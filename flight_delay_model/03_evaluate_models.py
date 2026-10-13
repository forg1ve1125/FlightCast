# -*- coding: utf-8 -*-
"""
步骤 3：在测试集（2023年数据）上评估模型存档。

会议纪要：测试集只能使用一次（反复调参请用验证集2022）。
  - 分类指标：准确率 accuracy、精确率 precision、召回率 recall、F1、AUC、混淆矩阵
    （阈值 0.5 一份；另附用验证集调出的最优F1阈值一份——调阈值只动验证集，不算泄露）
  - 回归指标：MAE、RMSE、R²（在实际延误>0的测试样本上，即"若延误，延误多少分钟"）
  - 两段式合成：期望延误分钟 = P(延误) × 预测延误分钟，给出全量样本上的MAE

评估报告写入存档目录 evaluation_report.json / evaluation_report.md。

用法：
    python 03_evaluate_models.py                 # 评估最近一次训练存档
    python 03_evaluate_models.py --run-id xxx    # 评估指定存档
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from features import align_categories, build_features, make_targets, task_mask, year_series


def load_run(run_id=None, models_dir=None):
    models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
    if run_id is None:
        latest = config.LATEST_RUN_JSON
        if not latest.exists():
            raise SystemExit("未找到任何训练存档，请先运行 02_train_models.py")
        with open(latest, encoding="utf-8") as f:
            run_info = json.load(f)
        run_id = run_info["run_id"]
    run_dir = models_dir / run_id
    if not run_dir.exists():
        raise SystemExit(f"存档目录不存在: {run_dir}")
    metas = sorted(run_dir.glob("*__metadata.json"))
    if not metas:
        raise SystemExit(f"存档目录中没有 metadata: {run_dir}")
    return run_dir, [json.load(open(m, encoding="utf-8")) for m in metas]


def load_model(meta):
    path = meta["model_path"]
    if meta["algo"] == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if meta["task"].startswith(("cls", "mcls")) else CatBoostRegressor
        model = cls()
        model.load_model(path)
    else:
        import lightgbm as lgb
        model = lgb.Booster(model_file=path)
    return model


def prepare_x(meta, x):
    """LightGBM 的分类特征必须对齐到训练时的类别集合。"""
    if meta["algo"] == "lightgbm":
        x = align_categories(x, meta.get("categories") or {})
    return x


def feature_set_of(meta):
    """根据 metadata 的特征清单判断特征集版本（兼容旧存档）。"""
    return "v2" if "HIST_ROUTE_RATE" in (meta.get("features") or []) else "v1"


def load_hist_stats(run_dir):
    """加载存档中的训练期历史统计（v2 特征需要）。"""
    p = Path(run_dir) / "historical_stats.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def build_x(meta, df, hist_stats):
    from features import resolve_stats
    return build_features(df, hist_stats=resolve_stats(hist_stats),
                          feature_set=feature_set_of(meta))


def predict(model, meta, x):
    x = prepare_x(meta, x)
    if meta["algo"] == "catboost":
        if meta["task"].startswith("mcls"):
            return model.predict_proba(x)          # 多分类：返回 (n, n_classes)
        if meta["task"].startswith("cls"):
            return model.predict_proba(x)[:, 1]    # 二分类：返回正类概率
        return model.predict(x)
    ni = model.best_iteration
    out = model.predict(x, num_iteration=ni if ni and ni > 0 else None)
    return out


def best_f1_threshold(y_true, proba):
    """在验证集上扫描阈值，找F1最大的阈值（只使用验证集，不算数据泄露）。"""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 2), round(best_f1, 4)


def eval_classification(y_true, proba, threshold):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_true, proba)), 4),
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
    }


def eval_regression(y_true, pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "mae_min": round(float(mean_absolute_error(y_true, pred)), 2),
        "rmse_min": round(float(np.sqrt(mean_squared_error(y_true, pred))), 2),
        "r2": round(float(r2_score(y_true, pred)), 4),
        "n_samples": int(len(y_true)),
    }


def main():
    ap = argparse.ArgumentParser(description="在2023测试集上评估模型（测试集只能用一次！）")
    ap.add_argument("--run-id", default=None, help="要评估的存档名（默认最近一次）")
    ap.add_argument("--data", default=str(config.CLEAN_CSV), help="清洗后的CSV路径")
    args = ap.parse_args()

    print("=" * 70)
    print("警告：2023 测试集按会议纪要只能使用一次；调参请改用 2022 验证集。")
    print("=" * 70)

    run_dir, metas = load_run(args.run_id)
    print(f"评估存档: {run_dir}  （{len(metas)} 个模型）\n")
    hist_stats = load_hist_stats(run_dir)

    print(f"读取数据: {args.data}")
    df = make_targets(pd.read_csv(args.data, low_memory=False))
    year = year_series(df)
    val_df = df[year.isin(config.VAL_YEARS)].copy()
    test_df = df[year.isin(config.TEST_YEARS)].copy()
    print(f"  验证集(2022): {len(val_df):,} 行 | 测试集(2023): {len(test_df):,} 行")

    report = {"run_id": run_dir.name, "evaluated_at": datetime.now().isoformat(timespec="seconds"),
              "test_years": list(config.TEST_YEARS), "note": "测试集只能使用一次", "models": []}

    # 分类模型先在验证集上调阈值（合法：验证集可反复使用）
    tuned_thresholds = {}
    for meta in metas:
        if meta["task"].startswith("cls"):
            model = load_model(meta)
            mask_v = task_mask(val_df, meta["task"])
            xv = build_x(meta, val_df[mask_v], hist_stats)
            yv = val_df[mask_v]["y_cls_dep" if "dep" in meta["task"] else "y_cls_arr"].astype(int)
            proba_v = predict(model, meta, xv)
            t, f1v = best_f1_threshold(yv.to_numpy(), np.asarray(proba_v))
            tuned_thresholds[meta["task"]] = {"threshold": t, "val_f1": f1v}
            print(f"[阈值调优|验证集] {meta['task']}: best_threshold={t} (val F1={f1v})")
    print()

    for meta in metas:
        task, algo = meta["task"], meta["algo"]
        model = load_model(meta)
        mask_t = task_mask(test_df, task)
        xt = build_x(meta, test_df[mask_t], hist_stats)
        entry = {"task": task, "algo": algo, "model_file": meta["model_file"],
                 "test_rows": int(mask_t.sum())}

        if task.startswith("cls"):
            yt = test_df[mask_t]["y_cls_dep" if "dep" in task else "y_cls_arr"].astype(int)
            proba = np.asarray(predict(model, meta, xt))
            entry["test_metrics@0.5"] = eval_classification(yt.to_numpy(), proba, 0.5)
            t = tuned_thresholds.get(task, {}).get("threshold", 0.5)
            entry["test_metrics@tuned"] = eval_classification(yt.to_numpy(), proba, t)
            entry["tuned_threshold_from_val"] = tuned_thresholds.get(task, {})
            m = entry["test_metrics@0.5"]
            print(f"[测试集] {task:14s} [{algo:9s}] acc={m['accuracy']:.4f} "
                  f"precision={m['precision']:.4f} recall={m['recall']:.4f} "
                  f"F1={m['f1']:.4f} AUC={m['auc']:.4f}")
            mt = entry["test_metrics@tuned"]
            print(f"{'':14s} 调优阈值{t}: acc={mt['accuracy']:.4f} precision={mt['precision']:.4f} "
                  f"recall={mt['recall']:.4f} F1={mt['f1']:.4f}")
        else:
            yt = test_df[mask_t]["y_reg_dep" if "dep" in task else "y_reg_arr"].astype(float)
            pred = np.asarray(predict(model, meta, xt), dtype=float)
            entry["test_metrics_delayed_only"] = eval_regression(yt.to_numpy(), pred)
            m = entry["test_metrics_delayed_only"]
            print(f"[测试集] {task:14s} [{algo:9s}] MAE={m['mae_min']:.2f}min "
                  f"RMSE={m['rmse_min']:.2f}min R2={m['r2']:.4f} (n={m['n_samples']:,})")

            # 两段式合成：P(延误)×预测分钟，在整个测试集上评估期望延误
            cls_task = task.replace("reg", "cls")
            cls_meta = next((mm for mm in metas if mm["task"] == cls_task and mm["algo"] == algo), None)
            if cls_meta is not None:
                cmask = task_mask(test_df, cls_task)
                xc = build_x(cls_meta, test_df[cmask], hist_stats)
                yc = test_df[cmask]["y_cls_dep" if "dep" in task else "y_cls_arr"].astype(int).to_numpy()
                proba = np.asarray(predict(load_model(cls_meta), cls_meta, xc))
                # 回归只对延误>0的样本有预测；把预测对齐回分类样本（未延误样本记0分钟）
                sub_mask = cmask & (test_df["y_reg_dep" if "dep" in task else "y_reg_arr"].fillna(-1) > 0)
                reg_aligned = pd.Series(pred, index=test_df.index[sub_mask])
                expected = proba * reg_aligned.reindex(test_df.index[cmask]).fillna(0).to_numpy()
                combined_mae = float(np.mean(np.abs(expected - yc)))
                entry["combined_expected_delay_mae_min"] = round(combined_mae, 2)
                print(f"{'':14s} 两段式合成期望延误 MAE={entry['combined_expected_delay_mae_min']:.2f}min")

        report["models"].append(entry)

    with open(run_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 生成 Markdown 报告
    lines = [f"# 测试集评估报告（2023）", f"- 存档: `{run_dir.name}`",
             f"- 评估时间: {report['evaluated_at']}",
             f"- 注意: 测试集只能使用一次；调参请用 2022 验证集", ""]
    for e in report["models"]:
        lines.append(f"## {e['task']} ({e['algo']})")
        if e["task"].startswith("cls"):
            for key in ["test_metrics@0.5", "test_metrics@tuned"]:
                m = e[key]
                lines.append(f"- **{key}**: 准确率={m['accuracy']}, 精确率={m['precision']}, "
                             f"召回率={m['recall']}, F1={m['f1']}, AUC={m['auc']}, "
                             f"混淆矩阵 TP={m['confusion_matrix']['TP']} FP={m['confusion_matrix']['FP']} "
                             f"TN={m['confusion_matrix']['TN']} FN={m['confusion_matrix']['FN']}")
        else:
            m = e["test_metrics_delayed_only"]
            lines.append(f"- **延误航班回归**: MAE={m['mae_min']}min, RMSE={m['rmse_min']}min, R²={m['r2']}")
            if "combined_expected_delay_mae_min" in e:
                lines.append(f"- **两段式合成期望延误 MAE**: {e['combined_expected_delay_mae_min']}min")
        lines.append("")
    with open(run_dir / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n报告已写入:\n  {run_dir / 'evaluation_report.md'}\n  {run_dir / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
