# -*- coding: utf-8 -*-
"""
步骤 9：训练"累积二分类"区间模型，与 08 的 MultiClass 对照。

思路：延误档位是有序的（0 < 1-15 < 16-30 < 31-60 < 60+）。与其一次分 5 类
（MultiClass，无视顺序），不如训 4 个链式二分类器：

    P(>0), P(>15), P(>30), P(>60)

再逐档相减得到 5 档概率：

    bucket0 = 1 - P(>0)
    bucket1 = P(>0) - P(>15)
    bucket2 = P(>15) - P(>30)
    bucket3 = P(>30) - P(>60)
    bucket4 = P(>60)

对 4 个累积概率先做单调投影（从左向右取累积最小，保证 P(>0)≥P(>15)≥P(>30)≥P(>60)），
再相减并归一化，因此输出天然有序。

数据纪律与 08 完全一致：2019-2021 训练 / 2022 验证（早停+评估）/ 2023 测试不触碰；
不用类别加权（保持各档概率的绝对值可信）；特征 v1；CatBoost。

事先约定的裁决标准（对 08 存档的 catboost MultiClass，同在 2022 验证集）：
  ① 区间准确率  ② 逐档校准平均偏差(pp)  ③ macro F1
  累积模型至少赢两项才建议采用；本脚本【不会】修改 bucket_latest.json，
  是否切换生产模型由人工决定。

产物（out_dir = models/<时间戳>_cumbin/）：
  cls_cumbin_{seg}_gt{edge}__catboost.cbm / __metadata.json   共 8 个模型
  bucket_stats.json               训练集各档占比/中位分钟（沿用 08 的口径）
  cumulative_val_metrics.json     全部指标与对照明细
  cumulative_report.md            总结文档（含裁决）
  charts/*.png                    关键图像（英文标注）
  models/cumulative_latest.json   指针（注意：不是 bucket_latest.json）

用法：
    python 09_train_cumulative.py                       # 完整训练（约30-60分钟）
    python 09_train_cumulative.py --quick               # 抽样冒烟（约3-5分钟）
    python 09_train_cumulative.py --run-id <目录名>     # 断点续训（已训好的模型跳过）
"""
import argparse
import json
import platform
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config
from features import CATEGORICAL_FEATURES, build_features, make_targets, task_mask, year_series
from importlib import import_module

train_mod = import_module("02_train_models")
m08 = import_module("08_train_bucket_models")     # 复用 bucket_stats / multiclass_val_metrics
eval_mod = import_module("03_evaluate_models")

EDGES = list(config.DELAY_BUCKET_EDGES)           # [0, 15, 30, 60]
BUCKET_SHORT_EN = ["On time", "1-15", "16-30", "31-60", ">60"]

# ------------------------------------------------------------------ 核心数学
def monotone_project(p_cum: np.ndarray) -> np.ndarray:
    """单调投影：从左向右取累积最小，保证 P(>0)>=P(>15)>=P(>30)>=P(>60)。"""
    return np.minimum.accumulate(p_cum, axis=1)


def cum_to_buckets(p_cum: np.ndarray) -> np.ndarray:
    """4 个累积概率 -> 5 档概率（先投影，再相减，clip+归一化）。"""
    p = monotone_project(np.asarray(p_cum, dtype=float))
    cols = [1.0 - p[:, 0]]
    for k in range(len(EDGES) - 1):
        cols.append(p[:, k] - p[:, k + 1])
    cols.append(p[:, -1])
    b = np.column_stack(cols)
    b = np.clip(b, 0.0, None)
    return b / b.sum(axis=1, keepdims=True)


def binary_val_metrics(y_true, p):
    """单个累积阈值上的二分类指标。"""
    from sklearn.metrics import log_loss, roc_auc_score
    return {
        "auc": round(float(roc_auc_score(y_true, p)), 4),
        "logloss": round(float(log_loss(y_true, p, labels=[0, 1])), 4),
        "brier": round(float(np.mean((p - y_true) ** 2)), 4),
        "positive_rate": round(float(np.mean(y_true)), 4),
    }


def calib_dev_pp(bucket_metrics) -> float:
    """逐档校准平均偏差：|平均预测概率 - 实际频率| 的均值，单位百分点。"""
    c = bucket_metrics["probability_calibration"]
    return round(float(np.mean([abs(x["mean_pred_prob"] - x["actual_rate"]) for x in c])) * 100, 2)


# ------------------------------------------------------------------ 训练
def train_catboost_bin(x_tr, y_tr, x_va, y_va, quick=False):
    from catboost import CatBoostClassifier, Pool
    cat_idx = [x_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    model = CatBoostClassifier(
        loss_function="Logloss", eval_metric="Logloss",
        iterations=300 if quick else 1500, learning_rate=0.1, depth=8,
        l2_leaf_reg=3.0, random_seed=config.RANDOM_SEED,
        early_stopping_rounds=50 if quick else 150,
        verbose=200, allow_writing_files=False)
    model.fit(Pool(x_tr, y_tr, cat_features=cat_idx),
              eval_set=Pool(x_va, y_va, cat_features=cat_idx), use_best_model=True)
    return model, cat_idx


def predict_catboost_bin(model, x, cat_idx):
    from catboost import Pool
    return model.predict_proba(Pool(x, cat_features=cat_idx))[:, 1]


# ------------------------------------------------------------------ 对照模型
def load_flat_models():
    """加载 08 存档的 catboost MultiClass（生产在用的那套）。返回 {seg: (run_id, meta, model)}。"""
    ptr = config.MODELS_DIR / "bucket_latest.json"
    out = {}
    if not ptr.exists():
        return out
    run_id = json.load(open(ptr, encoding="utf-8"))["run_id"]
    for seg in ("dep", "arr"):
        mp = config.MODELS_DIR / run_id / f"mcls_{seg}_delay__catboost__metadata.json"
        if mp.exists():
            meta = json.load(open(mp, encoding="utf-8"))
            out[seg] = (run_id, meta, eval_mod.load_model(meta))
    return out


# ------------------------------------------------------------------ 图像
def make_charts(seg, y_true, cum_b, flat_b, p_cum, metrics, out_dir):
    """关键图像（英文标注，150dpi）：可靠性、混淆矩阵、逐档校准、逐档F1。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_dir = out_dir / "charts"
    c_dir.mkdir(exist_ok=True)
    n = len(y_true)

    # 1) 可靠性图：4 个累积阈值 2x2
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for i, edge in enumerate(EDGES):
        ax = axes[i // 2][i % 2]
        p, yy = p_cum[:, i], (y_true > edge).astype(int)
        ids = np.digitize(p, np.linspace(0, 1, 11)) - 1
        xs, ys = [], []
        for b in range(10):
            m = ids == b
            if m.sum() >= max(50, n // 2000):
                xs.append(p[m].mean()); ys.append(yy[m].mean())
        ax.plot([0, 1], [0, 1], "--", color="#b9b7af", lw=1)
        ax.plot(xs, ys, "o-", color="#00645a", ms=4)
        ax.set_title(f"P(delay > {edge} min), AUC={metrics['cumulative']['auc_per_threshold'][f'gt{edge}']:.3f}",
                     fontsize=11)
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Empirical frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle(f"Cumulative classifiers reliability ({seg.upper()}, validation 2022)", fontsize=13)
    fig.tight_layout()
    fig.savefig(c_dir / f"{seg}_reliability.png", dpi=150); plt.close(fig)

    # 2) 混淆矩阵对照（按行归一化=召回视角）
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, (name, proba, acc) in zip(
            axes, [("Cumulative (this run)", cum_b, metrics["cumulative"]["bucket_metrics"]["accuracy"]),
                   ("MultiClass (08 archive)", flat_b, metrics["flat_multiclass"]["bucket_metrics"]["accuracy"]
                    if flat_b is not None else None)]):
        cm = np.zeros((config.N_BUCKETS, config.N_BUCKETS))
        if proba is not None:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, proba.argmax(axis=1), labels=list(range(config.N_BUCKETS)))
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
        for r in range(config.N_BUCKETS):
            for c in range(config.N_BUCKETS):
                if cm[r, c] > 0:
                    ax.text(c, r, f"{cmn[r, c]:.0f}%", ha="center", va="center", fontsize=9,
                            color="white" if cmn[r, c] > 55 else "#20302c")
        ax.set_xticks(range(config.N_BUCKETS), BUCKET_SHORT_EN)
        ax.set_yticks(range(config.N_BUCKETS), BUCKET_SHORT_EN)
        ax.set_xlabel("Predicted bucket"); ax.set_ylabel("Actual bucket")
        ax.set_title(f"{name}\naccuracy={acc if acc is not None else float('nan'):.4f}", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Confusion matrix, row-normalized ({seg.upper()})", fontsize=13)
    fig.tight_layout()
    fig.savefig(c_dir / f"{seg}_confusion.png", dpi=150); plt.close(fig)

    # 3) 逐档校准：预测概率均值 vs 实际频率
    cal_c = metrics["cumulative"]["bucket_metrics"]["probability_calibration"]
    cal_f = (metrics["flat_multiclass"]["bucket_metrics"]["probability_calibration"]
             if flat_b is not None else None)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    pos = np.arange(config.N_BUCKETS); w = 0.27
    actual = [x["actual_rate"] for x in cal_c]
    ax.bar(pos - w, actual, w, color="#2c4036", label="Actual rate")
    ax.bar(pos, [x["mean_pred_prob"] for x in cal_c], w, color="#00645a", label="Cumulative (pred)")
    if cal_f:
        ax.bar(pos + w, [x["mean_pred_prob"] for x in cal_f], w, color="#c9a227", label="MultiClass (pred)")
    ax.set_xticks(pos, BUCKET_SHORT_EN); ax.set_ylabel("Probability")
    ax.set_title(f"Per-bucket calibration ({seg.upper()}); mean |dev| "
                 f"cum={metrics['cumulative']['calib_mean_abs_dev_pp']:.2f}pp"
                 + (f" vs flat={metrics['flat_multiclass']['calib_mean_abs_dev_pp']:.2f}pp"
                    if flat_b is not None else ""))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(c_dir / f"{seg}_calibration.png", dpi=150); plt.close(fig)

    # 4) 逐档 F1 对照
    f1_c = metrics["cumulative"]["bucket_metrics"]["per_class_f1"]
    f1_f = (metrics["flat_multiclass"]["bucket_metrics"]["per_class_f1"] if flat_b is not None else None)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    keys = list(f1_c.keys())
    ax.bar(pos - 0.17, [f1_c[k] for k in keys], 0.34, color="#00645a", label="Cumulative")
    if f1_f:
        ax.bar(pos + 0.17, [f1_f[k] for k in keys], 0.34, color="#9aa5a0", label="MultiClass")
    ax.set_xticks(pos, BUCKET_SHORT_EN); ax.set_ylabel("F1")
    ax.set_ylim(0, 1); ax.legend(fontsize=9)
    ax.set_title(f"Per-bucket F1 ({seg.upper()}, validation 2022)")
    fig.tight_layout()
    fig.savefig(c_dir / f"{seg}_f1.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ 报告
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def write_report(out_dir, run_id, quick, results, flat_run_id):
    seg_cn = {"dep": "出发延误", "arr": "到达延误"}
    L = [f"# 累积二分类区间模型 — 训练与对照报告", "",
         f"- 运行ID：`{run_id}`（{'**quick 抽样模式，指标仅供参考**' if quick else '完整训练'}）",
         f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
         f"- 数据：`{config.CLEAN_CSV.name}`，训练 {list(config.TRAIN_YEARS)} / 验证 {list(config.VAL_YEARS)}，"
         f"**2023 测试集未触碰**；特征 v1；CatBoost Logloss（无类别加权，概率绝对值可信）",
         f"- 对照对象：08 存档 MultiClass `{flat_run_id}`（catboost）", "",
         "## 方法", "",
         "4 个链式二分类器 P(>0) / P(>15) / P(>30) / P(>60)，单调投影后逐档相减：", "",
         "```",
         "bucket0 = 1 - P(>0)      bucket1 = P(>0)  - P(>15)",
         "bucket2 = P(>15) - P(>30)  bucket3 = P(>30) - P(>60)   bucket4 = P(>60)",
         "```", ""]
    for seg, m in results.items():
        c, f = m["cumulative"], m["flat_multiclass"]
        L += [f"## {seg_cn[seg]}（验证集 n={m['n_val']:,}）", ""]
        rows = []
        # accuracy/macro_f1 在 bucket_metrics 里；calib_mean_abs_dev_pp 在段级顶层
        for name, key, better, where in [
                ("区间准确率", "accuracy", "high", "bucket"),
                ("macro F1", "macro_f1", "high", "bucket"),
                ("逐档校准平均偏差(pp)", "calib_mean_abs_dev_pp", "low", "seg")]:
            if where == "bucket":
                cv = c["bucket_metrics"][key]
                fv = f["bucket_metrics"][key] if f else None
            else:
                cv = c[key]
                fv = f[key] if f else None
            rows.append([name, cv, fv if fv is not None else "—",
                         "✅ 累积" if (fv is None or (cv > fv if better == "high" else cv < fv)) else "✅ MultiClass"])
        L += [md_table(["指标", "累积二分类", "MultiClass", "胜者"], rows), ""]

        auc = c["auc_per_threshold"]
        L += [f"各阈值 AUC：>0 分钟 **{auc['gt0']}** / >15 分钟 **{auc['gt15']}** / "
              f">30 分钟 **{auc['gt30']}** / >60 分钟 **{auc['gt60']}**", "",
              f"单调违例（投影前）：{c['monotone_violations_raw_pct']}% → 投影后恒为 0", ""]

        cal_rows = []
        for k in range(config.N_BUCKETS):
            cc = c["bucket_metrics"]["probability_calibration"][k]
            ff = f["bucket_metrics"]["probability_calibration"][k] if f else None
            cal_rows.append([f"bucket{k} {config.BUCKET_LABELS[k]}",
                             f"{cc['mean_pred_prob']:.3f}",
                             f"{ff['mean_pred_prob']:.3f}" if ff else "—",
                             f"{cc['actual_rate']:.3f}"])
        L += [md_table(["档位", "累积:平均预测概率", "MultiClass:平均预测概率", "实际频率"], cal_rows), ""]

        f1_c = c["bucket_metrics"]["per_class_f1"]
        f1_f = f["bucket_metrics"]["per_class_f1"] if f else None
        f1_rows = [[f"bucket{k} {config.BUCKET_LABELS[k]}", f1_c[f"bucket{k}_{config.BUCKET_LABELS[k]}"],
                    f1_f[f"bucket{k}_{config.BUCKET_LABELS[k]}"] if f1_f else "—"]
                   for k in range(config.N_BUCKETS)]
        L += [md_table(["档位", "累积 F1", "MultiClass F1"], f1_rows), ""]

        v = m["verdict"]
        L += [f"**裁决**：三项准则（准确率/校准/macro F1）累积模型胜出 **{v['wins']}/3** → "
              f"{v['recommend']}（标准：≥2 项才建议采用）", "",
              f"图像：`charts/{seg}_reliability.png`、`charts/{seg}_confusion.png`、"
              f"`charts/{seg}_calibration.png`、`charts/{seg}_f1.png`", ""]
    L += ["## 产物与后续", "",
          "- 模型：8 个 `cls_cumbin_{seg}_gt{edge}__catboost.cbm` + 同名 metadata",
          "- 指标明细：`cumulative_val_metrics.json`；指针：`models/cumulative_latest.json`",
          "- **本脚本没有改动生产配置**（`bucket_latest.json` 未动，网页仍用 08 的模型）。",
          "  若裁决为'建议采用'且你们确认切换：把 `models/bucket_latest.json` 中的 `run_id` 手动改为"
          f"`{run_id}` 之前，还需要给 04_predict.py 增加累积模型的加载分支（另行任务）。", ""]
    (out_dir / "cumulative_report.md").write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description="训练累积二分类区间模型（对照 08 MultiClass）")
    ap.add_argument("--data", default=str(config.CLEAN_CSV))
    ap.add_argument("--run-name", default="cumbin")
    ap.add_argument("--run-id", default=None, help="续训已有目录时传目录名")
    ap.add_argument("--quick", action="store_true", help="抽样冒烟（150k训练/80k验证/300轮）")
    args = ap.parse_args()

    run_id = args.run_id or train_mod.make_run_id(args.run_name, quick=args.quick)
    out_dir = config.MODELS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"存档目录: {out_dir}{'（quick 模式，指标仅供参考）' if args.quick else ''}")

    df = make_targets(pd.read_csv(args.data, low_memory=False))
    year = year_series(df)
    train = df[year.isin(config.TRAIN_YEARS)]
    val = df[year.isin(config.VAL_YEARS)].copy()

    flat = load_flat_models()
    flat_run_id = flat.get("dep", (None, None, None))[0]

    results = {}
    for seg, col in [("dep", "DEP_DELAY"), ("arr", "ARR_DELAY")]:
        task_base = f"mcls_{seg}_delay"
        m_tr = task_mask(train, task_base)
        m_va = task_mask(val, task_base)
        tr = train[m_tr]
        va = val[m_va]
        if args.quick:
            tr = tr.sample(min(len(tr), 150_000), random_state=config.RANDOM_SEED)
            va = va.sample(min(len(va), 80_000), random_state=config.RANDOM_SEED)
        x_tr = build_features(tr, feature_set="v1")
        x_va = build_features(va, feature_set="v1")
        delay_va = pd.to_numeric(va[col], errors="coerce")
        y_true = va["y_bucket_" + seg].astype(int).to_numpy()
        print(f"\n=== [{seg}] 训练 {len(tr):,} / 验证 {len(va):,} "
              f"（验证延误率 {float((delay_va > 0).mean()):.1%}） ===")

        # 4 个累积二分类
        p_cum = np.zeros((len(va), len(EDGES)))
        auc_per, ll_per = {}, {}
        for edge in EDGES:
            task = f"cls_cumbin_{seg}_gt{edge}"
            meta_path = out_dir / f"{task}__catboost__metadata.json"
            model_path = out_dir / f"{task}__catboost.cbm"
            y_tr = (pd.to_numeric(tr[col], errors="coerce") > edge).astype(int).to_numpy()
            y_va = (delay_va > edge).astype(int).to_numpy()
            t0 = time.time()
            if meta_path.exists() and model_path.exists():
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(); model.load_model(str(model_path))
                cat_idx = [x_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
                best_iter = int(model.get_best_iteration())
                print(f"  [>{edge:>3}min] 已存在，跳过训练")
            else:
                model, cat_idx = train_catboost_bin(x_tr, y_tr, x_va, y_va, quick=args.quick)
                model.save_model(str(model_path))
                best_iter = int(model.get_best_iteration())
            p = predict_catboost_bin(model, x_va, cat_idx)
            p_cum[:, EDGES.index(edge)] = p
            bm = binary_val_metrics(y_va, p)
            auc_per[f"gt{edge}"], ll_per[f"gt{edge}"] = bm["auc"], bm["logloss"]
            print(f"  [>{edge:>3}min] AUC={bm['auc']:.4f} logloss={bm['logloss']:.4f} "
                  f"正例率={bm['positive_rate']:.1%} ({time.time()-t0:.0f}s, best_iter={best_iter})")
            meta = {
                "run_id": run_id, "task": task,
                "task_desc": f"累积二分类: P({seg}延误 > {edge}分钟), 无类别加权",
                "algo": "catboost", "feature_set": "v1", "model_file": model_path.name,
                "model_path": str(model_path), "features": list(x_tr.columns),
                "categorical_features": CATEGORICAL_FEATURES, "categories": None,
                "edge_minutes": edge, "cumulative": True,
                "bucket_edges": config.DELAY_BUCKET_EDGES,
                "bucket_labels": config.BUCKET_LABELS,
                "split": {"train_years": list(config.TRAIN_YEARS),
                          "val_years": list(config.VAL_YEARS),
                          "train_rows": int(len(tr)), "val_rows": int(len(va))},
                "hyperparams": {"loss_function": "Logloss",
                                "iterations": 300 if args.quick else 1500,
                                "learning_rate": 0.1, "depth": 8, "l2_leaf_reg": 3.0,
                                "class_weighting": "无（保持概率绝对值可信）"},
                "random_seed": config.RANDOM_SEED,
                "val_metrics_2022": {**bm, "best_iteration": best_iter},
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "library_versions": {"python": platform.python_version(),
                                     "pandas": pd.__version__, "numpy": np.__version__,
                                     "catboost": __import__("catboost").__version__},
                "mode": "quick" if args.quick else "full",
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        viol_raw = float(np.mean(np.any(np.diff(p_cum, axis=1) > 1e-9, axis=1))) * 100
        cum_b = cum_to_buckets(p_cum)
        cum_metrics = m08.multiclass_val_metrics(y_true, cum_b)

        # 对照：08 的 catboost MultiClass，同一批验证行
        if seg in flat:
            _, fmeta, fmodel = flat[seg]
            fx = x_va[fmeta["features"]] if fmeta.get("features") else x_va
            flat_b = np.asarray(fmodel.predict_proba(fx))
            flat_metrics = m08.multiclass_val_metrics(y_true, flat_b)
        else:
            flat_b, flat_metrics = None, None

        res = {
            "n_val": int(len(va)),
            "cumulative": {
                "auc_per_threshold": auc_per, "logloss_per_threshold": ll_per,
                "monotone_violations_raw_pct": round(viol_raw, 2),
                "bucket_metrics": cum_metrics,
                "calib_mean_abs_dev_pp": calib_dev_pp(cum_metrics),
            },
            "flat_multiclass": (
                {"run_id": flat[seg][0], "bucket_metrics": flat_metrics,
                 "calib_mean_abs_dev_pp": calib_dev_pp(flat_metrics)}
                if flat_metrics else None),
        }
        cw = (cum_metrics["accuracy"] > flat_metrics["accuracy"]) if flat_metrics else True
        cc = (calib_dev_pp(cum_metrics) < calib_dev_pp(flat_metrics)) if flat_metrics else True
        cf = (cum_metrics["macro_f1"] > flat_metrics["macro_f1"]) if flat_metrics else True
        wins = int(cw) + int(cc) + int(cf)
        res["verdict"] = {"accuracy_win": bool(cw), "calibration_win": bool(cc),
                          "macro_f1_win": bool(cf), "wins": wins,
                          "recommend": "建议采用累积模型" if wins >= 2 else "保留现状(MultiClass)"}
        results[seg] = res
        make_charts(seg, y_true, cum_b, flat_b, p_cum, res, out_dir)

    # bucket_stats 沿用 08 口径（若续训目录已有则跳过）
    stats_path = out_dir / "bucket_stats.json"
    if not stats_path.exists():
        all_stats = {seg: m08.bucket_stats(train, col) for seg, col in
                     [("dep", "DEP_DELAY"), ("arr", "ARR_DELAY")]}
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)

    with open(out_dir / "cumulative_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "mode": "quick" if args.quick else "full",
                   "flat_multiclass_run_id": flat_run_id, "segments": results},
                  f, ensure_ascii=False, indent=2)
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "created_at": datetime.now().isoformat(timespec="seconds"),
                   "type": "delay_bucket_cumulative_binary",
                   "mode": "quick" if args.quick else "full",
                   "models": [str(p) for p in sorted(out_dir.glob("cls_cumbin_*.cbm"))]},
                  f, ensure_ascii=False, indent=2)
    with open(config.MODELS_DIR / "cumulative_latest.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id}, f, ensure_ascii=False, indent=2)

    write_report(out_dir, run_id, args.quick, results, flat_run_id)

    # ---------------- 控制台总结 ----------------
    print("\n" + "=" * 74)
    print("累积二分类 vs MultiClass 对照总结（2022 验证集）")
    print("=" * 74)
    for seg, m in results.items():
        c, f, v = m["cumulative"], m["flat_multiclass"], m["verdict"]
        fa = f" vs 平面 {f['bucket_metrics']['accuracy']:.4f}" if f else ""
        fc = f" vs 平面 {f['calib_mean_abs_dev_pp']:.2f}pp" if f else ""
        ff_ = f" vs 平面 {f['bucket_metrics']['macro_f1']:.4f}" if f else ""
        print(f"[{seg}] 区间准确率  : 累积 {c['bucket_metrics']['accuracy']:.4f}{fa}")
        print(f"[{seg}] 校准平均偏差: 累积 {c['calib_mean_abs_dev_pp']:.2f}pp{fc}")
        print(f"[{seg}] macro F1    : 累积 {c['bucket_metrics']['macro_f1']:.4f}{ff_}")
        print(f"[{seg}] 各阈值AUC   : " + "  ".join(
            f">{k}={c['auc_per_threshold'][f'gt{k}']}" for k in EDGES))
        print(f"[{seg}] 单调违例    : 投影前 {c['monotone_violations_raw_pct']}% → 投影后 0%")
        print(f"[{seg}] 裁决        : 胜出 {v['wins']}/3 → {v['recommend']}")
    print("-" * 74)
    print(f"报告: {out_dir / 'cumulative_report.md'}")
    print(f"图像: {out_dir / 'charts'}/ ({'_'.join(['reliability', 'confusion', 'calibration', 'f1'])} x dep/arr)")
    print(f"指标: {out_dir / 'cumulative_val_metrics.json'}")
    print("注意: 未修改 bucket_latest.json，生产网页仍用 08 的模型（采用与否人工决定）")
    print("=" * 74)


if __name__ == "__main__":
    main()
