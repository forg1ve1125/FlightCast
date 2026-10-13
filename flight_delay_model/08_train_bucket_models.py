# -*- coding: utf-8 -*-
"""
步骤 8：训练"延误区间"多分类模型（5档）——多情景概率输出。

对应会议纪要示例的完整形态：
  "航班A有62%的概率不延误，24%的概率延误1-15分钟，9%的概率延误16-30分钟..."

区间定义（config.DELAY_BUCKET_EDGES）：
  0: 不延误(<=0)   1: (0,15]   2: (15,30]   3: (30,60]   4: >60（分钟）

训练 2算法(CatBoost/LightGBM) × 2目标(出发/到达) = 4 个模型。
数据划分与主流程一致：2019-2021 训练 / 2022 验证（早停+评估）/ 2023 测试不触碰。
不用类别加权（保持概率的绝对值可信，直接作为各情景概率展示）。

同时保存 bucket_stats.json：训练集每个区间的样本占比与中位延误分钟数
（用于把"延误16-30分钟"翻译成"延误约X分钟"）。

用法：
    python 08_train_bucket_models.py [--algos catboost,lightgbm] [--run-name buckets]
"""
import argparse
import json
import platform
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config
from features import (CATEGORICAL_FEATURES, build_features, make_targets,
                      task_mask, year_series)
from importlib import import_module

train_mod = import_module("02_train_models")


def bucket_stats(train_df, col):
    """训练集各区间的占比与中位延误分钟（预测时用来换算'约X分钟'）。"""
    m = task_mask(train_df, f"mcls_{'dep' if 'DEP' in col else 'arr'}")
    d = pd.to_numeric(train_df[m][col], errors="coerce")
    b = train_mod.__dict__  # noqa
    from features import bucket_of_delay
    idx = bucket_of_delay(d)
    stats = []
    for k in range(config.N_BUCKETS):
        sub = d[idx == k]
        stats.append({
            "bucket": k,
            "label": config.BUCKET_LABELS[k],
            "train_share": round(float((idx == k).mean()), 4),
            "median_minutes": round(float(sub.median()), 1) if len(sub) else None,
            "mean_minutes": round(float(sub.mean()), 1) if len(sub) else None,
        })
    return stats


def multiclass_val_metrics(y_true, proba):
    """验证集多分类指标 + 每档概率的校准情况（预测概率 vs 实际频率）。"""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    pred = proba.argmax(axis=1)
    cm = confusion_matrix(y_true, pred, labels=list(range(config.N_BUCKETS)))
    report = classification_report(y_true, pred, labels=list(range(config.N_BUCKETS)),
                                   output_dict=True, zero_division=0)
    calib = []
    for k in range(config.N_BUCKETS):
        mask = np.ones(len(y_true), dtype=bool)  # 全样本上该档的预测概率均值 vs 实际频率
        calib.append({
            "bucket": k,
            "mean_pred_prob": round(float(proba[mask, k].mean()), 4),
            "actual_rate": round(float((y_true == k).mean()), 4),
        })
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "confusion_matrix": cm.tolist(),
        "per_class_f1": {f"bucket{k}_{config.BUCKET_LABELS[k]}":
                         round(float(report[str(k)]["f1-score"]), 4)
                         for k in range(config.N_BUCKETS)},
        "probability_calibration": calib,
    }


def train_catboost_mcls(x_tr, y_tr, x_va, y_va, algo_seed_params=None):
    from catboost import CatBoostClassifier, Pool
    cat_idx = [x_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    model = CatBoostClassifier(
        loss_function="MultiClass", eval_metric="MultiClass",
        iterations=1200, learning_rate=0.1, depth=7, l2_leaf_reg=3.0,
        random_seed=config.RANDOM_SEED, early_stopping_rounds=150,
        verbose=200, allow_writing_files=False)
    model.fit(Pool(x_tr, y_tr, cat_features=cat_idx),
              eval_set=Pool(x_va, y_va, cat_features=cat_idx), use_best_model=True)
    return model


def train_lightgbm_mcls(x_tr, y_tr, x_va, y_va):
    import lightgbm as lgb
    x_tr, x_va = x_tr.copy(), x_va.copy()
    cats_map = {}
    for c in CATEGORICAL_FEATURES:
        cats = sorted(x_tr[c].unique())
        cats_map[c] = cats
        x_tr[c] = pd.Categorical(x_tr[c], categories=cats)
        x_va[c] = pd.Categorical(x_va[c], categories=cats)
    params = dict(objective="multiclass", num_class=config.N_BUCKETS,
                  metric="multi_logloss", learning_rate=0.05, num_leaves=63,
                  min_child_samples=300, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, seed=config.RANDOM_SEED,
                  verbose=-1, force_row_wise=True)
    train_set = lgb.Dataset(x_tr, label=y_tr)
    val_set = lgb.Dataset(x_va, label=y_va, reference=train_set)
    booster = lgb.train(params, train_set, num_boost_round=1500,
                        valid_sets=[val_set],
                        callbacks=[lgb.early_stopping(150, verbose=False),
                                   lgb.log_evaluation(200)])
    return booster, cats_map


def main():
    ap = argparse.ArgumentParser(description="训练延误区间多分类模型（多情景概率）")
    ap.add_argument("--data", default=str(config.CLEAN_CSV))
    ap.add_argument("--algos", default="catboost,lightgbm")
    ap.add_argument("--run-name", default="buckets")
    ap.add_argument("--run-id", default=None, help="完整存档目录名（续训已有目录时用）")
    args = ap.parse_args()
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]

    run_id = args.run_id or train_mod.make_run_id(args.run_name)
    out_dir = config.MODELS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"存档目录: {out_dir}")

    df = make_targets(pd.read_csv(args.data, low_memory=False))
    year = year_series(df)
    train = df[year.isin(config.TRAIN_YEARS)]
    val = df[year.isin(config.VAL_YEARS)].copy()

    all_stats, metas = {}, []
    for seg, col, y_col in [("dep", "DEP_DELAY", "y_bucket_dep"),
                            ("arr", "ARR_DELAY", "y_bucket_arr")]:
        task = f"mcls_{seg}_delay"
        stats = bucket_stats(train, col)
        all_stats[seg] = stats
        print(f"\n[{seg}] 训练集区间分布: " +
              ", ".join(f"{s['label']}={s['train_share']:.1%}" for s in stats))

        m_tr = task_mask(train, task)
        m_va = task_mask(val, task)
        y_tr = train[m_tr][y_col].astype(int).to_numpy()
        y_va = val[m_va][y_col].astype(int).to_numpy()
        x_tr = build_features(train[m_tr], feature_set="v1")
        x_va = build_features(val[m_va], feature_set="v1")
        print(f"=== [{seg}] 训练 {len(y_tr):,} / 验证 {len(y_va):,} ===")

        for algo in algos:
            meta_path = out_dir / f"{task}__{algo}__metadata.json"
            model_path = out_dir / (f"{task}__{algo}.cbm" if algo == "catboost"
                                    else f"{task}__{algo}.txt")
            if meta_path.exists() and model_path.exists():
                meta = json.load(open(meta_path, encoding="utf-8"))
                metas.append(meta)
                print(f"  [{algo}] 已存在，跳过（acc={meta['val_metrics_2022']['accuracy']}）")
                continue
            t0 = time.time()
            if algo == "catboost":
                model = train_catboost_mcls(x_tr, y_tr, x_va, y_va)
                proba = model.predict_proba(x_va)
                path = model_path
                model.save_model(str(path))
                cats_map, best_iter = None, int(model.get_best_iteration())
            else:
                model, cats_map = train_lightgbm_mcls(x_tr, y_tr, x_va, y_va)
                from features import align_categories
                proba = model.predict(align_categories(x_va, cats_map),
                                      num_iteration=model.best_iteration)
                path = model_path
                model.save_model(str(path), num_iteration=model.best_iteration)
                best_iter = int(model.best_iteration)

            metrics = multiclass_val_metrics(y_va, np.asarray(proba))
            metrics["best_iteration"] = best_iter
            metrics["seconds"] = round(time.time() - t0, 1)
            print(f"  [{algo}] acc={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f} "
                  f"({metrics['seconds']}s, best_iter={best_iter})")

            meta = {
                "run_id": run_id, "task": task, "task_desc": config.TASKS[task],
                "algo": algo, "feature_set": "v1", "model_file": path.name,
                "model_path": str(path), "features": list(x_tr.columns),
                "categorical_features": CATEGORICAL_FEATURES,
                "categories": cats_map,
                "n_classes": config.N_BUCKETS,
                "bucket_edges": config.DELAY_BUCKET_EDGES,
                "bucket_labels": config.BUCKET_LABELS,
                "split": {"train_years": list(config.TRAIN_YEARS),
                          "val_years": list(config.VAL_YEARS),
                          "test_years": list(config.TEST_YEARS),
                          "train_rows": int(len(y_tr)), "val_rows": int(len(y_va))},
                "hyperparams": {"iterations": 1200 if algo == "catboost" else 1500,
                                "learning_rate": 0.1 if algo == "catboost" else 0.05,
                                "depth": 7 if algo == "catboost" else None,
                                "num_leaves": None if algo == "catboost" else 63,
                                "class_weighting": "无（保持概率绝对值可信）"},
                "random_seed": config.RANDOM_SEED,
                "val_metrics_2022": metrics,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "library_versions": {"python": platform.python_version(),
                                     "pandas": pd.__version__, "numpy": np.__version__,
                                     "catboost": __import__("catboost").__version__ if algo == "catboost" else None,
                                     "lightgbm": __import__("lightgbm").__version__ if algo == "lightgbm" else None},
            }
            with open(out_dir / f"{task}__{algo}__metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            metas.append(meta)

    with open(out_dir / "bucket_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "created_at": datetime.now().isoformat(timespec="seconds"),
                   "type": "delay_bucket_multiclass", "models": [m["model_path"] for m in metas]},
                  f, ensure_ascii=False, indent=2)
    with open(config.MODELS_DIR / "bucket_latest.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id}, f, ensure_ascii=False, indent=2)

    print(f"\n完成。区间模型存档: {out_dir}")


if __name__ == "__main__":
    main()
