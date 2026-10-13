# -*- coding: utf-8 -*-
"""
步骤 2b：超参数/特征调优——【只用 2019-2021 训练 + 2022 验证集】，绝不触碰 2023 测试集。

会议纪要：验证集可以反复使用，测试集只能用一次。
本脚本对一组候选配置（特征集版本、学习率、树深、正则等）在验证集上一一评估，
输出排名，选出最优配置后再去 02_train_models.py 正式训练/评估。

用法：
    python 02b_tune_on_validation.py               # 全部候选（约25-35分钟）
    python 02b_tune_on_validation.py --task cls_dep_delay --algos catboost
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config
from features import (align_categories, build_features, compute_asof_stats,
                      make_targets, task_mask, year_series)
from importlib import import_module

train_mod = import_module("02_train_models")


def evaluate_classification(y_true, proba):
    """验证集指标：AUC + 最优F1工作点 + precision@recall。"""
    from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score
    auc = float(roc_auc_score(y_true, proba))
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-9)
    k = int(np.argmax(f1s))
    pred = (proba >= thr[k]).astype(int)
    order = np.argsort(rec)
    p_at_r60 = float(np.interp(0.60, rec[order], prec[order]))
    return {
        "auc": round(auc, 4),
        "best_f1": round(float(f1s[k]), 4),
        "threshold": round(float(thr[k]), 3),
        "precision@best_f1": round(float(prec[k]), 4),
        "recall@best_f1": round(float(rec[k]), 4),
        "accuracy@best_f1": round(float(accuracy_score(y_true, pred)), 4),
        "precision@recall0.6": round(p_at_r60, 4),
    }


def train_eval_config(name, algo, task, train, val, stats, feature_set, overrides,
                      use_asof=False):
    """stats = asof 包装 {snapshots, full}。训练集用 as-of（诚实），验证集用 full。"""
    t0 = time.time()
    flat = stats["full"] if isinstance(stats, dict) and "full" in stats else stats
    asof = stats if (use_asof and isinstance(stats, dict) and "snapshots" in stats) else None
    x_tr, y_tr = train_mod.get_xy(train, task, flat, feature_set, asof_stats=asof)
    x_va, y_va = train_mod.get_xy(val, task, flat, feature_set)

    if algo == "catboost":
        model = train_mod.train_catboost(task, x_tr, y_tr, x_va, y_va, False, overrides)
        proba = np.asarray(model.predict_proba(x_va)[:, 1])
    else:
        model, cats = train_mod.train_lightgbm(task, x_tr, y_tr, x_va, y_va, False, overrides)
        proba = np.asarray(model.predict(align_categories(x_va, cats),
                                         num_iteration=model.best_iteration))
    m = evaluate_classification(y_va.to_numpy(), proba)
    m["seconds"] = round(time.time() - t0, 1)
    print(f"[{name:28s}] AUC={m['auc']:.4f} F1={m['best_f1']:.4f} "
          f"P={m['precision@best_f1']:.3f} R={m['recall@best_f1']:.3f} "
          f"acc={m['accuracy@best_f1']:.4f}  ({m['seconds']}s)", flush=True)
    return {"name": name, "algo": algo, "feature_set": feature_set,
            "use_asof": use_asof, "overrides": overrides or {}, "metrics": m}


def main():
    ap = argparse.ArgumentParser(description="仅用验证集调参（不碰测试集）")
    ap.add_argument("--data", default=str(config.CLEAN_CSV))
    ap.add_argument("--task", default="cls_dep_delay")
    ap.add_argument("--out", default="tuning_results.json")
    args = ap.parse_args()

    print(f"读取 {args.data}")
    df = make_targets(pd.read_csv(args.data, low_memory=False))
    year = year_series(df)
    train = df[year.isin(config.TRAIN_YEARS)].copy()
    val = df[year.isin(config.VAL_YEARS)].copy()
    del df
    print(f"训练 {len(train):,} 行 / 验证 {len(val):,} 行（2023 测试集不加载！）")

    hist = compute_asof_stats(train)
    print(f"历史统计(as-of): route_month={len(hist['full']['route_month']):,} "
          f"全局延误率={hist['full']['global'][0]:.4f}\n", flush=True)

    CAT = "catboost"; LGB = "lightgbm"
    candidates = [
        # ---- CatBoost：v1 超参数变体（当前最优特征集）----
        ("CB-v1-基线",               CAT, "v1", None, False),
        ("CB-v1-lr0.05",             CAT, "v1", {"learning_rate": 0.05}, False),
        ("CB-v1-lr0.05-d10",         CAT, "v1", {"learning_rate": 0.05, "depth": 10}, False),
        ("CB-v1-lr0.05-d6-l2=9",     CAT, "v1", {"learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 9.0}, False),
        ("CB-v1-lr0.15-d8-l2=6",     CAT, "v1", {"learning_rate": 0.15, "l2_leaf_reg": 6.0}, False),
        # ---- CatBoost：v2 as-of 修复版（诚实历史特征是否带来增益）----
        ("CB-v2asof-lr0.1",          CAT, "v2", None, True),
        ("CB-v2asof-lr0.05",         CAT, "v2", {"learning_rate": 0.05}, True),
        # ---- LightGBM ----
        ("LGB-v1-基线",              LGB, "v1", None, False),
        ("LGB-v1-lr0.05-l63-mcs300", LGB, "v1", {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 300}, False),
        ("LGB-v1-lr0.05-l255",       LGB, "v1", {"learning_rate": 0.05, "num_leaves": 255}, False),
    ]

    results = []
    for name, algo, fs, ov, asof in candidates:
        try:
            results.append(train_eval_config(name, algo, args.task, train, val,
                                             hist, fs, ov, use_asof=asof))
        except Exception as e:
            print(f"[{name}] 失败: {e}", flush=True)

    results.sort(key=lambda r: r["metrics"]["auc"], reverse=True)
    print("\n===== 验证集(2022)排名（按AUC） =====")
    for i, r in enumerate(results, 1):
        m = r["metrics"]
        print(f"{i:2d}. {r['name']:28s} AUC={m['auc']:.4f} F1={m['best_f1']:.4f} "
              f"P={m['precision@best_f1']:.3f} R={m['recall@best_f1']:.3f} "
              f"acc={m['accuracy@best_f1']:.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"task": args.task, "tuned_at": datetime.now().isoformat(timespec="seconds"),
                   "note": "仅使用2019-2021训练/2022验证，未触碰测试集",
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {args.out}")


if __name__ == "__main__":
    main()
