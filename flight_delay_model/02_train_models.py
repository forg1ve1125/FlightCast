# -*- coding: utf-8 -*-
"""
步骤 2：特征工程 + 训练模型（CatBoost / LightGBM）

任务（会议纪要）：
  cls_dep_delay  分类：出发延误概率 P(DEP_DELAY > 0)
  cls_arr_delay  分类：到达延误概率 P(ARR_DELAY > 0)
  reg_dep_delay  回归：出发延误分钟数（仅在实际延误>0 的航班上训练，两段式第二段）
  reg_arr_delay  回归：到达延误分钟数

数据划分（会议纪要）：2019-2021 训练 / 2022 验证（早停+调参，可反复用）/ 2023 测试（本脚本不触碰）。

防过拟合：早停（early stopping）、L2正则、限制树复杂度、类别不平衡加权。

存档（会议纪要要求）：每个模型保存在 models/<run_id>/ 下，
并写 metadata.json 记录"是怎么训练的"（特征、超参数、划分、指标、随机种子、库版本等）。

用法：
    python 02_train_models.py                                  # 完整训练全部 8 个模型
    python 02_train_models.py --algos lightgbm                 # 只训练 LightGBM
    python 02_train_models.py --tasks cls_dep_delay            # 只训练指定任务
    python 02_train_models.py --quick                          # 冒烟测试模式（抽样+少迭代）
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    align_categories,
    build_features,
    compute_asof_stats,
    make_targets,
    split_by_year,
    task_mask,
    year_series,
)

# ------------------------------------------------------------------ 超参数
CATBOOST_PARAMS = {
    "cls": dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1500,
        learning_rate=0.1,
        depth=8,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",   # 类别不平衡（延误率约18%）
        random_seed=config.RANDOM_SEED,
        verbose=200,
        allow_writing_files=False,
    ),
    "reg": dict(
        loss_function="MAE",             # 延误分钟数长尾，MAE更稳健
        eval_metric="MAE",
        iterations=1500,
        learning_rate=0.1,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=config.RANDOM_SEED,
        verbose=200,
        allow_writing_files=False,
    ),
}
CATBOOST_EARLY_STOPPING = 150

LIGHTGBM_PARAMS = {
    "cls": dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,              # 2022验证集调优选定（02b_tune_on_validation.py）
        num_leaves=63,
        min_child_samples=300,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        seed=config.RANDOM_SEED,
        verbose=-1,
        force_row_wise=True,
    ),
    "reg": dict(
        objective="regression_l1",       # MAE
        metric="l1",
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=300,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        seed=config.RANDOM_SEED,
        verbose=-1,
        force_row_wise=True,
    ),
}
LIGHTGBM_NUM_ROUNDS = 1500
LIGHTGBM_EARLY_STOPPING = 150

QUICK_PARAMS_OVERRIDE = dict(iterations=120, learning_rate=0.2, depth=6, num_leaves=63)
QUICK_N_ROUNDS = 120
QUICK_MAX_TRAIN_ROWS = 150_000   # 冒烟模式训练集抽样上限


def load_clean_data(csv_path):
    print(f"读取清洗后数据集: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  共 {len(df):,} 行 x {df.shape[1]} 列")
    return df


def split_dataset(df):
    """按年份划分 train/val/test（会议纪要规定）。"""
    year = year_series(df)
    train = df[year.isin(config.TRAIN_YEARS)].copy()
    val = df[year.isin(config.VAL_YEARS)].copy()
    test = df[year.isin(config.TEST_YEARS)].copy()
    print(
        f"数据划分: 训练集 {TRAIN_YEARS_STR}: {len(train):,} 行 | "
        f"验证集 {VAL_YEARS_STR}: {len(val):,} 行 | "
        f"测试集 {TEST_YEARS_STR}: {len(test):,} 行（本次训练不使用）"
    )
    return train, val, test


TRAIN_YEARS_STR = "-".join(map(str, config.TRAIN_YEARS))
VAL_YEARS_STR = "-".join(map(str, config.VAL_YEARS))
TEST_YEARS_STR = "-".join(map(str, config.TEST_YEARS))


def get_xy(df, task, hist_stats=None, feature_set="v2", asof_stats=None):
    """按任务返回 (X, y)。已按 task_mask 过滤（去取消/改航/缺失；回归再要求延误>0）。
    asof_stats 提供时（训练集），历史特征按'只用过去年份'逐 year 构造，防止自泄漏。"""
    mask = task_mask(df, task)
    sub = df[mask]
    if asof_stats is not None:
        parts = []
        yr = year_series(sub)
        for _, sub_y in sub.groupby(yr.to_numpy()):
            key = str(int(pd.to_datetime(sub_y["FL_DATE"]).dt.year.iloc[0]))
            snap = asof_stats["snapshots"].get(key) or {"global": asof_stats["full"]["global"],
                                                        "route_month": {}, "origin_month": {},
                                                        "dest_month": {}, "airline": {}}
            parts.append(build_features(sub_y, hist_stats=snap, feature_set=feature_set))
        # 按年份拼接后必须恢复原始行序，保证 X 与 y 逐行对齐
        x = pd.concat(parts).reindex(sub.index)
    else:
        x = build_features(sub, hist_stats=hist_stats, feature_set=feature_set)
    y_col = "y_cls_dep" if task.startswith("cls_dep") else \
            "y_cls_arr" if task.startswith("cls_arr") else \
            "y_reg_dep" if task.startswith("reg_dep") else "y_reg_arr"
    y = sub[y_col].astype("float32")
    return x, y


# ------------------------------------------------------------------ CatBoost
def train_catboost(task, x_train, y_train, x_val, y_val, quick,
                   param_overrides=None):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    params = dict(CATBOOST_PARAMS["cls" if task.startswith("cls") else "reg"])
    if quick:
        for k, v in QUICK_PARAMS_OVERRIDE.items():
            if k in params:
                params[k] = v
    if param_overrides:
        params.update({k: v for k, v in param_overrides.items() if k in params})
    params["early_stopping_rounds"] = 30 if quick else CATBOOST_EARLY_STOPPING

    cat_idx = [x_train.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    train_pool = Pool(x_train, y_train, cat_features=cat_idx)
    val_pool = Pool(x_val, y_val, cat_features=cat_idx)

    model_cls = CatBoostClassifier if task.startswith("cls") else CatBoostRegressor
    model = model_cls(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=params["verbose"])
    return model


# ------------------------------------------------------------------ LightGBM
def train_lightgbm(task, x_train, y_train, x_val, y_val, quick,
                   param_overrides=None):
    import lightgbm as lgb

    params = dict(LIGHTGBM_PARAMS["cls" if task.startswith("cls") else "reg"])
    if quick:
        params.update(learning_rate=QUICK_PARAMS_OVERRIDE["learning_rate"], num_leaves=QUICK_PARAMS_OVERRIDE["num_leaves"])
    if param_overrides:
        params.update({k: v for k, v in param_overrides.items() if k in params or k in ("scale_pos_weight",)})
    params["scale_pos_weight"] = float((y_train == 0).sum() / max((y_train == 1).sum(), 1)) if task.startswith("cls") else None
    params = {k: v for k, v in params.items() if v is not None}

    x_tr = x_train.copy()
    x_va = x_val.copy()
    categories = {}
    for c in CATEGORICAL_FEATURES:
        cats = sorted(x_train[c].unique())
        categories[c] = cats
        x_tr[c] = pd.Categorical(x_tr[c], categories=cats)
        x_va[c] = pd.Categorical(x_va[c], categories=cats)

    train_set = lgb.Dataset(x_tr, label=y_train)
    val_set = lgb.Dataset(x_va, label=y_val, reference=train_set)
    num_rounds = QUICK_N_ROUNDS if quick else LIGHTGBM_NUM_ROUNDS
    callbacks = [lgb.early_stopping(30 if quick else LIGHTGBM_EARLY_STOPPING, verbose=False),
                 lgb.log_evaluation(period=200)]
    booster = lgb.train(params, train_set, num_boost_round=num_rounds,
                        valid_sets=[val_set], callbacks=callbacks)
    return booster, categories


# ------------------------------------------------------------------ 验证集指标
def validate(model, algo, task, x_val, y_val):
    """在验证集(2022)上算指标，写入metadata（测试集2023不在此触碰）。"""
    if algo == "catboost":
        pred = model.predict(x_val)
        proba = model.predict_proba(x_val)[:, 1] if task.startswith("cls") else None
        best_iter = model.get_best_iteration()
    else:
        pred = model.predict(x_val, num_iteration=model.best_iteration)
        proba = model.predict(x_val, num_iteration=model.best_iteration) if task.startswith("cls") else None
        best_iter = model.best_iteration

    y = y_val.to_numpy()
    metrics = {}
    if task.startswith("cls"):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        pred_bin = (np.asarray(proba) >= 0.5).astype(int)
        metrics = {
            "threshold": 0.5,
            "accuracy": round(float(accuracy_score(y, pred_bin)), 4),
            "precision": round(float(precision_score(y, pred_bin, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred_bin, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred_bin, zero_division=0)), 4),
            "auc": round(float(roc_auc_score(y, proba)), 4),
        }
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        metrics = {
            "mae_min": round(float(mean_absolute_error(y, pred)), 2),
            "rmse_min": round(float(np.sqrt(mean_squared_error(y, pred))), 2),
            "r2": round(float(r2_score(y, pred)), 4),
        }
    metrics["best_iteration"] = int(best_iter) if best_iter is not None else None
    return metrics


def train_one(algo, task, train, val, out_dir, quick, run_id,
              hist_stats=None, asof_stats=None, feature_set="v2", param_overrides=None):
    t0 = time.time()
    x_train, y_train = get_xy(train, task, hist_stats, feature_set, asof_stats=asof_stats)
    x_val, y_val = get_xy(val, task, hist_stats, feature_set)
    if quick and len(x_train) > QUICK_MAX_TRAIN_ROWS:
        idx = np.random.RandomState(config.RANDOM_SEED).choice(
            len(x_train), QUICK_MAX_TRAIN_ROWS, replace=False)
        x_train, y_train = x_train.iloc[idx], y_train.iloc[idx]
    print(f"\n=== [{algo}] {task} : 训练 {len(x_train):,} 行 / 验证 {len(x_val):,} 行 ===")

    feat_list = FEATURES if feature_set == "v2" else __import__("features").FEATURES_V1
    categories = None
    if algo == "catboost":
        model = train_catboost(task, x_train, y_train, x_val, y_val, quick,
                               param_overrides)
        model_path = out_dir / f"{task}__catboost.cbm"
        model.save_model(str(model_path))
    else:
        model, categories = train_lightgbm(task, x_train, y_train, x_val, y_val, quick,
                                           param_overrides)
        model_path = out_dir / f"{task}__lightgbm.txt"
        model.save_model(str(model_path), num_iteration=model.best_iteration)
        # 验证/预测时分类特征必须对齐到训练时的类别集合
        x_val = align_categories(x_val, categories)

    val_metrics = validate(model, algo, task, x_val, y_val)
    duration = round(time.time() - t0, 1)
    print(f"验证集(2022)指标: {val_metrics}  耗时 {duration}s")

    base_params = (CATBOOST_PARAMS if algo == "catboost" else LIGHTGBM_PARAMS)[
        "cls" if task.startswith("cls") else "reg"]
    effective_params = dict(base_params)
    if param_overrides:
        effective_params.update(param_overrides)

    metadata = {
        "run_id": run_id,
        "task": task,
        "task_desc": config.TASKS[task],
        "algo": algo,
        "feature_set": feature_set,
        "quick_mode": quick,
        "model_file": model_path.name,
        "model_path": str(model_path),
        "features": feat_list,
        "categorical_features": CATEGORICAL_FEATURES,
        "historical_stats_file": "historical_stats.json" if hist_stats is not None else None,
        "categories": categories,  # LightGBM 预测时需对齐类别
        "target": ("DEP_DELAY" if "dep" in task else "ARR_DELAY") + (
            " > 0min (1/0)" if task.startswith("cls") else " 分钟数(仅延误>0样本)"),
        "delay_threshold_min": config.DELAY_THRESHOLD_MIN,
        "split": {
            "train_years": list(config.TRAIN_YEARS),
            "val_years": list(config.VAL_YEARS),
            "test_years": list(config.TEST_YEARS),
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
        },
        "hyperparams": effective_params,
        "param_overrides": param_overrides or {},
        "early_stopping_rounds": 30 if quick else (CATBOOST_EARLY_STOPPING if algo == "catboost" else LIGHTGBM_EARLY_STOPPING),
        "random_seed": config.RANDOM_SEED,
        "val_metrics_2022": val_metrics,
        "train_duration_sec": duration,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "catboost": __import__("catboost").__version__ if algo == "catboost" else None,
            "lightgbm": __import__("lightgbm").__version__ if algo == "lightgbm" else None,
        },
    }
    with open(out_dir / f"{task}__{algo}__metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def make_run_id(run_name=None, quick=False, ts=None):
    """存档目录名统一为 时间戳[_run名][_quick]，如 20261006_142130_final_gt0。"""
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [ts]
    if run_name:
        parts.append(run_name)
    if quick:
        parts.append("quick")
    return "_".join(parts)


def run_training(data_csv, algos, tasks, quick, models_dir=None, run_id=None,
                 run_name=None, feature_set="v2", param_overrides=None):
    models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
    run_id = run_id or make_run_id(run_name, quick)
    out_dir = models_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean_data(data_csv)
    df = make_targets(df)
    train, val, test = split_dataset(df)

    # 历史统计只用训练集年份计算，保存到存档目录（预测/评估时需要）。
    # as-of 包装：训练集每年只用"过去年份"的统计，防止训练内自泄漏。
    hist_stats = asof_stats = None
    if feature_set == "v2":
        print("计算历史延误统计（as-of：每年只用过去年份，防止自泄漏）...")
        asof_stats = compute_asof_stats(train)
        hist_stats = asof_stats["full"]
        with open(out_dir / "historical_stats.json", "w", encoding="utf-8") as f:
            json.dump(asof_stats, f, ensure_ascii=False)
        print(f"  full: route_month={len(hist_stats['route_month']):,} 条, "
              f"origin_month={len(hist_stats['origin_month']):,}, "
              f"dest_month={len(hist_stats['dest_month']):,}, "
              f"airline={len(hist_stats['airline']):,}, "
              f"全局延误率={hist_stats['global'][0]:.4f}")

    all_meta = []
    for algo in algos:
        for task in tasks:
            meta = train_one(algo, task, train, val, out_dir, quick, run_id,
                             hist_stats=hist_stats, asof_stats=asof_stats,
                             feature_set=feature_set,
                             param_overrides=param_overrides)
            all_meta.append(meta)

    run_info = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "quick_mode": quick,
        "feature_set": feature_set,
        "data_csv": str(data_csv),
        "models": [m["model_path"] for m in all_meta],
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    if models_dir == config.MODELS_DIR:
        with open(config.LATEST_RUN_JSON, "w", encoding="utf-8") as f:
            json.dump(run_info, f, ensure_ascii=False, indent=2)

    print(f"\n全部完成。模型存档目录: {out_dir}")
    return run_info


def main():
    ap = argparse.ArgumentParser(description="训练延误预测模型（CatBoost/LightGBM）")
    ap.add_argument("--data", default=str(config.CLEAN_CSV), help="清洗后的CSV路径")
    ap.add_argument("--algos", default="catboost,lightgbm", help="逗号分隔: catboost,lightgbm")
    ap.add_argument("--tasks", default=",".join(config.TASKS), help="逗号分隔的任务名")
    ap.add_argument("--quick", action="store_true", help="冒烟测试模式（抽样+少迭代，快速验证流程）")
    ap.add_argument("--run-name", default=None,
                    help="存档名（自动加时间戳前缀，如 --run-name final_gt0 -> 20261006_142130_final_gt0）")
    ap.add_argument("--run-id", default=None, help="完整存档目录名（优先于 --run-name，需自带时间戳）")
    ap.add_argument("--feature-set", default="v2", choices=["v1", "v2"],
                    help="v1=原始11特征; v2=+节假日+历史延误统计（默认）")
    args = ap.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in config.TASKS:
            sys.exit(f"未知任务: {t}，可选: {list(config.TASKS)}")

    run_training(args.data, algos, tasks, args.quick, run_id=args.run_id,
                 run_name=args.run_name, feature_set=args.feature_set)


if __name__ == "__main__":
    main()
