# -*- coding: utf-8 -*-
"""
步骤 7：改进空间对照实验（全部只在 2022 验证集上评估，不触碰测试集）

回答三个问题：
  E1 概率集成（CatBoost+LightGBM 平均）能否提升？
  E2 更小学习率+更多轮数（CatBoost）还有空间吗？
  E3 新增15分钟时刻粒度特征（当前只用小时）是否有增益？
  E4 换 XGBoost 是否更强？
  E5 单段式回归（直接预测 max(延误,0)）vs 两段式，哪个更适合"期望式延误"产品？
  E6 概率校准（期望延误 = P×分钟，P 的绝对准度很关键）现有模型校准如何？
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, precision_score,
                             recall_score, roc_auc_score)

import config
from features import build_features, make_targets, task_mask, year_series
from importlib import import_module

eval_mod = import_module("03_evaluate_models")

RUN = "20261005_155440_final_gt0"


def cls_report(y, proba):
    pred = (np.asarray(proba) >= 0.5).astype(int)
    return {"auc": round(float(roc_auc_score(y, proba)), 4),
            "acc": round(float(accuracy_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4)}


def reg_report(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = np.abs(pred - y)
    return {"mae": round(float(err.mean()), 2),
            "within_5": round(float((err <= 5).mean()), 4),
            "within_10": round(float((err <= 10).mean()), 4)}


def main():
    t00 = time.time()
    df = make_targets(pd.read_csv(config.CLEAN_CSV, low_memory=False))
    year = year_series(df)
    train = df[year.isin(config.TRAIN_YEARS)]
    val = df[year.isin(config.VAL_YEARS)].copy()
    del df

    m_tr = task_mask(train, "cls_dep_delay")
    m_va = task_mask(val, "cls_dep_delay")
    y_tr = train[m_tr]["y_cls_dep"].astype(int).to_numpy()
    y_va = val[m_va]["y_cls_dep"].astype(int).to_numpy()
    x_tr = build_features(train[m_tr], feature_set="v1")
    x_va = build_features(val[m_va], feature_set="v1")
    print(f"训练 {len(y_tr):,} / 验证 {len(y_va):,}，正例率 val={y_va.mean():.4f}", flush=True)

    results = {}

    # ---------- 基线：现有 final_gt0 两个模型 ----------
    run_dir, metas = eval_mod.load_run(RUN)
    cb_meta = next(m for m in metas if m["task"] == "cls_dep_delay" and m["algo"] == "catboost")
    lg_meta = next(m for m in metas if m["task"] == "cls_dep_delay" and m["algo"] == "lightgbm")
    p_cb = np.asarray(eval_mod.predict(eval_mod.load_model(cb_meta), cb_meta, x_va))
    p_lg = np.asarray(eval_mod.predict(eval_mod.load_model(lg_meta), lg_meta, x_va))
    results["base_catboost"] = cls_report(y_va, p_cb)
    results["base_lightgbm"] = cls_report(y_va, p_lg)
    print("基线:", {k: v for k, v in results.items() if k.startswith('base')}, flush=True)

    # ---------- E1 概率集成 ----------
    p_ens = (p_cb + p_lg) / 2
    results["E1_ensemble_mean"] = cls_report(y_va, p_ens)
    print("E1 集成:", results["E1_ensemble_mean"], flush=True)

    # ---------- E2 CatBoost 更小学习率 ----------
    from catboost import CatBoostClassifier, Pool
    cat_idx = [x_tr.columns.get_loc(c) for c in ["AIRLINE", "ORIGIN", "DEST", "ROUTE"]]
    t0 = time.time()
    m2 = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", iterations=3000,
                            learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                            auto_class_weights="Balanced", random_seed=config.RANDOM_SEED,
                            early_stopping_rounds=200, verbose=0, allow_writing_files=False)
    m2.fit(Pool(x_tr, y_tr, cat_features=cat_idx),
           eval_set=Pool(x_va, y_va, cat_features=cat_idx))
    p_e2 = m2.predict_proba(x_va)[:, 1]
    results["E2_cb_lr0.03"] = cls_report(y_va, p_e2)
    results["E2_cb_lr0.03"]["best_iter"] = int(m2.get_best_iteration())
    results["E2_cb_lr0.03"]["seconds"] = round(time.time() - t0, 1)
    print("E2 慢学习率:", results["E2_cb_lr0.03"], flush=True)

    # ---------- E3 15分钟时刻粒度特征 ----------
    def add_slot(x_src, df_src):
        hhmm = pd.to_numeric(df_src["CRS_DEP_TIME"]).to_numpy()
        slot = (hhmm // 100 * 60 + hhmm % 100) // 15
        out = x_src.copy()
        out["DEP_SLOT15"] = slot.astype("int16")
        return out

    xs_tr = add_slot(x_tr, train[m_tr])
    xs_va = add_slot(x_va, val[m_va])
    cat_idx_s = cat_idx
    t0 = time.time()
    m3 = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", iterations=1500,
                            learning_rate=0.1, depth=8, l2_leaf_reg=3.0,
                            auto_class_weights="Balanced", random_seed=config.RANDOM_SEED,
                            early_stopping_rounds=150, verbose=0, allow_writing_files=False)
    m3.fit(Pool(xs_tr, y_tr, cat_features=cat_idx_s),
           eval_set=Pool(xs_va, y_va, cat_features=cat_idx_s))
    p_e3 = m3.predict_proba(xs_va)[:, 1]
    results["E3_dep_slot15"] = cls_report(y_va, p_e3)
    results["E3_dep_slot15"]["seconds"] = round(time.time() - t0, 1)
    print("E3 15分钟粒度:", results["E3_dep_slot15"], flush=True)

    # ---------- E4 XGBoost ----------
    try:
        from xgboost import XGBClassifier
        xg_tr, xg_va = x_tr.copy(), x_va.copy()
        for c in ["AIRLINE", "ORIGIN", "DEST", "ROUTE"]:
            cats = sorted(x_tr[c].unique())
            xg_tr[c] = pd.Categorical(xg_tr[c], categories=cats)
            xg_va[c] = pd.Categorical(xg_va[c], categories=cats)
        t0 = time.time()
        m4 = XGBClassifier(n_estimators=1500, learning_rate=0.1, max_depth=8,
                           reg_lambda=1.0, tree_method="hist", enable_categorical=True,
                           random_state=config.RANDOM_SEED, eval_metric="auc",
                           early_stopping_rounds=150, n_jobs=-1)
        m4.fit(xg_tr, y_tr, eval_set=[(xg_va, y_va)], verbose=False)
        p_e4 = m4.predict_proba(xg_va)[:, 1]
        results["E4_xgboost"] = cls_report(y_va, p_e4)
        results["E4_xgboost"]["best_iter"] = int(m4.best_iteration)
        results["E4_xgboost"]["seconds"] = round(time.time() - t0, 1)
        print("E4 XGBoost:", results["E4_xgboost"], flush=True)
    except Exception as e:
        results["E4_xgboost"] = {"error": str(e)}
        print("E4 失败:", e, flush=True)

    # ---------- E5 单段式回归期望延误 vs 两段式 ----------
    from catboost import CatBoostRegressor
    y_all_tr = train[m_tr]["DEP_DELAY"].astype(float).clip(lower=0).to_numpy()
    y_all_va = val[m_va]["DEP_DELAY"].astype(float).clip(lower=0).to_numpy()
    t0 = time.time()
    m5 = CatBoostRegressor(loss_function="MAE", eval_metric="MAE", iterations=1500,
                           learning_rate=0.1, depth=8, l2_leaf_reg=3.0,
                           random_seed=config.RANDOM_SEED, early_stopping_rounds=150,
                           verbose=0, allow_writing_files=False)
    m5.fit(Pool(x_tr, y_all_tr, cat_features=cat_idx),
           eval_set=Pool(x_va, y_all_va, cat_features=cat_idx))
    p_min = np.asarray(eval_mod.predict(
        eval_mod.load_model(next(m for m in metas if m["task"] == "reg_dep_delay" and m["algo"] == "catboost")),
        next(m for m in metas if m["task"] == "reg_dep_delay" and m["algo"] == "catboost"),
        build_features(val[task_mask(val, "reg_dep_delay")], feature_set="v1")))
    delayed_va = m_va & (val["DEP_DELAY"].fillna(-1) > 0)
    reg_aligned = pd.Series(p_min, index=val.index[delayed_va])
    expected_two_stage = p_cb * reg_aligned.reindex(val.index[m_va]).fillna(0).to_numpy()
    results["E5_single_stage_reg"] = reg_report(y_all_va, m5.predict(x_va))
    results["E5_single_stage_reg"]["seconds"] = round(time.time() - t0, 1)
    results["E5_two_stage"] = reg_report(y_all_va, expected_two_stage)
    print("E5 单段式:", results["E5_single_stage_reg"], flush=True)
    print("E5 两段式:", results["E5_two_stage"], flush=True)

    # ---------- E6 概率校准 ----------
    calib = {}
    for name, p in [("catboost", p_cb), ("lightgbm", p_lg), ("ensemble", p_ens)]:
        dec = pd.qcut(pd.Series(p), 10, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y_va}).groupby(dec, observed=True).mean()
        calib[name] = {"brier": round(float(brier_score_loss(y_va, p)), 4),
                       "decile_pred_vs_actual": [[round(float(r.p), 3), round(float(r.y), 3)]
                                                 for r in g.itertuples()]}
    results["E6_calibration"] = calib
    print("E6 Brier:", {k: v["brier"] for k, v in calib.items()}, flush=True)

    out = run_dir / "experiments_val2022.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"run": RUN, "total_seconds": round(time.time() - t00, 1),
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {out}，总耗时 {round(time.time()-t00)}s")


if __name__ == "__main__":
    main()
