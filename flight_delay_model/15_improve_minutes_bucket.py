# -*- coding: utf-8 -*-
"""
步骤 15：五档/分钟组件的针对性优化（A/C/D 三方案 + 可选 --drop-2020 假设检验）。

背景（2026-10-13 基线对照发现的三个病灶）：
  A. 两段式期望延误在诚实口径下输给"常数0"基线（17.10 vs 15.40，2022 验证）——
     架构选择曾被 06 号评估泄露带偏（12.93 虚高），修正后需要重做架构对照：
     单段式（直接学 max(DEP_DELAY,0)，全体航班，MAE 损失=中位回归）。
  C. 数据集内还有一块干净的增量信息没用：机场×时段的"典型计划流量"
     （只用训练年统计、纯计划口径——线上可用同构特征，无泄露、无训练/部署漂移）。
     注意：不能用"当天实际架次"——线上预测未来日期时拿不到当天全国排班。
  D. 五档概率在 2022 延误率漂移（训练~30% -> 2022 39%）下校准偏差 3.5-3.8pp，
     用 2022 上半年拟合 isotonic、下半年验证（2023 不触碰）。
  B.（可选 --drop-2020）剔除 2020 疫情年的假设检验：不是决策，是让证据说话——
     带标志跑一遍与本报告对照，赢了才考虑采纳。

对照物（全部同一批 2022 验证行）：
  - 常数0基线（期望延误永远猜 0 的 MAE）——A 必须赢它
  - 两段式 od（现役：cls od × reg od，17.10）
  - 五档 od 现役模型（0.6108/0.1666/17.44）——C/D 的参照

纪律：训练 2019-2021（--drop-2020 时剔除其中 2020 行）/ 裁决只用 2022 验证 /
2023 完全不读取 / 不修改任何生产指针。断点续训：模型+metadata 存在则跳过。

用法：
    python 15_improve_minutes_bucket.py             # A + C + D（约45分钟，其中五档占40）
    python 15_improve_minutes_bucket.py --drop-2020 # 同上但训练行剔除 2020（假设检验）
    python 15_improve_minutes_bucket.py --quick      # 冒烟（约3分钟，指标仅供参考）
"""
import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from features import (CATEGORICAL_FEATURES, build_features, make_targets,
                      task_mask, year_series)
from importlib import import_module

train_mod = import_module("02_train_models")
m08 = import_module("08_train_bucket_models")       # multiclass_val_metrics
m11 = import_module("11_train_weather")             # 双端天气
m12 = import_module("12_train_bucket_weather")      # expected_delay_mae

# 现役参照（存档冻结，只加载不重训）
OD_CLS = ("20261009_193645_wx_dep", "cls_dep_delay__v1wx_od__catboost.cbm")
OD_REG = ("20261013_130457_wx_reg_dep", "reg_dep_delay__v1wx_od__catboost.cbm")
OD_BUCKET = ("20261013_122152_buckets_wx_dep", "mcls_dep_delay__v1wx_od__catboost.cbm")
VOL_FEATURES = ["TYP_ORIG_HR_VOL", "TYP_DEST_HR_VOL"]


# ---------------------------------------------------------------- C：典型流量特征
def typical_volume_stats(train_df):
    """训练年限内 每个(机场, 小时) 的典型计划架次/日（按日期计数后取中位数）。

    只用计划口径字段（FL_DATE/机场/CRS 时刻），无任何结果字段 -> 无泄露；
    线上可用同一张表查表 -> 训练/部署同构（这是不用"当天实际架次"的原因）。
    """
    dep_h = (pd.to_numeric(train_df["CRS_DEP_TIME"], errors="coerce") // 100)
    arr_h = (pd.to_numeric(train_df["CRS_ARR_TIME"], errors="coerce") // 100)

    def _stat(airport, hours, tag):
        g = pd.DataFrame({"k": airport + "|" + hours.astype("Int64").astype(str),
                          "d": train_df["FL_DATE"].astype(str)})
        per_day = g.groupby(["k", "d"]).size().reset_index(name="n")
        med = per_day.groupby("k")["n"].median()
        return {k: round(float(v), 2) for k, v in med.items()}

    return {"origin_dep": _stat(train_df["ORIGIN"], dep_h, "o"),
            "dest_arr": _stat(train_df["DEST"], arr_h, "d"),
            "_note": "训练年(2019-2021)每个机场×小时的典型计划架次中位数（每日期计数后取中位）"}


def attach_volume_features(df, stats):
    """把典型流量表映射到任意行（查不到用全局中位数，与线上兜底一致）。"""
    all_vals = [v for v in stats["origin_dep"].values()] + \
               [v for v in stats["dest_arr"].values()]
    fallback = float(np.median(all_vals)) if all_vals else 0.0
    dep_h = (pd.to_numeric(df["CRS_DEP_TIME"], errors="coerce") // 100)
    arr_h = (pd.to_numeric(df["CRS_ARR_TIME"], errors="coerce") // 100)
    df = df.copy()
    df["TYP_ORIG_HR_VOL"] = (df["ORIGIN"] + "|" + dep_h.astype("Int64").astype(str)) \
        .map(stats["origin_dep"]).astype(float).fillna(fallback)
    df["TYP_DEST_HR_VOL"] = (df["DEST"] + "|" + arr_h.astype("Int64").astype(str)) \
        .map(stats["dest_arr"]).astype(float).fillna(fallback)
    return df, fallback


# ---------------------------------------------------------------- 训练器
def train_catboost_reg(x_tr, y_tr, x_va, y_va, quick=False):
    from catboost import CatBoostRegressor, Pool
    cat_idx = [x_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    model = CatBoostRegressor(
        loss_function="MAE", eval_metric="MAE",
        iterations=300 if quick else 1500, learning_rate=0.1, depth=8,
        l2_leaf_reg=3.0, random_seed=config.RANDOM_SEED,
        early_stopping_rounds=50 if quick else 150,
        verbose=200, allow_writing_files=False)
    model.fit(Pool(x_tr, y_tr, cat_features=cat_idx),
              eval_set=Pool(x_va, y_va, cat_features=cat_idx), use_best_model=True)
    return model, cat_idx


def train_catboost_bucket(x_tr, y_tr, x_va, y_va, quick=False):
    from catboost import CatBoostClassifier, Pool
    cat_idx = [x_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    model = CatBoostClassifier(
        loss_function="MultiClass", eval_metric="MultiClass",
        iterations=300 if quick else 1200, learning_rate=0.1, depth=7,
        l2_leaf_reg=3.0, random_seed=config.RANDOM_SEED,
        early_stopping_rounds=50 if quick else 150,
        verbose=200, allow_writing_files=False)
    model.fit(Pool(x_tr, y_tr, cat_features=cat_idx),
              eval_set=Pool(x_va, y_va, cat_features=cat_idx), use_best_model=True)
    return model, cat_idx


def load_one(run_id, fname):
    path = config.MODELS_DIR / run_id / fname
    if not path.exists():
        raise SystemExit(f"缺少模型: {path}")
    meta = json.load(open(path.with_name(path.stem + "__metadata.json"), encoding="utf-8"))
    if meta["task"].startswith(("cls", "mcls")):
        from catboost import CatBoostClassifier as M
    else:
        from catboost import CatBoostRegressor as M
    model = M()
    model.load_model(str(path))
    return model, meta


def predict_with(model, meta, x_container):
    from catboost import Pool
    x = x_container[meta["features"]]
    ci = [x.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    if meta["task"].startswith("mcls"):
        return np.asarray(model.predict_proba(Pool(x, cat_features=ci)))
    if meta["task"].startswith("cls"):
        return np.asarray(model.predict_proba(Pool(x, cat_features=ci)))[:, 1]
    return np.asarray(model.predict(Pool(x, cat_features=ci))).ravel()


def expected_mae_metrics(y_clip, pred):
    return {"mae": round(float(np.mean(np.abs(pred - y_clip))), 2),
            "hit_5min": round(float(np.mean(np.abs(pred - y_clip) <= 5)), 4),
            "hit_10min": round(float(np.mean(np.abs(pred - y_clip) <= 10)), 4)}


# ---------------------------------------------------------------- D：isotonic 校准
def isotonic_calibrate(y_h1, proba_h1, y_h2, proba_h2):
    """2022H1 拟合逐类 isotonic，2022H2 应用后归一化。返回校准后的 H2 概率。"""
    from sklearn.isotonic import IsotonicRegression
    calib = []
    for k in range(config.N_BUCKETS):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        ir.fit(proba_h1[:, k], (y_h1 == k).astype(float))
        calib.append(ir)
    out = np.column_stack([c.predict(proba_h2[:, k]) for k, c in enumerate(calib)])
    out = np.clip(out, 1e-6, None)
    return out / out.sum(axis=1, keepdims=True)


def calib_pp(y, proba):
    """逐档 |平均预测概率 - 实际频率| 的均值（百分点）。"""
    devs = [abs(proba[:, k].mean() - (y == k).mean()) * 100
            for k in range(config.N_BUCKETS)]
    return round(float(np.mean(devs)), 2)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="单段式/典型流量/再校准（A/C/D，2022裁决）")
    ap.add_argument("--data", default=str(config.CLEAN_CSV))
    ap.add_argument("--run-name", default="improve")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--drop-2020", action="store_true",
                    help="假设检验：训练行剔除2020（对照本脚本不带标志的报告）")
    ap.add_argument("--only-a", action="store_true",
                    help="只跑 A（单段式，约3-5分钟），跳过 C/D（五档重训约40分钟）")
    args = ap.parse_args()

    tz_map = json.load(open(m11.WX_DIR / "tz_map.json", encoding="utf-8"))
    tag = "_d2020" if args.drop_2020 else ""
    run_id = args.run_id or train_mod.make_run_id(args.run_name + tag, quick=args.quick)
    out_dir = config.MODELS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "charts").mkdir(exist_ok=True)
    print(f"存档目录: {out_dir}{'（训练行剔除2020——假设检验）' if args.drop_2020 else ''}")

    # ---- 数据（2019-2022；2023 不读取） ----
    df = make_targets(pd.read_csv(args.data, low_memory=False))
    year = year_series(df)
    df = df[year.isin(list(config.TRAIN_YEARS) + list(config.VAL_YEARS))]
    df = m11.load_weather_features(df, tz_map)

    # C：典型流量表只用训练年统计（--drop-2020 时统计口径同步剔除，保持一致）
    train_full = df[year_series(df).isin(config.TRAIN_YEARS)]
    if args.drop_2020:
        train_full = train_full[year_series(train_full) != 2020]
    stats = typical_volume_stats(train_full)
    with open(out_dir / "congestion_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    df, vol_fallback = attach_volume_features(df, stats)
    print(f"典型流量特征: origin_dep {len(stats['origin_dep'])} 项 / "
          f"dest_arr {len(stats['dest_arr'])} 项，兜底值 {vol_fallback:.1f}")

    train = df[year_series(df).isin(config.TRAIN_YEARS)]
    if args.drop_2020:
        train = train[year_series(train) != 2020]
    val = df[year_series(df).isin(config.VAL_YEARS)].copy()

    m_cls = task_mask(val, "cls_dep_delay")
    va = val[m_cls]
    m_tr = task_mask(train, "cls_dep_delay")   # 单段式与两段式同口径：有有效延误标签的全体航班
    if args.quick:
        tr_s = train[m_tr].sample(min((m_tr).sum(), 120_000),
                                  random_state=config.RANDOM_SEED)
    else:
        tr_s = train[m_tr]
    y_tr_clip = np.clip(pd.to_numeric(tr_s["DEP_DELAY"], errors="coerce")
                        .to_numpy(), 0, None)
    x_v1_tr = build_features(tr_s, feature_set="v1").reset_index(drop=True)
    x_v1_va = build_features(va, feature_set="v1").reset_index(drop=True)
    wx_cols = m11.WX_ORIGIN_FEATURE_COLS + m11.WX_DEST_FEATURE_COLS
    base_tr = pd.concat([x_v1_tr, tr_s[wx_cols].reset_index(drop=True)], axis=1)
    base_va = pd.concat([x_v1_va, va[wx_cols].reset_index(drop=True)], axis=1)
    full_va = pd.concat([base_va, va[VOL_FEATURES].reset_index(drop=True)], axis=1)
    full_tr = pd.concat([base_tr, tr_s[VOL_FEATURES].reset_index(drop=True)], axis=1)

    y_all = np.clip(pd.to_numeric(va["DEP_DELAY"], errors="coerce").to_numpy(), 0, None)

    # ================= A：单段式回归 =================
    print(f"\n=== A. 单段式回归（目标 max(DEP_DELAY,0)，全体航班） 训练 {len(tr_s):,} / 验证 {len(va):,} ===")
    ref = {"constant_zero": expected_mae_metrics(y_all, np.zeros_like(y_all))}
    # 两段式 od 参照（现役模型，同一批验证行）
    cls_m, cls_meta = load_one(*OD_CLS)
    reg_m, reg_meta = load_one(*OD_REG)
    p_delay = predict_with(cls_m, cls_meta, base_va)
    p_min = predict_with(reg_m, reg_meta, base_va)
    ref["two_stage_od"] = expected_mae_metrics(y_all, p_delay * p_min)
    print(f"  参照: 常数0 MAE={ref['constant_zero']['mae']} | "
          f"两段式od MAE={ref['two_stage_od']['mae']}")

    single_results = {}
    for variant, x_tr, x_va_ in [("single39", base_tr, base_va),
                                 ("single41", full_tr, full_va)]:
        meta_path = out_dir / f"reg_dep_single__{variant}{tag}__metadata.json"
        model_path = out_dir / f"reg_dep_single__{variant}{tag}.cbm"
        t0 = time.time()
        if meta_path.exists() and model_path.exists():
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(); model.load_model(str(model_path))
            cat_idx = [x_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
            best_iter = int(model.get_best_iteration())
            print(f"  [{variant}] 已存在，跳过训练")
        else:
            model, cat_idx = train_catboost_reg(x_tr, y_tr_clip, x_va_, y_all,
                                                quick=args.quick)
            model.save_model(str(model_path))
            best_iter = int(model.get_best_iteration())
        from catboost import Pool
        pred = np.asarray(model.predict(Pool(x_va_, cat_features=cat_idx))).ravel()
        met = expected_mae_metrics(y_all, pred)
        met.update({"best_iteration": best_iter, "seconds": round(time.time() - t0, 1),
                    "pred_positive_share": round(float((pred > 0.5).mean()), 4),
                    "pred_mean": round(float(pred.mean()), 2)})
        single_results[variant] = met
        print(f"  [{variant}] MAE={met['mae']} ±5={met['hit_5min']:.1%} "
              f"±10={met['hit_10min']:.1%} 预测>0比例={met['pred_positive_share']:.1%} "
              f"均值={met['pred_mean']} (best_iter={best_iter}, {met['seconds']}s)")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_id, "task": "reg_dep_single",
                       "task_desc": f"单段式期望延误回归 max(DEP_DELAY,0) 全体航班 {variant}{tag}",
                       "algo": "catboost", "feature_set": f"single_{variant}{tag}",
                       "model_file": model_path.name, "model_path": str(model_path),
                       "features": list(x_tr.columns),
                       "categorical_features": CATEGORICAL_FEATURES,
                       "target": "max(DEP_DELAY,0) all flights, MAE loss (median regression)",
                       "split": {"train_years": list(config.TRAIN_YEARS),
                                 "drop_2020": bool(args.drop_2020),
                                 "val_years": list(config.VAL_YEARS)},
                       "val_metrics_2022": met, "references_2022": ref,
                       "trained_at": datetime.now().isoformat(timespec="seconds"),
                       "library_versions": {"python": platform.python_version(),
                                            "catboost": __import__("catboost").__version__},
                       "mode": "quick" if args.quick else "full"}, f,
                     ensure_ascii=False, indent=2)

    # ================= C：五档 + 典型流量（--only-a 时提前收尾） =================
    if args.only_a:
        payload = {
            "run_id": run_id, "mode": "quick" if args.quick else "full",
            "drop_2020": bool(args.drop_2020), "only_a": True,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "A_single_stage": {"references": ref, "arms": single_results},
            "note": "--only-a：仅A（单段式）；C/D 未运行；2022裁决；未改生产指针",
        }
        with open(out_dir / "improve_val2022.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        best_s = min(single_results, key=lambda k: single_results[k]["mae"])
        print("\n" + "=" * 72)
        print(f"[A 单段式] 常数0={ref['constant_zero']['mae']} | 两段式od={ref['two_stage_od']['mae']} "
              f"| 最优 {best_s}={single_results[best_s]['mae']} "
              f"({'✅ 赢常数' if single_results[best_s]['mae'] < ref['constant_zero']['mae'] else '❌ 未赢常数'})")
        print(f"指标: {out_dir / 'improve_val2022.json'}")
        print("=" * 72)
        return

    print(f"\n=== C. 五档区间 + 典型流量（对照现役 od 模型） ===")
    m_bucket = task_mask(val, "mcls_dep_delay")
    va_b = val[m_bucket]
    y_b = va_b["y_bucket_dep"].astype(int).to_numpy()
    delay_b = pd.to_numeric(va_b["DEP_DELAY"], errors="coerce").to_numpy()
    base_va_b = pd.concat([build_features(va_b, feature_set="v1").reset_index(drop=True),
                           va_b[wx_cols].reset_index(drop=True)], axis=1)
    full_va_b = pd.concat([base_va_b, va_b[VOL_FEATURES].reset_index(drop=True)], axis=1)

    bm, bmeta = load_one(*OD_BUCKET)
    proba_od = predict_with(bm, bmeta, base_va_b)
    med = [s["median_minutes"] for s in json.load(
        open(config.MODELS_DIR / OD_BUCKET[0] / "bucket_stats.json",
             encoding="utf-8"))["dep"]]
    od_bucket_met = {"accuracy": None, "macro_f1": None,
                     "calib_pp": calib_pp(y_b, proba_od),
                     "expected_mae": m12.expected_delay_mae(delay_b, proba_od, med)}
    mm = m08.multiclass_val_metrics(y_b, proba_od)
    od_bucket_met["accuracy"], od_bucket_met["macro_f1"] = mm["accuracy"], mm["macro_f1"]
    print(f"  现役od参照: acc={od_bucket_met['accuracy']} macroF1={od_bucket_met['macro_f1']} "
          f"校准={od_bucket_met['calib_pp']}pp 期望MAE={od_bucket_met['expected_mae']}")

    b_variant = f"bucket41{tag}"
    b_meta_path = out_dir / f"mcls_dep_delay__{b_variant}__catboost__metadata.json"
    b_model_path = out_dir / f"mcls_dep_delay__{b_variant}__catboost.cbm"
    t0 = time.time()
    tr_b = tr_s[task_mask(tr_s, "mcls_dep_delay")]
    y_tr_b = tr_b["y_bucket_dep"].astype(int).to_numpy()
    x_tr_b = pd.concat([build_features(tr_b, feature_set="v1").reset_index(drop=True),
                        tr_b[wx_cols].reset_index(drop=True),
                        tr_b[VOL_FEATURES].reset_index(drop=True)], axis=1)
    if b_meta_path.exists() and b_model_path.exists():
        from catboost import CatBoostClassifier
        bmodel = CatBoostClassifier(); bmodel.load_model(str(b_model_path))
        print("  [bucket41] 已存在，跳过训练")
    else:
        bmodel, _ = train_catboost_bucket(x_tr_b, y_tr_b, full_va_b, y_b, quick=args.quick)
        bmodel.save_model(str(b_model_path))
    proba_new = predict_with(bmodel, {"task": "mcls", "features": list(x_tr_b.columns)},
                             full_va_b)
    nm = m08.multiclass_val_metrics(y_b, proba_new)
    bucket41_met = {"accuracy": nm["accuracy"], "macro_f1": nm["macro_f1"],
                    "calib_pp": calib_pp(y_b, proba_new),
                    "expected_mae": m12.expected_delay_mae(delay_b, proba_new, med),
                    "seconds": round(time.time() - t0, 1)}
    print(f"  [bucket41] acc={bucket41_met['accuracy']} macroF1={bucket41_met['macro_f1']} "
          f"校准={bucket41_met['calib_pp']}pp 期望MAE={bucket41_met['expected_mae']} "
          f"({bucket41_met['seconds']}s)")
    with open(b_meta_path, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "task": "mcls_dep_delay",
                   "task_desc": f"五档区间 + 典型流量特征 {b_variant}",
                   "algo": "catboost", "feature_set": b_variant,
                   "model_file": b_model_path.name, "model_path": str(b_model_path),
                   "features": list(x_tr_b.columns),
                   "categorical_features": CATEGORICAL_FEATURES,
                   "n_classes": config.N_BUCKETS,
                   "bucket_edges": config.DELAY_BUCKET_EDGES,
                   "bucket_labels": config.BUCKET_LABELS,
                   "split": {"train_years": list(config.TRAIN_YEARS),
                             "drop_2020": bool(args.drop_2020),
                             "val_years": list(config.VAL_YEARS)},
                   "val_metrics_2022": bucket41_met,
                   "od_reference_2022": od_bucket_met,
                   "trained_at": datetime.now().isoformat(timespec="seconds"),
                   "library_versions": {"python": platform.python_version(),
                                        "catboost": __import__("catboost").__version__},
                   "mode": "quick" if args.quick else "full"}, f, ensure_ascii=False, indent=2)

    # ================= D：isotonic 校准（2022 H1 拟合 / H2 验证） =================
    print(f"\n=== D. 五档概率 isotonic 校准（H1 拟合 / H2 验证，2023 不触碰） ===")
    months = pd.to_datetime(va_b["FL_DATE"]).dt.month
    h1 = (months <= 6).to_numpy()
    proba_raw = proba_new  # 用本脚本新模型（含典型流量）；对照=未校准
    proba_cal = isotonic_calibrate(y_b[h1], proba_raw[h1], y_b[~h1], proba_raw[~h1])
    before = {"calib_pp": calib_pp(y_b[~h1], proba_raw[~h1]),
              "expected_mae": m12.expected_delay_mae(delay_b[~h1], proba_raw[~h1], med)}
    mb = m08.multiclass_val_metrics(y_b[~h1], proba_raw[~h1])
    before.update({"accuracy": mb["accuracy"], "macro_f1": mb["macro_f1"]})
    after = {"calib_pp": calib_pp(y_b[~h1], proba_cal),
             "expected_mae": m12.expected_delay_mae(delay_b[~h1], proba_cal, med)}
    ma = m08.multiclass_val_metrics(y_b[~h1], proba_cal)
    after.update({"accuracy": ma["accuracy"], "macro_f1": ma["macro_f1"]})
    print(f"  未校准: 校准={before['calib_pp']}pp acc={before['accuracy']} "
          f"macroF1={before['macro_f1']} 期望MAE={before['expected_mae']}")
    print(f"  校准后: 校准={after['calib_pp']}pp acc={after['accuracy']} "
          f"macroF1={after['macro_f1']} 期望MAE={after['expected_mae']}")

    # ================= 汇总落盘 =================
    payload = {
        "run_id": run_id, "mode": "quick" if args.quick else "full",
        "drop_2020": bool(args.drop_2020),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "A_single_stage": {"references": ref, "arms": single_results},
        "C_bucket_volume": {"od_reference": od_bucket_met, "bucket41": bucket41_met},
        "D_isotonic_2022H1H2": {"before": before, "after": after},
        "note": "全部裁决基于2022验证集；2023未读取；未修改生产指针",
    }
    with open(out_dir / "improve_val2022.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    make_charts(out_dir, ref, single_results, od_bucket_met, bucket41_met, before, after)
    write_report(out_dir, run_id, payload)

    print("\n" + "=" * 72)
    print(f"优化实验总结（2022 验证集{'，训练剔除2020' if args.drop_2020 else ''}）")
    print("=" * 72)
    best_s = min(single_results, key=lambda k: single_results[k]["mae"])
    print(f"[A 单段式] 常数0={ref['constant_zero']['mae']} | 两段式od={ref['two_stage_od']['mae']} "
          f"| 最优单段式 {best_s}={single_results[best_s]['mae']} "
          f"({'✅ 赢常数' if single_results[best_s]['mae'] < ref['constant_zero']['mae'] else '❌ 未赢常数'})")
    print(f"[C 五档+流量] acc {bucket41_met['accuracy']} vs 现役 {od_bucket_met['accuracy']} | "
          f"macroF1 {bucket41_met['macro_f1']} vs {od_bucket_met['macro_f1']} | "
          f"校准 {bucket41_met['calib_pp']} vs {od_bucket_met['calib_pp']}pp")
    print(f"[D 校准] 校准偏差 {before['calib_pp']} -> {after['calib_pp']}pp | "
          f"期望MAE {before['expected_mae']} -> {after['expected_mae']}")
    print("-" * 72)
    print(f"报告: {out_dir / 'improve_report.md'}")
    print("注意: 未修改任何生产指针；采纳与否等对照结果")
    print("=" * 72)


# ---------------------------------------------------------------- 图与报告
def make_charts(out_dir, ref, single_results, od_bucket_met, bucket41_met, before, after):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["constant 0", "two-stage od"] + [f"single\n{k}" for k in single_results]
    vals = [ref["constant_zero"]["mae"], ref["two_stage_od"]["mae"]] + \
           [single_results[k]["mae"] for k in single_results]
    colors = ["#9aa5a0", "#c9a227", "#00645a", "#3f6fb5"][:len(vals)]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(range(len(vals)), vals, 0.55, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(vals)), names, fontsize=9)
    ax.set_ylabel("Expected-delay MAE (min), 2022 val")
    ax.set_title("A. Single-stage vs two-stage vs constant-zero")
    fig.tight_layout(); fig.savefig(out_dir / "charts" / "singlestage.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax_, key, title in [(axes[0], "macro_f1", "macro F1"),
                            (axes[1], "calib_pp", "calibration |dev| (pp)")]:
        v = [od_bucket_met[key], bucket41_met[key], before[key], after[key]]
        ax_.bar(range(4), v, 0.55, color=["#9aa5a0", "#00645a", "#c9a227", "#3f6fb5"])
        for i, vv in enumerate(v):
            ax_.text(i, vv, f"{vv:.4f}" if key == "macro_f1" else f"{vv:.2f}",
                     ha="center", fontsize=9)
        ax_.set_xticks(range(4), ["od\n(serving)", "od+volume", "raw\n(H2)", "isotonic\n(H2)"],
                       fontsize=8)
        ax_.set_title(title, fontsize=11)
    fig.suptitle("C/D. Bucket: volume features & isotonic calibration (2022)", fontsize=12)
    fig.tight_layout(); fig.savefig(out_dir / "charts" / "bucket_cd.png", dpi=150)
    plt.close(fig)


def write_report(out_dir, run_id, payload):
    ref = payload["A_single_stage"]["references"]
    sr = payload["A_single_stage"]["arms"]
    od, b41 = payload["C_bucket_volume"]["od_reference"], payload["C_bucket_volume"]["bucket41"]
    db, da = payload["D_isotonic_2022H1H2"]["before"], payload["D_isotonic_2022H1H2"]["after"]
    L = ["# 五档/分钟组件优化实验（A 单段式 / C 典型流量 / D 校准）", "",
         f"- 运行ID：`{run_id}`（{'quick 冒烟' if payload['mode'] == 'quick' else '完整训练'}"
         f"{'，训练行剔除2020（假设检验）' if payload['drop_2020'] else ''}），"
         f"生成于 {payload['evaluated_at']}",
         "- 纪律：训练 2019-2021 / 裁决只用 2022 验证 / 2023 未读取 / 未改生产指针", "",
         "## A. 单段式期望延误（全体航班，MAE=中位回归）", "",
         "| 方法 | MAE | ±5min | ±10min |", "|---|---|---|---|",
         f"| 常数0（必赢基线） | {ref['constant_zero']['mae']} | "
         f"{ref['constant_zero']['hit_5min']:.1%} | {ref['constant_zero']['hit_10min']:.1%} |",
         f"| 两段式 od（现役） | {ref['two_stage_od']['mae']} | "
         f"{ref['two_stage_od']['hit_5min']:.1%} | {ref['two_stage_od']['hit_10min']:.1%} |"]
    for k, v in sr.items():
        L.append(f"| 单段式 {k} | **{v['mae']}** | {v['hit_5min']:.1%} | {v['hit_10min']:.1%} |")
    L += ["", "## C. 五档 + 典型流量特征（机场×小时典型计划架次，训练年口径）", "",
          "| 模型 | acc | macro F1 | 校准pp | 期望MAE |", "|---|---|---|---|---|",
          f"| 现役 od（39特征） | {od['accuracy']} | {od['macro_f1']} | "
          f"{od['calib_pp']} | {od['expected_mae']} |",
          f"| od+流量（41特征） | {b41['accuracy']} | {b41['macro_f1']} | "
          f"{b41['calib_pp']} | {b41['expected_mae']} |", "",
          "## D. isotonic 概率校准（2022H1 拟合 / H2 验证）", "",
          "| 概率 | 校准pp | acc | macro F1 | 期望MAE |", "|---|---|---|---|---|",
          f"| 未校准 | {db['calib_pp']} | {db['accuracy']} | {db['macro_f1']} | "
          f"{db['expected_mae']} |",
          f"| isotonic | {da['calib_pp']} | {da['accuracy']} | {da['macro_f1']} | "
          f"{da['expected_mae']} |", "",
          "## 结论与采纳建议", "",
          "- A：单段式若同时低于常数0与两段式 → 建议采纳为\"期望延误\"来源"
          "（产品其余输出不变；中期展示的\"概率+若延误X分钟\"由 cls+五档继续承担）。",
          "- C：五档若 macro F1/校准改善且准确率不掉 → 建议采纳。",
          "- D：校准若明显收窄且 H2 上其他指标不恶化 → 建议在服务端加校准层。",
          "- B（--drop-2020）：对照本脚本两次运行的报告即可判断，证据不支持则不采纳。", ""]
    (out_dir / "improve_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
