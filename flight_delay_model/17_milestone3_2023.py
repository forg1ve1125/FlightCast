# -*- coding: utf-8 -*-
"""
步骤 17：最终形态 v2 收官（第三个冻结里程碑）。

做两件事（全部无训练）：
  1. 制作 isotonic 校准器：用【完整 2022 验证集】对五档模型逐类拟合，
     序列化为 models/<bucket存档>/isotonic_full2022.json（服务端加载使用）。
     其泛化能力由 16 号 H1/H2 实验（校准 2.68→1.85pp）与本脚本的 2023 评估背书。
  2. 2023 里程碑评估（第三次冻结里程碑，如实记录）：
     - 单段式(single39) 期望延误：MAE/±5/±10，对照常数0基线与旧两段式(od)
     - 五档：od 原始 vs od+isotonic(全2022拟合) 的四层指标对照

纪律：无训练；2022 仅用于拟合校准器（校准器是评估产物，不参与任何模型选择）；
2023 为第三次冻结里程碑评估（前两次：v1 基线时代 03 号、终评 14 号），如实入档。

用法：
    python 17_milestone3_2023.py
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

import config
from features import build_features, make_targets, task_mask, year_series
from importlib import import_module

train_mod = import_module("02_train_models")
m11 = import_module("11_train_weather")
m16 = import_module("16_bucket_imbalance")      # metrics_bundle（注意：其 load_one/predict_with
                                                # 是五档分类器专用，本脚本用下面的任务感知版本）


def load_one(run_id, fname):
    """按 metadata 的 task 选 Classifier/Regressor（m16 的同名函数是分类器专用）。"""
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
    ci = [x.columns.get_loc(c) for c in meta["categorical_features"]]
    if meta["task"].startswith("mcls"):
        return np.asarray(model.predict_proba(Pool(x, cat_features=ci)))
    if meta["task"].startswith("cls"):
        return np.asarray(model.predict_proba(Pool(x, cat_features=ci)))[:, 1]
    return np.asarray(model.predict(Pool(x, cat_features=ci))).ravel()

BUCKET = ("20261013_122152_buckets_wx_dep", "mcls_dep_delay__v1wx_od__catboost.cbm")
SINGLE = ("20261017_152303_improve", "reg_dep_single__single39.cbm")
COND = ("20261013_130457_wx_reg_dep", "reg_dep_delay__v1wx_od__catboost.cbm")
CLS = ("20261009_193645_wx_dep", "cls_dep_delay__v1wx_od__catboost.cbm")
MED = [s["median_minutes"] for s in json.load(
    open(config.MODELS_DIR / BUCKET[0] / "bucket_stats.json", encoding="utf-8"))["dep"]]


def expected_metrics(y, p):
    return {"mae": round(float(np.mean(np.abs(p - y))), 2),
            "hit_5min": round(float(np.mean(np.abs(p - y) <= 5)), 4),
            "hit_10min": round(float(np.mean(np.abs(p - y) <= 10)), 4)}


def main():
    tz_map = json.load(open(m11.WX_DIR / "tz_map.json", encoding="utf-8"))
    run_id = train_mod.make_run_id("milestone3")
    out_dir = config.MODELS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"存档目录: {out_dir}")

    # ---- 数据：2022 + 2023 一次读入（2023 为第三次冻结里程碑评估） ----
    df = make_targets(pd.read_csv(config.CLEAN_CSV, low_memory=False))
    df = df[year_series(df).isin(list(config.VAL_YEARS) + list(config.TEST_YEARS))]
    df = m11.load_weather_features(df, tz_map)
    wx_cols = m11.WX_ORIGIN_FEATURE_COLS + m11.WX_DEST_FEATURE_COLS

    def container(d):
        return pd.concat([build_features(d, feature_set="v1").reset_index(drop=True),
                          d[wx_cols].reset_index(drop=True)], axis=1)

    yr = year_series(df)
    val, test = df[yr.isin(config.VAL_YEARS)], df[yr.isin(config.TEST_YEARS)]
    bm, bmeta = load_one(*BUCKET)

    # ---- 1) isotonic 校准器：完整 2022 拟合并序列化 ----
    from sklearn.isotonic import IsotonicRegression
    va_b = val[task_mask(val, "mcls_dep_delay")]
    y22 = va_b["y_bucket_dep"].astype(int).to_numpy()
    proba22 = predict_with(bm, bmeta, container(va_b))
    fits = {}
    for k in range(config.N_BUCKETS):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        ir.fit(proba22[:, k], (y22 == k).astype(float))
        fits[str(k)] = {"x": [round(float(v), 6) for v in ir.X_thresholds_],
                        "y": [round(float(v), 6) for v in ir.y_thresholds_]}
    iso_path = config.MODELS_DIR / BUCKET[0] / "isotonic_full2022.json"
    with open(iso_path, "w", encoding="utf-8") as f:
        json.dump({"fit_on": "full 2022 validation (mcls rows)",
                   "model": f"{BUCKET[0]}/{BUCKET[1]}",
                   "created_at": datetime.now().isoformat(timespec="seconds"),
                   "classes": config.N_BUCKETS, "fit": fits}, f, ensure_ascii=False)
    print(f"isotonic 校准器已保存: {iso_path}")

    def apply_iso(proba):
        out = np.column_stack([
            np.interp(proba[:, k], fits[str(k)]["x"], fits[str(k)]["y"])
            for k in range(config.N_BUCKETS)])
        out = np.clip(out, 1e-6, None)
        return out / out.sum(axis=1, keepdims=True)

    # ---- 2) 2023：单段式 vs 基线 vs 旧两段式 ----
    te = test[task_mask(test, "cls_dep_delay")]
    x_te = container(te)
    y23 = np.clip(pd.to_numeric(te["DEP_DELAY"], errors="coerce").to_numpy(), 0, None)
    res = {"expected_delay_2023": {
        "constant_zero": expected_metrics(y23, np.zeros_like(y23))}}
    sm, smeta = load_one(*SINGLE)
    p_single = predict_with(sm, smeta, x_te)
    res["expected_delay_2023"]["single39"] = expected_metrics(y23, p_single)
    cm, cmeta = load_one(*CLS)
    dm, dmeta = load_one(*COND)
    p_delay = predict_with(cm, cmeta, x_te)
    p_min = predict_with(dm, dmeta, x_te)
    res["expected_delay_2023"]["two_stage_od_old"] = expected_metrics(y23, p_delay * p_min)
    r = res["expected_delay_2023"]
    print(f"\n[期望延误 2023] 常数0={r['constant_zero']['mae']} | "
          f"旧两段式={r['two_stage_od_old']['mae']} | 单段式={r['single39']['mae']} "
          f"(±5={r['single39']['hit_5min']:.1%})")

    # ---- 3) 2023：五档 原始 vs isotonic ----
    te_b = test[task_mask(test, "mcls_dep_delay")]
    y23b = te_b["y_bucket_dep"].astype(int).to_numpy()
    delay_b = pd.to_numeric(te_b["DEP_DELAY"], errors="coerce").to_numpy()
    proba_raw = predict_with(bm, bmeta, container(te_b))
    proba_cal = apply_iso(proba_raw)
    res["bucket_2023"] = {"raw": m16.metrics_bundle(y23b, proba_raw, delay_b, MED),
                          "isotonic_full2022": m16.metrics_bundle(y23b, proba_cal, delay_b, MED)}
    for name, met in [("原始", res["bucket_2023"]["raw"]),
                      ("isotonic", res["bucket_2023"]["isotonic_full2022"])]:
        print(f"[五档2023 {name:8s}] LogLoss={met['logloss']} Brier={met['brier']} "
              f"RPS={met['rps']} 校准={met['calib_pp']}pp QWK={met['qwk']} "
              f"macroF1={met['macro_f1']} acc={met['accuracy']}")

    payload = {"run_id": run_id,
               "evaluated_at": datetime.now().isoformat(timespec="seconds"),
               "discipline": ("第三个冻结里程碑（无训练）：2022 仅拟合校准器（不参与模型选择），"
                              "2023 评估 single39 与 isotonic 校准层；前两次里程碑见 "
                              "03(基线时代)/14(终评)。"),
               **res}
    with open(out_dir / "milestone3_2023.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_dir / 'milestone3_2023.json'}")


if __name__ == "__main__":
    main()
