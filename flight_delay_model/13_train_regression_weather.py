# -*- coding: utf-8 -*-
"""
步骤 13：给"若延误，大约几分钟"回归模型加上【双端】天气特征（仅出发口径）。

定位：两段式输出的第二段 E[分钟 | 延误>0]。11 号脚本升级了第一段（二分类），
本脚本升级第二段，特征与 11 号臂4 完全一致（v1 + 起飞机场天气 + 到达机场
@计划到达时刻天气，共 39 个），使两段式全链路天气化。到达侧动机：目的地
雷雨/大雾触发 FAA 地面延误程序(GDP)时航班被扣在起飞机场，分钟数也变长。

评估（2022 验证集，全程不碰 2023）：
  A. 条件延误分钟（仅延误>0 航班）: 双端天气版 vs 生产 v1 版 —— MAE / RMSE / R² /
     ±10min / ±15min 命中率
  B. 两段式期望延误（全体航班, 目标 max(DEP_DELAY,0)），三组合对比：
     ① 生产 v1×v1  ② 上代天气版 wx×wx（仅起飞侧 25 特征存档）
     ③ 本实验 od×od（11 号臂4 二分类 × 本脚本回归）
     组合③需要先跑完 11 号脚本（v1wx_od 臂）——未找到时自动跳过并提示。

用法：
    python 13_train_regression_weather.py                # 完整训练（约15-30分钟）
    python 13_train_regression_weather.py --run-id <目录> # 断点续训
    python 13_train_regression_weather.py --quick         # 抽样冒烟（约2分钟）
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
from features import CATEGORICAL_FEATURES, build_features, make_targets, task_mask, year_series
from importlib import import_module

train_mod = import_module("02_train_models")
m11 = import_module("11_train_weather")     # 天气关联/特征列/工具
eval_mod = import_module("03_evaluate_models")

BASELINE_RUN = "20261005_155440_final_gt0"   # 生产 v1 存档（cls + reg）
PREV_WX_CLS_RUN = "20261009_193645_wx_dep"   # 上代天气二分类（v1wx 25 特征，臂2）
PREV_WX_REG_RUN = "20261009_204247_wx_reg_dep"  # 上代天气回归（v1wx 25 特征）


def find_latest_od_cls_meta():
    """找最近一次训练的 11 号臂4（v1wx_od）二分类 metadata；无则 None。"""
    metas = sorted(config.MODELS_DIR.glob(
        "*/cls_dep_delay__v1wx_od__catboost__metadata.json"),
        key=lambda p: p.stat().st_mtime)
    return json.load(open(metas[-1], encoding="utf-8")) if metas else None


def reg_val_metrics(y_true, p):
    err = p - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": round(float(np.mean(np.abs(err))), 2),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "r2": round(1 - ss_res / ss_tot, 4) if ss_tot else None,
        "hit_10min": round(float(np.mean(np.abs(err) <= 10)), 4),
        "hit_15min": round(float(np.mean(np.abs(err) <= 15)), 4),
        "n": int(len(y_true)),
    }


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


def main():
    ap = argparse.ArgumentParser(description="天气特征版延误分钟回归（仅出发）")
    ap.add_argument("--data", default=str(config.CLEAN_CSV))
    ap.add_argument("--run-name", default="wx_reg_dep")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    for p in (m11.WX_DIR / "tz_map.json", m11.WX_DIR / "weather_hourly.csv.gz"):
        if not p.exists():
            raise SystemExit(f"缺少 {p}，请先运行: python 10_weather_download.py")
    tz_map = json.load(open(m11.WX_DIR / "tz_map.json", encoding="utf-8"))

    run_id = args.run_id or train_mod.make_run_id(args.run_name, quick=args.quick)
    out_dir = config.MODELS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "charts").mkdir(exist_ok=True)
    print(f"存档目录: {out_dir}{'（quick，指标仅供参考）' if args.quick else ''}")
    print("目标: E[出发延误分钟 | 延误>0] 回归 + 天气特征（两段式第二段天气化）")

    df = make_targets(pd.read_csv(args.data, low_memory=False))
    year = year_series(df)
    df = df[year.isin(list(config.TRAIN_YEARS) + list(config.VAL_YEARS))]
    df = m11.load_weather_features(df, tz_map)
    cov = {"origin": round(float(1 - df["wx_missing_o"].mean()), 4),
           "dest": round(float(1 - df["wx_missing_d"].mean()), 4)}
    print(f"天气覆盖率: 起飞机场 {cov['origin']:.1%} / 到达机场 {cov['dest']:.1%}")

    train = df[year_series(df).isin(config.TRAIN_YEARS)]
    val = df[year_series(df).isin(config.VAL_YEARS)].copy()

    task = "reg_dep_delay"
    m_tr, m_va = task_mask(train, task), task_mask(val, task)
    tr, va = train[m_tr], val[m_va]
    if args.quick:
        tr = tr.sample(min(len(tr), 100_000), random_state=config.RANDOM_SEED)
        va = va.sample(min(len(va), 50_000), random_state=config.RANDOM_SEED)
    y_tr = pd.to_numeric(tr["DEP_DELAY"], errors="coerce").to_numpy()
    y_va = pd.to_numeric(va["DEP_DELAY"], errors="coerce").to_numpy()
    print(f"=== [dep|delay>0] 训练 {len(tr):,} / 验证 {len(va):,} "
          f"(中位延误 {np.median(y_va):.0f} 分钟) ===")

    x_v1_tr = build_features(tr, feature_set="v1")
    x_v1_va = build_features(va, feature_set="v1")
    wx_cols = m11.WX_ORIGIN_FEATURE_COLS + m11.WX_DEST_FEATURE_COLS
    wx_tr = tr[wx_cols].reset_index(drop=True)
    wx_va = va[wx_cols].reset_index(drop=True)
    x_tr = pd.concat([x_v1_tr.reset_index(drop=True), wx_tr], axis=1)
    x_va = pd.concat([x_v1_va.reset_index(drop=True), wx_va], axis=1)

    meta_path = out_dir / f"{task}__v1wx_od__catboost__metadata.json"
    model_path = out_dir / f"{task}__v1wx_od__catboost.cbm"
    t0 = time.time()
    if meta_path.exists() and model_path.exists():
        from catboost import CatBoostRegressor, Pool
        model = CatBoostRegressor(); model.load_model(str(model_path))
        cat_idx = [x_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
        best_iter = int(model.get_best_iteration())
        print("  [v1wx_od] 已存在，跳过训练")
    else:
        model, cat_idx = train_catboost_reg(x_tr, y_tr, x_va, y_va, quick=args.quick)
        model.save_model(str(model_path))
        best_iter = int(model.get_best_iteration())
    from catboost import Pool
    p_wx = np.asarray(model.predict(Pool(x_va, cat_features=cat_idx))).ravel()
    met = reg_val_metrics(y_va, p_wx)
    met["best_iteration"] = best_iter
    met["seconds"] = round(time.time() - t0, 1)
    print(f"  [v1wx_od] MAE={met['mae']} RMSE={met['rmse']} ±10min={met['hit_10min']:.1%} "
          f"(best_iter={best_iter}, {met['seconds']}s)")

    # 对照A: 生产 v1 回归（同批验证行）
    base_meta_path = config.MODELS_DIR / BASELINE_RUN / f"{task}__catboost__metadata.json"
    old_met = None
    if base_meta_path.exists():
        base_meta = json.load(open(base_meta_path, encoding="utf-8"))
        old_model = eval_mod.load_model(base_meta)
        x_old = x_v1_va.reset_index(drop=True)
        if base_meta.get("features"):
            x_old = x_old[base_meta["features"]]
        from catboost import Pool
        cat_old = [x_old.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
        p_old = np.asarray(old_model.predict(Pool(x_old, cat_features=cat_old))).ravel()
        old_met = reg_val_metrics(y_va, p_old)

    # 对照B: 两段式期望延误（全体验证行, 目标 max(DEP_DELAY,0)），三组合
    two_stage = {}
    m_all = task_mask(val, "cls_dep_delay")
    va_all = val[m_all]
    x_all_v1 = build_features(va_all, feature_set="v1").reset_index(drop=True)
    wx_all = va_all[m11.WX_ORIGIN_FEATURE_COLS + m11.WX_DEST_FEATURE_COLS].reset_index(drop=True)
    x_all_wx = pd.concat([x_all_v1, wx_all], axis=1)   # v1 + 39 个天气特征
    y_all = np.clip(pd.to_numeric(va_all["DEP_DELAY"], errors="coerce").to_numpy(), 0, None)

    def two_stage_metrics(p_delay, p_min):
        exp = p_delay * p_min
        return {"mae": round(float(np.mean(np.abs(exp - y_all))), 2),
                "hit_5min": round(float(np.mean(np.abs(exp - y_all) <= 5)), 4),
                "hit_10min": round(float(np.mean(np.abs(exp - y_all) <= 10)), 4)}

    def _predict_with_meta(mod, mt, x_container):
        xi = x_container[mt["features"]] if mt.get("features") else x_container
        ci = [xi.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
        if mt["task"].startswith("cls"):
            return mod.predict_proba(Pool(xi, cat_features=ci))[:, 1]
        return np.asarray(mod.predict(Pool(xi, cat_features=ci))).ravel()

    from catboost import Pool
    # 组合①: 生产 cls(v1) × reg(v1)
    old_cls_meta = json.load(open(config.MODELS_DIR / BASELINE_RUN /
                                  "cls_dep_delay__catboost__metadata.json", encoding="utf-8"))
    old_cls = eval_mod.load_model(old_cls_meta)
    p_delay_old = _predict_with_meta(old_cls, old_cls_meta, x_all_v1)
    if base_meta_path.exists():
        p_min_old = _predict_with_meta(old_model, base_meta, x_all_v1)
        two_stage["old_v1"] = two_stage_metrics(p_delay_old, p_min_old)

    # 组合②: 上代天气版（仅起飞侧 25 特征存档，cls × reg 同批行重算，公平对比）
    try:
        prev_cls_meta = json.load(open(config.MODELS_DIR / PREV_WX_CLS_RUN /
                                       "cls_dep_delay__v1wx__catboost__metadata.json",
                                       encoding="utf-8"))
        prev_reg_meta = json.load(open(config.MODELS_DIR / PREV_WX_REG_RUN /
                                       f"{task}__v1wx__catboost__metadata.json",
                                       encoding="utf-8"))
        prev_cls = eval_mod.load_model(prev_cls_meta)
        prev_reg = eval_mod.load_model(prev_reg_meta)
        two_stage["prev_wx"] = two_stage_metrics(
            _predict_with_meta(prev_cls, prev_cls_meta, x_all_wx),
            _predict_with_meta(prev_reg, prev_reg_meta, x_all_wx))
    except FileNotFoundError:
        print("  [两段式] 未找到上代天气版存档，跳过组合②")

    # 组合③: 本实验 od × od —— 11 号臂4 二分类(自动发现最近存档) × 本脚本回归
    od_cls_meta = find_latest_od_cls_meta()
    if od_cls_meta is not None:
        od_cls = eval_mod.load_model(od_cls_meta)
        this_meta = {"task": task, "features": list(x_tr.columns)}
        two_stage["new_od"] = two_stage_metrics(
            _predict_with_meta(od_cls, od_cls_meta, x_all_wx),
            _predict_with_meta(model, this_meta, x_all_wx))
        two_stage["new_od_cls_run"] = od_cls_meta["run_id"]
    else:
        print("  [两段式] 未找到 11 号臂4(v1wx_od) 二分类存档 —— 请先跑 11 号脚本，"
              "再重跑本脚本(--run-id 本目录)可得组合③")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id, "task": task,
            "task_desc": "E[出发延误分钟|延误>0] 回归 + 天气特征（两段式第二段）",
            "algo": "catboost", "feature_set": "v1wx_od", "model_file": model_path.name,
            "model_path": str(model_path), "features": list(x_tr.columns),
            "categorical_features": CATEGORICAL_FEATURES, "categories": None,
            "split": {"train_years": list(config.TRAIN_YEARS),
                      "val_years": list(config.VAL_YEARS),
                      "train_rows": int(len(tr)), "val_rows": int(len(va))},
            "hyperparams": {"loss_function": "MAE", "iterations": 300 if args.quick else 1500,
                            "learning_rate": 0.1, "depth": 8, "l2_leaf_reg": 3.0},
            "random_seed": config.RANDOM_SEED,
            "weather_coverage": cov,
            "val_metrics_2022": met, "baseline_v1_val_metrics": old_met,
            "two_stage_val": two_stage,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "library_versions": {"python": platform.python_version(),
                                 "pandas": pd.__version__, "numpy": np.__version__,
                                 "catboost": __import__("catboost").__version__},
            "mode": "quick" if args.quick else "full",
            "scope": "departure-only (2026-10-08 decision)",
        }, f, ensure_ascii=False, indent=2)
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id,
                   "created_at": datetime.now().isoformat(timespec="seconds"),
                   "type": "wx_reg_dep", "mode": "quick" if args.quick else "full",
                   "models": [str(model_path)]}, f, ensure_ascii=False, indent=2)

    make_outputs(out_dir, run_id, args.quick, cov, met, old_met, two_stage)

    print("\n" + "=" * 70)
    print(f"延误分钟回归+双端天气 对照总结（2022 验证集, 覆盖率 起飞 {cov['origin']:.1%}"
          f" / 到达 {cov['dest']:.1%}）")
    print("=" * 70)
    if old_met:
        print(f"[A 条件分钟] MAE   : 双端天气版 {met['mae']} vs v1版 {old_met['mae']}")
        print(f"[A 条件分钟] ±10min: 双端天气版 {met['hit_10min']:.1%} vs v1版 {old_met['hit_10min']:.1%}")
        print(f"[A 条件分钟] R²    : 双端天气版 {met['r2']} vs v1版 {old_met['r2']}")
    for combo, label in [("old_v1", "v1×v1（生产）"), ("prev_wx", "上代天气×上代天气"),
                         ("new_od", "双端天气×双端天气")]:
        if combo in two_stage:
            print(f"[B 两段式 {label:12s}] MAE={two_stage[combo]['mae']} "
                  f"±5min={two_stage[combo]['hit_5min']:.1%} "
                  f"±10min={two_stage[combo]['hit_10min']:.1%}")
    if "prev_wx" in two_stage and "new_od" in two_stage:
        print(f"[B 到达侧增益] 两段式MAE: {two_stage['new_od']['mae'] - two_stage['prev_wx']['mae']:+.2f} 分钟"
              f"（双端 vs 上代天气版）")
    print("-" * 70)
    print(f"报告: {out_dir / 'wx_reg_report.md'}")
    print("注意: 未修改 latest_run.json / wx_serving.json（生产不变）")
    print("=" * 70)


def make_outputs(out_dir, run_id, quick, cov, met, old_met, two_stage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    if old_met:
        pos = np.arange(3); w = 0.34
        vals_old = [old_met["mae"], old_met["hit_10min"] * 100, old_met["hit_15min"] * 100]
        vals_wx = [met["mae"], met["hit_10min"] * 100, met["hit_15min"] * 100]
        axes[0].bar(pos - w / 2, vals_old, w, color="#9aa5a0", label="v1")
        axes[0].bar(pos + w / 2, vals_wx, w, color="#00645a", label="v1 + weather")
        axes[0].set_xticks(pos, ["MAE (min)", "±10 min %", "±15 min %"])
        for i in range(3):
            axes[0].text(i - w / 2, vals_old[i], f"{vals_old[i]:.1f}", ha="center", fontsize=9)
            axes[0].text(i + w / 2, vals_wx[i], f"{vals_wx[i]:.1f}", ha="center", fontsize=9)
        axes[0].legend(fontsize=9); axes[0].set_title("A. Minutes-if-delayed (delayed only)")
    if two_stage:
        combos = [(k, lbl) for k, lbl in
                  [("old_v1", "v1×v1"), ("prev_wx", "wx×wx (origin)"),
                   ("new_od", "od×od (origin+dest)")] if k in two_stage]
        pos = np.arange(3); w = 0.8 / max(len(combos), 1)
        palette = {"old_v1": "#9aa5a0", "prev_wx": "#c9a227", "new_od": "#00645a"}
        for j, (k, lbl) in enumerate(combos):
            t = two_stage[k]
            vals = [t["mae"], t["hit_5min"] * 100, t["hit_10min"] * 100]
            xj = pos - 0.4 + w / 2 + j * w
            axes[1].bar(xj, vals, w, color=palette.get(k, "#777"), label=lbl)
            for i in range(3):
                axes[1].text(xj[i], vals[i], f"{vals[i]:.1f}", ha="center", fontsize=8)
        axes[1].set_xticks(pos, ["MAE (min)", "±5 min %", "±10 min %"])
        axes[1].legend(fontsize=8); axes[1].set_title("B. Expected delay (all flights)")
    fig.tight_layout(); fig.savefig(out_dir / "charts" / "regression_comparison.png", dpi=150)
    plt.close(fig)

    L = ["# 延误分钟回归+双端天气 实验报告（仅出发口径）", "",
         f"- 运行ID：`{run_id}`（{'**quick，仅供参考**' if quick else '完整训练'}），"
         f"生成于 {datetime.now().isoformat(timespec='seconds')}",
         f"- 特征与 11 号臂4 一致（v1+双端天气39特征）；训练 2019-2021 / 验证 2022，"
         f"2023 未读取；覆盖率 起飞 {cov['origin']:.1%} / 到达 {cov['dest']:.1%}", "",
         "## A. 条件延误分钟（仅延误>0 航班）", "",
         "| 模型 | MAE | RMSE | R² | ±10min | ±15min |", "|---|---|---|---|---|---|"]
    for name, m in [("v1+双端天气", met)] + ([("v1 生产", old_met)] if old_met else []):
        L.append(f"| {name} | {m['mae']} | {m['rmse']} | {m['r2']} | "
                 f"{m['hit_10min']:.1%} | {m['hit_15min']:.1%} |")
    if two_stage:
        L += ["", "## B. 两段式期望延误（全体航班，目标 max(DEP_DELAY,0)）", "",
              "| 两段式组合 | MAE | ±5min | ±10min |", "|---|---|---|---|"]
        for k, lbl in [("old_v1", "v1 × v1（生产）"),
                       ("prev_wx", f"上代天气 × 上代天气（{PREV_WX_CLS_RUN} 臂2 × {PREV_WX_REG_RUN}）"),
                       ("new_od", f"双端天气 × 双端天气（{two_stage.get('new_od_cls_run', '')} 臂4 × 本模型）")]:
            if k in two_stage:
                t = two_stage[k]
                L.append(f"| {lbl} | {t['mae']} | {t['hit_5min']:.1%} | {t['hit_10min']:.1%} |")
        if "prev_wx" in two_stage and "new_od" in two_stage:
            L += ["", f"到达侧天气增益（双端 vs 上代天气版）两段式 MAE："
                      f"**{two_stage['new_od']['mae'] - two_stage['prev_wx']['mae']:+.2f} 分钟**"]
    L += ["", "## 说明", "",
          "- 训练用实测 METAR；部署由 Open-Meteo 预报提供同构天气输入（server/weather_serving.py"
          "按模型元数据自动取起飞侧+到达侧）。",
          "- 本脚本不修改任何生产指针；切换由 wx_serving.json 配置决定。", ""]
    (out_dir / "wx_reg_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
