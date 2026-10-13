# -*- coding: utf-8 -*-
"""
步骤 14：2023 测试集【一次性收官评估】——项目至今从未读取 2023，本脚本是唯一入口。

纪律（同时写进报告）：
  - 所有参与评估的模型在评估前已冻结：训练用 2019-2021，早停/选模用 2022；
    本脚本只做预测，不做任何训练/调参/筛选。
  - 2023 全年数据在本脚本中被读取【一次】，结果即最终测试成绩，之后不再迭代。
    同一存档目录已有正式结果时拒绝重跑（如确需重跑加 --force）。
  - 本脚本不修改任何生产指针（latest_run.json / bucket_latest.json /
    wx_serving.json）；换装决策在看完本报告后人工进行。

评估对象（三档消融 × 三组件——项目最终形态候选）：
  v1 : 仅 11 个计划特征（生产基线存档 20261005_155440_final_gt0）
  wx : v1 + 起飞机场天气 25 特征（11号臂2 / 12号旧档 / 13号旧档）
  od : v1 + 起飞+到达双端天气 39 特征（11号臂4 / 12号新档 / 13号新档）
组件与指标：
  二分类(cls)  : AUC / logloss / acc@0.5 / P / R / F1（并列 2022 验证值看稳定性）
  区间5档(mcls) : 准确率 / macro F1 / 逐档校准偏差 / 期望延误MAE（中位分钟=训练集口径）
  回归(reg)    : 条件分钟 MAE / RMSE / R² / ±10min / ±15min（仅延误>0）
  两段式       : P(延误)×E(分钟) 对 max(DEP_DELAY,0) 的 MAE / ±5min / ±10min（全体航班）

输出：控制台总结 + charts/ 图 + test2023_report.md + test2023_results.json

用法：
    python 14_test_2023.py            # 正式评估（约3-6分钟，无训练）
    python 14_test_2023.py --quick    # 抽样冒烟（指标仅供参考，不落正式标记）
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
m08 = import_module("08_train_bucket_models")     # multiclass_val_metrics
m09 = import_module("09_train_cumulative")        # calib_dev_pp
m11 = import_module("11_train_weather")           # 双端天气关联
m12 = import_module("12_train_bucket_weather")    # expected_delay_mae
m13 = import_module("13_train_regression_weather")  # reg_val_metrics

# ---------------------------------------------------------------- 模型地图
# 评估前冻结的存档（训练 2019-2021 / 早停 2022，2023 从未参与任何环节）
RUNS = {
    "v1": {
        "cls": ("20261005_155440_final_gt0", "cls_dep_delay__catboost.cbm"),
        "reg": ("20261005_155440_final_gt0", "reg_dep_delay__catboost.cbm"),
        "bucket": None,   # 由 bucket_latest.json 指向的 08 存档
    },
    "wx": {
        "cls": ("20261009_193645_wx_dep", "cls_dep_delay__v1wx__catboost.cbm"),
        "reg": ("20261009_204247_wx_reg_dep", "reg_dep_delay__v1wx__catboost.cbm"),
        "bucket": ("20261009_202254_buckets_wx_dep", "mcls_dep_delay__v1wx__catboost.cbm"),
    },
    "od": {
        "cls": ("20261009_193645_wx_dep", "cls_dep_delay__v1wx_od__catboost.cbm"),
        "reg": ("20261013_130457_wx_reg_dep", "reg_dep_delay__v1wx_od__catboost.cbm"),
        "bucket": ("20261013_122152_buckets_wx_dep", "mcls_dep_delay__v1wx_od__catboost.cbm"),
    },
}
TIER_LABELS = {"v1": "v1 (schedule only)", "wx": "wx (+origin weather)",
               "od": "od (+origin+dest weather)"}
TIER_COLORS = {"v1": "#9aa5a0", "wx": "#c9a227", "od": "#00645a"}
# 2022 验证集两段式 MAE 的读取位置（13 号存档 metadata 的 two_stage_val）
TWO_STAGE_VAL_REF = {
    "v1": ("20261009_204247_wx_reg_dep", "reg_dep_delay__v1wx__catboost__metadata.json", "old_v1"),
    "wx": ("20261009_204247_wx_reg_dep", "reg_dep_delay__v1wx__catboost__metadata.json", "new_wx"),
    "od": ("20261013_130457_wx_reg_dep", "reg_dep_delay__v1wx_od__catboost__metadata.json", "new_od"),
}


def _bucket_run_of_v1():
    p = config.MODELS_DIR / "bucket_latest.json"
    return json.load(open(p, encoding="utf-8"))["run_id"] if p.exists() else None


def resolve_paths():
    """RUNS 里 bucket=None 的 v1 档，用 bucket_latest.json 补全为 08 存档。"""
    paths = {}
    for tier, comps in RUNS.items():
        paths[tier] = dict(comps)
        if comps["bucket"] is None:
            rid = _bucket_run_of_v1()
            if rid:
                paths[tier]["bucket"] = (rid, "mcls_dep_delay__catboost.cbm")
    return paths


def load_one(run_id, fname):
    path = config.MODELS_DIR / run_id / fname
    if not path.exists():
        raise SystemExit(f"缺少模型: {path}")
    meta = json.load(open(path.with_name(path.stem + "__metadata.json"),
                          encoding="utf-8"))
    if meta["task"].startswith(("cls", "mcls")):
        from catboost import CatBoostClassifier as M
    else:
        from catboost import CatBoostRegressor as M
    model = M()
    model.load_model(str(path))
    return model, meta


def predict_with(model, meta, x_container):
    """按 metadata 特征清单从容器取列预测；cls->P(延误), mcls->5档概率, reg->分钟。"""
    from catboost import Pool
    x = x_container[meta["features"]]
    ci = [x.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    if meta["task"].startswith("mcls"):
        return np.asarray(model.predict_proba(Pool(x, cat_features=ci)))
    if meta["task"].startswith("cls"):
        return np.asarray(model.predict_proba(Pool(x, cat_features=ci)))[:, 1]
    return np.asarray(model.predict(Pool(x, cat_features=ci))).ravel()


def main():
    ap = argparse.ArgumentParser(description="2023 测试集一次性收官评估（无训练）")
    ap.add_argument("--data", default=str(config.CLEAN_CSV))
    ap.add_argument("--run-name", default="test2023")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true", help="已有正式结果时强制重跑")
    args = ap.parse_args()

    tz_path = m11.WX_DIR / "tz_map.json"
    wx_path = m11.WX_DIR / "weather_hourly.csv.gz"
    for p in (tz_path, wx_path):
        if not p.exists():
            raise SystemExit(f"缺少 {p}，请先运行: python 10_weather_download.py")
    tz_map = json.load(open(tz_path, encoding="utf-8"))

    run_id = args.run_id or train_mod.make_run_id(args.run_name, quick=args.quick)
    out_dir = config.MODELS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "charts").mkdir(exist_ok=True)
    marker = out_dir / "test2023_results.json"
    if marker.exists() and not args.quick and not args.force:
        raise SystemExit(f"{marker} 已存在：2023 测试集只做一次正式评估。"
                         f"确需重跑请加 --force（并在报告中说明）。")
    print(f"存档目录: {out_dir}{'（quick 冒烟，不作为正式成绩）' if args.quick else ''}")

    # ---- 数据：只读 2023（第一次也是最后一次） ----
    df = make_targets(pd.read_csv(args.data, low_memory=False))
    year = year_series(df)
    df = df[year.isin(config.TEST_YEARS)]
    if df.empty:
        raise SystemExit("2023 无数据：请检查 TEST_YEARS / 数据文件")
    df = m11.load_weather_features(df, tz_map)
    cov = {"origin": round(float(1 - df["wx_missing_o"].mean()), 4),
           "dest": round(float(1 - df["wx_missing_d"].mean()), 4)}
    delay_all = pd.to_numeric(df["DEP_DELAY"], errors="coerce")
    ctx = {"rows_total": int(len(df)),
           "weather_coverage": cov,
           "dep_delay_share_gt0": round(float((delay_all > 0).mean()), 4),
           "dep_delay_median": round(float(delay_all.median()), 1),
           "dep_delay_mean": round(float(delay_all.mean()), 1)}
    print(f"2023 行数 {ctx['rows_total']:,}；延误>0 占比 {ctx['dep_delay_share_gt0']:.1%}；"
          f"中位 {ctx['dep_delay_median']} 分钟；天气覆盖率 起飞 {cov['origin']:.1%}"
          f" / 到达 {cov['dest']:.1%}")

    paths = resolve_paths()
    tiers = ["v1", "wx", "od"]
    containers = {
        "v1": build_features(df, feature_set="v1").reset_index(drop=True),
    }
    wx_cols = m11.WX_ORIGIN_FEATURE_COLS + m11.WX_DEST_FEATURE_COLS
    containers["wx"] = pd.concat(
        [containers["v1"], df[m11.WX_ORIGIN_FEATURE_COLS].reset_index(drop=True)], axis=1)
    containers["od"] = pd.concat(
        [containers["v1"], df[wx_cols].reset_index(drop=True)], axis=1)

    results = {t: {} for t in tiers}

    # ---- 二分类（同一批 cls 口径行） ----
    m_cls = task_mask(df, "cls_dep_delay")
    d_cls = df[m_cls]
    y_cls = d_cls["y_cls_dep"].astype(int).to_numpy()
    x_cls = {t: containers[t][m_cls.values] for t in tiers}
    print(f"\n=== 二分类: {len(y_cls):,} 行 ===")
    for t in tiers:
        model, meta = load_one(*paths[t]["cls"])
        p = predict_with(model, meta, x_cls[t])
        met = m11.cls_val_metrics(y_cls, p)
        met["val_auc_2022"] = meta.get("val_metrics_2022", {}).get("auc")
        results[t]["cls"] = met
        print(f"  [{t}] AUC={met['auc']:.4f} (2022验证 {met['val_auc_2022']:.4f}, "
              f"落差 {met['auc'] - met['val_auc_2022']:+.4f}) logloss={met['logloss']}")

    # ---- 区间5档（mcls 口径行；中位分钟用各存档的训练集口径） ----
    m_bucket = task_mask(df, "mcls_dep_delay")
    d_bucket = df[m_bucket]
    y_bucket = d_bucket["y_bucket_dep"].astype(int).to_numpy()
    delay_bucket = pd.to_numeric(d_bucket["DEP_DELAY"], errors="coerce").to_numpy()
    x_bucket = {t: containers[t][m_bucket.values] for t in tiers}
    print(f"\n=== 区间5档: {len(y_bucket):,} 行 ===")
    for t in tiers:
        run_id_b, fname_b = paths[t]["bucket"]
        if run_id_b is None or not (config.MODELS_DIR / run_id_b / fname_b).exists():
            print(f"  [{t}] 未找到区间模型存档，跳过")
            continue
        model, meta = load_one(run_id_b, fname_b)
        proba = predict_with(model, meta, x_bucket[t])
        met = m08.multiclass_val_metrics(y_bucket, proba)
        met["calib_mean_abs_dev_pp"] = m09.calib_dev_pp(met)
        stats_path = config.MODELS_DIR / run_id_b / "bucket_stats.json"
        med = ([s["median_minutes"] for s in
                json.load(open(stats_path, encoding="utf-8"))["dep"]]
               if stats_path.exists() else [0, 5, 22, 42, 104])
        met["expected_delay_mae_vs_max_delay0"] = m12.expected_delay_mae(
            delay_bucket, proba, med)
        vm = meta.get("val_metrics_2022", {})
        met["val_accuracy_2022"] = vm.get("accuracy")
        results[t]["bucket"] = met
        print(f"  [{t}] acc={met['accuracy']:.4f} (2022 {vm.get('accuracy')}) "
              f"macroF1={met['macro_f1']:.4f} 校准={met['calib_mean_abs_dev_pp']:.2f}pp "
              f"期望延误MAE={met['expected_delay_mae_vs_max_delay0']}")

    # ---- 回归（延误>0 行） + 两段式（全体 cls 口径行） ----
    m_reg = task_mask(df, "reg_dep_delay")
    d_reg = df[m_reg]
    y_reg = pd.to_numeric(d_reg["DEP_DELAY"], errors="coerce").to_numpy()
    x_reg = {t: containers[t][m_reg.values] for t in tiers}
    y_all = np.clip(pd.to_numeric(d_cls["DEP_DELAY"], errors="coerce").to_numpy(), 0, None)
    print(f"\n=== 回归(延误>0): {len(y_reg):,} 行 | 两段式(全体): {len(y_all):,} 行 ===")

    def two_stage_metrics(p_delay, p_min):
        exp = p_delay * p_min
        return {"mae": round(float(np.mean(np.abs(exp - y_all))), 2),
                "hit_5min": round(float(np.mean(np.abs(exp - y_all) <= 5)), 4),
                "hit_10min": round(float(np.mean(np.abs(exp - y_all) <= 10)), 4)}

    for t in tiers:
        cls_model, cls_meta = load_one(*paths[t]["cls"])
        model, meta = load_one(*paths[t]["reg"])
        # A. 条件分钟（仅延误>0 行）
        p_min_cond = predict_with(model, meta, x_reg[t])
        met = m13.reg_val_metrics(y_reg, p_min_cond)
        results[t]["reg"] = met
        # B. 两段式：分钟预测必须覆盖全体 cls 口径行（部署时不知道谁会延误）
        p_min_all = predict_with(model, meta, x_cls[t])
        two = two_stage_metrics(predict_with(cls_model, cls_meta, x_cls[t]), p_min_all)
        results[t]["two_stage"] = two
        print(f"  [{t}] 条件分钟 MAE={met['mae']} ±10min={met['hit_10min']:.1%} | "
              f"两段式 MAE={two['mae']} ±5min={two['hit_5min']:.1%} ±10min={two['hit_10min']:.1%}")

    # 2022 两段式参照值（图注用）
    val_ref = {}
    for t, (rid, mfile, key) in TWO_STAGE_VAL_REF.items():
        try:
            ts = json.load(open(config.MODELS_DIR / rid / mfile,
                                encoding="utf-8")).get("two_stage_val", {})
            if key in ts:
                val_ref[t] = ts[key]["mae"]
        except OSError:
            pass

    payload = {
        "run_id": run_id,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "discipline": ("2023 测试集一次性收官评估：所有模型评估前冻结"
                       "（训练2019-2021/早停2022），本脚本只预测不训练；"
                       "结果即最终成绩，不再迭代。"),
        "mode": "quick" if args.quick else "final",
        "context_2023": ctx,
        "model_paths": {t: {k: (f"{r}/{f}" if r else None) for k, (r, f) in c.items()}
                        for t, c in paths.items()},
        "tiers": results,
        "two_stage_val_ref_2022": val_ref,
        "library_versions": {"python": platform.python_version(),
                             "pandas": pd.__version__, "numpy": np.__version__},
    }
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    make_charts(out_dir, results, val_ref)
    write_report(out_dir, run_id, payload, results, ctx, val_ref)

    print("\n" + "=" * 72)
    print("2023 测试集·三档收官总结（一次性，最终成绩）")
    print("=" * 72)
    for t in tiers:
        r = results[t]
        print(f"[{TIER_LABELS[t]}]")
        print(f"  AUC {r['cls']['auc']:.4f} (2022 {r['cls']['val_auc_2022']:.4f}) | "
              f"区间acc {r.get('bucket', {}).get('accuracy', '—')} | "
              f"两段式MAE {r['two_stage']['mae']}")
    if all("bucket" in results[t] for t in tiers):
        print(f"测试集增益 od−v1: AUC {results['od']['cls']['auc'] - results['v1']['cls']['auc']:+.4f} | "
              f"两段式MAE {results['od']['two_stage']['mae'] - results['v1']['two_stage']['mae']:+.2f} 分钟")
        print(f"测试集增益 od−wx: AUC {results['od']['cls']['auc'] - results['wx']['cls']['auc']:+.4f} | "
              f"两段式MAE {results['od']['two_stage']['mae'] - results['wx']['two_stage']['mae']:+.2f} 分钟")
    print("-" * 72)
    print(f"报告: {out_dir / 'test2023_report.md'}")
    print(f"图像: {out_dir / 'charts'}/")
    print("注意: 未修改任何生产指针（latest_run/bucket_latest/wx_serving）")
    print("=" * 72)


# ---------------------------------------------------------------- 图与报告
def make_charts(out_dir, results, val_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tiers = [t for t in ("v1", "wx", "od") if t in results]

    # 1) 二分类 AUC：2023 测试柱 + 2022 验证横线
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    vals = [results[t]["cls"]["auc"] for t in tiers]
    ax.bar(range(len(tiers)), vals, 0.55, color=[TIER_COLORS[t] for t in tiers])
    for i, (t, v) in enumerate(zip(tiers, vals)):
        ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=10)
        ref = results[t]["cls"].get("val_auc_2022")
        if ref:
            ax.hlines(ref, i - 0.3, i + 0.3, color="#b23a3a", ls="--", lw=1.5)
            ax.text(i + 0.32, ref, f"2022: {ref:.4f}", fontsize=8, color="#b23a3a",
                    va="center")
    ax.set_xticks(range(len(tiers)), [TIER_LABELS[t] for t in tiers], fontsize=9)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.62, max(vals) + 0.02)
    ax.set_title("Departure-delay AUC on the untouched 2023 test set")
    fig.tight_layout(); fig.savefig(out_dir / "charts" / "auc_test2023.png", dpi=150)
    plt.close(fig)

    # 2) 两段式期望延误 MAE
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    vals = [results[t]["two_stage"]["mae"] for t in tiers]
    ax.bar(range(len(tiers)), vals, 0.55, color=[TIER_COLORS[t] for t in tiers])
    for i, (t, v) in enumerate(zip(tiers, vals)):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)
        if t in val_ref:
            ax.hlines(val_ref[t], i - 0.3, i + 0.3, color="#b23a3a", ls="--", lw=1.5)
            ax.text(i + 0.32, val_ref[t], f"2022: {val_ref[t]:.2f}", fontsize=8,
                    color="#b23a3a", va="center")
    ax.set_xticks(range(len(tiers)), [TIER_LABELS[t] for t in tiers], fontsize=9)
    ax.set_ylabel("Expected-delay MAE (min)")
    ax.set_title("Two-stage expected delay, all flights (2023 test)")
    fig.tight_layout(); fig.savefig(out_dir / "charts" / "twostage_test2023.png", dpi=150)
    plt.close(fig)

    # 3) 区间5档：准确率 + 期望延误 MAE
    if all("bucket" in results.get(t, {}) for t in tiers):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax_, key, title in [(axes[0], "accuracy", "Bucket accuracy"),
                                (axes[1], "expected_delay_mae_vs_max_delay0",
                                 "Expected-delay MAE (min)")]:
            vals = [results[t]["bucket"][key] for t in tiers]
            ax_.bar(range(len(tiers)), vals, 0.55,
                    color=[TIER_COLORS[t] for t in tiers])
            for i, v in enumerate(vals):
                ax_.text(i, v, f"{v}", ha="center", fontsize=9)
            ax_.set_xticks(range(len(tiers)), [t for t in tiers], fontsize=9)
            ax_.set_title(title, fontsize=11)
        fig.suptitle("5-bucket multiclass on the 2023 test set", fontsize=12)
        fig.tight_layout(); fig.savefig(out_dir / "charts" / "bucket_test2023.png", dpi=150)
        plt.close(fig)


def write_report(out_dir, run_id, payload, results, ctx, val_ref):
    tiers = [t for t in ("v1", "wx", "od") if t in results]
    L = ["# 2023 测试集·一次性收官评估报告", "",
         f"- 运行ID：`{run_id}`，评估时间 {payload['evaluated_at']}"
         f"（{'**quick 冒烟，非正式成绩**' if payload['mode'] == 'quick' else '**正式成绩**'}）",
         f"- 纪律：{payload['discipline']}",
         f"- 2023 数据：{ctx['rows_total']:,} 行，延误>0 占比 "
         f"{ctx['dep_delay_share_gt0']:.1%}，中位 {ctx['dep_delay_median']} 分钟；"
         f"天气覆盖率 起飞 {ctx['weather_coverage']['origin']:.1%} / "
         f"到达 {ctx['weather_coverage']['dest']:.1%}", "",
         "## 二分类（延误>0）", "",
         "| 档 | AUC(2023) | AUC(2022验证) | 落差 | logloss | acc@0.5 | F1 |",
         "|---|---|---|---|---|---|---|"]
    for t in tiers:
        m = results[t]["cls"]
        L.append(f"| {TIER_LABELS[t]} | **{m['auc']:.4f}** | {m['val_auc_2022']} | "
                 f"{m['auc'] - m['val_auc_2022']:+.4f} | {m['logloss']} | "
                 f"{m['accuracy']} | {m['f1']} |")
    L += ["", "## 区间5档（多分类）", "",
          "| 档 | 准确率 | macro F1 | 校准偏差 | 期望延误MAE |", "|---|---|---|---|---|"]
    for t in tiers:
        if "bucket" not in results[t]:
            continue
        m = results[t]["bucket"]
        L.append(f"| {TIER_LABELS[t]} | **{m['accuracy']:.4f}** | {m['macro_f1']:.4f} | "
                 f"{m['calib_mean_abs_dev_pp']:.2f}pp | "
                 f"{m['expected_delay_mae_vs_max_delay0']} |")
    L += ["", "## 回归（条件分钟，仅延误>0）", "",
          "| 档 | MAE | RMSE | ±10min | ±15min |", "|---|---|---|---|---|"]
    for t in tiers:
        m = results[t]["reg"]
        L.append(f"| {TIER_LABELS[t]} | **{m['mae']}** | {m['rmse']} | "
                 f"{m['hit_10min']:.1%} | {m['hit_15min']:.1%} |")
    L += ["", "## 两段式期望延误（全体航班，目标 max(DEP_DELAY,0)）", "",
          "| 档 | MAE | ±5min | ±10min |", "|---|---|---|---|"]
    for t in tiers:
        m = results[t]["two_stage"]
        L.append(f"| {TIER_LABELS[t]} | **{m['mae']}** | {m['hit_5min']:.1%} | "
                 f"{m['hit_10min']:.1%} |")
    L += ["", "## 结论要点", ""]
    if all(k in results for k in tiers):
        g_od_v1 = results["od"]["cls"]["auc"] - results["v1"]["cls"]["auc"]
        g_od_wx = results["od"]["cls"]["auc"] - results["wx"]["cls"]["auc"]
        ts_v1 = results["od"]["two_stage"]["mae"] - results["v1"]["two_stage"]["mae"]
        ts_wx = results["od"]["two_stage"]["mae"] - results["wx"]["two_stage"]["mae"]
        L.append(f"- 测试集总增益（天气全链路，od−v1）：AUC **{g_od_v1:+.4f}**，"
                 f"两段式 MAE **{ts_v1:+.2f} 分钟**")
        L.append(f"- 到达侧增益（od−wx）：AUC **{g_od_wx:+.4f}**，"
                 f"两段式 MAE **{ts_wx:+.2f} 分钟**")
    L += ["- 2022→2023 的落差属正常分布漂移（疫情后恢复期运量/延误结构变化），"
          "三档相对排序若保持不变，说明结论稳健。",
          "", "## 产物", "",
          "- 图：`charts/auc_test2023.png`、`charts/twostage_test2023.png`、"
          "`charts/bucket_test2023.png`",
          "- 指标：`test2023_results.json`（含全部模型路径，可复现）", ""]
    (out_dir / "test2023_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
