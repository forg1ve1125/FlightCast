# -*- coding: utf-8 -*-
"""
步骤 12：给"延误区间"模型（5档多分类）加上【双端】天气特征，重训【出发】端。

为什么需要它：产品要求输出"概率 + 该概率下延误多久"——
  > 52% On time · 26% 1-15 min delay (~5 min) · 9% 16-30 min delay (~22 min) · …
这正是区间模型（08_train_bucket_models.py）的职责。天气实验（11 号脚本）只升级了
二分类部件；本脚本把天气特征同样用于区间模型，让"概率+分钟"输出也享受天气增益。

设计：
  - 特征 = 11 号脚本臂4（v1wx_od）：11 计划特征 + 起飞机场@计划起飞时刻 14 个
    天气特征 + 到达机场@计划到达时刻 14 个天气特征（_d 后缀），共 39 个。
    到达侧动机：目的地雷雨/大雾触发 FAA 地面延误程序(GDP)时，起飞机场天气
    良好的航班也被扣在地面。
  - 标签/区间定义与 08 完全相同（config.DELAY_BUCKET_EDGES），模型配置与 08 的
    catboost 相同（MultiClass, depth7, lr0.1, 早停150），**不用类别加权**（保持各档
    概率绝对值可信——这是产品输出"X%概率延误Y分钟"的立身之本）
  - 对照1：08 存档的生产区间模型（bucket_latest.json 指向，catboost，dep 端），
    同一批验证行上比较：①区间准确率 ②逐档校准平均偏差 ③macro F1（赢≥2项才建议采用）
  - 对照2：上一代天气版区间模型（仅起飞机场 25 特征，PREV_WX_RUN 存档）——
    直接归因"到达侧天气带来多少"
  - 额外报告：期望延误 MAE（Σ概率×各档中位分钟 vs 实际，各模型对比）——
    直接衡量"概率+分钟"整体输出质量
  - 数据纪律：训练 2019-2021 / 验证 2022，**2023 不读取**；
    断点续训：模型+metadata 已存在则跳过训练。
  - 不修改 bucket_latest.json / wx_serving.json（生产不变；采用与否人工决定）。

前置：weather/weather_hourly.csv.gz 与 tz_map.json（由 10 号脚本生成）。

用法：
    python 12_train_bucket_weather.py                # 完整训练（1个模型，约15-30分钟）
    python 12_train_bucket_weather.py --run-id <目录> # 断点续训
    python 12_train_bucket_weather.py --quick         # 抽样冒烟（约2分钟）
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
m08 = import_module("08_train_bucket_models")          # bucket_stats / multiclass_val_metrics
m09 = import_module("09_train_cumulative")             # calib_dev_pp / load_flat_models
m11 = import_module("11_train_weather")                # 天气关联与特征列

BUCKET_SHORT_EN = ["On time", "1-15", "16-30", "31-60", ">60"]
PREV_WX_RUN = "20261009_202254_buckets_wx_dep"   # 上一代天气版（仅起飞侧 25 特征）


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


def expected_delay_mae(delay, proba, median_minutes):
    """期望延误 = Σ(各档概率 × 各档中位分钟)，与 max(实际延误,0) 比 MAE。"""
    med = np.array([0.0] + [median_minutes[k] if median_minutes[k] is not None else 0.0
                            for k in range(1, config.N_BUCKETS)])
    exp = proba @ med
    return round(float(np.mean(np.abs(exp - np.clip(delay, 0, None)))), 2)


def main():
    ap = argparse.ArgumentParser(description="天气特征版出发区间模型（对照08）")
    ap.add_argument("--data", default=str(config.CLEAN_CSV))
    ap.add_argument("--run-name", default="buckets_wx_dep")
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
    print("目标: 出发端5档区间模型 + 天气特征（产品输出\"概率+延误多久\"的核心部件）")

    df = make_targets(pd.read_csv(args.data, low_memory=False))
    year = year_series(df)
    df = df[year.isin(list(config.TRAIN_YEARS) + list(config.VAL_YEARS))]
    df = m11.load_weather_features(df, tz_map)
    cov = {"origin": round(float(1 - df["wx_missing_o"].mean()), 4),
           "dest": round(float(1 - df["wx_missing_d"].mean()), 4)}
    print(f"天气覆盖率: 起飞机场 {cov['origin']:.1%} / 到达机场 {cov['dest']:.1%}")

    train = df[year_series(df).isin(config.TRAIN_YEARS)]
    val = df[year_series(df).isin(config.VAL_YEARS)].copy()

    task = "mcls_dep_delay"
    m_tr, m_va = task_mask(train, task), task_mask(val, task)
    tr, va = train[m_tr], val[m_va]
    if args.quick:
        tr = tr.sample(min(len(tr), 150_000), random_state=config.RANDOM_SEED)
        va = va.sample(min(len(va), 80_000), random_state=config.RANDOM_SEED)
    y_tr = tr["y_bucket_dep"].astype(int).to_numpy()
    y_va = va["y_bucket_dep"].astype(int).to_numpy()
    delay_va = pd.to_numeric(va["DEP_DELAY"], errors="coerce").to_numpy()
    print(f"=== [dep] 训练 {len(tr):,} / 验证 {len(va):,} ===")

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
        from catboost import CatBoostClassifier, Pool
        model = CatBoostClassifier(); model.load_model(str(model_path))
        cat_idx = [x_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
        best_iter = int(model.get_best_iteration())
        print("  [v1wx_od] 已存在，跳过训练")
    else:
        model, cat_idx = train_catboost_bucket(x_tr, y_tr, x_va, y_va, quick=args.quick)
        model.save_model(str(model_path))
        best_iter = int(model.get_best_iteration())
    from catboost import Pool
    proba = np.asarray(model.predict_proba(Pool(x_va, cat_features=cat_idx)))
    met = m08.multiclass_val_metrics(y_va, proba)
    met["best_iteration"] = best_iter
    met["seconds"] = round(time.time() - t0, 1)

    # 各档中位分钟（产品里"约X分钟"的来源；口径同 08 的 bucket_stats）
    stats = m08.bucket_stats(train, "DEP_DELAY")
    med = [s["median_minutes"] for s in stats]
    met["expected_delay_mae_vs_max_delay0"] = expected_delay_mae(delay_va, proba, med)
    print(f"  [v1wx_od] acc={met['accuracy']:.4f} macroF1={met['macro_f1']:.4f} "
          f"期望延误MAE={met['expected_delay_mae_vs_max_delay0']} "
          f"(best_iter={best_iter}, {met['seconds']}s)")

    # 对照：08 生产区间模型（catboost, dep）
    flat = m09.load_flat_models().get("dep")
    if flat:
        flat_run_id, fmeta, fmodel = flat
        fx = x_va[fmeta["features"]] if fmeta.get("features") else x_va
        flat_proba = np.asarray(fmodel.predict_proba(fx))
        flat_met = m08.multiclass_val_metrics(y_va, flat_proba)
        flat_met["expected_delay_mae_vs_max_delay0"] = expected_delay_mae(
            delay_va, flat_proba, med)
    else:
        flat_run_id, flat_met, flat_proba = None, None, None

    # 对照2：上一代天气版区间模型（仅起飞侧 25 特征）——归因"到达侧天气带来多少"
    prev_wx, calib_p = None, None
    pj = config.MODELS_DIR / PREV_WX_RUN / "bucket_wx_val_metrics.json"
    if pj.exists():
        prev_wx = json.load(open(pj, encoding="utf-8")).get("v1wx")
        if prev_wx:
            calib_p = m09.calib_dev_pp(prev_wx)

    calib_wx = m09.calib_dev_pp(met)
    calib_f = m09.calib_dev_pp(flat_met) if flat_met else None
    wins = int(met["accuracy"] > flat_met["accuracy"]) if flat_met else 1
    wins += int(calib_wx < calib_f) if flat_met else 1
    wins += int(met["macro_f1"] > flat_met["macro_f1"]) if flat_met else 1
    verdict = {"accuracy_win": bool(met["accuracy"] > flat_met["accuracy"]) if flat_met else True,
               "calibration_win": bool(calib_wx < calib_f) if flat_met else True,
               "macro_f1_win": bool(met["macro_f1"] > flat_met["macro_f1"]) if flat_met else True,
               "wins": wins,
               "recommend": "建议采用天气版区间模型" if wins >= 2 else "保留现状(08区间模型)"}

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id, "task": task,
            "task_desc": "出发延误区间多分类(5档) + 天气特征 —— 产品\"概率+延误多久\"输出",
            "algo": "catboost", "feature_set": "v1wx_od", "model_file": model_path.name,
            "model_path": str(model_path), "features": list(x_tr.columns),
            "categorical_features": CATEGORICAL_FEATURES, "categories": None,
            "n_classes": config.N_BUCKETS, "bucket_edges": config.DELAY_BUCKET_EDGES,
            "bucket_labels": config.BUCKET_LABELS,
            "split": {"train_years": list(config.TRAIN_YEARS),
                      "val_years": list(config.VAL_YEARS),
                      "train_rows": int(len(tr)), "val_rows": int(len(va))},
            "hyperparams": {"loss_function": "MultiClass", "iterations": 300 if args.quick else 1200,
                            "learning_rate": 0.1, "depth": 7, "l2_leaf_reg": 3.0,
                            "class_weighting": "无（保持各档概率绝对值可信）"},
            "random_seed": config.RANDOM_SEED,
            "weather_coverage": cov,
            "val_metrics_2022": met, "vs_flat_verdict": verdict,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "library_versions": {"python": platform.python_version(),
                                 "pandas": pd.__version__, "numpy": np.__version__,
                                 "catboost": __import__("catboost").__version__},
            "mode": "quick" if args.quick else "full",
            "scope": "departure-only (2026-10-08 decision)",
        }, f, ensure_ascii=False, indent=2)
    with open(out_dir / "bucket_stats.json", "w", encoding="utf-8") as f:
        json.dump({"dep": stats}, f, ensure_ascii=False, indent=2)
    with open(out_dir / "bucket_wx_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "mode": "quick" if args.quick else "full",
                   "weather_coverage": cov,
                   "v1wx_od": met, "flat_v1": flat_met, "flat_run_id": flat_run_id,
                   "prev_wx": prev_wx, "prev_wx_run": PREV_WX_RUN,
                   "verdict": verdict}, f, ensure_ascii=False, indent=2)
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id,
                   "created_at": datetime.now().isoformat(timespec="seconds"),
                   "type": "bucket_wx_dep", "mode": "quick" if args.quick else "full",
                   "models": [str(model_path)]}, f, ensure_ascii=False, indent=2)

    make_charts(out_dir, y_va, proba, flat_proba, met, flat_met, calib_wx, calib_f)
    write_report(out_dir, run_id, args.quick, cov, met, flat_met, flat_run_id,
                 verdict, stats, prev_wx, calib_p)

    print("\n" + "=" * 70)
    print(f"区间模型(5档)+双端天气 对照总结（2022 验证集, 覆盖率 起飞 {cov['origin']:.1%}"
          f" / 到达 {cov['dest']:.1%}）")
    print("=" * 70)
    if flat_met:
        print(f"[dep] 区间准确率  : 双端天气版 {met['accuracy']:.4f} vs 08版 {flat_met['accuracy']:.4f}")
        print(f"[dep] 校准平均偏差: 双端天气版 {calib_wx:.2f}pp vs 08版 {calib_f:.2f}pp")
        print(f"[dep] macro F1    : 双端天气版 {met['macro_f1']:.4f} vs 08版 {flat_met['macro_f1']:.4f}")
        print(f"[dep] 期望延误MAE : 双端天气版 {met['expected_delay_mae_vs_max_delay0']} vs "
              f"08版 {flat_met['expected_delay_mae_vs_max_delay0']}")
    else:
        print("[dep] 未找到 08 存档，仅输出天气版指标")
    if prev_wx:
        print(f"[dep] vs 上代天气版(仅起飞): 准确率 {met['accuracy']:.4f} vs {prev_wx['accuracy']:.4f} "
              f"({met['accuracy'] - prev_wx['accuracy']:+.4f}) | macroF1 "
              f"{met['macro_f1']:.4f} vs {prev_wx['macro_f1']:.4f} "
              f"({met['macro_f1'] - prev_wx['macro_f1']:+.4f}) | 期望延误MAE "
              f"{met['expected_delay_mae_vs_max_delay0']} vs "
              f"{prev_wx['expected_delay_mae_vs_max_delay0']}")
    print(f"[dep] 裁决(vs 08版) : 胜出 {verdict['wins']}/3 → {verdict['recommend']}")
    print("-" * 70)
    print(f"报告: {out_dir / 'bucket_wx_report.md'}")
    print(f"图像: {out_dir / 'charts'}/")
    print("注意: 未修改 bucket_latest.json / wx_serving.json，网页仍用现役区间模型")
    print("=" * 70)


def make_charts(out_dir, y_true, proba_wx, proba_flat, met, flat_met, calib_wx, calib_f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    pos = np.arange(config.N_BUCKETS)

    # 1) 混淆矩阵对照（行归一化）
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, (name, proba) in zip(axes, [("v1 + weather (this run)", proba_wx),
                                        ("v1 MultiClass (08 archive)", proba_flat)]):
        cm = confusion_matrix(y_true, proba.argmax(axis=1),
                              labels=list(range(config.N_BUCKETS))) \
            if proba is not None else np.zeros((config.N_BUCKETS, config.N_BUCKETS))
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
        for r in range(config.N_BUCKETS):
            for c in range(config.N_BUCKETS):
                if cm[r, c] > 0:
                    ax.text(c, r, f"{cmn[r, c]:.0f}%", ha="center", va="center", fontsize=9,
                            color="white" if cmn[r, c] > 55 else "#20302c")
        ax.set_xticks(pos, BUCKET_SHORT_EN); ax.set_yticks(pos, BUCKET_SHORT_EN)
        ax.set_xlabel("Predicted bucket"); ax.set_ylabel("Actual bucket")
        ax.set_title(name, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Departure bucket confusion, row-normalized (validation 2022)", fontsize=13)
    fig.tight_layout(); fig.savefig(out_dir / "charts" / "confusion.png", dpi=150)
    plt.close(fig)

    # 2) 逐档校准柱状图
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.27
    actual = [x["actual_rate"] for x in met["probability_calibration"]]
    ax.bar(pos - w, actual, w, color="#2c4036", label="Actual rate")
    ax.bar(pos, [x["mean_pred_prob"] for x in met["probability_calibration"]], w,
           color="#00645a", label="v1+weather (pred)")
    if flat_met:
        ax.bar(pos + w, [x["mean_pred_prob"] for x in flat_met["probability_calibration"]],
               w, color="#c9a227", label="v1 MultiClass (pred)")
    ax.set_xticks(pos, BUCKET_SHORT_EN); ax.set_ylabel("Probability"); ax.legend(fontsize=9)
    ax.set_title(f"Per-bucket calibration (DEP); mean |dev| wx={calib_wx:.2f}pp"
                 + (f" vs v1={calib_f:.2f}pp" if calib_f else ""))
    fig.tight_layout(); fig.savefig(out_dir / "charts" / "calibration.png", dpi=150)
    plt.close(fig)


def write_report(out_dir, run_id, quick, cov, met, flat_met, flat_run_id, verdict,
                 stats, prev_wx=None, calib_p=None):
    L = ["# 区间模型(5档)+双端天气 实验报告（仅出发口径）", "",
         f"- 运行ID：`{run_id}`（{'**quick 抽样，仅供参考**' if quick else '完整训练'}），"
         f"生成于 {datetime.now().isoformat(timespec='seconds')}",
         f"- 目标：产品输出\"概率+该概率下延误多久\"（52% On time · 26% 1-15 min (~5 min)…）"
         f"的区间模型，升级为 v1+双端天气（39 特征，与 11 号脚本臂4一致）",
         f"- 到达侧动机：目的地雷雨/大雾触发 FAA 地面延误程序(GDP)时，起飞机场天气"
         f"良好的航班也被扣在地面",
         f"- 天气覆盖率 起飞 {cov['origin']:.1%} / 到达 {cov['dest']:.1%}；"
         f"纪律：训练 2019-2021 / 验证 2022，2023 未读取；无类别加权",
         f"- 对照1：08 存档 `{flat_run_id}` 的 catboost 区间模型（生产在用）",
         f"- 对照2：上一代天气版 `{PREV_WX_RUN}`（仅起飞侧 25 特征）", "",
         "## 对照结果（2022 验证集）", ""]
    rows = [["区间准确率", met["accuracy"], flat_met["accuracy"] if flat_met else "—"],
            ["macro F1", met["macro_f1"], flat_met["macro_f1"] if flat_met else "—"],
            ["期望延误 MAE(分钟)", met["expected_delay_mae_vs_max_delay0"],
             flat_met["expected_delay_mae_vs_max_delay0"] if flat_met else "—"]]
    L += ["| 指标 | v1+双端天气 | 08版(v1) |", "|---|---|---|"]
    L += [f"| {r[0]} | **{r[1]}** | {r[2]} |" for r in rows]
    L += ["", f"**裁决(vs 08版)**：三准则（准确率/校准/macro F1）胜出 **{verdict['wins']}/3** → "
              f"{verdict['recommend']}（标准：≥2 项才建议采用；未修改 bucket_latest.json）"]
    if prev_wx:
        L += ["", "## 到达侧天气增益（vs 上一代天气版，同验证口径）", "",
              "| 指标 | v1+双端天气 | 上代天气版(仅起飞) | 差值 |", "|---|---|---|---|",
              f"| 区间准确率 | {met['accuracy']:.4f} | {prev_wx['accuracy']:.4f} | "
              f"{met['accuracy'] - prev_wx['accuracy']:+.4f} |",
              f"| macro F1 | {met['macro_f1']:.4f} | {prev_wx['macro_f1']:.4f} | "
              f"{met['macro_f1'] - prev_wx['macro_f1']:+.4f} |",
              f"| 期望延误 MAE | {met['expected_delay_mae_vs_max_delay0']} | "
              f"{prev_wx['expected_delay_mae_vs_max_delay0']} | "
              f"{met['expected_delay_mae_vs_max_delay0'] - prev_wx['expected_delay_mae_vs_max_delay0']:+.2f} |"]
        if calib_p is not None:
            L += [f"| 校准平均偏差 | {m09.calib_dev_pp(met):.2f}pp | {calib_p:.2f}pp | "
                  f"{m09.calib_dev_pp(met) - calib_p:+.2f}pp |"]
    L += ["", "## 各档中位分钟（产品里\"约X分钟\"的换算口径）", ""]
    L += ["| 档位 | 训练集占比 | 中位分钟 |", "|---|---|---|"]
    L += [f"| {s['label']} | {s['train_share']:.1%} | {s['median_minutes']} |" for s in stats]
    L += ["", "## 部署提示", "",
          "本模型在**预测时需要双端天气输入**：训练用实测 METAR（文献标准做法）；线上由"
          "Open-Meteo 预报提供同构特征（server/weather_serving.py 按模型元数据自动"
          "同时取起飞侧与到达侧）。在切换 wx_serving.json 之前，生产网页继续使用现役模型。", ""]
    (out_dir / "bucket_wx_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
