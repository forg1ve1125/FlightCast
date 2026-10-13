# -*- coding: utf-8 -*-
"""
步骤 11：加入逐小时天气特征重训【出发延误】二分类（延误>0），四臂对照。

范围决策（2026-10-08，导师建议聚焦单一任务）：本项目只预测出发延误，
不再训练/输出到达延误模型。CRS_ARR_TIME 仍作为表单输入与特征原料
（用于唯一确定航班 + 块时长特征 CRS_TIME_DIFF），但不是预测目标。

前置：先运行 `python 10_weather_download.py` 生成
  weather/weather_hourly.csv.gz（机场逐小时天气，UTC）
  weather/tz_map.json（机场 UTC 偏移，从航班时刻表推导）

设计（四臂对照，全部在【同一批行】上训练，保证公平）：
  臂1 v1         : 原 11 个计划特征（基线）
  臂2 v1+天气    : v1 + 14 个天气特征（起飞机场@计划起飞时刻: 气温/露点/湿度/
                   风速/阵风/能见度/降水/云底/云量/雷暴·雪·雨·雾标志 + 缺测标志）
                   -> 干净归因"天气带来多少"
  臂3 v1+天气+块 : 臂2 + 3 个块时长特征(CRS_ELAPSED_TIME / 计划起降分钟差 /
                   两者之差≈时区差)
  臂4 v1+双端天气: v1 + 起飞机场天气(同臂2) + 到达机场@计划到达时刻的 14 个天气
                   特征(_d 后缀)。动机：目的地雷雨/大雾会触发 FAA 地面延误程序
                   (GDP)，航班在起飞机场天气良好时也被扣在地面——只看起飞侧
                   看不见这类延误。到达侧取样：到达当地时刻=FL_DATE+(到达分钟
                   <出发分钟?1天:0)+CRS_ARR_TIME，夏令时按到达机场/到达日期判定。
  另展示生产基线(20261005_155440_final_gt0)的出发验证集AUC作参照。
  - 数据纪律：训练 2019-2021 / 验证 2022，**2023 测试集完全不读取**；
    训练配置与生产二分类一致（CatBoost Logloss, lr0.1/depth8/Balanced, 早停150）。
  - 断点续训：模型+metadata 已存在则跳过训练、仅重算指标/图/报告。
    在旧三臂存档上续跑只训练新臂4：--run-id 20261009_193645_wx_dep

诚实披露（写进报告）：这里用的是"实测天气"（METAR）。学术文献的标准做法
即是如此；若未来部署到真实"提前数天预测"场景，应接入天气预报数据
（结构与本表一致，可直接替换）。这是研究与部署的标准差距，报告中已注明。

用法：
    python 11_train_weather.py                 # 完整训练（4个模型，约35-60分钟）
    python 11_train_weather.py --run-id 20261009_193645_wx_dep
                                               # 断点续训：旧三臂跳过，只训练臂4
    python 11_train_weather.py --quick         # 抽样冒烟（约4分钟）
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

WX_DIR = Path(__file__).resolve().parent / "weather"
WX_COLS = ["tmpf", "dwpf", "relh", "sknt", "gust", "vsby", "p01i",
           "skyl1", "skyc1_okta", "wx_ts", "wx_sn", "wx_ra", "wx_fg"]
WX_ORIGIN_FEATURE_COLS = [f"{c}_o" for c in WX_COLS] + ["wx_missing_o"]
WX_DEST_FEATURE_COLS = [f"{c}_d" for c in WX_COLS] + ["wx_missing_d"]
WX_FEATURE_COLS = WX_ORIGIN_FEATURE_COLS   # 兼容别名（12/13 号脚本旧引用）
BASELINE_RUN = "20261005_155440_final_gt0"


# ---------------------------------------------------------------- 工具
def _nth_sunday(year, month, n):
    from datetime import date, timedelta
    d0 = date(year, month, 1)
    return d0 + timedelta(days=(6 - d0.weekday()) % 7 + 7 * (n - 1))


def in_us_dst(d):
    """美国夏令时窗口：3月第2个周日 ~ 11月第1个周日。接受 date/Timestamp/str。"""
    d = pd.Timestamp(d).date()
    return _nth_sunday(d.year, 3, 2) <= d < _nth_sunday(d.year, 11, 1)


def _wx_keys(local_dates: pd.Series, airport: pd.Series, hhmm: pd.Series,
             tz_map: dict) -> pd.Series:
    """机场当地时间(日期列+hhmm) -> UTC 整点的 ns 键；偏移未知则哨兵值(必不命中)。

    夏令时按【该机场、该当地日期】判定（到达侧用到达日期，跨 DST 边界才正确）。
    """
    hhmm = pd.to_numeric(hhmm, errors="coerce")
    minutes = (hhmm // 100) * 60 + (hhmm % 100)
    local = pd.to_datetime(local_dates) + pd.to_timedelta(minutes, unit="m")
    base = airport.map(lambda c: (tz_map.get(c) or {}).get("base_offset_hours"))
    has_dst = airport.map(lambda c: (tz_map.get(c) or {}).get("has_dst", True))
    dst = local.dt.date.map(in_us_dst)
    off = pd.to_numeric(base, errors="coerce") + (dst & has_dst).astype(float)
    utc = local - pd.to_timedelta(off * 3600, unit="s")
    h = utc.dt.floor("h")
    key = pd.Series(np.iinfo(np.int64).min, index=h.index, dtype="int64")
    ok = h.notna()
    # 强制 ns 单位再转整数，与 weather_hourly.csv.gz 的键（ns）严格一致
    key[ok] = h[ok].astype("datetime64[ns]").astype("int64")
    return key


def arrival_local_dates(df: pd.DataFrame) -> pd.Series:
    """计划到达的【当地日期】：到达分钟<出发分钟 => 跨夜，FL_DATE+1 天。

    时刻按"分钟"比较（hhmm 整数直接相减是错的，如 0030-2350=-320）。
    US 国内不存在"时区增益>飞行时长"的航线，该启发式误判率可忽略。
    """
    dep = pd.to_numeric(df["CRS_DEP_TIME"], errors="coerce")
    arr = pd.to_numeric(df["CRS_ARR_TIME"], errors="coerce")
    dep_m = (dep // 100) * 60 + dep % 100
    arr_m = (arr // 100) * 60 + arr % 100
    next_day = (arr_m < dep_m).fillna(False)
    return (pd.to_datetime(df["FL_DATE"])
            + pd.to_timedelta(next_day.astype("int64"), unit="D"))


def load_weather_features(df: pd.DataFrame, tz_map: dict) -> pd.DataFrame:
    """拼上【双端】天气：起飞机场@计划起飞时刻(_o) + 到达机场@计划到达时刻(_d)。

    两侧共用 weather_hourly.csv.gz（380 站既覆盖出发也覆盖到达机场）。
    天气列统一降为 float32 控内存（28 列 × 全量行）。
    """
    wx = pd.read_csv(WX_DIR / "weather_hourly.csv.gz",
                     usecols=["station", "utc_hour"] + WX_COLS,
                     dtype={"station": str})
    wx["utc_hour"] = wx["utc_hour"].astype("int64")
    wx[WX_COLS] = wx[WX_COLS].astype("float32")

    df = df.copy()
    # 起飞侧
    df["_key_o"] = _wx_keys(df["FL_DATE"], df["ORIGIN"], df["CRS_DEP_TIME"], tz_map)
    wx_o = wx.add_suffix("_o").rename(
        columns={"station_o": "station", "utc_hour_o": "utc_hour"})
    df = df.merge(wx_o, how="left", left_on=["ORIGIN", "_key_o"],
                  right_on=["station", "utc_hour"]).drop(
                  columns=["station", "utc_hour", "_key_o"])
    df["wx_missing_o"] = df["tmpf_o"].isna().astype("int8")

    # 到达侧（跨夜+1天；按到达机场/到达日期判 DST）
    arr_dates = arrival_local_dates(df)
    df["_key_d"] = _wx_keys(arr_dates, df["DEST"], df["CRS_ARR_TIME"], tz_map)
    wx_d = wx.add_suffix("_d").rename(
        columns={"station_d": "station", "utc_hour_d": "utc_hour"})
    df = df.merge(wx_d, how="left", left_on=["DEST", "_key_d"],
                  right_on=["station", "utc_hour"]).drop(
                  columns=["station", "utc_hour", "_key_d"])
    df["wx_missing_d"] = df["tmpf_d"].isna().astype("int8")
    return df


def block_features(d: pd.DataFrame) -> pd.DataFrame:
    """臂3 的块时长特征：计划飞行时长 / 计划起降时刻差(分钟) / 两者之差(≈−时区差)。

    时刻差按"分钟"计算（hhmm 整数直接相减是错的，如 2015-1830=185 而非 105）。
    树模型无法自己做减法，显式给出 CRS_BLOCK_MINUS_DIFF 让它直接分裂；
    跨日取模伪影(±1440)对树无害（只是分裂点上移一段）。
    """
    dep = pd.to_numeric(d["CRS_DEP_TIME"], errors="coerce")
    arr = pd.to_numeric(d["CRS_ARR_TIME"], errors="coerce")
    diff = ((arr // 100) * 60 + arr % 100) - ((dep // 100) * 60 + dep % 100)
    diff = diff % 1440
    elapsed = pd.to_numeric(d["CRS_ELAPSED_TIME"], errors="coerce")
    return pd.DataFrame({
        "CRS_ELAPSED_TIME": elapsed,
        "CRS_TIME_DIFF": diff,
        "CRS_BLOCK_MINUS_DIFF": elapsed - diff,
    })


# ---------------------------------------------------------------- 训练/评估
def train_catboost_cls(x_tr, y_tr, x_va, y_va, quick=False):
    from catboost import CatBoostClassifier, Pool
    cat_idx = [x_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    model = CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC",
        iterations=300 if quick else 1500, learning_rate=0.1, depth=8,
        l2_leaf_reg=3.0, auto_class_weights="Balanced",
        random_seed=config.RANDOM_SEED,
        early_stopping_rounds=50 if quick else 150,
        verbose=200, allow_writing_files=False)
    model.fit(Pool(x_tr, y_tr, cat_features=cat_idx),
              eval_set=Pool(x_va, y_va, cat_features=cat_idx), use_best_model=True)
    return model, cat_idx


def cls_val_metrics(y_true, p):
    from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                                 precision_score, recall_score, roc_auc_score)
    pred = (p >= 0.5).astype(int)
    return {
        "threshold": 0.5,
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "auc": round(float(roc_auc_score(y_true, p)), 4),
        "logloss": round(float(log_loss(y_true, p)), 4),
    }


def main():
    ap = argparse.ArgumentParser(description="天气特征版出发延误二分类（三臂对照）")
    ap.add_argument("--data", default=str(config.CLEAN_CSV))
    ap.add_argument("--run-name", default="wx_dep")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    tz_path = WX_DIR / "tz_map.json"
    wx_path = WX_DIR / "weather_hourly.csv.gz"
    for p in (tz_path, wx_path):
        if not p.exists():
            raise SystemExit(f"缺少 {p}，请先运行: python 10_weather_download.py")
    tz_map = json.load(open(tz_path, encoding="utf-8"))

    run_id = args.run_id or train_mod.make_run_id(args.run_name, quick=args.quick)
    out_dir = config.MODELS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "charts").mkdir(exist_ok=True)
    print(f"存档目录: {out_dir}{'（quick，指标仅供参考）' if args.quick else ''}")
    print("范围: 仅出发延误（2026-10-08 起的项目口径），天气=起飞机场@计划起飞时刻")

    # 只读 2019-2022（2023 完全不触碰）
    df = make_targets(pd.read_csv(args.data, low_memory=False))
    year = year_series(df)
    df = df[year.isin(list(config.TRAIN_YEARS) + list(config.VAL_YEARS))]
    df = load_weather_features(df, tz_map)
    cov_o = 1 - df["wx_missing_o"].mean()
    cov_d = 1 - df["wx_missing_d"].mean()
    cov = {"origin": round(float(cov_o), 4), "dest": round(float(cov_d), 4)}
    print(f"天气覆盖率: 起飞机场 {cov_o:.1%} / 到达机场 {cov_d:.1%}")

    train = df[year_series(df).isin(config.TRAIN_YEARS)]
    val = df[year_series(df).isin(config.VAL_YEARS)].copy()

    results, importances = {}, {}
    seg, y_col, task = "dep", "y_cls_dep", "cls_dep_delay"
    m_tr, m_va = task_mask(train, task), task_mask(val, task)
    tr, va = train[m_tr], val[m_va]
    if args.quick:
        tr = tr.sample(min(len(tr), 150_000), random_state=config.RANDOM_SEED)
        va = va.sample(min(len(va), 80_000), random_state=config.RANDOM_SEED)
    y_tr, y_va = tr[y_col].astype(int).to_numpy(), va[y_col].astype(int).to_numpy()
    print(f"\n=== [dep] 训练 {len(tr):,} / 验证 {len(va):,} ===")

    x_v1_tr = build_features(tr, feature_set="v1")
    x_v1_va = build_features(va, feature_set="v1")
    wx_tr = tr[WX_ORIGIN_FEATURE_COLS].reset_index(drop=True)
    wx_va = va[WX_ORIGIN_FEATURE_COLS].reset_index(drop=True)
    wxd_tr = tr[WX_DEST_FEATURE_COLS].reset_index(drop=True)
    wxd_va = va[WX_DEST_FEATURE_COLS].reset_index(drop=True)
    x_wx_tr = pd.concat([x_v1_tr.reset_index(drop=True), wx_tr], axis=1)
    x_wx_va = pd.concat([x_v1_va.reset_index(drop=True), wx_va], axis=1)
    blk_tr = block_features(tr).reset_index(drop=True)
    blk_va = block_features(va).reset_index(drop=True)
    x_full_tr = pd.concat([x_wx_tr, blk_tr], axis=1)
    x_full_va = pd.concat([x_wx_va, blk_va], axis=1)
    x_od_tr = pd.concat([x_wx_tr, wxd_tr], axis=1)
    x_od_va = pd.concat([x_wx_va, wxd_va], axis=1)

    results[seg] = {}
    for variant, x_tr, x_va in [("v1", x_v1_tr, x_v1_va),
                                 ("v1wx", x_wx_tr, x_wx_va),
                                 ("v1wx_block", x_full_tr, x_full_va),
                                 ("v1wx_od", x_od_tr, x_od_va)]:
        meta_path = out_dir / f"{task}__{variant}__catboost__metadata.json"
        model_path = out_dir / f"{task}__{variant}__catboost.cbm"
        t0 = time.time()
        if meta_path.exists() and model_path.exists():
            from catboost import CatBoostClassifier, Pool
            model = CatBoostClassifier(); model.load_model(str(model_path))
            cat_idx = [x_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
            best_iter = int(model.get_best_iteration())
            print(f"  [{variant}] 已存在，跳过训练")
        else:
            model, cat_idx = train_catboost_cls(x_tr, y_tr, x_va, y_va, quick=args.quick)
            model.save_model(str(model_path))
            best_iter = int(model.get_best_iteration())
        p_va = model.predict_proba(
            __import__("catboost").Pool(x_va, cat_features=cat_idx))[:, 1]
        met = cls_val_metrics(y_va, p_va)
        met["best_iteration"] = best_iter
        met["seconds"] = round(time.time() - t0, 1)
        print(f"  [{variant}] AUC={met['auc']:.4f} (logloss={met['logloss']}, "
              f"best_iter={best_iter}, {met['seconds']}s)")
        if variant in ("v1wx_block", "v1wx_od"):
            imp = model.get_feature_importance()
            importances.setdefault(seg, {})[variant] = sorted(
                zip(x_tr.columns, imp), key=lambda t: -t[1])
        variant_desc = {"v1": "", "v1wx": " +天气特征(起飞机场)",
                        "v1wx_block": " +天气+块时长特征",
                        "v1wx_od": " +双端天气(起飞+到达机场)"}[variant]
        meta = {
            "run_id": run_id, "task": task,
            "task_desc": config.TASKS[task] + variant_desc + "（仅出发延误口径）",
            "algo": "catboost", "feature_set": variant, "model_file": model_path.name,
            "model_path": str(model_path), "features": list(x_tr.columns),
            "categorical_features": CATEGORICAL_FEATURES, "categories": None,
            "split": {"train_years": list(config.TRAIN_YEARS),
                      "val_years": list(config.VAL_YEARS),
                      "train_rows": int(len(tr)), "val_rows": int(len(va))},
            "hyperparams": {"loss_function": "Logloss", "eval_metric": "AUC",
                            "iterations": 300 if args.quick else 1500,
                            "learning_rate": 0.1, "depth": 8, "l2_leaf_reg": 3.0,
                            "auto_class_weights": "Balanced"},
            "random_seed": config.RANDOM_SEED,
            "weather_coverage": cov,
            "val_metrics_2022": met,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "library_versions": {"python": platform.python_version(),
                                 "pandas": pd.__version__, "numpy": np.__version__,
                                 "catboost": __import__("catboost").__version__},
            "mode": "quick" if args.quick else "full",
            "scope": "departure-only (2026-10-08 decision)",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        results[seg][variant] = met

    # 生产基线（参照值）
    baseline = {}
    mp = config.MODELS_DIR / BASELINE_RUN / "cls_dep_delay__catboost__metadata.json"
    if mp.exists():
        baseline["dep"] = json.load(open(mp, encoding="utf-8"))["val_metrics_2022"]["auc"]

    make_outputs(out_dir, run_id, args.quick, results, importances, baseline, cov)
    print_summary(out_dir, results, baseline, cov)

    # 存档规范（与 08/09 一致）：run_info + 汇总指标 json
    with open(out_dir / "wx_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "mode": "quick" if args.quick else "full",
                   "scope": "departure-only",
                   "weather_coverage": cov,
                   "segments": results}, f, ensure_ascii=False, indent=2)
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id,
                   "created_at": datetime.now().isoformat(timespec="seconds"),
                   "type": "wx_dep_four_arm",
                   "mode": "quick" if args.quick else "full",
                   "models": [str(p) for p in sorted(out_dir.glob("*.cbm"))]},
                  f, ensure_ascii=False, indent=2)


ARMS = [("v1", "arm1\nv1 (schedule)", "#9aa5a0", "臂1 基线"),
        ("v1wx", "arm2\nv1 + weather\n(origin)", "#00645a", "臂2 +天气(起飞)"),
        ("v1wx_block", "arm3\narm2 + block\ntime", "#c9a227", "臂3 +天气+块时长"),
        ("v1wx_od", "arm4\nv1 + weather\n(origin+dest)", "#3f6fb5", "臂4 +双端天气")]


def make_outputs(out_dir, run_id, quick, results, importances, baseline, cov):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = results["dep"]
    arms = [(v, lab, c) for v, lab, c, _ in ARMS if v in r]
    # 1) AUC 四臂对照柱状图（仅出发）
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    vals = [r[v]["auc"] for v, _, _ in arms]
    labels = [lab for _, lab, _ in arms]
    colors = [c for _, _, c in arms]
    ax.bar(range(len(arms)), vals, 0.55, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=10)
    if "dep" in baseline:
        ax.hlines(baseline["dep"], -0.4, len(arms) - 0.6, color="#b23a3a", ls="--",
                  lw=1.5, label=f"production baseline {baseline['dep']:.4f}")
        ax.legend(fontsize=9)
    ax.set_xticks(range(len(arms)), labels)
    ax.set_ylabel("Validation AUC (2022)")
    ax.set_ylim(0.55, max(vals) + 0.03)
    ax.set_title("Departure-delay AUC: four-arm comparison")
    fig.tight_layout(); fig.savefig(out_dir / "charts" / "auc_comparison.png", dpi=150)
    plt.close(fig)

    # 2) 全量模型(臂3/臂4)特征重要性 top20
    for seg, per_var in importances.items():
        for variant, imp in per_var.items():
            top = imp[:20][::-1]
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.barh([t[0] for t in top], [t[1] for t in top], color="#00645a")
            ax.set_xlabel("CatBoost feature importance")
            ax.set_title(f"Top-20 features, {variant} (DEP)")
            fig.tight_layout()
            fig.savefig(out_dir / "charts" / f"feature_importance_{seg}__{variant}.png",
                        dpi=150)
            plt.close(fig)

    # 3) MD 报告
    L = ["# 天气特征实验报告（仅出发延误口径）", "",
         f"- 运行ID：`{run_id}`（{'**quick 抽样，仅供参考**' if quick else '完整训练'}），"
         f"生成于 {datetime.now().isoformat(timespec='seconds')}",
         f"- 范围决策：2026-10-08 起仅预测出发延误（导师建议聚焦单一任务）",
         f"- 天气：起飞机场@计划起飞时刻 + 到达机场@计划到达时刻（跨夜+1天）的逐小时"
         f"METAR 实测（UTC 关联），覆盖率 起飞 {cov['origin']:.1%} / 到达 {cov['dest']:.1%}",
         f"- 臂4 动机：目的地雷雨/大雾触发 FAA 地面延误程序(GDP)时，起飞机场天气良好"
         f"的航班也被扣在地面——只看起飞侧看不见这类延误",
         f"- 对照：同批行四臂，生产基线 `{BASELINE_RUN}` 出发验证集AUC 作参照",
         f"- 纪律：训练 2019-2021 / 验证 2022，**2023 未读取**", "",
         "**诚实披露**：本实验用实测天气（文献标准做法）。若部署为提前数天的真实预测，"
         "应替换为数值天气预报（表结构一致可直接替换），届时增益会略小于本报告。", "",
         "## 出发延误（>0）", "",
         "| 臂 | 特征 | AUC | logloss | acc@0.5 | precision | recall | F1 |",
         "|---|---|---|---|---|---|---|---|"]
    for v, _, _, label in ARMS:
        if v not in r:
            continue
        m = r[v]
        L.append(f"| {label} | {v} | **{m['auc']:.4f}** | {m['logloss']} | "
                 f"{m['accuracy']} | {m['precision']} | {m['recall']} | {m['f1']} |")
    if "dep" in baseline:
        L += ["", f"生产基线(20261005 存档, 全量行训练)参照 AUC：**{baseline['dep']:.4f}**"]
    gains = []
    if "v1wx" in r and "v1" in r:
        gains.append(f"天气增益(臂2−臂1)：**{r['v1wx']['auc'] - r['v1']['auc']:+.4f}**")
    if "v1wx_block" in r and "v1wx" in r:
        gains.append(f"块时长增益(臂3−臂2)：**{r['v1wx_block']['auc'] - r['v1wx']['auc']:+.4f}**")
    if "v1wx_od" in r and "v1wx" in r:
        gains.append(f"到达侧天气增益(臂4−臂2)：**{r['v1wx_od']['auc'] - r['v1wx']['auc']:+.4f}**")
    if gains:
        L += ["", "；".join(gains)]
    best = max((r[v]["auc"] for v, _, _, _ in ARMS if v in r), default=0.0)
    L += ["", f"最优臂是否达到 0.70：**{'是 ✅' if best >= 0.70 else '否'}**", ""]
    for variant, per_var in [("v1wx_block", importances.get("dep", {})),
                             ("v1wx_od", importances.get("dep", {}))]:
        if variant in per_var:
            imp5 = ", ".join(f"{n}({v:.1f})" for n, v in per_var[variant][:10])
            L += [f"{variant} 模型重要性 Top-10：{imp5}", ""]
    L += ["", "## 产物", "",
          "- 图：`charts/auc_comparison.png`、`charts/feature_importance_dep__*.png`",
          "- 模型：`cls_dep_delay__{v1|v1wx|v1wx_block|v1wx_od}__catboost.cbm` × 4 + metadata",
          "- 本脚本未改动 `latest_run.json`（生产不变）", ""]
    (out_dir / "wx_report.md").write_text("\n".join(L), encoding="utf-8")


def print_summary(out_dir, results, baseline, cov):
    r = results["dep"]
    names = {"v1": "臂1 v1        ", "v1wx": "臂2 +天气    ",
             "v1wx_block": "臂3 +天气+块 ", "v1wx_od": "臂4 +双端天气"}
    b = f" | 生产基线 {baseline['dep']:.4f}" if "dep" in baseline else ""
    print("\n" + "=" * 70)
    print(f"出发延误·天气四臂总结（2022 验证集, 覆盖率 起飞 {cov['origin']:.1%}"
          f" / 到达 {cov['dest']:.1%}）")
    print("=" * 70)
    prev_auc = None
    for v, _, _, _ in ARMS:
        if v not in r:
            continue
        gain = f" ({r[v]['auc'] - prev_auc:+.4f})" if prev_auc is not None else ""
        print(f"[dep] {names[v]} = {r[v]['auc']:.4f}{gain}")
        prev_auc = r[v]["auc"]
    if "v1wx" in r and "v1wx_od" in r:
        print(f"[dep] 到达侧天气增益(臂4−臂2): {r['v1wx_od']['auc'] - r['v1wx']['auc']:+.4f}")
    best = max((r[v]["auc"] for v, _, _, _ in ARMS if v in r), default=0.0)
    print(f"[dep] 最优臂是否达到 0.70: {'是 ✅' if best >= 0.70 else '否'}{b}")
    print("-" * 70)
    print(f"报告: {out_dir / 'wx_report.md'}")
    print(f"图像: {out_dir / 'charts'}/")
    print("注意: 未修改 latest_run.json，生产与网页不受影响")
    print("=" * 70)


if __name__ == "__main__":
    main()
