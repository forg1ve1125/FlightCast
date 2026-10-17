# -*- coding: utf-8 -*-
"""
步骤 16：五档模型的不平衡处理消融（课件 L6 的方法，双指标汇报）。

背景：五档训练分布 70/15/5/5/5 严重不平衡，argmax 几乎永远选准点档
（档1-3 的 F1≈0），macro F1 只有 0.17。课件给出三类处理：
  ① 代价敏感（类别加权）——本脚本两个强度臂
  ② 采样法（过/欠采样、SMOTE）——对 CatBoost 数学上≈类别加权，且混合类别
     特征上插值别扭，故以加权臂代表，不重复实现
  ③ 决策规则（课件 88 页：分类时引入代价）——零训练：先验校正 argmax
外加 isotonic 校准（15 号冒烟已验证 macro F1 0.165→0.207 且校准更好）。

纪律（课件 89 页原文：Only rebalance the TRAINING data）：
  - 只有【训练行】做重平衡/加权；验证集保持自然分布评估（诚实口径）；
  - 全部裁决在 2022 验证集；2023 不读取；不改生产指针；断点续训。

双指标：每个方案同时汇报 macro F1（你关心的）和校准偏差（产品关心的），
外加准确率 / 期望延误 MAE / 三档口径 macro F1（任务定义参考）。

用法：
    python 16_bucket_imbalance.py                # 全部臂（2次训练，约80分钟）
    python 16_bucket_imbalance.py --arms w1      # 只跑强加权臂（约40分钟）
    python 16_bucket_imbalance.py --arms w2      # 只跑温和加权臂
    python 16_bucket_imbalance.py --quick        # 冒烟
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
m08 = import_module("08_train_bucket_models")      # multiclass_val_metrics
m11 = import_module("11_train_weather")            # 双端天气
m12 = import_module("12_train_bucket_weather")     # expected_delay_mae
m15 = import_module("15_improve_minutes_bucket")   # isotonic_calibrate / calib_pp

OD_BUCKET = ("20261013_122152_buckets_wx_dep", "mcls_dep_delay__v1wx_od__catboost.cbm")
BUCKET_SHORT = ["On time", "1-15", "16-30", "31-60", ">60"]


def class_weights_of(y, mode):
    """课件口径的代价敏感权重：Balanced=n/(K*n_k)；soft=其平方根（较温和）。"""
    counts = np.bincount(y, minlength=config.N_BUCKETS).astype(float)
    w = counts.sum() / (config.N_BUCKETS * np.maximum(counts, 1))
    return (w / np.sqrt(w)).tolist() if mode == "soft" else w.tolist()


def train_bucket(x_tr, y_tr, x_va, y_va, class_weights, quick=False):
    from catboost import CatBoostClassifier, Pool
    cat_idx = [x_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    model = CatBoostClassifier(
        loss_function="MultiClass", eval_metric="MultiClass",
        iterations=300 if quick else 1200, learning_rate=0.1, depth=7,
        l2_leaf_reg=3.0, random_seed=config.RANDOM_SEED,
        class_weights=class_weights,
        early_stopping_rounds=50 if quick else 150,
        verbose=200, allow_writing_files=False)
    model.fit(Pool(x_tr, y_tr, cat_features=cat_idx),
              eval_set=Pool(x_va, y_va, cat_features=cat_idx), use_best_model=True)
    return model


def load_one(run_id, fname):
    path = config.MODELS_DIR / run_id / fname
    if not path.exists():
        raise SystemExit(f"缺少模型: {path}")
    meta = json.load(open(path.with_name(path.stem + "__metadata.json"), encoding="utf-8"))
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model, meta


def predict_with(model, meta, x_container):
    from catboost import Pool
    x = x_container[meta["features"]]
    ci = [x.columns.get_loc(c) for c in meta["categorical_features"]]
    return np.asarray(model.predict_proba(Pool(x, cat_features=ci)))


def metrics_bundle(y, proba, delay, med, priors=None):
    """统一四层指标包（三层正式体系 + 决策层辅助）。priors 给出时附加先验校正 argmax。"""
    from sklearn.metrics import (f1_score, accuracy_score, log_loss,
                                 cohen_kappa_score, balanced_accuracy_score)
    K = config.N_BUCKETS
    out = {}
    pred = proba.argmax(axis=1)
    onehot = np.eye(K)[y]
    proba_c = np.clip(proba, 1e-9, None)
    # 概率质量层
    out["logloss"] = round(float(log_loss(y, proba_c)), 4)
    out["brier"] = round(float(((proba - onehot) ** 2).sum(axis=1).mean()), 4)
    out["rps"] = round(float(((np.cumsum(proba, 1)[:, :-1] - np.cumsum(onehot, 1)[:, :-1])
                              ** 2).sum(1).mean() / (K - 1)), 4)
    # 校准层
    out["calib_pp"] = m15.calib_pp(y, proba)
    # 决策层（顺序敏感）
    out["qwk"] = round(float(cohen_kappa_score(y, pred, weights="quadratic")), 4)
    out["band_mae"] = round(float(np.abs(pred - y).mean()), 3)
    out["bal_acc"] = round(float(balanced_accuracy_score(y, pred)), 4)
    # 决策层辅助 + 产品口径
    met = m08.multiclass_val_metrics(y, proba)
    out["accuracy"] = met["accuracy"]
    out["macro_f1"] = met["macro_f1"]
    out["expected_mae"] = m12.expected_delay_mae(delay, proba, med)
    # 三档口径（同概率合并，任务定义参考）
    p3 = np.column_stack([proba[:, 0], proba[:, 1] + proba[:, 2], proba[:, 3] + proba[:, 4]])
    y3 = np.select([y == 0, (y == 1) | (y == 2)], [0, 1], default=2)
    out["macro_f1_3band"] = round(float(f1_score(y3, p3.argmax(axis=1), average="macro")), 4)
    if priors is not None:
        adj = proba / priors          # 代价敏感决策规则：除以类先验后再 argmax
        out["prior_adj_macro_f1"] = round(float(f1_score(y, adj.argmax(axis=1),
                                                          average="macro")), 4)
        out["prior_adj_qwk"] = round(float(cohen_kappa_score(
            y, adj.argmax(axis=1), weights="quadratic")), 4)
        out["prior_adj_accuracy"] = round(float(accuracy_score(y, adj.argmax(axis=1))), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description="五档不平衡处理消融（双指标）")
    ap.add_argument("--data", default=str(config.CLEAN_CSV))
    ap.add_argument("--run-name", default="bucket_imb")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--arms", default="w1,w2", help="w1=Balanced加权, w2=温和加权")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    tz_map = json.load(open(m11.WX_DIR / "tz_map.json", encoding="utf-8"))
    run_id = args.run_id or train_mod.make_run_id(args.run_name, quick=args.quick)
    out_dir = config.MODELS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "charts").mkdir(exist_ok=True)
    print(f"存档目录: {out_dir}")

    # ---- 数据（2019-2022；2023 不读取） ----
    df = make_targets(pd.read_csv(args.data, low_memory=False))
    df = df[year_series(df).isin(list(config.TRAIN_YEARS) + list(config.VAL_YEARS))]
    df = m11.load_weather_features(df, tz_map)
    train = df[year_series(df).isin(config.TRAIN_YEARS)]
    val = df[year_series(df).isin(config.VAL_YEARS)]

    m_tr = task_mask(train, "mcls_dep_delay")
    m_va = task_mask(val, "mcls_dep_delay")
    tr, va = train[m_tr], val[m_va]
    if args.quick:
        tr = tr.sample(min(len(tr), 150_000), random_state=config.RANDOM_SEED)
    y_tr = tr["y_bucket_dep"].astype(int).to_numpy()
    y_va = va["y_bucket_dep"].astype(int).to_numpy()
    delay_va = pd.to_numeric(va["DEP_DELAY"], errors="coerce").to_numpy()

    wx_cols = m11.WX_ORIGIN_FEATURE_COLS + m11.WX_DEST_FEATURE_COLS
    x_tr = pd.concat([build_features(tr, feature_set="v1").reset_index(drop=True),
                      tr[wx_cols].reset_index(drop=True)], axis=1)
    x_va = pd.concat([build_features(va, feature_set="v1").reset_index(drop=True),
                      va[wx_cols].reset_index(drop=True)], axis=1)
    priors = np.bincount(y_tr, minlength=config.N_BUCKETS) / len(y_tr)
    med = [s["median_minutes"] for s in json.load(
        open(config.MODELS_DIR / OD_BUCKET[0] / "bucket_stats.json",
             encoding="utf-8"))["dep"]]
    print(f"训练 {len(tr):,} / 验证 {len(va):,} | 训练类先验 " +
          "/".join(f"{p:.3f}" for p in priors))

    results = {}

    # ---- 基准：现役 od（无加权）+ 其先验校正 argmax ----
    bm, bmeta = load_one(*OD_BUCKET)
    proba_od = predict_with(bm, bmeta, x_va)
    results["baseline_od"] = metrics_bundle(y_va, proba_od, delay_va, med, priors)
    print(f"\n[基准] 现役od(无加权): macroF1={results['baseline_od']['macro_f1']} "
          f"校准={results['baseline_od']['calib_pp']}pp acc={results['baseline_od']['accuracy']} "
          f"期望MAE={results['baseline_od']['expected_mae']} "
          f"| 先验校正argmax: macroF1={results['baseline_od']['prior_adj_macro_f1']} "
          f"acc={results['baseline_od']['prior_adj_accuracy']}")

    # ---- 加权臂（代价敏感，课件 88 页；只作用于训练损失） ----
    for arm in arms:
        mode = "soft" if arm == "w2" else "balanced"
        w = class_weights_of(y_tr, mode)
        print(f"\n[{arm}] 类别权重({mode}) = " +
              "/".join(f"{x:.2f}" for x in w))
        meta_path = out_dir / f"mcls_dep_delay__{arm}__catboost__metadata.json"
        model_path = out_dir / f"mcls_dep_delay__{arm}__catboost.cbm"
        t0 = time.time()
        if model_path.exists():   # 模型在即可续（评估中途崩过也不必重训）
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            print(f"  [{arm}] 已存在，跳过训练")
        else:
            model = train_bucket(x_tr, y_tr, x_va, y_va, w, quick=args.quick)
            model.save_model(str(model_path))
        proba = predict_with(model, {"task": "mcls", "features": list(x_tr.columns),
                                     "categorical_features": CATEGORICAL_FEATURES}, x_va)
        res = metrics_bundle(y_va, proba, delay_va, med, priors)
        res["class_weights"] = [round(v, 4) for v in w]
        res["seconds"] = round(time.time() - t0, 1)
        results[arm] = res
        print(f"  [{arm}] macroF1={res['macro_f1']} (Δ{res['macro_f1']-results['baseline_od']['macro_f1']:+.4f}) "
              f"校准={res['calib_pp']}pp acc={res['accuracy']} "
              f"期望MAE={res['expected_mae']} 三档F1={res['macro_f1_3band']} ({res['seconds']}s)")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_id, "task": "mcls_dep_delay",
                       "task_desc": f"五档+类别加权({mode})——不平衡处理消融",
                       "algo": "catboost", "feature_set": f"imb_{arm}",
                       "model_file": model_path.name, "model_path": str(model_path),
                       "features": list(x_tr.columns),
                       "categorical_features": CATEGORICAL_FEATURES,
                       "n_classes": config.N_BUCKETS,
                       "class_weights": res["class_weights"],
                       "split": {"train_years": list(config.TRAIN_YEARS),
                                 "val_years": list(config.VAL_YEARS)},
                       "val_metrics_2022": res,
                       "note": "仅训练集加权（课件: Only rebalance TRAINING data）；验证集自然分布",
                       "trained_at": datetime.now().isoformat(timespec="seconds"),
                       "mode": "quick" if args.quick else "full"}, f,
                     ensure_ascii=False, indent=2)

    # ---- isotonic（零训练：2022H1 拟合 / H2 验证） ----
    months = pd.to_datetime(va["FL_DATE"]).dt.month
    h1 = (months <= 6).to_numpy()
    proba_cal = m15.isotonic_calibrate(y_va[h1], proba_od[h1], y_va[~h1], proba_od[~h1])
    results["isotonic_H2"] = metrics_bundle(y_va[~h1], proba_cal, delay_va[~h1], med)
    raw_h2 = metrics_bundle(y_va[~h1], proba_od[~h1], delay_va[~h1], med)
    print(f"\n[isotonic H2] macroF1={results['isotonic_H2']['macro_f1']} "
          f"(未校准同批 {raw_h2['macro_f1']}) 校准={results['isotonic_H2']['calib_pp']}pp "
          f"(未校准 {raw_h2['calib_pp']}pp)")
    results["raw_H2_reference"] = raw_h2

    # ---- 汇总 ----
    payload = {"run_id": run_id, "mode": "quick" if args.quick else "full",
               "evaluated_at": datetime.now().isoformat(timespec="seconds"),
               "class_priors_train": [round(float(p), 4) for p in priors],
               "arms": results,
               "note": "双指标（macroF1=决策分辨, calib_pp=概率可信）；全部2022自然分布验证；2023未读取；未改生产指针"}
    with open(out_dir / "imbalance_val2022.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    make_charts(out_dir, results)
    write_report(out_dir, run_id, payload)

    print("\n" + "=" * 96)
    print("五档不平衡处理消融总结（2022 验证，四层指标）")
    print("=" * 96)
    print(f"{'方案':<22}{'LogLoss↓':>9}{'Brier↓':>8}{'RPS↓':>8}{'校准pp':>8}"
          f"{'QWK↑':>7}{'macroF1':>9}{'acc':>8}{'期望MAE':>9}")
    rows = [("现役od(无加权)", results["baseline_od"], None)] + \
           [(f"加权{a}", results[a], None) for a in arms if a in results] + \
           [("先验校正argmax*", results["baseline_od"], "adj"),
            ("isotonic(H2)", results["isotonic_H2"], None)]
    for name, r, mode in rows:
        if mode == "adj":
            mf, qwk = r["prior_adj_macro_f1"], r["prior_adj_qwk"]
        else:
            mf, qwk = r["macro_f1"], r["qwk"]
        print(f"{name:<22}{r['logloss']:>9}{r['brier']:>8}{r['rps']:>8}"
              f"{r['calib_pp']:>8}{qwk:>7}{mf:>9}{r['accuracy']:>8}{r['expected_mae']:>9}")
    print("*先验校正 argmax 只改 verdict 规则，概率/校准与现役完全相同")
    print("-" * 76)
    print(f"报告: {out_dir / 'imbalance_report.md'}")
    print("注意: 未修改生产指针；采纳与否看跷跷板选点")
    print("=" * 76)


def make_charts(out_dir, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [("baseline_od", "no weight"), ("w1", "Balanced"), ("w2", "soft weight")]
    names = [(k, lab) for k, lab in names if k in results]
    if "isotonic_H2" in results:
        names.append(("isotonic_H2", "isotonic (H2)"))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (k, lab) in enumerate(names):
        r = results[k]
        ax.scatter(r["calib_pp"], r["macro_f1"], s=90)
        ax.annotate(lab, (r["calib_pp"], r["macro_f1"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    b = results["baseline_od"]
    if b.get("prior_adj_macro_f1"):
        ax.scatter(b["calib_pp"], b["prior_adj_macro_f1"], s=90, marker="x", color="#b23a3a")
        ax.annotate("prior-adjusted argmax", (b["calib_pp"], b["prior_adj_macro_f1"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=9, color="#b23a3a")
    ax.set_xlabel("Calibration |dev| (pp)  ——越小概率越可信")
    ax.set_ylabel("macro F1 (5-band)")
    ax.set_title("Trade-off: macro F1 vs calibration (2022 val)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "charts" / "tradeoff.png", dpi=150)
    plt.close(fig)


def write_report(out_dir, run_id, payload):
    r = payload["arms"]
    L = ["# 五档不平衡处理消融（课件 L6 方法）", "",
         f"- 运行ID：`{run_id}`，生成于 {payload['evaluated_at']}",
         f"- 训练类先验: {payload['class_priors_train']}",
         "- 纪律：仅训练集加权/校准（课件：Only rebalance the TRAINING data），"
         "验证集保持自然分布；2022 裁决；2023 未读取；未改生产指针", "",
         "| 方案 | macro F1 | 校准pp | acc | 期望MAE | 三档F1 |", "|---|---|---|---|---|---|"]
    for k, lab in [("baseline_od", "现役 od（无加权）"), ("w1", "加权 Balanced"),
                   ("w2", "加权 soft"), ("isotonic_H2", "isotonic（H2 验证）")]:
        if k in r:
            m = r[k]
            L.append(f"| {lab} | {m['macro_f1']:.4f} | {m['calib_pp']:.2f} | "
                     f"{m['accuracy']:.4f} | {m['expected_mae']} | {m['macro_f1_3band']:.4f} |")
    if "prior_adj_macro_f1" in r.get("baseline_od", {}):
        m = r["baseline_od"]
        L.append(f"| 先验校正 argmax（零训练） | {m['prior_adj_macro_f1']:.4f} | "
                 f"{m['calib_pp']:.2f}(不变) | {m['prior_adj_accuracy']:.4f} | "
                 f"{m['expected_mae']} | {m['macro_f1_3band']:.4f} |")
    L += ["", "## 解读与选点建议", "",
          "- 加权臂：macro F1 上去的代价是校准变差（概率虚高）——课件 89 页警告的换价。",
          "- 先验校正 argmax / isotonic：不动概率、只动决策，是“两头兼顾”的位置。",
          "- 三档口径 F1 供任务定义参考（同概率合并，无需重训）。",
          "- 采纳任一加权臂时，报告中必须如实注明“概率未校准”。", ""]
    (out_dir / "imbalance_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
