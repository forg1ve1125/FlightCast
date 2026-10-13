# -*- coding: utf-8 -*-
"""
步骤 6：评估"延误时间（分钟数）预测得准不准"——在 2022 验证集上（不触碰测试集）。

评估两个层面：

A. 延误航班的分钟数回归（已知延误的前提下，预测延误多少分钟）
   指标：MAE / RMSE / R²（经典回归指标）
        ±5 / ±10 / ±15 分钟命中率（预测误差带内的比例，直观易懂）
   基线对照：恒定预测训练集均值 / 中位数（回归模型必须打败它们才有意义）

B. 两段式合成的期望延误（全体航班）：期望延误 = P(延误) × E[分钟|延误]
   指标：MAE / ±5 / ±10 / ±15 分钟命中率
   基线对照：全部预测0分钟 / 恒定预测训练集平均延误

用法：
    python 06_eval_regression.py                    # 评估最近一次存档
    python 06_eval_regression.py --run-id 20261005_155440_final_gt0
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

import config
from features import build_features, make_targets, task_mask, year_series
from importlib import import_module

eval_mod = import_module("03_evaluate_models")


def reg_metrics(y, pred):
    from sklearn.metrics import r2_score
    err = np.abs(np.asarray(pred, dtype=float) - np.asarray(y, dtype=float))
    return {
        "n": int(len(y)),
        "mae_min": round(float(err.mean()), 2),
        "rmse_min": round(float(np.sqrt((err ** 2).mean())), 2),
        "r2": round(float(r2_score(y, pred)), 4),
        "within_5min": round(float((err <= 5).mean()), 4),
        "within_10min": round(float((err <= 10).mean()), 4),
        "within_15min": round(float((err <= 15).mean()), 4),
    }


def fmt_row(name, m):
    return (f"  {name:34s} MAE={m['mae_min']:7.2f}  RMSE={m['rmse_min']:7.2f}  "
            f"R²={m['r2']:+.3f}  ±5min={m['within_5min']:.1%}  ±10min={m['within_10min']:.1%}  "
            f"±15min={m['within_15min']:.1%}")


def main():
    ap = argparse.ArgumentParser(description="回归(延误分钟)预测质量评估——验证集")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--data", default=str(config.CLEAN_CSV))
    args = ap.parse_args()

    run_dir, metas = eval_mod.load_run(args.run_id)
    hist = eval_mod.load_hist_stats(run_dir)
    print(f"评估存档: {run_dir.name}（验证集 2022）\n")

    df = make_targets(pd.read_csv(args.data, low_memory=False))
    year = year_series(df)
    train_df = df[year.isin(config.TRAIN_YEARS)]
    val_df = df[year.isin(config.VAL_YEARS)].copy()

    report = {"run_id": run_dir.name, "evaluated_at": datetime.now().isoformat(timespec="seconds"),
              "eval_on": "validation_2022", "tasks": {}}

    for seg, seg_col, cls_task, reg_task in [
        ("DEP", "DEP_DELAY", "cls_dep_delay", "reg_dep_delay"),
        ("ARR", "ARR_DELAY", "cls_arr_delay", "reg_arr_delay"),
    ]:
        seg_report = {}

        # ---------- A. 延误航班分钟数回归 ----------
        m_reg = task_mask(val_df, reg_task)
        sub_reg = val_df[m_reg]
        y_minutes = sub_reg[seg_col].astype(float).to_numpy()
        x_reg = build_features(sub_reg, feature_set="v1")

        # 基线只用训练集统计
        tr_minutes = train_df[task_mask(train_df, reg_task)][seg_col].astype(float)
        base_mean = float(tr_minutes.mean())
        base_median = float(tr_minutes.median())

        print(f"=== {seg} 延误分钟预测（仅实际延误>0 的航班, n={len(y_minutes):,}）===")
        print(f"  训练集延误分布: 均值={base_mean:.1f}min 中位数={base_median:.1f}min")
        seg_report["delayed_only"] = {
            "baseline_train_mean": reg_metrics(y_minutes, np.full_like(y_minutes, base_mean)),
            "baseline_train_median": reg_metrics(y_minutes, np.full_like(y_minutes, base_median)),
            "models": {},
        }
        print(fmt_row("基线:恒定预测训练均值", seg_report["delayed_only"]["baseline_train_mean"]))
        print(fmt_row("基线:恒定预测训练中位数", seg_report["delayed_only"]["baseline_train_median"]))

        reg_preds = {}
        for algo in ["catboost", "lightgbm"]:
            reg_meta = next(mm for mm in metas if mm["task"] == reg_task and mm["algo"] == algo)
            pred_min = np.asarray(eval_mod.predict(eval_mod.load_model(reg_meta), reg_meta,
                                                   x_reg), dtype=float)
            reg_preds[algo] = pred_min
            m = reg_metrics(y_minutes, pred_min)
            seg_report["delayed_only"]["models"][algo] = m
            print(fmt_row(f"模型:{algo}", m))

        # ---------- B. 两段式合成期望延误（全体航班） ----------
        m_cls = task_mask(val_df, cls_task)
        sub_cls = val_df[m_cls]
        y_all = sub_cls[seg_col].astype(float).clip(lower=0).to_numpy()  # 未延误=0分钟
        x_cls = build_features(sub_cls, feature_set="v1")

        tr_all = train_df[task_mask(train_df, cls_task)][seg_col].astype(float).clip(lower=0)
        base_all_mean = float(tr_all.mean())

        print(f"\n=== {seg} 两段式期望延误（全体航班, n={len(y_all):,}, 目标=max(延误,0)）===")
        seg_report["two_stage_all_flights"] = {
            "baseline_zero": reg_metrics(y_all, np.zeros_like(y_all)),
            "baseline_train_mean": reg_metrics(y_all, np.full_like(y_all, base_all_mean)),
            "models": {},
        }
        print(fmt_row("基线:全部预测0分钟", seg_report["two_stage_all_flights"]["baseline_zero"]))
        print(fmt_row("基线:恒定预测训练均值", seg_report["two_stage_all_flights"]["baseline_train_mean"]))

        for algo in ["catboost", "lightgbm"]:
            cls_meta = next(mm for mm in metas if mm["task"] == cls_task and mm["algo"] == algo)
            proba = np.asarray(eval_mod.predict(eval_mod.load_model(cls_meta), cls_meta,
                                                x_cls), dtype=float)
            # 修正(2026-10-09)：回归预测必须覆盖【全体】航班——部署时不知道谁会延误。
            # 历史版本按"真实是否延误"选择性置0（评估口径泄漏），两段式指标偏乐观，已废弃。
            reg_full = np.asarray(eval_mod.predict(eval_mod.load_model(reg_meta), reg_meta,
                                                   x_cls), dtype=float)
            expected = proba * reg_full
            m2 = reg_metrics(y_all, expected)
            seg_report["two_stage_all_flights"]["models"][algo] = m2
            print(fmt_row(f"两段式:{algo}", m2))

        report["tasks"][seg] = seg_report
        print()

    out = run_dir / "regression_eval_val2022.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
