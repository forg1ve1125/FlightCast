# -*- coding: utf-8 -*-
"""
步骤 4：用存档模型对新航班做预测（演示给前端对接）。

前端的输入字段 = 模型特征所需的原始字段（会议纪要：前端输入必须与训练参数匹配）：
    FL_DATE       航班日期           如 2024-07-04
    AIRLINE       航空公司全名       如 Delta Air Lines Inc.
    ORIGIN        起飞机场代码       如 ATL
    DEST          到达机场代码       如 DEN
    CRS_DEP_TIME  计划出发时刻 hhmm  如 1830
    CRS_ARR_TIME  计划到达时刻 hhmm  如 2015
    DISTANCE      里程(英里)         如 1199

输出（两段式，对应会议纪要示例"70%的概率延误X分钟"）：
    - 出发延误概率 / 若延误预计延误分钟 / 期望延误分钟
    - 到达延误概率 / 若延误预计延误分钟 / 期望延误分钟

用法：
    python 04_predict.py --demo                                  # 内置示例
    python 04_predict.py --fl-date 2024-07-04 --airline "Delta Air Lines Inc." \
        --origin ATL --dest DEN --crs-dep-time 1830 --crs-arr-time 2015 --distance 1199
"""
import argparse
import json

import numpy as np
import pandas as pd

import config
from features import build_features

from importlib import import_module

eval_mod = import_module("03_evaluate_models")


def load_bucket_run():
    """加载最近的延误区间多分类存档（08_train_bucket_models.py 产出）。"""
    pointer = config.MODELS_DIR / "bucket_latest.json"
    if not pointer.exists():
        return None, None, None
    with open(pointer, encoding="utf-8") as f:
        run_id = json.load(f)["run_id"]
    run_dir = config.MODELS_DIR / run_id
    if not run_dir.exists():
        return None, None, None
    metas = []
    for m in sorted(run_dir.glob("mcls_*__metadata.json")):
        with open(m, encoding="utf-8") as f:
            metas.append(json.load(f))
    stats_path = run_dir / "bucket_stats.json"
    stats = json.load(open(stats_path, encoding="utf-8")) if stats_path.exists() else {}
    return run_dir, metas, stats


def bucket_scenarios(meta, stats, x):
    """单航班的多情景概率：每档概率 + 换算'约X分钟' + 概率加权期望延误。"""
    model = eval_mod.load_model(meta)
    if meta["algo"] == "lightgbm":
        from features import align_categories
        xi = align_categories(x, meta.get("categories") or {})
        proba = np.asarray(model.predict(xi, num_iteration=model.best_iteration))[0]
    else:
        proba = np.asarray(model.predict_proba(x))[0]

    seg = "dep" if "_dep" in meta["task"] else "arr"
    seg_stats = stats.get(seg, [])
    scenarios = []
    for k, label in enumerate(config.BUCKET_LABELS):
        item = {"scenario": label, "probability": round(float(proba[k]), 4)}
        if seg_stats and k < len(seg_stats):
            med = seg_stats[k].get("median_minutes")
            if k == 0:
                item["typical_minutes"] = None  # 不延误档：提前/准点，无典型分钟数
            else:
                item["typical_minutes"] = med
        scenarios.append(item)
    scenarios.sort(key=lambda s: -s["probability"])

    # 各档概率×该档典型分钟数，得到区间版的期望延误
    expected = sum(s["probability"] * (s.get("typical_minutes") or 0) for s in scenarios)
    summary_parts = []
    for s in scenarios:
        if s["scenario"] == "不延误":
            summary_parts.append(f"{s['probability']:.0%}概率不延误(准点或提前)")
        else:
            summary_parts.append(f"{s['probability']:.0%}概率{s['scenario']}"
                                 + (f"(约{s['typical_minutes']:.0f}分钟)"
                                    if s.get("typical_minutes") is not None else ""))
    return {"scenarios": scenarios,
            "expected_delay_min_from_buckets": round(float(expected), 1),
            "summary": "；".join(summary_parts)}


def predict_flight(model_dir, flight: dict, models_dir=None, with_buckets=True):
    run_dir, metas = eval_mod.load_run(model_dir, models_dir=models_dir)
    hist_stats = eval_mod.load_hist_stats(run_dir)
    row = pd.DataFrame([flight])
    result = {"input": flight, "delay_threshold_min": config.DELAY_THRESHOLD_MIN, "predictions": {}}

    for meta in metas:
        model = eval_mod.load_model(meta)
        x = eval_mod.build_x(meta, row, hist_stats)  # 特征集版本按 metadata 自动判断
        pred = eval_mod.predict(model, meta, x)

        seg = "DEP" if "dep" in meta["task"] else "ARR"
        kind = "cls" if meta["task"].startswith("cls") else "reg"
        key = f"{'dep' if seg == 'DEP' else 'arr'}_delay"
        bucket = result["predictions"].setdefault(key, {})
        if kind == "cls":
            bucket["prob_delay"] = round(float(np.asarray(pred).ravel()[0]), 4)
        else:
            bucket["expected_minutes_if_delayed"] = round(float(np.asarray(pred).ravel()[0]), 1)

    # 组装两段式结论
    for key, label in [("dep_delay", "出发"), ("arr_delay", "到达")]:
        b = result["predictions"].get(key, {})
        p = b.get("prob_delay")
        mins = b.get("expected_minutes_if_delayed")
        if p is not None and mins is not None:
            b["expected_minutes_overall"] = round(p * mins, 1)
            b["summary"] = (f"{label}延误概率 {p*100:.1f}%；"
                            f"若延误，预计延误约 {mins:.0f} 分钟；"
                            f"期望延误 {p*mins:.0f} 分钟")

    # 多情景概率（区间多分类模型，若已训练）
    if with_buckets:
        b_dir, b_metas, b_stats = load_bucket_run()
        if b_metas:
            result["scenarios"] = {}
            for meta in b_metas:
                if meta["algo"] != "catboost":
                    continue  # 每个seg取一种算法（优先catboost）
                seg = "dep" if "_dep" in meta["task"] else "arr"
                key = f"{seg}_delay"
                if key in result["scenarios"]:
                    continue
                x = eval_mod.build_x(meta, row, None)
                result["scenarios"][key] = bucket_scenarios(meta, b_stats, x)
    return result


def main():
    ap = argparse.ArgumentParser(description="新航班延误预测（前端对接示例）")
    ap.add_argument("--run-id", default=None, help="模型存档名（默认最近一次）")
    ap.add_argument("--demo", action="store_true", help="使用内置示例航班")
    ap.add_argument("--fl-date", dest="fl_date")
    ap.add_argument("--airline")
    ap.add_argument("--origin")
    ap.add_argument("--dest")
    ap.add_argument("--crs-dep-time", dest="crs_dep_time", type=int)
    ap.add_argument("--crs-arr-time", dest="crs_arr_time", type=int)
    ap.add_argument("--distance", type=float)
    args = ap.parse_args()

    if args.demo:
        flight = {
            "FL_DATE": "2024-07-04",
            "AIRLINE": "Delta Air Lines Inc.",
            "ORIGIN": "ATL",
            "DEST": "DEN",
            "CRS_DEP_TIME": 1830,
            "CRS_ARR_TIME": 2015,
            "DISTANCE": 1199.0,
        }
    else:
        required = ["fl_date", "airline", "origin", "dest", "crs_dep_time", "crs_arr_time", "distance"]
        missing = [r for r in required if getattr(args, r) is None]
        if missing:
            ap.error(f"缺少参数: {missing}（或使用 --demo）")
        flight = {
            "FL_DATE": args.fl_date,
            "AIRLINE": args.airline,
            "ORIGIN": args.origin,
            "DEST": args.dest,
            "CRS_DEP_TIME": args.crs_dep_time,
            "CRS_ARR_TIME": args.crs_arr_time,
            "DISTANCE": args.distance,
        }

    result = predict_flight(args.run_id, flight)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
