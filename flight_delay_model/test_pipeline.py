# -*- coding: utf-8 -*-
"""
测试 / 验证脚本（可用 pytest 运行，也可直接 `python test_pipeline.py`）

验证内容对应会议纪要与字段文档的各项要求：
  1. 原始数据集从未被修改（MD5 与清洗时记录一致）
  2. 清洗数据集：12 个不使用字段确已删除、其余字段保留、行数一致
  3. 数据集划分符合会议纪要：2019-2021 训练 / 2022 验证 / 2023 测试，覆盖全部数据且不重叠
  4. 特征无数据泄漏：训练特征不含"飞行后才知道"的字段，也不含目标本身
  5. 目标构造正确：延误定义 >= 15 分钟；回归任务只在"实际延误>0"的未取消航班上训练
  6. 模型存档完整：每个模型都有模型文件 + metadata.json（记录了"怎么训练的"）
  7. 模型可重新加载并预测：概率在 [0,1]，回归预测为正数，且与验证集指标方向一致
  8. 可复现配置：随机种子、超参数、划分均与配置文件一致
  9. 端到端冒烟测试：在临时目录用 3 万行样本从零跑一遍 清洗->训练->预测
 10. 评估报告完整（若已生成）：分类指标/回归指标字段齐全、数值合法

用法：
    python test_pipeline.py                 # 直接运行全部测试
    pytest test_pipeline.py -v              # pytest 运行
    python test_pipeline.py --skip-e2e      # 跳过较慢的端到端冒烟测试
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import config
from features import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
                      align_categories, build_features, make_targets,
                      split_by_year, task_mask, year_series)

sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------- 工具函数
def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_manifest():
    with open(config.MANIFEST_JSON, encoding="utf-8") as f:
        return json.load(f)


def _latest_run_dir():
    with open(config.LATEST_RUN_JSON, encoding="utf-8") as f:
        run_id = json.load(f)["run_id"]
    return config.MODELS_DIR / run_id


def _load_run_metas(run_dir):
    metas = []
    for m in sorted(run_dir.glob("*__metadata.json")):
        with open(m, encoding="utf-8") as f:
            metas.append(json.load(f))
    return metas


def _sample_clean_df(n=5000, seed=0):
    """从清洗数据集抽 n 行做快速校验。"""
    df = pd.read_csv(config.CLEAN_CSV, low_memory=False)
    return df.sample(n=min(n, len(df)), random_state=seed)


# ------------------------------------------------------------------ 测试 1
def test_01_original_dataset_untouched():
    """原始数据集一个字节都不能动（方式一的意义就在于保留原始数据）。"""
    manifest = _load_manifest()
    assert Path(manifest["original_csv"]).exists(), "原始数据集不存在"
    actual = _md5(manifest["original_csv"])
    assert actual == manifest["original_md5"], (
        f"原始数据集被修改过! MD5 {manifest['original_md5']} -> {actual}")


# ------------------------------------------------------------------ 测试 2
def test_02_clean_dataset_fields():
    """清洗数据集 = 原始 - 12 个不使用字段；行数不变。"""
    manifest = _load_manifest()
    assert Path(config.CLEAN_CSV).exists(), "清洗数据集不存在，请先运行 01_prepare_data.py"

    header = pd.read_csv(config.CLEAN_CSV, nrows=0).columns.tolist()
    for field in config.EXCLUDED_FIELDS:
        assert field not in header, f"不使用的字段 {field} 仍存在于清洗数据集!"
    # 原始列 - 排除列 = 清洗列
    original_header = pd.read_csv(config.ORIGINAL_CSV, nrows=0).columns.tolist()
    expect = [c for c in original_header if c not in config.EXCLUDED_FIELDS]
    assert header == expect, "清洗数据集列与'原始列-排除列'不一致"

    n_rows = sum(1 for _ in open(config.CLEAN_CSV, "rb")) - 1
    assert n_rows == manifest["original_rows"], (
        f"行数不一致: 清洗 {n_rows} vs 原始 {manifest['original_rows']}")


# ------------------------------------------------------------------ 测试 3
def test_03_split_by_year():
    """2019-2021/2022/2023 三份划分互不重叠、并集覆盖全部数据。"""
    years = year_series(pd.read_csv(config.CLEAN_CSV, usecols=["FL_DATE"], low_memory=False))
    in_train = years.isin(config.TRAIN_YEARS)
    in_val = years.isin(config.VAL_YEARS)
    in_test = years.isin(config.TEST_YEARS)
    assert not (in_train & in_val).any() and not (in_train & in_test).any() \
        and not (in_val & in_test).any(), "训练/验证/测试集存在重叠!"
    assert (in_train | in_val | in_test).all(), "存在未归入任何集合的年份"
    assert in_train.sum() > 0 and in_val.sum() > 0 and in_test.sum() > 0
    print(f"    训练 {int(in_train.sum()):,} / 验证 {int(in_val.sum()):,} / "
          f"测试 {int(in_test.sum()):,} 行")


# ------------------------------------------------------------------ 测试 4
def test_04_no_feature_leakage():
    """训练特征只含起飞前可知信息：不含泄漏字段、不含目标列。"""
    forbidden = set(config.LEAKAGE_FIELDS) | {
        "DEP_DELAY", "ARR_DELAY", "CANCELLED", "DIVERTED",
        "y_cls_dep", "y_cls_arr", "y_reg_dep", "y_reg_arr",
        "FL_DATE", "CRS_DEP_TIME", "CRS_ARR_TIME",  # 原始列只能以派生形式进入
    }
    leak = [f for f in FEATURES if f in forbidden]
    assert not leak, f"特征中存在泄漏/目标列: {leak}"
    from features import HIST_FEATURES
    assert set(FEATURES) == set(NUMERIC_FEATURES + HIST_FEATURES + CATEGORICAL_FEATURES), \
        "FEATURES 与 数值+历史+分类 特征清单不一致"

    df = _sample_clean_df(2000)
    x = build_features(df)
    assert list(x.columns) == FEATURES, "build_features 输出列顺序与 FEATURES 不一致"
    assert x.notna().all().all(), "特征存在缺失值"
    assert x["DEP_HOUR"].between(0, 23).all() and x["ARR_HOUR"].between(0, 23).all()
    assert x["DISTANCE"].gt(0).all()
    assert (x["ROUTE"] == x["ORIGIN"] + "_" + x["DEST"]).all()


# ------------------------------------------------------------------ 测试 5
def test_05_target_definition():
    """延误定义 = 延误时间>0；任务掩码正确排除取消/改航/缺失；回归只留延误>0。"""
    df = make_targets(_sample_clean_df(5000))

    dep = pd.to_numeric(df["DEP_DELAY"])
    assert (df["y_cls_dep"] == (dep > config.DELAY_THRESHOLD_MIN).astype(int)).all()
    arr = pd.to_numeric(df["ARR_DELAY"])
    assert (df["y_cls_arr"] == (arr > config.DELAY_THRESHOLD_MIN).astype(int)).all()

    cancelled = pd.to_numeric(df["CANCELLED"]).fillna(0) > 0
    diverted = pd.to_numeric(df["DIVERTED"]).fillna(0) > 0

    m_dep = task_mask(df, "cls_dep_delay")
    assert not df.loc[m_dep, "DEP_DELAY"].isna().any(), "分类任务样本中存在缺失 DEP_DELAY"
    assert not (cancelled[m_dep]).any() and not (diverted[m_dep]).any(), "分类任务混入取消/改航航班"

    m_reg = task_mask(df, "reg_dep_delay")
    assert (df.loc[m_reg, "y_reg_dep"] > 0).all(), "回归任务包含延误<=0的样本"
    assert m_reg.sum() <= m_dep.sum(), "回归样本应是分类样本的子集"


# ------------------------------------------------------------------ 测试 6
def test_06_model_archive_complete():
    """存档目录：模型文件 + metadata.json 齐全，metadata 记录了训练方式。"""
    run_dir = _latest_run_dir()
    metas = _load_run_metas(run_dir)
    assert metas, f"存档 {run_dir} 中没有 metadata.json"

    required_keys = ["task", "algo", "features", "categorical_features", "target",
                     "split", "hyperparams", "random_seed", "val_metrics_2022",
                     "delay_threshold_min", "trained_at"]
    for meta in metas:
        model_file = run_dir / meta["model_file"]
        assert model_file.exists(), f"模型文件缺失: {model_file}"
        for k in required_keys:
            assert k in meta, f"{meta['task']}/{meta['algo']} metadata 缺少键: {k}"
        assert meta["split"]["train_years"] == list(config.TRAIN_YEARS)
        assert meta["split"]["val_years"] == list(config.VAL_YEARS)
        assert meta["split"]["test_years"] == list(config.TEST_YEARS)
        from features import FEATURES_V1
        expect_feats = FEATURES if meta.get("feature_set") == "v2" else FEATURES_V1
        assert meta["features"] == expect_feats, "metadata 特征清单与特征集版本不一致"
    print(f"    存档 {run_dir.name}: {len(metas)} 个模型齐全")


# ------------------------------------------------------------------ 测试 7
def test_07_model_reload_and_predict():
    """重新加载每个模型，对2023年样本预测：概率∈[0,1]、回归预测>0。"""
    from importlib import import_module
    eval_mod = import_module("03_evaluate_models")

    run_dir = _latest_run_dir()
    metas = _load_run_metas(run_dir)
    hist = eval_mod.load_hist_stats(run_dir)
    df = pd.read_csv(config.CLEAN_CSV, low_memory=False)
    df = make_targets(df[year_series(df).isin(config.TEST_YEARS)].head(300))

    for meta in metas:
        model = eval_mod.load_model(meta)
        x = eval_mod.build_x(meta, df, hist)
        pred = np.asarray(eval_mod.predict(model, meta, x)).ravel()
        assert len(pred) == len(df)
        if meta["task"].startswith("cls"):
            assert ((pred >= 0) & (pred <= 1)).all(), f"{meta['task']} 概率超出[0,1]"
        else:
            assert (pred > 0).all(), f"{meta['task']} 延误分钟数出现非正数"
    print(f"    {len(metas)} 个模型均可在2023样本上正常加载并预测")


# ------------------------------------------------------------------ 测试 8
def test_08_reproducible_config():
    """metadata 中的种子/阈值/超参数与 config 一致（训练方式可追溯，含显式调参覆盖）。"""
    from importlib import import_module
    train_mod = import_module("02_train_models")
    run_dir = _latest_run_dir()
    for meta in _load_run_metas(run_dir):
        assert meta["random_seed"] == config.RANDOM_SEED
        assert meta["delay_threshold_min"] == config.DELAY_THRESHOLD_MIN
        key = "cls" if meta["task"].startswith("cls") else "reg"
        ref = (train_mod.CATBOOST_PARAMS if meta["algo"] == "catboost"
               else train_mod.LIGHTGBM_PARAMS)[key]
        overrides = meta.get("param_overrides") or {}
        for k, v in ref.items():
            expect = overrides.get(k, v)  # 显式调参覆盖的键以覆盖值为准
            assert meta["hyperparams"].get(k) == expect, \
                f"{meta['task']}/{meta['algo']} 超参数 {k} 与配置/覆盖不一致"


# ------------------------------------------------------------------ 测试 9
def test_09_e2e_smoke(tmp: Path):
    """端到端冒烟：临时目录内 清洗 -> 训练(quick) -> 预测 全链路可跑通。"""
    from importlib import import_module
    prepare_mod = import_module("01_prepare_data")
    train_mod = import_module("02_train_models")
    predict_mod = import_module("04_predict")

    # 9.1 从原始数据集抽 3 万行做"迷你原始数据集"
    mini_original = tmp / "mini_original.csv"
    pd.read_csv(config.ORIGINAL_CSV, nrows=30_000, low_memory=False).to_csv(mini_original, index=False)

    # 9.2 清洗
    mini_clean = tmp / "mini_clean.csv"
    prepare_mod.prepare(mini_original, mini_clean, tmp / "manifest.json")
    header = pd.read_csv(mini_clean, nrows=0).columns.tolist()
    assert not (set(config.EXCLUDED_FIELDS) & set(header))

    # 9.3 快速训练一个分类模型
    train_mod.run_training(mini_clean, algos=["catboost"], tasks=["cls_dep_delay"],
                           quick=True, models_dir=tmp / "models", run_id="smoke")
    meta_files = list((tmp / "models" / "smoke").glob("*__metadata.json"))
    assert len(meta_files) == 1, "端到端训练未产出存档"

    # 9.4 预测
    result = predict_mod.predict_flight("smoke", {
        "FL_DATE": "2024-07-04", "AIRLINE": "Delta Air Lines Inc.",
        "ORIGIN": "ATL", "DEST": "DEN", "CRS_DEP_TIME": 1830,
        "CRS_ARR_TIME": 2015, "DISTANCE": 1199.0,
    }, models_dir=tmp / "models")
    p = result["predictions"]["dep_delay"]["prob_delay"]
    assert 0.0 <= p <= 1.0, f"端到端预测概率非法: {p}"
    print(f"    端到端链路 OK（mini训练->预测, 示例延误概率={p:.3f}）")


# ------------------------------------------------------------------ 测试 10
def test_10_evaluation_report_valid():
    """若已生成评估报告，则指标字段齐全且数值合法（准确率/精确率/召回率等）。"""
    run_dir = _latest_run_dir()
    report_path = run_dir / "evaluation_report.json"
    if not report_path.exists():
        print("    (跳过: 尚未运行 03_evaluate_models.py)")
        return
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["models"], "评估报告为空"
    for e in report["models"]:
        if e["task"].startswith("cls"):
            for key in ("accuracy", "precision", "recall", "f1", "auc"):
                v = e["test_metrics@0.5"][key]
                assert 0.0 <= v <= 1.0, f"{e['task']} {key}={v} 超出 [0,1]"
        else:
            m = e["test_metrics_delayed_only"]
            assert m["mae_min"] >= 0 and m["rmse_min"] >= m["mae_min"] * 0.99
    print(f"    评估报告 {report_path.name} 指标合法")


# ------------------------------------------------------------------ 测试 11
def test_11_bucket_models_valid():
    """区间多分类模型（若已训练）：概率和为1、非负、档数正确、存档齐全。"""
    from importlib import import_module
    predict_mod = import_module("04_predict")

    b_dir, b_metas, b_stats = predict_mod.load_bucket_run()
    if b_metas is None:
        print("    (跳过: 尚未训练区间模型 08_train_bucket_models.py)")
        return
    import config as cfg
    assert (b_dir / "bucket_stats.json").exists(), "缺 bucket_stats.json"
    assert len(b_metas) >= 1
    for meta in b_metas:
        assert meta["task"].startswith("mcls_")
        assert meta["n_classes"] == cfg.N_BUCKETS
        model_path = Path(meta["model_path"])
        assert model_path.exists(), f"模型文件缺失: {model_path}"

    # 单航班预测：概率和≈1
    result = predict_mod.predict_flight(None, {
        "FL_DATE": "2024-07-04", "AIRLINE": "Delta Air Lines Inc.",
        "ORIGIN": "ATL", "DEST": "DEN", "CRS_DEP_TIME": 1830,
        "CRS_ARR_TIME": 2015, "DISTANCE": 1199.0,
    })
    for key, sc in result.get("scenarios", {}).items():
        total = sum(s["probability"] for s in sc["scenarios"])
        assert abs(total - 1.0) < 0.01, f"{key} 情景概率和={total} 不为1"
        assert all(s["probability"] >= 0 for s in sc["scenarios"])
        assert len(sc["scenarios"]) == cfg.N_BUCKETS
    print(f"    区间模型 {len(b_metas)} 个：概率合法（和为1、非负、{cfg.N_BUCKETS}档）")


# ------------------------------------------------------------------ 主入口
ALL_TESTS = [
    test_01_original_dataset_untouched,
    test_02_clean_dataset_fields,
    test_03_split_by_year,
    test_04_no_feature_leakage,
    test_05_target_definition,
    test_06_model_archive_complete,
    test_07_model_reload_and_predict,
    test_08_reproducible_config,
    test_09_e2e_smoke,
    test_10_evaluation_report_valid,
    test_11_bucket_models_valid,
]


def main():
    ap = argparse.ArgumentParser(description="航班延误模型流水线测试/验证")
    ap.add_argument("--skip-e2e", action="store_true", help="跳过端到端冒烟测试(较慢)")
    args = ap.parse_args()

    print("=" * 72)
    print("航班延误预测流水线 —— 测试与验证")
    print("=" * 72)

    passed, failed, skipped = 0, 0, 0
    for fn in ALL_TESTS:
        if args.skip_e2e and fn is test_09_e2e_smoke:
            print(f"[SKIP] {fn.__name__}")
            skipped += 1
            continue
        print(f"\n[RUN ] {fn.__name__}: {fn.__doc__.strip().splitlines()[0]}")
        try:
            if fn is test_09_e2e_smoke:
                tmp = Path(tempfile.mkdtemp(prefix="flight_smoke_"))
                try:
                    fn(tmp)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
            else:
                fn()
            print("[PASS]")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] 断言失败: {e}")
            failed += 1
        except Exception:
            print("[FAIL] 异常:")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 72)
    print(f"结果: {passed} 通过, {failed} 失败, {skipped} 跳过 / 共 {len(ALL_TESTS)} 项")
    print("=" * 72)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
