# -*- coding: utf-8 -*-
"""
可行性分析：能否让准确率/精确率/召回率同时达到80%？
（只用 2022 验证集，不触碰测试集）

结论（详细数字见输出）：
在延误率约 18-21% 的类别分布下，P=R=80% 数学上要求模型在 FPR≈5% 处达到 TPR=80%，
对应 AUC≈0.95+；而本数据"起飞前可知特征"的 AUC 天花板约 0.68-0.70，
验证集上任何阈值都无法使精确率达到 80%。因此 80/80/80 不可行，
这不是调参问题，而是特征信息量（信号上限）决定的。

用法：python 05_feasibility_analysis.py
"""
import numpy as np
import pandas as pd

import config
from features import build_features, make_targets, task_mask, year_series
from importlib import import_module

eval_mod = import_module("03_evaluate_models")


def main():
    df = make_targets(pd.read_csv(config.CLEAN_CSV, low_memory=False))
    val = df[year_series(df).isin(config.VAL_YEARS)].copy()
    mask = task_mask(val, "cls_dep_delay")
    y = val[mask]["y_cls_dep"].astype(int).to_numpy()
    x = build_features(val[mask], feature_set="v1")
    pi = y.mean()
    print(f"验证集样本 {len(y):,}，延误率(正例率) = {pi:.4f}\n")

    run_dir, metas = eval_mod.load_run()
    for meta in [m for m in metas if m["task"] == "cls_dep_delay"]:
        model = eval_mod.load_model(meta)
        proba = np.asarray(eval_mod.predict(model, meta, x))

        from sklearn.metrics import (precision_recall_curve, roc_auc_score,
                                     roc_curve)
        auc = roc_auc_score(y, proba)
        fpr, tpr, _ = roc_curve(y, proba)
        prec, rec, thr = precision_recall_curve(y, proba)

        # P=R=80% 所需的 ROC 工作点
        req_fpr = 0.2 * pi / (1 - pi)
        tpr_at_req = float(np.interp(req_fpr, fpr, tpr))

        order = np.argsort(rec)
        p_at_r80 = float(np.interp(0.80, rec[order], prec[order]))
        p_at_r60 = float(np.interp(0.60, rec[order], prec[order]))

        f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-9)
        k = int(np.argmax(f1s))

        print(f"模型: {meta['algo']}  (验证集 2022)")
        print(f"  AUC = {auc:.4f}")
        print(f"  若要 P=R=80%：需要 ROC 点 FPR={req_fpr:.3f} 处 TPR=0.80，"
              f"实际该点 TPR 仅 {tpr_at_req:.3f}（对应 AUC≈0.95+，远超特征信息上限）")
        print(f"  recall=80% 时 precision 仅 {p_at_r80:.3f}")
        print(f"  recall=60% 时 precision 仅 {p_at_r60:.3f}")
        print(f"  最优F1工作点: threshold={thr[k]:.2f} P={prec[k]:.3f} "
              f"R={rec[k]:.3f} F1={f1s[k]:.3f}\n")


if __name__ == "__main__":
    main()
