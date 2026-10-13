# 累积二分类区间模型 — 训练与对照报告

- 运行ID：`20261007_152943_cumbin_quick`（完整训练）
- 生成时间：2026-10-07T15:33:16
- 数据：`flights_sample_3m_clean.csv`，训练 [2019, 2020, 2021] / 验证 [2022]，**2023 测试集未触碰**；特征 v1；CatBoost Logloss（无类别加权，概率绝对值可信）
- 对照对象：08 存档 MultiClass `20261006_124341_buckets`（catboost）

## 方法

4 个链式二分类器 P(>0) / P(>15) / P(>30) / P(>60)，单调投影后逐档相减：

```
bucket0 = 1 - P(>0)      bucket1 = P(>0)  - P(>15)
bucket2 = P(>15) - P(>30)  bucket3 = P(>30) - P(>60)   bucket4 = P(>60)
```

## 出发延误（验证集 n=667,762）

| 指标 | 累积二分类 | MultiClass | 胜者 |
|---|---|---|---|
| 区间准确率 | 0.6086 | 0.609 | ✅ MultiClass |
| macro F1 | 0.1514 | 0.1532 | ✅ MultiClass |
| 逐档校准平均偏差(pp) | 2.56 | 3.39 | ✅ 累积 |

各阈值 AUC：>0 分钟 **0.6812** / >15 分钟 **0.6679** / >30 分钟 **0.67** / >60 分钟 **0.6593**

单调违例（投影前）：2.09% → 投影后恒为 0

| 档位 | 累积:平均预测概率 | MultiClass:平均预测概率 | 实际频率 |
|---|---|---|---|
| bucket0 不延误 | 0.664 | 0.693 | 0.609 |
| bucket1 延误1-15分钟 | 0.157 | 0.153 | 0.186 |
| bucket2 延误16-30分钟 | 0.081 | 0.054 | 0.073 |
| bucket3 延误31-60分钟 | 0.046 | 0.046 | 0.064 |
| bucket4 延误60分钟以上 | 0.052 | 0.053 | 0.069 |

| 档位 | 累积 F1 | MultiClass F1 |
|---|---|---|
| bucket0 不延误 | 0.7567 | 0.7575 |
| bucket1 延误1-15分钟 | 0.0001 | 0.0083 |
| bucket2 延误16-30分钟 | 0.0 | 0.0 |
| bucket3 延误31-60分钟 | 0.0 | 0.0 |
| bucket4 延误60分钟以上 | 0.0 | 0.0001 |

**裁决**：三项准则（准确率/校准/macro F1）累积模型胜出 **1/3** → 保留现状(MultiClass)（标准：≥2 项才建议采用）

图像：`charts/dep_reliability.png`、`charts/dep_confusion.png`、`charts/dep_calibration.png`、`charts/dep_f1.png`

## 到达延误（验证集 n=667,761）

| 指标 | 累积二分类 | MultiClass | 胜者 |
|---|---|---|---|
| 区间准确率 | 0.6267 | 0.6267 | ✅ MultiClass |
| macro F1 | 0.1541 | 0.1541 | ✅ MultiClass |
| 逐档校准平均偏差(pp) | 1.29 | 1.71 | ✅ 累积 |

各阈值 AUC：>0 分钟 **0.6169** / >15 分钟 **0.6427** / >30 分钟 **0.6477** / >60 分钟 **0.6559**

单调违例（投影前）：0.31% → 投影后恒为 0

| 档位 | 累积:平均预测概率 | MultiClass:平均预测概率 | 实际频率 |
|---|---|---|---|
| bucket0 不延误 | 0.655 | 0.669 | 0.627 |
| bucket1 延误1-15分钟 | 0.156 | 0.153 | 0.170 |
| bucket2 延误16-30分钟 | 0.071 | 0.064 | 0.072 |
| bucket3 延误31-60分钟 | 0.067 | 0.054 | 0.062 |
| bucket4 延误60分钟以上 | 0.051 | 0.059 | 0.069 |

| 档位 | 累积 F1 | MultiClass F1 |
|---|---|---|
| bucket0 不延误 | 0.7705 | 0.7705 |
| bucket1 延误1-15分钟 | 0.0 | 0.0 |
| bucket2 延误16-30分钟 | 0.0 | 0.0 |
| bucket3 延误31-60分钟 | 0.0 | 0.0 |
| bucket4 延误60分钟以上 | 0.0 | 0.0 |

**裁决**：三项准则（准确率/校准/macro F1）累积模型胜出 **1/3** → 保留现状(MultiClass)（标准：≥2 项才建议采用）

图像：`charts/arr_reliability.png`、`charts/arr_confusion.png`、`charts/arr_calibration.png`、`charts/arr_f1.png`

## 产物与后续

- 模型：8 个 `cls_cumbin_{seg}_gt{edge}__catboost.cbm` + 同名 metadata
- 指标明细：`cumulative_val_metrics.json`；指针：`models/cumulative_latest.json`
- **本脚本没有改动生产配置**（`bucket_latest.json` 未动，网页仍用 08 的模型）。
  若裁决为'建议采用'且你们确认切换：把 `models/bucket_latest.json` 中的 `run_id` 手动改为`20261007_152943_cumbin_quick` 之前，还需要给 04_predict.py 增加累积模型的加载分支（另行任务）。
