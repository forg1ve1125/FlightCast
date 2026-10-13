# IS6400 小组项目：航班延误预测模型（CatBoost）

> **最终形态（2026-10-13 定格）**：仅预测**出发延误**，线上由三个 CatBoost 组件组成
> （二分类 / 五档多分类 / 回归），各 39 特征 = 11 计划特征 + 起飞机场天气 14 + 到达机场天气 14。
> 2023 测试集收官成绩：**AUC 0.7038 / 两段式期望延误 MAE 18.88 分钟**。
> 最终采用哪几个模型、如何切换，见下文[「最终采用的模型」](#最终采用的模型线上现役)一节。

按 **9.24 会议纪要** 和 **《训练模型不需要使用的字段》** 文档实现的完整流水线：
数据清洗 → 特征工程 → 模型训练 → 验证评估 → 天气特征 → 2023 收官测试 → 全栈网页部署。

## 我们要做什么（会议纪要 + 后续范围决策）

给定航班信息，预测：
1. **延误的概率**（分类：**延误时间 > 0 分钟**即为"延误"，小组统一口径）
2. **在延误概率的基础上，分别延误多少时间**（回归：延误分钟数）
3. ~~出发延误和到达延误两类~~ → **2026-10-08 导师建议聚焦单一任务后，只预测出发延误**
   （到达模型存档保留但不再训练/上线；两个计划时刻仍作为输入，见字段说明）

最终预测输出形如：*"63% on time · 25% 1-15 min delay (~5 min) · …"*（五档概率分布）
与 *"35.2% chance of any delay; if delayed, about 38 min; expected delay 13.3 min"*（两段式）。

## 文件说明

| 文件 | 作用 |
|---|---|
| `config.py` | 全局配置：路径、12个排除字段、数据划分、延误阈值/五档边界、随机种子 |
| `features.py` | 特征工程（只使用"起飞前可知"的信息，防数据泄漏） |
| `01_prepare_data.py` | **方式一（文档推荐）**：生成去除不使用字段的新数据集，保留原始数据集 |
| `02_train_models.py` | 训练 CatBoost/LightGBM（分类+回归，v1 基线），自动存档每个模型 |
| `02b_tune_on_validation.py` | **只用验证集调参**（特征集/超参候选在2022上一一评估，绝不碰测试集） |
| `03_evaluate_models.py` | 模型评估工具（被后续脚本复用：加载存档/构建特征/预测） |
| `04_predict.py` | 用存档模型预测新航班（演示与前端对接的输入/输出格式） |
| `05_feasibility_analysis.py` | 分析"80/80/80是否可行"（结论：不可行，数学+实证） |
| `06_eval_regression.py` | **延误分钟数预测质量评估**（含勘误：两段式改为全体航班口径） |
| `07_experiments_val.py` | 提升空间对照实验（集成/调参/新特征/XGBoost/单段式/校准，验证集） |
| `08_train_bucket_models.py` | **延误区间多分类**（5档，多情景概率输出，无天气版） |
| `09_train_cumulative.py` | 累积二分类对照实验（4 个链式 P(>k)，结论：1/3 胜，弃用） |
| `10_weather_download.py` | **IEM 天气数据下载**（380 机场逐站 2019–2023，断点续传+机场时区推导） |
| `11_train_weather.py` | 二分类**四臂对照**（v1 / +起飞天气 / +块时长 / +双端天气） |
| `12_train_bucket_weather.py` | 区间模型 + 双端天气（39 特征，双对照：08 版与上代天气版） |
| `13_train_regression_weather.py` | 回归模型 + 双端天气（两段式三组合对照） |
| `14_test_2023.py` | **2023 测试集一次性收官评估**（三档消融×三组件，已有结果拒绝重跑） |
| `test_pipeline.py` | 数据/特征层测试脚本（10项） |
| `server/` | Flask 后端：`app.py`（路由）、`predictor_service.py`（预测编排）、
  `weather_serving.py`（Open-Meteo 双端预报+模型注册表）、`verdict.py`（argmax 英文结论）、
  `auth.py`（注册/登录/历史，独立 app.db）、`build_meta_db.py`（元数据库）、`test_backend.py` |
| `frontend/` | Vue3 前端（国泰风格全英文，`npm run build` 后由 Flask 托管） |
| `weather/` | `tz_map.json`（机场时区）、`airport_coords.json`（坐标）、原始/整合天气数据（不入库） |
| `models/<时间戳_run名>/` | 每次训练的存档：模型文件 + metadata.json + 指标 + 图表 + 报告 |

### 存档命名规范（统一格式）

目录名一律为 `时间戳[_run名][_quick]`（本地时间，无冒号），例如 `20261005_155440_final_gt0`：

```bash
python 02_train_models.py --feature-set v1 --run-name final_gt0   # 自动生成 时间戳_final_gt0
python 02_train_models.py --feature-set v1 --run-id 20261006_101500_myrun  # 也可完整指定
```

## 最终采用的模型（线上现役）

三个组件全部为 CatBoost、39 特征（11 计划 + 起飞机场天气 14 + 到达机场天气 14），
由注册表 `models/wx_serving.json` 控制启用与文件名：

| 组件 | 回答的问题 | 存档 | 模型文件 | 2022验证 / 2023测试 |
|---|---|---|---|---|
| 二分类 | P(出发延误) | `20261009_193645_wx_dep`（臂4） | `cls_dep_delay__v1wx_od__catboost.cbm` | AUC 0.7042 / **0.7038** |
| 五档区间 | 各延误区间的概率分布 | `20261013_122152_buckets_wx_dep` | `mcls_dep_delay__v1wx_od__catboost.cbm` | acc 0.6108 / 0.5991 |
| 回归 | 若延误约几分钟 | `20261013_130457_wx_reg_dep` | `reg_dep_delay__v1wx_od__catboost.cbm` | 两段式 MAE 17.10 / 18.88 |

兜底链（预报取不到→组件失败→整体异常时逐级回退）：
- `models/latest_run.json` → `20261005_155440_final_gt0`（v1 基线 cls+reg，CatBoost/LightGBM）
- `models/bucket_latest.json` → `20261006_124341_buckets`（v1 五档区间）

换装/回滚：只改 `wx_serving.json` 的 run 与 `*_file` 字段后重启 Flask。线上在用的
13 个模型二进制已入 git 仓库（克隆即可跑网页），其余存档二进制不入库、可按脚本重训。

## 快速开始

```bash
# 0. 安装依赖（macOS 上 LightGBM 需要 libomp: brew install libomp）
pip install pandas numpy scikit-learn catboost lightgbm matplotlib flask

# 1. 生成清洗数据集（约1-2分钟，只需运行一次）
python 01_prepare_data.py

# 2. 启动网页（模型二进制已在仓库中；详见 server 部分说明）
python server/build_meta_db.py          # 首次：构建元数据库（约3秒）
cd frontend && npm install && npm run build && cd ..
python server/app.py                    # 打开 http://127.0.0.1:5000

# 3. 复现模型训练（可选；每个脚本支持断点续训，结束自动出总结+图表+报告）
python 02_train_models.py               # v1 基线（约30-60分钟）
python 10_weather_download.py           # 天气数据（约1-2小时，断点续传）
python 11_train_weather.py              # 二分类四臂对照
python 12_train_bucket_weather.py       # 区间模型（双端天气39特征）
python 13_train_regression_weather.py   # 回归模型（双端天气39特征）
python 14_test_2023.py                  # 2023 收官（已有结果会拒绝重跑）

# 4. 工具
python 04_predict.py --demo             # 命令行预测示例
python 06_eval_regression.py            # 分钟预测质量评估（验证集，可反复运行）
python test_pipeline.py                 # 数据/特征层 10 项测试
python server/test_backend.py           # 后端 6 项测试
python server/weather_serving.py --self-test   # 天气服务自检（含双端预报）
```

## 关键设计决策

### 1. 字段处理（按文档方式一）
`01_prepare_data.py` 从原始数据集复制生成 `Dataset/flights_sample_3m_clean.csv`，
删除文档列出的 12 个字段：`AIRLINE_DOT, AIRLINE_CODE, DOT_CODE, FL_NUMBER,
ORIGIN_CITY, DEST_CITY, CANCELLATION_CODE, DELAY_DUE_CARRIER, DELAY_DUE_WEATHER,
DELAY_DUE_NAS, DELAY_DUE_SECURITY, DELAY_DUE_LATE_AIRCRAFT`。
原始数据集永不修改（manifest 记录其 MD5，测试脚本会验证未被改动）。

### 2. 特征（防数据泄漏）
预测"未来"时只能用**订票/起飞前已知**的信息：

| 特征 | 来源 |
|---|---|
| 月份、日、星期几、是否周末 | FL_DATE |
| 计划出发/到达小时 | CRS_DEP_TIME / CRS_ARR_TIME |
| 航空公司、起飞/到达机场、航线(ORIGIN_DEST) | AIRLINE / ORIGIN / DEST |
| 飞行距离 | DISTANCE |

实际运行字段（DEP_TIME、TAXI_OUT、WHEELS_OFF、ARR_TIME、ELAPSED_TIME、AIR_TIME 等）
是飞行后才知道的，**绝不作为特征**，否则数据泄漏、线下虚高线上失效。

### 3. 数据集划分（会议纪要规定）
- **训练集**：2019-2021（1,791,628 行）— 训练模型
- **验证集**：2022（667,762 行）— 早停、调参、一切采用决策，可反复使用
- **测试集**：2023（463,484 行）— **仅在 `14_test_2023.py` 收官时读取一次**
  （2026-10-13，三档消融×三组件，结果已定格；对同一存档拒绝重跑）

### 4. 类别不平衡与防过拟合
- 延误率约 33-40%（延误>0口径）：CatBoost 用 `auto_class_weights=Balanced`，LightGBM 用 `scale_pos_weight`
- 早停（early stopping，验证集 AUC/MAE 150 轮不提升即停）+ L2 正则 + 限制树深/叶子数
- 回归任务（延误分钟数）长尾严重，用 **MAE** 作为损失函数更稳健

### 5. 两段式预测（对应会议纪要示例）
分类模型给概率 P(延误)，回归模型只在"实际延误>0"的航班上训练预测分钟数：
```
期望延误分钟 = P(延误) × E[分钟 | 延误]
```

### 6. 调参过程与结论（全程只用验证集，测试集只用了一次）

**"准确率/精确率/召回率同时 80%" 不可行**（`05_feasibility_analysis.py` 可复现）：
即使按"延误>0"口径（延误率约33-39%），P=R=80% 仍要求模型在 FPR≈13% 处 TPR=80%，
对应 AUC≈0.95+；而"起飞前可知特征"的 AUC 上限实测约 0.69，验证集上任何阈值都无法
同时满足两者（recall=80% 时 precision 仅约 0.48）。这是特征信息量的上限，不是调参能解决的。

两轮验证集调参（`02b_tune_on_validation.py`，20+个候选配置）：
- 第1轮：升级特征（美国节假日 + 航线×月份历史延误率）→ AUC 反而从 0.682 掉到 0.637。
  原因一：历史率统计包含样本自身标签（训练内自泄漏）；修复为"as-of 只用过去年份"后仍仅 0.60
  ——因为 CatBoost 内部的 ordered target encoding 已无泄漏地学到了同样的信号，
  显式加入的小样本组统计反而是噪声。**该负结果已记录存档（tuning_round1.log）。**
- 第2轮：v1 特征下扫超参（lr/depth/l2/num_leaves等10个配置）→ CatBoost 已饱和（±0.0007）；
  LightGBM 微调后 0.6735→**0.6756**，采纳（lr=0.05, num_leaves=63, min_child_samples=300）。

**最终配置（v1 时代的最终配置；现役线上模型见上文「最终采用的模型」）**
（存档 `models/20261005_155440_final_gt0/`，延误定义 = 延误时间>0）：v1 特征 + CatBoost 基线超参
+ LightGBM 调优超参。验证集(2022)与测试集(2023)指标高度一致，无过拟合。
历史15分钟口径的对比存档保留在 `models/20261005_153619_final_v1/`。

| 模型（出发延误分类, 延误>0） | 验证集AUC | 测试集AUC | 测试集 acc / P / R |
|---|---|---|---|
| CatBoost | 0.6923 | 0.6911 | 0.638 / 0.544 / 0.642 |
| LightGBM | 0.6898 | 0.6888 | 0.637 / 0.544 / 0.632 |

到达延误分类略低（测试集AUC≈0.629）。若把阈值从0.5降到0.42（验证集调优值），
召回率可提升到 0.81（精确率 0.487，F1 0.609）——按业务偏好权衡即可。

若想要更高指标，需要更强信息源（如天气数据、前序航班链路、实时机位数据），
或改变预测粒度（按航线/时段聚合预测）。

### 7. 延误时间（分钟数）预测得准不准？（`06_eval_regression.py`，验证集2022）

两个层面、每层都和朴素基线对比（基线只用训练集统计）：

**A. 已知延误的前提下，延误多少分钟**（仅延误>0航班，出发延误 n=261,370）：

| 方法 | MAE | RMSE | R² | ±10min命中率 | ±15min命中率 |
|---|---|---|---|---|---|
| 基线:恒定预测训练中位数(16min) | 32.19 | 78.59 | -0.095 | 39.4% | 66.9% |
| CatBoost | **31.57** | 77.67 | -0.070 | **41.9%** | 57.6% |
| LightGBM | 31.63 | 77.77 | -0.073 | 41.2% | 58.3% |

解读：模型MAE比最优朴素基线低约0.6分钟、±10分钟命中率略高；但 R² 为负
（均方误差意义下不如"恒定预测均值"）——说明**具体延误几分钟主要由当天随机因素
（天气、流量管制、机务）决定，起飞前特征只能解释一小部分**。这也是领域常识：
延误"会不会发生"可预测性较好，"确切几分钟"本质难预测。

**B. 两段式期望延误**（全体航班，期望延误 = P(延误)×预测分钟，目标=max(实际延误,0)）：

| 方法 | MAE | ±5min | ±10min |
|---|---|---|---|
| 基线:全部预测0分钟 | 15.33 | 69.8% | 75.4% |
| 基线:恒定预测训练均值 | 19.87 | 9.0% | 19.8% |
| 两段式 CatBoost | **12.93** | **72.8%** | **80.0%** |
| 两段式 LightGBM | 12.95 | 72.9% | 80.3% |

解读：合成后全面胜过两个基线（MAE 12.9 vs 15.3/19.9，约80%的航班期望延误
预测误差在±10分钟内）——概率×分钟的两段式是有效的产品化输出形式。

> **勘误（2026-10-09）**：上表"两段式"行含评估口径泄漏——历史代码按【真实】是否延误
> 决定哪些行的分钟数记 0（部署时不可知），导致 12.93 偏乐观。修正后（06 脚本已修，
> 可重跑复现）的诚实口径为：v1 两段式 MAE ≈ 17.4；加天气（13 号脚本）≈ 17.2。
> 产品网页展示的期望延误来自**区间模型**的 Σ(档位概率×档位中位分钟)（概率校准更好），
> 该口径的对照见 12 号脚本报告，不受本勘误影响。

### 8. 提升空间实验（`07_experiments_val.py`，验证集，结论：已到特征信息上限）

| 实验 | 验证集AUC | 结论 |
|---|---|---|
| 基线 CatBoost / LightGBM | 0.6923 / 0.6898 | — |
| E1 概率集成(两者平均) | **0.6934** | +0.001，边际收益 |
| E2 学习率0.03+3000轮 | 0.6927 | 无提升（已收敛） |
| E3 新增15分钟时刻粒度 | 0.6924 | 无提升（小时粒度已够） |
| E4 XGBoost | 0.6839 | 更差，换模型无意义 |
| E5 单段式直接回归期望延误 | MAE 15.2 | 劣于两段式12.9，架构选择正确 |

另测保序概率校准：期望延误MAE 12.93→13.32 反而变差（类别加权的概率偏高
与"回归学的是条件中位数"部分抵消），故保持现有概率输出。
**两个GBDT实现+6种改法全部落在0.684-0.693 → 瓶颈在特征信息量，不在算法/参数。
要突破需新信息源（天气、前序航班链路/机号）或改聚合粒度（按航线×时段预测）。**

### 9. 多情景概率输出（`08_train_bucket_models.py`，对应会议纪要原始示例）

会议纪要的示例是"航班A有70%的概率延误X分钟，有10%的概率延误Y分钟"——
为此训练了**延误区间多分类模型**（5档），预测时直接输出每个区间的概率：

> 出发延误：**52%概率不延误(准点或提前)；26%概率延误1-15分钟(约5分钟)；
> 9%概率延误16-30分钟(约22分钟)；7%概率延误31-60分钟(约42分钟)；
> 7%概率延误60分钟以上(约104分钟)**，概率加权期望延误10.5分钟

- 区间：不延误(≤0) / 1-15 / 16-30 / 31-60 / >60分钟（`config.DELAY_BUCKET_EDGES`）
- 模型：CatBoost/LightGBM 多分类 × 出发/到达（存档 `20261006_124341_buckets/`，
  `models/bucket_latest.json` 指向）
- **不用类别加权** → 各档概率的绝对值更可信（验证集每档预测概率 vs 实际频率
  偏差仅1-3个百分点，如 不延误档 0.693 vs 0.609；二分类模型的 P 因 Balanced
  加权偏高，如 demo 航班二分类说出发延误66% vs 区间模型48%）
- 验证集多分类准确率：出发 0.609 / 到达 0.627（5档随机基线=0.20-0.33）
- `04_predict.py --demo` 的 `scenarios` 字段即此输出，前端可直接渲染成分段条形图

## 存档与可追溯（会议纪要要求）

每次训练生成 `models/<时间戳[_quick]>/`：
- `cls_dep_delay__catboost.cbm` 等模型文件
- `<task>__<algo>__metadata.json`：特征清单、超参数、数据划分、随机种子、
  验证集指标、库版本、训练耗时——完整记录"是怎么训练的"
- `run_info.json`；`03` 评估后追加 `evaluation_report.md/json`
- `models/latest_run.json` 指向最近一次存档，`03/04` 默认使用它

> 注：即使步骤完全一致，重训结果也可能略有差异（随机噪声不可控），
> 所以每个存档请保留，不要覆盖。

## 测试验证脚本（test_pipeline.py，10 项）

| 测试 | 验证内容 |
|---|---|
| 01 | 原始数据集 MD5 未变（从未被修改） |
| 02 | 清洗数据集已删除12个字段、其余列与原始一致、行数一致 |
| 03 | 训练/验证/测试按年份划分、互不重叠、并集=全量 |
| 04 | 特征无泄漏（不含运行时字段与目标列）、无缺失、取值合法 |
| 05 | 延误定义正确(>0min)；任务样本正确排除取消/改航/延误值缺失 |
| 06 | 每个存档的模型文件+metadata齐全且内容合规 |
| 07 | 模型可重新加载：概率∈[0,1]、回归预测>0 |
| 08 | metadata 中的种子/超参数与 config 一致（可追溯） |
| 09 | 端到端冒烟：临时目录从零跑通 清洗→训练→预测 |
| 10 | 评估报告指标齐全且数值合法 |
| 11 | 区间多分类模型：概率和为1、非负、档数正确、存档齐全 |

```bash
python test_pipeline.py             # 全部（约2-4分钟）
python test_pipeline.py --skip-e2e  # 跳过最慢的端到端测试
pytest test_pipeline.py -v          # pytest 方式
```

## 给前端（11.4 成品）的对接说明 —— 已实现（Vue3 + Flask + SQLite，全英文界面）

前后端已全部实现并通过浏览器端到端验证，代码在 `server/`（Flask后端）和 `frontend/`（Vue3+Vite）。
**产品口径（2026-10-08，导师建议聚焦单一任务）：只预测出发延误，不再输出到达延误预测。**
CRS_ARR_TIME 仍是必填输入（唯一确定航班 + 块时长特征原料），到达模型存档保留但服务层不对外输出。
界面为全英文，视觉参考国泰航空官网（深林绿 `#00332E/#00645A` + 暖白 `#FAFAF7`、白色预订卡片+顶部标签页、
下划线式表单）。支持**可选的**用户注册/登录（SQLite 独立库）与查询历史回看；**游客无需登录即可使用全部预测功能**。

### 启动方式

```bash
# 1. 首次：构建元数据库（约3秒，数据来自原始数据集，只读）
python server/build_meta_db.py

# 2. 构建前端（改过前端代码后需要重新构建）
cd frontend && npm install && npm run build && cd ..

# 3. 启动后端（同时托管前端页面）
python server/app.py            # 打开 http://127.0.0.1:5000

# 前端开发模式（可选，热更新）：cd frontend && npm run dev  (5173，已配代理到5000)
```

### 前端输入（6必填 + 1可选）

| 字段 | 必填 | 输入方式 |
|---|---|---|
| FL_DATE | ✅ | 日期选择器（YYYY-MM-DD） |
| AIRLINE | ✅ | 下拉搜索：支持 IATA 代码（DL/dl）或部分名称（Delta/DELTA），大小写不敏感；也支持原样全名直接提交 |
| ORIGIN / DEST | ✅ | 下拉搜索：支持机场代码或**城市名**（如 new york → JFK/LGA；EWR 归 Newark, NJ）；也支持原样代码直接提交 |
| CRS_DEP_TIME / CRS_ARR_TIME | ✅ | hhmm 整数（0000-2359，后端校验） |
| DISTANCE | ⬜ | 留空时后端自动兜底（见下），填写则优先生效 |

### 后端距离兜底（DISTANCE 未填时按优先级）

1. **航线历史中位数**（routes 表，来自原始数据集 7,785 条航线）
2. 该起飞机场所有航线距离的中位数（该航线无历史时）
3. 该到达机场所有航线距离的中位数
4. 全局中位数

响应中 `distance_source` 字段标明实际使用层级，前端展示给用户（如 "1,199 mi (route history median)"）。

### API 一览

```
GET  /api/airlines?q=delta        # 航司模糊搜索（代码/名称）
GET  /api/airports?q=new%20york   # 机场模糊搜索（代码/城市）
POST /api/predict                 # 预测（JSON body；登录用户成功后自动存入历史）
POST /api/auth/register           # 注册并自动登录 {username,password}
POST /api/auth/login              # 登录
POST /api/auth/logout             # 登出
GET  /api/auth/me                 # 当前会话用户（未登录返回 user:null）
GET  /api/history                 # 当前用户的预测历史列表（需登录）
GET  /api/history/<id>            # 某条历史的完整预测结果（仅本人可见）
```

`POST /api/predict` 请求/响应示例（仅出发延误口径）：
```json
// 请求（CRS_ARR_TIME 仍必填：唯一确定航班 + 块时长特征原料）
{"FL_DATE":"2024-07-04","AIRLINE":"DL","ORIGIN":"ATL","DEST":"DEN",
 "CRS_DEP_TIME":1830,"CRS_ARR_TIME":2015}
// 响应（节选，全部英文字段；无任何 arr_* 输出）
{"resolved":{"airline":"Delta Air Lines Inc.","origin":"ATL","dest":"DEN","distance":1199.0},
 "distance_source":"route_history",
 "verdicts":{"dep_delay":{"level":"good","bucket":0,"phrase":"Expected on-time departure","p_delay":0.485}},
 "scenarios":{"dep_delay":{"scenarios":[{"bucket":0,"label":"On time","probability":0.515},
   {"bucket":1,"label":"1-15 min delay","probability":0.26,"typical_minutes":5.0},...],
   "expected_delay_min_from_buckets":13.1,"summary":"52% on time · 26% 1-15 min delay (~5 min) · …"}},
 "predictions":{"dep_delay":{...}}}   // 两段式对照输出（仅出发）
```

### 总结语规则（server/verdict.py，模型输出后的后端规则）

**纯 argmax（用户确认的规则）**：总结语永远跟随概率最高的那一档，不再设额外分级——
不延误最高 → `Expected on-time departure/arrival`；如"延误1-15分钟"最高 → `Expected 1-15 min departure/arrival delay`。
`level`（good绿/notice黄/warn橙/alert红）仅决定徽章配色，跟随获胜档位的严重程度。
模型层（config/04_predict）的中文档位标签仅作内部编号，由服务层在 API 出口统一翻译为英文。

### 用户账户与查询历史（server/auth.py，独立 app.db）

- 账户与历史存放在 `server/app.db`（users + query_history 两张表），与 `flight_meta.db` 完全分离——
  后者每次重建都会删库，故用户数据绝不能放那边。
- 密码 PBKDF2-SHA256 哈希（24万次迭代+随机盐，仅标准库）；会话用 Flask session（密钥持久化在 server/.secret_key）。
- **仅保存成功的预测**，且仅登录用户；游客预测不落库、无任何功能限制。
- 历史列表条目含航线/航司/日期/两条总结语，点击条目回看当时完整预测结果（GET /api/history/<id>，仅本人）。

### 天气模型线上服务（server/weather_serving.py + Open-Meteo 免费预报 API）

- 训练用 IEM 实测 METAR（10 号脚本），线上用 Open-Meteo 预报生成**同构的天气特征**
  （温度°F/露点/湿度/风速节/阵风/能见度英里/降水英寸/云量okta/雷暴·雨·雪·雾标志，skyl1 预报缺失）。
- **双端天气**：模型元数据含 `_d` 特征时自动额外取【到达机场@计划到达时刻】的预报
  （跨夜航班按到达分钟<出发分钟+1天）——对应训练侧的 GDP 机制特征（目的地雷雨/大雾
  会把航班扣在起飞机场）。旧 25 特征模型只取起飞侧，行为不变。
- 机场坐标来自 OpenFlights（首次自动构建 `weather/airport_coords.json`，含 PBI(2026年更名DJT)/WRG/USA 兜底）。
- 模型启用注册表 `models/wx_serving.json`：`cls_run`/`bucket_run`/`reg_run` 指向存档，
  可选 `cls_file`/`bucket_file`/`reg_file` 指定文件名（默认 `*_v1wx_*.cbm` 旧版）；
  **null 的组件自动沿用 v1 基线**；预报失败/超16天/无坐标 -> 天气缺测标志，模型照常预测；整体失败回退 v1。
- 响应新增 `weather_source` 字段（前端 meta 行显示 "Weather: Open-Meteo forecast"）。
- 自检：`python server/weather_serving.py --self-test`（验证坐标库与双端预报映射）。

### 最终形态与 2023 测试集收官成绩（14_test_2023.py，一次性评估）

**2026-10-13 定格**：三组件全部为 od 档（v1 计划特征 11 + 起飞侧天气 14 + 到达侧天气 14 = 39 特征），
注册表现值即 `wx_serving.json` 当前内容。2023 测试集（46.3 万行，延误>0 占 39.8%）已按纪律
**一次性评估完毕**（模型评估前冻结：训练 2019-2021 / 早停选模 2022），结果：

| 指标（2023 测试） | v1 仅计划 | wx +起飞天气 | **od +双端天气(线上)** |
|---|---|---|---|
| 二分类 AUC | 0.6911 | 0.7032 | **0.7038** |
| 区间5档准确率 | 0.5962 | 0.5983 | **0.5991** |
| 区间 macro F1 | 0.1506 | 0.1639 | **0.1696** |
| 期望延误 MAE(分) | 19.73 | 19.35 | **19.18** |
| 两段式 MAE(分) | 19.35 | 19.01 | **18.88** |
| 两段式 ±5min 命中 | 29.5% | 33.1% | **35.4%** |

- 天气全链路增益（od−v1）：AUC **+0.0127**、两段式 MAE **−0.47 分钟**；到达侧增益（od−wx）：AUC +0.0006、MAE −0.13 分钟。
- 2022→2023 落差极小（AUC ±0.004 内），三档排序在从未见过的测试集上完全保持——无验证集过拟合迹象。
- 存档：`models/20261013_133654_test2023/`（报告 test2023_report.md、图 charts/、可复现 results.json）；
  该脚本对同一存档拒绝重跑（`--force` 才放行），且不修改任何生产指针。

### 元数据库（server/flight_meta.db）——已与原始数据集严格核验

18家航司（代码↔全名一一对应）、380个机场（与 ORIGIN∪DEST 完全一致）、7,785条航线
（距离中位数抽查逐一吻合，航班数合计=3,000,000）。不重复、不遗漏。

### 后端测试

```bash
python server/test_backend.py    # 6项：航司/机场模糊搜索、距离兜底、argmax总结语、端到端预测(仅出发口径)、注册登录+历史
```
