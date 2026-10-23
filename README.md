# IS6400 · 航班延误预测（FlightCast）

美国国内航班**出发延误**预测：当前网页后端只运行 **CatBoost 五档 + isotonic 校准**，保留双端天气特征；另外三个模型和历史基线文件保留、停止调用。
Vue3 + Flask 全栈网页。**以下为四组件训练存档的历史成绩，不是当前单模型推导分钟数的回归成绩。** 2023 测试集收官成绩 **AUC 0.7038 / 五档 RPS 0.137（isotonic 校准 0.37pp）/ 期望延误 MAE 17.09 分钟（单段式）**。

## 单模型服务更新（2026-10-23 同步）

**新增版本说明：[README_SINGLE_MODEL.md](README_SINGLE_MODEL.md)**，包含启用状态、接口来源、VS Code / PyCharm 启动命令、测试、开销及演示。

- HTTP 入口直接运行五档模型，每次一次推理；原 isotonic 校准器保持不变。
- `cls_run`、`reg_run`、`cond_reg_run` 设为 `null`，不加载、不推理，也不再调用旧 v1 基线。原 run ID 记录在 `disabled_runs`，所有模型文件保留。
- 页面继续使用同一五档来源的结果。兼容 `predictions.dep_delay` 数值也由五档推导，并增加 `*_source`；不再返回三个独立模型的预测数值。
- 天气获取失败仍使用原缺测特征；模型或校准器故障返回 503，不恢复停用模型。旧历史记录保留保存时的结果。
- 原四个模型各有已验证优势，停用是小组在既定页面口径下减少开销的运行决策，不代表其训练工作或离线结果被否定。

本次通过 6 项单模型测试、29 项账号历史接口测试、6 项原后端测试、11 项前端时序测试。详见 [后端开销说明](BACKEND_PERFORMANCE.md)。

> 📖 **阅读顺序**：本文件 = 怎么操作；[`PROJECT_RECORD.md`](PROJECT_RECORD.md) = 完整项目记录
> （数据来源、特征、每一步提升的对比与分析）；[`flight_delay_model/README.md`](flight_delay_model/README.md) = 技术细节手册。

---

## 网页与账号更新（2026-10-20）

- 前端采用深蓝主色、浅色背景和圆角卡片，增加日期/时间选择器；历史记录每页 10 条，使用左右分屏查看预测。
- 账号记录由后端校验归属；登录用户的成功预测会保存，访客预测不入库。登录会话支持服务端注销与过期校验，避免退出后旧 Cookie 继续使用。
- 修复切换账号时慢请求带回旧记录、退出失败仍显示已退出、重复注册遗留事务，以及异常输入报错等问题。
- 原有模型权重、模型注册表和 API 地址保持不变。当前卡片的 `Chance of any delay` 为五档概率中的 `1 − 准时概率`，`Expected delay` 为五档加权分钟数，与二分类/单段式模型指标口径不同。
- 更新后重启后端并重新登录一次；首次启动会增加登录会话表，原账号与历史记录保留。

新增 29 项后端接口测试和 11 项前端请求时序测试，另通过原有 6 项后端测试。详见 [测试说明](AUTH_HISTORY_TEST_REPORT.md)。Windows 环境及启动入口见 [Windows 环境配置](Windows环境配置.md)；本机模型绝对路径、账号数据库、密钥、依赖和大数据集不提交。

```bash
python flight_delay_model/server/test_auth_history_api.py
cd flight_delay_model/frontend && npm run test:auth
```

Windows 可在项目根目录运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\test-auth.ps1`，统一执行新增的两组测试。

---

## 四组件训练存档与当前启用状态（仅五档在线）

四个组件全部为 **CatBoost**、各 **39 特征**（11 计划特征 + 起飞机场天气 14 + 到达机场天气 14），
外加 isotonic 概率校准层。下表保留 **2026-10-17 四组件训练方案**及其成绩；2026-10-22 小组决定精简运行，当前 HTTP 后端和页面仅使用五档+isotonic（见最后一列）。

| # | 组件 | 回答的问题 | 存档 | 模型文件 | 2022验证 / 2023测试 | 页面显示 |
|---|---|---|---|---|---|---|
| 1 | 二分类 | P(出发延误) | `models/20261009_193645_wx_dep`（11 号臂4） | `cls_dep_delay__v1wx_od__catboost.cbm` | AUC 0.7042 / **0.7038** | ⏸ 已停用（存档保留） |
| 2 | 五档+isotonic | 各延误区间的概率分布（"63% on time · 25% 1-15 min…"） | `models/20261013_122152_buckets_wx_dep`（12 号） | `mcls_dep_delay__v1wx_od__catboost.cbm` + `isotonic_full2022.json` | RPS 0.129/0.137 · 校准 1.85/**0.37**pp（三层指标详见技术 README） | ✅ **唯一上屏来源**（概率、期望延误、五档图） |
| 3 | 条件回归 | 若延误，大约几分钟 | `models/20261013_130457_wx_reg_dep`（13 号） | `reg_dep_delay__v1wx_od__catboost.cbm` | MAE 31.30 / 34.43（常数基线 32.28/35.52） | ⏸ 已停用（存档保留） |
| 4 | 单段式回归 | 期望延误（中位口径） | `models/20261017_152303_improve`（15 号） | `reg_dep_single__single39.cbm` | MAE **15.14 / 17.09**（常数0 基线 15.33/17.32；旧两段式 17.10/18.88） | ⏸ 已停用（存档保留） |

> **页面显示口径（2026-10-20 定稿）**：出于口径一致考虑，页面全部数字统一出自
> 五档+isotonic 一个分布——"Chance of any delay" = 1−准点档概率、"Expected delay" =
> 五档加权分钟数、五档条形图逐档展示，同源数字必然自洽，中期两个目标由这一个分布
> 同时兑现。该来源的概率绝对值经校准实测可信（ECE 0.21pp，2022 验证，平均预测 0.392
> vs 实际 0.391；对比：二分类 8.54pp、五档原始 8.67pp；2023 测试 0.37pp）。
> **四个模型存档全部保留、并非被淘汰**，各有已验证的强项（如分钟精度以单段式最优，MAE 15.14/17.09）。当前网页后端只计算五档来源，其他三者不再在线推理；将来恢复多模型需显式调整服务编排与测试。

14 个原有模型二进制继续保留在仓库中。旧基线注册表和存档保留作历史记录；当前 HTTP 服务不再调用它们，也不在故障时回退到它们。

> 曾训练并被**对照实验否决**的方案（存档保留、报告可查）：LightGBM 全家（AUC −0.003）、
> 块时长特征臂（−0.0004）、累积二分类（1/3 胜）、XGBoost（−0.008）、典型流量特征
> （单段式与五档双任务证伪）、**五档类别加权 w1**（决策层收益低于零训练的先验校正
> argmax，且概率层崩塌：LogLoss 1.11→1.46、校准 3.5→14.6pp，被严格支配）。
> 详见 [`PROJECT_RECORD.md`](PROJECT_RECORD.md) 第 6 节"提升之路"。

---

## 仓库里有什么

```
IS6400/
├── PROJECT_RECORD.md          # 完整项目记录（复盘/写报告用）
├── Dataset/                   # 数据清单与字段字典（CSV 本体不入库，见下）
├── docs/img/                  # 文档配图
├── 9.24会议纪要.pdf 等        # 小组会议纪要与课程材料
└── flight_delay_model/        # 全部代码
    ├── 01~17_*.py             # 数据清洗 → 训练 → 评估 → 天气 → 收官/优化（顺序即编号）
    ├── config.py / features.py
    ├── models/                # 每次训练的 metadata/指标/图表/报告（线上在用的模型二进制已入库）
    ├── weather/               # 时区表/机场坐标（原始天气数据不入库）
    ├── server/                # Flask 后端 + 测试
    └── frontend/              # Vue3 前端（全英文界面、深蓝主题）
```

**不入库的大文件**：原始/清洗 CSV、原始天气数据、额外训练存档二进制、node_modules 和运行时账号数据库；上述已追踪的 14 个模型文件继续保留。

---

## 如何操作（组员上手指南）

### 0. 环境准备（一次性）

```bash
git clone https://github.com/forg1ve1125/IS6400.git && cd IS6400
python3 -m pip install pandas numpy catboost lightgbm scikit-learn matplotlib flask
# 前端依赖（只在做网页时需要）
cd flight_delay_model/frontend && npm install && cd ../..
```

### 1. 放入数据集

从 Kaggle 下载 `flights_sample_3m.csv`（300 万行，2019–2023）放到 `Dataset/` 目录：

> 下载页：<https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023>
> （原始出处：BTS TranStats On-Time Performance，<https://www.transtats.bts.gov/ONTIME/>）

然后用 MD5 核对与 `Dataset/data_manifest.json` 里记录的一致（保证大家用同一份数据）：

```bash
md5 Dataset/flights_sample_3m.csv     # 应为 633c02070a85e801b6bf23d76acd3fdf
```

### 2. 把网页跑起来（最快路径）

```bash
cd flight_delay_model
python 01_prepare_data.py        # 清洗数据集（约1-2分钟，生成 *_clean.csv）
python server/build_meta_db.py   # 构建航司/机场/航线元数据库（约3秒）
# 线上在用的模型二进制已包含在仓库里，无需拷贝；重训全部存档见第 3 节
cd frontend && npm run build && cd ..
python server/app.py             # 打开 http://127.0.0.1:5000
```

网页玩法：填日期/航司/出发地/目的地/计划起降时刻（距离可留空自动回填）→ Predict delay
→ 得到五档概率、总体延误概率和五档加权分钟数；天气由 Open-Meteo 实时预报自动获取。
可选注册登录，登录后自动保存历史记录。

常用运维命令：

```bash
python server/test_backend.py                # 6项后端测试
python server/weather_serving.py --self-test # 天气服务自检（含双端预报）
lsof -ti:5000 | xargs kill                   # 停止 Flask
```

### 3. 复现模型训练（可选，编号即顺序）

| 步骤 | 命令 | 说明 |
|---|---|---|
| 基线模型 | `python 02_train_models.py` | v1 计划特征，CatBoost/LightGBM |
| 天气数据 | `python 10_weather_download.py` | IEM 逐站下载（[门户](https://mesonet.agron.iastate.edu/request/download.phtml)），断点续传（约1-2小时） |
| 二分类四臂 | `python 11_train_weather.py` | v1 / +起飞天气 / +块时长 / +双端天气 对照 |
| 区间模型 | `python 12_train_bucket_weather.py` | 五档概率，39 特征 |
| 回归模型 | `python 13_train_regression_weather.py` | "若延误约几分钟"，39 特征 |
| 2023 收官 | `python 14_test_2023.py` | 三档消融终评（已有结果拒绝重跑） |
| 单段式回归 | `python 15_improve_minutes_bucket.py --only-a` | 期望延误 v2（约10分钟） |
| 不平衡消融 | `python 16_bucket_imbalance.py --arms ""` | 零训练路线（isotonic 等）；`--arms w1` 加权臂（可选） |
| 里程碑3 | `python 17_milestone3_2023.py` | 制作 isotonic 校准器 + v2 配置的 2023 评估 |

每个脚本都支持断点续训（已存在的模型自动跳过）、结束后在控制台打总结、
在 `models/<运行id>/` 落盘指标 JSON + 图表 + Markdown 报告。

### 4. 换装线上模型（采用新训练结果）

当前配置为 `serving_mode=bucket_only`，只有 `bucket_run` 启用，其余三个槽位为 `null`，表示停止调用。更换五档存档需同时核对 metadata、bucket_stats 和 isotonic 校准器，然后重启 Flask。
本版入口明确限制为五档；恢复其他组件需调整服务编排和测试，不能只把槽位置为非空。详见 [新版本说明](README_SINGLE_MODEL.md#8-恢复其他模型时的注意事项)。

---

## 我们的工作方式（纪律）

1. **数据划分**：2019–2021 训练 / 2022 验证（一切决策只看它）/ **2023 测试仅在三个冻结里程碑
   各读取一次**（03 号 v1 基线 → 14 号 v1-wx-od 终评 → 17 号 v2 终评，对同一存档拒绝重跑）；
2. **防泄露**：运行后字段（实际起降时刻、`DELAY_DUE_*` 归因列）一律不进特征；
   评估口径出过一次错已在 README 勘误（两段式曾用真实标签筛选，MAE 12.93→17.45 修正）；
3. **对照实验驱动**：每个提升都要在同一批验证行上有对照（三臂/四臂消融），
   存档报告里赢过对照才换装，不凭感觉；
4. **可追溯**：原始数据 MD5 锚定；每次训练的配置/指标/图表全部入库；
5. **Git 协作**：改动后 `git add -A && git commit -m "说明" && git push`；
   大文件（数据/模型二进制/天气数据）永远不提交，由 .gitignore 把关。
