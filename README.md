# IS6400 · 航班延误预测（FlightCast）

美国国内航班**出发延误**预测：CatBoost 三组件（二分类 / 五档多分类 / 回归）+ 双端天气特征，
Vue3 + Flask 全栈网页。2023 测试集收官成绩 **AUC 0.7038 / 两段式期望延误 MAE 18.88 分钟**。

> 📖 **阅读顺序**：本文件 = 怎么操作；[`PROJECT_RECORD.md`](PROJECT_RECORD.md) = 完整项目记录
> （数据来源、特征、每一步提升的对比与分析）；[`flight_delay_model/README.md`](flight_delay_model/README.md) = 技术细节手册。

---

## 最终采用的模型（线上现役）

三个组件全部为 **CatBoost**、各 **39 特征**（11 计划特征 + 起飞机场天气 14 + 到达机场天气 14），
只预测**出发延误**，由 `flight_delay_model/models/wx_serving.json` 注册表控制启用：

| # | 组件 | 回答的问题 | 存档 | 模型文件 | 2022验证 / 2023测试 |
|---|---|---|---|---|---|
| 1 | 二分类 | P(出发延误) | `models/20261009_193645_wx_dep`（11 号臂4） | `cls_dep_delay__v1wx_od__catboost.cbm` | AUC 0.7042 / **0.7038** |
| 2 | 五档区间 | 各延误区间的概率分布（"63% on time · 25% 1-15 min…"） | `models/20261013_122152_buckets_wx_dep`（12 号） | `mcls_dep_delay__v1wx_od__catboost.cbm` | acc 0.6108 / 0.5991 |
| 3 | 回归 | 若延误，大约几分钟 | `models/20261013_130457_wx_reg_dep`（13 号） | `reg_dep_delay__v1wx_od__catboost.cbm` | 两段式 MAE 17.10 / 18.88 |

兜底链（Open-Meteo 预报取不到 → 组件失败 → 整体异常时逐级回退到 v1 基线）：
`latest_run.json → 20261005_155440_final_gt0`（cls+reg）、`bucket_latest.json → 20261006_124341_buckets`（五档）。
这 13 个模型二进制已包含在仓库中，**克隆即可运行网页**；换装/回滚只需改 `wx_serving.json` 后重启 Flask。

> 曾训练并被**对照实验否决**的方案（存档保留、报告可查）：LightGBM 全家（AUC −0.003）、
> 块时长特征臂（−0.0004）、累积二分类（1/3 胜）、XGBoost（−0.008）。详见
> [`PROJECT_RECORD.md`](PROJECT_RECORD.md) 第 6 节"提升之路"。

---

## 仓库里有什么

```
IS6400/
├── PROJECT_RECORD.md          # 完整项目记录（复盘/写报告用）
├── Dataset/                   # 数据清单与字段字典（CSV 本体不入库，见下）
├── docs/img/                  # 文档配图
├── 9.24会议纪要.pdf 等        # 小组会议纪要与课程材料
└── flight_delay_model/        # 全部代码
    ├── 01~14_*.py             # 数据清洗 → 训练 → 评估 → 天气 → 收官测试（顺序即编号）
    ├── config.py / features.py
    ├── models/                # 每次训练的 metadata/指标/图表/报告（模型二进制不入库）
    ├── weather/               # 时区表/机场坐标（原始天气数据不入库）
    ├── server/                # Flask 后端 + 测试
    └── frontend/              # Vue3 前端（国泰风格全英文）
```

**不入库的大文件**（克隆后需自行获得，见下节）：原始/清洗 CSV、天气数据、模型二进制
（.cbm）、node_modules、运行时数据库。

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
→ 得到延误概率、"若延误约几分钟"、五档概率分布；天气由 Open-Meteo 实时预报自动获取。
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
| 2023 收官 | `python 14_test_2023.py` | **测试集只能用一次**，已有结果会拒绝重跑 |

每个脚本都支持断点续训（已存在的模型自动跳过）、结束后在控制台打总结、
在 `models/<运行id>/` 落盘指标 JSON + 图表 + Markdown 报告。

### 4. 换装线上模型（采用新训练结果）

线上用哪套模型由 `flight_delay_model/models/wx_serving.json` 一个文件决定
（三组件各指向一个存档+文件名，`null` 表示沿用 v1 基线）。改完重启 Flask 即生效；
回滚就是把指向改回去。**换装前先看对应存档里的对照报告，赢了才换。**

---

## 我们的工作方式（纪律）

1. **数据划分**：2019–2021 训练 / 2022 验证（一切决策只看它）/ **2023 测试只在收官用了一次**
   （`14_test_2023.py`，对同一存档拒绝重跑）；
2. **防泄露**：运行后字段（实际起降时刻、`DELAY_DUE_*` 归因列）一律不进特征；
   评估口径出过一次错已在 README 勘误（两段式曾用真实标签筛选，MAE 12.93→17.45 修正）；
3. **对照实验驱动**：每个提升都要在同一批验证行上有对照（三臂/四臂消融），
   存档报告里赢过对照才换装，不凭感觉；
4. **可追溯**：原始数据 MD5 锚定；每次训练的配置/指标/图表全部入库；
5. **Git 协作**：改动后 `git add -A && git commit -m "说明" && git push`；
   大文件（数据/模型二进制/天气数据）永远不提交，由 .gitignore 把关。
