# -*- coding: utf-8 -*-
"""
全局配置：路径、字段清单、数据集划分、延误定义等。
所有脚本共享本配置，保证"每个人记住自己是怎么训练的"（会议纪要要求）。
"""
from pathlib import Path

# ---------------------------------------------------------------- 路径
CODE_DIR = Path(__file__).resolve().parent          # flight_delay_model/
PROJECT_ROOT = CODE_DIR.parent                       # Group Project/
DATASET_DIR = PROJECT_ROOT / "Dataset"
ORIGINAL_CSV = DATASET_DIR / "flights_sample_3m.csv"          # 原始数据集（永不修改）
CLEAN_CSV = DATASET_DIR / "flights_sample_3m_clean.csv"       # 去除不使用字段后的新数据集（方式一）
MANIFEST_JSON = DATASET_DIR / "data_manifest.json"            # 清洗清单（含原始文件MD5，用于验证原始数据未被改动）
MODELS_DIR = CODE_DIR / "models"                              # 模型存档根目录
LATEST_RUN_JSON = MODELS_DIR / "latest_run.json"              # 指向最近一次训练存档

# ------------------------------------------------- 训练不需要使用的字段
# 来自《训练模型不需要使用的字段》文档：
#   1) AIRLINE_DOT / AIRLINE_CODE / DOT_CODE：与 AIRLINE 重复（都用于唯一确定航空公司）
#   2) FL_NUMBER：同一航班号会随时间变化用于不同航线，对长期预测无意义
#   3) ORIGIN_CITY / DEST_CITY：分别与 ORIGIN / DEST 重复
#   4) CANCELLATION_CODE：没有给出对应的取消原因
#   5) DELAY_DUE_*：部分延误航班在此缺失，且我们不需要预测延误原因
EXCLUDED_FIELDS = [
    "AIRLINE_DOT",
    "AIRLINE_CODE",
    "DOT_CODE",
    "FL_NUMBER",
    "ORIGIN_CITY",
    "DEST_CITY",
    "CANCELLATION_CODE",
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
]

# ---------------------------------------------------------------- 防数据泄漏
# 以下字段是"飞行过程中/之后"才能知道的实际运行数据。
# 它们不在文档的排除清单里（清洗数据集保留它们，供EDA用），
# 但预测"未来"时不可知，因此绝不能作为训练特征。
LEAKAGE_FIELDS = [
    "DEP_TIME",        # 实际出发时刻
    "TAXI_OUT",        # 实际滑出时间
    "WHEELS_OFF",      # 实际离地时刻
    "WHEELS_ON",       # 实际落地时刻
    "TAXI_IN",         # 实际滑入时间
    "ARR_TIME",        # 实际到达时刻
    "ELAPSED_TIME",    # 实际飞行时长
    "AIR_TIME",        # 实际空中时间
    "CANCELLED",       # 是否取消（结果）
    "DIVERTED",        # 是否改航（结果）
]

# ------------------------------------------------------------- 数据集划分
# 会议纪要：2019-2021 训练；2022 验证（可反复用）；2023 测试（只能用一次）
TRAIN_YEARS = (2019, 2020, 2021)
VAL_YEARS = (2022,)
TEST_YEARS = (2023,)

# ------------------------------------------------------------ 延误相关定义
# 延误定义（小组统一口径）：延误时间 > 0 分钟即为"延误"（含出发/到达延误）
DELAY_THRESHOLD_MIN = 0

# 延误区间（多情景输出）：区间边界(分钟)，class 0 = 不延误(<=0)，之后依次为
# (0,15], (15,30], (30,60], >60 —— 对应会议纪要"70%概率延误X分钟、10%概率延误Y分钟"的展示
DELAY_BUCKET_EDGES = [0, 15, 30, 60]
BUCKET_LABELS = ["不延误", "延误1-15分钟", "延误16-30分钟", "延误31-60分钟", "延误60分钟以上"]
N_BUCKETS = len(BUCKET_LABELS)

# 训练目标（任务名 -> 定义）
TASKS = {
    "cls_dep_delay":  "分类：出发延误概率  P(DEP_DELAY > 0)",
    "cls_arr_delay":  "分类：到达延误概率  P(ARR_DELAY > 0)",
    "reg_dep_delay":  "回归：出发延误分钟数（仅在实际延误>0的航班上训练）",
    "reg_arr_delay":  "回归：到达延误分钟数（仅在实际延误>0的航班上训练）",
    "mcls_dep_delay": "多分类：出发延误落入哪个区间（5档，多情景概率输出）",
    "mcls_arr_delay": "多分类：到达延误落入哪个区间（5档，多情景概率输出）",
}

RANDOM_SEED = 42
