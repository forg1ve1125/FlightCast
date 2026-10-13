# -*- coding: utf-8 -*-
"""
特征工程：把清洗后数据集的原始列，转换成模型训练使用的特征矩阵。

设计原则（防止数据泄漏）：
只使用"起飞前/订票时就已可知"的信息：
    航班日期(年月日/星期/是否周末)、计划出发/到达时刻(小时)、
    航空公司、起飞机场、到达机场、航线、飞行距离。
绝不使用 LEAKAGE_FIELDS 中的实际运行数据（实际时刻、滑行、空中时间等）。
"""
import numpy as np
import pandas as pd

from config import DELAY_THRESHOLD_MIN

# 分类特征（CatBoost / LightGBM 均按类别特征处理）
CATEGORICAL_FEATURES = ["AIRLINE", "ORIGIN", "DEST", "ROUTE"]
# 数值特征
NUMERIC_FEATURES = [
    "FL_MONTH",
    "FL_DAY_OF_MONTH",
    "FL_DAY_OF_WEEK",
    "FL_IS_WEEKEND",
    "DEP_HOUR",
    "ARR_HOUR",
    "DISTANCE",
]
# 历史统计特征（只从训练集年份2019-2021计算，映射到验证/测试集——无泄漏）
HIST_FEATURES = [
    "IS_HOLIDAY",
    "HIST_ROUTE_RATE",
    "HIST_ROUTE_MIN",
    "HIST_ORIGIN_RATE",
    "HIST_DEST_RATE",
    "HIST_AIRLINE_RATE",
]
# 最终特征顺序（训练/预测必须严格一致）
FEATURES = NUMERIC_FEATURES + HIST_FEATURES + CATEGORICAL_FEATURES
# 第一版特征（不含历史统计），用于对比/兼容旧存档
FEATURES_V1 = NUMERIC_FEATURES + CATEGORICAL_FEATURES

HIST_SMOOTH_K = 50  # 历史率平滑先验样本数（样本少时向全局均值收缩）


# ------------------------------------------------------------- 美国联邦假日
def _nth_weekday(year, month, weekday, n):
    d = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - d.dayofweek) % 7
    return d + pd.Timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    offset = (d.dayofweek - weekday) % 7
    return d - pd.Timedelta(days=offset)


def us_holidays(index) -> np.ndarray:
    """返回是否美国联邦假日的 bool 数组（固定日期+规则假日）。接受 DatetimeIndex/Series。"""
    index = pd.DatetimeIndex(index)
    years = index.year.unique()
    days = set()
    for y in years:
        days |= {
            pd.Timestamp(y, 1, 1),                       # 元旦
            _nth_weekday(y, 1, 0, 3),                    # MLK 1月第3个周一
            _nth_weekday(y, 2, 0, 3),                    # 总统日 2月第3个周一
            _last_weekday(y, 5, 0),                      # 阵亡将士纪念日 5月最后周一
            pd.Timestamp(y, 7, 4),                       # 独立日
            _nth_weekday(y, 9, 0, 1),                    # 劳动节 9月第1个周一
            _nth_weekday(y, 11, 3, 4),                   # 感恩节 11月第4个周四
            pd.Timestamp(y, 12, 25),                     # 圣诞节
        }
    return index.isin(pd.DatetimeIndex(sorted(days)))


# ------------------------------------------------------------ 历史统计
def compute_historical_stats(train_df: pd.DataFrame) -> dict:
    """
    只用训练集年份(2019-2021)的未取消航班，统计"延误率/平均延误分钟"：
      ROUTE×月份、ORIGIN×月份、DEST×月份、AIRLINE。
    这些是"起飞前已知的 历史/统计信息"，映射到验证/测试集不构成数据泄漏。
    """
    d = train_df
    ok = ((pd.to_numeric(d["CANCELLED"], errors="coerce").fillna(0) == 0)
          & (pd.to_numeric(d["DIVERTED"], errors="coerce").fillna(0) == 0))
    d = d[ok]
    dep = pd.to_numeric(d["DEP_DELAY"], errors="coerce")
    y = (dep > DELAY_THRESHOLD_MIN).astype(float)
    mins = dep.clip(lower=0)

    glob_rate = float(y.mean())
    glob_min = float(mins.mean())

    month = pd.to_datetime(d["FL_DATE"]).dt.month

    def smoothed_table(keys: pd.Series) -> dict:
        """按给定键聚合 -> 平滑后的 {key: [rate, mean_min]}。"""
        g = pd.DataFrame({"k": keys, "y": y, "m": mins})
        grp = g.groupby("k").agg(n=("y", "size"), s=("y", "sum"), mm=("m", "mean"))
        rate = (grp["s"] + HIST_SMOOTH_K * glob_rate) / (grp["n"] + HIST_SMOOTH_K)
        return {k: [round(float(r), 5), round(float(m), 2)]
                for k, r, m in zip(grp.index, rate, grp["mm"])}

    route = (d["ORIGIN"].astype(str) + "_" + d["DEST"].astype(str))
    stats = {
        "global": [round(glob_rate, 5), round(glob_min, 2)],
        "route_month": smoothed_table(route + "|" + month.astype(str)),
        "origin_month": smoothed_table(d["ORIGIN"].astype(str) + "|" + month.astype(str)),
        "dest_month": smoothed_table(d["DEST"].astype(str) + "|" + month.astype(str)),
        "airline": smoothed_table(d["AIRLINE"].astype(str)),
    }
    return stats


def _hist_lookup(stats: dict, table: str, keys) -> tuple:
    """查历史统计，缺失时回退全局值。返回 (rate, mean_min) 两个数组。"""
    table_d = stats.get(table, {})
    g_rate, g_min = stats.get("global", [0.18, 11.0])
    rates, mins_ = [], []
    for k in keys:
        v = table_d.get(k)
        if v is None:
            rates.append(g_rate); mins_.append(g_min)
        else:
            rates.append(v[0]); mins_.append(v[1])
    return np.asarray(rates, dtype="float32"), np.asarray(mins_, dtype="float32")


def _global_only_stats(flat_stats: dict) -> dict:
    """只保留全局均值的历史统计（用于没有'过去年份'可用的最早年份）。"""
    return {"global": flat_stats.get("global", [0.18, 11.0]),
            "route_month": {}, "origin_month": {}, "dest_month": {}, "airline": {}}


def compute_asof_stats(train_df: pd.DataFrame) -> dict:
    """
    防止'训练集内自我包含泄漏'的 as-of 历史统计：
      2019 年样本 -> 只用全局均值（没有过去数据）
      2020 年样本 -> 只用 2019 年统计
      2021 年样本 -> 只用 2019-2020 统计
      验证/测试集  -> 用全部训练年份(2019-2021)统计（full）
    这样训练时模型见到的历史特征与预测时同样'诚实'，不会高估其作用。
    """
    year = year_series(train_df)
    flat = compute_historical_stats(train_df)
    snapshots = {}
    prior = []
    for y in sorted(int(v) for v in year.unique()):
        if prior:
            snapshots[str(y)] = compute_historical_stats(train_df[year.isin(prior)])
        else:
            snapshots[str(y)] = _global_only_stats(flat)
        prior.append(y)
    return {"snapshots": snapshots, "full": flat}


def resolve_stats(stats: dict) -> dict:
    """兼容两种格式：新的 as-of 包装 {snapshots, full} 与旧的扁平统计。"""
    if stats is None:
        return None
    return stats.get("full", stats) if isinstance(stats, dict) else None


def build_features(df: pd.DataFrame, hist_stats: dict = None,
                   feature_set: str = "v2") -> pd.DataFrame:
    """
    从原始列构造特征矩阵 X。
    feature_set='v1': 原始 11 个特征（兼容旧存档/对照实验）
    feature_set='v2': + 美国节假日 + 训练期历史延误统计（默认）
    hist_stats=None 时 v2 的历史特征以全局默认值填充（仅供结构测试）。
    """
    x = pd.DataFrame(index=df.index)

    fl_date = pd.to_datetime(df["FL_DATE"])
    x["FL_MONTH"] = fl_date.dt.month.astype("int16")
    x["FL_DAY_OF_MONTH"] = fl_date.dt.day.astype("int16")
    x["FL_DAY_OF_WEEK"] = fl_date.dt.dayofweek.astype("int16")  # 周一=0 ... 周日=6
    x["FL_IS_WEEKEND"] = (fl_date.dt.dayofweek >= 5).astype("int8")

    # 计划时刻 hhmm -> 小时（CRS_DEP_TIME=1155 表示 11:55）
    dep_hour = (pd.to_numeric(df["CRS_DEP_TIME"]) // 100).astype("int16")
    arr_hour = (pd.to_numeric(df["CRS_ARR_TIME"]) // 100).astype("int16")
    x["DEP_HOUR"] = dep_hour.clip(0, 23)
    x["ARR_HOUR"] = arr_hour.clip(0, 23)

    x["DISTANCE"] = pd.to_numeric(df["DISTANCE"]).astype("float32")

    x["AIRLINE"] = df["AIRLINE"].astype(str)
    x["ORIGIN"] = df["ORIGIN"].astype(str)
    x["DEST"] = df["DEST"].astype(str)
    # 航线 = 起飞机场_到达机场（会议纪要：预测航线比预测航班号更有意义）
    x["ROUTE"] = x["ORIGIN"] + "_" + x["DEST"]

    if feature_set == "v1":
        return x[FEATURES_V1]

    # ---- v2 历史统计特征 ----
    month = fl_date.dt.month.astype(int)
    x["IS_HOLIDAY"] = us_holidays(fl_date).astype("int8")

    if hist_stats is None:
        default = np.asarray([0.18, 11.0], dtype="float32")
        for col in ["HIST_ROUTE_RATE", "HIST_ROUTE_MIN", "HIST_ORIGIN_RATE",
                    "HIST_DEST_RATE", "HIST_AIRLINE_RATE"]:
            x[col] = default[0] if col.endswith("RATE") else default[1]
    else:
        r, m = _hist_lookup(hist_stats, "route_month",
                            (x["ROUTE"] + "|" + month.astype(str)).to_numpy())
        x["HIST_ROUTE_RATE"], x["HIST_ROUTE_MIN"] = r, m
        r, _ = _hist_lookup(hist_stats, "origin_month",
                            (x["ORIGIN"] + "|" + month.astype(str)).to_numpy())
        x["HIST_ORIGIN_RATE"] = r
        r, _ = _hist_lookup(hist_stats, "dest_month",
                            (x["DEST"] + "|" + month.astype(str)).to_numpy())
        x["HIST_DEST_RATE"] = r
        r, _ = _hist_lookup(hist_stats, "airline", x["AIRLINE"].to_numpy())
        x["HIST_AIRLINE_RATE"] = r

    return x[FEATURES]


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """构造各任务的目标列（y_*），不修改原列。"""
    out = df.copy()
    dep_delay = pd.to_numeric(out["DEP_DELAY"], errors="coerce")
    arr_delay = pd.to_numeric(out["ARR_DELAY"], errors="coerce")
    out["DEP_DELAY"] = dep_delay
    out["ARR_DELAY"] = arr_delay

    # 分类目标：延误 > 0 分钟（小组统一口径：任何正延误都算延误）
    out["y_cls_dep"] = (dep_delay > DELAY_THRESHOLD_MIN).astype("int8")
    out["y_cls_arr"] = (arr_delay > DELAY_THRESHOLD_MIN).astype("int8")
    # 延误区间目标（多情景输出）：0=不延误, 1=(0,15], 2=(15,30], 3=(30,60], 4=>60
    import config as _cfg
    edges = [-np.inf] + [float(e) for e in _cfg.DELAY_BUCKET_EDGES] + [np.inf]
    out["y_bucket_dep"] = pd.cut(dep_delay, bins=edges, labels=False)
    out["y_bucket_arr"] = pd.cut(arr_delay, bins=edges, labels=False)
    # 回归目标：延误分钟数（仅延误>0 的行有效，训练时再过滤）
    out["y_reg_dep"] = dep_delay
    out["y_reg_arr"] = arr_delay
    return out


def task_mask(df: pd.DataFrame, task: str) -> pd.Series:
    """每个任务的可用样本掩码：剔除取消/改航/目标缺失的行；回归任务再要求延误>0。
    分类任务的"目标缺失"= 对应延误分钟数为NaN（延误状态未知，不能当作未延误）。"""
    base = (pd.to_numeric(df["CANCELLED"], errors="coerce").fillna(0) == 0) & (
        pd.to_numeric(df["DIVERTED"], errors="coerce").fillna(0) == 0
    )
    if "_dep" in task:
        delay = pd.to_numeric(df["DEP_DELAY"], errors="coerce")
        m = base & delay.notna()
        if task.startswith("reg_"):
            m &= delay > 0
        return m
    if "_arr" in task:
        delay = pd.to_numeric(df["ARR_DELAY"], errors="coerce")
        m = base & delay.notna()
        if task.startswith("reg_"):
            m &= delay > 0
        return m
    raise ValueError(f"未知任务: {task}")


def bucket_of_delay(delay: pd.Series) -> pd.Series:
    """延误分钟数 -> 区间编号 0..4（与 config.DELAY_BUCKET_EDGES 一致）。"""
    import config as _cfg
    edges = [-np.inf] + [float(e) for e in _cfg.DELAY_BUCKET_EDGES] + [np.inf]
    return pd.cut(pd.to_numeric(delay, errors="coerce"), bins=edges, labels=False)


def split_by_year(df: pd.DataFrame, years) -> pd.DataFrame:
    """按 FL_DATE 的年份划分子集。"""
    year = pd.to_datetime(df["FL_DATE"]).dt.year
    return df[year.isin(list(years))].copy()


def year_series(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["FL_DATE"]).dt.year


def align_categories(x: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """预测时把分类特征对齐到训练时的类别集合（LightGBM 需要一致的 category 编码）。"""
    x = x.copy()
    for col, cats in categories.items():
        x[col] = pd.Categorical(x[col], categories=cats)
    return x
