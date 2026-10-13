# -*- coding: utf-8 -*-
"""
步骤 10：下载机场逐小时天气（METAR/ASOS），并聚合成可关联的逐小时天气表。

数据源（免费公开，请限速使用，本脚本已内置 2 秒间隔 + 指数退避）：
  Iowa Environmental Mesonet, Iowa State University
  门户页:  https://mesonet.agron.iastate.edu/request/download.phtml
  接口:    https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py

流程（全部支持断点重跑，随时 Ctrl-C，重跑同一命令即续传）：
  [1/3] 时区表：从我们自己的航班时刻表推导每个机场的 UTC 偏移
        （原理：CRS_ELAPSED_TIME − (到达-出发时刻差) ≈ −时区差，图上 BFS 传播），
        输出 weather/tz_map.json —— 不需要任何外部时区数据。
  [2/3] 逐机场下载原始 METAR（2018-12-25 ~ 2024-01-05，UTC），
        存 weather/raw/{机场码}.csv；已存在且非空则跳过；失败站点重跑时自动重试。
  [3/3] 聚合：每机场每小时取最接近整点的观测，解析天气代码为雷暴/雪/雨/雾标志，
        合并输出 weather/weather_hourly.csv.gz + weather/weather_report.json。

用法：
    python 10_weather_download.py                # 全流程（约20-40分钟）
    python 10_weather_download.py                # 中断后重跑同一命令 = 续传
    python 10_weather_download.py --force        # 强制重新聚合（重下原始文件仍会跳过）
    python 10_weather_download.py --only ATL     # 只处理单个机场（调试用）
"""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config

WX_DIR = Path(__file__).resolve().parent / "weather"
RAW_DIR = WX_DIR / "raw"

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
FIELDS = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust",
          "vsby", "p01i", "skyc1", "skyl1", "wxcodes"]

# IATA -> IEM 站号映射（2026-10-08 逐一实测验证）。这批机场在 asos.py 里用
# 3 字母 IATA 查不到数据，需要用 4 字母 ICAO 或现站号（多为 AK/HI/海外领地
# 及改名机场）。原始文件仍按 IATA 命名保存，保证与航班 ORIGIN 关联一致。
# 特例(2026-10-09 补)：PBI 已于 2026-08-18 更名 DJT（IEM 历史数据随之迁移，
# 见 flydjt.org/name-change-faqs）；USA 的 ICAO=KJQF；WRG 的 ICAO=PAWG。
STATION_OVERRIDES = {
    "ADK": "PADK", "ADQ": "PADQ", "AKN": "PAKN", "ANC": "PANC", "BET": "PABE",
    "BRW": "PABR", "CDB": "PACD", "CDV": "PACV", "DLG": "PADL", "FAI": "PAFA",
    "GST": "PAGS", "JNU": "PAJN", "KTN": "PAKT", "OME": "PAOM", "OTZ": "PAOT",
    "PSG": "PAPG", "SCC": "PASC", "SIT": "PASI", "YAK": "PAYA", "WRG": "PAWG",
    "HNL": "PHNL", "ITO": "PHTO", "KOA": "PHKO", "LIH": "PHLI", "OGG": "PHOG",
    "BQN": "TJBQ", "PSE": "TJPS", "SJU": "TJSJ", "STT": "TIST", "STX": "TISX",
    "PPG": "NSTU", "GUM": "PGUM", "SPN": "PGSN",
    "AZA": "KIWA", "BKG": "KBBG", "FCA": "KGPI", "HHH": "KHXD", "MQT": "KSAW",
    "SCE": "KUNV", "YUM": "KNYL", "PBI": "KDJT", "USA": "KJQF",
}

# 州 -> 标准时区偏移（小时，相对UTC）。仅作 BFS 覆盖不到的机场的兜底；
# 商业机场基本都能从航班表推导出精确值。
STATE_TZ_DEFAULT = {
    "CT": -5, "DE": -5, "FL": -5, "GA": -5, "IN": -5, "KY": -5, "ME": -5,
    "MD": -5, "MA": -5, "MI": -5, "NH": -5, "NJ": -5, "NY": -5, "NC": -5,
    "OH": -5, "PA": -5, "RI": -5, "SC": -5, "VT": -5, "VA": -5, "WV": -5,
    "DC": -5,
    "AL": -6, "AR": -6, "IL": -6, "IA": -6, "KS": -6, "LA": -6, "MN": -6,
    "MS": -6, "MO": -6, "NE": -6, "ND": -6, "OK": -6, "SD": -6, "TN": -6,
    "TX": -6, "WI": -6,
    "AZ": -7, "CO": -7, "ID": -7, "MT": -7, "NM": -7, "UT": -7, "WY": -7,
    "CA": -8, "NV": -8, "OR": -8, "WA": -8,
    "AK": -9, "HI": -10, "PR": -4, "VI": -4,
    "GU": 10, "MP": 10, "AS": 11,
}
# 不实行夏令时的州/领地：时刻表推导对它们有 ±1 小时歧义（全国同步切夏令时
# 的假设不成立），因此直接用州固定偏移覆盖（PHX=-7、HNL=-10、SJU=-4 等）
NO_DST_STATES = {"AZ", "HI", "PR", "VI", "GU", "MP", "AS"}


# ---------------------------------------------------------------- 时区推导
def derive_tz_offsets(clean_csv) -> dict:
    """从航班时刻表本身推导机场 UTC 偏移（无需外部时区数据）。

    原理：计划块时长 CRS_ELAPSED_TIME 与"计划起降的当地分钟差"之差 ≈ −时区差。
    分钟差必须先换算成分钟（hhmm 整数直接相减是错的）；|差|<=360 分钟的行
    排除跨日取模伪影(±1440)后，按航线取中位数得到时区差，再以 ATL（东部,-5）
    为根在航线图上 BFS 传播。夏令时两端同步移动，整数小时差全年稳定；
    亚利桑那/夏威夷用州标记关闭夏令时。
    """
    usecols = ["ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME",
               "CRS_ELAPSED_TIME", "DISTANCE"]
    df = pd.read_csv(clean_csv, usecols=usecols, low_memory=False)
    airports = sorted(set(df["ORIGIN"]) | set(df["DEST"]))

    dep = pd.to_numeric(df["CRS_DEP_TIME"], errors="coerce")
    arr = pd.to_numeric(df["CRS_ARR_TIME"], errors="coerce")
    wall_min = (((arr // 100) * 60 + arr % 100)
                - ((dep // 100) * 60 + dep % 100)) % 1440
    diff = df["CRS_ELAPSED_TIME"] - wall_min
    ok = diff.abs() <= 360
    sub = df[ok].assign(_d=diff[ok])
    g = sub.groupby(["ORIGIN", "DEST"])["_d"]
    med = g.median()
    cnt = g.size()
    edges = {}
    for (o, d_), m in med.items():
        if cnt[(o, d_)] >= 30:
            dtz = -round(m / 60)
            if abs(dtz) <= 6:                     # 美国境内合理的时区差
                edges.setdefault(o, []).append((d_, dtz))

    offset = {"ATL": -5}
    queue = deque(["ATL"])
    while queue:
        o = queue.popleft()
        for d, dtz in edges.get(o, []):
            if d not in offset:
                offset[d] = offset[o] + dtz
                if -11 <= offset[d] <= -4:
                    queue.append(d)
                else:
                    del offset[d]

    # 兜底：州默认值；再兜不住（如海外领地）标记 None，天气关联时自然缺失
    # 州信息从航线距离表（flight_meta.db）拿不到就用距离最长的航线两端同州假设不可靠，
    # 因此直接对未覆盖机场用全数据集 ORIGIN_CITY? —— 简化：读 airports 表的 state。
    import sqlite3
    meta_db = Path(__file__).resolve().parent / "server" / "flight_meta.db"
    state_of = {}
    if meta_db.exists():
        conn = sqlite3.connect(meta_db)
        for code, st in conn.execute("SELECT code, state FROM airports"):
            state_of[code] = st
        conn.close()

    tz_map = {}
    for a in airports:
        st = state_of.get(a, "")
        base = offset.get(a, STATE_TZ_DEFAULT.get(st))
        tz_map[a] = {
            "base_offset_hours": base,
            "state": st,
            "derived_from_schedule": a in offset,
            "has_dst": st not in NO_DST_STATES if st else True,
        }
    # 不实行夏令时的州/领地：用州固定偏移覆盖（推导对它们有 ±1 小时歧义）
    no_dst_fixed = 0
    for v in tz_map.values():
        st = v["state"]
        if st in NO_DST_STATES and st in STATE_TZ_DEFAULT:
            v["base_offset_hours"] = STATE_TZ_DEFAULT[st]
            v["has_dst"] = False
            v["derived_from_schedule"] = False
            v["note"] = f"{st} 不实行夏令时，使用州固定偏移"
            no_dst_fixed += 1
    derived = sum(1 for v in tz_map.values() if v["derived_from_schedule"])
    print(f"[1/3] 时区表: {len(tz_map)} 机场, 其中 {derived} 个从时刻表精确推导, "
          f"{no_dst_fixed} 个不实行夏令时的州用固定偏移(AZ/HI/领地), "
          f"{len(tz_map) - derived - no_dst_fixed} 个用州默认, "
          f"{sum(1 for v in tz_map.values() if v['base_offset_hours'] is None)} 个无偏移(将无天气)")
    return tz_map


# ---------------------------------------------------------------- 下载
def fetch_station(code, d1, d2, sleep_s):
    """下载单个机场的 METAR（UTC）。优先用 3 字母 IATA，无数据时改用
    STATION_OVERRIDES 里的 ICAO/现站号重试；report_type=3 空结果再去掉该参数。"""
    sid = STATION_OVERRIDES.get(code, code)

    def _get(station_id, report_type=True):
        params = {
            "station": station_id, "data": ",".join(FIELDS),
            "year1": str(d1.year), "month1": str(d1.month), "day1": str(d1.day),
            "year2": str(d2.year), "month2": str(d2.month), "day2": str(d2.day),
            "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no",
            "missing": "M", "trace": "T", "direct": "no",
        }
        if report_type:
            params["report_type"] = "3"
        url = IEM_URL + "?" + urllib.parse.urlencode(params)
        last_err = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "flight-delay-course-project/1.0"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                if len(text) > 200 and "valid" in text.split("\n")[0]:
                    return text
                last_err = f"bad payload ({len(text)} bytes)"
            except (urllib.error.URLError, OSError) as e:
                last_err = str(e)
            time.sleep(min(10 * 2 ** attempt, 120))
        raise RuntimeError(last_err)

    attempts = [code]
    if sid != code:
        attempts.append(sid)
    for st in attempts:
        try:
            for report_type in (True, False):
                try:
                    return _get(st, report_type=report_type), st
                except RuntimeError:
                    continue
        finally:
            pass
    raise RuntimeError("no data under any station id")


def download_all(stations, d1, d2, sleep_s, only=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stations if only is None or s == only]
    failed, resumed = [], 0
    t0 = time.time()
    for i, code in enumerate(todo, 1):
        out = RAW_DIR / f"{code}.csv"
        if out.exists() and out.stat().st_size > 200:
            resumed += 1
            continue
        try:
            text, used_sid = fetch_station(code, d1, d2, sleep_s)
            out.write_text(text, encoding="utf-8")
            n_rows = len(text.strip().split("\n")) - 1
            via = f" (via {used_sid})" if used_sid != code else ""
            print(f"  [{i:>3}/{len(todo)}] {code}: {n_rows:,} obs{via} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            failed.append(code)
            print(f"  [{i:>3}/{len(todo)}] {code}: FAILED ({e})", flush=True)
        time.sleep(sleep_s)
    print(f"[2/3] 下载完成: 成功 {len(todo)-len(failed)}, 跳过(已存在) {resumed}, "
          f"失败 {len(failed)}{'' if not failed else ' -> 重跑同一命令即可重试失败站点'}")
    return failed


# ---------------------------------------------------------------- 聚合
SKYC_OKTA = {"CLR": 0, "SKC": 0, "NSC": 0, "FEW": 2, "SCT": 4, "BKN": 6,
             "OVC": 8, "VV": 9}


def hourize_one(path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    if "valid" not in d.columns:
        return pd.DataFrame()
    t = pd.to_datetime(d["valid"], errors="coerce")
    d = d.assign(_h=t.dt.floor("h"), _s=(t - t.dt.floor("h")).dt.total_seconds())
    d = d.dropna(subset=["_h"]).sort_values("_s")
    hourly = d.groupby("_h", as_index=False).first()   # 每小时最接近整点的观测
    for c in ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "vsby", "p01i", "skyl1"]:
        if c in hourly.columns:
            hourly[c] = pd.to_numeric(
                hourly[c].astype(str).str.replace("T", "0", regex=False),
                errors="coerce")
    if "skyc1" in hourly.columns:
        hourly["skyc1_okta"] = hourly["skyc1"].astype(str).str.strip().str[:3].map(SKYC_OKTA)
    if "wxcodes" in hourly.columns:
        wx = hourly["wxcodes"].fillna("").astype(str)
        hourly["wx_ts"] = wx.str.contains("TS").astype("float")
        hourly["wx_sn"] = wx.str.contains("SN").astype("float")
        hourly["wx_ra"] = wx.str.contains("RA").astype("float")
        hourly["wx_fg"] = (wx.str.contains("FG") | wx.str.contains("BR")).astype("float")
    cols = ["_h", "tmpf", "dwpf", "relh", "sknt", "gust", "vsby", "p01i",
            "skyl1", "skyc1_okta", "wx_ts", "wx_sn", "wx_ra", "wx_fg"]
    cols = [c for c in cols if c in hourly.columns]
    hourly = hourly[cols].rename(columns={"_h": "utc_hour"})
    # 强制 ns 单位再转整数（pandas 3.x 的 astype(int64) 跟随原时间单位，需统一）
    hourly["utc_hour"] = hourly["utc_hour"].astype("datetime64[ns]").astype("int64")
    hourly["station"] = path.stem
    return hourly


def aggregate(force=False):
    out_gz = WX_DIR / "weather_hourly.csv.gz"
    if out_gz.exists() and not force:
        print(f"[3/3] 聚合表已存在, 跳过: {out_gz}")
        return out_gz
    frames, bad = [], []
    for i, p in enumerate(sorted(RAW_DIR.glob("*.csv")), 1):
        try:
            f = hourize_one(p)
            if len(f):
                frames.append(f)
            else:
                bad.append(p.stem)
        except Exception as e:
            bad.append(p.stem)
            print(f"  {p.stem}: 解析失败 ({e})")
        if i % 50 == 0:
            print(f"  聚合进度 {i}/{len(list(RAW_DIR.glob('*.csv')))}")
    if not frames:
        raise RuntimeError("没有任何可聚合的天气文件，请先完成下载")
    allw = pd.concat(frames, ignore_index=True)
    allw["utc_hour"] = allw["utc_hour"].astype("datetime64[ns]").astype("int64")  # ns 键
    allw = allw.sort_values(["station", "utc_hour"]).reset_index(drop=True)
    allw.to_csv(out_gz, index=False, compression="gzip")
    print(f"[3/3] 聚合完成: {len(allw):,} 机场小时记录, {allw['station'].nunique()} 机场 "
          f"-> {out_gz} (解析失败 {len(bad)} 个)")
    return out_gz


def main():
    ap = argparse.ArgumentParser(description="下载并聚合机场逐小时天气(METAR)")
    ap.add_argument("--data", default=str(config.CLEAN_CSV))
    ap.add_argument("--start", default="2018-12-25")
    ap.add_argument("--end", default="2024-01-05")
    ap.add_argument("--sleep", type=float, default=2.0, help="相邻请求间隔秒数(限速礼貌)")
    ap.add_argument("--only", default=None, help="只处理单个机场代码(调试)")
    ap.add_argument("--force", action="store_true", help="强制重新聚合")
    args = ap.parse_args()

    WX_DIR.mkdir(exist_ok=True)
    d1 = date.fromisoformat(args.start)
    d2 = date.fromisoformat(args.end)

    # 1) 时区表（存在即跳过，--force 也不重算：由航班表决定，与下载无关）
    tz_path = WX_DIR / "tz_map.json"
    if tz_path.exists():
        tz_map = json.load(open(tz_path, encoding="utf-8"))
        print(f"[1/3] 时区表已存在, 跳过: {tz_path}")
    else:
        tz_map = derive_tz_offsets(args.data)
        with open(tz_path, "w", encoding="utf-8") as f:
            json.dump(tz_map, f, ensure_ascii=False, indent=2)

    # 2) 下载（机场集合 = 数据集中出现过的 ORIGIN∪DEST）
    df_codes = pd.read_csv(args.data, usecols=["ORIGIN", "DEST"], low_memory=False)
    stations = sorted(set(df_codes["ORIGIN"]) | set(df_codes["DEST"]))
    failed = download_all(stations, d1, d2, args.sleep, only=args.only)

    # 3) 聚合 + 报告
    out_gz = aggregate(force=args.force)
    agg = pd.read_csv(out_gz, usecols=["station"])
    have = set(agg["station"].unique())
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": IEM_URL,
        "range": [args.start, args.end],
        "stations_total": len(stations),
        "stations_with_weather": len(have & set(stations)),
        "stations_failed_last_run": failed,
        "stations_no_weather": sorted(set(stations) - have),
        "raw_files": len(list(RAW_DIR.glob('*.csv'))),
        "output": str(out_gz),
        "resume_note": "重跑同一命令即续传：已下载文件自动跳过，失败站点自动重试",
    }
    with open(WX_DIR / "weather_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告: {WX_DIR / 'weather_report.json'}")
    if failed:
        print(f"注意: {len(failed)} 个站点下载失败, 重跑同一命令会自动重试它们")


if __name__ == "__main__":
    main()
