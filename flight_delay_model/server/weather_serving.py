# -*- coding: utf-8 -*-
"""
天气模型线上服务：用 Open-Meteo 免费预报 API 生成与训练(METAR)同构的天气特征，
加载天气版模型（11/12/13 号脚本存档）替换 v1 基线结果。

双端天气：模型元数据里含 _d 后缀特征（11 号臂4 / 12 / 13 的 v1wx_od 版）时，
自动额外取【到达机场@计划到达时刻】的预报（跨夜航班按到达分钟<出发分钟+1天）；
旧版 25 特征模型（_o 后缀）只取起飞侧，行为不变。

设计要点：
  - 机场坐标：首次调用时从 OpenFlights airports.dat 生成 weather/airport_coords.json
    （含 PBI/WRG/USA 等改名机场的手工兜底坐标），之后离线可用。
  - 时刻：机场当地时间 -> UTC（复用 tz_map.json，含夏令时；夏令时按各自机场、
    各自当地日期判定），向 Open-Meteo 请求该 UTC 日的逐小时预报，取整点匹配。
  - 变量映射（单位在请求端直接换成英制/节/英寸，尽量对齐 METAR 口径）：
      tmpf=temperature_2m(°F) dwpf=dew_point_2m(°F) relh=relative_humidity_2m(%)
      sknt=wind_speed_10m(kn) gust=wind_gusts_10m(kn) p01i=precipitation(in)
      vsby=visibility(m->mi) skyc1_okta=cloud_cover(%)×8/100
      wx_ts/ra/sn/fg = WMO weather_code 映射(雷暴/雨/雪/雾)
      skyl1=预报无云底高度 -> 恒缺失(树模型可处理)
  - 任何失败（无预报/无坐标/网络/超出16天预报窗）-> 该侧全部天气特征缺失 +
    wx_missing_{o,d}=1，模型按缺测处理；再失败则整体回退 v1 结果。
  - 模型注册表 models/wx_serving.json：
      {"cls_run":.., "bucket_run":.., "reg_run":..}
    可选 "<key>_file" 指定模型文件名（默认 *_v1wx_*.cbm，旧版）；
    null 的组件沿用 v1 基线（渐进式启用）。

自检（验证 API 与映射，不做预测）：
    python server/weather_serving.py --self-test
只构建坐标库：
    python server/weather_serving.py --build-coords
"""
import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SERVER_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SERVER_DIR))   # verdict 等同目录模块在任意加载方式下可导入

import config  # noqa: E402
from features import build_features  # noqa: E402

WX_DIR = PROJECT_DIR / "weather"
COORDS_JSON = WX_DIR / "airport_coords.json"
REGISTRY_JSON = config.MODELS_DIR / "wx_serving.json"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPENFLIGHTS_URL = ("https://raw.githubusercontent.com/jpatokal/"
                   "openflights/master/data/airports.dat")

# 手工兜底坐标（改名/极小机场在 openflights 主数据里可能滞后）
FALLBACK_COORDS = {
    "PBI": (26.6832, -80.0956),   # 2026-08 更名 DJT/KDJT，同一机场
    "WRG": (56.4847, -132.3698),
    "USA": (35.3866, -80.7099),   # ICAO KJQF
}

# WMO weather_code -> 天气标志
_WMO_TS = {95, 96, 99}
_WMO_RA = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}
_WMO_SN = {71, 73, 75, 77, 85, 86}
_WMO_FG = {45, 48}

_model_cache = {}
_coords_cache = None
_tz_cache = None
_iso_cache = None


def _load_isotonic(run_dir):
    """加载五档概率的 isotonic 校准器（17 号脚本产出 isotonic_full2022.json）。"""
    global _iso_cache
    if _iso_cache is None:
        path = run_dir / "isotonic_full2022.json"
        _iso_cache = json.load(open(path, encoding="utf-8")) if path.exists() else False
    return _iso_cache


def _apply_isotonic(proba, iso):
    """逐类分段线性插值后归一化（与 sklearn IsotonicRegression.predict 等价）。

    兼容单航班(1-D, 5)与批量为(N, 5)两种输入。
    """
    single = (np.ndim(proba) == 1)
    p = np.atleast_2d(np.asarray(proba, dtype=float))
    n_cls = int(iso.get("classes", p.shape[1]))
    out = np.column_stack([
        np.interp(p[:, k], iso["fit"][str(k)]["x"], iso["fit"][str(k)]["y"])
        for k in range(min(n_cls, p.shape[1]))])
    out = np.clip(out, 1e-6, None)
    out = out / out.sum(axis=1, keepdims=True)
    return out[0] if single else out


# ---------------------------------------------------------------- 坐标
def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def build_coords(force=False):
    """从 OpenFlights airports.dat 构建 IATA -> [lat, lon]，合并手工兜底。
    返回内层 coords dict（与 get_coords 的缓存一致）。"""
    global _coords_cache
    if COORDS_JSON.exists() and not force:
        _coords_cache = json.load(open(COORDS_JSON, encoding="utf-8"))["coords"]
        return _coords_cache
    req = urllib.request.Request(OPENFLIGHTS_URL,
                                 headers={"User-Agent": "flight-delay-course-project/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode("utf-8", errors="replace")
    coords = {}
    for line in text.splitlines():
        f = [x.replace('\\"', "").strip('"') for x in line.split(",")]
        if len(f) < 8 or not f[4]:
            continue
        try:
            lat, lon = float(f[6]), float(f[7])
        except ValueError:
            continue
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            coords[f[4]] = [lat, lon]
    for k, v in FALLBACK_COORDS.items():
        coords.setdefault(k, list(v))
    WX_DIR.mkdir(exist_ok=True)
    with open(COORDS_JSON, "w", encoding="utf-8") as f:
        json.dump({"built_at": _now_iso(), "source": OPENFLIGHTS_URL,
                   "count": len(coords), "coords": coords}, f, ensure_ascii=False)
    _coords_cache = coords
    return coords


def get_coords(iata):
    global _coords_cache
    if _coords_cache is None:
        if not COORDS_JSON.exists():
            build_coords()
        _coords_cache = json.load(open(COORDS_JSON, encoding="utf-8"))["coords"]
    return _coords_cache.get(iata)


# ---------------------------------------------------------------- 时刻
def _nth_sunday(year, month, n):
    d0 = datetime(year, month, 1).date()
    return d0 + timedelta(days=(6 - d0.weekday()) % 7 + 7 * (n - 1))


def _in_us_dst(d):
    return _nth_sunday(d.year, 3, 2) <= d < _nth_sunday(d.year, 11, 1)


def _load_tz_map():
    global _tz_cache
    if _tz_cache is None:
        tz_path = WX_DIR / "tz_map.json"
        if not tz_path.exists():
            return {}
        _tz_cache = json.load(open(tz_path, encoding="utf-8"))
    return _tz_cache


def _utc_of_local(iata, local_dt):
    """机场本地时刻(datetime) -> UTC datetime（tz_map + 夏令时按当地日期）。失败 None。"""
    info = _load_tz_map().get(iata) or {}
    base = info.get("base_offset_hours")
    if base is None:
        return None
    off = base + (1 if (info.get("has_dst", True) and _in_us_dst(local_dt.date())) else 0)
    return local_dt - timedelta(hours=off)


def utc_of_local(iata, fl_date, hhmm):
    """机场当地时间(日期+hhmm) -> UTC datetime。保留旧签名兼容。"""
    hhmm = int(hhmm)
    local = (datetime.fromisoformat(str(fl_date))
             + timedelta(hours=hhmm // 100, minutes=hhmm % 100))
    return _utc_of_local(iata, local)


def arrival_local_datetime(flight: dict) -> datetime:
    """计划到达的【到达机场当地】时刻：到达分钟<出发分钟 => 跨夜+1天。"""
    dep, arr = int(flight["CRS_DEP_TIME"]), int(flight["CRS_ARR_TIME"])
    d = datetime.fromisoformat(str(flight["FL_DATE"]))
    if (arr // 100) * 60 + arr % 100 < (dep // 100) * 60 + dep % 100:
        d = d + timedelta(days=1)
    return d + timedelta(hours=arr // 100, minutes=arr % 100)


# ---------------------------------------------------------------- 预报
WX_SOURCE_COLS = ["tmpf", "dwpf", "relh", "sknt", "gust", "vsby", "p01i",
                  "skyl1", "skyc1_okta", "wx_ts", "wx_sn", "wx_ra", "wx_fg"]


def _missing(side):
    m = {f"{c}_{side}": None for c in WX_SOURCE_COLS}
    m[f"wx_missing_{side}"] = 1
    return m


def fetch_weather_at_local(iata, local_dt, side="o"):
    """某机场某【当地时刻】的预报特征，键后缀 _o(起飞侧)/_d(到达侧)。

    任何失败返回该侧全缺失（wx_missing_{side}=1）。"""
    missing = _missing(side)
    try:
        utc = _utc_of_local(iata, local_dt)
        coords = get_coords(iata)
        if utc is None or coords is None:
            return missing
        params = {
            "latitude": coords[0], "longitude": coords[1],
            "hourly": ",".join(["temperature_2m", "dew_point_2m", "relative_humidity_2m",
                                "wind_speed_10m", "wind_gusts_10m", "precipitation",
                                "weather_code", "cloud_cover", "visibility"]),
            "start_date": (utc - timedelta(hours=6)).date().isoformat(),
            "end_date": (utc + timedelta(hours=6)).date().isoformat(),
            "timezone": "UTC",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "kn",
            "precipitation_unit": "inch",
        }
        url = OPEN_METEO_URL + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url,
                                     headers={"User-Agent": "flight-delay-course-project/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        h = data["hourly"]
        # 找最接近的整点
        times = [datetime.fromisoformat(t) for t in h["time"]]
        i = min(range(len(times)), key=lambda k: abs((times[k] - utc).total_seconds()))
        def val(name):
            v = h.get(name, [None] * len(times))[i]
            return None if v is None else float(v)
        tmpf = val("temperature_2m")
        out = {
            f"tmpf_{side}": tmpf,
            f"dwpf_{side}": val("dew_point_2m"),
            f"relh_{side}": val("relative_humidity_2m"),
            f"sknt_{side}": val("wind_speed_10m"),
            f"gust_{side}": val("wind_gusts_10m"),
            f"vsby_{side}": (None if val("visibility") is None
                             else min(val("visibility") / 1609.34, 10.0)),
            f"p01i_{side}": val("precipitation"),
            f"skyl1_{side}": None,                  # 预报无云底高度 -> 缺失
            f"skyc1_okta_{side}": (None if val("cloud_cover") is None
                                   else round(val("cloud_cover") * 8 / 100)),
            f"wx_ts_{side}": None, f"wx_sn_{side}": None,
            f"wx_ra_{side}": None, f"wx_fg_{side}": None,
        }
        wcode = h.get("weather_code", [None] * len(times))[i]
        if wcode is not None:
            out[f"wx_ts_{side}"] = float(int(wcode) in _WMO_TS)
            out[f"wx_sn_{side}"] = float(int(wcode) in _WMO_SN)
            out[f"wx_ra_{side}"] = float(int(wcode) in _WMO_RA)
            out[f"wx_fg_{side}"] = float(int(wcode) in _WMO_FG)
        out[f"wx_missing_{side}"] = 0 if tmpf is not None else 1
        return out
    except Exception:
        return missing


def fetch_forecast_weather(iata, fl_date, hhmm, side="o"):
    """起飞侧默认入口（旧行为不变：FL_DATE + hhmm 的当地时间）。"""
    hhmm = int(hhmm)
    local = (datetime.fromisoformat(str(fl_date))
             + timedelta(hours=hhmm // 100, minutes=hhmm % 100))
    return fetch_weather_at_local(iata, local, side)


def fetch_dest_forecast_weather(flight: dict):
    """到达侧入口：到达机场@计划到达时刻（当地，跨夜+1天）。"""
    try:
        return fetch_weather_at_local(flight["DEST"], arrival_local_datetime(flight), "d")
    except Exception:
        return _missing("d")


# ---------------------------------------------------------------- 模型注册
def load_registry():
    if not REGISTRY_JSON.exists():
        return {}
    return json.load(open(REGISTRY_JSON, encoding="utf-8"))


def enabled():
    """至少一个天气组件可用且坐标库就绪（或可构建）。"""
    reg = load_registry()
    return any(reg.get(k) for k in ("cls_run", "bucket_run", "reg_run"))


# 组件 -> 注册表里指定文件名的键 / 默认模型文件名（旧版 v1wx）
COMPONENT_FILE_KEYS = {"cls_run": "cls_file", "reg_run": "reg_file",
                       "bucket_run": "bucket_file", "cond_reg_run": "cond_reg_file"}
DEFAULT_COMPONENT_FILES = {
    "cls_run": "cls_dep_delay__v1wx__catboost.cbm",
    "reg_run": "reg_dep_delay__v1wx__catboost.cbm",
    "bucket_run": "mcls_dep_delay__v1wx__catboost.cbm",
    # cond_reg_run 无默认：必须显式指定（任务=若延误约X分钟的条件回归）
}


def _load_model(run_id, filename):
    key = f"{run_id}/{filename}"
    if key not in _model_cache:
        from catboost import CatBoostClassifier, CatBoostRegressor
        path = config.MODELS_DIR / run_id / filename
        meta = json.load(open(path.with_name(path.stem + "__metadata.json"),
                              encoding="utf-8"))
        if meta["task"].startswith(("cls", "mcls")):
            m = CatBoostClassifier()
        else:
            m = CatBoostRegressor()
        m.load_model(str(path))
        _model_cache[key] = (m, meta)
    return _model_cache[key]


def _predict(model, meta, x_full):
    from catboost import Pool
    x = x_full[meta["features"]]
    cat_idx = [x.columns.get_loc(c) for c in meta["categorical_features"]]
    if meta["task"].startswith("mcls"):
        return np.asarray(model.predict_proba(Pool(x, cat_features=cat_idx)))[0]
    if meta["task"].startswith("cls"):
        return float(np.asarray(model.predict_proba(Pool(x, cat_features=cat_idx)))[:, 1][0])
    return float(np.asarray(model.predict(Pool(x, cat_features=cat_idx))).ravel()[0])


# ---------------------------------------------------------------- 主入口
def predict_with_weather(flight: dict, base_result: dict) -> dict:
    """在 v1 基线结果上叠加天气模型输出；逐组件独立启用，失败组件保持 v1。

    四槽位（models/wx_serving.json，2026-10-17 最终形态 v2）：
      cls_run      -> P(出发延误)，页面 "X% chance of any delay"
      reg_run      -> 期望延误：单段式（task=reg_dep_single，目标 max(延误,0)）直接输出；
                      兼容旧版条件回归文件（走两段式 p×m 回退）
      bucket_run   -> 五档概率分布 +（可选）isotonic_full2022.json 校准层
      cond_reg_run -> "若延误约 X 分钟"（条件回归，实测优于五档推导）；
                      未配置时从五档条件分布推导作兜底
    模型元数据含 _d 特征时自动取【到达机场@计划到达时刻】的预报（双端天气）。
    """
    reg = load_registry()
    comps = []
    for key in ("cls_run", "reg_run", "bucket_run", "cond_reg_run"):
        run_id = reg.get(key)
        if not run_id:
            continue
        fname = reg.get(COMPONENT_FILE_KEYS[key]) or DEFAULT_COMPONENT_FILES.get(key)
        if not fname:
            print(f"[weather_serving] 警告: {key} 未指定文件名，该组件跳过")
            continue
        path = config.MODELS_DIR / run_id / fname
        if not path.exists():
            print(f"[weather_serving] 警告: {key} 指定的模型不存在，该组件沿用 v1: {path}")
            continue
        comps.append((key, *_load_model(run_id, fname)))
    if not comps:
        return base_result

    need_d = any("tmpf_d" in meta["features"] for _, _, meta in comps)
    wx_o = fetch_forecast_weather(flight["ORIGIN"], flight["FL_DATE"],
                                  flight["CRS_DEP_TIME"], "o")
    wx_d = (fetch_dest_forecast_weather(flight)
            if need_d and flight.get("CRS_ARR_TIME") else _missing("d"))

    x_v1 = build_features(pd.DataFrame([flight]), feature_set="v1").reset_index(drop=True)
    x_full = pd.concat([x_v1, pd.DataFrame([wx_o]), pd.DataFrame([wx_d])], axis=1)

    result = dict(base_result)
    result["predictions"] = dict(base_result.get("predictions", {}))
    result["scenarios"] = dict(base_result.get("scenarios", {}))
    src = []

    try:
        for key, model, meta in comps:
            if key == "cls_run":
                p = _predict(model, meta, x_full)
                result["predictions"].setdefault("dep_delay", {})["prob_delay"] = round(p, 4)
                src.append("cls")
            elif key == "reg_run" and meta.get("task") == "reg_dep_single":
                # 单段式：期望延误的直接来源（全体航班 max(延误,0) 中位回归）
                e = _predict(model, meta, x_full)
                b = result["predictions"].setdefault("dep_delay", {})
                b["expected_minutes_overall"] = round(max(float(e), 0.0), 1)  # MAE回归可能出微负值
                b["expected_delay_source"] = "single-stage"
                src.append("single")
            elif key == "reg_run":   # 旧版条件回归文件：兼容路径（两段式）
                m = _predict(model, meta, x_full)
                result["predictions"].setdefault("dep_delay", {})[
                    "expected_minutes_if_delayed"] = round(m, 1)
                src.append("reg")
            elif key == "cond_reg_run":
                m = _predict(model, meta, x_full)
                result["predictions"].setdefault("dep_delay", {})[
                    "expected_minutes_if_delayed"] = round(m, 1)
                src.append("cond")
            else:  # bucket_run（+ 可选 isotonic 校准层）
                from verdict import BUCKET_LABELS_EN
                run_dir = config.MODELS_DIR / reg["bucket_run"]
                proba = _predict(model, meta, x_full)
                iso = _load_isotonic(run_dir)
                if iso:
                    proba = _apply_isotonic(proba, iso)
                stats = json.load(open(run_dir / "bucket_stats.json",
                                       encoding="utf-8"))["dep"]
                med = [s["median_minutes"] for s in stats]
                items = [{"bucket": k, "label": BUCKET_LABELS_EN[k],
                          "probability": round(float(proba[k]), 4),
                          "typical_minutes": None if k == 0 else med[k]}
                         for k in range(config.N_BUCKETS)]
                items.sort(key=lambda s: -s["probability"])
                expected = sum(s["probability"] * (s["typical_minutes"] or 0) for s in items)
                parts = []
                for s in items:
                    if s["bucket"] == 0:
                        parts.append(f"{s['probability']:.0%} on time")
                    else:
                        mm = (f" (~{s['typical_minutes']:.0f} min)"
                              if s.get("typical_minutes") is not None else "")
                        parts.append(f"{s['probability']:.0%} {s['label'].lower()}{mm}")
                result["scenarios"]["dep_delay"] = {
                    "scenarios": items,
                    "expected_delay_min_from_buckets": round(float(expected), 1),
                    "summary": " · ".join(parts),
                }
                # 未配置 cond_reg 槽位时：从五档条件分布推导"若延误约X分钟"（粗粒度兜底）
                p0 = float(proba[0])
                b = result["predictions"].setdefault("dep_delay", {})
                if "expected_minutes_if_delayed" not in b and p0 < 1 - 1e-6:
                    b["expected_minutes_if_delayed"] = round(expected / max(1 - p0, 1e-9), 1)
                src.append("bucket+iso" if iso else "bucket")

        # 旧版回退与总结语组装
        b = result["predictions"].get("dep_delay", {})
        if "expected_minutes_overall" not in b and "prob_delay" in b \
                and "expected_minutes_if_delayed" in b:
            b["expected_minutes_overall"] = round(
                b["prob_delay"] * b["expected_minutes_if_delayed"], 1)
        if "prob_delay" in b and "expected_minutes_overall" in b:
            parts = [f"{b['prob_delay']*100:.1f}% chance of departure delay"]
            if "expected_minutes_if_delayed" in b:
                parts.append(f"if delayed, about {b['expected_minutes_if_delayed']:.0f} min")
            parts.append(f"expected delay {b['expected_minutes_overall']:.0f} min")
            b["summary"] = "; ".join(parts)
    except Exception:
        pass  # 任一组件失败 -> 保持 v1 对应部分

    if src:
        ok_o = wx_o.get("wx_missing_o") == 0
        ok_d = (not need_d) or (wx_d.get("wx_missing_d") == 0)
        if ok_o and ok_d:
            label = "Open-Meteo forecast"
        elif ok_o or ok_d:
            label = "Open-Meteo forecast (partial)"
        else:
            label = "Open-Meteo unavailable (weather features missing)"
        result["weather_source"] = label
        result["weather_components"] = src
    return result


# ---------------------------------------------------------------- CLI
def _self_test():
    build_coords()
    info = json.load(open(COORDS_JSON, encoding="utf-8"))
    print(f"坐标库: {info['count']} 个机场 -> {COORDS_JSON}")
    tomorrow = (datetime.now() + timedelta(days=1)).date().isoformat()
    wx = fetch_forecast_weather("ATL", tomorrow, 1830)
    print(f"ATL {tomorrow} 18:30 当地时 预报特征(起飞侧):")
    for k, v in wx.items():
        print(f"  {k:14s} = {v}")
    assert wx.get("wx_missing_o") == 0, "预报获取失败，请检查网络/API"
    flight = {"FL_DATE": tomorrow, "ORIGIN": "ATL", "DEST": "DEN",
              "CRS_DEP_TIME": 2200, "CRS_ARR_TIME": 2340}   # 跨夜+1天用例
    wx_d = fetch_dest_forecast_weather(flight)
    arr = arrival_local_datetime(flight)
    print(f"到达侧: DEN 计划到达当地时刻 {arr} (UTC {utc_of_local('DEN', arr.date(), arr.hour*100 + arr.minute)})")
    for k, v in wx_d.items():
        print(f"  {k:14s} = {v}")
    assert wx_d.get("wx_missing_d") == 0, "到达侧预报获取失败，请检查网络/API"
    print("自检通过 ✅")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Open-Meteo 天气特征服务")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--build-coords", action="store_true")
    args = ap.parse_args()
    if args.build_coords:
        c = build_coords(force=True)
        print(f"坐标库已重建: {c['count']} 个机场")
    elif args.self_test:
        _self_test()
    else:
        ap.print_help()
