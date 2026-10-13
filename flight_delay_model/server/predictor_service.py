# -*- coding: utf-8 -*-
"""
Prediction service: input validation -> airline resolution -> distance
fallback -> model call -> assemble result + English verdict phrases.

Distance fallback when DISTANCE is not provided (in priority order):
  1) route-history median (routes table)
  2) median distance of all routes from the origin airport
  3) median distance of all routes to the destination airport
  4) global median
`distance_source` in the response says which level was used.

All user-facing text is English (project requirement). The model layer
(04_predict.py) still produces Chinese bucket labels from config.BUCKET_LABELS;
this module translates them at the API boundary.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

SERVER_DIR = Path(__file__).resolve().parent
DB_PATH = SERVER_DIR / "flight_meta.db"
sys.path.insert(0, str(SERVER_DIR.parent))

import config  # noqa: E402
from verdict import BUCKET_LABELS_EN, verdict, verdict_from_prob  # noqa: E402

from importlib import import_module  # noqa: E402

predict_mod = import_module("04_predict")

import weather_serving  # noqa: E402  (Open-Meteo 天气模型服务，可渐进启用)

# config.BUCKET_LABELS (Chinese, model-side) -> bucket index
_LABEL_TO_BUCKET = {label: k for k, label in enumerate(config.BUCKET_LABELS)}


class MetaDB:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def resolve_airline(self, q: str):
        """Accept an IATA code (DL/dl) or the full airline name (any case). Returns the canonical name or None."""
        q = q.strip()
        if not q:
            return None
        row = self.conn.execute(
            "SELECT name FROM airlines WHERE UPPER(code)=UPPER(?)", (q,)).fetchone()
        if row:
            return row["name"]
        row = self.conn.execute(
            "SELECT name FROM airlines WHERE name=? COLLATE NOCASE", (q,)).fetchone()
        return row["name"] if row else None

    def search_airlines(self, q: str, limit=20):
        """Dropdown search: code or name substring (case-insensitive), exact code first."""
        like = f"%{q}%"
        rows = self.conn.execute(
            """SELECT code, name FROM airlines
               WHERE UPPER(code) LIKE UPPER(?) OR name LIKE ? COLLATE NOCASE
               ORDER BY (UPPER(code)=UPPER(?)) DESC, LENGTH(name), name LIMIT ?""",
            (like, like, q, limit)).fetchall()
        return [dict(r) for r in rows]

    def search_airports(self, q: str, limit=20):
        """Dropdown search: code or city substring (case-insensitive), e.g. 'new york' -> JFK/LGA."""
        like = f"%{q}%"
        rows = self.conn.execute(
            """SELECT code, city, state FROM airports
               WHERE UPPER(code) LIKE UPPER(?) OR city LIKE ? COLLATE NOCASE
               ORDER BY (UPPER(code)=UPPER(?)) DESC, code LIMIT ?""",
            (like, like, q, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_airport(self, code: str):
        row = self.conn.execute(
            "SELECT code, city, state FROM airports WHERE UPPER(code)=UPPER(?)",
            (code.strip(),)).fetchone()
        return dict(row) if row else None

    def fill_distance(self, origin: str, dest: str, user_distance=None):
        """Returns (distance, source). source in user/route_history/origin_median/dest_median/global_median"""
        if user_distance is not None and float(user_distance) > 0:
            return round(float(user_distance), 1), "user"
        row = self.conn.execute(
            "SELECT dist_median FROM routes WHERE origin=? AND dest=?",
            (origin, dest)).fetchone()
        if row and row["dist_median"]:
            return round(float(row["dist_median"]), 1), "route_history"
        # sqlite3 has no MEDIAN: sort in Python and take the middle (origin -> dest -> global)
        for key, src in [(origin, "origin_median"), (dest, "dest_median")]:
            col = "origin" if src == "origin_median" else "dest"
            rows = self.conn.execute(
                f"SELECT dist_median FROM routes WHERE {col}=? ORDER BY dist_median", (key,)).fetchall()
            if rows:
                vals = [r["dist_median"] for r in rows if r["dist_median"]]
                if vals:
                    mid = vals[len(vals) // 2]
                    return round(float(mid), 1), src
        rows = self.conn.execute(
            "SELECT dist_median FROM routes ORDER BY dist_median").fetchall()
        vals = [r["dist_median"] for r in rows if r["dist_median"]]
        return round(float(vals[len(vals) // 2]), 1), "global_median"


def _valid_hhmm(v):
    try:
        n = int(v)
    except (TypeError, ValueError):
        return False
    if not (0 <= n <= 2359):
        return False
    hh, mm = divmod(n, 100)
    return 0 <= hh <= 23 and 0 <= mm <= 59


def _scenarios_to_english(seg_scenarios: dict) -> dict:
    """Translate one segment's bucket scenario block to English display keys.

    Model side (04_predict.py): {scenarios: [{scenario: <zh label>, probability,
    typical_minutes}], expected_delay_min_from_buckets, summary}.
    API side: {scenarios: [{bucket, label, probability, typical_minutes}], ...}
    with an English `summary`.
    """
    items = []
    for s in seg_scenarios["scenarios"]:
        k = _LABEL_TO_BUCKET.get(s["scenario"])
        if k is None:            # unknown label: keep as-is, no bucket index
            k = -1
            label = s["scenario"]
        else:
            label = BUCKET_LABELS_EN[k]
        items.append({"bucket": k, "label": label,
                      "probability": s["probability"],
                      "typical_minutes": s.get("typical_minutes")})
    parts = []
    for s in items:
        if s["bucket"] == 0:
            parts.append(f"{s['probability']:.0%} on time")
        else:
            mins = (f" (~{s['typical_minutes']:.0f} min)"
                    if s.get("typical_minutes") is not None else "")
            parts.append(f"{s['probability']:.0%} {s['label'].lower()}{mins}")
    return {"scenarios": items,
            "expected_delay_min_from_buckets": seg_scenarios["expected_delay_min_from_buckets"],
            "summary": " · ".join(parts)}


def _predictions_to_english(result: dict) -> None:
    """Rewrite the two-stage model summaries in English (in place)."""
    seg_word = {"dep_delay": "departure", "arr_delay": "arrival"}
    for key, word in seg_word.items():
        b = result.get("predictions", {}).get(key)
        if not b or "summary" not in b:
            continue
        p, mins = b.get("prob_delay"), b.get("expected_minutes_if_delayed")
        if p is not None and mins is not None:
            b["summary"] = (f"{p*100:.1f}% chance of {word} delay; "
                            f"if delayed, about {mins:.0f} min; "
                            f"expected delay {p*mins:.0f} min")


def predict(payload: dict, db: MetaDB) -> dict:
    """Main entry. payload: FL_DATE/AIRLINE/ORIGIN/DEST/CRS_DEP_TIME/CRS_ARR_TIME/DISTANCE?"""
    required = ["FL_DATE", "AIRLINE", "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME"]
    missing = [k for k in required if not str(payload.get(k, "")).strip()]
    if missing:
        return {"error": f"Missing required fields: {missing}"}, 400

    airline = db.resolve_airline(str(payload["AIRLINE"]))
    if not airline:
        return {"error": f"Unrecognised airline '{payload['AIRLINE']}'. "
                         f"Search via /api/airlines (code or partial name)."}, 400

    origin = str(payload["ORIGIN"]).strip().upper()
    dest = str(payload["DEST"]).strip().upper()
    for code, label in [(origin, "ORIGIN"), (dest, "DEST")]:
        if not db.get_airport(code):
            return {"error": f"Unrecognised airport '{code}' ({label}). "
                             f"Search via /api/airports (airport code or city name)."}, 400

    if not _valid_hhmm(payload["CRS_DEP_TIME"]) or not _valid_hhmm(payload["CRS_ARR_TIME"]):
        return {"error": "Scheduled times must be hhmm integers (0000-2359)."}, 400

    try:
        fl_date = pd.Timestamp(str(payload["FL_DATE"]))
        if pd.isna(fl_date):
            raise ValueError
    except ValueError:
        return {"error": "FL_DATE must be a valid date (YYYY-MM-DD)."}, 400

    distance, dist_source = db.fill_distance(
        origin, dest, payload.get("DISTANCE") or None)

    flight = {
        "FL_DATE": fl_date.strftime("%Y-%m-%d"),
        "AIRLINE": airline,
        "ORIGIN": origin,
        "DEST": dest,
        "CRS_DEP_TIME": int(payload["CRS_DEP_TIME"]),
        "CRS_ARR_TIME": int(payload["CRS_ARR_TIME"]),
        "DISTANCE": float(distance),
    }
    result = predict_mod.predict_flight(None, flight)
    result["distance_source"] = dist_source
    result["resolved"] = {"airline": airline, "origin": origin, "dest": dest,
                          "distance": distance}

    # English display for the bucket scenarios + two-stage summaries
    if "scenarios" in result:
        result["scenarios"] = {k: _scenarios_to_english(v)
                               for k, v in result["scenarios"].items()}
    _predictions_to_english(result)

    # 产品口径(2026-10-08, 导师建议聚焦单一任务): 仅输出出发延误预测。
    # 模型存档仍保留到达模型(训练能力不变), 在服务层过滤, 不对外输出。
    # CRS_ARR_TIME 仍是必填输入: 用于唯一确定航班 + 块时长类特征原料。
    if "predictions" in result:
        result["predictions"] = {k: v for k, v in result["predictions"].items()
                                 if k == "dep_delay"}
    if "scenarios" in result:
        result["scenarios"] = {k: v for k, v in result["scenarios"].items()
                               if k == "dep_delay"}

    # 天气模型叠加（Open-Meteo 预报 -> 与训练同构的天气特征 -> 11/12/13 号存档模型）。
    # 必须放在英文翻译与仅出发过滤之后：其输出已自带英文结构、只覆盖 dep 组件。
    # 组件在 models/wx_serving.json 中逐个启用；任何失败保持 v1 基线结果。
    if weather_serving.enabled():
        try:
            result = weather_serving.predict_with_weather(flight, result)
        except Exception:
            pass

    # Verdict: pure argmax over the bucket scenarios (better calibrated);
    # fall back to the binary classifier probability when buckets are absent.
    result["verdicts"] = {}
    for seg in ["dep"]:
        key = f"{seg}_delay"
        if key in result.get("scenarios", {}):
            exp = result["scenarios"][key].get("expected_delay_min_from_buckets")
            result["verdicts"][key] = verdict(seg, result["scenarios"][key]["scenarios"], exp)
        elif key in result.get("predictions", {}):
            p_delay = result["predictions"][key].get("prob_delay")
            exp = result["predictions"][key].get("expected_minutes_overall")
            result["verdicts"][key] = verdict_from_prob(seg, p_delay, exp)
    return result
