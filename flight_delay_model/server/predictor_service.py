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

The HTTP service runs only the calibrated five-bucket departure model.
Archived models remain on disk and are never loaded by this entry point.
"""
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

SERVER_DIR = Path(__file__).resolve().parent
DB_PATH = SERVER_DIR / "flight_meta.db"
sys.path.insert(0, str(SERVER_DIR.parent))

from verdict import verdict  # noqa: E402

import weather_serving  # noqa: E402  (Open-Meteo 天气模型服务，可渐进启用)


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
    try:
        result = weather_serving.predict_bucket_only(flight)
    except Exception:
        # Missing/broken configuration must never reactivate archived models.
        logging.getLogger(__name__).exception("Calibrated departure model unavailable")
        return {"error": "The calibrated departure model is unavailable. Please try again later."}, 503
    result["distance_source"] = dist_source
    result["resolved"] = {"airline": airline, "origin": origin, "dest": dest,
                          "distance": distance}

    # Verdict and all visible statistics share the calibrated bucket distribution.
    result["verdicts"] = {}
    key = "dep_delay"
    if key in result.get("scenarios", {}):
        exp = result["scenarios"][key].get("expected_delay_min_from_buckets")
        result["verdicts"][key] = verdict("dep", result["scenarios"][key]["scenarios"], exp)
    return result
