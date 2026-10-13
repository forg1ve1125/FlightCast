# -*- coding: utf-8 -*-
"""
构建前后端用的 SQLite 元数据库 flight_meta.db（数据来源：原始数据集，只读取不修改）。

三张表：
  airlines(code, name)                      航空公司 IATA 代码 <-> 全名（模型用全名）
  airports(code, city, state)               机场代码 <-> 所服务城市（如 JFK -> New York, NY）
  routes(origin, dest, dist_median, n)      航线的历史飞行距离中位数与航班数（距离兜底用）

用法：python server/build_meta_db.py
"""
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

SERVER_DIR = Path(__file__).resolve().parent
DB_PATH = SERVER_DIR / "flight_meta.db"
sys.path.insert(0, str(SERVER_DIR.parent))
import config  # noqa: E402


def main():
    t0 = time.time()
    src = config.ORIGINAL_CSV
    print(f"读取原始数据集（只读）: {src}")
    df = pd.read_csv(src, usecols=[
        "AIRLINE", "AIRLINE_CODE", "ORIGIN", "ORIGIN_CITY", "DEST", "DEST_CITY", "DISTANCE"],
        low_memory=False)

    # ---- 航空公司：全名 + IATA 代码 ----
    airlines = (df[["AIRLINE", "AIRLINE_CODE"]]
                .dropna().drop_duplicates()
                .rename(columns={"AIRLINE": "name", "AIRLINE_CODE": "code"}))
    # AIRLINE_DOT 形如 "United Air Lines Inc.: UA"，AIRLINE_CODE 是纯代码；确保 code 干净
    airlines["code"] = airlines["code"].astype(str).str.strip()
    airlines = airlines[airlines["code"].str.len() <= 4]
    print(f"航空公司: {len(airlines)} 家")

    # ---- 机场：代码 + 城市/州（"Fort Lauderdale, FL" -> city, state）----
    o = df[["ORIGIN", "ORIGIN_CITY"]].rename(columns={"ORIGIN": "code", "ORIGIN_CITY": "city"})
    d = df[["DEST", "DEST_CITY"]].rename(columns={"DEST": "code", "DEST_CITY": "city"})
    airports = pd.concat([o, d]).dropna().drop_duplicates(subset=["code"])
    # 同一机场可能出现多个城市写法，保留最常见的
    full = pd.concat([o, d]).dropna()
    top_city = (full.groupby(["code", "city"]).size().reset_index(name="n")
                .sort_values("n", ascending=False).drop_duplicates("code"))
    airports = top_city.set_index("code")[["city"]].reset_index()
    city_state = airports["city"].str.extract(r"^(?P<city>.+?),\s*(?P<state>[A-Z]{2})$")
    airports["city"] = city_state["city"].fillna(airports["city"])
    airports["state"] = city_state["state"].fillna("")
    print(f"机场: {len(airports)} 个")

    # ---- 航线距离中位数 ----
    routes = (df.dropna(subset=["DISTANCE"])
              .groupby(["ORIGIN", "DEST"])["DISTANCE"]
              .agg(dist_median="median", n="size").reset_index())
    routes["dist_median"] = routes["dist_median"].round(1)
    print(f"航线: {len(routes):,} 条（含历史距离中位数）")

    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as conn:
        airlines.to_sql("airlines", conn, index=False, if_exists="replace")
        airports.to_sql("airports", conn, index=False, if_exists="replace")
        routes.to_sql("routes", conn, index=False, if_exists="replace")
        conn.execute("CREATE INDEX idx_routes_od ON routes(origin, dest)")
        conn.execute("CREATE INDEX idx_airports_city ON airports(city)")
    print(f"完成: {DB_PATH} （{time.time()-t0:.1f}s）")


if __name__ == "__main__":
    main()
