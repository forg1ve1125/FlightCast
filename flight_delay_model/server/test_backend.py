# -*- coding: utf-8 -*-
"""
Backend tests: metadata search, airline resolution, distance fallback,
argmax verdict, end-to-end prediction, auth + query history.
Run: python server/test_backend.py  (or pytest server/test_backend.py -v)
"""
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auth as auth_mod  # noqa: E402
from predictor_service import MetaDB, predict  # noqa: E402
from verdict import BUCKET_LABELS_EN, verdict, verdict_from_prob  # noqa: E402

db = MetaDB()


def test_airline_fuzzy():
    """Dropdown search: code DL/dl, partial name Delta/DELTA all hit Delta; full names resolve."""
    for q in ["DL", "dl", "Delta", "DELTA", "delta air"]:
        hits = db.search_airlines(q)
        assert any(h["code"] == "DL" for h in hits), f"'{q}' missed Delta: {hits[:3]}"
    assert db.resolve_airline("DL") == "Delta Air Lines Inc."
    assert db.resolve_airline("dl") == "Delta Air Lines Inc."
    assert db.resolve_airline("Delta Air Lines Inc.") == "Delta Air Lines Inc."
    assert db.resolve_airline("delta air lines inc.") == "Delta Air Lines Inc."
    assert db.resolve_airline("no such airline") is None


def test_airport_city_search():
    """City search: 'new york' finds JFK/LGA; EWR is Newark, NJ; raw codes resolve."""
    hits = db.search_airports("new york")
    codes = {h["code"] for h in hits}
    assert "JFK" in codes and "LGA" in codes, f"new york missing JFK/LGA: {codes}"
    ewr = db.get_airport("EWR")
    assert ewr and "Newark" in ewr["city"], f"EWR city info wrong: {ewr}"
    assert db.get_airport("atl")["code"] == "ATL"


def test_distance_fallback():
    """Distance fallback: known route -> route-history median; unknown -> airport/global median; user value wins."""
    d, src = db.fill_distance("ATL", "DEN")
    assert src == "route_history" and d > 0, (d, src)
    d, src = db.fill_distance("ATL", "DEN", user_distance=1234)
    assert src == "user" and d == 1234
    d, src = db.fill_distance("ATL", "XYZ")
    assert src in ("origin_median", "dest_median", "global_median") and d > 0, (d, src)


def _sc(probs):
    return [{"bucket": k, "label": BUCKET_LABELS_EN[k], "probability": p}
            for k, p in enumerate(probs)]


def test_verdict_argmax():
    """Pure argmax (user-confirmed rule): the phrase always follows the most probable
    scenario; level only colours the badge by the winning bucket's severity."""
    v = verdict("dep", _sc([0.52, 0.26, 0.12, 0.06, 0.04]), expected_minutes=5)
    assert v["level"] == "good" and v["phrase"] == "Expected on-time departure", v
    assert abs(v["p_delay"] - 0.48) < 1e-6 and v["expected_delay_min"] == 5
    v = verdict("dep", _sc([0.30, 0.45, 0.15, 0.07, 0.03]))
    assert v["level"] == "notice" and v["phrase"] == "Expected 1-15 min departure delay", v
    v = verdict("arr", _sc([0.20, 0.25, 0.30, 0.15, 0.10]))
    assert v["level"] == "warn" and v["phrase"] == "Expected 16-30 min arrival delay", v
    v = verdict("dep", _sc([0.10, 0.15, 0.20, 0.15, 0.40]))
    assert v["level"] == "alert" and v["phrase"] == "Expected over 60 min departure delay", v
    # probability tie -> the less severe bucket wins
    v = verdict("dep", _sc([0.5, 0.5, 0.0, 0.0, 0.0]))
    assert v["phrase"] == "Expected on-time departure", v
    # order must not matter (list comes sorted desc from the model layer)
    shuffled = _sc([0.10, 0.15, 0.20, 0.15, 0.40])[::-1]
    assert verdict("dep", shuffled)["phrase"] == "Expected over 60 min departure delay"
    # fallback with only the binary classifier available
    v = verdict_from_prob("arr", 0.70, expected_minutes=9)
    assert v["level"] == "warn" and v["phrase"] == "Expected arrival delay", v
    assert verdict_from_prob("dep", 0.30)["phrase"] == "Expected on-time departure"


def test_predict_end_to_end():
    """End-to-end: only required fields (no distance); backend fills distance, returns
    English bucket scenarios and argmax verdicts (departure-only scope since 2026-10-08);
    validation errors are English."""
    payload = {"FL_DATE": "2024-07-04", "AIRLINE": "DL", "ORIGIN": "ATL",
               "DEST": "DEN", "CRS_DEP_TIME": 1830, "CRS_ARR_TIME": 2015}
    result = predict(payload, db)
    if isinstance(result, tuple):
        raise AssertionError(f"prediction failed: {result[0]}")
    assert result["resolved"]["airline"] == "Delta Air Lines Inc."
    assert result["distance_source"] in ("route_history", "user")
    assert result["resolved"]["distance"] > 0
    assert "dep_delay" in result["scenarios"], "missing dep scenarios"
    assert "dep_delay" in result["verdicts"], "missing dep verdict"
    # 产品口径: 仅出发延误，不输出到达
    assert "arr_delay" not in result.get("scenarios", {})
    assert "arr_delay" not in result.get("verdicts", {})
    assert "arr_delay" not in result.get("predictions", {})
    items = result["scenarios"]["dep_delay"]["scenarios"]
    total = sum(s["probability"] for s in items)
    assert abs(total - 1.0) < 0.01
    assert all("bucket" in s and "label" in s for s in items), items
    assert any(s["bucket"] == 0 and s["label"] == "On time" for s in items)
    assert result["verdicts"]["dep_delay"]["phrase"].startswith("Expected"), result["verdicts"]["dep_delay"]
    # missing required field -> 400
    err = predict({"AIRLINE": "DL"}, db)
    assert isinstance(err, tuple) and "Missing required fields" in err[0]["error"]
    # unknown airport -> 400
    err = predict({**payload, "ORIGIN": "ZZZ"}, db)
    assert isinstance(err, tuple) and "airport" in err[0]["error"]
    # invalid time -> 400
    err = predict({**payload, "CRS_DEP_TIME": 2499}, db)
    assert isinstance(err, tuple) and "hhmm" in err[0]["error"]


def test_auth_and_history():
    """Register/login (PBKDF2), validation rejections, history save/list/get, owner scoping."""
    with tempfile.TemporaryDirectory() as td:
        conn = auth_mod.init_db(Path(td) / "app.db")
        try:
            user, err = auth_mod.register(conn, "alice", "secret123")
            assert err is None and user["username"] == "alice"
            # duplicate username is case-insensitive
            dup, err = auth_mod.register(conn, "ALICE", "secret123")
            assert dup is None and err == "USERNAME_TAKEN"
            # weak password / bad username rejected
            _, err = auth_mod.register(conn, "bob", "12345")
            assert err
            _, err = auth_mod.register(conn, "x", "123456")
            assert err
            # login ok / wrong password
            ok, err = auth_mod.login(conn, "alice", "secret123")
            assert err is None and ok["id"] == user["id"]
            bad, err = auth_mod.login(conn, "alice", "wrong-password")
            assert bad is None and err
            # password stored hashed, not plaintext
            row = conn.execute("SELECT password_hash FROM users WHERE id=?",
                               (user["id"],)).fetchone()
            assert "secret123" not in row["password_hash"]
            # history round-trip + verdict phrase surfaced in the list
            payload = {"FL_DATE": "2024-07-04", "AIRLINE": "DL", "ORIGIN": "ATL",
                       "DEST": "DEN", "CRS_DEP_TIME": 1830, "CRS_ARR_TIME": 2015}
            fake = {"input": {"FL_DATE": "2024-07-04", "AIRLINE": "Delta Air Lines Inc.",
                              "ORIGIN": "ATL", "DEST": "DEN"},
                    "verdicts": {"dep_delay": {"phrase": "Expected on-time departure"}}}
            hid = auth_mod.save_history(conn, user["id"], payload, fake)
            items = auth_mod.list_history(conn, user["id"])
            assert len(items) == 1 and items[0]["id"] == hid
            assert items[0]["origin"] == "ATL" and items[0]["dest"] == "DEN"
            assert items[0]["verdicts"]["dep_delay"] == "Expected on-time departure"
            item = auth_mod.get_history_item(conn, user["id"], hid)
            assert item["result"]["verdicts"]["dep_delay"]["phrase"] == "Expected on-time departure"
            assert item["payload"]["AIRLINE"] == "DL"
            # another user cannot read it
            other, err = auth_mod.register(conn, "carol", "secret123")
            assert err is None
            assert auth_mod.get_history_item(conn, other["id"], hid) is None
            assert auth_mod.list_history(conn, other["id"]) == []
        finally:
            conn.close()


ALL = [test_airline_fuzzy, test_airport_city_search, test_distance_fallback,
       test_verdict_argmax, test_predict_end_to_end, test_auth_and_history]

if __name__ == "__main__":
    passed = failed = 0
    for fn in ALL:
        print(f"[RUN ] {fn.__name__}: {fn.__doc__.strip().splitlines()[0]}")
        try:
            fn()
            print("[PASS]")
            passed += 1
        except Exception:
            traceback.print_exc()
            print("[FAIL]")
            failed += 1
    print(f"\nresult: {passed} passed, {failed} failed / {len(ALL)} total")
    sys.exit(1 if failed else 0)
