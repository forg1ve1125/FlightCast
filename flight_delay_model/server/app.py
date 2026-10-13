# -*- coding: utf-8 -*-
"""
Flask backend:
  GET  /api/airlines?q=delta      airline dropdown search (code/name, case-insensitive)
  GET  /api/airports?q=new york   airport dropdown search (code/city, case-insensitive)
  POST /api/predict               prediction (required FL_DATE/AIRLINE/ORIGIN/DEST/
                                  CRS_DEP_TIME/CRS_ARR_TIME, DISTANCE optional; the
                                  backend resolves airline codes, falls back for
                                  distance, attaches English verdict phrases).
                                  Saved to the signed-in user's history on success.
  POST /api/auth/register         {username, password} -> create account + sign in
  POST /api/auth/login            {username, password} -> sign in
  POST /api/auth/logout           sign out
  GET  /api/auth/me               current session user (or null)
  GET  /api/history               list saved predictions (signed-in users only)
  GET  /api/history/<id>          one stored prediction (owner only)
  GET  /                          serves frontend/dist in production

Guests can use every prediction feature without an account (nothing is saved).

Run: python server/app.py   (default http://127.0.0.1:5000)
The vite dev server (port 5173) proxies /api to this service.
"""
import secrets
from pathlib import Path

from flask import Flask, g, jsonify, request, send_from_directory, session

import auth
from predictor_service import MetaDB, predict

app = Flask(__name__, static_folder=None)

# Persisted random secret so sessions survive restarts.
_SECRET_FILE = Path(__file__).resolve().parent / ".secret_key"
if _SECRET_FILE.exists():
    app.secret_key = _SECRET_FILE.read_text().strip()
else:
    app.secret_key = secrets.token_hex(32)
    _SECRET_FILE.write_text(app.secret_key)

db = MetaDB()
app_db = auth.init_db()

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@app.before_request
def _load_user():
    g.user = None
    uid = session.get("user_id")
    if uid is not None:
        row = app_db.execute(
            "SELECT id, username FROM users WHERE id=?", (uid,)).fetchone()
        if row:
            g.user = {"id": row["id"], "username": row["username"]}


@app.get("/api/airlines")
def api_airlines():
    q = request.args.get("q", "").strip()
    if not q:
        rows = db.conn.execute("SELECT code, name FROM airlines ORDER BY name").fetchall()
        return jsonify([dict(r) for r in rows])
    return jsonify(db.search_airlines(q))


@app.get("/api/airports")
def api_airports():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Please provide q (airport code or city name)."}), 400
    return jsonify(db.search_airports(q))


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    outcome = predict(payload, db)
    if isinstance(outcome, tuple):          # (error_dict, 400)
        return jsonify(outcome[0]), 400
    if g.user:                              # save successful predictions only
        auth.save_history(app_db, g.user["id"], payload, outcome)
    return jsonify(outcome)


@app.post("/api/auth/register")
def api_register():
    body = request.get_json(silent=True) or {}
    user, err = auth.register(app_db, body.get("username", ""), body.get("password", ""))
    if err == "USERNAME_TAKEN":
        return jsonify({"error": "That username is already taken."}), 409
    if err:
        return jsonify({"error": err}), 400
    session["user_id"] = user["id"]
    return jsonify({"user": {"id": user["id"], "username": user["username"]}}), 201


@app.post("/api/auth/login")
def api_login():
    body = request.get_json(silent=True) or {}
    user, err = auth.login(app_db, body.get("username", ""), body.get("password", ""))
    if err:
        return jsonify({"error": err}), 401
    session["user_id"] = user["id"]
    return jsonify({"user": {"id": user["id"], "username": user["username"]}})


@app.post("/api/auth/logout")
def api_logout():
    session.pop("user_id", None)
    return jsonify({"ok": True})


@app.get("/api/auth/me")
def api_me():
    return jsonify({"user": g.user})


@app.get("/api/history")
def api_history():
    if not g.user:
        return jsonify({"error": "Sign in to view your prediction history."}), 401
    return jsonify({"items": auth.list_history(app_db, g.user["id"])})


@app.get("/api/history/<int:item_id>")
def api_history_item(item_id):
    if not g.user:
        return jsonify({"error": "Sign in to view your prediction history."}), 401
    item = auth.get_history_item(app_db, g.user["id"], item_id)
    if not item:
        return jsonify({"error": "Prediction not found."}), 404
    return jsonify(item)


@app.get("/")
def index():
    if (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({"message": "Frontend not built yet (frontend/dist missing). "
                               "Call the /api/* endpoints directly, or run "
                               "`npm run build` inside frontend/."})


@app.get("/<path:filename>")
def static_files(filename):
    target = FRONTEND_DIST / filename
    if target.exists() and target.is_file():
        return send_from_directory(FRONTEND_DIST, filename)
    return jsonify({"error": "not found"}), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
