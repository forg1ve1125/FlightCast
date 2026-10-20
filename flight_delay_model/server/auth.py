# -*- coding: utf-8 -*-
"""
User accounts & query history.

Lives in its own SQLite file (server/app.db), deliberately separate from
flight_meta.db: build_meta_db.py deletes and rebuilds flight_meta.db on every
run, so anything stored there would be lost.

Passwords are hashed with PBKDF2-SHA256 (stdlib hashlib only, no new deps).
Only successful predictions are saved, and only for signed-in users —
guests can use every prediction feature without an account.
"""
import hashlib
import json
import re
import secrets
import sqlite3
import time
from pathlib import Path

APP_DB_PATH = Path(__file__).resolve().parent / "app.db"
PBKDF2_ITERATIONS = 240_000
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_conn(db_path=None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or APP_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path=None) -> sqlite3.Connection:
    conn = get_conn(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            salt          TEXT NOT NULL,
            created_at    TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS query_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            fl_date    TEXT NOT NULL,
            airline    TEXT NOT NULL,
            origin     TEXT NOT NULL,
            dest       TEXT NOT NULL,
            payload    TEXT NOT NULL,   -- JSON sent by the frontend
            result     TEXT NOT NULL    -- JSON returned by /api/predict (success only)
        );
        CREATE INDEX IF NOT EXISTS idx_history_user ON query_history(user_id, id DESC);
        CREATE TABLE IF NOT EXISTS login_sessions (
            token_hash TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            expires_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON login_sessions(expires_at);
        """
    )
    conn.commit()
    return conn


def hash_password(password: str, salt_hex: str = None):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS).hex()
    return digest, salt.hex()


def _user_row(conn, user_id):
    return conn.execute(
        "SELECT id, username, created_at FROM users WHERE id=?", (user_id,)).fetchone()


def register(conn, username: str, password: str):
    """Returns (user_dict, None) on success or (None, error_message)."""
    if not isinstance(username, str) or not isinstance(password, str):
        return None, "Username and password must be text."
    username = (username or "").strip()
    if not _USERNAME_RE.match(username):
        return None, ("Username must be 3-32 characters: letters, digits, "
                      "underscore, dot or hyphen.")
    if not password or len(password) < 6:
        return None, "Password must be at least 6 characters."
    try:
        digest, salt = hash_password(password)
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) "
            "VALUES (?, ?, ?, ?)", (username, digest, salt, _now()))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        return None, "USERNAME_TAKEN"
    row = _user_row(conn, cur.lastrowid)
    return {"id": row["id"], "username": row["username"],
            "created_at": row["created_at"]}, None


def login(conn, username: str, password: str):
    """Returns (user_dict, None) or (None, error_message)."""
    if not isinstance(username, str) or not isinstance(password, str):
        return None, "Invalid username or password."
    row = conn.execute(
        "SELECT id, username, created_at, password_hash, salt FROM users "
        "WHERE username=? COLLATE NOCASE", ((username or "").strip(),)).fetchone()
    if row:
        digest, _ = hash_password(password, row["salt"])
        if secrets.compare_digest(digest, row["password_hash"]):
            return {"id": row["id"], "username": row["username"],
                    "created_at": row["created_at"]}, None
    return None, "Invalid username or password."


def _token_hash(token):
    if not isinstance(token, str) or not token or len(token) > 128:
        return None
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(conn, user_id: int, lifetime_seconds: int = 31 * 24 * 60 * 60):
    """Store a revocable login; the database keeps only a hash of its random token."""
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    with conn:
        conn.execute("DELETE FROM login_sessions WHERE expires_at<=?", (now,))
        conn.execute(
            "INSERT INTO login_sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (_token_hash(token), user_id, _now(), now + lifetime_seconds))
    return token


def session_user(conn, token):
    """Return the current account only for an unexpired, unrevoked login."""
    digest = _token_hash(token)
    if digest is None:
        return None
    row = conn.execute(
        "SELECT u.id, u.username FROM login_sessions s JOIN users u ON u.id=s.user_id "
        "WHERE s.token_hash=? AND s.expires_at>?", (digest, int(time.time()))).fetchone()
    return dict(row) if row else None


def revoke_session(conn, token):
    digest = _token_hash(token)
    if digest is not None:
        with conn:
            conn.execute("DELETE FROM login_sessions WHERE token_hash=?", (digest,))


def save_history(conn, user_id: int, payload: dict, result: dict) -> int:
    """Store one successful prediction. Returns the new history id."""
    inp = result.get("input", payload)
    cur = conn.execute(
        "INSERT INTO query_history (user_id, created_at, fl_date, airline, origin, dest, "
        "payload, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, _now(), str(inp.get("FL_DATE", "")), str(inp.get("AIRLINE", "")),
         str(inp.get("ORIGIN", "")), str(inp.get("DEST", "")),
         json.dumps(payload, ensure_ascii=False), json.dumps(result, ensure_ascii=False)))
    conn.commit()
    return cur.lastrowid


def list_history(conn, user_id: int, limit: int = 100):
    """History list (newest first) with the headline fields for rendering."""
    rows = conn.execute(
        """SELECT h.id, h.created_at, h.fl_date, h.airline, h.origin, h.dest, h.result
           FROM query_history h WHERE h.user_id=? ORDER BY h.id DESC LIMIT ?""",
        (user_id, limit)).fetchall()
    items = []
    for r in rows:
        result = json.loads(r["result"])
        items.append({
            "id": r["id"], "created_at": r["created_at"], "fl_date": r["fl_date"],
            "airline": r["airline"], "origin": r["origin"], "dest": r["dest"],
            "verdicts": {k: v.get("phrase") for k, v in (result.get("verdicts") or {}).items()},
        })
    return items


def get_history_item(conn, user_id: int, item_id: int):
    """Full stored prediction for the owner only (or None)."""
    row = conn.execute(
        "SELECT id, created_at, payload, result FROM query_history "
        "WHERE id=? AND user_id=?", (item_id, user_id)).fetchone()
    if not row:
        return None
    return {"id": row["id"], "created_at": row["created_at"],
            "payload": json.loads(row["payload"]), "result": json.loads(row["result"])}


def get_history_item_shared(conn, item_id: int):
    """Full stored prediction regardless of owner (used only by tests)."""
    row = conn.execute(
        "SELECT id, user_id, created_at, payload, result FROM query_history WHERE id=?",
        (item_id,)).fetchone()
    if not row:
        return None
    return {"id": row["id"], "user_id": row["user_id"], "created_at": row["created_at"],
            "payload": json.loads(row["payload"]), "result": json.loads(row["result"])}
