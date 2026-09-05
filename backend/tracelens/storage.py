"""SQLite persistence for evidence, media resources, and Agent sessions."""
from __future__ import annotations
import hashlib
import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any
from fastapi import HTTPException
from .config import DB_PATH, UPLOADS_DIR
from .models import Evidence

def ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'USER',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS video_owners (
            video_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            start_seconds REAL NOT NULL,
            end_seconds REAL NOT NULL,
            text TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'ASR'
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'COMPLETED',
            ocr_status TEXT NOT NULL DEFAULT 'UNKNOWN',
            transcript_text TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_sessions (
            session_id TEXT PRIMARY KEY,
            video_id TEXT,
            title TEXT,
            summary TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            answerable INTEGER NOT NULL,
            support_level TEXT NOT NULL,
            report_json TEXT NOT NULL,
            trace_json TEXT,
            report_type TEXT NOT NULL DEFAULT 'INITIAL',
            parent_report_id INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS media_tasks (
            task_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'QUEUED',
            progress_current INTEGER NOT NULL DEFAULT 0,
            progress_total INTEGER NOT NULL DEFAULT 0,
            progress_message TEXT,
            result_json TEXT,
            error TEXT,
            question TEXT,
            session_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            finished_at TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Keep databases created by earlier versions compatible with the current schema.
    ensure_column(connection, "evidence", "source", "TEXT NOT NULL DEFAULT 'ASR'")
    ensure_column(connection, "videos", "ocr_status", "TEXT NOT NULL DEFAULT 'UNKNOWN'")
    ensure_column(connection, "agent_sessions", "title", "TEXT")
    ensure_column(connection, "agent_sessions", "summary", "TEXT")
    ensure_column(connection, "agent_reports", "report_type", "TEXT NOT NULL DEFAULT 'INITIAL'")
    ensure_column(connection, "agent_reports", "parent_report_id", "INTEGER")
    ensure_column(connection, "media_tasks", "question", "TEXT")
    ensure_column(connection, "media_tasks", "session_id", "TEXT")
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'USER',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS video_owners (
                video_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                start_seconds REAL NOT NULL,
                end_seconds REAL NOT NULL,
                text TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'ASR'
            )
        """
        )

        ensure_column(connection, "evidence", "source", "TEXT NOT NULL DEFAULT 'ASR'")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_sessions (
                session_id TEXT PRIMARY KEY,
                video_id TEXT,
                title TEXT,
                summary TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        ensure_column(connection, "agent_sessions", "summary", "TEXT")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                answerable INTEGER NOT NULL,
                support_level TEXT NOT NULL,
                report_json TEXT NOT NULL,
                trace_json TEXT,
                report_type TEXT NOT NULL DEFAULT 'INITIAL',
                parent_report_id INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS media_tasks (
                task_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'QUEUED',
                progress_current INTEGER NOT NULL DEFAULT 0,
                progress_total INTEGER NOT NULL DEFAULT 0,
                progress_message TEXT,
                result_json TEXT,
            error TEXT,
            question TEXT,
            session_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def create_user(username: str, password_hash: str) -> dict[str, Any]:
    with get_connection() as connection:
        try:
            cursor = connection.execute(
                "INSERT INTO users(username, password_hash) VALUES (?, ?)",
                (username, password_hash),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("username already exists") from error
        return {"id": cursor.lastrowid, "username": username, "role": "USER", "is_active": True}


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["is_active"] = bool(result.get("is_active"))
    return result


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["is_active"] = bool(result.get("is_active"))
    return result


def claim_video(video_id: str, user_id: int) -> None:
    with get_connection() as connection:
        row = connection.execute("SELECT user_id FROM video_owners WHERE video_id = ?", (video_id,)).fetchone()
        if row is None:
            connection.execute("INSERT INTO video_owners(video_id, user_id) VALUES (?, ?)", (video_id, user_id))
        elif int(row["user_id"]) != int(user_id):
            raise HTTPException(status_code=404, detail="video not found")


def user_owns_video(video_id: str, user_id: int) -> bool:
    with get_connection() as connection:
        row = connection.execute("SELECT user_id FROM video_owners WHERE video_id = ?", (video_id,)).fetchone()
    return row is not None and int(row["user_id"]) == int(user_id)


def create_media_task(
    video_id: str,
    task_type: str,
    *,
    question: str | None = None,
    session_id: str | None = None,
) -> str:
    task_id = uuid.uuid4().hex
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO media_tasks(task_id, video_id, task_type, question, session_id) VALUES (?, ?, ?, ?, ?)",
            (task_id, video_id, task_type.upper(), question, session_id),
        )
    return task_id


def update_media_task(task_id: str, **values: Any) -> None:
    allowed = {
        "state", "progress_current", "progress_total", "progress_message",
        "result_json", "error", "started_at", "finished_at",
    }
    values = {key: value for key, value in values.items() if key in allowed}
    if not values:
        return
    assignments = ", ".join(f"{key} = ?" for key in values)
    with get_connection() as connection:
        connection.execute(
            f"UPDATE media_tasks SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
            (*values.values(), task_id),
        )


def get_media_task(task_id: str) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM media_tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
    return dict(row) if row else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register_legacy_videos() -> None:
    """Reconcile upload directories with the persistent video resource table."""
    if not UPLOADS_DIR.is_dir():
        return
    with get_connection() as connection:
        for video_dir in UPLOADS_DIR.iterdir():
            if not video_dir.is_dir() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", video_dir.name):
                continue
            existing = connection.execute(
                "SELECT video_id, filename, stored_path, content_hash FROM videos WHERE video_id = ?",
                (video_dir.name,),
            ).fetchone()
            candidates = [
                path for path in video_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}
            ]
            if not candidates:
                continue
            path = max(candidates, key=lambda item: item.stat().st_mtime)
            digest = file_sha256(path)
            if existing is None:
                evidence_count = connection.execute(
                    "SELECT COUNT(*) FROM evidence WHERE video_id = ?", (video_dir.name,)
                ).fetchone()[0]
                connection.execute(
                    """
                    INSERT INTO videos(video_id, filename, stored_path, content_hash, status, transcript_text)
                    VALUES (?, ?, ?, ?, 'COMPLETED', ?)
                    """,
                    (
                        video_dir.name,
                        path.name,
                        str(path),
                        digest,
                        "\n".join(
                            row[0] for row in connection.execute(
                                "SELECT text FROM evidence WHERE video_id = ? ORDER BY start_seconds, id",
                                (video_dir.name,),
                            ).fetchall()
                        ) if evidence_count else None,
                    ),
                )
            elif (
                existing["filename"] != path.name
                or existing["stored_path"] != str(path)
                or existing["content_hash"] != digest
            ):
                connection.execute(
                    "UPDATE videos SET filename = ?, stored_path = ?, content_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE video_id = ?",
                    (path.name, str(path), digest, video_dir.name),
                )
def get_or_create_session(session_id: str | None, video_id: str | None) -> str:
    if session_id and not re.fullmatch(r"[A-Za-z0-9-]{8,100}", session_id):
        raise HTTPException(status_code=422, detail="invalid session_id")
    with get_connection() as connection:
        if session_id:
            row = connection.execute(
                "SELECT video_id FROM agent_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is not None:
                if row["video_id"] != video_id:
                    raise HTTPException(status_code=409, detail="session belongs to another video")
                return session_id
        new_session_id = uuid.uuid4().hex
        connection.execute(
            "INSERT INTO agent_sessions(session_id, video_id, title) VALUES (?, ?, ?)",
            (new_session_id, video_id, None),
        )
        return new_session_id


def get_session_history(session_id: str, limit: int = 12) -> list[dict[str, str]]:
    limit = max(2, min(int(limit), 20))
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT role, content FROM agent_messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def get_latest_agent_result(session_id: str) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT report_json FROM agent_reports WHERE session_id = ? AND report_type = 'INITIAL' "
            "ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    if not row or not row["report_json"]:
        return None
    try:
        value = json.loads(row["report_json"])
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def save_agent_turn(session_id: str, question: str, result: dict[str, Any]) -> None:
    # Every submitted question gets a durable record, including safe refusals
    # and provider failures. The UI can then explain what happened and users
    # can remove the record explicitly.
    answer = str(result.get("answer") or result.get("error") or "未生成回答").strip()
    report_type = "FOLLOW_UP" if result.get("kind") == "follow_up" else "INITIAL"
    with get_connection() as connection:
        parent = None
        if report_type == "FOLLOW_UP":
            parent = connection.execute(
                "SELECT id FROM agent_reports WHERE session_id = ? AND report_type = 'INITIAL' "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        connection.executemany(
            "INSERT INTO agent_messages(session_id, role, content) VALUES (?, ?, ?)",
            [(session_id, "user", question), (session_id, "assistant", answer)],
        )
        connection.execute(
            """
            INSERT INTO agent_reports(
                session_id, question, answer, answerable, support_level, report_json, trace_json,
                report_type, parent_report_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                question,
                answer,
                int(bool(result.get("grounded"))),
                str(result.get("support_level") or ("DIRECT" if result.get("grounded") else "INSUFFICIENT")),
                json.dumps(result, ensure_ascii=False),
                json.dumps(result.get("tool_trace") or result.get("trace") or [], ensure_ascii=False),
                report_type,
                parent["id"] if parent else None,
            ),
        )
        connection.execute(
            "UPDATE agent_sessions SET title = COALESCE(title, ?), updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (question[:80], session_id),
        )
        recent = connection.execute(
            "SELECT role, content FROM agent_messages WHERE session_id = ? ORDER BY id DESC LIMIT 12",
            (session_id,),
        ).fetchall()
        recent.reverse()
        summary = "\n".join(
            f"{item['role']}: {str(item['content'])[:1200]}"
            for item in recent
        )[-6000:]
        connection.execute(
            "UPDATE agent_sessions SET summary = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (summary, session_id),
        )


def get_session_summary(session_id: str) -> str:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT summary FROM agent_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return str(row["summary"] or "") if row else ""
