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
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
        )
        """
    )
    # Keep databases created by earlier versions compatible with the current schema.
    ensure_column(connection, "evidence", "source", "TEXT NOT NULL DEFAULT 'ASR'")
    ensure_column(connection, "videos", "ocr_status", "TEXT NOT NULL DEFAULT 'UNKNOWN'")
    ensure_column(connection, "agent_sessions", "title", "TEXT")
    return connection


def init_db() -> None:
    with get_connection() as connection:
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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
            )
            """
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register_legacy_videos() -> None:
    """Register uploads created before the videos resource table existed."""
    if not UPLOADS_DIR.is_dir():
        return
    with get_connection() as connection:
        for video_dir in UPLOADS_DIR.iterdir():
            if not video_dir.is_dir() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", video_dir.name):
                continue
            existing = connection.execute(
                "SELECT 1 FROM videos WHERE video_id = ?", (video_dir.name,)
            ).fetchone()
            if existing:
                continue
            evidence_count = connection.execute(
                "SELECT COUNT(*) FROM evidence WHERE video_id = ?", (video_dir.name,)
            ).fetchone()[0]
            candidates = [
                path for path in video_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}
            ]
            if not candidates:
                continue
            path = max(candidates, key=lambda item: item.stat().st_mtime)
            connection.execute(
                """
                INSERT INTO videos(video_id, filename, stored_path, content_hash, status, transcript_text)
                VALUES (?, ?, ?, ?, 'COMPLETED', ?)
                """,
                (
                    video_dir.name,
                    path.name,
                    str(path),
                    file_sha256(path),
                    "\n".join(
                        row[0] for row in connection.execute(
                            "SELECT text FROM evidence WHERE video_id = ? ORDER BY start_seconds, id",
                            (video_dir.name,),
                        ).fetchall()
                    ) if evidence_count else None,
                ),
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


def save_agent_turn(session_id: str, question: str, result: dict[str, Any]) -> None:
    answer = str(result.get("answer") or "").strip()
    with get_connection() as connection:
        connection.executemany(
            "INSERT INTO agent_messages(session_id, role, content) VALUES (?, ?, ?)",
            [(session_id, "user", question), (session_id, "assistant", answer)],
        )
        connection.execute(
            """
            INSERT INTO agent_reports(
                session_id, question, answer, answerable, support_level, report_json, trace_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                question,
                answer,
                int(bool(result.get("grounded"))),
                str(result.get("support_level") or ("DIRECT" if result.get("grounded") else "INSUFFICIENT")),
                json.dumps(result, ensure_ascii=False),
                json.dumps(result.get("tool_trace") or result.get("trace") or [], ensure_ascii=False),
            ),
        )
        connection.execute(
            "UPDATE agent_sessions SET title = COALESCE(title, ?), updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (question[:80], session_id),
        )
