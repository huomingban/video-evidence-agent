"""FastAPI application and HTTP route composition."""
from __future__ import annotations
import hashlib
import os
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from . import agent as agent_module
from .agent import (
    answer_from_evidence,
    evidence_citations,
    kimi_settings,
    parse_kimi_json,
    run_structured_kimi_agent as _run_structured_kimi_agent,
)
from .agent_graph import AGENT_GRAPH, AGENT_TOOL_NAMES
from .config import env_flag, logger
from . import ocr_runner
from .ocr_runner import extract_transcript_from_video, merge_transcript_chunks, resolve_ffmpeg_path
from .media import validate_video_filename, validate_video_id
from .models import AskIn, DemoSeedIn, EvidenceIn
from .retrieval import _QDRANT_CLIENT, ensure_qdrant_collection, get_qdrant_client, search_evidence, sync_evidence_to_qdrant
from .storage import DB_PATH, UPLOADS_DIR, get_connection, get_or_create_session, get_session_history, init_db, register_legacy_videos, save_agent_turn

OpenAI = agent_module.OpenAI


def kimi_is_configured() -> bool:
    agent_module.OpenAI = OpenAI
    return agent_module.kimi_is_configured()


def run_kimi_agent(question: str, video_id: str | None, history: list[dict[str, str]] | None = None) -> dict[str, Any] | None:
    agent_module.OpenAI = OpenAI
    if os.getenv("KIMI_AGENT_WORKFLOW", "structured").strip().lower() in {
        "structured",
        "planner-retriever-verifier-writer-critic",
    }:
        structured_result = _run_structured_kimi_agent(question, video_id)
        if structured_result is not None:
            return structured_result
    return agent_module.run_kimi_agent(question, video_id, history=history)

app = FastAPI(title="VideoEvidence Agent", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup() -> None:
    init_db()
    register_legacy_videos()
    ensure_qdrant_collection()


@app.on_event("shutdown")
def shutdown() -> None:
    if _QDRANT_CLIENT is not None:
        try:
            _QDRANT_CLIENT.close()
        except Exception:
            logger.exception("Unable to close Qdrant")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "video-evidence-agent"}


@app.get("/api/evidence")
def list_evidence(video_id: str | None = None) -> dict[str, Any]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, video_id, start_seconds, end_seconds, text FROM evidence "
            "WHERE (? IS NULL OR video_id = ?) ORDER BY start_seconds, id",
            (video_id, video_id),
        ).fetchall()
    return {"items": [dict(row) for row in rows]}


@app.post("/api/evidence")
def create_evidence(payload: EvidenceIn) -> dict[str, Any]:
    if payload.end_seconds <= payload.start_seconds:
        raise HTTPException(status_code=422, detail="end_seconds must be greater than start_seconds")
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO evidence(video_id, start_seconds, end_seconds, text) VALUES (?, ?, ?, ?)",
            (payload.video_id, payload.start_seconds, payload.end_seconds, payload.text),
        )
        evidence_id = cursor.lastrowid
    sync_evidence_to_qdrant(payload.video_id)
    return {"id": evidence_id, **payload.model_dump()}


@app.post("/api/demo/seed")
def seed_demo(payload: DemoSeedIn) -> dict[str, Any]:
    demo_items = [
        (0, 42, "视频先介绍了项目式学习：先做一个能运行的小项目，再围绕项目补齐知识。"),
        (48, 96, "视频提到面试准备要围绕自己的项目，重点讲清楚数据流、技术选型和故障排查。"),
        (105, 158, "视频总结了证据链：视频转写得到带时间戳的文本，检索后再生成带引用的回答。"),
    ]
    with get_connection() as connection:
        connection.execute("DELETE FROM evidence WHERE video_id = ?", (payload.video_id,))
        connection.executemany(
            "INSERT INTO evidence(video_id, start_seconds, end_seconds, text) VALUES (?, ?, ?, ?)",
            [(payload.video_id, start, end, text) for start, end, text in demo_items],
        )
    sync_evidence_to_qdrant(payload.video_id)
    return {"video_id": payload.video_id, "seeded": len(demo_items)}


@app.get("/api/videos")
def list_videos() -> dict[str, Any]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT v.video_id, v.filename, v.content_hash, v.status, v.created_at, v.updated_at, "
            "COUNT(e.id) AS evidence_count "
            "FROM videos v LEFT JOIN evidence e ON e.video_id = v.video_id "
            "GROUP BY v.video_id ORDER BY v.updated_at DESC, v.video_id"
        ).fetchall()
    return {"items": [dict(row) for row in rows]}


@app.post("/api/videos/upload")
async def upload_video(
    video_id: str = Form(..., min_length=1),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    video_id = validate_video_id(video_id)
    if not file.filename:
        raise HTTPException(status_code=400, detail="file name is required")

    safe_name = validate_video_filename(file.filename)
    video_dir = UPLOADS_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    stored_path = video_dir / safe_name
    temporary_path = video_dir / f".{uuid.uuid4().hex}.uploading"
    max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
    total_bytes = 0
    digest = hashlib.sha256()
    try:
        with temporary_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_upload_bytes:
                    raise HTTPException(status_code=413, detail="video file is too large")
                digest.update(chunk)
                output.write(chunk)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    content_hash = digest.hexdigest()
    existing: sqlite3.Row | None = None
    with get_connection() as connection:
        existing = connection.execute(
            "SELECT video_id, filename, stored_path, content_hash FROM videos WHERE video_id = ?",
            (video_id,),
        ).fetchone()
        evidence_count = connection.execute(
            "SELECT COUNT(*) FROM evidence WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
    if (
        existing is not None
        and existing["content_hash"] == content_hash
        and evidence_count > 0
        and Path(existing["stored_path"]).is_file()
    ):
        temporary_path.unlink(missing_ok=True)
        return {
            "video_id": video_id,
            "filename": existing["filename"],
            "stored_path": existing["stored_path"],
            "evidence_count": evidence_count,
            "status": "already_processed",
            "deduplicated": True,
        }

    try:
        transcript = await run_in_threadpool(extract_transcript_from_video, temporary_path, safe_name)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(error)) from error
    if not transcript:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="no speech transcript was produced")

    old_path = Path(existing["stored_path"]) if existing is not None else None
    temporary_path.replace(stored_path)
    with get_connection() as connection:
        connection.execute("DELETE FROM evidence WHERE video_id = ?", (video_id,))
        connection.executemany(
            "INSERT INTO evidence(video_id, start_seconds, end_seconds, text) VALUES (?, ?, ?, ?)",
            [(video_id, start, end, text) for start, end, text in transcript],
        )
        connection.execute(
            """
            INSERT INTO videos(video_id, filename, stored_path, content_hash, status, transcript_text)
            VALUES (?, ?, ?, ?, 'COMPLETED', ?)
            ON CONFLICT(video_id) DO UPDATE SET
                filename = excluded.filename,
                stored_path = excluded.stored_path,
                content_hash = excluded.content_hash,
                status = excluded.status,
                transcript_text = excluded.transcript_text,
                updated_at = CURRENT_TIMESTAMP
            """,
            (video_id, safe_name, str(stored_path), content_hash, "\n".join(item[2] for item in transcript)),
        )
    if old_path is not None and old_path != stored_path:
        old_path.unlink(missing_ok=True)
    for path in video_dir.iterdir():
        if path.is_file() and path != stored_path and not path.name.endswith(".uploading"):
            path.unlink(missing_ok=True)
    await run_in_threadpool(sync_evidence_to_qdrant, video_id)

    return {
        "video_id": video_id,
        "filename": safe_name,
        "stored_path": str(stored_path),
        "evidence_count": len(transcript),
        "status": "uploaded",
        "deduplicated": False,
    }


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str) -> dict[str, Any]:
    video_id = validate_video_id(video_id)
    video_dir = UPLOADS_DIR / video_id
    with get_connection() as connection:
        row = connection.execute(
            "SELECT filename FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        evidence_count = connection.execute(
            "SELECT COUNT(*) FROM evidence WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
        if row is None and evidence_count == 0 and not video_dir.exists():
            raise HTTPException(status_code=404, detail="video not found")
        connection.execute("DELETE FROM evidence WHERE video_id = ?", (video_id,))
        connection.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        session_rows = connection.execute(
            "SELECT session_id FROM agent_sessions WHERE video_id = ?", (video_id,)
        ).fetchall()
        session_ids = [row["session_id"] for row in session_rows]
        if session_ids:
            placeholders = ",".join("?" for _ in session_ids)
            connection.execute(
                f"DELETE FROM agent_messages WHERE session_id IN ({placeholders})", session_ids
            )
            connection.execute(
                f"DELETE FROM agent_reports WHERE session_id IN ({placeholders})", session_ids
            )
            connection.execute(
                f"DELETE FROM agent_sessions WHERE session_id IN ({placeholders})", session_ids
            )
    shutil.rmtree(video_dir, ignore_errors=True)
    sync_evidence_to_qdrant(video_id)
    return {
        "video_id": video_id,
        "deleted": True,
        "filename": row["filename"] if row is not None else None,
        "evidence_count": evidence_count,
    }


@app.get("/api/videos/{video_id}/memory")
def get_video_memory(video_id: str) -> dict[str, Any]:
    video_id = validate_video_id(video_id)
    with get_connection() as connection:
        sessions = connection.execute(
            "SELECT session_id, title, created_at, updated_at FROM agent_sessions "
            "WHERE video_id = ? ORDER BY updated_at DESC, session_id DESC",
            (video_id,),
        ).fetchall()
        result = []
        for session in sessions:
            messages = connection.execute(
                "SELECT role, content, created_at FROM agent_messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT 200",
                (session["session_id"],),
            ).fetchall()
            reports = connection.execute(
                "SELECT id, question, answer, answerable, support_level, report_json, created_at "
                "FROM agent_reports WHERE session_id = ? ORDER BY id DESC LIMIT 50",
                (session["session_id"],),
            ).fetchall()
            result.append({
                "session_id": session["session_id"],
                "title": session["title"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "messages": [dict(item) for item in reversed(messages)],
                "reports": [
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "answerable": bool(item["answerable"]),
                        "support_level": item["support_level"],
                        "created_at": item["created_at"],
                        "report": parse_kimi_json(item["report_json"]) or {},
                    }
                    for item in reversed(reports)
                ],
            })
    return {
        "video_id": video_id,
        "latest_session_id": result[0]["session_id"] if result else None,
        "sessions": result,
    }


@app.post("/api/ask")
def ask(payload: AskIn) -> dict[str, Any]:
    session_id = get_or_create_session(payload.session_id, payload.video_id)
    history = get_session_history(session_id)
    if kimi_is_configured():
        agent_result = run_kimi_agent(payload.question, payload.video_id, history=history)
        if agent_result is not None:
            agent_result["session_id"] = session_id
            save_agent_turn(session_id, payload.question, agent_result)
            return agent_result
        result = {
            "question": payload.question,
            "answer": "Kimi Agent 暂时无法完成证据检索，请检查 API Key、模型名称和网络连接。",
            "grounded": False,
            "citations": [],
            "provider": "Kimi Agent error",
            "trace": ["Kimi Agent failed before submitting an accepted report"],
            "session_id": session_id,
        }
        save_agent_turn(session_id, payload.question, result)
        return result

    if AGENT_GRAPH is not None:
        try:
            state = AGENT_GRAPH.invoke({
                "question": payload.question,
                "video_id": payload.video_id,
                "evidence": [],
                "answer": "",
                "trace": [],
                "grounded": False,
                "citations": [],
                "provider": "",
                "adequate": False,
            })
            result = {
                "question": payload.question,
                "answer": state.get("answer", "当前无法生成回答。"),
                "grounded": bool(state.get("grounded", False)),
                "citations": state.get("citations", []),
                "provider": state.get("provider", ""),
                "trace": state.get("trace", []),
                "session_id": session_id,
            }
            save_agent_turn(session_id, payload.question, result)
            return result
        except Exception:
            logger.exception("Agent graph failed")

    evidence = search_evidence(payload.question, payload.video_id)
    result = {
        "question": payload.question,
        "trace": ["Fallback: Using direct search (LangGraph unavailable)"],
        **answer_from_evidence(payload.question, evidence),
        "session_id": session_id,
    }
    save_agent_turn(session_id, payload.question, result)
    return result


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    """Get Agent performance metrics and statistics"""
    with get_connection() as connection:
        total_evidence = connection.execute(
            "SELECT COUNT(*) as count FROM evidence"
        ).fetchone()["count"]
        
        video_count = connection.execute(
            "SELECT COUNT(DISTINCT video_id) as count FROM evidence"
        ).fetchone()["count"]
        
        avg_evidence_per_video = (
            total_evidence / video_count if video_count > 0 else 0
        )
    
    qdrant_active = get_qdrant_client() is not None
    embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_enabled = env_flag("EMBEDDING_ENABLED", False)
    
    return {
        "agent_version": "0.5.0",
        "capabilities": [
            "vector_retrieval",
            "keyword_retrieval",
            "evidence_verification",
            "answer_grounding",
            "citation_tracking",
            "agent_tracing",
            "tool_calling",
            "structured_planning",
            "critic_revision",
        ],
        "statistics": {
            "total_evidence": total_evidence,
            "distinct_videos": video_count,
            "avg_evidence_per_video": round(avg_evidence_per_video, 2),
        },
        "components": {
            "vector_database": "qdrant" if qdrant_active and embedding_enabled else "disabled",
            "embedding_model": embedding_model if embedding_enabled else "disabled",
            "agent_framework": "langgraph" if AGENT_GRAPH is not None else "fallback",
            "agent_workflow": os.getenv("KIMI_AGENT_WORKFLOW", "structured"),
            "transcription": "faster-whisper" if ocr_runner._WHISPER_MODEL is not None else "available on upload",
            "video_processing": "ffmpeg" if resolve_ffmpeg_path() else "unavailable",
            "llm": {
                "provider": "Kimi" if kimi_is_configured() else "local fallback",
                "configured": kimi_is_configured(),
                "model": kimi_settings()["model"],
            },
        },
        "available_tools": AGENT_TOOL_NAMES,
    }
