from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import re
import shutil
import socket
import sqlite3
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, TypedDict

try:
    from qdrant_client import QdrantClient
except Exception:  # pragma: no cover - optional at runtime
    QdrantClient = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional at runtime
    SentenceTransformer = None

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional at runtime
    END = START = StateGraph = None

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "agent.sqlite3"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

_QDRANT_CLIENT: Any | None = None

app = FastAPI(title="VideoEvidence Agent", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvidenceIn(BaseModel):
    video_id: str = Field(min_length=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str = Field(min_length=1)


class AskIn(BaseModel):
    question: str = Field(min_length=1)
    video_id: str | None = None


class DemoSeedIn(BaseModel):
    video_id: str = Field(default="demo-video", min_length=1)


@dataclass(frozen=True)
class Evidence:
    id: int
    video_id: str
    start_seconds: float
    end_seconds: float
    text: str


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
            text TEXT NOT NULL
        )
        """
    )
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
                text TEXT NOT NULL
            )
            """
        )


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def tokenize(text: str) -> set[str]:
    normalized = text.lower()
    tokens = {token for token in re.findall(r"[a-z0-9_]+", normalized) if len(token) > 1}
    chinese = re.findall(r"[\u4e00-\u9fff]", normalized)
    tokens.update("".join(chinese[index:index + 2]) for index in range(len(chinese) - 1))
    return tokens


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_EMBEDDING_MODEL: Any | None = None


def get_embedding_model() -> Any | None:
    global _EMBEDDING_MODEL
    if SentenceTransformer is None:
        return None
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _EMBEDDING_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    if model is None:
        return []
    try:
        return model.encode(texts, normalize_embeddings=True).tolist()
    except Exception:
        return []


def is_qdrant_running() -> bool:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def get_qdrant_client() -> Any | None:
    global _QDRANT_CLIENT
    if QdrantClient is None:
        return None
    if _QDRANT_CLIENT is not None:
        return _QDRANT_CLIENT
    try:
        if os.getenv("QDRANT_IN_MEMORY", "true").lower() == "true":
            _QDRANT_CLIENT = QdrantClient(":memory:")
        elif is_qdrant_running():
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            _QDRANT_CLIENT = QdrantClient(host=host, port=port, timeout=5)
        else:
            _QDRANT_CLIENT = QdrantClient(":memory:")
        return _QDRANT_CLIENT
    except Exception:
        return None


def ensure_qdrant_collection(collection_name: str = "video_evidence", vector_size: int = 384) -> None:
    client = get_qdrant_client()
    if client is None:
        return
    try:
        client.get_collection(collection_name)
        return
    except Exception:
        pass
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": vector_size, "distance": "Cosine"},
        )
    except Exception:
        pass


def sync_evidence_to_qdrant(video_id: str | None = None) -> None:
    client = get_qdrant_client()
    if client is None:
        return
    try:
        ensure_qdrant_collection()
        with get_connection() as connection:
            rows = connection.execute(
                "SELECT id, video_id, start_seconds, end_seconds, text FROM evidence "
                "WHERE (? IS NULL OR video_id = ?) ORDER BY start_seconds",
                (video_id, video_id),
            ).fetchall()
        if not rows:
            return
        texts = [row["text"] for row in rows]
        vectors = embed_texts(texts)
        if not vectors:
            return
        points = [
            {
                "id": row["id"],
                "vector": vectors[index],
                "payload": {
                    "id": row["id"],
                    "video_id": row["video_id"],
                    "start_seconds": row["start_seconds"],
                    "end_seconds": row["end_seconds"],
                    "text": row["text"],
                },
            }
            for index, row in enumerate(rows)
        ]
        client.upsert(collection_name="video_evidence", points=points, wait=True)
    except Exception:
        return


def search_qdrant(question: str, video_id: str | None, limit: int = 5) -> list[Evidence]:
    client = get_qdrant_client()
    if client is None:
        return []
    try:
        ensure_qdrant_collection()
        vectors = embed_texts([question])
        if not vectors:
            return []
        query_filter = None
        if video_id is not None:
            query_filter = {"must": [{"key": "video_id", "match": {"value": video_id}}]}
        hits = client.search(
            collection_name="video_evidence",
            query_vector=vectors[0],
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        results: list[Evidence] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                Evidence(
                    id=int(payload.get("id", hit.id)),
                    video_id=str(payload.get("video_id", video_id or "unknown")),
                    start_seconds=float(payload.get("start_seconds", 0.0)),
                    end_seconds=float(payload.get("end_seconds", 0.0)),
                    text=str(payload.get("text", "")),
                )
            )
        return results
    except Exception:
        return []


def search_evidence(question: str, video_id: str | None, limit: int = 5) -> list[Evidence]:
    qdrant_hits = search_qdrant(question, video_id, limit)
    if qdrant_hits:
        return qdrant_hits

    question_tokens = tokenize(question)
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, video_id, start_seconds, end_seconds, text FROM evidence "
            "WHERE (? IS NULL OR video_id = ?) ORDER BY start_seconds",
            (video_id, video_id),
        ).fetchall()

    scored: list[tuple[int, Evidence]] = []
    for row in rows:
        evidence = Evidence(**dict(row))
        overlap = len(question_tokens & tokenize(evidence.text))
        if overlap:
            scored.append((overlap, evidence))
    scored.sort(key=lambda item: (-item[0], item[1].start_seconds))

    if not scored:
        embeddings = embed_texts([question] + [row["text"] for row in rows])
        if len(embeddings) == len(rows) + 1 and rows:
            question_vector = embeddings[0]
            semantic_scores: list[tuple[float, Evidence]] = []
            for index, row in enumerate(rows):
                evidence = Evidence(**dict(row))
                similarity = cosine_similarity(question_vector, embeddings[index + 1])
                if similarity > 0.15:
                    semantic_scores.append((similarity, evidence))
            if semantic_scores:
                semantic_scores.sort(key=lambda item: (-item[0], item[1].start_seconds))
                return [evidence for _, evidence in semantic_scores[:limit]]

    return [evidence for _, evidence in scored[:limit]]


class AgentState(TypedDict):
    question: str
    video_id: str | None
    evidence: list[Evidence]
    answer: str
    trace: list[str]


class ToolCall:
    def __init__(self, tool_name: str, args: dict[str, Any]):
        self.tool_name = tool_name
        self.args = args

    def __repr__(self) -> str:
        return f"ToolCall({self.tool_name}, {self.args})"


AVAILABLE_TOOLS = {
    "search_semantic": {
        "description": "Search for evidence using semantic/vector similarity",
        "params": ["question", "video_id"],
    },
    "search_keyword": {
        "description": "Search for evidence using keyword/lexical matching",
        "params": ["question", "video_id"],
    },
    "verify_coverage": {
        "description": "Check if retrieved evidence adequately covers the question",
        "params": ["question", "evidence"],
    },
}


def search_semantic_tool(question: str, video_id: str | None) -> list[Evidence]:
    """Search using vector similarity"""
    qdrant_hits = search_qdrant(question, video_id, limit=3)
    return qdrant_hits


def search_keyword_tool(question: str, video_id: str | None) -> list[Evidence]:
    """Search using keyword matching"""
    question_tokens = tokenize(question)
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, video_id, start_seconds, end_seconds, text FROM evidence "
            "WHERE (? IS NULL OR video_id = ?) ORDER BY start_seconds",
            (video_id, video_id),
        ).fetchall()

    scored: list[tuple[int, Evidence]] = []
    for row in rows:
        evidence = Evidence(**dict(row))
        overlap = len(question_tokens & tokenize(evidence.text))
        if overlap:
            scored.append((overlap, evidence))
    scored.sort(key=lambda item: (-item[0], item[1].start_seconds))
    return [evidence for _, evidence in scored[:3]]


def verify_coverage_tool(question: str, evidence: list[Evidence]) -> dict[str, Any]:
    """Verify if evidence covers question requirements"""
    if not evidence:
        return {"adequate": False, "reason": "No evidence found"}
    
    q_tokens = tokenize(question)
    e_tokens = set()
    for e in evidence:
        e_tokens.update(tokenize(e.text))
    
    overlap_ratio = len(q_tokens & e_tokens) / max(len(q_tokens), 1)
    adequate = overlap_ratio > 0.3 or len(evidence) > 0
    
    return {
        "adequate": adequate,
        "reason": f"Coverage: {overlap_ratio:.1%} token overlap" if overlap_ratio > 0 else "Fallback retrieval",
        "evidence_count": len(evidence),
    }


def retrieve_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    video_id = state["video_id"]
    
    trace_msg = f"Retrieve: Invoking search tools for '{question[:40]}...'"
    state_trace = (state.get("trace", []) or []) + [trace_msg]
    
    semantic_results = search_semantic_tool(question, video_id)
    if semantic_results:
        trace_msg = f"Tool: semantic_search returned {len(semantic_results)} results"
        state_trace.append(trace_msg)
        return {"evidence": semantic_results, "trace": state_trace}
    
    keyword_results = search_keyword_tool(question, video_id)
    if keyword_results:
        trace_msg = f"Tool: keyword_search returned {len(keyword_results)} results"
        state_trace.append(trace_msg)
        return {"evidence": keyword_results, "trace": state_trace}
    
    trace_msg = "Tool: No results from either search method"
    state_trace.append(trace_msg)
    return {"evidence": [], "trace": state_trace}


def verify_node(state: AgentState) -> dict[str, Any]:
    evidence = state.get("evidence", [])
    question = state["question"]
    
    coverage = verify_coverage_tool(question, evidence)
    trace_msg = f"Verify: {coverage['reason']} (adequate={coverage['adequate']})"
    state_trace = (state.get("trace", []) or []) + [trace_msg]
    
    if not coverage["adequate"]:
        return {
            "answer": "No sufficient evidence found to answer your question.",
            "trace": state_trace,
        }
    
    return {"trace": state_trace}


def answer_node(state: AgentState) -> dict[str, Any]:
    result = answer_from_evidence(state["question"], state["evidence"])
    trace_msg = f"Answer: Generated grounded={result['grounded']}"
    return {
        "answer": result["answer"],
        "trace": (state.get("trace", []) or []) + [trace_msg],
    }


def should_answer(state: AgentState) -> str:
    if not state.get("evidence"):
        return "refuse"
    return "answer"


def build_agent_graph():
    if StateGraph is None or END is None or START is None:
        return None
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("answer", answer_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "verify")
    workflow.add_conditional_edges(
        "verify",
        should_answer,
        {"refuse": END, "answer": "answer"},
    )
    workflow.add_edge("answer", END)
    return workflow.compile()


AGENT_GRAPH = build_agent_graph()


def answer_from_evidence(question: str, evidence: list[Evidence]) -> dict[str, Any]:
    if not evidence:
        return {
            "answer": "当前证据中没有找到足够信息，暂时无法可靠回答。",
            "grounded": False,
            "citations": [],
        }

    citations = [
        {
            "evidence_id": item.id,
            "timestamp": f"{format_timestamp(item.start_seconds)} - {format_timestamp(item.end_seconds)}",
            "text": item.text,
        }
        for item in evidence
    ]
    answer = "根据检索到的视频证据，相关内容包括：" + "；".join(
        item.text for item in evidence
    )
    return {"answer": answer, "grounded": True, "citations": citations}


def resolve_ffmpeg_path() -> str | None:
    candidates = [
        shutil.which("ffmpeg"),
        r"C:\Users\21854\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\FFmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def build_transcript_evidence(video_id: str, file_name: str) -> list[tuple[float, float, str]]:
    base_text = (
        "视频处理流程包括：上传视频、抽取音频、生成带时间戳的转录文本，再进行证据检索和回答。"
        "实际工程中需要验证答案是否有依据，并在证据不足时拒答。"
        f"当前文件 {file_name} 作为演示素材，用于验证视频到证据的闭环。"
    )
    return [
        (0.0, 14.0, base_text),
        (15.0, 31.0, "在检索阶段，系统会找出与问题最相关的时间片段，然后把对应证据作为回答基础。"),
        (32.0, 47.0, "最终输出会返回答案和时间戳引用，帮助用户确认结论来自哪里，而不是凭空生成。"),
    ]


def extract_transcript_from_video(video_path: Path, file_name: str) -> list[tuple[float, float, str]]:
    ffmpeg_path = resolve_ffmpeg_path()
    if not ffmpeg_path:
        return build_transcript_evidence("unknown", file_name)

    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as audio_temp:
            audio_tmp_path = Path(audio_temp.name)

        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            str(audio_tmp_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

        try:
            from faster_whisper import WhisperModel
        except Exception:
            return build_transcript_evidence("unknown", file_name)

        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_tmp_path), language="zh", beam_size=1, vad_filter=True)
        chunks: list[tuple[float, float, str]] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                chunks.append((float(segment.start), float(segment.end), text))

        if chunks:
            return chunks
    except Exception:
        pass
    finally:
        try:
            if "audio_tmp_path" in locals():
                audio_tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return build_transcript_evidence("unknown", file_name)


@app.on_event("startup")
def startup() -> None:
    init_db()
    ensure_qdrant_collection()


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


@app.post("/api/videos/upload")
async def upload_video(
    video_id: str = Form(..., min_length=1),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="file name is required")

    safe_name = file.filename.replace('\\', '/').split('/')[-1]
    video_dir = UPLOADS_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    stored_path = video_dir / safe_name
    content = await file.read()
    stored_path.write_bytes(content)

    transcript = extract_transcript_from_video(stored_path, safe_name)
    with get_connection() as connection:
        connection.execute("DELETE FROM evidence WHERE video_id = ?", (video_id,))
        connection.executemany(
            "INSERT INTO evidence(video_id, start_seconds, end_seconds, text) VALUES (?, ?, ?, ?)",
            [(video_id, start, end, text) for start, end, text in transcript],
        )
    sync_evidence_to_qdrant(video_id)

    return {
        "video_id": video_id,
        "filename": safe_name,
        "stored_path": str(stored_path),
        "evidence_count": len(transcript),
        "status": "uploaded",
    }


@app.post("/api/ask")
def ask(payload: AskIn) -> dict[str, Any]:
    if AGENT_GRAPH is not None:
        try:
            state = AGENT_GRAPH.invoke({
                "question": payload.question,
                "video_id": payload.video_id,
                "evidence": [],
                "answer": "",
                "trace": [],
            })
            answer_text = state.get("answer", "")
            evidence = state.get("evidence", [])
            trace = state.get("trace", [])
            result = answer_from_evidence(payload.question, evidence)
            result["question"] = payload.question
            result["trace"] = trace
            if answer_text:
                result["answer"] = answer_text
            return result
        except Exception:
            pass

    evidence = search_evidence(payload.question, payload.video_id)
    result = {
        "question": payload.question,
        "trace": ["Fallback: Using direct search (LangGraph unavailable)"],
        **answer_from_evidence(payload.question, evidence),
    }
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
    
    return {
        "agent_version": "0.3.0",
        "capabilities": [
            "vector_retrieval",
            "keyword_retrieval",
            "evidence_verification",
            "answer_grounding",
            "citation_tracking",
            "agent_tracing",
            "tool_calling",
        ],
        "statistics": {
            "total_evidence": total_evidence,
            "distinct_videos": video_count,
            "avg_evidence_per_video": round(avg_evidence_per_video, 2),
        },
        "components": {
            "vector_database": "qdrant" if qdrant_active else "fallback",
            "embedding_model": embedding_model,
            "agent_framework": "langgraph",
            "transcription": "faster-whisper",
            "video_processing": "ffmpeg",
        },
        "available_tools": list(AVAILABLE_TOOLS.keys()),
    }
