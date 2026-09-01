from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
import math
import os
import re
import shutil
import socket
import sqlite3
import subprocess
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, TypedDict

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional when environment variables are set directly
    load_dotenv = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional until Kimi is configured
    OpenAI = None

try:
    import httpx2
except Exception:  # pragma: no cover - provided by the OpenAI client dependency
    httpx2 = None

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
from starlette.concurrency import run_in_threadpool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "agent.sqlite3"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
LOCAL_QDRANT_DIR = PROJECT_ROOT / "data" / "qdrant"
ENV_PATH = PROJECT_ROOT / "backend" / ".env"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_QDRANT_DIR.mkdir(parents=True, exist_ok=True)

if load_dotenv is not None:
    load_dotenv(ENV_PATH)

logger = logging.getLogger(__name__)

_QDRANT_CLIENT: Any | None = None
_EMBEDDING_MODEL: Any | None = None
_EMBEDDING_LOAD_FAILED = False
_WHISPER_MODEL: Any | None = None


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def kimi_settings() -> dict[str, Any]:
    return {
        "api_key": os.getenv("KIMI_API_KEY", "").strip(),
        "base_url": os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1").strip(),
        "model": os.getenv("KIMI_MODEL", "moonshot-v1-8k").strip(),
        "enabled": env_flag("KIMI_ENABLED", True),
        "timeout": float(os.getenv("KIMI_TIMEOUT_SECONDS", "45")),
        "trust_env": env_flag("KIMI_TRUST_ENV", False),
        "proxy": os.getenv("KIMI_PROXY", "").strip() or None,
    }


def kimi_is_configured() -> bool:
    settings = kimi_settings()
    return bool(OpenAI is not None and settings["enabled"] and settings["api_key"])


@contextmanager
def direct_connection_if_configured():
    """Temporarily bypass broken system proxies during model downloads."""
    if env_flag("WHISPER_TRUST_ENV", False):
        yield
        return
    proxy_names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    saved = {name: os.environ.pop(name, None) for name in proxy_names}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value

app = FastAPI(title="VideoEvidence Agent", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvidenceIn(BaseModel):
    video_id: str = Field(min_length=1, max_length=200)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str = Field(min_length=1, max_length=10000)


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    video_id: str | None = Field(default=None, max_length=200)


class DemoSeedIn(BaseModel):
    video_id: str = Field(default="demo-video", min_length=1, max_length=200)


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


def get_embedding_model() -> Any | None:
    global _EMBEDDING_MODEL, _EMBEDDING_LOAD_FAILED
    if not env_flag("EMBEDDING_ENABLED", False) or SentenceTransformer is None or _EMBEDDING_LOAD_FAILED:
        return None
    if _EMBEDDING_MODEL is None:
        try:
            model_name = os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
            _EMBEDDING_MODEL = SentenceTransformer(model_name)
        except Exception:
            _EMBEDDING_LOAD_FAILED = True
            logger.exception("Unable to load the embedding model")
            return None
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
        if env_flag("QDRANT_IN_MEMORY", False):
            _QDRANT_CLIENT = QdrantClient(":memory:")
        elif is_qdrant_running():
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            _QDRANT_CLIENT = QdrantClient(host=host, port=port, timeout=5)
        else:
            _QDRANT_CLIENT = QdrantClient(path=str(LOCAL_QDRANT_DIR))
        return _QDRANT_CLIENT
    except Exception:
        logger.exception("Unable to initialize Qdrant")
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
        if video_id is not None:
            client.delete(
                collection_name="video_evidence",
                points_selector={
                    "filter": {"must": [{"key": "video_id", "match": {"value": video_id}}]}
                },
                wait=True,
            )
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
        minimum_score = float(os.getenv("MIN_SEMANTIC_SCORE", "0.35"))
        for hit in hits:
            if getattr(hit, "score", 0.0) < minimum_score:
                continue
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

    semantic_results = search_qdrant(question, video_id, limit)
    combined: list[Evidence] = []
    seen_ids: set[int] = set()
    for item in semantic_results + [evidence for _, evidence in scored[:limit]]:
        if item.id not in seen_ids:
            combined.append(item)
            seen_ids.add(item.id)
    return combined[:limit]


def public_evidence(item: Evidence) -> dict[str, Any]:
    """Return the small, stable evidence shape exposed to the model."""
    return {
        "evidence_id": item.id,
        "source": "ASR",
        "start_ms": round(item.start_seconds * 1000),
        "end_ms": round(item.end_seconds * 1000),
        "timestamp": f"{format_timestamp(item.start_seconds)} - {format_timestamp(item.end_seconds)}",
        "text": item.text,
    }


def load_video_evidence(video_id: str | None) -> list[Evidence]:
    if not video_id:
        return []
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, video_id, start_seconds, end_seconds, text FROM evidence "
            "WHERE video_id = ? ORDER BY start_seconds, id",
            (video_id,),
        ).fetchall()
    return [Evidence(**dict(row)) for row in rows]


class AgentToolbox:
    """Tools the model may choose while investigating one video."""

    def __init__(self, video_id: str | None, question: str) -> None:
        self.video_id = video_id
        self.question = question
        self.evidence = load_video_evidence(video_id)
        self._trace: list[dict[str, Any]] = []

    @staticmethod
    def _evenly_spaced(items: list[Evidence], count: int) -> list[Evidence]:
        if len(items) <= count:
            return items
        if count <= 1:
            return [items[len(items) // 2]]
        indexes = {
            round(index * (len(items) - 1) / (count - 1))
            for index in range(count)
        }
        return [items[index] for index in sorted(indexes)]

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_video_metadata",
                    "description": "读取当前视频的文件状态、证据数量和时间范围。开始分析时调用。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_timeline_overview",
                    "description": "从完整时间轴均匀抽取代表性证据，用于总结视频主题。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_segments": {"type": "integer", "minimum": 4, "maximum": 20},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_timeline",
                    "description": "按问题检索最相关的带时间戳 ASR 证据。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "minLength": 1},
                            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_evidence_window",
                    "description": "展开某个时间点前后的连续证据，避免脱离上下文理解。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timestamp_ms": {"type": "integer", "minimum": 0},
                            "before_ms": {"type": "integer", "minimum": 0, "maximum": 120000},
                            "after_ms": {"type": "integer", "minimum": 0, "maximum": 120000},
                        },
                        "required": ["timestamp_ms"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "verify_citations",
                    "description": "检查候选引用是否真实存在并且能覆盖当前问题。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "citation_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "maxItems": 20,
                            },
                        },
                        "required": ["citation_ids"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_report",
                    "description": "提交最终结构化报告。必须是最后一步；证据不足时 answerable=false。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answerable": {"type": "boolean"},
                            "final_answer": {"type": "string", "minLength": 1, "maxLength": 2000},
                            "citation_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "maxItems": 20,
                            },
                            "support_level": {
                                "type": "string",
                                "enum": ["DIRECT", "SUMMARY", "INFERENCE", "INSUFFICIENT"],
                            },
                        },
                        "required": ["answerable", "final_answer", "citation_ids", "support_level"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "get_video_metadata":
                result = self._metadata()
            elif name == "get_timeline_overview":
                result = self._overview(arguments)
            elif name == "search_timeline":
                result = self._search(arguments)
            elif name == "get_evidence_window":
                result = self._window(arguments)
            elif name == "verify_citations":
                result = self._verify(arguments)
            elif name == "generate_report":
                result = self._report(arguments)
            else:
                result = {"ok": False, "error": f"未知 Agent 工具: {name}"}
        except (TypeError, ValueError, KeyError) as error:
            result = {"ok": False, "error": str(error)}
        self._trace.append({
            "tool": name,
            "arguments": arguments,
            "success": bool(result.get("ok")),
            "result_preview": json.dumps(result, ensure_ascii=False)[:500],
        })
        return result

    def trace(self) -> list[dict[str, Any]]:
        return list(self._trace)

    def _metadata(self) -> dict[str, Any]:
        files = []
        if self.video_id:
            directory = UPLOADS_DIR / self.video_id
            if directory.is_dir():
                files = [path.name for path in directory.iterdir() if path.is_file()]
        return {
            "ok": True,
            "video_id": self.video_id,
            "uploaded_files": files,
            "evidence_count": len(self.evidence),
            "timeline_end_ms": round(max((item.end_seconds for item in self.evidence), default=0) * 1000),
        }

    def _overview(self, arguments: dict[str, Any]) -> dict[str, Any]:
        count = max(4, min(int(arguments.get("max_segments", 12)), 20))
        selected = self._evenly_spaced(self.evidence, count)
        return {"ok": True, "sampling": "even_timeline", "segments": [public_evidence(item) for item in selected]}

    def _search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or self.question).strip()
        if not query:
            raise ValueError("query 不能为空")
        count = max(1, min(int(arguments.get("top_k", 6)), 10))
        matches = search_evidence(query, self.video_id, count)
        return {"ok": True, "query": query, "matches": [public_evidence(item) for item in matches]}

    def _window(self, arguments: dict[str, Any]) -> dict[str, Any]:
        timestamp = max(0, int(arguments["timestamp_ms"]))
        before = max(0, min(int(arguments.get("before_ms", 15000)), 120000))
        after = max(0, min(int(arguments.get("after_ms", 15000)), 120000))
        start, end = timestamp - before, timestamp + after
        matches = [
            item for item in self.evidence
            if item.end_seconds * 1000 >= start and item.start_seconds * 1000 <= end
        ][:40]
        return {
            "ok": True,
            "window_start_ms": max(0, start),
            "window_end_ms": end,
            "segments": [public_evidence(item) for item in matches],
        }

    def _ids(self, arguments: dict[str, Any]) -> list[int]:
        raw_ids = arguments.get("citation_ids", [])
        if not isinstance(raw_ids, list):
            return []
        ids = []
        for value in raw_ids:
            if isinstance(value, dict):
                value = value.get("evidence_id", value.get("id"))
            if isinstance(value, int) and value not in ids:
                ids.append(value)
        return ids

    def _verify(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ids = self._ids(arguments)
        by_id = {item.id: item for item in self.evidence}
        selected = [by_id[item_id] for item_id in ids if item_id in by_id]
        coverage = verify_coverage_tool(self.question, selected)
        return {
            "ok": True,
            "valid": len(selected) == len(ids) and bool(selected),
            "valid_ids": [item.id for item in selected],
            "coverage": coverage,
        }

    def _report(self, arguments: dict[str, Any]) -> dict[str, Any]:
        answerable = bool(arguments.get("answerable"))
        final_answer = str(arguments.get("final_answer") or "").strip()
        ids = self._ids(arguments)
        by_id = {item.id: item for item in self.evidence}
        selected = [by_id[item_id] for item_id in ids if item_id in by_id]
        invalid_ids = [item_id for item_id in ids if item_id not in by_id]
        coverage = verify_coverage_tool(self.question, selected)
        if not final_answer:
            return {"ok": False, "accepted": False, "error": "final_answer 不能为空"}
        if answerable and (not selected or invalid_ids or not coverage["adequate"]):
            return {
                "ok": True,
                "accepted": False,
                "error": "报告未通过引用或证据覆盖校验，请继续检索；证据不足则提交 answerable=false",
                "valid_ids": [item.id for item in selected],
                "coverage": coverage,
            }
        if not answerable:
            selected = []
        return {
            "ok": True,
            "accepted": True,
            "answerable": answerable,
            "final_answer": final_answer,
            "support_level": str(arguments.get("support_level") or ("DIRECT" if answerable else "INSUFFICIENT")),
            "citations": [public_evidence(item) for item in selected],
        }


class AgentState(TypedDict):
    question: str
    video_id: str | None
    evidence: list[Evidence]
    answer: str
    trace: list[str]
    grounded: bool
    citations: list[dict[str, Any]]
    provider: str
    adequate: bool


AGENT_TOOL_NAMES = [
    "get_video_metadata",
    "get_timeline_overview",
    "search_timeline",
    "get_evidence_window",
    "verify_citations",
    "generate_report",
]


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
    adequate = overlap_ratio >= float(os.getenv("MIN_EVIDENCE_COVERAGE", "0.3"))
    
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
    trace_msg = f"Tool: semantic_search returned {len(semantic_results)} results"
    state_trace.append(trace_msg)
    keyword_results = search_keyword_tool(question, video_id)
    trace_msg = f"Tool: keyword_search returned {len(keyword_results)} results"
    state_trace.append(trace_msg)
    combined: list[Evidence] = []
    seen_ids: set[int] = set()
    for item in semantic_results + keyword_results:
        if item.id not in seen_ids:
            combined.append(item)
            seen_ids.add(item.id)
    if not combined:
        state_trace.append("Retrieve: No evidence matched the question")
    return {"evidence": combined[:5], "trace": state_trace}


def verify_node(state: AgentState) -> dict[str, Any]:
    evidence = state.get("evidence", [])
    question = state["question"]
    
    coverage = verify_coverage_tool(question, evidence)
    trace_msg = f"Verify: {coverage['reason']} (adequate={coverage['adequate']})"
    state_trace = (state.get("trace", []) or []) + [trace_msg]
    
    if not coverage["adequate"]:
        return {
            "answer": "No sufficient evidence found to answer your question.",
            "grounded": False,
            "citations": [],
            "provider": "refusal",
            "adequate": False,
            "trace": state_trace,
        }
    
    return {"adequate": True, "trace": state_trace}


def answer_node(state: AgentState) -> dict[str, Any]:
    result = answer_from_evidence(state["question"], state["evidence"])
    provider = result["provider"]
    trace_msg = f"Answer: {provider} generated grounded={result['grounded']}"
    return {
        "answer": result["answer"],
        "grounded": result["grounded"],
        "citations": result["citations"],
        "provider": provider,
        "adequate": True,
        "trace": (state.get("trace", []) or []) + [trace_msg],
    }


def should_answer(state: AgentState) -> str:
    if not state.get("adequate", False):
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


def evidence_citations(evidence: list[Evidence]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": item.id,
            "timestamp": f"{format_timestamp(item.start_seconds)} - {format_timestamp(item.end_seconds)}",
            "text": item.text,
        }
        for item in evidence
    ]


def parse_kimi_json(content: str) -> dict[str, Any] | None:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _message_value(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def _tool_call_value(call: Any, name: str, default: Any = None) -> Any:
    if isinstance(call, dict):
        return call.get(name, default)
    return getattr(call, name, default)


def _function_value(call: Any, name: str, default: Any = None) -> Any:
    function = _tool_call_value(call, "function", {}) or {}
    if isinstance(function, dict):
        return function.get(name, default)
    return getattr(function, name, default)


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _assistant_tool_message(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
    normalized_calls = []
    for call in tool_calls:
        normalized_calls.append({
            "id": str(_tool_call_value(call, "id", "tool-call")),
            "type": "function",
            "function": {
                "name": str(_function_value(call, "name", "")),
                "arguments": str(_function_value(call, "arguments", "{}")),
            },
        })
    return {
        "role": "assistant",
        "content": _message_value(message, "content"),
        "tool_calls": normalized_calls,
    }


def run_kimi_agent(question: str, video_id: str | None) -> dict[str, Any] | None:
    """Let Kimi choose evidence tools until it submits an accepted report."""
    if not kimi_is_configured():
        return None

    settings = kimi_settings()
    toolbox = AgentToolbox(video_id, question)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "你是一个严谨的视频证据 Agent。只能使用工具返回的证据回答，不能补充外部知识。"
                "先根据问题选择合适的工具：概括问题使用时间轴概览，具体问题使用时间轴检索，"
                "需要上下文时展开证据窗口。完成调查后必须调用 generate_report。"
                "如果证据不足，必须提交 answerable=false，并明确说明视频没有提供足够依据。"
            ),
        },
        {"role": "user", "content": f"视频 ID：{video_id or '未指定'}\n问题：{question}"},
    ]
    try:
        http_client = None
        if httpx2 is not None:
            http_client = httpx2.Client(
                trust_env=settings["trust_env"],
                proxy=settings["proxy"],
                timeout=settings["timeout"],
            )
        client = OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            timeout=settings["timeout"],
            max_retries=1,
            http_client=http_client,
        )
        max_steps = max(2, min(int(os.getenv("AGENT_MAX_TOOL_STEPS", "8")), 12))
        for step in range(max_steps):
            response = client.chat.completions.create(
                model=settings["model"],
                temperature=0.1,
                messages=messages,
                tools=toolbox.schemas(),
                tool_choice="auto",
            )
            message = response.choices[0].message
            tool_calls = list(_message_value(message, "tool_calls", []) or [])
            content = str(_message_value(message, "content", "") or "")

            if tool_calls:
                messages.append(_assistant_tool_message(message, tool_calls))
                for call in tool_calls:
                    call_id = str(_tool_call_value(call, "id", f"tool-call-{step}"))
                    name = str(_function_value(call, "name", ""))
                    arguments = _parse_tool_arguments(_function_value(call, "arguments", "{}"))
                    result = toolbox.execute(name, arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    if name == "generate_report" and result.get("accepted"):
                        return {
                            "question": question,
                            "answer": result["final_answer"],
                            "grounded": bool(result["answerable"]),
                            "citations": result["citations"],
                            "provider": "Kimi",
                            "support_level": result["support_level"],
                            "trace": [
                                f"Agent step {step + 1}: Kimi selected {item['tool']}"
                                for item in toolbox.trace()
                            ],
                            "tool_trace": toolbox.trace(),
                        }
                continue

            # Some OpenAI-compatible endpoints do not return tool_calls reliably.
            # Accept a JSON report as a compatibility fallback, then keep the same gate.
            parsed = parse_kimi_json(content)
            if parsed:
                arguments = {
                    "answerable": parsed.get("answerable", True),
                    "final_answer": parsed.get("final_answer", parsed.get("answer", "")),
                    "citation_ids": parsed.get("citation_ids", parsed.get("citations", [])),
                    "support_level": parsed.get("support_level", "DIRECT"),
                }
                result = toolbox.execute("generate_report", arguments)
                if result.get("accepted"):
                    return {
                        "question": question,
                        "answer": result["final_answer"],
                        "grounded": bool(result["answerable"]),
                        "citations": result["citations"],
                        "provider": "Kimi",
                        "support_level": result["support_level"],
                        "trace": [
                            f"Agent step {step + 1}: compatibility report submitted"
                        ],
                        "tool_trace": toolbox.trace(),
                    }
            messages.extend([
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": "请继续调用证据工具，并通过 generate_report 提交最终报告，不要只输出普通文本。",
                },
            ])
        raise RuntimeError("Kimi Agent exceeded its tool-step budget without an accepted report")
    except Exception:
        logger.exception("Kimi Agent execution failed")
        return None


def generate_kimi_answer(question: str, evidence: list[Evidence]) -> tuple[str, list[Evidence]] | None:
    if not kimi_is_configured():
        return None

    settings = kimi_settings()
    context = [
        {
            "evidence_id": item.id,
            "timestamp": f"{format_timestamp(item.start_seconds)} - {format_timestamp(item.end_seconds)}",
            "text": item.text,
        }
        for item in evidence
    ]
    prompt = (
        "请只根据给定的视频证据回答问题。不要补充证据中没有的信息。"
        "必须返回 JSON，不要使用 Markdown，格式为："
        '{"answer":"...","citation_ids":[1,2]}。'
        "citation_ids 只能填写真正支持答案的 evidence_id；如果证据不足，answer 写明证据不足，"
        "并将 citation_ids 设为空数组。请用中文回答。\n\n"
        f"问题：{question}\n证据：{json.dumps(context, ensure_ascii=False)}"
    )
    try:
        http_client = None
        if httpx2 is not None:
            http_client = httpx2.Client(
                trust_env=settings["trust_env"],
                proxy=settings["proxy"],
                timeout=settings["timeout"],
            )
        client = OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            timeout=settings["timeout"],
            max_retries=1,
            http_client=http_client,
        )
        response = client.chat.completions.create(
            model=settings["model"],
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的视频证据问答助手。回答必须可由提供的证据支持。",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        parsed = parse_kimi_json(content)
        if not parsed or not isinstance(parsed.get("answer"), str):
            raise ValueError("Kimi returned an invalid JSON answer")
        ids = parsed.get("citation_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError("Kimi returned no valid citation ids")
        evidence_by_id = {item.id: item for item in evidence}
        selected = [evidence_by_id[int(item_id)] for item_id in ids if str(item_id).isdigit() and int(item_id) in evidence_by_id]
        if not selected:
            raise ValueError("Kimi cited evidence outside the retrieved set")
        return parsed["answer"].strip(), selected
    except Exception:
        logger.exception("Kimi answer generation failed")
        return None


def answer_from_evidence(question: str, evidence: list[Evidence]) -> dict[str, Any]:
    if not evidence:
        return {
            "answer": "当前证据中没有找到足够信息，暂时无法可靠回答。",
            "grounded": False,
            "citations": [],
            "provider": "refusal",
        }

    kimi_result = generate_kimi_answer(question, evidence)
    if kimi_result is not None:
        answer, cited_evidence = kimi_result
        return {
            "answer": answer,
            "grounded": True,
            "citations": evidence_citations(cited_evidence),
            "provider": "Kimi",
        }

    if kimi_is_configured():
        return {
            "answer": "Kimi 暂时不可用，未生成未经验证的回答。请检查 API Key、模型名称和网络连接。",
            "grounded": False,
            "citations": [],
            "provider": "Kimi error",
        }

    answer = "根据检索到的视频证据，相关内容包括：" + "；".join(
        item.text for item in evidence
    )
    return {
        "answer": answer,
        "grounded": True,
        "citations": evidence_citations(evidence),
        "provider": "local fallback",
    }


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


def extract_transcript_from_video(video_path: Path, file_name: str) -> list[tuple[float, float, str]]:
    ffmpeg_path = resolve_ffmpeg_path()
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg is not installed or cannot be found")

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

        global _WHISPER_MODEL
        try:
            from faster_whisper import WhisperModel
        except Exception as error:
            raise RuntimeError("faster-whisper is not installed") from error

        if _WHISPER_MODEL is None:
            with direct_connection_if_configured():
                _WHISPER_MODEL = WhisperModel(
                    os.getenv("WHISPER_MODEL", "tiny"),
                    device=os.getenv("WHISPER_DEVICE", "cpu"),
                    compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
                )
        model = _WHISPER_MODEL
        segments, _ = model.transcribe(str(audio_tmp_path), language="zh", beam_size=1, vad_filter=True)
        chunks: list[tuple[float, float, str]] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                chunks.append((float(segment.start), float(segment.end), text))

        if chunks:
            return merge_transcript_chunks(
                chunks,
                max_duration=float(os.getenv("TRANSCRIPT_CHUNK_SECONDS", "30")),
                max_chars=int(os.getenv("TRANSCRIPT_CHUNK_MAX_CHARS", "240")),
                max_gap=float(os.getenv("TRANSCRIPT_MAX_GAP_SECONDS", "2")),
            )
    except RuntimeError:
        raise
    except Exception as error:
        logger.exception("Video transcription failed")
        raise RuntimeError(f"Video transcription failed: {error}") from error
    finally:
        try:
            if "audio_tmp_path" in locals():
                audio_tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return []


def merge_transcript_chunks(
    segments: list[tuple[float, float, str]],
    max_duration: float = 30.0,
    max_chars: int = 240,
    max_gap: float = 2.0,
) -> list[tuple[float, float, str]]:
    """Group short Whisper segments into readable, retrieval-friendly evidence blocks."""
    merged: list[tuple[float, float, str]] = []
    current_start: float | None = None
    current_end = 0.0
    current_text = ""

    for start, end, text in segments:
        normalized = " ".join(text.split())
        if not normalized or end <= start:
            continue

        candidate_text = f"{current_text}{normalized}" if current_text else normalized
        should_flush = (
            current_start is not None
            and (start - current_end > max_gap
                 or end - current_start > max_duration
                 or len(candidate_text) > max_chars)
        )
        if should_flush:
            merged.append((current_start, current_end, current_text))
            current_start = None
            current_text = ""

        if current_start is None:
            current_start = start
        current_end = end
        current_text += normalized

    if current_start is not None and current_text:
        merged.append((current_start, current_end, current_text))
    return merged


def validate_video_id(video_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", video_id):
        raise HTTPException(
            status_code=422,
            detail="video_id may contain only letters, numbers, '.', '_' and '-'",
        )
    return video_id


def validate_video_filename(filename: str) -> str:
    safe_name = Path(filename.replace("\\", "/")).name
    allowed_extensions = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}
    if not safe_name or Path(safe_name).suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=415, detail="unsupported video file type")
    return safe_name


@app.on_event("startup")
def startup() -> None:
    init_db()
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
    max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
    total_bytes = 0
    try:
        with stored_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_upload_bytes:
                    raise HTTPException(status_code=413, detail="video file is too large")
                output.write(chunk)
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise

    try:
        transcript = await run_in_threadpool(extract_transcript_from_video, stored_path, safe_name)
    except Exception as error:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(error)) from error
    if not transcript:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="no speech transcript was produced")
    with get_connection() as connection:
        connection.execute("DELETE FROM evidence WHERE video_id = ?", (video_id,))
        connection.executemany(
            "INSERT INTO evidence(video_id, start_seconds, end_seconds, text) VALUES (?, ?, ?, ?)",
            [(video_id, start, end, text) for start, end, text in transcript],
        )
    await run_in_threadpool(sync_evidence_to_qdrant, video_id)

    return {
        "video_id": video_id,
        "filename": safe_name,
        "stored_path": str(stored_path),
        "evidence_count": len(transcript),
        "status": "uploaded",
    }


@app.post("/api/ask")
def ask(payload: AskIn) -> dict[str, Any]:
    if kimi_is_configured():
        agent_result = run_kimi_agent(payload.question, payload.video_id)
        if agent_result is not None:
            return agent_result
        return {
            "question": payload.question,
            "answer": "Kimi Agent 暂时无法完成证据检索，请检查 API Key、模型名称和网络连接。",
            "grounded": False,
            "citations": [],
            "provider": "Kimi Agent error",
            "trace": ["Kimi Agent failed before submitting an accepted report"],
        }

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
            return {
                "question": payload.question,
                "answer": state.get("answer", "当前无法生成回答。"),
                "grounded": bool(state.get("grounded", False)),
                "citations": state.get("citations", []),
                "provider": state.get("provider", ""),
                "trace": state.get("trace", []),
            }
        except Exception:
            logger.exception("Agent graph failed")

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
    embedding_enabled = env_flag("EMBEDDING_ENABLED", False)
    
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
            "vector_database": "qdrant" if qdrant_active and embedding_enabled else "disabled",
            "embedding_model": embedding_model if embedding_enabled else "disabled",
            "agent_framework": "langgraph" if AGENT_GRAPH is not None else "fallback",
            "transcription": "faster-whisper" if _WHISPER_MODEL is not None else "available on upload",
            "video_processing": "ffmpeg" if resolve_ffmpeg_path() else "unavailable",
            "llm": {
                "provider": "Kimi" if kimi_is_configured() else "local fallback",
                "configured": kimi_is_configured(),
                "model": kimi_settings()["model"],
            },
        },
        "available_tools": AGENT_TOOL_NAMES,
    }
