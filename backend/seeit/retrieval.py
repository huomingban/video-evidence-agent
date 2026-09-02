"""Lexical, embedding, and local Qdrant evidence retrieval."""
from __future__ import annotations
import math
import os
import re
import socket
from typing import Any
try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
from .config import LOCAL_QDRANT_DIR, env_flag, logger
from .models import Evidence
from .storage import get_connection

_QDRANT_CLIENT: Any | None = None
_EMBEDDING_MODEL: Any | None = None
_EMBEDDING_LOAD_FAILED = False

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
                "SELECT id, video_id, start_seconds, end_seconds, text, source FROM evidence "
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
                    "source": row["source"],
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
                    source=str(payload.get("source", "ASR")),
                )
            )
        return results
    except Exception:
        return []


def search_evidence(question: str, video_id: str | None, limit: int = 5) -> list[Evidence]:
    question_tokens = tokenize(question)
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, video_id, start_seconds, end_seconds, text, source FROM evidence "
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


def plan_evidence_requirements(query: str) -> dict[str, Any]:
    """Create a deterministic evidence-slot plan before model verification.

    The reference project uses this boundary to separate question
    decomposition from retrieval. The local implementation intentionally
    stays provider-independent and supplies conservative query slots.
    """
    normalized = " ".join(str(query).split()).strip()
    if not normalized:
        return {"strategy": "SINGLE_QUERY", "requirements": []}

    parts = [
        item.strip(" ，,、；;。！？!?")
        for item in re.split(r"\s*(?:以及|并且|同时|分别|和|与|及|以及|以及|以及|以及|以及|以及|and|also)\s*", normalized, flags=re.I)
        if item.strip(" ，,、；;。！？!?")
    ]
    if not parts:
        parts = [normalized]
    if len(parts) > 4:
        parts = parts[:4]

    strategy = "SINGLE_QUERY"
    if len(parts) > 1:
        strategy = "MULTI_REQUIREMENT"
    if re.search(r"比较|区别|不同|对比|versus|vs\.?", normalized, flags=re.I):
        strategy = "COMPARISON_DECOMPOSITION"
    return {
        "strategy": strategy,
        "requirements": [{"query": part} for part in parts],
    }

def public_evidence(item: Evidence) -> dict[str, Any]:
    """Return the small, stable evidence shape exposed to the model."""
    return {
        "evidence_id": item.id,
        "start_ms": round(item.start_seconds * 1000),
        "end_ms": round(item.end_seconds * 1000),
        "timestamp": f"{format_timestamp(item.start_seconds)} - {format_timestamp(item.end_seconds)}",
        "text": item.text,
        "source": item.source,
    }


def load_video_evidence(video_id: str | None) -> list[Evidence]:
    if not video_id:
        return []
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, video_id, start_seconds, end_seconds, text, source FROM evidence "
            "WHERE video_id = ? ORDER BY start_seconds, id",
            (video_id,),
        ).fetchall()
    return [Evidence(**dict(row)) for row in rows]
