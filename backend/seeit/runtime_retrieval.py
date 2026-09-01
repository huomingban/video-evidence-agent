"""Runtime retrieval policy and fallback selection."""

from __future__ import annotations

import os
from typing import Any, Callable

from .retrieval import search_evidence


def build_runtime_retriever(video_id: str | None) -> Callable[[str, int], list[Any]]:
    """Return the configured retrieval strategy for one video.

    The callable shape keeps Agent tools independent from the storage backend;
    Qdrant remains an optional enhancement behind ``search_evidence``.
    """
    profile = os.getenv("EVIDENCE_RETRIEVER_PROFILE", "hybrid-local").strip()
    if profile in {"lexical", "keyword"}:
        from .storage import get_connection
        from .models import Evidence
        from .retrieval import tokenize

        def lexical(query: str, top_k: int = 8) -> list[Any]:
            query_tokens = tokenize(query)
            with get_connection() as connection:
                rows = connection.execute(
                    "SELECT id, video_id, start_seconds, end_seconds, text FROM evidence "
                    "WHERE (? IS NULL OR video_id = ?) ORDER BY start_seconds",
                    (video_id, video_id),
                ).fetchall()
            ranked = [
                (len(query_tokens & tokenize(row["text"])), Evidence(**dict(row)))
                for row in rows
            ]
            ranked.sort(key=lambda item: (-item[0], item[1].start_seconds, item[1].id))
            return [item for score, item in ranked if score > 0][:top_k]

        return lexical

    return lambda query, top_k=8: search_evidence(query, video_id, top_k)


__all__ = ["build_runtime_retriever"]
