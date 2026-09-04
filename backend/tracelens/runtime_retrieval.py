"""Build the reference-style retrieval stack for one persisted video."""

from __future__ import annotations

from typing import Any

from .models import Evidence
from .retrieval import (
    CoverageAwareEvidenceRetriever,
    EvidenceRetriever,
    ContextualEvidenceRetriever,
    load_video_evidence,
)


def _segment(item: Evidence) -> dict[str, Any]:
    return {
        "evidenceId": item.id,
        "segmentId": f"evidence-{item.id}",
        "startMs": round(item.start_seconds * 1000),
        "endMs": round(item.end_seconds * 1000),
        "content": item.text,
        "source": str(item.source or "ASR").upper(),
    }


def build_runtime_retriever(video_id: str | None) -> CoverageAwareEvidenceRetriever:
    """Return Coverage -> Context -> Lexical retrieval for one video snapshot."""
    segments = [_segment(item) for item in load_video_evidence(video_id)]
    lexical = EvidenceRetriever(segments)
    contextual = ContextualEvidenceRetriever(segments, lexical)
    return CoverageAwareEvidenceRetriever(
        segments,
        contextual,
        candidate_depth=24,
        context_window_ms=20000,
        context_per_requirement=2,
        # Missing-anchor refusal belongs to the Verifier: ASR/OCR can contain
        # legitimate transcription variants of the user's wording.
        enable_anchor_gate=False,
    )


__all__ = ["build_runtime_retriever"]
