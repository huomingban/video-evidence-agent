"""Embedding backend boundary.

The current project uses SentenceTransformers as an optional local backend.
Keeping this boundary separate matches the reference project's retrieval
layout and leaves room for an ONNX backend later.
"""

from .retrieval import embed_texts, get_embedding_model

__all__ = ["embed_texts", "get_embedding_model"]
