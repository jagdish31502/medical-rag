from __future__ import annotations

import asyncio

from sentence_transformers import CrossEncoder

from app.config import settings
from app.models.chunk import RetrievedChunk
from app.utilities.logger import get_logger

log = get_logger(__name__)

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or settings.top_k_rerank
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [(query, c.text) for c in chunks]

    def _predict() -> list[float]:
        return reranker.predict(pairs).tolist()

    scores = await asyncio.to_thread(_predict)
    scored = sorted(
        zip(chunks, scores), key=lambda x: x[1], reverse=True
    )
    out: list[RetrievedChunk] = []
    for chunk, sc in scored[:top_k]:
        out.append(
            chunk.model_copy(update={"score": float(sc)})
        )
    log.info("[QUERY] rerank: selected %s chunks for LLM context", len(out))
    return out
