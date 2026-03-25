from __future__ import annotations

from openai import AsyncOpenAI

from app.config import settings
from app.db.faiss_store import FAISSStore
from app.models.chunk import RetrievedChunk
from app.utilities.logger import get_logger

log = get_logger(__name__)


def _meta_int(val: object) -> int | None:
    if val is None or val == "" or val == -1:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _document_to_retrieved(chunk_id: str, score: float, text: str, m: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        citation_url=(m.get("citation_url") or None) or None,
        chapter_title=(m.get("chapter_title") or None) or None,
        section=(m.get("section") or None) or None,
        pdf_page_number=_meta_int(m.get("pdf_page_number")),
        content_type=str(m.get("content_type", "text")),
    )


async def retrieve(
    query: str,
    pdf_id: str,
    faiss_store: FAISSStore,
    client: AsyncOpenAI,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or settings.top_k_retrieval
    _ = client

    k_fetch = min(max(top_k * 8, top_k + 16), 250)
    raw = await faiss_store.similarity_search_with_score_async(query, k=k_fetch)
    if not raw:
        return []

    out: list[RetrievedChunk] = []
    for doc, score in raw:
        m = doc.metadata
        if m.get("document_id") != pdf_id:
            continue
        cid = m.get("chunk_id")
        if not cid:
            continue
        out.append(_document_to_retrieved(str(cid), float(score), doc.page_content, m))
        if len(out) >= top_k:
            break

    if not out:
        log.info(
            "[QUERY] retrieve: no chunks for this document (raw_faiss_hits=%s) q_preview=%s",
            len(raw),
            query[:100],
        )
        return []

    log.info(
        "[QUERY] retrieve: faiss_hits=%s kept_after_pdf_id_filter=%s top_k=%s q_preview=%s",
        len(raw),
        len(out),
        top_k,
        query[:100],
    )
    return out
