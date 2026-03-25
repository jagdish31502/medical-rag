from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI

from app.config import settings
from app.db.faiss_store import FAISSStore
from app.models.chunk import RetrievedChunk
from app.models.query import CitationItem
from app.services.retrieval.reranking import rerank
from app.services.retrieval.retrieval import retrieve
from app.utilities.logger import get_logger
from app.utilities.prompt_formater import get_formatted_prompt
from app.utilities.prompts import (
    CHUNK_CONTEXT_SEPARATOR,
    RAG_CHUNK_HEADER_FORMAT,
    query_prompt,
)

log = get_logger(__name__)


def _format_conversation_context(chat_history: list[dict]) -> str:
    lines: list[str] = []
    for turn in chat_history[-10:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")
    return "\n\n".join(lines) if lines else "(none)"


def build_rag_messages(
    question: str,
    chunks: list[RetrievedChunk],
    chat_history: list[dict],
) -> list[dict]:
    context_lines: list[str] = []
    for c in chunks:
        chapter = c.chapter_title or "?"
        section = c.section or "?"
        page = c.pdf_page_number if c.pdf_page_number is not None else "?"
        header = RAG_CHUNK_HEADER_FORMAT.format(
            chunk_id=c.chunk_id,
            chapter=chapter,
            section=section,
            page=page,
            content_type=c.content_type,
        )
        context_lines.append(f"{header}\n{c.text}")
    context_block = CHUNK_CONTEXT_SEPARATOR.join(context_lines) or "(no chunks retrieved)"

    return get_formatted_prompt(
        query_prompt,
        user_query=question,
        conversation_context=_format_conversation_context(chat_history),
        context_block=context_block,
    )


def _citation_display_key(rc: RetrievedChunk) -> tuple:
    """One row per distinct source: URL when present, else chapter/section/page."""
    u = (rc.citation_url or "").strip()
    if u:
        return ("url", u)
    return (
        "loc",
        rc.chapter_title or "",
        rc.section or "",
        rc.pdf_page_number if rc.pdf_page_number is not None else -1,
    )


def citations_from_reranked(
    cited_chunk_ids: list[str],
    reranked_by_id: dict[str, RetrievedChunk],
) -> list[CitationItem]:
    """Structured citations from reranked chunks; unique by source URL or (chapter, section, page)."""
    out: list[CitationItem] = []
    seen_chunk_ids: set[str] = set()
    seen_display_keys: set[tuple] = set()
    for cid in cited_chunk_ids:
        if cid in seen_chunk_ids:
            continue
        rc = reranked_by_id.get(cid)
        if rc is None:
            continue
        dk = _citation_display_key(rc)
        if dk in seen_display_keys:
            continue
        seen_chunk_ids.add(cid)
        seen_display_keys.add(dk)
        out.append(
            CitationItem(
                chunk_id=cid,
                chapter_title=rc.chapter_title,
                section=rc.section,
                page=rc.pdf_page_number,
                url=rc.citation_url,
                content_type=rc.content_type,
            )
        )
    return out


def sources_from_citations(citations: list[CitationItem]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in citations:
        u = c.url
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


@dataclass
class RetrievalPipelineResult:
    answer: str
    cited_chunk_ids: list[str]
    reranked: list[RetrievedChunk]
    retrieved_count: int


async def retrieval_pipeline(
    question: str,
    pdf_id: str,
    faiss_store: FAISSStore,
    client: AsyncOpenAI,
    chat_history: list[dict],
) -> RetrievalPipelineResult:
    from app.utilities.query_llm import generate_answer

    log.info(
        "[QUERY] pipeline: start retrieval top_k=%s pdf_id=%s…",
        settings.top_k_retrieval,
        pdf_id[:16],
    )
    retrieved = await retrieve(
        question,
        pdf_id,
        faiss_store,
        client,
        top_k=settings.top_k_retrieval,
    )
    log.info(
        "[QUERY] pipeline: after vector search candidates=%s",
        len(retrieved),
    )
    reranked = await rerank(
        question,
        retrieved,
        top_k=settings.top_k_rerank,
    )
    log.info(
        "[QUERY] pipeline: after rerank context_chunks=%s (top_k=%s)",
        len(reranked),
        settings.top_k_rerank,
    )
    answer, cited_chunk_ids = await generate_answer(
        question,
        reranked,
        chat_history,
        client,
        settings.chat_model,
    )
    log.info(
        "[QUERY] pipeline: LLM done cited_chunk_ids=%s",
        len(cited_chunk_ids),
    )
    return RetrievalPipelineResult(
        answer=answer,
        cited_chunk_ids=cited_chunk_ids,
        reranked=reranked,
        retrieved_count=len(retrieved),
    )
