from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from openai import AsyncOpenAI

from app.db import conversations_store as db
from app.db.faiss_store import FAISSStore
from app.models.query import QueryRequest, QueryResponse, SessionsListResponse
from app.services.retrieval.query import (
    citations_from_reranked,
    retrieval_pipeline,
    sources_from_citations,
)
from app.utilities.logger import get_logger

router = APIRouter(tags=["query"])
log = get_logger(__name__)


@router.get("/sessions/{session_id}/messages")
async def session_messages(session_id: str):
    """Chat history for reloading a session in the UI."""
    if not await db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    rows = await db.get_messages(session_id)
    return {"messages": rows}


@router.get("/sessions", response_model=SessionsListResponse)
async def list_sessions_endpoint(pdf_id: str):
    """List chat sessions for a document (document_id == pdf_id hash)."""
    rows = await db.list_sessions(pdf_id)
    return SessionsListResponse(sessions=rows)


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(http_request: Request, request: QueryRequest):
    faiss_store: FAISSStore = http_request.app.state.faiss_store
    client: AsyncOpenAI = http_request.app.state.openai_client

    pdf_id = request.pdf_id
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    log.info(
        "[QUERY] 1. Request: pdf_id=%s… question_preview=%s session_id_provided=%s",
        pdf_id[:16],
        question[:120],
        request.session_id is not None,
    )

    session_id = request.session_id or str(uuid.uuid4())
    if request.session_id:
        doc = await db.get_session_document(session_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="session not found")
        if doc != pdf_id:
            raise HTTPException(
                status_code=400, detail="session does not belong to this pdf_id"
            )
        await db.touch_session(session_id)
    else:
        if not await db.ingested_pdf_exists(pdf_id):
            raise HTTPException(
                status_code=400,
                detail="pdf_id has not been ingested; call POST /api/ingest first",
            )
        await db.create_session(session_id, pdf_id)

    history_rows = await db.get_messages(session_id)
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history_rows]
    log.info(
        "[QUERY] 2. Session ready: session_id=%s… prior_turns=%s",
        session_id[:8],
        len(chat_history),
    )

    pipeline = await retrieval_pipeline(
        question,
        pdf_id,
        faiss_store,
        client,
        chat_history,
    )

    reranked_by_id = {c.chunk_id: c for c in pipeline.reranked}
    citations = citations_from_reranked(pipeline.cited_chunk_ids, reranked_by_id)
    sources = sources_from_citations(citations)
    cited_ids = [c.chunk_id for c in citations]

    await db.add_message(session_id, "user", question, None)
    await db.add_message(session_id, "assistant", pipeline.answer, cited_ids)
    await db.touch_session(session_id)

    log.info(
        "[QUERY] 3. Response: answer_chars=%s citations=%s unique_sources=%s",
        len(pipeline.answer),
        len(citations),
        len(sources),
    )
    log.info(
        "[QUERY] 4. Stored user+assistant messages session_id=%s…",
        session_id[:8],
    )

    return QueryResponse(
        answer=pipeline.answer,
        citations=citations,
        session_id=session_id,
        sources=sources,
    )
