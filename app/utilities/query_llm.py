"""LLM utilities for chatbot and RAG."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

from fastapi import HTTPException
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from app.config import settings
from app.models.chunk import RetrievedChunk
from app.services.retrieval.query import build_rag_messages
from app.utilities.logger import get_logger

log = get_logger(__name__)


def _estimate_prompt_tokens(messages: list[dict]) -> int:
    text = json.dumps(messages)
    return max(1, len(text) // 4)


async def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    chat_history: list[dict],
    client: AsyncOpenAI,
    model: str,
) -> tuple[str, list[str]]:
    messages = build_rag_messages(question, chunks, chat_history)
    est = _estimate_prompt_tokens(messages)

    response = await client.chat.completions.create(
        model=model,
        max_tokens=1024,
        response_format={"type": "json_object"},
        messages=messages,
    )
    raw = response.choices[0].message.content or "{}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("[LLM] invalid JSON, raw_preview=%s", raw[:200])
        data = {"answer": raw, "cited_chunk_ids": []}

    answer = str(data.get("answer", "")).strip()
    cited_ids = data.get("cited_chunk_ids") or []
    if not isinstance(cited_ids, list):
        cited_ids = []

    allowed = {c.chunk_id for c in chunks}
    validated: list[str] = []
    seen: set[str] = set()
    for cid in cited_ids:
        if not isinstance(cid, str):
            continue
        cid = cid.strip()
        if not cid or cid not in allowed or cid in seen:
            continue
        seen.add(cid)
        validated.append(cid)

    log.info(
        "[LLM] model=%s prompt_tokens_est=%s cited_chunk_ids=%s",
        model,
        est,
        validated,
    )
    return answer, validated


async def query_llm(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    temperature: float,
    schema_name: str,
    schema: dict[str, Any],
    schema_strictness: bool = True,
    model: Optional[str] = None,
) -> str:
    """
    Query LLM with structured output schema (OpenAI json_schema response format).
    """
    try:
        model_to_use = model or settings.chat_model
        response = await client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=temperature,
            seed=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": schema_strictness,
                },
            },
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    except RateLimitError as e:
        log.exception("RateLimitError from OpenAI")
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Error code: 429 - {str(e)}",
            },
        ) from e

    except APIConnectionError as e:
        log.exception("APIConnectionError from OpenAI")
        raise HTTPException(
            status_code=503, detail={"error": "api_connection_error", "message": str(e)}
        ) from e

    except APITimeoutError as e:
        log.exception("APITimeoutError from OpenAI")
        raise HTTPException(
            status_code=504,
            detail={
                "error": "openai_timeout",
                "message": "OpenAI request timed out. Please retry.",
            },
        ) from e

    except Exception as e:
        log.exception("General error in query_llm")
        raise HTTPException(
            status_code=500, detail={"error": "unknown_error", "message": str(e)}
        ) from e
