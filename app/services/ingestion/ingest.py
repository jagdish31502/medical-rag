"""Ingestion pipeline: persist PDF, embed chunks, record metadata."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from openai import AsyncOpenAI

from app.config import settings
from app.db import conversations_store as db
from app.db.faiss_store import FAISSStore
from app.models.chunk import ChunkMetadata
from app.models.query import IngestResponse
from app.services.ingestion.preprocessing import run_ingestion
from app.utilities.logger import get_logger

log = get_logger(__name__)


def _write_chunks_debug_json(
    *,
    chunks: list[ChunkMetadata],
    docstore_ids: list[str],
    pdf_id: str,
    filename: str,
    page_count: int,
) -> None:
    out_dir = Path(settings.ingest_debug_json_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chunks_{pdf_id[:16]}.json"
    payload = {
        "pdf_id": pdf_id,
        "filename": filename,
        "page_count": page_count,
        "chunk_count": len(chunks),
        "chunks": [
            {
                **c.model_dump(mode="json"),
                "docstore_id": did,
            }
            for c, did in zip(chunks, docstore_ids, strict=True)
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("[INGEST] debug: wrote chunks JSON path=%s", out_path.resolve())


def _chunk_to_document(c: ChunkMetadata) -> Document:
    return Document(
        page_content=c.text,
        metadata={
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "content_type": c.content_type,
            "pdf_page_number": c.pdf_page_number if c.pdf_page_number is not None else -1,
            "chapter": c.chapter if c.chapter is not None else -1,
            "chapter_title": c.chapter_title or "",
            "section": c.section or "",
            "section_page": c.section_page or "",
            "citation_url": c.citation_url or "",
            "libretexts_page_id": c.libretexts_page_id or "",
        },
    )


async def ingesting_pipeline(
    *,
    file_bytes: bytes,
    filename: str,
    faiss_store: FAISSStore,
    openai_client: AsyncOpenAI,
) -> IngestResponse:
    pdf_id = hashlib.sha256(file_bytes).hexdigest()

    log.info(
        "[INGEST] 1. PDF received: filename=%s size_bytes=%s",
        filename,
        len(file_bytes),
    )
    log.info("[INGEST] 2. document_id (sha256 prefix): %s…", pdf_id[:16])

    if await db.ingested_pdf_exists(pdf_id):
        log.info(
            "[INGEST] 3. Duplicate document — already ingested, skipping pipeline (pdf_id=%s…)",
            pdf_id[:16],
        )
        return IngestResponse(
            pdf_id=pdf_id,
            filename=filename,
            chunk_count=0,
            page_count=0,
            deduplicated=True,
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{pdf_id}.pdf"
    log.info("[INGEST] 3. Saving PDF to disk: %s", dest.resolve())
    dest.write_bytes(file_bytes)
    log.info(
        "[INGEST] 4. PDF saved: path=%s bytes_written=%s",
        dest.resolve(),
        len(file_bytes),
    )

    log.info(
        "[INGEST] 5. Parsing PDF (text, tables, image captions) vision_model=%s",
        settings.vision_model,
    )
    chunks, page_count = await run_ingestion(
        pdf_path=str(dest),
        filename=filename,
        document_id=pdf_id,
        openai_client=openai_client,
        vision_model=settings.vision_model,
    )

    by_type = Counter(c.content_type for c in chunks)
    log.info(
        "[INGEST] 6. Parse complete: pages=%s total_chunks=%s by_type=%s",
        page_count,
        len(chunks),
        dict(by_type),
    )

    documents = [_chunk_to_document(c) for c in chunks]
    ids = [str(uuid4()) for _ in documents]

    if settings.ingest_debug_json:
        _write_chunks_debug_json(
            chunks=chunks,
            docstore_ids=ids,
            pdf_id=pdf_id,
            filename=filename,
            page_count=page_count,
        )

    log.info(
        "[INGEST] 7. Vector index: embedding %s chunks into FAISS",
        len(documents),
    )
    await faiss_store.add_documents_async(documents, ids)
    log.info("[INGEST] 8. Persisting FAISS index to disk")
    await faiss_store.save_async()

    log.info(
        "[INGEST] 9. Database: recording ingested_pdf (pdf_id=%s… chunks=%s pages=%s)",
        pdf_id[:16],
        len(chunks),
        page_count,
    )
    await db.record_ingested_pdf(
        pdf_id=pdf_id,
        filename=filename,
        file_path=str(dest),
        page_count=page_count,
        chunk_count=len(chunks),
    )

    log.info(
        "[INGEST] 10. Ingest pipeline finished: pdf_id=%s… chunks=%s pages=%s",
        pdf_id[:16],
        len(chunks),
        page_count,
    )

    return IngestResponse(
        pdf_id=pdf_id,
        filename=filename,
        chunk_count=len(chunks),
        page_count=page_count,
        deduplicated=False,
    )
