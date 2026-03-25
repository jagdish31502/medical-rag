from __future__ import annotations

from fastapi import APIRouter, File, Request, UploadFile
from openai import AsyncOpenAI

from app.db.faiss_store import FAISSStore
from app.models.query import IngestResponse
from app.services.ingestion.ingest import ingesting_pipeline
from app.utilities.logger import get_logger

router = APIRouter(tags=["ingest"])
log = get_logger(__name__)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    request: Request,
    file: UploadFile = File(...),
):
    faiss_store: FAISSStore = request.app.state.faiss_store
    client: AsyncOpenAI = request.app.state.openai_client

    file_bytes = await file.read()
    filename = file.filename or "upload.pdf"
    log.info("[INGEST] POST /api/ingest: upload read filename=%s bytes=%s", filename, len(file_bytes))

    return await ingesting_pipeline(
        file_bytes=file_bytes,
        filename=filename,
        faiss_store=faiss_store,
        openai_client=client,
    )
