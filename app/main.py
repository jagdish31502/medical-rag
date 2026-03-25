from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncOpenAI

from app.config import settings
from app.db import conversations_store as db
from app.db.faiss_store import FAISSStore
from app.routes import ingest, query
from app.utilities.logger import get_logger

log = get_logger(__name__)

faiss_store = FAISSStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[MAIN] Initializing SQLite: path=%s", settings.db_path)
    await db.init_db()
    log.info("[MAIN] Loading FAISS: dir=%s", settings.faiss_local_dir)
    faiss_store.load_or_create()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    app.state.faiss_store = faiss_store
    app.state.openai_client = client
    log.info(
        "[MAIN] OpenAI client ready (chat_model=%s vision_model=%s embedding=%s)",
        settings.chat_model,
        settings.vision_model,
        settings.embedding_model,
    )
    log.info("[MAIN] RAG API startup complete — routes mounted at /api")
    yield
    await client.close()


app = FastAPI(title="RAG API", lifespan=lifespan)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Liveness/readiness probe for load balancers and orchestrators."""
    return {"status": "ok"}


app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")
