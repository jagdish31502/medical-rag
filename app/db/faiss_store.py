from __future__ import annotations

import asyncio
from pathlib import Path
# LangChain 0.2+: FAISS lives in langchain_community (not langchain.vectorstores).
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.utilities.logger import get_logger

log = get_logger(__name__)


class FAISSStore:
    """LangChain local FAISS: chunk text + full metadata on each ``Document``.

    Docstore ids are caller-supplied (e.g. UUIDs); logical chunk id stays in
    ``metadata["chunk_id"]`` for filtering and citations.
    """

    def __init__(self) -> None:
        self._write_lock = asyncio.Lock()
        self._dir = Path(settings.faiss_local_dir)
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._vs: FAISS | None = None

    def load_or_create(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        index_faiss = self._dir / "index.faiss"
        if index_faiss.exists():
            self._vs = FAISS.load_local(
                str(self._dir),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            log.info(
                "[FAISS] LangChain loaded ntotal=%s path=%s",
                self._vs.index.ntotal,
                self._dir,
            )
        else:
            self._vs = None
            log.info("[FAISS] LangChain no index yet; will create on first ingest path=%s", self._dir)

    def _ensure_vs(self) -> FAISS:
        if self._vs is None:
            raise RuntimeError("FAISS store is empty; ingest at least one document first")
        return self._vs

    def add_documents_sync(self, documents: list[Document], ids: list[str]) -> None:
        if len(documents) != len(ids):
            raise ValueError("documents and ids length mismatch")
        texts = [d.page_content for d in documents]
        metadatas = [dict(d.metadata) for d in documents]
        if self._vs is None:
            self._vs = FAISS.from_texts(
                texts,
                self._embeddings,
                metadatas=metadatas,
                ids=ids,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
        else:
            self._vs.add_texts(texts, metadatas=metadatas, ids=ids)
        log.info(
            "[FAISS] LangChain add_documents n=%s ntotal=%s",
            len(documents),
            self._vs.index.ntotal,
        )

    def save_local_sync(self) -> None:
        vs = self._ensure_vs()
        self._dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(self._dir))
        log.info("[FAISS] LangChain save_local path=%s ntotal=%s", self._dir, vs.index.ntotal)

    def similarity_search_with_score_sync(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]:
        if self._vs is None or self._vs.index.ntotal == 0:
            return []
        kk = min(k, max(1, int(self._vs.index.ntotal)))
        return self._vs.similarity_search_with_score(query, k=kk)

    async def add_documents_async(
        self, documents: list[Document], ids: list[str]
    ) -> None:
        async with self._write_lock:

            def _run() -> None:
                self.add_documents_sync(documents, ids)

            await asyncio.to_thread(_run)

    async def save_async(self) -> None:
        async with self._write_lock:

            def _run() -> None:
                self.save_local_sync()

            await asyncio.to_thread(_run)

    async def similarity_search_with_score_async(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]:
        def _run() -> list[tuple[Document, float]]:
            return self.similarity_search_with_score_sync(query, k)

        return await asyncio.to_thread(_run)
