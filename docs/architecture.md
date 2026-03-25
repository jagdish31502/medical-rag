# Architecture

High-level design of the RAG stack. For step-by-step runtime behavior, see [flow.md](flow.md).

---

## 1. Overview


| Layer                            | Role                                                                                                                                                       |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Routes** (`app/routes/`)       | HTTP only: validate input, call services, return Pydantic models.                                                                                          |
| **Services** (`app/services/`)   | **Ingestion** (PDF → chunks → FAISS), **retrieval** (search → rerank → pipeline). Chunk embeddings run inside **FAISSStore** (LangChain `OpenAIEmbeddings`). |
| **Data** (`app/db/`)             | **FAISSStore** (LangChain FAISS + OpenAI embeddings), **conversations_store** (SQLite: ingest registry + chat).                                            |
| **Utilities** (`app/utilities/`) | Prompts, prompt formatting, LLM helpers, logging.                                                                                                          |
| **Frontend** (`frontend/`)       | Streamlit UI over `httpx`.                                                                                                                                 |


**Configuration:** `app/config.py` + `.env` via **pydantic-settings** (`OPENAI_API_KEY` required).

---

## 2. Data stores

### Vector index (FAISS)

- **Path:** `FAISS_LOCAL_DIR` (e.g. `data/faiss` or `data/faiss_lc`).
- **Embeddings:** `langchain_openai.OpenAIEmbeddings` with `**EMBEDDING_MODEL`** (default `text-embedding-3-small`), inner-product distance.
- **Documents:** LangChain `**Document`**: `page_content` = chunk text; **metadata** includes `chunk_id`, `document_id`, `content_type`, page, chapter, section, LibreTexts URLs, etc.
- **Docstore IDs:** random UUIDs per vector row; logical identity for filtering and citations is `**metadata["chunk_id"]`**.

### SQLite (`DB_PATH`)

- `**ingested_pdfs`:** dedup key `pdf_id`, file path, page/chunk counts.
- `**sessions`:** `session_id` → `document_id` (`pdf_id`).
- `**messages`:** chat turns with optional `**cited_chunks`** JSON array of chunk ids.

No per-chunk table in SQL; chunks live only in FAISS + PDF on disk.

**Future:** originals may live in **S3 or Azure Blob**; API citations can expose **read-only** URLs (e.g. pre-signed GETs). See **[future_scop.md](future_scop.md)** (§3.6).

---

## 3. Ingestion stack

- `**services/ingestion/ingest.py` — `ingesting_pipeline`:** hash, dedup, save PDF, call preprocessing, optional debug JSON, FAISS add/save, SQLite record.
- `**services/ingestion/preprocessing.py`:** chapter/skip-page heuristics, tables → markdown, PyMuPDF images → `**caption_image_with_llm`** ( `**VISION_MODEL**` ), text splitting via `**CHUNK_MAX_TOKENS**` / overlap from settings.

---

## 4. Retrieval stack

- `**services/retrieval/retrieval.py` — `retrieve`:** similarity search, filter by `document_id`.
- `**services/retrieval/reranking.py`:** **CrossEncoder** `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU, `asyncio.to_thread`).
- `**services/retrieval/query.py` — `retrieval_pipeline`:** retrieve → rerank → `**generate_answer`**; `**citations_from_reranked**` dedupes display sources by URL or location.

---

## 5. LLM usage


| Use case       | Model settings     | Mechanism                                                                              |
| -------------- | ------------------ | -------------------------------------------------------------------------------------- |
| RAG answer     | `**CHAT_MODEL**`   | Chat Completions, `**response_format: json_object**`, prompts from `**query_prompt**`. |
| Image captions | `**VISION_MODEL**` | Multimodal chat with image URL + `**image_caption_prompt**`.                           |


Shared **AsyncOpenAI** client from app state (API key from env).

---

## 6. Process layout

- **API:** `uvicorn app.main:app` (e.g. port **8000**).
- **UI:** `streamlit run frontend/main.py` (e.g. port **8501**), `**API_BASE`** must point at the API.
- **Combined:** `python main.py` from repo root starts both subprocesses and sets `**API_BASE`** for Streamlit.

---

## 7. Notable files


| Path                            | Purpose                                         |
| ------------------------------- | ----------------------------------------------- |
| `app/main.py`                   | FastAPI app, lifespan, router include.          |
| `app/db/faiss_store.py`         | FAISS load/save/add/search.                     |
| `app/db/conversations_store.py` | SQLite async access.                            |
| `app/utilities/prompts.py`      | `query_prompt`, `image_caption_prompt`.         |
| `app/utilities/query_llm.py`    | `generate_answer`, optional structured helpers. |
| `main.py` (repo root)           | Uvicorn + Streamlit launcher.                   |


