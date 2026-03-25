# Application flows

End-to-end behavior for the RAG API and UI. For components and file layout, see [architecture.md](architecture.md).

---

## 1. Application startup

1. **FastAPI** (`app.main:app`) runs with a lifespan hook.
2. **SQLite** is initialized (`conversations_store.init_db`): schema for `ingested_pdfs`, `sessions`, `messages`.
3. **FAISS** is loaded from `FAISS_LOCAL_DIR` or left empty until first ingest.
4. A shared **AsyncOpenAI** client is attached to `app.state` for ingest (vision captions) and query (chat).

The **Streamlit** app (`frontend/main.py`) does not start with FastAPI; it calls the HTTP API using `API_BASE` (default `http://127.0.0.1:8000`). Use the root `main.py` runner to start both processes together.

---

## 2. Ingest flow (`POST /api/ingest`)

| Step | What happens |
|------|----------------|
| 1 | Multipart PDF is read; **SHA-256** of raw bytes becomes `pdf_id` / `document_id`. |
| 2 | If `pdf_id` already exists in SQLite **`ingested_pdfs`**, respond with **deduplicated** (no re-processing). |
| 3 | PDF saved to **`UPLOAD_DIR/{pdf_id}.pdf`**. |
| 4 | **Preprocessing** (`preprocessing.run_ingestion`): pdfplumber for text/tables/footer metadata; PyMuPDF for images; vision model captions images; LangChain splitter for body text → list of **`ChunkMetadata`**. |
| 5 | Each chunk becomes a LangChain **`Document`** (text + metadata); UUIDs assigned as FAISS docstore ids. |
| 6 | Optional: if **`INGEST_DEBUG_JSON=true`**, write `chunks_{prefix}.json` under **`INGEST_DEBUG_JSON_DIR`**. |
| 7 | **FAISS** `from_texts` / `add_texts` embeds chunk text (**OpenAIEmbeddings**) and updates the index; **save_local** persists to disk. |
| 8 | SQLite **`record_ingested_pdf`** stores filename, path, page/chunk counts. |

**Response:** `pdf_id`, `filename`, `chunk_count`, `page_count`, `deduplicated`.

---

## 3. Query flow (`POST /api/query`)

| Step | What happens |
|------|----------------|
| 1 | Validate `question` and `pdf_id`; resolve **session** (existing `session_id` or new UUID + `create_session`). |
| 2 | Load prior **`messages`** for session → chat history for the prompt. |
| 3 | **Retrieve:** FAISS similarity search with a larger internal `k`, then filter **`document_id == pdf_id`** → list of **`RetrievedChunk`**. |
| 4 | **Rerank:** cross-encoder scores `(query, chunk.text)`; keep top **`TOP_K_RERANK`**. |
| 5 | **LLM:** build messages from **`query_prompt`** + formatted context + history; OpenAI **`json_object`** with `answer` and `cited_chunk_ids`. |
| 6 | Validate cited ids against retrieved chunk ids; build **`CitationItem`** list (unique by citation URL or chapter/section/page). |
| 7 | Append user and assistant rows to **`messages`** (assistant row stores cited chunk ids); return **`QueryResponse`**. |

Supporting endpoints: **`GET /api/sessions`**, **`GET /api/sessions/{id}/messages`** for the UI.

---

## 4. Streamlit UI flow

1. User uploads a PDF → **`POST /api/ingest`** → `pdf_id` stored in session state.
2. User asks questions → **`POST /api/query`** with `pdf_id` and optional `session_id`.
3. UI renders answer, citations, and can list/reload sessions for the current document.

---

## 5. Shutdown

FastAPI lifespan closes the OpenAI client. FAISS index is already persisted on disk after each ingest save.
