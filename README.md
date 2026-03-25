# RAG PDF Q&A (FastAPI + FAISS + Streamlit)

Ask questions over ingested PDFs with citations. Backend: **FastAPI**, **LangChain FAISS**, **OpenAI** embeddings and chat. UI: **Streamlit**.

---

## Documentation

| Document | Description |
|----------|-------------|
| **[docs/architecture.md](docs/architecture.md)** | Components, data stores, retrieval stack, configuration. |
| **[docs/flow.md](docs/flow.md)** | Startup, ingest, query, and UI flows step by step. |
| **[docs/future_scop.md](docs/future_scop.md)** | Scaling, P95 goals, limitations, and production next steps. |

---

## Prerequisites

- **Python 3.11+** (3.12 recommended)
- **OpenAI API key** with access to chat and embedding models you configure

---

## Setup (step by step)

### 1. Clone and enter the project

```bash
cd takehomeassessment
```

(Use your actual repository path.)

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

First run may download the cross-encoder model for reranking (~90MB).

### 4. Configure environment variables

Copy the example env file and edit it:

```bash
copy .env.example .env
```

On macOS/Linux use `cp .env.example .env`.

**Required:**

- `OPENAI_API_KEY` — your secret key

**Common optional values** (defaults exist in `app/config.py`; see `.env.example`):

- `CHAT_MODEL`, `VISION_MODEL`, `EMBEDDING_MODEL`
- `FAISS_LOCAL_DIR`, `DB_PATH`, `UPLOAD_DIR`
- `TOP_K_RETRIEVAL`, `TOP_K_RERANK`
- `API_BASE` — base URL of the API (used by Streamlit; the root `main.py` overrides this for the UI child process)
- `INGEST_DEBUG_JSON=true` — write chunk dumps under `INGEST_DEBUG_JSON_DIR` during ingest

### 5. Run the application

**Option A — API and UI together (recommended for local demo)**

From the **repository root**:

```bash
python main.py
```

With hot reload on the API:

```bash
python main.py --reload
```

Custom ports:

```bash
python main.py --api-port 8000 --streamlit-port 8501 --host 127.0.0.1
```

Then open the Streamlit URL printed in the terminal (default **http://127.0.0.1:8501**). The API is typically at **http://127.0.0.1:8000**.

**Option B — Two terminals**

Terminal 1 — API:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — UI (set `API_BASE` if the API is not on port 8000):

```bash
streamlit run frontend/main.py
```

### 6. Use the system

1. In Streamlit, upload a PDF and wait for ingest to finish (or use the API: `POST /api/ingest`).
2. Ask questions; answers include citations when the model uses retrieved context.
3. API docs: **http://127.0.0.1:8000/docs** (when the API is running).

---

## Project layout (short)

- `app/` — FastAPI application (routes, services, DB, utilities)
- `frontend/` — Streamlit app
- `docs/` — **[architecture.md](docs/architecture.md)** and **[flow.md](docs/flow.md)**
- `main.py` — launches uvicorn + Streamlit
- `data/` — local FAISS index, SQLite DB, uploads (created at runtime; paths from `.env`)

---

## Troubleshooting

- **Empty retrieval:** confirm ingest completed for that `pdf_id` and `FAISS_LOCAL_DIR` matches the index you built.
- **Streamlit cannot reach API:** check `API_BASE` and firewall; ensure uvicorn is running before using the UI.
- **SQLite / schema:** legacy tables may be migrated on startup; re-ingest if instructed in logs.

For deeper detail, start with **[docs/flow.md](docs/flow.md)** and **[docs/architecture.md](docs/architecture.md)**.
