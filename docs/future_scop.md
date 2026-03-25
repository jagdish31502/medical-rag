# Future scope

Scaling assumptions, latency goals, limitations of the current take-home stack, and concrete next steps toward production. Related: [architecture.md](architecture.md), [flow.md](flow.md).

---

## 1. Scaling thought experiment

### Many documents

Today the index is a **single flat FAISS store** with `document_id` in metadata for filtering. As corpora grow:

- **Sharding / namespaces:** partition by tenant, product line, or category so each query searches a smaller index (or a small set of shards) instead of one global `ntotal`.
- **Metadata pre-filtering:** combine vector search with constraints (category, date, ACL) via multiple indexes or a system that supports filtered ANN.
- **Tiered storage:** hot documents in memory or fast SSD; cold archives with async refresh.

### Thousands of users

At high concurrency:

- The **API tier** must scale **horizontally** (multiple uvicorn/gunicorn workers or containers) behind a load balancer.
- **SQLite** and **in-process FAISS** become bottlenecks: move sessions and ingest registry to **Postgres** (or similar) and the vector tier to a **shared, replicated** store (e.g. managed vector DB or FAISS served by dedicated GPU/CPU workers).
- **Rate limiting**, **per-tenant quotas**, and **caching** (embeddings for frequent queries, reranker batching) reduce thundering herds.
- **OpenAI** calls are the dominant cost and latency; pool connections, enforce timeouts, and consider **smaller/faster models** for some tiers.

### Target P95 latency under ~2 seconds

End-to-end query time includes: embedding the question, vector search, rerank, LLM generation, and JSON parsing. To keep **P95 ≲ 2s**:

- **Cap work per request:** lower `top_k` retrieval, smaller rerank batches, tight `max_tokens` on the answer model.
- **Async everywhere:** non-blocking I/O to vector and LLM services; avoid holding a global lock during slow operations where possible.
- **SLAs on dependencies:** regional OpenAI routing, retries with backoff, circuit breakers.
- **Measure:** distributed tracing (trace ids per query) and histograms on each stage to see whether P95 is dominated by retrieval, rerank, or LLM.
- **Enable streaming:** **Server-Sent Events (SSE)** or chunked HTTP for the **LLM token stream** improves **time-to-first-token (TTFT)** and *perceived* latency even when total generation time is unchanged—users see partial answers immediately. 

Together, **many documents**, **thousands of users**, and **P95 &lt; ~2s** imply moving off single-node SQLite + local FAISS and treating the vector store and metadata DB as **shared, monitored services**, with the query path optimized and horizontally scaled.

---

## 2. Limitations and next steps toward a production-ready system

**Current limitations (illustrative):**

- Single-process FAISS and SQLite: no HA, limited write concurrency, backup/restore is manual.
- No authentication/authorization on API or documents.
- Ingest is synchronous: large PDFs block the HTTP request and tie up a worker.
- No formal eval harness (ground-truth Q&A, citation accuracy, latency SLOs).

**Next steps toward production:**

- **Observability:** structured logs, metrics (latency histograms, error rates), optional tracing.
- **Security:** API keys or OAuth, encryption at rest for uploads, secrets in a vault.
- **Data lifecycle:** retention policies, GDPR-style delete, versioned indexes per ingest.
- **CI/CD:** tests for ingest/query, linting, container images, staged rollouts.

The items below are specific product/engineering enhancements that align with this direction.

---

## 3. Planned enhancements (from roadmap)

### 3.1 Multi-upload and categories

- Support **multiple uploads** per session or batch API (zip / file list) with validation and size limits.
- Add a **category name** (and optionally tags or folder path) on ingest so documents stay **structured** in metadata and UI (filter search by category, list by tenant).
- Persist category on **`ingested_pdfs`** (schema migration) and thread it into chunk metadata for filtered retrieval.

### 3.2 Background ingest and status monitoring

- **Uploading and ingesting** large PDFs (especially with vision captions) is **slow**; it should not block the HTTP thread.
- Move ingestion to **background workers** (**Celery** + Redis/RabbitMQ, or **RQ**, **Arq**, or cloud task queues) with a durable **job id**.
- Expose **job status** APIs (`queued`, `running`, `succeeded`, `failed`, progress %) and a **monitor** in the UI (polling or WebSocket) so users stay up to date without timing out the browser.

### 3.3 Embedding model: `text-embedding-3-small`

- `text-embedding-3-small` balances **strong retrieval quality** with **low cost and latency** versus larger embedding models, and its context fits long chunks; it pairs well with OpenAI chat models in a single-vendor stack for simplicity.
- **Today (figures):** images are described with a **vision LLM caption**, then embedded like any other **text** chunk in the same FAISS index.

### 3.4 Chunk size (currently ~400 tokens)

- Chunk size is set to **~400 tokens** (`CHUNK_MAX_TOKENS`) as a reasonable default for mixed PDFs.
- **Future:** tune chunk size (and overlap) **per document type or corpus** after **analysis** (e.g. distribution of paragraph length, retrieval hit rate, manual eval on citations)—not a one-size-fits-all constant.

### 3.5 Chunking strategy: `RecursiveCharacterTextSplitter`

- **Current:** Body text is split with LangChain’s **`RecursiveCharacterTextSplitter`** (`langchain_text_splitters`), using a separator order such as paragraphs, newlines, sentence boundaries, then words—so splits prefer **coherent boundaries** before falling back to smaller breaks.
- Chunk budget is expressed in **characters** derived from `CHUNK_MAX_TOKENS` (approx. ×4 chars/token) plus configurable **overlap** (`CHUNK_OVERLAP_PCT`); this is a practical proxy, not exact tokenizer counts.
- **Future:** experiment with **structure-aware** splitting (headings, TOC, page boxes already used for body vs footer), **semantic chunking** (embed sentences then merge), or **late chunking** for long contexts—especially once corpus-specific chunk sizes are measured.

### 3.6 Object storage (S3 / Azure Blob) and read-only URLs in citations

- **Store originals in the cloud:** persist uploaded PDFs (or normalized derivatives) in **Amazon S3**, **Azure Blob Storage**, or equivalent instead of (or in addition to) local `UPLOAD_DIR`. Ingest workers read from `s3://` / `https://…blob…` via SDK or pre-signed GET; metadata DB holds **bucket, object key, version id**, and optional checksum.
- **Citations for humans:** expose **read-only** links in the API/UI—e.g. **time-limited pre-signed URLs** (S3 presigned GET, SAS tokens for Blob) or a tiny **download proxy** endpoint that checks auth then streams the object. Do not put long-lived secret-bearing URLs in logs; regenerate on demand.
- **Separation of concerns:** chunk text and LibreTexts-style **citation URLs** in metadata can remain as today for attribution; the **canonical document** link in `CitationItem` (or a sibling field) can point at the **tenant’s copy** in object storage when the source is user-uploaded, while public textbook URLs stay as-is.
- **Policies:** bucket **block public access**, encryption at rest, lifecycle rules for stale uploads, and IAM scoped to ingest/query roles.

### 3.7 Streaming (reduce perceived latency)

- Enable **token streaming** on the query path (e.g. **SSE**, **WebSocket**, or NDJSON) so the UI shows partial output as it is generated—better **time-to-first-byte** and perceived responsiveness.
- Pipe the LLM stream through FastAPI (**`StreamingResponse`** or similar) and consume it from Streamlit or other clients (reuse patterns such as an async iterator over model deltas).

### 3.8 Vision embedding models (images as input)

- **Direction:** use **multimodal / vision embedding** APIs that accept **images (or image patches) directly** and return a vector, instead of (or alongside) caption → `text-embedding-*`.
- **Benefits:** captures layout, color, and non-text structure that captions may omit; one fewer generative step for ingest if embeddings are enough for retrieval.
- **Design choices:** align **dimensionality and similarity metric** with text vectors if you use a **shared** index (e.g. joint text–image space), or keep **separate** image and text indexes and **fuse** scores at query time; for text-only queries, retrieve from the text index as today and optionally merge with an image index when the query or UI supplies a visual cue.
- **Operational:** batching, rate limits, image size caps, and fallbacks (e.g. caption path) when the vision embedder fails.

---

## 4. Summary

| Theme | Direction |
|-------|-----------|
| **Scale** | Sharded or filtered indexes, horizontal API tier, shared vector + OLTP stores |
| **Users** | Auth, quotas, caching, async workers |
| **P95 ~2s** | Stage-level budgets, smaller prompts where safe, **streaming** for TTFT/UX, observability, resilient LLM calls |
| **Production** | Background ingest with status, categories/multi-doc, evaluated chunking & splitting, object storage + read-only citation URLs, streaming query API, optional **vision embeddings** for figures, hardened ops |

This document is a living note for scope beyond the current take-home implementation.
