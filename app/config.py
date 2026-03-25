from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"
    vision_model: str = "gpt-4o"
    faiss_local_dir: str = "data/faiss_lc"
    db_path: str = "data/rag.db"
    upload_dir: str = "data/uploads"
    top_k_retrieval: int = 10
    top_k_rerank: int = 4

    # PDF preprocessing / ingestion (optional overrides)
    decorative_image_width: int = 183
    decorative_image_height: int = 51
    min_content_image_kb: float = 5.0
    chunk_max_tokens: int = 400
    chunk_overlap_pct: float = 0.10

    # When True, each ingest writes chunks + metadata as JSON under ingest_debug_json_dir
    ingest_debug_json: bool = False
    ingest_debug_json_dir: str = "data/debug/ingest"


settings = Settings()
