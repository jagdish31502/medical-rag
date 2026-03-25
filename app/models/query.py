from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    pdf_id: str = Field(description="SHA-256 hex from ingest")


class CitationItem(BaseModel):
    chunk_id: str
    chapter_title: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    url: Optional[str] = None
    content_type: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationItem]
    session_id: str
    sources: List[str]


class IngestResponse(BaseModel):
    pdf_id: str
    filename: str
    chunk_count: int
    page_count: int
    deduplicated: bool


class SessionsListResponse(BaseModel):
    sessions: list[dict]
