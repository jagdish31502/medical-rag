from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str
    pdf_id: str = Field(description="Same as document_id (SHA-256 hex) for this app")
    content_type: Literal["text", "table", "image_caption"]
    text: str
    pdf_page_number: Optional[int] = None
    chapter: Optional[int] = None
    chapter_title: Optional[str] = None
    section: Optional[str] = None
    section_page: Optional[str] = None
    citation_url: Optional[str] = None
    libretexts_page_id: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    citation_url: Optional[str] = None
    chapter_title: Optional[str] = None
    section: Optional[str] = None
    pdf_page_number: Optional[int] = None
    content_type: str
