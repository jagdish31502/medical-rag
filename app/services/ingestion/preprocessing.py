"""PDF preprocessing: text/table chunking, footers, and image captioning."""

from __future__ import annotations

import base64
import re
from collections import Counter
from typing import Any

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI

from app.config import settings
from app.models.chunk import ChunkMetadata
from app.utilities.logger import get_logger
from app.utilities.prompt_formater import get_formatted_prompt
from app.utilities.prompts import image_caption_prompt

log = get_logger(__name__)

# --- chunking / PDF helpers (from former chunker.py) ---


def extract_chapter_titles(pdf_path: str) -> dict[int, str]:
    chapter_titles: dict[int, str] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:10]:
            text = page.extract_text() or ""
            for line in text.splitlines():
                match = re.match(r"^(\d+):\s+(.+)$", line.strip())
                if match:
                    num = int(match.group(1))
                    title = match.group(2).strip()
                    chapter_titles[num] = title
    return chapter_titles


def detect_skip_pages(pdf_path: str) -> set[int]:
    skip_pages: set[int] = set()
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            footer_text = _extract_footer_text(page)
            has_url = bool(
                re.search(r"https://bio\.libretexts\.org/@go/page/\d+", footer_text)
            )
            if not has_url:
                skip_pages.add(page_num)
    return skip_pages


def _extract_footer_text(page: Any) -> str:
    words = page.extract_words()
    cutoff = page.height * 0.92
    return " ".join(w["text"] for w in words if w["top"] > cutoff)


def extract_footer_metadata(page: Any) -> dict[str, str | None]:
    footer_text = _extract_footer_text(page)
    section_page_match = re.search(r"\b(\d+\.\d+\.\d+)\b", footer_text)
    url_match = re.search(
        r"(https://bio\.libretexts\.org/@go/page/(\d+))", footer_text
    )
    return {
        "section_page": section_page_match.group(1) if section_page_match else None,
        "citation_url": url_match.group(1) if url_match else None,
        "libretexts_page_id": url_match.group(2) if url_match else None,
    }


def parse_section_from_footer(section_page: str | None) -> dict[str, int | str | None]:
    if not section_page:
        return {"chapter": None, "section": None}
    parts = section_page.split(".")
    return {
        "chapter": int(parts[0]),
        "section": f"{parts[0]}.{parts[1]}",
    }


def is_real_table(table: list) -> bool:
    if not table or len(table) < 2:
        return False
    if not table[0] or len(table[0]) < 2:
        return False
    max_cell_len = max(
        len(str(cell or "")) for row in table for cell in row
    )
    return max_cell_len < 400


def table_to_markdown(table: list) -> str:
    if not table or not table[0]:
        return ""

    def clean(cell: Any) -> str:
        if cell is None:
            return ""
        return str(cell).replace("\n", " ").strip()

    header = "| " + " | ".join(clean(c) for c in table[0]) + " |"
    separator = "| " + " | ".join("---" for _ in table[0]) + " |"
    rows = [
        "| " + " | ".join(clean(c) for c in row) + " |" for row in table[1:]
    ]
    return "\n".join([header, separator] + rows)


def is_decorative_image(img_meta: tuple) -> bool:
    width, height = img_meta[2], img_meta[3]
    img_type = img_meta[8] if len(img_meta) > 8 else ""
    if (width, height) == (
        settings.decorative_image_width,
        settings.decorative_image_height,
    ):
        return True
    if "smask" in str(img_type).lower():
        return True
    return False


def make_chunk_id(
    document_id: str,
    pdf_page: int,
    content_type: str,
    index: int,
    section: str | None = None,
) -> str:
    section_part = f"_s{section}" if section else ""
    return f"{document_id}{section_part}_p{pdf_page}_{content_type}_{index}"


def approximate_token_count(text: str) -> int:
    return max(1, len(text) // 4)


_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    chunk_size=settings.chunk_max_tokens * 4,
    chunk_overlap=int(
        settings.chunk_max_tokens * 4 * settings.chunk_overlap_pct
    ),
    length_function=len,
    is_separator_regex=False,
)


def split_into_chunks(text: str, max_tokens: int | None = None) -> list[str]:
    default_tokens = settings.chunk_max_tokens
    mt = default_tokens if max_tokens is None else max_tokens
    if mt == default_tokens:
        return _splitter.split_text(text)
    chunk_size = mt * 4
    chunk_overlap = int(chunk_size * settings.chunk_overlap_pct)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)


# --- image extraction & captions (from former image_caption.py) ---


def extract_image_bytes(doc: Any, xref: int) -> tuple[bytes, str] | None:
    try:
        base_image = doc.extract_image(xref)
        size_kb = len(base_image["image"]) / 1024
        if size_kb < settings.min_content_image_kb:
            return None
        return base_image["image"], base_image["ext"]
    except Exception:
        return None


async def caption_image_with_llm(
    image_bytes: bytes,
    image_ext: str,
    page_num: int,
    section: str | None,
    chapter_title: str | None,
    client: AsyncOpenAI,
    model: str,
) -> str:
    ext_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    media_type = ext_map.get(image_ext.lower(), "image/jpeg")
    b64_data = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{media_type};base64,{b64_data}"

    book_context = (
        f"chapter '{chapter_title}'"
        if chapter_title
        else "a biology/biosciences textbook"
    )
    section_context = f"section {section}" if section else "an unspecified section"

    messages = get_formatted_prompt(
        image_caption_prompt,
        page_num=page_num,
        section_context=section_context,
        book_context=book_context,
    )
    user_text = messages[1]["content"]
    messages[1]["content"] = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    response = await client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=messages,
    )
    choice = response.choices[0].message.content
    return (choice or "").strip()


# --- full PDF → ChunkMetadata (from former ingest.py service) ---


async def run_ingestion(
    pdf_path: str,
    filename: str,
    document_id: str,
    openai_client: AsyncOpenAI,
    vision_model: str,
) -> tuple[list[ChunkMetadata], int]:
    """
    Parse PDF at pdf_path; return (chunks, page_count).
    document_id and pdf_id are the SHA-256 hex (same value).
    """
    pdf_id = document_id
    log.info(
        "[INGEST] parse: opening PDF path=%s document_id=%s…",
        pdf_path,
        document_id[:16],
    )
    chapter_titles = extract_chapter_titles(pdf_path)
    skip_pages = detect_skip_pages(pdf_path)

    fitz_doc = None
    has_pymupdf = False
    try:
        import fitz

        fitz_doc = fitz.open(pdf_path)
        has_pymupdf = True
    except ImportError:
        log.warning(
            "[INGEST] parse: PyMuPDF not installed — image captioning disabled"
        )
    log.info(
        "[INGEST] parse: layout scan chapter_titles=%s skip_pages=%s (no LibreTexts footer) pymupdf=%s",
        len(chapter_titles),
        len(skip_pages),
        has_pymupdf,
    )

    all_chunks: list[ChunkMetadata] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)

            for pdf_page_num, page in enumerate(pdf.pages, start=1):
                if pdf_page_num in skip_pages:
                    continue

                footer = extract_footer_metadata(page)
                section_info = parse_section_from_footer(footer["section_page"])
                chapter_num = section_info["chapter"]
                section_str = section_info["section"]
                chapter_title = (
                    chapter_titles.get(chapter_num)
                    if chapter_num is not None
                    else None
                )

                footer_section_page = footer["section_page"]
                footer_citation = footer["citation_url"]
                footer_libre = footer["libretexts_page_id"]

                def base_kwargs() -> dict[str, Any]:
                    return {
                        "document_id": document_id,
                        "pdf_id": pdf_id,
                        "pdf_page_number": pdf_page_num,
                        "chapter": chapter_num,
                        "chapter_title": chapter_title,
                        "section": section_str,
                        "section_page": footer_section_page,
                        "citation_url": footer_citation,
                        "libretexts_page_id": footer_libre,
                    }

                tables = page.extract_tables()
                table_output_idx = 0
                for table in tables:
                    if not is_real_table(table):
                        continue
                    markdown = table_to_markdown(table)
                    if not markdown:
                        continue
                    table_output_idx += 1
                    chunk_id = make_chunk_id(
                        document_id,
                        pdf_page_num,
                        "table",
                        table_output_idx - 1,
                        section_str,
                    )
                    text = f"[TABLE — Page {pdf_page_num}]\n{markdown}"
                    all_chunks.append(
                        ChunkMetadata(
                            chunk_id=chunk_id,
                            content_type="table",
                            text=text,
                            **base_kwargs(),
                        )
                    )

                if has_pymupdf and fitz_doc is not None:
                    fitz_page = fitz_doc[pdf_page_num - 1]
                    image_idx = 0
                    for img_meta in fitz_page.get_images(full=True):
                        if is_decorative_image(img_meta):
                            continue
                        xref = img_meta[0]
                        result = extract_image_bytes(fitz_doc, xref)
                        if result is None:
                            continue
                        image_bytes, image_ext = result
                        try:
                            caption = await caption_image_with_llm(
                                image_bytes=image_bytes,
                                image_ext=image_ext,
                                page_num=pdf_page_num,
                                section=section_str,
                                chapter_title=chapter_title,
                                client=openai_client,
                                model=vision_model,
                            )
                        except Exception as e:
                            log.warning(
                                "Image caption failed page=%s idx=%s: %s",
                                pdf_page_num,
                                image_idx,
                                e,
                            )
                            caption = (
                                f"Image on page {pdf_page_num} (caption unavailable)."
                            )
                        chunk_id = make_chunk_id(
                            document_id,
                            pdf_page_num,
                            "image",
                            image_idx,
                            section_str,
                        )
                        text = f"[IMAGE — Page {pdf_page_num}]: {caption}"
                        all_chunks.append(
                            ChunkMetadata(
                                chunk_id=chunk_id,
                                content_type="image_caption",
                                text=text,
                                **base_kwargs(),
                            )
                        )
                        image_idx += 1

                body_bbox = (0, 0, page.width, page.height * 0.92)
                body_page = page.within_bbox(body_bbox)
                body_text = body_page.extract_text() or ""
                if not body_text.strip():
                    continue

                text_chunks = split_into_chunks(body_text)
                for idx, chunk_text in enumerate(text_chunks):
                    chunk_id = make_chunk_id(
                        document_id, pdf_page_num, "text", idx, section_str
                    )
                    all_chunks.append(
                        ChunkMetadata(
                            chunk_id=chunk_id,
                            content_type="text",
                            text=chunk_text,
                            **base_kwargs(),
                        )
                    )
    finally:
        if fitz_doc is not None:
            fitz_doc.close()

    by_type = Counter(c.content_type for c in all_chunks)
    log.info(
        "[INGEST] parse: page walk finished file=%s pages=%s chunks=%s by_type=%s",
        filename,
        page_count,
        len(all_chunks),
        dict(by_type),
    )
    return all_chunks, page_count
