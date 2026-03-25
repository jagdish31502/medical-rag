"""SQLite: ingested PDF registry (dedup), sessions, and chat messages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from app.config import settings


def _db_path() -> str:
    return settings.db_path


async def init_db() -> None:
    path = Path(_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(_db_path()) as db:
        cur = await db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'"
        )
        if await cur.fetchone():
            await db.executescript(
                """
                DROP TABLE IF EXISTS messages;
                DROP TABLE IF EXISTS sessions;
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS documents;
                """
            )
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS ingested_pdfs (
                pdf_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                page_count INTEGER,
                chunk_count INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES ingested_pdfs(pdf_id),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                cited_chunks TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_document ON sessions(document_id);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            """
        )
        await db.commit()


async def ingested_pdf_exists(pdf_id: str) -> bool:
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT 1 FROM ingested_pdfs WHERE pdf_id = ? LIMIT 1", (pdf_id,)
        )
        row = await cur.fetchone()
        return row is not None


async def record_ingested_pdf(
    pdf_id: str,
    filename: str,
    file_path: str,
    page_count: int | None,
    chunk_count: int | None,
) -> None:
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(
            """
            INSERT INTO ingested_pdfs (
                pdf_id, filename, file_path, page_count, chunk_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (pdf_id, filename, file_path, page_count, chunk_count),
        )
        await db.commit()


async def create_session(session_id: str, document_id: str) -> None:
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(
            """
            INSERT INTO sessions (session_id, document_id)
            VALUES (?, ?)
            """,
            (session_id, document_id),
        )
        await db.commit()


async def touch_session(session_id: str) -> None:
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(
            "UPDATE sessions SET last_active = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,),
        )
        await db.commit()


async def add_message(
    session_id: str,
    role: str,
    content: str,
    cited_chunks: list[str] | None = None,
) -> None:
    cited_json = json.dumps(cited_chunks) if cited_chunks is not None else None
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(
            """
            INSERT INTO messages (session_id, role, content, cited_chunks)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content, cited_json),
        )
        await db.commit()


async def get_messages(session_id: str) -> list[dict[str, Any]]:
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """
            SELECT role, content, cited_chunks, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        )
        rows = await cur.fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        cited = r["cited_chunks"]
        if cited:
            try:
                cited_parsed = json.loads(cited)
            except json.JSONDecodeError:
                cited_parsed = None
        else:
            cited_parsed = None
        out.append(
            {
                "role": r["role"],
                "content": r["content"],
                "cited_chunks": cited_parsed,
            }
        )
    return out


async def list_sessions(document_id: str) -> list[dict[str, Any]]:
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """
            SELECT session_id, created_at, last_active
            FROM sessions
            WHERE document_id = ?
            ORDER BY last_active DESC
            """,
            (document_id,),
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_session_document(session_id: str) -> str | None:
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT document_id FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cur.fetchone()
    return str(row["document_id"]) if row else None


async def session_exists(session_id: str) -> bool:
    async with aiosqlite.connect(_db_path()) as db:
        cur = await db.execute(
            "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1", (session_id,)
        )
        row = await cur.fetchone()
        return row is not None
