"""
Streamlit UI for RAG: upload PDF, chat with citations, session history.
Requires FastAPI backend: uvicorn app.main:app --reload
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")


def _api_url(path: str) -> str:
    return f"{API_BASE}{path}"


def init_session_state() -> None:
    if "pdf_id" not in st.session_state:
        st.session_state.pdf_id = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingest_info" not in st.session_state:
        st.session_state.ingest_info = None


def fetch_sessions(pdf_id: str) -> list[dict]:
    try:
        r = httpx.get(
            _api_url("/api/sessions"),
            params={"pdf_id": pdf_id},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json().get("sessions", [])
    except Exception as e:
        st.error(f"Could not load sessions: {e}")
        return []


def main() -> None:
    st.set_page_config(page_title="RAG — PDF Chat", layout="wide")
    init_session_state()

    st.title("RAG PDF assistant")
    st.caption(f"API: `{API_BASE}`")

    with st.sidebar:
        st.subheader("Document")
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded is not None:
            if st.button("Ingest PDF", type="primary"):
                with st.spinner("Ingesting (may take several minutes for large PDFs)..."):
                    try:
                        files = {
                            "file": (uploaded.name, uploaded.getvalue(), "application/pdf")
                        }
                        r = httpx.post(
                            _api_url("/api/ingest"),
                            files=files,
                            timeout=600.0,
                        )
                        r.raise_for_status()
                        data = r.json()
                    except Exception as e:
                        st.error(f"Ingest failed: {e}")
                    else:
                        st.session_state.pdf_id = data["pdf_id"]
                        st.session_state.session_id = None
                        st.session_state.messages = []
                        st.session_state.ingest_info = data
                        if data.get("deduplicated"):
                            st.info("PDF already indexed (deduplicated).")
                        else:
                            st.success(
                                f"Ingested: {data['chunk_count']} chunks, "
                                f"{data['page_count']} pages."
                            )

        info = st.session_state.ingest_info
        if info:
            st.write("**Last ingest**")
            st.write(f"File: {info.get('filename')}")
            st.write(f"Chunks: {info.get('chunk_count')}")
            st.write(f"Pages: {info.get('page_count')}")

        pdf_id = st.session_state.pdf_id
        if pdf_id:
            st.text_input("pdf_id (SHA-256)", value=pdf_id, disabled=True)
            sessions = fetch_sessions(pdf_id)
            labels = ["— New chat —"] + [
                f"{s['session_id'][:8]}… ({s.get('last_active', '')})"
                for s in sessions
            ]
            choice = st.selectbox("Session", range(len(labels)), format_func=lambda i: labels[i])
            if choice == 0:
                if st.session_state.session_id is not None:
                    if st.button("Start new chat"):
                        st.session_state.session_id = None
                        st.session_state.messages = []
                        st.rerun()
            else:
                sid = sessions[choice - 1]["session_id"]
                if st.button("Load session"):
                    st.session_state.session_id = sid
                    try:
                        r = httpx.get(
                            _api_url(f"/api/sessions/{sid}/messages"),
                            timeout=60.0,
                        )
                        r.raise_for_status()
                        raw = r.json().get("messages", [])
                        st.session_state.messages = [
                            {
                                "role": m["role"],
                                "content": m["content"],
                                "citations": [],
                            }
                            for m in raw
                        ]
                    except Exception as e:
                        st.error(f"Could not load messages: {e}")
                        st.session_state.messages = []
                    st.rerun()
        else:
            st.info("Upload and ingest a PDF to begin.")

    pdf_id = st.session_state.pdf_id
    if not pdf_id:
        st.markdown("Use the sidebar to **upload and ingest** a PDF first.")
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander("Citations"):
                    for c in msg["citations"]:
                        ch = c.get("chapter_title") or "?"
                        sec = c.get("section") or "?"
                        page = c.get("page")
                        url = c.get("url") or ""
                        st.markdown(
                            f"**{c.get('chunk_id', '')}** — "
                            f"{ch} | Section {sec} | Page {page}"
                            + (f" | [{url}]({url})" if url else "")
                        )

    if prompt := st.chat_input("Ask about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        payload = {
            "question": prompt,
            "pdf_id": pdf_id,
            "session_id": st.session_state.session_id,
        }
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    r = httpx.post(
                        _api_url("/api/query"),
                        json=payload,
                        timeout=120.0,
                    )
                    r.raise_for_status()
                    data = r.json()
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    return

            st.session_state.session_id = data["session_id"]
            answer = data.get("answer", "")
            citations = data.get("citations", [])
            st.markdown(answer)
            if citations:
                with st.expander("Citations"):
                    for c in citations:
                        ch = c.get("chapter_title") or "?"
                        sec = c.get("section") or "?"
                        page = c.get("page")
                        url = c.get("url") or ""
                        line = (
                            f"{ch} | Section {sec} | Page {page}"
                            + (f" | {url}" if url else "")
                        )
                        st.markdown(line)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                }
            )


if __name__ == "__main__":
    main()
