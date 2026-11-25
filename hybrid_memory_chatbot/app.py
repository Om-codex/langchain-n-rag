"""Streamlit UI for Phase 1: PostgreSQL-backed short-term memory."""

from pathlib import Path
import sys

# Make the package directory importable when launched from the repo root.
_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

import streamlit as st
from sqlalchemy import text

from chains.chatbot import ask, get_session_history
from db.connection import SessionFactory


st.set_page_config(page_title="Hybrid Memory - Phase 1", page_icon="💬")
st.title("Phase 1 — Short-term memory (PostgreSQL)")

st.sidebar.header("Conversation")
user_id = st.sidebar.text_input("User ID", value="123").strip() or "123"
session_id = f"user:{user_id}"

st.sidebar.caption(f"Session: `{session_id}`")


def load_rows(sid: str) -> list[dict]:
    """Read raw message rows so the UI reflects the database on every rerun."""
    with SessionFactory() as db:
        result = db.execute(
            text(
                "SELECT id, role, content, created_at "
                "FROM chat_messages WHERE session_id = :sid ORDER BY id ASC"
            ),
            {"sid": sid},
        ).mappings().all()

    rows: list[dict] = []
    for row in result:
        item = dict(row)
        item["created_at"] = str(item["created_at"])
        rows.append(item)

    return rows


# Reset the local view when the user switches.
if st.session_state.get("active_user_id") != user_id:
    st.session_state.active_user_id = user_id
    st.rerun()

rows = load_rows(session_id)

for row in rows:
    role = "user" if row["role"] == "human" else "assistant"
    with st.chat_message(role):
        st.write(row["content"])

st.sidebar.write(f"Messages in DB: {len(rows)}")

if st.sidebar.button("Clear short-term history"):
    get_session_history(session_id).clear()
    st.rerun()

with st.sidebar.expander("Database rows"):
    if rows:
        st.json(rows)
    else:
        st.info("No messages stored yet.")

question = st.chat_input("Ask something…")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Thinking…"):
        try:
            answer = ask(question, user_id)
        except Exception as exc:
            st.error(f"Chat failed: {type(exc).__name__}: {exc}")
            st.stop()

    with st.chat_message("assistant"):
        st.write(answer)

    st.rerun()
