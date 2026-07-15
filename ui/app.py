"""Streamlit chat client for the Dubz AI Bot.

STRICT BOUNDARY: this module talks to the backend exclusively over HTTP via
httpx. It imports NOTHING from ``app`` — no shared agent, data, or memory
code. The only contract is the FastAPI JSON API.

Run the backend first, then:
    uv run streamlit run ui/app.py
"""

from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 60.0

st.set_page_config(page_title="Dubz AI Bot", page_icon="🚗")


# --------------------------------------------------------------------------
# Backend HTTP helpers
# --------------------------------------------------------------------------

def backend_healthy() -> bool:
    """Return True if the backend health check responds ok."""
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=5.0)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except httpx.HTTPError:
        return False


def fetch_profile(user_id: str) -> dict | None:
    """Fetch a user's profile, or None if they have none / backend is down."""
    try:
        r = httpx.get(f"{BACKEND_URL}/users/{user_id}/profile", timeout=10.0)
        if r.status_code == 200:
            return r.json()
    except httpx.HTTPError:
        pass
    return None


def post_chat(user_id: str, session_id: str, message: str) -> dict:
    """POST a chat turn; raises httpx.HTTPError / returns parsed JSON."""
    r = httpx.post(
        f"{BACKEND_URL}/chat",
        json={"user_id": user_id, "session_id": session_id, "message": message},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------
# Session state (UI concern only — conversational memory lives in the backend)
# --------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, tool_calls}]


def render_tool_calls(tool_calls: list[dict]) -> None:
    """Render a collapsed transparency panel of the tool calls for a turn."""
    if not tool_calls:
        return
    with st.expander(f"🔧 behind the scenes ({len(tool_calls)} tool call(s))", expanded=False):
        for tc in tool_calls:
            st.markdown(f"**{tc.get('name')}**")
            st.json(tc.get("args", {}))


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Dubz AI Bot")

    user_id = st.text_input("User ID", value=st.session_state.get("user_id", "guest"))
    st.session_state.user_id = user_id

    if st.button("Start new session", use_container_width=True):
        # Same user_id, brand-new session_id — this is how we demo long-term
        # memory: the backend recalls the profile even though history is fresh.
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    st.divider()
    if backend_healthy():
        st.success("Backend: online")
    else:
        st.error("Backend: offline")
        st.caption(f"Expected at {BACKEND_URL}")

    st.divider()
    st.subheader("Your profile")
    profile = fetch_profile(user_id)
    if profile:
        st.write(f"**Name:** {profile.get('name') or '—'}")
        prefs = profile.get("preferences") or {}
        if prefs:
            st.json(prefs)
        else:
            st.caption("No saved preferences yet.")
    else:
        st.caption("No profile yet — it builds as you chat.")


# --------------------------------------------------------------------------
# Main chat surface
# --------------------------------------------------------------------------

st.title("🚗 Dubz AI Bot")
st.caption("Your dubizzle UAE used-car shopping assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_tool_calls(msg.get("tool_calls", []))

if prompt := st.chat_input("Ask about our cars…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                data = post_chat(user_id, st.session_state.session_id, prompt)
            reply = data.get("reply", "")
            tool_calls = data.get("tool_calls", [])
            st.markdown(reply)
            render_tool_calls(tool_calls)
            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "tool_calls": tool_calls}
            )
        except httpx.HTTPError as e:
            st.error(
                "Couldn't reach the assistant. Is the backend running at "
                f"{BACKEND_URL}?"
            )
            st.caption(str(e))

    # Rerun so the sidebar profile refreshes with any preferences just saved.
    st.rerun()
