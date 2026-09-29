import streamlit as st

from components.status import render_status
from config import MODELS
from services.api_client import ApiError, get_client


def new_chat() -> None:
    st.session_state.conversation_id = None
    st.session_state.messages = []


def open_conversation(conversation_id: str) -> None:
    try:
        msgs = get_client().messages(conversation_id)
    except ApiError as e:
        st.error(str(e))
        return
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in msgs]


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## NEXUS")
        st.caption("Agentic RAG Intelligence")
        st.button("＋ New chat", on_click=new_chat, width="stretch")

        st.markdown("#### Settings")
        st.selectbox("Model", list(MODELS), format_func=MODELS.get, key="model")
        st.toggle("Live web (GDELT / OpenAlex / RSS)", key="use_web")
        st.toggle("Private knowledge (your documents)", key="use_private")

        st.markdown("#### Recent conversations")
        try:
            convs = get_client().conversations()[:15]
        except ApiError as e:
            st.caption(f"Unavailable: {e.message}")
            convs = []
        for c in convs:
            active = c["id"] == st.session_state.get("conversation_id")
            st.button(("▸ " if active else "") + c["title"][:40], key=f"conv-{c['id']}", width="stretch",
                      on_click=open_conversation, args=(c["id"],), type="primary" if active else "secondary")
        if not convs:
            st.caption("No conversations yet.")

        st.divider()
        render_status()
