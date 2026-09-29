import streamlit as st

from config import BACKEND_URL
from services.api_client import ApiError, get_client


def render_status() -> None:
    try:
        h = get_client().health()
        st.caption(f"🟢 {h['service']} v{h['version']} · [API docs]({BACKEND_URL}/docs)")
    except ApiError as e:
        st.caption(f"🔴 Backend offline — {e.message}")
