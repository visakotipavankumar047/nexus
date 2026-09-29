import streamlit as st

from utils.formatting import score


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    private = [s for s in sources if s["type"] == "private"]
    web = [s for s in sources if s["type"] == "web"]
    with st.expander(f"Sources ({len(sources)})"):
        if private:
            st.caption("Private documents")
            for s in private:
                page = f" · page {s['page']}" if s.get("page") else ""
                st.markdown(f"`{s['id']}` **{s['title']}**{page} · score {score(s.get('score'))}")
        if web:
            st.caption("Live sources")
            for s in web:
                date = f" · {s['published_at']}" if s.get("published_at") else ""
                st.markdown(f"`{s['id']}` [{s['title']}]({s['url']}){date}")
