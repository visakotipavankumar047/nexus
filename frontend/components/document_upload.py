import streamlit as st

from services.api_client import ApiError, get_client
from utils.formatting import score, short


def render_documents_page() -> None:
    st.title("Documents")
    client = get_client()

    left, right = st.columns(2)
    with left, st.form("upload", clear_on_submit=True):
        st.markdown("#### Upload files")
        files = st.file_uploader("PDF, TXT, DOCX or Markdown (max 20 MB)", type=["pdf", "txt", "docx", "md", "markdown"],
                                 accept_multiple_files=True)
        if st.form_submit_button("Upload & index", type="primary") and files:
            for f in files:
                with st.spinner(f"Indexing {f.name}…"):
                    try:
                        doc = client.upload_file(f.name, f.getvalue(), f.type)
                        st.success(f"{f.name}: {doc['status']} ({doc['metadata'].get('chunks', '?')} chunks)")
                    except ApiError as e:
                        st.error(f"{f.name}: {e}")
    with right, st.form("url", clear_on_submit=True):
        st.markdown("#### Add a web page")
        url = st.text_input("URL", placeholder="https://…")
        if st.form_submit_button("Fetch & index") and url:
            with st.spinner("Fetching…"):
                try:
                    doc = client.upload_url(url)
                    st.success(f"Indexed ({doc['metadata'].get('chunks', '?')} chunks)")
                except ApiError as e:
                    st.error(str(e))

    st.markdown("#### Your documents")
    try:
        docs = client.documents()
    except ApiError as e:
        st.error(str(e))
        return
    if not docs:
        st.info("No documents yet. Upload one above.")
    for d in docs:
        c1, c2, c3, c4 = st.columns([5, 2, 2, 1])
        c1.markdown(f"**{d['filename']}**  \n`{d['id']}`")
        c2.markdown(f"{d['file_type'].upper()} · {d['metadata'].get('chunks', 0)} chunks")
        c3.markdown({"indexed": "🟢", "processing": "🟡", "failed": "🔴"}.get(d["status"], "⚪") + f" {d['status']}")
        with c4.popover("🗑", help="Delete document and its vectors"):
            st.write(f"Delete **{d['filename']}**?")
            if st.button("Delete", key=f"del-{d['id']}", type="primary"):
                try:
                    client.delete_document(d["id"])
                    st.rerun()
                except ApiError as e:
                    st.error(str(e))

    st.markdown("#### Test retrieval")
    with st.form("search"):
        q = st.text_input("Query", placeholder="What does my document say about…")
        c1, c2 = st.columns(2)
        k = c1.slider("Top K", 1, 20, 5)
        rewrite = c2.toggle("LLM query rewrite", value=True)
        if st.form_submit_button("Search") and q:
            try:
                results = client.vector_search(q, k, rewrite)
            except ApiError as e:
                st.error(str(e))
                return
            if not results:
                st.info("No matches.")
            for r in results:
                md = r["metadata"]
                page = f" · page {md['page']}" if md.get("page") else ""
                st.markdown(f"**{md.get('filename', '?')}**{page} · score {score(r['score'])}")
                st.caption(short(r["content"], 500))
