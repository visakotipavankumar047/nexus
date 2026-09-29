import streamlit as st

from components.agent_activity import render_activity, render_step
from components.sources import render_sources
from services.api_client import ApiError, get_client


def render_chat_page() -> None:
    st.title("What should I research?")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            render_activity(m.get("activity", []))
            st.markdown(m["content"])
            render_sources(m.get("sources", []))

    prompt = st.chat_input("Ask about your documents, current events or research…")
    if not prompt:
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.session_state.messages.append(_stream_answer(prompt))


def _stream_answer(prompt: str) -> dict:
    payload = {"message": prompt, "conversation_id": st.session_state.conversation_id,
               "model": st.session_state.model, "use_web": st.session_state.use_web,
               "use_private_knowledge": st.session_state.use_private}
    status = st.status("Agent working…", expanded=True)
    answer_box = st.empty()
    text, activity, sources, tools = "", [], [], 0
    try:
        for event, data in get_client().chat_stream(payload):
            match event:
                case "agent_step":
                    item = {"kind": "step", **data}
                    if data["step"] != "start":
                        tools += 1
                        text = ""  # text before a tool call is a preamble, not the answer
                        answer_box.empty()
                    activity.append(item)
                    render_step(status, item)
                case "tool_result":
                    item = {"kind": "result", **data}
                    activity.append(item)
                    render_step(status, item)
                case "source":
                    sources.append(data)
                case "token":
                    text += data["content"]
                    answer_box.markdown(text + "▌")
                case "done":
                    st.session_state.conversation_id = data["conversation_id"]
                    text, sources = data["answer"], data["sources"]
                case "error":
                    raise ApiError(data.get("code", "ERROR"), data.get("message", "Unknown error"),
                                   data.get("request_id"))
        grounded = "grounded in sources" if sources else "no sources used"
        status.update(label=f"Done · {tools} tool call{'s' if tools != 1 else ''} · {grounded}",
                      state="complete", expanded=False)
    except ApiError as e:
        status.update(label="Failed", state="error", expanded=False)
        text = text or f"⚠️ {e}"
        st.error(str(e))
    answer_box.markdown(text)
    render_sources(sources)
    return {"role": "assistant", "content": text, "activity": activity, "sources": sources}
