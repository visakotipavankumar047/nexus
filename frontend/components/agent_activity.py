import streamlit as st

from utils.formatting import short, step_label


def render_step(container, item: dict) -> None:
    """One activity line inside an st.status container (live) or expander (history)."""
    if item["kind"] == "step":
        args = item.get("input")
        container.markdown(f"**{step_label(item['step'])}**" + (f" — `{short(str(args), 120)}`" if args else ""))
    else:
        icon = "⚠️" if item.get("status") == "error" else "↳"
        container.caption(f"{icon} {short(item.get('content', ''), 280)}")


def render_activity(activity: list[dict]) -> None:
    tools = [a for a in activity if a["kind"] == "step" and a["step"] != "start"]
    if not tools:
        return
    with st.expander(f"Agent activity ({len(tools)} tool call{'s' if len(tools) != 1 else ''})"):
        for item in activity:
            render_step(st, item)
