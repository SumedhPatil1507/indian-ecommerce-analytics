"""
dashboard/copilot_tab.py
Merchant Insights Copilot  Streamlit UI.
"""
from __future__ import annotations
import os
import time

import pandas as pd
import streamlit as st

_SUGGESTED_QUESTIONS = [
    "Why did AOV drop in the South zone last month?",
    "Which customers are at risk of churning in Tier-1 cities?",
    "What discount should I offer on Electronics to maximise revenue?",
    "Are there any anomalous orders or fraud signals this quarter?",
    "How does customer lifetime value compare across product categories?",
]


def render_copilot_tab(df: pd.DataFrame, filters: dict | None = None) -> None:
    """Render the Merchant Insights Copilot chat tab."""
    # Demo mode banner
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.warning(
            " Running in demo mode  set `OPENAI_API_KEY` in your `.env` or Streamlit secrets "
            "for AI-generated answers.",
            icon="",
        )

    st.markdown(
        """
        <div style='background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);
                    border-radius:12px;padding:18px 24px;margin-bottom:16px;'>
          <span style='font-size:1.5rem'></span>
          <span style='color:#fff;font-size:1.2rem;font-weight:700;margin-left:8px'>
            Merchant Insights Copilot
          </span><br/>
          <span style='color:#c4b5fd;font-size:.9rem'>
            Ask any business question  the Copilot plans which analytics modules to run,
            retrieves relevant context, and synthesises a cited answer.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Suggested questions in sidebar area (within the tab)
    with st.expander(" Suggested Questions", expanded=True):
        cols = st.columns(2)
        for i, q in enumerate(_SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["copilot_prefill"] = q

    st.markdown("---")

    # Chat input
    prefill = st.session_state.pop("copilot_prefill", "")
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_input(
            "Ask your question",
            value=prefill,
            placeholder="e.g. Why did AOV drop in the South zone last month?",
            label_visibility="collapsed",
            key="copilot_input",
        )
    with col_btn:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    # Session history
    if "copilot_history" not in st.session_state:
        st.session_state["copilot_history"] = []

    if ask_clicked and question.strip():
        try:
            from modules.copilot import MerchantCopilot
            # Use a cached copilot instance per session for RAG memory continuity
            if "_copilot_instance" not in st.session_state:
                st.session_state["_copilot_instance"] = MerchantCopilot()
            copilot: MerchantCopilot = st.session_state["_copilot_instance"]

            with st.spinner("Analysing  thinking which modules to run..."):
                result = copilot.answer(question, df, filters)

            st.session_state["copilot_history"].append({
                "question":     question,
                "answer":       result["answer"],
                "sources":      result["sources"],
                "tool_outputs": result["tool_outputs"],
                "chunks_used":  result["chunks_used"],
            })
        except Exception as e:
            st.error(f" Something went wrong: {e}. Please check the logs or try a different question.")

    # Render conversation history
    history = st.session_state.get("copilot_history", [])
    if history:
        st.markdown("### Conversation")
        for i, turn in enumerate(reversed(history)):
            # User bubble
            st.markdown(
                f"<div style='background:#f1f5f9;border-radius:8px;padding:10px 14px;"
                f"margin-bottom:4px;border-left:3px solid #94a3b8'>"
                f"<strong> You:</strong> {turn['question']}</div>",
                unsafe_allow_html=True,
            )
            # Answer bubble
            answer_html = turn["answer"].replace(
                "[price_optimizer]", "**[price_optimizer]**"
            ).replace(
                "[at_risk]", "**[at_risk]**"
            ).replace(
                "[clv]", "**[clv]**"
            ).replace(
                "[anomaly]", "**[anomaly]**"
            ).replace(
                "[insights]", "**[insights]**"
            ).replace(
                "[inventory_alerts]", "**[inventory_alerts]**"
            ).replace(
                "[price_elasticity]", "**[price_elasticity]**"
            ).replace(
                "[cohort]", "**[cohort]**"
            ).replace(
                "[time_series]", "**[time_series]**"
            )
            st.markdown(
                f"<div style='background:#eef2ff;border-radius:8px;padding:12px 16px;"
                f"margin-bottom:8px;border-left:3px solid #4f46e5'>"
                f"<strong> Copilot</strong> "
                f"<span style='color:#64748b;font-size:.8rem'>"
                f"(used: {', '.join(turn['sources'])})</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(turn["answer"])

            # Expandable tool outputs
            if turn["tool_outputs"]:
                with st.expander(f" Tool Outputs ({len(turn['tool_outputs'])} modules)", expanded=False):
                    for tool_name, output in turn["tool_outputs"].items():
                        st.markdown(f"**{tool_name}**")
                        st.markdown(output)
                        st.markdown("---")

            # Expandable RAG context
            if turn["chunks_used"]:
                with st.expander(f" Retrieved Context ({len(turn['chunks_used'])} chunks)", expanded=False):
                    for j, chunk in enumerate(turn["chunks_used"], 1):
                        st.markdown(f"**Chunk {j}:**")
                        st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                        st.markdown("")

            if i < len(history) - 1:
                st.markdown("---")

        # Clear history button
        if st.button(" Clear conversation", key="clear_copilot"):
            st.session_state["copilot_history"] = []
            st.session_state.pop("_copilot_instance", None)
            st.rerun()
    else:
        st.markdown(
            "<div style='text-align:center;color:#94a3b8;padding:40px;'>"
            " Ask a question above to get started."
            "</div>",
            unsafe_allow_html=True,
        )
