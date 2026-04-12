"""AI Advisor — Claude-powered portfolio analysis and recommendations."""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.core.database import get_connection, load_config
from app.core.ai_advisor import get_api_key, get_ai_config, ask_advisor, get_market_analysis, get_trade_advice, get_flip_strategy

st.set_page_config(page_title="AI Advisor", page_icon="\U0001f916", layout="wide")

st.title("\U0001f916 AI Advisor")
st.caption("AI-powered portfolio analysis and recommendations")

# ---------------------------------------------------------------------------
# Provider Check
# ---------------------------------------------------------------------------
ai_config = get_ai_config()
has_key = ai_config["ready"]

if has_key:
    st.success(f"**Connected:** {ai_config['message']}")
else:
    st.warning(
        f"**AI Advisor not configured**\n\n{ai_config['message']}\n\n"
        "**Option 1 — Google Gemini** (recommended):\n"
        "Add your API key to `config.yaml` under `gemini.api_key`\n\n"
        "**Option 2 — Anthropic Claude:**\n"
        "Set `ai_provider: anthropic` and add `anthropic_api_key` to `config.yaml`"
    )

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------
conn = get_connection()


def _display_response(result: dict, state_key: str) -> None:
    """Store a response in session state and display it."""
    st.session_state[state_key] = result
    _render_response(result)


def _render_response(result: dict) -> None:
    """Render a previously-stored advisor response."""
    if result.get("error"):
        st.error(result["error"])
    elif result.get("response"):
        st.markdown(result["response"])
        tokens = result.get("tokens_used", 0)
        model = result.get("model", "unknown")
        st.caption(f"Tokens used: {tokens:,} | Model: {model}")


# ---------------------------------------------------------------------------
# Quick Analysis Buttons
# ---------------------------------------------------------------------------
st.subheader("Quick Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("\U0001f4ca Portfolio Analysis", type="primary", disabled=not has_key, use_container_width=True):
        with st.spinner("AI is analyzing your portfolio..."):
            result = get_market_analysis(conn)
        _display_response(result, "ai_response_portfolio")

with col2:
    if st.button("\U0001f4b0 Flip Strategy", disabled=not has_key, use_container_width=True):
        with st.spinner("AI is analyzing your portfolio..."):
            result = get_flip_strategy(conn)
        _display_response(result, "ai_response_flip")

with col3:
    show_custom = st.button("\u2753 Custom Question", disabled=not has_key, use_container_width=True)
    if show_custom:
        st.session_state["ai_show_custom"] = not st.session_state.get("ai_show_custom", False)

# Show persisted quick-analysis responses
if "ai_response_portfolio" in st.session_state:
    with st.expander("Portfolio Analysis", expanded=True):
        _render_response(st.session_state["ai_response_portfolio"])

if "ai_response_flip" in st.session_state:
    with st.expander("Flip Strategy", expanded=True):
        _render_response(st.session_state["ai_response_flip"])

# ---------------------------------------------------------------------------
# Custom Question Section
# ---------------------------------------------------------------------------
if st.session_state.get("ai_show_custom", False):
    st.divider()
    st.subheader("Custom Question")
    question = st.text_area(
        "Ask anything about your portfolio",
        placeholder="What should I prioritize? Is Ohtani worth the price? Which position should I upgrade next?",
        key="ai_custom_question_input",
    )
    if st.button("Ask AI", disabled=not has_key or not question, key="btn_ask_custom"):
        with st.spinner("AI is analyzing your portfolio..."):
            result = ask_advisor(question, conn)
        _display_response(result, "ai_response_custom")

    if "ai_response_custom" in st.session_state:
        _render_response(st.session_state["ai_response_custom"])

# ---------------------------------------------------------------------------
# Card-Specific Advice
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Card-Specific Advice")

card_name = st.text_input("Card name", placeholder="e.g. MLB 2026 Live Aaron Judge", key="ai_card_name_input")

buy_col, sell_col = st.columns(2)

with buy_col:
    if st.button("\U0001f6d2 Should I Buy?", disabled=not has_key or not card_name, use_container_width=True, key="btn_buy"):
        with st.spinner("AI is analyzing your portfolio..."):
            result = get_trade_advice(card_name, "buy", conn)
        _display_response(result, "ai_response_card_buy")

with sell_col:
    if st.button("\U0001f4b8 Should I Sell?", disabled=not has_key or not card_name, use_container_width=True, key="btn_sell"):
        with st.spinner("AI is analyzing your portfolio..."):
            result = get_trade_advice(card_name, "sell", conn)
        _display_response(result, "ai_response_card_sell")

if "ai_response_card_buy" in st.session_state:
    with st.expander("Buy Advice", expanded=True):
        _render_response(st.session_state["ai_response_card_buy"])

if "ai_response_card_sell" in st.session_state:
    with st.expander("Sell Advice", expanded=True):
        _render_response(st.session_state["ai_response_card_sell"])

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
conn.close()
