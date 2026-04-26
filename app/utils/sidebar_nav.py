"""Shared grouped sidebar navigation.

Streamlit's auto-discovered page list is hidden in ``.streamlit/config.toml``
(``client.showSidebarNavigation = false``) so this module is the only nav
surface the user sees. Every page should call ``render_sidebar_nav()`` once,
near the top after ``st.set_page_config``.

The grouping (Decide / Analyze / Library / Configure) addresses Tier-3 #12
from the product review: the alphabetical dump of 14 numbered pages was
hiding the workflow sequence ("decide what to act on → analyze data →
look things up → configure").
"""
from __future__ import annotations

import streamlit as st


# Each tuple: (relative file path from entry script, label, icon).
# Paths are relative to ``app/main.py`` (the streamlit entry point).
_HOME = ("main.py", "Dashboard", "\U0001f3e0")  # 🏠

# "Start Here" — the data refresh flow is the entry point to everything
# else. Every session begins by running it, so it gets its own prominent
# slot directly under Dashboard instead of being buried in Configure.
_START = [
    ("pages/0_Data_Refresh.py",     "Data Refresh",   "\U0001f4e5"),  # 📥
    ("pages/20_Game_Coverage.py",   "Game Coverage",  "\U0001f5d3️"),  # 🗓️
]

_DECIDE = [
    ("pages/1_Buy_Recommendations.py", "Buy Recs",          "\U0001f6d2"),  # 🛒
    ("pages/2_Sell_Recommendations.py", "Sell Recs",        "\U0001f4b0"),  # 💰
    ("pages/4_Roster_Optimizer.py",     "Roster Optimizer", "\U0001f4cb"),  # 📋
    ("pages/13_Tournament_Builder.py",  "Tournament Builder","\U0001f3c6"),  # 🏆
    ("pages/15_Export_Plan.py",         "Export Plan",      "\U0001f4e6"),  # 📦
    ("pages/9_Flip_Finder.py",          "Flip Finder",      "\U0001f504"),  # 🔄
]

_ANALYZE = [
    ("pages/7_Game_Stats.py",      "Game Stats",      "\U0001f4ca"),  # 📊
    ("pages/3_Price_Trends.py",    "Price Trends",    "\U0001f4c8"),  # 📈
    ("pages/6_Trends.py",          "Trends & History","\U0001f4c9"),  # 📉
    ("pages/10_Mission_Tracker.py","Mission Tracker", "\U0001f3af"),  # 🎯
    ("pages/19_Archetypes.py",     "Archetypes",      "\U0001f9ec"),  # 🧬
]

_LIBRARY = [
    ("pages/14_Card_Detail.py",       "Card Detail",       "\U0001f4c4"),  # 📄
    ("pages/6_Live_Card_Tracker.py",  "Live Card Tracker", "\U0001f4e1"),  # 📡
]

_CONFIGURE = [
    ("pages/12_AI_Advisor.py", "AI Advisor", "\U0001f916"),  # 🤖
    ("pages/5_Settings.py",    "Settings",   "\u2699\ufe0f"),  # ⚙️
]


def render_sidebar_nav() -> None:
    """Draw the grouped nav inside ``st.sidebar``.

    Idempotent on the page level — call exactly once per page (calling twice
    would render two copies). Pages with their own sidebar content can call
    this first, then add their content with a separate ``with st.sidebar:``
    block; Streamlit appends sidebar content in declaration order.
    """
    with st.sidebar:
        # Home
        path, label, icon = _HOME
        st.page_link(path, label=label, icon=icon)

        st.markdown("**Start Here**")
        for path, label, icon in _START:
            st.page_link(path, label=label, icon=icon)

        st.markdown("**Decide**")
        for path, label, icon in _DECIDE:
            st.page_link(path, label=label, icon=icon)

        st.markdown("**Analyze**")
        for path, label, icon in _ANALYZE:
            st.page_link(path, label=label, icon=icon)

        st.markdown("**Library**")
        for path, label, icon in _LIBRARY:
            st.page_link(path, label=label, icon=icon)

        st.markdown("**Configure**")
        for path, label, icon in _CONFIGURE:
            st.page_link(path, label=label, icon=icon)

        st.divider()
