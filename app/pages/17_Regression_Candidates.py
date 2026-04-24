"""Regression Candidates — buy-low / sell-high scanner.

Surfaces cards whose observed BABIP is badly out of line with their
quality-of-contact signature (LD% + EV). Classic luck-based regression
targets you can buy cheap (positive regression) or sell high before the
bottom falls out (negative regression).

Data source: game_log_at_bats (pitch-by-pitch from OOTP's HTML exports).
Pooled across every team instance of each card, so these are real
cross-team signals, not single-team noise.
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.core.database import get_connection, load_config
from app.core.superstats import regression_candidates
from app.utils.sidebar_nav import render_sidebar_nav

st.set_page_config(page_title="Regression Candidates", page_icon="📈", layout="wide")
render_sidebar_nav()
st.title("📈 Regression Candidates")
st.caption(
    "Cards whose BABIP is mismatched with their quality-of-contact signature "
    "(LD% + exit velocity). Buy the ones due to rebound, sell the ones "
    "riding lucky BABIP before the correction. Pooled across every team "
    "instance of each card in the league."
)

conn = get_connection()
config = load_config()
active_league = config.get('active_league', 'lb124')

# ── Controls ──
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    min_pa = st.number_input("Min PA", value=150, step=25, min_value=50,
                             help="Filter cards with at least this many cumulative PA in the league.")
with c2:
    show_owned = st.radio("Show", ["All", "Owned only", "Market only"], horizontal=True,
                          help="Focus on your roster (sell candidates) or market (buy candidates).")
with c3:
    direction_filter = st.radio("Direction", ["Both", "📈 Positive (buy-low)", "📉 Negative (sell-high)"],
                                 horizontal=True)

# Compute
with st.spinner(f"Scanning {active_league} for regression candidates..."):
    cands = regression_candidates(league_id=active_league, min_pa=min_pa, conn=conn)

# Apply filters
if show_owned == "Owned only":
    cands = [c for c in cands if c['owned']]
elif show_owned == "Market only":
    cands = [c for c in cands if not c['owned']]
if direction_filter.startswith("📈"):
    cands = [c for c in cands if c['direction'] == 'up']
elif direction_filter.startswith("📉"):
    cands = [c for c in cands if c['direction'] == 'down']

up_count = sum(1 for c in cands if c['direction'] == 'up')
down_count = sum(1 for c in cands if c['direction'] == 'down')

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total flagged", len(cands))
m2.metric("📈 Positive (buy-low)", up_count)
m3.metric("📉 Negative (sell-high)", down_count)
owned_cands = sum(1 for c in cands if c['owned'])
m4.metric("👤 On your roster", owned_cands)

# Build display table
if not cands:
    st.info("No regression candidates for these filters. Try lowering the min-PA "
            "or expanding the owned/direction filter.")
else:
    df = pd.DataFrame([{
        '': '👤' if c['owned'] else '',
        'Dir': '📈 UP' if c['direction'] == 'up' else '📉 DOWN',
        'Card': c['card_title'],
        'Pos': c['position'],
        'PA': c['pa'],
        'OPS+': c['ops_plus'],
        'BABIP': round(c['babip'], 3),
        'vs lg': round(c['babip_vs_league'], 3),
        'LD%': c['ld_pct'],
        'EV': c['ev_avg'],
        'Reason': c['reason'],
    } for c in cands])

    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            '': st.column_config.TextColumn(width=30),
            'Dir': st.column_config.TextColumn(width='small'),
            'Card': st.column_config.TextColumn(width='large'),
            'Pos': st.column_config.TextColumn(width=50),
            'PA': st.column_config.NumberColumn(format='%d', width=60),
            'OPS+': st.column_config.NumberColumn(format='%d', width=60),
            'BABIP': st.column_config.NumberColumn(format='%.3f', width=70),
            'vs lg': st.column_config.NumberColumn(
                format='%+.3f', width=70,
                help="BABIP vs league average. Negative = unlucky so far; positive = lucky.",
            ),
            'LD%': st.column_config.NumberColumn(
                format='%.0f', width=60,
                help="Line-drive rate from game logs. League avg ≈ 28%. Higher = harder contact quality.",
            ),
            'EV': st.column_config.NumberColumn(
                format='%.1f', width=60,
                help="Average exit velocity across all batted balls. Higher = better contact.",
            ),
            'Reason': st.column_config.TextColumn(width='large'),
        },
    )

st.divider()
st.markdown("### How this works")
st.markdown("""
Each candidate card passed all of:

- **≥ 150 PA** of cumulative batting stats in the active league
- **≥ 50 game-log at-bats** so LD% and EV are reliable
- **BABIP ≥ 0.025 off league average** (big enough to be meaningful)
- **Contact quality contradicts BABIP**:
    - **📈 Positive** — BABIP below league BUT LD% ≥ 25% or EV ≥ 83 mph → balls being hit hard but not falling.
    - **📉 Negative** — BABIP above league AND (LD% < 22% OR EV < 79 mph) → weak contact finding grass.

Regression target: BABIP should drift toward `(LD% × 1.0) + 0.120` for an average
hitter, so a 30% LD% hitter with a .250 BABIP will pull up, and an 18% LD% hitter
with a .330 BABIP will fall. Cross-team pooling (same card on multiple teams)
stabilizes the signal so these aren't just single-team streaks.

**Use the output:**

- 📈 Owned candidates — hold; expect rebound. Don't panic-sell.
- 📈 Market candidates — buy-low opportunities. Their price reflects recent bad OPS;
  LD%/EV says they're due up.
- 📉 Owned candidates — consider selling while the price is inflated.
- 📉 Market candidates — avoid; you'd be buying at peak.
""")
