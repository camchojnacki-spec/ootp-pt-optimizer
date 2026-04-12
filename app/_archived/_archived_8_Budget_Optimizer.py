"""Budget Optimizer page."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection, load_config
from app.core.optimizer import optimize_budget, simulate_transactions, get_roster_meta_total, ALL_POSITIONS

st.set_page_config(page_title="Budget Optimizer", page_icon="\U0001f9ee", layout="wide")
st.title("\U0001f9ee Budget Optimizer")

conn = get_connection()
config = load_config()
default_budget = config.get('pp_budget', 786)

# --- Controls row ---
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    pp_budget = st.slider("PP Budget", min_value=0, max_value=50000, value=default_budget, step=50)
with col2:
    opt_method = st.radio("Optimization Method",
                          options=["Smart (DP)", "Fast (Greedy)"],
                          index=0,
                          horizontal=True,
                          help="Smart uses dynamic programming for globally optimal results. Fast uses a greedy heuristic.")
    include_sell_revenue = st.checkbox("Include sell revenue",
                                       help="Add estimated sell revenue from sell recommendations to the budget")
with col3:
    st.write("")  # spacer for alignment
    optimize_clicked = st.button("Optimize!", type="primary")

# --- Position filters ---
filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    priority_positions = st.multiselect(
        "Position Priority",
        options=ALL_POSITIONS,
        default=[],
        help="Positions to prioritize filling first in the optimization.")
with filter_col2:
    exclude_positions = st.multiselect(
        "Exclude Positions",
        options=ALL_POSITIONS,
        default=[],
        help="Positions to skip -- the optimizer will not suggest upgrades here.")

# If include sell revenue, calculate it
if include_sell_revenue:
    sell_recs = conn.execute("""
        SELECT COALESCE(SUM(estimated_price), 0) as total
        FROM recommendations
        WHERE rec_type = 'sell' AND dismissed = 0
    """).fetchone()
    sell_revenue = int(sell_recs['total'])
    if sell_revenue > 0:
        st.info(f"Adding **{sell_revenue:,} PP** from sell recommendations to budget (total: **{pp_budget + sell_revenue:,} PP**)")
    effective_budget = pp_budget + sell_revenue
else:
    effective_budget = pp_budget

# --- Optimization results ---
if optimize_clicked:
    method_key = 'dp' if opt_method == "Smart (DP)" else 'greedy'
    with st.spinner("Optimizing..."):
        result = optimize_budget(
            effective_budget, conn,
            method=method_key,
            priority_positions=priority_positions or None,
            exclude_positions=exclude_positions or None,
        )

    # Summary metrics
    method_label = result.get('method', method_key)
    st.subheader("Optimization Results")
    st.caption(f"Method: **{method_label.upper()}**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Meta Gain", f"+{result['total_meta_gain']:.1f}")
    m2.metric("Total Cost", f"{result['total_cost']:,} PP")
    m3.metric("Remaining PP", f"{result['remaining_budget']:,} PP")
    m4.metric("# Upgrades", len(result['transactions']))

    # Before/after comparison
    roster_before = get_roster_meta_total(conn)
    roster_after = roster_before + result['total_meta_gain']

    ba1, ba2, ba3 = st.columns(3)
    ba1.metric("Roster Meta (Before)", f"{roster_before:.0f}")
    ba2.metric("Roster Meta (After)", f"{roster_after:.0f}")
    ba3.metric("Net Improvement", f"+{result['total_meta_gain']:.1f}",
               delta=f"{result['total_meta_gain']:.1f}")

    st.divider()

    # Transaction table
    if result['transactions']:
        st.subheader("Recommended Upgrades")
        tx_data = []
        for t in result['transactions']:
            tx_data.append({
                "Position": t['position'],
                "Current Player (meta)": f"{t['current_player']} ({t['current_meta']:.0f})",
                "\u2192 New Card (meta)": f"{t['card_title']} ({t['new_meta']:.0f})",
                "Meta Gain": f"+{t['meta_gain']:.1f}",
                "Cost (PP)": f"{t['price']:,}",
                "Efficiency": f"{t['efficiency']:.4f}",
            })
        st.dataframe(pd.DataFrame(tx_data), use_container_width=True, hide_index=True)
    else:
        st.info("No upgrades found within this budget. Try increasing the budget or importing more market data.")

# --- What-If Sandbox ---
st.divider()
st.subheader("What-If Sandbox")
st.caption("Simulate specific buy/sell transactions to see their combined impact.")

# Populate card lists for multiselect
# Buy candidates: unowned cards with prices
buy_options_raw = conn.execute("""
    SELECT card_id, card_title,
           COALESCE(pitcher_role_name, position_name) as pos,
           COALESCE(last_10_price, sell_order_low, 0) as price
    FROM cards
    WHERE owned = 0
      AND COALESCE(last_10_price, sell_order_low, 0) > 0
      AND COALESCE(meta_score_batting, meta_score_pitching, 0) > 0
    ORDER BY COALESCE(meta_score_batting, meta_score_pitching) DESC
    LIMIT 500
""").fetchall()

buy_options = {}
for r in buy_options_raw:
    label = f"{r['card_title']} - {r['pos']} - {r['price']:,} PP"
    buy_options[label] = r['card_id']

# Sell candidates: owned cards
sell_options_raw = conn.execute("""
    SELECT card_id, card_title,
           COALESCE(pitcher_role_name, position_name) as pos,
           COALESCE(buy_order_high, last_10_price, 0) as price
    FROM cards
    WHERE owned = 1
    ORDER BY COALESCE(meta_score_batting, meta_score_pitching) DESC
""").fetchall()

sell_options = {}
for r in sell_options_raw:
    label = f"{r['card_title']} - {r['pos']} - {r['price']:,} PP"
    sell_options[label] = r['card_id']

sandbox_col1, sandbox_col2 = st.columns(2)
with sandbox_col1:
    selected_buys = st.multiselect("Cards to Buy", options=list(buy_options.keys()),
                                    placeholder="Search for cards to buy...")
with sandbox_col2:
    selected_sells = st.multiselect("Cards to Sell", options=list(sell_options.keys()),
                                     placeholder="Search for owned cards to sell...")

simulate_clicked = st.button("Simulate", type="secondary")

if simulate_clicked:
    buy_ids = [buy_options[label] for label in selected_buys]
    sell_ids = [sell_options[label] for label in selected_sells]

    if not buy_ids and not sell_ids:
        st.warning("Select at least one card to buy or sell.")
    else:
        with st.spinner("Simulating..."):
            sim = simulate_transactions(buy_ids, sell_ids, conn)

        # Summary metrics
        s1, s2, s3 = st.columns(3)
        pp_label = f"+{sim['net_pp_change']:,}" if sim['net_pp_change'] >= 0 else f"{sim['net_pp_change']:,}"
        s1.metric("Net PP Change", f"{pp_label} PP")
        s2.metric("Roster Meta Delta", f"{sim['meta_delta']:+.1f}")
        s3.metric("Meta After", f"{sim['total_meta_after']:.0f}",
                  delta=f"{sim['meta_delta']:+.1f}")

        # Buy/sell cost breakdown
        if sim['buy_details']:
            st.caption(f"Buy cost: {sim['buy_cost']:,} PP")
        if sim['sell_details']:
            st.caption(f"Sell revenue: {sim['sell_revenue']:,} PP")

        st.divider()

        # Position-by-position impact table
        st.markdown("**Position-by-Position Impact**")
        impact_data = []
        for pos in ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH', 'SP', 'RP', 'CL']:
            before = sim['roster_before'].get(pos, 0)
            after = sim['roster_after'].get(pos, 0)
            delta = after - before
            if delta != 0:
                impact_data.append({
                    "Position": pos,
                    "Meta Before": f"{before:.0f}",
                    "Meta After": f"{after:.0f}",
                    "Delta": f"{delta:+.1f}",
                })

        if impact_data:
            st.dataframe(pd.DataFrame(impact_data), use_container_width=True, hide_index=True)
        else:
            st.info("No position changes detected. The selected cards may not affect roster starters.")

conn.close()
