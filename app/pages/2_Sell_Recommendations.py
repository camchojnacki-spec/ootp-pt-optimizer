"""Sell Recommendations page — categorized with selection and PP tracking."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection
from app.utils.sparklines import text_sparkline

st.set_page_config(page_title="Sell Recommendations", page_icon="💰", layout="wide")
st.title("Sell Recommendations")

conn = get_connection()

# Get all sell recommendations with extra context
rows = conn.execute("""
    SELECT r.card_id, r.card_title, r.position, r.reason, r.priority,
           r.estimated_price, r.meta_score, r.roster_impact,
           c.tier_name, c.tier, c.owned, c.last_10_price, c.buy_order_high, c.sell_order_low
    FROM recommendations r
    LEFT JOIN cards c ON r.card_id = c.card_id
    WHERE r.rec_type = 'sell' AND r.dismissed = 0
    ORDER BY r.priority ASC, r.estimated_price DESC
""").fetchall()

if rows:
    # Categorize sells
    duplicates = []
    off_roster = []
    outclassed = []

    # Get active roster names
    active_roster = conn.execute(
        "SELECT player_name, position, meta_score FROM roster_current WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')"
    ).fetchall()
    active_names = {r['player_name'] for r in active_roster}

    # Build roster meta by position for outclass check
    roster_meta_by_pos = {}
    for r in active_roster:
        pos = r['position']
        if pos not in roster_meta_by_pos or r['meta_score'] > roster_meta_by_pos[pos]:
            roster_meta_by_pos[pos] = r['meta_score']

    for r in rows:
        row_dict = dict(r)
        reason = r['reason'] or ''
        owned_count = r['owned'] or 0

        if 'duplicate' in reason.lower() or owned_count > 1:
            row_dict['category'] = 'duplicate'
            duplicates.append(row_dict)
        elif 'not on active' in reason.lower():
            # Check if this card is outclassed at its position
            pos = r['position'] or ''
            card_meta = r['meta_score'] or 0
            starter_meta = roster_meta_by_pos.get(pos, 0)
            if starter_meta > 0 and card_meta < starter_meta:
                row_dict['category'] = 'outclassed'
                row_dict['starter_meta'] = starter_meta
                outclassed.append(row_dict)
            else:
                row_dict['category'] = 'off_roster'
                off_roster.append(row_dict)
        else:
            row_dict['category'] = 'off_roster'
            off_roster.append(row_dict)

    # Summary metrics
    total_pp = sum(r['estimated_price'] or 0 for r in rows)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sell Value", f"{total_pp:,} PP")
    with col2:
        st.metric("Duplicates", f"{len(duplicates)} cards")
    with col3:
        st.metric("Off-Roster", f"{len(off_roster)} cards")
    with col4:
        st.metric("Outclassed", f"{len(outclassed)} cards")

    st.divider()

    # ── Quick Sells: Duplicates ──
    if duplicates:
        st.subheader("🔴 Quick Sell — Duplicates")
        st.caption("You own multiple copies. Sell the extras with zero risk.")
        dup_data = []
        for r in duplicates:
            dup_data.append({
                "Card": r['card_title'] or '',
                "Pos": r['position'] or '',
                "Tier": r['tier_name'] or '',
                "Trend": text_sparkline(r['card_id'], conn) if r.get('card_id') else '',
                "Owned": r['owned'] or 0,
                "Est. Price": r['estimated_price'] or 0,
                "Buy High": r['buy_order_high'] or 0,
                "Sell Low": r['sell_order_low'] or 0,
                "Action": r['roster_impact'] or '',
            })
        dup_pp = sum(d['Est. Price'] for d in dup_data)
        st.markdown(f"**Subtotal: {dup_pp:,} PP** from {len(dup_data)} duplicate sell(s)")
        st.dataframe(pd.DataFrame(dup_data), use_container_width=True, hide_index=True)

    # ── Consider Selling: Off-Roster ──
    if off_roster:
        st.subheader("🟠 Consider Selling — Off-Roster Cards")
        st.caption("Cards you own but aren't using on the active roster. Safe to sell unless you're keeping them as backup.")
        off_data = []
        for r in off_roster:
            off_data.append({
                "Card": r['card_title'] or '',
                "Pos": r['position'] or '',
                "Tier": r['tier_name'] or '',
                "Trend": text_sparkline(r['card_id'], conn) if r.get('card_id') else '',
                "Meta": round(r['meta_score'], 0) if r['meta_score'] else 0,
                "Est. Price": r['estimated_price'] or 0,
                "Buy High": r['buy_order_high'] or 0,
                "Reason": r['reason'] or '',
            })
        off_pp = sum(d['Est. Price'] for d in off_data)
        st.markdown(f"**Subtotal: {off_pp:,} PP** from {len(off_data)} off-roster sell(s)")
        st.dataframe(pd.DataFrame(off_data), use_container_width=True, hide_index=True)

    # ── Roster Downgrades: Outclassed at Position ──
    if outclassed:
        st.subheader("🟡 Roster Downgrades — Outclassed at Position")
        st.caption("You own a better card at this position. These are safe to sell unless you want bench depth.")
        out_data = []
        for r in outclassed:
            out_data.append({
                "Card": r['card_title'] or '',
                "Pos": r['position'] or '',
                "Tier": r['tier_name'] or '',
                "Trend": text_sparkline(r['card_id'], conn) if r.get('card_id') else '',
                "Card Meta": round(r['meta_score'], 0) if r['meta_score'] else 0,
                "Starter Meta": round(r.get('starter_meta', 0), 0),
                "Gap": round((r.get('starter_meta', 0) or 0) - (r['meta_score'] or 0), 0),
                "Est. Price": r['estimated_price'] or 0,
            })
        out_pp = sum(d['Est. Price'] for d in out_data)
        st.markdown(f"**Subtotal: {out_pp:,} PP** from {len(out_data)} outclassed sell(s)")
        st.dataframe(pd.DataFrame(out_data), use_container_width=True, hide_index=True)

    # ── Sell Planner: select cards and see total ──
    st.divider()
    st.subheader("Sell Planner")
    st.caption("Select which sells to execute — see the PP you'd gain to feed into Buy Recommendations.")

    all_sell_cards = [r['card_title'] for r in rows if r['card_title']]
    all_sell_prices = {r['card_title']: (r['estimated_price'] or 0) for r in rows}

    selected_sells = st.multiselect("Select cards to sell", all_sell_cards, default=all_sell_cards[:0])

    if selected_sells:
        selected_total = sum(all_sell_prices.get(c, 0) for c in selected_sells)
        st.metric("PP from Selected Sells", f"{selected_total:,} PP")
        st.info(f"💡 Add this to your PP budget ({selected_total:,}) in the Buy Recommendations page to plan your buys.")
    else:
        st.info("Select cards above to see projected PP from sales.")

else:
    st.info("No sell recommendations yet. Import your collection and market data first.")

conn.close()
