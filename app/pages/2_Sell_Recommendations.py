"""Sell Recommendations page — categorized with selection and PP tracking."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection
from app.core.price_analysis import find_price_anomalies
from app.core.meta_scoring import explain_meta
from app.utils.sparklines import text_sparkline
from app.utils.sidebar_nav import render_sidebar_nav


def _fetch_card_for_explainer(conn, card_id):
    """Pull the rating columns ``explain_meta`` needs for a given card_id."""
    if not card_id:
        return None
    row = conn.execute(
        """
        SELECT card_id, card_title, position, position_name, pitcher_role, pitcher_role_name,
               tier_name,
               contact, gap_power, power, eye, avoid_ks, babip,
               speed, stealing, baserunning,
               of_range, of_error, of_arm,
               catcher_ability, catcher_frame, catcher_arm,
               infield_range, infield_error, infield_arm,
               stuff, movement, control, p_hr, p_babip, stamina, hold
        FROM cards WHERE card_id = ?
        """,
        (card_id,),
    ).fetchone()
    return dict(row) if row else None


def _render_meta_explainer(exp: dict):
    """Same renderer as Buy Recs — duplicated here so each page is self-contained."""
    if not exp:
        st.caption("(no explainer available — card data is missing the needed ratings)")
        return
    ovr = (exp.get('diagnostics') or {}).get('ovr')
    if ovr:
        st.caption(
            f"Total breakdown: **{exp['total']:.0f}** meta · OOTP OVR **{ovr}** "
            "(comparison only — not a formula input). "
            "Components sorted by impact."
        )
    else:
        st.caption(f"Total breakdown: **{exp['total']:.0f}** meta — components sorted by impact.")
    if exp.get('components'):
        comp_df = pd.DataFrame([
            {"Rating": c['label'], "Raw": c['raw'],
             "Weight": c['weight'], "Points": c['points']}
            for c in exp['components']
        ])
        st.dataframe(
            comp_df, use_container_width=True, hide_index=True,
            column_config={
                "Raw": st.column_config.NumberColumn(format="%.0f", width="small"),
                "Weight": st.column_config.NumberColumn(format="%.2f", width="small"),
                "Points": st.column_config.NumberColumn(format="%+.1f", width="small"),
            },
        )
    if exp.get('bonuses'):
        st.markdown("**Bonuses**")
        for b in exp['bonuses']:
            st.markdown(f"\u2022 {b['label']} \u2192 **{b['points']:+.1f}**")
    if exp.get('penalties'):
        st.markdown("**Penalties**")
        for p in exp['penalties']:
            st.markdown(f"\u2022 {p['label']} \u2192 **{p['points']:+.1f}**")
    if exp.get('notes'):
        st.markdown("---")
        for n in exp['notes']:
            st.caption(f"\U0001f4dd {n}")

st.set_page_config(page_title="Sell Recommendations", page_icon="💰", layout="wide")
render_sidebar_nav()
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
        st.dataframe(
            pd.DataFrame(off_data), use_container_width=True, hide_index=True,
            column_config={
                "Meta": st.column_config.NumberColumn(
                    format="%d", width="small",
                    help="Overall meta — same number used everywhere in the app. "
                         "vs-LHP / vs-RHP splits are on Card Detail."),
                "Est. Price": st.column_config.NumberColumn(format="%d PP", width="small"),
                "Buy High": st.column_config.NumberColumn(format="%d PP", width="small"),
            },
        )

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
        _meta_help = ("Overall meta — same number used everywhere in the app. "
                      "vs-LHP / vs-RHP splits are on Card Detail.")
        st.dataframe(
            pd.DataFrame(out_data), use_container_width=True, hide_index=True,
            column_config={
                "Card Meta": st.column_config.NumberColumn(
                    format="%d", width="small", help=_meta_help),
                "Starter Meta": st.column_config.NumberColumn(
                    format="%d", width="small", help=_meta_help),
                "Gap": st.column_config.NumberColumn(
                    format="%d", width="small",
                    help="How many meta points the active starter at this position is ahead "
                         "of the card you might sell."),
                "Est. Price": st.column_config.NumberColumn(format="%d PP", width="small"),
            },
        )

    # ── Price Sanity Check ──
    # Catches cards whose Last 10 price is well above peer median for the same
    # tier + meta band. Could be a legit hot real-world player premium, could
    # be a data ingest bug, but either way it inflates Est. Price (and therefore
    # the projected sell budget feeding Buy Recs). Surfaced quietly so the user
    # can spot-check before trusting the number.
    try:
        _anomalies = find_price_anomalies(conn, owned_only=True, multiplier=2.5)
    except Exception:
        _anomalies = []
    if _anomalies:
        with st.expander(
            f"⚠️ Price Sanity Check — {len(_anomalies)} owned card"
            f"{'s' if len(_anomalies) != 1 else ''} priced above peer median",
            expanded=False,
        ):
            st.caption(
                "These cards' Last 10 prices are at least 2.5x the median for "
                "other cards in the same tier and meta band. **Often legitimate** "
                "(hot real-world player, low supply, scarcity premium) but worth "
                "spot-checking against the actual Card Shop before relying on "
                "Est. Price for sell decisions."
            )
            anom_data = []
            for a in _anomalies[:25]:  # cap display so the page doesn't bloat
                anom_data.append({
                    "Card": a['card_title'],
                    "Tier": a['tier_name'],
                    "Meta": a['meta_score'],
                    "Last 10": a['last_10_price'],
                    "Peer Median": a['peer_median'],
                    "Ratio": f"{a['ratio']}x",
                    "Sell Low": a['sell_order_low'] or 0,
                    "Buy High": a['buy_order_high'] or 0,
                })
            st.dataframe(
                pd.DataFrame(anom_data),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Card": st.column_config.TextColumn(width="large"),
                    "Tier": st.column_config.TextColumn(width="small"),
                    "Meta": st.column_config.NumberColumn(
                        format="%d", width="small",
                        help="Overall meta — same number used everywhere in the app. "
                             "vs-LHP / vs-RHP splits are on Card Detail."),
                    "Last 10": st.column_config.NumberColumn(
                        format="%d PP", width="small",
                        help="Average of the last 10 sales — what the app uses for Est. Price"),
                    "Peer Median": st.column_config.NumberColumn(
                        format="%d PP", width="small",
                        help="Median Last 10 of cards in the same tier + 100-meta band"),
                    "Ratio": st.column_config.TextColumn(
                        width="small",
                        help="Last 10 / Peer Median. >2.5x = flagged."),
                    "Sell Low": st.column_config.NumberColumn(format="%d PP", width="small"),
                    "Buy High": st.column_config.NumberColumn(format="%d PP", width="small"),
                },
            )

    # ── "Why this meta?" explainer (Tier-2 #10) ──
    # Sell tables don't support per-row expanders, so this single picker lets
    # the user pick any sell candidate and see exactly which ratings drove its
    # meta. Helps answer "is this card *actually* worse than the starter, or
    # is the meta formula penalizing something I don't care about?"
    st.divider()
    with st.expander("\U0001f50d Why is one of these rated this way?", expanded=False):
        _explainer_options = {}
        for _r in rows:
            if _r['card_id'] and _r['card_title']:
                _label = f"{_r['card_title']} ({_r['position'] or '?'}, meta {round(_r['meta_score'] or 0)})"
                _explainer_options[_label] = _r['card_id']
        if _explainer_options:
            _picked = st.selectbox(
                "Pick a sell candidate to see its meta breakdown",
                list(_explainer_options.keys()),
                index=0,
                key="sell_explainer_pick",
            )
            _picked_id = _explainer_options.get(_picked)
            if _picked_id:
                try:
                    _full = _fetch_card_for_explainer(conn, _picked_id)
                    _is_pit = bool(_full and (_full.get('pitcher_role') or _full.get('pitcher_role_name')))
                    _exp = explain_meta(_full or {}, is_pitcher=_is_pit) if _full else None
                except Exception as _e:
                    _exp = None
                    st.caption(f"(explainer unavailable: {_e})")
                _render_meta_explainer(_exp)
        else:
            st.caption("No cards with explainable meta data in the current sell list.")

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
