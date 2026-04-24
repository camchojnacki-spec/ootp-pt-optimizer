"""Card Detail page — deep analysis of a single card via query params or search."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection
from app.core.meta_scoring import (
    calc_batting_meta, calc_batting_meta_vs_rhp, calc_batting_meta_vs_lhp,
    calc_pitching_meta, calc_pitching_meta_vs_lhb, calc_pitching_meta_vs_rhb,
    explain_meta,
)
from app.utils.sparklines import get_price_history
from app.utils.sidebar_nav import render_sidebar_nav

st.set_page_config(page_title="Card Detail", page_icon="\U0001f4cb", layout="wide")
render_sidebar_nav()
st.title("Card Detail")

conn = get_connection()

# Get card_id from query params or let user search
card_id = st.query_params.get("card_id")
card = None

if card_id:
    try:
        card = conn.execute("SELECT * FROM cards WHERE card_id = ?", (int(card_id),)).fetchone()
    except (ValueError, TypeError):
        pass

if not card:
    # Search fallback
    search = st.text_input("Search for a card", placeholder="e.g., Babe Ruth, Mike Trout")
    if search:
        matches = conn.execute(
            "SELECT card_id, card_title, tier_name, position_name, pitcher_role_name "
            "FROM cards WHERE card_title LIKE ? ORDER BY COALESCE(meta_score_batting, meta_score_pitching) DESC LIMIT 20",
            (f"%{search}%",)
        ).fetchall()
        if matches:
            options = {f"{m['card_title']} ({m['tier_name']}) - {m['pitcher_role_name'] or m['position_name']}": m['card_id']
                       for m in matches}
            selected_label = st.selectbox("Select card", list(options.keys()))
            if selected_label:
                card_id = options[selected_label]
                card = conn.execute("SELECT * FROM cards WHERE card_id = ?", (card_id,)).fetchone()
        else:
            st.warning(f"No cards found matching '{search}'")

if not card:
    st.info("Enter a card name above or navigate here with ?card_id=123 in the URL.")
    conn.close()
    st.stop()

# ============================================================
# 1. CARD HEADER
# ============================================================
is_pitcher = bool(card['pitcher_role'])
pos_display = card['pitcher_role_name'] if is_pitcher else card['position_name']

tier_colors = {
    'Perfect': '#ff4500', 'Diamond': '#b9f2ff', 'Gold': '#ffd700',
    'Silver': '#c0c0c0', 'Bronze': '#cd7f32', 'Common': '#808080',
}
tier_name = card['tier_name'] or 'Common'
tier_color = tier_colors.get(tier_name, '#808080')

st.markdown(
    f"### {card['card_title']} "
    f"<span style='background-color:{tier_color};color:#000;padding:2px 10px;border-radius:4px;font-size:0.8em'>"
    f"{tier_name}</span>",
    unsafe_allow_html=True,
)

header_cols = st.columns(6)
with header_cols[0]:
    st.markdown(f"**Position:** {pos_display}")
with header_cols[1]:
    st.markdown(f"**Age:** {card['age'] or '---'}")
with header_cols[2]:
    st.markdown(f"**Team:** {card['team'] or '---'}")
with header_cols[3]:
    st.markdown(f"**Bats:** {card['bats'] or '---'}")
with header_cols[4]:
    st.markdown(f"**Throws:** {card['throws'] or '---'}")
with header_cols[5]:
    st.markdown(f"**OVR:** {card['card_value'] or '---'}")

st.divider()

# ============================================================
# 2. RATINGS PANEL
# ============================================================
ratings_col, market_col = st.columns(2)

with ratings_col:
    if is_pitcher:
        st.subheader("Pitching Ratings")
        pitch_labels = ["STU", "MOV", "CTRL", "HR", "BABIP"]
        pitch_overall = [card['stuff'], card['movement'], card['control'], card['p_hr'], card['p_babip']]
        pitch_vl = [card['stuff_vl'], card['movement_vl'], card['control_vl'], card['p_hr_vl'], card['p_babip_vl']]
        pitch_vr = [card['stuff_vr'], card['movement_vr'], card['control_vr'], card['p_hr_vr'], card['p_babip_vr']]

        pitch_df = pd.DataFrame({
            "Stat": pitch_labels,
            "Overall": [v or 0 for v in pitch_overall],
            "vs LHB": [v or 0 for v in pitch_vl],
            "vs RHB": [v or 0 for v in pitch_vr],
        })
        st.dataframe(pitch_df, use_container_width=True, hide_index=True)

        # Extra pitching info
        extra_cols = st.columns(3)
        with extra_cols[0]:
            st.metric("Stamina", card['stamina'] or 0)
        with extra_cols[1]:
            st.metric("Hold", card['hold'] or 0)
        with extra_cols[2]:
            st.metric("Velocity", card['velocity'] or "---")

        if card['meta_score_pitching']:
            # Compute split-aware metas for the tooltip. Overall is the canonical
            # number used elsewhere in the app; the splits show how much the
            # number shifts when this pitcher faces an L-heavy or R-heavy lineup.
            try:
                _row = dict(card)
                _meta_overall = card['meta_score_pitching']
                _meta_vlhb = calc_pitching_meta_vs_lhb(_row) or 0
                _meta_vrhb = calc_pitching_meta_vs_rhb(_row) or 0
            except Exception:
                _meta_overall = card['meta_score_pitching']
                _meta_vlhb = _meta_vrhb = 0
            st.metric(
                "Meta Score (Pitching)",
                f"{_meta_overall:.0f}",
                help=(
                    "Overall meta — the canonical score used everywhere in the app "
                    "(Buy Recs, Roster Optimizer, Sell Recs).\n\n"
                    f"• vs LHB split: {_meta_vlhb:.0f}\n"
                    f"• vs RHB split: {_meta_vrhb:.0f}\n\n"
                    "Splits are derived from this pitcher's vsL/vsR ratings. "
                    "Use them when your matchup planning calls for it; "
                    "Overall is the default sort key everywhere else."
                ),
            )
            if _meta_vlhb and _meta_vrhb:
                _split_cols = st.columns(2)
                with _split_cols[0]:
                    _delta_l = _meta_vlhb - _meta_overall
                    st.caption(
                        f"vs LHB: **{_meta_vlhb:.0f}** "
                        f"({'+' if _delta_l >= 0 else ''}{_delta_l:.0f})"
                    )
                with _split_cols[1]:
                    _delta_r = _meta_vrhb - _meta_overall
                    st.caption(
                        f"vs RHB: **{_meta_vrhb:.0f}** "
                        f"({'+' if _delta_r >= 0 else ''}{_delta_r:.0f})"
                    )
    else:
        st.subheader("Batting Ratings")
        bat_labels = ["CON", "GAP", "POW", "EYE", "K's", "BABIP"]
        bat_overall = [card['contact'], card['gap_power'], card['power'], card['eye'], card['avoid_ks'], card['babip']]
        bat_vl = [card['contact_vl'], card['gap_vl'], card['power_vl'], card['eye_vl'], card['avoid_ks_vl'], card['babip_vl']]
        bat_vr = [card['contact_vr'], card['gap_vr'], card['power_vr'], card['eye_vr'], card['avoid_ks_vr'], card['babip_vr']]

        bat_df = pd.DataFrame({
            "Stat": bat_labels,
            "Overall": [v or 0 for v in bat_overall],
            "vs LHP": [v or 0 for v in bat_vl],
            "vs RHP": [v or 0 for v in bat_vr],
        })
        st.dataframe(bat_df, use_container_width=True, hide_index=True)

        if card['meta_score_batting']:
            # Compute split-aware metas for the tooltip. Overall is the canonical
            # number used everywhere in the app; the vs-LHP/vs-RHP variants show
            # how much the number shifts based on this batter's handedness splits.
            try:
                _row = dict(card)
                _meta_overall = card['meta_score_batting']
                _meta_vlhp = calc_batting_meta_vs_lhp(_row) or 0
                _meta_vrhp = calc_batting_meta_vs_rhp(_row) or 0
            except Exception:
                _meta_overall = card['meta_score_batting']
                _meta_vlhp = _meta_vrhp = 0
            st.metric(
                "Meta Score (Batting)",
                f"{_meta_overall:.0f}",
                help=(
                    "Overall meta — the canonical score used everywhere in the app "
                    "(Buy Recs, Roster Optimizer, Sell Recs). All cross-page "
                    "comparisons use this number.\n\n"
                    f"• vs LHP split: {_meta_vlhp:.0f}\n"
                    f"• vs RHP split: {_meta_vrhp:.0f}\n\n"
                    "Splits are derived from this batter's vsL/vsR ratings. Use "
                    "them when planning platoons; Overall is the default sort "
                    "key everywhere else."
                ),
            )
            if _meta_vlhp and _meta_vrhp:
                _split_cols = st.columns(2)
                with _split_cols[0]:
                    _delta_l = _meta_vlhp - _meta_overall
                    st.caption(
                        f"vs LHP: **{_meta_vlhp:.0f}** "
                        f"({'+' if _delta_l >= 0 else ''}{_delta_l:.0f})"
                    )
                with _split_cols[1]:
                    _delta_r = _meta_vrhp - _meta_overall
                    st.caption(
                        f"vs RHP: **{_meta_vrhp:.0f}** "
                        f"({'+' if _delta_r >= 0 else ''}{_delta_r:.0f})"
                    )

    # ── Why this meta? — explainer expander (Tier-2 #10 from the review) ──
    # Cards expose their formula contributions so the user isn't reading the
    # meta number as a black box. Reconciles to within 0.1 of the canonical
    # calc_*_meta functions so the breakdown total matches the headline number.
    try:
        _exp = explain_meta(dict(card), is_pitcher=is_pitcher)
    except Exception:
        _exp = None
    if _exp:
        with st.expander("\U0001f50d Why this meta?", expanded=False):
            _ovr = (_exp.get('diagnostics') or {}).get('ovr')
            if _ovr:
                st.caption(
                    f"Total breakdown: **{_exp['total']:.0f}** meta · "
                    f"OOTP OVR **{_ovr}** (comparison only — not a formula input). "
                    "Contributions sorted by impact."
                )
            else:
                st.caption(
                    f"Total breakdown: **{_exp['total']:.0f}** meta — "
                    "contributions sorted by impact."
                )
            if _exp['components']:
                _comp_df = pd.DataFrame([
                    {
                        "Rating": c['label'],
                        "Raw": c['raw'],
                        "Weight": c['weight'],
                        "Points": c['points'],
                    }
                    for c in _exp['components']
                ])
                st.dataframe(
                    _comp_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rating": st.column_config.TextColumn(width="medium"),
                        "Raw": st.column_config.NumberColumn(format="%.0f", width="small",
                            help="The OOTP rating value (0-200)"),
                        "Weight": st.column_config.NumberColumn(format="%.2f", width="small",
                            help="Current calibrated weight from Settings"),
                        "Points": st.column_config.NumberColumn(format="%+.0f", width="small",
                            help="Contribution to the meta total (raw × weight, with diminishing returns above 110)"),
                    },
                )
            if _exp['bonuses']:
                st.markdown("**Bonuses**")
                for b in _exp['bonuses']:
                    st.markdown(f"• {b['label']} → **{b['points']:+.0f}**")
            if _exp['penalties']:
                st.markdown("**Penalties**")
                for p in _exp['penalties']:
                    st.markdown(f"• {p['label']} → **{p['points']:+.0f}**")
            if _exp['notes']:
                st.markdown("---")
                for n in _exp['notes']:
                    st.caption(f"\U0001f4dd {n}")

    # Speed / Baserunning
    if not is_pitcher:
        st.markdown("**Speed & Baserunning**")
        spd_cols = st.columns(4)
        with spd_cols[0]:
            st.metric("Speed", card['speed'] or 0)
        with spd_cols[1]:
            st.metric("Stealing", card['stealing'] or 0)
        with spd_cols[2]:
            st.metric("Baserunning", card['baserunning'] or 0)
        with spd_cols[3]:
            st.metric("Steal Rate", card['steal_rate'] or 0)

    # Defense ratings — show all positions with rating > 0
    st.markdown("**Defense Ratings**")
    pos_map = {
        'P': 'pos_rating_p', 'C': 'pos_rating_c', '1B': 'pos_rating_1b',
        '2B': 'pos_rating_2b', '3B': 'pos_rating_3b', 'SS': 'pos_rating_ss',
        'LF': 'pos_rating_lf', 'CF': 'pos_rating_cf', 'RF': 'pos_rating_rf',
    }
    def_ratings = {}
    for pos_label, col_name in pos_map.items():
        val = card[col_name]
        if val and val > 0:
            def_ratings[pos_label] = val

    if def_ratings:
        def_cols = st.columns(min(len(def_ratings), 5))
        for i, (pos_label, rating) in enumerate(def_ratings.items()):
            with def_cols[i % len(def_cols)]:
                st.metric(pos_label, rating)
    else:
        st.caption("No defensive ratings available.")

    # Fielding detail (if batter)
    if not is_pitcher:
        has_if = any(card[f] and card[f] > 0 for f in ['infield_range', 'infield_error', 'infield_arm', 'dp'])
        has_of = any(card[f] and card[f] > 0 for f in ['of_range', 'of_error', 'of_arm'])
        has_c = any(card[f] and card[f] > 0 for f in ['catcher_ability', 'catcher_frame', 'catcher_arm'])

        if has_if:
            st.markdown("**Infield**")
            if_cols = st.columns(4)
            with if_cols[0]:
                st.metric("Range", card['infield_range'] or 0)
            with if_cols[1]:
                st.metric("Error", card['infield_error'] or 0)
            with if_cols[2]:
                st.metric("Arm", card['infield_arm'] or 0)
            with if_cols[3]:
                st.metric("DP", card['dp'] or 0)

        if has_of:
            st.markdown("**Outfield**")
            of_cols = st.columns(3)
            with of_cols[0]:
                st.metric("Range", card['of_range'] or 0)
            with of_cols[1]:
                st.metric("Error", card['of_error'] or 0)
            with of_cols[2]:
                st.metric("Arm", card['of_arm'] or 0)

        if has_c:
            st.markdown("**Catcher**")
            c_cols = st.columns(3)
            with c_cols[0]:
                st.metric("Ability", card['catcher_ability'] or 0)
            with c_cols[1]:
                st.metric("Framing", card['catcher_frame'] or 0)
            with c_cols[2]:
                st.metric("Arm", card['catcher_arm'] or 0)

# ============================================================
# 3. MARKET PANEL
# ============================================================
with market_col:
    st.subheader("Market Data")

    price_cols = st.columns(2)
    with price_cols[0]:
        st.metric(
            "Buy Order High",
            f"{card['buy_order_high']:,} PP" if card['buy_order_high'] else "---",
            help=(
                "**The highest active BID** — what other players are currently "
                "willing to pay for this card. If you list at this price, it "
                "should sell instantly. Lower than Sell Order Low because that's "
                "what makes a market spread."
            ),
        )
        st.metric(
            "Last 10 Price",
            f"{card['last_10_price']:,} PP" if card['last_10_price'] else "---",
            help=(
                "**Average of the last 10 completed sales.** This is the most "
                "honest 'real' price — actual transactions, not just standing "
                "orders. The app uses this as the default Est. Price for sell "
                "recommendations because it's the least gameable."
            ),
        )
    with price_cols[1]:
        st.metric(
            "Sell Order Low",
            f"{card['sell_order_low']:,} PP" if card['sell_order_low'] else "---",
            help=(
                "**The lowest active ASK** — what other players are currently "
                "trying to sell this card for. If you bid at this price, you "
                "should fill instantly. Higher than Buy Order High because "
                "that's the spread sellers want to capture."
            ),
        )
        st.metric(
            "Last 10 Variance",
            f"{card['last_10_variance']:,}" if card['last_10_variance'] else "---",
            help=(
                "**Std. dev. of the last 10 sale prices.** High variance = "
                "volatile market (less predictable Est. Price). Low variance = "
                "stable market. Useful for spotting cards in price discovery."
            ),
        )

    # Spread analysis — disclose what wide vs tight spreads mean for liquidity.
    if card['buy_order_high'] and card['sell_order_low'] and card['sell_order_low'] > 0:
        spread = card['sell_order_low'] - card['buy_order_high']
        spread_pct = (spread / card['sell_order_low']) * 100
        if spread_pct >= 30:
            _liquidity_note = "  ⚠️ Wide spread → low liquidity (orders may take time to fill)"
        elif spread_pct <= 10:
            _liquidity_note = "  ✅ Tight spread → liquid market"
        else:
            _liquidity_note = ""
        st.caption(f"Spread: {spread:,} PP ({spread_pct:.1f}%){_liquidity_note}")
        st.caption(
            "**Spread** = Sell Low − Buy High. A spread under ~10% usually means "
            "a liquid card you can flip quickly; over ~30% means thin demand "
            "where you may have to wait or accept a worse fill."
        )

    # Price history chart
    st.markdown("**Price History**")
    history = get_price_history(card['card_id'], conn, days=30)
    if history:
        dates, prices = zip(*history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(dates), y=list(prices),
            mode='lines+markers',
            name='Last 10 Price',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Date",
            yaxis_title="Price (PP)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Price momentum
        if len(prices) >= 3:
            recent_avg = sum(prices[-3:]) / 3
            older_avg = sum(prices[:3]) / 3
            if older_avg > 0:
                momentum_pct = ((recent_avg - older_avg) / older_avg) * 100
                if momentum_pct > 5:
                    st.warning(f"Price trending UP: +{momentum_pct:.1f}% recent vs older snapshots")
                elif momentum_pct < -5:
                    st.success(f"Price trending DOWN: {momentum_pct:.1f}% recent vs older snapshots")
                else:
                    st.info(f"Price stable: {momentum_pct:+.1f}% change")
    else:
        st.caption("Not enough price snapshots for a chart (need 2+).")

    # Full price snapshot table
    snapshots = conn.execute("""
        SELECT snapshot_date, buy_order_high, sell_order_low, last_10_price, last_10_variance
        FROM price_snapshots WHERE card_id = ?
        ORDER BY snapshot_date DESC LIMIT 20
    """, (card['card_id'],)).fetchall()
    if snapshots:
        with st.expander(f"Price Snapshot History ({len(snapshots)} entries)"):
            snap_data = [{
                "Date": s['snapshot_date'],
                "Buy High": s['buy_order_high'] or 0,
                "Sell Low": s['sell_order_low'] or 0,
                "Last 10": s['last_10_price'] or 0,
                "Variance": s['last_10_variance'] or 0,
            } for s in snapshots]
            st.dataframe(
                pd.DataFrame(snap_data),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Buy High": st.column_config.NumberColumn(
                        format="%d PP", width="small",
                        help="Highest active BID at snapshot time — what buyers were "
                             "willing to pay. Lower than Sell Low by the spread."),
                    "Sell Low": st.column_config.NumberColumn(
                        format="%d PP", width="small",
                        help="Lowest active ASK at snapshot time — what sellers were "
                             "asking. Higher than Buy High by the spread."),
                    "Last 10": st.column_config.NumberColumn(
                        format="%d PP", width="small",
                        help="Average of the last 10 completed sales — what the app "
                             "uses as the default Est. Price."),
                    "Variance": st.column_config.NumberColumn(
                        format="%d", width="small",
                        help="Std. dev. of the last 10 sales — high = volatile market."),
                },
            )

st.divider()

# ============================================================
# 4. IN-GAME PERFORMANCE
# ============================================================
st.subheader("In-Game Performance")

card_name = card['card_title'] or ''
# Try exact card_id match first, then name match
if is_pitcher:
    pstat = conn.execute(
        "SELECT * FROM pitching_stats WHERE card_id = ? ORDER BY snapshot_date DESC LIMIT 1",
        (card['card_id'],)
    ).fetchone()
    if not pstat and card_name:
        # Fuzzy name match
        name_parts = card_name.split()
        if len(name_parts) >= 2:
            pstat = conn.execute(
                "SELECT * FROM pitching_stats WHERE player_name LIKE ? ORDER BY snapshot_date DESC LIMIT 1",
                (f"%{name_parts[-2]}%{name_parts[-1]}%",)
            ).fetchone()

    if pstat:
        ps1, ps2, ps3, ps4 = st.columns(4)
        with ps1:
            st.metric("ERA", f"{pstat['era']:.2f}")
            st.metric("Games", pstat['games'] or 0)
        with ps2:
            st.metric("WHIP", f"{pstat['whip']:.2f}")
            st.metric("IP", f"{pstat['ip']:.1f}")
        with ps3:
            st.metric("K/9", f"{pstat['k_per_9']:.2f}")
            st.metric("BB/9", f"{pstat['bb_per_9']:.2f}")
        with ps4:
            st.metric("WAR", f"{pstat['war']:.1f}")
            st.metric("FIP", f"{pstat['fip']:.2f}")

        # ERA+ context
        if pstat['era_plus'] and pstat['era_plus'] > 0:
            era_plus = pstat['era_plus']
            if era_plus >= 150:
                st.success(f"ERA+ {era_plus} -- Elite pitcher")
            elif era_plus >= 100:
                st.info(f"ERA+ {era_plus} -- Above average")
            else:
                st.warning(f"ERA+ {era_plus} -- Below average")
    else:
        st.caption("No pitching stats found for this card.")
else:
    bstat = conn.execute(
        "SELECT * FROM batting_stats WHERE card_id = ? ORDER BY snapshot_date DESC LIMIT 1",
        (card['card_id'],)
    ).fetchone()
    if not bstat and card_name:
        name_parts = card_name.split()
        if len(name_parts) >= 2:
            bstat = conn.execute(
                "SELECT * FROM batting_stats WHERE player_name LIKE ? ORDER BY snapshot_date DESC LIMIT 1",
                (f"%{name_parts[-2]}%{name_parts[-1]}%",)
            ).fetchone()

    if bstat:
        bs1, bs2, bs3, bs4 = st.columns(4)
        with bs1:
            st.metric("AVG", f"{bstat['avg']:.3f}")
            st.metric("Games", bstat['games'] or 0)
        with bs2:
            st.metric("OPS", f"{bstat['ops']:.3f}")
            st.metric("PA", bstat['pa'] or 0)
        with bs3:
            st.metric("HR", bstat['hr'] or 0)
            st.metric("RBI", bstat['rbi'] or 0)
        with bs4:
            st.metric("WAR", f"{bstat['war']:.1f}")
            st.metric("SB", bstat['sb'] or 0)

        # OPS+ context
        if bstat['ops_plus'] and bstat['ops_plus'] > 0:
            ops_plus = bstat['ops_plus']
            if ops_plus >= 150:
                st.success(f"OPS+ {ops_plus} -- Elite hitter")
            elif ops_plus >= 100:
                st.info(f"OPS+ {ops_plus} -- Above average")
            else:
                st.warning(f"OPS+ {ops_plus} -- Below average")
    else:
        st.caption("No batting stats found for this card.")

st.divider()

# ============================================================
# 4b. SUPERSTATS (game-log-derived)
# ============================================================
# From the HTML game logs: per-at-bat exit velocity, batted ball type,
# observed handedness splits, and opponent-quality-adjusted performance.
# Pooled across every team in the league that has this card (cross-team
# sample, dramatically stabilizes vs single-team stats).
try:
    from app.core.superstats import card_full_superstat_report
    from app.core.card_aggregation import card_confidence
    _sr = card_full_superstat_report(card_id=card['card_id'], conn=conn)
    _ss = _sr.get('superstats') or {}
    _split = _sr.get('observed_splits') or {}
    _oq = _sr.get('opponent_adjusted') or {}
    _conf = card_confidence(card_id=card['card_id'], conn=conn)
    _n = _ss.get('n_at_bats', 0)
    if _n >= 30 or (_oq and _oq.get('pa', 0) >= 30):
        st.subheader("Superstats — from game logs")
        st.caption(
            f"Pooled across every team instance of this card in the league. "
            f"Based on {_n} at-bats parsed from OOTP HTML game logs."
        )
        # Top metrics
        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        if _ss.get('ev_avg') is not None:
            sm1.metric("Avg Exit Velocity", f"{_ss['ev_avg']:.1f} mph",
                       delta=f"p90 {_ss.get('ev_p90')}" if _ss.get('ev_p90') else None,
                       delta_color="off")
        if _ss.get('ld_pct') is not None:
            sm2.metric("Line Drive %", f"{_ss['ld_pct']:.0f}%",
                       delta=f"lg avg ~28%", delta_color="off")
        if _ss.get('k_pct') is not None:
            sm3.metric("True K %", f"{_ss['k_pct']:.1f}%",
                       delta=f"{_ss.get('strikeouts_swinging', 0)}sw / {_ss.get('strikeouts_looking', 0)}look",
                       delta_color="off")
        if _ss.get('hard_hit_rate') is not None:
            sm4.metric("Hard-hit % (≥90 mph)", f"{_ss['hard_hit_rate']:.0f}%",
                       delta=f"barrel {_ss.get('barrel_rate', 0):.0f}%", delta_color="off")
        if _conf and _conf.get('score') is not None:
            lbl = _conf.get('label', 'none')
            emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(lbl, '⚪')
            sm5.metric("Confidence", f"{emoji} {_conf['score']}",
                       delta=lbl.title(), delta_color="off")

        # Batted-ball distribution
        bb_types = _ss.get('batted_balls') or {}
        if bb_types:
            st.markdown("**Batted ball mix**")
            bbc1, bbc2, bbc3 = st.columns(3)
            bbc1.write(f"Line Drive — **{_ss.get('ld_pct') or 0:.0f}%** "
                       f"(EV {_ss.get('ev_on_ld') or 0:.0f} mph)")
            bbc2.write(f"Groundball — **{_ss.get('gb_pct') or 0:.0f}%** "
                       f"(EV {_ss.get('ev_on_gb') or 0:.0f} mph)")
            bbc3.write(f"Fly/Popup — **{_ss.get('fb_pct') or 0:.0f}%** "
                       f"(EV {_ss.get('ev_on_fb') or 0:.0f} mph)")

        # Observed splits
        vs_l = _split.get('vs_LHP') or {}
        vs_r = _split.get('vs_RHP') or {}
        if (vs_l.get('pa') or 0) >= 20 or (vs_r.get('pa') or 0) >= 20:
            st.markdown("**Observed platoon splits (from game logs, not card ratings)**")
            sp1, sp2 = st.columns(2)
            with sp1:
                st.markdown("vs LHP")
                if vs_l.get('insufficient_sample') or (vs_l.get('pa') or 0) < 20:
                    st.caption(f"Only {vs_l.get('pa', 0)} PA — insufficient")
                else:
                    st.write(f"**{vs_l['pa']}** PA · hit% **{vs_l['hit_pct']:.0f}** · "
                             f"K% {vs_l['k_pct']:.0f} · BB% {vs_l['bb_pct']:.0f} · "
                             f"HR% {vs_l['hr_pct']:.1f} · EV {vs_l.get('ev_avg') or 0:.0f}")
            with sp2:
                st.markdown("vs RHP")
                if vs_r.get('insufficient_sample') or (vs_r.get('pa') or 0) < 20:
                    st.caption(f"Only {vs_r.get('pa', 0)} PA — insufficient")
                else:
                    st.write(f"**{vs_r['pa']}** PA · hit% **{vs_r['hit_pct']:.0f}** · "
                             f"K% {vs_r['k_pct']:.0f} · BB% {vs_r['bb_pct']:.0f} · "
                             f"HR% {vs_r['hr_pct']:.1f} · EV {vs_r.get('ev_avg') or 0:.0f}")
            gap = _split.get('split_gap_hit_pct')
            if gap is not None and gap >= 8:
                st.warning(f"Platoon gap: **{gap:.1f}pp** difference in hit% between LHP/RHP splits — "
                           f"use this card in favorable matchups and platoon the weak side.")

        # Opponent quality
        if _oq.get('available'):
            st.markdown("**Opponent-quality context**")
            oqc1, oqc2, oqc3, oqc4 = st.columns(4)
            oqc1.metric("Avg opposing ERA+ faced",
                        f"{_oq.get('avg_opp_era_plus') or 0:.0f}",
                        delta="harder than avg" if (_oq.get('avg_opp_era_plus') or 100) > 105
                              else ("easier than avg" if (_oq.get('avg_opp_era_plus') or 100) < 95 else "average"),
                        delta_color="off")
            oqc2.metric("PA vs aces (ERA+ ≥125)", _oq.get('faced_aces', 0))
            oqc3.metric("PA vs filler (ERA+ ≤80)", _oq.get('faced_filler', 0))
            oqc4.metric("Adjusted hit%",
                        f"{_oq.get('adjusted_hit_pct') or 0:.1f}%",
                        delta=f"raw {_oq.get('raw_hit_pct') or 0:.1f}%", delta_color="off")
    else:
        st.caption("No game-log at-bats yet for this card (need HTML logs "
                   "from OOTP — see Almanac setting).")
except Exception as _e:
    # Don't break the page if superstats aren't available
    st.caption(f"Superstats unavailable: {_e}")

st.divider()

# ============================================================
# 5. ROSTER CONTEXT
# ============================================================
st.subheader("Roster Context")

roster_player = conn.execute("""
    SELECT player_name, ovr, meta_score, position, lineup_role
    FROM roster_current
    WHERE position = ? AND lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
    ORDER BY meta_score DESC LIMIT 1
""", (pos_display,)).fetchone()

card_meta = card['meta_score_pitching'] if is_pitcher else card['meta_score_batting']
card_meta = card_meta or 0

if roster_player:
    ctx1, ctx2, ctx3 = st.columns(3)
    with ctx1:
        st.markdown(f"**Current {pos_display} Starter**")
        st.write(f"{roster_player['player_name']}")
        st.write(f"OVR: {roster_player['ovr']} | Meta: {roster_player['meta_score']:.0f}" if roster_player['meta_score'] else "")
    with ctx2:
        delta = card_meta - (roster_player['meta_score'] or 0)
        st.markdown("**Meta Improvement**")
        if delta > 0:
            st.metric("Upgrade", f"+{delta:.0f} meta")
        elif delta < 0:
            st.metric("Downgrade", f"{delta:.0f} meta")
        else:
            st.metric("Same", "0 meta")
    with ctx3:
        price = card['last_10_price'] or card['sell_order_low'] or 0
        if price > 0 and card_meta > 0:
            value_ratio = round((card_meta * card_meta) / price, 1)
            st.markdown("**Value Ratio**")
            st.metric("Meta^2 / Price", f"{value_ratio:.1f}")
            if value_ratio > 50:
                st.success("Excellent value")
            elif value_ratio > 20:
                st.info("Good value")
            else:
                st.caption("Standard value")
        else:
            st.markdown("**Value Ratio**")
            st.caption("Insufficient data")
else:
    st.info(f"No active roster starter found at {pos_display}. This card would fill an open slot.")
    if card_meta > 0:
        st.metric("Card Meta Score", f"{card_meta:.0f}")

# Ownership status
st.divider()
owned = card['owned'] or 0
if owned:
    st.success(f"You own this card (owned count: {owned})")
else:
    st.info("You do not own this card.")

conn.close()
