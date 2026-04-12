"""Tournament Roster Builder — optimize your roster for Perfect Team tournaments."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection
from app.core.tournament import (
    get_tournament_presets, auto_build_roster, get_eligible_cards,
    calculate_chemistry, calculate_salary, validate_roster,
    BATTING_POSITIONS, ROSTER_SIZE, EMPTY_SLOT_VALUE,
)

st.set_page_config(page_title="Tournament Builder", page_icon="🏆", layout="wide")
st.title("🏆 Tournament Roster Builder")
st.caption("Build and optimize your roster for Perfect Team tournaments")

conn = get_connection()

# ============================================================
# Tournament Configuration
# ============================================================
st.subheader("Tournament Settings")

presets = get_tournament_presets()
preset_names = ['Custom'] + list(presets.keys())

col_preset, col_format = st.columns(2)
with col_preset:
    selected_preset = st.selectbox("Preset", preset_names,
                                    help="Choose a common tournament type or customize")
with col_format:
    tournament_format = st.selectbox("Format", [
        "Best of 5 (Bracket)", "Best of 7 (Bracket)", "Do or Die (Single Elim)",
        "Round Robin (Pool Play)", "Double Elimination"
    ], help="Tournament format affects pitching depth needs")

# Load preset values or defaults
if selected_preset != 'Custom' and selected_preset in presets:
    preset = presets[selected_preset]
    st.info(f"**{preset['name']}**: {preset['description']}")
    default_cap = preset['salary_cap']
    default_min_ovr = preset['min_ovr']
    default_max_ovr = preset['max_ovr']
    default_chemistry = preset['chemistry_enabled']
else:
    default_cap = 0
    default_min_ovr = 0
    default_max_ovr = 0
    default_chemistry = True

# Constraint inputs
with st.expander("Advanced Constraints", expanded=(selected_preset == 'Custom')):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        salary_cap = st.number_input("Salary Cap (0 = none)", min_value=0, max_value=5000,
                                      value=default_cap, step=50,
                                      help="Sum of all roster OVR ratings. Empty slots = 40 each.")
    with c2:
        min_ovr = st.number_input("Min Card OVR (0 = none)", min_value=0, max_value=100,
                                   value=default_min_ovr, step=5)
    with c3:
        max_ovr = st.number_input("Max Card OVR (0 = none)", min_value=0, max_value=100,
                                   value=default_max_ovr, step=5)
    with c4:
        chemistry_enabled = st.checkbox("Chemistry Enabled", value=default_chemistry)

    c5, c6 = st.columns(2)
    with c5:
        # Get distinct card types from owned cards
        try:
            card_type_rows = conn.execute(
                "SELECT DISTINCT card_sub_type FROM cards WHERE owned > 0 AND card_sub_type IS NOT NULL ORDER BY card_sub_type"
            ).fetchall()
            available_types = [r['card_sub_type'] for r in card_type_rows if r['card_sub_type']]
        except Exception:
            available_types = []
        card_types = st.multiselect("Card Types (empty = all)", available_types,
                                     help="Restrict to specific card types")
    with c6:
        try:
            series_rows = conn.execute(
                "SELECT DISTINCT card_series FROM cards WHERE owned > 0 AND card_series IS NOT NULL ORDER BY card_series"
            ).fetchall()
            available_series = [r['card_series'] for r in series_rows if r['card_series']]
        except Exception:
            available_series = []
        card_series = st.multiselect("Card Series (empty = all)", available_series,
                                      help="Restrict to specific card series")

    year_filter = st.number_input("Card Year (0 = any)", min_value=0, max_value=2030, value=0, step=1)

constraints = {
    'salary_cap': salary_cap,
    'min_ovr': min_ovr,
    'max_ovr': max_ovr,
    'card_types': card_types,
    'card_series': card_series,
    'max_combinators': -1,
    'year': year_filter,
    'chemistry_enabled': chemistry_enabled,
}

# ============================================================
# Build Roster
# ============================================================
st.divider()

build_col1, build_col2 = st.columns([3, 1])
with build_col1:
    build_clicked = st.button("Build Optimal Roster", type="primary", use_container_width=True)
with build_col2:
    eligible_cards = get_eligible_cards(conn, constraints)
    st.metric("Eligible Cards", len(eligible_cards))

if build_clicked:
    if not eligible_cards:
        st.error("No eligible cards found. Check your constraints or import collection data first.")
    else:
        with st.spinner("Building optimal tournament roster..."):
            result = auto_build_roster(conn, constraints)

        roster = result['roster']
        salary = result['salary']
        chemistry = result['chemistry']
        validation = result['validation']

        # ── Validation Status ──
        if validation['valid']:
            st.success(f"Valid roster: {len(roster)} players selected")
        else:
            for err in validation['errors']:
                st.error(err)

        for warn in validation.get('warnings', []):
            st.warning(warn)

        # ── Summary Metrics ──
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Roster Size", f"{salary['filled_slots']}/{ROSTER_SIZE}")
        with m2:
            total_meta = sum(c.get('meta_score', 0) for c in roster)
            st.metric("Total Meta", f"{total_meta:,.0f}")
        with m3:
            cap_display = f"{salary['total_salary']}"
            if salary_cap > 0:
                cap_display += f" / {salary_cap}"
                if salary['total_salary'] > salary_cap:
                    st.metric("Salary", cap_display, delta="OVER CAP", delta_color="inverse")
                else:
                    remaining = salary_cap - salary['total_salary']
                    st.metric("Salary", cap_display, delta=f"{remaining} under", delta_color="normal")
            else:
                st.metric("Salary (OVR)", cap_display)
        with m4:
            avg_ovr = sum(c.get('ovr', 0) or 0 for c in roster) / len(roster) if roster else 0
            st.metric("Avg OVR", f"{avg_ovr:.1f}")
        with m5:
            if chemistry_enabled:
                st.metric("Chemistry", f"{chemistry['total_score']:.1f}")
            else:
                st.metric("Chemistry", "Disabled")

        # ── Salary Cap Visualization ──
        if salary_cap > 0:
            usage_pct = min(1.0, salary['total_salary'] / salary_cap)
            st.progress(usage_pct, text=f"Salary: {salary['total_salary']} / {salary_cap} ({usage_pct*100:.0f}%)")

        st.divider()

        # ── Roster Display ──
        tab_lineup, tab_chemistry, tab_acquire, tab_swap = st.tabs([
            "Lineup", "Chemistry", "Acquire", "Swap Cards"
        ])

        with tab_lineup:
            st.subheader("Tournament Lineup")

            # Position map for display
            pos_order = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH', 'SP', 'RP', 'CL']

            # Group roster by position
            by_pos = defaultdict(list)
            for c in roster:
                by_pos[c.get('role', '?')].append(c)

            # Batting lineup
            st.markdown("**Starting Lineup**")
            lineup_data = []
            for pos in BATTING_POSITIONS:
                players = by_pos.get(pos, [])
                if players:
                    best = max(players, key=lambda x: x.get('meta_score', 0))
                    lineup_data.append({
                        "Pos": pos,
                        "Player": best['card_title'] or '',
                        "OVR": best.get('ovr', 0),
                        "Meta": round(best.get('meta_score', 0)),
                        "Tier": best.get('tier_name', ''),
                        "Bats": best.get('bats', ''),
                    })
                else:
                    lineup_data.append({
                        "Pos": pos, "Player": "-- EMPTY --", "OVR": 0,
                        "Meta": 0, "Tier": "", "Bats": "",
                    })

            st.dataframe(pd.DataFrame(lineup_data), use_container_width=True, hide_index=True)

            # Pitching staff
            st.markdown("**Pitching Staff**")
            pitch_data = []
            for pos in ['SP', 'RP', 'CL']:
                for p in sorted(by_pos.get(pos, []), key=lambda x: x.get('meta_score', 0), reverse=True):
                    pitch_data.append({
                        "Role": pos,
                        "Player": p['card_title'] or '',
                        "OVR": p.get('ovr', 0),
                        "Meta": round(p.get('meta_score', 0)),
                        "Tier": p.get('tier_name', ''),
                        "Throws": p.get('throws', ''),
                    })
            if pitch_data:
                st.dataframe(pd.DataFrame(pitch_data), use_container_width=True, hide_index=True)

            # Bench
            bench_players = [c for c in roster if c not in result.get('starters', roster)]
            if bench_players:
                st.markdown("**Bench**")
                bench_data = []
                for b in bench_players:
                    bench_data.append({
                        "Pos": b.get('role', '?'),
                        "Player": b['card_title'] or '',
                        "OVR": b.get('ovr', 0),
                        "Meta": round(b.get('meta_score', 0)),
                        "Tier": b.get('tier_name', ''),
                    })
                st.dataframe(pd.DataFrame(bench_data), use_container_width=True, hide_index=True)

            # Position coverage chart
            st.markdown("**Position Coverage**")
            pos_counts = validation['position_counts']
            fig = go.Figure(data=[
                go.Bar(
                    x=pos_order,
                    y=[pos_counts.get(p, 0) for p in pos_order],
                    marker_color=['#2ecc71' if pos_counts.get(p, 0) >= 1 else '#e74c3c' for p in pos_order],
                    text=[pos_counts.get(p, 0) for p in pos_order],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                height=250, margin=dict(t=10, b=30, l=40, r=10),
                xaxis_title="Position", yaxis_title="Count",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_chemistry:
            st.subheader("Team Chemistry Analysis")

            if not chemistry_enabled:
                st.info("Chemistry is disabled for this tournament.")
            else:
                # Chemistry breakdown
                chem_col1, chem_col2 = st.columns(2)
                with chem_col1:
                    st.metric("Overall Chemistry", f"{chemistry['total_score']:.1f} / 100")

                    # Breakdown bars
                    if chemistry['breakdown']:
                        for factor, score in chemistry['breakdown'].items():
                            st.caption(factor)
                            st.progress(min(1.0, score / 100), text=f"{score:.1f}%")

                with chem_col2:
                    if chemistry.get('pairs'):
                        st.markdown("**Notable Chemistry Pairs**")
                        for pair in chemistry['pairs']:
                            st.write(f"- {pair}")
                    else:
                        st.info("No strong chemistry pairs found. Consider cards from the same franchise, year, or card type.")

                # Chemistry optimization tips
                st.divider()
                st.markdown("**Chemistry Tips for This Roster**")

                # Analyze dominant traits
                franchises = defaultdict(int)
                card_types = defaultdict(int)
                years = defaultdict(int)
                for c in roster:
                    if c.get('franchise'):
                        franchises[c['franchise']] += 1
                    if c.get('card_sub_type'):
                        card_types[c['card_sub_type']] += 1
                    if c.get('year'):
                        years[c['year']] += 1

                if franchises:
                    top_franchise = max(franchises.items(), key=lambda x: x[1])
                    st.write(f"Most common franchise: **{top_franchise[0]}** ({top_franchise[1]} cards)")
                    if top_franchise[1] >= 5:
                        st.success(f"Strong franchise chemistry with {top_franchise[1]} {top_franchise[0]} cards!")

                if card_types:
                    top_type = max(card_types.items(), key=lambda x: x[1])
                    st.write(f"Most common card type: **{top_type[0]}** ({top_type[1]} cards)")

                if years:
                    top_year = max(years.items(), key=lambda x: x[1])
                    st.write(f"Most common year: **{top_year[0]}** ({top_year[1]} cards)")

        with tab_acquire:
            st.subheader("Cards to Acquire")
            st.caption("Market cards that would improve your tournament roster")

            recs = result.get('recommendations', [])
            if recs:
                rec_data = []
                for r in recs:
                    rec_data.append({
                        "Card": r['card_title'] or '',
                        "Pos": r['position'] or '',
                        "OVR": r.get('ovr', 0),
                        "Meta": round(r.get('meta_score', 0)),
                        "Price": r.get('price', 0),
                        "Tier": r.get('tier', ''),
                        "+Meta": round(r.get('improvement', 0)),
                        "Reason": r.get('reason', ''),
                    })
                st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

                total_cost = sum(r.get('price', 0) for r in recs)
                st.caption(f"Total acquisition cost: **{total_cost:,} PP** for top {len(recs)} upgrades")
            else:
                st.success("Your roster looks fully optimized for these constraints!")

        with tab_swap:
            st.subheader("Manual Card Swaps")
            st.caption("Swap cards in and out of the roster manually")

            # Show excluded eligible cards
            excluded = result.get('excluded', [])
            if excluded:
                swap_col1, swap_col2 = st.columns(2)
                with swap_col1:
                    roster_options = {f"{c['card_title']} ({c.get('role','?')}, OVR {c.get('ovr',0)})": c['card_id']
                                     for c in roster}
                    remove_card = st.selectbox("Remove from roster", ["(none)"] + list(roster_options.keys()))

                with swap_col2:
                    excluded_options = {f"{c['card_title']} ({c.get('role','?')}, OVR {c.get('ovr',0)})": c['card_id']
                                        for c in excluded}
                    add_card = st.selectbox("Add to roster", ["(none)"] + list(excluded_options.keys()))

                if remove_card != "(none)" or add_card != "(none)":
                    # Show impact preview
                    preview_roster = list(roster)
                    if remove_card != "(none)":
                        remove_id = roster_options[remove_card]
                        preview_roster = [c for c in preview_roster if c['card_id'] != remove_id]
                    if add_card != "(none)":
                        add_id = excluded_options[add_card]
                        add_card_data = next((c for c in excluded if c['card_id'] == add_id), None)
                        if add_card_data:
                            preview_roster.append(add_card_data)

                    new_salary = calculate_salary(preview_roster)
                    new_meta = sum(c.get('meta_score', 0) for c in preview_roster)
                    old_meta = sum(c.get('meta_score', 0) for c in roster)

                    p1, p2, p3 = st.columns(3)
                    with p1:
                        meta_delta = new_meta - old_meta
                        st.metric("Meta Change", f"{meta_delta:+.0f}")
                    with p2:
                        sal_delta = new_salary['total_salary'] - salary['total_salary']
                        st.metric("Salary Change", f"{sal_delta:+d}")
                    with p3:
                        if salary_cap > 0 and new_salary['total_salary'] > salary_cap:
                            st.error(f"Over salary cap! ({new_salary['total_salary']} > {salary_cap})")
                        else:
                            st.success("Within constraints")
            else:
                st.info("No excluded cards available for swapping.")

        # ── Format-Specific Tips ──
        st.divider()
        st.subheader("Format Strategy Tips")

        sp_count = sum(1 for c in roster if c.get('role') == 'SP')
        rp_count = sum(1 for c in roster if c.get('role') in ('RP', 'CL'))

        if "Best of 7" in tournament_format:
            st.markdown("""
            **Best of 7 Series Strategy:**
            - Need **5 strong SP** for full rotation coverage
            - Deep bullpen (6-7 RP) critical for long series
            - Stamina matters — check SP stamina ratings
            - Consider platooning: have both L and R bats available
            """)
            if sp_count < 5:
                st.warning(f"You only have {sp_count} SP — need 5 for Bo7 coverage")
        elif "Best of 5" in tournament_format:
            st.markdown("""
            **Best of 5 Series Strategy:**
            - **4-5 SP** sufficient — your top 3 could go twice
            - Bullpen depth still important (5-6 RP)
            - Ace quality matters more than rotation depth
            """)
        elif "Do or Die" in tournament_format:
            st.markdown("""
            **Single Elimination Strategy:**
            - Your **#1 SP** is everything — maximize ace quality
            - Bullpen depth less critical, but late-game relief matters
            - Every lineup spot counts — no room for cold bats
            - Consider high-OPS bats over balanced lineups
            """)
        elif "Round Robin" in tournament_format:
            st.markdown("""
            **Round Robin Strategy:**
            - Playing 3-9 games in pool — need full rotation
            - Consistency matters more than peak performance
            - Deep pitching prevents burnout across pool games
            - Every position player will get lots of ABs
            """)
        elif "Double Elimination" in tournament_format:
            st.markdown("""
            **Double Elimination Strategy:**
            - You can afford one loss — be aggressive early
            - Deep rotation needed for potentially many games
            - Bullpen management critical in late bracket
            - 7-8 RP recommended for modern era tournaments
            """)

        # Handedness balance
        left_bats = sum(1 for c in roster if not c.get('is_pitcher') and c.get('bats') == 'L')
        right_bats = sum(1 for c in roster if not c.get('is_pitcher') and c.get('bats') == 'R')
        switch_bats = sum(1 for c in roster if not c.get('is_pitcher') and c.get('bats') == 'S')
        left_throws = sum(1 for c in roster if c.get('is_pitcher') and c.get('throws') == 'L')
        right_throws = sum(1 for c in roster if c.get('is_pitcher') and c.get('throws') == 'R')

        h1, h2, h3, h4, h5 = st.columns(5)
        with h1:
            st.metric("L Bats", left_bats)
        with h2:
            st.metric("R Bats", right_bats)
        with h3:
            st.metric("Switch", switch_bats)
        with h4:
            st.metric("LHP", left_throws)
        with h5:
            st.metric("RHP", right_throws)

        if left_bats == 0:
            st.warning("No left-handed batters — vulnerable to RHP-heavy opponents")
        if right_bats == 0:
            st.warning("No right-handed batters — vulnerable to LHP-heavy opponents")
        if left_throws == 0:
            st.warning("No left-handed pitchers — consider adding a lefty for bullpen matchups")

# ============================================================
# Quick Tournament Card Search
# ============================================================
st.divider()
st.subheader("Tournament Card Search")
st.caption("Search your collection for tournament-eligible cards")

search_query = st.text_input("Search owned cards", placeholder="e.g., Babe Ruth, or position like SS")

if search_query:
    search_results = conn.execute("""
        SELECT card_id, card_title,
               COALESCE(pitcher_role_name, position_name) as pos,
               card_value as ovr, tier_name,
               COALESCE(meta_score_batting, meta_score_pitching, 0) as meta_score,
               card_sub_type, card_series, year, franchise, bats, throws
        FROM cards
        WHERE owned > 0
            AND (card_title LIKE ? OR position_name LIKE ? OR pitcher_role_name LIKE ?)
        ORDER BY COALESCE(meta_score_batting, meta_score_pitching, 0) DESC
        LIMIT 50
    """, (f"%{search_query}%", f"%{search_query}%", f"%{search_query}%")).fetchall()

    if search_results:
        search_data = []
        for r in search_results:
            search_data.append({
                "Card": r['card_title'] or '',
                "Pos": r['pos'] or '',
                "OVR": r['ovr'] or 0,
                "Meta": round(r['meta_score']),
                "Tier": r['tier_name'] or '',
                "Type": r['card_sub_type'] or '',
                "Series": r['card_series'] or '',
                "Year": r['year'] or '',
                "Team": r['franchise'] or '',
                "Bats": r['bats'] or '',
                "Throws": r['throws'] or '',
            })
        st.dataframe(pd.DataFrame(search_data), use_container_width=True, hide_index=True)
    else:
        st.info(f"No owned cards matching '{search_query}'")

conn.close()
