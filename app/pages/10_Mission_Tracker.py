"""Mission Tracker — Live Card collection progress for team missions."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection
from app.core.mission_tracker import get_mission_progress, get_mission_summary, get_best_mission_buys

st.set_page_config(page_title="Mission Tracker", page_icon="🎯", layout="wide")
st.title("Mission Tracker")
st.caption("Track Live Card collection by team — own at least 1 per team for mission eligibility")

conn = get_connection()

# Summary metrics
summary = get_mission_summary(conn)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Teams Covered", f"{summary['teams_covered']}/30")
with col2:
    st.metric("Teams Missing", len(summary['teams_needed']))
with col3:
    st.metric("Cost to Complete", f"{summary['total_cost_to_complete']:,} PP",
              help="Estimated PP to buy the cheapest Live card for each missing team")
with col4:
    st.metric("Total Mission Value", f"{summary['total_mission_value']:,}")

if summary['teams_covered'] == 30:
    st.success("All 30 teams covered! You're mission-eligible for every team.")

st.divider()

# Shopping list for missing teams
if summary['teams_needed']:
    st.subheader("Shopping List — Complete Your Missions")
    st.caption("Cheapest Live card per missing team. Most are under 10 PP — quick wins.")

    shopping = get_best_mission_buys(conn, max_price=5000)
    if shopping:
        total_cost = sum(s['price'] for s in shopping)
        shop_data = []
        for s in shopping:
            shop_data.append({
                "Team": s['team'],
                "Card": s['card_title'],
                "Price": s['price'],
                "OVR": s['card_value'],
                "Mission Value": s['mission_value'],
            })
        st.markdown(f"**{len(shopping)} cards needed — total cost: {total_cost:,} PP**")
        st.dataframe(pd.DataFrame(shop_data), use_container_width=True, hide_index=True)
    else:
        st.info("No affordable Live cards found for missing teams.")

st.divider()

# Full team-by-team breakdown
st.subheader("Collection Progress by Team")

progress = get_mission_progress(conn)

# Separate into covered and missing for visual clarity
prog_data = []
for t in progress:
    status = "✅" if t['has_any'] else "❌"
    prog_data.append({
        "Status": status,
        "Team": t['team'],
        "Owned": t['owned_count'],
        "Available": t['total_cards'] - t['owned_count'],
        "Total Cards": t['total_cards'],
        "Mission Value": t['mission_value_total'],
        "Cheapest (PP)": f"{t['cheapest_available']:,}" if t['cheapest_available'] else "—",
    })

df = pd.DataFrame(prog_data)
st.dataframe(df, use_container_width=True, hide_index=True)

# Expandable owned cards per team
st.divider()
st.subheader("Owned Live Cards Detail")

owned_teams = [t for t in progress if t['has_any']]
if owned_teams:
    for t in owned_teams:
        with st.expander(f"{t['team']} ({t['owned_count']} cards)"):
            team_cards = conn.execute("""
                SELECT card_title, card_value, tier_name,
                       COALESCE(pitcher_role_name, position_name) as pos,
                       COALESCE(meta_score_batting, meta_score_pitching) as meta,
                       last_10_price, mission_value
                FROM cards
                WHERE card_title LIKE 'MLB 2026 Live%' AND team = ? AND owned > 0
                ORDER BY card_value DESC
            """, (t['team'],)).fetchall()

            if team_cards:
                tc_data = []
                for c in team_cards:
                    tc_data.append({
                        "Card": c['card_title'],
                        "Pos": c['pos'] or '',
                        "OVR": c['card_value'],
                        "Meta": f"{c['meta']:.0f}" if c['meta'] else "—",
                        "Price": f"{c['last_10_price']:,}" if c['last_10_price'] else "—",
                        "Mission": c['mission_value'] or 0,
                    })
                st.dataframe(pd.DataFrame(tc_data), use_container_width=True, hide_index=True)

conn.close()
