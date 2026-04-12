"""Live Card Tracker — MLB stats vs OOTP ratings for upgrade/downgrade prediction."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection
from app.core.live_card_tracker import get_live_cards, analyze_live_cards
from app.core.recommendations import cache_live_card_analysis

st.set_page_config(page_title="Live Card Tracker", page_icon="📡", layout="wide")
st.title("Live Card Tracker")
st.caption("Compare OOTP Live card ratings to real MLB performance — spot upgrades before they happen")

conn = get_connection()

# Count live cards
live_total = conn.execute("SELECT COUNT(*) FROM cards WHERE card_title LIKE 'MLB 2026 Live%'").fetchone()[0]
live_owned = conn.execute("SELECT COUNT(*) FROM cards WHERE card_title LIKE 'MLB 2026 Live%' AND owned > 0").fetchone()[0]

col1, col2, col3 = st.columns(3)
col1.metric("Total Live Cards", live_total)
col2.metric("Live Cards Owned", live_owned)
col3.metric("Live Cards on Market", live_total - live_owned)

st.divider()

# Analysis controls
st.subheader("Analyze Live Cards vs MLB Stats")
st.info("This pulls real-time stats from the MLB Stats API. Each card takes ~0.3s to look up, so analyzing many cards takes a moment.")

acol1, acol2, acol3 = st.columns(3)
with acol1:
    scope = st.radio("Scope", ["My Cards", "Top Market Cards", "All (slow)"], index=0)
with acol2:
    max_cards = st.slider("Max cards to analyze", 10, 200, 30, step=10)
with acol3:
    sort_option = st.selectbox("Sort results by", ["Upgrade Signal", "Downgrade Signal", "Price (High)", "OVR (High)"])

if st.button("Run Analysis", type="primary"):
    owned_only = (scope == "My Cards")
    limit = max_cards if scope != "All (slow)" else 500

    progress_bar = st.progress(0, text="Starting analysis...")

    def update_progress(current, total, name):
        pct = current / total
        progress_bar.progress(pct, text=f"Analyzing {name} ({current}/{total})...")

    with st.spinner("Fetching MLB stats..."):
        results = analyze_live_cards(
            max_cards=limit,
            owned_only=owned_only,
            progress_callback=update_progress,
        )

    progress_bar.progress(1.0, text="Analysis complete!")

    if not results:
        st.warning("No Live cards found to analyze.")
    else:
        # Sort based on user preference
        if sort_option == "Upgrade Signal":
            results.sort(key=lambda x: x['analysis']['score'], reverse=True)
        elif sort_option == "Downgrade Signal":
            results.sort(key=lambda x: x['analysis']['score'])
        elif sort_option == "Price (High)":
            results.sort(key=lambda x: x['card'].get('last_10_price', 0) or 0, reverse=True)
        elif sort_option == "OVR (High)":
            results.sort(key=lambda x: x['card'].get('card_value', 0) or 0, reverse=True)

        # Split into upgrade / hold / downgrade
        upgrades = [r for r in results if r['analysis']['signal'] == 'upgrade']
        downgrades = [r for r in results if r['analysis']['signal'] == 'downgrade']
        holds = [r for r in results if r['analysis']['signal'] in ('hold', 'unknown')]

        st.subheader(f"Results: {len(upgrades)} upgrades, {len(downgrades)} downgrades, {len(holds)} holds")

        # Upgrade opportunities
        if upgrades:
            st.markdown("### 🟢 Upgrade Candidates — BUY before rating increase")
            upgrade_data = []
            for r in upgrades:
                card = r['card']
                mlb = r['mlb_stats'] or {}
                a = r['analysis']
                pos = card.get('pitcher_role_name') or card.get('position_name') or '?'

                row = {
                    "Signal": f"⬆️ +{a['score']}",
                    "Confidence": a['confidence'].title(),
                    "Player": r['player_name'],
                    "Pos": pos,
                    "OVR": card.get('card_value', 0),
                    "Tier": card.get('tier_name', ''),
                    "Price": card.get('last_10_price', 0),
                    "Owned": "Yes" if card.get('owned', 0) > 0 else "",
                }

                if r['is_pitcher']:
                    row["ERA"] = f"{mlb.get('era', 0):.2f}" if mlb else "—"
                    row["WHIP"] = f"{mlb.get('whip', 0):.2f}" if mlb else "—"
                    row["K/9"] = f"{mlb.get('k_per_9', 0):.1f}" if mlb else "—"
                    row["IP"] = f"{mlb.get('ip', 0):.1f}" if mlb else "—"
                else:
                    row["AVG"] = f"{mlb.get('avg', 0):.3f}" if mlb else "—"
                    row["OPS"] = f"{mlb.get('ops', 0):.3f}" if mlb else "—"
                    row["HR"] = mlb.get('hr', 0) if mlb else "—"
                    row["PA"] = mlb.get('pa', 0) if mlb else "—"

                row["Reasons"] = " | ".join(a['reasons'])
                upgrade_data.append(row)

            st.dataframe(pd.DataFrame(upgrade_data), use_container_width=True, hide_index=True)

        # Downgrade warnings
        if downgrades:
            st.markdown("### 🔴 Downgrade Risk — SELL before rating decrease")
            downgrade_data = []
            for r in downgrades:
                card = r['card']
                mlb = r['mlb_stats'] or {}
                a = r['analysis']
                pos = card.get('pitcher_role_name') or card.get('position_name') or '?'

                row = {
                    "Signal": f"⬇️ {a['score']}",
                    "Confidence": a['confidence'].title(),
                    "Player": r['player_name'],
                    "Pos": pos,
                    "OVR": card.get('card_value', 0),
                    "Tier": card.get('tier_name', ''),
                    "Price": card.get('last_10_price', 0),
                    "Owned": "Yes" if card.get('owned', 0) > 0 else "",
                }

                if r['is_pitcher']:
                    row["ERA"] = f"{mlb.get('era', 0):.2f}" if mlb else "—"
                    row["WHIP"] = f"{mlb.get('whip', 0):.2f}" if mlb else "—"
                    row["K/9"] = f"{mlb.get('k_per_9', 0):.1f}" if mlb else "—"
                    row["IP"] = f"{mlb.get('ip', 0):.1f}" if mlb else "—"
                else:
                    row["AVG"] = f"{mlb.get('avg', 0):.3f}" if mlb else "—"
                    row["OPS"] = f"{mlb.get('ops', 0):.3f}" if mlb else "—"
                    row["HR"] = mlb.get('hr', 0) if mlb else "—"
                    row["PA"] = mlb.get('pa', 0) if mlb else "—"

                row["Reasons"] = " | ".join(a['reasons'])
                downgrade_data.append(row)

            st.dataframe(pd.DataFrame(downgrade_data), use_container_width=True, hide_index=True)

        # Holds
        if holds:
            with st.expander(f"🟡 Holds / Unknown ({len(holds)} cards)"):
                hold_data = []
                for r in holds:
                    card = r['card']
                    a = r['analysis']
                    pos = card.get('pitcher_role_name') or card.get('position_name') or '?'
                    hold_data.append({
                        "Player": r['player_name'],
                        "Pos": pos,
                        "OVR": card.get('card_value', 0),
                        "Price": card.get('last_10_price', 0),
                        "Owned": "Yes" if card.get('owned', 0) > 0 else "",
                        "Notes": " | ".join(a['reasons']),
                    })
                st.dataframe(pd.DataFrame(hold_data), use_container_width=True, hide_index=True)

        # Cache results for recommendation engine integration
        cache_live_card_analysis(results)
        st.success("✅ Analysis cached — upgrade/downgrade signals will now boost Buy Recommendations on next import.")

        # Store results in session state for the detail view below
        st.session_state['live_analysis_results'] = results

st.divider()

# Individual card lookup
st.subheader("Quick Player Lookup")
player_search = st.text_input("Search Live cards by player name")
if player_search:
    matches = conn.execute("""
        SELECT card_id, card_title, position_name, pitcher_role_name, card_value, tier_name,
               contact, gap_power, power, eye, avoid_ks, babip,
               stuff, movement, control, p_hr,
               owned, last_10_price, sell_order_low, buy_order_high,
               meta_score_batting, meta_score_pitching
        FROM cards
        WHERE card_title LIKE ? AND card_title LIKE 'MLB 2026 Live%'
        ORDER BY card_value DESC
        LIMIT 10
    """, (f"%{player_search}%",)).fetchall()

    if matches:
        for card in matches:
            pos = card['pitcher_role_name'] or card['position_name'] or '?'
            is_pitcher = card['pitcher_role_name'] in ('SP', 'RP', 'CL') if card['pitcher_role_name'] else False
            owned_str = " (OWNED)" if card['owned'] and card['owned'] > 0 else ""

            with st.expander(f"{card['card_title']} — OVR {card['card_value']} {card['tier_name']}{owned_str}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Card Ratings**")
                    if is_pitcher:
                        st.write(f"STU: {card['stuff']} | MOV: {card['movement']} | CTRL: {card['control']} | pHR: {card['p_hr']}")
                        st.write(f"Pitching Meta: {card['meta_score_pitching']:.0f}" if card['meta_score_pitching'] else "")
                    else:
                        st.write(f"CON: {card['contact']} | GAP: {card['gap_power']} | POW: {card['power']}")
                        st.write(f"EYE: {card['eye']} | AvK: {card['avoid_ks']} | BABIP: {card['babip']}")
                        st.write(f"Batting Meta: {card['meta_score_batting']:.0f}" if card['meta_score_batting'] else "")
                    st.write(f"Price: {card['last_10_price']:,} PP | Buy: {card['buy_order_high']:,} | Sell: {card['sell_order_low']:,}")

                with c2:
                    # Extract player name from card title
                    title_parts = card['card_title'].replace('MLB 2026 Live ', '').split()
                    # Remove position prefix and team suffix
                    if len(title_parts) >= 3:
                        player_name = ' '.join(title_parts[1:-1])  # skip POS and TEAM
                    else:
                        player_name = ' '.join(title_parts)

                    if st.button(f"Fetch MLB Stats", key=f"fetch_{card['card_id']}"):
                        from app.core.live_card_tracker import fetch_mlb_stats_for_player, estimate_rating_direction
                        mlb = fetch_mlb_stats_for_player(player_name, is_pitcher)
                        if mlb:
                            st.markdown("**2026 MLB Stats**")
                            if is_pitcher:
                                st.write(f"ERA: {mlb['era']:.2f} | WHIP: {mlb['whip']:.2f} | K/9: {mlb['k_per_9']:.1f}")
                                st.write(f"Record: {mlb['wins']}-{mlb['losses']} | IP: {mlb['ip']:.1f} | Saves: {mlb['saves']}")
                            else:
                                st.write(f"AVG: {mlb['avg']:.3f} | OBP: {mlb['obp']:.3f} | SLG: {mlb['slg']:.3f} | OPS: {mlb['ops']:.3f}")
                                st.write(f"HR: {mlb['hr']} | RBI: {mlb['rbi']} | SB: {mlb['sb']} | PA: {mlb['pa']}")

                            analysis = estimate_rating_direction(dict(card), mlb, is_pitcher)
                            signal_emoji = {"upgrade": "🟢⬆️", "downgrade": "🔴⬇️", "hold": "🟡➡️"}.get(analysis['signal'], "❓")
                            st.markdown(f"**Prediction: {signal_emoji} {analysis['signal'].upper()}** (confidence: {analysis['confidence']}, score: {analysis['score']:+d})")
                            for reason in analysis['reasons']:
                                st.write(f"  - {reason}")
                        else:
                            st.warning(f"Could not find MLB stats for {player_name}")
    else:
        st.info("No Live cards found matching your search.")

conn.close()
