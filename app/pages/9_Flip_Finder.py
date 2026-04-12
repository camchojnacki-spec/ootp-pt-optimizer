"""Flip Finder — find cards to buy and resell for PP profit."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection
from app.core.flip_finder import (
    find_spread_flips,
    find_volatility_flips,
    find_trend_flips,
    find_live_card_flips,
    find_matchup_flips,
    find_hot_streak_flips,
    get_flip_summary,
)

st.set_page_config(page_title="Flip Finder", page_icon="🔄", layout="wide")
st.title("Flip Finder")
st.caption("Find cards to buy low and sell high for PP profit")

conn = get_connection()

# Summary metrics
summary = get_flip_summary(conn=conn)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Spread Opportunities", summary['spread_flip_count'])
with col2:
    best = summary.get('best_spread_flip')
    st.metric("Best Spread Profit", f"{best['profit']:,} PP" if best else "—")
with col3:
    st.metric("Volatility Plays", summary['volatility_play_count'])
with col4:
    tc = summary['trend_play_count']
    st.metric("Trend Plays", tc if tc > 0 else "Need more snapshots")

st.divider()

# Tabs
tab_spread, tab_vol, tab_trend, tab_live, tab_matchup, tab_hot = st.tabs([
    "Spread Flips", "Volatility Plays", "Trend Plays", "Live Card Plays",
    "Matchup Plays", "Hot Streaks"
])

# ── Spread Flips ──
with tab_spread:
    sc1, sc2 = st.columns(2)
    with sc1:
        min_profit = st.slider("Min Profit (PP)", 0, 500, 20, step=10, key="sp_profit")
    with sc2:
        min_margin = st.slider("Min Margin %", 0, 50, 10, step=5, key="sp_margin")

    spreads = find_spread_flips(min_profit=min_profit, min_margin_pct=min_margin, conn=conn)

    if spreads:
        risk_emoji = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}
        spread_data = []
        for s in spreads:
            spread_data.append({
                "Card": s['card_title'],
                "Pos": s['position'] or '',
                "Tier": s['tier_name'] or '',
                "Buy At": s['buy_at'],
                "Sell At": s['sell_at'],
                "Profit": s['profit'],
                "Margin%": s['margin_pct'],
                "Score": s['flip_score'],
                "Risk": risk_emoji.get(s['risk_level'], s['risk_level']),
            })

        total_profit = sum(d['Profit'] for d in spread_data)
        st.markdown(f"**{len(spread_data)} opportunities — total potential profit: {total_profit:,} PP**")
        st.dataframe(pd.DataFrame(spread_data), use_container_width=True, hide_index=True)
    else:
        st.info("No spread flip opportunities found. Try lowering the thresholds.")

# ── Volatility Plays ──
with tab_vol:
    min_var = st.slider("Min Variance Ratio", 0.05, 0.50, 0.15, step=0.05, key="vol_var")

    vols = find_volatility_flips(min_variance_ratio=min_var, conn=conn)

    if vols:
        vol_data = []
        for v in vols:
            vol_data.append({
                "Card": v['card_title'],
                "Pos": v['position'] or '',
                "Tier": v['tier_name'] or '',
                "Avg Price": v['last_10_price'],
                "Variance": v['variance'],
                "Var%": f"{v['variance_ratio']:.1%}",
                "Est. Low": max(v['estimated_low'], 0),
                "Est. High": v['estimated_high'],
                "Potential Profit": v['potential_profit'],
            })

        st.markdown(f"**{len(vol_data)} high-volatility cards**")
        st.dataframe(pd.DataFrame(vol_data), use_container_width=True, hide_index=True)
        st.caption("These cards have wide price swings. Set buy orders near Est. Low and sell orders near Est. High.")
    else:
        st.info("No volatility plays found with current filter.")

# ── Trend Plays ──
with tab_trend:
    trends = find_trend_flips(conn=conn)

    if trends:
        trend_data = []
        for t in trends:
            trend_data.append({
                "Card": t['card_title'],
                "Pos": t['position'] or '',
                "Tier": t['tier_name'] or '',
                "Current Price": t['current_price'],
                "Historical Avg": t['avg_historical_price'],
                "Drop%": f"{t['price_drop_pct']:.1f}%",
                "Recovery Target": t['recovery_target'],
                "Potential Profit": t['potential_profit'],
                "Snapshots": t['snapshot_count'],
            })

        st.markdown(f"**{len(trend_data)} cards trading below historical average**")
        st.dataframe(pd.DataFrame(trend_data), use_container_width=True, hide_index=True)
    else:
        st.info("Need multiple market imports over time to detect price trends. Keep importing regularly!")

# ── Live Card Plays ──
with tab_live:
    lives = find_live_card_flips(conn=conn)

    if lives:
        live_data = []
        for l in lives:
            live_data.append({
                "Card": l['card_title'],
                "Pos": l['position'] or '',
                "Tier": l['tier_name'] or '',
                "Price": l['current_price'],
                "Upgrade Score": l['upgrade_score'],
                "Confidence": l['confidence'].title(),
                "Meta": l['meta_score'],
                "Reasons": l['reasons'],
            })

        st.markdown(f"**{len(live_data)} Live cards with upgrade signals**")
        st.dataframe(pd.DataFrame(live_data), use_container_width=True, hide_index=True)
        st.caption("Buy these before OOTP updates ratings. Run Live Card Tracker analysis first to populate this data.")
    else:
        st.info("No live card flip data. Go to **Live Card Tracker** and run analysis first — results will appear here.")

# ── Matchup Plays ──
with tab_matchup:
    st.caption("Live cards with favorable upcoming MLB matchups — more games = more chances to put up stats and trigger upgrades.")
    days = st.slider("Days ahead", 3, 14, 7, step=1, key="matchup_days")

    if st.button("Scan Matchups", key="scan_matchups"):
        with st.spinner("Fetching MLB schedule..."):
            matchups = find_matchup_flips(days_ahead=days, conn=conn)

        if matchups:
            match_data = []
            for m in matchups:
                match_data.append({
                    "Card": m['card_title'],
                    "Pos": m['position'] or '',
                    "Tier": m['tier_name'] or '',
                    "Price": m['price'],
                    "Team": m['team'],
                    "Games": m['games_in_window'],
                    "Home": m['home_games'],
                    "Score": round(m['matchup_score'], 1),
                    "Reasoning": m['flip_reasoning'],
                })

            st.markdown(f"**{len(match_data)} cards with upcoming matchup advantage**")
            st.dataframe(pd.DataFrame(match_data), use_container_width=True, hide_index=True)
        else:
            st.info("No matchup flip data available. This requires the MLB Stats API to fetch upcoming schedules.")
    else:
        st.info("Click 'Scan Matchups' to fetch the upcoming MLB schedule and find flip targets.")

# ── Hot Streaks ──
with tab_hot:
    st.caption("Live cards where the player is hot but the market hasn't caught up yet. Buy before the price spikes.")

    hot = find_hot_streak_flips(conn=conn)

    if hot:
        hot_data = []
        for h in hot:
            hot_data.append({
                "Card": h['card_title'],
                "Pos": h['position'] or '',
                "Tier": h['tier_name'] or '',
                "Price": h['price'],
                "Upgrade Score": h['upgrade_score'],
                "Confidence": h['confidence'].title(),
                "Price Lag": f"{h['price_lag_ratio']:.2f}x",
                "Reasons": h['reasons'],
            })

        st.markdown(f"**{len(hot_data)} hot streak cards — market hasn't priced in yet**")
        st.dataframe(pd.DataFrame(hot_data), use_container_width=True, hide_index=True)
        st.caption("Price Lag < 1.2x means the buy orders haven't spiked relative to recent sales — market is slow to react.")
    else:
        st.info("No hot streak flips found. Run **Live Card Tracker** analysis first to populate upgrade signals.")

conn.close()
