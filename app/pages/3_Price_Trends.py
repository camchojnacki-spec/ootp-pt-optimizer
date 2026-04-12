"""Price Trends page."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection
from app.core.price_analysis import (
    get_price_history, get_biggest_movers, get_price_stats,
    get_price_momentum, get_market_momentum_summary,
)

st.set_page_config(page_title="Price Trends", page_icon="📈", layout="wide")
st.title("Price Trends")

conn = get_connection()

# --- Market Momentum Section ---
st.subheader("Market Momentum")
summary = get_market_momentum_summary(conn)

col_r, col_f, col_b = st.columns(3)
col_r.metric("Rising", summary['rising_count'])
col_f.metric("Falling", summary['falling_count'])
col_b.metric("Buy Signals", len(summary['buy_signals']))

DISPLAY_COLS = ['Card', 'Pos', 'Tier', 'Price', 'Momentum', 'Direction', 'Signal']

tab_buy, tab_sell, tab_vol = st.tabs(["Buy Low Signals", "Sell High Signals", "Most Volatile"])

with tab_buy:
    if summary['buy_signals']:
        st.dataframe(
            pd.DataFrame(summary['buy_signals'])[DISPLAY_COLS],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No buy-low signals detected. Need more price snapshots or price movement.")

with tab_sell:
    if summary['sell_signals']:
        st.dataframe(
            pd.DataFrame(summary['sell_signals'])[DISPLAY_COLS],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No sell-high signals detected. Need more price snapshots or price movement.")

with tab_vol:
    if summary['most_volatile']:
        vol_cols = ['Card', 'Pos', 'Tier', 'Price', 'Volatility', 'Momentum', 'Direction', 'Signal']
        st.dataframe(
            pd.DataFrame(summary['most_volatile'])[vol_cols],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No volatility data yet. Need more price snapshots.")

st.divider()

# Card search
search = st.text_input("Search cards by name")
if search:
    cards = conn.execute("""
        SELECT card_id, card_title, position_name, pitcher_role_name, tier_name,
               last_10_price, buy_order_high, sell_order_low,
               COALESCE(meta_score_batting, meta_score_pitching) as meta_score
        FROM cards
        WHERE card_title LIKE ? AND last_10_price > 0
        ORDER BY last_10_price DESC
        LIMIT 50
    """, (f"%{search}%",)).fetchall()

    if cards:
        card_options = {f"{c['card_title']} ({c['tier_name']}, {c['last_10_price']:,} PP)": c['card_id'] for c in cards}
        selected = st.selectbox("Select card", list(card_options.keys()))
        card_id = card_options[selected]

        # Price chart
        history = get_price_history(card_id, conn)
        if history:
            fig = go.Figure()
            dates = [h['snapshot_date'] for h in history]
            fig.add_trace(go.Scatter(x=dates, y=[h['last_10_price'] for h in history],
                                      mode='lines+markers', name='Last 10 Price'))
            fig.add_trace(go.Scatter(x=dates, y=[h['sell_order_low'] for h in history],
                                      mode='lines', name='Sell Low', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=dates, y=[h['buy_order_high'] for h in history],
                                      mode='lines', name='Buy High', line=dict(dash='dot')))
            fig.update_layout(title=f"Price History", xaxis_title="Date", yaxis_title="PP",
                              height=400)
            st.plotly_chart(fig, use_container_width=True)

        stats = get_price_stats(card_id, conn)
        if stats and stats['snapshot_count']:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Min Price", f"{stats['min_price']:,}")
            col2.metric("Max Price", f"{stats['max_price']:,}")
            col3.metric("Avg Price", f"{stats['avg_price']:,.0f}")
            col4.metric("Snapshots", stats['snapshot_count'])

        # Momentum indicators for selected card
        momentum = get_price_momentum(card_id, conn)
        if momentum:
            st.markdown("**Momentum Indicators**")
            mc1, mc2, mc3, mc4 = st.columns(4)
            direction_icon = {"rising": "+", "falling": "-", "stable": "~"}.get(momentum['direction'], '')
            mc1.metric("Direction", f"{direction_icon} {momentum['direction'].title()}")
            mc2.metric("Momentum", f"{momentum['momentum_score']:+.1f}")
            mc3.metric("Volatility", f"{momentum['volatility_score']:.1f}")
            signal_map = {"buy_low": "BUY LOW", "sell_high": "SELL HIGH", "hold": "HOLD", "watch": "WATCH"}
            mc4.metric("Signal", signal_map.get(momentum['signal'], momentum['signal']))

            ma1, ma2, ma3 = st.columns(3)
            ma1.metric("3-Day Avg", f"{momentum['avg_3day']:,}")
            ma2.metric("7-Day Avg", f"{momentum['avg_7day']:,}")
            ma3.metric("14-Day Avg", f"{momentum['avg_14day']:,}")
    else:
        st.info("No cards found matching your search.")

st.divider()

# Biggest movers
st.subheader("Biggest Movers")
movers = get_biggest_movers(days=7, limit=20, conn=conn)
if movers:
    mover_data = []
    for m in movers:
        pos = m['pitcher_role_name'] or m['position_name'] or ''
        mover_data.append({
            "Card": m['card_title'],
            "Pos": pos,
            "Tier": m['tier_name'],
            "Old Price": m['old_price'],
            "Current": m['current_price'],
            "Change": m['price_change'],
            "% Change": f"{m['pct_change']}%",
        })
    st.dataframe(pd.DataFrame(mover_data), use_container_width=True, hide_index=True)
else:
    st.info("Need multiple price snapshots to show movers. Import market data over multiple sessions.")

conn.close()
