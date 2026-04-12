"""Sparkline utilities for inline price trend display."""
import plotly.graph_objects as go


def get_price_history(card_id, conn, days=14):
    """Query last N days of last_10_price from price_snapshots."""
    rows = conn.execute("""
        SELECT snapshot_date, last_10_price
        FROM price_snapshots
        WHERE card_id = ? AND last_10_price > 0
        ORDER BY snapshot_date DESC
        LIMIT ?
    """, (card_id, days)).fetchall()
    if len(rows) < 2:
        return None
    # Return in chronological order
    return list(reversed([(r['snapshot_date'], r['last_10_price']) for r in rows]))


def make_sparkline_figure(card_id, conn, days=14):
    """Create a tiny Plotly sparkline figure for a card's price trend.

    Returns a plotly Figure (80x25px, no axes/labels) or None if insufficient data.
    """
    history = get_price_history(card_id, conn, days)
    if history is None:
        return None

    dates, prices = zip(*history)

    fig = go.Figure(
        go.Scatter(
            x=list(range(len(prices))),
            y=list(prices),
            mode='lines',
            line=dict(width=1.5, color='#1f77b4'),
            hoverinfo='skip',
        )
    )
    fig.update_layout(
        width=80,
        height=25,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def text_sparkline(card_id, conn, days=14):
    """Generate a unicode text sparkline using block characters.

    Returns a string like '▁▃▅▇▆▄▂' or '' if insufficient data.
    """
    history = get_price_history(card_id, conn, days)
    if history is None:
        return ''

    prices = [p for _, p in history]
    lo = min(prices)
    hi = max(prices)

    blocks = '▁▂▃▄▅▆▇█'

    if hi == lo:
        return blocks[4] * len(prices)

    spark = ''
    for p in prices:
        idx = int((p - lo) / (hi - lo) * 7)
        idx = min(idx, 7)
        spark += blocks[idx]
    return spark


def add_trend_column(rows, conn, card_id_key='card_id'):
    """Given a list of row dicts, add a 'Trend' sparkline string for each.

    Expects each row dict to have a key matching card_id_key.
    Returns a dict mapping card_id -> sparkline string (for batch use).
    """
    cache = {}
    for r in rows:
        cid = r.get(card_id_key) if isinstance(r, dict) else r[card_id_key]
        if cid and cid not in cache:
            cache[cid] = text_sparkline(cid, conn)
    return cache
