"""Price trend analysis and signals."""
import statistics
from app.core.database import get_connection


def get_price_history(card_id: int, conn=None):
    """Get price history for a card."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    rows = conn.execute("""
        SELECT snapshot_date, buy_order_high, sell_order_low, last_10_price, last_10_variance
        FROM price_snapshots
        WHERE card_id = ?
        ORDER BY snapshot_date ASC
    """, (card_id,)).fetchall()

    if close_conn:
        conn.close()
    return rows


def get_biggest_movers(days=7, limit=20, conn=None):
    """Find cards with the biggest price changes over the given period."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    rows = conn.execute("""
        WITH latest AS (
            SELECT card_id, last_10_price as current_price,
                   ROW_NUMBER() OVER (PARTITION BY card_id ORDER BY snapshot_date DESC) as rn
            FROM price_snapshots
        ),
        older AS (
            SELECT card_id, last_10_price as old_price,
                   ROW_NUMBER() OVER (PARTITION BY card_id ORDER BY snapshot_date ASC) as rn
            FROM price_snapshots
            WHERE snapshot_date >= datetime('now', ?)
        )
        SELECT l.card_id, c.card_title, c.position_name, c.pitcher_role_name, c.tier_name,
               o.old_price, l.current_price,
               (l.current_price - o.old_price) as price_change,
               CASE WHEN o.old_price > 0
                    THEN ROUND((l.current_price - o.old_price) * 100.0 / o.old_price, 1)
                    ELSE 0 END as pct_change
        FROM latest l
        JOIN older o ON l.card_id = o.card_id AND o.rn = 1
        JOIN cards c ON l.card_id = c.card_id
        WHERE l.rn = 1 AND o.old_price > 0 AND l.current_price > 0
        ORDER BY ABS(l.current_price - o.old_price) DESC
        LIMIT ?
    """, (f'-{days} days', limit)).fetchall()

    if close_conn:
        conn.close()
    return rows


def get_price_stats(card_id: int, conn=None):
    """Get price statistics for a card."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    row = conn.execute("""
        SELECT
            COUNT(*) as snapshot_count,
            MIN(last_10_price) as min_price,
            MAX(last_10_price) as max_price,
            AVG(last_10_price) as avg_price,
            MIN(snapshot_date) as first_seen,
            MAX(snapshot_date) as last_seen
        FROM price_snapshots
        WHERE card_id = ? AND last_10_price > 0
    """, (card_id,)).fetchone()

    if close_conn:
        conn.close()
    return row


def get_price_momentum(card_id, conn=None):
    """Calculate momentum indicators for a card.

    Returns dict with:
    - direction: 'rising', 'falling', 'stable'
    - momentum_score: -100 to +100 (negative=falling, positive=rising)
    - volatility_score: 0-100
    - signal: 'buy_low', 'sell_high', 'hold', 'watch'
    - avg_3day, avg_7day, avg_14day: moving averages
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    rows = conn.execute("""
        SELECT snapshot_date, last_10_price
        FROM price_snapshots
        WHERE card_id = ? AND last_10_price > 0
        ORDER BY snapshot_date ASC
    """, (card_id,)).fetchall()

    if close_conn:
        conn.close()

    if len(rows) < 3:
        return None

    prices = [r['last_10_price'] for r in rows]

    # Moving averages (use available data if fewer than window size)
    avg_3day = statistics.mean(prices[-3:])
    avg_7day = statistics.mean(prices[-7:]) if len(prices) >= 7 else statistics.mean(prices)
    avg_14day = statistics.mean(prices[-14:]) if len(prices) >= 14 else statistics.mean(prices)

    # Momentum score: (short_avg - long_avg) / long_avg * 100, clamped
    if avg_14day > 0:
        momentum_raw = (avg_3day - avg_14day) / avg_14day * 100
    else:
        momentum_raw = 0.0
    momentum_score = max(-100, min(100, momentum_raw))

    # Direction
    if momentum_score > 5:
        direction = 'rising'
    elif momentum_score < -5:
        direction = 'falling'
    else:
        direction = 'stable'

    # Volatility: stdev of last 7 / avg of last 7 * 100
    recent_prices = prices[-7:] if len(prices) >= 7 else prices
    if len(recent_prices) >= 2 and avg_7day > 0:
        volatility_score = min(100, statistics.stdev(recent_prices) / avg_7day * 100)
    else:
        volatility_score = 0.0

    # Signal logic
    if direction == 'falling' and volatility_score < 20:
        signal = 'buy_low'
    elif direction == 'rising' and momentum_score > 15:
        signal = 'sell_high'
    elif volatility_score > 40:
        signal = 'watch'
    else:
        signal = 'hold'

    return {
        'direction': direction,
        'momentum_score': round(momentum_score, 1),
        'volatility_score': round(volatility_score, 1),
        'signal': signal,
        'avg_3day': round(avg_3day),
        'avg_7day': round(avg_7day),
        'avg_14day': round(avg_14day),
    }


def get_market_momentum_summary(conn=None):
    """Get momentum signals across all tracked cards.

    Returns dict with:
    - rising_count, falling_count, stable_count
    - buy_signals: list of top 10 buy-low signal cards
    - sell_signals: list of top 10 sell-high signal cards
    - most_volatile: top 10 by volatility_score
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    # Get all cards with 3+ snapshots
    card_rows = conn.execute("""
        SELECT ps.card_id, c.card_title, c.position_name, c.pitcher_role_name,
               c.tier_name, c.last_10_price
        FROM price_snapshots ps
        JOIN cards c ON ps.card_id = c.card_id
        WHERE ps.last_10_price > 0
        GROUP BY ps.card_id
        HAVING COUNT(*) >= 3
    """).fetchall()

    rising_count = 0
    falling_count = 0
    stable_count = 0
    buy_signals = []
    sell_signals = []
    all_entries = []

    for card in card_rows:
        momentum = get_price_momentum(card['card_id'], conn)
        if momentum is None:
            continue

        pos = card['pitcher_role_name'] or card['position_name'] or ''
        entry = {
            'card_id': card['card_id'],
            'Card': card['card_title'],
            'Pos': pos,
            'Tier': card['tier_name'],
            'Price': card['last_10_price'] or 0,
            'Momentum': momentum['momentum_score'],
            'Direction': momentum['direction'],
            'Signal': momentum['signal'],
            'Volatility': momentum['volatility_score'],
        }

        if momentum['direction'] == 'rising':
            rising_count += 1
        elif momentum['direction'] == 'falling':
            falling_count += 1
        else:
            stable_count += 1

        if momentum['signal'] == 'buy_low':
            buy_signals.append(entry)
        elif momentum['signal'] == 'sell_high':
            sell_signals.append(entry)

        all_entries.append(entry)

    if close_conn:
        conn.close()

    # Sort buy signals by momentum (most negative = deepest dip)
    buy_signals.sort(key=lambda x: x['Momentum'])
    # Sort sell signals by momentum (most positive = biggest spike)
    sell_signals.sort(key=lambda x: x['Momentum'], reverse=True)
    # Sort volatile by volatility
    most_volatile = sorted(all_entries, key=lambda x: x['Volatility'], reverse=True)

    return {
        'rising_count': rising_count,
        'falling_count': falling_count,
        'stable_count': stable_count,
        'buy_signals': buy_signals[:10],
        'sell_signals': sell_signals[:10],
        'most_volatile': most_volatile[:10],
    }
