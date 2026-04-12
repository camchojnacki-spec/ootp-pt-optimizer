"""Buy/sell recommendation engine — with live card upgrade signal integration."""
import sqlite3
from datetime import datetime
from app.core.database import get_connection, load_config


def generate_recommendations():
    """Main entry point — regenerate all recommendations."""
    conn = get_connection()

    # Archive old recommendations (mark dismissed)
    conn.execute("UPDATE recommendations SET dismissed = 1 WHERE dismissed = 0")
    conn.commit()

    config = load_config()
    budget = config.get('pp_budget', 500)
    rec_config = config.get('recommendations', {})
    min_meta_improvement = rec_config.get('min_meta_improvement', 10)
    max_budget_pct = rec_config.get('max_budget_pct', 0.80)
    max_spend = int(budget * max_budget_pct)

    # Generate buy recommendations
    _generate_buy_recs(conn, max_spend, min_meta_improvement)

    # Integrate live card upgrade signals into buy recs
    _boost_live_card_upgrades(conn)

    # Generate sell recommendations (categorized)
    _generate_sell_recs(conn)

    # Flag underperformers for sell
    _flag_underperformer_sells(conn)

    conn.commit()

    # Passive AI enrichment (non-blocking)
    try:
        from app.core.ai_advisor import generate_ai_insights
        generate_ai_insights(conn)
    except Exception:
        pass  # AI enrichment is optional — don't break recommendations

    # Auto-calibrate meta weights if enough data
    try:
        from app.core.meta_calibration import auto_calibrate_if_ready
        auto_calibrate_if_ready(conn)
    except Exception:
        pass  # Calibration is optional

    conn.close()


def _generate_buy_recs(conn, max_spend, min_meta_improvement):
    """Generate buy recommendations based on roster gaps and market value.

    Scans the FULL market (not limited by budget) so the Recommendations tab
    has useful data at any slider position. Budget filtering is done at display time.
    Priority is tiered: budget-friendly cards get higher priority.
    """
    cursor = conn.cursor()

    # Get current roster by position (active roster only)
    roster = cursor.execute("""
        SELECT player_name, position, meta_score, ovr
        FROM roster_current WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
    """).fetchall()

    roster_by_pos = {}
    for r in roster:
        pos = r['position']
        if pos not in roster_by_pos or r['meta_score'] > roster_by_pos[pos]['meta_score']:
            roster_by_pos[pos] = dict(r)

    # All field positions we expect filled
    # DH excluded — any batter can DH, no need for a dedicated DH card
    batting_positions = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
    pitching_positions = ['SP', 'RP', 'CL']

    # 1. Roster gap analysis — positions with no starter or weak starter
    for pos in batting_positions:
        current = roster_by_pos.get(pos)
        current_meta = current['meta_score'] if current else 0

        # Scan full market — no budget cap, higher per-position limit
        market_cards = cursor.execute("""
            SELECT card_id, card_title, position_name, meta_score_batting as meta_score,
                   last_10_price, sell_order_low, buy_order_high, tier_name, tier
            FROM cards
            WHERE position_name = ? AND owned = 0
                AND last_10_price > 0
                AND meta_score_batting > ?
            ORDER BY meta_score_batting DESC
        """, (pos, current_meta + min_meta_improvement)).fetchall()

        for card in market_cards:
            price = card['last_10_price'] or card['sell_order_low'] or 0
            if price <= 0:
                continue

            meta = card['meta_score']
            value_ratio = (meta * meta * 1.0 / price) if price > 0 else 0
            delta = meta - current_meta

            if not current:
                reason = f"Empty {pos} slot — {card['card_title']} fills gap"
                priority = 1
            elif price <= max_spend:
                reason = f"Upgrade {pos}: +{delta:.0f} meta over {current['player_name']}"
                priority = 2 if delta > 100 else 3
            else:
                reason = f"Upgrade {pos}: +{delta:.0f} meta over {current['player_name']} (save up)"
                priority = 3 if delta > 200 else 4

            cursor.execute("""
                INSERT INTO recommendations (rec_type, card_id, card_title, position, reason,
                    priority, estimated_price, meta_score, value_ratio, roster_impact)
                VALUES ('buy', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card['card_id'], card['card_title'], pos, reason,
                priority, price, meta, round(value_ratio, 2),
                f"Replace {current['player_name']}" if current else f"Fill {pos}"
            ))

    # Same for pitching positions
    for pos in pitching_positions:
        current = roster_by_pos.get(pos)
        current_meta = current['meta_score'] if current else 0

        market_cards = cursor.execute("""
            SELECT card_id, card_title, pitcher_role_name, meta_score_pitching as meta_score,
                   last_10_price, sell_order_low, buy_order_high, tier_name, tier
            FROM cards
            WHERE pitcher_role_name = ? AND owned = 0
                AND last_10_price > 0
                AND meta_score_pitching > ?
            ORDER BY meta_score_pitching DESC
        """, (pos, current_meta + min_meta_improvement)).fetchall()

        for card in market_cards:
            price = card['last_10_price'] or card['sell_order_low'] or 0
            if price <= 0:
                continue

            meta = card['meta_score']
            value_ratio = (meta * meta * 1.0 / price) if price > 0 else 0
            delta = meta - current_meta

            if not current:
                reason = f"Empty {pos} slot — {card['card_title']} fills gap"
                priority = 1
            elif price <= max_spend:
                reason = f"Upgrade {pos}: +{delta:.0f} meta over {current['player_name']}"
                priority = 2 if delta > 100 else 3
            else:
                reason = f"Upgrade {pos}: +{delta:.0f} meta over {current['player_name']} (save up)"
                priority = 3 if delta > 200 else 4

            cursor.execute("""
                INSERT INTO recommendations (rec_type, card_id, card_title, position, reason,
                    priority, estimated_price, meta_score, value_ratio, roster_impact)
                VALUES ('buy', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card['card_id'], card['card_title'], pos, reason,
                priority, price, meta, round(value_ratio, 2),
                f"Replace {current['player_name']}" if current else f"Fill {pos}"
            ))

    # 1b. Multi-position eligibility scan — find cards that can play secondary positions
    # Maps position abbreviations to their pos_rating column names
    pos_rating_map = {
        'C': 'pos_rating_c', '1B': 'pos_rating_1b', '2B': 'pos_rating_2b',
        '3B': 'pos_rating_3b', 'SS': 'pos_rating_ss',
        'LF': 'pos_rating_lf', 'CF': 'pos_rating_cf', 'RF': 'pos_rating_rf',
    }

    # Collect card_ids already recommended to avoid duplicates
    already_recommended = set()
    existing_recs = cursor.execute(
        "SELECT card_id, position FROM recommendations WHERE rec_type = 'buy' AND dismissed = 0"
    ).fetchall()
    for er in existing_recs:
        already_recommended.add((er['card_id'], er['position']))

    for pos in batting_positions:
        current = roster_by_pos.get(pos)
        current_meta = current['meta_score'] if current else 0
        rating_col = pos_rating_map[pos]

        # Find unowned batting cards with a decent rating at this secondary position
        # that are listed at a different primary position
        multi_pos_cards = cursor.execute(f"""
            SELECT card_id, card_title, position_name, meta_score_batting as meta_score,
                   last_10_price, sell_order_low, buy_order_high, tier_name, tier,
                   {rating_col} as sec_pos_rating
            FROM cards
            WHERE position_name != ? AND owned = 0
                AND last_10_price > 0
                AND meta_score_batting > ?
                AND {rating_col} >= 100
                AND pitcher_role_name IS NULL
            ORDER BY meta_score_batting DESC
            LIMIT 20
        """, (pos, current_meta + min_meta_improvement)).fetchall()

        for card in multi_pos_cards:
            # Skip if already recommended for this position
            if (card['card_id'], pos) in already_recommended:
                continue
            # Skip if already recommended for their primary position
            if (card['card_id'], card['position_name']) in already_recommended:
                continue

            price = card['last_10_price'] or card['sell_order_low'] or 0
            if price <= 0:
                continue

            meta = card['meta_score']
            value_ratio = (meta * meta * 1.0 / price) if price > 0 else 0
            delta = meta - current_meta
            sec_rating = card['sec_pos_rating']

            if not current:
                reason = f"Empty {pos} slot — {card['card_title']} (primary {card['position_name']}). Can also play {pos} (rating: {sec_rating})"
                priority = 2
            elif price <= max_spend:
                reason = f"Upgrade {pos}: +{delta:.0f} meta over {current['player_name']}. Can also play {pos} (rating: {sec_rating})"
                priority = 3
            else:
                reason = f"Upgrade {pos}: +{delta:.0f} meta over {current['player_name']} (save up). Can also play {pos} (rating: {sec_rating})"
                priority = 4

            cursor.execute("""
                INSERT INTO recommendations (rec_type, card_id, card_title, position, reason,
                    priority, estimated_price, meta_score, value_ratio, roster_impact)
                VALUES ('buy', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card['card_id'], card['card_title'], pos, reason,
                priority, price, meta, round(value_ratio, 2),
                f"Secondary pos: {card['position_name']} -> {pos}"
            ))
            already_recommended.add((card['card_id'], pos))

    # 2. Best value buys across all positions (no budget cap, min meta 200)
    value_buys = cursor.execute("""
        SELECT card_id, card_title, position_name, pitcher_role_name,
               COALESCE(meta_score_batting, meta_score_pitching) as meta_score,
               last_10_price, sell_order_low, tier_name
        FROM cards
        WHERE owned = 0 AND last_10_price > 0
            AND COALESCE(meta_score_batting, meta_score_pitching) > 200
        ORDER BY (COALESCE(meta_score_batting, meta_score_pitching) * 1.0 / NULLIF(last_10_price, 0)) DESC
        LIMIT 30
    """).fetchall()

    for card in value_buys:
        price = card['last_10_price'] or card['sell_order_low'] or 0
        if price <= 0:
            continue
        meta = card['meta_score']
        value_ratio = (meta / (price / 1000)) if price > 0 else 0
        pos = card['pitcher_role_name'] or card['position_name'] or '?'

        # Check if we already have a rec for this card
        existing = cursor.execute(
            "SELECT 1 FROM recommendations WHERE card_id = ? AND dismissed = 0", (card['card_id'],)
        ).fetchone()
        if existing:
            continue

        cursor.execute("""
            INSERT INTO recommendations (rec_type, card_id, card_title, position, reason,
                priority, estimated_price, meta_score, value_ratio, roster_impact)
            VALUES ('buy', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            card['card_id'], card['card_title'], pos,
            f"High value: {meta:.0f} meta for {price:,} PP ({card['tier_name']})",
            3 if price <= max_spend else 4, price, meta, round(value_ratio, 2), "Value buy"
        ))


def _boost_live_card_upgrades(conn):
    """Boost priority of buy recommendations for Live cards with upgrade signals.

    Checks if any recommended buy cards are Live cards, and if cached upgrade
    analysis exists, adjusts their priority and adds upgrade context to reason.
    """
    cursor = conn.cursor()

    # Find buy recs that are Live cards
    live_recs = cursor.execute("""
        SELECT r.id, r.card_id, r.card_title, r.reason, r.priority, r.value_ratio
        FROM recommendations r
        JOIN cards c ON r.card_id = c.card_id
        WHERE r.rec_type = 'buy' AND r.dismissed = 0
            AND c.card_title LIKE 'MLB 2026 Live%'
    """).fetchall()

    if not live_recs:
        return

    # Check if we have cached live card analysis
    # We store this in a simple table — create if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS live_card_cache (
            card_id INTEGER PRIMARY KEY,
            signal TEXT,
            confidence TEXT,
            score INTEGER,
            reasons TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    for rec in live_recs:
        cached = cursor.execute(
            "SELECT signal, score, reasons FROM live_card_cache WHERE card_id = ?",
            (rec['card_id'],)
        ).fetchone()

        if cached and cached['signal'] == 'upgrade':
            # Boost priority (lower number = higher priority)
            new_priority = max(1, rec['priority'] - 1)
            boost_pct = min(50, cached['score'])  # up to 50% value ratio boost
            new_value = rec['value_ratio'] * (1 + boost_pct / 100) if rec['value_ratio'] else 0
            upgrade_note = f" | 📈 MLB upgrade signal ({cached['confidence']} confidence)"

            cursor.execute("""
                UPDATE recommendations
                SET priority = ?, value_ratio = ?, reason = reason || ?
                WHERE id = ?
            """, (new_priority, round(new_value, 2), upgrade_note, rec['id']))

        elif cached and cached['signal'] == 'downgrade':
            # Deprioritize
            new_priority = min(4, rec['priority'] + 1)
            downgrade_note = f" | ⚠️ MLB downgrade risk ({cached['confidence']})"

            cursor.execute("""
                UPDATE recommendations
                SET priority = ?, reason = reason || ?
                WHERE id = ?
            """, (new_priority, downgrade_note, rec['id']))


def _generate_sell_recs(conn):
    """Generate categorized sell recommendations for owned cards."""
    cursor = conn.cursor()

    # Find owned cards that are not on the active roster
    owned_cards = cursor.execute("""
        SELECT c.card_id, c.card_title, c.position_name, c.pitcher_role_name,
               COALESCE(c.meta_score_batting, c.meta_score_pitching) as meta_score,
               c.last_10_price, c.sell_order_low, c.buy_order_high, c.tier_name, c.owned
        FROM cards c
        WHERE c.owned > 0 AND c.last_10_price > 0
    """).fetchall()

    # Get active roster with meta scores
    active_roster = cursor.execute(
        "SELECT player_name, position, meta_score FROM roster_current WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')"
    ).fetchall()
    active_names = {r['player_name'] for r in active_roster}

    # Best meta by position for outclass detection
    roster_meta_by_pos = {}
    for r in active_roster:
        pos = r['position']
        if pos not in roster_meta_by_pos or r['meta_score'] > roster_meta_by_pos[pos]:
            roster_meta_by_pos[pos] = r['meta_score']

    for card in owned_cards:
        pos = card['pitcher_role_name'] or card['position_name'] or '?'
        meta = card['meta_score'] or 0
        price = card['last_10_price'] or 0

        card_title = card['card_title'] or ''
        is_active = any(name in card_title for name in active_names)

        # Duplicate detection
        if card['owned'] > 1:
            reason = f"Own {card['owned']}x — sell extras for ~{price} PP each"
            cursor.execute("""
                INSERT INTO recommendations (rec_type, card_id, card_title, position, reason,
                    priority, estimated_price, meta_score, value_ratio, roster_impact)
                VALUES ('sell', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card['card_id'], card['card_title'], pos, reason,
                1, price, meta, 0, f"Sell {card['owned'] - 1} duplicate(s)"
            ))

        # Off-roster / outclassed detection
        if not is_active and price > 50:
            starter_meta = roster_meta_by_pos.get(pos, 0)

            if starter_meta > 0 and meta < starter_meta:
                # Outclassed at position
                gap = starter_meta - meta
                reason = f"Outclassed at {pos} by {gap:.0f} meta — sell for ~{price} PP"
                priority = 2
                impact = f"Your {pos} starter has {starter_meta:.0f} meta"
            else:
                reason = f"Not on active roster — sell for ~{price} PP"
                priority = 3
                impact = "Free up PP"

            cursor.execute("""
                INSERT INTO recommendations (rec_type, card_id, card_title, position, reason,
                    priority, estimated_price, meta_score, value_ratio, roster_impact)
                VALUES ('sell', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card['card_id'], card['card_title'], pos, reason,
                priority, price, meta, 0, impact
            ))


def _flag_underperformer_sells(conn):
    """Flag owned players whose in-game stats don't match their meta scores.

    Batters: meta_score > 600 but OPS < 0.650 with 50+ AB
    Pitchers: meta_score > 400 but ERA > 5.00 with 30+ IP
    """
    cursor = conn.cursor()

    try:
        # Underperforming batters
        bat_underperformers = cursor.execute("""
            SELECT r.player_name, r.position, r.meta_score,
                   bs.ops, bs.ab,
                   c.card_id, c.card_title, c.last_10_price
            FROM roster_current r
            INNER JOIN batting_stats bs ON bs.player_name = r.player_name
            INNER JOIN (
                SELECT player_name, MAX(snapshot_date) as max_date
                FROM batting_stats GROUP BY player_name
            ) latest ON bs.player_name = latest.player_name
                    AND bs.snapshot_date = latest.max_date
            LEFT JOIN cards c ON c.card_title LIKE '%' || r.player_name || '%' AND c.owned > 0
            WHERE r.meta_score > 600
                AND bs.ops < 0.650
                AND bs.ab >= 50
                AND r.lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
        """).fetchall()

        for row in bat_underperformers:
            price = row['last_10_price'] or 0
            card_id = row['card_id']
            card_title = row['card_title'] or row['player_name']
            meta = row['meta_score']
            ops = row['ops']

            # Skip if we already have a sell rec for this card
            if card_id:
                existing = cursor.execute(
                    "SELECT 1 FROM recommendations WHERE card_id = ? AND rec_type = 'sell' AND dismissed = 0",
                    (card_id,)
                ).fetchone()
                if existing:
                    continue

            reason = f"Underperformer: meta {meta:.0f} but .{int(ops * 1000):03d} OPS in-game. Consider selling for ~{price:,} PP"

            cursor.execute("""
                INSERT INTO recommendations (rec_type, card_id, card_title, position, reason,
                    priority, estimated_price, meta_score, value_ratio, roster_impact)
                VALUES ('sell', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card_id, card_title, row['position'], reason,
                2, price, meta, 0, "Underperforming in-game"
            ))

        # Underperforming pitchers
        pitch_underperformers = cursor.execute("""
            SELECT r.player_name, r.position, r.meta_score,
                   ps.era, ps.ip,
                   c.card_id, c.card_title, c.last_10_price
            FROM roster_current r
            INNER JOIN pitching_stats ps ON ps.player_name = r.player_name
            INNER JOIN (
                SELECT player_name, MAX(snapshot_date) as max_date
                FROM pitching_stats GROUP BY player_name
            ) latest ON ps.player_name = latest.player_name
                    AND ps.snapshot_date = latest.max_date
            LEFT JOIN cards c ON c.card_title LIKE '%' || r.player_name || '%' AND c.owned > 0
            WHERE r.meta_score > 400
                AND ps.era > 5.00
                AND ps.ip >= 30
                AND r.lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
        """).fetchall()

        for row in pitch_underperformers:
            price = row['last_10_price'] or 0
            card_id = row['card_id']
            card_title = row['card_title'] or row['player_name']
            meta = row['meta_score']
            era = row['era']

            if card_id:
                existing = cursor.execute(
                    "SELECT 1 FROM recommendations WHERE card_id = ? AND rec_type = 'sell' AND dismissed = 0",
                    (card_id,)
                ).fetchone()
                if existing:
                    continue

            reason = f"Underperformer: meta {meta:.0f} but {era:.2f} ERA in-game. Consider selling for ~{price:,} PP"

            cursor.execute("""
                INSERT INTO recommendations (rec_type, card_id, card_title, position, reason,
                    priority, estimated_price, meta_score, value_ratio, roster_impact)
                VALUES ('sell', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card_id, card_title, row['position'], reason,
                2, price, meta, 0, "Underperforming in-game"
            ))

    except Exception:
        pass  # Stats tables may not exist yet — this is optional


def cache_live_card_analysis(results: list):
    """Cache live card analysis results for use by the recommendation engine.

    Called from the Live Card Tracker page after running analysis.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS live_card_cache (
            card_id INTEGER PRIMARY KEY,
            signal TEXT,
            confidence TEXT,
            score INTEGER,
            reasons TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    for result in results:
        card = result.get('card', {})
        analysis = result.get('analysis', {})
        card_id = card.get('card_id')
        if not card_id:
            continue

        reasons_str = '; '.join(analysis.get('reasons', []))
        cursor.execute("""
            INSERT INTO live_card_cache (card_id, signal, confidence, score, reasons, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(card_id) DO UPDATE SET
                signal = excluded.signal,
                confidence = excluded.confidence,
                score = excluded.score,
                reasons = excluded.reasons,
                updated_at = CURRENT_TIMESTAMP
        """, (card_id, analysis.get('signal', 'hold'), analysis.get('confidence', 'low'),
              analysis.get('score', 0), reasons_str))

    conn.commit()
    conn.close()


def get_buy_recommendations(conn=None, limit=50, position=None, max_price=None, min_tier=None):
    """Fetch current buy recommendations with optional filters."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    query = """
        SELECT r.*, c.tier_name, c.tier, c.position_name, c.pitcher_role_name
        FROM recommendations r
        LEFT JOIN cards c ON r.card_id = c.card_id
        WHERE r.rec_type = 'buy' AND r.dismissed = 0
    """
    params = []

    if position:
        query += " AND r.position = ?"
        params.append(position)
    if max_price:
        query += " AND r.estimated_price <= ?"
        params.append(max_price)
    if min_tier:
        query += " AND c.tier >= ?"
        params.append(min_tier)

    query += " ORDER BY r.priority ASC, r.value_ratio DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    if close_conn:
        conn.close()
    return rows


def get_sell_recommendations(conn=None, limit=50):
    """Fetch current sell recommendations."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    rows = conn.execute("""
        SELECT r.*, c.tier_name, c.tier
        FROM recommendations r
        LEFT JOIN cards c ON r.card_id = c.card_id
        WHERE r.rec_type = 'sell' AND r.dismissed = 0
        ORDER BY r.priority ASC, r.estimated_price DESC
        LIMIT ?
    """, (limit,)).fetchall()

    if close_conn:
        conn.close()
    return rows
