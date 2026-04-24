"""Buy/sell recommendation engine — with live card upgrade signal integration."""
import sqlite3
from datetime import datetime
from app.core.database import get_connection, load_config
from app.core.tier_context import tier_budget_ceiling, get_active_league_tier
from app.core.position_eligibility import (
    BATTING_POSITIONS,
    POS_RATING_COL,
    build_eligible_where_clause,
    select_rating_columns,
    position_meta_penalty,
    format_position_annotation,
)


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

    # Tier-aware per-card ceiling: don't recommend 50k PP diamonds to a
    # Low Bronze team. The user's PP budget caps the upper bound, but the
    # tier ceiling caps it further when the user is sitting on a large PP
    # pile (which otherwise tempts the engine to surface expensive cards).
    league_tier = get_active_league_tier(conn)
    tier_ceiling = tier_budget_ceiling(league_tier)
    tier_max_spend = min(max_spend, tier_ceiling) if tier_ceiling else max_spend

    # Generate buy recommendations
    _generate_buy_recs(conn, tier_max_spend, min_meta_improvement)

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

    Pitching now tracks ALL occupants per role (SP1–SP5, the bullpen, CL) and
    benchmarks against the WEAKEST slot, not the best. Previously an ace-
    quality market card looked unneeded because SP1 already held that slot,
    even though SP5 was bleeding runs.
    """
    cursor = conn.cursor()

    # Get current roster by position (active roster only)
    roster = cursor.execute("""
        SELECT player_name, position, meta_score, ovr
        FROM roster_current WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
    """).fetchall()

    # Batting: one starter per position → track the best (for replacement comparisons).
    # Pitching: multiple slots per role → track the full sorted list so we can
    # suggest upgrades to the worst slot, which is where real improvement lives.
    roster_by_pos = {}
    pitchers_by_role: dict[str, list[dict]] = {'SP': [], 'RP': [], 'CL': []}
    for r in roster:
        pos = r['position']
        r_dict = dict(r)
        if pos in pitchers_by_role:
            pitchers_by_role[pos].append(r_dict)
        else:
            if pos not in roster_by_pos or r_dict['meta_score'] > roster_by_pos[pos]['meta_score']:
                roster_by_pos[pos] = r_dict
    # Sort each pitching role ascending (weakest first) so [0] is the upgrade target.
    for role in pitchers_by_role:
        pitchers_by_role[role].sort(key=lambda p: p.get('meta_score') or 0)

    # All field positions we expect filled.
    # DH excluded — any batter can DH, no need for a dedicated DH card.
    # Uses the shared BATTING_POSITIONS constant so every rec engine agrees
    # on the set of slots it optimizes for.
    batting_positions = list(BATTING_POSITIONS)
    # Expected slot counts per pitching role — matches PT 40-man / active
    # pitching staff conventions. RP count is approximate; real staffs range
    # 6–8 but 7 is the normative target. CL is always a single slot.
    pitching_slots = {'SP': 5, 'RP': 7, 'CL': 1}

    # Track (card_id, position) pairs we've already written so the multi-position
    # loop doesn't emit duplicate recs (a card eligible at CF, LF, and RF should
    # surface at most once per slot, and we don't want two LF recs for the same
    # card if the query fires twice).
    already_recommended: set[tuple[int, str]] = set()

    # Pre-build the SELECT fragment for pos_rating_* columns — we need the
    # ratings in-hand so we can compute per-position penalties without a
    # second query per card.
    rating_cols_sql = select_rating_columns()

    # 1. Roster gap analysis — positions with no starter or weak starter.
    # Each position scan considers cards whose PRIMARY position matches OR
    # whose pos_rating_{slot} meets ELIGIBILITY_THRESHOLD. A meta penalty is
    # applied for non-primary assignments so low-rating corner-OF cards
    # don't silently beat true specialists on raw meta alone.
    for pos in batting_positions:
        current = roster_by_pos.get(pos)
        current_meta = current['meta_score'] if current else 0

        where_frag, where_params = build_eligible_where_clause(pos)

        # Scan full market — no budget cap, higher per-position limit.
        # We fetch pos_rating columns too so ``position_meta_penalty`` can
        # read the defensive rating for the target slot without a re-query.
        market_cards = cursor.execute(f"""
            SELECT card_id, card_title, position_name, meta_score_batting as raw_meta,
                   last_10_price, sell_order_low, buy_order_high, tier_name, tier,
                   {rating_cols_sql}
            FROM cards
            WHERE {where_frag} AND owned = 0
                AND last_10_price > 0
                AND pitcher_role IS NULL
                AND meta_score_batting > ?
            ORDER BY meta_score_batting DESC
        """, (*where_params, current_meta)).fetchall()

        for card in market_cards:
            card_dict = dict(card)
            card_id = card_dict['card_id']
            # Guard against double-insertion if the same card qualifies at
            # multiple target positions via different code paths.
            if (card_id, pos) in already_recommended:
                continue

            price = card_dict['last_10_price'] or card_dict['sell_order_low'] or 0
            if price <= 0:
                continue

            # Effective meta at THIS slot = raw meta minus defensive penalty.
            # A primary-CF card at RF (rating 67) pays 0; the same card at LF
            # (rating 32) pays ~7 — enough to keep ranking honest, not enough
            # to hide a genuine upgrade.
            raw_meta = card_dict['raw_meta'] or 0
            penalty = position_meta_penalty(card_dict, pos)
            meta = raw_meta - penalty
            # Drop cards that fall below the min-improvement bar AFTER
            # penalty. The SQL filter only screened raw meta; the penalty
            # can push a borderline card back under the threshold.
            if meta <= current_meta + min_meta_improvement:
                continue

            value_ratio = (meta * meta * 1.0 / price) if price > 0 else 0
            delta = meta - current_meta
            annotation = format_position_annotation(card_dict, pos)

            if not current:
                reason = f"Empty {pos} slot — {card_dict['card_title']} fills gap{annotation}"
                priority = 1
            elif price <= max_spend:
                reason = f"Upgrade {pos}: +{delta:.0f} meta over {current['player_name']}{annotation}"
                priority = 2 if delta > 100 else 3
            else:
                # Aspirational: exceeds current budget but clearly better.
                # Kept at Priority 3/4 (not hidden) so the user sees targets
                # worth saving for; the display layer is responsible for
                # tagging these as "save up".
                reason = f"Upgrade {pos}: +{delta:.0f} meta over {current['player_name']} (save up){annotation}"
                priority = 3 if delta > 200 else 4

            cursor.execute("""
                INSERT INTO recommendations (rec_type, card_id, card_title, position, reason,
                    priority, estimated_price, meta_score, value_ratio, roster_impact)
                VALUES ('buy', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card_id, card_dict['card_title'], pos, reason,
                priority, price, round(meta, 1), round(value_ratio, 2),
                f"Replace {current['player_name']}" if current else f"Fill {pos}"
            ))
            already_recommended.add((card_id, pos))

    # Pitching — benchmark against the WEAKEST slot per role, not the best.
    # If you have SP1=800 meta and SP5=520 meta, a 600-meta SP is a real
    # upgrade (replaces SP5) even though it doesn't beat SP1.
    for pos, expected_slots in pitching_slots.items():
        occupants = pitchers_by_role[pos]
        # Open slots matter too: SP staff of 4 with 5 expected = one empty slot.
        open_slots = max(0, expected_slots - len(occupants))
        # Benchmark meta: weakest occupant, or 0 if any slot is empty.
        weakest = occupants[0] if occupants else None
        if open_slots > 0:
            benchmark_meta = 0
            benchmark_name = None
        else:
            benchmark_meta = (weakest['meta_score'] if weakest else 0) or 0
            benchmark_name = weakest['player_name'] if weakest else None

        market_cards = cursor.execute("""
            SELECT card_id, card_title, pitcher_role_name, meta_score_pitching as meta_score,
                   last_10_price, sell_order_low, buy_order_high, tier_name, tier
            FROM cards
            WHERE pitcher_role_name = ? AND owned = 0
                AND last_10_price > 0
                AND meta_score_pitching > ?
            ORDER BY meta_score_pitching DESC
        """, (pos, benchmark_meta + min_meta_improvement)).fetchall()

        for card in market_cards:
            price = card['last_10_price'] or card['sell_order_low'] or 0
            if price <= 0:
                continue

            meta = card['meta_score']
            value_ratio = (meta * meta * 1.0 / price) if price > 0 else 0
            delta = meta - benchmark_meta

            # Mirror the comparison and priority logic used above for batters.
            # Re-bind ``current`` and ``current_meta`` so the downstream reason/
            # impact strings read identically to the batting branch.
            current = weakest if open_slots == 0 else None
            current_meta = benchmark_meta

            # Weakest-slot label: SP5 when the rotation is full, SP3 when two
            # slots are open and this card would be the 3rd body, etc.
            weakest_slot_num = expected_slots if open_slots == 0 else (expected_slots - open_slots + 1)
            slot_label = f"{pos}{weakest_slot_num}" if expected_slots > 1 else pos

            if not current:
                reason = f"Empty {slot_label} slot — {card['card_title']} fills gap"
                priority = 1
            elif price <= max_spend:
                reason = f"Upgrade {slot_label}: +{delta:.0f} meta over {current['player_name']}"
                priority = 2 if delta > 100 else 3
            else:
                reason = f"Upgrade {slot_label}: +{delta:.0f} meta over {current['player_name']} (save up)"
                priority = 3 if delta > 200 else 4

            cursor.execute("""
                INSERT INTO recommendations (rec_type, card_id, card_title, position, reason,
                    priority, estimated_price, meta_score, value_ratio, roster_impact)
                VALUES ('buy', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                card['card_id'], card['card_title'], pos, reason,
                priority, price, meta, round(value_ratio, 2),
                f"Replace {current['player_name']} ({slot_label})" if current else f"Fill {slot_label}"
            ))

    # (Multi-position scan removed 2026-04-20: the primary loop above now
    # honors pos_rating_* natively via build_eligible_where_clause, so a
    # CF-primary card with LF=32 competes for the LF slot in the SAME pass
    # that handles true-LF cards. The old secondary scan gated on rating
    # >= 100 — too strict; Van Haltren's LF=32 never qualified — and
    # skipped cards already matched at their primary position, so versatile
    # cards were actively hidden from secondary slots. See
    # memory/project_position_flexibility_gap.md for the backstory.)

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
            "SELECT signal, confidence, score, reasons FROM live_card_cache WHERE card_id = ?",
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

    Uses performance gaps, not hardcoded meta thresholds. Prior code had
    ``meta > 600`` for batters and ``meta > 400`` for pitchers, which
    silently excluded every Low Bronze roster from ever being flagged
    (their card metas rarely reach those floors). ``lineup_role`` already
    gates to active roster, so the meta threshold was redundant AND
    tier-biased. Removed.
    """
    import logging
    logger = logging.getLogger(__name__)
    cursor = conn.cursor()

    try:
        # Underperforming batters: active roster + 50+ AB + OPS below .650.
        # (.650 is roughly the low-end "starter-worthy" floor across tiers.)
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
            WHERE bs.ops < 0.650
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

            # OPS formatted explicitly for values >= 1.000 (fixes .1050 bug).
            ops_str = f"{ops:.3f}" if ops >= 1.0 else f".{int(round(ops * 1000)):03d}"
            reason = f"Underperformer: meta {meta:.0f} but {ops_str} OPS in-game. Consider selling for ~{price:,} PP"

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
            WHERE ps.era > 5.00
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

    except sqlite3.OperationalError as e:
        # Stats tables may not exist yet on fresh install — that's expected
        # and non-fatal. Anything else we want to see in the logs.
        if "no such table" in str(e).lower():
            logger.info("Skipping underperformer detection: stats tables not yet created")
        else:
            logger.warning(f"Underperformer detection failed: {e}", exc_info=True)
    except Exception as e:
        logger.warning(f"Underperformer detection failed: {e}", exc_info=True)


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
