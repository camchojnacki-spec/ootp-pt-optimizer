"""Card flipping analysis engine — finds profitable buy/sell opportunities."""
import logging
from datetime import date, timedelta
from app.core.database import get_connection

logger = logging.getLogger(__name__)


def find_spread_flips(min_profit=20, min_margin_pct=10, conn=None):
    """Find cards with a profitable buy-order/sell-order spread.

    Looks for cards where buy_order_high > sell_order_low (immediate flip
    opportunity). Ranks by a composite flip_score that accounts for both
    absolute profit and margin percentage, penalizing high-variance cards.

    Args:
        min_profit: minimum absolute profit in PP.
        min_margin_pct: minimum margin as a percentage of buy cost.
        conn: optional existing DB connection.

    Returns:
        List of dicts sorted by flip_score DESC (max 100).
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    rows = conn.execute("""
        SELECT card_id, card_title, position_name, pitcher_role_name, tier_name,
               buy_order_high, sell_order_low, last_10_price, last_10_variance
        FROM cards
        WHERE buy_order_high > 0
          AND sell_order_low > 0
          AND buy_order_high > sell_order_low
    """).fetchall()

    if close_conn:
        conn.close()

    results = []
    for row in rows:
        sell_order_low = row["sell_order_low"]
        buy_order_high = row["buy_order_high"]
        last_10_price = row["last_10_price"] or 0
        variance = row["last_10_variance"] or 0

        profit = buy_order_high - sell_order_low
        margin_pct = (profit / sell_order_low) * 100 if sell_order_low > 0 else 0

        if profit < min_profit or margin_pct < min_margin_pct:
            continue

        # Safety factor based on variance relative to price
        if last_10_price > 0 and variance < last_10_price * 0.3:
            safety_factor = 1.0
            risk_level = "Low"
        elif last_10_price > 0 and variance < last_10_price * 0.5:
            safety_factor = 0.7
            risk_level = "Medium"
        else:
            safety_factor = 0.7
            risk_level = "High"

        flip_score = profit * (margin_pct / 100) * safety_factor

        position = row["pitcher_role_name"] if row["pitcher_role_name"] else row["position_name"]

        results.append({
            "card_id": row["card_id"],
            "card_title": row["card_title"],
            "position": position,
            "tier_name": row["tier_name"],
            "buy_at": sell_order_low,
            "sell_at": buy_order_high,
            "profit": profit,
            "margin_pct": round(margin_pct, 1),
            "flip_score": round(flip_score, 1),
            "last_10_price": last_10_price,
            "variance": variance,
            "risk_level": risk_level,
        })

    results.sort(key=lambda x: x["flip_score"], reverse=True)
    return results[:100]


def find_volatility_flips(min_variance_ratio=0.15, conn=None):
    """Find cards with high price volatility — wide swings create buying opportunities.

    Cards with a high variance-to-price ratio see frequent price dips that can
    be scooped up with low buy orders.

    Args:
        min_variance_ratio: minimum (last_10_variance / last_10_price).
        conn: optional existing DB connection.

    Returns:
        List of dicts sorted by potential_profit DESC (max 100).
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    rows = conn.execute("""
        SELECT card_id, card_title, position_name, pitcher_role_name, tier_name,
               last_10_price, last_10_variance,
               meta_score_batting, meta_score_pitching
        FROM cards
        WHERE last_10_price > 0
          AND last_10_variance > 0
    """).fetchall()

    if close_conn:
        conn.close()

    results = []
    for row in rows:
        last_10_price = row["last_10_price"]
        variance = row["last_10_variance"]
        variance_ratio = variance / last_10_price

        if variance_ratio < min_variance_ratio:
            continue

        estimated_low = last_10_price - variance
        estimated_high = last_10_price + variance
        potential_profit = estimated_high - estimated_low  # 2 * variance

        meta_batting = row["meta_score_batting"] or 0
        meta_pitching = row["meta_score_pitching"] or 0
        meta_score = max(meta_batting, meta_pitching)

        position = row["pitcher_role_name"] if row["pitcher_role_name"] else row["position_name"]

        results.append({
            "card_id": row["card_id"],
            "card_title": row["card_title"],
            "position": position,
            "tier_name": row["tier_name"],
            "last_10_price": last_10_price,
            "variance": variance,
            "variance_ratio": round(variance_ratio, 3),
            "estimated_low": max(estimated_low, 0),
            "estimated_high": estimated_high,
            "potential_profit": potential_profit,
            "meta_score": round(meta_score, 1),
        })

    results.sort(key=lambda x: x["potential_profit"], reverse=True)
    return results[:100]


def find_trend_flips(conn=None):
    """Find cards whose current price is significantly below their historical average.

    Requires at least 2 price snapshots. Cards that have dropped 15%+ from
    their historical average are recovery candidates.

    Args:
        conn: optional existing DB connection.

    Returns:
        List of dicts sorted by potential_profit DESC (max 50).
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    rows = conn.execute("""
        SELECT c.card_id, c.card_title, c.position_name, c.pitcher_role_name,
               c.tier_name, c.last_10_price,
               AVG(ps.last_10_price) as avg_historical_price,
               COUNT(ps.id) as snapshot_count
        FROM cards c
        JOIN price_snapshots ps ON c.card_id = ps.card_id
        WHERE c.last_10_price > 0
          AND ps.last_10_price > 0
        GROUP BY c.card_id
        HAVING snapshot_count >= 2
           AND avg_historical_price > 0
    """).fetchall()

    if close_conn:
        conn.close()

    results = []
    for row in rows:
        current_price = row["last_10_price"]
        avg_historical = row["avg_historical_price"]

        if avg_historical <= 0:
            continue

        price_drop_pct = ((avg_historical - current_price) / avg_historical) * 100

        if price_drop_pct < 15:
            continue

        recovery_target = round(avg_historical)
        potential_profit = recovery_target - current_price

        position = row["pitcher_role_name"] if row["pitcher_role_name"] else row["position_name"]

        results.append({
            "card_id": row["card_id"],
            "card_title": row["card_title"],
            "position": position,
            "tier_name": row["tier_name"],
            "current_price": current_price,
            "avg_historical_price": round(avg_historical),
            "price_drop_pct": round(price_drop_pct, 1),
            "recovery_target": recovery_target,
            "potential_profit": potential_profit,
            "snapshot_count": row["snapshot_count"],
        })

    results.sort(key=lambda x: x["potential_profit"], reverse=True)
    return results[:50]


def find_live_card_flips(conn=None):
    """Find Live cards predicted to receive a ratings upgrade.

    Cross-references the live_card_cache table (populated by the Live Card
    Tracker) with the cards table to find upgrade candidates worth buying
    before the ratings boost.

    Args:
        conn: optional existing DB connection.

    Returns:
        List of dicts sorted by upgrade_score DESC. Empty list if
        live_card_cache table does not exist.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    # Check if live_card_cache table exists
    table_check = conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='live_card_cache'
    """).fetchone()

    if not table_check:
        if close_conn:
            conn.close()
        return []

    rows = conn.execute("""
        SELECT c.card_id, c.card_title, c.position_name, c.pitcher_role_name,
               c.tier_name, c.last_10_price,
               c.meta_score_batting, c.meta_score_pitching,
               lcc.signal, lcc.confidence, lcc.score, lcc.reasons
        FROM live_card_cache lcc
        JOIN cards c ON lcc.card_id = c.card_id
        WHERE lcc.signal = 'upgrade'
          AND lcc.confidence IN ('high', 'medium')
        ORDER BY lcc.score DESC
    """).fetchall()

    if close_conn:
        conn.close()

    results = []
    for row in rows:
        meta_batting = row["meta_score_batting"] or 0
        meta_pitching = row["meta_score_pitching"] or 0
        meta_score = max(meta_batting, meta_pitching)

        position = row["pitcher_role_name"] if row["pitcher_role_name"] else row["position_name"]

        results.append({
            "card_id": row["card_id"],
            "card_title": row["card_title"],
            "position": position,
            "tier_name": row["tier_name"],
            "current_price": row["last_10_price"] or 0,
            "upgrade_score": row["score"],
            "confidence": row["confidence"],
            "reasons": row["reasons"],
            "meta_score": round(meta_score, 1),
        })

    return results


def get_flip_summary(conn=None):
    """Quick summary stats across all flip strategies.

    Returns:
        Dict with counts and highlights for each flip type.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    spread_flips = find_spread_flips(conn=conn)
    volatility_flips = find_volatility_flips(conn=conn)

    # Trend flips — only if snapshots exist
    snapshot_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM price_snapshots"
    ).fetchone()["cnt"]

    trend_flips = find_trend_flips(conn=conn) if snapshot_count > 0 else []

    if close_conn:
        conn.close()

    best_spread = spread_flips[0] if spread_flips else None

    return {
        "spread_flip_count": len(spread_flips),
        "best_spread_flip": best_spread,
        "volatility_play_count": len(volatility_flips),
        "trend_play_count": len(trend_flips),
    }


def find_matchup_flips(days_ahead=7, conn=None):
    """Find Live cards with favorable upcoming MLB matchups that could boost their stats and trigger upgrades.

    Uses MLB Stats API to check upcoming schedules. Players facing weak opponents
    or in hot streaks are likely to put up numbers that push OOTP to upgrade their Live card.

    Strategy: Buy cards of players with easy upcoming matchups, sell after the stat boost.
    """
    try:
        import statsapi
    except ImportError:
        logger.error("statsapi package not installed — cannot find matchup flips")
        return []

    try:
        from app.core.live_card_tracker import OOTP_TEAM_MAP
    except ImportError:
        logger.error("Could not import OOTP_TEAM_MAP from live_card_tracker")
        return []

    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        # 1. Get all unowned Live cards with a valid price
        rows = conn.execute("""
            SELECT card_id, card_title, position_name, pitcher_role_name,
                   tier_name, last_10_price, buy_order_high, team
            FROM cards
            WHERE card_title LIKE 'MLB 2026 Live%'
              AND owned = 0
              AND last_10_price > 0
        """).fetchall()

        # Check for live_card_cache table
        cache_exists = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='live_card_cache'
        """).fetchone()

        upgrade_map = {}
        if cache_exists:
            cache_rows = conn.execute("""
                SELECT card_id, signal, score FROM live_card_cache
                WHERE signal = 'upgrade'
            """).fetchall()
            for cr in cache_rows:
                upgrade_map[cr["card_id"]] = cr["score"]

        if close_conn:
            conn.close()
            close_conn = False

        if not rows:
            return []

        # 2. Fetch upcoming schedule
        start_dt = date.today()
        end_dt = start_dt + timedelta(days=days_ahead)
        start_str = start_dt.strftime("%m/%d/%Y")
        end_str = end_dt.strftime("%m/%d/%Y")

        schedule = statsapi.schedule(start_date=start_str, end_date=end_str)

        # 3. Build team game counts — keyed by full team name
        team_games = {}      # full team name -> total games
        team_home_games = {}  # full team name -> home games
        for game in schedule:
            home = game.get("home_name", "")
            away = game.get("away_name", "")
            for t in (home, away):
                if t:
                    team_games[t] = team_games.get(t, 0) + 1
            if home:
                team_home_games[home] = team_home_games.get(home, 0) + 1

        # Build reverse map: full name -> abbr, and abbr -> full name
        abbr_to_full = dict(OOTP_TEAM_MAP)
        full_to_full = {v: v for v in OOTP_TEAM_MAP.values()}

        # 4. Score each card
        results = []
        for row in rows:
            card_team = (row["team"] or "").strip()
            # Try to resolve to a full MLB team name
            full_name = None
            if card_team in full_to_full:
                full_name = card_team
            elif card_team in abbr_to_full:
                full_name = abbr_to_full[card_team]
            else:
                # Fuzzy: check if the card_team is a substring of any full name
                for fn in full_to_full:
                    if card_team and card_team.lower() in fn.lower():
                        full_name = fn
                        break

            games = team_games.get(full_name, 0) if full_name else 0
            home = team_home_games.get(full_name, 0) if full_name else 0

            if games == 0:
                continue

            price = row["last_10_price"] or 0
            position = row["pitcher_role_name"] if row["pitcher_role_name"] else row["position_name"]
            is_pitcher = bool(row["pitcher_role_name"])

            # Base score: games in window
            score = games * 10.0

            # Home game bonus (slight advantage)
            score += home * 2.0

            # Upgrade signal bonus from live_card_cache
            upgrade_score = upgrade_map.get(row["card_id"], 0)
            if upgrade_score > 0:
                score += upgrade_score * 5.0

            # Pitcher bonus: more games = more potential starts
            if is_pitcher:
                score += games * 3.0

            # Price efficiency: cheaper cards are better flip targets
            if price > 0:
                score *= (1000.0 / (price + 500.0))

            # Build reasoning
            reasons = []
            reasons.append(f"{games} games in next {days_ahead} days ({home} home)")
            if upgrade_score > 0:
                reasons.append(f"Upgrade signal (score={upgrade_score})")
            if is_pitcher:
                reasons.append("Pitcher — potential starts")
            if price < 200:
                reasons.append("Low price — high flip ROI potential")

            results.append({
                "card_id": row["card_id"],
                "card_title": row["card_title"],
                "position": position,
                "tier_name": row["tier_name"],
                "price": price,
                "team": full_name or card_team,
                "games_in_window": games,
                "home_games": home,
                "matchup_score": round(score, 1),
                "flip_reasoning": "; ".join(reasons),
            })

        results.sort(key=lambda x: x["matchup_score"], reverse=True)
        return results

    except Exception as e:
        logger.error("Error in find_matchup_flips: %s", e, exc_info=True)
        if close_conn:
            try:
                conn.close()
            except Exception:
                pass
        return []


def find_hot_streak_flips(conn=None):
    """Find Live cards where a player is performing well but the market price hasn't caught up.

    Cross-references live_card_cache upgrade signals with card pricing to find
    cards where the price-to-performance ratio suggests the market hasn't priced
    in the hot streak yet.

    Strategy: Buy before the market notices, sell after the price adjusts upward.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        # Check if live_card_cache table exists
        cache_check = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='live_card_cache'
        """).fetchone()

        if not cache_check:
            if close_conn:
                conn.close()
            return []

        rows = conn.execute("""
            SELECT c.card_id, c.card_title, c.position_name, c.pitcher_role_name,
                   c.tier_name, c.last_10_price, c.buy_order_high,
                   lcc.signal, lcc.confidence, lcc.score, lcc.reasons
            FROM live_card_cache lcc
            JOIN cards c ON lcc.card_id = c.card_id
            WHERE lcc.signal = 'upgrade'
              AND c.owned = 0
              AND c.last_10_price > 0
        """).fetchall()

        if close_conn:
            conn.close()
            close_conn = False

        results = []
        for row in rows:
            price = row["last_10_price"]
            buy_high = row["buy_order_high"] or 0

            # Price lag ratio: if buy orders haven't spiked relative to recent price,
            # the market hasn't caught up to the hot streak
            price_lag_ratio = (buy_high / price) if price > 0 else 999

            # Only include cards where price hasn't caught up (ratio < 1.2)
            if price_lag_ratio >= 1.2:
                continue

            upgrade_score = row["score"] or 0
            confidence = row["confidence"] or "low"

            # Score: upgrade signal strength weighted by how much price lag remains
            # Lower price_lag_ratio = more opportunity
            lag_multiplier = max(0.1, 1.2 - price_lag_ratio)
            score = upgrade_score * lag_multiplier

            # Confidence boost
            if confidence == "high":
                score *= 1.5
            elif confidence == "medium":
                score *= 1.0
            else:
                score *= 0.6

            position = row["pitcher_role_name"] if row["pitcher_role_name"] else row["position_name"]

            # Build reasons list
            reasons_parts = []
            db_reasons = row["reasons"] or ""
            if db_reasons:
                reasons_parts.append(db_reasons)
            reasons_parts.append(f"Price lag ratio: {price_lag_ratio:.2f} (buy_high={buy_high}, last_10={price})")
            if price_lag_ratio < 0.9:
                reasons_parts.append("Buy orders well below recent price — strong opportunity")
            elif price_lag_ratio < 1.0:
                reasons_parts.append("Buy orders slightly below recent price — good window")

            results.append({
                "card_id": row["card_id"],
                "card_title": row["card_title"],
                "position": position,
                "tier_name": row["tier_name"],
                "price": price,
                "upgrade_score": upgrade_score,
                "confidence": confidence,
                "price_lag_ratio": round(price_lag_ratio, 3),
                "score": round(score, 1),
                "reasons": "; ".join(reasons_parts),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    except Exception as e:
        logger.error("Error in find_hot_streak_flips: %s", e, exc_info=True)
        if close_conn:
            try:
                conn.close()
            except Exception:
                pass
        return []
