"""Budget-constrained roster optimizer with DP knapsack and greedy fallback."""
import sqlite3
from app.core.database import get_connection


# DH excluded — any batter can DH, no dedicated DH card needed
BATTING_POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
PITCHING_POSITIONS = ['SP', 'RP', 'CL']
ALL_POSITIONS = BATTING_POSITIONS + PITCHING_POSITIONS


def get_roster_meta_total(conn=None):
    """Return total meta score of active roster starters."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    rows = conn.execute("""
        SELECT position, meta_score
        FROM roster_current
        WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
    """).fetchall()

    total = sum(r['meta_score'] or 0 for r in rows)

    if close_conn:
        conn.close()
    return total


def _get_roster_starters(conn):
    """Get the best starter at each position from the roster table."""
    rows = conn.execute("""
        SELECT player_name, position, meta_score
        FROM roster_current
        WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
    """).fetchall()

    by_pos = {}
    for r in rows:
        pos = r['position']
        if pos not in by_pos or (r['meta_score'] or 0) > (by_pos[pos]['meta_score'] or 0):
            by_pos[pos] = {
                'player_name': r['player_name'],
                'position': pos,
                'meta_score': r['meta_score'] or 0,
            }
    return by_pos


def _get_upgrade_candidates(conn, position, current_meta, budget):
    """Find market cards that are upgrades for a position within budget."""
    is_pitching = position in PITCHING_POSITIONS

    if is_pitching:
        rows = conn.execute("""
            SELECT card_id, card_title, pitcher_role_name AS position,
                   meta_score_pitching AS meta_score,
                   last_10_price, sell_order_low, buy_order_high, tier_name
            FROM cards
            WHERE pitcher_role_name = ? AND owned = 0
                AND last_10_price > 0 AND last_10_price <= ?
                AND meta_score_pitching > ?
            ORDER BY meta_score_pitching DESC
        """, (position, budget, current_meta)).fetchall()
    else:
        rows = conn.execute("""
            SELECT card_id, card_title, position_name AS position,
                   meta_score_batting AS meta_score,
                   last_10_price, sell_order_low, buy_order_high, tier_name
            FROM cards
            WHERE position_name = ? AND owned = 0
                AND last_10_price > 0 AND last_10_price <= ?
                AND meta_score_batting > ?
            ORDER BY meta_score_batting DESC
        """, (position, budget, current_meta)).fetchall()

    return rows


def optimize_budget_dp(budget_pp, conn=None, priority_positions=None, exclude_positions=None):
    """Dynamic programming optimizer -- finds globally optimal upgrades.

    Uses recursive DP with memoization over (position_index, budget_bucket)
    to maximize total meta gain across all positions within the PP budget.

    Args:
        budget_pp: total PP budget available
        conn: optional sqlite3 connection
        priority_positions: list of positions to fill first (optional)
        exclude_positions: list of positions to skip (optional)

    Returns:
        dict with keys:
            - transactions: list of recommended buys
            - total_meta_gain: sum of meta improvements
            - total_cost: sum of PP spent
            - remaining_budget: leftover PP
            - method: 'dp'
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    exclude_positions = set(exclude_positions or [])
    priority_positions = list(priority_positions or [])

    # Build ordered position list: priority positions first, then the rest
    remaining_pos = [p for p in ALL_POSITIONS if p not in exclude_positions and p not in priority_positions]
    positions = [p for p in priority_positions if p not in exclude_positions] + remaining_pos
    n_positions = len(positions)

    starters = _get_roster_starters(conn)

    # Budget discretization: 50 PP buckets
    BUCKET_SIZE = 50
    max_buckets = budget_pp // BUCKET_SIZE

    def price_to_buckets(price):
        """Convert a PP price to the number of budget buckets it consumes."""
        return (price + BUCKET_SIZE - 1) // BUCKET_SIZE  # ceiling division

    # For each position, gather top 10 upgrade candidates
    TOP_N = 10
    candidates_by_pos = []
    for pos in positions:
        current = starters.get(pos)
        current_meta = current['meta_score'] if current else 0
        current_player = current['player_name'] if current else '(empty)'

        raw = _get_upgrade_candidates(conn, pos, current_meta, budget_pp)
        pos_candidates = []
        for card in raw[:TOP_N]:
            price = card['last_10_price']
            if price <= 0:
                continue
            meta_gain = card['meta_score'] - current_meta
            if meta_gain <= 0:
                continue
            pos_candidates.append({
                'card_id': card['card_id'],
                'card_title': card['card_title'],
                'position': pos,
                'current_player': current_player,
                'current_meta': current_meta,
                'new_meta': card['meta_score'],
                'meta_gain': meta_gain,
                'price': price,
                'buckets': price_to_buckets(price),
                'efficiency': round(meta_gain / price, 6),
            })
        candidates_by_pos.append(pos_candidates)

    # DP with memoization: state = (position_index, remaining_budget_buckets)
    # Returns best total meta_gain achievable from positions[pos_idx:] with
    # remaining_buckets of budget left.
    memo = {}

    def dp(pos_idx, remaining_buckets):
        if pos_idx >= n_positions:
            return 0.0
        key = (pos_idx, remaining_buckets)
        if key in memo:
            return memo[key]

        # Option 1: skip this position
        best = dp(pos_idx + 1, remaining_buckets)

        # Option 2: pick one candidate at this position
        for cand in candidates_by_pos[pos_idx]:
            if cand['buckets'] <= remaining_buckets:
                val = cand['meta_gain'] + dp(pos_idx + 1, remaining_buckets - cand['buckets'])
                if val > best:
                    best = val

        memo[key] = best
        return best

    # Run DP forward pass
    optimal_gain = dp(0, max_buckets)

    # Backtrack to recover which picks were made
    transactions = []
    remaining_buckets = max_buckets
    for pos_idx in range(n_positions):
        # Check: did we skip this position?
        skip_val = dp(pos_idx + 1, remaining_buckets) if pos_idx + 1 < n_positions else 0.0

        picked = False
        for cand in candidates_by_pos[pos_idx]:
            if cand['buckets'] <= remaining_buckets:
                future = dp(pos_idx + 1, remaining_buckets - cand['buckets']) if pos_idx + 1 < n_positions else 0.0
                val = cand['meta_gain'] + future
                if abs(val - dp(pos_idx, remaining_buckets)) < 1e-9:
                    transactions.append(cand)
                    remaining_buckets -= cand['buckets']
                    picked = True
                    break
        # If not picked, we skipped -- continue

    total_meta_gain = sum(t['meta_gain'] for t in transactions)
    total_cost = sum(t['price'] for t in transactions)

    result = {
        'transactions': transactions,
        'total_meta_gain': round(total_meta_gain, 2),
        'total_cost': total_cost,
        'remaining_budget': budget_pp - total_cost,
        'method': 'dp',
    }

    if close_conn:
        conn.close()
    return result


def _optimize_budget_greedy(budget_pp, conn, priority_positions=None, exclude_positions=None):
    """Original greedy optimizer (internal). Used as fallback."""
    exclude_positions = set(exclude_positions or [])
    priority_positions = list(priority_positions or [])

    # Build ordered position list: priority first, then rest
    remaining_pos = [p for p in ALL_POSITIONS if p not in exclude_positions and p not in priority_positions]
    ordered_positions = [p for p in priority_positions if p not in exclude_positions] + remaining_pos

    starters = _get_roster_starters(conn)
    remaining = budget_pp
    transactions = []
    filled_positions = set()

    while remaining > 0:
        best_candidate = None
        best_efficiency = -1
        best_position = None

        for pos in ordered_positions:
            if pos in filled_positions:
                continue

            current = starters.get(pos)
            current_meta = current['meta_score'] if current else 0
            current_player = current['player_name'] if current else '(empty)'

            candidates = _get_upgrade_candidates(conn, pos, current_meta, remaining)
            if not candidates:
                continue

            for card in candidates:
                price = card['last_10_price']
                if price <= 0:
                    continue
                meta_gain = card['meta_score'] - current_meta
                if meta_gain <= 0:
                    continue
                efficiency = meta_gain / price

                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_position = pos
                    best_candidate = {
                        'card_id': card['card_id'],
                        'card_title': card['card_title'],
                        'position': pos,
                        'current_player': current_player,
                        'current_meta': current_meta,
                        'new_meta': card['meta_score'],
                        'meta_gain': meta_gain,
                        'price': price,
                        'efficiency': round(efficiency, 6),
                    }

        if best_candidate is None:
            break

        transactions.append(best_candidate)
        filled_positions.add(best_position)
        remaining -= best_candidate['price']

    total_meta_gain = sum(t['meta_gain'] for t in transactions)
    total_cost = sum(t['price'] for t in transactions)

    return {
        'transactions': transactions,
        'total_meta_gain': round(total_meta_gain, 2),
        'total_cost': total_cost,
        'remaining_budget': remaining,
        'method': 'greedy',
    }


def optimize_budget(budget_pp, conn=None, method='dp', priority_positions=None, exclude_positions=None):
    """Optimize roster upgrades within a PP budget.

    Delegates to DP (default) or greedy optimizer. Falls back to greedy on error.

    Args:
        budget_pp: total PP budget available
        conn: optional sqlite3 connection
        method: 'dp' for dynamic programming, 'greedy' for fast greedy
        priority_positions: list of positions to fill first (optional)
        exclude_positions: list of positions to skip (optional)

    Returns:
        dict with keys:
            - transactions: list of recommended buys
            - total_meta_gain: sum of meta improvements
            - total_cost: sum of PP spent
            - remaining_budget: leftover PP
            - method: which algorithm was used ('dp' or 'greedy')
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        if method == 'dp':
            result = optimize_budget_dp(budget_pp, conn, priority_positions, exclude_positions)
        else:
            result = _optimize_budget_greedy(budget_pp, conn, priority_positions, exclude_positions)
    except Exception:
        # Fall back to greedy on any DP error
        result = _optimize_budget_greedy(budget_pp, conn, priority_positions, exclude_positions)
        result['method'] = 'greedy (fallback)'

    if close_conn:
        conn.close()
    return result


def simulate_transactions(buys, sells, conn=None):
    """What-if sandbox: simulate buying and selling specific cards.

    Args:
        buys: list of card_ids to buy
        sells: list of card_ids to sell
        conn: optional sqlite3 connection

    Returns:
        dict with:
            - buy_cost: total cost of buys (using last_10_price)
            - sell_revenue: total revenue from sells (using buy_order_high)
            - net_pp_change: sell_revenue - buy_cost
            - roster_before: dict of position -> meta_score (current)
            - roster_after: dict of position -> meta_score (projected)
            - total_meta_before: total meta of current roster
            - total_meta_after: projected total meta
            - meta_delta: total_meta_after - total_meta_before
            - buy_details: list of buy card info
            - sell_details: list of sell card info
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    starters = _get_roster_starters(conn)

    # Build roster_before from current starters
    roster_before = {}
    for pos in ALL_POSITIONS:
        current = starters.get(pos)
        roster_before[pos] = current['meta_score'] if current else 0

    total_meta_before = sum(roster_before.values())

    # Fetch buy card details
    buy_details = []
    buy_cost = 0
    buy_by_pos = {}
    for card_id in buys:
        row = conn.execute("""
            SELECT card_id, card_title, position_name, pitcher_role_name,
                   meta_score_batting, meta_score_pitching,
                   last_10_price, sell_order_low
            FROM cards WHERE card_id = ?
        """, (card_id,)).fetchone()
        if not row:
            continue

        is_pitching = row['pitcher_role_name'] in PITCHING_POSITIONS
        pos = row['pitcher_role_name'] if is_pitching else row['position_name']
        meta = row['meta_score_pitching'] if is_pitching else row['meta_score_batting']
        price = row['last_10_price'] or row['sell_order_low'] or 0

        buy_details.append({
            'card_id': row['card_id'],
            'card_title': row['card_title'],
            'position': pos,
            'meta_score': meta or 0,
            'price': price,
        })
        buy_cost += price

        # Track best buy per position for roster projection
        if pos not in buy_by_pos or (meta or 0) > buy_by_pos[pos]:
            buy_by_pos[pos] = meta or 0

    # Fetch sell card details
    sell_details = []
    sell_revenue = 0
    sell_positions = set()
    for card_id in sells:
        row = conn.execute("""
            SELECT card_id, card_title, position_name, pitcher_role_name,
                   meta_score_batting, meta_score_pitching,
                   buy_order_high, last_10_price
            FROM cards WHERE card_id = ?
        """, (card_id,)).fetchone()
        if not row:
            continue

        is_pitching = row['pitcher_role_name'] in PITCHING_POSITIONS
        pos = row['pitcher_role_name'] if is_pitching else row['position_name']
        meta = row['meta_score_pitching'] if is_pitching else row['meta_score_batting']
        revenue = row['buy_order_high'] or row['last_10_price'] or 0

        sell_details.append({
            'card_id': row['card_id'],
            'card_title': row['card_title'],
            'position': pos,
            'meta_score': meta or 0,
            'revenue': revenue,
        })
        sell_revenue += revenue
        sell_positions.add(pos)

    # Project roster_after
    roster_after = dict(roster_before)

    # Apply sells: if selling the current starter at a position, meta drops to 0
    # (simplified — assumes the sold card is the starter)
    for pos in sell_positions:
        sold_at_pos = [s for s in sell_details if s['position'] == pos]
        if sold_at_pos:
            current = starters.get(pos)
            if current:
                # Check if any sold card matches the starter
                for s in sold_at_pos:
                    if current['player_name'] in s['card_title']:
                        roster_after[pos] = 0
                        break

    # Apply buys: if buying a card better than current at a position, upgrade
    for pos, buy_meta in buy_by_pos.items():
        if buy_meta > roster_after.get(pos, 0):
            roster_after[pos] = buy_meta

    total_meta_after = sum(roster_after.values())

    result = {
        'buy_cost': buy_cost,
        'sell_revenue': sell_revenue,
        'net_pp_change': sell_revenue - buy_cost,
        'roster_before': roster_before,
        'roster_after': roster_after,
        'total_meta_before': round(total_meta_before, 2),
        'total_meta_after': round(total_meta_after, 2),
        'meta_delta': round(total_meta_after - total_meta_before, 2),
        'buy_details': buy_details,
        'sell_details': sell_details,
    }

    if close_conn:
        conn.close()
    return result
