"""Budget-constrained roster optimizer with DP knapsack and greedy fallback.

UAT 2026-04-25 §5.1 / Tier-2 #7: pitching is now multi-slot (5 SP, 6 RP,
1 CL) instead of one slot per role. The DP knapsack iterates over each
slot and enforces a unique-card constraint so the same RP can't fill
multiple bullpen slots in one optimization pass.
"""
import sqlite3
from app.core.database import get_connection


# DH excluded — any batter can DH, no dedicated DH card needed.
BATTING_POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']

# Pitching role tags (used for SQL filtering and display).
PITCHING_ROLE_TAGS = ['SP', 'RP', 'CL']

# Per-role slot expansion. UAT noted the optimizer was treating SP/RP/CL
# as 1-slot positions, ignoring rotation depth and bullpen breadth.
# A real PT roster carries 5 SPs (full rotation), 6 RPs (middle / setup
# / long), and 1 CL. CL stays at 1; bench depth is still recommended via
# the page's promotion logic, not via the DP knapsack.
PITCHING_SLOTS_BY_ROLE = {
    'SP': ['SP1', 'SP2', 'SP3', 'SP4', 'SP5'],
    'RP': ['RP1', 'RP2', 'RP3', 'RP4', 'RP5', 'RP6'],
    'CL': ['CL'],
}
PITCHING_SLOTS = [
    s for tag in PITCHING_ROLE_TAGS for s in PITCHING_SLOTS_BY_ROLE[tag]
]
ALL_POSITIONS = BATTING_POSITIONS + PITCHING_SLOTS


def _slot_role(slot: str) -> str:
    """Map a slot label back to its role tag (SP1 -> SP, RP3 -> RP, CL -> CL)."""
    if slot.startswith('SP'):
        return 'SP'
    if slot.startswith('RP'):
        return 'RP'
    if slot == 'CL':
        return 'CL'
    return slot  # batting positions are their own "role"


def _slot_index(slot: str) -> int:
    """Order index within a role (SP1 -> 1, SP5 -> 5, CL -> 1)."""
    if slot in ('CL',):
        return 1
    if slot.startswith('SP') or slot.startswith('RP'):
        try:
            return int(slot[2:])
        except (ValueError, TypeError):
            return 1
    return 1


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
    """Return the current best card per slot (batting positions + pitching slots).

    Batting: one per position, max meta. Same as before.
    Pitching: top-N within each role, where N = len(PITCHING_SLOTS_BY_ROLE[role]).
        SP1 = best SP, SP2 = 2nd-best, ..., SP5 = 5th-best.
        RP1..RP6 likewise. CL = best CL.

    Each slot dict carries ``card_id`` so callers can enforce
    uniqueness during upgrade selection (a card already in SP1 must
    not be re-picked for SP2).
    """
    rows = conn.execute("""
        SELECT player_name, position, meta_score, card_id
        FROM roster_current
        WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
    """).fetchall()

    by_slot: dict[str, dict] = {}

    # Batting — use the platoon-aware resolver from roster_analysis (single
    # source of truth, UAT Tier-2 #9 / #12). For optimization we need ONE
    # "current meta" per slot, so when a real platoon is detected we take
    # the platoon-effective meta = mean of the two halves (the optimizer
    # would need to upgrade BOTH halves to actually retire the platoon).
    try:
        from app.core.roster_analysis import get_position_starters
        starters_resolved = get_position_starters(conn, platoon_aware=True)
    except Exception:
        starters_resolved = {}

    for pos in BATTING_POSITIONS:
        starters_here = starters_resolved.get(pos) or []
        if not starters_here:
            continue
        if len(starters_here) >= 2:
            # Platoon: aggregate to a representative meta. The
            # platoon-effective value is the mean — beating that with
            # one card means the platoon's worse half is replaceable;
            # beating both halves means the platoon is fully replaced.
            metas = [s.get('meta_score') or 0 for s in starters_here]
            agg_meta = sum(metas) / len(metas)
            primary = max(starters_here, key=lambda s: s.get('meta_score') or 0)
            by_slot[pos] = {
                'player_name': f"{primary['player_name']} (platoon)",
                'position': pos,
                'meta_score': agg_meta,
                'card_id': primary.get('card_id'),
                'is_platoon': True,
                'platoon_card_ids': [s.get('card_id') for s in starters_here if s.get('card_id')],
            }
        else:
            s = starters_here[0]
            by_slot[pos] = {
                'player_name': s['player_name'],
                'position': pos,
                'meta_score': s.get('meta_score') or 0,
                'card_id': s.get('card_id'),
                'is_platoon': False,
            }

    # Pitching — group by role, sort desc by meta, fill SP1..SP5 / RP1..RP6 / CL.
    # Use per-league meta when an active league is set so depth-chart
    # ordering reflects how the cards rank in the user's tier (e.g.
    # Kerkering 666 lb124 → 591 lb122 — different RP1 candidate).
    league = _active_league_for_meta()
    per_league_meta: dict[int, float] = {}
    if league:
        try:
            for r in conn.execute(
                "SELECT card_id, meta_score FROM card_meta_by_league "
                "WHERE league_id = ? AND side = 'pitching'",
                (league,),
            ).fetchall():
                per_league_meta[r['card_id']] = r['meta_score']
        except Exception:
            pass

    by_role: dict[str, list[dict]] = {tag: [] for tag in PITCHING_ROLE_TAGS}
    for r in rows:
        role = r['position']
        if role not in PITCHING_ROLE_TAGS:
            continue
        # Prefer per-league meta when available — global meta is the
        # fallback for cards that haven't been recalced yet.
        cid = r['card_id']
        meta = per_league_meta.get(cid) if cid is not None else None
        if meta is None:
            meta = r['meta_score'] or 0
        by_role[role].append({
            'player_name': r['player_name'],
            'meta_score': meta,
            'card_id': cid,
        })
    for role, players in by_role.items():
        players.sort(key=lambda x: x['meta_score'] or 0, reverse=True)
        for slot, p in zip(PITCHING_SLOTS_BY_ROLE[role], players):
            by_slot[slot] = {
                'player_name': p['player_name'],
                'position': slot,
                'meta_score': p['meta_score'],
                'card_id': p['card_id'],
            }
        # Empty slots (e.g. only 4 RPs on roster) — populate with placeholders
        # so the optimizer treats them as "fill from market" candidates rather
        # than skipping the slot entirely.
        for slot in PITCHING_SLOTS_BY_ROLE[role][len(players):]:
            by_slot[slot] = {
                'player_name': '(empty)',
                'position': slot,
                'meta_score': 0,
                'card_id': None,
            }

    return by_slot


def _active_league_for_meta() -> str | None:
    """Return the active league for per-tier meta lookups.

    Reads the process-level override set by the Roster Optimizer page
    (via ``meta_scoring.set_active_league_override``). Falls back to
    config.yaml's ``active_league``. ``None`` means "use global meta".
    """
    try:
        from app.core.meta_scoring import _ACTIVE_LEAGUE_OVERRIDE  # type: ignore
        if _ACTIVE_LEAGUE_OVERRIDE:
            return _ACTIVE_LEAGUE_OVERRIDE
    except Exception:
        pass
    try:
        from app.core.database import load_config
        return (load_config() or {}).get('active_league')
    except Exception:
        return None


def _get_upgrade_candidates(conn, position, current_meta, budget,
                             excluded_card_ids: set | None = None):
    """Find market cards that are upgrades for a position within budget.

    For pitching slots (SP1..SP5, RP1..RP6, CL) the SQL filter uses the
    ROLE tag (SP/RP/CL) — every SP-eligible card is a candidate for any
    SP slot. The DP knapsack picks which slot they fill, with uniqueness
    enforced via ``excluded_card_ids``.

    UAT 2026-04-25: when an active league is set, we LEFT JOIN
    ``card_meta_by_league`` and prefer the per-tier meta score. Cards
    without a per-tier row (no recalc yet) fall back to global meta.
    The ORDER BY uses the per-tier value when available so candidates
    rank against the right scale for the user's selected tier.
    """
    role = _slot_role(position)
    is_pitching = role in PITCHING_ROLE_TAGS
    excluded_card_ids = excluded_card_ids or set()
    league = _active_league_for_meta()
    side = 'pitching' if is_pitching else 'batting'

    if is_pitching:
        global_col = "meta_score_pitching"
        pos_col = "pitcher_role_name"
        pos_arg = role
    else:
        global_col = "meta_score_batting"
        pos_col = "position_name"
        pos_arg = position

    if league:
        # COALESCE(per_league, global) — cards with a fresh per-tier
        # score get ranked by it; cards without fall back to global.
        sql = f"""
            SELECT c.card_id, c.card_title, c.{pos_col} AS position,
                   COALESCE(cmbl.meta_score, c.{global_col}) AS meta_score,
                   c.last_10_price, c.sell_order_low, c.buy_order_high, c.tier_name
            FROM cards c
            LEFT JOIN card_meta_by_league cmbl
              ON cmbl.card_id = c.card_id
             AND cmbl.league_id = ?
             AND cmbl.side = ?
            WHERE c.{pos_col} = ? AND c.owned = 0
                AND c.last_10_price > 0 AND c.last_10_price <= ?
                AND COALESCE(cmbl.meta_score, c.{global_col}) > ?
            ORDER BY COALESCE(cmbl.meta_score, c.{global_col}) DESC
        """
        rows = conn.execute(sql, (league, side, pos_arg, budget, current_meta)).fetchall()
    else:
        sql = f"""
            SELECT card_id, card_title, {pos_col} AS position,
                   {global_col} AS meta_score,
                   last_10_price, sell_order_low, buy_order_high, tier_name
            FROM cards
            WHERE {pos_col} = ? AND owned = 0
                AND last_10_price > 0 AND last_10_price <= ?
                AND {global_col} > ?
            ORDER BY {global_col} DESC
        """
        rows = conn.execute(sql, (pos_arg, budget, current_meta)).fetchall()

    if excluded_card_ids:
        rows = [r for r in rows if r['card_id'] not in excluded_card_ids]
    return rows


def optimize_budget_dp(budget_pp, conn=None, priority_positions=None,
                        exclude_positions=None):
    """Dynamic programming optimizer with multi-slot pitching.

    Each pitching role expands into multiple slots (SP1..SP5, RP1..RP6,
    CL). The DP iterates over every slot and picks at most one card per
    slot, with a global uniqueness constraint: the same market card_id
    can't fill two different slots in one pass.

    Returns the same dict shape as before; ``transactions[i]['position']``
    is now a slot label (SP3, RP2, ...) so the caller can show "upgrade
    SP3 from X to Y" instead of "upgrade SP from X to Y".
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    exclude_positions = set(exclude_positions or [])
    priority_positions = list(priority_positions or [])

    # Build ordered slot list (priority first, then the rest).
    remaining_pos = [p for p in ALL_POSITIONS
                     if p not in exclude_positions and p not in priority_positions]
    positions = ([p for p in priority_positions if p not in exclude_positions]
                 + remaining_pos)
    n_positions = len(positions)

    starters = _get_roster_starters(conn)

    # Owned-card guard: never let the optimizer "buy" a card that's already
    # on the roster as the current starter at the slot we're considering.
    owned_card_ids = {s['card_id'] for s in starters.values() if s.get('card_id')}

    BUCKET_SIZE = 50
    max_buckets = budget_pp // BUCKET_SIZE

    def price_to_buckets(price):
        return (price + BUCKET_SIZE - 1) // BUCKET_SIZE

    # Top-N candidates per slot, keyed by slot index. Pitching candidates
    # are role-pooled (every SP card is a candidate for any SPx slot)
    # — uniqueness during pick is enforced in backtrack.
    TOP_N = 12
    candidates_by_pos = []
    for pos in positions:
        current = starters.get(pos)
        current_meta = current['meta_score'] if current else 0
        current_player = current['player_name'] if current else '(empty)'

        raw = _get_upgrade_candidates(conn, pos, current_meta, budget_pp,
                                      excluded_card_ids=owned_card_ids)
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

    # DP with memoization. State now includes the set of card_ids already
    # picked, BUT we keep memoization on (pos_idx, remaining_buckets) only
    # — the picked-set is threaded as a runtime argument and we pessimize
    # by always trying the top-meta-gain card first per slot. With TOP_N=12
    # and 20 slots this stays tractable.
    memo: dict = {}

    def dp(pos_idx, remaining_buckets, picked_ids):
        if pos_idx >= n_positions:
            return 0.0
        # Memo key includes a frozen snapshot of picked_ids only when it
        # affects decisions for the remaining slots. Since pitching cards
        # can only fill pitching slots (and we group SP/RP/CL distinctly),
        # we can safely cache without picked_ids when the remaining slots
        # are all batting OR all pitching but distinct from prior picks.
        # To keep it simple and correct, key on (pos_idx, remaining_buckets,
        # frozenset of pitching picks only).
        pitching_picks = frozenset(
            cid for cid in picked_ids if cid in _ALL_PITCHING_CAND_IDS
        )
        key = (pos_idx, remaining_buckets, pitching_picks)
        if key in memo:
            return memo[key]

        best = dp(pos_idx + 1, remaining_buckets, picked_ids)

        for cand in candidates_by_pos[pos_idx]:
            if cand['card_id'] in picked_ids:
                continue
            if cand['buckets'] <= remaining_buckets:
                new_picked = picked_ids | {cand['card_id']}
                val = cand['meta_gain'] + dp(
                    pos_idx + 1,
                    remaining_buckets - cand['buckets'],
                    new_picked,
                )
                if val > best:
                    best = val

        memo[key] = best
        return best

    # Pre-compute the universe of pitching candidate ids for memo-key
    # narrowing (only pitching picks need to flow through the picked set).
    _ALL_PITCHING_CAND_IDS = set()
    for pos_idx, pos in enumerate(positions):
        if _slot_role(pos) in PITCHING_ROLE_TAGS:
            for c in candidates_by_pos[pos_idx]:
                _ALL_PITCHING_CAND_IDS.add(c['card_id'])

    initial_picked = frozenset()
    optimal_gain = dp(0, max_buckets, initial_picked)

    # Backtrack to recover the actual picks.
    transactions = []
    remaining_buckets = max_buckets
    picked_ids: set = set()
    for pos_idx in range(n_positions):
        pitching_picks_now = frozenset(
            cid for cid in picked_ids if cid in _ALL_PITCHING_CAND_IDS
        )
        full_val = dp(pos_idx, remaining_buckets, picked_ids)

        picked = False
        for cand in candidates_by_pos[pos_idx]:
            if cand['card_id'] in picked_ids:
                continue
            if cand['buckets'] <= remaining_buckets:
                future_pitching = (pitching_picks_now | {cand['card_id']}
                                    if cand['card_id'] in _ALL_PITCHING_CAND_IDS
                                    else pitching_picks_now)
                future = dp(
                    pos_idx + 1,
                    remaining_buckets - cand['buckets'],
                    picked_ids | {cand['card_id']},
                )
                val = cand['meta_gain'] + future
                if abs(val - full_val) < 1e-9:
                    transactions.append(cand)
                    remaining_buckets -= cand['buckets']
                    picked_ids.add(cand['card_id'])
                    picked = True
                    break

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


def _optimize_budget_greedy(budget_pp, conn, priority_positions=None,
                              exclude_positions=None):
    """Greedy fallback. Same multi-slot pitching as the DP path."""
    exclude_positions = set(exclude_positions or [])
    priority_positions = list(priority_positions or [])

    remaining_pos = [p for p in ALL_POSITIONS
                     if p not in exclude_positions and p not in priority_positions]
    ordered_positions = ([p for p in priority_positions if p not in exclude_positions]
                         + remaining_pos)

    starters = _get_roster_starters(conn)
    owned_card_ids = {s['card_id'] for s in starters.values() if s.get('card_id')}

    remaining = budget_pp
    transactions = []
    filled_positions = set()
    picked_card_ids: set = set()

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

            candidates = _get_upgrade_candidates(
                conn, pos, current_meta, remaining,
                excluded_card_ids=owned_card_ids | picked_card_ids,
            )
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
        picked_card_ids.add(best_candidate['card_id'])
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


def optimize_budget(budget_pp, conn=None, method='dp', priority_positions=None,
                     exclude_positions=None):
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
            - transactions: list of recommended buys (positions are SLOT
              labels — 'SP3', 'RP2', etc. — not just role tags)
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
        result = _optimize_budget_greedy(budget_pp, conn, priority_positions, exclude_positions)
        result['method'] = 'greedy (fallback)'

    # Free in-house bullpen role swaps (UAT Tier-2 #8). Surface alongside
    # the paid upgrades so the user sees "promote Kerkering to CL — free
    # +108 meta" before being asked to spend PP on a CL upgrade.
    try:
        result['role_reassignments'] = recommend_role_reassignments(conn)
    except Exception:
        result['role_reassignments'] = []

    if close_conn:
        conn.close()
    return result


def recommend_role_reassignments(conn=None) -> list[dict]:
    """Find free in-house bullpen role swaps that lift total pitching meta.

    UAT 2026-04-25 §5.2 / Tier-2 #8: the optimizer used to lock cards
    into whatever role they appeared under (CL stays CL, RPs stay RPs)
    and never suggest promoting the highest-meta reliever to closer.
    The user reported Kerkering (META 666) sitting in middle relief
    while Lovelady (META 558) was the closer — a free +108 meta the
    system never surfaced.

    This pre-pass looks at every owned RP/CL and pairs the highest-meta
    reliever-eligible card with the CL slot, the second-highest with
    SU/RP1, and so on. Returns a list of suggestions with the SAME
    dict shape as ``transactions`` so the page can render them in the
    Top Priority Moves panel under the "Free promotion" badge.

    Each suggestion is "swap card X (currently slot A, meta a) into
    slot B (currently filled by card Y, meta b)" — a zero-cost move.
    Only emitted when meta_gain >= 30 (smaller swaps are noise).
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        # Pull all owned reliever-eligible cards + their current slot fill.
        # COALESCE with per-league meta when an override is active so the
        # CL-promotion ranking reflects the user's selected tier.
        league = _active_league_for_meta()
        per_league_meta: dict[int, float] = {}
        if league:
            try:
                for r in conn.execute(
                    "SELECT card_id, meta_score FROM card_meta_by_league "
                    "WHERE league_id = ? AND side = 'pitching'",
                    (league,),
                ).fetchall():
                    per_league_meta[r['card_id']] = r['meta_score']
            except Exception:
                pass

        rp_pool = conn.execute("""
            SELECT rc.player_name, rc.position AS current_role,
                   rc.meta_score, rc.card_id, c.card_title,
                   c.pitcher_role_name
            FROM roster_current rc
            LEFT JOIN cards c ON c.card_id = rc.card_id
            WHERE rc.lineup_role IN ('rotation', 'closer', 'bullpen')
              AND rc.position IN ('RP', 'CL')
              AND rc.meta_score IS NOT NULL
        """).fetchall()
        if not rp_pool:
            return []

        # Build the ranking using per-league meta when present.
        rp_dicts = []
        for r in rp_pool:
            d = dict(r)
            cid = d.get('card_id')
            d['_rank_meta'] = (per_league_meta.get(cid)
                               if cid is not None and cid in per_league_meta
                               else (d.get('meta_score') or 0))
            rp_dicts.append(d)

        sorted_pool = sorted(
            rp_dicts,
            key=lambda r: r.get('_rank_meta') or 0,
            reverse=True,
        )
        ideal_assignments = {}  # slot -> card dict
        slot_order = ['CL'] + PITCHING_SLOTS_BY_ROLE['RP']  # CL gets best, then RP1..RP6
        for slot, p in zip(slot_order, sorted_pool):
            ideal_assignments[slot] = p

        # Compare against current. The CL slot's *current* fill is the
        # highest-meta card whose ``current_role`` is 'CL'; others are RP.
        cur_cl = next(
            (p for p in sorted_pool if p['current_role'] == 'CL'),
            None,
        )
        suggestions = []
        if 'CL' in ideal_assignments and cur_cl:
            ideal_cl = ideal_assignments['CL']
            if ideal_cl['card_id'] != cur_cl['card_id']:
                # Use per-league meta for the gain when active so the
                # "+N meta" matches what the user sees in the page.
                cur_meta = cur_cl.get('_rank_meta', cur_cl.get('meta_score') or 0)
                ideal_meta = ideal_cl.get('_rank_meta', ideal_cl.get('meta_score') or 0)
                gain = ideal_meta - cur_meta
                if gain >= 30:
                    suggestions.append({
                        'kind': 'role_promotion',
                        'card_id': ideal_cl['card_id'],
                        'card_title': ideal_cl.get('card_title') or ideal_cl['player_name'],
                        'position': 'CL',
                        'current_player': cur_cl['player_name'],
                        'current_meta': cur_meta,
                        'new_meta': ideal_meta,
                        'meta_gain': gain,
                        'price': 0,
                        'note': (
                            f"Promote {ideal_cl['player_name']} from "
                            f"{ideal_cl['current_role']} to CL "
                            f"(free, +{gain:.0f} meta"
                            + (f" · per-{league}" if league else "")
                            + ")"
                        ),
                    })

        return suggestions
    finally:
        if close_conn:
            conn.close()


def simulate_transactions(buys, sells, conn=None):
    """What-if sandbox: simulate buying and selling specific cards.

    Multi-slot aware: a buy aligned to SP/RP/CL will replace the lowest-meta
    current slot fill if the buy's meta exceeds it, mirroring how the DP
    knapsack would assign the upgrade.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    starters = _get_roster_starters(conn)

    roster_before = {}
    for slot in ALL_POSITIONS:
        current = starters.get(slot)
        roster_before[slot] = current['meta_score'] if current else 0

    total_meta_before = sum(roster_before.values())

    # Fetch buy details, group by role.
    buy_details = []
    buy_cost = 0
    buy_by_role: dict[str, list[float]] = {}
    for card_id in buys:
        row = conn.execute("""
            SELECT card_id, card_title, position_name, pitcher_role_name,
                   meta_score_batting, meta_score_pitching,
                   last_10_price, sell_order_low
            FROM cards WHERE card_id = ?
        """, (card_id,)).fetchone()
        if not row:
            continue

        is_pitching = row['pitcher_role_name'] in PITCHING_ROLE_TAGS
        role = row['pitcher_role_name'] if is_pitching else row['position_name']
        meta = row['meta_score_pitching'] if is_pitching else row['meta_score_batting']
        price = row['last_10_price'] or row['sell_order_low'] or 0

        buy_details.append({
            'card_id': row['card_id'],
            'card_title': row['card_title'],
            'position': role,
            'meta_score': meta or 0,
            'price': price,
        })
        buy_cost += price
        buy_by_role.setdefault(role, []).append(meta or 0)

    sell_details = []
    sell_revenue = 0
    sell_card_ids: set = set()
    for card_id in sells:
        row = conn.execute("""
            SELECT card_id, card_title, position_name, pitcher_role_name,
                   meta_score_batting, meta_score_pitching,
                   buy_order_high, last_10_price
            FROM cards WHERE card_id = ?
        """, (card_id,)).fetchone()
        if not row:
            continue

        is_pitching = row['pitcher_role_name'] in PITCHING_ROLE_TAGS
        role = row['pitcher_role_name'] if is_pitching else row['position_name']
        meta = row['meta_score_pitching'] if is_pitching else row['meta_score_batting']
        revenue = row['buy_order_high'] or row['last_10_price'] or 0

        sell_details.append({
            'card_id': row['card_id'],
            'card_title': row['card_title'],
            'position': role,
            'meta_score': meta or 0,
            'revenue': revenue,
        })
        sell_revenue += revenue
        sell_card_ids.add(row['card_id'])

    # Project roster_after — apply sells (drop slots whose card_id matches a
    # sell), then apply buys (replace lowest-meta slot in role if buy > slot).
    roster_after = dict(roster_before)

    for slot, meta in list(roster_after.items()):
        cur = starters.get(slot) or {}
        if cur.get('card_id') in sell_card_ids:
            roster_after[slot] = 0

    for role, buys_meta_list in buy_by_role.items():
        if role in BATTING_POSITIONS:
            slots = [role]
        else:
            slots = PITCHING_SLOTS_BY_ROLE.get(role, [role])
        # For each buy, place it in the slot with lowest current meta if
        # the buy's meta beats that slot.
        sorted_buys = sorted(buys_meta_list, reverse=True)
        for buy_meta in sorted_buys:
            # find the slot with the smallest current meta
            slot_metas = [(s, roster_after.get(s, 0)) for s in slots]
            slot_metas.sort(key=lambda x: x[1])
            target_slot, cur_meta = slot_metas[0]
            if buy_meta > cur_meta:
                roster_after[target_slot] = buy_meta

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
