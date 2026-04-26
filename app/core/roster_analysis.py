"""Roster analysis — gap detection, strength mapping, lineup optimization."""
from app.core.database import get_connection


# Slots known to be platoon-able. The OOTP CSV export emits separate
# ``vs_lhp`` and ``vs_rhp`` rows for these positions when the user has
# a platoon pair set; the dedup step needs to keep BOTH halves of a
# legitimate platoon.
PLATOON_POSITIONS = ('C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH')


def get_position_starters(conn=None, platoon_aware: bool = True) -> dict:
    """Single source of truth for "who starts at each position" (UAT Tier-2 #9 / #12).

    The optimizer used to call ``_get_roster_starters`` in two places —
    one in ``core/optimizer.py``, one inline on the page — and both
    de-duped multi-row positions to a single max-meta entry. That
    collapsed legitimate platoon pairs (vs-RHP starter + vs-LHP
    starter) into one starter + one "bench but better than the
    starter" alert.

    This helper returns ``{position: [list_of_starters]}`` where each
    list has 1 entry for non-platoon slots and 2 for platoon pairs,
    distinguishable by ``bats``. Callers that need the "best single
    starter at this position" can take ``max(list, key=meta)``;
    callers that want platoon-aware comparisons can iterate the list
    and use ``meta_vs_rhp`` / ``meta_vs_lhp`` accordingly.

    ``platoon_aware``:
        True  — keep both halves of a vs-RHP/vs-LHP pair (default).
        False — collapse to one max-meta entry per position (legacy).
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        rows = conn.execute("""
            SELECT player_name, position, lineup_role, ovr, meta_score,
                   meta_vs_rhp, meta_vs_lhp, bats, card_id, card_title
            FROM roster_current
            WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
        """).fetchall()
    finally:
        if close_conn:
            conn.close()

    by_pos: dict[str, list[dict]] = {}
    for r in rows:
        d = dict(r)
        pos = d['position']
        by_pos.setdefault(pos, []).append(d)

    if not platoon_aware:
        # Legacy mode — collapse to one per position by max meta.
        collapsed = {}
        for pos, players in by_pos.items():
            best = max(players, key=lambda p: p.get('meta_score') or 0)
            collapsed[pos] = [best]
        return collapsed

    # Platoon-aware: when a position has multiple starter rows, keep
    # rows whose ``bats`` differ — that's the signal of a real platoon
    # pair (vs-RHP card bats L, vs-LHP card bats R, or some combination
    # plus switch-hitters). Rows with identical ``bats`` collapse to
    # the max-meta one (true duplicates from CSV import quirks).
    resolved: dict[str, list[dict]] = {}
    for pos, players in by_pos.items():
        if pos not in PLATOON_POSITIONS or len(players) <= 1:
            best = max(players, key=lambda p: p.get('meta_score') or 0)
            resolved[pos] = [best]
            continue

        # Group by bats (None counts as 'unknown')
        by_bats: dict[str, dict] = {}
        for p in players:
            b = (p.get('bats') or '').upper() or 'unknown'
            cur = by_bats.get(b)
            if cur is None or (p.get('meta_score') or 0) > (cur.get('meta_score') or 0):
                by_bats[b] = p

        # If only one bats group survived, this isn't a platoon — just
        # one starter with possible duplicate rows. Keep the max.
        if len(by_bats) <= 1:
            resolved[pos] = [list(by_bats.values())[0]]
        else:
            # Real platoon pair — keep both halves.
            resolved[pos] = list(by_bats.values())
    return resolved


def get_roster_summary(conn=None):
    """Get current roster with meta scores by position."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    rows = conn.execute("""
        SELECT player_name, position, lineup_role, ovr, meta_score
        FROM roster_current
        ORDER BY
            CASE lineup_role
                WHEN 'starter' THEN 1 WHEN 'rotation' THEN 2
                WHEN 'closer' THEN 3 WHEN 'bullpen' THEN 4
                WHEN 'bench' THEN 5 WHEN 'reserve' THEN 6
            END,
            meta_score DESC
    """).fetchall()

    if close_conn:
        conn.close()
    return rows


def get_position_strength(conn=None, platoon_aware: bool = True):
    """Analyze roster strength by position. Returns dict of position -> info.

    UAT Tier-2 #9: when ``platoon_aware`` is True (default), platoon
    pairs are reported as a single "platoon" entry showing both halves
    rather than collapsing one half into bench-mistake-alert noise.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    roster = conn.execute("""
        SELECT player_name, position, lineup_role, ovr, meta_score
        FROM roster_current WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
    """).fetchall()

    positions = {}
    for r in roster:
        pos = r['position']
        if pos not in positions:
            positions[pos] = []
        positions[pos].append(dict(r))

    # Calculate strength rating for each position
    result = {}
    all_metas = [r['meta_score'] for r in roster if r['meta_score']]
    avg_meta = sum(all_metas) / len(all_metas) if all_metas else 0

    starters_resolved = get_position_starters(conn, platoon_aware=platoon_aware)

    for pos, players in positions.items():
        starters_here = starters_resolved.get(pos) or []
        if len(starters_here) >= 2:
            # Platoon pair — describe both halves but keep the higher-meta
            # card as the "primary" for legacy callers reading the dict.
            primary = max(starters_here, key=lambda p: p.get('meta_score') or 0)
            partner = next(
                (p for p in starters_here if p.get('player_name') != primary.get('player_name')),
                None,
            )
            result[pos] = {
                'player': primary['player_name'],
                'ovr': primary['ovr'],
                'meta_score': primary['meta_score'],
                'depth': len(players),
                'strength': 'strong' if primary['meta_score'] and primary['meta_score'] > avg_meta * 1.1
                            else ('average' if primary['meta_score'] and primary['meta_score'] > avg_meta * 0.9
                                  else 'weak'),
                'is_platoon': True,
                'platoon_partner': partner['player_name'] if partner else None,
                'platoon_partner_meta': (partner or {}).get('meta_score'),
                'platoon_partner_bats': (partner or {}).get('bats'),
            }
        else:
            best = starters_here[0] if starters_here else max(players, key=lambda x: x['meta_score'] or 0)
            result[pos] = {
                'player': best['player_name'],
                'ovr': best['ovr'],
                'meta_score': best['meta_score'],
                'depth': len(players),
                'strength': 'strong' if best['meta_score'] and best['meta_score'] > avg_meta * 1.1
                            else ('average' if best['meta_score'] and best['meta_score'] > avg_meta * 0.9
                                  else 'weak'),
                'is_platoon': False,
            }

    # Check for missing positions
    # DH excluded — any batter can DH
    expected_batting = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
    expected_pitching = ['SP', 'RP', 'CL']
    for pos in expected_batting + expected_pitching:
        if pos not in result:
            result[pos] = {
                'player': None, 'ovr': 0, 'meta_score': 0,
                'depth': 0, 'strength': 'empty'
            }

    if close_conn:
        conn.close()
    return result


def get_best_available_by_position(position: str, limit: int = 10, conn=None):
    """Get best available cards on market for a given position."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    if position in ('SP', 'RP', 'CL'):
        rows = conn.execute("""
            SELECT card_id, card_title, pitcher_role_name as pos, tier_name,
                   meta_score_pitching as meta_score, last_10_price, sell_order_low
            FROM cards
            WHERE pitcher_role_name = ? AND owned = 0 AND last_10_price > 0
            ORDER BY meta_score_pitching DESC
            LIMIT ?
        """, (position, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT card_id, card_title, position_name as pos, tier_name,
                   meta_score_batting as meta_score, last_10_price, sell_order_low
            FROM cards
            WHERE position_name = ? AND owned = 0 AND last_10_price > 0
            ORDER BY meta_score_batting DESC
            LIMIT ?
        """, (position, limit)).fetchall()

    if close_conn:
        conn.close()
    return rows


def get_collection_by_position(conn=None):
    """Get all owned cards grouped by position."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    rows = conn.execute("""
        SELECT player_name, position, ovr, status, meta_score
        FROM collection_current
        ORDER BY position, meta_score DESC
    """).fetchall()

    if close_conn:
        conn.close()

    by_pos = {}
    for r in rows:
        pos = r['position']
        if pos not in by_pos:
            by_pos[pos] = []
        by_pos[pos].append(dict(r))
    return by_pos
