"""Roster analysis — gap detection, strength mapping, lineup optimization."""
from app.core.database import get_connection


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


def get_position_strength(conn=None):
    """Analyze roster strength by position. Returns dict of position -> info."""
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

    for pos, players in positions.items():
        best = max(players, key=lambda x: x['meta_score'] or 0)
        result[pos] = {
            'player': best['player_name'],
            'ovr': best['ovr'],
            'meta_score': best['meta_score'],
            'depth': len(players),
            'strength': 'strong' if best['meta_score'] and best['meta_score'] > avg_meta * 1.1
                        else ('average' if best['meta_score'] and best['meta_score'] > avg_meta * 0.9
                              else 'weak'),
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
