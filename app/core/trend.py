"""Player trend/momentum computation from player_history.

Surfaces two signals that aren't captured by the current meta:

1. **Meta trend** — did this card's meta go up or down across snapshots?
   A rising meta usually means OOTP upgraded the underlying Live card
   (hidden attribute improvement, new splits). A falling meta means the
   card got downgraded or aged. This is invisible to the current static
   meta number.

2. **Performance trend** — is this player's WAR / OPS+ / ERA+ getting
   better or worse across snapshots? This is cumulative-stats delta
   (today vs N snapshots ago), so a player who was league-average early
   and is now torrid shows a big positive trend even if their cumulative
   numbers still look average.

Both are per-card features derived from ``player_history`` with no
changes to the meta formula itself. The UI can surface them as arrows
(↑/↓) alongside meta so the user sees momentum.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

from app.core.database import get_connection, load_config


def _active_league(conn) -> str:
    try:
        return load_config().get('active_league', 'lb124')
    except Exception:
        return 'lb124'


def player_trend(card_id: int, league_id: Optional[str] = None,
                 window: int = 3, conn=None) -> dict:
    """Compute trend metrics for a single card over its last ``window`` snapshots.

    Returns::

        {
            'card_id': 12345,
            'snapshots_used': 3,
            'meta_delta': +12.5,        # latest - earliest meta_score
            'meta_slope': +4.2,         # per-snapshot slope
            'ops_delta': +0.045,        # stat trend (batters)
            'ops_plus_delta': +8,
            'war_delta': +0.6,
            'era_delta': -0.40,         # lower = better (pitchers)
            'era_plus_delta': +12,
            'direction': 'up' | 'down' | 'flat',
            'signal': 'hot' | 'cold' | 'stable' | 'insufficient',
        }

    ``signal`` summarizes the overall trend — 'hot' means metrics are
    trending favorably (meta up, stats up for batters / down for ERA).
    """
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    try:
        league_id = league_id or _active_league(conn)
        # Filter to snapshots with stabilized sample sizes (PA>=50 or IP>=15).
        # Early-season snapshots with tiny samples produce wild rate-stat
        # swings (OPS+ swings of 200+ points, ERA+ swings of 900+) that
        # swamp any real trend signal. Only include snapshots past that
        # threshold so the delta reflects meaningful changes.
        rows = conn.execute(
            """SELECT snapshot_date, export_number, meta_score, ops, ops_plus, war,
                      era, era_plus, p_war, pa, ip
               FROM player_history
               WHERE card_id = ? AND league_id = ?
                 AND (
                     (pa IS NOT NULL AND pa >= 50)
                     OR (ip IS NOT NULL AND ip >= 15)
                     OR (pa IS NULL AND ip IS NULL)
                 )
               ORDER BY export_number DESC, snapshot_date DESC
               LIMIT ?""",
            (card_id, league_id, window),
        ).fetchall()
        if len(rows) < 2:
            return {
                'card_id': card_id, 'snapshots_used': len(rows),
                'direction': 'flat', 'signal': 'insufficient',
            }
        # rows[0] is latest, rows[-1] is earliest in window
        latest, earliest = rows[0], rows[-1]

        def _f(x):
            try:
                return float(x) if x is not None else None
            except (ValueError, TypeError):
                return None

        def _delta(key):
            a = _f(latest[key])
            b = _f(earliest[key])
            if a is None or b is None:
                return None
            return a - b

        n = len(rows)
        meta_delta = _delta('meta_score')
        meta_slope = None
        if n >= 2:
            metas = [(i, _f(r['meta_score'])) for i, r in enumerate(reversed(rows))]
            metas = [(i, v) for i, v in metas if v is not None]
            if len(metas) >= 2:
                # Simple slope: (last - first) / (n - 1)
                meta_slope = (metas[-1][1] - metas[0][1]) / (len(metas) - 1)

        ops_delta = _delta('ops')
        ops_plus_delta = _delta('ops_plus')
        war_delta = _delta('war')
        era_delta = _delta('era')
        era_plus_delta = _delta('era_plus')
        p_war_delta = _delta('p_war')

        # Classify from STATS, not meta. Meta deltas are unreliable across
        # formula revisions (the current meta formula is v6b; earlier history
        # snapshots used v5/v4 with different OVR and weight behavior).
        # Performance-stat deltas are stable across formula changes.
        signal = 'stable'
        bat_up = (ops_plus_delta or 0) >= 8 or (war_delta or 0) >= 0.5
        bat_down = (ops_plus_delta or 0) <= -8 or (war_delta or 0) <= -0.5
        pit_up = (era_plus_delta or 0) >= 8 or (p_war_delta or 0) >= 0.5
        pit_down = (era_plus_delta or 0) <= -8 or (p_war_delta or 0) <= -0.5
        if bat_up or pit_up:
            signal = 'hot'
        elif bat_down or pit_down:
            signal = 'cold'

        # Direction mirrors the signal so the UI can show arrows consistently.
        direction = {'hot': 'up', 'cold': 'down'}.get(signal, 'flat')

        return {
            'card_id': card_id,
            'snapshots_used': n,
            'meta_delta': round(meta_delta, 1) if meta_delta is not None else None,
            'meta_slope': round(meta_slope, 1) if meta_slope is not None else None,
            'ops_delta': round(ops_delta, 3) if ops_delta is not None else None,
            'ops_plus_delta': round(ops_plus_delta, 1) if ops_plus_delta is not None else None,
            'war_delta': round(war_delta, 2) if war_delta is not None else None,
            'era_delta': round(era_delta, 2) if era_delta is not None else None,
            'era_plus_delta': round(era_plus_delta, 1) if era_plus_delta is not None else None,
            'p_war_delta': round(p_war_delta, 2) if p_war_delta is not None else None,
            'direction': direction,
            'signal': signal,
        }
    finally:
        if close:
            conn.close()


def roster_trends(window: int = 3, conn=None) -> dict[int, dict]:
    """Compute trend for every card on the active roster.

    Returns a dict keyed by ``card_id``. Rows without a card_id are skipped.
    Only considers the active league.
    """
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    try:
        league_id = _active_league(conn)
        rows = conn.execute(
            """SELECT DISTINCT card_id FROM roster_current
               WHERE card_id IS NOT NULL
                 AND lineup_role IN ('starter','rotation','closer','bullpen','bench','reserve')"""
        ).fetchall()
        out = {}
        for r in rows:
            cid = r[0]
            if cid is None:
                continue
            out[int(cid)] = player_trend(int(cid), league_id=league_id,
                                         window=window, conn=conn)
        return out
    finally:
        if close:
            conn.close()


def summarize_trends(trends: dict[int, dict]) -> dict:
    """Summary stats for a trends dict. Useful for the roster overview."""
    hot = [t for t in trends.values() if t.get('signal') == 'hot']
    cold = [t for t in trends.values() if t.get('signal') == 'cold']
    insufficient = [t for t in trends.values() if t.get('signal') == 'insufficient']
    stable = [t for t in trends.values() if t.get('signal') == 'stable']
    meta_ups = [t for t in trends.values() if t.get('direction') == 'up']
    meta_downs = [t for t in trends.values() if t.get('direction') == 'down']
    return {
        'total': len(trends),
        'hot': len(hot),
        'cold': len(cold),
        'stable': len(stable),
        'insufficient': len(insufficient),
        'meta_trending_up': len(meta_ups),
        'meta_trending_down': len(meta_downs),
    }
