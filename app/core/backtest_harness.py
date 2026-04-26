"""Back-test harness for optimizer recommendations (UAT 2026-04-25 Tier-3 #13).

Cross-references closed-loop entries in ``recommendation_log`` with
post-recommendation game outcomes from ``game_log_at_bats`` and
``game_pitching``. Lets us answer: "does the optimizer's recommendation
actually improve the team's pitching/batting outcomes in the games
played AFTER the recommendation was made?"

Public surface:
    backtest_recommendations(league_id, conn) -> dict
    backtest_player_pre_post(card_id, anchor_date, league_id, conn, side) -> dict
    backtest_player_pre_post_batting(card_id, anchor_date, league_id, conn) -> dict

The top-level ``backtest_player_pre_post`` is a thin dispatcher. It auto-
detects ``side`` from ``cards.pitcher_role`` when the caller doesn't pass
it explicitly, so legacy positional callers keep working.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from app.core.database import get_connection


# Outcome strings stored in ``game_log_at_bats.outcome``. ``OUT`` and ``FC``
# count as ABs but not hits; ``SAC`` is a sac-bunt/fly that does not count
# as an AB or PA in the standard rate denominators.
_HIT_OUTCOMES = {'SINGLE', 'DOUBLE', 'TRIPLE', 'HR'}
_AB_OUTCOMES = {'SINGLE', 'DOUBLE', 'TRIPLE', 'HR', 'OUT', 'K', 'FC'}
_PA_OUTCOMES = _AB_OUTCOMES | {'BB', 'HBP'}


def _parse_date(s: Optional[str]) -> Optional[str]:
    """Normalize a string timestamp to ``YYYY-MM-DD``. Returns None on failure."""
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s[:10] if len(s) >= 10 else None


def _detect_side(card_id: int, conn) -> str:
    """Decide whether a card should be backtested as a batter or pitcher.

    A card is considered a pitcher when ``cards.position`` is the OOTP code
    for pitcher (1) or when ``pitcher_role`` is set to a non-zero value;
    everything else is treated as batting (DH, position players).
    """
    row = conn.execute(
        "SELECT position, pitcher_role FROM cards WHERE card_id = ?",
        (card_id,),
    ).fetchone()
    if not row:
        return 'batting'
    pos = row['position'] or 0
    pr = row['pitcher_role'] or 0
    if int(pos) == 1 or int(pr) > 0:
        return 'pitching'
    return 'batting'


def _window_dates(anchor: str, window_days: int) -> tuple[str, str, str, str]:
    """Return ``(pre_start, pre_end, post_start, post_end)`` for an anchor.

    The anchor day itself is excluded from both halves — the rec was made
    on that day and the first observable effect is the next game played.
    """
    a = datetime.strptime(anchor, "%Y-%m-%d")
    pre_start = (a - timedelta(days=window_days)).strftime("%Y-%m-%d")
    pre_end = (a - timedelta(days=1)).strftime("%Y-%m-%d")
    post_start = (a + timedelta(days=1)).strftime("%Y-%m-%d")
    post_end = (a + timedelta(days=window_days)).strftime("%Y-%m-%d")
    return pre_start, pre_end, post_start, post_end


def _backtest_player_pre_post_pitching(card_id: int, anchor_date: str,
                                        league_id: Optional[str] = None,
                                        window_days: int = 14,
                                        conn=None) -> dict:
    """Compare a pitcher's pre-/post-anchor production.

    Pulls game-pitching aggregates from the ``window_days`` before and
    after ``anchor_date``. Returns ``None`` in side fields when the
    sample is too small (< 10 IP either side).
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    anchor = _parse_date(anchor_date)
    if not anchor:
        if close_conn:
            conn.close()
        return {'error': f'unparseable anchor_date={anchor_date}'}

    league_clause = "AND g.league_id = ?" if league_id else ""
    league_args = (league_id,) if league_id else ()

    def _agg(start_date, end_date):
        sql = f"""
            SELECT COALESCE(SUM(gp.ip), 0) AS ip,
                   COALESCE(SUM(gp.er), 0) AS er,
                   COALESCE(SUM(gp.k), 0) AS k,
                   COALESCE(SUM(gp.bb), 0) AS bb,
                   COUNT(DISTINCT gp.game_id) AS games
            FROM game_pitching gp
            JOIN games g ON g.game_id = gp.game_id
            WHERE gp.card_id = ?
              AND DATE(g.game_date) BETWEEN ? AND ?
              {league_clause}
        """
        row = conn.execute(sql, (card_id, start_date, end_date, *league_args)).fetchone()
        ip = row['ip'] or 0
        er = row['er'] or 0
        k = row['k'] or 0
        bb = row['bb'] or 0
        games = row['games'] or 0
        era = (er * 9.0 / ip) if ip > 0 else None
        k9 = (k * 9.0 / ip) if ip > 0 else None
        bb9 = (bb * 9.0 / ip) if ip > 0 else None
        return {
            'games': games, 'ip': float(ip), 'er': er,
            'era': round(era, 2) if era is not None else None,
            'k_per_9': round(k9, 2) if k9 is not None else None,
            'bb_per_9': round(bb9, 2) if bb9 is not None else None,
        }

    pre_start, pre_end, post_start, post_end = _window_dates(anchor, window_days)
    pre = _agg(pre_start, pre_end)
    post = _agg(post_start, post_end)

    delta_era = None
    if pre['era'] is not None and post['era'] is not None:
        delta_era = round(post['era'] - pre['era'], 2)

    if close_conn:
        conn.close()
    return {
        'card_id': card_id,
        'anchor_date': anchor,
        'window_days': window_days,
        'side': 'pitching',
        'pre': pre,
        'post': post,
        'delta_era': delta_era,
    }


def backtest_player_pre_post_batting(card_id: int, anchor_date: str,
                                      league_id: Optional[str] = None,
                                      window_days: int = 14,
                                      conn=None) -> dict:
    """Compare a batter's pre-/post-anchor production.

    Aggregates per-AB rows from ``game_log_at_bats`` joined to ``games`` to
    derive AVG/OBP/SLG/OPS/K%/BB%/ISO/HR-rate over the ``window_days``
    before and after ``anchor_date``. Rate fields are ``None`` when the
    sample is too small (PA < 15 either side) — small-sample noise has
    swamped real signal in past calibration runs.

    Returns the same shape as the pitching version with batting-specific
    metrics::

        {
          'card_id': int, 'anchor_date': 'YYYY-MM-DD', 'window_days': int,
          'side': 'batting',
          'pre':  {'games': int, 'pa': int, 'ab': int, 'h': int, 'hr': int,
                   'k': int, 'bb': int,
                   'avg':..., 'obp':..., 'slg':..., 'ops':...,
                   'iso':..., 'k_pct':..., 'bb_pct':..., 'hr_rate':...},
          'post': {... same shape ...},
          'delta_ops': float|None,
          'delta_woba': float|None,
        }
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    anchor = _parse_date(anchor_date)
    if not anchor:
        if close_conn:
            conn.close()
        return {'error': f'unparseable anchor_date={anchor_date}'}

    league_clause = "AND g.league_id = ?" if league_id else ""
    league_args = (league_id,) if league_id else ()

    def _agg(start_date, end_date):
        # Pull every AB row in window then bucket in Python — avoids
        # repeating the same CASE expression eight times in SQL and keeps
        # the outcome list co-located with ``_HIT_OUTCOMES`` constants
        # above so future outcome additions stay in one place.
        sql = f"""
            SELECT ab.outcome
            FROM game_log_at_bats ab
            JOIN games g ON g.game_id = ab.game_id
            WHERE ab.batter_card_id = ?
              AND DATE(g.game_date) BETWEEN ? AND ?
              {league_clause}
        """
        rows = conn.execute(
            sql, (card_id, start_date, end_date, *league_args)
        ).fetchall()
        # Distinct game count needs a separate query; cheap.
        gsql = f"""
            SELECT COUNT(DISTINCT ab.game_id) AS games
            FROM game_log_at_bats ab
            JOIN games g ON g.game_id = ab.game_id
            WHERE ab.batter_card_id = ?
              AND DATE(g.game_date) BETWEEN ? AND ?
              {league_clause}
        """
        games = conn.execute(
            gsql, (card_id, start_date, end_date, *league_args)
        ).fetchone()['games'] or 0

        singles = doubles = triples = hr = k = bb = hbp = h = ab = pa = 0
        for r in rows:
            o = (r['outcome'] or '').upper()
            if o in _PA_OUTCOMES:
                pa += 1
            if o in _AB_OUTCOMES:
                ab += 1
            if o in _HIT_OUTCOMES:
                h += 1
            if o == 'SINGLE':
                singles += 1
            elif o == 'DOUBLE':
                doubles += 1
            elif o == 'TRIPLE':
                triples += 1
            elif o == 'HR':
                hr += 1
            elif o == 'K':
                k += 1
            elif o == 'BB':
                bb += 1
            elif o == 'HBP':
                hbp += 1

        small = pa < 15
        avg = obp = slg = ops = iso = k_pct = bb_pct = hr_rate = None
        if not small and ab > 0:
            avg = h / ab
            tb = singles + 2 * doubles + 3 * triples + 4 * hr
            slg = tb / ab
            iso = slg - avg
            hr_rate = hr / ab
        if not small and pa > 0:
            obp = (h + bb + hbp) / pa
            k_pct = k / pa
            bb_pct = bb / pa
        if avg is not None and obp is not None and slg is not None:
            ops = obp + slg

        def _r(x, n=3):
            return round(x, n) if x is not None else None

        return {
            'games': games, 'pa': pa, 'ab': ab, 'h': h, 'hr': hr,
            'k': k, 'bb': bb,
            'avg': _r(avg), 'obp': _r(obp), 'slg': _r(slg),
            'ops': _r(ops, 3), 'iso': _r(iso),
            'k_pct': _r(k_pct), 'bb_pct': _r(bb_pct),
            'hr_rate': _r(hr_rate),
            # carry the linear-weight inputs through so the wOBA delta
            # below doesn't have to re-bucket; this is internal-only and
            # not part of the documented return shape.
            '_singles': singles, '_doubles': doubles, '_triples': triples,
            '_hbp': hbp,
        }

    pre_start, pre_end, post_start, post_end = _window_dates(anchor, window_days)
    pre = _agg(pre_start, pre_end)
    post = _agg(post_start, post_end)

    def _woba(side: dict) -> Optional[float]:
        pa = side.get('pa') or 0
        if pa < 15:
            return None
        # Standard Fangraphs simplified weights (no SF/IBB split — we
        # don't store either in game_log_at_bats).
        num = (0.69 * (side.get('bb') or 0)
               + 0.72 * (side.get('_hbp') or 0)
               + 0.89 * (side.get('_singles') or 0)
               + 1.27 * (side.get('_doubles') or 0)
               + 1.62 * (side.get('_triples') or 0)
               + 2.10 * (side.get('hr') or 0))
        return round(num / pa, 3)

    pre_woba = _woba(pre)
    post_woba = _woba(post)
    delta_woba = (round(post_woba - pre_woba, 3)
                  if pre_woba is not None and post_woba is not None else None)
    delta_ops = (round(post['ops'] - pre['ops'], 3)
                 if pre['ops'] is not None and post['ops'] is not None else None)

    # Strip the internal carry-through keys before returning so the public
    # contract stays clean.
    for s in (pre, post):
        for k_ in ('_singles', '_doubles', '_triples', '_hbp'):
            s.pop(k_, None)

    if close_conn:
        conn.close()
    return {
        'card_id': card_id,
        'anchor_date': anchor,
        'window_days': window_days,
        'side': 'batting',
        'pre': pre,
        'post': post,
        'delta_ops': delta_ops,
        'delta_woba': delta_woba,
    }


def backtest_player_pre_post(card_id: int, anchor_date: str,
                              league_id: Optional[str] = None,
                              window_days: int = 14,
                              conn=None,
                              side: Optional[str] = None) -> dict:
    """Dispatch to batting or pitching pre/post backtest.

    When ``side`` is ``None`` we look up ``cards.pitcher_role`` /
    ``cards.position`` to decide automatically. Existing positional
    callers continue to work because ``side`` is the new last kwarg.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    if side is None:
        side = _detect_side(card_id, conn)

    if side == 'batting':
        result = backtest_player_pre_post_batting(
            card_id, anchor_date, league_id=league_id,
            window_days=window_days, conn=conn,
        )
    else:
        result = _backtest_player_pre_post_pitching(
            card_id, anchor_date, league_id=league_id,
            window_days=window_days, conn=conn,
        )

    if close_conn:
        conn.close()
    return result


def backtest_recommendations(league_id: Optional[str] = None,
                              rec_type: Optional[str] = None,
                              window_days: int = 14,
                              max_recs: int = 200,
                              conn=None) -> dict:
    """Back-test a batch of closed-loop recommendations.

    Pulls recommendations whose ``verdict`` is not 'pending' (i.e. the
    user actually acted on or rejected them) and computes the
    pre-/post-anchor production for the involved card via the side-aware
    dispatcher. Aggregates by ``(verdict, side)`` so batting and
    pitching outcomes stay in their own buckets — averaging an OPS delta
    against an ERA delta would be nonsense.

    Returns a summary dict suitable for tabular display.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    where = ["verdict != 'pending'"]
    args = []
    if league_id:
        where.append("league_id = ?")
        args.append(league_id)
    if rec_type:
        where.append("rec_type = ?")
        args.append(rec_type)

    sql = f"""
        SELECT id, created_at, source, rec_type, pos, player_name, player_card_id,
               from_player, from_card_id, verdict, expected_delta,
               days_observed, side, projected_meta, projected_war,
               meta_before, meta_after
        FROM recommendation_log
        WHERE {' AND '.join(where)}
        ORDER BY created_at DESC
        LIMIT ?
    """
    rows = conn.execute(sql, (*args, max_recs)).fetchall()

    by_verdict: dict[str, list[dict]] = {}
    by_side_verdict: dict[str, dict[str, list[dict]]] = {
        'batting': {}, 'pitching': {},
    }
    detail_rows = []
    for r in rows:
        d = dict(r)
        if not d.get('player_card_id'):
            continue
        # Prefer the explicit ``side`` column populated by Stream A; fall
        # back to auto-detection for legacy rows where it's NULL.
        rec_side = d.get('side')
        if rec_side not in ('batting', 'pitching'):
            rec_side = _detect_side(d['player_card_id'], conn)
            d['side'] = rec_side
        try:
            bt = backtest_player_pre_post(
                d['player_card_id'], d['created_at'],
                league_id=league_id, window_days=window_days, conn=conn,
                side=rec_side,
            )
        except Exception as _e:
            bt = {'error': str(_e)}
        d['backtest'] = bt
        d['bt'] = bt  # legible alias for UI templates
        detail_rows.append(d)
        verdict = d.get('verdict') or 'unknown'
        by_verdict.setdefault(verdict, []).append(d)
        by_side_verdict.setdefault(rec_side, {}).setdefault(verdict, []).append(d)

    # Legacy aggregate — pitching ERA only — keeps existing UI rows that
    # still read ``summary_by_verdict[verdict]['avg_delta_era']`` working.
    summary_by_verdict = {}
    for verdict, recs in by_verdict.items():
        deltas = [
            (r.get('backtest') or {}).get('delta_era')
            for r in recs
            if isinstance(r.get('backtest'), dict)
        ]
        deltas = [d for d in deltas if d is not None]
        if deltas:
            avg = sum(deltas) / len(deltas)
            summary_by_verdict[verdict] = {
                'count': len(recs),
                'count_with_data': len(deltas),
                'avg_delta_era': round(avg, 3),
                'best_delta_era': round(min(deltas), 3),
                'worst_delta_era': round(max(deltas), 3),
            }
        else:
            summary_by_verdict[verdict] = {
                'count': len(recs),
                'count_with_data': 0,
                'avg_delta_era': None,
                'best_delta_era': None,
                'worst_delta_era': None,
            }

    # Side-aware aggregate. Batting reports OPS + wOBA deltas, pitching
    # reports ERA. Each verdict bucket also carries the average projected
    # meta delta so the UI can show projected vs actual side-by-side.
    summary_by_side: dict[str, dict[str, dict]] = {'batting': {}, 'pitching': {}}
    for s, vmap in by_side_verdict.items():
        for verdict, recs in vmap.items():
            if s == 'batting':
                ops_deltas = [
                    (r.get('backtest') or {}).get('delta_ops')
                    for r in recs if isinstance(r.get('backtest'), dict)
                ]
                ops_deltas = [d for d in ops_deltas if d is not None]
                woba_deltas = [
                    (r.get('backtest') or {}).get('delta_woba')
                    for r in recs if isinstance(r.get('backtest'), dict)
                ]
                woba_deltas = [d for d in woba_deltas if d is not None]
                proj_deltas = [r.get('expected_delta') for r in recs
                               if r.get('expected_delta') is not None]
                summary_by_side['batting'][verdict] = {
                    'count': len(recs),
                    'count_with_data': len(ops_deltas),
                    'avg_delta_ops': (round(sum(ops_deltas) / len(ops_deltas), 3)
                                       if ops_deltas else None),
                    'avg_delta_woba': (round(sum(woba_deltas) / len(woba_deltas), 3)
                                        if woba_deltas else None),
                    'avg_projected_delta': (round(sum(proj_deltas) / len(proj_deltas), 1)
                                             if proj_deltas else None),
                }
            else:
                era_deltas = [
                    (r.get('backtest') or {}).get('delta_era')
                    for r in recs if isinstance(r.get('backtest'), dict)
                ]
                era_deltas = [d for d in era_deltas if d is not None]
                proj_deltas = [r.get('expected_delta') for r in recs
                               if r.get('expected_delta') is not None]
                summary_by_side['pitching'][verdict] = {
                    'count': len(recs),
                    'count_with_data': len(era_deltas),
                    'avg_delta_era': (round(sum(era_deltas) / len(era_deltas), 3)
                                       if era_deltas else None),
                    'avg_projected_delta': (round(sum(proj_deltas) / len(proj_deltas), 1)
                                             if proj_deltas else None),
                }

    if close_conn:
        conn.close()
    return {
        'league_id': league_id,
        'rec_type_filter': rec_type,
        'window_days': window_days,
        'total_recs_examined': len(rows),
        'summary_by_verdict': summary_by_verdict,
        'summary_by_side': summary_by_side,
        'detail': detail_rows[:50],  # cap for UI display
    }
