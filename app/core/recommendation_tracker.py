"""Recommendation audit log + outcome tracker.

Flow:
    1. Each engine that emits recommendations (AI Optimize All, Manager's Eye,
       meta-based lineup advice) calls ``log_recommendations(...)`` with its
       list of picks. Each pick becomes a row in ``recommendation_log``.
    2. On every roster refresh (triggered by the background worker), the
       ``detect_actioned_recommendations`` function diffs the new roster
       against prior roster and marks pending recs as followed/partial/ignored.
    3. After an action is detected, ``measure_outcomes`` runs periodically
       (also on refresh) to compute the WAR/meta delta for the player in
       their new role and writes a verdict.
    4. UI surfaces verdict roll-ups so Cameron can see which recommender
       has the best hit rate and the meta engine can calibrate against
       systematic misses.

Public API:
    - log_recommendation(source, rec_type, ...)
    - log_recommendations(source, list_of_dicts)
    - detect_actioned_recommendations(conn) → count
    - measure_outcomes(conn) → count
    - get_recent_recommendations(limit, status_filter) → list
    - get_scoreboard() → {source: {total, followed, hit_rate, avg_delta}}
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Logging recommendations
# ──────────────────────────────────────────────────────────────────────

def log_recommendation(
    source: str,
    rec_type: str,
    *,
    pos: Optional[str] = None,
    player_name: Optional[str] = None,
    player_card_id: Optional[int] = None,
    from_player: Optional[str] = None,
    from_card_id: Optional[int] = None,
    expected_delta: Optional[float] = None,
    cost_pp: Optional[int] = None,
    reasoning: Optional[str] = None,
    league_id: Optional[str] = None,
    data_version: Optional[int] = None,
) -> int:
    """Insert a single recommendation. Returns the row id."""
    from app.core.database import get_connection
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO recommendation_log
            (source, rec_type, pos, player_name, player_card_id,
             from_player, from_card_id, expected_delta, cost_pp,
             reasoning, league_id, data_version, verdict)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """,
        (source, rec_type, pos, player_name, player_card_id,
         from_player, from_card_id, expected_delta, cost_pp,
         reasoning, league_id, data_version),
    )
    conn.commit()
    return int(cur.lastrowid)


def log_recommendations(source: str, picks: Iterable[dict],
                        league_id: Optional[str] = None,
                        data_version: Optional[int] = None) -> int:
    """Bulk-log a batch of picks from one engine run.

    ``picks`` is a list of dicts shaped like the AI Optimize All picks_data:
        {'pos', 'action', 'card_name', 'current_name', 'cost',
         'reason', 'card_id', 'current_card_id', 'expected_delta'}

    Only non-'Keep' actions are logged (Keeps aren't recommendations).
    Dedup window widened from 1h → 24h: the engine re-renders the same
    upgrade plan every page load, and anything meaningful hasn't shifted
    within a day. 1h was producing 60% duplicate rec rows.
    """
    from app.core.database import get_connection
    conn = get_connection()
    logged = 0
    one_day_ago = (datetime.now() - timedelta(hours=24)).isoformat()

    for p in picks:
        action = (p.get('action') or '').lower()
        if not action or action == 'keep':
            continue
        rec_type = {
            'buy': 'buy', 'promote': 'promote', 'platoon': 'platoon',
            'bench': 'bench', 'sell': 'sell', 'demote': 'demote', 'hook': 'hook',
        }.get(action, action)

        player_name = p.get('card_name') or p.get('player_name')
        from_player = p.get('current_name')
        pos = p.get('pos')

        # Resolve to card_ids for reliable dedup — two LLM runs may emit the
        # same pick with slightly different name strings; matching on
        # card_id avoids logging the same rec twice.
        player_card_id = p.get('card_id')
        from_card_id = p.get('current_card_id')
        if player_card_id is None and player_name:
            try:
                from app.core.name_resolver import resolve_to_card_id
                player_card_id = resolve_to_card_id(player_name, conn=conn)
            except Exception:
                pass
        if from_card_id is None and from_player:
            try:
                from app.core.name_resolver import resolve_to_card_id
                from_card_id = resolve_to_card_id(from_player, conn=conn)
            except Exception:
                pass

        # Dedupe: prefer card_id equality when available, fall back to
        # name equality for name-only recs (e.g. manual entries).
        dup = conn.execute(
            """
            SELECT id FROM recommendation_log
            WHERE source = ? AND rec_type = ? AND pos = ?
              AND (
                (? IS NOT NULL AND player_card_id = ?)
                OR (? IS NULL AND COALESCE(player_name,'') = COALESCE(?,''))
              )
              AND (
                (? IS NOT NULL AND from_card_id = ?)
                OR (? IS NULL AND COALESCE(from_player,'') = COALESCE(?,''))
              )
              AND created_at > ?
            LIMIT 1
            """,
            (source, rec_type, pos,
             player_card_id, player_card_id,
             player_card_id, player_name,
             from_card_id, from_card_id,
             from_card_id, from_player,
             one_day_ago),
        ).fetchone()
        if dup:
            continue

        conn.execute(
            """
            INSERT INTO recommendation_log
                (source, rec_type, pos, player_name, player_card_id,
                 from_player, from_card_id, expected_delta, cost_pp,
                 reasoning, league_id, data_version, verdict)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (source, rec_type, pos, player_name, p.get('card_id'),
             from_player, p.get('current_card_id'),
             p.get('expected_delta'), p.get('cost'),
             p.get('reason') or p.get('reasoning'),
             league_id, data_version),
        )
        logged += 1
    conn.commit()
    return logged


# ──────────────────────────────────────────────────────────────────────
# Detecting which recommendations were actioned
# ──────────────────────────────────────────────────────────────────────

def detect_actioned_recommendations(conn=None) -> int:
    """Scan pending recommendations and mark the ones the user followed.

    Detection rule per rec_type:
      - promote/buy: target player now has an active lineup_role
      - platoon: both target AND from_player have active lineup_roles
      - bench/demote: from_player no longer has an active lineup_role
      - sell: target player no longer owned
      - hook: relevant only if we later add a hook-tracking concept

    If the engine's advice matches the user's actual roster state, we mark
    the rec as "followed" and stamp action_detected_at. Partial matches
    (e.g. the user promoted someone else in the recommended slot) get
    marked "partial". Recs older than 14 days without a detected action
    get flipped to "ignored".
    """
    from app.core.database import get_connection
    from app.core.name_resolver import resolve_to_card_id
    if conn is None:
        conn = get_connection()

    now = datetime.now().isoformat(timespec='seconds')
    updated = 0

    # Build maps indexed by card_id — stopped relying on substring name
    # matching because it false-positives when two players share a last
    # name (Cain/Cain, Rodriguez/Rodriguez, etc.) and false-negatives
    # on the card_title form "MLB 2026 Live CF Lorenzo Cain KC 2015".
    # We now key everything by card_id and fall back to canonical name
    # strings only when card_id is missing.
    active_card_ids: set[int] = set()
    active_names: set[str] = set()
    bench_card_ids: set[int] = set()     # owned but not currently active
    bench_names: set[str] = set()
    owned_card_ids: set[int] = set()
    try:
        for r in conn.execute("""
            SELECT player_name, position, lineup_role, card_id
            FROM roster
            WHERE lineup_role != 'league'
              AND DATE(snapshot_date) = (
                  SELECT MAX(DATE(snapshot_date)) FROM roster
                  WHERE lineup_role != 'league'
              )
        """).fetchall():
            name = r['player_name'] or ''
            cid = r['card_id']
            role = (r['lineup_role'] or '').lower()
            if role in ('starter', 'rotation', 'closer', 'primary', 'active'):
                if cid: active_card_ids.add(int(cid))
                if name: active_names.add(name)
            elif role in ('bench', 'reserve', 'bullpen'):
                # 'bullpen' is wrongly applied to batters (ingest bug) —
                # treating as bench is safer than active for detection.
                if cid: bench_card_ids.add(int(cid))
                if name: bench_names.add(name)
    except Exception as e:
        logger.warning("Active roster fetch failed: %s", e)

    try:
        for r in conn.execute("SELECT card_id FROM cards WHERE owned = 1 AND card_id IS NOT NULL").fetchall():
            owned_card_ids.add(int(r['card_id']))
    except Exception:
        pass

    # Pending recs — scan and classify
    pending = conn.execute(
        "SELECT * FROM recommendation_log WHERE action_type IS NULL "
        "AND verdict = 'pending' AND created_at > datetime('now', '-30 days')"
    ).fetchall()

    def _cid_or_resolve(rec, field_cid: str, field_name: str) -> Optional[int]:
        """Return card_id for a rec column — use stored value first, else
        resolve from name. Persists the resolved value so subsequent runs
        skip the lookup."""
        cid = rec[field_cid]
        if cid:
            return int(cid)
        name = rec[field_name]
        if not name:
            return None
        try:
            resolved = resolve_to_card_id(name, prefer_owned=True, conn=conn)
        except Exception:
            resolved = None
        if resolved:
            # Persist so we don't re-resolve next time
            try:
                conn.execute(
                    f"UPDATE recommendation_log SET {field_cid} = ? WHERE id = ?",
                    (resolved, rec['id'])
                )
            except Exception:
                pass
        return resolved

    def _is_active(cid: Optional[int], name: str) -> bool:
        if cid is not None and cid in active_card_ids:
            return True
        # Fallback — name must match EXACTLY (not substring)
        return bool(name and name in active_names)

    def _is_bench_or_active(cid: Optional[int], name: str) -> bool:
        if cid is not None and (cid in active_card_ids or cid in bench_card_ids):
            return True
        return bool(name and (name in active_names or name in bench_names))

    for rec in pending:
        rec_type = rec['rec_type']
        target_name = rec['player_name'] or ''
        from_name = rec['from_player'] or ''
        pos = rec['pos'] or ''

        target_cid = _cid_or_resolve(rec, 'player_card_id', 'player_name')
        from_cid = _cid_or_resolve(rec, 'from_card_id', 'from_player')

        action_type: Optional[str] = None

        target_active = _is_active(target_cid, target_name)
        from_active = _is_active(from_cid, from_name)
        target_owned = (target_cid in owned_card_ids) if target_cid else False

        if rec_type in ('promote', 'buy'):
            if target_active:
                action_type = 'followed'
            elif target_owned or _is_bench_or_active(target_cid, target_name):
                action_type = 'partial'   # owned/benched but not active
        elif rec_type == 'platoon':
            if target_active and from_active:
                action_type = 'followed'
            elif target_active or from_active:
                action_type = 'partial'
        elif rec_type in ('bench', 'demote'):
            # "Success" = the from-player is no longer active
            if from_name and not from_active:
                action_type = 'followed'
        elif rec_type == 'sell':
            # Sold when no longer owned
            if target_cid and target_cid not in owned_card_ids:
                action_type = 'followed'
            elif target_name and target_cid is None:
                # Couldn't resolve — fall back to owned name check
                pass
        elif rec_type == 'intake':
            # New-arrival intake is "followed" once the card shows up in
            # bench or active roster (user acknowledged it into the team).
            if target_active:
                action_type = 'followed'
            elif _is_bench_or_active(target_cid, target_name):
                action_type = 'partial'

        if action_type:
            conn.execute(
                """
                UPDATE recommendation_log
                SET action_type = ?, action_detected_at = ?
                WHERE id = ?
                """,
                (action_type, now, rec['id'])
            )
            updated += 1

    # Age out unactioned recs after 14 days
    conn.execute(
        """
        UPDATE recommendation_log
        SET action_type = 'ignored', verdict = 'neutral'
        WHERE action_type IS NULL
          AND verdict = 'pending'
          AND created_at < datetime('now', '-14 days')
        """
    )
    conn.commit()
    return updated


# ──────────────────────────────────────────────────────────────────────
# Measuring outcomes
# ──────────────────────────────────────────────────────────────────────

def measure_outcomes(conn=None) -> int:
    """For followed recs, compute WAR/meta before/after and a verdict.

    Runs each refresh cycle. Takes current WAR/meta snapshot for target
    and compares to the pre-action snapshot (stored at action_detected_at
    time). Verdict:
      - positive: observed delta ≥ 50% of expected delta
      - neutral: within ±50%
      - negative: observed delta < -50% of expected (recommendation hurt)
      - pending: not enough data yet (days_observed < 3)
    """
    from app.core.database import get_connection
    if conn is None:
        conn = get_connection()

    updated = 0
    now = datetime.now()

    # Recs that have been followed but not yet measured (or need re-measure)
    followed = conn.execute(
        """
        SELECT * FROM recommendation_log
        WHERE action_type = 'followed'
          AND (verdict = 'pending' OR verdict IS NULL)
          AND action_detected_at IS NOT NULL
        """
    ).fetchall()

    for rec in followed:
        target = rec['player_name'] or ''
        if not target:
            continue

        # Days since action
        try:
            ad = datetime.fromisoformat(rec['action_detected_at'])
            days_observed = (now - ad).days
        except Exception:
            days_observed = 0

        # Grab current meta + most recent WAR for the target
        meta_after: Optional[float] = None
        war_after: Optional[float] = None
        try:
            row = conn.execute(
                """
                SELECT c.meta_score_batting, c.meta_score_pitching, c.pitcher_role,
                       bs.war as bat_war, bs.pa as bat_pa,
                       ps.war as pit_war, ps.ip as pit_ip
                FROM cards c
                LEFT JOIN batting_stats bs ON bs.card_id = c.card_id
                LEFT JOIN pitching_stats ps ON ps.card_id = c.card_id
                WHERE c.card_title LIKE ?
                LIMIT 1
                """,
                (f'%{target}%',),
            ).fetchone()
            if row:
                meta_after = row['meta_score_batting'] or row['meta_score_pitching']
                if row['pitcher_role']:
                    war_after = (row['pit_war'] or 0) * 200.0 / (row['pit_ip'] or 1) if row['pit_ip'] else None
                else:
                    war_after = (row['bat_war'] or 0) * 600.0 / (row['bat_pa'] or 1) if row['bat_pa'] else None
        except Exception as e:
            logger.debug("Outcome fetch failed for %s: %s", target, e)
            continue

        # Compute verdict
        verdict = 'pending'
        expected = rec['expected_delta']
        meta_before = rec['meta_before']
        if days_observed >= 3 and meta_after is not None:
            if meta_before is None:
                # First measurement — just store current as baseline
                meta_before = meta_after
                verdict = 'pending'
            else:
                observed_delta = meta_after - meta_before
                if expected and expected > 0:
                    ratio = observed_delta / expected
                    if ratio >= 0.5:
                        verdict = 'positive'
                    elif ratio >= -0.2:
                        verdict = 'neutral'
                    else:
                        verdict = 'negative'
                else:
                    # No expected_delta captured — use absolute observed
                    if observed_delta > 5:
                        verdict = 'positive'
                    elif observed_delta > -5:
                        verdict = 'neutral'
                    else:
                        verdict = 'negative'

        conn.execute(
            """
            UPDATE recommendation_log
            SET meta_after = ?, war_after = ?, verdict = ?,
                verdict_measured_at = ?, days_observed = ?,
                meta_before = COALESCE(meta_before, ?)
            WHERE id = ?
            """,
            (meta_after, war_after, verdict,
             now.isoformat(timespec='seconds'), days_observed,
             meta_before, rec['id'])
        )
        updated += 1

    conn.commit()
    return updated


# ──────────────────────────────────────────────────────────────────────
# Read API — used by UI
# ──────────────────────────────────────────────────────────────────────

def get_recent_recommendations(limit: int = 50,
                                status_filter: Optional[str] = None) -> list[dict]:
    """Return most-recent recommendations for the tracker UI."""
    from app.core.database import get_connection
    conn = get_connection()
    sql = "SELECT * FROM recommendation_log"
    params: list = []
    if status_filter == 'pending':
        sql += " WHERE verdict = 'pending'"
    elif status_filter == 'followed':
        sql += " WHERE action_type = 'followed'"
    elif status_filter == 'ignored':
        sql += " WHERE action_type = 'ignored'"
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_scoreboard() -> dict:
    """Hit-rate + avg-delta roll-up per source engine.

    Returns:
        {
          'ai_optimize_all': {
             'total': N, 'followed': F, 'ignored': I, 'pending': P,
             'positive': X, 'neutral': Y, 'negative': Z,
             'hit_rate': 0.67, 'avg_meta_delta': 12.3,
          },
          'manager_eye': {...},
          ...
        }
    """
    from app.core.database import get_connection
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT source,
               COUNT(*) as total,
               SUM(CASE WHEN action_type = 'followed' THEN 1 ELSE 0 END) as followed,
               SUM(CASE WHEN action_type = 'partial' THEN 1 ELSE 0 END) as partial_acted,
               SUM(CASE WHEN action_type = 'ignored' THEN 1 ELSE 0 END) as ignored,
               SUM(CASE WHEN action_type IS NULL THEN 1 ELSE 0 END) as undetected,
               SUM(CASE WHEN verdict = 'pending' THEN 1 ELSE 0 END) as pending,
               SUM(CASE WHEN verdict = 'positive' THEN 1 ELSE 0 END) as positive,
               SUM(CASE WHEN verdict = 'neutral' THEN 1 ELSE 0 END) as neutral,
               SUM(CASE WHEN verdict = 'negative' THEN 1 ELSE 0 END) as negative,
               AVG(CASE WHEN verdict IN ('positive','neutral','negative')
                   THEN (COALESCE(meta_after,0) - COALESCE(meta_before,0))
                   ELSE NULL END) as avg_meta_delta,
               AVG(CASE WHEN verdict IN ('positive','neutral','negative')
                   THEN expected_delta
                   ELSE NULL END) as avg_expected_delta
        FROM recommendation_log
        GROUP BY source
        """
    ).fetchall()
    out: dict = {}
    for r in rows:
        src = r['source']
        scored = (r['positive'] or 0) + (r['negative'] or 0) + (r['neutral'] or 0)
        hit_rate = ((r['positive'] or 0) / scored) if scored else None
        followed = int(r['followed'] or 0)
        partial = int(r['partial_acted'] or 0)
        # "Acted-on rate" = portion the user did something about (full or
        # partial action). More interpretable than raw followed count since
        # many recs are never acted on by design.
        total = int(r['total'] or 0)
        acted_rate = ((followed + partial) / total) if total else None
        out[src] = {
            'total': total,
            'followed': followed,
            'partial': partial,
            'ignored': int(r['ignored'] or 0),
            'undetected': int(r['undetected'] or 0),
            'pending': int(r['pending'] or 0),
            'positive': int(r['positive'] or 0),
            'neutral': int(r['neutral'] or 0),
            'negative': int(r['negative'] or 0),
            'hit_rate': hit_rate,
            'acted_rate': acted_rate,
            'avg_meta_delta': r['avg_meta_delta'],
            'avg_expected_delta': r['avg_expected_delta'],
        }
    return out
