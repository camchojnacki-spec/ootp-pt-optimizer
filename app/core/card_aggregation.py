"""Card-level cross-team aggregation + confidence ratings.

The key insight: in OOTP Perfect Team, the **same card** (e.g., "MLB 2026
Live SS Francisco Lindor NYM") exists on MULTIPLE teams in the league.
Each team's instance of that card plays games and accumulates stats
independently. When we ingest box scores / game logs for the whole
league, we see the SAME CARD performing across multiple team contexts.

Aggregating all instances of a card dramatically expands our sample size
for estimating the card's true performance distribution:

- **Single-team view:** Toronto's Adael Amador has 26 PA → 0.635 OPS.
  That's noise. 26 PA can swing from .400 to .900 by luck alone.
- **Cross-team view:** aggregate across all teams holding Amador → ~500 PA.
  Now the sampling error is tiny and we see the card's real distribution.

This module provides:

1. ``card_stats_aggregate(card_title=..., card_id=..., league_id=...)`` —
   pools PA, hits, HR, WAR, exit velocity, LD%, K%, BB% across every team
   instance seen in our data.

2. ``card_confidence(card_title_or_id)`` — a 0–100 confidence score for a
   card's meta estimate. Combines:
     - Sample size (PA or IP accumulated across all instances)
     - Consistency (variance across team instances — low variance = high
       confidence; volatile results across teams = low confidence)
     - Data source mix (game-log at-bat data > box-score rollups >
       cumulative stats).

3. ``all_card_confidences(league_id=..., role=...)`` — batch compute for
   every card appearing in the DB, useful for the confidence chip shown
   in the UI and for filtering recommendations to "high-confidence only".

Design choice: we group by ``player_name`` (free text from OOTP's CSVs
and HTML) as the primary key, with ``card_id`` as a secondary key when
available. OOTP PT cards share the same ``player_name`` across team
copies — "Adael Amador" on Toronto and "Adael Amador" on Flatbush are
both the Amador card. Our card_id matcher collapses them to the same
integer, so aggregating by card_id gives us the cross-team pool for free.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from typing import Optional

from app.core.database import get_connection, load_config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════

def _f(x, default=0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def card_stats_aggregate(player_name: Optional[str] = None,
                         card_id: Optional[int] = None,
                         league_id: Optional[str] = None,
                         conn=None) -> dict:
    """Aggregate game-log + box-score stats for one card across all team
    instances in the league.

    Returns a dict with pooled batting and pitching stats plus quality-of-
    contact features from the game logs. Fields that have no data are
    None (not zero) so the caller can tell "no samples" from "real zero".
    """
    if not player_name and not card_id:
        raise ValueError("card_stats_aggregate needs player_name or card_id")

    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        # Filter by card_id when available (most reliable — same card on
        # different teams shares one card_id post-matcher). Fall back to
        # player_name if no card_id.
        if card_id:
            key_clause = "card_id = ?"
            key_args: tuple = (card_id,)
            gl_batter_key = "batter_card_id = ?"
            gl_pitcher_key = "pitcher_card_id = ?"
        else:
            key_clause = "player_name = ?"
            key_args = (player_name,)
            gl_batter_key = "batter = ?"
            gl_pitcher_key = "pitcher = ?"

        # Latest batting_stats row per (card_id, team) pair. "Team" here
        # is implicit — each team's version of the card produces its own
        # row each snapshot, so we take the latest snapshot per card_id.
        # We then SUM across all instances to get the cross-team pool.
        league_filter = "AND league_id = ?" if league_id else ""
        league_args = (league_id,) if league_id else ()

        out: dict = {
            'player_name': player_name,
            'card_id': card_id,
            'league_id': league_id,
            'batting': None,
            'pitching': None,
            'game_log': None,
            'team_instances': 0,
        }

        # ── Batting: sum of latest stats across every team instance ──
        bat_rows = conn.execute(f"""
            SELECT pa, ab, hits, hr, doubles, triples, bb, k, sb, war,
                   ops, ops_plus, babip, iso
            FROM batting_stats
            WHERE {key_clause} {league_filter}
              AND pa > 0
              AND id IN (
                  SELECT MAX(id) FROM batting_stats
                  WHERE {key_clause} {league_filter}
                  GROUP BY player_name, COALESCE(league_id, ''), COALESCE(position, ''),
                           bats, throws
              )
        """, key_args + league_args + key_args + league_args).fetchall()

        if bat_rows:
            pa = sum(_f(r['pa']) for r in bat_rows)
            ab = sum(_f(r['ab']) for r in bat_rows)
            hits = sum(_f(r['hits']) for r in bat_rows)
            hr = sum(_f(r['hr']) for r in bat_rows)
            dbl = sum(_f(r['doubles']) for r in bat_rows)
            trp = sum(_f(r['triples']) for r in bat_rows)
            bb = sum(_f(r['bb']) for r in bat_rows)
            k = sum(_f(r['k']) for r in bat_rows)
            sb = sum(_f(r['sb']) for r in bat_rows)
            war_total = sum(_f(r['war']) for r in bat_rows)
            # Weighted averages for rate stats
            ops_plus_avg = None
            ops_avg = None
            if pa > 0:
                ops_plus_avg = sum(_f(r['ops_plus']) * _f(r['pa']) for r in bat_rows) / pa
                ops_avg = sum(_f(r['ops']) * _f(r['pa']) for r in bat_rows) / pa
            out['batting'] = {
                'n_instances': len(bat_rows),
                'pa': int(pa), 'ab': int(ab), 'hits': int(hits),
                'hr': int(hr), 'doubles': int(dbl), 'triples': int(trp),
                'bb': int(bb), 'k': int(k), 'sb': int(sb),
                'war': round(war_total, 2),
                'ops': round(ops_avg, 3) if ops_avg is not None else None,
                'ops_plus': round(ops_plus_avg, 1) if ops_plus_avg is not None else None,
                'avg': round(hits / ab, 3) if ab > 0 else None,
                'k_pct': round(100 * k / pa, 1) if pa > 0 else None,
                'bb_pct': round(100 * bb / pa, 1) if pa > 0 else None,
                'war_per_600': round(war_total * 600.0 / pa, 2) if pa > 0 else None,
                # Per-instance values for consistency (std-dev) computation
                'instance_ops_plus': [int(_f(r['ops_plus'])) for r in bat_rows
                                      if r['ops_plus'] is not None],
                'instance_war_per_600': [round(_f(r['war']) * 600.0 / _f(r['pa']), 2)
                                          for r in bat_rows if _f(r['pa']) > 0],
            }
            out['team_instances'] = max(out['team_instances'], len(bat_rows))

        # ── Pitching ──
        pit_rows = conn.execute(f"""
            SELECT ip, k, bb, hr_allowed, hits_allowed, war, era, era_plus,
                   whip, saves, wins, losses
            FROM pitching_stats
            WHERE {key_clause} {league_filter}
              AND ip > 0
              AND id IN (
                  SELECT MAX(id) FROM pitching_stats
                  WHERE {key_clause} {league_filter}
                  GROUP BY player_name, COALESCE(league_id, ''), COALESCE(position, '')
              )
        """, key_args + league_args + key_args + league_args).fetchall()

        if pit_rows:
            ip = sum(_f(r['ip']) for r in pit_rows)
            k = sum(_f(r['k']) for r in pit_rows)
            bb = sum(_f(r['bb']) for r in pit_rows)
            hra = sum(_f(r['hr_allowed']) for r in pit_rows)
            ha = sum(_f(r['hits_allowed']) for r in pit_rows)
            war_total = sum(_f(r['war']) for r in pit_rows)
            era_avg = None
            era_plus_avg = None
            if ip > 0:
                era_avg = sum(_f(r['era']) * _f(r['ip']) for r in pit_rows) / ip
                era_plus_avg = sum(_f(r['era_plus']) * _f(r['ip']) for r in pit_rows) / ip
            out['pitching'] = {
                'n_instances': len(pit_rows),
                'ip': round(ip, 1),
                'k': int(k), 'bb': int(bb),
                'hr_allowed': int(hra), 'hits_allowed': int(ha),
                'saves': sum(int(_f(r['saves'])) for r in pit_rows),
                'wins': sum(int(_f(r['wins'])) for r in pit_rows),
                'losses': sum(int(_f(r['losses'])) for r in pit_rows),
                'war': round(war_total, 2),
                'era': round(era_avg, 2) if era_avg is not None else None,
                'era_plus': round(era_plus_avg, 1) if era_plus_avg is not None else None,
                'k_per_9': round(9 * k / ip, 1) if ip > 0 else None,
                'bb_per_9': round(9 * bb / ip, 1) if ip > 0 else None,
                'war_per_200': round(war_total * 200.0 / ip, 2) if ip > 0 else None,
                'instance_era_plus': [int(_f(r['era_plus'])) for r in pit_rows
                                       if r['era_plus'] is not None],
                'instance_war_per_200': [round(_f(r['war']) * 200.0 / _f(r['ip']), 2)
                                          for r in pit_rows if _f(r['ip']) > 0],
            }
            out['team_instances'] = max(out['team_instances'], len(pit_rows))

        # ── Game log: at-bat level features ──
        bat_gl = conn.execute(f"""
            SELECT outcome, batted_ball_type, exit_velocity
            FROM game_log_at_bats
            WHERE {gl_batter_key}
        """, key_args).fetchall()
        pit_gl = conn.execute(f"""
            SELECT outcome, batted_ball_type, exit_velocity
            FROM game_log_at_bats
            WHERE {gl_pitcher_key}
        """, key_args).fetchall()

        def _gl_summary(rows):
            if not rows:
                return None
            pa_logged = len(rows)
            k_n = sum(1 for r in rows if r['outcome'] == 'K')
            bb_n = sum(1 for r in rows if r['outcome'] == 'BB')
            hr_n = sum(1 for r in rows if r['outcome'] == 'HR')
            hit_outcomes = {'SINGLE', 'DOUBLE', 'TRIPLE', 'HR'}
            hits_n = sum(1 for r in rows if r['outcome'] in hit_outcomes)
            ld_n = sum(1 for r in rows
                       if (r['batted_ball_type'] or '').lower().startswith('line'))
            gb_n = sum(1 for r in rows
                       if (r['batted_ball_type'] or '').lower().startswith('ground'))
            fb_n = sum(1 for r in rows
                       if (r['batted_ball_type'] or '').lower().startswith(('fly', 'popup')))
            bb_types_n = ld_n + gb_n + fb_n
            ev_list = [r['exit_velocity'] for r in rows if r['exit_velocity'] is not None]
            avg_ev = sum(ev_list) / len(ev_list) if ev_list else None
            return {
                'pa_or_bf': pa_logged,
                'k_pct': round(100 * k_n / pa_logged, 1),
                'bb_pct': round(100 * bb_n / pa_logged, 1),
                'hit_pct': round(100 * hits_n / pa_logged, 1),
                'hr_pct': round(100 * hr_n / pa_logged, 1),
                'ld_pct': round(100 * ld_n / bb_types_n, 1) if bb_types_n else None,
                'gb_pct': round(100 * gb_n / bb_types_n, 1) if bb_types_n else None,
                'fb_pct': round(100 * fb_n / bb_types_n, 1) if bb_types_n else None,
                'avg_ev': round(avg_ev, 1) if avg_ev is not None else None,
                'ev_samples': len(ev_list),
            }

        batting_gl = _gl_summary(bat_gl)
        pitching_gl = _gl_summary(pit_gl)
        if batting_gl or pitching_gl:
            out['game_log'] = {
                'batting': batting_gl,
                'pitching': pitching_gl,
            }

        return out
    finally:
        if close_conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════
# Confidence scoring
# ══════════════════════════════════════════════════════════════════════

# Saturation thresholds — the numbers of PA / IP at which sample-size
# confidence saturates at 1.0. Below these, confidence scales with the
# square root of n / threshold. These were picked to match the point at
# which rate stats (OPS+, ERA+) stabilize per common sabermetric guidance:
# ~600 PA for batting rate stats, ~150 IP for ERA-based rate stats.
_PA_SATURATION = 600
_IP_SATURATION = 150
# Game-log at-bat counts have their own saturation since each game-log
# at-bat carries more info (EV + batted ball type) than a raw box-score
# line. 300 at-bats of EV data is very stable.
_GL_AB_SATURATION = 300


def _stddev(xs: list[float]) -> float:
    if not xs or len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def card_confidence(player_name: Optional[str] = None,
                    card_id: Optional[int] = None,
                    league_id: Optional[str] = None,
                    conn=None) -> dict:
    """Compute a 0–100 confidence score for a card's stats/meta.

    Returns::

        {
            'score': 72,                 # 0..100
            'label': 'high' | 'medium' | 'low' | 'none',
            'drivers': [ ... ],          # human-readable reasons
            'aggregate': {...},          # full aggregate snapshot
        }

    Formula (batting example, pitching mirrors):
        n_factor = sqrt(min(total_PA / 600, 1.0))
        consistency = 1 - min(std(ops_plus_across_instances) / 40, 1.0)
        gl_bonus = 0.3 if game-log at-bats >= 100 else 0
        base_score = 0.6 * n_factor + 0.4 * consistency
        final = min(100, int(100 * base_score * (1 + gl_bonus)))
    """
    agg = card_stats_aggregate(player_name=player_name, card_id=card_id,
                               league_id=league_id, conn=conn)
    drivers = []

    # Pick the dominant role (batting or pitching) based on which has more samples
    bat = agg.get('batting')
    pit = agg.get('pitching')
    gl = agg.get('game_log') or {}
    gl_bat = gl.get('batting') or {}
    gl_pit = gl.get('pitching') or {}

    role = None
    if bat and (not pit or bat.get('pa', 0) >= pit.get('ip', 0) * 4):
        role = 'batting'
    elif pit and pit.get('ip', 0) > 0:
        role = 'pitching'

    if role is None:
        return {
            'score': 0, 'label': 'none',
            'drivers': ['No stat samples found for this card yet.'],
            'aggregate': agg,
        }

    if role == 'batting':
        pa = bat.get('pa') or 0
        n_factor = math.sqrt(min(pa / _PA_SATURATION, 1.0))
        drivers.append(f"{pa} PA across {bat.get('n_instances')} team instance(s)"
                       f" (saturates at {_PA_SATURATION}).")

        consistency = 1.0
        instance_opsp = bat.get('instance_ops_plus') or []
        if len(instance_opsp) >= 2:
            sd = _stddev([float(x) for x in instance_opsp])
            consistency = max(0.0, 1.0 - sd / 40.0)
            drivers.append(f"Across {len(instance_opsp)} instances OPS+ std-dev = {sd:.1f}"
                           f" → consistency {consistency:.2f}.")
        else:
            drivers.append("Only one team instance — no cross-team consistency check.")

        gl_bonus = 0.0
        gl_pa = gl_bat.get('pa_or_bf') or 0
        if gl_pa >= _GL_AB_SATURATION * 0.3:
            gl_bonus = min(0.30, 0.30 * gl_pa / _GL_AB_SATURATION)
            drivers.append(f"{gl_pa} game-log at-bats with EV / batted-ball type data"
                           f" → bonus {gl_bonus:.2f}.")
        base = 0.60 * n_factor + 0.40 * consistency
        score = min(100, int(round(100 * base * (1 + gl_bonus))))

    else:  # pitching
        ip = pit.get('ip') or 0
        # Pitching sample-size fix (2026-04-18): Relievers pitch 50-70 IP/yr;
        # the old 150 IP saturation meant they maxed at n_factor ≈ 0.65 no
        # matter how many game-log batters-faced we had. Now we combine
        # season-level IP (converted to BF at 4.3 per inning) with game-log
        # BF and saturate at 600 total BF — which equates to ~140 IP for a
        # starter or ~80 appearances for a reliever.
        gl_bf = gl_pit.get('pa_or_bf') or 0
        ip_bf = float(ip) * 4.3  # avg BF per inning
        total_bf = ip_bf + gl_bf
        _BF_SATURATION = 600
        n_factor = math.sqrt(min(total_bf / _BF_SATURATION, 1.0))
        if ip:
            drivers.append(
                f"{ip} IP (~{ip_bf:.0f} BF) + {gl_bf} game-log BF = "
                f"{total_bf:.0f} total across {pit.get('n_instances')} "
                f"team instance(s) (saturates at {_BF_SATURATION})."
            )
        else:
            drivers.append(f"{gl_bf} game-log BF only (saturates at {_BF_SATURATION}).")

        consistency = 1.0
        instance_erap = pit.get('instance_era_plus') or []
        if len(instance_erap) >= 2:
            sd = _stddev([float(x) for x in instance_erap])
            consistency = max(0.0, 1.0 - sd / 40.0)
            drivers.append(f"Across {len(instance_erap)} instances ERA+ std-dev = {sd:.1f}"
                           f" → consistency {consistency:.2f}.")
        else:
            drivers.append("Only one team instance — no cross-team consistency check.")

        # Game-log bonus — fires at a LOWER threshold now (30 BF vs old 90)
        # so relievers with a handful of appearances still get credit.
        gl_bonus = 0.0
        if gl_bf >= 30:
            gl_bonus = min(0.30, 0.30 * gl_bf / _GL_AB_SATURATION)
            drivers.append(f"{gl_bf} game-log BF with EV / batted-ball data"
                           f" → bonus {gl_bonus:.2f}.")
        base = 0.60 * n_factor + 0.40 * consistency
        score = min(100, int(round(100 * base * (1 + gl_bonus))))

    if score >= 70:
        label = 'high'
    elif score >= 40:
        label = 'medium'
    elif score > 0:
        label = 'low'
    else:
        label = 'none'

    return {
        'score': score,
        'label': label,
        'role': role,
        'drivers': drivers,
        'aggregate': agg,
    }


def all_card_confidences(league_id: Optional[str] = None,
                         owned_only: bool = False,
                         conn=None) -> list[dict]:
    """Batch-compute confidence for every card with stat samples.

    ``owned_only=True`` restricts to cards the user owns. Returns a list
    sorted by confidence score descending.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        try:
            league_id = league_id or load_config().get('active_league', 'lb124')
        except Exception:
            league_id = league_id or 'lb124'

        owned_clause = "AND c.owned > 0" if owned_only else ""
        # Cards with stat samples in the active league (either batting or pitching)
        rows = conn.execute(f"""
            SELECT DISTINCT c.card_id, c.card_title, c.first_name || ' ' || c.last_name AS player_name,
                   c.position_name, c.pitcher_role_name, c.owned,
                   c.meta_score_batting, c.meta_score_pitching
            FROM cards c
            WHERE c.card_id IN (
                SELECT card_id FROM batting_stats
                  WHERE league_id = ? AND card_id IS NOT NULL
                UNION
                SELECT card_id FROM pitching_stats
                  WHERE league_id = ? AND card_id IS NOT NULL
            ) {owned_clause}
        """, (league_id, league_id)).fetchall()

        out = []
        for r in rows:
            conf = card_confidence(card_id=r['card_id'], league_id=league_id, conn=conn)
            out.append({
                'card_id': r['card_id'],
                'card_title': r['card_title'],
                'player_name': r['player_name'],
                'position_name': r['position_name'],
                'pitcher_role_name': r['pitcher_role_name'],
                'owned': r['owned'],
                'meta_score': r['meta_score_batting'] or r['meta_score_pitching'],
                'confidence_score': conf['score'],
                'confidence_label': conf['label'],
                'role': conf.get('role'),
                'drivers': conf['drivers'],
                'aggregate': conf['aggregate'],
            })
        out.sort(key=lambda x: -x['confidence_score'])
        return out
    finally:
        if close_conn:
            conn.close()
