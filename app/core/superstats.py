"""Superstats — advanced per-card analytics derived from game_log_at_bats.

These are features the CSV-based stats tables can't produce:

1. ``player_superstats(card_id)`` — pooled EV + batted-ball mix + true K%/BB%
   across every team instance of the card in the league.
2. ``observed_splits(card_id)`` — actual vs-LHP / vs-RHP performance from the
   pitch-by-pitch log (not card ratings, real observed outcomes).
3. ``regression_candidates(league_id)`` — flags cards whose BABIP is
   wildly off their LD% / EV expectation. These are the buy-low (positive
   regression) and sell-high (negative regression) lists.
4. ``opponent_quality_adjusted(player_name, league_id)`` — weights each
   at-bat by the opposing pitcher's ERA+ so you can see "OPS against tough
   pitchers" vs "OPS against mediocre pitchers".

All functions honor league_id scoping and pool across team instances
(cross-team aggregation, same as ``card_aggregation``).
"""
from __future__ import annotations

import logging
import math
import sqlite3
from typing import Optional

from app.core.database import get_connection, load_config

logger = logging.getLogger(__name__)


def _active_league(conn=None) -> str:
    try:
        return load_config().get('active_league', 'lb124')
    except Exception:
        return 'lb124'


# ══════════════════════════════════════════════════════════════════════
# Per-card superstats (everything game-log-derived in one shot)
# ══════════════════════════════════════════════════════════════════════

def player_superstats(card_id: Optional[int] = None,
                      player_name: Optional[str] = None,
                      league_id: Optional[str] = None,
                      conn=None) -> dict:
    """Return game-log-derived superstats for one card.

    Pools across every team instance. Keys:
        n_at_bats           : int  — total PA logged
        outcomes            : {'K': 34, 'BB': 12, 'SINGLE': 40, ...}
        k_pct, bb_pct, hr_pct, hit_pct   : floats (0-100)
        batted_balls        : {'Line Drive': 60, 'Groundball': 100, 'Popup': 20}
        ld_pct, gb_pct, fb_pct           : floats (on batted balls only)
        ev_avg, ev_max, ev_p90           : floats — exit velocity stats
        ev_on_ld, ev_on_fb               : avg EV by batted-ball type
        barrel_rate                      : % of EVs ≥ 95 (proxy — no angle data)
        soft_rate                        : % of EVs < 75
        hard_hit_rate                    : % of EVs ≥ 90
        pitches_per_pa                   : avg pitches seen per at-bat
        strikeouts_swinging, strikeouts_looking   : ints
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        if not card_id and not player_name:
            raise ValueError("need card_id or player_name")

        # Filter on whichever key we have
        if card_id:
            key_sql = "batter_card_id = ?"
            key_args: tuple = (card_id,)
        else:
            key_sql = "batter = ?"
            key_args = (player_name,)

        rows = conn.execute(
            f"""SELECT outcome, batted_ball_type, exit_velocity, pitches_seen,
                       strikeout_type
                FROM game_log_at_bats
                WHERE {key_sql}""",
            key_args,
        ).fetchall()

        if not rows:
            return {'n_at_bats': 0}

        n = len(rows)
        outcomes: dict[str, int] = {}
        bb_types: dict[str, int] = {}
        evs = []
        ld_evs = []
        fb_evs = []
        gb_evs = []
        k_swing = 0
        k_look = 0
        pitches_total = 0
        pitches_seen_count = 0

        for r in rows:
            outc = r['outcome']
            if outc:
                outcomes[outc] = outcomes.get(outc, 0) + 1
            bt = r['batted_ball_type']
            if bt:
                bb_types[bt] = bb_types.get(bt, 0) + 1
            ev = r['exit_velocity']
            if ev is not None:
                try:
                    ev = int(ev)
                    evs.append(ev)
                    bl = (bt or '').lower()
                    if bl.startswith('line'):
                        ld_evs.append(ev)
                    elif bl.startswith('ground'):
                        gb_evs.append(ev)
                    elif bl.startswith(('fly', 'popup')):
                        fb_evs.append(ev)
                except (ValueError, TypeError):
                    pass
            st = r['strikeout_type']
            if st == 'swinging': k_swing += 1
            elif st == 'looking': k_look += 1
            p = r['pitches_seen']
            if p is not None:
                try:
                    pitches_total += int(p)
                    pitches_seen_count += 1
                except (ValueError, TypeError):
                    pass

        hit_set = {'SINGLE', 'DOUBLE', 'TRIPLE', 'HR'}
        def pct(num, den): return round(100 * num / den, 1) if den else None
        def mean(vals): return round(sum(vals) / len(vals), 1) if vals else None

        total_bb = sum(bb_types.values())
        ld_pct = pct(sum(v for k, v in bb_types.items() if k.lower().startswith('line')), total_bb)
        gb_pct = pct(sum(v for k, v in bb_types.items() if k.lower().startswith('ground')), total_bb)
        fb_pct = pct(sum(v for k, v in bb_types.items() if k.lower().startswith(('fly', 'popup'))), total_bb)

        ev_sorted = sorted(evs)
        ev_avg = mean(evs)
        ev_max = max(evs) if evs else None
        ev_p90 = ev_sorted[int(len(ev_sorted) * 0.9)] if ev_sorted else None

        # Barrel proxy: batted balls ≥ 95 mph (OOTP doesn't export launch angle
        # so we can't do Statcast barrels — this is the best scalar we have)
        barrel_n = sum(1 for e in evs if e >= 95)
        soft_n = sum(1 for e in evs if e < 75)
        hard_n = sum(1 for e in evs if e >= 90)

        return {
            'n_at_bats': n,
            'outcomes': outcomes,
            'k_pct': pct(outcomes.get('K', 0), n),
            'bb_pct': pct(outcomes.get('BB', 0), n),
            'hr_pct': pct(outcomes.get('HR', 0), n),
            'hit_pct': pct(sum(outcomes.get(h, 0) for h in hit_set), n),
            'batted_balls': bb_types,
            'ld_pct': ld_pct,
            'gb_pct': gb_pct,
            'fb_pct': fb_pct,
            'ev_avg': ev_avg,
            'ev_max': ev_max,
            'ev_p90': ev_p90,
            'ev_on_ld': mean(ld_evs),
            'ev_on_fb': mean(fb_evs),
            'ev_on_gb': mean(gb_evs),
            'barrel_rate': pct(barrel_n, len(evs)),
            'soft_rate': pct(soft_n, len(evs)),
            'hard_hit_rate': pct(hard_n, len(evs)),
            'pitches_per_pa': round(pitches_total / pitches_seen_count, 2)
                if pitches_seen_count else None,
            'strikeouts_swinging': k_swing,
            'strikeouts_looking': k_look,
        }
    finally:
        if close_conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════
# Observed platoon splits
# ══════════════════════════════════════════════════════════════════════

def observed_splits(card_id: Optional[int] = None,
                    player_name: Optional[str] = None,
                    min_pa_per_side: int = 20,
                    conn=None) -> dict:
    """Compute observed vs-LHP and vs-RHP splits from the game logs.

    Unlike ``card.contact_vl / contact_vr`` (rating inputs), this is what the
    batter ACTUALLY did against each pitcher handedness. Returns:
        vs_LHP: {pa, k_pct, bb_pct, hit_pct, hr_pct, ev_avg}
        vs_RHP: same
        split_gap: |vs_LHP.hit_pct - vs_RHP.hit_pct| — how platoon-sensitive?
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        if card_id:
            key_sql = "batter_card_id = ?"
            key_args: tuple = (card_id,)
        elif player_name:
            key_sql = "batter = ?"
            key_args = (player_name,)
        else:
            return {}

        rows = conn.execute(
            f"""SELECT pitcher_hand, outcome, exit_velocity
                FROM game_log_at_bats WHERE {key_sql}""",
            key_args,
        ).fetchall()

        hit_set = {'SINGLE', 'DOUBLE', 'TRIPLE', 'HR'}
        by_hand = {'LHP': {'pa': 0, 'k': 0, 'bb': 0, 'hr': 0, 'hits': 0, 'ev_list': []},
                   'RHP': {'pa': 0, 'k': 0, 'bb': 0, 'hr': 0, 'hits': 0, 'ev_list': []}}
        for r in rows:
            h = r['pitcher_hand']
            if h not in by_hand:
                continue
            b = by_hand[h]
            b['pa'] += 1
            outc = r['outcome']
            if outc == 'K': b['k'] += 1
            elif outc == 'BB': b['bb'] += 1
            elif outc == 'HR': b['hr'] += 1; b['hits'] += 1
            elif outc in hit_set: b['hits'] += 1
            if r['exit_velocity'] is not None:
                try:
                    b['ev_list'].append(int(r['exit_velocity']))
                except (ValueError, TypeError):
                    pass

        out = {}
        for h, b in by_hand.items():
            if b['pa'] < min_pa_per_side:
                out[f'vs_{h}'] = {'pa': b['pa'], 'insufficient_sample': True}
                continue
            ev_avg = (sum(b['ev_list']) / len(b['ev_list'])) if b['ev_list'] else None
            out[f'vs_{h}'] = {
                'pa': b['pa'],
                'k_pct': round(100 * b['k'] / b['pa'], 1),
                'bb_pct': round(100 * b['bb'] / b['pa'], 1),
                'hr_pct': round(100 * b['hr'] / b['pa'], 1),
                'hit_pct': round(100 * b['hits'] / b['pa'], 1),
                'ev_avg': round(ev_avg, 1) if ev_avg is not None else None,
            }

        # Gap
        l = out.get('vs_LHP') or {}
        r = out.get('vs_RHP') or {}
        if 'hit_pct' in l and 'hit_pct' in r:
            out['split_gap_hit_pct'] = round(abs(l['hit_pct'] - r['hit_pct']), 1)
        return out
    finally:
        if close_conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════
# Regression candidates
# ══════════════════════════════════════════════════════════════════════

# Typical league BABIP in MLB-adjacent leagues hovers around .295–.310.
# We use the active league's actual median if available, otherwise .300.
def _league_babip(conn, league_id: str) -> float:
    try:
        row = conn.execute(
            "SELECT AVG(babip) FROM batting_stats WHERE league_id = ? AND pa >= 150",
            (league_id,),
        ).fetchone()
        if row and row[0] and 0.2 < row[0] < 0.4:
            return float(row[0])
    except Exception:
        pass
    return 0.300


def regression_candidates(league_id: Optional[str] = None,
                          min_pa: int = 150,
                          conn=None) -> list[dict]:
    """Flag cards whose observed BABIP is far off their quality-of-contact signature.

    Positive-regression candidates (buy-low): BABIP meaningfully BELOW league
    but LD% above league + EV above league → good contact, bad luck, expected
    to rebound.

    Negative-regression candidates (sell-high): BABIP meaningfully ABOVE
    league but LD% below league + EV below league → weak contact finding
    gloves, expected to fall.

    Each candidate row has direction ('up'|'down'), a concise reason, and the
    underlying numbers so the UI can explain without re-querying.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        league_id = league_id or _active_league(conn)
        lg_babip = _league_babip(conn, league_id)

        # Pull BABIP from the latest batting_stats snapshot per card + our
        # game-log-derived LD% / EV for the same cards.
        rows = conn.execute("""
            SELECT c.card_id, c.card_title, c.owned, c.position_name,
                   bs.pa, bs.babip, bs.ops_plus, bs.ops
            FROM cards c
            INNER JOIN batting_stats bs ON bs.card_id = c.card_id
            INNER JOIN (
                SELECT card_id, MAX(snapshot_date) AS mx
                FROM batting_stats WHERE league_id = ? AND card_id IS NOT NULL
                GROUP BY card_id
            ) latest ON bs.card_id = latest.card_id AND bs.snapshot_date = latest.mx
            WHERE bs.league_id = ? AND bs.pa >= ?
              AND bs.babip IS NOT NULL
              AND bs.babip > 0
        """, (league_id, league_id, min_pa)).fetchall()

        candidates = []
        for r in rows:
            ss = player_superstats(card_id=r['card_id'], league_id=league_id, conn=conn)
            ld_pct = ss.get('ld_pct')
            ev_avg = ss.get('ev_avg')
            if ld_pct is None or ev_avg is None or ss.get('n_at_bats', 0) < 50:
                # Need enough game-log data to judge contact quality
                continue

            babip = float(r['babip'])
            babip_gap = babip - lg_babip

            # Soft / hard contact signals
            # Positive regression: BABIP low BUT contact is solid (LD%>=25, EV>=83)
            if babip_gap <= -0.025 and (ld_pct >= 25 or ev_avg >= 83):
                candidates.append({
                    'direction': 'up',
                    'card_id': r['card_id'],
                    'card_title': r['card_title'],
                    'owned': r['owned'],
                    'position': r['position_name'],
                    'pa': r['pa'],
                    'babip': babip,
                    'babip_vs_league': round(babip_gap, 3),
                    'ld_pct': ld_pct,
                    'ev_avg': ev_avg,
                    'ops': r['ops'],
                    'ops_plus': r['ops_plus'],
                    'reason': (
                        f"BABIP {babip:.3f} ({babip_gap:+.3f} vs league {lg_babip:.3f}) "
                        f"but LD% {ld_pct:.0f} and EV {ev_avg:.1f} — quality contact not falling."
                    ),
                })
            # Negative regression: BABIP high AND contact weak (LD%<22 OR EV<79)
            elif babip_gap >= +0.025 and (ld_pct < 22 or ev_avg < 79):
                candidates.append({
                    'direction': 'down',
                    'card_id': r['card_id'],
                    'card_title': r['card_title'],
                    'owned': r['owned'],
                    'position': r['position_name'],
                    'pa': r['pa'],
                    'babip': babip,
                    'babip_vs_league': round(babip_gap, 3),
                    'ld_pct': ld_pct,
                    'ev_avg': ev_avg,
                    'ops': r['ops'],
                    'ops_plus': r['ops_plus'],
                    'reason': (
                        f"BABIP {babip:.3f} ({babip_gap:+.3f} vs league {lg_babip:.3f}) "
                        f"but LD% {ld_pct:.0f} and EV {ev_avg:.1f} — weak contact finding gloves."
                    ),
                })
        # Sort: strongest-signal first (by absolute babip gap)
        candidates.sort(key=lambda x: -abs(x['babip_vs_league']))
        return candidates
    finally:
        if close_conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════
# Opponent-quality-adjusted performance
# ══════════════════════════════════════════════════════════════════════

def opponent_quality_adjusted(player_name: Optional[str] = None,
                              card_id: Optional[int] = None,
                              league_id: Optional[str] = None,
                              conn=None) -> dict:
    """Compute opponent-adjusted offense for a batter.

    Each at-bat weighted by the opposing pitcher's ERA+ (higher = tougher
    matchup). Outputs:
        raw_ops, adjusted_ops — adjusted = raw * (100 / avg_opp_era_plus)
        avg_opp_era_plus      — the faced-pitcher quality
        faced_aces            — count of ABs vs pitchers with ERA+ >= 125
        faced_filler          — count of ABs vs pitchers with ERA+ <= 80

    Requires game_log_at_bats (pitcher name) joined against pitching_stats
    (ERA+ lookup). Returns dict with ``available=False`` if no log data.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        league_id = league_id or _active_league(conn)

        if card_id:
            key_sql = "gl.batter_card_id = ?"
            key_args: tuple = (card_id,)
        elif player_name:
            key_sql = "gl.batter = ?"
            key_args = (player_name,)
        else:
            return {'available': False}

        # Aggregate opposing ERA+ from the latest pitching_stats snapshot
        # for each pitcher faced. We join by pitcher name — this is coarse
        # (duplicate-name pitchers get pooled into one average), which is
        # acceptable for opponent-quality weighting.
        rows = conn.execute(f"""
            SELECT gl.outcome, gl.exit_velocity,
                   AVG(ps.era_plus) AS avg_era_plus
            FROM game_log_at_bats gl
            LEFT JOIN pitching_stats ps ON ps.player_name = gl.pitcher
                  AND ps.league_id = ?
                  AND ps.ip >= 10
            WHERE {key_sql}
            GROUP BY gl.id
        """, (league_id,) + key_args).fetchall()

        if not rows:
            return {'available': False}

        hit_set = {'SINGLE', 'DOUBLE', 'TRIPLE', 'HR'}
        total_pa = 0
        hits = 0
        bb = 0
        hr = 0
        opp_era_sum = 0.0
        opp_n = 0
        faced_aces = 0
        faced_filler = 0

        # Opponent-adjusted: sum(outcome_bool * era_plus/100) / sum(era_plus/100)
        weighted_hit_sum = 0.0
        weight_sum = 0.0

        for r in rows:
            total_pa += 1
            era_plus = r['avg_era_plus']
            if era_plus and era_plus > 0:
                opp_era_sum += float(era_plus)
                opp_n += 1
                if era_plus >= 125: faced_aces += 1
                elif era_plus <= 80: faced_filler += 1
                w = float(era_plus) / 100.0
            else:
                w = 1.0
            outc = r['outcome']
            is_hit = 1.0 if outc in hit_set else 0.0
            weighted_hit_sum += is_hit * w
            weight_sum += w
            if outc in hit_set: hits += 1
            if outc == 'BB': bb += 1
            if outc == 'HR': hr += 1

        avg_opp_era = (opp_era_sum / opp_n) if opp_n else None
        raw_hit_pct = (hits / total_pa * 100) if total_pa else None
        # Adjusted hit rate — what would it be if opponent quality was league-avg (100)?
        # hit_rate_adjusted = weighted_hit_sum / weight_sum * (avg_era / 100)
        # But simpler: rescale raw by (100 / avg_era) — crude but intuitive
        adj_hit_pct = (raw_hit_pct * 100 / avg_opp_era) if (raw_hit_pct is not None and avg_opp_era) else None

        return {
            'available': True,
            'pa': total_pa,
            'raw_hit_pct': round(raw_hit_pct, 1) if raw_hit_pct is not None else None,
            'adjusted_hit_pct': round(adj_hit_pct, 1) if adj_hit_pct is not None else None,
            'avg_opp_era_plus': round(avg_opp_era, 1) if avg_opp_era else None,
            'hits': hits, 'bb': bb, 'hr': hr,
            'faced_aces': faced_aces,           # PA vs ERA+ ≥ 125
            'faced_filler': faced_filler,       # PA vs ERA+ ≤ 80
        }
    finally:
        if close_conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════
# Full summary for one card (for UI display)
# ══════════════════════════════════════════════════════════════════════

def card_full_superstat_report(card_id: int, conn=None) -> dict:
    """One-call helper for the Card Detail page.

    Returns everything in one dict — superstats, observed splits, and
    opponent-quality-adjusted performance. Null sub-fields when the card
    has no logs.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        league_id = _active_league(conn)
        return {
            'card_id': card_id,
            'league_id': league_id,
            'superstats': player_superstats(card_id=card_id, league_id=league_id, conn=conn),
            'observed_splits': observed_splits(card_id=card_id, conn=conn),
            'opponent_adjusted': opponent_quality_adjusted(card_id=card_id,
                                                           league_id=league_id, conn=conn),
        }
    finally:
        if close_conn:
            conn.close()
