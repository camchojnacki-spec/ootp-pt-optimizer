"""Comprehensive correlation scan — every available feature vs WAR.

Surfaces any feature that correlates with WAR but isn't in the meta
formula yet. Covers:
  - All card rating columns (including split-vL/vR, weak-side mins, split gaps)
  - Fielding and position ratings
  - Card metadata (age, tier, card_type, badge, sub_type, series)
  - Game-log derived (EV, LD%, hard-hit, barrel, pitches/PA, observed K/BB)
  - Box-score clutch events (2OUT_RBI, LOB_RISP_2OUT, ERRORS, SB, INHERITED)
  - Opponent quality (avg ERA+ faced)

Pooled across lb124 + i76 for sample size. Features with |r| > 0.10 that
aren't already a meta input are flagged as new overlay candidates.
"""
import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from app.core.database import get_db_path


def pearson(xs, ys):
    xs = np.array([float(x) if x is not None else np.nan for x in xs])
    ys = np.array(ys, dtype=float)
    keep = ~np.isnan(xs) & ~np.isnan(ys)
    if keep.sum() < 10:
        return None, 0
    xs = xs[keep]; ys = ys[keep]
    if xs.std() < 1e-9 or ys.std() < 1e-9:
        return None, len(xs)
    return float(np.corrcoef(xs, ys)[0, 1]), int(keep.sum())


def _to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return np.nan
    if '-' in s:
        parts = s.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2.0
        except (ValueError, IndexError):
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def scan_batting():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row

    print("=" * 80)
    print("BATTING CORRELATION SCAN (pooled lb124 + i76, PA >= 150)")
    print("=" * 80)

    rows = conn.execute("""
        SELECT c.card_id, c.card_value AS ovr, c.card_type, c.tier, c.age,
               c.contact, c.gap_power, c.power, c.eye, c.avoid_ks, c.babip,
               c.contact_vl, c.contact_vr, c.gap_vl, c.gap_vr,
               c.power_vl, c.power_vr, c.eye_vl, c.eye_vr,
               c.avoid_ks_vl, c.avoid_ks_vr, c.babip_vl, c.babip_vr,
               c.speed, c.stealing, c.baserunning, c.sac_bunt, c.bunt_for_hit,
               c.of_range, c.of_error, c.of_arm,
               c.infield_range, c.infield_error, c.infield_arm, c.dp,
               c.catcher_ability, c.catcher_frame, c.catcher_arm,
               c.pos_rating_c, c.pos_rating_1b, c.pos_rating_2b,
               c.pos_rating_3b, c.pos_rating_ss, c.pos_rating_lf,
               c.pos_rating_cf, c.pos_rating_rf,
               bs.pa, bs.ab, bs.war, bs.ops, bs.ops_plus, bs.babip AS obs_babip,
               bs.iso, bs.league_id
        FROM cards c
        INNER JOIN batting_stats bs ON bs.card_id = c.card_id
        INNER JOIN (
            SELECT card_id, MAX(snapshot_date) mx FROM batting_stats
            WHERE card_id IS NOT NULL GROUP BY card_id, league_id
        ) latest ON bs.card_id = latest.card_id AND bs.snapshot_date = latest.mx
        WHERE bs.pa >= 150 AND bs.war IS NOT NULL AND c.card_value > 0
    """).fetchall()

    n_total = len(rows)
    print(f"Total sample: {n_total}")
    print()

    # Target: WAR/600
    target = [r['war'] * 600.0 / r['pa'] for r in rows]

    # Base rating correlations (known; re-check to frame the rest)
    already = {
        'contact', 'gap_power', 'power', 'eye', 'avoid_ks', 'babip',
        'contact_vl', 'contact_vr', 'power_vl', 'power_vr', 'eye_vl', 'eye_vr',
        'gap_vl', 'gap_vr', 'avoid_ks_vl', 'avoid_ks_vr', 'babip_vl', 'babip_vr',
        'speed', 'stealing', 'baserunning',
        'of_range', 'of_error', 'of_arm',
        'infield_range', 'infield_error', 'infield_arm',
        'catcher_ability', 'catcher_frame', 'catcher_arm',
        'pos_rating_c', 'pos_rating_1b', 'pos_rating_2b', 'pos_rating_3b',
        'pos_rating_ss', 'pos_rating_lf', 'pos_rating_cf', 'pos_rating_rf',
        'ovr',
    }

    features = {}
    # Raw columns
    for col in ['ovr', 'age', 'tier',
                'contact', 'gap_power', 'power', 'eye', 'avoid_ks', 'babip',
                'contact_vl', 'contact_vr', 'power_vl', 'power_vr',
                'eye_vl', 'eye_vr', 'gap_vl', 'gap_vr',
                'avoid_ks_vl', 'avoid_ks_vr', 'babip_vl', 'babip_vr',
                'speed', 'stealing', 'baserunning', 'sac_bunt', 'bunt_for_hit',
                'of_range', 'of_error', 'of_arm', 'dp',
                'infield_range', 'infield_error', 'infield_arm',
                'catcher_ability', 'catcher_frame', 'catcher_arm',
                'pos_rating_c', 'pos_rating_1b', 'pos_rating_2b',
                'pos_rating_3b', 'pos_rating_ss', 'pos_rating_lf',
                'pos_rating_cf', 'pos_rating_rf',
                'obs_babip', 'iso']:
        features[col] = [r[col] for r in rows]

    # Derived: platoon split gaps + weak-side mins
    for rating in ['contact', 'power', 'eye', 'gap', 'avoid_ks', 'babip']:
        vl = f'{rating}_vl'
        vr = f'{rating}_vr'
        features[f'{rating}_split_gap'] = [
            abs((r[vl] or 0) - (r[vr] or 0)) for r in rows]
        features[f'{rating}_weak_side'] = [
            min(r[vl] or 0, r[vr] or 0) for r in rows]
        features[f'{rating}_strong_side'] = [
            max(r[vl] or 0, r[vr] or 0) for r in rows]

    # Game-log derived aggregates per card_id
    gl = conn.execute("""
        SELECT batter_card_id AS cid, COUNT(*) ab,
               AVG(exit_velocity) ev_avg,
               SUM(CASE WHEN LOWER(batted_ball_type) LIKE 'line%' THEN 1 ELSE 0 END)
                 * 1.0 / NULLIF(SUM(CASE WHEN batted_ball_type IS NOT NULL THEN 1 ELSE 0 END), 0)
                 * 100 AS ld_pct,
               SUM(CASE WHEN LOWER(batted_ball_type) LIKE 'ground%' THEN 1 ELSE 0 END)
                 * 1.0 / NULLIF(SUM(CASE WHEN batted_ball_type IS NOT NULL THEN 1 ELSE 0 END), 0)
                 * 100 AS gb_pct,
               SUM(CASE WHEN outcome = 'K' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS k_pct,
               SUM(CASE WHEN outcome = 'BB' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS bb_pct,
               SUM(CASE WHEN exit_velocity >= 95 THEN 1 ELSE 0 END) * 100.0
                 / NULLIF(SUM(CASE WHEN exit_velocity IS NOT NULL THEN 1 ELSE 0 END), 0)
                 AS barrel_pct,
               SUM(CASE WHEN exit_velocity >= 90 THEN 1 ELSE 0 END) * 100.0
                 / NULLIF(SUM(CASE WHEN exit_velocity IS NOT NULL THEN 1 ELSE 0 END), 0)
                 AS hard_pct,
               AVG(pitches_seen) AS pitches_per_pa
        FROM game_log_at_bats
        WHERE batter_card_id IS NOT NULL
        GROUP BY batter_card_id
    """).fetchall()
    gl_map = {g['cid']: dict(g) for g in gl}

    for key in ['ev_avg', 'ld_pct', 'gb_pct', 'k_pct', 'bb_pct',
                'barrel_pct', 'hard_pct', 'pitches_per_pa']:
        features[f'gl_{key}'] = [
            (gl_map.get(r['card_id']) or {}).get(key) for r in rows]

    # Clutch aggregates per card
    cl = conn.execute("""
        SELECT card_id,
               SUM(CASE WHEN event_type = '2OUT_RBI' THEN 1 ELSE 0 END) rbi,
               SUM(CASE WHEN event_type = 'LOB_RISP_2OUT' THEN 1 ELSE 0 END) lob,
               SUM(CASE WHEN event_type = 'ERROR' THEN 1 ELSE 0 END) errs,
               SUM(CASE WHEN event_type = 'SB' THEN 1 ELSE 0 END) sb_events,
               SUM(CASE WHEN event_type = 'GIDP' THEN 1 ELSE 0 END) gidp,
               SUM(CASE WHEN event_type = 'DOUBLE' THEN 1 ELSE 0 END) doubles,
               SUM(CASE WHEN event_type = 'HR' THEN 1 ELSE 0 END) hr_events,
               SUM(CASE WHEN event_type = 'SAC_FLY' THEN 1 ELSE 0 END) sf
        FROM game_clutch_events WHERE card_id IS NOT NULL GROUP BY card_id
    """).fetchall()
    cl_map = {c['card_id']: dict(c) for c in cl}
    for key in ['rbi', 'lob', 'errs', 'sb_events', 'gidp', 'doubles', 'hr_events', 'sf']:
        features[f'clutch_{key}'] = [
            (cl_map.get(r['card_id']) or {}).get(key) for r in rows]
    # Normalized clutch rates (per game_batting PA, not cumulative PA — fair)
    gb_pa = conn.execute("""
        SELECT card_id, SUM(ab+bb) pa FROM game_batting
        WHERE card_id IS NOT NULL AND ab > 0 GROUP BY card_id
    """).fetchall()
    pa_map = {r['card_id']: r['pa'] for r in gb_pa}
    for r in rows:
        pa_log = pa_map.get(r['card_id']) or 0
    features['clutch_net_per_100pa'] = [
        (((cl_map.get(r['card_id']) or {}).get('rbi') or 0) -
         0.5 * ((cl_map.get(r['card_id']) or {}).get('lob') or 0)) * 100.0
        / (pa_map.get(r['card_id']) or 1) if pa_map.get(r['card_id']) else 0
        for r in rows]

    # Compute correlations
    results = []
    for name, vals in features.items():
        r, n = pearson(vals, target)
        if r is None:
            continue
        is_new = name not in already and not any(a in name for a in already)
        results.append((abs(r), name, r, n, is_new))
    results.sort(reverse=True)

    print(f"{'Feature':28s} {'r':>8s} {'n':>5s}  new?")
    for abs_r, name, r, n, is_new in results:
        if abs_r < 0.05:
            continue
        star = ' ***' if abs_r >= 0.15 else (' **' if abs_r >= 0.10 else ' *')
        new_mark = ' ← NOT in meta yet' if is_new else ''
        print(f"  {name:28s} {r:+.4f} {n:>5d}{star}{new_mark}")


def scan_pitching():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row

    print()
    print("=" * 80)
    print("PITCHING CORRELATION SCAN (pooled, IP >= 30)")
    print("=" * 80)

    rows = conn.execute("""
        SELECT c.card_id, c.card_value AS ovr, c.card_type, c.tier, c.age,
               c.stuff, c.movement, c.control, c.p_hr, c.p_babip,
               c.stuff_vl, c.stuff_vr, c.movement_vl, c.movement_vr,
               c.control_vl, c.control_vr, c.p_hr_vl, c.p_hr_vr,
               c.p_babip_vl, c.p_babip_vr,
               c.stamina, c.hold, c.velocity, c.pitcher_role,
               ps.ip, ps.war, ps.era, ps.era_plus, ps.whip, ps.league_id
        FROM cards c
        INNER JOIN pitching_stats ps ON ps.card_id = c.card_id
        INNER JOIN (
            SELECT card_id, MAX(snapshot_date) mx FROM pitching_stats
            WHERE card_id IS NOT NULL GROUP BY card_id, league_id
        ) latest ON ps.card_id = latest.card_id AND ps.snapshot_date = latest.mx
        WHERE ps.ip >= 30 AND ps.war IS NOT NULL AND c.card_value > 0
    """).fetchall()

    n_total = len(rows)
    print(f"Total sample: {n_total}")
    print()

    target = [r['war'] * 200.0 / r['ip'] for r in rows]

    already = {
        'stuff', 'movement', 'control', 'p_hr', 'stamina', 'hold',
        'stuff_vl', 'stuff_vr', 'movement_vl', 'movement_vr',
        'control_vl', 'control_vr',
        'stuff_x_movement', 'stuff_x_control', 'movement_x_control',
        'ovr',
    }

    features = {}
    for col in ['ovr', 'age', 'tier',
                'stuff', 'movement', 'control', 'p_hr', 'p_babip',
                'stuff_vl', 'stuff_vr', 'movement_vl', 'movement_vr',
                'control_vl', 'control_vr', 'p_hr_vl', 'p_hr_vr',
                'p_babip_vl', 'p_babip_vr',
                'stamina', 'hold']:
        features[col] = [r[col] for r in rows]

    # Velocity (stored as "94-96" range)
    features['velocity'] = [_to_float(r['velocity']) for r in rows]

    # Interactions
    features['stuff_x_movement'] = [
        (r['stuff'] or 0) * (r['movement'] or 0) for r in rows]
    features['stuff_x_control'] = [
        (r['stuff'] or 0) * (r['control'] or 0) for r in rows]
    features['movement_x_control'] = [
        (r['movement'] or 0) * (r['control'] or 0) for r in rows]
    features['stuff_x_stamina'] = [
        (r['stuff'] or 0) * (r['stamina'] or 0) for r in rows]

    # Platoon splits + weak sides
    for rating in ['stuff', 'movement', 'control', 'p_hr', 'p_babip']:
        vl, vr = f'{rating}_vl', f'{rating}_vr'
        features[f'{rating}_split_gap'] = [
            abs((r[vl] or 0) - (r[vr] or 0)) for r in rows]
        features[f'{rating}_weak_side'] = [
            min(r[vl] or 0, r[vr] or 0) for r in rows]

    # Pitch arsenal per card_id
    pr = conn.execute("""
        SELECT pr.card_id AS card_id, fb, ch, cb, sl, si, sp, ct, fo, cc, sc, kc, kn,
               pitch_count, velocity
        FROM pitch_ratings pr
        INNER JOIN (SELECT card_id, MAX(snapshot_date) mx FROM pitch_ratings
                    WHERE card_id IS NOT NULL GROUP BY card_id) l
          ON pr.card_id = l.card_id AND pr.snapshot_date = l.mx
        WHERE pr.card_id IS NOT NULL
    """).fetchall()
    pr_map = {p['card_id']: dict(p) for p in pr}
    for key in ['fb', 'ch', 'cb', 'sl', 'si', 'sp', 'ct', 'fo', 'cc',
                'sc', 'kc', 'kn', 'pitch_count']:
        features[f'pitch_{key}'] = [
            (pr_map.get(r['card_id']) or {}).get(key) for r in rows]
    # Max pitch quality (best single pitch)
    features['pitch_best'] = [
        max([(pr_map.get(r['card_id']) or {}).get(k) or 0
             for k in ['fb', 'ch', 'cb', 'sl', 'si', 'sp', 'ct', 'fo']])
        for r in rows
    ]

    # Game-log per pitcher
    gl = conn.execute("""
        SELECT pitcher_card_id AS cid, COUNT(*) bf,
               AVG(exit_velocity) ev_allowed,
               SUM(CASE WHEN LOWER(batted_ball_type) LIKE 'line%' THEN 1 ELSE 0 END)
                 * 1.0 / NULLIF(SUM(CASE WHEN batted_ball_type IS NOT NULL THEN 1 ELSE 0 END), 0)
                 * 100 AS ld_pct_allowed,
               SUM(CASE WHEN LOWER(batted_ball_type) LIKE 'ground%' THEN 1 ELSE 0 END)
                 * 1.0 / NULLIF(SUM(CASE WHEN batted_ball_type IS NOT NULL THEN 1 ELSE 0 END), 0)
                 * 100 AS gb_pct_allowed,
               SUM(CASE WHEN outcome = 'K' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS k_pct,
               SUM(CASE WHEN outcome = 'BB' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS bb_pct,
               SUM(CASE WHEN outcome = 'HR' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS hr_pct,
               SUM(CASE WHEN exit_velocity >= 95 THEN 1 ELSE 0 END) * 100.0
                 / NULLIF(SUM(CASE WHEN exit_velocity IS NOT NULL THEN 1 ELSE 0 END), 0)
                 AS barrel_pct_allowed,
               AVG(pitches_seen) AS pitches_per_bf
        FROM game_log_at_bats WHERE pitcher_card_id IS NOT NULL
        GROUP BY pitcher_card_id
    """).fetchall()
    gl_map = {g['cid']: dict(g) for g in gl}
    for key in ['ev_allowed', 'ld_pct_allowed', 'gb_pct_allowed',
                'k_pct', 'bb_pct', 'hr_pct', 'barrel_pct_allowed',
                'pitches_per_bf']:
        features[f'gl_{key}'] = [
            (gl_map.get(r['card_id']) or {}).get(key) for r in rows]

    # Box score per-game aggregates
    bs = conn.execute("""
        SELECT card_id,
               AVG(CASE WHEN game_score IS NOT NULL THEN game_score END) AS avg_gs,
               SUM(ground_outs) * 1.0 / NULLIF(SUM(fly_outs), 0) AS go_fo_ratio,
               SUM(CASE WHEN role_flag = 'W' THEN 1 ELSE 0 END) AS w_count,
               SUM(CASE WHEN role_flag = 'L' THEN 1 ELSE 0 END) AS l_count,
               SUM(CASE WHEN role_flag = 'SV' THEN 1 ELSE 0 END) AS sv_count,
               SUM(CASE WHEN role_flag = 'BS' THEN 1 ELSE 0 END) AS bs_count,
               SUM(CASE WHEN role_flag = 'HLD' THEN 1 ELSE 0 END) AS hld_count,
               AVG(batters_faced) AS avg_bf
        FROM game_pitching WHERE card_id IS NOT NULL GROUP BY card_id
    """).fetchall()
    bs_map = {r['card_id']: dict(r) for r in bs}
    for key in ['avg_gs', 'go_fo_ratio', 'w_count', 'l_count',
                'sv_count', 'bs_count', 'hld_count', 'avg_bf']:
        features[f'box_{key}'] = [
            (bs_map.get(r['card_id']) or {}).get(key) for r in rows]

    # Inherited runner discipline — true rate
    ir = conn.execute("""
        SELECT card_id,
               SUM(CASE WHEN event_type = 'INHERITED_RUNNERS' THEN event_count ELSE 0 END) inh,
               SUM(CASE WHEN event_type = 'INHERITED_SCORED' THEN event_count ELSE 0 END) sc
        FROM game_clutch_events WHERE card_id IS NOT NULL
          AND event_type IN ('INHERITED_RUNNERS', 'INHERITED_SCORED')
        GROUP BY card_id
    """).fetchall()
    ir_map = {}
    for r in ir:
        inh = r['inh'] or 0
        sc = r['sc'] or 0
        ir_map[r['card_id']] = {
            'inherited_total': inh,
            'inherited_scored': sc,
            'inherited_score_rate': (sc / inh) if inh >= 3 else None,
            'inherited_stranded_rate': ((inh - sc) / inh) if inh >= 3 else None,
        }
    for key in ['inherited_total', 'inherited_scored',
                'inherited_score_rate', 'inherited_stranded_rate']:
        features[key] = [
            (ir_map.get(r['card_id']) or {}).get(key) for r in rows]

    # Results
    results = []
    for name, vals in features.items():
        r, n = pearson(vals, target)
        if r is None:
            continue
        is_new = not any(a == name or a in name for a in already)
        results.append((abs(r), name, r, n, is_new))
    results.sort(reverse=True)

    print(f"{'Feature':28s} {'r':>8s} {'n':>5s}  new?")
    for abs_r, name, r, n, is_new in results:
        if abs_r < 0.05:
            continue
        star = ' ***' if abs_r >= 0.15 else (' **' if abs_r >= 0.10 else ' *')
        new_mark = ' ← NOT in meta yet' if is_new else ''
        print(f"  {name:28s} {r:+.4f} {n:>5d}{star}{new_mark}")


if __name__ == '__main__':
    scan_batting()
    scan_pitching()
