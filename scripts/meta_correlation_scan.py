"""Meta-correlation scan — deeper patterns beyond univariate r.

Goes beyond the simple 'feature vs WAR' scan. Looks for:
  1. Non-linear transforms of top features (squared, log, sqrt)
  2. Interaction terms (feature × feature)
  3. Ratios (feature_a / feature_b)
  4. Position-specific conditional correlations
  5. Observed-vs-rating gaps (stat overperformance vs expected)
  6. Residual-of-residual after all current overlays

Run after all overlays are applied (OPS+, OBP, ISO, FIP).
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.database import get_connection


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx * dy > 0 else 0.0


def standardized_residuals(meta, target):
    mt = sum(target) / len(target)
    mm = sum(meta) / len(meta)
    st = math.sqrt(sum((x - mt) ** 2 for x in target) / len(target))
    sm = math.sqrt(sum((x - mm) ** 2 for x in meta) / len(meta))
    return [(t - mt) / st - (m - mm) / sm for t, m in zip(target, meta)]


def safe(v, default=0.0):
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def show(title, rows, thresh=0.05):
    print(f"\n{title}")
    print(f"{'feature':<36} {'r':>8} {'n':>6}")
    for name, r, n in rows:
        if abs(r) >= thresh:
            flag = " ***" if abs(r) >= 0.20 else " **" if abs(r) >= 0.15 else " *" if abs(r) >= 0.10 else ""
            print(f"  {name:<34} {r:+8.4f} {n:>6}{flag}")


def scan_batting():
    conn = get_connection()
    rows = conn.execute("""
        SELECT c.card_id, c.contact, c.gap_power, c.power, c.eye, c.avoid_ks,
               c.babip AS cbabip, c.speed, c.stealing, c.baserunning,
               c.card_value AS ovr, c.tier, c.card_type, c.position,
               c.contact_vl, c.contact_vr, c.gap_vl, c.gap_vr,
               c.power_vl, c.power_vr, c.eye_vl, c.eye_vr,
               c.of_range, c.of_error, c.of_arm,
               c.infield_range, c.infield_error, c.infield_arm,
               c.catcher_ability, c.catcher_frame, c.catcher_arm,
               c.meta_score_batting AS meta,
               bs.war, bs.pa, bs.ops, bs.ops_plus, bs.obp, bs.slg, bs.iso,
               bs.babip AS obabip, bs.league_id
        FROM cards c
        INNER JOIN batting_stats bs ON bs.card_id = c.card_id
        INNER JOIN (
            SELECT card_id, MAX(snapshot_date) mx FROM batting_stats
            WHERE card_id IS NOT NULL GROUP BY card_id, league_id
        ) latest ON bs.card_id = latest.card_id AND bs.snapshot_date = latest.mx
        WHERE bs.pa >= 150 AND bs.war IS NOT NULL AND c.card_value > 0
    """).fetchall()
    print(f"Batting sample: {len(rows)}")

    tgt = [r['war'] * 600.0 / r['pa'] for r in rows]
    meta = [r['meta'] or 0 for r in rows]
    res = standardized_residuals(meta, tgt)

    # ============================================================
    # 1. NON-LINEAR TRANSFORMS on top features
    # ============================================================
    nl_results = []
    for feat in ['contact', 'power', 'gap_power', 'eye', 'avoid_ks', 'speed',
                 'stealing', 'baserunning', 'cbabip']:
        vals = [safe(r[feat]) for r in rows]
        # vs WAR
        nl_results.append((f'{feat}', pearson(vals, tgt), len(vals)))
        nl_results.append((f'{feat}²', pearson([v**2 for v in vals], tgt), len(vals)))
        nl_results.append((f'log(1+{feat})', pearson([math.log1p(max(v, 0)) for v in vals], tgt), len(vals)))
        nl_results.append((f'sqrt({feat})', pearson([math.sqrt(max(v, 0)) for v in vals], tgt), len(vals)))
    nl_results.sort(key=lambda x: -abs(x[1]))
    show('A) NON-LINEAR TRANSFORMS (batting vs WAR/600)', nl_results[:15])

    # ============================================================
    # 2. INTERACTION TERMS between ratings + between rating×obs
    # ============================================================
    def mk(r, key):
        return safe(r[key])

    rating_keys = ['contact', 'gap_power', 'power', 'eye', 'avoid_ks', 'speed']
    obs_keys = ['ops_plus', 'obp', 'slg', 'iso', 'obabip']
    int_results = []
    # Rating × rating
    for i, a in enumerate(rating_keys):
        for b in rating_keys[i+1:]:
            prod = [mk(r, a) * mk(r, b) for r in rows]
            int_results.append((f'{a}×{b}', pearson(prod, tgt), len(prod)))
            # vs residuals too
            int_results.append((f'{a}×{b} vs RES', pearson(prod, res), len(prod)))
    # Rating × observed
    for a in rating_keys:
        for b in obs_keys:
            prod = [mk(r, a) * (safe(r[b]) if r[b] is not None else 0) for r in rows if r[b] is not None]
            t_sub = [r_['war']*600.0/r_['pa'] for r_ in rows if r_[b] is not None]
            if len(prod) < 50: continue
            int_results.append((f'{a}×{b}', pearson(prod, t_sub), len(prod)))
    int_results.sort(key=lambda x: -abs(x[1]))
    show('B) INTERACTION TERMS (batting)', int_results[:20])

    # ============================================================
    # 3. RATIOS (rating_a / rating_b)
    # ============================================================
    ratio_defs = [
        ('contact/avoid_ks', 'contact', 'avoid_ks'),
        ('power/contact', 'power', 'contact'),
        ('gap_power/power', 'gap_power', 'power'),
        ('eye/contact', 'eye', 'contact'),
        ('speed/baserunning', 'speed', 'baserunning'),
        ('stealing/speed', 'stealing', 'speed'),
        ('contact_vl/contact_vr', 'contact_vl', 'contact_vr'),
        ('power_vl/power_vr', 'power_vl', 'power_vr'),
    ]
    ratio_results = []
    for name, num, den in ratio_defs:
        vals = []
        ts = []
        for r, t in zip(rows, tgt):
            a = safe(r[num]); b = safe(r[den])
            if b > 0:
                vals.append(a / b); ts.append(t)
        if len(vals) < 50: continue
        ratio_results.append((name, pearson(vals, ts), len(vals)))
    ratio_results.sort(key=lambda x: -abs(x[1]))
    show('C) RATIO FEATURES (batting)', ratio_results, thresh=0.03)

    # ============================================================
    # 4. POSITION-SPECIFIC CONDITIONAL CORRELATIONS
    # ============================================================
    pos_results = []
    for pos in [2, 3, 4, 5, 6, 7, 8, 9, 10]:  # C through DH
        sub = [(r, t, rs) for r, t, rs in zip(rows, tgt, res) if (r['position'] == pos)]
        if len(sub) < 40: continue
        sub_rows = [x[0] for x in sub]
        sub_tgt = [x[1] for x in sub]
        pos_name = {2:'C',3:'1B',4:'2B',5:'3B',6:'SS',7:'LF',8:'CF',9:'RF',10:'DH'}.get(pos, str(pos))
        # For each rating, see conditional r
        for feat in ['contact', 'power', 'gap_power', 'eye', 'speed', 'avoid_ks']:
            vals = [safe(r[feat]) for r in sub_rows]
            r_val = pearson(vals, sub_tgt)
            if abs(r_val) >= 0.25:
                pos_results.append((f'{pos_name}:{feat}', r_val, len(sub)))
    pos_results.sort(key=lambda x: -abs(x[1]))
    show('D) POSITION-CONDITIONAL CORRELATIONS (r≥0.25)', pos_results, thresh=0.25)

    # ============================================================
    # 5. OBSERVED-vs-RATING GAP FEATURES
    #    "Does this card overperform its rating?"
    # ============================================================
    gap_results = []
    # Expected ops_plus given contact/power rating (linear fit)
    # Use a simple proxy: contact + power + gap_power = rating composite
    cp_comp = [(safe(r['contact']) + safe(r['power']) + safe(r['gap_power'])) / 3 for r in rows]
    # Observed over-performance: normalized diff of ops_plus vs its predicted
    from statistics import mean
    if cp_comp and all(r['ops_plus'] is not None for r in rows):
        # Simple OLS fit: ops_plus ~ a + b * cp_comp
        mx = mean(cp_comp); my = mean([r['ops_plus'] for r in rows])
        num = sum((x-mx)*(safe(r['ops_plus'])-my) for x,r in zip(cp_comp, rows))
        den = sum((x-mx)**2 for x in cp_comp) or 1
        b = num / den; a = my - b * mx
        predicted = [a + b * x for x in cp_comp]
        overperf = [safe(r['ops_plus']) - p for r, p in zip(rows, predicted)]
        gap_results.append(('obs_ops+ over rating', pearson(overperf, tgt), len(overperf)))
        gap_results.append(('obs_ops+ over rating vs RES', pearson(overperf, res), len(overperf)))
    # OBP delta from expected (eye composite)
    eye_comp = [safe(r['eye']) for r in rows]
    if all(r['obp'] is not None for r in rows):
        mx = mean(eye_comp); my = mean([r['obp'] for r in rows])
        num = sum((x-mx)*(safe(r['obp'])-my) for x,r in zip(eye_comp, rows))
        den = sum((x-mx)**2 for x in eye_comp) or 1
        b = num/den; a = my - b*mx
        pred = [a + b*x for x in eye_comp]
        op = [safe(r['obp']) - p for r, p in zip(rows, pred)]
        gap_results.append(('obs_obp over eye', pearson(op, tgt), len(op)))
        gap_results.append(('obs_obp over eye vs RES', pearson(op, res), len(op)))
    gap_results.sort(key=lambda x: -abs(x[1]))
    show('E) OBSERVED-OVER-RATING GAPS', gap_results, thresh=0.05)

    # ============================================================
    # 6. RESIDUAL-OF-RESIDUAL — signals still missing
    # ============================================================
    print('\nF) RESIDUAL CORRELATIONS (after all current overlays)')
    rr = []
    # ratings
    for feat in ['contact', 'gap_power', 'power', 'eye', 'avoid_ks', 'speed',
                 'stealing', 'baserunning', 'ovr', 'tier', 'pa']:
        vals = [safe(r[feat]) for r in rows]
        rr.append((feat, pearson(vals, res), len(vals)))
    # observed
    for feat in ['ops_plus', 'obp', 'slg', 'iso', 'obabip']:
        vals = [safe(r[feat]) for r in rows if r[feat] is not None]
        r_sub = [rs for r_, rs in zip(rows, res) if r_[feat] is not None]
        if len(vals) < 50: continue
        rr.append((feat, pearson(vals, r_sub), len(vals)))
    # squared features against residuals (non-linear residual)
    for feat in ['power', 'gap_power', 'contact']:
        vals = [safe(r[feat]) ** 2 for r in rows]
        rr.append((f'{feat}²', pearson(vals, res), len(vals)))
    rr.sort(key=lambda x: -abs(x[1]))
    show('', rr, thresh=0.08)


def scan_pitching():
    conn = get_connection()
    rows = conn.execute("""
        SELECT c.card_id, c.stuff, c.movement, c.control, c.p_hr, c.p_babip,
               c.stamina, c.hold, c.card_value AS ovr, c.tier, c.pitcher_role,
               c.stuff_vl, c.stuff_vr, c.movement_vl, c.movement_vr,
               c.control_vl, c.control_vr,
               c.meta_score_pitching AS meta,
               ps.war, ps.ip, ps.era_plus, ps.fip, ps.whip,
               ps.k_per_9, ps.bb_per_9, ps.hr_per_9, ps.babip AS obabip,
               ps.league_id,
               pr.fb, pr.ch, pr.cb, pr.sl, pr.si, pr.sp, pr.ct, pr.fo,
               pr.velocity, pr.pitch_count
        FROM cards c
        INNER JOIN pitching_stats ps ON ps.card_id = c.card_id
        LEFT JOIN pitch_ratings pr ON pr.card_id = c.card_id
        INNER JOIN (
            SELECT card_id, MAX(snapshot_date) mx FROM pitching_stats
            WHERE card_id IS NOT NULL GROUP BY card_id, league_id
        ) latest ON ps.card_id = latest.card_id AND ps.snapshot_date = latest.mx
        WHERE ps.ip >= 30 AND ps.war IS NOT NULL AND c.card_value > 0
    """).fetchall()
    print(f"\n{'='*70}\nPitching sample: {len(rows)}")

    tgt = [r['war'] * 200.0 / r['ip'] for r in rows]
    meta = [r['meta'] or 0 for r in rows]
    res = standardized_residuals(meta, tgt)

    # Non-linear transforms
    nl = []
    for feat in ['stuff', 'movement', 'control', 'p_hr', 'velocity', 'fb', 'sl', 'cb']:
        vals = [safe(r[feat]) for r in rows]
        nl.append((f'{feat}', pearson(vals, tgt), len(vals)))
        nl.append((f'{feat}²', pearson([v**2 for v in vals], tgt), len(vals)))
        nl.append((f'log(1+{feat})', pearson([math.log1p(max(v,0)) for v in vals], tgt), len(vals)))
    nl.sort(key=lambda x: -abs(x[1]))
    show('A) NON-LINEAR TRANSFORMS (pitching vs WAR/200)', nl[:15])

    # Interactions
    rating_keys = ['stuff', 'movement', 'control', 'p_hr', 'velocity']
    obs_keys = ['era_plus', 'fip', 'k_per_9', 'bb_per_9', 'hr_per_9']
    int_r = []
    for i, a in enumerate(rating_keys):
        for b in rating_keys[i+1:]:
            prod = [safe(r[a]) * safe(r[b]) for r in rows]
            int_r.append((f'{a}×{b}', pearson(prod, tgt), len(prod)))
            int_r.append((f'{a}×{b} vs RES', pearson(prod, res), len(prod)))
    for a in rating_keys:
        for b in obs_keys:
            pairs = [(safe(r[a]) * safe(r[b]), t) for r, t in zip(rows, tgt)
                     if r[b] is not None]
            if len(pairs) < 50: continue
            int_r.append((f'{a}×{b}', pearson([x[0] for x in pairs], [x[1] for x in pairs]),
                          len(pairs)))
    int_r.sort(key=lambda x: -abs(x[1]))
    show('B) INTERACTION TERMS (pitching)', int_r[:20])

    # Role-conditional
    role_r = []
    for role_id in [1, 2]:  # SP, RP
        sub = [(r, t, rs) for r, t, rs in zip(rows, tgt, res)
               if r['pitcher_role'] == role_id]
        if len(sub) < 40: continue
        role_name = 'SP' if role_id == 1 else 'RP'
        for feat in ['stuff', 'movement', 'control', 'velocity', 'fb', 'stamina']:
            vals = [safe(x[0][feat]) for x in sub]
            sub_t = [x[1] for x in sub]
            r_val = pearson(vals, sub_t)
            if abs(r_val) >= 0.20:
                role_r.append((f'{role_name}:{feat}', r_val, len(sub)))
    role_r.sort(key=lambda x: -abs(x[1]))
    show('C) ROLE-CONDITIONAL CORRELATIONS (pitching)', role_r, thresh=0.20)

    # Pitch arsenal compound scores
    arsenal_r = []
    # Best pitch, 2nd best, etc
    pitch_keys = ['fb', 'ch', 'cb', 'sl', 'si', 'sp', 'ct', 'fo']
    for r in rows:
        ratings = sorted([safe(r[p]) for p in pitch_keys if r[p] is not None],
                         reverse=True)
    # Compute compound features
    arsenal_features = {}
    for feat_name in ['pitch_sum', 'pitch_mean_top3', 'pitch_count_plus',
                      'pitch_count_elite', 'arsenal_gap']:
        arsenal_features[feat_name] = []
    for r in rows:
        ratings = sorted([safe(r[p]) for p in pitch_keys if r[p] is not None],
                         reverse=True)
        if not ratings:
            for f in arsenal_features:
                arsenal_features[f].append(0)
            continue
        arsenal_features['pitch_sum'].append(sum(ratings))
        arsenal_features['pitch_mean_top3'].append(sum(ratings[:3]) / min(3, len(ratings)))
        arsenal_features['pitch_count_plus'].append(sum(1 for x in ratings if x >= 70))
        arsenal_features['pitch_count_elite'].append(sum(1 for x in ratings if x >= 85))
        arsenal_features['arsenal_gap'].append(ratings[0] - (ratings[1] if len(ratings)>1 else 0))
    for name, vals in arsenal_features.items():
        arsenal_r.append((name, pearson(vals, tgt), len(vals)))
        arsenal_r.append((f'{name} vs RES', pearson(vals, res), len(vals)))
    arsenal_r.sort(key=lambda x: -abs(x[1]))
    show('D) PITCH ARSENAL COMPOUND FEATURES', arsenal_r, thresh=0.08)

    # Residuals
    rr = []
    for feat in ['stuff', 'movement', 'control', 'p_hr', 'velocity', 'ovr', 'tier',
                 'era_plus', 'fip', 'k_per_9', 'bb_per_9', 'hr_per_9', 'whip',
                 'obabip', 'fb', 'sl', 'cb', 'stamina', 'hold']:
        vals, r_sub = [], []
        for r_, rs in zip(rows, res):
            v = r_[feat] if feat in r_.keys() else None
            if v is None: continue
            vals.append(safe(v)); r_sub.append(rs)
        if len(vals) < 50: continue
        rr.append((feat, pearson(vals, r_sub), len(vals)))
    rr.sort(key=lambda x: -abs(x[1]))
    show('E) RESIDUAL CORRELATIONS (pitching, after FIP overlay)', rr, thresh=0.08)


if __name__ == '__main__':
    print('=' * 80)
    print('META-CORRELATION SCAN: non-linear, interactions, conditional')
    print('=' * 80)
    scan_batting()
    scan_pitching()
