"""Residual analysis — what does the current meta miss?

Pools batters from lb124 + i76, fits WAR ~ meta (linear), then regresses
residuals against every available feature. Features with |r| > 0.10 are
real signals the meta formula isn't capturing.
"""
import sys, sqlite3
sys.path.insert(0, '.')
import numpy as np
from app.core.database import get_db_path

CARD_TYPE_NAMES = {
    1: 'Live', 2: 'Legend', 3: 'Hardware Heroes', 4: 'Unsung Heroes',
    5: 'Historical All-Star', 6: 'Future Legend', 7: 'Snapshot',
    8: 'Veteran Presence', 9: 'All-Time Legend', 10: 'Live Reward',
}

POS_COLS = ['pos_rating_c', 'pos_rating_1b', 'pos_rating_2b', 'pos_rating_3b',
            'pos_rating_ss', 'pos_rating_lf', 'pos_rating_cf', 'pos_rating_rf']

conn = sqlite3.connect(get_db_path())
conn.row_factory = sqlite3.Row

rows = conn.execute(f"""
    SELECT c.card_id, c.meta_score_batting m, c.card_value ovr, c.card_type,
           c.card_sub_type, c.card_badge, c.tier_name, c.age, c.bats,
           c.contact, c.gap_power, c.power, c.eye, c.avoid_ks, c.babip,
           c.contact_vl, c.contact_vr, c.power_vl, c.power_vr, c.eye_vl, c.eye_vr,
           c.speed, c.stealing, c.baserunning,
           {','.join('c.'+x for x in POS_COLS)},
           c.position, c.position_name,
           bs.war, bs.pa, bs.ops_plus, bs.league_id
    FROM cards c
    JOIN batting_stats bs ON bs.card_id = c.card_id
    JOIN (SELECT card_id, MAX(snapshot_date) mx FROM batting_stats
          WHERE card_id IS NOT NULL GROUP BY card_id, league_id) l
      ON bs.card_id = l.card_id AND bs.snapshot_date = l.mx
    WHERE bs.pa >= 150 AND c.meta_score_batting > 0
      AND c.card_value > 0 AND bs.war IS NOT NULL
""").fetchall()

print(f'Pooled batting sample (lb124 + i76): n={len(rows)}')

metas = np.array([r['m'] for r in rows])
wars = np.array([r['war'] * 600.0 / r['pa'] for r in rows])
slope, intercept = np.polyfit(metas, wars, 1)
preds = slope * metas + intercept
residuals = wars - preds
print(f'Baseline fit: WAR/600 = {slope:.5f} * meta + {intercept:.3f}')
print(f'Pearson(meta, WAR): r = {np.corrcoef(metas, wars)[0,1]:+.4f}')
print(f'Residual RMSE: {np.sqrt(np.mean(residuals**2)):.3f} WAR/600')

def correl_resid(xs, resid):
    xs = np.array([float(x) if x is not None else np.nan for x in xs])
    keep = ~np.isnan(xs)
    if keep.sum() < 10:
        return None, 0
    xs2 = xs[keep]
    r2 = resid[keep]
    if xs2.std() < 1e-9:
        return None, len(xs2)
    return float(np.corrcoef(xs2, r2)[0, 1]), len(xs2)

features = {
    'OVR': [r['ovr'] for r in rows],
    'age': [r['age'] for r in rows],
    'contact': [r['contact'] for r in rows],
    'gap_power': [r['gap_power'] for r in rows],
    'power': [r['power'] for r in rows],
    'eye': [r['eye'] for r in rows],
    'avoid_ks': [r['avoid_ks'] for r in rows],
    'babip': [r['babip'] for r in rows],
    'speed': [r['speed'] for r in rows],
    'stealing': [r['stealing'] for r in rows],
    'baserunning': [r['baserunning'] for r in rows],
    'meta (self)': [r['m'] for r in rows],
    'platoon_split_con': [abs((r['contact_vl'] or 0) - (r['contact_vr'] or 0)) for r in rows],
    'platoon_split_pow': [abs((r['power_vl'] or 0) - (r['power_vr'] or 0)) for r in rows],
    'contact_weak_side': [min(r['contact_vl'] or 0, r['contact_vr'] or 0) for r in rows],
    'power_weak_side':   [min(r['power_vl'] or 0, r['power_vr'] or 0) for r in rows],
    'multi_pos_count':   [sum(1 for k in POS_COLS if (r[k] or 0) >= 20) for r in rows],
    'max_pos_rating':    [max([r[k] or 0 for k in POS_COLS]) for r in rows],
    'ops_plus':          [r['ops_plus'] for r in rows],
    'pa':                [r['pa'] for r in rows],
}

print()
print('FEATURE CORRELATION WITH RESIDUAL (positive = predicts beating meta):')
print('  Features sorted by |r|. |r|>0.10 = ** (real signal), |r|>0.15 = *** (strong)')
results = []
for name, vals in features.items():
    c, n = correl_resid(vals, residuals)
    if c is None:
        continue
    results.append((abs(c), name, c, n))
results.sort(reverse=True)
for _, name, c, n in results:
    if abs(c) > 0.15:
        sig = '***'
    elif abs(c) > 0.10:
        sig = '**'
    elif abs(c) > 0.05:
        sig = '*'
    else:
        sig = ''
    print(f'  {name:24s} r={c:+.4f}  n={n:>4}  {sig}')

print()
print('CATEGORICAL RESIDUAL MEANS (positive = this bucket beats meta on average):')
for cat_name, getter in [
    ('card_type',     lambda r: CARD_TYPE_NAMES.get(r['card_type'], f"type={r['card_type']}")),
    ('tier',          lambda r: r['tier_name'] or '(none)'),
    ('card_badge',    lambda r: r['card_badge'] or '(none)'),
    ('card_sub_type', lambda r: r['card_sub_type'] or '(none)'),
    ('bats',          lambda r: r['bats'] or '?'),
    ('position_name', lambda r: r['position_name'] or '?'),
    ('league',        lambda r: r['league_id'] or '?'),
]:
    print(f'\n  By {cat_name}:')
    groups = {}
    for row, resid in zip(rows, residuals):
        k = getter(row)
        groups.setdefault(k, []).append(resid)
    for k, rs in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(rs) < 5:
            continue
        m = float(np.mean(rs))
        se = float(np.std(rs, ddof=0) / (len(rs) ** 0.5))
        if abs(m) > 2 * se:
            sig = ' ***'
        elif abs(m) > se:
            sig = ' *'
        else:
            sig = ''
        print(f'    {str(k):30s}  n={len(rs):>4}  mean_resid={m:+.3f} WAR/600  SE={se:.3f}{sig}')
