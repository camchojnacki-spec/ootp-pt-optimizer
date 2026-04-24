"""Pitching residual analysis — what does the pitching meta miss?"""
import sys, sqlite3
sys.path.insert(0, '.')
import numpy as np
from app.core.database import get_db_path

CARD_TYPE_NAMES = {
    1: 'Live', 2: 'Legend', 3: 'Hardware Heroes', 4: 'Unsung Heroes',
    5: 'Historical All-Star', 6: 'Future Legend', 7: 'Snapshot',
    8: 'Veteran Presence', 9: 'All-Time Legend', 10: 'Live Reward',
}

conn = sqlite3.connect(get_db_path())
conn.row_factory = sqlite3.Row

rows = conn.execute("""
    SELECT c.card_id, c.meta_score_pitching m, c.card_value ovr, c.card_type,
           c.card_sub_type, c.card_badge, c.tier_name, c.age, c.throws,
           c.stuff, c.movement, c.control, c.p_hr, c.p_babip,
           c.stuff_vl, c.stuff_vr, c.movement_vl, c.movement_vr,
           c.control_vl, c.control_vr, c.p_hr_vl, c.p_hr_vr,
           c.stamina, c.hold, c.velocity,
           c.pitcher_role_name, c.pitcher_role,
           ps.war, ps.ip, ps.era_plus, ps.league_id
    FROM cards c
    JOIN pitching_stats ps ON ps.card_id = c.card_id
    JOIN (SELECT card_id, MAX(snapshot_date) mx FROM pitching_stats
          WHERE card_id IS NOT NULL GROUP BY card_id, league_id) l
      ON ps.card_id = l.card_id AND ps.snapshot_date = l.mx
    WHERE ps.ip >= 30 AND c.meta_score_pitching > 0
      AND c.card_value > 0 AND ps.war IS NOT NULL
""").fetchall()

print(f'Pooled pitching sample (lb124 + i76): n={len(rows)}')

metas = np.array([r['m'] for r in rows])
wars = np.array([r['war'] * 200.0 / r['ip'] for r in rows])
slope, intercept = np.polyfit(metas, wars, 1)
preds = slope * metas + intercept
residuals = wars - preds
print(f'Baseline fit: WAR/200 = {slope:.5f} * meta + {intercept:.3f}')
print(f'Pearson(meta, WAR): r = {np.corrcoef(metas, wars)[0,1]:+.4f}')
print(f'Residual RMSE: {np.sqrt(np.mean(residuals**2)):.3f} WAR/200')

def _to_float(x):
    """Coerce to float, handling range strings like '94-96' by taking mean."""
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

def correl_resid(xs, resid):
    xs = np.array([_to_float(x) for x in xs])
    keep = ~np.isnan(xs)
    if keep.sum() < 10:
        return None, 0
    xs2 = xs[keep]; r2 = resid[keep]
    if xs2.std() < 1e-9:
        return None, len(xs2)
    return float(np.corrcoef(xs2, r2)[0, 1]), len(xs2)

features = {
    'OVR': [r['ovr'] for r in rows],
    'age': [r['age'] for r in rows],
    'stuff': [r['stuff'] for r in rows],
    'movement': [r['movement'] for r in rows],
    'control': [r['control'] for r in rows],
    'p_hr': [r['p_hr'] for r in rows],
    'p_babip': [r['p_babip'] for r in rows],
    'stamina': [r['stamina'] for r in rows],
    'hold': [r['hold'] for r in rows],
    'velocity': [r['velocity'] for r in rows],
    'stuff_x_movement': [(r['stuff'] or 0) * (r['movement'] or 0) for r in rows],
    'stuff_x_control': [(r['stuff'] or 0) * (r['control'] or 0) for r in rows],
    'movement_x_control': [(r['movement'] or 0) * (r['control'] or 0) for r in rows],
    'platoon_split_stuff': [abs((r['stuff_vl'] or 0) - (r['stuff_vr'] or 0)) for r in rows],
    'platoon_split_mov':   [abs((r['movement_vl'] or 0) - (r['movement_vr'] or 0)) for r in rows],
    'platoon_split_ctrl':  [abs((r['control_vl'] or 0) - (r['control_vr'] or 0)) for r in rows],
    'stuff_weak_side':  [min(r['stuff_vl'] or 0, r['stuff_vr'] or 0) for r in rows],
    'meta (self)': [r['m'] for r in rows],
    'era_plus': [r['era_plus'] for r in rows],
    'ip': [r['ip'] for r in rows],
}

print()
print('FEATURE CORRELATION WITH PITCHING RESIDUAL:')
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
print('CATEGORICAL RESIDUAL MEANS:')
for cat_name, getter in [
    ('card_type',     lambda r: CARD_TYPE_NAMES.get(r['card_type'], f"type={r['card_type']}")),
    ('tier',          lambda r: r['tier_name'] or '(none)'),
    ('throws',        lambda r: r['throws'] or '?'),
    ('pitcher_role',  lambda r: r['pitcher_role_name'] or '?'),
    ('league',        lambda r: r['league_id'] or '?'),
]:
    print(f'\n  By {cat_name}:')
    groups = {}
    for row, resid in zip(rows, residuals):
        k = getter(row)
        groups.setdefault(k, []).append(resid)
    for k, rs in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(rs) < 5: continue
        m = float(np.mean(rs))
        se = float(np.std(rs, ddof=0) / (len(rs) ** 0.5))
        if abs(m) > 2 * se:
            sig = ' ***'
        elif abs(m) > se:
            sig = ' *'
        else:
            sig = ''
        print(f'    {str(k):30s}  n={len(rs):>4}  mean_resid={m:+.3f} WAR/200  SE={se:.3f}{sig}')
