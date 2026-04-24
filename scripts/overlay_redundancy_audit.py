"""Overlay-redundancy audit.

Compute per-card contributions of every observed-stat overlay, then:

    1. Cross-correlation matrix — how much each overlay predicts the others.
       If two overlays have |r| > 0.8, they're measuring the same thing
       and the additive formula is double-counting.
    2. Marginal r with WAR — univariate target correlation.
    3. Partial r with WAR after controlling for the BEST single overlay —
       does each additional overlay still add signal, or just noise?
    4. Sum-of-overlays distribution — how compressed is the "good offense"
       effect across many overlays? (Bichette's -66 total = 5 stacked
       negative signals on the same underperformance.)

Run: python scripts/overlay_redundancy_audit.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.database import get_connection
from app.core.meta_scoring import (
    _calc_performance_adjustment_batting,
    _calc_superstat_overlay_batting,
    _calc_clutch_overlay_batting,
    _calc_iso_overlay_batting,
    _calc_opsplus_overlay_batting,
    _calc_obp_overlay_batting,
    _calc_babip_overlay_batting,
    _calc_overperformance_overlay_batting,
)


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


def partial_r(x, y, control):
    """Partial correlation of x and y controlling for control.

    r(x,y|control) = (r_xy - r_xc * r_yc) / sqrt((1 - r_xc²)(1 - r_yc²))
    """
    r_xy = pearson(x, y)
    r_xc = pearson(x, control)
    r_yc = pearson(y, control)
    denom = math.sqrt(max(1e-9, (1 - r_xc * r_xc) * (1 - r_yc * r_yc)))
    return (r_xy - r_xc * r_yc) / denom if denom > 0 else 0.0


def build_overlay_matrix():
    conn = get_connection()
    # Load all overlay inputs per card_id from the ingestion-equivalent queries.
    rows = conn.execute("""
        SELECT c.card_id, c.card_title,
               c.contact, c.gap_power, c.power, c.eye, c.avoid_ks, c.babip,
               c.speed, c.stealing, c.baserunning,
               c.contact_vl, c.contact_vr, c.power_vl, c.power_vr,
               c.eye_vl, c.eye_vr,
               c.tier, c.card_type, c.position
        FROM cards c
        WHERE c.owned >= 0 AND c.position IS NOT NULL
          AND c.pitcher_role IS NULL
    """).fetchall()
    cards = {r['card_id']: dict(r) for r in rows}

    # Pull observed stats
    obs_bat = {}
    for r in conn.execute("""
        SELECT bs.card_id,
               SUM(COALESCE(bs.ops_plus * bs.pa, 0)) * 1.0 / NULLIF(SUM(bs.pa), 0) AS ops_plus,
               SUM(COALESCE(bs.obp * bs.pa, 0)) * 1.0 / NULLIF(SUM(bs.pa), 0) AS obp,
               SUM(COALESCE(bs.iso * bs.pa, 0)) * 1.0 / NULLIF(SUM(bs.pa), 0) AS iso,
               SUM(COALESCE(bs.babip * bs.pa, 0)) * 1.0 / NULLIF(SUM(bs.pa), 0) AS babip,
               SUM(bs.pa) AS pa,
               SUM(bs.war) AS war
        FROM batting_stats bs
        WHERE bs.card_id IS NOT NULL AND bs.pa > 0
        GROUP BY bs.card_id
        HAVING SUM(bs.pa) >= 150
    """).fetchall():
        obs_bat[r['card_id']] = dict(r)

    # League baselines for delta overlays
    league_obp = {}
    league_babip = {}
    league_tier_iso = {}
    for r in conn.execute("""
        SELECT league_id, AVG(obp) AS obp, AVG(babip) AS babip
        FROM batting_stats
        WHERE pa >= 150 AND obp IS NOT NULL
        GROUP BY league_id
    """).fetchall():
        if r['league_id']:
            league_obp[r['league_id']] = r['obp']
            league_babip[r['league_id']] = r['babip']
    for r in conn.execute("""
        SELECT bs.league_id, c.tier, AVG(bs.iso) AS iso
        FROM batting_stats bs
        INNER JOIN cards c ON c.card_id = bs.card_id
        WHERE bs.pa >= 150 AND bs.iso IS NOT NULL
        GROUP BY bs.league_id, c.tier
        HAVING COUNT(*) >= 20
    """).fetchall():
        league_tier_iso[(r['league_id'], r['tier'])] = r['iso']

    # wOBA for performance overlay
    adv = {}
    for r in conn.execute("""
        SELECT ba.card_id, ba.woba, ba.pa
        FROM batting_stats_adv ba
        WHERE ba.card_id IS NOT NULL
    """).fetchall():
        adv[r['card_id']] = dict(r)

    # Now build the per-card overlay contributions
    headers = ['ops_plus_o', 'obp_o', 'iso_o', 'babip_o',
               'woba_o', 'overperf_o', 'superstat_o', 'clutch_o']
    data: dict[str, list[float]] = {k: [] for k in headers}
    wars = []
    pas = []
    metas = []
    names = []

    # Need rating composite for overperf
    xs_comp = []
    ys_op = []
    for cid, ob in obs_bat.items():
        if cid not in cards:
            continue
        c = cards[cid]
        comp = (c.get('contact') or 0) + (c.get('power') or 0) + (c.get('gap_power') or 0)
        comp /= 3.0
        xs_comp.append(comp)
        ys_op.append(ob.get('ops_plus') or 0)
    if len(xs_comp) >= 30:
        mx = sum(xs_comp)/len(xs_comp); my = sum(ys_op)/len(ys_op)
        num = sum((x-mx)*(y-my) for x,y in zip(xs_comp, ys_op))
        den = sum((x-mx)**2 for x in xs_comp) or 1.0
        op_b = num / den; op_a = my - op_b * mx
    else:
        op_a, op_b = 100.0, 0.0

    for cid, ob in obs_bat.items():
        if cid not in cards:
            continue
        c = cards[cid]
        pa = ob.get('pa') or 0
        war = ob.get('war') or 0
        # Build input dict for each overlay
        d = dict(c)
        d['_obs_ops_plus'] = ob.get('ops_plus')
        d['_obs_ops_plus_pa'] = pa
        # OBP delta
        # NOTE: we pool across leagues for simplicity; ingestion uses
        # per-league baselines.
        d['_obs_obp_delta'] = (ob.get('obp') or 0) - 0.320
        d['_obs_obp_pa'] = pa
        # BABIP delta
        d['_obs_babip_delta'] = (ob.get('babip') or 0) - 0.290
        d['_obs_babip_pa'] = pa
        # ISO delta vs tier
        tier = c.get('tier')
        baseline = 0.146
        d['_obs_iso_delta'] = (ob.get('iso') or 0) - baseline
        d['_obs_iso_pa'] = pa
        # Overperf OPS+
        comp = (c.get('contact') or 0) + (c.get('power') or 0) + (c.get('gap_power') or 0)
        comp /= 3.0
        predicted = op_a + op_b * comp
        d['_over_ops_plus'] = (ob.get('ops_plus') or 0) - predicted
        d['_over_ops_plus_pa'] = pa
        # wOBA overlay
        a = adv.get(cid)
        if a:
            d['adv_woba'] = a.get('woba')
            d['adv_pa'] = a.get('pa')
            d['_league_avg_woba'] = 0.320

        data['ops_plus_o'].append(_calc_opsplus_overlay_batting(d))
        data['obp_o'].append(_calc_obp_overlay_batting(d))
        data['iso_o'].append(_calc_iso_overlay_batting(d))
        data['babip_o'].append(_calc_babip_overlay_batting(d))
        data['woba_o'].append(_calc_performance_adjustment_batting(d))
        data['overperf_o'].append(_calc_overperformance_overlay_batting(d))
        data['superstat_o'].append(_calc_superstat_overlay_batting(d))
        data['clutch_o'].append(_calc_clutch_overlay_batting(d))
        wars.append(war * 600.0 / pa if pa else 0)
        pas.append(pa)
        metas.append(c.get('meta_score_batting') or 0)
        names.append(c.get('card_title') or '?')

    return headers, data, wars, pas, metas, names


def print_heatmap(headers, data):
    print('\nCROSS-CORRELATION MATRIX (r between overlay contributions):')
    print(f'{"":12s}', ' '.join(f'{h:>10s}' for h in headers))
    for h1 in headers:
        row = [f'{h1:12s}']
        for h2 in headers:
            r = pearson(data[h1], data[h2])
            flag = ''
            if h1 != h2 and abs(r) > 0.80:
                flag = '!!!'
            elif h1 != h2 and abs(r) > 0.60:
                flag = '!!'
            elif h1 != h2 and abs(r) > 0.40:
                flag = '!'
            row.append(f'{r:+.2f}{flag:>3s}')
        print('  '.join(row))


def print_target_r(headers, data, target, label):
    print(f'\nUNIVARIATE r of each overlay with {label}:')
    rs = [(h, pearson(data[h], target)) for h in headers]
    rs.sort(key=lambda x: -abs(x[1]))
    for h, r in rs:
        stars = '***' if abs(r) > 0.2 else '**' if abs(r) > 0.1 else '*' if abs(r) > 0.05 else ''
        print(f'  {h:12s} r={r:+.4f} {stars}')


def print_partials(headers, data, target):
    print(f'\nPARTIAL r with WAR/600 (controlling for the best single overlay):')
    # Find the overlay with highest |r| to WAR
    rs = [(h, pearson(data[h], target)) for h in headers]
    rs.sort(key=lambda x: -abs(x[1]))
    best_h, best_r = rs[0]
    print(f'  Best overlay = {best_h} (r={best_r:+.3f})')
    print(f'  Others, partial r after controlling for {best_h}:')
    for h in headers:
        if h == best_h:
            continue
        pr = partial_r(data[h], target, data[best_h])
        base_r = pearson(data[h], target)
        shrink = base_r - pr if abs(base_r) > 0.01 else 0
        print(f'    {h:12s} partial={pr:+.3f}  (univariate {base_r:+.3f}, '
              f'shrink {shrink:+.3f})')


def print_sum_distribution(data):
    total = [sum(vals) for vals in zip(*data.values())]
    total.sort()
    n = len(total)
    def pct(p):
        i = int((n-1) * p)
        return total[i]
    print(f'\nSUM of overlays across cards (distribution):')
    print(f'  count: {n}')
    print(f'   min:  {pct(0):+.1f}')
    print(f'  p10:  {pct(0.10):+.1f}')
    print(f'  p25:  {pct(0.25):+.1f}')
    print(f'  p50:  {pct(0.50):+.1f}')
    print(f'  p75:  {pct(0.75):+.1f}')
    print(f'  p90:  {pct(0.90):+.1f}')
    print(f'   max: {pct(1.0):+.1f}')


def main():
    print('=' * 72)
    print('OVERLAY REDUNDANCY AUDIT (batting)')
    print('=' * 72)
    headers, data, wars, pas, metas, names = build_overlay_matrix()
    print(f'Sample: {len(wars)} cards with PA>=150')
    print_heatmap(headers, data)
    print_target_r(headers, data, wars, 'WAR/600')
    print_partials(headers, data, wars)
    print_sum_distribution(data)


if __name__ == '__main__':
    main()
