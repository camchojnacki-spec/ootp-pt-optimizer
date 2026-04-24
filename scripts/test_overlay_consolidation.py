"""Test a consolidated overlay config: drop redundant wOBA + ISO, bump OPS+,
shrink BABIP + overperf magnitudes. Measure cross-league r vs WAR.

Does NOT persist anything — purely a what-if simulator. If the consolidated
config matches or beats current, we apply it to meta_scoring.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.database import get_connection


def pearson(xs, ys):
    n = len(xs)
    if n < 2: return 0.0
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx*dy > 0 else 0.0


def test():
    """Compare CURRENT vs CONSOLIDATED overlay configs.

    CURRENT: wOBA + OPS+ + OBP + ISO + BABIP + overperf all firing
    CONSOLIDATED: OPS+(x1.0 bumped) + BABIP(half) + overperf(half), rest zeroed
    """
    conn = get_connection()

    # Per-league validation against WAR/600
    for league in ('lb124', 'i76'):
        rows = conn.execute("""
            SELECT c.meta_score_batting AS meta, c.card_value AS ovr,
                   bs.war, bs.pa
            FROM cards c INNER JOIN batting_stats bs ON bs.card_id = c.card_id
            INNER JOIN (
                SELECT card_id, MAX(snapshot_date) mx FROM batting_stats
                WHERE card_id IS NOT NULL AND league_id = ?
                GROUP BY card_id
            ) l ON bs.card_id = l.card_id AND bs.snapshot_date = l.mx
            WHERE bs.pa >= 150 AND bs.war IS NOT NULL AND c.card_value > 0
              AND bs.league_id = ?
        """, (league, league)).fetchall()
        tgt = [r['war'] * 600 / r['pa'] for r in rows]
        m_r = pearson([r['meta'] or 0 for r in rows], tgt)
        o_r = pearson([r['ovr'] for r in rows], tgt)
        print(f'{league:6s} CURRENT BAT n={len(rows)} meta r={m_r:+.4f} '
              f'ovr r={o_r:+.4f} delta={m_r - o_r:+.4f}')

    # Reconstruct per-card CONSOLIDATED meta by subtracting redundant
    # overlays and re-adding with new magnitudes
    print('\n=== Consolidated overlay simulation ===')
    # For each card, compute: consolidated = current_meta - (woba_o + iso_o)
    #                                      + (delta adjustment)
    from app.core.meta_scoring import (
        _calc_performance_adjustment_batting,
        _calc_iso_overlay_batting,
        _calc_opsplus_overlay_batting,
        _calc_obp_overlay_batting,
        _calc_babip_overlay_batting,
        _calc_overperformance_overlay_batting,
    )

    # Pull everything we need to re-fire overlays
    cards = {r['card_id']: dict(r) for r in conn.execute("""
        SELECT c.card_id, c.meta_score_batting AS meta,
               c.contact, c.power, c.gap_power, c.tier
        FROM cards c
        WHERE c.meta_score_batting IS NOT NULL AND c.pitcher_role IS NULL
    """).fetchall()}

    obs = {}
    for r in conn.execute("""
        SELECT card_id,
               SUM(COALESCE(ops_plus*pa,0))*1.0/NULLIF(SUM(pa),0) AS ops_plus,
               SUM(COALESCE(obp*pa,0))*1.0/NULLIF(SUM(pa),0) AS obp,
               SUM(COALESCE(iso*pa,0))*1.0/NULLIF(SUM(pa),0) AS iso,
               SUM(COALESCE(babip*pa,0))*1.0/NULLIF(SUM(pa),0) AS babip,
               SUM(pa) AS pa
        FROM batting_stats
        WHERE card_id IS NOT NULL AND pa > 0
        GROUP BY card_id
        HAVING SUM(pa) >= 50
    """).fetchall():
        obs[r['card_id']] = dict(r)

    adv = {}
    for r in conn.execute("""
        SELECT card_id, woba, pa FROM batting_stats_adv WHERE card_id IS NOT NULL
    """).fetchall():
        adv[r['card_id']] = dict(r)

    # OPS+ overperf regression (same as ingestion)
    train = []
    for cid, ob in obs.items():
        c = cards.get(cid)
        if not c: continue
        comp = ((c.get('contact') or 0) + (c.get('power') or 0)
                + (c.get('gap_power') or 0)) / 3.0
        train.append((comp, ob.get('ops_plus') or 0))
    if len(train) >= 30:
        mx = sum(t[0] for t in train)/len(train)
        my = sum(t[1] for t in train)/len(train)
        num = sum((t[0]-mx)*(t[1]-my) for t in train)
        den = sum((t[0]-mx)**2 for t in train) or 1.0
        op_b = num/den; op_a = my - op_b*mx
    else:
        op_a, op_b = 100.0, 0.0

    def _consolidated_delta(cid):
        """Recompute what the overlay sum would be under consolidated config."""
        c = cards.get(cid)
        ob = obs.get(cid)
        if not c or not ob:
            return 0.0, 0.0
        d = dict(c)
        d['_obs_ops_plus'] = ob.get('ops_plus')
        d['_obs_ops_plus_pa'] = ob.get('pa')
        d['_obs_obp_delta'] = (ob.get('obp') or 0) - 0.320
        d['_obs_obp_pa'] = ob.get('pa')
        d['_obs_iso_delta'] = (ob.get('iso') or 0) - 0.146
        d['_obs_iso_pa'] = ob.get('pa')
        d['_obs_babip_delta'] = (ob.get('babip') or 0) - 0.290
        d['_obs_babip_pa'] = ob.get('pa')
        comp = ((c.get('contact') or 0) + (c.get('power') or 0)
                + (c.get('gap_power') or 0)) / 3.0
        predicted = op_a + op_b * comp
        d['_over_ops_plus'] = (ob.get('ops_plus') or 0) - predicted
        d['_over_ops_plus_pa'] = ob.get('pa')
        a = adv.get(cid)
        if a:
            d['adv_woba'] = a.get('woba')
            d['adv_pa'] = a.get('pa')
            d['_league_avg_woba'] = 0.320
        # CURRENT overlays
        curr = (
            _calc_performance_adjustment_batting(d)
            + _calc_opsplus_overlay_batting(d)
            + _calc_obp_overlay_batting(d)
            + _calc_iso_overlay_batting(d)
            + _calc_babip_overlay_batting(d)
            + _calc_overperformance_overlay_batting(d)
        )
        # CONSOLIDATED: bump OPS+ by 1.5x, keep BABIP+overperf at 0.5x,
        # drop wOBA + OBP + ISO
        ops_o = _calc_opsplus_overlay_batting(d) * 1.5
        bab_o = _calc_babip_overlay_batting(d) * 0.5
        ovp_o = _calc_overperformance_overlay_batting(d) * 0.5
        cons = ops_o + bab_o + ovp_o
        return curr, cons

    # Simulate consolidated meta and measure r vs WAR for each league
    print()
    for league in ('lb124', 'i76'):
        rows = conn.execute("""
            SELECT c.card_id, c.meta_score_batting AS meta,
                   bs.war, bs.pa
            FROM cards c INNER JOIN batting_stats bs ON bs.card_id = c.card_id
            INNER JOIN (
                SELECT card_id, MAX(snapshot_date) mx FROM batting_stats
                WHERE card_id IS NOT NULL AND league_id = ?
                GROUP BY card_id
            ) l ON bs.card_id = l.card_id AND bs.snapshot_date = l.mx
            WHERE bs.pa >= 150 AND bs.war IS NOT NULL AND c.card_value > 0
              AND bs.league_id = ?
        """, (league, league)).fetchall()

        tgt = []
        curr_metas = []
        cons_metas = []
        for r in rows:
            pa = r['pa'] or 1
            tgt.append(r['war'] * 600 / pa)
            curr, cons = _consolidated_delta(r['card_id'])
            curr_metas.append(r['meta'] or 0)
            # Apply the delta: replace current overlay sum with consolidated
            cons_metas.append((r['meta'] or 0) - curr + cons)

        r_curr = pearson(curr_metas, tgt)
        r_cons = pearson(cons_metas, tgt)
        print(f'{league:6s} n={len(rows)}  '
              f'CURRENT r={r_curr:+.4f}  CONSOLIDATED r={r_cons:+.4f}  '
              f'Δ={r_cons - r_curr:+.4f}')


if __name__ == '__main__':
    test()
