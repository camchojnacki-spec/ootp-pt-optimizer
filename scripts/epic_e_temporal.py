"""Epic E — Temporal dynamics study.

Q1: wOBA stabilization vs final-snapshot wOBA, by PA bucket.
Q2: Re-applies derived_stats regression-candidate scoring to every
    historical batting_stats row; checks next-snapshot OPS+ direction.
Q3: Per-card consecutive delta_meta vs delta_price, with 1-snap lag.

Usage: python scripts/epic_e_temporal.py
"""
from __future__ import annotations

import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DB = Path(__file__).resolve().parent.parent / "data" / "ootp_optimizer.db"

# Match derived_stats.build_regression_candidates_v2
BABIP_BASELINE = 0.300
K_PCT_BASELINE = 0.22
DOWN_THRESH = 0.30
UP_THRESH = -0.30


def pearson(xs, ys):
    if len(xs) < 3:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pa_bucket(pa):
    for lo, hi, name in [
        (0, 50, "0-49"), (50, 100, "50-99"), (100, 150, "100-149"),
        (150, 200, "150-199"), (200, 300, "200-299"),
        (300, 400, "300-399"), (400, 500, "400-499"),
        (500, 99999, "500+"),
    ]:
        if lo <= pa < hi:
            return name
    return None


# ──────────────────────────────────────────────────────────────────────
# Q1. wOBA stabilization
# ──────────────────────────────────────────────────────────────────────

def q1_woba_stabilization(conn):
    print("\n" + "=" * 72)
    print("Q1. wOBA Stabilization")
    print("=" * 72)

    rows = conn.execute("""
        SELECT bs.card_id, bs.league_id, bs.snapshot_date, bs.pa, ba.woba
        FROM batting_stats bs
        JOIN batting_stats_adv ba
          ON ba.card_id = bs.card_id
         AND ba.snapshot_date = bs.snapshot_date
         AND COALESCE(ba.league_id, '') = COALESCE(bs.league_id, '')
        WHERE bs.pa >= 1 AND ba.woba IS NOT NULL
        ORDER BY bs.card_id, bs.league_id, bs.snapshot_date
    """).fetchall()

    trajs = defaultdict(list)
    for r in rows:
        key = (r["card_id"], r["league_id"] or "")
        trajs[key].append((r["snapshot_date"], r["pa"] or 0,
                           float(r["woba"])))

    elig = {k: v for k, v in trajs.items() if len(v) >= 3}
    print(f"Cards with >=3 snapshots & wOBA: {len(elig)}")

    BAND = 0.020
    snap_to_final = defaultdict(list)
    pa_pairs_all = defaultdict(list)
    final_diffs = []

    for traj in elig.values():
        traj.sort(key=lambda t: t[0])
        woba_final = traj[-1][2]
        first_stab = None
        for idx, (_, pa, woba) in enumerate(traj):
            if idx == len(traj) - 1:
                break
            snap_to_final[idx].append((woba, woba_final))
            b = pa_bucket(pa)
            if b:
                pa_pairs_all[b].append((woba, woba_final))
            if first_stab is None and abs(woba - woba_final) <= BAND:
                if all(abs(t[2] - woba_final) <= BAND
                       for t in traj[idx:]):
                    first_stab = (idx, pa)
        if first_stab is not None:
            final_diffs.append((traj[-1][1], first_stab[0], first_stab[1]))

    print(f"Cards reaching ±{BAND:.3f} wOBA band of final: "
          f"{len(final_diffs)} / {len(elig)}")
    if final_diffs:
        print(f"  Mean snapshots to stabilize: "
              f"{statistics.mean(x[1] for x in final_diffs):.2f}")
        print(f"  Median PA at first stabilization: "
              f"{statistics.median(x[2] for x in final_diffs):.0f}")

    # PA bucket cross-card r — primary view
    print("\nr(wOBA at PA bucket X, wOBA final) — primary stabilization view")
    print(f"  {'bucket':<12}{'n':<8}{'r':<10}")
    for b in ["0-49", "50-99", "100-149", "150-199", "200-299",
              "300-399", "400-499", "500+"]:
        pairs = pa_pairs_all.get(b, [])
        if len(pairs) < 5:
            print(f"  {b:<12}{len(pairs):<8}--")
            continue
        r = pearson([p[0] for p in pairs], [p[1] for p in pairs])
        print(f"  {b:<12}{len(pairs):<8}{r:+.3f}")

    # Restrict to "full-season" cards (final PA >= 500)
    elig_fs = {k: v for k, v in trajs.items()
               if len(v) >= 3 and v[-1][1] >= 500}
    print(f"\nFull-season cohort (final PA >= 500): n={len(elig_fs)}")
    pa_pairs_fs = defaultdict(list)
    for traj in elig_fs.values():
        traj.sort(key=lambda t: t[0])
        woba_final = traj[-1][2]
        for (_, pa, woba) in traj[:-1]:
            b = pa_bucket(pa)
            if b:
                pa_pairs_fs[b].append((woba, woba_final))
    for b in ["0-49", "50-99", "100-149", "150-199", "200-299",
              "300-399", "400-499", "500+"]:
        pairs = pa_pairs_fs.get(b, [])
        if len(pairs) < 5:
            print(f"  {b:<12}{len(pairs):<8}--")
            continue
        r = pearson([p[0] for p in pairs], [p[1] for p in pairs])
        print(f"  {b:<12}{len(pairs):<8}{r:+.3f}")

    # Late-season volatility check
    elig_300 = {k: v for k, v in trajs.items()
                if len(v) >= 3 and v[-1][1] >= 300}
    last3_ranges = [max(t[2] for t in v[-3:]) - min(t[2] for t in v[-3:])
                    for v in elig_300.values()]
    if last3_ranges:
        print(f"\nFinal PA>=300 cohort, n={len(elig_300)}")
        print(f"  Median last-3-snap wOBA range: "
              f"{statistics.median(last3_ranges):.4f}")
        frac = sum(1 for r in last3_ranges if r <= 0.020) / len(last3_ranges)
        print(f"  Fraction with last-3 wOBA range <= 0.020: {frac:.3f}")


# ──────────────────────────────────────────────────────────────────────
# Q2. Regression-candidate convergence
# ──────────────────────────────────────────────────────────────────────

def _expected_ops_plus(card_value, curve):
    if card_value is None or card_value <= 0 or not curve:
        return None
    bucket = card_value // 5
    if bucket in curve:
        return curve[bucket]
    keys = sorted(curve.keys())
    return curve[min(keys, key=lambda k: abs(k - bucket))]


def _regression_score(ops_plus_obs, expected_ops, babip_obs, rating_babip,
                      k_pct_obs, rating_avoid_ks):
    components = []
    if ops_plus_obs is not None and expected_ops is not None:
        components.append((ops_plus_obs - expected_ops) / 30.0)
    if babip_obs is not None and rating_babip is not None:
        expected_babip = BABIP_BASELINE * (1 + 0.003 * (rating_babip - 50))
        if expected_babip:
            components.append(((babip_obs / expected_babip) - 1) * 3.0)
    if k_pct_obs is not None and rating_avoid_ks is not None:
        expected_k = K_PCT_BASELINE * (1 - 0.008 * (rating_avoid_ks - 50))
        if expected_k > 0:
            components.append(-(k_pct_obs - expected_k) * 3.0)
    return (sum(components) / len(components)) if components else None


def q2_regression_convergence(conn):
    print("\n" + "=" * 72)
    print("Q2. Regression Candidate Convergence")
    print("=" * 72)

    curve_rows = conn.execute("""
        WITH latest AS (
            SELECT bs.card_id, bs.league_id, bs.ops_plus, bs.pa
            FROM batting_stats bs
            WHERE bs.id IN (
                SELECT MAX(id) FROM batting_stats
                WHERE card_id IS NOT NULL
                GROUP BY card_id, COALESCE(league_id, '')
            ) AND bs.pa >= 25 AND bs.ops_plus IS NOT NULL
        )
        SELECT (c.card_value / 5) AS bucket, AVG(l.ops_plus) AS m,
               COUNT(*) AS n
        FROM cards c JOIN latest l ON l.card_id = c.card_id
        WHERE c.card_value > 0
          AND (c.position_name IS NULL
               OR c.position_name NOT IN ('SP','RP','CL','P'))
          AND (c.pitcher_role_name IS NULL OR c.pitcher_role_name = '')
        GROUP BY bucket HAVING n >= 3 ORDER BY bucket
    """).fetchall()
    curve = {r["bucket"]: r["m"] for r in curve_rows}
    print(f"Fitted card_value -> OPS+ buckets: {len(curve)}")

    rows = conn.execute("""
        SELECT bs.card_id, bs.league_id, bs.snapshot_date, bs.pa,
               bs.ops_plus, bs.babip,
               CAST(bs.k AS REAL) / NULLIF(bs.pa, 0) AS k_pct,
               c.card_value, c.babip AS rating_babip,
               c.avoid_ks AS rating_avoid_ks
        FROM batting_stats bs
        JOIN cards c ON c.card_id = bs.card_id
        WHERE bs.pa >= 50
          AND (c.position_name IS NULL
               OR c.position_name NOT IN ('SP','RP','CL','P'))
          AND (c.pitcher_role_name IS NULL OR c.pitcher_role_name = '')
        ORDER BY bs.card_id, bs.league_id, bs.snapshot_date
    """).fetchall()

    by_card = defaultdict(list)
    for r in rows:
        by_card[(r["card_id"], r["league_id"] or "")].append(r)

    flag_counts = {"down": 0, "up": 0, "sus": 0}
    hits = {"down": 0, "up": 0, "sus": 0}
    moves = {"down": [], "up": [], "sus": []}
    pairs_total = 0

    for traj in by_card.values():
        traj.sort(key=lambda r: r["snapshot_date"])
        for i in range(len(traj) - 1):
            cur, nxt = traj[i], traj[i + 1]
            if (cur["snapshot_date"] == nxt["snapshot_date"]
                    or cur["ops_plus"] is None or nxt["ops_plus"] is None):
                continue
            score = _regression_score(
                cur["ops_plus"],
                _expected_ops_plus(cur["card_value"], curve),
                cur["babip"], cur["rating_babip"],
                cur["k_pct"], cur["rating_avoid_ks"],
            )
            if score is None:
                continue
            d = nxt["ops_plus"] - cur["ops_plus"]
            pairs_total += 1
            if score > DOWN_THRESH:
                flag_counts["down"] += 1
                moves["down"].append(d)
                if d < 0:
                    hits["down"] += 1
            elif score < UP_THRESH:
                flag_counts["up"] += 1
                moves["up"].append(d)
                if d > 0:
                    hits["up"] += 1
            else:
                flag_counts["sus"] += 1
                moves["sus"].append(d)
                if abs(d) <= 10:
                    hits["sus"] += 1

    print(f"\nTotal consecutive-snapshot pairs analysed: {pairs_total}")

    def pct(n, d):
        return f"{100.0 * n / d:.1f}%" if d else "n/a"

    print(f"\nregress_down (score > +{DOWN_THRESH}):")
    print(f"  count: {flag_counts['down']}, "
          f"hits (next OPS+ down): {hits['down']} "
          f"({pct(hits['down'], flag_counts['down'])})")
    if moves["down"]:
        print(f"  mean dOPS+: {statistics.mean(moves['down']):+.2f}, "
              f"median dOPS+: {statistics.median(moves['down']):+.2f}")

    print(f"\nregress_up (score < {UP_THRESH}):")
    print(f"  count: {flag_counts['up']}, "
          f"hits (next OPS+ up): {hits['up']} "
          f"({pct(hits['up'], flag_counts['up'])})")
    if moves["up"]:
        print(f"  mean dOPS+: {statistics.mean(moves['up']):+.2f}, "
              f"median dOPS+: {statistics.median(moves['up']):+.2f}")

    print(f"\nsustainable (|score| <= {DOWN_THRESH}):")
    print(f"  count: {flag_counts['sus']}, "
          f"stable (|dOPS+|<=10): {hits['sus']} "
          f"({pct(hits['sus'], flag_counts['sus'])})")

    all_d = moves["down"] + moves["up"] + moves["sus"]
    if all_d:
        bd = sum(1 for d in all_d if d < 0) / len(all_d)
        bu = sum(1 for d in all_d if d > 0) / len(all_d)
        print(f"\nBaseline (any pair): P(down)={bd:.3f}, P(up)={bu:.3f}")


# ──────────────────────────────────────────────────────────────────────
# Q3. Meta drift vs price velocity
# ──────────────────────────────────────────────────────────────────────

def q3_meta_drift_vs_price(conn):
    print("\n" + "=" * 72)
    print("Q3. Meta-score Drift vs Price Velocity")
    print("=" * 72)

    rows = conn.execute("""
        SELECT card_id, snapshot_date, meta_score, last_10_price
        FROM player_history
        WHERE meta_score IS NOT NULL AND last_10_price IS NOT NULL
        ORDER BY card_id, snapshot_date
    """).fetchall()

    by_card = defaultdict(list)
    for r in rows:
        by_card[r["card_id"]].append(
            (r["snapshot_date"], float(r["meta_score"]),
             float(r["last_10_price"]))
        )

    same, meta_lead, price_lead = [], [], []
    cards_used = 0
    for traj in by_card.values():
        if len(traj) < 3:
            continue
        traj.sort(key=lambda t: t[0])
        # Dedupe near-identical timestamps (some snapshots came in pairs)
        dedup = [traj[0]]
        for t in traj[1:]:
            if t[0] != dedup[-1][0]:
                dedup.append(t)
        if len(dedup) < 3:
            continue
        cards_used += 1
        dM = [dedup[i + 1][1] - dedup[i][1] for i in range(len(dedup) - 1)]
        dP = [dedup[i + 1][2] - dedup[i][2] for i in range(len(dedup) - 1)]
        for m, p in zip(dM, dP):
            same.append((m, p))
        for i in range(len(dM) - 1):
            meta_lead.append((dM[i], dP[i + 1]))
            price_lead.append((dP[i], dM[i + 1]))

    print(f"Cards used (>=3 deduped snaps): {cards_used}")
    print(f"Total deltas: {len(same)}")

    def report(label, pairs):
        if len(pairs) < 5:
            print(f"  {label:<46} n={len(pairs)} (insufficient)")
            return None
        r = pearson([p[0] for p in pairs], [p[1] for p in pairs])
        print(f"  {label:<46} n={len(pairs):4d}  r={r:+.4f}")
        return r

    print("\nCoincident:")
    report("dMeta_t vs dPrice_t", same)
    print("\n1-snap lag tests:")
    report("dMeta_t -> dPrice_{t+1} (meta leads)", meta_lead)
    report("dPrice_t -> dMeta_{t+1} (price leads)", price_lead)

    print("\nFiltered |dMeta| > 5 (drop noise floor):")
    report("dMeta_t vs dPrice_t",
           [p for p in same if abs(p[0]) > 5])
    report("dMeta_t -> dPrice_{t+1}",
           [p for p in meta_lead if abs(p[0]) > 5])


def main():
    if not DB.exists():
        print(f"DB not found at {DB}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    try:
        q1_woba_stabilization(conn)
        q2_regression_convergence(conn)
        q3_meta_drift_vs_price(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
