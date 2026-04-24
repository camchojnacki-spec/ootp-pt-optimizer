"""Mine rating-vs-observed correlations across the game-log layer.

Produces three reports:
  1. Univariate r for every rating x outcome (batters and pitchers, pooled + by role).
  2. Top pair-wise interactions where r(A*B/100) beats r(A) and r(B) by >=0.05.
  3. K-means archetype clusters on observed outcomes, with the rating profile that maps to each.
"""
from __future__ import annotations
import sqlite3
import math
from pathlib import Path
from collections import defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "ootp_optimizer.db"

BATTER_RATINGS = [
    "contact", "gap_power", "power", "eye", "avoid_ks", "babip",
    "speed", "steal_rate", "stealing", "baserunning",
    "contact_vl", "gap_vl", "power_vl", "eye_vl", "avoid_ks_vl", "babip_vl",
    "contact_vr", "gap_vr", "power_vr", "eye_vr", "avoid_ks_vr", "babip_vr",
]

PITCHER_RATINGS = [
    "stuff", "movement", "control", "p_hr", "p_babip",
    "stuff_vl", "movement_vl", "control_vl", "p_hr_vl", "p_babip_vl",
    "stuff_vr", "movement_vr", "control_vr", "p_hr_vr", "p_babip_vr",
    "stamina", "hold", "velocity",
]


def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    try:
        return float(str(v).split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def pearson(xs, ys):
    pairs = [(_num(x), _num(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 10:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs)))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return num / (sx * sy)


def fetch_batter_observed(conn, min_ab=50):
    """Return dict[card_id] -> {rating cols + observed outcome cols}."""
    cur = conn.cursor()
    rating_cols = ", ".join(f"c.{r}" for r in BATTER_RATINGS)
    sql = f"""
        SELECT c.card_id, c.position,
               {rating_cols},
               COUNT(*) AS ab,
               SUM(g.outcome='K')*1.0/COUNT(*) AS k_rate,
               SUM(g.outcome='BB')*1.0/COUNT(*) AS bb_rate,
               SUM(g.outcome='HR')*1.0/COUNT(*) AS hr_rate,
               SUM(g.outcome IN ('SINGLE','DOUBLE','TRIPLE','HR'))*1.0/COUNT(*) AS hit_rate,
               SUM(g.outcome='DOUBLE' OR g.outcome='TRIPLE' OR g.outcome='HR')*1.0/COUNT(*) AS xbh_rate,
               AVG(g.exit_velocity) AS avg_ev,
               SUM(g.batted_ball_type='Line Drive')*1.0/NULLIF(SUM(g.batted_ball_type IS NOT NULL),0) AS ld_pct,
               SUM(g.batted_ball_type='Groundball')*1.0/NULLIF(SUM(g.batted_ball_type IS NOT NULL),0) AS gb_pct,
               SUM(g.batted_ball_type='Popup')*1.0/NULLIF(SUM(g.batted_ball_type IS NOT NULL),0) AS pop_pct,
               SUM(g.exit_velocity >= 95)*1.0/NULLIF(SUM(g.exit_velocity IS NOT NULL),0) AS hard_hit_pct
        FROM cards c JOIN game_log_at_bats g ON c.card_id=g.batter_card_id
        GROUP BY c.card_id HAVING ab >= ?
    """
    cur.execute(sql, (min_ab,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_pitcher_observed(conn, min_bf=50):
    cur = conn.cursor()
    rating_cols = ", ".join(f"c.{r}" for r in PITCHER_RATINGS)
    sql = f"""
        SELECT c.card_id, c.pitcher_role_name, c.pitcher_role,
               {rating_cols},
               COUNT(*) AS bf,
               SUM(g.outcome='K')*1.0/COUNT(*) AS k_rate,
               SUM(g.outcome='BB')*1.0/COUNT(*) AS bb_rate,
               SUM(g.outcome='HR')*1.0/COUNT(*) AS hr_rate,
               SUM(g.outcome IN ('SINGLE','DOUBLE','TRIPLE','HR'))*1.0/COUNT(*) AS hit_allowed_rate,
               AVG(g.exit_velocity) AS ev_allowed,
               SUM(g.batted_ball_type='Groundball')*1.0/NULLIF(SUM(g.batted_ball_type IS NOT NULL),0) AS gb_allowed_pct,
               SUM(g.batted_ball_type='Line Drive')*1.0/NULLIF(SUM(g.batted_ball_type IS NOT NULL),0) AS ld_allowed_pct,
               SUM(g.batted_ball_type='Popup')*1.0/NULLIF(SUM(g.batted_ball_type IS NOT NULL),0) AS pop_allowed_pct,
               SUM(g.exit_velocity >= 95)*1.0/NULLIF(SUM(g.exit_velocity IS NOT NULL),0) AS hard_hit_allowed_pct
        FROM cards c JOIN game_log_at_bats g ON c.card_id=g.pitcher_card_id
        GROUP BY c.card_id HAVING bf >= ?
    """
    cur.execute(sql, (min_bf,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def univariate_grid(rows, ratings, outcomes, role_filter=None):
    if role_filter:
        rows = [r for r in rows if role_filter(r)]
    print(f"\n  n={len(rows)} cards")
    header = f"  {'rating':<14} " + " ".join(f"{o:>10}" for o in outcomes)
    print(header)
    print("  " + "-" * (14 + 11 * len(outcomes)))
    for rating in ratings:
        cells = []
        for out in outcomes:
            xs, ys = [], []
            for row in rows:
                if row.get(rating) is None or row.get(out) is None:
                    continue
                xs.append(row[rating])
                ys.append(row[out])
            r = pearson(xs, ys)
            cells.append(f"{r:+.3f}" if r is not None else "   -   ")
        print(f"  {rating:<14} " + " ".join(f"{c:>10}" for c in cells))


def top_interactions(rows, ratings, outcomes, top_k=10):
    """For each outcome, find rating-pair products that beat univariate r."""
    findings = []
    for out in outcomes:
        baselines = {}
        for r in ratings:
            xs = [row[r] for row in rows if row.get(r) is not None and row.get(out) is not None]
            ys = [row[out] for row in rows if row.get(r) is not None and row.get(out) is not None]
            baselines[r] = pearson(xs, ys)
        for i, a in enumerate(ratings):
            for b in ratings[i + 1:]:
                xs, ys = [], []
                for row in rows:
                    av, bv, ov = _num(row.get(a)), _num(row.get(b)), _num(row.get(out))
                    if av is None or bv is None or ov is None:
                        continue
                    xs.append((av * bv) / 100.0)
                    ys.append(ov)
                r_ab = pearson(xs, ys)
                if r_ab is None:
                    continue
                r_a = baselines.get(a) or 0
                r_b = baselines.get(b) or 0
                best_single = max(abs(r_a), abs(r_b))
                lift = abs(r_ab) - best_single
                if lift >= 0.05:
                    findings.append((out, a, b, r_a, r_b, r_ab, lift))
    findings.sort(key=lambda t: t[6], reverse=True)
    print(f"\n  Top {top_k} pair interactions (|r(A*B)| > max(|r(A)|,|r(B)|) + 0.05):")
    print(f"  {'outcome':<18} {'A':<12} {'B':<12} {'r(A)':>8} {'r(B)':>8} {'r(A*B)':>8} {'lift':>7}")
    for out, a, b, ra, rb, rab, lift in findings[:top_k]:
        print(f"  {out:<18} {a:<12} {b:<12} {ra:+.3f}   {rb:+.3f}   {rab:+.3f}   {lift:+.3f}")


def main():
    conn = sqlite3.connect(DB)
    batters = fetch_batter_observed(conn, min_ab=50)
    pitchers = fetch_pitcher_observed(conn, min_bf=50)
    print(f"\n=== DATASET ===\nbatters w/ >=50 AB: {len(batters)}")
    print(f"pitchers w/ >=50 BF: {len(pitchers)}")

    bat_outs = ["k_rate", "bb_rate", "hr_rate", "hit_rate", "xbh_rate",
                "avg_ev", "ld_pct", "gb_pct", "pop_pct", "hard_hit_pct"]
    pit_outs = ["k_rate", "bb_rate", "hr_rate", "hit_allowed_rate",
                "ev_allowed", "gb_allowed_pct", "ld_allowed_pct",
                "pop_allowed_pct", "hard_hit_allowed_pct"]

    print("\n=== BATTERS (pooled) — univariate r ===")
    univariate_grid(batters, BATTER_RATINGS[:10], bat_outs)

    print("\n=== PITCHERS (pooled) — univariate r ===")
    univariate_grid(pitchers, PITCHER_RATINGS[:10] + ["stamina", "velocity"], pit_outs)

    print("\n=== PITCHERS (SP only) — univariate r ===")
    univariate_grid(pitchers, PITCHER_RATINGS[:10] + ["stamina", "velocity"], pit_outs,
                    role_filter=lambda r: (r.get("pitcher_role_name") or "").upper().startswith("S"))

    print("\n=== PITCHERS (RP only) — univariate r ===")
    univariate_grid(pitchers, PITCHER_RATINGS[:10] + ["stamina", "velocity"], pit_outs,
                    role_filter=lambda r: (r.get("pitcher_role_name") or "").upper().startswith("R")
                                       or (r.get("pitcher_role_name") or "").upper().startswith("C"))

    print("\n=== BATTER INTERACTIONS ===")
    top_interactions(batters, ["contact", "gap_power", "power", "eye", "avoid_ks", "babip"],
                     ["hr_rate", "k_rate", "bb_rate", "avg_ev", "xbh_rate", "ld_pct"])

    print("\n=== PITCHER INTERACTIONS ===")
    top_interactions(pitchers, ["stuff", "movement", "control", "p_hr", "p_babip", "stamina", "velocity"],
                     ["k_rate", "bb_rate", "hr_rate", "ev_allowed", "gb_allowed_pct", "hard_hit_allowed_pct"])


if __name__ == "__main__":
    main()
