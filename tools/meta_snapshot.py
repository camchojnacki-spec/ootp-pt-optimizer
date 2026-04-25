"""Capture a point-in-time snapshot of the meta engine state.

Writes a JSON blob to ``data/snapshots/meta_{YYYY-MM-DD-HHMMSS}.json`` with
calibration results, top-50 meta leaders per role, distribution stats, and
the current roster's meta + fit. Designed to be diffed across ingests — run
before a data dump, run after, compare to see how meta moved.

Usage:
    python tools/meta_snapshot.py                    # write snapshot
    python tools/meta_snapshot.py --label pre-dump   # custom filename tag
    python tools/meta_snapshot.py --diff pre-dump    # diff against prior label
"""
from __future__ import annotations
import argparse
import json
import sqlite3
import statistics
from datetime import datetime
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "ootp_optimizer.db"
SNAP_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"


def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pct(values, p):
    vs = sorted(v for v in values if v is not None)
    if not vs:
        return None
    idx = max(0, min(len(vs) - 1, int(round((p / 100.0) * (len(vs) - 1)))))
    return vs[idx]


def _distribution(values):
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return {
        "n": len(vs),
        "mean": round(statistics.mean(vs), 2),
        "median": round(statistics.median(vs), 2),
        "p25": round(_pct(vs, 25), 2),
        "p75": round(_pct(vs, 75), 2),
        "p95": round(_pct(vs, 95), 2),
        "min": round(min(vs), 2),
        "max": round(max(vs), 2),
    }


def capture_calibration(conn):
    """Most-recent calibration row per type."""
    rows = conn.execute("""
        SELECT calibration_type, r_squared, correlation, sample_size,
               confidence, created_at
        FROM meta_calibration mc
        WHERE created_at = (
            SELECT MAX(created_at) FROM meta_calibration
            WHERE calibration_type = mc.calibration_type
        )
        ORDER BY calibration_type
    """).fetchall()
    return [
        {
            "type": r["calibration_type"],
            "cv_r2": round(r["r_squared"], 4) if r["r_squared"] is not None else None,
            "pearson_r": round(r["correlation"], 4) if r["correlation"] is not None else None,
            "n": r["sample_size"],
            "confidence": round(r["confidence"], 3) if r["confidence"] is not None else None,
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def capture_top_meta(conn, role, limit=50):
    """Top-N cards by meta score for a given role.

    role in {'batting', 'SP', 'RP'}. Returns a list with rank, card_id,
    title, meta, ovr, tier, price, fit, archetype.
    """
    if role == "batting":
        meta_col = "meta_score_batting"
        where = "c.pitcher_role IS NULL"
        arch_role_filter = "ca.role = 'batting'"
    elif role == "SP":
        meta_col = "meta_score_pitching"
        where = "c.pitcher_role_name = 'SP'"
        arch_role_filter = "ca.role = 'SP'"
    elif role == "RP":
        meta_col = "meta_score_pitching"
        where = "c.pitcher_role_name IN ('RP', 'CL')"
        arch_role_filter = "ca.role = 'RP'"
    else:
        raise ValueError(f"unknown role: {role}")

    rows = conn.execute(f"""
        SELECT c.card_id, c.card_title, c.position_name, c.pitcher_role_name,
               c.tier_name, c.card_type, c.card_value AS ovr,
               c.{meta_col} AS meta, c.last_10_price, c.owned,
               ca.fit_score, ca.archetype_name, ca.archetype_war
        FROM cards c
        LEFT JOIN card_archetypes ca ON ca.card_id = c.card_id AND {arch_role_filter}
        WHERE {where} AND c.{meta_col} IS NOT NULL
        ORDER BY c.{meta_col} DESC
        LIMIT ?
    """, (limit,)).fetchall()

    return [
        {
            "rank": i + 1,
            "card_id": r["card_id"],
            "title": r["card_title"],
            "position": r["position_name"] or r["pitcher_role_name"],
            "tier": r["tier_name"],
            "card_type": r["card_type"],
            "ovr": r["ovr"],
            "meta": round(r["meta"], 1) if r["meta"] is not None else None,
            "price": r["last_10_price"],
            "owned": bool(r["owned"]),
            "fit": round(r["fit_score"], 1) if r["fit_score"] is not None else None,
            "archetype": r["archetype_name"],
            "arch_war": round(r["archetype_war"], 3)
            if r["archetype_war"] is not None else None,
        }
        for i, r in enumerate(rows)
    ]


def capture_distributions(conn):
    """Meta distribution per role, broken down by card_type."""
    out = {}
    for role, meta_col, where in [
        ("batting", "meta_score_batting", "pitcher_role IS NULL"),
        ("pitching_sp", "meta_score_pitching", "pitcher_role_name = 'SP'"),
        ("pitching_rp", "meta_score_pitching", "pitcher_role_name IN ('RP', 'CL')"),
    ]:
        rows = conn.execute(f"""
            SELECT card_type, {meta_col} AS meta
            FROM cards
            WHERE {where} AND {meta_col} IS NOT NULL
        """).fetchall()
        by_type = {}
        all_meta = []
        for r in rows:
            meta = _num(r["meta"])
            if meta is None:
                continue
            all_meta.append(meta)
            by_type.setdefault(r["card_type"] or "Unknown", []).append(meta)
        out[role] = {
            "overall": _distribution(all_meta),
            "by_card_type": {k: _distribution(v) for k, v in sorted(by_type.items())},
        }
    return out


def capture_roster(conn):
    """Current active roster state (meta + fit for each lineup slot)."""
    rows = conn.execute("""
        SELECT r.lineup_role, r.card_id, r.position AS slot_position,
               c.card_title, c.position_name, c.pitcher_role_name,
               c.card_value AS ovr, c.meta_score_batting, c.meta_score_pitching,
               c.last_10_price,
               ca_bat.fit_score AS fit_bat, ca_bat.archetype_name AS arch_bat,
               ca_bat.archetype_war AS arch_war_bat,
               ca_pit.fit_score AS fit_pit, ca_pit.archetype_name AS arch_pit,
               ca_pit.archetype_war AS arch_war_pit
        FROM roster r
        JOIN cards c ON c.card_id = r.card_id
        LEFT JOIN card_archetypes ca_bat
            ON ca_bat.card_id = c.card_id AND ca_bat.role = 'batting'
        LEFT JOIN card_archetypes ca_pit
            ON ca_pit.card_id = c.card_id AND ca_pit.role IN ('SP', 'RP')
        WHERE r.lineup_role = 'starter'
          AND r.snapshot_date = (SELECT MAX(snapshot_date) FROM roster)
        ORDER BY r.position, c.card_title
    """).fetchall()

    out = []
    for r in rows:
        is_pit = bool(r["pitcher_role_name"])
        meta = r["meta_score_pitching"] if is_pit else r["meta_score_batting"]
        fit = r["fit_pit"] if is_pit else r["fit_bat"]
        arch = r["arch_pit"] if is_pit else r["arch_bat"]
        arch_war = r["arch_war_pit"] if is_pit else r["arch_war_bat"]
        out.append({
            "slot": r["slot_position"],
            "card_id": r["card_id"],
            "title": r["card_title"],
            "position": r["position_name"] or r["pitcher_role_name"],
            "ovr": r["ovr"],
            "meta": round(meta, 1) if meta is not None else None,
            "price": r["last_10_price"],
            "fit": round(fit, 1) if fit is not None else None,
            "archetype": arch,
            "arch_war": round(arch_war, 3) if arch_war is not None else None,
        })
    return out


def capture_metadata(conn):
    """DB-level metadata: row counts, last snapshot, active league."""
    counts = {}
    for table in ("cards", "batting_stats", "pitching_stats", "game_log_at_bats",
                  "games", "player_history", "card_archetypes", "roster",
                  "game_clutch_events"):
        try:
            counts[table] = conn.execute(
                f"SELECT COUNT(*) AS n FROM {table}"
            ).fetchone()["n"]
        except sqlite3.OperationalError:
            counts[table] = None

    last_hist = conn.execute(
        "SELECT MAX(snapshot_date) AS d FROM player_history"
    ).fetchone()

    return {
        "counts": counts,
        "last_player_history_snapshot": last_hist["d"] if last_hist else None,
    }


def build_snapshot(db_path=DB):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return {
            "snapshot_at": datetime.now().isoformat(timespec="seconds"),
            "db_path": str(db_path),
            "metadata": capture_metadata(conn),
            "calibration": capture_calibration(conn),
            "top_meta": {
                "batting": capture_top_meta(conn, "batting"),
                "sp": capture_top_meta(conn, "SP"),
                "rp": capture_top_meta(conn, "RP"),
            },
            "distributions": capture_distributions(conn),
            "roster_starters": capture_roster(conn),
        }
    finally:
        conn.close()


def write_snapshot(snapshot, label=None):
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    tag = f"-{label}" if label else ""
    path = SNAP_DIR / f"meta_{stamp}{tag}.json"
    path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    return path


def _find_snapshot(label):
    matches = sorted(SNAP_DIR.glob(f"meta_*-{label}.json"))
    if not matches:
        raise FileNotFoundError(f"no snapshot with label '{label}'")
    return matches[-1]


def diff_snapshots(before_label, after_label="latest"):
    before = json.loads(_find_snapshot(before_label).read_text(encoding="utf-8"))
    if after_label == "latest":
        latest = sorted(SNAP_DIR.glob("meta_*.json"))
        if not latest:
            raise FileNotFoundError("no snapshots available for 'latest'")
        after = json.loads(latest[-1].read_text(encoding="utf-8"))
    else:
        after = json.loads(_find_snapshot(after_label).read_text(encoding="utf-8"))

    lines = [
        f"=== Snapshot diff: {before['snapshot_at']} -> {after['snapshot_at']} ==="
    ]

    def _fmt_delta(a, b):
        if a is None or b is None:
            return "?"
        return f"{b - a:+.4f}"

    lines.append("\n[Calibration]")
    before_cal = {c["type"]: c for c in before["calibration"]}
    after_cal = {c["type"]: c for c in after["calibration"]}
    for t in sorted(set(before_cal) | set(after_cal)):
        a = before_cal.get(t, {})
        b = after_cal.get(t, {})
        lines.append(
            f"  {t}: r²={a.get('cv_r2')}->{b.get('cv_r2')} "
            f"({_fmt_delta(a.get('cv_r2'), b.get('cv_r2'))})  "
            f"r={a.get('pearson_r')}->{b.get('pearson_r')} "
            f"({_fmt_delta(a.get('pearson_r'), b.get('pearson_r'))})  "
            f"n={a.get('n')}->{b.get('n')}"
        )

    lines.append("\n[Top-meta rank shifts]")
    for role in ("batting", "sp", "rp"):
        a_ranks = {c["card_id"]: c["rank"] for c in before["top_meta"][role]}
        b_ranks = {c["card_id"]: c["rank"] for c in after["top_meta"][role]}
        b_cards = {c["card_id"]: c for c in after["top_meta"][role]}
        shifts = []
        for cid, br in b_ranks.items():
            ar = a_ranks.get(cid)
            if ar is None:
                shifts.append((99, cid, f"NEW @ {br}: {b_cards[cid]['title']}"))
            elif ar != br:
                delta = ar - br
                if abs(delta) >= 3:
                    shifts.append(
                        (abs(delta), cid,
                         f"{b_cards[cid]['title']}: #{ar} -> #{br} ({delta:+d})")
                    )
        shifts.sort(reverse=True)
        lines.append(f"  {role.upper()}: {len(shifts)} notable shifts")
        for _, _, line in shifts[:10]:
            lines.append(f"    {line}")

    lines.append("\n[Roster starters — meta deltas]")
    a_roster = {r["card_id"]: r for r in before["roster_starters"]}
    b_roster = {r["card_id"]: r for r in after["roster_starters"]}
    for cid, br in b_roster.items():
        ar = a_roster.get(cid)
        if ar and ar.get("meta") is not None and br.get("meta") is not None:
            delta = br["meta"] - ar["meta"]
            if abs(delta) >= 2:
                lines.append(
                    f"  {br['slot']:<4} {br['title'][:30]:<30} "
                    f"meta {ar['meta']:.1f} -> {br['meta']:.1f} ({delta:+.1f})"
                )

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", help="Filename tag (e.g. 'pre-dump')")
    ap.add_argument("--diff", help="Label to diff against the latest snapshot")
    args = ap.parse_args()

    if args.diff:
        print(diff_snapshots(args.diff))
        return

    snap = build_snapshot()
    path = write_snapshot(snap, args.label)
    cal = {c["type"]: c for c in snap["calibration"]}
    print(f"Snapshot written: {path}")
    print(f"  Cards: {snap['metadata']['counts'].get('cards')}, "
          f"last history: {snap['metadata']['last_player_history_snapshot']}")
    for t in ("batting", "pitching_sp", "pitching_rp", "pitching"):
        c = cal.get(t)
        if c:
            print(f"  {t}: CV R²={c['cv_r2']}  r={c['pearson_r']}  n={c['n']}")
    print(f"  Top batting: {snap['top_meta']['batting'][0]['title']} "
          f"(meta {snap['top_meta']['batting'][0]['meta']})")


if __name__ == "__main__":
    main()
