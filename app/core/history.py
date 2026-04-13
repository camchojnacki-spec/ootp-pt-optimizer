"""Meta history tracking -- snapshot and trend meta scores across leagues."""
import hashlib
import json
import math
from datetime import datetime, timedelta

from app.core.database import get_connection
from app.core.meta_scoring import (
    calc_batting_meta,
    calc_pitching_meta,
    calc_batting_meta_vs_rhp,
    calc_batting_meta_vs_lhp,
    calc_pitching_meta_vs_lhb,
    calc_pitching_meta_vs_rhb,
    get_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val, default=0.0) -> float:
    """Safely convert a value to float."""
    try:
        v = float(val)
        return v if not math.isnan(v) else default
    except (ValueError, TypeError):
        return default


def _weights_hash(batting_w: dict, pitching_w: dict) -> str:
    """Return a short hash representing the current weight configuration."""
    blob = json.dumps({"b": batting_w, "p": pitching_w}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _is_pitcher(row) -> bool:
    """Return True if the card row represents a pitcher."""
    pos = row["position"] if "position" in row.keys() else row.get("position_name", "")
    if isinstance(pos, int):
        return pos == 1
    if isinstance(pos, str):
        return pos.upper() in ("P", "SP", "RP", "CL", "MR")
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_league_exists(league_id: str, league_name: str = None,
                         league_tier: str = None, conn=None):
    """Insert a league record if it doesn't exist."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        existing = conn.execute(
            "SELECT 1 FROM leagues WHERE league_id = ?", (league_id,)
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO leagues (league_id, league_name, league_tier) "
                "VALUES (?, ?, ?)",
                (league_id, league_name or league_id, league_tier),
            )
            conn.commit()
    finally:
        if close_conn:
            conn.close()


def snapshot_meta_scores(league_id: str = None, weights_version: str = None):
    """Calculate and store meta scores for all players in collection + roster.

    Reads current cards with owned=1 or appearing in roster_current,
    calculates meta with current weights, and inserts rows into meta_history.

    Returns dict with count of snapshots taken.
    """
    conn = get_connection()
    try:
        batting_w, pitching_w = get_weights()
        if weights_version is None:
            weights_version = _weights_hash(batting_w, pitching_w)

        # Collect card_ids from owned cards and current roster
        owned_rows = conn.execute(
            "SELECT * FROM cards WHERE owned = 1"
        ).fetchall()

        roster_ids = conn.execute(
            "SELECT DISTINCT card_id FROM roster WHERE card_id IS NOT NULL "
            "AND DATE(snapshot_date) = (SELECT MAX(DATE(snapshot_date)) FROM roster)"
        ).fetchall()
        roster_id_set = {r["card_id"] for r in roster_ids}

        # Build combined set -- use owned cards as base, add any roster cards
        # that aren't already owned (edge case: card in roster but owned=0)
        card_map = {r["card_id"]: r for r in owned_rows}
        if roster_id_set - card_map.keys():
            extra_ids = roster_id_set - card_map.keys()
            placeholders = ",".join("?" * len(extra_ids))
            extra_rows = conn.execute(
                f"SELECT * FROM cards WHERE card_id IN ({placeholders})",
                tuple(extra_ids),
            ).fetchall()
            for r in extra_rows:
                card_map[r["card_id"]] = r

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        inserted = 0

        for card_id, row in card_map.items():
            row_dict = dict(row)
            pos_name = row_dict.get("position_name", "")
            pos_num = row_dict.get("position", 0)
            pitcher = (pos_num == 1) or (
                isinstance(pos_name, str) and pos_name.upper() in ("P", "SP", "RP", "CL", "MR")
            )

            if pitcher:
                meta = calc_pitching_meta(row_dict, pitching_w)
                meta_vs_rhp = calc_pitching_meta_vs_rhb(row_dict, pitching_w)
                meta_vs_lhp = calc_pitching_meta_vs_lhb(row_dict, pitching_w)
            else:
                meta = calc_batting_meta(row_dict, batting_w)
                meta_vs_rhp = calc_batting_meta_vs_rhp(row_dict, batting_w)
                meta_vs_lhp = calc_batting_meta_vs_lhp(row_dict, batting_w)

            player_name = (
                row_dict.get("card_title")
                or f"{row_dict.get('first_name', '')} {row_dict.get('last_name', '')}".strip()
            )
            position = pos_name or str(pos_num)

            conn.execute(
                "INSERT INTO meta_history "
                "(card_id, player_name, position, meta_score, "
                " meta_vs_rhp, meta_vs_lhp, league_id, weights_version, snapshot_date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (card_id, player_name, position, meta,
                 meta_vs_rhp, meta_vs_lhp, league_id, weights_version, now),
            )
            inserted += 1

        conn.commit()
        return {"count": inserted, "weights_version": weights_version,
                "snapshot_date": now, "league_id": league_id}
    finally:
        conn.close()


def get_meta_trend(player_name: str, limit: int = 30) -> list[dict]:
    """Get meta score history for a player across all snapshots.

    Returns list of {snapshot_date, meta_score, meta_vs_rhp, meta_vs_lhp,
    league_id, weights_version}.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT snapshot_date, meta_score, meta_vs_rhp, meta_vs_lhp, "
            "       league_id, weights_version "
            "FROM meta_history "
            "WHERE player_name = ? "
            "ORDER BY snapshot_date DESC LIMIT ?",
            (player_name, limit),
        ).fetchall()

        return [
            {
                "snapshot_date": r["snapshot_date"],
                "meta_score": _safe_float(r["meta_score"]),
                "meta_vs_rhp": _safe_float(r["meta_vs_rhp"]),
                "meta_vs_lhp": _safe_float(r["meta_vs_lhp"]),
                "league_id": r["league_id"],
                "weights_version": r["weights_version"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_league_comparison(card_id: int = None, player_name: str = None) -> list[dict]:
    """Compare a player's meta across different leagues.

    Requires either card_id or player_name. Joins meta_history with leagues
    and, where available, batting/pitching stats for OPS/ERA context.

    Returns list of {league_id, league_tier, avg_meta, games_played, war,
    ops_or_era}.
    """
    if card_id is None and player_name is None:
        raise ValueError("Provide either card_id or player_name")

    conn = get_connection()
    try:
        # Build the WHERE clause for meta_history
        if card_id is not None:
            where = "mh.card_id = ?"
            params: list = [card_id]
        else:
            where = "mh.player_name = ?"
            params = [player_name]

        rows = conn.execute(
            f"""
            SELECT mh.league_id,
                   l.league_tier,
                   ROUND(AVG(mh.meta_score), 2) AS avg_meta,
                   COUNT(*) AS snapshot_count
            FROM meta_history mh
            LEFT JOIN leagues l ON l.league_id = mh.league_id
            WHERE {where}
            GROUP BY mh.league_id
            ORDER BY avg_meta DESC
            """,
            tuple(params),
        ).fetchall()

        # Resolve player_name for stat lookups if only card_id was given
        lookup_name = player_name
        if lookup_name is None and card_id is not None:
            name_row = conn.execute(
                "SELECT player_name FROM meta_history WHERE card_id = ? LIMIT 1",
                (card_id,),
            ).fetchone()
            if name_row:
                lookup_name = name_row["player_name"]

        results = []
        for r in rows:
            lid = r["league_id"]
            entry = {
                "league_id": lid,
                "league_tier": r["league_tier"],
                "avg_meta": _safe_float(r["avg_meta"]),
                "snapshot_count": r["snapshot_count"],
                "games_played": 0,
                "war": 0.0,
                "ops_or_era": None,
            }

            # Try to pull aggregate game stats for this player + league
            if lookup_name:
                # Batting stats
                bat = conn.execute(
                    "SELECT SUM(games) as g, SUM(war) as w, "
                    "       AVG(ops) as ops "
                    "FROM batting_stats WHERE player_name = ? "
                    "AND league_id = ?",
                    (lookup_name, lid),
                ).fetchone()
                if bat and bat["g"]:
                    entry["games_played"] = int(bat["g"] or 0)
                    entry["war"] = _safe_float(bat["w"])
                    entry["ops_or_era"] = round(_safe_float(bat["ops"]), 3)

                # Pitching stats (overwrite if pitcher)
                pit = conn.execute(
                    "SELECT SUM(games) as g, SUM(war) as w, "
                    "       AVG(era) as era "
                    "FROM pitching_stats WHERE player_name = ? "
                    "AND league_id = ?",
                    (lookup_name, lid),
                ).fetchone()
                if pit and pit["g"] and int(pit["g"]) > 0:
                    entry["games_played"] = int(pit["g"])
                    entry["war"] = _safe_float(pit["w"])
                    entry["ops_or_era"] = round(_safe_float(pit["era"]), 2)

            results.append(entry)

        return results
    finally:
        conn.close()


def get_meta_movers(league_id: str = None, days: int = 7,
                    top_n: int = 20) -> dict:
    """Find players whose meta changed most between snapshots.

    Compares the most recent snapshot to the snapshot closest to *days* ago.
    If league_id is provided, only considers snapshots for that league.

    Returns {risers: [...], fallers: [...]} with delta info.
    Each entry: {player_name, position, meta_old, meta_new, delta, pct_change}.
    """
    conn = get_connection()
    try:
        league_filter = ""
        params: list = []
        if league_id:
            league_filter = "AND league_id = ? "
            params = [league_id]

        # Find the latest snapshot date
        latest = conn.execute(
            f"SELECT MAX(DATE(snapshot_date)) as d FROM meta_history "
            f"WHERE 1=1 {league_filter}",
            tuple(params),
        ).fetchone()
        if not latest or not latest["d"]:
            return {"risers": [], "fallers": []}
        latest_date = latest["d"]

        # Target date for comparison
        cutoff = (datetime.strptime(latest_date, "%Y-%m-%d") - timedelta(days=days)
                  ).strftime("%Y-%m-%d")

        # Find closest earlier snapshot date
        earlier = conn.execute(
            f"SELECT DISTINCT DATE(snapshot_date) as d FROM meta_history "
            f"WHERE DATE(snapshot_date) <= ? {league_filter} "
            f"ORDER BY DATE(snapshot_date) DESC LIMIT 1",
            (cutoff, *params),
        ).fetchone()
        if not earlier or not earlier["d"]:
            # Fall back: just take the earliest available snapshot
            earlier = conn.execute(
                f"SELECT MIN(DATE(snapshot_date)) as d FROM meta_history "
                f"WHERE DATE(snapshot_date) < ? {league_filter}",
                (latest_date, *params),
            ).fetchone()
        if not earlier or not earlier["d"]:
            return {"risers": [], "fallers": []}
        earlier_date = earlier["d"]

        # Fetch latest scores
        new_rows = conn.execute(
            f"SELECT player_name, position, meta_score, card_id "
            f"FROM meta_history "
            f"WHERE DATE(snapshot_date) = ? {league_filter}",
            (latest_date, *params),
        ).fetchall()
        new_map = {r["player_name"]: dict(r) for r in new_rows}

        # Fetch earlier scores
        old_rows = conn.execute(
            f"SELECT player_name, position, meta_score, card_id "
            f"FROM meta_history "
            f"WHERE DATE(snapshot_date) = ? {league_filter}",
            (earlier_date, *params),
        ).fetchall()
        old_map = {r["player_name"]: dict(r) for r in old_rows}

        # Calculate deltas for players present in both snapshots
        deltas = []
        for name in new_map.keys() & old_map.keys():
            meta_new = _safe_float(new_map[name]["meta_score"])
            meta_old = _safe_float(old_map[name]["meta_score"])
            delta = round(meta_new - meta_old, 2)
            pct = round((delta / meta_old) * 100, 2) if meta_old else 0.0
            deltas.append({
                "player_name": name,
                "position": new_map[name].get("position", ""),
                "card_id": new_map[name].get("card_id"),
                "meta_old": meta_old,
                "meta_new": meta_new,
                "delta": delta,
                "pct_change": pct,
                "old_date": earlier_date,
                "new_date": latest_date,
            })

        # Sort and split into risers / fallers
        deltas.sort(key=lambda d: d["delta"], reverse=True)
        risers = [d for d in deltas if d["delta"] > 0][:top_n]
        fallers = [d for d in deltas if d["delta"] < 0]
        fallers.sort(key=lambda d: d["delta"])
        fallers = fallers[:top_n]

        return {"risers": risers, "fallers": fallers}
    finally:
        conn.close()


def snapshot_player_history(league_id: str, export_number: int = None,
                            games_into_season: int = None,
                            team_record: str = None) -> dict:
    """Take a comprehensive snapshot of ALL player data for trending.

    Captures card value, meta scores, key ratings, AND in-game performance
    in a single row per player. Call this after each data export/import.

    Returns dict with count, export_number, league_id.
    """
    conn = get_connection()
    try:
        batting_w, pitching_w = get_weights()
        wv = _weights_hash(batting_w, pitching_w)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Auto-determine export_number if not provided
        if export_number is None:
            row = conn.execute(
                "SELECT COALESCE(MAX(export_number), 0) FROM player_history "
                "WHERE league_id = ?", (league_id,)
            ).fetchone()
            export_number = (row[0] or 0) + 1

        # Log this export event
        conn.execute(
            "INSERT INTO export_log (league_id, export_number, games_played, "
            "team_record, files_imported) VALUES (?, ?, ?, ?, ?)",
            (league_id, export_number, games_into_season, team_record, 0)
        )

        # Get all cards with their market data
        cards = conn.execute("""
            SELECT card_id, card_title, position, position_name,
                   pitcher_role, pitcher_role_name,
                   card_value, buy_order_high, sell_order_low,
                   last_10_price, last_10_variance,
                   contact, gap_power, power, eye, avoid_ks, babip,
                   speed, stealing,
                   stuff, movement, control, p_hr, stamina, hold,
                   infield_range, infield_error, infield_arm,
                   of_range, of_error, of_arm,
                   catcher_ability, catcher_frame, catcher_arm,
                   meta_score_batting, meta_score_pitching
            FROM cards WHERE owned = 1
        """).fetchall()

        # Get latest batting stats keyed by card_id
        bat_stats = {}
        for r in conn.execute("""
            SELECT card_id, games, pa, avg, obp, slg, ops, ops_plus,
                   hr, rbi, war, sb
            FROM batting_stats
            WHERE card_id IS NOT NULL
              AND snapshot_date = (SELECT MAX(snapshot_date) FROM batting_stats)
        """).fetchall():
            bat_stats[r["card_id"]] = dict(r)

        # Get latest pitching stats keyed by card_id
        pit_stats = {}
        for r in conn.execute("""
            SELECT card_id, games, ip, era, whip, k_per_9, bb_per_9,
                   fip, era_plus, war
            FROM pitching_stats
            WHERE card_id IS NOT NULL
              AND snapshot_date = (SELECT MAX(snapshot_date) FROM pitching_stats)
        """).fetchall():
            pit_stats[r["card_id"]] = dict(r)

        inserted = 0
        for c in cards:
            cd = dict(c)
            cid = cd["card_id"]
            is_pitcher = cd.get("pitcher_role") is not None

            # Recalculate meta with current weights
            if is_pitcher:
                meta = calc_pitching_meta(cd, pitching_w)
                meta_vr = calc_pitching_meta_vs_rhb(cd, pitching_w)
                meta_vl = calc_pitching_meta_vs_lhb(cd, pitching_w)
            else:
                meta = calc_batting_meta(cd, batting_w)
                meta_vr = calc_batting_meta_vs_rhp(cd, batting_w)
                meta_vl = calc_batting_meta_vs_lhp(cd, batting_w)

            # Pull stats
            bs = bat_stats.get(cid, {})
            ps = pit_stats.get(cid, {})

            player_name = cd.get("card_title", "")
            pos = cd.get("position_name", "") or str(cd.get("position", ""))
            role = cd.get("pitcher_role_name", "")

            conn.execute("""
                INSERT INTO player_history (
                    card_id, player_name, position, pitcher_role, league_id,
                    card_value, buy_order_high, sell_order_low,
                    last_10_price, last_10_variance,
                    meta_score, meta_vs_rhp, meta_vs_lhp,
                    contact, gap_power, power, eye, avoid_ks, babip,
                    speed, stealing,
                    stuff, movement, control, p_hr, stamina, hold,
                    games, pa, avg, obp, slg, ops, ops_plus,
                    hr, rbi, war, sb,
                    ip, era, whip, k_per_9, bb_per_9, fip, era_plus, p_war,
                    weights_version, export_number, games_into_season,
                    snapshot_date
                ) VALUES (
                    ?,?,?,?,?, ?,?,?,?,?, ?,?,?, ?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?, ?,?,?,?,?,?,?, ?,?,?,?,
                    ?,?,?,?,?,?,?,?, ?,?,?, ?
                )
            """, (
                cid, player_name, pos, role, league_id,
                cd.get("card_value"), cd.get("buy_order_high"),
                cd.get("sell_order_low"), cd.get("last_10_price"),
                cd.get("last_10_variance"),
                meta, meta_vr, meta_vl,
                cd.get("contact"), cd.get("gap_power"), cd.get("power"),
                cd.get("eye"), cd.get("avoid_ks"), cd.get("babip"),
                cd.get("speed"), cd.get("stealing"),
                cd.get("stuff"), cd.get("movement"), cd.get("control"),
                cd.get("p_hr"), cd.get("stamina"), cd.get("hold"),
                # batting stats
                bs.get("games"), bs.get("pa"), bs.get("avg"), bs.get("obp"),
                bs.get("slg"), bs.get("ops"), bs.get("ops_plus"),
                bs.get("hr"), bs.get("rbi"), bs.get("war"), bs.get("sb"),
                # pitching stats
                ps.get("ip"), ps.get("era"), ps.get("whip"),
                ps.get("k_per_9"), ps.get("bb_per_9"),
                ps.get("fip"), ps.get("era_plus"), ps.get("war"),
                wv, export_number, games_into_season, now,
            ))
            inserted += 1

        conn.commit()
        return {
            "count": inserted, "export_number": export_number,
            "league_id": league_id, "weights_version": wv,
            "snapshot_date": now,
        }
    finally:
        conn.close()


def get_player_trend(player_name: str = None, card_id: int = None,
                     league_id: str = None) -> list[dict]:
    """Get full history for a player across exports.

    Returns list of dicts with meta, stats, card value per export.
    """
    conn = get_connection()
    try:
        conditions = []
        params = []
        if player_name:
            conditions.append("player_name LIKE ?")
            params.append(f"%{player_name}%")
        if card_id:
            conditions.append("card_id = ?")
            params.append(card_id)
        if league_id:
            conditions.append("league_id = ?")
            params.append(league_id)

        if not conditions:
            return []

        where = " AND ".join(conditions)
        rows = conn.execute(f"""
            SELECT * FROM player_history
            WHERE {where}
            ORDER BY snapshot_date ASC
        """, tuple(params)).fetchall()

        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_market_trends(league_id: str = None, min_exports: int = 2,
                      top_n: int = 20) -> dict:
    """Find cards with biggest price movements across exports.

    Returns {risers: [...], fallers: [...]} based on sell_order_low changes.
    """
    conn = get_connection()
    try:
        league_filter = "AND league_id = ?" if league_id else ""
        params = [league_id] if league_id else []

        # Get first and last export for each card
        rows = conn.execute(f"""
            SELECT card_id, player_name, position,
                   MIN(export_number) as first_exp,
                   MAX(export_number) as last_exp,
                   COUNT(DISTINCT export_number) as num_exports
            FROM player_history
            WHERE sell_order_low IS NOT NULL AND sell_order_low > 0
              {league_filter}
            GROUP BY card_id
            HAVING num_exports >= ?
        """, (*params, min_exports)).fetchall()

        deltas = []
        for r in rows:
            # Get first and last prices
            first = conn.execute(
                "SELECT sell_order_low, meta_score FROM player_history "
                "WHERE card_id = ? AND export_number = ? "
                f"{league_filter} LIMIT 1",
                (r["card_id"], r["first_exp"], *params)
            ).fetchone()
            last = conn.execute(
                "SELECT sell_order_low, meta_score FROM player_history "
                "WHERE card_id = ? AND export_number = ? "
                f"{league_filter} LIMIT 1",
                (r["card_id"], r["last_exp"], *params)
            ).fetchone()

            if first and last and first["sell_order_low"] and last["sell_order_low"]:
                price_delta = (last["sell_order_low"] or 0) - (first["sell_order_low"] or 0)
                pct = round(price_delta / first["sell_order_low"] * 100, 1) if first["sell_order_low"] else 0
                deltas.append({
                    "card_id": r["card_id"],
                    "player_name": r["player_name"],
                    "position": r["position"],
                    "price_old": first["sell_order_low"],
                    "price_new": last["sell_order_low"],
                    "price_delta": price_delta,
                    "pct_change": pct,
                    "meta_score": _safe_float(last["meta_score"]),
                    "num_exports": r["num_exports"],
                })

        deltas.sort(key=lambda d: d["price_delta"], reverse=True)
        risers = [d for d in deltas if d["price_delta"] > 0][:top_n]
        fallers = sorted([d for d in deltas if d["price_delta"] < 0],
                         key=lambda d: d["price_delta"])[:top_n]

        return {"risers": risers, "fallers": fallers}
    finally:
        conn.close()


def tag_existing_data(conn=None):
    """One-time migration: tag existing batting_stats and pitching_stats rows
    with league_id.

    Logic:
    - snapshot_date on 2026-04-11 or 2026-04-12 -> league_id = 'i76'
    - snapshot_date on 2026-04-13+            -> league_id = 'lb124'

    Also inserts league records into the leagues table:
    - i76:   Iron league, ended
    - lb124: Low Bronze, current
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        # Ensure league_id columns exist on stats tables
        for table in ("batting_stats", "pitching_stats"):
            cols = [
                r[1]
                for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
            ]
            if "league_id" not in cols:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN league_id TEXT"
                )

        # Insert league records
        ensure_league_exists("i76", "Iron 76", "Iron", conn=conn)
        ensure_league_exists("lb124", "Low Bronze 124", "Low Bronze", conn=conn)

        # Update end_date for i76 (finished league)
        conn.execute(
            "UPDATE leagues SET end_date = '2026-04-12' "
            "WHERE league_id = 'i76' AND end_date IS NULL"
        )

        # Tag stats rows
        for table in ("batting_stats", "pitching_stats"):
            conn.execute(
                f"UPDATE {table} SET league_id = 'i76' "
                f"WHERE league_id IS NULL "
                f"AND DATE(snapshot_date) IN ('2026-04-11', '2026-04-12')"
            )
            conn.execute(
                f"UPDATE {table} SET league_id = 'lb124' "
                f"WHERE league_id IS NULL "
                f"AND DATE(snapshot_date) >= '2026-04-13'"
            )

        conn.commit()
    finally:
        if close_conn:
            conn.close()
