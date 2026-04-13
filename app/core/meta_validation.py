"""Meta Validation Engine — compare meta score predictions against actual in-game performance.

Uses batting_stats and pitching_stats tables (from CSV stat exports) to validate
whether the meta scoring system accurately predicts real in-game outcomes.
"""
import json
import logging
import math

from app.core.database import get_connection

logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy matching."""
    return name.strip().lower().replace(".", "").replace("'", "")


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Calculate Pearson correlation coefficient between two lists."""
    n = len(xs)
    if n < 3:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / (den_x * den_y)


def _rank_correlation(xs: list[float], ys: list[float]) -> float:
    """Calculate Spearman rank correlation between two lists."""
    n = len(xs)
    if n < 3:
        return 0.0

    def _ranks(vals):
        sorted_idx = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
        ranks = [0.0] * len(vals)
        for rank, idx in enumerate(sorted_idx, 1):
            ranks[idx] = float(rank)
        return ranks

    rx = _ranks(xs)
    ry = _ranks(ys)
    return _pearson_correlation(rx, ry)


def _safe_float(val, default=0.0) -> float:
    """Safely convert a value to float."""
    try:
        v = float(val)
        return v if not math.isnan(v) else default
    except (ValueError, TypeError):
        return default


def validate_meta_vs_performance(conn=None) -> dict:
    """Compare roster meta scores against actual in-game performance (batting + pitching).

    Uses the batting_stats and pitching_stats tables from CSV stat exports.

    Returns dict with:
        players: list of player dicts with meta_score, performance stats, etc.
        correlation: Pearson correlation between meta_score and performance_rating
        rank_correlation: Spearman rank correlation
        overperformers: players performing above their meta prediction
        underperformers: players performing below their meta prediction
        batting_correlation: batting-specific correlation
        pitching_correlation: pitching-specific correlation
        weight_suggestions: suggested weight adjustments
        message: status/error message
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    empty_result = {
        "players": [],
        "correlation": 0.0,
        "rank_correlation": 0.0,
        "batting_correlation": 0.0,
        "pitching_correlation": 0.0,
        "overperformers": [],
        "underperformers": [],
        "weight_suggestions": {},
        "message": "",
    }

    try:
        # Check if stats tables exist and have data
        bat_count = conn.execute(
            "SELECT COUNT(*) as c FROM batting_stats"
        ).fetchone()["c"]
        pitch_count = conn.execute(
            "SELECT COUNT(*) as c FROM pitching_stats"
        ).fetchone()["c"]

        if bat_count == 0 and pitch_count == 0:
            empty_result["message"] = (
                "No game stats found. Export your batting and pitching stats CSVs "
                "from OOTP and import them on the main page."
            )
            return empty_result

        # --- Load roster with meta scores ---
        roster_rows = conn.execute(
            "SELECT player_name, position, lineup_role, ovr, meta_score "
            "FROM roster_current ORDER BY meta_score DESC"
        ).fetchall()

        if not roster_rows:
            empty_result["message"] = "No roster data found. Import your roster first."
            return empty_result

        roster_lookup = {}
        for r in roster_rows:
            key = _normalize_name(r["player_name"])
            roster_lookup[key] = dict(r)

        all_players = []
        matched_roster_keys = set()  # Prevent duplicate matches

        # --- Match batters ---
        # Get latest snapshot for each batter
        bat_rows = conn.execute("""
            SELECT bs.player_name, bs.position, bs.games, bs.ab, bs.pa,
                   bs.avg, bs.obp, bs.slg, bs.ops, bs.ops_plus, bs.iso,
                   bs.hr, bs.rbi, bs.runs, bs.bb, bs.k, bs.war, bs.babip,
                   bs.sb, bs.cs, bs.hits, bs.doubles, bs.triples
            FROM batting_stats bs
            INNER JOIN (
                SELECT player_name, MAX(snapshot_date) as max_date
                FROM batting_stats GROUP BY player_name
            ) latest ON bs.player_name = latest.player_name
                    AND bs.snapshot_date = latest.max_date
            WHERE bs.ab >= 10
            ORDER BY bs.ops DESC
        """).fetchall()

        for row in bat_rows:
            norm = _normalize_name(row["player_name"])

            # Match to roster (exact first, then partial)
            roster_entry = None
            matched_key = None

            if norm in roster_lookup and norm not in matched_roster_keys:
                roster_entry = roster_lookup[norm]
                matched_key = norm
            else:
                for rkey, rval in roster_lookup.items():
                    if rkey in matched_roster_keys:
                        continue
                    if rkey in norm or norm in rkey:
                        roster_entry = rval
                        matched_key = rkey
                        break

            if roster_entry is None:
                continue
            matched_roster_keys.add(matched_key)

            meta_score = _safe_float(roster_entry.get("meta_score", 0))
            ops = _safe_float(row["ops"])
            performance_rating = round(ops * 1000, 1)
            meta_vs_perf_gap = round(meta_score - performance_rating, 1)

            all_players.append({
                "player_name": roster_entry["player_name"],
                "position": roster_entry.get("position", row["position"] or ""),
                "player_type": "batter",
                "meta_score": meta_score,
                "performance_rating": performance_rating,
                "meta_vs_perf_gap": meta_vs_perf_gap,
                # Batting stats
                "games": int(row["games"]),
                "ab": int(row["ab"]),
                "in_game_avg": round(_safe_float(row["avg"]), 3),
                "in_game_obp": round(_safe_float(row["obp"]), 3),
                "in_game_slg": round(_safe_float(row["slg"]), 3),
                "in_game_ops": round(ops, 3),
                "in_game_ops_plus": int(row["ops_plus"]),
                "in_game_iso": round(_safe_float(row["iso"]), 3),
                "in_game_hr": int(row["hr"]),
                "in_game_rbi": int(row["rbi"]),
                "in_game_war": round(_safe_float(row["war"]), 1),
                "in_game_babip": round(_safe_float(row["babip"]), 3),
                "in_game_k": int(row["k"]),
                "in_game_bb": int(row["bb"]),
                "in_game_sb": int(row["sb"]),
            })

        # --- Match pitchers ---
        pitch_rows = conn.execute("""
            SELECT ps.player_name, ps.position, ps.games, ps.gs,
                   ps.wins, ps.losses, ps.saves, ps.holds,
                   ps.ip, ps.era, ps.whip, ps.k, ps.bb, ps.hr_allowed,
                   ps.k_per_9, ps.bb_per_9, ps.era_plus, ps.fip, ps.war,
                   ps.babip, ps.avg_against
            FROM pitching_stats ps
            INNER JOIN (
                SELECT player_name, MAX(snapshot_date) as max_date
                FROM pitching_stats GROUP BY player_name
            ) latest ON ps.player_name = latest.player_name
                    AND ps.snapshot_date = latest.max_date
            WHERE ps.ip >= 10
            ORDER BY ps.era ASC
        """).fetchall()

        for row in pitch_rows:
            norm = _normalize_name(row["player_name"])

            roster_entry = None
            matched_key = None
            if norm in roster_lookup and norm not in matched_roster_keys:
                roster_entry = roster_lookup[norm]
                matched_key = norm
            else:
                for rkey, rval in roster_lookup.items():
                    if rkey in matched_roster_keys:
                        continue
                    if rkey in norm or norm in rkey:
                        roster_entry = rval
                        matched_key = rkey
                        break

            if roster_entry is None:
                continue
            matched_roster_keys.add(matched_key)

            meta_score = _safe_float(roster_entry.get("meta_score", 0))
            era = _safe_float(row["era"])
            era_plus = _safe_float(row["era_plus"])

            # For pitchers: use ERA+ as performance (higher = better, like meta)
            # Cap ERA+ at 200 to prevent extreme reliever values from skewing
            capped_era_plus = min(era_plus, 200)
            performance_rating = round(capped_era_plus * 5, 1)  # ERA+ 100 -> 500, ERA+ 150 -> 750
            meta_vs_perf_gap = round(meta_score - performance_rating, 1)

            all_players.append({
                "player_name": roster_entry["player_name"],
                "position": roster_entry.get("position", row["position"] or ""),
                "player_type": "pitcher",
                "meta_score": meta_score,
                "performance_rating": performance_rating,
                "meta_vs_perf_gap": meta_vs_perf_gap,
                # Pitching stats
                "games": int(row["games"]),
                "ip": round(_safe_float(row["ip"]), 1),
                "in_game_era": round(era, 2),
                "in_game_whip": round(_safe_float(row["whip"]), 2),
                "in_game_k": int(row["k"]),
                "in_game_bb": int(row["bb"]),
                "in_game_k_per_9": round(_safe_float(row["k_per_9"]), 1),
                "in_game_era_plus": int(era_plus),
                "in_game_fip": round(_safe_float(row["fip"]), 2),
                "in_game_war": round(_safe_float(row["war"]), 1),
                "in_game_wins": int(row["wins"]),
                "in_game_losses": int(row["losses"]),
                "in_game_saves": int(row["saves"]),
            })

        if not all_players:
            empty_result["message"] = (
                "No player matches found between roster and game stats. "
                "Make sure your roster and stats CSVs have matching player names."
            )
            return empty_result

        # --- Overall correlation ---
        meta_scores = [p["meta_score"] for p in all_players]
        perf_ratings = [p["performance_rating"] for p in all_players]

        correlation = round(_pearson_correlation(meta_scores, perf_ratings), 3)
        rank_corr = round(_rank_correlation(meta_scores, perf_ratings), 3)

        # --- Batting-only correlation ---
        bat_players = [p for p in all_players if p["player_type"] == "batter"]
        if len(bat_players) >= 3:
            bat_metas = [p["meta_score"] for p in bat_players]
            bat_perfs = [p["performance_rating"] for p in bat_players]
            batting_corr = round(_pearson_correlation(bat_metas, bat_perfs), 3)
        else:
            batting_corr = 0.0

        # --- Pitching-only correlation ---
        pitch_players = [p for p in all_players if p["player_type"] == "pitcher"]
        if len(pitch_players) >= 3:
            pitch_metas = [p["meta_score"] for p in pitch_players]
            pitch_perfs = [p["performance_rating"] for p in pitch_players]
            pitching_corr = round(_pearson_correlation(pitch_metas, pitch_perfs), 3)
        else:
            pitching_corr = 0.0

        # --- Classify over/underperformers ---
        overperformers = sorted(
            [p for p in all_players if p["meta_vs_perf_gap"] < -20],
            key=lambda p: p["meta_vs_perf_gap"],
        )
        underperformers = sorted(
            [p for p in all_players if p["meta_vs_perf_gap"] > 20],
            key=lambda p: p["meta_vs_perf_gap"],
            reverse=True,
        )

        # --- Weight suggestions ---
        weight_suggestions = suggest_weight_adjustments(bat_players, conn)

        return {
            "players": sorted(all_players, key=lambda p: p["meta_vs_perf_gap"]),
            "correlation": correlation,
            "rank_correlation": rank_corr,
            "batting_correlation": batting_corr,
            "pitching_correlation": pitching_corr,
            "overperformers": overperformers,
            "underperformers": underperformers,
            "weight_suggestions": weight_suggestions,
            "batter_count": len(bat_players),
            "pitcher_count": len(pitch_players),
            "message": (
                f"Matched {len(bat_players)} batters + {len(pitch_players)} pitchers. "
                f"Overall correlation: {correlation:.3f}"
            ),
        }

    except Exception as e:
        logger.error(f"Meta validation error: {e}", exc_info=True)
        empty_result["message"] = f"Error during validation: {e}"
        return empty_result

    finally:
        if close_conn:
            conn.close()


def suggest_weight_adjustments(players: list, conn=None) -> dict:
    """Analyze gap patterns and suggest meta weight adjustments.

    Correlates individual card rating components (contact, gap, power, etc.)
    with actual in-game OPS performance.

    Returns dict of { component: { 'adjustment': float, 'reason': str } }.
    """
    if not players or len(players) < 3:
        return {}

    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        player_names = [p["player_name"] for p in players]
        placeholders = ",".join("?" * len(player_names))

        card_rows = conn.execute(
            f"SELECT card_title, contact, gap_power, power, eye, avoid_ks, babip "
            f"FROM cards WHERE card_title IN ({placeholders}) AND owned = 1",
            player_names,
        ).fetchall()

        if not card_rows:
            card_rows = []
            for name in player_names:
                parts = name.strip().split()
                if len(parts) >= 2:
                    last = parts[-1]
                    rows = conn.execute(
                        "SELECT card_title, contact, gap_power, power, eye, avoid_ks, babip "
                        "FROM cards WHERE card_title LIKE ? AND owned = 1",
                        (f"%{last}%",),
                    ).fetchall()
                    card_rows.extend(rows)

        if not card_rows:
            return {"message": "No card rating data found to analyze weight adjustments."}

        card_lookup = {}
        for cr in card_rows:
            key = _normalize_name(cr["card_title"])
            card_lookup[key] = dict(cr)

        components = {
            "contact": [],
            "gap_power": [],
            "power": [],
            "eye": [],
            "avoid_ks": [],
            "babip": [],
        }
        perf_vals = []

        for p in players:
            norm = _normalize_name(p["player_name"])
            card = card_lookup.get(norm)
            if card is None:
                for ck, cv in card_lookup.items():
                    if ck in norm or norm in ck:
                        card = cv
                        break
            if card is None:
                continue

            perf_vals.append(p["performance_rating"])
            for comp in components:
                components[comp].append(_safe_float(card.get(comp, 0)))

        if len(perf_vals) < 3:
            return {"message": "Not enough matched cards to suggest adjustments."}

        component_corrs = {}
        for comp, vals in components.items():
            if len(vals) == len(perf_vals):
                component_corrs[comp] = _pearson_correlation(vals, perf_vals)

        from app.utils.constants import DEFAULT_BATTING_WEIGHTS

        weight_key_map = {
            "contact": "contact",
            "gap_power": "gap_power",
            "power": "power",
            "eye": "eye",
            "avoid_ks": "avoid_ks",
            "babip": "babip",
        }

        sorted_by_corr = sorted(component_corrs.items(), key=lambda x: x[1], reverse=True)
        sorted_by_weight = sorted(
            [(k, DEFAULT_BATTING_WEIGHTS.get(weight_key_map.get(k, k), 0))
             for k in component_corrs],
            key=lambda x: x[1],
            reverse=True,
        )

        corr_ranks = {comp: i for i, (comp, _) in enumerate(sorted_by_corr)}
        weight_ranks = {comp: i for i, (comp, _) in enumerate(sorted_by_weight)}

        suggestions = {}
        for comp in component_corrs:
            corr_rank = corr_ranks[comp]
            weight_rank = weight_ranks[comp]
            rank_diff = weight_rank - corr_rank

            current_weight = DEFAULT_BATTING_WEIGHTS.get(weight_key_map.get(comp, comp), 1.0)
            corr_val = component_corrs[comp]

            if abs(rank_diff) >= 2:
                adjustment = round(rank_diff * 0.10, 2)
                if rank_diff > 0:
                    reason = (
                        f"{comp} correlates strongly with performance (r={corr_val:.2f}) "
                        f"but has a relatively low weight ({current_weight:.2f}). "
                        f"Consider increasing."
                    )
                else:
                    reason = (
                        f"{comp} has a high weight ({current_weight:.2f}) but weak "
                        f"correlation with performance (r={corr_val:.2f}). "
                        f"Consider decreasing."
                    )
                suggestions[comp] = {
                    "adjustment": adjustment,
                    "current_weight": current_weight,
                    "correlation": round(corr_val, 3),
                    "reason": reason,
                }
            elif abs(corr_val) < 0.1 and current_weight > 1.0:
                suggestions[comp] = {
                    "adjustment": -0.15,
                    "current_weight": current_weight,
                    "correlation": round(corr_val, 3),
                    "reason": (
                        f"{comp} shows near-zero correlation with actual performance "
                        f"(r={corr_val:.2f}) despite weight of {current_weight:.2f}."
                    ),
                }

        return suggestions

    except Exception as e:
        logger.error(f"Weight adjustment suggestion error: {e}", exc_info=True)
        return {"message": f"Error analyzing weights: {e}"}

    finally:
        if close_conn:
            conn.close()


def get_meta_accuracy_score(conn=None) -> dict:
    """Quick summary of how well the meta correlates with actual performance.

    Returns dict with:
        accuracy_pct: 0-100 score
        sample_size: number of players matched
        top_overperformer: player outperforming meta the most
        top_underperformer: player underperforming meta the most
        message: status message
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        result = validate_meta_vs_performance(conn)

        players = result.get("players", [])
        if not players:
            return {
                "accuracy_pct": 0,
                "sample_size": 0,
                "top_overperformer": None,
                "top_underperformer": None,
                "message": result.get("message", "No data available."),
            }

        rank_corr = result.get("rank_correlation", 0.0)
        accuracy_pct = round(max(0, min(100, (rank_corr + 1) * 50)), 1)

        overperformers = result.get("overperformers", [])
        underperformers = result.get("underperformers", [])

        top_over = None
        if overperformers:
            p = overperformers[0]
            top_over = {
                "player_name": p["player_name"],
                "meta_score": p["meta_score"],
                "performance_rating": p["performance_rating"],
                "gap": p["meta_vs_perf_gap"],
            }

        top_under = None
        if underperformers:
            p = underperformers[0]
            top_under = {
                "player_name": p["player_name"],
                "meta_score": p["meta_score"],
                "performance_rating": p["performance_rating"],
                "gap": p["meta_vs_perf_gap"],
            }

        return {
            "accuracy_pct": accuracy_pct,
            "sample_size": len(players),
            "top_overperformer": top_over,
            "top_underperformer": top_under,
            "message": (
                f"Meta accuracy: {accuracy_pct}% based on {len(players)} players. "
                f"Rank correlation: {rank_corr:.3f}"
            ),
        }

    except Exception as e:
        logger.error(f"Meta accuracy score error: {e}", exc_info=True)
        return {
            "accuracy_pct": 0,
            "sample_size": 0,
            "top_overperformer": None,
            "top_underperformer": None,
            "message": f"Error: {e}",
        }

    finally:
        if close_conn:
            conn.close()


def get_stats_summary(conn=None) -> dict:
    """Get a quick summary of stored game stats for display on the dashboard.

    Returns dict with batting and pitching leader info.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        result = {
            "has_batting_stats": False,
            "has_pitching_stats": False,
            "batting_count": 0,
            "pitching_count": 0,
            "mvp": None,
            "cy_young": None,
            "team_avg": None,
            "team_era": None,
            "team_ops": None,
        }

        # Batting summary
        bat_count = conn.execute("SELECT COUNT(DISTINCT player_name) as c FROM batting_stats").fetchone()["c"]
        result["batting_count"] = bat_count
        result["has_batting_stats"] = bat_count > 0

        if bat_count > 0:
            # Team averages
            team = conn.execute("""
                SELECT AVG(avg) as team_avg, AVG(ops) as team_ops
                FROM batting_stats WHERE ab >= 50
            """).fetchone()
            if team:
                result["team_avg"] = round(team["team_avg"] or 0, 3)
                result["team_ops"] = round(team["team_ops"] or 0, 3)

            # MVP (highest WAR batter)
            mvp = conn.execute("""
                SELECT player_name, position, war, ops, hr, rbi, avg
                FROM batting_stats WHERE ab >= 50
                ORDER BY war DESC LIMIT 1
            """).fetchone()
            if mvp:
                result["mvp"] = dict(mvp)

        # Pitching summary
        pitch_count = conn.execute("SELECT COUNT(DISTINCT player_name) as c FROM pitching_stats").fetchone()["c"]
        result["pitching_count"] = pitch_count
        result["has_pitching_stats"] = pitch_count > 0

        if pitch_count > 0:
            # Team ERA
            team_era = conn.execute("""
                SELECT AVG(era) as team_era FROM pitching_stats WHERE ip >= 30
            """).fetchone()
            if team_era:
                result["team_era"] = round(team_era["team_era"] or 0, 2)

            # Cy Young (highest WAR pitcher)
            cy = conn.execute("""
                SELECT player_name, position, war, era, k, wins, losses
                FROM pitching_stats WHERE ip >= 30
                ORDER BY war DESC LIMIT 1
            """).fetchone()
            if cy:
                result["cy_young"] = dict(cy)

        return result

    except Exception as e:
        logger.error(f"Stats summary error: {e}", exc_info=True)
        return {
            "has_batting_stats": False, "has_pitching_stats": False,
            "batting_count": 0, "pitching_count": 0,
            "mvp": None, "cy_young": None,
            "team_avg": None, "team_era": None, "team_ops": None,
        }

    finally:
        if close_conn:
            conn.close()


# ---------------------------------------------------------------------------
# Auto-calibration system
# ---------------------------------------------------------------------------

_META_CALIBRATION_DDL = """
CREATE TABLE IF NOT EXISTS meta_calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calibration_type TEXT NOT NULL,
    weights_json TEXT NOT NULL,
    r_squared REAL,
    correlation REAL,
    sample_size INTEGER,
    confidence REAL,
    changes_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def _ensure_calibration_table(conn):
    """Create the meta_calibration table if it does not exist."""
    conn.executescript(_META_CALIBRATION_DDL)


def _r_squared(actual: list[float], predicted: list[float]) -> float:
    """Compute R-squared (coefficient of determination)."""
    n = len(actual)
    if n < 3:
        return 0.0
    mean_a = sum(actual) / n
    ss_tot = sum((a - mean_a) ** 2 for a in actual)
    if ss_tot == 0:
        return 0.0
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return 1.0 - ss_res / ss_tot


def auto_calibrate_weights(conn=None) -> dict:
    """Run auto-calibration using team's actual performance data.

    Uses robust individual-correlation-based weighting instead of OLS regression.
    Blends empirical weights with default weights using a confidence factor
    based on sample size (more data = more trust in empirical weights).

    Returns dict with:
        batting_weights: calibrated batting weights dict
        pitching_weights: calibrated pitching weights dict
        batting_r2: R-squared of calibrated model
        pitching_r2: R-squared of calibrated model
        changes: list of {stat, old_weight, new_weight, reason} dicts
        confidence: 0-1 confidence in calibration (based on sample size)
        message: status
    """
    from app.utils.constants import DEFAULT_BATTING_WEIGHTS, DEFAULT_PITCHING_WEIGHTS

    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    result = {
        "batting_weights": dict(DEFAULT_BATTING_WEIGHTS),
        "pitching_weights": dict(DEFAULT_PITCHING_WEIGHTS),
        "batting_r2": 0.0,
        "pitching_r2": 0.0,
        "changes": [],
        "confidence": 0.0,
        "message": "",
    }

    try:
        _ensure_calibration_table(conn)

        # ---------------------------------------------------------------
        # BATTING CALIBRATION
        # ---------------------------------------------------------------
        bat_stat_cols = ["contact", "gap_power", "power", "eye", "avoid_ks", "babip"]
        bat_weight_keys = ["contact", "gap_power", "power", "eye", "avoid_ks", "babip"]
        # defense handled separately (computed, not a single card column)

        # Join cards with batting_stats using card_id (most reliable) or name fallback
        bat_rows = conn.execute("""
            SELECT c.card_title, c.contact, c.gap_power, c.power, c.eye,
                   c.avoid_ks, c.babip, c.card_value,
                   c.infield_range, c.infield_error, c.infield_arm,
                   c.of_range, c.of_error, c.of_arm,
                   c.catcher_ability, c.catcher_frame, c.catcher_arm,
                   c.position, c.speed, c.stealing, c.baserunning,
                   bs.war, bs.ops, bs.pa
            FROM cards c
            INNER JOIN batting_stats bs ON bs.card_id = c.card_id
            INNER JOIN (
                SELECT card_id, MAX(snapshot_date) as max_date
                FROM batting_stats WHERE card_id IS NOT NULL
                GROUP BY card_id
            ) latest ON bs.card_id = latest.card_id
                    AND bs.snapshot_date = latest.max_date
            WHERE c.position != 1 AND bs.pa >= 100
        """).fetchall()

        # Fallback: name matching for stats without card_id
        if len(bat_rows) < 15:
            try:
                seen_ids = {r["card_title"] for r in bat_rows}
                extra = conn.execute("""
                    SELECT c.card_title, c.contact, c.gap_power, c.power, c.eye,
                           c.avoid_ks, c.babip, c.card_value,
                           c.infield_range, c.infield_error, c.infield_arm,
                           c.of_range, c.of_error, c.of_arm,
                           c.catcher_ability, c.catcher_frame, c.catcher_arm,
                           c.position, c.speed, c.stealing, c.baserunning,
                           bs.war, bs.ops, bs.pa
                    FROM cards c
                    INNER JOIN batting_stats bs
                        ON c.card_title LIKE '%' || bs.player_name || '%'
                    INNER JOIN (
                        SELECT player_name, MAX(snapshot_date) as max_date
                        FROM batting_stats WHERE card_id IS NULL
                        GROUP BY player_name
                    ) latest ON bs.player_name = latest.player_name
                            AND bs.snapshot_date = latest.max_date
                    WHERE c.position != 1 AND bs.pa >= 100
                """).fetchall()
                for r in extra:
                    if r["card_title"] not in seen_ids:
                        bat_rows.append(r)
                        seen_ids.add(r["card_title"])
            except Exception:
                pass

        bat_sample_size = len(bat_rows)
        bat_calibrated = dict(DEFAULT_BATTING_WEIGHTS)
        bat_confidence = 0.0
        bat_r2 = 0.0

        if bat_sample_size >= 15:
            bat_confidence = min(1.0, bat_sample_size / 100.0)

            # Compute defense score for each row
            from app.core.meta_scoring import calc_defense_score

            # Build parallel arrays for each stat and performance metrics
            stat_arrays = {k: [] for k in bat_stat_cols + ["defense"]}
            war_vals = []
            ops_vals = []

            for row in bat_rows:
                war_vals.append(_safe_float(row["war"]))
                ops_vals.append(_safe_float(row["ops"]))
                for col in bat_stat_cols:
                    stat_arrays[col].append(_safe_float(row[col]))
                stat_arrays["defense"].append(calc_defense_score(dict(row)))

            # Compute correlations with WAR and OPS for each stat
            all_bat_keys = bat_weight_keys + ["defense"]
            corr_war = {}
            corr_ops = {}
            for key in all_bat_keys:
                corr_war[key] = _pearson_correlation(stat_arrays[key], war_vals)
                corr_ops[key] = _pearson_correlation(stat_arrays[key], ops_vals)

            # Combined correlation: average of WAR corr and OPS corr
            combined_corr = {}
            for key in all_bat_keys:
                combined_corr[key] = (corr_war[key] + corr_ops[key]) / 2.0

            # Convert to proportional weights: use max(0, corr) so negative
            # correlations get zeroed out
            positive_corrs = {k: max(0.0, v) for k, v in combined_corr.items()}
            corr_sum = sum(positive_corrs.values())

            if corr_sum > 0:
                # Scale to match the total weight of the defaults (excluding ovr
                # and stamina_hold which are handled differently)
                default_total = sum(
                    DEFAULT_BATTING_WEIGHTS.get(k, 0.0) for k in all_bat_keys
                )
                empirical = {}
                for key in all_bat_keys:
                    empirical[key] = (positive_corrs[key] / corr_sum) * default_total

                # Blend: final = default*(1-conf*0.6) + empirical*(conf*0.6)
                blend = bat_confidence * 0.6
                for key in all_bat_keys:
                    default_w = DEFAULT_BATTING_WEIGHTS.get(key, 0.0)
                    emp_w = empirical[key]
                    new_w = round(default_w * (1.0 - blend) + emp_w * blend, 2)
                    bat_calibrated[key] = new_w

                # Compute R-squared using calibrated weights to predict WAR
                predicted_war = []
                for i in range(bat_sample_size):
                    pred = sum(
                        stat_arrays[key][i] * bat_calibrated.get(key, 0.0)
                        for key in all_bat_keys
                    )
                    predicted_war.append(pred)
                bat_r2 = _r_squared(war_vals, predicted_war)

                # Record changes
                for key in all_bat_keys:
                    old_w = DEFAULT_BATTING_WEIGHTS.get(key, 0.0)
                    new_w = bat_calibrated[key]
                    if abs(new_w - old_w) >= 0.05:
                        direction = "increased" if new_w > old_w else "decreased"
                        result["changes"].append({
                            "stat": key,
                            "type": "batting",
                            "old_weight": old_w,
                            "new_weight": new_w,
                            "reason": (
                                f"{key} {direction} from {old_w:.2f} to {new_w:.2f} "
                                f"(WAR r={corr_war[key]:.3f}, OPS r={corr_ops[key]:.3f}, "
                                f"n={bat_sample_size})"
                            ),
                        })
        else:
            if bat_sample_size > 0:
                result["message"] += (
                    f"Batting: only {bat_sample_size} players with 100+ PA "
                    f"(need 15). Using defaults. "
                )
            else:
                result["message"] += "Batting: no matched players found. Using defaults. "

        # ---------------------------------------------------------------
        # PITCHING CALIBRATION
        # ---------------------------------------------------------------
        pitch_stat_cols = ["stuff", "movement", "control", "p_hr"]
        pitch_weight_keys = ["stuff", "movement", "control", "p_hr"]

        pitch_rows = conn.execute("""
            SELECT c.card_title, c.stuff, c.movement, c.control, c.p_hr,
                   c.card_value, c.stamina, c.hold,
                   ps.war, ps.era, ps.ip
            FROM cards c
            INNER JOIN pitching_stats ps ON ps.card_id = c.card_id
            INNER JOIN (
                SELECT card_id, MAX(snapshot_date) as max_date
                FROM pitching_stats WHERE card_id IS NOT NULL
                GROUP BY card_id
            ) latest ON ps.card_id = latest.card_id
                    AND ps.snapshot_date = latest.max_date
            WHERE c.pitcher_role IS NOT NULL AND ps.ip >= 30
        """).fetchall()

        # Fallback: name matching for stats without card_id
        if len(pitch_rows) < 10:
            try:
                seen = {r["card_title"] for r in pitch_rows}
                extra = conn.execute("""
                    SELECT c.card_title, c.stuff, c.movement, c.control, c.p_hr,
                           c.card_value, c.stamina, c.hold,
                           ps.war, ps.era, ps.ip
                    FROM cards c
                    INNER JOIN pitching_stats ps
                        ON c.card_title LIKE '%' || ps.player_name || '%'
                    INNER JOIN (
                        SELECT player_name, MAX(snapshot_date) as max_date
                        FROM pitching_stats WHERE card_id IS NULL
                        GROUP BY player_name
                    ) latest ON ps.player_name = latest.player_name
                            AND ps.snapshot_date = latest.max_date
                    WHERE c.pitcher_role IS NOT NULL AND ps.ip >= 30
                """).fetchall()
                for r in extra:
                    if r["card_title"] not in seen:
                        pitch_rows.append(r)
                        seen.add(r["card_title"])
            except Exception:
                pass

        pitch_sample_size = len(pitch_rows)
        pitch_calibrated = dict(DEFAULT_PITCHING_WEIGHTS)
        pitch_confidence = 0.0
        pitch_r2 = 0.0

        if pitch_sample_size >= 10:
            pitch_confidence = min(1.0, pitch_sample_size / 100.0)

            stat_arrays_p = {k: [] for k in pitch_stat_cols}
            war_vals_p = []
            neg_era_vals = []  # Negated so higher = better

            for row in pitch_rows:
                war_vals_p.append(_safe_float(row["war"]))
                neg_era_vals.append(-_safe_float(row["era"]))
                for col in pitch_stat_cols:
                    stat_arrays_p[col].append(_safe_float(row[col]))

            corr_war_p = {}
            corr_era_p = {}
            for key in pitch_weight_keys:
                corr_war_p[key] = _pearson_correlation(stat_arrays_p[key], war_vals_p)
                corr_era_p[key] = _pearson_correlation(stat_arrays_p[key], neg_era_vals)

            combined_corr_p = {}
            for key in pitch_weight_keys:
                combined_corr_p[key] = (corr_war_p[key] + corr_era_p[key]) / 2.0

            positive_corrs_p = {k: max(0.0, v) for k, v in combined_corr_p.items()}
            corr_sum_p = sum(positive_corrs_p.values())

            if corr_sum_p > 0:
                default_total_p = sum(
                    DEFAULT_PITCHING_WEIGHTS.get(k, 0.0) for k in pitch_weight_keys
                )
                empirical_p = {}
                for key in pitch_weight_keys:
                    empirical_p[key] = (positive_corrs_p[key] / corr_sum_p) * default_total_p

                blend_p = pitch_confidence * 0.6
                for key in pitch_weight_keys:
                    default_w = DEFAULT_PITCHING_WEIGHTS.get(key, 0.0)
                    emp_w = empirical_p[key]
                    new_w = round(default_w * (1.0 - blend_p) + emp_w * blend_p, 2)
                    pitch_calibrated[key] = new_w

                # R-squared for pitching (predict WAR)
                predicted_war_p = []
                for i in range(pitch_sample_size):
                    pred = sum(
                        stat_arrays_p[key][i] * pitch_calibrated.get(key, 0.0)
                        for key in pitch_weight_keys
                    )
                    predicted_war_p.append(pred)
                pitch_r2 = _r_squared(war_vals_p, predicted_war_p)

                for key in pitch_weight_keys:
                    old_w = DEFAULT_PITCHING_WEIGHTS.get(key, 0.0)
                    new_w = pitch_calibrated[key]
                    if abs(new_w - old_w) >= 0.05:
                        direction = "increased" if new_w > old_w else "decreased"
                        result["changes"].append({
                            "stat": key,
                            "type": "pitching",
                            "old_weight": old_w,
                            "new_weight": new_w,
                            "reason": (
                                f"{key} {direction} from {old_w:.2f} to {new_w:.2f} "
                                f"(WAR r={corr_war_p[key]:.3f}, "
                                f"-ERA r={corr_era_p[key]:.3f}, "
                                f"n={pitch_sample_size})"
                            ),
                        })
        else:
            if pitch_sample_size > 0:
                result["message"] += (
                    f"Pitching: only {pitch_sample_size} pitchers with 30+ IP "
                    f"(need 10). Using defaults. "
                )
            else:
                result["message"] += "Pitching: no matched pitchers found. Using defaults. "

        # ---------------------------------------------------------------
        # Store results
        # ---------------------------------------------------------------
        overall_confidence = 0.0
        if bat_sample_size >= 15 and pitch_sample_size >= 10:
            overall_confidence = (bat_confidence + pitch_confidence) / 2.0
        elif bat_sample_size >= 15:
            overall_confidence = bat_confidence * 0.5
        elif pitch_sample_size >= 10:
            overall_confidence = pitch_confidence * 0.5

        result["batting_weights"] = bat_calibrated
        result["pitching_weights"] = pitch_calibrated
        result["batting_r2"] = round(bat_r2, 4)
        result["pitching_r2"] = round(pitch_r2, 4)
        result["confidence"] = round(overall_confidence, 3)

        # Persist to DB
        if bat_sample_size >= 15:
            conn.execute(
                "INSERT INTO meta_calibration "
                "(calibration_type, weights_json, r_squared, correlation, "
                "sample_size, confidence, changes_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "batting",
                    json.dumps(bat_calibrated),
                    bat_r2,
                    _pearson_correlation(
                        [_safe_float(p["war"]) for p in bat_rows],
                        [sum(
                            _safe_float(p[col]) * bat_calibrated.get(col, 0.0)
                            for col in bat_stat_cols
                        ) for p in bat_rows],
                    ) if bat_rows else 0.0,
                    bat_sample_size,
                    bat_confidence,
                    json.dumps([c for c in result["changes"] if c["type"] == "batting"]),
                ),
            )

        if pitch_sample_size >= 10:
            conn.execute(
                "INSERT INTO meta_calibration "
                "(calibration_type, weights_json, r_squared, correlation, "
                "sample_size, confidence, changes_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "pitching",
                    json.dumps(pitch_calibrated),
                    pitch_r2,
                    _pearson_correlation(
                        [_safe_float(p["war"]) for p in pitch_rows],
                        [sum(
                            _safe_float(p[col]) * pitch_calibrated.get(col, 0.0)
                            for col in pitch_stat_cols
                        ) for p in pitch_rows],
                    ) if pitch_rows else 0.0,
                    pitch_sample_size,
                    pitch_confidence,
                    json.dumps([c for c in result["changes"] if c["type"] == "pitching"]),
                ),
            )

        conn.commit()

        changes_count = len(result["changes"])
        if changes_count > 0:
            result["message"] += (
                f"Calibration complete: {changes_count} weight(s) adjusted. "
                f"Batting R2={bat_r2:.3f} (n={bat_sample_size}), "
                f"Pitching R2={pitch_r2:.3f} (n={pitch_sample_size}). "
                f"Confidence={overall_confidence:.1%}."
            )
        elif bat_sample_size >= 15 or pitch_sample_size >= 10:
            result["message"] += (
                "Calibration ran but defaults are already well-aligned with performance. "
                "No weight changes needed."
            )

        return result

    except Exception as e:
        logger.error(f"Auto-calibration error: {e}", exc_info=True)
        result["message"] = f"Calibration error: {e}"
        return result

    finally:
        if close_conn:
            conn.close()


def apply_calibrated_weights(conn=None) -> tuple[dict, dict]:
    """Load calibrated weights from DB if available, otherwise return defaults.

    Returns a tuple of (batting_weights, pitching_weights).
    """
    from app.utils.constants import DEFAULT_BATTING_WEIGHTS, DEFAULT_PITCHING_WEIGHTS

    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        _ensure_calibration_table(conn)

        # Get most recent batting calibration
        bat_row = conn.execute(
            "SELECT weights_json, confidence FROM meta_calibration "
            "WHERE calibration_type = 'batting' "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        # Get most recent pitching calibration
        pitch_row = conn.execute(
            "SELECT weights_json, confidence FROM meta_calibration "
            "WHERE calibration_type = 'pitching' "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        batting_weights = dict(DEFAULT_BATTING_WEIGHTS)
        pitching_weights = dict(DEFAULT_PITCHING_WEIGHTS)

        if bat_row and _safe_float(bat_row["confidence"]) > 0:
            try:
                stored = json.loads(bat_row["weights_json"])
                # Only override keys that exist in defaults
                for key in DEFAULT_BATTING_WEIGHTS:
                    if key in stored:
                        batting_weights[key] = stored[key]
            except (json.JSONDecodeError, TypeError):
                pass

        if pitch_row and _safe_float(pitch_row["confidence"]) > 0:
            try:
                stored = json.loads(pitch_row["weights_json"])
                for key in DEFAULT_PITCHING_WEIGHTS:
                    if key in stored:
                        pitching_weights[key] = stored[key]
            except (json.JSONDecodeError, TypeError):
                pass

        return batting_weights, pitching_weights

    except Exception as e:
        logger.error(f"Error loading calibrated weights: {e}", exc_info=True)
        return dict(DEFAULT_BATTING_WEIGHTS), dict(DEFAULT_PITCHING_WEIGHTS)

    finally:
        if close_conn:
            conn.close()


def get_calibration_history(conn=None) -> list:
    """Get history of calibration runs for tracking improvement.

    Returns a list of dicts, most recent first, with keys:
        id, calibration_type, weights (parsed), r_squared, correlation,
        sample_size, confidence, changes (parsed), created_at.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        _ensure_calibration_table(conn)

        rows = conn.execute(
            "SELECT id, calibration_type, weights_json, r_squared, correlation, "
            "sample_size, confidence, changes_json, created_at "
            "FROM meta_calibration ORDER BY created_at DESC LIMIT 50"
        ).fetchall()

        history = []
        for row in rows:
            entry = dict(row)
            # Parse JSON fields
            try:
                entry["weights"] = json.loads(entry.pop("weights_json"))
            except (json.JSONDecodeError, TypeError):
                entry["weights"] = {}
                entry.pop("weights_json", None)
            try:
                entry["changes"] = json.loads(entry.pop("changes_json"))
            except (json.JSONDecodeError, TypeError):
                entry["changes"] = []
                entry.pop("changes_json", None)
            history.append(entry)

        return history

    except Exception as e:
        logger.error(f"Error fetching calibration history: {e}", exc_info=True)
        return []

    finally:
        if close_conn:
            conn.close()
