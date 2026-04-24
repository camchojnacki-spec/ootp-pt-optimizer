"""Meta Validation Engine — compare meta score predictions against actual in-game performance.

Uses batting_stats and pitching_stats tables (from CSV stat exports) to validate
whether the meta scoring system accurately predicts real in-game outcomes.
"""
import json
import logging
import math

from app.core.database import get_connection, load_config
from app.utils.constants import DEFAULT_BATTING_WEIGHTS, DEFAULT_PITCHING_WEIGHTS
# MIN_CALIBRATION_R2 is the threshold a calibration run must clear before its
# weights are written to meta_calibration. Kept in meta_scoring.py as the
# single source of truth so the load path and the write path agree.
from app.core.meta_scoring import MIN_CALIBRATION_R2

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
);
-- Covers the hot query path in meta_scoring._load_calibration_row, which
-- hits the table once per weight load with (calibration_type, MAX(created_at)).
-- Without the index, the table scan cost grows linearly with calibration
-- history (per-position rows multiply the row count) and calibration loads
-- happen on every rec regen, every roster scorer call, and every page render
-- that uses weights. Added 2026-04-15 with the meta_calibration rewrite.
CREATE INDEX IF NOT EXISTS idx_meta_calibration_type_created
    ON meta_calibration(calibration_type, created_at DESC);
"""


def _ensure_calibration_table(conn):
    """Create the meta_calibration table and its hot-path index if missing."""
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
    """Run auto-calibration — now a passthrough to ``meta_calibration.run_calibration``.

    **2026-04-15 rewrite**: the old 480-line ElasticNetCV implementation was
    replaced with a RidgeCV + NNLS + Bayesian-prior-blend calibrator living in
    ``app.core.meta_calibration``. This function is kept for backward
    compatibility with the ``7_Game_Stats.py`` UI button and any external
    callers — it simply delegates to ``run_calibration`` and packages the
    result into the legacy ``{batting_weights, pitching_weights, changes,
    message, confidence, ...}`` shape.

    The per-position calibration sub-step (``calibrate_per_position``) still
    runs here since it's a separate concern — those rows drive the per-position
    confidence chips on Buy Recs / Roster Optimizer.
    """
    from app.core.meta_calibration import run_calibration
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
        "rejected_writes": [],
    }

    try:
        _ensure_calibration_table(conn)

        # ── Delegate to the new calibrator ─────────────────────────────
        summary = run_calibration(conn)

        bat = summary.get("batting") or {}
        sp = summary.get("pitching_sp") or {}
        rp = summary.get("pitching_rp") or {}
        combined = summary.get("pitching") or {}

        # Batting result shaping
        if "error" not in bat and bat.get("weights"):
            result["batting_weights"] = bat["weights"]
            result["batting_r2"] = bat.get("r_squared", 0.0) or 0.0
            bat_confidence = bat.get("confidence", 0.0) or 0.0
            bat_sample = bat.get("sample_size", 0) or 0

            # Build a changes list from the prior (config) → blended values
            try:
                cfg_bat = load_config().get("batting_weights", DEFAULT_BATTING_WEIGHTS)
            except Exception:
                cfg_bat = DEFAULT_BATTING_WEIGHTS
            for stat, new_w in bat["weights"].items():
                old_w = cfg_bat.get(stat, DEFAULT_BATTING_WEIGHTS.get(stat, 0.0))
                if abs(new_w - old_w) >= 0.05:
                    direction = "increased" if new_w > old_w else "decreased"
                    result["changes"].append({
                        "stat": stat,
                        "type": "batting",
                        "old_weight": round(float(old_w), 4),
                        "new_weight": round(float(new_w), 4),
                        "reason": (
                            f"{stat} {direction} from {old_w:.2f} to {new_w:.2f} "
                            f"(Ridge+NNLS, n={bat_sample}, "
                            f"R²={result['batting_r2']:.3f})"
                        ),
                    })
        else:
            bat_confidence, bat_sample = 0.0, 0
            if bat.get("error"):
                result["message"] += f"Batting: {bat['error']} "

        # Pitching result shaping — use the combined row for legacy weights,
        # but flag SP and RP separately in the message so the user sees both.
        if "error" not in combined and combined.get("weights"):
            result["pitching_weights"] = combined["weights"]
            result["pitching_r2"] = combined.get("r_squared", 0.0) or 0.0
            pitch_confidence = combined.get("confidence", 0.0) or 0.0
            pitch_sample = combined.get("sample_size", 0) or 0

            try:
                cfg_pit = load_config().get("pitching_weights", DEFAULT_PITCHING_WEIGHTS)
            except Exception:
                cfg_pit = DEFAULT_PITCHING_WEIGHTS
            for stat, new_w in combined["weights"].items():
                old_w = cfg_pit.get(stat, DEFAULT_PITCHING_WEIGHTS.get(stat, 0.0))
                threshold = 0.01 if "_x_" in stat else 0.05
                if abs(new_w - old_w) >= threshold:
                    direction = "increased" if new_w > old_w else "decreased"
                    result["changes"].append({
                        "stat": stat,
                        "type": "pitching",
                        "old_weight": round(float(old_w), 4),
                        "new_weight": round(float(new_w), 4),
                        "reason": (
                            f"{stat} {direction} from {old_w:.3f} to {new_w:.3f} "
                            f"(Ridge+NNLS combined SP+RP, n={pitch_sample}, "
                            f"R²={result['pitching_r2']:.3f})"
                        ),
                    })
        else:
            pitch_confidence, pitch_sample = 0.0, 0
            if combined.get("error"):
                result["message"] += f"Pitching (combined): {combined['error']} "

        # Report SP/RP split R² in the message even when below gate
        if "error" not in sp:
            result["message"] += (
                f"SP R²={sp.get('r_squared', 0):.3f} (n={sp.get('sample_size', 0)}). "
            )
        if "error" not in rp:
            result["message"] += (
                f"RP R²={rp.get('r_squared', 0):.3f} (n={rp.get('sample_size', 0)}). "
            )

        # Overall confidence: average of the two sub-confidences, weighted
        if bat_sample >= 15 and pitch_sample >= 10:
            overall_confidence = (bat_confidence + pitch_confidence) / 2.0
        elif bat_sample >= 15:
            overall_confidence = bat_confidence * 0.5
        elif pitch_sample >= 10:
            overall_confidence = pitch_confidence * 0.5
        else:
            overall_confidence = 0.0
        result["confidence"] = round(overall_confidence, 3)

        conn.commit()

        # ── Per-position calibration (Tier-2 #8) ──
        # Run alongside the global calibration so the per-position chips on
        # Buy Recs / Roster Optimizer always reflect the latest stats. Keep
        # any failure non-fatal — the global calibration is the load-bearing
        # path and shouldn't break if a position bucket has weird data.
        try:
            pos_result = calibrate_per_position(conn)
            result["per_position"] = pos_result.get("by_position", {})
            if pos_result.get("message"):
                result["message"] += " " + pos_result["message"]
        except Exception as e:
            logger.warning(f"Per-position calibration sub-step failed: {e}")

        changes_count = len(result["changes"])
        if changes_count > 0:
            result["message"] += (
                f"Calibration complete: {changes_count} weight(s) adjusted. "
                f"Batting R²={result['batting_r2']:.3f} (n={bat_sample}), "
                f"Pitching R²={result['pitching_r2']:.3f} (n={pitch_sample}). "
                f"Confidence={overall_confidence:.1%}."
            )
        elif bat_sample >= 15 or pitch_sample >= 10:
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


def calibrate_per_position(conn=None) -> dict:
    """Compute per-position correlation/R\u00b2 between meta_score and observed WAR.

    The global ``auto_calibrate_weights`` answers "how well does the meta
    formula predict performance overall?" but it can't tell the user whether
    e.g. catcher meta is much less reliable than shortstop meta. Per-position
    calibration fills that gap so the UI can render position-specific trust
    chips on Buy Recs / Roster Optimizer / Sell Recs.

    For each position (batting and pitching role), this:
      1. Loads the latest stat snapshot per player at that position.
      2. Computes Pearson r and R\u00b2 between meta_score and a perf metric:
         - Batters: WAR scaled to /600 PA so part-time players don't anchor low
         - Pitchers: WAR scaled to /200 IP for the same reason
      3. Persists one row per position into ``meta_calibration`` with
         ``calibration_type = 'pos:<POSITION>'``.

    Returns a dict::
        {
          "by_position": {
            "C":  {"correlation": 0.41, "r_squared": 0.17, "sample_size": 8},
            "SS": {"correlation": 0.62, "r_squared": 0.38, "sample_size": 11},
            ...
          },
          "message": "...",
        }

    The per-position rows live alongside the existing ``batting`` / ``pitching``
    rows; they're discoverable via ``get_meta_confidence_by_position``.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    out: dict = {"by_position": {}, "message": ""}

    try:
        _ensure_calibration_table(conn)

        # ── Batters ──
        # Pull every batter with a current meta + a perf row, latest snapshot
        # per player. Keep PA gating modest (>= 30) so early-season catchers
        # and platoon guys still contribute — the per-position table is meant
        # to be descriptive, not predictive.
        bat_rows = conn.execute("""
            SELECT c.position_name as pos, c.meta_score_batting as meta,
                   bs.war, bs.pa
            FROM cards c
            INNER JOIN batting_stats bs ON bs.card_id = c.card_id
            INNER JOIN (
                SELECT card_id, MAX(snapshot_date) as max_date
                FROM batting_stats WHERE card_id IS NOT NULL
                GROUP BY card_id
            ) latest ON bs.card_id = latest.card_id
                    AND bs.snapshot_date = latest.max_date
            WHERE c.position_name IS NOT NULL
              AND c.meta_score_batting > 0
              AND bs.pa >= 30
        """).fetchall()

        # Bucket by position
        pos_buckets: dict[str, list[tuple[float, float]]] = {}
        for r in bat_rows:
            pos = r["pos"]
            war = _safe_float(r["war"])
            pa = _safe_float(r["pa"])
            meta = _safe_float(r["meta"])
            if pa <= 0 or meta <= 0:
                continue
            war600 = war * 600.0 / pa
            pos_buckets.setdefault(pos, []).append((meta, war600))

        # ── Pitchers ──
        pit_rows = conn.execute("""
            SELECT c.pitcher_role_name as pos, c.meta_score_pitching as meta,
                   ps.war, ps.ip
            FROM cards c
            INNER JOIN pitching_stats ps ON ps.card_id = c.card_id
            INNER JOIN (
                SELECT card_id, MAX(snapshot_date) as max_date
                FROM pitching_stats WHERE card_id IS NOT NULL
                GROUP BY card_id
            ) latest ON ps.card_id = latest.card_id
                    AND ps.snapshot_date = latest.max_date
            WHERE c.pitcher_role_name IS NOT NULL
              AND c.meta_score_pitching > 0
              AND ps.ip >= 10
        """).fetchall()

        for r in pit_rows:
            pos = r["pos"]
            war = _safe_float(r["war"])
            ip = _safe_float(r["ip"])
            meta = _safe_float(r["meta"])
            if ip <= 0 or meta <= 0:
                continue
            war200 = war * 200.0 / ip
            pos_buckets.setdefault(pos, []).append((meta, war200))

        # Compute per-position stats and persist
        persisted = 0
        for pos, pairs in pos_buckets.items():
            n = len(pairs)
            if n < 5:
                # Too few samples to trust the correlation — still record so
                # the UI can show "n=3 — insufficient" instead of going dark.
                out["by_position"][pos] = {
                    "correlation": 0.0,
                    "r_squared": 0.0,
                    "sample_size": n,
                }
                continue
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            corr = _pearson_correlation(xs, ys)
            # R\u00b2 here means "squared correlation" — i.e. fraction of
            # variance in WAR explainable by a linear meta map. We deliberately
            # do NOT use the residual-based _r_squared() helper because we have
            # no fitted model in this lightweight per-position pass.
            r2 = corr * corr
            out["by_position"][pos] = {
                "correlation": round(corr, 3),
                "r_squared": round(r2, 3),
                "sample_size": n,
            }
            try:
                conn.execute(
                    "INSERT INTO meta_calibration "
                    "(calibration_type, weights_json, r_squared, correlation, "
                    "sample_size, confidence, changes_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"pos:{pos}",
                        "{}",  # no per-position weights stored — only stats
                        r2,
                        corr,
                        n,
                        min(1.0, n / 50.0),
                        "[]",
                    ),
                )
                persisted += 1
            except Exception as e:
                logger.warning(f"Failed to persist per-position calibration for {pos}: {e}")

        conn.commit()
        out["message"] = (
            f"Per-position calibration: {persisted} positions stored "
            f"({len(out['by_position'])} buckets total)."
        )
        return out

    except Exception as e:
        logger.error(f"Per-position calibration error: {e}", exc_info=True)
        out["message"] = f"Error: {e}"
        return out

    finally:
        if close_conn:
            conn.close()


def get_meta_confidence_by_position(position: str, conn=None) -> dict:
    """Return the latest per-position confidence chip for ``position``.

    Same shape as ``get_meta_confidence`` but reads ``calibration_type =
    'pos:<position>'`` rows. Falls back to the broader batting/pitching chip
    when the position-specific row is missing or has too few samples.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    BATTING_POSITIONS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"}

    try:
        _ensure_calibration_table(conn)
        row = conn.execute(
            "SELECT correlation, r_squared, sample_size "
            "FROM meta_calibration WHERE calibration_type = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (f"pos:{position}",),
        ).fetchone()

        if not row or int(row["sample_size"] or 0) < 5:
            # Fallback to broader chip — better than going dark
            fallback_cat = "batting" if position in BATTING_POSITIONS else "pitching"
            base = get_meta_confidence(fallback_cat, conn)
            base["category"] = position
            base["fallback"] = True
            return base

        corr = _safe_float(row["correlation"])
        r2 = _safe_float(row["r_squared"])
        n = int(row["sample_size"])
        abs_corr = abs(corr)

        if n < 5:
            level, label_prefix, emoji, color = (
                "none", "Insufficient data", "\u26aa", "#555555",
            )
        elif abs_corr >= 0.5:
            level, label_prefix, emoji, color = (
                "high", "High", "\U0001f7e2", "#2E7D32",
            )
        elif abs_corr >= 0.3:
            level, label_prefix, emoji, color = (
                "medium", "Medium", "\U0001f7e1", "#F9A825",
            )
        elif abs_corr >= 0.1:
            level, label_prefix, emoji, color = (
                "low", "Low", "\U0001f7e0", "#EF6C00",
            )
        else:
            level, label_prefix, emoji, color = (
                "low", "Very Low", "\U0001f534", "#C62828",
            )

        return {
            "category": position,
            "correlation": round(corr, 3),
            "r_squared": round(r2, 3),
            "sample_size": n,
            "level": level,
            "label": f"{label_prefix} (r={corr:.2f}, n={n})",
            "emoji": emoji,
            "color": color,
            "fallback": False,
        }
    except Exception as e:
        logger.error(f"get_meta_confidence_by_position failed for {position}: {e}", exc_info=True)
        return {
            "category": position,
            "correlation": 0.0,
            "r_squared": 0.0,
            "sample_size": 0,
            "level": "none",
            "label": "Error",
            "emoji": "\u26aa",
            "color": "#555555",
            "fallback": True,
        }
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


def get_meta_confidence(category: str = "batting", conn=None) -> dict:
    """Return the latest observed correlation between meta_score and real
    in-game performance for a category ("batting" or "pitching"), bucketed
    into a coarse label the UI can show as a chip.

    This reads the newest row from `meta_calibration` for the category. The
    `correlation` field on that table is the Pearson r from the last full
    calibration run (see `calibrate_meta_weights`). The chip gives users a
    one-glance answer to "can I trust the meta sort on this page?" — the
    review made this a P0 ask: every page that shows a meta-ordered list is
    implicitly claiming the ordering is useful, and without the chip the user
    has no way to know whether that claim holds.

    Returns a dict with:
        category:   passed-through ("batting" / "pitching")
        correlation: float  (0.0 if no calibration has been run)
        r_squared:   float
        sample_size: int
        level:       "high" / "medium" / "low" / "none"
        label:       short text e.g. "High (r=0.52)" / "No data"
        emoji:       single char visual marker
        color:       hex color for the chip background

    Thresholds are deliberately generous — the review calls out that even
    r=0.3 is more useful than nothing, and r=0.5+ is the "trust it" zone.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    default = {
        "category": category,
        "correlation": 0.0,
        "r_squared": 0.0,
        "sample_size": 0,
        "level": "none",
        "label": "No calibration yet",
        "emoji": "\u26aa",
        "color": "#555555",
    }

    try:
        _ensure_calibration_table(conn)
        row = conn.execute(
            "SELECT correlation, r_squared, sample_size "
            "FROM meta_calibration WHERE calibration_type = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (category,),
        ).fetchone()
        if not row:
            return default

        corr = _safe_float(row["correlation"])
        r2 = _safe_float(row["r_squared"])
        n = int(row["sample_size"] or 0)

        # Treat absolute correlation — a strongly negative correlation is
        # still informative, just inverted. In practice correlation here is
        # nonnegative because the formula is sign-aligned, but guard for it.
        abs_corr = abs(corr)
        if n < 20:
            level, label_prefix, emoji, color = (
                "none", "Insufficient data", "\u26aa", "#555555",
            )
        elif abs_corr >= 0.5:
            level, label_prefix, emoji, color = (
                "high", "High", "\U0001f7e2", "#2E7D32",
            )
        elif abs_corr >= 0.3:
            level, label_prefix, emoji, color = (
                "medium", "Medium", "\U0001f7e1", "#F9A825",
            )
        elif abs_corr >= 0.1:
            level, label_prefix, emoji, color = (
                "low", "Low", "\U0001f7e0", "#EF6C00",
            )
        else:
            level, label_prefix, emoji, color = (
                "low", "Very Low", "\U0001f534", "#C62828",
            )

        label = f"{label_prefix} (r={corr:.2f})" if level != "none" else label_prefix

        return {
            "category": category,
            "correlation": round(corr, 3),
            "r_squared": round(r2, 3),
            "sample_size": n,
            "level": level,
            "label": label,
            "emoji": emoji,
            "color": color,
        }

    except Exception as e:
        logger.error(f"get_meta_confidence failed: {e}", exc_info=True)
        return default

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
