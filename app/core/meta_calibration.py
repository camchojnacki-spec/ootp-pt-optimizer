"""Regression-based meta weight calibration using actual game stats.

Uses pure Python normal equations (no numpy/scipy) to find optimal weights
that best predict in-game performance from card ratings.
"""
import logging
import math
import json
from datetime import datetime

from app.core.database import get_connection, load_config

logger = logging.getLogger(__name__)


def _dot(a, b):
    """Dot product of two vectors (lists)."""
    return sum(x * y for x, y in zip(a, b))


def _mat_mul(A, B):
    """Multiply two matrices represented as list-of-lists."""
    rows_a = len(A)
    cols_a = len(A[0])
    cols_b = len(B[0])
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            s = 0.0
            for k in range(cols_a):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result


def _transpose(M):
    """Transpose a matrix."""
    rows = len(M)
    cols = len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def _mat_vec_mul(M, v):
    """Multiply matrix by vector."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _invert_matrix(M):
    """Invert a square matrix using Gauss-Jordan elimination. Pure Python."""
    n = len(M)
    # Augment with identity
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]

    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < 1e-12:
            return None  # Singular matrix
        aug[col], aug[max_row] = aug[max_row], aug[col]

        # Scale pivot row
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        # Eliminate column
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    # Extract inverse
    return [row[n:] for row in aug]


def _r_squared(y_actual, y_predicted):
    """Calculate R-squared."""
    n = len(y_actual)
    if n < 2:
        return 0.0
    mean_y = sum(y_actual) / n
    ss_tot = sum((y - mean_y) ** 2 for y in y_actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(y_actual, y_predicted))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def _ols_regression(X, y):
    """Ordinary Least Squares regression using normal equations.

    X: list of lists (each row is a sample, each col is a feature)
    y: list of floats (target)

    Returns: list of coefficients (one per feature), R-squared
    """
    n = len(y)
    k = len(X[0])
    if n < k + 1:
        return None, 0.0

    Xt = _transpose(X)
    XtX = _mat_mul(Xt, [[X[i][j] for j in range(k)] for i in range(n)])
    Xty = _mat_vec_mul(Xt, y)

    XtX_inv = _invert_matrix(XtX)
    if XtX_inv is None:
        return None, 0.0

    beta = _mat_vec_mul(XtX_inv, Xty)

    # Calculate R-squared
    y_pred = [_dot(X[i], beta) for i in range(n)]
    r2 = _r_squared(y, y_pred)

    return beta, r2


def _normalize_name(name):
    """Normalize player name for matching."""
    return name.strip().lower().replace(".", "").replace("'", "")


def calibrate_batting_weights(conn=None):
    """Run OLS regression: card batting ratings -> in-game OPS.

    Returns dict with:
        weights: dict of {rating_name: weight}
        r_squared: float
        sample_size: int
        error: str or None
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        # Get batting stats with sufficient AB
        bat_rows = conn.execute("""
            SELECT bs.player_name, bs.ops
            FROM batting_stats bs
            INNER JOIN (
                SELECT player_name, MAX(snapshot_date) as max_date
                FROM batting_stats GROUP BY player_name
            ) latest ON bs.player_name = latest.player_name
                    AND bs.snapshot_date = latest.max_date
            WHERE bs.ab >= 50
        """).fetchall()

        if not bat_rows:
            return {"weights": {}, "r_squared": 0.0, "sample_size": 0,
                    "error": "No batting stats with 50+ AB found."}

        # Build lookup of stats by normalized name
        stats_lookup = {}
        for row in bat_rows:
            key = _normalize_name(row["player_name"])
            stats_lookup[key] = float(row["ops"])

        # Get card ratings for matching players
        rating_cols = ["contact", "gap_power", "power", "eye", "avoid_ks", "babip"]
        card_rows = conn.execute("""
            SELECT card_title, contact, gap_power, power, eye, avoid_ks, babip
            FROM cards
            WHERE contact IS NOT NULL AND contact > 0
        """).fetchall()

        X = []
        y = []
        for card in card_rows:
            key = _normalize_name(card["card_title"])
            ops = None
            # Exact match
            if key in stats_lookup:
                ops = stats_lookup[key]
            else:
                # Partial match
                for skey, sval in stats_lookup.items():
                    if skey in key or key in skey:
                        ops = sval
                        break
            if ops is None:
                continue

            row_x = [float(card[col] or 0) for col in rating_cols]
            X.append(row_x)
            y.append(ops)

        if len(X) < 5:
            return {"weights": {}, "r_squared": 0.0, "sample_size": len(X),
                    "error": f"Only {len(X)} matched players (need at least 5)."}

        beta, r2 = _ols_regression(X, y)
        if beta is None:
            return {"weights": {}, "r_squared": 0.0, "sample_size": len(X),
                    "error": "Regression failed (singular matrix)."}

        # Normalize weights so they sum to same total as current config weights
        config = load_config()
        current_batting = config.get("batting_weights", {})
        current_total = sum(current_batting.get(col, 1.0) for col in rating_cols)

        # Handle negative coefficients: clamp to small positive value
        raw_weights = {col: max(beta[i], 0.01) for i, col in enumerate(rating_cols)}
        raw_total = sum(raw_weights.values())

        if raw_total < 1e-12:
            return {"weights": {}, "r_squared": r2, "sample_size": len(X),
                    "error": "All regression coefficients were near zero."}

        scale = current_total / raw_total
        normalized = {col: round(w * scale, 2) for col, w in raw_weights.items()}

        return {
            "weights": normalized,
            "r_squared": round(r2, 4),
            "sample_size": len(X),
            "error": None,
        }

    except Exception as e:
        logger.error(f"Batting calibration error: {e}", exc_info=True)
        return {"weights": {}, "r_squared": 0.0, "sample_size": 0,
                "error": str(e)}
    finally:
        if close_conn:
            conn.close()


def calibrate_pitching_weights(conn=None):
    """Run OLS regression: card pitching ratings -> in-game ERA+.

    Returns dict with:
        weights: dict of {rating_name: weight}
        r_squared: float
        sample_size: int
        error: str or None
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        pitch_rows = conn.execute("""
            SELECT ps.player_name, ps.era_plus
            FROM pitching_stats ps
            INNER JOIN (
                SELECT player_name, MAX(snapshot_date) as max_date
                FROM pitching_stats GROUP BY player_name
            ) latest ON ps.player_name = latest.player_name
                    AND ps.snapshot_date = latest.max_date
            WHERE ps.ip >= 30 AND ps.era_plus > 0
        """).fetchall()

        if not pitch_rows:
            return {"weights": {}, "r_squared": 0.0, "sample_size": 0,
                    "error": "No pitching stats with 30+ IP found."}

        stats_lookup = {}
        for row in pitch_rows:
            key = _normalize_name(row["player_name"])
            stats_lookup[key] = float(row["era_plus"])

        rating_cols = ["stuff", "movement", "control", "p_hr"]
        card_rows = conn.execute("""
            SELECT card_title, stuff, movement, control, p_hr
            FROM cards
            WHERE stuff IS NOT NULL AND stuff > 0
        """).fetchall()

        X = []
        y = []
        for card in card_rows:
            key = _normalize_name(card["card_title"])
            era_plus = None
            if key in stats_lookup:
                era_plus = stats_lookup[key]
            else:
                for skey, sval in stats_lookup.items():
                    if skey in key or key in skey:
                        era_plus = sval
                        break
            if era_plus is None:
                continue

            row_x = [float(card[col] or 0) for col in rating_cols]
            X.append(row_x)
            y.append(era_plus)

        if len(X) < 5:
            return {"weights": {}, "r_squared": 0.0, "sample_size": len(X),
                    "error": f"Only {len(X)} matched pitchers (need at least 5)."}

        beta, r2 = _ols_regression(X, y)
        if beta is None:
            return {"weights": {}, "r_squared": 0.0, "sample_size": len(X),
                    "error": "Regression failed (singular matrix)."}

        config = load_config()
        current_pitching = config.get("pitching_weights", {})
        current_total = sum(current_pitching.get(col, 1.0) for col in rating_cols)

        raw_weights = {col: max(beta[i], 0.01) for i, col in enumerate(rating_cols)}
        raw_total = sum(raw_weights.values())

        if raw_total < 1e-12:
            return {"weights": {}, "r_squared": r2, "sample_size": len(X),
                    "error": "All regression coefficients were near zero."}

        scale = current_total / raw_total
        normalized = {col: round(w * scale, 2) for col, w in raw_weights.items()}

        return {
            "weights": normalized,
            "r_squared": round(r2, 4),
            "sample_size": len(X),
            "error": None,
        }

    except Exception as e:
        logger.error(f"Pitching calibration error: {e}", exc_info=True)
        return {"weights": {}, "r_squared": 0.0, "sample_size": 0,
                "error": str(e)}
    finally:
        if close_conn:
            conn.close()


def get_calibration_comparison(conn=None):
    """Compare current config weights vs regression-suggested weights.

    Returns dict with:
        batting: {current: {}, suggested: {}, r_squared: float, sample_size: int}
        pitching: {current: {}, suggested: {}, r_squared: float, sample_size: int}
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        config = load_config()
        current_batting = config.get("batting_weights", {})
        current_pitching = config.get("pitching_weights", {})

        bat_result = calibrate_batting_weights(conn)
        pitch_result = calibrate_pitching_weights(conn)

        return {
            "batting": {
                "current": current_batting,
                "suggested": bat_result["weights"],
                "r_squared": bat_result["r_squared"],
                "sample_size": bat_result["sample_size"],
                "error": bat_result["error"],
            },
            "pitching": {
                "current": current_pitching,
                "suggested": pitch_result["weights"],
                "r_squared": pitch_result["r_squared"],
                "sample_size": pitch_result["sample_size"],
                "error": pitch_result["error"],
            },
        }

    except Exception as e:
        logger.error(f"Calibration comparison error: {e}", exc_info=True)
        return {
            "batting": {"current": {}, "suggested": {}, "r_squared": 0, "sample_size": 0, "error": str(e)},
            "pitching": {"current": {}, "suggested": {}, "r_squared": 0, "sample_size": 0, "error": str(e)},
        }
    finally:
        if close_conn:
            conn.close()


def auto_calibrate_if_ready(conn=None):
    """Run calibration if enough data exists (min 20 matched players).

    Stores result in ai_insights table as insight_type 'calibration'.
    Returns True if calibration ran, False otherwise.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        comparison = get_calibration_comparison(conn)

        bat_size = comparison["batting"]["sample_size"]
        pitch_size = comparison["pitching"]["sample_size"]

        if bat_size < 20 and pitch_size < 20:
            return False

        # Build content string
        lines = []
        if bat_size >= 20 and comparison["batting"]["suggested"]:
            lines.append(f"BATTING CALIBRATION (n={bat_size}, R2={comparison['batting']['r_squared']:.3f}):")
            for key in sorted(comparison["batting"]["current"].keys()):
                current = comparison["batting"]["current"].get(key, 0)
                suggested = comparison["batting"]["suggested"].get(key, 0)
                diff = suggested - current
                if abs(diff) >= 0.05:
                    lines.append(f"  {key}: {current:.2f} -> {suggested:.2f} ({diff:+.2f})")

        if pitch_size >= 20 and comparison["pitching"]["suggested"]:
            lines.append(f"PITCHING CALIBRATION (n={pitch_size}, R2={comparison['pitching']['r_squared']:.3f}):")
            for key in sorted(comparison["pitching"]["current"].keys()):
                current = comparison["pitching"]["current"].get(key, 0)
                suggested = comparison["pitching"]["suggested"].get(key, 0)
                diff = suggested - current
                if abs(diff) >= 0.05:
                    lines.append(f"  {key}: {current:.2f} -> {suggested:.2f} ({diff:+.2f})")

        if not lines:
            lines.append("Calibration ran but current weights are already close to optimal.")

        content = "\n".join(lines)

        # Store in ai_insights
        conn.execute(
            "INSERT INTO ai_insights (insight_type, content, created_at) VALUES (?, ?, ?)",
            ("calibration", content, datetime.now().isoformat()),
        )
        conn.commit()

        return True

    except Exception as e:
        logger.error(f"Auto-calibration error: {e}", exc_info=True)
        return False
    finally:
        if close_conn:
            conn.close()
