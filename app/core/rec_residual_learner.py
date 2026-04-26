"""Per-rec residual learner — shadow-mode factor calibration (UAT 2026-04-25 Stream C).

When the optimizer makes a recommendation it stamps a ``factor_snapshot_json``
on the row in ``recommendation_log`` describing exactly which factor
contributions (contact +X, perf overlay +Y, etc.) drove its projected delta.
After the player accumulates post-recommendation game logs we can ask: did
the player actually outperform/underperform the projection, and which
factors systematically over- or under-predicted? That is a per-factor
*residual* signal — orthogonal to the league-wide WAR-vs-rating regression
that ``meta_calibration.py`` runs.

This module mines that signal in shadow mode. It produces:
  - per-factor avg-residual-per-unit (with std-error and direction tag),
  - pairwise interaction warnings where one factor's predictive power
    collapses or amplifies in the presence of another,
  - diagnostics on sample size and gating.

The output is persisted to ``rec_residual_calibration`` and surfaced in the
Roster Optimizer UI for inspection. It is **not** fed back into calibration
weights — adoption is a separate decision once the sample size and signal
quality clear a hand-set bar. ``shadow_mode`` flags every row so the
calibrator and any downstream consumer can tell.

Public API:
    analyze_rec_residuals(league_id, side, window_days, min_observations, conn) -> dict
    get_latest_residual_calibration(league_id, side, conn) -> dict | None
    run_residual_analysis_all(conn) -> dict
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional

import numpy as np

from app.core.backtest_harness import backtest_player_pre_post
from app.core.database import get_connection, load_config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Tunables
# ══════════════════════════════════════════════════════════════════════════

# Below this slope magnitude (in standardized residual units per factor
# point), a factor is reported as 'calibrated' rather than over/under-predicted.
# 0.005 was picked so a 60-point factor swing has to move the standardized
# residual by ≥0.3σ before we call it directional — anything smaller is
# noise at our typical sample sizes (n=30-100).
SLOPE_DIRECTIONAL_THRESHOLD = 0.005

# Interaction mining: bucket recs into 2×2 by median split on each pair of
# top factors. A pair is flagged when the spread between cell means exceeds
# this many residual std-deviations and every cell has ≥ 5 observations.
INTERACTION_SPREAD_SIGMA = 0.6
INTERACTION_MIN_CELL_N = 5

# Top-N factors (by abs slope) considered for pairwise interaction mining.
INTERACTION_TOP_N = 5

# Bootstrap iterations for slope std-error estimation. 200 is the balance
# point: fewer becomes noisy at our small n, more is wasted compute when
# the slope itself is already small.
BOOTSTRAP_ITERS = 200


# ══════════════════════════════════════════════════════════════════════════
# Public entry points
# ══════════════════════════════════════════════════════════════════════════

def analyze_rec_residuals(league_id: Optional[str] = None,
                          side: str = 'batting',
                          window_days: int = 14,
                          min_observations: int = 20,
                          conn=None) -> dict:
    """Pull closed-loop recs with factor snapshots, regress residuals, persist.

    For each rec where (a) ``factor_snapshot_json`` is populated, (b) verdict
    is not 'pending', and (c) the post-anchor sample is sufficient, we
    compute::

        residual = z(observed_delta) - z(projected_delta)

    where ``observed_delta`` is the wOBA delta (batting) or sign-flipped ERA
    delta (pitching) returned by ``backtest_player_pre_post`` and
    ``projected_delta`` is the rec's stored ``expected_delta``.

    The standardization (z-score within the cohort) is the cleanest way to
    handle the scale mismatch between the projected delta (meta-points,
    typically ~5) and the observed delta (wOBA tenths, typically ~0.020)
    without overfitting a conversion factor. Residuals are unit-free
    standard-deviation deviations — directly interpretable as "this factor
    over-predicts by X σ per rating point on average".

    Per-factor regression: for each factor key in the union of all snapshot
    component/bonus/penalty keys, regress residual on that factor's points
    contribution (univariate, weighted equally). The slope is the
    avg-residual-per-unit. Std error is a 200-iteration bootstrap.

    Interaction mining: take the top-N factors by ``abs(slope)``, bucket the
    cohort into 2×2 by median splits on each factor pair, and flag any
    pair whose cell-mean spread exceeds ``INTERACTION_SPREAD_SIGMA`` × the
    cohort residual std-dev (with ≥``INTERACTION_MIN_CELL_N`` per cell).

    Sample size gate: if fewer than ``min_observations`` recs have both a
    factor snapshot AND an observable outcome, the call returns
    diagnostics-only output (empty ``factor_calibration``) and persists a
    row anyway so the UI can render "insufficient data" cleanly.

    Returns the full output dict (the same shape that gets persisted).
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        if side not in ('batting', 'pitching'):
            raise ValueError(f"side must be 'batting' or 'pitching', got {side!r}")

        league_id = league_id or _current_league_id()

        # 1) Pull candidate recs.
        recs = _fetch_candidate_recs(conn, league_id, side)
        n_examined = len(recs)
        n_with_factors = sum(1 for r in recs if r.get('factor_snapshot_json'))

        # 2) For each rec with both a snapshot and a projected_delta, compute
        # observed delta via backtest. Skip silently when the post-anchor
        # window is empty — that's just lack of game data, not a bug.
        observations = []
        for r in recs:
            snap_raw = r.get('factor_snapshot_json')
            proj = r.get('expected_delta')
            card_id = r.get('player_card_id')
            anchor = r.get('created_at')
            if not snap_raw or proj is None or not card_id or not anchor:
                continue
            try:
                snapshot = json.loads(snap_raw)
                if not isinstance(snapshot, dict):
                    continue
            except (TypeError, json.JSONDecodeError):
                continue

            try:
                bt = backtest_player_pre_post(
                    card_id, anchor,
                    league_id=league_id, window_days=window_days, conn=conn,
                )
            except Exception as bt_err:
                logger.debug("backtest failed for rec %s: %s", r.get('id'), bt_err)
                continue

            obs = _extract_observed_delta(bt, side)
            if obs is None:
                continue

            observations.append({
                'rec_id': r.get('id'),
                'snapshot': _flatten_snapshot(snapshot),
                'projected_delta': float(proj),
                'observed_delta': float(obs),
            })

        n_with_outcomes = len(observations)

        diagnostics = {
            'n_recs_examined': n_examined,
            'n_recs_with_factors': n_with_factors,
            'n_recs_with_outcomes': n_with_outcomes,
            'min_observations_threshold': min_observations,
            'shadow_mode': True,
            'ran_at': datetime.now().isoformat(),
            'window_days': window_days,
            'side': side,
            'league_id': league_id,
        }

        # 3) Sample size gate.
        if n_with_outcomes < min_observations:
            output = {
                'factor_calibration': {},
                'interaction_warnings': [],
                'diagnostics': diagnostics,
                'low_sample': True,
            }
            _persist(conn, league_id, side, n_with_outcomes, output)
            return output

        # 4) Standardize observed and projected within the cohort, residuals
        # are the difference.
        residuals, _projected_z, _observed_z = _residuals_from_obs(observations)

        # 5) Per-factor univariate slope + bootstrap std-err.
        factor_calibration = _per_factor_slopes(observations, residuals)

        # 6) Interaction warnings (top-N factor pairs).
        interaction_warnings = _mine_interactions(observations, residuals, factor_calibration)

        output = {
            'factor_calibration': factor_calibration,
            'interaction_warnings': interaction_warnings,
            'diagnostics': diagnostics,
            'low_sample': False,
        }
        _persist(conn, league_id, side, n_with_outcomes, output)
        return output
    except Exception as e:
        logger.error("analyze_rec_residuals failed: %s", e, exc_info=True)
        return {
            'factor_calibration': {},
            'interaction_warnings': [],
            'diagnostics': {
                'error': str(e),
                'shadow_mode': True,
                'ran_at': datetime.now().isoformat(),
                'side': side,
                'league_id': league_id,
            },
            'low_sample': True,
        }
    finally:
        if close_conn:
            conn.close()


def get_latest_residual_calibration(league_id: Optional[str] = None,
                                     side: Optional[str] = None,
                                     conn=None) -> Optional[dict]:
    """Return the most recent persisted run as a dict, or ``None`` if no runs.

    JSON blobs are parsed into nested dicts/lists. ``shadow_mode`` is
    coerced to bool. Caller can filter by league + side; either filter
    falling through to ``None`` means "any".
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        clauses = []
        args: list = []
        if league_id is not None:
            clauses.append("league_id = ?")
            args.append(league_id)
        if side is not None:
            clauses.append("side = ?")
            args.append(side)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT id, created_at, league_id, side, n_observations,
                   factor_calibration_json, interaction_warnings_json,
                   diagnostics_json, shadow_mode
            FROM rec_residual_calibration
            {where}
            ORDER BY created_at DESC
            LIMIT 1
        """
        try:
            row = conn.execute(sql, args).fetchone()
        except sqlite3.Error as e:
            logger.warning("get_latest_residual_calibration query failed: %s", e)
            return None
        if not row:
            return None

        d = dict(row)
        for k in ('factor_calibration_json', 'interaction_warnings_json', 'diagnostics_json'):
            raw = d.pop(k, None)
            target_key = k[:-len('_json')]
            try:
                d[target_key] = json.loads(raw) if raw else None
            except (TypeError, json.JSONDecodeError):
                d[target_key] = None
        d['shadow_mode'] = bool(d.get('shadow_mode'))
        return d
    finally:
        if close_conn:
            conn.close()


def run_residual_analysis_all(conn=None) -> dict:
    """Run residual analysis for batting + pitching at the active league.

    Persists each side's row independently. Returns a dict shaped::

        {'league_id': str, 'batting': {...}, 'pitching': {...},
         'started_at': iso, 'finished_at': iso}

    Failures in either side are caught and reported under that side's key
    so a batting-side bug can't break the pitching run.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        league_id = _current_league_id()
        out = {
            'league_id': league_id,
            'started_at': datetime.now().isoformat(),
        }
        for side in ('batting', 'pitching'):
            try:
                out[side] = analyze_rec_residuals(
                    league_id=league_id, side=side, conn=conn,
                )
            except Exception as e:
                logger.error("residual analysis %s failed: %s", side, e, exc_info=True)
                out[side] = {
                    'factor_calibration': {},
                    'interaction_warnings': [],
                    'diagnostics': {'error': str(e), 'side': side},
                    'low_sample': True,
                }
        out['finished_at'] = datetime.now().isoformat()
        return out
    finally:
        if close_conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════
# Internals
# ══════════════════════════════════════════════════════════════════════════

def _current_league_id() -> str:
    """Read ``active_league`` from config.yaml, defaulting to ``lb124``.

    Mirrors the helper in ``meta_calibration.py`` — duplicated here to keep
    this module independent of the calibrator's import order.
    """
    try:
        return load_config().get('active_league', 'lb124')
    except Exception:
        return 'lb124'


def _fetch_candidate_recs(conn, league_id: str, side: str) -> list[dict]:
    """Pull closed-loop recs with the right side/league.

    Both ``side`` and ``league_id`` filters are applied loosely — older rows
    may have NULL side (Stream A backfill hasn't reached them) or different
    league strings, so we fall back to "side IS NULL OR side = ?" rather
    than dropping all the un-tagged history. Sample integrity is preserved
    because rows without a populated factor snapshot are dropped downstream.
    """
    sql = """
        SELECT id, created_at, league_id, side, source, rec_type, pos,
               player_name, player_card_id, expected_delta,
               factor_snapshot_json, projected_meta, projected_war,
               verdict, action_type, war_before, war_after,
               meta_before, meta_after
        FROM recommendation_log
        WHERE verdict IS NOT NULL
          AND verdict != 'pending'
          AND factor_snapshot_json IS NOT NULL
          AND (side IS NULL OR side = ?)
          AND (league_id IS NULL OR league_id = ?)
        ORDER BY created_at DESC
        LIMIT 2000
    """
    try:
        rows = conn.execute(sql, (side, league_id)).fetchall()
    except sqlite3.Error as e:
        # Likely missing columns on a very stale schema — degrade to empty
        # rather than crashing the calibration run.
        logger.warning("rec fetch failed (schema drift?): %s", e)
        return []
    return [dict(r) for r in rows]


def _flatten_snapshot(snap: dict) -> dict[str, float]:
    """Coerce a factor snapshot into a flat ``{factor_name: points}`` dict.

    Stream A's snapshot shape isn't fully nailed down yet — some recs may
    write a flat dict, others may write ``{'components': {...},
    'bonuses': {...}, 'penalties': {...}}`` with nested floats. We accept
    both. Non-numeric values are dropped silently. Penalty values are
    preserved at their original sign (negatives are penalties).
    """
    out: dict[str, float] = {}
    for k, v in snap.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                _val = _coerce_float(sub_v)
                if _val is not None:
                    out[sub_k] = _val
        else:
            _val = _coerce_float(v)
            if _val is not None:
                out[k] = _val
    return out


def _coerce_float(v) -> Optional[float]:
    """Best-effort float coercion. Returns None on failure (incl. None, str)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (ValueError, TypeError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _extract_observed_delta(bt: dict, side: str) -> Optional[float]:
    """Pull the observed delta from a backtest result.

    Batting: ``bt['delta_woba']`` (if Stream B has shipped its batting
    branch) or fall back to a meta-side delta computed from ``meta_after -
    meta_before`` if recorded on the rec itself. Pitching: ``bt['delta_era']``
    flipped so positive = improvement (lower ERA is better).

    Returns None when neither signal is available — caller skips the rec.
    """
    if not isinstance(bt, dict):
        return None
    if 'error' in bt:
        return None

    if side == 'batting':
        woba = bt.get('delta_woba')
        if woba is not None:
            try:
                return float(woba)
            except (ValueError, TypeError):
                return None
        # Fallback: some batting-rec verdicts have already cached
        # observed wOBA-like deltas in alternate keys. None of these
        # are guaranteed to be present pre-Stream-B; absence is fine.
        for alt_key in ('delta_ops', 'delta_wrcplus'):
            v = bt.get(alt_key)
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return None

    # pitching: lower ERA = better, so flip sign so positive = improvement.
    era = bt.get('delta_era')
    if era is None:
        # Fallback to FIP if Stream B exposes it later.
        era = bt.get('delta_fip')
    if era is None:
        return None
    try:
        return -float(era)
    except (ValueError, TypeError):
        return None


def _residuals_from_obs(observations: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize observed + projected within the cohort, residual = z_obs - z_proj.

    Returning each side's z-vector alongside the residual lets the interaction
    miner reuse them without recomputing.
    """
    obs = np.array([o['observed_delta'] for o in observations], dtype=float)
    proj = np.array([o['projected_delta'] for o in observations], dtype=float)

    obs_z = _zscore(obs)
    proj_z = _zscore(proj)
    return (obs_z - proj_z), proj_z, obs_z


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score with safe std (returns zeros if std≈0 to avoid div-by-zero)."""
    if arr.size == 0:
        return arr
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _per_factor_slopes(observations: list[dict],
                        residuals: np.ndarray) -> dict[str, dict]:
    """Run univariate residual ~ factor regressions for every factor key.

    A factor key gets included only if at least 5 observations have a
    non-zero contribution for it — otherwise the slope is dominated by
    a handful of points and the bootstrap is meaningless.

    Direction tags:
        slope < -SLOPE_DIRECTIONAL_THRESHOLD → 'over_predicted'
            (more of this factor → residual goes negative → observed
             underperformed projection)
        slope > +SLOPE_DIRECTIONAL_THRESHOLD → 'under_predicted'
        else → 'calibrated'
    """
    if len(observations) == 0 or residuals.size == 0:
        return {}

    # Union of all factor keys across all snapshots.
    all_keys = set()
    for o in observations:
        all_keys.update(o.get('snapshot', {}).keys())

    out: dict[str, dict] = {}
    rng = np.random.default_rng(seed=20260425)
    n = len(observations)

    for key in sorted(all_keys):
        x = np.array(
            [float(o['snapshot'].get(key) or 0.0) for o in observations],
            dtype=float,
        )

        # Drop factors that are essentially constant across the cohort —
        # a slope is undefined and the bootstrap will spin its wheels.
        nonzero_n = int(np.sum(np.abs(x) > 1e-9))
        if nonzero_n < 5 or float(x.std()) < 1e-9:
            continue

        try:
            slope, _intercept = np.polyfit(x, residuals, 1)
        except (np.linalg.LinAlgError, ValueError):
            continue
        slope = float(slope)

        # Bootstrap std-error: resample (x, residual) pairs with replacement,
        # refit, take stdev of the slope distribution. 200 iters is plenty
        # for a one-number summary — running more buys precision the n=30-100
        # cohort can't actually deliver.
        slopes = np.empty(BOOTSTRAP_ITERS, dtype=float)
        slopes_filled = 0
        for i in range(BOOTSTRAP_ITERS):
            idx = rng.integers(0, n, size=n)
            xs = x[idx]
            rs = residuals[idx]
            if float(xs.std()) < 1e-9:
                continue
            try:
                s, _ = np.polyfit(xs, rs, 1)
            except (np.linalg.LinAlgError, ValueError):
                continue
            slopes[slopes_filled] = float(s)
            slopes_filled += 1
        std_err = float(np.std(slopes[:slopes_filled])) if slopes_filled > 5 else None

        if slope < -SLOPE_DIRECTIONAL_THRESHOLD:
            direction = 'over_predicted'
        elif slope > SLOPE_DIRECTIONAL_THRESHOLD:
            direction = 'under_predicted'
        else:
            direction = 'calibrated'

        out[key] = {
            'avg_residual_per_unit': round(slope, 6),
            'n': nonzero_n,
            'std_err': round(std_err, 6) if std_err is not None else None,
            'direction': direction,
        }
    return out


def _mine_interactions(observations: list[dict],
                       residuals: np.ndarray,
                       factor_calibration: dict[str, dict]) -> list[dict]:
    """Find factor pairs where one's predictive power flips on the other's value.

    Take the top-N factors by ``abs(slope)``, then for every unordered pair
    bucket the cohort into 2×2 by median split on each factor's points. A
    pair is flagged when the spread between cell means exceeds
    ``INTERACTION_SPREAD_SIGMA`` × cohort residual std-dev AND every cell
    has at least ``INTERACTION_MIN_CELL_N`` observations.

    Reports the worst-offending split (max |residual|) along with its
    "comparison split" (the same primary high but flipped secondary), so
    the warning text reads like "power's predictive power collapses when
    avoid_ks is low".
    """
    n = len(observations)
    if n < 2 * INTERACTION_MIN_CELL_N:
        return []

    # Rank factors by |slope|; only consider ones with a real slope.
    ranked = sorted(
        ((k, v) for k, v in factor_calibration.items() if v.get('avg_residual_per_unit')),
        key=lambda kv: abs(kv[1]['avg_residual_per_unit'] or 0.0),
        reverse=True,
    )[:INTERACTION_TOP_N]
    if len(ranked) < 2:
        return []

    cohort_std = float(np.std(residuals)) if residuals.size > 0 else 1.0
    if cohort_std < 1e-9:
        cohort_std = 1.0
    threshold = INTERACTION_SPREAD_SIGMA * cohort_std

    # Pre-compute factor vectors + their medians.
    factor_vecs: dict[str, np.ndarray] = {}
    factor_medians: dict[str, float] = {}
    for k, _ in ranked:
        v = np.array(
            [float(o['snapshot'].get(k) or 0.0) for o in observations],
            dtype=float,
        )
        factor_vecs[k] = v
        factor_medians[k] = float(np.median(v))

    warnings: list[dict] = []
    keys = [k for k, _ in ranked]
    for i, fa in enumerate(keys):
        for fb in keys[i + 1:]:
            xa = factor_vecs[fa]
            xb = factor_vecs[fb]
            ma = factor_medians[fa]
            mb = factor_medians[fb]

            # 2×2 cells: (a_high|low, b_high|low). Use strict > median so
            # ties don't all fall into one bucket.
            cells: dict[tuple[bool, bool], np.ndarray] = {}
            for hi_a in (True, False):
                for hi_b in (True, False):
                    mask = ((xa > ma) == hi_a) & ((xb > mb) == hi_b)
                    cells[(hi_a, hi_b)] = residuals[mask]

            # Need a minimum population in every cell — otherwise the
            # interaction signal is unstable.
            if any(c.size < INTERACTION_MIN_CELL_N for c in cells.values()):
                continue

            cell_means = {k: float(c.mean()) for k, c in cells.items()}
            spread = max(cell_means.values()) - min(cell_means.values())
            if spread < threshold:
                continue

            # Find the cell with max |mean| residual and its same-primary,
            # flipped-secondary comparison cell. The comparison provides
            # the "predictive power collapses when X is low" framing.
            worst_key = max(cell_means.items(), key=lambda kv: abs(kv[1]))[0]
            comp_key = (worst_key[0], not worst_key[1])
            worst_cell = cells[worst_key]
            comp_cell = cells[comp_key]

            split_str = (
                f"{fa}_{'high' if worst_key[0] else 'low'} × "
                f"{fb}_{'high' if worst_key[1] else 'low'}"
            )
            comp_str = (
                f"{fa}_{'high' if comp_key[0] else 'low'} × "
                f"{fb}_{'high' if comp_key[1] else 'low'}"
            )
            interp = _interaction_interpretation(
                fa, fb, worst_key, cell_means[worst_key], cell_means[comp_key],
                int(worst_cell.size),
            )
            warnings.append({
                'factors': [fa, fb],
                'split': split_str,
                'n': int(worst_cell.size),
                'avg_residual': round(cell_means[worst_key], 4),
                'comparison_split': comp_str,
                'comparison_residual': round(cell_means[comp_key], 4),
                'interpretation': interp,
            })
    return warnings


def _interaction_interpretation(fa: str, fb: str,
                                 worst_key: tuple[bool, bool],
                                 worst_mean: float,
                                 comp_mean: float,
                                 n: int) -> str:
    """Render a one-sentence diagnosis of the worst-cell × comparison-cell gap.

    ``worst_key`` is (a_high, b_high). The phrasing names the *primary*
    factor (the one that appears to have a larger calibration slope) and
    flags how the *secondary* moderates it. Sample size is included in
    parens so the user can immediately discount low-n warnings.
    """
    a_hi, b_hi = worst_key
    a_state = 'high' if a_hi else 'low'
    b_state = 'high' if b_hi else 'low'
    direction = 'over-predicts' if worst_mean < 0 else 'under-predicts'
    if abs(comp_mean) < abs(worst_mean) * 0.5:
        return (
            f"{fa} {direction} when {fb} is {b_state} but stays calibrated "
            f"when {fb} is {'low' if b_hi else 'high'} (sample n={n})"
        )
    return (
        f"{fa}×{fb} interaction: residual flips between "
        f"({fa}={a_state}, {fb}={b_state}) and the same-primary cell "
        f"with opposite secondary (sample n={n})"
    )


def _persist(conn, league_id: str, side: str, n_obs: int, output: dict) -> None:
    """Write one row to ``rec_residual_calibration``.

    The shadow_mode flag is hard-coded to 1 in this module — adoption is a
    separate decision tracked via config + a manual flip when sample size
    + signal quality justify it.
    """
    try:
        conn.execute(
            """
            INSERT INTO rec_residual_calibration
                (league_id, side, n_observations, factor_calibration_json,
                 interaction_warnings_json, diagnostics_json, shadow_mode)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            (
                league_id, side, int(n_obs),
                json.dumps(output.get('factor_calibration') or {}),
                json.dumps(output.get('interaction_warnings') or []),
                json.dumps(output.get('diagnostics') or {}),
            ),
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.warning("persist residual calibration failed: %s", e)
