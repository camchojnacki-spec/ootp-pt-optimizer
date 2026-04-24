"""Meta weight calibration — fits card ratings against in-game performance.

This is the 2026-04-15 rewrite that replaced the pure-Python OLS calibrator.
The old implementation had four structural problems, all fixed here:

1. **No league filter.** It regressed against every stats row in the DB
   regardless of league, so stale i76 noise and the current lb124 signal
   got mixed into one fit. The new loader accepts ``league_id`` (defaults
   to ``config.yaml:active_league``) and scopes every query.
2. **Brittle name matching.** It normalized player_name and did partial
   substring matching, which routinely mis-joined "Jose Ramirez" across
   card eras. The new loader joins on ``card_id`` (populated by
   ``_match_card_id`` in ingestion) and only falls back to name matching
   when explicitly asked.
3. **Wrong target.** It regressed against raw OPS / ERA+, which are
   noisy and PA-weighted. The new loader targets WAR-per-600-PA (batting)
   and WAR-per-200-IP (pitching), which are the single-number value stats
   the optimizer actually cares about.
4. **No SP/RP split.** Starters and relievers have radically different
   WAR/IP dynamics (leverage, bullpen usage, etc.). A single pitching
   regression blends them into incoherent weights. The new fit produces
   three pitching rows: ``pitching_sp``, ``pitching_rp``, and a combined
   ``pitching`` for legacy callers.

The fitting approach is **RidgeCV → NNLS → Bayesian prior blend**:

- RidgeCV picks the regularization alpha and gives us a 5-fold CV R² for
  gating. Ridge handles multicollinearity between stuff/movement/control
  that crushed the old OLS fit.
- NNLS (non-negative least squares) refits the standardized design matrix
  to produce interpretable non-negative empirical weights. Ridge coefs on
  multicollinear features can flip sign in ways that don't match the
  "ratings help, they don't hurt" intuition — NNLS prevents that.
- Bayesian blend: ``final = confidence * empirical + (1 - confidence) *
  prior`` where confidence = n_factor × r2_factor × 0.80. n_factor saturates
  at n=200; r2_factor saturates at CV R²=0.15. So a big, clean fit earns
  ~0.80 confidence; a big, noisy fit stays closer to prior. Unfitted prior
  keys are carried forward at prior value so the emitted weight dict is
  always complete (prevents meta_scoring from hitting stale fallbacks).

Two additional safety rails:

- **Sanity gate** — any new weight that would swing more than −60% or
  more than +100% from the previous run is clipped to the boundary.
  The user's 2026-04-15 calibration (movement 2.2→1.2, a −45% cut) was
  the motivating case for relaxing from the original −50% cap.
- **Winsorization** — y is clipped at the 1st/99th percentile before
  fitting so one outlier card doesn't dominate the loss.

Public API (stable):
    run_calibration(conn=None, league_id=None) -> dict
    maybe_calibrate(min_new_rows=50, conn=None, league_id=None) -> bool
    get_calibration_comparison(conn=None) -> dict       # legacy UI
    auto_calibrate_if_ready(conn=None) -> bool          # legacy caller
    calibrate_batting(conn, league_id, prior_weights) -> dict
    calibrate_pitching(conn, league_id, role_filter, prior_weights) -> dict
"""
import json
import logging
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

from app.core.database import get_connection, load_config
from app.utils.constants import DEFAULT_BATTING_WEIGHTS, DEFAULT_PITCHING_WEIGHTS

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Configuration constants
# ══════════════════════════════════════════════════════════════════════════

# Features fed to the batting regression.
# avoid_ks dropped: univariate r=+0.069 vs WAR/600, statistical noise.
# speed_stealing dropped: 2026-04-15 inspection showed only 57/297 lb124
#   batters clear the speed>=70 floor that calc_speed_score enforces; below
#   the n=100 threshold for a useful regression signal. Kept at config prior.
# defense dropped: computed post-hoc from per-position ratings, not a single
#   card column.
BATTING_FEATURES = ['contact', 'gap_power', 'power', 'eye', 'babip']

# Features fed to the pitching regression.
PITCHING_FEATURES = ['stuff', 'movement', 'control', 'p_hr']
# Interaction terms — SIERA-style non-additive synergies. Each tuple is
# (output_name, rating_a, rating_b).
#
# 2026-04-15: Set empty. First calibration run with interactions showed
# severe overfit on RP (CV R²=−0.113 vs in-sample 0.183) — 7 features /
# 169 samples with low signal-to-noise was fitting pure noise. The 4
# main features alone hold up in CV. Can re-enable once sample size
# grows past ~500/role.
PITCHING_INTERACTIONS = []

# Minimum sample sizes per fit. Pitching is lower because the SP/RP split
# halves each sub-sample.
MIN_BATTING_N = 30
MIN_PITCHING_N = 20

# Bayesian prior blending:
#   confidence = n_factor * r2_factor * CONFIDENCE_CAP
# where:
#   n_factor  = min(n / CONFIDENCE_N_SATURATION, 1.0)       # sample size
#   r2_factor = max(0, min(cv_r² / R2_QUALITY_TARGET, 1.0)) # fit quality
#
# 2026-04-17 v3: Cap restored to 0.80 (its original design value) but the
# path to the cap is now R²-gated. Previous v2 forced a flat 0.40 cap after
# multicollinearity artifacts produced "whack" rankings at 0.80 — that was a
# symptom of the bad fallbacks (B01) and circular OVR anchor, both fixed in
# the 2026-04-17 meta engine surgery. With those gone, a high-R² fit on
# fresh stats should be trusted; a low-R² fit on any stats still defers to
# the prior.
#
# Behavior across sample size × R² combos (CONFIDENCE_CAP=0.80, target=0.15):
#     n=300, CV R²=0.20  -> 1.0 * 1.0  * 0.80 = 0.80   (full trust)
#     n=300, CV R²=0.075 -> 1.0 * 0.50 * 0.80 = 0.40   (equivalent to v2)
#     n=300, CV R²=0.03  -> 1.0 * 0.20 * 0.80 = 0.16   (stay near prior)
#     n=80,  CV R²=0.20  -> 0.4 * 1.0  * 0.80 = 0.32   (small-sample anchor)
CONFIDENCE_N_SATURATION = 200.0
CONFIDENCE_CAP = 0.80
R2_QUALITY_TARGET = 0.15

# Sanity gate: reject/clip weight moves between consecutive runs.
# Tightened in 3 stages: -60%/+2× → -40%/+1.5× → -25%/+1.4×.
# With CV R² at 0.03-0.16, the regression barely outperforms the mean
# and the empirical weights carry significant multicollinearity bias
# (Ridge redistributing stuff↔movement shared variance). The gate keeps
# the calibration's influence to a ±20% nudge from prior rather than
# structural re-weighting that produced "whack" rankings.
SANITY_DOWN_FRAC = 0.25   # max 25% down-swing
SANITY_UP_MULT = 1.40     # max 1.4× up-swing

# Throttle for maybe_calibrate: only re-run when at least this many new
# stat rows have landed since the last calibration.
DEFAULT_MIN_NEW_ROWS = 50


# ══════════════════════════════════════════════════════════════════════════
# Core fit: Ridge → NNLS → prior blend
# ══════════════════════════════════════════════════════════════════════════

def _build_design_matrix(pairs, feature_names, interactions=None, weights=None):
    """Turn a list of (features_dict, target_value[, weight]) triples into (X, y, w).

    Each row of X is ``[features[f] for f in feature_names]`` followed by
    ``features[a] * features[b]`` for each interaction tuple. If ``weights``
    is a list of per-row weights the function returns ``(X, y, w)``;
    otherwise it returns ``(X, y, None)`` so callers can skip sample
    weighting.
    """
    X_list, y_list, w_list = [], [], []
    for i, (features, target) in enumerate(pairs):
        if target is None:
            continue
        row = [float(features.get(f) or 0) for f in feature_names]
        if interactions:
            for _, a, b in interactions:
                row.append(float(features.get(a) or 0) * float(features.get(b) or 0))
        X_list.append(row)
        y_list.append(float(target))
        if weights is not None:
            w_list.append(float(weights[i]))
    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    w = np.asarray(w_list, dtype=float) if weights is not None else None
    return X, y, w


def _full_feature_names(base_features, interactions=None):
    """Concat main features + interaction output names in order."""
    names = list(base_features)
    if interactions:
        for label, _, _ in interactions:
            names.append(label)
    return names


def _fit_weights(X, y, feature_names, prior_weights, min_samples, sample_weights=None):
    """RidgeCV + NNLS + Bayesian blend with the supplied prior.

    Returns None if the sample is too small or the fit blows up.
    Otherwise returns a dict with ``weights``, ``r_squared`` (cross-validated),
    ``correlation``, ``sample_size``, ``confidence``, ``alpha``, and diagnostic
    ``empirical_raw`` / ``target_range`` fields.

    ``sample_weights`` (optional) — per-row weights, typically ``sqrt(PA)`` or
    ``sqrt(IP)`` so low-sample cards don't dominate the loss with their
    noisy target estimates. Passed to both RidgeCV and NNLS (the latter via
    √w row-scaling, which is mathematically equivalent to weighted LS).
    """
    if X.ndim != 2 or y.ndim != 1:
        return None
    n, k = X.shape
    if n < min_samples or k == 0:
        return None

    try:
        # Winsorize target at 1st/99th pct so one outlier doesn't dominate.
        lo_q, hi_q = np.percentile(y, [1, 99])
        y_w = np.clip(y, lo_q, hi_q)

        # Normalize sample weights so they sum to n (keeps loss scale
        # comparable to the unweighted case — useful for debugging).
        if sample_weights is not None and len(sample_weights) == n:
            sw = np.asarray(sample_weights, dtype=float)
            sw = np.where(sw > 0, sw, 1e-6)
            sw = sw * (n / sw.sum())
        else:
            sw = None

        # RidgeCV picks the alpha and handles multicollinear features.
        # Grid extended on 2026-04-15: previous runs always picked alpha=100
        # (the old max), meaning regularization wanted to go stronger.
        alphas = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
        ridge = RidgeCV(alphas=alphas)
        if sw is not None:
            ridge.fit(X, y_w, sample_weight=sw)
            in_sample_r2 = float(ridge.score(X, y_w, sample_weight=sw))
        else:
            ridge.fit(X, y_w)
            in_sample_r2 = float(ridge.score(X, y_w))

        # Cross-validated R² is the gate metric (we care about
        # out-of-sample generalization, not memorized fit).
        cv_folds = min(5, max(2, n // 20))
        try:
            fit_params = {'sample_weight': sw} if sw is not None else None
            cv_scores = cross_val_score(
                RidgeCV(alphas=alphas), X, y_w, cv=cv_folds, scoring='r2',
                params=fit_params,
            )
            cv_r2 = float(np.nanmean(cv_scores))
        except TypeError:
            # Older sklearn: cross_val_score uses fit_params kwarg.
            try:
                cv_scores = cross_val_score(
                    RidgeCV(alphas=alphas), X, y_w, cv=cv_folds, scoring='r2',
                    fit_params=fit_params,
                )
                cv_r2 = float(np.nanmean(cv_scores))
            except Exception:
                cv_r2 = in_sample_r2
        except Exception:
            cv_r2 = in_sample_r2

        # Pearson r between Ridge's predictions and actual target.
        y_pred = ridge.predict(X)
        if np.std(y_pred) > 1e-9 and np.std(y_w) > 1e-9:
            corr = float(np.corrcoef(y_pred, y_w)[0, 1])
        else:
            corr = 0.0

        # NNLS refit for interpretable non-negative weights. We center
        # and scale first so the solver sees features on comparable scales,
        # then un-scale the coefficients back to per-raw-rating units.
        # Sample weighting: multiply both X and y rows by sqrt(w), which
        # gives weighted least squares equivalence.
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std_safe = np.where(X_std > 1e-9, X_std, 1.0)
        X_scaled = (X - X_mean) / X_std_safe
        y_centered = y_w - y_w.mean()

        if sw is not None:
            sqrt_w = np.sqrt(sw)[:, None]
            X_fit = X_scaled * sqrt_w
            y_fit = y_centered * sqrt_w[:, 0]
        else:
            X_fit = X_scaled
            y_fit = y_centered

        try:
            nnls_coef_scaled, _ = nnls(X_fit, y_fit)
        except Exception as fit_err:
            logger.warning(f"NNLS failed ({fit_err}), falling back to clipped Ridge")
            nnls_coef_scaled = np.clip(ridge.coef_ * X_std_safe, 0, None)

        nnls_coef_raw = nnls_coef_scaled / X_std_safe

        # Build empirical weights dict and normalize to the prior's total
        # so the scorer's overall magnitude stays stable.
        empirical = {
            f: max(float(nnls_coef_raw[i]), 0.0)
            for i, f in enumerate(feature_names)
        }
        prior_total = sum(max(float(prior_weights.get(f, 0.0)), 0.0) for f in feature_names)
        emp_total = sum(empirical.values())
        if emp_total > 1e-9 and prior_total > 1e-9:
            scale = prior_total / emp_total
            empirical_scaled = {f: v * scale for f, v in empirical.items()}
        else:
            empirical_scaled = dict(empirical)

        # Bayesian blend with prior — scaled by BOTH sample size and CV R².
        # A big, clean fit earns high confidence; a big, noisy fit stays near
        # the prior even at n=1000.
        n_factor = min(n / CONFIDENCE_N_SATURATION, 1.0)
        r2_factor = max(0.0, min(cv_r2 / R2_QUALITY_TARGET, 1.0))
        confidence = n_factor * r2_factor * CONFIDENCE_CAP

        blended = {}
        # Fitted features: empirical * confidence + prior * (1 - confidence)
        for f in feature_names:
            emp = empirical_scaled.get(f, 0.0)
            pri = float(prior_weights.get(f, 0.0))
            blended[f] = round(confidence * emp + (1 - confidence) * pri, 4)
        # Carry forward every other prior key at its prior value so the
        # output dict is complete. This prevents meta_scoring.py's
        # ``weights.get(key, DEFAULT)`` from silently reverting unfitted
        # features (defense, speed_stealing, stamina_hold, interactions)
        # to stale v4 fallbacks when the calibration only fits 4–5 features.
        for f, pri in prior_weights.items():
            if f not in blended:
                blended[f] = round(float(pri), 4)

        return {
            'weights': blended,
            'empirical_raw': {f: round(empirical.get(f, 0.0), 6) for f in feature_names},
            'empirical_scaled': {f: round(empirical_scaled.get(f, 0.0), 4) for f in feature_names},
            'r_squared': round(cv_r2, 4),
            'r_squared_in_sample': round(in_sample_r2, 4),
            'correlation': round(corr, 4),
            'confidence': round(confidence, 4),
            'sample_size': n,
            'alpha': float(ridge.alpha_),
            'target_range': [round(float(y_w.min()), 2), round(float(y_w.max()), 2)],
            'cv_folds': cv_folds,
        }
    except Exception as e:
        logger.error(f"_fit_weights failed: {e}", exc_info=True)
        return None


# ══════════════════════════════════════════════════════════════════════════
# Sanity gate
# ══════════════════════════════════════════════════════════════════════════

def _sanity_clip(new_weights, old_weights,
                 down_frac=SANITY_DOWN_FRAC, up_mult=SANITY_UP_MULT):
    """Clip new weights into ``[old*(1-down_frac), old*up_mult]``.

    A missing or zero old weight passes through unchanged (no prior to clip
    against). Returns ``(clipped_dict, list_of_clip_events)`` where each
    clip event describes the direction, old/new values, and clipped target.
    """
    clipped = {}
    events = []
    for k, new_v in new_weights.items():
        old_v = old_weights.get(k)
        if old_v is None or old_v <= 0:
            clipped[k] = new_v
            continue
        lo = old_v * (1 - down_frac)
        hi = old_v * up_mult
        if new_v < lo:
            clipped[k] = round(lo, 4)
            events.append({
                'key': k, 'direction': 'down',
                'old': round(float(old_v), 4),
                'new': round(float(new_v), 4),
                'clipped_to': round(lo, 4),
            })
        elif new_v > hi:
            clipped[k] = round(hi, 4)
            events.append({
                'key': k, 'direction': 'up',
                'old': round(float(old_v), 4),
                'new': round(float(new_v), 4),
                'clipped_to': round(hi, 4),
            })
        else:
            clipped[k] = new_v
    return clipped, events


# ══════════════════════════════════════════════════════════════════════════
# Data loaders (league-scoped, card_id-joined)
# ══════════════════════════════════════════════════════════════════════════

def _current_league_id():
    """Read ``active_league`` from config.yaml, defaulting to ``lb124``."""
    try:
        return load_config().get('active_league', 'lb124')
    except Exception:
        return 'lb124'


def _fetch_latest_batting_stats(conn, league_id, min_pa=150):
    """Pull the latest snapshot per card_id from ``batting_stats``.

    Joins on ``card_id`` (no brittle name matching) and filters to the
    active league. Excludes pitchers (position=1) and cards with no
    contact rating.
    """
    rows = conn.execute("""
        SELECT c.card_id, c.contact, c.gap_power, c.power, c.eye,
               c.avoid_ks, c.babip,
               bs.pa, bs.war, bs.ops_plus, bs.ops
        FROM cards c
        INNER JOIN batting_stats bs ON bs.card_id = c.card_id
        INNER JOIN (
            SELECT card_id, MAX(snapshot_date) AS max_date
            FROM batting_stats
            WHERE league_id = ? AND card_id IS NOT NULL
            GROUP BY card_id
        ) latest ON bs.card_id = latest.card_id
                 AND bs.snapshot_date = latest.max_date
        WHERE bs.league_id = ?
          AND bs.pa >= ?
          AND (c.position IS NULL OR c.position != 1)
          AND c.contact IS NOT NULL
          AND c.contact > 0
    """, (league_id, league_id, min_pa)).fetchall()
    return [dict(r) for r in rows]


def _is_sp_row(row):
    """True if this card row represents a starting pitcher.

    Handles the OOTP 27 encoding used in the ``cards`` table:
      - ``pitcher_role_name`` is a short code: 'SP', 'RP', 'CL', 'MR', 'SU',
        'LR', 'CP', 'SW' (not the long name).
      - Numeric ``pitcher_role`` is 11 (SP), 12 (RP), 13 (CL).

    Falls back to numeric only when the short code is missing or unknown.
    """
    role_name = row.get('pitcher_role_name')
    if isinstance(role_name, str):
        stripped = role_name.strip().upper()
        if stripped == 'SP' or 'STARTING' in stripped or stripped == 'STARTER':
            return True
        if stripped in ('RP', 'CL', 'MR', 'SU', 'LR', 'CP', 'SW'):
            return False
    role_num = row.get('pitcher_role')
    try:
        return int(role_num) == 11
    except (TypeError, ValueError):
        return False


def _fetch_latest_pitching_stats(conn, league_id, min_ip=50, role_filter=None):
    """Pull the latest pitching snapshot per card_id for the league.

    ``role_filter``:
        None → all pitchers
        'SP' → only starting pitchers (``pitcher_role_name`` == 'SP' or
               numeric ``pitcher_role`` == 11)
        'RP' → everyone else (RP/CL/MR/SU/LR/CP/SW)
    """
    rows = conn.execute("""
        SELECT c.card_id, c.pitcher_role_name, c.pitcher_role,
               c.stuff, c.movement, c.control, c.p_hr,
               ps.ip, ps.war, ps.era_plus
        FROM cards c
        INNER JOIN pitching_stats ps ON ps.card_id = c.card_id
        INNER JOIN (
            SELECT card_id, MAX(snapshot_date) AS max_date
            FROM pitching_stats
            WHERE league_id = ? AND card_id IS NOT NULL
            GROUP BY card_id
        ) latest ON ps.card_id = latest.card_id
                 AND ps.snapshot_date = latest.max_date
        WHERE ps.league_id = ?
          AND ps.ip >= ?
          AND c.stuff IS NOT NULL
          AND c.stuff > 0
    """, (league_id, league_id, min_ip)).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        is_sp = _is_sp_row(d)
        if role_filter == 'SP' and not is_sp:
            continue
        if role_filter == 'RP' and is_sp:
            continue
        out.append(d)
    return out


def _pairs_batting(rows):
    """Turn fetched batting rows into (features_dict, target, weight) triples.

    Target is WAR per 600 PA. Weight is sqrt(PA) so high-PA cards get more
    influence than low-PA noise — without this a 155-PA card has the same
    pull on the regression as a 650-PA regular, even though its WAR estimate
    is ~2× noisier.
    """
    pairs = []
    weights = []
    for r in rows:
        pa = float(r.get('pa') or 0)
        war = r.get('war')
        if pa <= 0 or war is None:
            continue
        target = float(war) * 600.0 / pa
        pairs.append((r, target))
        weights.append(np.sqrt(pa))
    return pairs, weights


def _pairs_pitching(rows):
    """Turn fetched pitching rows into (features_dict, target, weight) triples.

    Target is WAR per 200 IP (SP scale) so SP and RP fits are comparable.
    Weight is sqrt(IP) for the same noise-budget reason as batting.
    """
    pairs = []
    weights = []
    for r in rows:
        ip = float(r.get('ip') or 0)
        war = r.get('war')
        if ip <= 0 or war is None:
            continue
        target = float(war) * 200.0 / ip
        pairs.append((r, target))
        weights.append(np.sqrt(ip))
    return pairs, weights


# ══════════════════════════════════════════════════════════════════════════
# Category fits
# ══════════════════════════════════════════════════════════════════════════

def calibrate_batting(conn, league_id=None, prior_weights=None):
    """Fit batting weights for the given league.

    Returns either ``{'error': str, 'sample_size': n}`` or the full fit
    dict (with ``weights``, ``r_squared``, ``correlation``, etc.).
    """
    league_id = league_id or _current_league_id()
    if prior_weights is None:
        try:
            cfg = load_config()
            prior_weights = cfg.get('batting_weights', DEFAULT_BATTING_WEIGHTS)
        except Exception:
            prior_weights = DEFAULT_BATTING_WEIGHTS

    rows = _fetch_latest_batting_stats(conn, league_id)
    pairs, sample_weights = _pairs_batting(rows)
    if len(pairs) < MIN_BATTING_N:
        return {
            'error': f'Only {len(pairs)} batters with WAR+PA in {league_id} '
                     f'(need ≥{MIN_BATTING_N}).',
            'sample_size': len(pairs),
            'league_id': league_id,
        }

    X, y, w = _build_design_matrix(pairs, BATTING_FEATURES, weights=sample_weights)
    result = _fit_weights(X, y, BATTING_FEATURES, prior_weights, MIN_BATTING_N,
                          sample_weights=w)
    if result is None:
        return {
            'error': 'Ridge/NNLS fit failed',
            'sample_size': len(pairs),
            'league_id': league_id,
        }

    result['league_id'] = league_id
    result['target'] = 'WAR_per_600_PA'
    return result


def calibrate_pitching(conn, league_id=None, role_filter=None, prior_weights=None):
    """Fit pitching weights for the given league + role slice.

    ``role_filter``: None (all), 'SP', or 'RP'. Uses a lower IP threshold
    for RP-only so we capture more bullpen samples.
    """
    league_id = league_id or _current_league_id()
    if prior_weights is None:
        try:
            cfg = load_config()
            prior_weights = cfg.get('pitching_weights', DEFAULT_PITCHING_WEIGHTS)
        except Exception:
            prior_weights = DEFAULT_PITCHING_WEIGHTS

    # IP thresholds dialed per-role against the lb124 distribution.
    # SP median is ~70 IP so 50 catches ~p50-p60 = 140 samples.
    # RP: raised from 25→40 on 2026-04-18. At 25 IP the RP fit had
    # CV R² = 0.005 (essentially zero out-of-sample predictive power)
    # because 25-IP samples are still dominated by leverage/usage noise
    # rather than true talent. IP≥40 trims the noisiest ~30% of RPs while
    # leaving enough (~120) to clear MIN_PITCHING_N=20 comfortably.
    # Combined fits stay at 30 IP as a compromise; callers who need the
    # RP-specific signal should read the 'pitching_rp' row directly.
    if role_filter == 'RP':
        min_ip = 40
    elif role_filter == 'SP':
        min_ip = 50
    else:
        min_ip = 30
    rows = _fetch_latest_pitching_stats(conn, league_id, min_ip=min_ip,
                                        role_filter=role_filter)
    pairs, sample_weights = _pairs_pitching(rows)
    if len(pairs) < MIN_PITCHING_N:
        return {
            'error': f'Only {len(pairs)} pitchers ({role_filter or "all"}) '
                     f'with WAR+IP in {league_id} (need ≥{MIN_PITCHING_N}).',
            'sample_size': len(pairs),
            'league_id': league_id,
            'role_filter': role_filter or 'all',
        }

    full_features = _full_feature_names(PITCHING_FEATURES, PITCHING_INTERACTIONS)
    X, y, w = _build_design_matrix(pairs, PITCHING_FEATURES, PITCHING_INTERACTIONS,
                                    weights=sample_weights)
    result = _fit_weights(X, y, full_features, prior_weights, MIN_PITCHING_N,
                          sample_weights=w)
    if result is None:
        return {
            'error': 'Ridge/NNLS fit failed',
            'sample_size': len(pairs),
            'league_id': league_id,
            'role_filter': role_filter or 'all',
        }

    result['league_id'] = league_id
    result['target'] = 'WAR_per_200_IP'
    result['role_filter'] = role_filter or 'all'
    return result


# ══════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════

def _save_calibration(conn, calibration_type, fit_result, sanity_events):
    """Write a fit result into ``meta_calibration``."""
    if not fit_result or 'error' in fit_result:
        return False
    weights_json = json.dumps(fit_result['weights'])
    changes = {
        'empirical_raw': fit_result.get('empirical_raw', {}),
        'empirical_scaled': fit_result.get('empirical_scaled', {}),
        'sanity_clip_events': sanity_events,
        'alpha': fit_result.get('alpha'),
        'target': fit_result.get('target'),
        'target_range': fit_result.get('target_range'),
        'league_id': fit_result.get('league_id'),
        'role_filter': fit_result.get('role_filter'),
        'r_squared_in_sample': fit_result.get('r_squared_in_sample'),
        'cv_folds': fit_result.get('cv_folds'),
    }
    conn.execute("""
        INSERT INTO meta_calibration
            (calibration_type, weights_json, r_squared, correlation,
             sample_size, confidence, changes_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        calibration_type,
        weights_json,
        fit_result.get('r_squared'),
        fit_result.get('correlation'),
        fit_result.get('sample_size'),
        fit_result.get('confidence'),
        json.dumps(changes),
        datetime.now().isoformat(),
    ))
    return True


# ══════════════════════════════════════════════════════════════════════════
# Main entry: run_calibration
# ══════════════════════════════════════════════════════════════════════════

def run_calibration(conn=None, league_id=None):
    """Run the full calibration suite for the active league.

    Writes up to 4 rows to ``meta_calibration``:
      - ``batting``      (WAR/600 target)
      - ``pitching_sp``  (SP-only, WAR/200 target)
      - ``pitching_rp``  (RP-only, WAR/200 target)
      - ``pitching``     (combined SP+RP for legacy callers)

    Returns a summary dict with each fit result nested under its key.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        league_id = league_id or _current_league_id()
        try:
            cfg = load_config()
            prior_bat = cfg.get('batting_weights', DEFAULT_BATTING_WEIGHTS)
            prior_pit = cfg.get('pitching_weights', DEFAULT_PITCHING_WEIGHTS)
        except Exception:
            prior_bat = DEFAULT_BATTING_WEIGHTS
            prior_pit = DEFAULT_PITCHING_WEIGHTS

        summary = {
            'league_id': league_id,
            'started_at': datetime.now().isoformat(),
            'batting': None,
            'pitching_sp': None,
            'pitching_rp': None,
            'pitching': None,
        }

        # Batting
        bat_fit = calibrate_batting(conn, league_id=league_id, prior_weights=prior_bat)
        if 'error' not in bat_fit:
            clipped, events = _sanity_clip(bat_fit['weights'], prior_bat)
            bat_fit['weights'] = clipped
            bat_fit['sanity_clip_events'] = events
            _save_calibration(conn, 'batting', bat_fit, events)
        summary['batting'] = bat_fit

        # Pitching — SP
        sp_fit = calibrate_pitching(conn, league_id=league_id,
                                    role_filter='SP', prior_weights=prior_pit)
        if 'error' not in sp_fit:
            clipped, events = _sanity_clip(sp_fit['weights'], prior_pit)
            sp_fit['weights'] = clipped
            sp_fit['sanity_clip_events'] = events
            _save_calibration(conn, 'pitching_sp', sp_fit, events)
        summary['pitching_sp'] = sp_fit

        # Pitching — RP
        rp_fit = calibrate_pitching(conn, league_id=league_id,
                                    role_filter='RP', prior_weights=prior_pit)
        if 'error' not in rp_fit:
            clipped, events = _sanity_clip(rp_fit['weights'], prior_pit)
            rp_fit['weights'] = clipped
            rp_fit['sanity_clip_events'] = events
            _save_calibration(conn, 'pitching_rp', rp_fit, events)
        summary['pitching_rp'] = rp_fit

        # Pitching — combined (legacy compat for meta_scoring's old loader
        # and meta_validation.load_calibrated_weights).
        combined_fit = calibrate_pitching(conn, league_id=league_id,
                                          role_filter=None, prior_weights=prior_pit)
        if 'error' not in combined_fit:
            clipped, events = _sanity_clip(combined_fit['weights'], prior_pit)
            combined_fit['weights'] = clipped
            combined_fit['sanity_clip_events'] = events
            _save_calibration(conn, 'pitching', combined_fit, events)
        summary['pitching'] = combined_fit

        conn.commit()
        summary['finished_at'] = datetime.now().isoformat()
        return summary
    finally:
        if close_conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════
# Throttled entry: maybe_calibrate
# ══════════════════════════════════════════════════════════════════════════

def _count_new_rows_since_last_calibration(conn, league_id):
    """Count (batting_stats + pitching_stats) rows added since the last run.

    Used to avoid recalibrating on every ingest when little has changed.
    """
    last = conn.execute(
        "SELECT created_at FROM meta_calibration "
        "WHERE calibration_type IN ('batting', 'pitching', 'pitching_sp') "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not last or not last[0]:
        return 1_000_000  # no prior run → always calibrate

    last_at = last[0]
    try:
        bat = conn.execute(
            "SELECT COUNT(*) FROM batting_stats "
            "WHERE league_id = ? AND datetime(snapshot_date) > datetime(?)",
            (league_id, last_at)
        ).fetchone()[0]
        pit = conn.execute(
            "SELECT COUNT(*) FROM pitching_stats "
            "WHERE league_id = ? AND datetime(snapshot_date) > datetime(?)",
            (league_id, last_at)
        ).fetchone()[0]
        return int(bat or 0) + int(pit or 0)
    except Exception as e:
        logger.warning(f"_count_new_rows failed: {e}")
        return 0


def maybe_calibrate(min_new_rows=DEFAULT_MIN_NEW_ROWS, conn=None, league_id=None):
    """Run calibration only if enough new stat rows have arrived.

    Called from ``recommendations.generate_recommendations`` after each
    ingest. Returns True if a run actually happened, False if skipped.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        league_id = league_id or _current_league_id()
        new_rows = _count_new_rows_since_last_calibration(conn, league_id)
        if new_rows < min_new_rows:
            logger.info(
                f"Skip calibration: only {new_rows} new stat rows since "
                f"last run (threshold {min_new_rows})"
            )
            return False

        summary = run_calibration(conn, league_id=league_id)
        bat_r2 = (summary.get('batting') or {}).get('r_squared')
        sp_r2 = (summary.get('pitching_sp') or {}).get('r_squared')
        rp_r2 = (summary.get('pitching_rp') or {}).get('r_squared')
        logger.info(
            f"Calibration ran: batting R²={bat_r2}, SP R²={sp_r2}, RP R²={rp_r2}"
        )
        return True
    except Exception as e:
        logger.error(f"maybe_calibrate failed: {e}", exc_info=True)
        return False
    finally:
        if close_conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════
# Legacy-compat shims
# ══════════════════════════════════════════════════════════════════════════

def auto_calibrate_if_ready(conn=None):
    """Legacy entry point called by ``recommendations.generate_recommendations``.

    Routed through ``maybe_calibrate`` with the default 50-row throttle so
    recommendations.py doesn't have to change. Returns True if calibration
    actually ran, False if skipped.
    """
    return maybe_calibrate(min_new_rows=DEFAULT_MIN_NEW_ROWS, conn=conn)


def calibrate_batting_weights(conn=None):
    """Legacy wrapper for the old UI page. Returns the new calibrate_batting
    result in a shape compatible with get_calibration_comparison callers.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        fit = calibrate_batting(conn)
        if 'error' in fit:
            return {
                'weights': {}, 'r_squared': 0.0,
                'sample_size': fit.get('sample_size', 0),
                'error': fit['error'],
            }
        return {
            'weights': fit['weights'],
            'r_squared': fit['r_squared'],
            'sample_size': fit['sample_size'],
            'error': None,
        }
    finally:
        if close_conn:
            conn.close()


def calibrate_pitching_weights(conn=None):
    """Legacy wrapper — returns combined SP+RP pitching fit."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        fit = calibrate_pitching(conn, role_filter=None)
        if 'error' in fit:
            return {
                'weights': {}, 'r_squared': 0.0,
                'sample_size': fit.get('sample_size', 0),
                'error': fit['error'],
            }
        return {
            'weights': fit['weights'],
            'r_squared': fit['r_squared'],
            'sample_size': fit['sample_size'],
            'error': None,
        }
    finally:
        if close_conn:
            conn.close()


def get_calibration_comparison(conn=None):
    """Return ``{'batting': {...}, 'pitching': {...}}`` for the UI.

    Each sub-dict has ``current`` (config), ``suggested`` (fresh fit),
    ``r_squared``, ``sample_size``, and ``error``. The suggested weights
    are the pre-sanity-clip empirical values so the user can see the raw
    recommendation before it's gated.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        try:
            cfg = load_config()
            cur_bat = cfg.get('batting_weights', dict(DEFAULT_BATTING_WEIGHTS))
            cur_pit = cfg.get('pitching_weights', dict(DEFAULT_PITCHING_WEIGHTS))
        except Exception:
            cur_bat = dict(DEFAULT_BATTING_WEIGHTS)
            cur_pit = dict(DEFAULT_PITCHING_WEIGHTS)

        bat_fit = calibrate_batting(conn, prior_weights=cur_bat)
        pit_fit = calibrate_pitching(conn, role_filter=None, prior_weights=cur_pit)

        def _pack(cur, fit):
            if 'error' in fit:
                return {
                    'current': cur, 'suggested': {}, 'r_squared': 0.0,
                    'sample_size': fit.get('sample_size', 0),
                    'error': fit['error'],
                }
            return {
                'current': cur,
                'suggested': fit.get('weights', {}),
                'r_squared': fit.get('r_squared', 0.0),
                'correlation': fit.get('correlation', 0.0),
                'sample_size': fit.get('sample_size', 0),
                'error': None,
            }

        return {
            'batting': _pack(cur_bat, bat_fit),
            'pitching': _pack(cur_pit, pit_fit),
        }
    except Exception as e:
        logger.error(f"get_calibration_comparison failed: {e}", exc_info=True)
        return {
            'batting': {'current': {}, 'suggested': {}, 'r_squared': 0.0,
                        'sample_size': 0, 'error': str(e)},
            'pitching': {'current': {}, 'suggested': {}, 'r_squared': 0.0,
                         'sample_size': 0, 'error': str(e)},
        }
    finally:
        if close_conn:
            conn.close()
