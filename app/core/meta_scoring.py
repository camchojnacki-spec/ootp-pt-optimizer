"""Meta score calculations for batters and pitchers."""
import json
import math
from app.core.database import load_config, get_db_path
from app.utils.constants import (
    DEFAULT_BATTING_WEIGHTS, DEFAULT_PITCHING_WEIGHTS,
    DEFAULT_PITCHING_WEIGHTS_SP, DEFAULT_PITCHING_WEIGHTS_RP,
    PITCHING_STAT_FLOOR, BATTING_STAT_FLOOR,
    POSITION_DEFENSE_MULTIPLIERS, POSITIONAL_VALUE_BONUS,
    BATTING_CARD_TYPE_OFFSET, PITCHING_CARD_TYPE_OFFSET,
)

# Diminishing returns threshold — stats above this get sqrt-scaled benefit
DIMINISHING_RETURNS_THRESHOLD = 110

# Minimum cross-validated R² a calibration run needs to clear before its
# weights are trusted. Below this, the live formula falls through to
# config.yaml / defaults.
#
# History: this was originally 0.40 under the theory that a calibration
# explaining less than half the variance would actively misrank players.
# That was too aggressive — the rebuilt 2026-04-15 calibrator targets WAR
# (single-number value) instead of OPS/ERA+ (noisy rate stats), and the
# achievable out-of-sample R² is structurally lower on WAR because WAR
# itself is noisy at small IP/PA.
#
# Observed CV R² on the 2026-04-15 lb124 run (with sqrt(PA)/sqrt(IP)
# sample weighting, no interactions, and RidgeCV alpha=1000):
#     batting     0.158   (Pearson r=0.528, in-sample 0.277)
#     pitching_sp 0.050   (Pearson r=0.477, in-sample 0.232)
#     pitching_rp 0.035   (Pearson r=0.439, in-sample 0.191)
#     pitching    0.089   (Pearson r=0.403, in-sample 0.174)
#
# Pearson r² on the predictions is 0.16–0.28, so the model IS learning
# real ranking signal — it's just losing ground to CV because WAR has a
# large irreducible per-card noise floor that rating features can't
# explain. We gate on two conditions (either suffices):
#   1. CV R² ≥ the floor below, OR
#   2. Pearson r ≥ MIN_CALIBRATION_CORR (0.30), which catches the case
#      where the model ranks correctly but scale/offset is off.
#
# The Bayesian prior blend (cap=0.80) keeps the adopted weights from
# straying too far from config even when both gates barely clear.
MIN_CALIBRATION_R2 = 0.10
MIN_PITCHING_R2 = 0.03
MIN_CALIBRATION_CORR = 0.30

# How the last weight-load decision was reached. Populated by
# _load_calibrated_weights / get_weights_with_source so the UI can surface
# "calibration rejected because R² below threshold" rather than silently
# using defaults.
# Format: dict with keys batting/pitching/pitching_sp/pitching_rp →
#   {status, r_squared, sample_size, reason}
# status is one of: 'used', 'rejected_low_r2', 'missing', 'error'
LAST_WEIGHT_LOAD_DIAGNOSTICS: dict = {}


def _diminished(value: float, threshold: float = DIMINISHING_RETURNS_THRESHOLD) -> float:
    """Apply diminishing returns to stats above the threshold.

    Below threshold: linear (1:1).
    Above threshold: sqrt-scaled so that extreme spikes don't dominate.
    e.g. 150 -> 110 + sqrt(40)*4 ≈ 135 (effective), not raw 150.
    """
    if value <= threshold:
        return value
    excess = value - threshold
    return threshold + math.sqrt(excess) * 4


# ════════════════════════════════════════════════════════════════════════════
# MULTI-FACTOR META LAYERS — v5 (2026-04-16)
#
# These layers inject signals that OVR does NOT capture:
#   1. Platoon-adjusted base: penalizes extreme L/R splits (VALIDATED)
#   2. Arsenal diversity: theoretical value for PT lineup trips (NOT validated
#      by lb124 WAR data — pitch_count r=+0.024; scaled down 60%)
#   3. Position flexibility: theoretical roster value (NOT validated —
#      playable_positions r=-0.023; scaled down 75%)
#   4. Performance overlay: actual game results vs expectations (VALIDATED —
#      wOBA r=+0.884 with WAR, SIERA r=-0.654)
#
# Each layer is backward-compatible — when the row dict doesn't contain
# the needed columns, the layer returns 0 (no adjustment).
# ════════════════════════════════════════════════════════════════════════════

# Handedness distribution in a typical OOTP PT league.
# Batters face ~60% RHP / ~40% LHP; pitchers face ~55% RHB / ~45% LHB.
_BATTER_VS_RHP_WEIGHT = 0.60
_PITCHER_VS_RHB_WEIGHT = 0.55

# Below this threshold on the weak side, the player is functionally unplayable
# against that handedness — an automatic out (batting) or a batting-practice
# pitcher (pitching).
_PLATOON_CRITICAL_BAT = 50
_PLATOON_CRITICAL_PIT = 55

# Position flexibility: minimum OOTP pos_rating to be "playable" at a spot.
# Scale: 0 = can't play, ~20 = playable with defensive cost, 100+ = gold glove.
_POS_PLAYABLE_THRESHOLD = 20

# Arsenal diversity: pitch rating thresholds.
_PITCH_USABLE_THRESHOLD = 50     # real pitch in-game
_PITCH_QUALITY_THRESHOLD = 65    # good enough to miss bats consistently

# Performance overlay: minimum sample sizes before we trust actual stats.
_PERF_MIN_PA = 100
_PERF_MIN_IP = 30


def _apply_splits(row, overall, vl_key, vr_key, vs_r_weight=_BATTER_VS_RHP_WEIGHT):
    """Return platoon-weighted stat if splits available, else the overall value.

    For batters ``vs_r_weight=0.60`` (face 60% RHP).
    For pitchers ``vs_r_weight=0.55`` (face 55% RHB).
    Falls back gracefully when split columns are absent from the row dict.
    """
    vl = row.get(vl_key)
    vr = row.get(vr_key)
    if vl is not None and vr is not None:
        try:
            vl_f, vr_f = float(vl), float(vr)
            if vl_f > 0 and vr_f > 0:
                return vs_r_weight * vr_f + (1 - vs_r_weight) * vl_f
        except (ValueError, TypeError):
            pass
    return overall


def _calc_platoon_penalty_batting(row):
    """Penalty for extreme batting platoon splits.

    In OOTP PT you set one lineup and face random opponents — you can't
    platoon. A player with 120 contact vs RHP / 40 vs LHP is exposed at
    40 contact ~40% of the time. Below the critical threshold (~50), the
    player is essentially an automatic out against that hand.

    Returns a negative value (penalty) or 0.
    """
    penalty = 0.0
    splits = [
        ('contact', 'contact_vl', 'contact_vr', 1.5),   # contact most important
        ('power', 'power_vl', 'power_vr', 1.2),          # power matters a lot
        ('eye', 'eye_vl', 'eye_vr', 0.5),                # eye less critical
    ]
    for _overall_key, vl_key, vr_key, coeff in splits:
        vl_raw = row.get(vl_key)
        vr_raw = row.get(vr_key)
        if vl_raw is None or vr_raw is None:
            continue
        try:
            vl = float(vl_raw or 0)
            vr = float(vr_raw or 0)
        except (ValueError, TypeError):
            continue
        if vl <= 0 or vr <= 0:
            continue
        weak_side = min(vl, vr)
        if weak_side < _PLATOON_CRITICAL_BAT:
            # Exposure = how often you face the hand you're weak against
            exposure = (1 - _BATTER_VS_RHP_WEIGHT) if vl < vr else _BATTER_VS_RHP_WEIGHT
            penalty -= (_PLATOON_CRITICAL_BAT - weak_side) * exposure * coeff
    return penalty


def _calc_platoon_penalty_pitching(row):
    """Penalty for extreme pitching platoon splits.

    A pitcher with 40 stuff vs LHB gets torched by lefties ~45% of the time.
    Same principle as batting: below the critical threshold you're unplayable
    against that hand, and you can't avoid them in PT.
    """
    penalty = 0.0
    splits = [
        ('stuff', 'stuff_vl', 'stuff_vr', 1.2),
        ('control', 'control_vl', 'control_vr', 0.8),
    ]
    for _overall_key, vl_key, vr_key, coeff in splits:
        vl_raw = row.get(vl_key)
        vr_raw = row.get(vr_key)
        if vl_raw is None or vr_raw is None:
            continue
        try:
            vl = float(vl_raw or 0)
            vr = float(vr_raw or 0)
        except (ValueError, TypeError):
            continue
        if vl <= 0 or vr <= 0:
            continue
        weak_side = min(vl, vr)
        if weak_side < _PLATOON_CRITICAL_PIT:
            exposure = (1 - _PITCHER_VS_RHB_WEIGHT) if vl < vr else _PITCHER_VS_RHB_WEIGHT
            penalty -= (_PLATOON_CRITICAL_PIT - weak_side) * exposure * coeff
    return penalty


def _calc_position_flexibility(row):
    """Bonus for multi-position eligibility.

    OOTP PT rosters have 25 slots. A player who can fill 4 positions is
    far more roster-efficient than a 1B-only player — they cover injuries,
    enable platoon matchups, and allow lineup flexibility. OVR is completely
    blind to this.

    Premium positions (C, SS, CF) get extra credit because covering those
    scarce spots with a bench player saves a roster slot at a premium position.

    Returns 0–35 bonus points.
    """
    POS_COLS = {
        'pos_rating_c':  True,   # premium
        'pos_rating_1b': False,
        'pos_rating_2b': False,
        'pos_rating_3b': False,
        'pos_rating_ss': True,   # premium
        'pos_rating_lf': False,
        'pos_rating_cf': True,   # premium
        'pos_rating_rf': False,
    }
    playable = 0
    premium = 0
    has_any = False
    for col, is_premium in POS_COLS.items():
        rating = row.get(col)
        if rating is None:
            continue
        has_any = True
        try:
            if float(rating or 0) >= _POS_PLAYABLE_THRESHOLD:
                playable += 1
                if is_premium:
                    premium += 1
        except (ValueError, TypeError):
            continue

    if not has_any or playable <= 1:
        return 0.0

    # Scaled down 75% from v5 original — lb124 data shows playable_positions
    # count has r=-0.023 with WAR (no signal). Kept small because multi-position
    # flexibility has real PT roster value even if WAR doesn't capture it.
    bonus = (playable - 1) * 1.5 + premium * 1.0
    return min(bonus, 10.0)


def _calc_arsenal_factor(row, is_sp=True):
    """Bonus for pitch arsenal diversity.

    OOTP gives batters a "familiarity" bonus on 2nd and 3rd trips through
    the order. More quality pitch types = less familiarity penalty for the
    pitcher. This is invisible to Stuff/Movement/Control ratings and to OVR.

    For SP: full bonus (face lineup 3×, diversity critical). Up to +40.
    For RP: reduced bonus (typically face batters once). Up to +25.
    """
    PITCH_TYPES = ['fb', 'ch', 'cb', 'sl', 'si', 'sp',
                   'ct', 'fo', 'cc', 'sc', 'kc', 'kn']

    usable = 0
    quality = 0
    has_any = False
    for p in PITCH_TYPES:
        rating = row.get(p)
        if rating is None:
            continue
        has_any = True
        try:
            val = float(rating or 0)
        except (ValueError, TypeError):
            continue
        if val >= _PITCH_USABLE_THRESHOLD:
            usable += 1
        if val >= _PITCH_QUALITY_THRESHOLD:
            quality += 1

    if not has_any:
        return 0.0

    # Scaled down 60% from v5 original — lb124 data shows pitch_count has
    # r=+0.024 with WAR (no signal). Kept small because arsenal diversity
    # has real theoretical value for SP facing lineups 3× even if the current
    # sample doesn't capture it.
    if is_sp:
        diversity = max(0, usable - 2) * 2.5   # 3→2.5, 4→5, 5→7.5
        quality_bonus = quality * 1.5
        return min(diversity + quality_bonus, 15.0)
    else:
        diversity = max(0, usable - 2) * 1.5
        quality_bonus = quality * 1.0
        return min(diversity + quality_bonus, 10.0)


def _calc_superstat_overlay_batting(row):
    """Adjust meta based on game-log-derived batting signals.

    Separate from ``_calc_performance_adjustment_batting`` (which uses
    wOBA) because game-log signals are INDEPENDENT of rate stats — they
    measure underlying contact quality (EV, LD%) plus plate-discipline
    process (true K% / BB% from pitch-by-pitch data) rather than
    downstream outcomes.

    Inputs (injected onto the row at recalc time; None if no logs):
        ``_gl_ev_avg``       — pooled exit velocity (mph).
        ``_gl_ld_pct``       — line drive rate (0–100).
        ``_gl_k_pct``        — observed strikeout rate (0–100).
        ``_gl_bb_pct``       — observed walk rate (0–100).
        ``_gl_at_bats``      — total pooled game-log PA.

    Scale: capped at ±35 meta. Each component is additive; confidence
    saturates at 200 game-log at-bats. The EV/LD% half captures contact
    quality; the K%/BB% half captures plate discipline — both add real
    info the rating system can miss (especially for cards whose current
    ratings lag their actual in-sim behavior).
    """
    ev_avg = row.get('_gl_ev_avg')
    ld_pct = row.get('_gl_ld_pct')
    k_pct = row.get('_gl_k_pct')
    bb_pct = row.get('_gl_bb_pct')
    ab = row.get('_gl_at_bats') or 0
    if (ev_avg is None and k_pct is None) or ab < 50:
        return 0.0

    try:
        ab = int(ab)
    except (ValueError, TypeError):
        return 0.0

    # League anchors — typical OOTP PT at these tiers
    EV_ANCHOR = 82.0   # mph
    LD_ANCHOR = 28.0   # %
    K_ANCHOR = 22.0    # % — below anchor = good contact, above = whiffy
    BB_ANCHOR = 8.5    # % — above anchor = plus discipline

    raw = 0.0

    # Contact quality block
    if ev_avg is not None:
        try:
            raw += (float(ev_avg) - EV_ANCHOR) * 1.0
        except (ValueError, TypeError):
            pass
    if ld_pct is not None:
        try:
            raw += (float(ld_pct) - LD_ANCHOR) * 1.0
        except (ValueError, TypeError):
            pass

    # Plate-discipline block (inverted: lower K = better, higher BB = better)
    if k_pct is not None:
        try:
            raw += (K_ANCHOR - float(k_pct)) * 0.6
        except (ValueError, TypeError):
            pass
    if bb_pct is not None:
        try:
            raw += (float(bb_pct) - BB_ANCHOR) * 0.8
        except (ValueError, TypeError):
            pass

    confidence = min(math.sqrt(ab / 200.0), 1.0)
    return max(-35.0, min(35.0, raw * confidence))


def _calc_clutch_overlay_batting(row):
    """Adjust batter meta based on situational clutch events from box scores.

    Uses:
        _clutch_2out_rbi       — count of 2-out RBI events (positive clutch)
        _clutch_lob_risp_2out  — count of runners-left-RISP-2-out events (neg clutch)
        _clutch_pa             — denominator (total game-batting PA across instances)

    The net clutch rate is (2OUT_RBI - 0.5 * LOB_RISP_2OUT) / PA_100, scaled
    to meta points. LOB gets 0.5 weight because leaving runners stranded
    isn't 100% the batter's fault (they may have walked or hit a clean
    single that the next hitter failed to drive in) — RBI is a cleaner
    positive signal. Capped at ±20 meta, confidence saturates at 300 PA.
    """
    rbi = row.get('_clutch_2out_rbi') or 0
    lob = row.get('_clutch_lob_risp_2out') or 0
    pa = row.get('_clutch_pa') or 0
    if pa < 100:
        return 0.0
    try:
        rbi = float(rbi); lob = float(lob); pa = float(pa)
    except (ValueError, TypeError):
        return 0.0
    # Net clutch signal per 100 PA
    net = (rbi - 0.5 * lob) * 100.0 / pa
    # League baseline ~ 2-3 net RBI per 100 PA on a Low Bronze team. Anchor
    # at 0 — positive = above league clutch, negative = below.
    confidence = min(math.sqrt(pa / 300.0), 1.0)
    raw = net * 4.0  # 1 net event per 100 PA → 4 meta
    return max(-20.0, min(20.0, raw * confidence))


# Tier-relative observed ISO baselines (pooled lb124 + i76, PA >= 150).
# Comes from analyzing batting_stats per tier: Diamond 0.215 > Gold 0.177 >
# Silver 0.152 > Bronze 0.137 > Regular 0.129 > unrated 0.117.
_TIER_ISO_BASELINES = {
    0: 0.117, 1: 0.129, 2: 0.137, 3: 0.152, 4: 0.177, 5: 0.215,
}


def _calc_observed_defense_overlay_batting(row):
    """Adjust meta based on observed Zone Rating from fielding_stats (Epic B).

    Rating-based defense has r≈0.046 with WAR — essentially noise. Observed
    defensive range (OOTP's Zone Rating, `zr` in fielding_stats) is the
    empirical equivalent of UZR/DRS and carries the real signal. A card
    posting zr=+5 over 81 games has demonstrably added runs; zr=-5 has
    given them back.

    Aggregation (see ingestion.py): games-weighted mean zr across all
    positions played, so a CF-primary who also plays some LF contributes
    both his CF-zr and LF-zr proportional to games. This is fairer than
    picking the primary position's zr alone, especially for utility types.

    Inputs:
        ``_obs_def_zr``    — games-weighted mean Zone Rating (runs above/below)
        ``_obs_def_games`` — total defensive games (confidence)

    Scale: zr=+5 × 1.0 × 81-game confidence = +5 meta. Capped ±10. That
    keeps defense a genuine-but-modest meta contributor — consistent with
    PT sims where offensive variance dominates short series.
    """
    zr = row.get('_obs_def_zr')
    games = row.get('_obs_def_games') or 0
    if zr is None or games is None:
        return 0.0
    try:
        zr = float(zr)
        games = float(games)
    except (ValueError, TypeError):
        return 0.0
    if games < 20:
        return 0.0  # too few games to trust

    # Confidence saturates at 81 games (half a full season).
    confidence = min(math.sqrt(games / 81.0), 1.0)
    # Scale: zr=+5 × 1.0 × full confidence = +5 meta. A full-season +10
    # zr elite defender (rare — top 1%) earns the +10 cap.
    adjustment = zr * 1.0 * confidence
    return max(-10.0, min(10.0, adjustment))


def _calc_iso_overlay_batting(row):
    """Adjust meta based on observed Isolated Power (SLG - AVG).

    Correlation scan (2026-04-17, pooled lb124+i76, n=1105) shows observed
    ISO has r=+0.580 with WAR/600 — stronger than OVR (r=+0.478) and
    stronger than any single rating. The signal is stable because ISO
    saturates quickly (200 PA is already meaningful power evidence).

    **Park adjustment (Epic C, 2026-04-24):** ISO is a power stat — it
    scales directly with the home-park HR factor. A .250 ISO in a hitter's
    park (pf_hr=1.15) is really ~.217 in a neutral park. Dividing the
    observed delta by ``_park_factor_hr`` neutralizes that environment
    before the overlay fires. Neutral pf=1.0 → no change (safe default).

    This is fully additive to rating-based power/gap_power because it's
    a directly-measured sample outcome, not a prediction.

    Uses a PA-weighted, league+tier-adjusted delta computed at ingestion
    time (``_obs_iso_delta``). Each league's tier mean is subtracted
    before pooling, so a card's delta reflects relative power vs its
    same-tier peers in the league it actually played — not a cross-league
    scoring-environment artifact.

    Inputs:
        ``_obs_iso_delta``     — PA-weighted ISO delta vs league+tier mean
        ``_obs_iso_pa``        — total PAs across all leagues
        ``_park_factor_hr``    — home park HR factor (1.0 = neutral)

    Scales with sqrt(PA/300) confidence. Capped ±15.
    """
    delta = row.get('_obs_iso_delta')
    pa = row.get('_obs_iso_pa') or 0
    if delta is None or pa is None:
        return 0.0
    try:
        delta = float(delta)
        pa = float(pa)
    except (ValueError, TypeError):
        return 0.0
    if pa < 100:
        return 0.0  # too few PA to trust

    # Park-neutralize the ISO delta: divide by pf_hr so a hitter in a +15%
    # HR park doesn't get over-credited for power he wouldn't maintain in
    # a neutral park. Clamp the pf to [0.5, 2.0] for hygiene (implausible
    # park factors treated as neutral).
    pf_hr = row.get('_park_factor_hr') or 1.0
    try:
        pf_hr = float(pf_hr)
        if pf_hr <= 0.5 or pf_hr >= 2.0:
            pf_hr = 1.0
    except (ValueError, TypeError):
        pf_hr = 1.0
    if pf_hr != 1.0:
        delta = delta / pf_hr

    confidence = min(math.sqrt(pa / 300.0), 1.0)

    # Scale: 0.050 ISO advantage above peers × 300 × 1.0 confidence = 15 meta.
    # Mid-range magnitude — not so large it overwhelms the rating base,
    # but large enough to meaningfully separate sustained power hitters
    # from their tier peers. Cap ±15 prevents runaway from huge samples.
    adjustment = delta * 300.0 * confidence
    return max(-15.0, min(15.0, adjustment))


def calc_mix_diagnostic(row: dict, side: str = 'batting') -> dict:
    """Compute mix-balance diagnostics for a card. NOT a meta formula term.

    2026-04-20 residual analysis showed that mix features (min_top3,
    count_>=75, mean_top4) are already ABSORBED by the current meta formula
    — adding them as additive terms would hurt (ridge coef went negative).
    However, the raw mix profile carries signal that users want to SEE
    alongside meta: is this card a balanced 80-across or a one-trick 95/50?

    Treat this like OVR: a separate diagnostic column displayed next to
    meta, useful for archetype matching and replacement search, but NOT
    folded into calc_*_meta.

    Returns a dict with:
        core_max:         max of the core rating group
        core_min:         min (weakest core)
        core_mean:        average core rating
        min_top3:         3rd-best core rating (mix floor)
        mean_top4:        average of the best 4 (mix ceiling)
        count_elite:      # ratings >= 80
        count_solid:      # ratings >= 70
        spread:           max - min (one-trick → high spread)
        mix_score:        [0-100] scaled blend of min_top3 + count_elite,
                           higher = more balanced elite-across-the-board

    ``side`` selects which ratings go into 'core':
        'batting'  → [contact, gap_power, power, eye, avoid_ks, babip]
        'pitching' → [stuff, movement, control, p_hr, p_babip]
    """
    if side == 'pitching':
        core_keys = ['stuff', 'movement', 'control', 'p_hr', 'p_babip']
    else:
        core_keys = ['contact', 'gap_power', 'power', 'eye', 'avoid_ks', 'babip']

    vals = []
    for k in core_keys:
        v = row.get(k)
        try:
            vals.append(float(v) if v is not None else 0.0)
        except (ValueError, TypeError):
            vals.append(0.0)

    if not vals or all(v == 0 for v in vals):
        return {
            'core_max': 0, 'core_min': 0, 'core_mean': 0,
            'min_top3': 0, 'mean_top4': 0,
            'count_elite': 0, 'count_solid': 0,
            'spread': 0, 'mix_score': 0,
        }

    sorted_desc = sorted(vals, reverse=True)
    min_top3 = sorted_desc[2] if len(sorted_desc) >= 3 else sorted_desc[-1]
    top4 = sorted_desc[:4]
    mean_top4 = sum(top4) / len(top4)
    count_elite = sum(1 for v in vals if v >= 80)
    count_solid = sum(1 for v in vals if v >= 70)
    core_max = max(vals); core_min = min(vals); core_mean = sum(vals)/len(vals)

    # Blended mix score 0-100:
    #   min_top3 maps 50..95 -> 0..60 (balance floor)
    #   count_elite maps 0..6 -> 0..40 (how many truly elite ratings)
    #   cap at 100.
    mix_balance = max(0.0, min(60.0, (min_top3 - 50) / 45.0 * 60.0))
    mix_ceiling = max(0.0, min(40.0, count_elite / 6.0 * 40.0))
    mix_score = round(mix_balance + mix_ceiling, 1)

    return {
        'core_max': round(core_max, 1),
        'core_min': round(core_min, 1),
        'core_mean': round(core_mean, 1),
        'min_top3': round(min_top3, 1),
        'mean_top4': round(mean_top4, 1),
        'count_elite': count_elite,
        'count_solid': count_solid,
        'spread': round(core_max - core_min, 1),
        'mix_score': mix_score,
    }


def _calc_iso_gap_overlay_batting(row):
    """Reward/penalize cards based on observed ISO vs power-rating prediction.

    Distinct from ``_calc_iso_overlay_batting`` (which compares to tier-peer
    baseline). This overlay compares observed ISO to what the card's POWER
    RATING predicts — catching "paper-tiger sluggers" whose power rating
    promises slugging they don't deliver, and rewarding "sneaky power" cards
    that out-slug their rating.

    Residual analysis (2026-04-20, n=1903 PA>=250) showed:
      - iso_gap vs WAR residual: r=+0.466 (biggest single untapped signal)
      - Adding this to meta lifts overall r from +0.7356 to +0.8153 (+0.08)

    The gap is computed at ingestion time by regressing obs ISO on power
    rating (``_obs_iso_gap`` / ``_obs_iso_gap_pa``). Positive gap = card
    over-delivered its power rating; negative = card under-delivered.

    Scale: 0.050 ISO gap * 200 * confidence = 10 meta points at 1.0 conf.
    Confidence saturates at 400 PA. Capped ±25 (larger than tier-relative
    ISO overlay because the signal is stronger).
    """
    gap = row.get('_obs_iso_gap')
    pa = row.get('_obs_iso_gap_pa') or 0
    if gap is None or pa is None:
        return 0.0
    try:
        gap = float(gap); pa = float(pa)
    except (ValueError, TypeError):
        return 0.0
    if pa < 150:
        return 0.0  # below stable sample threshold

    confidence = min(math.sqrt(pa / 400.0), 1.0)
    # Scale 200 chosen from regression slope: iso_gap has ~22.7 WAR/600 per
    # ISO-unit coefficient, and inverse slope is ~78 meta per WAR/600.
    # 0.050 gap * 22.7 WAR * 78 meta/WAR = 88 meta — cap hard at ±25 so a
    # single ISO reading doesn't swamp the ratings base.
    adjustment = gap * 200.0 * confidence
    return max(-25.0, min(25.0, adjustment))


def _calc_rating_diminishing_penalty_batting(row):
    """Penalize cards with multiple elite ratings (true diminishing returns).

    Meta-correlation scan (2026-04-17) found that ``contact × power`` has
    r=-0.439 with meta residuals: cards with BOTH high contact AND high
    power are systematically over-predicted. The rating-based meta adds
    contact and power linearly, but a card can only physically collect so
    many WAR per 600 PA — you can't hit 50 HRs AND .340 at the same time
    at the level the additive meta implies.

    Applied as a small negative multiplicative interaction for the top-tier
    stat combinations:
      - contact × power (elite quality cards)
      - gap_power × power (elite extra-base hitters)

    Returns a negative value; magnitude proportional to how far above the
    elite-threshold (85) each stat is.
    """
    con = float(row.get('contact') or 0)
    pwr = float(row.get('power') or 0)
    gap = float(row.get('gap_power') or 0)

    penalty = 0.0
    # contact × power — punish dual-elite (residual r=-0.44)
    if con >= 70 and pwr >= 70:
        con_ex = max(0, con - 70)
        pwr_ex = max(0, pwr - 70)
        penalty -= (con_ex * pwr_ex) * 0.018  # was 0.010
    # gap_power × power (residual -0.25 area)
    if gap >= 70 and pwr >= 70:
        gap_ex = max(0, gap - 70)
        pwr_ex = max(0, pwr - 70)
        penalty -= (gap_ex * pwr_ex) * 0.014  # was 0.008
    # Standalone power damping — linear across the full distribution.
    # Residual -0.30 for power alone; strengthened to address that.
    if pwr > 60:
        penalty -= (pwr - 60) * 0.20
    # Cap at -40 to avoid runaway on 99/99/99 cards (was -25)
    return max(-40.0, penalty)


def _calc_babip_overlay_batting(row):
    """Reward observed BABIP above league baseline.

    Residual analysis (2026-04-17): obs BABIP has r=+0.26 with meta
    residuals after ISO/OPS+/OBP/overperf overlays. Captures hit-quality
    signal distinct from slugging/on-base — fast runners beating out
    grounders, hard-hit balls finding holes. The BABIP rating captures
    expected; the observed residual captures SUSTAINED over-performance
    on balls in play.

    Inputs:
        ``_obs_babip_delta`` — PA-weighted BABIP delta vs league baseline
        ``_obs_babip_pa``    — total PAs

    Scales with sqrt(PA/400) confidence. Capped ±12 (BABIP is noisier
    than OBP/OPS+ so we use a tighter cap).
    """
    delta = row.get('_obs_babip_delta')
    pa = row.get('_obs_babip_pa') or 0
    if delta is None or pa is None:
        return 0.0
    try:
        delta = float(delta); pa = float(pa)
    except (ValueError, TypeError):
        return 0.0
    if pa < 100:
        return 0.0

    confidence = min(math.sqrt(pa / 400.0), 1.0)
    # Scale: 0.040 BABIP advantage × 200 × 1.0 = 8 meta. BABIP ranges
    # 0.250-0.350 typically, so this is the realistic spread.
    adjustment = delta * 200.0 * confidence
    return max(-12.0, min(12.0, adjustment))


def _calc_overperformance_overlay_batting(row):
    """Reward cards whose observed OPS+ exceeds what their ratings predict.

    Meta-correlation scan (2026-04-17): ``obs_ops_plus`` regressed onto a
    rating composite (contact+power+gap_power) gives an overperformance
    residual with r=+0.551 against meta residuals — signal that persists
    BEYOND the OPS+ delta overlay. Captures cards that systematically
    outperform their rating profile (e.g. "plays up" intangibles, favorable
    matchups, OOTP hidden attributes).

    Expected OPS+ is computed at ingestion time by linear regression of
    observed OPS+ on the rating composite, pooled across leagues; the
    residual (obs - expected) is injected as ``_over_ops_plus``.

    Inputs:
        ``_over_ops_plus``    — observed - rating-predicted OPS+
        ``_over_ops_plus_pa`` — total PA across leagues

    Scales with sqrt(PA/400) confidence. Capped ±20.
    """
    over = row.get('_over_ops_plus')
    pa = row.get('_over_ops_plus_pa') or 0
    if over is None or pa is None:
        return 0.0
    try:
        over = float(over)
        pa = float(pa)
    except (ValueError, TypeError):
        return 0.0
    if pa < 100:
        return 0.0

    confidence = min(math.sqrt(pa / 400.0), 1.0)

    # Scale: 30 OPS+ above rating-predicted × 0.5 × 1.0 = 15 meta points.
    # Conservative — OPS+ and OBP overlays already absorb the league-relative
    # component; this captures the pure rating-residual overperformance.
    adjustment = over * 0.5 * confidence
    return max(-20.0, min(20.0, adjustment))


def _calc_obp_overlay_batting(row):
    """Adjust meta based on observed On-Base Percentage.

    Residual analysis (2026-04-17) after OPS+/ISO overlays: OBP has
    r=+0.337 with remaining residuals, making it the single biggest
    additional signal. OPS+ absorbs some OBP (OPS+ = OBP + SLG relative
    to league), but observed OBP standalone still has strong predictive
    power for WAR above what meta currently captures.

    Uses league-specific baseline to handle scoring-environment differences.
    Computed with PA-weighted delta at ingestion time.

    Inputs:
        ``_obs_obp_delta`` — PA-weighted OBP delta vs league baseline
        ``_obs_obp_pa``    — total PAs across all leagues

    Scales with sqrt(PA/400) confidence. Capped ±20.
    """
    delta = row.get('_obs_obp_delta')
    pa = row.get('_obs_obp_pa') or 0
    if delta is None or pa is None:
        return 0.0
    try:
        delta = float(delta)
        pa = float(pa)
    except (ValueError, TypeError):
        return 0.0
    if pa < 100:
        return 0.0

    confidence = min(math.sqrt(pa / 400.0), 1.0)

    # Scale: 0.050 OBP above avg × 300 × 1.0 confidence = 15 meta points.
    # Conservative because OPS+ overlay already absorbs a portion of this.
    adjustment = delta * 300.0 * confidence
    return max(-20.0, min(20.0, adjustment))


def _calc_opsplus_overlay_batting(row):
    """Adjust meta based on observed OPS+ (park- and league-adjusted).

    Residual analysis (2026-04-17, pooled lb124+i76, n=1105): OPS+ has
    r=+0.325 with meta residuals, meaning it carries strong signal that
    the rating-based meta and wOBA overlay are missing. OPS+ is centered
    at 100 (league average) and already park-adjusted, so it naturally
    crosses league boundaries.

    Partially overlaps with wOBA overlay (both measure offense), but the
    residual correlation proves meaningful additive signal remains.

    Inputs:
        ``_obs_ops_plus``    — PA-weighted OPS+ across all league instances
        ``_obs_ops_plus_pa`` — total PAs in that sample

    Scales with sqrt(PA/400) confidence. Capped ±25 to avoid overwhelming
    the rating base.
    """
    ops_plus = row.get('_obs_ops_plus')
    pa = row.get('_obs_ops_plus_pa') or 0
    if ops_plus is None or pa is None:
        return 0.0
    try:
        ops_plus = float(ops_plus)
        pa = float(pa)
    except (ValueError, TypeError):
        return 0.0
    if pa < 100:
        return 0.0

    # OPS+ = 100 at league average by definition
    delta = ops_plus - 100.0
    confidence = min(math.sqrt(pa / 400.0), 1.0)

    # Scale: 40 OPS+ above avg × 0.6 × 1.0 = 24 meta points.
    # Calibrated conservatively because wOBA overlay already absorbs
    # part of this signal; keep magnitude sub-wOBA (±40) to avoid
    # double-counting runaway.
    adjustment = delta * 0.6 * confidence
    return max(-25.0, min(25.0, adjustment))


def _calc_error_overlay_batting(row):
    """Adjust batter meta based on observed error rate vs position expectation.

    OOTP exports defensive ratings, but they're predictions. Observed error
    rate from box scores is a direct measurement. For positions with ≥20
    games of observed play, we compare error-per-game to a position
    expectation and adjust meta.

    Position-expected error rate per game (rough baselines from MLB):
        SS  0.13   2B  0.09   3B  0.12   1B  0.05
        LF  0.04   CF  0.04   RF  0.04   C   0.06

    Inputs:
        _errors_count      — total errors observed
        _def_games_count   — games the player appeared defensively
        position           — already on row
    """
    games = row.get('_def_games_count') or 0
    if games < 20:
        return 0.0
    errs = row.get('_errors_count') or 0
    try:
        games = int(games); errs = int(errs)
    except (ValueError, TypeError):
        return 0.0
    pos = row.get('position') or 0
    if isinstance(pos, str):
        pos_map = {"C": 2, "1B": 3, "2B": 4, "3B": 5, "SS": 6,
                   "LF": 7, "CF": 8, "RF": 9, "DH": 10}
        pos = pos_map.get(pos, 0)
    try:
        pos = int(pos)
    except (ValueError, TypeError):
        pos = 0
    # Expected errors per game by position (see docstring)
    _EXPECTED = {2: 0.06, 3: 0.05, 4: 0.09, 5: 0.12, 6: 0.13,
                 7: 0.04, 8: 0.04, 9: 0.04, 10: 0.0}
    expected = _EXPECTED.get(pos, 0.07)
    actual = errs / games
    # Difference: positive means MORE errors than expected → penalty.
    # Scale: 0.1 excess errors per game × 100 = 10 meta penalty. Cap ±15.
    delta = (expected - actual) * 100.0
    confidence = min(math.sqrt(games / 60.0), 1.0)
    return max(-15.0, min(15.0, delta * confidence))


def _calc_superstat_overlay_pitching(row):
    """Adjust pitcher meta based on game-log-allowed signals.

    Mirror of the batting overlay, from the pitcher's side. Pitchers who
    suppress exit velocity and line drives are genuinely good even when
    their ERA looks ordinary — this is the OOTP analogue of "hidden"
    run-prevention skill the rating system doesn't fully price in.

    Inputs (injected at recalc time):
        ``_gl_p_ev_allowed``       — avg EV on balls put in play.
        ``_gl_p_ld_pct_allowed``   — line drive % on balls in play.
        ``_gl_p_k_pct``            — observed strikeout rate (batters faced).
        ``_gl_p_bb_pct``           — observed walk rate.
        ``_gl_p_batters_faced``    — sample size.

    Scale: capped at ±35 meta. Inverted signs vs batting (low EV allowed
    = good for pitchers). Confidence saturates at 300 BF (pitchers see
    more batters per game than batters see pitchers).
    """
    ev_allowed = row.get('_gl_p_ev_allowed')
    ld_pct_allowed = row.get('_gl_p_ld_pct_allowed')
    k_pct = row.get('_gl_p_k_pct')
    bb_pct = row.get('_gl_p_bb_pct')
    bf = row.get('_gl_p_batters_faced') or 0
    if (ev_allowed is None and k_pct is None) or bf < 75:
        return 0.0

    try:
        bf = int(bf)
    except (ValueError, TypeError):
        return 0.0

    # Same anchors as batting — we're measuring the same physical thing
    # from the pitcher's side.
    EV_ANCHOR = 82.0
    LD_ANCHOR = 28.0
    K_ANCHOR = 22.0
    BB_ANCHOR = 8.5

    raw = 0.0

    # Contact suppression (inverted signs: low EV allowed = good)
    if ev_allowed is not None:
        try:
            raw += (EV_ANCHOR - float(ev_allowed)) * 1.0
        except (ValueError, TypeError):
            pass
    if ld_pct_allowed is not None:
        try:
            raw += (LD_ANCHOR - float(ld_pct_allowed)) * 1.0
        except (ValueError, TypeError):
            pass

    # Rate stats (pitcher perspective: high K good, low BB good)
    if k_pct is not None:
        try:
            raw += (float(k_pct) - K_ANCHOR) * 0.6
        except (ValueError, TypeError):
            pass
    if bb_pct is not None:
        try:
            raw += (BB_ANCHOR - float(bb_pct)) * 0.8
        except (ValueError, TypeError):
            pass

    confidence = min(math.sqrt(bf / 300.0), 1.0)
    return max(-35.0, min(35.0, raw * confidence))


def _calc_bb9_overlay_pitching(row):
    """Penalize observed walk rate above league baseline.

    Residual analysis (2026-04-17): bb_per_9 has r=-0.34 with meta
    residuals even after FIP overlay. FIP weights walks (3×BB) but the
    residual says walks are STILL undercharged — likely because FIP
    averages high-K-high-BB arms to middling, and real WAR punishes
    walks harder than strikeouts reward.

    Inputs:
        ``_obs_bb9_delta`` — BF-weighted BB/9 delta vs league baseline
        ``_obs_bb9_bf``    — total BF

    Scales with sqrt(BF/300) confidence. Capped ±20.
    """
    delta = row.get('_obs_bb9_delta')
    bf = row.get('_obs_bb9_bf') or 0
    if delta is None or bf is None:
        return 0.0
    try:
        delta = float(delta); bf = float(bf)
    except (ValueError, TypeError):
        return 0.0
    if bf < 100:
        return 0.0

    confidence = min(math.sqrt(bf / 300.0), 1.0)
    # Scale: 1.0 BB/9 above league × -10 × 1.0 = -10 meta.
    # Negative coefficient — higher BB/9 is worse. Capped ±20.
    adjustment = -delta * 10.0 * confidence
    return max(-20.0, min(20.0, adjustment))


def _calc_overperformance_overlay_pitching(row):
    """Reward pitchers whose observed ERA+ exceeds rating-predicted.

    Parallel to batting's over-performance overlay. Meta-correlation scan
    shows that pitchers with strong rating composites (stuff+movement+
    control) still vary meaningfully in observed ERA+, and the residual
    overperformance carries signal beyond SIERA/FIP overlays.

    Inputs:
        ``_over_era_plus``    — observed - rating-predicted ERA+
        ``_over_era_plus_bf`` — total BF faced

    Scales with sqrt(BF/300) confidence. Capped ±20.
    """
    over = row.get('_over_era_plus')
    bf = row.get('_over_era_plus_bf') or 0
    if over is None or bf is None:
        return 0.0
    try:
        over = float(over)
        bf = float(bf)
    except (ValueError, TypeError):
        return 0.0
    if bf < 100:
        return 0.0

    confidence = min(math.sqrt(bf / 300.0), 1.0)

    # Scale: 30 ERA+ above rating-predicted × 0.4 × 1.0 = 12 meta points.
    # Smaller than batting overperf because pitching has more noise per PA.
    adjustment = over * 0.4 * confidence
    return max(-20.0, min(20.0, adjustment))


# ════════════════════════════════════════════════════════════════════════════
# RECENT-FORM OVERLAYS (2026-04-24)
#
# All other overlays freeze at last ingest, blending the player's full season
# (or career across owners) into one number. That means a card that's been
# white-hot or ice-cold over the last 30 days carries the SAME meta as the
# day before its hot/cold streak began. The Roster Optimizer can show a
# "Recent Performance" panel and an Outlook (hot/cold) label, but neither
# nudges meta — so a cold high-meta starter still ranks "Optimal" and locks
# out genuine bench upgrades that have heated up.
#
# These overlays close the gap with a small, sample-weighted recency nudge:
# observed last-30-day production minus the season baseline, scaled and
# capped so it can shift meta within a band but never override the rating
# foundation.
#
# Inputs come from `recent_form_batting` / `recent_form_pitching` aggregates
# built in `recalculate_all_meta_scores`. Per-league windows are used so
# multi-league users don't cross-contaminate.
# ════════════════════════════════════════════════════════════════════════════

# Sample-size gates for the recent-form overlay. Below these, the signal is
# too noisy to nudge meta. Confidence saturates near these saturation points.
_RECENT_FORM_MIN_PA = 25
_RECENT_FORM_PA_SATURATION = 100
_RECENT_FORM_MIN_IP = 10
_RECENT_FORM_IP_SATURATION = 30


def _calc_recent_form_overlay_batting(row):
    """Nudge meta toward recent-30-day OBP performance vs season baseline.

    Inputs (set by ingestion pre-scan):
        ``_rf_pa``          — total PA in the last-30-day window
        ``_rf_obp_proxy``   — (h+bb)/(ab+bb) over the window
        ``_rf_season_obp``  — PA-weighted season OBP across all card instances

    Why OBP-proxy and not OPS+: ``game_batting`` only carries ab/h/bb/k per
    game (no doubles/triples/HR), so SLG can't be reconstructed at the game
    level. OBP is robust at the small samples we typically see in 30 days
    and directly comparable to season OBP.

    Scale: 0.060 OBP swing × 200 × full-confidence = 12 meta. Cap ±15 so the
    overlay can shift a borderline upgrade decision but never dominate
    ratings + season-long observed signals.
    """
    pa = row.get('_rf_pa') or 0
    rec_obp = row.get('_rf_obp_proxy')
    season_obp = row.get('_rf_season_obp')
    if not pa or rec_obp is None or season_obp is None:
        return 0.0
    try:
        pa = float(pa)
        rec_obp = float(rec_obp)
        season_obp = float(season_obp)
    except (ValueError, TypeError):
        return 0.0
    if pa < _RECENT_FORM_MIN_PA or season_obp <= 0:
        return 0.0

    confidence = min(math.sqrt(pa / _RECENT_FORM_PA_SATURATION), 1.0)
    delta = rec_obp - season_obp
    adjustment = delta * 200.0 * confidence
    return max(-15.0, min(15.0, adjustment))


def _calc_recent_form_overlay_pitching(row):
    """Nudge meta toward recent-30-day ERA performance vs season baseline.

    Inputs (set by ingestion pre-scan):
        ``_rf_ip``         — innings pitched in the last-30-day window
        ``_rf_era``        — ERA over the window
        ``_rf_season_era`` — IP-weighted season ERA across all card instances

    Sign convention: positive adjustment when recent ERA is BETTER (lower)
    than season — a guy turning a corner gets meta nudged up, a guy melting
    down gets nudged down. ERA is noisy at low IP, so the gate is 10 IP
    minimum and confidence saturates at 30 IP.

    Scale: 1.50 ERA gap × 6 × full-confidence = 9 meta. Cap ±15.
    """
    ip = row.get('_rf_ip') or 0
    rec_era = row.get('_rf_era')
    season_era = row.get('_rf_season_era')
    if not ip or rec_era is None or season_era is None:
        return 0.0
    try:
        ip = float(ip)
        rec_era = float(rec_era)
        season_era = float(season_era)
    except (ValueError, TypeError):
        return 0.0
    if ip < _RECENT_FORM_MIN_IP or season_era <= 0:
        return 0.0

    confidence = min(math.sqrt(ip / _RECENT_FORM_IP_SATURATION), 1.0)
    # Positive delta = recent better than season.
    delta = season_era - rec_era
    adjustment = delta * 6.0 * confidence
    return max(-15.0, min(15.0, adjustment))


def _calc_stamina_bonus_pitching(row):
    """Linear stamina bonus for starters.

    Calibrated stamina_hold weight is 0.10 but residual r=+0.38 showed
    stamina severely under-weighted for SPs. Initial correction at 0.3/pt
    still left residual +0.38; bumped to 0.6/pt (stamina=100 → +30).

    Relievers excluded — they pitch 1-inning bursts, stamina irrelevant.
    """
    if not _is_starting_pitcher(row):
        return 0.0
    stamina = float(row.get('stamina') or 0)
    if stamina <= 0:
        return 0.0
    # +0.6 per stamina point above 50, cap at +30 (stamina=100 → +30).
    bonus = max(0, stamina - 50) * 0.6
    return min(30.0, bonus)


def _calc_stuff_overcredit_damping_pitching(row):
    """Linear damping to correct stuff over-weighting residual r=-0.52.

    Even after dual-elite diminishing, stuff residual persisted at r=-0.52.
    Initial correction at 0.10/pt didn't dent it; bumped to 0.25/pt.
    Calibration assigned stuff=2.04 but observed WAR shows diminishing
    returns across the full distribution, not just elite.
    """
    stu = float(row.get('stuff') or 0)
    if stu <= 60:
        return 0.0
    # -0.25 per stuff point above 60, cap at -12.
    return -min(12.0, (stu - 60) * 0.25)


def _calc_stuff_diminishing_pitching(row):
    """Penalize cards with elite stuff × movement (fix additive over-credit).

    Residuals show stuff alone has r=-0.519 with residuals — the meta
    weights stuff (2.20) too heavily at the top end. A 99-stuff pitcher
    doesn't produce proportionally more WAR than an 85-stuff one once
    the batter quality ceiling is reached.

    Returns negative; applies only to high-rating combinations.
    """
    stu = float(row.get('stuff') or 0)
    mov = float(row.get('movement') or 0)
    ctrl = float(row.get('control') or 0)

    penalty = 0.0
    if stu >= 80 and mov >= 70:
        stu_ex = max(0, stu - 80)
        mov_ex = max(0, mov - 70)
        penalty -= (stu_ex * mov_ex) * 0.015
    if stu >= 85 and ctrl >= 70:
        stu_ex = max(0, stu - 85)
        ctrl_ex = max(0, ctrl - 70)
        penalty -= (stu_ex * ctrl_ex) * 0.010

    return max(-30.0, penalty)


def _calc_fip_overlay_pitching(row):
    """Adjust meta based on observed FIP (Fielding Independent Pitching).

    Residual analysis (2026-04-17, pooled lb124+i76, n=1229): FIP has
    r=-0.359 with meta residuals, meaning it carries strong signal that
    SIERA overlay and rating-based meta are missing. FIP weights HR
    allowed heavily (13×HR), which SIERA de-emphasizes — so FIP
    captures the HR-per-9 residual (r=-0.414 with residuals) that is
    currently the biggest pitching prediction gap.

    Uses league-specific baseline to avoid scoring-environment bias:
    lb124 avg FIP 4.05, i76 avg 4.14. PA-weighted delta is computed
    at ingestion time.

    Inputs:
        ``_obs_fip_delta`` — IP-weighted FIP delta vs league baseline
        ``_obs_fip_bf``    — total batters faced (for confidence)

    Scales with sqrt(BF/300) confidence. Capped ±25 (lower than FIP is
    better — we INVERT the delta when scoring).
    """
    delta = row.get('_obs_fip_delta')
    bf = row.get('_obs_fip_bf') or 0
    if delta is None or bf is None:
        return 0.0
    try:
        delta = float(delta)
        bf = float(bf)
    except (ValueError, TypeError):
        return 0.0
    if bf < 100:
        return 0.0

    # Park adjustment (Epic C, 2026-04-24): FIP is HR-weighted (13×HR in
    # the formula), so a pitcher at a hitter-friendly park (pf_hr > 1.0)
    # has their FIP inflated by environment. Multiply the delta by pf_hr
    # to neutralize — a pitcher who posted 3.80 FIP in a +15% HR park is
    # really putting up ~3.30 FIP-neutral, which should register as a
    # bigger gap below league (-0.70 vs observed -0.20). Clamp to [0.5,
    # 2.0] for hygiene.
    pf_hr = row.get('_park_factor_hr') or 1.0
    try:
        pf_hr = float(pf_hr)
        if pf_hr <= 0.5 or pf_hr >= 2.0:
            pf_hr = 1.0
    except (ValueError, TypeError):
        pf_hr = 1.0
    if pf_hr != 1.0:
        delta = delta * pf_hr

    confidence = min(math.sqrt(bf / 300.0), 1.0)

    # Scale: 1.0 FIP below league avg × scale × 1.0 confidence.
    # Negative coefficient because lower FIP is better. Scale history:
    #   v1: 15  → residual still −0.324, undersized.
    #   v2: 25  → residual dropped to −0.30 post-overlay.
    #   v3: 30  (2026-04-18) — bumped for combined pitching r=0.40 ceiling.
    #   v4: SP-specific 45, RP-specific 30 (2026-04-24) — SP residual analysis
    #       (n=1834) found FIP_obs r=-0.644 vs WAR/200 residuals AFTER all
    #       prior overlays. SPs face 28 BF/start so 5+ starts saturates the
    #       confidence weight; the prior cap at ±40 was being hit on tail
    #       cases without clearing enough of the residual. SPs get 50% more
    #       FIP weight + a wider cap; RPs unchanged because their FIP
    #       residual is much smaller and the existing pair-interaction
    #       overlay already extracts the role-specific signal.
    role = (row.get('pitcher_role_name') or '').upper()
    is_sp_role = role.startswith('SP')
    # SP FIP scale bumped 45 → 65 after first-pass measure showed residual
    # only fell from -0.644 to -0.611 (+0.029 project-wide r). Raising scale
    # absorbs more remaining FIP signal; cap also bumped 60 → 80 so true
    # outlier FIP gaps (1.5+ above league) don't get clipped.
    scale = 65.0 if is_sp_role else 30.0
    cap = 80.0 if is_sp_role else 40.0
    adjustment = -delta * scale * confidence
    return max(-cap, min(cap, adjustment))


def _calc_hr9_overlay_pitching(row):
    """SP HR/9 overlay — captures HR-rate signal beyond what FIP absorbs.

    SP residual analysis (2026-04-24, n=1834): observed HR/9 still has
    r=-0.614 with WAR/200 residuals AFTER the FIP overlay fires. FIP
    weights HR at 13× but is diluted by K/BB; in OOTP PT the gap between
    a 0.7 HR/9 and 1.5 HR/9 SP is wider than FIP captures because lineups
    cycle 3+ times and the second-time-through power penalty compounds.

    Inputs (computed in the recent-form pre-scan and reused here):
        ``_obs_hr9_delta`` — IP-weighted HR/9 delta vs league baseline
        ``_obs_hr9_ip``    — total IP across all instances

    SP-only because:
      - RP HR/9 already absorbed by `_calc_pair_interaction_overlay_pitching`
        and `_calc_reliever_discipline_overlay_pitching`
      - RP samples are tiny (avg 50 IP) so HR/9 is too noisy to act on
      - SP-specific residual analysis showed the signal concentrates on SPs

    Scale: 0.50 HR/9 above league × -8 × 1.0 = -4 meta. Cap ±20 so it can
    matter without dominating FIP.
    """
    role = (row.get('pitcher_role_name') or '').upper()
    if not role.startswith('SP'):
        return 0.0
    delta = row.get('_obs_hr9_delta')
    ip = row.get('_obs_hr9_ip') or 0
    if delta is None or ip is None:
        return 0.0
    try:
        delta = float(delta)
        ip = float(ip)
    except (ValueError, TypeError):
        return 0.0
    if ip < 30:
        return 0.0

    # Park adjustment — same rationale as FIP overlay. HR/9 is the cleanest
    # signal in a homer-friendly park and we want to credit suppression
    # achieved against environment, not get fooled by it.
    pf_hr = row.get('_park_factor_hr') or 1.0
    try:
        pf_hr = float(pf_hr)
        if pf_hr <= 0.5 or pf_hr >= 2.0:
            pf_hr = 1.0
    except (ValueError, TypeError):
        pf_hr = 1.0
    if pf_hr != 1.0:
        delta = delta * pf_hr

    confidence = min(math.sqrt(ip / 100.0), 1.0)
    # Scale 12 (was 8 in v1, bumped after first-pass results showed HR/9
    # residual only dropped -0.614 → -0.583). 0.50 HR/9 above league × -12
    # × 1.0 = -6 meta. Cap ±25 (was ±20).
    adjustment = -delta * 12.0 * confidence
    return max(-25.0, min(25.0, adjustment))


def _calc_pair_interaction_overlay_pitching(row):
    """Role-specific pair-interaction overlay for relievers.

    Residual analysis (2026-04-20, n=1142 RP IP>=30) found two RP-specific
    rating pairs carrying independent signal AFTER the current meta:
      - (control-70)*(p_hr-70) — residual r=+0.180  "command fireman"
      - (movement-70)*(control-70) — residual r=+0.177  "junk-ball precision"

    These same pairs have near-zero residual for starters (r=-0.029, -0.012),
    so the overlay is RP-GATED. It also zeros out when either stat is below
    70 — the signal is specifically about dual-elite command/HR-suppression,
    not about two mediocre ratings multiplying each other.

    Scale: 0.0007 per unit of (a-70)*(b-70) matches the slopes from the
    calibration regression (meta + (ctl-70)*(p_hr-70) gave slope 0.00067).
    Capped +15 so a 90/90/90 RP doesn't runaway.

    Returns 0 for SPs, swing-men, and RPs where both ratings don't clear 70.
    """
    role = (row.get('pitcher_role_name') or '').upper()
    if role.startswith('SP'):
        return 0.0
    # Accept all non-SP roles: RP, CL, SU, MR, LR, SW (swing-man still gets it
    # since leverage distribution is closer to RP than SP in PT sim).
    if not role or role.startswith(('SP',)):
        return 0.0

    control = float(row.get('control') or 0)
    p_hr = float(row.get('p_hr') or 0)
    movement = float(row.get('movement') or 0)

    adjustment = 0.0
    # "Command fireman" — ctl+HR-suppression
    if control >= 70 and p_hr >= 70:
        adjustment += (control - 70) * (p_hr - 70) * 0.0007
    # "Junk-ball precision" — movement+control
    if movement >= 70 and control >= 70:
        adjustment += (movement - 70) * (control - 70) * 0.0009

    # Cap to prevent a single extreme card from dominating meta
    return max(0.0, min(15.0, adjustment))


def _calc_reliever_discipline_overlay_pitching(row):
    """Adjust reliever meta based on inherited-runner-scored rate.

    ERA doesn't charge a reliever for runners they inherit — a reliever who
    enters with bases loaded and gives up a 2-run single stays clean on ERA.
    Observed inherited-score-rate exposes that leak.

    Inputs:
        _inherited_runners  — total runners inherited across all team instances
        _inherited_scored   — how many of those runners scored

    MLB baseline inherited-runner-scored rate is ~30%. Below that = good
    stranding; above = poor. Capped at ±18 meta, confidence saturates at
    25 inherited runners.
    """
    inh = row.get('_inherited_runners') or 0
    sc = row.get('_inherited_scored') or 0
    if inh < 5:
        return 0.0
    try:
        inh = int(inh); sc = int(sc)
    except (ValueError, TypeError):
        return 0.0
    rate = sc / inh
    # Baseline 0.30 → 0% delta. Rate 0.10 → +20pp below baseline → boost.
    # Rate 0.60 → -30pp above baseline → penalty.
    delta_pp = (0.30 - rate) * 100.0
    confidence = min(math.sqrt(inh / 25.0), 1.0)
    # Scale: 10pp off baseline = 6 meta points
    return max(-18.0, min(18.0, delta_pp * 0.6 * confidence))


def _calc_performance_adjustment_batting(row):
    """Adjust meta based on actual wOBA performance, park-adjusted.

    A .370 wOBA in a hitter's park (pf_hr=1.15, pf_avg=1.10) is less
    impressive than .370 in a neutral park. When ``_park_factor_bat`` is
    on the row (typically injected from league_team_stats), we divide the
    observed wOBA by it before computing the gap vs league. Missing park
    factor = 1.0 (neutral), so the overlay still fires on cards without
    a park join — just without the correction.

    Capped at ±40. wOBA r=+0.884 with WAR/600 in lb124 — the strongest
    single signal in the dataset. Higher cap than rating layers because
    actual performance IS value.
    """
    woba = row.get('adv_woba')
    pa = row.get('adv_pa')
    if not woba or not pa:
        return 0.0
    try:
        woba = float(woba)
        pa = float(pa)
    except (ValueError, TypeError):
        return 0.0
    if woba <= 0 or pa < _PERF_MIN_PA:
        return 0.0

    # Park adjustment — normalize observed wOBA to a neutral park before
    # comparing to league average. Try the overall batting park factor first,
    # then an explicit HR factor (cleaner signal for power bats), then fall
    # back to neutral 1.0 so cards without a park join still get the overlay.
    pf = row.get('_park_factor_bat') or row.get('pf_overall') or row.get('pf_avg') or 1.0
    try:
        pf = float(pf)
        if pf <= 0.5 or pf >= 2.0:
            pf = 1.0  # implausible park factor → treat as neutral
    except (ValueError, TypeError):
        pf = 1.0
    woba_neutral = woba / pf

    league_avg = float(row.get('_league_avg_woba') or 0.320)
    diff = woba_neutral - league_avg

    # Confidence increases with sample size, saturates at ~500 PA
    confidence = min(math.sqrt(pa / 500.0), 1.0)

    # Scale: .050 wOBA diff × 600 × 1.0 confidence = 30 meta points
    adjustment = diff * 600 * confidence
    return max(-40.0, min(40.0, adjustment))


def _calc_performance_adjustment_pitching(row):
    """Adjust meta based on actual SIERA performance.

    SIERA (Skill-Interactive ERA) captures pitching effectiveness beyond
    what ratings predict. A low SIERA relative to league average means
    the pitcher is generating outs more efficiently than ratings suggest.

    Capped at ±40. SIERA has r=-0.654 with WAR/200 in lb124 — the
    strongest pitching signal by a wide margin.
    """
    siera = row.get('adv_siera')
    ip = row.get('adv_ip')
    if not siera or not ip:
        return 0.0
    try:
        siera = float(siera)
        ip = float(ip)
    except (ValueError, TypeError):
        return 0.0
    if siera <= 0 or ip < _PERF_MIN_IP:
        return 0.0

    # Park adjustment — multiply SIERA by pitcher-favoring park factor
    # (pf < 1.0 means hitter's park, so pitcher's raw SIERA understates
    # their true skill and we scale it DOWN to normalize). Pitching uses
    # the reciprocal orientation of batting: a hitter's park hurts pitchers.
    pf = row.get('_park_factor_pit') or row.get('pf_overall') or 1.0
    try:
        pf = float(pf)
        if pf <= 0.5 or pf >= 2.0:
            pf = 1.0
    except (ValueError, TypeError):
        pf = 1.0
    # SIERA scales with runs → divide by pf so that a pitcher in a hitter's
    # park (pf>1) gets SIERA nudged down (more favorable), matching what a
    # neutral-park version would look like.
    siera_neutral = siera / pf

    league_avg = float(row.get('_league_avg_siera') or 4.00)
    diff = league_avg - siera_neutral  # lower SIERA is better → positive diff = good

    confidence = min(math.sqrt(ip / 150.0), 1.0)

    # Scale: 1.00 SIERA diff × 30 × 1.0 confidence = 30 meta points
    adjustment = diff * 30 * confidence
    return max(-40.0, min(40.0, adjustment))


def _r2_gate_for(cal_type: str) -> float:
    """Return the minimum R² threshold for a given calibration_type."""
    if cal_type == 'batting':
        return MIN_CALIBRATION_R2
    # pitching / pitching_sp / pitching_rp all use the relaxed pitching gate
    return MIN_PITCHING_R2


def _active_league_id() -> str | None:
    """Return the active league from config, or None if unreadable."""
    try:
        from app.core.database import load_config
        return load_config().get('active_league')
    except Exception:
        return None


def _load_calibration_row(cal_type: str):
    """Read the latest row for a calibration_type and gate it.

    Returns the parsed weights dict on success, or None on any failure
    (missing row, below threshold, unparseable JSON). Populates
    ``LAST_WEIGHT_LOAD_DIAGNOSTICS[cal_type]`` with the reason so the UI
    can surface it.

    **Cross-league lookup (Epic G, 2026-04-24):** if an active league is
    configured, this tries ``{cal_type}:{league}`` first (league-specific
    fit) and falls back to ``{cal_type}`` (pooled fit) when the
    league-specific row is missing or fails the gate. This resolves the
    opposite-sign residuals between lb124 and i76 by letting each league
    carry its own weight profile when sample size supports it.

    Gate logic (either suffices):
        CV R² ≥ _r2_gate_for(cal_type), OR
        Pearson correlation ≥ MIN_CALIBRATION_CORR
    """
    import sqlite3 as _sqlite3
    # Build the lookup order: league-specific first if active league
    # exists, then the pooled fallback. The diagnostics dict is keyed on
    # `cal_type` (the caller's requested key) so the UI still shows the
    # right chip even after a league fallback.
    active = _active_league_id()
    lookup_types = []
    if active:
        lookup_types.append(f"{cal_type}:{active}")
    lookup_types.append(cal_type)

    try:
        db_path = get_db_path()
        conn = _sqlite3.connect(db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=10000")
        except Exception:
            pass
        cursor = conn.cursor()
        row = None
        source_type = None
        for lt in lookup_types:
            cursor.execute(
                "SELECT weights_json, r_squared, correlation, sample_size "
                "FROM meta_calibration WHERE calibration_type = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (lt,),
            )
            r = cursor.fetchone()
            if r and r[0]:
                # Gate-check each candidate; if league-specific fails the gate
                # we fall through to pooled.
                threshold = _r2_gate_for(cal_type)
                r2_ok = r[1] is not None and r[1] >= threshold
                corr_ok = r[2] is not None and r[2] >= MIN_CALIBRATION_CORR
                if r2_ok or corr_ok:
                    row = r
                    source_type = lt
                    break
        conn.close()
    except Exception as e:
        LAST_WEIGHT_LOAD_DIAGNOSTICS[cal_type] = {
            "status": "error", "r_squared": None, "correlation": None,
            "sample_size": None,
            "reason": f"exception reading meta_calibration: {e}",
        }
        return None

    if not row or not row[0]:
        LAST_WEIGHT_LOAD_DIAGNOSTICS[cal_type] = {
            "status": "missing", "r_squared": None, "correlation": None,
            "sample_size": None,
            "reason": "no calibration row found",
        }
        return None

    weights_json, r_squared, correlation, sample_size = row
    threshold = _r2_gate_for(cal_type)
    # Reject rows with NULL / negative R² (the old broken OLS-with-OVR
    # runs wrote rows with r_squared ≈ -105000) unless Pearson r clears
    # the secondary gate.
    r2_ok = r_squared is not None and r_squared >= threshold
    corr_ok = correlation is not None and correlation >= MIN_CALIBRATION_CORR
    if not (r2_ok or corr_ok):
        LAST_WEIGHT_LOAD_DIAGNOSTICS[cal_type] = {
            "status": "rejected_low_r2",
            "r_squared": r_squared,
            "correlation": correlation,
            "sample_size": sample_size,
            "reason": (
                f"calibration failed both gates: R²={r_squared} < {threshold} "
                f"AND Pearson r={correlation} < {MIN_CALIBRATION_CORR}; "
                f"falling back to config/defaults"
            ),
        }
        return None

    try:
        parsed = json.loads(weights_json)
    except (json.JSONDecodeError, TypeError):
        LAST_WEIGHT_LOAD_DIAGNOSTICS[cal_type] = {
            "status": "error",
            "r_squared": r_squared,
            "correlation": correlation,
            "sample_size": sample_size,
            "reason": "weights_json could not be parsed",
        }
        return None

    reason_parts = []
    if r2_ok:
        reason_parts.append(f"CV R²={r_squared} ≥ {threshold}")
    if corr_ok:
        reason_parts.append(f"Pearson r={correlation} ≥ {MIN_CALIBRATION_CORR}")
    # Note which calibration row actually got picked — league-specific or
    # pooled — so the UI can show "used lb124 weights" vs "used pooled weights".
    scope_note = f" [scope={source_type}]" if source_type and source_type != cal_type else ""
    LAST_WEIGHT_LOAD_DIAGNOSTICS[cal_type] = {
        "status": "used",
        "r_squared": r_squared,
        "correlation": correlation,
        "sample_size": sample_size,
        "source_type": source_type,
        "reason": "calibration passed: " + " and ".join(reason_parts) + scope_note,
    }
    return parsed


def _load_calibrated_weights() -> tuple:
    """Load (batting, pitching_combined) calibrated weights from the DB.

    Each type is gated independently on its own R² threshold. Returns
    ``(bw, pw)`` where either can be None if that type's calibration was
    missing, errored, or below threshold. The pitching slot is the
    SP+RP *combined* row — callers that need role-aware dispatch should
    use ``get_pitching_weights_for(row)`` instead.
    """
    LAST_WEIGHT_LOAD_DIAGNOSTICS.clear()
    bw = _load_calibration_row('batting')
    pw = _load_calibration_row('pitching')
    return bw, pw


def _load_pitching_weights_for_role(role: str) -> dict:
    """Load role-specific pitching weights (``'SP'`` or ``'RP'``).

    Reads ``pitching_sp`` or ``pitching_rp`` from meta_calibration. Returns
    None if that row is missing or below threshold, so the caller can
    fall back to the combined pitching weights or config.
    """
    cal_type = 'pitching_sp' if role == 'SP' else 'pitching_rp'
    return _load_calibration_row(cal_type)


def get_weights():
    """Load weights: calibrated DB weights > config.yaml > defaults."""
    # 1. Try calibrated weights from DB
    cal_bw, cal_pw = _load_calibrated_weights()
    if cal_bw is not None and cal_pw is not None:
        return cal_bw, cal_pw

    # 2. Fall back to config.yaml
    try:
        config = load_config()
        bw = cal_bw or config.get('batting_weights', DEFAULT_BATTING_WEIGHTS)
        pw = cal_pw or config.get('pitching_weights', DEFAULT_PITCHING_WEIGHTS)
    except Exception:
        bw = cal_bw or DEFAULT_BATTING_WEIGHTS
        pw = cal_pw or DEFAULT_PITCHING_WEIGHTS
    return bw, pw


def get_weights_with_source() -> tuple:
    """Load weights and return (batting_weights, pitching_weights, source_label).

    source_label summarizes the overall load outcome:
      'calibrated'        — both batting AND pitching loaded from DB calibration
      'calibrated_mixed'  — one of the two is calibrated, the other is
                            config/default (e.g. pitching calibration R² too
                            low while batting passed)
      'config'            — neither calibrated, both from config.yaml
      'default'           — neither calibrated and no config; using hardcoded
                            DEFAULT_BATTING_WEIGHTS / DEFAULT_PITCHING_WEIGHTS

    For a finer-grained view (per-type status, R², rejection reason) call
    ``get_weight_diagnostics()``.
    """
    # 1. Try calibrated weights from DB (each type gated independently by R²)
    cal_bw, cal_pw = _load_calibrated_weights()
    if cal_bw is not None and cal_pw is not None:
        return cal_bw, cal_pw, 'calibrated'

    # 2. Try config.yaml for whichever type wasn't calibrated
    cfg_bw, cfg_pw = None, None
    try:
        config = load_config()
        cfg_bw = config.get('batting_weights')
        cfg_pw = config.get('pitching_weights')
    except Exception:
        pass

    bw = cal_bw or cfg_bw or DEFAULT_BATTING_WEIGHTS
    pw = cal_pw or cfg_pw or DEFAULT_PITCHING_WEIGHTS

    # 3. Decide the overall source label
    if cal_bw is not None or cal_pw is not None:
        # Exactly one type loaded from calibration
        return bw, pw, 'calibrated_mixed'
    if cfg_bw is not None or cfg_pw is not None:
        return bw, pw, 'config'
    return bw, pw, 'default'


def _is_starting_pitcher(row: dict) -> bool:
    """Classify a card row as SP vs RP for role-aware weight dispatch.

    OOTP's ``pitcher_role_name`` column in this DB stores the short form:
    ``'SP'``, ``'RP'``, ``'CL'``. Numeric ``pitcher_role`` is 11 (SP), 12
    (RP), or 13 (CL) — these are OOTP's internal position codes, not 1/2/3.
    Only 'SP' is treated as a starter; relievers and closers both route to
    the RP weight set, which matches how PT games actually usage them.
    """
    role_name = (row.get('pitcher_role_name')
                 or row.get('Pitcher Role')
                 or row.get('PitcherRole')
                 or '')
    if isinstance(role_name, str):
        stripped = role_name.strip().upper()
        if stripped == 'SP' or 'STARTING' in stripped or stripped == 'STARTER':
            return True
        if stripped in ('RP', 'CL', 'MR', 'SU', 'LR', 'CP', 'SW'):
            return False
    # Numeric role: 11 = SP in this DB
    role_num = row.get('pitcher_role')
    try:
        return int(role_num) == 11
    except (TypeError, ValueError):
        return False


def get_pitching_weights_for(row: dict) -> dict:
    """Return the pitching weight dict appropriate for this card's role.

    Dispatches to SP or RP calibration based on ``pitcher_role_name``.
    Falls back to the combined ``pitching`` calibration, then to
    config.yaml, then to ``DEFAULT_PITCHING_WEIGHTS``.

    This is how we thread SP vs RP role awareness through the scorer
    instead of blending the two into a single set of weights at load
    time. A starting pitcher and a closer have different optimal weight
    profiles (starters care more about stamina, relievers care more about
    peak stuff) and averaging them degrades both.
    """
    is_sp = _is_starting_pitcher(row)
    role_key = 'SP' if is_sp else 'RP'
    role_weights = _load_pitching_weights_for_role(role_key)
    if role_weights is not None:
        return role_weights

    # 2026-04-20: When per-role calibration is missing, fall back to role-
    # specific DEFAULTS rather than the combined pitching weights. The
    # pooled-pitching residuals hid a sharp RP-specific pattern (stuff
    # over-weighted, control under-weighted) that DEFAULT_PITCHING_WEIGHTS_RP
    # corrects. See constants.py docstring for the residual numbers.
    if is_sp:
        return dict(DEFAULT_PITCHING_WEIGHTS_SP)
    return dict(DEFAULT_PITCHING_WEIGHTS_RP)


def get_weight_diagnostics() -> dict:
    """Return per-type diagnostics from the last weight-load attempt.

    Structure::

        {
          'batting':  {'status': 'used'|'rejected_low_r2'|'missing'|'error',
                       'r_squared': float|None,
                       'correlation': float|None,
                       'sample_size': int|None,
                       'reason': str},
          'pitching': {...},
          'pitching_sp': {...},
          'pitching_rp': {...},
          'min_calibration_r2': 0.10,
          'min_pitching_r2': 0.03,
        }

    Call ``get_weights()`` or ``get_weights_with_source()`` first — those
    populate the underlying diagnostics dict as a side effect. If called
    before any weight load has been attempted this session, the values
    will be empty.

    Used by the Settings UI and by the top-of-page banner on the Roster
    Optimizer to surface "calibration rejected because R² too low".
    """
    # Trigger a fresh load so diagnostics reflect the current DB state.
    _load_calibrated_weights()
    # Also probe the SP/RP rows so the UI can show per-role gate outcomes.
    _load_pitching_weights_for_role('SP')
    _load_pitching_weights_for_role('RP')
    return {
        "batting": dict(LAST_WEIGHT_LOAD_DIAGNOSTICS.get("batting", {"status": "missing", "reason": "not loaded"})),
        "pitching": dict(LAST_WEIGHT_LOAD_DIAGNOSTICS.get("pitching", {"status": "missing", "reason": "not loaded"})),
        "pitching_sp": dict(LAST_WEIGHT_LOAD_DIAGNOSTICS.get("pitching_sp", {"status": "missing", "reason": "not loaded"})),
        "pitching_rp": dict(LAST_WEIGHT_LOAD_DIAGNOSTICS.get("pitching_rp", {"status": "missing", "reason": "not loaded"})),
        "min_calibration_r2": MIN_CALIBRATION_R2,
        "min_pitching_r2": MIN_PITCHING_R2,
    }


def calc_defense_score(row: dict, apply_position_multiplier: bool = True) -> float:
    """Calculate defense component from card data.

    When apply_position_multiplier is True (default), the raw defensive average
    is scaled by POSITION_DEFENSE_MULTIPLIERS so that SS/CF/C defense counts
    much more than 1B/DH defense.  Pass False to get the raw average.
    """
    # Determine position type and calc appropriate defense average
    pos = row.get('position') or row.get('Position') or 0
    if isinstance(pos, str):
        # Map string positions
        pos_map = {"C": 2, "1B": 3, "2B": 4, "3B": 5, "SS": 6, "LF": 7, "CF": 8, "RF": 9, "DH": 10, "P": 1}
        pos = pos_map.get(pos, 0)

    try:
        pos = int(pos)
    except (ValueError, TypeError):
        pos = 0

    raw = 0.0
    if pos in (7, 8, 9):  # OF
        vals = [row.get('of_range', 0) or 0, row.get('of_error', 0) or 0, row.get('of_arm', 0) or 0]
        # Also check alternate column names
        if not any(vals):
            vals = [row.get('OF Range', 0) or 0, row.get('OF Error', 0) or 0, row.get('OF Arm', 0) or 0]
        raw = sum(vals) / max(len([v for v in vals if v]), 1)
    elif pos == 2:  # C
        vals = [row.get('catcher_ability', 0) or 0, row.get('catcher_frame', 0) or 0, row.get('catcher_arm', 0) or 0]
        if not any(vals):
            vals = [row.get('CatcherAbil', 0) or 0, row.get('CatcherFrame', 0) or 0, row.get('Catcher Arm', 0) or 0]
        raw = sum(vals) / max(len([v for v in vals if v]), 1)
    elif pos in (3, 4, 5, 6):  # IF
        vals = [row.get('infield_range', 0) or 0, row.get('infield_error', 0) or 0, row.get('infield_arm', 0) or 0]
        if not any(vals):
            vals = [row.get('Infield Range', 0) or 0, row.get('Infield Error', 0) or 0, row.get('Infield Arm', 0) or 0]
        raw = sum(vals) / max(len([v for v in vals if v]), 1)

    if not apply_position_multiplier:
        return raw

    # Scale by position importance: SS defense is worth 1.4x, 1B is 0.5x, etc.
    multiplier = POSITION_DEFENSE_MULTIPLIERS.get(pos, 1.0)
    return raw * multiplier


def calc_speed_score(row: dict) -> float:
    """Calculate speed/baserunning component from card data.

    Combines Speed, Stealing, and Baserunning ratings.
    Speed alone is conditionally valuable — it only matters if you get on base.
    We weight Stealing highest (directly produces SB/runs), then Baserunning
    (extra bases taken), then Speed (raw tool).
    """
    speed = float(row.get('speed') or row.get('Speed') or row.get('SPD') or 0)
    stealing = float(row.get('stealing') or row.get('Stealing') or row.get('STL') or 0)
    baserunning = float(row.get('baserunning') or row.get('Baserunning') or row.get('BR') or 0)

    if speed <= 0 and stealing <= 0 and baserunning <= 0:
        return 0.0

    # Weighted composite: Stealing 40%, Baserunning 35%, Speed 25%
    # Stealing is most directly predictive of SB (r=+0.344)
    composite = (stealing * 0.40 + baserunning * 0.35 + speed * 0.25)

    # Only count speed bonus above average (70) — slow players shouldn't be
    # penalized since the batting stats already capture their output
    if composite < 70:
        return 0.0

    return composite - 70  # Excess above average baseline


def calc_batting_meta(row: dict, weights: dict = None) -> float:
    """Calculate batter meta score.

    v6 (2026-04-17): **OVR demoted to diagnostic.** The meta formula no longer
    includes OVR as a term — calibration explicitly drops it for VIF>10, so
    hardcoding it back at weight 2.00 was circular (meta ≈ OVR + noise).
    Use ``card['ovr']`` as a separate sanity-check column, not a formula input.

    Layers:
      1. **Platoon-adjusted base** — handedness-weighted vs-R/vs-L splits.
      2. **Position flexibility** — small theoretical roster bonus (not WAR-correlated).
      3. **Performance overlay** — actual wOBA vs league (when sample is large enough).
      4. **Positional value bonus** — fWAR ladder (SS/C premium, 1B/DH discount).
      5. **Speed/stealing** — conditional bonus above 70 composite.
      6. **Defense** — position-scaled by fWAR positional adjustment.

    All fallback weights inside ``weights.get()`` now resolve to
    ``DEFAULT_BATTING_WEIGHTS[key]`` so an incomplete calibration dict no longer
    silently reverts to the pre-calibration v4 values.
    """
    if weights is None:
        weights, _ = get_weights()

    # ── Platoon-adjusted base stats ──
    # Use handedness-weighted effective values (60% vR + 40% vL) when splits
    # are available; falls back to overall ratings when split data is absent.
    gap_ovr = float(row.get('gap_power') or row.get('Gap') or row.get('GAP') or 0)
    con_ovr = float(row.get('contact') or row.get('Contact') or row.get('CON') or 0)
    eye_ovr = float(row.get('eye') or row.get('Eye') or row.get('EYE') or 0)
    pwr_ovr = float(row.get('power') or row.get('Power') or row.get('POW') or 0)
    gap = _apply_splits(row, gap_ovr, 'gap_vl', 'gap_vr', _BATTER_VS_RHP_WEIGHT)
    con = _apply_splits(row, con_ovr, 'contact_vl', 'contact_vr', _BATTER_VS_RHP_WEIGHT)
    eye = _apply_splits(row, eye_ovr, 'eye_vl', 'eye_vr', _BATTER_VS_RHP_WEIGHT)
    pwr = _apply_splits(row, pwr_ovr, 'power_vl', 'power_vr', _BATTER_VS_RHP_WEIGHT)
    avk = float(row.get('avoid_ks') or row.get('Avoid Ks') or row.get("K's") or 0)
    bab = float(row.get('babip') or row.get('BABIP') or 0)
    defense = float(row.get('defense_score') or calc_defense_score(row))
    speed_score = float(row.get('speed_score') or calc_speed_score(row))

    try:
        # Core weighted sum — OVR is intentionally absent (see docstring).
        # Fallbacks now resolve to the authoritative DEFAULT_BATTING_WEIGHTS
        # instead of stale pre-calibration v4 numbers.
        meta = (_diminished(gap) * weights.get('gap_power', DEFAULT_BATTING_WEIGHTS['gap_power']) +
                _diminished(con) * weights.get('contact', DEFAULT_BATTING_WEIGHTS['contact']) +
                _diminished(avk) * weights.get('avoid_ks', DEFAULT_BATTING_WEIGHTS['avoid_ks']) +
                _diminished(eye) * weights.get('eye', DEFAULT_BATTING_WEIGHTS['eye']) +
                _diminished(pwr) * weights.get('power', DEFAULT_BATTING_WEIGHTS['power']) +
                _diminished(bab) * weights.get('babip', DEFAULT_BATTING_WEIGHTS['babip']) +
                defense * weights.get('defense', DEFAULT_BATTING_WEIGHTS['defense']))

        # Speed/Stealing component — conditionally valuable (amplifies OBP)
        spd_weight = weights.get('speed_stealing', DEFAULT_BATTING_WEIGHTS['speed_stealing'])
        if speed_score > 0 and spd_weight > 0:
            meta += _diminished(speed_score + 70) * spd_weight

        # ── Platoon penalty: weak-side exposure ──
        meta += _calc_platoon_penalty_batting(row)

        # Balance penalty — key stats below floor get penalized.
        # Uses effective (platoon-weighted) stats so weak-side holes are caught.
        floor = BATTING_STAT_FLOOR
        for stat, coeff in [(con, 0.4), (gap, 0.4), (pwr, 1.0)]:
            if 0 < stat < floor:
                penalty = (floor - stat) * coeff
                meta -= penalty

        # Position flexibility REMOVED from WAR-prediction meta (v6).
        # lb124 data: playable_positions count r=-0.023 with WAR (p=0.74).
        # The "roster utility" value is real for lineup construction but
        # provably orthogonal to a card's WAR output — keeping it in the
        # meta formula added variance without signal. Multi-position
        # recommendations are still surfaced separately in the Buy Recs
        # page via the multi-position eligibility scan.

        # ── Performance overlay (wOBA-based) ──
        meta += _calc_performance_adjustment_batting(row)

        # ── Superstat overlay (game-log EV + LD%) ──
        # Only fires when the recalc path has injected game-log aggregates
        # onto the row (see recalculate_all_meta_scores). Cards without
        # HTML game logs get no adjustment.
        meta += _calc_superstat_overlay_batting(row)

        # ── ISO overlay (observed isolated power) ──
        # Comprehensive correlation scan (2026-04-17) shows obs ISO has
        # r=+0.580 with WAR/600 pooled across lb124+i76, stronger than
        # any rating. Tier-relative baseline used so low-tier cards
        # aren't unfairly discounted.
        meta += _calc_iso_overlay_batting(row)

        # ── OPS+ overlay (park/league-adjusted observed offense) ──
        # Residual analysis shows OPS+ has r=+0.325 with meta residuals
        # — strong additive signal beyond wOBA. OPS+ is already league-
        # normalized (100 = avg), so it's naturally cross-league stable.
        meta += _calc_opsplus_overlay_batting(row)

        # ── OBP overlay (on-base% delta vs league) ──
        # Post-OPS+ residual shows OBP still r=+0.337 with residuals.
        # OPS+ absorbs some OBP but standalone OBP carries distinct
        # walk/HBP signal beyond slugging. Magnitude conservative to
        # avoid double-counting OPS+.
        meta += _calc_obp_overlay_batting(row)

        # ── BABIP overlay (obs BABIP delta vs league) ──
        # Residual r=+0.26 after other obs overlays — captures ball-in-play
        # quality distinct from slugging/on-base. Tight cap to avoid noise.
        meta += _calc_babip_overlay_batting(row)

        # ── Over-performance overlay (obs OPS+ beyond rating prediction) ──
        # Meta-correlation: overperf vs residual has r=+0.551 beyond all
        # other overlays. Captures cards that "play up" their ratings.
        meta += _calc_overperformance_overlay_batting(row)

        # ── Rating diminishing-returns penalty ──
        # contact × power has r=-0.44 with residuals — dual-elite cards
        # are over-predicted by additive meta. Penalize mild interaction.
        meta += _calc_rating_diminishing_penalty_batting(row)

        # ── ISO-gap overlay (obs ISO vs power-rating prediction) ──
        # 2026-04-20 residual analysis: iso_gap has r=+0.466 with WAR
        # residuals AFTER the current meta (which already has iso tier-
        # delta, OPS+, OBP, BABIP, and overperformance overlays). This is
        # the single biggest untapped signal — observed ISO sometimes
        # dramatically under/over-performs the power rating, and that
        # gap predicts WAR very strongly. Unlike _calc_iso_overlay_batting
        # (tier-peer delta), this uses the power rating itself as baseline.
        meta += _calc_iso_gap_overlay_batting(row)

        # ── Clutch overlay (box-score 2-out RBI vs LOB-RISP-2-out) ──
        # Captures high-leverage production that rate stats don't reflect.
        meta += _calc_clutch_overlay_batting(row)

        # ── Defensive error-rate overlay ──
        # Observed errors per game vs position baseline. Zero for players
        # without enough defensive game sample (≥20 games).
        meta += _calc_error_overlay_batting(row)

        # ── Observed defense (Zone Rating) overlay — Epic B ──
        # Games-weighted mean zr from fielding_stats. The rating-based
        # defense term (calc_defense_score above) has r≈0 with WAR; this
        # observed term adds a small ±10 adjustment based on actual range
        # demonstrated in-game. Zero when the card has <20 defensive games.
        meta += _calc_observed_defense_overlay_batting(row)

        # ── Recent-form (last 30 days) overlay — 2026-04-24 ──
        # All other observed overlays freeze at last ingest, blending the
        # full season into one number. A cold high-meta starter still ranks
        # "Optimal" because the static meta hides the streak. This nudges
        # meta toward the last 30 days of OBP — capped ±15 so it can shift
        # borderline upgrade decisions without overruling the foundation.
        meta += _calc_recent_form_overlay_batting(row)

        # Positional value bonus — fWAR ladder (SS > C > CF > ... > 1B > DH)
        pos = row.get('position') or row.get('Position') or 0
        if isinstance(pos, str):
            pos_map = {"C": 2, "1B": 3, "2B": 4, "3B": 5, "SS": 6,
                       "LF": 7, "CF": 8, "RF": 9, "DH": 10}
            pos = pos_map.get(pos, 0)
        try:
            pos = int(pos)
        except (ValueError, TypeError):
            pos = 0
        meta += POSITIONAL_VALUE_BONUS.get(pos, 0)

        # ── Card-type systematic bias (2026-04-17 residual analysis) ──
        # See BATTING_CARD_TYPE_OFFSET in constants.py. Captures meta error
        # that's systematic per card category (e.g. Live underperforms by
        # ~0.2 WAR/600, Snapshots beat by ~0.35). Zero for types with no
        # significant bias, so behavior is unchanged for the majority.
        ct = row.get('card_type')
        if ct is not None:
            try:
                meta += BATTING_CARD_TYPE_OFFSET.get(int(ct), 0)
            except (ValueError, TypeError):
                pass

    except (ValueError, TypeError):
        meta = 0.0

    return round(meta, 2)


def calc_pitching_meta(row: dict, weights: dict = None) -> float:
    """Calculate pitcher meta score.

    v6 (2026-04-17): **OVR demoted to diagnostic.** Same rationale as batting —
    OVR is dropped from calibration for multicollinearity, so including it as
    a hardcoded term made the pitching meta also mostly-OVR + noise.

    Layers:
      1. **Platoon-adjusted base** — handedness-weighted stuff/movement/control.
      2. **Arsenal diversity** — theoretical repertoire bonus (small weight).
      3. **Performance overlay** — SIERA vs league (when IP is large enough).
      4. **SIERA-style interactions** — stuff×movement, stuff×control synergies.
      5. **Stamina/Hold** — small bonus for usage flexibility.

    All fallbacks now resolve to ``DEFAULT_PITCHING_WEIGHTS[key]``.
    """
    if weights is None:
        weights = get_pitching_weights_for(row)

    is_sp = _is_starting_pitcher(row)

    # ── Platoon-adjusted base stats ──
    mov_ovr = float(row.get('movement') or row.get('Movement') or row.get('MOV') or 0)
    stu_ovr = float(row.get('stuff') or row.get('Stuff') or row.get('STU') or 0)
    ctrl_ovr = float(row.get('control') or row.get('Control') or row.get('CON') or 0)
    mov = _apply_splits(row, mov_ovr, 'movement_vl', 'movement_vr', _PITCHER_VS_RHB_WEIGHT)
    stu = _apply_splits(row, stu_ovr, 'stuff_vl', 'stuff_vr', _PITCHER_VS_RHB_WEIGHT)
    ctrl = _apply_splits(row, ctrl_ovr, 'control_vl', 'control_vr', _PITCHER_VS_RHB_WEIGHT)
    phr = float(row.get('p_hr') or row.get('pHR') or row.get('HRA') or 0)
    stamina = float(row.get('stamina') or row.get('Stamina') or row.get('STA') or 0)
    hold = float(row.get('hold') or row.get('Hold') or 0)

    try:
        # Core ratings with diminishing returns on extreme values.
        # OVR is intentionally NOT a formula term (see docstring) — use the
        # separate ``ovr`` column for sanity-check comparisons.
        meta = (_diminished(mov) * weights.get('movement', DEFAULT_PITCHING_WEIGHTS['movement']) +
                _diminished(stu) * weights.get('stuff', DEFAULT_PITCHING_WEIGHTS['stuff']) +
                _diminished(ctrl) * weights.get('control', DEFAULT_PITCHING_WEIGHTS['control']) +
                _diminished(phr) * weights.get('p_hr', DEFAULT_PITCHING_WEIGHTS['p_hr']))

        # SIERA-inspired interaction terms — pitching is non-additive.
        # Ks + weak contact together are more valuable than the sum of parts.
        # Use raw (not diminished) values for interactions to avoid double-sqrt.
        sxm_w = weights.get('stuff_x_movement', DEFAULT_PITCHING_WEIGHTS['stuff_x_movement'])
        sxc_w = weights.get('stuff_x_control', DEFAULT_PITCHING_WEIGHTS['stuff_x_control'])
        mxc_w = weights.get('movement_x_control', DEFAULT_PITCHING_WEIGHTS['movement_x_control'])
        if sxm_w > 0:
            meta += stu * mov * sxm_w
        if sxc_w > 0:
            meta += stu * ctrl * sxc_w
        if mxc_w > 0:
            meta += mov * ctrl * mxc_w

        # Stamina/Hold component (matters for relievers and starters alike)
        sh_weight = weights.get('stamina_hold', DEFAULT_PITCHING_WEIGHTS['stamina_hold'])
        if sh_weight > 0:
            sh_avg = 0
            sh_count = 0
            if stamina > 0:
                sh_avg += stamina
                sh_count += 1
            if hold > 0:
                sh_avg += hold
                sh_count += 1
            if sh_count > 0:
                meta += (sh_avg / sh_count) * sh_weight

        # ── Platoon penalty ──
        meta += _calc_platoon_penalty_pitching(row)

        # Balance penalty — key ratings below floor get penalized.
        floor = PITCHING_STAT_FLOOR
        for stat in [stu, mov, ctrl]:
            if 0 < stat < floor:
                shortfall = floor - stat
                penalty = shortfall * 1.0
                meta -= penalty

        # Arsenal diversity REMOVED from WAR-prediction meta (v6).
        # lb124 data: pitch_count r=+0.024 with WAR (p=0.73). Like position
        # flexibility for batters, this layer had theoretical value that the
        # data doesn't support. The scaffolding (_calc_arsenal_factor) stays
        # around in case a future analysis re-justifies it.

        # ── Performance overlay (SIERA-based) ──
        meta += _calc_performance_adjustment_pitching(row)

        # ── Superstat overlay (game-log EV-allowed + LD%-allowed + K%/BB%) ──
        # Only fires when the recalc path injected game-log aggregates.
        meta += _calc_superstat_overlay_pitching(row)

        # ── Reliever discipline overlay (inherited-runners-scored rate) ──
        # Rewards pitchers who strand inherited runners; penalizes leakers.
        # ERA doesn't charge for inherited runs, so this is a hidden signal.
        meta += _calc_reliever_discipline_overlay_pitching(row)

        # ── FIP overlay (fielding-independent pitching) ──
        # Residual analysis shows FIP has r=-0.359 with meta residuals;
        # captures HR-per-9 signal that SIERA undershoots (HR r=-0.414
        # with residuals is the biggest single pitching prediction gap).
        meta += _calc_fip_overlay_pitching(row)

        # ── SP HR/9 overlay (2026-04-24) ──
        # SP residual analysis (n=1834) found HR/9 still has r=-0.614 with
        # WAR/200 residuals AFTER FIP. FIP dilutes HR with K/BB; OOTP PT
        # third-time-through penalties make standalone HR rate matter more
        # for SPs specifically. RPs no-op.
        meta += _calc_hr9_overlay_pitching(row)

        # ── BB/9 overlay (observed walk rate) ──
        # Post-FIP residual r=-0.34 for bb_per_9 — walks still undercharged.
        meta += _calc_bb9_overlay_pitching(row)

        # ── Over-performance overlay (obs ERA+ beyond rating prediction) ──
        # Mirrors batting overperformance. Captures "plays up" pitchers.
        meta += _calc_overperformance_overlay_pitching(row)

        # ── Stuff-diminishing penalty ──
        # Residual shows stuff r=-0.519 — dual-elite pitchers over-predicted.
        meta += _calc_stuff_diminishing_pitching(row)

        # ── Stamina bonus for SPs ──
        # Calibrated stamina_hold=0.10 is severely under-weighted (residual
        # r=+0.377). Linear correction for SPs only.
        meta += _calc_stamina_bonus_pitching(row)

        # ── Stuff over-credit linear damping ──
        # Even after dual-elite penalty, stuff residual r=-0.52. Gentle
        # linear negative correction across the distribution.
        meta += _calc_stuff_overcredit_damping_pitching(row)

        # ── RP pair-interaction overlay ──
        # 2026-04-20 residual analysis (RP n=1142) found:
        #   (control-70)*(p_hr-70)  resid r=+0.180  "command fireman"
        #   (movement-70)*(control-70) resid r=+0.177 "junk-ball precision"
        # SP residuals for the same pairs are ~0, so the overlay is
        # role-gated inside the function. No-op for SPs.
        meta += _calc_pair_interaction_overlay_pitching(row)

        # ── Recent-form (last 30 days) overlay — 2026-04-24 ──
        # Mirrors the batting overlay: nudge meta toward the last 30 days of
        # ERA vs season ERA. Captures cold starters (Guzman 7+ ERA recent /
        # acceptable full-season) the static overlays miss, and rewards
        # heaters. ±15 cap keeps it advisory, not dominant.
        meta += _calc_recent_form_overlay_pitching(row)

        # ── Card-type systematic bias (2026-04-17 residual analysis) ──
        # See PITCHING_CARD_TYPE_OFFSET. Pitching has stronger card_type
        # effects than batting — Legend +65, Hardware Heroes -76. Opposite
        # sign to batting in some cases (All-Time Legend +44 bat / -30 pit).
        ct = row.get('card_type')
        if ct is not None:
            try:
                meta += PITCHING_CARD_TYPE_OFFSET.get(int(ct), 0)
            except (ValueError, TypeError):
                pass

    except (ValueError, TypeError):
        meta = 0.0

    return round(meta, 2)


def calc_batting_meta_vs_rhp(row: dict, weights: dict = None) -> float:
    """Calculate batter meta score vs right-handed pitching.

    Uses the batter's 'vR' split ratings where available, falling back to overall.
    When facing RHP, OOTP uses the batter's CON vR, POW vR, EYE vR ratings.
    """
    if weights is None:
        weights, _ = get_weights()

    con = float(row.get('con_vr') or row.get('CON vR') or row.get('contact_vr') or
                row.get('contact') or row.get('CON') or 0)
    pwr = float(row.get('pow_vr') or row.get('POW vR') or row.get('power_vr') or
                row.get('power') or row.get('POW') or 0)
    eye = float(row.get('eye_vr') or row.get('EYE vR') or row.get('eye_vr') or
                row.get('eye') or row.get('EYE') or 0)
    gap = float(row.get('gap_power') or row.get('Gap') or row.get('GAP') or 0)
    avk = float(row.get('avoid_ks') or row.get('Avoid Ks') or row.get("K's") or 0)
    bab = float(row.get('babip') or row.get('BABIP') or 0)
    defense = float(row.get('defense_score') or calc_defense_score(row))
    speed_score = float(row.get('speed_score') or calc_speed_score(row))

    try:
        meta = (_diminished(gap) * weights.get('gap_power', DEFAULT_BATTING_WEIGHTS['gap_power']) +
                _diminished(con) * weights.get('contact', DEFAULT_BATTING_WEIGHTS['contact']) +
                _diminished(avk) * weights.get('avoid_ks', DEFAULT_BATTING_WEIGHTS['avoid_ks']) +
                _diminished(eye) * weights.get('eye', DEFAULT_BATTING_WEIGHTS['eye']) +
                _diminished(pwr) * weights.get('power', DEFAULT_BATTING_WEIGHTS['power']) +
                _diminished(bab) * weights.get('babip', DEFAULT_BATTING_WEIGHTS['babip']) +
                defense * weights.get('defense', DEFAULT_BATTING_WEIGHTS['defense']))

        spd_weight = weights.get('speed_stealing', DEFAULT_BATTING_WEIGHTS['speed_stealing'])
        if speed_score > 0 and spd_weight > 0:
            meta += _diminished(speed_score + 70) * spd_weight

        floor = BATTING_STAT_FLOOR
        key_stats = [con, gap]
        for stat in key_stats:
            if 0 < stat < floor:
                meta -= (floor - stat) * 0.4
    except (ValueError, TypeError):
        meta = 0.0

    return round(meta, 2)


def calc_batting_meta_vs_lhp(row: dict, weights: dict = None) -> float:
    """Calculate batter meta score vs left-handed pitching.

    Uses the batter's 'vL' split ratings where available, falling back to overall.
    """
    if weights is None:
        weights, _ = get_weights()

    con = float(row.get('con_vl') or row.get('CON vL') or row.get('contact_vl') or
                row.get('contact') or row.get('CON') or 0)
    pwr = float(row.get('pow_vl') or row.get('POW vL') or row.get('power_vl') or
                row.get('power') or row.get('POW') or 0)
    eye = float(row.get('eye_vl') or row.get('EYE vL') or row.get('eye_vl') or
                row.get('eye') or row.get('EYE') or 0)
    gap = float(row.get('gap_power') or row.get('Gap') or row.get('GAP') or 0)
    avk = float(row.get('avoid_ks') or row.get('Avoid Ks') or row.get("K's") or 0)
    bab = float(row.get('babip') or row.get('BABIP') or 0)
    defense = float(row.get('defense_score') or calc_defense_score(row))
    speed_score = float(row.get('speed_score') or calc_speed_score(row))

    try:
        meta = (_diminished(gap) * weights.get('gap_power', DEFAULT_BATTING_WEIGHTS['gap_power']) +
                _diminished(con) * weights.get('contact', DEFAULT_BATTING_WEIGHTS['contact']) +
                _diminished(avk) * weights.get('avoid_ks', DEFAULT_BATTING_WEIGHTS['avoid_ks']) +
                _diminished(eye) * weights.get('eye', DEFAULT_BATTING_WEIGHTS['eye']) +
                _diminished(pwr) * weights.get('power', DEFAULT_BATTING_WEIGHTS['power']) +
                _diminished(bab) * weights.get('babip', DEFAULT_BATTING_WEIGHTS['babip']) +
                defense * weights.get('defense', DEFAULT_BATTING_WEIGHTS['defense']))

        spd_weight = weights.get('speed_stealing', DEFAULT_BATTING_WEIGHTS['speed_stealing'])
        if speed_score > 0 and spd_weight > 0:
            meta += _diminished(speed_score + 70) * spd_weight

        floor = BATTING_STAT_FLOOR
        key_stats = [con, gap]
        for stat in key_stats:
            if 0 < stat < floor:
                meta -= (floor - stat) * 0.4
    except (ValueError, TypeError):
        meta = 0.0

    return round(meta, 2)


def _pitching_split_meta(stu, mov, ctrl, phr, stamina, hold, weights):
    """Shared logic for pitching split metas (vs LHB / vs RHB)."""
    try:
        meta = (_diminished(mov) * weights.get('movement', DEFAULT_PITCHING_WEIGHTS['movement']) +
                _diminished(stu) * weights.get('stuff', DEFAULT_PITCHING_WEIGHTS['stuff']) +
                _diminished(ctrl) * weights.get('control', DEFAULT_PITCHING_WEIGHTS['control']) +
                _diminished(phr) * weights.get('p_hr', DEFAULT_PITCHING_WEIGHTS['p_hr']))

        # Interaction terms (fallbacks match DEFAULT_PITCHING_WEIGHTS)
        sxm_w = weights.get('stuff_x_movement', DEFAULT_PITCHING_WEIGHTS['stuff_x_movement'])
        sxc_w = weights.get('stuff_x_control', DEFAULT_PITCHING_WEIGHTS['stuff_x_control'])
        mxc_w = weights.get('movement_x_control', DEFAULT_PITCHING_WEIGHTS['movement_x_control'])
        if sxm_w > 0:
            meta += stu * mov * sxm_w
        if sxc_w > 0:
            meta += stu * ctrl * sxc_w
        if mxc_w > 0:
            meta += mov * ctrl * mxc_w

        sh_weight = weights.get('stamina_hold', DEFAULT_PITCHING_WEIGHTS['stamina_hold'])
        if sh_weight > 0:
            sh_avg, sh_count = 0, 0
            if stamina > 0: sh_avg += stamina; sh_count += 1
            if hold > 0: sh_avg += hold; sh_count += 1
            if sh_count > 0: meta += (sh_avg / sh_count) * sh_weight

        floor = PITCHING_STAT_FLOOR
        for stat in [stu, mov, ctrl]:
            if 0 < stat < floor:
                meta -= (floor - stat) * 1.0
    except (ValueError, TypeError):
        meta = 0.0
    return round(meta, 2)


def calc_pitching_meta_vs_lhb(row: dict, weights: dict = None) -> float:
    """Calculate pitcher meta score vs left-handed batters."""
    if weights is None:
        weights = get_pitching_weights_for(row)

    stu = float(row.get('stu_vl') or row.get('STU vL') or row.get('stuff_vl') or
                row.get('stuff') or row.get('STU') or 0)
    mov = float(row.get('movement') or row.get('Movement') or row.get('MOV') or 0)
    ctrl = float(row.get('control') or row.get('Control') or row.get('CON') or 0)
    phr = float(row.get('p_hr') or row.get('pHR') or row.get('HRA') or 0)
    stamina = float(row.get('stamina') or row.get('Stamina') or row.get('STA') or 0)
    hold = float(row.get('hold') or row.get('Hold') or 0)
    return _pitching_split_meta(stu, mov, ctrl, phr, stamina, hold, weights)


def calc_pitching_meta_vs_rhb(row: dict, weights: dict = None) -> float:
    """Calculate pitcher meta score vs right-handed batters."""
    if weights is None:
        weights = get_pitching_weights_for(row)

    stu = float(row.get('stu_vr') or row.get('STU vR') or row.get('stuff_vr') or
                row.get('stuff') or row.get('STU') or 0)
    mov = float(row.get('movement') or row.get('Movement') or row.get('MOV') or 0)
    ctrl = float(row.get('control') or row.get('Control') or row.get('CON') or 0)
    phr = float(row.get('p_hr') or row.get('pHR') or row.get('HRA') or 0)
    stamina = float(row.get('stamina') or row.get('Stamina') or row.get('STA') or 0)
    hold = float(row.get('hold') or row.get('Hold') or 0)
    return _pitching_split_meta(stu, mov, ctrl, phr, stamina, hold, weights)


# ════════════════════════════════════════════════════════════════════════════
# META EXPLAINER — produces a structured breakdown of how a meta score was
# computed. Used by the "Why this rec?" UX on Buy/Sell rows so users can see
# the formula contributions instead of treating meta as a black box.
# ════════════════════════════════════════════════════════════════════════════

def explain_batting_meta(row: dict, weights: dict = None) -> dict:
    """Return a structured breakdown of a batter's meta contributions.

    The breakdown mirrors ``calc_batting_meta`` so the totals reconcile.
    Used by the Card Detail "Why this meta?" expander and by the Buy/Sell row
    explainers so users can see exactly which ratings drove the score.

    Returns dict with::
        total: rounded total (float)
        components: list of {label, raw, weight, points} sorted by abs(points)
        bonuses: list of {label, points} for non-rating bonuses
        penalties: list of {label, points} for floor penalties
        notes: list of strings explaining special rules that fired
    """
    if weights is None:
        weights, _ = get_weights()

    # Mirror calc_batting_meta: platoon-adjusted stats
    gap_ovr = float(row.get('gap_power') or row.get('Gap') or row.get('GAP') or 0)
    con_ovr = float(row.get('contact') or row.get('Contact') or row.get('CON') or 0)
    eye_ovr = float(row.get('eye') or row.get('Eye') or row.get('EYE') or 0)
    pwr_ovr = float(row.get('power') or row.get('Power') or row.get('POW') or 0)
    gap = _apply_splits(row, gap_ovr, 'gap_vl', 'gap_vr', _BATTER_VS_RHP_WEIGHT)
    con = _apply_splits(row, con_ovr, 'contact_vl', 'contact_vr', _BATTER_VS_RHP_WEIGHT)
    eye = _apply_splits(row, eye_ovr, 'eye_vl', 'eye_vr', _BATTER_VS_RHP_WEIGHT)
    pwr = _apply_splits(row, pwr_ovr, 'power_vl', 'power_vr', _BATTER_VS_RHP_WEIGHT)
    avk = float(row.get('avoid_ks') or row.get('Avoid Ks') or row.get("K's") or 0)
    bab = float(row.get('babip') or row.get('BABIP') or 0)
    defense = float(row.get('defense_score') or calc_defense_score(row))
    speed_score = float(row.get('speed_score') or calc_speed_score(row))

    rating_components = [
        ("Contact",  con,     weights.get('contact', DEFAULT_BATTING_WEIGHTS['contact']),   _diminished(con)  * weights.get('contact', DEFAULT_BATTING_WEIGHTS['contact'])),
        ("Gap Power", gap,    weights.get('gap_power', DEFAULT_BATTING_WEIGHTS['gap_power']), _diminished(gap) * weights.get('gap_power', DEFAULT_BATTING_WEIGHTS['gap_power'])),
        ("Power",    pwr,     weights.get('power', DEFAULT_BATTING_WEIGHTS['power']),       _diminished(pwr)  * weights.get('power', DEFAULT_BATTING_WEIGHTS['power'])),
        ("Defense",  defense, weights.get('defense', DEFAULT_BATTING_WEIGHTS['defense']),   defense           * weights.get('defense', DEFAULT_BATTING_WEIGHTS['defense'])),
        ("Eye",      eye,     weights.get('eye', DEFAULT_BATTING_WEIGHTS['eye']),           _diminished(eye)  * weights.get('eye', DEFAULT_BATTING_WEIGHTS['eye'])),
        ("Avoid K's", avk,    weights.get('avoid_ks', DEFAULT_BATTING_WEIGHTS['avoid_ks']), _diminished(avk)  * weights.get('avoid_ks', DEFAULT_BATTING_WEIGHTS['avoid_ks'])),
        ("BABIP",    bab,     weights.get('babip', DEFAULT_BATTING_WEIGHTS['babip']),       _diminished(bab)  * weights.get('babip', DEFAULT_BATTING_WEIGHTS['babip'])),
    ]
    components = [
        {"label": label, "raw": round(raw, 1), "weight": weight, "points": round(points, 1)}
        for label, raw, weight, points in rating_components
        if points != 0
    ]

    bonuses, penalties, notes = [], [], []

    # Show platoon info when effective differs from overall
    for stat_name, eff, ovr_val in [("Contact", con, con_ovr), ("Power", pwr, pwr_ovr),
                                     ("Eye", eye, eye_ovr), ("Gap", gap, gap_ovr)]:
        if abs(eff - ovr_val) > 1:
            notes.append(f"{stat_name}: {ovr_val:.0f} overall -> {eff:.0f} effective (platoon-weighted)")

    spd_weight = weights.get('speed_stealing', DEFAULT_BATTING_WEIGHTS['speed_stealing'])
    if speed_score > 0 and spd_weight > 0:
        spd_points = _diminished(speed_score + 70) * spd_weight
        bonuses.append({"label": f"Speed/Stealing ({speed_score:.0f})", "points": round(spd_points, 1)})

    # Platoon penalty
    platoon_pen = _calc_platoon_penalty_batting(row)
    if platoon_pen < -0.5:
        penalties.append({"label": "Platoon split weakness", "points": round(platoon_pen, 1)})
        splits_detail = []
        for stat_name, vl_key, vr_key in [('Contact', 'contact_vl', 'contact_vr'),
                                            ('Power', 'power_vl', 'power_vr'),
                                            ('Eye', 'eye_vl', 'eye_vr')]:
            vl = row.get(vl_key)
            vr = row.get(vr_key)
            if vl is not None and vr is not None:
                vl_f, vr_f = float(vl or 0), float(vr or 0)
                if abs(vl_f - vr_f) > 15:
                    splits_detail.append(f"{stat_name}: {vl_f:.0f} vL / {vr_f:.0f} vR")
        if splits_detail:
            notes.append(f"Weak-side exposure: {'; '.join(splits_detail)}")

    # Balance penalty (uses effective stats)
    floor = BATTING_STAT_FLOOR
    for stat_name, stat_val, coeff in [("Contact", con, 0.4), ("Gap Power", gap, 0.4), ("Power", pwr, 1.0)]:
        if 0 < stat_val < floor:
            pen = (floor - stat_val) * coeff
            penalties.append({"label": f"{stat_name} below floor ({stat_val:.0f}<{floor})", "points": -round(pen, 1)})

    # Position flexibility — calculated for info but NOT added to the meta
    # total (v6: removed from WAR-prediction formula due to r=-0.023 signal).
    flex_bonus = _calc_position_flexibility(row)
    if flex_bonus > 0:
        notes.append(
            f"Can play multiple positions (+{flex_bonus:.0f} would be the "
            "roster-utility bonus, but it's excluded from meta — not WAR-correlated)."
        )

    # Performance overlay
    perf_adj = _calc_performance_adjustment_batting(row)
    if abs(perf_adj) > 0.5:
        woba = row.get('adv_woba')
        label = f"wOBA performance ({float(woba):.3f})" if woba else "Performance adjustment"
        if perf_adj > 0:
            bonuses.append({"label": label, "points": round(perf_adj, 1)})
        else:
            penalties.append({"label": label, "points": round(perf_adj, 1)})

    # Superstat overlay (EV + LD% from game logs)
    ss_adj = _calc_superstat_overlay_batting(row)
    if abs(ss_adj) > 0.5:
        ev = row.get('_gl_ev_avg') or 0
        ld = row.get('_gl_ld_pct') or 0
        ab = row.get('_gl_at_bats') or 0
        label = f"Game-log EV/LD% (EV {ev:.0f}, LD% {ld:.0f} over {ab} AB)"
        if ss_adj > 0:
            bonuses.append({"label": label, "points": round(ss_adj, 1)})
        else:
            penalties.append({"label": label, "points": round(ss_adj, 1)})

    # Clutch overlay (2-out RBI vs LOB-RISP-2-out from box scores)
    clutch_adj = _calc_clutch_overlay_batting(row)
    if abs(clutch_adj) > 0.5:
        rbi = row.get('_clutch_2out_rbi') or 0
        lob = row.get('_clutch_lob_risp_2out') or 0
        label = f"Clutch (2-out RBI {rbi} - LOB-RISP-2out {lob})"
        if clutch_adj > 0:
            bonuses.append({"label": label, "points": round(clutch_adj, 1)})
        else:
            penalties.append({"label": label, "points": round(clutch_adj, 1)})

    # Defensive error-rate overlay
    err_adj = _calc_error_overlay_batting(row)
    if abs(err_adj) > 0.5:
        errs = row.get('_errors_count') or 0
        games = row.get('_def_games_count') or 0
        rate = errs / games if games else 0
        label = f"Defensive errors ({errs} in {games}G, rate {rate:.2f}/G)"
        if err_adj > 0:
            bonuses.append({"label": label, "points": round(err_adj, 1)})
        else:
            penalties.append({"label": label, "points": round(err_adj, 1)})

    # Recent-form (last 30 days) overlay — 2026-04-24
    rf_adj = _calc_recent_form_overlay_batting(row)
    if abs(rf_adj) > 0.5:
        rf_pa = row.get('_rf_pa') or 0
        rec_obp = row.get('_rf_obp_proxy') or 0
        season_obp = row.get('_rf_season_obp') or 0
        delta = (rec_obp - season_obp)
        sign = '+' if delta >= 0 else ''
        label = (f"Recent form (30d OBP {rec_obp:.3f} vs season {season_obp:.3f}, "
                 f"{sign}{delta:.3f} over {rf_pa} PA)")
        if rf_adj > 0:
            bonuses.append({"label": label, "points": round(rf_adj, 1)})
        else:
            penalties.append({"label": label, "points": round(rf_adj, 1)})

    pos = row.get('position') or row.get('Position') or 0
    if isinstance(pos, str):
        pos_map = {"C": 2, "1B": 3, "2B": 4, "3B": 5, "SS": 6,
                   "LF": 7, "CF": 8, "RF": 9, "DH": 10}
        pos_int = pos_map.get(pos, 0)
    else:
        try:
            pos_int = int(pos)
        except (ValueError, TypeError):
            pos_int = 0
    pos_bonus = POSITIONAL_VALUE_BONUS.get(pos_int, 0)
    if pos_bonus:
        pos_label_map = {2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS",
                         7: "LF", 8: "CF", 9: "RF", 10: "DH"}
        pos_label = pos_label_map.get(pos_int, "?")
        bonuses.append({"label": f"Position scarcity ({pos_label})", "points": pos_bonus})
        if pos_bonus > 0:
            notes.append(f"Premium defensive position ({pos_label}) gets +{pos_bonus} fWAR-ladder bonus.")
        elif pos_bonus < 0:
            notes.append(f"Bat-only position ({pos_label}) gets {pos_bonus} fWAR-ladder penalty.")

    if any(c['raw'] > DIMINISHING_RETURNS_THRESHOLD for c in components if c['label'] != 'Defense'):
        notes.append(f"Ratings above {DIMINISHING_RETURNS_THRESHOLD} get sqrt-scaled to prevent extreme spikes from dominating.")

    # Card-type offset (after all other components, before the sort)
    ct = row.get('card_type')
    card_type_offset = 0
    if ct is not None:
        try:
            card_type_offset = BATTING_CARD_TYPE_OFFSET.get(int(ct), 0)
        except (ValueError, TypeError):
            pass
    if card_type_offset:
        ct_name_map = {1:'Live', 2:'Legend', 3:'Hardware Heroes', 4:'Unsung Heroes',
                       5:'Historical All-Star', 6:'Future Legend', 7:'Snapshot',
                       8:'Veteran Presence', 9:'All-Time Legend', 10:'Live Reward'}
        ct_label = ct_name_map.get(int(ct), f'type={ct}')
        entry = {"label": f"Card-type bias ({ct_label})", "points": card_type_offset}
        if card_type_offset > 0:
            bonuses.append(entry)
        else:
            penalties.append(entry)
        notes.append(f"{ct_label} cards empirically beat/lag meta by {card_type_offset:+d} (from residual analysis).")

    components.sort(key=lambda c: abs(c['points']), reverse=True)
    total = sum(c['points'] for c in components) + sum(b['points'] for b in bonuses) + sum(p['points'] for p in penalties)

    # ── OVR diagnostic (not a formula input) ──
    # Surfaced so the UI can show "OOTP says 80 OVR, our meta says X" side by
    # side. Large divergences are the interesting signal — that's the case
    # where the card ratings and the game's black-box quality estimate
    # disagree. Never added into ``total``.
    ovr_val = row.get('ovr') or row.get('card_value') or row.get('OVR')
    diagnostics = {"ovr": int(float(ovr_val)) if ovr_val else None}

    return {
        "total": round(total, 1),
        "components": components,
        "bonuses": bonuses,
        "penalties": penalties,
        "notes": notes,
        "diagnostics": diagnostics,
    }


def explain_pitching_meta(row: dict, weights: dict = None) -> dict:
    """Return a structured breakdown of a pitcher's meta contributions.

    Mirrors ``calc_pitching_meta`` so the totals reconcile. Uses the
    role-aware dispatcher when no explicit weights are passed, so the
    explainer shows the same coefficients that actually drove the score.
    """
    if weights is None:
        weights = get_pitching_weights_for(row)

    is_sp = _is_starting_pitcher(row)

    # Mirror calc_pitching_meta: platoon-adjusted stats
    mov_ovr = float(row.get('movement') or row.get('Movement') or row.get('MOV') or 0)
    stu_ovr = float(row.get('stuff') or row.get('Stuff') or row.get('STU') or 0)
    ctrl_ovr = float(row.get('control') or row.get('Control') or row.get('CON') or 0)
    mov = _apply_splits(row, mov_ovr, 'movement_vl', 'movement_vr', _PITCHER_VS_RHB_WEIGHT)
    stu = _apply_splits(row, stu_ovr, 'stuff_vl', 'stuff_vr', _PITCHER_VS_RHB_WEIGHT)
    ctrl = _apply_splits(row, ctrl_ovr, 'control_vl', 'control_vr', _PITCHER_VS_RHB_WEIGHT)
    phr = float(row.get('p_hr') or row.get('pHR') or row.get('HRA') or 0)
    stamina = float(row.get('stamina') or row.get('Stamina') or row.get('STA') or 0)
    hold = float(row.get('hold') or row.get('Hold') or 0)

    rating_components = [
        ("Movement", mov,  weights.get('movement', DEFAULT_PITCHING_WEIGHTS['movement']), _diminished(mov)  * weights.get('movement', DEFAULT_PITCHING_WEIGHTS['movement'])),
        ("HR Suppression", phr, weights.get('p_hr', DEFAULT_PITCHING_WEIGHTS['p_hr']),     _diminished(phr)  * weights.get('p_hr', DEFAULT_PITCHING_WEIGHTS['p_hr'])),
        ("Stuff",    stu,  weights.get('stuff', DEFAULT_PITCHING_WEIGHTS['stuff']),        _diminished(stu)  * weights.get('stuff', DEFAULT_PITCHING_WEIGHTS['stuff'])),
        ("Control",  ctrl, weights.get('control', DEFAULT_PITCHING_WEIGHTS['control']),    _diminished(ctrl) * weights.get('control', DEFAULT_PITCHING_WEIGHTS['control'])),
    ]
    components = [
        {"label": label, "raw": round(raw, 1), "weight": weight, "points": round(points, 1)}
        for label, raw, weight, points in rating_components
        if points != 0
    ]

    bonuses, penalties, notes = [], [], []

    # Show platoon info when effective differs from overall
    for stat_name, eff, ovr_val in [("Stuff", stu, stu_ovr), ("Movement", mov, mov_ovr),
                                     ("Control", ctrl, ctrl_ovr)]:
        if abs(eff - ovr_val) > 1:
            notes.append(f"{stat_name}: {ovr_val:.0f} overall -> {eff:.0f} effective (platoon-weighted)")

    sxm_w = weights.get('stuff_x_movement', DEFAULT_PITCHING_WEIGHTS['stuff_x_movement'])
    sxc_w = weights.get('stuff_x_control', DEFAULT_PITCHING_WEIGHTS['stuff_x_control'])
    mxc_w = weights.get('movement_x_control', DEFAULT_PITCHING_WEIGHTS['movement_x_control'])
    if sxm_w > 0 and stu and mov:
        bonuses.append({"label": f"Stuff x Movement (Ks + weak contact)", "points": round(stu * mov * sxm_w, 1)})
    if sxc_w > 0 and stu and ctrl:
        bonuses.append({"label": f"Stuff x Control (dominant + commanded)", "points": round(stu * ctrl * sxc_w, 1)})
    if mxc_w > 0 and mov and ctrl:
        bonuses.append({"label": f"Movement x Control (groundballs)", "points": round(mov * ctrl * mxc_w, 1)})

    sh_weight = weights.get('stamina_hold', DEFAULT_PITCHING_WEIGHTS['stamina_hold'])
    if sh_weight > 0:
        sh_avg, sh_count = 0, 0
        if stamina > 0: sh_avg += stamina; sh_count += 1
        if hold > 0: sh_avg += hold; sh_count += 1
        if sh_count > 0:
            sh_points = (sh_avg / sh_count) * sh_weight
            bonuses.append({"label": f"Stamina/Hold ({sh_avg / sh_count:.0f})", "points": round(sh_points, 1)})

    # Platoon penalty
    platoon_pen = _calc_platoon_penalty_pitching(row)
    if platoon_pen < -0.5:
        penalties.append({"label": "Platoon split weakness", "points": round(platoon_pen, 1)})
        splits_detail = []
        for stat_name, vl_key, vr_key in [('Stuff', 'stuff_vl', 'stuff_vr'),
                                            ('Control', 'control_vl', 'control_vr')]:
            vl = row.get(vl_key)
            vr = row.get(vr_key)
            if vl is not None and vr is not None:
                vl_f, vr_f = float(vl or 0), float(vr or 0)
                if abs(vl_f - vr_f) > 15:
                    splits_detail.append(f"{stat_name}: {vl_f:.0f} vL / {vr_f:.0f} vR")
        if splits_detail:
            notes.append(f"Weak-side exposure: {'; '.join(splits_detail)}")

    floor = PITCHING_STAT_FLOOR
    for stat_name, stat_val in [("Stuff", stu), ("Movement", mov), ("Control", ctrl)]:
        if 0 < stat_val < floor:
            pen = (floor - stat_val) * 1.0
            penalties.append({"label": f"{stat_name} below floor ({stat_val:.0f}<{floor})", "points": -round(pen, 1)})

    # Arsenal diversity — computed for info but NOT added to the total
    # (v6: removed from WAR-prediction meta due to r=+0.024 with WAR).
    arsenal_bonus = _calc_arsenal_factor(row, is_sp=is_sp)
    if arsenal_bonus > 0:
        role_label = "SP" if is_sp else "RP"
        notes.append(
            f"Diverse {role_label} arsenal (+{arsenal_bonus:.0f} theoretical "
            "lineup-trip bonus, but excluded from meta — not WAR-correlated)."
        )

    # Performance overlay
    perf_adj = _calc_performance_adjustment_pitching(row)
    if abs(perf_adj) > 0.5:
        siera = row.get('adv_siera')
        label = f"SIERA performance ({float(siera):.2f})" if siera else "Performance adjustment"
        if perf_adj > 0:
            bonuses.append({"label": label, "points": round(perf_adj, 1)})
        else:
            penalties.append({"label": label, "points": round(perf_adj, 1)})

    # Superstat overlay (pitcher)
    p_ss_adj = _calc_superstat_overlay_pitching(row)
    if abs(p_ss_adj) > 0.5:
        ev = row.get('_gl_p_ev_allowed') or 0
        ld = row.get('_gl_p_ld_pct_allowed') or 0
        bf = row.get('_gl_p_batters_faced') or 0
        kp = row.get('_gl_p_k_pct')
        bbp = row.get('_gl_p_bb_pct')
        rate_piece = ''
        if kp is not None and bbp is not None:
            rate_piece = f", K% {kp:.0f}, BB% {bbp:.0f}"
        label = f"Game-log contact allowed (EV {ev:.0f}, LD% {ld:.0f}{rate_piece} over {bf} BF)"
        if p_ss_adj > 0:
            bonuses.append({"label": label, "points": round(p_ss_adj, 1)})
        else:
            penalties.append({"label": label, "points": round(p_ss_adj, 1)})

    # Reliever discipline overlay (inherited-runners-scored rate)
    rd_adj = _calc_reliever_discipline_overlay_pitching(row)
    if abs(rd_adj) > 0.5:
        inh = row.get('_inherited_runners') or 0
        sc = row.get('_inherited_scored') or 0
        rate = (100 * sc / inh) if inh else 0
        label = f"Inherited runners stranded ({sc}/{inh} scored, {rate:.0f}% vs 30% baseline)"
        if rd_adj > 0:
            bonuses.append({"label": label, "points": round(rd_adj, 1)})
        else:
            penalties.append({"label": label, "points": round(rd_adj, 1)})

    # SP HR/9 overlay (SPs only — RPs no-op)
    hr9_adj = _calc_hr9_overlay_pitching(row)
    if abs(hr9_adj) > 0.5:
        delta = row.get('_obs_hr9_delta') or 0
        ip = row.get('_obs_hr9_ip') or 0
        sign = '+' if delta >= 0 else ''
        label = f"SP HR/9 vs league ({sign}{delta:.2f} HR/9 over {ip:.0f} IP)"
        if hr9_adj > 0:
            bonuses.append({"label": label, "points": round(hr9_adj, 1)})
        else:
            penalties.append({"label": label, "points": round(hr9_adj, 1)})

    # Recent-form (last 30 days) overlay — 2026-04-24
    p_rf_adj = _calc_recent_form_overlay_pitching(row)
    if abs(p_rf_adj) > 0.5:
        rf_ip = row.get('_rf_ip') or 0
        rec_era = row.get('_rf_era') or 0
        season_era = row.get('_rf_season_era') or 0
        delta = (season_era - rec_era)
        sign = '+' if delta >= 0 else ''
        label = (f"Recent form (30d ERA {rec_era:.2f} vs season {season_era:.2f}, "
                 f"{sign}{delta:.2f} over {rf_ip:.1f} IP)")
        if p_rf_adj > 0:
            bonuses.append({"label": label, "points": round(p_rf_adj, 1)})
        else:
            penalties.append({"label": label, "points": round(p_rf_adj, 1)})

    if bonuses and any('x' in b['label'].lower() for b in bonuses):
        notes.append("Pitching is non-additive — interaction terms reflect SIERA-style synergy.")

    # Pitching card-type offset
    ct = row.get('card_type')
    card_type_offset = 0
    if ct is not None:
        try:
            card_type_offset = PITCHING_CARD_TYPE_OFFSET.get(int(ct), 0)
        except (ValueError, TypeError):
            pass
    if card_type_offset:
        ct_name_map = {1:'Live', 2:'Legend', 3:'Hardware Heroes', 4:'Unsung Heroes',
                       5:'Historical All-Star', 6:'Future Legend', 7:'Snapshot',
                       8:'Veteran Presence', 9:'All-Time Legend', 10:'Live Reward'}
        ct_label = ct_name_map.get(int(ct), f'type={ct}')
        entry = {"label": f"Card-type bias ({ct_label})", "points": card_type_offset}
        if card_type_offset > 0:
            bonuses.append(entry)
        else:
            penalties.append(entry)
        notes.append(f"{ct_label} pitchers empirically beat/lag meta by {card_type_offset:+d} (from residual analysis).")

    components.sort(key=lambda c: abs(c['points']), reverse=True)
    total = sum(c['points'] for c in components) + sum(b['points'] for b in bonuses) + sum(p['points'] for p in penalties)

    # ── OVR diagnostic (not a formula input) ──
    # Same contract as the batter explainer: surface OVR so the UI can flag
    # divergences between meta and OVR, but never add it into ``total``.
    ovr_val = row.get('ovr') or row.get('card_value') or row.get('OVR')
    diagnostics = {"ovr": int(float(ovr_val)) if ovr_val else None}

    return {
        "total": round(total, 1),
        "components": components,
        "bonuses": bonuses,
        "penalties": penalties,
        "notes": notes,
        "diagnostics": diagnostics,
    }


def explain_meta(card: dict, is_pitcher: bool = None, weights: dict = None) -> dict:
    """Convenience dispatcher — picks batting or pitching explainer based on the card."""
    if is_pitcher is None:
        is_pitcher = bool(card.get('pitcher_role') or card.get('pitcher_role_name'))
    if is_pitcher:
        return explain_pitching_meta(card, weights)
    return explain_batting_meta(card, weights)
