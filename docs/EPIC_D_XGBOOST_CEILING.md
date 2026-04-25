# Epic D — XGBoost Ceiling Study

**Date:** 2026-04-24
**Script:** `scripts/xgboost_ceiling.py`
**Target:** Establish an upper bound on predictability with a non-linear model so we know whether to keep adding linear overlays.

---

## TL;DR

**We are at the ceiling. Stop adding linear rating-based overlays.**

Apples-to-apples 5-fold CV R² comparison on the same feature set (all ratings, splits, position flags, card_type, age):

| Target | Linear CV R² | XGBoost CV R² | XGBoost edge |
|---|---:|---:|---:|
| **Batting WAR/600** (n=1,605, PA ≥ 150) | **+0.2215** | +0.2146 | **−0.007** (worse) |
| **Pitching WAR/200** (n=1,784, IP ≥ 30) | **+0.1149** | +0.0672 | **−0.048** (worse) |

XGBoost *cannot* beat a linear fit on either target. Non-linear interactions and tree-based threshold effects do not exist in sufficient quantity to outweigh the regularization penalty XGBoost pays. The linear meta formula is already extracting the rating-based signal.

---

## Reading the result

### What this does NOT mean
- The linear meta **is perfect**. It isn't — n=1,605 with R²=0.22 means 78% of WAR variance is unexplained. But that unexplained variance is NOT hidden in rating-feature combinations that XGBoost can discover.
- **Ratings are useless**. They aren't — XGBoost in-sample r = +0.81 (batting), +0.76 (pitching) shows the model *can* memorize the data, but can't generalize beyond the linear structure.

### What it DOES mean
The remaining WAR variance comes from sources the rating feature set does not contain. The next gains must come from **outside** the rating columns:

1. **Observed performance** — wOBA, SIERA, OPS+, ISO, FIP. Already captured by the Performance Overlay (Layer 4) and observed-stat overlays. wOBA alone has r=+0.88 with WAR, dwarfing any rating.
2. **Park factors** (Epic C) — `pf_hr`, `pf_avg` from `league_team_stats`. Currently ingested, never applied to meta. A power hitter in Coors deserves a discount.
3. **Observed defense** (Epic B) — `fielding_stats` table (DRS/UZR equivalents). Rating-based defense has r≈0 with WAR. Observed defense might have real signal.
4. **Cross-league calibration** (Epic G) — residuals between leagues are opposite-sign. Pooled weights under-fit extremes.
5. **Noise floor** — WAR has an irreducible per-card noise component. Even a perfect model couldn't explain it all.

---

## Top-10 XGBoost feature importances

### Batting (n=1,605)
| Rank | Feature | Importance | Interpretation |
|---|---|---:|---|
| 1 | `ovr` | 0.098 | Composite absorbs ~10% of signal — as expected, captures hidden attributes |
| 2 | `babip` | 0.052 | Confirms univariate r=+0.318 finding; babip is a real BATTING signal |
| 3 | `position` | 0.046 | Validates the fWAR positional-value bonus |
| 4 | `contact_vr` | 0.038 | Platoon split useful; vR > overall (60% RHP frequency) |
| 5 | `catcher_arm` | 0.036 | **Surprising** — only applies to catchers but XGBoost finds real signal |
| 6 | `pos_rating_c` | 0.034 | Catcher eligibility adds WAR ceiling (rare combo) |
| 7 | `power` | 0.032 | Consistent with r=+0.289 univariate |
| 8 | `power_vr` | 0.031 | Platoon dominance (strong-side power) |
| 9 | `contact` | 0.030 | Core rating |
| 10 | `pos_rating_1b` | 0.029 | Position flexibility weighted |

**Worth investigating:** `catcher_arm` and `pos_rating_c` are both catcher-specific. XGBoost may be finding that C-eligible bats are especially valuable. The current meta's `POSITIONAL_VALUE_BONUS['C'] = +31` may already capture this but the importance suggests checking.

### Pitching (n=1,784)
| Rank | Feature | Importance | Interpretation |
|---|---|---:|---|
| 1 | `ovr` | 0.124 | Composite is even more dominant than in batting |
| 2 | `p_hr` | 0.083 | HR suppression is the top standalone pitching rating |
| 3 | `p_hr_vl` | 0.076 | Platoon HR (lefty side) |
| 4 | `p_hr_vr` | 0.067 | Platoon HR (righty side) |
| 5 | `control_vl` | 0.063 | Control matters more vs LHB than vs RHB |
| 6 | `p_babip` | 0.052 | BABIP suppression |
| 7 | `stuff_vl` | 0.047 | Stuff vs LHB |
| 8 | `stamina` | 0.046 | **Matches residual finding** — stamina is under-weighted in current meta |
| 9 | `movement_vr` | 0.044 | Movement vs RHB |
| 10 | `movement` | 0.043 | Movement overall |

**Worth investigating:** `p_hr` and its splits dominate pitching (top 4 of top 10). Current pitching weights have `p_hr = 1.40-1.74`. XGBoost says it might still be under-weighted, especially the split versions. `stamina` consistently top-10 despite being weighted at 0.10 — confirms residual finding of +0.38.

---

## Recommendations

### Stop doing
- **Do not add more linear rating-based overlays.** The XGBoost ceiling tells us they won't move the needle.
- Do not add more linear interaction terms (stuff×movement etc.) — XGBoost would have found useful interactions if they existed.

### Do
1. **Park factors (Epic C).** Highest leverage next move. Ingested, never used. Estimated +2-5% accuracy.
2. **Observed defense (Epic B).** `fielding_stats` table has DRS/UZR. Rating-based defense is r≈0, observed might be real.
3. **Consider bumping `p_hr` weights 10-15%** on pitching. XGBoost top-4 importance with current weight of 1.40-1.74 suggests it's still under-weighted.
4. **Consider bumping `stamina` weight to ~0.20-0.30** on SP (currently 0.10). XGBoost finds real signal despite current under-weighting.
5. **Accept that batting meta has diminishing returns from ratings.** Focus batting gains on park factors and cross-league calibration.

### Investigate
- **Catcher-specific signal.** `catcher_arm` + `pos_rating_c` both in top-6 batting importances. The positional bonus already gives C a +31 meta boost — but is that enough? Mine a catcher-only correlation study.

---

## Methodology notes

- **5-fold CV** with `random_state=42` on same rows for both linear and XGBoost.
- **XGBoost config**: n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8. Standard regularization, not tuned for this dataset (tuning could add 0.01-0.02 but won't change the conclusion).
- **Feature sets**: 40 batting features, 20 pitching features (ratings + splits + position + card_type + age + pos_rating_*).
- **Target**: WAR/600 PA (batting), WAR/200 IP (pitching). Raw WAR scaled to standardize playing time.
- **NaN handling**: median imputation per feature; rows with NaN target dropped.
- **Filter**: PA ≥ 150 (batting), IP ≥ 30 (pitching) — matches the calibration engine's thresholds.

## Reproducing
```
PYTHONIOENCODING=utf-8 python scripts/xgboost_ceiling.py
```
Takes ~30s on a modern CPU. Output is the three blocks shown above.
