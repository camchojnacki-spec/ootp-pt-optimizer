# OOTP 27 Perfect Team — Meta Knowledge Dump

**Last updated:** 2026-04-17
**Purpose:** Hand-off document for external research agents. Contains the full
state of meta analysis, residual findings, overlay coefficients, data
statistics, and open research questions. Pair with the SQLite DB (`app.db`)
for reproducible analysis.

---

## 1. Core Thesis

**OVR is a diagnostic column, not a formula term.** The meta score is what
the app uses to predict WAR per standardized PA/BF. OVR is displayed
alongside it for sanity checking, but the meta must **independently beat OVR**
when validated against observed WAR — not rationalize it.

Target metrics:
- **Batters:** WAR / 600 PA
- **Pitchers:** WAR / 200 IP

---

## 2. Data Inventory (as of 2026-04-17)

### Structured stats
| Table | Rows | Notes |
|---|---|---|
| `cards` | ~7,000+ | Every ratings record (card_id keyed) |
| `batting_stats` | 1,306 seasons w/ PA≥150 | Pooled lb124 + i76 |
| `pitching_stats` | 1,229 seasons w/ IP≥30 | Pooled |
| `batting_stats_adv` | ~500 | wOBA, wpa |
| `pitching_stats_adv` | 2,212 | SIERA |
| `pitch_ratings` | ~1,200 | Per-pitch grades (fb, sl, cb, etc.) |
| `league_team_stats` | — | Park factors |

### Game-log / box score extractions
| Table | Rows | Extracted from |
|---|---|---|
| `games` | 110 | HTML box scores |
| `game_batting` | 2,227 | Per-player per-game batting lines |
| `game_pitching` | 898 | Per-player per-game pitching lines |
| `game_log_at_bats` | 8,621 | Pitch-by-pitch ABs (100% card_id resolved) |
| `game_narratives` | 229 | Recap text for sentiment (unused by meta) |
| `game_clutch_events` | 4,680 | 2-out RBI, LOB-RISP-2out, errors, SB, HR, inherited runners |

### Leagues
| League | Tier | Batter-seasons | Pitcher-seasons |
|---|---|---|---|
| lb124 | Low Bronze | 895 | 988 |
| i76 | Independent | 381 | 402 |

---

## 3. Meta Formula Structure

### Batting (`calc_batting_meta` in `app/core/meta_scoring.py`)

```
meta =
    _diminished(gap)       × weights['gap_power']
  + _diminished(con)       × weights['contact']
  + _diminished(avk)       × weights['avoid_ks']
  + _diminished(eye)       × weights['eye']
  + _diminished(pwr)       × weights['power']
  + _diminished(bab)       × weights['babip']
  + defense                × weights['defense']
  + conditional speed      × weights['speed_stealing']
  + platoon_penalty                              (negative when splits extreme)
  + balance_penalty                              (negative when weak-side <floor)
  + performance_adjustment  (wOBA vs league, park-adjusted, ±40)
  + superstat_overlay       (EV, LD%, K%, BB% from game logs, ±N)
  + clutch_overlay          (2-out RBI vs LOB-RISP-2out, ±10)
  + error_overlay           (obs errors/game vs positional expectation, ±15)
  + iso_overlay             (tier+league-adjusted obs ISO delta, ±15)
  + opsplus_overlay         (obs OPS+ vs 100, ±25)
  + obp_overlay             (obs OBP delta vs league, ±20)
  + overperformance_overlay (obs OPS+ - rating-predicted OPS+, ±20)
  + rating_diminishing_penalty (contact×power, gap×power, ≤0)
  + positional_value_bonus  (POSITIONAL_VALUE_BONUS[pos])
  + card_type_offset        (BATTING_CARD_TYPE_OFFSET[card_type])
```

`_diminished(x)` = linear up to threshold (60), then gentler slope above.

### Pitching (`calc_pitching_meta`)

```
meta =
    _diminished(mov)  × weights['movement']
  + _diminished(stu)  × weights['stuff']
  + _diminished(ctrl) × weights['control']
  + _diminished(phr)  × weights['p_hr']
  + conditional       stuff×movement, stuff×control, movement×control
  + stamina_hold      × weights['stamina_hold']
  + platoon_penalty
  + balance_penalty
  + performance_adjustment  (SIERA vs league, ±35)
  + superstat_overlay       (EV-allowed, LD%-allowed, K%, BB%, ±N)
  + reliever_discipline     (inherited-runner strand rate, RP-only, ±10)
  + fip_overlay             (BF-weighted FIP delta, league-adjusted, ±35)
  + overperformance_overlay (obs ERA+ - rating-predicted, ±20)
  + stuff_diminishing       (stuff×mov, stuff×ctrl, ≤0)
  + stamina_bonus           (SP-only, (stamina-50)×coeff, ≤+15)
  + stuff_overcredit_damping (-(stuff-60)×coeff, ≤-6)
  + card_type_offset
```

---

## 4. Calibrated Weights (2026-04-17)

### Batting (Ridge → NNLS → Bayesian prior blend)
```python
{
  "contact":   2.52,   # was 1.80
  "power":     1.69,   # was 2.00
  "gap_power": 0.90,
  "eye":       0.92,
  "babip":     0.90,
  "avoid_ks":  0.00,   # zeroed by NNLS — significant finding
  "defense":   0.40,
  "speed_stealing": 0.15,
}
```

### Pitching (SP)
```python
{
  "stuff":       2.04,   # calibrated
  "movement":    0.60,
  "control":     0.42,
  "p_hr":        1.74,
  "stamina_hold": 0.10,  # strongly under-weighted — residual +0.38
  "stuff_x_movement": 0.008,
  "stuff_x_control":  0.010,
  "movement_x_control": 0.002,
}
```

**Key insight:** `avoid_ks=0` came out of NNLS — statistically it adds nothing
after contact is in the formula. This is counter-intuitive but validated.

---

## 5. Validation Results (Cross-League)

Pearson r of meta vs WAR/600 (batters) or WAR/200 (pitchers), compared to OVR.

| Slice | n | OVR r | Meta r | Δ vs OVR |
|---|---|---|---|---|
| lb124 batting | 330 | +0.540 | **+0.754** | **+0.215** |
| i76 batting | 381 | +0.530 | **+0.704** | **+0.174** |
| lb124 pitching | 379 | +0.478 | **+0.595** | **+0.117** |
| i76 pitching | 402 | +0.547 | **+0.622** | **+0.075** |

XGBoost CV R²:
- Batting: 0.142 (lower than linear 0.216 → we're at the ceiling)
- Pitching: 0.119 (lower than linear 0.149)

---

## 6. Overlay Catalog

All overlays live in `app/core/meta_scoring.py` and are injected via
`recalculate_all_meta_scores()` in `app/core/ingestion.py`.

### Batting overlays

| Function | Input data | Scale | Cap | Signal r (at intro) |
|---|---|---|---|---|
| `_calc_platoon_penalty_batting` | vL/vR splits | — | — | — |
| `_calc_performance_adjustment_batting` | `adv_woba`, `adv_pa`, park | 600× | ±40 | r=+0.88 wOBA→WAR |
| `_calc_superstat_overlay_batting` | `_gl_ev_avg`, `_gl_ld_pct`, `_gl_k_pct`, `_gl_bb_pct` | varies | ±N | — |
| `_calc_clutch_overlay_batting` | `_clutch_2out_rbi`, `_clutch_lob_risp_2out` | 15× | ±10 | — |
| `_calc_error_overlay_batting` | `_errors_count`, `_def_games_count`, position | 100× | ±15 | — |
| `_calc_iso_overlay_batting` | `_obs_iso_delta` (league+tier adj), `_obs_iso_pa` | 300× | ±15 | univariate +0.58 |
| `_calc_opsplus_overlay_batting` | `_obs_ops_plus`, `_obs_ops_plus_pa` | 0.6× | ±25 | residual +0.33 |
| `_calc_obp_overlay_batting` | `_obs_obp_delta`, `_obs_obp_pa` | 300× | ±20 | residual +0.34 |
| `_calc_babip_overlay_batting` | `_obs_babip_delta`, `_obs_babip_pa` | 200× | ±12 | residual +0.26 |
| `_calc_overperformance_overlay_batting` | `_over_ops_plus` (OLS residual) | 0.5× | ±20 | residual +0.55 |
| `_calc_rating_diminishing_penalty_batting` | contact×power / gap×power ≥70, plus standalone power>60 × 0.20 | — | −40 | residual −0.44 (contact×power), −0.30 (power) |

### Pitching overlays

| Function | Input data | Scale | Cap | Signal r (at intro) |
|---|---|---|---|---|
| `_calc_platoon_penalty_pitching` | stuff/control splits | — | — | — |
| `_calc_performance_adjustment_pitching` | `adv_siera`, `adv_ip` | — | ±35 | r=-0.65 SIERA→WAR |
| `_calc_superstat_overlay_pitching` | `_gl_p_ev_allowed`, `_gl_p_ld_pct_allowed`, etc. | varies | ±N | — |
| `_calc_reliever_discipline_overlay_pitching` | `_inherited_runners`, `_inherited_scored` | — | ±10 | — |
| `_calc_fip_overlay_pitching` | `_obs_fip_delta`, `_obs_fip_bf` | −25× | ±35 | residual −0.36 |
| `_calc_bb9_overlay_pitching` | `_obs_bb9_delta`, `_obs_bb9_bf` | −10× | ±20 | residual −0.34 |
| `_calc_overperformance_overlay_pitching` | `_over_era_plus`, `_over_era_plus_bf` | 0.4× | ±20 | — |
| `_calc_stuff_diminishing_pitching` | stuff, movement, control ≥70-85 | — | −30 | residual −0.52 (stuff) |
| `_calc_stamina_bonus_pitching` | stamina, SP-only | 0.6× | +30 | residual +0.38 |
| `_calc_stuff_overcredit_damping_pitching` | stuff>60 | −0.25× | −12 | residual −0.52 |

---

## 7. Residuals After All Overlays (Open Signal)

Cross-league pooled, WAR/600 for batting, WAR/200 for pitching.

### Batting residuals (what's still unexplained)

| Feature | r with residuals | Interpretation |
|---|---|---|
| `power` | **−0.29** | High-power cards still over-predicted (diminishing penalty too weak) |
| `tier` | **−0.27** | High-tier cards over-predicted generally |
| `obp` | +0.25 | OBP overlay could push harder |
| `babip` | +0.26 | **No BABIP overlay yet — gap here** |
| `ops_plus` | +0.25 | OPS+ overlay could push harder |
| `gap_power` | −0.20 | Same family as power diminishing |
| `contact` | −0.15 | Over-predicted (weaker than power) |
| `slg` | +0.17 | (subsumed by OPS+) |
| `speed` | +0.15 | Under-weighted in core formula |
| `baserunning` | +0.10 | Under-weighted |

### Pitching residuals

| Feature | r with residuals | Interpretation |
|---|---|---|
| `stuff` | **−0.52** | Stuff damping/diminishing still under-sized |
| `stamina` | **+0.38** | Stamina bonus magnitude still under-sized |
| `hr_per_9` | −0.35 | FIP partially captures; still gap |
| `bb_per_9` | −0.34 | **No standalone BB/9 overlay yet** |
| `fip` | −0.28 | FIP overlay could push harder |
| `k_per_9` | −0.27 | Curious negative — high-K pitchers don't WAR-deliver |
| `whip` | −0.20 | Somewhat in FIP |
| `era_plus` | +0.14 | Observed quality signal |

---

## 8. Key Insights & Surprises

### What the data confirmed
- **OPS+ > any rating** for predicting WAR once you have ≥150 PA sample
- **FIP is the single most additive pitching overlay** (captures HR suppression SIERA undershoots)
- **League environment matters** — lb124 avg OPS+ = 100 is NOT the same card quality as i76 avg OPS+ = 100
- **Position-conditional correlations are sharp**:
  - CF:contact r=+0.58 (lead-off archetype)
  - C:power r=+0.49 (rare power Cs are elite)
  - SS:avoid_ks r=−0.27 (high-K SSs are surprisingly fine)
- **Multiplicative interactions dominate**: contact × OPS+ r=+0.79, but additive formula never captures this

### What was surprising
- `avoid_ks` weight zeroed by NNLS calibration — contact already subsumes it
- `arsenal diversity` has r≈0 with WAR (we kept scaffolding but zeroed weight)
- `position flexibility` has r≈0 with WAR (roster utility ≠ WAR)
- OOTP .1 IP notation means 1 out, not 0.1 innings — silent bug source
- Same card_id can appear on 2 different PT teams — pool or conflate intentionally

### Persistent mysteries
- **`stuff` residual −0.52 after aggressive damping** — either the rating is noisy or there's a hidden opposing signal (stuff is paired with some hidden negative)
- **High-K pitchers under-deliver WAR** (k_per_9 residual −0.27) — suggests high-K pitchers in this dataset are also high-walk (three-true-outcomes)
- **Tier over-prediction across the board** (r=−0.27) — we credit tier implicitly via ratings, but actual WAR doesn't scale tier-linearly at the top end

---

## 9. Known Gaps & Research Questions

**Q1: Is `stuff` noisy, redundant, or something else?**
After linear damping (-0.10/pt) and dual-elite penalty (stuff×movement), residual is still −0.52. Testing: fit a GAM or spline on stuff alone. If the curve is non-monotonic, we've missed a turning point.

**Q2: Why does k_per_9 have negative residual?**
Univariate k_per_9 → WAR is positive. But after the meta, the residual is negative, meaning high-K pitchers we predict to be elite under-deliver. Hypothesis: high-K comes with high-BB in this dataset. Partial correlation test needed.

**Q3: Position-specific weights — how much would they help?**
Current meta uses same weights for C / CF / SS. Position-conditional correlations show big spread (CF:contact +0.58, C:contact +0.31). Training per-position weights on n=150-250 samples may overfit. Need cross-validation design.

**Q4: Park factor integration**
`_park_factor_bat` fires in wOBA overlay only. OPS+ and FIP overlays rely on `league_id` baselines but don't re-park-adjust. For players on extreme parks, their overlays may still be noisy.

**Q5: Card type × league interaction**
`BATTING_CARD_TYPE_OFFSET` was calibrated pooled. Some card types have n<30 per league — per-league offsets may differ but sample can't support fitting them.

**Q6: Clutch durability**
`game_clutch_events` gives us 4,680 events. Does clutch performance stabilize per-card? PA-per-player is ~50 which is thin. Longitudinal test: does a player's clutch rate in months 1-3 predict months 4-6?

---

## 10. Formula Code Map (for research agents)

| File | Purpose |
|---|---|
| `app/core/meta_scoring.py` | All `calc_*_meta` functions + overlays |
| `app/core/meta_calibration.py` | Ridge → NNLS → Bayesian weight fitting |
| `app/core/meta_validation.py` | Cross-league r, XGBoost ceiling test |
| `app/core/ingestion.py` | `recalculate_all_meta_scores` — where overlays are injected |
| `app/core/html_ingest.py` | Box score + game log parsing |
| `app/core/card_aggregation.py` | Cross-team card pooling, confidence |
| `app/core/superstats.py` | EV/LD%/opponent-quality splits |
| `app/utils/constants.py` | `DEFAULT_BATTING_WEIGHTS`, positional bonuses, card-type offsets |
| `scripts/comprehensive_correlation_scan.py` | Univariate feature sweep |
| `scripts/meta_correlation_scan.py` | Non-linear, interactions, residual-of-residual |

---

## 11. For a Research Agent — Suggested Attacks

Given the DB and this document, these are the highest-value unexplored angles:

1. **Bayesian hierarchical model** — per-position, per-card-type random effects. Current flat regression pools everyone.
2. **GAM/spline on stuff** — straight linear + elite-interaction terms don't fit. Non-monotonic shape suspected.
3. **Partial correlation matrix** — `k_per_9` residual is negative but univariate positive. Partial r (controlling for BB/9, HR/9) may explain it.
4. **Gradient-boosted interactions** — XGBoost already tested, gave 0.14 CV R² vs linear 0.22. But SHAP values could reveal the specific interactions that matter.
5. **Survival-style metric for roster slots** — WAR/600 normalizes, but some positions have shallower supply (C, SS). A "replacement-level adjusted" metric may separate elite-scarce from elite-abundant.
6. **Temporal decay / hot-cold** — we have `game_batting` at per-game resolution. Does a 7-game hot streak predict next-7? If yes, that's a deployable live-roster signal.

---

## 12. Data Quality Notes

- **roster.card_id**: previously NULL for 114 batters; fixed 2026-04-17. Historical snapshots may have gaps.
- **Park factors**: single snapshot per league, not time-series.
- **game_clutch_events deduplication**: early ingest triple-counted events in BATTING/BASERUNNING/FIELDING markers. Fixed by container-id dedup in `html_ingest.py`.
- **Card conflation**: same card_id on multiple PT teams is intentional (cross-team pooling). For per-team stats, always filter by team_id.
- **.1 IP parsing**: OOTP uses .1=1out, .2=2outs — decimal IP is buggy. Use `ip_whole*3 + ip_outs` in SQL.

---

*Generated by the meta analysis engine on 2026-04-17. Update this document
whenever new overlays ship, coefficients change, or new leagues are added.*
