# Epic E — Temporal Dynamics Study

**Generated**: 2026-04-24 against `data/ootp_optimizer.db` (player_history n=2,949; batting_stats × batting_stats_adv joined snapshots = 6,073 rows; 553 cards with ≥3 wOBA snapshots).

## TL;DR

- **wOBA does not stabilize** to r ≥ 0.80 against final-snapshot wOBA at any PA bucket within the observed window. Best-case r is ~0.69–0.75 in the 100–149 PA range, and the relationship actually weakens past 200 PA because "final" wOBA is itself a moving mid-season target. The honest answer: in this dataset, the snapshot history is too short (9 days, 7 distinct dates) to claim wOBA reliably converges.
- **Regression flags work asymmetrically**: `regress_down` predictions hit 65.6% (vs 37.5% baseline) — a real, useful signal. `regress_up` predictions hit only 51.3% (vs 32.6% baseline) — directionally right but barely better than chance per-pair, and the median move is just +1 OPS+. `sustainable` flags are stickier than expected (74% stayed within ±10 OPS+).
- **Meta-score drift does NOT predict price velocity**, in either direction. Coincident r = +0.002, meta-leads-price r = −0.007, price-leads-meta r = −0.014 across 2,228+ delta pairs. Even filtering for big meta moves (|ΔM|>5), correlation stays ≈0. Price and meta drift independently in this snapshot regime.

## Q1. wOBA Stabilization

**Method**: 553 cards have ≥3 joined batting_stats / batting_stats_adv snapshots. For each, the "final" wOBA is the most recent snapshot (2026-04-24). I bucketed each prior snapshot by its PA at that moment and computed cross-card Pearson r between snapshot wOBA and that card's final wOBA.

**All cards with ≥3 snapshots (n=553):**

| PA bucket | n pairs | r(wOBA_at_PA, wOBA_final) |
|-----------|--------:|--------------------------:|
| 0–49      |     622 | +0.393 |
| 50–99     |     579 | +0.563 |
| 100–149   |     469 | **+0.752** |
| 150–199   |     356 | +0.578 |
| 200–299   |   1,056 | +0.595 |
| 300–399   |     633 | +0.539 |
| 400–499   |     739 | +0.602 |
| 500+      |   1,211 | +0.428 |

**Full-season cohort, final PA ≥ 500 (n=223):** the 100–149 PA peak persists at r=+0.445, climbs to +0.628 by 200–299 PA, and never reaches 0.80.

**Why r doesn't reach 0.80**: the snapshot window is 2026-04-15 to 2026-04-24 — 9 days, 7 distinct snapshot dates. The "final" wOBA is mid-season, not season-end. In the final-PA-≥300 cohort (n=346), the median *last-3-snapshot* wOBA range is 0.013, and only 44% of cards had ≤0.010 wOBA range across their last 3 snapshots. The denominator (final wOBA) is itself volatile, so per-snap r is bounded.

**Stickiness interpretation**: Pearson r(wOBA_t, wOBA_{t+1}) peaks at 0.69 around 100–149 PA and falls back to ≈0.40 at 500+ PA. This isn't a stabilization story — it's a re-shuffling story driven by overlapping samples and short windows.

**Recommendation**: 
- Until snapshots span an entire season (or two), report wOBA bands as a "current wOBA, ±0.020 wOBA noise floor" rather than treating them as stable predictors.
- For the meta engine, do not weight wOBA as a high-confidence signal until card PA exceeds ~150 — below that, r-to-final stays under 0.55.
- Implement a `pa_band` column on observed-stat overlays (the existing `superstats` overlay would benefit) and gate signal weights by it.

## Q2. Regression-Candidate Convergence

**Method**: `regression_candidates_v2` only stores the latest snapshot, so I retroactively re-applied `derived_stats.build_regression_candidates_v2` logic to every historical batting_stats row (PA ≥ 50, non-pitchers), using the **same** card-value → expected OPS+ curve fit on the latest snapshot. For each consecutive-snapshot pair, I scored snapshot N and checked whether snapshot N+1's OPS+ moved in the predicted direction.

**Sample size**: 1,923 consecutive-snapshot pairs across the 553 eligible cards.

| Direction       | Count | Hit rate | Mean ΔOPS+ | Median ΔOPS+ |
|-----------------|------:|---------:|-----------:|-------------:|
| `regress_down`  |   224 | **65.6%** | −20.41 | −11.00 |
| `regress_up`    |   595 |   51.3% |  +8.92 |  +1.00 |
| `sustainable`   | 1,104 |   74.0% (\|Δ\|≤10) |   — |   — |

**Baseline** (any consecutive pair, ignoring score): P(OPS+ down) = 37.5%, P(OPS+ up) = 32.6%, P(stable ±10) ≈ 30%.

**Findings**:

- **`regress_down` is real signal.** 65.6% hit rate vs 37.5% baseline is a +28-point lift — equivalent to +0.6σ. Median move is −11 OPS+, mean −20. Hot streaks at the time of flagging do come down.
- **`regress_up` is weak.** 51.3% hit rate vs 32.6% baseline is +19 points, but the median move is just +1 OPS+. Most "regression-up" candidates stay roughly where they are; only a minority bounce hard. Likely because cold streaks have a real-talent component that flagging based on rating-vs-outcome can't disambiguate.
- **`sustainable` flags are sticky** (74% within ±10 OPS+ vs ~30% baseline) — useful as a *negative* signal: cards in this band are unlikely to swing.

**Recommendation**:
- Surface `regress_down` flags prominently — they're predictive enough to act on for sell-recommendations.
- Demote `regress_up` to "watch list" tier; do not auto-recommend buys based on it.
- Add a confidence-weighted version: combine PA gate with regression_score to filter the low-PA noise.

## Q3. Meta Drift vs Price Velocity

**Method**: From `player_history`, 290 cards have ≥3 distinct snapshots with both `meta_score` and `last_10_price`. For each card, computed Δmeta and Δprice between consecutive (deduped) snapshots — 2,518 delta pairs total. Tested three correlation regimes:

1. Coincident: Δmeta_t vs Δprice_t (same window)
2. Meta-leads-price: Δmeta_t vs Δprice_{t+1} (1-snap lag, n=2,228)
3. Price-leads-meta: Δprice_t vs Δmeta_{t+1} (1-snap lag, n=2,228)

**Results**:

| Regime              |     n | Pearson r |
|---------------------|------:|----------:|
| Coincident          | 2,518 |   +0.002 |
| Meta leads price    | 2,228 |   −0.007 |
| Price leads meta    | 2,228 |   −0.014 |
| Filtered \|Δmeta\|>5 (coincident) | 1,145 |   +0.000 |
| Filtered \|Δmeta\|>5 (meta-leads) |   950 |   −0.012 |

**Interpretation**: There is no detectable linear relationship between meta-score drift and price drift in either direction. Even after filtering to large meta moves (which should be the most informative), the signal is null.

**Why this is consistent with the project thesis**: 
- The market price reflects perceived value (often anchored to OVR + recent results), while meta_score reflects modeled WAR contribution. The two updates are essentially uncorrelated within the 1–9 day cadence of these snapshots.
- Meta_score updates within this dataset are dominated by **calibration changes** (the engine itself was retuned across the 2026-04-13 to 2026-04-24 window — see commits 0ee88c2, e993e73, 4a619e2). Cards saw their meta jump 100+ points overnight from formula changes, not from new-game evidence. Those engine-driven jumps are independent of the market.
- Real price-vs-meta causality probably operates on a slower clock than 1-snap lag captures.

**Recommendation**:
- Do NOT add a "meta-momentum buy signal" to the recommender. Meta drift is not a leading indicator of price.
- Re-run this study after 2 months of *stable-engine* snapshots (no calibration churn) and after `price_velocity` has dense daily history. Today's null is largely a measurement-period artifact.
- Until then, frame price commentary as orthogonal to meta drift — they answer different questions on different clocks.

## What This Means for the Meta

1. **wOBA reliability is a function of PA, not snapshot count.** Below ~150 PA, observed wOBA is noisy enough to mislead recommendations. The meta engine should down-weight wOBA-derived overlays for cards under that threshold.
2. **`regress_down` flags are the strongest temporal signal in the system.** They beat baseline by +28 points on hit rate and produce a median 11-point OPS+ drop in the next snapshot. Wire them into sell-recommendation logic now.
3. **`regress_up` flags are not actionable on their own.** Combine with another signal (recent xwOBA, contact-quality overlay) before surfacing buys.
4. **Meta-score drift is not a price predictor in this dataset, full stop.** Any "meta is moving, price will follow" UI claim should be removed or deferred until a stable-engine, longer-window study can be run.
5. **Snapshot cadence is the limiting factor.** 7 distinct stat-dates over 9 days is not enough to characterize stabilization or causality. The single most valuable infrastructure investment for temporal analysis would be a **daily** (not ad-hoc) `derived_stats.build_regression_candidates_v2` run that *retains* historical rows so this study can be re-run against richer data in 30, 60, 90 days. Right now the table is overwritten on every rebuild; switching to append-only with `snapshot_date` would unlock real prospective validation.
