# OOTP PT Optimizer — Dev Team Handoff & Roadmap

**Document date:** 2026-04-24
**Repo:** https://github.com/camchojnacki-spec/ootp-pt-optimizer
**Active branch:** `main`
**Point of contact:** Cameron Chojnacki (project owner, operates the app daily)

This document is the authoritative picture of where the project is today and what a handoff dev team should do next. It pairs with three deep-dive references — read those before touching code:

1. [`ARCHITECTURE.md`](../ARCHITECTURE.md) — technical architecture v5.0 (data pipeline, schema, formulas, layers)
2. [`docs/META_KNOWLEDGE.md`](META_KNOWLEDGE.md) — meta research hand-off, overlay coefficients, open questions
3. [`docs/META_CALCULATOR_OVERVIEW.md`](META_CALCULATOR_OVERVIEW.md) — the meta formula line by line

---

## 1. Project in One Paragraph

Streamlit + SQLite local app that ingests ~30 CSV exports (and HTML box scores) from OOTP 27 Perfect Team, computes a multi-factor "meta score" per card, empirically calibrates that score against observed WAR, and generates buy / sell / flip / roster-replacement recommendations. Core thesis: OVR captures ~50% of WAR variance but is a black box. The meta must **independently beat OVR** at predicting WAR — that's the only scoreboard that matters.

---

## 2. Current State (2026-04-24)

### Data scale (live DB)

| Table | Rows | Notes |
|---|---|---|
| `cards` | 2,756 | Market catalog, 98 columns of ratings + metadata |
| `batting_stats` / `pitching_stats` | 6.5k / 5.4k | Pooled lb124 + i76 |
| `batting_stats_adv` / `pitching_stats_adv` | 6.7k / 5.5k | wOBA, SIERA |
| `game_log_at_bats` | **31,729** | Pitch-by-pitch, **100% card_id resolved** |
| `game_batting` / `game_pitching` | 8.3k / 3.3k | 99%+ card_id resolved |
| `game_clutch_events` | 9,201 | 2-out RBI, LOB-RISP-2out, inherited-scored, HR, errors |
| `games` | 417 | Box scores parsed from HTML |
| `card_archetypes` | 2,756 | k-means clusters — 8 batting / 6 SP / 6 RP |
| `player_history` | 2,548 | Per-refresh snapshots, enables temporal analysis |
| `roster` / `league_rosters` | 6.9k / 2.4k | Active + bench + full-league ownership |

### Calibration performance (2026-04-20 run)

| Type | CV R² | Pearson r | n |
|---|---|---|---|
| **Batting** | 0.180 | **+0.545** | 385 |
| **Pitching SP** | 0.148 | **+0.541** | 222 |
| Pitching (combined) | 0.055 | +0.433 | 405 |
| pos:CF | 0.458 | **+0.677** | 126 |
| pos:2B | 0.414 | +0.643 | 116 |
| pos:LF | 0.372 | +0.610 | 87 |
| pos:RF | 0.274 | +0.524 | 101 |

**Meta vs OVR head-to-head (post-overlays, pooled lb124+i76):**
- Batting: meta r=+0.743 vs OVR r=+0.540 → **meta beats OVR by +0.203**
- Pitching: meta r=+0.562-0.598 vs OVR r=+0.478-0.547 → **+0.05-0.08**

This is the main KPI. If a change drops meta below OVR at predicting WAR, revert it.

---

## 3. What's Shipped Since the ARCHITECTURE.md v5.0 Snapshot

ARCHITECTURE.md was written 2026-04-16. Since then:

### Correctness fixes
- **DB audit Phase 0-5** (2026-04-18): roster de-bloat (-1,665 rows, UNIQUE index), `league_rosters` populated from `_default.csv`, `team_aliases` bridges short↔full team names, `name_resolver.py` raised game-log card_id coverage from **0.4% → 99%+**, 0B orphan `.db` files archived, background worker mtime-throttled.
- **Outlook logic rewrite** (2026-04-17): size-aware blend + WAR-on-track override eliminated Bellinger/Pierre false-cold flags.
- **Multi-position eligibility** (2026-04-20): `position_eligibility.py` shared helper; Van Haltren now surfaces at LF/CF/RF with defensive-fit penalty.

### New signal
- **Observed-stat overlays** (OPS+, OBP, ISO for batting; FIP for pitching) — biggest single accuracy gain (+0.18 r batting).
- **Superstat overlay** — game-log EV, LD%, K%, BB% feed into meta once a card has ≥50 AB / ≥75 BF.
- **ISO-gap overlay** (2026-04-20) — observed ISO vs power-rating prediction (r=+0.466 with WAR residual).
- **RP pair-interaction overlay** (2026-04-20) — `(control-70)*(p_hr-70)` "command fireman" + `(movement-70)*(control-70)` "junk-ball precision". RP r +0.053 — biggest single pitching win.

### New analytics layer
- **`app/core/derived_stats.py`** (1,412 lines, 8 tables): `batter_vs_pitcher_hand`, `pitcher_fatigue`, `clutch_profile_card`, `park_adjusted_stats`, `regression_candidates_v2`, `opponent_scouting`, `price_velocity`, `meta_confidence`. Rebuilt by BG worker after every ingest.
- **`card_archetypes`** (k-means on z-scored ratings, 2,756 cards): each card gets `archetype_name`, `fit_score` (0-100), `archetype_war` (cluster-average WAR), `mix_score`, `min_top3`, `count_elite`.

### New pages
- `0_Data_Refresh` — watch-folder scan with streaming progress + per-file results
- `16_Stats_Reconciliation` — stats-vs-ratings audit
- `17_Regression_Candidates` — BABIP/LD%/EV regression candidates v2
- `18_Sim_Correlations` — rating × outcome heatmap + empirical fit rankings (diagnostic only)
- `19_Archetypes` — archetype leaderboard + Find Similar under PP cap

### Roster Optimizer (4,530 lines — the monolith)
- Confidence / Status / Owned Promotion / Market Upgrade columns with rich tooltip HTML.
- Hot/cold outlook + regression arrows folded into Status cell.
- AI Manager's Eye + Council Review (Gemini/Claude, DB-cached per roster hash — no auto-fire).
- **Fit column + Mix Analysis expander** (2026-04-24): bargain replacements ranked by empirical attribute-mix fit when the meta-delta gate returns "Optimal". Respects sidebar **Max PP per card** cap.

---

## 4. Known Issues & Follow-Ups

Prioritized for a dev team to pick up.

### P0 — Known-broken, blocks trust

Nothing currently P0. The 2026-04-18 DB audit closed the resolver gap that was P0 for weeks.

### P1 — High leverage

**1. Fold fit-score into the Owned/Market upgrade gate itself (not just the Mix Analysis expander).**
Today the "Optimal" verdict fires when `market_delta_meta < 10`. A market card with `Δfit ≥ 15` but small meta delta should be promoted to the `Market Upgrade` column itself. Files: [`app/pages/4_Roster_Optimizer.py:1506` (`find_market_upgrades`)](../app/pages/4_Roster_Optimizer.py), [`app/pages/4_Roster_Optimizer.py:2833` (market-action rendering)](../app/pages/4_Roster_Optimizer.py).

**2. Fielding stats integration (fielding_stats table exists, 684 rows, not used in meta).**
OOTP exports DRS/UZR-equivalents per card. The meta currently uses rating-based defense, which has r=+0.046 with WAR (noise). Substituting/blending observed fielding data could move the defense term from decorative to meaningful. Target file: `app/core/meta_scoring.py` (add `calc_observed_defense_score()`).

**3. Park factor integration.**
`league_team_stats` carries `pf_hr`, `pf_avg`, `pf_avg_l/r`, `pf_hr_l/r` per team-season. Currently ingested, never read by meta. A power hitter in Coors should have ISO discounted; a pitcher at Great American should have HR/9 adjusted. Low-risk change — the overlay would apply where a card has `park_adjusted_stats` data.

**4. Cross-league weight generalization study.**
Weights were fit pooled lb124 + i76 but the residual analysis (2026-04-17) showed opposite-sign residuals between leagues: `i76 +0.33 vs lb124 -0.33`. Meta scale is not cross-league consistent. Either (a) fit per league and blend, (b) add a tier-calibration term, or (c) accept league-specific weights entirely.

**5. Refactor: Roster Optimizer monolith (4,530 lines).**
Split into: `_roster_data.py` (load + precompute), `_upgrade_plan.py` (find_owned / find_market / _build_slot), `_row_render.py` (Fit/Confidence/Status/Upgrade column builders), `_expanders.py` (perf_outlook, mix_analysis, roster_moves), and a thin page file that orchestrates. Risk: regression on a hot path the user touches daily — cover with page-render smoke tests first.

### P2 — Polish

**6. Sell rec suppression for benched-but-eligible cards.**
`_generate_sell_recs` still flags Van Haltren-type cards as "Sell — not on active roster" even though `position_eligibility` now recognizes them as legitimate LF/RF bench options. Downgrade or suppress when a card has a matching eligibility upgrade hit.

**7. Hot-player market-buy suppression.**
Currently a Hot Palmeiro (+78 over meta) still generates "Buy Sam Leslie 1,271 PP" recs. Suppress market-buys when current player is Hot AND the delta is small (<80 meta).

**8. Stamina over-credit for SPs.**
`_calc_stamina_bonus_pitching` adds up to +15 meta for SPs, but stamina shows r≈0 with per-BF outcomes. The bonus is predicting volume (innings), not quality. Either reframe the bonus as volume-adjusted (multiply by `expected_IP/200`) or scale it down. Confirmed via `tools/mine_correlations.py`.

**9. Archetype WAR average is sparse for small clusters.**
Some archetypes have n<20 and their `archetype_war` is noisy. Add a confidence weight or enforce a minimum cluster size in `build_card_archetypes`.

**10. Streamlit deprecation cleanup.**
`use_container_width` is deprecated after 2025-12-31 in favor of `width='stretch'` / `width='content'`. Logs are noisy. Sweep-replace across the codebase.

**11. i76 game-log coverage gap.**
i76 has no `_default.csv` export, so `league_rosters` + `game_batting` will stay sparse for i76 until Cameron plays + exports games there. Not a bug — a data-availability note.

---

## 5. Recommended Next-Quarter Roadmap

Ordered by expected value × feasibility.

### Epic A: Close the Fit / Meta integration loop
**Goal:** The Roster Optimizer's main `Market Upgrade` column surfaces mix-based upgrades, not just meta-based. User already sees this in the Mix Analysis expander — it needs to be promoted to the primary view.
**Scope:** P1 #1 above. Also update `find_market_upgrades` to `ORDER BY` on `max(meta_delta, fit_delta)` rather than meta alone.
**Exit criteria:** When running on Cameron's current staff, the `Market Upgrade` column for SPs 2-5 shows cards like Tom Hughes (129 PP, fit 90) instead of `✅ Optimal`.
**Estimate:** 2-3 days.

### Epic B: Observed defense integration
**Goal:** Replace / blend rating-based defense with DRS-equivalent from `fielding_stats`.
**Scope:** P1 #2. New function `calc_observed_defense_score()` with sample-size gating. Weight empirically against WAR residual after current meta.
**Exit criteria:** Batting meta r vs WAR improves by ≥+0.02 in both leagues, without regressing OVR beat.
**Estimate:** 5-7 days (includes correlation mining + weight calibration).

### Epic C: Park-factor overlay
**Goal:** Discount power stats at hitter-friendly parks; discount pitching ERA at pitcher-friendly parks.
**Scope:** P1 #3. Pull `pf_hr`, `pf_avg` from `league_team_stats` JOIN'd on the card's team. Apply in the observed-stat overlays (ISO gets multiplied by 1/pf, not raw).
**Exit criteria:** Cards from extreme parks (pf_hr > 1.15 or < 0.85) see meaningful meta shifts. Cross-park i76 test shows tighter fit.
**Estimate:** 3-5 days.

### Epic D: XGBoost ceiling study
**Goal:** Establish an upper bound on predictability with a non-linear model. If linear meta r=+0.74 and XGBoost r=+0.80, we have ~0.06 headroom worth chasing. If XGBoost is +0.75, we're near the ceiling and can stop adding linear overlays.
**Scope:** `scripts/xgboost_ceiling.py` exists as a stub — wire it up. Train on full feature set including interactions, report CV r. Inspect feature importance for non-obvious signals.
**Exit criteria:** A single report doc with (a) XGBoost CV r, (b) top-10 importances, (c) recommendation on whether to add any feature to the linear meta.
**Estimate:** 3-4 days.

### Epic E: Temporal dynamics study
**Goal:** `player_history` has 2,548 snapshots. Are early-season wOBA leaders sustained? Do BABIP-regression v2 predictions verify? Does meta-score drift track price-velocity?
**Scope:** Analysis-only, no meta changes. Produces a `docs/TEMPORAL_STUDY.md`.
**Exit criteria:** Evidence-based answers to three questions: (1) how quickly does wOBA stabilize, (2) do regression candidates converge as predicted, (3) which meta changes predict price changes.
**Estimate:** 5-7 days.

### Epic F: Roster Optimizer refactor
**Goal:** Split the 4,530-line monolith into testable units.
**Scope:** P1 #5. New structure + smoke tests.
**Exit criteria:** No file > 1,000 lines. Pages render identically pre/post. Cameron can't tell it changed.
**Estimate:** 7-10 days. **Prerequisite:** Page-render smoke tests covering the batting + pitching tabs with a fixture roster. Without those, this epic is high-risk.

### Epic G: Cross-league calibration
**Goal:** Resolve the opposite-sign residuals between lb124 and i76.
**Scope:** P1 #4. Either per-league fits blended at query time, or a tier-calibration overlay. Evaluate both.
**Exit criteria:** i76 residual shrinks from ±0.33 to <±0.10 without hurting lb124.
**Estimate:** 7-10 days.

### Epic H: Streamlit deprecation cleanup
**Goal:** Clean log output.
**Scope:** P2 #10.
**Estimate:** 1 day of sweep-replace.

---

## 6. Guardrails — Things Not to Break

Every item below has cost an entire session to get right. Any PR that regresses one of these should be rejected.

**1. OVR is a diagnostic column, not a formula term.**
The meta's job is to beat OVR. If OVR goes back into the formula, the system starts rationalizing rather than predicting. Ref: [`memory/feedback_meta_architecture.md`](../../.claude/projects/C--Users-Cameron-OneDrive-Documents-Claude-Projects-OOTPBUYNSELL/memory/feedback_meta_architecture.md) — the session where this was re-learned the hard way.

**2. Every market-filtering page reads the sidebar `Max PP per card` input.**
Defaults from `config.yaml['pp_budget']`. Never hardcode a ceiling. Pattern: `budget = config.get('pp_budget', 500); max_spend = st.number_input("Max PP per card", ..., value=budget)`. Ref: `memory/feedback_sidebar_budget_filter.md`.

**3. Attribute-mix framing.**
Cards are evaluated by combinations that drive outcomes in *this* sim, not by single-rating lookups or a monolithic meta. Replacement finders should use `card_archetypes.fit_score` + `archetype_war` as primary signals alongside meta. Ref: `memory/feedback_attribute_mix_framing.md`.

**4. Name resolver.**
`name_resolver.py` + `team_aliases` + `league_rosters` together achieve 99%+ game-log card_id resolution. If any of those three tables/modules gets "cleaned up" independently, coverage drops to near zero. Don't touch without reading `memory/project_db_audit_2026_04_18.md` first.

**5. Meta weight load precedence.**
`DB calibration → config.yaml → DEFAULT_*_WEIGHTS`. If a change bypasses calibrated weights, the whole empirical loop breaks. Calibration has safety rails (dual-gate on CV R² or Pearson r ≥ 0.30, ±25% sanity clip vs prior, 40% confidence cap in Bayesian blend) — don't disable those.

**6. Balance penalties + diminishing returns exist for reasons.**
A card with one 95 and two 60s is worse than three 80s in this sim (see `project_attribute_mix_2026_04_20.md`: min-of-top-3 predicts WAR at r=+0.452 vs max at +0.244). The `_diminished()` function + balance floors capture this. Don't remove them as "redundant" — they're not.

**7. Roster dedup + UNIQUE index.**
`idx_roster_dedup` prevents the 5.8× bloat that existed before Phase 0-5. Any new `INSERT INTO roster` must use `INSERT OR IGNORE` and respect the `(card_id, DATE(snapshot_date), lineup_role)` uniqueness.

**8. Background worker mtime throttle.**
`background_worker.py` filters `actionable` by `last_import is None or modified > last_import`. Without this gate the worker re-ingested unchanged files 80-280× per file per day and the DB was constantly locked.

---

## 7. Files to Read, in Order, on Day 1

1. **This file** — you're here.
2. **`ARCHITECTURE.md`** (45 min read) — system architecture, data pipeline, schema.
3. **`docs/META_KNOWLEDGE.md`** (30 min) — meta research state, overlay coefficients, open questions.
4. **`docs/META_CALCULATOR_OVERVIEW.md`** (30 min) — formula line by line.
5. **`app/core/meta_scoring.py`** (2,449 lines — skim `calc_batting_meta` + `calc_pitching_meta` + the `_calc_*_overlay_*` functions).
6. **`app/core/ingestion.py`** (3,068 lines — skim the `_get_active_league` gate + the file-type dispatch in `_refresh_file()`).
7. **`app/core/derived_stats.py`** (1,412 lines — 8 analytical tables, each built by its own function. Read `build_card_archetypes` closely since the fit layer depends on it).
8. **`app/core/name_resolver.py`** (bridges game-log names to `card_id`).
9. **`app/pages/4_Roster_Optimizer.py`** (4,530 lines — don't try to read linearly. Use Ctrl+F for `_build_slot`, `find_market_upgrades`, `_render_mix_analysis_expander`).
10. **`tools/mine_correlations.py`** — if you ever want to re-verify what the meta formula *should* be emphasizing.

---

## 8. Local Dev Setup

```bash
# Prereqs: Python 3.12+, Windows (paths are Windows-rooted)
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app/main.py --server.port=8502

# Watch directory (configure in config.yaml):
# C:\Users\Cameron\OneDrive\Documents\Out of the Park Developments\OOTP Baseball 27\online_data

# Re-run correlation mining against current DB:
python tools/mine_correlations.py

# DB lives at: data/ootp_optimizer.db (~50MB after full ingest)
```

`config.yaml` is gitignored (contains API keys for Gemini + Claude). Create from the keys Cameron shares.

---

## 9. Questions to Validate With Cameron Before Building

1. **Scope of Epic A:** should `find_market_upgrades` return cards qualified by *either* meta delta *or* fit delta, or should there be two lists (one per signal)?
2. **Fielding priority:** for Epic B, is the target "defense signal that matters in meta" or "defense signal that matters in lineup construction advice"? Different work streams.
3. **Park-factor leagues:** lb124 parks are mostly ~1.0. Is there an upcoming league with real park effects to validate against?
4. **Council Review:** Cameron currently triggers it manually. Is on-ingest auto-fire desired, or does the cost cap ($ per Gemini call) make the gate worth keeping?
5. **Multi-league UI:** i76 support is half-wired. Should the UI get a league switcher or stay lb124-first?

---

*End of handoff plan. Ping Cameron with questions — he operates the app daily and will spot any unrealistic assumption in about ten seconds.*
