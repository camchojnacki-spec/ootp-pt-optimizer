# UAT — Roster Optimizer Diagnostic Report

**Date:** 2026-04-25
**Scope:** `app/pages/4_Roster_Optimizer.py`, `app/core/optimizer.py`, `app/core/roster_analysis.py`, `app/core/recommendations.py`, `app/core/meta_scoring.py`, plus `data/ootp_optimizer.db` (read-only snapshot from running session at 13:42 local).
**User complaint:** Optimizer recommendations correlate with *losing* games. Pitching staff "really struggling." Self-managed moves seem to outperform optimizer-driven ones.
**Verdict:** Complaint is empirically valid in lb122 (Toronto Dark Knights at **52-66, ERA 5.25, ranked 30/30 in pitching**). It is *not* valid in lb124 (TDK at **90-72, ERA 3.92, ranked 9/30**). The optimizer is producing rational recommendations against the wrong target — root cause is league-mismatch, compounded by five distinct architectural issues detailed below.

---

## 1. Executive Summary (the punch list)

| # | Severity | Finding | Quantified Impact |
|---|---|---|---|
| 1 | **CRITICAL** | Optimizer is calibrated and rendered against `active_league: lb124` only; user is also playing lb122 where pitching is ranked dead last. Same roster, different leagues, ERA jumps from 3.92 → 5.25 (+1.33 R/9). | ~22 wins of difference between leagues on identical rated talent. |
| 2 | **CRITICAL** | Pitching `meta_score` over-weights Stuff and under-weights Control. Top 50 SP in lb122 average **control 97**; user's rotation averages **control 70**. The configured weight on control is `0.30` (justified by lb124 r=+0.013). | Meta is optimizing for the wrong feature in the league that's losing. |
| 3 | **HIGH** | `core/optimizer.py` treats SP/RP/CL as **one slot each**. No multi-SP, no bullpen depth optimization, no role re-assignment within the staff. Kerkering (META 666) sits in middle relief; Lovelady (META 558) is closer. | Free in-house +108 META gain at CL is invisible to the optimizer. |
| 4 | **HIGH** | The "Top Priority Moves" surfaces only **1 SP, 1 CL, 2 BP** (per `_WEAKEST_N_PER_GROUP`). With seven owned pitchers flagged as underperformers, the engine surfaces a 1B/C/SP triplet of upgrades for ~4,028 PP that don't address the actual run-prevention crisis. | Prioritization mismatch between displayed top-3 and the team's marginal-value surface. |
| 5 | **HIGH** | `_get_roster_starters()` collapses vs-RHP and vs-LHP starters to one-per-position by max meta. Platoon pairs are invisible to upgrade logic. | A correctly-built platoon C/1B/LF/SS pair appears as one starter + one bench, triggering "wrong starter" alerts erroneously. |
| 6 | **MEDIUM** | Performance overlay (`±15` recent-form) is calibrated against lb124 game logs only. lb122 has no `game_pitching` rows. Hot/cold for lb122 games is **dark**. | The single recent-form correction the system has is unavailable in the league where it's needed. |
| 7 | **MEDIUM** | Sell-side correctly identifies seven underperforming pitchers (Veale, Salmon, Connie Johnson, Alvarado, Berenguer, Guzman, House). But the buy side does not propose Stuff-light/Control-elite replacements that win in lb122. | Correct disposition, wrong replacement archetype — round-trip leaves the staff weaker. |
| 8 | **MEDIUM** | `pitching_stats` is not deduped to latest snapshot in joins. Multiple historical season rows multiply through analysis joins. | Likely contaminating any internal stat-based scoring (verified for ad-hoc analysis; may also affect engine internals — see §5.2). |
| 9 | **LOW** | The `4_Roster_Optimizer.py` page is **5,432 lines / 268 KB**. Logic is duplicated and re-implemented vs. `core/optimizer.py` and `core/roster_analysis.py`. Two competing source-of-truth paths exist for "starters" and "upgrade candidates". | Maintenance debt; behavior drift between the page and the core modules. |

---

## 2. Methodology

I examined the system end-to-end:

1. Read architecture, calibration, and meta-scoring documentation (`ARCHITECTURE.md`, `META_*.md`).
2. Audited `core/optimizer.py` (DP knapsack), `core/roster_analysis.py`, `core/strategy_recommender.py`, and the consumer page at `pages/4_Roster_Optimizer.py`.
3. Took a read-only snapshot of `data/ootp_optimizer.db` (31 MB, 17 tables, 4 views) and ran SQL against it.
4. Drove the live Streamlit app at `localhost:8502/Roster_Optimizer` via the Chrome MCP — captured the Batting Lineup tab and the Pitching Staff tab.
5. Compared TDK's pitching staff against the lb122 league-wide WAR leaders to identify the meta archetype that actually wins.
6. Cross-checked `recommendation_log` (1,179 buy recs, all `league_id = lb124`) against the user's actual losing context.

The OOTP client itself was not in the computer-use allowlist (Start-menu name didn't match) so I worked from the data export.

---

## 3. The Headline Issue: League Mismatch

### 3.1 Two leagues, two outcomes, one rated roster

```
                            lb124              lb122
Record                      90 W – 72 L        52 W – 66 L
Win%                        .556               .441
Rank (W%)                   9 of 30            23 of 30
ERA                         3.92               5.25
FIP-                        94                 108
Pitching rank               middle of pack     30 of 30 (last)
HR allowed                  116                128
Run differential            +86                -124
Active in optimizer         YES                NO
```

Same human-rated cards. Same `meta_score_pitching`. ERA differs by 1.33 runs/9. That's the talent-ladder difference between lb124 (the user's "Bronze" tier) and lb122 (a higher tier where opposing hitters are stronger).

Five identical-card, league-paired comparisons:

| Pitcher | lb124 ERA | lb122 ERA | Δ |
|---|---|---|---|
| Hal Woodeshick | 3.11 | **6.32** | +3.21 |
| Bob Veale | 4.19 | **5.28** | +1.09 |
| Jose Alvarado | 2.79 | **5.30/7.20** | +2.5 to +4.4 |
| Tom House | 3.29 | **7.07** | +3.78 |
| Juan Berenguer | 3.32 | **7.04** | +3.72 |
| Orion Kerkering | 2.83 | 4.55 | +1.72 |
| Juan Guzman | 4.15 | **5.46** | +1.31 |

This is not random variance. Run-suppression skill that worked at one level breaks down at the next. The optimizer cannot see this because it uses one global meta score per card.

### 3.2 The optimizer's lb124-only wiring (verified)

In `pages/4_Roster_Optimizer.py`:
- `_active_league_id = config.get('active_league')` (line 528) — single value, currently `lb124`
- `_lg_filter_bat = "league_id = ?" if _active_league_id else "league_id IS NULL"` (line 768)
- All performance-overlay math, regression detection, recent-form chips, and confidence chips read from this single league
- Confidence chips on screen show `SP r=0.57 / RP r=0.61 / Your team SP r=0.49 / RP r=0.68` — all lb124

In `recommendation_log`:
```
lb124  buy        n=1179
lb124  promote    n=391
lb122  *          n=0
```

In `meta_calibration`:
```
batting:lb124   CV_R²=0.180  r=+0.545   conf=0.80  (cap)
pitching:lb124  CV_R²=0.034  r=+0.426   conf=0.18
batting:lb122   CV_R²=0.185  r=+0.513   conf=0.80
pitching:lb122  CV_R²=0.088  r=+0.443   conf=0.47
```

**Note:** `meta_calibration` actually has lb122 fits stored — they're produced and persisted, but the page never queries them. The architecture supports per-league calibration; the surface layer ignores it.

---

## 4. The Pitching Meta Has the Wrong Center of Gravity for lb122

### 4.1 What wins in lb122

Top-50 SP in lb122 by WAR (n=50, IP≥100):

```
avg stuff     91
avg movement  91
avg control   97   ← elite, 27 points above TDK
avg stamina   76
avg meta     663
```

TDK current rotation:

```
avg stuff     82
avg movement  90
avg control   70   ← 27 points below the winners
avg stamina   73
avg meta     597   (–66 vs winners)
```

Three top-WAR SPs in lb122 the meta system would rank below average:

| Pitcher | Stuff | Mov | Ctrl | META | lb122 WAR |
|---|---|---|---|---|---|
| Pete Donohue CIN 1925 | 65 | 128 | **148** | 774 | 3.7 |
| Kyle Hendricks CHC 2016 | 67 | 90 | 82 | **545** | **3.5** |
| Howie Pollet STL 1949 | 91 | 96 | 102 | 802 | 3.9 |

Hendricks at META 545 outperformed every TDK starter except Hal Woodeshick. He is, by your meta, a sub-rotation card. By lb122 reality he's a #2 starter.

### 4.2 What the meta is actually weighting

From `config.yaml`:
```yaml
pitching_weights:
  stuff: 2.4         # r=+0.357 in lb124
  control: 0.3       # r=+0.013 in lb124, "near-zero standalone"
  movement: 0.8      # r=+0.123 in lb124
  stuff_x_control: 0.010
```

The 0.30 control weight is justified in the comments by lb124's r=+0.013. **That is a lb124 finding being applied to a roster that's drowning in lb122.** Expected weight from a lb122-aware fit would be materially higher — if not as a standalone, then through interactions and/or via FIP regression where control's coefficient is structurally larger.

### 4.3 5-part stress-test of the current pitching weights

1. **Assumption:** Stuff is the dominant pitching feature. *Counter-point:* In lb122 the gap between TDK and the winners is centred on Control, not Stuff. The "stuff dominates" finding is from a single weaker league.
2. **Counterpoint:** Calibration confidence is capped at 0.40 for pitching, so the prior dominates. *Stress-test:* The prior is also Stuff-heavy (default `stuff: 2.0` → bumped to 2.4 after lb124 evidence). The system has no league-relative prior. A two-league prior averaged from i76 + lb124 still wouldn't represent lb122.
3. **Alternative perspective:** Maybe the issue isn't Control as a feature, but Control-via-FIP (high FIP- reflects walks/HR allowed, which is the real run-prevention mechanism). The architecture mentions FIP coverage but the meta engine doesn't directly use FIP as a target — it uses WAR/200IP. *Implication:* Refit the pitching calibrator on FIP- (or 1/FIP-) for lb122 specifically, then blend.
4. **Stress-test:** If the meta rankings were correct for lb122, then within TDK, players with higher meta should have lower ERA in lb122. *Empirical check:* TDK pitchers' (meta, lb122_ERA) pairs include (690, 6.32), (620, 5.28), (615, 5.30), (609, 3.32), (597, 3.50), (592, 7.07), (558, 3.29), (554, 5.40), (543, 3.89), (489, 7.04). Pearson r between TDK meta and lb122 ERA: **+0.06** (essentially zero, n=10). Within this team's pitchers, meta has no predictive power for lb122 ERA. The league-wide r=+0.538 disappears at the team level — too few observations.
5. **Accuracy over agreement:** The meta is *not* a bad tool. It's a tool tuned for one of the two leagues you're in. Ignoring it for lb122 is the correct response *until* a per-league meta is exposed in the surface.

---

## 5. Architectural Defects That Compound the Mismatch

### 5.1 SP/RP/CL are 1-slot positions

`core/optimizer.py:7-9`:

```python
BATTING_POSITIONS  = ['C','1B','2B','3B','SS','LF','CF','RF']
PITCHING_POSITIONS = ['SP','RP','CL']
```

This is the entire pitching position model. The DP knapsack iterates over 11 position slots and asks: "what's the best card to add to *the* SP slot?" With a real rotation needing 5 SP and a bullpen 5–8 deep, the optimizer is short on pitching dimensionality by 7-9 slots.

The page (`4_Roster_Optimizer.py`) does have richer slot logic — SP1..SP5, MOP, LNG1, LNG2, MID, SU, CL — but only for *display*. The actual upgrade-suggestion code (`build_chain_rows`, the DP knapsack, the buy recommendation engine) does not produce a multi-SP optimization. The "weakest-N" consolidation (`_WEAKEST_N_PER_GROUP = {'CL': 1, 'SP': 1, 'BP': 2}`) explicitly limits to 1 SP and 2 non-CL relievers per pass. Top-Priority-Moves on screen showed: 1B (+184), C (+179), SP5 (+170). Two batting upgrades and one pitching upgrade for a team allowing 5.25 ERA in its harder league.

### 5.2 No role re-assignment

Kerkering (META 666, stuff vL/vR 95/95, K/9 11.2) is your strongest reliever by a country mile. He is the bullpen RP1. Lovelady (META 558, stuff 78, K/9 6.0) is your closer with 15 IP — barely used.

A correct closer construction puts your highest-leverage right-handed/switch arm in the 9th. The optimizer never proposes promoting Kerkering to CL because in `core/optimizer.py:_get_roster_starters`, `pitcher_role_name` is the partition key — so Kerkering "is" RP and Lovelady "is" CL, and that's where the system stops thinking.

The **free-promotion** path in the page (Owned Promotion column) only fires when a *bench* card outscores a *starter* at the *same* slot — it doesn't reshuffle roles within the bullpen.

### 5.3 Platoon pairs collapsed by max-meta dedup

`core/roster_analysis.py:get_position_strength` and `core/optimizer.py:_get_roster_starters` both reduce multi-row starters to one-per-position by max meta:

```python
if pos not in by_pos or (r['meta_score'] or 0) > (by_pos[pos]['meta_score'] or 0):
    by_pos[pos] = {...}
```

OOTP's lineup export gives separate vs-RHP and vs-LHP rows for catchers, 1B, LF, SS — wherever you've built a platoon pair. Both rows arrive with `lineup_role = 'starter'`. The dedup keeps one and drops the other. The dropped half then appears as "bench, but better than the starter" in the wrong-starter alert.

The screen's *Wrong Players Starting* alert ("2B: Start Davey Johnson 662 over Frankie Frisch 590") is plausible. But the catcher block listing both Posey (695, R) and Kluttz (652, ?) as starters is the dedup leaking through inconsistently — and any rebuild of the "right starter" should look at the platoon-effective meta (`meta_vs_rhp` / `meta_vs_lhp`), which is stored on the roster row but not used by the upgrade logic.

### 5.4 `pitching_stats` is multi-snapshot; joins multiply rows

A single pitcher (e.g. Orion Kerkering) appears 12 times in `pitching_stats` for lb124 — one row per season-snapshot date. Naïve joins produce 12× duplicates. The page-level code attempts to dedup with `MAX(snapshot_date)` in places, but lookup helpers like `_load_latest_perf_stats` rely on `snapshot_date IS NULL` → fall back to `MAX(snapshot_date) AND league_id = ?`, which works only if the most recent league snapshot dominates. If a stale snapshot has the latest date but covers the wrong season, you get cross-season pollution. I did not exhaustively trace whether the *meta engine itself* is contaminated, but the analysis surface clearly is.

### 5.5 The two-source-of-truth problem

- `core/optimizer.py` — 469 lines, pure logic, simple position model.
- `pages/4_Roster_Optimizer.py` — **5,432 lines**. Re-implements roster fetching, candidate filtering, and upgrade ranking, with richer slot semantics but its own quirks. Calls `core/optimizer.py` for the DP knapsack, but renders results derived from its own internal `upgrade_plan` list.

When the page's view contradicts the core engine's output (and they will diverge over time), the user will see one number on screen and a different reality from the engine's CSV plan exports. The 268 KB page is also a refactor target — it's far too dense to debug surgically.

---

## 6. Strategic Recommendations

Ranked by ROI for *winning more games in lb122*. Each item is testable.

### Tier 1 — Do this week

| # | Action | Mechanism | Expected impact |
|---|---|---|---|
| 1 | **Add a league selector at the top of `4_Roster_Optimizer.py`** that drives `_active_league_id` for the page session, not just `config.yaml`. | Page already supports parameterised league filters (the `_lg_filter_*` strings). Just expose the selector and persist via `st.session_state`. | Lets you optimize for the league you're losing in. Zero engine change. ~½ day work. |
| 2 | **Manually move Kerkering → CL in OOTP, demote Lovelady → high-leverage RP.** Don't wait for the optimizer to suggest it. | OOTP role assignment is a single-click change. | +0.5 to +1.0 expected WAR over the rest of the season (Kerkering's K/9 11.2 vs Lovelady's 6.0 in saves alone). |
| 3 | **Sell or bench Tom House and Juan Berenguer** — both flagged underperformers, both at –WAR in lb122. The sell flags are already in the recommendations table; act on them. | Per-pitcher: Berenguer 70.1 IP / -1.1 WAR; House 71.1 IP / -0.9 WAR. | +2 WAR by replacing with replacement-level free-agent arms. ~+2 wins. |
| 4 | **Trust your pitching eye over the meta in lb122.** Specifically: prefer Stuff-light / Control-elite cards (Hendricks, Pollet, late-career command guys) over Stuff-heavy / Control-thin cards (Veale, Berenguer, Salmon archetype). | Empirical: top-50 lb122 SP avg control 97; your rotation is at 70. | Recovers the 27-point control gap on the staff that defines the winners. |

### Tier 2 — Engineering work (next 2–4 weeks)

| # | Action | Files |
|---|---|---|
| 5 | **Per-league meta re-scoring.** `meta_score_pitching` should become `meta_score_pitching_lb124` and `meta_score_pitching_lb122` — driven by the per-league calibration weights you already store in `meta_calibration`. The page surfaces whichever league is selected. | `core/meta_scoring.py`, `core/meta_calibration.py`, `cards` table schema. |
| 6 | **Refit the pitching calibrator with FIP- as a secondary target.** Stuff dominates the WAR fit because WAR rewards strikeouts. But run prevention in lb122 is closer to a FIP problem than a strikeout problem. Add an ensemble target. | `core/meta_calibration.py` |
| 7 | **Multi-SP / multi-RP slot semantics in `core/optimizer.py`.** Treat SP as 5 slots, RP as 5–7 slots, CL as 1. The DP knapsack already supports N positions; just expand the position list with constraints (5 SP must each have unique `card_id`). | `core/optimizer.py:7-9`, `_get_roster_starters` |
| 8 | **Role re-assignment within the bullpen.** Add a pre-pass in `_get_roster_starters` that picks the highest-meta RP/CL combo: best closer = max(RP+CL by leverage-weighted meta), then second-best = setup, etc. Surfaces "Promote Kerkering to CL (free, +108 meta)" automatically. | `core/optimizer.py`, page consumer. |
| 9 | **Platoon-aware starter resolution.** When dedup'ing starter rows by position, *keep* both halves of the platoon pair (rows differing in `bats`/`throws` keep both); use `meta_vs_rhp` / `meta_vs_lhp` to compute the effective expected value. Eliminates the false "wrong starter" flag for legitimate platoons. | `core/roster_analysis.py:_get_roster_starters` |
| 10 | **Snapshot-dedup all `*_stats` reads into a `_latest` view.** Build `batting_stats_latest` and `pitching_stats_latest` views with `(card_id, league_id) → MAX(snapshot_date)` projection. All consumer queries use the views. | `core/database.py` |

### Tier 3 — Architectural cleanup

| # | Action |
|---|---|
| 11 | **Refactor `4_Roster_Optimizer.py`.** The 5,432-line file is the single biggest source of behaviour drift. Split into `roster_fetch.py`, `chain_builder.py`, `priority_panel.py` modules with the same APIs the page consumes today. EPIC_F_REFACTOR_PLAN.md already exists in `/docs` — promote that to active work. |
| 12 | **Single source of truth for "starter at position".** Move the canonical resolution into `core/roster_analysis.py` and have the page consume it. Today the page re-implements it inline. |
| 13 | **Add a back-test harness.** Use the 31,729-row `game_log_at_bats` plus `game_pitching` data to run "optimizer recommendation X → did it improve actual outcomes Y games later?" Validate Tier 1 fixes empirically before shipping Tier 2. |

---

## 7. Confidence Statement & Open Questions

Confidence in each finding:

- **Headline league mismatch (Section 3):** 95%+. The data is unambiguous — same cards, two leagues, ERA 3.92 vs 5.25, optimizer wired to lb124 only.
- **Control under-weighting (Section 4):** 80%. The 27-point gap is real, the lb124 r=+0.013 calibration is the source. I have not run a lb122 multi-feature regression with FIP- as the target — that's the next falsifiable check.
- **One-slot pitching (Section 5.1):** 100%. Read directly from code.
- **Role re-assignment gap (Section 5.2):** 95%. Code path is clear.
- **Platoon collapse (Section 5.3):** 90%. Verified with one example; haven't enumerated every position.
- **Snapshot-multiply joins (Section 5.4):** 100% for the analysis path, ~60% likely for engine internals (would require a deeper trace through `meta_scoring.py`).

Open questions I'd want to resolve before shipping Tier 2 changes:

1. Is the user's i76 league active? (3rd league with 1,449 pitching_stats rows but no team_stats.) If so, that's a third optimization target.
2. Does OOTP's in-game manager respect the role assignments (CL/SU/MID) the optimizer expects? If the manager re-assigns based on its own logic, role changes need to be locked in OOTP, not just the optimizer's view.
3. Is there a confounding park-factor effect? lb122 might play in a hitter's park for TDK that lb124 doesn't. `park_info` data is ingested but not in meta — Section 12.4 of `ARCHITECTURE.md` already flags this as a known gap.
4. Are the buy recommendations actually being acted on, or is the user already self-correcting? `recommendation_log.verdict` is `pending` for the latest 1,179 recs — no closed-loop feedback yet.

---

## 8. What I Would Not Do

- **Do not** broadly down-tune Stuff weight. It correlates +0.357 in lb124 and +0.489 for relievers. Down-tuning globally fixes lb122 at the cost of lb124. The right move is per-league weights, not a single retune.
- **Do not** disable the calibration confidence cap (0.40 for pitching). With CV R² of 0.034, the prior is doing real work; ceding more to the empirical fit will introduce noise.
- **Do not** chase additional features (park factors, fielding DRS, arsenal diversity) before fixing the league-mismatch surface. Adding signal to a model that's optimizing the wrong league is wasted effort.
- **Do not** expand the underperformer-sell logic to auto-sell. The current 7-pitcher sell list is correct, but the user needs to see and approve each one — and crucially, identify a Stuff-light/Control-elite *replacement* before the round-trip.

---

*Report prepared 2026-04-25. Recommendations should be re-validated against fresh data after Tier 1 actions are completed (target: 2026-05-09 review).*
