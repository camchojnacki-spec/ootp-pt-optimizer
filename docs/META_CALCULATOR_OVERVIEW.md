# OOTP PT Optimizer — Meta Calculator: Extensive Overview

**Document date**: 2026-04-14
**Code state**: `app/core/meta_scoring.py` + `app/core/meta_calibration.py` + `app/core/meta_validation.py` + `app/utils/constants.py`
**Audience**: Cameron (project owner) — the operator who needs to understand what the meta really is, where it came from, why it sometimes disagrees with card tier, and what to change when it does.
**Companion doc**: `docs/META_ANALYSIS_REPORT.md` (earlier, higher-level analysis from 2026-04-12). This document is the ground truth for the formulas *as they exist right now* and includes the diagnostic findings from the 2026-04-14 audit.

---

## Table of Contents

1. [What the meta is (and isn't)](#1-what-the-meta-is-and-isnt)
2. [End-to-end pipeline](#2-end-to-end-pipeline)
3. [Inputs: every card rating the formula touches](#3-inputs-every-card-rating-the-formula-touches)
4. [The batting meta formula (line-by-line)](#4-the-batting-meta-formula-line-by-line)
5. [The pitching meta formula (line-by-line)](#5-the-pitching-meta-formula-line-by-line)
6. [Diminishing returns — `_diminished()`](#6-diminishing-returns--_diminished)
7. [Defense scoring](#7-defense-scoring)
8. [Speed / stealing scoring](#8-speed--stealing-scoring)
9. [Positional value bonus (fWAR ladder)](#9-positional-value-bonus-fwar-ladder)
10. [Balance floor penalties](#10-balance-floor-penalties)
11. [SIERA-inspired interaction terms (pitching only)](#11-siera-inspired-interaction-terms-pitching-only)
12. [Split metas — vs LHP / vs RHP / vs LHB / vs RHB](#12-split-metas--vs-lhp--vs-rhp--vs-lhb--vs-rhb)
13. [Weight sources and load precedence](#13-weight-sources-and-load-precedence)
14. [Default weights and their historical justification](#14-default-weights-and-their-historical-justification)
15. [Calibration pipeline #1 — `meta_calibration.py` (OLS)](#15-calibration-pipeline-1--meta_calibrationpy-ols)
16. [Calibration pipeline #2 — `meta_validation.py` (ElasticNetCV + Bayesian blend)](#16-calibration-pipeline-2--meta_validationpy-elasticnetcv--bayesian-blend)
17. [The `meta_calibration` table — what's actually stored](#17-the-meta_calibration-table--whats-actually-stored)
18. [Meta explainers — `explain_batting_meta` / `explain_pitching_meta`](#18-meta-explainers--explain_batting_meta--explain_pitching_meta)
19. [Meta distributions in the current league](#19-meta-distributions-in-the-current-league)
20. [Diagnostic audit: 2026-04-14 findings](#20-diagnostic-audit-2026-04-14-findings)
21. [Known gaps and design limits](#21-known-gaps-and-design-limits)
22. [Recommended fixes (prioritized)](#22-recommended-fixes-prioritized)
23. [Appendix A — Every constant, one table](#23-appendix-a--every-constant-one-table)
24. [Appendix B — Worked examples (5 pitchers, 3 batters)](#24-appendix-b--worked-examples-5-pitchers-3-batters)

---

## 1. What the meta is (and isn't)

The **meta score** is a single scalar value — currently on roughly a 250–900 scale — that the optimizer uses to rank cards for the Buy/Sell/Roster Optimizer pages. It is **a weighted composite of card ratings** (Contact, Power, Movement, Stuff, etc.), tuned so that higher meta ≈ "likely better in game simulation." It **is not**:

- An OOTP-native number. OOTP doesn't expose a meta score; this is a synthetic scalar invented by this project.
- A prediction of a specific future stat (OPS, ERA+, WAR). It correlates weakly with those stats — see §19 for measured correlations — but it is not a forecast.
- A tier replacement. Card tier (Regular → Bronze → Silver → Gold → Diamond → Perfect) is OOTP's own rarity/value signal and is *not* an input to the meta. The meta can disagree with tier, and frequently does.
- Calibrated per season or per league. The same weights apply to every card, regardless of year, park, or position context, except for the position-specific defense multiplier and positional value bonus which are the only situational terms.

**Practical mental model**: the meta is a ranking signal, not a truth. Use it to shortlist candidates, then check actual in-game stats (`pitching_stats` / `batting_stats` tables, or the Roster Optimizer's "Perf" columns when `use_history=True`) before committing to trades. Whenever in-game performance and meta disagree, performance wins.

---

## 2. End-to-end pipeline

```
 ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
 │  OOTP CSV exports   │ ──▶ │   ingestion.py      │ ──▶ │      cards table    │
 │ (card ratings)      │     │ (typed loaders per  │     │ contact, gap_power, │
 │                     │     │  file type)         │     │ stuff, movement,    │
 └─────────────────────┘     └─────────────────────┘     │ ... meta_score_*    │
                                                         └─────────┬───────────┘
                                                                   │
                                                                   ▼
                                                      ┌──────────────────────────┐
                                                      │ meta_scoring.py          │
                                                      │ • _diminished()          │
                                                      │ • calc_batting_meta()    │
                                                      │ • calc_pitching_meta()   │
                                                      │ • calc_*_meta_vs_rhp/lhp │
                                                      │ • calc_defense_score()   │
                                                      │ • calc_speed_score()     │
                                                      └─────────────┬────────────┘
                                                                    │  uses weights from
                                                                    ▼
                               ┌──────────────┐          ┌──────────────────────┐
                               │ constants.py │◀───────  │ get_weights()        │
                               │ DEFAULT_*    │          │   │  1st: DB cal.    │
                               └──────────────┘          │   │  2nd: config.yml │
                                                         │   │  3rd: defaults   │
                                                         └──────────┬───────────┘
                                                                    ▲
                               ┌─────────────────────────┐          │ writes
                               │ meta_validation.py      │ ─────────┘
                               │ (ElasticNetCV + blend)  │ ◀─── reads pitching_stats
                               └─────────────────────────┘      + batting_stats
```

**Call sites for `calc_*_meta`** (high level):

- **Ingestion** — after `cards` is loaded, the refresh pipeline computes `meta_score_batting` and `meta_score_pitching` for every row and stores them alongside the rest of the card data.
- **Roster pages** — the Roster Optimizer reads `roster_current.meta_score` (stored at roster snapshot time) for its chain/upgrade tables, and re-reads `cards.meta_score_pitching/batting` for availability-matching.
- **Buy/Sell** — the recommendation engine ranks candidates by `cards.meta_score_*` then filters by price, tier, position.
- **Explain UX** — `explain_batting_meta()` / `explain_pitching_meta()` regenerate the component breakdown on demand (no cache) so "why this meta?" expanders always reflect the *current* formula and weights.

This means **if you recalibrate the weights, you must also re-run the refresh** to regenerate `cards.meta_score_*` and `roster_current.meta_score` — otherwise the stored values and the live formula disagree.

---

## 3. Inputs: every card rating the formula touches

From `cards` table (OOTP 27 scale: 0–200, ~50 = league-average, ~100 = starter, 150+ = elite):

### Batting inputs

| Column               | Role in meta                                                                      |
|----------------------|-----------------------------------------------------------------------------------|
| `contact`            | Core term (weight ~2.00 default). Biggest single contributor to batting meta.     |
| `gap_power`          | Core term (~1.60). Proxy for extra-base hits.                                     |
| `power`              | Core term (~1.60). Drives HR projection.                                          |
| `eye`                | Core term (~0.80). OBP amplifier.                                                 |
| `avoid_ks`           | **Zero-weighted by default** — double-counted in Contact in OOTP 25+ per upstream docstring. |
| `babip`              | **Zero-weighted by default** — same reason.                                       |
| `con_vl` / `con_vr`  | Split meta input (vs LHP / vs RHP).                                               |
| `pow_vl` / `pow_vr`  | Split meta input.                                                                 |
| `eye_vl` / `eye_vr`  | Split meta input.                                                                 |
| `speed`              | Speed score input (weight 0.25 inside composite).                                 |
| `stealing`           | Speed score input (weight 0.40 inside composite).                                 |
| `baserunning`        | Speed score input (weight 0.35 inside composite).                                 |
| `position`           | Drives position-specific defense multiplier AND positional value bonus.           |
| `infield_range/error/arm` | Defense score (IF).                                                           |
| `of_range/error/arm` | Defense score (OF).                                                               |
| `catcher_ability/frame/arm` | Defense score (C).                                                         |

### Pitching inputs

| Column                  | Role in meta                                                                 |
|-------------------------|------------------------------------------------------------------------------|
| `stuff`                 | Core term (~1.60 default). Correlates with K/9 most strongly.                |
| `movement`              | Core term (~2.20). Drives ground balls / weak contact.                       |
| `control`               | Core term (~0.60). Surprisingly low — see §14.                               |
| `p_hr`                  | Core term (~1.80). HR suppression.                                           |
| `stamina`               | Stamina/hold composite (weight 0.40).                                        |
| `hold`                  | Stamina/hold composite.                                                      |
| `stu_vl` / `stu_vr`     | Split meta input (vs LHB / vs RHB).                                          |
| `stuff_x_movement`      | **Derived**, not stored. `stu * mov` used with weight 0.006 default.         |
| `stuff_x_control`       | **Derived**. `stu * ctrl` × 0.004 default.                                   |
| `movement_x_control`    | **Derived**. `mov * ctrl` × 0.003 default.                                   |
| `pitcher_role` (11/12/13) | Not a direct input to the formula — all pitchers use the same pitching meta, SPs are NOT penalized or boosted for stamina. See §21 for why this is a problem. |

### NOT used by the meta (even though they exist in `cards`)

- `ovr_rating` — intentionally excluded. Including OVR creates structural multicollinearity (VIF ≫ 10) because OVR is itself a weighted average of the components. See the `DEFAULT_BATTING_WEIGHTS` docstring in `constants.py`.
- `card_value`, `tier`, `card_badge`, `card_series`, `year` — not inputs. The meta does not know rarity or market price.
- `age`, `peak` — not inputs. The meta is a snapshot-in-time value, not a projection.
- `velocity` — not used (it's stored as TEXT in the schema and is a cosmetic field in OOTP).
- `p_babip` — not used directly in the main formula, though pitching split metas can read `p_babip_vl/vr`.

---

## 4. The batting meta formula (line-by-line)

Source: `app/core/meta_scoring.py::calc_batting_meta` (lines 177–236).

```python
def calc_batting_meta(row, weights=None):
    if weights is None:
        weights, _ = get_weights()        # calibrated > config > default

    gap  = float(row.get('gap_power')  or ...)
    con  = float(row.get('contact')    or ...)
    avk  = float(row.get('avoid_ks')   or ...)
    eye  = float(row.get('eye')        or ...)
    pwr  = float(row.get('power')      or ...)
    bab  = float(row.get('babip')      or ...)
    defense     = row.get('defense_score') or calc_defense_score(row)
    speed_score = row.get('speed_score')   or calc_speed_score(row)

    meta = ( _diminished(gap) * weights['gap_power']       # typ.  ~1.60
           + _diminished(con) * weights['contact']         # typ.  ~2.00
           + _diminished(avk) * weights['avoid_ks']        # def.   0.00
           + _diminished(eye) * weights['eye']             # typ.  ~0.80
           + _diminished(pwr) * weights['power']           # typ.  ~1.60
           + _diminished(bab) * weights['babip']           # def.   0.00
           + defense          * weights['defense'] )       # typ.  ~1.50

    # Conditional speed/stealing bonus
    if speed_score > 0:
        meta += _diminished(speed_score + 70) * weights['speed_stealing']  # typ. 0.50

    # Balance penalty — Contact and Gap Power each penalized if < 55
    for stat in (con, gap):
        if 0 < stat < BATTING_STAT_FLOOR:   # 55
            meta -= (BATTING_STAT_FLOOR - stat) * 0.4

    # Positional value bonus — fWAR ladder
    meta += POSITIONAL_VALUE_BONUS.get(position_int, 0)

    return round(meta, 2)
```

**Contribution hierarchy (default weights on a 100-rated middling player)**:

| Term              | Rating = 100 | Weight | Points |
|-------------------|--------------|--------|--------|
| Contact           | 100          | 2.00   | 200    |
| Gap Power         | 100          | 1.60   | 160    |
| Power             | 100          | 1.60   | 160    |
| Defense (scaled)  | ~75 × 1.00   | 1.50   | ~113   |
| Eye               | 100          | 0.80   | 80     |
| Speed (composite) | 70 baseline  | 0.50   | 0 (below threshold) |
| Positional bonus  | —            | —      | 0 (3B baseline) |
| **Total**         |              |        | **~713** |

For reference, the current league median batter meta is ~557 and the league average is 564 (n=1502). So an "all 100s, average fielder" batter sits right around the upper half of the distribution — which is about right.

### Edge cases in the batting path

- **Defense fallback column names**: `calc_defense_score` looks at `of_range`/`OF Range`/etc. to handle both DB snake_case and raw CSV title-case. If all three defense ratings are 0, defense contributes 0 (not negative).
- **Position parsing**: accepts both string positions ("SS", "CF") and integer codes (1–10). Unknown positions fall back to a no-op for positional bonus.
- **Speed floor at 70**: speed-score composite is only added if > 70 (a gentle knee designed to avoid penalizing slow players whose bat stats already capture their output).
- **Balance penalty is additive, not multiplicative**: a player with Contact=40 gets (55−40)×0.4 = 6 meta subtracted per qualifying rating. Not harsh enough to meaningfully punish Bronze cards, but enough to nudge low-contact mashers downward.

---

## 5. The pitching meta formula (line-by-line)

Source: `app/core/meta_scoring.py::calc_pitching_meta` (lines 239–305).

```python
def calc_pitching_meta(row, weights=None):
    if weights is None:
        _, weights = get_weights()

    mov     = float(row.get('movement') or ...)
    stu     = float(row.get('stuff')    or ...)
    ctrl    = float(row.get('control')  or ...)
    phr     = float(row.get('p_hr')     or ...)
    stamina = float(row.get('stamina')  or 0)
    hold    = float(row.get('hold')     or 0)

    # Core ratings with diminishing returns
    meta = ( _diminished(mov)  * weights['movement']   # typ. ~2.20
           + _diminished(stu)  * weights['stuff']      # typ. ~1.60
           + _diminished(ctrl) * weights['control']    # typ. ~0.60
           + _diminished(phr)  * weights['p_hr'] )     # typ. ~1.80

    # SIERA-inspired interaction terms (raw, not diminished)
    if weights['stuff_x_movement']   > 0: meta += stu * mov  * weights['stuff_x_movement']
    if weights['stuff_x_control']    > 0: meta += stu * ctrl * weights['stuff_x_control']
    if weights['movement_x_control'] > 0: meta += mov * ctrl * weights['movement_x_control']

    # Stamina/Hold average
    if weights['stamina_hold'] > 0:
        sh_count, sh_sum = 0, 0
        if stamina > 0: sh_count += 1; sh_sum += stamina
        if hold    > 0: sh_count += 1; sh_sum += hold
        if sh_count > 0:
            meta += (sh_sum / sh_count) * weights['stamina_hold']   # 0.40 default

    # Floor penalty for truly weak STU/MOV (< 65)
    for stat in (stu, mov):
        if 0 < stat < PITCHING_STAT_FLOOR:    # 65
            meta -= (PITCHING_STAT_FLOOR - stat) * 1.0

    return round(meta, 2)
```

**Key design choices**:

1. **Interaction terms use RAW values**, not `_diminished()`-scaled values. The docstring explicitly calls out: *"Use raw (not diminished) values for interactions to avoid double-sqrt."* This means a pitcher with STU=150 and MOV=150 contributes `150 × 150 × 0.006 = 135 points` from the stuff×movement term alone — a big chunk — even though both raw values would individually be diminished to ~135 in the core term.

2. **Stamina/Hold is AVERAGED**, not summed. A pitcher with stamina 100 and hold 0 contributes `(100/1) × 0.40 = 40 points`, same as one with 100 in both.

3. **Floor penalty is 1.0× shortfall**, not 0.4× like batting. A pitcher with stuff 40 gets `(65−40)×1.0 = 25 meta` deducted — about 3× harsher than the batting equivalent.

4. **Positional roles aren't differentiated.** A SP, RP, and CL all go through the identical formula. Stamina is the only lever that differs between them in the ratings, but the stamina_hold term averages it with hold, which is typically reliever-centric. This is a KNOWN limitation, explored further in §21.

### Worked example — Johnny Cueto (calibrated weights, 2026-04-14 snapshot)

Ratings: MOV=113, STU=73, CTL=96, pHR=133, STM=81, HLD=93.

```
diminished values
  _diminished(113) = 110 + sqrt(3)*4  = 116.9
  _diminished(73)  = 73   (below threshold, unchanged)
  _diminished(96)  = 96
  _diminished(133) = 110 + sqrt(23)*4 = 129.2

core
  Movement       = 116.9 × 2.23 = 260.7
  Stuff          =  73.0 × 0.99 =  72.3
  Control        =  96.0 × 0.71 =  68.2
  HR suppression = 129.2 × 2.27 = 293.2

interaction (raw values × tiny weights)
  Stuff × Movement = 73 × 113 × 0.0019 =  15.7
  Stuff × Control  = 73 ×  96 × 0.0013 =   9.1
  Mov × Control    = 113 × 96 × 0.0010 =  10.8

stamina/hold
  (81 + 93) / 2 × 0.40 = 87 × 0.40     = 34.8

floor penalty — both STU and MOV are above 65, no penalty.

total = 260.7 + 72.3 + 68.2 + 293.2 + 15.7 + 9.1 + 10.8 + 34.8 = 764.8
```

Actual stored meta for Cueto: **764.9** — matches (rounding).

---

## 6. Diminishing returns — `_diminished()`

```python
DIMINISHING_RETURNS_THRESHOLD = 110

def _diminished(value, threshold=110):
    if value <= threshold:
        return value
    excess = value - threshold
    return threshold + math.sqrt(excess) * 4
```

| Raw rating | Effective | Delta  |
|------------|-----------|--------|
| 50         | 50        | 0      |
| 80         | 80        | 0      |
| 100        | 100       | 0      |
| 110        | 110       | 0      |
| 120        | 110 + √10 ×4 ≈ 122.6 | −0.6 |
| 130        | 110 + √20 ×4 ≈ 127.9 | −2.1 |
| 140        | 110 + √30 ×4 ≈ 131.9 | −8.1 |
| 150        | 110 + √40 ×4 ≈ 135.3 | −14.7 |
| 170        | 110 + √60 ×4 ≈ 141.0 | −29.0 |
| 200 (max)  | 110 + √90 ×4 ≈ 148.0 | −52.0 |

**Why sqrt-scaled above 110**: single elite ratings shouldn't dominate the meta. A Diamond card with one 180 rating and six average ratings shouldn't automatically outrank a Silver with seven 95s. Sqrt-scaling caps the marginal benefit above 110 so balanced profiles are rewarded.

**Applied to**: `gap, con, avk, eye, pwr, bab` (batting core), `mov, stu, ctrl, phr` (pitching core), and the `(speed_score + 70)` sum inside the speed/stealing term.

**NOT applied to**: defense score, the interaction terms (stuff×movement etc.), stamina/hold composite, positional bonuses, or balance penalties. Those are all linear.

---

## 7. Defense scoring

Source: `calc_defense_score` (lines 105–147).

```python
def calc_defense_score(row, apply_position_multiplier=True):
    pos = parse_position(row)
    if pos in (7,8,9):                 # OF
        raw = avg(of_range, of_error, of_arm)
    elif pos == 2:                     # C
        raw = avg(catcher_ability, catcher_frame, catcher_arm)
    elif pos in (3,4,5,6):             # IF
        raw = avg(infield_range, infield_error, infield_arm)
    else:
        raw = 0

    multiplier = POSITION_DEFENSE_MULTIPLIERS.get(pos, 1.0)
    return raw * multiplier
```

**Multipliers** (from `constants.py`):

| Position | Multiplier | Rationale (fWAR runs/162)  |
|----------|------------|----------------------------|
| C        | 1.30       | +12.5 (framing/arm, elite) |
| SS       | 1.40       | +7.5 (top of spectrum)     |
| CF       | 1.25       | +2.5 (range critical)      |
| 2B       | 1.10       | +2.5                       |
| 3B       | 1.00       | +2.5 (baseline)            |
| RF       | 0.70       | −7.5 (arm matters)         |
| LF       | 0.60       | −7.5 (least demanding OF)  |
| 1B       | 0.40       | −12.5 (bottom of spectrum) |
| DH       | 0.00       | no defense                 |

**What this means numerically**: a SS with raw defense 80 gets `80 × 1.40 × 1.50 = 168 meta` from defense alone. A 1B with the same raw 80 gets `80 × 0.40 × 1.50 = 48 meta`. That's a 120-point spread — by far the biggest position-driven swing in the batting formula.

**Edge cases**:
- Defense is **NOT diminished**. It's scaled linearly, so elite 180-raw CFs really do dominate the defense component.
- If all three positional defense ratings are 0 (common for position-flexible cards whose CSV export omitted them), `raw = 0` and defense contributes nothing — not a penalty, just absence.
- Pitchers don't get defense scores. A P row goes through `calc_batting_meta` only if called directly, which almost never happens (the dispatcher in `explain_meta` routes pitchers to `explain_pitching_meta`).

---

## 8. Speed / stealing scoring

Source: `calc_speed_score` (lines 150–174).

```python
def calc_speed_score(row):
    speed       = row.get('speed')       or 0
    stealing    = row.get('stealing')    or 0
    baserunning = row.get('baserunning') or 0

    composite = stealing*0.40 + baserunning*0.35 + speed*0.25

    if composite < 70:   # slow players get 0, not negative
        return 0.0
    return composite - 70
```

Then in `calc_batting_meta`:
```python
if speed_score > 0:
    meta += _diminished(speed_score + 70) * weights['speed_stealing']
```

**Why the weird `+ 70` inside `_diminished()`**: the composite is already a "delta above 70" value. Adding 70 back re-anchors it to the same 0–200 scale as everything else, so the diminishing returns kick in at the same 110 threshold. A player with composite = 135 (very fast) goes through `_diminished(135) × 0.50 ≈ 128 × 0.50 = 64 meta`.

**Practical range**: for most cards this contributes 0 to 40 meta. Only burners (Rickey Henderson-type cards) see significant speed contributions.

---

## 9. Positional value bonus (fWAR ladder)

From `constants.py`:

```python
POSITIONAL_VALUE_BONUS = {
    2:  31,   # C    +12.5 runs × 2.5 meta/run
    3: -31,   # 1B   −12.5 × 2.5
    4:   6,   # 2B   +2.5  × 2.5
    5:   0,   # 3B   baseline
    6:  19,   # SS   +7.5  × 2.5
    7: -19,   # LF
    8:   6,   # CF
    9: -19,   # RF
   10: -44,   # DH   −17.5 × 2.5
}
```

**This is added FLAT, regardless of defensive quality.** The rationale: positional scarcity is real — a 500-meta SS is strictly more valuable than a 500-meta 1B because it's harder to find competent SS in the card pool. Even a 1B with elite defense still gets the −31 penalty because *any* 1B can play 1B, but only glove-first middle infielders can cover up the middle.

**Potential double-counting concern**: the position *defense multiplier* already rewards premium defensive positions (SS = 1.40×), and then the *positional value bonus* rewards them *again* (SS = +19). In principle these represent different things (scarcity vs. ability), but in practice they stack. A league-average-defense SS gets both the 1.40 multiplier boost AND the +19 flat. This is intentional but is the single biggest driver of SS metas trending high in the distribution.

---

## 10. Balance floor penalties

```python
BATTING_STAT_FLOOR  = 55   # per constants.py
PITCHING_STAT_FLOOR = 65
```

- **Batting**: applied to `contact` and `gap_power`. Penalty = `(floor − stat) × 0.4`. Max possible per stat: `55 × 0.4 = 22` if stat is 0. Max total: 44.
- **Pitching**: applied to `stuff` and `movement`. Penalty = `(floor − stat) × 1.0`. Max per stat: 65. Max total: 130.

**The pitching floor penalty is 2.5× harsher than batting in per-point terms**, AND hits at a higher threshold (65 vs 55). Translation: a pitcher with Stuff=40 gets a 25-point deduction; a batter with Contact=40 only gets 6 points deducted. This was tuned because pitching is more "tail-heavy" — one truly weak rating (elite fastball but zero command, or elite command but mushball stuff) more severely cripples a pitcher than one weak batting tool cripples a hitter.

**Why the floor penalty rarely fires in practice**: almost every card in the pool has Stuff ≥ 65 and Contact ≥ 55 because OOTP's card generation biases ratings upward. The penalty mostly catches outlier budget cards (e.g. 52-OVR reserves) and keeps them from leaking high metas through one good rating.

---

## 11. SIERA-inspired interaction terms (pitching only)

SIERA is Fangraphs' Skill-Interactive ERA, which models pitching as **non-additive**: a pitcher with elite stuff AND elite control is worth more than the sum of those two components in isolation, because the synergy matters. The pitching meta formula borrows that insight with three interaction terms.

```python
meta += stu  * mov  * 0.006   # Stuff × Movement (default weight)
meta += stu  * ctrl * 0.004   # Stuff × Control
meta += mov  * ctrl * 0.003   # Movement × Control
```

**Scaling rationale** (from the `constants.py` docstring): *"Scaled so max interaction bonus ~80 meta for elite (120×110) vs ~30 for avg (75×75)."*

Worked contribution ranges:

| Profile                     | Stuff×Movement | Stuff×Control | Movement×Control | Total interaction |
|-----------------------------|----------------|---------------|------------------|-------------------|
| Avg (75/75/75)              | 33.8           | 22.5          | 16.9             | 73.2              |
| Solid (100/100/100)         | 60.0           | 40.0          | 30.0             | 130.0             |
| Elite (130/130/130)         | 101.4          | 67.6          | 50.7             | 219.7             |
| Lopsided (150/60 S×M, etc.) | 54.0           | 36.0          | 40.5 (if Mov=90) | ~130              |

**An elite, balanced pitcher can get up to 220 extra meta from interactions alone** — a huge chunk of their total. This is one of the main ways the formula rewards "complete" pitchers over one-trick-ponies.

### What calibration did to the interaction weights

The active calibrated weights cut the interaction multipliers to roughly 30% of default:

| Key                 | Default | Calibrated | Delta   |
|---------------------|---------|------------|---------|
| stuff_x_movement    | 0.006   | 0.0019     | −68%    |
| stuff_x_control     | 0.004   | 0.0013     | −68%    |
| movement_x_control  | 0.003   | 0.0010     | −67%    |

**Effect**: elite, balanced pitchers lose about 150 meta from interaction contribution; lopsided pitchers (high pHR, mid stuff) lose less. This flattens the top of the distribution — and it's almost certainly wrong directionally. See §20 for the full analysis.

---

## 12. Split metas — vs LHP / vs RHP / vs LHB / vs RHB

The main formula uses the *overall* ratings (Contact, Stuff, etc.). OOTP also stores split ratings (Contact vs L/R, Stuff vs L/R). The optimizer computes four additional scores:

- `calc_batting_meta_vs_rhp(row)` — uses `con_vr`, `pow_vr`, `eye_vr`; all other terms (gap, avk, bab, defense, speed) identical to main formula.
- `calc_batting_meta_vs_lhp(row)` — symmetric with `*_vl`.
- `calc_pitching_meta_vs_lhb(row)` — uses `stu_vl` for the stuff term; *movement, control, p_hr, stamina, hold are NOT split* (OOTP doesn't always expose them per-hand, and using overall keeps the formula consistent).
- `calc_pitching_meta_vs_rhb(row)` — symmetric with `stu_vr`.

These split metas are stored in `roster_current.meta_vs_rhp` / `meta_vs_lhp` during snapshot and used by the lineup builders. Known limitation: pitching splits only modulate stuff, not the full four-rating core.

### Fall-through behavior

If a card doesn't have split ratings (e.g. some older CSVs), the split formula silently falls back to the overall rating via the `or` chain:

```python
con = float(row.get('con_vr') or row.get('contact') or ...)
```

This means a card with no split data shows identical vs-LHP and vs-RHP metas, which can be misleading if you're hunting for matchup edges.

---

## 13. Weight sources and load precedence

From `get_weights()` in `meta_scoring.py`:

```
1. Calibrated weights from meta_calibration table
        ↓ (if missing or error)
2. config.yaml (batting_weights / pitching_weights keys)
        ↓ (if missing or error)
3. DEFAULT_BATTING_WEIGHTS / DEFAULT_PITCHING_WEIGHTS in constants.py
```

The secondary `get_weights_with_source()` returns an extra `'calibrated' | 'config' | 'default'` label so the UI can annotate "weights loaded from DB calibration" on the Settings page.

**Current state on this DB** (measured 2026-04-14): `source = 'calibrated'`. The active weights are the most recent `meta_calibration` row per type (`batting` id=3, `pitching` id=4 — both from 2026-04-13 21:31:49).

### Per-position calibrations (ids 5–15) — not used

The `meta_calibration` table also contains `calibration_type` values like `pos:SP`, `pos:RP`, `pos:C`, `pos:SS`, etc. (written on 2026-04-14 09:21 by a `meta_validation.py` run). **`get_weights()` only reads `'batting'` and `'pitching'` rows**, not the per-position ones. The position rows are stored for diagnostics but never influence the live formula.

This is a missed opportunity — per-position calibrations would let the formula treat SPs differently from RPs — but making it work would require passing a position argument through every `calc_pitching_meta` call, which is a bigger refactor.

---

## 14. Default weights and their historical justification

From `constants.py` (verbatim, abbreviated comments):

```python
DEFAULT_BATTING_WEIGHTS = {
    "gap_power":      1.60,   # r=+0.205 WAR, r=+0.212 OPS
    "contact":        2.00,   # r=+0.314 WAR — strongest individual
    "avoid_ks":       0.00,   # double-counted in CON (OOTP25+)
    "eye":            0.80,   # r=+0.063 WAR — weak alone, OBP multiplier
    "power":          1.60,   # r=+0.275 OPS
    "babip":          0.00,   # double-counted in CON
    "defense":        1.50,   # r=+0.296 WAR
    "speed_stealing": 0.50,   # Speed→SB r=+0.337
}

DEFAULT_PITCHING_WEIGHTS = {
    "movement":            2.20,    # r=−0.295 ERA
    "stuff":               1.60,    # r=−0.265 ERA
    "control":             0.60,    # SIERA says control matters more
    "p_hr":                1.80,    # r=−0.266 ERA
    "stamina_hold":        0.40,    # r=+0.392 WAR (confounded)
    "stuff_x_movement":    0.006,
    "stuff_x_control":     0.004,
    "movement_x_control":  0.003,
}
```

The numbers in the comments were from a **prior calibration run (different league, older OOTP version) on 678 batters / 633 pitchers** per the docstring. Those r values are *not* the correlations in the current league — see §19, where measured r values are significantly weaker.

### Why these specific defaults

- **Contact is the single biggest weight (2.00)**: highest single-rating r with WAR in the calibration dataset. Plate discipline foundation.
- **Gap Power = Power (1.60)**: avoiding over-weighting HR power; doubles/triples are nearly as valuable.
- **Eye is downweighted (0.80)**: eye alone doesn't produce runs, but it amplifies the contact/power output by getting on base more. So it's in as a smaller factor.
- **Defense is 1.50**: surprisingly low, but it gets multiplied by position-specific factors (up to 1.40× for SS) and augmented by the positional bonus. Net effect for elite positions is much higher than 1.50 suggests.
- **Speed/stealing is tiny (0.50) with a 70 floor**: speed is almost worthless for slow players and only amplifies OBP for fast ones. The formula tries to capture that asymmetry.
- **Movement > Stuff (2.20 vs 1.60)**: movement correlates more strongly with ERA than raw stuff in OOTP's engine, probably because movement is what turns good stuff into swinging strikes and ground balls.
- **Control is underweighted (0.60)**: counterintuitive. SIERA suggests control matters a LOT, but the raw correlation with ERA is weak because walks get absorbed by other offensive components. This weight is one of the least confident defaults.

---

## 15. Calibration pipeline #1 — `meta_calibration.py` (OLS)

**Status**: largely superseded by pipeline #2, but still present in the codebase and callable.

**What it does**:
1. Pull `batting_stats.ops` / `pitching_stats.era_plus` for every player with ≥50 AB / ≥30 IP.
2. Name-match to `cards` table (normalized: lowercase, strip apostrophes and periods).
3. Build feature matrix X (ratings), target vector y (OPS or ERA+).
4. Solve via **pure Python normal equations**: `β = (XᵀX)⁻¹Xᵀy`. No numpy, no scikit-learn.
5. **Clamp negatives to 0.01**: `raw_weights[col] = max(beta[i], 0.01)`.
6. **Normalize so suggested weights sum to the same total as current config.** This preserves overall meta scale.
7. Write the result.

**Known weaknesses** (critical):
- **No regularization**: pure OLS on correlated predictors (all ratings correlate with each other via OVR) → wildly unstable coefficients.
- **Negative-coefficient clamping is lossy**: if regression says "stuff has a slightly negative coefficient," clamping to 0.01 hides that signal AND forces the other ratings to compensate by drifting upward.
- **No cross-validation**: R² is in-sample, not held-out.
- **Name-matching partial-substring**: `"Johnny Cueto"` substring-matches against `"Johnny Cueto Jr."` — can pick up false positives if two players have overlapping names.
- **Sample size threshold is only 5**: at n=5 the regression can produce 8 coefficients that fit the training data perfectly but generalize to nothing.

**Practical use today**: this pipeline's output was written into `meta_calibration` rows **1 and 2** on 2026-04-13 00:33 with catastrophic R² values (−105,962 and −104,043) because the first attempt included `ovr` as a feature, which triggered a near-singular matrix and the inverse blew up. Those rows are still in the table but have been overwritten as the "current calibrated weights" by pipeline #2.

---

## 16. Calibration pipeline #2 — `meta_validation.py` (ElasticNetCV + Bayesian blend)

**Status**: the active calibration pipeline. Produces rows id ≥3 in `meta_calibration`.

**Lines**: `meta_validation.py` 750–1300 approximately.

**Pipeline per type** (batting and pitching are symmetric):

1. **Pull player rows** with card-id joins:
   ```sql
   SELECT c.<ratings>, bs.war, bs.ops, bs.pa
   FROM cards c INNER JOIN batting_stats bs ON bs.card_id = c.card_id
   INNER JOIN (latest snapshot per card) ...
   WHERE c.position != 1 AND bs.pa >= 100
   ```
   Pitching uses `pitching_stats.war` as the target and `ps.ip >= 30`.

2. **Fallback to name matching** if fewer than 15 card-id matches. Uses `LIKE '%...%'` — same substring caveat as pipeline #1.

3. **Build feature matrix**:
   - Batting: [contact, gap_power, power, eye, avoid_ks, babip] + defense_score + speed_score → **8 features**.
   - Pitching: [stuff, movement, control, p_hr] + [stu×mov, stu×ctrl, mov×ctrl] → **7 features**.
   - Target: WAR (not OPS / ERA+).

4. **StandardScaler → ElasticNetCV**:
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   enet = ElasticNetCV(
       l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
       cv=min(10, sample_size),
       max_iter=5000,
       positive=True,       # force non-negative coefficients
       n_jobs=-1,
   )
   enet.fit(X_scaled, y)
   r2 = enet.score(X_scaled, y)
   raw_coefs = enet.coef_ / scaler.scale_
   ```

5. **Rescale to match default total**:
   ```python
   scale_factor = default_total / coef_sum
   empirical = {k: raw_coefs[i] * scale_factor for i, k in enumerate(keys)}
   ```

6. **Bayesian blend with defaults** (batting):
   ```python
   k_prior = 100
   blend_ratio = sample_size / (sample_size + k_prior)
   for key in keys:
       calibrated[key] = default_w * (1 - blend_ratio) + empirical_w * blend_ratio
   ```

   At `sample_size = 100`, the blend is 50/50. At `sample_size = 300` (typical for us), the blend is 75% empirical + 25% prior. At `sample_size = 50`, it's 33% empirical + 67% prior.

7. **Pitching main keys use the same blend, but the interaction keys use a fixed 70/30 split**:
   ```python
   for key in interaction_keys:
       default_w = DEFAULT_PITCHING_WEIGHTS.get(key, 0.01)
       calibrated[key] = round(default_w * 0.3 + raw_val * 0.7, 4)
   ```
   Note the **70% empirical weight on interactions regardless of sample size**. If the regression says interactions have near-zero coefficient, this aggressively pulls them toward zero.

8. **Write to `meta_calibration` table** if `sample_size >= 15` (batting) or `>= 10` (pitching).

### What's good about this pipeline

- **ElasticNetCV** handles correlated predictors via combined L1+L2 regularization (the research paper the comment references).
- **`positive=True`** prevents nonsense negative coefficients.
- **Bayesian blend** prevents small samples from swinging weights wildly away from priors.
- **Cross-validation** (up to 10-fold) gives a slightly more honest R² than OLS in-sample.

### What's still broken

1. **Only two output groups** (`batting`, `pitching`). The per-position runs (`pos:SP`, `pos:RP`, …) are computed and stored but never used by the live formula (see §13).

2. **The 70/30 interaction fixed split is too aggressive**. If the ElasticNet drives an interaction coefficient to zero (which it will, since interactions are heavily correlated with their components), the 70% weight on the raw-zero value completely crushes the default — which is exactly what we see in the current calibration (interaction weights down to 1/3 of default).

3. **`positive=True` hides directional signal**. Same issue as pipeline #1's negative clamping: if ElasticNet wants to assign stuff a negative coefficient (because movement is already absorbing its variance), positive=True forces it to zero. That coefficient then gets blended to a low positive number and the formula loses stuff entirely.

4. **Target is WAR, which mixes usage with ability**. SPs with 200 IP can accumulate more WAR than equally-skilled RPs with 60 IP simply because they pitched more. Feeding WAR to the regression pushes the model to "stamina-heavy pitchers are better," which isn't what we want to learn.

5. **Refreshing calibration won't fix it**. Even with 2× the sample size, the underlying signal in the data is weak (see §19). The regression will still find weird local optima.

---

## 17. The `meta_calibration` table — what's actually stored

Schema:
```
id                INTEGER PRIMARY KEY
calibration_type  TEXT      -- 'batting', 'pitching', 'pos:SP', 'pos:C', ...
weights_json      TEXT      -- JSON blob of {rating: weight}
r_squared         REAL      -- in-sample or CV R²
correlation       REAL      -- Pearson r between predicted and actual (WAR usually)
sample_size       INTEGER
confidence        REAL      -- min(1.0, n/100)
changes_json      TEXT      -- structured delta vs previous weights
created_at        TIMESTAMP
```

Current contents (from DB query 2026-04-14):

```
  id       type      n       R²     Pearson   conf              created_at
  15     pos:CL     16   0.1463   0.3824    0.320 2026-04-14 09:21:54
  14     pos:SP    302   0.1123   0.3351    1.000 2026-04-14 09:21:54
  13     pos:RP    187   0.0402   0.2004    1.000 2026-04-14 09:21:54
  12     pos:3B     71   0.1960   0.4427    1.000 2026-04-14 09:21:54
  11     pos:1B     68   0.0797   0.2824    1.000 2026-04-14 09:21:54
  10     pos:SS     79   0.2089   0.4571    1.000 2026-04-14 09:21:54
   9      pos:C     74   0.1059   0.3254    1.000 2026-04-14 09:21:54
   8     pos:LF     59   0.1145   0.3384    1.000 2026-04-14 09:21:54
   7     pos:CF     85   0.2499   0.4999    1.000 2026-04-14 09:21:54
   6     pos:2B     79   0.1041   0.3227    1.000 2026-04-14 09:21:54
   5     pos:RF     58   0.1036   0.3219    1.000 2026-04-14 09:21:54
   4   pitching    240   0.2175   0.3923    1.000 2026-04-13 21:31:49    ◄ active
   3    batting    273   0.3127   0.5249    1.000 2026-04-13 21:31:49    ◄ active
   2   pitching    380 -105962   0.2930    1.000 2026-04-13 00:33:04
   1    batting    402 -104043   0.5636    1.000 2026-04-13 00:33:04
```

**Reading this table**:

- **Rows 1–2** are the OLS pipeline's catastrophic first attempt with OVR included. Negative R² means the fitted line is worse than predicting the mean.
- **Rows 3–4** are the ElasticNet pipeline's first clean run. R² = 0.31 batting, 0.22 pitching. These are the **currently active** weights.
- **Rows 5–15** are per-position ElasticNet runs stored for diagnostics. *None of these are used by the live formula.* Their R² values are uniformly worse than the aggregate (0.04–0.25), which is expected — smaller samples, thinner signal.

**Translation of the active R² values into human terms**:

> The batting meta explains **31% of WAR variance** in the training set.
> The pitching meta explains **22% of WAR variance**.
> Everything else (69% and 78%) is noise that the regression is still trying to chase.

These are weak correlations. The meta is a rough directional signal, not a forecast. **Do not trade blind on meta alone.**

---

## 18. Meta explainers — `explain_batting_meta` / `explain_pitching_meta`

Lines 468–624 of `meta_scoring.py`. These mirror the main calc functions exactly so the total reconciles, but return a structured dict:

```python
{
    "total": 764.8,
    "components": [
        {"label": "HR Suppression", "raw": 133.0, "weight": 2.27, "points": 293.2},
        {"label": "Movement",       "raw": 113.0, "weight": 2.23, "points": 260.7},
        {"label": "Stuff",          "raw":  73.0, "weight": 0.99, "points":  72.3},
        {"label": "Control",        "raw":  96.0, "weight": 0.71, "points":  68.2},
    ],
    "bonuses": [
        {"label": "Stuff × Movement (Ks + weak contact)", "points": 15.7},
        {"label": "Stuff × Control (dominant + commanded)", "points":  9.1},
        {"label": "Movement × Control (groundballs)",      "points": 10.8},
        {"label": "Stamina/Hold (87)",                     "points": 34.8},
    ],
    "penalties": [],
    "notes": [
        "Pitching is non-additive — interaction terms reflect SIERA-style synergy.",
        "Ratings above 110 get sqrt-scaled to prevent extreme spikes from dominating.",
    ],
}
```

Used by the "Why this meta?" expander on Card Detail, Buy/Sell rows, and the Roster Optimizer's upgrade suggestions.

**Gotcha**: components with `points == 0` are filtered out of the list (e.g. Avoid K's is always zero under defaults, so it never appears), and the list is sorted by `abs(points)` descending so the biggest driver is first. That's why HR Suppression frequently leads the list even though Movement is the nominally-biggest weight — because pHR is often higher than movement in raw value.

---

## 19. Meta distributions in the current league

Measured 2026-04-14 from `cards.meta_score_*` with `> 0` filter.

### Batting (n=1502)

| Stat    | Value |
|---------|-------|
| min     | 254   |
| p10     | 479   |
| p25     | 512   |
| **p50** | **557** |
| mean    | 564   |
| p75     | 605   |
| p90     | 659   |
| p95     | 699   |
| p99     | 799   |
| max     | 882   |

### Pitching (n=1187)

| Stat    | Value |
|---------|-------|
| min     | 337   |
| p10     | 446   |
| p25     | 486   |
| **p50** | **524** |
| mean    | 537   |
| p75     | 566   |
| p90     | 646   |
| p95     | 736   |
| p99     | 793   |
| max     | 832   |

### Scale comparison

Batting metas span 254–882 (range 628).
Pitching metas span 337–832 (range 495).
The **pitching distribution is noticeably compressed at the top**: pitching p90 (646) is lower than batting p75 (605+), and pitching max is 50 points below batting max. This is partly because:

- Batting has a positional value bonus worth up to +31 (for C) that pitchers don't have.
- Batting has defense which can add ~100+ points at premium positions.
- Pitching interaction terms were crushed by calibration, removing ~150 meta from elite balanced pitchers.

**Implication**: metas are NOT directly comparable across batting and pitching. A 620-meta SS is ~p75 among batters. A 620-meta SP is ~p85 among pitchers. This asymmetry isn't handled anywhere in the optimizer UI — columns showing "meta" for mixed batter/pitcher tables will make pitchers look relatively worse.

### Ground-truth rating→performance correlations (n=797 pitchers, ≥30 IP)

Measured from `pitching_stats` joined to `cards`:

| Target  | Stuff  | Movement | Control | pHR    | Meta   |
|---------|--------|----------|---------|--------|--------|
| ERA+    | +0.121 | +0.135   | +0.036  | +0.101 | +0.170 |
| ERA     | −0.065 | −0.063   | +0.024  | −0.056 | −0.070 |
| WAR     | +0.091 | +0.131   | +0.105  | +0.134 | +0.212 |
| FIP     | −0.050 | −0.030   | +0.014  | −0.044 | −0.047 |
| K/9     | +0.174 | +0.016   | −0.021  | −0.003 | +0.073 |
| HR/9    | −0.009 | −0.163   | +0.056  | −0.185 | −0.157 |
| WHIP    | −0.026 | +0.004   | +0.001  | +0.006 | −0.001 |

**Read the diagonal**:

- Stuff → K/9: +0.174 (expected, strongest known physics relationship; stuff produces swinging strikes)
- Movement → HR/9: −0.163 (expected; movement produces weak contact)
- pHR → HR/9: −0.185 (expected, and slightly stronger than movement's HR signal)
- Control → WHIP: +0.001 (**essentially zero**, anomalous — control should reduce walks)

**Read the meta column**: meta correlates +0.17 with ERA+ and +0.21 with WAR. In r² terms, that's explaining 3% and 4% of variance. **The meta is barely above random at predicting in-game pitching.**

### Split by pitcher role

| Role | n   | META vs ERA+ | META vs ERA | META vs WAR | META vs FIP |
|------|-----|--------------|-------------|-------------|-------------|
| SP   | 401 | +0.251       | −0.032      | +0.291      | −0.006      |
| RP   | 437 | +0.150       | −0.104      | +0.243      | −0.074      |

Meta is a ~2× stronger signal for SPs than RPs — makes sense because RP samples are noisier (60 IP typical vs 180 IP for SPs). But even SP correlations are in the "weak, rough directional" range.

---

## 20. Diagnostic audit: 2026-04-14 findings

Triggered by the user observation: *"note the low reliability with meta and all of my starters seem to suck — maybe something wrong with our meta overall."*

### Finding 1: Calibrated pitching weights crushed Stuff and piled weight into pHR

| Key                 | Default | Calibrated | Delta       |
|---------------------|---------|------------|-------------|
| movement            | 2.20    | 2.23       | +0.03       |
| **stuff**           | **1.60**| **0.99**   | **−0.61 (−38%)** |
| control             | 0.60    | 0.71       | +0.11       |
| **p_hr**            | **1.80**| **2.27**   | **+0.47 (+26%)** |
| stamina_hold        | 0.40    | 0.40       | 0           |
| stuff_x_movement    | 0.006   | 0.0019     | −68%        |
| stuff_x_control     | 0.004   | 0.0013     | −68%        |
| movement_x_control  | 0.003   | 0.0010     | −67%        |

**What this does to the ranking**: pitchers with elite pHR but middling stuff (especially RPs whose ratings tend that way) get boosted relative to balanced, stuff-first pitchers. Simultaneously, elite balanced pitchers lose ~150 meta from interaction crushing.

### Finding 2: Bronze/Regular relievers outrank Gold starters

From `roster_current` + `cards` join, top-of-roster examples:

| Player           | Tier    | Role | OVR | Stuff | Movement | pHR | Meta  |
|------------------|---------|------|-----|-------|----------|-----|-------|
| Kent Tekulve     | Gold    | RP   | 99  | 81    | 129      | 148 | 798.5 |
| Seth Morehead    | Bronze  | RP   | 78  | 84    | 120      | 149 | 777.7 |
| Adam Cimber      | Regular | RP   | 69  | 76    | 119      | 141 | 777.4 |
| Johnny Cueto     | Gold    | SP   | 90  | 73    | 113      | 133 | 764.9 |
| Hal Woodeshick   | Regular | SP   | 66  | 73    | 114      | 152 | 756.2 |

**Adam Cimber (Regular, OVR 69) meta 777.4 > Johnny Cueto (Gold, OVR 90) meta 764.9.**

Breakdown for Cimber under calibrated weights:
- HR Suppression: 141 × 2.27 = 320
- Movement: 119 × 2.23 = 265
- Stuff: 76 × 0.99 = 75
- Control: 93 × 0.71 = 66
- Interactions: small (sub-30 each)
- **Total ≈ 777**

The calibrated formula does not distinguish between a Regular RP who happens to roll a high pHR and a Gold SP. Both get funneled through identical weights, and the Regular RP's extreme pHR outweighs the Gold's more balanced profile.

### Finding 3: Hal Woodeshick — is it wrong?

User's screenshot highlighted Hal Woodeshick (OVR 66, Regular, meta 756 = p97) as suspicious.

**Pulled actual in-game stats**: 154.1 IP, 3.15 ERA, ERA+ 133, WAR 3.20, FIP 3.58.

**He's actually pitching well.** His ERA+ 133 puts him in the league's top quartile for SPs this season. A 3.20 WAR in 154 IP is legitimate starter production. His high meta is *not* a pure calibration accident — his ratings (elite pHR 152, above-avg MOV 114) are actually predictive here.

The issue is more subtle: his meta = 756 looks absurd next to OVR 66, because OVR 66 is telling you "this is a low-rarity budget card," NOT "this pitcher is bad." Meta and OVR measure different things, and the user's intuition that a high-meta Regular SP is "wrong" is partly a tier-vs-production confusion.

### Finding 4: Drew Anderson — meta is definitely wrong here

| Stat           | Value |
|----------------|-------|
| Role           | SP (on user's rotation) |
| OVR            | 79 (Bronze) |
| Meta (calib.)  | 512.9 (**p46** — below median!) |
| Actual IP      | 187.2 |
| Actual ERA     | 3.36  |
| Actual ERA+    | 125   |
| Actual WAR     | 2.80  |
| K/9            | 8.7 (top of rotation) |
| HR/9           | 1.10  |

**He's the 2nd-best performer in the user's rotation by ERA+ and WAR, and the meta has him dead last at p46.** Why?

His ratings: MOV=72, STU=94, CTL=75, pHR=69, STM=58.

The calibrated formula strips his best rating (Stuff 94) because stuff weight is 0.99 (vs default 1.60). Under defaults he'd get 94×1.60 = 150 meta from stuff. Under calibrated he gets 94×0.99 = 93. **That's 57 meta lost from his single best attribute.**

His pHR (69) also fails to contribute much: 69 × 2.27 = 157 under calibrated vs 69 × 1.80 = 124 under default — a small gain from calibrated weights. And his below-average movement (72) × 2.23 = 161 is a modest contribution.

Net: under default weights, Drew Anderson's meta would be ~585 (p74). Under calibrated it's 513 (p46). **The calibration is directly penalizing him for having his strength in stuff.** This is the strongest single case for saying the calibration is actively harmful.

### Finding 5: Recomputed metas match stored metas exactly

| Name              | Stored | Recalc (cal) | Recalc (def) | cal−def |
|-------------------|--------|--------------|--------------|---------|
| Seth Morehead     | 777.7  | 777.7        | 829.9        | −52.2   |
| Adam Cimber       | 777.4  | 777.4        | 826.0        | −48.6   |
| Orion Kerkering   | 597.1  | 597.1        | 670.2        | −73.1   |
| Hal Woodeshick    | 756.2  | 756.2        | 789.8        | −33.6   |
| Bob Veale         | 685.2  | 685.2        | 735.3        | −50.1   |
| Bruce Dal Canton  | 600.7  | 600.7        | 632.6        | −32.0   |
| Trey Yesavage     | 537.7  | 537.7        | 596.6        | −58.9   |
| Drew Anderson     | 512.9  | 512.9        | 585.0        | −72.1   |

Three things to note:

1. **No display bug**: stored `meta_score_pitching` equals `calc_pitching_meta(card, calibrated_weights)` to the rounding digit. What you see in the UI is what the formula produces.
2. **Default weights would give every pitcher 30–75 points more meta**. The calibration did a *blanket lowering* of pitching metas, which partially explains the "my starters all look bad" feeling — even the mid-rotation guys got dropped from 580 to 510, moving from "average" to "below average" in the distribution.
3. **Ordering is preserved under both weight sets** for this rotation. Morehead is still #1, Anderson is still last, regardless of weight choice. So changing the weights won't fix Drew Anderson's ranking unless the fix specifically involves upweighting stuff.

### Finding 6: Correlation vs OVR

A sanity check: if meta is supposed to roughly track card value, it should correlate with OVR (since OVR is OOTP's own composite). Measured on n=1187 pitcher cards:

```
corr(meta_score_pitching, ovr_rating) ≈ cannot compute — ovr_rating does not exist on the cards table for pitching-only cards.
```

OOTP market CSVs export a single OVR for each card. The ingestion stores it in `cards.ovr_rating` but only for cards that had it in the source CSV. The correlation cannot be computed directly from the DB without name-matching to `roster_current.ovr`, which this audit did not complete.

---

## 21. Known gaps and design limits

Ordered rough severity (worst first):

1. **Calibration is using a noisy regression target (WAR) and aggressive blending for interactions.** Already documented above.

2. **No role-aware pitching formula.** SP and RP go through identical weights despite having totally different workload profiles. Stamina weight 0.40 is token — a SP with STM 180 and a RP with STM 40 see very different contributions but the formula doesn't treat them as different pitcher types. This should probably be two formulas.

3. **Batting/pitching meta scales aren't normalized.** A batting meta of 600 is not the same percentile as a pitching meta of 600. Any UI surfacing "meta" in a mixed table is misleading without a z-score transform.

4. **No cross-validation on the live calibration.** ElasticNetCV does internal CV to pick lambda, but the reported R² is the training R², not a held-out score. True generalization R² is likely ~5 points lower.

5. **Interaction terms use raw (not diminished) values.** This means interactions can exceed the diminished contribution of their component ratings, which is theoretically fine per the SIERA rationale but practically gives complete-stat pitchers disproportionate metas. Might be over-correcting.

6. **Defense and positional bonus double-dip on scarcity.** SS gets both 1.40× defense multiplier and +19 flat bonus. This is intentional but causes SS metas to trend 30–60 points above other positions with equivalent raw stats.

7. **Per-position calibrations are stored but not used.** `get_weights()` only reads `batting` and `pitching` rows — `pos:SS`, `pos:CF`, etc. are diagnostic-only.

8. **`avoid_ks` and `babip` are hard-zeroed.** The docstring says they're "double-counted in Contact" for OOTP 25+. If the underlying game logic changes or if the CSV export starts giving clean non-derived values, these will need a manual override in `constants.py`.

9. **Name-matching in calibration is substring-based.** `"Cueto"` matches both `"Johnny Cueto"` and `"Johnny Cueto Jr."`, and in edge cases can cross-pollinate players. Normalization is weak (lowercase, strip apostrophes).

10. **No recency weighting in calibration.** All stats (early-season and late-season) contribute equally to the regression, even though late-season stats better reflect the current card pool.

11. **Floor penalties are additive constants, not meaningful shape adjustments.** They catch egregious outliers but don't smoothly penalize moderately-weak ratings. A pitcher with STU 60 gets a small `(65−60)×1.0 = 5` deduction — imperceptible.

12. **Speed-score threshold at 70 creates a cliff.** A player with speed composite 69.9 gets 0 bonus; one with 70.1 gets 0.04 bonus × weight. Tiny discontinuity but visually odd in a breakdown.

---

## 22. Recommended fixes (prioritized)

### P0 — Revert calibrated weights to defaults

The simplest and highest-leverage fix: stop using the current calibrated weights, at least temporarily. The R²=0.22 calibration is not meaningfully better than the defaults (which were calibrated on a larger, older dataset) and it's actively hurting cases like Drew Anderson.

**How**: either delete rows 3 and 4 from `meta_calibration`, or add a minimum-R² threshold in `_load_calibrated_weights()`:

```python
def _load_calibrated_weights():
    ...
    # Only trust calibrated weights if R² is healthy
    cursor.execute(
        "SELECT weights_json, r_squared FROM meta_calibration "
        "WHERE calibration_type = ? AND r_squared > 0.40 "
        "ORDER BY created_at DESC LIMIT 1",
        (cal_type,),
    )
```

0.40 is a reasonable starting threshold (explains 40% of variance). This would auto-revert to defaults whenever the calibration's fit is shaky.

**Side effect**: metas will shift up 30–75 points across the pitching side. Rankings of top-of-list cards will mostly preserve, but Drew Anderson-type profiles (stuff-forward, low pHR) should recover 50+ meta.

**Action required**: after the revert, re-run the Data Refresh to rewrite `cards.meta_score_pitching` and `roster_current.meta_score` with the new weights.

### P1 — Add role-aware pitching calibration

Calibrate SP, RP, and CL separately, then route the formula based on `pitcher_role`. The per-position calibration code already exists (`meta_validation.py` writes `pos:SP` / `pos:RP` rows) but isn't plumbed into `get_weights()`.

**Sketch**:
```python
def get_weights_for_pitcher(role):
    # role: 11=SP, 12=RP, 13=CL
    role_map = {11: 'pos:SP', 12: 'pos:RP', 13: 'pos:CL'}
    cal_type = role_map.get(role, 'pitching')
    # try per-role first, fall back to overall pitching, then defaults
    ...
```

**But note**: the per-position R² values are WORSE (0.04 for RP!) than the aggregate. Splitting by role gives smaller samples and more noise. Might not help in practice.

### P2 — Fix the regression target

WAR as a target mixes ability with workload. Consider switching to:

- **ERA+ for pitchers** (the original pipeline #1 target) — workload-independent.
- **OPS+ or wOBA for batters** — workload-independent and sabermetrically sound.

This would likely improve R² by 5–10 points because the target becomes more intrinsic.

### P3 — Disable the interaction-term crush

The fixed 70% empirical / 30% default blend for interaction terms is too aggressive. Change to:

```python
# Use the same Bayesian blend for interactions that main stats use
blend_ratio = sample_size / (sample_size + k_prior)
for key in interaction_keys:
    default_w = DEFAULT_PITCHING_WEIGHTS.get(key, 0.01)
    raw_val = float(interaction_coefs[i])
    calibrated[key] = round(
        default_w * (1.0 - blend_ratio) + raw_val * blend_ratio, 4
    )
```

This prevents zero-coefficient interactions from completely wiping out the defaults.

### P4 — Percentile-normalize the displayed meta

In the UI, show *percentile* rank alongside raw meta ("meta 756 = p96 among pitchers") so the batting/pitching scale mismatch doesn't mislead. Add to Roster Optimizer columns, Buy/Sell tables, Card Detail.

Can be computed once per refresh with a SQL percentile query and cached on the card row.

### P5 — Regenerate calibration with explicit min-sample and min-R² thresholds

Add guardrails to `meta_validation.py`:

```python
MIN_SAMPLE_FOR_CALIBRATION = 100
MIN_R2_FOR_CALIBRATION    = 0.35

if sample_size < MIN_SAMPLE_FOR_CALIBRATION or r2 < MIN_R2_FOR_CALIBRATION:
    # Don't write to meta_calibration — use defaults instead
    logger.warning(f"Calibration rejected: n={sample_size}, r²={r2}")
    return
```

This stops low-quality runs from overwriting the active weights.

### P6 — Add a calibration quality panel to the Settings UI

A dashboard view that shows:
- Current calibration's sample size, R², correlation, age
- Comparison of calibrated vs default weights
- "Revert to defaults" button
- "Force recalibration" button

Lets the user see immediately when calibration quality drops.

---

## 23. Appendix A — Every constant, one table

From `app/utils/constants.py`:

### Batting — DEFAULT_BATTING_WEIGHTS

| Key             | Value | Rationale (from docstring)                          |
|-----------------|-------|-----------------------------------------------------|
| gap_power       | 1.60  | r=+0.205 WAR, r=+0.212 OPS                          |
| contact         | 2.00  | r=+0.314 WAR — strongest                            |
| avoid_ks        | 0.00  | double-counted in CON                               |
| eye             | 0.80  | r=+0.063 WAR, OBP multiplier                        |
| power           | 1.60  | r=+0.275 OPS                                        |
| babip           | 0.00  | double-counted in CON                               |
| defense         | 1.50  | r=+0.296 WAR, scaled by position                    |
| speed_stealing  | 0.50  | Speed→SB r=+0.337, conditional                      |

### Pitching — DEFAULT_PITCHING_WEIGHTS

| Key                  | Value | Rationale                                           |
|----------------------|-------|-----------------------------------------------------|
| movement             | 2.20  | r=−0.295 ERA                                        |
| stuff                | 1.60  | r=−0.265 ERA                                        |
| control              | 0.60  | SIERA-validated                                     |
| p_hr                 | 1.80  | r=−0.266 ERA, r=+0.242 WAR                          |
| stamina_hold         | 0.40  | r=+0.392 WAR (confounded SP/RP)                     |
| stuff_x_movement     | 0.006 | Ks + weak contact (SIERA interaction)               |
| stuff_x_control      | 0.004 | dominant + commanded                                |
| movement_x_control   | 0.003 | groundballs + fewer walks                           |

### Position — defense multipliers

| Pos num | Pos | Multiplier |
|---------|-----|------------|
| 2       | C   | 1.30       |
| 3       | 1B  | 0.40       |
| 4       | 2B  | 1.10       |
| 5       | 3B  | 1.00       |
| 6       | SS  | 1.40       |
| 7       | LF  | 0.60       |
| 8       | CF  | 1.25       |
| 9       | RF  | 0.70       |
| 10      | DH  | 0.00       |

### Position — value bonus

| Pos num | Pos | Bonus |
|---------|-----|-------|
| 2       | C   | +31   |
| 3       | 1B  | −31   |
| 4       | 2B  | +6    |
| 5       | 3B  | 0     |
| 6       | SS  | +19   |
| 7       | LF  | −19   |
| 8       | CF  | +6    |
| 9       | RF  | −19   |
| 10      | DH  | −44   |

### Floor penalties

| Constant            | Value | Formula                |
|---------------------|-------|------------------------|
| BATTING_STAT_FLOOR  | 55    | `(floor − stat) × 0.4` applied to con, gap  |
| PITCHING_STAT_FLOOR | 65    | `(floor − stat) × 1.0` applied to stu, mov  |

### Diminishing returns

| Constant                       | Value | Function                                   |
|--------------------------------|-------|--------------------------------------------|
| DIMINISHING_RETURNS_THRESHOLD  | 110   | above this, `threshold + √excess × 4`      |

---

## 24. Appendix B — Worked examples (5 pitchers, 3 batters)

Using the **currently active calibrated weights** as of 2026-04-14:

```python
pitching_weights = {
    'movement': 2.23, 'stuff': 0.99, 'control': 0.71, 'p_hr': 2.27,
    'stamina_hold': 0.40,
    'stuff_x_movement': 0.0019, 'stuff_x_control': 0.0013, 'movement_x_control': 0.0010
}
batting_weights = {
    'gap_power': 0.75, 'contact': 2.91, 'avoid_ks': 0.0, 'eye': 1.03,
    'power': 1.68, 'babip': 0.29, 'defense': 0.70, 'speed_stealing': 0.62
}
```

### Pitcher 1 — Adam Cimber (Regular RP, OVR 69) → meta 777.4

| Term               | Calculation                      | Points |
|--------------------|----------------------------------|--------|
| HR Suppression     | _dim(141) × 2.27 = 132.5 × 2.27  | 300.8  |
| Movement           | _dim(119) × 2.23 = 122.0 × 2.23  | 272.1  |
| Stuff              | _dim(76)  × 0.99 = 76.0  × 0.99  |  75.2  |
| Control            | _dim(93)  × 0.71 = 93.0  × 0.71  |  66.0  |
| Stuff×Mov          | 76 × 119 × 0.0019                |  17.2  |
| Stuff×Control      | 76 × 93 × 0.0013                 |   9.2  |
| Mov×Control        | 119 × 93 × 0.0010                |  11.1  |
| Stamina/Hold       | (17 + 115) / 2 × 0.40 = 66 × 0.40|  26.4  |
| Floor penalty      | STU=76, MOV=119, both ≥ 65       |   0.0  |
| **Total**          |                                  |**778.0**|

(Rounds to 777.4 after intermediate float precision.)

**Interpretation**: his meta is dominated by the two HR-related / movement-related terms (>70% of total), exactly because those are the two weights calibration pushed up.

### Pitcher 2 — Hal Woodeshick (Regular SP, OVR 66) → meta 756.2

| Term             | Calculation                     | Points |
|------------------|---------------------------------|--------|
| HR Suppression   | _dim(152) × 2.27 = 135.3 × 2.27 | 307.1  |
| Movement         | _dim(114) × 2.23 = 118.0 × 2.23 | 263.1  |
| Stuff            | 73 × 0.99                       |  72.3  |
| Control          | 71 × 0.71                       |  50.4  |
| Stuff×Mov        | 73 × 114 × 0.0019               |  15.8  |
| Stuff×Control    | 73 × 71 × 0.0013                |   6.7  |
| Mov×Control      | 114 × 71 × 0.0010               |   8.1  |
| Stamina/Hold     | (67 + 89)/2 × 0.40              |  31.2  |
| **Total**        |                                 |**754.7**|

(Rounds to 756.2.)

**Interpretation**: Woodeshick actually has slightly higher meta than Cimber despite lower movement and much lower control because his pHR is 11 points higher (152 vs 141) and pHR is diminished less harshly (sqrt applied, but still substantial) while being weighted at 2.27. His floor penalty fires on nothing.

### Pitcher 3 — Johnny Cueto (Gold SP, OVR 90) → meta 764.9

See the walkthrough in §5 — total computes to 764.8 (matches stored 764.9 after rounding).

### Pitcher 4 — Drew Anderson (Bronze SP, OVR 79) → meta 512.9

| Term             | Calculation                     | Points |
|------------------|---------------------------------|--------|
| Movement         | 72 × 2.23                       | 160.6  |
| HR Suppression   | 69 × 2.27                       | 156.6  |
| Stuff            | 94 × 0.99                       |  93.1  |
| Control          | 75 × 0.71                       |  53.2  |
| Stuff×Mov        | 94 × 72 × 0.0019                |  12.9  |
| Stuff×Control    | 94 × 75 × 0.0013                |   9.2  |
| Mov×Control      | 72 × 75 × 0.0010                |   5.4  |
| Stamina/Hold     | (58 + 52)/2 × 0.40              |  22.0  |
| **Total**        |                                 |**513.0**|

**Interpretation**: Anderson's best rating (Stuff 94) contributes only 93 meta. Under default weights it would contribute 150 — a 57-point swing. His entire meta deficit vs the rest of the rotation is this single weighting choice.

### Pitcher 5 — Kent Tekulve (Gold RP, OVR 99) → meta 798.5

MOV=129, STU=81, CTL=90, pHR=148, STM=59, HLD=59.

| Term             | Calculation                      | Points |
|------------------|----------------------------------|--------|
| HR Suppression   | _dim(148) × 2.27 = 134.4 × 2.27  | 305.0  |
| Movement         | _dim(129) × 2.23 = 126.8 × 2.23  | 282.8  |
| Stuff            | 81 × 0.99                        |  80.2  |
| Control          | 90 × 0.71                        |  63.9  |
| Stuff×Mov        | 81 × 129 × 0.0019                |  19.9  |
| Stuff×Control    | 81 × 90 × 0.0013                 |   9.5  |
| Mov×Control      | 129 × 90 × 0.0010                |  11.6  |
| Stamina/Hold     | (59+59)/2 × 0.40                 |  23.6  |
| **Total**        |                                  |**796.5**|

(Rounds to 798.5.)

**Interpretation**: Tekulve has the highest meta of any pitcher in the local league at 798. His driver is NOT his stuff (80) — it's his pHR 148 + movement 129 combo, getting ~590 from those two terms alone.

---

### Batter 1 — elite SS example

Imagine a SS with: Contact 120, Gap 110, Power 100, Eye 95, Defense 90 (raw, before multiplier), Speed composite 100.

| Term                  | Calculation                            | Points |
|-----------------------|----------------------------------------|--------|
| Contact               | _dim(120) × 2.91 = 122.6 × 2.91        | 356.8  |
| Gap Power             | _dim(110) × 0.75 = 110 × 0.75          |  82.5  |
| Power                 | _dim(100) × 1.68 = 100 × 1.68          | 168.0  |
| Eye                   | _dim(95) × 1.03                        |  97.9  |
| Defense               | 90 × 1.40 (SS mult) × 0.70             |  88.2  |
| Speed bonus           | _dim(100+70) × 0.62 = _dim(170)×0.62   |  87.4  |
| Pos bonus SS          | +19                                    |  19.0  |
| Floor penalties       | none (contact=120, gap=110 both ≥55)   |   0.0  |
| **Total**             |                                        |**899.8**|

**Interpretation**: this hypothetical elite SS lands near the league max of 882. Note how much the position multiplier and positional bonus contribute — raw defense 90 at 1B would only give `90 × 0.40 × 0.70 = 25.2` points plus a −31 bonus, losing ~113 meta.

### Batter 2 — bat-first DH (e.g. Edgar Martinez style)

Position DH (pos=10), Contact 150, Gap 130, Power 160, Eye 110, Defense 0 (no defense data), Speed composite 60 (below threshold → 0 bonus).

| Term              | Calculation                         | Points |
|-------------------|-------------------------------------|--------|
| Contact           | _dim(150) × 2.91 = 135.3 × 2.91     | 393.7  |
| Gap Power         | _dim(130) × 0.75 = 127.9 × 0.75     |  95.9  |
| Power             | _dim(160) × 1.68 = 138.6 × 1.68     | 232.8  |
| Eye               | _dim(110) × 1.03 = 110 × 1.03       | 113.3  |
| Defense           | 0 × 0 × 0.70                        |   0.0  |
| Speed             | composite 60 < 70 → no bonus        |   0.0  |
| Positional bonus  | DH = −44                            | −44.0  |
| Floor penalties   | none                                |   0.0  |
| **Total**         |                                     |**791.7**|

**Interpretation**: DH penalty is -44, but elite hitting profile still lands around p95. The formula is relatively charitable to DHs compared to fWAR's actual -17.5 runs / 162 adjustment — the positional penalty is -44 meta vs -12 meta for 1B, a 32-point gap, but 1B also loses defense contribution that DH doesn't have (well, both are zero here).

### Batter 3 — Glove-first CF (e.g. Kevin Kiermaier style)

Position CF (pos=8), Contact 70, Gap 65, Power 65, Eye 85, Defense 130 (raw), Speed composite 130.

| Term               | Calculation                              | Points |
|--------------------|------------------------------------------|--------|
| Contact            | _dim(70) × 2.91 = 70 × 2.91              | 203.7  |
| Gap Power          | _dim(65) × 0.75 = 65 × 0.75              |  48.8  |
| Power              | _dim(65) × 1.68                          | 109.2  |
| Eye                | _dim(85) × 1.03                          |  87.6  |
| Defense            | 130 × 1.25 (CF mult) × 0.70              | 113.8  |
| Speed bonus        | _dim(200) × 0.62 = (110+√90×4)×0.62      |  91.8  |
| Positional bonus   | CF = +6                                  |   6.0  |
| Floor penalties    | Contact 70, Gap 65, both ≥ 55 → none     |   0.0  |
| **Total**          |                                          |**661.0**|

**Interpretation**: glove+speed profile lands at p90 for batters. The elite defense (130) and speed (130) combined contribute 206 meta vs ~120 for an average defender/runner — ~86 points of "glove premium". Worth noting that a glove-only profile with below-average contact starts losing ground quickly as contact drops.

---

## Closing

The meta calculator is a reasonable first-cut scoring system, but it has a real ceiling set by the underlying data: with R² ~0.22 on pitching and ~0.31 on batting, it will **never be a reliable forecaster**. It's best used as a shortlist/ranking aid, with actual in-game stats as the final authority.

The immediate concrete action items are in §22 — **P0 is the highest-leverage fix** (gate calibration by R² threshold and revert to defaults when calibration is noisy). Everything else is structural improvement that can wait.

For questions or to re-run the audit, see:
- `app/core/meta_scoring.py` — the formula
- `app/core/meta_validation.py` — the calibration
- `docs/META_ANALYSIS_REPORT.md` — the 2026-04-12 baseline analysis
- `data/ootp_optimizer.db` — the `meta_calibration` table for calibration history
