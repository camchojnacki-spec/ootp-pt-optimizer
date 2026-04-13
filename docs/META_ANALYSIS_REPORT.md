# OOTP 27 Perfect Team Optimizer -- Technical Analysis Report

**System**: OOTP 27 PT Optimizer  
**Purpose**: Streamlit dashboard for buy/sell card recommendations, roster optimization, and meta-scoring for the OOTP Baseball 27 Perfect Team game mode  
**Team**: Toronto Dark Knights  
**Date**: 2026-04-12  
**Repository**: `C:\Users\Cameron\OneDrive\Documents\Claude\Projects\OOTPBUYNSELL`

---

## Table of Contents

1. [META FORMULA -- How Player Scores Are Calculated](#1-meta-formula----how-player-scores-are-calculated)
2. [AUTO-CALIBRATION SYSTEM](#2-auto-calibration-system)
3. [ROSTER OPTIMIZER](#3-roster-optimizer)
4. [BUY RECOMMENDATIONS](#4-buy-recommendations)
5. [DATA MODEL & INGESTION](#5-data-model--ingestion)
6. [KNOWN ISSUES, DESIGN DECISIONS, AND TRADE-OFFS](#6-known-issues-design-decisions-and-trade-offs)

---

## 1. META FORMULA -- How Player Scores Are Calculated

**Source file**: `app/core/meta_scoring.py`  
**Weight definitions**: `app/utils/constants.py`

The meta formula is the central scoring engine. Every card in the system receives a numeric meta score that represents its predicted in-game value. The formula has separate paths for batters and pitchers, with handedness splits, defense scoring, speed scoring, balance penalties, and OVR anchoring.

### 1.1 Diminishing Returns: `_diminished()`

All raw stat values pass through a diminishing-returns function before being weighted.

```python
DIMINISHING_RETURNS_THRESHOLD = 110

def _diminished(value, threshold=110):
    if value <= threshold:
        return value  # Linear 1:1
    excess = value - threshold
    return threshold + math.sqrt(excess) * 4
```

**Mechanics**:
- Stats at or below 110: counted at face value (linear).
- Stats above 110: the excess above 110 is sqrt-scaled, then multiplied by 4.
- Example: a raw 150 stat becomes `110 + sqrt(40) * 4 = 110 + 25.3 = ~135.3` effective.
- Purpose: prevents extreme single-stat spikes from dominating the composite score.

### 1.2 Batting Meta: `calc_batting_meta()`

The batting formula computes a weighted sum of diminished stat values, adds speed/defense, applies a balance penalty, then scales by OVR.

**Step 1 -- Core Weighted Sum**:
```
meta = (
    _diminished(gap)   * weight['gap_power']   +
    _diminished(con)   * weight['contact']      +
    _diminished(avk)   * weight['avoid_ks']     +
    _diminished(eye)   * weight['eye']          +
    _diminished(pwr)   * weight['power']        +
    _diminished(bab)   * weight['babip']        +
    defense            * weight['defense']
)
```

Note: `defense` is NOT passed through `_diminished()`. It is a pre-computed composite that already accounts for position multipliers (see Section 1.5).

**Step 2 -- Speed/Stealing Component** (conditional):
```python
spd_weight = weights.get('speed_stealing', 0.50)
if speed_score > 0 and spd_weight > 0:
    meta += _diminished(speed_score + 70) * spd_weight
```
The speed_score is an excess-above-70 value (see Section 1.6). When adding it to the formula, 70 is re-added to the baseline so the diminishing-returns function can operate on the full composite value. This means a player with a speed composite of exactly 70 contributes nothing. A player with composite 90 contributes `_diminished(90) * 0.50 = 90 * 0.50 = 45`.

**Step 3 -- Balance Penalty**:
```python
floor = BATTING_STAT_FLOOR  # 55
key_stats = [con, gap]
for stat in key_stats:
    if 0 < stat < floor:
        penalty = (floor - stat) * 0.4
        meta -= penalty
```
- Only Contact and Gap Power are subject to the penalty.
- Penalty rate: 0.4 per point below 55.
- A player with CON=40 gets penalized `(55-40) * 0.4 = 6.0` points.
- The `0 < stat` guard prevents penalizing stats that are simply missing/zero (e.g., pitchers).

**Step 4 -- OVR Anchoring**:
```python
ovr_weight = weights.get('ovr', 1.25)
if ovr > 0 and ovr_weight > 0:
    ovr_factor = (ovr / 80.0) ** (ovr_weight * 0.35)
    meta *= ovr_factor
```
- OVR 80 = neutral (factor = 1.0).
- OVR 100 with weight 1.25: `(100/80) ^ (1.25 * 0.35) = 1.25 ^ 0.4375 = ~1.098` (a ~10% boost).
- OVR 60 with weight 1.25: `(60/80) ^ 0.4375 = 0.75 ^ 0.4375 = ~0.879` (a ~12% penalty).
- This is a **multiplicative** factor, not additive -- it scales the entire weighted sum.
- Rationale: OVR has the strongest individual correlation with WAR (r=+0.529 for batters, r=+0.534 for pitchers) in league data.

**Final**: `round(meta, 2)`

### 1.3 Pitching Meta: `calc_pitching_meta()`

Similar structure to batting but with different stats and weights.

**Step 1 -- Core Weighted Sum**:
```
meta = (
    _diminished(mov)   * weight['movement']   +
    _diminished(stu)   * weight['stuff']       +
    _diminished(ctrl)  * weight['control']     +
    _diminished(phr)   * weight['p_hr']
)
```

**Step 2 -- Stamina/Hold Component**:
```python
sh_weight = weights.get('stamina_hold', 0.30)
if sh_weight > 0:
    sh_avg = 0
    sh_count = 0
    if stamina > 0: sh_avg += stamina; sh_count += 1
    if hold > 0: sh_avg += hold; sh_count += 1
    if sh_count > 0:
        meta += (sh_avg / sh_count) * sh_weight
```
- Averages whichever of stamina/hold are non-zero.
- The average is multiplied by the stamina_hold weight (default 0.30, config 0.40).
- Not passed through `_diminished()`.

**Step 3 -- Balance Penalty**:
```python
floor = 65  # PITCHING_STAT_FLOOR
key_stats = [stu, mov]  # Only STU and MOV, NOT Control
for stat in key_stats:
    if 0 < stat < floor:
        shortfall = floor - stat
        penalty = shortfall * 1.0  # Harsher: 1.0 per point
        meta -= penalty
```
- Pitching floor is 65 (higher than batting's 55).
- Only Stuff and Movement trigger the penalty. Control is deliberately excluded because data shows Control below 75 has near-zero impact on ERA (r=-0.002).
- Penalty rate is 1.0 per point (2.5x harsher than batting's 0.4 rate).

**Step 4 -- OVR Anchoring** (same formula as batting but different default weight):
```python
ovr_weight = weights.get('ovr', 1.50)  # Default 1.50 for pitchers (vs 1.25 batters)
ovr_factor = (ovr / 80.0) ** (ovr_weight * 0.35)
meta *= ovr_factor
```
- OVR 100 with weight 1.50: `(100/80) ^ (1.50 * 0.35) = 1.25 ^ 0.525 = ~1.123` (~12% boost).
- Pitching OVR weight is higher because data shows OVR->WAR r=+0.601 for pitchers (vs r=+0.529 for batters).

### 1.4 Split Formulas (vs RHP / vs LHP / vs RHB / vs LHB)

Four additional functions calculate split-specific meta scores:

**Batting splits** (`calc_batting_meta_vs_rhp`, `calc_batting_meta_vs_lhp`):
- Identical formula structure to `calc_batting_meta`.
- Input difference: CON, POW, and EYE are sourced from split columns (`con_vr`/`con_vl`, `pow_vr`/`pow_vl`, `eye_vr`/`eye_vl`) when available, falling back to overall ratings.
- GAP, AvK, BABIP do NOT have splits in OOTP roster CSVs -- always use overall values.
- Defense, speed, OVR anchoring: identical to overall formula.

**Pitching splits** (`calc_pitching_meta_vs_lhb`, `calc_pitching_meta_vs_rhb`):
- Identical formula structure to `calc_pitching_meta`.
- Input difference: STU is sourced from split columns (`stu_vl`/`stu_vr`) when available.
- MOV, Control, and pHR do NOT have splits in roster CSVs -- always use overall values.
- Stamina/Hold, OVR anchoring: identical to overall formula.

### 1.5 Defense Score: `calc_defense_score()`

**Position type detection**: Uses numeric position codes (2=C, 3=1B, ..., 9=RF, 10=DH).

**Raw defense calculation by position type**:
- **Outfield (pos 7,8,9)**: Average of `of_range`, `of_error`, `of_arm`
- **Catcher (pos 2)**: Average of `catcher_ability`, `catcher_frame`, `catcher_arm`
- **Infield (pos 3,4,5,6)**: Average of `infield_range`, `infield_error`, `infield_arm`
- Average calculation: `sum(vals) / count(nonzero vals)` -- only non-zero values count toward the denominator.

**Position multipliers** (`POSITION_DEFENSE_MULTIPLIERS`):
```
C  (2):  1.30  -- premium position, framing/arm hugely valuable
1B (3):  0.50  -- anyone can play 1B, low defensive spectrum value
2B (4):  1.10  -- middle infield, moderate value
3B (5):  1.00  -- moderate, hot corner needs arm/range
SS (6):  1.40  -- premium position, highest defensive spectrum
LF (7):  0.70  -- least demanding OF spot
CF (8):  1.25  -- premium OF, range matters most here
RF (9):  0.80  -- arm matters but less demanding than CF
DH (10): 0.00  -- no defense
```

**Final**: `raw_avg * position_multiplier`

The multipliers are a blend of league correlation data (e.g., CF def->WAR r=+0.257, SS r=+0.250) and traditional positional scarcity. When `apply_position_multiplier=False` is passed, returns the raw average without scaling.

### 1.6 Speed/Stealing Composite: `calc_speed_score()`

```python
composite = (stealing * 0.40 + baserunning * 0.35 + speed * 0.25)
```

**Component weights** (chosen based on predictive value):
- Stealing: 40% -- most directly predictive of stolen bases (r=+0.344)
- Baserunning: 35% -- extra bases taken on hits
- Speed: 25% -- raw tool, least directly productive

**Threshold behavior**:
```python
if composite < 70:
    return 0.0
return composite - 70  # Excess above average baseline
```
- Below composite 70: returns 0.0 (no speed bonus or penalty).
- Above 70: returns excess only.
- Slow players are NOT penalized because their batting stats already capture their lower output.
- Example: composite 85 returns 15. Composite 65 returns 0.

### 1.7 DEFAULT_BATTING_WEIGHTS (Every Weight and Source)

From `app/utils/constants.py` (also in `config.yaml`):

| Weight Key       | Default Value | Correlation Basis                                | Notes                                          |
|------------------|---------------|--------------------------------------------------|------------------------------------------------|
| `gap_power`      | 1.40          | WAR r=+0.205, OPS r=+0.212                      | Solid but not strongest                        |
| `contact`        | 1.80          | WAR r=+0.314                                    | Strongest individual rating                    |
| `avoid_ks`       | 0.00          | Double-counted in CON                            | OOTP25+ makes CON a derived stat from BABIP+AvK |
| `eye`            | 0.60          | WAR r=+0.063                                    | Weak predictor                                 |
| `power`          | 1.40          | OPS r=+0.275                                    | Strongest OPS predictor                        |
| `babip`          | 0.00          | Double-counted in CON                            | Same OOTP25+ derived stat reason               |
| `defense`        | 1.50          | WAR r=+0.296                                    | 2nd strongest; was massively underweighted     |
| `ovr`            | 1.25          | WAR r=+0.529                                    | Dominant but partially redundant with ratings  |
| `speed_stealing` | 0.50          | Speed->SB r=+0.337                              | Conditional value (amplifies OBP)              |

**Correlation source**: Analysis of 678 batters in league i76.

### 1.8 DEFAULT_PITCHING_WEIGHTS

| Weight Key       | Default Value | Correlation Basis                                | Notes                                          |
|------------------|---------------|--------------------------------------------------|------------------------------------------------|
| `movement`       | 2.40          | ERA r=-0.295, WAR r=+0.228                      | Strongest ERA predictor                        |
| `stuff`          | 1.40          | ERA r=-0.265                                    | 2nd strongest for ERA                          |
| `control`        | 0.20          | ERA r=-0.002                                    | Near zero! Was massively overweighted at 1.8   |
| `p_hr`           | 1.80          | ERA r=-0.266, WAR r=+0.242                      | Was underweighted at 0.6                       |
| `ovr`            | 1.50          | WAR r=+0.534                                    | Dominant                                       |
| `stamina_hold`   | 0.40          | WAR r=+0.392                                    | Confounded (starters vs relievers)             |

**Correlation source**: Analysis of 633 pitchers in league i76.

### 1.9 Stat Floor Values

```python
BATTING_STAT_FLOOR  = 55   # Contact and Gap below this get penalized
PITCHING_STAT_FLOOR = 65   # Stuff and Movement below this get penalized
```

---

## 2. AUTO-CALIBRATION SYSTEM

**Source file**: `app/core/meta_validation.py`

The auto-calibration system adjusts meta formula weights based on actual in-game performance data. It uses individual stat-to-WAR correlations to empirically determine how much each rating matters, then blends the results with default weights.

### 2.1 `auto_calibrate_weights()` -- End-to-End Flow

**Overall strategy**: Robust individual-correlation-based weighting (not OLS regression). Computes the Pearson correlation of each individual card rating with WAR and OPS, then converts those correlations into proportional weights.

#### 2.1.1 Batting Calibration

**Data query** (primary -- card_id join):
```sql
SELECT c.card_title, c.contact, c.gap_power, c.power, c.eye,
       c.avoid_ks, c.babip, c.card_value,
       c.infield_range, c.infield_error, c.infield_arm,
       c.of_range, c.of_error, c.of_arm,
       c.catcher_ability, c.catcher_frame, c.catcher_arm,
       c.position, c.speed, c.stealing, c.baserunning,
       bs.war, bs.ops, bs.pa
FROM cards c
INNER JOIN batting_stats bs ON bs.card_id = c.card_id
INNER JOIN (
    SELECT card_id, MAX(snapshot_date) as max_date
    FROM batting_stats WHERE card_id IS NOT NULL
    GROUP BY card_id
) latest ON bs.card_id = latest.card_id AND bs.snapshot_date = latest.max_date
WHERE c.position != 1 AND bs.pa >= 100
```

**Sample size requirements**:
- Batters: PA >= 100 minimum per player, need >= 15 total matched players.
- Falls back to name-LIKE matching if card_id join returns < 15 results:
  ```sql
  INNER JOIN batting_stats bs ON c.card_title LIKE '%' || bs.player_name || '%'
  ```

**Correlation computation**:
- For each stat key in `[contact, gap_power, power, eye, avoid_ks, babip, defense]`:
  - Compute Pearson correlation with WAR values.
  - Compute Pearson correlation with OPS values.
  - Combined correlation: `(corr_war + corr_ops) / 2.0`
- Defense is computed on-the-fly via `calc_defense_score(row)` for each matched player.

**Proportional weight conversion**:
```python
positive_corrs = {k: max(0.0, v) for k, v in combined_corr.items()}
corr_sum = sum(positive_corrs.values())

default_total = sum(DEFAULT_BATTING_WEIGHTS[k] for k in all_bat_keys)
empirical = {}
for key in all_bat_keys:
    empirical[key] = (positive_corrs[key] / corr_sum) * default_total
```
- Negative correlations are zeroed out (they would get zero weight).
- The empirical weights are scaled so their sum matches the sum of default weights (preserving overall magnitude).

**Blending with defaults**:
```python
bat_confidence = min(1.0, bat_sample_size / 100.0)
blend = bat_confidence * 0.6  # Max 60% empirical

for key in all_bat_keys:
    new_w = default_w * (1.0 - blend) + empirical_w * blend
```
- Confidence: `sample_size / 100`, capped at 1.0.
- At 100+ samples and full confidence: 60% empirical, 40% default.
- At 50 samples: `(50/100) * 0.6 = 0.30` blend -- 30% empirical, 70% default.
- At 15 samples (minimum): `(15/100) * 0.6 = 0.09` blend -- 9% empirical, 91% default.

#### 2.1.2 Pitching Calibration

**Data query** (primary -- card_id join):
```sql
SELECT c.card_title, c.stuff, c.movement, c.control, c.p_hr,
       c.card_value, c.stamina, c.hold,
       ps.war, ps.era, ps.ip
FROM cards c
INNER JOIN pitching_stats ps ON ps.card_id = c.card_id
INNER JOIN (
    SELECT card_id, MAX(snapshot_date) as max_date
    FROM pitching_stats WHERE card_id IS NOT NULL GROUP BY card_id
) latest ON ps.card_id = latest.card_id AND ps.snapshot_date = latest.max_date
WHERE c.pitcher_role IS NOT NULL AND ps.ip >= 30
```

**Sample size requirements**:
- Pitchers: IP >= 30 minimum per player, need >= 10 total matched players.
- Fallback: name-LIKE matching if card_id join returns < 10 results.

**Correlation computation**:
- ERA is negated (`-ERA`) so that "higher = better" aligns with the correlation direction.
- For each stat: compute Pearson with WAR and with negated ERA.
- Combined: `(corr_war + corr_neg_era) / 2.0`

**Blending**: Same formula as batting, with `pitch_confidence = min(1.0, pitch_sample_size / 100.0)`.

### 2.2 `meta_calibration` Table Structure

```sql
CREATE TABLE IF NOT EXISTS meta_calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calibration_type TEXT NOT NULL,     -- 'batting' or 'pitching'
    weights_json TEXT NOT NULL,          -- JSON dict of weight name -> value
    r_squared REAL,                      -- R-squared of calibrated predictions vs actual WAR
    correlation REAL,                    -- Pearson correlation of predictions vs actual WAR
    sample_size INTEGER,                 -- Number of players in calibration set
    confidence REAL,                     -- 0.0-1.0 confidence factor
    changes_json TEXT,                   -- JSON list of {stat, old_weight, new_weight, reason}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

Both batting and pitching calibrations are stored as separate rows. The most recent row per `calibration_type` is used.

### 2.3 Weight Loading Cascade: `get_weights()` and `get_weights_with_source()`

Three-tier priority system:

1. **Calibrated** (DB): Queries `meta_calibration` table for most recent `weights_json` per type. Both batting AND pitching must be present for this tier to activate.
2. **Config** (YAML): Falls back to `config.yaml` -> `batting_weights` / `pitching_weights`.
3. **Defaults** (constants.py): Falls back to `DEFAULT_BATTING_WEIGHTS` / `DEFAULT_PITCHING_WEIGHTS`.

`get_weights_with_source()` returns a third value: `'calibrated'`, `'config'`, or `'default'` indicating which tier was used.

### 2.4 Validation: `validate_meta_vs_performance()`

Compares roster meta scores against actual in-game performance to measure formula accuracy.

**Batter matching**: Joins `batting_stats` with `roster_current` by normalized name (case-insensitive, stripped dots and apostrophes). Exact match first, then substring fallback.

**Performance rating for batters**: `OPS * 1000` (so .800 OPS = 800 performance rating).

**Performance rating for pitchers**: `min(ERA+, 200) * 5` (ERA+ 100 = 500, ERA+ 150 = 750; capped at 200 to prevent extreme reliever values from skewing).

**Outputs**: Pearson correlation, Spearman rank correlation, over/underperformer lists (threshold: >20 point gap between meta and performance).

### 2.5 Calibration Results Referenced in Codebase

The commit history and code comments reference these correlation improvements:
- Batting: 0.330 -> 0.369 (after calibration)
- Pitching: 0.083 -> 0.298 (after calibration)

---

## 3. ROSTER OPTIMIZER

**Source file**: `app/pages/4_Roster_Optimizer.py`

The Roster Optimizer is the primary lineup management page. It displays the current roster as a card-based lineup view with upgrade recommendations from both owned cards and the market.

### 3.1 Roster Data Loading

**SQL query**:
```sql
SELECT r.player_name, r.position, r.lineup_role, r.ovr, r.meta_score,
       r.meta_vs_rhp, r.meta_vs_lhp, r.bats as roster_bats, c.bats
FROM roster r
LEFT JOIN cards c ON c.card_title LIKE '%' || r.player_name || '%'
    AND c.owned = 1
WHERE r.lineup_role != 'league'
  AND DATE(r.snapshot_date) = (
      SELECT MAX(DATE(snapshot_date)) FROM roster WHERE lineup_role != 'league'
  )
GROUP BY r.id
ORDER BY r.position, r.meta_score DESC
```

- Filters out `'league'` entries (league-wide rating imports).
- Joins to `cards` table for the `bats` handedness field.
- Uses most recent snapshot date.
- `GROUP BY r.id` deduplicates when multiple cards match.

**Active roles**: `{'starter', 'rotation', 'closer', 'bullpen'}`

### 3.2 Data Dictionaries

Three dictionaries are built from the roster query:

- **`starters`**: Best active player per position. Keyed by position string. For single-slot positions, keeps only the highest-meta active player.
- **`active_by_pos`**: All active players per position (list). Used for multi-slot positions like SP (5 slots), RP (7 slots).
- **`all_by_pos`**: Every rostered player per position, including bench/reserve. Used for exclude lists and mismatch detection.

### 3.3 Performance Data Loading

Two dictionaries loaded for in-game context:
- **`_perf_bat`**: Batting stats for players with PA >= 50. Computes `war600 = war * 600.0 / pa`.
- **`_perf_pit`**: Pitching stats for players with IP >= 10. Computes `war200 = war * 200.0 / ip`.

### 3.4 DH Inference Logic

The DH slot is not explicitly assigned in the roster data. Instead, it is **inferred**:

```python
if pos == 'DH':
    # Collect all starters not already shown at a field position.
    _field_shown = {e['pos'] for e in upgrade_plan if e['pos'] in bat_field_positions}
    _field_names = {e['current_name'] for e in upgrade_plan if e['pos'] in bat_field_positions}
    
    dh_candidates = []
    for fpos in bat_field_positions:
        for p in active_by_pos.get(fpos, []):
            if p['player_name'] not in _field_names:
                dh_candidates.append(p)
    dh_candidates.sort(key=lambda p: p['meta_score'] or 0, reverse=True)
```

Logic: For each field position, the best player is the fielder (already shown). Any remaining starters at that position who were NOT shown as the fielder become DH candidates. The best DH candidate (by meta) gets the DH slot.

If no candidates exist, the slot is marked `(empty)` and only upgrade suggestions are shown.

DH upgrades search ALL batting positions (`pitcher_role IS NULL`) since any non-pitcher can DH.

### 3.5 Position Processing Loop

The main loop iterates over `show_positions` and handles each type differently:

**SP Handling**:
- Takes up to 5 active rotation pitchers from `active_by_pos.get('SP', [])`.
- Processes WEAKEST first (sorted ascending by meta) so the best free upgrades go to the worst slots.
- Labels: `SP1`, `SP2`, `SP3`, `SP4`, `SP5`.
- Tracks used names (both player names and card titles) to prevent the same card being recommended for multiple slots.

**RP Handling**:
- Takes up to 7 active bullpen pitchers from `active_by_pos.get('RP', [])`.
- Also processes WEAKEST first.
- Slot names: `["SU1", "SU2", "MID1", "MID2", "LNG1", "LNG2", "MOP"]`.
- Falls back to `RP{i+1}` if index exceeds slot_names length.

**CL Handling**: Treated as a single standard position (same as batting positions).

**Batting Position Handling**: Single slot per position. Best active player by meta score starts. Upgrade searches use both the position-specific owned search and market search.

### 3.6 `find_roster_bench_upgrades()`

Searches for bench/reserve players at a specific roster position who beat the current starter's meta by at least `min_improvement` (default 20).

**Key behaviors**:
- Only considers players whose `lineup_role` is NOT in `{'starter', 'rotation', 'closer', 'bullpen'}`.
- Checks `exclude_names` against both exact match and substring match (handles both player names and card titles in the list).
- **Performance gate**: If the current starter has performance data with `WAR/600 >= 1.5` (or `WAR/200` for pitchers) AND the bench candidate has NO performance data, the candidate is skipped. This prevents benching a player who is producing in-game for someone unproven.
- Sorted by meta descending.

### 3.7 `find_owned_upgrades()`

Searches the `cards` table for owned cards that beat the current starter.

**Flow**:
1. First calls `find_roster_bench_upgrades()` for roster-level bench players.
2. Then queries the `cards` table by the card's native position (`position_name` or `pitcher_role_name`).
3. **DH special case**: When `pos_value == 'DH'`, queries ALL non-pitcher cards (`pitcher_role IS NULL`) instead of filtering by position.
4. Joins `my_collection` for status and `roster` for current role.
5. Filters out cards whose titles appear in `exclude_names`.
6. Deduplicates against bench names already captured.
7. Assigns action labels: `'Activate'`, `'Promote'`, `'Move Up'`, or `'Swap In'` based on collection status and roster role.
8. Merges bench + cards-table results, sorted by meta, limited to `limit` (default 5).

**SQL for standard position**:
```sql
SELECT c.card_id, c.card_title, c.tier_name, c.card_value,
       c.{meta_col} as meta_score, c.last_10_price,
       mc.status as collection_status, r.lineup_role as roster_role
FROM cards c
LEFT JOIN my_collection mc ON mc.card_id = c.card_id
LEFT JOIN roster r ON c.card_title LIKE '%' || r.player_name || '%'
    AND r.position = c.{pos_col}
WHERE c.{pos_col} = ? AND c.owned = 1 AND c.{meta_col} > ?
GROUP BY c.card_id ORDER BY c.{meta_col} DESC LIMIT ?
```

### 3.8 `find_market_upgrades()`

Searches unowned cards on the market within the price budget.

```sql
SELECT card_id, card_title, tier_name, card_value,
       {meta_col} as meta_score, last_10_price
FROM cards
WHERE {pos_col} = ? AND owned = 0 AND last_10_price > 0
    AND last_10_price <= ? AND {meta_col} > ?
ORDER BY {meta_col} DESC LIMIT ?
```

**DH special case**: Same as owned -- queries `pitcher_role IS NULL` for all non-pitchers.

Excludes cards whose `card_id` is in `exclude_ids` set.

### 3.9 Performance Gate: WAR/600 >= 1.5

The performance gate appears in two places:

1. **`find_roster_bench_upgrades()`**: If the starter has performance data and `WAR/600 >= 1.5` (decent+ production) while the bench candidate has NO performance data, skip the recommendation.
2. **Roster mismatch detection**: Same logic -- if the starter is producing real WAR and the "better" bench player has no stats, don't flag it as a mismatch.

Rationale: A card with high meta but no game data might just be new or unlucky. A starter producing at 1.5+ WAR/600 is proving their value regardless of meta predictions.

### 3.10 `_all_active_names` -- Cross-Position Exclusion

```python
_all_active_names = set()
for _pos_key, _players in active_by_pos.items():
    for _p in _players:
        _all_active_names.add(_p['player_name'])
```

This set is passed as `exclude_names` to upgrade finders. It prevents recommending an active player at one position as an upgrade for another slot. For example, the CL should not be suggested as a MOP upgrade.

### 3.11 Roster Mismatch Detection (`roster_fixes`)

```python
for pos in bat_field_positions + ['CL']:
    pp = all_by_pos.get(pos, [])
    if len(pp) < 2: continue
    best = pp[0]  # Highest meta at this position
    if best.lineup_role not in ACTIVE_ROLES:
        for p in pp:
            if p.lineup_role in ACTIVE_ROLES:
                d = best.meta_score - p.meta_score
                if d >= min_improvement:
                    # Performance gate check...
                    roster_fixes.append(...)
```

Flags cases where the bench player at a position has higher meta than the active starter, subject to the performance gate.

### 3.12 `build_chain_rows()` -- Table Builder

**Position matching logic**:
```python
pos = u['pos'].rstrip(" ...")
if pos not in positions_list and not any(
    pos.startswith(p) and len(pos) > len(p) and pos[len(p)].isdigit()
    for p in positions_list
):
    continue
```
- First checks exact position match (e.g., `'SS'` in `['SS']`).
- Then checks prefix-with-digit: `'SP1'` matches `'SP'` because `'SP1'.startswith('SP')` and `'SP1'[2]` is `'1'` (a digit).
- The digit check (`pos[len(p)].isdigit()`) is critical to prevent `CL` from matching `C`. `'CL'.startswith('C')` is true, but `'CL'[1]` is `'L'` (not a digit), so it does NOT match.

**Table columns** (`CHAIN_COL_CONFIG`):
| Column   | Type             | Width  | Description                                                      |
|----------|------------------|--------|------------------------------------------------------------------|
| Pos      | TextColumn       | small  | Position label, with platoon warning icon if applicable          |
| Current  | TextColumn       | medium | "PlayerName (OVR Hand)" format                                  |
| Meta     | ProgressColumn   | small  | Min 300, max 800, integer format                                 |
| Perf     | TextColumn       | small  | In-game stats: ERA/WAR for pitchers, OPS/WAR600 for batters     |
| Action   | TextColumn       | medium | What to do: Optimal / Promote X / Buy X 2,350PP                 |
| Why      | TextColumn       | medium | AI reasoning or meta explanation                                 |

**Perf column emoji codes**:
- Clover leaf = outperforming (ERA - FIP < -0.5, meaning ERA is better than FIP predicts, "lucky")
- Warning sign = underperforming (ERA - FIP > 0.5, meaning ERA is worse than FIP predicts, "unlucky")

### 3.13 Team Grading

Computed from average meta across all active starters:

| Avg Meta | Grade |
|----------|-------|
| >= 700   | A+    |
| >= 650   | A     |
| >= 600   | A-    |
| >= 560   | B+    |
| >= 520   | B     |
| >= 480   | B-    |
| >= 440   | C+    |
| >= 400   | C     |
| < 400    | D     |

---

## 4. BUY RECOMMENDATIONS

**Source file**: `app/pages/1_Buy_Recommendations.py`

The Investment Advisor page helps decide how to spend PP (Perfect Points, the in-game currency).

### 4.1 Position Gaps Analysis: `_get_position_gaps()`

For each position (8 batting + 3 pitching):
1. Gets current starter's meta from `roster_current`.
2. Queries the market ceiling: `MAX(meta_score_batting)` from `cards` where `position_name = pos AND last_10_price > 0`.
3. Checks for bench upgrades: owned cards with meta > current + 5.
4. Computes `gap = market_ceiling - current_meta`.
5. Returns a sorted list (descending by gap) including current name, meta, performance-adjusted meta, WAR, OPS/ERA.

### 4.2 Upgrade Candidates: `_get_upgrade_candidates()`

For a given position and budget:
- Queries unowned cards where `last_10_price <= budget AND meta > current + 5`.
- Computes:
  - `delta`: raw meta improvement over current starter.
  - `value_ratio`: `meta * meta / price` -- quadratic scaling rewards high-meta cards.
  - `efficiency`: `delta / (price / 100)` -- meta gain per 100 PP spent.
- Enriches with performance data by fuzzy-matching the card's last name against stats tables.

### 4.3 Performance Outliers: `_get_performance_outliers()`

**Pitching outliers** (outperforming their meta):
```sql
SELECT p.player_name, p.era, p.fip, p.war, p.ip, p.era_plus, p.whip,
       r.meta_score, r.position, r.lineup_role
FROM pitching_stats p
JOIN roster r ON r.player_name = p.player_name
    AND DATE(r.snapshot_date) = (SELECT MAX(DATE(snapshot_date)) FROM roster WHERE lineup_role != 'league')
WHERE p.ip > 40 AND p.war > 1.5
    AND DATE(p.snapshot_date) = (SELECT MAX(DATE(snapshot_date)) FROM pitching_stats)
GROUP BY p.player_name
ORDER BY p.war DESC
```

**Batting outliers**:
```sql
WHERE b.pa > 80 AND b.war > 1.0
```

**Deduplication**: Both queries use `GROUP BY p.player_name` / `GROUP BY b.player_name` to ensure one row per player even if multiple stat snapshots exist.

### 4.4 Performance-Adjusted Meta

Blends the card's meta score with actual in-game production:

**For pitchers** (IP > 50):
```python
war_per_200 = war * 200 / ip
perf_score = 400 + war_per_200 * 40  # Scale WAR to meta range
perf_meta = meta_score * 0.6 + perf_score * 0.4
```

**For batters** (PA > 100):
```python
war_per_600 = war * 600 / pa
perf_score = 400 + war_per_600 * 35
perf_meta = meta_score * 0.6 + perf_score * 0.4
```

**Formula**: 60% meta score + 40% WAR-based performance score. The performance score maps WAR/rate to the same ~400-700 range as meta scores via the `400 + war_rate * multiplier` formula.

### 4.5 Investment Scenarios: `_build_budget_scenarios()`

Three scenarios are computed for the player's total PP budget:

**Scenario 1 -- "One Big Buy"**:
- Searches every gap position for the single best card within the full budget.
- Picks the card with the highest `delta` (meta improvement over current).
- Skips positions that have free owned upgrades.

**Scenario 2 -- "Two Balanced Buys"**:
- Budget split: `total_budget // 2` per card.
- Searches the two highest-gap positions for the best card within half budget each.
- Only includes cards with `delta > 5`.
- Uses position deduplication to ensure two different positions.

**Scenario 3 -- "Max Efficiency"**:
- Collects ALL upgrade candidates across ALL positions (up to 3 per position).
- Sorts by `efficiency = delta / (price / 100)`.
- Greedily packs cards: picks best-efficiency card, deducts from remaining budget, moves to next (different position), until 3 cards or budget exhausted.

Each scenario includes: label, description, card list, total cost, total meta delta, remaining PP, positions improved.

### 4.6 Free Upgrades Display

Prominently shown at the top of the Investment Plan tab:
- Identifies positions where `has_owned_upgrade == True`.
- Shows a table: Pos | Current | Meta | Promote | New Meta | +Gain.
- Shows total meta gain from free moves.
- Instructional text: "Do these first!"

---

## 5. DATA MODEL & INGESTION

### 5.1 Full Database Schema

**Source file**: `app/core/database.py`

The database is SQLite, stored at `data/ootp_optimizer.db` (configurable via `config.yaml`).

#### 5.1.1 `cards` Table
Primary table for all card data from the market CSV.

| Column             | Type      | Notes                                        |
|--------------------|-----------|----------------------------------------------|
| card_id            | INTEGER   | PRIMARY KEY                                  |
| card_title         | TEXT      | Full card title (e.g., "Snapshot CF Mike Trout LAA 2026") |
| first_name         | TEXT      |                                              |
| last_name          | TEXT      |                                              |
| nickname           | TEXT      |                                              |
| position           | INTEGER   | Numeric position code (1-10)                 |
| position_name      | TEXT      | String position (C, 1B, ..., RF, DH)        |
| pitcher_role       | INTEGER   | 11=SP, 12=RP, 13=CL                         |
| pitcher_role_name  | TEXT      | SP, RP, CL                                  |
| bats / throws      | TEXT      |                                              |
| age                | INTEGER   | Computed from YearOB                         |
| team / franchise   | TEXT      |                                              |
| card_type          | INTEGER   |                                              |
| card_sub_type      | TEXT      |                                              |
| card_badge         | TEXT      |                                              |
| card_series        | TEXT      |                                              |
| card_value         | INTEGER   | OVR rating                                   |
| year / peak        | INTEGER/TEXT |                                            |
| tier / tier_name   | INTEGER/TEXT | 1=Regular, 2=Bronze, ..., 6=Perfect       |
| nation             | TEXT      |                                              |
| contact ... babip  | INTEGER   | Overall batting ratings (6 stats)            |
| contact_vl ... babip_vl | INTEGER | vs LHP batting splits (6 stats)         |
| contact_vr ... babip_vr | INTEGER | vs RHP batting splits (6 stats)         |
| speed, steal_rate, stealing, baserunning | INTEGER | Speed tools              |
| sac_bunt, bunt_for_hit | INTEGER |                                          |
| stuff ... p_babip  | INTEGER   | Overall pitching ratings (5 stats)           |
| stuff_vl ... p_babip_vl | INTEGER | vs LHB pitching splits (5 stats)        |
| stuff_vr ... p_babip_vr | INTEGER | vs RHB pitching splits (5 stats)        |
| stamina, hold      | INTEGER   |                                              |
| velocity           | TEXT      |                                              |
| infield_range/error/arm | INTEGER | Infield defense (3 stats)              |
| dp                 | INTEGER   | Double play ability                          |
| catcher_ability/frame/arm | INTEGER | Catcher defense (3 stats)            |
| of_range/error/arm | INTEGER   | Outfield defense (3 stats)                   |
| pos_rating_p ... pos_rating_rf | INTEGER | Position eligibility ratings (9 positions) |
| mission_value      | INTEGER   |                                              |
| card_limit         | INTEGER   |                                              |
| bref_id            | TEXT      | Baseball Reference ID                        |
| packs              | TEXT      |                                              |
| owned              | INTEGER   | 0 or 1                                       |
| meta_score_batting | REAL      | Computed batting meta                        |
| meta_score_pitching | REAL     | Computed pitching meta                       |
| buy_order_high     | INTEGER   |                                              |
| sell_order_low     | INTEGER   |                                              |
| last_10_price      | INTEGER   | Most recent avg price of last 10 trades      |
| last_10_variance   | INTEGER   |                                              |
| first_seen_at      | TIMESTAMP |                                              |
| last_updated_at    | TIMESTAMP |                                              |

#### 5.1.2 `price_snapshots` Table

```sql
CREATE TABLE price_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id INTEGER NOT NULL,
    snapshot_date TEXT NOT NULL,
    buy_order_high INTEGER,
    sell_order_low INTEGER,
    last_10_price INTEGER,
    last_10_variance INTEGER,
    FOREIGN KEY (card_id) REFERENCES cards(card_id)
);
CREATE UNIQUE INDEX idx_price_card_date ON price_snapshots(card_id, snapshot_date);
```

**Dedup mechanism**: The UNIQUE index on `(card_id, snapshot_date)` combined with `INSERT OR IGNORE` ensures only one price snapshot per card per day. The snapshot_date is truncated to date-only (no time component): `datetime.now().strftime("%Y-%m-%d")`.

#### 5.1.3 `roster` Table

```sql
CREATE TABLE roster (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT NOT NULL,
    position TEXT,
    lineup_role TEXT,       -- 'starter', 'rotation', 'closer', 'bullpen',
                            -- 'bench', 'reserve', 'league'
    ovr INTEGER,
    meta_score REAL,
    snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Migrated columns:
    meta_vs_rhp REAL,
    meta_vs_lhp REAL,
    con_vl INTEGER, pow_vl INTEGER, eye_vl INTEGER,
    con_vr INTEGER, pow_vr INTEGER, eye_vr INTEGER,
    stu_vl INTEGER, stu_vr INTEGER,
    card_title TEXT,
    card_id INTEGER,
    bats TEXT
);
```

#### 5.1.4 `my_collection` Table

Stores cards the player owns with their ratings and meta scores.

#### 5.1.5 `batting_stats` Table

Full set of standard batting statistics per player per snapshot date:
- Counting stats: G, PA, AB, H, 2B, 3B, HR, RBI, R, BB, IBB, HBP, K, GIDP, SB, CS
- Rate stats: AVG, OBP, SLG, ISO, OPS, OPS+, BABIP, WAR
- card_id, snapshot_date

Indexes: `player_name`, `snapshot_date`.

#### 5.1.6 `pitching_stats` Table

Full set of standard pitching statistics:
- Counting: G, GS, W, L, SV, HLD, IP, HA, HR, R, ER, BB, K, HBP
- Rate: ERA, AVG against, BABIP, WHIP, HR/9, BB/9, K/9, K/BB, ERA+, FIP, WAR

#### 5.1.7 Other Tables

- **`batting_stats_adv`**: Advanced batting metrics (wOBA, BB%, K%, RC/27, WPA, etc.).
- **`pitching_stats_adv`**: Advanced pitching metrics (SIERA, QS%, WPA, pLi, etc.).
- **`team_lineup`**: Stores lineup exports by type (`vs_rhp`, `vs_lhp`, `overview`, `pitching`).
- **`fielding_stats`**: In-game fielding performance (PCT, RNG, ZR, EFF, FRM, ARM, CERA).
- **`pitch_ratings`**: Individual pitch type ratings (FB, CH, CB, SL, SI, etc.).
- **`recommendations`**: Generated buy/sell recommendations.
- **`ingestion_log`**: Log of every CSV import (file_type, file_name, row_count, timestamp).
- **`ai_insights`**: Cached AI-generated analysis.
- **`price_alerts`**: User-configured price triggers (alert_type: 'below'/'above', target_price).
- **`meta_calibration`**: Calibrated weight storage (see Section 2.2).

#### 5.1.8 Views

```sql
roster_current:     Latest non-league roster snapshot
roster_league:      Latest league-wide roster snapshot
collection_current: Latest collection snapshot
lineup_current:     Latest team lineup snapshot
```

### 5.2 Ingestion Pipeline

**Source file**: `app/core/ingestion.py`

#### 5.2.1 File Type Identification

Uses `FILE_PATTERNS` dictionary in `constants.py`:
```python
FILE_PATTERNS = {
    "market": "pt_card_list",
    "roster_batting": "rosters_-_player_list_batting_ratings",
    "roster_pitching": "rosters_-_player_list_pitching_ratings",
    "collection_batting": "collection_-_manage_cards_..._batting_ratings",
    "collection_pitching": "collection_-_manage_cards_..._pitching_ratings",
    "stats_batting": "sortable_stats_batting_stats",
    "stats_pitching": "sortable_stats_pitching_stats",
    "roster_batting_stats": "player_list_batting_stats",
    "roster_pitching_stats": "player_list_pitching_stats",
    "stats_batting_ratings": "sortable_stats_batting_ratings",
    "stats_pitching_ratings": "sortable_stats_pitching_ratings",
    "lineup_vs_rhp": "lineups_-_vs_rhp",
    "lineup_vs_lhp": "lineups_-_vs_lhp",
    "lineup_overview": "lineups_-_overview",
    "team_pitching": "_pitching_default",
    "fielding_stats": "fielding_stats",
    "position_ratings": "position_ratings",
    "pitch_ratings": "individual_pitch_ratings",
}
```

The `identify_file_type()` function (in `csv_parser.py`) matches CSV filenames against these patterns.

#### 5.2.2 Market Data Ingestion: `ingest_market_data()`

1. Parses CSV via `parse_market_csv()`.
2. For each row:
   - Extracts card_id, position, pitcher_role, tier.
   - Computes `batting_meta` (if not a pitcher) or `pitching_meta` (if pitcher) using current weights.
   - Computes age from YearOB: `2026 - year_ob`.
   - **Upserts** into `cards` table: `INSERT ... ON CONFLICT(card_id) DO UPDATE SET` -- updates title, value, owned status, meta scores, prices, timestamp.
   - **Inserts** into `price_snapshots`: `INSERT OR IGNORE` -- skips if a snapshot for this card+date already exists (UNIQUE index dedup).
3. Snapshot date: extracted from CSV `date` column if present (truncated to date-only), else `datetime.now().strftime("%Y-%m-%d")`.

#### 5.2.3 Roster Batting Ingestion: `ingest_roster_batting()`

1. Clears today's non-league, non-pitching roster entries.
2. For each player:
   - Extracts ratings: CON, GAP, K's, EYE, POW, BABIP, DEF.
   - Extracts split ratings: CON vL/vR, POW vL/vR, EYE vL/vR.
   - Computes three meta scores: overall, vs RHP, vs LHP.
   - **`lineup_role` determination**: `'bench'` if status column is `"Reserve Roster"`, else `'starter'`.
   - **Card matching**: `SELECT card_id FROM cards WHERE (first_name || ' ' || last_name) = ? OR card_title LIKE ? LIMIT 1`.
   - Inserts into both `roster` and `my_collection` tables.
   - Stores bats handedness from the `B` column.

#### 5.2.4 Roster Pitching Ingestion: `ingest_roster_pitching()`

1. Clears today's pitching roster entries (SP/RP/CL that aren't league).
2. For each pitcher:
   - Extracts: STU, MOV, CON (control), HRA, STA/STM, HLD, STU vL, STU vR.
   - Computes three meta scores: overall, vs LHB, vs RHB.
   - **`lineup_role` determination**: `'rotation'` for SP, `'closer'` for CL, `'bullpen'` for RP. If status is `"Reserve Roster"`, overridden to `'reserve'`.
3. Same card matching and dual-insert pattern.

#### 5.2.5 Stats Ingestion

**Batting stats** (`ingest_stats_batting()`):
- Detects format: if CSV contains `wOBA` or `BB%`, routes to advanced handler.
- Deletes any existing snapshot from today (re-import overwrites).
- Calls `_match_card_id()` to link stats to cards.
- Standard stats insert: G, PA, AB, H, 2B, 3B, HR, RBI, R, BB, IBB, HBP, K, GIDP, AVG, OBP, SLG, ISO, OPS, OPS+, BABIP, WAR, SB, CS.

**Pitching stats** (`ingest_stats_pitching()`):
- Same format detection (SIERA or WIN% -> advanced handler).
- Same delete-today + insert pattern.
- Stats: G, GS, W, L, SV, HLD, IP, HA, HR, R, ER, BB, K, HBP, ERA, AVG, BABIP, WHIP, HR/9, BB/9, K/9, K/BB, ERA+, FIP, WAR.

**Roster stats variants** (`ingest_roster_batting_stats()`, `ingest_roster_pitching_stats()`):
- Detect stats_1 (standard: AB, AVG) vs stats_2 (advanced: wOBA, BB%) format.
- Route to the appropriate handler.
- Standard format uses upsert logic: if player+date exists, UPDATE; else INSERT.

#### 5.2.6 New File Types

- **`ingest_fielding_stats()`**: G, GS, TC, A, PO, E, DP, PCT, RNG, ZR, EFF, SBA, RTO, RTO%, IP, PB, CERA, FRM, ARM.
- **`ingest_pitch_ratings()`**: Individual pitch grades (FB, CH, CB, SL, SI, SP, CT, FO, CC, SC, KC, KN), pitch count, velocity, slot, stamina.
- **`ingest_position_ratings()`**: Position eligibility ratings across all 9 field positions.

#### 5.2.7 `_match_card_id()` -- Name-to-Card Matching

```python
def _match_card_id(cursor, player_name):
    card_row = cursor.execute(
        "SELECT card_id FROM cards WHERE (first_name || ' ' || last_name) = ? "
        "OR card_title LIKE ? LIMIT 1",
        (player_name, f"%{player_name}%")
    ).fetchone()
    return card_row[0] if card_row else None
```

Two strategies:
1. Exact match: `first_name || ' ' || last_name = player_name`.
2. Fuzzy match: `card_title LIKE '%player_name%'`.

### 5.3 `recalculate_all_meta_scores()`

Called after weight calibration to update all stored meta scores without re-importing.

**Flow**:
1. Loads current weights via `get_weights_with_source()`.
2. **Batting recalculation**: Queries ALL non-pitcher cards. For each:
   - Computes `defense_score` and `speed_score` from raw card data.
   - Calls `calc_batting_meta()` with the full card dict.
   - Updates `cards.meta_score_batting`.
3. **Pitching recalculation**: Queries ALL pitcher cards. Similar process.
4. **Roster meta sync** (critical -- prevents stale roster metas):
   - **Primary path (card_id match)**: Updates `roster.meta_score` from `cards.meta_score_batting` or `cards.meta_score_pitching` where `roster.card_id = cards.card_id`.
   - **Fallback path (name + OVR match)**: For roster rows without `card_id`:
     ```sql
     UPDATE roster SET meta_score = (
         SELECT c.meta_score_batting FROM cards c
         WHERE c.card_title LIKE '%' || roster.player_name || '%'
           AND c.card_value = roster.ovr
           AND c.meta_score_batting IS NOT NULL AND c.position != 1
         LIMIT 1
     )
     WHERE card_id IS NULL AND position NOT IN ('SP', 'RP', 'CL')
     ```
   - Four separate UPDATE statements cover: batting by card_id, pitching by card_id, batting by name+OVR, pitching by name+OVR.

**Return value**: `{status, batters_updated, pitchers_updated, roster_synced, weight_source, message}`.

---

## 6. KNOWN ISSUES, DESIGN DECISIONS, AND TRADE-OFFS

### 6.1 Why Position Defense Multipliers Were Chosen

The multipliers in `POSITION_DEFENSE_MULTIPLIERS` blend two sources:

1. **Empirical league data correlations** (cited in `constants.py` comments):
   - CF def->WAR r=+0.257
   - SS def->WAR r=+0.250
   - 1B def->WAR r=+0.258 (surprisingly high -- likely confounded with playing time)
   - 3B r=+0.185
   - LF r=+0.205
   - 2B r=+0.125
   - RF r=+0.054
   - C r=+0.034

2. **Traditional positional scarcity**: SS, C, and CF are premium positions with limited supply of elite defenders. The multipliers intentionally override the raw correlation data in some cases (e.g., C gets 1.30 despite low r=+0.034, because framing and arm value are not fully captured in WAR; 1B gets only 0.50 despite high r=+0.258, because "anyone can play 1B").

### 6.2 Why Speed Scoring Is Conditional (Only Above 70 Composite)

```python
if composite < 70:
    return 0.0
```

Design rationale:
- Speed below average should NOT penalize a player. Slow players are already penalized by their lower in-game production (fewer infield hits, fewer stolen bases, etc.), which shows up in their real WAR/OPS.
- Only elite or above-average speed creates marginal value beyond what batting stats capture -- it converts singles into doubles via extra bases taken, and converts walks into scoring opportunities via stolen bases.
- The 70 threshold represents "average" on the OOTP rating scale.
- Speed contribution is conditional on getting on base (amplifies OBP), which is why the comment says "conditional value."

### 6.3 Why Calibration Uses card_id Joins Instead of Name Matching

The primary calibration query joins `cards` and `batting_stats` via `bs.card_id = c.card_id`. This was implemented AFTER a bug where name-based matching returned 0 matches.

**The 0-match bug**: The original calibration used `c.card_title LIKE '%' || bs.player_name || '%'` as the primary join. This failed when:
- Card titles have set prefixes (e.g., "Snapshot CF Mike Trout LAA 2026") that don't match the plain "Mike Trout" in stats tables.
- Multiple cards share the same player name (different card years/variants), causing ambiguous or incorrect matches.

**Solution**: The `card_id` field was added to `batting_stats` and `pitching_stats` tables, populated during ingestion via `_match_card_id()`. The calibration now uses `card_id` as the primary join and only falls back to name-LIKE matching if the card_id join returns too few results (< 15 batters or < 10 pitchers).

### 6.4 The Bellinger vs Carbo Saga: Why Performance Gates Were Added

The performance gate (`WAR/600 >= 1.5`) was added because of real roster situations where:
- A bench player had a HIGHER meta score than the active starter.
- The meta formula would recommend benching the starter.
- But the starter was actually producing excellent in-game results (high WAR, good OPS/ERA).
- The bench player had no game data to validate their meta prediction.

The classic example: a player like Bellinger might have lower card ratings than a bench alternative like Carbo, but Bellinger is producing 2.0+ WAR/600 in actual games. Benching a proven producer for an unproven (but theoretically better) card is a bad strategy.

**Gate logic**: Only skip the recommendation when ALL of:
1. The starter has performance data.
2. The starter's WAR rate >= 1.5 (decent+ production).
3. The bench candidate has NO performance data.

If both players have performance data, or the starter is NOT producing well, the recommendation proceeds normally.

### 6.5 Why Platoon Splitting Was Removed From Batting (The C1/C2/DH Problem)

Earlier versions of the system attempted platoon splitting: showing two starters per position (one for vs RHP, one for vs LHP) based on their split meta scores. This was removed because:

1. **The DH interaction**: If C1 starts vs RHP and C2 starts vs LHP, the non-starting catcher should DH. But the system couldn't reliably model this second-order effect -- it would show a "DH" candidate who was actually supposed to be a platoon partner.

2. **Lineup validation complexity**: OOTP only allows 25 active roster spots. A full platoon system across 8 batting positions would need up to 16 lineup slots, which is impossible. The system would need complex constraint satisfaction to determine which positions to platoon and which to use a single starter.

3. **Handedness data reliability**: The `bats` field from roster CSVs doesn't always reliably populate, leading to `?` handedness for many players.

**Current approach**: Platoon metadata is still computed and stored (`meta_vs_rhp`, `meta_vs_lhp`), and the roster optimizer shows a platoon warning when multiple same-handed batters share a position, but it does NOT recommend specific platoon splits.

### 6.6 The CL-in-Batting-Tab Bug (startswith('C') Matching CL)

In the `build_chain_rows()` function, positions are matched against a filter list. Early code used simple `pos.startswith(p)` to match numbered slots (e.g., `SP1` matching `SP`). This caused `CL` (closer) to match `C` (catcher) because `'CL'.startswith('C')` is `True`.

**Fix**: Added a digit check:
```python
pos.startswith(p) and len(pos) > len(p) and pos[len(p)].isdigit()
```
Now `CL` does not match `C` because `'L'` is not a digit. But `C1` (platoon catcher) would still match `C` because `'1'` is a digit. Similarly, `SP1` matches `SP`, `MID2` matches `MID`, etc.

### 6.7 Duplicate Recommendation Prevention

The system uses multiple mechanisms to prevent the same player/card being recommended for multiple positions:

1. **`_all_active_names`**: Set of all active roster player names. Passed as `exclude_names` to prevent recommending someone already starting elsewhere.

2. **`used_names` (per multi-slot position)**: For SP and RP processing, a running set tracks both player names AND card titles of already-recommended upgrades:
   ```python
   pname = bo.get('player_name', entry['owned_name'])
   used_names.add(pname)           # Player name
   used_names.add(entry['owned_name'])  # Card title (for cards-table exclusion)
   ```
   This handles the case where a card title like "Snapshot SP Bob Smith NYY 2026" needs to match both the player name "Bob Smith" and the full card title.

3. **`used_market_ids`**: Set of card_ids already recommended from the market. Passed to `find_market_upgrades()` as `exclude_ids`.

4. **`used_owned_titles`**: Set of card titles already recommended from owned cards. Prevents the same owned card appearing in multiple slots.

5. **Substring matching in exclude_names**: `find_roster_bench_upgrades()` checks both exact match AND substring match:
   ```python
   if any(pname in ex or ex in pname for ex in exclude_names)
   ```
   This catches cases where the exclude list contains either a full card title or just a player name.

### 6.8 AvK and BABIP Zeroed Out (Double-Counting Prevention)

Both `avoid_ks` and `babip` have weights of 0.00 in the default weights. The code comment explains:

> CON is derived stat in OOTP25+ that already incorporates them.

In OOTP 25 and later, the Contact rating is a derived stat that combines BABIP and Avoid K's ability. Weighting all three would double-count the effect. Setting AvK and BABIP weights to zero while using Contact at 1.80 avoids this.

### 6.9 Control at 0.20 (Near-Zero Weight for Pitchers)

The pitching Control weight of 0.20 is dramatically lower than community consensus (which often weights it at 1.5-2.0). This is based on empirical data: `ERA r=-0.002` -- essentially zero correlation between Control rating and actual ERA in the sample of 633 pitchers.

The code comment notes it was "massively overweighted at 1.8" previously. The data suggests that in OOTP 27 Perfect Team, Movement and Stuff drive pitching results far more than Control.

### 6.10 OVR as Multiplicative Rather Than Additive

OVR is applied as a **multiplicative scaling factor**, not an additive component. This means:
- A high OVR amplifies the entire weighted sum (a 10% boost on a 600 meta = +60).
- A low OVR dampens it (a 12% penalty on a 600 meta = -72).
- This makes OVR disproportionately important for cards that are already strong. Two cards with identical ratings but different OVR will have significantly different metas.

This design choice is supported by OVR having the highest individual correlation with WAR (r=+0.529 batting, r=+0.601 pitching) -- it captures information about card quality that individual ratings miss.

---

## Appendix A: config.yaml Reference

```yaml
ai_provider: gemini
batting_weights:
  avoid_ks: 0.0
  babip: 0.0
  contact: 1.8
  defense: 1.5
  eye: 0.6
  gap_power: 1.4
  ovr: 1.25
  power: 1.4
pitching_weights:
  control: 0.2
  movement: 2.4
  ovr: 1.5
  p_hr: 1.8
  stamina_hold: 0.4
  stuff: 1.4
pp_budget: 2000
team_name: Toronto Dark Knights
database_path: data/ootp_optimizer.db
watch_directory: C:\Users\Cameron\OneDrive\Documents\Out of the Park Developments\OOTP Baseball 27\online_data
recommendations:
  max_budget_pct: 0.8
  min_meta_improvement: 10
  price_drop_threshold: 0.15
  price_spike_threshold: 0.2
  value_ratio_threshold: 50.0
dashboard:
  port: 8501
  theme: dark
watcher:
  auto_start: true
  debounce_seconds: 2
  poll_interval_seconds: 5
```

Note: `speed_stealing: 0.50` is present in the code defaults but NOT in `config.yaml`. This means it falls through to the `DEFAULT_BATTING_WEIGHTS` value of 0.50 unless calibrated weights override it. Similarly, `stamina_hold` is in `config.yaml` as 0.40 but defaults to 0.30 in constants.py -- the config value takes precedence.

## Appendix B: File Topology

```
OOTPBUYNSELL/
  config.yaml                   -- Configuration (weights, paths, budget)
  app/
    main.py                     -- Streamlit entry point, sidebar, data import UI
    core/
      database.py               -- Schema DDL, init_db(), connections, views
      meta_scoring.py           -- Meta formula engine (all calc_* functions)
      meta_validation.py        -- Auto-calibration, validation, accuracy scoring
      ingestion.py              -- CSV parsing + DB writes (18 file type handlers)
      recommendations.py        -- Buy/sell recommendation generation
      ai_advisor.py             -- Gemini/Claude API integration for AI analysis
      price_alerts.py           -- Price trigger monitoring
    utils/
      constants.py              -- Weights, multipliers, file patterns, position maps
      csv_parser.py             -- CSV column mapping and parsing
      sparklines.py             -- Price trend sparklines
    pages/
      1_Buy_Recommendations.py  -- Investment advisor (gaps, scenarios, outliers)
      2_Sell_Recommendations.py -- Sell analysis
      4_Roster_Optimizer.py     -- Lineup card view with upgrade chains
      7_Game_Stats.py           -- In-game stats dashboard
      13_Tournament_Builder.py  -- Tournament mode
      15_Export_Plan.py         -- Export upgrade plan
  data/
    ootp_optimizer.db           -- SQLite database (auto-created)
```
