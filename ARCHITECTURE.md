# OOTP Perfect Team Optimizer — Technical Architecture & Feature Document

**Version:** 5.0 (Multi-Factor Meta Architecture)
**Date:** 2026-04-16
**Classification:** Internal Technical Reference
**Author:** Cameron Chojnacki / Claude AI Engineering
**Team:** Toronto Dark Knights — OOTP 27 Perfect Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Database Schema](#4-database-schema)
5. [Meta Scoring Engine](#5-meta-scoring-engine)
6. [Meta Calibration System](#6-meta-calibration-system)
7. [Correlation Analysis & Empirical Findings](#7-correlation-analysis--empirical-findings)
8. [Multi-Factor Meta Layers](#8-multi-factor-meta-layers)
9. [Recommendation Engine](#9-recommendation-engine)
10. [Presentation Layer](#10-presentation-layer)
11. [AI Integration](#11-ai-integration)
12. [Current Limitations & Research Opportunities](#12-current-limitations--research-opportunities)

---

## 1. Executive Summary

The OOTP Perfect Team Optimizer is a decision-support system for competitive play in Out of the Park Baseball 27's Perfect Team mode — a collectible card game where players build MLB rosters from historical and current player cards, then compete in simulated leagues.

The system ingests ~30 CSV export files from the game client (2,700+ cards, 600+ player performance records, 172 pitch arsenal entries, 684 fielding records, park factors for 30 stadiums), computes a composite "meta score" for every card using a multi-factor model calibrated against actual in-game WAR, and generates prioritized buy/sell/flip recommendations optimized for the user's budget and roster needs.

**Core thesis:** OOTP's built-in Overall Rating (OVR) captures ~50% of the variance in player WAR (r=+0.517 for batting, r=+0.416 for pitching), but it is a black box that misses platoon vulnerability, pitch arsenal diversity, positional flexibility, and actual in-game performance divergence from ratings. The meta score aims to capture these additional signals to rank players more accurately than OVR alone.

**Current correlation performance (league lb124, 2026-04-16):**
- Batting meta vs WAR/600PA: r=0.509 (n=321)
- Pitching meta vs WAR/200IP: r=0.416 (n=282)
- For comparison: raw OVR vs batting WAR: r=0.517; raw OVR vs pitching WAR: r=0.416

---

## 2. System Architecture

### 2.1 Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Application Framework | Streamlit | ≥1.30.0 |
| Database | SQLite3 | Built-in Python |
| Data Processing | Pandas | ≥2.0.0 |
| Visualization | Plotly | ≥5.18.0 |
| Statistical Modeling | scikit-learn (RidgeCV), scipy (NNLS), numpy | ≥1.4.0, ≥1.12.0, ≥1.26.0 |
| AI Integration | Google Gemini (primary), Anthropic Claude (secondary) | google-genai ≥1.0.0, anthropic ≥0.40.0 |
| File Watching | Watchdog | ≥3.0.0 |
| Configuration | YAML | pyyaml ≥6.0 |
| MLB Live Data | MLB-StatsAPI | ≥1.7.0 |

### 2.2 Directory Structure

```
OOTPBUYNSELL/
├── app/
│   ├── main.py                    # Streamlit entry point, dashboard
│   ├── core/                      # Business logic (no Streamlit imports)
│   │   ├── database.py            # Schema, connections, migrations (981 lines)
│   │   ├── ingestion.py           # CSV parsing + DB writes (2,190 lines)
│   │   ├── data_refresh.py        # Scan → plan → execute pipeline (534 lines)
│   │   ├── meta_scoring.py        # Meta formula engine (1,300+ lines)
│   │   ├── meta_calibration.py    # Ridge → NNLS → Bayesian blend (915 lines)
│   │   ├── recommendations.py     # Buy/sell/flip generation (616 lines)
│   │   ├── ai_advisor.py          # Gemini/Claude strategic analysis
│   │   ├── history.py             # Player history snapshots + trending
│   │   ├── flip_finder.py         # Arbitrage detection
│   │   ├── live_card_tracker.py   # MLB live card upgrade prediction
│   │   ├── price_analysis.py      # Price anomaly detection
│   │   ├── tournament.py          # Tournament roster optimization
│   │   └── csv_parser.py          # 25+ parse functions for OOTP CSVs
│   ├── pages/                     # 15 Streamlit page files
│   │   ├── 0_Data_Refresh.py      # Import orchestration UI
│   │   ├── 1_Buy_Recommendations.py
│   │   ├── 2_Sell_Recommendations.py
│   │   ├── 3_Price_Trends.py
│   │   ├── 4_Roster_Optimizer.py
│   │   ├── 5_Settings.py
│   │   └── ... (10 more pages)
│   └── utils/
│       ├── constants.py           # Weights, maps, patterns
│       ├── sidebar_nav.py         # Grouped navigation
│       ├── sparklines.py          # ASCII price sparklines
│       └── csv_parser.py
├── config.yaml                    # User configuration
├── data/
│   └── ootp_optimizer.db          # SQLite database (~50MB typical)
└── requirements.txt
```

### 2.3 Data Flow Architecture

```
OOTP 27 Game Client
        │
        ▼  (CSV export to watch folder)
┌─────────────────────┐
│  Watch Directory     │  44 CSV files, ~2MB total
│  (Watchdog monitor)  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Data Refresh Page   │  scan_watch_directory() → plan_refresh() → execute_refresh()
│  (0_Data_Refresh.py) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Ingestion Engine    │  22 handler functions, file-type dispatch
│  (ingestion.py)      │  CSV → parse → meta_score → SQLite
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│ SQLite │ │ Meta Scoring  │  calc_batting_meta() / calc_pitching_meta()
│   DB   │ │   Engine      │  + 4 multi-factor layers
└────┬───┘ └──────┬───────┘
     │            │
     ▼            ▼
┌─────────────────────┐
│  Calibration Engine  │  RidgeCV → NNLS → Bayesian prior blend
│  (meta_calibration)  │  Targets: WAR/600PA (batting), WAR/200IP (pitching)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Recommendation      │  Gap analysis → value scoring → priority ranking
│  Engine              │  Buy / Sell / Flip / Live Card recs
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Presentation Layer  │  15 Streamlit pages
│  (Dashboard + Pages) │  Charts, tables, explainers, AI insights
└─────────────────────┘
```

---

## 3. Data Pipeline

### 3.1 Source Data

OOTP 27 exports CSV files to a configurable watch directory. The system recognizes 26 distinct file patterns via longest-substring matching:

| Category | File Types | Typical Row Count | Key Columns |
|----------|-----------|-------------------|-------------|
| **Market** | `market` (pt_card_list.csv) | 2,702 | card_id, all ratings (72 cols), buy/sell prices, ownership |
| **Roster** | `roster_batting`, `roster_pitching` | 200 | Lineup role, split ratings (vL/vR), position |
| **Collection** | `collection_batting`, `collection_pitching` | 263 | Owned cards with ratings |
| **League Stats** | `stats_batting`, `stats_pitching` + advanced variants | 1,648 | PA, AB, HR, WAR, OPS+, wOBA, SIERA |
| **League Ratings** | `league_batting_ratings`, `league_pitching_ratings` | 824 | Full league card ratings for cross-team comparison |
| **Fielding** | `fielding_stats`, `fielding_ratings`, `position_ratings` | 684 | DRS, UZR, errors, position eligibility ratings |
| **Pitch Arsenal** | `pitch_ratings` (individual + roster) | 172 | FB/CH/CB/SL/SI ratings, velocity, pitch count |
| **Park Factors** | `park_info` | 30 | pf_hr, pf_avg, pf_overall per stadium |
| **Team Stats** | `team_stats_batting/pitching/fielding` | 60 | Team-level aggregate stats + park factors |
| **Lineups** | `lineups_overview`, `vs_rhp`, `vs_lhp` | 52 | Lineup slot assignments by matchup type |

### 3.2 Ingestion Pipeline

**File identification** (`csv_parser.identify_file_type`): Matches filename substrings against 26 patterns sorted by length descending. Longer patterns win (e.g., `team_statistics___info_-_sortable_stats_batting_stats` wins over `sortable_stats_batting_stats`).

**League gating** (`ingestion._get_active_league`): Files with a detected `league_id` that doesn't match `config.yaml:active_league` are skipped with an audit trail entry. This prevents stale data from inactive leagues from polluting the active dataset.

**Card ID resolution** (`ingestion._match_card_id`): Links performance stats to cards via exact `card_id` match (preferred) or fuzzy name matching (fallback). Fuzzy matching prefers owned cards to avoid cross-era confusion (e.g., multiple "Jose Ramirez" cards).

**Meta score computation**: During market data ingestion, `calc_batting_meta()` and `calc_pitching_meta()` are called for every card. The computed scores are written to `cards.meta_score_batting` and `cards.meta_score_pitching`. During full recalculation, the multi-factor layers (platoon splits, position flexibility, arsenal diversity, performance overlay) are also incorporated.

### 3.3 History Snapshots

After each successful refresh, `snapshot_player_history()` captures the complete state of every player (ratings, meta scores, market prices, in-game stats) into `player_history`. This enables:
- Meta score trending over time
- Market price trending
- Performance trajectory tracking
- Pre/post calibration comparison

---

## 4. Database Schema

### 4.1 Core Tables (17 tables, 4 views)

The SQLite database contains 17 tables and 4 convenience views. Key tables:

**`cards`** (primary card catalog, ~90 columns):
- Identity: `card_id` (PK), `card_title`, `first_name`, `last_name`
- Card metadata: `card_type`, `card_sub_type`, `card_badge`, `card_series`, `tier`, `tier_name`
- Overall: `card_value` (OVR), `position`, `pitcher_role`, `pitcher_role_name`
- Batting ratings (overall): `contact`, `gap_power`, `power`, `eye`, `avoid_ks`, `babip`
- Batting splits (vs LHP): `contact_vl`, `gap_vl`, `power_vl`, `eye_vl`, `avoid_ks_vl`, `babip_vl`
- Batting splits (vs RHP): `contact_vr`, `gap_vr`, `power_vr`, `eye_vr`, `avoid_ks_vr`, `babip_vr`
- Pitching ratings (overall): `stuff`, `movement`, `control`, `p_hr`, `p_babip`
- Pitching splits: `stuff_vl/vr`, `movement_vl/vr`, `control_vl/vr`, `p_hr_vl/vr`, `p_babip_vl/vr`
- Speed/baserunning: `speed`, `steal_rate`, `stealing`, `baserunning`, `sac_bunt`, `bunt_for_hit`
- Pitching mechanics: `stamina`, `hold`, `velocity`
- Infield defense: `infield_range`, `infield_error`, `infield_arm`, `dp`
- Catcher defense: `catcher_ability`, `catcher_frame`, `catcher_arm`
- Outfield defense: `of_range`, `of_error`, `of_arm`
- Position eligibility: `pos_rating_p`, `pos_rating_c`, `pos_rating_1b`, ..., `pos_rating_rf` (9 columns)
- Market: `buy_order_high`, `sell_order_low`, `last_10_price`, `last_10_variance`, `owned`
- Computed: `meta_score_batting`, `meta_score_pitching`

**`batting_stats` / `pitching_stats`** (partitioned by `league_id`):
- Linked to `cards` via `card_id`
- Batting: PA, AB, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, AVG, OBP, SLG, OPS, OPS+, WAR
- Pitching: W, L, SV, IP, H, ER, HR, BB, SO, ERA, FIP, WHIP, ERA+, WAR

**`batting_stats_adv` / `pitching_stats_adv`** (advanced analytics):
- Batting: wOBA (r=+0.884 with WAR), WPA, ISO, BB%, K%, RC/27
- Pitching: SIERA (r=-0.654 with WAR), WPA, GO%, QS, QS%, pLI

**`pitch_ratings`** (per-pitch-type arsenal):
- 12 pitch types: FB, CH, CB, SL, SI, SP, CT, FO, CC, SC, KC, KN
- Plus: `pitch_count`, `velocity`, `slot`

**`league_team_stats`** (team-level aggregates + park factors):
- pf_hr, pf_avg, pf_overall, pf_hr_l, pf_hr_r, pf_avg_l, pf_avg_r, pf_d, pf_t

**`meta_calibration`** (calibration run results):
- `calibration_type`: 'batting', 'pitching_sp', 'pitching_rp', 'pitching'
- `weights_json`: Calibrated weight dict
- `r_squared`: Cross-validated R²
- `correlation`: Pearson r
- `sample_size`, `confidence`, `changes_json`, `created_at`

---

## 5. Meta Scoring Engine

### 5.1 Architecture Overview

The meta scoring engine (`meta_scoring.py`, 1,300+ lines) computes a single composite score for every card that predicts in-game value (WAR) better than any individual rating. The architecture is a weighted-sum base layer plus four additive adjustment layers:

```
Meta Score = Base Rating Score
           + Platoon Penalty (Layer 1)
           + Position Flexibility Bonus (Layer 2, batting only)
           + Arsenal Diversity Bonus (Layer 3, pitching only)
           + Performance Overlay (Layer 4)
           + Positional Value Bonus (batting only)
```

### 5.2 Batting Meta Formula

```python
def calc_batting_meta(row, weights):
    # Platoon-adjusted base stats (60% vR + 40% vL when splits available)
    gap = _apply_splits(row, gap_overall, 'gap_vl', 'gap_vr', 0.60)
    con = _apply_splits(row, con_overall, 'contact_vl', 'contact_vr', 0.60)
    eye = _apply_splits(row, eye_overall, 'eye_vl', 'eye_vr', 0.60)
    pwr = _apply_splits(row, pwr_overall, 'power_vl', 'power_vr', 0.60)

    # Core weighted sum with diminishing returns above 110
    meta = (diminished(gap) * w['gap_power']     # default: 1.20
          + diminished(con) * w['contact']        # default: 1.80
          + diminished(avk) * w['avoid_ks']       # default: 0.00 (dead signal)
          + diminished(eye) * w['eye']             # default: 0.80
          + diminished(pwr) * w['power']           # default: 2.00
          + diminished(bab) * w['babip']           # default: 1.20
          + diminished(ovr) * w['ovr']             # default: 2.00
          + defense * w['defense'])                # default: 0.40

    # Speed/Stealing (conditional, only above avg 70)
    meta += diminished(speed_score + 70) * w['speed_stealing']  # default: 0.15

    # Layer 1: Platoon penalty
    meta += platoon_penalty_batting(row)

    # Balance penalty: key stats below floor (55) get penalized
    # Power coefficient 1.0 (hard ceiling), Contact/Gap 0.4
    for stat, coeff in [(con, 0.4), (gap, 0.4), (pwr, 1.0)]:
        if 0 < stat < 55: meta -= (55 - stat) * coeff

    # Layer 2: Position flexibility (+0 to +10)
    meta += position_flexibility(row)

    # Layer 4: Performance overlay (wOBA-based, ±40)
    meta += performance_adjustment_batting(row)

    # Positional value bonus (fWAR ladder: C +31, SS +19, DH -44)
    meta += POSITIONAL_VALUE_BONUS[position]
```

### 5.3 Pitching Meta Formula

```python
def calc_pitching_meta(row, weights):
    # Platoon-adjusted base (55% vR + 45% vL)
    mov = _apply_splits(row, mov_overall, 'movement_vl', 'movement_vr', 0.55)
    stu = _apply_splits(row, stu_overall, 'stuff_vl', 'stuff_vr', 0.55)
    ctrl = _apply_splits(row, ctrl_overall, 'control_vl', 'control_vr', 0.55)

    meta = (diminished(mov)  * w['movement']        # default: 0.80
          + diminished(stu)  * w['stuff']            # default: 2.40
          + diminished(ctrl) * w['control']          # default: 0.30
          + diminished(phr)  * w['p_hr']             # default: 1.40
          + diminished(ovr)  * w['ovr'])             # default: 2.00

    # SIERA-inspired interaction terms (non-additive synergy)
    meta += stu * mov * w['stuff_x_movement']        # default: 0.008
    meta += stu * ctrl * w['stuff_x_control']        # default: 0.010
    meta += mov * ctrl * w['movement_x_control']     # default: 0.002

    # Stamina/Hold component
    meta += avg(stamina, hold) * w['stamina_hold']   # default: 0.10

    # Layer 1: Platoon penalty
    meta += platoon_penalty_pitching(row)

    # Balance penalty: STU/MOV/CTL below 65 floor
    for stat in [stu, mov, ctrl]:
        if 0 < stat < 65: meta -= (65 - stat) * 1.0

    # Layer 3: Arsenal diversity (+0 to +15 SP, +0 to +10 RP)
    meta += arsenal_factor(row, is_sp)

    # Layer 4: Performance overlay (SIERA-based, ±40)
    meta += performance_adjustment_pitching(row)
```

### 5.4 Diminishing Returns Function

```
_diminished(value, threshold=110):
    if value <= 110: return value          # Linear 1:1
    excess = value - 110
    return 110 + sqrt(excess) * 4          # sqrt-scaled above threshold
```

**Rationale:** Prevents extreme single-stat spikes from dominating. A 150-rated stat contributes as if it were ~135. This is grounded in OOTP's internal simulation engine, which has its own diminishing returns at extreme ratings — our function approximates that non-linearity.

### 5.5 Weight Loading Priority Chain

```
1. Calibrated weights from meta_calibration table (if CV R² ≥ threshold OR Pearson r ≥ 0.30)
2. config.yaml weights (user-tunable)
3. DEFAULT_*_WEIGHTS from constants.py (hardcoded evidence-based defaults)
```

Role-aware dispatch for pitching: SP and RP have separate calibrated weight rows. A closer with high stuff but low stamina gets the RP weight profile (stuff-dominant), not the SP profile (more balanced).

---

## 6. Meta Calibration System

### 6.1 Pipeline: RidgeCV → NNLS → Bayesian Prior Blend

The calibration engine (`meta_calibration.py`, 915 lines) fits card ratings against actual in-game WAR to learn optimal weights. The three-stage pipeline addresses specific statistical challenges:

**Stage 1 — RidgeCV (regularized regression):**
- Solves multicollinearity between correlated ratings (stuff/movement/control share variance)
- Alpha grid: [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
- 5-fold cross-validation provides the gate metric (out-of-sample R²)
- Sample weights: sqrt(PA) for batting, sqrt(IP) for pitching — high-sample cards get more influence than noisy low-sample ones

**Stage 2 — NNLS (non-negative least squares):**
- Ridge coefficients can flip sign due to multicollinearity (movement goes negative when stuff absorbs shared variance)
- NNLS refits on standardized features to produce interpretable non-negative weights
- Coefficients are un-scaled back to per-raw-rating units

**Stage 3 — Bayesian prior blend:**
```
confidence = min(n / 200.0, 0.40)    # Capped at 40% empirical influence
final_weight = confidence * empirical + (1 - confidence) * prior
```

**Rationale for 0.40 cap:** With CV R² of 0.03–0.16, the regression captures real signal but also multicollinearity artifacts. At full confidence, these artifacts dominated rankings. At 0.40, calibration nudges weights ~15-20% from prior — appropriate for this R² range.

### 6.2 Safety Rails

**Sanity gate:** New weights clipped to ±25% of prior values (`SANITY_DOWN_FRAC=0.25`, `SANITY_UP_MULT=1.40`).

**Winsorization:** Target variable (WAR) clipped at 1st/99th percentile before fitting to prevent outlier cards from dominating the loss.

**Dual-gate acceptance:** Calibration results are trusted if EITHER:
- Cross-validated R² ≥ 0.10 (batting) or ≥ 0.03 (pitching), OR
- Pearson correlation ≥ 0.30

### 6.3 Fit Targets

| Type | Target | Sample Filter | Typical n | Typical CV R² | Pearson r |
|------|--------|--------------|-----------|---------------|-----------|
| Batting | WAR per 600 PA | PA ≥ 150 | 278 | 0.158 | 0.528 |
| Pitching SP | WAR per 200 IP | IP ≥ 50 | 142 | 0.050 | 0.477 |
| Pitching RP | WAR per 200 IP | IP ≥ 25 | 140 | 0.035 | 0.439 |
| Pitching Combined | WAR per 200 IP | IP ≥ 30 | 282 | 0.089 | 0.403 |

### 6.4 Feature Sets

**Batting features:** contact, gap_power, power, eye, babip (5 features)

**Pitching features:** stuff, movement, control, p_hr (4 features). Interaction terms currently disabled (PITCHING_INTERACTIONS = []) due to overfit at current sample sizes.

---

## 7. Correlation Analysis & Empirical Findings

### 7.1 Methodology

Pearson correlations computed on lb124 league data (2026-04-16 snapshot). Batting: 209 players with PA ≥ 150. Pitching: 220 players with IP ≥ 30. Target: WAR per 600 PA (batting) and WAR per 200 IP (pitching).

### 7.2 Batting Correlations with WAR/600PA (n=209)

| Rank | Feature | r | p-value | Status |
|------|---------|---|---------|--------|
| 1 | wOBA (actual advanced stat) | +0.884 | <0.001 | **Strongest signal by far** |
| 2 | OVR (card_value) | +0.517 | <0.001 | **Captures hidden attributes** |
| 3 | BABIP rating | +0.318 | <0.001 | **3rd strongest — NOT double-counted** |
| 4 | Power vR | +0.298 | <0.001 | vR split > overall for power |
| 5 | Power (overall) | +0.289 | <0.001 | **Strongest individual rating** |
| 6 | Contact (overall) | +0.270 | <0.001 | 2nd strongest rating |
| 7 | Contact vR | +0.264 | <0.001 | |
| 8 | Power weak side | +0.260 | <0.001 | Weak-side power still predictive |
| 9 | Power vL | +0.226 | <0.01 | |
| 10 | Contact weak side | +0.225 | <0.01 | |
| 11 | Eye vL | +0.203 | <0.01 | |
| 12 | Contact vL | +0.184 | <0.01 | |
| 13 | Gap Power | +0.183 | <0.01 | Moderate |
| 14 | Eye (overall) | +0.160 | <0.05 | Weak but significant |
| 15 | Eye vR | +0.130 | =0.060 | Marginal |
| — | Speed | +0.076 | =0.27 | **Not significant** |
| — | Avoid Ks | +0.063 | =0.36 | **Dead signal** |
| — | Defense Score | +0.046 | =0.51 | **OOTP overvalues defense** |
| — | Playable Positions | -0.023 | =0.74 | **No flexibility signal** |
| — | Stealing | -0.014 | =0.84 | **No signal** |

**Key insight:** Defense (r=+0.046, p=0.51) and speed/stealing (r≈0) have essentially zero correlation with WAR in this dataset. The prior defense weight of 1.50 was dramatically overweighted. OOTP's simulation engine apparently does not translate defensive ratings into WAR at the rate real baseball does. This may be because OOTP PT games are shorter series where offensive variance dominates.

### 7.3 Pitching Correlations with WAR/200IP (n=220)

| Rank | Feature | r | p-value | Status |
|------|---------|---|---------|--------|
| 1 | SIERA (actual advanced stat) | -0.654 | <0.001 | **Strongest pitching signal** |
| 2 | OVR (card_value) | +0.416 | <0.001 | **Strong anchor** |
| 3 | Stuff vL | +0.363 | <0.001 | |
| 4 | Stuff (overall) | +0.357 | <0.001 | **Dominant #1 rating** |
| 5 | Stuff × Control (interaction) | +0.338 | <0.001 | **Strongest interaction** |
| 6 | Stuff × Movement (interaction) | +0.332 | <0.001 | Strong interaction |
| 7 | Stuff vR | +0.303 | <0.001 | |
| 8 | Movement vR | +0.146 | <0.05 | |
| 9 | HR Suppression (p_hr) | +0.142 | <0.05 | Significant overall |
| 10 | Movement (overall) | +0.123 | =0.068 | **Marginal — was overweighted at 2.20** |
| — | Control (overall) | +0.013 | =0.85 | **Zero standalone — value is in interactions** |
| — | Stamina | +0.013 | =0.85 | **No signal** |
| — | Hold | -0.073 | =0.28 | **No signal** |
| — | Pitch count (arsenal) | +0.024 | =0.73 | **No arsenal signal** |
| — | Best pitch rating | +0.006 | =0.93 | **No signal** |

**By role (SP vs RP):**

| Feature | SP (n=151) | RP (n=69) |
|---------|-----------|-----------|
| Stuff | +0.314 | **+0.489** (dominant) |
| Stuff × Movement | **+0.380** | — |
| HR Suppression | **+0.333** | -0.077 (not sig for RP) |
| Stuff × Control | +0.330 | +0.392 |
| Movement | **+0.307** | -0.100 (slightly negative!) |
| Control | +0.038 (not sig) | — |
| Stamina | -0.060 (not sig) | — |

**Key insight:** Pitching is dominated by Stuff. Movement's standalone correlation (+0.123) is marginal and actually negative for relievers. The real signal is in the **interaction terms** — Stuff × Control (+0.338) and Stuff × Movement (+0.332) both outperform Movement and Control as standalone features. This validates the SIERA-inspired non-additive architecture.

### 7.4 Implications for Weight Assignment

Based on these findings, weights were revised on 2026-04-16:

**Batting weight changes:**
| Feature | Old Weight | New Weight | Rationale |
|---------|-----------|-----------|-----------|
| power | 1.60 | **2.00** | Strongest rating (r=+0.289) |
| contact | 2.00 | **1.80** | Strong but overweighted vs power |
| babip | 0.00 | **1.20** | r=+0.318, 3rd strongest — data refutes "double-counted" assumption |
| gap_power | 1.60 | **1.20** | Moderate signal (r=+0.183) |
| defense | 1.50 | **0.40** | r=+0.046, p=0.51 — dramatically overweighted |
| speed_stealing | 0.50 | **0.15** | r=+0.076, p=0.27 — near-zero signal |

**Pitching weight changes:**
| Feature | Old Weight | New Weight | Rationale |
|---------|-----------|-----------|-----------|
| stuff | 2.00 | **2.40** | Dominant signal (r=+0.357, +0.489 RP) |
| movement | 1.20 | **0.80** | Marginal standalone (r=+0.123) |
| control | 0.60 | **0.30** | Zero standalone (r=+0.013); value in interactions |
| stuff_x_control | 0.008 | **0.010** | Strongest interaction (r=+0.338) |
| stuff_x_movement | 0.006 | **0.008** | Strong synergy (r=+0.332) |
| stamina_hold | 0.40 | **0.10** | No signal (r=+0.013) |
| p_hr | 1.80 | **1.40** | Moderate, SP-specific |

---

## 8. Multi-Factor Meta Layers

### 8.1 Layer 1: Platoon-Adjusted Base (VALIDATED)

**Mechanism:** For key batting stats (contact, power, eye, gap), the base value uses a handedness-weighted effective rating: `effective = 0.60 × vR + 0.40 × vL` (batting) or `0.55 × vR + 0.45 × vL` (pitching). Falls back to overall rating when split data is absent.

**Platoon penalty:** When the weak-side rating falls below a critical threshold (50 for batting, 55 for pitching), an additional penalty is applied proportional to the shortfall × handedness exposure rate × stat importance coefficient.

```
Example: Batter with Contact vL=40, Contact vR=110
- Effective contact: 0.60 × 110 + 0.40 × 40 = 82
- Weak side (40) < threshold (50): penalty = (50-40) × 0.40 × 1.5 = -6.0
- Net effect: uses 82 effective (vs 80 overall) but takes a -6 platoon penalty
```

**Empirical support:** Power vR (+0.298) correlates more strongly than Power overall (+0.289), and Contact vR (+0.264) more than Contact vL (+0.184) — consistent with 60% RHP frequency. Weak-side ratings still correlate significantly (Power weak-side +0.260).

### 8.2 Layer 2: Position Flexibility Bonus (NOT VALIDATED — scaled to ±10)

**Mechanism:** Counts positions where `pos_rating >= 20` (playable in OOTP). Each extra playable position beyond the first adds +1.5 meta. Premium positions (C, SS, CF) add an additional +1.0 each. Capped at 10.

**Empirical status:** Playable position count shows r=-0.023 (p=0.74) with WAR — no signal. However, this layer was retained at reduced scale because multi-position flexibility has genuine PT roster construction value (25-man roster, injury coverage) that WAR per individual player cannot capture.

### 8.3 Layer 3: Arsenal Diversity Bonus (NOT VALIDATED — scaled to ±15 SP / ±10 RP)

**Mechanism:** Counts pitches with rating ≥ 50 (usable) and ≥ 65 (quality). SP get a diversity bonus starting at 3+ pitches (max +15). RP get a smaller bonus (max +10).

**Empirical status:** Pitch count, best pitch, and arsenal average all show r≈0 with WAR. The OOTP simulation engine apparently does not penalize limited-repertoire pitchers as strongly as real baseball does. Layer retained at reduced scale for theoretical lineup-trip-through-order value.

### 8.4 Layer 4: Performance Overlay (STRONGLY VALIDATED)

**Mechanism:** When advanced stats exist for a card, the meta adjusts based on actual vs expected performance:
- Batting: wOBA vs league average (0.320 default), scaled by sqrt(PA/500) confidence, ±40 cap
- Pitching: SIERA vs league average (4.00 default), scaled by sqrt(IP/150) confidence, ±40 cap

**Empirical support:** wOBA has r=+0.884 with batting WAR (the strongest signal in the entire dataset). SIERA has r=-0.654 with pitching WAR. These are by far the most predictive features available. The performance overlay captures hidden attributes (clutch, consistency, durability) that ratings alone miss.

**Scaling:** Bumped from ±25 to ±40 cap and from 400 to 600 coefficient (batting) / 20 to 30 (pitching) based on the extraordinary correlation strength.

---

## 9. Recommendation Engine

### 9.1 Buy Recommendation Algorithm

Three phases executed in sequence:

**Phase 1 — Roster gap analysis:**
For each position (C, 1B, 2B, 3B, SS, LF, CF, RF, DH, SP×5, RP×4, CL):
1. Identify current starter's meta score
2. Query market for affordable upgrades: `meta > current_meta + min_improvement AND sell_order_low <= max_spend`
3. Score by value ratio: `meta² / price` (quadratic rewards elite cards)
4. Assign priority 1 (empty slot), 2 (large upgrade), or 3 (moderate upgrade)

**Phase 2 — Multi-position eligibility scan:**
Find cards where `pos_rating >= 100` at a secondary position that upgrades a different roster slot. This catches utility players who improve the team at a non-obvious position.

**Phase 3 — Best-value sweep:**
Top 30 cards across all positions by `meta² / price`, regardless of roster need. Captures market inefficiencies where high-meta cards are underpriced.

### 9.2 Sell Recommendation Algorithm

Categorized detection:
1. **Duplicates:** Cards with `owned > 1` — sell the excess
2. **Off-roster:** Owned cards not in any lineup slot — potential dead weight
3. **Outclassed:** Cards worse than the current starter at the same position — no reason to hold
4. **Underperformers:** Cards with meta > 600 but OPS < .650 (batters) or meta > 400 but ERA > 5.00 (pitchers) — ratings not translating to production

### 9.3 Flip Finder

Three arbitrage strategies:
1. **Spread flips:** Buy order < sell order by meaningful margin
2. **Undervalued:** High `meta / price` ratio cards the market hasn't priced correctly
3. **Tier arbitrage:** Cards priced below their tier's typical price floor

---

## 10. Presentation Layer

### 10.1 Page Architecture (15 pages across 5 groups)

**Start Here:**
- **Data Refresh:** Watch folder scanner with category-grouped preview, league auto-selection (persists to config.yaml), streaming progress bar, Refresh All button at both top and bottom for convenience, post-run summary with per-file results

**Decide (6 pages):**
- **Buy Recommendations:** Budget investment advisor with three spending scenarios (One Big Buy, Two Balanced, Max Efficiency). Budget slider, position/tier filters. Meta explainer breakdowns showing each formula component. Sparkline price trends inline
- **Sell Recommendations:** Categorized sell targets (duplicates, off-roster, outclassed, underperformers) with selection checkboxes and running PP total
- **Roster Optimizer:** Position-by-position depth chart with platoon splits (vs RHP/LHP), bench upgrade recommendations, AI Manager's Eye team assessment
- **Tournament Builder:** Salary-cap constrained roster optimization for PT tournaments
- **Export Plan:** Downloadable action plan document with prioritized buy/sell targets
- **Flip Finder:** Spread, undervalued, and tier arbitrage opportunities

**Analyze (4 pages):**
- **Game Stats:** Statcast-inspired analytics — OPS vs WAR scatter, ERA vs FIP luck chart, K% vs BB% dominance chart, meta vs reality validation, calibration status with per-type R²/r chips
- **Price Trends:** Multi-card price overlay charts, buy/sell order history, price anomaly detection
- **Trends & History:** Meta movers, market trends, per-player trend charts across time
- **Mission Tracker:** Team mission completion progress, best buys to complete missions

**Library (2 pages):**
- **Card Detail:** Deep single-card analysis — full meta breakdown with "Why this score?" explainer, price history chart, platoon splits visualization, comparable cards, fielding ratings
- **Live Card Tracker:** MLB Live card upgrade/downgrade prediction based on real-world stats vs OOTP ratings

**Configure (2 pages):**
- **AI Advisor:** Claude/Gemini-powered chat interface for strategic analysis, trade advice, market timing
- **Settings:** Meta weight presets and custom tuning, watch folder path, database management (backup/export/reset), meta recalculation trigger

### 10.2 Meta Explainer UX

Every meta score in the UI can be expanded to show a structured breakdown:

```
Components (sorted by contribution):
  Contact (82 effective)  × 1.80  = 147.6
  Power (91 effective)    × 2.00  = 182.0
  OVR (85)                × 2.00  = 170.0
  BABIP (78)              × 1.20  =  93.6
  Defense (72 × 1.10)     × 0.40  =  31.7
  Eye (68 effective)      × 0.80  =  54.4
  Gap Power (75)          × 1.20  =  90.0

Bonuses:
  Speed/Stealing (12)              =   7.2
  Position scarcity (SS)           = +19.0
  wOBA performance (.358)          = +22.4
  Multi-position flexibility       =  +4.5

Penalties:
  Platoon split weakness           = -11.2

Notes:
  Contact: 80 overall -> 82 effective (platoon-weighted)
  Weak-side exposure: Contact: 65 vL / 95 vR
  Premium defensive position (SS) gets +19 fWAR-ladder bonus
```

---

## 11. AI Integration

### 11.1 Gemini Integration (Primary)

Model: `gemini-2.5-flash`. Used for:
- **Manager's Eye:** Full-roster team assessment with strengths/weaknesses analysis
- **AI Advisor chat:** Strategic Q&A about market timing, trade advice, lineup construction
- **AI Insights:** Passive background generation of buy/sell recommendations after each refresh

### 11.2 Claude Integration (Secondary)

Model: Anthropic Claude. Available as alternate AI provider in Settings. Same feature set as Gemini.

---

## 12. Current Limitations & Research Opportunities

### 12.1 Known Limitations

1. **Calibration sample size:** 200-280 players per fit type. CV R² is structurally low (0.03-0.16) because WAR has a large irreducible per-card noise floor. Need 500+ players per role for robust interaction term estimation.

2. **Defense signal:** Individual defense ratings show r≈0 with WAR. This could mean (a) OOTP PT doesn't reward defense in WAR, (b) our defense formula (position-weighted average) is wrong, or (c) the sample is too noisy. The fielding_stats table (DRS, UZR equivalents) has never been incorporated into the meta — this is a potential signal source.

3. **Arsenal diversity:** No WAR correlation found. Could be (a) OOTP doesn't model lineup-trip-through-order familiarity, (b) the pitch_count metric is too crude, or (c) arsenal quality is already captured by Stuff rating.

4. **Park factor adjustment:** Park factors are ingested and stored but not used in meta scoring. A batter playing in a hitter's park (pf_hr > 1.10) should have their power production discounted relative to a neutral park.

5. **Market efficiency:** Buy/sell spread, variance, and liquidity data exist but aren't used in meta scoring. They affect recommendation prioritization but not the meta itself.

### 12.2 High-Priority Research Opportunities

1. **Multivariate regression with expanded feature set:** The current calibrator uses 4-5 features. Expanding to include platoon split gap magnitude, wOBA residual (wOBA - expected_wOBA_from_ratings), and fielding DRS would test whether these signals survive in a multivariate context.

2. **Non-linear modeling:** The linear + interaction architecture may miss non-linear relationships. A gradient boosted model (XGBoost/LightGBM) trained on all available features could establish an upper bound on predictability and identify which non-linearities matter most.

3. **Fielding stats integration:** The `fielding_stats` table contains per-position DRS/UZR-equivalent data. Correlating these actual defensive results (rather than rating predictions) with WAR could reveal whether OOTP rewards defense differently than the ratings predict.

4. **Temporal dynamics:** Player history snapshots enable studying how meta scores evolve over a season. Are early-season wOBA leaders sustained? Do calibrated weights from early-season hold up in late-season validation?

5. **Cross-league validation:** Data from league i76 exists alongside lb124. Running the same correlation analysis on i76 would test whether the weight structure generalizes across leagues or is lb124-specific.

6. **Platoon split value quantification:** The current platoon penalty uses theoretically-derived coefficients. A regression specifically targeting platoon gap magnitude × WAR would empirically calibrate the penalty strength.

7. **Park-factor-adjusted meta:** Incorporating pf_hr and pf_avg from league_team_stats into the batting meta would normalize for park effects, potentially improving WAR prediction by 2-5% (estimated based on real baseball park factor impact).

8. **Ensemble approach:** Combining the rating-based meta (current system) with a pure-performance meta (wOBA/SIERA-based) via a weighted ensemble could capture both the "what should this card do" and "what is this card actually doing" signals simultaneously.

---

## Appendix A: Configuration Reference

```yaml
# config.yaml
active_league: lb124          # Auto-persisted after each refresh
ai_provider: gemini           # 'gemini' or 'claude'
batting_weights:              # Calibration-aware weights (see Section 7.4)
  power: 2.0
  contact: 1.8
  babip: 1.2
  gap_power: 1.2
  eye: 0.8
  defense: 0.4
  speed_stealing: 0.15
  avoid_ks: 0.0
pitching_weights:
  stuff: 2.4
  p_hr: 1.4
  movement: 0.8
  control: 0.3
  stuff_x_control: 0.010
  stuff_x_movement: 0.008
  movement_x_control: 0.002
  stamina_hold: 0.1
pp_budget: 2000               # Perfect Points budget for recommendations
recommendations:
  max_budget_pct: 0.8
  min_meta_improvement: 10
  value_ratio_threshold: 50.0
team_name: Toronto Dark Knights
watch_directory: C:\Users\Cameron\OneDrive\Documents\Out of the Park Developments\OOTP Baseball 27\online_data
```

## Appendix B: Database Entity-Relationship Summary

```
cards (PK: card_id)
  ├── batting_stats (FK: card_id, partitioned by league_id)
  ├── pitching_stats (FK: card_id, partitioned by league_id)
  ├── batting_stats_adv (FK: card_id)
  ├── pitching_stats_adv (FK: card_id)
  ├── pitch_ratings (FK: card_id)
  ├── fielding_stats (FK: card_id)
  ├── price_snapshots (FK: card_id)
  ├── recommendations (FK: card_id)
  └── player_history (FK: card_id)

roster (lineup_role partitions: 'starter'/'bench'/'league')
  └── my_collection

meta_calibration (calibration_type: batting/pitching_sp/pitching_rp/pitching)

league_team_stats (PK: league_id + team_name + snapshot_date)

export_log / ingestion_log (audit trail)
```

---

*Document generated 2026-04-16. For questions or collaboration inquiries, reference this document as "OOTP-PTO-ARCH-v5.0".*
