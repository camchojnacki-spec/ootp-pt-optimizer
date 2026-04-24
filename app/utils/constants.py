"""Constants for OOTP PT Optimizer."""

POSITION_MAP = {
    1: "P", 2: "C", 3: "1B", 4: "2B", 5: "3B",
    6: "SS", 7: "LF", 8: "CF", 9: "RF", 10: "DH"
}

POSITION_TO_NUM = {v: k for k, v in POSITION_MAP.items()}

PITCHER_ROLE_MAP = {11: "SP", 12: "RP", 13: "CL"}

TIER_MAP = {
    1: "Regular", 2: "Bronze", 3: "Silver",
    4: "Gold", 5: "Diamond", 6: "Perfect"
}

TIER_TO_NUM = {v: k for k, v in TIER_MAP.items()}

# Default meta score weights — v2 (research-paper-informed)
# OVR REMOVED: causes structural multicollinearity (VIF >> 10) because
# OVR is derived from the component ratings. Including it poisons the
# regression and makes weight estimates unstable. (See research brief §1)
#
# Weights are starting priors for Elastic Net calibration.
# Derived from correlation analysis of 209 batters in league lb124 (2026-04-16).
# Correlations measured vs WAR/600PA with PA >= 150 sample filter.
DEFAULT_BATTING_WEIGHTS = {
    "power": 2.00,           # r=+0.289 WAR — strongest individual rating in lb124
    "contact": 1.80,         # r=+0.270 WAR — 2nd strongest, was overweighted at 2.0
    "babip": 1.20,           # r=+0.318 WAR — 3rd strongest! NOT double-counted (lb124 data)
    "gap_power": 1.20,       # r=+0.183 WAR — moderate, was overweighted at 1.6
    "eye": 0.80,             # r=+0.160 WAR — weak but real signal
    # 2026-04-17 v6b: Residual analysis on pooled lb124+i76 (n=1103) showed
    # avoid_ks has r=+0.138 with meta residual (the predictive-power left over
    # after meta). The univariate WAR correlation was dead, but once meta
    # absorbs contact/power/eye, the *remaining* WAR variance tracks avoid_ks.
    # Bumping from 0.00 → 0.60 so the signal is captured.
    "avoid_ks": 0.60,
    "defense": 0.40,         # r=+0.046 WAR, p=0.51 — OOTP overvalues defense; was 1.5!
    # 2026-04-20 attribute-mix residual analysis (n=2385 PA>=150 pooled
    # lb124+i76) showed speed still carries r=+0.145 on the meta residual
    # AND (speed+baserunning)/2 carries r=+0.163. Adding speed to the meta
    # lifts meta r from +0.7294 to +0.7381 — a genuine independent signal
    # not absorbed by observed-stat overlays. Bumping 0.30 → 0.45 moves
    # weight ~50% closer to regression-implied slope (~0.008 WAR per
    # rating point * 78 meta/WAR = 0.62, but capping to preserve stability).
    "speed_stealing": 0.45,
}

# Position-specific defense multipliers — how much defense matters by position.
# Rebased on fWAR positional adjustment ladder (runs/162 games):
#   C +12.5, SS +7.5, 2B/3B/CF +2.5, LF/RF -7.5, 1B -12.5, DH -17.5
# Normalized so SS = 1.40 (top of defensive spectrum).
POSITION_DEFENSE_MULTIPLIERS = {
    2: 1.30,   # C  — +12.5 runs, framing/arm hugely valuable, premium position
    3: 0.40,   # 1B — -12.5 runs, lowest defensive spectrum
    4: 1.10,   # 2B — +2.5 runs, middle infield
    5: 1.00,   # 3B — +2.5 runs, hot corner
    6: 1.40,   # SS — +7.5 runs, highest defensive spectrum
    7: 0.60,   # LF — -7.5 runs, least demanding OF spot
    8: 1.25,   # CF — +2.5 runs, premium OF, range critical
    9: 0.70,   # RF — -7.5 runs, arm matters but less demanding than CF
    10: 0.00,  # DH — -17.5 runs, no defense
}

# Positional value adjustment — added to meta REGARDLESS of defense quality.
# A 500-meta SS is worth more than a 500-meta 1B due to positional scarcity.
#
# 2026-04-17 v2: Rebalanced based on residual analysis (n=1103 pooled
# lb124 + i76). The fWAR ladder values were correct in *shape* but too
# extreme in this game's WAR output:
#   C residual  = -0.39 WAR/600 (***) — meta OVERvalues catchers → reduce bonus
#   1B residual = +0.27 WAR/600 (***) — meta too harsh → lighten penalty
#   3B residual = +0.28 WAR/600 (***) — 3B is undervalued at baseline → positive bonus
#   2B residual = +0.20 WAR/600 (*)   — slight bump
#   RF residual = -0.27 WAR/600 (*)   — overvalued → more negative
# Values still respect the fWAR spectrum (SS/C premium > 2B/3B > corners > DH).
POSITIONAL_VALUE_BONUS = {
    2: 10,    # C  — was +31, residual said too generous
    3: -10,   # 1B — was -31, residual said too harsh
    4: 14,    # 2B — was +6, slight bump (+0.20 residual)
    5: 19,    # 3B — was 0, residual +0.28 says undervalued
    6: 19,    # SS — unchanged; SS residual was -0.12 (not significant)
    7: -5,    # LF — was -19, residual near zero
    8: 6,     # CF — unchanged; residual near zero
    9: -19,   # RF — was -19, residual -0.27 confirmed overvalue
    10: -44,  # DH — unchanged; no direct residual data (DH is rare in PT rosters)
}

DEFAULT_PITCHING_WEIGHTS = {
    # 2026-04-17 v6b: Residual analysis (n=1228 pooled) showed:
    #   stuff residual r=-0.09 *  (slightly overweighted at 2.40 → trim)
    #   control residual r=+0.15 *** (undervalued → raise)
    #   stamina residual r=+0.18 *** (undervalued → raise)
    #   movement_x_control residual r=+0.13 ** (interaction underweighted)
    "stuff": 2.20,           # trimmed from 2.40 per residual
    "movement": 0.80,        # r=+0.123 WAR, p=0.068 — unchanged
    "control": 0.50,         # was 0.30; residual r=+0.15 *** — real undervalue
    "p_hr": 1.40,            # r=+0.142 WAR — significant for SP
    "stamina_hold": 0.30,    # was 0.10; residual r=+0.18 *** — matters for SP durability
    # Interaction terms — SIERA-style synergies
    "stuff_x_movement": 0.008,
    "stuff_x_control": 0.010,
    "movement_x_control": 0.004,  # was 0.002; residual r=+0.13 ** says raise
}

# Role-specific pitching defaults — used when no calibration row exists for
# the role in meta_calibration. 2026-04-20 residual analysis on RP-only
# sample (n=1142 IP>=30) found SHARPLY different residuals than the pooled
# pitching fit:
#   RP stuff residual r=-0.276 *** (SIGNIFICANTLY over-weighted)
#   RP control residual r=+0.205 *** (significantly under-weighted)
#   RP stamina residual r=+0.149 ** (under-weighted; matters for multi-inning RP)
#   RP movement_x_control residual r=+0.177 *** (the junk-ball pair is very real)
#   RP stuff_x_movement residual r=-0.166 (the "all-stuff" pair over-predicts)
# Calibration's combined pitching fit diluted these signals because SP is
# double the sample; explicit RP defaults preserve the role-specific shape.
DEFAULT_PITCHING_WEIGHTS_RP = {
    "stuff": 1.40,           # pooled 2.20 → 1.40 (residual -0.276 ***)
    "movement": 0.80,        # unchanged
    "control": 1.10,         # pooled 0.50 → 1.10 (residual +0.205 ***)
    "p_hr": 1.40,            # unchanged, still a strong signal
    "stamina_hold": 0.50,    # pooled 0.30 → 0.50 (residual +0.149 **)
    "stuff_x_movement": 0.004,  # pooled 0.008 → 0.004 (residual -0.166)
    "stuff_x_control": 0.010,   # unchanged
    "movement_x_control": 0.008,  # pooled 0.004 → 0.008 (residual +0.177 ***)
}

# SP-specific defaults match the combined defaults closely — SP dominates the
# pooled sample, so there's no signal separating them. Kept as an explicit
# dict so future role-specific tuning has a natural home.
DEFAULT_PITCHING_WEIGHTS_SP = dict(DEFAULT_PITCHING_WEIGHTS)

# ══════════════════════════════════════════════════════════════════════════
# Card-type meta offsets (derived 2026-04-17 from residual analysis,
# n=1103 batters / 1228 pitchers pooled across lb124 + i76)
#
# The residual analysis fit `WAR ~ meta` (linear), then computed the mean
# residual per card_type. Systematic bias in a card type means the formula
# doesn't capture something type-specific (e.g., Snapshots tend to be
# peak-season reproductions, All-Time Legends come from a prestige pool).
# We convert residual WAR units to meta units via the inverse slope of
# the baseline fit:
#   batting:  1 WAR/600 = ~69 meta  (slope 0.01444 meta→WAR)
#   pitching: 1 WAR/200 = ~91 meta  (slope 0.01103 meta→WAR)
#
# Card_type integer mapping (OOTP internal):
#   1 Live, 2 Legend, 3 Hardware Heroes, 4 Unsung Heroes,
#   5 Historical All-Star, 6 Future Legend, 7 Snapshot,
#   8 Veteran Presence, 9 All-Time Legend, 10 Live Reward
#
# Only types with |residual| > SE (at least one-star significance) get
# an offset. Small-n groups (<20) are damped 50% to avoid overfitting.
# ══════════════════════════════════════════════════════════════════════════
BATTING_CARD_TYPE_OFFSET = {
    1: -14,   # Live   (n=561, residual -0.20 ***)
    2:   0,   # Legend (n=92, residual +0.03, not significant)
    3: -16,   # Hardware Heroes (n=26, residual -0.47 *, damped)
    4:   5,   # Unsung Heroes  (n=23, residual +0.15, damped)
    5:   0,   # Historical All-Star (n=52, residual -0.14, not significant at SE)
    6:   0,   # Future Legend (no data yet)
    7:  24,   # Snapshot (n=233, residual +0.35 ***)
    8:   0,   # Veteran Presence (n=31, residual +0.14, not significant)
    9:  44,   # All-Time Legend (n=41, residual +0.64 ***)
    10: 19,   # Live Reward (n=42, residual +0.28 *)
}
PITCHING_CARD_TYPE_OFFSET = {
    1:   0,   # Live   (n=775, residual +0.00)
    2:  65,   # Legend (n=40, residual +0.72 ***)
    3: -76,   # Hardware Heroes (n=62, residual -0.84 ***)
    4:  23,   # Unsung Heroes  (n=22, residual +0.51 *, damped)
    5:  15,   # Historical All-Star (n=16, residual +0.16, damped not significant)
    6:   0,   # Future Legend (no data)
    7:   5,   # Snapshot (n=236, residual +0.06, weak)
    8: -36,   # Veteran Presence (n=16, residual -0.79 *, damped)
    9: -30,   # All-Time Legend (n=31, residual -0.33 *)
    10: 24,   # Live Reward (n=30, residual +0.53 *, damped)
}

# Minimum floor for key stats — cards below this get penalized
PITCHING_STAT_FLOOR = 65
BATTING_STAT_FLOOR = 55

# File pattern matching for CSV identification.
# NOTE: identify_file_type() sorts by pattern length descending so more
# specific substrings win. Example: a team-level "team_statistics___info_-_
# sortable_stats_batting_stats" file must not be confused with the
# player-level "sortable_stats_batting_stats" pattern.
FILE_PATTERNS = {
    "market": "pt_card_list",
    "roster_batting": "rosters_-_player_list_batting_ratings",
    "roster_pitching": "rosters_-_player_list_pitching_ratings",
    "collection_batting": "collection_-_manage_cards_collection_-_manage_cards_batting_ratings",
    "collection_pitching": "collection_-_manage_cards_collection_-_manage_cards_pitching_ratings",
    # Collection default view — recognized so staleness can detect "user
    # clicked Export but forgot to switch to Batting Ratings / Pitching
    # Ratings views." The columns (EYE/CON, FLD/STA) are aggregates, not
    # the per-skill ratings meta scoring needs, so we don't ingest it —
    # just use its mtime as evidence the user tried to refresh.
    "collection_default": "collection_-_manage_cards_collection_-_manage_cards_default",
    # League-wide TEAM stats (framing context — one row per team).
    # These MUST be listed before the player-level stats_batting/pitching
    # patterns so longest-match identification picks them up.
    "team_stats_batting": "team_statistics___info_-_sortable_stats_batting_stats",
    "team_stats_pitching": "team_statistics___info_-_sortable_stats_pitching_stats",
    # Team-level fielding stats (Team Name, G, IP, PO, A, DP, E, PCT, ZR).
    # Distinct schema from the player-level fielding_stats file — the
    # longer pattern wins so the player handler doesn't trip on Team Name.
    # Currently a recognized-noop; no consumer feature yet.
    "team_stats_fielding": "team_statistics___info_-_sortable_stats_fielding_stats",
    "team_stats_park": "sortable_stats_park_info",
    # Player-level stats (league-wide player rows).
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
    # fielding_ratings MUST come before fielding_stats — longer pattern wins.
    # This is OOTP 27's new per-player fielding ratings export (C ABI, IF RNG,
    # OF ARM, etc.) — separate from the older position_ratings export (which
    # has DEF, P, C, 1B… per-position ability ladders).
    "fielding_ratings": "fielding_ratings",
    "fielding_stats": "fielding_stats",
    "position_ratings": "position_ratings",
    "pitch_ratings": "individual_pitch_ratings",
    # Team-level "default" view = standings (W/L/pct/GB/streak). We recognize
    # it so the refresh UI doesn't flag it as unknown, but the handler is
    # currently a noop — league_team_stats doesn't have standings columns yet
    # and adding them is a migration we can defer until a consumer exists.
    "team_standings": "team_statistics___info_-_sortable_stats_default",
    # League-wide player DEFAULT view — this is the only export that carries
    # the TM (short team name) + LG columns needed for card-ownership
    # attribution. Paired row-for-row with stats_batting_ratings /
    # stats_pitching_ratings. Feeds the `league_rosters` table.
    "league_player_default": "player_statistics_-_sortable_stats_default",
}
