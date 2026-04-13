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
# Derived from correlation analysis of 678 batters / 633 pitchers in league i76.
DEFAULT_BATTING_WEIGHTS = {
    "gap_power": 1.60,      # r=+0.205 WAR, r=+0.212 OPS — solid extra-base hit proxy
    "contact": 2.00,         # r=+0.314 WAR — strongest individual rating
    "avoid_ks": 0.00,        # double-counted in CON (OOTP25+ derived stat)
    "eye": 0.80,             # r=+0.063 WAR — weak alone but OBP multiplier
    "power": 1.60,           # r=+0.275 OPS — strongest OPS predictor
    "babip": 0.00,           # double-counted in CON (OOTP25+ derived stat)
    "defense": 1.50,         # r=+0.296 WAR — scaled by position multiplier
    "speed_stealing": 0.50,  # Speed→SB r=+0.337, conditional value (amplifies OBP)
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
# Values are runs/162 from fWAR ladder, scaled to meta points (~2.5 meta per run).
POSITIONAL_VALUE_BONUS = {
    2: 31,    # C  — +12.5 runs × 2.5
    3: -31,   # 1B — -12.5 runs × 2.5
    4: 6,     # 2B — +2.5 runs × 2.5
    5: 0,     # 3B — baseline
    6: 19,    # SS — +7.5 runs × 2.5
    7: -19,   # LF — -7.5 runs × 2.5
    8: 6,     # CF — +2.5 runs × 2.5
    9: -19,   # RF — -7.5 runs × 2.5
    10: -44,  # DH — -17.5 runs × 2.5
}

DEFAULT_PITCHING_WEIGHTS = {
    "movement": 2.20,        # r=-0.295 ERA — strongest ERA predictor
    "stuff": 1.60,           # r=-0.265 ERA — 2nd strongest for ERA
    "control": 0.60,         # Was 0.20, but SIERA validates control matters more
    "p_hr": 1.80,            # r=-0.266 ERA, r=+0.242 WAR
    "stamina_hold": 0.40,    # r=+0.392 WAR but confounded (SP vs RP)
    # Interaction terms (SIERA precedent — pitching is non-additive)
    # Scaled so max interaction bonus ~80 meta for elite (120×110) vs ~30 for avg (75×75)
    "stuff_x_movement": 0.006,   # K% × GB% analog — Ks + weak contact = devastating
    "stuff_x_control": 0.004,    # Dominant stuff + command = elite
    "movement_x_control": 0.003, # Groundballs + fewer walks = value when runners on
}

# Minimum floor for key stats — cards below this get penalized
PITCHING_STAT_FLOOR = 65
BATTING_STAT_FLOOR = 55

# File pattern matching for CSV identification
FILE_PATTERNS = {
    "market": "pt_card_list",
    "roster_batting": "rosters_-_player_list_batting_ratings",
    "roster_pitching": "rosters_-_player_list_pitching_ratings",
    "collection_batting": "collection_-_manage_cards_collection_-_manage_cards_batting_ratings",
    "collection_pitching": "collection_-_manage_cards_collection_-_manage_cards_pitching_ratings",
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
