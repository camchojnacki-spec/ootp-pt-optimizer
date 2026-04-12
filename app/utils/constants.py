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

# Default meta score weights
# Derived from correlation analysis of 678 batters / 633 pitchers in league i76.
# Key findings:
#   CON->WAR r=+0.314 (strongest individual), DEF->WAR r=+0.296 (2nd),
#   GAP->WAR r=+0.205, POW->OPS r=+0.275, OVR->WAR r=+0.529 (dominant)
#   Note: CON is derived stat in OOTP25+ (combines BABIP + AvK),
#   so AvK/BABIP are zeroed to avoid double-counting.
DEFAULT_BATTING_WEIGHTS = {
    "gap_power": 1.40,      # r=+0.205 WAR, r=+0.212 OPS — solid but not #1
    "contact": 1.80,         # r=+0.314 WAR — strongest individual rating
    "avoid_ks": 0.00,        # double-counted in CON (OOTP25+ derived stat)
    "eye": 0.60,             # r=+0.063 WAR — weak predictor
    "power": 1.40,           # r=+0.275 OPS — strongest OPS predictor
    "babip": 0.00,           # double-counted in CON (OOTP25+ derived stat)
    "defense": 1.50,         # r=+0.296 WAR — 2nd strongest, was massively underweighted
    "ovr": 1.25,             # r=+0.529 WAR — dominant but partially redundant with ratings
}

DEFAULT_PITCHING_WEIGHTS = {
    "movement": 2.40,        # r=-0.295 ERA, r=+0.228 WAR — strongest ERA predictor
    "stuff": 1.40,           # r=-0.265 ERA — 2nd strongest for ERA
    "control": 0.20,         # r=-0.002 ERA — near zero! Was massively overweighted at 1.8
    "p_hr": 1.80,            # r=-0.266 ERA, r=+0.242 WAR — was underweighted at 0.6
    "ovr": 1.50,             # r=+0.534 WAR — dominant
    "stamina_hold": 0.40,    # r=+0.392 WAR but confounded (starters vs relievers)
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
}
