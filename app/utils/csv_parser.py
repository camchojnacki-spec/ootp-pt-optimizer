"""Handle OOTP CSV parsing quirks."""
import pandas as pd
from pathlib import Path
from app.utils.constants import FILE_PATTERNS


def identify_file_type(filepath: str) -> str | None:
    """Identify CSV file type from filename.

    Uses longest-match semantics: when multiple patterns match the same
    filename, the longest (most specific) pattern wins. This prevents e.g.
    a team-level "...sortable_stats_batting_stats.csv" file from being
    misidentified as the player-level `stats_batting` type when a more
    specific `team_stats_batting` pattern is present.
    """
    name = Path(filepath).name.lower()
    best_type: str | None = None
    best_len = -1
    for file_type, pattern in FILE_PATTERNS.items():
        pat = pattern.lower()
        if pat in name and len(pat) > best_len:
            best_type = file_type
            best_len = len(pat)
    return best_type


def parse_market_csv(filepath: str) -> pd.DataFrame:
    """Parse pt_card_list.csv, handling the // header prefix."""
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()

    # Strip // prefix from header
    if first_line.startswith('//'):
        import io
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        content = content[2:]  # Remove leading //
        df = pd.read_csv(io.StringIO(content), index_col=False)
    else:
        df = pd.read_csv(filepath, index_col=False)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Fill NaN numeric fields with 0
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0).astype(int)

    return df


def parse_roster_batting_csv(filepath: str) -> pd.DataFrame:
    """Parse roster batting ratings CSV."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def parse_roster_pitching_csv(filepath: str) -> pd.DataFrame:
    """Parse roster pitching ratings CSV."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def parse_collection_batting_csv(filepath: str) -> pd.DataFrame:
    """Parse collection batting ratings CSV."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    # Drop Actions column if present
    if 'Actions' in df.columns:
        df = df.drop(columns=['Actions'])
    return df


def parse_collection_pitching_csv(filepath: str) -> pd.DataFrame:
    """Parse collection pitching ratings CSV."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    if 'Actions' in df.columns:
        df = df.drop(columns=['Actions'])
    return df


def parse_stats_batting_csv(filepath: str) -> pd.DataFrame:
    """Parse sortable batting stats CSV export.

    Expected columns: POS, #, Name, Inf, B, T, G, PA, AB, H, 2B, 3B, HR,
    RBI, R, BB, IBB, HP, K, GIDP, AVG, OBP, SLG, ISO, OPS, OPS+, BABIP, WAR, SB, CS
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def parse_stats_pitching_csv(filepath: str) -> pd.DataFrame:
    """Parse sortable pitching stats CSV export.

    Expected columns: POS, #, Name, Inf, B, T, G, GS, W, L, SV, HLD, IP,
    HA, HR, R, ER, BB, K, HP, ERA, AVG, BABIP, WHIP, HR/9, BB/9, K/9, K/BB, ERA+, FIP, WAR
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def parse_stats_batting_adv_csv(filepath: str) -> pd.DataFrame:
    """Parse batting stats_2 CSV (advanced metrics).

    Expected columns: POS, #, Name, Inf, B, T, G, PA, BB, BB%, SH, SF, CI,
    K, K%, GIDP, EBH, TB, RC, RC/27, ISO, wOBA, WPA, PI/PA
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def parse_stats_pitching_adv_csv(filepath: str) -> pd.DataFrame:
    """Parse pitching stats_2 CSV (advanced metrics).

    Expected columns: POS, #, Name, Inf, B, T, G, WIN%, SV%, BS, SD, MD, IP,
    BF, DP, RA, GF, IR, IRS%, pLi, QS, QS%, CG, CG%, SHO, PPG, RSG, GO%, SIERA, SB, CS, WPA
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def parse_lineup_csv(filepath: str) -> pd.DataFrame:
    """Parse team lineup CSV (vs RHP, vs LHP, or overview).

    Expected columns: POS, #, Name, Inf, Age, NAT, HT, WT, B, T, OVR, CTM, CType, Title
    The Title column is key — it contains the exact card title for precise card matching.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def parse_team_pitching_csv(filepath: str) -> pd.DataFrame:
    """Parse team pitching roster CSV.

    Same format as lineup CSV but for pitchers. Shows full staff including reserve.
    Expected columns: POS, #, Name, Inf, Age, NAT, HT, WT, B, T, OVR, CTM, CType, Title
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def parse_league_batting_ratings_csv(filepath: str) -> pd.DataFrame:
    """Parse league-wide batting ratings CSV.

    Same format as roster batting ratings but for all league players.
    Expected columns: POS, #, Name, Inf, Age, B, T, OVR, CON, BABIP, K's, GAP, POW, EYE,
    CON vL, POW vL, EYE vL, CON vR, POW vR, EYE vR, BUN, BFH, SPE, STE, DEF, St
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def parse_league_pitching_ratings_csv(filepath: str) -> pd.DataFrame:
    """Parse league-wide pitching ratings CSV.

    Same format as roster pitching ratings but for all league players.
    Expected columns: POS, #, Name, Inf, Age, B, T, OVR, STU, MOV, HRA, PBABIP, CON,
    STU vL, STU vR, VELO, STM, G/F, HLD, St
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def parse_league_player_default_csv(filepath: str) -> pd.DataFrame:
    """Parse the league-wide player `default` CSV.

    Expected columns: POS, #, Name, Inf, TM, LG, Lev, DOB, Age, NAT, HT, WT, B, T, OVR

    This is the only league-wide player export that carries the TM (short team
    name) + LG columns. One row per card-instance-per-team, aligned row-for-row
    with the ratings files from the same league/date.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def parse_fielding_stats_csv(filepath: str) -> pd.DataFrame:
    """Parse fielding stats CSV export.

    Expected columns: POS, #, Name, Inf, B, T, G, GS, TC, A, PO, E, DP, PCT,
    RNG, ZR, EFF, SBA, RTO, RTO%, IP, PB, CERA, FRM, ARM
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')
    df.columns = df.columns.str.strip()
    # Filter out empty rows
    df = df.dropna(subset=['Name'], how='all')
    return df


def parse_position_ratings_csv(filepath: str) -> pd.DataFrame:
    """Parse position ratings CSV export.

    Expected columns: POS, #, Name, Inf, Age, B, T, DEF, P, C, 1B, 2B, 3B, SS, LF, CF, RF, St
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Name'], how='all')
    return df


def parse_team_stats_batting_csv(filepath: str) -> pd.DataFrame:
    """Parse league-wide team batting stats CSV (one row per team).

    Expected columns: Team Name, G, 2B, 3B, HR, R, BB, SO, SB, AVG, OBP, SLG, OPS
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Team Name'])
    return df


def parse_team_stats_pitching_csv(filepath: str) -> pd.DataFrame:
    """Parse league-wide team pitching stats CSV (one row per team).

    Expected columns: Team Name, W, L, SV, IP, R, HR, BB, K, WHIP, AVG, ERA, FIP-
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Team Name'])
    return df


def parse_team_stats_park_csv(filepath: str) -> pd.DataFrame:
    """Parse league-wide team park factor CSV (one row per team).

    Expected columns: Team Name, Park, PF AVG, AVG L, AVG R, PF HR, HR L, HR R, PF D, PF T, PF
    Park factors are centered at 1.000 (neutral). >1 = hitter-friendly.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Team Name'])
    return df


def parse_pitch_ratings_csv(filepath: str) -> pd.DataFrame:
    """Parse individual pitch ratings CSV export.

    Expected columns: POS, #, Name, Inf, Age, T, FB, CH, CB, SL, SI, SP, CT, FO,
    CC, SC, KC, KN, PIT, VELO, Slot, STM
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Name'], how='all')
    return df


def parse_fielding_ratings_csv(filepath: str) -> pd.DataFrame:
    """Parse player fielding ratings CSV export (OOTP 27+).

    Expected columns: POS, #, Name, Inf, Age, B, T, C ABI, C ARM, IF RNG,
    IF ERR, IF ARM, TDP, OF RNG, OF ERR, OF ARM, St

    Note the spaces in column names ('C ABI', 'IF RNG'...) — the handler
    reads them via row.get('C ABI') etc., not via pandas attribute access.
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Name'], how='all')
    return df
