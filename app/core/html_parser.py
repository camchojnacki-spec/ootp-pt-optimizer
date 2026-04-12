"""Parse OOTP HTML report files into DataFrames."""
import pandas as pd
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)

# Path to saved game HTML directory
SAVED_GAME_HTML_BASE = (
    r"C:\Users\Cameron\OneDrive\Documents\Out of the Park Developments"
    r"\OOTP Baseball 27\saved_games\7ea0000000000000000002ea.pt\news\html"
)


def get_html_base_path() -> Path:
    """Get the base path for HTML game data."""
    return Path(SAVED_GAME_HTML_BASE)


def parse_html_table(filepath: str) -> list[pd.DataFrame]:
    """Parse HTML file and extract all sortable tables as DataFrames.

    Uses pandas.read_html which handles the HTML table parsing robustly.
    Returns a list of DataFrames (one per table found).
    """
    try:
        dfs = pd.read_html(filepath, flavor='lxml')
        # Clean up: strip whitespace from string columns
        for i, df in enumerate(dfs):
            df.columns = [str(c).strip() for c in df.columns]
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].str.strip() if hasattr(df[col], 'str') else df[col]
            dfs[i] = df
        return dfs
    except Exception as e:
        logger.error(f"Error parsing HTML {filepath}: {e}")
        return []


def parse_standings(filepath: str = None) -> pd.DataFrame | None:
    """Parse league standings HTML. Returns DataFrame with Team, W, L, PCT, GB, etc."""
    if filepath is None:
        filepath = str(get_html_base_path() / "leagues" / "league_100_standings.html")

    if not Path(filepath).exists():
        return None

    dfs = parse_html_table(filepath)

    # Find the table(s) with standings data (has W, L, PCT columns)
    standings_dfs = []
    for df in dfs:
        cols = [str(c).upper() for c in df.columns]
        if 'W' in cols and 'L' in cols and 'PCT' in cols:
            standings_dfs.append(df)

    if not standings_dfs:
        return None

    # Concatenate all division standings
    result = pd.concat(standings_dfs, ignore_index=True)

    # Clean team names (remove any HTML artifacts)
    if 'Team' in result.columns:
        result['Team'] = result['Team'].astype(str).str.strip()

    return result


def parse_team_batting_stats(filepath: str = None) -> pd.DataFrame | None:
    """Parse team batting stats HTML. Returns DataFrame with player performance stats."""
    if filepath is None:
        filepath = str(get_html_base_path() / "teams" / "team_26_batting_stats_0_1.html")

    if not Path(filepath).exists():
        return None

    dfs = parse_html_table(filepath)

    # Find the table with batting stats (has AB, H, AVG columns)
    for df in dfs:
        cols = [str(c).upper() for c in df.columns]
        if 'AB' in cols and 'H' in cols and 'AVG' in cols:
            # Clean player names - extract name and position from "Name POS" format
            name_col = None
            for c in df.columns:
                if c.upper() in ('NAME', 'PLAYER'):
                    name_col = c
                    break
            if name_col is None:
                # First text column is likely the name
                for c in df.columns:
                    if df[c].dtype == object:
                        name_col = c
                        break

            if name_col:
                # Parse "PlayerName POS" into separate columns
                df['Player'] = df[name_col].astype(str).apply(
                    lambda x: re.sub(r'\s+(P|C|1B|2B|3B|SS|LF|CF|RF|DH|SP|RP|CL)\s*$', '', x).strip()
                )
                df['POS'] = df[name_col].astype(str).apply(
                    lambda x: m.group(1) if (m := re.search(r'\s+(P|C|1B|2B|3B|SS|LF|CF|RF|DH|SP|RP|CL)\s*$', x)) else ''
                )

            return df

    return None


def parse_sortable_stats_export(filepath: str) -> pd.DataFrame | None:
    """Parse the sortable stats HTML export (from /temp/ directory).

    These have comprehensive stats: POS, Name, G, PA, AB, H, 2B, 3B, HR, RBI, R,
    BB, IBB, HP, K, GIDP, AVG, OBP, SLG, ISO, OPS, OPS+, BABIP, WAR, SB, CS
    """
    if not Path(filepath).exists():
        return None

    dfs = parse_html_table(filepath)

    for df in dfs:
        cols = [str(c).upper() for c in df.columns]
        # Look for the comprehensive stats table
        if 'PA' in cols and 'OPS' in cols and 'WAR' in cols:
            return df

    # Fallback: return largest table
    if dfs:
        return max(dfs, key=len)
    return None


def find_latest_sortable_export() -> str | None:
    """Find the most recent sortable stats export in the temp directory."""
    temp_dir = get_html_base_path() / "temp"
    if not temp_dir.exists():
        return None

    html_files = sorted(temp_dir.glob("*.html"), key=lambda f: f.stat().st_mtime, reverse=True)
    return str(html_files[0]) if html_files else None


def get_my_team_standing(standings_df: pd.DataFrame, team_name: str = "Toronto Dark Knights") -> dict | None:
    """Extract our team's standing from the standings DataFrame."""
    if standings_df is None or standings_df.empty:
        return None

    for _, row in standings_df.iterrows():
        team = str(row.get('Team', ''))
        if team_name.lower() in team.lower():
            return {
                'team': team,
                'wins': int(row.get('W', 0)),
                'losses': int(row.get('L', 0)),
                'pct': str(row.get('PCT', '')),
                'gb': str(row.get('GB', '')),
                'streak': str(row.get('Streak', '')),
                'last10': str(row.get('Last10', '')),
            }
    return None
