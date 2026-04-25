"""Parse OOTP's news/html exports — starting with box scores.

OOTP saves one HTML file per game under `saved_games/<save_id>/news/html/
box_scores/game_box_<game_id>.html`. Each file has:

- A title bar with league / teams / date
- A linescore table with per-team inning-by-inning runs + final totals
- A batting table per team (AB / R / H / RBI / BB / K / LOB / running AVG / HR / RBI)
- A pitching table per team (IP / H / R / ER / BB / K / HR / PI / ERA + role flags
  like "W (4-3)", "SV (19)")
- A context block: "Game Score", "Batters Faced", "Ground Outs - Fly Outs",
  "Pitches - Strikes"

Parsing these gives us true game-log granularity. The schema lives in
`database.py` — tables `games`, `game_batting`, `game_pitching`.

## Design notes

- `parse_box_score(filepath)` returns a dict with all extracted fields.
  Pure function, no DB side effects. Safe to unit-test offline.
- `ingest_box_score(filepath, conn)` calls `parse_box_score` and writes
  the three rows into the tables. Idempotent on (game_id) — re-running
  overwrites an existing game cleanly.
- `scan_box_score_dir(save_dir)` walks `<save_dir>/news/html/box_scores/`
  and returns the list of HTML paths.
- `execute_refresh` in data_refresh.py accepts an optional `save_dir`
  parameter that triggers box-score ingestion after CSV ingestion.

The parser is deliberately defensive — OOTP's HTML is hand-generated and
occasionally inconsistent. Missing fields are stored as NULL rather than
raising. Stats that can't be parsed as numbers fall through to defaults.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

from app.core.database import get_connection, load_config


def _resolve_game_card_id(
    cursor,
    player_name: Optional[str],
    team_name: Optional[str],
    league_id: Optional[str],
) -> Optional[int]:
    """Resolve an abbreviated box-score name to a card_id.

    Uses the team-scoped resolver (league_rosters-backed) first — that
    disambiguates common-surname cases like "W. Contreras" which exists on
    three different teams. Falls back to the league-wide resolver for
    cases not in league_rosters yet (new call-ups, free-agent pool).

    Returns None rather than raising so a single bad row can't fail ingest.
    """
    if not player_name:
        return None
    try:
        from app.core.name_resolver import resolve_to_card_id_with_team
        conn = getattr(cursor, "connection", None)
        return resolve_to_card_id_with_team(
            player_name, team_name, league_id, prefer_owned=False, conn=conn,
        )
    except Exception as e:
        logger.debug("resolve_to_card_id_with_team failed for %r: %s", player_name, e)
        try:
            from app.core.ingestion import _match_card_id
            return _match_card_id(cursor, player_name)
        except Exception:
            return None

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Low-level helpers
# ══════════════════════════════════════════════════════════════════════

def _ip_to_outs(ip) -> int:
    """Convert OOTP's .1-is-one-out notation to integer outs.

    6.1 IP → 6*3 + 1 = 19 outs. 5.2 IP → 5*3 + 2 = 17.
    Returns 0 for None / unparseable.
    """
    if ip is None:
        return 0
    try:
        ip_f = float(ip)
    except (ValueError, TypeError):
        return 0
    whole = int(ip_f)
    frac = round((ip_f - whole) * 10)
    return whole * 3 + frac


def _safe_int(x, default=0):
    if x is None:
        return default
    try:
        return int(str(x).strip().replace(',', ''))
    except (ValueError, TypeError):
        return default


def _safe_float(x, default=0.0):
    if x is None:
        return default
    try:
        return float(str(x).strip().replace(',', ''))
    except (ValueError, TypeError):
        return default


def _safe_str(x) -> str:
    return str(x).strip() if x is not None else ''


# OOTP player-name links look like "B. Dal Canton SV (19)" — name is the
# text inside the <a>, and anything after is the role annotation.
_ROLE_RE = re.compile(
    r"(?P<flag>[WLSB]V?|SV|BS|H|HLD|CG|SHO)\s*(?:\((?P<num>[\d-]+)\))?",
)


def _extract_role_and_number(trailing_text: str) -> tuple[Optional[str], Optional[int]]:
    """Given text after the player-name anchor (e.g. 'SV (19)' or 'W (4-3)'),
    return (role_flag, role_number). Role_number is the save count / win count
    when parenthesized as a single int; W-L records return the wins only.

    Empty / None input → (None, None).
    """
    if not trailing_text:
        return None, None
    t = trailing_text.strip(' ,')
    if not t:
        return None, None
    # W (4-3) → flag=W, num=4 (wins to-date)
    # SV (19) → flag=SV, num=19
    # BS (1) → flag=BS, num=1
    # H (10) → flag=HLD (alias), num=10
    m = re.match(r"^(?P<flag>[A-Z]+)\s*(?:\((?P<num>[\d-]+)\))?", t)
    if not m:
        return None, None
    flag = m.group('flag').upper()
    if flag == 'H':
        flag = 'HLD'
    num_s = m.group('num') or ''
    if not num_s:
        return flag, None
    if '-' in num_s:
        num_s = num_s.split('-')[0]
    try:
        return flag, int(num_s)
    except ValueError:
        return flag, None


# ══════════════════════════════════════════════════════════════════════
# Parsing
# ══════════════════════════════════════════════════════════════════════

_TITLE_RE = re.compile(
    r"(?P<league>[A-Z0-9]+)\s+Box Score,\s+"
    r"(?P<away>.+?)\s+at\s+(?P<home>.+?),\s+"
    r"(?P<date>\d{2}/\d{2}/\d{4})"
)

_GAME_ID_RE = re.compile(r"game_box_(\d+)", re.IGNORECASE)


def _parse_title(soup: BeautifulSoup) -> dict:
    """Pull league id, away team, home team, game date from the <title>."""
    title = (soup.title.string or '') if soup.title else ''
    m = _TITLE_RE.search(title)
    if not m:
        return {'league_id': None, 'home_team': None, 'away_team': None,
                'game_date': None}
    # OOTP date is MM/DD/YYYY — convert to YYYY-MM-DD
    mm, dd, yyyy = m.group('date').split('/')
    return {
        'league_id': m.group('league').lower(),   # 'lb124', 'i76'
        'home_team': m.group('home').strip(),
        'away_team': m.group('away').strip(),
        'game_date': f"{yyyy}-{mm}-{dd}",
    }


def _parse_linescore(soup: BeautifulSoup) -> dict:
    """Pull home/away total runs from the linescore block.

    Structure: a <table> containing a <thead>-ish row (E R H or 1 2 3…)
    followed by two team rows. The team rows end with three <b>-wrapped
    totals: R, H, E. We scan all tables and pick the first one that has
    exactly 2 team-name-leading rows each ending in a <b>...</b> cell.
    """
    result = {'home_score': None, 'away_score': None,
              'away_team_line': None, 'home_team_line': None}
    for tbl in soup.find_all('table'):
        team_rows = []
        for tr in tbl.find_all('tr'):
            cells = tr.find_all('td')
            if len(cells) < 4:
                continue
            first_text = cells[0].get_text(strip=True)
            if not first_text or not any(c.isalpha() for c in first_text):
                continue
            # Need at least one bolded trailing cell (R/H/E totals)
            trailing_bold = [c for c in cells[-3:] if c.find('b')]
            if len(trailing_bold) >= 1:
                team_rows.append((first_text, cells))
        if len(team_rows) == 2:
            (away_name, away_cells), (home_name, home_cells) = team_rows
            try:
                # The last 3 cells are R / H / E — the first of the three is
                # the runs total.
                away_totals = [c.get_text(strip=True) for c in away_cells[-3:]]
                home_totals = [c.get_text(strip=True) for c in home_cells[-3:]]
                # Strip (record) suffix from team name: "Toronto (58-56)" → "Toronto"
                away_name = re.sub(r'\s*\(\d+-\d+\)\s*$', '', away_name).strip()
                home_name = re.sub(r'\s*\(\d+-\d+\)\s*$', '', home_name).strip()
                result['away_score'] = _safe_int(away_totals[0])
                result['home_score'] = _safe_int(home_totals[0])
                result['away_team_line'] = away_name
                result['home_team_line'] = home_name
                return result
            except Exception:
                continue
    return result


def _parse_stat_table(table, columns: list[str]) -> list[dict]:
    """Generic row-reader: pull rows where first cell has an <a> and return
    each as a dict keyed by ``columns``.

    Columns after the first (player + trailing text) must match the <td>
    order exactly. Returns empty list if no valid rows.
    """
    rows = []
    trs = table.find_all('tr')
    for tr in trs:
        cells = tr.find_all('td')
        if len(cells) < len(columns):
            continue
        first = cells[0]
        anchor = first.find('a')
        if not anchor:
            continue
        name = anchor.get_text(strip=True)
        # Trailing text after </a> (used for pitcher role flag)
        trailing = first.get_text(strip=True).replace(name, '', 1).strip()
        row = {'player_name': name, 'trailing': trailing}
        # Remaining cells map to columns[1:]
        for i, col in enumerate(columns[1:], start=1):
            if i < len(cells):
                row[col] = cells[i].get_text(strip=True)
        rows.append(row)
    return rows


_BATTING_COLS = ['player_name', 'ab', 'r', 'h', 'rbi', 'bb', 'k', 'lob',
                 'season_avg', 'season_hr', 'season_rbi']
_PITCHING_COLS = ['player_name', 'ip', 'h', 'r', 'er', 'bb', 'k', 'hr',
                  'pitches', 'season_era']


def _find_stat_tables(soup: BeautifulSoup) -> dict:
    """Locate the four stat tables in a box score.

    Returns dict with keys 'away_batting', 'home_batting', 'away_pitching',
    'home_pitching', each mapping to a <table> element or None.
    """
    # The stat tables have class "data sortable" and are inside a <td class="databg">.
    # Order on the page: [away_batting, home_batting, away_pitching, home_pitching]
    stat_tables = soup.select('table.data.sortable')
    # Filter to ones whose header row contains 'AB' (batting) or 'IP' (pitching)
    batting = []
    pitching = []
    for tbl in stat_tables:
        header_row = tbl.find('tr')
        if not header_row:
            continue
        headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
        header_set = set(headers)
        if 'AB' in header_set and 'AVG' in header_set:
            batting.append(tbl)
        elif 'IP' in header_set and 'ERA' in header_set:
            pitching.append(tbl)
    return {
        'away_batting': batting[0] if len(batting) >= 1 else None,
        'home_batting': batting[1] if len(batting) >= 2 else None,
        'away_pitching': pitching[0] if len(pitching) >= 1 else None,
        'home_pitching': pitching[1] if len(pitching) >= 2 else None,
    }


# Parse a single "Batters Faced:" or "Ground Outs - Fly Outs:" string from
# the pitching context block. The text looks like:
#   "R. Ruffing 20, C. Booser 7, E. Orze 4, ..."
# We split on commas and pull "name number" pairs.
_PITCH_CTX_PAIR_RE = re.compile(
    r"([A-Z]\.\s+[A-Za-z. ']+?)\s+(\d+[-]?\d*)"
)


def _parse_pitching_context(block_text: str) -> dict:
    """Parse the pitching context block for one team.

    Returns dict of {label: {player_name: value}} for each of the labels
    we care about (batters_faced, ground_outs_fly_outs, pitches_strikes,
    game_score).
    """
    result = {
        'game_score': {},
        'batters_faced': {},
        'ground_fly': {},   # values are (go, fo) tuples
        'pitches_strikes': {},  # values are (pi, st) tuples
    }
    if not block_text:
        return result

    def _after(label: str) -> str:
        """Return the text after `label:` up to the next bold label (or end)."""
        idx = block_text.find(label)
        if idx < 0:
            return ''
        tail = block_text[idx + len(label):]
        # Cut at the next bold label if present
        for stop in ['Game Score:', 'Batters Faced:', 'Ground Outs - Fly Outs:',
                     'Pitches - Strikes:']:
            sidx = tail.find(stop)
            if sidx >= 0:
                tail = tail[:sidx]
        return tail.strip()

    gs = _after('Game Score:')
    if gs:
        # "R. Ruffing 32" — one pitcher (the starter) at most
        for name, num in _PITCH_CTX_PAIR_RE.findall(gs):
            try:
                result['game_score'][name.strip()] = int(num.split('-')[0])
            except ValueError:
                pass

    bf = _after('Batters Faced:')
    for name, num in _PITCH_CTX_PAIR_RE.findall(bf):
        try:
            result['batters_faced'][name.strip()] = int(num.split('-')[0])
        except ValueError:
            pass

    gf = _after('Ground Outs - Fly Outs:')
    for name, num in _PITCH_CTX_PAIR_RE.findall(gf):
        # num is "9-7" format → split
        if '-' in num:
            try:
                go_s, fo_s = num.split('-', 1)
                result['ground_fly'][name.strip()] = (int(go_s), int(fo_s))
            except (ValueError, IndexError):
                pass

    ps = _after('Pitches - Strikes:')
    for name, num in _PITCH_CTX_PAIR_RE.findall(ps):
        if '-' in num:
            try:
                pi_s, st_s = num.split('-', 1)
                result['pitches_strikes'][name.strip()] = (int(pi_s), int(st_s))
            except (ValueError, IndexError):
                pass

    return result


def parse_box_score(filepath: str) -> Optional[dict]:
    """Parse one box score HTML into a structured dict.

    Returns None if the file can't be parsed. Never raises.

    Schema::

        {
            'game_id': 1705,
            'league_id': 'lb124',
            'game_date': '2030-08-13',
            'home_team': 'Spring Fire Fighters',
            'away_team': 'Toronto Dark Knights',
            'home_score': 3,
            'away_score': 10,
            'batting': [
                {'team': 'Toronto...', 'player_name': 'J. Pierre', 'position': 'CF',
                 'batting_order': 1, 'ab': 5, 'r': 2, 'h': 2, 'rbi': 0,
                 'bb': 0, 'k': 0, 'lob': 1, 'season_avg': 0.279,
                 'season_hr': 1, 'season_rbi': 40},
                ...
            ],
            'pitching': [
                {'team': 'Toronto...', 'player_name': 'B. Dal Canton',
                 'role_flag': None, 'role_number': None, 'appearance_order': 1,
                 'ip': 6.0, 'h': 7, 'r': 5, 'er': 5, 'bb': 5, 'k': 0, 'hr': 1,
                 'pitches': 108, 'strikes': 65, 'batters_faced': 28,
                 'ground_outs': 9, 'fly_outs': 7, 'game_score': 33,
                 'season_era': 4.20},
                ...
            ],
            'source_file': '/path/to/game_box_1705.html',
        }
    """
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        html = path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        logger.warning(f"box score read failed: {filepath}: {e}")
        return None

    # game_id from filename
    m = _GAME_ID_RE.search(path.name)
    game_id = int(m.group(1)) if m else None
    if not game_id:
        logger.warning(f"box score has no game_id: {filepath}")
        return None

    try:
        soup = BeautifulSoup(html, 'lxml')
    except Exception as e:
        logger.warning(f"box score parse failed: {filepath}: {e}")
        return None

    title_info = _parse_title(soup)
    linescore = _parse_linescore(soup)
    tables = _find_stat_tables(soup)

    # Prefer linescore team names (cleaner — no (W-L) suffix) when present
    if linescore.get('home_team_line'):
        title_info['home_team'] = linescore['home_team_line']
    if linescore.get('away_team_line'):
        title_info['away_team'] = linescore['away_team_line']

    # Read batting
    batting: list[dict] = []
    for side, team in [('away', title_info.get('away_team')),
                       ('home', title_info.get('home_team'))]:
        tbl = tables.get(f'{side}_batting')
        if tbl is None:
            continue
        raw = _parse_stat_table(tbl, _BATTING_COLS)
        for order_idx, row in enumerate(raw, start=1):
            # Position is sometimes in the trailing text (e.g. 'CF', 'PR', 'PH')
            pos = row.get('trailing') or None
            batting.append({
                'team': team,
                'player_name': row['player_name'],
                'position': pos,
                'batting_order': order_idx,
                'ab': _safe_int(row.get('ab')),
                'r': _safe_int(row.get('r')),
                'h': _safe_int(row.get('h')),
                'rbi': _safe_int(row.get('rbi')),
                'bb': _safe_int(row.get('bb')),
                'k': _safe_int(row.get('k')),
                'lob': _safe_int(row.get('lob')),
                'season_avg': _safe_float(row.get('season_avg')),
                'season_hr': _safe_int(row.get('season_hr')),
                'season_rbi': _safe_int(row.get('season_rbi')),
            })

    # Pull pitching context blocks — each team has its own "Game Score: …"
    # block. We find every <b>PITCHING</b> marker and parse the surrounding
    # text independently so both teams' game scores / batters-faced / etc.
    # are captured (the previous text.find('Game Score:') only caught the
    # first team's block).
    pitch_ctx = {'game_score': {}, 'batters_faced': {},
                 'ground_fly': {}, 'pitches_strikes': {}}
    for b in soup.find_all('b'):
        if b.get_text(strip=True) != 'PITCHING':
            continue
        # The context block is the text content of the enclosing <td> — walk
        # up to the nearest <td> and take its get_text.
        container = b.find_parent('td') or b.parent
        block_text = container.get_text('\n', strip=True) if container else ''
        block = _parse_pitching_context(block_text)
        for key in pitch_ctx:
            pitch_ctx[key].update(block.get(key, {}))

    # Read pitching
    pitching: list[dict] = []
    for side, team in [('away', title_info.get('away_team')),
                       ('home', title_info.get('home_team'))]:
        tbl = tables.get(f'{side}_pitching')
        if tbl is None:
            continue
        raw = _parse_stat_table(tbl, _PITCHING_COLS)
        for order_idx, row in enumerate(raw, start=1):
            flag, num = _extract_role_and_number(row.get('trailing') or '')
            if order_idx == 1 and not flag:
                flag = 'SP'
            gf = pitch_ctx['ground_fly'].get(row['player_name'], (None, None))
            ps = pitch_ctx['pitches_strikes'].get(row['player_name'], (None, None))
            pitching.append({
                'team': team,
                'player_name': row['player_name'],
                'role_flag': flag,
                'role_number': num,
                'appearance_order': order_idx,
                'ip': _safe_float(row.get('ip')),
                'h': _safe_int(row.get('h')),
                'r': _safe_int(row.get('r')),
                'er': _safe_int(row.get('er')),
                'bb': _safe_int(row.get('bb')),
                'k': _safe_int(row.get('k')),
                'hr': _safe_int(row.get('hr')),
                'pitches': _safe_int(row.get('pitches')),
                'strikes': ps[1] if ps[1] is not None else 0,
                'batters_faced': pitch_ctx['batters_faced'].get(row['player_name']),
                'ground_outs': gf[0],
                'fly_outs': gf[1],
                'game_score': pitch_ctx['game_score'].get(row['player_name']),
                'season_era': _safe_float(row.get('season_era')),
            })

    home = title_info.get('home_team')
    away = title_info.get('away_team')
    home_s = linescore.get('home_score')
    away_s = linescore.get('away_score')
    winner = None
    if home_s is not None and away_s is not None:
        if home_s > away_s:
            winner = 'home'
        elif away_s > home_s:
            winner = 'away'
    toronto_role = None
    if home and 'Toronto' in home:
        toronto_role = 'home'
    elif away and 'Toronto' in away:
        toronto_role = 'away'

    # Narrative + ambient metadata
    recap_headline, recap_body = _parse_recap(html)
    notes = _parse_game_notes(soup)

    # Clutch event lists — each team's batting and pitching events live in
    # its own <td class="databg"> cell. We find each unique cell that has
    # a BATTING or PITCHING marker, get its full text once, attribute it
    # to the correct team by matching player names, and parse labels.
    # Dedup by container id() so we don't re-process the same cell
    # multiple times (each cell has BATTING + BASERUNNING + FIELDING
    # markers, so naively iterating the <b> tags triple-counts).
    clutch_events: list[dict] = []
    seen_containers: set[int] = set()
    away_batter_names = {b['player_name'] for b in batting if b.get('team') == away}
    home_batter_names = {b['player_name'] for b in batting if b.get('team') == home}
    for marker in soup.find_all('b'):
        if marker.get_text(strip=True) not in ('BATTING', 'PITCHING'):
            continue
        container = marker.find_parent('td')
        if container is None or id(container) in seen_containers:
            continue
        seen_containers.add(id(container))
        block_text = container.get_text('\n', strip=True)

        # Team attribution: which team's roster has more name matches in
        # this block?
        away_hits = sum(1 for n in away_batter_names if n and n in block_text)
        home_hits = sum(1 for n in home_batter_names if n and n in block_text)
        if away_hits > home_hits and away:
            _team = away
        elif home_hits >= away_hits and home:
            _team = home
        else:
            _team = None

        for label, etype in [
            ('Doubles', 'DOUBLE'),
            ('Triples', 'TRIPLE'),
            ('Home Runs', 'HR'),
            ('2-out RBI', '2OUT_RBI'),
            ('Runners left in scoring position, 2 outs', 'LOB_RISP_2OUT'),
            ('Sac Fly', 'SAC_FLY'),
            ('GIDP', 'GIDP'),
            ('SB', 'SB'),
            ('CS', 'CS'),
            ('Errors', 'ERROR'),
        ]:
            if etype in ('DOUBLE', 'TRIPLE', 'HR'):
                parsed_ev = _parse_event_block(block_text, label)
            else:
                parsed_ev = _parse_simple_player_list(block_text, label)
            for entry in parsed_ev:
                entry['event_type'] = etype
                entry['team_name'] = _team
                clutch_events.append(entry)

        # Inherited runners is a special "X-Y" format — produces two event
        # rows per player so we get both the denominator (INHERITED_RUNNERS)
        # and the numerator (INHERITED_SCORED) for a real rate calc.
        inh_runs, inh_scored = _parse_inherited_runners(block_text)
        for entry in inh_runs:
            entry['event_type'] = 'INHERITED_RUNNERS'
            entry['team_name'] = _team
            clutch_events.append(entry)
        for entry in inh_scored:
            entry['event_type'] = 'INHERITED_SCORED'
            entry['team_name'] = _team
            clutch_events.append(entry)

    return {
        'game_id': game_id,
        'league_id': title_info.get('league_id'),
        'game_date': title_info.get('game_date'),
        'home_team': home,
        'away_team': away,
        'home_score': home_s,
        'away_score': away_s,
        'winner_team': winner,
        'toronto_role': toronto_role,
        'source_file': str(path),
        'batting': batting,
        'pitching': pitching,
        # New: narrative + situational events
        'recap_headline': recap_headline,
        'recap_body': recap_body,
        'game_notes': notes,
        'clutch_events': clutch_events,
    }


# ══════════════════════════════════════════════════════════════════════
# Ingestion
# ══════════════════════════════════════════════════════════════════════

def ingest_box_score(filepath: str, conn=None) -> dict:
    """Parse a box score HTML and write it to the DB. Idempotent on game_id.

    Returns {'status': 'success'|'skipped'|'error', 'game_id': int, 'rows': int}.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        parsed = parse_box_score(filepath)
        if not parsed:
            return {'status': 'error', 'rows': 0,
                    'reason': 'parse returned None', 'file': filepath}

        cursor = conn.cursor()
        # Idempotent: delete existing rows for this game_id, then re-insert
        cursor.execute("DELETE FROM game_batting WHERE game_id = ?", (parsed['game_id'],))
        cursor.execute("DELETE FROM game_pitching WHERE game_id = ?", (parsed['game_id'],))
        cursor.execute("DELETE FROM games WHERE game_id = ?", (parsed['game_id'],))

        cursor.execute(
            """INSERT INTO games
               (game_id, league_id, game_date, home_team, away_team,
                home_score, away_score, winner_team, toronto_role, source_file)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (parsed['game_id'], parsed['league_id'], parsed['game_date'],
             parsed['home_team'], parsed['away_team'],
             parsed['home_score'], parsed['away_score'],
             parsed['winner_team'], parsed['toronto_role'],
             parsed['source_file']),
        )

        rows = 0
        league_id_for_resolve = parsed.get('league_id')
        for b in parsed['batting']:
            card_id = _resolve_game_card_id(
                cursor, b['player_name'], b.get('team'), league_id_for_resolve,
            )
            cursor.execute(
                """INSERT OR REPLACE INTO game_batting
                   (game_id, league_id, player_name, card_id, team_name, position, batting_order,
                    ab, r, h, rbi, bb, k, lob,
                    season_avg, season_hr, season_rbi)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (parsed['game_id'], parsed.get('league_id'), b['player_name'], card_id, b['team'],
                 b['position'], b['batting_order'],
                 b['ab'], b['r'], b['h'], b['rbi'], b['bb'], b['k'], b['lob'],
                 b['season_avg'], b['season_hr'], b['season_rbi']),
            )
            rows += 1

        for p in parsed['pitching']:
            card_id = _resolve_game_card_id(
                cursor, p['player_name'], p.get('team'), league_id_for_resolve,
            )
            cursor.execute(
                """INSERT OR REPLACE INTO game_pitching
                   (game_id, league_id, player_name, card_id, team_name, appearance_order,
                    role_flag, role_number,
                    ip, h, r, er, bb, k, hr, pitches, strikes,
                    batters_faced, ground_outs, fly_outs, game_score, season_era)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (parsed['game_id'], parsed.get('league_id'), p['player_name'], card_id, p['team'],
                 p['appearance_order'], p['role_flag'], p['role_number'],
                 p['ip'], p['h'], p['r'], p['er'], p['bb'], p['k'], p['hr'],
                 p['pitches'], p['strikes'],
                 p['batters_faced'], p['ground_outs'], p['fly_outs'],
                 p['game_score'], p['season_era']),
            )
            rows += 1

        # Narrative + ambient metadata (one row per game_id). Idempotent via
        # REPLACE on the primary key.
        notes = parsed.get('game_notes') or {}
        if parsed.get('recap_headline') or parsed.get('recap_body') or notes.get('player_of_game'):
            cursor.execute("DELETE FROM game_narratives WHERE game_id = ?",
                           (parsed['game_id'],))
            cursor.execute(
                """INSERT INTO game_narratives
                   (game_id, recap_headline, recap_body, player_of_game,
                    ballpark, weather, game_duration, start_time, attendance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (parsed['game_id'],
                 parsed.get('recap_headline'), parsed.get('recap_body'),
                 notes.get('player_of_game'), notes.get('ballpark'),
                 notes.get('weather'), notes.get('game_duration'),
                 notes.get('start_time'), notes.get('attendance')),
            )
            rows += 1

        # Clutch / situational events — one row per event per player-game.
        # Idempotent via full DELETE of the game_id, then bulk re-insert.
        cursor.execute("DELETE FROM game_clutch_events WHERE game_id = ?",
                       (parsed['game_id'],))
        for ev in (parsed.get('clutch_events') or []):
            cid = _resolve_game_card_id(
                cursor, ev['player_name'], ev.get('team_name'), league_id_for_resolve,
            )
            cursor.execute(
                """INSERT INTO game_clutch_events
                   (game_id, league_id, player_name, card_id, team_name, event_type,
                    event_count, inning, context, opposing_pitcher,
                    season_count, raw_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (parsed['game_id'], parsed.get('league_id'), ev['player_name'], cid,
                 ev.get('team_name'), ev['event_type'],
                 ev.get('event_count', 1), ev.get('inning'),
                 ev.get('context'), ev.get('opposing_pitcher'),
                 ev.get('season_count'), ev.get('raw_text')),
            )
            rows += 1

        conn.commit()
        return {'status': 'success', 'game_id': parsed['game_id'],
                'rows': rows, 'file': filepath}
    except Exception as e:
        logger.error(f"ingest_box_score failed for {filepath}: {e}", exc_info=True)
        return {'status': 'error', 'rows': 0, 'reason': str(e), 'file': filepath}
    finally:
        if close_conn:
            conn.close()


def scan_box_score_dir(save_dir: str) -> list[str]:
    """Walk `<save_dir>/news/html/box_scores/` and return HTML file paths."""
    if not save_dir or not os.path.isdir(save_dir):
        return []
    box_dir = os.path.join(save_dir, 'news', 'html', 'box_scores')
    if not os.path.isdir(box_dir):
        return []
    out = []
    for name in os.listdir(box_dir):
        if name.lower().endswith('.html'):
            out.append(os.path.join(box_dir, name))
    return sorted(out)


def scan_game_log_dir(save_dir: str) -> list[str]:
    """Walk `<save_dir>/news/html/game_logs/` and return HTML file paths."""
    if not save_dir or not os.path.isdir(save_dir):
        return []
    gl_dir = os.path.join(save_dir, 'news', 'html', 'game_logs')
    if not os.path.isdir(gl_dir):
        return []
    out = []
    for name in os.listdir(gl_dir):
        if name.lower().endswith('.html'):
            out.append(os.path.join(gl_dir, name))
    return sorted(out)


# ══════════════════════════════════════════════════════════════════════
# Game log parser — pitch-by-pitch at-bat extraction
# ══════════════════════════════════════════════════════════════════════

_EV_RE = re.compile(r'EV\s+(\d+)\s+MPH')
_BATTED_BALL_RE = re.compile(
    r'\((Line Drive|Groundball|Fly Ball|Popup|Bunt)[^)]*\)', re.IGNORECASE,
)
_LOC_RE = re.compile(
    r'\(\s*(?:Line Drive|Groundball|Fly Ball|Popup|Bunt)[^,]*,\s*([0-9A-Z]+)'
)
_GAME_LOG_ID_RE = re.compile(r'log_(\d+)', re.IGNORECASE)

# ── Box score narrative / event parsing ─────────────────────────────
_RECAP_SUBJECT_RE = re.compile(
    r'<!--RECAP_SUBJECT_START-->(.*?)<!--RECAP_SUBJECT_END-->',
    re.DOTALL,
)
_RECAP_TEXT_RE = re.compile(
    r'<!--RECAP_TEXT_START-->(.*?)<!--RECAP_TEXT_END-->',
    re.DOTALL,
)
# "Doubles: <a>B. Wallace</a> (19, 9th Inning off B. Bernardino, 1 on, 0 outs)"
_EVENT_ENTRY_RE = re.compile(
    r'(?P<name>[A-Z]\.\s+[A-Za-z.\' -]+?)'
    r'(?:\s+\((?P<season>\d+)(?:,\s*(?P<inning>\d+)(?:st|nd|rd|th)?\s*Inning)?'
    r'(?:\s+off\s+(?P<pitcher>[A-Z]\.\s+[A-Za-z.\' -]+?))?'
    r'(?:,\s*(?P<context>[^)]+))?\))?(?=[,\s]|$)',
    re.MULTILINE,
)


def _kv_from_block_text(text: str) -> dict:
    """Parse a "Label:\\nvalue\\nLabel:\\nvalue\\n..." block into a dict.

    OOTP's rendered box-score text has labels on their own line followed
    by the value on the next line (sometimes multi-line for things like
    Weather). Returns a case-preserved dict of label→value.
    """
    if not text:
        return {}
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    out: dict[str, str] = {}
    current_label = None
    current_value: list[str] = []
    for ln in lines:
        if ln.endswith(':'):
            # Flush prior
            if current_label is not None:
                out[current_label] = ' '.join(current_value).strip()
            current_label = ln[:-1].strip()
            current_value = []
        else:
            if current_label is not None:
                current_value.append(ln)
    if current_label is not None:
        out[current_label] = ' '.join(current_value).strip()
    return out


def _parse_game_notes(soup: BeautifulSoup) -> dict:
    """Extract the ambient game metadata from the GAME NOTES block.

    Looks for a <td class="databg"> whose text contains "Player of the
    Game" and "Ballpark" (the canonical GAME NOTES markers), then parses
    the Label:\\nvalue structure within.
    """
    out = {'player_of_game': None, 'ballpark': None, 'weather': None,
           'game_duration': None, 'start_time': None, 'attendance': None}
    for cell in soup.find_all('td', class_='databg'):
        t = cell.get_text('\n', strip=True)
        if 'Player of the Game' not in t:
            continue
        kv = _kv_from_block_text(t)
        out['player_of_game'] = kv.get('Player of the Game')
        out['ballpark'] = kv.get('Ballpark')
        out['weather'] = kv.get('Weather')
        out['start_time'] = kv.get('Start Time')
        out['game_duration'] = kv.get('Time')
        attn = kv.get('Attendance')
        if attn:
            try:
                out['attendance'] = int(str(attn).replace(',', ''))
            except (ValueError, TypeError):
                pass
        break
    return out


def _parse_recap(html: str) -> tuple[Optional[str], Optional[str]]:
    """Return (headline, body) from the RECAP comment markers, stripping
    HTML tags from the body but preserving text flow.
    """
    h = _RECAP_SUBJECT_RE.search(html)
    t = _RECAP_TEXT_RE.search(html)
    headline = None
    body = None
    if h:
        headline = BeautifulSoup(h.group(1), 'lxml').get_text(' ', strip=True)
    if t:
        # Body contains <br> and <a> tags; flatten to text but preserve
        # paragraph breaks (double-<br> in OOTP's output).
        raw = t.group(1)
        raw = re.sub(r'(?i)<br\s*/?>\s*<br\s*/?>', '\n\n', raw)
        raw = re.sub(r'(?i)<br\s*/?>', ' ', raw)
        body = BeautifulSoup(raw, 'lxml').get_text(' ', strip=True)
        body = re.sub(r'[ \t]+', ' ', body)
        body = re.sub(r'\n ', '\n', body)
    return headline, body


def _section_from_block(block_text: str, label: str) -> list[str]:
    """Return the lines under ``label:`` up to the next LABEL: line.

    Operates on line-based split so we don't need regex tail-cutting.
    """
    if not block_text:
        return []
    lines = [ln.strip() for ln in block_text.split('\n')]
    target = label.strip()
    if not target.endswith(':'):
        target = target + ':'
    try:
        start = next(i for i, ln in enumerate(lines) if ln == target)
    except StopIteration:
        return []
    out = []
    for ln in lines[start + 1:]:
        if not ln:
            continue
        if ln.endswith(':') and ln != target:
            break
        out.append(ln)
    return out


# Match "(19, 9th Inning off B. Bernardino, 1 on, 0 outs)" etc. — lazy
# to catch whatever variants OOTP emits.
_PARENS_CTX_RE = re.compile(
    r'^\((?P<season>\d+)(?:,\s*(?P<inning>\d+)(?:st|nd|rd|th)?\s*Inning)?'
    r'(?:\s+off\s+(?P<pitcher>[A-Z]\.\s+[A-Za-z.\' -]+?))?'
    r'(?:,\s*(?P<context>[^)]+))?\)'
)
# Name line like "B. Wallace", "R. Palmeiro", or "Roy Campanella". OOTP
# abbreviates the first name to a single initial plus period in box score
# event blocks, so the first token can be just one letter.
_NAME_LINE_RE = re.compile(r"^[A-Z][A-Za-z]*\.?(?:\s+[A-Z][A-Za-z.\'\- ]+){1,4}$")


def _parse_event_block(block_text: str, label: str) -> list[dict]:
    """Parse an event section with per-item metadata (inning, context, pitcher).

    Used for Doubles / Triples / Home Runs lines. Walks the lines under
    ``label:`` pairing each name with the next parenthesized metadata.
    """
    lines = _section_from_block(block_text, label)
    if not lines:
        return []
    entries: list[dict] = []
    current_name: Optional[str] = None
    for ln in lines:
        if ln == ',' or ln.startswith(','):
            continue
        if _NAME_LINE_RE.match(ln):
            current_name = ln
            continue
        if ln.startswith('(') and current_name:
            m = _PARENS_CTX_RE.search(ln)
            if m:
                entries.append({
                    'player_name': current_name,
                    'season_count': int(m.group('season')) if m.group('season') else None,
                    'inning': int(m.group('inning')) if m.group('inning') else None,
                    'opposing_pitcher': (m.group('pitcher') or '').strip() or None,
                    'context': (m.group('context') or '').strip() or None,
                    'raw_text': f"{current_name} {ln}",
                })
            current_name = None
    return entries


def _parse_simple_player_list(block_text: str, label: str) -> list[dict]:
    """Parse a list of player names: '2-out RBI: A. Wells, G. Wilson'.

    OOTP renders these as newline-separated: NAME / , / NAME / (count?) / ,...
    We walk the section lines, treating standalone names as entries and
    parenthesized counts as annotations for the preceding name.
    """
    lines = _section_from_block(block_text, label)
    if not lines:
        return []
    entries: list[dict] = []
    for ln in lines:
        if ln == ',' or ln.startswith(','):
            continue
        # "(5)" → season count for the previous entry
        m_count = re.match(r'^\((\d+)\)$', ln)
        if m_count and entries:
            entries[-1]['season_count'] = int(m_count.group(1))
            continue
        if _NAME_LINE_RE.match(ln):
            entries.append({
                'player_name': ln,
                'season_count': None,
                'raw_text': ln,
            })
    return entries


def _parse_inherited_runners(block_text: str) -> tuple[list[dict], list[dict]]:
    """Parse the 'Inherited Runners - Scored' line format: Name X-Y.

    X = runners inherited when entering, Y = runners that scored. Returns
    two parallel lists of dicts: (inherited_events, scored_events). Each
    inherited event has event_count = X; each scored event has count = Y.
    This gives us the denominator we need for inherited-runner-scored%
    (a reliever-discipline signal ERA doesn't capture).
    """
    lines = _section_from_block(block_text, 'Inherited Runners - Scored')
    if not lines:
        return [], []
    inherited: list[dict] = []
    scored: list[dict] = []
    current_name: Optional[str] = None
    for ln in lines:
        if ln == ',' or ln.startswith(','):
            continue
        m = re.match(r'^(\d+)\s*-\s*(\d+)$', ln)
        if m and current_name:
            inh = int(m.group(1))
            sc = int(m.group(2))
            inherited.append({
                'player_name': current_name,
                'event_count': inh,
                'raw_text': f"{current_name} {ln}",
            })
            scored.append({
                'player_name': current_name,
                'event_count': sc,
                'raw_text': f"{current_name} {ln}",
            })
            current_name = None
            continue
        if _NAME_LINE_RE.match(ln):
            current_name = ln
    return inherited, scored
_INNING_HEADER_RE = re.compile(r'(TOP|BOTTOM)\s+OF\s+THE\s+(\d+)(?:ST|ND|RD|TH)?',
                               re.IGNORECASE)


def _classify_outcome(text: str) -> tuple[Optional[str], Optional[str]]:
    """Classify at-bat outcome + strikeout type from rendered text.

    Returns (outcome, strikeout_type) where strikeout_type is 'looking' |
    'swinging' | None.
    """
    upper = text.upper()
    lower = text.lower()
    for tok, label in [('GRAND SLAM', 'HR'), ('HOME RUN', 'HR'),
                       ('TRIPLE', 'TRIPLE'), ('DOUBLE', 'DOUBLE'),
                       ('SINGLE', 'SINGLE')]:
        if tok in upper:
            return label, None
    if 'strikes out swinging' in lower:
        return 'K', 'swinging'
    if 'strikes out looking' in lower or 'called out on strikes' in lower:
        return 'K', 'looking'
    if 'strikes out' in lower or 'strikeout' in lower:
        return 'K', None
    if 'base on balls' in lower or 'walked' in lower or 'walks' in lower:
        return 'BB', None
    if 'hit by pitch' in lower or 'hbp' in lower:
        return 'HBP', None
    if "sacrifice" in lower or 'sac bunt' in lower or 'sac fly' in lower:
        return 'SAC', None
    if "fielder's choice" in lower or 'fielders choice' in lower:
        return 'FC', None
    if any(k in lower for k in ['grounds out', 'ground out', 'fly out',
                                 'flies out', 'pops out', 'popout',
                                 'lineout', 'line out', 'forceout',
                                 'force out', 'double play', 'triple play']):
        return 'OUT', None
    return None, None


def _parse_at_bat_features(outcome_html: str) -> dict:
    """Extract features from one at-bat outcome cell."""
    text = BeautifulSoup(outcome_html, 'lxml').get_text(' ', strip=True)

    ev_match = _EV_RE.search(text)
    ev = int(ev_match.group(1)) if ev_match else None

    bb_match = _BATTED_BALL_RE.search(text)
    batted_ball = bb_match.group(1) if bb_match else None

    loc_match = _LOC_RE.search(text)
    location = loc_match.group(1) if loc_match else None

    outcome, k_type = _classify_outcome(text)

    pitches = len(re.findall(r'\d+\s*-\s*\d+\s*:', text))

    return {
        'outcome': outcome,
        'strikeout_type': k_type,
        'batted_ball_type': batted_ball,
        'location': location,
        'exit_velocity': ev,
        'pitches_seen': pitches,
    }


def parse_game_log(filepath: str) -> Optional[dict]:
    """Parse one game log HTML into structured at-bats.

    Returns ``{'game_id': int, 'at_bats': list[dict]}`` or None on failure.
    """
    path = Path(filepath)
    if not path.exists():
        return None

    m = _GAME_LOG_ID_RE.search(path.name)
    game_id = int(m.group(1)) if m else None
    if not game_id:
        return None

    try:
        html = path.read_text(encoding='utf-8', errors='replace')
        soup = BeautifulSoup(html, 'lxml')
    except Exception as e:
        logger.warning(f"game log read/parse failed: {filepath}: {e}")
        return None

    at_bats: list[dict] = []
    current_pitcher = None
    current_pitcher_hand = None
    current_inning = None
    current_half = None

    for tr in soup.find_all('tr'):
        cells = tr.find_all(['th', 'td'])
        # Inning header: single <th> with colspan="2"
        if len(cells) == 1:
            head_text = cells[0].get_text(' ', strip=True)
            m_inn = _INNING_HEADER_RE.search(head_text)
            if m_inn:
                current_half = 'top' if m_inn.group(1).upper() == 'TOP' else 'bottom'
                try:
                    current_inning = int(m_inn.group(2))
                except (ValueError, TypeError):
                    current_inning = None
                continue

        if len(cells) < 2:
            continue
        left_text = cells[0].get_text(' ', strip=True)
        right_html = str(cells[1])

        # Pitching change: "Pitching: RHP Red Ruffing"
        m_pit = re.match(r'Pitching:\s*(LHP|RHP)\s+(.+)', left_text)
        if m_pit:
            current_pitcher_hand = m_pit.group(1)
            current_pitcher = m_pit.group(2).strip()
            continue

        # Batter line: "Batting: LHB Juan Pierre"
        m_bat = re.match(r'Batting:\s*(LHB|RHB|SHB)\s+(.+)', left_text)
        if not m_bat:
            continue
        batter_hand = m_bat.group(1)
        batter = m_bat.group(2).strip()

        features = _parse_at_bat_features(right_html)
        at_bats.append({
            'inning': current_inning,
            'half_inning': current_half,
            'pitcher': current_pitcher,
            'pitcher_hand': current_pitcher_hand,
            'batter': batter,
            'batter_hand': batter_hand,
            **features,
        })

    return {'game_id': game_id, 'at_bats': at_bats, 'source_file': str(path)}


def ingest_game_log(filepath: str, conn=None) -> dict:
    """Parse + write a game log to the DB. Idempotent on game_id."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        parsed = parse_game_log(filepath)
        if not parsed:
            return {'status': 'error', 'file': filepath,
                    'reason': 'parse returned None'}

        cursor = conn.cursor()

        # Ensure a parent `games` row exists before inserting at-bats. Raw
        # game_log.html files don't carry team/date/score headers, so when
        # the matching game_box.html isn't also ingested (missing export,
        # OOTP skipped writing it, etc.) we'd previously leave orphan
        # at-bats that violated the FK. Stub the row with just the game_id
        # + source_file so downstream joins don't drop rows; a subsequent
        # box-score ingest will `DELETE + INSERT` the full record.
        existing = cursor.execute(
            "SELECT 1 FROM games WHERE game_id = ?", (parsed['game_id'],)
        ).fetchone()
        if not existing:
            cursor.execute(
                """INSERT INTO games (game_id, league_id, source_file)
                   VALUES (?, ?, ?)""",
                (parsed['game_id'], parsed.get('league_id'), parsed.get('source_file')),
            )

        cursor.execute("DELETE FROM game_log_at_bats WHERE game_id = ?",
                       (parsed['game_id'],))

        try:
            from app.core.ingestion import _match_card_id
        except Exception:
            _match_card_id = None

        rows = 0
        for ab in parsed['at_bats']:
            batter_cid = None
            pitcher_cid = None
            if _match_card_id:
                try:
                    batter_cid = _match_card_id(cursor, ab['batter']) if ab['batter'] else None
                    pitcher_cid = _match_card_id(cursor, ab['pitcher']) if ab['pitcher'] else None
                except Exception:
                    pass

            cursor.execute(
                """INSERT INTO game_log_at_bats
                   (game_id, league_id, inning, half_inning, pitcher, pitcher_hand,
                    pitcher_card_id, batter, batter_hand, batter_card_id,
                    outcome, batted_ball_type, location, exit_velocity,
                    pitches_seen, strikeout_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (parsed['game_id'], parsed.get('league_id'),
                 ab['inning'], ab['half_inning'],
                 ab['pitcher'], ab['pitcher_hand'], pitcher_cid,
                 ab['batter'], ab['batter_hand'], batter_cid,
                 ab['outcome'], ab['batted_ball_type'], ab['location'],
                 ab['exit_velocity'], ab['pitches_seen'],
                 ab['strikeout_type']),
            )
            rows += 1
        conn.commit()
        return {'status': 'success', 'game_id': parsed['game_id'],
                'at_bats': rows, 'file': filepath}
    except Exception as e:
        logger.error(f"ingest_game_log failed for {filepath}: {e}", exc_info=True)
        return {'status': 'error', 'file': filepath, 'reason': str(e)}
    finally:
        if close_conn:
            conn.close()


def ingest_all_game_logs(save_dir: str, conn=None,
                         skip_existing: bool = True) -> dict:
    """Ingest every game log HTML under ``save_dir``. Idempotent."""
    paths = scan_game_log_dir(save_dir)
    if not paths:
        return {'status': 'empty', 'scanned': 0, 'ingested': 0}

    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        existing = set()
        if skip_existing:
            existing = {r[0] for r in conn.execute(
                "SELECT DISTINCT game_id FROM game_log_at_bats"
            ).fetchall()}

        ingested = 0
        skipped = 0
        errors = 0
        total_at_bats = 0
        for p in paths:
            m = _GAME_LOG_ID_RE.search(os.path.basename(p))
            gid = int(m.group(1)) if m else None
            if gid and gid in existing:
                skipped += 1
                continue
            res = ingest_game_log(p, conn=conn)
            if res.get('status') == 'success':
                ingested += 1
                total_at_bats += res.get('at_bats', 0)
            else:
                errors += 1
        return {'status': 'success' if errors == 0 else 'partial',
                'scanned': len(paths), 'ingested': ingested,
                'skipped': skipped, 'errors': errors,
                'at_bats': total_at_bats}
    finally:
        if close_conn:
            conn.close()


def ingest_all_box_scores(save_dir: str, conn=None,
                          skip_existing: bool = True) -> dict:
    """Ingest every HTML box score under ``save_dir``. Idempotent.

    ``skip_existing``: if True, skip games already in the DB (by game_id).
    """
    paths = scan_box_score_dir(save_dir)
    if not paths:
        return {'status': 'empty', 'scanned': 0, 'ingested': 0, 'skipped': 0}

    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        existing = set()
        if skip_existing:
            existing = {r[0] for r in conn.execute("SELECT game_id FROM games").fetchall()}

        ingested = 0
        skipped = 0
        errors = 0
        for p in paths:
            m = _GAME_ID_RE.search(os.path.basename(p))
            gid = int(m.group(1)) if m else None
            if gid and gid in existing:
                skipped += 1
                continue
            result = ingest_box_score(p, conn=conn)
            if result['status'] == 'success':
                ingested += 1
            else:
                errors += 1
        return {
            'status': 'success' if errors == 0 else 'partial',
            'scanned': len(paths), 'ingested': ingested,
            'skipped': skipped, 'errors': errors,
        }
    finally:
        if close_conn:
            conn.close()
