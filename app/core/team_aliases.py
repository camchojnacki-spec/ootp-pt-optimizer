"""Team name normalization across OOTP data sources.

OOTP PT CSV exports use short team names ("Sassy", "Lakeville", "KT") while
HTML box scores use full franchise names ("Sassy Kitties", "Lakeville Tourists",
"KT Rolster"). This module owns the `team_aliases` table that bridges them.

The seeder is safe to re-run — it only INSERTs new (league_id, short_name)
pairs thanks to the UNIQUE constraint.
"""

from __future__ import annotations

import csv
import os
import sqlite3
from pathlib import Path
from typing import Iterable

from app.core.database import get_connection


def _load_short_names_from_default_csv(online_data_dir: str, league_id: str) -> set[str]:
    """Read the league-wide default CSV and return distinct `TM` values."""
    p = Path(online_data_dir) / (
        f"{league_id}_statistics_player_statistics_-_sortable_stats_default.csv"
    )
    if not p.exists():
        return set()
    names: set[str] = set()
    with p.open(encoding="utf-8", errors="replace") as fp:
        for row in csv.DictReader(fp):
            tm = (row.get("TM") or "").strip()
            if tm and tm != "-":
                names.add(tm)
    return names


def _load_full_names_from_games(conn: sqlite3.Connection, league_id: str) -> set[str]:
    """Return distinct team_name values from game_batting + game_pitching.

    Scopes by the games.league_id when possible, else falls back to all games
    (during early bootstrap when league_id on games may be sparse).
    """
    rows = conn.execute(
        """
        SELECT DISTINCT gb.team_name
        FROM game_batting gb
        JOIN games g ON g.game_id = gb.game_id
        WHERE g.league_id = ? AND gb.team_name IS NOT NULL AND gb.team_name <> ''
        """,
        (league_id,),
    ).fetchall()
    if rows:
        return {r[0] for r in rows}
    return {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT team_name FROM game_batting "
            "WHERE team_name IS NOT NULL AND team_name <> ''"
        )
    }


def _match_prefix(shorts: Iterable[str], fulls: Iterable[str]) -> dict[str, str]:
    """Map short→full where short (case-insensitive) is a word-boundary prefix of full."""
    fulls_list = list(fulls)
    out: dict[str, str] = {}
    for s in shorts:
        low = s.lower().strip()
        cands = [f for f in fulls_list if f.lower().startswith(low + " ") or f.lower() == low]
        if cands:
            out[s] = min(cands, key=len)  # tightest prefix wins on ties
    return out


def seed_team_aliases(
    online_data_dir: str,
    league_id: str,
    conn: sqlite3.Connection | None = None,
) -> dict:
    """Seed/refresh `team_aliases` for one league.

    Returns a summary dict with counts + unmatched names so a caller can
    surface coverage gaps in the UI.
    """
    owns_conn = conn is None
    if conn is None:
        conn = get_connection()
    try:
        shorts = _load_short_names_from_default_csv(online_data_dir, league_id)
        fulls = _load_full_names_from_games(conn, league_id)
        matches = _match_prefix(shorts, fulls)

        inserted = 0
        for short, full in matches.items():
            cur = conn.execute(
                """INSERT OR IGNORE INTO team_aliases
                   (league_id, short_name, full_name, source)
                   VALUES (?, ?, ?, 'csv_default+game_batting')""",
                (league_id, short, full),
            )
            inserted += cur.rowcount

        # Also record shorts we couldn't match (NULL full_name) so a human
        # can backfill them later. IGNORE if already present.
        unmatched_shorts = [s for s in shorts if s not in matches]
        for s in unmatched_shorts:
            conn.execute(
                """INSERT OR IGNORE INTO team_aliases
                   (league_id, short_name, full_name, source)
                   VALUES (?, ?, NULL, 'csv_default:unmatched')""",
                (league_id, s),
            )

        conn.commit()
        unmatched_fulls = [f for f in fulls if f not in matches.values()]
        return {
            "league_id": league_id,
            "shorts_found": len(shorts),
            "fulls_found": len(fulls),
            "matched": len(matches),
            "inserted": inserted,
            "unmatched_shorts": unmatched_shorts,
            "unmatched_fulls": unmatched_fulls,
        }
    finally:
        if owns_conn:
            conn.close()


def resolve_full_name(
    short_name: str, league_id: str, conn: sqlite3.Connection | None = None
) -> str | None:
    """Look up the full team name for a short name within a league.

    Returns None when no alias is known — callers should fall back to the
    short name rather than crash.
    """
    owns_conn = conn is None
    if conn is None:
        conn = get_connection()
    try:
        row = conn.execute(
            "SELECT full_name FROM team_aliases WHERE league_id = ? AND short_name = ?",
            (league_id, short_name),
        ).fetchone()
        return row[0] if row and row[0] else None
    finally:
        if owns_conn:
            conn.close()
