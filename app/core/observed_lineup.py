"""Observed-lineup extractor.

Derives "who actually starts at each position" from recent box scores. This
is more reliable than OOTP's lineup CSV because the CSV exports the depth
chart (sorted by games-played historically) while box scores reflect the
USER'S pinned starters from recent games.

Used by Roster Optimizer so recommendations target the lineup you're
actually running, not whoever looks best by meta.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ObservedStarter:
    """One position's observed starter derived from recent box scores."""
    position: str
    player_name: str           # as it appears in box scores ("C. Kluttz" / "R. Palmeiro")
    games_started: int
    batting_order: Optional[int]  # modal BO slot across those games
    last_seen: str            # ISO date
    confidence: float         # 0-1


def _resolve_full_name(conn, box_name: str,
                       team_name: str = 'Toronto Dark Knights') -> Optional[str]:
    """Box score names are usually abbreviated (e.g. 'C. Kluttz'). Resolve
    to the canonical full name via the shared name_resolver — which handles
    suffixes, refuses to guess on ambiguous pairs, and caches the index.
    """
    if not box_name:
        return None
    try:
        from app.core.name_resolver import expand_abbreviated_name
        return expand_abbreviated_name(box_name, conn=conn) or box_name
    except Exception as _e:
        logger.debug("name_resolver unavailable, fallback: %s", _e)
    # Legacy LIKE fallback (rare — only if resolver import fails)
    import re as _re
    m = _re.match(r'^\s*([A-Z])\.?\s+([A-Za-z\'\-]+)\s*$', box_name.strip())
    if not m:
        return box_name
    initial, last = m.group(1), m.group(2)
    try:
        row = conn.execute(
            "SELECT DISTINCT player_name FROM roster "
            "WHERE player_name LIKE ? AND player_name LIKE ? LIMIT 1",
            (f'{initial}%', f'%{last}%'),
        ).fetchone()
        if row:
            return row['player_name']
    except Exception:
        pass
    return box_name


def get_pinned_starters(
    team_name: str = 'Toronto Dark Knights',
    last_n_games: int = 3,
    min_games: int = 1,
) -> dict[str, ObservedStarter]:
    """Return {position: ObservedStarter} for positions where a starter is clear.

    Uses the MOST RECENT game's batting order for slot assignment (so each
    BO slot 1-9 is occupied by exactly one player, no collisions). Falls
    back to the most-recent multi-game window for starter identification
    when a position's recent game didn't include a BO 1-9 appearance.

    Pinch-hitters / substitution positions (prefixes like 'a-', 'b-', 'PH',
    'PR') are excluded so we only see the actual lineup.
    """
    from app.core.database import get_connection
    conn = get_connection()

    # Step 1 — most recent game. Grab every 1-9 BO row from that game so the
    # slot assignments are coherent (one player per slot).
    try:
        last_game_row = conn.execute(
            """
            SELECT game_id, game_date FROM games
            WHERE home_team = ? OR away_team = ?
            ORDER BY game_date DESC LIMIT 1
            """,
            (team_name, team_name),
        ).fetchone()
    except Exception as e:
        logger.warning("observed_lineup last-game query failed: %s", e)
        return {}
    if not last_game_row:
        return {}

    out: dict[str, ObservedStarter] = {}
    seen_slots: set[int] = set()
    try:
        last_game_starters = conn.execute(
            """
            SELECT player_name, position, batting_order
            FROM game_batting
            WHERE game_id = ? AND team_name = ?
              AND batting_order BETWEEN 1 AND 9
              AND position IN ('C','1B','2B','3B','SS','LF','CF','RF','DH')
            ORDER BY batting_order
            """,
            (last_game_row['game_id'], team_name),
        ).fetchall()
    except Exception as e:
        logger.warning("last-game starters query failed: %s", e)
        last_game_starters = []

    for r in last_game_starters:
        bo = int(r['batting_order'])
        if bo in seen_slots:
            # Guard against weird data — BO should be unique in a game.
            continue
        seen_slots.add(bo)
        pos = r['position']
        full = _resolve_full_name(conn, r['player_name'], team_name) or r['player_name']
        out[pos] = ObservedStarter(
            position=pos,
            player_name=full,
            games_started=1,
            batting_order=bo,
            last_seen=last_game_row['game_date'],
            confidence=1.0,
        )

    # Step 2 — for any primary position NOT covered by the most recent game
    # (e.g. off-day pinch-hit shuffle), fall back to the wider window's modal
    # starter WITHOUT a BO slot, so the chain table still shows them as
    # pinned even if we can't give them a 1-9 number right now.
    missing_positions = {'C','1B','2B','3B','SS','LF','CF','RF','DH'} - set(out.keys())
    if missing_positions:
        try:
            rows = conn.execute(
                f"""
                SELECT gb.player_name, gb.position,
                       COUNT(*) AS games,
                       MAX(g.game_date) AS last_date
                FROM game_batting gb
                INNER JOIN games g ON g.game_id = gb.game_id
                WHERE (g.home_team = ? OR g.away_team = ?)
                  AND gb.team_name = ?
                  AND gb.batting_order BETWEEN 1 AND 9
                  AND gb.position IN ({','.join(repr(p) for p in missing_positions)})
                  AND g.game_date IN (
                      SELECT game_date FROM games
                      WHERE home_team = ? OR away_team = ?
                      ORDER BY game_date DESC LIMIT ?
                  )
                GROUP BY gb.player_name, gb.position
                """,
                (team_name, team_name, team_name,
                 team_name, team_name, last_n_games),
            ).fetchall()
        except Exception as e:
            logger.warning("fallback observed_lineup query failed: %s", e)
            rows = []
        # Best per position by games started
        best: dict[str, tuple[str, int, str]] = {}
        for r in rows:
            pos = r['position']
            games = int(r['games'] or 0)
            if games < min_games:
                continue
            prev = best.get(pos)
            if prev is None or games > prev[1]:
                best[pos] = (r['player_name'], games, r['last_date'])
        for pos, (player, games, last) in best.items():
            full = _resolve_full_name(conn, player, team_name) or player
            out[pos] = ObservedStarter(
                position=pos,
                player_name=full,
                games_started=games,
                batting_order=None,   # no slot; player appeared in window but not last game
                last_seen=last or '',
                confidence=min(1.0, games / float(last_n_games)),
            )
    return out


def get_observed_batting_order(
    team_name: str = 'Toronto Dark Knights',
    last_n_games: int = 10,
) -> list[tuple[int, str]]:
    """Return [(batting_order, player_full_name), ...] in batting order 1..9.

    Derived from the modal BO slot per position's pinned starter. Used by
    the chain table's BO column so the user sees their ACTUAL batting
    order, not a hypothetical ideal.
    """
    starters = get_pinned_starters(team_name=team_name,
                                    last_n_games=last_n_games)
    out: list[tuple[int, str]] = []
    for starter in starters.values():
        if starter.batting_order:
            out.append((starter.batting_order, starter.player_name))
    out.sort(key=lambda x: x[0])
    return out
