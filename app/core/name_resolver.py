"""Canonical player-name + card_id resolver.

Single source of truth for matching OOTP player-name strings to the right
``cards.card_id``. The codebase has FIVE different naming schemes floating
around:

    * ``roster.player_name``       = "Clyde Kluttz"              (full)
    * ``cards.card_title``         = "MLB 2026 Live C Clyde Kluttz SD"
    * ``batting_stats.player_name``= "Clyde Kluttz"              (full)
    * ``game_batting.player_name`` = "C. Kluttz"                  (abbreviated)
    * ``game_pitching.player_name``= "D. Lovelady"                (abbreviated)

Box-score ingestion was failing to resolve card_id for abbreviated names
(0.4% resolved on game_batting, 0% on game_pitching) because the legacy
``_match_card_id`` used exact ``first_name || ' ' || last_name`` equality
plus a ``card_title LIKE '%name%'`` fallback — neither matches "C. Kluttz"
against "Clyde Kluttz".

This module builds a **small cached index** on first use:

    {
      ('C', 'Kluttz'):    'Clyde Kluttz',
      ('R', 'Palmeiro'):  'Rafael Palmeiro',
      …
    }

…and exposes:

    * ``expand_abbreviated_name(name)`` → canonical full name (or input if no hit)
    * ``resolve_to_card_id(name, prefer_owned=True)`` → card_id or None

**Safety — never guess:**
    * If ``(initial, lastname)`` maps to >1 full name (two players with same
      initial + lastname), we refuse to guess and return the input unchanged.
    * The resolver is side-effect-free — it never writes to the DB.

Callers that need DB writes (e.g. backfilling game_batting.card_id) should
call ``resolve_to_card_id`` explicitly and skip rows that return None.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level cache — built lazily, invalidated via ``refresh_cache()``.
_CACHE_LOCK = threading.Lock()
_ABBREV_TO_FULL: dict[tuple[str, str], str] = {}
_AMBIGUOUS: set[tuple[str, str]] = set()
_FULL_NAMES: set[str] = set()
_CACHE_BUILT = False

# Team-scoped cache: (league_id, team_name, initial, lastname) -> card_id.
# Built from `league_rosters` which records exactly which PT teams own each
# card in a given league. Lets us resolve "W. Contreras" unambiguously even
# when three teams each have a different William/Willson Contreras.
_TEAM_CACHE: dict[tuple[str, str, str, str], int] = {}
_TEAM_AMBIGUOUS: set[tuple[str, str, str, str]] = set()
_TEAM_CACHE_BUILT = False


# "C. Kluttz" / "C.Kluttz" — pull (initial, lastname).
# Also handles "J.P. Arencibia" (first + middle initials, each with '.').
# Lastname must be ≥2 chars so we don't accidentally eat one lastname
# letter as a "middle initial" (bug seen: "C. Kluttz" → ('C','luttz')).
# Recognized suffixes stripped from abbreviated/full names before lookup.
# Suffixes may appear with or without a period, with varying case.
_SUFFIXES = {'jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'v'}


def _strip_suffix(name: str) -> str:
    """Remove trailing Jr./Sr./II/III suffixes from a name."""
    if not name:
        return name
    tokens = name.strip().split()
    while tokens and tokens[-1].lower().rstrip('.') in {s.rstrip('.') for s in _SUFFIXES}:
        tokens.pop()
    return ' '.join(tokens)


_ABBREV_RE = re.compile(
    r'^\s*'
    r'([A-Z])\.?'                          # required first initial, optional '.'
    r'(?:\s*[A-Z]\.)*'                     # optional middle initials — REQUIRED '.'
    r'\s+'                                 # mandatory whitespace before lastname
    r'([A-Z][A-Za-z\'\-]{1,}'              # lastname (cap + ≥1 more chars)
    r'(?:\s+[A-Za-z\'\-]+)?)'              # optional space-separated 2nd part ("De La Rosa")
    r'\s*$'
)
_FULLNAME_RE = re.compile(
    r'^\s*([A-Za-z]+)\s+([A-Za-z\'\-]+(?:\s+[A-Za-z\'\-]+)?)\s*$'
)


def _is_abbreviated(name: str) -> bool:
    """Is `name` a box-score abbreviation like 'C. Kluttz'?"""
    if not name or len(name) < 3:
        return False
    # Quick check: first token is one letter (optionally followed by '.')
    first = name.split()[0].rstrip('.')
    return len(first) == 1 and first.isalpha()


def _parse_abbreviated(name: str) -> Optional[tuple[str, str]]:
    """Return (initial, lastname) from an abbreviated name, or None.

    Strips Jr./Sr./II/III suffixes first so "J. Chisholm Jr." → ('J', 'Chisholm').
    """
    if not name:
        return None
    stripped = _strip_suffix(name)
    m = _ABBREV_RE.match(stripped)
    if not m:
        return None
    initial = m.group(1).upper()
    lastname = m.group(2).strip()
    return (initial, lastname)


def _parse_fullname(name: str) -> Optional[tuple[str, str]]:
    """Return (first_initial, lastname) from a full name, or None.

    Strips Jr./Sr./II/III suffixes first.
    """
    if not name:
        return None
    stripped = _strip_suffix(name)
    m = _FULLNAME_RE.match(stripped)
    if not m:
        return None
    first = m.group(1)
    lastname = m.group(2).strip()
    if not first:
        return None
    return (first[0].upper(), lastname)


def _build_cache(conn: sqlite3.Connection) -> None:
    """Scan ``cards`` + ``roster`` to build (initial, lastname) → full_name.

    Rows where the (initial, lastname) pair maps to >1 distinct full name
    are marked AMBIGUOUS and excluded from resolution (we refuse to guess).
    """
    global _CACHE_BUILT
    _ABBREV_TO_FULL.clear()
    _AMBIGUOUS.clear()
    _FULL_NAMES.clear()

    # Pull distinct "first_name last_name" pairs from cards
    seen_pairs: dict[tuple[str, str], set[str]] = {}
    try:
        for r in conn.execute(
            "SELECT DISTINCT first_name, last_name FROM cards "
            "WHERE first_name IS NOT NULL AND last_name IS NOT NULL"
        ).fetchall():
            first = (r[0] or '').strip()
            last = (r[1] or '').strip()
            if not first or not last:
                continue
            full = f"{first} {last}"
            key = (first[0].upper(), last)
            seen_pairs.setdefault(key, set()).add(full)
            _FULL_NAMES.add(full)
    except Exception as e:
        logger.warning("name_resolver cards scan failed: %s", e)

    # Also add from roster (catches players who may not be in cards yet)
    try:
        for r in conn.execute(
            "SELECT DISTINCT player_name FROM roster WHERE player_name IS NOT NULL"
        ).fetchall():
            full = (r[0] or '').strip()
            parsed = _parse_fullname(full)
            if parsed:
                seen_pairs.setdefault(parsed, set()).add(full)
                _FULL_NAMES.add(full)
    except Exception as e:
        logger.warning("name_resolver roster scan failed: %s", e)

    for key, fulls in seen_pairs.items():
        if len(fulls) == 1:
            _ABBREV_TO_FULL[key] = next(iter(fulls))
        else:
            _AMBIGUOUS.add(key)

    _CACHE_BUILT = True
    logger.info(
        "name_resolver cache built: %d unique abbrev→full mappings, %d ambiguous",
        len(_ABBREV_TO_FULL), len(_AMBIGUOUS),
    )


def _ensure_cache(conn: Optional[sqlite3.Connection] = None) -> None:
    if _CACHE_BUILT:
        return
    with _CACHE_LOCK:
        if _CACHE_BUILT:
            return
        if conn is None:
            from app.core.database import get_connection
            conn = get_connection()
        _build_cache(conn)


def refresh_cache() -> None:
    """Force a rebuild of the cache (call after major ingestion)."""
    global _CACHE_BUILT
    with _CACHE_LOCK:
        _CACHE_BUILT = False
    _ensure_cache()


def expand_abbreviated_name(name: str,
                             conn: Optional[sqlite3.Connection] = None) -> str:
    """Return a canonical full name for ``name`` if ``name`` is abbreviated.

    If ``name`` is already a full name, ambiguous, or not in our index,
    returns the input unchanged. Never guesses when (initial, lastname)
    maps to >1 full name.
    """
    if not name:
        return name
    _ensure_cache(conn)
    if not _is_abbreviated(name):
        return name
    parsed = _parse_abbreviated(name)
    if not parsed:
        return name
    if parsed in _AMBIGUOUS:
        logger.debug("name_resolver: refusing to guess ambiguous %r", name)
        return name
    return _ABBREV_TO_FULL.get(parsed, name)


def resolve_to_card_id(
    name: str,
    *,
    prefer_owned: bool = True,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[int]:
    """Return the best-match ``card_id`` for ``name``, or None.

    Steps:
        1. If ``name`` is abbreviated, expand to canonical full name.
        2. Match cards on exact ``first_name + ' ' + last_name = full``.
        3. When ``prefer_owned`` is True, prefer owned cards.
        4. If multiple matches for a full name (name collision across
           different cards), pick by ``card_value`` desc.
    """
    if not name:
        return None
    _ensure_cache(conn)
    if conn is None:
        from app.core.database import get_connection
        conn = get_connection()
    full = expand_abbreviated_name(name, conn)

    # Short-circuit: if we STILL don't have a fullname form, try LIKE on title.
    # Narrow this so we don't match substrings wildly.
    if _is_abbreviated(full):
        parsed = _parse_abbreviated(full)
        if parsed and parsed not in _AMBIGUOUS:
            # Still abbreviated AND not in index — rare; give up.
            return None
        # Ambiguous — refuse.
        return None

    # Exact fullname match via first_name + last_name
    if prefer_owned:
        owned = conn.execute(
            "SELECT card_id FROM cards "
            "WHERE owned >= 1 "
            "  AND (first_name || ' ' || last_name) = ? "
            "ORDER BY card_value DESC LIMIT 1",
            (full,),
        ).fetchone()
        if owned:
            return int(owned[0])

    any_ = conn.execute(
        "SELECT card_id FROM cards "
        "WHERE (first_name || ' ' || last_name) = ? "
        "ORDER BY card_value DESC LIMIT 1",
        (full,),
    ).fetchone()
    return int(any_[0]) if any_ else None


def cache_stats() -> dict:
    """Introspection for diagnostics / tests."""
    return {
        'built': _CACHE_BUILT,
        'unique_mappings': len(_ABBREV_TO_FULL),
        'ambiguous_pairs': len(_AMBIGUOUS),
        'full_name_count': len(_FULL_NAMES),
        'team_cache_built': _TEAM_CACHE_BUILT,
        'team_cache_size': len(_TEAM_CACHE),
        'team_cache_ambiguous': len(_TEAM_AMBIGUOUS),
    }


# ──────────────────────────────────────────────────────────────────────
# Team-scoped resolution (for game_batting / game_pitching)
# ──────────────────────────────────────────────────────────────────────


def _build_team_cache(conn: sqlite3.Connection) -> None:
    """Build (league_id, team_name, initial, lastname) -> card_id from league_rosters.

    A league lineup can contain multiple "J. Rodriguez" entries spread across
    different teams, but within a single team the (initial, lastname) pair
    is essentially always unique. The ~1% of intra-team collisions (twins,
    same-initial brothers) are recorded in _TEAM_AMBIGUOUS and the resolver
    refuses to guess for those.
    """
    global _TEAM_CACHE_BUILT
    _TEAM_CACHE.clear()
    _TEAM_AMBIGUOUS.clear()

    per_key: dict[tuple[str, str, str, str], set[int]] = {}
    try:
        for r in conn.execute(
            "SELECT league_id, team_name, player_name, card_id "
            "FROM league_rosters "
            "WHERE card_id IS NOT NULL AND player_name IS NOT NULL "
            "  AND team_name IS NOT NULL AND league_id IS NOT NULL"
        ).fetchall():
            league_id, team_name, player_name, card_id = r
            parsed = _parse_fullname(player_name)
            if not parsed:
                continue
            initial, lastname = parsed
            key = (league_id, team_name, initial, lastname)
            per_key.setdefault(key, set()).add(int(card_id))
    except Exception as e:
        logger.warning("name_resolver team cache scan failed: %s", e)

    for key, cids in per_key.items():
        if len(cids) == 1:
            _TEAM_CACHE[key] = next(iter(cids))
        else:
            _TEAM_AMBIGUOUS.add(key)

    _TEAM_CACHE_BUILT = True
    logger.info(
        "name_resolver team cache built: %d unique team-scoped mappings, %d ambiguous",
        len(_TEAM_CACHE), len(_TEAM_AMBIGUOUS),
    )


def _ensure_team_cache(conn: Optional[sqlite3.Connection] = None) -> None:
    if _TEAM_CACHE_BUILT:
        return
    with _CACHE_LOCK:
        if _TEAM_CACHE_BUILT:
            return
        if conn is None:
            from app.core.database import get_connection
            conn = get_connection()
        _build_team_cache(conn)


def refresh_team_cache() -> None:
    """Force a rebuild of the team cache (call after ingesting league_rosters)."""
    global _TEAM_CACHE_BUILT
    with _CACHE_LOCK:
        _TEAM_CACHE_BUILT = False
    _ensure_team_cache()


def resolve_to_card_id_with_team(
    name: str,
    team_name: Optional[str],
    league_id: Optional[str],
    *,
    prefer_owned: bool = True,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[int]:
    """Team-scoped card_id resolution for game-log names.

    Resolution order:
        1. If (league_id, team_name, initial, lastname) is in the team cache,
           return that card_id. This handles league-wide ambiguity (many
           "J. Rodriguez" across teams) by using the team as the tie-breaker.
        2. If the team pair is flagged ambiguous (two players on the same
           team with the same initial+lastname), refuse to guess.
        3. Otherwise fall through to the league-wide resolver
           (resolve_to_card_id). That handles names that aren't in
           league_rosters yet — new call-ups, free-agent pool, etc.
    """
    if not name:
        return None

    if team_name and league_id:
        _ensure_team_cache(conn)
        parsed: Optional[tuple[str, str]] = None
        if _is_abbreviated(name):
            parsed = _parse_abbreviated(name)
        else:
            parsed = _parse_fullname(name)
        if parsed:
            key = (league_id, team_name, parsed[0], parsed[1])
            cid = _TEAM_CACHE.get(key)
            if cid is not None:
                return cid
            if key in _TEAM_AMBIGUOUS:
                # Two different players with the same initial+lastname on
                # the same team — without more context we can't pick one.
                return None

    # Fall through to league-wide resolver.
    return resolve_to_card_id(name, prefer_owned=prefer_owned, conn=conn)
