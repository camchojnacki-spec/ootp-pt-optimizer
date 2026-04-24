"""Tier-aware competitive context.

OOTP Perfect Team has 8 league tiers: Rookie → Stone → Iron → Bronze →
Silver → Gold → Diamond → Perfect. The simulation engine is identical
across all tiers — cards play the same everywhere — but opponent roster
quality differs enormously. A 550-meta roster dominates Iron and gets
swept in Diamond.

Before this module existed the app was tier-blind: it benchmarked every
roster against Diamond+ rosters, recommended 50k-PP cards to a Low Bronze
team with 4,970 PP, and labeled every player "Cold" because the
perf-to-meta mapping assumed Diamond-tier expectations.

This module provides the tier-aware context that makes recommendations
actionable:

- ``tier_benchmarks(league_id)`` — meta percentiles per position drawn from
  the user's own history (what do competitive rosters in *this* league look
  like?), not hardcoded tier tables.
- ``tier_budget_ceiling(tier)`` — rough price guardrail so Buy Recs don't
  suggest diamond cards to a bronze team.
- ``promotion_readiness(league_id)`` — gap to the next tier's competitive
  threshold, per position, so the user can see where to invest.
- ``next_tier(current)`` / ``tier_rank(tier)`` — ordering helpers.

The benchmarks are **data-driven where possible** (queries against
``player_history`` and ``league_team_stats``) so they improve as the user
accumulates more snapshots. Where no data exists for a tier (e.g., the
user has never played Gold), we fall back to order-of-magnitude heuristics
with a ``source='heuristic'`` tag so the UI can flag the lower confidence.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from app.core.database import get_connection, load_config

logger = logging.getLogger(__name__)


# Canonical tier order. OOTP's internal tiers — we support the "Low/High"
# sub-tier variant the UI sometimes shows (lb124 reads as "Low Bronze") by
# stripping the modifier when ranking.
TIER_ORDER = [
    'Rookie', 'Stone', 'Iron', 'Bronze', 'Silver', 'Gold', 'Diamond', 'Perfect',
]

# Rough budget ceilings per tier — the max single-card PP you'd realistically
# spend for a competitive card at that tier. Used as a hint, not a hard cap.
# Values informed by Low Bronze observed pricing (avg 500 PP, high-end 4k PP)
# and scaled by tier rank. A user with an unusual PP budget can override via
# config.yaml:tier_budget_ceilings.
_DEFAULT_BUDGET_CEILINGS = {
    'Rookie':   500,
    'Stone':    1_000,
    'Iron':     2_500,
    'Bronze':   5_000,
    'Silver':   15_000,
    'Gold':     40_000,
    'Diamond':  120_000,
    'Perfect':  500_000,
}

# Heuristic meta P50 per tier (used when no data exists for that tier yet).
# These are educated guesses for the 50th percentile roster card meta;
# the module prefers real data from player_history when it's available.
_HEURISTIC_TIER_P50 = {
    'Rookie':   350,
    'Stone':    420,
    'Iron':     490,
    'Bronze':   560,
    'Silver':   630,
    'Gold':     700,
    'Diamond':  780,
    'Perfect':  860,
}


def _normalize_tier(tier: str | None) -> str | None:
    """Strip 'Low'/'High' modifiers and normalize casing.

    ``"Low Bronze"`` → ``"Bronze"``. Returns None for empty/unparseable input.
    """
    if not tier or not isinstance(tier, str):
        return None
    parts = tier.strip().split()
    if not parts:
        return None
    last = parts[-1].capitalize()
    if last in TIER_ORDER:
        return last
    # Maybe the whole string is a tier already
    cap = tier.strip().capitalize()
    if cap in TIER_ORDER:
        return cap
    return None


def tier_rank(tier: str | None) -> int:
    """Return the 0-based rank of a tier, or -1 if unknown."""
    norm = _normalize_tier(tier)
    return TIER_ORDER.index(norm) if norm in TIER_ORDER else -1


def next_tier(current: str | None) -> str | None:
    """Return the name of the tier above ``current``, or None if at the top."""
    rank = tier_rank(current)
    if rank < 0 or rank >= len(TIER_ORDER) - 1:
        return None
    return TIER_ORDER[rank + 1]


def get_active_league_tier(conn=None) -> Optional[str]:
    """Look up the active league's tier.

    Checks the ``leagues`` table for the active_league from config. Returns
    the raw tier string (may include Low/High modifier) or None if not set.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        try:
            cfg = load_config()
            league_id = cfg.get('active_league', 'lb124')
        except Exception:
            league_id = 'lb124'
        try:
            row = conn.execute(
                "SELECT league_tier FROM leagues WHERE league_id = ?",
                (league_id,),
            ).fetchone()
        except sqlite3.Error:
            return None
        return row[0] if row and row[0] else None
    finally:
        if close_conn:
            conn.close()


def tier_budget_ceiling(tier: str | None) -> int:
    """Single-card PP ceiling appropriate for a given tier.

    Cards priced above this are treated as over-budget for that tier —
    they may still show in Buy Recs, but with a lower priority and a
    "save up" flag instead of a green-light upgrade.
    """
    norm = _normalize_tier(tier)
    if norm and norm in _DEFAULT_BUDGET_CEILINGS:
        return _DEFAULT_BUDGET_CEILINGS[norm]
    # Unknown tier → assume Bronze ceiling (conservative middle ground)
    return _DEFAULT_BUDGET_CEILINGS['Bronze']


def tier_meta_p50(tier: str | None, conn=None) -> float:
    """Typical (50th percentile) owned-card meta for a tier.

    Prefers real data from ``player_history`` joined with ``leagues`` on
    ``league_tier``. Falls back to a heuristic table when no history exists
    for that tier in this user's DB.
    """
    norm = _normalize_tier(tier)
    if norm is None:
        return _HEURISTIC_TIER_P50['Bronze']

    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        try:
            rows = conn.execute("""
                SELECT ph.meta_score
                FROM player_history ph
                JOIN leagues l ON l.league_id = ph.league_id
                WHERE ph.meta_score IS NOT NULL
                  AND ph.meta_score > 0
                  AND UPPER(l.league_tier) LIKE ?
            """, (f'%{norm.upper()}%',)).fetchall()
        except sqlite3.Error:
            rows = []
        metas = sorted(float(r[0]) for r in rows if r and r[0] is not None)
        if len(metas) >= 30:
            # Enough data for a real P50 — pick the median.
            mid = len(metas) // 2
            if len(metas) % 2 == 0:
                return (metas[mid - 1] + metas[mid]) / 2.0
            return metas[mid]
        return float(_HEURISTIC_TIER_P50.get(norm, 500))
    finally:
        if close_conn:
            conn.close()


def tier_benchmarks(league_id: str | None = None, conn=None) -> dict:
    """Return per-position meta percentile benchmarks for the active league.

    Returns a dict shaped like::

        {
            'league_id': 'lb124',
            'league_tier': 'Low Bronze',
            'tier_normalized': 'Bronze',
            'sample_size': 430,
            'source': 'history' | 'heuristic',
            'positions': {
                'C':  {'p50': 540, 'p75': 620, 'p90': 700, 'n': 45},
                'SS': {'p50': 560, 'p75': 630, 'p90': 710, 'n': 52},
                ...
            },
            'overall': {'p50': 560, 'p75': 640, 'p90': 720, 'n': 430},
        }

    Positions with < 10 samples fall back to the overall percentiles so the
    UI still has something to display for rare positions.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        try:
            cfg = load_config()
            league_id = league_id or cfg.get('active_league', 'lb124')
        except Exception:
            league_id = league_id or 'lb124'

        tier_row = None
        try:
            tier_row = conn.execute(
                "SELECT league_tier FROM leagues WHERE league_id = ?",
                (league_id,),
            ).fetchone()
        except sqlite3.Error:
            pass
        league_tier = tier_row[0] if tier_row else None
        tier_norm = _normalize_tier(league_tier) or 'Bronze'

        # Pull all owned-card meta snapshots for this league from player_history.
        rows = []
        try:
            rows = conn.execute("""
                SELECT position, meta_score
                FROM player_history
                WHERE league_id = ?
                  AND meta_score IS NOT NULL
                  AND meta_score > 0
            """, (league_id,)).fetchall()
        except sqlite3.Error:
            pass

        positions: dict[str, list[float]] = {}
        all_metas: list[float] = []
        for r in rows:
            pos = (r[0] or '').strip().upper()
            meta = float(r[1] or 0)
            if meta <= 0:
                continue
            all_metas.append(meta)
            if pos:
                positions.setdefault(pos, []).append(meta)

        def _pcts(vals: list[float]) -> dict:
            if not vals:
                return {'p50': None, 'p75': None, 'p90': None, 'n': 0}
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            def q(p: float) -> float:
                idx = min(int(p * n), n - 1)
                return round(vals_sorted[idx], 1)
            return {'p50': q(0.50), 'p75': q(0.75), 'p90': q(0.90), 'n': n}

        overall = _pcts(all_metas)
        position_benchmarks = {}
        for pos, vals in positions.items():
            bench = _pcts(vals)
            # Fall back to overall for sparse positions.
            if bench['n'] < 10:
                bench = {**overall, 'fallback': True}
            position_benchmarks[pos] = bench

        source = 'history' if overall['n'] >= 30 else 'heuristic'
        if source == 'heuristic':
            # Backfill from heuristic table so the UI has something sensible.
            p50 = _HEURISTIC_TIER_P50.get(tier_norm, 500)
            overall = {'p50': p50, 'p75': p50 + 80, 'p90': p50 + 160, 'n': overall['n']}

        return {
            'league_id': league_id,
            'league_tier': league_tier,
            'tier_normalized': tier_norm,
            'sample_size': overall['n'],
            'source': source,
            'positions': position_benchmarks,
            'overall': overall,
            'budget_ceiling': tier_budget_ceiling(league_tier),
        }
    finally:
        if close_conn:
            conn.close()


def promotion_readiness(league_id: str | None = None, conn=None) -> dict:
    """Gap analysis between the current roster and the next-tier competitive threshold.

    Returns::

        {
            'current_tier': 'Bronze',
            'target_tier': 'Silver',
            'target_threshold_p75': 630,   # proxy for competitive Silver
            'roster_p50': 560,
            'overall_gap': 70,             # meta points to close
            'positions': {
                'SS': {'current_meta': 520, 'target_meta': 650, 'gap': 130, 'priority': 'high'},
                'C':  {'current_meta': 610, 'target_meta': 620, 'gap': 10,  'priority': 'low'},
                ...
            },
            'notes': [...],
        }

    Positions where ``gap > 50`` are flagged high priority; that's where
    investment moves the needle for promotion.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        try:
            cfg = load_config()
            league_id = league_id or cfg.get('active_league', 'lb124')
        except Exception:
            league_id = league_id or 'lb124'

        tier_row = conn.execute(
            "SELECT league_tier FROM leagues WHERE league_id = ?",
            (league_id,),
        ).fetchone()
        current_tier = tier_row[0] if tier_row else None
        current_norm = _normalize_tier(current_tier)
        target_norm = next_tier(current_norm) if current_norm else None

        # Target threshold: P75 of the target tier's meta. If we have real
        # data for that tier, use it; otherwise use the heuristic P50 + 80
        # (a rough proxy for "competitive rather than median").
        target_p50 = tier_meta_p50(target_norm, conn=conn) if target_norm else None
        target_threshold = (target_p50 + 80) if target_p50 else None

        # Current roster: active starters + rotation + closer + bullpen.
        try:
            roster_rows = conn.execute("""
                SELECT position, player_name, meta_score
                FROM roster_current
                WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
                  AND meta_score IS NOT NULL
            """).fetchall()
        except sqlite3.Error:
            roster_rows = []

        position_gaps = {}
        all_metas = []
        for r in roster_rows:
            pos = (r[0] or '').strip().upper()
            name = r[1] or ''
            meta = float(r[2] or 0)
            if meta <= 0 or not pos:
                continue
            all_metas.append(meta)
            # For multi-slot positions (SP, RP), keep the weakest occupant
            # since that's where an upgrade lands.
            existing = position_gaps.get(pos)
            if existing is None or meta < existing['current_meta']:
                position_gaps[pos] = {
                    'current_meta': round(meta, 1),
                    'player_name': name,
                }

        if target_threshold is None:
            target_threshold_for_pos = None
        else:
            target_threshold_for_pos = round(target_threshold, 0)

        for pos, entry in position_gaps.items():
            if target_threshold_for_pos is None:
                entry['target_meta'] = None
                entry['gap'] = None
                entry['priority'] = 'unknown'
            else:
                gap = round(target_threshold_for_pos - entry['current_meta'], 1)
                entry['target_meta'] = target_threshold_for_pos
                entry['gap'] = gap
                if gap <= 20:
                    entry['priority'] = 'met'
                elif gap <= 50:
                    entry['priority'] = 'low'
                elif gap <= 120:
                    entry['priority'] = 'medium'
                else:
                    entry['priority'] = 'high'

        # Overall roster P50
        if all_metas:
            roster_sorted = sorted(all_metas)
            mid = len(roster_sorted) // 2
            roster_p50 = roster_sorted[mid] if len(roster_sorted) % 2 else (
                roster_sorted[mid - 1] + roster_sorted[mid]) / 2.0
        else:
            roster_p50 = None

        overall_gap = (
            round(target_threshold - roster_p50, 1)
            if (target_threshold is not None and roster_p50 is not None)
            else None
        )

        notes = []
        if not target_norm:
            notes.append("Already at the top tier — no promotion target.")
        if not roster_rows:
            notes.append("Active roster is empty; import a roster CSV to enable promotion analysis.")

        return {
            'current_tier': current_tier,
            'current_tier_normalized': current_norm,
            'target_tier': target_norm,
            'target_threshold_p75': round(target_threshold, 0) if target_threshold else None,
            'roster_p50': round(roster_p50, 1) if roster_p50 else None,
            'overall_gap': overall_gap,
            'positions': position_gaps,
            'notes': notes,
        }
    finally:
        if close_conn:
            conn.close()
