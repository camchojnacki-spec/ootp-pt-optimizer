"""Anti-oscillation guard for engine recommendations.

Problem: the meta engine currently has no memory of recent decisions. If
a user follows a "swap X out for Y" rec, the next refresh may see Y's
rating-only meta drop below X's (which has accumulated real PA-performance
overlay data) and recommend swapping BACK. The user flip-flops.

Solution: a hysteresis filter. Before emitting a rec that would reverse
a recently-actioned swap, require the delta to clear a stability floor
(default 50 meta points). Below that, suppress the rec.

Public API:
    - should_suppress_reversal(pos, proposed_in, proposed_out, delta,
                                 conn=None, lookback_hours=72,
                                 min_flip_delta=50.0) -> (bool, reason)

The filter reads ``recommendation_log`` for actioned recs in the lookback
window and checks whether the proposed swap is a direct reversal.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _names_match(a: Optional[str], b: Optional[str]) -> bool:
    """Forgiving name match — card titles often differ in prefix/suffix."""
    if not a or not b:
        return False
    a_l = str(a).lower()
    b_l = str(b).lower()
    # Strip common OOTP card-title prefixes so "MLB 2026 Live SS Bo Bichette TOR"
    # matches "Bo Bichette".
    for prefix in ('mlb 2026 live', 'mlb 2025 live', 'mlb 2024 live'):
        a_l = a_l.replace(prefix, '').strip()
        b_l = b_l.replace(prefix, '').strip()
    return (a_l in b_l) or (b_l in a_l)


def should_suppress_reversal(
    pos: str,
    proposed_in: str,            # the player the rec wants to install
    proposed_out: Optional[str], # the player being replaced (current starter)
    delta: float,                # meta points claimed gain
    *,
    conn=None,
    lookback_hours: int = 72,
    min_flip_delta: float = 50.0,
) -> tuple[bool, str]:
    """Return (suppress, reason).

    Suppression triggers if:
      * Within ``lookback_hours``, a followed rec moved ``proposed_in``
        OUT of this slot AND installed ``proposed_out`` (the reverse swap).
      * AND the claimed ``delta`` is below ``min_flip_delta``.

    In other words: the user already tried this, then reversed it. Don't
    recommend flipping it BACK unless the gain is substantial.
    """
    if not pos or not proposed_in:
        return False, ''
    try:
        from app.core.database import get_connection
        if conn is None:
            conn = get_connection()
        rows = conn.execute(
            """
            SELECT rec_type, player_name, from_player, expected_delta,
                   action_type, created_at
            FROM recommendation_log
            WHERE pos = ?
              AND created_at > datetime('now', ?)
              AND rec_type IN ('promote', 'platoon', 'buy')
            ORDER BY created_at DESC
            """,
            (pos, f'-{int(lookback_hours)} hours'),
        ).fetchall()
    except Exception as e:
        logger.debug("hysteresis lookup failed: %s", e)
        return False, ''

    for r in rows:
        prior_in = r['player_name']   # who WAS recommended to install
        prior_out = r['from_player']  # who was recommended to remove
        # A "reversal" is: the current proposal flips roles with a prior one.
        if (_names_match(prior_in, proposed_out)
                and _names_match(prior_out, proposed_in)):
            if delta < min_flip_delta:
                return True, (
                    f"Suppressed: {proposed_in} ↔ {proposed_out} was "
                    f"recommended the other way within the last "
                    f"{lookback_hours}h (action={r['action_type'] or 'pending'}). "
                    f"Current delta {delta:.0f} < stability floor "
                    f"{min_flip_delta:.0f} — avoiding oscillation."
                )
            else:
                return False, (
                    f"Reversal allowed: delta {delta:.0f} clears "
                    f"stability floor {min_flip_delta:.0f}."
                )
    return False, ''


def filter_upgrade_plan(upgrade_plan: list[dict],
                        *,
                        min_flip_delta: float = 50.0,
                        lookback_hours: int = 72) -> list[dict]:
    """Mutate entries in ``upgrade_plan`` to suppress oscillating owned
    promotions. Adds an ``oscillation_suppressed`` flag + clears the
    owned_name / owned_delta if we decide to hide the rec.

    Market upgrades are NOT suppressed (a buy is rarely a reversal — you
    don't un-buy a card). We only filter owned promotions.

    Returns the list back for chaining.
    """
    try:
        from app.core.database import get_connection
        conn = get_connection()
    except Exception:
        return upgrade_plan

    for u in upgrade_plan:
        owned_name = u.get('owned_name')
        current_name = u.get('current_name')
        delta = u.get('owned_delta') or 0
        pos = u.get('pos') or ''
        if not owned_name or delta <= 0:
            continue
        suppress, reason = should_suppress_reversal(
            pos, owned_name, current_name, delta,
            conn=conn,
            lookback_hours=lookback_hours,
            min_flip_delta=min_flip_delta,
        )
        if suppress:
            u['oscillation_suppressed'] = True
            u['oscillation_reason'] = reason
            u['owned_name'] = None
            u['owned_delta'] = 0
            u['owned_action'] = 'stable'
    return upgrade_plan
