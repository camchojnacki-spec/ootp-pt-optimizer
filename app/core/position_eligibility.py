"""Position eligibility helpers for multi-position recommendation matching.

Why this module exists
----------------------
OOTP tracks each batter's defensive ability at all eight field positions via
``pos_rating_c``, ``pos_rating_1b``, ``pos_rating_2b``, ``pos_rating_3b``,
``pos_rating_ss``, ``pos_rating_lf``, ``pos_rating_cf``, ``pos_rating_rf``.
The card's ``position_name`` is just the primary — the position where the
card earns top-line ratings — but a CF-primary card with ``pos_rating_rf=67``
is perfectly usable in RF, and a 3B-primary with ``pos_rating_lf=50`` covers
the outfield corner fine.

Before this helper, every recommendation query used
``WHERE position_name = ?``, so a CF-primary like George Van Haltren never
surfaced as an LF or RF upgrade even when his meta clearly beat the incumbent
corner outfielder. The user flagged this as a recurring miss (see
``memory/project_position_flexibility_gap.md``); this module is the fix.

Design
------
Two thresholds:

* ``ELIGIBILITY_THRESHOLD`` (30) — the rating below which a card is NOT
  considered playable for recommendation purposes. Purposefully a touch above
  ``meta_scoring._POS_PLAYABLE_THRESHOLD`` (20, the "playable with defensive
  cost" floor) because recommendations should only surface cards whose
  defensive fit won't embarrass the user. 30 is also just below
  Van Haltren's LF=32, so the user's poster-child case qualifies.
* ``RECOMMENDED_THRESHOLD`` (60) — at or above this, treat the card as a
  natural fit. Below, apply a small meta penalty proportional to how far
  below 60 the rating sits.

The meta penalty is deliberately small (capped at ~15 meta). It exists to
avoid a low-rating corner-OF card silently beating a true OF specialist on
raw meta alone; it does NOT exist to hide multi-position options from the
user. A rating of 30 costs ~10 meta — Van Haltren's 649 at LF=32 becomes
~640, which still clobbers Tyler Soderstrom's 603.
"""
from __future__ import annotations

from typing import Iterable

# Ordered to match the batting field positions everywhere else in the code.
BATTING_POSITIONS: tuple[str, ...] = ('C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF')

# Canonical map from position abbreviation to the ``cards`` table column that
# holds that position's defensive rating. DH is omitted — any batter can DH,
# so eligibility is trivially True.
POS_RATING_COL: dict[str, str] = {
    'C':  'pos_rating_c',
    '1B': 'pos_rating_1b',
    '2B': 'pos_rating_2b',
    '3B': 'pos_rating_3b',
    'SS': 'pos_rating_ss',
    'LF': 'pos_rating_lf',
    'CF': 'pos_rating_cf',
    'RF': 'pos_rating_rf',
}

# Minimum rating required for a card to appear as a recommendation at a
# non-primary position. Tuned just below Van Haltren's LF=32 so the
# "qualified to learn" case surfaces; tuned above noise (<20) so bad fits
# don't flood the list.
ELIGIBILITY_THRESHOLD: int = 30

# Rating at or above which we consider the defensive fit clean — no meta
# penalty applied. Below this, we scale a penalty in.
RECOMMENDED_THRESHOLD: int = 60

# Cap on the meta penalty a non-primary assignment can incur. Keeps the
# penalty from swamping the underlying meta signal for genuinely strong
# cards playing out of position.
MAX_POSITION_PENALTY: float = 15.0


def get_eligible_positions(
    card_row: dict,
    threshold: int = ELIGIBILITY_THRESHOLD,
) -> list[str]:
    """Return every batting position ``card_row`` is eligible to play.

    The card's ``position_name`` is always included (primary position is
    always eligible). Secondary positions are included when their
    ``pos_rating_*`` value meets ``threshold``.

    ``card_row`` is any mapping with the relevant keys — a ``sqlite3.Row``,
    a plain ``dict``, anything supporting ``.get``.
    """
    eligible: list[str] = []
    primary = card_row.get('position_name')
    if primary in BATTING_POSITIONS:
        eligible.append(primary)

    for pos, col in POS_RATING_COL.items():
        if pos == primary:
            continue
        rating = card_row.get(col)
        try:
            if rating is not None and float(rating) >= threshold:
                eligible.append(pos)
        except (TypeError, ValueError):
            continue
    return eligible


def position_meta_penalty(
    card_row: dict,
    target_pos: str,
) -> float:
    """Return a meta penalty (>= 0) for assigning ``card_row`` to ``target_pos``.

    Zero penalty when the card's primary position matches, when the rating
    at the target sits at or above ``RECOMMENDED_THRESHOLD`` (60), or when
    no rating is available (conservatively charge nothing rather than
    hide the option).

    Below 60, penalty scales linearly toward a cap at ``MAX_POSITION_PENALTY``
    so a rating of ``RECOMMENDED_THRESHOLD`` costs 0 and a rating of 0 costs
    the full cap.
    """
    if not target_pos or target_pos not in POS_RATING_COL:
        return 0.0
    if card_row.get('position_name') == target_pos:
        return 0.0

    rating = card_row.get(POS_RATING_COL[target_pos])
    try:
        r = float(rating) if rating is not None else None
    except (TypeError, ValueError):
        r = None
    if r is None or r >= RECOMMENDED_THRESHOLD:
        return 0.0

    # Linear scale: 60 → 0 penalty, 0 → MAX_POSITION_PENALTY.
    penalty = (RECOMMENDED_THRESHOLD - r) / RECOMMENDED_THRESHOLD * MAX_POSITION_PENALTY
    return max(0.0, min(MAX_POSITION_PENALTY, penalty))


def build_eligible_where_clause(
    target_pos: str,
    threshold: int = ELIGIBILITY_THRESHOLD,
    table_alias: str = '',
) -> tuple[str, list]:
    """Build a SQL WHERE fragment that matches cards eligible at ``target_pos``.

    Returns a tuple ``(fragment, params)`` suitable to splice into a query.
    The fragment is wrapped in parentheses so it's safe to AND with other
    clauses. ``table_alias`` (e.g. ``"c"``) prefixes the column references;
    pass ``""`` for a bare query.

    Example::

        frag, params = build_eligible_where_clause('LF', table_alias='c')
        # frag = "(c.position_name = ? OR c.pos_rating_lf >= ?)"
        # params = ['LF', 30]

    For ``DH`` the fragment matches any non-pitcher (``pitcher_role IS NULL``)
    since anyone can DH; ``threshold`` is ignored.
    """
    prefix = f"{table_alias}." if table_alias else ""

    if target_pos == 'DH':
        return (f"({prefix}pitcher_role IS NULL)", [])

    if target_pos not in POS_RATING_COL:
        # Fall back to exact-match for unknown positions (e.g. pitcher roles).
        return (f"({prefix}position_name = ?)", [target_pos])

    col = POS_RATING_COL[target_pos]
    frag = f"({prefix}position_name = ? OR {prefix}{col} >= ?)"
    return (frag, [target_pos, threshold])


def select_rating_columns(table_alias: str = '') -> str:
    """Return the comma-separated list of pos_rating columns for a SELECT.

    Useful when you need the rating values downstream (e.g. to compute
    penalties). Always returns a trailing-newline-free fragment::

        f"SELECT card_id, meta_score_batting, {select_rating_columns('c')} FROM cards c ..."
    """
    prefix = f"{table_alias}." if table_alias else ""
    return ", ".join(f"{prefix}{col}" for col in POS_RATING_COL.values())


def format_position_annotation(
    card_row: dict,
    target_pos: str,
) -> str:
    """Human-readable annotation for recommendations.

    Returns ``""`` when the card is playing its primary position.
    Otherwise returns e.g. ``" (played as LF, rating 32)"`` so the reason
    string in the recommendation makes the position switch explicit.
    """
    primary = card_row.get('position_name')
    if not primary or primary == target_pos:
        return ""
    rating = card_row.get(POS_RATING_COL.get(target_pos, ''))
    if rating is None:
        return f" (played as {target_pos})"
    try:
        return f" (played as {target_pos}, rating {int(float(rating))})"
    except (TypeError, ValueError):
        return f" (played as {target_pos})"


def is_eligible(card_row: dict, target_pos: str, threshold: int = ELIGIBILITY_THRESHOLD) -> bool:
    """Quick predicate: can this card play ``target_pos`` at ``threshold`` or above?

    Wraps ``get_eligible_positions`` for callers that only need a yes/no.
    """
    return target_pos in get_eligible_positions(card_row, threshold=threshold)


__all__ = [
    'BATTING_POSITIONS',
    'POS_RATING_COL',
    'ELIGIBILITY_THRESHOLD',
    'RECOMMENDED_THRESHOLD',
    'MAX_POSITION_PENALTY',
    'get_eligible_positions',
    'position_meta_penalty',
    'build_eligible_where_clause',
    'select_rating_columns',
    'format_position_annotation',
    'is_eligible',
]
