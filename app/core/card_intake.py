"""New-card intake detection.

Every refresh, diff the current ``cards WHERE owned=1`` set against the
``card_intake`` tracking table to find cards that appeared since last
seen. Each newly-seen card gets:

    1. An ``intake`` row (card_id, first_seen_at).
    2. A ``recommendation_log`` row (``rec_type='intake'``) so the LLM
       council reviews "where should this card go on the roster?".
    3. Surfaced in the UI's "New Arrivals" section until acknowledged.

This closes the loop: acquiring a card in OOTP → reincarnated as a rec
with LLM commentary → user sees it with deployment guidance.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _best_slot_for_card(card: dict) -> Optional[str]:
    """Quick rating-based 'where does this card fit' guess.

    Batters: return their primary position.
    Pitchers: return SP or RP based on stamina.
    """
    pos = (card.get('position_name') or '').upper()
    if pos in ('SP', 'RP', 'CL'):
        return pos
    role = (card.get('pitcher_role_name') or '').upper()
    if role in ('SP', 'RP', 'CL'):
        return role
    if card.get('pitcher_role'):
        stam = card.get('stamina') or 0
        return 'SP' if stam >= 65 else 'RP'
    # Batter
    return pos or None


def detect_new_cards() -> int:
    """Scan for owned cards not yet in card_intake. Returns count logged.

    Each newly-seen card creates:
      - card_intake row
      - recommendation_log row (rec_type='intake') with a deployment hint
    """
    from app.core.database import get_connection
    conn = get_connection()

    try:
        new_cards = conn.execute(
            """
            SELECT c.card_id, c.card_title, c.position_name,
                   c.pitcher_role_name, c.pitcher_role,
                   c.stamina, c.tier_name, c.card_type,
                   c.meta_score_batting, c.meta_score_pitching,
                   c.card_value
            FROM cards c
            LEFT JOIN card_intake ci ON ci.card_id = c.card_id
            WHERE c.owned = 1
              AND c.card_id IS NOT NULL
              AND ci.card_id IS NULL
            LIMIT 200
            """
        ).fetchall()
    except Exception as e:
        logger.warning("detect_new_cards query failed: %s", e)
        return 0

    if not new_cards:
        return 0

    # Bulk insert intake rows
    logged = 0
    for c in new_cards:
        cid = c['card_id']
        slot = _best_slot_for_card(dict(c))
        meta = c['meta_score_batting'] or c['meta_score_pitching']
        reasoning = (
            f"New arrival: {c['card_title']} "
            f"(OVR {c['card_value']}, meta {meta or '?'}). "
            f"Suggested slot: {slot or '—'}."
        )
        try:
            # Try to create the intake rec first so we can FK it
            cur = conn.execute(
                """
                INSERT INTO recommendation_log
                    (source, rec_type, pos, player_name, player_card_id,
                     reasoning, league_id, verdict)
                VALUES ('meta_engine', 'intake', ?, ?, ?, ?, NULL, 'pending')
                """,
                (slot, c['card_title'], cid, reasoning),
            )
            rec_id = int(cur.lastrowid)
            conn.execute(
                "INSERT OR IGNORE INTO card_intake (card_id, rec_id) VALUES (?, ?)",
                (cid, rec_id),
            )
            logged += 1
        except Exception as e:
            logger.debug("intake insert failed for card_id=%s: %s", cid, e)

    conn.commit()
    return logged


def get_unacknowledged_intake(limit: int = 20) -> list[dict]:
    """Return new arrivals the user hasn't acknowledged yet."""
    from app.core.database import get_connection
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT ci.card_id, ci.first_seen_at, ci.rec_id,
               c.card_title, c.position_name, c.pitcher_role_name,
               c.tier_name, c.card_value,
               c.meta_score_batting, c.meta_score_pitching,
               rl.reasoning
        FROM card_intake ci
        INNER JOIN cards c ON c.card_id = ci.card_id
        LEFT JOIN recommendation_log rl ON rl.id = ci.rec_id
        WHERE ci.acknowledged_at IS NULL
        ORDER BY ci.first_seen_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def acknowledge_intake(card_id: int) -> None:
    """Mark a new-arrival notification as acknowledged."""
    from app.core.database import get_connection
    conn = get_connection()
    conn.execute(
        "UPDATE card_intake SET acknowledged_at = CURRENT_TIMESTAMP "
        "WHERE card_id = ?",
        (card_id,),
    )
    conn.commit()
