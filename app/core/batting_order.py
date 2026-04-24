"""Engine-recommended batting order (1–9).

Given the 9 starters you're playing, produce the optimal lineup slot for
each based on ratings. This is prescriptive — "who SHOULD bat where" —
rather than descriptive (what you're currently doing). Used by the chain
table's BO column so the user can see which slot each hitter belongs in.

Scoring rules (high score = better fit for that slot):
  1 Leadoff:   OBP + speed (get on, steal)
  2 Two-hole:  contact + OBP + some gap (move runners, avoid DP)
  3 Three:     best overall hitter (everything matters)
  4 Cleanup:   power focus (drive in runs)
  5 Five:      power/RBI secondary
  6–9:         remaining by meta descending

Greedy assignment — for each slot in order, pick the remaining player
with the highest slot-fit score; remove from pool; move to next slot.
"""
from __future__ import annotations

from typing import Optional


def compute_batting_order(
    players: list[dict],
) -> dict[str, int]:
    """Return {player_name: slot_1_through_9}.

    ``players`` must each have keys: 'player_name', plus rating columns
    'contact', 'gap_power', 'power', 'eye', 'avoid_ks', 'speed',
    'stealing', and optionally 'meta_score' for slot-6-9 ordering.
    """
    if not players:
        return {}

    def _f(p: dict, key: str) -> float:
        v = p.get(key)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    # Score each player for each slot
    scored: list[dict] = []
    for p in players:
        con = _f(p, 'contact')
        gap = _f(p, 'gap_power')
        pwr = _f(p, 'power')
        eye = _f(p, 'eye')
        avk = _f(p, 'avoid_ks')
        spd = _f(p, 'speed')
        stl = _f(p, 'stealing')
        meta = _f(p, 'meta_score')
        scored.append({
            'player_name': p.get('player_name'),
            'leadoff':  con * 1.2 + eye * 1.5 + avk * 0.8 + spd * 0.6 + stl * 0.3,
            'two_hole': con * 1.3 + eye * 1.2 + gap * 0.8 + avk * 0.7,
            'three':    con * 1.0 + gap * 1.2 + pwr * 1.0 + eye * 1.0 + avk * 0.5,
            'cleanup':  pwr * 1.5 + gap * 1.3 + con * 0.6 + eye * 0.5,
            'five':     pwr * 1.2 + gap * 1.1 + con * 0.7 + eye * 0.6,
            'meta':     meta,
        })

    # Greedy-assign slots 1..5 by slot-specific score
    slot_keys = [
        (1, 'leadoff'),
        (2, 'two_hole'),
        (3, 'three'),
        (4, 'cleanup'),
        (5, 'five'),
    ]
    order: dict[str, int] = {}
    remaining = list(scored)
    for slot, key in slot_keys:
        if not remaining:
            break
        best = max(remaining, key=lambda p: p.get(key, 0))
        name = best['player_name']
        if name:
            order[name] = slot
        remaining.remove(best)

    # Slots 6..9 by meta descending
    remaining.sort(key=lambda p: -(p.get('meta') or 0))
    for i, p in enumerate(remaining):
        slot = 6 + i
        if slot > 9:
            break
        name = p['player_name']
        if name:
            order[name] = slot
    return order
