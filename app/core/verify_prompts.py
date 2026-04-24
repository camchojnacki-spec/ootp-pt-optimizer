"""Prompt templates for the LLM verification council.

Each template takes a meta-engine recommendation + structured context and
asks the LLM to verify. Output is a structured verdict the app can parse:

    VERDICT: CONCUR | DISSENT | NEEDS_MORE_INFO
    CONFIDENCE: 0-10
    REASONING: <one paragraph, cite specifics>
    RISKS: <one line or "none">
    ALTERNATIVE: <optional — if DISSENT, what would you do instead?>

Templates by rec_type:
    - buy, promote, platoon, bench, demote, sell
    - strategy_slider (team-level)
    - player_strategy (per-player override)
    - hook_sp, hook_rp

The caller assembles the full prompt by combining:
    * SYSTEM_PREAMBLE (one per app, sets expert persona)
    * template_for(rec_type) — the structured verify ask
    * build_context(rec, player_row, meta_explainer, observed_stats)
      — facts about the player, team, league
    * CLOSING — output format spec

Consumers: ``app/core/council.py``.
"""
from __future__ import annotations

from typing import Optional


SYSTEM_PREAMBLE = """You are an OOTP 27 Perfect Team strategy analyst
reviewing a recommendation from a meta-analysis engine. The engine has
extensive data: calibrated rating weights, observed WAR correlation,
residual analysis, league-adjusted overlays (OPS+, OBP, ISO, FIP, BB/9,
BABIP), clutch events, inherited-runner stranding, park factors, and
platoon splits.

Your job is NOT to make the recommendation from scratch — the engine did
that. Your job is to VERIFY: does this recommendation hold up given the
specific context? Look for blind spots, counterfactuals, and edge cases
the engine may have missed. Be direct. If you agree, say so and explain
why. If you disagree, dissent with a concrete alternative."""


OUTPUT_FORMAT = """Reply in exactly this structure (no preamble):

VERDICT: <CONCUR | DISSENT | NEEDS_MORE_INFO>
CONFIDENCE: <integer 0-10>
REASONING: <2-4 sentences, cite specific numbers from the context>
RISKS: <one line of what could go wrong, or "none">
ALTERNATIVE: <if DISSENT, name the alternative player/setting and why>
"""


# ──────────────────────────────────────────────────────────────────────
# Per-rec-type templates
# ──────────────────────────────────────────────────────────────────────

BUY_TEMPLATE = """An AUCTION-HOUSE BUY has been recommended.

ENGINE'S PICK:
  Slot: {pos}
  Target: {player_name}
  Replacing: {from_player}
  Expected meta gain: {expected_delta:+.0f} points
  Cost: {cost_pp} PP
  Reason (engine): {reasoning}

{context}

Evaluate:
- Is the expected meta gain realistic given the target's ratings + observed stats?
- Is the cost justified given the meta delta and the user's PP budget?
- Are there hidden risks (platoon hole, role mismatch, stamina, hot/cold streak)?
- Would a different acquisition deliver more WAR per PP?
"""


PROMOTE_TEMPLATE = """A ROSTER PROMOTION (bench → active) has been recommended.

ENGINE'S PICK:
  Slot: {pos}
  Promote: {player_name}
  Demote/bench: {from_player}
  Expected meta gain: {expected_delta:+.0f} points
  Reason (engine): {reasoning}

{context}

Evaluate:
- Does the player currently on the bench actually fit the active slot?
  (e.g. pitcher role, platoon split, position defense)
- Is the player being demoted genuinely worse, or does their observed
  performance argue for keeping them?
- Is there a subtle chemistry/lineup-balance cost to the swap?
"""


PLATOON_TEMPLATE = """A PLATOON PAIRING has been recommended.

ENGINE'S PICK:
  Slot: {pos}
  Player A: {player_name}
  Player B: {from_player}
  Expected meta gain: {expected_delta:+.0f} points
  Reason (engine): {reasoning}

{context}

Evaluate:
- Are the platoon splits actually large enough to justify a platoon vs
  picking the better overall hitter?
- Is either player being wasted? (e.g. a .900 OPS bat benched vs LHP)
- Would a straight start work better for roster flexibility?
"""


BENCH_TEMPLATE = """BENCHING a current starter has been recommended.

ENGINE'S PICK:
  Slot: {pos}
  Bench: {from_player}
  Expected meta gain (from the replacement): {expected_delta:+.0f} points
  Reason (engine): {reasoning}

{context}

Evaluate:
- Is the observed performance actually bad, or sample-size noise?
- Has the player been benched too early based on a cold streak?
- Is there a role the benched player could still fill (PH, defensive sub)?
"""


SELL_TEMPLATE = """SELLING a card has been recommended.

ENGINE'S PICK:
  Sell: {player_name}
  Expected meta gain from freed PP: {expected_delta:+.0f} points
  Reason (engine): {reasoning}

{context}

Evaluate:
- Is the market price truly above the card's meta-justified value?
- Will the user regret the sale (roster depth, platoon partner, prospect)?
- Would holding and re-evaluating in 1 week be safer?
"""


STRATEGY_SLIDER_TEMPLATE = """A TEAM STRATEGY SLIDER setting has been recommended.

ENGINE'S PICK:
  Slider: {slider}
  Recommended setting: {bucket} (position {position:.0f}/100)
  Reason (engine): {reasoning}

{context}

Evaluate:
- Does the setting match the roster composition evidence?
- Is the magnitude right? (e.g. "Frequently" vs "Often" matters)
- Are there game-state scenarios (behind late, vs ace pitcher) where the
  slider should have different settings?
"""


PLAYER_STRATEGY_TEMPLATE = """A PER-PLAYER STRATEGY OVERRIDE has been recommended.

ENGINE'S PICK:
  Player: {player_name}
  Override: {slider} → {bucket}
  Reason (engine): {reasoning}

{context}

Evaluate:
- Is the player's role (starter / bench / platoon) consistent with the
  override?
- Does the override create a gap elsewhere on the roster?
- For power/speed-based sliders: does observed performance support it
  beyond just the rating?
"""


HOOK_TEMPLATE = """A PITCHING HOOK strategy has been recommended for a specific pitcher.

ENGINE'S PICK:
  Pitcher: {player_name}
  Role: {pos}
  Hook setting: {bucket}
  Reason (engine): {reasoning}

{context}

Evaluate:
- Does the pitcher's stamina + observed performance support this hook?
- How does the bullpen behind them compare? (quick hook requires strong pen)
- Any late-inning matchup concerns against lefty/righty-heavy opponents?
"""


# ──────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────

_TEMPLATES = {
    'buy': BUY_TEMPLATE,
    'promote': PROMOTE_TEMPLATE,
    'platoon': PLATOON_TEMPLATE,
    'bench': BENCH_TEMPLATE,
    'demote': BENCH_TEMPLATE,
    'sell': SELL_TEMPLATE,
    'strategy_slider': STRATEGY_SLIDER_TEMPLATE,
    'player_strategy': PLAYER_STRATEGY_TEMPLATE,
    'hook': HOOK_TEMPLATE,
}


def template_for(rec_type: str) -> str:
    """Return the raw template string for a recommendation type."""
    return _TEMPLATES.get(rec_type, BUY_TEMPLATE)


def build_context(rec: dict, extra: Optional[dict] = None) -> str:
    """Render a structured CONTEXT section from a recommendation dict.

    ``rec`` is a row from ``recommendation_log`` (or an AI Optimize All
    pick dict). ``extra`` can carry enriched fields: meta_explainer,
    observed_stats, team_context, league_context.
    """
    extra = extra or {}
    lines = ['CONTEXT:']

    # Player card data
    card = extra.get('card') or {}
    if card:
        lines.append('  Card ratings:')
        for k in ('contact', 'gap_power', 'power', 'eye', 'avoid_ks',
                  'speed', 'stealing', 'baserunning', 'babip'):
            v = card.get(k)
            if v is not None:
                lines.append(f'    {k}: {v}')
        for k in ('stuff', 'movement', 'control', 'p_hr', 'stamina', 'hold'):
            v = card.get(k)
            if v is not None:
                lines.append(f'    {k}: {v}')
        splits = [(a, b) for a, b in
                  [('contact_vl','contact_vr'), ('power_vl','power_vr'),
                   ('stuff_vl','stuff_vr')]
                  if card.get(a) is not None or card.get(b) is not None]
        if splits:
            lines.append('  Platoon splits:')
            for l, r in splits:
                lines.append(f'    {l}={card.get(l)} / {r}={card.get(r)}')

    # Observed stats
    obs = extra.get('observed') or {}
    if obs:
        lines.append('  Observed stats:')
        for k, v in obs.items():
            if v is not None:
                lines.append(f'    {k}: {v}')

    # Meta explainer (from explain_meta)
    meta_explainer = extra.get('meta_explainer')
    if meta_explainer:
        lines.append('  Meta-score breakdown:')
        for line in str(meta_explainer).splitlines():
            lines.append(f'    {line}')

    # Team context
    team = extra.get('team') or {}
    if team:
        lines.append('  Team context:')
        for k, v in team.items():
            if v is not None:
                lines.append(f'    {k}: {v}')

    # League context
    league = extra.get('league') or {}
    if league:
        lines.append('  League context:')
        for k, v in league.items():
            if v is not None:
                lines.append(f'    {k}: {v}')

    return '\n'.join(lines)


def build_verify_prompt(rec: dict, extra: Optional[dict] = None) -> str:
    """Assemble the full prompt for a verification call.

    ``rec`` must include ``rec_type``. Missing fields render as "unknown".
    """
    rec_type = rec.get('rec_type') or 'buy'
    template = template_for(rec_type)

    # Fill template placeholders safely
    fill = {
        'pos': rec.get('pos') or '—',
        'player_name': rec.get('player_name') or '—',
        'from_player': rec.get('from_player') or '—',
        'expected_delta': rec.get('expected_delta') or 0.0,
        'cost_pp': rec.get('cost_pp') or 0,
        'reasoning': rec.get('reasoning') or '—',
        'slider': rec.get('slider') or rec.get('pos') or '—',
        'bucket': rec.get('bucket') or rec.get('setting') or '—',
        'position': rec.get('position') or 50.0,
        'context': build_context(rec, extra),
    }
    try:
        body = template.format(**fill)
    except KeyError as e:
        body = template + f"\n\n(template placeholder {e} missing)"

    return f"{body}\n\n{OUTPUT_FORMAT}"


# ──────────────────────────────────────────────────────────────────────
# Response parser — extracts the structured fields
# ──────────────────────────────────────────────────────────────────────

def parse_verdict(text: str) -> dict:
    """Parse the OUTPUT_FORMAT response into a structured dict.

    Returns:
        {
          'verdict': 'CONCUR'|'DISSENT'|'NEEDS_MORE_INFO'|'UNKNOWN',
          'confidence': int or None,
          'reasoning': str,
          'risks': str,
          'alternative': str or None,
          'raw': original text,
        }
    """
    out = {
        'verdict': 'UNKNOWN', 'confidence': None,
        'reasoning': '', 'risks': '', 'alternative': None, 'raw': text,
    }
    if not text:
        return out
    current_key = None
    buffer: dict[str, list[str]] = {
        'verdict': [], 'confidence': [], 'reasoning': [],
        'risks': [], 'alternative': [],
    }
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current_key:
                buffer[current_key].append('')
            continue
        lower = stripped.lower()
        if lower.startswith('verdict:'):
            current_key = 'verdict'
            buffer[current_key].append(stripped.split(':', 1)[1].strip())
        elif lower.startswith('confidence:'):
            current_key = 'confidence'
            buffer[current_key].append(stripped.split(':', 1)[1].strip())
        elif lower.startswith('reasoning:'):
            current_key = 'reasoning'
            buffer[current_key].append(stripped.split(':', 1)[1].strip())
        elif lower.startswith('risks:'):
            current_key = 'risks'
            buffer[current_key].append(stripped.split(':', 1)[1].strip())
        elif lower.startswith('alternative:'):
            current_key = 'alternative'
            buffer[current_key].append(stripped.split(':', 1)[1].strip())
        elif current_key:
            buffer[current_key].append(stripped)

    verdict_raw = (' '.join(buffer['verdict'])).strip().upper()
    if 'CONCUR' in verdict_raw:
        out['verdict'] = 'CONCUR'
    elif 'DISSENT' in verdict_raw:
        out['verdict'] = 'DISSENT'
    elif 'NEEDS' in verdict_raw or 'MORE INFO' in verdict_raw:
        out['verdict'] = 'NEEDS_MORE_INFO'

    conf_raw = (' '.join(buffer['confidence'])).strip()
    try:
        import re
        m = re.search(r'(\d+)', conf_raw)
        if m:
            out['confidence'] = int(m.group(1))
    except Exception:
        pass

    out['reasoning'] = ' '.join(buffer['reasoning']).strip()
    out['risks'] = ' '.join(buffer['risks']).strip() or 'none'
    alt = ' '.join(buffer['alternative']).strip()
    if alt and alt.lower() not in ('none', '—', 'n/a', 'na'):
        out['alternative'] = alt

    return out
