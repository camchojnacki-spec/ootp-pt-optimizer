"""Team identity + tactical readout from active roster ratings.

Synthesizes a deterministic team profile (Power Mash, Small-Ball Speed,
Pitching & Defense, etc.) from the active lineup and pitching staff and
emits strengths, weaknesses, tactical directives, and the numeric profile
the park recommender consumes.

This is the rule-based counterpart to the LLM-driven Manager's Eye panel —
it runs instantly with no API call and re-evaluates whenever the roster
changes.

Inputs are lists of dicts (one per player) with the same shape used by
strategy_recommender.recommend_strategy. Required keys:
  Batters:  contact, gap_power, power, eye, avoid_ks, speed, stealing,
            baserunning, bats / bats_hand
  Pitchers: stuff, movement, control, p_hr, stamina,
            pitcher_role_name, throws / throws_hand

Missing values default to 0 — partial rosters still produce a usable
profile but the confidence indicators in the consumer should account for
that.
"""
from __future__ import annotations

import statistics

# Rating thresholds (OOTP 0-100 scale, calibrated from the team's own
# distributions in the OOTP 27 PT environment — not generic baseball
# benchmarks).
ELITE = 75
SOLID = 65
NEUTRAL = 50
WEAK = 40


def _safe(v, default=0.0):
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _avg(values):
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else 0.0


def build_team_profile(
    active_batters: list[dict],
    active_pitchers: list[dict],
    bench_batters: list[dict] | None = None,
    starters: list[dict] | None = None,
    bullpen: list[dict] | None = None,
) -> dict:
    """Compute a structured team profile (numeric scores + classifications).

    Returns a dict with:
      identity_label   — "Power Mash + Elite Pitching" style headline
      identity_summary — 1-2 sentence narrative
      identity_codes   — list of internal codes for downstream logic
      batter           — composite scores (power_composite, contact_composite,
                         speed_composite, discipline)
      starter / bullpen — composite scores
      sp_quality, bp_quality, pitch_quality — single-number staff scores
      bat_hand_pct     — {L/R/S → %} of active lineup
      pit_throw_pct    — {L/R → %} of active staff
      strengths        — top-3 [{label, value, note}]
      weaknesses       — top-3 [{label, value, note}]
      directives       — high-level prescriptions [{icon, rule, detail}]
      roster_size      — counts for the badge row
    """
    bench_batters = bench_batters or []
    starters = starters or []
    bullpen = bullpen or []

    # ── Batter aggregate scores ──
    bat = {
        'contact':     _avg([_safe(b.get('contact')) for b in active_batters]),
        'power':       _avg([_safe(b.get('power')) for b in active_batters]),
        'gap':         _avg([_safe(b.get('gap_power')) for b in active_batters]),
        'eye':         _avg([_safe(b.get('eye')) for b in active_batters]),
        'avoid_ks':    _avg([_safe(b.get('avoid_ks')) for b in active_batters]),
        'speed':       _avg([_safe(b.get('speed')) for b in active_batters]),
        'stealing':    _avg([_safe(b.get('stealing')) for b in active_batters]),
        'baserunning': _avg([_safe(b.get('baserunning')) for b in active_batters]),
    }
    bat['power_composite']   = bat['power'] * 0.65 + bat['gap'] * 0.35
    bat['contact_composite'] = bat['contact'] * 0.6 + bat['avoid_ks'] * 0.25 + bat['eye'] * 0.15
    bat['speed_composite']   = bat['speed'] * 0.5 + bat['stealing'] * 0.25 + bat['baserunning'] * 0.25
    bat['discipline']        = bat['eye'] * 0.55 + bat['avoid_ks'] * 0.45

    # ── Lineup handedness (counts + percent) ──
    bat_hand = {'L': 0, 'R': 0, 'S': 0}
    for b in active_batters:
        h = (b.get('bats_hand') or b.get('bats') or '').upper()
        if h in bat_hand:
            bat_hand[h] += 1
    n_bat = max(1, sum(bat_hand.values()))
    bat_hand_pct = {k: v / n_bat * 100 for k, v in bat_hand.items()}

    # ── Pitcher aggregate scores ──
    sp = {
        'stuff':    _avg([_safe(p.get('stuff')) for p in starters]),
        'control':  _avg([_safe(p.get('control')) for p in starters]),
        'movement': _avg([_safe(p.get('movement')) for p in starters]),
        'p_hr':     _avg([_safe(p.get('p_hr')) for p in starters]),
        'stamina':  _avg([_safe(p.get('stamina')) for p in starters]),
    }
    bp = {
        'stuff':    _avg([_safe(p.get('stuff')) for p in bullpen]),
        'control':  _avg([_safe(p.get('control')) for p in bullpen]),
        'movement': _avg([_safe(p.get('movement')) for p in bullpen]),
        'p_hr':     _avg([_safe(p.get('p_hr')) for p in bullpen]),
    }
    sp_quality = sp['stuff'] * 0.4 + sp['movement'] * 0.25 + sp['control'] * 0.25 + sp['p_hr'] * 0.1
    bp_quality = bp['stuff'] * 0.4 + bp['movement'] * 0.25 + bp['control'] * 0.25 + bp['p_hr'] * 0.1
    pitch_quality = (sp_quality + bp_quality) / 2 if (starters and bullpen) else max(sp_quality, bp_quality)

    # ── Pitcher handedness (counts + percent) ──
    pit_throw = {'L': 0, 'R': 0}
    for p in active_pitchers:
        h = (p.get('throws_hand') or p.get('throws') or '').upper()
        if h in pit_throw:
            pit_throw[h] += 1
    n_pit = max(1, sum(pit_throw.values()))
    pit_throw_pct = {k: v / n_pit * 100 for k, v in pit_throw.items()}

    # ── Identity classification ──
    label, summary, codes = _classify_identity(bat, sp_quality, bp_quality, pitch_quality)

    # ── Strengths / weaknesses ──
    strengths, weaknesses = _strengths_weaknesses(
        bat, sp, bp, sp_quality, bp_quality, pitch_quality, bat_hand_pct
    )

    # ── Tactical directives ──
    directives = _tactical_directives(
        bat, sp_quality, bp_quality, codes, bat_hand_pct, pit_throw_pct
    )

    return {
        'identity_label':   label,
        'identity_summary': summary,
        'identity_codes':   codes,
        'batter':           bat,
        'starter':          sp,
        'bullpen':          bp,
        'sp_quality':       sp_quality,
        'bp_quality':       bp_quality,
        'pitch_quality':    pitch_quality,
        'bat_hand':         bat_hand,
        'bat_hand_pct':     bat_hand_pct,
        'pit_throw':        pit_throw,
        'pit_throw_pct':    pit_throw_pct,
        'strengths':        strengths,
        'weaknesses':       weaknesses,
        'directives':       directives,
        'roster_size': {
            'active_batters': len(active_batters),
            'starters':       len(starters),
            'bullpen':        len(bullpen),
            'bench':          len(bench_batters),
        },
    }


def _classify_identity(bat, sp_quality, bp_quality, pitch_quality):
    """Return (label, summary, codes) — compact identity headline + narrative."""
    p, c, s = bat['power_composite'], bat['contact_composite'], bat['speed_composite']
    codes: list[str] = []

    # Batter archetype — order matters; first match wins
    if p >= ELITE and c >= SOLID:
        codes.append('mash')
        bat_arch = 'Power Mash'
    elif p >= ELITE:
        codes.append('power')
        bat_arch = 'Pure Power'
    elif s >= SOLID and c >= SOLID and p < SOLID:
        codes.append('smallball')
        bat_arch = 'Small-Ball Speed'
    elif s >= SOLID and p < NEUTRAL:
        codes.append('speed')
        bat_arch = 'Speed-First'
    elif c >= ELITE and p < SOLID:
        codes.append('contact')
        bat_arch = 'Contact-First'
    elif p >= SOLID and c >= SOLID:
        codes.append('balanced_offense')
        bat_arch = 'Balanced Offense'
    elif p < NEUTRAL and c < NEUTRAL and s < NEUTRAL:
        codes.append('weakbat')
        bat_arch = 'Below-Avg Bats'
    else:
        codes.append('balanced_offense')
        bat_arch = 'Balanced Offense'

    # Pitching archetype
    if pitch_quality >= ELITE and bp_quality > sp_quality + 5:
        codes.append('penlean')
        pit_arch = 'Bullpen-Leaning Staff'
    elif pitch_quality >= ELITE:
        codes.append('elitepitch')
        pit_arch = 'Elite Pitching'
    elif sp_quality >= SOLID and bp_quality < NEUTRAL:
        codes.append('rotation')
        pit_arch = 'Rotation Carries'
    elif bp_quality >= SOLID and sp_quality < NEUTRAL:
        codes.append('penstrong')
        pit_arch = 'Bullpen Carries'
    elif pitch_quality < NEUTRAL:
        codes.append('weakpitch')
        pit_arch = 'Below-Avg Staff'
    else:
        codes.append('balanced_pitch')
        pit_arch = 'Average Staff'

    label = f'{bat_arch} + {pit_arch}'

    parts = []
    if 'mash' in codes:
        parts.append("Lineup is built to bludgeon — bat your power up and don't waste outs bunting.")
    elif 'power' in codes:
        parts.append("Power-only offense — protect the bats with quality OBP guys ahead of them.")
    elif 'smallball' in codes:
        parts.append("Manufactured-runs offense — go aggressive on the bases, skip swing-for-the-fences.")
    elif 'speed' in codes:
        parts.append("Speed-driven — wheels do the damage; pair with situational hitting.")
    elif 'contact' in codes:
        parts.append("Bat-to-ball offense — minimize whiffs and run when the count says so.")
    elif 'weakbat' in codes:
        parts.append("Lineup needs help — prioritize bat upgrades over pitching luxuries.")
    else:
        parts.append("Balanced bats — flex either small-ball or power depending on opponent.")

    if 'elitepitch' in codes or 'penlean' in codes:
        parts.append("Pitching is the carry — protect leads with quick hooks to the pen.")
    elif 'penstrong' in codes:
        parts.append("Bullpen advantage — pull SPs at the first sign of trouble.")
    elif 'weakpitch' in codes:
        parts.append("Staff is the weak link — fix this before chasing offense upgrades.")
    elif 'rotation' in codes:
        parts.append("SPs eat innings — let them; thin pen can't stretch.")

    summary = ' '.join(parts)
    return label, summary, codes


def _strengths_weaknesses(bat, sp, bp, sp_q, bp_q, pq, bat_hand_pct):
    """Return (strengths[], weaknesses[]) — each is list of {label, value, note}."""
    candidates = [
        ('Lineup Power',      bat['power_composite'],   "HR + XBH threat across the order"),
        ('Gap Power',         bat['gap'],               "Doubles damage in non-HR parks"),
        ('Plate Discipline',  bat['discipline'],        "Walks + low whiffs sustain rallies"),
        ('Contact Skill',     bat['contact_composite'], "Puts bat on ball, advances runners"),
        ('Team Speed',        bat['speed_composite'],   "Stretches bases, pressures defense"),
        ('Starting Rotation', sp_q,                     "Innings-eaters set the tone"),
        ('Bullpen Depth',     bp_q,                     "Late-inning leverage arms"),
    ]
    weakness_extra = []
    if sp.get('p_hr', 0) and sp['p_hr'] < WEAK:
        weakness_extra.append(('SP HR-Suppression', sp['p_hr'],
                               "Long balls hurt — favor pitcher's parks"))
    if bp.get('p_hr', 0) and bp['p_hr'] < WEAK:
        weakness_extra.append(('BP HR-Suppression', bp['p_hr'],
                               "Late-inning HRs sink leads"))

    sorted_high = sorted(candidates, key=lambda c: -c[1])
    sorted_low  = sorted(candidates, key=lambda c: c[1]) + weakness_extra

    strengths = [
        {'label': lbl, 'value': val, 'note': note}
        for lbl, val, note in sorted_high if val >= SOLID
    ][:3]
    weaknesses = [
        {'label': lbl, 'value': val, 'note': note}
        for lbl, val, note in sorted_low
        if val < NEUTRAL or 'HR-Suppression' in lbl
    ][:3]

    # Handedness imbalance is its own callout — appended only if extreme
    if bat_hand_pct.get('L', 0) < 20 and bat_hand_pct.get('R', 0) > 70:
        weaknesses.append({
            'label': 'L/R Bat Imbalance',
            'value': bat_hand_pct.get('L', 0),
            'note': f'Lineup is {bat_hand_pct.get("R", 0):.0f}% RHB '
                    '— vulnerable to RHP-heavy opponents',
        })
    elif bat_hand_pct.get('R', 0) < 20 and bat_hand_pct.get('L', 0) > 70:
        weaknesses.append({
            'label': 'L/R Bat Imbalance',
            'value': bat_hand_pct.get('R', 0),
            'note': f'Lineup is {bat_hand_pct.get("L", 0):.0f}% LHB '
                    '— vulnerable to LHP-heavy opponents',
        })

    return strengths, weaknesses[:3]


def _tactical_directives(bat, sp_q, bp_q, codes, bat_hand_pct, pit_throw_pct):
    """Higher-order prescriptions (lineup ordering, leverage, gaps to fill).

    These are NOT slider settings — those come from strategy_recommender.
    These are the strategic narrative the player should run their team by.
    """
    out: list[dict] = []
    p_comp, c_comp, s_comp = bat['power_composite'], bat['contact_composite'], bat['speed_composite']

    if 'mash' in codes or 'power' in codes:
        out.append({
            'icon': '\U0001f4a3',
            'rule': 'Stack power 2-4-5',
            'detail': "Cluster HR threats — your run gain comes from XBH "
                      "concentration, not splitting them across the order",
        })
    if 'smallball' in codes:
        out.append({
            'icon': '\U0001f3c3',
            'rule': 'Aggressive baserunning',
            'detail': "Steal sliders to Frequently; hit-and-run on 2-1, 3-1 counts",
        })
    if 'speed' in codes and p_comp < NEUTRAL:
        out.append({
            'icon': '⚡',
            'rule': 'Lead off your fastest bat',
            'detail': "SB attempts compound when the speedster bats first — "
                      "even 0.250 OBP earns a stolen base over 162 games",
        })
    if c_comp >= ELITE:
        out.append({
            'icon': '\U0001f3af',
            'rule': 'Hit-and-run favorable',
            'detail': "Low team K% — protect runners with the bat, the math "
                      "favors it more than for whiff-heavy lineups",
        })
    if 'penlean' in codes or bp_q > sp_q + 8:
        out.append({
            'icon': '\U0001f6aa',
            'rule': 'Quick hook on starters',
            'detail': f"BP rates {bp_q:.0f} vs SP {sp_q:.0f} — ride the pen "
                      "for 4+ innings/game",
        })
    if 'rotation' in codes:
        out.append({
            'icon': '⏳',
            'rule': 'Slow hook — protect the pen',
            'detail': "Pen falls off after the first 2-3 arms; let SPs grind "
                      "through the 7th when they have it",
        })
    if 'elitepitch' in codes:
        out.append({
            'icon': '\U0001f6e1️',
            'rule': 'Play for low-scoring games',
            'detail': "Tight games favor your staff — prioritize defensive "
                      "subs in close late innings, don't pinch-hit the glove",
        })
    if 'weakpitch' in codes:
        out.append({
            'icon': '⚠️',
            'rule': 'Score early, score often',
            'detail': "Staff can't protect 1-run leads — push for big innings "
                      "in the first 4 frames, don't play for one run",
        })
    if pit_throw_pct.get('L', 0) < 15 and len(codes) > 0:
        out.append({
            'icon': '✋',
            'rule': 'Add a LOOGY',
            'detail': f"Only {pit_throw_pct.get('L', 0):.0f}% LHP — opponents "
                      "can platoon your pen out of leverage spots",
        })
    if bat_hand_pct.get('S', 0) >= 25:
        out.append({
            'icon': '\U0001f501',
            'rule': 'Switch-hitter advantage',
            'detail': f"{bat_hand_pct.get('S', 0):.0f}% switch — your platoon "
                      "matchup setting can stay neutral without giving up edge",
        })

    return out[:6]
