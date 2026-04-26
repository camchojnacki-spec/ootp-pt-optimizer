"""OOTP ballpark recommender from active roster profile.

Scores each ballpark in the OOTP 27 PT catalog by how well its tendencies
(hitter/pitcher orientation, LHB/RHB orientation, surface) align with the
team's offensive and pitching profile.

Inputs the team profile dict produced by team_identity.build_team_profile.
Returns a ranked list of park fits with reasons + a short list of parks
to avoid.

The catalog is hand-curated from the OOTP Customize Team → Select Ballpark
screen — each park's tendency string is parsed into two numeric axes:

  hp_score  — hitter/pitcher tilt
              -2 = Extreme Pitcher,  -1.5 = Moderate Pitcher
              -1 = Slightly Pitcher,    0 = Neutral
              +1 = Slightly Hitter,  +1.5 = Moderate Hitter, +2 = Extreme Hitter

  lr_score  — handedness tilt
              -2 = Extreme LHB,  -1.5 = Moderate LHB
              -1 = Slightly LHB,    0 = Neutral
              +1 = Slightly RHB, +1.5 = Moderate RHB, +2 = Extreme RHB

Required levels mirror OOTP's PT progression gates. Parks the user can't
yet unlock are still scored (so they show up in the "stretch" bucket) but
default to filtered-out in the top recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Park:
    name: str
    team: str
    year: int
    hp_score: float        # -2 (extreme pitcher) → +2 (extreme hitter)
    lr_score: float        # -2 (extreme LHB) → +2 (extreme RHB)
    park_type: str         # 'Open' / 'Dome' / 'Retractable Roof'
    turf: str              # 'Grass' / 'Artificial Turf'
    tendency: str          # raw OOTP tendency label, for display
    required_level: str    # 'Default' / 'Iron' / 'Low Bronze' / 'Mission Reward' / ...


# Catalog of OOTP 27 PT ballparks visible in the Customize Team selector
# (capacity / type / turf / tendency / required-level transcribed directly
# from the in-game ballpark picker). Tendencies are converted to the
# numeric scale documented above.
PARK_CATALOG: list[Park] = [
    Park('Chase Field',                 'Arizona Diamondbacks', 2026, -1,   +1,   'Retractable Roof', 'Artificial Turf', 'Slightly Pitcher / Slightly RHB',  'Mission Reward'),
    Park('Oriole Park at Camden Yards', 'Baltimore Orioles',    2026,  0,   -1.5, 'Open',             'Grass',           'Neutral / Moderate LHB',           'Mission Reward'),
    Park('Fenway Park',                 'Boston Red Sox',       2026, +1,   +1,   'Open',             'Grass',           'Slightly Hitter / Slightly RHB',   'Mission Reward'),
    Park('Coors Field',                 'Colorado Rockies',     2026, +2,   +1,   'Open',             'Grass',           'Extreme Hitter / Slightly RHB',    'Mission Reward'),
    Park('Angel Stadium of Anaheim',    'Los Angeles Angels',   2026, +1,    0,   'Open',             'Grass',           'Slightly Hitter / Neutral',        'Mission Reward'),
    Park('Yankee Stadium',              'New York Yankees',     2026,  0,   -1,   'Open',             'Grass',           'Neutral / Slightly LHB',           'Mission Reward'),
    Park('Citizens Bank Park',          'Philadelphia Phillies',2026, +1,   -1,   'Open',             'Grass',           'Slightly Hitter / Slightly LHB',   'Mission Reward'),
    Park('Tropicana Field',             'Tampa Bay Rays',       2026, +1.5, +2,   'Dome',             'Artificial Turf', 'Moderate Hitter / Extreme RHB',    'Mission Reward'),
    Park('Rogers Centre',               'Toronto Blue Jays',    2026,  0,   +1.5, 'Retractable Roof', 'Artificial Turf', 'Neutral / Moderate RHB',           'Mission Reward'),
    Park('Sajik Baseball Stadium',      'Lotte Giants',         2026,  0,    0,   'Open',             'Grass',           'Neutral / Neutral',                'Iron'),
    Park('Tropicana Field (2024)',      'Tampa Bay Rays',       2024, -1,   +1.5, 'Dome',             'Artificial Turf', 'Slightly Pitcher / Moderate RHB',  'Low Bronze'),
    Park('Angel Stadium (2015)',        'Los Angeles Angels',   2015, -1,   -1,   'Open',             'Grass',           'Slightly Pitcher / Slightly LHB',  'Low Bronze'),
    Park('Citi Field',                  'New York Mets',        2013,  0,   +1.5, 'Open',             'Grass',           'Neutral / Moderate RHB',           'Low Bronze'),
    Park('McAfee Coliseum',             'Oakland Athletics',    2008, -1.5, -1,   'Open',             'Grass',           'Moderate Pitcher / Slightly LHB',  'Iron'),
    Park('Metrodome',                   'Minnesota Twins',      1990,  0,    0,   'Dome',             'Artificial Turf', 'Neutral / Neutral',                'Iron'),
    Park('Veterans Stadium',            'Philadelphia Phillies',1985,  0,   -1,   'Open',             'Artificial Turf', 'Neutral / Slightly LHB',           'Low Bronze'),
    Park('Busch Stadium',               'St. Louis Cardinals',  1984, -1.5, -1,   'Open',             'Artificial Turf', 'Moderate Pitcher / Slightly LHB',  'Iron'),
    Park('Oakland Coliseum',            'Oakland Athletics',    1977, -1,    0,   'Open',             'Grass',           'Slightly Pitcher / Neutral',       'Low Bronze'),
    Park('Seals Stadium',               'San Francisco Giants', 1959, -1,    0,   'Open',             'Grass',           'Slightly Pitcher / Neutral',       'Iron'),
    Park('Ruppert Stadium',             'Newark Eagles',        1946, +1,    0,   'Open',             'Grass',           'Slightly Hitter / Neutral',        'Low Bronze'),
    Park('Harrison Park',               'Newark Pepper',        1915, -1,    0,   'Open',             'Grass',           'Slightly Pitcher / Neutral',       'Iron'),
    Park('Keystone Park',               'Philadelphia Keystones',1884,-1,    0,   'Open',             'Grass',           'Slightly Pitcher / Neutral',       'Low Bronze'),
    Park('Lake Front Park',             'Chicago White Stockings',1881,+1,   0,   'Open',             'Grass',           'Slightly Hitter / Neutral',        'Iron'),
    Park('Hartford Ball Club Grounds',  'Hartford Dark Blues',  1876, -1,    0,   'Open',             'Grass',           'Slightly Pitcher / Neutral',       'Low Bronze'),
    Park('Waverly Fairgrounds',         'Elizabeth Resolutes',  1873,  0,    0,   'Open',             'Grass',           'Neutral / Neutral',                'Iron'),
    Park('Union Grounds (Brooklyn)',    'New York Mutuals',     1871, -1.5,  0,   'Open',             'Grass',           'Moderate Pitcher / Neutral',       'Low Bronze'),
    Park('South End Grounds I',         'Boston Red Stockings', 1871, +1,    0,   'Open',             'Grass',           'Slightly Hitter / Neutral',        'Low Bronze'),
]


# Levels accessible at "Iron" and below (the typical entry-level tier).
# Mission Reward parks are always accessible if the user has unlocked
# the relevant mission — we surface them by default since most managers
# pursue them.
ACCESSIBLE_LEVELS = frozenset({'Default', 'Iron', 'Low Bronze', 'Mission Reward'})


# Rating thresholds shared with team_identity (kept in sync intentionally).
ELITE = 75
SOLID = 65
NEUTRAL = 50


def _desired_park_tilts(profile: dict) -> tuple[float, float, dict]:
    """Compute the ideal (hp_score, lr_score) target for this roster.

    Returns (desired_hp, desired_lr, debug_dict) so callers can show the
    user *why* a park is being recommended, not just *which*.
    """
    bat = profile['batter']
    sp = profile.get('starter', {})
    bp = profile.get('bullpen', {})
    pq = profile['pitch_quality']
    bat_hand_pct = profile['bat_hand_pct']

    # Hitter/pitcher tilt: lean the park toward your stronger side.
    # A hitter park boosts BOTH offenses equally — but if YOUR offense is
    # the bigger one, the absolute run gap grows in your favor. Same for
    # pitcher parks and pitching staffs.
    bat_strength = (bat['power_composite'] + bat['contact_composite']) / 2
    desired_hp = (bat_strength - pq) / 25.0  # roughly -1.5 → +1.5

    # HR-suppression overlay: weak HR pitchers should hide in pitcher parks
    # regardless of offensive profile (a weak HR-suppression staff
    # surrenders MORE HRs in a hitter park than the team's bats can
    # claw back).
    sp_hr = sp.get('p_hr', 50)
    bp_hr = bp.get('p_hr', 50)
    hr_floor = min(sp_hr, bp_hr)
    if hr_floor < NEUTRAL:
        desired_hp -= (NEUTRAL - hr_floor) / 25.0

    desired_hp = max(-2, min(2, desired_hp))

    # L/R tilt: favor the dominant half. Switch-hitters split 50/50 since
    # they hit from whichever side faces the platoon advantage.
    lhb_weight = bat_hand_pct.get('L', 0) + bat_hand_pct.get('S', 0) * 0.5
    rhb_weight = bat_hand_pct.get('R', 0) + bat_hand_pct.get('S', 0) * 0.5
    desired_lr = (rhb_weight - lhb_weight) / 30.0  # roughly -1.5 → +1.5
    desired_lr = max(-2, min(2, desired_lr))

    debug = {
        'bat_strength': bat_strength,
        'pitch_quality': pq,
        'lhb_weight': lhb_weight,
        'rhb_weight': rhb_weight,
        'sp_hr': sp_hr,
        'bp_hr': bp_hr,
        'hr_floor': hr_floor,
    }
    return desired_hp, desired_lr, debug


def _score_park(park: Park, profile: dict,
                desired_hp: float, desired_lr: float, debug: dict
                ) -> tuple[float, list[str]]:
    """Score a park (0-100) for this roster + return list of reason strings."""
    bat = profile['batter']
    sp_q = profile['sp_quality']
    pq = profile['pitch_quality']
    reasons: list[str] = []

    # 1) Hitter/pitcher fit (55% weight)
    hp_diff = abs(park.hp_score - desired_hp)
    hp_match = max(0.0, 100.0 - hp_diff * 35.0)

    if hp_diff <= 0.5:
        if desired_hp >= 0.7:
            reasons.append(f"hitter tilt fits a {debug['bat_strength']:.0f}-rated lineup")
        elif desired_hp <= -0.7:
            reasons.append(f"pitcher tilt protects a {pq:.0f}-rated staff")
        else:
            reasons.append("neutral run environment matches a balanced roster")
    elif park.hp_score - desired_hp >= 1:
        reasons.append("more hitter-friendly than ideal — opponent bats benefit equally")
    elif desired_hp - park.hp_score >= 1:
        reasons.append("more pitcher-friendly than ideal — your offense loses runs too")

    # HR-suppression callout (only when it actually drove the desired_hp shift)
    if debug['hr_floor'] < NEUTRAL and park.hp_score <= 0:
        reasons.append(f"HR suppression matters — staff p_hr {debug['hr_floor']:.0f}")

    # 2) L/R fit (30% weight)
    lr_diff = abs(park.lr_score - desired_lr)
    lr_match = max(0.0, 100.0 - lr_diff * 28.0)

    if lr_diff <= 0.7 and abs(park.lr_score) >= 0.5:
        if park.lr_score > 0:
            reasons.append(f"RHB tilt rewards {debug['rhb_weight']:.0f}% right-side bat")
        else:
            reasons.append(f"LHB tilt rewards {debug['lhb_weight']:.0f}% left-side bat")
    elif lr_diff >= 1.5:
        # Only flag a "wrong-way" lean if the park actually leans the wrong way.
        # A neutral park doesn't "favor" anyone — it just fails to amplify
        # the team's dominant side.
        if park.lr_score >= 0.5 and desired_lr <= -0.5:
            reasons.append("park favors RHB but lineup leans left — split goes to opponent")
        elif park.lr_score <= -0.5 and desired_lr >= 0.5:
            reasons.append("park favors LHB but lineup leans right — split goes to opponent")
        elif abs(park.lr_score) < 0.5 and abs(desired_lr) >= 1.5:
            side = 'RHB' if desired_lr > 0 else 'LHB'
            reasons.append(f"neutral L/R — leaves your {side}-heavy lineup's edge on the table")

    # 3) Surface bonus — turf rewards speed (15 pts max, ~5% of total)
    surface_bonus = 0.0
    if 'Artificial' in park.turf and bat['speed_composite'] >= SOLID:
        surface_bonus = 8.0
        reasons.append("turf rewards your team speed")
    elif 'Artificial' in park.turf and bat['speed_composite'] < NEUTRAL:
        surface_bonus = -4.0
        reasons.append("turf doesn't help — slow lineup")

    # 4) Roof bonus — domes/retractable add a tiny tiebreaker for control
    roof_bonus = 2.0 if park.park_type in ('Dome', 'Retractable Roof') else 0.0

    score = hp_match * 0.55 + lr_match * 0.30 + surface_bonus + roof_bonus
    return score, reasons[:4]  # cap reasons so the UI stays readable


def recommend_parks(profile: dict, top_n: int = 5,
                    accessible_only: bool = True) -> dict:
    """Score and rank the OOTP park catalog for this roster profile.

    Args:
        profile: dict from team_identity.build_team_profile
        top_n: number of top parks to return (default 5)
        accessible_only: if True, restrict the top list to parks at
            commonly-accessible PT levels (Default/Iron/Low Bronze/
            Mission Reward). Higher-tier parks still appear in the
            'stretch' bucket.

    Returns:
        {
          'top':        [{park, score, reasons}, ...]   — top_n recommendations
          'stretch':    [...]                          — best parks user can't yet access
          'avoid':      [...]                          — bottom 3 for this roster
          'desired_hp': float, 'desired_lr': float     — the target tilts
          'debug':      {...}                          — composition stats
        }
    """
    desired_hp, desired_lr, debug = _desired_park_tilts(profile)

    scored = []
    for p in PARK_CATALOG:
        s, reasons = _score_park(p, profile, desired_hp, desired_lr, debug)
        scored.append({
            'park': p,
            'score': s,
            'reasons': reasons,
            'accessible': p.required_level in ACCESSIBLE_LEVELS,
        })
    scored.sort(key=lambda x: -x['score'])

    if accessible_only:
        top = [s for s in scored if s['accessible']][:top_n]
        stretch = [s for s in scored if not s['accessible']][:3]
    else:
        top = scored[:top_n]
        stretch = []

    avoid = sorted(scored, key=lambda x: x['score'])[:3]

    return {
        'top': top,
        'stretch': stretch,
        'avoid': avoid,
        'desired_hp': desired_hp,
        'desired_lr': desired_lr,
        'debug': debug,
        'all_scored': scored,
    }


def tendency_label(hp: float, lr: float) -> str:
    """Render a (hp, lr) pair as an OOTP-style tendency string."""
    def _hp_label(v):
        if v >= 1.75: return 'Extreme Hitter'
        if v >= 1.25: return 'Moderate Hitter'
        if v >= 0.5:  return 'Slightly Hitter'
        if v <= -1.75:return 'Extreme Pitcher'
        if v <= -1.25:return 'Moderate Pitcher'
        if v <= -0.5: return 'Slightly Pitcher'
        return 'Neutral'

    def _lr_label(v):
        if v >= 1.75: return 'Extreme RHB'
        if v >= 1.25: return 'Moderate RHB'
        if v >= 0.5:  return 'Slightly RHB'
        if v <= -1.75:return 'Extreme LHB'
        if v <= -1.25:return 'Moderate LHB'
        if v <= -0.5: return 'Slightly LHB'
        return 'Neutral'

    return f'{_hp_label(hp)} / {_lr_label(lr)}'
