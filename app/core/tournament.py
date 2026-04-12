"""Tournament roster builder and optimizer for OOTP Perfect Team.

Tournament rules from OOTP wiki:
- 26-man roster max, salary cap = sum of OVR ratings
- Empty slots count as 40 points (or min tier value if restricted)
- Card restrictions: min/max OVR, card type/year/era, combinator limits
- Chemistry bonuses: card type matching, franchise, year/decade, historical teammates
- Minimum lineup: 4 SP, 4 RP, 7 position players per lineup slot
"""
import sqlite3
from collections import defaultdict
from app.core.database import get_connection


# Position requirements for a valid tournament roster
BATTING_POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
PITCHING_POSITIONS_STARTER = ['SP']
PITCHING_POSITIONS_RELIEF = ['RP', 'CL']
MIN_SP = 4
MIN_RP = 4  # includes CL
MIN_BATTERS_PER_POS = 1  # at least 1 per batting position
ROSTER_SIZE = 26
EMPTY_SLOT_VALUE = 40


def get_eligible_cards(conn, constraints):
    """Get all owned cards that meet tournament constraints.

    Args:
        conn: database connection
        constraints: dict with keys:
            - salary_cap: int (0 = no cap)
            - min_ovr: int (0 = no min)
            - max_ovr: int (0 = no max)
            - card_types: list of allowed card_sub_type strings (empty = all)
            - card_series: list of allowed card_series strings (empty = all)
            - max_combinators: int (-1 = unlimited)
            - year: int (0 = any year)
            - chemistry_enabled: bool

    Returns:
        list of card dicts with: card_id, card_title, position, pitcher_role,
        ovr, meta_score, tier, tier_name, card_type, card_series, year, team, franchise
    """
    query = """
        SELECT c.card_id, c.card_title, c.position_name, c.pitcher_role_name,
               c.card_value as ovr,
               COALESCE(c.meta_score_batting, c.meta_score_pitching, 0) as meta_score,
               c.meta_score_batting, c.meta_score_pitching,
               c.tier, c.tier_name, c.card_sub_type, c.card_series,
               c.year, c.team, c.franchise, c.bats, c.throws, c.age,
               c.contact, c.gap_power, c.power, c.eye, c.avoid_ks, c.babip,
               c.stuff, c.movement, c.control, c.p_hr,
               c.pos_rating_c, c.pos_rating_1b, c.pos_rating_2b,
               c.pos_rating_3b, c.pos_rating_ss, c.pos_rating_lf,
               c.pos_rating_cf, c.pos_rating_rf
        FROM cards c
        WHERE c.owned > 0
    """
    params = []

    min_ovr = constraints.get('min_ovr', 0)
    max_ovr = constraints.get('max_ovr', 0)

    if min_ovr > 0:
        query += " AND c.card_value >= ?"
        params.append(min_ovr)
    if max_ovr > 0:
        query += " AND c.card_value <= ?"
        params.append(max_ovr)

    year = constraints.get('year', 0)
    if year > 0:
        query += " AND c.year = ?"
        params.append(year)

    query += " ORDER BY COALESCE(c.meta_score_batting, c.meta_score_pitching, 0) DESC"

    rows = conn.execute(query, params).fetchall()

    cards = []
    for r in rows:
        card = dict(r)
        # Determine primary role
        if r['pitcher_role_name'] in ('SP', 'RP', 'CL'):
            card['role'] = r['pitcher_role_name']
            card['is_pitcher'] = True
        else:
            card['role'] = r['position_name']
            card['is_pitcher'] = False
        cards.append(card)

    # Filter by card types if specified
    card_types = constraints.get('card_types', [])
    if card_types:
        cards = [c for c in cards if c['card_sub_type'] in card_types]

    card_series_filter = constraints.get('card_series', [])
    if card_series_filter:
        cards = [c for c in cards if c['card_series'] in card_series_filter]

    return cards


def calculate_chemistry(roster_cards):
    """Calculate chemistry score for a set of cards.

    Chemistry factors (from OOTP wiki):
    1. Card type matching (same card_sub_type)
    2. Franchise matching (same franchise)
    3. Year matching (same year = big bonus, same decade = small bonus)
    4. Card series matching

    Returns:
        dict with:
            - total_score: float (0-100 scale)
            - breakdown: dict of factor -> score
            - pairs: list of notable chemistry pairs
    """
    if len(roster_cards) < 2:
        return {'total_score': 0, 'breakdown': {}, 'pairs': []}

    n = len(roster_cards)
    max_pairs = n * (n - 1) / 2

    # Factor 1: Card type matching
    type_matches = 0
    type_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = roster_cards[i], roster_cards[j]
            if a.get('card_sub_type') and a['card_sub_type'] == b.get('card_sub_type'):
                type_matches += 1
                if len(type_pairs) < 5:
                    type_pairs.append(f"{a['card_title'][:25]} + {b['card_title'][:25]} ({a['card_sub_type']})")

    type_score = (type_matches / max_pairs * 100) if max_pairs > 0 else 0

    # Factor 2: Franchise matching
    franchise_matches = 0
    franchise_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = roster_cards[i], roster_cards[j]
            if a.get('franchise') and a['franchise'] == b.get('franchise'):
                franchise_matches += 1
                if len(franchise_pairs) < 5:
                    franchise_pairs.append(f"{a['card_title'][:25]} + {b['card_title'][:25]} ({a['franchise']})")

    franchise_score = (franchise_matches / max_pairs * 100) if max_pairs > 0 else 0

    # Factor 3: Year/decade matching
    year_matches = 0
    decade_matches = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = roster_cards[i], roster_cards[j]
            ya, yb = a.get('year', 0), b.get('year', 0)
            if ya and yb:
                if ya == yb:
                    year_matches += 1
                elif ya // 10 == yb // 10:
                    decade_matches += 1

    year_score = (year_matches / max_pairs * 100) if max_pairs > 0 else 0
    decade_score = (decade_matches / max_pairs * 50) if max_pairs > 0 else 0  # half weight

    # Factor 4: Card series matching
    series_matches = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = roster_cards[i], roster_cards[j]
            if a.get('card_series') and a['card_series'] == b.get('card_series'):
                series_matches += 1

    series_score = (series_matches / max_pairs * 100) if max_pairs > 0 else 0

    # Weighted total
    total = (type_score * 0.30 + franchise_score * 0.25 +
             year_score * 0.20 + decade_score * 0.10 + series_score * 0.15)

    # Collect notable pairs
    all_pairs = type_pairs + franchise_pairs

    return {
        'total_score': round(total, 1),
        'breakdown': {
            'Card Type': round(type_score, 1),
            'Franchise': round(franchise_score, 1),
            'Same Year': round(year_score, 1),
            'Same Decade': round(decade_score, 1),
            'Card Series': round(series_score, 1),
        },
        'pairs': all_pairs[:10],
    }


def calculate_salary(roster_cards):
    """Calculate total salary for a tournament roster.

    Salary = sum of OVR for filled slots + 40 per empty slot.
    """
    filled = len(roster_cards)
    total_ovr = sum(c.get('ovr', 0) or 0 for c in roster_cards)
    empty_slots = max(0, ROSTER_SIZE - filled)
    empty_cost = empty_slots * EMPTY_SLOT_VALUE

    return {
        'total_salary': total_ovr + empty_cost,
        'player_salary': total_ovr,
        'empty_penalty': empty_cost,
        'filled_slots': filled,
        'empty_slots': empty_slots,
    }


def validate_roster(roster_cards):
    """Validate a tournament roster meets minimum requirements.

    Returns:
        dict with:
            - valid: bool
            - errors: list of error strings
            - warnings: list of warning strings
            - position_counts: dict of position -> count
    """
    errors = []
    warnings = []

    if len(roster_cards) > ROSTER_SIZE:
        errors.append(f"Roster exceeds {ROSTER_SIZE} players ({len(roster_cards)})")

    if len(roster_cards) < 20:
        errors.append(f"Roster needs at least 20 players (have {len(roster_cards)})")

    # Count by role
    sp_count = sum(1 for c in roster_cards if c.get('role') == 'SP')
    rp_count = sum(1 for c in roster_cards if c.get('role') in ('RP', 'CL'))
    cl_count = sum(1 for c in roster_cards if c.get('role') == 'CL')

    position_counts = defaultdict(int)
    for c in roster_cards:
        position_counts[c.get('role', '?')] += 1

    if sp_count < MIN_SP:
        errors.append(f"Need at least {MIN_SP} starting pitchers (have {sp_count})")
    if rp_count < MIN_RP:
        errors.append(f"Need at least {MIN_RP} relievers (have {rp_count})")

    # Check batting positions filled
    for pos in BATTING_POSITIONS:
        count = position_counts.get(pos, 0)
        if count < MIN_BATTERS_PER_POS:
            errors.append(f"No player at {pos}")

    # Warnings
    if sp_count < 5:
        warnings.append(f"Only {sp_count} SP — consider 5 for rotation depth")
    if rp_count < 5:
        warnings.append(f"Only {rp_count} RP/CL — bullpen may be thin")
    if cl_count == 0:
        warnings.append("No dedicated closer (CL)")

    # Check for duplicate players (same player name, different card version)
    names = [c.get('card_title', '').split(' - ')[0] for c in roster_cards]
    seen = set()
    dupes = set()
    for name in names:
        # Extract base player name (remove card type prefixes)
        base = name.strip()
        if base in seen:
            dupes.add(base)
        seen.add(base)
    if dupes:
        for d in dupes:
            warnings.append(f"Possible duplicate player: {d}")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'position_counts': dict(position_counts),
    }


def auto_build_roster(conn, constraints):
    """Automatically build the best tournament roster given constraints.

    Strategy:
    1. Get all eligible cards
    2. Fill required positions (best meta per position)
    3. If salary cap, use greedy knapsack to maximize meta within cap
    4. Fill remaining slots with best available bench depth
    5. Optimize for chemistry if enabled

    Returns:
        dict with:
            - roster: list of selected card dicts
            - bench: list of bench/depth cards
            - salary: salary calculation dict
            - chemistry: chemistry calculation dict
            - validation: validation dict
            - excluded: list of eligible cards not selected
            - recommendations: list of cards to acquire
    """
    eligible = get_eligible_cards(conn, constraints)

    if not eligible:
        return {
            'roster': [], 'bench': [], 'salary': calculate_salary([]),
            'chemistry': calculate_chemistry([]), 'validation': validate_roster([]),
            'excluded': [], 'recommendations': [],
        }

    salary_cap = constraints.get('salary_cap', 0)
    chemistry_enabled = constraints.get('chemistry_enabled', True)

    # Categorize eligible cards by role
    by_role = defaultdict(list)
    for c in eligible:
        by_role[c['role']].append(c)

    # Sort each role by meta_score descending
    for role in by_role:
        by_role[role].sort(key=lambda x: x.get('meta_score', 0), reverse=True)

    selected = []
    selected_ids = set()

    def pick_best(role, count):
        picked = 0
        for c in by_role.get(role, []):
            if c['card_id'] not in selected_ids:
                selected.append(c)
                selected_ids.add(c['card_id'])
                picked += 1
                if picked >= count:
                    break

    # Phase 1: Fill minimum requirements
    # 1 per batting position
    for pos in BATTING_POSITIONS:
        pick_best(pos, 1)

    # Min SP
    pick_best('SP', MIN_SP)

    # Min RP/CL
    pick_best('CL', 1)  # Try to get a closer first
    rp_have = sum(1 for c in selected if c.get('role') in ('RP', 'CL'))
    if rp_have < MIN_RP:
        pick_best('RP', MIN_RP - rp_have)

    # Phase 2: Fill remaining slots with best available
    remaining_slots = ROSTER_SIZE - len(selected)
    all_remaining = [c for c in eligible if c['card_id'] not in selected_ids]

    if salary_cap > 0:
        # Greedy approach: fill within salary cap
        current_salary = calculate_salary(selected)['total_salary']
        # Removing empty slot penalties as we fill
        for c in sorted(all_remaining, key=lambda x: x.get('meta_score', 0), reverse=True):
            if len(selected) >= ROSTER_SIZE:
                break
            ovr = c.get('ovr', 0) or 0
            # Adding this player: +ovr, -40 (one fewer empty slot)
            net_salary_change = ovr - EMPTY_SLOT_VALUE
            if current_salary + net_salary_change <= salary_cap:
                selected.append(c)
                selected_ids.add(c['card_id'])
                current_salary += net_salary_change
    else:
        # No salary cap — just pick best meta
        all_remaining.sort(key=lambda x: x.get('meta_score', 0), reverse=True)
        for c in all_remaining:
            if len(selected) >= ROSTER_SIZE:
                break
            selected.append(c)
            selected_ids.add(c['card_id'])

    # Phase 3: Categorize selected into starters vs bench
    starters = []
    bench = []

    # Starters: best at each position
    starter_by_pos = {}
    for c in sorted(selected, key=lambda x: x.get('meta_score', 0), reverse=True):
        role = c.get('role', '?')
        if role not in starter_by_pos:
            starter_by_pos[role] = c
            starters.append(c)
        elif role == 'SP' and sum(1 for s in starters if s.get('role') == 'SP') < 5:
            starters.append(c)
        elif role in ('RP', 'CL') and sum(1 for s in starters if s.get('role') in ('RP', 'CL')) < 6:
            starters.append(c)
        else:
            bench.append(c)

    # Calculate results
    salary = calculate_salary(selected)
    chemistry = calculate_chemistry(selected) if chemistry_enabled else {'total_score': 0, 'breakdown': {}, 'pairs': []}
    validation = validate_roster(selected)

    excluded = [c for c in eligible if c['card_id'] not in selected_ids]

    # Generate acquisition recommendations
    recommendations = _get_tournament_recommendations(conn, constraints, selected, validation)

    return {
        'roster': selected,
        'starters': starters,
        'bench': bench,
        'salary': salary,
        'chemistry': chemistry,
        'validation': validation,
        'excluded': excluded,
        'recommendations': recommendations,
    }


def _get_tournament_recommendations(conn, constraints, current_roster, validation):
    """Recommend cards to acquire for tournament play.

    Looks at market for cards that:
    1. Fill position gaps in the roster
    2. Would be upgrades over current selections
    3. Meet tournament constraints
    """
    recs = []
    min_ovr = constraints.get('min_ovr', 0)
    max_ovr = constraints.get('max_ovr', 0)
    salary_cap = constraints.get('salary_cap', 0)

    # Current roster meta by position
    roster_meta = {}
    for c in current_roster:
        role = c.get('role', '?')
        meta = c.get('meta_score', 0)
        if role not in roster_meta or meta > roster_meta[role]:
            roster_meta[role] = meta

    # Check positions with gaps
    all_positions = BATTING_POSITIONS + ['SP', 'RP', 'CL']
    for pos in all_positions:
        current_meta = roster_meta.get(pos, 0)

        # Find market cards at this position
        if pos in ('SP', 'RP', 'CL'):
            query = """
                SELECT card_id, card_title, pitcher_role_name as role, card_value as ovr,
                       meta_score_pitching as meta_score, last_10_price, tier_name
                FROM cards
                WHERE pitcher_role_name = ? AND owned = 0 AND last_10_price > 0
                    AND meta_score_pitching > ?
            """
        else:
            query = """
                SELECT card_id, card_title, position_name as role, card_value as ovr,
                       meta_score_batting as meta_score, last_10_price, tier_name
                FROM cards
                WHERE position_name = ? AND owned = 0 AND last_10_price > 0
                    AND meta_score_batting > ?
            """

        params = [pos, current_meta + 20]

        if min_ovr > 0:
            query += " AND card_value >= ?"
            params.append(min_ovr)
        if max_ovr > 0:
            query += " AND card_value <= ?"
            params.append(max_ovr)

        query += " ORDER BY COALESCE(meta_score_batting, meta_score_pitching) DESC LIMIT 3"

        market_cards = conn.execute(query, params).fetchall()

        for mc in market_cards:
            improvement = (mc['meta_score'] or 0) - current_meta
            recs.append({
                'card_title': mc['card_title'],
                'position': pos,
                'ovr': mc['ovr'],
                'meta_score': mc['meta_score'],
                'price': mc['last_10_price'],
                'tier': mc['tier_name'],
                'improvement': improvement,
                'reason': f"+{improvement:.0f} meta upgrade at {pos}" if current_meta > 0
                         else f"Fills {pos} gap",
            })

    # Sort by improvement
    recs.sort(key=lambda x: x.get('improvement', 0), reverse=True)
    return recs[:15]


def get_tournament_presets():
    """Return common tournament configuration presets."""
    return {
        'Open': {
            'name': 'Open Tournament',
            'description': 'No restrictions — bring your best roster',
            'salary_cap': 0,
            'min_ovr': 0,
            'max_ovr': 0,
            'card_types': [],
            'card_series': [],
            'max_combinators': -1,
            'chemistry_enabled': True,
        },
        'Salary Cap': {
            'name': 'Salary Cap Tournament',
            'description': 'Total roster OVR capped — balance stars with role players',
            'salary_cap': 1600,
            'min_ovr': 0,
            'max_ovr': 0,
            'card_types': [],
            'card_series': [],
            'max_combinators': -1,
            'chemistry_enabled': True,
        },
        'Budget': {
            'name': 'Budget Tournament',
            'description': 'Max OVR per card restricted — find hidden gems',
            'salary_cap': 0,
            'min_ovr': 0,
            'max_ovr': 70,
            'card_types': [],
            'card_series': [],
            'max_combinators': -1,
            'chemistry_enabled': True,
        },
        'No Chemistry': {
            'name': 'No Chemistry Tournament',
            'description': 'Chemistry disabled — pure ratings matter',
            'salary_cap': 0,
            'min_ovr': 0,
            'max_ovr': 0,
            'card_types': [],
            'card_series': [],
            'max_combinators': -1,
            'chemistry_enabled': False,
        },
        'Tight Cap': {
            'name': 'Tight Salary Cap',
            'description': 'Very restrictive cap — every OVR point counts',
            'salary_cap': 1300,
            'min_ovr': 0,
            'max_ovr': 0,
            'card_types': [],
            'card_series': [],
            'max_combinators': -1,
            'chemistry_enabled': True,
        },
    }
