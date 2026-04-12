"""Mission Tracker — tracks Live Card collection progress for team missions."""
from app.core.database import get_connection


# All 30 MLB teams
MLB_TEAMS = [
    'Arizona Diamondbacks', 'Athletics', 'Atlanta Braves', 'Baltimore Orioles',
    'Boston Red Sox', 'Chicago Cubs', 'Chicago White Sox', 'Cincinnati Reds',
    'Cleveland Guardians', 'Colorado Rockies', 'Detroit Tigers', 'Houston Astros',
    'Kansas City Royals', 'Los Angeles Angels', 'Los Angeles Dodgers', 'Miami Marlins',
    'Milwaukee Brewers', 'Minnesota Twins', 'New York Mets', 'New York Yankees',
    'Philadelphia Phillies', 'Pittsburgh Pirates', 'San Diego Padres',
    'San Francisco Giants', 'Seattle Mariners', 'St. Louis Cardinals',
    'Tampa Bay Rays', 'Texas Rangers', 'Toronto Blue Jays', 'Washington Nationals',
]


def get_mission_progress(conn=None):
    """Get Live Card collection progress by team for missions.

    Returns a list of dicts per team:
        - team: team name
        - total_cards: how many Live cards exist for this team
        - owned_count: how many you own
        - has_any: True if you own at least 1 (mission eligible)
        - cheapest_available: cheapest unowned Live card for this team (PP)
        - cheapest_card: title of that cheapest card
        - owned_cards: list of owned card titles
        - mission_value_total: sum of mission_value for owned cards
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    # Get all Live cards grouped by team
    rows = conn.execute("""
        SELECT card_id, card_title, team, owned, last_10_price, sell_order_low,
               mission_value, card_value, tier_name,
               COALESCE(meta_score_batting, meta_score_pitching) as meta_score
        FROM cards
        WHERE card_title LIKE 'MLB 2026 Live%'
        ORDER BY team, last_10_price ASC
    """).fetchall()

    by_team = {}
    for r in rows:
        team = r['team']
        if team not in by_team:
            by_team[team] = {
                'team': team,
                'total_cards': 0,
                'owned_count': 0,
                'has_any': False,
                'cheapest_available': None,
                'cheapest_card': None,
                'owned_cards': [],
                'mission_value_total': 0,
            }

        entry = by_team[team]
        entry['total_cards'] += 1

        if r['owned'] and r['owned'] > 0:
            entry['owned_count'] += r['owned']
            entry['has_any'] = True
            entry['owned_cards'].append(r['card_title'])
            entry['mission_value_total'] += (r['mission_value'] or 0)
        else:
            price = r['last_10_price'] or r['sell_order_low'] or 0
            if price > 0 and (entry['cheapest_available'] is None or price < entry['cheapest_available']):
                entry['cheapest_available'] = price
                entry['cheapest_card'] = r['card_title']

    # Ensure all 30 teams are represented even if no cards found
    result = []
    for team in MLB_TEAMS:
        if team in by_team:
            result.append(by_team[team])
        else:
            result.append({
                'team': team, 'total_cards': 0, 'owned_count': 0,
                'has_any': False, 'cheapest_available': None, 'cheapest_card': None,
                'owned_cards': [], 'mission_value_total': 0,
            })

    if close_conn:
        conn.close()
    return result


def get_mission_summary(conn=None):
    """Quick mission progress summary.

    Returns dict:
        - teams_covered: how many teams you have at least 1 Live card for
        - teams_needed: list of team names still missing
        - total_cost_to_complete: estimated PP to buy cheapest card for each missing team
        - total_mission_value: sum of mission_value for all owned Live cards
    """
    progress = get_mission_progress(conn)

    covered = [t for t in progress if t['has_any']]
    needed = [t for t in progress if not t['has_any']]
    cost = sum(t['cheapest_available'] or 0 for t in needed)
    total_mv = sum(t['mission_value_total'] for t in progress)

    return {
        'teams_covered': len(covered),
        'teams_needed': [t['team'] for t in needed],
        'needed_details': needed,
        'total_cost_to_complete': cost,
        'total_mission_value': total_mv,
    }


def get_best_mission_buys(conn=None, max_price=500):
    """Get the cheapest Live card per missing team — shopping list to complete missions.

    Returns list of dicts: team, card_title, price, card_value, meta_score
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    summary = get_mission_summary(conn)
    needed_teams = summary['teams_needed']

    if not needed_teams:
        if close_conn:
            conn.close()
        return []

    shopping_list = []
    for team in needed_teams:
        card = conn.execute("""
            SELECT card_id, card_title, team, last_10_price, sell_order_low,
                   card_value, tier_name, mission_value,
                   COALESCE(meta_score_batting, meta_score_pitching) as meta_score
            FROM cards
            WHERE card_title LIKE 'MLB 2026 Live%'
                AND team = ? AND owned = 0
                AND (last_10_price > 0 OR sell_order_low > 0)
            ORDER BY COALESCE(last_10_price, sell_order_low) ASC
            LIMIT 1
        """, (team,)).fetchone()

        if card:
            price = card['last_10_price'] or card['sell_order_low'] or 0
            if price <= max_price:
                shopping_list.append({
                    'card_id': card['card_id'],
                    'team': team,
                    'card_title': card['card_title'],
                    'price': price,
                    'card_value': card['card_value'],
                    'tier_name': card['tier_name'],
                    'meta_score': card['meta_score'] or 0,
                    'mission_value': card['mission_value'] or 0,
                })

    if close_conn:
        conn.close()
    return shopping_list
