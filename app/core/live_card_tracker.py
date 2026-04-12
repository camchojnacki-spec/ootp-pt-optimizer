"""Live Card MLB Tracker — compares real MLB stats to card ratings for upgrade/downgrade prediction."""
import logging
import time
from datetime import datetime
from app.core.database import get_connection

logger = logging.getLogger(__name__)

# OOTP team abbreviation -> MLB team name mapping (for display, not critical for matching)
OOTP_TEAM_MAP = {
    'AZ': 'Arizona Diamondbacks', 'ATL': 'Atlanta Braves', 'BAL': 'Baltimore Orioles',
    'BOS': 'Boston Red Sox', 'CHC': 'Chicago Cubs', 'CWS': 'Chicago White Sox',
    'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians', 'COL': 'Colorado Rockies',
    'DET': 'Detroit Tigers', 'HOU': 'Houston Astros', 'KC': 'Kansas City Royals',
    'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers', 'MIA': 'Miami Marlins',
    'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins', 'NYM': 'New York Mets',
    'NYY': 'New York Yankees', 'ATH': 'Oakland Athletics', 'PHI': 'Philadelphia Phillies',
    'PIT': 'Pittsburgh Pirates', 'SD': 'San Diego Padres', 'SF': 'San Francisco Giants',
    'SEA': 'Seattle Mariners', 'STL': 'St. Louis Cardinals', 'TB': 'Tampa Bay Rays',
    'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays', 'WSH': 'Washington Nationals',
}


def _safe_float(val, default=0.0):
    """Safely convert a stat value to float."""
    if val is None:
        return default
    try:
        s = str(val).strip().replace('.---', '0')
        return float(s)
    except (ValueError, TypeError):
        return default


def fetch_mlb_stats_for_player(player_name: str, is_pitcher: bool) -> dict | None:
    """Look up a player by name and fetch their current season MLB stats.

    Returns a dict with normalized stat fields, or None if not found.
    """
    import statsapi

    try:
        # Try last name lookup first (more reliable)
        parts = player_name.strip().split()
        if len(parts) >= 2:
            last_name = parts[-1]
            first_name = parts[0]
        else:
            last_name = player_name
            first_name = ""

        results = statsapi.lookup_player(last_name)

        if not results:
            return None

        # Find the best match by full name
        match = None
        for r in results:
            full = r.get('fullName', '').lower()
            if first_name.lower() in full and last_name.lower() in full:
                match = r
                break

        # Fallback: first result with matching last name
        if not match:
            for r in results:
                if last_name.lower() in r.get('fullName', '').lower():
                    match = r
                    break

        if not match:
            return None

        player_id = match['id']
        group = 'pitching' if is_pitcher else 'hitting'

        stat_data = statsapi.player_stat_data(player_id, group=group, type='season')
        seasons = stat_data.get('stats', [])

        if not seasons:
            return None

        # Get the most recent season (should be 2026)
        current = seasons[-1]
        stats = current.get('stats', {})

        if not stats:
            return None

        result = {
            'mlb_id': player_id,
            'mlb_name': match.get('fullName', ''),
            'mlb_team': match.get('currentTeam', {}).get('id', 0),
            'season': current.get('season', ''),
        }

        if is_pitcher:
            result.update({
                'games': stats.get('gamesPlayed', 0),
                'games_started': stats.get('gamesStarted', 0),
                'ip': _safe_float(stats.get('inningsPitched', 0)),
                'era': _safe_float(stats.get('era', 0)),
                'whip': _safe_float(stats.get('whip', 0)),
                'k_per_9': _safe_float(stats.get('strikeoutsPer9Inn', 0)),
                'bb_per_9': _safe_float(stats.get('walksPer9Inn', 0)),
                'hr_per_9': _safe_float(stats.get('homeRunsPer9', 0)),
                'wins': stats.get('wins', 0),
                'losses': stats.get('losses', 0),
                'saves': stats.get('saves', 0),
                'holds': stats.get('holds', 0),
                'strikeouts': stats.get('strikeOuts', 0),
                'walks': stats.get('baseOnBalls', 0),
                'hits_allowed': stats.get('hits', 0),
                'earned_runs': stats.get('earnedRuns', 0),
                'avg_against': _safe_float(stats.get('avg', 0)),
            })
        else:
            result.update({
                'games': stats.get('gamesPlayed', 0),
                'pa': stats.get('plateAppearances', 0),
                'ab': stats.get('atBats', 0),
                'hits': stats.get('hits', 0),
                'hr': stats.get('homeRuns', 0),
                'rbi': stats.get('rbi', 0),
                'runs': stats.get('runs', 0),
                'sb': stats.get('stolenBases', 0),
                'avg': _safe_float(stats.get('avg', 0)),
                'obp': _safe_float(stats.get('obp', 0)),
                'slg': _safe_float(stats.get('slg', 0)),
                'ops': _safe_float(stats.get('ops', 0)),
                'strikeouts': stats.get('strikeOuts', 0),
                'walks': stats.get('baseOnBalls', 0),
                'doubles': stats.get('doubles', 0),
                'triples': stats.get('triples', 0),
                'babip': _safe_float(stats.get('babip', 0)),
            })

        return result

    except Exception as e:
        logger.warning(f"Error fetching MLB stats for {player_name}: {e}")
        return None


def estimate_rating_direction(card: dict, mlb_stats: dict, is_pitcher: bool) -> dict:
    """Compare MLB performance to card ratings and estimate upgrade/downgrade direction.

    Returns a dict with:
        - signal: 'upgrade', 'downgrade', or 'hold'
        - confidence: 'high', 'medium', 'low'
        - reasons: list of strings explaining why
        - score: numeric score (-100 to +100, positive = upgrade likely)
    """
    reasons = []
    score = 0

    if is_pitcher:
        era = mlb_stats.get('era', 0)
        whip = mlb_stats.get('whip', 0)
        k9 = mlb_stats.get('k_per_9', 0)
        bb9 = mlb_stats.get('bb_per_9', 0)
        hr9 = mlb_stats.get('hr_per_9', 0)
        ip = mlb_stats.get('ip', 0)

        card_stuff = card.get('stuff') or 0
        card_movement = card.get('movement') or 0
        card_control = card.get('control') or 0
        card_phr = card.get('p_hr') or 0
        card_value = card.get('card_value') or 50

        # ERA assessment (lower = better)
        if era < 2.50 and card_value < 85:
            score += 30
            reasons.append(f"Elite ERA ({era:.2f}) exceeds {card_value} OVR card")
        elif era < 3.50 and card_value < 75:
            score += 20
            reasons.append(f"Strong ERA ({era:.2f}) above card level")
        elif era > 5.00 and card_value > 65:
            score -= 25
            reasons.append(f"Poor ERA ({era:.2f}) below card level")
        elif era > 6.00:
            score -= 35
            reasons.append(f"Terrible ERA ({era:.2f})")

        # K rate (higher = better stuff)
        if k9 > 10.0 and card_stuff < 80:
            score += 20
            reasons.append(f"High K rate ({k9:.1f}/9) suggests stuff upgrade")
        elif k9 < 5.0 and card_stuff > 60:
            score -= 15
            reasons.append(f"Low K rate ({k9:.1f}/9) suggests stuff downgrade")

        # Walk rate (lower = better control)
        if bb9 < 2.0 and card_control < 80:
            score += 15
            reasons.append(f"Excellent walk rate ({bb9:.1f}/9) suggests control upgrade")
        elif bb9 > 4.5 and card_control > 60:
            score -= 15
            reasons.append(f"High walk rate ({bb9:.1f}/9) suggests control downgrade")

        # HR rate
        if hr9 < 0.5 and card_phr < 75:
            score += 10
            reasons.append(f"Very low HR rate ({hr9:.2f}/9)")
        elif hr9 > 1.5 and card_phr > 50:
            score -= 10
            reasons.append(f"High HR rate ({hr9:.2f}/9)")

        # Small sample penalty
        if ip < 10:
            score = int(score * 0.3)
            reasons.append(f"Small sample ({ip:.1f} IP) — low confidence")
        elif ip < 25:
            score = int(score * 0.6)
            reasons.append(f"Limited sample ({ip:.1f} IP)")

    else:  # Hitter
        ops = mlb_stats.get('ops', 0)
        avg = mlb_stats.get('avg', 0)
        slg = mlb_stats.get('slg', 0)
        babip = mlb_stats.get('babip', 0)
        pa = mlb_stats.get('pa', 0)
        hr = mlb_stats.get('hr', 0)
        sb = mlb_stats.get('sb', 0)

        card_contact = card.get('contact') or 0
        card_power = card.get('power') or 0
        card_eye = card.get('eye') or 0
        card_value = card.get('card_value') or 50

        # OPS assessment
        if ops > 0.900 and card_value < 85:
            score += 30
            reasons.append(f"Elite OPS ({ops:.3f}) exceeds {card_value} OVR card")
        elif ops > 0.800 and card_value < 75:
            score += 20
            reasons.append(f"Strong OPS ({ops:.3f}) above card level")
        elif ops < 0.600 and card_value > 65:
            score -= 25
            reasons.append(f"Poor OPS ({ops:.3f}) below card level")
        elif ops < 0.550:
            score -= 35
            reasons.append(f"Terrible OPS ({ops:.3f})")

        # AVG assessment
        if avg > 0.300 and card_contact < 80:
            score += 15
            reasons.append(f"High AVG ({avg:.3f}) suggests contact upgrade")
        elif avg < 0.200 and card_contact > 60:
            score -= 15
            reasons.append(f"Low AVG ({avg:.3f}) suggests contact downgrade")

        # Power (HR rate)
        games = mlb_stats.get('games', 1) or 1
        hr_per_game = hr / games
        if hr_per_game > 0.25 and card_power < 80:
            score += 15
            reasons.append(f"HR pace ({hr} in {games}G) suggests power upgrade")
        elif hr_per_game < 0.05 and card_power > 70:
            score -= 10
            reasons.append(f"Low HR pace suggests power downgrade")

        # BABIP luck check
        if babip > 0.370:
            score -= 5
            reasons.append(f"High BABIP ({babip:.3f}) — some luck inflation")
        elif babip < 0.250 and avg < 0.230:
            score += 5
            reasons.append(f"Low BABIP ({babip:.3f}) — may be unlucky")

        # Small sample penalty
        if pa < 30:
            score = int(score * 0.3)
            reasons.append(f"Small sample ({pa} PA) — low confidence")
        elif pa < 75:
            score = int(score * 0.6)
            reasons.append(f"Limited sample ({pa} PA)")

    # Determine signal and confidence
    if score >= 20:
        signal = 'upgrade'
    elif score <= -20:
        signal = 'downgrade'
    else:
        signal = 'hold'

    abs_score = abs(score)
    if abs_score >= 40:
        confidence = 'high'
    elif abs_score >= 20:
        confidence = 'medium'
    else:
        confidence = 'low'

    if not reasons:
        reasons.append("Performing roughly in line with card ratings")

    return {
        'signal': signal,
        'confidence': confidence,
        'score': score,
        'reasons': reasons,
    }


def get_live_cards(conn=None):
    """Get all Live cards from the database."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    rows = conn.execute("""
        SELECT card_id, card_title, first_name, last_name, position_name,
               pitcher_role, pitcher_role_name, card_value, tier, tier_name,
               contact, gap_power, power, eye, avoid_ks, babip,
               stuff, movement, control, p_hr, p_babip,
               speed, stealing,
               meta_score_batting, meta_score_pitching,
               owned, buy_order_high, sell_order_low, last_10_price, last_10_variance
        FROM cards
        WHERE card_title LIKE 'MLB 2026 Live%'
        ORDER BY card_value DESC, last_10_price DESC
    """).fetchall()

    if close_conn:
        conn.close()
    return rows


def analyze_live_cards(card_ids: list = None, max_cards: int = 50, owned_only: bool = False,
                       progress_callback=None) -> list:
    """Analyze Live cards against real MLB performance.

    Args:
        card_ids: specific card IDs to analyze (None = auto-select)
        max_cards: maximum number of cards to analyze (API rate limiting)
        owned_only: if True, only analyze cards you own
        progress_callback: optional callable(current, total, card_name) for progress updates

    Returns list of dicts with card info + MLB stats + upgrade/downgrade signal.
    """
    conn = get_connection()

    if card_ids:
        placeholders = ','.join(['?'] * len(card_ids))
        rows = conn.execute(f"""
            SELECT card_id, card_title, first_name, last_name, position_name,
                   pitcher_role, pitcher_role_name, card_value, tier, tier_name,
                   contact, gap_power, power, eye, avoid_ks, babip,
                   stuff, movement, control, p_hr, p_babip,
                   speed, stealing,
                   meta_score_batting, meta_score_pitching,
                   owned, buy_order_high, sell_order_low, last_10_price, last_10_variance
            FROM cards
            WHERE card_id IN ({placeholders}) AND card_title LIKE 'MLB 2026 Live%'
        """, card_ids).fetchall()
    else:
        query = """
            SELECT card_id, card_title, first_name, last_name, position_name,
                   pitcher_role, pitcher_role_name, card_value, tier, tier_name,
                   contact, gap_power, power, eye, avoid_ks, babip,
                   stuff, movement, control, p_hr, p_babip,
                   speed, stealing,
                   meta_score_batting, meta_score_pitching,
                   owned, buy_order_high, sell_order_low, last_10_price, last_10_variance
            FROM cards
            WHERE card_title LIKE 'MLB 2026 Live%'
        """
        if owned_only:
            query += " AND owned > 0"
        query += " ORDER BY last_10_price DESC LIMIT ?"
        rows = conn.execute(query, (max_cards,)).fetchall()

    conn.close()

    results = []
    total = len(rows)

    for i, card in enumerate(rows):
        card_dict = dict(card)
        player_name = f"{card['first_name']} {card['last_name']}".strip()
        is_pitcher = card['pitcher_role'] in (11, 12, 13) if card['pitcher_role'] else False

        if progress_callback:
            progress_callback(i + 1, total, player_name)

        # Fetch MLB stats
        mlb_stats = fetch_mlb_stats_for_player(player_name, is_pitcher)

        if mlb_stats:
            analysis = estimate_rating_direction(card_dict, mlb_stats, is_pitcher)
            results.append({
                'card': card_dict,
                'mlb_stats': mlb_stats,
                'analysis': analysis,
                'player_name': player_name,
                'is_pitcher': is_pitcher,
            })
        else:
            results.append({
                'card': card_dict,
                'mlb_stats': None,
                'analysis': {'signal': 'unknown', 'confidence': 'none', 'score': 0,
                            'reasons': ['Could not find MLB stats']},
                'player_name': player_name,
                'is_pitcher': is_pitcher,
            })

        # Rate limiting — be kind to the MLB API
        time.sleep(0.3)

    # Sort by analysis score (biggest upgrade signals first)
    results.sort(key=lambda x: x['analysis']['score'], reverse=True)

    return results
