"""Game log parser DEMO — extracts at-bat-level superstats from OOTP's
``news/html/game_logs/log_N.html`` files.

Each at-bat in a game log has:
  - pitcher hand + name (LHP/RHP Red Ruffing)
  - batter hand + name (LHB/RHB/SHB Juan Pierre)
  - pitch-by-pitch progression (0-0: Foul, 0-1: Ball, ...)
  - final outcome, which for batted balls includes:
      hit type (SINGLE / DOUBLE / TRIPLE / HOME RUN / out)
      batted ball type (Line Drive / Groundball / Fly Ball / Popup)
      location code (9LD, 2F, 6MS, 89D, ...)
      exit velocity (EV XX MPH)

This demo parses one or more game logs and reports:
  - per-batter: at-bats, avg EV, hit distribution, strikeout rate
  - per-pitcher: batters faced, avg EV allowed, K rate, BB rate
  - vs-handedness splits per player
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup


EV_RE = re.compile(r'EV\s+(\d+)\s+MPH')
BATTED_BALL_RE = re.compile(
    r'\((Line Drive|Groundball|Fly Ball|Popup|Bunt)[^)]*\)',
    re.IGNORECASE,
)
LOC_RE = re.compile(r'\(\s*(?:Line Drive|Groundball|Fly Ball|Popup|Bunt)[^,]*,\s*([0-9A-Z]+)')
OUTCOME_TOKENS = [
    'HOME RUN', 'GRAND SLAM', 'TRIPLE', 'DOUBLE', 'SINGLE',
    'Strikes out', 'Base on Balls', 'HBP', 'Sacrifice',
]


def _parse_at_bat(outcome_html: str) -> dict:
    """Extract features from one at-bat outcome cell HTML."""
    text = BeautifulSoup(outcome_html, 'lxml').get_text(' ', strip=True)

    ev_match = EV_RE.search(text)
    ev = int(ev_match.group(1)) if ev_match else None

    bb_match = BATTED_BALL_RE.search(text)
    batted_ball = bb_match.group(1) if bb_match else None

    loc_match = LOC_RE.search(text)
    location = loc_match.group(1) if loc_match else None

    # Find the final outcome token — scan in order of priority (HR > ..)
    outcome = None
    upper = text.upper()
    for tok in ['GRAND SLAM', 'HOME RUN', 'TRIPLE', 'DOUBLE', 'SINGLE']:
        if tok in upper:
            outcome = tok
            break
    if outcome is None:
        if 'strikes out' in text.lower() or 'strikeout' in text.lower():
            outcome = 'K'
        elif 'base on balls' in text.lower() or 'walked' in text.lower():
            outcome = 'BB'
        elif 'hit by pitch' in text.lower() or 'hbp' in text.lower():
            outcome = 'HBP'
        elif any(k in text.lower() for k in ['grounds out', 'ground out', 'fly out', 'fly ball',
                                              'pops out', 'popout', 'lineout', 'line out',
                                              'fielder\'s choice']):
            outcome = 'OUT'

    # Pitch count: count pitch lines (each line looks like "N-N: something")
    pitches = len(re.findall(r'\d+\s*-\s*\d+\s*:', text))

    return {
        'outcome': outcome,
        'batted_ball': batted_ball,
        'location': location,
        'exit_velocity': ev,
        'pitches_seen': pitches,
    }


def parse_game_log(filepath: str) -> list[dict]:
    """Return a list of at-bat dicts for one game log."""
    html = Path(filepath).read_text(encoding='utf-8', errors='replace')
    soup = BeautifulSoup(html, 'lxml')

    current_pitcher = None
    current_pitcher_hand = None
    at_bats = []

    for tr in soup.find_all('tr'):
        cells = tr.find_all('td')
        if len(cells) < 2:
            continue
        left = cells[0].get_text(' ', strip=True)
        right_html = str(cells[1])

        # Pitching change line: "Pitching: RHP <name>"
        m_pit = re.match(r'Pitching:\s*(LHP|RHP)\s+(.+)', left)
        if m_pit:
            current_pitcher_hand = m_pit.group(1)
            current_pitcher = m_pit.group(2).strip()
            continue

        # Batter line: "Batting: LHB/RHB/SHB <name>"
        m_bat = re.match(r'Batting:\s*(LHB|RHB|SHB)\s+(.+)', left)
        if not m_bat:
            continue
        batter_hand = m_bat.group(1)
        batter = m_bat.group(2).strip()

        parsed = _parse_at_bat(right_html)
        parsed.update({
            'pitcher': current_pitcher,
            'pitcher_hand': current_pitcher_hand,
            'batter': batter,
            'batter_hand': batter_hand,
        })
        at_bats.append(parsed)

    return at_bats


def summarize(at_bats: list[dict]):
    """Aggregate: per-batter and per-pitcher rollups."""
    by_batter = defaultdict(lambda: {'pa': 0, 'k': 0, 'bb': 0, 'hits': 0, 'hr': 0,
                                      'ev_sum': 0, 'ev_n': 0, 'ld': 0, 'gb': 0, 'fb': 0})
    by_pitcher = defaultdict(lambda: {'bf': 0, 'k': 0, 'bb': 0, 'hits_allowed': 0, 'hr_allowed': 0,
                                       'ev_allowed_sum': 0, 'ev_allowed_n': 0})

    for ab in at_bats:
        b = by_batter[ab['batter']]
        b['pa'] += 1
        if ab['outcome'] == 'K': b['k'] += 1
        elif ab['outcome'] == 'BB': b['bb'] += 1
        elif ab['outcome'] in ('SINGLE', 'DOUBLE', 'TRIPLE', 'HOME RUN', 'GRAND SLAM'):
            b['hits'] += 1
            if ab['outcome'] in ('HOME RUN', 'GRAND SLAM'):
                b['hr'] += 1
        if ab['exit_velocity']:
            b['ev_sum'] += ab['exit_velocity']
            b['ev_n'] += 1
        bt = (ab['batted_ball'] or '').lower()
        if 'line' in bt: b['ld'] += 1
        elif 'ground' in bt: b['gb'] += 1
        elif 'fly' in bt or 'popup' in bt: b['fb'] += 1

        if ab['pitcher']:
            p = by_pitcher[ab['pitcher']]
            p['bf'] += 1
            if ab['outcome'] == 'K': p['k'] += 1
            elif ab['outcome'] == 'BB': p['bb'] += 1
            elif ab['outcome'] in ('SINGLE', 'DOUBLE', 'TRIPLE', 'HOME RUN', 'GRAND SLAM'):
                p['hits_allowed'] += 1
                if ab['outcome'] in ('HOME RUN', 'GRAND SLAM'):
                    p['hr_allowed'] += 1
            if ab['exit_velocity']:
                p['ev_allowed_sum'] += ab['exit_velocity']
                p['ev_allowed_n'] += 1

    return by_batter, by_pitcher


def main():
    save_dir = Path(
        r'C:\Users\Cameron\OneDrive\Documents\Out of the Park Developments'
        r'\OOTP Baseball 27\saved_games\7ea0000000000000000003a4.pt'
    )
    log_dir = save_dir / 'news' / 'html' / 'game_logs'
    logs = sorted(log_dir.glob('log_*.html'))
    print(f'Found {len(logs)} game logs')

    all_at_bats = []
    for log in logs:
        all_at_bats.extend(parse_game_log(str(log)))
    print(f'Total at-bats across all games: {len(all_at_bats)}')

    by_batter, by_pitcher = summarize(all_at_bats)

    print()
    print('=== TOP 10 BATTERS BY PA (with EV + hit-type breakdown) ===')
    print(f'{"Batter":22s}  PA   K%  BB%  AVG EV  HR  LD%  GB%  FB%')
    top = sorted(by_batter.items(), key=lambda x: -x[1]['pa'])[:10]
    for name, s in top:
        k_pct = s['k'] / s['pa'] * 100 if s['pa'] else 0
        bb_pct = s['bb'] / s['pa'] * 100 if s['pa'] else 0
        avg_ev = s['ev_sum'] / s['ev_n'] if s['ev_n'] else 0
        ld_pct = s['ld'] / (s['ld'] + s['gb'] + s['fb']) * 100 if (s['ld'] + s['gb'] + s['fb']) else 0
        gb_pct = s['gb'] / (s['ld'] + s['gb'] + s['fb']) * 100 if (s['ld'] + s['gb'] + s['fb']) else 0
        fb_pct = s['fb'] / (s['ld'] + s['gb'] + s['fb']) * 100 if (s['ld'] + s['gb'] + s['fb']) else 0
        print(f'{name:22s}  {s["pa"]:>3}  {k_pct:>4.1f}  {bb_pct:>4.1f}  {avg_ev:>5.1f}   {s["hr"]:>2}  '
              f'{ld_pct:>4.0f}  {gb_pct:>4.0f}  {fb_pct:>4.0f}')

    print()
    print('=== TOP 10 PITCHERS BY BF (K% / BB% / avg-EV-allowed) ===')
    print(f'{"Pitcher":22s}  BF  K%   BB%  HR  AVG EV')
    top = sorted(by_pitcher.items(), key=lambda x: -x[1]['bf'])[:10]
    for name, s in top:
        k_pct = s['k'] / s['bf'] * 100 if s['bf'] else 0
        bb_pct = s['bb'] / s['bf'] * 100 if s['bf'] else 0
        avg_ev = s['ev_allowed_sum'] / s['ev_allowed_n'] if s['ev_allowed_n'] else 0
        print(f'{name:22s}  {s["bf"]:>3}  {k_pct:>4.1f}  {bb_pct:>4.1f}  {s["hr_allowed"]:>2}  {avg_ev:>5.1f}')


if __name__ == '__main__':
    main()
