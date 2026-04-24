"""Recent-performance roll-ups for the franchise dashboard.

Pulls per-game batting + pitching lines from the last N games the user's
team played and aggregates them into:

    * Top 5 batters by recent production (OPS / wRC+ proxy / hot streak)
    * Top 5 pitchers by recent run prevention
    * Trend lines per key player — their last-10 rolling stat vs season
    * Clutch events in the recent window (2-out RBIs, LOB failures)

The output feeds a compact "Last N games" panel on the Roster Optimizer.
All data is local to the SQLite DB — no network calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RecentBatter:
    player_name: str
    card_id: Optional[int]
    games: int
    pa: int
    hits: int
    hr: int
    rbi: int
    bb: int
    k: int
    avg: float
    ops: float           # computed from raw counts
    wrc_proxy: float     # simple linear weight proxy


@dataclass
class RecentPitcher:
    player_name: str
    card_id: Optional[int]
    games: int
    ip: float
    k: int
    bb: int
    hr: int
    er: int
    era: float
    whip: float
    k_per_9: float


@dataclass
class RecentPerformance:
    window_games: int
    date_from: Optional[str]
    date_to: Optional[str]
    team_games: int
    top_batters: list[RecentBatter] = field(default_factory=list)
    top_pitchers: list[RecentPitcher] = field(default_factory=list)
    clutch_recent: list[dict] = field(default_factory=list)


def _ops_from_line(pa: int, h: int, bb: int, hr: int, ab: int) -> tuple[float, float]:
    """Rough OPS: OBP ≈ (H+BB)/PA; SLG ≈ (H+3*HR)/AB (no doubles/triples data)."""
    obp = ((h + bb) / pa) if pa > 0 else 0.0
    slg = ((h + 3 * hr) / ab) if ab > 0 else 0.0
    return obp, slg


def _ip_to_outs(ip: float) -> int:
    """OOTP uses .1 = 1 out, .2 = 2 outs decimal-encoded."""
    if ip is None:
        return 0
    whole = int(ip)
    frac = round((ip - whole) * 10)
    return whole * 3 + frac


def recent_performance(
    team_name: str,
    window_games: int = 10,
    *,
    min_pa: int = 5,
    min_ip: float = 3.0,
) -> RecentPerformance:
    """Build recent-performance summary for `team_name`.

    window_games: number of most-recent team games to include.
    """
    from app.core.database import get_connection
    conn = get_connection()

    # Find the last N Toronto games
    games = conn.execute(
        """
        SELECT game_id, game_date FROM games
        WHERE home_team = ? OR away_team = ?
        ORDER BY game_date DESC
        LIMIT ?
        """,
        (team_name, team_name, window_games),
    ).fetchall()
    if not games:
        return RecentPerformance(
            window_games=window_games, date_from=None, date_to=None,
            team_games=0,
        )

    game_ids = [g['game_id'] for g in games]
    date_from = games[-1]['game_date']
    date_to = games[0]['game_date']
    placeholders = ','.join(['?'] * len(game_ids))

    # Batters — aggregate across those games
    bat_rows = conn.execute(
        f"""
        SELECT player_name, card_id,
               COUNT(*) as games,
               SUM(ab) as ab, SUM(h) as h, SUM(rbi) as rbi,
               SUM(bb) as bb, SUM(k) as k,
               SUM(ab + bb) as pa_est,
               SUM(CASE WHEN season_hr IS NOT NULL THEN 1 ELSE 0 END) as hr_games
        FROM game_batting
        WHERE game_id IN ({placeholders}) AND team_name = ?
        GROUP BY player_name, card_id
        HAVING SUM(ab + bb) >= ?
        """,
        (*game_ids, team_name, min_pa),
    ).fetchall()

    batters: list[RecentBatter] = []
    for r in bat_rows:
        ab = int(r['ab'] or 0); h = int(r['h'] or 0); bb = int(r['bb'] or 0)
        pa = int(r['pa_est'] or 0); k = int(r['k'] or 0); rbi = int(r['rbi'] or 0)
        # HR proxy — game_batting doesn't have per-game HR count, use season_hr trend
        hr_est = int(r['hr_games'] or 0)  # approximation
        obp, slg = _ops_from_line(pa, h, bb, hr_est, ab)
        avg = (h / ab) if ab > 0 else 0.0
        # wRC+ proxy: (1.8*OBP + SLG) * 100 / league-mean (approx 0.72)
        wrc = (1.8 * obp + slg) * 100 / 0.72 if pa > 0 else 0.0
        batters.append(RecentBatter(
            player_name=r['player_name'] or '?',
            card_id=r['card_id'],
            games=int(r['games'] or 0),
            pa=pa, hits=h, hr=hr_est, rbi=rbi, bb=bb, k=k,
            avg=avg, ops=obp + slg, wrc_proxy=wrc,
        ))
    batters.sort(key=lambda b: -b.wrc_proxy)

    # Pitchers — aggregate
    pit_rows = conn.execute(
        f"""
        SELECT player_name, card_id,
               COUNT(*) as games,
               SUM(ip) as ip, SUM(k) as k, SUM(bb) as bb, SUM(hr) as hr,
               SUM(er) as er, SUM(h) as h,
               SUM(batters_faced) as bf
        FROM game_pitching
        WHERE game_id IN ({placeholders}) AND team_name = ?
        GROUP BY player_name, card_id
        HAVING SUM(ip) >= ?
        """,
        (*game_ids, team_name, min_ip),
    ).fetchall()

    pitchers: list[RecentPitcher] = []
    for r in pit_rows:
        ip = float(r['ip'] or 0)
        outs = _ip_to_outs(ip)
        innings = outs / 3.0 if outs else 0.001
        er = int(r['er'] or 0); bb = int(r['bb'] or 0); h = int(r['h'] or 0)
        k = int(r['k'] or 0); hr = int(r['hr'] or 0)
        era = (er * 9.0 / innings) if innings > 0 else 0.0
        whip = ((bb + h) / innings) if innings > 0 else 0.0
        k9 = (k * 9.0 / innings) if innings > 0 else 0.0
        pitchers.append(RecentPitcher(
            player_name=r['player_name'] or '?',
            card_id=r['card_id'],
            games=int(r['games'] or 0),
            ip=ip, k=k, bb=bb, hr=hr, er=er,
            era=era, whip=whip, k_per_9=k9,
        ))
    # Sort by fewer runs allowed per inning (relevant recency)
    pitchers.sort(key=lambda p: (p.era, -p.k_per_9))

    # Clutch events in the window
    clutch_rows = conn.execute(
        f"""
        SELECT event_type, player_name, card_id, game_id, event_count
        FROM game_clutch_events
        WHERE game_id IN ({placeholders})
          AND event_type IN ('2OUT_RBI', 'LOB_RISP_2OUT', 'HR', 'ERROR',
                             'INHERITED_SCORED')
        ORDER BY id DESC
        LIMIT 50
        """,
        tuple(game_ids),
    ).fetchall()
    clutch = [dict(r) for r in clutch_rows]

    return RecentPerformance(
        window_games=window_games,
        date_from=date_from, date_to=date_to,
        team_games=len(game_ids),
        top_batters=batters[:8],
        top_pitchers=pitchers[:8],
        clutch_recent=clutch,
    )
