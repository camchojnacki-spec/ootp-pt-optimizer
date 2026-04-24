"""Derived analytics tables — rebuilt from raw ingested data.

Each `build_*()` function below rebuilds one derived table from scratch so
ordering between them doesn't matter. They're idempotent (DELETE+INSERT
style) and safe to re-run on every background-worker refresh.

Design rules:
    * No new raw data is created here. Every column is computed from
      already-ingested tables (`game_log_at_bats`, `game_batting`,
      `batting_stats`, `price_snapshots`, `league_team_stats`, etc.).
    * Prefer observed data (game logs) over ratings-derived signals —
      these tables exist precisely to expose where reality diverges from
      the card-rating implied expectation.
    * Every table scopes by `card_id` + `league_id` where meaningful so
      multi-league play stays un-polluted.

`build_all()` wraps the full rebuild in a single connection and returns a
summary dict so the worker can log row counts.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from app.core.database import get_connection

logger = logging.getLogger(__name__)


# Normalize 'LHP'/'RHP' -> 'L'/'R' for the split dimension. 'SHP' has never
# been seen in OOTP exports but we defensively accept it.
_HAND_NORM = {'LHP': 'L', 'RHP': 'R', 'L': 'L', 'R': 'R'}


# ──────────────────────────────────────────────────────────────────────
# Table 1: batter_vs_pitcher_hand — observed L/R splits per card
# ──────────────────────────────────────────────────────────────────────

def build_batter_vs_pitcher_hand(conn: Optional[sqlite3.Connection] = None,
                                  min_pa: int = 5) -> dict:
    """Aggregate game_log_at_bats into per-card L/R split lines.

    Args:
        conn: optional existing connection. Opens a new one if None.
        min_pa: minimum plate appearances before we emit a row. Below this
            the sample is too noisy to be useful — OOTP runs short seasons
            and a 2-PA line would be misleading.

    Returns a dict with `rows_written` and `cards_covered`. The caller can
    log this or surface it in the data-refresh UI.
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()

    try:
        cursor = conn.cursor()

        # Compute everything in one SQL pass — portable across SQLite
        # versions and fast enough at our row count (~24k at-bats).
        # We normalize hand here and scope by league_id via games.
        rows = cursor.execute("""
            WITH ab_joined AS (
                SELECT
                    ab.batter_card_id AS card_id,
                    CASE ab.pitcher_hand
                        WHEN 'LHP' THEN 'L'
                        WHEN 'RHP' THEN 'R'
                        ELSE NULL END               AS pitcher_hand,
                    g.league_id                      AS league_id,
                    ab.outcome                       AS outcome,
                    ab.game_id                       AS game_id
                FROM game_log_at_bats ab
                JOIN games g ON g.game_id = ab.game_id
                WHERE ab.batter_card_id IS NOT NULL
                  AND ab.pitcher_hand IN ('LHP','RHP')
                  AND ab.outcome IS NOT NULL
            ),
            agg AS (
                SELECT
                    card_id,
                    pitcher_hand,
                    league_id,
                    COUNT(*)                                       AS pa,
                    SUM(CASE WHEN outcome NOT IN ('BB','HBP','SAC') THEN 1 ELSE 0 END) AS ab,
                    SUM(CASE WHEN outcome IN ('SINGLE','DOUBLE','TRIPLE','HR') THEN 1 ELSE 0 END) AS h,
                    SUM(CASE WHEN outcome = 'DOUBLE' THEN 1 ELSE 0 END)   AS doubles,
                    SUM(CASE WHEN outcome = 'TRIPLE' THEN 1 ELSE 0 END)   AS triples,
                    SUM(CASE WHEN outcome = 'HR'     THEN 1 ELSE 0 END)   AS hr,
                    SUM(CASE WHEN outcome = 'BB'     THEN 1 ELSE 0 END)   AS bb,
                    SUM(CASE WHEN outcome = 'K'      THEN 1 ELSE 0 END)   AS k,
                    SUM(CASE WHEN outcome = 'HBP'    THEN 1 ELSE 0 END)   AS hbp,
                    COUNT(DISTINCT game_id)                        AS games_sample
                FROM ab_joined
                GROUP BY card_id, pitcher_hand, league_id
            )
            SELECT *,
                   -- Derived rates. NULLIF guards against div-by-zero at
                   -- edges. SLG uses total bases: 1B + 2*2B + 3*3B + 4*HR.
                   CAST(h AS REAL) / NULLIF(ab, 0)                                 AS avg,
                   CAST(h + bb + hbp AS REAL) / NULLIF(pa, 0)                      AS obp,
                   CAST((h - doubles - triples - hr) + 2*doubles + 3*triples + 4*hr AS REAL)
                       / NULLIF(ab, 0)                                             AS slg,
                   CAST(k  AS REAL) / NULLIF(pa, 0)                                AS k_rate,
                   CAST(bb AS REAL) / NULLIF(pa, 0)                                AS bb_rate,
                   CAST(hr AS REAL) / NULLIF(pa, 0)                                AS hr_rate
            FROM agg
            WHERE pa >= ?
        """, (min_pa,)).fetchall()

        # Clear + insert fresh. We don't want stale rows hanging around
        # when a card drops below min_pa after data corrections.
        cursor.execute("DELETE FROM batter_vs_pitcher_hand")

        cards_covered: set[int] = set()
        for r in rows:
            (card_id, pitcher_hand, league_id, pa, ab, h, doubles, triples,
             hr, bb, k, hbp, games_sample,
             avg, obp, slg, k_rate, bb_rate, hr_rate) = r
            ops = (obp or 0.0) + (slg or 0.0) if (obp is not None and slg is not None) else None
            iso = (slg - avg) if (slg is not None and avg is not None) else None
            cursor.execute("""
                INSERT INTO batter_vs_pitcher_hand
                    (card_id, pitcher_hand, league_id,
                     pa, ab, h, doubles, triples, hr, bb, k, hbp,
                     avg, obp, slg, ops, iso, k_rate, bb_rate, hr_rate,
                     games_sample)
                VALUES (?,?,?, ?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?,?, ?)
            """, (card_id, pitcher_hand, league_id,
                  pa, ab, h, doubles, triples, hr, bb, k, hbp,
                  avg, obp, slg, ops, iso, k_rate, bb_rate, hr_rate,
                  games_sample))
            cards_covered.add(card_id)

        conn.commit()
        result = {
            "table": "batter_vs_pitcher_hand",
            "rows_written": len(rows),
            "cards_covered": len(cards_covered),
            "min_pa": min_pa,
        }
        logger.info("derived_stats: %s", result)
        return result
    finally:
        if owns_conn:
            conn.close()


# ──────────────────────────────────────────────────────────────────────
# Table 2: pitcher_fatigue — rolling usage window per card
# ──────────────────────────────────────────────────────────────────────

def _derive_role_profile(counts: dict[str, int]) -> str:
    """Pick the most-seen role for a pitcher (SP / RP / CL)."""
    if not counts:
        return 'RP'
    # Collapse all save situations into CL, starter flags into SP.
    buckets = {'SP': 0, 'CL': 0, 'RP': 0}
    for role, n in counts.items():
        r = (role or '').upper()
        if r in ('SP',):
            buckets['SP'] += n
        elif r in ('SV', 'BS'):
            buckets['CL'] += n
        else:
            # W/L/HLD/None all lean RP (could be starter W but the
            # non-starter appearances dominate for starters-in-relief).
            buckets['RP'] += n
    return max(buckets, key=buckets.get)


def _availability_signal(days_rest: Optional[int],
                         pitches_last_3: int,
                         pitches_last_7: int,
                         apps_last_3: int,
                         last_outing_pitches: Optional[int],
                         role_profile: str) -> str:
    """Rough availability gauge for tonight's game.

    Thresholds tuned for PT where starters go every 5th day and relievers
    typically should not throw 3 consecutive days or >40 pitches in 3 days.
    """
    if days_rest is None:
        return 'unknown'

    if role_profile == 'SP':
        # Starters: standard 5-day rotation. Fresh at day 4+.
        if days_rest >= 4:
            return 'fresh'
        if days_rest >= 2:
            return 'available'   # short rest outings
        return 'unavailable'

    # Relievers (RP/CL)
    if days_rest == 0:
        # Pitched today — unavailable unless they threw <10 pitches (rare).
        if last_outing_pitches is not None and last_outing_pitches <= 10:
            return 'tired'
        return 'unavailable'
    if apps_last_3 >= 3:
        return 'unavailable'     # three days straight
    if pitches_last_3 >= 45:
        return 'tired'
    if pitches_last_7 >= 90:
        return 'tired'
    if days_rest >= 1:
        return 'available'
    return 'available'


def build_pitcher_fatigue(conn: Optional[sqlite3.Connection] = None) -> dict:
    """Summarize each pitcher's rolling workload up to the latest game date.

    The "anchor" is the latest `games.game_date` in the DB — simulated season
    clock, not wall-clock today. Callers who want a fatigue view as of a
    specific day can extend this later to accept an anchor override.
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()

    try:
        cursor = conn.cursor()

        # Anchor date per league — latest game we've ingested.
        leagues = [r[0] for r in cursor.execute(
            "SELECT DISTINCT league_id FROM games WHERE league_id IS NOT NULL"
        )]
        if not leagues:
            logger.info("pitcher_fatigue: no games ingested yet")
            cursor.execute("DELETE FROM pitcher_fatigue")
            conn.commit()
            return {"table": "pitcher_fatigue", "rows_written": 0,
                    "cards_covered": 0, "reason": "no games"}

        cursor.execute("DELETE FROM pitcher_fatigue")
        rows_written = 0
        cards_covered: set[int] = set()

        for league_id in leagues:
            anchor = cursor.execute(
                "SELECT MAX(g.game_date) FROM games g WHERE g.league_id = ?",
                (league_id,),
            ).fetchone()[0]
            if not anchor:
                continue

            # Pull per-card aggregates in one pass.
            agg = cursor.execute("""
                WITH joined AS (
                    SELECT gp.card_id, gp.pitches, gp.role_flag,
                           g.game_date,
                           julianday(?) - julianday(g.game_date) AS d
                    FROM game_pitching gp
                    JOIN games g ON g.game_id = gp.game_id
                    WHERE gp.card_id IS NOT NULL
                      AND gp.card_id > 0
                      AND g.league_id = ?
                )
                SELECT card_id,
                       MAX(game_date) AS last_date,
                       COUNT(*) AS total_apps,
                       SUM(pitches) AS total_pitches,
                       SUM(CASE WHEN d <= 3 THEN 1 ELSE 0 END) AS apps3,
                       SUM(CASE WHEN d <= 7 THEN 1 ELSE 0 END) AS apps7,
                       SUM(CASE WHEN d <= 3 THEN COALESCE(pitches,0) ELSE 0 END) AS pitches3,
                       SUM(CASE WHEN d <= 7 THEN COALESCE(pitches,0) ELSE 0 END) AS pitches7
                FROM joined
                GROUP BY card_id
            """, (anchor, league_id)).fetchall()

            for row in agg:
                (card_id, last_date, total_apps, total_pitches,
                 apps3, apps7, pitches3, pitches7) = row

                # Role mix: collect role_flag distribution
                role_rows = cursor.execute("""
                    SELECT COALESCE(gp.role_flag, '') role, COUNT(*) n
                    FROM game_pitching gp
                    JOIN games g ON g.game_id = gp.game_id
                    WHERE gp.card_id = ? AND g.league_id = ?
                    GROUP BY role
                """, (card_id, league_id)).fetchall()
                role_counts = {r[0]: r[1] for r in role_rows}
                role_profile = _derive_role_profile(role_counts)

                # Last outing detail
                last_outing = cursor.execute("""
                    SELECT gp.pitches, gp.role_flag
                    FROM game_pitching gp
                    JOIN games g ON g.game_id = gp.game_id
                    WHERE gp.card_id = ? AND g.league_id = ?
                      AND g.game_date = ?
                    ORDER BY gp.appearance_order DESC LIMIT 1
                """, (card_id, league_id, last_date)).fetchone()
                last_pitches = last_outing[0] if last_outing else None
                last_role = last_outing[1] if last_outing else None

                days_rest = None
                if last_date:
                    days_rest = int(cursor.execute(
                        "SELECT CAST(julianday(?) - julianday(?) AS INTEGER)",
                        (anchor, last_date),
                    ).fetchone()[0])

                avail = _availability_signal(
                    days_rest, pitches3 or 0, pitches7 or 0, apps3 or 0,
                    last_pitches, role_profile,
                )

                cursor.execute("""
                    INSERT INTO pitcher_fatigue
                        (card_id, league_id, last_appearance_date, days_rest,
                         appearances_last_3_days, appearances_last_7_days,
                         pitches_last_3_days, pitches_last_7_days,
                         last_appearance_pitches, last_appearance_role,
                         total_appearances, total_pitches, role_profile,
                         availability, anchor_date)
                    VALUES (?,?,?,?, ?,?, ?,?, ?,?, ?,?, ?, ?, ?)
                """, (card_id, league_id, last_date, days_rest,
                      apps3 or 0, apps7 or 0,
                      pitches3 or 0, pitches7 or 0,
                      last_pitches, last_role,
                      total_apps, total_pitches, role_profile,
                      avail, anchor))
                cards_covered.add(card_id)
                rows_written += 1

        conn.commit()
        result = {
            "table": "pitcher_fatigue",
            "rows_written": rows_written,
            "cards_covered": len(cards_covered),
            "leagues": leagues,
        }
        logger.info("derived_stats: %s", result)
        return result
    finally:
        if owns_conn:
            conn.close()


# ──────────────────────────────────────────────────────────────────────
# Table 3: clutch_profile_card — aggregated clutch/situational events
# ──────────────────────────────────────────────────────────────────────

def build_clutch_profile_card(conn: Optional[sqlite3.Connection] = None,
                               min_events: int = 3) -> dict:
    """Roll game_clutch_events up to per-card totals + derived rates.

    Two headline rates:
      * clutch_rbi_rate = 2OUT_RBI / (2OUT_RBI + LOB_RISP_2OUT) — the
        batter's conversion rate in the highest-leverage PA of the inning.
      * inherited_strand_rate = 1 - (inh_scored / inh_runners) — the
        reliever's "fireman" metric; 1.0 means every inherited runner
        stranded.

    min_events filters out cards with <N total clutch events so the rates
    aren't noise. We preserve raw counts regardless.
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()

    try:
        cursor = conn.cursor()
        rows = cursor.execute("""
            SELECT
                ce.card_id,
                g.league_id,
                SUM(CASE WHEN ce.event_type='2OUT_RBI'          THEN COALESCE(ce.event_count,1) ELSE 0 END) AS two_out_rbi,
                SUM(CASE WHEN ce.event_type='LOB_RISP_2OUT'     THEN COALESCE(ce.event_count,1) ELSE 0 END) AS lob_risp,
                SUM(CASE WHEN ce.event_type='SAC_FLY'           THEN COALESCE(ce.event_count,1) ELSE 0 END) AS sac_fly,
                SUM(CASE WHEN ce.event_type='GIDP'              THEN COALESCE(ce.event_count,1) ELSE 0 END) AS gidp,
                SUM(CASE WHEN ce.event_type='DOUBLE'            THEN COALESCE(ce.event_count,1) ELSE 0 END) AS doubles,
                SUM(CASE WHEN ce.event_type='TRIPLE'            THEN COALESCE(ce.event_count,1) ELSE 0 END) AS triples,
                SUM(CASE WHEN ce.event_type='HR'                THEN COALESCE(ce.event_count,1) ELSE 0 END) AS hr_clutch,
                SUM(CASE WHEN ce.event_type='SB'                THEN COALESCE(ce.event_count,1) ELSE 0 END) AS sb,
                SUM(CASE WHEN ce.event_type='CS'                THEN COALESCE(ce.event_count,1) ELSE 0 END) AS cs,
                SUM(CASE WHEN ce.event_type='INHERITED_RUNNERS' THEN COALESCE(ce.event_count,1) ELSE 0 END) AS inh_runners,
                SUM(CASE WHEN ce.event_type='INHERITED_SCORED'  THEN COALESCE(ce.event_count,1) ELSE 0 END) AS inh_scored,
                SUM(CASE WHEN ce.event_type='ERROR'             THEN COALESCE(ce.event_count,1) ELSE 0 END) AS errors,
                COUNT(DISTINCT ce.game_id) AS games_sample,
                COUNT(*) AS total_events
            FROM game_clutch_events ce
            JOIN games g ON g.game_id = ce.game_id
            WHERE ce.card_id IS NOT NULL
              AND ce.card_id > 0
            GROUP BY ce.card_id, g.league_id
        """).fetchall()

        cursor.execute("DELETE FROM clutch_profile_card")
        rows_written = 0
        cards_covered: set[int] = set()

        for r in rows:
            (card_id, league_id,
             two_out_rbi, lob_risp, sac_fly, gidp,
             doubles, triples, hr_clutch, sb, cs,
             inh_runners, inh_scored, errors,
             games_sample, total_events) = r

            if total_events < min_events:
                continue

            # Batter clutch conversion rate
            batter_clutch_total = two_out_rbi + lob_risp
            clutch_rbi_rate = (two_out_rbi / batter_clutch_total) if batter_clutch_total > 0 else None

            # Reliever fireman rate (1 = stranded all, 0 = scored all)
            strand_rate = None
            if inh_runners and inh_runners > 0:
                strand_rate = 1.0 - (inh_scored / inh_runners)

            net_steals = (sb or 0) - (cs or 0)

            cursor.execute("""
                INSERT INTO clutch_profile_card
                    (card_id, league_id,
                     two_out_rbi, lob_risp_2out, sac_fly, gidp,
                     doubles, triples, hr_clutch, sb, cs,
                     inherited_runners, inherited_scored, errors,
                     clutch_rbi_rate, inherited_strand_rate, net_steals,
                     games_sample)
                VALUES (?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?, ?,?,?, ?)
            """, (card_id, league_id,
                  two_out_rbi, lob_risp, sac_fly, gidp,
                  doubles, triples, hr_clutch, sb, cs,
                  inh_runners, inh_scored, errors,
                  clutch_rbi_rate, strand_rate, net_steals,
                  games_sample))
            cards_covered.add(card_id)
            rows_written += 1

        conn.commit()
        result = {
            "table": "clutch_profile_card",
            "rows_written": rows_written,
            "cards_covered": len(cards_covered),
            "min_events": min_events,
        }
        logger.info("derived_stats: %s", result)
        return result
    finally:
        if owns_conn:
            conn.close()


# ──────────────────────────────────────────────────────────────────────
# Table 4: park_adjusted_stats — weighted park factor per card
# ──────────────────────────────────────────────────────────────────────

def build_park_adjusted_stats(conn: Optional[sqlite3.Connection] = None,
                               min_pa: int = 20) -> dict:
    """Compute each card's PA-weighted park factor and adjusted OPS.

    Game-by-game: the park where a game happens is the home team's park.
    We pull `league_team_stats.pf_overall` (+ `pf_hr`, `pf_avg`) for the
    latest snapshot per (league, team), then weight by games_played
    at that park to produce a card-level park factor.

    Adjusted = raw / park_factor — straight OPS+ style. If park factors
    are ~1.0 (as in LB124 currently), adjusted ≈ raw, and the table is
    a no-op. The table still exists so queries can join it safely in
    leagues where park effects matter.
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()
    try:
        cursor = conn.cursor()

        # Latest park factor snapshot per (league, team).
        park_rows = cursor.execute("""
            SELECT league_id, team_name, pf_overall, pf_hr, pf_avg
            FROM league_team_stats lts1
            WHERE snapshot_date = (
                SELECT MAX(snapshot_date) FROM league_team_stats lts2
                WHERE lts2.league_id = lts1.league_id
                  AND lts2.team_name = lts1.team_name
            )
        """).fetchall()
        park_by_team = {
            (r[0], r[1]): (r[2] or 1.0, r[3] or 1.0, r[4] or 1.0)
            for r in park_rows
        }

        # Per (card, league) aggregate over game_batting, with park factor
        # weighted by # of distinct games at each park.
        # home_team from `games` = the park hosting the game.
        cursor.execute("""
            SELECT gb.card_id,
                   g.league_id,
                   g.home_team,
                   COUNT(DISTINCT g.game_id) AS games_here,
                   SUM(COALESCE(gb.ab, 0))    AS ab,
                   SUM(COALESCE(gb.h, 0))     AS h,
                   SUM(COALESCE(gb.bb, 0))    AS bb,
                   SUM(COALESCE(gb.k, 0))     AS k
                   -- NB: game_batting doesn't separate doubles/triples/hr
                   -- at the per-row level; we'll approximate SLG using
                   -- season_hr and assume ISO patterns. For LB124 where
                   -- PFs=1 the adjustment is a no-op anyway.
            FROM game_batting gb
            JOIN games g ON g.game_id = gb.game_id
            WHERE gb.card_id IS NOT NULL
              AND gb.card_id > 0
            GROUP BY gb.card_id, g.league_id, g.home_team
        """)
        per_game = cursor.fetchall()

        # Group by (card, league) — accumulate totals and weighted PFs.
        buckets: dict[tuple, dict] = {}
        for r in per_game:
            card_id, league_id, home_team, games_here, ab, h, bb, k = r
            pf = park_by_team.get((league_id, home_team), (1.0, 1.0, 1.0))
            key = (card_id, league_id)
            b = buckets.setdefault(key, {
                "games": 0, "ab": 0, "h": 0, "bb": 0, "k": 0,
                "pf_overall_w": 0.0, "pf_hr_w": 0.0, "pf_avg_w": 0.0,
            })
            b["games"] += games_here or 0
            b["ab"] += ab or 0
            b["h"] += h or 0
            b["bb"] += bb or 0
            b["k"] += k or 0
            b["pf_overall_w"] += (games_here or 0) * pf[0]
            b["pf_hr_w"]      += (games_here or 0) * pf[1]
            b["pf_avg_w"]     += (games_here or 0) * pf[2]

        # Pull season HR totals from batting_stats per card so we have
        # something resembling SLG/HR-rate. Fall back to 0 if missing.
        hr_by_card: dict[tuple, int] = {}
        pa_by_card: dict[tuple, int] = {}
        for r in cursor.execute("""
            SELECT bs.card_id, bs.league_id, MAX(bs.hr) AS hr, MAX(bs.pa) AS pa
            FROM batting_stats bs
            WHERE bs.card_id IS NOT NULL
            GROUP BY bs.card_id, bs.league_id
        """):
            hr_by_card[(r[0], r[1])] = r[2] or 0
            pa_by_card[(r[0], r[1])] = r[3] or 0

        cursor.execute("DELETE FROM park_adjusted_stats")
        rows_written = 0
        cards_covered: set[int] = set()

        for (card_id, league_id), b in buckets.items():
            pa_season = pa_by_card.get((card_id, league_id), 0)
            hr_season = hr_by_card.get((card_id, league_id), 0)
            pa_used = max(pa_season, b["ab"] + b["bb"])
            if pa_used < min_pa:
                continue

            pf_overall = (b["pf_overall_w"] / b["games"]) if b["games"] else 1.0
            pf_hr      = (b["pf_hr_w"]      / b["games"]) if b["games"] else 1.0
            pf_avg     = (b["pf_avg_w"]     / b["games"]) if b["games"] else 1.0

            raw_avg = (b["h"] / b["ab"]) if b["ab"] else None
            raw_obp = ((b["h"] + b["bb"]) / (b["ab"] + b["bb"])) if (b["ab"] + b["bb"]) else None
            # SLG: we don't know doubles/triples; approximate using season HR
            # as 4-base weight, all other hits treated as singles. Crude but
            # consistent across cards and good enough until game_batting
            # starts breaking down hit types per row.
            if b["ab"]:
                approx_tb = (b["h"] - hr_season) + 4 * hr_season if b["h"] >= hr_season else b["h"]
                raw_slg = approx_tb / b["ab"]
            else:
                raw_slg = None
            raw_ops = ((raw_obp or 0) + (raw_slg or 0)) if (raw_obp is not None and raw_slg is not None) else None

            adj_avg = (raw_avg / pf_avg) if (raw_avg is not None and pf_avg) else None
            adj_ops = (raw_ops / pf_overall) if (raw_ops is not None and pf_overall) else None
            adj_hr_rate = ((hr_season / pa_used) / pf_hr) if (pa_used and pf_hr) else None

            cursor.execute("""
                INSERT INTO park_adjusted_stats
                    (card_id, league_id, games_sample, pa, ab, h, hr,
                     raw_avg, raw_obp, raw_slg, raw_ops,
                     pf_overall_weighted, pf_hr_weighted, pf_avg_weighted,
                     adj_ops, adj_avg, adj_hr_rate)
                VALUES (?,?,?,?,?,?,?, ?,?,?,?, ?,?,?, ?,?,?)
            """, (card_id, league_id, b["games"], pa_used, b["ab"], b["h"], hr_season,
                  raw_avg, raw_obp, raw_slg, raw_ops,
                  pf_overall, pf_hr, pf_avg,
                  adj_ops, adj_avg, adj_hr_rate))
            cards_covered.add(card_id)
            rows_written += 1

        conn.commit()
        result = {
            "table": "park_adjusted_stats",
            "rows_written": rows_written,
            "cards_covered": len(cards_covered),
            "min_pa": min_pa,
            "parks_known": len(park_by_team),
        }
        logger.info("derived_stats: %s", result)
        return result
    finally:
        if owns_conn:
            conn.close()


# ──────────────────────────────────────────────────────────────────────
# Table 5: regression_candidates_v2 — ratings-vs-outcomes regression flags
# ──────────────────────────────────────────────────────────────────────

# Rating -> expected rate baselines. These are rough calibrations that let
# us flag *divergence*, not hit the exact rate. Better than nothing for
# small-sample PT leagues and cheap to adjust here if the league's rate
# distribution shifts meaningfully.
_BABIP_BASELINE = 0.300           # league average BABIP
_K_PCT_BASELINE = 0.22            # league average K rate (modern MLB ~22%)


def build_regression_candidates_v2(conn: Optional[sqlite3.Connection] = None,
                                    min_pa: int = 50) -> dict:
    """Flag cards whose observed outcomes diverge from rating expectations.

    Three signals combined into a regression_score (z-like scale, roughly
    standardized). Positive score = running hotter than expected → likely
    to regress down. Negative = underperforming → likely to regress up.

    * ops_plus_delta: observed OPS+ minus expected OPS+ (empirical curve
      fit from the league — 100 is league average).
    * babip_delta: (observed BABIP / expected BABIP) - 1, where expected
      scales the .300 baseline by the rated BABIP tendency.
    * k_pct_delta: (observed K%) - expected K%, where expected scales the
      .22 baseline inversely by avoid_ks rating.

    direction:
      * 'regress_down' if score > +0.30
      * 'regress_up'   if score < -0.30
      * 'sustainable'  otherwise
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()

    try:
        cursor = conn.cursor()

        # Step 1: empirical OPS+ curve by card_value bucket, using only the
        # latest batting_stats snapshot per card-league to avoid double
        # counting. Cards with tiny PA samples would pull the curve toward
        # noise, so gate at min_pa/2 (loose) for curve fitting.
        buckets = cursor.execute(f"""
            WITH latest AS (
                SELECT bs.card_id, bs.league_id, bs.ops_plus, bs.pa
                FROM batting_stats bs
                WHERE bs.id IN (
                    SELECT MAX(id) FROM batting_stats
                    WHERE card_id IS NOT NULL
                    GROUP BY card_id, COALESCE(league_id, '')
                )
                AND bs.pa >= {max(25, min_pa // 2)}
                AND bs.ops_plus IS NOT NULL
            )
            SELECT (c.card_value / 5) AS bucket,
                   AVG(l.ops_plus)    AS mean_ops_plus,
                   COUNT(*)           AS n
            FROM cards c JOIN latest l ON l.card_id = c.card_id
            WHERE c.card_value IS NOT NULL AND c.card_value > 0
              AND (c.position_name IS NULL
                   OR c.position_name NOT IN ('SP','RP','CL','P'))
              AND (c.pitcher_role_name IS NULL OR c.pitcher_role_name = '')
            GROUP BY bucket
            HAVING n >= 3
            ORDER BY bucket
        """).fetchall()
        expected_by_bucket = {r[0]: r[1] for r in buckets}

        def _expected_ops_plus(card_value: int) -> Optional[float]:
            if card_value is None or card_value <= 0:
                return None
            bucket = card_value // 5
            if bucket in expected_by_bucket:
                return expected_by_bucket[bucket]
            # Fall back to nearest bucket
            keys = sorted(expected_by_bucket.keys())
            if not keys:
                return None
            # Linear search for closest bucket
            best = min(keys, key=lambda k: abs(k - bucket))
            return expected_by_bucket[best]

        # Step 2: pull observed + ratings per card. Exclude pitchers — they
        # have batting rows (OOTP tracks pitcher bats) but OPS+ is
        # meaningless there and would flood the regress-up list.
        rows = cursor.execute("""
            SELECT bs.card_id, bs.league_id, bs.pa,
                   bs.ops_plus, bs.babip,
                   CAST(bs.k AS REAL) / NULLIF(bs.pa, 0) AS k_pct,
                   c.card_value,
                   c.babip    AS rating_babip,
                   c.avoid_ks AS rating_avoid_ks
            FROM batting_stats bs
            JOIN cards c ON c.card_id = bs.card_id
            WHERE bs.id IN (
                SELECT MAX(id) FROM batting_stats
                WHERE card_id IS NOT NULL
                GROUP BY card_id, COALESCE(league_id, '')
            )
              AND bs.pa >= ?
              AND (c.position_name IS NULL
                   OR c.position_name NOT IN ('SP','RP','CL','P'))
              AND (c.pitcher_role_name IS NULL OR c.pitcher_role_name = '')
        """, (min_pa,)).fetchall()

        cursor.execute("DELETE FROM regression_candidates_v2")
        rows_written = 0

        for r in rows:
            (card_id, league_id, pa,
             ops_plus_obs, babip_obs, k_pct_obs,
             card_value, rating_babip, rating_avoid_ks) = r

            expected_ops_plus = _expected_ops_plus(card_value)
            ops_plus_delta = None
            if ops_plus_obs is not None and expected_ops_plus is not None:
                ops_plus_delta = ops_plus_obs - expected_ops_plus

            # BABIP: rating 50 maps to baseline .300. Rating 80 → higher
            # expected BABIP, rating 20 → lower. Linear scaling within a
            # modest band (±15% of baseline) since OOTP's BABIP rating
            # tends to have narrow true impact.
            babip_delta = None
            if babip_obs is not None and rating_babip is not None:
                # Scale ±15% across 0-100 rating range, pivot at 50
                expected_babip = _BABIP_BASELINE * (1 + 0.003 * (rating_babip - 50))
                babip_delta = (babip_obs / expected_babip) - 1 if expected_babip else None

            # K%: higher avoid_ks → lower expected K%. Baseline .22 at
            # rating 50; ±40% swing across rating range.
            k_pct_delta = None
            if k_pct_obs is not None and rating_avoid_ks is not None:
                expected_k = _K_PCT_BASELINE * (1 - 0.008 * (rating_avoid_ks - 50))
                if expected_k > 0:
                    k_pct_delta = k_pct_obs - expected_k

            # Composite regression score. Scale each component so they're
            # comparable, then sum. Positive = hotter than expected.
            components = []
            if ops_plus_delta is not None:
                components.append(ops_plus_delta / 30.0)         # ~1σ ≈ 30 OPS+
            if babip_delta is not None:
                components.append(babip_delta * 3.0)             # ±10% BABIP = ±0.30
            if k_pct_delta is not None:
                # Higher K% is BAD (sustainable weakness). Invert so
                # "lower K than expected" reads as "running hot".
                components.append(-k_pct_delta * 3.0)
            regression_score = (sum(components) / len(components)) if components else None

            direction = 'sustainable'
            if regression_score is not None:
                if regression_score > 0.30:
                    direction = 'regress_down'
                elif regression_score < -0.30:
                    direction = 'regress_up'

            confidence = min(1.0, (pa or 0) / 300.0)

            cursor.execute("""
                INSERT INTO regression_candidates_v2
                    (card_id, league_id, pa,
                     ops_plus_observed, babip_observed, k_pct_observed,
                     ops_plus_expected, ops_plus_delta, babip_delta, k_pct_delta,
                     regression_score, direction, confidence)
                VALUES (?,?,?, ?,?,?, ?,?,?,?, ?,?,?)
            """, (card_id, league_id, pa,
                  ops_plus_obs, babip_obs, k_pct_obs,
                  expected_ops_plus, ops_plus_delta, babip_delta, k_pct_delta,
                  regression_score, direction, confidence))
            rows_written += 1

        conn.commit()
        result = {
            "table": "regression_candidates_v2",
            "rows_written": rows_written,
            "buckets_fitted": len(expected_by_bucket),
            "min_pa": min_pa,
        }
        logger.info("derived_stats: %s", result)
        return result
    finally:
        if owns_conn:
            conn.close()


# ──────────────────────────────────────────────────────────────────────
# Table 6: opponent_scouting — per-team rolling form + bullpen snapshot
# ──────────────────────────────────────────────────────────────────────

def build_opponent_scouting(conn: Optional[sqlite3.Connection] = None) -> dict:
    """Per-team form over last 5/10/20 games + bullpen availability.

    Drives the "tonight's opponent" lookup. Anchored to the latest
    `games.game_date` per league.
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()
    try:
        cursor = conn.cursor()

        leagues = [r[0] for r in cursor.execute(
            "SELECT DISTINCT league_id FROM games WHERE league_id IS NOT NULL"
        )]
        if not leagues:
            cursor.execute("DELETE FROM opponent_scouting")
            conn.commit()
            return {"table": "opponent_scouting", "rows_written": 0,
                    "reason": "no games"}

        cursor.execute("DELETE FROM opponent_scouting")
        rows_written = 0

        for league_id in leagues:
            anchor = cursor.execute(
                "SELECT MAX(game_date) FROM games WHERE league_id = ?",
                (league_id,),
            ).fetchone()[0]
            if not anchor:
                continue

            # Distinct teams that played in this league
            teams = [r[0] for r in cursor.execute("""
                SELECT DISTINCT team FROM (
                    SELECT home_team AS team FROM games WHERE league_id = ?
                    UNION
                    SELECT away_team FROM games WHERE league_id = ?
                ) ORDER BY team
            """, (league_id, league_id))]

            for team in teams:
                # Per-game rows from this team's perspective (last N games).
                # Use a UNION where we flip home/away so the team's runs
                # are always "our_runs" and opponent's are "their_runs".
                # games.winner_team stores literal 'home' or 'away', not
                # the team name — decode it against home/away_team instead.
                games_rows = cursor.execute("""
                    SELECT game_date,
                           CASE WHEN home_team = ? THEN home_score ELSE away_score END AS our_r,
                           CASE WHEN home_team = ? THEN away_score ELSE home_score END AS their_r,
                           CASE
                               WHEN winner_team = 'home' AND home_team = ? THEN 1
                               WHEN winner_team = 'away' AND away_team = ? THEN 1
                               ELSE 0
                           END AS win
                    FROM games
                    WHERE league_id = ? AND (home_team = ? OR away_team = ?)
                    ORDER BY game_date DESC, game_id DESC
                    LIMIT 20
                """, (team, team, team, team, league_id, team, team)).fetchall()

                def _window(rows, n):
                    w = rows[:n]
                    wins = sum(r[3] for r in w)
                    rf = sum((r[1] or 0) for r in w)
                    ra = sum((r[2] or 0) for r in w)
                    return wins, len(w) - wins, rf, ra

                w5, l5, rf5, ra5 = _window(games_rows, 5)
                w10, l10, rf10, ra10 = _window(games_rows, 10)
                w20, l20, _, _ = _window(games_rows, 20)

                # Bullpen snapshot via pitcher_fatigue joined to the team
                # through league_rosters. Skip if nothing to aggregate.
                pen = cursor.execute("""
                    SELECT availability, COUNT(*) n
                    FROM pitcher_fatigue pf
                    JOIN league_rosters lr ON lr.card_id = pf.card_id
                      AND lr.league_id = pf.league_id
                    WHERE pf.league_id = ? AND lr.team_name = ?
                      AND (lr.position IN ('SP','RP','CL') OR lr.position IS NULL)
                    GROUP BY availability
                """, (league_id, team)).fetchall()
                pen_counts = {r[0]: r[1] for r in pen}

                cursor.execute("""
                    INSERT INTO opponent_scouting
                        (league_id, team_name, anchor_date,
                         last_5_wins, last_5_losses, last_5_runs_for, last_5_runs_against,
                         last_10_wins, last_10_losses, last_10_runs_for, last_10_runs_against,
                         last_20_wins, last_20_losses,
                         pitchers_fresh, pitchers_available,
                         pitchers_tired, pitchers_unavailable)
                    VALUES (?,?,?, ?,?,?,?, ?,?,?,?, ?,?, ?,?,?,?)
                """, (league_id, team, anchor,
                      w5, l5, rf5, ra5,
                      w10, l10, rf10, ra10,
                      w20, l20,
                      pen_counts.get('fresh', 0),
                      pen_counts.get('available', 0),
                      pen_counts.get('tired', 0),
                      pen_counts.get('unavailable', 0)))
                rows_written += 1

        conn.commit()
        result = {
            "table": "opponent_scouting",
            "rows_written": rows_written,
            "leagues": leagues,
        }
        logger.info("derived_stats: %s", result)
        return result
    finally:
        if owns_conn:
            conn.close()


# ──────────────────────────────────────────────────────────────────────
# Table 7: price_velocity — precomputed trend per card
# ──────────────────────────────────────────────────────────────────────

def build_price_velocity(conn: Optional[sqlite3.Connection] = None,
                          min_snapshots: int = 2) -> dict:
    """Compute per-card price trend from `player_history`.

    Skips cards with fewer than `min_snapshots` data points (can't compute
    a slope with one observation). As more snapshots accumulate with daily
    exports, this table's coverage expands automatically.

    Slopes use last_10_price (most stable series on thin markets); pct
    changes are relative to the window's starting price.
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()
    try:
        import statistics
        cursor = conn.cursor()

        # Latest snapshot date so "3d" and "7d" windows anchor consistently.
        latest_overall = cursor.execute(
            "SELECT MAX(DATE(snapshot_date)) FROM player_history"
        ).fetchone()[0]
        if not latest_overall:
            cursor.execute("DELETE FROM price_velocity")
            conn.commit()
            return {"table": "price_velocity", "rows_written": 0,
                    "reason": "no player_history"}

        # Pull all (card_id, day, l10_price, buy_high, sell_low) tuples.
        # Dedup within a day by keeping the most recent timestamp — when
        # two exports happen in one day we want the final price for that day.
        rows = cursor.execute("""
            SELECT card_id, DATE(snapshot_date) AS d,
                   last_10_price, buy_order_high, sell_order_low
            FROM player_history
            WHERE card_id IS NOT NULL
              AND id IN (
                  SELECT MAX(id) FROM player_history
                  WHERE card_id IS NOT NULL
                  GROUP BY card_id, DATE(snapshot_date)
              )
            ORDER BY card_id, d
        """).fetchall()

        # Group by card
        from collections import defaultdict
        per_card: dict[int, list[tuple[str, int, int, int]]] = defaultdict(list)
        for r in rows:
            per_card[r[0]].append((r[1], r[2], r[3], r[4]))

        cursor.execute("DELETE FROM price_velocity")
        rows_written = 0

        # Compute julian day for latest_overall once for anchoring
        latest_jd = cursor.execute(
            "SELECT julianday(?)", (latest_overall,)
        ).fetchone()[0]

        for card_id, series in per_card.items():
            if len(series) < min_snapshots:
                continue
            # series is already sorted by date ascending
            latest_date, latest_l10, latest_buy, latest_sell = series[-1]

            # Build a quick map day -> l10 price
            # Use cursor.execute per card is expensive; compute diffs in Python
            def _days_ago(d: str) -> int:
                jd = cursor.execute("SELECT julianday(?)", (d,)).fetchone()[0]
                return int(latest_jd - jd)

            # Filter to last 7d and last 3d windows
            win7 = [(d, l10) for (d, l10, _, _) in series if _days_ago(d) <= 7 and l10 is not None]
            win3 = [(d, l10) for (d, l10, _, _) in series if _days_ago(d) <= 3 and l10 is not None]

            avg_l10_3d = statistics.mean([l for _, l in win3]) if win3 else None
            avg_l10_7d = statistics.mean([l for _, l in win7]) if win7 else None
            vol_7d = statistics.stdev([l for _, l in win7]) if len(win7) >= 2 else None

            # Slope over the whole window: (last - first) / days
            first_d, first_l10 = None, None
            for (d, l10, _, _) in series:
                if l10 is not None:
                    first_d = d
                    first_l10 = l10
                    break
            window_days = _days_ago(first_d) if first_d else 0
            slope_per_day = None
            if first_l10 is not None and latest_l10 is not None and window_days > 0:
                slope_per_day = (latest_l10 - first_l10) / window_days

            # Percent changes relative to beginning of each window
            def _pct_change(win):
                if not win or win[0][1] in (None, 0) or win[-1][1] is None:
                    return None
                start = win[0][1]
                end = win[-1][1]
                if start == 0:
                    return None
                return (end - start) / start

            pct_3d = _pct_change(win3)
            pct_7d = _pct_change(win7)

            cursor.execute("""
                INSERT INTO price_velocity
                    (card_id, latest_date, latest_buy_high, latest_sell_low, latest_l10,
                     avg_l10_3d, avg_l10_7d, slope_l10_per_day,
                     pct_change_3d, pct_change_7d, volatility_7d,
                     snapshots_n, window_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (card_id, latest_date, latest_buy, latest_sell, latest_l10,
                  avg_l10_3d, avg_l10_7d, slope_per_day,
                  pct_3d, pct_7d, vol_7d,
                  len(series), window_days))
            rows_written += 1

        conn.commit()
        result = {
            "table": "price_velocity",
            "rows_written": rows_written,
            "latest_date": latest_overall,
            "min_snapshots": min_snapshots,
        }
        logger.info("derived_stats: %s", result)
        return result
    finally:
        if owns_conn:
            conn.close()


# ──────────────────────────────────────────────────────────────────────
# Table 8: meta_confidence — persisted card_aggregation.card_confidence()
# ──────────────────────────────────────────────────────────────────────

def build_meta_confidence(conn: Optional[sqlite3.Connection] = None,
                           league_id: Optional[str] = None) -> dict:
    """Persist the `card_confidence` score for every owned/active card.

    The live function does the heavy lifting — a card_aggregation call per
    card. Running it for ~2500 market cards would be slow, so we limit to
    cards that (a) are owned by a team in the league (via league_rosters),
    or (b) appear on the user's roster (via `roster`). That's ~900 cards
    per league, which completes in a few seconds.

    If `league_id` is None we loop over every registered league.
    """
    import json
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()
    try:
        from app.core.card_aggregation import card_confidence

        cursor = conn.cursor()
        leagues: list[Optional[str]] = []
        if league_id is not None:
            leagues = [league_id]
        else:
            rows = cursor.execute(
                "SELECT DISTINCT league_id FROM leagues WHERE league_id IS NOT NULL"
            ).fetchall()
            leagues = [r[0] for r in rows] or [None]

        cursor.execute("DELETE FROM meta_confidence")
        rows_written = 0

        for lg in leagues:
            # Candidate cards: anyone owned on Toronto or owned anywhere in
            # the league. `roster` covers user's team; `league_rosters`
            # covers all league teams (populated in Phase 1).
            card_ids = cursor.execute("""
                SELECT DISTINCT card_id FROM (
                    SELECT card_id FROM roster
                    WHERE card_id IS NOT NULL
                    UNION
                    SELECT card_id FROM league_rosters
                    WHERE card_id IS NOT NULL AND (league_id = ? OR ? IS NULL)
                )
            """, (lg, lg)).fetchall()
            card_ids = [r[0] for r in card_ids]

            for cid in card_ids:
                try:
                    result = card_confidence(card_id=cid, league_id=lg, conn=conn)
                except Exception as e:
                    logger.debug("card_confidence(%s, %s) failed: %s", cid, lg, e)
                    continue

                score = int(result.get("score") or 0)
                label = result.get("label")
                role = result.get("role")
                drivers = result.get("drivers") or []
                agg = result.get("aggregate") or {}
                sample = 0
                n_inst = 0
                if role == "batting":
                    bat = agg.get("batting") or {}
                    sample = int(bat.get("pa") or 0)
                    n_inst = int(bat.get("n_instances") or 0)
                elif role == "pitching":
                    pit = agg.get("pitching") or {}
                    sample = int((pit.get("ip") or 0) * 10)
                    n_inst = int(pit.get("n_instances") or 0)

                cursor.execute("""
                    INSERT INTO meta_confidence
                        (card_id, league_id, score, label, role,
                         drivers_json, sample_size, n_instances)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (cid, lg, score, label, role,
                      json.dumps(drivers), sample, n_inst))
                rows_written += 1

        conn.commit()
        result_summary = {
            "table": "meta_confidence",
            "rows_written": rows_written,
            "leagues": leagues,
        }
        logger.info("derived_stats: %s", result_summary)
        return result_summary
    finally:
        if owns_conn:
            conn.close()


# ──────────────────────────────────────────────────────────────────────
# Table 9: card_archetypes — k-means clustering of rating profiles
# ──────────────────────────────────────────────────────────────────────

def build_card_archetypes(conn: Optional[sqlite3.Connection] = None) -> dict:
    """Cluster cards by rating profile and persist archetype labels.

    The 2026-04-20 attribute-mix analysis showed that individual ratings are
    poorer predictors than how ratings COMBINE — a card with three 80s often
    out-produces a card with one 95 and two 60s. Clusters expose those mix
    patterns as named archetypes ("Power+Discipline", "Command Fireman") with
    an empirical WAR/600 or WAR/200 attached.

    For each card we persist:
      - archetype_id (0..K-1 within role)
      - archetype_name (human-readable label)
      - role ('batting' | 'SP' | 'RP')
      - fit_score 0..100 (how close to centroid; 100=perfect match)
      - archetype_war (cluster's sample-weighted mean WAR rate)
      - n_in_archetype (cluster size — sanity check)
      - mix_score, min_top3, count_elite (diagnostic mix columns)

    This table enables the replacement-finder to match "find me another
    Command Fireman under 5,000 PP" instead of "find me a meta >= X" —
    role-fit over monolithic score.

    Implementation: k-means with scikit-learn on z-scored ratings. Fit each
    role independently (8 batting / 6 SP / 6 RP clusters). Cluster names
    are derived from the top 2 positive deviations from population mean.
    """
    import json
    import numpy as np

    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()
    try:
        cursor = conn.cursor()

        # Lazy-import sklearn so the base import path stays fast; only
        # derived-stats rebuilds pay the startup cost.
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            logger.warning("sklearn unavailable — skipping archetype build: %s", e)
            return {"table": "card_archetypes", "rows_written": 0, "error": str(e)}

        # Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS card_archetypes (
                card_id INTEGER,
                role TEXT,
                archetype_id INTEGER,
                archetype_name TEXT,
                fit_score REAL,
                archetype_war REAL,
                n_in_archetype INTEGER,
                mix_score REAL,
                min_top3 REAL,
                count_elite INTEGER,
                centroid_json TEXT,
                PRIMARY KEY (card_id, role)
            )
        """)
        cursor.execute("DELETE FROM card_archetypes")

        rows_written = 0

        def _cluster_and_store(rows, rating_names, role_label, K):
            """Fit k-means on `rows`, label and persist to card_archetypes."""
            nonlocal rows_written
            if len(rows) < K * 3:
                logger.info(
                    "archetypes[%s]: n=%d too small for k=%d — skipping",
                    role_label, len(rows), K,
                )
                return

            X = np.array([[float(r[k] or 0) for k in rating_names] for r in rows], dtype=float)
            # WAR rate per row — stored so we can compute cluster-avg WAR.
            # batting uses WAR/600, pitching WAR/200 (stored via caller).
            wars = np.array([r['_war_rate'] for r in rows])
            ws = np.array([r['_sample_weight'] for r in rows])

            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X)
            km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Xs)
            labels = km.labels_

            # Name clusters by top 2 positive deviations from pop mean.
            pop_mean = X.mean(axis=0)
            names = []
            for cid in range(K):
                mask = labels == cid
                if not mask.any():
                    names.append(f"Cluster {cid}")
                    continue
                profile = X[mask].mean(axis=0)
                dev = profile - pop_mean
                top_idx = np.argsort(dev)[-2:][::-1]
                # Abbreviated rating name shortcuts for readability
                abbrev = {
                    'gap_power': 'gap', 'avoid_ks': 'avoid-K', 'baserunning': 'brn',
                    'p_hr': 'HR-supp', 'p_babip': 'BABIP-supp', 'stealing': 'steal',
                }
                parts = []
                for i in top_idx:
                    n = rating_names[i]
                    short = abbrev.get(n, n).title()
                    parts.append(f"{short}({int(profile[i])})")
                names.append(" + ".join(parts))

            # Cluster-level aggregates
            cluster_war = {}
            cluster_n = {}
            for cid in range(K):
                mask = labels == cid
                if mask.any():
                    w_sum = ws[mask].sum()
                    cluster_war[cid] = float((wars[mask] * ws[mask]).sum() / w_sum) if w_sum > 0 else 0.0
                    cluster_n[cid] = int(mask.sum())
                else:
                    cluster_war[cid] = 0.0; cluster_n[cid] = 0

            # Fit score: 100 * (1 - dist_to_centroid / max_dist_in_cluster)
            # Use Xs (scaled) so all features contribute equally.
            dist = np.linalg.norm(Xs - km.cluster_centers_[labels], axis=1)
            # Normalize per-cluster so fit_score is comparable within role
            fit_scores = np.zeros(len(rows))
            for cid in range(K):
                mask = labels == cid
                if mask.any():
                    d = dist[mask]
                    mx = max(d.max(), 1e-6)
                    fit_scores[mask] = 100.0 * (1.0 - d / mx)

            for i, r in enumerate(rows):
                cid = int(labels[i])
                profile = {name: int(X[i][j]) for j, name in enumerate(rating_names)}
                sorted_vals = sorted(X[i], reverse=True)
                min_top3 = float(sorted_vals[2]) if len(sorted_vals) >= 3 else float(sorted_vals[-1])
                count_elite = int(sum(1 for v in X[i] if v >= 80))
                # mix_score matches calc_mix_diagnostic formula
                mix_balance = max(0.0, min(60.0, (min_top3 - 50) / 45.0 * 60.0))
                mix_ceiling = max(0.0, min(40.0, count_elite / 6.0 * 40.0))
                mix_score = round(mix_balance + mix_ceiling, 1)

                cursor.execute("""
                    INSERT OR REPLACE INTO card_archetypes
                        (card_id, role, archetype_id, archetype_name,
                         fit_score, archetype_war, n_in_archetype,
                         mix_score, min_top3, count_elite, centroid_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r['card_id'], role_label, cid, names[cid],
                    round(float(fit_scores[i]), 1),
                    round(cluster_war[cid], 3),
                    cluster_n[cid],
                    mix_score, round(min_top3, 1), count_elite,
                    json.dumps(profile),
                ))
                rows_written += 1

        # ── BATTING — pull every batter with a full rating profile ──
        bat_names = ['contact', 'gap_power', 'power', 'eye', 'avoid_ks',
                     'babip', 'speed', 'baserunning']
        # Join with batting_stats to get WAR rate. Cards w/o stats get rate=0
        # and sample_weight=1 so they still get clustered on ratings alone.
        batters = cursor.execute(f"""
            SELECT c.card_id, c.contact, c.gap_power, c.power, c.eye,
                   c.avoid_ks, c.babip, c.speed, c.baserunning,
                   COALESCE(
                       (SELECT SUM(b.war)*1.0 / NULLIF(SUM(b.pa)/600.0, 0)
                        FROM batting_stats b WHERE b.card_id=c.card_id AND b.pa IS NOT NULL),
                       0
                   ) AS _war_rate,
                   COALESCE(
                       (SELECT SUM(b.pa)
                        FROM batting_stats b WHERE b.card_id=c.card_id AND b.pa>=50),
                       1
                   ) AS _sample_weight
            FROM cards c
            WHERE c.position IS NOT NULL AND c.pitcher_role IS NULL
              AND c.contact IS NOT NULL AND c.power IS NOT NULL
        """).fetchall()
        batters = [dict(b) for b in batters]
        _cluster_and_store(batters, bat_names, 'batting', K=8)

        # ── PITCHERS — split SP from RP ──
        pit_names_sp = ['stuff', 'movement', 'control', 'p_hr', 'p_babip', 'stamina']
        pit_names_rp = ['stuff', 'movement', 'control', 'p_hr', 'p_babip', 'hold']

        def _load_pitchers(role_filter_sql):
            return [dict(r) for r in cursor.execute(f"""
                SELECT c.card_id, c.stuff, c.movement, c.control, c.p_hr,
                       c.p_babip, c.stamina, c.hold,
                       COALESCE(
                           (SELECT SUM(p.war)*1.0 / NULLIF(SUM(p.ip)/200.0, 0)
                            FROM pitching_stats p WHERE p.card_id=c.card_id
                              AND p.ip IS NOT NULL),
                           0
                       ) AS _war_rate,
                       COALESCE(
                           (SELECT SUM(p.ip)
                            FROM pitching_stats p WHERE p.card_id=c.card_id
                              AND p.ip>=10),
                           1
                       ) AS _sample_weight
                FROM cards c
                WHERE c.pitcher_role IS NOT NULL
                  AND c.stuff IS NOT NULL AND c.control IS NOT NULL
                  AND {role_filter_sql}
            """).fetchall()]

        sps = _load_pitchers("UPPER(COALESCE(c.pitcher_role_name,'')) LIKE 'SP%'")
        _cluster_and_store(sps, pit_names_sp, 'SP', K=6)

        rps = _load_pitchers("UPPER(COALESCE(c.pitcher_role_name,'')) IN ('RP','CL','SU','MR','LR','SW','CP')")
        _cluster_and_store(rps, pit_names_rp, 'RP', K=6)

        # Helpful index for role-based lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_archetypes_role ON card_archetypes(role, archetype_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_archetypes_name ON card_archetypes(archetype_name)")
        conn.commit()

        result_summary = {
            "table": "card_archetypes",
            "rows_written": rows_written,
            "batters": len(batters),
            "sps": len(sps),
            "rps": len(rps),
        }
        logger.info("derived_stats: %s", result_summary)
        return result_summary
    finally:
        if owns_conn:
            conn.close()


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

def build_all(conn: Optional[sqlite3.Connection] = None) -> dict:
    """Rebuild every derived table. Single connection for efficiency.

    Returns per-table summary so the worker can log or surface counts.
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()
    try:
        out: dict[str, dict] = {}
        out["batter_vs_pitcher_hand"] = build_batter_vs_pitcher_hand(conn=conn)
        out["pitcher_fatigue"] = build_pitcher_fatigue(conn=conn)
        out["clutch_profile_card"] = build_clutch_profile_card(conn=conn)
        out["park_adjusted_stats"] = build_park_adjusted_stats(conn=conn)
        out["regression_candidates_v2"] = build_regression_candidates_v2(conn=conn)
        out["opponent_scouting"] = build_opponent_scouting(conn=conn)
        out["price_velocity"] = build_price_velocity(conn=conn)
        out["meta_confidence"] = build_meta_confidence(conn=conn)
        # 2026-04-20: k-means archetype clustering for role-fit matching
        out["card_archetypes"] = build_card_archetypes(conn=conn)
        return out
    finally:
        if owns_conn:
            conn.close()
