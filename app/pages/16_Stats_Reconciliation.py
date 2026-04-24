"""Stats Reconciliation — does the data we see match reality?

Surfaces three trust signals so the user knows what to believe:

1. **Team total vs roster-sum delta** — for each ingested snapshot, the team
   total (from ``league_team_stats``) compared against the sum of individual
   player stats on the current roster. Systematic gaps indicate departed
   players (players no longer on the team whose stats still count in the
   team total).

2. **Roster presence map** — which active roster players appear in the
   latest league-wide stats, and which don't (bench / not-yet-played).

3. **Data freshness** — latest snapshot per table and ingestion log
   summary so the user can see which CSVs were last imported and when.

This page doesn't modify any data — it's a read-only audit view.
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.core.database import get_connection, load_config
from app.utils.sidebar_nav import render_sidebar_nav

st.set_page_config(page_title="Stats Reconciliation", page_icon="🧮", layout="wide")
render_sidebar_nav()
st.title("🧮 Stats Reconciliation")
st.caption(
    "Does the data match reality? Cross-checks team totals against the sum "
    "of roster player stats, flags departed players, and shows data freshness."
)

conn = get_connection()
config = load_config()
active_league = config.get('active_league', 'lb124')

# ── 1. Freshness panel ──
st.markdown("## 🗓️ Data Freshness")
fresh_cols = st.columns(4)
TABLES = [
    ('batting_stats', 'Batting Stats'),
    ('pitching_stats', 'Pitching Stats'),
    ('batting_stats_adv', 'Batting Adv (wOBA)'),
    ('pitching_stats_adv', 'Pitching Adv (SIERA)'),
    ('league_team_stats', 'League Team Stats'),
    ('fielding_stats', 'Fielding Stats'),
    ('pitch_ratings', 'Pitch Ratings'),
    ('roster', 'Roster'),
]
for i, (tbl, label) in enumerate(TABLES):
    try:
        row = conn.execute(
            f"SELECT COUNT(*) n, MAX(snapshot_date) d FROM {tbl}"
        ).fetchone()
        with fresh_cols[i % 4]:
            st.metric(label, f"{row['n']:,}", delta=str(row['d'] or 'never'), delta_color="off")
    except Exception as e:
        with fresh_cols[i % 4]:
            st.metric(label, "ERR", delta=str(e)[:30], delta_color="off")

# ── 2. Team total vs roster sum ──
st.markdown("## ⚖️ Team Totals vs Roster Sums")
st.caption(
    "Toronto's team-level totals (from `league_team_stats`) compared against "
    "the sum of individual player stats for players currently on the roster. "
    "A positive diff means the team total is higher — usually departed players "
    "whose stats still count for the team."
)
team = conn.execute(
    """SELECT * FROM league_team_stats
       WHERE league_id = ? AND team_name LIKE '%Toronto%'
       ORDER BY snapshot_date DESC LIMIT 1""",
    (active_league,),
).fetchone()

if not team:
    st.info("No team-level stats ingested yet. Run a refresh with the "
            "`team_statistics_*_batting_stats` / `pitching_stats` CSVs to "
            "enable this panel.")
else:
    # Active roster player names
    roster_names = [r['player_name'] for r in conn.execute(
        """SELECT DISTINCT player_name FROM roster_current
           WHERE lineup_role IN ('starter','rotation','closer','bullpen')"""
    ).fetchall()]

    latest_bat = conn.execute(
        "SELECT MAX(snapshot_date) FROM batting_stats WHERE league_id = ?",
        (active_league,),
    ).fetchone()[0]
    latest_pit = conn.execute(
        "SELECT MAX(snapshot_date) FROM pitching_stats WHERE league_id = ?",
        (active_league,),
    ).fetchone()[0]

    if roster_names and latest_bat:
        placeholders = ','.join('?' * len(roster_names))
        bat_sum = conn.execute(
            f"""SELECT COUNT(*) n, SUM(pa) pa, SUM(ab) ab, SUM(hits) hits,
                       SUM(hr) hr, SUM(runs) runs, SUM(bb) bb, SUM(k) k,
                       SUM(doubles) d2, SUM(triples) d3, SUM(sb) sb
                FROM batting_stats
                WHERE league_id = ? AND snapshot_date = ?
                  AND player_name IN ({placeholders})""",
            [active_league, latest_bat] + roster_names,
        ).fetchone()
        pit_sum = conn.execute(
            f"""SELECT COUNT(*) n, SUM(ip) ip, SUM(k) k, SUM(bb) bb,
                       SUM(hr_allowed) hra, SUM(hits_allowed) ha,
                       SUM(saves) sv, SUM(wins) w, SUM(losses) l
                FROM pitching_stats
                WHERE league_id = ? AND snapshot_date = ?
                  AND player_name IN ({placeholders})""",
            [active_league, latest_pit] + roster_names,
        ).fetchone()

        reco_rows = [
            ('HR',     team['hr'],       bat_sum['hr']),
            ('Runs',   team['runs'],     bat_sum['runs']),
            ('BB',     team['bb'],       bat_sum['bb']),
            ('SO',     team['so'],       bat_sum['k']),
            ('2B',     team['doubles'],  bat_sum['d2']),
            ('3B',     team['triples'],  bat_sum['d3']),
            ('SB',     team['sb'],       bat_sum['sb']),
            ('Wins',   team['wins'],     pit_sum['w']),
            ('Losses', team['losses'],   pit_sum['l']),
            ('Saves',  team['saves'],    pit_sum['sv']),
            ('Pit-K',  team['k'],        pit_sum['k']),
        ]
        df = pd.DataFrame(
            [{'Stat': s, 'Team Total': tt or 0, 'Roster Sum': rs or 0,
              'Diff': (tt or 0) - (rs or 0),
              'Diff %': round(((tt or 0) - (rs or 0)) / (tt or 1) * 100, 1)}
             for s, tt, rs in reco_rows]
        )
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Team Total": st.column_config.NumberColumn(format="%d"),
                "Roster Sum": st.column_config.NumberColumn(format="%d"),
                "Diff": st.column_config.NumberColumn(format="%+d"),
                "Diff %": st.column_config.NumberColumn(format="%+.1f%%"),
            },
        )

        avg_diff_pct = df['Diff %'].abs().mean()
        if avg_diff_pct < 10:
            st.success(
                f"✅ Reconciliation OK — average delta {avg_diff_pct:.1f}% across "
                f"{len(df)} stats. Gaps within the 5–10% band are typically "
                f"departed players still counted in the team total."
            )
        else:
            st.warning(
                f"⚠️ Average delta {avg_diff_pct:.1f}% — above the expected "
                f"5–10% departed-players band. Investigate: it could mean "
                f"stale data, duplicate-name card_id conflation across PT "
                f"teams, or a missing team-stats import."
            )
    else:
        st.info("Roster or stats table empty — import roster + stats CSVs first.")

# ── 3. Roster presence map ──
st.markdown("## 👥 Roster Presence in League Stats")
st.caption(
    "Which players on your active roster appear in the most recent league-wide "
    "batting/pitching stats file? Missing players are usually bench cards that "
    "haven't appeared in a game yet, not a data error."
)
roster_rows = conn.execute(
    """SELECT player_name, position, lineup_role, meta_score, card_id, ovr
       FROM roster_current
       WHERE lineup_role IN ('starter','rotation','closer','bullpen','bench')
       ORDER BY lineup_role, position, player_name"""
).fetchall()

presence_data = []
for r in roster_rows:
    is_pit = (r['position'] or '') in ('SP', 'RP', 'CL')
    if is_pit:
        hit = conn.execute(
            """SELECT ip FROM pitching_stats
               WHERE league_id = ? AND player_name = ?
               ORDER BY snapshot_date DESC LIMIT 1""",
            (active_league, r['player_name'])
        ).fetchone()
        sample = f"{hit['ip']} IP" if hit and hit['ip'] is not None else "—"
    else:
        hit = conn.execute(
            """SELECT pa FROM batting_stats
               WHERE league_id = ? AND player_name = ?
               ORDER BY snapshot_date DESC LIMIT 1""",
            (active_league, r['player_name'])
        ).fetchone()
        sample = f"{hit['pa']} PA" if hit and hit['pa'] is not None else "—"
    presence_data.append({
        'Role': r['lineup_role'],
        'Pos': r['position'],
        'Player': r['player_name'],
        'Meta': round(r['meta_score'], 0) if r['meta_score'] else 0,
        'OVR': r['ovr'],
        'In Stats?': '✅' if hit else '❌',
        'Sample': sample,
    })

pres_df = pd.DataFrame(presence_data)
# Split active vs bench
active_df = pres_df[pres_df['Role'] != 'bench']
bench_df = pres_df[pres_df['Role'] == 'bench']

pc1, pc2 = st.columns([2, 1])
with pc1:
    st.markdown(f"**Active Roster ({len(active_df)} players)**")
    st.dataframe(active_df, use_container_width=True, hide_index=True)
with pc2:
    matched = int((active_df['In Stats?'] == '✅').sum())
    st.metric("Active players in stats", f"{matched}/{len(active_df)}")
    st.metric("Bench (not expected in stats)", len(bench_df))

if bench_df.shape[0] > 0:
    with st.expander(f"Bench / collection ({len(bench_df)} cards)"):
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

# ── 4. Departed-player drill-down ──
st.markdown("## 🚶 Possibly Departed Players")
st.caption(
    "Players who appear in league-wide stats with a Toronto-flavored presence "
    "(e.g., were on prior roster snapshots) but aren't on the current roster. "
    "These account for most of the team-total vs roster-sum gap."
)
try:
    # Names that appeared in prior roster snapshots but not current
    cur_names = set(r['player_name'] for r in roster_rows)
    prior_names = set(r[0] for r in conn.execute(
        """SELECT DISTINCT player_name FROM roster
           WHERE DATE(snapshot_date) < (SELECT MAX(DATE(snapshot_date)) FROM roster)
             AND lineup_role IN ('starter','rotation','closer','bullpen','bench','reserve')"""
    ).fetchall() if r[0])
    departed = prior_names - cur_names
    if departed:
        departed_rows = []
        for name in sorted(departed):
            last_seen = conn.execute(
                """SELECT MAX(snapshot_date) d FROM roster
                   WHERE player_name = ? AND lineup_role != 'league'""",
                (name,)
            ).fetchone()
            stat = conn.execute(
                """SELECT pa, hr FROM batting_stats
                   WHERE league_id = ? AND player_name = ?
                   ORDER BY snapshot_date DESC LIMIT 1""",
                (active_league, name)
            ).fetchone()
            if stat:
                departed_rows.append({
                    'Player': name,
                    'Last on roster': last_seen['d'] if last_seen else '?',
                    'Latest PA': stat['pa'] or 0,
                    'Latest HR': stat['hr'] or 0,
                })
        if departed_rows:
            st.dataframe(pd.DataFrame(departed_rows),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No departed players identifiable from current history.")
    else:
        st.info("No players in prior roster snapshots who aren't on current roster.")
except Exception as e:
    st.info(f"Departed-player analysis unavailable: {e}")

# ── 5. Duplicate card_id warning ──
st.markdown("## ⚠️ Known Data Issues")
dup_rows = conn.execute(
    """SELECT player_name, COUNT(*) n
       FROM batting_stats
       WHERE league_id = ? AND snapshot_date = (
           SELECT MAX(snapshot_date) FROM batting_stats WHERE league_id = ?
       )
       GROUP BY player_name
       HAVING COUNT(*) > 1
       ORDER BY COUNT(*) DESC""",
    (active_league, active_league),
).fetchall()
if dup_rows:
    st.warning(
        f"**{len(dup_rows)}** player names appear on multiple teams within "
        f"**{active_league}** (same card on different PT team rosters). "
        f"OOTP's CSV lists them separately; our card_id matcher collapses them to "
        f"one card, which is correct for card-level joins but means summing by "
        f"card_id over-counts for any of these names."
    )
    dup_df = pd.DataFrame([{'Player': r['player_name'], 'Rows': r['n']} for r in dup_rows])
    with st.expander(f"See {len(dup_rows)} ambiguous names"):
        st.dataframe(dup_df, use_container_width=True, hide_index=True)
else:
    st.success("No duplicate-name cards detected in the latest league stats snapshot.")

conn.close()
