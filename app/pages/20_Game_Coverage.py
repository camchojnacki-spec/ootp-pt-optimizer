"""Game Coverage — calendar view of which game-dates have data ingested.

Answers "did I miss exporting any games?" with these lenses:

1. **Heatmap calendar** — one cell per in-universe game date, colored by
   status (TDK + league sample / TDK only / league only / partial / empty).
2. **Per-team coverage** — games + batting/pitching row counts + last-seen
   per team, flags under-sampled teams (median-based threshold).
3. **Per-date detail table** — completeness flags for the five game-related
   tables (batting, pitching, narrative, clutch, PBP).
4. **Date drill-down** — pick a date, see the full game list with TDK role,
   scores, completeness, recap headline.
5. **Data quality** — card-id resolution rate (name → card_id mapping that
   feeds the meta engine) plus source-file integrity (HTML still on disk?).
6. **Missing exports** — orphan box scores on disk, PBP catch-up (with
   actionable vs unrecoverable split), TDK-missing date stretches with
   confidence scoring (🔴 High = bracketed by TDK games), and date gaps.
7. **Export checklist** — downloadable CSV of every actionable miss.

Date coordinate is the in-universe `game_date` (the date the OOTP sim
played the game), not real-world wall-clock — that's what matters for
roster/meta analysis.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection, load_config
from app.utils.sidebar_nav import render_sidebar_nav


# ── Page setup ────────────────────────────────────────────────────────
st.set_page_config(page_title="Game Coverage", page_icon="🗓️", layout="wide")
render_sidebar_nav()

config = load_config()
conn = get_connection()


# ── Palette ──────────────────────────────────────────────────────────
COLOR_BOTH      = "#00c853"   # TDK + league sample, full
COLOR_TDK       = "#4ba3ff"   # TDK only, full
COLOR_LEAGUE    = "#ff9800"   # League sample only — TDK missing (probable miss)
COLOR_PARTIAL   = "#ff4b4b"   # Some sub-tables missing
COLOR_EMPTY     = "#2a2a2a"   # Date in range with no games at all (gap)
COLOR_OUT       = "rgba(0,0,0,0)"  # Outside range / blank cell

STATUS_LABELS = {
    "both":    "TDK + league sample",
    "tdk":     "TDK only",
    "league":  "League sample (no TDK)",
    "partial": "Partial ingest",
    "empty":   "No data",
}
STATUS_COLORS = {
    "both":    COLOR_BOTH,
    "tdk":     COLOR_TDK,
    "league":  COLOR_LEAGUE,
    "partial": COLOR_PARTIAL,
    "empty":   COLOR_EMPTY,
}


# ── Data loaders ─────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_coverage(league_id: str | None) -> pd.DataFrame:
    """One row per game_date with per-table completeness counts."""
    where = "WHERE g.game_date IS NOT NULL"
    params: tuple = ()
    if league_id and league_id != "all":
        where += " AND g.league_id = ?"
        params = (league_id,)

    # EXISTS subqueries (not LEFT JOINs) — joining 5 child tables would
    # multiply rows and explode the TDK sum. Each EXISTS is an index lookup.
    q = f"""
        SELECT
            g.game_date AS d,
            COUNT(*) AS games,
            SUM(CASE WHEN g.toronto_role IS NOT NULL THEN 1 ELSE 0 END) AS tdk_games,
            SUM(CASE WHEN EXISTS (SELECT 1 FROM game_batting       WHERE game_id = g.game_id) THEN 1 ELSE 0 END) AS w_batting,
            SUM(CASE WHEN EXISTS (SELECT 1 FROM game_pitching      WHERE game_id = g.game_id) THEN 1 ELSE 0 END) AS w_pitching,
            SUM(CASE WHEN EXISTS (SELECT 1 FROM game_narratives    WHERE game_id = g.game_id) THEN 1 ELSE 0 END) AS w_narrative,
            SUM(CASE WHEN EXISTS (SELECT 1 FROM game_clutch_events WHERE game_id = g.game_id) THEN 1 ELSE 0 END) AS w_clutch,
            SUM(CASE WHEN EXISTS (SELECT 1 FROM game_log_at_bats   WHERE game_id = g.game_id) THEN 1 ELSE 0 END) AS w_atbats
        FROM games g
        {where}
        GROUP BY g.game_date
        ORDER BY g.game_date
    """
    rows = conn.execute(q, params).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["d"] = pd.to_datetime(df["d"]).dt.date
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_leagues() -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT league_id FROM games "
        "WHERE league_id IS NOT NULL ORDER BY league_id"
    ).fetchall()
    return [r["league_id"] for r in rows]


def classify(row: pd.Series) -> str:
    """Decide the status bucket for a date row."""
    if row["games"] == 0:
        return "empty"
    full = (
        row["w_batting"] == row["games"]
        and row["w_pitching"] == row["games"]
        and row["w_narrative"] == row["games"]
        and row["w_clutch"] == row["games"]
        and row["w_atbats"] == row["games"]
    )
    if not full:
        return "partial"
    has_tdk = row["tdk_games"] > 0
    has_league = (row["games"] - row["tdk_games"]) > 0
    if has_tdk and has_league:
        return "both"
    if has_tdk:
        return "tdk"
    return "league"


def build_calendar_grid(df: pd.DataFrame, year: int, month: int) -> go.Figure:
    """Render one month as a 7-col Sun-Sat grid using a plotly heatmap.

    Cells are colored by status. Day numbers are drawn as annotations.
    Hovertext shows the date + status + game counts.
    """
    # Map status -> z value (numeric) for the heatmap
    status_to_z = {"out": 0, "empty": 1, "league": 2, "partial": 3, "tdk": 4, "both": 5}
    colorscale = [
        [0.00, COLOR_OUT],
        [0.16, COLOR_OUT],
        [0.17, COLOR_EMPTY],
        [0.33, COLOR_EMPTY],
        [0.34, COLOR_LEAGUE],
        [0.50, COLOR_LEAGUE],
        [0.51, COLOR_PARTIAL],
        [0.66, COLOR_PARTIAL],
        [0.67, COLOR_TDK],
        [0.83, COLOR_TDK],
        [0.84, COLOR_BOTH],
        [1.00, COLOR_BOTH],
    ]

    # Build the 6-week × 7-day grid
    first = date(year, month, 1)
    # Sunday-start: Python date.weekday() is Mon=0..Sun=6; we want Sun=0..Sat=6
    start_offset = (first.weekday() + 1) % 7
    last_day = (date(year + (month // 12), (month % 12) + 1, 1) - timedelta(days=1)).day

    grid_z = [[0] * 7 for _ in range(6)]
    grid_text = [[""] * 7 for _ in range(6)]
    grid_hover = [[""] * 7 for _ in range(6)]

    by_date = {r["d"]: r for _, r in df.iterrows()} if not df.empty else {}

    for day in range(1, last_day + 1):
        d = date(year, month, day)
        idx = start_offset + (day - 1)
        row, col = divmod(idx, 7)
        if row >= 6:
            break
        if d in by_date:
            r = by_date[d]
            status = classify(r)
            grid_z[row][col] = status_to_z[status]
            grid_text[row][col] = str(day)
            tdk_g = int(r["tdk_games"])
            league_g = int(r["games"]) - tdk_g
            grid_hover[row][col] = (
                f"<b>{d.isoformat()}</b><br>"
                f"Status: {STATUS_LABELS[status]}<br>"
                f"TDK games: {tdk_g}<br>"
                f"League games: {league_g}<br>"
                f"Batting {int(r['w_batting'])}/{int(r['games'])} · "
                f"Pitching {int(r['w_pitching'])}/{int(r['games'])}<br>"
                f"Recap {int(r['w_narrative'])}/{int(r['games'])} · "
                f"Clutch {int(r['w_clutch'])}/{int(r['games'])} · "
                f"PBP {int(r['w_atbats'])}/{int(r['games'])}"
            )
        else:
            grid_z[row][col] = status_to_z["empty"]
            grid_text[row][col] = str(day)
            grid_hover[row][col] = f"<b>{d.isoformat()}</b><br>No games"

    # Reverse rows so week 1 is at top
    grid_z = grid_z[::-1]
    grid_text = grid_text[::-1]
    grid_hover = grid_hover[::-1]

    fig = go.Figure(
        data=go.Heatmap(
            z=grid_z,
            text=grid_text,
            customdata=grid_hover,
            hovertemplate="%{customdata}<extra></extra>",
            colorscale=colorscale,
            zmin=0,
            zmax=5,
            showscale=False,
            xgap=3,
            ygap=3,
        )
    )
    # Day numbers
    annotations = []
    for r_idx, row in enumerate(grid_text):
        for c_idx, txt in enumerate(row):
            if txt:
                annotations.append(
                    dict(
                        x=c_idx, y=r_idx, text=txt,
                        showarrow=False,
                        font=dict(size=11, color="rgba(255,255,255,0.85)"),
                    )
                )
    fig.update_layout(
        annotations=annotations,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(7)),
            ticktext=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
            showgrid=False, zeroline=False, side="top",
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        title=dict(text=first.strftime("%B %Y"), x=0.5, font=dict(size=16)),
        height=260,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def find_orphan_box_scores(games_df: pd.DataFrame) -> tuple[str | None, list[str]]:
    """Return (save_dir, list_of_orphan_html_paths) — files on disk not in DB.

    Derives save_dir from any source_file row in games. If multiple distinct
    save_dirs exist, picks the most recent one by ingested_at.
    """
    row = conn.execute(
        "SELECT source_file FROM games "
        "WHERE source_file IS NOT NULL AND source_file LIKE '%box_scores%' "
        "ORDER BY ingested_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None, []
    sample = row["source_file"]
    box_dir = os.path.dirname(sample)
    if not os.path.isdir(box_dir):
        return box_dir, []

    on_disk = {
        os.path.join(box_dir, n)
        for n in os.listdir(box_dir)
        if n.lower().endswith(".html")
    }
    in_db = {
        r["source_file"]
        for r in conn.execute(
            "SELECT DISTINCT source_file FROM games WHERE source_file LIKE ?",
            (os.path.join(box_dir, "%"),),
        ).fetchall()
        if r["source_file"]
    }
    orphans = sorted(on_disk - in_db)
    return box_dir, orphans


# ── Header ───────────────────────────────────────────────────────────
st.title("🗓️ Game Coverage")
st.caption(
    "Calendar view of every game-date in the database. "
    "Spots TDK exports you missed and league-sample gaps that would weaken meta analysis."
)

leagues = load_leagues()
if not leagues:
    st.info("No games ingested yet. Run **Data Refresh** to import box scores.")
    st.stop()

cols = st.columns([2, 2, 2, 2, 2])
with cols[0]:
    league_choice = st.selectbox(
        "League", ["all"] + leagues,
        index=(["all"] + leagues).index(config.get("active_league", "all"))
        if config.get("active_league") in leagues else 0,
    )
with cols[1]:
    show_partial = st.toggle("Highlight partial ingests", value=True,
                             help="Treat dates where some sub-tables are missing as red instead of green.")

df = load_coverage(None if league_choice == "all" else league_choice)

if df.empty:
    st.warning(f"No games for league `{league_choice}`.")
    st.stop()

# When the toggle is off, fold partial → status from TDK/league presence
if not show_partial:
    def _classify_no_partial(row):
        if row["games"] == 0:
            return "empty"
        has_tdk = row["tdk_games"] > 0
        has_league = (row["games"] - row["tdk_games"]) > 0
        if has_tdk and has_league:
            return "both"
        if has_tdk:
            return "tdk"
        return "league"
    df["status"] = df.apply(_classify_no_partial, axis=1)
    # Patch classify() for the calendar via monkey-patch alternative:
    # we re-classify in build_calendar_grid via the row data, so keep the
    # toggle only affecting the table — the calendar always reflects truth.
    # (Hiding partial in the calendar would lie about the data.)
else:
    df["status"] = df.apply(classify, axis=1)


# ── KPI row ──────────────────────────────────────────────────────────
total_games = int(df["games"].sum())
tdk_games = int(df["tdk_games"].sum())
league_games = total_games - tdk_games
covered_dates = len(df)
date_min = df["d"].min()
date_max = df["d"].max()
span_days = (date_max - date_min).days + 1
gap_days = span_days - covered_dates

# Days since last TDK game (within the in-universe timeline)
tdk_dates = df[df["tdk_games"] > 0]["d"]
last_tdk = tdk_dates.max() if len(tdk_dates) else None
gap_since_tdk = (date_max - last_tdk).days if last_tdk else None

partial_dates = int((df["status"] == "partial").sum())
league_only_dates = int((df["status"] == "league").sum())

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Game-dates covered", f"{covered_dates}", f"{gap_days} gaps in span")
k2.metric("TDK games", f"{tdk_games}")
k3.metric("League sample", f"{league_games}")
k4.metric("Span", f"{date_min} → {date_max}", f"{span_days} days")
k5.metric(
    "Partial ingests",
    f"{partial_dates}",
    "dates with missing sub-tables" if partial_dates else "all complete",
    delta_color=("inverse" if partial_dates else "off"),
)
k6.metric(
    "Last TDK game",
    str(last_tdk) if last_tdk else "—",
    f"{gap_since_tdk} days ago" if gap_since_tdk is not None else "—",
    delta_color=("inverse" if (gap_since_tdk or 0) > 2 else "off"),
)

# Split TDK-missing dates into "mid-season miss" vs "post-elimination".
# Heuristic: if TDK has 0 games on a date AFTER its last appearance, it's
# postseason absence (TDK eliminated, OOTP didn't generate TDK files).
# If it's BEFORE the last TDK game date, TDK was active that week and the
# missing data is a real export gap worth investigating.
#
# NOTE: condition is tdk_games == 0 (not status == "league") because a
# league-only date that's also missing some sub-tables would be tagged
# "partial" and would get hidden from the alert otherwise.
tdk_missing_mask = df["tdk_games"] == 0
if last_tdk is not None:
    midseason_misses = df[tdk_missing_mask & (df["d"] <= last_tdk)].copy()
    postseason_only  = df[tdk_missing_mask & (df["d"] >  last_tdk)].copy()
else:
    midseason_misses = df[tdk_missing_mask].copy()
    postseason_only  = df.iloc[0:0].copy()

# Confidence scoring for mid-season misses — if TDK played both the day
# before AND the day after, the miss is high-confidence (TDK was active
# all week). If TDK was already idle adjacent days, it might just be a
# scheduled bye, not an export miss.
tdk_active_dates = set(df[df["tdk_games"] > 0]["d"].tolist())

def _miss_confidence(d: date) -> str:
    before = (d - timedelta(days=1)) in tdk_active_dates
    after  = (d + timedelta(days=1)) in tdk_active_dates
    if before and after:
        return "🔴 High"      # bracketed by TDK games — definitely a miss
    if before or after:
        return "🟡 Medium"    # adjacent to a TDK game — likely a miss
    return "🟢 Low"           # TDK already idle nearby — possibly a bye

if not midseason_misses.empty:
    midseason_misses["confidence"] = midseason_misses["d"].apply(_miss_confidence)
    midseason_misses = midseason_misses.sort_values("d", ascending=False)
n_high_mid = int((midseason_misses["confidence"] == "🔴 High").sum()) if not midseason_misses.empty else 0

n_mid = len(midseason_misses)
n_post = len(postseason_only)

if n_high_mid:
    st.warning(
        f"⚠️ **{n_high_mid} high-confidence missed TDK export(s)** "
        f"(bracketed by TDK games — TDK was active that week). "
        f"{n_mid - n_high_mid} more mid-season date(s) are medium/low confidence. "
        f"{('Plus ' + str(n_post) + ' post-elimination dates (informational, not misses).') if n_post else ''}"
    )
elif n_mid:
    st.info(
        f"ℹ️ {n_mid} mid-season date(s) had league games but no TDK — "
        "all are low-confidence (TDK was already idle nearby), so likely byes, not misses."
    )
elif n_post:
    st.info(
        f"ℹ️ {n_post} dates after {last_tdk} have league games but no TDK — "
        "TDK was eliminated, so OOTP produced no TDK files for those days. Not a miss."
    )


# ── Calendar grid ────────────────────────────────────────────────────
st.markdown("### Calendar")

# Legend with density counts
status_counts = df["status"].value_counts().to_dict()
legend_html = " &nbsp;&nbsp; ".join(
    f"<span style='display:inline-block;width:12px;height:12px;background:{c};"
    f"border-radius:2px;vertical-align:middle;'></span> "
    f"{STATUS_LABELS[k]} <b>({status_counts.get(k, 0)})</b>"
    for k, c in STATUS_COLORS.items()
)
st.markdown(legend_html, unsafe_allow_html=True)
st.caption("In-universe game dates. Click a calendar cell to read the hover detail.")

# Build month list spanning the data range
months: list[tuple[int, int]] = []
y, m = date_min.year, date_min.month
while (y, m) <= (date_max.year, date_max.month):
    months.append((y, m))
    m += 1
    if m > 12:
        m = 1
        y += 1

# Show 3 months per row
for i in range(0, len(months), 3):
    chunk = months[i : i + 3]
    cs = st.columns(len(chunk))
    for col, (yr, mo) in zip(cs, chunk):
        with col:
            st.plotly_chart(
                build_calendar_grid(df, yr, mo),
                config={"displayModeBar": False},
                use_container_width=True,
            )


# ── Per-team coverage ────────────────────────────────────────────────
st.markdown("### Per-team coverage")
st.caption(
    "How many games per team are in the database, and how recent. "
    "Under-sampled teams weaken cross-card learning in the meta engine — "
    "their cards get less residual signal."
)

team_where = ""
team_params: tuple = ()
if league_choice != "all":
    team_where = "WHERE g.league_id = ?"
    team_params = (league_choice,)

# Combine home + away counts so each team gets credited per game it played
team_q = f"""
    WITH appearances AS (
        SELECT g.home_team AS team, g.game_id, g.game_date FROM games g {team_where}
        UNION ALL
        SELECT g.away_team AS team, g.game_id, g.game_date FROM games g {team_where}
    )
    SELECT
        a.team,
        COUNT(DISTINCT a.game_id)        AS games,
        MIN(a.game_date)                 AS first_seen,
        MAX(a.game_date)                 AS last_seen,
        COUNT(DISTINCT gb.id)            AS batting_rows,
        COUNT(DISTINCT gp.id)            AS pitching_rows
    FROM appearances a
    LEFT JOIN game_batting  gb ON gb.team_name = a.team AND gb.game_id = a.game_id
    LEFT JOIN game_pitching gp ON gp.team_name = a.team AND gp.game_id = a.game_id
    WHERE a.team IS NOT NULL
    GROUP BY a.team
    ORDER BY games DESC
"""
# Same params bound twice (once per CTE branch)
team_rows = conn.execute(team_q, team_params * 2).fetchall()
team_df = pd.DataFrame([dict(r) for r in team_rows])

if team_df.empty:
    st.info("No team data yet.")
else:
    median_games = team_df["games"].median()
    under_thr = max(5, int(median_games * 0.4))
    team_df["status"] = team_df["games"].apply(
        lambda g: "🟢 Strong" if g >= median_games
        else ("🟡 OK" if g >= under_thr else "🔴 Under-sampled")
    )
    today_anchor = df["d"].max() if not df.empty else None
    if today_anchor is not None:
        team_df["days_since_last"] = team_df["last_seen"].apply(
            lambda s: (today_anchor - pd.to_datetime(s).date()).days if s else None
        )
    else:
        team_df["days_since_last"] = None

    display_cols = pd.DataFrame({
        "Team":          team_df["team"],
        "Status":        team_df["status"],
        "Games":         team_df["games"].astype(int),
        "Batting rows":  team_df["batting_rows"].astype(int),
        "Pitching rows": team_df["pitching_rows"].astype(int),
        "First seen":    team_df["first_seen"].astype(str),
        "Last seen":     team_df["last_seen"].astype(str),
        "Days since":    team_df["days_since_last"],
    })
    st.dataframe(display_cols, hide_index=True, use_container_width=True)

    n_under = int((team_df["status"] == "🔴 Under-sampled").sum())
    n_total = len(team_df)
    if n_under:
        st.caption(
            f"{n_under} of {n_total} teams are under-sampled (<{under_thr} games). "
            "Their card-level meta residuals will be noisier."
        )

# TDK home/away balance + opponent distribution
ha_where = ""
ha_params: tuple = ()
if league_choice != "all":
    ha_where = "AND league_id = ?"
    ha_params = (league_choice,)

ha_rows = conn.execute(
    f"SELECT toronto_role, COUNT(*) c FROM games "
    f"WHERE toronto_role IS NOT NULL {ha_where} GROUP BY toronto_role",
    ha_params,
).fetchall()
home_n = next((r["c"] for r in ha_rows if r["toronto_role"] == "home"), 0)
away_n = next((r["c"] for r in ha_rows if r["toronto_role"] == "away"), 0)

if home_n + away_n > 0:
    st.markdown("**TDK home/away balance**")
    imbalance = abs(home_n - away_n)
    total_tdk = home_n + away_n
    home_pct = home_n / total_tdk * 100
    away_pct = away_n / total_tdk * 100
    ha_cols = st.columns([1, 1, 2])
    ha_cols[0].metric("Home games", home_n, f"{home_pct:.0f}%")
    ha_cols[1].metric("Away games", away_n, f"{away_pct:.0f}%")
    # Real schedules are roughly 50/50; >5-game imbalance is suspicious
    if imbalance > 5:
        side = "away" if away_n > home_n else "home"
        opposite = "home" if side == "away" else "away"
        ha_cols[2].warning(
            f"⚠️ {imbalance}-game imbalance ({side} > {opposite}). "
            f"Could be schedule asymmetry, but worth checking that you're "
            f"exporting {opposite} games as diligently as {side} ones."
        )
    else:
        ha_cols[2].success(f"Balanced (within {imbalance} games of 50/50). ✅")

    # Opponent distribution
    opp_rows = conn.execute(f"""
        SELECT
            CASE WHEN toronto_role = 'home' THEN away_team ELSE home_team END AS opp,
            SUM(CASE WHEN toronto_role = 'home' THEN 1 ELSE 0 END) AS home_vs,
            SUM(CASE WHEN toronto_role = 'away' THEN 1 ELSE 0 END) AS away_vs,
            COUNT(*) AS total
        FROM games WHERE toronto_role IS NOT NULL {ha_where}
        GROUP BY opp ORDER BY total DESC
    """, ha_params).fetchall()
    if opp_rows:
        opp_df = pd.DataFrame([dict(r) for r in opp_rows])
        opp_df["balance"] = opp_df.apply(
            lambda r: "✅" if abs(r["home_vs"] - r["away_vs"]) <= 2
            else f"⚠️ {abs(r['home_vs'] - r['away_vs'])}-game gap",
            axis=1,
        )
        with st.expander(f"TDK opponent distribution ({len(opp_df)} teams faced)", expanded=False):
            st.caption(
                "How often TDK faced each opponent, split by venue. Big home/away "
                "gaps against a single opponent can mean missed exports for that series."
            )
            st.dataframe(
                pd.DataFrame({
                    "Opponent": opp_df["opp"],
                    "Total":    opp_df["total"].astype(int),
                    "Home vs":  opp_df["home_vs"].astype(int),
                    "Away vs":  opp_df["away_vs"].astype(int),
                    "Balance":  opp_df["balance"],
                }),
                hide_index=True, use_container_width=True,
            )


# ── Data quality ─────────────────────────────────────────────────────
st.markdown("### Data quality")
st.caption(
    "Resolution rates for the name → card_id mapping that powers the meta engine, "
    "plus integrity checks on the source HTML files we ingested from."
)

dq_cols = st.columns(2)

with dq_cols[0]:
    st.markdown("**Card-id resolution rate**")
    st.caption("Per game-table, per league. Below 95% means the resolver is "
               "missing players and meta-residual signal will leak.")
    dq_where = ""
    dq_params: tuple = ()
    if league_choice != "all":
        dq_where = "AND g.league_id = ?"
        dq_params = (league_choice,)

    rows: list[dict] = []
    for tbl, label in [
        ("game_batting",       "Batting"),
        ("game_pitching",      "Pitching"),
        ("game_clutch_events", "Clutch events"),
    ]:
        for r in conn.execute(f"""
            SELECT g.league_id, COUNT(*) AS total,
                   SUM(CASE WHEN t.card_id IS NOT NULL THEN 1 ELSE 0 END) AS resolved
            FROM {tbl} t JOIN games g ON g.game_id = t.game_id
            WHERE g.league_id IS NOT NULL {dq_where}
            GROUP BY g.league_id
        """, dq_params):
            total = r["total"] or 0
            resolved = r["resolved"] or 0
            rate = (resolved / total * 100) if total else 0.0
            rows.append({
                "Table": label,
                "League": r["league_id"],
                "Resolved": f"{resolved:,}",
                "Total": f"{total:,}",
                "Rate": f"{rate:.1f}%",
                "_rate": rate,
            })

    if rows:
        dq_df = pd.DataFrame(rows)
        st.dataframe(
            dq_df.drop(columns=["_rate"]),
            hide_index=True, use_container_width=True,
        )
        worst = min(r["_rate"] for r in rows)
        if worst < 95:
            st.warning(
                f"Worst resolver rate is {worst:.1f}% — "
                "check `name_resolver` and `league_rosters` for the affected league/table."
            )
    else:
        st.info("No game tables to evaluate yet.")

with dq_cols[1]:
    st.markdown("**Source-file integrity**")
    st.caption("Every game's source HTML — does it still exist on disk? "
               "Moved/deleted files mean re-ingest is impossible without finding them.")
    src_rows = conn.execute(
        "SELECT source_file FROM games "
        "WHERE source_file IS NOT NULL "
        + ("AND league_id = ?" if league_choice != "all" else "")
        + " GROUP BY source_file",
        (league_choice,) if league_choice != "all" else (),
    ).fetchall()
    total_src = len(src_rows)
    if total_src == 0:
        st.info("No source-file paths recorded.")
    else:
        missing_src = [r["source_file"] for r in src_rows if not os.path.exists(r["source_file"])]
        present = total_src - len(missing_src)
        rate = present / total_src * 100
        st.metric(
            "Files still on disk",
            f"{present}/{total_src}",
            f"{rate:.1f}%",
            delta_color=("off" if rate == 100 else "inverse"),
        )
        if missing_src:
            st.warning(f"{len(missing_src)} source HTML file(s) are no longer on disk.")
            with st.expander("Show missing paths", expanded=False):
                st.code("\n".join(missing_src[:50]))
                if len(missing_src) > 50:
                    st.caption(f"… and {len(missing_src) - 50} more")
        else:
            st.success("All source HTML files still on disk. ✅")


# ── Detail table ─────────────────────────────────────────────────────
st.markdown("### Per-date detail")

month_labels = [f"{date(y, m, 1):%b %Y}" for (y, m) in months]
default_idx = len(months) - 1
sel_label = st.selectbox(
    "Month",
    month_labels + ["All months"],
    index=default_idx,
)
if sel_label == "All months":
    df_view = df.copy()
else:
    sel_idx = month_labels.index(sel_label)
    sy, sm = months[sel_idx]
    df_view = df[df["d"].apply(lambda x: x.year == sy and x.month == sm)].copy()

if df_view.empty:
    st.info("No games in the selected month.")
else:
    def _flag(have, total):
        if total == 0:
            return "—"
        return "✓" if have == total else f"{have}/{total}"

    table = pd.DataFrame({
        "Date":     df_view["d"].astype(str),
        "Day":      [d.strftime("%a") for d in df_view["d"]],
        "Status":   df_view["status"].map(STATUS_LABELS),
        "TDK":      df_view["tdk_games"].astype(int),
        "League":   (df_view["games"] - df_view["tdk_games"]).astype(int),
        "Total":    df_view["games"].astype(int),
        "Batting":  [_flag(h, t) for h, t in zip(df_view["w_batting"], df_view["games"])],
        "Pitching": [_flag(h, t) for h, t in zip(df_view["w_pitching"], df_view["games"])],
        "Recap":    [_flag(h, t) for h, t in zip(df_view["w_narrative"], df_view["games"])],
        "Clutch":   [_flag(h, t) for h, t in zip(df_view["w_clutch"], df_view["games"])],
        "PBP":      [_flag(h, t) for h, t in zip(df_view["w_atbats"], df_view["games"])],
    }).sort_values("Date", ascending=False)

    st.dataframe(table, hide_index=True, use_container_width=True)


# ── Date drill-down ──────────────────────────────────────────────────
st.markdown("### Date drill-down")
st.caption("Pick any date with games to see the full game list and per-game completeness.")

covered_dates_list = sorted(df["d"].tolist(), reverse=True)
default_drill = last_tdk if last_tdk in covered_dates_list else covered_dates_list[0]
drill_date = st.selectbox(
    "Date",
    covered_dates_list,
    index=covered_dates_list.index(default_drill),
    format_func=lambda d: f"{d.isoformat()} ({d.strftime('%a')})",
)

drill_where = "g.game_date = ?"
drill_params: tuple = (drill_date.isoformat(),)
if league_choice != "all":
    drill_where += " AND g.league_id = ?"
    drill_params = drill_params + (league_choice,)

drill_q = f"""
    SELECT
        g.game_id,
        g.league_id,
        g.home_team,
        g.away_team,
        g.home_score,
        g.away_score,
        g.winner_team,
        g.toronto_role,
        EXISTS (SELECT 1 FROM game_batting       WHERE game_id = g.game_id) AS has_batting,
        EXISTS (SELECT 1 FROM game_pitching      WHERE game_id = g.game_id) AS has_pitching,
        EXISTS (SELECT 1 FROM game_narratives    WHERE game_id = g.game_id) AS has_narrative,
        EXISTS (SELECT 1 FROM game_clutch_events WHERE game_id = g.game_id) AS has_clutch,
        EXISTS (SELECT 1 FROM game_log_at_bats   WHERE game_id = g.game_id) AS has_atbats,
        gn.recap_headline,
        gn.ballpark
    FROM games g
    LEFT JOIN game_narratives gn ON gn.game_id = g.game_id
    WHERE {drill_where}
    ORDER BY g.game_id
"""
drill_rows = conn.execute(drill_q, drill_params).fetchall()
drill_df = pd.DataFrame([dict(r) for r in drill_rows])

if drill_df.empty:
    st.info(f"No games on {drill_date}.")
else:
    def _matchup(row):
        h, a = row["home_team"], row["away_team"]
        hs, as_ = row["home_score"], row["away_score"]
        if hs is None or as_ is None:
            return f"{a} @ {h}"
        return f"{a} {as_} @ {h} {hs}"

    def _tdk_tag(role):
        if role == "home": return "🏠 TDK home"
        if role == "away": return "✈️ TDK away"
        return "—"

    def _check(b):
        return "✓" if b else "✗"

    drill_table = pd.DataFrame({
        "Game ID":   drill_df["game_id"].astype(int),
        "League":    drill_df["league_id"].fillna("?"),
        "Matchup":   drill_df.apply(_matchup, axis=1),
        "TDK":       drill_df["toronto_role"].apply(_tdk_tag),
        "Winner":    drill_df["winner_team"].fillna("—"),
        "Bat":       drill_df["has_batting"].apply(_check),
        "Pit":       drill_df["has_pitching"].apply(_check),
        "Recap":     drill_df["has_narrative"].apply(_check),
        "Clutch":    drill_df["has_clutch"].apply(_check),
        "PBP":       drill_df["has_atbats"].apply(_check),
        "Headline":  drill_df["recap_headline"].fillna(""),
    })
    st.dataframe(drill_table, hide_index=True, use_container_width=True)

    incomplete = drill_df[
        ~(drill_df["has_batting"].astype(bool)
          & drill_df["has_pitching"].astype(bool)
          & drill_df["has_narrative"].astype(bool)
          & drill_df["has_clutch"].astype(bool)
          & drill_df["has_atbats"].astype(bool))
    ]
    if not incomplete.empty:
        missing_pbp = incomplete[~incomplete["has_atbats"].astype(bool)]
        if not missing_pbp.empty:
            st.warning(
                f"⚠️ {len(missing_pbp)} game(s) on this date are missing play-by-play data. "
                f"Look for `log_<game_id>.html` files in `news/html/game_logs/` for game IDs: "
                f"{', '.join(str(i) for i in missing_pbp['game_id'].head(10))}"
                f"{'…' if len(missing_pbp) > 10 else ''}"
            )


# ── Missing exports panel ────────────────────────────────────────────
st.markdown("### Missing exports")

with st.expander("Box-score HTML files on disk vs ingested", expanded=False):
    box_dir, orphans = find_orphan_box_scores(df)
    if not box_dir:
        st.info("No source-file paths recorded yet — ingest at least one box score first.")
    else:
        st.caption(f"Scan path: `{box_dir}`")
        if not os.path.isdir(box_dir):
            st.error(
                f"That directory doesn't exist on this machine. "
                f"OOTP save folder may have moved — update `save_game_dir` in config.yaml."
            )
        elif not orphans:
            st.success(f"All HTML files in `{box_dir}` have been ingested. ✅")
        else:
            st.warning(
                f"**{len(orphans)} HTML file(s)** on disk have not been ingested. "
                "Run **Data Refresh** to pull them in."
            )
            sample_n = min(50, len(orphans))
            st.code("\n".join(os.path.basename(p) for p in orphans[:sample_n]))
            if len(orphans) > sample_n:
                st.caption(f"… and {len(orphans) - sample_n} more")

# Missing PBP — diff game_log_at_bats coverage against game_logs/log_<id>.html on disk
pbp_missing_q = """
    SELECT g.game_id, g.game_date, g.home_team, g.away_team, g.toronto_role,
           g.source_file
    FROM games g
    WHERE NOT EXISTS (SELECT 1 FROM game_log_at_bats WHERE game_id = g.game_id)
    """ + ("AND g.league_id = ?" if league_choice != "all" else "") + """
    ORDER BY g.game_date DESC, g.game_id
"""
pbp_missing_rows = conn.execute(
    pbp_missing_q,
    (league_choice,) if league_choice != "all" else (),
).fetchall()
pbp_actionable: list[int] = []
pbp_unrecoverable: list[int] = []
gl_dir = None
if pbp_missing_rows:
    # Derive game_logs dir as sibling of box_scores dir
    sample_src = next((r["source_file"] for r in pbp_missing_rows if r["source_file"]), None)
    if sample_src is None:
        # Fall back to any source_file in games
        any_src = conn.execute(
            "SELECT source_file FROM games WHERE source_file LIKE '%box_scores%' LIMIT 1"
        ).fetchone()
        sample_src = any_src["source_file"] if any_src else None
    if sample_src:
        box_dir_local = os.path.dirname(sample_src)
        gl_dir = os.path.join(os.path.dirname(box_dir_local), "game_logs")
    if gl_dir and os.path.isdir(gl_dir):
        for r in pbp_missing_rows:
            log_path = os.path.join(gl_dir, f"log_{r['game_id']}.html")
            (pbp_actionable if os.path.exists(log_path) else pbp_unrecoverable).append(r["game_id"])
    else:
        # Treat all as unrecoverable since we can't see the dir
        pbp_unrecoverable = [r["game_id"] for r in pbp_missing_rows]

with st.expander(
    f"Play-by-play catch-up ({len(pbp_missing_rows)} games missing PBP)",
    expanded=bool(pbp_missing_rows),
):
    if not pbp_missing_rows:
        st.success("Every ingested game has play-by-play data. ✅")
    else:
        st.caption(
            "Games whose box score was ingested but no `log_<game_id>.html` rows landed "
            "in `game_log_at_bats`. PBP enables exit-velocity, LD%, and true K% overlays — "
            "missing it leaves those overlays blank for that game."
        )
        if gl_dir:
            st.caption(f"Game-log scan path: `{gl_dir}`")
        c1, c2 = st.columns(2)
        c1.metric("Actionable (log file on disk)", len(pbp_actionable))
        c2.metric("Unrecoverable (no log file)", len(pbp_unrecoverable))

        if pbp_actionable:
            st.warning(
                f"**{len(pbp_actionable)} game(s)** can be ingested right now — "
                "the log files exist on disk. Run **Data Refresh** to pick them up."
            )
            st.code("\n".join(f"log_{gid}.html" for gid in pbp_actionable[:20]))
            if len(pbp_actionable) > 20:
                st.caption(f"… and {len(pbp_actionable) - 20} more")
        if pbp_unrecoverable:
            st.info(
                f"**{len(pbp_unrecoverable)} game(s)** have no matching log file on disk — "
                "OOTP didn't generate PBP for these (likely sims with PBP disabled). "
                "Re-running ingest won't help; the data simply wasn't produced. "
                "Material impact on analysis is small (~"
                f"{len(pbp_unrecoverable) / max(1, len(df)) * 100:.1f}% of game-dates affected)."
            )
            with st.expander("Show unrecoverable game IDs", expanded=False):
                rows_view = pd.DataFrame([
                    {"Game ID": r["game_id"], "Date": r["game_date"],
                     "Matchup": f"{r['away_team']} @ {r['home_team']}",
                     "TDK": r["toronto_role"] or "—"}
                    for r in pbp_missing_rows if r["game_id"] in pbp_unrecoverable
                ])
                st.dataframe(rows_view, hide_index=True, use_container_width=True)

        # Mystery logs: log files on disk that aren't in game_log_at_bats AND aren't
        # in pbp_actionable above. These are logs we have but never tried to ingest.
        if gl_dir and os.path.isdir(gl_dir):
            try:
                ids_on_disk = set()
                for n in os.listdir(gl_dir):
                    if n.lower().startswith("log_") and n.lower().endswith(".html"):
                        try:
                            ids_on_disk.add(int(n[4:-5]))
                        except ValueError:
                            pass
                ids_with_pbp = {
                    r["game_id"] for r in conn.execute(
                        "SELECT DISTINCT game_id FROM game_log_at_bats"
                    ).fetchall()
                }
                mystery = sorted(ids_on_disk - ids_with_pbp - set(pbp_actionable))
                if mystery:
                    st.caption(
                        f"Plus {len(mystery)} log file(s) on disk for games not in our "
                        "`games` table — these are orphaned PBP logs (game itself was "
                        "never box-score-ingested)."
                    )
            except OSError:
                pass


with st.expander(
    f"TDK-missing date stretches ({n_mid} mid-season misses, {n_post} post-elimination)",
    expanded=bool(n_high_mid),
):
    if n_mid == 0 and n_post == 0:
        st.success("Every date with games has at least one TDK game. ✅")
    else:
        if n_mid:
            st.markdown("**Mid-season — likely missed exports**")
            st.caption(
                "Confidence reflects whether TDK was active around this date. "
                "🔴 High = TDK played the day before AND after (definitely a miss). "
                "🟡 Medium = TDK played one adjacent day. "
                "🟢 Low = TDK already idle nearby (possibly a scheduled bye, not a miss)."
            )
            st.dataframe(
                pd.DataFrame({
                    "Date":       midseason_misses["d"].astype(str),
                    "Day":        [d.strftime("%a") for d in midseason_misses["d"]],
                    "Confidence": midseason_misses["confidence"],
                    "League games": midseason_misses["games"].astype(int),
                }),
                hide_index=True, use_container_width=True,
            )
        if n_post:
            st.markdown("**Post-elimination — informational**")
            st.caption(
                f"After {last_tdk} (last TDK game), the league continued into the "
                "playoffs without TDK. OOTP doesn't generate files for teams that "
                "aren't playing, so these aren't misses."
            )
            st.dataframe(
                pd.DataFrame({
                    "Date":   postseason_only["d"].astype(str),
                    "Day":    [d.strftime("%a") for d in postseason_only["d"]],
                    "League games that day": postseason_only["games"].astype(int),
                }).sort_values("Date", ascending=False),
                hide_index=True, use_container_width=True,
            )

with st.expander("Date gaps within the covered span", expanded=False):
    have = set(df["d"])
    cur = date_min
    gaps: list[date] = []
    while cur <= date_max:
        if cur not in have:
            gaps.append(cur)
        cur += timedelta(days=1)
    if not gaps:
        st.success("No gap days — every calendar date in span has at least one game. ✅")
    else:
        # Group into runs
        runs: list[tuple[date, date, int]] = []
        run_start = gaps[0]
        prev = gaps[0]
        for g in gaps[1:]:
            if (g - prev).days == 1:
                prev = g
                continue
            runs.append((run_start, prev, (prev - run_start).days + 1))
            run_start = g
            prev = g
        runs.append((run_start, prev, (prev - run_start).days + 1))

        st.dataframe(
            pd.DataFrame({
                "Start":  [str(s) for s, _, _ in runs],
                "End":    [str(e) for _, e, _ in runs],
                "Days":   [n for _, _, n in runs],
            }).sort_values("Days", ascending=False),
            hide_index=True, use_container_width=True,
        )
        st.caption(
            f"{len(gaps)} total gap days across {len(runs)} runs. "
            "Long runs at the start of the season are usually pre-launch; "
            "mid-season runs are likely missed exports."
        )


# ── Export checklist ─────────────────────────────────────────────────
st.markdown("### Export checklist")
st.caption(
    "Download a CSV bundle of every actionable miss for offline reference. "
    "Useful for tracking down lost OOTP exports between sessions."
)

checklist_rows: list[dict] = []
for _, row in midseason_misses.iterrows():
    checklist_rows.append({
        "kind": "missing_tdk_box_score",
        "date": row["d"].isoformat(),
        "confidence": row["confidence"],
        "league_games_that_day": int(row["games"]),
        "game_id": "",
        "matchup": "",
        "note": "TDK played adjacent days — likely a missed export"
                if row["confidence"] == "🔴 High"
                else "TDK was active in the area — possible miss",
    })
for gid in pbp_actionable:
    matching = next((r for r in pbp_missing_rows if r["game_id"] == gid), None)
    checklist_rows.append({
        "kind": "actionable_missing_pbp",
        "date": matching["game_date"] if matching else "",
        "confidence": "🔴 High",
        "league_games_that_day": "",
        "game_id": gid,
        "matchup": f"{matching['away_team']} @ {matching['home_team']}" if matching else "",
        "note": f"log_{gid}.html exists on disk — re-run Data Refresh",
    })
# Box-score orphans (recompute lightly here so we don't depend on expander state)
_, orphan_files = find_orphan_box_scores(df)
for p in orphan_files:
    checklist_rows.append({
        "kind": "orphan_box_score",
        "date": "",
        "confidence": "🔴 High",
        "league_games_that_day": "",
        "game_id": "",
        "matchup": "",
        "note": f"HTML on disk not yet ingested: {os.path.basename(p)}",
    })

if checklist_rows:
    chk_df = pd.DataFrame(checklist_rows)
    st.dataframe(chk_df, hide_index=True, use_container_width=True)
    st.download_button(
        "📥 Download checklist as CSV",
        data=chk_df.to_csv(index=False).encode("utf-8"),
        file_name=f"game_coverage_checklist_{date_max.isoformat()}.csv",
        mime="text/csv",
    )
else:
    st.success("Nothing actionable — coverage is clean. ✅")


conn.close()
