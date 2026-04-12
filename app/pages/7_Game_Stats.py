"""Game Stats — Statcast-inspired performance analytics dashboard."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection

st.set_page_config(page_title="Game Stats", page_icon="📊", layout="wide")

conn = get_connection()

# ── Palette ──
FIRE = "#ff4b4b"
ICE = "#4ba3ff"
GOLD = "#ffc107"
GREEN = "#00c853"
PURPLE = "#a855f7"
GREY = "#555"
BG = "rgba(0,0,0,0)"
GRID = "rgba(255,255,255,0.06)"


def _df(query, params=()):
    rows = conn.execute(query, params).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def _latest(table):
    row = conn.execute(f"SELECT MAX(snapshot_date) as d FROM {table}").fetchone()
    return row["d"] if row else None


def _pctile_color(val, reverse=False):
    """Return color based on percentile (0-100). Red=bad, Blue=avg, Green=good."""
    if reverse:
        val = 100 - val
    if val >= 90: return "#ff1744"  # elite red
    if val >= 75: return "#ff6d00"  # great orange
    if val >= 50: return GOLD       # above avg
    if val >= 25: return ICE        # below avg
    return GREY                     # poor


# ── Load data ──
latest_bat = _latest("batting_stats")
latest_pit = _latest("pitching_stats")
has_batting = latest_bat is not None
has_pitching = latest_pit is not None

MIN_PA = 50
MIN_IP = 20

if not has_batting and not has_pitching:
    st.title("📊 Game Stats")
    st.info("No batting or pitching stats in the database yet. Import game stats to populate this dashboard.")
    conn.close()
    st.stop()

# Load full datasets
bat_df = pd.DataFrame()
pit_df = pd.DataFrame()

if has_batting:
    bat_df = _df(f"""
        SELECT b.player_name, b.position, b.pa, b.ab, b.hits, b.doubles, b.triples,
               b.hr, b.rbi, b.runs, b.bb, b.ibb, b.hbp, b.k, b.sb, b.cs,
               b.avg, b.obp, b.slg, b.ops, b.iso, b.babip, b.war, b.ops_plus,
               c.meta_score_batting as meta, c.contact, c.power, c.eye, c.speed
        FROM batting_stats b
        LEFT JOIN cards c ON c.card_title LIKE '%' || b.player_name || '%' AND c.owned = 1
        WHERE b.snapshot_date = ? AND b.ab > 0 AND b.pa >= {MIN_PA}
        GROUP BY b.player_name
        ORDER BY b.war DESC
    """, (latest_bat,))

    if not bat_df.empty:
        bat_df["K%"] = (bat_df["k"] / bat_df["pa"] * 100).round(1)
        bat_df["BB%"] = (bat_df["bb"] / bat_df["pa"] * 100).round(1)
        bat_df["WAR/600"] = (bat_df["war"] * 600 / bat_df["pa"]).round(1)
        singles = bat_df["hits"] - bat_df["doubles"] - bat_df["triples"] - bat_df["hr"]
        woba_num = (0.69 * (bat_df["bb"] - bat_df["ibb"]) + 0.72 * bat_df["hbp"]
                    + 0.88 * singles + 1.24 * bat_df["doubles"]
                    + 1.56 * bat_df["triples"] + 2.01 * bat_df["hr"])
        woba_den = bat_df["ab"] + bat_df["bb"] - bat_df["ibb"] + bat_df["hbp"]
        bat_df["wOBA"] = (woba_num / woba_den.replace(0, 1)).round(3)

if has_pitching:
    pit_df = _df(f"""
        SELECT p.player_name, p.position, p.games, p.gs, p.ip, p.era, p.whip, p.fip,
               p.k, p.bb, p.hbp, p.hr_allowed, p.hits_allowed,
               p.k_per_9, p.bb_per_9, p.hr_per_9, p.k_per_bb, p.war,
               p.saves, p.holds, p.wins, p.losses,
               c.meta_score_pitching as meta, c.stuff, c.movement, c.control
        FROM pitching_stats p
        LEFT JOIN cards c ON c.card_title LIKE '%' || p.player_name || '%' AND c.owned = 1
        WHERE p.snapshot_date = ? AND p.ip >= {MIN_IP} AND (p.k > 0 OR p.era > 0)
        GROUP BY p.player_name
        ORDER BY p.war DESC
    """, (latest_pit,))

    if not pit_df.empty:
        pit_df["BF"] = (pit_df["ip"] * 3 + pit_df["hits_allowed"] + pit_df["bb"] + pit_df["hbp"]).astype(int).replace(0, 1)
        pit_df["K%"] = (pit_df["k"] / pit_df["BF"] * 100).round(1)
        pit_df["BB%"] = (pit_df["bb"] / pit_df["BF"] * 100).round(1)
        pit_df["K-BB%"] = (pit_df["K%"] - pit_df["BB%"]).round(1)
        pit_df["WAR/200"] = (pit_df["war"] * 200 / pit_df["ip"]).round(1)
        pit_df["ERA-FIP"] = (pit_df["era"] - pit_df["fip"]).round(2)
        pit_df["Role"] = pit_df.apply(
            lambda r: "SP" if r["gs"] > 0 else ("CL" if r["saves"] > 0 else "RP"), axis=1)


# ════════════════════════════════════════════════════════════════════════════════
# HEADER — Team vitals
# ════════════════════════════════════════════════════════════════════════════════
st.title("📊 Game Stats")

c1, c2, c3, c4, c5, c6 = st.columns(6)
if not bat_df.empty:
    c1.metric("Team AVG", f"{bat_df['avg'].mean():.3f}")
    c2.metric("Team OPS", f"{bat_df['ops'].mean():.3f}")
    c3.metric("Team HR", int(bat_df["hr"].sum()))
if not pit_df.empty:
    sp_df = pit_df[pit_df["gs"] > 0]
    c4.metric("Team ERA", f"{pit_df['era'].mean():.2f}")
    c5.metric("Team WHIP", f"{pit_df['whip'].mean():.2f}")
    c6.metric("SP ERA", f"{sp_df['era'].mean():.2f}" if not sp_df.empty else "N/A")

st.caption(f"Latest snapshot: {latest_bat or latest_pit}")

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tab_overview, tab_batting, tab_pitching, tab_player, tab_meta = st.tabs([
    "🏟️ Overview", "⚾ Batting", "🎯 Pitching", "🔍 Player Card", "🧪 Meta vs Reality"
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW — hero charts
# ════════════════════════════════════════════════════════════════════════════════
with tab_overview:
    if not bat_df.empty:
        st.markdown("### Offense: OPS vs WAR/600")
        st.caption("Bigger bubble = more plate appearances. Top-right = elite producer.")

        fig = go.Figure()

        # Quadrant lines
        ops_med = bat_df["ops"].median()
        war_med = bat_df["WAR/600"].median()

        fig.add_hline(y=war_med, line_dash="dot", line_color=GREY, opacity=0.4)
        fig.add_vline(x=ops_med, line_dash="dot", line_color=GREY, opacity=0.4)

        # Quadrant labels
        fig.add_annotation(x=ops_med + 0.08, y=war_med + 2.5, text="⭐ Stars",
                          font=dict(color=GOLD, size=11), showarrow=False)
        fig.add_annotation(x=ops_med - 0.08, y=war_med + 2.5, text="🔮 Hidden Gems",
                          font=dict(color=GREEN, size=11), showarrow=False)
        fig.add_annotation(x=ops_med + 0.08, y=war_med - 2.0, text="📉 Declining",
                          font=dict(color=FIRE, size=11), showarrow=False)
        fig.add_annotation(x=ops_med - 0.08, y=war_med - 2.0, text="⚠️ Struggling",
                          font=dict(color=GREY, size=11), showarrow=False)

        fig.add_trace(go.Scatter(
            x=bat_df["ops"], y=bat_df["WAR/600"],
            mode="markers+text",
            text=bat_df["player_name"].apply(lambda n: n.split()[-1]),
            textposition="top center",
            textfont=dict(size=9, color="rgba(255,255,255,0.7)"),
            marker=dict(
                size=bat_df["pa"] / bat_df["pa"].max() * 30 + 8,
                color=bat_df["WAR/600"],
                colorscale=[[0, ICE], [0.5, GOLD], [1, FIRE]],
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
            ),
            hovertemplate="<b>%{customdata[0]}</b><br>OPS: %{x:.3f}<br>WAR/600: %{y:.1f}<br>PA: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack([bat_df["player_name"], bat_df["pa"]]),
        ))

        fig.update_layout(
            height=450, xaxis_title="OPS", yaxis_title="WAR/600",
            plot_bgcolor=BG, paper_bgcolor=BG,
            xaxis=dict(gridcolor=GRID, zeroline=False),
            yaxis=dict(gridcolor=GRID, zeroline=False),
            margin=dict(t=20, b=40, l=50, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    if not pit_df.empty:
        col_luck, col_dom = st.columns(2)

        with col_luck:
            st.markdown("### Pitching: Luck Chart (ERA vs FIP)")
            st.caption("Above the line = lucky (ERA < FIP). Below = unlucky. Size = innings.")

            fig2 = go.Figure()

            # Perfect luck line
            era_range = [pit_df["fip"].min() - 0.5, pit_df["fip"].max() + 0.5]
            fig2.add_trace(go.Scatter(
                x=era_range, y=era_range, mode="lines",
                line=dict(color=GREY, dash="dash", width=1),
                showlegend=False, hoverinfo="skip",
            ))

            colors = [GREEN if row["ERA-FIP"] > 0.3 else (FIRE if row["ERA-FIP"] < -0.3 else ICE)
                      for _, row in pit_df.iterrows()]

            fig2.add_trace(go.Scatter(
                x=pit_df["fip"], y=pit_df["era"],
                mode="markers+text",
                text=pit_df["player_name"].apply(lambda n: n.split()[-1]),
                textposition="top center",
                textfont=dict(size=9, color="rgba(255,255,255,0.6)"),
                marker=dict(
                    size=pit_df["ip"] / pit_df["ip"].max() * 25 + 6,
                    color=colors,
                    line=dict(width=1, color="rgba(255,255,255,0.3)"),
                ),
                hovertemplate="<b>%{customdata[0]}</b><br>ERA: %{y:.2f}<br>FIP: %{x:.2f}<br>Gap: %{customdata[1]}<br>IP: %{customdata[2]:.0f}<extra></extra>",
                customdata=np.column_stack([pit_df["player_name"], pit_df["ERA-FIP"], pit_df["ip"]]),
            ))

            fig2.update_layout(
                height=380, xaxis_title="FIP (skill)", yaxis_title="ERA (results)",
                plot_bgcolor=BG, paper_bgcolor=BG,
                xaxis=dict(gridcolor=GRID, zeroline=False),
                yaxis=dict(gridcolor=GRID, zeroline=False),
                margin=dict(t=20, b=40, l=50, r=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_dom:
            st.markdown("### Pitching: Dominance (K% vs BB%)")
            st.caption("Top-left = elite command + stuff. Bottom-right = wild and hittable.")

            fig3 = go.Figure()

            kbb_med = pit_df["K-BB%"].median()
            fig3.add_trace(go.Scatter(
                x=pit_df["BB%"], y=pit_df["K%"],
                mode="markers+text",
                text=pit_df["player_name"].apply(lambda n: n.split()[-1]),
                textposition="top center",
                textfont=dict(size=9, color="rgba(255,255,255,0.6)"),
                marker=dict(
                    size=pit_df["ip"] / pit_df["ip"].max() * 25 + 6,
                    color=pit_df["K-BB%"],
                    colorscale=[[0, GREY], [0.5, ICE], [1, FIRE]],
                    line=dict(width=1, color="rgba(255,255,255,0.3)"),
                ),
                hovertemplate="<b>%{customdata[0]}</b><br>K%: %{y:.1f}<br>BB%: %{x:.1f}<br>K-BB%: %{customdata[1]:.1f}<extra></extra>",
                customdata=np.column_stack([pit_df["player_name"], pit_df["K-BB%"]]),
            ))

            fig3.update_layout(
                height=380, xaxis_title="BB% (lower = better)", yaxis_title="K% (higher = better)",
                plot_bgcolor=BG, paper_bgcolor=BG,
                xaxis=dict(gridcolor=GRID, zeroline=False, autorange="reversed"),
                yaxis=dict(gridcolor=GRID, zeroline=False),
                margin=dict(t=20, b=40, l=50, r=20),
            )
            st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: BATTING — leaderboards + advanced
# ════════════════════════════════════════════════════════════════════════════════
with tab_batting:
    if bat_df.empty:
        st.info("No qualifying batters yet.")
    else:
        st.markdown("### Batting Leaderboard")

        # Sortable main leaderboard
        display_bat = bat_df[["player_name", "position", "pa", "avg", "obp", "slg",
                              "ops", "wOBA", "iso", "hr", "K%", "BB%", "WAR/600", "war", "babip"]].copy()
        display_bat.columns = ["Player", "Pos", "PA", "AVG", "OBP", "SLG",
                               "OPS", "wOBA", "ISO", "HR", "K%", "BB%", "WAR/600", "WAR", "BABIP"]

        st.dataframe(display_bat, use_container_width=True, hide_index=True,
                     height=min(35 * len(display_bat) + 40, 500),
                     column_config={
                         "AVG": st.column_config.NumberColumn(format="%.3f"),
                         "OBP": st.column_config.NumberColumn(format="%.3f"),
                         "SLG": st.column_config.NumberColumn(format="%.3f"),
                         "OPS": st.column_config.NumberColumn(format="%.3f"),
                         "wOBA": st.column_config.NumberColumn(format="%.3f"),
                         "ISO": st.column_config.NumberColumn(format="%.3f"),
                         "BABIP": st.column_config.NumberColumn(format="%.3f",
                             help="Batting avg on balls in play. League avg ~.300. High = lucky or elite contact."),
                         "WAR/600": st.column_config.NumberColumn(format="%.1f",
                             help="WAR projected to 600 PA full season"),
                     })

        # Luck indicators
        st.markdown("### 🍀 BABIP Luck Detector")
        st.caption("BABIP far from .300 often regresses. High BABIP + low meta = lucky streak.")
        lucky = bat_df[bat_df["babip"] > 0.340][["player_name", "babip", "avg", "ops", "war", "meta"]].copy()
        unlucky = bat_df[bat_df["babip"] < 0.260][["player_name", "babip", "avg", "ops", "war", "meta"]].copy()

        col_lucky, col_unlucky = st.columns(2)
        with col_lucky:
            st.markdown("**🍀 Running Hot (BABIP > .340)**")
            if not lucky.empty:
                lucky.columns = ["Player", "BABIP", "AVG", "OPS", "WAR", "Meta"]
                st.dataframe(lucky.sort_values("BABIP", ascending=False),
                             use_container_width=True, hide_index=True,
                             column_config={"BABIP": st.column_config.NumberColumn(format="%.3f"),
                                           "AVG": st.column_config.NumberColumn(format="%.3f"),
                                           "OPS": st.column_config.NumberColumn(format="%.3f")})
            else:
                st.caption("No batters with BABIP > .340")
        with col_unlucky:
            st.markdown("**🥶 Running Cold (BABIP < .260)**")
            if not unlucky.empty:
                unlucky.columns = ["Player", "BABIP", "AVG", "OPS", "WAR", "Meta"]
                st.dataframe(unlucky.sort_values("BABIP"),
                             use_container_width=True, hide_index=True,
                             column_config={"BABIP": st.column_config.NumberColumn(format="%.3f"),
                                           "AVG": st.column_config.NumberColumn(format="%.3f"),
                                           "OPS": st.column_config.NumberColumn(format="%.3f")})
            else:
                st.caption("No batters with BABIP < .260")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: PITCHING — leaderboards + luck
# ════════════════════════════════════════════════════════════════════════════════
with tab_pitching:
    if pit_df.empty:
        st.info("No qualifying pitchers yet.")
    else:
        st.markdown("### Pitching Leaderboard")

        display_pit = pit_df[["player_name", "Role", "ip", "era", "fip", "ERA-FIP",
                              "whip", "K%", "BB%", "K-BB%", "WAR/200", "war"]].copy()
        display_pit.columns = ["Player", "Role", "IP", "ERA", "FIP", "ERA-FIP",
                               "WHIP", "K%", "BB%", "K-BB%", "WAR/200", "WAR"]

        st.dataframe(display_pit, use_container_width=True, hide_index=True,
                     height=min(35 * len(display_pit) + 40, 500),
                     column_config={
                         "ERA": st.column_config.NumberColumn(format="%.2f"),
                         "FIP": st.column_config.NumberColumn(format="%.2f"),
                         "ERA-FIP": st.column_config.NumberColumn(format="%.2f",
                             help="Positive = unlucky (ERA > FIP, should improve). Negative = lucky (ERA < FIP, may regress)."),
                         "WHIP": st.column_config.NumberColumn(format="%.2f"),
                         "IP": st.column_config.NumberColumn(format="%.1f"),
                         "K-BB%": st.column_config.NumberColumn(format="%.1f",
                             help="Elite >20%, Good >15%, Average 10-15%, Poor <10%"),
                         "WAR/200": st.column_config.NumberColumn(format="%.1f",
                             help="WAR projected to 200 IP full season"),
                     })

        # Luck table
        st.markdown("### 🎲 Regression Watch")
        st.caption("ERA-FIP gap shows luck. Negative = getting lucky. Positive = due for a bounce-back.")

        col_reg_lucky, col_reg_unlucky = st.columns(2)
        with col_reg_lucky:
            st.markdown("**🍀 Lucky (ERA << FIP) — may regress**")
            lucky_p = pit_df[pit_df["ERA-FIP"] < -0.5][["player_name", "Role", "era", "fip", "ERA-FIP", "ip"]].copy()
            if not lucky_p.empty:
                lucky_p.columns = ["Player", "Role", "ERA", "FIP", "Gap", "IP"]
                st.dataframe(lucky_p.sort_values("Gap"),
                             use_container_width=True, hide_index=True,
                             column_config={"ERA": st.column_config.NumberColumn(format="%.2f"),
                                           "FIP": st.column_config.NumberColumn(format="%.2f"),
                                           "Gap": st.column_config.NumberColumn(format="%.2f")})
            else:
                st.caption("No pitchers significantly outperforming their FIP")

        with col_reg_unlucky:
            st.markdown("**🔥 Unlucky (ERA >> FIP) — should improve**")
            unlucky_p = pit_df[pit_df["ERA-FIP"] > 0.5][["player_name", "Role", "era", "fip", "ERA-FIP", "ip"]].copy()
            if not unlucky_p.empty:
                unlucky_p.columns = ["Player", "Role", "ERA", "FIP", "Gap", "IP"]
                st.dataframe(unlucky_p.sort_values("Gap", ascending=False),
                             use_container_width=True, hide_index=True,
                             column_config={"ERA": st.column_config.NumberColumn(format="%.2f"),
                                           "FIP": st.column_config.NumberColumn(format="%.2f"),
                                           "Gap": st.column_config.NumberColumn(format="%.2f")})
            else:
                st.caption("No pitchers significantly underperforming their FIP")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: PLAYER CARD — Statcast-style percentile card
# ════════════════════════════════════════════════════════════════════════════════
with tab_player:
    all_players = set()
    if not bat_df.empty:
        all_players.update(bat_df["player_name"].tolist())
    if not pit_df.empty:
        all_players.update(pit_df["player_name"].tolist())

    selected = st.selectbox("Search for a player", [""] + sorted(all_players),
                            format_func=lambda x: "Type to search..." if x == "" else x,
                            key="player_card_search")

    if selected:
        is_batter = selected in bat_df["player_name"].values if not bat_df.empty else False
        is_pitcher = selected in pit_df["player_name"].values if not pit_df.empty else False

        if is_batter:
            row = bat_df[bat_df["player_name"] == selected].iloc[0]

            st.markdown(f"### {selected} — Batter")

            # Hero metrics
            mc = st.columns(8)
            mc[0].metric("AVG", f"{row['avg']:.3f}")
            mc[1].metric("OPS", f"{row['ops']:.3f}")
            mc[2].metric("wOBA", f"{row['wOBA']:.3f}")
            mc[3].metric("WAR", f"{row['war']:.1f}")
            mc[4].metric("HR", int(row["hr"]))
            mc[5].metric("K%", f"{row['K%']:.1f}%")
            mc[6].metric("BB%", f"{row['BB%']:.1f}%")
            mc[7].metric("BABIP", f"{row['babip']:.3f}")

            # Percentile bars — Statcast style
            st.markdown("#### Percentile Rankings")
            st.caption("Compared to all qualified batters on your team")

            stats_for_pctile = {
                "OPS": ("ops", False),
                "WAR/600": ("WAR/600", False),
                "wOBA": ("wOBA", False),
                "ISO (Power)": ("iso", False),
                "K%": ("K%", True),       # lower is better
                "BB%": ("BB%", False),
                "BABIP": ("babip", False),
                "Speed": ("sb", False),
            }

            pctile_data = []
            for label, (col, reverse) in stats_for_pctile.items():
                if col in bat_df.columns:
                    val = row[col]
                    if reverse:
                        pctile = 100 - (bat_df[col] < val).mean() * 100
                    else:
                        pctile = (bat_df[col] < val).mean() * 100
                    pctile_data.append({"stat": label, "value": val, "pctile": round(pctile)})

            if pctile_data:
                fig_pct = go.Figure()

                stats_labels = [p["stat"] for p in pctile_data]
                pctiles = [p["pctile"] for p in pctile_data]
                colors = []
                for p in pctiles:
                    if p >= 90: colors.append("#ff1744")
                    elif p >= 75: colors.append("#ff6d00")
                    elif p >= 50: colors.append(GOLD)
                    elif p >= 25: colors.append(ICE)
                    else: colors.append(GREY)

                fig_pct.add_trace(go.Bar(
                    y=stats_labels, x=pctiles,
                    orientation="h",
                    marker=dict(color=colors, line=dict(width=0)),
                    text=[f"{p}th" for p in pctiles],
                    textposition="outside",
                    textfont=dict(size=12, color="white"),
                    hovertemplate="%{y}: %{x}th percentile<extra></extra>",
                ))

                fig_pct.update_layout(
                    height=300, xaxis=dict(range=[0, 105], title="Percentile",
                                          gridcolor=GRID, zeroline=False),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor=BG, paper_bgcolor=BG,
                    margin=dict(t=10, b=30, l=100, r=40),
                )
                st.plotly_chart(fig_pct, use_container_width=True)

            # Meta vs actual
            if row.get("meta") and row["meta"] > 0:
                perf_rating = row["ops"] * 1000  # rough scale to compare with meta
                gap = perf_rating - row["meta"]
                if gap > 50:
                    st.success(f"🔥 **Overperforming** meta by {gap:.0f} points (OPS-scaled {perf_rating:.0f} vs {row['meta']:.0f} meta)")
                elif gap < -50:
                    st.error(f"📉 **Underperforming** meta by {abs(gap):.0f} points (OPS-scaled {perf_rating:.0f} vs {row['meta']:.0f} meta)")
                else:
                    st.info(f"✅ **Performing as expected** — OPS-scaled {perf_rating:.0f} vs {row['meta']:.0f} meta")

            # Trend chart (if multiple snapshots)
            bat_history = _df("""
                SELECT snapshot_date, avg, ops, war, pa FROM batting_stats
                WHERE player_name = ? ORDER BY snapshot_date
            """, (selected,))

            if len(bat_history) >= 2:
                st.markdown("#### Performance Trend")
                fig_trend = make_subplots(rows=1, cols=3, subplot_titles=("AVG", "OPS", "WAR"),
                                         horizontal_spacing=0.08)
                for i, (col, color) in enumerate([(("avg", "#1f77b4")), (("ops", "#ff7f0e")), (("war", "#2ca02c"))]):
                    fig_trend.add_trace(go.Scatter(
                        x=bat_history["snapshot_date"], y=bat_history[col],
                        mode="lines+markers", line=dict(color=color, width=2),
                    ), row=1, col=i+1)
                fig_trend.update_layout(height=280, showlegend=False,
                                       plot_bgcolor=BG, paper_bgcolor=BG,
                                       margin=dict(t=40, b=30, l=40, r=20))
                st.plotly_chart(fig_trend, use_container_width=True)

        if is_pitcher:
            row = pit_df[pit_df["player_name"] == selected].iloc[0]

            st.markdown(f"### {selected} — Pitcher ({row['Role']})")

            mc = st.columns(8)
            mc[0].metric("ERA", f"{row['era']:.2f}")
            mc[1].metric("FIP", f"{row['fip']:.2f}")
            mc[2].metric("ERA-FIP", f"{row['ERA-FIP']:+.2f}")
            mc[3].metric("WAR", f"{row['war']:.1f}")
            mc[4].metric("K%", f"{row['K%']:.1f}%")
            mc[5].metric("BB%", f"{row['BB%']:.1f}%")
            mc[6].metric("K-BB%", f"{row['K-BB%']:.1f}%")
            mc[7].metric("IP", f"{row['ip']:.1f}")

            # Percentile bars
            st.markdown("#### Percentile Rankings")

            pit_stats = {
                "ERA": ("era", True),       # lower is better
                "FIP": ("fip", True),
                "K%": ("K%", False),
                "BB%": ("BB%", True),       # lower is better
                "K-BB%": ("K-BB%", False),
                "WAR/200": ("WAR/200", False),
                "WHIP": ("whip", True),     # lower is better
            }

            pctile_data = []
            for label, (col, reverse) in pit_stats.items():
                if col in pit_df.columns:
                    val = row[col]
                    if reverse:
                        pctile = (pit_df[col] > val).mean() * 100  # lower = better percentile
                    else:
                        pctile = (pit_df[col] < val).mean() * 100
                    pctile_data.append({"stat": label, "value": val, "pctile": round(pctile)})

            if pctile_data:
                fig_pct = go.Figure()
                stats_labels = [p["stat"] for p in pctile_data]
                pctiles = [p["pctile"] for p in pctile_data]
                colors = []
                for p in pctiles:
                    if p >= 90: colors.append("#ff1744")
                    elif p >= 75: colors.append("#ff6d00")
                    elif p >= 50: colors.append(GOLD)
                    elif p >= 25: colors.append(ICE)
                    else: colors.append(GREY)

                fig_pct.add_trace(go.Bar(
                    y=stats_labels, x=pctiles, orientation="h",
                    marker=dict(color=colors, line=dict(width=0)),
                    text=[f"{p}th" for p in pctiles],
                    textposition="outside",
                    textfont=dict(size=12, color="white"),
                ))
                fig_pct.update_layout(
                    height=280, xaxis=dict(range=[0, 105], title="Percentile", gridcolor=GRID),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor=BG, paper_bgcolor=BG,
                    margin=dict(t=10, b=30, l=80, r=40),
                )
                st.plotly_chart(fig_pct, use_container_width=True)

            # Luck assessment
            gap = row["ERA-FIP"]
            if gap < -0.5:
                st.warning(f"🍀 **Lucky** — ERA {row['era']:.2f} is {abs(gap):.2f} below FIP {row['fip']:.2f}. Expect regression.")
            elif gap > 0.5:
                st.success(f"🔥 **Unlucky** — ERA {row['era']:.2f} is {gap:.2f} above FIP {row['fip']:.2f}. Should improve.")
            else:
                st.info(f"✅ **Sustainable** — ERA {row['era']:.2f} matches FIP {row['fip']:.2f}")

            # Trend
            pit_history = _df("""
                SELECT snapshot_date, era, whip, war, ip FROM pitching_stats
                WHERE player_name = ? ORDER BY snapshot_date
            """, (selected,))

            if len(pit_history) >= 2:
                st.markdown("#### Performance Trend")
                fig_trend = make_subplots(rows=1, cols=3, subplot_titles=("ERA", "WHIP", "WAR"),
                                         horizontal_spacing=0.08)
                fig_trend.add_trace(go.Scatter(
                    x=pit_history["snapshot_date"], y=pit_history["era"],
                    mode="lines+markers", line=dict(color="#d62728", width=2),
                ), row=1, col=1)
                fig_trend.add_trace(go.Scatter(
                    x=pit_history["snapshot_date"], y=pit_history["whip"],
                    mode="lines+markers", line=dict(color="#9467bd", width=2),
                ), row=1, col=2)
                fig_trend.add_trace(go.Scatter(
                    x=pit_history["snapshot_date"], y=pit_history["war"],
                    mode="lines+markers", line=dict(color="#2ca02c", width=2),
                ), row=1, col=3)
                fig_trend.update_layout(height=280, showlegend=False,
                                       plot_bgcolor=BG, paper_bgcolor=BG,
                                       margin=dict(t=40, b=30, l=40, r=20))
                fig_trend.update_yaxes(autorange="reversed", row=1, col=1)
                st.plotly_chart(fig_trend, use_container_width=True)

        if not is_batter and not is_pitcher:
            st.warning(f"No qualifying stats for {selected} (need {MIN_PA}+ PA or {MIN_IP}+ IP).")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5: META VS REALITY — does meta predict performance?
# ════════════════════════════════════════════════════════════════════════════════
with tab_meta:
    st.markdown("### 🧪 Meta Score vs Actual Performance")
    st.caption("Does our meta scoring system actually predict in-game results?")

    if not bat_df.empty:
        meta_bat = bat_df[bat_df["meta"].notna() & (bat_df["meta"] > 0)].copy()

        if not meta_bat.empty:
            meta_bat["perf_score"] = meta_bat["ops"] * 1000  # Scale to meta range

            # Correlation
            corr = meta_bat[["meta", "perf_score"]].corr().iloc[0, 1]

            col_corr, col_scatter = st.columns([1, 3])
            with col_corr:
                st.metric("Batting Correlation", f"{corr:.3f}",
                          help="1.0 = perfect prediction, 0 = random, <0 = backwards")
                if corr > 0.7:
                    st.success("Meta is a strong predictor")
                elif corr > 0.4:
                    st.info("Meta is a moderate predictor")
                elif corr > 0:
                    st.warning("Meta is a weak predictor")
                else:
                    st.error("Meta is not predictive")

                # Biggest mismatches
                meta_bat["gap"] = meta_bat["perf_score"] - meta_bat["meta"]
                overperformers = meta_bat.nlargest(3, "gap")[["player_name", "meta", "ops", "gap"]]
                underperformers = meta_bat.nsmallest(3, "gap")[["player_name", "meta", "ops", "gap"]]

                st.markdown("**🔥 Overperformers**")
                for _, r in overperformers.iterrows():
                    st.caption(f"{r['player_name']}: +{r['gap']:.0f} ({r['ops']:.3f} OPS vs {r['meta']:.0f} meta)")

                st.markdown("**📉 Underperformers**")
                for _, r in underperformers.iterrows():
                    st.caption(f"{r['player_name']}: {r['gap']:.0f} ({r['ops']:.3f} OPS vs {r['meta']:.0f} meta)")

            with col_scatter:
                fig_meta = go.Figure()

                # Perfect prediction line
                meta_range = [meta_bat["meta"].min() - 20, meta_bat["meta"].max() + 20]
                fig_meta.add_trace(go.Scatter(
                    x=meta_range, y=meta_range, mode="lines",
                    line=dict(color=GREY, dash="dash", width=1),
                    showlegend=False, hoverinfo="skip",
                ))

                # Color by gap (over/under)
                colors = [GREEN if g > 30 else (FIRE if g < -30 else ICE) for g in meta_bat["gap"]]

                fig_meta.add_trace(go.Scatter(
                    x=meta_bat["meta"], y=meta_bat["perf_score"],
                    mode="markers+text",
                    text=meta_bat["player_name"].apply(lambda n: n.split()[-1]),
                    textposition="top center",
                    textfont=dict(size=9, color="rgba(255,255,255,0.6)"),
                    marker=dict(size=12, color=colors,
                                line=dict(width=1, color="rgba(255,255,255,0.3)")),
                    hovertemplate="<b>%{customdata[0]}</b><br>Meta: %{x:.0f}<br>OPS×1000: %{y:.0f}<br>Gap: %{customdata[1]:+.0f}<extra></extra>",
                    customdata=np.column_stack([meta_bat["player_name"], meta_bat["gap"]]),
                ))

                fig_meta.update_layout(
                    height=420,
                    xaxis_title="Meta Score (predicted)", yaxis_title="OPS × 1000 (actual)",
                    plot_bgcolor=BG, paper_bgcolor=BG,
                    xaxis=dict(gridcolor=GRID, zeroline=False),
                    yaxis=dict(gridcolor=GRID, zeroline=False),
                    margin=dict(t=20, b=40, l=50, r=20),
                )

                # Quadrant labels
                mid_meta = meta_bat["meta"].median()
                mid_perf = meta_bat["perf_score"].median()
                fig_meta.add_annotation(x=mid_meta - 30, y=mid_perf + 80, text="🔮 Hidden Gems",
                                       font=dict(color=GREEN, size=10), showarrow=False)
                fig_meta.add_annotation(x=mid_meta + 30, y=mid_perf - 80, text="📉 Overrated",
                                       font=dict(color=FIRE, size=10), showarrow=False)
                fig_meta.add_annotation(x=mid_meta + 30, y=mid_perf + 80, text="⭐ As Advertised",
                                       font=dict(color=GOLD, size=10), showarrow=False)

                st.plotly_chart(fig_meta, use_container_width=True)
        else:
            st.info("No batters with both meta scores and in-game stats. Make sure cards are linked to roster.")

    if not pit_df.empty:
        meta_pit = pit_df[pit_df["meta"].notna() & (pit_df["meta"] > 0)].copy()

        if not meta_pit.empty:
            # ERA+ style: lower ERA = higher performance score
            max_era = meta_pit["era"].max()
            meta_pit["perf_score"] = ((max_era - meta_pit["era"]) / max_era * 500 + 300).clip(300, 800)

            corr_p = meta_pit[["meta", "perf_score"]].corr().iloc[0, 1]

            st.divider()
            st.markdown("### Pitching Meta vs Performance")
            col_pc, col_ps = st.columns([1, 3])
            with col_pc:
                st.metric("Pitching Correlation", f"{corr_p:.3f}")

            with col_ps:
                fig_mp = go.Figure()
                meta_range_p = [meta_pit["meta"].min() - 20, meta_pit["meta"].max() + 20]
                fig_mp.add_trace(go.Scatter(x=meta_range_p, y=meta_range_p, mode="lines",
                                           line=dict(color=GREY, dash="dash", width=1),
                                           showlegend=False, hoverinfo="skip"))

                meta_pit["gap"] = meta_pit["perf_score"] - meta_pit["meta"]
                colors_p = [GREEN if g > 30 else (FIRE if g < -30 else ICE) for g in meta_pit["gap"]]

                fig_mp.add_trace(go.Scatter(
                    x=meta_pit["meta"], y=meta_pit["perf_score"],
                    mode="markers+text",
                    text=meta_pit["player_name"].apply(lambda n: n.split()[-1]),
                    textposition="top center",
                    textfont=dict(size=9, color="rgba(255,255,255,0.6)"),
                    marker=dict(size=12, color=colors_p,
                                line=dict(width=1, color="rgba(255,255,255,0.3)")),
                    hovertemplate="<b>%{customdata[0]}</b><br>Meta: %{x:.0f}<br>Perf: %{y:.0f}<br>ERA: %{customdata[1]:.2f}<extra></extra>",
                    customdata=np.column_stack([meta_pit["player_name"], meta_pit["era"]]),
                ))

                fig_mp.update_layout(
                    height=380,
                    xaxis_title="Meta Score", yaxis_title="Performance Score (ERA-based)",
                    plot_bgcolor=BG, paper_bgcolor=BG,
                    xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
                    margin=dict(t=20, b=40, l=50, r=20),
                )
                st.plotly_chart(fig_mp, use_container_width=True)

    # ── Calibration Section ──
    st.divider()
    st.subheader("🔧 Auto-Calibrate Meta Weights")
    st.caption(
        "Use your team's actual performance data to improve the meta formula. "
        "The system learns which card attributes actually predict in-game success for YOUR cards."
    )

    try:
        from app.core.meta_validation import auto_calibrate_weights, get_calibration_history
        from app.core.meta_scoring import get_weights_with_source

        # Show current weight source
        _, _, source = get_weights_with_source()
        source_labels = {
            'calibrated': '🎯 Calibrated (learned from your data)',
            'config': '⚙️ Config (manual weights from config.yaml)',
            'default': '📊 Default (league-wide correlation analysis)',
        }
        st.info(f"**Current weights:** {source_labels.get(source, source)}")

        # Show calibration history
        cal_history = get_calibration_history(conn)
        if cal_history:
            last_cal = cal_history[0]
            st.caption(
                f"Last calibration: {last_cal.get('created_at', 'unknown')} — "
                f"Batting R²: {last_cal.get('r_squared', 0):.3f}, "
                f"Confidence: {last_cal.get('confidence', 0):.0%}"
            )

        cal_col1, cal_col2 = st.columns([1, 3])
        with cal_col1:
            if st.button("🔄 Run Calibration", type="primary", use_container_width=True,
                         help="Analyze your team's performance data to find better meta weights"):
                with st.spinner("Analyzing card ratings vs in-game performance..."):
                    result = auto_calibrate_weights(conn)

                if result.get("message"):
                    if "error" in result["message"].lower() or "need" in result["message"].lower():
                        st.error(result["message"])
                    else:
                        st.success(result["message"])

                changes = result.get("changes", [])
                if changes:
                    st.markdown("**Weight Changes Applied:**")
                    for ch in changes:
                        delta = ch['new_weight'] - ch['old_weight']
                        arrow = "⬆️" if delta > 0 else "⬇️"
                        st.caption(
                            f"{arrow} **{ch['stat']}**: {ch['old_weight']:.2f} → {ch['new_weight']:.2f} "
                            f"({delta:+.2f}) — {ch.get('reason', '')}"
                        )
                    st.info("✅ New weights saved. They'll be used for all meta calculations going forward. "
                            "Re-import your market data to recalculate all card meta scores.")
                elif not result.get("message", "").startswith("Error"):
                    st.info("No significant weight changes needed — current weights are performing well.")

        with cal_col2:
            # Show current vs calibrated weight comparison
            from app.utils.constants import DEFAULT_BATTING_WEIGHTS, DEFAULT_PITCHING_WEIGHTS
            bw, pw, _ = get_weights_with_source()

            w_col1, w_col2 = st.columns(2)
            with w_col1:
                st.markdown("**Batting Weights**")
                bat_w_data = []
                for stat in ['contact', 'gap_power', 'power', 'eye', 'avoid_ks', 'babip', 'defense', 'ovr']:
                    default = DEFAULT_BATTING_WEIGHTS.get(stat, 0)
                    current = bw.get(stat, default)
                    bat_w_data.append({
                        "Stat": stat.replace('_', ' ').title(),
                        "Default": round(default, 2),
                        "Current": round(current, 2),
                        "Δ": round(current - default, 2),
                    })
                st.dataframe(pd.DataFrame(bat_w_data), use_container_width=True, hide_index=True,
                             height=320)

            with w_col2:
                st.markdown("**Pitching Weights**")
                pit_w_data = []
                for stat in ['stuff', 'movement', 'control', 'p_hr', 'ovr', 'stamina_hold']:
                    default = DEFAULT_PITCHING_WEIGHTS.get(stat, 0)
                    current = pw.get(stat, default)
                    pit_w_data.append({
                        "Stat": stat.replace('_', ' ').title(),
                        "Default": round(default, 2),
                        "Current": round(current, 2),
                        "Δ": round(current - default, 2),
                    })
                st.dataframe(pd.DataFrame(pit_w_data), use_container_width=True, hide_index=True,
                             height=250)

    except Exception as e:
        st.warning(f"Calibration module not available: {e}")


conn.close()
