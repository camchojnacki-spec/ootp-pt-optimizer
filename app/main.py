"""OOTP Perfect Team Optimizer — Streamlit entry point."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import init_db, get_connection, load_config
from app.utils.sparklines import text_sparkline

# Initialize database on startup
init_db()

st.set_page_config(
    page_title="OOTP PT Optimizer",
    page_icon="\u26be",
    layout="wide",
    initial_sidebar_state="expanded",
)

conn = get_connection()
config = load_config()
budget = config.get('pp_budget', 500)

# Last ingestion times
logs = conn.execute("""
    SELECT file_type, MAX(ingested_at) as last_import, SUM(row_count) as total_rows
    FROM ingestion_log GROUP BY file_type ORDER BY last_import DESC
""").fetchall()

# ============================================================
# SIDEBAR — team info, quick stats, navigation shortcuts
# ============================================================
with st.sidebar:
    st.markdown(f"### \u26be {config.get('team_name', 'My Team')}")
    st.metric("PP Budget", f"{budget:,}")

    if logs:
        buy_count = conn.execute("SELECT COUNT(*) FROM recommendations WHERE rec_type='buy' AND dismissed=0").fetchone()[0]
        sell_count = conn.execute("SELECT COUNT(*) FROM recommendations WHERE rec_type='sell' AND dismissed=0").fetchone()[0]
        sell_pp_row = conn.execute("SELECT COALESCE(SUM(estimated_price), 0) as t FROM recommendations WHERE rec_type='sell' AND dismissed=0").fetchone()
        sell_pp = sell_pp_row['t'] or 0

        st.caption(f"{buy_count} buy recs | {sell_count} sell recs")
        if sell_pp > 0:
            st.caption(f"Sellable: {sell_pp:,} PP")

    st.divider()

    # Quick nav
    st.markdown("**Quick Links**")
    st.page_link("pages/1_Buy_Recommendations.py", label="Buy Recs", icon="\U0001f6d2")
    st.page_link("pages/2_Sell_Recommendations.py", label="Sell Recs", icon="\U0001f4b0")
    st.page_link("pages/4_Roster_Optimizer.py", label="Roster Optimizer", icon="📋")
    st.page_link("pages/7_Game_Stats.py", label="Game Stats", icon="📊")
    st.page_link("pages/13_Tournament_Builder.py", label="Tournament Builder", icon="\U0001f3c6")
    st.page_link("pages/15_Export_Plan.py", label="Export Plan", icon="\U0001f4cb")

    st.divider()
    st.caption("Updated: " + (str(logs[0]['last_import'])[:16] if logs else "Never"))

# ============================================================
# HEADER
# ============================================================
st.title(f"\u26be {config.get('team_name', 'OOTP PT Optimizer')}")

# Price alert notifications
try:
    from app.core.price_alerts import check_alerts, get_triggered_alerts
    new_alerts = check_alerts(conn)
    if new_alerts:
        for alert in new_alerts:
            st.toast(f"\U0001f514 {alert['card_title']} hit {alert['current_price']:,} PP!")

    recent_alerts = get_triggered_alerts(conn)
    if recent_alerts:
        with st.container():
            for a in recent_alerts[:3]:
                direction = "dropped below" if a.get('alert_type') == 'below' else "rose above"
                st.warning(f"\U0001f514 **{a['card_title']}** {direction} {a['target_price']:,} PP (now {a.get('current_price', '?'):,} PP)")
except Exception:
    pass

# ============================================================
# IMPORT SECTION
# ============================================================
with st.expander("\U0001f4e5 Import Data from OOTP", expanded=not bool(logs)):
    st.markdown("""
**Card Ratings & Market** (export every session):
1. **Card Shop** > CSV export button
2. **Roster > Player List > Batting Ratings** > Export CSV
3. **Roster > Player List > Pitching Ratings** > Export CSV
4. **Collection > Manage Cards > Batting Ratings** > Export CSV
5. **Collection > Manage Cards > Pitching Ratings** > Export CSV

**Game Stats** (export periodically):
6. **Roster > Batting Stats (1 & 2)** > Export CSV
7. **Roster > Pitching Stats (1 & 2)** > Export CSV
8. **League > Sortable Stats > Batting (Stats & Ratings)** > Export CSV
9. **League > Sortable Stats > Pitching (Stats & Ratings)** > Export CSV

**Lineups & Pitching Staff** (export when lineup changes):
10. **Lineups > vs RHP + DH** > Export CSV
11. **Lineups > vs LHP + DH** > Export CSV
12. **Lineups > Overview** > Export CSV
13. **Pitching** (team pitching roster) > Export CSV

All CSVs go to: `{config.get('watch_directory', 'your watch directory')}`
    """)

    col_import1, col_import2 = st.columns(2)

    with col_import1:
        if st.button("\U0001f4e5 Import All from Watch Folder", type="primary", use_container_width=True):
            from app.core.ingestion import ingest_file
            from app.core.recommendations import generate_recommendations
            watch_dir = config.get('watch_directory', '')
            if Path(watch_dir).exists():
                csvs = list(Path(watch_dir).glob('*.csv'))
                if csvs:
                    progress = st.progress(0, text="Importing...")
                    import_results = []
                    for i, csv_path in enumerate(csvs):
                        progress.progress((i + 1) / len(csvs), text=f"Importing {csv_path.name}...")
                        result = ingest_file(str(csv_path))
                        result['filename'] = csv_path.name
                        import_results.append(result)
                    progress.progress(1.0, text="Generating recommendations...")
                    generate_recommendations()

                    success_count = sum(1 for r in import_results if r.get('status') == 'success')
                    total_rows = sum(r.get('rows', 0) for r in import_results)
                    st.success(f"Imported {success_count}/{len(csvs)} files ({total_rows:,} rows)")

                    with st.expander("Import details"):
                        for r in import_results:
                            status = r.get('status', 'unknown')
                            rows = r.get('rows', 0)
                            fname = r.get('filename', '?')
                            ftype = r.get('file_type', '?')
                            icon = "\u2705" if status == 'success' and rows > 0 else ("\u2796" if rows == 0 else "\u274c")
                            st.write(f"{icon} `{ftype}` {rows:,} rows - *{fname}*")

                    st.rerun()
                else:
                    st.warning("No CSV files in watch folder.")
            else:
                st.error(f"Watch folder not found: {watch_dir}")

    with col_import2:
        uploaded = st.file_uploader("Or drag & drop CSVs", type=['csv'],
                                     accept_multiple_files=True, key="main_upload")
        if uploaded:
            import tempfile, os
            from app.core.ingestion import ingest_file
            from app.core.recommendations import generate_recommendations
            for f in uploaded:
                tmp_path = os.path.join(tempfile.gettempdir(), f.name)
                with open(tmp_path, 'wb') as tmp:
                    tmp.write(f.getbuffer())
                result = ingest_file(tmp_path)
                st.write(f"**{f.name}**: {result.get('rows', 0):,} rows")
            generate_recommendations()
            st.success("Import complete!")
            st.rerun()

# ============================================================
# DASHBOARD — only when data exists
# ============================================================
if not logs:
    st.info("\U0001f449 No data yet. Expand the import section above to get started.")
    conn.close()
    st.stop()

# ── Key Metrics Bar ──
card_count = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
roster_count = conn.execute("SELECT COUNT(*) FROM roster_current").fetchone()[0]

roster_meta_row = conn.execute("""
    SELECT SUM(meta_score) as total_meta FROM roster_current
    WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
""").fetchone()
total_roster_meta = roster_meta_row['total_meta'] or 0

weakest_pos = conn.execute("""
    SELECT position, MAX(meta_score) as best_meta, player_name
    FROM roster_current WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
    GROUP BY position ORDER BY best_meta ASC LIMIT 1
""").fetchone()

avg_roster_meta_row = conn.execute("""
    SELECT AVG(meta_score) as avg_meta FROM roster_current
    WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen') AND meta_score > 0
""").fetchone()
avg_roster_meta = avg_roster_meta_row['avg_meta'] or 0

avg_market_top_row = conn.execute("""
    SELECT AVG(COALESCE(meta_score_batting, meta_score_pitching)) as avg_meta
    FROM cards WHERE tier >= 5 AND COALESCE(meta_score_batting, meta_score_pitching) > 0
""").fetchone()
avg_market_top = avg_market_top_row['avg_meta'] or 1

health_pct = min(100, int((avg_roster_meta / avg_market_top) * 100)) if avg_market_top > 0 else 0

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Roster Meta", f"{total_roster_meta:,.0f}")
with m2:
    st.metric("Team Health", f"{health_pct}%")
with m3:
    if weakest_pos:
        st.metric("Weakest Pos", weakest_pos['position'],
                   delta=f"{weakest_pos['best_meta']:.0f} meta", delta_color="off")
    else:
        st.metric("Weakest Pos", "\u2014")
with m4:
    st.metric("Cards", f"{card_count:,}")
with m5:
    st.metric("Roster", roster_count)

# ── Main Content Tabs ──
tab_overview, tab_recs, tab_roster, tab_data = st.tabs([
    "\U0001f3e0 Overview", "\U0001f4cb Recommendations", "\U0001f3af Roster", "\U0001f4be Data"
])

# ============================================================
# TAB: Overview
# ============================================================
with tab_overview:
    # Team Health Progress Bars
    try:
        bat_starter_meta_row = conn.execute("""
            SELECT AVG(meta_score) as avg_meta FROM roster_current
            WHERE lineup_role = 'starter' AND position NOT IN ('SP', 'RP', 'CL') AND meta_score > 0
        """).fetchone()
        bat_starter_meta = bat_starter_meta_row['avg_meta'] or 0 if bat_starter_meta_row else 0

        market_bat_avg_row = conn.execute("""
            SELECT AVG(meta_score_batting) as avg_meta FROM cards WHERE tier >= 5 AND meta_score_batting > 0
        """).fetchone()
        market_bat_avg = market_bat_avg_row['avg_meta'] or 1 if market_bat_avg_row else 1

        batting_depth_pct = min(100, int((bat_starter_meta / market_bat_avg) * 100)) if market_bat_avg > 0 else 0

        pit_starter_meta_row = conn.execute("""
            SELECT AVG(meta_score) as avg_meta FROM roster_current
            WHERE lineup_role IN ('rotation', 'closer', 'bullpen') AND meta_score > 0
        """).fetchone()
        pit_starter_meta = pit_starter_meta_row['avg_meta'] or 0 if pit_starter_meta_row else 0

        market_pit_avg_row = conn.execute("""
            SELECT AVG(meta_score_pitching) as avg_meta FROM cards WHERE tier >= 5 AND meta_score_pitching > 0
        """).fetchone()
        market_pit_avg = market_pit_avg_row['avg_meta'] or 1 if market_pit_avg_row else 1

        pitching_depth_pct = min(100, int((pit_starter_meta / market_pit_avg) * 100)) if market_pit_avg > 0 else 0

        bp_good = conn.execute("""
            SELECT COUNT(*) as c FROM roster_current
            WHERE position IN ('RP', 'CL') AND lineup_role IN ('bullpen', 'closer') AND meta_score > 400
        """).fetchone()['c']
        bp_total = conn.execute("""
            SELECT COUNT(*) as c FROM roster_current
            WHERE position IN ('RP', 'CL') AND lineup_role IN ('bullpen', 'closer')
        """).fetchone()['c']
        bullpen_pct = int((bp_good / bp_total) * 100) if bp_total > 0 else 0

        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.markdown("**Overall**")
            st.progress(min(1.0, health_pct / 100))
            st.caption(f"{health_pct}% vs Diamond+ avg")
        with h2:
            st.markdown("**Batting**")
            st.progress(min(1.0, batting_depth_pct / 100))
            st.caption(f"{batting_depth_pct}% depth")
        with h3:
            st.markdown("**Pitching**")
            st.progress(min(1.0, pitching_depth_pct / 100))
            st.caption(f"{pitching_depth_pct}% depth")
        with h4:
            st.markdown("**Bullpen**")
            st.progress(min(1.0, bullpen_pct / 100))
            st.caption(f"{bp_good}/{bp_total} quality arms")
    except Exception:
        pass

    # Game Stats Summary
    try:
        bat_stat_count = conn.execute("SELECT COUNT(DISTINCT player_name) as c FROM batting_stats").fetchone()["c"]
        pit_stat_count = conn.execute("SELECT COUNT(DISTINCT player_name) as c FROM pitching_stats").fetchone()["c"]
        if bat_stat_count > 0 or pit_stat_count > 0:
            st.divider()
            team_ops_row = conn.execute("SELECT AVG(ops) as v FROM batting_stats WHERE ab >= 50").fetchone()
            team_era_row = conn.execute("SELECT AVG(era) as v FROM pitching_stats WHERE ip >= 30").fetchone()
            mvp_row = conn.execute("SELECT player_name, war FROM batting_stats WHERE ab >= 50 ORDER BY war DESC LIMIT 1").fetchone()
            cy_row = conn.execute("SELECT player_name, war FROM pitching_stats WHERE ip >= 30 ORDER BY war DESC LIMIT 1").fetchone()

            gs1, gs2, gs3, gs4 = st.columns(4)
            with gs1:
                team_ops = team_ops_row["v"] if team_ops_row and team_ops_row["v"] else 0
                st.metric("Team OPS", f"{team_ops:.3f}")
            with gs2:
                team_era = team_era_row["v"] if team_era_row and team_era_row["v"] else 0
                st.metric("Team ERA", f"{team_era:.2f}")
            with gs3:
                if mvp_row:
                    st.metric("MVP", mvp_row["player_name"], delta=f"{mvp_row['war']:.1f} WAR", delta_color="off")
                else:
                    st.metric("MVP", "\u2014")
            with gs4:
                if cy_row:
                    st.metric("Cy Young", cy_row["player_name"], delta=f"{cy_row['war']:.1f} WAR", delta_color="off")
                else:
                    st.metric("Cy Young", "\u2014")
    except Exception:
        pass

    # AI Strategic Insights
    try:
        ai_table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ai_insights'"
        ).fetchone()
        if ai_table_exists:
            latest_insight = conn.execute("""
                SELECT content, created_at FROM ai_insights
                WHERE insight_type = 'strategic_summary'
                ORDER BY created_at DESC LIMIT 1
            """).fetchone()
            if latest_insight and latest_insight['content']:
                st.divider()
                st.markdown("**AI Strategic Insights**")
                st.info(latest_insight['content'])
                st.caption(f"Generated: {str(latest_insight['created_at'])[:19]}")
    except Exception:
        pass

# ============================================================
# TAB: Recommendations
# ============================================================
with tab_recs:
    col_buy, col_sell = st.columns(2)

    with col_buy:
        st.markdown("#### \U0001f6d2 Buy")
        buys = conn.execute("""
            SELECT card_id, card_title, position, reason, priority, estimated_price, meta_score, value_ratio
            FROM recommendations
            WHERE rec_type = 'buy' AND dismissed = 0
            ORDER BY priority ASC, value_ratio DESC
            LIMIT 10
        """).fetchall()

        if buys:
            buy_data = []
            for b in buys:
                buy_data.append({
                    "Card": b['card_title'],
                    "Pos": b['position'],
                    "Trend": text_sparkline(b['card_id'], conn) if b['card_id'] else '',
                    "Meta": f"{b['meta_score']:.0f}" if b['meta_score'] else "\u2014",
                    "Price": f"{b['estimated_price']:,}" if b['estimated_price'] else "\u2014",
                    "Value": f"{b['value_ratio']:.1f}" if b['value_ratio'] else "\u2014",
                })
            st.dataframe(pd.DataFrame(buy_data), use_container_width=True, hide_index=True)
            st.page_link("pages/1_Buy_Recommendations.py", label="Browse all \u2192", icon="\U0001f6d2")
        else:
            st.info("No buy recommendations yet.")

    with col_sell:
        st.markdown("#### \U0001f4b0 Sell")
        sells = conn.execute("""
            SELECT card_id, card_title, position, reason, priority, estimated_price
            FROM recommendations
            WHERE rec_type = 'sell' AND dismissed = 0
            ORDER BY priority ASC, estimated_price DESC
            LIMIT 10
        """).fetchall()

        if sells:
            sell_data = []
            for s in sells:
                sell_data.append({
                    "Card": s['card_title'],
                    "Pos": s['position'],
                    "Trend": text_sparkline(s['card_id'], conn) if s['card_id'] else '',
                    "Price": f"{s['estimated_price']:,}" if s['estimated_price'] else "\u2014",
                    "Reason": s['reason'],
                })
            st.dataframe(pd.DataFrame(sell_data), use_container_width=True, hide_index=True)
            st.page_link("pages/2_Sell_Recommendations.py", label="Browse all \u2192", icon="\U0001f4b0")
        else:
            st.info("No sell recommendations yet.")

# ============================================================
# TAB: Roster
# ============================================================
with tab_roster:
    roster = conn.execute("""
        SELECT player_name, position, lineup_role, ovr, meta_score
        FROM roster_current
        ORDER BY
            CASE lineup_role
                WHEN 'starter' THEN 1 WHEN 'rotation' THEN 2
                WHEN 'closer' THEN 3 WHEN 'bullpen' THEN 4
                WHEN 'bench' THEN 5 WHEN 'reserve' THEN 6
            END, meta_score DESC
    """).fetchall()

    if roster:
        # ── Depth Chart: every position with starter, backup, and bench ──
        st.markdown("**Depth Chart**")

        # Build depth chart by position, sorted by meta within each position
        from collections import defaultdict
        depth = defaultdict(list)
        for r in roster:
            depth[r['position']].append(dict(r))
        for pos in depth:
            depth[pos].sort(key=lambda x: -(x['meta_score'] or 0))

        # Batting depth chart — starter + top 2 backups per position
        bat_positions = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
        bat_rows = []
        for pos in bat_positions:
            players = depth.get(pos, [])[:3]  # Starter + 2 backups max
            starter_meta = players[0]['meta_score'] if players else 0
            for i, p in enumerate(players):
                role_label = "\u2b50 Starter" if i == 0 else f"Backup {i}"
                meta = p['meta_score'] or 0
                gap_to_starter = ""
                if i > 0 and starter_meta:
                    diff = meta - starter_meta
                    gap_to_starter = f"{diff:+.0f}"

                # Grab in-game stats
                war_str = ""
                ops_str = ""
                bstat = conn.execute(
                    "SELECT ops, war, ab FROM batting_stats WHERE player_name = ? ORDER BY snapshot_date DESC LIMIT 1",
                    (p['player_name'],)
                ).fetchone()
                if bstat and bstat['ab'] and bstat['ab'] > 0:
                    ops_str = f"{bstat['ops']:.3f}"
                    war_str = f"{bstat['war']:.1f}"

                bat_rows.append({
                    "Pos": pos,
                    "Depth": role_label,
                    "Player": p['player_name'],
                    "OVR": p['ovr'],
                    "Meta": round(meta),
                    "Gap": gap_to_starter,
                    "OPS": ops_str,
                    "WAR": war_str,
                })

        if bat_rows:
            st.markdown("*Batting*")
            st.dataframe(
                pd.DataFrame(bat_rows),
                use_container_width=True,
                hide_index=True,
                height=min(35 * len(bat_rows) + 38, 500),
            )

        # Pitching depth chart
        pitch_rows = []

        # Rotation — top 5 SP + 2 depth arms
        sp_list = depth.get('SP', [])[:7]
        ace_meta = sp_list[0]['meta_score'] if sp_list else 0
        for i, p in enumerate(sp_list):
            slot = f"\u2b50 SP{i+1}" if i < 5 else "Depth SP"
            meta = p['meta_score'] or 0
            gap = f"{meta - ace_meta:+.0f}" if i > 0 and ace_meta else ""

            era_str = ""
            war_str = ""
            pstat = conn.execute(
                "SELECT era, war, games FROM pitching_stats WHERE player_name = ? ORDER BY snapshot_date DESC LIMIT 1",
                (p['player_name'],)
            ).fetchone()
            if pstat and pstat['games'] and pstat['games'] > 0:
                era_str = f"{pstat['era']:.2f}"
                war_str = f"{pstat['war']:.1f}"

            pitch_rows.append({
                "Pos": "SP",
                "Slot": slot,
                "Player": p['player_name'],
                "OVR": p['ovr'],
                "Meta": round(meta),
                "Gap": gap,
                "ERA": era_str,
                "WAR": war_str,
            })

        # Bullpen — CL first, then top 7 RP
        cl_list = depth.get('CL', [])[:1]
        rp_list = depth.get('RP', [])[:7]
        bp_all = [(p, 'CL') for p in cl_list] + [(p, 'RP') for p in rp_list]

        slot_names = ["Closer", "Setup 1", "Setup 2", "Middle 1", "Middle 2", "Long 1", "Long 2", "Mop-up"]
        for i, (p, role) in enumerate(bp_all):
            slot = slot_names[i] if i < len(slot_names) else "Extra"
            if i == 0 and role == 'CL':
                slot = "\u2b50 Closer"
            meta = p['meta_score'] or 0

            era_str = ""
            war_str = ""
            pstat = conn.execute(
                "SELECT era, war, games FROM pitching_stats WHERE player_name = ? ORDER BY snapshot_date DESC LIMIT 1",
                (p['player_name'],)
            ).fetchone()
            if pstat and pstat['games'] and pstat['games'] > 0:
                era_str = f"{pstat['era']:.2f}"
                war_str = f"{pstat['war']:.1f}"

            pitch_rows.append({
                "Pos": role,
                "Slot": slot,
                "Player": p['player_name'],
                "OVR": p['ovr'],
                "Meta": round(meta),
                "Gap": "",
                "ERA": era_str,
                "WAR": war_str,
            })

        if pitch_rows:
            st.markdown("*Pitching*")
            st.dataframe(
                pd.DataFrame(pitch_rows),
                use_container_width=True,
                hide_index=True,
                height=min(35 * len(pitch_rows) + 38, 500),
            )

        st.divider()
        roster_data = []
        for r in roster:
            row = {
                "Player": r['player_name'],
                "Pos": r['position'],
                "Role": r['lineup_role'],
                "OVR": r['ovr'],
                "Meta": f"{r['meta_score']:.0f}" if r['meta_score'] else "\u2014",
            }

            is_pitcher = r['position'] in ('SP', 'RP', 'CL')
            if is_pitcher:
                pstat = conn.execute(
                    "SELECT era, whip, k, war, games FROM pitching_stats WHERE player_name = ? ORDER BY snapshot_date DESC LIMIT 1",
                    (r['player_name'],)
                ).fetchone()
                if pstat and pstat['games'] and pstat['games'] > 0:
                    row["ERA"] = f"{pstat['era']:.2f}"
                    row["WHIP"] = f"{pstat['whip']:.2f}"
                    row["K"] = str(pstat['k'])
                    row["WAR"] = f"{pstat['war']:.1f}"
                else:
                    row["ERA"] = "\u2014"
                    row["WHIP"] = "\u2014"
                    row["K"] = "\u2014"
                    row["WAR"] = "\u2014"
            else:
                bstat = conn.execute(
                    "SELECT avg, ops, hr, rbi, war, ab FROM batting_stats WHERE player_name = ? ORDER BY snapshot_date DESC LIMIT 1",
                    (r['player_name'],)
                ).fetchone()
                if bstat and bstat['ab'] and bstat['ab'] > 0:
                    row["AVG"] = f"{bstat['avg']:.3f}"
                    row["OPS"] = f"{bstat['ops']:.3f}"
                    row["HR"] = str(bstat['hr'])
                    row["WAR"] = f"{bstat['war']:.1f}"
                else:
                    row["AVG"] = "\u2014"
                    row["OPS"] = "\u2014"
                    row["HR"] = "\u2014"
                    row["WAR"] = "\u2014"

            roster_data.append(row)

        batter_rows = [r for r in roster_data if r.get("AVG") is not None]
        pitcher_rows = [r for r in roster_data if r.get("ERA") is not None]

        if batter_rows:
            st.markdown("**Batters**")
            bat_df = pd.DataFrame(batter_rows)
            bat_cols = ["Player", "Pos", "Role", "OVR", "Meta", "AVG", "OPS", "HR", "WAR"]
            bat_df = bat_df[[c for c in bat_cols if c in bat_df.columns]]
            st.dataframe(bat_df, use_container_width=True, hide_index=True,
                column_config={
                    "OVR": st.column_config.NumberColumn(format="%d"),
                    "WAR": st.column_config.NumberColumn(format="%.1f"),
                })

        if pitcher_rows:
            st.markdown("**Pitchers**")
            pit_df = pd.DataFrame(pitcher_rows)
            pit_cols = ["Player", "Pos", "Role", "OVR", "Meta", "ERA", "WHIP", "K", "WAR"]
            pit_df = pit_df[[c for c in pit_cols if c in pit_df.columns]]
            st.dataframe(pit_df, use_container_width=True, hide_index=True,
                column_config={
                    "OVR": st.column_config.NumberColumn(format="%d"),
                    "WAR": st.column_config.NumberColumn(format="%.1f"),
                })

        st.page_link("pages/4_Roster_Optimizer.py", label="Roster Optimizer \u2192", icon="\U0001f4cb")
    else:
        st.info("No roster data. Import your roster CSV first.")

# ============================================================
# TAB: Data
# ============================================================
with tab_data:
    # Data freshness table
    st.markdown("**Data Freshness**")
    freshness_data = []
    file_type_labels = {
        'market': 'Card Shop (pt_card_list)',
        'roster_batting': 'Roster Batting Ratings',
        'roster_pitching': 'Roster Pitching Ratings',
        'collection_batting': 'Collection Batting',
        'collection_pitching': 'Collection Pitching',
        'stats_batting': 'League Batting Stats',
        'stats_pitching': 'League Pitching Stats',
        'roster_batting_stats': 'Roster Batting Stats',
        'roster_pitching_stats': 'Roster Pitching Stats',
        'roster_batting_stats_adv': 'Roster Batting Stats (Adv)',
        'roster_pitching_stats_adv': 'Roster Pitching Stats (Adv)',
    }
    for log in logs:
        freshness_data.append({
            "Source": file_type_labels.get(log['file_type'], log['file_type']),
            "Last Import": str(log['last_import'])[:16],
            "Rows": log['total_rows'],
        })
    st.dataframe(pd.DataFrame(freshness_data), use_container_width=True, hide_index=True)

    # Import validation summary
    st.divider()
    st.markdown("**Coverage**")
    stats_counts = {}
    for label, query in [
        ("Cards", "SELECT COUNT(*) as c FROM cards"),
        ("Roster", "SELECT COUNT(*) as c FROM roster_current"),
        ("Collection", "SELECT COUNT(*) as c FROM collection_current"),
        ("Batting Stats", "SELECT COUNT(DISTINCT player_name) as c FROM batting_stats"),
        ("Pitching Stats", "SELECT COUNT(DISTINCT player_name) as c FROM pitching_stats"),
        ("Price Snapshots", "SELECT COUNT(DISTINCT snapshot_date) as c FROM price_snapshots"),
    ]:
        try:
            stats_counts[label] = conn.execute(query).fetchone()["c"]
        except Exception:
            stats_counts[label] = 0

    dc1, dc2, dc3, dc4, dc5, dc6 = st.columns(6)
    dc1.metric("Cards", f"{stats_counts['Cards']:,}")
    dc2.metric("Roster", stats_counts["Roster"])
    dc3.metric("Collection", stats_counts["Collection"])
    dc4.metric("Bat Stats", stats_counts["Batting Stats"])
    dc5.metric("Pit Stats", stats_counts["Pitching Stats"])
    dc6.metric("Snapshots", stats_counts["Price Snapshots"])

    snapshots = stats_counts.get("Price Snapshots", 0)
    if snapshots >= 2:
        st.success(f"Trending active: {snapshots} price snapshots collected.")
    elif snapshots == 1:
        st.info("First snapshot recorded. Import again after your next session to start trending.")
    else:
        st.caption("Import market data across multiple sessions to enable price trending.")

conn.close()
