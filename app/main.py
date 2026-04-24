"""OOTP Perfect Team Optimizer — Streamlit entry point."""
import streamlit as st
import pandas as pd
import yaml
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import init_db, get_connection, load_config
from app.utils.sparklines import text_sparkline
from app.utils.sidebar_nav import render_sidebar_nav

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
config_path = Path(__file__).parent.parent / "config.yaml"
budget = config.get('pp_budget', 500)

# Last ingestion times
logs = conn.execute("""
    SELECT file_type, MAX(ingested_at) as last_import, SUM(row_count) as total_rows
    FROM ingestion_log GROUP BY file_type ORDER BY last_import DESC
""").fetchall()

# ============================================================
# SIDEBAR — grouped nav (helper) + team info / quick stats
# Auto-discovered nav is hidden via .streamlit/config.toml so this is the
# only navigation surface the user sees.
# ============================================================
render_sidebar_nav()

with st.sidebar:
    st.markdown(f"### \u26be {config.get('team_name', 'My Team')}")

    # Editable PP Budget — saves to config.yaml on change
    new_budget = st.number_input(
        "PP Budget",
        min_value=0,
        max_value=999999,
        value=budget,
        step=100,
        key="sidebar_pp_budget",
        label_visibility="collapsed",
    )
    # Show it big like a metric
    st.markdown(f"<div style='text-align:center;margin-top:-15px;margin-bottom:5px;'>"
                f"<span style='font-size:0.8em;color:#888;'>PP Budget</span><br>"
                f"<span style='font-size:1.8em;font-weight:bold;'>{new_budget:,}</span></div>",
                unsafe_allow_html=True)

    if new_budget != budget:
        config['pp_budget'] = new_budget
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

    if logs:
        buy_count = conn.execute("SELECT COUNT(*) FROM recommendations WHERE rec_type='buy' AND dismissed=0").fetchone()[0]
        sell_count = conn.execute("SELECT COUNT(*) FROM recommendations WHERE rec_type='sell' AND dismissed=0").fetchone()[0]
        sell_pp_row = conn.execute("SELECT COALESCE(SUM(estimated_price), 0) as t FROM recommendations WHERE rec_type='sell' AND dismissed=0").fetchone()
        sell_pp = sell_pp_row['t'] or 0

        st.caption(f"{buy_count} buy recs | {sell_count} sell recs")
        if sell_pp > 0:
            st.caption(f"Sellable: {sell_pp:,} PP")

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
# IMPORT STATUS STRIP — lives on Dashboard, links to Data Refresh page
# The old inline import expander (80+ lines of ingest_file + file_uploader)
# moved to app/pages/0_Data_Refresh.py. Dashboard now shows a thin "is
# your data fresh?" strip and a button to the refresh page.
# ============================================================
if logs:
    latest_import_raw = logs[0]['last_import']
    try:
        latest_dt = datetime.fromisoformat(str(latest_import_raw).replace("T", " "))
        age_hours = (datetime.now() - latest_dt).total_seconds() / 3600.0
        if age_hours < 1:
            age_lbl = f"{int(age_hours * 60)}m ago"
        elif age_hours < 24:
            age_lbl = f"{int(age_hours)}h ago"
        else:
            age_lbl = f"{int(age_hours / 24)}d ago"
    except Exception:
        age_lbl = str(latest_import_raw)[:16]
        age_hours = 0

    stale = age_hours >= 24
    status_cols = st.columns([5, 2])
    with status_cols[0]:
        if stale:
            st.warning(
                f"\u23f0 Last refresh: **{age_lbl}** — your data may be stale. "
                "Run a fresh import before acting on recommendations."
            )
        else:
            st.caption(f"\u2714 Last refresh: {age_lbl}")
    with status_cols[1]:
        st.page_link(
            "pages/0_Data_Refresh.py",
            label="Open Data Refresh",
            icon="\U0001f4e5",
            use_container_width=True,
        )
else:
    st.info(
        "\U0001f449 No data yet. Open **Data Refresh** to scan your OOTP "
        "watch folder and run your first import."
    )
    st.page_link(
        "pages/0_Data_Refresh.py",
        label="Open Data Refresh",
        icon="\U0001f4e5",
    )
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

# Team Health now benchmarks against the active tier's P75 (a competitive
# roster at this tier), NOT hardcoded Diamond+ averages. Previously every
# Low Bronze roster showed 45% health by construction — the benchmark was
# Diamond meta. Source switches to tier_context so the bar scales with the
# league the user is actually playing in.
try:
    from app.core.tier_context import tier_benchmarks as _tb, tier_meta_p50 as _tp50
    _tier_bench = _tb(conn=conn)
    _tier_norm_header = _tier_bench.get('tier_normalized') or 'Bronze'
    avg_market_top = (_tier_bench.get('overall') or {}).get('p75') or _tp50(_tier_norm_header, conn=conn) + 80
except Exception:
    # Fallback if tier_context can't read leagues table — use the old
    # Diamond+ benchmark so the page still renders.
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
    # Team Health Progress Bars — tier-relative, not "vs Diamond+"
    try:
        from app.core.tier_context import tier_benchmarks, promotion_readiness, tier_meta_p50

        # Pull tier benchmarks once; the UI stays readable even when the
        # sample is thin (falls back to heuristic P50 per tier).
        _bench = tier_benchmarks(conn=conn)
        _tier_norm = _bench.get('tier_normalized') or 'Bronze'
        _tier_label = _bench.get('league_tier') or _tier_norm
        # P75 of the roster's own tier is the "competitive" benchmark — a
        # card at that meta is a solid contributor in this tier. For Low
        # Bronze this lands around 580-640 rather than the Diamond+ 850.
        _tier_p75 = (_bench.get('overall') or {}).get('p75') or tier_meta_p50(_tier_norm, conn=conn) + 80

        bat_starter_meta_row = conn.execute("""
            SELECT AVG(meta_score) as avg_meta FROM roster_current
            WHERE lineup_role = 'starter' AND position NOT IN ('SP', 'RP', 'CL') AND meta_score > 0
        """).fetchone()
        bat_starter_meta = bat_starter_meta_row['avg_meta'] or 0 if bat_starter_meta_row else 0
        batting_depth_pct = min(100, int((bat_starter_meta / _tier_p75) * 100)) if _tier_p75 > 0 else 0

        pit_starter_meta_row = conn.execute("""
            SELECT AVG(meta_score) as avg_meta FROM roster_current
            WHERE lineup_role IN ('rotation', 'closer', 'bullpen') AND meta_score > 0
        """).fetchone()
        pit_starter_meta = pit_starter_meta_row['avg_meta'] or 0 if pit_starter_meta_row else 0
        pitching_depth_pct = min(100, int((pit_starter_meta / _tier_p75) * 100)) if _tier_p75 > 0 else 0

        # "Good arm" threshold is tier-relative: the P50 of owned cards at
        # this tier. Previously hardcoded at 400, which meant no bullpen
        # arm in Silver+ ever counted as good (the bar was too low) AND no
        # bullpen arm in Stone ever counted as good (the bar was too high).
        _bp_good_threshold = tier_meta_p50(_tier_norm, conn=conn)
        bp_good = conn.execute("""
            SELECT COUNT(*) as c FROM roster_current
            WHERE position IN ('RP', 'CL') AND lineup_role IN ('bullpen', 'closer') AND meta_score > ?
        """, (_bp_good_threshold,)).fetchone()['c']
        bp_total = conn.execute("""
            SELECT COUNT(*) as c FROM roster_current
            WHERE position IN ('RP', 'CL') AND lineup_role IN ('bullpen', 'closer')
        """).fetchone()['c']
        bullpen_pct = int((bp_good / bp_total) * 100) if bp_total > 0 else 0

        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.markdown("**Overall**")
            st.progress(min(1.0, health_pct / 100))
            st.caption(f"{health_pct}% of {_tier_label} P75")
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

    # ── Promotion Readiness ──
    # Surfaces the gap between the current roster and the next tier's
    # competitive threshold, with a per-position breakdown so the user
    # knows *where* to spend PP for the push.
    try:
        from app.core.tier_context import promotion_readiness
        readiness = promotion_readiness(conn=conn)
        if readiness.get('target_tier'):
            current_t = readiness.get('current_tier') or readiness.get('current_tier_normalized') or '?'
            target_t = readiness['target_tier']
            gap = readiness.get('overall_gap')
            threshold = readiness.get('target_threshold_p75')
            st.divider()
            st.markdown(f"#### 🎯 Promotion Readiness — {current_t} → {target_t}")
            pr1, pr2, pr3 = st.columns(3)
            with pr1:
                roster_p50 = readiness.get('roster_p50') or 0
                st.metric("Roster P50 meta", f"{roster_p50:.0f}")
            with pr2:
                if threshold:
                    st.metric(f"{target_t} threshold", f"{threshold:.0f}")
                else:
                    st.metric(f"{target_t} threshold", "—")
            with pr3:
                if gap is not None:
                    if gap <= 0:
                        st.metric("Overall gap", "Met", delta="ready for promotion",
                                  delta_color="normal")
                    else:
                        st.metric("Overall gap", f"+{gap:.0f} meta",
                                  delta="needed to close",
                                  delta_color="inverse")
                else:
                    st.metric("Overall gap", "—")

            # Position-level investment priorities — sorted so the biggest
            # gaps surface first. Only show slots the roster currently fills
            # (empty slots are already flagged by the Weakest Pos metric).
            positions = readiness.get('positions') or {}
            ordered = sorted(
                ((pos, d) for pos, d in positions.items() if d.get('gap') is not None),
                key=lambda x: -(x[1].get('gap') or 0),
            )[:6]
            if ordered:
                invest_lines = []
                for pos, d in ordered:
                    cur = d.get('current_meta') or 0
                    tgt = d.get('target_meta') or 0
                    gp = d.get('gap') or 0
                    prio = d.get('priority') or 'unknown'
                    emoji = {
                        'met': '✅', 'low': '🟢', 'medium': '🟡',
                        'high': '🔴', 'unknown': '⚪',
                    }.get(prio, '⚪')
                    invest_lines.append(
                        f"- {emoji} **{pos}** — current {cur:.0f} / target {tgt:.0f} "
                        f"({'+' if gp > 0 else ''}{gp:.0f})"
                    )
                st.markdown("\n".join(invest_lines))
    except Exception:
        # Readiness is a nice-to-have, don't block the page on it.
        pass

    # ── Roster Momentum (hot/cold from history snapshots) ──
    # Computed from player_history: how has each roster card's PERFORMANCE
    # moved across the last 3 snapshots? Independent signal from meta —
    # meta is static, momentum says "this guy is turning around" or
    # "this guy is sliding" based on actual game output.
    try:
        from app.core.trend import roster_trends, summarize_trends
        _trends = roster_trends(window=3, conn=conn)
        _summary = summarize_trends(_trends)
        if _summary['hot'] + _summary['cold'] > 0:
            st.divider()
            st.markdown("#### 📈 Roster Momentum (last 3 snapshots)")
            mm1, mm2, mm3 = st.columns(3)
            with mm1:
                st.metric("🔥 Hot", _summary['hot'],
                          delta=f"trending up", delta_color="normal")
            with mm2:
                st.metric("🧊 Cold", _summary['cold'],
                          delta=f"trending down", delta_color="inverse")
            with mm3:
                st.metric("— Stable", _summary['stable'])

            # Pull top 3 hot and cold by war/p_war delta, show inline
            hot_sorted = sorted(
                ((cid, t) for cid, t in _trends.items() if t.get('signal') == 'hot'),
                key=lambda x: -(x[1].get('war_delta') or x[1].get('p_war_delta') or 0),
            )[:5]
            cold_sorted = sorted(
                ((cid, t) for cid, t in _trends.items() if t.get('signal') == 'cold'),
                key=lambda x: (x[1].get('war_delta') or x[1].get('p_war_delta') or 0),
            )[:5]
            # Resolve names
            def _name_for(cid):
                row = conn.execute("SELECT card_title FROM cards WHERE card_id = ?", (cid,)).fetchone()
                return (row['card_title'] if row and row['card_title'] else f"card {cid}")[:44]
            if hot_sorted:
                st.markdown("**Hottest 5:**")
                for cid, t in hot_sorted:
                    w = t.get('war_delta') or t.get('p_war_delta') or 0
                    bat = t.get('ops_plus_delta')
                    pit = t.get('era_plus_delta')
                    rate = (f"OPS+ {bat:+.0f}" if bat is not None else
                            (f"ERA+ {pit:+.0f}" if pit is not None else ""))
                    st.markdown(f"- 🔥 {_name_for(cid)} — WAR {w:+.1f}  {rate}")
            if cold_sorted:
                st.markdown("**Coldest 5:**")
                for cid, t in cold_sorted:
                    w = t.get('war_delta') or t.get('p_war_delta') or 0
                    bat = t.get('ops_plus_delta')
                    pit = t.get('era_plus_delta')
                    rate = (f"OPS+ {bat:+.0f}" if bat is not None else
                            (f"ERA+ {pit:+.0f}" if pit is not None else ""))
                    st.markdown(f"- 🧊 {_name_for(cid)} — WAR {w:+.1f}  {rate}")
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
                ts = str(latest_insight['created_at'])[:16].replace('T', ' ')
                st.caption(f"Generated: {ts}")
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
        'market': 'Card Shop',
        'roster_batting': 'Roster Batting Ratings',
        'roster_pitching': 'Roster Pitching Ratings',
        'collection_batting': 'Collection Batting',
        'collection_pitching': 'Collection Pitching',
        'stats_batting': 'League Batting Stats',
        'stats_pitching': 'League Pitching Stats',
        'roster_batting_stats': 'Roster Batting Stats',
        'roster_pitching_stats': 'Roster Pitching Stats',
        'roster_batting_stats_adv': 'Batting Stats (Adv)',
        'roster_pitching_stats_adv': 'Pitching Stats (Adv)',
        'stats_batting_ratings': 'League Batting Ratings',
        'stats_pitching_ratings': 'League Pitching Ratings',
        'lineup_vs_rhp': 'Lineup vs RHP',
        'lineup_vs_lhp': 'Lineup vs LHP',
        'lineup_overview': 'Lineup Overview',
        'team_pitching': 'Pitching Staff',
        'fielding_stats': 'Fielding Stats',
        'position_ratings': 'Position Ratings',
        'pitch_ratings': 'Pitch Ratings',
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
