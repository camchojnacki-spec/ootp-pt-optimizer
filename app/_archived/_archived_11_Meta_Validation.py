"""Meta Validation — does our meta formula actually predict in-game performance?"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.core.database import get_connection, load_config
from app.core.meta_validation import (
    validate_meta_vs_performance, get_meta_accuracy_score,
    get_stats_summary,
)

st.set_page_config(page_title="Meta Validation", page_icon="🔬", layout="wide")
st.title("Meta Validation")
st.caption("Does our meta formula actually predict who plays well? Import batting & pitching stat CSVs to find out.")

conn = get_connection()

# --- Stats Summary (quick view of what data we have) ---
stats = get_stats_summary(conn)
if stats["has_batting_stats"] or stats["has_pitching_stats"]:
    scol1, scol2, scol3, scol4, scol5 = st.columns(5)
    scol1.metric("Batters Tracked", stats["batting_count"])
    scol2.metric("Pitchers Tracked", stats["pitching_count"])
    scol3.metric("Team AVG", f".{int((stats['team_avg'] or 0) * 1000):03d}" if stats["team_avg"] else "—")
    scol4.metric("Team OPS", f"{stats['team_ops']:.3f}" if stats["team_ops"] else "—")
    scol5.metric("Team ERA", f"{stats['team_era']:.2f}" if stats["team_era"] else "—")

    # MVP and Cy Young
    mcol1, mcol2 = st.columns(2)
    mvp = stats.get("mvp")
    if mvp:
        mcol1.info(
            f"**MVP: {mvp['player_name']}** ({mvp['position']}) — "
            f".{int(mvp['avg'] * 1000):03d} AVG, {mvp['ops']:.3f} OPS, "
            f"{mvp['hr']} HR, {mvp['rbi']} RBI, {mvp['war']:.1f} WAR"
        )
    cy = stats.get("cy_young")
    if cy:
        mcol2.info(
            f"**Cy Young: {cy['player_name']}** ({cy['position']}) — "
            f"{cy['era']:.2f} ERA, {cy['wins']}W-{cy['losses']}L, "
            f"{cy['k']} K, {cy['war']:.1f} WAR"
        )
    st.divider()
else:
    st.warning(
        "No game stats found. Export your **batting stats** and **pitching stats** CSVs "
        "from OOTP's sortable stats page and import them on the main dashboard."
    )

# --- Accuracy Summary ---
try:
    accuracy = get_meta_accuracy_score(conn)
except Exception:
    accuracy = None

if accuracy and accuracy.get("sample_size", 0) > 0:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy Score", f"{accuracy['accuracy_pct']:.0f}%")
    col2.metric("Sample Size", f"{accuracy['sample_size']} players")

    top_over = accuracy.get("top_overperformer")
    col3.metric("Top Overperformer",
                top_over['player_name'] if top_over else "—",
                delta=f"gap: {top_over['gap']:.0f}" if top_over else None,
                delta_color="off",
                help="Playing better than meta predicts")

    top_under = accuracy.get("top_underperformer")
    col4.metric("Top Underperformer",
                top_under['player_name'] if top_under else "—",
                delta=f"gap: +{top_under['gap']:.0f}" if top_under else None,
                delta_color="off",
                help="Playing worse than meta predicts")

    st.divider()

# --- Full Validation ---
if st.button("Run Full Validation", type="primary"):
    with st.spinner("Running validation..."):
        try:
            result = validate_meta_vs_performance(conn)
        except Exception as e:
            st.error(f"Validation failed: {e}")
            result = None

    # Generate passive AI meta insight
    if result and result.get("players"):
        try:
            from app.core.ai_advisor import generate_meta_insight
            generate_meta_insight(result, conn)
        except Exception:
            pass  # AI insight is optional

    if result is None or not result.get("players"):
        msg = result.get('message', '') if result else ''
        st.info(f"No matched data. {msg}")
    else:
        players = result["players"]
        batter_count = result.get("batter_count", 0)
        pitcher_count = result.get("pitcher_count", 0)

        st.success(result["message"])

        # --- Correlation Summary ---
        cc1, cc2, cc3, cc4 = st.columns(4)
        correlation = result.get("correlation", 0)
        rank_corr = result.get("rank_correlation", 0)
        bat_corr = result.get("batting_correlation", 0)
        pitch_corr = result.get("pitching_correlation", 0)

        cc1.metric("Overall Pearson r", f"{correlation:.3f}")
        cc2.metric("Overall Rank r", f"{rank_corr:.3f}")
        cc3.metric("Batting Correlation", f"{bat_corr:.3f}")
        cc4.metric("Pitching Correlation", f"{pitch_corr:.3f}")

        if correlation >= 0.6:
            interpretation = "Strong positive — meta is working well"
        elif correlation >= 0.3:
            interpretation = "Moderate positive — meta has some predictive value"
        elif correlation >= 0:
            interpretation = "Weak positive — meta needs adjustment"
        else:
            interpretation = "Negative — meta is backwards, weights need rework"
        st.info(interpretation)

        # --- Tabs: Batters / Pitchers / All ---
        tab_bat, tab_pitch, tab_all = st.tabs([
            f"Batters ({batter_count})",
            f"Pitchers ({pitcher_count})",
            f"All Players ({len(players)})"
        ])

        batters = [p for p in players if p["player_type"] == "batter"]
        pitchers = [p for p in players if p["player_type"] == "pitcher"]

        with tab_bat:
            if batters:
                # Scatter plot with over/underperformance coloring
                bat_df = pd.DataFrame(batters)
                bat_df["Performance"] = bat_df["meta_vs_perf_gap"].apply(
                    lambda g: "Overperformer" if g < 0 else "Underperformer"
                )
                bat_df["OPS"] = bat_df.apply(lambda r: r.get("in_game_ops", 0), axis=1)

                fig = px.scatter(
                    bat_df, x="meta_score", y="performance_rating",
                    color="Performance",
                    color_discrete_map={"Overperformer": "#2ecc71", "Underperformer": "#e74c3c"},
                    hover_name="player_name",
                    hover_data={
                        "position": True,
                        "meta_score": ":.0f",
                        "performance_rating": ":.0f",
                        "OPS": ":.3f",
                        "Performance": False,
                    },
                    labels={
                        "meta_score": "Meta Score",
                        "performance_rating": "In-Game OPS x 1000",
                        "position": "Position",
                        "OPS": "OPS",
                    },
                    title="Batters: Meta Score vs In-Game Performance",
                )
                fig.update_traces(marker=dict(size=10), textposition="top center")

                # y=x reference line
                all_vals = bat_df["meta_score"].tolist() + bat_df["performance_rating"].tolist()
                ref_min, ref_max = min(all_vals), max(all_vals)
                fig.add_trace(go.Scatter(
                    x=[ref_min, ref_max], y=[ref_min, ref_max],
                    mode="lines", line=dict(color="gray", dash="dash", width=1),
                    name="y = x (perfect prediction)", showlegend=True,
                ))

                # Regression trend line
                if len(bat_df) >= 2:
                    xs = bat_df["meta_score"].values
                    ys = bat_df["performance_rating"].values
                    coeffs = np.polyfit(xs, ys, 1)
                    x_min, x_max = xs.min(), xs.max()
                    fig.add_trace(go.Scatter(
                        x=[x_min, x_max],
                        y=[coeffs[0] * x_min + coeffs[1], coeffs[0] * x_max + coeffs[1]],
                        mode="lines", line=dict(color="red", dash="dot", width=2),
                        name=f"Regression (slope={coeffs[0]:.2f})",
                    ))

                fig.update_layout(height=550, xaxis_title="Meta Score", yaxis_title="In-Game OPS x 1000")
                st.plotly_chart(fig, use_container_width=True)

                # Table
                table_data = []
                for p in sorted(batters, key=lambda x: x['meta_vs_perf_gap'], reverse=True):
                    table_data.append({
                        "Player": p['player_name'],
                        "Pos": p['position'],
                        "Meta": f"{p['meta_score']:.0f}",
                        "AVG": f"{p.get('in_game_avg', 0):.3f}",
                        "OPS": f"{p.get('in_game_ops', 0):.3f}",
                        "OPS+": p.get('in_game_ops_plus', 0),
                        "HR": p.get('in_game_hr', 0),
                        "RBI": p.get('in_game_rbi', 0),
                        "WAR": p.get('in_game_war', 0),
                        "Games": p.get('games', 0),
                        "Perf Rating": f"{p['performance_rating']:.0f}",
                        "Gap": f"{p['meta_vs_perf_gap']:.0f}",
                    })
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
            else:
                st.info("No batting data matched.")

        with tab_pitch:
            if pitchers:
                # Scatter plot with over/underperformance coloring
                pit_df = pd.DataFrame(pitchers)
                pit_df["Performance"] = pit_df["meta_vs_perf_gap"].apply(
                    lambda g: "Overperformer" if g < 0 else "Underperformer"
                )
                pit_df["ERA"] = pit_df.apply(lambda r: r.get("in_game_era", 0), axis=1)
                pit_df["ERA_plus"] = pit_df.apply(lambda r: r.get("in_game_era_plus", 0), axis=1)

                fig = px.scatter(
                    pit_df, x="meta_score", y="performance_rating",
                    color="Performance",
                    color_discrete_map={"Overperformer": "#2ecc71", "Underperformer": "#e74c3c"},
                    hover_name="player_name",
                    hover_data={
                        "position": True,
                        "meta_score": ":.0f",
                        "performance_rating": ":.0f",
                        "ERA": ":.2f",
                        "ERA_plus": ":.0f",
                        "Performance": False,
                    },
                    labels={
                        "meta_score": "Meta Score",
                        "performance_rating": "Performance Rating (ERA+ x 5)",
                        "position": "Position",
                        "ERA": "ERA",
                        "ERA_plus": "ERA+",
                    },
                    title="Pitchers: Meta Score vs In-Game Performance",
                )
                fig.update_traces(marker=dict(size=10), textposition="top center")

                # y=x reference line
                all_vals = pit_df["meta_score"].tolist() + pit_df["performance_rating"].tolist()
                ref_min, ref_max = min(all_vals), max(all_vals)
                fig.add_trace(go.Scatter(
                    x=[ref_min, ref_max], y=[ref_min, ref_max],
                    mode="lines", line=dict(color="gray", dash="dash", width=1),
                    name="y = x (perfect prediction)", showlegend=True,
                ))

                # Regression trend line
                if len(pit_df) >= 2:
                    xs = pit_df["meta_score"].values
                    ys = pit_df["performance_rating"].values
                    coeffs = np.polyfit(xs, ys, 1)
                    x_min, x_max = xs.min(), xs.max()
                    fig.add_trace(go.Scatter(
                        x=[x_min, x_max],
                        y=[coeffs[0] * x_min + coeffs[1], coeffs[0] * x_max + coeffs[1]],
                        mode="lines", line=dict(color="red", dash="dot", width=2),
                        name=f"Regression (slope={coeffs[0]:.2f})",
                    ))

                fig.update_layout(height=550, xaxis_title="Meta Score", yaxis_title="Performance Rating (ERA+ x 5)")
                st.plotly_chart(fig, use_container_width=True)

                # Table
                table_data = []
                for p in sorted(pitchers, key=lambda x: x['meta_vs_perf_gap'], reverse=True):
                    table_data.append({
                        "Player": p['player_name'],
                        "Pos": p['position'],
                        "Meta": f"{p['meta_score']:.0f}",
                        "ERA": p.get('in_game_era', 0),
                        "WHIP": p.get('in_game_whip', 0),
                        "K/9": p.get('in_game_k_per_9', 0),
                        "ERA+": p.get('in_game_era_plus', 0),
                        "FIP": p.get('in_game_fip', 0),
                        "W-L": f"{p.get('in_game_wins', 0)}-{p.get('in_game_losses', 0)}",
                        "WAR": p.get('in_game_war', 0),
                        "IP": p.get('ip', 0),
                        "Perf Rating": f"{p['performance_rating']:.0f}",
                        "Gap": f"{p['meta_vs_perf_gap']:.0f}",
                    })
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
            else:
                st.info("No pitching data matched.")

        with tab_all:
            # Combined over/underperformers
            overperformers = result.get("overperformers", [])
            if overperformers:
                st.subheader(f"Overperformers ({len(overperformers)}) — meta underrates them")
                over_data = []
                for p in overperformers[:15]:
                    row = {
                        "Player": p['player_name'],
                        "Pos": p['position'],
                        "Type": p['player_type'].title(),
                        "Meta": f"{p['meta_score']:.0f}",
                        "Perf": f"{p['performance_rating']:.0f}",
                        "Gap": f"{p['meta_vs_perf_gap']:.0f}",
                    }
                    if p['player_type'] == 'batter':
                        row["Key Stat"] = f".{int(p.get('in_game_ops', 0) * 1000):03d} OPS, {p.get('in_game_war', 0):.1f} WAR"
                    else:
                        row["Key Stat"] = f"{p.get('in_game_era', 0):.2f} ERA, {p.get('in_game_war', 0):.1f} WAR"
                    over_data.append(row)
                st.dataframe(pd.DataFrame(over_data), use_container_width=True, hide_index=True)

            underperformers = result.get("underperformers", [])
            if underperformers:
                st.subheader(f"Underperformers ({len(underperformers)}) — meta overrates them")
                under_data = []
                for p in underperformers[:15]:
                    row = {
                        "Player": p['player_name'],
                        "Pos": p['position'],
                        "Type": p['player_type'].title(),
                        "Meta": f"{p['meta_score']:.0f}",
                        "Perf": f"{p['performance_rating']:.0f}",
                        "Gap": f"+{p['meta_vs_perf_gap']:.0f}",
                    }
                    if p['player_type'] == 'batter':
                        row["Key Stat"] = f".{int(p.get('in_game_ops', 0) * 1000):03d} OPS, {p.get('in_game_war', 0):.1f} WAR"
                    else:
                        row["Key Stat"] = f"{p.get('in_game_era', 0):.2f} ERA, {p.get('in_game_war', 0):.1f} WAR"
                    under_data.append(row)
                st.dataframe(pd.DataFrame(under_data), use_container_width=True, hide_index=True)

        # --- Weight Suggestions ---
        st.divider()
        st.subheader("Weight Adjustment Suggestions")
        suggestions = result.get("weight_suggestions", {})

        actual_suggestions = {k: v for k, v in suggestions.items()
                              if isinstance(v, dict) and 'adjustment' in v}

        if actual_suggestions:
            sugg_data = []
            for comp, info in actual_suggestions.items():
                sugg_data.append({
                    "Rating": comp.replace('_', ' ').title(),
                    "Current Weight": info.get('current_weight', '—'),
                    "Correlation": f"{info.get('correlation', 0):.3f}",
                    "Suggested Change": f"{info.get('adjustment', 0):+.2f}",
                    "Reasoning": info.get('reason', ''),
                })
            st.dataframe(pd.DataFrame(sugg_data), use_container_width=True, hide_index=True)

            if st.button("Apply Suggestions to Settings"):
                config = load_config()
                bw = config.get('batting_weights', {})
                changes = []
                for comp, info in actual_suggestions.items():
                    key = comp
                    if key in bw:
                        old_val = bw[key]
                        new_val = round(old_val + info['adjustment'], 2)
                        bw[key] = new_val
                        changes.append(f"{comp}: {old_val:.2f} -> {new_val:.2f}")
                config['batting_weights'] = bw

                import yaml
                config_path = Path(__file__).parent.parent.parent / "config.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                st.success("Weights updated! Changes: " + ", ".join(changes))
                st.info("Re-import data to recalculate meta scores with the new weights.")
        else:
            msg = suggestions.get('message', 'No weight adjustments suggested — current weights look reasonable, or insufficient data.')
            st.info(msg)

# --- AI Meta Insight (passive — generated after validation runs) ---
try:
    ai_table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ai_insights'"
    ).fetchone()
    if ai_table_exists:
        meta_insight = conn.execute("""
            SELECT content, created_at FROM ai_insights
            WHERE insight_type = 'meta_insight'
            ORDER BY created_at DESC LIMIT 1
        """).fetchone()
        if meta_insight and meta_insight['content']:
            st.divider()
            st.subheader("AI Meta Insight")
            st.info(meta_insight['content'])
            st.caption(f"Generated: {str(meta_insight['created_at'])[:19]}")

        # Calibration insight — regression weights vs current weights
        cal_insight = conn.execute("""
            SELECT content, created_at FROM ai_insights
            WHERE insight_type = 'calibration'
            ORDER BY created_at DESC LIMIT 1
        """).fetchone()
        if cal_insight and cal_insight['content']:
            st.divider()
            st.subheader("Regression-Based Weight Calibration")
            st.code(cal_insight['content'])
            st.caption(f"Last calibrated: {str(cal_insight['created_at'])[:19]}")

        # Also show live calibration comparison if enough data
        try:
            from app.core.meta_calibration import get_calibration_comparison
            comparison = get_calibration_comparison(conn)

            bat_comp = comparison.get("batting", {})
            pit_comp = comparison.get("pitching", {})

            has_bat = bat_comp.get("sample_size", 0) >= 5 and bat_comp.get("suggested")
            has_pit = pit_comp.get("sample_size", 0) >= 5 and pit_comp.get("suggested")

            if has_bat or has_pit:
                st.divider()
                st.subheader("Live Calibration: Current vs Regression Weights")

                if has_bat:
                    st.caption(f"Batting (n={bat_comp['sample_size']}, R2={bat_comp['r_squared']:.3f})")
                    bat_rows = []
                    for key in sorted(set(list(bat_comp['current'].keys()) + list(bat_comp['suggested'].keys()))):
                        cur = bat_comp['current'].get(key, 0)
                        sug = bat_comp['suggested'].get(key, 0)
                        diff = sug - cur
                        bat_rows.append({
                            "Rating": key.replace("_", " ").title(),
                            "Current": f"{cur:.2f}",
                            "Regression": f"{sug:.2f}",
                            "Diff": f"{diff:+.2f}",
                        })
                    st.dataframe(pd.DataFrame(bat_rows), use_container_width=True, hide_index=True)

                if has_pit:
                    st.caption(f"Pitching (n={pit_comp['sample_size']}, R2={pit_comp['r_squared']:.3f})")
                    pit_rows = []
                    for key in sorted(set(list(pit_comp['current'].keys()) + list(pit_comp['suggested'].keys()))):
                        cur = pit_comp['current'].get(key, 0)
                        sug = pit_comp['suggested'].get(key, 0)
                        diff = sug - cur
                        pit_rows.append({
                            "Rating": key.replace("_", " ").title(),
                            "Current": f"{cur:.2f}",
                            "Regression": f"{sug:.2f}",
                            "Diff": f"{diff:+.2f}",
                        })
                    st.dataframe(pd.DataFrame(pit_rows), use_container_width=True, hide_index=True)
        except Exception:
            pass  # Calibration module may not be available
except Exception:
    pass  # AI insights are optional

conn.close()
