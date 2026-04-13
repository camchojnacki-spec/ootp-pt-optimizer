"""Trends & History -- visualize player history and meta trends across leagues."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection
from app.core.history import get_meta_movers, get_market_trends, get_player_trend

st.set_page_config(page_title="Trends & History", page_icon="\U0001f4c8", layout="wide")
st.title("\U0001f4c8 Trends & History")

# ============================================================================
# Section 1: League Overview
# ============================================================================
st.header("League Overview")

try:
    conn = get_connection()
    leagues = conn.execute("""
        SELECT l.league_id, l.league_name, l.league_tier, l.start_date,
               l.end_date, l.final_record, l.team_name
        FROM leagues l
        ORDER BY l.created_at DESC
    """).fetchall()

    if not leagues:
        st.info("No leagues found yet -- import your first export to get started.")
    else:
        # Count exports and date range per league from export_log
        league_stats = {}
        for lg in leagues:
            lid = lg["league_id"]
            stats = conn.execute("""
                SELECT COUNT(*) as num_exports,
                       MIN(created_at) as first_export,
                       MAX(created_at) as last_export,
                       MAX(team_record) as latest_record
                FROM export_log
                WHERE league_id = ?
            """, (lid,)).fetchone()
            league_stats[lid] = stats

        cols = st.columns(min(len(leagues), 3))
        for i, lg in enumerate(leagues):
            lid = lg["league_id"]
            stats = league_stats.get(lid)
            col = cols[i % len(cols)]
            with col:
                tier = lg["league_tier"] or "Unknown"
                name = lg["league_name"] or lid
                st.subheader(f"{name}")
                st.caption(f"Tier: {tier}")

                record = lg["final_record"] or (stats["latest_record"] if stats else None)
                if record:
                    st.metric("Record", record)

                num_exports = stats["num_exports"] if stats else 0
                st.metric("Exports", num_exports)

                first = stats["first_export"] if stats else lg["start_date"]
                last = stats["last_export"] if stats else lg["end_date"]
                if first and last:
                    st.text(f"{str(first)[:10]}  to  {str(last)[:10]}")
                elif first:
                    st.text(f"Since {str(first)[:10]}")

                if lg["end_date"]:
                    st.caption("Completed")
                else:
                    st.caption("Active")
                st.divider()

    conn.close()
except Exception as e:
    st.warning(f"Could not load league data: {e}")

# ============================================================================
# Section 2: Player Lookup
# ============================================================================
st.header("Player Lookup")

search = st.text_input("Search player name", placeholder="e.g. Babe Ruth, Pedro Martinez")

if search and len(search.strip()) >= 2:
    try:
        trend_data = get_player_trend(player_name=search.strip())

        if not trend_data:
            st.info(f"No history found for '{search}'. Player must appear in at least one export snapshot.")
        else:
            # Build DataFrame
            df = pd.DataFrame(trend_data)

            # Determine if pitcher or batter based on available stats
            has_era = df["era"].notna().any() if "era" in df.columns else False

            # --- Meta Score Chart ---
            st.subheader("Meta Score Over Time")

            if len(df) >= 2:
                chart_df = df[["snapshot_date", "meta_score", "meta_vs_rhp", "meta_vs_lhp"]].copy()
                chart_df["snapshot_date"] = pd.to_datetime(chart_df["snapshot_date"])
                chart_df = chart_df.set_index("snapshot_date").sort_index()
                chart_df.columns = ["Meta Score", "vs RHP", "vs LHP"]
                st.line_chart(chart_df)
            else:
                latest = df.iloc[-1]
                c1, c2, c3 = st.columns(3)
                c1.metric("Meta Score", f"{latest['meta_score']:.1f}" if pd.notna(latest["meta_score"]) else "N/A")
                c2.metric("vs RHP", f"{latest['meta_vs_rhp']:.1f}" if pd.notna(latest.get("meta_vs_rhp")) else "N/A")
                c3.metric("vs LHP", f"{latest['meta_vs_lhp']:.1f}" if pd.notna(latest.get("meta_vs_lhp")) else "N/A")
                st.caption("Only one snapshot available -- chart requires 2+ data points.")

            # --- Snapshot Table ---
            st.subheader("Snapshot History")

            display_cols = ["export_number", "league_id", "meta_score", "card_value", "sell_order_low"]
            if has_era:
                display_cols += ["era", "whip", "k_per_9", "p_war"]
                col_labels = {
                    "export_number": "Export #", "league_id": "League",
                    "meta_score": "Meta", "card_value": "Card Value",
                    "sell_order_low": "Sell Price", "era": "ERA",
                    "whip": "WHIP", "k_per_9": "K/9", "p_war": "WAR",
                }
            else:
                display_cols += ["ops", "ops_plus", "hr", "war"]
                col_labels = {
                    "export_number": "Export #", "league_id": "League",
                    "meta_score": "Meta", "card_value": "Card Value",
                    "sell_order_low": "Sell Price", "ops": "OPS",
                    "ops_plus": "OPS+", "hr": "HR", "war": "WAR",
                }

            # Only include columns that exist in the DataFrame
            valid_cols = [c for c in display_cols if c in df.columns]
            table_df = df[valid_cols].copy()
            table_df = table_df.rename(columns={c: col_labels.get(c, c) for c in valid_cols})

            st.dataframe(table_df, use_container_width=True, hide_index=True)

            # --- Cross-League Comparison ---
            unique_leagues = df["league_id"].dropna().unique()
            if len(unique_leagues) > 1:
                st.subheader("Cross-League Comparison")

                league_summary = []
                for lid in unique_leagues:
                    league_rows = df[df["league_id"] == lid]
                    summary = {
                        "League": lid,
                        "Snapshots": len(league_rows),
                        "Avg Meta": round(league_rows["meta_score"].mean(), 1),
                        "Min Meta": round(league_rows["meta_score"].min(), 1),
                        "Max Meta": round(league_rows["meta_score"].max(), 1),
                    }
                    if has_era and "era" in league_rows.columns:
                        era_vals = league_rows["era"].dropna()
                        if len(era_vals) > 0:
                            summary["Avg ERA"] = round(era_vals.mean(), 2)
                    elif "ops" in league_rows.columns:
                        ops_vals = league_rows["ops"].dropna()
                        if len(ops_vals) > 0:
                            summary["Avg OPS"] = round(ops_vals.mean(), 3)
                    league_summary.append(summary)

                st.dataframe(pd.DataFrame(league_summary), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error looking up player: {e}")

# ============================================================================
# Section 3: Meta Movers
# ============================================================================
st.header("Meta Movers")

try:
    movers = get_meta_movers()
    risers = movers.get("risers", [])
    fallers = movers.get("fallers", [])

    if not risers and not fallers:
        st.info("Need 2+ exports to show meta movers. Import more data to track changes over time.")
    else:
        col_rise, col_fall = st.columns(2)

        with col_rise:
            st.subheader("Risers")
            if risers:
                rise_df = pd.DataFrame(risers)
                display = rise_df[["player_name", "position", "meta_old", "meta_new", "delta", "pct_change"]].copy()
                display.columns = ["Player", "Pos", "Old Meta", "New Meta", "Change", "% Change"]
                display["Change"] = display["Change"].apply(lambda x: f"+{x:.1f}")
                display["% Change"] = display["% Change"].apply(lambda x: f"+{x:.1f}%")
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                st.caption("No risers in this period.")

        with col_fall:
            st.subheader("Fallers")
            if fallers:
                fall_df = pd.DataFrame(fallers)
                display = fall_df[["player_name", "position", "meta_old", "meta_new", "delta", "pct_change"]].copy()
                display.columns = ["Player", "Pos", "Old Meta", "New Meta", "Change", "% Change"]
                display["Change"] = display["Change"].apply(lambda x: f"{x:.1f}")
                display["% Change"] = display["% Change"].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                st.caption("No fallers in this period.")

        # Show date range context if available
        if risers:
            st.caption(f"Comparing {risers[0].get('old_date', '?')} to {risers[0].get('new_date', '?')}")
        elif fallers:
            st.caption(f"Comparing {fallers[0].get('old_date', '?')} to {fallers[0].get('new_date', '?')}")

except Exception as e:
    st.warning(f"Could not load meta movers: {e}")

# ============================================================================
# Section 4: Market Trends
# ============================================================================
st.header("Market Trends")

try:
    market = get_market_trends()
    price_risers = market.get("risers", [])
    price_fallers = market.get("fallers", [])

    if not price_risers and not price_fallers:
        st.info("Need 2+ exports with price data to show market trends. Keep importing to build history.")
    else:
        col_up, col_down = st.columns(2)

        with col_up:
            st.subheader("Price Risers")
            if price_risers:
                pr_df = pd.DataFrame(price_risers)
                display = pr_df[["player_name", "position", "price_old", "price_new",
                                 "price_delta", "pct_change", "meta_score"]].copy()
                display.columns = ["Player", "Pos", "Old Price", "New Price",
                                   "Change", "% Change", "Meta"]
                display["Change"] = display["Change"].apply(lambda x: f"+{x:,.0f}")
                display["% Change"] = display["% Change"].apply(lambda x: f"+{x:.1f}%")
                display["Old Price"] = pr_df["price_old"].apply(lambda x: f"{x:,.0f}")
                display["New Price"] = pr_df["price_new"].apply(lambda x: f"{x:,.0f}")
                display["Meta"] = display["Meta"].apply(lambda x: f"{x:.0f}")
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                st.caption("No price risers detected.")

        with col_down:
            st.subheader("Price Fallers")
            if price_fallers:
                pf_df = pd.DataFrame(price_fallers)
                display = pf_df[["player_name", "position", "price_old", "price_new",
                                 "price_delta", "pct_change", "meta_score"]].copy()
                display.columns = ["Player", "Pos", "Old Price", "New Price",
                                   "Change", "% Change", "Meta"]
                display["Change"] = display["Change"].apply(lambda x: f"{x:,.0f}")
                display["% Change"] = display["% Change"].apply(lambda x: f"{x:.1f}%")
                display["Old Price"] = pf_df["price_old"].apply(lambda x: f"{x:,.0f}")
                display["New Price"] = pf_df["price_new"].apply(lambda x: f"{x:,.0f}")
                display["Meta"] = display["Meta"].apply(lambda x: f"{x:.0f}")
                st.dataframe(display, use_container_width=True, hide_index=True)

                # Flag buy opportunities: high meta + falling price
                buys = pf_df[pf_df["meta_score"] >= 200]
                if len(buys) > 0:
                    st.success(
                        f"Buy opportunity: {len(buys)} card(s) with strong meta "
                        f"(200+) and falling price."
                    )
                    for _, row in buys.iterrows():
                        st.caption(
                            f"  {row['player_name']} -- Meta {row['meta_score']:.0f}, "
                            f"price dropped {row['pct_change']:.1f}%"
                        )
            else:
                st.caption("No price fallers detected.")

except Exception as e:
    st.warning(f"Could not load market trends: {e}")

# ============================================================================
# Section 5: Export History
# ============================================================================
st.header("Export History")

try:
    conn = get_connection()
    exports = conn.execute("""
        SELECT el.league_id, el.export_number, el.games_played,
               el.team_record, el.files_imported, el.notes, el.created_at,
               l.league_name, l.league_tier
        FROM export_log el
        LEFT JOIN leagues l ON l.league_id = el.league_id
        ORDER BY el.created_at DESC
    """).fetchall()
    conn.close()

    if not exports:
        st.info("No exports recorded yet. Import data to start building your timeline.")
    else:
        export_data = []
        for ex in exports:
            row = {
                "Date": str(ex["created_at"])[:16] if ex["created_at"] else "Unknown",
                "League": ex["league_name"] or ex["league_id"] or "Unknown",
                "Tier": ex["league_tier"] or "",
                "Export #": ex["export_number"],
                "Games Played": ex["games_played"] if ex["games_played"] else "",
                "Record": ex["team_record"] or "",
                "Files Imported": ex["files_imported"] if ex["files_imported"] else "",
            }
            if ex["notes"]:
                row["Notes"] = ex["notes"]
            export_data.append(row)

        export_df = pd.DataFrame(export_data)
        st.dataframe(export_df, use_container_width=True, hide_index=True)

        st.caption(f"Total exports: {len(exports)}")

except Exception as e:
    st.warning(f"Could not load export history: {e}")
