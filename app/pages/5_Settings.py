"""Settings page with meta weight presets."""
import streamlit as st
import pandas as pd
import yaml
import shutil
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import load_config, get_connection, init_db

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
st.title("Settings")

config_path = Path(__file__).parent.parent.parent / "config.yaml"
config = load_config()

# Team info
st.subheader("Team Info")
col1, col2 = st.columns(2)
with col1:
    team_name = st.text_input("Team Name", value=config.get('team_name', ''))
with col2:
    pp_budget = st.number_input("PP Budget", min_value=0, value=config.get('pp_budget', 500), step=50)

# ── Meta Weight Presets ──
st.subheader("Meta Weight Presets")
st.caption("Quick-load a team-building philosophy, then fine-tune below.")

PRESETS = {
    "Data-Driven (Default)": {
        "batting": {"gap_power": 1.40, "contact": 1.80, "avoid_ks": 0.00, "eye": 0.60, "power": 1.40, "babip": 0.00, "defense": 1.50, "ovr": 1.25},
        "pitching": {"movement": 2.40, "stuff": 1.40, "control": 0.20, "p_hr": 1.80, "ovr": 1.50, "stamina_hold": 0.40},
    },
    "Power Slugger": {
        "batting": {"gap_power": 1.80, "contact": 1.20, "avoid_ks": 0.00, "eye": 0.50, "power": 2.00, "babip": 0.00, "defense": 0.80, "ovr": 1.00},
        "pitching": {"movement": 2.20, "stuff": 1.60, "control": 0.20, "p_hr": 1.60, "ovr": 1.25, "stamina_hold": 0.30},
    },
    "Contact & Defense": {
        "batting": {"gap_power": 1.20, "contact": 2.00, "avoid_ks": 0.00, "eye": 0.80, "power": 1.00, "babip": 0.00, "defense": 2.00, "ovr": 1.25},
        "pitching": {"movement": 2.40, "stuff": 1.20, "control": 0.40, "p_hr": 1.80, "ovr": 1.50, "stamina_hold": 0.35},
    },
    "Pitching-Dominant": {
        "batting": {"gap_power": 1.30, "contact": 1.60, "avoid_ks": 0.00, "eye": 0.60, "power": 1.20, "babip": 0.00, "defense": 1.20, "ovr": 1.25},
        "pitching": {"movement": 2.80, "stuff": 1.60, "control": 0.20, "p_hr": 2.00, "ovr": 1.75, "stamina_hold": 0.50},
    },
    "OVR Trust": {
        "batting": {"gap_power": 1.20, "contact": 1.40, "avoid_ks": 0.00, "eye": 0.50, "power": 1.20, "babip": 0.00, "defense": 1.20, "ovr": 2.00},
        "pitching": {"movement": 2.00, "stuff": 1.20, "control": 0.20, "p_hr": 1.40, "ovr": 2.00, "stamina_hold": 0.40},
    },
}

preset_cols = st.columns(len(PRESETS))
for i, (preset_name, preset_vals) in enumerate(PRESETS.items()):
    with preset_cols[i]:
        if st.button(preset_name, use_container_width=True, key=f"preset_{i}"):
            st.session_state['preset_batting'] = preset_vals['batting']
            st.session_state['preset_pitching'] = preset_vals['pitching']
            st.rerun()

# Use preset values if just loaded, otherwise use config
active_batting = st.session_state.pop('preset_batting', None) or config.get('batting_weights', {})
active_pitching = st.session_state.pop('preset_pitching', None) or config.get('pitching_weights', {})

# Meta score weights
st.subheader("Batting Meta Weights")
bcol1, bcol2, bcol3, bcol4 = st.columns(4)
with bcol1:
    bw_gap = st.number_input("Gap Power", value=active_batting.get('gap_power', 1.65), step=0.05, format="%.2f")
    bw_contact = st.number_input("Contact", value=active_batting.get('contact', 1.55), step=0.05, format="%.2f")
with bcol2:
    bw_avk = st.number_input("Avoid K's", value=active_batting.get('avoid_ks', 1.35), step=0.05, format="%.2f")
    bw_eye = st.number_input("Eye", value=active_batting.get('eye', 1.20), step=0.05, format="%.2f")
with bcol3:
    bw_pow = st.number_input("Power", value=active_batting.get('power', 1.05), step=0.05, format="%.2f")
    bw_bab = st.number_input("BABIP", value=active_batting.get('babip', 0.80), step=0.05, format="%.2f")
with bcol4:
    bw_def = st.number_input("Defense", value=active_batting.get('defense', 0.50), step=0.05, format="%.2f")
    bw_ovr = st.number_input("OVR (Batting)", value=active_batting.get('ovr', 0.75), step=0.05, format="%.2f",
                               help="Anchors meta to the game's overall rating. Higher = trust OVR more.")

st.subheader("Pitching Meta Weights")
pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
with pcol1:
    pw_mov = st.number_input("Movement", value=active_pitching.get('movement', 2.00), step=0.05, format="%.2f")
with pcol2:
    pw_stu = st.number_input("Stuff", value=active_pitching.get('stuff', 1.50), step=0.05, format="%.2f")
with pcol3:
    pw_ctrl = st.number_input("Control", value=active_pitching.get('control', 1.00), step=0.05, format="%.2f")
with pcol4:
    pw_phr = st.number_input("pHR", value=active_pitching.get('p_hr', 0.80), step=0.05, format="%.2f")
with pcol5:
    pw_ovr = st.number_input("OVR (Pitching)", value=active_pitching.get('ovr', 1.00), step=0.05, format="%.2f",
                               help="Anchors meta to the game's overall rating.")
    pw_sh = st.number_input("Stamina/Hold", value=active_pitching.get('stamina_hold', 0.30), step=0.05, format="%.2f")

# Save settings
if st.button("Save Settings", type="primary"):
    config['team_name'] = team_name
    config['pp_budget'] = pp_budget
    config['batting_weights'] = {
        'gap_power': bw_gap, 'contact': bw_contact, 'avoid_ks': bw_avk,
        'eye': bw_eye, 'power': bw_pow, 'babip': bw_bab, 'defense': bw_def,
        'ovr': bw_ovr,
    }
    config['pitching_weights'] = {
        'movement': pw_mov, 'stuff': pw_stu, 'control': pw_ctrl, 'p_hr': pw_phr,
        'ovr': pw_ovr, 'stamina_hold': pw_sh,
    }

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    st.success("Settings saved! Recommendations will update on next data import.")

st.divider()

# ── Price Alerts ──
st.subheader("Price Alerts")
st.caption("Set target prices for cards. Alerts fire on the dashboard when a card hits your target.")

conn = get_connection()
init_db()  # Ensure price_alerts table exists

from app.core.price_alerts import create_alert, get_active_alerts, dismiss_alert, get_triggered_alerts

# Search for a card
alert_search = st.text_input("Search card by name", key="alert_search", placeholder="e.g. Mike Trout")

if alert_search:
    matches = conn.execute(
        "SELECT card_id, card_title, last_10_price, position_name, tier_name FROM cards WHERE card_title LIKE ? ORDER BY last_10_price DESC LIMIT 10",
        (f"%{alert_search}%",),
    ).fetchall()

    if matches:
        card_options = {f"{m['card_title']} ({m['position_name'] or '?'}, {m['tier_name'] or '?'}) — {m['last_10_price']:,} PP" if m['last_10_price'] else f"{m['card_title']} ({m['position_name'] or '?'}, {m['tier_name'] or '?'}) — no price": m for m in matches}
        selected_label = st.selectbox("Select card", list(card_options.keys()), key="alert_card_select")
        selected_card = card_options[selected_label]

        st.write(f"**Current price:** {selected_card['last_10_price']:,} PP" if selected_card['last_10_price'] else "**Current price:** unknown")

        acol1, acol2, acol3 = st.columns([1, 1, 1])
        with acol1:
            alert_type = st.selectbox("Alert type", ["below", "above"], key="alert_type",
                                       format_func=lambda x: "Price drops to or below" if x == "below" else "Price rises to or above")
        with acol2:
            default_target = (selected_card['last_10_price'] // 2) if selected_card['last_10_price'] else 100
            alert_target = st.number_input("Target Price (PP)", min_value=1, value=max(1, default_target), step=25, key="alert_target")
        with acol3:
            st.write("")
            st.write("")
            if st.button("Create Alert", type="primary", key="add_alert"):
                create_alert(selected_card['card_id'], alert_type, alert_target, conn=conn)
                st.success(f"Alert created: {selected_card['card_title']} {'below' if alert_type == 'below' else 'above'} {alert_target:,} PP")
                st.rerun()
    else:
        st.warning(f"No cards found matching '{alert_search}'")

# Active alerts table
active_alerts = get_active_alerts(conn=conn)
if active_alerts:
    st.markdown("**Active Alerts**")
    for i, a in enumerate(active_alerts):
        cols = st.columns([3, 1, 1, 1, 1])
        with cols[0]:
            st.write(a['card_title'])
        with cols[1]:
            st.write(f"{'Below' if a['alert_type'] == 'below' else 'Above'} {a['target_price']:,}")
        with cols[2]:
            st.write(f"Now: {a['last_10_price']:,}" if a['last_10_price'] else "Now: --")
        with cols[3]:
            st.write(str(a['created_at'])[:10])
        with cols[4]:
            if st.button("Dismiss", key=f"dismiss_{a['id']}"):
                dismiss_alert(a['id'], conn=conn)
                st.rerun()
else:
    st.info("No active price alerts. Search for a card above to create one.")

# Recently triggered alerts
triggered = get_triggered_alerts(conn=conn)
if triggered:
    st.markdown("**Recently Triggered (last 7 days)**")
    trig_data = []
    for t in triggered:
        trig_data.append({
            "Card": t['card_title'],
            "Type": "Below" if t['alert_type'] == 'below' else "Above",
            "Target": f"{t['target_price']:,}",
            "Current": f"{t['current_price']:,}" if t['current_price'] else "--",
            "Triggered": str(t['triggered_at'])[:16] if t['triggered_at'] else "--",
        })
    st.dataframe(pd.DataFrame(trig_data), use_container_width=True, hide_index=True)

st.divider()

# Database management
st.subheader("Database Management")
col1, col2 = st.columns(2)
with col1:
    if st.button("Backup Database"):
        db_path = Path(__file__).parent.parent.parent / "data" / "ootp_optimizer.db"
        backup_dir = Path(__file__).parent.parent.parent / "data" / "backups"
        backup_dir.mkdir(exist_ok=True)
        if db_path.exists():
            backup_name = f"ootp_optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(db_path, backup_dir / backup_name)
            st.success(f"Backup created: {backup_name}")
        else:
            st.warning("No database file found.")

with col2:
    if st.button("Reset Database", type="secondary"):
        db_path = Path(__file__).parent.parent.parent / "data" / "ootp_optimizer.db"
        if db_path.exists():
            db_path.unlink()
        init_db()
        st.success("Database reset. Import your data again.")

st.divider()

# ── Data Import ──
st.subheader("Data Import")

watch_dir = config.get('watch_directory', '')
watch_path = Path(watch_dir) if watch_dir else None

# Show current watch directory and let user change it
new_watch = st.text_input("Watch Directory (folder with OOTP CSV exports)",
                          value=watch_dir,
                          help="Point this to the folder where OOTP exports CSV files")
if new_watch != watch_dir:
    config['watch_directory'] = new_watch
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    st.success("Watch directory updated!")
    watch_dir = new_watch
    watch_path = Path(watch_dir) if watch_dir else None

# Define import categories for clear display
IMPORT_CATEGORIES = {
    "Market": {
        "icon": "🏪", "types": ["market"],
        "desc": "Card list with prices & ratings",
    },
    "Roster": {
        "icon": "📋", "types": ["roster_batting", "roster_pitching"],
        "desc": "Active + reserve roster ratings (with splits)",
    },
    "Collection": {
        "icon": "📦", "types": ["collection_batting", "collection_pitching"],
        "desc": "Full collection of owned cards",
    },
    "Stats": {
        "icon": "📊", "types": ["stats_batting", "stats_pitching",
                                  "roster_batting_stats", "roster_pitching_stats"],
        "desc": "Game stats (basic + advanced wOBA/SIERA)",
    },
    "Lineups": {
        "icon": "⚾", "types": ["lineup_vs_rhp", "lineup_vs_lhp",
                                  "lineup_overview", "team_pitching"],
        "desc": "Lineup cards (vs RHP/LHP) + pitching staff",
    },
    "League": {
        "icon": "🌐", "types": ["stats_batting_ratings", "stats_pitching_ratings"],
        "desc": "League-wide ratings for all players",
    },
}

if watch_path and watch_path.exists():
    csvs = sorted(watch_path.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)

    if csvs:
        # Categorise found files
        from app.utils.csv_parser import identify_file_type
        categorised = {}
        uncategorised = []
        for csv_path in csvs:
            ftype = identify_file_type(str(csv_path))
            if ftype:
                categorised[csv_path] = ftype
            else:
                uncategorised.append(csv_path)

        # Show what's available by category
        st.markdown(f"**{len(csvs)} CSV files found** in `{watch_dir}`")

        cat_cols = st.columns(len(IMPORT_CATEGORIES))
        for i, (cat_name, cat_info) in enumerate(IMPORT_CATEGORIES.items()):
            with cat_cols[i]:
                matched = [p for p, t in categorised.items() if t in cat_info['types']]
                icon = cat_info['icon']
                if matched:
                    st.success(f"{icon} **{cat_name}**\n\n{len(matched)} file(s)")
                else:
                    st.warning(f"{icon} **{cat_name}**\n\nNo files found")

        # Import button with progress
        col_import, col_status = st.columns([1, 2])
        with col_import:
            run_import = st.button("Import All", type="primary", use_container_width=True)

        if run_import:
            from app.core.ingestion import ingest_file
            from app.core.recommendations import generate_recommendations

            progress = st.progress(0, text="Starting import...")
            results = []
            for idx, csv_path in enumerate(csvs):
                ftype = categorised.get(csv_path)
                if not ftype:
                    results.append(("skip", csv_path.name, "Unknown file type", 0))
                    continue
                progress.progress((idx + 1) / len(csvs),
                                  text=f"Importing {csv_path.name}...")
                try:
                    result = ingest_file(str(csv_path))
                    results.append(("ok", csv_path.name, ftype, result.get('rows', 0)))
                except Exception as e:
                    results.append(("error", csv_path.name, str(e), 0))

            progress.progress(1.0, text="Generating recommendations...")
            try:
                generate_recommendations()
            except Exception:
                pass  # recommendations are optional
            progress.empty()

            # Show results grouped by status
            ok_results = [r for r in results if r[0] == "ok"]
            err_results = [r for r in results if r[0] == "error"]
            skip_results = [r for r in results if r[0] == "skip"]

            total_rows = sum(r[3] for r in ok_results)
            st.success(f"Imported **{len(ok_results)}** files ({total_rows:,} total rows)")

            if ok_results:
                with st.expander(f"Import details ({len(ok_results)} files)", expanded=False):
                    import_df = pd.DataFrame([
                        {"File": r[1], "Type": r[2], "Rows": r[3]}
                        for r in sorted(ok_results, key=lambda x: x[2])
                    ])
                    st.dataframe(import_df, use_container_width=True, hide_index=True)

            if err_results:
                st.error(f"{len(err_results)} files failed:")
                for r in err_results:
                    st.write(f"  - **{r[1]}**: {r[2]}")

            if skip_results:
                st.caption(f"{len(skip_results)} unrecognised files skipped")
    else:
        st.info("No CSV files found in the watch directory.")
elif watch_dir:
    st.error(f"Watch directory not found: `{watch_dir}`")
else:
    st.info("Set a watch directory above to enable CSV import.")

# Last import log
st.markdown("**Recent Imports**")
log_rows = conn.execute("""
    SELECT file_type, file_name, row_count, ingested_at
    FROM ingestion_log ORDER BY ingested_at DESC LIMIT 15
""").fetchall()
if log_rows:
    log_df = pd.DataFrame([dict(r) for r in log_rows])
    log_df.columns = ["Type", "File", "Rows", "Time"]
    st.dataframe(log_df, use_container_width=True, hide_index=True, height=200)
else:
    st.caption("No import history yet.")

# ── Meta Recalculation ──
st.divider()
st.subheader("🔄 Recalculate Meta Scores")
st.caption(
    "After changing weights (manually or via auto-calibration on Game Stats), "
    "recalculate all card meta scores without re-importing CSV files."
)
try:
    from app.core.meta_scoring import get_weights_with_source
    _, _, source = get_weights_with_source()
    st.info(f"Current weight source: **{source}**")
except Exception:
    pass

if st.button("🔄 Recalculate All Meta Scores", type="primary"):
    with st.spinner("Recalculating..."):
        from app.core.ingestion import recalculate_all_meta_scores
        result = recalculate_all_meta_scores()
    st.success(result['message'])

conn.close()
