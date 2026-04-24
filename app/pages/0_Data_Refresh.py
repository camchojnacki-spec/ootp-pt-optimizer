"""Data Refresh — scan the watch folder, preview the plan, run ingestion.

This page is the *single* place for importing OOTP exports. It wraps
``app/core/data_refresh.py`` (pure logic, no Streamlit) with a rich UI:
per-category status cards, a pre-scan preview grouped by file type,
optional league picker when multiple leagues live in the watch folder,
streaming progress during the run, and a post-run summary with raw
results for debugging.

The old import expander in ``main.py`` now just points here.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Ensure the project root is on sys.path so ``app.*`` imports resolve
# when Streamlit runs this page directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection, init_db, load_config
from app.core.data_refresh import (
    plan_refresh,
    execute_refresh,
    format_age,
    format_size,
    FileDescriptor,
    RefreshPlan,
)
from app.core.recommendations import generate_recommendations
from app.utils.sidebar_nav import render_sidebar_nav


# ──────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────
init_db()

st.set_page_config(
    page_title="Data Refresh",
    page_icon="\U0001f4e5",  # 📥
    layout="wide",
)

render_sidebar_nav()

config = load_config()
conn = get_connection()
watch_dir = config.get("watch_directory", "") or ""

st.title("\U0001f4e5 Data Refresh")
st.caption(
    "Scan your OOTP watch folder, preview exactly what will import, "
    "then refresh in one click."
)

# ──────────────────────────────────────────────────────────────────────
# Guard: watch directory exists
# ──────────────────────────────────────────────────────────────────────
if not watch_dir:
    st.error(
        "No `watch_directory` set in `config.yaml`. "
        "Open **Settings** and point it at your OOTP `online_data` folder."
    )
    conn.close()
    st.stop()

if not Path(watch_dir).exists():
    st.error(f"Watch directory not found: `{watch_dir}`")
    st.caption(
        "Edit `config.yaml` or open **Settings** to update the path."
    )
    conn.close()
    st.stop()

with st.expander("Watch folder", expanded=False):
    st.code(watch_dir, language=None)

# ──────────────────────────────────────────────────────────────────────
# Build the plan (scan + ingestion_log join)
# ──────────────────────────────────────────────────────────────────────
try:
    plan: RefreshPlan = plan_refresh(watch_dir, conn)
except Exception as e:
    st.error(f"Failed to scan watch directory: {e}")
    conn.close()
    st.stop()


def _visible_files(p: RefreshPlan, league: str | None) -> list[FileDescriptor]:
    """Files that belong to the selected league (or are league-agnostic)."""
    if league is None:
        return [d for d in p.files if d.file_type]
    return [
        d for d in p.files
        if d.file_type and d.league_id in (league, None)
    ]


# ──────────────────────────────────────────────────────────────────────
# League picker — only shown when >1 league is detected in the folder
# ──────────────────────────────────────────────────────────────────────
# Auto-default to the most recently used league from config.yaml.
# After each successful refresh, active_league is persisted back so the
# user never has to pick the same league twice.
_cfg_league = load_config().get("active_league")

selected_league: str | None = None
if len(plan.leagues_detected) > 1:
    # Pre-select the config league if it's in the detected list
    _default_idx = 0
    if _cfg_league and _cfg_league in plan.leagues_detected:
        _default_idx = plan.leagues_detected.index(_cfg_league)
    selected_league = st.selectbox(
        "Target league",
        options=plan.leagues_detected,
        index=_default_idx,
        help="Auto-defaults to your last-used league. Saved after each refresh.",
    )
elif len(plan.leagues_detected) == 1:
    selected_league = plan.leagues_detected[0]

visible_files = _visible_files(plan, selected_league)

# Re-group visible files into categories for rendering
visible_categories: dict[str, list[FileDescriptor]] = {}
for d in visible_files:
    visible_categories.setdefault(d.category, []).append(d)
for cat in visible_categories:
    visible_categories[cat].sort(key=lambda x: x.modified, reverse=True)

# ──────────────────────────────────────────────────────────────────────
# Top status strip — at-a-glance counts
# ──────────────────────────────────────────────────────────────────────
importable = [d for d in visible_files if not d.will_skip]
skipped = [d for d in visible_files if d.will_skip]
never = [d for d in importable if d.last_import is None]

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Files in watch", len(plan.files))
with m2:
    st.metric("Ready to import", len(importable))
with m3:
    st.metric("New (never imported)", len(never))
with m4:
    st.metric("Unrecognized", plan.total_unrecognized)
with m5:
    leagues_label = ", ".join(plan.leagues_detected) or "—"
    st.metric("Leagues", leagues_label)

# ──────────────────────────────────────────────────────────────────────
# Quick-refresh button at the top (duplicated at bottom for convenience)
# ──────────────────────────────────────────────────────────────────────
_top_importable = [d for d in visible_files if not d.will_skip]
_top_btn_cols = st.columns([3, 3, 6])
with _top_btn_cols[0]:
    _top_run_all = st.button(
        f"\U0001f4e5 Refresh All ({len(_top_importable)})",
        key="top_refresh_all",
        type="primary",
        use_container_width=True,
        disabled=len(_top_importable) == 0,
    )
with _top_btn_cols[1]:
    st.caption("")  # spacer

# ──────────────────────────────────────────────────────────────────────
# Missing-core warning
# ──────────────────────────────────────────────────────────────────────
if plan.missing_core:
    st.warning(
        "\u26a0\ufe0f **Missing core file types:** "
        + ", ".join(f"`{ft}`" for ft in plan.missing_core)
        + ". Re-export these from OOTP before running the refresh — "
        "recommendations will be incomplete without them."
    )

st.divider()

# ──────────────────────────────────────────────────────────────────────
# Status by category — 4-up grid of cards, wrapping into new rows
# ──────────────────────────────────────────────────────────────────────
st.subheader("Status by category")

# Pretty order — known buckets first, 'Other' last
PREFERRED_CATEGORY_ORDER = [
    "Market",
    "Roster",
    "Collection",
    "Lineups",
    "Roster Stats",
    "League Stats",
    "League Ratings",
    "League Team Stats",
    "Park Factors",
    "Fielding",
    "Pitch Arsenal",
    "Standings",
    "Other",
]
ordered_cats = [c for c in PREFERRED_CATEGORY_ORDER if c in visible_categories]
# Any unexpected categories get appended at the end
for c in visible_categories:
    if c not in ordered_cats:
        ordered_cats.append(c)

if not ordered_cats:
    st.info("No recognized CSV files in the watch folder yet.")
else:
    COLS_PER_ROW = 4
    rows = math.ceil(len(ordered_cats) / COLS_PER_ROW)
    now = datetime.now()
    idx = 0
    for _ in range(rows):
        cols = st.columns(COLS_PER_ROW)
        for col in cols:
            if idx >= len(ordered_cats):
                break
            cat = ordered_cats[idx]
            files = visible_categories[cat]
            last_imports = [d.last_import for d in files if d.last_import]
            latest = max(last_imports) if last_imports else None
            total_rows = sum(d.last_row_count for d in files)

            with col:
                if latest:
                    age = format_age((now - latest).total_seconds() / 3600.0)
                    status_html = (
                        f"<span style='color:#888;font-size:0.82em;'>"
                        f"last: {age} ago &bull; {total_rows:,} rows</span>"
                    )
                else:
                    status_html = (
                        "<span style='color:#c44;font-size:0.82em;'>"
                        "never imported</span>"
                    )
                st.markdown(
                    f"**{cat}**  \n"
                    f"<span style='font-size:1.1em;'>{len(files)} file(s)</span>  \n"
                    f"{status_html}",
                    unsafe_allow_html=True,
                )
            idx += 1

st.divider()

# ──────────────────────────────────────────────────────────────────────
# File-level preview — expanders per category with per-file checkboxes
# ──────────────────────────────────────────────────────────────────────
st.subheader("Files queued for refresh")
st.caption(
    "Uncheck any files you want to skip this round. "
    "Recognized-noop files (like team standings) can't be run yet and "
    "are disabled."
)

selected_rows: dict[str, bool] = {}
now_dt = datetime.now()
for cat in ordered_cats:
    files = visible_categories[cat]
    with st.expander(f"{cat} ({len(files)})", expanded=True):
        for d in files:
            # Build the status string
            if d.will_skip:
                icon = "\u23ed\ufe0f"  # ⏭
                status_txt = f"skip — {d.skip_reason}"
            elif d.last_import is None:
                icon = "\U0001f195"  # 🆕
                status_txt = "never imported"
            else:
                icon = "\u2714\ufe0f"  # ✔
                ago = format_age(
                    (now_dt - d.last_import).total_seconds() / 3600.0
                )
                status_txt = f"last: {ago} ago \u2022 {d.last_row_count:,} rows"

            age_lbl = format_age(d.age_hours)
            size_lbl = format_size(d.size_bytes)
            league_tag = f" \u2022 {d.league_id}" if d.league_id else ""

            cb_col, name_col, meta_col = st.columns([0.4, 6.6, 3.0])
            with cb_col:
                default_checked = (not d.will_skip) and (
                    selected_league is None
                    or d.league_id in (selected_league, None)
                )
                checked = st.checkbox(
                    " ",
                    value=default_checked,
                    key=f"refresh_sel_{d.path}",
                    label_visibility="collapsed",
                    disabled=d.will_skip,
                )
                selected_rows[d.path] = checked
            with name_col:
                st.markdown(
                    f"{icon} `{d.filename}`  \n"
                    f"<span style='color:#888;font-size:0.82em;'>"
                    f"{d.file_type} \u2022 {size_lbl} \u2022 {age_lbl} old"
                    f"{league_tag}</span>",
                    unsafe_allow_html=True,
                )
            with meta_col:
                st.markdown(
                    f"<div style='text-align:right;color:#888;font-size:0.82em;"
                    f"padding-top:4px;'>{status_txt}</div>",
                    unsafe_allow_html=True,
                )

if plan.unrecognized:
    with st.expander(
        f"Unrecognized ({len(plan.unrecognized)})", expanded=False
    ):
        st.caption(
            "These CSVs are in the watch folder but don't match any known "
            "file pattern. Usually a typo in the export name or a new "
            "export type we haven't wired up yet."
        )
        for d in plan.unrecognized:
            st.text(
                f"\u2754 {d.filename}  "
                f"{format_size(d.size_bytes)}  \u2022  "
                f"{format_age(d.age_hours)} old"
            )

# ──────────────────────────────────────────────────────────────────────
# Run options
# ──────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Run options")

opt_cols = st.columns([2, 2, 2, 3])
with opt_cols[0]:
    use_history = st.checkbox(
        "Snapshot history",
        value=True,
        help=(
            "After importing, capture roster + meta + stats into the "
            "history table so trend charts and calibration keep working."
        ),
    )
with opt_cols[1]:
    regen_recs = st.checkbox(
        "Regen recs",
        value=True,
        help="Re-run buy/sell/flip scoring after import completes.",
    )
with opt_cols[2]:
    games_into_season_raw = st.number_input(
        "Games in",
        min_value=0,
        max_value=162,
        value=0,
        step=1,
        help=(
            "Tags the history snapshot with how deep into the season "
            "this refresh is. Lets trend charts show 'at game 42 vs now'."
        ),
    )
with opt_cols[3]:
    team_record = st.text_input(
        "Team record",
        value="",
        placeholder="20-18",
        help="Optional record to tag the snapshot with (e.g. '20-18').",
    )

# ──────────────────────────────────────────────────────────────────────
# Run buttons
# ──────────────────────────────────────────────────────────────────────
selected_paths = [p for p, chk in selected_rows.items() if chk]
all_importable_paths = [d.path for d in importable]

btn_cols = st.columns([2, 2, 6])
with btn_cols[0]:
    run_all = st.button(
        f"\U0001f4e5 Refresh All ({len(all_importable_paths)})",
        type="primary",
        use_container_width=True,
        disabled=len(all_importable_paths) == 0,
    )
with btn_cols[1]:
    run_sel = st.button(
        f"Refresh Selected ({len(selected_paths)})",
        use_container_width=True,
        disabled=len(selected_paths) == 0,
    )


def _run_refresh(paths: list[str], label: str) -> None:
    """Execute a refresh with live progress and show the summary."""
    if not paths:
        st.warning("Nothing selected.")
        return

    st.markdown(f"#### {label}")
    progress = st.progress(0.0, text="Starting…")
    live_status = st.empty()

    def on_progress(i: int, total: int, fname: str) -> None:
        if total <= 0:
            progress.progress(1.0, text="Done.")
            return
        pct = min(1.0, (i + 1) / total)
        progress.progress(pct, text=f"[{i+1}/{total}] {fname}")

    try:
        games_param: int | None = (
            int(games_into_season_raw) if games_into_season_raw else None
        )
        result = execute_refresh(
            files=paths,
            league_id=selected_league,
            use_history=use_history,
            games_into_season=games_param,
            team_record=(team_record or None),
            on_progress=on_progress,
        )
    except Exception as e:
        progress.empty()
        st.error(f"Refresh failed: {e}")
        return

    progress.progress(1.0, text="Import complete.")

    # Persist the selected league to config.yaml so it auto-defaults next time
    if selected_league and result.files_succeeded > 0:
        try:
            import yaml as _yaml
            from app.core.database import PROJECT_ROOT
            _cfg_path = PROJECT_ROOT / "config.yaml"
            _cfg = load_config()
            if _cfg.get("active_league") != selected_league:
                _cfg["active_league"] = selected_league
                with open(_cfg_path, "w", encoding="utf-8") as _f:
                    _yaml.dump(_cfg, _f, default_flow_style=False)
        except Exception:
            pass  # non-critical — don't block the refresh summary

    # Regenerate recommendations (optional)
    if regen_recs and result.files_succeeded > 0:
        live_status.caption("Regenerating recommendations…")
        try:
            generate_recommendations()
            live_status.caption("Recommendations regenerated.")
        except Exception as e:
            st.warning(f"Recommendation regen failed: {e}")

    # Store timestamp so a rerun (e.g. clicking the button again)
    # still shows something meaningful.
    st.session_state["_last_refresh_at"] = result.finished_at.isoformat()

    # ── Summary metrics ─────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Files OK", result.files_succeeded)
    with s2:
        st.metric("Rows ingested", f"{result.rows_ingested:,}")
    with s3:
        st.metric("Duration", f"{result.duration_seconds:.1f}s")
    with s4:
        st.metric(
            "Failed",
            result.files_failed,
            delta_color="inverse" if result.files_failed > 0 else "off",
        )

    # ── History snapshot outcome ────────────────────────────────────
    if result.history_snapshot is not None:
        snap = result.history_snapshot
        if snap.get("error"):
            st.warning(f"History snapshot failed: {snap['error']}")
        else:
            count = snap.get("count", snap.get("rows", "?"))
            export_num = snap.get("export_number", "?")
            st.success(
                f"\U0001f4f8 History snapshot: {count} players captured "
                f"(export #{export_num}) for league `{result.league_id}`."
            )

    # ── Errors ──────────────────────────────────────────────────────
    if result.errors:
        with st.expander(
            f"\u26a0\ufe0f {len(result.errors)} error(s)", expanded=True
        ):
            for err in result.errors:
                st.text(f"\u2022 {err}")

    # ── Per-file detail ─────────────────────────────────────────────
    with st.expander("Per-file results", expanded=False):
        for r in result.per_file:
            status = r.get("status", "unknown")
            rows = r.get("rows", 0)
            fname = r.get("filename", "?")
            ftype = r.get("file_type", "?")
            if status == "success" and rows > 0:
                icon = "\u2705"  # ✅
            elif status == "success" and rows == 0:
                icon = "\u2796"  # ➖
            elif status == "skipped":
                icon = "\u23ed\ufe0f"  # ⏭
            else:
                icon = "\u274c"  # ❌
            st.text(f"{icon} {ftype:<28} {rows:>6,} rows  {fname}")

    with st.expander("Raw result (debug)", expanded=False):
        st.json(result.to_dict())


if run_all or _top_run_all:
    _run_refresh(all_importable_paths, "Refresh All")

if run_sel:
    _run_refresh(selected_paths, "Refresh Selected")

# ──────────────────────────────────────────────────────────────────────
# Footer — last refresh timestamp + quick links
# ──────────────────────────────────────────────────────────────────────
st.divider()
footer_cols = st.columns([4, 1, 1])
with footer_cols[0]:
    last_run = st.session_state.get("_last_refresh_at")
    if last_run:
        st.caption(f"Last refresh this session: {last_run[:19]}")
    else:
        # Fall back to the most recent ingestion_log entry
        row = conn.execute(
            "SELECT MAX(ingested_at) AS t FROM ingestion_log"
        ).fetchone()
        if row and row["t"]:
            st.caption(f"Last import (DB): {str(row['t'])[:19]}")
        else:
            st.caption("No prior imports on record.")
with footer_cols[1]:
    st.page_link("main.py", label="Dashboard", icon="\U0001f3e0")
with footer_cols[2]:
    st.page_link("pages/5_Settings.py", label="Settings", icon="\u2699\ufe0f")

conn.close()
