"""Streamlit helpers for the background worker + staleness panel.

Usage on any page:
    from app.utils.live_status import ensure_worker_running, live_header
    ensure_worker_running()
    live_header()  # renders the badge + auto-refreshes every 10s

Usage for staleness reminder:
    from app.utils.live_status import staleness_reminder
    staleness_reminder()  # renders the "what to re-export" panel
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import streamlit as st

from app.core.background_worker import (
    start_worker, get_worker_status, trigger_refresh_now, is_running,
)
from app.core.staleness import (
    check_csv_staleness, check_html_staleness, overall_staleness,
    build_staleness_report,
)
from app.core.database import load_config


# How often the live badge fragment re-renders. Worker debounces at 3s,
# so anything under 5s is wasted effort. 10s is a good UX cadence.
LIVE_REFRESH_SECONDS = 10


def ensure_worker_running() -> bool:
    """Idempotent — kick the background worker on first call per process.

    Safe to call from every page's top-of-script. Stores a session flag
    to avoid spam-logging the "already running" message.
    """
    if st.session_state.get('_bg_worker_started'):
        return True
    ok = start_worker()
    st.session_state['_bg_worker_started'] = True
    return ok


def _fmt_age(seconds: float) -> str:
    if seconds < 5:
        return 'just now'
    if seconds < 60:
        return f'{int(seconds)}s ago'
    if seconds < 3600:
        return f'{int(seconds/60)}m ago'
    if seconds < 86400:
        return f'{int(seconds/3600)}h ago'
    return f'{int(seconds/86400)}d ago'


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


@st.fragment(run_every=LIVE_REFRESH_SECONDS)
def live_header() -> None:
    """Compact badge: 🟢 Live · last refresh 8s ago · [Refresh now].

    Rendered inside a fragment so it refreshes every 10s independently
    of the rest of the page. Fragment returns nothing — just draws UI.
    """
    status = get_worker_status()
    running = is_running()
    last_refresh = _parse_iso(status.get('last_refresh_at'))
    now = datetime.now()
    age_sec = (now - last_refresh).total_seconds() if last_refresh else None

    # Decide the dot color
    worker_state = status.get('status') or 'idle'
    if not running:
        dot = '🔴'
        label = 'Worker stopped'
    elif worker_state == 'error':
        dot = '🔴'
        label = 'Worker error'
    elif worker_state in ('ingesting', 'recalc', 'scanning'):
        dot = '🟡'
        label = f'Worker {worker_state}…'
    else:
        dot = '🟢'
        label = 'Live'

    # Detect newer data_version than last seen (visual flash could go here)
    dv = int(status.get('data_version') or 0)
    last_seen = st.session_state.get('_live_last_seen_dv', dv)
    new_data = dv > last_seen
    st.session_state['_live_last_seen_dv'] = dv

    cols = st.columns([4, 2, 1])
    with cols[0]:
        age_str = f'· refreshed {_fmt_age(age_sec)}' if age_sec is not None else '· no refresh yet'
        flash = ' · ✨ new data' if new_data else ''
        st.caption(f"{dot} **{label}** {age_str}{flash}")
    with cols[1]:
        msg = (status.get('last_message') or '')[:70]
        if msg:
            st.caption(msg)
    with cols[2]:
        if st.button('Refresh now', key='live_refresh_now', width='stretch'):
            trigger_refresh_now()
            st.toast('Refresh triggered — will run in ~3s', icon='🔄')


@st.fragment(run_every=30)
def staleness_glance() -> None:
    """Compact always-on staleness banner.

    One-line summary that shows the worst group by relative+absolute age,
    highlights groups that are lagging behind a recent export session,
    and auto-refreshes every 30s so the banner turns green seconds after
    a new export lands.
    """
    cfg = load_config()
    watch_dir = cfg.get('watch_directory')
    if not watch_dir:
        st.caption('⚠️ No watch_directory in config.yaml — staleness checks disabled.')
        return
    report = build_staleness_report(watch_dir)
    status = report.overall_status
    dot = {'fresh':'🟢','aging':'🟡','stale':'🟠','missing':'🔴'}.get(status, '⚪')

    if status == 'fresh':
        st.success(f"{dot} **Data fresh** · {report.overall_label}")
        return
    if status == 'aging':
        st.warning(f"{dot} **{report.overall_label}**")
    elif status == 'stale':
        st.error(f"{dot} **{report.overall_label}** — your analysis may be outdated")
    else:
        st.error(f"{dot} **{report.overall_label}**")

    # Top 3 groups that most need attention (worst first). View-lag
    # groups (intra-bucket spread — e.g. lineup default view stale while
    # other lineup views are fresh) sort with the lagging groups so the
    # user sees them even when the bucket's newest file looks fresh.
    ranked = sorted(
        [g for g in report.groups if g.priority == 1],
        key=lambda g: (
            {'missing':0,'stale':1,'aging':2,'fresh':3}[g.absolute_status],
            0 if (g.relative_status == 'lagging' or g.view_lag_h is not None) else 1,
            -(g.newest_age_h or 0),
        ),
    )
    bad = [g for g in ranked
           if g.absolute_status in ('stale', 'missing')
              or g.relative_status == 'lagging'
              or g.view_lag_h is not None]
    if bad:
        lines = []
        for g in bad[:4]:
            age_str = ('never' if g.newest_age_h is None
                       else f'{g.newest_age_h:.0f}h')
            lag_str = ''
            if g.relative_status == 'lagging' and g.relative_lag_h:
                lag_str = f' · {g.relative_lag_h:.0f}h behind fresh exports'
            lines.append(f"- **{g.label}** ({age_str}{lag_str}) — *{g.hint}*")
            # Inline view-lag detail: name the specific stale view file(s)
            # so the user knows which OOTP view dropdown to switch to.
            if g.view_lag_h is not None and g.stale_views:
                top_stale = ', '.join(
                    f"`{name}` ({age:.0f}h)" for name, age in g.stale_views[:2]
                )
                more = (f" (+{len(g.stale_views) - 2} more)"
                        if len(g.stale_views) > 2 else '')
                lines.append(
                    f"  - 🔄 **View variant lag**: {top_stale}{more} — "
                    "switch OOTP view dropdown and re-export"
                )
        st.markdown('\n'.join(lines))


@st.fragment(run_every=30)
def staleness_reminder() -> None:
    """Full "what to re-export" panel — expanded view with all groups.

    Auto-refreshes every 30s so new exports flip the UI from red→green.
    """
    cfg = load_config()
    watch_dir = cfg.get('watch_directory')
    save_dir = cfg.get('save_game_dir')
    if not watch_dir:
        st.caption('No watch_directory configured.')
        return
    report = build_staleness_report(watch_dir)

    # Group table
    import pandas as pd
    dot_map = {'fresh':'🟢','aging':'🟡','stale':'🟠','missing':'🔴'}
    rel_dot_map = {'fresh':'=','same':'=','lagging':'⬆ behind','missing':'—'}
    rows = []
    for g in sorted(report.groups, key=lambda x: (
        {'missing':0,'stale':1,'aging':2,'fresh':3}[x.absolute_status],
        0 if (x.relative_status == 'lagging' or x.view_lag_h is not None) else 1,
        x.priority,
        x.label,
    )):
        age_str = 'never' if g.newest_age_h is None else f'{g.newest_age_h:.1f}h'
        lag_str = ''
        if g.relative_lag_h is not None and g.relative_lag_h > 0.1:
            lag_str = f'+{g.relative_lag_h:.1f}h'
        view_str = '—'
        if g.view_lag_h is not None:
            view_str = f'⚠ {g.view_lag_h:.1f}h spread'
        rows.append({
            '': dot_map.get(g.absolute_status, '⚪'),
            'Export group': g.label,
            'Age': age_str,
            'Relative': lag_str if lag_str else '—',
            'Views': view_str,
            'Present': f'{g.files_present}/{g.files_expected}',
            'Priority': g.priority,
            'Where in OOTP': g.hint,
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True,
                 column_config={
                     '': st.column_config.TextColumn(width='small'),
                     'Export group': st.column_config.TextColumn(width='medium'),
                     'Age': st.column_config.TextColumn(width='small',
                         help="Hours since the newest file in this group."),
                     'Relative': st.column_config.TextColumn(width='small',
                         help="How much older this group's newest file is compared to "
                              "the freshest export across ALL groups. A big number here "
                              "means you exported something recently but skipped this group."),
                     'Views': st.column_config.TextColumn(width='small',
                         help="Spread between the newest and oldest file inside this "
                              "group. A big spread means you re-exported some OOTP view "
                              "dropdowns but missed others (e.g. exported Lineups → "
                              "Batting Ratings but not Lineups → Default)."),
                     'Present': st.column_config.TextColumn(width='small',
                         help="How many of the expected CSVs in this group are actually present."),
                     'Priority': st.column_config.NumberColumn(width='small',
                         help="1 = core export for roster analysis; 2 = supplemental league stats."),
                     'Where in OOTP': st.column_config.TextColumn(width='large'),
                 })

    # If any group has stale views, list the specific files below the table
    # so the user can see exactly which view to re-export.
    view_lag_groups = [g for g in report.groups if g.stale_views]
    if view_lag_groups:
        st.markdown("**🔄 View variant lag** — files inside an export group "
                    "that are older than other files in the same group. "
                    "Switch the OOTP view dropdown and re-export each one.")
        for g in view_lag_groups:
            st.markdown(f"- **{g.label}** ({g.hint})")
            for name, age in g.stale_views:
                st.markdown(f"  - `{name}` — **{age:.0f}h old**")

    # HTML reports (box scores / game logs)
    html = check_html_staleness(save_dir) if save_dir else {}
    if html:
        b = html.get('box_scores', {})
        g = html.get('game_logs', {})
        st.markdown('**HTML reports** (game logs + box scores)')
        for key, meta in [('box_scores', b), ('game_logs', g)]:
            d = dot_map.get(meta.get('status'), '⚪')
            age = meta.get('newest_hours')
            age_str = 'missing' if age is None else f'{age:.1f}h'
            count = meta.get('count') or 0
            st.markdown(f"- {d} **{key.replace('_',' ')}** · {count} files · newest {age_str}")


def get_data_version() -> int:
    """Current data_version — used by AI cache keys so new data re-fires AI."""
    return int(get_worker_status().get('data_version') or 0)
