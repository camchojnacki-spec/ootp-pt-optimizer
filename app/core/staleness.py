"""What does the user need to re-export from OOTP?

Group-aware staleness: files that come from the same OOTP export session
are grouped together, and a group is flagged when its newest file lags
meaningfully behind the newest export across ALL groups. This catches
the common case where a user exports Market but forgets Roster — the
engine notices the time skew and reminds them.

Three levels of signal:
    * "Oldest at a glance" — the single oldest required export
    * "Relative stale" — groups that are >6h older than the freshest group
    * "Absolute stale" — anything >24h old regardless of relative position

Used by the Roster Optimizer staleness banner + expander.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.core.data_refresh import (
    scan_watch_directory,
    FileDescriptor,
)


# ──────────────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────────────
FRESH_HOURS = 6          # < 6h old → fresh
AGING_HOURS = 24         # 6-24h → aging
RELATIVE_GAP_HOURS = 6   # group >6h older than freshest group → relative-stale


# ──────────────────────────────────────────────────────────────────────
# Export groups — one OOTP export click usually produces multiple CSVs
# that should all be refreshed together. If one file in a group is fresh
# but another is old, the user likely missed that tab/button.
# ──────────────────────────────────────────────────────────────────────

EXPORT_GROUPS: dict[str, dict] = {
    'market': {
        'label': 'Market',
        'hint': 'League → Auction House → Export Market CSV',
        'file_types': ['market'],
        'priority': 1,
    },
    'collection': {
        # Label deliberately avoids "card list" / "all cards" phrasing, which
        # users conflate with the Market / Auction House export
        # (pt_card_list.csv). This label is the Collection → Manage Cards
        # Ratings CSVs — a completely different OOTP menu.
        'label': 'Collection Ratings (Manage Cards)',
        # OOTP's Collection Export only writes the currently-selected view.
        # Users routinely click Export while on the default "Overview" view,
        # which produces collection_..._default.csv (aggregates only, useless
        # for meta scoring). The hint must spell out that the view dropdown
        # has to be switched to Batting Ratings, then Pitching Ratings, with
        # Export clicked after each switch.
        'hint': ('Collection → Manage Cards → switch view to **Batting Ratings** '
                 '→ Export, then **Pitching Ratings** → Export (default view is '
                 'not enough; this is NOT the Market / pt_card_list export)'),
        'file_types': ['collection_batting', 'collection_pitching'],
        'priority': 1,
    },
    'team_roster': {
        'label': 'Team Roster',
        'hint': 'Team → Roster → Export (batting + pitching + fielding)',
        'file_types': [
            'roster_batting', 'roster_pitching',
            'roster_batting_stats', 'roster_pitching_stats',
        ],
        'priority': 1,
    },
    'team_lineups': {
        'label': 'Team Lineups',
        'hint': 'Team → Lineups → export Overview + vs LHP + vs RHP',
        'file_types': ['lineup_overview', 'lineup_vs_lhp', 'lineup_vs_rhp'],
        'priority': 1,
    },
    'team_pitching': {
        'label': 'Team Pitching Staff',
        'hint': 'Team → Pitching → Export',
        'file_types': ['team_pitching'],
        'priority': 1,
    },
    'league_batting': {
        'label': 'League Batting Stats',
        'hint': 'League → Stats → Batting → Export (all tabs)',
        'file_types': ['stats_batting', 'stats_batting_ratings'],
        'priority': 2,
    },
    'league_pitching': {
        'label': 'League Pitching Stats',
        'hint': 'League → Stats → Pitching → Export (all tabs)',
        'file_types': ['stats_pitching', 'stats_pitching_ratings',
                       'pitch_ratings'],
        'priority': 2,
    },
    'league_team_stats': {
        'label': 'League Team / Park',
        'hint': 'League → Stats → Team → Export + Park Factors tab',
        'file_types': ['team_stats_batting', 'team_stats_pitching',
                       'team_stats_park'],
        'priority': 2,
    },
    'fielding': {
        'label': 'Fielding / Position',
        'hint': 'League → Stats → Fielding → Export + Position Ratings',
        'file_types': ['fielding_ratings', 'position_ratings', 'fielding_stats'],
        'priority': 2,
    },
}

# Reverse lookup: file_type → group key
_TYPE_TO_GROUP: dict[str, str] = {}
for _gk, _gm in EXPORT_GROUPS.items():
    for _ft in _gm['file_types']:
        _TYPE_TO_GROUP[_ft] = _gk


def _classify_age(hours: Optional[float]) -> str:
    if hours is None:
        return 'missing'
    if hours < FRESH_HOURS:
        return 'fresh'
    if hours < AGING_HOURS:
        return 'aging'
    return 'stale'


@dataclass
class GroupStatus:
    """Aggregated staleness for one export group."""
    group_key: str
    label: str
    hint: str
    priority: int
    files_present: int
    files_expected: int
    newest_age_h: Optional[float]   # hours since newest file in this group
    oldest_age_h: Optional[float]
    absolute_status: str            # fresh | aging | stale | missing
    relative_status: str            # same | lagging | fresh | missing
    relative_lag_h: Optional[float] # how much older than freshest group
    missing_types: list[str] = field(default_factory=list)
    # Intra-group view-lag: catches the common case where a single file_type
    # pattern matches multiple OOTP "view dropdown" variants (e.g. all
    # `lineups_-_overview_*.csv` files share the `lineup_overview` file_type
    # but represent different views — default, batting_ratings, position_ratings,
    # fielding_*). If the user re-exports some views but forgets others, the
    # group's newest_age_h still looks fresh; this field surfaces the spread.
    view_lag_h: Optional[float] = None
    stale_views: list[tuple[str, float]] = field(default_factory=list)
                                              # [(filename, age_h), ...]


@dataclass
class StalenessReport:
    """Full staleness summary for the UI."""
    overall_status: str            # fresh | aging | stale | missing
    overall_label: str
    freshest_age_h: Optional[float]
    oldest_age_h: Optional[float]
    oldest_group: Optional[str]
    groups: list[GroupStatus] = field(default_factory=list)
    worst_group_key: Optional[str] = None


def build_staleness_report(watch_dir: str) -> StalenessReport:
    """Build a grouped, relative-aware staleness report.

    Key semantic change vs old per-file logic:
        * Each file type is classified into a group.
        * Group age = newest file in the group.
        * Relative staleness = how much older this group's newest is
          than the freshest group's newest. If >6h lag, flag as "lagging"
          — the user exported something but missed this group.
    """
    if not watch_dir or not os.path.isdir(watch_dir):
        return StalenessReport(overall_status='missing',
                               overall_label='No watch directory configured',
                               freshest_age_h=None, oldest_age_h=None,
                               oldest_group=None)

    descriptors = scan_watch_directory(watch_dir)

    # Bucket EVERY descriptor by file_type (NOT just the newest). A single
    # file_type pattern often matches multiple OOTP view-dropdown exports
    # (e.g. lineups_-_overview matches default, batting_ratings,
    # position_ratings, fielding_ratings, fielding_stats variants), so we
    # need the full set to detect view-lag within a group.
    by_type: dict[str, list[FileDescriptor]] = {}
    for d in descriptors:
        if not d.file_type:
            continue
        by_type.setdefault(d.file_type, []).append(d)
    newest_by_type: dict[str, FileDescriptor] = {
        ft: max(ds, key=lambda d: d.modified) for ft, ds in by_type.items()
    }

    # Build group status
    groups: list[GroupStatus] = []
    for gk, gm in EXPORT_GROUPS.items():
        ages = []
        present = 0
        missing_types = []
        for ft in gm['file_types']:
            d = newest_by_type.get(ft)
            if d:
                ages.append(d.age_hours)
                present += 1
            else:
                missing_types.append(ft)
        if ages:
            newest_age = min(ages)
            oldest_age = max(ages)
        else:
            newest_age = None
            oldest_age = None

        # View-lag: gather every descriptor in the group across all
        # file_types and compute the spread. If the spread exceeds the
        # relative-gap threshold, the user re-exported part of the group
        # but missed at least one view — surface the lagging filenames.
        #
        # We restrict the comparison to descriptors matching the freshest
        # file's league_id so cross-league residue (e.g. old `i76_*` files
        # left over from a previous league) doesn't masquerade as a view
        # mismatch in the active league. Files with no league prefix
        # (team-specific exports like `toronto_dark_knights_*`) are
        # always included.
        all_descs: list[FileDescriptor] = []
        for ft in gm['file_types']:
            all_descs.extend(by_type.get(ft, []))
        view_lag_h: Optional[float] = None
        stale_views: list[tuple[str, float]] = []
        if len(all_descs) >= 2:
            freshest = min(all_descs, key=lambda d: d.age_hours)
            active_league = freshest.league_id
            in_scope = [
                d for d in all_descs
                if d.league_id is None or d.league_id == active_league
            ]
            if len(in_scope) >= 2:
                youngest = min(d.age_hours for d in in_scope)
                oldest = max(d.age_hours for d in in_scope)
                spread = oldest - youngest
                if spread >= RELATIVE_GAP_HOURS:
                    view_lag_h = spread
                    threshold = youngest + RELATIVE_GAP_HOURS
                    stale_views = [
                        (d.filename, d.age_hours)
                        for d in sorted(in_scope, key=lambda d: -d.age_hours)
                        if d.age_hours >= threshold
                    ]

        absolute_status = _classify_age(newest_age)
        groups.append(GroupStatus(
            group_key=gk, label=gm['label'], hint=gm['hint'],
            priority=gm['priority'],
            files_present=present,
            files_expected=len(gm['file_types']),
            newest_age_h=newest_age, oldest_age_h=oldest_age,
            absolute_status=absolute_status,
            relative_status='same',  # filled below
            relative_lag_h=None,
            missing_types=missing_types,
            view_lag_h=view_lag_h,
            stale_views=stale_views,
        ))

    # Relative staleness: compare each group's newest age vs freshest group
    present_ages = [g.newest_age_h for g in groups if g.newest_age_h is not None]
    freshest_age = min(present_ages) if present_ages else None

    for g in groups:
        if g.newest_age_h is None:
            g.relative_status = 'missing'
            g.relative_lag_h = None
            continue
        if freshest_age is None:
            g.relative_status = 'fresh'
            g.relative_lag_h = 0.0
            continue
        lag = g.newest_age_h - freshest_age
        g.relative_lag_h = lag
        if lag >= RELATIVE_GAP_HOURS:
            g.relative_status = 'lagging'
        elif g.absolute_status == 'fresh':
            g.relative_status = 'fresh'
        else:
            g.relative_status = 'same'

    # Overall status: worst of the priority-1 groups
    priority_groups = [g for g in groups if g.priority == 1]
    if not priority_groups:
        priority_groups = groups

    # Missing wins, then stale, then lagging, then aging, then fresh
    status_rank = {'missing': 4, 'stale': 3, 'aging': 2, 'fresh': 0}
    # relative lagging or view-lag bumps one level
    def worst_rank(g: GroupStatus) -> int:
        r = status_rank.get(g.absolute_status, 0)
        if g.relative_status == 'lagging' and r < 3:
            r = max(r, 2)  # at least aging
        # Intra-group view-lag also escalates to "aging" — a fresh
        # bucket where the canonical view is stale shouldn't read green.
        if g.view_lag_h is not None and r < 3:
            r = max(r, 2)
        return r
    worst = max(priority_groups, key=worst_rank) if priority_groups else None
    worst_rank_val = worst_rank(worst) if worst else 0
    if worst_rank_val >= 4:
        overall_status = 'missing'
        overall_label = f"{worst.label} never exported — please run now"
    elif worst_rank_val >= 3:
        overall_status = 'stale'
        # Collection stale headlines get the inline fix — the generic
        # "refresh now" wording leads users to re-export the Market CSV
        # (pt_card_list) thinking that's the card list, which doesn't
        # update the Ratings views we actually consume.
        if worst.group_key == 'collection':
            overall_label = (
                f"{worst.label} is {worst.newest_age_h:.0f}h old — in OOTP: "
                "Collection → Manage Cards → switch view to Batting Ratings "
                "→ Export, then Pitching Ratings → Export"
            )
        else:
            overall_label = f"{worst.label} is {worst.newest_age_h:.0f}h old — refresh now"
    elif worst and worst.relative_status == 'lagging':
        overall_status = 'aging'
        # Special case: Collection lagging while a FRESH collection_default
        # file exists means the user clicked Export but was on the wrong
        # view. Surface the specific fix in the banner rather than the
        # generic "re-export to sync" message.
        if worst.group_key == 'collection':
            default_desc = newest_by_type.get('collection_default')
            if default_desc and default_desc.age_hours < FRESH_HOURS:
                overall_label = (
                    "Collection exported as Default view only — in OOTP, "
                    "switch view to Batting Ratings → Export, then "
                    "Pitching Ratings → Export"
                )
            else:
                overall_label = (f"{worst.label} is {worst.relative_lag_h:.0f}h behind "
                                 f"the rest of your exports — re-export to sync")
        else:
            overall_label = (f"{worst.label} is {worst.relative_lag_h:.0f}h behind "
                             f"the rest of your exports — re-export to sync")
    elif worst and worst.view_lag_h is not None:
        # The bucket itself looks fresh (newest file < 6h) but at least one
        # view inside it is much older — call it out by name.
        overall_status = 'aging'
        if worst.stale_views:
            stale_basename = worst.stale_views[0][0]
            stale_age = worst.stale_views[0][1]
            overall_label = (
                f"{worst.label}: \"{stale_basename}\" is {stale_age:.0f}h old "
                f"while other views in this group are fresh — switch the OOTP "
                f"view dropdown and re-export the missing variant"
            )
        else:
            overall_label = (
                f"{worst.label} has a {worst.view_lag_h:.0f}h spread between "
                "view variants — re-export the older view from OOTP"
            )
    elif worst_rank_val >= 2:
        overall_status = 'aging'
        overall_label = f"{worst.label} is {worst.newest_age_h:.0f}h old"
    else:
        # All primary groups fresh
        overall_status = 'fresh'
        if freshest_age is not None:
            overall_label = f"All primary exports fresh (newest {freshest_age:.1f}h ago)"
        else:
            overall_label = "All primary exports fresh"

    # Oldest group for the glance banner
    oldest_g = None
    oldest_age = None
    for g in priority_groups:
        if g.newest_age_h is None:
            continue
        if oldest_age is None or g.newest_age_h > oldest_age:
            oldest_age = g.newest_age_h
            oldest_g = g.group_key

    return StalenessReport(
        overall_status=overall_status,
        overall_label=overall_label,
        freshest_age_h=freshest_age,
        oldest_age_h=oldest_age,
        oldest_group=oldest_g,
        groups=groups,
        worst_group_key=worst.group_key if worst else None,
    )


# ──────────────────────────────────────────────────────────────────────
# HTML staleness for game-log + box-score (unchanged from v1)
# ──────────────────────────────────────────────────────────────────────

def check_html_staleness(save_dir: str) -> dict:
    """Latest file + total count in box_scores/ + game_logs/ HTML dirs."""
    out = {
        'box_scores': {'count': 0, 'newest_hours': None, 'status': 'missing'},
        'game_logs': {'count': 0, 'newest_hours': None, 'status': 'missing'},
    }
    if not save_dir:
        return out
    for key, sub in (('box_scores', 'news/html/box_scores'),
                     ('game_logs', 'news/html/game_logs')):
        full = os.path.join(save_dir, sub)
        if not os.path.isdir(full):
            continue
        files = []
        try:
            for name in os.listdir(full):
                if not name.lower().endswith('.html'):
                    continue
                fp = os.path.join(full, name)
                try:
                    files.append(os.path.getmtime(fp))
                except OSError:
                    continue
        except OSError:
            continue
        out[key]['count'] = len(files)
        if files:
            newest_ts = max(files)
            age_h = (datetime.now().timestamp() - newest_ts) / 3600.0
            out[key]['newest_hours'] = age_h
            out[key]['status'] = _classify_age(age_h)
    return out


# ──────────────────────────────────────────────────────────────────────
# Back-compat shims — old per-file API used by a few callers
# ──────────────────────────────────────────────────────────────────────

@dataclass
class StalenessEntry:
    """Legacy per-file row — kept for anything still using the old API."""
    file_type: str
    label: str
    status: str
    age_hours: Optional[float]
    hint: str
    is_primary: bool


def check_csv_staleness(watch_dir: str) -> list[StalenessEntry]:
    """Legacy API — returns per-file rows. New code should use build_staleness_report()."""
    report = build_staleness_report(watch_dir)
    out = []
    for g in report.groups:
        out.append(StalenessEntry(
            file_type=g.group_key,
            label=g.label,
            status=g.absolute_status,
            age_hours=g.newest_age_h,
            hint=g.hint,
            is_primary=(g.priority == 1),
        ))
    out.sort(key=lambda e: (
        {'missing':0,'stale':1,'aging':2,'fresh':3}[e.status],
        not e.is_primary,
        e.label,
    ))
    return out


def overall_staleness(watch_dir: str, save_dir: str) -> dict:
    """Legacy API — used by live_status.staleness_reminder()."""
    report = build_staleness_report(watch_dir)
    return {'status': report.overall_status, 'label': report.overall_label}
