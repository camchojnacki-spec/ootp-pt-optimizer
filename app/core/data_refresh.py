"""Data refresh orchestration — scan, plan, and execute OOTP CSV imports.

This module is the single source of truth for the "refresh my data" workflow.
It's deliberately pure Python (no Streamlit imports) so it can be called from
the dedicated refresh page, the legacy main.py import expander, the file
watcher, or a CLI. The Streamlit page wraps these functions with progress
bars and expanders — the logic lives here.

Public API:
    - scan_watch_directory(watch_dir) → list of file descriptors
    - get_last_ingestion_status(conn) → per-file-type status from ingestion_log
    - plan_refresh(watch_dir, conn) → full preview: what would run, why,
      and what's new vs. already-imported
    - execute_refresh(files, league_id, use_history, on_progress) → runs
      ingestion, returns per-file results + optional history snapshot
    - detect_leagues_in_dir(watch_dir) → set of league_ids in the folder
    - get_file_category(file_type) → UI bucket label for grouping

The "plan then execute" split lets the UI show a detailed dry-run before the
user commits. `execute_refresh` accepts a callback so progress can stream.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from app.core.ingestion import ingest_file, detect_league_id
from app.utils.csv_parser import identify_file_type
from app.utils.constants import FILE_PATTERNS


# ──────────────────────────────────────────────────────────────────────
# Category groupings — used by the UI to bucket file types into cards.
# A fresh refresh touches all categories; missing categories signal a
# stale/incomplete export set.
# ──────────────────────────────────────────────────────────────────────
FILE_CATEGORIES: dict[str, str] = {
    "market": "Market",
    "roster_batting": "Roster",
    "roster_pitching": "Roster",
    "roster_batting_stats": "Roster Stats",
    "roster_pitching_stats": "Roster Stats",
    "collection_batting": "Collection",
    "collection_pitching": "Collection",
    "collection_default": "Collection",
    "stats_batting": "League Stats",
    "stats_pitching": "League Stats",
    "stats_batting_ratings": "League Ratings",
    "stats_pitching_ratings": "League Ratings",
    "league_player_default": "League Rosters",
    "team_stats_batting": "League Team Stats",
    "team_stats_pitching": "League Team Stats",
    "team_stats_fielding": "League Team Stats",
    "team_stats_park": "Park Factors",
    "team_standings": "Standings",
    "lineup_overview": "Lineups",
    "lineup_vs_lhp": "Lineups",
    "lineup_vs_rhp": "Lineups",
    "team_pitching": "Lineups",
    "fielding_stats": "Fielding",
    "fielding_ratings": "Fielding",
    "position_ratings": "Fielding",
    "pitch_ratings": "Pitch Arsenal",
}

# File types whose handler is a recognized-noop (file is understood but
# we don't write anything to the DB yet — usually because no consumer
# feature needs the data). The scanner marks these as `will_skip` so the
# UI can show a clear "skip — not ingested" badge instead of a silent
# success.
NOOP_FILE_TYPES: set[str] = {
    "team_standings",
    "team_stats_fielding",
    # collection_default is the "Overview" view of Collection → Manage Cards.
    # Its columns are aggregates (EYE/CON, FLD/STA), not the per-skill
    # ratings meta scoring needs. We recognize it so staleness can detect
    # the common "user clicked Export without switching view" case.
    "collection_default",
}

# File types the system considers "core" — a refresh with any of these
# missing is incomplete. Used to highlight gaps in the plan preview.
CORE_FILE_TYPES: set[str] = {
    "market",
    "roster_batting",
    "roster_pitching",
    "collection_batting",
    "collection_pitching",
    "lineup_overview",
    "team_pitching",
}


def get_file_category(file_type: str | None) -> str:
    """Map a file_type to its UI bucket ('Market', 'Roster', etc.)."""
    if not file_type:
        return "Unknown"
    return FILE_CATEGORIES.get(file_type, "Other")


# ──────────────────────────────────────────────────────────────────────
# Scan — walk the watch directory and classify every CSV
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FileDescriptor:
    """A single CSV in the watch directory, classified and timestamped."""
    path: str                  # absolute path
    filename: str              # basename
    file_type: str | None      # None = unrecognized (won't import)
    category: str              # UI bucket
    size_bytes: int
    modified: datetime
    league_id: str | None      # lb124, i76, etc. (None for non-league files)
    age_hours: float           # hours since modification (for freshness UI)
    last_import: datetime | None = None  # from ingestion_log per file_type
    last_row_count: int = 0
    # Per-FILENAME last-import timestamp. Different from ``last_import``
    # which is per-file_type. Multiple sibling files share a file_type
    # (e.g. all _overview_*.csv → lineup_overview), so the per-type
    # timestamp collapses them. The per-filename value tells us when
    # exactly THIS file was last ingested. Used by the actionable
    # filter so re-saving a single sibling triggers a re-ingest.
    last_import_for_file: datetime | None = None
    is_core: bool = False      # True if file_type in CORE_FILE_TYPES
    will_skip: bool = False    # True if handler is a known noop (team_standings)
    skip_reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["modified"] = self.modified.isoformat() if self.modified else None
        d["last_import"] = self.last_import.isoformat() if self.last_import else None
        return d


def scan_watch_directory(watch_dir: str) -> list[FileDescriptor]:
    """Walk the watch directory and return a classified list of CSVs.

    Does NOT touch the database — pure filesystem + classification. Use
    plan_refresh() to join this with last-import status from ingestion_log.

    Returns an empty list if the directory doesn't exist. Non-CSV files are
    ignored. Files are sorted by modification time descending (newest first).
    """
    if not watch_dir or not os.path.isdir(watch_dir):
        return []

    now = datetime.now()
    results: list[FileDescriptor] = []
    try:
        entries = os.listdir(watch_dir)
    except OSError:
        return []

    for name in entries:
        full = os.path.join(watch_dir, name)
        if not os.path.isfile(full):
            continue
        if not name.lower().endswith(".csv"):
            continue
        try:
            stat = os.stat(full)
        except OSError:
            continue

        modified = datetime.fromtimestamp(stat.st_mtime)
        age_hours = (now - modified).total_seconds() / 3600.0
        ftype = identify_file_type(full)
        league = detect_league_id(full)

        # Noop file types are recognized but intentionally skipped — mark
        # them so the UI can show a clear "noop" rather than a silent
        # success. Reasons are per-type so the user knows why.
        will_skip = bool(ftype and ftype in NOOP_FILE_TYPES)
        if ftype == "team_standings":
            skip_reason = "recognized but not ingested (standings schema not in DB yet)"
        elif ftype == "team_stats_fielding":
            skip_reason = "recognized but not ingested (team fielding columns not yet consumed)"
        elif ftype == "collection_default":
            skip_reason = ("default Collection view — aggregates only, no per-skill "
                           "ratings. Switch view to Batting Ratings + Pitching Ratings "
                           "and Export each.")
        else:
            skip_reason = ""

        results.append(
            FileDescriptor(
                path=full,
                filename=name,
                file_type=ftype,
                category=get_file_category(ftype),
                size_bytes=stat.st_size,
                modified=modified,
                league_id=league,
                age_hours=age_hours,
                is_core=bool(ftype and ftype in CORE_FILE_TYPES),
                will_skip=will_skip,
                skip_reason=skip_reason,
            )
        )

    results.sort(key=lambda d: d.modified, reverse=True)
    return results


def detect_leagues_in_dir(watch_dir: str) -> list[str]:
    """Return sorted list of distinct league_ids present in the watch dir.

    The user may play multiple leagues (lb124, i76, …) and OOTP exports
    into the same folder. The refresh UI uses this list to offer a league
    picker when more than one is detected.
    """
    descriptors = scan_watch_directory(watch_dir)
    leagues = {d.league_id for d in descriptors if d.league_id}
    return sorted(leagues)


# ──────────────────────────────────────────────────────────────────────
# Ingestion status — what does the DB already know about each file type?
# ──────────────────────────────────────────────────────────────────────

@dataclass
class IngestionStatus:
    """Per-file-type summary from the ingestion_log table."""
    file_type: str
    last_import: datetime | None
    last_row_count: int
    total_imports: int

    def to_dict(self) -> dict:
        return {
            "file_type": self.file_type,
            "last_import": self.last_import.isoformat() if self.last_import else None,
            "last_row_count": self.last_row_count,
            "total_imports": self.total_imports,
        }


def get_last_ingestion_status(conn) -> dict[str, IngestionStatus]:
    """Query ingestion_log for the most recent run per file_type.

    Returns a dict keyed by file_type → IngestionStatus. Missing file_types
    (never imported) are NOT included — callers should treat absence as
    "never run". The UI surfaces both cases (old imports and never-run).
    """
    cursor = conn.execute(
        """
        SELECT file_type,
               MAX(ingested_at) as last_import,
               COUNT(*) as total_imports,
               (SELECT row_count FROM ingestion_log il2
                WHERE il2.file_type = il.file_type
                ORDER BY il2.ingested_at DESC LIMIT 1) as last_rows
        FROM ingestion_log il
        GROUP BY file_type
        """
    )
    status: dict[str, IngestionStatus] = {}
    for row in cursor.fetchall():
        # Row accessor — sqlite3.Row supports dict-style
        ft = row["file_type"]
        last_import_raw = row["last_import"]
        parsed_dt: datetime | None = None
        if last_import_raw:
            try:
                # ingestion_log.ingested_at is stored as TEXT in local time
                parsed_dt = datetime.fromisoformat(str(last_import_raw).replace("T", " "))
            except ValueError:
                parsed_dt = None
        status[ft] = IngestionStatus(
            file_type=ft,
            last_import=parsed_dt,
            last_row_count=int(row["last_rows"] or 0),
            total_imports=int(row["total_imports"] or 0),
        )
    return status


def get_last_ingestion_per_filename(conn) -> dict[str, datetime]:
    """Like ``get_last_ingestion_status`` but keyed on filename.

    Each ``_overview_*.csv`` sibling shares ``file_type='lineup_overview'``,
    so the per-type query collapses them and dedups all of them once any
    one is imported. The UAT 2026-04-25 user feedback was specifically that
    files weren't picking up because of this aggregation: re-saving
    ``_default.csv`` after a sibling refresh kept it filtered out as
    "already imported" via the type's last_import.

    Returning per-file timestamps lets the actionable filter compare each
    file's mtime against the last time THAT file was ingested. Re-saving
    a single sibling now properly re-runs that one file.
    """
    cursor = conn.execute(
        """
        SELECT file_name, MAX(ingested_at) AS last_import
        FROM ingestion_log
        WHERE file_name IS NOT NULL
        GROUP BY file_name
        """
    )
    out: dict[str, datetime] = {}
    for row in cursor.fetchall():
        fn = row["file_name"]
        raw = row["last_import"]
        if not fn or not raw:
            continue
        try:
            out[fn] = datetime.fromisoformat(str(raw).replace("T", " "))
        except ValueError:
            continue
    return out


# ──────────────────────────────────────────────────────────────────────
# Plan — scan + status + gap analysis → structured preview
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RefreshPlan:
    """Preview of what a refresh would do, before committing to run it."""
    watch_dir: str
    files: list[FileDescriptor]
    leagues_detected: list[str]
    missing_core: list[str] = field(default_factory=list)   # core file types absent from disk
    unrecognized: list[FileDescriptor] = field(default_factory=list)  # in dir but no matcher
    categories: dict[str, list[FileDescriptor]] = field(default_factory=dict)  # grouped by category
    total_importable: int = 0
    total_unrecognized: int = 0
    total_will_skip: int = 0

    def summary_line(self) -> str:
        """Human-readable summary for the UI status strip."""
        bits = [
            f"{self.total_importable} file(s) ready to import",
        ]
        if self.total_will_skip:
            bits.append(f"{self.total_will_skip} recognized-noop")
        if self.total_unrecognized:
            bits.append(f"{self.total_unrecognized} unrecognized")
        if self.missing_core:
            bits.append(f"{len(self.missing_core)} core file(s) missing")
        return " • ".join(bits)


def plan_refresh(watch_dir: str, conn) -> RefreshPlan:
    """Build a dry-run preview of a refresh.

    Joins a scan of the watch directory with ingestion_log status so the UI
    can show:
      - files grouped by category (Market, Roster, Lineups, …)
      - each file's last-imported timestamp (or "never")
      - which core file types are MISSING from the directory
      - which files are unrecognized (would be skipped)
    """
    descriptors = scan_watch_directory(watch_dir)
    status_map = get_last_ingestion_status(conn) if conn is not None else {}
    file_status_map = (get_last_ingestion_per_filename(conn)
                        if conn is not None else {})

    # Join last-import info onto each descriptor — both per-type
    # (UI display) and per-filename (actionable filter).
    for d in descriptors:
        if d.file_type and d.file_type in status_map:
            s = status_map[d.file_type]
            d.last_import = s.last_import
            d.last_row_count = s.last_row_count
        # Per-filename: catches sibling files (lineup_overview siblings,
        # league stats variants) that share a file_type. ingestion_log's
        # file_name field is the basename, not the absolute path, so we
        # match on basename here.
        if file_status_map:
            d.last_import_for_file = file_status_map.get(d.filename)

    # Group by category (only recognized files — unrecognized go in their own bucket)
    categories: dict[str, list[FileDescriptor]] = {}
    unrecognized: list[FileDescriptor] = []
    for d in descriptors:
        if d.file_type is None:
            unrecognized.append(d)
            continue
        categories.setdefault(d.category, []).append(d)

    # Stable category order by category name, then files within by modified desc
    categories = {
        k: sorted(v, key=lambda x: x.modified, reverse=True)
        for k, v in sorted(categories.items(), key=lambda kv: kv[0])
    }

    present_types = {d.file_type for d in descriptors if d.file_type}
    missing_core = sorted(CORE_FILE_TYPES - present_types)

    leagues = sorted({d.league_id for d in descriptors if d.league_id})

    total_importable = sum(
        1 for d in descriptors if d.file_type and not d.will_skip
    )
    total_will_skip = sum(1 for d in descriptors if d.will_skip)

    return RefreshPlan(
        watch_dir=watch_dir,
        files=descriptors,
        leagues_detected=leagues,
        missing_core=missing_core,
        unrecognized=unrecognized,
        categories=categories,
        total_importable=total_importable,
        total_unrecognized=len(unrecognized),
        total_will_skip=total_will_skip,
    )


# ──────────────────────────────────────────────────────────────────────
# Execute — run ingestion with progress streaming
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RefreshResult:
    """Aggregated outcome of a refresh run."""
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    files_total: int
    files_succeeded: int
    files_skipped: int
    files_failed: int
    rows_ingested: int
    per_file: list[dict]              # raw ingest_file results (one per file)
    league_id: str | None
    history_snapshot: dict | None     # result of snapshot_player_history, if run
    errors: list[str]                 # one-line summaries of each failure
    html_snapshot: dict | None = None # result of HTML box-score + game-log ingest

    def to_dict(self) -> dict:
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat()
        d["finished_at"] = self.finished_at.isoformat()
        return d


def execute_refresh(
    files: Iterable[FileDescriptor | str],
    league_id: str | None = None,
    use_history: bool = True,
    games_into_season: int | None = None,
    team_record: str | None = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    allow_cross_league: bool = False,
) -> RefreshResult:
    """Run ingestion for a set of files and return an aggregated result.

    Parameters
    ----------
    files : iterable of FileDescriptor or str paths
        Files to import. Accepts either FileDescriptors from plan_refresh()
        or raw paths (handy for file-watcher or CLI callers).
    league_id : str, optional
        If provided, passed to the ingestion pipeline and used as the
        league for history snapshots. Auto-detected from file prefixes
        when omitted.
    use_history : bool
        If True, take a player history snapshot after ingesting all files.
        This is what makes trending + calibration views work — the snapshot
        locks today's roster/meta/stats into a history table.
    games_into_season : int, optional
        Passed through to the history snapshot (lets the user tag how
        many games into the season the snapshot represents).
    team_record : str, optional
        Passed through to the history snapshot (e.g. "20-18").
    on_progress : callable(current, total, filename) → None, optional
        Called before each file is processed so the UI can stream progress.

    Returns
    -------
    RefreshResult with aggregated counts, per-file details, and any
    history snapshot info.
    """
    # Normalize inputs: accept FileDescriptor or raw string paths
    file_paths: list[str] = []
    for f in files:
        if isinstance(f, FileDescriptor):
            file_paths.append(f.path)
        else:
            file_paths.append(str(f))

    # Sort imports so the market CSV lands first, then roster / lineup /
    # collection files, then stats / fielding / park data. The card_id
    # matching done by roster and stats ingestion depends on the cards table
    # being populated — if a stats file arrives before the market file is
    # processed, every card_id lookup comes back NULL and the rows get
    # orphaned. Previously the order was whatever mtime-sorted, which was
    # unstable if OOTP re-exported the market file a second time.
    _INGEST_ORDER = {
        # Tier 0: market must populate cards table first.
        "market": 0,
        # Tier 1: collection / roster / lineups set ownership + lineup roles.
        "collection_batting": 1, "collection_pitching": 1,
        "roster_batting": 1, "roster_pitching": 1,
        "lineup_vs_rhp": 1, "lineup_vs_lhp": 1, "lineup_overview": 1,
        # Tier 2: ratings-adjacent files (pitch, fielding, position) refine
        # card metadata with extended ratings.
        "fielding_ratings": 2, "fielding_stats": 2,
        "pitch_ratings": 2, "position_ratings": 2,
        "stats_batting_ratings": 2, "stats_pitching_ratings": 2,
        # League player default: must run AFTER the two ratings files so the
        # companion files exist in the same directory when we read them.
        "league_player_default": 2,
        # Tier 3: stats — join to card_id set up by tier 0.
        "stats_batting": 3, "stats_pitching": 3,
        "roster_batting_stats": 3, "roster_pitching_stats": 3,
        # Tier 4: team-level park / team stats (framing context).
        "team_stats_batting": 4, "team_stats_pitching": 4,
        "team_stats_fielding": 4, "team_stats_park": 4,
        "team_standings": 4, "team_pitching": 4,
    }

    def _ingest_priority(path: str) -> tuple[int, str]:
        ftype = identify_file_type(path)
        # Unknown types land in the middle (tier 5) in stable filename order.
        # They won't crash ingestion (the handlers no-op on unknown), but we
        # still want deterministic ordering for reproducibility.
        return (_INGEST_ORDER.get(ftype, 5), Path(path).name)

    file_paths.sort(key=_ingest_priority)

    # Auto-detect league_id if not provided
    if league_id is None:
        for fp in file_paths:
            lid = detect_league_id(fp)
            if lid:
                league_id = lid
                break

    started = datetime.now()
    per_file: list[dict] = []
    errors: list[str] = []
    succeeded = 0
    skipped = 0
    failed = 0
    rows_total = 0

    total = len(file_paths)
    for i, fp in enumerate(file_paths):
        if on_progress is not None:
            try:
                on_progress(i, total, Path(fp).name)
            except Exception:
                pass  # progress callback errors must never block ingestion
        try:
            result = ingest_file(fp, league_id=league_id, allow_cross_league=allow_cross_league)
            result["filename"] = Path(fp).name
            per_file.append(result)
            status = result.get("status")
            if status == "success":
                succeeded += 1
                rows_total += int(result.get("rows", 0) or 0)
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                errors.append(f"{Path(fp).name}: {result.get('reason', 'unknown')}")
        except Exception as e:
            failed += 1
            per_file.append({
                "filename": Path(fp).name,
                "status": "error",
                "rows": 0,
                "error": str(e),
            })
            errors.append(f"{Path(fp).name}: {e}")

    # History snapshot (optional)
    history_result: dict | None = None
    if use_history and league_id and succeeded > 0:
        try:
            from app.core.history import snapshot_player_history, ensure_league_exists
            ensure_league_exists(league_id)
            history_result = snapshot_player_history(
                league_id=league_id,
                games_into_season=games_into_season,
                team_record=team_record,
            )
        except Exception as e:
            history_result = {"error": str(e)}

    # HTML ingest: box scores + game logs from the OOTP save directory.
    # Configured via ``config.yaml:save_game_dir`` — the .pt folder where
    # OOTP drops news/html/box_scores/*.html and news/html/game_logs/*.html.
    # Both ingests are idempotent (skip_existing=True) so re-running a
    # refresh only picks up new files.
    html_result: dict | None = None
    try:
        from app.core.database import load_config
        cfg = load_config()
        save_dir = cfg.get('save_game_dir')
        if save_dir:
            from app.core.html_ingest import ingest_all_box_scores, ingest_all_game_logs
            box_r = ingest_all_box_scores(save_dir, skip_existing=True)
            log_r = ingest_all_game_logs(save_dir, skip_existing=True)
            html_result = {'box_scores': box_r, 'game_logs': log_r}
    except Exception as e:
        html_result = {'error': str(e)}

    # Final progress tick so the UI can land on 100%
    if on_progress is not None:
        try:
            on_progress(total, total, "done")
        except Exception:
            pass

    finished = datetime.now()
    return RefreshResult(
        started_at=started,
        finished_at=finished,
        duration_seconds=(finished - started).total_seconds(),
        files_total=total,
        files_succeeded=succeeded,
        files_skipped=skipped,
        files_failed=failed,
        rows_ingested=rows_total,
        per_file=per_file,
        league_id=league_id,
        history_snapshot=history_result,
        errors=errors,
        html_snapshot=html_result,
    )


# ──────────────────────────────────────────────────────────────────────
# Convenience: format helpers the UI uses often enough to centralize
# ──────────────────────────────────────────────────────────────────────

def format_age(hours: float) -> str:
    """Compact age label: '3m', '2h', '1d', '2w'."""
    if hours < 1 / 60:
        return "<1m"
    if hours < 1:
        return f"{int(hours * 60)}m"
    if hours < 24:
        return f"{int(hours)}h"
    days = hours / 24
    if days < 14:
        return f"{int(days)}d"
    weeks = days / 7
    return f"{int(weeks)}w"


def format_size(bytes_: int) -> str:
    """Compact size label: '1.2K', '3.4M'."""
    if bytes_ < 1024:
        return f"{bytes_}B"
    if bytes_ < 1024 * 1024:
        return f"{bytes_ / 1024:.0f}K"
    return f"{bytes_ / (1024 * 1024):.1f}M"
