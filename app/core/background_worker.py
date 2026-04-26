"""Background file-watcher + ingestion worker.

Watches the OOTP CSV dump directory + save-game HTML directories. When new
files land (or existing ones change), debounces the burst, then runs the
full ingestion pipeline (CSV → DB → meta recalc) without requiring a manual
refresh click from the user.

Lifecycle:
    * ``start_worker()`` — idempotent; spawns a daemon thread the first
      time it's called in this Python process. Safe to call from every
      Streamlit page render (they'll no-op after the first).
    * The worker sits in a ``threading.Event.wait()`` loop. File events
      set the ``trigger`` event; the worker wakes, waits for the debounce
      window to be quiet, and runs ingestion.
    * State is persisted in the ``worker_status`` singleton row so every
      Streamlit session (or external reader) sees the same state.

Design notes:
    * Module-level singleton, NOT per-session — Streamlit spins up fresh
      script contexts per user but we only want one watcher per process.
    * The ``Observer`` from ``watchdog`` runs its own thread; our worker
      thread just waits on an Event it sets when a handler fires.
    * All DB writes open fresh connections. SQLite is safe for concurrent
      reads + one writer; Streamlit pages read, worker writes.
    * Errors are caught and written to ``worker_status.last_message`` so
      the UI can surface them instead of silently dying.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Module-level singleton — one worker per Python process.
# ──────────────────────────────────────────────────────────────────────
_WORKER_LOCK = threading.Lock()
_WORKER_THREAD: Optional[threading.Thread] = None
_WORKER_OBSERVER = None  # watchdog Observer
_TRIGGER_EVENT = threading.Event()
_SHUTDOWN_EVENT = threading.Event()
_LAST_EVENT_TS = 0.0    # monotonic time of most recent file change


# Debounce window: after the last file event, wait this long before
# running the refresh. OOTP dumps ~20 files when you click "Export" —
# we want one refresh run, not twenty.
DEBOUNCE_SECONDS = 3.0

# Poll fallback: even when watchdog misses an event (network drives,
# OneDrive sync, etc.), wake up every N seconds to check for changes.
POLL_INTERVAL_SECONDS = 30.0


def _update_status(**fields) -> None:
    """Write a partial update to the worker_status singleton row."""
    if not fields:
        return
    try:
        from app.core.database import get_connection
        conn = get_connection()
        # SET clause
        cols = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values())
        conn.execute(
            f"UPDATE worker_status SET {cols} WHERE id = 1",
            values,
        )
        conn.commit()
    except Exception as e:
        logger.warning("Worker status update failed: %s", e)


def _bump_data_version() -> int:
    """Increment data_version (monotonic). Returns the new value."""
    try:
        from app.core.database import get_connection
        conn = get_connection()
        conn.execute("UPDATE worker_status SET data_version = data_version + 1 WHERE id = 1")
        conn.commit()
        row = conn.execute("SELECT data_version FROM worker_status WHERE id = 1").fetchone()
        return int(row[0]) if row else 0
    except Exception as e:
        logger.warning("data_version bump failed: %s", e)
        return 0


def _do_refresh() -> dict:
    """Run the full refresh pipeline once. Returns summary dict."""
    from app.core.database import load_config, get_connection
    from app.core.data_refresh import plan_refresh, execute_refresh
    from app.core.ingestion import recalculate_all_meta_scores

    started = time.time()
    cfg = load_config()
    watch_dir = cfg.get('watch_directory')
    summary = {
        'files_succeeded': 0, 'rows': 0, 'box_scores': 0, 'game_logs': 0,
        'duration': 0.0, 'message': '', 'error': None,
    }

    # CSV ingestion (if watch_dir configured and populated)
    if watch_dir and os.path.isdir(watch_dir):
        try:
            _update_status(status='ingesting', last_message='Scanning CSVs…')
            conn = get_connection()
            plan = plan_refresh(watch_dir, conn)
            # Skip files whose mtime is <= last_import — OOTP doesn't
            # rewrite a CSV until the user exports again. Prior behavior
            # re-ran ingestion 100-300x/day per file because this check
            # didn't exist; each run was a full DELETE+INSERT cycle on
            # tables like batting_stats. With the filter, a "nothing new"
            # poll becomes a no-op.
            # Per-file actionable filter (UAT 2026-04-25 user feedback).
            # Compare each file's mtime against its OWN last_import
            # (filename-keyed) instead of the type's aggregated
            # last_import. Sibling files like _overview_default.csv vs
            # _overview_batting_stats_1.csv share file_type=lineup_overview
            # so the type-level filter falsely dedups them. Per-file
            # comparison fixes "I re-saved _default.csv but it didn't
            # re-ingest" by tracking each file independently.
            def _file_is_fresh(d):
                stamp = d.last_import_for_file or d.last_import
                if stamp is None:
                    return True  # never ingested
                if not d.modified:
                    return False
                return d.modified > stamp

            actionable = [
                d for d in plan.files
                if d.file_type and not d.will_skip and _file_is_fresh(d)
            ]
            skipped_unchanged = sum(
                1 for d in plan.files
                if d.file_type and not d.will_skip and not _file_is_fresh(d)
            )
            if skipped_unchanged:
                logger.info("Worker skipped %d unchanged files", skipped_unchanged)
            if actionable:
                result = execute_refresh(
                    actionable,
                    league_id=cfg.get('active_league'),
                    use_history=False,
                    allow_cross_league=True,
                )
                summary['files_succeeded'] = result.files_succeeded
                summary['rows'] = result.rows_ingested
                if result.html_snapshot and isinstance(result.html_snapshot, dict):
                    b = result.html_snapshot.get('box_scores') or {}
                    g = result.html_snapshot.get('game_logs') or {}
                    summary['box_scores'] = int(b.get('processed') or 0)
                    summary['game_logs'] = int(g.get('processed') or 0)
        except Exception as e:
            logger.exception("CSV refresh failed")
            summary['error'] = f"CSV refresh: {e}"

    # HTML-only ingest (runs even without watch_dir if save_game_dir exists)
    save_dir = cfg.get('save_game_dir')
    if save_dir and (not summary['box_scores'] and not summary['game_logs']):
        try:
            _update_status(status='ingesting', last_message='Scanning HTML…')
            from app.core.html_ingest import (
                ingest_all_box_scores, ingest_all_game_logs,
            )
            bx = ingest_all_box_scores(save_dir, skip_existing=True)
            gl = ingest_all_game_logs(save_dir, skip_existing=True)
            summary['box_scores'] += int(bx.get('processed') or 0)
            summary['game_logs'] += int(gl.get('processed') or 0)
        except Exception as e:
            logger.exception("HTML refresh failed")
            err = f"HTML refresh: {e}"
            summary['error'] = err if not summary['error'] else summary['error'] + '; ' + err

    # Meta recalc — only if anything actually ingested.
    # Retry on DB lock so stored meta doesn't end up stale when the
    # Streamlit process happens to hold a write lock during the first
    # attempt. 3 attempts with exponential backoff; failure is logged
    # AND reflected in status so the UI can surface it.
    summary['recalc_ok'] = False
    if (summary['files_succeeded'] or summary['box_scores'] or summary['game_logs']):
        for attempt in range(3):
            try:
                _update_status(status='recalc',
                               last_message=f'Recalculating meta… (attempt {attempt+1})')
                result = recalculate_all_meta_scores()
                # Accept 'success' only. 'partial' means roster sync failed
                # so stored roster metas are stale even though cards are fresh.
                if isinstance(result, dict) and result.get('status') == 'partial':
                    logger.warning("Meta recalc partial: %s",
                                   result.get('message', '?'))
                    summary['recalc_ok'] = False
                    summary['error'] = (
                        f"Partial recalc: {result.get('roster_sync_error') or 'unknown'}"
                    )
                else:
                    summary['recalc_ok'] = True
                break
            except Exception as e:
                msg = str(e)
                if 'locked' in msg.lower() and attempt < 2:
                    logger.warning("Meta recalc locked, retry %d/3 in %ds",
                                    attempt + 1, 2 ** attempt)
                    time.sleep(2 ** attempt)
                    continue
                logger.exception("Meta recalc failed (attempt %d)", attempt + 1)
                err = f"Meta recalc: {e}"
                summary['error'] = err if not summary['error'] else summary['error'] + '; ' + err
                break

        # Recommendation-tracker sweep — if new roster data landed, detect
        # which pending recs Cameron actioned, then re-measure outcomes.
        try:
            from app.core.recommendation_tracker import (
                detect_actioned_recommendations, measure_outcomes,
            )
            detect_actioned_recommendations()
            measure_outcomes()
        except Exception as e:
            logger.exception("Rec tracker sweep failed")

        # Derived analytics rebuild — batter vs pitcher splits, pitcher
        # fatigue, clutch profile, park-adjusted, regression_v2,
        # opponent scouting, price velocity, meta confidence. All read
        # from the freshly-ingested raw tables. Safe to fail: these are
        # cosmetic for the UI and next refresh will pick up any failure.
        try:
            _update_status(status='recalc', last_message='Rebuilding derived stats…')
            from app.core.derived_stats import build_all as build_derived_all
            derived = build_derived_all()
            totals = sum(v.get('rows_written', 0) for v in derived.values())
            logger.info("derived_stats rebuild: %d total rows across %d tables",
                        totals, len(derived))
        except Exception as e:
            logger.exception("derived_stats rebuild failed")
            err = f"derived rebuild: {e}"
            summary['error'] = err if not summary['error'] else summary['error'] + '; ' + err

        # Per-league meta recalc (UAT 2026-04-25 #5). Refreshes
        # ``card_meta_by_league`` for every active tier so the optimizer
        # page doesn't have to wait for the user to click the recalc
        # button after fresh data lands. Cheap (~0.2s per league on the
        # current DB) so it's safe to run every refresh.
        try:
            _update_status(status='recalc', last_message='Per-league meta recalc…')
            from app.core.database import get_connection as _gc_pl
            from app.core.meta_scoring import recalc_meta_scores_per_league
            _conn_pl = _gc_pl()
            try:
                _leagues = [r[0] for r in _conn_pl.execute(
                    "SELECT DISTINCT league_id FROM batting_stats "
                    "WHERE league_id IS NOT NULL"
                ).fetchall() if r[0]]
            finally:
                _conn_pl.close()
            for lg in _leagues:
                try:
                    recalc_meta_scores_per_league(lg, only_owned=False)
                except Exception:
                    logger.exception("per-league recalc failed for %s", lg)
            logger.info("per-league meta recalc done for %d tiers", len(_leagues))
        except Exception as e:
            logger.exception("per-league meta recalc orchestration failed")

        # New-card intake — any owned cards not yet tracked get an intake
        # row + a pending recommendation_log entry, which the council
        # sweep below will review next.
        try:
            from app.core.card_intake import detect_new_cards
            n_new = detect_new_cards()
            if n_new:
                logger.info("card_intake: %d new arrivals logged", n_new)
        except Exception:
            logger.exception("card_intake sweep failed")

        # Council sweep — silent LLM verification on recent recs. Results
        # get folded into chain-table tooltips only when a verdict DISSENTS
        # or a provider has low confidence. Concurrences never show UI.
        try:
            _sweep_council_reviews(max_per_sweep=8)
        except Exception as e:
            logger.exception("Council sweep failed")

    # Autonomous sweeps — run EVERY tick, even when no files changed.
    # Catches silent drift (calibration rotated, stored meta outdated,
    # baselines shifted) without waiting for explicit user action.
    try:
        from app.core.autonomous_sweeps import run_autonomous_sweeps
        auto = run_autonomous_sweeps()
        summary['stale_cards_fixed'] = auto.get('stale_cards_fixed') or 0
        if auto.get('stale_cards_fixed'):
            logger.info(
                "autonomous sweep: fixed %d stale cards (rating-base drift)",
                auto['stale_cards_fixed'],
            )
    except Exception as e:
        logger.exception("Autonomous sweep failed")

    # Retention sweep — keep the DB from growing unboundedly:
    #   - ingestion_log: keep 30 days (file mtime debugging)
    #   - recommendations: keep last batch per calendar day for 30 days,
    #     drop older days entirely (intraday churn is noise, daily snapshots
    #     are enough for "what did we recommend when" history)
    # Run quietly; errors shouldn't block the refresh.
    try:
        from app.core.database import get_connection as _gc
        _rc = _gc()
        try:
            r1 = _rc.execute(
                "DELETE FROM ingestion_log WHERE ingested_at < datetime('now', '-30 days')"
            ).rowcount
            r2 = _rc.execute(
                """DELETE FROM recommendations
                   WHERE DATE(created_at) < DATE('now', '-30 days')
                      OR created_at NOT IN (
                          SELECT MAX(created_at) FROM recommendations
                          GROUP BY DATE(created_at)
                      )"""
            ).rowcount
            _rc.commit()
            if r1 or r2:
                logger.info(
                    "retention sweep: %d ingestion_log, %d recommendations pruned",
                    r1, r2,
                )
        finally:
            _rc.close()
    except Exception:
        logger.exception("retention sweep failed")

    summary['duration'] = time.time() - started

    # Persist status
    message = (
        f"Ingested {summary['files_succeeded']} CSV file(s), "
        f"{summary['box_scores']} box score(s), "
        f"{summary['game_logs']} game log(s) in {summary['duration']:.1f}s"
    )
    if summary['error']:
        message += f" — error: {summary['error']}"
        status = 'error'
    else:
        status = 'idle'
    summary['message'] = message

    _update_status(
        status=status,
        last_refresh_at=datetime.now().isoformat(timespec='seconds'),
        last_refresh_duration_s=summary['duration'],
        last_refresh_files=summary['files_succeeded'],
        last_refresh_rows=summary['rows'],
        last_box_scores=summary['box_scores'],
        last_game_logs=summary['game_logs'],
        last_message=message,
    )
    # Only bump data_version when recalc actually succeeded. If recalc
    # failed (DB lock, formula error), the stored meta is still stale
    # and we don't want the UI to think fresh data landed.
    any_ingested = (summary['files_succeeded'] or summary['box_scores']
                     or summary['game_logs'])
    if any_ingested and summary.get('recalc_ok'):
        _bump_data_version()
    elif any_ingested and not summary.get('recalc_ok'):
        # Flag stale-meta state in status so the UI can surface a warning.
        _update_status(status='error',
                       last_message=message + ' — meta is STALE, retry pending')

    return summary


def _worker_loop() -> None:
    """Daemon loop: wait for trigger, debounce, run refresh."""
    logger.info("background_worker loop started")
    while not _SHUTDOWN_EVENT.is_set():
        # Wait for a trigger or poll timeout, whichever comes first
        triggered = _TRIGGER_EVENT.wait(timeout=POLL_INTERVAL_SECONDS)
        if _SHUTDOWN_EVENT.is_set():
            break

        if triggered:
            # Debounce: wait until DEBOUNCE_SECONDS have passed with no
            # new events, up to a max of 30s (to avoid starvation if OOTP
            # is actively writing).
            _TRIGGER_EVENT.clear()
            wait_start = time.monotonic()
            while time.monotonic() - _LAST_EVENT_TS < DEBOUNCE_SECONDS:
                if _SHUTDOWN_EVENT.is_set():
                    return
                if time.monotonic() - wait_start > 30.0:
                    break
                time.sleep(0.25)

        # Fire the refresh
        try:
            _do_refresh()
        except Exception as e:
            logger.exception("background refresh iteration failed")
            _update_status(status='error', last_message=f"Refresh error: {e}")


class _ChangeHandler:
    """watchdog FileSystemEventHandler — just sets the trigger event.

    We don't differentiate file types here; the refresh pipeline handles
    categorization. We DO skip temporary / partial writes (OOTP creates
    .tmp, .bak files during export) to avoid false wakes.
    """
    def _relevant(self, path: str) -> bool:
        p = path.lower()
        if p.endswith('.tmp') or p.endswith('.bak') or p.endswith('~'):
            return False
        # Skip Streamlit itself rewriting files in app/
        if '.streamlit' in p or '\\app\\' in p or '/app/' in p:
            return False
        return p.endswith('.csv') or p.endswith('.html')

    def on_created(self, event):
        if getattr(event, 'is_directory', False):
            return
        if not self._relevant(event.src_path):
            return
        global _LAST_EVENT_TS
        _LAST_EVENT_TS = time.monotonic()
        _TRIGGER_EVENT.set()

    def on_modified(self, event):
        if getattr(event, 'is_directory', False):
            return
        if not self._relevant(event.src_path):
            return
        global _LAST_EVENT_TS
        _LAST_EVENT_TS = time.monotonic()
        _TRIGGER_EVENT.set()

    def on_moved(self, event):
        # File moved into watched dir (e.g. atomic rename after write)
        if getattr(event, 'is_directory', False):
            return
        dest = getattr(event, 'dest_path', '') or ''
        if dest and self._relevant(dest):
            global _LAST_EVENT_TS
            _LAST_EVENT_TS = time.monotonic()
            _TRIGGER_EVENT.set()


def start_worker(force: bool = False) -> bool:
    """Idempotent — spawn the watcher + refresh thread once per process.

    Returns True if the worker started (or was already running), False if
    no watch directories are configured and nothing was started.
    """
    global _WORKER_THREAD, _WORKER_OBSERVER

    with _WORKER_LOCK:
        if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive() and not force:
            return True

        from app.core.database import load_config
        cfg = load_config()
        watch_dir = cfg.get('watch_directory')
        save_dir = cfg.get('save_game_dir')

        dirs_to_watch = []
        if watch_dir and os.path.isdir(watch_dir):
            dirs_to_watch.append(watch_dir)
        if save_dir:
            for sub in ('news/html/box_scores', 'news/html/game_logs'):
                p = os.path.join(save_dir, sub)
                if os.path.isdir(p):
                    dirs_to_watch.append(p)

        if not dirs_to_watch:
            logger.warning("No watch directories — worker not started")
            _update_status(status='idle',
                           last_message='No watch directories configured')
            return False

        # Spin up watchdog Observer
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            logger.error("watchdog not installed — worker cannot start")
            _update_status(status='error', last_message='watchdog not installed')
            return False

        # watchdog requires a proper subclass; wrap our _ChangeHandler.
        handler_inner = _ChangeHandler()
        class _Adapter(FileSystemEventHandler):
            def on_created(self, event):  handler_inner.on_created(event)
            def on_modified(self, event): handler_inner.on_modified(event)
            def on_moved(self, event):    handler_inner.on_moved(event)

        observer = Observer()
        adapter = _Adapter()
        for d in dirs_to_watch:
            observer.schedule(adapter, d, recursive=False)
        observer.daemon = True
        observer.start()
        _WORKER_OBSERVER = observer

        # Spin up refresh worker
        _SHUTDOWN_EVENT.clear()
        t = threading.Thread(
            target=_worker_loop, name='ootp-bg-worker', daemon=True,
        )
        t.start()
        _WORKER_THREAD = t

        _update_status(
            status='idle',
            last_message=f"Watching {len(dirs_to_watch)} director(ies)",
        )

        # Kick an initial refresh so the UI immediately reflects current
        # disk state (catches files landed while the app was offline).
        _LAST_EVENT_TS = time.monotonic()
        _TRIGGER_EVENT.set()

        return True


def stop_worker() -> None:
    """Shut down the observer + worker thread. Rarely used — mainly tests."""
    global _WORKER_THREAD, _WORKER_OBSERVER
    _SHUTDOWN_EVENT.set()
    _TRIGGER_EVENT.set()
    if _WORKER_OBSERVER is not None:
        try:
            _WORKER_OBSERVER.stop()
            _WORKER_OBSERVER.join(timeout=2.0)
        except Exception:
            pass
        _WORKER_OBSERVER = None
    if _WORKER_THREAD is not None:
        try:
            _WORKER_THREAD.join(timeout=3.0)
        except Exception:
            pass
        _WORKER_THREAD = None


def is_running() -> bool:
    return _WORKER_THREAD is not None and _WORKER_THREAD.is_alive()


def trigger_refresh_now() -> None:
    """External API — force an immediate refresh (used by UI buttons)."""
    global _LAST_EVENT_TS
    _LAST_EVENT_TS = time.monotonic() - DEBOUNCE_SECONDS - 1  # skip debounce
    _TRIGGER_EVENT.set()


def _sweep_council_reviews(max_per_sweep: int = 8) -> int:
    """Fire LLM verification on recent engine recommendations that lack a
    council review. Returns count of recs reviewed in this sweep.

    The LLM is embedded in the analysis pipeline rather than a user-facing
    feature — verdicts land in ``council_review`` and are rendered as
    tooltip annotations on the canonical engine picks.
    """
    try:
        from app.core.database import get_connection
        from app.core.council import review_recommendation
        from app.core.llm_providers import list_providers
    except Exception:
        return 0
    if not list_providers():
        return 0
    conn = get_connection()
    # Find recs from the last 24h that don't have a council_review row yet.
    try:
        rows = conn.execute(
            """
            SELECT rl.id FROM recommendation_log rl
            LEFT JOIN council_review cr ON cr.rec_id = rl.id
            WHERE cr.id IS NULL
              AND rl.created_at > datetime('now', '-24 hours')
              AND rl.verdict = 'pending'
              AND (rl.rec_type IN ('buy','promote','platoon','bench','sell','hook')
                   OR rl.rec_type IS NOT NULL)
            ORDER BY rl.created_at DESC
            LIMIT ?
            """,
            (max_per_sweep,),
        ).fetchall()
    except Exception:
        return 0
    reviewed = 0
    for r in rows:
        try:
            review_recommendation(int(r['id']), primary_only=True)
            reviewed += 1
        except Exception as e:
            logger.warning("review_recommendation(%s) failed: %s", r['id'], e)
    if reviewed:
        _bump_data_version()   # so the UI refreshes to show new verdicts
    return reviewed


def get_worker_status() -> dict:
    """Read the worker_status singleton as a plain dict."""
    try:
        from app.core.database import get_connection
        conn = get_connection()
        row = conn.execute("SELECT * FROM worker_status WHERE id = 1").fetchone()
        if not row:
            return {}
        return dict(row)
    except Exception:
        return {}
