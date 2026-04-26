"""SQLite database module for OOTP PT Optimizer."""
import sqlite3
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config() -> dict:
    """Read config.yaml from the project root and return the dict."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict) -> None:
    """Write the full config dict back to config.yaml.

    Preserves YAML formatting as best we can via safe_dump. Used by the
    Settings UI when editing LLM providers or other runtime settings.
    """
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def get_db_path() -> str:
    """Get the database path from config, defaulting to data/ootp_optimizer.db."""
    try:
        config = load_config()
        db_rel = config.get("database_path", "data/ootp_optimizer.db")
    except Exception:
        db_rel = "data/ootp_optimizer.db"
    return str(PROJECT_ROOT / db_rel)


# Process-level guard for the auto-sync. We only want to run the roster ↔ cards
# meta alignment once per Python process — Streamlit re-imports modules between
# script runs but the same module object persists for the lifetime of the worker.
_AUTO_META_SYNC_DONE = False

# Process-level guard for the latest-snapshot views. ``init_db`` only fires
# on the first call per process and pages may be loaded against a DB that
# was created by an older build. This flag lets ``get_connection`` install
# the views on demand without hammering the DDL on every connection.
_LATEST_VIEWS_ENSURED = False


def _ensure_latest_views(conn: sqlite3.Connection) -> None:
    """Idempotently install ``batting_stats_latest`` / ``pitching_stats_latest``
    plus the per-league meta cache table ``card_meta_by_league``.

    These views encode the snapshot+jersey dedup that callers used to do
    inline (or skip — UAT 2026-04-25 §5.4 found multiple consumers reading
    raw stats and getting per-card row multiplication). Centralizing the
    dedup keeps consumer queries single-line and prevents drift.

    Called once per Python process from ``get_connection``. Safe to re-run
    — the DDL itself uses ``CREATE VIEW IF NOT EXISTS`` semantics via
    ``DROP VIEW IF EXISTS`` first.
    """
    global _LATEST_VIEWS_ENSURED
    if _LATEST_VIEWS_ENSURED:
        return
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS card_meta_by_league (
                card_id INTEGER NOT NULL,
                league_id TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('batting', 'pitching')),
                meta_score REAL NOT NULL,
                weight_source TEXT,
                recalc_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (card_id, league_id, side),
                FOREIGN KEY (card_id) REFERENCES cards(card_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cmbl_card_league "
            "ON card_meta_by_league(card_id, league_id)"
        )
        conn.executescript("""
        DROP VIEW IF EXISTS batting_stats_latest;
        CREATE VIEW batting_stats_latest AS
            SELECT bs.* FROM batting_stats bs
            INNER JOIN (
                SELECT card_id, league_id, jersey_number,
                       MAX(snapshot_date) AS max_date
                FROM batting_stats
                WHERE card_id IS NOT NULL
                GROUP BY card_id, league_id, jersey_number
            ) latest
              ON bs.card_id = latest.card_id
             AND IFNULL(bs.league_id, '') = IFNULL(latest.league_id, '')
             AND IFNULL(bs.jersey_number, -1) = IFNULL(latest.jersey_number, -1)
             AND bs.snapshot_date = latest.max_date;

        DROP VIEW IF EXISTS pitching_stats_latest;
        CREATE VIEW pitching_stats_latest AS
            SELECT ps.* FROM pitching_stats ps
            INNER JOIN (
                SELECT card_id, league_id, jersey_number,
                       MAX(snapshot_date) AS max_date
                FROM pitching_stats
                WHERE card_id IS NOT NULL
                GROUP BY card_id, league_id, jersey_number
            ) latest
              ON ps.card_id = latest.card_id
             AND IFNULL(ps.league_id, '') = IFNULL(latest.league_id, '')
             AND IFNULL(ps.jersey_number, -1) = IFNULL(latest.jersey_number, -1)
             AND ps.snapshot_date = latest.max_date;
        """)
        conn.commit()
        _LATEST_VIEWS_ENSURED = True
    except Exception:
        # Worst case: views remain missing; consumers continue to use the
        # base tables. Logged at the call site if needed.
        pass


def get_connection() -> sqlite3.Connection:
    """Return a sqlite3 connection with row_factory = sqlite3.Row.

    Side effect: on the first call in this process, runs
    ``sync_roster_meta_from_cards`` to align stale roster.meta_score values
    with the live cards table. This closes the cross-page Bichette 622-vs-666
    discrepancy without requiring every page to remember to call the sync.
    """
    global _AUTO_META_SYNC_DONE
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row

    # Enable WAL mode + busy_timeout so the background worker and Streamlit
    # pages can read/write concurrently without "database is locked" errors.
    # WAL allows one writer + many readers at once. busy_timeout makes the
    # second writer wait rather than fail immediately.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")   # 10s
        conn.execute("PRAGMA synchronous=NORMAL")   # WAL-safe, faster commits
    except Exception:
        pass

    if not _AUTO_META_SYNC_DONE:
        _AUTO_META_SYNC_DONE = True  # set first to avoid re-entry on errors
        try:
            sync_roster_meta_from_cards(conn)
        except Exception:
            pass  # Sync is best-effort; never block a page from loading.

    _ensure_latest_views(conn)

    return conn


def get_data_freshness(conn: sqlite3.Connection | None = None) -> dict:
    """Detect when the cards list is newer than the Toronto team roster.

    Uses `ingestion_log.ingested_at` as the source of truth — it always
    writes via CURRENT_TIMESTAMP (UTC) regardless of which ingestion
    function was called, so comparisons are apples-to-apples even if the
    individual data tables disagree on timezone conventions.

    Staleness is the gap in hours between the most recent `market` import
    (pt_card_list.csv → cards.owned) and the most recent TEAM roster
    import (the group of CSVs that tell us Toronto's active lineup). If
    the cards import is materially newer than the roster import, the
    optimizer may be recommending promotions from stale active/reserve
    assignments.

    Returns:
        {
            'cards_updated_at': ISO timestamp (UTC) or None,
            'roster_updated_at': ISO timestamp (UTC) or None,
            'staleness_hours': float (positive = roster older than cards),
            'is_stale': bool,
            'stale_threshold_hours': float,
            'stale_file_types': list[str] of missing/old team CSV types,
        }
    """
    # File types that ONLY come from Toronto-team exports. If any of these
    # is older than the market import, the roster snapshot is stale.
    team_roster_types = (
        "roster_batting",
        "roster_pitching",
        "lineup_overview",
        "lineup_vs_rhp",
        "lineup_vs_lhp",
        "team_pitching",
    )

    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True
    try:
        cards_at = conn.execute(
            "SELECT MAX(ingested_at) FROM ingestion_log WHERE file_type = 'market'"
        ).fetchone()[0]

        placeholders = ",".join(["?"] * len(team_roster_types))
        roster_at = conn.execute(
            f"SELECT MAX(ingested_at) FROM ingestion_log WHERE file_type IN ({placeholders})",
            team_roster_types,
        ).fetchone()[0]

        # Find which team file types are older than the cards import (or missing).
        stale_types: list[str] = []
        if cards_at:
            rows = conn.execute(
                f"""
                SELECT file_type, MAX(ingested_at) AS last_ingested
                FROM ingestion_log
                WHERE file_type IN ({placeholders})
                GROUP BY file_type
                """,
                team_roster_types,
            ).fetchall()
            seen = {r["file_type"]: r["last_ingested"] for r in rows}
            for ft in team_roster_types:
                last = seen.get(ft)
                if last is None or last < cards_at:
                    stale_types.append(ft)

        staleness_hours = None
        if cards_at and roster_at:
            diff = conn.execute(
                "SELECT (julianday(?) - julianday(?)) * 24.0",
                (cards_at, roster_at),
            ).fetchone()[0]
            staleness_hours = float(diff) if diff is not None else None

        threshold = 1.0
        is_stale = bool(
            (staleness_hours is not None and staleness_hours > threshold)
            or (cards_at and not roster_at)
        )

        return {
            "cards_updated_at": cards_at,
            "roster_updated_at": roster_at,
            "staleness_hours": staleness_hours,
            "is_stale": is_stale,
            "stale_threshold_hours": threshold,
            "stale_file_types": stale_types,
        }
    finally:
        if close_after:
            conn.close()


def migrate_add_league_columns(cursor: sqlite3.Cursor) -> None:
    """Safely add league_id column to existing tables if not already present.

    Covers the basic stats tables (batting_stats, pitching_stats, roster)
    which were originally partitioned in Stage 1, and the advanced stats
    tables (batting_stats_adv, pitching_stats_adv) which had the same
    bug as Stage 1: league-wide CSVs and team-scoped CSVs both wrote
    into an unpartitioned pool, so each team file would wipe the
    preceding league-wide insert. Added 2026-04-15.
    """
    for table in (
        "batting_stats",
        "pitching_stats",
        "roster",
        "batting_stats_adv",
        "pitching_stats_adv",
    ):
        try:
            cols = [row[1] for row in cursor.execute(f"PRAGMA table_info({table})").fetchall()]
            if "league_id" not in cols:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN league_id TEXT")
        except Exception:
            pass  # Table may not exist yet on fresh install


def _stage1c_cleanup(cursor: sqlite3.Cursor) -> None:
    """One-shot cleanup for the pre-Stage-1 stats pollution.

    Before the Stage 1 fix, both league-wide stats files and team-scoped
    stats files wrote rows with ``league_id IS NULL``. This helper:

    1. Deletes any ``league_id IS NULL`` rows in ``batting_stats`` /
       ``pitching_stats`` whose ``card_id`` does not match an owned card.
       Team-file ingestion pins every row to an owned card_id via
       ``_match_card_id(prefer_owned=True)``, so any NULL-league row that
       references a non-owned card (or card_id IS NULL) must have come
       from the old league-file code path.
    2. Installs a partial unique index on ``(card_id, snapshot_date) WHERE
       league_id IS NULL`` so future team-file upserts are idempotent.

    Both steps are idempotent and cheap. Runs on every ``init_db()`` so
    stale rows can't come back after a corrupted re-import.
    """
    for table in ("batting_stats", "pitching_stats"):
        try:
            # 1a. Delete NULL-league rows whose card_id is not owned.
            # Team-file ingestion always pins to an owned card_id, so any
            # NULL-league row referencing a non-owned (or NULL) card_id
            # must have come from the pre-Stage-1 league-file code path.
            # Note: ``owned`` can be 1 or 2 (duplicate copies both count),
            # so match ``owned >= 1``.
            cursor.execute(
                f"""
                DELETE FROM {table}
                WHERE league_id IS NULL
                  AND (
                      card_id IS NULL
                      OR card_id NOT IN (SELECT card_id FROM cards WHERE owned >= 1)
                  )
                """
            )
        except Exception:
            pass  # Fresh install / tables not yet populated

        try:
            # 1b. Dedupe (card_id, snapshot_date) groups inside the
            # NULL-league partition by keeping only the highest id per group.
            # Before Stage 1, league-file imports could write multiple rows
            # for the same owned card_id — one per other team that held a
            # card with the user's player name. After ``_match_card_id``
            # became owned-greedy in a prior release, all those rows ended
            # up pointing to the user's own card_id, so the owned-filter in
            # step 1a can't distinguish them. Collapse to one row per group
            # so the partial unique index can be installed. The user's next
            # team-file import will refresh the kept row with correct stats.
            cursor.execute(
                f"""
                DELETE FROM {table}
                WHERE league_id IS NULL
                  AND id NOT IN (
                      SELECT MAX(id)
                      FROM {table}
                      WHERE league_id IS NULL AND card_id IS NOT NULL
                      GROUP BY card_id, snapshot_date
                  )
                  AND card_id IS NOT NULL
                """
            )
        except Exception:
            pass

    # 2. Partial unique index: only one (card_id, snapshot_date) row per
    #    card per day in the team-scoped partition. Snapshot_date values
    #    for team-file rows are always plain date strings ('YYYY-MM-DD'),
    #    so a direct column index is sufficient and portable.
    for table, idx_name in (
        ("batting_stats", "idx_bat_stats_team_card_date"),
        ("pitching_stats", "idx_pit_stats_team_card_date"),
    ):
        try:
            cursor.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {idx_name}
                ON {table}(card_id, snapshot_date)
                WHERE league_id IS NULL AND card_id IS NOT NULL
                """
            )
        except Exception:
            pass  # Index may already exist with a conflicting signature


def sync_roster_meta_from_cards(conn: sqlite3.Connection | None = None) -> dict:
    """Re-align roster.meta_score with the live cards.meta_score_batting /
    meta_score_pitching values.

    Background: roster.meta_score is written at import time from whatever
    weights were active then. When the user later changes meta weights or we
    deploy a new formula version, cards.meta_score_batting gets recalc'd but
    the roster table stays frozen. This produced the visible Bichette
    "Roster Optimizer says 622, Card Detail says 666" gap that the product
    review flagged: same card, two different numbers because the roster row
    was stale by ~44 meta points across 84% of batters.

    This helper is cheap (two indexed UPDATEs joined on card_id) and idempotent.
    Call it once per Streamlit session at the top of any page that displays
    roster meta values; we gate via session_state in the page.

    Returns: {batters_synced, pitchers_synced, drift_before}
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        cur = conn.cursor()

        # How many rows would actually move? Used for the optional UI notice.
        try:
            drift_before = cur.execute("""
                SELECT COUNT(*)
                FROM roster r JOIN cards c ON r.card_id = c.card_id
                WHERE r.card_id IS NOT NULL
                  AND (
                    (r.position NOT IN ('SP','RP','CL') AND c.meta_score_batting IS NOT NULL
                       AND ABS(COALESCE(r.meta_score, 0) - c.meta_score_batting) > 0.5)
                    OR
                    (r.position IN ('SP','RP','CL') AND c.meta_score_pitching IS NOT NULL
                       AND ABS(COALESCE(r.meta_score, 0) - c.meta_score_pitching) > 0.5)
                  )
            """).fetchone()[0]
        except Exception:
            drift_before = 0

        bat = 0
        pit = 0
        try:
            cur.execute("""
                UPDATE roster
                SET meta_score = (
                    SELECT c.meta_score_batting FROM cards c
                    WHERE c.card_id = roster.card_id
                      AND c.meta_score_batting IS NOT NULL
                    LIMIT 1
                )
                WHERE card_id IS NOT NULL
                  AND position NOT IN ('SP','RP','CL')
                  AND EXISTS (
                      SELECT 1 FROM cards c
                      WHERE c.card_id = roster.card_id
                        AND c.meta_score_batting IS NOT NULL
                  )
            """)
            bat = cur.rowcount
        except Exception:
            pass

        try:
            cur.execute("""
                UPDATE roster
                SET meta_score = (
                    SELECT c.meta_score_pitching FROM cards c
                    WHERE c.card_id = roster.card_id
                      AND c.meta_score_pitching IS NOT NULL
                    LIMIT 1
                )
                WHERE card_id IS NOT NULL
                  AND position IN ('SP','RP','CL')
                  AND EXISTS (
                      SELECT 1 FROM cards c
                      WHERE c.card_id = roster.card_id
                        AND c.meta_score_pitching IS NOT NULL
                  )
            """)
            pit = cur.rowcount
        except Exception:
            pass

        conn.commit()
        return {
            "batters_synced": bat,
            "pitchers_synced": pit,
            "drift_before": drift_before,
        }
    finally:
        if own_conn:
            conn.close()


_INIT_DB_DONE = False  # process-level guard


def init_db() -> None:
    """Create all tables if they do not exist.

    Guarded with a module-level flag so it only runs once per Python
    process — Streamlit re-runs the script on every page load, and the
    CREATE TABLE IF NOT EXISTS scripts would otherwise compete with the
    background worker for a write lock on every rerun.
    """
    global _INIT_DB_DONE  # must appear before any assignment below
    if _INIT_DB_DONE:
        return

    db_path = get_db_path()
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Open with WAL + busy_timeout so we don't collide with the background
    # worker or other Streamlit sessions during the initial schema run.
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS cards (
        card_id INTEGER PRIMARY KEY,
        card_title TEXT NOT NULL,
        first_name TEXT,
        last_name TEXT,
        nickname TEXT,
        position INTEGER,
        position_name TEXT,
        pitcher_role INTEGER,
        pitcher_role_name TEXT,
        bats TEXT,
        throws TEXT,
        age INTEGER,
        team TEXT,
        franchise TEXT,
        card_type INTEGER,
        card_sub_type TEXT,
        card_badge TEXT,
        card_series TEXT,
        card_value INTEGER,
        year INTEGER,
        peak TEXT,
        tier INTEGER,
        tier_name TEXT,
        nation TEXT,
        contact INTEGER, gap_power INTEGER, power INTEGER, eye INTEGER, avoid_ks INTEGER, babip INTEGER,
        contact_vl INTEGER, gap_vl INTEGER, power_vl INTEGER, eye_vl INTEGER, avoid_ks_vl INTEGER, babip_vl INTEGER,
        contact_vr INTEGER, gap_vr INTEGER, power_vr INTEGER, eye_vr INTEGER, avoid_ks_vr INTEGER, babip_vr INTEGER,
        speed INTEGER, steal_rate INTEGER, stealing INTEGER, baserunning INTEGER, sac_bunt INTEGER, bunt_for_hit INTEGER,
        stuff INTEGER, movement INTEGER, control INTEGER, p_hr INTEGER, p_babip INTEGER,
        stuff_vl INTEGER, movement_vl INTEGER, control_vl INTEGER, p_hr_vl INTEGER, p_babip_vl INTEGER,
        stuff_vr INTEGER, movement_vr INTEGER, control_vr INTEGER, p_hr_vr INTEGER, p_babip_vr INTEGER,
        stamina INTEGER, hold INTEGER, velocity TEXT,
        infield_range INTEGER, infield_error INTEGER, infield_arm INTEGER, dp INTEGER,
        catcher_ability INTEGER, catcher_frame INTEGER, catcher_arm INTEGER,
        of_range INTEGER, of_error INTEGER, of_arm INTEGER,
        pos_rating_p INTEGER, pos_rating_c INTEGER, pos_rating_1b INTEGER,
        pos_rating_2b INTEGER, pos_rating_3b INTEGER, pos_rating_ss INTEGER,
        pos_rating_lf INTEGER, pos_rating_cf INTEGER, pos_rating_rf INTEGER,
        mission_value INTEGER, card_limit INTEGER, bref_id TEXT, packs TEXT,
        owned INTEGER DEFAULT 0,
        meta_score_batting REAL,
        meta_score_pitching REAL,
        buy_order_high INTEGER,
        sell_order_low INTEGER,
        last_10_price INTEGER,
        last_10_variance INTEGER,
        first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS price_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        snapshot_date TEXT NOT NULL,
        buy_order_high INTEGER,
        sell_order_low INTEGER,
        last_10_price INTEGER,
        last_10_variance INTEGER,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_price_card_date ON price_snapshots(card_id, snapshot_date);

    CREATE TABLE IF NOT EXISTS my_collection (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER,
        player_name TEXT NOT NULL,
        position TEXT,
        ovr INTEGER,
        status TEXT,
        contact INTEGER, gap_power INTEGER, avoid_ks INTEGER, eye INTEGER, power INTEGER,
        babip INTEGER, defense_score REAL,
        stuff INTEGER, movement INTEGER, ctrl INTEGER, p_hr INTEGER,
        meta_score REAL,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );

    CREATE TABLE IF NOT EXISTS roster (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT NOT NULL,
        position TEXT,
        lineup_role TEXT,
        ovr INTEGER,
        meta_score REAL,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rec_type TEXT NOT NULL,
        card_id INTEGER,
        card_title TEXT,
        position TEXT,
        reason TEXT,
        priority INTEGER,
        estimated_price INTEGER,
        meta_score REAL,
        value_ratio REAL,
        roster_impact TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        dismissed INTEGER DEFAULT 0,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );

    CREATE TABLE IF NOT EXISTS ingestion_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_type TEXT NOT NULL,
        file_name TEXT NOT NULL,
        row_count INTEGER,
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Singleton row tracks background-worker state so every Streamlit
    -- page can render a "live" badge + invalidate AI caches via data_version.
    -- Written by app/core/background_worker.py, read by app/utils/live_status.py.
    CREATE TABLE IF NOT EXISTS worker_status (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        data_version INTEGER DEFAULT 0,
        last_refresh_at TEXT,
        last_refresh_duration_s REAL,
        last_refresh_files INTEGER DEFAULT 0,
        last_refresh_rows INTEGER DEFAULT 0,
        last_box_scores INTEGER DEFAULT 0,
        last_game_logs INTEGER DEFAULT 0,
        last_message TEXT,
        status TEXT DEFAULT 'idle'   -- idle | scanning | ingesting | recalc | error
    );
    INSERT OR IGNORE INTO worker_status (id) VALUES (1);

    -- Recommendation audit log. Every suggestion the engine emits (AI Optimize
    -- All, Manager's Eye, meta-based lineup advice) writes a row so we can
    -- later detect whether Cameron actioned it and measure how the player
    -- performed post-action. Outcome rows populate asynchronously after the
    -- next roster refresh + N days of stats accrual.
    CREATE TABLE IF NOT EXISTS recommendation_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        source TEXT NOT NULL,           -- ai_optimize_all, manager_eye, meta_based, manual
        rec_type TEXT NOT NULL,         -- buy, promote, platoon, bench, demote, sell, hook
        pos TEXT,                       -- position slot
        player_name TEXT,               -- target player (who to acquire/promote/bench)
        player_card_id INTEGER,
        from_player TEXT,               -- who they replace (for promotes/buys)
        from_card_id INTEGER,
        expected_delta REAL,            -- meta points expected to gain
        cost_pp INTEGER,                -- PP cost if a buy
        reasoning TEXT,                 -- short text rationale from engine
        league_id TEXT,
        data_version INTEGER,           -- snapshot of data_version when emitted
        -- Outcome tracking (populated later):
        action_detected_at TEXT,        -- when we noticed the user followed the rec
        action_type TEXT,               -- followed, partial, ignored (NULL until detected)
        verdict TEXT,                   -- positive, neutral, negative, pending
        verdict_measured_at TEXT,
        war_before REAL,                -- player's WAR/600 before the action
        war_after REAL,                 -- WAR/600 since the action
        meta_before REAL,
        meta_after REAL,
        days_observed INTEGER           -- how long we've watched outcome
    );
    CREATE INDEX IF NOT EXISTS idx_reclog_player ON recommendation_log(player_name);
    CREATE INDEX IF NOT EXISTS idx_reclog_status ON recommendation_log(action_type, verdict);
    CREATE INDEX IF NOT EXISTS idx_reclog_created ON recommendation_log(created_at DESC);

    -- Tracks which card_ids are 'new' (first seen in a given refresh) so the
    -- dashboard can flag new arrivals and auto-queue a deployment rec for
    -- the LLM council to review. Written by background_worker._detect_new_cards.
    CREATE TABLE IF NOT EXISTS card_intake (
        card_id INTEGER PRIMARY KEY,
        first_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
        acknowledged_at TEXT,
        rec_id INTEGER                -- optional FK to recommendation_log
    );

    CREATE TABLE IF NOT EXISTS batting_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT NOT NULL,
        position TEXT,
        bats TEXT,
        throws TEXT,
        games INTEGER DEFAULT 0,
        pa INTEGER DEFAULT 0,
        ab INTEGER DEFAULT 0,
        hits INTEGER DEFAULT 0,
        doubles INTEGER DEFAULT 0,
        triples INTEGER DEFAULT 0,
        hr INTEGER DEFAULT 0,
        rbi INTEGER DEFAULT 0,
        runs INTEGER DEFAULT 0,
        bb INTEGER DEFAULT 0,
        ibb INTEGER DEFAULT 0,
        hbp INTEGER DEFAULT 0,
        k INTEGER DEFAULT 0,
        gidp INTEGER DEFAULT 0,
        avg REAL DEFAULT 0,
        obp REAL DEFAULT 0,
        slg REAL DEFAULT 0,
        iso REAL DEFAULT 0,
        ops REAL DEFAULT 0,
        ops_plus INTEGER DEFAULT 0,
        babip REAL DEFAULT 0,
        war REAL DEFAULT 0,
        sb INTEGER DEFAULT 0,
        cs INTEGER DEFAULT 0,
        card_id INTEGER,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE INDEX IF NOT EXISTS idx_batting_stats_player ON batting_stats(player_name);
    CREATE INDEX IF NOT EXISTS idx_batting_stats_date ON batting_stats(snapshot_date);

    CREATE TABLE IF NOT EXISTS pitching_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT NOT NULL,
        position TEXT,
        bats TEXT,
        throws TEXT,
        games INTEGER DEFAULT 0,
        gs INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        saves INTEGER DEFAULT 0,
        holds INTEGER DEFAULT 0,
        ip REAL DEFAULT 0,
        hits_allowed INTEGER DEFAULT 0,
        hr_allowed INTEGER DEFAULT 0,
        runs_allowed INTEGER DEFAULT 0,
        er INTEGER DEFAULT 0,
        bb INTEGER DEFAULT 0,
        k INTEGER DEFAULT 0,
        hbp INTEGER DEFAULT 0,
        era REAL DEFAULT 0,
        avg_against REAL DEFAULT 0,
        babip REAL DEFAULT 0,
        whip REAL DEFAULT 0,
        hr_per_9 REAL DEFAULT 0,
        bb_per_9 REAL DEFAULT 0,
        k_per_9 REAL DEFAULT 0,
        k_per_bb REAL DEFAULT 0,
        era_plus INTEGER DEFAULT 0,
        fip REAL DEFAULT 0,
        war REAL DEFAULT 0,
        card_id INTEGER,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE INDEX IF NOT EXISTS idx_pitching_stats_player ON pitching_stats(player_name);
    CREATE INDEX IF NOT EXISTS idx_pitching_stats_date ON pitching_stats(snapshot_date);

    CREATE TABLE IF NOT EXISTS ai_insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        insight_type TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_ai_insights_type ON ai_insights(insight_type, created_at);

    -- Manager's Eye AI cache. Keyed on a hash of the active 26-man roster so
    -- the (expensive) Gemini call only fires when the lineup actually changes.
    -- payload is a JSON blob of the full unified analysis dict.
    CREATE TABLE IF NOT EXISTS manager_eye_cache (
        roster_hash TEXT PRIMARY KEY,
        payload     TEXT NOT NULL,
        model       TEXT,
        tokens      INTEGER,
        created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS price_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        alert_type TEXT NOT NULL DEFAULT 'below',
        target_price INTEGER NOT NULL,
        active INTEGER DEFAULT 1,
        triggered INTEGER DEFAULT 0,
        triggered_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    """)

    # Migrate old price_alerts table if it's missing new columns
    try:
        cols = [row[1] for row in cursor.execute("PRAGMA table_info(price_alerts)").fetchall()]
        if 'active' not in cols:
            cursor.execute("ALTER TABLE price_alerts ADD COLUMN active INTEGER DEFAULT 1")
        if 'alert_type' not in cols:
            cursor.execute("ALTER TABLE price_alerts ADD COLUMN alert_type TEXT NOT NULL DEFAULT 'below'")
        if 'triggered_at' not in cols:
            cursor.execute("ALTER TABLE price_alerts ADD COLUMN triggered_at TIMESTAMP")
    except Exception:
        pass  # Table may not exist yet on fresh install

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_alerts_active ON price_alerts(active, triggered)")

    # Recommendation reinforcement loop (2026-04-25). The optimizer logs each
    # rec with the *factor decomposition* that drove it — contact +X, power
    # +Y, perf overlay +Z — so we can later regress (actual − projected)
    # outcomes on those factors and learn which signals over- or under-
    # predicted. ``side`` distinguishes batting vs pitching so the backtest
    # picks the right game-log table; ``projected_meta`` / ``projected_war``
    # are the engine's projection at rec time, used as the regression
    # baseline. ``factor_snapshot_json`` is the canonical contract written
    # by recommendation_tracker.extract_factor_snapshot.
    try:
        cols = [row[1] for row in cursor.execute("PRAGMA table_info(recommendation_log)").fetchall()]
        if 'factor_snapshot_json' not in cols:
            cursor.execute("ALTER TABLE recommendation_log ADD COLUMN factor_snapshot_json TEXT")
        if 'projected_meta' not in cols:
            cursor.execute("ALTER TABLE recommendation_log ADD COLUMN projected_meta REAL")
        if 'projected_war' not in cols:
            cursor.execute("ALTER TABLE recommendation_log ADD COLUMN projected_war REAL")
        if 'side' not in cols:
            cursor.execute("ALTER TABLE recommendation_log ADD COLUMN side TEXT")
    except Exception:
        pass  # Table may not exist yet on fresh install

    # Residual learner output (Stream C). One row per learner run; the
    # latest row per (league_id, side) is what the UI renders. Stored as
    # JSON blobs so the schema doesn't need to evolve as we add factors
    # or interaction-mining heuristics. ``shadow_mode`` is True when the
    # multipliers are reported but NOT fed back into meta_calibration —
    # flips to False once the user opts in or sample size clears the gate.
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS rec_residual_calibration (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        league_id TEXT,
        side TEXT NOT NULL,
        n_observations INTEGER NOT NULL,
        factor_calibration_json TEXT,
        interaction_warnings_json TEXT,
        diagnostics_json TEXT,
        shadow_mode INTEGER DEFAULT 1
    );
    CREATE INDEX IF NOT EXISTS idx_rec_residual_recent
        ON rec_residual_calibration(side, league_id, created_at DESC);
    """)

    # Card-lookup indexes. Ingestion does name-based LIKE / exact matches
    # thousands of times per CSV; without these the inner loop is an O(N*M)
    # full table scan that dominates import wall time on a 2500-card market.
    for idx_sql in (
        "CREATE INDEX IF NOT EXISTS idx_cards_name ON cards(first_name, last_name)",
        "CREATE INDEX IF NOT EXISTS idx_cards_last_first ON cards(last_name, first_name)",
        "CREATE INDEX IF NOT EXISTS idx_cards_title ON cards(card_title)",
        "CREATE INDEX IF NOT EXISTS idx_cards_owned ON cards(owned)",
        "CREATE INDEX IF NOT EXISTS idx_roster_card_id ON roster(card_id)",
        "CREATE INDEX IF NOT EXISTS idx_roster_player ON roster(player_name)",
        # Dedup guard: one roster row per (card_id, day, role). The league
        # ratings CSVs list a card N times (once per owning team) — those
        # per-team rows live in `league_rosters` now, so `roster` doesn't
        # need to carry them. Scoped to non-null card_id so name-only legacy
        # rows (if any) don't block the index.
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_roster_dedup "
        "ON roster (card_id, DATE(snapshot_date), lineup_role) "
        "WHERE card_id IS NOT NULL",

        # Hot-query indexes added 2026-04-24 to speed up the Buy Recs /
        # Roster Optimizer / overlay query paths. Added after verifying
        # missing indexes left several common filters doing table scans.
        "CREATE INDEX IF NOT EXISTS idx_roster_role_date ON roster(lineup_role, snapshot_date)",
        "CREATE INDEX IF NOT EXISTS idx_bs_league_card_date ON batting_stats(league_id, card_id, snapshot_date)",
        "CREATE INDEX IF NOT EXISTS idx_ps_league_card_date ON pitching_stats(league_id, card_id, snapshot_date)",
        "CREATE INDEX IF NOT EXISTS idx_bat_adv_card_date ON batting_stats_adv(card_id, snapshot_date)",
        "CREATE INDEX IF NOT EXISTS idx_pit_adv_card_date ON pitching_stats_adv(card_id, snapshot_date)",
        "CREATE INDEX IF NOT EXISTS idx_recs_created ON recommendations(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_recs_card ON recommendations(card_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_ingest_log_date ON ingestion_log(ingested_at)",
        "CREATE INDEX IF NOT EXISTS idx_meta_hist_card_date ON meta_history(card_id, snapshot_date)",
        # Dedup guard for meta_history: re-running recalc on the same day
        # with the same weights should refresh the existing row, not append
        # a duplicate. Paired with INSERT OR REPLACE in snapshot_meta_scores.
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_meta_hist_dedup ON meta_history"
        "(card_id, league_id, weights_version, DATE(snapshot_date))",
    ):
        try:
            cursor.execute(idx_sql)
        except Exception:
            # A column referenced by an index may not exist yet on very old
            # DBs that haven't run the roster.card_id migration below. Those
            # ALTER TABLEs run later in init_db; a retry pass happens via
            # the re-entrant init_db call at app startup.
            pass

    # Migrate price_snapshots index to UNIQUE (needed for INSERT OR IGNORE dedup)
    try:
        idx_info = cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_price_card_date'"
        ).fetchone()
        if idx_info and 'UNIQUE' not in (idx_info[0] or '').upper():
            cursor.execute("DROP INDEX IF EXISTS idx_price_card_date")
            cursor.execute(
                "CREATE UNIQUE INDEX idx_price_card_date ON price_snapshots(card_id, snapshot_date)"
            )
    except Exception:
        pass

    # --- Advanced stats tables (stats_2 format) ---
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS batting_stats_adv (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT NOT NULL,
        position TEXT,
        bats TEXT,
        throws TEXT,
        games INTEGER DEFAULT 0,
        pa INTEGER DEFAULT 0,
        bb INTEGER DEFAULT 0,
        bb_pct REAL DEFAULT 0,
        sh INTEGER DEFAULT 0,
        sf INTEGER DEFAULT 0,
        ci INTEGER DEFAULT 0,
        k INTEGER DEFAULT 0,
        k_pct REAL DEFAULT 0,
        gidp INTEGER DEFAULT 0,
        ebh INTEGER DEFAULT 0,
        tb INTEGER DEFAULT 0,
        rc REAL DEFAULT 0,
        rc27 REAL DEFAULT 0,
        iso REAL DEFAULT 0,
        woba REAL DEFAULT 0,
        wpa REAL DEFAULT 0,
        pi_pa REAL DEFAULT 0,
        card_id INTEGER,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE INDEX IF NOT EXISTS idx_bat_adv_player ON batting_stats_adv(player_name);
    CREATE INDEX IF NOT EXISTS idx_bat_adv_date ON batting_stats_adv(snapshot_date);

    CREATE TABLE IF NOT EXISTS pitching_stats_adv (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT NOT NULL,
        position TEXT,
        bats TEXT,
        throws TEXT,
        games INTEGER DEFAULT 0,
        win_pct REAL DEFAULT 0,
        sv_pct REAL DEFAULT 0,
        bs INTEGER DEFAULT 0,
        sd INTEGER DEFAULT 0,
        md INTEGER DEFAULT 0,
        ip REAL DEFAULT 0,
        bf INTEGER DEFAULT 0,
        dp INTEGER DEFAULT 0,
        ra INTEGER DEFAULT 0,
        gf INTEGER DEFAULT 0,
        ir INTEGER DEFAULT 0,
        irs_pct REAL DEFAULT 0,
        pli REAL DEFAULT 0,
        qs INTEGER DEFAULT 0,
        qs_pct REAL DEFAULT 0,
        cg INTEGER DEFAULT 0,
        cg_pct REAL DEFAULT 0,
        sho INTEGER DEFAULT 0,
        ppg INTEGER DEFAULT 0,
        rsg REAL DEFAULT 0,
        go_pct REAL DEFAULT 0,
        siera REAL DEFAULT 0,
        sb INTEGER DEFAULT 0,
        cs INTEGER DEFAULT 0,
        wpa REAL DEFAULT 0,
        card_id INTEGER,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE INDEX IF NOT EXISTS idx_pit_adv_player ON pitching_stats_adv(player_name);
    CREATE INDEX IF NOT EXISTS idx_pit_adv_date ON pitching_stats_adv(snapshot_date);

    CREATE TABLE IF NOT EXISTS team_lineup (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lineup_type TEXT NOT NULL,
        position TEXT,
        player_name TEXT NOT NULL,
        card_title TEXT,
        card_id INTEGER,
        ovr INTEGER,
        bats TEXT,
        throws TEXT,
        age INTEGER,
        slot_order INTEGER,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE INDEX IF NOT EXISTS idx_lineup_type ON team_lineup(lineup_type);

    CREATE TABLE IF NOT EXISTS fielding_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT NOT NULL,
        position TEXT,
        bats TEXT,
        throws TEXT,
        games INTEGER DEFAULT 0,
        gs INTEGER DEFAULT 0,
        tc INTEGER DEFAULT 0,
        assists INTEGER DEFAULT 0,
        putouts INTEGER DEFAULT 0,
        errors INTEGER DEFAULT 0,
        dp INTEGER DEFAULT 0,
        pct REAL DEFAULT 0,
        rng REAL DEFAULT 0,
        zr REAL DEFAULT 0,
        eff REAL DEFAULT 0,
        sba INTEGER DEFAULT 0,
        rto INTEGER DEFAULT 0,
        rto_pct REAL DEFAULT 0,
        ip REAL DEFAULT 0,
        pb INTEGER DEFAULT 0,
        cera REAL DEFAULT 0,
        frm REAL DEFAULT 0,
        arm REAL DEFAULT 0,
        card_id INTEGER,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE INDEX IF NOT EXISTS idx_fielding_player ON fielding_stats(player_name);

    CREATE TABLE IF NOT EXISTS pitch_ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT NOT NULL,
        position TEXT,
        throws TEXT,
        age INTEGER,
        fb INTEGER DEFAULT 0,
        ch INTEGER DEFAULT 0,
        cb INTEGER DEFAULT 0,
        sl INTEGER DEFAULT 0,
        si INTEGER DEFAULT 0,
        sp INTEGER DEFAULT 0,
        ct INTEGER DEFAULT 0,
        fo INTEGER DEFAULT 0,
        cc INTEGER DEFAULT 0,
        sc INTEGER DEFAULT 0,
        kc INTEGER DEFAULT 0,
        kn INTEGER DEFAULT 0,
        pitch_count INTEGER DEFAULT 0,
        velocity TEXT,
        slot TEXT,
        stamina INTEGER DEFAULT 0,
        card_id INTEGER,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE INDEX IF NOT EXISTS idx_pitch_ratings_player ON pitch_ratings(player_name);

    -- League tracking tables
    CREATE TABLE IF NOT EXISTS leagues (
        league_id TEXT PRIMARY KEY,
        league_name TEXT,
        league_tier TEXT,
        start_date TEXT,
        end_date TEXT,
        final_record TEXT,
        team_name TEXT DEFAULT 'Toronto Dark Knights',
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS meta_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER,
        player_name TEXT NOT NULL,
        position TEXT,
        meta_score REAL,
        meta_vs_rhp REAL,
        meta_vs_lhp REAL,
        league_id TEXT,
        weights_version TEXT,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (league_id) REFERENCES leagues(league_id)
    );
    CREATE INDEX IF NOT EXISTS idx_meta_history_player ON meta_history(player_name, snapshot_date);
    CREATE INDEX IF NOT EXISTS idx_meta_history_league ON meta_history(league_id, snapshot_date);

    CREATE TABLE IF NOT EXISTS league_rosters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        league_id TEXT,
        team_name TEXT,
        player_name TEXT,
        card_id INTEGER,
        position TEXT,
        ovr INTEGER,
        meta_score REAL,
        contact INTEGER, gap_power INTEGER, power INTEGER, eye INTEGER,
        stuff INTEGER, movement INTEGER, control INTEGER, p_hr INTEGER,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (league_id) REFERENCES leagues(league_id)
    );
    CREATE INDEX IF NOT EXISTS idx_league_rosters_team ON league_rosters(league_id, team_name);

    -- Team name normalization.
    -- Sources disagree: OOTP CSV league exports use short names ("Sassy",
    -- "Lakeville"), HTML box scores use full franchise names ("Sassy Kitties",
    -- "Lakeville Tourists"). This table is the single join key across the two.
    CREATE TABLE IF NOT EXISTS team_aliases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        league_id TEXT NOT NULL,
        short_name TEXT NOT NULL,
        full_name TEXT,
        franchise_code TEXT,
        source TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(league_id, short_name)
    );
    CREATE INDEX IF NOT EXISTS idx_team_aliases_full ON team_aliases(league_id, full_name);

    -- ─── Derived analytics tables ─────────────────────────────────────
    -- These are all computed from the raw ingested tables (game_log_at_bats,
    -- game_batting, batting_stats, price_snapshots, etc). They're rebuilt
    -- idempotently by app/core/derived_stats.build_all() on every refresh.

    -- Observed L/R splits per batter card from real at-bats. Distinct from
    -- the ratings-based `roster.meta_vs_rhp` / `meta_vs_lhp` which come
    -- from OOTP card ratings — this table tells you how the batter ACTUALLY
    -- performs vs each pitcher hand in games we've ingested.
    CREATE TABLE IF NOT EXISTS batter_vs_pitcher_hand (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        pitcher_hand TEXT NOT NULL,                 -- 'L' or 'R' (from LHP/RHP)
        league_id TEXT,
        pa INTEGER DEFAULT 0,
        ab INTEGER DEFAULT 0,
        h INTEGER DEFAULT 0,
        doubles INTEGER DEFAULT 0,
        triples INTEGER DEFAULT 0,
        hr INTEGER DEFAULT 0,
        bb INTEGER DEFAULT 0,
        k INTEGER DEFAULT 0,
        hbp INTEGER DEFAULT 0,
        avg REAL,
        obp REAL,
        slg REAL,
        ops REAL,
        iso REAL,
        k_rate REAL,
        bb_rate REAL,
        hr_rate REAL,
        games_sample INTEGER,                        -- # distinct games
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_bvph_unique
        ON batter_vs_pitcher_hand(card_id, pitcher_hand, COALESCE(league_id, ''));
    CREATE INDEX IF NOT EXISTS idx_bvph_card ON batter_vs_pitcher_hand(card_id);

    -- Rolling-window pitcher usage. Answers "who's rested tonight?" for
    -- both my own bullpen and the opponent's. Anchored to the LATEST game
    -- date in `games` (simulated season clock) — not real-world today.
    CREATE TABLE IF NOT EXISTS pitcher_fatigue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        league_id TEXT,
        last_appearance_date TEXT,
        days_rest INTEGER,
        appearances_last_3_days INTEGER DEFAULT 0,
        appearances_last_7_days INTEGER DEFAULT 0,
        pitches_last_3_days INTEGER DEFAULT 0,
        pitches_last_7_days INTEGER DEFAULT 0,
        last_appearance_pitches INTEGER,
        last_appearance_role TEXT,               -- SP, RP, CL, etc.
        total_appearances INTEGER,               -- season-to-date
        total_pitches INTEGER,
        role_profile TEXT,                       -- SP / RP / CL (most common)
        availability TEXT,                       -- fresh / available / tired / unavailable
        anchor_date TEXT,                        -- the game_date used as 'today'
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_pf_unique
        ON pitcher_fatigue(card_id, COALESCE(league_id, ''));
    CREATE INDEX IF NOT EXISTS idx_pf_avail ON pitcher_fatigue(availability);

    -- Observed clutch / situational profile per card. Built from the
    -- game_clutch_events event stream. Distinguishes hitters who actually
    -- drive in runners in 2-out RISP spots (clutch_rbi_rate) from hitters
    -- whose surface stats mask a weak clutch record, and relievers who
    -- actually strand inherited runners vs. those who just put up good
    -- ERAs on their own baserunners.
    CREATE TABLE IF NOT EXISTS clutch_profile_card (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        league_id TEXT,
        -- Batter clutch
        two_out_rbi INTEGER DEFAULT 0,
        lob_risp_2out INTEGER DEFAULT 0,
        sac_fly INTEGER DEFAULT 0,
        gidp INTEGER DEFAULT 0,
        -- Power/baserunning (clutch-table counts, not surface box scores)
        doubles INTEGER DEFAULT 0,
        triples INTEGER DEFAULT 0,
        hr_clutch INTEGER DEFAULT 0,
        sb INTEGER DEFAULT 0,
        cs INTEGER DEFAULT 0,
        -- Pitching clutch (relievers)
        inherited_runners INTEGER DEFAULT 0,
        inherited_scored INTEGER DEFAULT 0,
        -- Defense
        errors INTEGER DEFAULT 0,
        -- Derived rates
        clutch_rbi_rate REAL,                    -- 2OUT_RBI / (2OUT_RBI + LOB_RISP_2OUT)
        inherited_strand_rate REAL,              -- 1 - inh_scored/inh_runners
        net_steals INTEGER,                      -- sb - cs
        games_sample INTEGER,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_cpc_unique
        ON clutch_profile_card(card_id, COALESCE(league_id, ''));

    -- Per-card park-adjusted batting line. Most LB124 parks are ~1.00 so
    -- adjusted ≈ raw there, but this kicks in when the user plays a league
    -- with real park effects (e.g. a Coors-style home park). The weight is
    -- the card's games_played at each home park.
    CREATE TABLE IF NOT EXISTS park_adjusted_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        league_id TEXT,
        games_sample INTEGER,
        pa INTEGER,
        ab INTEGER,
        h INTEGER,
        hr INTEGER,
        -- Raw (observed from game_batting)
        raw_avg REAL, raw_obp REAL, raw_slg REAL, raw_ops REAL,
        -- Weighted park factors the card experienced
        pf_overall_weighted REAL,
        pf_hr_weighted REAL,
        pf_avg_weighted REAL,
        -- Adjusted = raw / pf_overall (OPS+)-style normalization
        adj_ops REAL,
        adj_avg REAL,
        adj_hr_rate REAL,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_pas_unique
        ON park_adjusted_stats(card_id, COALESCE(league_id, ''));

    -- Persisted regression signal per card. The existing Regression
    -- Candidates page computes BABIP-vs-contact-quality live; this table
    -- complements that with a ratings-vs-outcomes lens (OPS+ observed vs
    -- what we'd empirically expect for a card of this value, K% vs
    -- avoid_ks rating, etc). Stored so Buy/Sell pages can JOIN and flag
    -- "also running hot" without re-running the whole calc per view.
    CREATE TABLE IF NOT EXISTS regression_candidates_v2 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        league_id TEXT,
        pa INTEGER,
        -- Observed
        ops_plus_observed INTEGER,
        babip_observed REAL,
        k_pct_observed REAL,
        -- Expected (empirical card_value curve + card ratings)
        ops_plus_expected REAL,
        -- Deltas (positive = running hotter than expected → regress down)
        ops_plus_delta REAL,
        babip_delta REAL,                         -- vs card.babip rating mapped to .300 baseline
        k_pct_delta REAL,                         -- vs avoid_ks rating mapped to .22 baseline
        -- Composite
        regression_score REAL,                    -- standardized combined signal
        direction TEXT,                           -- 'regress_down' | 'regress_up' | 'sustainable'
        confidence REAL,                          -- 0-1 based on PA
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_rcv2_unique
        ON regression_candidates_v2(card_id, COALESCE(league_id, ''));
    CREATE INDEX IF NOT EXISTS idx_rcv2_direction ON regression_candidates_v2(direction);

    -- Per-team rolling form + bullpen availability. Lets the user look up
    -- tonight's opponent and see record trend + who's gassed before
    -- setting the lineup. "Who's hot this week" regardless of matchup.
    CREATE TABLE IF NOT EXISTS opponent_scouting (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        league_id TEXT NOT NULL,
        team_name TEXT NOT NULL,
        anchor_date TEXT,                        -- latest game date considered
        last_5_wins INTEGER,
        last_5_losses INTEGER,
        last_5_runs_for INTEGER,
        last_5_runs_against INTEGER,
        last_10_wins INTEGER,
        last_10_losses INTEGER,
        last_10_runs_for INTEGER,
        last_10_runs_against INTEGER,
        last_20_wins INTEGER,
        last_20_losses INTEGER,
        -- Bullpen snapshot (via team's pitchers in pitcher_fatigue)
        pitchers_fresh INTEGER,
        pitchers_available INTEGER,
        pitchers_tired INTEGER,
        pitchers_unavailable INTEGER,
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(league_id, team_name)
    );
    CREATE INDEX IF NOT EXISTS idx_os_team ON opponent_scouting(team_name);

    -- Precomputed price trends per card from `player_history`. Saves
    -- Flip Finder + Buy Recs from re-scanning snapshots on every render.
    -- Uses `last_10_price` as the primary series (most stable on thin markets).
    CREATE TABLE IF NOT EXISTS price_velocity (
        card_id INTEGER PRIMARY KEY,
        latest_date TEXT,
        latest_buy_high INTEGER,
        latest_sell_low INTEGER,
        latest_l10 INTEGER,
        avg_l10_3d REAL,
        avg_l10_7d REAL,
        slope_l10_per_day REAL,                  -- over the whole window
        pct_change_3d REAL,
        pct_change_7d REAL,
        volatility_7d REAL,                      -- stddev of l10 price
        snapshots_n INTEGER,
        window_days INTEGER,                     -- actual span of data (may be < 7)
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE INDEX IF NOT EXISTS idx_pv_slope ON price_velocity(slope_l10_per_day);

    -- Persisted output of card_aggregation.card_confidence(). That
    -- function runs on every page load; persisting it lets Buy/Sell/
    -- Roster pages join for a fast "confidence label" column without
    -- paying the aggregation cost per-render.
    CREATE TABLE IF NOT EXISTS meta_confidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        league_id TEXT,
        score INTEGER,
        label TEXT,                              -- high / medium / low / none
        role TEXT,                               -- batting / pitching
        drivers_json TEXT,
        sample_size INTEGER,                     -- PA for batters, IP*10 for pitchers
        n_instances INTEGER,                     -- # team copies seen
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_mc_unique
        ON meta_confidence(card_id, COALESCE(league_id, ''));
    CREATE INDEX IF NOT EXISTS idx_mc_label ON meta_confidence(label);

    -- Player card value + stats history (per-export snapshots for trending)
    CREATE TABLE IF NOT EXISTS player_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER,
        player_name TEXT NOT NULL,
        position TEXT,
        pitcher_role TEXT,
        league_id TEXT,
        -- card market data
        card_value INTEGER,
        buy_order_high INTEGER,
        sell_order_low INTEGER,
        last_10_price INTEGER,
        last_10_variance INTEGER,
        -- meta scores (at time of snapshot)
        meta_score REAL,
        meta_vs_rhp REAL,
        meta_vs_lhp REAL,
        -- key batting ratings
        contact INTEGER, gap_power INTEGER, power INTEGER, eye INTEGER,
        avoid_ks INTEGER, babip INTEGER, speed INTEGER, stealing INTEGER,
        -- key pitching ratings
        stuff INTEGER, movement INTEGER, control INTEGER, p_hr INTEGER,
        stamina INTEGER, hold INTEGER,
        -- in-game performance (cumulative to this point in season)
        games INTEGER,
        pa INTEGER,
        avg REAL, obp REAL, slg REAL, ops REAL, ops_plus INTEGER,
        hr INTEGER, rbi INTEGER, war REAL, sb INTEGER,
        -- pitching performance
        ip REAL, era REAL, whip REAL, k_per_9 REAL, bb_per_9 REAL,
        fip REAL, era_plus INTEGER, p_war REAL,
        -- context
        weights_version TEXT,
        export_number INTEGER,           -- which export within this league (1st, 2nd, 3rd...)
        games_into_season INTEGER,       -- how many team games played at this point
        snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (league_id) REFERENCES leagues(league_id)
    );
    CREATE INDEX IF NOT EXISTS idx_player_history_card ON player_history(card_id, league_id, snapshot_date);
    CREATE INDEX IF NOT EXISTS idx_player_history_player ON player_history(player_name, snapshot_date);
    CREATE INDEX IF NOT EXISTS idx_player_history_league ON player_history(league_id, export_number);

    -- Track each data export event for sequencing
    CREATE TABLE IF NOT EXISTS export_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        league_id TEXT,
        export_number INTEGER,
        games_played INTEGER,
        team_record TEXT,
        files_imported INTEGER,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (league_id) REFERENCES leagues(league_id)
    );

    -- League-wide TEAM-level stats (one row per team per snapshot).
    -- Used for framing: "your OPS is 9th of 30 in lb124, league avg .680".
    -- Combines batting + pitching + park factors into a single wide row
    -- because OOTP exports these in separate files but we always want them
    -- together when building team context.
    CREATE TABLE IF NOT EXISTS league_team_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        league_id TEXT,
        team_name TEXT NOT NULL,
        snapshot_date TEXT NOT NULL,
        -- Batting. Note: OOTP's league-wide team-batting CSV does NOT export
        -- AB or H columns — only 2B/3B/HR/R/BB/SO/SB + rate stats. We keep
        -- ab/hits columns for reconciliation-by-summing-players (the Toronto-
        -- specific roster CSV provides H totals) but they stay NULL when only
        -- league-wide data is ingested.
        games INTEGER, ab INTEGER, hits INTEGER,
        doubles INTEGER, triples INTEGER, hr INTEGER,
        runs INTEGER, bb INTEGER, so INTEGER, sb INTEGER,
        avg REAL, obp REAL, slg REAL, ops REAL,
        -- Pitching
        wins INTEGER, losses INTEGER, saves INTEGER, ip REAL,
        runs_allowed INTEGER, hr_allowed INTEGER, bb_allowed INTEGER,
        k INTEGER, whip REAL, batting_avg_against REAL, era REAL,
        fip_minus INTEGER,
        -- Park factors (centered at 1.000, >1 = hitter-friendly)
        park_name TEXT, pf_avg REAL, pf_avg_l REAL, pf_avg_r REAL,
        pf_hr REAL, pf_hr_l REAL, pf_hr_r REAL,
        pf_d REAL, pf_t REAL, pf_overall REAL,
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (league_id) REFERENCES leagues(league_id),
        UNIQUE(league_id, team_name, snapshot_date)
    );
    CREATE INDEX IF NOT EXISTS idx_league_team_stats_league
        ON league_team_stats(league_id, snapshot_date);
    CREATE INDEX IF NOT EXISTS idx_league_team_stats_team
        ON league_team_stats(team_name, league_id);

    -- ══════════════════════════════════════════════════════════════════
    -- GAMES / BOX SCORES — parsed from OOTP's news/html/box_scores/*.html
    --
    -- Each PT league sim generates a per-game HTML box score. Ingesting
    -- these gives us game-by-game granularity (vs the cumulative stat
    -- snapshots we get from CSV). Unlocks:
    --   * rolling windows (last 7 games OPS) that aren't smeared by N-day
    --     snapshot cadence
    --   * opposing-pitcher context (went 3-for-4 against X, who has ERA+ Y)
    --   * matchup history between teams and between players
    --   * pitcher usage (W/L/SV/BS attribution by card_id)
    -- ══════════════════════════════════════════════════════════════════
    CREATE TABLE IF NOT EXISTS games (
        game_id INTEGER PRIMARY KEY,            -- OOTP's internal game ID
        league_id TEXT,
        game_date TEXT,                         -- yyyy-mm-dd parsed from HTML header
        home_team TEXT,
        away_team TEXT,
        home_score INTEGER,
        away_score INTEGER,
        winner_team TEXT,                       -- 'home' or 'away' (or NULL if tied)
        toronto_role TEXT,                      -- 'home' | 'away' | NULL (if neither)
        source_file TEXT,                       -- path to the parsed HTML
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
    CREATE INDEX IF NOT EXISTS idx_games_league ON games(league_id, game_date);
    CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team, away_team);

    -- One row per (game, batter). Stats are for this single game only.
    -- Season-to-date running stats (season_avg etc.) are captured from
    -- the box HTML for convenience — they're the AVG/HR/RBI shown at
    -- the right of each line.
    CREATE TABLE IF NOT EXISTS game_batting (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER NOT NULL,
        player_name TEXT NOT NULL,
        card_id INTEGER,                        -- resolved by _match_card_id at ingest
        team_name TEXT,                         -- team the player was on in this game
        position TEXT,                          -- C/1B/…/DH/PR/PH
        batting_order INTEGER,                  -- 1..9, or NULL for subs
        ab INTEGER DEFAULT 0,
        r INTEGER DEFAULT 0,
        h INTEGER DEFAULT 0,
        rbi INTEGER DEFAULT 0,
        bb INTEGER DEFAULT 0,
        k INTEGER DEFAULT 0,
        lob INTEGER DEFAULT 0,
        -- Season-to-date at time of game (running)
        season_avg REAL,
        season_hr INTEGER,
        season_rbi INTEGER,
        FOREIGN KEY (game_id) REFERENCES games(game_id),
        UNIQUE(game_id, player_name, team_name)
    );
    CREATE INDEX IF NOT EXISTS idx_game_bat_player ON game_batting(player_name);
    CREATE INDEX IF NOT EXISTS idx_game_bat_card ON game_batting(card_id);
    CREATE INDEX IF NOT EXISTS idx_game_bat_team ON game_batting(team_name);

    -- One row per (game, pitcher). role_flag captures W/L/SV/BS/HLD for
    -- pitcher attribution (used by usage-pattern analysis).
    CREATE TABLE IF NOT EXISTS game_pitching (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER NOT NULL,
        player_name TEXT NOT NULL,
        card_id INTEGER,
        team_name TEXT,
        appearance_order INTEGER,               -- 1 = starter, 2 = first reliever, etc.
        role_flag TEXT,                         -- 'SP' | 'W' | 'L' | 'SV' | 'BS' | 'HLD' | ''
        role_number INTEGER,                    -- parenthesized count if present (e.g. SV (19))
        ip REAL DEFAULT 0,                      -- OOTP notation: 6.1 = 6 IP + 1 out
        h INTEGER DEFAULT 0,
        r INTEGER DEFAULT 0,
        er INTEGER DEFAULT 0,
        bb INTEGER DEFAULT 0,
        k INTEGER DEFAULT 0,
        hr INTEGER DEFAULT 0,
        pitches INTEGER DEFAULT 0,
        strikes INTEGER DEFAULT 0,
        batters_faced INTEGER,
        ground_outs INTEGER,
        fly_outs INTEGER,
        game_score INTEGER,
        season_era REAL,                        -- ERA to-date at time of game
        FOREIGN KEY (game_id) REFERENCES games(game_id),
        UNIQUE(game_id, player_name, team_name)
    );
    CREATE INDEX IF NOT EXISTS idx_game_pit_player ON game_pitching(player_name);
    CREATE INDEX IF NOT EXISTS idx_game_pit_card ON game_pitching(card_id);
    CREATE INDEX IF NOT EXISTS idx_game_pit_team ON game_pitching(team_name);

    -- At-bat level data from game log HTML (news/html/game_logs/log_N.html).
    -- Each game log is a play-by-play — per pitch, per at-bat, per inning.
    -- This table captures the at-bat result with its quality-of-contact
    -- features (exit velocity, batted ball type, location). Feeds the
    -- superstats (LD%, true K%, EV-based confidence) that the aggregate
    -- box-score tables can't produce.
    CREATE TABLE IF NOT EXISTS game_log_at_bats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER NOT NULL,
        inning INTEGER,                     -- 1..9+, NULL if unparseable
        half_inning TEXT,                   -- 'top' | 'bottom'
        pitcher TEXT,
        pitcher_hand TEXT,                  -- 'LHP' | 'RHP'
        pitcher_card_id INTEGER,
        batter TEXT NOT NULL,
        batter_hand TEXT,                   -- 'LHB' | 'RHB' | 'SHB'
        batter_card_id INTEGER,
        outcome TEXT,                       -- 'SINGLE'|'DOUBLE'|'TRIPLE'|'HR'|'K'|'BB'|'HBP'|'OUT'|'SAC'|'FC'
        batted_ball_type TEXT,              -- 'Line Drive'|'Groundball'|'Fly Ball'|'Popup'|'Bunt'|NULL
        location TEXT,                      -- OOTP position code: '9LD','6MS','89D',...
        exit_velocity INTEGER,              -- mph, NULL if no contact (K/BB/HBP)
        pitches_seen INTEGER,
        strikeout_type TEXT,                -- 'looking'|'swinging'|NULL
        FOREIGN KEY (game_id) REFERENCES games(game_id)
    );
    CREATE INDEX IF NOT EXISTS idx_gl_batter ON game_log_at_bats(batter);
    CREATE INDEX IF NOT EXISTS idx_gl_pitcher ON game_log_at_bats(pitcher);
    CREATE INDEX IF NOT EXISTS idx_gl_batter_card ON game_log_at_bats(batter_card_id);
    CREATE INDEX IF NOT EXISTS idx_gl_pitcher_card ON game_log_at_bats(pitcher_card_id);
    CREATE INDEX IF NOT EXISTS idx_gl_game ON game_log_at_bats(game_id);

    -- Game narrative + ambient metadata parsed from box score HTML.
    -- Feeds sentiment analysis, "notable game" surfacing, ballpark/weather
    -- context for performance overlays. One row per game_id.
    CREATE TABLE IF NOT EXISTS game_narratives (
        game_id INTEGER PRIMARY KEY,
        recap_headline TEXT,
        recap_body TEXT,              -- Full prose recap with player names inline
        player_of_game TEXT,
        ballpark TEXT,
        weather TEXT,                 -- "Cloudy (72 degrees), wind blowing in from left at 12 mph"
        game_duration TEXT,
        start_time TEXT,
        attendance INTEGER,
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (game_id) REFERENCES games(game_id)
    );

    -- Clutch + situational events parsed from the box score event blocks
    -- ("Doubles:", "2-out RBI:", "Runners LOB in RISP 2-outs:", "SB:",
    --  "Errors:", "Inherited Runners - Scored:", "GIDP:", "Sac Fly:").
    -- These are the high-leverage per-player signals stats tables can't
    -- give us — used downstream for clutch-rate + defensive-reliability
    -- adjustments to meta calcs.
    CREATE TABLE IF NOT EXISTS game_clutch_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER NOT NULL,
        player_name TEXT NOT NULL,
        card_id INTEGER,
        team_name TEXT,
        event_type TEXT NOT NULL,     -- '2OUT_RBI','LOB_RISP_2OUT','ERROR','SB','CS',
                                      -- 'INHERITED_SCORED','SAC_FLY','GIDP','DP_FIELDER',
                                      -- 'DOUBLE','TRIPLE','HR','HBP_TAKEN'
        event_count INTEGER DEFAULT 1,
        inning INTEGER,               -- When the box score notes the inning
        context TEXT,                 -- "1 on, 0 outs", "bases empty, 2 outs"
        opposing_pitcher TEXT,        -- For hits
        season_count INTEGER,         -- Season total to-date (e.g. 19 for Wallace's 19th double)
        raw_text TEXT,                -- Verbatim line for auditing / future reparse
        FOREIGN KEY (game_id) REFERENCES games(game_id)
    );
    CREATE INDEX IF NOT EXISTS idx_clutch_game ON game_clutch_events(game_id);
    CREATE INDEX IF NOT EXISTS idx_clutch_player ON game_clutch_events(player_name);
    CREATE INDEX IF NOT EXISTS idx_clutch_card ON game_clutch_events(card_id);
    CREATE INDEX IF NOT EXISTS idx_clutch_type ON game_clutch_events(event_type);
    """)

    # --- Roster table migration: add split meta columns ---
    try:
        roster_cols = [row[1] for row in cursor.execute("PRAGMA table_info(roster)").fetchall()]
        if 'meta_vs_rhp' not in roster_cols:
            cursor.execute("ALTER TABLE roster ADD COLUMN meta_vs_rhp REAL")
        if 'meta_vs_lhp' not in roster_cols:
            cursor.execute("ALTER TABLE roster ADD COLUMN meta_vs_lhp REAL")
        if 'con_vl' not in roster_cols:
            for col in ['con_vl', 'pow_vl', 'eye_vl', 'con_vr', 'pow_vr', 'eye_vr']:
                cursor.execute(f"ALTER TABLE roster ADD COLUMN {col} INTEGER")
        if 'stu_vl' not in roster_cols:
            for col in ['stu_vl', 'stu_vr']:
                cursor.execute(f"ALTER TABLE roster ADD COLUMN {col} INTEGER")
        if 'card_title' not in roster_cols:
            cursor.execute("ALTER TABLE roster ADD COLUMN card_title TEXT")
        if 'card_id' not in roster_cols:
            cursor.execute("ALTER TABLE roster ADD COLUMN card_id INTEGER")
        if 'bats' not in roster_cols:
            cursor.execute("ALTER TABLE roster ADD COLUMN bats TEXT")
    except Exception:
        pass

    # --- league_team_stats migration: add ab + hits for team/player reconciliation ---
    try:
        lts_cols = [row[1] for row in cursor.execute("PRAGMA table_info(league_team_stats)").fetchall()]
        if lts_cols and 'ab' not in lts_cols:
            cursor.execute("ALTER TABLE league_team_stats ADD COLUMN ab INTEGER")
        if lts_cols and 'hits' not in lts_cols:
            cursor.execute("ALTER TABLE league_team_stats ADD COLUMN hits INTEGER")
    except Exception:
        pass

    # --- Add league_id to batting_stats, pitching_stats, roster ---
    migrate_add_league_columns(cursor)

    # --- Add jersey_number to batting_stats, pitching_stats, league_rosters ---
    # Same card_id can appear on multiple teams in OOTP PT (e.g. Aranda owned
    # by Vancouver #28, Toronto #8, EyeBeez #2). The league-wide CSV exposes
    # the jersey `#` per row but we previously dropped it, leaving N rows
    # per card with no way to disambiguate which team's instance is which.
    # Adding jersey_number lets us:
    #   * dedupe at insert time when needed,
    #   * filter to a user's specific team instance (via league_rosters join),
    #   * keep cross-team aggregates intact for portable card-level meta.
    for _table in ('batting_stats', 'pitching_stats', 'league_rosters'):
        try:
            _cols = [r[1] for r in cursor.execute(f"PRAGMA table_info({_table})").fetchall()]
            if _cols and 'jersey_number' not in _cols:
                cursor.execute(f"ALTER TABLE {_table} ADD COLUMN jersey_number INTEGER")
        except Exception:
            pass  # Table may not exist on a fresh install

    # Index on (card_id, league_id, jersey_number) so the team-specific
    # lookup helper (get_user_team_card_stats) is O(1) instead of full scan.
    try:
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_batting_stats_card_jersey "
            "ON batting_stats(card_id, league_id, jersey_number)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pitching_stats_card_jersey "
            "ON pitching_stats(card_id, league_id, jersey_number)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_league_rosters_card_team "
            "ON league_rosters(card_id, league_id, team_name)"
        )
    except Exception:
        pass

    # --- Stage 1c: partition batting_stats / pitching_stats by league_id ---
    #
    # Prior to the Stage 1 fix, league-wide stats files (``lb124_statistics_…``)
    # and team-scoped stats files (``toronto_dark_knights_…``) both wrote rows
    # with ``league_id IS NULL``. That mixed the user's own stats together with
    # 14 other teams' rows for the same player names — the visible symptom was
    # the Roster Optimizer showing wrong ERAs because it picked up an arbitrary
    # league-file row instead of the team-file row.
    #
    # Post-Stage-1 rule:
    #   * ``league_id IS NULL``  → team-scoped (the user's own team)
    #   * ``league_id = 'lb124'`` → league-wide row from the `lb124_…` file
    #
    # The cleanup below runs once to evict legacy polluted NULL-league rows
    # that came from the league-wide file path. A row is considered polluted
    # if its ``card_id`` doesn't match an owned card — team-file rows are
    # always pinned to owned cards by ``_match_card_id(prefer_owned=True)``.
    # Then we install a partial unique index so future team-file upserts
    # can't accidentally create duplicates.
    _stage1c_cleanup(cursor)

    # --- Views for latest snapshot (preserves history while keeping queries simple) ---
    cursor.executescript("""
    DROP VIEW IF EXISTS roster_current;
    CREATE VIEW roster_current AS
        SELECT * FROM roster
        WHERE lineup_role != 'league'
          AND DATE(snapshot_date) = (
              SELECT MAX(DATE(snapshot_date)) FROM roster WHERE lineup_role != 'league'
          );

    DROP VIEW IF EXISTS roster_league;
    CREATE VIEW roster_league AS
        SELECT * FROM roster
        WHERE lineup_role = 'league'
          AND DATE(snapshot_date) = (
              SELECT MAX(DATE(snapshot_date)) FROM roster WHERE lineup_role = 'league'
          );

    DROP VIEW IF EXISTS collection_current;
    CREATE VIEW collection_current AS
        SELECT * FROM my_collection
        WHERE DATE(snapshot_date) = (
            SELECT MAX(DATE(snapshot_date)) FROM my_collection
        );

    DROP VIEW IF EXISTS lineup_current;
    CREATE VIEW lineup_current AS
        SELECT * FROM team_lineup
        WHERE DATE(snapshot_date) = (
            SELECT MAX(DATE(snapshot_date)) FROM team_lineup
        );

    -- Per-league meta surface (UAT 2026-04-25 §4 / Tier-2 #5).
    -- ``cards.meta_score_pitching`` and ``cards.meta_score_batting`` are
    -- single global values, computed under the active-league weights at
    -- recalc time. With per-league calibration in ``meta_calibration``
    -- (``pitching:lb122``, ``pitching:lb124``...) we now have weights
    -- that differ across tiers — and therefore meta scores that should
    -- differ. This table caches per-tier meta scores so the optimizer
    -- can rank cards using the league/tier the user has selected.
    --
    -- Populated by ``meta_scoring.recalc_meta_scores_per_league(league_id)``
    -- (one row per card per league per side). A card with no row in
    -- this table for the chosen league falls back to ``cards.meta_score_*``
    -- via ``meta_scoring.get_meta_for_league``.
    CREATE TABLE IF NOT EXISTS card_meta_by_league (
        card_id INTEGER NOT NULL,
        league_id TEXT NOT NULL,
        side TEXT NOT NULL CHECK (side IN ('batting', 'pitching')),
        meta_score REAL NOT NULL,
        weight_source TEXT,                  -- 'pitching:lb122' / 'batting' / etc.
        recalc_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (card_id, league_id, side),
        FOREIGN KEY (card_id) REFERENCES cards(card_id)
    );
    CREATE INDEX IF NOT EXISTS idx_cmbl_card_league
        ON card_meta_by_league(card_id, league_id);

    -- Snapshot-dedup views (UAT 2026-04-25 §5.4 / Tier-2 #10): each card_id
    -- can have many rows in batting_stats / pitching_stats — one per
    -- (league_id, jersey_number, snapshot_date). Naive joins fan that
    -- multiplication out across dependent tables. These views project to
    -- the latest snapshot per (card_id, league_id, jersey_number) — i.e.
    -- the most recent stat-line for each (card, team-instance, league).
    --
    -- IMPORTANT: jersey# IS in the dedup key on purpose. Multiple teams
    -- in the same league can own the same card (e.g. Aranda owned by
    -- Vancouver, Toronto, and EyeBeez — three rows under one league at
    -- the same snapshot, distinguished only by jersey#). Collapsing them
    -- would discard real per-team data and bias the user's view toward
    -- whichever team happened to have the highest PA/IP. Historical
    -- snapshots are still preserved in the base table; this view just
    -- picks the most recent one per team-instance.
    --
    -- Consumers that want the league-wide picture should aggregate across
    -- jersey# explicitly (or sum). Consumers wanting "this card on this
    -- team" should filter on jersey_number. The page's
    -- ``_load_latest_perf_stats`` already does this for the user's team.
    DROP VIEW IF EXISTS batting_stats_latest;
    CREATE VIEW batting_stats_latest AS
        SELECT bs.* FROM batting_stats bs
        INNER JOIN (
            SELECT card_id, league_id, jersey_number,
                   MAX(snapshot_date) AS max_date
            FROM batting_stats
            WHERE card_id IS NOT NULL
            GROUP BY card_id, league_id, jersey_number
        ) latest
          ON bs.card_id = latest.card_id
         AND IFNULL(bs.league_id, '') = IFNULL(latest.league_id, '')
         AND IFNULL(bs.jersey_number, -1) = IFNULL(latest.jersey_number, -1)
         AND bs.snapshot_date = latest.max_date;

    DROP VIEW IF EXISTS pitching_stats_latest;
    CREATE VIEW pitching_stats_latest AS
        SELECT ps.* FROM pitching_stats ps
        INNER JOIN (
            SELECT card_id, league_id, jersey_number,
                   MAX(snapshot_date) AS max_date
            FROM pitching_stats
            WHERE card_id IS NOT NULL
            GROUP BY card_id, league_id, jersey_number
        ) latest
          ON ps.card_id = latest.card_id
         AND IFNULL(ps.league_id, '') = IFNULL(latest.league_id, '')
         AND IFNULL(ps.jersey_number, -1) = IFNULL(latest.jersey_number, -1)
         AND ps.snapshot_date = latest.max_date;
    """)

    conn.commit()
    conn.close()
    _INIT_DB_DONE = True
