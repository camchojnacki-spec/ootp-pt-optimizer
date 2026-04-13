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


def get_db_path() -> str:
    """Get the database path from config, defaulting to data/ootp_optimizer.db."""
    try:
        config = load_config()
        db_rel = config.get("database_path", "data/ootp_optimizer.db")
    except Exception:
        db_rel = "data/ootp_optimizer.db"
    return str(PROJECT_ROOT / db_rel)


def get_connection() -> sqlite3.Connection:
    """Return a sqlite3 connection with row_factory = sqlite3.Row."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def migrate_add_league_columns(cursor: sqlite3.Cursor) -> None:
    """Safely add league_id column to existing tables if not already present."""
    for table in ("batting_stats", "pitching_stats", "roster"):
        try:
            cols = [row[1] for row in cursor.execute(f"PRAGMA table_info({table})").fetchall()]
            if "league_id" not in cols:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN league_id TEXT")
        except Exception:
            pass  # Table may not exist yet on fresh install


def init_db() -> None:
    """Create all tables if they do not exist."""
    db_path = get_db_path()
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
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

    # --- Add league_id to batting_stats, pitching_stats, roster ---
    migrate_add_league_columns(cursor)

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
    """)

    conn.commit()
    conn.close()
