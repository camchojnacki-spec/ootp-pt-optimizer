"""CSV ingestion engine — parses CSVs and writes to SQLite."""
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from app.utils.constants import POSITION_MAP, PITCHER_ROLE_MAP, TIER_MAP
from app.utils.csv_parser import (
    parse_market_csv, parse_roster_batting_csv, parse_roster_pitching_csv,
    parse_collection_batting_csv, parse_collection_pitching_csv,
    parse_stats_batting_csv, parse_stats_pitching_csv, identify_file_type,
    parse_stats_batting_adv_csv, parse_stats_pitching_adv_csv,
    parse_lineup_csv, parse_team_pitching_csv,
    parse_league_batting_ratings_csv, parse_league_pitching_ratings_csv,
    parse_league_player_default_csv,
    parse_fielding_stats_csv, parse_position_ratings_csv, parse_pitch_ratings_csv,
    parse_team_stats_batting_csv, parse_team_stats_pitching_csv, parse_team_stats_park_csv,
)
from app.core.database import get_connection, load_config
from app.core.meta_scoring import calc_batting_meta, calc_pitching_meta, calc_defense_score, calc_speed_score


def _safe_int(value, default=0):
    """Safely convert a value to int, handling NaN and None."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _safe_float(value, default=0.0):
    """Safely convert a value to float, handling NaN and None."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_str(value, default=''):
    """Safely convert a value to str, handling NaN and None."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value)


def detect_league_id(filepath: str) -> str | None:
    """Try to detect league_id from the filename prefix (e.g. 'lb124_...' -> 'lb124')."""
    fname = Path(filepath).name.lower()
    # League stats files are prefixed with league_id + '_statistics_'
    if '_statistics_' in fname:
        prefix = fname.split('_statistics_')[0]
        if prefix and len(prefix) <= 10:
            return prefix
    return None


def _get_active_league() -> str | None:
    """Read the currently active league_id from config.yaml.

    Used by the stale-league gate in ingest_file() to skip stats/ratings
    files from non-current leagues (e.g. i76_* files after moving to lb124).
    Returns None if config is missing or the key is unset, which disables
    the gate (legacy behavior).
    """
    try:
        cfg = load_config()
        return cfg.get("active_league")
    except Exception:
        return None


def ingest_file(filepath: str, league_id: str = None, allow_cross_league: bool = False) -> dict:
    """Auto-detect file type and ingest. Returns dict with status info.

    If league_id is provided, it's passed to handlers that support it
    and stored in the ingestion log for tracking.

    Stale-league gate: if the file has a detected league_id (league-scoped
    exports like `i76_statistics_...`, `lb124_statistics_...`) and that
    league_id doesn't match config.active_league, the file is skipped so
    its stale data can't pollute pitching_stats / batting_stats / roster.
    Team-scoped files (prefix `toronto_dark_knights_`) have league_id=None
    and always import.

    ``allow_cross_league=True`` bypasses the gate. Only use this for
    explicit cross-league analysis (e.g., ingesting i76 alongside lb124
    to validate whether meta weights generalize across tiers).
    """
    file_type = identify_file_type(filepath)
    if file_type is None:
        return {"status": "skipped", "reason": "Unknown file type", "file": filepath}

    # Auto-detect league from filename if not provided
    if league_id is None:
        league_id = detect_league_id(filepath)

    # Stale-league gate: skip league-wide files from non-active leagues.
    # The active league is set in config.yaml. Team-scoped files have
    # league_id=None and pass through. If active_league is unset, the
    # gate is disabled (legacy behavior).
    _active = _get_active_league()
    if league_id and _active and league_id != _active and not allow_cross_league:
        # Log the skip so freshness/UI layers can still see that we saw
        # the file (with row_count=0 so it's clearly a noop).
        try:
            conn = get_connection()
            conn.execute(
                "INSERT INTO ingestion_log (file_type, file_name, row_count) VALUES (?, ?, ?)",
                (file_type + ":stale_league", Path(filepath).name, 0),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        return {
            "status": "skipped",
            "file_type": file_type,
            "rows": 0,
            "reason": f"file from league {league_id!r} but active league is {_active!r}",
            "league_id": league_id,
        }

    handlers = {
        "market": ingest_market_data,
        "roster_batting": ingest_roster_batting,
        "roster_pitching": ingest_roster_pitching,
        "collection_batting": ingest_collection_batting,
        "collection_pitching": ingest_collection_pitching,
        "stats_batting": ingest_stats_batting,
        "stats_pitching": ingest_stats_pitching,
        "team_stats_batting": ingest_team_stats_batting,
        "team_stats_pitching": ingest_team_stats_pitching,
        "team_stats_park": ingest_team_stats_park,
        "roster_batting_stats": ingest_roster_batting_stats,
        "roster_pitching_stats": ingest_roster_pitching_stats,
        "stats_batting_ratings": ingest_league_batting_ratings,
        "stats_pitching_ratings": ingest_league_pitching_ratings,
        "league_player_default": ingest_league_default,
        "lineup_vs_rhp": ingest_lineup,
        "lineup_vs_lhp": ingest_lineup,
        "lineup_overview": ingest_lineup,
        "team_pitching": ingest_team_pitching,
        "fielding_stats": ingest_fielding_stats,
        "fielding_ratings": ingest_fielding_ratings,
        "pitch_ratings": ingest_pitch_ratings,
        "position_ratings": ingest_position_ratings,
        "team_standings": ingest_team_standings,
        "team_stats_fielding": ingest_team_stats_fielding,
    }

    handler = handlers[file_type]
    # Some handlers need the file_type (e.g. lineup variants)
    import inspect
    sig = inspect.signature(handler)
    if len(sig.parameters) >= 2:
        result = handler(filepath, file_type)
    else:
        result = handler(filepath)

    # Store league_id in result for downstream use
    result["league_id"] = league_id

    # Log ingestion with league_id
    conn = get_connection()
    conn.execute(
        "INSERT INTO ingestion_log (file_type, file_name, row_count) VALUES (?, ?, ?)",
        (file_type, Path(filepath).name, result.get("rows", 0))
    )
    conn.commit()
    conn.close()

    return result


def ingest_batch_with_history(filepaths: list[str], league_id: str = None,
                              games_into_season: int = None,
                              team_record: str = None) -> dict:
    """Import a batch of files AND automatically take a player history snapshot.

    This is the recommended way to import — it ensures every export is
    tracked for trending. Call this instead of individual ingest_file() calls.
    """
    from app.core.history import snapshot_player_history, ensure_league_exists

    # Auto-detect league from the first league-prefixed file
    if league_id is None:
        for fp in filepaths:
            lid = detect_league_id(fp)
            if lid:
                league_id = lid
                break

    if league_id:
        ensure_league_exists(league_id)

    results = []
    for fp in filepaths:
        try:
            r = ingest_file(fp, league_id=league_id)
            results.append(r)
        except Exception as e:
            results.append({"status": "error", "file": fp, "error": str(e)})

    # Take comprehensive snapshot after all files imported
    snapshot = None
    if league_id:
        try:
            snapshot = snapshot_player_history(
                league_id=league_id,
                games_into_season=games_into_season,
                team_record=team_record,
            )
        except Exception as e:
            snapshot = {"error": str(e)}

    ok = sum(1 for r in results if r.get("status") == "success")
    return {
        "files_imported": ok,
        "files_total": len(filepaths),
        "results": results,
        "snapshot": snapshot,
        "league_id": league_id,
    }


def ingest_market_data(filepath: str) -> dict:
    """Ingest market card list CSV into cards table and price_snapshots."""
    df = parse_market_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    # Try to get date from the CSV data (date-only for daily dedup)
    if 'date' in df.columns and len(df) > 0:
        csv_date = df['date'].iloc[0]
        if pd.notna(csv_date) and str(csv_date).strip():
            snapshot_date = str(csv_date).strip()[:10]

    count = 0
    for _, row in df.iterrows():
        card_id = _safe_int(row.get('Card ID', 0))
        if card_id == 0:
            continue

        pos = _safe_int(row.get('Position', 0))
        pos_name = POSITION_MAP.get(pos, "")
        pr = _safe_int(row.get('Pitcher Role', 0))
        pr_name = PITCHER_ROLE_MAP.get(pr, "")
        tier_val = _safe_int(row.get('tier', 1))
        tier_name = TIER_MAP.get(tier_val, "Regular")

        # Calculate meta scores
        card_dict = {
            'position': pos,
            'card_value': _safe_int(row.get('Card Value', 0)),
            'gap_power': _safe_int(row.get('Gap', 0)),
            'contact': _safe_int(row.get('Contact', 0)),
            'avoid_ks': _safe_int(row.get('Avoid Ks', 0)),
            'eye': _safe_int(row.get('Eye', 0)),
            'power': _safe_int(row.get('Power', 0)),
            'babip': _safe_int(row.get('BABIP', 0)),
            'of_range': _safe_int(row.get('OF Range', 0)),
            'of_error': _safe_int(row.get('OF Error', 0)),
            'of_arm': _safe_int(row.get('OF Arm', 0)),
            'catcher_ability': _safe_int(row.get('CatcherAbil', 0)),
            'catcher_frame': _safe_int(row.get('CatcherFrame', 0)),
            'catcher_arm': _safe_int(row.get('Catcher Arm', 0)),
            'infield_range': _safe_int(row.get('Infield Range', 0)),
            'infield_error': _safe_int(row.get('Infield Error', 0)),
            'infield_arm': _safe_int(row.get('Infield Arm', 0)),
            'movement': _safe_int(row.get('Movement', 0)),
            'stuff': _safe_int(row.get('Stuff', 0)),
            'control': _safe_int(row.get('Control', 0)),
            'p_hr': _safe_int(row.get('pHR', 0)),
            'stamina': _safe_int(row.get('Stamina', 0)),
            'hold': _safe_int(row.get('Hold', 0)),
        }

        is_pitcher = (pr in (11, 12, 13))
        batting_meta = calc_batting_meta(card_dict) if not is_pitcher else None
        pitching_meta = calc_pitching_meta(card_dict) if is_pitcher else None

        # Calculate age from YearOB (approximate)
        year_ob = _safe_int(row.get('YearOB', 0))
        age = 2026 - year_ob if year_ob > 0 else None

        # Build values tuple
        values = (
            card_id, _safe_str(row.get('Card Title', '')), _safe_str(row.get('FirstName', '')),
            _safe_str(row.get('LastName', '')), _safe_str(row.get('NickName', '')),
            pos, pos_name, pr if pr else None, pr_name if pr_name else None,
            _safe_str(row.get('Bats', '')), _safe_str(row.get('Throws', '')),
            age,
            _safe_str(row.get('Team', '')), _safe_str(row.get('Franchise', '')),
            _safe_int(row.get('Card Type', 0)), _safe_str(row.get('Card Sub Type', '')),
            _safe_str(row.get('Card Badge', '')), _safe_str(row.get('Card Series', '')),
            _safe_int(row.get('Card Value', 0)),
            _safe_int(row.get('Year', 0)), _safe_str(row.get('Peak', '')),
            tier_val, tier_name, _safe_str(row.get('Nation', '')),
            _safe_int(row.get('Contact', 0)), _safe_int(row.get('Gap', 0)),
            _safe_int(row.get('Power', 0)), _safe_int(row.get('Eye', 0)),
            _safe_int(row.get('Avoid Ks', 0)), _safe_int(row.get('BABIP', 0)),
            _safe_int(row.get('Contact vL', 0)), _safe_int(row.get('Gap vL', 0)),
            _safe_int(row.get('Power vL', 0)), _safe_int(row.get('Eye vL', 0)),
            _safe_int(row.get('Avoid K vL', 0)), _safe_int(row.get('BABIP vL', 0)),
            _safe_int(row.get('Contact vR', 0)), _safe_int(row.get('Gap vR', 0)),
            _safe_int(row.get('Power vR', 0)), _safe_int(row.get('Eye vR', 0)),
            _safe_int(row.get('Avoid K vR', 0)), _safe_int(row.get('BABIP vR', 0)),
            _safe_int(row.get('Speed', 0)), _safe_int(row.get('Steal Rate', 0)),
            _safe_int(row.get('Stealing', 0)), _safe_int(row.get('Baserunning', 0)),
            _safe_int(row.get('Sac bunt', 0)), _safe_int(row.get('Bunt for hit', 0)),
            _safe_int(row.get('Stuff', 0)), _safe_int(row.get('Movement', 0)),
            _safe_int(row.get('Control', 0)), _safe_int(row.get('pHR', 0)),
            _safe_int(row.get('pBABIP', 0)),
            _safe_int(row.get('Stuff vL', 0)), _safe_int(row.get('Movement vL', 0)),
            _safe_int(row.get('Control vL', 0)), _safe_int(row.get('pHR vL', 0)),
            _safe_int(row.get('pBABIP vL', 0)),
            _safe_int(row.get('Stuff vR', 0)), _safe_int(row.get('Movement vR', 0)),
            _safe_int(row.get('Control vR', 0)), _safe_int(row.get('pHR vR', 0)),
            _safe_int(row.get('pBABIP vR', 0)),
            _safe_int(row.get('Stamina', 0)), _safe_int(row.get('Hold', 0)),
            _safe_str(row.get('Velocity', '')),
            _safe_int(row.get('Infield Range', 0)), _safe_int(row.get('Infield Error', 0)),
            _safe_int(row.get('Infield Arm', 0)), _safe_int(row.get('DP', 0)),
            _safe_int(row.get('CatcherAbil', 0)), _safe_int(row.get('CatcherFrame', 0)),
            _safe_int(row.get('Catcher Arm', 0)),
            _safe_int(row.get('OF Range', 0)), _safe_int(row.get('OF Error', 0)),
            _safe_int(row.get('OF Arm', 0)),
            _safe_int(row.get('Pos Rating P', 0)), _safe_int(row.get('Pos Rating C', 0)),
            _safe_int(row.get('Pos Rating 1B', 0)), _safe_int(row.get('Pos Rating 2B', 0)),
            _safe_int(row.get('Pos Rating 3B', 0)), _safe_int(row.get('Pos Rating SS', 0)),
            _safe_int(row.get('Pos Rating LF', 0)), _safe_int(row.get('Pos Rating CF', 0)),
            _safe_int(row.get('Pos Rating RF', 0)),
            _safe_int(row.get('MissionValue', 0)), _safe_int(row.get('limit', 0)),
            _safe_str(row.get('brefid', '')), _safe_str(row.get('packs', '')),
            _safe_int(row.get('owned', 0)),
            batting_meta, pitching_meta,
            _safe_int(row.get('Buy Order High', 0)), _safe_int(row.get('Sell Order Low', 0)),
            _safe_int(row.get('Last 10 Price', 0)), _safe_int(row.get('Last 10 Price(VAR)', 0)),
            # UTC to match roster.snapshot_date (which uses SQLite CURRENT_TIMESTAMP / UTC).
            # This lets the data-freshness check compare timestamps apples-to-apples.
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        )

        # Upsert into cards table
        cols = [
            'card_id', 'card_title', 'first_name', 'last_name', 'nickname',
            'position', 'position_name', 'pitcher_role', 'pitcher_role_name',
            'bats', 'throws', 'age', 'team', 'franchise',
            'card_type', 'card_sub_type', 'card_badge', 'card_series', 'card_value',
            'year', 'peak', 'tier', 'tier_name', 'nation',
            'contact', 'gap_power', 'power', 'eye', 'avoid_ks', 'babip',
            'contact_vl', 'gap_vl', 'power_vl', 'eye_vl', 'avoid_ks_vl', 'babip_vl',
            'contact_vr', 'gap_vr', 'power_vr', 'eye_vr', 'avoid_ks_vr', 'babip_vr',
            'speed', 'steal_rate', 'stealing', 'baserunning', 'sac_bunt', 'bunt_for_hit',
            'stuff', 'movement', 'control', 'p_hr', 'p_babip',
            'stuff_vl', 'movement_vl', 'control_vl', 'p_hr_vl', 'p_babip_vl',
            'stuff_vr', 'movement_vr', 'control_vr', 'p_hr_vr', 'p_babip_vr',
            'stamina', 'hold', 'velocity',
            'infield_range', 'infield_error', 'infield_arm', 'dp',
            'catcher_ability', 'catcher_frame', 'catcher_arm',
            'of_range', 'of_error', 'of_arm',
            'pos_rating_p', 'pos_rating_c', 'pos_rating_1b',
            'pos_rating_2b', 'pos_rating_3b', 'pos_rating_ss',
            'pos_rating_lf', 'pos_rating_cf', 'pos_rating_rf',
            'mission_value', 'card_limit', 'bref_id', 'packs', 'owned',
            'meta_score_batting', 'meta_score_pitching',
            'buy_order_high', 'sell_order_low', 'last_10_price', 'last_10_variance',
            'last_updated_at',
        ]
        placeholders = ','.join(['?'] * len(cols))
        col_names = ','.join(cols)
        # Update every column except the primary key on conflict. Prior code
        # only refreshed ~10 of ~90 columns, so card ratings (contact, power,
        # stuff, splits, fielding) stayed pinned to the *first* market
        # snapshot for each card_id — even after OOTP recalibrated ratings
        # mid-season. recalculate_all_meta_scores then operated on stale
        # inputs forever. Explicit list so new columns are opted in on add.
        update_clause = ','.join(f"{c}=excluded.{c}" for c in cols if c != 'card_id')
        cursor.execute(f"""
            INSERT INTO cards ({col_names}) VALUES ({placeholders})
            ON CONFLICT(card_id) DO UPDATE SET {update_clause}
        """, values)

        # Insert price snapshot (deduplicate by card_id + date via unique index)
        cursor.execute("""
            INSERT OR IGNORE INTO price_snapshots
                (card_id, snapshot_date, buy_order_high, sell_order_low, last_10_price, last_10_variance)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            card_id, snapshot_date,
            _safe_int(row.get('Buy Order High', 0)), _safe_int(row.get('Sell Order Low', 0)),
            _safe_int(row.get('Last 10 Price', 0)), _safe_int(row.get('Last 10 Price(VAR)', 0)),
        ))

        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "market", "rows": count}


def ingest_roster_batting(filepath: str) -> dict:
    """Ingest roster batting ratings into roster and my_collection tables."""
    from app.core.meta_scoring import calc_batting_meta_vs_rhp, calc_batting_meta_vs_lhp
    df = parse_roster_batting_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    # Clear current roster batting entries (not league entries)
    cursor.execute("DELETE FROM roster WHERE position NOT IN ('SP', 'RP', 'CL') AND lineup_role != 'league' AND DATE(snapshot_date) = DATE('now')")

    _BATTER_POSITIONS = {'C','1B','2B','3B','SS','LF','CF','RF','DH','OF','IF','UT'}
    _skipped_non_batter = 0
    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        pos = _safe_str(row.get('POS', '')).strip()
        ovr = _safe_int(row.get('OVR', 0))
        status = _safe_str(row.get('St', '')).strip()

        # OOTP's batting-ratings CSV includes pitchers too (SP/RP/CL rows).
        # Inserting them with a batter ``lineup_role`` gave every pitcher a
        # phantom "starter" row at their pitcher slot which confused the
        # observed-starter override and Bench Bats dedup. Skip any row
        # whose POS isn't a batter slot.
        if pos not in _BATTER_POSITIONS:
            _skipped_non_batter += 1
            continue

        # Calculate meta with split ratings
        card_dict = {
            'contact': _safe_int(row.get('CON', 0)),
            'gap_power': _safe_int(row.get('GAP', 0)),
            'avoid_ks': _safe_int(row.get("K's", 0)),
            'eye': _safe_int(row.get('EYE', 0)),
            'power': _safe_int(row.get('POW', 0)),
            'babip': _safe_int(row.get('BABIP', 0)),
            'defense_score': _safe_float(row.get('DEF', 0)),
            'card_value': ovr,
            # Split ratings for platoon meta
            'con_vl': _safe_int(row.get('CON vL', 0)),
            'pow_vl': _safe_int(row.get('POW vL', 0)),
            'eye_vl': _safe_int(row.get('EYE vL', 0)),
            'con_vr': _safe_int(row.get('CON vR', 0)),
            'pow_vr': _safe_int(row.get('POW vR', 0)),
            'eye_vr': _safe_int(row.get('EYE vR', 0)),
        }
        meta = calc_batting_meta(card_dict)
        meta_rhp = calc_batting_meta_vs_rhp(card_dict)
        meta_lhp = calc_batting_meta_vs_lhp(card_dict)

        # Determine lineup role
        lineup_role = "bench" if status == "Reserve Roster" else "starter"

        # Route through the shared resolver (expands "C. Kluttz" etc.).
        card_id = _match_card_id(cursor, name, prefer_owned=True)

        cursor.execute("""
            INSERT OR IGNORE INTO roster (player_name, position, lineup_role, ovr, meta_score,
                meta_vs_rhp, meta_vs_lhp, con_vl, pow_vl, eye_vl, con_vr, pow_vr, eye_vr, bats, card_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, pos, lineup_role, ovr, meta, meta_rhp, meta_lhp,
              _safe_int(row.get('CON vL', 0)), _safe_int(row.get('POW vL', 0)),
              _safe_int(row.get('EYE vL', 0)), _safe_int(row.get('CON vR', 0)),
              _safe_int(row.get('POW vR', 0)), _safe_int(row.get('EYE vR', 0)),
              _safe_str(row.get('B', '')).strip(), card_id))

        cursor.execute("""
            INSERT INTO my_collection (card_id, player_name, position, ovr, status,
                contact, gap_power, avoid_ks, eye, power, babip, defense_score, meta_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            card_id, name, pos, ovr, status,
            _safe_int(row.get('CON', 0)), _safe_int(row.get('GAP', 0)),
            _safe_int(row.get("K's", 0)), _safe_int(row.get('EYE', 0)),
            _safe_int(row.get('POW', 0)), _safe_int(row.get('BABIP', 0)),
            _safe_float(row.get('DEF', 0)), meta
        ))

        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "roster_batting", "rows": count}


def ingest_roster_pitching(filepath: str) -> dict:
    """Ingest roster pitching ratings."""
    from app.core.meta_scoring import calc_pitching_meta_vs_lhb, calc_pitching_meta_vs_rhb
    df = parse_roster_pitching_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    # Clear current roster pitching entries (not league entries)
    cursor.execute("DELETE FROM roster WHERE position IN ('SP', 'RP', 'CL') AND lineup_role != 'league' AND DATE(snapshot_date) = DATE('now')")

    count = 0
    _PITCHER_POSITIONS = {'SP', 'RP', 'CL', 'MOP', 'MID', 'LNG', 'SU'}
    _skipped_non_pitcher = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        pos = _safe_str(row.get('POS', '')).strip()
        ovr = _safe_int(row.get('OVR', 0))
        status = _safe_str(row.get('St', '')).strip()

        # OOTP's pitching-ratings CSV includes EVERY player on the team (not
        # just pitchers). Inserting them all with a pitcher ``lineup_role``
        # gave every batter a phantom "bullpen" row (Bo Bichette SS bullpen,
        # Jonathan Aranda 1B bullpen, etc.) which bloated ``active_by_pos``
        # and caused duplicate Bench Bats + wrong-role classifications in
        # the rec tracker. Skip any row whose POS isn't a pitcher slot.
        if pos not in _PITCHER_POSITIONS:
            _skipped_non_pitcher += 1
            continue

        card_dict = {
            'stuff': _safe_int(row.get('STU', 0)),
            'movement': _safe_int(row.get('MOV', 0)),
            'control': _safe_int(row.get('CON', 0)),
            'p_hr': _safe_int(row.get('HRA', 0)),
            'card_value': ovr,
            'stamina': _safe_int(row.get('STM', 0)) or _safe_int(row.get('STA', 0)),
            'hold': _safe_int(row.get('HLD', 0)) or _safe_int(row.get('Hold', 0)),
            'stu_vl': _safe_int(row.get('STU vL', 0)),
            'stu_vr': _safe_int(row.get('STU vR', 0)),
        }
        meta = calc_pitching_meta(card_dict)
        meta_vs_lhb = calc_pitching_meta_vs_lhb(card_dict)
        meta_vs_rhb = calc_pitching_meta_vs_rhb(card_dict)

        lineup_role = "rotation" if pos == "SP" else ("closer" if pos == "CL" else "bullpen")
        if status == "Reserve Roster":
            lineup_role = "reserve"

        # Route through the shared resolver (same as batting ingest above).
        card_id = _match_card_id(cursor, name, prefer_owned=True)

        cursor.execute("""
            INSERT OR IGNORE INTO roster (player_name, position, lineup_role, ovr, meta_score,
                meta_vs_rhp, meta_vs_lhp, stu_vl, stu_vr, card_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, pos, lineup_role, ovr, meta, meta_vs_rhb, meta_vs_lhb,
              _safe_int(row.get('STU vL', 0)), _safe_int(row.get('STU vR', 0)), card_id))

        cursor.execute("""
            INSERT INTO my_collection (card_id, player_name, position, ovr, status,
                stuff, movement, ctrl, p_hr, meta_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            card_id, name, pos, ovr, status,
            _safe_int(row.get('STU', 0)), _safe_int(row.get('MOV', 0)),
            _safe_int(row.get('CON', 0)), _safe_int(row.get('HRA', 0)), meta
        ))

        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "roster_pitching", "rows": count}


def ingest_collection_batting(filepath: str) -> dict:
    """Ingest collection batting ratings into my_collection."""
    df = parse_collection_batting_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    # Clear collection batting entries (not from roster files)
    # We keep roster entries separate; collection replaces itself
    cursor.execute("DELETE FROM my_collection WHERE position NOT IN ('SP', 'RP', 'CL') AND DATE(snapshot_date) = DATE('now')")

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        pos = _safe_str(row.get('POS', '')).strip()
        ovr = _safe_int(row.get('OVR', 0))
        status = _safe_str(row.get('St', '')).strip()

        card_dict = {
            'contact': _safe_int(row.get('CON', 0)),
            'gap_power': _safe_int(row.get('GAP', 0)),
            'avoid_ks': _safe_int(row.get("K's", 0)),
            'eye': _safe_int(row.get('EYE', 0)),
            'power': _safe_int(row.get('POW', 0)),
            'babip': _safe_int(row.get('BABIP', 0)),
            'defense_score': _safe_float(row.get('DEF', 0)),
            'card_value': ovr,
        }
        meta = calc_batting_meta(card_dict)

        card_id = _match_card_id(cursor, name, prefer_owned=False)

        cursor.execute("""
            INSERT INTO my_collection (card_id, player_name, position, ovr, status,
                contact, gap_power, avoid_ks, eye, power, babip, defense_score, meta_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            card_id, name, pos, ovr, status,
            _safe_int(row.get('CON', 0)), _safe_int(row.get('GAP', 0)),
            _safe_int(row.get("K's", 0)), _safe_int(row.get('EYE', 0)),
            _safe_int(row.get('POW', 0)), _safe_int(row.get('BABIP', 0)),
            _safe_float(row.get('DEF', 0)), meta
        ))

        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "collection_batting", "rows": count}


def ingest_collection_pitching(filepath: str) -> dict:
    """Ingest collection pitching ratings into my_collection."""
    df = parse_collection_pitching_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM my_collection WHERE position IN ('SP', 'RP', 'CL') AND DATE(snapshot_date) = DATE('now')")

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        pos = _safe_str(row.get('POS', '')).strip()
        ovr = _safe_int(row.get('OVR', 0))
        status = _safe_str(row.get('St', '')).strip()

        card_dict = {
            'stuff': _safe_int(row.get('STU', 0)),
            'movement': _safe_int(row.get('MOV', 0)),
            'control': _safe_int(row.get('CON', 0)),
            'p_hr': _safe_int(row.get('HRA', 0)),
            'card_value': ovr,
            'stamina': _safe_int(row.get('STA', 0)),
            'hold': _safe_int(row.get('Hold', 0)),
        }
        meta = calc_pitching_meta(card_dict)

        card_id = _match_card_id(cursor, name, prefer_owned=False)

        cursor.execute("""
            INSERT INTO my_collection (card_id, player_name, position, ovr, status,
                stuff, movement, ctrl, p_hr, meta_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            card_id, name, pos, ovr, status,
            _safe_int(row.get('STU', 0)), _safe_int(row.get('MOV', 0)),
            _safe_int(row.get('CON', 0)), _safe_int(row.get('HRA', 0)), meta
        ))

        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "collection_pitching", "rows": count}


def _match_card_id(cursor, player_name: str, prefer_owned: bool = True) -> int | None:
    """Try to find a card_id by matching a player name in the cards table.

    Routes through ``app.core.name_resolver.resolve_to_card_id`` which:
      * Expands abbreviated names ("C. Kluttz" → "Clyde Kluttz")
      * Refuses to guess when (initial, lastname) is ambiguous
      * Exact first_name+last_name match (no unsafe LIKE substring)
      * Prefers owned cards when requested

    Falls back to the legacy LIKE-substring path if the resolver returns
    None (e.g. a new player not yet in the cards table) — that keeps the
    old success cases intact while dramatically lifting abbreviated-name
    resolution from 0.4% → ~100%.
    """
    if not player_name:
        return None
    # Grab a connection from the cursor if possible; fall back to global.
    try:
        conn = cursor.connection
    except AttributeError:
        conn = None

    try:
        from app.core.name_resolver import resolve_to_card_id
        cid = resolve_to_card_id(player_name, prefer_owned=prefer_owned, conn=conn)
        if cid is not None:
            return cid
    except Exception as _e:
        # If resolver breaks for any reason, fall through to the legacy path
        # so ingestion doesn't stall.
        import logging as _lg
        _lg.getLogger(__name__).debug("resolve_to_card_id failed: %s", _e)

    # Legacy fallback — unchanged from prior behavior.
    if prefer_owned:
        owned_row = cursor.execute(
            "SELECT card_id FROM cards "
            "WHERE owned >= 1 AND ((first_name || ' ' || last_name) = ? OR card_title LIKE ?) "
            "ORDER BY card_value DESC LIMIT 1",
            (player_name, f"%{player_name}%"),
        ).fetchone()
        if owned_row:
            return owned_row[0]
    card_row = cursor.execute(
        "SELECT card_id FROM cards "
        "WHERE (first_name || ' ' || last_name) = ? OR card_title LIKE ? LIMIT 1",
        (player_name, f"%{player_name}%"),
    ).fetchone()
    return card_row[0] if card_row else None


def ingest_stats_batting(filepath: str) -> dict:
    """Ingest sortable batting stats CSV into batting_stats table.

    Each import replaces current stats (latest snapshot) and preserves history
    by checking for an existing snapshot from the same date.
    Detects stats_2 format (wOBA, BB%, K%) and routes to advanced handler.

    League scoping: these files are league-wide (e.g. ``lb124_statistics_...``)
    and have a non-null ``league_id``. The DELETE/INSERT are scoped by
    ``league_id`` so they don't wipe or collide with the team-file path, which
    writes ``league_id IS NULL`` rows for the user's own team stats.
    """
    df = parse_stats_batting_csv(filepath)
    cols = set(df.columns)
    # Detect stats_2 format
    if 'wOBA' in cols or 'BB%' in cols:
        return ingest_batting_stats_adv(filepath)

    conn = get_connection()
    cursor = conn.cursor()

    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    league_id = detect_league_id(filepath)

    # Delete any existing snapshot from today for THIS league only.
    # Team-scoped rows (league_id IS NULL) are NOT touched.
    if league_id is None:
        cursor.execute(
            "DELETE FROM batting_stats WHERE DATE(snapshot_date) = ? AND league_id IS NULL",
            (snapshot_date,),
        )
    else:
        cursor.execute(
            "DELETE FROM batting_stats WHERE DATE(snapshot_date) = ? AND league_id = ?",
            (snapshot_date, league_id),
        )

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name or name.lower() in ('total', 'totals', 'team'):
            continue

        pos = _safe_str(row.get('POS', '')).strip()
        bats = _safe_str(row.get('B', '')).strip()
        throws = _safe_str(row.get('T', '')).strip()

        # League rows represent other teams' cards; do NOT prefer owned or
        # they'll all get pinned to the user's own card_id.
        card_id = _match_card_id(cursor, name, prefer_owned=False)

        cursor.execute("""
            INSERT INTO batting_stats (
                player_name, position, bats, throws,
                games, pa, ab, hits, doubles, triples, hr, rbi, runs,
                bb, ibb, hbp, k, gidp,
                avg, obp, slg, iso, ops, ops_plus, babip, war,
                sb, cs, card_id, snapshot_date, league_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, pos, bats, throws,
            _safe_int(row.get('G', 0)),
            _safe_int(row.get('PA', 0)),
            _safe_int(row.get('AB', 0)),
            _safe_int(row.get('H', 0)),
            _safe_int(row.get('2B', 0)),
            _safe_int(row.get('3B', 0)),
            _safe_int(row.get('HR', 0)),
            _safe_int(row.get('RBI', 0)),
            _safe_int(row.get('R', 0)),
            _safe_int(row.get('BB', 0)),
            _safe_int(row.get('IBB', 0)),
            _safe_int(row.get('HP', 0)),
            _safe_int(row.get('K', 0)),
            _safe_int(row.get('GIDP', 0)),
            _safe_float(row.get('AVG', 0)),
            _safe_float(row.get('OBP', 0)),
            _safe_float(row.get('SLG', 0)),
            _safe_float(row.get('ISO', 0)),
            _safe_float(row.get('OPS', 0)),
            _safe_int(row.get('OPS+', 0)),
            _safe_float(row.get('BABIP', 0)),
            _safe_float(row.get('WAR', 0)),
            _safe_int(row.get('SB', 0)),
            _safe_int(row.get('CS', 0)),
            card_id,
            snapshot_date,
            league_id,
        ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "stats_batting", "rows": count, "league_id": league_id}


def ingest_stats_pitching(filepath: str) -> dict:
    """Ingest sortable pitching stats CSV into pitching_stats table.

    Each import replaces current stats (latest snapshot) and preserves history.
    Detects stats_2 format (SIERA, WIN%) and routes to advanced handler.

    League scoping: these files are league-wide (e.g. ``lb124_statistics_...``)
    and have a non-null ``league_id``. The DELETE/INSERT are scoped by
    ``league_id`` so they don't wipe or collide with the team-file path, which
    writes ``league_id IS NULL`` rows for the user's own team stats.
    """
    df = parse_stats_pitching_csv(filepath)
    cols = set(df.columns)
    if 'SIERA' in cols or 'WIN%' in cols:
        return ingest_pitching_stats_adv(filepath)

    conn = get_connection()
    cursor = conn.cursor()

    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    league_id = detect_league_id(filepath)

    # Delete any existing snapshot from today for THIS league only.
    # Team-scoped rows (league_id IS NULL) are NOT touched.
    if league_id is None:
        cursor.execute(
            "DELETE FROM pitching_stats WHERE DATE(snapshot_date) = ? AND league_id IS NULL",
            (snapshot_date,),
        )
    else:
        cursor.execute(
            "DELETE FROM pitching_stats WHERE DATE(snapshot_date) = ? AND league_id = ?",
            (snapshot_date, league_id),
        )

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name or name.lower() in ('total', 'totals', 'team'):
            continue

        pos = _safe_str(row.get('POS', '')).strip()
        bats = _safe_str(row.get('B', '')).strip()
        throws = _safe_str(row.get('T', '')).strip()

        # League rows represent other teams' cards; do NOT prefer owned or
        # they'll all get pinned to the user's own card_id.
        card_id = _match_card_id(cursor, name, prefer_owned=False)

        cursor.execute("""
            INSERT INTO pitching_stats (
                player_name, position, bats, throws,
                games, gs, wins, losses, saves, holds,
                ip, hits_allowed, hr_allowed, runs_allowed, er,
                bb, k, hbp,
                era, avg_against, babip, whip,
                hr_per_9, bb_per_9, k_per_9, k_per_bb,
                era_plus, fip, war,
                card_id, snapshot_date, league_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, pos, bats, throws,
            _safe_int(row.get('G', 0)),
            _safe_int(row.get('GS', 0)),
            _safe_int(row.get('W', 0)),
            _safe_int(row.get('L', 0)),
            _safe_int(row.get('SV', 0)),
            _safe_int(row.get('HLD', 0)),
            _safe_float(row.get('IP', 0)),
            _safe_int(row.get('HA', 0)),
            _safe_int(row.get('HR', 0)),
            _safe_int(row.get('R', 0)),
            _safe_int(row.get('ER', 0)),
            _safe_int(row.get('BB', 0)),
            _safe_int(row.get('K', 0)),
            _safe_int(row.get('HP', 0)),
            _safe_float(row.get('ERA', 0)),
            _safe_float(row.get('AVG', 0)),
            _safe_float(row.get('BABIP', 0)),
            _safe_float(row.get('WHIP', 0)),
            _safe_float(row.get('HR/9', 0)),
            _safe_float(row.get('BB/9', 0)),
            _safe_float(row.get('K/9', 0)),
            _safe_float(row.get('K/BB', 0)),
            _safe_int(row.get('ERA+', 0)),
            _safe_float(row.get('FIP', 0)),
            _safe_float(row.get('WAR', 0)),
            card_id,
            snapshot_date,
            league_id,
        ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "stats_pitching", "rows": count, "league_id": league_id}


def ingest_roster_batting_stats(filepath: str) -> dict:
    """Ingest roster batting stats CSV (stats_1 or stats_2 format).

    stats_1 has: POS, Name, G, PA, AB, H, 2B, 3B, HR, RBI, R, BB, K, AVG, OBP, SLG, OPS, WAR...
    stats_2 has: POS, Name, G, PA, BB, BB%, K, K%, wOBA, WPA, ISO, RC/27...

    For stats_1: same columns as league-wide sortable stats — reuse batting_stats table.
    For stats_2: skip (advanced metrics, less critical for now).
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Detect format: stats_1 has 'AB' and 'AVG', stats_2 has 'BB%' and 'wOBA'
    cols = set(df.columns)
    if 'AB' in cols and 'AVG' in cols:
        # stats_1 format — same as league sortable stats
        return _ingest_batting_stats_standard(df, "roster_batting_stats")
    elif 'wOBA' in cols or 'BB%' in cols:
        # stats_2 format — advanced metrics
        return ingest_batting_stats_adv(filepath)
    else:
        return {"status": "skipped", "file_type": "roster_batting_stats", "rows": 0,
                "reason": "Unrecognized batting stats format"}


def ingest_roster_pitching_stats(filepath: str) -> dict:
    """Ingest roster pitching stats CSV (stats_1 or stats_2 format).

    stats_1 has: POS, Name, G, GS, W, L, SV, IP, ERA, WHIP, K, WAR...
    stats_2 has: POS, Name, WIN%, SV%, BS, QS, SIERA, WPA...
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    cols = set(df.columns)
    if 'ERA' in cols and 'W' in cols:
        # stats_1 format — same as league sortable stats
        return _ingest_pitching_stats_standard(df, "roster_pitching_stats")
    elif 'SIERA' in cols or 'WIN%' in cols:
        # stats_2 format — advanced metrics
        return ingest_pitching_stats_adv(filepath)
    else:
        return {"status": "skipped", "file_type": "roster_pitching_stats", "rows": 0,
                "reason": "Unrecognized pitching stats format"}


def _ingest_batting_stats_standard(df: pd.DataFrame, file_type: str) -> dict:
    """Shared logic for ingesting standard batting stats (team-scoped).

    Called for team-file paths (e.g. ``toronto_dark_knights_..._batting_stats``),
    which represent the user's OWN team and are authoritative for their
    pitchers'/batters' performance. Rows are written with ``league_id=NULL``
    and upserted keyed on ``(card_id, snapshot_date)``. The owned-preferred
    ``_match_card_id`` ensures we pin to the user's card_id even when other
    teams in the league own cards with the same player name.
    """
    conn = get_connection()
    cursor = conn.cursor()
    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name or name.lower() in ('total', 'totals', 'team'):
            continue

        pos = _safe_str(row.get('POS', '')).strip()
        bats = _safe_str(row.get('B', '')).strip()
        throws = _safe_str(row.get('T', '')).strip()
        card_id = _match_card_id(cursor, name, prefer_owned=True)

        # Upsert keyed on (card_id, date) within the team-scoped partition
        # (league_id IS NULL). Fall back to player_name if card_id is unknown.
        existing = None
        if card_id is not None:
            existing = cursor.execute(
                "SELECT id FROM batting_stats "
                "WHERE card_id = ? AND DATE(snapshot_date) = ? AND league_id IS NULL",
                (card_id, snapshot_date),
            ).fetchone()
        if existing is None:
            existing = cursor.execute(
                "SELECT id FROM batting_stats "
                "WHERE player_name = ? AND DATE(snapshot_date) = ? AND league_id IS NULL",
                (name, snapshot_date),
            ).fetchone()

        if existing:
            cursor.execute("""
                UPDATE batting_stats SET
                    position=?, games=?, pa=?, ab=?, hits=?, doubles=?, triples=?,
                    hr=?, rbi=?, runs=?, bb=?, ibb=?, hbp=?, k=?, gidp=?,
                    avg=?, obp=?, slg=?, iso=?, ops=?, ops_plus=?, babip=?, war=?,
                    sb=?, cs=?, card_id=?
                WHERE id=?
            """, (
                pos, _safe_int(row.get('G', 0)), _safe_int(row.get('PA', 0)),
                _safe_int(row.get('AB', 0)), _safe_int(row.get('H', 0)),
                _safe_int(row.get('2B', 0)), _safe_int(row.get('3B', 0)),
                _safe_int(row.get('HR', 0)), _safe_int(row.get('RBI', 0)),
                _safe_int(row.get('R', 0)), _safe_int(row.get('BB', 0)),
                _safe_int(row.get('IBB', 0)), _safe_int(row.get('HP', 0)),
                _safe_int(row.get('K', 0)), _safe_int(row.get('GIDP', 0)),
                _safe_float(row.get('AVG', 0)), _safe_float(row.get('OBP', 0)),
                _safe_float(row.get('SLG', 0)), _safe_float(row.get('ISO', 0)),
                _safe_float(row.get('OPS', 0)), _safe_int(row.get('OPS+', 0)),
                _safe_float(row.get('BABIP', 0)), _safe_float(row.get('WAR', 0)),
                _safe_int(row.get('SB', 0)), _safe_int(row.get('CS', 0)),
                card_id, existing[0],
            ))
        else:
            cursor.execute("""
                INSERT INTO batting_stats (
                    player_name, position, bats, throws,
                    games, pa, ab, hits, doubles, triples, hr, rbi, runs,
                    bb, ibb, hbp, k, gidp,
                    avg, obp, slg, iso, ops, ops_plus, babip, war,
                    sb, cs, card_id, snapshot_date, league_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """, (
                name, pos, bats, throws,
                _safe_int(row.get('G', 0)), _safe_int(row.get('PA', 0)),
                _safe_int(row.get('AB', 0)), _safe_int(row.get('H', 0)),
                _safe_int(row.get('2B', 0)), _safe_int(row.get('3B', 0)),
                _safe_int(row.get('HR', 0)), _safe_int(row.get('RBI', 0)),
                _safe_int(row.get('R', 0)), _safe_int(row.get('BB', 0)),
                _safe_int(row.get('IBB', 0)), _safe_int(row.get('HP', 0)),
                _safe_int(row.get('K', 0)), _safe_int(row.get('GIDP', 0)),
                _safe_float(row.get('AVG', 0)), _safe_float(row.get('OBP', 0)),
                _safe_float(row.get('SLG', 0)), _safe_float(row.get('ISO', 0)),
                _safe_float(row.get('OPS', 0)), _safe_int(row.get('OPS+', 0)),
                _safe_float(row.get('BABIP', 0)), _safe_float(row.get('WAR', 0)),
                _safe_int(row.get('SB', 0)), _safe_int(row.get('CS', 0)),
                card_id, snapshot_date,
            ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": file_type, "rows": count}


def _ingest_pitching_stats_standard(df: pd.DataFrame, file_type: str) -> dict:
    """Shared logic for ingesting standard pitching stats (team-scoped).

    Called for team-file paths (e.g. ``toronto_dark_knights_..._pitching_stats``),
    which represent the user's OWN team and are authoritative for their
    pitchers' performance. Rows are written with ``league_id=NULL`` and
    upserted keyed on ``(card_id, snapshot_date)``. The owned-preferred
    ``_match_card_id`` ensures we pin to the user's card_id even when other
    teams in the league own cards with the same player name.
    """
    conn = get_connection()
    cursor = conn.cursor()
    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name or name.lower() in ('total', 'totals', 'team'):
            continue

        pos = _safe_str(row.get('POS', '')).strip()
        bats = _safe_str(row.get('B', '')).strip()
        throws = _safe_str(row.get('T', '')).strip()
        card_id = _match_card_id(cursor, name, prefer_owned=True)

        # Upsert keyed on (card_id, date) within the team-scoped partition
        # (league_id IS NULL). Fall back to player_name if card_id is unknown.
        existing = None
        if card_id is not None:
            existing = cursor.execute(
                "SELECT id FROM pitching_stats "
                "WHERE card_id = ? AND DATE(snapshot_date) = ? AND league_id IS NULL",
                (card_id, snapshot_date),
            ).fetchone()
        if existing is None:
            existing = cursor.execute(
                "SELECT id FROM pitching_stats "
                "WHERE player_name = ? AND DATE(snapshot_date) = ? AND league_id IS NULL",
                (name, snapshot_date),
            ).fetchone()

        stat_vals = (
            _safe_int(row.get('G', 0)), _safe_int(row.get('GS', 0)),
            _safe_int(row.get('W', 0)), _safe_int(row.get('L', 0)),
            _safe_int(row.get('SV', 0)), _safe_int(row.get('HLD', 0)),
            _safe_float(row.get('IP', 0)), _safe_int(row.get('HA', 0)),
            _safe_int(row.get('HR', 0)), _safe_int(row.get('R', 0)),
            _safe_int(row.get('ER', 0)), _safe_int(row.get('BB', 0)),
            _safe_int(row.get('K', 0)), _safe_int(row.get('HP', 0)),
            _safe_float(row.get('ERA', 0)), _safe_float(row.get('AVG', 0)),
            _safe_float(row.get('BABIP', 0)), _safe_float(row.get('WHIP', 0)),
            _safe_float(row.get('HR/9', 0)), _safe_float(row.get('BB/9', 0)),
            _safe_float(row.get('K/9', 0)), _safe_float(row.get('K/BB', 0)),
            _safe_int(row.get('ERA+', 0)), _safe_float(row.get('FIP', 0)),
            _safe_float(row.get('WAR', 0)), card_id,
        )

        if existing:
            cursor.execute("""
                UPDATE pitching_stats SET
                    position=?,
                    games=?, gs=?, wins=?, losses=?, saves=?, holds=?,
                    ip=?, hits_allowed=?, hr_allowed=?, runs_allowed=?, er=?,
                    bb=?, k=?, hbp=?, era=?, avg_against=?, babip=?, whip=?,
                    hr_per_9=?, bb_per_9=?, k_per_9=?, k_per_bb=?,
                    era_plus=?, fip=?, war=?, card_id=?
                WHERE id=?
            """, (pos,) + stat_vals + (existing[0],))
        else:
            cursor.execute("""
                INSERT INTO pitching_stats (
                    player_name, position, bats, throws,
                    games, gs, wins, losses, saves, holds,
                    ip, hits_allowed, hr_allowed, runs_allowed, er,
                    bb, k, hbp,
                    era, avg_against, babip, whip,
                    hr_per_9, bb_per_9, k_per_9, k_per_bb,
                    era_plus, fip, war,
                    card_id, snapshot_date, league_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """, (name, pos, bats, throws) + stat_vals + (snapshot_date,))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": file_type, "rows": count}


# ─── New handlers for additional file types ───────────────────────────


def ingest_league_batting_ratings(filepath: str) -> dict:
    """Ingest league-wide batting ratings into roster table as 'league' entries.

    These contain split ratings (CON vL/vR, POW vL/vR, EYE vL/vR) for every
    player in the league — useful for market analysis and platoon comparisons.
    We store them in the roster table with lineup_role='league'.
    """
    from app.core.meta_scoring import calc_batting_meta, calc_batting_meta_vs_rhp, calc_batting_meta_vs_lhp
    df = parse_league_batting_ratings_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    # Clear previous league batting entries
    cursor.execute("DELETE FROM roster WHERE lineup_role = 'league' AND position NOT IN ('SP', 'RP', 'CL') AND DATE(snapshot_date) = DATE('now')")

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name:
            continue
        pos = _safe_str(row.get('POS', '')).strip()
        ovr = _safe_int(row.get('OVR', 0))

        card_dict = {
            'contact': _safe_int(row.get('CON', 0)),
            'gap_power': _safe_int(row.get('GAP', 0)),
            'avoid_ks': _safe_int(row.get("K's", 0)),
            'eye': _safe_int(row.get('EYE', 0)),
            'power': _safe_int(row.get('POW', 0)),
            'babip': _safe_int(row.get('BABIP', 0)),
            'defense_score': _safe_float(row.get('DEF', 0)),
            'card_value': ovr,
            'con_vl': _safe_int(row.get('CON vL', 0)),
            'pow_vl': _safe_int(row.get('POW vL', 0)),
            'eye_vl': _safe_int(row.get('EYE vL', 0)),
            'con_vr': _safe_int(row.get('CON vR', 0)),
            'pow_vr': _safe_int(row.get('POW vR', 0)),
            'eye_vr': _safe_int(row.get('EYE vR', 0)),
        }
        meta = calc_batting_meta(card_dict)
        meta_rhp = calc_batting_meta_vs_rhp(card_dict)
        meta_lhp = calc_batting_meta_vs_lhp(card_dict)

        card_id = _match_card_id(cursor, name)

        cursor.execute("""
            INSERT OR IGNORE INTO roster (player_name, position, lineup_role, ovr, meta_score,
                meta_vs_rhp, meta_vs_lhp, con_vl, pow_vl, eye_vl, con_vr, pow_vr, eye_vr,
                card_id, bats)
            VALUES (?, ?, 'league', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, pos, ovr, meta, meta_rhp, meta_lhp,
            _safe_int(row.get('CON vL', 0)), _safe_int(row.get('POW vL', 0)),
            _safe_int(row.get('EYE vL', 0)), _safe_int(row.get('CON vR', 0)),
            _safe_int(row.get('POW vR', 0)), _safe_int(row.get('EYE vR', 0)),
            card_id, _safe_str(row.get('B', '')).strip(),
        ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "stats_batting_ratings", "rows": count}


def ingest_league_pitching_ratings(filepath: str) -> dict:
    """Ingest league-wide pitching ratings into roster table as 'league' entries."""
    from app.core.meta_scoring import calc_pitching_meta, calc_pitching_meta_vs_lhb, calc_pitching_meta_vs_rhb
    df = parse_league_pitching_ratings_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM roster WHERE lineup_role = 'league' AND position IN ('SP', 'RP', 'CL') AND DATE(snapshot_date) = DATE('now')")

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name:
            continue
        pos = _safe_str(row.get('POS', '')).strip()
        ovr = _safe_int(row.get('OVR', 0))

        card_dict = {
            'stuff': _safe_int(row.get('STU', 0)),
            'movement': _safe_int(row.get('MOV', 0)),
            'control': _safe_int(row.get('CON', 0)),
            'p_hr': _safe_int(row.get('HRA', 0)),
            'card_value': ovr,
            'stamina': _safe_int(row.get('STM', 0)),
            'hold': _safe_int(row.get('HLD', 0)),
            'stu_vl': _safe_int(row.get('STU vL', 0)),
            'stu_vr': _safe_int(row.get('STU vR', 0)),
        }
        meta = calc_pitching_meta(card_dict)
        meta_vs_lhb = calc_pitching_meta_vs_lhb(card_dict)
        meta_vs_rhb = calc_pitching_meta_vs_rhb(card_dict)

        card_id = _match_card_id(cursor, name)

        cursor.execute("""
            INSERT OR IGNORE INTO roster (player_name, position, lineup_role, ovr, meta_score,
                meta_vs_rhp, meta_vs_lhp, stu_vl, stu_vr, card_id)
            VALUES (?, ?, 'league', ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, pos, ovr, meta, meta_vs_rhb, meta_vs_lhb,
            _safe_int(row.get('STU vL', 0)), _safe_int(row.get('STU vR', 0)),
            card_id,
        ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "stats_pitching_ratings", "rows": count}


def ingest_league_default(filepath: str) -> dict:
    """Populate `league_rosters` from the league-wide player `_default.csv`.

    This is the ONLY league-wide export that carries the TM (short team name)
    and LG columns, so it's the only way to attribute a card to the PT team
    that owns it. Row-aligned with the sibling `_batting_ratings` and
    `_pitching_ratings` files — we read those too and pick the appropriate
    meta formula based on POS.

    Team short_name ("Sassy") is translated to full_name ("Sassy Kitties") via
    `team_aliases` so joins against game_batting.team_name succeed. If no
    alias exists the short_name is stored as-is and Phase 0 re-seeding can
    backfill it later.
    """
    from app.core.meta_scoring import (
        calc_batting_meta, calc_pitching_meta,
        calc_pitching_meta_vs_lhb, calc_pitching_meta_vs_rhb,
    )
    from app.core.team_aliases import resolve_full_name

    league_id = detect_league_id(filepath)
    if not league_id:
        return {"status": "skipped", "file_type": "league_player_default",
                "rows": 0, "reason": "could not detect league_id from filename"}

    # Find companion ratings files in the same directory. OOTP exports them
    # atomically, so they always coexist — but handle a missing companion
    # gracefully rather than crashing the whole refresh.
    base = Path(filepath)
    dir_ = base.parent
    stem = base.name.replace("_default.csv", "")
    bat_path = dir_ / f"{stem}_batting_ratings.csv"
    pit_path = dir_ / f"{stem}_pitching_ratings.csv"
    if not bat_path.exists() or not pit_path.exists():
        return {"status": "skipped", "file_type": "league_player_default",
                "rows": 0,
                "reason": f"missing companion file(s): "
                          f"{bat_path.name if not bat_path.exists() else ''} "
                          f"{pit_path.name if not pit_path.exists() else ''}".strip()}

    df_def = parse_league_player_default_csv(filepath)
    df_bat = parse_league_batting_ratings_csv(str(bat_path))
    df_pit = parse_league_pitching_ratings_csv(str(pit_path))

    # Row-alignment sanity check — if OOTP ever changes the sort order these
    # will diverge. Rather than fail hard we count mismatches and fall back
    # to per-row name matching within the companion frames.
    n = min(len(df_def), len(df_bat), len(df_pit))
    mismatches = 0

    conn = get_connection()
    cursor = conn.cursor()

    # Clear today's rows for this league so the ingest is idempotent.
    cursor.execute(
        "DELETE FROM league_rosters WHERE league_id = ? AND DATE(snapshot_date) = DATE('now')",
        (league_id,),
    )

    inserted = 0
    for i in range(n):
        dr = df_def.iloc[i]
        br = df_bat.iloc[i]
        pr = df_pit.iloc[i]

        name = _safe_str(dr.get("Name", "")).strip()
        if not name:
            continue

        # Detect drift in row alignment — don't silently mix rows.
        if (_safe_str(br.get("Name", "")).strip() != name or
                _safe_str(pr.get("Name", "")).strip() != name):
            mismatches += 1
            continue

        short_tm = _safe_str(dr.get("TM", "")).strip()
        if not short_tm or short_tm == "-":
            # League's free-agent pool / unassigned — no team attribution.
            continue
        full_tm = resolve_full_name(short_tm, league_id, conn=conn) or short_tm

        pos = _safe_str(dr.get("POS", "")).strip()
        ovr = _safe_int(dr.get("OVR", 0))
        is_pitcher = pos in ("SP", "RP", "CL")

        # Meta score: pitching formula for pitchers, batting for everyone else.
        if is_pitcher:
            card_dict = {
                "stuff": _safe_int(pr.get("STU", 0)),
                "movement": _safe_int(pr.get("MOV", 0)),
                "control": _safe_int(pr.get("CON", 0)),
                "p_hr": _safe_int(pr.get("HRA", 0)),
                "card_value": ovr,
                "stamina": _safe_int(pr.get("STM", 0)),
                "hold": _safe_int(pr.get("HLD", 0)),
                "stu_vl": _safe_int(pr.get("STU vL", 0)),
                "stu_vr": _safe_int(pr.get("STU vR", 0)),
            }
            meta = calc_pitching_meta(card_dict)
            contact = gap_power = power = eye = None
            stuff = card_dict["stuff"]
            movement = card_dict["movement"]
            control = card_dict["control"]
            p_hr = card_dict["p_hr"]
        else:
            card_dict = {
                "contact": _safe_int(br.get("CON", 0)),
                "gap_power": _safe_int(br.get("GAP", 0)),
                "avoid_ks": _safe_int(br.get("K's", 0)),
                "eye": _safe_int(br.get("EYE", 0)),
                "power": _safe_int(br.get("POW", 0)),
                "babip": _safe_int(br.get("BABIP", 0)),
                "defense_score": _safe_float(br.get("DEF", 0)),
                "card_value": ovr,
                "con_vl": _safe_int(br.get("CON vL", 0)),
                "pow_vl": _safe_int(br.get("POW vL", 0)),
                "eye_vl": _safe_int(br.get("EYE vL", 0)),
                "con_vr": _safe_int(br.get("CON vR", 0)),
                "pow_vr": _safe_int(br.get("POW vR", 0)),
                "eye_vr": _safe_int(br.get("EYE vR", 0)),
            }
            meta = calc_batting_meta(card_dict)
            contact = card_dict["contact"]
            gap_power = card_dict["gap_power"]
            power = card_dict["power"]
            eye = card_dict["eye"]
            stuff = movement = control = p_hr = None

        # Prefer an owned card match when one exists (user's own Campanella
        # card_id will outrank other cards if multiple exist for the name);
        # otherwise fall back to any card.
        card_id = _match_card_id(cursor, name, prefer_owned=False)

        cursor.execute("""
            INSERT INTO league_rosters (league_id, team_name, player_name, card_id,
                position, ovr, meta_score,
                contact, gap_power, power, eye,
                stuff, movement, control, p_hr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            league_id, full_tm, name, card_id,
            pos, ovr, meta,
            contact, gap_power, power, eye,
            stuff, movement, control, p_hr,
        ))
        inserted += 1

    conn.commit()
    conn.close()

    # New league_rosters rows mean the team-scoped name resolver cache is
    # stale. Refresh so the next box-score ingest sees the latest
    # (team, initial, lastname) -> card_id mappings.
    try:
        from app.core.name_resolver import refresh_team_cache
        refresh_team_cache()
    except Exception as _e:
        logger.debug("refresh_team_cache post-ingest failed: %s", _e)

    return {
        "status": "success",
        "file_type": "league_player_default",
        "rows": inserted,
        "league_id": league_id,
        "source_rows": n,
        "row_mismatches": mismatches,
    }


def ingest_lineup(filepath: str, file_type: str = None) -> dict:
    """Ingest team lineup CSV. The Title column enables exact card matching.

    file_type is one of: lineup_vs_rhp, lineup_vs_lhp, lineup_overview
    """
    df = parse_lineup_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    # Determine lineup type from file_type or filename
    if file_type is None:
        fname = Path(filepath).name.lower()
        if 'vs_rhp' in fname:
            file_type = 'lineup_vs_rhp'
        elif 'vs_lhp' in fname:
            file_type = 'lineup_vs_lhp'
        else:
            file_type = 'lineup_overview'

    lineup_type_map = {
        'lineup_vs_rhp': 'vs_rhp',
        'lineup_vs_lhp': 'vs_lhp',
        'lineup_overview': 'overview',
    }
    lineup_type = lineup_type_map.get(file_type, 'overview')

    # Clear previous entries for this lineup type
    cursor.execute("DELETE FROM team_lineup WHERE lineup_type = ? AND DATE(snapshot_date) = DATE('now')", (lineup_type,))

    count = 0
    for idx, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name:
            continue
        pos = _safe_str(row.get('POS', '')).strip()
        title = _safe_str(row.get('Title', '')).strip()
        ovr = _safe_int(row.get('OVR', 0))
        bats = _safe_str(row.get('B', '')).strip()
        throws = _safe_str(row.get('T', '')).strip()
        age = _safe_int(row.get('Age', 0))

        # Use Title for exact card matching (much more reliable than name)
        card_id = None
        if title:
            card_row = cursor.execute(
                "SELECT card_id FROM cards WHERE card_title = ? LIMIT 1", (title,)
            ).fetchone()
            card_id = card_row[0] if card_row else None

        # Fall back to name matching
        if card_id is None:
            card_id = _match_card_id(cursor, name)

        # Also update the roster table with card_id and card_title if we have a match.
        # Prefer card_id match (safer — no name mismatches) and fall back to
        # player_name match when card_id wasn't yet resolved in roster.
        if card_id and title:
            cursor.execute("""
                UPDATE roster SET card_id = ?, card_title = ?
                WHERE (card_id = ? OR (card_id IS NULL AND player_name = ?))
                  AND lineup_role != 'league'
            """, (card_id, title, card_id, name))

        cursor.execute("""
            INSERT INTO team_lineup (lineup_type, position, player_name, card_title,
                card_id, ovr, bats, throws, age, slot_order)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (lineup_type, pos, name, title, card_id, ovr, bats, throws, age, idx))
        count += 1

    # ── Sync roster.lineup_role: keep one starter per batting position ──
    # Roster ingestion marks every active-roster non-reserve player as
    # 'starter' because the roster CSV has no "in lineup?" signal. The
    # lineup CSV ALSO doesn't help: OOTP's "Lineups → Overview" export
    # dumps the full depth chart (e.g. both Kluttz AND Posey as C), not
    # the slot assignments. So we fall back to a heuristic: at each
    # batting position, the highest-META active-roster player is the
    # starter; everyone else at the same position becomes 'bench'.
    #
    # Meta (not OVR) as the tiebreaker because meta is calibrated to
    # predict WAR, which is what OOTP users actually optimize when they
    # pick a starter. OVR can disagree — e.g. the user has Jonathan Aranda
    # (OVR 87, meta 636) as a 1B backup and Rafael Palmeiro (OVR 77, meta
    # 711) as the actual starter. OVR would mis-pick Aranda; meta picks
    # Palmeiro correctly. Verified on Posey/Kluttz (C) and Soderstrom/
    # Bellinger (LF) too.
    #
    # Scoped to the LATEST roster snapshot so older snapshots aren't
    # modified. Pitching roles (SP/RP/CL/rotation/bullpen/closer) are
    # left alone — they have their own depth signals from the team
    # pitching staff export.
    latest_snap = cursor.execute(
        "SELECT MAX(DATE(snapshot_date)) FROM roster WHERE lineup_role != 'league'"
    ).fetchone()[0]

    demoted = 0
    if latest_snap:
        demoted = cursor.execute("""
            UPDATE roster SET lineup_role = 'bench'
            WHERE lineup_role = 'starter'
              AND position NOT IN ('SP', 'RP', 'CL')
              AND DATE(snapshot_date) = ?
              AND rowid NOT IN (
                  SELECT rowid FROM (
                      SELECT rowid,
                             ROW_NUMBER() OVER (
                                 PARTITION BY position
                                 ORDER BY COALESCE(meta_score, 0) DESC,
                                          ovr DESC, rowid DESC
                             ) as rn
                      FROM roster
                      WHERE lineup_role = 'starter'
                        AND position NOT IN ('SP', 'RP', 'CL')
                        AND DATE(snapshot_date) = ?
                  ) WHERE rn = 1
              )
        """, (latest_snap, latest_snap)).rowcount

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": file_type, "rows": count,
            "roster_demoted": demoted, "roster_promoted": 0}


def ingest_team_pitching(filepath: str) -> dict:
    """Ingest team pitching roster CSV. Shows full staff with order and Title."""
    df = parse_team_pitching_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    # Store in team_lineup as 'pitching' type
    cursor.execute("DELETE FROM team_lineup WHERE lineup_type = 'pitching' AND DATE(snapshot_date) = DATE('now')")

    count = 0
    for idx, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name:
            continue
        pos = _safe_str(row.get('POS', '')).strip()
        title = _safe_str(row.get('Title', '')).strip()
        ovr = _safe_int(row.get('OVR', 0))
        bats = _safe_str(row.get('B', '')).strip()
        throws = _safe_str(row.get('T', '')).strip()
        age = _safe_int(row.get('Age', 0))

        card_id = None
        if title:
            card_row = cursor.execute(
                "SELECT card_id FROM cards WHERE card_title = ? LIMIT 1", (title,)
            ).fetchone()
            card_id = card_row[0] if card_row else None
        if card_id is None:
            card_id = _match_card_id(cursor, name)

        if card_id and title:
            cursor.execute("""
                UPDATE roster SET card_id = ?, card_title = ?
                WHERE (card_id = ? OR (card_id IS NULL AND player_name = ?))
                  AND lineup_role != 'league'
            """, (card_id, title, card_id, name))

        cursor.execute("""
            INSERT INTO team_lineup (lineup_type, position, player_name, card_title,
                card_id, ovr, bats, throws, age, slot_order)
            VALUES ('pitching', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (pos, name, title, card_id, ovr, bats, throws, age, idx))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "team_pitching", "rows": count}


def ingest_batting_stats_adv(filepath: str) -> dict:
    """Ingest batting stats_2 CSV (advanced metrics: wOBA, BB%, K%, WPA, etc.).

    League scoping: league-wide CSVs (``lb124_...``) and team CSVs
    (``toronto_dark_knights_...``) both hit this function. Before the
    2026-04-15 fix the DELETE was unscoped, so whichever file ran second
    wiped the other. Now the DELETE is scoped by ``league_id`` so league
    and team partitions coexist.
    """
    df = parse_stats_batting_adv_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()
    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    league_id = detect_league_id(filepath)

    if league_id is None:
        cursor.execute(
            "DELETE FROM batting_stats_adv WHERE DATE(snapshot_date) = ? AND league_id IS NULL",
            (snapshot_date,),
        )
    else:
        cursor.execute(
            "DELETE FROM batting_stats_adv WHERE DATE(snapshot_date) = ? AND league_id = ?",
            (snapshot_date, league_id),
        )

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name or name.lower() in ('total', 'totals', 'team'):
            continue
        # League rows represent other teams' cards; don't prefer owned
        # or they'll all get pinned to the user's own card_id.
        card_id = _match_card_id(cursor, name, prefer_owned=(league_id is None))

        cursor.execute("""
            INSERT INTO batting_stats_adv (
                player_name, position, bats, throws,
                games, pa, bb, bb_pct, sh, sf, ci, k, k_pct, gidp,
                ebh, tb, rc, rc27, iso, woba, wpa, pi_pa,
                card_id, snapshot_date, league_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            _safe_str(row.get('POS', '')).strip(),
            _safe_str(row.get('B', '')).strip(),
            _safe_str(row.get('T', '')).strip(),
            _safe_int(row.get('G', 0)),
            _safe_int(row.get('PA', 0)),
            _safe_int(row.get('BB', 0)),
            _safe_float(row.get('BB%', 0)),
            _safe_int(row.get('SH', 0)),
            _safe_int(row.get('SF', 0)),
            _safe_int(row.get('CI', 0)),
            _safe_int(row.get('K', 0)),
            _safe_float(row.get('K%', 0)),
            _safe_int(row.get('GIDP', 0)),
            _safe_int(row.get('EBH', 0)),
            _safe_int(row.get('TB', 0)),
            _safe_float(row.get('RC', 0)),
            _safe_float(row.get('RC/27', 0)),
            _safe_float(row.get('ISO', 0)),
            _safe_float(row.get('wOBA', 0)),
            _safe_float(row.get('WPA', 0)),
            _safe_float(row.get('PI/PA', 0)),
            card_id, snapshot_date, league_id,
        ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "batting_stats_adv", "rows": count, "league_id": league_id}


def ingest_pitching_stats_adv(filepath: str) -> dict:
    """Ingest pitching stats_2 CSV (advanced metrics: SIERA, WPA, QS%, etc.).

    League scoping: same story as ``ingest_batting_stats_adv`` — league
    and team CSVs both land here, and before the 2026-04-15 fix the
    DELETE was unscoped so whichever ran second wiped the other.
    """
    df = parse_stats_pitching_adv_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()
    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    league_id = detect_league_id(filepath)

    if league_id is None:
        cursor.execute(
            "DELETE FROM pitching_stats_adv WHERE DATE(snapshot_date) = ? AND league_id IS NULL",
            (snapshot_date,),
        )
    else:
        cursor.execute(
            "DELETE FROM pitching_stats_adv WHERE DATE(snapshot_date) = ? AND league_id = ?",
            (snapshot_date, league_id),
        )

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name or name.lower() in ('total', 'totals', 'team'):
            continue
        card_id = _match_card_id(cursor, name, prefer_owned=(league_id is None))

        cursor.execute("""
            INSERT INTO pitching_stats_adv (
                player_name, position, bats, throws,
                games, win_pct, sv_pct, bs, sd, md, ip, bf, dp, ra, gf,
                ir, irs_pct, pli, qs, qs_pct, cg, cg_pct, sho,
                ppg, rsg, go_pct, siera, sb, cs, wpa,
                card_id, snapshot_date, league_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            _safe_str(row.get('POS', '')).strip(),
            _safe_str(row.get('B', '')).strip(),
            _safe_str(row.get('T', '')).strip(),
            _safe_int(row.get('G', 0)),
            _safe_float(row.get('WIN%', 0)),
            _safe_float(row.get('SV%', 0)),
            _safe_int(row.get('BS', 0)),
            _safe_int(row.get('SD', 0)),
            _safe_int(row.get('MD', 0)),
            _safe_float(row.get('IP', 0)),
            _safe_int(row.get('BF', 0)),
            _safe_int(row.get('DP', 0)),
            _safe_int(row.get('RA', 0)),
            _safe_int(row.get('GF', 0)),
            _safe_int(row.get('IR', 0)),
            _safe_float(row.get('IRS%', 0)),
            _safe_float(row.get('pLi', 0)),
            _safe_int(row.get('QS', 0)),
            _safe_float(row.get('QS%', 0)),
            _safe_int(row.get('CG', 0)),
            _safe_float(row.get('CG%', 0)),
            _safe_int(row.get('SHO', 0)),
            _safe_int(row.get('PPG', 0)),
            _safe_float(row.get('RSG', 0)),
            _safe_float(row.get('GO%', 0)),
            _safe_float(row.get('SIERA', 0)),
            _safe_int(row.get('SB', 0)),
            _safe_int(row.get('CS', 0)),
            _safe_float(row.get('WPA', 0)),
            card_id, snapshot_date, league_id,
        ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "pitching_stats_adv", "rows": count, "league_id": league_id}


def recalculate_all_meta_scores() -> dict:
    """Recalculate meta scores for ALL cards using current weights.

    Call this after calibrating weights so every card's meta_score_batting /
    meta_score_pitching reflects the updated formula — without re-importing.
    """
    from app.core.meta_scoring import (
        calc_batting_meta, calc_pitching_meta, calc_defense_score,
        calc_speed_score, get_weights_with_source,
    )

    conn = get_connection()
    cursor = conn.cursor()

    _, _, source = get_weights_with_source()

    # ── Batch-fetch supplementary data for the multi-factor meta layers ──
    # These lookups feed platoon splits, position flexibility, arsenal
    # diversity, and performance overlay into the scoring functions.

    # Advanced batting stats (performance overlay: wOBA)
    adv_batting = {}
    try:
        for r in cursor.execute("""
            SELECT ba.card_id, ba.woba, ba.wpa, ba.pa
            FROM batting_stats_adv ba
            INNER JOIN (
                SELECT card_id, MAX(snapshot_date) AS max_date
                FROM batting_stats_adv WHERE card_id IS NOT NULL
                GROUP BY card_id
            ) latest ON ba.card_id = latest.card_id
                     AND ba.snapshot_date = latest.max_date
            WHERE ba.card_id IS NOT NULL
        """).fetchall():
            adv_batting[r['card_id']] = dict(r)
    except Exception as _e:
        logger.warning("adv_batting fetch failed (wOBA overlay disabled): %s", _e)

    # Observed BABIP per card_id — PA-weighted, league-adjusted delta.
    # Residual r=+0.26 after other overlays — ball-in-play quality signal.
    obs_babip_batting = {}
    try:
        league_babip_baseline = {}
        for r in cursor.execute("""
            SELECT league_id AS lg, AVG(babip) AS mean_bab, COUNT(*) AS n
            FROM batting_stats
            WHERE babip IS NOT NULL AND pa >= 150 AND card_id IS NOT NULL
            GROUP BY league_id HAVING COUNT(*) >= 30
        """).fetchall():
            league_babip_baseline[r['lg']] = r['mean_bab']
        for r in cursor.execute("""
            SELECT card_id, league_id AS lg, babip, pa
            FROM batting_stats
            WHERE card_id IS NOT NULL AND pa IS NOT NULL AND pa > 0
              AND babip IS NOT NULL
        """).fetchall():
            baseline = league_babip_baseline.get(r['lg'])
            if baseline is None:
                continue
            delta = float(r['babip']) - float(baseline)
            pa = r['pa'] or 0
            entry = obs_babip_batting.setdefault(r['card_id'],
                                                  {'dw': 0.0, 'pa': 0})
            entry['dw'] += delta * pa
            entry['pa'] += pa
        for cid, entry in obs_babip_batting.items():
            entry['babip_delta'] = (entry['dw'] / entry['pa']) if entry['pa'] else 0.0
    except Exception as _e:
        logger.warning("obs_babip_batting fetch failed (BABIP overlay disabled): %s", _e)

    # Observed OBP per card_id — PA-weighted, league-adjusted delta.
    # Residual r=+0.337 with WAR/600 after OPS+ overlay — captures walk/HBP
    # signal OPS+ compresses via SLG combination. Feeds _calc_obp_overlay_batting.
    obs_obp_batting = {}
    try:
        # Per-league OBP baselines
        league_obp_baseline = {}
        for r in cursor.execute("""
            SELECT league_id AS lg, AVG(obp) AS mean_obp, COUNT(*) AS n
            FROM batting_stats
            WHERE obp IS NOT NULL AND pa >= 150 AND card_id IS NOT NULL
            GROUP BY league_id HAVING COUNT(*) >= 30
        """).fetchall():
            league_obp_baseline[r['lg']] = r['mean_obp']

        for r in cursor.execute("""
            SELECT card_id, league_id AS lg, obp, pa
            FROM batting_stats
            WHERE card_id IS NOT NULL AND pa IS NOT NULL AND pa > 0
              AND obp IS NOT NULL
        """).fetchall():
            baseline = league_obp_baseline.get(r['lg'])
            if baseline is None:
                continue
            delta = float(r['obp']) - float(baseline)
            pa = r['pa'] or 0
            entry = obs_obp_batting.setdefault(r['card_id'],
                                                {'dw': 0.0, 'pa': 0})
            entry['dw'] += delta * pa
            entry['pa'] += pa
        for cid, entry in obs_obp_batting.items():
            entry['obp_delta'] = (entry['dw'] / entry['pa']) if entry['pa'] else 0.0
    except Exception as _e:
        logger.warning("obs_obp_batting fetch failed (OBP overlay disabled): %s", _e)

    # Observed OPS+ per card_id (PA-weighted across leagues). OPS+ is
    # park- and league-adjusted (100 = league avg), so pooling across
    # leagues is safe. Residual r=+0.325 with WAR/600 — strongest single
    # untapped signal. Feeds _calc_opsplus_overlay_batting.
    obs_opsplus_batting = {}
    try:
        for r in cursor.execute("""
            SELECT card_id,
                   SUM(COALESCE(ops_plus * pa, 0)) * 1.0 / NULLIF(SUM(pa), 0) AS opp_wt,
                   SUM(pa) AS pa_total
            FROM batting_stats
            WHERE card_id IS NOT NULL AND pa IS NOT NULL AND pa > 0
              AND ops_plus IS NOT NULL
            GROUP BY card_id
        """).fetchall():
            obs_opsplus_batting[r['card_id']] = {
                'ops_plus': r['opp_wt'], 'pa': r['pa_total'] or 0,
            }
    except Exception as _e:
        logger.warning("obs_opsplus_batting fetch failed (OPS+ overlay disabled): %s", _e)

    # Per-card OPS+ overperformance: observed OPS+ minus rating-predicted OPS+.
    # Fit a simple OLS line on (rating_composite → ops_plus) using the full
    # population of batters with PA>=150, then compute residual per card.
    # Meta-correlation scan showed this residual has r=+0.551 vs meta
    # residuals — the strongest untapped overperformance signal.
    overperf_batting = {}
    try:
        # Grab rating composite + observed OPS+ for all stable-sample batters.
        train_rows = cursor.execute("""
            SELECT c.card_id,
                   (COALESCE(c.contact,0) + COALESCE(c.power,0)
                    + COALESCE(c.gap_power,0)) / 3.0 AS comp,
                   SUM(COALESCE(bs.ops_plus * bs.pa, 0)) * 1.0
                     / NULLIF(SUM(bs.pa), 0) AS obs_op,
                   SUM(bs.pa) AS pa_total
            FROM cards c
            INNER JOIN batting_stats bs ON bs.card_id = c.card_id
            WHERE c.position IS NOT NULL AND c.pitcher_role IS NULL
              AND bs.pa IS NOT NULL AND bs.pa > 0 AND bs.ops_plus IS NOT NULL
              AND c.card_value > 0
            GROUP BY c.card_id
            HAVING SUM(bs.pa) >= 150
        """).fetchall()
        # OLS: obs_op = a + b * comp
        xs = [float(r['comp'] or 0) for r in train_rows]
        ys = [float(r['obs_op'] or 0) for r in train_rows]
        n = len(xs)
        if n >= 30:
            mx = sum(xs)/n; my = sum(ys)/n
            num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
            den = sum((x-mx)**2 for x in xs) or 1.0
            b = num / den
            a = my - b * mx
            # Now compute per-card overperformance on all cards (not just train)
            for r in cursor.execute("""
                SELECT c.card_id,
                       (COALESCE(c.contact,0) + COALESCE(c.power,0)
                        + COALESCE(c.gap_power,0)) / 3.0 AS comp,
                       SUM(COALESCE(bs.ops_plus * bs.pa, 0)) * 1.0
                         / NULLIF(SUM(bs.pa), 0) AS obs_op,
                       SUM(bs.pa) AS pa_total
                FROM cards c
                INNER JOIN batting_stats bs ON bs.card_id = c.card_id
                WHERE c.position IS NOT NULL AND c.pitcher_role IS NULL
                  AND bs.pa IS NOT NULL AND bs.pa > 0 AND bs.ops_plus IS NOT NULL
                GROUP BY c.card_id
            """).fetchall():
                predicted = a + b * float(r['comp'] or 0)
                over = float(r['obs_op'] or 0) - predicted
                overperf_batting[r['card_id']] = {
                    'over': over, 'pa': r['pa_total'] or 0,
                }
    except Exception as _e:
        logger.warning("overperf_batting fit failed (overperformance overlay disabled): %s", _e)

    # Observed ISO per card_id — league-adjusted to avoid cross-league
    # scoring-environment bias. We compute a delta vs the card's own
    # tier+league mean in each league, then weight by PA. This way i76
    # (higher-scoring) cards don't get inflated bonuses from a pooled
    # baseline. Feeds _calc_iso_overlay_batting (r=+0.580 w/ WAR/600).
    obs_iso_batting = {}
    try:
        # Per-(league, tier) mean ISO as baselines
        tier_league_iso = {}
        for r in cursor.execute("""
            SELECT bs.league_id AS lg, c.tier AS tier,
                   AVG(bs.iso) AS iso_mean, COUNT(*) AS n
            FROM batting_stats bs
            INNER JOIN cards c ON c.card_id = bs.card_id
            WHERE bs.pa >= 150 AND bs.iso IS NOT NULL
              AND bs.card_id IS NOT NULL AND c.card_value > 0
            GROUP BY bs.league_id, c.tier
            HAVING COUNT(*) >= 20
        """).fetchall():
            tier_league_iso[(r['lg'], r['tier'])] = r['iso_mean']

        # For each card, compute PA-weighted delta over all leagues
        for r in cursor.execute("""
            SELECT bs.card_id, bs.league_id AS lg, c.tier AS tier,
                   bs.iso AS iso, bs.pa AS pa
            FROM batting_stats bs
            INNER JOIN cards c ON c.card_id = bs.card_id
            WHERE bs.card_id IS NOT NULL AND bs.pa IS NOT NULL AND bs.pa > 0
              AND bs.iso IS NOT NULL
        """).fetchall():
            cid = r['card_id']
            baseline = tier_league_iso.get((r['lg'], r['tier']))
            if baseline is None:
                continue  # no stable baseline for this league+tier
            delta = (r['iso'] or 0) - baseline
            pa = r['pa'] or 0
            entry = obs_iso_batting.setdefault(cid, {'delta_wt': 0.0, 'pa': 0})
            entry['delta_wt'] += delta * pa
            entry['pa'] += pa
        # Finalize: PA-weighted mean delta
        for cid, entry in obs_iso_batting.items():
            if entry['pa'] > 0:
                entry['iso_delta'] = entry['delta_wt'] / entry['pa']
            else:
                entry['iso_delta'] = 0.0
    except Exception as _e:
        logger.warning("obs_iso_batting fit failed (ISO overlay disabled): %s", _e)

    # 2026-04-20: ISO gap vs power rating (iso_gap overlay).
    # Residual analysis (n=1903 PA>=250) found that observed ISO, residualized
    # on POWER RATING (not tier-peer baseline), carries r=+0.466 with meta
    # residual — the single biggest untapped signal AFTER the current
    # overlays. Semantically: "does the card under- or over-deliver what
    # its power rating implies?" Feeds _calc_iso_gap_overlay_batting.
    #
    # Distinct from obs_iso_batting (tier-peer delta). Both overlays can
    # fire on the same card without double-counting — the tier-delta
    # captures "slugs better than peers with same tier" and iso-gap
    # captures "slugs better than its power rating predicts." Correlation
    # between the two signals is ~0.4 on our sample.
    obs_iso_gap_batting = {}
    try:
        # Per-card aggregates (PA-weighted obs ISO + power rating)
        train_rows = cursor.execute("""
            SELECT c.card_id,
                   COALESCE(c.power, 0) AS power,
                   SUM(COALESCE(bs.iso * bs.pa, 0)) * 1.0
                     / NULLIF(SUM(bs.pa), 0) AS obs_iso,
                   SUM(bs.pa) AS pa_total
            FROM cards c
            INNER JOIN batting_stats bs ON bs.card_id = c.card_id
            WHERE c.position IS NOT NULL AND c.pitcher_role IS NULL
              AND bs.pa IS NOT NULL AND bs.pa > 0 AND bs.iso IS NOT NULL
              AND c.power IS NOT NULL AND c.card_value > 0
            GROUP BY c.card_id
            HAVING SUM(bs.pa) >= 150
        """).fetchall()
        xs = [float(r['power'] or 0) for r in train_rows]
        ys = [float(r['obs_iso'] or 0) for r in train_rows]
        n = len(xs)
        if n >= 30:
            mx = sum(xs)/n; my = sum(ys)/n
            num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
            den = sum((x-mx)**2 for x in xs) or 1.0
            slope = num / den
            intercept = my - slope * mx
            # Compute gap for every card with obs ISO, not just the train set
            for r in cursor.execute("""
                SELECT c.card_id, COALESCE(c.power, 0) AS power,
                       SUM(COALESCE(bs.iso * bs.pa, 0)) * 1.0
                         / NULLIF(SUM(bs.pa), 0) AS obs_iso,
                       SUM(bs.pa) AS pa_total
                FROM cards c
                INNER JOIN batting_stats bs ON bs.card_id = c.card_id
                WHERE c.position IS NOT NULL AND c.pitcher_role IS NULL
                  AND bs.pa IS NOT NULL AND bs.pa > 0 AND bs.iso IS NOT NULL
                  AND c.power IS NOT NULL
                GROUP BY c.card_id
            """).fetchall():
                pred = intercept + slope * float(r['power'] or 0)
                gap = float(r['obs_iso'] or 0) - pred
                obs_iso_gap_batting[r['card_id']] = {
                    'gap': gap, 'pa': r['pa_total'] or 0,
                }
    except Exception as _e:
        logger.warning("obs_iso_gap_batting fit failed (ISO-gap overlay disabled): %s", _e)

    # Advanced pitching stats (performance overlay: SIERA)
    adv_pitching = {}
    try:
        for r in cursor.execute("""
            SELECT pa.card_id, pa.siera, pa.wpa, pa.ip
            FROM pitching_stats_adv pa
            INNER JOIN (
                SELECT card_id, MAX(snapshot_date) AS max_date
                FROM pitching_stats_adv WHERE card_id IS NOT NULL
                GROUP BY card_id
            ) latest ON pa.card_id = latest.card_id
                     AND pa.snapshot_date = latest.max_date
            WHERE pa.card_id IS NOT NULL
        """).fetchall():
            adv_pitching[r['card_id']] = dict(r)
    except Exception as _e:
        logger.warning("adv_pitching fetch failed (SIERA overlay disabled): %s", _e)

    # Pitch arsenal data (arsenal diversity factor)
    arsenal_data = {}
    try:
        for r in cursor.execute("""
            SELECT pr.card_id, pr.fb, pr.ch, pr.cb, pr.sl, pr.si, pr.sp,
                   pr.ct, pr.fo, pr.cc, pr.sc, pr.kc, pr.kn,
                   pr.pitch_count, pr.velocity
            FROM pitch_ratings pr
            INNER JOIN (
                SELECT card_id, MAX(snapshot_date) AS max_date
                FROM pitch_ratings WHERE card_id IS NOT NULL
                GROUP BY card_id
            ) latest ON pr.card_id = latest.card_id
                     AND pr.snapshot_date = latest.max_date
            WHERE pr.card_id IS NOT NULL
        """).fetchall():
            arsenal_data[r['card_id']] = dict(r)
    except Exception as _e:
        logger.warning("pitch_ratings fetch failed (arsenal diversity disabled): %s", _e)

    # League averages for performance overlay baseline
    try:
        avg_woba_row = cursor.execute(
            "SELECT AVG(woba) FROM batting_stats_adv WHERE woba > 0"
        ).fetchone()
        avg_woba = float(avg_woba_row[0]) if avg_woba_row and avg_woba_row[0] else 0.320
    except Exception:
        avg_woba = 0.320

    try:
        avg_siera_row = cursor.execute(
            "SELECT AVG(siera) FROM pitching_stats_adv WHERE siera > 0"
        ).fetchone()
        avg_siera = float(avg_siera_row[0]) if avg_siera_row and avg_siera_row[0] else 4.00
    except Exception:
        avg_siera = 4.00

    # Pre-compute game-log aggregates per batter_card_id so each batter's
    # meta recalc can use EV + LD% + observed K%/BB% as a superstat overlay.
    # One scan of game_log_at_bats keyed by batter_card_id — O(N) vs
    # per-card queries.
    gl_agg_batting: dict = {}
    try:
        for r in cursor.execute("""
            SELECT batter_card_id AS cid, COUNT(*) AS ab,
                   AVG(exit_velocity) AS ev_avg,
                   SUM(CASE WHEN LOWER(batted_ball_type) LIKE 'line%' THEN 1 ELSE 0 END)
                    * 1.0 / NULLIF(SUM(CASE WHEN batted_ball_type IS NOT NULL THEN 1 ELSE 0 END), 0)
                    * 100 AS ld_pct,
                   SUM(CASE WHEN outcome = 'K' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS k_pct,
                   SUM(CASE WHEN outcome = 'BB' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS bb_pct
            FROM game_log_at_bats
            WHERE batter_card_id IS NOT NULL
            GROUP BY batter_card_id
        """).fetchall():
            gl_agg_batting[r['cid']] = {
                'ev_avg': r['ev_avg'], 'ld_pct': r['ld_pct'], 'ab': r['ab'],
                'k_pct': r['k_pct'], 'bb_pct': r['bb_pct'],
            }
    except Exception as _e:
        logger.warning("game-log batting agg failed (EV/LD/K/BB overlay disabled): %s", _e)

    # Clutch event aggregates per batter card_id — sums 2-out RBI,
    # LOB-RISP-2-out, and total game-batting PA so the clutch overlay
    # can compute net clutch-per-PA. One combined query for efficiency.
    clutch_batting: dict = {}
    try:
        # Per-card 2-out RBI + LOB-RISP-2-out counts
        for r in cursor.execute("""
            SELECT card_id,
                   SUM(CASE WHEN event_type = '2OUT_RBI' THEN 1 ELSE 0 END) AS rbi_n,
                   SUM(CASE WHEN event_type = 'LOB_RISP_2OUT' THEN 1 ELSE 0 END) AS lob_n,
                   SUM(CASE WHEN event_type = 'ERROR' THEN 1 ELSE 0 END) AS err_n
            FROM game_clutch_events
            WHERE card_id IS NOT NULL
            GROUP BY card_id
        """).fetchall():
            clutch_batting[r['card_id']] = {
                'rbi_2out': r['rbi_n'] or 0,
                'lob_risp_2out': r['lob_n'] or 0,
                'errors': r['err_n'] or 0,
            }
        # Denominator PA + defensive games count
        for r in cursor.execute("""
            SELECT card_id,
                   SUM(COALESCE(ab, 0) + COALESCE(bb, 0)) AS total_pa,
                   COUNT(DISTINCT game_id) AS games
            FROM game_batting
            WHERE card_id IS NOT NULL AND ab > 0
            GROUP BY card_id
        """).fetchall():
            d = clutch_batting.setdefault(r['card_id'], {})
            d['pa'] = r['total_pa'] or 0
            d['games'] = r['games'] or 0
    except Exception as _e:
        logger.warning("clutch batting agg failed (clutch overlay disabled): %s", _e)

    # Per-pitcher ERA+ overperformance: observed ERA+ minus rating-predicted.
    # Fit OLS on (stuff+movement+control → ERA+) across the population, then
    # compute residual per pitcher. Parallels batting over-performance overlay.
    overperf_pitching = {}
    try:
        train_rows = cursor.execute("""
            SELECT c.card_id,
                   (COALESCE(c.stuff,0) + COALESCE(c.movement,0)
                    + COALESCE(c.control,0)) / 3.0 AS comp,
                   SUM(COALESCE(ps.era_plus * (ps.ip * 3 + COALESCE(ps.bb,0)
                                                + COALESCE(ps.hits_allowed,0)), 0)) * 1.0
                     / NULLIF(SUM(ps.ip * 3 + COALESCE(ps.bb,0)
                                  + COALESCE(ps.hits_allowed,0)), 0) AS obs_ep,
                   SUM(ps.ip * 3 + COALESCE(ps.bb,0)
                       + COALESCE(ps.hits_allowed,0)) AS bf_est
            FROM cards c
            INNER JOIN pitching_stats ps ON ps.card_id = c.card_id
            WHERE c.pitcher_role IS NOT NULL AND ps.ip IS NOT NULL AND ps.ip > 0
              AND ps.era_plus IS NOT NULL AND c.card_value > 0
            GROUP BY c.card_id
            HAVING SUM(ps.ip * 3) >= 90
        """).fetchall()
        xs = [float(r['comp'] or 0) for r in train_rows]
        ys = [float(r['obs_ep'] or 0) for r in train_rows]
        n = len(xs)
        if n >= 30:
            mx = sum(xs)/n; my = sum(ys)/n
            num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
            den = sum((x-mx)**2 for x in xs) or 1.0
            b = num / den
            a = my - b * mx
            for r in cursor.execute("""
                SELECT c.card_id,
                       (COALESCE(c.stuff,0) + COALESCE(c.movement,0)
                        + COALESCE(c.control,0)) / 3.0 AS comp,
                       SUM(COALESCE(ps.era_plus * (ps.ip * 3 + COALESCE(ps.bb,0)
                                                    + COALESCE(ps.hits_allowed,0)), 0)) * 1.0
                         / NULLIF(SUM(ps.ip * 3 + COALESCE(ps.bb,0)
                                      + COALESCE(ps.hits_allowed,0)), 0) AS obs_ep,
                       SUM(ps.ip * 3 + COALESCE(ps.bb,0)
                           + COALESCE(ps.hits_allowed,0)) AS bf_est
                FROM cards c
                INNER JOIN pitching_stats ps ON ps.card_id = c.card_id
                WHERE c.pitcher_role IS NOT NULL AND ps.ip IS NOT NULL AND ps.ip > 0
                  AND ps.era_plus IS NOT NULL
                GROUP BY c.card_id
            """).fetchall():
                predicted = a + b * float(r['comp'] or 0)
                over = float(r['obs_ep'] or 0) - predicted
                overperf_pitching[r['card_id']] = {
                    'over': over, 'bf': int(r['bf_est'] or 0),
                }
    except Exception as _e:
        logger.warning("overperf_pitching fit failed (pitcher overperf overlay disabled): %s", _e)

    # Observed BB/9 delta per pitcher card_id — BF-weighted, league-adjusted.
    # Residual r=-0.34 after FIP overlay — walks still undercharged.
    obs_bb9_pitching = {}
    try:
        league_bb9_baseline = {}
        for r in cursor.execute("""
            SELECT league_id AS lg, AVG(bb_per_9) AS mean_bb9, COUNT(*) AS n
            FROM pitching_stats
            WHERE bb_per_9 IS NOT NULL AND ip >= 30 AND card_id IS NOT NULL
            GROUP BY league_id HAVING COUNT(*) >= 30
        """).fetchall():
            league_bb9_baseline[r['lg']] = r['mean_bb9']
        for r in cursor.execute("""
            SELECT card_id, league_id AS lg, bb_per_9, ip,
                   (COALESCE(hits_allowed,0) + COALESCE(bb,0) + COALESCE(hbp,0)
                    + ROUND(COALESCE(ip,0) * 3)) AS bf_est
            FROM pitching_stats
            WHERE card_id IS NOT NULL AND bb_per_9 IS NOT NULL AND ip > 0
        """).fetchall():
            baseline = league_bb9_baseline.get(r['lg'])
            if baseline is None:
                continue
            bf = max(int(r['bf_est'] or 0), 1)
            delta = float(r['bb_per_9']) - float(baseline)
            entry = obs_bb9_pitching.setdefault(r['card_id'],
                                                  {'dw': 0.0, 'bf': 0})
            entry['dw'] += delta * bf
            entry['bf'] += bf
        for cid, entry in obs_bb9_pitching.items():
            entry['bb9_delta'] = (entry['dw'] / entry['bf']) if entry['bf'] else 0.0
    except Exception as _e:
        logger.warning("obs_bb9_pitching fit failed (BB/9 overlay disabled): %s", _e)

    # Observed FIP delta per pitcher card_id, BF-weighted and league-adjusted.
    # Residual r=-0.359 with WAR/200 (pooled) — captures HR-per-9 signal
    # (r=-0.414 with residuals) that SIERA overlay undershoots because SIERA
    # weights HR less heavily than FIP does.
    obs_fip_pitching: dict = {}
    try:
        # Per-league FIP baselines
        league_fip_baseline = {}
        for r in cursor.execute("""
            SELECT league_id AS lg, AVG(fip) AS mean_fip, COUNT(*) AS n
            FROM pitching_stats
            WHERE fip IS NOT NULL AND ip >= 30 AND card_id IS NOT NULL
            GROUP BY league_id HAVING COUNT(*) >= 30
        """).fetchall():
            league_fip_baseline[r['lg']] = r['mean_fip']

        # BF-weighted delta per card
        for r in cursor.execute("""
            SELECT card_id, league_id AS lg, fip, ip,
                   (COALESCE(hits_allowed,0) + COALESCE(bb,0) + COALESCE(hbp,0)
                    + ROUND(COALESCE(ip,0) * 3)) AS bf_est
            FROM pitching_stats
            WHERE card_id IS NOT NULL AND fip IS NOT NULL AND ip IS NOT NULL
              AND ip > 0
        """).fetchall():
            baseline = league_fip_baseline.get(r['lg'])
            if baseline is None:
                continue
            bf = max(int(r['bf_est'] or 0), 1)
            delta = float(r['fip']) - float(baseline)
            entry = obs_fip_pitching.setdefault(r['card_id'],
                                                 {'dw': 0.0, 'bf': 0})
            entry['dw'] += delta * bf
            entry['bf'] += bf
        for cid, entry in obs_fip_pitching.items():
            entry['fip_delta'] = (entry['dw'] / entry['bf']) if entry['bf'] else 0.0
    except Exception as _e:
        logger.warning("obs_fip_pitching fit failed (FIP overlay disabled): %s", _e)

    # Reliever discipline: inherited runners + scored per pitcher card_id
    reliever_discipline: dict = {}
    try:
        for r in cursor.execute("""
            SELECT card_id,
                   SUM(CASE WHEN event_type = 'INHERITED_RUNNERS' THEN event_count ELSE 0 END) AS inh,
                   SUM(CASE WHEN event_type = 'INHERITED_SCORED' THEN event_count ELSE 0 END) AS sc
            FROM game_clutch_events
            WHERE card_id IS NOT NULL
              AND event_type IN ('INHERITED_RUNNERS', 'INHERITED_SCORED')
            GROUP BY card_id
        """).fetchall():
            reliever_discipline[r['card_id']] = {
                'inh': r['inh'] or 0,
                'scored': r['sc'] or 0,
            }
    except Exception as _e:
        logger.warning("reliever_discipline fetch failed (inherited-runner overlay disabled): %s", _e)

    # Same for pitchers — aggregate what THEY allowed across the game logs.
    # The pitcher side uses batted_ball_type + exit_velocity from the same
    # rows keyed on pitcher_card_id instead of batter_card_id.
    gl_agg_pitching: dict = {}
    try:
        for r in cursor.execute("""
            SELECT pitcher_card_id AS cid, COUNT(*) AS bf,
                   AVG(exit_velocity) AS ev_allowed,
                   SUM(CASE WHEN LOWER(batted_ball_type) LIKE 'line%' THEN 1 ELSE 0 END)
                    * 1.0 / NULLIF(SUM(CASE WHEN batted_ball_type IS NOT NULL THEN 1 ELSE 0 END), 0)
                    * 100 AS ld_pct_allowed,
                   SUM(CASE WHEN outcome = 'K' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS k_pct,
                   SUM(CASE WHEN outcome = 'BB' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS bb_pct
            FROM game_log_at_bats
            WHERE pitcher_card_id IS NOT NULL
            GROUP BY pitcher_card_id
        """).fetchall():
            gl_agg_pitching[r['cid']] = {
                'ev_allowed': r['ev_allowed'],
                'ld_pct_allowed': r['ld_pct_allowed'],
                'k_pct': r['k_pct'], 'bb_pct': r['bb_pct'],
                'bf': r['bf'],
            }
    except Exception as _e:
        logger.warning("game-log pitching agg failed (pitcher EV/LD overlay disabled): %s", _e)

    # ── Recalc batting ──
    # Includes platoon splits, position flexibility, defense, speed, card_type bias
    batters = cursor.execute("""
        SELECT card_id, contact, gap_power, power, eye, avoid_ks, babip,
               card_value, position, card_type, tier,
               infield_range, infield_error, infield_arm,
               catcher_ability, catcher_frame, catcher_arm,
               of_range, of_error, of_arm,
               speed, stealing, baserunning,
               contact_vl, contact_vr, gap_vl, gap_vr,
               power_vl, power_vr, eye_vl, eye_vr,
               pos_rating_c, pos_rating_1b, pos_rating_2b, pos_rating_3b,
               pos_rating_ss, pos_rating_lf, pos_rating_cf, pos_rating_rf
        FROM cards WHERE position IS NOT NULL AND pitcher_role IS NULL
    """).fetchall()

    bat_count = 0
    for row in batters:
        d = dict(row)
        d['defense_score'] = calc_defense_score(d)
        d['speed_score'] = calc_speed_score(d)
        d['ovr'] = d.get('card_value', 0)
        # Merge advanced batting stats for performance overlay
        adv = adv_batting.get(d['card_id'])
        if adv:
            d['adv_woba'] = adv.get('woba')
            d['adv_wpa'] = adv.get('wpa')
            d['adv_pa'] = adv.get('pa')
        d['_league_avg_woba'] = avg_woba
        # Game-log superstats for the quality-of-contact + discipline overlay.
        # Only injected when the card has game_log_at_bats data; otherwise
        # the overlay returns 0 and behavior is unchanged.
        gl = gl_agg_batting.get(d['card_id'])
        if gl:
            d['_gl_ev_avg'] = gl.get('ev_avg')
            d['_gl_ld_pct'] = gl.get('ld_pct')
            d['_gl_k_pct'] = gl.get('k_pct')
            d['_gl_bb_pct'] = gl.get('bb_pct')
            d['_gl_at_bats'] = gl.get('ab') or 0
        # Clutch + error aggregates from box score event tables
        cb = clutch_batting.get(d['card_id'])
        if cb:
            d['_clutch_2out_rbi'] = cb.get('rbi_2out') or 0
            d['_clutch_lob_risp_2out'] = cb.get('lob_risp_2out') or 0
            d['_clutch_pa'] = cb.get('pa') or 0
            d['_errors_count'] = cb.get('errors') or 0
            d['_def_games_count'] = cb.get('games') or 0
        # League-adjusted ISO delta for iso overlay (r=+0.580 with WAR).
        # delta is already relative to league+tier baseline, so cross-
        # league scoring environments don't bias the signal.
        iso = obs_iso_batting.get(d['card_id'])
        if iso and 'iso_delta' in iso:
            d['_obs_iso_delta'] = iso['iso_delta']
            d['_obs_iso_pa'] = iso.get('pa') or 0
        # Observed OPS+ for park/league-adjusted offense overlay
        # (residual r=+0.325 with WAR/600 — strongest untapped signal).
        opp = obs_opsplus_batting.get(d['card_id'])
        if opp:
            d['_obs_ops_plus'] = opp.get('ops_plus')
            d['_obs_ops_plus_pa'] = opp.get('pa') or 0
        # Observed OBP delta for on-base overlay (residual r=+0.337).
        obp = obs_obp_batting.get(d['card_id'])
        if obp and 'obp_delta' in obp:
            d['_obs_obp_delta'] = obp['obp_delta']
            d['_obs_obp_pa'] = obp.get('pa') or 0
        # OPS+ overperformance for over-performance overlay (residual r=+0.551).
        op = overperf_batting.get(d['card_id'])
        if op:
            d['_over_ops_plus'] = op.get('over')
            d['_over_ops_plus_pa'] = op.get('pa') or 0
        # Observed BABIP delta for babip overlay (residual r=+0.26).
        bab = obs_babip_batting.get(d['card_id'])
        if bab and 'babip_delta' in bab:
            d['_obs_babip_delta'] = bab['babip_delta']
            d['_obs_babip_pa'] = bab.get('pa') or 0
        # ISO gap vs power-rating prediction (residual r=+0.466 — biggest
        # 2026-04-20 win). Distinct from obs_iso_delta (tier-peer baseline).
        isg = obs_iso_gap_batting.get(d['card_id'])
        if isg and 'gap' in isg:
            d['_obs_iso_gap'] = isg['gap']
            d['_obs_iso_gap_pa'] = isg.get('pa') or 0
        meta = calc_batting_meta(d)
        if meta and meta > 0:
            cursor.execute("UPDATE cards SET meta_score_batting = ? WHERE card_id = ?",
                          (meta, d['card_id']))
            bat_count += 1
            # Commit every 250 rows so we don't hold a write lock for the
            # entire batter sweep. Concurrent readers (Streamlit pages) can
            # proceed between chunks; concurrent writers get a quick turn.
            if bat_count % 250 == 0:
                conn.commit()

    conn.commit()  # flush any remaining batters

    # ── Recalc pitching ──
    # Includes platoon splits, arsenal diversity, performance overlay, card_type bias
    pitchers = cursor.execute("""
        SELECT card_id, stuff, movement, control, p_hr, p_babip,
               card_value, stamina, hold, card_type,
               pitcher_role_name, pitcher_role,
               stuff_vl, stuff_vr, movement_vl, movement_vr,
               control_vl, control_vr
        FROM cards WHERE pitcher_role IS NOT NULL
    """).fetchall()

    pit_count = 0
    for row in pitchers:
        d = dict(row)
        d['ovr'] = d.get('card_value', 0)
        # Merge arsenal data for diversity factor
        arsenal = arsenal_data.get(d['card_id'])
        if arsenal:
            for k, v in arsenal.items():
                if k != 'card_id':
                    d[k] = v
        # Merge advanced pitching stats for performance overlay
        adv = adv_pitching.get(d['card_id'])
        if adv:
            d['adv_siera'] = adv.get('siera')
            d['adv_wpa'] = adv.get('wpa')
            d['adv_ip'] = adv.get('ip')
        d['_league_avg_siera'] = avg_siera
        # Game-log pitcher superstats (EV-allowed + LD%-allowed + K%/BB%)
        # from game_log_at_bats keyed by pitcher_card_id.
        gl_p = gl_agg_pitching.get(d['card_id'])
        if gl_p:
            d['_gl_p_ev_allowed'] = gl_p.get('ev_allowed')
            d['_gl_p_ld_pct_allowed'] = gl_p.get('ld_pct_allowed')
            d['_gl_p_k_pct'] = gl_p.get('k_pct')
            d['_gl_p_bb_pct'] = gl_p.get('bb_pct')
            d['_gl_p_batters_faced'] = gl_p.get('bf') or 0
        # Reliever discipline — inherited-runner strand rate from box scores
        rd = reliever_discipline.get(d['card_id'])
        if rd:
            d['_inherited_runners'] = rd.get('inh') or 0
            d['_inherited_scored'] = rd.get('scored') or 0
        # Observed FIP delta (league-adjusted) for FIP overlay.
        # Residual r=-0.359 — captures HR-per-9 signal SIERA undershoots.
        fip = obs_fip_pitching.get(d['card_id'])
        if fip and 'fip_delta' in fip:
            d['_obs_fip_delta'] = fip['fip_delta']
            d['_obs_fip_bf'] = fip.get('bf') or 0
        # ERA+ overperformance for over-performance overlay.
        op = overperf_pitching.get(d['card_id'])
        if op:
            d['_over_era_plus'] = op.get('over')
            d['_over_era_plus_bf'] = op.get('bf') or 0
        # Observed BB/9 delta for BB/9 overlay (residual r=-0.34).
        bb9 = obs_bb9_pitching.get(d['card_id'])
        if bb9 and 'bb9_delta' in bb9:
            d['_obs_bb9_delta'] = bb9['bb9_delta']
            d['_obs_bb9_bf'] = bb9.get('bf') or 0
        meta = calc_pitching_meta(d)
        if meta and meta > 0:
            cursor.execute("UPDATE cards SET meta_score_pitching = ? WHERE card_id = ?",
                          (meta, d['card_id']))
            pit_count += 1
            if pit_count % 250 == 0:
                conn.commit()

    conn.commit()  # flush any remaining pitchers

    # ── Sync roster table metas with recalculated card metas ──
    # The roster table stores meta_score from import time, which goes stale
    # when weights change. Update roster metas by joining against cards.
    # Uses card_id when available (exact match), falls back to name + OVR match.
    roster_synced = 0
    try:
        # Best: match by card_id (exact)
        cursor.execute("""
            UPDATE roster SET meta_score = (
                SELECT c.meta_score_batting FROM cards c
                WHERE c.card_id = roster.card_id
                  AND c.meta_score_batting IS NOT NULL AND c.position != 1
                LIMIT 1
            )
            WHERE card_id IS NOT NULL
              AND position NOT IN ('SP', 'RP', 'CL')
              AND EXISTS (
                  SELECT 1 FROM cards c WHERE c.card_id = roster.card_id
                    AND c.meta_score_batting IS NOT NULL AND c.position != 1
              )
        """)
        roster_synced += cursor.rowcount

        cursor.execute("""
            UPDATE roster SET meta_score = (
                SELECT c.meta_score_pitching FROM cards c
                WHERE c.card_id = roster.card_id
                  AND c.meta_score_pitching IS NOT NULL AND c.pitcher_role IS NOT NULL
                LIMIT 1
            )
            WHERE card_id IS NOT NULL
              AND position IN ('SP', 'RP', 'CL')
              AND EXISTS (
                  SELECT 1 FROM cards c WHERE c.card_id = roster.card_id
                    AND c.meta_score_pitching IS NOT NULL AND c.pitcher_role IS NOT NULL
              )
        """)
        roster_synced += cursor.rowcount

        # Fallback: match by name + OVR for rows without card_id
        cursor.execute("""
            UPDATE roster SET meta_score = (
                SELECT c.meta_score_batting FROM cards c
                WHERE c.card_title LIKE '%' || roster.player_name || '%'
                  AND c.card_value = roster.ovr
                  AND c.meta_score_batting IS NOT NULL AND c.position != 1
                LIMIT 1
            )
            WHERE card_id IS NULL
              AND position NOT IN ('SP', 'RP', 'CL')
              AND EXISTS (
                  SELECT 1 FROM cards c
                  WHERE c.card_title LIKE '%' || roster.player_name || '%'
                    AND c.card_value = roster.ovr
                    AND c.meta_score_batting IS NOT NULL AND c.position != 1
              )
        """)
        roster_synced += cursor.rowcount

        cursor.execute("""
            UPDATE roster SET meta_score = (
                SELECT c.meta_score_pitching FROM cards c
                WHERE c.card_title LIKE '%' || roster.player_name || '%'
                  AND c.card_value = roster.ovr
                  AND c.meta_score_pitching IS NOT NULL AND c.pitcher_role IS NOT NULL
                LIMIT 1
            )
            WHERE card_id IS NULL
              AND position IN ('SP', 'RP', 'CL')
              AND EXISTS (
                  SELECT 1 FROM cards c
                  WHERE c.card_title LIKE '%' || roster.player_name || '%'
                    AND c.card_value = roster.ovr
                    AND c.meta_score_pitching IS NOT NULL AND c.pitcher_role IS NOT NULL
              )
        """)
        roster_synced += cursor.rowcount
    except Exception as _roster_sync_err:
        # Previously this was a bare `pass` that silently left roster metas
        # stale (same failure mode as the Bichette flip-flop bug).  Now we
        # retry once on lock errors, log other failures loudly, and tag
        # the result so the background worker can gate data_version bumps
        # on it.
        import logging as _lg
        _logger = _lg.getLogger(__name__)
        _err_str = str(_roster_sync_err)
        _roster_sync_error = _err_str
        if 'locked' in _err_str.lower():
            try:
                import time as _t
                _t.sleep(1.0)
                conn.commit()  # release any pending txn state
                # Do not re-run the 4-stage sync here — keep error signaled
                # so the worker retries the whole recalc instead.
            except Exception:
                pass
        _logger.warning("roster meta sync failed: %s", _err_str)
    else:
        _roster_sync_error = None

    conn.commit()
    conn.close()

    return {
        "status": "success" if _roster_sync_error is None else "partial",
        "batters_updated": bat_count,
        "pitchers_updated": pit_count,
        "roster_synced": roster_synced,
        "roster_sync_error": _roster_sync_error,
        "weight_source": source,
        "message": (
            f"Recalculated {bat_count} batters + {pit_count} pitchers using "
            f"{source} weights. Synced {roster_synced} roster metas."
            + (f" WARNING: roster sync failed — {_roster_sync_error}"
               if _roster_sync_error else "")
        ),
    }


def ingest_fielding_stats(filepath: str) -> dict:
    """Ingest fielding stats CSV into fielding_stats table."""
    df = parse_fielding_stats_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    # Delete any existing snapshot from today (re-import overwrites)
    cursor.execute("DELETE FROM fielding_stats WHERE DATE(snapshot_date) = ?", (snapshot_date,))

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name or name.lower() in ('total', 'totals', 'team'):
            continue

        pos = _safe_str(row.get('POS', '')).strip()
        bats = _safe_str(row.get('B', '')).strip()
        throws = _safe_str(row.get('T', '')).strip()

        card_id = _match_card_id(cursor, name)

        cursor.execute("""
            INSERT INTO fielding_stats (
                player_name, position, bats, throws,
                games, gs, tc, assists, putouts, errors, dp,
                pct, rng, zr, eff,
                sba, rto, rto_pct, ip, pb, cera, frm, arm,
                card_id, snapshot_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, pos, bats, throws,
            _safe_int(row.get('G', 0)),
            _safe_int(row.get('GS', 0)),
            _safe_int(row.get('TC', 0)),
            _safe_int(row.get('A', 0)),
            _safe_int(row.get('PO', 0)),
            _safe_int(row.get('E', 0)),
            _safe_int(row.get('DP', 0)),
            _safe_float(row.get('PCT', 0)),
            _safe_float(row.get('RNG', 0)),
            _safe_float(row.get('ZR', 0)),
            _safe_float(row.get('EFF', 0)),
            _safe_int(row.get('SBA', 0)),
            _safe_int(row.get('RTO', 0)),
            _safe_float(row.get('RTO%', 0)),
            _safe_float(row.get('IP', 0)),
            _safe_int(row.get('PB', 0)),
            _safe_float(row.get('CERA', 0)),
            _safe_float(row.get('FRM', 0)),
            _safe_float(row.get('ARM', 0)),
            card_id,
            snapshot_date,
        ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "fielding_stats", "rows": count}


def ingest_pitch_ratings(filepath: str) -> dict:
    """Ingest individual pitch ratings CSV into pitch_ratings table."""
    df = parse_pitch_ratings_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    # Delete any existing snapshot from today (re-import overwrites)
    cursor.execute("DELETE FROM pitch_ratings WHERE DATE(snapshot_date) = ?", (snapshot_date,))

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name or name.lower() in ('total', 'totals', 'team'):
            continue

        pos = _safe_str(row.get('POS', '')).strip()
        throws = _safe_str(row.get('T', '')).strip()
        age = _safe_int(row.get('Age', 0))

        card_id = _match_card_id(cursor, name)

        cursor.execute("""
            INSERT INTO pitch_ratings (
                player_name, position, throws, age,
                fb, ch, cb, sl, si, sp, ct, fo, cc, sc, kc, kn,
                pitch_count, velocity, slot, stamina,
                card_id, snapshot_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, pos, throws, age,
            _safe_int(row.get('FB', 0)),
            _safe_int(row.get('CH', 0)),
            _safe_int(row.get('CB', 0)),
            _safe_int(row.get('SL', 0)),
            _safe_int(row.get('SI', 0)),
            _safe_int(row.get('SP', 0)),
            _safe_int(row.get('CT', 0)),
            _safe_int(row.get('FO', 0)),
            _safe_int(row.get('CC', 0)),
            _safe_int(row.get('SC', 0)),
            _safe_int(row.get('KC', 0)),
            _safe_int(row.get('KN', 0)),
            _safe_int(row.get('PIT', 0)),
            _safe_str(row.get('VELO', '')).strip(),
            _safe_str(row.get('Slot', '')).strip(),
            _safe_int(row.get('STM', 0)),
            card_id,
            snapshot_date,
        ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "pitch_ratings", "rows": count}


def ingest_position_ratings(filepath: str) -> dict:
    """Ingest position ratings CSV and update cards table with position ratings.

    This does not use its own table — it updates pos_rating_* columns on the cards table.
    Handles '-' values in position columns (they mean 0/not applicable).
    """
    df = parse_position_ratings_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name or name.lower() in ('total', 'totals', 'team'):
            continue

        card_id = _match_card_id(cursor, name)
        if card_id is None:
            continue

        # Convert position ratings, handling '-' as 0
        def _pos_val(col):
            val = row.get(col, '-')
            if val == '-' or val is None:
                return 0
            return _safe_int(val, 0)

        cursor.execute("""
            UPDATE cards SET
                pos_rating_p = ?,
                pos_rating_c = ?,
                pos_rating_1b = ?,
                pos_rating_2b = ?,
                pos_rating_3b = ?,
                pos_rating_ss = ?,
                pos_rating_lf = ?,
                pos_rating_cf = ?,
                pos_rating_rf = ?,
                last_updated_at = ?
            WHERE card_id = ?
        """, (
            _pos_val('P'),
            _pos_val('C'),
            _pos_val('1B'),
            _pos_val('2B'),
            _pos_val('3B'),
            _pos_val('SS'),
            _pos_val('LF'),
            _pos_val('CF'),
            _pos_val('RF'),
            # UTC to match CURRENT_TIMESTAMP default on this column.
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            card_id,
        ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "position_ratings", "rows": count}


# ══════════════════════════════════════════════════════════════════════
# LEAGUE-WIDE TEAM STATS — batting + pitching + park factors
# Each of these writes to the same league_team_stats wide table using
# upsert semantics so partial data is OK. Keyed on (league_id, team, date).
# ══════════════════════════════════════════════════════════════════════

def _upsert_league_team_row(cursor, league_id: str, team_name: str,
                            snapshot_date: str, field_map: dict) -> None:
    """Upsert a partial row into league_team_stats.

    If the (league_id, team, date) row exists, update only the provided
    columns. Otherwise insert with the provided columns. This lets each
    of the three team-stats CSVs (batting / pitching / park) contribute
    independently without overwriting each other.
    """
    row = cursor.execute(
        "SELECT id FROM league_team_stats WHERE league_id IS ? AND team_name = ? AND snapshot_date = ?",
        (league_id, team_name, snapshot_date),
    ).fetchone()
    if row:
        if not field_map:
            return
        set_clause = ", ".join(f"{col} = ?" for col in field_map.keys())
        cursor.execute(
            f"UPDATE league_team_stats SET {set_clause} WHERE id = ?",
            list(field_map.values()) + [row[0]],
        )
    else:
        cols = ["league_id", "team_name", "snapshot_date"] + list(field_map.keys())
        placeholders = ", ".join(["?"] * len(cols))
        cursor.execute(
            f"INSERT INTO league_team_stats ({', '.join(cols)}) VALUES ({placeholders})",
            [league_id, team_name, snapshot_date] + list(field_map.values()),
        )


def ingest_team_stats_batting(filepath: str) -> dict:
    """Ingest league-wide TEAM batting stats CSV (one row per team)."""
    df = parse_team_stats_batting_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    league_id = detect_league_id(filepath)
    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    count = 0
    for _, row in df.iterrows():
        team = _safe_str(row.get('Team Name', '')).strip()
        if not team:
            continue
        field_map = {
            'games': _safe_int(row.get('G', 0)),
            'doubles': _safe_int(row.get('2B', 0)),
            'triples': _safe_int(row.get('3B', 0)),
            'hr': _safe_int(row.get('HR', 0)),
            'runs': _safe_int(row.get('R', 0)),
            'bb': _safe_int(row.get('BB', 0)),
            'so': _safe_int(row.get('SO', 0)),
            'sb': _safe_int(row.get('SB', 0)),
            'avg': _safe_float(row.get('AVG', 0)),
            'obp': _safe_float(row.get('OBP', 0)),
            'slg': _safe_float(row.get('SLG', 0)),
            'ops': _safe_float(row.get('OPS', 0)),
        }
        _upsert_league_team_row(cursor, league_id, team, snapshot_date, field_map)
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "team_stats_batting", "rows": count}


def ingest_team_stats_pitching(filepath: str) -> dict:
    """Ingest league-wide TEAM pitching stats CSV (one row per team)."""
    df = parse_team_stats_pitching_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    league_id = detect_league_id(filepath)
    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    count = 0
    for _, row in df.iterrows():
        team = _safe_str(row.get('Team Name', '')).strip()
        if not team:
            continue
        field_map = {
            'wins': _safe_int(row.get('W', 0)),
            'losses': _safe_int(row.get('L', 0)),
            'saves': _safe_int(row.get('SV', 0)),
            'ip': _safe_float(row.get('IP', 0)),
            'runs_allowed': _safe_int(row.get('R', 0)),
            'hr_allowed': _safe_int(row.get('HR', 0)),
            'bb_allowed': _safe_int(row.get('BB', 0)),
            'k': _safe_int(row.get('K', 0)),
            'whip': _safe_float(row.get('WHIP', 0)),
            'batting_avg_against': _safe_float(row.get('AVG', 0)),
            'era': _safe_float(row.get('ERA', 0)),
            'fip_minus': _safe_int(row.get('FIP-', 0)),
        }
        _upsert_league_team_row(cursor, league_id, team, snapshot_date, field_map)
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "team_stats_pitching", "rows": count}


def ingest_team_stats_park(filepath: str) -> dict:
    """Ingest league-wide park factors CSV (one row per team)."""
    df = parse_team_stats_park_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    league_id = detect_league_id(filepath)
    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    count = 0
    for _, row in df.iterrows():
        team = _safe_str(row.get('Team Name', '')).strip()
        if not team:
            continue
        field_map = {
            'park_name': _safe_str(row.get('Park', '')),
            'pf_avg': _safe_float(row.get('PF AVG', 0)),
            'pf_avg_l': _safe_float(row.get('AVG L', 0)),
            'pf_avg_r': _safe_float(row.get('AVG R', 0)),
            'pf_hr': _safe_float(row.get('PF HR', 0)),
            'pf_hr_l': _safe_float(row.get('HR L', 0)),
            'pf_hr_r': _safe_float(row.get('HR R', 0)),
            'pf_d': _safe_float(row.get('PF D', 0)),
            'pf_t': _safe_float(row.get('PF T', 0)),
            'pf_overall': _safe_float(row.get('PF', 0)),
        }
        _upsert_league_team_row(cursor, league_id, team, snapshot_date, field_map)
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "team_stats_park", "rows": count}


# ══════════════════════════════════════════════════════════════════════
# FIELDING RATINGS (OOTP 27 granular fielding export)
# Columns: POS, #, Name, Inf, Age, B, T, C ABI, C ARM, IF RNG, IF ERR,
#          IF ARM, TDP, OF RNG, OF ERR, OF ARM, St
# These update the cards table's fielding columns directly — same fields
# the market CSV provides, but refreshed independently so per-player
# fielding data stays current between market exports. Unlike position_
# ratings (which writes pos_rating_P/C/1B/… ability ladders), this file
# carries the granular SKILL numbers (range, arm, error).
# ══════════════════════════════════════════════════════════════════════

def ingest_fielding_ratings(filepath: str) -> dict:
    """Ingest player fielding ratings CSV and update cards table skill columns.

    The fielding_ratings export (new in OOTP 27) carries granular per-player
    fielding skills: catcher ability/arm, infield range/error/arm + TDP,
    outfield range/error/arm. We match by player name and refresh the
    matching card's existing columns — no new table required.

    '-' values in unused position groups mean "not applicable" → stored as 0.
    """
    from app.utils.csv_parser import parse_fielding_ratings_csv
    df = parse_fielding_ratings_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name or name.lower() in ('total', 'totals', 'team'):
            continue

        card_id = _match_card_id(cursor, name)
        if card_id is None:
            continue

        def _rating(col):
            val = row.get(col, '-')
            if val == '-' or val is None:
                return 0
            return _safe_int(val, 0)

        cursor.execute(
            """
            UPDATE cards SET
                catcher_ability = ?,
                catcher_arm     = ?,
                infield_range   = ?,
                infield_error   = ?,
                infield_arm     = ?,
                of_range        = ?,
                of_error        = ?,
                of_arm          = ?,
                last_updated_at = ?
            WHERE card_id = ?
            """,
            (
                _rating('C ABI'),
                _rating('C ARM'),
                _rating('IF RNG'),
                _rating('IF ERR'),
                _rating('IF ARM'),
                _rating('OF RNG'),
                _rating('OF ERR'),
                _rating('OF ARM'),
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                card_id,
            ),
        )
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "fielding_ratings", "rows": count}


def ingest_team_standings(filepath: str) -> dict:
    """Recognize the league-wide team standings export without ingesting.

    The `team_statistics___info_-_sortable_stats_default.csv` export is the
    standings view (team W/L, division position, games back, streak). It's
    useful context — which teams are in contention, who's rebuilding — but
    the current `league_team_stats` wide table doesn't have standings
    columns and adding them is a migration we can defer until a feature
    actually consumes it (trade-scout, schedule context, etc.).

    For now we RECOGNIZE the file so the Data Refresh UI doesn't flag it as
    "unknown", but we skip ingestion. Once a consumer lands, swap this for
    a real handler that writes W/L/pct/games_back/streak into either
    league_team_stats or a dedicated league_team_standings table.
    """
    return {
        "status": "skipped",
        "file_type": "team_standings",
        "rows": 0,
        "reason": "standings columns not yet in schema — recognized but not ingested",
    }


def ingest_team_stats_fielding(filepath: str) -> dict:
    """Recognize the league-wide TEAM fielding stats export without ingesting.

    The `team_statistics___info_-_sortable_stats_fielding_stats.csv` export
    has columns (Team Name, G, IP, PO, A, DP, E, PCT, ZR) — one row per
    team, not per player. This is a completely different schema from the
    player-level `fielding_stats.csv` handler, which expects a Name column
    and per-player defensive ratings.

    The longer FILE_PATTERNS entry (`team_statistics___info_-_sortable_
    stats_fielding_stats`) wins the longest-match in identify_file_type,
    which routes the file here instead of ingest_fielding_stats. We
    RECOGNIZE it so the refresh UI doesn't crash or flag it as unknown,
    but skip ingestion until a team-defense feature wants it — at that
    point this becomes a real handler writing into league_team_stats
    (which already has fielding columns for team_stats_batting/pitching
    sibling handlers).
    """
    return {
        "status": "skipped",
        "file_type": "team_stats_fielding",
        "rows": 0,
        "reason": "team-level fielding columns not yet consumed — recognized but not ingested",
    }
