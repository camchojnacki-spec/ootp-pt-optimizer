"""CSV ingestion engine — parses CSVs and writes to SQLite."""
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

from app.utils.constants import POSITION_MAP, PITCHER_ROLE_MAP, TIER_MAP
from app.utils.csv_parser import (
    parse_market_csv, parse_roster_batting_csv, parse_roster_pitching_csv,
    parse_collection_batting_csv, parse_collection_pitching_csv,
    parse_stats_batting_csv, parse_stats_pitching_csv, identify_file_type,
    parse_stats_batting_adv_csv, parse_stats_pitching_adv_csv,
    parse_lineup_csv, parse_team_pitching_csv,
    parse_league_batting_ratings_csv, parse_league_pitching_ratings_csv,
)
from app.core.database import get_connection, load_config
from app.core.meta_scoring import calc_batting_meta, calc_pitching_meta, calc_defense_score


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


def ingest_file(filepath: str) -> dict:
    """Auto-detect file type and ingest. Returns dict with status info."""
    file_type = identify_file_type(filepath)
    if file_type is None:
        return {"status": "skipped", "reason": "Unknown file type", "file": filepath}

    handlers = {
        "market": ingest_market_data,
        "roster_batting": ingest_roster_batting,
        "roster_pitching": ingest_roster_pitching,
        "collection_batting": ingest_collection_batting,
        "collection_pitching": ingest_collection_pitching,
        "stats_batting": ingest_stats_batting,
        "stats_pitching": ingest_stats_pitching,
        "roster_batting_stats": ingest_roster_batting_stats,
        "roster_pitching_stats": ingest_roster_pitching_stats,
        "stats_batting_ratings": ingest_league_batting_ratings,
        "stats_pitching_ratings": ingest_league_pitching_ratings,
        "lineup_vs_rhp": ingest_lineup,
        "lineup_vs_lhp": ingest_lineup,
        "lineup_overview": ingest_lineup,
        "team_pitching": ingest_team_pitching,
    }

    handler = handlers[file_type]
    # Some handlers need the file_type (e.g. lineup variants)
    import inspect
    sig = inspect.signature(handler)
    if len(sig.parameters) >= 2:
        result = handler(filepath, file_type)
    else:
        result = handler(filepath)

    # Log ingestion
    conn = get_connection()
    conn.execute(
        "INSERT INTO ingestion_log (file_type, file_name, row_count) VALUES (?, ?, ?)",
        (file_type, Path(filepath).name, result.get("rows", 0))
    )
    conn.commit()
    conn.close()

    return result


def ingest_market_data(filepath: str) -> dict:
    """Ingest market card list CSV into cards table and price_snapshots."""
    df = parse_market_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()

    snapshot_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Try to get date from the CSV data
    if 'date' in df.columns and len(df) > 0:
        csv_date = df['date'].iloc[0]
        if pd.notna(csv_date) and str(csv_date).strip():
            snapshot_date = str(csv_date).strip()

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
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        cursor.execute(f"""
            INSERT INTO cards ({col_names}) VALUES ({placeholders})
            ON CONFLICT(card_id) DO UPDATE SET
                card_title=excluded.card_title, card_value=excluded.card_value,
                owned=excluded.owned, meta_score_batting=excluded.meta_score_batting,
                meta_score_pitching=excluded.meta_score_pitching,
                buy_order_high=excluded.buy_order_high, sell_order_low=excluded.sell_order_low,
                last_10_price=excluded.last_10_price, last_10_variance=excluded.last_10_variance,
                last_updated_at=excluded.last_updated_at
        """, values)

        # Insert price snapshot (deduplicate by card_id + date)
        cursor.execute("""
            INSERT OR REPLACE INTO price_snapshots (card_id, snapshot_date, buy_order_high, sell_order_low, last_10_price, last_10_variance)
            SELECT ?, ?, ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM price_snapshots
                WHERE card_id = ? AND snapshot_date = ?
            )
        """, (
            card_id, snapshot_date,
            _safe_int(row.get('Buy Order High', 0)), _safe_int(row.get('Sell Order Low', 0)),
            _safe_int(row.get('Last 10 Price', 0)), _safe_int(row.get('Last 10 Price(VAR)', 0)),
            card_id, snapshot_date
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

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        pos = _safe_str(row.get('POS', '')).strip()
        ovr = _safe_int(row.get('OVR', 0))
        status = _safe_str(row.get('St', '')).strip()

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

        cursor.execute("""
            INSERT INTO roster (player_name, position, lineup_role, ovr, meta_score,
                meta_vs_rhp, meta_vs_lhp, con_vl, pow_vl, eye_vl, con_vr, pow_vr, eye_vr, bats)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, pos, lineup_role, ovr, meta, meta_rhp, meta_lhp,
              _safe_int(row.get('CON vL', 0)), _safe_int(row.get('POW vL', 0)),
              _safe_int(row.get('EYE vL', 0)), _safe_int(row.get('CON vR', 0)),
              _safe_int(row.get('POW vR', 0)), _safe_int(row.get('EYE vR', 0)),
              _safe_str(row.get('B', '')).strip()))

        # Try to find card_id by matching name in cards table
        card_row = cursor.execute(
            "SELECT card_id FROM cards WHERE (first_name || ' ' || last_name) = ? OR card_title LIKE ? LIMIT 1",
            (name, f"%{name}%")
        ).fetchone()
        card_id = card_row[0] if card_row else None

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

        cursor.execute("""
            INSERT INTO roster (player_name, position, lineup_role, ovr, meta_score,
                meta_vs_rhp, meta_vs_lhp, stu_vl, stu_vr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, pos, lineup_role, ovr, meta, meta_vs_rhb, meta_vs_lhb,
              _safe_int(row.get('STU vL', 0)), _safe_int(row.get('STU vR', 0))))

        card_row = cursor.execute(
            "SELECT card_id FROM cards WHERE (first_name || ' ' || last_name) = ? OR card_title LIKE ? LIMIT 1",
            (name, f"%{name}%")
        ).fetchone()
        card_id = card_row[0] if card_row else None

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

        card_row = cursor.execute(
            "SELECT card_id FROM cards WHERE (first_name || ' ' || last_name) = ? OR card_title LIKE ? LIMIT 1",
            (name, f"%{name}%")
        ).fetchone()
        card_id = card_row[0] if card_row else None

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

        card_row = cursor.execute(
            "SELECT card_id FROM cards WHERE (first_name || ' ' || last_name) = ? OR card_title LIKE ? LIMIT 1",
            (name, f"%{name}%")
        ).fetchone()
        card_id = card_row[0] if card_row else None

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


def _match_card_id(cursor, player_name: str) -> int | None:
    """Try to find a card_id by matching player name in cards table."""
    card_row = cursor.execute(
        "SELECT card_id FROM cards WHERE (first_name || ' ' || last_name) = ? OR card_title LIKE ? LIMIT 1",
        (player_name, f"%{player_name}%")
    ).fetchone()
    return card_row[0] if card_row else None


def ingest_stats_batting(filepath: str) -> dict:
    """Ingest sortable batting stats CSV into batting_stats table.

    Each import replaces current stats (latest snapshot) and preserves history
    by checking for an existing snapshot from the same date.
    Detects stats_2 format (wOBA, BB%, K%) and routes to advanced handler.
    """
    df = parse_stats_batting_csv(filepath)
    cols = set(df.columns)
    # Detect stats_2 format
    if 'wOBA' in cols or 'BB%' in cols:
        return ingest_batting_stats_adv(filepath)

    conn = get_connection()
    cursor = conn.cursor()

    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    # Delete any existing snapshot from today (re-import overwrites)
    cursor.execute("DELETE FROM batting_stats WHERE DATE(snapshot_date) = ?", (snapshot_date,))

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
            INSERT INTO batting_stats (
                player_name, position, bats, throws,
                games, pa, ab, hits, doubles, triples, hr, rbi, runs,
                bb, ibb, hbp, k, gidp,
                avg, obp, slg, iso, ops, ops_plus, babip, war,
                sb, cs, card_id, snapshot_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "stats_batting", "rows": count}


def ingest_stats_pitching(filepath: str) -> dict:
    """Ingest sortable pitching stats CSV into pitching_stats table.

    Each import replaces current stats (latest snapshot) and preserves history.
    Detects stats_2 format (SIERA, WIN%) and routes to advanced handler.
    """
    df = parse_stats_pitching_csv(filepath)
    cols = set(df.columns)
    if 'SIERA' in cols or 'WIN%' in cols:
        return ingest_pitching_stats_adv(filepath)

    conn = get_connection()
    cursor = conn.cursor()

    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    # Delete any existing snapshot from today (re-import overwrites)
    cursor.execute("DELETE FROM pitching_stats WHERE DATE(snapshot_date) = ?", (snapshot_date,))

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
            INSERT INTO pitching_stats (
                player_name, position, bats, throws,
                games, gs, wins, losses, saves, holds,
                ip, hits_allowed, hr_allowed, runs_allowed, er,
                bb, k, hbp,
                era, avg_against, babip, whip,
                hr_per_9, bb_per_9, k_per_9, k_per_bb,
                era_plus, fip, war,
                card_id, snapshot_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "stats_pitching", "rows": count}


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
    """Shared logic for ingesting standard batting stats (league or roster)."""
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
        card_id = _match_card_id(cursor, name)

        # Upsert: if player already has a record for today, update it
        existing = cursor.execute(
            "SELECT id FROM batting_stats WHERE player_name = ? AND DATE(snapshot_date) = ?",
            (name, snapshot_date)
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
                    sb, cs, card_id, snapshot_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    """Shared logic for ingesting standard pitching stats (league or roster)."""
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
        card_id = _match_card_id(cursor, name)

        existing = cursor.execute(
            "SELECT id FROM pitching_stats WHERE player_name = ? AND DATE(snapshot_date) = ?",
            (name, snapshot_date)
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
                    card_id, snapshot_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            INSERT INTO roster (player_name, position, lineup_role, ovr, meta_score,
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
            INSERT INTO roster (player_name, position, lineup_role, ovr, meta_score,
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

        # Also update the roster table with card_id and card_title if we have a match
        if card_id and title:
            cursor.execute("""
                UPDATE roster SET card_id = ?, card_title = ?
                WHERE player_name = ? AND lineup_role != 'league'
            """, (card_id, title, name))

        cursor.execute("""
            INSERT INTO team_lineup (lineup_type, position, player_name, card_title,
                card_id, ovr, bats, throws, age, slot_order)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (lineup_type, pos, name, title, card_id, ovr, bats, throws, age, idx))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": file_type, "rows": count}


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
                WHERE player_name = ? AND lineup_role != 'league'
            """, (card_id, title, name))

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
    """Ingest batting stats_2 CSV (advanced metrics: wOBA, BB%, K%, WPA, etc.)."""
    df = parse_stats_batting_adv_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()
    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("DELETE FROM batting_stats_adv WHERE DATE(snapshot_date) = ?", (snapshot_date,))

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name or name.lower() in ('total', 'totals', 'team'):
            continue
        card_id = _match_card_id(cursor, name)

        cursor.execute("""
            INSERT INTO batting_stats_adv (
                player_name, position, bats, throws,
                games, pa, bb, bb_pct, sh, sf, ci, k, k_pct, gidp,
                ebh, tb, rc, rc27, iso, woba, wpa, pi_pa,
                card_id, snapshot_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            card_id, snapshot_date,
        ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "batting_stats_adv", "rows": count}


def ingest_pitching_stats_adv(filepath: str) -> dict:
    """Ingest pitching stats_2 CSV (advanced metrics: SIERA, WPA, QS%, etc.)."""
    df = parse_stats_pitching_adv_csv(filepath)
    conn = get_connection()
    cursor = conn.cursor()
    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("DELETE FROM pitching_stats_adv WHERE DATE(snapshot_date) = ?", (snapshot_date,))

    count = 0
    for _, row in df.iterrows():
        name = _safe_str(row.get('Name', '')).strip()
        if not name or name.lower() in ('total', 'totals', 'team'):
            continue
        card_id = _match_card_id(cursor, name)

        cursor.execute("""
            INSERT INTO pitching_stats_adv (
                player_name, position, bats, throws,
                games, win_pct, sv_pct, bs, sd, md, ip, bf, dp, ra, gf,
                ir, irs_pct, pli, qs, qs_pct, cg, cg_pct, sho,
                ppg, rsg, go_pct, siera, sb, cs, wpa,
                card_id, snapshot_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            card_id, snapshot_date,
        ))
        count += 1

    conn.commit()
    conn.close()
    return {"status": "success", "file_type": "pitching_stats_adv", "rows": count}
