"""Meta score calculations for batters and pitchers."""
import json
import math
from app.core.database import load_config, get_db_path
from app.utils.constants import (
    DEFAULT_BATTING_WEIGHTS, DEFAULT_PITCHING_WEIGHTS,
    PITCHING_STAT_FLOOR, BATTING_STAT_FLOOR,
    POSITION_DEFENSE_MULTIPLIERS,
)

# Diminishing returns threshold — stats above this get sqrt-scaled benefit
DIMINISHING_RETURNS_THRESHOLD = 110


def _diminished(value: float, threshold: float = DIMINISHING_RETURNS_THRESHOLD) -> float:
    """Apply diminishing returns to stats above the threshold.

    Below threshold: linear (1:1).
    Above threshold: sqrt-scaled so that extreme spikes don't dominate.
    e.g. 150 -> 110 + sqrt(40)*4 ≈ 135 (effective), not raw 150.
    """
    if value <= threshold:
        return value
    excess = value - threshold
    return threshold + math.sqrt(excess) * 4


def _load_calibrated_weights() -> tuple:
    """Try to load calibrated weights from the meta_calibration DB table.

    Returns (batting_weights, pitching_weights) or (None, None) if unavailable.
    """
    import sqlite3 as _sqlite3
    try:
        db_path = get_db_path()
        conn = _sqlite3.connect(db_path)
        cursor = conn.cursor()
        bw, pw = None, None
        for cal_type, target in [('batting', 'bw'), ('pitching', 'pw')]:
            cursor.execute(
                "SELECT weights_json FROM meta_calibration "
                "WHERE calibration_type = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (cal_type,),
            )
            row = cursor.fetchone()
            if row and row[0]:
                parsed = json.loads(row[0])
                if cal_type == 'batting':
                    bw = parsed
                else:
                    pw = parsed
        conn.close()
        return bw, pw
    except Exception:
        return None, None


def get_weights():
    """Load weights: calibrated DB weights > config.yaml > defaults."""
    # 1. Try calibrated weights from DB
    cal_bw, cal_pw = _load_calibrated_weights()
    if cal_bw is not None and cal_pw is not None:
        return cal_bw, cal_pw

    # 2. Fall back to config.yaml
    try:
        config = load_config()
        bw = cal_bw or config.get('batting_weights', DEFAULT_BATTING_WEIGHTS)
        pw = cal_pw or config.get('pitching_weights', DEFAULT_PITCHING_WEIGHTS)
    except Exception:
        bw = cal_bw or DEFAULT_BATTING_WEIGHTS
        pw = cal_pw or DEFAULT_PITCHING_WEIGHTS
    return bw, pw


def get_weights_with_source() -> tuple:
    """Load weights and return (batting_weights, pitching_weights, source_label).

    source_label is one of: 'calibrated', 'config', 'default'
    """
    # 1. Try calibrated weights from DB
    cal_bw, cal_pw = _load_calibrated_weights()
    if cal_bw is not None and cal_pw is not None:
        return cal_bw, cal_pw, 'calibrated'

    # 2. Try config.yaml
    try:
        config = load_config()
        bw = config.get('batting_weights')
        pw = config.get('pitching_weights')
        if bw is not None and pw is not None:
            return (cal_bw or bw), (cal_pw or pw), 'config'
        # Partial: mix calibrated + config + defaults
        bw = cal_bw or bw or DEFAULT_BATTING_WEIGHTS
        pw = cal_pw or pw or DEFAULT_PITCHING_WEIGHTS
        return bw, pw, 'config'
    except Exception:
        pass

    # 3. Defaults
    return (cal_bw or DEFAULT_BATTING_WEIGHTS), (cal_pw or DEFAULT_PITCHING_WEIGHTS), 'default'


def calc_defense_score(row: dict, apply_position_multiplier: bool = True) -> float:
    """Calculate defense component from card data.

    When apply_position_multiplier is True (default), the raw defensive average
    is scaled by POSITION_DEFENSE_MULTIPLIERS so that SS/CF/C defense counts
    much more than 1B/DH defense.  Pass False to get the raw average.
    """
    # Determine position type and calc appropriate defense average
    pos = row.get('position') or row.get('Position') or 0
    if isinstance(pos, str):
        # Map string positions
        pos_map = {"C": 2, "1B": 3, "2B": 4, "3B": 5, "SS": 6, "LF": 7, "CF": 8, "RF": 9, "DH": 10, "P": 1}
        pos = pos_map.get(pos, 0)

    try:
        pos = int(pos)
    except (ValueError, TypeError):
        pos = 0

    raw = 0.0
    if pos in (7, 8, 9):  # OF
        vals = [row.get('of_range', 0) or 0, row.get('of_error', 0) or 0, row.get('of_arm', 0) or 0]
        # Also check alternate column names
        if not any(vals):
            vals = [row.get('OF Range', 0) or 0, row.get('OF Error', 0) or 0, row.get('OF Arm', 0) or 0]
        raw = sum(vals) / max(len([v for v in vals if v]), 1)
    elif pos == 2:  # C
        vals = [row.get('catcher_ability', 0) or 0, row.get('catcher_frame', 0) or 0, row.get('catcher_arm', 0) or 0]
        if not any(vals):
            vals = [row.get('CatcherAbil', 0) or 0, row.get('CatcherFrame', 0) or 0, row.get('Catcher Arm', 0) or 0]
        raw = sum(vals) / max(len([v for v in vals if v]), 1)
    elif pos in (3, 4, 5, 6):  # IF
        vals = [row.get('infield_range', 0) or 0, row.get('infield_error', 0) or 0, row.get('infield_arm', 0) or 0]
        if not any(vals):
            vals = [row.get('Infield Range', 0) or 0, row.get('Infield Error', 0) or 0, row.get('Infield Arm', 0) or 0]
        raw = sum(vals) / max(len([v for v in vals if v]), 1)

    if not apply_position_multiplier:
        return raw

    # Scale by position importance: SS defense is worth 1.4x, 1B is 0.5x, etc.
    multiplier = POSITION_DEFENSE_MULTIPLIERS.get(pos, 1.0)
    return raw * multiplier


def calc_speed_score(row: dict) -> float:
    """Calculate speed/baserunning component from card data.

    Combines Speed, Stealing, and Baserunning ratings.
    Speed alone is conditionally valuable — it only matters if you get on base.
    We weight Stealing highest (directly produces SB/runs), then Baserunning
    (extra bases taken), then Speed (raw tool).
    """
    speed = float(row.get('speed') or row.get('Speed') or row.get('SPD') or 0)
    stealing = float(row.get('stealing') or row.get('Stealing') or row.get('STL') or 0)
    baserunning = float(row.get('baserunning') or row.get('Baserunning') or row.get('BR') or 0)

    if speed <= 0 and stealing <= 0 and baserunning <= 0:
        return 0.0

    # Weighted composite: Stealing 40%, Baserunning 35%, Speed 25%
    # Stealing is most directly predictive of SB (r=+0.344)
    composite = (stealing * 0.40 + baserunning * 0.35 + speed * 0.25)

    # Only count speed bonus above average (70) — slow players shouldn't be
    # penalized since the batting stats already capture their output
    if composite < 70:
        return 0.0

    return composite - 70  # Excess above average baseline


def calc_batting_meta(row: dict, weights: dict = None) -> float:
    """Calculate batter meta score.

    Includes OVR anchoring, speed/stealing value, and position-specific
    defense scaling so the game's own evaluation is factored in.
    """
    if weights is None:
        weights, _ = get_weights()

    # Support both DB column names and CSV column names
    gap = float(row.get('gap_power') or row.get('Gap') or row.get('GAP') or 0)
    con = float(row.get('contact') or row.get('Contact') or row.get('CON') or 0)
    avk = float(row.get('avoid_ks') or row.get('Avoid Ks') or row.get("K's") or 0)
    eye = float(row.get('eye') or row.get('Eye') or row.get('EYE') or 0)
    pwr = float(row.get('power') or row.get('Power') or row.get('POW') or 0)
    bab = float(row.get('babip') or row.get('BABIP') or 0)
    defense = float(row.get('defense_score') or calc_defense_score(row))
    ovr = float(row.get('card_value') or row.get('OVR') or row.get('ovr') or 0)
    speed_score = float(row.get('speed_score') or calc_speed_score(row))

    try:
        # Core weighted sum — AvK and BABIP zeroed by default since CON is
        # a derived stat in OOTP25+ that already incorporates them.
        meta = (_diminished(gap) * weights.get('gap_power', 1.40) +
                _diminished(con) * weights.get('contact', 1.80) +
                _diminished(avk) * weights.get('avoid_ks', 0.00) +
                _diminished(eye) * weights.get('eye', 0.60) +
                _diminished(pwr) * weights.get('power', 1.40) +
                _diminished(bab) * weights.get('babip', 0.00) +
                defense * weights.get('defense', 1.50))

        # Speed/Stealing component — conditionally valuable (amplifies OBP)
        # Only counts above-average speed; elite speedsters get a real boost
        spd_weight = weights.get('speed_stealing', 0.50)
        if speed_score > 0 and spd_weight > 0:
            meta += _diminished(speed_score + 70) * spd_weight  # re-add baseline for diminishing calc

        # Balance penalty — if any key batting stat is below floor
        floor = BATTING_STAT_FLOOR
        key_stats = [con, gap]
        for stat in key_stats:
            if 0 < stat < floor:
                penalty = (floor - stat) * 0.4
                meta -= penalty

        # OVR multiplier — data shows OVR->WAR r=+0.529 for batters
        ovr_weight = weights.get('ovr', 1.25)
        if ovr > 0 and ovr_weight > 0:
            ovr_factor = (ovr / 80.0) ** (ovr_weight * 0.35)
            meta *= ovr_factor

    except (ValueError, TypeError):
        meta = 0.0

    return round(meta, 2)


def calc_pitching_meta(row: dict, weights: dict = None) -> float:
    """Calculate pitcher meta score.

    Includes OVR anchoring and balance penalty so that one-trick-pony
    cards with extreme single-stat spikes don't outscore well-rounded arms.
    """
    if weights is None:
        _, weights = get_weights()

    mov = float(row.get('movement') or row.get('Movement') or row.get('MOV') or 0)
    stu = float(row.get('stuff') or row.get('Stuff') or row.get('STU') or 0)
    ctrl = float(row.get('control') or row.get('Control') or row.get('CON') or 0)
    phr = float(row.get('p_hr') or row.get('pHR') or row.get('HRA') or 0)
    ovr = float(row.get('card_value') or row.get('OVR') or row.get('ovr') or 0)
    stamina = float(row.get('stamina') or row.get('Stamina') or row.get('STA') or 0)
    hold = float(row.get('hold') or row.get('Hold') or 0)

    try:
        # Core ratings with diminishing returns on extreme values
        # League data: MOV r=-0.295 ERA, STU r=-0.265, CTRL r=-0.002, HRA r=-0.266
        meta = (_diminished(mov) * weights.get('movement', 2.40) +
                _diminished(stu) * weights.get('stuff', 1.40) +
                _diminished(ctrl) * weights.get('control', 0.20) +
                _diminished(phr) * weights.get('p_hr', 1.80))

        # Stamina/Hold component (matters for relievers and starters alike)
        sh_weight = weights.get('stamina_hold', 0.30)
        if sh_weight > 0:
            sh_avg = 0
            sh_count = 0
            if stamina > 0:
                sh_avg += stamina
                sh_count += 1
            if hold > 0:
                sh_avg += hold
                sh_count += 1
            if sh_count > 0:
                meta += (sh_avg / sh_count) * sh_weight

        # Balance penalty — only penalize truly weak ratings (sub-65)
        # Data shows CTRL below 75 doesn't hurt ERA much in practice,
        # so we use a softer floor and lighter penalty
        floor = 65
        key_stats = [stu, mov]  # Only penalize STU/MOV weakness, not CTRL
        for stat in key_stats:
            if 0 < stat < floor:
                shortfall = floor - stat
                penalty = shortfall * 1.0
                meta -= penalty

        # OVR multiplier — data shows OVR->WAR r=+0.601 for pitchers
        # Stronger than community suggested but validated by league data
        # OVR 80 = neutral (1.0x), OVR 100 = ~1.17x at weight=1.5
        ovr_weight = weights.get('ovr', 1.50)
        if ovr > 0 and ovr_weight > 0:
            ovr_factor = (ovr / 80.0) ** (ovr_weight * 0.35)
            meta *= ovr_factor

    except (ValueError, TypeError):
        meta = 0.0

    return round(meta, 2)


def calc_batting_meta_vs_rhp(row: dict, weights: dict = None) -> float:
    """Calculate batter meta score vs right-handed pitching.

    Uses the batter's 'vR' split ratings where available, falling back to overall.
    When facing RHP, OOTP uses the batter's CON vR, POW vR, EYE vR ratings.
    """
    if weights is None:
        weights, _ = get_weights()

    # Use vR splits where available, fall back to overall
    con = float(row.get('con_vr') or row.get('CON vR') or row.get('contact_vr') or
                row.get('contact') or row.get('CON') or 0)
    pwr = float(row.get('pow_vr') or row.get('POW vR') or row.get('power_vr') or
                row.get('power') or row.get('POW') or 0)
    eye = float(row.get('eye_vr') or row.get('EYE vR') or row.get('eye_vr') or
                row.get('eye') or row.get('EYE') or 0)
    # GAP and BABIP don't have splits in roster CSV, use overall
    gap = float(row.get('gap_power') or row.get('Gap') or row.get('GAP') or 0)
    avk = float(row.get('avoid_ks') or row.get('Avoid Ks') or row.get("K's") or 0)
    bab = float(row.get('babip') or row.get('BABIP') or 0)
    defense = float(row.get('defense_score') or calc_defense_score(row))
    ovr = float(row.get('card_value') or row.get('OVR') or row.get('ovr') or 0)
    speed_score = float(row.get('speed_score') or calc_speed_score(row))

    try:
        meta = (_diminished(gap) * weights.get('gap_power', 1.40) +
                _diminished(con) * weights.get('contact', 1.80) +
                _diminished(avk) * weights.get('avoid_ks', 0.00) +
                _diminished(eye) * weights.get('eye', 0.60) +
                _diminished(pwr) * weights.get('power', 1.40) +
                _diminished(bab) * weights.get('babip', 0.00) +
                defense * weights.get('defense', 1.50))

        spd_weight = weights.get('speed_stealing', 0.50)
        if speed_score > 0 and spd_weight > 0:
            meta += _diminished(speed_score + 70) * spd_weight

        floor = BATTING_STAT_FLOOR
        key_stats = [con, gap]
        for stat in key_stats:
            if 0 < stat < floor:
                penalty = (floor - stat) * 0.4
                meta -= penalty

        ovr_weight = weights.get('ovr', 1.25)
        if ovr > 0 and ovr_weight > 0:
            ovr_factor = (ovr / 80.0) ** (ovr_weight * 0.35)
            meta *= ovr_factor
    except (ValueError, TypeError):
        meta = 0.0

    return round(meta, 2)


def calc_batting_meta_vs_lhp(row: dict, weights: dict = None) -> float:
    """Calculate batter meta score vs left-handed pitching.

    Uses the batter's 'vL' split ratings where available, falling back to overall.
    When facing LHP, OOTP uses the batter's CON vL, POW vL, EYE vL ratings.
    """
    if weights is None:
        weights, _ = get_weights()

    con = float(row.get('con_vl') or row.get('CON vL') or row.get('contact_vl') or
                row.get('contact') or row.get('CON') or 0)
    pwr = float(row.get('pow_vl') or row.get('POW vL') or row.get('power_vl') or
                row.get('power') or row.get('POW') or 0)
    eye = float(row.get('eye_vl') or row.get('EYE vL') or row.get('eye_vl') or
                row.get('eye') or row.get('EYE') or 0)
    gap = float(row.get('gap_power') or row.get('Gap') or row.get('GAP') or 0)
    avk = float(row.get('avoid_ks') or row.get('Avoid Ks') or row.get("K's") or 0)
    bab = float(row.get('babip') or row.get('BABIP') or 0)
    defense = float(row.get('defense_score') or calc_defense_score(row))
    ovr = float(row.get('card_value') or row.get('OVR') or row.get('ovr') or 0)
    speed_score = float(row.get('speed_score') or calc_speed_score(row))

    try:
        meta = (_diminished(gap) * weights.get('gap_power', 1.40) +
                _diminished(con) * weights.get('contact', 1.80) +
                _diminished(avk) * weights.get('avoid_ks', 0.00) +
                _diminished(eye) * weights.get('eye', 0.60) +
                _diminished(pwr) * weights.get('power', 1.40) +
                _diminished(bab) * weights.get('babip', 0.00) +
                defense * weights.get('defense', 1.50))

        spd_weight = weights.get('speed_stealing', 0.50)
        if speed_score > 0 and spd_weight > 0:
            meta += _diminished(speed_score + 70) * spd_weight

        floor = BATTING_STAT_FLOOR
        key_stats = [con, gap]
        for stat in key_stats:
            if 0 < stat < floor:
                penalty = (floor - stat) * 0.4
                meta -= penalty

        ovr_weight = weights.get('ovr', 1.25)
        if ovr > 0 and ovr_weight > 0:
            ovr_factor = (ovr / 80.0) ** (ovr_weight * 0.35)
            meta *= ovr_factor
    except (ValueError, TypeError):
        meta = 0.0

    return round(meta, 2)


def calc_pitching_meta_vs_lhb(row: dict, weights: dict = None) -> float:
    """Calculate pitcher meta score vs left-handed batters.

    Uses STU vL split where available. MOV/CON(ctrl)/HRA don't have splits in roster CSV.
    """
    if weights is None:
        _, weights = get_weights()

    stu = float(row.get('stu_vl') or row.get('STU vL') or row.get('stuff_vl') or
                row.get('stuff') or row.get('STU') or 0)
    mov = float(row.get('movement') or row.get('Movement') or row.get('MOV') or 0)
    ctrl = float(row.get('control') or row.get('Control') or row.get('CON') or 0)
    phr = float(row.get('p_hr') or row.get('pHR') or row.get('HRA') or 0)
    ovr = float(row.get('card_value') or row.get('OVR') or row.get('ovr') or 0)
    stamina = float(row.get('stamina') or row.get('Stamina') or row.get('STA') or 0)
    hold = float(row.get('hold') or row.get('Hold') or 0)

    try:
        meta = (_diminished(mov) * weights.get('movement', 2.40) +
                _diminished(stu) * weights.get('stuff', 1.40) +
                _diminished(ctrl) * weights.get('control', 0.20) +
                _diminished(phr) * weights.get('p_hr', 1.80))

        sh_weight = weights.get('stamina_hold', 0.30)
        if sh_weight > 0:
            sh_avg, sh_count = 0, 0
            if stamina > 0: sh_avg += stamina; sh_count += 1
            if hold > 0: sh_avg += hold; sh_count += 1
            if sh_count > 0: meta += (sh_avg / sh_count) * sh_weight

        floor = 65
        key_stats = [stu, mov]
        for stat in key_stats:
            if 0 < stat < floor:
                meta -= (floor - stat) * 1.0

        ovr_weight = weights.get('ovr', 1.50)
        if ovr > 0 and ovr_weight > 0:
            meta *= (ovr / 80.0) ** (ovr_weight * 0.35)
    except (ValueError, TypeError):
        meta = 0.0

    return round(meta, 2)


def calc_pitching_meta_vs_rhb(row: dict, weights: dict = None) -> float:
    """Calculate pitcher meta score vs right-handed batters.

    Uses STU vR split where available. MOV/CON(ctrl)/HRA don't have splits in roster CSV.
    """
    if weights is None:
        _, weights = get_weights()

    stu = float(row.get('stu_vr') or row.get('STU vR') or row.get('stuff_vr') or
                row.get('stuff') or row.get('STU') or 0)
    mov = float(row.get('movement') or row.get('Movement') or row.get('MOV') or 0)
    ctrl = float(row.get('control') or row.get('Control') or row.get('CON') or 0)
    phr = float(row.get('p_hr') or row.get('pHR') or row.get('HRA') or 0)
    ovr = float(row.get('card_value') or row.get('OVR') or row.get('ovr') or 0)
    stamina = float(row.get('stamina') or row.get('Stamina') or row.get('STA') or 0)
    hold = float(row.get('hold') or row.get('Hold') or 0)

    try:
        meta = (_diminished(mov) * weights.get('movement', 2.40) +
                _diminished(stu) * weights.get('stuff', 1.40) +
                _diminished(ctrl) * weights.get('control', 0.20) +
                _diminished(phr) * weights.get('p_hr', 1.80))

        sh_weight = weights.get('stamina_hold', 0.30)
        if sh_weight > 0:
            sh_avg, sh_count = 0, 0
            if stamina > 0: sh_avg += stamina; sh_count += 1
            if hold > 0: sh_avg += hold; sh_count += 1
            if sh_count > 0: meta += (sh_avg / sh_count) * sh_weight

        floor = 65
        key_stats = [stu, mov]
        for stat in key_stats:
            if 0 < stat < floor:
                meta -= (floor - stat) * 1.0

        ovr_weight = weights.get('ovr', 1.50)
        if ovr > 0 and ovr_weight > 0:
            meta *= (ovr / 80.0) ** (ovr_weight * 0.35)
    except (ValueError, TypeError):
        meta = 0.0

    return round(meta, 2)
