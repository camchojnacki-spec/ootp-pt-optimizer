"""Roster Optimizer — lineup card view with AI team assessment."""
import streamlit as st
import pandas as pd
import sys
from bisect import bisect_right
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection, load_config, get_data_freshness
from app.core.meta_validation import get_meta_confidence
from app.core.meta_scoring import explain_meta
from app.core.ai_advisor import (
    get_ai_config, get_upgrade_scouting_report, build_team_context, get_full_card_data,
    ai_optimize_all_positions, get_unified_roster_analysis,
)
from app.core.position_eligibility import (
    POS_RATING_COL,
    build_eligible_where_clause,
    select_rating_columns,
    position_meta_penalty,
    format_position_annotation,
    is_eligible,
)
from app.utils.sidebar_nav import render_sidebar_nav


def _fetch_card_for_explainer(_conn, card_id):
    """Pull the rating columns ``explain_meta`` needs for a given card_id."""
    if not card_id:
        return None
    row = _conn.execute(
        """
        SELECT card_id, card_title, position, position_name, pitcher_role, pitcher_role_name,
               tier_name,
               contact, gap_power, power, eye, avoid_ks, babip,
               speed, stealing, baserunning,
               of_range, of_error, of_arm,
               catcher_ability, catcher_frame, catcher_arm,
               infield_range, infield_error, infield_arm,
               stuff, movement, control, p_hr, p_babip, stamina, hold
        FROM cards WHERE card_id = ?
        """,
        (card_id,),
    ).fetchone()
    return dict(row) if row else None


def _render_meta_explainer(exp: dict):
    """Render explain_meta() output. Same shape as Buy/Sell pages."""
    if not exp:
        st.caption("(no explainer available — card data is missing the needed ratings)")
        return
    ovr = (exp.get('diagnostics') or {}).get('ovr')
    if ovr:
        st.caption(
            f"Total breakdown: **{exp['total']:.0f}** meta · OOTP OVR **{ovr}** "
            "(comparison only — not a formula input). "
            "Components sorted by impact."
        )
    else:
        st.caption(f"Total breakdown: **{exp['total']:.0f}** meta — components sorted by impact.")
    if exp.get('components'):
        comp_df = pd.DataFrame([
            {"Rating": c['label'], "Raw": c['raw'],
             "Weight": c['weight'], "Points": c['points']}
            for c in exp['components']
        ])
        st.dataframe(
            comp_df, use_container_width=True, hide_index=True,
            column_config={
                "Raw": st.column_config.NumberColumn(format="%.0f", width="small"),
                "Weight": st.column_config.NumberColumn(format="%.2f", width="small"),
                "Points": st.column_config.NumberColumn(format="%+.1f", width="small"),
            },
        )
    if exp.get('bonuses'):
        st.markdown("**Bonuses**")
        for b in exp['bonuses']:
            st.markdown(f"\u2022 {b['label']} \u2192 **{b['points']:+.1f}**")
    if exp.get('penalties'):
        st.markdown("**Penalties**")
        for p in exp['penalties']:
            st.markdown(f"\u2022 {p['label']} \u2192 **{p['points']:+.1f}**")
    if exp.get('notes'):
        st.markdown("---")
        for n in exp['notes']:
            st.caption(f"\U0001f4dd {n}")


def _meta_confidence_chip(category: str, display_name: str | None = None) -> str:
    """Render a compact meta-confidence chip (HTML) for a category.

    Uses the latest calibration run. The chip tells users how much to trust
    the meta ordering at a glance — see the review's P0 ask that every
    meta-sorted view surface its observed predictive power.

    ``category`` matches a ``meta_calibration.calibration_type`` row:
    'batting', 'pitching' (combined), 'pitching_sp', 'pitching_rp'.
    ``display_name`` optionally overrides the chip label (e.g. 'SP' or
    'RP' when rendering the two pitching chips side-by-side so the user
    can tell them apart without a tooltip)."""
    c = get_meta_confidence(category, conn)
    tip = (
        f"Meta vs observed performance: r={c['correlation']}, "
        f"R\u00b2={c['r_squared']}, n={c['sample_size']}. "
        f"Run /Game_Stats \u2192 Meta vs Reality to recalibrate."
    )
    prefix = f"{display_name} " if display_name else ""
    return (
        f'<span title="{tip}" style="display:inline-block; padding:2px 8px; '
        f'border-radius:10px; background:{c["color"]}; color:#fff; '
        f'font-size:0.78em; font-weight:600; margin-left:8px;">'
        f'{c["emoji"]} {prefix}Confidence: {c["label"]}</span>'
    )

st.set_page_config(page_title="Roster Optimizer", page_icon="\U0001f4cb", layout="wide")
render_sidebar_nav()

conn = get_connection()  # auto-syncs roster.meta_score from cards on first call
config = load_config()
budget = config.get('pp_budget', 500)

# ── Archetype / fit layer (project_attribute_mix_2026_04_20) ──
# `card_archetypes` is rebuilt by the BG worker after each ingest. We use it
# to show a Fit column alongside Meta and to surface mix-aligned replacements
# that Meta under-ranks. Falls back silently if the table isn't populated.
_archetypes_by_card: dict[int, dict] = {}
try:
    for _ar in conn.execute(
        "SELECT card_id, role, fit_score, archetype_name, archetype_war, "
        "mix_score, min_top3, count_elite FROM card_archetypes"
    ).fetchall():
        _archetypes_by_card[_ar['card_id']] = dict(_ar)
except Exception:
    pass


def _fit_for(card_id):
    """Return the archetype record for a card_id, or an empty dict."""
    if not card_id:
        return {}
    return _archetypes_by_card.get(card_id, {}) or {}

# ── Sidebar controls ──
with st.sidebar:
    st.header("Filters")
    max_spend = st.number_input("Max PP per card", min_value=0, max_value=999999,
                                value=budget, step=500, format="%d",
                                help="Maximum PP you're willing to spend on a single card")
    # Threshold defaults to 10 meta points so marginal-but-real upgrades
    # still surface. Bumping up to 25-50 filters to only meaningful moves
    # when the user wants less noise. Previous default of 20 was hiding
    # a lot of legitimate 10-15 point upgrades on pitching slots.
    min_improvement = st.number_input(
        "Min meta improvement", min_value=0, max_value=500,
        value=10, step=5,
        help=("Only show upgrades with at least this much meta gain. "
              "Lower = more suggestions (some marginal), higher = only "
              "the most impactful moves. Default 10 surfaces all "
              "meaningful upgrades; raise to 25-50 for stricter filtering."))
    focus = st.selectbox("Focus", ["All Positions", "Batting Only", "Pitching Only", "Weakest First"])

# ── Build roster data ──
# Active roles = players actually in the game lineup right now
ACTIVE_ROLES = {'starter', 'rotation', 'closer', 'bullpen'}

# Slot-name prefixes that identify pitching slots (SP1..SP5, RP1..RP8, CL, and
# the named bullpen roles SU/MID/LNG/MOP). Used by the perf-driver outperformance
# loop AND by _find_drop_candidate to keep alt drops on the same side (a
# batting upgrade shouldn't suggest dropping a reliever or vice versa — that
# changes the 13/13 roster composition in ways the user probably doesn't want).
_PIT_PREFIXES = ('SP', 'RP', 'CL', 'SU', 'MID', 'LNG', 'MOP')

def _is_pitching_pos(pos_label: str) -> bool:
    """True if a slot label (SP3, RP5, CL, MID, 1B, CF, etc.) is a pitching slot."""
    if not pos_label:
        return False
    clean = pos_label.rstrip(" \u26a0\ufe0f")
    return any(clean.startswith(p) for p in _PIT_PREFIXES)

BATS_MAP = {1: 'R', 2: 'L', 3: 'S'}

roster_rows = conn.execute("""
    SELECT r.player_name, r.position, r.lineup_role, r.ovr, r.meta_score,
           r.meta_vs_rhp, r.meta_vs_lhp, r.bats as roster_bats, c.bats,
           c.card_id, c.card_title, c.position_name as card_position_name,
           c.pos_rating_c, c.pos_rating_1b, c.pos_rating_2b, c.pos_rating_3b,
           c.pos_rating_ss, c.pos_rating_lf, c.pos_rating_cf, c.pos_rating_rf
    FROM roster r
    LEFT JOIN cards c ON c.card_title LIKE '%' || r.player_name || '%'
        AND c.owned = 1
    WHERE r.lineup_role != 'league'
      AND DATE(r.snapshot_date) = (
          SELECT MAX(DATE(snapshot_date)) FROM roster WHERE lineup_role != 'league'
      )
    GROUP BY r.id
    ORDER BY r.position, r.meta_score DESC
""").fetchall()

starters = {}          # best active player per position (for single-slot positions)
active_by_pos = {}     # all ACTIVE players per position (for multi-slot: SP, RP, platoons)
all_by_pos = {}        # every rostered player per position (for exclude lists)
# Flat list of every NON-active (bench/reserve) roster row, with card-level
# pos_rating fields attached. Used by find_roster_bench_upgrades to catch
# multi-position candidates whose card's primary position doesn't match the
# roster slot being evaluated (e.g. Van Haltren on the bench with roster
# position='CF' still appearing as an LF/RF bench upgrade candidate).
bench_pool: list[dict] = []
for r in roster_rows:
    pos = r['position']
    d = dict(r)
    bats_raw = d.get('bats')
    roster_bats = d.get('roster_bats', '')
    # Use roster_bats (text R/L/S) if available, else map numeric from cards table
    if roster_bats and roster_bats in ('R', 'L', 'S'):
        d['bats_hand'] = roster_bats
    else:
        d['bats_hand'] = BATS_MAP.get(int(bats_raw) if bats_raw else 0, '?')
    # Expose the card's natural position under the key the position_eligibility
    # helpers expect (they key on 'position_name'). Roster queries previously
    # named this field card_position_name to avoid clashing with r.position.
    if d.get('card_position_name') and not d.get('position_name'):
        d['position_name'] = d['card_position_name']
    if pos not in all_by_pos:
        all_by_pos[pos] = []
    all_by_pos[pos].append(d)
    if d.get('lineup_role') not in ACTIVE_ROLES:
        bench_pool.append(d)

    if d.get('lineup_role') in ACTIVE_ROLES:
        if pos not in active_by_pos:
            active_by_pos[pos] = []
        active_by_pos[pos].append(d)
        # For single-slot bat positions, pick best active player by meta as
        # a provisional default; we override below with observed-starter
        # data from recent box scores where available.
        if pos not in starters or (d['meta_score'] or 0) > (starters[pos]['meta_score'] or 0):
            starters[pos] = d

# ── Observed-starter override ──
# OOTP's lineup CSV exports the depth chart (sorted by historical games
# played), not the user's currently pinned starter. Recent box scores are
# the source of truth for who's actually starting right now. Override
# ``starters[pos]`` for bat positions when box-score data shows a
# different player pinned in the last few games.
try:
    from app.core.observed_lineup import get_pinned_starters
    _cfg_team_name = load_config().get('team_name') or 'Toronto Dark Knights'
    _pinned = get_pinned_starters(team_name=_cfg_team_name,
                                    last_n_games=3, min_games=1)
    _observed_bo_by_name: dict[str, int] = {}
    for _pos, _ps in _pinned.items():
        # Find the active player matching the observed starter. We try three
        # increasingly tolerant passes, all strict enough to avoid false
        # positives:
        #   1. Exact player_name equality
        #   2. card_id equality (resolved via the name_resolver)
        #   3. Substring containment as a last resort
        _pos_pool = active_by_pos.get(_pos, [])
        _obs_name = _ps.player_name or ''
        _matched = None
        for _p in _pos_pool:
            if _p.get('player_name', '') == _obs_name:
                _matched = _p
                break
        if _matched is None and _obs_name:
            try:
                from app.core.name_resolver import resolve_to_card_id
                _obs_cid = resolve_to_card_id(_obs_name, prefer_owned=True, conn=conn)
            except Exception:
                _obs_cid = None
            if _obs_cid:
                for _p in _pos_pool:
                    if _p.get('card_id') == _obs_cid:
                        _matched = _p
                        break
        if _matched is None:
            for _p in _pos_pool:
                pn = _p.get('player_name', '')
                if _obs_name and (_obs_name in pn or pn in _obs_name):
                    _matched = _p
                    break
        if _matched is not None:
            # Stale-override guard: when the observed starter's meta is much
            # lower than the best active player at the same position, the
            # observed data is probably stale (user just replaced the old
            # starter with a much better card, but new games haven't been
            # played yet). Skip the override so the meta leader wins. Delta
            # of 50 meta ≈ one tier of card quality — large enough to be
            # meaningful but small enough to respect user lineup choices
            # between comparable cards (e.g. platoon splits).
            best_meta = max((p.get('meta_score') or 0) for p in _pos_pool)
            matched_meta = _matched.get('meta_score') or 0
            if best_meta - matched_meta >= 50:
                import logging as _lg
                _lg.getLogger(__name__).info(
                    "observed_lineup override SKIPPED: %s pinned=%s (meta %.0f) "
                    "but active pool best meta=%.0f (delta %.0f) — observed "
                    "data likely stale",
                    _pos, _matched.get('player_name'), matched_meta,
                    best_meta, best_meta - matched_meta,
                )
            else:
                prev = (starters.get(_pos) or {}).get('player_name')
                if prev != _matched.get('player_name'):
                    import logging as _lg
                    _lg.getLogger(__name__).info(
                        "observed_lineup override: %s %s → %s (pinned in last 3 games)",
                        _pos, prev, _matched.get('player_name'),
                    )
                starters[_pos] = _matched
        # Track observed BO for the inline batting-order column
        if _ps.batting_order:
            _observed_bo_by_name[_matched.get('player_name') if _matched
                                  else _obs_name] = _ps.batting_order
except Exception as _obs_err:
    import logging as _lg
    _lg.getLogger(__name__).debug("observed_lineup override failed: %s", _obs_err)
    _observed_bo_by_name = {}

# ── DH manual-override sidebar control ──
# OOTP's CSV export doesn't include a DH row in team_lineup (confirmed across
# all lineup_types: overview, vs_lhp, vs_rhp, pitching). Players DH'ing are
# exported under their natural defensive position, so we have to guess who
# the user's actual DH is. The default heuristic (highest-meta "extra"
# starter not already winning their defensive slot) gets the right answer
# most of the time but can misfire — e.g. if two players share a defensive
# position, the heuristic flips a coin on which one is the DH vs the starter.
# This sidebar dropdown lets the user explicitly pin the DH so the rest of
# the optimizer (who gets displaced, what bats vs RHP/LHP, etc.) uses the
# right identity. Stored in session_state so it survives reruns.
# DH override candidates: include DH position so a file-exported DH can be
# re-pinned or replaced via the dropdown. When Cameron sets a player's team
# position to DH in OOTP, the roster CSVs now export POS=DH and that row
# lands in active_by_pos['DH'] — we want it visible in the manual dropdown.
_dh_override_candidates = sorted({
    p.get('player_name') or ''
    for _bat_pos in ('C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH')
    for p in active_by_pos.get(_bat_pos, [])
    if p.get('player_name')
})
with st.sidebar:
    st.markdown("---")
    st.markdown("**Lineup overrides**")
    _dh_current = st.session_state.get('_dh_override_name', '(auto — best guess)')
    if _dh_current != '(auto — best guess)' and _dh_current not in _dh_override_candidates:
        # User's previously-set override is stale (player no longer on roster)
        _dh_current = '(auto — best guess)'
    _dh_choice = st.selectbox(
        "DH (manual)",
        options=['(auto — best guess)'] + _dh_override_candidates,
        index=(['(auto — best guess)'] + _dh_override_candidates).index(_dh_current),
        help=(
            "Normally read from your OOTP export — if you set a player's team "
            "position to DH in-game, OOTP writes a POS=DH row and we pick that "
            "up automatically. Use this dropdown only when you want to force a "
            "different player into the DH slot than what OOTP exported."
        ),
    )
    if _dh_choice != '(auto — best guess)':
        st.session_state['_dh_override_name'] = _dh_choice
    else:
        st.session_state.pop('_dh_override_name', None)

# ── Load in-game performance for context ──
# We pull enough stats for the driver analyzer below: BABIP (luck on balls in
# play), ISO (power rate), strikeout/walk rates, and HR rate for batters; and
# ERA vs FIP, BABIP, WHIP, K/9, BB/9 for pitchers. The whole point is to let
# the user see *why* a slot is over/underperforming — small-sample luck, a
# real skill signal, or something that looks like league fit.
#
# Stage 1d partition: per-player stats live in the team-scoped partition
# (league_id IS NULL) and are keyed by card_id, with player_name as a
# secondary alias for legacy call sites that still look up by name. League
# averages for the outlook driver pull from the CURRENT league's partition
# (``config.active_league``), falling back to NULL if the league partition
# hasn't been populated yet.
_active_league_id = config.get('active_league')

# ── Cross-team confidence map (card_id → confidence dict) ──
# Pulls stats from every team's instance of each roster card (e.g. other teams
# in the league also have this card) and computes a 0-100 confidence score.
# Driven by the game logs + box scores now ingested. Keyed by card_id so the
# row renderer can look up in O(1).
_conf_by_card: dict = {}
_conf_by_name: dict = {}
try:
    from app.core.card_aggregation import card_confidence as _card_conf
    # Only compute for cards actually on the active roster — 26 calls, fast.
    for _row in conn.execute(
        """SELECT DISTINCT r.player_name, r.card_id
           FROM roster_current r
           WHERE r.lineup_role IN ('starter','rotation','closer','bullpen')
             AND r.player_name IS NOT NULL"""
    ).fetchall():
        try:
            cid = _row['card_id']
            # Pool confidence across leagues — the meta overlays are
            # cross-league (pooled PA/BF), so the confidence score that
            # gauges stability of that meta should be cross-league too.
            # Filtering to active_league alone produced spurious 🟡 50 scores
            # on relievers who have 150+ pooled IP but only 3-5 IP in lb124.
            _c = _card_conf(
                player_name=_row['player_name'] if not cid else None,
                card_id=cid,
                league_id=None,
                conn=conn,
            )
            if cid:
                _conf_by_card[cid] = _c
            _conf_by_name[_row['player_name']] = _c
        except Exception:
            continue
except Exception:
    pass

# ── Regression candidate map (card_id → direction) ──
# Pulls the regression scan once and indexes by card_id so each row can show
# its flag in O(1). Both UP (buy-low / positive regression) and DOWN
# (sell-high / negative regression) are surfaced in the Regression column.
_regress_by_card: dict = {}
_regress_by_name: dict = {}
try:
    from app.core.superstats import regression_candidates as _regress_scan
    for _rc in _regress_scan(league_id=_active_league_id, min_pa=100, conn=conn):
        _regress_by_card[_rc['card_id']] = _rc
        # card_title is the closest thing we have to player-name-indexable
        # if card_id lookup ever misses — we key by both. Won't hurt if both hit.
except Exception:
    pass

_perf_bat = {}
_perf_pit = {}
_perf_bat_by_card = {}
_perf_pit_by_card = {}
_latest_snap = conn.execute(
    "SELECT MAX(snapshot_date) as d FROM batting_stats WHERE league_id IS NULL"
).fetchone()
if _latest_snap and _latest_snap['d']:
    # Lowered the PA gate from 50 → 5 so freshly-pinned starters (e.g.
    # Kluttz/Al Dark just moved into the lineup with 4-8 PA) still get a
    # Status cell. The Status renderer marks anything <50 PA as "small
    # sample" so the reader knows the OPS reading is noisy, but at least
    # it's VISIBLE instead of a confusing blank cell.
    for r in conn.execute(
        """SELECT player_name, card_id, pa, ab, hits, hr, k, bb, war, ops, ops_plus, babip, iso,
                  CASE WHEN pa > 0 THEN war * 600.0 / pa ELSE 0 END as war600
           FROM batting_stats
           WHERE snapshot_date = ? AND league_id IS NULL AND pa >= 5 AND ab > 0""",
        (_latest_snap['d'],),
    ).fetchall():
        d = dict(r)
        _perf_bat[r['player_name']] = d
        if r['card_id'] is not None:
            _perf_bat_by_card[r['card_id']] = d
_latest_psnap = conn.execute(
    "SELECT MAX(snapshot_date) as d FROM pitching_stats WHERE league_id IS NULL"
).fetchone()
if _latest_psnap and _latest_psnap['d']:
    # IP gate lowered from 10 → 1 so relievers with only a few innings
    # (e.g. Dicky Lovelady with 4 IP on current team but 159 pooled) still
    # surface a Status cell. The Status renderer tags ``< 20 IP`` as a
    # small-sample warning so the reader knows the ERA is noisy.
    for r in conn.execute(
        """SELECT player_name, card_id, ip, era, era_plus, fip, war, babip, whip,
                  k_per_9, bb_per_9, hr_per_9,
                  CASE WHEN ip > 0 THEN war * 200.0 / ip ELSE 0 END as war200
           FROM pitching_stats
           WHERE snapshot_date = ? AND league_id IS NULL AND ip >= 1
             AND (k > 0 OR era > 0 OR hits_allowed > 0)""",
        (_latest_psnap['d'],),
    ).fetchall():
        d = dict(r)
        _perf_pit[r['player_name']] = d
        if r['card_id'] is not None:
            _perf_pit_by_card[r['card_id']] = d

# League-wide averages for the same snapshot — used by _analyze_perf_driver()
# to decide whether a player's BABIP / rate stats are outliers vs the league.
# We pull aggregates once so the per-player analysis is just dictionary math.
# Scoped to the CURRENT league (lb124), with a graceful fall-through: if the
# league-wide file hasn't been re-imported since the Stage 1 fix, the NULL
# partition is a decent proxy (it's the user's team, so averages will be
# noisy but non-zero).
_lg_bat = {}
_lg_pit = {}
_lg_filter_bat = "league_id = ?" if _active_league_id else "league_id IS NULL"
_lg_filter_pit = "league_id = ?" if _active_league_id else "league_id IS NULL"
_lg_args_bat = (_active_league_id,) if _active_league_id else ()
_lg_args_pit = (_active_league_id,) if _active_league_id else ()


def _fetch_lg_avg(sql: str, args: tuple) -> dict:
    try:
        row = conn.execute(sql, args).fetchone()
        if not row:
            return {}
        d = {k: (v or 0) for k, v in dict(row).items()}
        # If every column is 0, treat as empty and let caller fall back.
        if all((v == 0) for v in d.values()):
            return {}
        return d
    except Exception:
        return {}


# Latest date for the active league (lb124) — may differ from the team
# snapshot date if the league file was imported on a different day.
_lg_latest_bat = conn.execute(
    f"SELECT MAX(snapshot_date) as d FROM batting_stats WHERE {_lg_filter_bat}",
    _lg_args_bat,
).fetchone()
if _lg_latest_bat and _lg_latest_bat['d']:
    _lg_bat = _fetch_lg_avg(
        f"""SELECT AVG(babip) as babip, AVG(ops) as ops, AVG(iso) as iso,
                   SUM(hr) * 1.0 / NULLIF(SUM(ab), 0) as hr_rate,
                   SUM(k)  * 1.0 / NULLIF(SUM(pa), 0) as k_rate,
                   SUM(bb) * 1.0 / NULLIF(SUM(pa), 0) as bb_rate
            FROM batting_stats
            WHERE snapshot_date = ? AND {_lg_filter_bat} AND pa >= 50 AND ab > 0""",
        (_lg_latest_bat['d'],) + _lg_args_bat,
    )
# Fallback to NULL partition if league partition is empty (pre-Stage-1 install)
if not _lg_bat and _latest_snap and _latest_snap['d']:
    _lg_bat = _fetch_lg_avg(
        """SELECT AVG(babip) as babip, AVG(ops) as ops, AVG(iso) as iso,
                  SUM(hr) * 1.0 / NULLIF(SUM(ab), 0) as hr_rate,
                  SUM(k)  * 1.0 / NULLIF(SUM(pa), 0) as k_rate,
                  SUM(bb) * 1.0 / NULLIF(SUM(pa), 0) as bb_rate
           FROM batting_stats
           WHERE snapshot_date = ? AND league_id IS NULL AND pa >= 50 AND ab > 0""",
        (_latest_snap['d'],),
    )

_lg_latest_pit = conn.execute(
    f"SELECT MAX(snapshot_date) as d FROM pitching_stats WHERE {_lg_filter_pit}",
    _lg_args_pit,
).fetchone()
if _lg_latest_pit and _lg_latest_pit['d']:
    _lg_pit = _fetch_lg_avg(
        f"""SELECT AVG(era) as era, AVG(fip) as fip, AVG(babip) as babip,
                   AVG(whip) as whip, AVG(k_per_9) as k_per_9, AVG(bb_per_9) as bb_per_9
            FROM pitching_stats
            WHERE snapshot_date = ? AND {_lg_filter_pit} AND ip >= 10 AND (k > 0 OR era > 0)""",
        (_lg_latest_pit['d'],) + _lg_args_pit,
    )
if not _lg_pit and _latest_psnap and _latest_psnap['d']:
    _lg_pit = _fetch_lg_avg(
        """SELECT AVG(era) as era, AVG(fip) as fip, AVG(babip) as babip,
                  AVG(whip) as whip, AVG(k_per_9) as k_per_9, AVG(bb_per_9) as bb_per_9
           FROM pitching_stats
           WHERE snapshot_date = ? AND league_id IS NULL AND ip >= 10 AND (k > 0 OR era > 0)""",
        (_latest_psnap['d'],),
    )


# ── League-relative peer pools ──
# For the driver analysis we want to say "ERA+ 172 — top 5% of lb124 pitchers"
# rather than just quoting the raw number. We load a sorted list of ERA+ /
# OPS+ values from the *active* league partition so the percentile lookup is
# vs real league counterparts. If the league file hasn't been re-imported
# since the Stage 1 partition fix, we fall back to the team-only NULL
# partition — the pool will be tiny (~18 players) and percentiles become
# "vs your own team", so we gate display on pool size >= 20. Until then,
# we still surface raw ERA+/OPS+ (which is league-relative by definition —
# 100 = league average) via a "N% better/worse than league" phrasing.
#
# Direct answer to "how are they doing relative to their league
# counterparts?": this IS the relative comparison. ERA+/OPS+ are already
# league-normalized; the percentile just adds rank context when we have it.
_peer_era_plus: list[float] = []
_peer_ops_plus: list[float] = []
_peer_pool_label_bat = f"{_active_league_id or 'league'} hitters"
_peer_pool_label_pit = f"{_active_league_id or 'league'} pitchers"

# Prefer the active-league partition. The league file has per-player ERA+/
# OPS+ rows for every qualifying pitcher/batter in lb124, which is the
# actual peer pool the user cares about.
if _lg_latest_bat and _lg_latest_bat['d']:
    try:
        _peer_ops_plus = sorted(
            (r['ops_plus'] for r in conn.execute(
                f"""SELECT ops_plus FROM batting_stats
                    WHERE snapshot_date = ? AND {_lg_filter_bat}
                      AND pa >= 50 AND ops_plus IS NOT NULL AND ops_plus > 0""",
                (_lg_latest_bat['d'],) + _lg_args_bat,
            ).fetchall()),
        )
    except Exception:
        _peer_ops_plus = []
# Fall back to NULL partition (team-only) if the league partition is empty.
# Pool is tiny here — the percentile gate (>=20) will skip the rank phrase
# but raw OPS+ still gets surfaced.
if not _peer_ops_plus and _latest_snap and _latest_snap['d']:
    try:
        _peer_ops_plus = sorted(
            (r['ops_plus'] for r in conn.execute(
                """SELECT ops_plus FROM batting_stats
                   WHERE snapshot_date = ? AND league_id IS NULL
                     AND pa >= 50 AND ops_plus IS NOT NULL AND ops_plus > 0""",
                (_latest_snap['d'],),
            ).fetchall()),
        )
        _peer_pool_label_bat = "team hitters"
    except Exception:
        _peer_ops_plus = []

if _lg_latest_pit and _lg_latest_pit['d']:
    try:
        _peer_era_plus = sorted(
            (r['era_plus'] for r in conn.execute(
                f"""SELECT era_plus FROM pitching_stats
                    WHERE snapshot_date = ? AND {_lg_filter_pit}
                      AND ip >= 10 AND era_plus IS NOT NULL AND era_plus > 0""",
                (_lg_latest_pit['d'],) + _lg_args_pit,
            ).fetchall()),
        )
    except Exception:
        _peer_era_plus = []
if not _peer_era_plus and _latest_psnap and _latest_psnap['d']:
    try:
        _peer_era_plus = sorted(
            (r['era_plus'] for r in conn.execute(
                """SELECT era_plus FROM pitching_stats
                   WHERE snapshot_date = ? AND league_id IS NULL
                     AND ip >= 10 AND era_plus IS NOT NULL AND era_plus > 0""",
                (_latest_psnap['d'],),
            ).fetchall()),
        )
        _peer_pool_label_pit = "team pitchers"
    except Exception:
        _peer_era_plus = []


def _percentile_rank(value: float | None, sorted_peers: list) -> float | None:
    """Return where `value` sits in `sorted_peers` as a 0..1 percentile.

    Uses ``bisect_right`` so ties with value count as at-or-below. Returns
    None if the pool is empty or the value is invalid. 1.0 = top of the
    distribution, 0.0 = bottom. Caller is responsible for gating on pool
    size (a 3-player pool gives nonsense percentiles).
    """
    if not sorted_peers or value is None or value <= 0:
        return None
    idx = bisect_right(sorted_peers, value)
    return idx / len(sorted_peers)


def _format_league_rel_line(
    rate_val: float | None,
    pool: list,
    label: str,
    pool_label: str,
    min_pool: int = 20,
) -> str | None:
    """Build a one-line league-relative driver string from a 100-anchored rate.

    `rate_val` is an ERA+/OPS+ style stat where 100 = league average. We
    return None if the player is inside ±15 of average (no signal to
    surface). If ``len(pool) >= min_pool``, we append a percentile phrase
    ("top 5% of lb124 pitchers"); otherwise we fall back to a plain
    "N% better/worse than league" phrasing that's still accurate because
    ERA+/OPS+ are league-normalized by construction.
    """
    if rate_val is None or rate_val <= 0:
        return None
    delta = rate_val - 100
    if abs(delta) < 15:
        return None

    pct = _percentile_rank(rate_val, pool) if len(pool) >= min_pool else None
    if pct is not None and pct >= 0.90:
        tail = f"top {max(1, 100 - int(pct * 100))}% of {pool_label}"
    elif pct is not None and pct <= 0.10:
        tail = f"bottom {max(1, int(pct * 100))}% of {pool_label}"
    elif pct is not None and pct >= 0.75:
        tail = f"top quartile of {pool_label}"
    elif pct is not None and pct <= 0.25:
        tail = f"bottom quartile of {pool_label}"
    else:
        direction_word = 'better' if delta > 0 else 'worse'
        tail = f"{abs(delta):.0f}% {direction_word} than league avg"
    return f"{label} {rate_val:.0f} \u2014 {tail}"


# ── Performance driver analysis ──
# We used to "lock" slots where a player was outperforming their card meta —
# the optimizer would refuse to recommend a market buy on the theory that you
# shouldn't pay PP to replace a producing starter. In practice this was
# backwards: small samples + hot streaks explain most overperformance, so
# locking hides upgrades during exactly the window when regression is most
# likely. Now we *always* show the buy recommendation and treat over/under
# performance as **informational** — surfaced in an Outlook column with a
# Outlook thresholds — used BOTH as absolute (legacy) and as a fraction of
# the card's own meta (new). Whichever is more permissive wins, so a small
# card with an absolute 30-point gap still registers while a big card
# avoids getting flagged "cold" for a -80 point gap that's only -10%.
_DRIVER_GAP = 50            # meta points — minimum absolute gap
_DRIVER_GAP_PCT = 0.12      # 12% of card meta — minimum relative gap
_BAT_DRIVER_PA_MIN = 50     # informational thresholds are lower than the old lock
_PIT_DRIVER_IP_MIN = 15     # the numbers are labeled "low sample" not suppressed


# ── Perf→meta anchor rescale (2026-04-17) ──
# After demoting OVR to diagnostic, the meta formula's output range compressed
# (no more +160 OVR bump). The prior perf anchors were too flat: OPS+ 100 →
# 500 with slope 3 meant OPS+ 120 = only 560 meta, while a roster card at
# meta 700 was expected to produce OPS+ ~120. Every decent performer looked
# "cold" because perf_meta was stuck near the OPS+ 100 baseline.
#
# Rescaled so typical expected output for a given card meta lands at that meta:
#   meta 550 ≈ OPS+ 105, 1.5 WAR
#   meta 700 ≈ OPS+ 120, 3.0 WAR
#   meta 850 ≈ OPS+ 135, 4.5 WAR
_WAR_SLOPE = 100.0          # meta per WAR/full-season
_WAR_BASELINE = 400.0       # ~replacement = meta 400
_OPS_PLUS_SLOPE = 10.0      # meta per OPS+ point
_OPS_PLUS_BASELINE = 500.0  # OPS+ 100 = meta 500
_ERA_PLUS_SLOPE = 10.0      # meta per ERA+ point (same slope as OPS+)
_ERA_PLUS_BASELINE = 500.0  # ERA+ 100 = meta 500


def _perf_to_meta_equivalent(war_per_full_season: float) -> float:
    """Map a WAR/600 (or WAR/200) value to an approximate meta score.

    v2 (2026-04-17): slope bumped from 56 to 100 meta-per-WAR. The old slope
    was calibrated when meta included OVR (range 700-900); now that OVR is
    diagnostic-only (range 500-750), we need a steeper slope so real-world
    WAR output maps into the same range as card meta.

    Kept for callers that only have WAR. Prefer ``_perf_meta_pitcher`` /
    ``_perf_meta_batter`` which blend WAR with league-relative rate stats.
    """
    if war_per_full_season is None or war_per_full_season <= 0:
        return _WAR_BASELINE
    return _WAR_BASELINE + war_per_full_season * _WAR_SLOPE


def _era_plus_to_meta(era_plus: float | None) -> float:
    """Map ERA+ to an approximate meta score (league-relative).

    v2: slope bumped from 3.5 to 10 per ERA+ point (see rescale comment).
    ERA+ 100 = league average → 500 meta; ERA+ 150 → 1000 meta;
    ERA+ 75 → 250 meta. High slope is intentional — a 50%-better-than-league
    rate pitcher is genuinely elite and should lap the meta distribution.
    """
    if era_plus is None or era_plus <= 0:
        return _ERA_PLUS_BASELINE - 250.0  # 250 floor
    return max(200.0, _ERA_PLUS_BASELINE + (era_plus - 100) * _ERA_PLUS_SLOPE)


def _ops_plus_to_meta(ops_plus: float | None) -> float:
    """Map OPS+ to an approximate meta score (league-relative).

    v2: slope bumped from 3 to 10 per OPS+ point. See _perf_to_meta_equivalent
    comment for rescale rationale.
    """
    if ops_plus is None or ops_plus <= 0:
        return _OPS_PLUS_BASELINE - 250.0  # 250 floor
    return max(200.0, _OPS_PLUS_BASELINE + (ops_plus - 100) * _OPS_PLUS_SLOPE)


def _perf_meta_pitcher(war200: float, era_plus: float | None, ip: float) -> float:
    """Blend WAR-rate and ERA+ into a single perf-meta for a pitcher.

    v3 (2026-04-17): Blend tilts toward WAR as the sample stabilizes. At low
    IP, WAR is noisy (1 bad start dominates WAR/200), so we lean on ERA+ as
    the stable league-relative rate stat. At full-season IP, WAR is the
    better value estimate (it captures leverage, BABIP luck, etc.) so we
    invert the blend and lean on WAR. Prior static 60/40 toward ERA+ at all
    IP was over-weighting rate stats and flagging WAR-positive pitchers as
    "cold" whenever their ERA+ drifted below 100.

        IP < 20         : 75% ERA+, 25% WAR  (small sample → rate-anchored)
        20 ≤ IP < 100   : linear ramp
        IP ≥ 100        : 40% ERA+, 60% WAR  (stabilized → value-anchored)
    """
    war_meta = _perf_to_meta_equivalent(war200)
    era_meta = _era_plus_to_meta(era_plus)
    if ip < 20:
        return 0.75 * era_meta + 0.25 * war_meta
    if ip >= 100:
        return 0.40 * era_meta + 0.60 * war_meta
    # Linear ramp between the two anchors
    t = (ip - 20) / 80.0  # 0 at IP=20, 1 at IP=100
    era_w = 0.75 - 0.35 * t
    war_w = 1.0 - era_w
    return era_w * era_meta + war_w * war_meta


def _perf_meta_batter(war600: float, ops_plus: float | None, pa: int) -> float:
    """Blend WAR-rate and OPS+ into a single perf-meta for a batter.

    v3 (2026-04-17): Same IP-stabilization logic as the pitcher version,
    applied to PA. WAR captures the whole picture (offense + defense +
    baserunning); OPS+ only captures batting. For stabilized samples,
    trust WAR. Prior 60/40 toward OPS+ produced Cold flags for players
    like Bellinger (OPS+ 92, WAR/600 2.14 — WAR above expected but OPS+
    below) because rate stats dragged perf_meta down even though value
    production was on-track.

        PA < 80         : 75% OPS+, 25% WAR
        80 ≤ PA < 300   : linear ramp
        PA ≥ 300        : 40% OPS+, 60% WAR
    """
    war_meta = _perf_to_meta_equivalent(war600)
    ops_meta = _ops_plus_to_meta(ops_plus)
    if pa < 80:
        return 0.75 * ops_meta + 0.25 * war_meta
    if pa >= 300:
        return 0.40 * ops_meta + 0.60 * war_meta
    t = (pa - 80) / 220.0
    ops_w = 0.75 - 0.35 * t
    war_w = 1.0 - ops_w
    return ops_w * ops_meta + war_w * war_meta


def _analyze_perf_driver(player_name: str, current_meta: float, is_pitcher: bool) -> dict | None:
    """Diagnose why a player is over- or under-performing their card meta.

    Returns a dict with:
      ``perf_meta``       perf-derived meta equivalent (int)
      ``gap``             perf_meta - current_meta (signed int)
      ``direction``       'hot' | 'cold' | 'inline'
      ``verdict``         'luck' | 'skill' | 'mixed' | 'small_sample' | 'inline'
      ``outlook``         short display string with emoji for the Outlook column
      ``drivers``         list[str] — human-readable driver lines for the expander
      ``regression``      bool — True if we think performance is unsustainable
      ``sample_label``    e.g. "88 PA" or "22.1 IP"
      ``sample_ok``       bool — False = too small to judge

    Returns None if the player has no stats row at all.
    """
    if not player_name:
        return None
    perf = (_perf_pit if is_pitcher else _perf_bat).get(player_name)
    if not perf:
        return None

    if is_pitcher:
        ip = float(perf.get('ip') or 0)
        sample_label = f"{ip:.1f} IP"
        sample_ok = ip >= _PIT_DRIVER_IP_MIN
        war_full = perf.get('war200') or 0
        era_plus = perf.get('era_plus')
        perf_meta = round(_perf_meta_pitcher(war_full, era_plus, ip))
    else:
        pa = int(perf.get('pa') or 0)
        sample_label = f"{pa} PA"
        sample_ok = pa >= _BAT_DRIVER_PA_MIN
        war_full = perf.get('war600') or 0
        ops_plus = perf.get('ops_plus')
        perf_meta = round(_perf_meta_batter(war_full, ops_plus, pa))

    current_meta_int = round(current_meta or 0)
    gap = perf_meta - current_meta_int

    # Hybrid threshold: flag as hot/cold only if BOTH absolute and relative
    # thresholds are cleared. An 80-point gap on a 200-meta filler card is
    # meaningful (40%); an 80-point gap on an 800-meta ace is noise (10%).
    # Using OR produced universal "cold" labels because even healthy cards
    # routinely have 50-meta differences against their card score.
    pct_gap = (abs(gap) / current_meta_int) if current_meta_int > 0 else 0.0
    if abs(gap) < _DRIVER_GAP or pct_gap < _DRIVER_GAP_PCT:
        return {
            'perf_meta': perf_meta, 'gap': gap,
            'direction': 'inline', 'verdict': 'inline',
            'outlook': '', 'drivers': [],
            'regression': False,
            'sample_label': sample_label, 'sample_ok': sample_ok,
        }

    # WAR-positive override: if we're about to flag "cold" but the WAR rate is
    # AT OR ABOVE what the card meta expects, downgrade to "inline". A card's
    # meta implies an expected WAR production (roughly meta - 400) / 100 for
    # batters, (meta - 400) / 90 for pitchers). If the player's WAR-rate meets
    # that expectation, they're delivering value — rate-stat softness alone
    # shouldn't tag them cold. Prevents Bellinger/Pierre-style false cold flags
    # on WAR-positive but OPS+-average players.
    if gap < 0:
        expected_war_rate = max(0.0, (current_meta_int - 400) / 100.0)
        if is_pitcher:
            # Pitchers: slightly higher expectation per meta point (slope 0.011 vs 0.014)
            expected_war_rate = max(0.0, (current_meta_int - 400) / 90.0)
        # A 10% tolerance around expected — being slightly below shouldn't save
        # them from cold either.
        if war_full >= expected_war_rate * 0.90:
            return {
                'perf_meta': perf_meta, 'gap': gap,
                'direction': 'inline', 'verdict': 'mixed',
                'outlook': '',
                'drivers': [
                    f"Rate stats soft but WAR on-track "
                    f"({war_full:.1f} vs ~{expected_war_rate:.1f} expected) — not cold."
                ],
                'regression': False,
                'sample_label': sample_label, 'sample_ok': sample_ok,
            }

    direction = 'hot' if gap > 0 else 'cold'
    drivers: list[str] = []
    luck_votes = 0
    skill_votes = 0

    if is_pitcher:
        # ── Pitcher driver analysis ──
        era = perf.get('era')
        fip = perf.get('fip')
        babip = perf.get('babip')
        whip = perf.get('whip')
        k9 = perf.get('k_per_9')
        bb9 = perf.get('bb_per_9')
        era_plus = perf.get('era_plus')
        lg_era = _lg_pit.get('era') or 4.00
        lg_fip = _lg_pit.get('fip') or 4.00
        lg_babip = _lg_pit.get('babip') or 0.295
        lg_k9 = _lg_pit.get('k_per_9') or 8.5
        lg_bb9 = _lg_pit.get('bb_per_9') or 3.0

        # Lead the driver list with the league-relative ERA+ line — this is
        # the most interpretable signal (100 = league avg, normalized by
        # construction) and directly answers "how are they doing vs their
        # league counterparts". We don't vote on this one; it's a headline
        # summary. The underlying FIP/BABIP/K%/BB% votes still drive the
        # luck-vs-skill verdict below.
        rel_pit_line = _format_league_rel_line(
            era_plus, _peer_era_plus, 'ERA+', _peer_pool_label_pit,
        )
        if rel_pit_line:
            drivers.append(rel_pit_line)

        if era is not None and fip is not None:
            era_minus_fip = era - fip
            if direction == 'hot' and era_minus_fip <= -0.50:
                drivers.append(
                    f"ERA {era:.2f} well below FIP {fip:.2f} "
                    f"(\u0394{era_minus_fip:+.2f}) \u2014 sequencing/strand luck"
                )
                luck_votes += 1
            elif direction == 'cold' and era_minus_fip >= 0.50:
                drivers.append(
                    f"ERA {era:.2f} well above FIP {fip:.2f} "
                    f"(\u0394{era_minus_fip:+.2f}) \u2014 bad sequencing luck"
                )
                luck_votes += 1
            elif abs(era_minus_fip) < 0.30:
                drivers.append(
                    f"ERA {era:.2f} tracks FIP {fip:.2f} \u2014 no sequencing luck"
                )
                skill_votes += 1

        if babip is not None and lg_babip:
            d_babip = babip - lg_babip
            if direction == 'hot' and d_babip <= -0.025:
                drivers.append(
                    f"BABIP {babip:.3f} vs league {lg_babip:.3f} "
                    f"({d_babip:+.3f}) \u2014 low BIP rate (luck on contact)"
                )
                luck_votes += 1
            elif direction == 'cold' and d_babip >= 0.025:
                drivers.append(
                    f"BABIP {babip:.3f} vs league {lg_babip:.3f} "
                    f"({d_babip:+.3f}) \u2014 high BIP rate (bad luck)"
                )
                luck_votes += 1

        # K/9 — elevated = skill win (either direction it's a real signal);
        # depressed = skill concern (also a real signal, just the bad kind).
        if k9 is not None and lg_k9:
            d_k9 = k9 - lg_k9
            if d_k9 >= 1.5:
                drivers.append(
                    f"K/9 {k9:.1f} vs league {lg_k9:.1f} "
                    f"(+{d_k9:.1f}) \u2014 missing bats (skill)"
                )
                skill_votes += 1
            elif d_k9 <= -1.5:
                drivers.append(
                    f"K/9 {k9:.1f} vs league {lg_k9:.1f} "
                    f"({d_k9:+.1f}) \u2014 not missing bats (real skill gap)"
                )
                skill_votes += 1

        # BB/9 — plus command vs wildness. Mirror both directions.
        if bb9 is not None and lg_bb9:
            d_bb9 = bb9 - lg_bb9
            if d_bb9 <= -1.0:
                drivers.append(
                    f"BB/9 {bb9:.1f} vs league {lg_bb9:.1f} "
                    f"({d_bb9:+.1f}) \u2014 plus command (skill)"
                )
                skill_votes += 1
            elif d_bb9 >= 1.0:
                drivers.append(
                    f"BB/9 {bb9:.1f} vs league {lg_bb9:.1f} "
                    f"(+{d_bb9:.1f}) \u2014 walking too many (real skill gap)"
                )
                skill_votes += 1

        # WHIP flags — elite or bloated, both are real signals.
        if whip is not None:
            if direction == 'hot' and whip <= 1.05:
                drivers.append(f"WHIP {whip:.2f} \u2014 elite baserunner prevention")
                skill_votes += 1
            elif direction == 'cold' and whip >= 1.45:
                drivers.append(f"WHIP {whip:.2f} \u2014 elevated baserunners (real)")
                skill_votes += 1

    else:
        # ── Batter driver analysis ──
        pa = int(perf.get('pa') or 0)
        ab = int(perf.get('ab') or 0)
        hr = int(perf.get('hr') or 0)
        k = int(perf.get('k') or 0)
        bb = int(perf.get('bb') or 0)
        babip = perf.get('babip')
        iso = perf.get('iso')
        ops = perf.get('ops') or 0
        ops_plus = perf.get('ops_plus')
        lg_babip = _lg_bat.get('babip') or 0.300
        lg_iso = _lg_bat.get('iso') or 0.150
        lg_hr_rate = _lg_bat.get('hr_rate') or 0.030
        lg_k_rate = _lg_bat.get('k_rate') or 0.225
        lg_bb_rate = _lg_bat.get('bb_rate') or 0.085

        hr_rate = (hr / ab) if ab > 0 else 0
        k_rate = (k / pa) if pa > 0 else 0
        bb_rate = (bb / pa) if pa > 0 else 0

        # Lead with OPS+ — league-relative headline, no vote (same pattern
        # as the pitcher ERA+ lead-off).
        rel_bat_line = _format_league_rel_line(
            ops_plus, _peer_ops_plus, 'OPS+', _peer_pool_label_bat,
        )
        if rel_bat_line:
            drivers.append(rel_bat_line)

        if babip is not None and lg_babip:
            d_babip = babip - lg_babip
            if direction == 'hot' and d_babip >= 0.040:
                drivers.append(
                    f"BABIP .{int(babip*1000):03d} vs league "
                    f".{int(lg_babip*1000):03d} ({d_babip:+.3f}) "
                    f"\u2014 hot on balls in play (regression likely)"
                )
                luck_votes += 1
            elif direction == 'cold' and d_babip <= -0.040:
                drivers.append(
                    f"BABIP .{int(babip*1000):03d} vs league "
                    f".{int(lg_babip*1000):03d} ({d_babip:+.3f}) "
                    f"\u2014 unlucky on balls in play"
                )
                luck_votes += 1

        # ISO — elevated = real power, depressed = real power drought.
        # Mirror both directions so cold sluggers get the skill signal too.
        if iso is not None and lg_iso:
            d_iso = iso - lg_iso
            if d_iso >= 0.050:
                drivers.append(
                    f"ISO .{int(iso*1000):03d} vs league .{int(lg_iso*1000):03d} "
                    f"(+{d_iso:.3f}) \u2014 real power output"
                )
                skill_votes += 1
            elif d_iso <= -0.060 and direction == 'cold':
                drivers.append(
                    f"ISO .{int(iso*1000):03d} vs league .{int(lg_iso*1000):03d} "
                    f"({d_iso:+.3f}) \u2014 real power drought"
                )
                skill_votes += 1

        if direction == 'hot' and hr_rate - lg_hr_rate >= 0.025:
            drivers.append(
                f"HR/AB {hr_rate*100:.1f}% vs league {lg_hr_rate*100:.1f}% "
                f"\u2014 elevated HR rate (sustainability depends on FB%)"
            )
            # Power rate is part-skill, part-luck; lean luck on small samples.
            if pa < 150:
                luck_votes += 1
            else:
                skill_votes += 1

        # K% — mirror both directions. Elite contact = skill win;
        # elevated K% = real contact problem. This fixes the Aranda case
        # where a cold slump driven by strikeouts was mis-labeled "unlucky"
        # because the analyzer only voted when K% was *better* than league.
        if k_rate and lg_k_rate:
            d_k_rate = k_rate - lg_k_rate  # positive = striking out more
            if -d_k_rate >= 0.040:  # player K% < league K% by 4pp+
                drivers.append(
                    f"K% {k_rate*100:.1f}% vs league {lg_k_rate*100:.1f}% "
                    f"\u2014 strong contact rate (skill, sticky)"
                )
                skill_votes += 1
            elif d_k_rate >= 0.040:  # player K% > league K% by 4pp+
                drivers.append(
                    f"K% {k_rate*100:.1f}% vs league {lg_k_rate*100:.1f}% "
                    f"\u2014 elevated K rate (real contact issue)"
                )
                skill_votes += 1

        # BB% — mirror both directions. Plus discipline = skill win;
        # depressed walk rate = real discipline problem.
        if bb_rate and lg_bb_rate:
            d_bb_rate = bb_rate - lg_bb_rate
            if d_bb_rate >= 0.030:
                drivers.append(
                    f"BB% {bb_rate*100:.1f}% vs league {lg_bb_rate*100:.1f}% "
                    f"\u2014 strong plate discipline (skill, sticky)"
                )
                skill_votes += 1
            elif d_bb_rate <= -0.035 and direction == 'cold':
                drivers.append(
                    f"BB% {bb_rate*100:.1f}% vs league {lg_bb_rate*100:.1f}% "
                    f"({d_bb_rate*100:+.1f}pp) \u2014 weak plate discipline (real)"
                )
                skill_votes += 1

        if direction == 'hot' and not drivers and ops >= 0.850:
            drivers.append(
                f"OPS .{int(ops*1000):03d} with no standout rate outlier "
                f"\u2014 possible league fit or balanced hot streak"
            )

    # ── Verdict from the vote tally ──
    if not sample_ok:
        verdict = 'small_sample'
    elif luck_votes > skill_votes:
        verdict = 'luck'
    elif skill_votes > luck_votes:
        verdict = 'skill'
    elif luck_votes > 0 or skill_votes > 0:
        verdict = 'mixed'
    else:
        verdict = 'mixed' if direction != 'inline' else 'inline'

    # Regression flag: hot + luck verdict OR small sample + hot
    regression = (direction == 'hot') and verdict in ('luck', 'small_sample', 'mixed')

    # Short display label for the Outlook column
    if direction == 'hot':
        if verdict == 'luck':
            outlook = f"\U0001f525 Hot \u2014 lucky (+{gap})"
        elif verdict == 'skill':
            outlook = f"\U0001f4aa Hot \u2014 real (+{gap})"
        elif verdict == 'small_sample':
            outlook = f"\u23f3 Hot \u2014 small sample (+{gap})"
        else:
            outlook = f"\U0001f525 Hot \u2014 mixed (+{gap})"
    elif direction == 'cold':
        if verdict == 'luck':
            outlook = f"\U0001f9ca Cold \u2014 unlucky ({gap})"
        elif verdict == 'skill':
            outlook = f"\u2744\ufe0f Cold \u2014 real ({gap})"
        elif verdict == 'small_sample':
            outlook = f"\u23f3 Cold \u2014 small sample ({gap})"
        else:
            outlook = f"\U0001f9ca Cold \u2014 mixed ({gap})"
    else:
        outlook = ''

    return {
        'perf_meta': perf_meta, 'gap': gap,
        'direction': direction, 'verdict': verdict,
        'outlook': outlook, 'drivers': drivers,
        'regression': regression,
        'sample_label': sample_label, 'sample_ok': sample_ok,
    }


def _is_outperforming_meta(player_name: str, current_meta: float, is_pitcher: bool) -> tuple:
    """Back-compat shim — the old hard-lock always returns False now.

    Kept so any lingering call sites still work, but the lock is dead:
    we never suppress a buy recommendation because the current player is
    hot. The informational story moves to _analyze_perf_driver().
    """
    a = _analyze_perf_driver(player_name, current_meta, is_pitcher)
    return (False, a['perf_meta'] if a else None)


def _keep_reason_text(analysis: dict) -> str:
    """Build a short 'why keep them' sentence from a perf analysis dict."""
    if not analysis:
        return ""
    v = analysis.get('verdict')
    g = analysis.get('gap', 0)
    s = analysis.get('sample_label', '')
    if v == 'luck':
        return f"unlucky ({g:+d}) \u2014 rate stats say rebound likely"
    if v == 'small_sample':
        return f"too early to cut ({s}) \u2014 small-sample noise"
    if v == 'mixed':
        return f"mixed signals ({g:+d}) \u2014 partial rebound case"
    return f"cold ({g:+d})"


def _find_drop_candidate(upgrade_entry: dict, active_pool: list) -> dict | None:
    """Suggest an alternative roster drop when the default is a rebound case.

    The 'default drop' for any promote is always the current starter at that
    slot — they're who gets displaced. When the current starter has cold-but-
    rebound-y perf signals (cold + luck/small_sample/mixed verdict), we'd
    rather keep them on the bench than cut them. This helper looks for a
    safer alternative drop on the active 26-man, with three rules:

    1. **Same side only** — a batting upgrade never suggests dropping a
       pitcher (and vice versa). Cross-side drops change the 13/13 roster
       composition in ways the user probably doesn't want.

    2. **No other rebound cases** — we re-check the candidate's own perf
       analysis. If they're *also* cold/unlucky with rebound upside, we
       skip them (dropping one rebound candidate to save another is a wash).

    3. **Role redundancy** — dropping the only backup C, the last usable
       SS, or the last long reliever would create a bigger hole than the
       upgrade fills. We count active-roster depth at each position and
       require `>= 3 total` before a player is droppable (preferred), or
       `>= 2 total` as a fallback — the #2 at a 2-deep position is a
       critical backup but at least the starter remains.

    `active_pool` should be a list of dicts with keys: name, pos, ovr, meta,
    role, perf_analysis, is_pit. The perf_analysis is precomputed at build
    time so we don't re-run the analyzer here.

    Returns None if the default drop is fine (no rebound upside on current)
    or if no safe alternative exists (every candidate is either a critical
    backup or a rebound case themselves).
    """
    analysis = upgrade_entry.get('perf_analysis')
    current = upgrade_entry.get('current_name', '')
    if not current or current == '(empty)':
        return None
    if not analysis:
        return None
    if analysis.get('direction') != 'cold':
        return None
    if analysis.get('verdict') not in ('luck', 'small_sample', 'mixed'):
        return None

    # Side filter — batting upgrades drop batters, pitching drops pitchers.
    is_pit_upgrade = _is_pitching_pos(upgrade_entry.get('pos') or '')

    # Build a position-count map for role-redundancy checks. We count the
    # whole active pool (batters or pitchers only, depending on the side).
    pos_counts: dict[str, int] = {}
    for p in active_pool:
        if bool(p.get('is_pit')) != is_pit_upgrade:
            continue
        pos_label = p.get('pos') or ''
        pos_counts[pos_label] = pos_counts.get(pos_label, 0) + 1

    def _is_rebound_case(cand_analysis):
        if not cand_analysis:
            return False
        if cand_analysis.get('direction') != 'cold':
            return False
        return cand_analysis.get('verdict') in ('luck', 'small_sample', 'mixed')

    # Partition candidates into surplus-safe (3+ at pos) and fallback (2 at
    # pos). Anything with only 1 at pos is off-limits — dropping the only
    # player at that position is worse than the upgrade we're making.
    surplus_safe = []
    fallback_safe = []
    for p in active_pool:
        if p['name'] == current:
            continue
        if bool(p.get('is_pit')) != is_pit_upgrade:
            continue
        if (p.get('meta') or 0) <= 0:
            continue
        if _is_rebound_case(p.get('perf_analysis')):
            continue  # Don't drop one rebound case to save another
        depth = pos_counts.get(p.get('pos') or '', 0)
        if depth >= 3:
            surplus_safe.append(p)
        elif depth >= 2:
            fallback_safe.append(p)
        # depth == 1: position would be empty after drop — skip entirely

    pool = surplus_safe if surplus_safe else fallback_safe
    if not pool:
        return None

    # Lowest-meta eligible piece — we're picking "the least valuable thing
    # we can safely cut" so the user loses the minimum possible production.
    alt = min(pool, key=lambda x: x['meta'] or 0)

    # Tag whether the pick came from surplus or fallback, so the caption can
    # explain *why* this drop is safer than the default.
    alt_depth_tag = 'surplus' if surplus_safe else 'thin'

    return {
        'keep_name': current,
        'keep_meta': int(upgrade_entry.get('current_meta') or 0),
        'keep_reason': _keep_reason_text(analysis),
        'alt_name': alt['name'],
        'alt_pos': alt['pos'],
        'alt_meta': int(alt['meta'] or 0),
        'alt_ovr': alt['ovr'],
        'alt_role': alt['role'],
        'alt_depth': alt_depth_tag,  # 'surplus' or 'thin'
        'alt_depth_count': pos_counts.get(alt['pos'] or '', 0),
    }

bat_field_positions = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
bat_positions = bat_field_positions + ['DH']
pitch_positions = ['SP', 'RP', 'CL']
if focus == "Batting Only":
    show_positions = bat_positions
elif focus == "Pitching Only":
    show_positions = pitch_positions
else:
    show_positions = bat_positions + pitch_positions


# ── Helpers ──
def find_roster_bench_upgrades(pos_value, current_meta, exclude_names=None, current_player_name=None):
    """Find bench/reserve players who beat the starter at ``pos_value``.

    Two independent matching paths, unioned:

    1. ROSTER-position match — anyone on the bench currently assigned to the
       same field position in the CSV. Catches the common case (backup CF
       for the CF slot).
    2. CARD-eligibility match — bench players whose CARD can play
       ``pos_value`` based on ``pos_rating_*`` (>= ELIGIBILITY_THRESHOLD)
       OR whose card's primary position IS ``pos_value`` even if the
       roster CSV assigns them elsewhere. Catches the position-flexibility
       case: Van Haltren on the bench (roster position=CF) should appear
       as an LF upgrade because his LF rating is above the eligibility
       floor and his meta clears the incumbent.

    Non-primary assignments get a small meta penalty via
    ``position_meta_penalty`` so a low-rating corner-OF card doesn't win
    the slot purely on raw meta.

    Performance gate: if the current starter is producing well
    (WAR/600 >= 1.5) and the bench candidate has no game data, skip —
    the existing starter has proven themselves and a rating-based swap
    isn't warranted.

    exclude_names should contain PLAYER NAMES (not card titles).
    """
    exclude_names = exclude_names or []

    # Check if current starter is performing well
    starter_perf = None
    if current_player_name:
        starter_perf = _perf_bat.get(current_player_name) or _perf_pit.get(current_player_name)

    bench_upgrades: list[dict] = []
    seen_keys: set = set()  # dedupe across the two matching paths

    # Pitching slots still use strict role-based matching — there's no
    # cross-eligibility concept for SP/RP/CL. Only batting slots benefit
    # from pos_rating-based matching.
    is_batting_slot = pos_value in POS_RATING_COL

    # Iterate every non-active roster row (bench + reserve). For batting
    # slots we check multi-position eligibility; for pitching slots we fall
    # back to exact roster-position match (the old behavior).
    candidate_pool = bench_pool if is_batting_slot else all_by_pos.get(pos_value, [])

    for p in candidate_pool:
        # Pitching path already filtered to this position via all_by_pos.
        # Batting path: match if roster position equals pos_value OR the
        # card's ratings make it eligible at pos_value.
        if is_batting_slot:
            matches_roster = p.get('position') == pos_value
            # card position_name may be None when the name-LIKE join didn't
            # find a card row — fall back to roster position match in that case.
            matches_card = is_eligible(p, pos_value) if p.get('position_name') else False
            if not (matches_roster or matches_card):
                continue

        # Skip active players (starter / rotation / closer / bullpen)
        if p.get('lineup_role') in ACTIVE_ROLES:
            continue

        pname = p['player_name']
        if pname in exclude_names:
            continue
        if any(pname in ex or ex in pname for ex in exclude_names):
            continue

        # Dedupe by card_id when available (bench_pool can contain the same
        # player under multiple roster positions on old snapshots).
        dedupe_key = p.get('card_id') or pname
        if dedupe_key in seen_keys:
            continue

        raw_m = p.get('meta_score') or 0
        # Apply defensive-fit penalty when assigning to a non-primary slot.
        penalty = position_meta_penalty(p, pos_value) if is_batting_slot else 0.0
        m = raw_m - penalty
        if m <= current_meta + min_improvement:
            continue

        # Performance gate: don't bench a producing starter for someone
        # with no game data.
        bench_perf = _perf_bat.get(pname) or _perf_pit.get(pname)
        if starter_perf and not bench_perf:
            war600 = starter_perf.get('war600', starter_perf.get('war200', 0))
            if war600 >= 1.5:
                continue

        seen_keys.add(dedupe_key)
        bench_upgrades.append({
            'card_id': p.get('card_id'),
            'card_title': p.get('card_title') or pname,
            'player_name': pname,
            'card_value': p.get('ovr'),
            'meta_score': m,
            'raw_meta': raw_m,
            'position_penalty': round(penalty, 1),
            'position_annotation': format_position_annotation(p, pos_value) if is_batting_slot else "",
            'last_10_price': 0,
            'action': 'Promote',
        })
    bench_upgrades.sort(key=lambda x: -(x['meta_score'] or 0))
    return bench_upgrades


def find_owned_upgrades(pos_value, current_meta, is_pitching, exclude_names=None, limit=5, current_player_name=None):
    """Find owned cards (active + reserve + collection) that upgrade the slot.

    Batting slots use the shared ``build_eligible_where_clause`` helper so
    multi-position cards compete for every slot they can play. A meta
    penalty is applied for non-primary assignments — small enough that
    genuine upgrades still surface, large enough to keep true specialists
    ahead of marginal cross-position fits.

    Pitching slots retain strict role-based matching (no cross-role
    eligibility exists for SP/RP/CL).
    """
    exclude_names = exclude_names or []

    # First: check roster bench/reserve at this position (catches position mismatches)
    bench_ups = find_roster_bench_upgrades(pos_value, current_meta, exclude_names, current_player_name)

    meta_col = "meta_score_pitching" if is_pitching else "meta_score_batting"
    pos_col = "pitcher_role_name" if is_pitching else "position_name"

    if is_pitching:
        # Pitching: unchanged — exact role match.
        results = conn.execute(f"""
            SELECT c.card_id, c.card_title, c.tier_name, c.card_value,
                   c.{meta_col} as meta_score, c.{meta_col} as raw_meta,
                   c.last_10_price, c.position_name,
                   mc.status as collection_status, r.lineup_role as roster_role
            FROM cards c
            LEFT JOIN my_collection mc ON mc.card_id = c.card_id
            LEFT JOIN roster r ON c.card_title LIKE '%' || r.player_name || '%' AND r.position = c.{pos_col}
            WHERE c.{pos_col} = ? AND c.owned = 1 AND c.{meta_col} > ?
            GROUP BY c.card_id ORDER BY c.{meta_col} DESC LIMIT ?
        """, (pos_value, current_meta + min_improvement, limit + len(exclude_names) + 5)).fetchall()
        # No penalty for pitching — pos_value is a role, not a defensive
        # position; pass-through matches the legacy shape.
        penalty_rows = [(dict(r), 0.0) for r in results]
    else:
        # Batting: use the eligibility helper so CF-primary cards compete
        # at LF/RF, a 3B-primary competes at 1B, etc. We pull the full set
        # of pos_rating_* columns so ``position_meta_penalty`` can score
        # the defensive fit without a re-query.
        rating_sql = select_rating_columns('c')
        where_frag, where_params = build_eligible_where_clause(pos_value, table_alias='c')
        results = conn.execute(f"""
            SELECT c.card_id, c.card_title, c.tier_name, c.card_value,
                   c.{meta_col} as raw_meta,
                   c.last_10_price, c.position_name,
                   {rating_sql},
                   mc.status as collection_status, r.lineup_role as roster_role
            FROM cards c
            LEFT JOIN my_collection mc ON mc.card_id = c.card_id
            LEFT JOIN roster r ON c.card_title LIKE '%' || r.player_name || '%'
            WHERE {where_frag} AND c.owned = 1
                AND c.pitcher_role IS NULL
                AND c.{meta_col} > ?
            GROUP BY c.card_id ORDER BY c.{meta_col} DESC LIMIT ?
        """, (*where_params, current_meta, limit + len(exclude_names) + 10)).fetchall()

        # Apply the defensive-fit penalty AFTER the SQL fetch so the penalty
        # has the actual rating in hand. Re-filter to enforce the
        # min_improvement bar on post-penalty meta.
        penalty_rows = []
        for r in results:
            d = dict(r)
            penalty = position_meta_penalty(d, pos_value)
            effective_meta = (d.get('raw_meta') or 0) - penalty
            if effective_meta <= current_meta + min_improvement:
                continue
            d['meta_score'] = effective_meta
            penalty_rows.append((d, penalty))

    filtered = []
    # Track names already added from bench to avoid duplicates
    bench_names = {b['card_title'] for b in bench_ups}

    for d, penalty in penalty_rows:
        title = d.get('card_title') or ''
        if any(name in title for name in exclude_names):
            continue
        if title in bench_names:
            continue  # Already captured from roster bench
        status, role = d.get('collection_status', ''), d.get('roster_role', '')
        if status == 'Inactive':
            d['action'] = 'Activate'
        elif status == 'Reserve Roster':
            d['action'] = 'Promote'
        elif role in ('bench', 'reserve'):
            d['action'] = 'Move Up'
        else:
            d['action'] = 'Swap In'
        d['position_penalty'] = round(penalty, 1)
        d['position_annotation'] = (
            format_position_annotation(d, pos_value) if not is_pitching else ""
        )
        filtered.append(d)

    # Merge: bench players first (they're already on the team), then cards table results
    combined = bench_ups + filtered
    combined.sort(key=lambda x: -(x['meta_score'] or 0))
    return combined[:limit]


def find_market_upgrades(pos_value, current_meta, is_pitching, exclude_ids=None, limit=5):
    """Find unowned market cards that upgrade the slot.

    Batting uses multi-position eligibility (primary position OR
    pos_rating_{slot} >= threshold) with a small penalty for non-primary
    assignments. Pitching keeps strict role matching.

    Budget note: instead of the hard ``last_10_price <= budget`` filter
    that silently hid aspirational upgrades, we now split results into
    two buckets and return both — in-budget cards first (ranked by meta),
    then "save up" cards up to 2× budget. The caller tags them via
    ``aspirational=True`` so the display layer can render a
    "Save Up" badge rather than pretending they don't exist.
    """
    exclude_ids = exclude_ids or set()
    meta_col = "meta_score_pitching" if is_pitching else "meta_score_batting"
    pos_col = "pitcher_role_name" if is_pitching else "position_name"
    max_price = max_spend if max_spend > 0 else 999999999
    # Aspirational cap: 2× budget. Catches genuine "save up" targets
    # without flooding the list with 10× cards the user will never buy.
    aspirational_cap = max_price * 2

    if is_pitching:
        # Pitching retains strict role matching.
        results = conn.execute(f"""
            SELECT card_id, card_title, tier_name, card_value,
                   {meta_col} as raw_meta, last_10_price, position_name
            FROM cards
            WHERE {pos_col} = ? AND owned = 0 AND last_10_price > 0
                AND last_10_price <= ? AND {meta_col} > ?
            ORDER BY {meta_col} DESC LIMIT ?
        """, (pos_value, aspirational_cap, current_meta + min_improvement,
              limit + len(exclude_ids) + 10)).fetchall()
        rows = []
        for r in results:
            d = dict(r)
            d['meta_score'] = d['raw_meta']
            d['position_penalty'] = 0.0
            d['position_annotation'] = ""
            d['aspirational'] = (d.get('last_10_price') or 0) > max_price
            rows.append(d)
    else:
        # Batting: eligibility-aware.
        rating_sql = select_rating_columns()
        where_frag, where_params = build_eligible_where_clause(pos_value)
        results = conn.execute(f"""
            SELECT card_id, card_title, tier_name, card_value,
                   {meta_col} as raw_meta, last_10_price, position_name,
                   {rating_sql}
            FROM cards
            WHERE {where_frag} AND owned = 0 AND last_10_price > 0
                AND pitcher_role IS NULL
                AND last_10_price <= ? AND {meta_col} > ?
            ORDER BY {meta_col} DESC LIMIT ?
        """, (*where_params, aspirational_cap, current_meta,
              limit + len(exclude_ids) + 15)).fetchall()

        rows = []
        for r in results:
            d = dict(r)
            penalty = position_meta_penalty(d, pos_value)
            effective = (d.get('raw_meta') or 0) - penalty
            if effective <= current_meta + min_improvement:
                continue
            d['meta_score'] = effective
            d['position_penalty'] = round(penalty, 1)
            d['position_annotation'] = format_position_annotation(d, pos_value)
            d['aspirational'] = (d.get('last_10_price') or 0) > max_price
            rows.append(d)

    filtered = [r for r in rows if r['card_id'] not in exclude_ids]
    # Rank: in-budget first (by meta), then aspirational (by meta).
    # This keeps the default list buyable while still surfacing the
    # "save up" targets below.
    affordable = [r for r in filtered if not r['aspirational']]
    aspirational = [r for r in filtered if r['aspirational']]
    affordable.sort(key=lambda x: -(x['meta_score'] or 0))
    aspirational.sort(key=lambda x: -(x['meta_score'] or 0))
    return (affordable + aspirational)[:limit]


def action_tag(owned_card):
    """Compact action label for an owned upgrade card."""
    a = owned_card.get('action', 'Swap')
    return f"FREE • {a}"


def price_tag(price):
    """Compact price label."""
    p = price or 0
    if p >= 10000:
        return f"Buy {p // 1000:.0f}K"
    elif p > 0:
        return f"Buy {p:,}"
    return "Buy"


_POS_TAGS = {'C ', '1B ', '2B ', '3B ', 'SS ', 'LF ', 'CF ', 'RF ', 'SP ', 'RP ', 'CL '}
_SET_PREFIXES = [
    "Live Collection Reward - ",
    "Veteran Presence ",
    "Historical All-Star ",
    "Hardware Heroes ",
    "All-Time Legend ",
    "Future Legend ",
    "Unsung Heroes ",
    "MLB 2026 Live ",
    "Snapshot ",
]


def _strip_set_prefix(card_title: str) -> tuple[str, str]:
    """Strip set prefix(es) + position tag from a card title.

    Returns (player_core, set_tag) where player_core is like
    'Steve O'Neill CLE 1919' and set_tag is like 'Snapshot'.
    Handles nested prefixes like 'Live Collection Reward - Historical All-Star 1B ...'.
    """
    if not card_title:
        return ("", "")
    t = card_title.strip()
    tag = ""
    # Strip prefixes repeatedly (handles "Live Collection Reward - Historical All-Star ...")
    changed = True
    while changed:
        changed = False
        for pfx in _SET_PREFIXES:
            if t.startswith(pfx):
                if not tag:
                    tag = pfx.strip().split()[0]  # First prefix becomes the tag
                t = t[len(pfx):]
                changed = True
                break
    # Strip position tag (e.g. "C ", "SP ", "1B ")
    if len(t) >= 3 and t[:3] in _POS_TAGS:
        t = t[3:]
    elif len(t) >= 2 and t[:2] == 'C ':
        t = t[2:]
    return (t.strip(), tag)


def short_name(card_title, max_len=28):
    """Smart truncation — strips set prefix to show player name + team.

    'Snapshot C Steve O'Neill CLE 1919' → 'Steve O'Neill CLE 1919'
    'Live Collection Reward - Historical All-Star 1B ...' → player name
    Falls back to raw truncation if stripping doesn't help.
    """
    if not card_title:
        return "\u2014"
    core, _ = _strip_set_prefix(card_title)
    if not core:
        core = card_title.strip()
    if len(core) <= max_len:
        return core
    return core[:max_len].rstrip() + "\u2026"


def full_card_tooltip(card_title):
    """Return the full card title for tooltip/mouseover."""
    return card_title or ""


# ── Build upgrade plan (chain: Current → Owned → Market) ──
used_market_ids = set()
used_owned_titles = set()
upgrade_plan = []

# Collect ALL active roster player names up-front so no active player can be
# recommended as an upgrade for another slot (e.g. CL shouldn't be suggested for MOP).
_all_active_names = set()
for _pos_key, _players in active_by_pos.items():
    for _p in _players:
        _all_active_names.add(_p['player_name'])


def _build_slot(pos_label, current_name, current_ovr, current_meta, owned_ups, market_ups,
                bats_hand='?', current_card_id=None):
    """Build one upgrade-plan entry with both owned and market stored separately."""
    bo = owned_ups[0] if owned_ups else None
    bm = market_ups[0] if market_ups else None

    # Owned upgrade: delta vs current
    owned_meta = round(bo['meta_score']) if bo else None
    owned_delta = round(bo['meta_score'] - current_meta) if bo else 0

    # Market upgrade: delta vs the owned upgrade if one exists, else vs current
    baseline_for_market = bo['meta_score'] if bo else current_meta
    market_meta = round(bm['meta_score']) if bm else None
    market_delta = round(bm['meta_score'] - baseline_for_market) if bm else 0

    # Track used IDs to prevent duplicates
    if bo:
        used_owned_titles.add(bo['card_title'])
    if bm:
        used_market_ids.add(bm['card_id'])

    # ── Attribute-mix / archetype layer ──
    # Attach fit_score + archetype for current / owned / market so the
    # Roster Optimizer can show the mix view alongside meta without a
    # second page. Deltas are archetype-fit deltas (0-100 scale), which
    # surface cards whose rating MIX predicts better outcomes even when
    # their headline meta is close.
    cur_arch = _fit_for(current_card_id)
    owned_arch = _fit_for(bo.get('card_id') if bo else None)
    market_arch = _fit_for(bm.get('card_id') if bm else None)
    cur_fit = cur_arch.get('fit_score')
    owned_fit_delta = (owned_arch.get('fit_score') - cur_fit) \
        if (owned_arch.get('fit_score') is not None and cur_fit is not None) else None
    market_fit_delta = (market_arch.get('fit_score') - cur_fit) \
        if (market_arch.get('fit_score') is not None and cur_fit is not None) else None

    return {
        'pos': pos_label,
        'current_name': current_name,
        'current_ovr': current_ovr,
        'current_meta': round(current_meta),
        'current_card_id': current_card_id,
        'bats': bats_hand,
        # Mix/archetype layer (from card_archetypes)
        'current_fit': cur_arch.get('fit_score'),
        'current_archetype': cur_arch.get('archetype_name'),
        'current_archetype_war': cur_arch.get('archetype_war'),
        'current_mix_score': cur_arch.get('mix_score'),
        'current_count_elite': cur_arch.get('count_elite'),
        # Owned upgrade (free)
        'owned_name': bo['card_title'] if bo else None,
        'owned_ovr': bo.get('card_value') if bo else None,
        'owned_meta': owned_meta,
        'owned_delta': owned_delta,
        'owned_action': bo.get('action') if bo else None,
        'owned_fit': owned_arch.get('fit_score'),
        'owned_fit_delta': owned_fit_delta,
        'owned_archetype': owned_arch.get('archetype_name'),
        # Market upgrade (paid)
        'market_name': bm['card_title'] if bm else None,
        'market_ovr': bm.get('card_value') if bm else None,
        'market_meta': market_meta,
        'market_delta': market_delta,
        'market_price': bm.get('last_10_price') if bm else None,
        'market_fit': market_arch.get('fit_score'),
        'market_fit_delta': market_fit_delta,
        'market_archetype': market_arch.get('archetype_name'),
        # For detail expanders
        '_owned_upgrades': owned_ups,
        '_market_upgrades': market_ups,
        # Best overall delta (for sorting priorities)
        'best_delta': max(
            round(bo['meta_score'] - current_meta) if bo else 0,
            round(bm['meta_score'] - current_meta) if bm else 0,
        ),
    }


for pos in show_positions:
    is_pitching = pos in ('SP', 'RP', 'CL')

    if pos == 'SP':
        # Use only actual rotation pitchers as "current", not bench/reserve
        sp_players = active_by_pos.get('SP', [])[:5]
        used_names = _all_active_names.copy()
        # Process WEAKEST first so best free upgrades go to worst slots
        order = sorted(range(len(sp_players)), key=lambda i: sp_players[i]['meta_score'] or 0)
        sp_entries = [None] * len(sp_players)
        for i in order:
            sp = sp_players[i]
            m = sp['meta_score'] or 0
            ow = find_owned_upgrades('SP', m, True, list(used_names), 3, current_player_name=sp['player_name'])
            mk = find_market_upgrades('SP', m, True, used_market_ids, 3)
            entry = _build_slot(f"SP{i+1}", sp['player_name'], sp['ovr'], m, ow, mk,
                                current_card_id=sp.get('card_id'))
            if entry['owned_name']:
                # Track PLAYER NAME to prevent same card recommended twice
                # Extract from the upgrade dict if available, else from card title
                bo = ow[0] if ow else None
                pname = bo.get('player_name', entry['owned_name']) if bo else entry['owned_name']
                used_names.add(pname)
                used_names.add(entry['owned_name'])  # Also add card title for cards-table exclusion
            sp_entries[i] = entry
        upgrade_plan.extend(sp_entries)
        continue

    if pos == 'RP':
        # Use only actual bullpen pitchers as "current"
        rp_players = active_by_pos.get('RP', [])[:7]
        used_names = _all_active_names.copy()
        slot_names = ["SU1", "SU2", "MID1", "MID2", "LNG1", "LNG2", "MOP"]
        # Process WEAKEST first so best free upgrades go to worst slots
        order = sorted(range(len(rp_players)), key=lambda i: rp_players[i]['meta_score'] or 0)
        rp_entries = [None] * len(rp_players)
        for i in order:
            rp = rp_players[i]
            m = rp['meta_score'] or 0
            ow = find_owned_upgrades('RP', m, True, list(used_names), 3, current_player_name=rp['player_name'])
            mk = find_market_upgrades('RP', m, True, used_market_ids, 3)
            label = slot_names[i] if i < len(slot_names) else f"RP{i+1}"
            entry = _build_slot(label, rp['player_name'], rp['ovr'], m, ow, mk,
                                current_card_id=rp.get('card_id'))
            if entry['owned_name']:
                bo = ow[0] if ow else None
                pname = bo.get('player_name', entry['owned_name']) if bo else entry['owned_name']
                used_names.add(pname)
                used_names.add(entry['owned_name'])
            rp_entries[i] = entry
        upgrade_plan.extend(rp_entries)
        continue

    # ── DH slot: file > manual override > inferred from "extras" ──
    # OOTP 27 DOES export a DH row — IF you manually set a player's team
    # position to DH in-game. When that happens, the roster / lineup CSVs
    # include POS=DH and we pick it up into active_by_pos['DH']. Priority:
    #   1. File-exported DH (authoritative — OOTP knows the truth).
    #   2. Manual sidebar override (power user pinning someone else).
    #   3. Inference from active-roster "extras" (legacy fallback for when
    #      the user hasn't set a DH in OOTP yet).
    if pos == 'DH':
        player = None
        dh_source = 'inferred'  # 'file' | 'manual' | 'inferred' | 'empty'

        # Step 1: manual override wins if set (explicit user intent).
        _manual_dh = st.session_state.get('_dh_override_name')
        if _manual_dh:
            for _fp in bat_field_positions + ['DH']:
                for _p in active_by_pos.get(_fp, []):
                    if _p.get('player_name') == _manual_dh:
                        player = _p
                        dh_source = 'manual'
                        break
                if player is not None:
                    break

        # Step 2: file-exported DH (only when there's no manual override).
        # If multiple DH rows exist, take the highest-meta one.
        if player is None:
            _file_dh = active_by_pos.get('DH', [])
            if _file_dh:
                player = sorted(_file_dh, key=lambda p: p.get('meta_score') or 0, reverse=True)[0]
                dh_source = 'file'

        # Step 3: legacy inference fallback (highest-meta extra starter at
        # any field pos). Only runs when Cameron hasn't marked anyone as DH
        # in OOTP AND hasn't pinned one via the sidebar.
        if player is None:
            _field_names = {e['current_name'] for e in upgrade_plan if e['pos'] in bat_field_positions}
            dh_candidates = []
            for fpos in bat_field_positions:
                for p in active_by_pos.get(fpos, []):
                    if p['player_name'] not in _field_names:
                        dh_candidates.append(p)
            dh_candidates.sort(key=lambda p: p['meta_score'] or 0, reverse=True)
            if dh_candidates:
                player = dh_candidates[0]
                dh_source = 'inferred'

        if player is not None:
            m = player['meta_score'] or 0
            bh = player.get('bats_hand', '?')
            # Exclude already-committed owned upgrades so Jackie Bradley Jr
            # (promoted for CF) can't be promoted again for DH. Titles go
            # into the exclude list so find_owned_upgrades' substring filter
            # drops them. Plus the current DH player's own name so they
            # don't show up as their own "upgrade".
            active_names = list(_all_active_names) + list(used_owned_titles)
            # DH upgrades search ALL batting positions — anyone can DH
            ow = find_owned_upgrades('DH', m, False, active_names, 3, current_player_name=player['player_name'])
            mk = find_market_upgrades('DH', m, False, used_market_ids, 3)
            entry = _build_slot('DH', player['player_name'], player['ovr'], m, ow, mk, bats_hand=bh,
                                current_card_id=player.get('card_id'))
            entry['is_platoon'] = False
            # 'dh_inferred' is True ONLY when we're guessing. File-exported
            # DH and manual overrides are both treated as truth — no "?" marker.
            entry['dh_inferred'] = (dh_source not in ('manual', 'file'))
            entry['dh_source'] = dh_source
            if entry['owned_name']:
                bo = ow[0] if ow else None
                pname = bo.get('player_name', entry['owned_name']) if bo else entry['owned_name']
                used_owned_titles.add(pname)
            upgrade_plan.append(entry)
        else:
            # No extra starters AND no override — DH is empty, suggest best
            # available hitter. This also fires when the roster snapshot is
            # stale. The staleness banner at the top should already warn.
            entry = _build_slot('DH', '(empty)', 0, 0,
                find_owned_upgrades('DH', 0, False, list(_all_active_names) + list(used_owned_titles), 5),
                find_market_upgrades('DH', 0, False, used_market_ids, 5))
            entry['dh_inferred'] = True
            entry['dh_source'] = 'empty'
            upgrade_plan.append(entry)
        continue

    # ── Standard batting positions: 1 slot per position (best starter) ──
    active_players = active_by_pos.get(pos, [])
    if not active_players:
        upgrade_plan.append(_build_slot(pos, '(empty)', 0, 0,
            find_owned_upgrades(pos, 0, is_pitching, list(used_owned_titles), 5),
            find_market_upgrades(pos, 0, is_pitching, used_market_ids, 5)))
        continue

    # Prefer the observed-starter override (from recent box-score pins) if
    # one is defined for this position; else fall back to highest-meta
    # active player. This ensures the chain table reflects the lineup the
    # user is ACTUALLY running (e.g. Kluttz at C, Al Dark at SS) rather
    # than the best-by-meta player in the depth chart.
    _observed_starter = starters.get(pos)
    if (_observed_starter
            and _observed_starter.get('player_name') in {p.get('player_name') for p in active_players}):
        # Move the observed starter to the front of active_players; keep
        # the rest in meta desc order so upgrade candidates still rank.
        _obs_name = _observed_starter.get('player_name')
        _head = [p for p in active_players if p.get('player_name') == _obs_name]
        _tail = sorted(
            [p for p in active_players if p.get('player_name') != _obs_name],
            key=lambda p: p['meta_score'] or 0, reverse=True,
        )
        active_players = _head + _tail
    else:
        active_players = sorted(active_players,
                                 key=lambda p: p['meta_score'] or 0, reverse=True)
    player = active_players[0]
    m = player['meta_score'] or 0
    bh = player.get('bats_hand', '?')
    # Exclude already-committed owned upgrade titles so the same card can't
    # be recommended across multiple slots (e.g. Jackie Bradley Jr for CF
    # AND DH). used_owned_titles is populated inside _build_slot each time
    # a slot commits an upgrade, and find_owned_upgrades treats the list as
    # substring exclusions.
    active_names = list(_all_active_names) + [p['player_name'] for p in active_players] + list(used_owned_titles)
    ow = find_owned_upgrades(pos, m, is_pitching, active_names, 3, current_player_name=player['player_name'])
    mk = find_market_upgrades(pos, m, is_pitching, used_market_ids, 3)
    entry = _build_slot(pos, player['player_name'], player['ovr'], m, ow, mk, bats_hand=bh,
                        current_card_id=player.get('card_id'))
    entry['is_platoon'] = False
    if entry['owned_name']:
        bo = ow[0] if ow else None
        pname = bo.get('player_name', entry['owned_name']) if bo else entry['owned_name']
        used_owned_titles.add(pname)
    # Platoon gap warning — DISABLED for batting positions.
    # Rationale: each batting position has one starter, so "platoon issues"
    # at individual slots aren't meaningful. The bigger-picture handedness
    # balance gets surfaced by Manager's Eye as a critical gap when it matters.
    upgrade_plan.append(entry)

if focus == "Weakest First":
    upgrade_plan.sort(key=lambda x: x['current_meta'])

# ── Performance driver pass ──
# Tag each slot with its full driver analysis (perf_analysis dict). This is
# purely informational now — we do NOT filter upgrades based on it. The lock
# that used to live here was backwards: small-sample hot streaks are the most
# common cause of overperformance, and locking the slot hid the upgrade
# during exactly the window when regression was most likely. Pitchers vs
# batters is determined by the slot prefix via _is_pitching_pos().
for _u in upgrade_plan:
    _is_pit_slot = _is_pitching_pos(_u.get('pos') or '')
    _analysis = _analyze_perf_driver(
        _u['current_name'], _u['current_meta'], _is_pit_slot
    )
    _u['perf_analysis'] = _analysis
    # Back-compat fields for any code that still reads them — the lock is dead.
    _u['outperforming'] = False
    _u['perf_meta_estimate'] = _analysis['perf_meta'] if _analysis else None

# ── Classify ──
collection_swaps = [u for u in upgrade_plan if u['owned_name']]
# Market buys are no longer filtered by outperformance. The upgrade list is
# the upgrade list — we just annotate with driver info so the user can see
# which slots are riding hot streaks and which are producing for real.
market_buys = [u for u in upgrade_plan if u['market_name']]

# ── Anti-oscillation filter ──
# Suppress owned-promotion recs that would reverse a recently-actioned
# swap unless the delta clears a stability floor. Fixes the flip-flop
# where Bo Bichette → Al Dark gets recommended, then next refresh says
# Al Dark → Bo Bichette because performance overlay caught up.
try:
    from app.core.rec_hysteresis import filter_upgrade_plan
    upgrade_plan = filter_upgrade_plan(upgrade_plan, min_flip_delta=50.0,
                                         lookback_hours=72)
    market_buys = [u for u in upgrade_plan if u['market_name']]
except Exception as _hys_err:
    import logging as _lg
    _lg.getLogger(__name__).debug("hysteresis filter failed: %s", _hys_err)

all_upgrades = [u for u in upgrade_plan if u['owned_name'] or u['market_name']]
top_priorities = sorted(all_upgrades, key=lambda x: -x['best_delta'])[:3]

# ── Recommended batting order (1–9) ──
# Compute the engine's prescriptive "who should bat where" from ratings,
# override the observed-BO map so the chain table shows the IDEAL order
# rather than whatever happened in the last played game. Chain rows also
# get sorted by this BO (see build_chain_rows rendering logic below).
try:
    from app.core.batting_order import compute_batting_order as _compute_bo_fn
    _bo_starter_names = [
        u.get('current_name') for u in upgrade_plan
        if u.get('pos') in bat_field_positions + ['DH']
        and u.get('current_name') and u['current_name'] != '(empty)'
    ]
    _bo_players: list[dict] = []
    for _name in _bo_starter_names:
        _card = conn.execute(
            """SELECT contact, gap_power, power, eye, avoid_ks, babip,
                      speed, stealing, baserunning
               FROM cards WHERE card_title LIKE ? AND owned = 1 LIMIT 1""",
            (f'%{_name}%',),
        ).fetchone()
        _meta = next((e.get('current_meta') for e in upgrade_plan
                      if e.get('current_name') == _name), 0)
        entry = dict(_card) if _card else {}
        entry['player_name'] = _name
        entry['meta_score'] = _meta
        _bo_players.append(entry)
    _observed_bo_by_name = _compute_bo_fn(_bo_players)
except Exception as _bo_err:
    import logging as _lg
    _lg.getLogger(__name__).debug("recommended BO compute failed: %s", _bo_err)
    _observed_bo_by_name = {}

# ── Log the canonical engine picks to the recommendation tracker ──
# This is the single source of truth for recommendations. Each slot's
# Owned Promotion + Market Upgrade becomes one rec row. The background
# worker's council_sweep picks these up, fires LLM verification, and
# stores verdicts against them — which surface in the chain-table
# tooltips as inline council commentary.
try:
    from app.core.recommendation_tracker import log_recommendations
    _engine_picks = []
    for _u in upgrade_plan:
        if _u.get('owned_name') and (_u.get('owned_delta') or 0) > 0:
            _engine_picks.append({
                'pos': _u.get('pos'),
                'action': 'Promote',
                'card_name': _u.get('owned_name'),
                'current_name': _u.get('current_name'),
                'expected_delta': _u.get('owned_delta'),
                'reason': _u.get('owned_action') or 'engine: owned promotion',
            })
        if _u.get('market_name') and (_u.get('market_delta') or 0) > 0:
            _engine_picks.append({
                'pos': _u.get('pos'),
                'action': 'Buy',
                'card_name': _u.get('market_name'),
                'current_name': _u.get('current_name'),
                'cost': _u.get('market_price'),
                'expected_delta': _u.get('market_delta'),
                'reason': f"engine: market upgrade ({_u.get('market_price','?')} PP)",
            })
    if _engine_picks:
        try:
            from app.utils.live_status import get_data_version as _gdv
            _rec_dv = _gdv()
        except Exception:
            _rec_dv = 0
        _rec_league = None
        try:
            _rec_league = (load_config() or {}).get('active_league')
        except Exception:
            pass
        log_recommendations(
            'meta_engine',
            _engine_picks,
            league_id=_rec_league,
            data_version=_rec_dv,
        )
except Exception as _rec_log_err:
    # Silent: don't spam caption on every rerun. Tracker sweep in the
    # background worker will retry and log its own errors if needed.
    import logging as _lg
    _lg.getLogger(__name__).debug("upgrade_plan rec log failed: %s", _rec_log_err)

# Flat pool of active 26-man players — used by _find_drop_candidate() to
# suggest alternative drops when the default "displace current starter"
# would cut a cold-but-unlucky piece with rebound upside. Each entry
# includes the precomputed perf_analysis so the drop picker can filter out
# candidates who are *themselves* rebound cases without re-running the
# analyzer on every call.
_active_pool = []
for _pos_players in active_by_pos.values():
    for _p in _pos_players:
        _p_pos = _p.get('position') or ''
        _p_is_pit = _is_pitching_pos(_p_pos)
        _p_analysis = _analyze_perf_driver(
            _p.get('player_name'), _p.get('meta_score') or 0, _p_is_pit
        )
        _active_pool.append({
            'name': _p.get('player_name'),
            'pos': _p_pos,
            'ovr': _p.get('ovr'),
            'meta': _p.get('meta_score'),
            'role': _p.get('lineup_role'),
            'is_pit': _p_is_pit,
            'perf_analysis': _p_analysis,
        })

# ── Roster mismatches ──
# Only flag when a bench player genuinely beats the starter AND the starter
# isn't already producing real WAR in-game (WAR/600 >= 1.5 is the gate — a
# producing starter shouldn't be benched for someone with no stats).
roster_fixes = []
for pos in bat_field_positions + ['CL']:
    pp = all_by_pos.get(pos, [])
    if len(pp) < 2: continue
    best = pp[0]
    if best.get('lineup_role') not in ('starter', 'rotation', 'closer', 'bullpen'):
        for p in pp:
            if p.get('lineup_role') in ('starter', 'rotation', 'closer', 'bullpen'):
                d = round((best['meta_score'] or 0) - (p['meta_score'] or 0))
                if d >= min_improvement:
                    # Performance gate: don't bench a player producing real WAR
                    # for someone with no performance data
                    starter_perf = _perf_bat.get(p['player_name']) or _perf_pit.get(p['player_name'])
                    bench_perf = _perf_bat.get(best['player_name']) or _perf_pit.get(best['player_name'])
                    if starter_perf and not bench_perf:
                        # Starter has stats, bench player doesn't — skip if starter is producing
                        war600 = starter_perf.get('war600', starter_perf.get('war200', 0))
                        if war600 >= 1.5:  # decent+ production
                            break  # Don't flag — starter is proving their value in-game
                    roster_fixes.append({'pos': pos, 'starter': p['player_name'],
                        'starter_meta': round(p['meta_score'] or 0),
                        'better': best['player_name'],
                        'better_meta': round(best['meta_score'] or 0),
                        'role': best['lineup_role'], 'delta': d})
                break

# ════════════════════════════════════════════════════════════════
# HEADER — Team Grade + AI Assessment + Quick Stats
# ════════════════════════════════════════════════════════════════
st.title("Roster Optimizer")

# Background worker + live status — starts the watcher thread on first load
# and renders a live badge that auto-refreshes every 10s so the user can see
# when new exports have been ingested without reloading the page.
try:
    from app.utils.live_status import (
        ensure_worker_running, live_header, staleness_reminder, staleness_glance,
    )
    ensure_worker_running()
    live_header()
    # Always-on glance banner (auto-refreshes every 30s). Shows the single
    # most pressing export gap — relative lag or absolute staleness — so
    # Cameron knows what to re-export without expanding anything.
    staleness_glance()
    with st.expander("\U0001f4cb All export groups", expanded=False):
        staleness_reminder()
except Exception as _live_err:
    st.caption(f"Live status unavailable: {_live_err}")

# Freshness banner — warn when the cards list is newer than the roster snapshot.
# If you import pt_card_list.csv after acquiring a card but don't re-export the
# Toronto team CSVs, the optimizer will still see that card as reserve and
# recommend it as a "free promote" — misleading. This banner catches that.
_fresh = get_data_freshness(conn)
if _fresh["is_stale"]:
    _gap_h = _fresh["staleness_hours"] or 0
    _gap_txt = f"{_gap_h:.1f}h" if _gap_h >= 1 else f"{int(_gap_h * 60)}min"
    _stale_list = ", ".join(_fresh["stale_file_types"]) or "team roster exports"
    st.warning(
        f"\u26a0\ufe0f **Roster snapshot is {_gap_txt} older than your card list.** "
        f"Promotion recommendations may be based on stale active/reserve assignments. "
        f"Re-export Toronto team CSVs ({_stale_list}) from OOTP and run Import All in Settings.",
        icon="\U0001f552",
    )

# Calculate team grade
bat_metas = [starters[p]['meta_score'] for p in bat_field_positions if p in starters]
pit_metas = [p['meta_score'] for p in active_by_pos.get('SP', [])[:5]]
pit_metas += [p['meta_score'] for p in active_by_pos.get('RP', [])[:7]]
if 'CL' in starters:
    pit_metas.append(starters['CL']['meta_score'])
all_metas = [m for m in bat_metas + pit_metas if m]
avg_meta = sum(all_metas) / len(all_metas) if all_metas else 0
total_meta = sum(all_metas)

# ── Team grade: avg_meta baseline + optimization penalty ──
# A+ should mean "elite cards AND you're executing on every available move."
# A roster with 7 free upgrades on the bench isn't A+ no matter how high the
# avg_meta is — those free upgrades represent an optimization gap you haven't
# closed yet. Grade reflects roster management, not just card quality.
_GRADE_LADDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D"]

# Base grade from avg_meta (tightened — prior thresholds were too lenient;
# A+ was given at avg_meta=700, but any reasonable Bronze-tier team clears
# that bar. True A+ should require ~850+ which is elite territory.)
if avg_meta >= 850:   _base_idx = 0   # A+
elif avg_meta >= 780: _base_idx = 1   # A
elif avg_meta >= 720: _base_idx = 2   # A-
elif avg_meta >= 660: _base_idx = 3   # B+
elif avg_meta >= 600: _base_idx = 4   # B
elif avg_meta >= 540: _base_idx = 5   # B-
elif avg_meta >= 480: _base_idx = 6   # C+
elif avg_meta >= 420: _base_idx = 7   # C
elif avg_meta >= 360: _base_idx = 8   # C-
else:                 _base_idx = 9   # D

# Optimization penalties:
#   - Every 3 free upgrades sitting uncommitted = -1 step (½ letter)
#   - Every wrong starter = -1 step (fundamental management error)
# Free upgrades are worse than "roster could be better" — they're moves the
# user literally has made no excuse not to make, so each one compounds.
_free_upgrade_count = len(collection_swaps)
_wrong_starter_count = len(roster_fixes) if roster_fixes else 0
_penalty_steps = (_free_upgrade_count // 3) + _wrong_starter_count

_final_idx = min(_base_idx + _penalty_steps, len(_GRADE_LADDER) - 1)
grade = _GRADE_LADDER[_final_idx]

# Keep the reason so we can surface it in the tooltip
_grade_reason_parts = []
if _free_upgrade_count >= 3:
    _grade_reason_parts.append(f"{_free_upgrade_count} free upgrades (-{_free_upgrade_count // 3} steps)")
if _wrong_starter_count:
    _grade_reason_parts.append(f"{_wrong_starter_count} wrong starter(s) (-{_wrong_starter_count} step)")
_grade_reason = (
    f"Base {_GRADE_LADDER[_base_idx]} from avg meta {avg_meta:.0f}; penalties: "
    + "; ".join(_grade_reason_parts)
    if _grade_reason_parts else
    f"Base {_GRADE_LADDER[_base_idx]} from avg meta {avg_meta:.0f}; no optimization gaps"
)

col_grade, col_ai, col_stats = st.columns([1, 4, 2])
with col_grade:
    st.metric("Team Grade", grade, help=_grade_reason)
    st.metric("Avg Meta", f"{avg_meta:.0f}")

with col_stats:
    st.metric("Free Upgrades", len(collection_swaps))
    mkt_cost = sum(u['market_price'] or 0 for u in market_buys)
    st.metric("Market Upgrades", f"{len(market_buys)} ({mkt_cost:,.0f} PP)")
    if roster_fixes:
        st.metric("Roster Fixes", len(roster_fixes), delta="wrong starters", delta_color="inverse")

with col_ai:
    ai_config = get_ai_config()
    if ai_config["ready"]:
        # Build a focused prompt for team assessment
        team_ctx = build_team_context(conn)
        # Summarize upgrade plan for AI
        upgrade_summary = []
        for u in top_priorities:
            if u['owned_name']:
                upgrade_summary.append(f"{u['pos']}: {u['current_name']} -> {short_name(u['owned_name'])} (+{u['owned_delta']} meta, FREE {u['owned_action']})")
            if u['market_name']:
                upgrade_summary.append(f"{u['pos']}: Buy {short_name(u['market_name'])} (+{u['market_delta']} meta, {price_tag(u['market_price'])})")
        for fix in roster_fixes:
            upgrade_summary.append(f"MISMATCH {fix['pos']}: {fix['starter']} should be {fix['better']} (+{fix['delta']})")

        assessment_prompt = (
            f"Team: Toronto Dark Knights\nGrade: {grade} (avg meta {avg_meta:.0f})\n"
            f"Batting avg meta: {sum(bat_metas) / len(bat_metas):.0f}\n"
            f"Pitching avg meta: {sum(m for m in pit_metas if m) / max(len([m for m in pit_metas if m]), 1):.0f}\n\n"
            f"Top moves:\n" + "\n".join(upgrade_summary) + "\n\n"
            f"Team context:\n{team_ctx}"
        )

        @st.cache_data(ttl=3600, show_spinner=False)
        def _get_assessment(prompt_hash):
            from app.core.ai_advisor import _call_gemini, _call_anthropic
            sys_prompt = (
                "You are a baseball GM evaluating a Perfect Team roster. In 4-5 concise sentences: "
                "1) Grade the overall roster strength and identify the biggest gap. "
                "2) Name the top 3 priority moves with specific player names. "
                "3) Tag each as FREE or BUY with cost. "
                "4) Note any strategic concerns (e.g. platoon balance, defense gaps). "
                "Be direct and specific. No fluff."
            )
            ai_cfg = get_ai_config()
            if ai_cfg["provider"] == "gemini":
                return _call_gemini(sys_prompt, prompt_hash, ai_cfg)
            return _call_anthropic(sys_prompt, prompt_hash, ai_cfg)

        try:
            result = _get_assessment(assessment_prompt)
            if result.get('response'):
                st.info(result['response'])
            else:
                st.caption("AI assessment unavailable")
        except Exception as e:
            st.caption(f"AI assessment error: {e}")
    else:
        # Fallback static assessment
        weakest = min(upgrade_plan, key=lambda x: x['current_meta'])
        strongest = max(upgrade_plan, key=lambda x: x['current_meta'])
        st.info(
            f"**{grade} roster** (avg meta {avg_meta:.0f}). "
            f"Strongest: **{strongest['pos']}** ({strongest['current_name']}, {strongest['current_meta']}). "
            f"Weakest: **{weakest['pos']}** ({weakest['current_name']}, {weakest['current_meta']}). "
            f"**{len(collection_swaps)}** free upgrades available, **{len(market_buys)}** on market."
        )

# ════════════════════════════════════════════════════════════════
# AI OPTIMIZE ALL — auto-fires on first load, cached per roster hash
#
# UX goal: no manual button clicks. The AI reasoning kicks off in the
# background as soon as the page renders, uses its result to enrich the
# lineup tables (via _get_ai_pick_for_pos), and surfaces the full picks
# summary in a dedicated panel below the tabs.
#
# Cache key is the 26-man roster hash (same hash Manager's Eye uses) so
# results persist across page refreshes within a session — you only pay
# for the API call when the roster actually changes.
# ════════════════════════════════════════════════════════════════
ai_config_check = get_ai_config()
# Compute the roster hash here (earlier than the Manager's Eye block
# needs it) so both AI auto-fires share the same cache key and invalidate
# together when the lineup actually changes. Manager's Eye will pick up
# this same _roster_hash variable when its block runs below.
_roster_sig = tuple(sorted(
    (p['player_name'], p.get('meta_score') or 0, p.get('lineup_role') or '')
    for _pp in active_by_pos.values() for p in _pp
))
_roster_hash = str(hash(_roster_sig))
# Include data_version so new HTML/CSV ingestion invalidates cached AI results.
# Without this, the AI picks would stay stale even after new box scores land.
try:
    from app.utils.live_status import get_data_version
    _dv_for_cache = get_data_version()
except Exception:
    _dv_for_cache = 0
_ai_optimize_cache_key = f'ai_optimize_result_{_roster_hash}_v{_dv_for_cache}'
# ── NO auto-firing AI ──
# The page used to block 10-15s on AI Optimize All here. That blocking call
# has been removed per the LLM-as-verifier vision: the meta engine is the
# source of recommendations, LLMs only verify on-demand. A cached AI result
# for THIS exact roster/data_version is still restored if one exists, so
# previously-run verifications survive across page loads.
_cached_ai_result = st.session_state.get(_ai_optimize_cache_key) if ai_config_check["ready"] else None
if _cached_ai_result:
    st.session_state['ai_optimize_result'] = _cached_ai_result
    st.session_state['ai_optimize_picks'] = _cached_ai_result.get('picks', {})
    st.session_state['ai_optimize_picks_data'] = _cached_ai_result.get('picks_data', [])

def _get_ai_pick_for_pos(pos):
    """Get parsed AI pick data for a position, if AI has been run."""
    picks_data = st.session_state.get('ai_optimize_picks_data', [])
    for p in picks_data:
        if p['pos'] == pos:
            return p
    return None


# ════════════════════════════════════════════════════════════════
# TOP 3 PRIORITIES — AI-aware when available
# ════════════════════════════════════════════════════════════════
if top_priorities:
    st.markdown("##### Top Priority Moves")
    pri_cols = st.columns(min(len(top_priorities), 3))
    for i, u in enumerate(top_priorities):
        with pri_cols[i]:
            with st.container(border=True):
                ai_pick = _get_ai_pick_for_pos(u['pos'])
                st.caption(f"PRIORITY #{i+1} — {u['pos']}: {u['current_name']}")
                if ai_pick and ai_pick['action'] != 'Keep':
                    # Show AI pick as the priority
                    emoji = ai_pick.get('emoji', '')
                    if ai_pick['action'] == 'Promote':
                        st.success(f"{emoji} 📦 **{ai_pick['card_name']}** • {ai_pick.get('reason', '')}")
                    elif ai_pick['action'] == 'Buy':
                        cost = ai_pick.get('cost')
                        cost_str = f" • {cost:,}PP" if cost else ""
                        st.warning(f"{emoji} 🛒 **{ai_pick['card_name']}**{cost_str} • {ai_pick.get('reason', '')}")
                    elif ai_pick['action'] == 'Platoon':
                        st.info(f"{emoji} 🤝 **{ai_pick['card_name']}** • {ai_pick.get('reason', '')}")
                else:
                    # No AI — fall back to meta-based
                    if u['owned_name']:
                        st.success(f"📦 {short_name(u['owned_name'])}  **+{u['owned_delta']}** meta  •  {u['owned_action']}")
                    if u['market_name']:
                        st.warning(f"🛒 {short_name(u['market_name'])}  **+{u['market_delta']}** meta  •  {price_tag(u['market_price'])}")

# ════════════════════════════════════════════════════════════════
# ROSTER MISMATCHES — compact warning
# ════════════════════════════════════════════════════════════════
if roster_fixes:
    with st.container(border=True):
        st.markdown("**\u26a0\ufe0f Wrong Players Starting** — promote in-game now:")
        for fix in sorted(roster_fixes, key=lambda x: -x['delta']):
            st.markdown(
                f"\u2022 **{fix['pos']}**: Start **{fix['better']}** ({fix['better_meta']}) "
                f"over {fix['starter']} ({fix['starter_meta']}) — "
                f"currently on {fix['role']} **(+{fix['delta']})**"
            )

# ════════════════════════════════════════════════════════════════
# 🎯 MANAGER'S EYE — Staged load
# We reserve an st.empty() slot here so the panel appears near the top,
# but defer the blocking Gemini call until AFTER the lineup tabs render.
# That way the user can interact with the batting/pitching tables
# while the model thinks. Cached per-roster so we only hit the API
# when the 26-man roster actually changes.
# ════════════════════════════════════════════════════════════════
st.divider()

_ai_ready = ai_config["ready"]

# _roster_hash / _roster_sig were computed earlier (above the AI Optimize
# All block) so both AI auto-fires share the same cache invalidation key.
# Just derive the Manager's Eye session-state key from it here.
_unified_key = f"unified_analysis_{_roster_hash}"

# Reserve the DOM slot at the top of the page — filled either immediately
# from session cache, or deferred until the bottom of the script.
_mgr_eye_slot = st.empty()


def _render_managers_eye(slot, ua):
    """Render full Manager's Eye output into an st.empty() slot.

    Called twice in the page lifecycle:
      1. Inline, if we already have a cached analysis for this roster.
      2. At the bottom of the script, after the blocking Gemini call
         fills st.session_state[_unified_key] — this is the staged-load
         path so the user can use the tables while waiting.
    """
    if not ua:
        return
    _analysis = ua.get("analysis") or {}
    if ua.get("error") and not _analysis:
        slot.warning(f"Manager's Eye unavailable: {ua['error']}")
        return
    if not _analysis:
        return
    with slot.container():
        _team_id = _analysis.get("team_identity", "")
        _gaps = _analysis.get("critical_gaps", []) or []
        _sanity = _analysis.get("sanity_check", "")
        with st.container(border=True):
            if _team_id:
                st.markdown(f"**🎯 Team Identity:** {_team_id}")
            if _gaps:
                _gap_cols = st.columns(min(len(_gaps), 4))
                for i, g in enumerate(_gaps[:4]):
                    with _gap_cols[i]:
                        sev = (g.get("severity") or "moderate").lower()
                        icon = "🔴" if sev == "critical" else "🟡" if sev == "moderate" else "🔵"
                        affects = g.get("affects_position", "")
                        pos_tag = f" *({affects})*" if affects else ""
                        st.markdown(f"{icon} **{g.get('gap', '')}**{pos_tag}")
            if _sanity:
                st.caption(f"💬 {_sanity}")

        _sliders = _analysis.get("strategy_sliders", []) or []
        _priorities = _analysis.get("upgrade_priorities", []) or []
        _comp_notes = _analysis.get("composition_notes", []) or []

        _detail_cols = st.columns([3, 2])
        with _detail_cols[0]:
            with st.expander("🎮 Strategy Slider Recommendations", expanded=False):
                if _sliders:
                    st.caption("Apply in OOTP under **Strategy → Overall Strategy**. "
                               "Scale: 1=Never/Conservative, 3=Neutral, 5=Aggressive.")
                    _slider_rows = [{
                        "Category": s.get("category", ""),
                        "Slider": s.get("name", ""),
                        "Value": s.get("value", 3),
                        "Reason": s.get("reason", ""),
                    } for s in _sliders]
                    st.dataframe(pd.DataFrame(_slider_rows), use_container_width=True,
                                 hide_index=True,
                                 column_config={
                                     "Category": st.column_config.TextColumn(width="small"),
                                     "Slider": st.column_config.TextColumn(width="medium"),
                                     "Value": st.column_config.ProgressColumn(
                                         min_value=1, max_value=5, format="%d", width="small"),
                                     "Reason": st.column_config.TextColumn(width="large"),
                                 })
                else:
                    st.caption("(no slider recommendations returned)")

        with _detail_cols[1]:
            with st.expander("🎯 What to Acquire Next", expanded=False):
                if _priorities:
                    for p in _priorities[:5]:
                        st.markdown(f"**{p.get('role', '')}** — {p.get('reason', '')}")
                else:
                    st.caption("(no role priorities returned)")
                if _comp_notes:
                    st.markdown("---")
                    st.caption("**Composition notes:**")
                    for n in _comp_notes[:3]:
                        st.caption(f"• {n}")

        _tokens = ua.get("tokens_used", 0)
        _model = ua.get("model", "?")
        _from_cache = ua.get("_from_cache")
        _cached_at = ua.get("_cached_at")
        if _from_cache and _cached_at:
            _cache_note = f" • cached {_cached_at} UTC"
        elif _from_cache:
            _cache_note = " • cached"
        else:
            _cache_note = ""
        _foot_l, _foot_r = st.columns([5, 1])
        with _foot_l:
            st.caption(
                f"*Manager's Eye: {_model} • {_tokens:,} tokens"
                f"{_cache_note} • only re-runs when your active roster changes*"
            )
        with _foot_r:
            if st.button("🔄 Re-run", key="mgr_eye_refresh_btn",
                         help="Force a fresh Gemini call (ignores cache)."):
                # Bust both caches and let the deferred call below re-fire.
                st.session_state.pop(_unified_key, None)
                try:
                    conn.execute(
                        "DELETE FROM manager_eye_cache WHERE roster_hash = ?",
                        (_roster_hash,),
                    )
                    conn.commit()
                except Exception:
                    pass
                st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# Cache strategy (cost-aware):
#   1. Session cache  → instant, free
#   2. DB cache       → instant, free, survives page refreshes
#   3. Explicit click → fires the Gemini call, writes both caches
#
# We do NOT auto-fire Gemini on first page load. Earlier behavior was that any
# fresh page visit (including a refresh) triggered an unsolicited API call, so
# even a quick "did the page render?" check burned tokens. The button gate
# preserves the staged-load UX (panel near the top, lineup tables usable while
# the model thinks) but only after explicit user intent.
# ────────────────────────────────────────────────────────────────────────────
import json as _mgr_json

def _load_mgr_eye_from_db(rh: str):
    """Return cached unified-analysis dict for this roster_hash, or None."""
    try:
        row = conn.execute(
            "SELECT payload, model, tokens, created_at FROM manager_eye_cache WHERE roster_hash = ?",
            (rh,),
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    try:
        analysis = _mgr_json.loads(row['payload'])
    except Exception:
        return None
    return {
        "analysis": analysis,
        "raw_response": None,
        "tokens_used": row['tokens'] or 0,
        "model": row['model'] or "?",
        "error": None,
        "_from_cache": True,
        "_cached_at": row['created_at'],
    }

def _save_mgr_eye_to_db(rh: str, ua: dict) -> None:
    """Persist a successful unified analysis so refreshes don't re-fire Gemini."""
    if not ua or ua.get('error') or not ua.get('analysis'):
        return
    try:
        conn.execute(
            "INSERT OR REPLACE INTO manager_eye_cache "
            "(roster_hash, payload, model, tokens, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (
                rh,
                _mgr_json.dumps(ua['analysis']),
                ua.get('model') or "?",
                int(ua.get('tokens_used') or 0),
            ),
        )
        conn.commit()
    except Exception:
        pass

# Cache-only rendering — no auto-fire. Manager's Eye is deprecated in favor
# of the on-demand Council Review flow. If a prior session or DB cache has
# a result for this roster, we render it (free); otherwise the slot shows a
# button that lets the user opt into a council review on demand.
_need_mgr_eye_call = False   # kept for legacy code paths below; always False now
if not _ai_ready:
    _mgr_eye_slot.info(f"Council Review needs AI advisor configured: {ai_config.get('message', 'not configured')}")
elif _unified_key in st.session_state:
    _render_managers_eye(_mgr_eye_slot, st.session_state[_unified_key])
else:
    _db_cached = _load_mgr_eye_from_db(_roster_hash)
    if _db_cached:
        st.session_state[_unified_key] = _db_cached
        _render_managers_eye(_mgr_eye_slot, _db_cached)
    else:
        # No auto-fire. The user can trigger a council review from the
        # Council Review section added later in the page.
        _mgr_eye_slot.empty()


# ════════════════════════════════════════════════════════════════
# LINEUP CARD — batting + pitching side by side
# ════════════════════════════════════════════════════════════════
st.divider()


# Role grouping for upgrade-suggestion consolidation. Within a group,
# only the N weakest slots (by current_meta) surface market-buy actions;
# the rest read "Optimal". Non-CL bullpen roles (SU/MID/LNG/MOP) all
# pool into one 'BP' group — the fiction that they were independent
# categories flagged a hot MOP for upgrade just because he was the only
# MOP. In practice the bullpen is a pool; rank all non-CL relievers by
# meta and surface the two weakest.
_ROLE_GROUP = {
    'CL': 'CL', 'SP': 'SP',
    'SU': 'BP', 'MID': 'BP', 'LNG': 'BP', 'MOP': 'BP',
}
_WEAKEST_N_PER_GROUP = {'CL': 1, 'SP': 1, 'BP': 2}


def _weakest_slot_per_role(entries):
    """Return slot labels that are 'weakest' in their role group.

    Slot labels are mapped to a role group via _ROLE_GROUP: CL and SP
    each stand alone; all non-CL bullpen roles (SU/MID/LNG/MOP) pool
    together as 'BP'. Within each group, the N lowest-meta slots (per
    _WEAKEST_N_PER_GROUP) surface a market-buy action; the rest display
    'Optimal' to avoid padding the shopping list.

    Free promotes are surfaced for ALL slots regardless (zero cost, no
    reason to hide them). Only paid market-buy actions get collapsed.
    """
    by_group: dict[str, list] = {}
    for u in entries:
        label = u['pos']
        # Strip trailing digits: 'SP1' → 'SP', 'MID2' → 'MID', 'CL' → 'CL'
        role = label.rstrip('0123456789')
        group = _ROLE_GROUP.get(role, role)
        by_group.setdefault(group, []).append(u)
    weakest = set()
    for group, members in by_group.items():
        n = _WEAKEST_N_PER_GROUP.get(group, 1)
        members_sorted = sorted(members, key=lambda x: x['current_meta'] or 0)
        for u in members_sorted[:n]:
            weakest.add(u['pos'])
    return weakest


def build_chain_rows(positions_list, show_bats=False, show_perf=False,
                     consolidate_by_role=False):
    """Build compact lineup rows for the roster optimizer table.

    Columns: Pos | Current | Meta | Perf | Action | Why
    - Action: concise "what to do" (Optimal / Promote X / Buy X 2,350PP)
    - Why: short AI or meta reason (platoon warning, AI insight)

    consolidate_by_role: when True, only the N weakest slots in each
    role group show a market-buy action (see _ROLE_GROUP /
    _WEAKEST_N_PER_GROUP). CL stands alone; SP stands alone; SU/MID/
    LNG/MOP pool into one 'BP' group ranked by meta. Other slots display
    "Optimal" for their Action. Free promotes are kept on every slot
    regardless. Used for the pitching staff view so users aren't pushed
    to buy a market card for every role when the real decision is
    "upgrade the two weakest relievers overall."
    """
    # Pre-pass: if consolidating, identify the weakest slot in each role.
    _weakest_set: set[str] = set()
    if consolidate_by_role:
        _filtered = []
        for u in upgrade_plan:
            pos_strip = u['pos'].rstrip(" ⚠️")
            if pos_strip in positions_list or any(
                pos_strip.startswith(p) and len(pos_strip) > len(p) and pos_strip[len(p)].isdigit()
                for p in positions_list
            ):
                _filtered.append(u)
        _weakest_set = _weakest_slot_per_role(_filtered)

    rows = []
    for u in upgrade_plan:
        # Match exact position OR numbered slot (SP1→SP, SU2→SU) but NOT
        # substring collisions like CL→C.  Prefix only counts when followed
        # by a digit (e.g. SP1, MID2, C1 for platoons).
        pos = u['pos'].rstrip(" ⚠️")  # strip inline warning emoji if present
        if pos not in positions_list and not any(
            pos.startswith(p) and len(pos) > len(p) and pos[len(p)].isdigit()
            for p in positions_list
        ):
            continue

        # Current player — compact: "Name (H)" — OVR is now a separate column
        # so it can be sorted independently (matches OOTP's roster-sort behavior).
        ovr = u['current_ovr'] or 0
        bh = f" ({u.get('bats', '')})" if show_bats and u.get('bats', '?') != '?' else ""
        current_display = f"{u['current_name']}{bh}"

        # Mark the DH slot as inferred — OOTP doesn't export a DH row so we
        # guess from "extra" starters. The "?" tells the user this slot may
        # not match their actual in-OOTP DH assignment.
        pos_display = u['pos']
        if u.get('dh_inferred'):
            pos_display = f"{u['pos']} ?"

        # Batting order from observed box scores (1..9 slot) — pinned
        # batters only, e.g. Juan Pierre #1, Josh Lowe #4 etc. Empty for
        # pitchers.
        _bo_val = _observed_bo_by_name.get(u.get('current_name') or '', '') \
            if pos in ('C','1B','2B','3B','SS','LF','CF','RF','DH') else ''
        row = {
            "Pos": pos_display,
            "Current": current_display,
            "BO": f"#{_bo_val}" if _bo_val else '',
            "OVR": int(ovr) if ovr else 0,
            "Meta": u['current_meta'],
        }

        # ── Fit column (attribute-mix layer, from card_archetypes) ──
        # Compact format: "72 · Power+AvoidK" — fit_score (0-100) + short
        # archetype name. Tells the user whether the current holder's rating
        # MIX predicts role-fit outcomes, separately from the meta score.
        _cur_fit = u.get('current_fit')
        _cur_arch = u.get('current_archetype') or ''
        if _cur_fit is not None:
            # Shrink archetype name for fit-cell density: drop the "(NN)" rating
            # suffixes and common prefixes so the column stays scannable.
            import re as _re
            _arch_short = _re.sub(r"\s*\(\d+\)", "", _cur_arch)
            _arch_short = _arch_short.replace("Avoid-K", "AvoidK").replace("HR-supp", "HRsupp") \
                .replace("BABIP-supp", "BABIPsupp")
            _arch_short = _arch_short[:22] + ("\u2026" if len(_arch_short) > 22 else "")
            row["Fit"] = f"{_cur_fit:.0f} \u00b7 {_arch_short}" if _arch_short else f"{_cur_fit:.0f}"
        else:
            row["Fit"] = ""

        # ── Confidence chip with inline breakdown ──
        # Cross-team sample-size + consistency score for this card's stats.
        # Pulls from the precomputed _conf_by_name map (built above, once
        # per page load, across all team instances of each roster card).
        # Cell format:
        #   🟢 95 · 680 PA · 2T · σ1.4 · +GL21
        # Components: score · sample · team-instances · consistency-sigma · game-log-bonus
        _conf_entry = _conf_by_name.get(u['current_name'])
        if _conf_entry:
            _score = _conf_entry.get('score', 0)
            _label = _conf_entry.get('label', 'none')
            _agg = _conf_entry.get('aggregate') or {}
            _ins = _agg.get('team_instances') or 1
            if _conf_entry.get('role') == 'pitching':
                _sample = (_agg.get('pitching') or {}).get('ip') or 0
                _sample_label = f"{_sample:.0f} IP"
                _instance_values = (_agg.get('pitching') or {}).get('instance_era_plus') or []
                _gl_sample = ((_agg.get('game_log') or {}).get('pitching') or {}).get('pa_or_bf') or 0
                _gl_saturation = 300
            else:
                _sample = (_agg.get('batting') or {}).get('pa') or 0
                _sample_label = f"{_sample} PA"
                _instance_values = (_agg.get('batting') or {}).get('instance_ops_plus') or []
                _gl_sample = ((_agg.get('game_log') or {}).get('batting') or {}).get('pa_or_bf') or 0
                _gl_saturation = 300
            _icon = {'high': '🟢', 'medium': '🟡', 'low': '🔴', 'none': '⚪'}.get(_label, '⚪')

            # Cross-team std-dev for consistency check
            _sigma_part = ""
            if len(_instance_values) >= 2:
                import statistics as _stat
                try:
                    _sigma = _stat.stdev([float(x) for x in _instance_values])
                    _sigma_part = f" · σ{_sigma:.1f}"
                except Exception:
                    pass

            # Game-log bonus (out of 30 max)
            _gl_part = ""
            if _gl_sample and _gl_sample >= _gl_saturation * 0.3:
                _gl_bonus = min(30, int(30 * _gl_sample / _gl_saturation))
                _gl_part = f" · +GL{_gl_bonus}"

            _instance_part = f" · {_ins}T" if _ins > 1 else ""
            row["Confidence"] = (f"{_icon} {_score} · {_sample_label}"
                                 + _instance_part + _sigma_part + _gl_part)
        else:
            row["Confidence"] = ""

        # ── Regression flag ──
        # From the game-log-derived LD%+EV+BABIP scan. 📈 = positive regression
        # (underperforming BUT quality-of-contact says they should bounce back),
        # 📉 = negative regression (overperforming BUT weak contact under the hood).
        _reg = None
        # Find the card_id for this current roster entry. `u` doesn't have it
        # directly, so we resolve via the active_by_pos map we populated from
        # roster_current earlier (which includes card_id).
        _reg_card_id = None
        for _apos_players in active_by_pos.values():
            for _ap in _apos_players:
                if _ap.get('player_name') == u['current_name']:
                    _reg_card_id = _ap.get('card_id')
                    break
            if _reg_card_id:
                break
        if _reg_card_id and _reg_card_id in _regress_by_card:
            _reg = _regress_by_card[_reg_card_id]
        # Track the regression verdict but don't emit it as its own column —
        # it's folded into the single Status column below alongside the
        # raw Perf stat and the hot/cold Outlook label.
        _reg_badge = ''
        if _reg:
            _arrow = '📈' if _reg['direction'] == 'up' else '📉'
            _reg_badge = _arrow

        # ── Action columns — split into Owned Promotion + Market Upgrade ──
        # Two independent signals:
        #   Owned Promotion: best owned card that beats current starter (FREE
        #                    in-game move), or "Optimal" if none.
        #   Market Upgrade:  best available market card that beats current
        #                    starter by ≥ min_meta_improvement, or "Optimal".
        # Each column evaluated independently so the user sees whether they
        # have an in-house option AND whether the market has something better.
        ai_pick = _get_ai_pick_for_pos(u['pos'])

        # When consolidating by role, suppress market-buy actions for slots
        # that aren't the weakest in their category. Free promotes stay
        # visible since they cost nothing.
        _suppress_market = consolidate_by_role and u['pos'] not in _weakest_set

        # Performance driver analysis — informational only (no lock).
        _pa = u.get('perf_analysis') or None

        # Formatting convention for both upgrade columns:
        #   "📦 +NN · Card Name"           — owned (delta first, left-aligned)
        #   "🛒 +NN · Card Name · NNN PP"  — market (delta on left, cost on right)
        # The +NN delta prefix is what the eye catches first; cost is the
        # right-most piece. "·" is used as a visual separator so the pieces
        # remain scannable without being a dense blob.

        # Context hints that make "Optimal" more informative:
        # • 📈 bounce-back regression candidate — "hold, don't sell"
        # • 📉 fall-off regression candidate    — "consider selling before drop"
        # • ❄️ cold outlook                     — "no upgrade available but underperforming"
        # These get appended to an Optimal label so the user knows WHICH kind of
        # optimal they're looking at. Avoids the trap of "why is a cold Cold
        # player marked Optimal" — now it says "Optimal · ❄️ cold (no upgrade)".
        _optimal_suffix = ""
        if _reg:
            if _reg['direction'] == 'up':
                _optimal_suffix = " · \U0001f4c8 hold (bounce-back)"
            else:
                _optimal_suffix = " · \U0001f4c9 consider sell"
        elif _pa and _pa.get('direction') == 'cold':
            _optimal_suffix = " · \u2744\ufe0f still cold"
        elif _pa and _pa.get('direction') == 'hot':
            _optimal_suffix = " · \U0001f525 riding hot"

        # ── Owned Promotion column ──
        owned_action = ""
        if ai_pick and ai_pick['action'] == 'Promote':
            emoji = ai_pick.get('emoji', '')
            card = short_name(ai_pick['card_name'], 25)
            # AI picks come with their own delta info; fall back to slot delta
            _d = u.get('owned_delta') or 0
            delta_str = f"+{_d} · " if _d else ""
            owned_action = f"{emoji} \U0001f4e6 {delta_str}{card}"
        elif ai_pick and ai_pick['action'] == 'Platoon':
            emoji = ai_pick.get('emoji', '')
            card = short_name(ai_pick['card_name'], 25)
            partner = ai_pick.get('platoon_partner', '')
            owned_action = (f"{emoji} \U0001f91d {card}"
                            + (f" + {short_name(partner, 15)}" if partner else ""))
        elif u['owned_name']:
            _d = u.get('owned_delta') or 0
            # Surface the position annotation (e.g. "as LF") when the
            # owned upgrade is assigned to a non-primary slot, so the user
            # immediately knows why a CF card is listed under LF/RF.
            _first_owned = (u.get('_owned_upgrades') or [None])[0] or {}
            _pos_note = _first_owned.get('position_annotation') or ''
            # Compact the annotation into "[as LF r32]" form for the table.
            _note_badge = ''
            if _pos_note:
                # position_annotation returns e.g. " (played as LF, rating 32)"
                # Convert to "[LF r32]" for space.
                _inner = _pos_note.strip(' ()').replace('played as ', '').replace(', rating ', ' r')
                _note_badge = f" [{_inner}]"
            owned_action = f"\U0001f4e6 +{_d} · {short_name(u['owned_name'], 25)}{_note_badge}"
        else:
            owned_action = "\u2705 Optimal" + _optimal_suffix

        # ── Market Upgrade column ──
        market_action = ""
        # Figure out the right "total" delta to show. market_delta is computed
        # relative to the owned baseline; if we want the delta over CURRENT
        # starter instead (when there's also an owned promote), add them.
        _market_d = u.get('market_delta') or 0
        # For "total meta gain vs the current starter", add owned_delta if
        # there's an owned option being leapfrogged. We use this for the UI
        # so the user sees the full picture, not the nested delta.
        _market_total = _market_d + (u.get('owned_delta') or 0) if u.get('owned_name') else _market_d

        if _suppress_market:
            market_action = "\u2705 Optimal" + _optimal_suffix
        elif ai_pick and ai_pick['action'] == 'Buy':
            emoji = ai_pick.get('emoji', '')
            card = short_name(ai_pick['card_name'], 25)
            p = ai_pick.get('cost') or (u['market_price'] if u.get('market_price') else 0)
            cost = f"{p:,}" if p else "?"
            delta_str = f"+{_market_total} · " if _market_total else ""
            market_action = f"{emoji} \U0001f6d2 {delta_str}{card} · {cost}PP"
        elif u['market_name']:
            # Show if market option is meaningfully better than the best
            # owned option (or better than current when no owned upgrade).
            _market_better = True
            if u.get('owned_name'):
                # Only show market if it beats owned by ≥ the configured
                # min-meta-improvement threshold (from config.yaml's
                # recommendations block, default 10).
                try:
                    _min_delta = int((config.get('recommendations') or {}).get('min_meta_improvement', 10))
                except (ValueError, TypeError, AttributeError):
                    _min_delta = 10
                _market_better = _market_d >= _min_delta
            if _market_better:
                p = u['market_price'] or 0
                cost = f"{p:,}" if p else "?"
                delta_str = f"+{_market_total} · " if _market_total else ""
                market_action = f"\U0001f6d2 {delta_str}{short_name(u['market_name'], 25)} · {cost}PP"
            else:
                market_action = "\u2705 Optimal" + _optimal_suffix
        else:
            market_action = "\u2705 Optimal" + _optimal_suffix

        # Single Status column — compact, packs three signals:
        #   1. Raw rate stat (ERA for pitchers, OPS for batters)
        #   2. Hot/cold outlook emoji + gap (from _analyze_perf_driver)
        #   3. Regression arrow (📈/📉) when applicable
        # Previously these were three separate columns (Perf / Outlook /
        # Regression) which bloated the table. One consolidated column
        # keeps the scanning density high without losing information.
        if show_perf:
            name = u['current_name']
            pb = _perf_bat.get(name)
            pp = _perf_pit.get(name)
            status_parts = []
            # Small-sample disclaimer when sample size hasn't stabilized.
            # For batters: <50 PA, for pitchers: <20 IP.
            _small_sample = False
            if pp:
                status_parts.append(f"{pp['era']:.2f} ERA")
                if (pp.get('ip') or 0) < 20:
                    _small_sample = True
            elif pb:
                status_parts.append(f".{int(pb['ops']*1000):03d} OPS")
                if (pb.get('pa') or 0) < 50:
                    _small_sample = True
            else:
                # No stats at all — explicit placeholder so the cell isn't
                # visually ambiguous (blank could mean "no data" or "stable").
                status_parts.append('\u2014 no stats yet')
            outlook_text = _pa.get('outlook') if (_pa and _pa.get('outlook')) else ''
            if outlook_text:
                status_parts.append(outlook_text)
            if _reg_badge:
                status_parts.append(_reg_badge)
            # Only add the small-sample tag if the outlook text doesn't
            # already mention it (the perf driver uses "⏳" too for small
            # sample hot/cold verdicts).
            if _small_sample and '\u23f3' not in outlook_text:
                status_parts.append('\u23f3 small sample')
            row["Status"] = " · ".join(status_parts)

        # Platoon warnings: tag onto whichever column is non-Optimal so the
        # info is surfaced without a separate "Why" column.
        if u.get('platoon_warning'):
            warn_text = u['platoon_warning']
            if owned_action != "\u2705 Optimal":
                owned_action = f"{owned_action} ⚠"
            if market_action != "\u2705 Optimal":
                market_action = f"{market_action} ⚠"

        row["Owned Promotion"] = owned_action
        row["Market Upgrade"] = market_action

        # ── Hidden fields the tooltip builder uses ──
        # These carry full card titles + deltas so the mini-player-card
        # tooltip can look up ratings/stats without having to reverse-parse
        # the truncated display text. Hidden from the table render itself
        # (not in `columns` list passed to render_tooltip_table).
        row["_current_card_title"] = u.get('current_name') or ''
        row["_owned_target_name"] = u.get('owned_name') or ''
        row["_owned_delta"] = u.get('owned_delta') or 0
        row["_market_target_name"] = u.get('market_name') or ''
        row["_market_delta"] = u.get('market_delta') or 0
        row["_market_price"] = u.get('market_price') or 0
        row["_pos"] = u.get('pos') or ''
        row["_platoon_warning"] = u.get('platoon_warning') or ''

        rows.append(row)
    return rows


def _add_priority_and_sort(rows: list[dict], sort_by_priority: bool = False) -> list[dict]:
    """Assign a fix-order Priority rank to each row based on current Meta.

    Lower meta → higher priority (rank 1 = fix first). The ranking is
    computed within the ``rows`` group, so rotation slots and bullpen
    slots are ranked independently of each other.

    If ``sort_by_priority=True`` (used for the bullpen where slot order
    SU1→MOP is less meaningful than "who's weakest"), the output is
    reordered so Priority #1 is at the top.
    """
    if not rows:
        return rows
    # Rank by meta ascending — weakest slot = Priority 1
    ranked = sorted(
        enumerate(rows), key=lambda x: (x[1].get('Meta') or 999)
    )
    priority_map = {orig_idx: rank + 1 for rank, (orig_idx, _) in enumerate(ranked)}
    out = []
    for idx, row in enumerate(rows):
        new_row = {'Pri': f"#{priority_map[idx]}", **row}
        out.append(new_row)
    if sort_by_priority:
        out.sort(key=lambda r: int(r['Pri'].lstrip('#')))
    return out


CHAIN_COL_CONFIG = {
    "Pri": st.column_config.TextColumn(width="small",
        help=(
            "Fix-order priority within this group (rotation / bullpen / lineup). "
            "#1 = weakest slot = highest priority to upgrade first. Ranked by "
            "current Meta ascending. After you fix the #1 slot, the next-weakest "
            "becomes the new #1 on your next refresh."
        )),
    "Pos": st.column_config.TextColumn(width="small"),
    "Current": st.column_config.TextColumn(width="medium"),
    "BO": st.column_config.TextColumn(width="small",
        help=(
            "RECOMMENDED batting order slot (1–9) based on the player's "
            "ratings — where the engine thinks they should bat, regardless "
            "of your current pinned order.\n\n"
            "Slot logic:\n"
            "  1 Leadoff  → OBP + speed\n"
            "  2 Two-hole → contact + OBP + gap\n"
            "  3          → best overall hitter\n"
            "  4 Cleanup  → power focus\n"
            "  5          → secondary power\n"
            "  6–9        → remaining, by meta descending\n\n"
            "Rows in the batting lineup table are sorted by this BO so you "
            "can read the recommended lineup top-to-bottom. Blank = not "
            "enough rating data to place. Always blank for pitchers."
        )),
    "OVR": st.column_config.NumberColumn(
        format="%d", width="small",
        help=(
            "OOTP's built-in overall rating. Shown as its own column so you "
            "can sort by it (matches OOTP's default roster sort). Remember: "
            "OVR is the black-box quality rating; our Meta number is the "
            "data-driven replacement that usually beats OVR at predicting "
            "WAR. If Meta and OVR disagree, trust Meta."
        ),
    ),
    "Meta": st.column_config.ProgressColumn(
        min_value=300, max_value=800, format="%d", width="small",
        help=(
            "Overall meta — the canonical score used everywhere in the app. "
            "Same number you'll see on Card Detail, Buy Recs, and Sell Recs. "
            "vs-LHP / vs-RHP splits are available on the Card Detail page if "
            "you need them for matchup planning."
        ),
    ),
    "Fit": st.column_config.TextColumn(width="medium",
        help=(
            "Attribute-mix fit for this role (0\u2013100) + the card's archetype. "
            "Built from k-means clusters on z-scored ratings — higher = rating "
            "MIX better matches what actually drives outcomes in the sim.\n\n"
            "Cell format: `NN \u00b7 Archetype`\n\n"
            "**Use it to spot meta/mix divergences.** A high Meta + low Fit "
            "means the card has strong individual ratings but a combination the "
            "sim doesn't reward (e.g. stamina + stuff with weak control). A "
            "low Meta + high Fit is a bargain: the mix wins games even if the "
            "headline number is average.\n\n"
            "Expand **\u2728 Mix Analysis** below the table for per-slot "
            "archetype comparisons + top in-archetype replacements under your "
            "PP cap."
        )),
    "Confidence": st.column_config.TextColumn(width="medium",
        help=(
            "Data quality score (0–100). Composition visible INLINE in the cell text.\n\n"
            "Cell format (batters): 🟢 SCORE · PA · NT · σNN · +GLNN\n"
            "  · SCORE (0–100)  — final confidence; 🟢≥70 🟡40–69 🔴<40.\n"
            "  · PA             — pooled plate appearances across every team "
            "instance of this card in the league.\n"
            "  · NT             — number of team instances pooled (only shown when >1 "
            "— cross-team consistency check).\n"
            "  · σNN            — OPS+ cross-team standard deviation (only when NT>1). "
            "Small σ = card behaves the same for everyone; big σ = manager/strategy "
            "matters, treat the meta as tentative.\n"
            "  · +GLNN          — game-log data bonus (out of 30 max), based on how "
            "many plays we have with exit velocity + batted-ball type.\n\n"
            "Formula: final = 0.6·sample_factor + 0.4·consistency_factor, then "
            "scaled by (1 + game_log_bonus). Caps at 100.\n"
            "Pitchers use IP (saturates at 150) and cross-team ERA+ std-dev instead."
        )),
    "Status": st.column_config.TextColumn(width="medium",
        help=(
            "Per-player status in one cell. Format: "
            "`{rate stat} · {outlook} · {regression}`\n\n"
            "Rate stat — ERA for pitchers, OPS for batters (from the most recent stats snapshot).\n"
            "Outlook — hot/cold verdict vs card meta:\n"
            "  \U0001f525 hot lucky · \U0001f4aa hot real · \u23f3 small sample · "
            "\U0001f9ca cold unlucky · \u2744\ufe0f cold real.\n"
            "Regression — \U0001f4c8 bounce-back (BABIP below but quality contact; hold) · "
            "\U0001f4c9 fall-off (BABIP above but weak contact; sell).\n\n"
            "Open the Performance Outlook expander below the lineup for the full "
            "driver breakdown (BABIP vs league, ERA-FIP delta, etc)."
        )),
    "Owned Promotion": st.column_config.TextColumn(width="large",
        help=(
            "Best owned card you could promote to fill this slot — FREE in-game move.\n\n"
            "Format: \U0001f4e6 +NN · Card Name\n"
            "  · +NN = meta points gained over the current starter.\n"
            "  · Card Name = your best internal option.\n"
            "\U0001f91d = platoon pairing (two owned cards share the slot).\n"
            "\u2705 Optimal = no owned card beats the current starter at this position."
        )),
    "Market Upgrade": st.column_config.TextColumn(width="large",
        help=(
            "Best market card that would upgrade this slot beyond your owned options "
            "by at least the configured meta-improvement threshold.\n\n"
            "Format: \U0001f6d2 +NN · Card Name · NNN PP\n"
            "  · +NN (left) = total meta gained over the current starter (includes "
            "any owned promotion that the market card leapfrogs).\n"
            "  · Card Name = the buy target.\n"
            "  · NNN PP (right) = estimated cost from recent sales.\n"
            "\u2705 Optimal = no market card meaningfully beats what you already have "
            "(either current starter or best owned promote).\n\n"
            "⚠ appended = card has a platoon warning (see Performance Outlook expander)."
        )),
}

# Legend removed — the emoji meanings live in the column tooltips
# (Status / Owned Promotion / Market Upgrade) where they're one hover away
# when a user needs them, instead of eating vertical space on every visit.

# ── Performance Outlook expander ──
# Full driver breakdown for over/underperformers — shown below the lineup
# tables so users can see exactly *why* a slot's current player is riding
# hot or cold. This is the "calculator" view: card meta vs perf meta, the
# signed gap, luck-vs-skill verdict, and the specific rate-stat evidence
# (BABIP vs league, K%, ISO, ERA-FIP, etc.) that drove the verdict.
def _render_perf_outlook_expander(plan_entries, title, empty_hint):
    """Render the Performance Outlook expander for a subset of slots.

    `plan_entries` is the filtered list of upgrade_plan entries for the
    current lineup view (batting or pitching). Only slots with a non-empty
    perf_analysis whose `direction` is hot or cold are included.
    """
    rows = []
    for u in plan_entries:
        pa = u.get('perf_analysis')
        if not pa:
            continue
        if pa.get('direction') not in ('hot', 'cold'):
            continue
        drivers = pa.get('drivers') or []
        rows.append({
            "Pos": u['pos'],
            "Player": u['current_name'],
            "Sample": pa.get('sample_label', ''),
            "Card Meta": int(u['current_meta'] or 0),
            "Playing Like": int(pa['perf_meta']),
            "Gap": int(pa['gap']),
            "Verdict": pa['outlook'],
            "Drivers": " \u2022 ".join(drivers) if drivers
                       else "No standout rate outlier \u2014 balanced hot/cold streak",
            "Regression?": "\u2705 likely" if pa.get('regression') else "\u2014",
        })

    with st.expander(f"\U0001f4ca {title} \u2014 {len(rows)} slot(s) off-card"):
        if not rows:
            st.caption(empty_hint)
            return
        # Sort by absolute gap descending — biggest divergence first
        rows.sort(key=lambda r: -abs(r["Gap"]))
        # Describe what's in the peer pool so the user knows whether the
        # percentile phrasing is real or the thin "vs your own team"
        # fallback. When the lb124 league-wide stats file is imported,
        # the phrasing auto-upgrades to "top X% of lb124 pitchers".
        _peer_note = ""
        if _peer_era_plus or _peer_ops_plus:
            if 'team' in _peer_pool_label_pit or 'team' in _peer_pool_label_bat:
                _peer_note = (
                    " League-relative ERA+/OPS+ in the drivers column is shown "
                    "as \"N% better/worse than league avg\" \u2014 import a "
                    "league-wide stats file to upgrade to true league percentiles."
                )
            else:
                _peer_note = (
                    f" ERA+/OPS+ percentiles are ranked against "
                    f"**{len(_peer_era_plus)}** {_peer_pool_label_pit} and "
                    f"**{len(_peer_ops_plus)}** {_peer_pool_label_bat}."
                )
        st.caption(
            "**How to read this:** *Card Meta* is what the meta formula thinks "
            "the player is worth. *Playing Like* blends WAR-rate (WAR/600 for "
            "batters, WAR/200 for pitchers) with league-relative OPS+/ERA+ \u2014 "
            "60/40 favoring the rate stat on normal samples, 75/25 favoring "
            "the rate stat when the sample is very small. This fixes the "
            "old WAR-only calc that mislabeled low-leverage relievers with "
            "elite ERAs as \"cold\". *Gap* = Playing Like \u2212 Card Meta; "
            "anything inside \u00b150 is ignored. *Drivers* lists the rate-stat "
            "outliers vs league average \u2014 the top line is the league-relative "
            "headline (ERA+/OPS+), followed by the BABIP/FIP/K%/BB% votes that "
            "drove the luck-vs-skill verdict." + _peer_note
        )
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pos": st.column_config.TextColumn(width="small"),
                "Player": st.column_config.TextColumn(width="medium"),
                "Sample": st.column_config.TextColumn(width="small",
                    help="Plate appearances (batters) or innings pitched (pitchers) "
                         "in the current stats snapshot. Smaller samples = less "
                         "trustworthy conclusions."),
                "Card Meta": st.column_config.NumberColumn(format="%d", width="small",
                    help="The meta score on the card \u2014 what the formula thinks "
                         "the player is worth."),
                "Playing Like": st.column_config.NumberColumn(format="%d", width="small",
                    help="Perf-derived meta equivalent. Blends WAR-rate "
                         "(WAR/600 or WAR/200) with league-relative OPS+/ERA+ "
                         "so low-leverage relievers with elite ERAs aren't "
                         "mislabeled as cold. Lean is 60/40 rate-over-WAR on "
                         "normal samples, 75/25 when the sample is very small."),
                "Gap": st.column_config.NumberColumn(format="%+d", width="small",
                    help="Playing Like \u2212 Card Meta. Positive = hot streak, "
                         "negative = slump. Anything inside \u00b150 is ignored."),
                "Verdict": st.column_config.TextColumn(width="medium",
                    help="Luck-vs-skill verdict based on rate-stat outliers. "
                         "\U0001f525 = hot, \U0001f9ca = cold, \u23f3 = small sample."),
                "Drivers": st.column_config.TextColumn(width="large",
                    help="Specific rate-stat outliers that drove the verdict."),
                "Regression?": st.column_config.TextColumn(width="small",
                    help="\"likely\" flags hot-streak + luck / small-sample cases "
                         "where you should expect performance to fall toward card meta."),
            },
        )


# ── Roster Move Suggestions expander ──
# The "promote" side of the upgrade plan tells you who to bring UP, but
# never tells you who comes OFF. That's fine when the displaced starter is
# a cut candidate anyway, but when the default drop has rebound upside
# (cold-but-unlucky or small-sample noise) we should suggest benching them
# and dropping a less valuable piece instead. This section surfaces those
# pairs so the full 26-man move is visible.
def _render_mix_analysis_expander(plan_entries, title, side):
    """Per-slot archetype / fit replacement view.

    Layered on top of the main Owned/Market Upgrade columns. Where those
    columns gate strictly on meta_delta, the Mix Analysis expander widens
    the net to include market cards whose attribute MIX (fit_score,
    archetype_war) says they'll outperform the current holder — even when
    meta delta is small. Respects the sidebar **Max PP per card** filter.

    ``side`` is 'batting' or 'pitching' — determines whether we search
    the archetype role set {'batting'} vs {'sp', 'rp'}.
    """
    if not _archetypes_by_card:
        return  # Table not yet built; silently skip.

    role_filter_sp = side == 'pitching'
    bullets = []
    for u in plan_entries:
        cur_fit = u.get('current_fit')
        cur_arch = u.get('current_archetype') or ''
        cur_war = u.get('current_archetype_war')
        cur_id = u.get('current_card_id')
        if cur_fit is None or not cur_id:
            continue

        # Pick archetype role based on position
        pos = (u.get('pos') or '').rstrip(" \u26a0\ufe0f")
        if role_filter_sp:
            # SP vs RP/CL subset. card_archetypes.role uses uppercase 'SP'/'RP'
            # (verified 2026-04-20 — lowercase returns zero rows and silently
            # renders a "0 mix upgrades" expander that's actually a bug).
            if pos.startswith('SP'):
                role_values = ('SP',)
            else:
                role_values = ('RP',)
        else:
            role_values = ('batting',)

        price_cap = max_spend if max_spend > 0 else 999_999_999
        # Fetch top-5 replacements by fit_score, same role, within budget.
        # We look at unowned + owned-but-not-active cards as replacement candidates.
        placeholders = ','.join('?' * len(role_values))
        q = f"""
            SELECT ca.card_id, c.card_title, ca.archetype_name, ca.fit_score,
                   ca.archetype_war, c.last_10_price,
                   CASE WHEN {'pitching' == side} THEN c.meta_score_pitching
                        ELSE c.meta_score_batting END AS meta,
                   c.owned, c.tier_name
            FROM card_archetypes ca
            JOIN cards c ON c.card_id = ca.card_id
            WHERE ca.role IN ({placeholders})
              AND ca.card_id != ?
              AND ca.fit_score > ?
              AND (c.last_10_price IS NULL OR c.last_10_price <= ?)
            ORDER BY ca.fit_score DESC LIMIT 5
        """
        try:
            rows = conn.execute(
                q, (*role_values, cur_id, cur_fit, price_cap),
            ).fetchall()
        except Exception:
            rows = []

        if not rows:
            continue

        # Build a compact bullet per slot
        war_str = f" (archetype avg WAR {cur_war:.2f})" if cur_war else ""
        cand_lines = []
        for r in rows:
            d = dict(r)
            price = d.get('last_10_price')
            owned_tag = " \U0001f4e6" if d.get('owned') else ""
            price_tag_ = f"{int(price):,} PP" if price else "free" if d.get('owned') else "?"
            fit_delta = d['fit_score'] - cur_fit
            war_delta = ""
            if d.get('archetype_war') and cur_war:
                war_delta = f", \u0394WAR {(d['archetype_war'] - cur_war):+.2f}"
            cand_lines.append(
                f"  \u2022 **{d['card_title']}**{owned_tag} \u2014 "
                f"fit {d['fit_score']:.0f} (\u0394{fit_delta:+.0f}), "
                f"meta {int(d.get('meta') or 0)}, "
                f"*{d.get('archetype_name', '')}*{war_delta} \u00b7 {price_tag_}"
            )
        bullets.append(
            f"**{pos} \u2014 {u.get('current_name', '')}** \u00b7 "
            f"fit {cur_fit:.0f} \u00b7 *{cur_arch}*{war_str}\n" + "\n".join(cand_lines)
        )

    with st.expander(f"\u2728 {title} \u2014 {len(bullets)} slot(s) with mix upgrades"):
        if not bullets:
            st.caption(
                "No in-archetype replacements beat the current holder's fit score "
                "under the sidebar PP cap. Raise **Max PP per card** to widen the "
                "search. Mix Analysis is additive to the Owned/Market columns \u2014 "
                "if both show `\u2705 Optimal` and this is also empty, the slot is "
                "genuinely locked for now."
            )
            return
        st.caption(
            "For each slot, the top 5 cards in the same archetype role with **higher fit** "
            "than the current holder, under your **Max PP per card** sidebar cap. \u0394fit "
            "is points above the current starter (0\u2013100 scale); \u0394WAR is the expected "
            "WAR delta if the archetype average holds."
        )
        for b in bullets:
            st.markdown(b)


def _render_roster_moves_expander(plan_entries, title, empty_hint):
    """Render the Roster Move Suggestions expander for a subset of slots.

    For each slot with a promote recommendation where the current player
    has bounce-back signals, show the suggested alternate drop.
    """
    rows = []
    for u in plan_entries:
        # Only slots where we actually have a promote lined up. Market-only
        # buys don't need a drop suggestion — the user is buying a card
        # from the market, not moving someone up from bench.
        if not u.get('owned_name'):
            continue
        suggestion = _find_drop_candidate(u, _active_pool)
        if not suggestion:
            continue
        depth = suggestion.get('alt_depth', 'thin')
        depth_count = suggestion.get('alt_depth_count', 0)
        if depth == 'surplus':
            depth_tag = f" \u2014 surplus ({depth_count} at pos)"
        else:
            depth_tag = f" \u2014 thin ({depth_count} at pos)"
        rows.append({
            "Slot": u['pos'],
            "Promote": short_name(u['owned_name'], 25),
            "Keep (don't cut)": (
                f"{suggestion['keep_name']} (meta {suggestion['keep_meta']})"
            ),
            "Why keep": suggestion['keep_reason'],
            "Drop instead": (
                f"{suggestion['alt_name']} \u2014 {suggestion['alt_pos']} "
                f"(OVR {suggestion['alt_ovr']}, meta {suggestion['alt_meta']}, "
                f"{suggestion['alt_role']}){depth_tag}"
            ),
        })

    with st.expander(f"\U0001f504 {title} \u2014 {len(rows)} suggestion(s)"):
        if not rows:
            st.caption(empty_hint)
            return
        st.caption(
            "**How to read this:** When the default drop for a promote is a "
            "player with rebound upside (cold but unlucky, or small-sample "
            "noise), you'd rather bench them than cut them. The *Drop instead* "
            "column points at the lowest-meta eligible piece on your active "
            "26-man. We filter out (a) players who are themselves rebound "
            "cases, (b) the only player at a position (dropping your only "
            "backup C is worse than the upgrade), and (c) players on the "
            "opposite side of the roster (batting upgrades only drop batters). "
            "**Surplus (3+ at pos)** picks are safer than **thin (2 at pos)** "
            "picks \u2014 the latter means your #2 at that spot becomes "
            "the only backup."
        )
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Slot": st.column_config.TextColumn(width="small"),
                "Promote": st.column_config.TextColumn(width="medium",
                    help="The owned card the optimizer is recommending you "
                         "bring up to this slot."),
                "Keep (don't cut)": st.column_config.TextColumn(width="medium",
                    help="The current starter at this slot. Normally they'd "
                         "be displaced \u2014 but the perf analyzer says they "
                         "have rebound upside, so don't cut them."),
                "Why keep": st.column_config.TextColumn(width="medium",
                    help="The perf-analyzer verdict that flagged rebound upside."),
                "Drop instead": st.column_config.TextColumn(width="large",
                    help="Lowest-meta active 26-man piece \u2014 drop them "
                         "from the 26-man to make room for the promote, "
                         "and move the kept player to the bench."),
            },
        )


# ── TABBED LAYOUT — batting + pitching only; AI picks are threaded inline ──
tab_bat, tab_pit = st.tabs(["⚾ Batting Lineup", "🎯 Pitching Staff"])

with tab_bat:
    # Confidence chip reads the latest `meta_calibration` row for batting and
    # tells the user how well the meta formula tracks observed performance.
    # Without this, every meta-sorted view is implicitly claiming its ordering
    # is useful — the chip turns that implicit claim into a visible input.
    st.markdown(
        f'<div style="margin-bottom: 6px;">Meta ordering reliability: '
        f'{_meta_confidence_chip("batting")}</div>',
        unsafe_allow_html=True,
    )
    bat_rows = build_chain_rows(bat_positions, show_bats=True, show_perf=True)
    # Batting lineup keeps its natural position order (C, 1B, 2B, ...) but
    # gets a Priority rank column so the weakest slot is easy to spot.
    bat_rows = _add_priority_and_sort(bat_rows, sort_by_priority=False)
    # Sort the batting-lineup table by recommended batting order (1→9),
    # with any player lacking a BO (e.g. no ratings fetched) sinking to
    # the bottom. The Pri column still shows fix-priority rank; the row
    # order reflects "how the lineup should be arranged top-to-bottom".
    def _bo_sort_key(r):
        bo = r.get('BO') or ''
        try:
            return int(bo.lstrip('#')) if bo else 99
        except (ValueError, TypeError):
            return 99
    bat_rows.sort(key=_bo_sort_key)
    if bat_rows:
        h = min(35 * len(bat_rows) + 40, 600)
        # Single unified view — HTML table with per-cell tooltips that include
        # LLM verdict annotations (populated asynchronously by the background
        # worker). No pre/post-AI fork: there is ONE Owned Promotion and ONE
        # Market Upgrade per slot; LLM verification is embedded metadata on
        # that single pick, surfaced inline as badges + on hover.
        from app.utils.tooltip_html import render_tooltip_table
        from app.pages._chain_tooltips import (
            chain_header_help, chain_cell_tooltip,
        )
        _chain_cols = [c for c in ['Pri','Pos','Current','BO','OVR','Meta','Fit','Confidence',
                                    'Status','Owned Promotion','Market Upgrade']
                       if c in bat_rows[0]]
        render_tooltip_table(
            rows=bat_rows,
            columns=_chain_cols,
            header_help=chain_header_help(CHAIN_COL_CONFIG),
            cell_tooltip=chain_cell_tooltip(conn),
            max_height_px=h,
        )
        # Flag inferred DH so the user knows it's a guess, not OOTP data.
        # This only fires when OOTP didn't export a POS=DH row (i.e. you
        # haven't manually set anyone's team position to DH in-game).
        if any(u.get('dh_inferred') for u in upgrade_plan):
            st.caption(
                "\u2139\ufe0f **DH ?** — OOTP hasn't exported a DH row, so the DH "
                "slot is inferred from your highest-meta \"extra\" starter (an "
                "active-roster batter who isn't the top player at their defensive "
                "position). **The fix:** in OOTP, open your team roster, change "
                "your DH's *Team Position* to DH, and re-export — the roster CSVs "
                "will then include a POS=DH row and the optimizer will use that "
                "identity automatically. Temporary workaround: pin a DH via "
                "**Sidebar \u2192 Lineup overrides \u2192 DH (manual)**."
            )

        # Performance Outlook — full driver breakdown for hot/cold slots
        _bat_plan = [u for u in upgrade_plan if u['pos'] in bat_positions]
        _render_perf_outlook_expander(
            _bat_plan,
            "Performance Outlook (batting)",
            "No batters are meaningfully off their card meta right now \u2014 "
            "everyone's performance is within \u00b150 meta of their card grade.",
        )
        _render_mix_analysis_expander(
            _bat_plan, "Mix Analysis (batting)", side='batting',
        )
        _render_roster_moves_expander(
            _bat_plan,
            "Roster Move Suggestions (batting)",
            "No batting promotes currently displace a player with rebound "
            "upside \u2014 the default drops are fine.",
        )

    # Batting order (1–9) is folded INTO the chain table via a "BO" column,
    # computed upstream from ratings before the chain table builds. See the
    # `_observed_bo_by_name` population done earlier (before build_chain_rows).
    #
    # `_all_lineup_names` is still needed below for Bench Bats selection:
    # it's the set of players currently occupying a field or DH slot, and
    # anyone else in active_by_pos is bench.
    _shown_field = {e['current_name'] for e in upgrade_plan if e['pos'] in bat_field_positions}
    _shown_dh = [e for e in upgrade_plan if e['pos'] == 'DH']
    _dh_name = _shown_dh[0]['current_name'] if _shown_dh else None
    _all_lineup_names = list(_shown_field)
    if _dh_name and _dh_name != '(empty)':
        _all_lineup_names.append(_dh_name)

    # ── Bench Bats ──
    st.divider()
    st.markdown("##### Bench Bats")
    st.caption("Your 4 reserve batters — pinch hitters, platoon partners, defensive subs")

    # Bench bats = starters NOT in the 9-man lineup.
    # Dedupe — the same player can appear in active_by_pos under multiple
    # positions if roster CSV paths insert multiple rows (seen with
    # Aranda at 1B twice, Bellinger at LF twice due to starter/bench/
    # reserve role churn). Key by card_id when available, else player_name.
    _bench_bats = []
    _bench_seen: set = set()
    for fpos in bat_field_positions:
        for p in active_by_pos.get(fpos, []):
            if p['player_name'] in _all_lineup_names:
                continue
            key = p.get('card_id') or p.get('player_name')
            if key in _bench_seen:
                continue
            _bench_seen.add(key)
            _bench_bats.append(p)
    _bench_bats.sort(key=lambda p: p['meta_score'] or 0, reverse=True)
    _bench_bats = _bench_bats[:4]  # 26-man roster has ~4 bench bats

    # All rostered player names — no rostered player should be recommended as an upgrade
    _all_rostered_names = set()
    for _pos_players in all_by_pos.values():
        for _rp in _pos_players:
            _all_rostered_names.add(_rp['player_name'])
    _used_bench_upgrades = set()  # track already-recommended upgrades to avoid dupes

    # Baseball roster rules: a 4-man bench usually has 1 backup C + 3 flex bats.
    # Figure out which positions the bench ALREADY covers so we don't recommend a
    # duplicate catcher (or other single-slot role) into a flex slot.
    _bench_pos_counts = {}
    for _bp in _bench_bats:
        _pkey = _bp['position'] or '?'
        _bench_pos_counts[_pkey] = _bench_pos_counts.get(_pkey, 0) + 1
    # Roles where one bench slot is enough — duplicating = waste of roster space
    _single_slot_bench_roles = {'C'}
    # Positions already saturated on the bench (can't recommend more of these)
    _saturated_bench_positions = {
        _pos for _pos, _cnt in _bench_pos_counts.items()
        if _pos in _single_slot_bench_roles and _cnt >= 1
    }

    if _bench_bats:
        bench_rows = []
        for bp in _bench_bats:
            pname = bp['player_name']
            bpos = bp['position'] or '?'
            bmeta = round(bp['meta_score'] or 0)
            bh = bp.get('bats_hand', '?')
            bperf = _perf_bat.get(pname)

            perf_str = ""
            if bperf:
                perf_str = f".{int(bperf['ops']*1000):03d} OPS  {bperf['war600']:.1f}W"

            # Find best upgrade from collection (not already rostered).
            # Position-match: a bench SS upgrade should play SS (or adjacent IF), not
            # a catcher. Plus baseball rules: if the bench already has a backup C,
            # don't recommend a second C for any other slot (wasted roster space).
            _pos_match_map = {
                'C':  ('C',),
                '1B': ('1B', '3B', 'LF', 'RF'),
                '2B': ('2B', 'SS', '3B'),
                'SS': ('SS', '2B', '3B'),
                '3B': ('3B', '1B', 'SS', '2B'),
                'LF': ('LF', 'RF', 'CF', '1B'),
                'CF': ('CF', 'LF', 'RF'),
                'RF': ('RF', 'LF', 'CF', '1B'),
                'DH': ('DH', '1B', 'LF', 'RF', '3B'),
            }
            _slot_compat = _pos_match_map.get(bpos, (bpos,))
            # Remove positions already saturated elsewhere on the bench.
            # Exception: if THIS slot IS the saturated position (e.g. the actual backup C
            # slot), keep it — we still want a better C here.
            _acceptable_positions = tuple(
                p for p in _slot_compat
                if p not in _saturated_bench_positions or p == bpos
            )
            if not _acceptable_positions:
                _acceptable_positions = (bpos,)

            # Build an eligibility-aware WHERE: a card qualifies if its
            # primary position is in the compat list OR its pos_rating at
            # the bench slot meets the eligibility floor. This catches
            # multi-position cards whose primary is outside the compat set
            # but whose defensive ratings are fine for this slot.
            _or_clauses = []
            _or_params: list = []
            for _p in _acceptable_positions:
                frag, params = build_eligible_where_clause(_p, table_alias='c')
                _or_clauses.append(frag)
                _or_params.extend(params)
            _where_or = " OR ".join(_or_clauses)

            _rating_sql = select_rating_columns('c')
            _bench_candidates = conn.execute(f"""
                SELECT c.card_id, c.card_title, c.team,
                       c.meta_score_batting as raw_meta, c.position_name, c.card_value,
                       {_rating_sql}
                FROM cards c
                WHERE c.owned = 1 AND c.meta_score_batting > ?
                    AND c.pitcher_role IS NULL
                    AND ({_where_or})
                ORDER BY c.meta_score_batting DESC LIMIT 40
            """, (bmeta + 10, *_or_params)).fetchall()

            # Collect up to 3 qualifying candidates (with penalty applied
            # for non-primary assignments). Previously this loop returned
            # the first match only — so a bench CF with a better LF card
            # immediately behind it in meta-rank was invisible.
            picked: list[dict] = []
            for _cand in _bench_candidates:
                _cand_title = _cand['card_title'] or ''
                if any(rname in _cand_title for rname in _all_rostered_names):
                    continue  # Player already on active roster
                if _cand_title in _used_bench_upgrades:
                    continue  # Already recommended for another bench slot
                _cand_d = dict(_cand)
                _penalty = position_meta_penalty(_cand_d, bpos)
                _eff_meta = (_cand_d.get('raw_meta') or 0) - _penalty
                if _eff_meta <= bmeta + 10:
                    continue  # Penalty knocks it back under the threshold
                _cand_d['meta'] = _eff_meta
                _cand_d['_penalty'] = _penalty
                picked.append(_cand_d)
                if len(picked) >= 3:
                    break

            # Claim only the TOP pick globally so a given card doesn't
            # double-book into two bench slots; leave alternates free in
            # case another slot prefers them.
            if picked:
                _used_bench_upgrades.add(picked[0]['card_title'] or '')

            def _fmt_bench_upgrade(cand: dict) -> str:
                _full_title = cand.get('card_title') or ''
                up_meta = round(cand.get('meta') or 0)
                delta = up_meta - bmeta
                up_pos = cand.get('position_name') or '?'
                up_value = cand.get('card_value') or 0
                # Strip "MLB YYYY Live POS" prefix for readable compact display
                _clean = _full_title
                for _prefix in ['MLB 2026 Live ', 'MLB 2025 Live ', 'MLB 2024 Live ']:
                    if _clean.startswith(_prefix):
                        _clean = _clean[len(_prefix):]
                        _parts = _clean.split(' ', 1)
                        if _parts and _parts[0] in ('C','1B','2B','3B','SS','LF','CF','RF','DH','SP','RP','CP'):
                            _clean = _parts[1] if len(_parts) > 1 else _clean
                        break
                _clean = short_name(_clean, 28)
                # Note when the card plays out of its primary position —
                # the user sees why a CF-primary is appearing under the LF
                # bench slot (rating hint in parens).
                pen_hint = ""
                if up_pos and bpos and up_pos != bpos:
                    rating = cand.get(POS_RATING_COL.get(bpos, ''))
                    if rating is not None:
                        try:
                            pen_hint = f" [as {bpos} r{int(float(rating))}]"
                        except (TypeError, ValueError):
                            pen_hint = f" [as {bpos}]"
                    else:
                        pen_hint = f" [as {bpos}]"
                return f"\U0001f4e6 {_clean}{pen_hint} • {up_pos} • {up_meta}m (+{delta}) • {up_value}pp"

            if picked:
                # Top pick on the main line; additional picks rendered as
                # "also: …" so the user sees the full bench-upgrade stack
                # without drilling into the Alternative Options expander.
                action = _fmt_bench_upgrade(picked[0])
                if len(picked) > 1:
                    _alts = " ; ".join(_fmt_bench_upgrade(p) for p in picked[1:])
                    action = f"{action}\n\u2937 also: {_alts}"
            else:
                action = ""

            bench_rows.append({
                "Pos": bpos,
                "Player": f"{pname} ({bp.get('ovr', '?')} {bh})",
                "Meta": bmeta,
                "Perf": perf_str,
                "Upgrade": action if action else "\u2705 Best available",
            })

        st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True,
                     column_config={
                         "Pos": st.column_config.TextColumn(width="small"),
                         "Player": st.column_config.TextColumn(width="medium"),
                         "Meta": st.column_config.ProgressColumn(
                             min_value=300, max_value=800, format="%d", width="small",
                             help="Overall meta — same number used everywhere in the app. "
                                  "vs-LHP / vs-RHP splits are on Card Detail."),
                         "Perf": st.column_config.TextColumn(width="small"),
                         "Upgrade": st.column_config.TextColumn(width="large"),
                     })

        # ── Collection pool: owned batters NOT currently rostered ──
        # This answers "do I actually have that player?" for anyone shown in Upgrade column.
        with st.expander("\U0001f4e6 Owned batters on the bench pool (not on active roster)", expanded=False):
            st.caption("These are batters you own but haven't assigned to your 26-man active roster. "
                       "Any of them can be promoted — this is where bench upgrades come from.")
            _pool_rows = conn.execute("""
                SELECT c.card_title, c.team, c.position_name, c.bats, c.card_value,
                       c.meta_score_batting as meta
                FROM cards c
                WHERE c.owned = 1 AND c.pitcher_role IS NULL
                  AND c.meta_score_batting IS NOT NULL
                ORDER BY c.meta_score_batting DESC
                LIMIT 60
            """).fetchall()
            _bats_map = {1: 'R', 2: 'L', 3: 'S'}
            _pool_display = []
            for _p in _pool_rows:
                _title = _p['card_title'] or ''
                # Skip if actively rostered
                if any(rname in _title for rname in _all_rostered_names):
                    continue
                # Clean display name
                _clean = _title
                for _prefix in ['MLB 2026 Live ', 'MLB 2025 Live ', 'MLB 2024 Live ']:
                    if _clean.startswith(_prefix):
                        _clean = _clean[len(_prefix):]
                        _parts = _clean.split(' ', 1)
                        if _parts and _parts[0] in ('C','1B','2B','3B','SS','LF','CF','RF','DH'):
                            _clean = _parts[1] if len(_parts) > 1 else _clean
                        break
                _pool_display.append({
                    "Player": _clean,
                    "Pos": _p['position_name'] or '?',
                    "Team": _p['team'] or '',
                    "B": _bats_map.get(_p['bats'], '?'),
                    "Meta": round(_p['meta'] or 0),
                    "Value": _p['card_value'] or 0,
                })
            if _pool_display:
                st.dataframe(pd.DataFrame(_pool_display[:25]), use_container_width=True, hide_index=True,
                             column_config={
                                 "Player": st.column_config.TextColumn(width="medium"),
                                 "Pos": st.column_config.TextColumn(width="small"),
                                 "Team": st.column_config.TextColumn(width="small"),
                                 "B": st.column_config.TextColumn(width="small"),
                                 "Meta": st.column_config.ProgressColumn(
                                     min_value=300, max_value=800, format="%d", width="small",
                                     help="Overall meta — same number used everywhere in the app. "
                                          "vs-LHP / vs-RHP splits are on Card Detail."),
                                 "Value": st.column_config.NumberColumn(format="%d pp", width="small"),
                             })
                st.caption(f"Showing top {min(25, len(_pool_display))} of {len(_pool_display)} owned non-rostered batters by meta.")
            else:
                st.info("No owned batters are sitting in the pool — everyone is assigned.")

        # Bench composition analysis
        bench_hands = [bp.get('bats_hand', '?') for bp in _bench_bats]
        l_count = bench_hands.count('L')
        r_count = bench_hands.count('R')
        s_count = bench_hands.count('S')
        bench_positions = set(bp['position'] for bp in _bench_bats)
        if l_count == 0:
            st.warning("No left-handed bench bat. Consider adding one for pinch-hitting vs RHP.")
        elif r_count == 0:
            st.warning("No right-handed bench bat. Consider adding one for pinch-hitting vs LHP.")
        if 'C' not in bench_positions:
            # Check if there's a backup catcher at all
            backup_c = [p for p in all_by_pos.get('C', []) if p['player_name'] not in _all_lineup_names]
            if not backup_c:
                st.warning("No backup catcher on the bench.")
    else:
        st.info("No bench bats identified. Check roster data.")

with tab_pit:
    # Confidence chips for pitching meta — rendered SP and RP separately.
    # Combined pitching typically masks the split: SPs calibrate similarly
    # to batting (ratings → WAR is fairly predictable at 50+ IP), while RPs
    # are noise-dominated at any realistic sample size (leverage/usage
    # variance swamps talent signal). Showing one combined chip hid the
    # fact that SP meta is trustworthy while RP meta often isn't.
    #
    # Reads pos:SP / pos:RP (direct meta→WAR Pearson) rather than the
    # 'pitching_sp' / 'pitching_rp' Ridge-fit rows. The chip is labeled
    # "Meta ordering reliability" — so we should show how well the
    # CURRENT meta formula (ratings + overlays + bonuses + card_type
    # offsets) predicts WAR, not how well a weights-only Ridge model
    # predicts from ratings. The overlays are meta's secret sauce and
    # the Ridge-fit doesn't see them.
    st.markdown(
        f'<div style="margin-bottom: 6px;">Meta ordering reliability: '
        f'{_meta_confidence_chip("pos:SP", "SP")}'
        f'{_meta_confidence_chip("pos:RP", "RP")}</div>',
        unsafe_allow_html=True,
    )
    # One-line caption covering both "weakest-slot only" market-buy logic
    # and the bullpen-weakest-first sort. (Previously two captions said
    # overlapping things.)
    st.caption(
        "SP1\u2192SP5 in depth-chart order · bullpen sorted by priority "
        "(weakest first). Market-buys surface on the weakest SP, weakest "
        "CL, and the 2 weakest non-CL relievers (SU/MID/LNG/MOP pooled); "
        "other slots read \"\u2705 Optimal\"."
    )

    sp_rows = build_chain_rows(['SP'], show_perf=True, consolidate_by_role=True)
    sp_rows = _add_priority_and_sort(sp_rows, sort_by_priority=False)

    pen_rows = build_chain_rows(['CL', 'SU', 'MID', 'LNG', 'MOP'], show_perf=True, consolidate_by_role=True)
    pen_rows = _add_priority_and_sort(pen_rows, sort_by_priority=True)

    unified_pitching = sp_rows + pen_rows
    if unified_pitching:
        # Use the same rich-tooltip HTML table as the batting tab so pitchers
        # also get mini-player-card tooltips on Current / Owned Promotion /
        # Market Upgrade cells.
        from app.utils.tooltip_html import render_tooltip_table
        from app.pages._chain_tooltips import chain_header_help, chain_cell_tooltip
        _pit_cols = [c for c in ['Pri','Pos','Current','BO','OVR','Meta','Fit','Confidence',
                                  'Status','Owned Promotion','Market Upgrade']
                     if c in unified_pitching[0]]
        render_tooltip_table(
            rows=unified_pitching,
            columns=_pit_cols,
            header_help=chain_header_help(CHAIN_COL_CONFIG),
            cell_tooltip=chain_cell_tooltip(conn),
            max_height_px=min(35 * len(unified_pitching) + 40, 550),
        )

    # Performance Outlook — full driver breakdown for hot/cold pitching slots
    _pit_labels = set()
    for u in upgrade_plan:
        _clean = (u.get('pos') or '').rstrip(" \u26a0\ufe0f")
        if any(_clean.startswith(p) and (len(_clean) == len(p) or _clean[len(p)].isdigit())
               for p in pitch_positions):
            _pit_labels.add(u['pos'])
    _pit_plan = [u for u in upgrade_plan if u['pos'] in _pit_labels]
    _render_perf_outlook_expander(
        _pit_plan,
        "Performance Outlook (pitching)",
        "No pitchers are meaningfully off their card meta right now \u2014 "
        "all arms are within \u00b150 meta of their card grade.",
    )
    _render_mix_analysis_expander(
        _pit_plan, "Mix Analysis (pitching)", side='pitching',
    )
    _render_roster_moves_expander(
        _pit_plan,
        "Roster Move Suggestions (pitching)",
        "No pitching promotes currently displace an arm with rebound "
        "upside \u2014 the default drops are fine.",
    )

# AI OPTIMIZED ROSTER section REMOVED. There is no pre/post-AI fork.
# The chain tables above already show ONE Owned Promotion + ONE Market
# Upgrade per slot (the meta engine's canonical pick). LLM verdicts on
# those picks arrive asynchronously via the background worker and render
# inline in the cell tooltips / as verdict badges — not as a parallel
# Gemini-picks table.

# ════════════════════════════════════════════════════════════════
# RECOMMENDATION OUTCOMES — did past recs pan out?
# Pulls every logged rec, shows scoreboard + recent outcomes. Gives
# Cameron + the meta calibrator a signal on which recommender is
# trustworthy and which is leaking WAR.
# ════════════════════════════════════════════════════════════════
try:
    from app.core.recommendation_tracker import (
        get_scoreboard, get_recent_recommendations,
    )
    _scoreboard = get_scoreboard()
    _recent = get_recent_recommendations(limit=30)
    if _scoreboard or _recent:
        st.divider()
        st.markdown("### \U0001f4c8 Recommendation Outcomes")
        st.caption(
            "Every rec the engines emit is logged. When you follow one, "
            "we track the player's WAR/meta over the next few days and score it."
        )
        if _scoreboard:
            _score_cols = st.columns(min(len(_scoreboard), 4))
            for _i, (_src, _stats) in enumerate(sorted(_scoreboard.items())):
                with _score_cols[_i % len(_score_cols)]:
                    total = _stats.get('total') or 0
                    followed = _stats.get('followed') or 0
                    partial = _stats.get('partial') or 0
                    acted = followed + partial
                    acted_rate = _stats.get('acted_rate')
                    acted_str = f"{acted_rate*100:.0f}% acted" if acted_rate is not None else '—'
                    st.metric(
                        _src.replace('_', ' ').title(),
                        acted_str,
                        delta=f"{followed} followed · {partial} partial · {total} total",
                        delta_color='off',
                    )
                    _hr = _stats.get('hit_rate')
                    _avg_delta = _stats.get('avg_meta_delta')
                    if _hr is not None:
                        st.caption(f"hit rate (positive verdict): {_hr*100:.0f}%")
                    if _avg_delta is not None:
                        st.caption(f"avg meta Δ: {_avg_delta:+.1f}")
        if _recent:
            with st.expander("Recent recommendations (last 30)", expanded=False):
                _rec_rows = []
                for _r in _recent:
                    _ver = _r.get('verdict') or '—'
                    _icon = {'positive':'\U0001f7e2','neutral':'\u26aa',
                              'negative':'\U0001f534','pending':'\u23f3'}.get(_ver, '')
                    _acted = _r.get('action_type') or '—'
                    _rec_rows.append({
                        'When': (_r.get('created_at') or '')[:16],
                        'Source': (_r.get('source') or '').replace('_', ' '),
                        'Type': _r.get('rec_type') or '',
                        'Pos': _r.get('pos') or '',
                        'Target': (_r.get('player_name') or '')[:28],
                        'Replacing': (_r.get('from_player') or '')[:24],
                        'Action': _acted,
                        'Verdict': f"{_icon} {_ver}".strip(),
                        'ExpΔ': (f"{_r.get('expected_delta'):+.0f}"
                                 if _r.get('expected_delta') is not None else ''),
                        'ObsΔ': (f"{(_r.get('meta_after') or 0) - (_r.get('meta_before') or 0):+.0f}"
                                 if _r.get('meta_after') is not None and _r.get('meta_before') is not None else ''),
                    })
                st.dataframe(
                    pd.DataFrame(_rec_rows), width='stretch', hide_index=True,
                    height=min(35 * len(_rec_rows) + 40, 420),
                )
except Exception as _rec_panel_err:
    st.caption(f"Rec outcomes unavailable: {_rec_panel_err}")

# ════════════════════════════════════════════════════════════════
# STRATEGY SLIDERS — mirrors OOTP's in-game strategy panel
# Recommended positions derived from active roster composition.
# Fires automatically after AI Optimize All finishes.
# ════════════════════════════════════════════════════════════════
try:
    from app.core.strategy_recommender import recommend_strategy

    # Pull rating data for active batters / pitchers by name via cards table.
    # active_by_pos + starters + all_by_pos are already built earlier in the
    # page. We enrich them with ratings needed by the recommender.
    _bat_names = sorted({
        p.get('player_name') or ''
        for _pos, _lst in active_by_pos.items()
        for p in _lst
        if _pos in ('C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH')
    })
    _pit_names = sorted({
        p.get('player_name') or ''
        for _pos, _lst in active_by_pos.items()
        for p in _lst
        if _pos in ('SP', 'RP', 'CL')
    })
    _strategy_bat: list[dict] = []
    if _bat_names:
        _placeholders = ','.join('?' * len(_bat_names))
        for _row in conn.execute(f"""
            SELECT card_title, contact, gap_power, power, eye, avoid_ks, babip,
                   speed, stealing, baserunning, bunt_for_hit,
                   contact_vl, contact_vr, power_vl, power_vr,
                   catcher_arm, position
            FROM cards
            WHERE owned = 1
              AND card_title IN ({_placeholders})
        """, _bat_names).fetchall():
            d = dict(_row)
            d['player_name'] = d.get('card_title') or ''
            # Map position integer to label so the catcher filter works
            _pos_num = d.get('position') or 0
            d['position'] = {2: 'C', 3: '1B', 4: '2B', 5: '3B', 6: 'SS',
                             7: 'LF', 8: 'CF', 9: 'RF', 10: 'DH'}.get(
                int(_pos_num) if _pos_num else 0, '')
            _strategy_bat.append(d)
    _strategy_pit: list[dict] = []
    if _pit_names:
        _placeholders = ','.join('?' * len(_pit_names))
        for _row in conn.execute(f"""
            SELECT card_title, stuff, movement, control, p_hr, stamina, hold,
                   stuff_vl, stuff_vr, pitcher_role_name, meta_score_pitching
            FROM cards
            WHERE owned = 1
              AND card_title IN ({_placeholders})
        """, _pit_names).fetchall():
            d = dict(_row)
            d['player_name'] = d.get('card_title') or ''
            _strategy_pit.append(d)

    _sp_list = [p for p in _strategy_pit
                if (p.get('pitcher_role_name') or '').upper() == 'SP']
    _bp_list = [p for p in _strategy_pit
                if (p.get('pitcher_role_name') or '').upper() in ('RP', 'CL')]
    # Bench batters: all owned batters NOT in the active lineup
    _active_names = {p.get('player_name') for _lst in active_by_pos.values()
                     for p in _lst}
    _strategy_bench: list[dict] = []
    _bench_rows = conn.execute("""
        SELECT card_title, contact, gap_power, power, eye, avoid_ks,
               speed, stealing, baserunning, bunt_for_hit, position
        FROM cards
        WHERE owned = 1 AND position NOT IN (1, 0, NULL)
    """).fetchall()
    for _row in _bench_rows:
        if (_row['card_title'] or '') in _active_names:
            continue
        d = dict(_row)
        d['player_name'] = d.get('card_title') or ''
        _strategy_bench.append(d)

    _strategy_recs = recommend_strategy(
        active_batters=_strategy_bat,
        active_pitchers=_strategy_pit,
        bench_batters=_strategy_bench,
        bullpen_pitchers=_bp_list,
        starters=_sp_list,
    )

    if _strategy_recs:
        st.divider()
        st.markdown("### \U0001f39a\ufe0f Strategy Sliders")
        st.caption(
            "Recommended OOTP slider positions based on your active roster. "
            "Mirror these in **Team \u2192 Strategy** to align in-game AI with "
            "your lineup's strengths."
        )
        # Two columns of sliders with labels on left and bucket on right
        _cols = st.columns(2)
        for _i, _rec in enumerate(_strategy_recs):
            with _cols[_i % 2]:
                # Section color accent
                _accent = {
                    'Offensive':    '\U0001f3c3',   # runner
                    'Pitching':     '\u26be',        # ball
                    'Substitution': '\U0001f501',    # swap
                }.get(_rec['section'], '')
                _pos = _rec['position']
                _bucket = _rec['bucket']
                _impact = _rec['impact']
                # Visual slider: markdown-rendered bar
                _bar_pos = int(_pos / 100 * 20)
                _bar = '─' * _bar_pos + '●' + '─' * (20 - _bar_pos)
                st.markdown(
                    f"**{_accent} {_rec['label']}**  \n"
                    f"`{_bar}` **{_bucket}**"
                )
                st.caption(f"\U0001f4a1 {_rec['reason']}")

    # ── Per-player strategy overrides ──
    # Surfaces OOTP's Player Strategy tab recommendations. We only flag a
    # slider when the player's ratings clearly diverge from whatever the
    # team-level default would be.
    from app.core.strategy_recommender import recommend_all_player_strategy
    _player_ovr = recommend_all_player_strategy(
        active_batters=_strategy_bat,
        active_pitchers=_strategy_pit,
    )
    if _player_ovr:
        st.markdown("#### \U0001f464 Per-Player Strategy Overrides")
        st.caption(
            "Set these in **Strategy \u2192 Player Strategy \u2192 [player]** "
            "to override the team default. Players with 0 overrides inherit the "
            "team setting."
        )
        _rows = []
        for _p in _player_ovr:
            for _o in _p['overrides']:
                _rows.append({
                    'Player': _p['name'],
                    'Pos': _p['pos'],
                    'Role': _p['role'],
                    'Slider': _o['slider'],
                    'Setting': _o['bucket'],
                    'Why': _o['reason'],
                })
        if _rows:
            st.dataframe(
                pd.DataFrame(_rows),
                use_container_width=True,
                hide_index=True,
                height=min(35 * len(_rows) + 40, 450),
                column_config={
                    'Player': st.column_config.TextColumn(width='medium'),
                    'Pos': st.column_config.TextColumn(width='small'),
                    'Role': st.column_config.TextColumn(width='small'),
                    'Slider': st.column_config.TextColumn(width='medium'),
                    'Setting': st.column_config.TextColumn(width='small'),
                    'Why': st.column_config.TextColumn(width='large'),
                },
            )
            st.caption(
                f"**{len({r['Player'] for r in _rows})} players** with "
                f"**{len(_rows)} total overrides**. All other players "
                f"inherit the team strategy defaults above."
            )
except Exception as _strategy_err:
    st.caption(f"Strategy sliders unavailable: {_strategy_err}")

# ════════════════════════════════════════════════════════════════
# PP-AWARE PURCHASE PLAN — fits the best engine buys into your budget
# ════════════════════════════════════════════════════════════════
try:
    from app.core.purchase_planner import build_purchase_plan
    _cfg_pp = load_config().get('pp_budget', 2000)
    _pp_col_hdr, _pp_col_budget = st.columns([4, 1])
    with _pp_col_hdr:
        st.markdown("### \U0001f4b0 Purchase Plan")
        st.caption(
            "Budget-fit list of the highest-efficiency market buys from "
            "your engine recs. Greedy-packed by meta-Δ per 1,000 PP."
        )
    with _pp_col_budget:
        _pp_budget = st.number_input(
            "Budget (PP)", min_value=0, max_value=1_000_000,
            value=int(_cfg_pp), step=500, key="pp_budget_input",
            label_visibility="visible",
        )
    _pp_plan = build_purchase_plan(upgrade_plan, int(_pp_budget))
    if _pp_plan.picked:
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric("Picks", len(_pp_plan.picked))
        _mc2.metric("Spend", f"{_pp_plan.total_cost:,} PP")
        _mc3.metric("+Meta total", f"+{int(_pp_plan.total_delta)}")
        _mc4.metric("Budget left", f"{_pp_plan.remaining_budget:,} PP")
        st.dataframe(
            pd.DataFrame(_pp_plan.to_table_rows()),
            width='stretch', hide_index=True,
            height=min(35 * len(_pp_plan.picked) + 40, 360),
            column_config={
                'Pos': st.column_config.TextColumn(width='small'),
                'Buy': st.column_config.TextColumn(width='medium'),
                'Replacing': st.column_config.TextColumn(width='medium'),
                '+Meta': st.column_config.TextColumn(width='small',
                    help="Meta points gained over the current starter."),
                'Cost PP': st.column_config.TextColumn(width='small',
                    help="Estimated PP cost from recent market sales."),
                'Δ / 1kPP': st.column_config.TextColumn(width='small',
                    help="Meta points gained per 1,000 PP — higher is better. "
                         "Picks are sorted by this."),
                'After buy': st.column_config.TextColumn(width='small'),
            },
        )
        if _pp_plan.skipped_too_expensive:
            with st.expander(
                f"\u26a0\ufe0f {len(_pp_plan.skipped_too_expensive)} "
                f"upgrades didn't fit your budget", expanded=False,
            ):
                st.caption(
                    "These picks would improve the roster but cost more than "
                    "remaining budget at their position in the ranking."
                )
                _rows = [{
                    'Pos': p.pos, 'Buy': p.card_name,
                    'Cost PP': f'{p.price:,}',
                    '+Meta': f'+{int(p.delta)}',
                    'Δ / 1kPP': f'{p.efficiency:.1f}',
                } for p in _pp_plan.skipped_too_expensive[:20]]
                st.dataframe(pd.DataFrame(_rows), width='stretch', hide_index=True)
    else:
        st.caption(
            "No market upgrades fit the current budget. Try increasing it, "
            "or drop the engine's min-meta-improvement threshold in Settings."
        )
except Exception as _pp_err:
    st.caption(f"Purchase Plan unavailable: {_pp_err}")


# ════════════════════════════════════════════════════════════════
# NEW ARRIVALS — cards acquired since last run, auto-logged as intake recs
# ════════════════════════════════════════════════════════════════
try:
    from app.core.card_intake import get_unacknowledged_intake, acknowledge_intake
    _new_arrivals = get_unacknowledged_intake(limit=20)
    if _new_arrivals:
        st.markdown("### \U0001f4e5 New Arrivals")
        st.caption(
            "Cards the engine detected as new since you last checked. "
            "The engine will analyze each and surface any risks inline in "
            "your chain-table tooltips."
        )
        _na_rows = []
        for _c in _new_arrivals:
            meta = _c.get('meta_score_batting') or _c.get('meta_score_pitching')
            _na_rows.append({
                'When': (_c.get('first_seen_at') or '')[:16],
                'Card': (_c.get('card_title') or '')[:36],
                'Tier': _c.get('tier_name') or '',
                'Pos': _c.get('position_name') or _c.get('pitcher_role_name') or '',
                'OVR': _c.get('card_value') or '',
                'Meta': int(meta) if meta else '',
                'Suggestion': (_c.get('reasoning') or '')[:80],
            })
        st.dataframe(
            pd.DataFrame(_na_rows), width='stretch', hide_index=True,
            height=min(35 * len(_na_rows) + 40, 300),
        )
        if st.button("\u2705 Acknowledge all new arrivals",
                     key='ack_new_arrivals'):
            for _c in _new_arrivals:
                try:
                    acknowledge_intake(int(_c['card_id']))
                except Exception:
                    pass
            st.toast(f"Acknowledged {len(_new_arrivals)} new arrival(s)", icon='\u2705')
            st.rerun()
except Exception as _intake_err:
    st.caption(f"New Arrivals unavailable: {_intake_err}")


# ════════════════════════════════════════════════════════════════
# RECENT PERFORMANCE — last-N-game batting + pitching leaders
# ════════════════════════════════════════════════════════════════
try:
    from app.core.recent_performance import recent_performance
    _team = load_config().get('team_name') or 'Toronto Dark Knights'
    _rp_window = st.slider(
        "Recent games window", min_value=5, max_value=30, value=10, step=1,
        key="rp_window", help="How many most-recent team games to aggregate."
    )
    _rp = recent_performance(_team, window_games=int(_rp_window))
    if _rp.team_games:
        st.markdown(
            f"### \U0001f4ca Recent Performance \u2014 last {_rp.team_games} games "
            f"({_rp.date_from} → {_rp.date_to})"
        )
        _rp_c1, _rp_c2 = st.columns(2)
        with _rp_c1:
            st.markdown("**Top batters (recent)**")
            if _rp.top_batters:
                _rp_bat = [{
                    'Player': b.player_name,
                    'PA': b.pa,
                    'AVG': f"{b.avg:.3f}",
                    'OPS': f"{b.ops:.3f}",
                    'HR/RBI': f"{b.hr}/{b.rbi}",
                    'K': b.k,
                    'wRC+ ~': int(b.wrc_proxy) if b.wrc_proxy else '',
                } for b in _rp.top_batters]
                st.dataframe(pd.DataFrame(_rp_bat), width='stretch',
                             hide_index=True,
                             height=min(35 * len(_rp_bat) + 40, 320))
            else:
                st.caption("No batting data in window.")
        with _rp_c2:
            st.markdown("**Top pitchers (recent)**")
            if _rp.top_pitchers:
                _rp_pit = [{
                    'Player': p.player_name,
                    'G': p.games,
                    'IP': p.ip,
                    'ERA': f"{p.era:.2f}",
                    'WHIP': f"{p.whip:.2f}",
                    'K/9': f"{p.k_per_9:.1f}",
                    'K/BB': (f"{p.k/max(p.bb,1):.1f}" if p.bb else f"{p.k}"),
                } for p in _rp.top_pitchers]
                st.dataframe(pd.DataFrame(_rp_pit), width='stretch',
                             hide_index=True,
                             height=min(35 * len(_rp_pit) + 40, 320))
            else:
                st.caption("No pitching data in window.")
        # Clutch events roll-up
        if _rp.clutch_recent:
            with st.expander(
                f"\u26be Clutch events in window ({len(_rp.clutch_recent)})",
                expanded=False,
            ):
                _cl_rows = [{
                    'Event': c.get('event_type'),
                    'Player': c.get('player_name') or '?',
                    'Count': c.get('event_count') or 1,
                } for c in _rp.clutch_recent]
                st.dataframe(pd.DataFrame(_cl_rows), width='stretch',
                             hide_index=True)
    else:
        st.caption(
            f"No games found for {_team} in the recent window \u2014 ingest "
            "some HTML box scores via the background watcher."
        )
except Exception as _rp_err:
    st.caption(f"Recent Performance unavailable: {_rp_err}")


# ════════════════════════════════════════════════════════════════
# DETAIL EXPANDERS — below the fold
# ════════════════════════════════════════════════════════════════
st.divider()

# Market shopping list — AI-enhanced when available
if market_buys:
    # If AI has been run, prioritize AI buy picks and add AI reasoning
    ai_buy_picks = {}
    picks_data = st.session_state.get('ai_optimize_picks_data', [])
    for p in picks_data:
        if p['action'] == 'Buy':
            ai_buy_picks[p['pos']] = p

    # Sort: AI-recommended buys first, then by meta/PP efficiency
    def _market_sort_key(u):
        is_ai = 1 if u['pos'] in ai_buy_picks else 0
        p = u['market_price'] or 1
        eff = (u['market_delta'] / p) if p > 0 else 0
        return (-is_ai, -eff)

    market_sorted = sorted(market_buys, key=_market_sort_key)
    total_cost = sum(u['market_price'] or 0 for u in market_buys)
    ai_label = " (AI-prioritized)" if ai_buy_picks else ""
    with st.expander(f"🛒 Market Shopping List{ai_label} — {len(market_buys)} cards, {total_cost:,} PP"):
        mkt_rows = []
        for u in market_sorted:
            p = u['market_price'] or 0
            delta_total = round((u['market_meta'] or 0) - u['current_meta'])
            eff = (delta_total / p) if p > 0 else 0
            ai_p = ai_buy_picks.get(u['pos'])
            ai_note = ""
            if ai_p:
                ai_note = f"🧠 {ai_p.get('reason', 'AI recommended')}"
            mkt_rows.append({
                "Slot": u['pos'],
                "Replaces": u['current_name'],
                "Buy": short_name(u['market_name']),
                "+Meta": f"+{delta_total}",
                "Cost": p,
                "Eff": round(eff, 3),
                "AI": ai_note,
            })
        st.dataframe(pd.DataFrame(mkt_rows), use_container_width=True, hide_index=True,
                     column_config={
                         "Cost": st.column_config.NumberColumn(format="%d PP"),
                         "Eff": st.column_config.NumberColumn(format="%.2f meta/PP", help="Total meta gained per PP spent"),
                         "AI": st.column_config.TextColumn(width="large", help="AI reasoning for this purchase"),
                     })

# Alternatives per position
with st.expander("Alternative Upgrade Options (per position)"):
    for u in upgrade_plan:
        ow = u.get('_owned_upgrades', [])
        mk = u.get('_market_upgrades', [])
        if not ow and not mk:
            continue
        st.markdown(f"**{u['pos']}: {u['current_name']}** (meta {u['current_meta']})")
        alt_rows = []

        def _compact_pos_note(a: dict) -> str:
            """Short "(as LF r32)" note for non-primary assignments; empty otherwise."""
            note = a.get('position_annotation') or ''
            if not note:
                return ''
            inner = note.strip(' ()').replace('played as ', '').replace(', rating ', ' r')
            return f" ({inner})"

        for a in ow:
            alt_rows.append({
                "Card": short_name(a.get('card_title', '')) + _compact_pos_note(a),
                "OVR": a.get('card_value', 0), "Meta": round(a.get('meta_score', 0) or 0),
                "+": round((a.get('meta_score', 0) or 0) - u['current_meta']),
                "Source": f"\U0001f4e6 {a.get('action', 'FREE')}",
            })
        for a in mk:
            p = a.get('last_10_price', 0) or 0
            # Badge aspirational (over-budget) cards so the user sees them
            # as "save up" targets rather than immediately-buyable options.
            if a.get('aspirational'):
                source = f"\U0001f6d2 {p:,} PP (save up)" if p else "\U0001f6d2 Market (save up)"
            else:
                source = f"\U0001f6d2 {p:,} PP" if p else "\U0001f6d2 Market"
            alt_rows.append({
                "Card": short_name(a.get('card_title', '')) + _compact_pos_note(a),
                "OVR": a.get('card_value', 0), "Meta": round(a.get('meta_score', 0) or 0),
                "+": round((a.get('meta_score', 0) or 0) - u['current_meta']),
                "Source": source,
            })
        if alt_rows:
            st.dataframe(pd.DataFrame(alt_rows), use_container_width=True, hide_index=True)

# AI Scouting
with st.expander("\U0001f9e0 AI Scouting Reports"):
    ai_config = get_ai_config()
    if not ai_config["ready"]:
        st.warning(f"AI scouting unavailable: {ai_config['message']}")
    else:
        scout_slots = [u for u in upgrade_plan if u['owned_name'] or u['market_name']]
        if not scout_slots:
            st.info("No upgrade candidates to scout.")
        else:
            def _scout_label(u):
                parts = [f"{u['pos']}: {u['current_name']} →"]
                if u['owned_name']:
                    parts.append(f"{short_name(u['owned_name'])} (+{u['owned_delta']})")
                if u['market_name']:
                    parts.append(f"/ {short_name(u['market_name'])} (+{u['market_delta']})")
                return " ".join(parts)
            slot_options = [_scout_label(u) for u in scout_slots]
            col_pick, col_btn = st.columns([3, 1])
            with col_pick:
                selected = st.selectbox("Scout a position", slot_options, key="scout_select")
            with col_btn:
                st.write("")
                run_scout = st.button("\U0001f50d Scout", type="primary")

            if run_scout and selected:
                idx = slot_options.index(selected)
                u = scout_slots[idx]
                with st.spinner(f"Scouting {u['pos']}..."):
                    current_full = get_full_card_data(u['current_name'], conn)
                    if not current_full:
                        current_full = {'player_name': u['current_name'], 'ovr': u['current_ovr']}
                    candidates = []
                    for a in u.get('_owned_upgrades', [])[:2]:
                        cd = get_full_card_data(a.get('card_id') or a.get('card_title', ''), conn)
                        if cd: cd['_source'] = 'collection'; candidates.append(cd)
                    for a in u.get('_market_upgrades', [])[:2]:
                        cd = get_full_card_data(a.get('card_id') or a.get('card_title', ''), conn)
                        if cd: cd['_source'] = 'market'; candidates.append(cd)
                    if candidates:
                        team_ctx = build_team_context(conn)
                        result = get_upgrade_scouting_report(u['pos'], current_full, candidates, team_ctx, conn=conn)
                        if result.get('response'):
                            st.markdown(result['response'])
                        elif result.get('error'):
                            st.error(result['error'])

# ════════════════════════════════════════════════════════════════
# "Why this meta?" explainer (Tier-2 #10)
# ════════════════════════════════════════════════════════════════
# Single picker that lets the user open the meta breakdown for ANY upgrade
# candidate surfaced on this page (current starter, owned upgrade target, or
# market upgrade target). Aimed at answering "the formula says X is better
# than Y by 30 — what's actually driving that gap?" without forcing the user
# to navigate to Card Detail for each comparison.
with st.expander("\U0001f50d Why this meta? (formula breakdown)", expanded=False):
    _explainer_options = {}
    for u in upgrade_plan:
        # Owned upgrade target (free promote)
        for a in u.get('_owned_upgrades', [])[:3] or []:
            cid = a.get('card_id')
            ct = a.get('card_title')
            if cid and ct:
                _label = f"\U0001f4e6 {u['pos']}: {ct} (meta {round(a.get('meta_score') or 0)})"
                _explainer_options[_label] = cid
        # Market upgrade target (paid)
        for a in u.get('_market_upgrades', [])[:3] or []:
            cid = a.get('card_id')
            ct = a.get('card_title')
            if cid and ct:
                _price = a.get('last_10_price') or 0
                _label = f"\U0001f6d2 {u['pos']}: {ct} ({_price:,} PP, meta {round(a.get('meta_score') or 0)})"
                _explainer_options[_label] = cid
    if _explainer_options:
        st.caption(
            "Pick any upgrade candidate to see exactly which ratings drove its meta score. "
            "Use this to sanity-check why one card outranks another \u2014 e.g. is the gap "
            "from raw power, position scarcity, or a SIERA interaction term?"
        )
        _picked = st.selectbox(
            "Upgrade candidate",
            list(_explainer_options.keys()),
            index=0,
            key="optimizer_explainer_pick",
        )
        _picked_id = _explainer_options.get(_picked)
        if _picked_id:
            try:
                _full = _fetch_card_for_explainer(conn, _picked_id)
                _is_pit = bool(_full and (_full.get('pitcher_role') or _full.get('pitcher_role_name')))
                _exp = explain_meta(_full or {}, is_pitcher=_is_pit) if _full else None
            except Exception as _e:
                _exp = None
                st.caption(f"(explainer unavailable: {_e})")
            _render_meta_explainer(_exp)
    else:
        st.caption("No upgrade candidates available to explain — load roster + market data first.")

# Deferred Manager's Eye auto-call has been REMOVED. The LLM is now a
# verifier, not an auto-analyzer — page renders instantly with engine recs
# and the user opts into LLM review per-recommendation via the Council
# Review section.

conn.close()
