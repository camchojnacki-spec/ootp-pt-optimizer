"""Roster Optimizer — lineup card view with AI team assessment."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection, load_config
from app.core.ai_advisor import (
    get_ai_config, get_upgrade_scouting_report, build_team_context, get_full_card_data,
    ai_optimize_all_positions,
)

st.set_page_config(page_title="Roster Optimizer", page_icon="\U0001f4cb", layout="wide")

conn = get_connection()
config = load_config()
budget = config.get('pp_budget', 500)

# ── Sidebar controls ──
with st.sidebar:
    st.header("Filters")
    max_spend = st.number_input("Max PP per card", min_value=0, max_value=999999,
                                value=budget, step=500, format="%d",
                                help="Maximum PP you're willing to spend on a single card")
    min_improvement = st.number_input("Min meta improvement", min_value=0, max_value=500,
                                      value=20, step=5,
                                      help="Only show upgrades with at least this much meta gain")
    focus = st.selectbox("Focus", ["All Positions", "Batting Only", "Pitching Only", "Weakest First"])

# ── Build roster data ──
# Active roles = players actually in the game lineup right now
ACTIVE_ROLES = {'starter', 'rotation', 'closer', 'bullpen'}

BATS_MAP = {1: 'R', 2: 'L', 3: 'S'}

roster_rows = conn.execute("""
    SELECT r.player_name, r.position, r.lineup_role, r.ovr, r.meta_score,
           r.meta_vs_rhp, r.meta_vs_lhp, r.bats as roster_bats, c.bats
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
    if pos not in all_by_pos:
        all_by_pos[pos] = []
    all_by_pos[pos].append(d)

    if d.get('lineup_role') in ACTIVE_ROLES:
        if pos not in active_by_pos:
            active_by_pos[pos] = []
        active_by_pos[pos].append(d)
        # For single-slot bat positions, pick best active player
        if pos not in starters or (d['meta_score'] or 0) > (starters[pos]['meta_score'] or 0):
            starters[pos] = d

# ── Load in-game performance for context ──
_perf_bat = {}
_perf_pit = {}
_latest_snap = conn.execute("SELECT MAX(snapshot_date) as d FROM batting_stats").fetchone()
if _latest_snap and _latest_snap['d']:
    for r in conn.execute("""SELECT player_name, pa, war, ops,
            CASE WHEN pa > 0 THEN war * 600.0 / pa ELSE 0 END as war600
        FROM batting_stats WHERE snapshot_date = ? AND pa >= 50 AND ab > 0""",
        (_latest_snap['d'],)).fetchall():
        _perf_bat[r['player_name']] = dict(r)
_latest_psnap = conn.execute("SELECT MAX(snapshot_date) as d FROM pitching_stats").fetchone()
if _latest_psnap and _latest_psnap['d']:
    for r in conn.execute("""SELECT player_name, ip, era, fip, war,
            CASE WHEN ip > 0 THEN war * 200.0 / ip ELSE 0 END as war200
        FROM pitching_stats WHERE snapshot_date = ? AND ip >= 10 AND (k > 0 OR era > 0)""",
        (_latest_psnap['d'],)).fetchall():
        _perf_pit[r['player_name']] = dict(r)

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
    """Find bench/reserve players at this ROSTER position who beat the starter.

    This catches players whose card position differs from their roster position
    (e.g., a CF card assigned to LF by the manager). These players wouldn't
    appear in find_owned_upgrades which queries by card position_name.

    Performance gate: if the current starter is producing well in-game (WAR/600 >= 1.5)
    and the bench player has no performance data, don't recommend the swap.

    exclude_names should contain PLAYER NAMES (not card titles).
    """
    exclude_names = exclude_names or []

    # Check if current starter is performing well
    starter_perf = None
    if current_player_name:
        starter_perf = _perf_bat.get(current_player_name) or _perf_pit.get(current_player_name)

    bench_upgrades = []
    for p in all_by_pos.get(pos_value, []):
        if p.get('lineup_role') in ('starter', 'rotation', 'closer', 'bullpen'):
            continue  # Skip active players — we only want bench/reserve
        pname = p['player_name']
        if pname in exclude_names:
            continue
        # Also check if player name appears in any exclude entry (handles card titles in list)
        if any(pname in ex or ex in pname for ex in exclude_names):
            continue
        m = p.get('meta_score') or 0
        if m > current_meta + min_improvement:
            # Performance gate: don't recommend benching a producing starter
            # for someone with no game data
            bench_perf = _perf_bat.get(pname) or _perf_pit.get(pname)
            if starter_perf and not bench_perf:
                war600 = starter_perf.get('war600', starter_perf.get('war200', 0))
                if war600 >= 1.5:
                    continue  # Starter proving their value — skip this bench player

            bench_upgrades.append({
                'card_id': p.get('card_id'),
                'card_title': p.get('card_title') or pname,
                'player_name': pname,
                'card_value': p.get('ovr'),
                'meta_score': m,
                'last_10_price': 0,
                'action': 'Promote',
            })
    bench_upgrades.sort(key=lambda x: -(x['meta_score'] or 0))
    return bench_upgrades


def find_owned_upgrades(pos_value, current_meta, is_pitching, exclude_names=None, limit=5, current_player_name=None):
    exclude_names = exclude_names or []

    # First: check roster bench/reserve at this position (catches position mismatches)
    bench_ups = find_roster_bench_upgrades(pos_value, current_meta, exclude_names, current_player_name)

    # Then: search cards table by card's natural position
    meta_col = "meta_score_pitching" if is_pitching else "meta_score_batting"
    pos_col = "pitcher_role_name" if is_pitching else "position_name"

    # DH special case: any non-pitcher can DH, so search ALL batting positions
    if pos_value == 'DH' and not is_pitching:
        results = conn.execute(f"""
            SELECT c.card_id, c.card_title, c.tier_name, c.card_value,
                   c.{meta_col} as meta_score, c.last_10_price,
                   mc.status as collection_status, r.lineup_role as roster_role
            FROM cards c
            LEFT JOIN my_collection mc ON mc.card_id = c.card_id
            LEFT JOIN roster r ON c.card_title LIKE '%' || r.player_name || '%'
            WHERE c.owned = 1 AND c.{meta_col} > ?
                AND c.pitcher_role IS NULL
            GROUP BY c.card_id ORDER BY c.{meta_col} DESC LIMIT ?
        """, (current_meta + min_improvement, limit + len(exclude_names) + 5)).fetchall()
    else:
        results = conn.execute(f"""
            SELECT c.card_id, c.card_title, c.tier_name, c.card_value,
                   c.{meta_col} as meta_score, c.last_10_price,
                   mc.status as collection_status, r.lineup_role as roster_role
            FROM cards c
            LEFT JOIN my_collection mc ON mc.card_id = c.card_id
            LEFT JOIN roster r ON c.card_title LIKE '%' || r.player_name || '%' AND r.position = c.{pos_col}
            WHERE c.{pos_col} = ? AND c.owned = 1 AND c.{meta_col} > ?
            GROUP BY c.card_id ORDER BY c.{meta_col} DESC LIMIT ?
        """, (pos_value, current_meta + min_improvement, limit + len(exclude_names) + 5)).fetchall()

    filtered = []
    # Track names already added from bench to avoid duplicates
    bench_names = {b['card_title'] for b in bench_ups}

    for r in results:
        title = r['card_title'] or ''
        if any(name in title for name in exclude_names):
            continue
        if title in bench_names:
            continue  # Already captured from roster bench
        d = dict(r)
        status, role = d.get('collection_status', ''), d.get('roster_role', '')
        if status == 'Inactive':
            d['action'] = 'Activate'
        elif status == 'Reserve Roster':
            d['action'] = 'Promote'
        elif role in ('bench', 'reserve'):
            d['action'] = 'Move Up'
        else:
            d['action'] = 'Swap In'
        filtered.append(d)

    # Merge: bench players first (they're already on the team), then cards table results
    combined = bench_ups + filtered
    combined.sort(key=lambda x: -(x['meta_score'] or 0))
    return combined[:limit]


def find_market_upgrades(pos_value, current_meta, is_pitching, exclude_ids=None, limit=5):
    exclude_ids = exclude_ids or set()
    meta_col = "meta_score_pitching" if is_pitching else "meta_score_batting"
    pos_col = "pitcher_role_name" if is_pitching else "position_name"
    max_price = max_spend if max_spend > 0 else 999999999

    # DH: any non-pitcher on the market can DH
    if pos_value == 'DH' and not is_pitching:
        results = conn.execute(f"""
            SELECT card_id, card_title, tier_name, card_value,
                   {meta_col} as meta_score, last_10_price
            FROM cards
            WHERE owned = 0 AND last_10_price > 0 AND pitcher_role IS NULL
                AND last_10_price <= ? AND {meta_col} > ?
            ORDER BY {meta_col} DESC LIMIT ?
        """, (max_price, current_meta + min_improvement, limit + len(exclude_ids) + 5)).fetchall()
    else:
        results = conn.execute(f"""
            SELECT card_id, card_title, tier_name, card_value,
                   {meta_col} as meta_score, last_10_price
            FROM cards
            WHERE {pos_col} = ? AND owned = 0 AND last_10_price > 0
                AND last_10_price <= ? AND {meta_col} > ?
            ORDER BY {meta_col} DESC LIMIT ?
        """, (pos_value, max_price, current_meta + min_improvement, limit + len(exclude_ids) + 5)).fetchall()
    return [dict(r) for r in results if r['card_id'] not in exclude_ids][:limit]


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


def _build_slot(pos_label, current_name, current_ovr, current_meta, owned_ups, market_ups, bats_hand='?'):
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

    return {
        'pos': pos_label,
        'current_name': current_name,
        'current_ovr': current_ovr,
        'current_meta': round(current_meta),
        'bats': bats_hand,
        # Owned upgrade (free)
        'owned_name': bo['card_title'] if bo else None,
        'owned_ovr': bo.get('card_value') if bo else None,
        'owned_meta': owned_meta,
        'owned_delta': owned_delta,
        'owned_action': bo.get('action') if bo else None,
        # Market upgrade (paid)
        'market_name': bm['card_title'] if bm else None,
        'market_ovr': bm.get('card_value') if bm else None,
        'market_meta': market_meta,
        'market_delta': market_delta,
        'market_price': bm.get('last_10_price') if bm else None,
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
            entry = _build_slot(f"SP{i+1}", sp['player_name'], sp['ovr'], m, ow, mk)
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
            entry = _build_slot(label, rp['player_name'], rp['ovr'], m, ow, mk)
            if entry['owned_name']:
                bo = ow[0] if ow else None
                pname = bo.get('player_name', entry['owned_name']) if bo else entry['owned_name']
                used_names.add(pname)
                used_names.add(entry['owned_name'])
            rp_entries[i] = entry
        upgrade_plan.extend(rp_entries)
        continue

    # ── DH slot: inferred from "extra" starters at other positions ──
    if pos == 'DH':
        # Collect all starters not already shown at a field position.
        # The best one at each position is the fielder; extras are DH candidates.
        _field_shown = {e['pos'] for e in upgrade_plan if e['pos'] in bat_field_positions}
        _field_names = {e['current_name'] for e in upgrade_plan if e['pos'] in bat_field_positions}
        dh_candidates = []
        for fpos in bat_field_positions:
            for p in active_by_pos.get(fpos, []):
                if p['player_name'] not in _field_names:
                    dh_candidates.append(p)
        dh_candidates.sort(key=lambda p: p['meta_score'] or 0, reverse=True)

        if dh_candidates:
            player = dh_candidates[0]
        else:
            # No extra starters — DH is empty, suggest best available hitter
            upgrade_plan.append(_build_slot('DH', '(empty)', 0, 0,
                find_owned_upgrades('DH', 0, False, list(_all_active_names), 5),
                find_market_upgrades('DH', 0, False, used_market_ids, 5)))
            continue

        m = player['meta_score'] or 0
        bh = player.get('bats_hand', '?')
        active_names = list(_all_active_names)
        # DH upgrades search ALL batting positions — anyone can DH
        ow = find_owned_upgrades('DH', m, False, active_names, 3, current_player_name=player['player_name'])
        mk = find_market_upgrades('DH', m, False, used_market_ids, 3)
        entry = _build_slot('DH', player['player_name'], player['ovr'], m, ow, mk, bats_hand=bh)
        entry['is_platoon'] = False
        if entry['owned_name']:
            bo = ow[0] if ow else None
            pname = bo.get('player_name', entry['owned_name']) if bo else entry['owned_name']
            used_owned_titles.add(pname)
        upgrade_plan.append(entry)
        continue

    # ── Standard batting positions: 1 slot per position (best starter) ──
    active_players = active_by_pos.get(pos, [])
    if not active_players:
        upgrade_plan.append(_build_slot(pos, '(empty)', 0, 0,
            find_owned_upgrades(pos, 0, is_pitching, [], 5),
            find_market_upgrades(pos, 0, is_pitching, used_market_ids, 5)))
        continue

    # Sort by meta descending — best player starts at this position
    active_players = sorted(active_players, key=lambda p: p['meta_score'] or 0, reverse=True)
    player = active_players[0]
    m = player['meta_score'] or 0
    bh = player.get('bats_hand', '?')
    active_names = list(_all_active_names) + [p['player_name'] for p in active_players]
    ow = find_owned_upgrades(pos, m, is_pitching, active_names, 3, current_player_name=player['player_name'])
    mk = find_market_upgrades(pos, m, is_pitching, used_market_ids, 3)
    entry = _build_slot(pos, player['player_name'], player['ovr'], m, ow, mk, bats_hand=bh)
    entry['is_platoon'] = False
    if entry['owned_name']:
        bo = ow[0] if ow else None
        pname = bo.get('player_name', entry['owned_name']) if bo else entry['owned_name']
        used_owned_titles.add(pname)
    # Flag same-handed starters (platoon gap warning)
    primary_hand = bh
    if len(active_players) > 1 and primary_hand in ('L', 'R'):
        same_hand_count = sum(1 for p in active_players if p.get('bats_hand') == primary_hand)
        if same_hand_count > 1:
            entry['platoon_warning'] = f"\u26a0\ufe0f {same_hand_count} {primary_hand}-batters, no platoon partner"
    upgrade_plan.append(entry)

if focus == "Weakest First":
    upgrade_plan.sort(key=lambda x: x['current_meta'])

# ── Classify ──
collection_swaps = [u for u in upgrade_plan if u['owned_name']]
market_buys = [u for u in upgrade_plan if u['market_name']]
all_upgrades = [u for u in upgrade_plan if u['owned_name'] or u['market_name']]
top_priorities = sorted(all_upgrades, key=lambda x: -x['best_delta'])[:3]

# ── Roster mismatches ──
# Only flag when a bench player genuinely beats the starter AND the starter
# isn't outperforming their meta in real games.
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

# Calculate team grade
bat_metas = [starters[p]['meta_score'] for p in bat_field_positions if p in starters]
pit_metas = [p['meta_score'] for p in active_by_pos.get('SP', [])[:5]]
pit_metas += [p['meta_score'] for p in active_by_pos.get('RP', [])[:7]]
if 'CL' in starters:
    pit_metas.append(starters['CL']['meta_score'])
all_metas = [m for m in bat_metas + pit_metas if m]
avg_meta = sum(all_metas) / len(all_metas) if all_metas else 0
total_meta = sum(all_metas)

if avg_meta >= 700: grade = "A+"
elif avg_meta >= 650: grade = "A"
elif avg_meta >= 600: grade = "A-"
elif avg_meta >= 560: grade = "B+"
elif avg_meta >= 520: grade = "B"
elif avg_meta >= 480: grade = "B-"
elif avg_meta >= 440: grade = "C+"
elif avg_meta >= 400: grade = "C"
else: grade = "D"

col_grade, col_ai, col_stats = st.columns([1, 4, 2])
with col_grade:
    st.metric("Team Grade", grade)
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
# AI OPTIMIZE ALL — button stays above tabs, results go in AI tab
# ════════════════════════════════════════════════════════════════
ai_config_check = get_ai_config()
if ai_config_check["ready"]:
    col_ai_btn, col_ai_status = st.columns([1, 4])
    with col_ai_btn:
        run_ai_optimize = st.button("\U0001f9e0 AI Optimize All", type="primary",
                                     use_container_width=True,
                                     help="Run AI reasoning across all positions — considers splits, performance, team fit")
    if run_ai_optimize:
        with st.spinner("AI analyzing all positions... (10-15 seconds)"):
            ai_result = ai_optimize_all_positions(upgrade_plan, conn, _perf_bat, _perf_pit,
                                                          max_spend_per_card=max_spend)
            if ai_result.get('response'):
                st.session_state['ai_optimize_result'] = ai_result
                st.session_state['ai_optimize_picks'] = ai_result.get('picks', {})
                st.session_state['ai_optimize_picks_data'] = ai_result.get('picks_data', [])
            elif ai_result.get('error'):
                st.error(f"AI error: {ai_result['error']}")

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
# LINEUP CARD — batting + pitching side by side
# ════════════════════════════════════════════════════════════════
st.divider()


def build_chain_rows(positions_list, show_bats=False, show_perf=False):
    """Build compact lineup rows for the roster optimizer table.

    Columns: Pos | Current | Meta | Perf | Action | Why
    - Action: concise "what to do" (Optimal / Promote X / Buy X 2,350PP)
    - Why: short AI or meta reason (platoon warning, AI insight)
    """
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

        # Current player — compact: "Name (OVR H)"
        ovr = u['current_ovr'] or ""
        bh = f" {u.get('bats', '')}" if show_bats and u.get('bats', '?') != '?' else ""
        current_display = f"{u['current_name']} ({ovr}{bh})" if ovr else u['current_name']

        # Pos column — append platoon warning inline
        pos_display = u['pos']
        if u.get('platoon_warning'):
            pos_display += " \u26a0\ufe0f"

        row = {
            "Pos": pos_display,
            "Current": current_display,
            "Meta": u['current_meta'],
        }

        # In-game performance context
        if show_perf:
            name = u['current_name']
            pb = _perf_bat.get(name)
            pp = _perf_pit.get(name)
            if pp:
                era_fip = pp['era'] - pp['fip']
                flag = " \U0001f340" if era_fip < -0.5 else (" \u26a0\ufe0f" if era_fip > 0.5 else "")
                row["Perf"] = f"{pp['era']:.2f} ERA  {pp['war']:.1f}W{flag}"
            elif pb:
                row["Perf"] = f".{int(pb['ops']*1000):03d} OPS  {pb['war600']:.1f}W"
            else:
                row["Perf"] = ""

        # ── Action column — concise "what to do" ──
        ai_pick = _get_ai_pick_for_pos(u['pos'])

        action = ""
        why = ""

        if ai_pick and ai_pick['action'] in ('Promote', 'Buy', 'Platoon'):
            emoji = ai_pick.get('emoji', '')
            card = short_name(ai_pick['card_name'], 25)
            if ai_pick['action'] == 'Promote':
                action = f"{emoji} \U0001f4e6 {card} FREE"
            elif ai_pick['action'] == 'Buy':
                p = ai_pick.get('cost') or (u['market_price'] if u.get('market_price') else 0)
                cost = f"{p:,}" if p else "?"
                action = f"{emoji} \U0001f6d2 {card} {cost}PP"
            elif ai_pick['action'] == 'Platoon':
                partner = ai_pick.get('platoon_partner', '')
                action = f"{emoji} \U0001f91d {card}" + (f" + {short_name(partner, 15)}" if partner else "")
            # AI reason as the "why"
            if ai_pick.get('reason'):
                why = ai_pick['reason']
        elif u['owned_name']:
            action = f"\U0001f4e6 {short_name(u['owned_name'], 25)} +{u['owned_delta']}"
            why = "Promote owned card"
        elif u['market_name']:
            p = u['market_price'] or 0
            cost = f"{p:,}" if p else "?"
            action = f"\U0001f6d2 {short_name(u['market_name'], 25)} {cost}PP"
            why = f"+{u['market_delta']} meta"
        else:
            action = "\u2705 Optimal"

        # Append platoon warning to why
        if u.get('platoon_warning'):
            warn_text = u['platoon_warning'].replace('\u26a0\ufe0f ', '')
            why = f"{warn_text}" if not why else f"{why} | {warn_text}"

        row["Action"] = action
        row["Why"] = why

        rows.append(row)
    return rows


CHAIN_COL_CONFIG = {
    "Pos": st.column_config.TextColumn(width="small",
        help="\u26a0\ufe0f = platoon issue (same-handed batters, no opposite-hand partner)"),
    "Current": st.column_config.TextColumn(width="medium"),
    "Meta": st.column_config.ProgressColumn(min_value=300, max_value=800, format="%d", width="small"),
    "Perf": st.column_config.TextColumn(width="small",
        help="In-game stats. \U0001f340=outperforming (lucky), \u26a0\ufe0f=underperforming (unlucky)"),
    "Action": st.column_config.TextColumn(width="medium",
        help="\U0001f4e6=promote owned card (FREE), \U0001f6d2=market buy, \u2705=no upgrade needed"),
    "Why": st.column_config.TextColumn(width="medium",
        help="AI reasoning or meta-based explanation for the recommendation"),
}

# ── TABBED LAYOUT — full width, no horizontal scrolling ──
tab_bat, tab_pit, tab_ai = st.tabs(["⚾ Batting Lineup", "🎯 Pitching Staff", "🧠 AI Recommendations"])

with tab_bat:
    bat_rows = build_chain_rows(bat_positions, show_bats=True, show_perf=True)
    if bat_rows:
        h = min(35 * len(bat_rows) + 40, 600)
        st.dataframe(pd.DataFrame(bat_rows), use_container_width=True, hide_index=True,
                     height=h, column_config=CHAIN_COL_CONFIG)

    # ── Batting Order Recommendation ──
    st.divider()
    st.markdown("##### Suggested Batting Order")
    st.caption("Based on ratings: OBP/speed at top, power in middle, weakest at bottom")

    # Gather the 9 starters' ratings from cards table
    _lineup_starters = []
    _shown_field = {e['current_name'] for e in upgrade_plan if e['pos'] in bat_field_positions}
    _shown_dh = [e for e in upgrade_plan if e['pos'] == 'DH']
    _dh_name = _shown_dh[0]['current_name'] if _shown_dh else None
    _all_lineup_names = list(_shown_field)
    if _dh_name and _dh_name != '(empty)':
        _all_lineup_names.append(_dh_name)

    for _name in _all_lineup_names:
        # Get position from upgrade_plan
        _pos = '?'
        for e in upgrade_plan:
            if e['current_name'] == _name and e['pos'] in bat_positions:
                _pos = e['pos']
                break
        _card = conn.execute("""
            SELECT contact, gap_power, power, eye, avoid_ks, speed, stealing, baserunning, babip
            FROM cards WHERE card_title LIKE ? AND owned = 1 LIMIT 1
        """, (f'%{_name}%',)).fetchone()
        _perf = _perf_bat.get(_name)
        _meta = 0
        for e in upgrade_plan:
            if e['current_name'] == _name:
                _meta = e['current_meta']
                break
        if _card:
            d = dict(_card)
            d['player_name'] = _name
            d['pos'] = _pos
            d['meta'] = _meta
            d['perf'] = _perf
            # Compute slot fitness scores
            con = d.get('contact') or 0
            gap = d.get('gap_power') or 0
            pwr = d.get('power') or 0
            eye_r = d.get('eye') or 0
            avk = d.get('avoid_ks') or 0
            spd = d.get('speed') or 0
            stl = d.get('stealing') or 0
            # Leadoff score: OBP + speed (get on base, steal)
            d['leadoff'] = con * 1.2 + eye_r * 1.5 + avk * 0.8 + spd * 0.6 + stl * 0.3
            # 2-hole: contact + OBP + some gap (move runners, avoid DP)
            d['two_hole'] = con * 1.3 + eye_r * 1.2 + gap * 0.8 + avk * 0.7
            # 3-hole: best overall hitter (everything matters)
            d['three_hole'] = con * 1.0 + gap * 1.2 + pwr * 1.0 + eye_r * 1.0 + avk * 0.5
            # Cleanup (4): power focus (drive in runs)
            d['cleanup'] = pwr * 1.5 + gap * 1.3 + con * 0.6 + eye_r * 0.5
            # 5-hole: power/RBI secondary
            d['five_hole'] = pwr * 1.2 + gap * 1.1 + con * 0.7 + eye_r * 0.6
            _lineup_starters.append(d)

    if _lineup_starters:
        # Greedy assignment: assign best-fit player to each slot
        _available = list(_lineup_starters)
        _order = []
        _slot_keys = [
            ('leadoff', '1 (Leadoff)'),
            ('two_hole', '2'),
            ('three_hole', '3'),
            ('cleanup', '4 (Cleanup)'),
            ('five_hole', '5'),
        ]
        # Top 5 by slot-specific scores
        for key, label in _slot_keys:
            if not _available:
                break
            best = max(_available, key=lambda p: p.get(key, 0))
            _order.append((label, best))
            _available.remove(best)
        # Remaining: sort by meta descending for slots 6-9
        _available.sort(key=lambda p: p['meta'] or 0, reverse=True)
        for i, p in enumerate(_available):
            _order.append((str(6 + i), p))

        order_rows = []
        for slot, p in _order:
            perf_str = ""
            if p.get('perf'):
                perf_str = f".{int(p['perf']['ops']*1000):03d} OPS  {p['perf']['war600']:.1f}W"
            order_rows.append({
                "#": slot,
                "Player": f"{p['player_name']} ({p['pos']})",
                "CON": p.get('contact', 0),
                "POW": p.get('power', 0),
                "GAP": p.get('gap_power', 0),
                "EYE": p.get('eye', 0),
                "SPD": p.get('speed', 0),
                "Meta": p['meta'],
                "Perf": perf_str,
            })
        st.dataframe(pd.DataFrame(order_rows), use_container_width=True, hide_index=True,
                     height=min(35 * len(order_rows) + 40, 400),
                     column_config={
                         "#": st.column_config.TextColumn(width="small"),
                         "Player": st.column_config.TextColumn(width="medium"),
                         "CON": st.column_config.NumberColumn(width="small"),
                         "POW": st.column_config.NumberColumn(width="small"),
                         "GAP": st.column_config.NumberColumn(width="small"),
                         "EYE": st.column_config.NumberColumn(width="small"),
                         "SPD": st.column_config.NumberColumn(width="small"),
                         "Meta": st.column_config.ProgressColumn(min_value=300, max_value=800, format="%d", width="small"),
                         "Perf": st.column_config.TextColumn(width="small"),
                     })

    # ── Bench Bats ──
    st.divider()
    st.markdown("##### Bench Bats")
    st.caption("Your 4 reserve batters — pinch hitters, platoon partners, defensive subs")

    # Bench bats = starters NOT in the 9-man lineup
    _bench_bats = []
    for fpos in bat_field_positions:
        for p in active_by_pos.get(fpos, []):
            if p['player_name'] not in _all_lineup_names:
                _bench_bats.append(p)
    _bench_bats.sort(key=lambda p: p['meta_score'] or 0, reverse=True)
    _bench_bats = _bench_bats[:4]  # 26-man roster has ~4 bench bats

    # All rostered player names — no rostered player should be recommended as an upgrade
    _all_rostered_names = set()
    for _pos_players in all_by_pos.values():
        for _rp in _pos_players:
            _all_rostered_names.add(_rp['player_name'])
    _used_bench_upgrades = set()  # track already-recommended upgrades to avoid dupes

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

            # Find best upgrade from collection (not already rostered)
            # Old query used card_title matching against roster, but roster.card_title
            # is NULL for all entries. Instead, fetch top candidates and filter in Python
            # against the known set of all rostered player names.
            # Position-match: a bench SS upgrade should play SS (or a corner IF), not
            # a catcher — so we score by position flexibility first.
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
            _acceptable_positions = _pos_match_map.get(bpos, (bpos,))
            _placeholders = ','.join(['?'] * len(_acceptable_positions))
            _bench_candidates = conn.execute(f"""
                SELECT c.card_id, c.card_title, c.team,
                       c.meta_score_batting as meta, c.position_name, c.card_value
                FROM cards c
                WHERE c.owned = 1 AND c.meta_score_batting > ?
                    AND c.pitcher_role IS NULL
                    AND c.position_name IN ({_placeholders})
                ORDER BY c.meta_score_batting DESC LIMIT 30
            """, (bmeta + 10, *_acceptable_positions)).fetchall()

            bench_upgrade = None
            for _cand in _bench_candidates:
                _cand_title = _cand['card_title'] or ''
                # Skip if this player is on the active roster (name appears in card_title)
                if any(rname in _cand_title for rname in _all_rostered_names):
                    continue
                # Skip if already recommended for another bench slot
                if _cand_title in _used_bench_upgrades:
                    continue
                bench_upgrade = _cand
                _used_bench_upgrades.add(_cand_title)
                break

            action = ""
            if bench_upgrade:
                # Extract just the player name from the card_title for a compact display
                _full_title = bench_upgrade['card_title'] or ''
                up_meta = round(bench_upgrade['meta'])
                delta = up_meta - bmeta
                up_pos = bench_upgrade['position_name'] or '?'
                up_value = bench_upgrade['card_value'] or 0
                # Cleaner short name — drop "MLB YYYY Live POS" prefix if present
                _clean = _full_title
                for _prefix in ['MLB 2026 Live ', 'MLB 2025 Live ', 'MLB 2024 Live ']:
                    if _clean.startswith(_prefix):
                        _clean = _clean[len(_prefix):]
                        # Drop the position code that follows (e.g. "C Carlos Perez HOU")
                        _parts = _clean.split(' ', 1)
                        if _parts and _parts[0] in ('C','1B','2B','3B','SS','LF','CF','RF','DH','SP','RP','CP'):
                            _clean = _parts[1] if len(_parts) > 1 else _clean
                        break
                _clean = short_name(_clean, 28)
                action = f"\U0001f4e6 {_clean} • {up_pos} • {up_meta}m (+{delta}) • {up_value}pp"

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
                         "Meta": st.column_config.ProgressColumn(min_value=300, max_value=800, format="%d", width="small"),
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
                                 "Meta": st.column_config.ProgressColumn(min_value=300, max_value=800, format="%d", width="small"),
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
    st.markdown("##### Rotation")
    sp_rows = build_chain_rows(['SP'], show_perf=True)
    if sp_rows:
        st.dataframe(pd.DataFrame(sp_rows), use_container_width=True, hide_index=True,
                     height=min(35 * len(sp_rows) + 40, 250), column_config=CHAIN_COL_CONFIG)

    st.markdown("##### Bullpen")
    pen_rows = build_chain_rows(['CL', 'SU', 'MID', 'LNG', 'MOP'], show_perf=True)
    if pen_rows:
        st.dataframe(pd.DataFrame(pen_rows), use_container_width=True, hide_index=True,
                     height=min(35 * len(pen_rows) + 40, 370), column_config=CHAIN_COL_CONFIG)

with tab_ai:
    if 'ai_optimize_result' in st.session_state and st.session_state['ai_optimize_result'].get('response'):
        picks_data = st.session_state['ai_optimize_result'].get('picks_data', [])
        if picks_data:
            # Split into changes and keeps
            changes = [p for p in picks_data if p['action'] in ('Promote', 'Buy', 'Platoon')]
            keeps = [p for p in picks_data if p['action'] == 'Keep']

            if changes:
                st.markdown("##### Recommended Moves")
                change_rows = []
                for p in changes:
                    cost_str = f"{p['cost']:,} PP" if p.get('cost') else "FREE"
                    action_display = p['action']
                    if action_display == 'Promote':
                        action_display = '📦 Promote'
                    elif action_display == 'Buy':
                        action_display = '🛒 Buy'
                    elif action_display == 'Platoon':
                        action_display = '🤝 Platoon'

                    card_display = p['card_name']
                    if p.get('platoon_partner'):
                        card_display += f" + {p['platoon_partner']}"

                    change_rows.append({
                        "Pos": p['pos'],
                        "": p.get('emoji', ''),
                        "Action": action_display,
                        "In": card_display,
                        "Out": p.get('replaces', '—'),
                        "Cost": cost_str,
                        "Why": p['reason'],
                    })
                st.dataframe(pd.DataFrame(change_rows), use_container_width=True, hide_index=True,
                             height=min(35 * len(change_rows) + 40, 400),
                             column_config={
                                 "Pos": st.column_config.TextColumn(width="small"),
                                 "": st.column_config.TextColumn(width="small"),
                                 "Action": st.column_config.TextColumn(width="small"),
                                 "In": st.column_config.TextColumn(width="medium"),
                                 "Out": st.column_config.TextColumn(width="medium"),
                                 "Cost": st.column_config.TextColumn(width="small"),
                                 "Why": st.column_config.TextColumn(width="large"),
                             })

            if keeps:
                with st.expander(f"✅ No Change — {len(keeps)} positions", expanded=False):
                    keep_rows = []
                    for p in keeps:
                        keep_rows.append({
                            "Pos": p['pos'],
                            "": p.get('emoji', ''),
                            "Player": p['card_name'],
                            "Reason": p['reason'],
                        })
                    st.dataframe(pd.DataFrame(keep_rows), use_container_width=True, hide_index=True,
                                 column_config={
                                     "Pos": st.column_config.TextColumn(width="small"),
                                     "": st.column_config.TextColumn(width="small"),
                                     "Player": st.column_config.TextColumn(width="medium"),
                                     "Reason": st.column_config.TextColumn(width="large"),
                                 })

            tokens = st.session_state['ai_optimize_result'].get('tokens_used', 0)
            model = st.session_state['ai_optimize_result'].get('model', '?')
            st.caption(f"*{model} • {tokens:,} tokens*")
            with st.expander("Raw AI Analysis", expanded=False):
                st.markdown(st.session_state['ai_optimize_result']['response'])
        else:
            # Fallback: raw text if parsing failed
            st.markdown(st.session_state['ai_optimize_result']['response'])
            tokens = st.session_state['ai_optimize_result'].get('tokens_used', 0)
            model = st.session_state['ai_optimize_result'].get('model', '?')
            st.caption(f"*{model} • {tokens:,} tokens*")
    else:
        st.info("Click **🧠 AI Optimize All** above to generate AI-powered roster recommendations. "
                "The AI evaluates your full collection, in-game performance, platoon splits, and market options.")

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
        for a in ow:
            alt_rows.append({
                "Card": short_name(a.get('card_title', '')),
                "OVR": a.get('card_value', 0), "Meta": round(a.get('meta_score', 0) or 0),
                "+": round((a.get('meta_score', 0) or 0) - u['current_meta']),
                "Source": f"\U0001f4e6 {a.get('action', 'FREE')}",
            })
        for a in mk:
            p = a.get('last_10_price', 0) or 0
            alt_rows.append({
                "Card": short_name(a.get('card_title', '')),
                "OVR": a.get('card_value', 0), "Meta": round(a.get('meta_score', 0) or 0),
                "+": round((a.get('meta_score', 0) or 0) - u['current_meta']),
                "Source": f"\U0001f6d2 {p:,} PP" if p else "\U0001f6d2 Market",
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

conn.close()
