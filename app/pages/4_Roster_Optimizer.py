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

bat_positions = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
pitch_positions = ['SP', 'RP', 'CL']
if focus == "Batting Only":
    show_positions = bat_positions
elif focus == "Pitching Only":
    show_positions = pitch_positions
else:
    show_positions = bat_positions + pitch_positions


# ── Helpers ──
def find_owned_upgrades(pos_value, current_meta, is_pitching, exclude_names=None, limit=5):
    exclude_names = exclude_names or []
    meta_col = "meta_score_pitching" if is_pitching else "meta_score_batting"
    pos_col = "pitcher_role_name" if is_pitching else "position_name"
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
    for r in results:
        title = r['card_title'] or ''
        if any(name in title for name in exclude_names):
            continue
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
    return filtered[:limit]


def find_market_upgrades(pos_value, current_meta, is_pitching, exclude_ids=None, limit=5):
    exclude_ids = exclude_ids or set()
    meta_col = "meta_score_pitching" if is_pitching else "meta_score_batting"
    pos_col = "pitcher_role_name" if is_pitching else "position_name"
    max_price = max_spend if max_spend > 0 else 999999999
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
        used_names = {sp['player_name'] for sp in sp_players}
        # Process WEAKEST first so best free upgrades go to worst slots
        order = sorted(range(len(sp_players)), key=lambda i: sp_players[i]['meta_score'] or 0)
        sp_entries = [None] * len(sp_players)
        for i in order:
            sp = sp_players[i]
            m = sp['meta_score'] or 0
            ow = find_owned_upgrades('SP', m, True, list(used_names), 3)
            mk = find_market_upgrades('SP', m, True, used_market_ids, 3)
            entry = _build_slot(f"SP{i+1}", sp['player_name'], sp['ovr'], m, ow, mk)
            if entry['owned_name']:
                used_names.add(entry['owned_name'])
            sp_entries[i] = entry
        upgrade_plan.extend(sp_entries)
        continue

    if pos == 'RP':
        # Use only actual bullpen pitchers as "current"
        rp_players = active_by_pos.get('RP', [])[:7]
        used_names = {rp['player_name'] for rp in rp_players}
        slot_names = ["SU1", "SU2", "MID1", "MID2", "LNG1", "LNG2", "MOP"]
        # Process WEAKEST first so best free upgrades go to worst slots
        order = sorted(range(len(rp_players)), key=lambda i: rp_players[i]['meta_score'] or 0)
        rp_entries = [None] * len(rp_players)
        for i in order:
            rp = rp_players[i]
            m = rp['meta_score'] or 0
            ow = find_owned_upgrades('RP', m, True, list(used_names), 3)
            mk = find_market_upgrades('RP', m, True, used_market_ids, 3)
            label = slot_names[i] if i < len(slot_names) else f"RP{i+1}"
            entry = _build_slot(label, rp['player_name'], rp['ovr'], m, ow, mk)
            if entry['owned_name']:
                used_names.add(entry['owned_name'])
            rp_entries[i] = entry
        upgrade_plan.extend(rp_entries)
        continue

    # For batting positions: show ALL active-roster players at this position.
    # The game has separate vs-RHP and vs-LHP lineups (platoons), and the
    # roster table can't tell us who's actually starting, so show everyone
    # on the 26-man and let the user identify their lineup vs bench.
    active_players = active_by_pos.get(pos, [])
    if not active_players:
        # No one active at this position — show empty slot
        upgrade_plan.append(_build_slot(pos, '(empty)', 0, 0,
            find_owned_upgrades(pos, 0, is_pitching, [], 5),
            find_market_upgrades(pos, 0, is_pitching, used_market_ids, 5)))
        continue

    # Process WEAKEST first so best free upgrades go to worst slots
    active_names = [p['player_name'] for p in active_players]
    order = sorted(range(len(active_players)), key=lambda i: active_players[i]['meta_score'] or 0)
    bat_entries = [None] * len(active_players)
    for i in order:
        player = active_players[i]
        m = player['meta_score'] or 0
        label = pos if len(active_players) == 1 else f"{pos}{i+1}"
        ow = find_owned_upgrades(pos, m, is_pitching, active_names, 3)
        mk = find_market_upgrades(pos, m, is_pitching, used_market_ids, 3)
        bh = player.get('bats_hand', '?')
        entry = _build_slot(label, player['player_name'], player['ovr'], m, ow, mk, bats_hand=bh)
        if entry['owned_name']:
            active_names.append(entry['owned_name'])
        bat_entries[i] = entry
    upgrade_plan.extend(bat_entries)

if focus == "Weakest First":
    upgrade_plan.sort(key=lambda x: x['current_meta'])

# ── Classify ──
collection_swaps = [u for u in upgrade_plan if u['owned_name']]
market_buys = [u for u in upgrade_plan if u['market_name']]
all_upgrades = [u for u in upgrade_plan if u['owned_name'] or u['market_name']]
top_priorities = sorted(all_upgrades, key=lambda x: -x['best_delta'])[:3]

# ── Roster mismatches ──
roster_fixes = []
for pos in bat_positions + ['CL']:
    pp = all_by_pos.get(pos, [])
    if len(pp) < 2: continue
    best = pp[0]
    if best.get('lineup_role') not in ('starter', 'rotation', 'closer', 'bullpen'):
        for p in pp:
            if p.get('lineup_role') in ('starter', 'rotation', 'closer', 'bullpen'):
                d = round((best['meta_score'] or 0) - (p['meta_score'] or 0))
                if d >= min_improvement:
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
bat_metas = [starters[p]['meta_score'] for p in bat_positions if p in starters]
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
    """Build compact lineup rows with a single unified Recommendation column.

    Cascade priority for the recommendation:
    1. AI pick (if AI has been run) — shows promote, buy, keep, or platoon
    2. Free promote (if you own a better card)
    3. Market buy (if no free option and market has one)
    4. "✅ Optimal" if no upgrades exist
    """
    rows = []
    for u in upgrade_plan:
        if u['pos'] not in positions_list and not any(u['pos'].startswith(p) for p in positions_list):
            continue

        # Current player — compact: "Name (OVR)"
        ovr = u['current_ovr'] or ""
        bh = f" {u.get('bats', '')}" if show_bats and u.get('bats', '?') != '?' else ""
        current_display = f"{u['current_name']} ({ovr}{bh})" if ovr else u['current_name']

        row = {
            "Pos": u['pos'],
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
                flag = "\U0001f340" if era_fip < -0.5 else ("\u26a0\ufe0f" if era_fip > 0.5 else "")
                row["Perf"] = f"{pp['era']:.2f} ERA  {pp['war']:.1f}W {flag}"
            elif pb:
                row["Perf"] = f".{int(pb['ops']*1000):03d} OPS  {pb['war600']:.1f}W/600"
            else:
                row["Perf"] = ""

        # ── Unified Recommendation column ──
        # Priority: meta-based first (promote > buy), then AI overrides ONLY
        # when AI suggests a DIFFERENT/BETTER card. AI "Keep" does NOT wipe
        # out existing meta recommendations — it just means AI didn't find
        # something better, so the meta pick stands.
        ai_pick = _get_ai_pick_for_pos(u['pos'])

        # Start with the meta-based recommendation
        meta_rec = ""
        if u['owned_name']:
            meta_rec = f"📦 {short_name(u['owned_name'], 30)} (+{u['owned_delta']}) FREE"
        elif u['market_name']:
            p = u['market_price'] or 0
            cost = f"{p:,}" if p else "?"
            meta_rec = f"🛒 {short_name(u['market_name'], 30)} (+{u['market_delta']}) {cost}PP"

        # AI overrides only when it has an actionable pick (not Keep)
        if ai_pick and ai_pick['action'] in ('Promote', 'Buy', 'Platoon'):
            emoji = ai_pick.get('emoji', '')
            card = ai_pick['card_name']
            if ai_pick['action'] == 'Promote':
                delta = u['owned_delta'] or '?'
                row["Recommendation"] = f"{emoji} 📦 {short_name(card, 30)} (+{delta}) FREE"
            elif ai_pick['action'] == 'Buy':
                p = ai_pick.get('cost') or (u['market_price'] if u.get('market_price') else 0)
                cost = f"{p:,}" if p else "?"
                row["Recommendation"] = f"{emoji} 🛒 {short_name(card, 30)} {cost}PP"
            elif ai_pick['action'] == 'Platoon':
                partner = ai_pick.get('platoon_partner', '')
                row["Recommendation"] = f"{emoji} 🤝 {short_name(card, 20)}" + (f" + {short_name(partner, 20)}" if partner else "")
        elif meta_rec:
            row["Recommendation"] = meta_rec
        else:
            row["Recommendation"] = "✅ Optimal"

        # Detail column — full card name + AI reason (widen column or hover to see)
        detail_parts = []
        if ai_pick and ai_pick['action'] != 'Keep' and ai_pick.get('reason'):
            detail_parts.append(f"AI: {ai_pick['reason']}")
            if ai_pick.get('replaces'):
                detail_parts.append(f"Replaces: {ai_pick['replaces']}")
        if u['owned_name']:
            detail_parts.append(f"Owned: {u['owned_name']}")
        if u['market_name']:
            detail_parts.append(f"Market: {u['market_name']}")
        row["Detail"] = " | ".join(detail_parts) if detail_parts else ""

        rows.append(row)
    return rows


CHAIN_COL_CONFIG = {
    "Pos": st.column_config.TextColumn(width="small"),
    "Current": st.column_config.TextColumn(width="medium"),
    "Meta": st.column_config.ProgressColumn(min_value=300, max_value=800, format="%d", width="small"),
    "Perf": st.column_config.TextColumn(width="small",
        help="In-game: ERA/OPS + WAR. \U0001f340=lucky (ERA << FIP), \u26a0\ufe0f=unlucky"),
    "Recommendation": st.column_config.TextColumn(width="large",
        help="Best move for this slot. 📦=promote owned card (FREE), 🛒=market buy, ✅=optimal. "
             "AI picks override meta-only recommendations when AI Optimize has been run."),
    "Detail": st.column_config.TextColumn(width="small",
        help="Expand column to see full card name, AI reasoning, and who gets replaced. "
             "Drag column border to widen."),
}

# ── TABBED LAYOUT — full width, no horizontal scrolling ──
tab_bat, tab_pit, tab_ai = st.tabs(["⚾ Batting Lineup", "🎯 Pitching Staff", "🧠 AI Recommendations"])

with tab_bat:
    bat_rows = build_chain_rows(bat_positions, show_bats=True, show_perf=True)
    if bat_rows:
        h = min(35 * len(bat_rows) + 40, 600)
        st.dataframe(pd.DataFrame(bat_rows), use_container_width=True, hide_index=True,
                     height=h, column_config=CHAIN_COL_CONFIG)

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
