"""Buy Recommendations — Budget Investment Advisor.

Helps decide: Should I spend 2000 PP on 1 elite card or 2 good cards?
Which investment path gives the best short-term AND long-term team health?
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection, load_config
from app.utils.sparklines import text_sparkline

st.set_page_config(page_title="Buy Recommendations", page_icon="🛒", layout="wide")

conn = get_connection()
config = load_config()
default_budget = config.get('pp_budget', 500)

# ── Theme colors ──
FIRE = "#FF6B35"
ICE = "#4FC3F7"
GOLD = "#FFD700"
GREEN = "#66BB6A"
PURPLE = "#AB47BC"
GREY = "#9E9E9E"
BG_DARK = "#0E1117"

# ── Set prefix stripping (shared with Roster Optimizer) ──
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


def _strip_set_prefix(card_title: str) -> tuple:
    if not card_title:
        return ("", "")
    t = card_title.strip()
    tag = ""
    changed = True
    while changed:
        changed = False
        for pfx in _SET_PREFIXES:
            if t.startswith(pfx):
                if not tag:
                    tag = pfx.strip().split()[0]
                t = t[len(pfx):]
                changed = True
                break
    if len(t) >= 3 and t[:3] in _POS_TAGS:
        t = t[3:]
    elif len(t) >= 2 and t[:2] == 'C ':
        t = t[2:]
    return (t.strip(), tag)


def short_name(card_title, max_len=30):
    if not card_title:
        return "—"
    core, _ = _strip_set_prefix(card_title)
    if not core:
        core = card_title.strip()
    if len(core) <= max_len:
        return core
    return core[:max_len].rstrip() + "…"


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=30)
def _get_roster_by_pos():
    """Current starters by position with meta + performance data."""
    c = get_connection()
    rows = c.execute("""
        SELECT r.player_name, r.position, r.meta_score, r.ovr, r.lineup_role,
               r.meta_vs_rhp, r.meta_vs_lhp
        FROM roster_current r
        WHERE r.lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
    """).fetchall()
    by_pos = {}
    for r in rows:
        pos = r['position']
        d = dict(r)
        # Add performance data
        if pos in ('SP', 'RP', 'CL'):
            stat = c.execute("""
                SELECT era, fip, war, ip, k_per_9, bb_per_9, whip, era_plus, babip
                FROM pitching_stats WHERE player_name = ?
                ORDER BY snapshot_date DESC LIMIT 1
            """, (r['player_name'],)).fetchone()
            if stat:
                d.update(dict(stat))
                d['era_fip_gap'] = (stat['era'] or 0) - (stat['fip'] or 0)
                # Performance-adjusted meta: blend meta with WAR-based score
                war = stat['war'] or 0
                ip = stat['ip'] or 0
                if ip > 50:
                    war_per_200 = war * 200 / ip
                    perf_score = 400 + war_per_200 * 40  # Scale WAR to meta range
                    d['perf_meta'] = round(d['meta_score'] * 0.6 + perf_score * 0.4)
                else:
                    d['perf_meta'] = d['meta_score']
        else:
            stat = c.execute("""
                SELECT ops, war, avg, obp, slg, iso, babip, pa
                FROM batting_stats WHERE player_name = ?
                ORDER BY snapshot_date DESC LIMIT 1
            """, (r['player_name'],)).fetchone()
            if stat:
                d.update(dict(stat))
                pa = stat['pa'] or 0
                war = stat['war'] or 0
                if pa > 100:
                    war_per_600 = war * 600 / pa
                    perf_score = 400 + war_per_600 * 35
                    d['perf_meta'] = round(d['meta_score'] * 0.6 + perf_score * 0.4)
                else:
                    d['perf_meta'] = d['meta_score']

        if pos not in by_pos or (d.get('meta_score') or 0) > (by_pos[pos].get('meta_score') or 0):
            by_pos[pos] = d
    c.close()
    return by_pos


@st.cache_data(ttl=30)
def _get_position_gaps():
    """Compute gap score for each position: how far below optimal the current starter is."""
    c = get_connection()
    roster = _get_roster_by_pos()

    gaps = []
    batting_pos = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
    pitching_pos = ['SP', 'RP', 'CL']

    for pos in batting_pos:
        current = roster.get(pos)
        current_meta = current['meta_score'] if current else 0
        current_name = current['player_name'] if current else '(empty)'

        # Best available card on market
        best = c.execute("""
            SELECT MAX(meta_score_batting) as best_meta FROM cards
            WHERE position_name = ? AND last_10_price > 0
        """, (pos,)).fetchone()
        market_ceiling = best['best_meta'] if best and best['best_meta'] else current_meta

        # Best owned but not starting
        bench_best = c.execute("""
            SELECT card_title, meta_score_batting as meta, last_10_price
            FROM cards WHERE position_name = ? AND owned = 1
              AND meta_score_batting > ?
            ORDER BY meta_score_batting DESC LIMIT 1
        """, (pos, current_meta + 5)).fetchone()

        gaps.append({
            'pos': pos,
            'type': 'bat',
            'current_name': current_name,
            'current_meta': round(current_meta),
            'perf_meta': round(current.get('perf_meta', current_meta)) if current else 0,
            'market_ceiling': round(market_ceiling or 0),
            'gap': round((market_ceiling or 0) - current_meta),
            'has_owned_upgrade': bench_best is not None,
            'owned_upgrade': short_name(bench_best['card_title']) if bench_best else None,
            'owned_upgrade_meta': round(bench_best['meta']) if bench_best else None,
            'war': round(current.get('war', 0), 1) if current else 0,
            'ops': round(current.get('ops', 0), 3) if current else 0,
        })

    for pos in pitching_pos:
        current = roster.get(pos)
        current_meta = current['meta_score'] if current else 0
        current_name = current['player_name'] if current else '(empty)'

        best = c.execute("""
            SELECT MAX(meta_score_pitching) as best_meta FROM cards
            WHERE pitcher_role_name = ? AND last_10_price > 0
        """, (pos,)).fetchone()
        market_ceiling = best['best_meta'] if best and best['best_meta'] else current_meta

        bench_best = c.execute("""
            SELECT card_title, meta_score_pitching as meta, last_10_price
            FROM cards WHERE pitcher_role_name = ? AND owned = 1
              AND meta_score_pitching > ?
            ORDER BY meta_score_pitching DESC LIMIT 1
        """, (pos, current_meta + 5)).fetchone()

        gaps.append({
            'pos': pos,
            'type': 'pit',
            'current_name': current_name,
            'current_meta': round(current_meta),
            'perf_meta': round(current.get('perf_meta', current_meta)) if current else 0,
            'market_ceiling': round(market_ceiling or 0),
            'gap': round((market_ceiling or 0) - current_meta),
            'has_owned_upgrade': bench_best is not None,
            'owned_upgrade': short_name(bench_best['card_title']) if bench_best else None,
            'owned_upgrade_meta': round(bench_best['meta']) if bench_best else None,
            'war': round(current.get('war', 0), 1) if current else 0,
            'era': round(current.get('era', 0), 2) if current else 0,
        })

    gaps.sort(key=lambda x: -x['gap'])
    c.close()
    return gaps


@st.cache_data(ttl=30)
def _get_upgrade_candidates(budget, pos, is_pitcher=False):
    """Get best upgrade candidates for a position within budget."""
    c = get_connection()
    roster = _get_roster_by_pos()
    current = roster.get(pos)
    current_meta = current['meta_score'] if current else 0

    if is_pitcher:
        rows = c.execute("""
            SELECT card_id, card_title, pitcher_role_name as pos, tier_name, tier,
                   meta_score_pitching as meta_score, last_10_price, sell_order_low,
                   stuff, movement, control, p_hr, p_babip,
                   stuff_vl, stuff_vr, movement_vl, movement_vr, control_vl, control_vr,
                   stamina, velocity
            FROM cards
            WHERE pitcher_role_name = ? AND owned = 0 AND last_10_price > 0
                AND last_10_price <= ? AND meta_score_pitching > ?
            ORDER BY meta_score_pitching DESC LIMIT 20
        """, (pos, budget, current_meta + 5)).fetchall()
    else:
        rows = c.execute("""
            SELECT card_id, card_title, position_name as pos, tier_name, tier,
                   meta_score_batting as meta_score, last_10_price, sell_order_low,
                   contact, gap_power, power, eye, avoid_ks, babip,
                   contact_vl, contact_vr, power_vl, power_vr, eye_vl, eye_vr,
                   speed, stealing, bats
            FROM cards
            WHERE position_name = ? AND owned = 0 AND last_10_price > 0
                AND last_10_price <= ? AND meta_score_batting > ?
            ORDER BY meta_score_batting DESC LIMIT 20
        """, (pos, budget, current_meta + 5)).fetchall()

    results = []
    for card in rows:
        d = dict(card)
        price = d['last_10_price'] or 0
        meta = d['meta_score'] or 0
        delta = meta - current_meta
        d['delta'] = round(delta)
        d['value_ratio'] = round(meta * meta / price, 1) if price > 0 else 0
        d['efficiency'] = round(delta / (price / 100), 1) if price > 0 else 0  # meta gain per 100PP
        d['short_name'] = short_name(d['card_title'])

        # Check if this card has in-game performance data (same name playing elsewhere)
        name_core, _ = _strip_set_prefix(d['card_title'])
        last_name = name_core.split()[1] if len(name_core.split()) > 1 else name_core.split()[0] if name_core.split() else ''
        if is_pitcher and last_name:
            perf = c.execute("""
                SELECT era, fip, war, ip FROM pitching_stats
                WHERE player_name LIKE ? ORDER BY snapshot_date DESC LIMIT 1
            """, (f"%{last_name}%",)).fetchone()
            if perf and (perf['ip'] or 0) > 30:
                d['perf_era'] = perf['era']
                d['perf_fip'] = perf['fip']
                d['perf_war'] = perf['war']
        elif last_name:
            perf = c.execute("""
                SELECT ops, war, pa FROM batting_stats
                WHERE player_name LIKE ? ORDER BY snapshot_date DESC LIMIT 1
            """, (f"%{last_name}%",)).fetchone()
            if perf and (perf['pa'] or 0) > 50:
                d['perf_ops'] = perf['ops']
                d['perf_war'] = perf['war']

        results.append(d)

    c.close()
    return results


@st.cache_data(ttl=30)
def _build_budget_scenarios(total_budget):
    """Build comparison scenarios for budget allocation.

    Computes: what's the best 1-card buy, best 2-card buy, best 3-card buy
    and compares total meta gain across the team.
    """
    c = get_connection()
    gaps = _get_position_gaps()
    roster = _get_roster_by_pos()

    scenarios = []

    # Scenario 1: Best single card (spend it all on biggest gap)
    best_single = None
    best_single_delta = 0
    for g in gaps:
        if g['has_owned_upgrade']:
            continue  # Skip positions with free upgrades
        pos = g['pos']
        is_pit = g['type'] == 'pit'
        current_meta = g['current_meta']

        if is_pit:
            card = c.execute("""
                SELECT card_id, card_title, meta_score_pitching as meta, last_10_price, tier_name
                FROM cards WHERE pitcher_role_name = ? AND owned = 0 AND last_10_price > 0
                    AND last_10_price <= ? AND meta_score_pitching > ?
                ORDER BY meta_score_pitching DESC LIMIT 1
            """, (pos, total_budget, current_meta + 5)).fetchone()
        else:
            card = c.execute("""
                SELECT card_id, card_title, meta_score_batting as meta, last_10_price, tier_name
                FROM cards WHERE position_name = ? AND owned = 0 AND last_10_price > 0
                    AND last_10_price <= ? AND meta_score_batting > ?
                ORDER BY meta_score_batting DESC LIMIT 1
            """, (pos, total_budget, current_meta + 5)).fetchone()

        if card:
            delta = (card['meta'] or 0) - current_meta
            if delta > best_single_delta:
                best_single_delta = delta
                best_single = {
                    'pos': pos,
                    'card': short_name(card['card_title']),
                    'card_title': card['card_title'],
                    'meta': round(card['meta']),
                    'price': card['last_10_price'],
                    'delta': round(delta),
                    'tier': card['tier_name'],
                    'replaces': g['current_name'],
                    'card_id': card['card_id'],
                }

    if best_single:
        scenarios.append({
            'label': '🎯 One Big Buy',
            'description': f"Spend {best_single['price']:,} PP on a single elite upgrade",
            'cards': [best_single],
            'total_cost': best_single['price'],
            'total_delta': best_single['delta'],
            'remaining_pp': total_budget - best_single['price'],
            'positions_improved': 1,
        })

    # Scenario 2: Best two cards (half budget each)
    half_budget = total_budget // 2
    best_pair = []
    used_positions = set()

    for g in gaps:
        if len(best_pair) >= 2:
            break
        if g['has_owned_upgrade']:
            continue
        pos = g['pos']
        if pos in used_positions:
            continue
        is_pit = g['type'] == 'pit'
        current_meta = g['current_meta']

        if is_pit:
            card = c.execute("""
                SELECT card_id, card_title, meta_score_pitching as meta, last_10_price, tier_name
                FROM cards WHERE pitcher_role_name = ? AND owned = 0 AND last_10_price > 0
                    AND last_10_price <= ? AND meta_score_pitching > ?
                ORDER BY meta_score_pitching DESC LIMIT 1
            """, (pos, half_budget, current_meta + 5)).fetchone()
        else:
            card = c.execute("""
                SELECT card_id, card_title, meta_score_batting as meta, last_10_price, tier_name
                FROM cards WHERE position_name = ? AND owned = 0 AND last_10_price > 0
                    AND last_10_price <= ? AND meta_score_batting > ?
                ORDER BY meta_score_batting DESC LIMIT 1
            """, (pos, half_budget, current_meta + 5)).fetchone()

        if card:
            delta = (card['meta'] or 0) - current_meta
            if delta > 5:
                best_pair.append({
                    'pos': pos,
                    'card': short_name(card['card_title']),
                    'card_title': card['card_title'],
                    'meta': round(card['meta']),
                    'price': card['last_10_price'],
                    'delta': round(delta),
                    'tier': card['tier_name'],
                    'replaces': g['current_name'],
                    'card_id': card['card_id'],
                })
                used_positions.add(pos)

    if len(best_pair) == 2:
        total_cost = sum(c['price'] for c in best_pair)
        total_delta = sum(c['delta'] for c in best_pair)
        scenarios.append({
            'label': '⚖️ Two Balanced Buys',
            'description': f"Spread across 2 weak spots ({half_budget:,} PP each)",
            'cards': best_pair,
            'total_cost': total_cost,
            'total_delta': total_delta,
            'remaining_pp': total_budget - total_cost,
            'positions_improved': 2,
        })

    # Scenario 3: Best efficiency — maximize meta gain per PP
    third_budget = total_budget // 3
    efficiency_picks = []
    used_pos_3 = set()

    # Collect all candidates across positions, sort by efficiency
    all_efficient = []
    for g in gaps:
        if g['has_owned_upgrade']:
            continue
        pos = g['pos']
        is_pit = g['type'] == 'pit'
        current_meta = g['current_meta']

        if is_pit:
            cards = c.execute("""
                SELECT card_id, card_title, meta_score_pitching as meta, last_10_price, tier_name
                FROM cards WHERE pitcher_role_name = ? AND owned = 0 AND last_10_price > 0
                    AND last_10_price <= ? AND meta_score_pitching > ?
                ORDER BY (meta_score_pitching - ?) * 1.0 / last_10_price DESC LIMIT 3
            """, (pos, total_budget, current_meta + 5, current_meta)).fetchall()
        else:
            cards = c.execute("""
                SELECT card_id, card_title, meta_score_batting as meta, last_10_price, tier_name
                FROM cards WHERE position_name = ? AND owned = 0 AND last_10_price > 0
                    AND last_10_price <= ? AND meta_score_batting > ?
                ORDER BY (meta_score_batting - ?) * 1.0 / last_10_price DESC LIMIT 3
            """, (pos, total_budget, current_meta + 5, current_meta)).fetchall()

        for card in cards:
            delta = (card['meta'] or 0) - current_meta
            price = card['last_10_price'] or 1
            all_efficient.append({
                'pos': pos,
                'card': short_name(card['card_title']),
                'card_title': card['card_title'],
                'meta': round(card['meta']),
                'price': price,
                'delta': round(delta),
                'efficiency': round(delta / (price / 100), 1),
                'tier': card['tier_name'],
                'replaces': g['current_name'],
                'card_id': card['card_id'],
            })

    all_efficient.sort(key=lambda x: -x['efficiency'])

    # Greedy pack: pick best efficiency cards that fit within budget
    remaining = total_budget
    for cand in all_efficient:
        if cand['pos'] in used_pos_3:
            continue
        if cand['price'] <= remaining:
            efficiency_picks.append(cand)
            used_pos_3.add(cand['pos'])
            remaining -= cand['price']
            if len(efficiency_picks) >= 3:
                break

    if len(efficiency_picks) >= 2:
        total_cost = sum(c['price'] for c in efficiency_picks)
        total_delta = sum(c['delta'] for c in efficiency_picks)
        scenarios.append({
            'label': '📈 Max Efficiency',
            'description': f"Best bang-for-buck: {len(efficiency_picks)} cards, max team meta gain",
            'cards': efficiency_picks,
            'total_cost': total_cost,
            'total_delta': total_delta,
            'remaining_pp': total_budget - total_cost,
            'positions_improved': len(efficiency_picks),
        })

    c.close()
    return scenarios


@st.cache_data(ttl=30)
def _get_performance_outliers():
    """Find players whose in-game performance doesn't match their meta score.

    These are 'hidden gem' or 'sell high' candidates.
    """
    c = get_connection()

    # Pitchers outperforming their meta (like Pfaadt)
    outperforming = [dict(r) for r in c.execute("""
        SELECT p.player_name, p.era, p.fip, p.war, p.ip, p.era_plus, p.whip,
               r.meta_score, r.position, r.lineup_role
        FROM pitching_stats p
        JOIN roster r ON r.player_name = p.player_name
            AND DATE(r.snapshot_date) = (SELECT MAX(DATE(snapshot_date)) FROM roster WHERE lineup_role != 'league')
        WHERE p.ip > 40 AND p.war > 1.5
            AND DATE(p.snapshot_date) = (SELECT MAX(DATE(snapshot_date)) FROM pitching_stats)
        ORDER BY p.war DESC
    """).fetchall()]

    bat_outperforming = [dict(r) for r in c.execute("""
        SELECT b.player_name, b.ops, b.war, b.pa, b.avg, b.obp, b.slg, b.iso,
               r.meta_score, r.position, r.lineup_role
        FROM batting_stats b
        JOIN roster r ON r.player_name = b.player_name
            AND DATE(r.snapshot_date) = (SELECT MAX(DATE(snapshot_date)) FROM roster WHERE lineup_role != 'league')
        WHERE b.pa > 80 AND b.war > 1.0
            AND DATE(b.snapshot_date) = (SELECT MAX(DATE(snapshot_date)) FROM batting_stats)
        ORDER BY b.war DESC
    """).fetchall()]

    c.close()
    return outperforming, bat_outperforming


# ============================================================
# HEADER + CONTROLS
# ============================================================
st.title("🛒 Investment Advisor")

# Budget control
sell_pp = conn.execute(
    "SELECT COALESCE(SUM(estimated_price), 0) FROM recommendations WHERE rec_type='sell' AND dismissed=0"
).fetchone()[0] or 0

max_market_price = conn.execute(
    "SELECT MAX(last_10_price) FROM cards WHERE owned = 0 AND last_10_price > 0"
).fetchone()[0] or 10000

effective_budget = default_budget + sell_pp

col_b1, col_b2 = st.columns([3, 1])
with col_b1:
    pp_budget = st.slider(
        "💰 Total PP Budget",
        min_value=0,
        max_value=min(max_market_price, 500000),
        value=min(default_budget, min(max_market_price, 500000)),
        step=100,
        format="%d PP",
    )
with col_b2:
    if sell_pp > 0:
        st.metric("Effective Budget", f"{pp_budget + sell_pp:,} PP",
                  delta=f"+{sell_pp:,} from sells")
    else:
        st.metric("Budget", f"{pp_budget:,} PP")

# ============================================================
# TAB LAYOUT
# ============================================================
tab_invest, tab_gaps, tab_browse, tab_perf, tab_ai = st.tabs([
    "💡 Investment Plan", "🔍 Position Gaps", "🏪 Market Browser",
    "📊 Performance vs Meta", "🤖 AI Advisor"
])

# ============================================================
# TAB: Investment Plan (THE core feature)
# ============================================================
with tab_invest:
    # Free upgrades first
    gaps = _get_position_gaps()
    free_upgrades = [g for g in gaps if g['has_owned_upgrade']]

    if free_upgrades:
        st.subheader("🆓 Free Upgrades — Promote From Your Collection")
        st.caption("These cost nothing — just swap from your bench to the lineup")
        free_data = []
        for g in free_upgrades:
            free_data.append({
                "Pos": g['pos'],
                "Current": g['current_name'],
                "Meta": g['current_meta'],
                "➡️ Promote": g['owned_upgrade'],
                "New Meta": g['owned_upgrade_meta'],
                "+Gain": g['owned_upgrade_meta'] - g['current_meta'],
            })
        st.dataframe(
            pd.DataFrame(free_data),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Meta": st.column_config.ProgressColumn(min_value=200, max_value=800, format="%d"),
                "New Meta": st.column_config.ProgressColumn(min_value=200, max_value=800, format="%d"),
                "+Gain": st.column_config.NumberColumn(format="+%d"),
            }
        )
        total_free_gain = sum(g['owned_upgrade_meta'] - g['current_meta'] for g in free_upgrades)
        st.success(f"**+{total_free_gain} total meta** from {len(free_upgrades)} free moves. Do these first!")
        st.divider()

    # Budget scenarios
    st.subheader("💰 How Should You Spend Your PP?")

    scenarios = _build_budget_scenarios(pp_budget)

    if not scenarios:
        if pp_budget == 0:
            st.info("Set your budget above to see investment options.")
        else:
            st.info(f"No meaningful upgrades found at {pp_budget:,} PP. Try increasing your budget.")
    else:
        # Scenario comparison cards
        cols = st.columns(len(scenarios))
        best_scenario = max(scenarios, key=lambda s: s['total_delta'])

        for i, scenario in enumerate(scenarios):
            with cols[i]:
                is_best = scenario == best_scenario
                border = f"border: 2px solid {GOLD};" if is_best else "border: 1px solid #333;"
                badge = " 👑" if is_best else ""

                st.markdown(f"""
                <div style="background: #1a1a2e; {border} border-radius: 10px; padding: 15px; text-align: center;">
                    <h3 style="margin: 0; color: {'#FFD700' if is_best else '#fff'};">{scenario['label']}{badge}</h3>
                    <p style="color: #aaa; font-size: 0.85em; margin: 5px 0;">{scenario['description']}</p>
                    <hr style="border-color: #333;">
                    <div style="font-size: 2em; font-weight: bold; color: {'#66BB6A' if is_best else '#4FC3F7'};">
                        +{scenario['total_delta']}
                    </div>
                    <p style="color: #aaa; margin: 0;">Total Meta Gain</p>
                    <p style="color: #ccc; margin-top: 8px;">
                        <strong>{scenario['total_cost']:,} PP</strong> • {scenario['positions_improved']} position{'s' if scenario['positions_improved'] > 1 else ''}
                    </p>
                    <p style="color: #888; font-size: 0.8em;">
                        {scenario['remaining_pp']:,} PP remaining
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Detailed breakdown for each scenario
        for scenario in scenarios:
            with st.expander(f"{scenario['label']} — Details", expanded=(scenario == best_scenario)):
                for card in scenario['cards']:
                    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                    with c1:
                        st.markdown(f"**{card['pos']}**: {card['card']}  \n"
                                   f"*{card.get('tier', '')}* — replaces {card['replaces']}")
                    with c2:
                        st.metric("Meta", card['meta'], delta=f"+{card['delta']}")
                    with c3:
                        st.metric("Cost", f"{card['price']:,} PP")
                    with c4:
                        trend = text_sparkline(card.get('card_id'), conn) if card.get('card_id') else ''
                        if trend:
                            st.markdown(f"**Trend** {trend}")
                        eff = card.get('efficiency')
                        if eff:
                            st.metric("Eff", f"{eff} meta/100PP")

        # Visual: meta gain per PP spent comparison
        if len(scenarios) >= 2:
            st.divider()
            st.subheader("📊 Scenario Comparison")

            fig = go.Figure()
            colors = [FIRE, ICE, GREEN, PURPLE]

            for i, s in enumerate(scenarios):
                fig.add_trace(go.Bar(
                    name=s['label'],
                    x=['Total Meta Gain', 'Positions Fixed', 'PP Remaining'],
                    y=[s['total_delta'], s['positions_improved'] * 50, s['remaining_pp'] / 50],
                    marker_color=colors[i % len(colors)],
                    text=[f"+{s['total_delta']}", str(s['positions_improved']),
                          f"{s['remaining_pp']:,}"],
                    textposition='auto',
                ))

            fig.update_layout(
                barmode='group',
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=True,
                legend=dict(orientation="h", y=1.15),
                yaxis=dict(visible=False),
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB: Position Gaps
# ============================================================
with tab_gaps:
    st.subheader("🔍 Position Strength Map")
    st.caption("Sorted by biggest gap between your starter and the best available card on market")

    gaps = _get_position_gaps()

    # Visual gap chart
    gap_df = pd.DataFrame(gaps)
    if not gap_df.empty:
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=gap_df['pos'],
            x=gap_df['current_meta'],
            orientation='h',
            name='Current Meta',
            marker_color=ICE,
            text=[f"{m}" for m in gap_df['current_meta']],
            textposition='inside',
        ))

        fig.add_trace(go.Bar(
            y=gap_df['pos'],
            x=gap_df['gap'],
            orientation='h',
            name='Gap to Best Available',
            marker_color=[FIRE if g > 100 else GOLD if g > 50 else GREY for g in gap_df['gap']],
            text=[f"+{g}" for g in gap_df['gap']],
            textposition='inside',
        ))

        fig.update_layout(
            barmode='stack',
            height=max(350, len(gaps) * 35),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            legend=dict(orientation="h", y=1.1),
            xaxis_title="Meta Score",
            yaxis=dict(autorange='reversed'),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Gap table
        gap_table = []
        for g in gaps:
            row = {
                "Pos": g['pos'],
                "Current": g['current_name'],
                "Meta": g['current_meta'],
                "Gap": g['gap'],
            }
            if g['type'] == 'bat':
                row["OPS"] = f".{int(g.get('ops', 0) * 1000):03d}" if g.get('ops') else "—"
                row["WAR"] = g.get('war', 0)
            else:
                row["ERA"] = f"{g.get('era', 0):.2f}" if g.get('era') else "—"
                row["WAR"] = g.get('war', 0)

            if g['has_owned_upgrade']:
                row["Action"] = f"🆓 Promote {g['owned_upgrade']} (+{g['owned_upgrade_meta'] - g['current_meta']})"
            elif g['gap'] > 50:
                row["Action"] = f"🛒 Buy upgrade (+{g['gap']} available)"
            else:
                row["Action"] = "✅ Strong"

            gap_table.append(row)

        st.dataframe(pd.DataFrame(gap_table), use_container_width=True, hide_index=True)

# ============================================================
# TAB: Browse Market
# ============================================================
with tab_browse:
    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        positions = ['All', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'SP', 'RP', 'CL']
        pos_filter = st.selectbox("Position", positions, key="browse_pos")
    with col2:
        tiers = {'All': 0, 'Bronze+': 2, 'Silver+': 3, 'Gold+': 4, 'Diamond+': 5, 'Perfect': 6}
        tier_filter = st.selectbox("Min Tier", list(tiers.keys()), key="browse_tier")
    with col3:
        sort_options = ["Best Value", "Biggest Upgrade", "Cheapest First", "Best Efficiency"]
        sort_by = st.selectbox("Sort By", sort_options, key="browse_sort")

    roster = _get_roster_by_pos()

    market_query = """
        SELECT c.card_id, c.card_title,
               COALESCE(c.pitcher_role_name, c.position_name) as pos,
               c.tier_name, c.tier,
               COALESCE(c.meta_score_batting, c.meta_score_pitching) as meta_score,
               c.last_10_price, c.sell_order_low, c.buy_order_high
        FROM cards c
        WHERE c.owned = 0 AND c.last_10_price > 0
            AND c.last_10_price <= ?
            AND COALESCE(c.meta_score_batting, c.meta_score_pitching) > 0
    """
    market_params = [pp_budget if pp_budget > 0 else 999999999]

    if pos_filter != 'All':
        market_query += " AND (c.position_name = ? OR c.pitcher_role_name = ?)"
        market_params.extend([pos_filter, pos_filter])
    if tiers[tier_filter] > 0:
        market_query += " AND c.tier >= ?"
        market_params.append(tiers[tier_filter])

    sort_map = {
        "Best Value": "meta_score * meta_score * 1.0 / c.last_10_price DESC",
        "Biggest Upgrade": "meta_score DESC",
        "Cheapest First": "c.last_10_price ASC",
        "Best Efficiency": "meta_score * 1.0 / c.last_10_price DESC",
    }
    market_query += f" ORDER BY {sort_map[sort_by]} LIMIT 150"
    market_rows = conn.execute(market_query, market_params).fetchall()

    if market_rows:
        st.caption(f"{len(market_rows)} cards within {pp_budget:,} PP budget")
        data = []
        for r in market_rows:
            meta = r['meta_score'] or 0
            price = r['last_10_price'] or 0
            pos = r['pos'] or ''
            current = roster.get(pos, {})
            current_meta = current.get('meta_score', 0) if current else 0
            delta = meta - current_meta

            data.append({
                "Card": short_name(r['card_title']),
                "Pos": pos,
                "Tier": r['tier_name'] or '',
                "Trend": text_sparkline(r['card_id'], conn) if r['card_id'] else '',
                "+Meta": round(delta),
                "Meta": round(meta),
                "Price": price,
                "Eff": round(delta / (price / 100), 1) if price > 0 and delta > 0 else 0,
            })

        st.dataframe(
            pd.DataFrame(data),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Meta": st.column_config.ProgressColumn(min_value=200, max_value=800, format="%d"),
                "Price": st.column_config.NumberColumn(format="%d PP"),
                "+Meta": st.column_config.NumberColumn(format="+%d"),
                "Eff": st.column_config.NumberColumn(format="%.1f", help="Meta gain per 100 PP spent"),
            }
        )

        # Scatter: Meta vs Price with budget line
        st.divider()
        viz_data = []
        for r in market_rows:
            meta = r['meta_score'] or 0
            price = r['last_10_price'] or 0
            pos = r['pos'] or ''
            current = roster.get(pos, {})
            current_meta = current.get('meta_score', 0) if current else 0
            delta = meta - current_meta
            vr = (meta * meta / price) if price > 0 else 0
            viz_data.append({
                "Card": short_name(r['card_title']),
                "Position": pos,
                "Tier": r['tier_name'] or '',
                "Meta": meta,
                "Price": price,
                "+Meta": round(delta),
                "Value": round(vr, 1),
                "size": max(min(vr / 20, 15), 3),
            })

        viz_df = pd.DataFrame(viz_data)
        fig = px.scatter(
            viz_df, x="Price", y="Meta",
            color="Position" if pos_filter == 'All' else "Tier",
            size="size",
            hover_name="Card",
            hover_data={"Position": True, "Meta": ":.0f", "Price": ":,.0f",
                       "+Meta": True, "Value": ":.1f", "size": False},
        )

        # Add current starter line for specific position
        if pos_filter != 'All' and pos_filter in roster:
            fig.add_hline(
                y=roster[pos_filter].get('meta_score', 0),
                line_dash="dash", line_color="red", line_width=2,
                annotation_text=f"Your {pos_filter}: {roster[pos_filter].get('player_name', '')}",
                annotation_position="top left",
            )

        fig.add_vline(
            x=pp_budget, line_dash="dot", line_color=GOLD, line_width=1,
            annotation_text=f"Budget: {pp_budget:,}",
        )

        fig.update_layout(
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_title="Price (PP)",
            yaxis_title="Meta Score",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cards found. Try increasing the budget slider.")

# ============================================================
# TAB: Performance vs Meta
# ============================================================
with tab_perf:
    st.subheader("📊 Performance-Adjusted Analysis")
    st.caption("Players whose in-game results DON'T match their meta — hidden gems and overrated cards")

    pit_outliers, bat_outliers = _get_performance_outliers()

    # Pitching performance analysis
    if pit_outliers:
        st.markdown("### ⚾ Pitching: Who's Really Delivering?")

        pit_data = []
        for p in pit_outliers:
            meta = p['meta_score'] or 0
            war = p['war'] or 0
            ip = p['ip'] or 0
            era = p['era'] or 0
            fip = p['fip'] or 0
            era_fip = era - fip

            # Performance-adjusted meta
            if ip > 50:
                war_200 = war * 200 / ip
                perf_meta = round(400 + war_200 * 40)
            else:
                perf_meta = round(meta)

            gap = perf_meta - round(meta)
            role = p['lineup_role']

            if gap > 30 and role != 'rotation' and role != 'closer':
                status = "⚠️ BENCHED ACE"
            elif gap > 30:
                status = "🔥 Outperforming"
            elif gap < -30:
                status = "📉 Overrated"
            elif abs(era_fip) > 0.5:
                status = "🍀 Lucky" if era_fip < 0 else "😤 Unlucky"
            else:
                status = "✅ Fair"

            pit_data.append({
                "Player": p['player_name'],
                "Role": role,
                "ERA": round(era, 2),
                "FIP": round(fip, 2),
                "Gap": round(era_fip, 2),
                "WAR": round(war, 1),
                "IP": round(ip),
                "Meta": round(meta),
                "Perf Meta": perf_meta,
                "Δ": gap,
                "Status": status,
            })

        pit_data.sort(key=lambda x: -abs(x['Δ']))

        st.dataframe(
            pd.DataFrame(pit_data),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Meta": st.column_config.ProgressColumn(min_value=200, max_value=800, format="%d"),
                "Perf Meta": st.column_config.ProgressColumn(min_value=200, max_value=800, format="%d"),
                "Δ": st.column_config.NumberColumn(format="%+d", help="Performance meta - card meta"),
                "Gap": st.column_config.NumberColumn(format="%.2f", help="ERA - FIP (negative = lucky)"),
            }
        )

        # Callout for benched aces
        benched_aces = [p for p in pit_data if "BENCHED" in p.get('Status', '')]
        if benched_aces:
            for ace in benched_aces:
                st.warning(
                    f"**{ace['Player']}** has {ace['WAR']} WAR / {ace['ERA']} ERA in {ace['IP']} IP "
                    f"but is only a *{ace['Role']}*. Performance meta ({ace['Perf Meta']}) is "
                    f"**+{ace['Δ']}** above card meta ({ace['Meta']}). Consider starting them!"
                )

    # Batting performance analysis
    if bat_outliers:
        st.markdown("### 🏏 Batting: Who's Really Delivering?")

        bat_data = []
        for b in bat_outliers:
            meta = b['meta_score'] or 0
            war = b['war'] or 0
            pa = b['pa'] or 0
            ops = b['ops'] or 0

            if pa > 100:
                war_600 = war * 600 / pa
                perf_meta = round(400 + war_600 * 35)
            else:
                perf_meta = round(meta)

            gap = perf_meta - round(meta)
            role = b['lineup_role']

            if gap > 30 and role not in ('starter',):
                status = "⚠️ BENCHED STAR"
            elif gap > 30:
                status = "🔥 Outperforming"
            elif gap < -30:
                status = "📉 Overrated"
            else:
                status = "✅ Fair"

            bat_data.append({
                "Player": b['player_name'],
                "Pos": b['position'],
                "Role": role,
                "OPS": round(ops, 3),
                "WAR": round(war, 1),
                "PA": pa,
                "Meta": round(meta),
                "Perf Meta": perf_meta,
                "Δ": gap,
                "Status": status,
            })

        bat_data.sort(key=lambda x: -abs(x['Δ']))

        st.dataframe(
            pd.DataFrame(bat_data),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Meta": st.column_config.ProgressColumn(min_value=200, max_value=800, format="%d"),
                "Perf Meta": st.column_config.ProgressColumn(min_value=200, max_value=800, format="%d"),
                "Δ": st.column_config.NumberColumn(format="%+d"),
            }
        )

    if not pit_outliers and not bat_outliers:
        st.info("No performance data available yet. Import game stats first.")

# ============================================================
# TAB: AI Advisor
# ============================================================
with tab_ai:
    st.subheader("🤖 AI Investment Advisor")

    try:
        from app.core.ai_advisor import get_ai_config, ask_advisor, get_market_analysis
        ai_config = get_ai_config()

        if not ai_config["ready"]:
            st.warning(f"AI Advisor not configured: {ai_config['message']}")
        else:
            st.info(f"Connected: {ai_config['message']}")

            # Pre-built questions with budget context
            ai_col1, ai_col2, ai_col3 = st.columns(3)
            with ai_col1:
                if st.button("🎯 Best single buy?", type="primary", use_container_width=True):
                    with st.spinner("Analyzing your roster + market..."):
                        result = ask_advisor(
                            f"My budget is {pp_budget:,} PP. What is the single best card I can buy "
                            f"to improve my team the most? Consider my weakest positions, platoon splits, "
                            f"and in-game performance. Be specific with card names and prices.",
                            conn
                        )
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        st.markdown(result["response"])

            with ai_col2:
                if st.button("⚖️ 1 big vs 2 small?", use_container_width=True):
                    with st.spinner("Comparing investment strategies..."):
                        result = ask_advisor(
                            f"I have {pp_budget:,} PP. Should I buy 1 expensive card or 2 cheaper cards? "
                            f"Compare the options — which gives better short-term improvement vs long-term "
                            f"team health? Consider which positions have the biggest gaps.",
                            conn
                        )
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        st.markdown(result["response"])

            with ai_col3:
                if st.button("📈 Long-term plan?", use_container_width=True):
                    with st.spinner("Building investment roadmap..."):
                        result = ask_advisor(
                            f"I have {pp_budget:,} PP now. Build me a 3-step investment plan: "
                            f"what to buy now, what to save for next, and what my endgame targets are "
                            f"at each weak position. Include price targets and priority order.",
                            conn
                        )
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        st.markdown(result["response"])

            custom_q = st.text_input("Ask a custom question",
                                      placeholder="e.g., Best SS under 5K PP? Should I buy Pfaadt or Anderson?")
            if custom_q:
                with st.spinner("Analyzing..."):
                    result = ask_advisor(custom_q, conn)
                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.markdown(result["response"])
    except ImportError:
        st.warning("AI Advisor not available.")

conn.close()
