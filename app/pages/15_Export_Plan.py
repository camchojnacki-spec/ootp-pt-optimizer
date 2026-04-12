"""Export Roster Plan — generate a downloadable action plan."""
import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.database import get_connection, load_config

st.set_page_config(page_title="Export Roster Plan", page_icon="\U0001f4e6", layout="wide")
st.title("Export Roster Plan")

conn = get_connection()
config = load_config()
budget = config.get('pp_budget', 500)
team_name = config.get('team_name', 'My Team')

# ============================================================
# 1. CURRENT ROSTER SUMMARY
# ============================================================
st.subheader("Current Roster Summary")

roster = conn.execute("""
    SELECT player_name, position, lineup_role, ovr, meta_score
    FROM roster_current
    ORDER BY
        CASE lineup_role
            WHEN 'starter' THEN 1 WHEN 'rotation' THEN 2
            WHEN 'closer' THEN 3 WHEN 'bullpen' THEN 4
            WHEN 'bench' THEN 5 WHEN 'reserve' THEN 6
        END, meta_score DESC
""").fetchall()

if roster:
    starters = [r for r in roster if r['lineup_role'] in ('starter', 'rotation', 'closer', 'bullpen')]
    bench = [r for r in roster if r['lineup_role'] in ('bench', 'reserve')]

    total_meta = sum(r['meta_score'] or 0 for r in starters)

    rm1, rm2, rm3, rm4 = st.columns(4)
    with rm1:
        st.metric("Active Starters", len(starters))
    with rm2:
        st.metric("Bench/Reserve", len(bench))
    with rm3:
        st.metric("Total Starter Meta", f"{total_meta:,.0f}")
    with rm4:
        st.metric("PP Budget", f"{budget:,}")

    # Roster table
    roster_data = []
    for r in roster:
        roster_data.append({
            "Player": r['player_name'],
            "Pos": r['position'],
            "Role": r['lineup_role'],
            "OVR": r['ovr'] or 0,
            "Meta": round(r['meta_score'], 0) if r['meta_score'] else 0,
        })
    with st.expander("Full Roster", expanded=False):
        st.dataframe(pd.DataFrame(roster_data), use_container_width=True, hide_index=True)
else:
    st.info("No roster data found. Import your roster first.")

st.divider()

# ============================================================
# 2. SELL RECOMMENDATIONS
# ============================================================
st.subheader("Sell Recommendations")

sells = conn.execute("""
    SELECT r.card_id, r.card_title, r.position, r.reason, r.priority,
           r.estimated_price, r.meta_score, r.roster_impact,
           c.tier_name, c.buy_order_high, c.sell_order_low
    FROM recommendations r
    LEFT JOIN cards c ON r.card_id = c.card_id
    WHERE r.rec_type = 'sell' AND r.dismissed = 0
    ORDER BY r.priority ASC, r.estimated_price DESC
""").fetchall()

total_sell_pp = sum(r['estimated_price'] or 0 for r in sells)

if sells:
    # Categorize
    duplicates = [r for r in sells if 'duplicate' in (r['reason'] or '').lower()]
    off_roster = [r for r in sells if r not in duplicates and 'not on active' in (r['reason'] or '').lower()]
    other_sells = [r for r in sells if r not in duplicates and r not in off_roster]

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Total Sell Value", f"{total_sell_pp:,} PP")
    with sc2:
        st.metric("Duplicates", f"{len(duplicates)} ({sum(r['estimated_price'] or 0 for r in duplicates):,} PP)")
    with sc3:
        st.metric("Off-Roster / Other", f"{len(off_roster) + len(other_sells)} cards")

    sell_data = []
    for r in sells:
        sell_data.append({
            "Card": r['card_title'] or '',
            "Pos": r['position'] or '',
            "Tier": r['tier_name'] or '',
            "Est. Price": r['estimated_price'] or 0,
            "Reason": r['reason'] or '',
        })
    st.dataframe(pd.DataFrame(sell_data), use_container_width=True, hide_index=True)
else:
    st.info("No sell recommendations.")

st.divider()

# ============================================================
# 3. BUY RECOMMENDATIONS
# ============================================================
st.subheader("Buy Recommendations (Prioritized)")

buys = conn.execute("""
    SELECT r.card_id, r.card_title, r.position, r.reason, r.priority,
           r.estimated_price, r.meta_score, r.value_ratio, r.roster_impact,
           c.tier_name
    FROM recommendations r
    LEFT JOIN cards c ON r.card_id = c.card_id
    WHERE r.rec_type = 'buy' AND r.dismissed = 0
    ORDER BY r.priority ASC, r.value_ratio DESC
""").fetchall()

total_buy_pp = sum(r['estimated_price'] or 0 for r in buys)

if buys:
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.metric("Total Buy Cost (All)", f"{total_buy_pp:,} PP")
    with bc2:
        st.metric("Buy Recommendations", len(buys))
    with bc3:
        affordable = [b for b in buys if (b['estimated_price'] or 0) <= budget + total_sell_pp]
        st.metric("Affordable (after sells)", len(affordable))

    priority_labels = {1: "Urgent", 2: "High", 3: "Medium", 4: "Low"}
    buy_data = []
    for r in buys:
        buy_data.append({
            "Priority": priority_labels.get(r['priority'], str(r['priority'])),
            "Card": r['card_title'] or '',
            "Pos": r['position'] or '',
            "Tier": r['tier_name'] or '',
            "Meta": round(r['meta_score'], 0) if r['meta_score'] else 0,
            "Price": r['estimated_price'] or 0,
            "Value": round(r['value_ratio'], 1) if r['value_ratio'] else 0,
            "Reason": r['reason'] or '',
        })
    st.dataframe(pd.DataFrame(buy_data), use_container_width=True, hide_index=True)
else:
    st.info("No buy recommendations.")

st.divider()

# ============================================================
# 4. BUDGET OPTIMIZER RESULTS
# ============================================================
st.subheader("Budget Optimizer Preview")

try:
    from app.core.optimizer import optimize_budget, get_roster_meta_total
    effective_budget = budget + total_sell_pp

    opt_col1, opt_col2 = st.columns([3, 1])
    with opt_col1:
        st.write(f"Optimizing with **{effective_budget:,} PP** (budget {budget:,} + sell revenue {total_sell_pp:,})")
    with opt_col2:
        run_optimizer = st.button("Run Optimizer", type="primary")

    if run_optimizer:
        with st.spinner("Optimizing..."):
            result = optimize_budget(effective_budget, conn)

        roster_before = get_roster_meta_total(conn)
        roster_after = roster_before + result['total_meta_gain']

        om1, om2, om3, om4 = st.columns(4)
        with om1:
            st.metric("Meta Gain", f"+{result['total_meta_gain']:.1f}")
        with om2:
            st.metric("Total Cost", f"{result['total_cost']:,} PP")
        with om3:
            st.metric("Remaining PP", f"{result['remaining_budget']:,} PP")
        with om4:
            st.metric("Roster Meta After", f"{roster_after:.0f}")

        if result['transactions']:
            opt_data = []
            for t in result['transactions']:
                opt_data.append({
                    "Position": t['position'],
                    "Current": f"{t['current_player']} ({t['current_meta']:.0f})",
                    "Upgrade To": f"{t['card_title']} ({t['new_meta']:.0f})",
                    "Meta Gain": f"+{t['meta_gain']:.1f}",
                    "Cost": f"{t['price']:,} PP",
                })
            st.dataframe(pd.DataFrame(opt_data), use_container_width=True, hide_index=True)

            # Store in session state for export
            st.session_state['optimizer_result'] = result
            st.session_state['roster_before'] = roster_before
            st.session_state['roster_after'] = roster_after
except ImportError:
    st.caption("Budget optimizer module not available.")

st.divider()

# ============================================================
# 5. EXPORT DOWNLOADABLE PLAN
# ============================================================
st.subheader("Download Action Plan")

# Build the text report
now = datetime.now().strftime("%Y-%m-%d %H:%M")

lines = []
lines.append(f"{'='*60}")
lines.append(f"  OOTP Perfect Team Roster Plan")
lines.append(f"  {team_name}")
lines.append(f"  Generated: {now}")
lines.append(f"{'='*60}")
lines.append("")

# Roster summary
if roster:
    lines.append(f"CURRENT ROSTER ({len(starters)} starters, {len(bench)} bench/reserve)")
    lines.append(f"  Total Starter Meta: {total_meta:,.0f}")
    lines.append(f"  PP Budget: {budget:,}")
    lines.append("")

    lines.append(f"  {'Player':<30} {'Pos':<5} {'Role':<10} {'OVR':<5} {'Meta':<8}")
    lines.append(f"  {'-'*58}")
    for r in roster:
        meta_str = f"{r['meta_score']:.0f}" if r['meta_score'] else "---"
        lines.append(f"  {r['player_name']:<30} {r['position'] or '':<5} {r['lineup_role'] or '':<10} {r['ovr'] or 0:<5} {meta_str:<8}")
    lines.append("")

# SELL FIRST section
lines.append(f"{'='*60}")
lines.append("  STEP 1: SELL FIRST")
lines.append(f"{'='*60}")
lines.append("")

if sells:
    lines.append(f"  Expected Revenue: {total_sell_pp:,} PP")
    lines.append("")
    lines.append(f"  {'Card':<35} {'Pos':<5} {'Est Price':<12} {'Reason'}")
    lines.append(f"  {'-'*75}")
    for r in sells:
        price_str = f"{r['estimated_price']:,}" if r['estimated_price'] else "---"
        lines.append(f"  {(r['card_title'] or ''):<35} {(r['position'] or ''):<5} {price_str:<12} {r['reason'] or ''}")
    lines.append("")
else:
    lines.append("  No cards to sell.")
    lines.append("")

# THEN BUY section
lines.append(f"{'='*60}")
lines.append("  STEP 2: THEN BUY (in priority order)")
lines.append(f"{'='*60}")
lines.append("")

if buys:
    lines.append(f"  Total Buy Cost: {total_buy_pp:,} PP")
    lines.append(f"  Available (budget + sells): {budget + total_sell_pp:,} PP")
    lines.append("")

    priority_labels = {1: "URGENT", 2: "HIGH", 3: "MEDIUM", 4: "LOW"}
    lines.append(f"  {'Pri':<8} {'Card':<35} {'Pos':<5} {'Price':<12} {'Meta':<8} {'Reason'}")
    lines.append(f"  {'-'*90}")

    running_cost = 0
    for r in buys:
        pri = priority_labels.get(r['priority'], str(r['priority']))
        price = r['estimated_price'] or 0
        running_cost += price
        meta_str = f"{r['meta_score']:.0f}" if r['meta_score'] else "---"
        price_str = f"{price:,}"
        over_budget = " [OVER BUDGET]" if running_cost > (budget + total_sell_pp) else ""
        lines.append(f"  {pri:<8} {(r['card_title'] or ''):<35} {(r['position'] or ''):<5} {price_str:<12} {meta_str:<8} {(r['reason'] or '')}{over_budget}")
    lines.append("")
else:
    lines.append("  No buy recommendations.")
    lines.append("")

# Net PP calculation
lines.append(f"{'='*60}")
lines.append("  NET PP CALCULATION")
lines.append(f"{'='*60}")
lines.append("")
lines.append(f"  Starting Budget:      {budget:>10,} PP")
lines.append(f"  + Sell Revenue:       {total_sell_pp:>10,} PP")
lines.append(f"  = Available:          {budget + total_sell_pp:>10,} PP")
lines.append(f"  - Total Buy Cost:     {total_buy_pp:>10,} PP")
net = budget + total_sell_pp - total_buy_pp
lines.append(f"  = Remaining:          {net:>10,} PP")
lines.append("")

# Optimizer results if available
opt_result = st.session_state.get('optimizer_result')
if opt_result and opt_result.get('transactions'):
    rb = st.session_state.get('roster_before', 0)
    ra = st.session_state.get('roster_after', 0)
    lines.append(f"{'='*60}")
    lines.append("  OPTIMIZED UPGRADES")
    lines.append(f"{'='*60}")
    lines.append("")
    lines.append(f"  Roster Meta Before: {rb:.0f}")
    lines.append(f"  Roster Meta After:  {ra:.0f}")
    lines.append(f"  Total Meta Gain:    +{opt_result['total_meta_gain']:.1f}")
    lines.append(f"  Total Cost:         {opt_result['total_cost']:,} PP")
    lines.append("")
    lines.append(f"  {'Position':<6} {'Current':<30} {'Upgrade To':<30} {'Gain':<8} {'Cost'}")
    lines.append(f"  {'-'*80}")
    for t in opt_result['transactions']:
        lines.append(
            f"  {t['position']:<6} "
            f"{t['current_player']:<30} "
            f"{t['card_title']:<30} "
            f"+{t['meta_gain']:<7.1f} "
            f"{t['price']:,} PP"
        )
    lines.append("")

# Roster before/after comparison
if roster and buys:
    lines.append(f"{'='*60}")
    lines.append("  ROSTER BEFORE / AFTER")
    lines.append(f"{'='*60}")
    lines.append("")

    # Build position-level comparison
    roster_by_pos = {}
    for r in starters:
        pos = r['position']
        if pos not in roster_by_pos or (r['meta_score'] or 0) > roster_by_pos[pos]['meta']:
            roster_by_pos[pos] = {'player': r['player_name'], 'meta': r['meta_score'] or 0}

    buy_by_pos = {}
    for b in buys:
        pos = b['position']
        if pos and (pos not in buy_by_pos or (b['meta_score'] or 0) > buy_by_pos[pos]['meta']):
            buy_by_pos[pos] = {'card': b['card_title'], 'meta': b['meta_score'] or 0, 'price': b['estimated_price'] or 0}

    all_positions = sorted(set(list(roster_by_pos.keys()) + list(buy_by_pos.keys())),
                          key=lambda p: ['C','1B','2B','3B','SS','LF','CF','RF','DH','SP','RP','CL'].index(p)
                          if p in ['C','1B','2B','3B','SS','LF','CF','RF','DH','SP','RP','CL'] else 99)

    lines.append(f"  {'Pos':<5} {'Current Player':<25} {'Meta':<8} {'Upgrade To':<25} {'New Meta':<8} {'Delta'}")
    lines.append(f"  {'-'*80}")

    for pos in all_positions:
        current = roster_by_pos.get(pos, {'player': '(empty)', 'meta': 0})
        upgrade = buy_by_pos.get(pos)
        if upgrade and upgrade['meta'] > current['meta']:
            delta = upgrade['meta'] - current['meta']
            lines.append(
                f"  {pos:<5} {current['player']:<25} {current['meta']:<8.0f} "
                f"{upgrade['card']:<25} {upgrade['meta']:<8.0f} +{delta:.0f}"
            )
        else:
            lines.append(f"  {pos:<5} {current['player']:<25} {current['meta']:<8.0f} {'(no change)':<25}")

    total_before = sum(v['meta'] for v in roster_by_pos.values())
    total_after = total_before
    for pos, upg in buy_by_pos.items():
        current_meta = roster_by_pos.get(pos, {}).get('meta', 0)
        if upg['meta'] > current_meta:
            total_after += (upg['meta'] - current_meta)

    lines.append("")
    lines.append(f"  Total Meta Before: {total_before:.0f}")
    lines.append(f"  Total Meta After:  {total_after:.0f}")
    lines.append(f"  Net Improvement:   +{total_after - total_before:.0f}")

lines.append("")
lines.append(f"{'='*60}")
lines.append("  END OF PLAN")
lines.append(f"{'='*60}")

report_text = "\n".join(lines)

# Show preview
st.text_area("Plan Preview", report_text, height=400)

# Download buttons
dl_col1, dl_col2 = st.columns(2)

with dl_col1:
    st.download_button(
        label="Download as Text",
        data=report_text,
        file_name=f"roster_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        type="primary",
        use_container_width=True,
    )

with dl_col2:
    # CSV export: one row per action (sell/buy)
    csv_rows = []
    for r in sells:
        csv_rows.append({
            "Action": "SELL",
            "Card": r['card_title'] or '',
            "Position": r['position'] or '',
            "Tier": r['tier_name'] or '',
            "Est Price": r['estimated_price'] or 0,
            "Meta Score": round(r['meta_score'], 0) if r['meta_score'] else 0,
            "Reason": r['reason'] or '',
        })
    for r in buys:
        csv_rows.append({
            "Action": "BUY",
            "Card": r['card_title'] or '',
            "Position": r['position'] or '',
            "Tier": r['tier_name'] or '',
            "Est Price": r['estimated_price'] or 0,
            "Meta Score": round(r['meta_score'], 0) if r['meta_score'] else 0,
            "Reason": r['reason'] or '',
            "Priority": r['priority'],
            "Value Ratio": round(r['value_ratio'], 1) if r['value_ratio'] else 0,
        })

    if csv_rows:
        csv_df = pd.DataFrame(csv_rows)
        csv_data = csv_df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f"roster_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.button("Download as CSV", disabled=True, use_container_width=True,
                   help="No recommendations to export")

conn.close()
