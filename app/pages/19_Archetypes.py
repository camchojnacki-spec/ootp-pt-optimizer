"""Archetype + Role-Fit — mix-aware card evaluation.

Cards are clustered by their rating profile into named archetypes
("Power+Discipline", "Command Fireman", "Athletic Gap-Power", ...). Each
archetype carries a sample-weighted empirical WAR/600 (batters) or WAR/200
(pitchers), so you can see at a glance which mix patterns actually
produce vs. which are one-trick-pony traps.

Built from ``card_archetypes`` derived table — rebuilt by the background
worker after every ingestion. Clustering uses k-means on z-scored ratings
(8 batting clusters, 6 SP clusters, 6 RP clusters).

Product hook: the **Find Similar Cards** search takes any card the user
owns (or picks) and returns market cards in the SAME archetype under
their PP budget — fixing the "replacement finder returned 0 suggestions"
pain point where meta-threshold matching was too narrow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.core.database import get_connection, load_config
from app.utils.sidebar_nav import render_sidebar_nav

st.set_page_config(page_title="Archetypes", page_icon="\U0001f9ec", layout="wide")
render_sidebar_nav()

conn = get_connection()
config = load_config()
default_budget = config.get("pp_budget", 500)

with st.sidebar:
    st.header("Filters")
    max_spend = st.number_input(
        "Max PP per card", min_value=0, max_value=9_999_999,
        value=int(default_budget), step=500, format="%d",
        help="Caps the Find-Similar search. 0 = unlimited.",
    )
    role_filter = st.radio(
        "Role",
        ["All", "Batting", "SP", "RP"],
        horizontal=True,
        help="Archetypes are computed per-role — batting, SP, and RP get separate cluster fits.",
    )

st.title("\U0001f9ec Archetypes & Role-Fit")
st.caption(
    "Cards clustered by rating **mix** rather than meta score. Each archetype shows the "
    "empirical WAR rate its members actually produced — the mix patterns that win this sim. "
    "Use the **Find Similar** search to replace a card with same-archetype options under budget."
)

# ── Top pane: archetype leaderboard ─────────────────────────────────
role_clause = ""
if role_filter != "All":
    role_code = {"Batting": "batting", "SP": "SP", "RP": "RP"}[role_filter]
    role_clause = f"WHERE role = '{role_code}'"

st.subheader("Archetype leaderboard")
st.caption("Sample-weighted mean WAR rate per archetype. Batting = WAR/600, pitching = WAR/200.")

arch_df = pd.read_sql(f"""
    SELECT
        role,
        archetype_name,
        ROUND(AVG(archetype_war), 3) AS war_rate,
        MAX(n_in_archetype) AS cluster_size,
        ROUND(AVG(mix_score), 1) AS avg_mix,
        ROUND(AVG(count_elite), 2) AS avg_elite_ratings
    FROM card_archetypes
    {role_clause}
    GROUP BY role, archetype_name
    ORDER BY war_rate DESC
""", conn)

if arch_df.empty:
    st.info("No archetype data yet — run a data refresh to populate.")
    st.stop()

# Pretty-print: war_rate header reflects whichever roles are visible.
st.dataframe(
    arch_df.rename(columns={
        "role": "Role",
        "archetype_name": "Archetype",
        "war_rate": "WAR rate",
        "cluster_size": "N cards",
        "avg_mix": "Avg mix-score",
        "avg_elite_ratings": "Avg elite ratings (>=80)",
    }),
    use_container_width=True,
    hide_index=True,
)

# ── Middle pane: top cards in each archetype ────────────────────────
st.subheader("Top cards per archetype")
st.caption("Best-fit members of each archetype — fit_score is distance to cluster centroid (100 = perfect match).")

top_cards = pd.read_sql(f"""
    SELECT
        c.card_id,
        c.card_title,
        c.tier_name,
        c.card_type,
        c.position_name,
        c.pitcher_role_name,
        ca.role,
        ca.archetype_name,
        ca.fit_score,
        ca.mix_score,
        ca.count_elite,
        ca.archetype_war,
        c.meta_score_batting,
        c.meta_score_pitching,
        c.sell_order_low,
        c.buy_order_high
    FROM card_archetypes ca
    JOIN cards c USING (card_id)
    {role_clause.replace('role', 'ca.role') if role_clause else ''}
    ORDER BY ca.archetype_war DESC, ca.fit_score DESC
    LIMIT 500
""", conn)

# Display price column: pick sell_order_low if available, else buy_order_high
def _market_price(row):
    sl = row.get("sell_order_low")
    bh = row.get("buy_order_high")
    if sl and sl > 0:
        return int(sl)
    if bh and bh > 0:
        return int(bh)
    return None
top_cards["market_pp"] = top_cards.apply(_market_price, axis=1)

# Combined meta column (batting OR pitching)
top_cards["meta"] = top_cards["meta_score_batting"].fillna(top_cards["meta_score_pitching"])
display_cols = [
    "card_title", "role", "archetype_name", "archetype_war",
    "fit_score", "mix_score", "count_elite", "meta", "market_pp",
    "tier_name", "position_name", "pitcher_role_name",
]
st.dataframe(
    top_cards[display_cols].rename(columns={
        "card_title": "Card",
        "role": "Role",
        "archetype_name": "Archetype",
        "archetype_war": "WAR (arch mean)",
        "fit_score": "Fit",
        "mix_score": "Mix",
        "count_elite": "# elite",
        "meta": "Meta",
        "market_pp": "PP",
        "tier_name": "Tier",
        "position_name": "Pos",
        "pitcher_role_name": "P Role",
    }),
    use_container_width=True,
    hide_index=True,
    height=400,
)

# ── Bottom pane: Find Similar Cards (the product hook) ──────────────
st.subheader("\U0001f50d Find similar cards under budget")
st.caption(
    "Pick a source card (owned, roster, or any market card). We return other cards in the SAME archetype "
    "that are also within your PP budget — ordered by fit_score so you get the closest profile match first."
)

# Source selector — default to user's roster, fallback to market cards
card_source_df = pd.read_sql("""
    SELECT DISTINCT c.card_id, c.card_title, c.tier_name,
           COALESCE(r.card_id IS NOT NULL, 0) AS owned
    FROM cards c
    LEFT JOIN roster r ON r.card_id = c.card_id
    WHERE c.card_title IS NOT NULL
    ORDER BY owned DESC, c.card_title
    LIMIT 2000
""", conn)

options = {
    row["card_title"]: row["card_id"] for _, row in card_source_df.iterrows()
}
if not options:
    st.info("No cards available. Run a data refresh first.")
    st.stop()

picked_title = st.selectbox(
    "Source card",
    list(options.keys()),
    help="Pick any card. We'll find others in its archetype.",
)
picked_id = options[picked_title]

# Resolve archetype of source
source = pd.read_sql(f"""
    SELECT ca.role, ca.archetype_name, ca.archetype_war, ca.fit_score,
           ca.mix_score, ca.count_elite
    FROM card_archetypes ca WHERE ca.card_id = ?
""", conn, params=(picked_id,))
if source.empty:
    st.warning(f"'{picked_title}' has no archetype assignment yet. "
               "Pick another card or run a refresh.")
    st.stop()

src = source.iloc[0]
c_info1, c_info2, c_info3, c_info4 = st.columns(4)
c_info1.metric("Source archetype", src["archetype_name"])
c_info2.metric("Role", src["role"])
c_info3.metric("Archetype WAR", f"{src['archetype_war']:.2f}")
c_info4.metric("Source fit", f"{src['fit_score']:.0f}")

# Candidate cards: same archetype, under budget, NOT the same card_id
budget_clause = "AND (COALESCE(c.sell_order_low, c.buy_order_high, 0) <= ?)" if max_spend > 0 else ""
params = [src["role"], src["archetype_name"], picked_id]
if max_spend > 0:
    params.append(max_spend)

candidates = pd.read_sql(f"""
    SELECT
        c.card_id,
        c.card_title,
        c.tier_name,
        c.position_name,
        c.pitcher_role_name,
        ca.fit_score,
        ca.mix_score,
        ca.count_elite,
        ca.archetype_war,
        c.meta_score_batting,
        c.meta_score_pitching,
        c.sell_order_low,
        c.buy_order_high
    FROM card_archetypes ca
    JOIN cards c USING (card_id)
    WHERE ca.role = ?
      AND ca.archetype_name = ?
      AND c.card_id != ?
      {budget_clause}
    ORDER BY ca.fit_score DESC
    LIMIT 75
""", conn, params=params)

if candidates.empty:
    st.warning(
        "No similar cards found under budget. Raise **Max PP per card** in the sidebar — "
        "the archetype set is small at lower budgets."
    )
else:
    candidates["market_pp"] = candidates.apply(_market_price, axis=1)
    candidates["meta"] = candidates["meta_score_batting"].fillna(candidates["meta_score_pitching"])
    show_cols = [
        "card_title", "fit_score", "mix_score", "count_elite",
        "meta", "market_pp", "tier_name", "position_name", "pitcher_role_name",
    ]
    st.dataframe(
        candidates[show_cols].rename(columns={
            "card_title": "Card",
            "fit_score": "Fit",
            "mix_score": "Mix",
            "count_elite": "# elite",
            "meta": "Meta",
            "market_pp": "PP",
            "tier_name": "Tier",
            "position_name": "Pos",
            "pitcher_role_name": "P Role",
        }),
        use_container_width=True,
        hide_index=True,
        height=400,
    )
    st.caption(
        f"Showing up to 75 cards in archetype **{src['archetype_name']}** "
        f"({'unlimited' if max_spend == 0 else f'PP ≤ {max_spend:,}'}), ranked by fit to centroid."
    )

with st.expander("About archetypes"):
    st.markdown("""
The 2026-04-20 attribute-mix analysis (n=2,385 batter-seasons + 2,565 pitcher-seasons
pooled across lb124 + i76) showed that individual ratings poorly predict WAR:

- **max(core ratings) r=+0.244** — the single biggest outlier stat
- **min-of-top-3 core r=+0.452** — the third-best rating
- **count(core ≥ 80) r=+0.426** — how many elite ratings

A card with three 80s meaningfully out-produces a card with one 95 and two 60s —
*mixes beat outliers*. This page surfaces those mix patterns as named clusters, each
with an empirical WAR rate attached.

- **Archetype**: which cluster this card belongs to (k-means on z-scored ratings)
- **Fit score**: 0–100, how close the card is to the cluster centroid (perfect match = 100)
- **Mix score**: 0–100, blend of min-of-top-3 and count-elite
- **Archetype WAR**: sample-weighted mean WAR/600 (batters) or WAR/200 (pitchers) across cluster
    """)
