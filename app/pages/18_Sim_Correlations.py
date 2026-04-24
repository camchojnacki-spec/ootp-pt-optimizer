"""Sim Correlations — mine what attribute mixes actually drive outcomes in this sim.

Three tabs:
  1. Rating x Outcome Grid — pooled correlations, pick batter / SP / RP
  2. Empirical Fit Rankings — cards ranked by observed-r-weighted composite, price-filtered
  3. Replacement Finder — for each active roster slot, bargain replacements under the sidebar PP ceiling

The page derives weights empirically from game-log outcomes rather than the meta formula. Use it
to find cards whose mix of ratings predicts actual sim outcomes (K rate / BB rate / HR rate /
hard-hit% / gb%) better than what meta_score_* surfaces.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.core.database import get_connection, load_config
from app.utils.sidebar_nav import render_sidebar_nav


# Ratings we correlate against outcomes.
BATTER_RATINGS = [
    "contact", "gap_power", "power", "eye", "avoid_ks", "babip",
    "speed", "stealing", "baserunning",
]
PITCHER_RATINGS = [
    "stuff", "movement", "control", "p_hr", "p_babip",
    "stuff_vl", "control_vl", "p_hr_vl",
    "stuff_vr", "control_vr", "p_hr_vr",
    "stamina", "hold",
]

BATTER_OUTCOMES = [
    "k_rate", "bb_rate", "hr_rate", "hit_rate", "xbh_rate",
    "avg_ev", "ld_pct", "gb_pct", "pop_pct", "hard_hit_pct",
]
PITCHER_OUTCOMES = [
    "k_rate", "bb_rate", "hr_rate", "hit_allowed_rate",
    "ev_allowed", "gb_allowed_pct", "ld_allowed_pct",
    "pop_allowed_pct", "hard_hit_allowed_pct",
]


def _num(v):
    """Coerce rating values (velocity is stored as '92 mph' in some cards)."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    try:
        return float(str(v).split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def pearson(xs, ys):
    pairs = [(_num(x), _num(y)) for x, y in zip(xs, ys)]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 10:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs)))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return num / (sx * sy)


@st.cache_data(ttl=300)
def fetch_batter_observed(min_ab: int):
    conn = get_connection()
    rating_cols = ", ".join(f"c.{r}" for r in BATTER_RATINGS)
    sql = f"""
        SELECT c.card_id, c.card_title, c.position, c.bats,
               c.last_10_price, c.meta_score_batting, c.owned,
               {rating_cols},
               COUNT(*) AS ab,
               SUM(g.outcome='K')*1.0/COUNT(*) AS k_rate,
               SUM(g.outcome='BB')*1.0/COUNT(*) AS bb_rate,
               SUM(g.outcome='HR')*1.0/COUNT(*) AS hr_rate,
               SUM(g.outcome IN ('SINGLE','DOUBLE','TRIPLE','HR'))*1.0/COUNT(*) AS hit_rate,
               SUM(g.outcome IN ('DOUBLE','TRIPLE','HR'))*1.0/COUNT(*) AS xbh_rate,
               AVG(g.exit_velocity) AS avg_ev,
               SUM(g.batted_ball_type='Line Drive')*1.0/NULLIF(SUM(g.batted_ball_type IS NOT NULL),0) AS ld_pct,
               SUM(g.batted_ball_type='Groundball')*1.0/NULLIF(SUM(g.batted_ball_type IS NOT NULL),0) AS gb_pct,
               SUM(g.batted_ball_type='Popup')*1.0/NULLIF(SUM(g.batted_ball_type IS NOT NULL),0) AS pop_pct,
               SUM(g.exit_velocity >= 95)*1.0/NULLIF(SUM(g.exit_velocity IS NOT NULL),0) AS hard_hit_pct
        FROM cards c JOIN game_log_at_bats g ON c.card_id=g.batter_card_id
        GROUP BY c.card_id HAVING ab >= ?
    """
    rows = conn.execute(sql, (min_ab,)).fetchall()
    cols = [d[0] for d in conn.execute(sql, (min_ab,)).description]
    return pd.DataFrame(rows, columns=cols)


@st.cache_data(ttl=300)
def fetch_pitcher_observed(min_bf: int):
    conn = get_connection()
    rating_cols = ", ".join(f"c.{r}" for r in PITCHER_RATINGS)
    sql = f"""
        SELECT c.card_id, c.card_title, c.position, c.pitcher_role, c.pitcher_role_name,
               c.last_10_price, c.meta_score_pitching, c.owned,
               {rating_cols},
               COUNT(*) AS bf,
               SUM(g.outcome='K')*1.0/COUNT(*) AS k_rate,
               SUM(g.outcome='BB')*1.0/COUNT(*) AS bb_rate,
               SUM(g.outcome='HR')*1.0/COUNT(*) AS hr_rate,
               SUM(g.outcome IN ('SINGLE','DOUBLE','TRIPLE','HR'))*1.0/COUNT(*) AS hit_allowed_rate,
               AVG(g.exit_velocity) AS ev_allowed,
               SUM(g.batted_ball_type='Groundball')*1.0/NULLIF(SUM(g.batted_ball_type IS NOT NULL),0) AS gb_allowed_pct,
               SUM(g.batted_ball_type='Line Drive')*1.0/NULLIF(SUM(g.batted_ball_type IS NOT NULL),0) AS ld_allowed_pct,
               SUM(g.batted_ball_type='Popup')*1.0/NULLIF(SUM(g.batted_ball_type IS NOT NULL),0) AS pop_allowed_pct,
               SUM(g.exit_velocity >= 95)*1.0/NULLIF(SUM(g.exit_velocity IS NOT NULL),0) AS hard_hit_allowed_pct
        FROM cards c JOIN game_log_at_bats g ON c.card_id=g.pitcher_card_id
        GROUP BY c.card_id HAVING bf >= ?
    """
    rows = conn.execute(sql, (min_bf,)).fetchall()
    cols = [d[0] for d in conn.execute(sql, (min_bf,)).description]
    return pd.DataFrame(rows, columns=cols)


def correlation_grid(df: pd.DataFrame, ratings, outcomes) -> pd.DataFrame:
    """Compute r(rating, outcome) for each pair. Returns a DataFrame indexed by rating."""
    if df.empty:
        return pd.DataFrame()
    data = {}
    for out in outcomes:
        cells = []
        for r in ratings:
            xs = df[r].tolist()
            ys = df[out].tolist()
            cells.append(pearson(xs, ys))
        data[out] = cells
    return pd.DataFrame(data, index=ratings)


def build_weights_from_grid(grid: pd.DataFrame, outcomes_direction: dict) -> pd.Series:
    """From the correlation grid build a per-rating composite weight.

    ``outcomes_direction`` maps outcome_name -> +1 (more = better) or -1 (more = worse).
    For a pitcher, ``k_rate`` should be +1 (strike out more is better); ``bb_rate`` is -1.
    """
    scores = pd.Series(0.0, index=grid.index)
    for out, sign in outcomes_direction.items():
        if out not in grid.columns:
            continue
        col = grid[out].fillna(0.0)
        # Each rating contributes its r (signed) * direction. Net sign tells us
        # whether higher rating predicts a better outcome.
        scores = scores + col * sign
    return scores


def fit_score(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Z-score each rating across the observed population, dot with weights."""
    out = pd.Series(0.0, index=df.index)
    total_abs = float(weights.abs().sum()) or 1.0
    for rating, w in weights.items():
        if rating not in df.columns:
            continue
        vals = df[rating].apply(_num)
        mu = vals.mean(skipna=True)
        sd = vals.std(skipna=True) or 1.0
        z = (vals - mu) / sd
        out = out + z.fillna(0.0) * w
    return out / total_abs


def fit_score_for_all_cards(all_df: pd.DataFrame, ratings: list, weights: pd.Series,
                            ref_means: dict, ref_sds: dict) -> pd.Series:
    """Apply observed-population-normalized weights to the full cards table."""
    out = pd.Series(0.0, index=all_df.index)
    total_abs = float(weights.abs().sum()) or 1.0
    for rating in ratings:
        if rating not in all_df.columns or rating not in weights.index:
            continue
        w = weights[rating]
        vals = all_df[rating].apply(_num)
        mu = ref_means.get(rating, vals.mean(skipna=True))
        sd = ref_sds.get(rating, vals.std(skipna=True)) or 1.0
        z = (vals - mu) / sd
        out = out + z.fillna(0.0) * w
    return out / total_abs


# ══════════════════════════════════════════════════════════════════════════════
#   Page setup
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Sim Correlations", page_icon="\U0001f9ec", layout="wide")
render_sidebar_nav()

conn = get_connection()
config = load_config()
default_budget = config.get("pp_budget", 500)

with st.sidebar:
    st.header("Filters")
    max_spend = st.number_input(
        "Max PP per card", min_value=0, max_value=999999,
        value=int(default_budget), step=500, format="%d",
        help="Replacement finder uses this as the cap. Set 0 for unlimited.",
    )
    min_ab = st.slider("Batter: min AB for inclusion", 20, 300, 50, 10,
                       help="Cards with fewer observed ABs are excluded from correlation + ranking.")
    min_bf = st.slider("Pitcher: min BF for inclusion", 20, 300, 50, 10,
                       help="Cards with fewer observed BFs are excluded.")

st.title("\U0001f9ec Sim Correlations")
st.caption(
    "Mines the game-log layer to find **which attribute mixes actually predict outcomes** in this sim. "
    "Weights are derived empirically from observed Pearson r, not from the meta formula. "
    "Use this to find cards the meta ranking is underrating because their mix is role-right."
)

bat_df = fetch_batter_observed(min_ab)
pit_df = fetch_pitcher_observed(min_bf)

if bat_df.empty or pit_df.empty:
    st.warning("Not enough game-log data yet. Lower the min-sample thresholds in the sidebar.")
    st.stop()

# Correlation grids (all computed once; filtered by role for display).
bat_grid = correlation_grid(bat_df, BATTER_RATINGS, BATTER_OUTCOMES)

is_sp = pit_df["pitcher_role_name"].fillna("").str.upper().str.startswith("S")
is_rp = pit_df["pitcher_role_name"].fillna("").str.upper().str.startswith(("R", "C"))
sp_grid = correlation_grid(pit_df[is_sp], PITCHER_RATINGS, PITCHER_OUTCOMES)
rp_grid = correlation_grid(pit_df[is_rp], PITCHER_RATINGS, PITCHER_OUTCOMES)
pit_grid = correlation_grid(pit_df, PITCHER_RATINGS, PITCHER_OUTCOMES)

# Build empirical weights — higher = more of that rating predicts better outcomes for this role.
BATTER_DIRECTION = {
    "k_rate": -1, "bb_rate": +1, "hr_rate": +1, "hit_rate": +1, "xbh_rate": +1,
    "avg_ev": +1, "hard_hit_pct": +1, "ld_pct": +1, "pop_pct": -1, "gb_pct": -1,
}
PITCHER_DIRECTION = {
    "k_rate": +1, "bb_rate": -1, "hr_rate": -1, "hit_allowed_rate": -1,
    "ev_allowed": -1, "hard_hit_allowed_pct": -1, "ld_allowed_pct": -1,
    "gb_allowed_pct": +1, "pop_allowed_pct": +1,
}

bat_weights = build_weights_from_grid(bat_grid, BATTER_DIRECTION)
sp_weights = build_weights_from_grid(sp_grid, PITCHER_DIRECTION)
rp_weights = build_weights_from_grid(rp_grid, PITCHER_DIRECTION)

# Per-rating means/sds from the observed populations (for projecting onto ALL cards).
bat_means = {r: bat_df[r].apply(_num).mean(skipna=True) for r in BATTER_RATINGS}
bat_sds = {r: bat_df[r].apply(_num).std(skipna=True) for r in BATTER_RATINGS}
sp_means = {r: pit_df[is_sp][r].apply(_num).mean(skipna=True) for r in PITCHER_RATINGS}
sp_sds = {r: pit_df[is_sp][r].apply(_num).std(skipna=True) for r in PITCHER_RATINGS}
rp_means = {r: pit_df[is_rp][r].apply(_num).mean(skipna=True) for r in PITCHER_RATINGS}
rp_sds = {r: pit_df[is_rp][r].apply(_num).std(skipna=True) for r in PITCHER_RATINGS}


# ══════════════════════════════════════════════════════════════════════════════
#   Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_grid, tab_ranks = st.tabs([
    "\U0001f4ca Rating \u00d7 Outcome Grid",
    "\U0001f3c6 Empirical Fit Rankings",
])

st.info(
    "**This page is diagnostic.** Per-slot replacement recommendations (fit vs meta deltas, "
    "archetype matching, price-filtered candidates) live on the **Roster Optimizer** page — "
    "look for the Fit column and the \u2728 Mix Analysis expander per lineup. The grid below "
    "explains _why_ those recommendations look the way they do."
)


# ── Tab 1: Correlation grid ───────────────────────────────────────────────────
with tab_grid:
    st.markdown(
        "Each cell is the Pearson r between a card rating (rows) and an observed outcome (columns). "
        "Green = higher rating predicts higher outcome; red = negative. "
        "Cells with no value lack sample size or variance."
    )
    role_pick = st.radio("View", ["Batters", "Pitchers (SP)", "Pitchers (RP)", "Pitchers (All)"],
                         horizontal=True, key="grid_role")
    if role_pick == "Batters":
        grid = bat_grid
        st.caption(f"Batters, n={len(bat_df)} (min {min_ab} AB)")
    elif role_pick == "Pitchers (SP)":
        grid = sp_grid
        st.caption(f"Starting pitchers, n={int(is_sp.sum())} (min {min_bf} BF)")
    elif role_pick == "Pitchers (RP)":
        grid = rp_grid
        st.caption(f"Relief pitchers, n={int(is_rp.sum())} (min {min_bf} BF)")
    else:
        grid = pit_grid
        st.caption(f"All pitchers, n={len(pit_df)} (min {min_bf} BF)")

    if grid.empty:
        st.info("No rows match the current filters.")
    else:
        styled = grid.style.format("{:+.3f}", na_rep="\u2014").background_gradient(
            cmap="RdYlGn", vmin=-0.5, vmax=0.5, axis=None,
        )
        st.dataframe(styled, use_container_width=True)

        # Top absolute-r cells so the user sees the strongest signals first.
        flat = grid.stack(future_stack=True).dropna().reset_index()
        flat.columns = ["Rating", "Outcome", "r"]
        flat["|r|"] = flat["r"].abs()
        flat = flat.sort_values("|r|", ascending=False).head(12).drop(columns=["|r|"])
        st.markdown("**Strongest 12 signals in this view:**")
        st.dataframe(flat, use_container_width=True, hide_index=True,
                     column_config={"r": st.column_config.NumberColumn(format="%+.3f")})


# ── Tab 2: Empirical fit rankings ─────────────────────────────────────────────
with tab_ranks:
    st.markdown(
        "Cards ranked by **empirical fit score** — a z-score composite built from the observed "
        "r-weights in the correlation grid. Score is in standard deviations of expected outcome. "
        "Apply the sidebar **Max PP per card** filter to shop your price range."
    )
    role_pick = st.radio("Role", ["Batter", "SP", "RP"], horizontal=True, key="rank_role")
    owned_only = st.checkbox("Owned cards only", value=False)

    all_cards = pd.read_sql(
        "SELECT card_id, card_title, position, position_name, pitcher_role, pitcher_role_name, "
        "bats, last_10_price, meta_score_batting, meta_score_pitching, owned, tier_name, "
        + ", ".join(set(BATTER_RATINGS + PITCHER_RATINGS))
        + " FROM cards", conn,
    )

    if role_pick == "Batter":
        weights = bat_weights
        means = bat_means
        sds = bat_sds
        ratings = BATTER_RATINGS
        subset = all_cards[all_cards["position_name"].fillna("").str.contains(
            "P", regex=False) == False].copy()
        meta_col = "meta_score_batting"
    elif role_pick == "SP":
        weights = sp_weights
        means = sp_means
        sds = sp_sds
        ratings = PITCHER_RATINGS
        subset = all_cards[all_cards["pitcher_role_name"].fillna("").str.upper().str.startswith("S")].copy()
        meta_col = "meta_score_pitching"
    else:
        weights = rp_weights
        means = rp_means
        sds = rp_sds
        ratings = PITCHER_RATINGS
        subset = all_cards[all_cards["pitcher_role_name"].fillna("").str.upper().str.startswith(("R", "C"))].copy()
        meta_col = "meta_score_pitching"

    subset = subset.reset_index(drop=True)
    subset["fit"] = fit_score_for_all_cards(subset, ratings, weights, means, sds)

    if owned_only:
        subset = subset[subset["owned"].fillna(0) > 0]

    if max_spend > 0:
        subset = subset[(subset["last_10_price"].fillna(0) <= max_spend) |
                        (subset["owned"].fillna(0) > 0)]

    subset = subset.sort_values("fit", ascending=False).head(40)

    display_ratings = [r for r in ratings if r in ("stuff", "movement", "control", "p_hr",
                                                   "stamina", "contact", "power", "eye",
                                                   "avoid_ks", "babip")]
    display_cols = ["card_title", "tier_name", "fit", meta_col, "last_10_price", "owned"] + display_ratings
    display_cols = [c for c in display_cols if c in subset.columns]

    st.dataframe(
        subset[display_cols].rename(columns={
            "card_title": "Card", "tier_name": "Tier",
            meta_col: "Meta", "last_10_price": "Price", "fit": "Fit (\u03c3)",
            "owned": "Owned",
        }),
        use_container_width=True, hide_index=True,
        column_config={
            "Fit (\u03c3)": st.column_config.NumberColumn(format="%+.2f",
                help="Z-score composite. +1.0 = 1 SD above mean on the outcomes that matter for this role."),
            "Meta": st.column_config.NumberColumn(format="%.0f"),
            "Price": st.column_config.NumberColumn(format="%d"),
        },
    )

    with st.expander("Weights used for this role"):
        weights_df = pd.DataFrame({
            "Rating": weights.index,
            "Net weight (sum of signed r \u00d7 direction)": weights.values,
        }).sort_values("Net weight (sum of signed r \u00d7 direction)",
                       key=lambda s: s.abs(), ascending=False)
        st.dataframe(weights_df, use_container_width=True, hide_index=True,
                     column_config={
                         "Net weight (sum of signed r \u00d7 direction)":
                             st.column_config.NumberColumn(format="%+.3f"),
                     })
        st.caption(
            "These weights come from the correlation grid. A rating gains weight when its observed r "
            "predicts the outcomes we care about for the role. Direction vector: K=+1/BB=-1/HR=-1/EV_allowed=-1 "
            "for pitchers, inverse for batters."
        )


# Replacement Finder deliberately removed — that lives on the Roster Optimizer
# page now (per-slot, integrated with Meta / Confidence / Status columns).
# Keeping it here duplicated the recommendation surface and fragmented trust.
