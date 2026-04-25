# Epic F — Roster Optimizer Refactor Plan

**Status:** Prereq shipped (smoke tests). Actual refactor is queued.
**Date:** 2026-04-24
**Prereq test file:** `tests/test_roster_optimizer_smoke.py` (10 tests, all passing)
**Target file:** `app/pages/4_Roster_Optimizer.py` (~4,600 lines)
**Exit criteria from handoff:** No file > 1,000 lines; pages render identically pre/post.

---

## Why this is a separate session

The file is long because it's been a high-velocity surface — Epic A, P2 #7, the no-stats fix, and the `_perf_bat` fallback all landed here in one session. Every feature Cameron operates daily touches this page. The guardrails doc says this epic is "high-risk without smoke tests first." Smoke tests are now in place, but:

1. **Rendering is not covered by any test** — the smoke tests hit the Python functions directly. A regression in the Streamlit rendering (wrong column alignment, missing chip, expander breakage) would still slip through. This epic needs **at least one screenshot-diff or DOM-shape test** before moving render code.
2. **Extraction isn't the hard part** — the hard part is the implicit module-scope state (`conn`, `max_spend`, `min_improvement`, `config`, `_perf_bat`, `_archetypes_by_card`) that dozens of nested functions close over. Rewriting these to take explicit params is the real refactor.
3. **Time-in-session trade-off** — a 7-10 day refactor doesn't fit a single session. Partial extraction risks leaving the file in a worse state than it started (two places defining the same function, or a thin shim that re-proxies everything).

Ship the plan. Execute when there's a multi-day window.

---

## Target module structure

```
app/pages/4_Roster_Optimizer.py            (≤ 800 lines — page orchestration only)
app/core/upgrade_plan.py                   (≤ 500 lines — finders, slot builder)
app/core/roster_data.py                    (≤ 400 lines — load roster/perf/archetype state)
app/pages/_roster_optimizer/row_render.py  (≤ 500 lines — Fit/Confidence/Status/Upgrade chips)
app/pages/_roster_optimizer/expanders.py   (≤ 400 lines — perf_outlook, mix_analysis, moves)
```

Everything in `app/core/*` should be **Streamlit-free** — pure Python with explicit dependencies. Page-specific helpers stay under `app/pages/_roster_optimizer/`.

---

## Extraction roadmap (in order)

### Step 1 — Extract upgrade finders (low risk, lots of value)
**From:** `4_Roster_Optimizer.py` lines ~1450–1780
**To:** `app/core/upgrade_plan.py`
**Functions:**
- `find_owned_upgrades(conn, ...)` — current line 1458
- `find_market_upgrades(conn, ..., current_fit, ..., min_improvement, max_spend, config)` — current line 1546 (expanded signature)
- `_build_slot(...)` — current line 1725
- Helpers: `action_tag`, `price_tag`, `short_name`, `_strip_set_prefix`, `_POS_TAGS`, `_SET_PREFIXES` — current lines 1628–1680

Every function needs explicit params for `conn`, `max_spend`, `min_improvement`, `config`, `_archetypes_by_card`. The page then passes them at call sites.

**Risk:** medium. Tests cover the functions (via smoke tests), but the rendering chain (`_build_slot → _render_market_action`) is tightly coupled to the page's rendering loop.

### Step 2 — Extract data loaders
**From:** `4_Roster_Optimizer.py` lines ~440–720 (batting/pitching perf loaders, league averages, peer baselines, archetypes)
**To:** `app/core/roster_data.py`
**Functions:**
- `load_latest_perf_stats(conn, active_league)` — already a helper from the 2026-04-24 no-stats fix; now extract to its own module
- `load_league_averages(conn, active_league)` — current lines 500–560
- `load_peer_baselines(conn, active_league)` — current lines 660–720
- `load_archetypes_by_card(conn)` — current lines 127–142

**Risk:** low. These are pure reads with no Streamlit calls.

### Step 3 — Extract row renderers
**From:** `4_Roster_Optimizer.py` lines ~2800–2990 (Status / Owned / Market / Fit column builders)
**To:** `app/pages/_roster_optimizer/row_render.py`
**Functions:**
- `render_fit_cell(u)` — current line 2735
- `render_status_cell(u, show_perf, _perf_bat, _perf_pit, ...)` — current line 3055
- `render_owned_action(u, ai_pick, ...)` — current line 2965
- `render_market_action(u, _pa, _market_d, _fit_d, _trigger, ...)` — current line 2998

**Risk:** high. These are tightly coupled to Streamlit's `st.markdown` + HTML emission. A screenshot test is a hard prereq — do NOT move these functions without at least a DOM-shape test in place.

### Step 4 — Extract expanders
**From:** `4_Roster_Optimizer.py` lines ~3275–3650 (`_render_mix_analysis_expander`, `_render_roster_moves_expander`, `_render_perf_outlook_expander`)
**To:** `app/pages/_roster_optimizer/expanders.py`

**Risk:** medium. Each expander is self-contained but reads from module-scope state (`conn`, `_archetypes_by_card`, etc). Pass those as args.

### Step 5 — Trim the page file
After steps 1–4, the page should be under 1,000 lines and read like orchestration — load data, compute plan, render tabs.

---

## New test coverage needed

Before step 3, add:
1. **Rendering DOM-shape test** — launch Streamlit headless, load the page, extract the rendered table, assert the expected column count + a few known-value chips. Use `preview_eval` + `preview_snapshot` or Playwright.
2. **Fixture-roster test** — seed a small DB fixture (10 cards, 2 starters, 1 obvious upgrade per position) and assert specific chip text appears. This is the gold standard regression guard.

---

## Commit strategy

Each step is a separate PR:
- Smallest viable diff per PR
- Run smoke tests + visual smoke (load the page, eyeball) on each
- If any PR feels like >500 LOC moving, split it further

Don't combine steps. Half-done extractions leave the file in limbo.

---

## Status and next actions

| Item | Status |
|---|---|
| Smoke tests written | ✅ `tests/test_roster_optimizer_smoke.py` |
| Smoke tests passing | ✅ 10/10 |
| Rendering-DOM test | ❌ needed before step 3 |
| Fixture-roster test | ❌ needed before step 3 |
| Step 1 extraction | ⬜ queued |
| Step 2 extraction | ⬜ queued |
| Step 3 extraction | ⬜ queued (blocked by rendering test) |
| Step 4 extraction | ⬜ queued |
| Step 5 final trim | ⬜ queued |

**Estimated effort to complete the full epic:** 6-9 days of focused work across 4-5 sessions.

**Next session starting point:** Step 1 (upgrade finders). Low risk, smoke-test-covered, immediate payoff (a testable pure-Python module), and prepares the ground for every subsequent step.
