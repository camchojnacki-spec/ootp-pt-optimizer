"""PP-aware purchase planning.

Takes the canonical engine recommendations (market buys in ``upgrade_plan``)
and greedy-packs them into the user's available PP budget, optimizing for
total meta delta. Also surfaces per-pick efficiency (delta / 1000 PP) so
the user can sanity-check individual buys.

Public API:
    - build_purchase_plan(upgrade_plan, budget_pp, min_delta=None) → PurchasePlan
    - PurchasePlan.to_table_rows() → list[dict] for Streamlit
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PurchasePick:
    """One recommended buy inside a plan."""
    pos: str
    card_name: str
    current_name: str
    delta: float                # meta points gained
    price: int                  # PP cost
    efficiency: float           # delta per 1000 PP
    cumulative_cost: int = 0    # running total within the plan
    remaining_budget: int = 0   # budget left after this pick


@dataclass
class PurchasePlan:
    """Budget-constrained purchase list + rejected picks."""
    budget_pp: int
    picked: list[PurchasePick] = field(default_factory=list)
    skipped_too_expensive: list[PurchasePick] = field(default_factory=list)
    skipped_below_threshold: list[PurchasePick] = field(default_factory=list)

    @property
    def total_cost(self) -> int:
        return sum(p.price for p in self.picked)

    @property
    def total_delta(self) -> float:
        return sum(p.delta for p in self.picked)

    @property
    def remaining_budget(self) -> int:
        return self.budget_pp - self.total_cost

    @property
    def avg_efficiency(self) -> float:
        if not self.picked:
            return 0.0
        return sum(p.efficiency for p in self.picked) / len(self.picked)

    def to_table_rows(self) -> list[dict]:
        """Dataframe-friendly representation for the Streamlit UI."""
        return [
            {
                'Pos': p.pos,
                'Buy': p.card_name,
                'Replacing': p.current_name,
                '+Meta': f'+{int(p.delta)}',
                'Cost PP': f'{p.price:,}',
                'Δ / 1kPP': f'{p.efficiency:.1f}',
                'After buy': f'{p.remaining_budget:,} PP left',
            }
            for p in self.picked
        ]


def build_purchase_plan(
    upgrade_plan: list[dict],
    budget_pp: int,
    *,
    min_delta: Optional[float] = None,
    min_efficiency: Optional[float] = None,
) -> PurchasePlan:
    """Greedy-pack the best market buys into the PP budget.

    Strategy:
        1. Filter to entries with a market buy + positive delta.
        2. Optionally drop picks under `min_delta` or `min_efficiency`.
        3. Sort by efficiency (delta per 1000 PP) descending.
        4. Add picks in order as long as the remaining budget covers them.
        5. Track skipped-too-expensive separately so the UI can show
           near-misses (e.g. "this would've been #1 but costs 3,000 PP").

    Args:
        upgrade_plan: the canonical engine upgrade list from Roster Optimizer.
            Each entry must have ``pos``, ``current_name``, ``market_name``,
            ``market_delta``, ``market_price``.
        budget_pp: total PP available to spend.
        min_delta: if set, drop picks with delta < this.
        min_efficiency: if set, drop picks with efficiency < this.
    """
    plan = PurchasePlan(budget_pp=int(budget_pp or 0))

    candidates: list[PurchasePick] = []
    for u in upgrade_plan:
        name = u.get('market_name')
        delta = u.get('market_delta') or 0
        price = int(u.get('market_price') or 0)
        if not name or delta <= 0 or price <= 0:
            continue
        if min_delta is not None and delta < min_delta:
            continue
        eff = (delta / price * 1000.0) if price > 0 else 0.0
        if min_efficiency is not None and eff < min_efficiency:
            continue
        candidates.append(PurchasePick(
            pos=u.get('pos') or '',
            card_name=name,
            current_name=u.get('current_name') or '',
            delta=float(delta),
            price=price,
            efficiency=eff,
        ))

    # Greedy: highest efficiency first. (Not optimal knapsack, but PT
    # cards are heterogeneous so 0/1 knapsack rarely changes the answer
    # at common budgets. Greedy is fast + legible.)
    candidates.sort(key=lambda c: (-c.efficiency, -c.delta))

    remaining = plan.budget_pp
    for c in candidates:
        if c.price <= remaining:
            c.cumulative_cost = plan.total_cost + c.price
            c.remaining_budget = remaining - c.price
            plan.picked.append(c)
            remaining -= c.price
        else:
            c.cumulative_cost = plan.total_cost
            c.remaining_budget = remaining
            plan.skipped_too_expensive.append(c)

    return plan
