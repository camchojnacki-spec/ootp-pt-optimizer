"""Plan history — capture exported plans and diff them against later imports.

Tier-2 #7 from the product review: the Export Plan page can download a plan,
but nothing imports back "which parts of the plan did you actually execute and
what happened?". Without this, the app can never learn from its own advice.

Design:
    - One row per export in ``plan_history``. Sells/buys live as JSON blobs on
      the row so we don't need a separate ``plan_actions`` table — adherence is
      computed at view time, not stored.
    - Each plan row records the ``owned`` count at export time per card. That's
      what makes the diff robust: ``owned_now > owned_at_export`` means a buy
      was executed even if the user only acted on one of several recommended
      copies, and ``owned_now < owned_at_export`` flags a sell. Going purely
      by "is the card still in the collection" would mis-classify partial
      duplicate sells.
    - Adherence is descriptive, not prescriptive — we never tell the user
      "you should have followed the plan". The point is to make the loop
      visible so the user (and future calibration passes) can see which kinds
      of recs actually get acted on.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime

from app.core.database import get_connection

logger = logging.getLogger(__name__)


# ─── Schema ──────────────────────────────────────────────────────────────────
def ensure_plan_history_table(conn: sqlite3.Connection) -> None:
    """Idempotent CREATE TABLE for ``plan_history``.

    Uses the same lazy pattern as ``_ensure_calibration_table`` in
    meta_validation — first call creates the table, subsequent calls are
    no-ops via ``IF NOT EXISTS``.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS plan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            team_name TEXT,
            pp_budget INTEGER,
            roster_meta_at_export REAL,
            sells_json TEXT NOT NULL,
            buys_json TEXT NOT NULL,
            sells_count INTEGER DEFAULT 0,
            buys_count INTEGER DEFAULT 0,
            sells_total_pp INTEGER DEFAULT 0,
            buys_total_pp INTEGER DEFAULT 0,
            notes TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_plan_history_date "
        "ON plan_history(exported_at DESC)"
    )
    conn.commit()


# ─── Capture ─────────────────────────────────────────────────────────────────
def save_plan_snapshot(
    team_name: str,
    pp_budget: int,
    sells: list,
    buys: list,
    roster_meta_at_export: float,
    conn: sqlite3.Connection | None = None,
    notes: str | None = None,
) -> int | None:
    """Persist a plan snapshot so a future import can diff against it.

    Args:
        team_name: User's team name (from config).
        pp_budget: Active PP budget at export time.
        sells: Iterable of sell rows. Each must be a dict-like or sqlite Row
            with ``card_id``, ``card_title``, ``position``, ``estimated_price``,
            optionally ``reason``, ``meta_score``, ``tier_name``.
        buys: Iterable of buy rows with the same shape plus ``priority``.
        roster_meta_at_export: Total starter meta at export time (lets us
            measure plan-induced gain after execution).
        conn: Optional connection (we open one if not provided).
        notes: Optional free-text label for this snapshot.

    Returns: The new plan_history row id, or None on failure.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    try:
        ensure_plan_history_table(conn)

        # For every card in the plan, snapshot its current ``owned`` count.
        # This is what makes the diff robust against partial-duplicate sells.
        def _snapshot_one(row, kind: str) -> dict:
            cid = row["card_id"] if hasattr(row, "keys") else row.get("card_id")
            owned_at = 0
            if cid:
                try:
                    cur = conn.execute(
                        "SELECT owned FROM cards WHERE card_id = ?", (cid,)
                    ).fetchone()
                    if cur and cur[0] is not None:
                        owned_at = int(cur[0])
                except Exception:
                    owned_at = 0
            base = {
                "card_id": cid,
                "card_title": row["card_title"] if hasattr(row, "keys") else row.get("card_title"),
                "position": row["position"] if hasattr(row, "keys") else row.get("position"),
                "estimated_price": (
                    row["estimated_price"] if hasattr(row, "keys")
                    else row.get("estimated_price")
                ) or 0,
                "meta_score": (
                    row["meta_score"] if hasattr(row, "keys")
                    else row.get("meta_score")
                ),
                "tier_name": (
                    row["tier_name"] if hasattr(row, "keys")
                    else row.get("tier_name")
                ),
                "reason": (
                    row["reason"] if hasattr(row, "keys")
                    else row.get("reason")
                ) or "",
                "owned_at_export": owned_at,
                "kind": kind,
            }
            if kind == "buy":
                base["priority"] = (
                    row["priority"] if hasattr(row, "keys")
                    else row.get("priority")
                )
            return base

        sells_data = [_snapshot_one(r, "sell") for r in sells]
        buys_data = [_snapshot_one(r, "buy") for r in buys]

        sells_total = sum(int(s["estimated_price"] or 0) for s in sells_data)
        buys_total = sum(int(b["estimated_price"] or 0) for b in buys_data)

        cur = conn.execute(
            """
            INSERT INTO plan_history (
                team_name, pp_budget, roster_meta_at_export,
                sells_json, buys_json,
                sells_count, buys_count,
                sells_total_pp, buys_total_pp,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                team_name,
                int(pp_budget or 0),
                float(roster_meta_at_export or 0),
                json.dumps(sells_data),
                json.dumps(buys_data),
                len(sells_data),
                len(buys_data),
                sells_total,
                buys_total,
                notes,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    except Exception as e:
        logger.error(f"save_plan_snapshot failed: {e}", exc_info=True)
        return None
    finally:
        if own_conn:
            conn.close()


# ─── Read ────────────────────────────────────────────────────────────────────
def get_latest_plan(conn: sqlite3.Connection | None = None) -> dict | None:
    """Return the most recent plan_history row as a dict, or None."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        ensure_plan_history_table(conn)
        row = conn.execute(
            "SELECT * FROM plan_history ORDER BY exported_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"get_latest_plan failed: {e}", exc_info=True)
        return None
    finally:
        if own_conn:
            conn.close()


def list_plans(conn: sqlite3.Connection | None = None, limit: int = 10) -> list:
    """Return the most recent ``limit`` plans, newest first."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        ensure_plan_history_table(conn)
        rows = conn.execute(
            "SELECT id, exported_at, team_name, pp_budget, "
            "roster_meta_at_export, sells_count, buys_count, "
            "sells_total_pp, buys_total_pp, notes "
            "FROM plan_history ORDER BY exported_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"list_plans failed: {e}", exc_info=True)
        return []
    finally:
        if own_conn:
            conn.close()


# ─── Diff ────────────────────────────────────────────────────────────────────
def diff_plan_against_current(
    plan_id: int,
    conn: sqlite3.Connection | None = None,
) -> dict | None:
    """Compare a saved plan against the current cards/roster state.

    For each sell:
        executed = current owned < owned_at_export  (user actually parted with a copy)
    For each buy:
        executed = current owned > owned_at_export  (user picked up a copy)

    The "owned delta" approach means we don't get fooled by duplicates: if the
    user owned 2 copies of a duplicate at export time and 1 copy now, that
    counts as one sell executed even though the card is still in the collection.

    Returns a dict::
        {
          "plan": {row data},
          "sells": [{...sell row..., "owned_now": int, "executed": bool}],
          "buys":  [{...buy row...,  "owned_now": int, "executed": bool}],
          "sells_executed": int, "sells_total": int,
          "buys_executed":  int, "buys_total":  int,
          "exec_pp_sells":  int,  # PP captured from executed sells
          "exec_pp_buys":   int,  # PP spent on executed buys
          "current_roster_meta": float,
          "meta_delta": float,    # current - at-export
        }
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        ensure_plan_history_table(conn)
        plan_row = conn.execute(
            "SELECT * FROM plan_history WHERE id = ?", (int(plan_id),)
        ).fetchone()
        if not plan_row:
            return None

        plan = dict(plan_row)
        try:
            sells = json.loads(plan.get("sells_json") or "[]")
            buys = json.loads(plan.get("buys_json") or "[]")
        except Exception:
            sells, buys = [], []

        # Build a single owned-lookup so we don't N+1 against the cards table.
        all_ids = [s.get("card_id") for s in sells if s.get("card_id")] + [
            b.get("card_id") for b in buys if b.get("card_id")
        ]
        owned_now: dict[int, int] = {}
        if all_ids:
            placeholders = ",".join("?" * len(all_ids))
            for r in conn.execute(
                f"SELECT card_id, owned FROM cards WHERE card_id IN ({placeholders})",
                all_ids,
            ):
                owned_now[int(r["card_id"])] = int(r["owned"] or 0)

        # When multiple plan rows share a card_id (rare — happens when the
        # recommendations table has duplicate entries for the same card), the
        # naive "owned_now < owned_at_export" check would mark *all* matching
        # rows as executed even if the user only acted on one copy. We allocate
        # executed credit per card_id by sorting same-card rows by price
        # descending and only crediting up to the actual owned-delta.
        def _classify(rows: list, kind: str) -> tuple[list, int, int]:
            # Index rows by card_id and compute total executed count per card
            by_card: dict[int, list[int]] = {}
            for idx, r in enumerate(rows):
                cid = r.get("card_id")
                if cid:
                    by_card.setdefault(int(cid), []).append(idx)

            executed_idx: set[int] = set()
            for cid, idxs in by_card.items():
                on_now = owned_now.get(cid, 0)
                on_then_max = max(
                    int(rows[i].get("owned_at_export", 0) or 0) for i in idxs
                )
                if kind == "sell":
                    delta = max(0, on_then_max - on_now)
                else:  # buy
                    delta = max(0, on_now - on_then_max)
                if delta <= 0:
                    continue
                # Prefer to credit the highest-priced rows first — gives the
                # user a nicer "we counted the expensive sell as executed" UX.
                idxs_sorted = sorted(
                    idxs,
                    key=lambda i: int(rows[i].get("estimated_price") or 0),
                    reverse=True,
                )
                for i in idxs_sorted[:delta]:
                    executed_idx.add(i)

            out: list = []
            count = 0
            pp = 0
            for idx, r in enumerate(rows):
                cid = r.get("card_id")
                on = owned_now.get(int(cid), 0) if cid else 0
                # Rows without a card_id can't be diffed — leave as pending.
                executed = idx in executed_idx
                out.append({**r, "owned_now": on, "executed": executed})
                if executed:
                    count += 1
                    pp += int(r.get("estimated_price") or 0)
            return out, count, pp

        sells_out, sells_executed, exec_pp_sells = _classify(sells, "sell")
        buys_out, buys_executed, exec_pp_buys = _classify(buys, "buy")

        # Current roster meta — same definition the Export Plan page uses for
        # ``total_meta`` (sum over starters/rotation/closer/bullpen).
        cur_meta = 0.0
        try:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(meta_score), 0) AS total
                FROM roster_current
                WHERE lineup_role IN ('starter','rotation','closer','bullpen')
                """
            ).fetchone()
            if row and row["total"] is not None:
                cur_meta = float(row["total"])
        except Exception:
            cur_meta = 0.0

        meta_delta = cur_meta - float(plan.get("roster_meta_at_export") or 0)

        return {
            "plan": plan,
            "sells": sells_out,
            "buys": buys_out,
            "sells_executed": sells_executed,
            "sells_total": len(sells_out),
            "buys_executed": buys_executed,
            "buys_total": len(buys_out),
            "exec_pp_sells": exec_pp_sells,
            "exec_pp_buys": exec_pp_buys,
            "current_roster_meta": cur_meta,
            "meta_delta": meta_delta,
        }
    except Exception as e:
        logger.error(f"diff_plan_against_current failed: {e}", exc_info=True)
        return None
    finally:
        if own_conn:
            conn.close()
