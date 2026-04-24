"""LLM council orchestration.

Flow:
    1. Engine emits a recommendation (already in ``recommendation_log``).
    2. Caller invokes ``review_recommendation(rec_id)``.
    3. Council fans out the verification prompt to all configured
       ``primary`` providers in parallel.
    4. If the primary verdicts disagree or any have low confidence,
       ``verifier`` providers are invoked for a second opinion.
    5. All verdicts are persisted to ``council_review`` (one row per
       provider response) linked back to the rec.
    6. A composite verdict (CONCUR/DISSENT/MIXED) is computed and cached
       on the rec for UI display.

Public API:
    - review_recommendation(rec_id, force=False) → CouncilResult
    - review_many(rec_ids) → list[CouncilResult]
    - get_council_reviews(rec_id) → list[ProviderReview]
    - composite_verdict(reviews) → str

The UI calls ``review_recommendation`` in response to a button click.
No auto-firing — LLMs are verifiers, not auto-analyzers.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ProviderReview:
    """One LLM's verdict on one recommendation."""
    provider_id: str
    vendor: str
    model: str
    role: str                  # primary | verifier | critic
    verdict: str               # CONCUR | DISSENT | NEEDS_MORE_INFO | ERROR | UNKNOWN
    confidence: Optional[int]
    reasoning: str
    risks: str
    alternative: Optional[str]
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None
    raw: str = ''


@dataclass
class CouncilResult:
    """Aggregate of all provider reviews for one recommendation."""
    rec_id: int
    composite_verdict: str     # CONCUR | DISSENT | MIXED | ERROR | NEEDS_MORE_INFO
    composite_confidence: Optional[float]
    reviews: list[ProviderReview]
    reviewed_at: str

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ──────────────────────────────────────────────────────────────────────
# Table ensurer
# ──────────────────────────────────────────────────────────────────────

def _ensure_table(conn) -> None:
    """Create council_review table on first use."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS council_review (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rec_id INTEGER NOT NULL,
            provider_id TEXT NOT NULL,
            vendor TEXT,
            model TEXT,
            role TEXT,
            verdict TEXT,
            confidence INTEGER,
            reasoning TEXT,
            risks TEXT,
            alternative TEXT,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            latency_s REAL DEFAULT 0,
            error TEXT,
            raw TEXT,
            reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(rec_id) REFERENCES recommendation_log(id)
        );
        CREATE INDEX IF NOT EXISTS idx_council_rec ON council_review(rec_id);
    """)
    conn.commit()


# ──────────────────────────────────────────────────────────────────────
# Context assembly — enrich the rec before sending to LLM
# ──────────────────────────────────────────────────────────────────────

def _enrich_rec_context(rec: dict, conn) -> dict:
    """Gather card data, observed stats, meta explainer for the prompt."""
    extra: dict = {}
    target_name = rec.get('player_name')
    if not target_name:
        return extra

    # Resolve target_name to a specific card_id via the canonical resolver
    # (expands abbreviations, refuses ambiguous matches). Falls back to LIKE
    # only if the resolver can't place the name (e.g. genuinely a stranger
    # player not in our cards table).
    from app.core.name_resolver import resolve_to_card_id
    resolved_card_id = resolve_to_card_id(target_name, prefer_owned=True, conn=conn)

    # Card ratings
    try:
        if resolved_card_id:
            row = conn.execute(
                """
                SELECT c.card_id, c.card_title, c.contact, c.gap_power, c.power,
                       c.eye, c.avoid_ks, c.babip,
                       c.speed, c.stealing, c.baserunning,
                       c.contact_vl, c.contact_vr, c.power_vl, c.power_vr,
                       c.stuff, c.movement, c.control, c.p_hr, c.stamina, c.hold,
                       c.stuff_vl, c.stuff_vr,
                       c.pitcher_role_name, c.tier_name, c.position_name,
                       c.meta_score_batting, c.meta_score_pitching, c.card_value
                FROM cards c
                WHERE c.card_id = ?
                LIMIT 1
                """,
                (resolved_card_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT c.card_id, c.card_title, c.contact, c.gap_power, c.power,
                       c.eye, c.avoid_ks, c.babip,
                       c.speed, c.stealing, c.baserunning,
                       c.contact_vl, c.contact_vr, c.power_vl, c.power_vr,
                       c.stuff, c.movement, c.control, c.p_hr, c.stamina, c.hold,
                       c.stuff_vl, c.stuff_vr,
                       c.pitcher_role_name, c.tier_name, c.position_name,
                       c.meta_score_batting, c.meta_score_pitching, c.card_value
                FROM cards c
                WHERE c.card_title LIKE ?
                ORDER BY c.owned DESC LIMIT 1
                """,
                (f'%{target_name}%',),
            ).fetchone()
        if row:
            extra['card'] = {k: row[k] for k in row.keys() if row[k] is not None}
    except Exception as e:
        logger.debug("card enrich failed: %s", e)

    # Observed stats — latest PA-weighted row per card_id (safer than LIKE)
    try:
        card_id_for_stats = resolved_card_id or (extra.get('card') or {}).get('card_id')
        if card_id_for_stats:
            bs = conn.execute(
                """
                SELECT bs.war, bs.pa, bs.ops, bs.ops_plus, bs.obp, bs.slg,
                       bs.iso, bs.babip AS obs_babip, bs.league_id
                FROM batting_stats bs
                WHERE bs.card_id = ? AND bs.pa >= 50
                ORDER BY bs.pa DESC LIMIT 1
                """,
                (card_id_for_stats,),
            ).fetchone()
            if bs:
                extra.setdefault('observed', {}).update(
                    {k: bs[k] for k in bs.keys() if bs[k] is not None}
                )
            ps = conn.execute(
                """
                SELECT ps.war, ps.ip, ps.era, ps.era_plus, ps.fip, ps.whip,
                       ps.k_per_9, ps.bb_per_9, ps.hr_per_9, ps.league_id
                FROM pitching_stats ps
                WHERE ps.card_id = ? AND ps.ip >= 10
                ORDER BY ps.ip DESC LIMIT 1
                """,
                (card_id_for_stats,),
            ).fetchone()
            if ps:
                extra.setdefault('observed', {}).update(
                    {k: ps[k] for k in ps.keys() if ps[k] is not None}
                )
    except Exception as e:
        logger.debug("observed enrich failed: %s", e)

    # Meta explainer
    try:
        from app.core.meta_scoring import explain_meta
        card = extra.get('card') or {}
        if card.get('card_id'):
            ex = explain_meta(card)
            if ex:
                extra['meta_explainer'] = ex
    except Exception as e:
        logger.debug("meta_explainer failed: %s", e)

    return extra


# ──────────────────────────────────────────────────────────────────────
# The main review call
# ──────────────────────────────────────────────────────────────────────

def review_recommendation(
    rec_id: int,
    *,
    primary_only: bool = False,
    force: bool = False,
) -> CouncilResult:
    """Run the council on one recommendation.

    Args:
        rec_id: primary key of the ``recommendation_log`` row.
        primary_only: if True, only call ``primary`` providers (skip
            verifier/critic escalation). Use this for quick first-pass
            verification when budget is tight.
        force: re-run even if a recent review exists for this rec.
    """
    from app.core.database import get_connection
    from app.core.llm_providers import (
        complete_many, get_providers_for_role, list_providers,
    )
    from app.core.verify_prompts import (
        SYSTEM_PREAMBLE, build_verify_prompt, parse_verdict,
    )

    conn = get_connection()
    _ensure_table(conn)

    # Pull the rec
    rec_row = conn.execute(
        "SELECT * FROM recommendation_log WHERE id = ?", (rec_id,),
    ).fetchone()
    if not rec_row:
        return CouncilResult(
            rec_id=rec_id, composite_verdict='ERROR',
            composite_confidence=None, reviews=[],
            reviewed_at=datetime.now().isoformat(timespec='seconds'),
        )

    rec = dict(rec_row)

    # Short-circuit: existing recent review
    if not force:
        existing = conn.execute(
            "SELECT COUNT(*) FROM council_review WHERE rec_id = ? "
            "AND reviewed_at > datetime('now', '-1 hour')",
            (rec_id,),
        ).fetchone()
        if existing and existing[0] > 0:
            return CouncilResult(
                rec_id=rec_id,
                composite_verdict='CACHED',
                composite_confidence=None,
                reviews=get_council_reviews(rec_id),
                reviewed_at=datetime.now().isoformat(timespec='seconds'),
            )

    # Build prompt
    extra = _enrich_rec_context(rec, conn)
    prompt = build_verify_prompt(rec, extra)

    # Primary round
    primaries = get_providers_for_role('primary')
    if not primaries:
        # Fallback: any configured provider
        primaries = list_providers()[:1]

    if not primaries:
        return CouncilResult(
            rec_id=rec_id, composite_verdict='ERROR',
            composite_confidence=None, reviews=[],
            reviewed_at=datetime.now().isoformat(timespec='seconds'),
        )

    reviews: list[ProviderReview] = []
    primary_ids = [p.id for p in primaries]
    responses = complete_many(primary_ids, prompt, system=SYSTEM_PREAMBLE)
    for p, resp in zip(primaries, responses):
        parsed = parse_verdict(resp.text or '')
        reviews.append(ProviderReview(
            provider_id=p.id, vendor=p.vendor, model=p.model, role='primary',
            verdict=parsed['verdict'] if resp.ok else 'ERROR',
            confidence=parsed['confidence'],
            reasoning=parsed['reasoning'],
            risks=parsed['risks'],
            alternative=parsed['alternative'],
            tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
            latency_s=resp.latency_s, error=resp.error,
            raw=resp.text or '',
        ))

    # Escalate to verifier if primaries disagree or have low confidence
    verdicts_set = {r.verdict for r in reviews
                    if r.verdict not in ('ERROR', 'UNKNOWN')}
    low_conf = any((r.confidence or 0) < 6 for r in reviews)
    needs_escalation = (len(verdicts_set) > 1 or low_conf or 'DISSENT' in verdicts_set)

    if needs_escalation and not primary_only:
        verifiers = [p for p in get_providers_for_role('verifier')
                     if p.id not in primary_ids]
        if verifiers:
            v_ids = [p.id for p in verifiers]
            v_responses = complete_many(v_ids, prompt, system=SYSTEM_PREAMBLE)
            for p, resp in zip(verifiers, v_responses):
                parsed = parse_verdict(resp.text or '')
                reviews.append(ProviderReview(
                    provider_id=p.id, vendor=p.vendor, model=p.model,
                    role='verifier',
                    verdict=parsed['verdict'] if resp.ok else 'ERROR',
                    confidence=parsed['confidence'],
                    reasoning=parsed['reasoning'],
                    risks=parsed['risks'],
                    alternative=parsed['alternative'],
                    tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                    latency_s=resp.latency_s, error=resp.error,
                    raw=resp.text or '',
                ))

    # Persist
    reviewed_at = datetime.now().isoformat(timespec='seconds')
    for r in reviews:
        conn.execute(
            """
            INSERT INTO council_review
                (rec_id, provider_id, vendor, model, role, verdict, confidence,
                 reasoning, risks, alternative, tokens_in, tokens_out,
                 latency_s, error, raw, reviewed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (rec_id, r.provider_id, r.vendor, r.model, r.role, r.verdict,
             r.confidence, r.reasoning, r.risks, r.alternative,
             r.tokens_in, r.tokens_out, r.latency_s, r.error, r.raw,
             reviewed_at),
        )
    conn.commit()

    composite, composite_conf = composite_verdict(reviews)
    # Store the composite on the rec for quick UI lookup
    try:
        conn.execute(
            "UPDATE recommendation_log SET reasoning = COALESCE(reasoning,'') "
            "|| ' | council=' || ? WHERE id = ?",
            (composite, rec_id),
        )
        conn.commit()
    except Exception:
        pass

    return CouncilResult(
        rec_id=rec_id, composite_verdict=composite,
        composite_confidence=composite_conf, reviews=reviews,
        reviewed_at=reviewed_at,
    )


def composite_verdict(reviews: list[ProviderReview]) -> tuple[str, Optional[float]]:
    """Aggregate per-provider verdicts into one label + avg confidence."""
    valid = [r for r in reviews
             if r.verdict not in ('ERROR', 'UNKNOWN')]
    if not valid:
        return 'ERROR', None
    concur = sum(1 for r in valid if r.verdict == 'CONCUR')
    dissent = sum(1 for r in valid if r.verdict == 'DISSENT')
    needs = sum(1 for r in valid if r.verdict == 'NEEDS_MORE_INFO')
    total = len(valid)

    # Unanimous
    if concur == total:
        label = 'CONCUR'
    elif dissent == total:
        label = 'DISSENT'
    elif needs == total:
        label = 'NEEDS_MORE_INFO'
    else:
        # Split verdict — majority rules, but mark MIXED if not decisive
        if concur > dissent and concur >= total * 0.6:
            label = 'CONCUR'
        elif dissent > concur and dissent >= total * 0.6:
            label = 'DISSENT'
        else:
            label = 'MIXED'

    confidences = [r.confidence for r in valid if r.confidence is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    return label, avg_conf


def review_many(rec_ids: list[int]) -> list[CouncilResult]:
    """Run council on multiple recs sequentially.

    Each rec's providers run in parallel internally, but recs run
    sequentially to keep total API burst polite.
    """
    return [review_recommendation(rid) for rid in rec_ids]


def get_council_reviews(rec_id: int) -> list[ProviderReview]:
    """Fetch persisted reviews for a rec (newest first)."""
    from app.core.database import get_connection
    conn = get_connection()
    _ensure_table(conn)
    rows = conn.execute(
        """
        SELECT * FROM council_review
        WHERE rec_id = ?
        ORDER BY reviewed_at DESC, id DESC
        """,
        (rec_id,),
    ).fetchall()
    return [
        ProviderReview(
            provider_id=r['provider_id'], vendor=r['vendor'] or '',
            model=r['model'] or '', role=r['role'] or '',
            verdict=r['verdict'] or 'UNKNOWN',
            confidence=r['confidence'],
            reasoning=r['reasoning'] or '',
            risks=r['risks'] or '',
            alternative=r['alternative'],
            tokens_in=r['tokens_in'] or 0,
            tokens_out=r['tokens_out'] or 0,
            latency_s=r['latency_s'] or 0.0,
            error=r['error'],
            raw=r['raw'] or '',
        )
        for r in rows
    ]


# ──────────────────────────────────────────────────────────────────────
# Chat follow-up on a specific rec
# ──────────────────────────────────────────────────────────────────────

def chat_with_rec(
    rec_id: int,
    user_message: str,
    history: Optional[list[dict]] = None,
    provider_id: Optional[str] = None,
) -> str:
    """Threaded chat about a specific recommendation.

    ``history`` is an optional list of prior turns: [{'role': 'user|assistant', 'content': '…'}].
    Returns the assistant's reply text.
    """
    from app.core.database import get_connection
    from app.core.llm_providers import (
        complete, get_providers_for_role, list_providers,
    )
    from app.core.verify_prompts import SYSTEM_PREAMBLE, build_verify_prompt

    conn = get_connection()
    rec = conn.execute(
        "SELECT * FROM recommendation_log WHERE id = ?", (rec_id,),
    ).fetchone()
    if not rec:
        return f"Recommendation #{rec_id} not found."
    rec_dict = dict(rec)

    extra = _enrich_rec_context(rec_dict, conn)
    rec_context = build_verify_prompt(rec_dict, extra)

    # Concatenate history into a single prompt
    parts = [f"ORIGINAL RECOMMENDATION & CONTEXT:\n{rec_context}",
             "\n\nPRIOR CONVERSATION:"]
    for turn in (history or []):
        who = turn.get('role', 'user').upper()
        parts.append(f"{who}: {turn.get('content','')}")
    parts.append(f"\nUSER: {user_message}\nASSISTANT:")
    prompt = '\n'.join(parts)

    # Pick provider — prefer chat role, fall back to primary
    if provider_id:
        pid = provider_id
    else:
        chat_ps = get_providers_for_role('chat')
        if chat_ps:
            pid = chat_ps[0].id
        else:
            ps = get_providers_for_role('primary') or list_providers()
            pid = ps[0].id if ps else ''

    if not pid:
        return "No LLM provider configured."

    resp = complete(pid, prompt, system=SYSTEM_PREAMBLE,
                    max_tokens=1024, temperature=0.5)
    if resp.error:
        return f"(LLM error: {resp.error})"
    return resp.text or '(empty response)'
