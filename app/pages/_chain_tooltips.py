"""Tooltip builders for the Roster Optimizer chain tables.

Centralizes the per-cell tooltip logic so the Roster Optimizer page stays
readable. Each cell tooltip pulls from:

    * The row dict itself (rating summaries already computed for display).
    * The recommendation_log (latest rec row for this slot + rec_type).
    * The council_review table (latest verdict per provider on that rec).

The LLM verdict is rendered INLINE in the tooltip text — never in a
separate column — so there is one unified recommendation per slot.
"""
from __future__ import annotations

from typing import Callable, Optional


def chain_header_help(col_config: dict) -> dict[str, str]:
    """Extract the `help` text from Streamlit ColumnConfig objects.

    Streamlit exposes TextColumn config as dataclass-adjacent objects;
    the help text lives in a private ``_help`` attr on the underlying
    config. We fall back through several known storage locations.
    """
    out: dict[str, str] = {}
    for col, cfg in (col_config or {}).items():
        if cfg is None:
            continue
        help_txt = ''
        # Streamlit 1.3x+ stores TextColumn config as a typed dict internally
        try:
            # Try the public attribute
            help_txt = getattr(cfg, 'help', '') or ''
        except Exception:
            pass
        if not help_txt:
            try:
                d = getattr(cfg, '__dict__', {}) or {}
                help_txt = d.get('help') or d.get('_help') or ''
            except Exception:
                pass
        if not help_txt and isinstance(cfg, dict):
            help_txt = cfg.get('help') or cfg.get('_help') or ''
        if help_txt:
            out[col] = str(help_txt)
    return out


def _latest_rec_and_verdict(conn, pos: str, rec_type: str,
                             target_name: Optional[str]) -> Optional[dict]:
    """Pull the latest rec for this (pos, rec_type, target_name) + composite
    council verdict if one exists.

    Returns None if no matching rec. Otherwise a dict with:
        rec_id, verdict (composite), confidence (avg), per_provider[]
    """
    if not pos or not rec_type:
        return None
    try:
        # Latest rec matching pos/rec_type — optionally the target_name
        sql = ("SELECT * FROM recommendation_log "
               "WHERE pos = ? AND rec_type = ? "
               "  AND created_at > datetime('now', '-30 days')")
        params: list = [pos, rec_type]
        if target_name:
            sql += " AND COALESCE(player_name,'') LIKE ?"
            params.append(f"%{target_name}%")
        sql += " ORDER BY created_at DESC LIMIT 1"
        row = conn.execute(sql, params).fetchone()
    except Exception:
        return None
    if not row:
        return None
    rec_id = row['id']

    # Pull all council_review rows for this rec
    providers: list[dict] = []
    composite = None
    avg_conf = None
    try:
        rev_rows = conn.execute(
            "SELECT * FROM council_review WHERE rec_id = ? "
            "ORDER BY reviewed_at DESC, id DESC", (rec_id,)
        ).fetchall()
        for rr in rev_rows:
            providers.append({
                'provider_id': rr['provider_id'],
                'model': rr['model'], 'role': rr['role'],
                'verdict': rr['verdict'], 'confidence': rr['confidence'],
                'reasoning': (rr['reasoning'] or '')[:280],
            })
        # Composite = majority vote
        valid = [p for p in providers if p['verdict'] in ('CONCUR','DISSENT','NEEDS_MORE_INFO')]
        if valid:
            c = sum(1 for p in valid if p['verdict'] == 'CONCUR')
            d = sum(1 for p in valid if p['verdict'] == 'DISSENT')
            if c > d and c >= len(valid) * 0.6:
                composite = 'CONCUR'
            elif d > c and d >= len(valid) * 0.6:
                composite = 'DISSENT'
            elif any(p['verdict'] == 'NEEDS_MORE_INFO' for p in valid):
                composite = 'NEEDS_MORE_INFO'
            else:
                composite = 'MIXED'
            confs = [p['confidence'] for p in valid if p['confidence'] is not None]
            avg_conf = sum(confs) / len(confs) if confs else None
    except Exception:
        pass

    return {
        'rec_id': rec_id, 'verdict': composite, 'confidence': avg_conf,
        'providers': providers,
        'expected_delta': row['expected_delta'],
        'reasoning': row['reasoning'],
    }


def _verdict_line(verdict_info: Optional[dict]) -> str:
    """Surface LLM verdict ONLY when the engine's pick is contested.

    Concurrences, pending reviews, and no-review cases are silent — the
    engine's recommendation speaks for itself. A dissent or a low-confidence
    provider opinion becomes a one-line risk hint appended to the tooltip.
    """
    if not verdict_info:
        return ''
    v = verdict_info.get('verdict') or ''
    providers = verdict_info.get('providers') or []
    dissents = [p for p in providers if p.get('verdict') == 'DISSENT']
    if dissents:
        reason = (dissents[0].get('reasoning') or '').strip()
        if len(reason) > 180:
            reason = reason[:180] + '…'
        return f'⚠ engine review flagged a risk — {reason}' if reason else ''
    # Low-confidence concurrences: still worth a quiet note
    low_conf = [p for p in providers
                if p.get('verdict') == 'CONCUR'
                and p.get('confidence') is not None
                and int(p['confidence']) < 6]
    if low_conf:
        reason = (low_conf[0].get('reasoning') or '').strip()
        if len(reason) > 140:
            reason = reason[:140] + '…'
        return f'⚠ low confidence — {reason}' if reason else ''
    return ''


def chain_cell_tooltip(conn) -> dict[str, Callable[[dict], str]]:
    """Return the cell_tooltip dict for render_tooltip_table.

    The Current / Owned Promotion / Market Upgrade cells use the rich
    mini-player-card builder so hovers show ratings, observed stats,
    overlay contributions, clutch events, and LLM verdict in one shot.
    """
    from app.core.player_card_tooltip import build_player_card_tooltip

    def _strip_display_prefix(s: str) -> str:
        """Strip '📦 +60 · ', '🛒 +100 · … · 1,234 PP' etc. to get a name."""
        if not s:
            return ''
        # Remove leading emoji + delta (e.g. "📦 +60 · ")
        import re
        # Strip leading non-alphanum run (emoji + whitespace + +delta)
        s = re.sub(r'^[^\w]*[+\-]?\d+\s*[·•]\s*', '', s).strip()
        # Strip trailing ' · 1,234 PP' cost suffix
        s = re.sub(r'\s*[·•]\s*[\d,]+\s*PP.*$', '', s).strip()
        # Strip trailing ⚠ and " riding hot"/"still cold" tag words
        s = re.sub(r'\s*⚠.*$', '', s).strip()
        s = re.sub(r'\s*·\s*(riding hot|still cold|Optimal|cold|hot).*$', '',
                   s, flags=re.IGNORECASE).strip()
        return s

    def current_tt(r: dict) -> str:
        name = r.get('_current_card_title') or r.get('_full_card_title') \
               or r.get('Current') or ''
        # Strip the bats-hand suffix "(R)" / "(L)" / "(S)" for lookup
        import re
        lookup = re.sub(r'\s*\([RLS]\)\s*$', '', name).strip()
        if not lookup:
            return name
        return build_player_card_tooltip(conn, lookup)

    def owned_tt(r: dict) -> str:
        raw = r.get('_owned_target_name') or r.get('Owned Promotion') or ''
        clean = _strip_display_prefix(raw)
        if not clean or clean.startswith('Optimal') or 'Optimal' == raw.strip():
            # No upgrade — show why
            current = r.get('Current') or ''
            return (f"\U0001f4e6 No owned promotion beats your current starter.\n"
                    f"Current: {current}")
        header = f"\U0001f4e6 Owned promotion candidate"
        delta = r.get('_owned_delta')
        if delta:
            header += f" · +{int(delta)} meta"
        return build_player_card_tooltip(
            conn, clean, rec_type='promote', header_prefix=header,
        )

    def market_tt(r: dict) -> str:
        raw = r.get('_market_target_name') or r.get('Market Upgrade') or ''
        clean = _strip_display_prefix(raw)
        if not clean or 'Optimal' in raw[:12]:
            current = r.get('Current') or ''
            return (f"\U0001f6d2 No market buy meaningfully upgrades this slot.\n"
                    f"Current: {current}")
        header = f"\U0001f6d2 Market upgrade candidate"
        delta = r.get('_market_delta')
        price = r.get('_market_price')
        if delta:
            header += f" · +{int(delta)} meta"
        if price:
            header += f" · {int(price):,} PP"
        return build_player_card_tooltip(
            conn, clean, rec_type='buy', header_prefix=header,
        )

    def ovr_tt(r: dict) -> str:
        return (
            "OVR is OOTP's built-in overall rating (diagnostic column).\n"
            f"Meta score for this card: {r.get('Meta','?')}\n"
            "Meta beats OVR for WAR prediction across lb124 + i76 "
            "(Δ≈+0.20 r on batting). When they disagree, trust Meta."
        )

    def meta_tt(r: dict) -> str:
        bd = r.get('_meta_breakdown')
        if bd:
            return bd
        return (
            f"Meta: {r.get('Meta')}\n"
            "Composed of rating base + platoon adj + performance overlay "
            "(wOBA) + OPS+/OBP/ISO/BABIP overlays + clutch + diminishing."
        )

    def conf_tt(r: dict) -> str:
        return (
            r.get('_confidence_breakdown')
            or "Confidence (0-100). Composition is inline in the cell.\n"
               "Drivers: pooled PA, # of team instances, OPS+ std-dev "
               "across instances, game-log sample size."
        )

    def status_tt(r: dict) -> str:
        return (
            r.get('_status_breakdown')
            or "Format: rate stat · outlook · regression.\n"
               "Outlook logic: WAR-weighted when PA is stabilized; "
               "OPS+-driven for small samples. Open Performance Outlook "
               "expander below for drivers."
        )

    # NOTE: the rich player-card versions of owned_tt / market_tt are
    # defined earlier in this function. The dict below binds to those
    # (closures capture by name at call time, so both versions existed
    # historically — the duplicates were removed).

    return {
        'Current': current_tt,
        'OVR': ovr_tt,
        'Meta': meta_tt,
        'Confidence': conf_tt,
        'Status': status_tt,
        'Owned Promotion': owned_tt,
        'Market Upgrade': market_tt,
    }
