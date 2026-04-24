"""Mini player-card text for use in table tooltips.

Given a player name (or card_id), assemble a multi-line text block that
looks like a compact baseball card. Used in the chain table tooltips so
the user can inspect a recommended card's ratings, observed stats,
overlay contributions, clutch history, and LLM verdict in one hover.

The output is plain text with newlines — the tooltip_html helper
converts \\n to &#10; so browsers render the multi-line title natively.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _fmt_num(v, digits=3) -> str:
    if v is None:
        return '-'
    try:
        return f'{float(v):.{digits}f}'
    except (TypeError, ValueError):
        return str(v)


def _fmt_int(v) -> str:
    if v is None:
        return '-'
    try:
        return f'{int(v):,}'
    except (TypeError, ValueError):
        return str(v)


def _fetch_card(conn, name_or_id) -> Optional[dict]:
    """Look up a card by id (int) or name (substring LIKE match)."""
    if name_or_id is None:
        return None
    try:
        if isinstance(name_or_id, int) or str(name_or_id).isdigit():
            row = conn.execute(
                "SELECT * FROM cards WHERE card_id = ? LIMIT 1",
                (int(name_or_id),),
            ).fetchone()
        else:
            # Prefer owned cards when a name collides across instances
            row = conn.execute(
                """
                SELECT * FROM cards
                WHERE card_title LIKE ?
                ORDER BY owned DESC, card_value DESC
                LIMIT 1
                """,
                (f'%{name_or_id}%',),
            ).fetchone()
        return dict(row) if row else None
    except Exception as e:
        logger.debug("card lookup failed for %r: %s", name_or_id, e)
        return None


def _fetch_observed_batting(conn, card_id) -> Optional[dict]:
    if not card_id:
        return None
    try:
        row = conn.execute(
            """
            SELECT bs.pa, bs.ab, bs.hits, bs.hr, bs.rbi, bs.bb, bs.k,
                   bs.avg, bs.obp, bs.slg, bs.ops, bs.ops_plus, bs.babip,
                   bs.iso, bs.war, bs.league_id
            FROM batting_stats bs
            WHERE bs.card_id = ?
            ORDER BY bs.pa DESC LIMIT 1
            """,
            (card_id,),
        ).fetchone()
        return dict(row) if row else None
    except Exception:
        return None


def _fetch_observed_pitching(conn, card_id) -> Optional[dict]:
    if not card_id:
        return None
    try:
        row = conn.execute(
            """
            SELECT ps.ip, ps.games, ps.gs, ps.saves, ps.wins, ps.losses,
                   ps.era, ps.era_plus, ps.fip, ps.whip,
                   ps.k_per_9, ps.bb_per_9, ps.hr_per_9,
                   ps.babip, ps.war, ps.league_id
            FROM pitching_stats ps
            WHERE ps.card_id = ?
            ORDER BY ps.ip DESC LIMIT 1
            """,
            (card_id,),
        ).fetchone()
        return dict(row) if row else None
    except Exception:
        return None


def _fetch_clutch(conn, card_id) -> dict:
    if not card_id:
        return {}
    out = {}
    try:
        rows = conn.execute(
            """
            SELECT event_type, SUM(event_count) AS n
            FROM game_clutch_events
            WHERE card_id = ?
            GROUP BY event_type
            """,
            (card_id,),
        ).fetchall()
        for r in rows:
            out[r['event_type']] = int(r['n'] or 0)
    except Exception:
        pass
    return out


def _fetch_overlays(conn, card_id, is_pitcher: bool = False) -> dict:
    """Rough inverse of the overlay formulas — approximate each overlay's
    contribution to the meta score for display purposes only.

    Skips batting overlays for pitchers and vice versa to avoid showing
    irrelevant meta contributions (e.g. FIP on a catcher).
    """
    if not card_id:
        return {}
    out: dict = {}
    if not is_pitcher:
        try:
            # OPS+ delta vs 100
            bs = conn.execute(
                """
                SELECT AVG(ops_plus) AS op, SUM(pa) AS pa
                FROM batting_stats
                WHERE card_id = ? AND ops_plus IS NOT NULL AND pa > 0
                """,
                (card_id,),
            ).fetchone()
            if bs and bs['op'] is not None:
                delta = float(bs['op']) - 100.0
                out['OPS+'] = f"{delta:+.0f} × 0.6 ≈ {delta*0.6:+.1f} meta"
        except Exception:
            pass
    else:
        try:
            # FIP delta vs league (pitchers only)
            ps = conn.execute(
                """
                SELECT AVG(ps.fip) AS fip,
                       (SELECT AVG(fip) FROM pitching_stats
                        WHERE league_id = ps.league_id) AS lg_fip
                FROM pitching_stats ps
                WHERE card_id = ? AND fip IS NOT NULL
                """,
                (card_id,),
            ).fetchone()
            if ps and ps['fip'] is not None and ps['lg_fip']:
                delta = float(ps['fip']) - float(ps['lg_fip'])
                out['FIP'] = f"{delta:+.2f} × -25 ≈ {-delta*25:+.1f} meta"
        except Exception:
            pass
    return out


def _fetch_council_verdict(conn, card_title: str,
                            rec_type: Optional[str] = None) -> Optional[dict]:
    """Find the most-recent rec + its council verdict for this card."""
    if not card_title:
        return None
    try:
        sql = """
            SELECT rl.id, rl.rec_type, rl.verdict AS rl_verdict,
                   rl.expected_delta, rl.pos
            FROM recommendation_log rl
            WHERE COALESCE(rl.player_name,'') LIKE ?
              AND rl.created_at > datetime('now', '-30 days')
        """
        params: list = [f'%{card_title}%']
        if rec_type:
            sql += " AND rl.rec_type = ?"
            params.append(rec_type)
        sql += " ORDER BY rl.created_at DESC LIMIT 1"
        rl = conn.execute(sql, params).fetchone()
        if not rl:
            return None
        reviews = conn.execute(
            """
            SELECT provider_id, model, role, verdict, confidence, reasoning
            FROM council_review
            WHERE rec_id = ?
            ORDER BY reviewed_at DESC, id DESC
            """,
            (rl['id'],),
        ).fetchall()
        return {
            'rec_id': rl['id'],
            'rec_type': rl['rec_type'],
            'expected_delta': rl['expected_delta'],
            'reviews': [dict(r) for r in reviews],
        }
    except Exception as e:
        logger.debug("council verdict fetch failed: %s", e)
        return None


def build_player_card_tooltip(
    conn,
    name_or_id,
    *,
    rec_type: Optional[str] = None,
    header_prefix: Optional[str] = None,
) -> str:
    """Assemble a mini-card tooltip string for a player.

    Args:
        conn: sqlite connection.
        name_or_id: card_title substring or card_id int.
        rec_type: if provided, look up a matching recommendation_log row so
            the council verdict is included.
        header_prefix: optional first line (e.g. "🛒 +100 meta upgrade").
    """
    card = _fetch_card(conn, name_or_id)
    if not card:
        # Fall back to a minimal line so the tooltip isn't empty
        bits = []
        if header_prefix:
            bits.append(header_prefix)
        bits.append(str(name_or_id))
        return '\n'.join(bits)

    lines: list[str] = []
    if header_prefix:
        lines.append(header_prefix)

    # Header — full title + card_type + tier + age
    title = card.get('card_title') or '(unknown)'
    tier = card.get('tier_name') or ''
    ct = card.get('card_type_name') or ''
    age = card.get('age')
    pos = card.get('position_name') or ''
    role = card.get('pitcher_role_name') or ''
    meta_b = card.get('meta_score_batting') or 0
    meta_p = card.get('meta_score_pitching') or 0
    ovr = card.get('card_value') or 0

    header_bits = [title]
    meta_line_bits = []
    if tier: meta_line_bits.append(tier)
    if ct: meta_line_bits.append(ct)
    if age: meta_line_bits.append(f'age {age}')
    if pos and not role: meta_line_bits.append(pos)
    if role: meta_line_bits.append(role)
    lines.append(title)
    if meta_line_bits:
        lines.append(' · '.join(meta_line_bits))
    lines.append(f'OVR {ovr} · Meta {meta_b or meta_p}')

    # Batting ratings
    is_pitcher = bool(card.get('pitcher_role'))
    if not is_pitcher:
        r_bits = []
        for key, label in [('contact','CON'), ('gap_power','GAP'),
                           ('power','PWR'), ('eye','EYE'),
                           ('avoid_ks','AVK'), ('babip','BABIP')]:
            v = card.get(key)
            if v is not None:
                r_bits.append(f'{label} {int(v)}')
        if r_bits:
            lines.append('Ratings: ' + ' · '.join(r_bits))
        # Platoon splits
        splits = []
        for side, keys in (('vL', [('contact_vl','CON'),('power_vl','PWR'),('eye_vl','EYE')]),
                           ('vR', [('contact_vr','CON'),('power_vr','PWR'),('eye_vr','EYE')])):
            parts = []
            for k, lbl in keys:
                v = card.get(k)
                if v is not None:
                    parts.append(f'{lbl} {int(v)}')
            if parts:
                splits.append(f'{side}: {" ".join(parts)}')
        if splits:
            lines.append('Splits: ' + ' | '.join(splits))
        # Speed/running
        sbb = []
        for k, lbl in (('speed','SPD'),('stealing','STL'),('baserunning','BR')):
            v = card.get(k)
            if v is not None:
                sbb.append(f'{lbl} {int(v)}')
        if sbb:
            lines.append('Speed: ' + ' · '.join(sbb))
    else:
        # Pitching ratings
        p_bits = []
        for key, label in [('stuff','STU'),('movement','MOV'),('control','CTL'),
                           ('p_hr','HRx'),('stamina','STA'),('hold','HLD')]:
            v = card.get(key)
            if v is not None:
                p_bits.append(f'{label} {int(v)}')
        if p_bits:
            lines.append('Ratings: ' + ' · '.join(p_bits))
        # Platoon splits
        splits = []
        for side, keys in (('vL', [('stuff_vl','STU'),('movement_vl','MOV'),('control_vl','CTL')]),
                           ('vR', [('stuff_vr','STU'),('movement_vr','MOV'),('control_vr','CTL')])):
            parts = []
            for k, lbl in keys:
                v = card.get(k)
                if v is not None:
                    parts.append(f'{lbl} {int(v)}')
            if parts:
                splits.append(f'{side}: {" ".join(parts)}')
        if splits:
            lines.append('Splits: ' + ' | '.join(splits))
        # Pitch arsenal
        pitches = []
        for k in ('fb','sl','cb','ch','si','sp','ct','fo','cc','sc','kc','kn'):
            v = card.get(k)
            if v is not None and int(v) > 0:
                pitches.append(f'{k.upper()} {int(v)}')
        if pitches:
            lines.append('Arsenal: ' + ' · '.join(pitches[:6]))

    # Observed stats
    card_id = card.get('card_id')
    if not is_pitcher:
        obs = _fetch_observed_batting(conn, card_id)
        if obs and obs.get('pa'):
            avg = _fmt_num(obs.get('avg'))
            obp = _fmt_num(obs.get('obp'))
            slg = _fmt_num(obs.get('slg'))
            iso = _fmt_num(obs.get('iso'))
            war = obs.get('war') or 0
            pa = obs.get('pa') or 1
            war600 = (float(war) * 600.0 / pa) if pa else 0
            lines.append(
                f"Obs ({obs.get('league_id','?')} · {pa} PA): "
                f"{avg}/{obp}/{slg} · OPS+ {obs.get('ops_plus','?')} "
                f"· ISO {iso} · WAR/600 {war600:.2f}"
            )
    else:
        obs = _fetch_observed_pitching(conn, card_id)
        if obs and obs.get('ip'):
            lines.append(
                f"Obs ({obs.get('league_id','?')} · {obs.get('ip')} IP): "
                f"ERA {_fmt_num(obs.get('era'),2)} · ERA+ {obs.get('era_plus','?')} "
                f"· FIP {_fmt_num(obs.get('fip'),2)} · WHIP {_fmt_num(obs.get('whip'),2)} "
                f"· K/9 {_fmt_num(obs.get('k_per_9'),1)} · BB/9 {_fmt_num(obs.get('bb_per_9'),1)}"
            )

    # Overlay contributions — only the side relevant to this player's role
    overlays = _fetch_overlays(conn, card_id, is_pitcher=is_pitcher)
    if overlays:
        lines.append('Overlays: ' + ' · '.join(
            f'{k}={v}' for k, v in overlays.items()
        ))

    # Clutch events
    clutch = _fetch_clutch(conn, card_id)
    if clutch:
        ch_bits = []
        for k in ('2OUT_RBI', 'LOB_RISP_2OUT', 'HR', 'ERROR',
                  'INHERITED_RUNNERS', 'INHERITED_SCORED'):
            if k in clutch:
                ch_bits.append(f'{k.replace("_"," ").lower()} {clutch[k]}')
        if ch_bits:
            lines.append('Clutch: ' + ' · '.join(ch_bits))

    # Silent LLM verification is folded back into the ENGINE's reasoning
    # rather than surfaced as a separate "council" callout. When a verdict
    # concurs it's invisible (engine pick stands); when it dissents, we
    # append a concise "⚠" risk-line so the user sees WHY not a verdict.
    verdict = _fetch_council_verdict(conn, title, rec_type=rec_type)
    if verdict and verdict.get('reviews'):
        revs = verdict['reviews']
        # Only surface if a provider dissented or flagged a concrete risk.
        dissents = [r for r in revs if r.get('verdict') == 'DISSENT']
        risks = [r for r in revs
                 if (r.get('reasoning') or '').strip() and r.get('confidence')
                 and int(r['confidence']) < 6]
        if dissents:
            r0 = dissents[0]
            msg = (r0.get('reasoning') or '').strip()
            if len(msg) > 180:
                msg = msg[:180] + '…'
            if msg:
                lines.append(f'⚠ engine review flagged a risk — {msg}')
        elif risks:
            r0 = risks[0]
            msg = (r0.get('reasoning') or '').strip()
            if len(msg) > 180:
                msg = msg[:180] + '…'
            if msg:
                lines.append(f'⚠ low confidence — {msg}')
        # Concurrences and pending reviews render no extra line.

    return '\n'.join(lines)
