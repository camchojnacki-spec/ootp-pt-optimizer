"""AI-powered advisor for OOTP 27 Perfect Team — uses Gemini or Anthropic API for strategic analysis."""
import os
import logging

logger = logging.getLogger(__name__)


def get_ai_config() -> dict:
    """Get AI provider configuration.

    Returns dict with:
        provider: "gemini" or "anthropic"
        ready: True if enough config to make API calls
        message: status/error message
        (plus provider-specific keys)
    """
    from app.core.database import load_config
    try:
        config = load_config()
    except Exception:
        config = {}

    provider = config.get("ai_provider", "gemini")

    if provider == "gemini":
        gemini_config = config.get("gemini", {})
        api_key = gemini_config.get("api_key", "")
        model = gemini_config.get("model", "gemini-2.0-flash")

        if not api_key:
            return {
                "provider": "gemini",
                "api_key": None,
                "model": model,
                "ready": False,
                "message": "Gemini API key not set. Add it to config.yaml under gemini.api_key",
            }
        return {
            "provider": "gemini",
            "api_key": api_key,
            "model": model,
            "ready": True,
            "message": f"Google Gemini ({model})",
        }
    else:
        # Direct Anthropic API
        api_key = os.environ.get("ANTHROPIC_API_KEY") or config.get("anthropic_api_key")
        if not api_key:
            return {
                "provider": "anthropic",
                "api_key": None,
                "ready": False,
                "message": "No API key. Set ANTHROPIC_API_KEY env var or add anthropic_api_key to config.yaml",
            }
        return {
            "provider": "anthropic",
            "api_key": api_key,
            "ready": True,
            "message": "Anthropic API (direct)",
        }


def get_api_key() -> str | None:
    """Legacy helper — returns API key if provider is ready."""
    ai_config = get_ai_config()
    if ai_config["ready"]:
        return ai_config.get("api_key")
    return None


def _build_portfolio_context(conn) -> str:
    """Build a concise text summary of the user's current portfolio state.

    This becomes the context window for the AI advisor prompt (~500-800 words).
    """
    from app.core.database import load_config, get_connection
    from app.core.roster_analysis import get_roster_summary, get_position_strength
    from app.core.recommendations import get_buy_recommendations, get_sell_recommendations
    from app.core.mission_tracker import get_mission_summary
    from app.core.flip_finder import get_flip_summary
    from app.core.price_analysis import get_biggest_movers

    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    sections = []

    # --- Team & Budget ---
    try:
        config = load_config()
        team_name = config.get("team_name", "Toronto Dark Knights")
        budget = config.get("pp_budget", 500)
        sections.append(f"TEAM: {team_name} | PP Budget: {budget:,}")
    except Exception:
        sections.append("TEAM: Toronto Dark Knights | PP Budget: unknown")

    # --- Roster Summary ---
    try:
        pos_strength = get_position_strength(conn)
        roster_lines = []
        total_meta = 0
        weakest_pos = None
        weakest_meta = float("inf")

        for pos in ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "SP", "RP", "CL"]:
            info = pos_strength.get(pos, {})
            player = info.get("player", "EMPTY")
            meta = info.get("meta_score", 0) or 0
            strength = info.get("strength", "empty")
            roster_lines.append(f"  {pos}: {player} (meta {meta:.0f}, {strength})")
            total_meta += meta
            if meta < weakest_meta:
                weakest_meta = meta
                weakest_pos = pos

        sections.append("ROSTER:")
        sections.extend(roster_lines)
        sections.append(f"Total roster meta: {total_meta:.0f} | Weakest: {weakest_pos} ({weakest_meta:.0f})")
    except Exception as e:
        sections.append(f"ROSTER: Error loading — {e}")

    # --- Buy Recommendations ---
    try:
        buy_recs = get_buy_recommendations(conn, limit=5)
        if buy_recs:
            sections.append("TOP BUY RECOMMENDATIONS:")
            for r in buy_recs:
                sections.append(
                    f"  {r['card_title']} ({r['position']}) — meta {r['meta_score']:.0f}, "
                    f"~{r['estimated_price']:,} PP | {r['reason']}"
                )
        else:
            sections.append("TOP BUY RECOMMENDATIONS: None generated yet.")
    except Exception:
        sections.append("TOP BUY RECOMMENDATIONS: Unavailable.")

    # --- Sell Recommendations ---
    try:
        sell_recs = get_sell_recommendations(conn, limit=5)
        if sell_recs:
            sections.append("TOP SELL RECOMMENDATIONS:")
            for r in sell_recs:
                sections.append(
                    f"  {r['card_title']} ({r['position']}) — ~{r['estimated_price']:,} PP | {r['reason']}"
                )
        else:
            sections.append("TOP SELL RECOMMENDATIONS: None generated yet.")
    except Exception:
        sections.append("TOP SELL RECOMMENDATIONS: Unavailable.")

    # --- In-Game Performance Stats ---
    try:
        from app.core.meta_validation import get_stats_summary
        stats_sum = get_stats_summary(conn)
        if stats_sum["has_batting_stats"]:
            parts = [f"IN-GAME STATS: {stats_sum['batting_count']} batters, {stats_sum['pitching_count']} pitchers tracked"]
            if stats_sum["team_avg"]:
                parts.append(f"Team AVG: .{int(stats_sum['team_avg']*1000):03d}, OPS: {stats_sum['team_ops']:.3f}")
            if stats_sum["team_era"]:
                parts.append(f"Team ERA: {stats_sum['team_era']:.2f}")
            mvp = stats_sum.get("mvp")
            if mvp:
                parts.append(f"MVP: {mvp['player_name']} ({mvp['war']:.1f} WAR, {mvp['ops']:.3f} OPS)")
            cy = stats_sum.get("cy_young")
            if cy:
                parts.append(f"Cy Young: {cy['player_name']} ({cy['war']:.1f} WAR, {cy['era']:.2f} ERA)")
            sections.append(" | ".join(parts))

            # Top/bottom performers by WAR
            top_war = conn.execute(
                "SELECT player_name, war, ops FROM batting_stats WHERE ab >= 50 ORDER BY war DESC LIMIT 3"
            ).fetchall()
            bot_war = conn.execute(
                "SELECT player_name, war, ops FROM batting_stats WHERE ab >= 50 ORDER BY war ASC LIMIT 3"
            ).fetchall()
            if top_war:
                sections.append("TOP WAR BATTERS: " + ", ".join(
                    f"{r['player_name']} ({r['war']:.1f})" for r in top_war
                ))
            if bot_war:
                sections.append("WORST WAR BATTERS: " + ", ".join(
                    f"{r['player_name']} ({r['war']:.1f})" for r in bot_war
                ))
        else:
            sections.append("IN-GAME STATS: Not imported yet.")
    except Exception:
        sections.append("IN-GAME STATS: Unavailable.")

    # --- Price Alerts ---
    try:
        alerts = conn.execute(
            "SELECT card_title, target_price, direction FROM price_alerts WHERE triggered = 1"
        ).fetchall()
        if alerts:
            sections.append("TRIGGERED PRICE ALERTS:")
            for a in alerts:
                sections.append(f"  {a['card_title']} hit {a['direction']} target {a['target_price']:,} PP")
        else:
            sections.append("PRICE ALERTS: None triggered.")
    except Exception:
        # Table may not exist
        sections.append("PRICE ALERTS: Not configured.")

    # --- Live Card Analysis ---
    try:
        table_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='live_card_cache'"
        ).fetchone()
        if table_check:
            upgrades = conn.execute(
                "SELECT COUNT(*) as cnt FROM live_card_cache WHERE signal = 'upgrade'"
            ).fetchone()["cnt"]
            downgrades = conn.execute(
                "SELECT COUNT(*) as cnt FROM live_card_cache WHERE signal = 'downgrade'"
            ).fetchone()["cnt"]
            sections.append(f"LIVE CARD SIGNALS: {upgrades} upgrade(s), {downgrades} downgrade(s)")
        else:
            sections.append("LIVE CARD SIGNALS: No analysis cached.")
    except Exception:
        sections.append("LIVE CARD SIGNALS: Unavailable.")

    # --- Meta Validation ---
    try:
        from app.core.meta_validation import get_meta_accuracy_score
        accuracy = get_meta_accuracy_score(conn)
        if accuracy["sample_size"] > 0:
            msg = f"META VALIDATION: {accuracy['accuracy_pct']}% accuracy ({accuracy['sample_size']} players)"
            if accuracy["top_overperformer"]:
                p = accuracy["top_overperformer"]
                msg += f" | Top overperformer: {p['player_name']}"
            if accuracy["top_underperformer"]:
                p = accuracy["top_underperformer"]
                msg += f" | Top underperformer: {p['player_name']}"
            sections.append(msg)
        else:
            sections.append(f"META VALIDATION: {accuracy['message']}")
    except Exception:
        sections.append("META VALIDATION: Unavailable.")

    # --- Mission Tracker ---
    try:
        mission = get_mission_summary(conn)
        covered = mission["teams_covered"]
        needed = mission["teams_needed"]
        cost = mission["total_cost_to_complete"]
        msg = f"MISSIONS: {covered}/30 teams covered"
        if needed:
            missing_sample = ", ".join(needed[:5])
            if len(needed) > 5:
                missing_sample += f" (+{len(needed) - 5} more)"
            msg += f" | Missing: {missing_sample} | Est. cost to complete: {cost:,} PP"
        sections.append(msg)
    except Exception:
        sections.append("MISSIONS: Unavailable.")

    # --- Recent Price Trends ---
    try:
        movers = get_biggest_movers(days=7, limit=5, conn=conn)
        if movers:
            sections.append("BIGGEST PRICE MOVERS (7d):")
            for m in movers:
                pos = m["pitcher_role_name"] or m["position_name"] or "?"
                change = m["price_change"]
                pct = m["pct_change"]
                direction = "+" if change > 0 else ""
                sections.append(
                    f"  {m['card_title']} ({pos}) — {direction}{change:,} PP ({direction}{pct}%)"
                )
        else:
            sections.append("PRICE TRENDS: No snapshot history yet.")
    except Exception:
        sections.append("PRICE TRENDS: Unavailable.")

    # --- Flip Opportunities ---
    try:
        flip_summary = get_flip_summary(conn)
        sections.append(
            f"FLIP OPPORTUNITIES: {flip_summary['spread_flip_count']} spread, "
            f"{flip_summary['volatility_play_count']} volatility, "
            f"{flip_summary['trend_play_count']} trend"
        )
    except Exception:
        sections.append("FLIP OPPORTUNITIES: Unavailable.")

    if close_conn:
        conn.close()

    return "\n".join(sections)


def _call_gemini(system_prompt: str, user_message: str, ai_config: dict,
                  max_tokens: int = 1024) -> dict:
    """Call Google Gemini API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=ai_config["api_key"])
    model = ai_config.get("model", "gemini-2.0-flash")

    resp = client.models.generate_content(
        model=model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
            max_output_tokens=max_tokens,
        ),
    )

    text = resp.text or ""
    tokens_used = resp.usage_metadata.total_token_count if resp.usage_metadata else 0

    return {
        "response": text,
        "tokens_used": tokens_used,
        "model": model,
        "error": None,
    }


def _call_anthropic(system_prompt: str, user_message: str, ai_config: dict,
                     max_tokens: int = 1024) -> dict:
    """Call Anthropic Claude API."""
    import anthropic

    model = "claude-sonnet-4-20250514"
    client = anthropic.Anthropic(api_key=ai_config["api_key"])

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.3,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    text = response.content[0].text if response.content else ""
    tokens_used = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)

    return {
        "response": text,
        "tokens_used": tokens_used,
        "model": response.model,
        "error": None,
    }


def ask_advisor(question: str, conn=None) -> dict:
    """Send a question to the AI advisor with full portfolio context.

    Args:
        question: The user's question or request.
        conn: Optional existing DB connection.

    Returns:
        Dict with keys: response, tokens_used, model, error
    """
    ai_config = get_ai_config()
    if not ai_config["ready"]:
        return {
            "response": None,
            "tokens_used": 0,
            "model": None,
            "error": ai_config["message"],
        }

    # Build context
    try:
        from app.core.database import get_connection
        close_conn = False
        if conn is None:
            conn = get_connection()
            close_conn = True

        portfolio_context = _build_portfolio_context(conn)

        if close_conn:
            conn.close()
    except Exception as e:
        portfolio_context = f"(Portfolio data unavailable: {e})"

    system_prompt = (
        "You are an expert OOTP 27 Perfect Team advisor. You analyze card portfolios, "
        "market dynamics, and roster construction to give actionable buy/sell/flip "
        "recommendations. Be specific with card names and prices. Be concise and direct. "
        "The user's team is the Toronto Dark Knights."
    )

    user_message = f"=== CURRENT PORTFOLIO STATE ===\n{portfolio_context}\n\n=== QUESTION ===\n{question}"

    try:
        if ai_config["provider"] == "gemini":
            return _call_gemini(system_prompt, user_message, ai_config)
        else:
            return _call_anthropic(system_prompt, user_message, ai_config)

    except ImportError as e:
        pkg = "google-genai" if ai_config["provider"] == "gemini" else "anthropic"
        return {
            "response": None,
            "tokens_used": 0,
            "model": None,
            "error": f"Missing package. Run: pip install {pkg}",
        }
    except Exception as e:
        logger.error(f"AI Advisor API error: {e}", exc_info=True)
        return {
            "response": None,
            "tokens_used": 0,
            "model": None,
            "error": f"API error: {e}",
        }


def get_market_analysis(conn=None) -> dict:
    """Get a comprehensive market and portfolio analysis from the AI advisor."""
    question = (
        "Analyze my current portfolio and market position. What are my top 3 priorities "
        "right now? Consider roster gaps, underpriced cards, upcoming upgrades, and flip "
        "opportunities. Give me a specific action plan with card names and prices."
    )
    return ask_advisor(question, conn=conn)


def get_trade_advice(card_name: str, action: str, conn=None) -> dict:
    """Get AI advice on whether to buy or sell a specific card."""
    question = (
        f"Should I {action} {card_name}? Consider the current price, meta score, "
        f"roster impact, upgrade/downgrade signals, and alternative options. "
        f"Give me a clear yes/no with reasoning."
    )
    return ask_advisor(question, conn=conn)


def get_upgrade_scouting_report(slot_label: str, current_card: dict, candidates: list,
                                 team_context: str, conn=None) -> dict:
    """Get AI scouting report comparing upgrade candidates for a roster slot.

    Goes beyond meta score to evaluate platoon splits, role fit, team composition,
    stat shape, and in-game performance.

    Args:
        slot_label: e.g. "SS", "SP3", "Setup 1"
        current_card: dict with full card stats for current starter
        candidates: list of dicts with full card stats for upgrade options
        team_context: brief summary of team composition/needs
        conn: optional DB connection

    Returns:
        Dict with keys: response, tokens_used, model, error
    """
    ai_config = get_ai_config()
    if not ai_config["ready"]:
        return {"response": None, "tokens_used": 0, "model": None, "error": ai_config["message"]}

    # Build the comparison prompt
    def _card_profile(card: dict, label: str) -> str:
        _bats_map = {'1': 'R', '2': 'L', '3': 'S'}
        _throws_map = {'1': 'R', '2': 'L'}
        bats = _bats_map.get(str(card.get('bats', '')), '?')
        throws = _throws_map.get(str(card.get('throws', '')), '?')

        lines = [f"  {label}: {card.get('card_title', card.get('player_name', '?'))}"]
        lines.append(f"    OVR: {card.get('card_value', card.get('ovr', '?'))} | "
                     f"Tier: {card.get('tier_name', '?')} | Bats: {bats} Throws: {throws}")

        # Batting stats
        if card.get('contact') or card.get('gap_power'):
            lines.append(f"    Overall: CON={card.get('contact',0)} GAP={card.get('gap_power',0)} "
                        f"POW={card.get('power',0)} EYE={card.get('eye',0)} AVK={card.get('avoid_ks',0)} "
                        f"BABIP={card.get('babip',0)}")
            if card.get('contact_vl'):
                lines.append(f"    vs LHP: CON={card.get('contact_vl',0)} GAP={card.get('gap_vl',0)} "
                            f"POW={card.get('power_vl',0)} EYE={card.get('eye_vl',0)} AVK={card.get('avoid_ks_vl',0)}")
            if card.get('contact_vr'):
                lines.append(f"    vs RHP: CON={card.get('contact_vr',0)} GAP={card.get('gap_vr',0)} "
                            f"POW={card.get('power_vr',0)} EYE={card.get('eye_vr',0)} AVK={card.get('avoid_ks_vr',0)}")
            if card.get('speed'):
                lines.append(f"    Speed={card.get('speed',0)} Steal={card.get('stealing',0)} "
                            f"BR={card.get('baserunning',0)}")
            # Defense
            for def_type in [('infield_range', 'infield_error', 'infield_arm'),
                             ('of_range', 'of_error', 'of_arm'),
                             ('catcher_ability', 'catcher_frame', 'catcher_arm')]:
                vals = [card.get(k, 0) or 0 for k in def_type]
                if any(vals):
                    labels = [k.split('_')[-1].title() for k in def_type]
                    lines.append(f"    Defense: {' / '.join(f'{l}={v}' for l, v in zip(labels, vals))}")
                    break

        # Pitching stats
        if card.get('stuff') or card.get('movement'):
            lines.append(f"    Overall: STU={card.get('stuff',0)} MOV={card.get('movement',0)} "
                        f"CTRL={card.get('control',0)} pHR={card.get('p_hr',0)} "
                        f"STA={card.get('stamina',0)} HOLD={card.get('hold',0)}")
            if card.get('stuff_vl'):
                lines.append(f"    vs LHB: STU={card.get('stuff_vl',0)} MOV={card.get('movement_vl',0)} "
                            f"CTRL={card.get('control_vl',0)}")
            if card.get('stuff_vr'):
                lines.append(f"    vs RHB: STU={card.get('stuff_vr',0)} MOV={card.get('movement_vr',0)} "
                            f"CTRL={card.get('control_vr',0)}")

        # Meta score
        meta_bat = card.get('meta_score_batting') or card.get('meta_score')
        meta_pit = card.get('meta_score_pitching')
        if meta_bat:
            lines.append(f"    Meta (bat): {meta_bat:.0f}")
        if meta_pit:
            lines.append(f"    Meta (pitch): {meta_pit:.0f}")

        # In-game performance if available
        if card.get('_ingame'):
            ig = card['_ingame']
            if ig.get('ops'):
                lines.append(f"    In-Game: .{int((ig.get('avg',0) or 0)*1000):03d}/{ig.get('obp',0):.3f}/{ig.get('slg',0):.3f} "
                            f"OPS={ig['ops']:.3f} WAR={ig.get('war',0):.1f} ({ig.get('pa',0)} PA)")
            elif ig.get('era'):
                lines.append(f"    In-Game: {ig['era']:.2f} ERA, {ig.get('whip',0):.2f} WHIP, "
                            f"{ig.get('k_per_9',0):.1f} K/9, WAR={ig.get('war',0):.1f} ({ig.get('ip',0):.0f} IP)")

        return "\n".join(lines)

    sections = [f"ROSTER SLOT: {slot_label}", ""]
    sections.append(_card_profile(current_card, "CURRENT STARTER"))
    sections.append("")

    for i, cand in enumerate(candidates[:3]):
        source = "OWNED" if cand.get('_source') == 'collection' else "MARKET"
        price_str = f" ({cand.get('last_10_price', 0):,} PP)" if source == "MARKET" else " (FREE)"
        sections.append(_card_profile(cand, f"CANDIDATE {i+1} [{source}{price_str}]"))
        sections.append("")

    sections.append(f"TEAM CONTEXT:\n{team_context}")

    system_prompt = (
        "You are an expert OOTP 27 Perfect Team scout. Analyze these upgrade candidates "
        "for a roster slot. Go BEYOND the meta score number and evaluate:\n"
        "1. PLATOON SPLITS: Do they hit/pitch better vs L or R? Major gaps?\n"
        "2. ROLE FIT: Does this card fit the slot (leadoff needs OBP, cleanup needs power, "
        "closer needs high-leverage stuff, long relief needs stamina)?\n"
        "3. STAT BALANCE: Is the card well-rounded or a one-trick pony? Hidden weaknesses?\n"
        "4. TEAM FIT: Does the team need what this card offers, or more of the same?\n"
        "5. IN-GAME TRACK RECORD: If performance data available, is the card over/underperforming?\n\n"
        "Give a VERDICT for each candidate: Strong Upgrade / Slight Upgrade / Sidegrade / Avoid.\n"
        "Then give your TOP PICK with reasoning. Be concise (150 words max)."
    )

    user_message = "\n".join(sections)

    try:
        if ai_config["provider"] == "gemini":
            return _call_gemini(system_prompt, user_message, ai_config)
        else:
            return _call_anthropic(system_prompt, user_message, ai_config)
    except Exception as e:
        logger.error(f"Scouting report API error: {e}", exc_info=True)
        return {"response": None, "tokens_used": 0, "model": None, "error": f"API error: {e}"}


def build_team_context(conn) -> str:
    """Build a brief team composition summary for scouting reports."""
    lines = []

    # Batting composition — join on last_name or full name match
    bat_stats = conn.execute("""
        SELECT r.position, r.player_name, r.meta_score,
               c.contact, c.gap_power, c.power, c.eye, c.avoid_ks, c.speed, c.bats
        FROM roster_current r
        LEFT JOIN cards c ON (
            c.card_title LIKE '%' || r.player_name || '%'
            OR (c.last_name IS NOT NULL AND c.first_name IS NOT NULL
                AND r.player_name = c.first_name || ' ' || c.last_name)
        ) AND c.owned = 1
        WHERE r.position IN ('C','1B','2B','3B','SS','LF','CF','RF')
        GROUP BY r.player_name
        ORDER BY r.meta_score DESC
    """).fetchall()

    # Aggregate team tendencies
    power_count = sum(1 for b in bat_stats if (b['power'] or 0) >= 90)
    contact_count = sum(1 for b in bat_stats if (b['contact'] or 0) >= 90)
    speed_count = sum(1 for b in bat_stats if (b['speed'] or 0) >= 60)

    # bats: 1=R, 2=L, 3=S (numeric in DB)
    bats_r = sum(1 for b in bat_stats if str(b['bats']) == '1')
    bats_l = sum(1 for b in bat_stats if str(b['bats']) == '2')
    bats_s = sum(1 for b in bat_stats if str(b['bats']) == '3')

    lines.append(f"Lineup: {power_count} power bats (POW>=90), {contact_count} contact bats (CON>=90), "
                 f"{speed_count} speed threats (SPD>=60)")
    lines.append(f"Handedness: {bats_l}L / {bats_r}R / {bats_s}S")

    # Pitching composition
    pitch_stats = conn.execute("""
        SELECT r.position, r.player_name, r.meta_score,
               c.stuff, c.movement, c.control, c.stamina, c.throws
        FROM roster_current r
        LEFT JOIN cards c ON (
            c.card_title LIKE '%' || r.player_name || '%'
            OR (c.last_name IS NOT NULL AND c.first_name IS NOT NULL
                AND r.player_name = c.first_name || ' ' || c.last_name)
        ) AND c.owned = 1
        WHERE r.position IN ('SP','RP','CL')
        GROUP BY r.player_name
        ORDER BY r.position, r.meta_score DESC
    """).fetchall()

    throws_r = sum(1 for p in pitch_stats if str(p['throws']) == '1')
    throws_l = sum(1 for p in pitch_stats if str(p['throws']) == '2')
    lines.append(f"Pitching: {throws_l}L / {throws_r}R arms")

    # Team in-game stats if available
    try:
        team_ops = conn.execute(
            "SELECT AVG(ops) as avg_ops FROM batting_stats WHERE ab >= 50"
        ).fetchone()
        team_era = conn.execute(
            "SELECT AVG(era) as avg_era FROM pitching_stats WHERE ip >= 20"
        ).fetchone()
        if team_ops and team_ops['avg_ops']:
            lines.append(f"Team avg OPS: {team_ops['avg_ops']:.3f}")
        if team_era and team_era['avg_era']:
            lines.append(f"Team avg ERA: {team_era['avg_era']:.2f}")
    except Exception:
        pass

    return "\n".join(lines)


def get_full_card_data(card_id_or_title, conn) -> dict:
    """Fetch full card data including splits and in-game stats for scouting reports."""
    # Try by card_id first, then by title match
    card = None
    if isinstance(card_id_or_title, int):
        card = conn.execute("SELECT * FROM cards WHERE card_id = ?", (card_id_or_title,)).fetchone()
    if card is None:
        card = conn.execute(
            "SELECT * FROM cards WHERE card_title LIKE ? LIMIT 1",
            (f"%{card_id_or_title}%",)
        ).fetchone()

    if card is None:
        return {}

    data = dict(card)

    # Attach in-game batting stats
    bat_stats = conn.execute(
        "SELECT * FROM batting_stats WHERE card_id = ? OR player_name LIKE ? ORDER BY snapshot_date DESC LIMIT 1",
        (data.get('card_id', -1), f"%{card_id_or_title}%")
    ).fetchone()
    if bat_stats and (bat_stats['pa'] or 0) > 0:
        data['_ingame'] = dict(bat_stats)
    else:
        # Try pitching stats
        pitch_stats = conn.execute(
            "SELECT * FROM pitching_stats WHERE card_id = ? OR player_name LIKE ? ORDER BY snapshot_date DESC LIMIT 1",
            (data.get('card_id', -1), f"%{card_id_or_title}%")
        ).fetchone()
        if pitch_stats and (pitch_stats['ip'] or 0) > 0:
            data['_ingame'] = dict(pitch_stats)

    return data


def get_flip_strategy(conn=None) -> dict:
    """Get AI-powered flipping strategy recommendations."""
    question = (
        "What's my best flipping strategy right now? Look at spread opportunities, "
        "volatile cards, underpriced live cards with upgrade signals, and upcoming "
        "matchup advantages. Give me 3-5 specific flip trades with buy/sell prices "
        "and expected profit."
    )
    return ask_advisor(question, conn=conn)


def _ensure_ai_insights_table(conn):
    """Create ai_insights table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            insight_type TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def generate_ai_insights(conn):
    """Passively generate AI strategic insights after recommendations run.

    Called automatically at the end of generate_recommendations().
    Sends a concise summary to Gemini and stores the response.
    """
    from datetime import datetime

    ai_config = get_ai_config()
    if not ai_config["ready"]:
        return

    _ensure_ai_insights_table(conn)

    try:
        # Build concise context (keep under 500 tokens)
        lines = []

        # Top 5 buy recs
        buys = conn.execute("""
            SELECT card_title, position, meta_score, estimated_price, reason
            FROM recommendations WHERE rec_type = 'buy' AND dismissed = 0
            ORDER BY priority ASC, value_ratio DESC LIMIT 5
        """).fetchall()
        if buys:
            lines.append("TOP BUYS:")
            for b in buys:
                lines.append(f"  {b['card_title']} ({b['position']}) meta:{b['meta_score']:.0f} ~{b['estimated_price']}PP")

        # Top 5 sell recs
        sells = conn.execute("""
            SELECT card_title, position, estimated_price, reason
            FROM recommendations WHERE rec_type = 'sell' AND dismissed = 0
            ORDER BY priority ASC, estimated_price DESC LIMIT 5
        """).fetchall()
        if sells:
            lines.append("TOP SELLS:")
            for s in sells:
                lines.append(f"  {s['card_title']} ({s['position']}) ~{s['estimated_price']}PP")

        # Roster weaknesses
        weak = conn.execute("""
            SELECT position, MAX(meta_score) as best_meta, player_name
            FROM roster_current WHERE lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
            GROUP BY position ORDER BY best_meta ASC LIMIT 3
        """).fetchall()
        if weak:
            lines.append("WEAKEST POSITIONS:")
            for w in weak:
                lines.append(f"  {w['position']}: {w['player_name']} ({w['best_meta']:.0f} meta)")

        # Meta validation accuracy (if available)
        try:
            from app.core.meta_validation import get_meta_accuracy_score
            accuracy = get_meta_accuracy_score(conn)
            if accuracy["sample_size"] > 0:
                lines.append(f"META ACCURACY: {accuracy['accuracy_pct']}% ({accuracy['sample_size']} players)")
        except Exception:
            pass

        if not lines:
            return

        context = "\n".join(lines)

        system_prompt = (
            "You are an OOTP 27 Perfect Team advisor. Give a 3-4 bullet strategic summary. "
            "Be specific with card names. Keep it under 100 words total."
        )
        user_message = f"Current state:\n{context}\n\nGive 3-4 bullet AI Strategic Summary."

        if ai_config["provider"] == "gemini":
            result = _call_gemini(system_prompt, user_message, ai_config)
        else:
            result = _call_anthropic(system_prompt, user_message, ai_config)

        if result.get("response"):
            conn.execute(
                "INSERT INTO ai_insights (insight_type, content, created_at) VALUES (?, ?, ?)",
                ("strategic_summary", result["response"], datetime.now().isoformat()),
            )
            conn.commit()

    except Exception as e:
        logger.warning(f"Passive AI insights failed (non-fatal): {e}")


def generate_meta_insight(validation_result, conn):
    """Generate AI interpretation of meta validation results.

    Called after validate_meta_vs_performance() runs.
    """
    from datetime import datetime

    ai_config = get_ai_config()
    if not ai_config["ready"]:
        return

    _ensure_ai_insights_table(conn)

    try:
        correlation = validation_result.get("correlation", 0)
        bat_corr = validation_result.get("batting_correlation", 0)
        pitch_corr = validation_result.get("pitching_correlation", 0)

        overperformers = validation_result.get("overperformers", [])[:3]
        underperformers = validation_result.get("underperformers", [])[:3]

        lines = [
            f"Overall r={correlation:.3f}, Batting r={bat_corr:.3f}, Pitching r={pitch_corr:.3f}",
        ]
        if overperformers:
            lines.append("Overperformers: " + ", ".join(
                f"{p['player_name']} (meta {p['meta_score']:.0f}, perf {p['performance_rating']:.0f})"
                for p in overperformers
            ))
        if underperformers:
            lines.append("Underperformers: " + ", ".join(
                f"{p['player_name']} (meta {p['meta_score']:.0f}, perf {p['performance_rating']:.0f})"
                for p in underperformers
            ))

        context = "\n".join(lines)

        system_prompt = (
            "You are an OOTP 27 Perfect Team meta formula analyst. "
            "Interpret what the meta formula is getting wrong in 2-3 sentences."
        )
        user_message = f"Meta validation data:\n{context}\n\nWhat is the meta formula getting wrong?"

        if ai_config["provider"] == "gemini":
            result = _call_gemini(system_prompt, user_message, ai_config)
        else:
            result = _call_anthropic(system_prompt, user_message, ai_config)

        if result.get("response"):
            conn.execute(
                "INSERT INTO ai_insights (insight_type, content, created_at) VALUES (?, ?, ?)",
                ("meta_insight", result["response"], datetime.now().isoformat()),
            )
            conn.commit()

    except Exception as e:
        logger.warning(f"Meta insight generation failed (non-fatal): {e}")


def ai_optimize_all_positions(upgrade_plan: list, conn, perf_bat: dict = None,
                               perf_pit: dict = None, max_spend_per_card: int = 5000) -> dict:
    """Run AI reasoning across ALL roster positions to produce smarter recommendations.

    Instead of pure meta-score ordering, the AI considers:
    - Platoon splits (vs LHP/RHP)
    - In-game performance vs meta predictions
    - Defensive value and role fit
    - Team composition balance
    - Card handedness and lineup construction

    Args:
        upgrade_plan: list of slot dicts from the optimizer grid
        conn: DB connection
        perf_bat: dict of batter in-game performance {name: {ops, war, war600, ...}}
        perf_pit: dict of pitcher in-game performance {name: {era, fip, war, ...}}

    Returns:
        Dict with 'response' (raw AI text), 'picks' (parsed per-position picks),
        'tokens_used', 'model', 'error'
    """
    ai_config = get_ai_config()
    if not ai_config["ready"]:
        return {"response": None, "picks": {}, "tokens_used": 0,
                "model": None, "error": ai_config["message"]}

    perf_bat = perf_bat or {}
    perf_pit = perf_pit or {}

    # Build comprehensive prompt with all positions
    sections = []

    # Team context
    team_ctx = build_team_context(conn)
    sections.append(f"TEAM CONTEXT:\n{team_ctx}\n")

    pp_budget = max_spend_per_card

    # Load split meta from roster table
    roster_splits = {}
    for r in conn.execute("""
        SELECT player_name, position, meta_vs_rhp, meta_vs_lhp, bats,
               con_vl, pow_vl, eye_vl, con_vr, pow_vr, eye_vr,
               stu_vl, stu_vr
        FROM roster
        WHERE lineup_role != 'league' AND DATE(snapshot_date) = (
            SELECT MAX(DATE(snapshot_date)) FROM roster WHERE lineup_role != 'league'
        )
    """).fetchall():
        roster_splits[r['player_name']] = dict(r)

    # ── ALL OWNED CARDS by position (so AI can reason about full collection) ──
    sections.append("=== ALL OWNED CARDS (your collection) ===")
    sections.append("These are ALL cards you own, organized by position. Use these to find the BEST roster from what you already have.\n")

    # Batting positions
    for pos in ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']:
        owned_at_pos = conn.execute("""
            SELECT card_title, card_value, meta_score_batting,
                   contact, gap_power, power, eye, avoid_ks,
                   contact_vl, contact_vr, power_vl, power_vr, eye_vl, eye_vr,
                   speed, stealing, bats, tier_name
            FROM cards WHERE position_name = ? AND owned = 1
            ORDER BY meta_score_batting DESC LIMIT 8
        """, (pos,)).fetchall()
        if owned_at_pos:
            lines = [f"{pos}: "]
            for c in owned_at_pos:
                bats_map = {'1': 'R', '2': 'L', '3': 'S'}
                bh = bats_map.get(str(c['bats']), '?')
                meta = c['meta_score_batting'] or 0
                line = f"  {short_name_for_ai(c['card_title'])} (OVR {c['card_value']}, Meta {meta:.0f}, {bh})"
                if c['contact_vl'] and c['contact_vr']:
                    line += f" CONvL={c['contact_vl']}/vR={c['contact_vr']}"
                lines.append(line)
            sections.append("\n".join(lines))

    # Pitching positions
    for role in ['SP', 'RP', 'CL']:
        owned_at_pos = conn.execute("""
            SELECT card_title, card_value, meta_score_pitching,
                   stuff, movement, control, stamina,
                   stuff_vl, stuff_vr, throws, tier_name
            FROM cards WHERE pitcher_role_name = ? AND owned = 1
            ORDER BY meta_score_pitching DESC LIMIT 10
        """, (role,)).fetchall()
        if owned_at_pos:
            lines = [f"{role}: "]
            for c in owned_at_pos:
                meta = c['meta_score_pitching'] or 0
                line = f"  {short_name_for_ai(c['card_title'])} (OVR {c['card_value']}, Meta {meta:.0f}, STU {c['stuff'] or 0})"
                if c['stuff_vl'] and c['stuff_vr']:
                    line += f" STUvL={c['stuff_vl']}/vR={c['stuff_vr']}"
                if c['stamina']:
                    line += f" STA={c['stamina']}"
                lines.append(line)
            sections.append("\n".join(lines))

    sections.append(f"\n=== MAX SPEND: {pp_budget:,} PP per card (do NOT recommend cards above this price) ===\n")
    sections.append("=== CURRENT ROSTER & UPGRADE CANDIDATES ===")
    sections.append("First optimize using OWNED cards above, then suggest market purchases within budget.\n")

    slots_with_upgrades = []
    for u in upgrade_plan:
        # Include ALL positions so AI can evaluate full collection even without meta-based upgrades
        name = u['current_name']
        splits = roster_splits.get(name, {})

        bats_h = u.get('bats', '?')
        slot_lines = [f"\n--- {u['pos']}: {name} (OVR {u['current_ovr']}, Meta {u['current_meta']}, Bats {bats_h}) ---"]

        # Add split info
        rhp = splits.get('meta_vs_rhp')
        lhp = splits.get('meta_vs_lhp')
        bats = splits.get('bats', '?')
        if rhp and lhp:
            diff = abs(rhp - lhp)
            split_note = ""
            if diff > 30:
                better = "vs RHP" if rhp > lhp else "vs LHP"
                split_note = f" ** BIG SPLIT: {better} by {diff:.0f} **"
            slot_lines.append(f"  Bats: {bats} | vsRHP: {rhp:.0f} | vsLHP: {lhp:.0f}{split_note}")

        # In-game performance
        pb = perf_bat.get(name)
        pp = perf_pit.get(name)
        if pp:
            era_fip = pp['era'] - pp['fip']
            luck = "LUCKY (ERA << FIP, will regress)" if era_fip < -0.5 else (
                "UNLUCKY (ERA >> FIP, should improve)" if era_fip > 0.5 else "FAIR")
            slot_lines.append(f"  In-Game: {pp['era']:.2f} ERA, {pp['fip']:.2f} FIP, "
                            f"{pp['war']:.1f} WAR ({pp['ip']:.0f} IP) — {luck}")
        elif pb:
            slot_lines.append(f"  In-Game: .{int(pb['ops']*1000):03d} OPS, {pb['war']:.1f} WAR, "
                            f"{pb['war600']:.1f} WAR/600 ({pb['pa']} PA)")

        # Candidates
        if u['owned_name']:
            ow_card = get_full_card_data(u['owned_name'], conn)
            ow_splits = roster_splits.get(u['owned_name'].split()[-1] if u['owned_name'] else '', {})
            owned_line = f"  OWNED: {short_name_for_ai(u['owned_name'])} (OVR {u['owned_ovr']}, Meta {u['owned_meta']}, +{u['owned_delta']}) — {u['owned_action']}"
            # Check owned card splits
            if ow_card:
                if ow_card.get('contact_vl') and ow_card.get('contact_vr'):
                    owned_line += f" | CONvL={ow_card['contact_vl']} CONvR={ow_card['contact_vr']}"
                if ow_card.get('stuff_vl') and ow_card.get('stuff_vr'):
                    owned_line += f" | STUvL={ow_card['stuff_vl']} STUvR={ow_card['stuff_vr']}"
            # Owned card in-game perf
            ow_name_short = u['owned_name'].split('  ')[-1].rsplit('  ', 1)[0] if u['owned_name'] else ''
            for pname in [ow_name_short, u.get('owned_name', '')]:
                op = perf_pit.get(pname) or perf_bat.get(pname)
                if op:
                    if 'era' in op:
                        owned_line += f" | In-Game: {op['era']:.2f} ERA {op['war']:.1f}W"
                    elif 'ops' in op:
                        owned_line += f" | In-Game: .{int(op['ops']*1000):03d} OPS {op['war']:.1f}W"
                    break
            slot_lines.append(owned_line)

        if u['market_name']:
            mk_card = get_full_card_data(u['market_name'], conn)
            market_line = f"  MARKET: {short_name_for_ai(u['market_name'])} (OVR {u['market_ovr']}, Meta {u['market_meta']}, +{u['market_delta']}) — {u['market_price']:,} PP"
            if mk_card:
                if mk_card.get('contact_vl') and mk_card.get('contact_vr'):
                    market_line += f" | CONvL={mk_card['contact_vl']} CONvR={mk_card['contact_vr']}"
                if mk_card.get('stuff_vl') and mk_card.get('stuff_vr'):
                    market_line += f" | STUvL={mk_card['stuff_vl']} STUvR={mk_card['stuff_vr']}"
            slot_lines.append(market_line)

        # Other candidates from alternatives (give AI more options to evaluate)
        for alt in u.get('_owned_upgrades', [])[1:4]:
            slot_lines.append(f"  ALT OWNED: {short_name_for_ai(alt.get('card_title',''))} "
                            f"(Meta {round(alt.get('meta_score',0))}, {alt.get('action','FREE')})")
        for alt in u.get('_market_upgrades', [])[1:3]:
            p = alt.get('last_10_price', 0) or 0
            slot_lines.append(f"  ALT MARKET: {short_name_for_ai(alt.get('card_title',''))} "
                            f"(Meta {round(alt.get('meta_score',0))}, {p:,} PP)")

        sections.append("\n".join(slot_lines))
        slots_with_upgrades.append(u['pos'])

    if not slots_with_upgrades:
        return {"response": "No upgrade candidates to evaluate.", "picks": {},
                "tokens_used": 0, "model": None, "error": None}

    system_prompt = (
        "You are an expert OOTP 27 Perfect Team GM optimizing a roster. "
        "You have access to the user's FULL collection of owned cards AND market options.\n\n"
        "YOUR JOB: For EVERY position, find the BEST possible card — first from owned cards, "
        "then from the market. You should almost always be recommending PROMOTE or BUY. "
        "KEEP should be RARE — only when the current starter is genuinely the best available "
        "option AND no market card within budget is better.\n\n"
        "STRATEGY — TWO PHASES:\n"
        "Phase 1: OWNED CARDS. For each position, check the full owned collection list. "
        "If there's a higher-meta owned card NOT in the lineup, recommend PROMOTE. "
        "Consider splits, performance, and role fit — but a +20 meta owned card should almost "
        "always be promoted unless there's a strong performance reason not to.\n"
        "Phase 2: MARKET BUYS. If the best owned card is already starting, look at market options. "
        "If a market card is a meaningful upgrade (+30+ meta or fills a clear weakness), recommend BUY. "
        "The user WANTS to spend PP to improve — don't be conservative about market buys.\n\n"
        "IMPORTANT: The meta-based system has already identified OWNED and MARKET candidates for each slot. "
        "Your job is to VALIDATE or IMPROVE those picks using context the meta system can't see "
        "(splits, performance, luck, team fit). If the meta pick looks right, CONFIRM it as PROMOTE or BUY. "
        "Don't override good recommendations with KEEP just because the current player is 'decent'.\n\n"
        "Use these factors to decide between candidates:\n"
        "1. PLATOON RULES: A valid platoon is ONE L-batter + ONE R-batter at the same position. "
        "Two same-handed batters (L+L or R+R) is NOT a platoon — it's a weakness. "
        "Switch hitters (S) pair with anyone. If a position has 2 same-handed batters, "
        "recommend replacing the weaker one with an opposite-hand batter.\n"
        "2. PLATOON SPLITS: Big splits (>30 gap between vRHP and vLHP) = platoon candidate.\n"
        "3. IN-GAME PERFORMANCE: ERA << FIP = lucky (will regress). ERA >> FIP = unlucky (will improve).\n"
        "4. DEFENSE & ROLE FIT: Catchers need defense. CF needs speed. Closers need stuff.\n"
        "5. TEAM BALANCE: Mix handedness, power/contact.\n\n"
        "FORMAT your response EXACTLY like this for each position (one per line):\n\n"
        "For owned upgrades:\n"
        "**[POS]**: [EMOJI] PROMOTE [New card name] >> [Current player being replaced] | [1-sentence reason]\n\n"
        "For market purchases:\n"
        "**[POS]**: [EMOJI] BUY [New card name] >> [Current player being replaced] | [1-sentence reason]\n\n"
        "For platoons:\n"
        "**[POS]**: [EMOJI] PLATOON [Player A] + [Player B] >> [Player being dropped] | [1-sentence reason]\n\n"
        "Only when NO better option exists anywhere:\n"
        "**[POS]**: [EMOJI] KEEP [Current player name] | [1-sentence reason]\n\n"
        "The >> arrow is CRITICAL — it tells the user WHO gets removed. "
        "Every PROMOTE or BUY MUST include >> [replaced player].\n\n"
        "Verdict emojis: \U0001f525 = Strong Upgrade, \u2b06\ufe0f = Slight Upgrade, "
        "\U0001f4b0 = Worth the Buy, \U0001f91d = Platoon Both, "
        "\u2744\ufe0f = Keep (RARE — only when truly optimal)\n\n"
        "After all positions, add:\n"
        "**TOP 3 PRIORITIES**: numbered list of the 3 most impactful moves (include >> swap details)\n"
        "Be concise. 1 sentence per position max."
    )

    user_message = "\n".join(sections)

    try:
        if ai_config["provider"] == "gemini":
            result = _call_gemini(system_prompt, user_message, ai_config, max_tokens=2048)
        else:
            result = _call_anthropic(system_prompt, user_message, ai_config, max_tokens=2048)

        # Parse AI picks from response into structured data
        picks = {}       # pos -> verdict_text (legacy)
        picks_data = []  # list of structured dicts for table display
        if result.get('response'):
            import re
            for line in result['response'].split('\n'):
                line = line.strip()
                if line.startswith('**') and '**:' in line:
                    pos_part = line.split('**:')[0].replace('**', '').strip()
                    verdict_part = line.split('**:', 1)[1].strip() if '**:' in line else ''

                    # Skip non-position lines like TOP 3 PRIORITIES
                    if pos_part in slots_with_upgrades or any(pos_part.startswith(s) for s in slots_with_upgrades):
                        picks[pos_part] = verdict_part

                        # Parse structured fields: EMOJI ACTION Card Name >> Replaced | reason
                        emoji = ''
                        action = ''
                        card_name = ''
                        replaces = ''
                        reason = ''

                        # Split on pipe for reason
                        if '|' in verdict_part:
                            pick_part, reason = verdict_part.split('|', 1)
                            reason = reason.strip()
                        else:
                            pick_part = verdict_part

                        # Extract emoji (first 1-2 chars if emoji)
                        pick_part = pick_part.strip()
                        # Match leading emoji(s)
                        emoji_match = re.match(r'^([\U0001f300-\U0001f9ff\u2b06\u2744\u274c\ufe0f\u26a0]+)\s*', pick_part)
                        if emoji_match:
                            emoji = emoji_match.group(1).strip()
                            pick_part = pick_part[emoji_match.end():].strip()

                        # Extract ACTION keyword
                        for act in ['PROMOTE', 'BUY', 'KEEP', 'PLATOON']:
                            if pick_part.upper().startswith(act):
                                action = act.title()
                                pick_part = pick_part[len(act):].strip()
                                break

                        # Extract >> replaced player
                        if '>>' in pick_part:
                            card_part, replaces = pick_part.split('>>', 1)
                            card_name = card_part.strip()
                            replaces = replaces.strip()
                        else:
                            card_name = pick_part.strip()

                        # For PLATOON, parse "Player A + Player B"
                        platoon_partner = ''
                        if action == 'Platoon' and '+' in card_name:
                            parts = card_name.split('+', 1)
                            card_name = parts[0].strip()
                            platoon_partner = parts[1].strip()

                        # Look up cost for BUY actions from upgrade_plan
                        cost = None
                        current_name = ''
                        for u in upgrade_plan:
                            if u['pos'] == pos_part:
                                current_name = u['current_name']
                                if action == 'Buy' and u.get('market_price'):
                                    cost = u['market_price']
                                break

                        # If replaces wasn't parsed but action is PROMOTE/BUY,
                        # fall back to the current starter at that position
                        if not replaces and action in ('Promote', 'Buy'):
                            replaces = current_name

                        picks_data.append({
                            'pos': pos_part,
                            'emoji': emoji,
                            'action': action or 'Keep',
                            'card_name': card_name,
                            'replaces': replaces,
                            'platoon_partner': platoon_partner,
                            'reason': reason,
                            'cost': cost,
                        })

            # Store in ai_insights for history
            from datetime import datetime
            _ensure_ai_insights_table(conn)
            conn.execute(
                "INSERT INTO ai_insights (insight_type, content, created_at) VALUES (?, ?, ?)",
                ("roster_ai_optimize", result['response'], datetime.now().isoformat()),
            )
            conn.commit()

        result['picks'] = picks
        result['picks_data'] = picks_data
        return result

    except Exception as e:
        logger.error(f"AI Optimize All error: {e}", exc_info=True)
        return {"response": None, "picks": {}, "tokens_used": 0,
                "model": None, "error": f"API error: {e}"}


def short_name_for_ai(card_title, max_len=40):
    """Truncate card title for AI prompts."""
    if not card_title:
        return "?"
    t = card_title.strip()
    return t[:max_len] if len(t) <= max_len else t[:max_len] + "..."


# ============================================================================
# Manager's Eye — holistic roster sanity check
# ============================================================================
def _build_manager_review_context(conn) -> str:
    """Build a detailed roster composition snapshot for the manager's review.

    This is different from _build_portfolio_context — it focuses on ROSTER
    CONSTRUCTION (positional coverage, handedness, platoon fits, bullpen roles)
    rather than markets/flips/missions.
    """
    lines = []

    # Determine the ACTIVE league.
    # Priority: config.yaml's active_league setting > roster tagging > export_log.
    active_league = None
    try:
        from app.core.database import load_config
        cfg = load_config()
        active_league = cfg.get("active_league")
    except Exception:
        pass
    if not active_league:
        try:
            row = conn.execute("""
                SELECT league_id, MAX(snapshot_date) as latest FROM roster
                WHERE lineup_role != 'league' AND league_id IS NOT NULL
                GROUP BY league_id ORDER BY latest DESC LIMIT 1
            """).fetchone()
            if row:
                active_league = row['league_id']
        except Exception:
            pass
    if not active_league:
        try:
            row = conn.execute(
                "SELECT league_id FROM export_log ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if row:
                active_league = row['league_id']
        except Exception:
            pass

    # Roster query — always use latest snapshot, regardless of league tagging.
    # (If the user has multiple leagues tagged, they're usually from
    # different sessions and the newest snapshot is what matters.)
    rows = conn.execute("""
        SELECT r.position, r.player_name, r.lineup_role, r.ovr, r.meta_score,
               c.bats, c.throws, c.contact, c.gap_power, c.power, c.eye,
               c.speed, c.stealing, c.baserunning,
               c.stuff, c.movement, c.control, c.p_hr, c.stamina, c.hold,
               c.infield_range, c.of_range, c.catcher_ability
        FROM roster r
        LEFT JOIN cards c ON (
            c.card_title LIKE '%' || r.player_name || '%' AND c.owned = 1
        )
        WHERE r.lineup_role IN ('starter', 'rotation', 'closer', 'bullpen')
          AND DATE(r.snapshot_date) = (
              SELECT MAX(DATE(snapshot_date)) FROM roster WHERE lineup_role != 'league'
          )
        GROUP BY r.id
        ORDER BY
            CASE r.position
                WHEN 'C' THEN 1 WHEN '1B' THEN 2 WHEN '2B' THEN 3 WHEN '3B' THEN 4
                WHEN 'SS' THEN 5 WHEN 'LF' THEN 6 WHEN 'CF' THEN 7 WHEN 'RF' THEN 8
                WHEN 'DH' THEN 9 WHEN 'SP' THEN 10 WHEN 'RP' THEN 11 WHEN 'CL' THEN 12
                ELSE 99 END,
            r.meta_score DESC
    """).fetchall()

    if active_league:
        lines.append(f"ACTIVE LEAGUE: {active_league}")
        lines.append("")

    _bats_map = {1: 'R', 2: 'L', 3: 'S', '1': 'R', '2': 'L', '3': 'S'}
    _throws_map = {1: 'R', 2: 'L', '1': 'R', '2': 'L'}

    # Split into batters and pitchers
    batters, pitchers = [], []
    for r in rows:
        if r['position'] in ('SP', 'RP', 'CL'):
            pitchers.append(r)
        else:
            batters.append(r)

    lines.append("BATTING ROSTER:")
    for b in batters:
        bh = _bats_map.get(b['bats'], '?')
        meta = b['meta_score'] or 0
        con = b['contact'] or 0
        gap = b['gap_power'] or 0
        pow_ = b['power'] or 0
        eye = b['eye'] or 0
        spd = b['speed'] or 0
        role = b['lineup_role']
        lines.append(
            f"  {b['position']}: {b['player_name']} ({bh}) ovr={b['ovr']} meta={meta:.0f} "
            f"CON={con} GAP={gap} POW={pow_} EYE={eye} SPD={spd} [{role}]"
        )

    lines.append("")
    lines.append("PITCHING STAFF:")
    for p in pitchers:
        th = _throws_map.get(p['throws'], '?')
        meta = p['meta_score'] or 0
        stu = p['stuff'] or 0
        mov = p['movement'] or 0
        ctrl = p['control'] or 0
        phr = p['p_hr'] or 0
        sta = p['stamina'] or 0
        hold = p['hold'] or 0
        role = p['lineup_role']
        lines.append(
            f"  {p['position']}: {p['player_name']} ({th}) ovr={p['ovr']} meta={meta:.0f} "
            f"STU={stu} MOV={mov} CTRL={ctrl} pHR={phr} STA={sta} HOLD={hold} [{role}]"
        )

    # Handedness summary
    lines.append("")
    bats_l = sum(1 for b in batters if _bats_map.get(b['bats']) == 'L')
    bats_r = sum(1 for b in batters if _bats_map.get(b['bats']) == 'R')
    bats_s = sum(1 for b in batters if _bats_map.get(b['bats']) == 'S')
    throws_l = sum(1 for p in pitchers if _throws_map.get(p['throws']) == 'L')
    throws_r = sum(1 for p in pitchers if _throws_map.get(p['throws']) == 'R')
    lines.append(f"HANDEDNESS: Batters {bats_l}L / {bats_r}R / {bats_s}S | "
                 f"Pitchers {throws_l}L / {throws_r}R")

    # Team averages (hints at identity)
    if batters:
        avg_con = sum(b['contact'] or 0 for b in batters) / len(batters)
        avg_pow = sum(b['power'] or 0 for b in batters) / len(batters)
        avg_spd = sum(b['speed'] or 0 for b in batters) / len(batters)
        avg_eye = sum(b['eye'] or 0 for b in batters) / len(batters)
        lines.append(f"BATTING PROFILE: avg CON={avg_con:.0f} POW={avg_pow:.0f} "
                     f"EYE={avg_eye:.0f} SPD={avg_spd:.0f}")

    if pitchers:
        avg_stu = sum(p['stuff'] or 0 for p in pitchers) / len(pitchers)
        avg_ctrl = sum(p['control'] or 0 for p in pitchers) / len(pitchers)
        avg_mov = sum(p['movement'] or 0 for p in pitchers) / len(pitchers)
        lines.append(f"PITCHING PROFILE: avg STU={avg_stu:.0f} MOV={avg_mov:.0f} "
                     f"CTRL={avg_ctrl:.0f}")

    # In-game team performance if available — use latest snapshot
    try:
        team_perf = conn.execute("""
            SELECT AVG(ops) as ops, AVG(war) as war_bat
            FROM batting_stats
            WHERE pa >= 20 AND snapshot_date = (SELECT MAX(snapshot_date) FROM batting_stats)
        """).fetchone()
        team_pitch = conn.execute("""
            SELECT AVG(era) as era, AVG(whip) as whip
            FROM pitching_stats
            WHERE ip >= 5 AND snapshot_date = (SELECT MAX(snapshot_date) FROM pitching_stats)
        """).fetchone()
        if active_league:
            record_row = conn.execute("""
                SELECT team_record, games_played FROM export_log
                WHERE league_id = ?
                ORDER BY created_at DESC LIMIT 1
            """, (active_league,)).fetchone()
        else:
            record_row = conn.execute("""
                SELECT team_record, games_played FROM export_log
                ORDER BY created_at DESC LIMIT 1
            """).fetchone()
        if record_row:
            lines.append(f"CURRENT RECORD: {record_row['team_record']} ({record_row['games_played']} games)")
        if team_perf and team_perf['ops']:
            lines.append(f"TEAM OPS: {team_perf['ops']:.3f}")
        if team_pitch and team_pitch['era']:
            lines.append(f"TEAM ERA: {team_pitch['era']:.2f} WHIP: {team_pitch['whip']:.2f}")
    except Exception:
        pass

    return "\n".join(lines)


def get_roster_manager_review(conn=None) -> dict:
    """Get a holistic 'manager's eye' review of the roster.

    Goes beyond position-by-position upgrade scoring to identify:
    - Critical coverage gaps (missing backup C, no utility IF, no LH bat, etc.)
    - Composition issues (handedness imbalance, too many of one skill type)
    - Strategic concerns (bullpen construction, platoon matchups, defense gaps)
    - Team identity summary (contact/power/speed/pitching-oriented)

    Returns dict: response, tokens_used, model, error
    """
    ai_config = get_ai_config()
    if not ai_config["ready"]:
        return {"response": None, "tokens_used": 0, "model": None, "error": ai_config["message"]}

    from app.core.database import get_connection
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        context = _build_manager_review_context(conn)
    except Exception as e:
        context = f"(Could not build context: {e})"
    finally:
        if close_conn:
            conn.close()

    system_prompt = (
        "You are an experienced Major League Baseball manager and bench coach reviewing "
        "your team's 26-man roster before making lineup decisions. You are NOT recommending "
        "specific player acquisitions — you are identifying ROSTER CONSTRUCTION problems "
        "that need to be addressed.\n\n"
        "Baseball roster rules to evaluate against:\n"
        "- 26-man roster: typically 13 position players + 13 pitchers\n"
        "- Position players: 2 C (starter + backup), 4-6 IF including utility, 4-5 OF, 1 DH\n"
        "- Pitchers: 5 SP, 7-8 RP (including a closer and 1-2 setup men)\n"
        "- Bench should have: 1 backup C, 1 utility IF (plays SS/2B/3B), 1 4th OF, 1 platoon bat\n"
        "- Handedness: want 2-3 LH bats minimum (vs RHP), 1+ LH reliever for matchups\n"
        "- Pitching: balance of power (high stuff) and control (high ctrl) is needed\n\n"
        "Your response MUST be structured with these sections:\n"
        "1. **Team Identity** (1 line): 'This team is a ___ team' (e.g. 'contact-and-speed', "
        "'power-heavy', 'groundball pitching')\n"
        "2. **Critical Gaps** (bullet list, 2-4 items): The biggest roster-construction problems. "
        "Focus on ROLES missing, not specific players.\n"
        "3. **Composition Concerns** (bullet list, 2-3 items): Subtler issues — handedness "
        "imbalance, overlap, platoon coverage.\n"
        "4. **Strategic Fit** (1-2 sentences): How should this team be PLAYED given its "
        "strengths/weaknesses?\n\n"
        "Be blunt. If the roster is well-built, say so. Don't invent problems. Max 250 words."
    )

    user_message = f"=== ROSTER STATE ===\n{context}\n\n=== REVIEW ==="

    try:
        if ai_config["provider"] == "gemini":
            return _call_gemini(system_prompt, user_message, ai_config, max_tokens=800)
        else:
            return _call_anthropic(system_prompt, user_message, ai_config, max_tokens=800)
    except Exception as e:
        logger.error(f"Manager review API error: {e}", exc_info=True)
        return {"response": None, "tokens_used": 0, "model": None, "error": f"API error: {e}"}


# ============================================================================
# Strategy Recommendations — AI-powered OOTP strategy slider suggestions
# ============================================================================
def get_strategy_recommendations(conn=None) -> dict:
    """Recommend OOTP in-game strategy slider values based on roster composition.

    Analyzes the team's batting profile (speed vs power, contact vs TTO),
    pitching profile (power vs control, starter depth, closer stuff), and
    defensive alignment to suggest values for OOTP's strategy sliders:

    Offensive: Stealing Bases, Base-Running, Hit&Run, Sac Bunt, Squeeze, Bunt for Hit
    Pitching/Defense: Pitch Around, IBB, Hold Runners, Play Infield In,
                      Play Corners In, Guard Lines, Infield Shifts, Outfield Shifts
    Substitution: Hook SP, Hook RP, L/R Pitching Matchups, L/R Batting Matchups,
                  Pinch-Hit for Position Players, Use Pinch Runners, Use of Openers

    Returns dict: response (AI text with recommendations + reasoning),
                  tokens_used, model, error
    """
    ai_config = get_ai_config()
    if not ai_config["ready"]:
        return {"response": None, "tokens_used": 0, "model": None, "error": ai_config["message"]}

    from app.core.database import get_connection
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        context = _build_manager_review_context(conn)
    except Exception as e:
        context = f"(Could not build context: {e})"
    finally:
        if close_conn:
            conn.close()

    system_prompt = (
        "You are an OOTP 27 tactical coordinator helping set in-game strategy sliders "
        "to MATCH the team's roster strengths. OOTP's strategy settings drive AI manager "
        "decisions during simulated games, so wrong settings waste roster value.\n\n"
        "Scale for each slider: 1 (Never/Conservative/Quick) through 5 (Frequently/Aggressive/Slow), "
        "with 3 as neutral.\n\n"
        "KEY TACTICAL PRINCIPLES:\n"
        "- High team SPEED (avg >=55) + low POWER -> aggressive baserunning, steal more, hit&run\n"
        "- High POWER (avg >=75) + low SPEED -> swing away, don't bunt/steal (waste of outs)\n"
        "- High CONTACT + low POWER -> hit&run, bunt for hit, manufacture runs\n"
        "- Weak bullpen -> quick hook on SP, aggressive L/R matchups on RP\n"
        "- Strong closer but weak setup -> hook SP late, protect leads aggressively\n"
        "- Weak defense (low IF/OF range) -> shift more, don't play infield in\n"
        "- Power pitchers (high STU) -> let them work, slower hooks\n"
        "- Control pitchers (high CTRL, low STU) -> hook quicker if struggling\n"
        "- No LH bullpen arms -> don't bother with LR pitching matchups\n"
        "- LH-heavy lineup vs RHP-heavy league -> prefer LR batting matchups\n\n"
        "Your response format — use EXACTLY this structure:\n\n"
        "**Team Identity**: (one line)\n\n"
        "**OFFENSIVE STRATEGY**\n"
        "- Stealing Bases: [1-5] — (one-line reason)\n"
        "- Base-Running: [1-5] — (reason)\n"
        "- Hit & Run: [1-5] — (reason)\n"
        "- Sacrifice Bunt: [1-5] — (reason)\n"
        "- Bunt for Hit: [1-5] — (reason)\n\n"
        "**PITCHING & DEFENSE**\n"
        "- Hook Starting Pitchers: [1-5] — (reason)\n"
        "- Hook Relievers: [1-5] — (reason)\n"
        "- Infield Shifts: [1-5] — (reason)\n"
        "- Outfield Shifts: [1-5] — (reason)\n"
        "- Play Infield In: [1-5] — (reason)\n\n"
        "**SUBSTITUTION**\n"
        "- L/R Pitching Matchups: [1-5] — (reason)\n"
        "- L/R Batting Matchups: [1-5] — (reason)\n"
        "- Pinch-Hit for Position Players: [1-5] — (reason)\n"
        "- Use Pinch Runners: [1-5] — (reason)\n\n"
        "**One-line summary**: (how to play this team)\n\n"
        "Be blunt and specific. Tie recommendations to actual roster numbers. Max 450 words."
    )

    user_message = f"=== ROSTER STATE ===\n{context}\n\n=== RECOMMEND STRATEGY ==="

    try:
        if ai_config["provider"] == "gemini":
            return _call_gemini(system_prompt, user_message, ai_config, max_tokens=1400)
        else:
            return _call_anthropic(system_prompt, user_message, ai_config, max_tokens=1400)
    except Exception as e:
        logger.error(f"Strategy recommendations API error: {e}", exc_info=True)
        return {"response": None, "tokens_used": 0, "model": None, "error": f"API error: {e}"}
