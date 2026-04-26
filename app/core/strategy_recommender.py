"""Strategy-slider recommendations from active roster composition.

Mirrors OOTP's in-game Strategy panel (Offensive / Pitching & Defensive /
Substitution) and recommends slider positions based on ratings/splits of
the players actually on the roster.

Output slider positions are 0-100 to match OOTP's slider range:
  0   = hard-left label (e.g. "Never", "Quick", "Conservative", "Don't Care")
  50  = neutral
  100 = hard-right label (e.g. "Frequently", "Slow", "Aggressive", "Prefer")

UAT 2026-04-25 user feedback: cover EVERY slider OOTP exposes, not just
the top 8 by impact. The user wants a 1:1 mapping with the in-game
Strategy screen so they can mirror our recs directly.

Coverage map (✓ = implemented):
  Offensive:           Stealing Bases ✓, Base-Running ✓, Use Hit & Run ✓,
                       Sacrifice Bunt ✓, Use Squeeze Bunt Play ✓,
                       Bunt for Hit ✓
  Pitching/Defensive:  Pitch Around ✓, Intentional Walk ✓, Hold Runners ✓,
                       Play Infield In ✓, Play Corners In ✓, Guard Lines ✓,
                       Use Infield Shifts ✓, Use Outfield Shifts ✓,
                       Shift OF Depth ✓
  Substitution:        Hook Starting Pitchers ✓, Hook Relievers ✓,
                       L/R Pitching Matchups ✓, L/R Batting Matchups ✓,
                       Pinch-Hit for Position Players ✓, Use Pinch Runners ✓

Each recommendation includes:
  - key         slider id (matches OOTP label)
  - label       display label
  - section     "Offensive" / "Pitching" / "Substitution"
  - position    0-100
  - bucket      "Never" / "Rarely" / "Normal" / "Frequently" / etc.
  - reason      one-line justification keyed to roster data
  - impact      priority score (higher = more impactful for THIS roster)
"""
from __future__ import annotations

import statistics


def _safe(v, default=0.0):
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bucket_by_position(pos: float, labels: tuple[str, ...]) -> str:
    """Given a 0-100 slider position and N ordered labels, pick the bucket."""
    n = len(labels)
    idx = int(min(n - 1, max(0, pos / 100.0 * n)))
    return labels[idx]


OFFENSE_LABELS_5 = ("Never", "Rarely", "Normal", "Often", "Frequently")
OFFENSE_LABELS_3 = ("Conservative", "Normal", "Aggressive")
HOOK_LABELS = ("Quick", "Normal-Quick", "Normal", "Normal-Slow", "Slow")
MATCHUP_LABELS = ("Don't Care", "Light", "Normal", "Lean", "Prefer")


def _avg(values):
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else 0.0


def recommend_strategy(
    active_batters: list[dict],
    active_pitchers: list[dict],
    bench_batters: list[dict] | None = None,
    bullpen_pitchers: list[dict] | None = None,
    starters: list[dict] | None = None,
) -> list[dict]:
    """Generate ranked slider recommendations for the active roster.

    Each input list is a list of dicts. Expected rating keys (per player):
      Batters:  contact, power, gap_power, eye, avoid_ks,
                speed, stealing, baserunning, bunt_for_hit,
                contact_vl, contact_vr, power_vl, power_vr
      Pitchers: stuff, movement, control, stamina, hold, p_hr,
                stuff_vl, stuff_vr, pitcher_role_name, meta_score_pitching

    Missing players or empty lists are handled gracefully — any slider that
    can't be scored confidently is dropped from the output.
    """
    bench_batters = bench_batters or []
    bullpen_pitchers = bullpen_pitchers or []
    starters = starters or [p for p in active_pitchers
                            if (p.get('pitcher_role_name') or '').upper() == 'SP']

    recs: list[dict] = []

    # ────────────────────────────────────────────────────────────
    # OFFENSIVE STRATEGY
    # ────────────────────────────────────────────────────────────

    # 1) Stealing Bases — speed + stealing composite
    if active_batters:
        spd = _avg([_safe(b.get('speed')) for b in active_batters])
        stl = _avg([_safe(b.get('stealing')) for b in active_batters])
        comp = (spd + stl) / 2.0
        # 30 → Never, 50 → Normal, 70+ → Frequently
        pos = max(0, min(100, (comp - 30) / 40 * 100))
        recs.append({
            'key': 'stealing',
            'label': 'Stealing Bases',
            'section': 'Offensive',
            'position': pos,
            'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
            'reason': f"Team speed {spd:.0f} / stealing {stl:.0f}",
            'impact': abs(pos - 50),  # further from neutral = more impactful
        })

    # 2) Base-Running — baserunning + speed
    if active_batters:
        br = _avg([_safe(b.get('baserunning')) for b in active_batters])
        sp = _avg([_safe(b.get('speed')) for b in active_batters])
        comp = (br + sp) / 2.0
        pos = max(0, min(100, (comp - 30) / 40 * 100))
        recs.append({
            'key': 'baserunning',
            'label': 'Base-Running',
            'section': 'Offensive',
            'position': pos,
            'bucket': _bucket_by_position(pos, OFFENSE_LABELS_3),
            'reason': f"Baserunning {br:.0f} / speed {sp:.0f}",
            'impact': abs(pos - 50),
        })

    # 3) Hit & Run — high contact + low K% is ideal
    if active_batters:
        con = _avg([_safe(b.get('contact')) for b in active_batters])
        avk = _avg([_safe(b.get('avoid_ks')) for b in active_batters])
        comp = (con + avk) / 2.0
        pos = max(0, min(100, (comp - 40) / 40 * 100))
        recs.append({
            'key': 'hit_and_run',
            'label': 'Use Hit & Run',
            'section': 'Offensive',
            'position': pos,
            'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
            'reason': f"Contact {con:.0f} / avoid-K {avk:.0f}",
            'impact': abs(pos - 50),
        })

    # 4) Sacrifice Bunt — more often if low team power (small ball)
    if active_batters:
        pwr = _avg([_safe(b.get('power')) for b in active_batters])
        gap = _avg([_safe(b.get('gap_power')) for b in active_batters])
        comp = (pwr + gap) / 2.0
        # Inverse — low power → bunt more
        # 70+ power → Rarely (pos=20), 40 power → Often (pos=70)
        pos = max(0, min(100, 110 - (comp - 30)))
        recs.append({
            'key': 'sac_bunt',
            'label': 'Sacrifice Bunt',
            'section': 'Offensive',
            'position': pos,
            'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
            'reason': f"Team power {pwr:.0f}/gap {gap:.0f} — "
                      + ("small ball" if pos > 50 else "swing away"),
            'impact': abs(pos - 50) * 0.7,  # less impactful than stealing
        })

    # 4b) Squeeze Bunt Play — same logic as sacrifice bunt but tied to
    # team contact (need to lay it down) AND speed (runner from 3rd has
    # to score). Low power + high contact + decent speed → use it.
    if active_batters:
        pwr = _avg([_safe(b.get('power')) for b in active_batters])
        con = _avg([_safe(b.get('contact')) for b in active_batters])
        spd = _avg([_safe(b.get('speed')) for b in active_batters])
        # Low power AND high contact AND speed — small ball squeeze
        if pwr < 65 and con > 70 and spd > 55:
            pos = 70
        elif pwr >= 75:
            pos = 5  # power team — never squeeze
        else:
            pos = 30
        recs.append({
            'key': 'squeeze_bunt',
            'label': 'Use Squeeze Bunt Play',
            'section': 'Offensive',
            'position': pos,
            'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
            'reason': f"Team power {pwr:.0f}/contact {con:.0f}/speed {spd:.0f}",
            'impact': abs(pos - 50) * 0.4,
        })

    # 4c) Bunt for Hit — speed + bunt-for-hit skill drives this. The
    # per-player overrides handle individual cases; the team default
    # leans toward "Often" only when the lineup has multiple speed +
    # bunt-skill combos.
    if active_batters:
        spd = _avg([_safe(b.get('speed')) for b in active_batters])
        bnh = _avg([_safe(b.get('bunt_for_hit')) for b in active_batters])
        # spd 70+ AND bnh 65+ → Often (70). Otherwise Rarely.
        if spd >= 70 and bnh >= 65:
            pos = 70
        elif spd <= 45:
            pos = 10
        else:
            pos = 35
        recs.append({
            'key': 'bunt_for_hit',
            'label': 'Bunt for Hit',
            'section': 'Offensive',
            'position': pos,
            'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
            'reason': f"Team speed {spd:.0f}/bunt-for-hit {bnh:.0f}",
            'impact': abs(pos - 50) * 0.4,
        })

    # ────────────────────────────────────────────────────────────
    # PITCHING & DEFENSIVE STRATEGY
    # ────────────────────────────────────────────────────────────

    # 5) Hold Runners — based on catcher arm of active catchers
    catchers = [b for b in active_batters if (b.get('position') or '').upper() == 'C']
    if catchers:
        arm = _avg([_safe(c.get('catcher_arm')) for c in catchers])
        # arm 100 → pos 50 (Normal); arm 50 → pos 100 (Frequently); arm 130 → pos 20 (Rarely)
        pos = max(0, min(100, 110 - (arm - 40)))
        if arm >= 95:
            tail = "elite arm, let them run"
        elif arm >= 75:
            tail = "above-avg arm, normal hold"
        elif arm >= 55:
            tail = "average arm, hold more"
        else:
            tail = "weak arm, hold often"
        recs.append({
            'key': 'hold_runners',
            'label': 'Hold Runners',
            'section': 'Pitching',
            'position': pos,
            'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
            'reason': f"Catcher arm {arm:.0f} — {tail}",
            'impact': abs(pos - 50) * 0.6,
        })

    # 5b) Pitch Around — only safe with high control. Recommend pitching
    # around dangerous hitters more often when YOUR pitchers can hit
    # spots without walking the next batter into trouble.
    if active_pitchers:
        ctrl = _avg([_safe(p.get('control')) for p in active_pitchers])
        # ctrl 75+ → Often (70); ctrl 50- → Never (10); 50-75 linear.
        pos = max(0, min(100, (ctrl - 50) / 30 * 70))
        recs.append({
            'key': 'pitch_around',
            'label': 'Pitch Around',
            'section': 'Pitching',
            'position': pos,
            'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
            'reason': f"Staff control {ctrl:.0f} — "
                      + ("trust the command" if pos > 50
                         else "can't afford the walk"),
            'impact': abs(pos - 50) * 0.5,
        })

    # 5c) Intentional Walk — also control-driven but more conservative
    # than Pitch Around. Default Normal unless control is very high or
    # very low.
    if active_pitchers:
        ctrl = _avg([_safe(p.get('control')) for p in active_pitchers])
        if ctrl >= 78:
            pos = 60   # plus-command staff can spot the IBB
        elif ctrl <= 50:
            pos = 25   # poor command — walking on purpose costs runs
        else:
            pos = 45   # default Normal-ish
        recs.append({
            'key': 'intentional_walk',
            'label': 'Intentional Walk',
            'section': 'Pitching',
            'position': pos,
            'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
            'reason': f"Staff control {ctrl:.0f}",
            'impact': abs(pos - 50) * 0.4,
        })

    # 5d) Play Infield In — situational defensive aggression. Helped by
    # strong infield arm (throw runner out at home) and high range
    # (recover from in-pulled position). Weak infield → don't risk it.
    if active_batters:
        infielders = [b for b in active_batters
                       if (b.get('position') or '').upper() in ('1B', '2B', 'SS', '3B')]
        if infielders:
            inf_arm = _avg([_safe(b.get('infield_arm')) for b in infielders])
            inf_rng = _avg([_safe(b.get('infield_range')) for b in infielders])
            comp = (inf_arm + inf_rng) / 2
            # comp 70+ → Frequently (75); 45- → Rarely (25)
            pos = max(0, min(100, (comp - 45) / 30 * 60 + 25))
            recs.append({
                'key': 'play_infield_in',
                'label': 'Play Infield In',
                'section': 'Pitching',
                'position': pos,
                'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
                'reason': f"Infield arm {inf_arm:.0f}/range {inf_rng:.0f}",
                'impact': abs(pos - 50) * 0.4,
            })

            # 5e) Play Corners In — same signal but only weighs 1B/3B arm.
            corners = [b for b in active_batters
                        if (b.get('position') or '').upper() in ('1B', '3B')]
            if corners:
                c_arm = _avg([_safe(b.get('infield_arm')) for b in corners])
                pos = max(0, min(100, (c_arm - 45) / 30 * 60 + 25))
                recs.append({
                    'key': 'play_corners_in',
                    'label': 'Play Corners In',
                    'section': 'Pitching',
                    'position': pos,
                    'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
                    'reason': f"Corner-INF arm {c_arm:.0f}",
                    'impact': abs(pos - 50) * 0.3,
                })

    # 5f) Guard Lines — usually situational; default Normal. Tilt
    # toward "Frequently" only when corner infielders have weak range
    # (can't recover from a line drive without help).
    if active_batters:
        corners = [b for b in active_batters
                    if (b.get('position') or '').upper() in ('1B', '3B')]
        if corners:
            c_rng = _avg([_safe(b.get('infield_range')) for b in corners])
            # rng 70+ → Rarely (30); rng 45- → Frequently (75)
            pos = max(0, min(100, 110 - c_rng))
            recs.append({
                'key': 'guard_lines',
                'label': 'Guard Lines',
                'section': 'Pitching',
                'position': pos,
                'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
                'reason': f"Corner-INF range {c_rng:.0f}",
                'impact': abs(pos - 50) * 0.25,
            })

    # 5g) Use Infield Shifts — modern analytics default Frequently.
    # Tilt toward "Always" when infielders have the range to get back
    # in time; pull back if range is poor (defensive-OOM risk).
    if active_batters:
        infielders = [b for b in active_batters
                       if (b.get('position') or '').upper() in ('1B', '2B', 'SS', '3B')]
        if infielders:
            inf_rng = _avg([_safe(b.get('infield_range')) for b in infielders])
            # rng 70+ → Frequently (85); 50- → Rarely (30)
            pos = max(0, min(100, (inf_rng - 50) / 30 * 60 + 30))
            recs.append({
                'key': 'infield_shifts',
                'label': 'Use Infield Shifts',
                'section': 'Pitching',
                'position': pos,
                'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
                'reason': f"Infield range {inf_rng:.0f} — "
                          + ("rangy fielders, shift away" if pos > 50
                             else "limited range, stay home"),
                'impact': abs(pos - 50) * 0.4,
            })

    # 5h) Use Outfield Shifts — same idea on OF range.
    if active_batters:
        outfielders = [b for b in active_batters
                        if (b.get('position') or '').upper() in ('LF', 'CF', 'RF')]
        if outfielders:
            of_rng = _avg([_safe(b.get('of_range')) for b in outfielders])
            pos = max(0, min(100, (of_rng - 50) / 30 * 60 + 30))
            recs.append({
                'key': 'outfield_shifts',
                'label': 'Use Outfield Shifts',
                'section': 'Pitching',
                'position': pos,
                'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
                'reason': f"OF range {of_rng:.0f}",
                'impact': abs(pos - 50) * 0.3,
            })

            # 5i) Shift OF Depth — default Normal. Tilt deep when OF arm
            # is weak (can't throw from shallow); shallow when OF range
            # is great + opposing power is unknown (default to data).
            of_arm = _avg([_safe(b.get('of_arm')) for b in outfielders])
            if of_arm <= 50:
                pos = 70   # shift deep — weak arm, want to keep balls in front
                reason = f"OF arm {of_arm:.0f} — shift deeper to keep balls in front"
            elif of_rng >= 75:
                pos = 35   # play shallow — speed makes up for it
                reason = f"OF range {of_rng:.0f} — play shallow, run things down"
            else:
                pos = 50
                reason = f"OF range {of_rng:.0f}/arm {of_arm:.0f} — neutral"
            recs.append({
                'key': 'of_depth',
                'label': 'Shift OF Depth',
                'section': 'Pitching',
                'position': pos,
                'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
                'reason': reason,
                'impact': abs(pos - 50) * 0.3,
            })

    # ────────────────────────────────────────────────────────────
    # SUBSTITUTION STRATEGY
    # ────────────────────────────────────────────────────────────

    # 6) Hook Starting Pitchers — if bullpen is STRONG vs SPs, hook quickly
    if starters and bullpen_pitchers:
        sp_meta = _avg([_safe(p.get('meta_score_pitching') or p.get('meta_score'))
                        for p in starters])
        bp_meta = _avg([_safe(p.get('meta_score_pitching') or p.get('meta_score'))
                        for p in bullpen_pitchers])
        # Ratio <1: bullpen weaker → slow hook; >1: bullpen stronger → quick hook
        ratio = bp_meta / sp_meta if sp_meta > 0 else 1.0
        # ratio=0.6 → Slow (80), ratio=1.2 → Quick (10)
        pos = max(0, min(100, 110 - (ratio - 0.5) * 100))
        recs.append({
            'key': 'hook_sp',
            'label': 'Hook Starting Pitchers',
            'section': 'Substitution',
            'position': pos,
            'bucket': _bucket_by_position(pos, HOOK_LABELS),
            'reason': f"Bullpen/SP meta ratio {ratio:.2f} "
                      + ("— pen stronger" if ratio > 1.05
                         else "— SPs stronger" if ratio < 0.95
                         else "— balanced"),
            'impact': abs(pos - 50),
        })

    # 6b) Hook Relievers — driven by bullpen depth. Many high-meta RPs
    # → quick hook (always have a fresh arm). Thin bullpen → slow hook
    # (can't afford to burn arms early).
    if bullpen_pitchers:
        bp_metas = sorted(
            [_safe(p.get('meta_score_pitching') or p.get('meta_score'))
             for p in bullpen_pitchers],
            reverse=True,
        )
        # Top 3 RPs vs bottom RPs — wider gap means deeper pen
        top3 = _avg(bp_metas[:3]) if len(bp_metas) >= 3 else _avg(bp_metas)
        bot = _avg(bp_metas[-2:]) if len(bp_metas) >= 5 else top3
        # gap > 100 → Quick (deep pen); gap < 30 → Slow (only 2-3 trusted arms)
        if top3 > 0 and bot > 0:
            gap = top3 - bot
            pos = max(0, min(100, 80 - (gap - 30) * 0.7))
            recs.append({
                'key': 'hook_rp',
                'label': 'Hook Relievers',
                'section': 'Substitution',
                'position': pos,
                'bucket': _bucket_by_position(pos, HOOK_LABELS),
                'reason': f"BP top-3 meta {top3:.0f} vs bottom {bot:.0f} — "
                          + ("deep pen, quick hook" if pos < 50
                             else "thin pen, ride starters"),
                'impact': abs(pos - 50) * 0.6,
            })

    # 7) L/R Pitching Matchups — if platoon split variance in bullpen is big
    if bullpen_pitchers and len(bullpen_pitchers) >= 3:
        splits = []
        for p in bullpen_pitchers:
            vl = _safe(p.get('stuff_vl'))
            vr = _safe(p.get('stuff_vr'))
            if vl > 0 and vr > 0:
                splits.append(abs(vl - vr))
        if splits:
            avg_split = _avg(splits)
            # avg_split 5 → Don't Care (20), avg_split 25 → Prefer (90)
            pos = max(0, min(100, (avg_split - 3) / 25 * 100))
            recs.append({
                'key': 'lr_pitching',
                'label': 'L/R Pitching Matchups',
                'section': 'Substitution',
                'position': pos,
                'bucket': _bucket_by_position(pos, MATCHUP_LABELS),
                'reason': f"Avg bullpen platoon gap {avg_split:.0f} pts",
                'impact': abs(pos - 50) * 0.9,
            })

    # 8) L/R Batting Matchups — if batters have big splits, prefer matchups
    if active_batters and len(active_batters) >= 5:
        splits = []
        for b in active_batters:
            vl = _safe(b.get('contact_vl'))
            vr = _safe(b.get('contact_vr'))
            if vl > 0 and vr > 0:
                splits.append(abs(vl - vr))
        if splits:
            avg_split = _avg(splits)
            pos = max(0, min(100, (avg_split - 3) / 25 * 100))
            recs.append({
                'key': 'lr_batting',
                'label': 'L/R Batting Matchups',
                'section': 'Substitution',
                'position': pos,
                'bucket': _bucket_by_position(pos, MATCHUP_LABELS),
                'reason': f"Avg lineup platoon gap {avg_split:.0f} pts",
                'impact': abs(pos - 50) * 0.9,
            })

    # 9) Pinch Runners — if bench has fast players + starters are slow
    if bench_batters and active_batters:
        bench_spd = max(
            (_safe(b.get('speed')) for b in bench_batters), default=0
        )
        starter_spd = _avg([_safe(b.get('speed')) for b in active_batters])
        gap = bench_spd - starter_spd
        # gap=15 → Frequently, gap<=0 → Never
        pos = max(0, min(100, (gap + 5) / 30 * 100))
        recs.append({
            'key': 'pinch_runners',
            'label': 'Use Pinch Runners',
            'section': 'Substitution',
            'position': pos,
            'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
            'reason': f"Bench top speed {bench_spd:.0f} vs lineup avg {starter_spd:.0f}",
            'impact': abs(pos - 50) * 0.5,
        })

    # 10) Pinch-Hit for Position Players — strong bench contact vs weak starter
    if bench_batters and active_batters:
        bench_con = max(
            (_safe(b.get('contact')) for b in bench_batters), default=0
        )
        starter_con_min = min(
            (_safe(b.get('contact')) for b in active_batters), default=100
        )
        diff = bench_con - starter_con_min
        # diff=20 → Frequently, diff<=0 → Never
        pos = max(0, min(100, diff / 30 * 100))
        recs.append({
            'key': 'pinch_hit',
            'label': 'Pinch-Hit for Position Players',
            'section': 'Substitution',
            'position': pos,
            'bucket': _bucket_by_position(pos, OFFENSE_LABELS_5),
            'reason': f"Best bench contact {bench_con:.0f} vs weakest starter {starter_con_min:.0f}",
            'impact': abs(pos - 50) * 0.5,
        })

    # Order by section first, then by impact within each section. Returns
    # ALL recommendations — the user explicitly wants every OOTP slider
    # surfaced so the in-game Strategy screen can be mirrored 1:1.
    section_order = {'Offensive': 0, 'Pitching': 1, 'Substitution': 2}
    recs.sort(key=lambda r: (section_order.get(r['section'], 99), -r['impact']))
    return recs


# ════════════════════════════════════════════════════════════════════
# PER-PLAYER STRATEGY OVERRIDES
# Surface rating-driven overrides for OOTP's Player Strategy tab.
# We only flag an override when the player's ratings clearly warrant
# deviation from the team default — middle-of-the-road players inherit.
# ════════════════════════════════════════════════════════════════════


def recommend_batter_strategy(player: dict) -> list[dict]:
    """Per-batter offensive strategy overrides.

    Returns a list of overrides (only non-default recommendations). Each
    entry: {slider, bucket, position, reason}. Empty list = inherit team.
    """
    overrides: list[dict] = []
    spd = _safe(player.get('speed'))
    stl = _safe(player.get('stealing'))
    brs = _safe(player.get('baserunning'))
    con = _safe(player.get('contact'))
    avk = _safe(player.get('avoid_ks'))
    pwr = _safe(player.get('power'))
    gap = _safe(player.get('gap_power'))
    bnh = _safe(player.get('bunt_for_hit'))
    sac = _safe(player.get('sac_bunt'))
    eye = _safe(player.get('eye'))

    # Stealing Bases — override for clear extremes
    steal_comp = (spd + stl) / 2
    if steal_comp >= 75:
        overrides.append({
            'slider': 'Stealing Bases', 'bucket': 'Frequently', 'position': 90,
            'reason': f'Speed {spd:.0f}/Stealing {stl:.0f} — green light'})
    elif steal_comp <= 30:
        overrides.append({
            'slider': 'Stealing Bases', 'bucket': 'Never', 'position': 5,
            'reason': f'Speed {spd:.0f}/Stealing {stl:.0f} — anchor'})

    # Base-Running
    br_comp = (brs + spd) / 2
    if br_comp >= 75:
        overrides.append({
            'slider': 'Base-Running', 'bucket': 'Aggressive', 'position': 90,
            'reason': f'Baserunning {brs:.0f}/Speed {spd:.0f}'})
    elif br_comp <= 35:
        overrides.append({
            'slider': 'Base-Running', 'bucket': 'Conservative', 'position': 10,
            'reason': f'Baserunning {brs:.0f}/Speed {spd:.0f}'})

    # Hit & Run — high contact + avoid_ks
    hr_comp = (con + avk) / 2
    if hr_comp >= 78 and con >= 75:
        overrides.append({
            'slider': 'Hit & Run', 'bucket': 'Frequently', 'position': 80,
            'reason': f'Contact {con:.0f}/AvK {avk:.0f} — protect the bat'})
    elif con <= 50 or avk <= 40:
        overrides.append({
            'slider': 'Hit & Run', 'bucket': 'Never', 'position': 5,
            'reason': f'Contact {con:.0f}/AvK {avk:.0f} — too many whiffs'})

    # Sacrifice Bunt — ABSOLUTELY never for power hitters
    if pwr >= 70 or gap >= 75:
        overrides.append({
            'slider': 'Sacrifice Bunt', 'bucket': 'Never', 'position': 0,
            'reason': f'Power {pwr:.0f}/Gap {gap:.0f} — never waste the out'})
    elif sac >= 75 and pwr <= 45:
        overrides.append({
            'slider': 'Sacrifice Bunt', 'bucket': 'Often', 'position': 70,
            'reason': f'Sac-bunt {sac:.0f} + low power {pwr:.0f}'})

    # Bunt for Hit — fast player with bunt skill
    if bnh >= 75 and spd >= 70:
        overrides.append({
            'slider': 'Bunt for Hit', 'bucket': 'Often', 'position': 75,
            'reason': f'Bunt-for-hit {bnh:.0f}/Speed {spd:.0f}'})
    elif pwr >= 75:
        overrides.append({
            'slider': 'Bunt for Hit', 'bucket': 'Never', 'position': 0,
            'reason': f'Power {pwr:.0f} — swing the bat'})

    return overrides


def recommend_pitcher_strategy(player: dict) -> list[dict]:
    """Per-pitcher strategy overrides.

    Returns list of overrides. Pitcher role matters — SP-only recommendations
    (Hook as Starter, Use as Opener) only fire for SPs.
    """
    overrides: list[dict] = []
    role = (player.get('pitcher_role_name') or '').upper()
    is_sp = role == 'SP'
    is_rp = role in ('RP', 'CL')
    is_cl = role == 'CL'

    stu = _safe(player.get('stuff'))
    mov = _safe(player.get('movement'))
    ctrl = _safe(player.get('control'))
    hld = _safe(player.get('hold'))
    stam = _safe(player.get('stamina'))
    meta = _safe(player.get('meta_score_pitching') or player.get('meta_score'))
    p_hr = _safe(player.get('p_hr'))

    # Hook as Starter — stamina-driven (SP only)
    if is_sp:
        if stam >= 80:
            overrides.append({
                'slider': 'Hook as Starter', 'bucket': 'Slow', 'position': 85,
                'reason': f'Stamina {stam:.0f} — trust him deep'})
        elif stam <= 55:
            overrides.append({
                'slider': 'Hook as Starter', 'bucket': 'Quick', 'position': 15,
                'reason': f'Stamina {stam:.0f} — get him out early'})

    # Hook as Reliever — for elite relievers, slow hook (let them finish)
    if is_rp:
        if meta >= 400 or (stu >= 80 and ctrl >= 65):
            overrides.append({
                'slider': 'Hook as Reliever', 'bucket': 'Slow', 'position': 80,
                'reason': f'Elite RP (stuff {stu:.0f}/ctrl {ctrl:.0f}) — finish the inning'})
        elif stu <= 50:
            overrides.append({
                'slider': 'Hook as Reliever', 'bucket': 'Quick', 'position': 15,
                'reason': f'Stuff {stu:.0f} — short leash'})

    # Hold Runner — hold rating driven
    if hld >= 75:
        overrides.append({
            'slider': 'Hold Runner', 'bucket': 'Frequently', 'position': 85,
            'reason': f'Hold {hld:.0f} — shut the run game down'})
    elif hld <= 40:
        overrides.append({
            'slider': 'Hold Runner', 'bucket': 'Rarely', 'position': 20,
            'reason': f'Hold {hld:.0f} — focus on the hitter'})

    # Pitch Around — high control + low HR tolerance → pitch around more
    if ctrl >= 75 and p_hr >= 70:
        overrides.append({
            'slider': 'Pitch Around', 'bucket': 'Often', 'position': 70,
            'reason': f'Control {ctrl:.0f}/HR-suppress {p_hr:.0f} — safe nibble'})
    elif ctrl <= 45:
        overrides.append({
            'slider': 'Pitch Around', 'bucket': 'Never', 'position': 5,
            'reason': f'Control {ctrl:.0f} — can\'t afford the walk'})

    # Intentional Walk — high control pitchers can afford it
    if ctrl >= 75:
        overrides.append({
            'slider': 'Intentional Walk', 'bucket': 'Normal', 'position': 50,
            'reason': f'Control {ctrl:.0f} — trust the command'})

    # Use as Opener — only if stuff high + stamina low
    if is_sp and stam <= 50 and stu >= 70:
        overrides.append({
            'slider': 'Use as Opener', 'bucket': 'Yes', 'position': 100,
            'reason': f'Low stamina ({stam:.0f}) but strong stuff ({stu:.0f}) — opener candidate'})

    return overrides


def recommend_all_player_strategy(
    active_batters: list[dict],
    active_pitchers: list[dict],
) -> list[dict]:
    """Build per-player override list sorted by override count.

    Returns one entry per player: {name, pos, role, overrides[]}. Players
    with 0 overrides are dropped (they inherit team strategy).
    """
    out: list[dict] = []

    for p in active_batters:
        ov = recommend_batter_strategy(p)
        if ov:
            out.append({
                'name': p.get('player_name') or p.get('card_title') or '?',
                'pos': p.get('position') or '?',
                'role': 'Batter',
                'overrides': ov,
            })

    for p in active_pitchers:
        ov = recommend_pitcher_strategy(p)
        if ov:
            out.append({
                'name': p.get('player_name') or p.get('card_title') or '?',
                'pos': (p.get('pitcher_role_name') or '?'),
                'role': 'Pitcher',
                'overrides': ov,
            })

    # Sort by override count descending (most impactful first)
    out.sort(key=lambda x: -len(x['overrides']))
    return out
