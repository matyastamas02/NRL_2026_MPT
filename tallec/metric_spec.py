# -*- coding: utf-8 -*-
"""Mike's position metric specification, as code.

Source: `BOSC Positions.xlsx`, sent 31 August 2026, with the forwards regrouped on
19 September at Leeds's request. Eight position blocks, each with four categories, each
category holding three metrics, and every metric given in two forms — a **volume** (how
much he contributes) and a **rate** (how efficiently or often he contributes when given
the chance). Twenty-four figures per position.

This module is the single translation of that workbook into the codebase: the metric
names, which feed column each one comes from, how a rate is derived, and — the part the
workbook cannot know — whether the data actually supports it. Anything downstream reads
the spec from here rather than re-deriving it from the spreadsheet, so the spreadsheet
can be re-sent without silently changing what the app computes.

Five things in the workbook could not be answered from the data alone. Mike settled all
five on 31 August 2026 and his answers are recorded in `RESOLVED`, because in six months
nobody will remember which choices were decisions and which were guesses. Two of the
answers name data this extract does not hold; those became `DATA_REQUESTS` rather than
open questions, since they are things to ask Stats Perform for, not things to decide.
"""

# ── position blocks ──────────────────────────────────────────────────────────
# Mike's label -> the rating engine's position group. The note below records how the
# forwards came to be grouped this way, because it has changed twice.
POSITIONS = {
    "FB":       dict(engine_group="Fullback",  label="Fullback"),
    "WG":       dict(engine_group="Winger",    label="Winger"),
    "CT":       dict(engine_group="Centre",    label="Centre"),
    "Halves":   dict(engine_group="Halves",    label="Halves"),
    "Hooker":   dict(engine_group="Hooker",    label="Hooker"),
    "Middles":  dict(engine_group="Middles",   label="Middles (prop and lock)"),
    "Edge":     dict(engine_group="Edge",      label="Edge (second row)"),
    "IT":       dict(engine_group="Bench",     label="Interchange"),
}

# The forwards have been grouped three ways in three weeks, so the sequence is worth
# recording rather than just the endpoint.
#
#   until 31 Aug   Prop | Back Row (second row and lock together)
#   31 Aug         Prop | Back Row | Lock — Mike: "Lock plays in the middle, sometimes
#                  like a halfback and sometimes like a prop, whereas Back Row is more
#                  defined as a role."
#   19 Sep         Middles (prop and lock) | Edge (second row) — Leeds asked for it.
#                  Mike: "I don't really agree with it but that's what they asked for so
#                  we should change. Makes Edge more distinct anyway."
#
# It is the client's competition to describe, and it happens to repair the weakest pool
# in the system: Super League Lock stood at 45 players, where one place in the order was
# worth more than two points on the 0-100 scale. Middles is 179 there and 225 in the NRL.
#
# The Prop and Lock blocks in Mike's sheet differed in exactly one slot — hit-up metres
# against line-break assists — so Middles is the Prop profile. Hit-ups are what every
# middle forward does; line-break assists describe the ball-playing lock, which is the
# distinction Leeds is asking us to stop drawing. If they would rather keep it, the
# alternative is to carry both and let Middles run to thirteen metrics instead of twelve.

# Interchange is a role, and a player whose whole career reads Interchange has never
# started. A secondary position can be recovered from his starts in ANOTHER competition
# — a permanent bench forward in the NRL who started in the NSW Cup. That resolves 41
# of 349 cases (12%): 21 props, 9 back row, 8 hookers, 2 halves, 1 centre, which is the
# shape Mike described. The other 297 look unresolvable but mostly do not matter: their
# median is two matches, so they would not earn a rating either way. For the handful
# who do matter, a manual override is the right answer — Mike knows the players.
IT_SUBPOSITION = dict(
    method="career starting position in any other competition",
    resolved=41, unresolved=297, unresolved_median_matches=2,
    override="a small player_id -> position table Mike fills in by hand")

CATEGORIES = ["Yardage", "Attack", "Involvement", "Defence & Discipline"]

# ── feed columns ─────────────────────────────────────────────────────────────
# short code -> the Stats Perform column in player_match_raw
FEED = {
    "RM":        "Ball Runs - Metres Gained",
    "TB":        "Tackle Break",
    "KRM":       "Ball Run - Kick Return Metres",
    "PCM":       "Ball Runs - Post Contact Metres",
    "HUM":       "Ball Run - Hitup Metres",
    "DHM":       "Ball Run - Dummy Half Run Metres",
    "DHR":       "Ball Run - Dummy Half Run",
    "LB":        "Line Break",
    "LBA":       "Line Break Assist",
    "LBI":       "Line Break Involvement",
    "TryAssist": "Try Assists",
    "Tries":     "Try Scored - Total",
    "Offloads":  "Offload - Successful",
    "Runs":      "Ball Runs - Total",
    "Receipts":  "Receipts",
    "Support":   "Support",
    "Decoy":     "Decoy",
    "Passes":    "Pass - Attempted",
    "Errors":    "Errors",
    "Penalties": "Penalty - Total",
    "SetRestart": "Set Restart Conceded",
    "BreakCause": "Break Cause",
    "LongToOpen": "Kick - Long To Open",
    "KickDefused": "Kick Defused",
    "KickDefusedAtt": "Kick Defused Attempt",
    "TacklesMissed": "Tackle - Total Missed",
    "KDpct":     "Kick Defused %",
    "TEpct":     "Made Tackle %",
    "TacklesMade": "Tackle - Total Made",
    "PTBW":      "PTB - Won",
    "PTBtot":    "PTB - Total",
    "PTBWpct":   "PTB Won %",
    "KM":        "Kicks - Total Metres",
    "Kicks":     "Kicks - Total",
    "LongKicks": "Long Kicks - Total",
    "AttKicks":  "Kicks - In Attacking Half",
    "FortyTwenty": "Kick - 40/20",
    "GLDO":      "Goal Line Dropout - Total",
    "LER":       "Ball Run - Line Engaged",
    "Minutes":   "Minutes",
}

# ── what the data supports ───────────────────────────────────────────────────
# Measured 31 August 2026 over all 122,359 player-match rows. `usable_from` is the
# first season the column is populated; `note` says what limits it.
AVAILABILITY = {
    "PCM": dict(usable_from=2025, note=(
        "Absent before 2025 — 0-6% of rows carry a value, against 96-98% from 2025. "
        "Post-contact metres appear in six of the nine position profiles, so their "
        "Yardage category cannot be computed on the earlier seasons at all.")),
    "KTA": dict(missing_in=["NRL", "SL", "NSW", "QLD"], note=(
        "Kick try assists. Not in this extract at all — it has no column, which is why "
        "it is absent from FEED as well. The conversion metric names it and is computed "
        "without it, so every row of that metric is a floor rather than the figure the "
        "name promises. Declared missing everywhere so the degraded flag fires on all of "
        "them rather than the shortfall living only in a note nobody reads — see "
        "DATA-2.")),
    "SetRestart": dict(missing_in=["SL"], note=(
        "NULL on all 31,769 Super League rows; 8,012 conceded in the NRL, 2,456 in the "
        "NSW Cup, 3,295 in the Queensland Cup. Six-again is played in Super League, so "
        "this is a gap in the extract rather than in the sport, and it is worth asking "
        "Stats Perform for. Until it arrives, 'Combined Infringements' means penalties "
        "plus set restarts in Australia and penalties alone in Super League, which is "
        "not the same measure — see DATA-1.")),
    "KM": dict(usable_from=2023, note=(
        "Kick metres are effectively empty for NSW Cup and Queensland Cup in 2021-2022 "
        "(0-6% of rows) and populated from 2023. Affects the Halves Yardage category "
        "on those seasons.")),
    "LER": dict(note=(
        "Line-engaged runs swing by a factor of six between seasons of the same "
        "competition — 2.6% of runs in NSW Cup 2021 against 16% in 2023 — which is a "
        "change in recording, not in football. Usable within a season, not across "
        "them.")),
    "FortyTwenty": dict(note=(
        "Real but very rare: 214 in seven NRL seasons. Sound as a volume count, but at "
        "player-season level most halves will have nought or one, so it carries almost "
        "no signal for a rating.")),
    "GLDO": dict(note=(
        "3,731 in the NRL across seven seasons — real, but sparse per player, and it "
        "is the dropout TAKEN by the defending side rather than the kick that forced "
        "it. Mike named 'Kick - Forced Dropout', which is not in this extract; see "
        "DATA-2.")),
    "BreakCause": dict(note=(
        "A count, not a category code. 64,617 rows at nought, 35,154 at one, decaying "
        "monotonically to twelve, and it tracks missed tackles cleanly (1.4 missed at "
        "nought, 10.0 at twelve). An earlier reading of this module had it as a coded "
        "reason and was wrong.")),
}

# ── settled, and what is still outstanding ───────────────────────────────────
# Mike's answers, 31 August 2026. Kept because each one is a judgement that shapes the
# numbers, and in six months nobody will remember which were decisions and which were
# guesses.
RESOLVED = {
    "ninth block": (
        "It is Interchange, mislabelled. Mike also wants interchange players given a "
        "sub-position — 'usually 2x props, 1x hooker/half and 1x something else'. See "
        "IT_SUBPOSITION: it can be derived for 12% of them from their starts in another "
        "competition, and the rest have a median of two matches, so a manual override "
        "for the few who matter is the sensible completion."),
    "Back Row and Lock": (
        "Separate peer groups. 'Lock plays in the middle, sometimes like a halfback and "
        "sometimes like a prop, whereas Back Row is more defined as a role.' Requires "
        "sp_schema.POSITION_GROUP to stop folding Lock into Back Row."),
    "Combined Infringements": (
        "All penalties plus all set restarts conceded: 'Penalty - Total' plus "
        "'Set Restart Conceded'. Answered, but only computable in Australia until the "
        "Super League set-restart data arrives — DATA-1."),
    "LKS": (
        "Long Kick to Open, which is 'Kick - Long To Open' in the feed. Confirmed "
        "present."),
    "Break Cause": (
        "'When a player gets blamed for a line break conceded' — and it is a genuine "
        "count, so it can be used as a volume exactly as the sheet has it. The earlier "
        "objection in this module was mistaken."),
    "PCM and LER": (
        "'We have to keep this in if at all possible, so we might just have to deal "
        "with the inconsistencies.' Kept. The handling is to compute them where the "
        "data supports it and to mark every figure that rests on them, rather than "
        "quietly averaging across a boundary the data cannot cross."),
}

# What is left is not a decision any more — it is data we do not have. Both are worth
# asking Stats Perform for, since neither is a limitation of the sport.
DATA_REQUESTS = {
    "DATA-1": (
        "Set restarts conceded for Super League. NULL on every row of the extract, "
        "present throughout the three Australian competitions. Without it, discipline "
        "cannot be compared between a Super League forward and an NRL one."),
    "DATA-2": (
        "Two columns Mike named that this extract does not contain: 'Try Assist - "
        "Kick' (KTA, for the CV formula) and 'Kick - Forced Dropout' (FDO). The feed "
        "has 'Try Assists' and 'Try Scored - From Kick', neither of which is a kick try "
        "assist credited to the kicker, and it has the dropout taken rather than the "
        "kick that forced it. Both may exist in a fuller export."),
}

# ── the specification ────────────────────────────────────────────────────────
# For each position: category -> list of (volume, rate) pairs, exactly as the
# workbook gives them. `v` and `r` are Mike's own labels; `inputs` names the FEED
# codes each needs, so coverage can be resolved per competition and season.
def M(v, r, inputs, note=None, optional=(), blocked=None):
    """One metric pair.

    `optional` names inputs the metric can be computed without, at the cost of meaning
    something slightly narrower. The benchmark ranks players within one competition and
    season, so a narrower definition still orders that pool correctly — what it stops
    being is comparable ACROSS competitions, and rows computed that way are flagged
    `degraded` so nothing quietly compares them.

    `blocked` is the stronger statement: the figure this would compute is not a narrower
    version of what the metric names, it is a different quantity, and ranking players on
    it would be worse than not ranking them. Such a metric is not computed at all, and
    the string says why so the gap appears in the report instead of vanishing.

    The distinction is the one the external review of 2026-09-22 drew and it is worth
    holding onto. A floor can still order a pool; a wrong quantity cannot.
    """
    return dict(volume=v, rate=r, inputs=inputs, note=note, optional=tuple(optional),
                blocked=blocked)


# ── how each figure is actually computed ─────────────────────────────────────
# Everything below takes `s`, a mapping of FEED code to that player's SEASON TOTAL,
# plus s["matches"] and s["minutes"]. Two conventions, both of them choices:
#
#   A volume is a per-match average, not a season total. A man with five matches and a
#   man with twenty are then on the same scale, which is what a benchmark needs; season
#   totals would rank availability rather than contribution.
#
#   A percentage is recomputed from its components over the whole season rather than
#   averaged across matches. Averaging per-match percentages weights a quiet game as
#   heavily as a busy one — a player who made one tackle and missed none does not have a
#   100% season.

def _div(a, b):
    """a / b, and nothing rather than infinity when there were no opportunities."""
    import numpy as _np
    b = _np.asarray(b, dtype=float)
    return _np.where(b > 0, _np.asarray(a, dtype=float) / _np.where(b > 0, b, 1), _np.nan)


def _opt(s, code):
    """An optional input, or zero where the feed does not record it at all."""
    import numpy as _np
    v = s.get(code)
    return _np.nan_to_num(_np.asarray(v, dtype=float)) if v is not None else 0.0


def _per_match(code):
    return lambda s: _div(s[code], s["matches"])


VOLUME = {
    "RM": _per_match("RM"), "TB": _per_match("TB"), "KRM": _per_match("KRM"),
    "PCM": _per_match("PCM"), "HUM": _per_match("HUM"), "DHM": _per_match("DHM"),
    "LB": _per_match("LB"), "LBA": _per_match("LBA"), "Tries": _per_match("Tries"),
    "Offloads": _per_match("Offloads"), "Runs": _per_match("Runs"),
    "Receipts": _per_match("Receipts"), "Errors": _per_match("Errors"),
    "PTBW": _per_match("PTBW"), "KM": _per_match("KM"), "LER": _per_match("LER"),
    "Tackles Made": _per_match("TacklesMade"), "40/20s": _per_match("FortyTwenty"),
    "Forced Drop Outs": _per_match("GLDO"), "Break Cause": _per_match("BreakCause"),
    "Break involvements (LB+LBA)": lambda s: _div(s["LB"] + s["LBA"], s["matches"]),
    "Combined Offball (supports + decoys)":
        lambda s: _div(s["Support"] + s["Decoy"], s["matches"]),
    "Combined Infringements":
        lambda s: _div(s["Penalties"] + _opt(s, "SetRestart"), s["matches"]),
    # already a rate in Mike's sheet, and recomputed from its components
    "KD%": lambda s: 100 * _div(s["KickDefused"], s["KickDefusedAtt"]),
}

RATE = {
    "MpR": lambda s: _div(s["RM"], s["Runs"]),
    "TBpR": lambda s: _div(s["TB"], s["Runs"]),
    "PCMpR": lambda s: _div(s["PCM"], s["Runs"]),
    "HUMpR": lambda s: _div(s["HUM"], s["Runs"]),
    "DHMpR": lambda s: _div(s["DHM"], s["DHR"]),
    "DHRpRec": lambda s: _div(s["DHR"], s["Receipts"]),
    "Earned Metres (RM-KRM/Runs)": lambda s: _div(s["RM"] - s["KRM"], s["Runs"]),
    "LBpR": lambda s: _div(s["LB"], s["Runs"]),
    "LBpRun": lambda s: _div(s["LB"], s["Runs"]),
    "LBApRec": lambda s: _div(s["LBA"], s["Receipts"]),
    "LERpRec": lambda s: _div(s["LER"], s["Receipts"]),
    "OffloadspRun": lambda s: _div(s["Offloads"], s["Runs"]),
    # KTA — kick try assists — is not in this extract (DATA-2), so the numerator is
    # line breaks plus line-break assists and the figure is a floor, not the full CV
    "CV (LB+LBA/Rec)": lambda s: _div(s["LB"] + s["LBA"], s["Receipts"]),
    "Break conversion (Tries pLB)": lambda s: _div(s["Tries"], s["LB"]),
    "PPR": lambda s: _div(s["Passes"], s["Receipts"]),
    "Involvement Rate (Runs+Tackles/Min)":
        lambda s: _div(s["Runs"] + s["TacklesMade"], s["minutes"]),
    "OBV (Support+Decoy/Min)": lambda s: _div(s["Support"] + s["Decoy"], s["minutes"]),
    "Off Ball Volume (Support+Decoy/Min)":
        lambda s: _div(s["Support"] + s["Decoy"], s["minutes"]),
    "Error rate (Errors pReceipt)": lambda s: _div(s["Errors"], s["Receipts"]),
    "Combined Infringements pTackle":
        lambda s: _div(s["Penalties"] + _opt(s, "SetRestart"), s["TacklesMade"]),
    "PTBW%": lambda s: 100 * _div(s["PTBW"], s["PTBtot"]),
    "TE%": lambda s: 100 * _div(s["TacklesMade"], s["TacklesMade"] + s["TacklesMissed"]),
    # a long kick is every kick that was not taken in the attacking half
    "KM per Long Kick (Long kick is K total-AK)":
        lambda s: _div(s["KM"], s["Kicks"] - s["AttKicks"]),
    "LKS p Long Kick": lambda s: _div(s["LongToOpen"], s["LongKicks"]),
    "FDO p Attacking Kick": lambda s: _div(s["GLDO"], s["AttKicks"]),
}

# Lower is better for these, so their 0-100 benchmark is inverted: a player who concedes
# few penalties should score high, not low.
LOWER_IS_BETTER = {
    "Errors", "Combined Infringements", "Break Cause",
    "Error rate (Errors pReceipt)", "Combined Infringements pTackle",
}


_YARD_OUTSIDE = [
    M("RM", "MpR", ["RM", "Runs"]),
    M("TB", "TBpR", ["TB", "Runs"]),
    M("PCM", "PCMpR", ["PCM", "Runs"]),
]
_INVOLVE = [
    M("Runs", "PPR", ["Runs", "Passes", "Receipts"]),
    M("Receipts", "Involvement Rate (Runs+Tackles/Min)",
      ["Receipts", "Runs", "TacklesMade", "Minutes"]),
    M("Combined Offball (supports + decoys)", "OBV (Support+Decoy/Min)",
      ["Support", "Decoy", "Minutes"]),
]
_DEF = [
    M("Errors", "Error rate (Errors pReceipt)", ["Errors", "Receipts"]),
    M("Combined Infringements", "Combined Infringements pTackle",
      ["Penalties", "SetRestart", "TacklesMade"],
      "penalties plus set restarts conceded; Super League has no set-restart data, so "
      "there it degrades to penalties alone — correct within that competition, not "
      "comparable across them until DATA-1 arrives",
      optional=["SetRestart"]),
]

SPEC = {
    "FB": {
        "Yardage": [M("RM", "MpR", ["RM", "Runs"]),
                    M("TB", "TBpR", ["TB", "Runs"]),
                    M("KRM", "Earned Metres (RM-KRM/Runs)", ["KRM", "RM", "Runs"])],
        "Attack": [M("LB", "LBpR", ["LB", "Runs"]),
                   M("LBA", "CV (LB+LBA/Rec)", ["LB", "LBA", "KTA", "Receipts"],
                     "KTA column absent — DATA-2", optional=["KTA"]),
                   M("Tries", "Break conversion (Tries pLB)", ["Tries", "LB"])],
        "Involvement": _INVOLVE,
        "Defence & Discipline": _DEF + [M("KD%", "TE%", ["KDpct", "TEpct"])],
    },
    "WG": {
        "Yardage": _YARD_OUTSIDE,
        "Attack": [M("LB", "LBpR", ["LB", "Runs"]),
                   M("Offloads", "OffloadspRun", ["Offloads", "Runs"]),
                   M("Tries", "Break conversion (Tries pLB)", ["Tries", "LB"])],
        "Involvement": _INVOLVE,
        "Defence & Discipline": _DEF + [M("KD%", "TE%", ["KDpct", "TEpct"])],
    },
    "CT": {
        "Yardage": _YARD_OUTSIDE,
        "Attack": [M("Break involvements (LB+LBA)", "CV (LB+LBA/Rec)",
                     ["LB", "LBA", "KTA", "Receipts"],
                     "KTA column absent — DATA-2", optional=["KTA"]),
                   M("Offloads", "OffloadspRun", ["Offloads", "Runs"]),
                   M("Tries", "Break conversion (Tries pLB)", ["Tries", "LB"])],
        "Involvement": _INVOLVE,
        "Defence & Discipline": _DEF + [M("Break Cause", "TE%", ["BreakCause", "TEpct"])],
    },
    "Halves": {
        "Yardage": [M("KM", "KM per Long Kick (Long kick is K total-AK)",
                      ["KM", "Kicks", "AttKicks"]),
                    M("40/20s", "LKS p Long Kick",
                      ["FortyTwenty", "LongToOpen", "LongKicks"],
                      "40/20 is genuine but very rare per player"),
                    M("Forced Drop Outs", "FDO p Attacking Kick", ["GLDO", "AttKicks"],
                      blocked="GLDO is the dropout the player's own side TOOK "
                              "while defending, not the kick he made that forced "
                              "one. Dividing it by his attacking kicks measures "
                              "nothing about him. Needs 'Kick - Forced Dropout' "
                              "— DATA-2.")],
        "Attack": [M("LBA", "LBApRec", ["LBA", "Receipts"]),
                   M("LER", "LERpRec", ["LER", "Receipts"],
                     "recording drifts between seasons"),
                   M("Tries", "CV (LB+LBA/Rec)", ["LB", "LBA", "KTA", "Receipts"],
                     "KTA column absent — DATA-2", optional=["KTA"])],
        "Involvement": _INVOLVE,
        "Defence & Discipline": _DEF + [M("Break Cause", "TE%", ["BreakCause", "TEpct"])],
    },
    "Hooker": {
        "Yardage": [M("DHM", "DHMpR", ["DHM", "DHR"]),
                    M("40/20s", "LKS p Long Kick",
                      ["FortyTwenty", "LongToOpen", "LongKicks"],
                      "40/20 is genuine but very rare per player"),
                    M("Forced Drop Outs", "FDO p Attacking Kick", ["GLDO", "AttKicks"],
                      blocked="GLDO is the dropout the player's own side TOOK "
                              "while defending, not the kick he made that forced "
                              "one. Dividing it by his attacking kicks measures "
                              "nothing about him. Needs 'Kick - Forced Dropout' "
                              "— DATA-2.")],
        "Attack": [M("LBA", "LBApRec", ["LBA", "Receipts"]),
                   M("LER", "LERpRec", ["LER", "Receipts"],
                     "recording drifts between seasons"),
                   M("Tries", "CV (LB+LBA/Rec)", ["LB", "LBA", "KTA", "Receipts"],
                     "KTA column absent — DATA-2", optional=["KTA"])],
        "Involvement": [M("Runs", "PPR", ["Runs", "Passes", "Receipts"]),
                        M("Receipts", "DHRpRec", ["DHR", "Receipts"]),
                        M("Combined Offball (supports + decoys)",
                          "Off Ball Volume (Support+Decoy/Min)",
                          ["Support", "Decoy", "Minutes"])],
        "Defence & Discipline": _DEF + [M("Tackles Made", "TE%",
                                          ["TacklesMade", "TEpct"])],
    },
    "Middles": {
        "Yardage": _YARD_OUTSIDE,
        "Attack": [M("HUM", "HUMpR", ["HUM", "Runs"]),
                   M("Offloads", "OffloadspRun", ["Offloads", "Runs"]),
                   M("PTBW", "PTBW%", ["PTBW", "PTBtot"])],
        "Involvement": _INVOLVE,
        "Defence & Discipline": _DEF + [M("Tackles Made", "TE%",
                                          ["TacklesMade", "TEpct"])],
    },
    "Edge": {
        "Yardage": _YARD_OUTSIDE,
        "Attack": [M("LB", "LBpRun", ["LB", "Runs"]),
                   M("Offloads", "OffloadspRun", ["Offloads", "Runs"]),
                   M("PTBW", "PTBW%", ["PTBW", "PTBtot"])],
        "Involvement": _INVOLVE,
        "Defence & Discipline": _DEF + [M("Tackles Made", "TE%",
                                          ["TacklesMade", "TEpct"])],
    },
    # Mike: "This is IT, I mislabelled it." The metric set is the one he gave, which is
    # the Prop profile — reasonable, since an interchange forward is doing a prop's job
    # in shorter bursts. Where a sub-position can be recovered (IT_SUBPOSITION), that
    # player is better judged against the position he actually plays.
    "IT": {
        "Yardage": _YARD_OUTSIDE,
        "Attack": [M("HUM", "HUMpR", ["HUM", "Runs"]),
                   M("Offloads", "OffloadspRun", ["Offloads", "Runs"]),
                   M("PTBW", "PTBW%", ["PTBW", "PTBtot"])],
        "Involvement": _INVOLVE,
        "Defence & Discipline": _DEF + [M("Tackles Made", "TE%",
                                          ["TacklesMade", "TEpct"])],
    },
}

# The Lock block from Mike's sheet, kept because Leeds may want it back and because it
# records what was given up. It differed from the prop profile in one slot only: line
# break assists per receipt where the prop has hit-up metres per run — the ball-playing
# lock, which is precisely the distinction the Middles grouping stops drawing.
LOCK_BLOCK_RETIRED = {
    "Attack": [M("LBA", "LBApRec", ["LBA", "Receipts"]),
               M("Offloads", "OffloadspRun", ["Offloads", "Runs"]),
               M("PTBW", "PTBW%", ["PTBW", "PTBtot"])],
    "retired": "2026-09-19, folded into Middles at Leeds's request",
}


def all_inputs():
    """Every feed code the specification touches."""
    out = set()
    for pos in SPEC.values():
        for cat in pos.values():
            for m in cat:
                out.update(m["inputs"])
    return sorted(out)


def resolve(position, competition, season):
    """Which metrics of a position are computable for a competition-season, and why not.

    A metric is blocked when any of its inputs is unavailable there. This is the
    function the rating layer and the app should ask, rather than each deciding for
    itself what the data supports.
    """
    out = {}
    for cat, metrics in SPEC[position].items():
        rows = []
        for m in metrics:
            blocked = []
            for code in m["inputs"]:
                av = AVAILABILITY.get(code)
                if not av:
                    continue
                if av.get("usable_from") and season < av["usable_from"]:
                    blocked.append(f"{code} not recorded before {av['usable_from']}")
                if competition in (av.get("missing_in") or []):
                    blocked.append(f"{code} absent in {competition}")
            rows.append(dict(volume=m["volume"], rate=m["rate"], inputs=m["inputs"],
                             note=m["note"], blocked_by=blocked,
                             computable=not blocked))
        out[cat] = rows
    return out


def counts():
    """How many of the 24 figures per position the spec defines, as a sanity check."""
    return {p: sum(len(c) for c in cats.values()) * 2 for p, cats in SPEC.items()}
