# -*- coding: utf-8 -*-
"""Translate a player rating from one competition to another.

Two numbers come back and they answer different questions. The distinction is not
pedantic — it is the one the sabermetric literature draws between a *translation*
("what would his current production look like in that league") and a *projection*
("what will he actually do there next season"), and this project spent months
conflating them.

  score_target  **the headline**: what he is expected to rate in the new competition
                next season. A Ridge fit on his source rating, position group and
                competition pair, which deliberately pulls extreme ratings toward the
                middle because part of any extreme rating is luck. Age, minutes per game
                and matches played were features until 2026-09-22 and were removed after
                an ablation found they did not earn their place; the live set is in
                `config.json -> translation.model_features` rather than here, so this
                paragraph cannot drift from it again.
  score_ladder  the translation: the measured average change for this pair applied to
                his rating, kept separately for the same-season and next-season
                horizons. It says what a player of his standard has historically scored
                over there. A true statement about levels, and a poor forecast of
                one man.

**The headline used to be the ladder and the evidence says it should not be.** A
rolling-origin backtest — each season forecast using only what was known before it —
finds the ladder beaten by predicting 50 for everyone. `ROLLING_REPORT.md` carries the
current figures and the cohort they rest on; they are not repeated here, because the
version that was repeated here described a 1,140-row inferred move set that
`transition_events.py` replaced with 441 genuine entries, and a scale that has since
been recalibrated.

Two things that report says which matter at the point of use. The claim holds against
the rating this system publishes and NOT against an unshrunk measure of the same season,
so it is a statement about the product rather than about the player. And on the cohort
the client actually asks about — a feeder player entering the NRL for the first time —
the model is not distinguishable from predicting 50 for everyone.

The ladder is kept and shown, because "what is his level worth over there" is a real
question a recruiter asks. It is simply not the answer to "what will he do".

The v3 model works directly in points of the 0-100 rating, because that is what it was
fitted on — the same cumulative Class score the app displays. v2 worked in composite
z-scores and needed a round trip through the normal distribution to be read; that path
is kept below, unchanged, because `v1_holdout_record.json` seals a result produced by
it and a sealed result that cannot be reproduced is not much of a seal.

**Position is two separate arguments.** `raw_position` takes a Stats Perform label
("Half Back", "Second Row"); `position_group` takes an engine group ("Halves", "Back
Row"). Passing a group where a raw label was expected used to resolve to nothing and
silently drop the position — the single argument that allowed it is gone, and giving
one raises rather than guessing.
"""
import os
import pickle
import sqlite3
from statistics import NormalDist

import numpy as np
import pandas as pd

import translation_features as tf

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
MODEL_V3 = os.path.join(BASE, "translation_model_v3.pkl")
MODEL_V2 = os.path.join(BASE, "translation_model_v2.pkl")

COMP_NAME = {"NRL": "NRL", "SL": "Super League", "NSW": "NSW Cup",
             "QLD": "Queensland Cup"}
_ND = NormalDist()
# which fitted layer answers which question
LAYER_SAME_SEASON = "A_same_season"
LAYER_NEXT_SEASON = "B_next_season"
# below this many moves a horizon's own estimate is too thin to prefer over the pooled
# one; the caller is told which was used either way
MIN_LADDER_OBS = 10


def score_to_z(score):
    """0-100 benchmark -> rating z. Exact inverse of the engine's mapping."""
    s = min(max(float(score), 0.01), 99.99) / 100.0
    return _ND.inv_cdf(s)


def z_to_score(z):
    """Rating z -> 0-100 benchmark."""
    return 100.0 * _ND.cdf(float(z))


def _load():
    if os.path.exists(MODEL_V3):
        with open(MODEL_V3, "rb") as f:
            pkl = pickle.load(f)
        return pkl, 3, pd.DataFrame(pkl["ladder"])
    with open(MODEL_V2, "rb") as f:
        pkl = pickle.load(f)
    con = sqlite3.connect(DB)
    ladder = pd.read_sql("SELECT * FROM translation_ladder", con)
    meta = pd.read_sql("SELECT * FROM translation_model_meta", con)
    try:
        pairs = pd.read_sql("SELECT age, mins_pg, g_feed FROM translation_pairs", con)
        rng = {}
        for name, col in [("age", "age"), ("mins_pg", "mins_pg"),
                          ("games_src", "g_feed")]:
            v = pd.to_numeric(pairs[col], errors="coerce").dropna()
            rng[name] = (float(v.min()), float(v.max()))
    except Exception:
        rng = {}
    con.close()
    pkl["_meta"], pkl["_range"] = meta, rng
    return pkl, 2, ladder


_PKL, VERSION, LADDER = _load()

# v2 only — see _translate_v2
LAYER_PAIRS = {
    "A": {("NSW", "NRL"), ("QLD", "NRL")},
    "B": {("NRL", "SL"), ("NSW", "SL"), ("QLD", "SL"),
          ("SL", "NRL"), ("SL", "NSW"), ("SL", "QLD")},
}
POS_GROUP = _PKL.get("position_group", {})


# Moves into these competitions are forecast by a straight line, target ~ source, fitted
# per direction on the same pairs the shipped model was fitted on (translation_pairs_v3).
# In the 2023-2025 rolling backtest the conditional model showed no clear advantage over
# that line for the 90 moves into Super League (MAE 17.98 against 17.72, difference -1.56
# to +1.06), which is not equivalence. The line is a provisional default because it is
# simpler to explain, not because it was shown to be as accurate. Decided 2026-10-08.
LINE_TARGETS = ("SL",)
MIN_LINE_PAIRS = 25          # the backtest's threshold for a direction's own line
# Only the next-season horizon was backtested, so only it gets a line. The fallback is
# the backtest's own: a direction with fewer than MIN_LINE_PAIRS pairs uses the line
# pooled over every next-season pair, which is what was scored in 38 of the 90
# historical forecasts into Super League. Any other horizon keeps the model.
LINE_LAYERS = ("B_next_season",)
# the deployed app holds only the copy build_app_db.py makes; the pairs are in both
_DB_READ = DB if os.path.exists(DB) else os.path.join(BASE, "tallec_app.db")
_LINES = {}


def _fit_line(p, basis):
    slope, icept = np.polyfit(p.class_source, p.class_target, 1)
    resid = p.class_target - (slope * p.class_source + icept)
    return dict(slope=float(slope), intercept=float(icept), n=len(p), basis=basis,
                resid_sd=float(np.sqrt((resid ** 2).sum() / (len(p) - 2))))


def _line(source, target, layer):
    """target ~ source for one direction, its pooled fallback, or None.

    `basis` says which: "direction" when the direction has MIN_LINE_PAIRS pairs of its
    own, "pooled" when the line is fitted on every pair of the layer instead.
    """
    key = (source, target, layer)
    if key not in _LINES:
        if layer not in LINE_LAYERS:
            _LINES[key] = None
            return None
        con = sqlite3.connect(f"file:{_DB_READ}?mode=ro", uri=True)
        p = pd.read_sql("SELECT source, target, class_source, class_target "
                        "FROM translation_pairs_v3 WHERE layer = ?", con, params=(layer,))
        con.close()
        own = p[(p.source == source) & (p.target == target)]
        # a direction gets its own line only with enough pairs and a source rating that
        # varies; a constant source cannot identify a slope (the backtest's rule too)
        _LINES[key] = (_fit_line(own, "direction")
                       if len(own) >= MIN_LINE_PAIRS and own.class_source.std() > 0
                       else _fit_line(p, "pooled") if len(p) >= MIN_LINE_PAIRS else None)
    return _LINES[key]


# ── comparable past entrants ──────────────────────────────────────────────────
# The real outcomes of earlier players who entered the same competition from the same
# one. The rows are the explicit entry cohort (`entry_cohort`, built by
# build_entry_cohort.py the way the frozen 2026 test builds it): first and returning
# entries, rated in the new competition, landing up to the freeze season. Until the
# seventh review this read the translation pairs, which are not entrants.
#
# The rule is fixed in code, before anyone looks at a player: same direction, source
# rating within COMP_WINDOW points; one case per player, the one with the closest source
# rating, then the earliest landing season, then the lowest id; the same position group
# when that leaves COMP_MIN players, otherwise any position, and the card says which;
# the window doubled once; below that, "not enough data". The queried player is left out
# entirely. Only rated entrants -- three or more matches in the new competition -- exist
# here, so a move that failed before three matches is invisible.
COMP_WINDOW = 7.5
COMP_MIN = 8            # distinct players
COMP_WIDE_MIN = 20      # below this the 10-90% range is not shown


def _one_per_player(rows, score):
    r = rows.assign(gap=(rows.class_source - float(score)).abs(),
                    _id=rows.player_id.astype(str))
    r = r.sort_values(["gap", "season_tgt", "_id"]).drop_duplicates("_id", keep="first")
    return r.drop(columns="_id")


def comparables(score, source, target, position_group=None, exclude_player=None):
    """Earlier entrants like this one, their outcomes and the rule that chose them."""
    import sp_schema as sp
    con = sqlite3.connect(f"file:{_DB_READ}?mode=ro", uri=True)
    try:
        p = pd.read_sql("SELECT * FROM entry_cohort WHERE source = ? AND target = ?",
                        con, params=(source, target))
    except Exception:
        p = pd.DataFrame(columns=["player_id", "class_source", "class_target",
                                  "raw_position", "season_tgt"])
    con.close()
    p["group"] = p.raw_position.map(sp.POSITION_GROUP)
    if exclude_player is not None:
        p = p[p.player_id.astype(str) != str(exclude_player)]
    out = dict(n=0, n_direction=int(p.player_id.nunique()), n_direction_rows=len(p),
               basis="not enough data", window=None, wide=False, rows=p.iloc[0:0])
    for window in (COMP_WINDOW, 2 * COMP_WINDOW):
        near = _one_per_player(p[(p.class_source - float(score)).abs() <= window], score)
        same = near[near.group == position_group] if position_group else near.iloc[0:0]
        if len(same) >= COMP_MIN:
            rows, basis = same, f"same position group ({position_group})"
        elif len(near) >= COMP_MIN:
            rows, basis = near, "any position"
        else:
            continue
        q = rows.class_target.quantile([.1, .25, .5, .75, .9])
        return dict(out, n=len(rows), basis=basis, window=window,
                    wide=len(rows) >= COMP_WIDE_MIN,
                    q10=float(q[.1]), q25=float(q[.25]), median=float(q[.5]),
                    q75=float(q[.75]), q90=float(q[.9]), rows=rows.sort_values("gap"))
    return out


def available_pairs():
    """Competition pairs with a measured shift, most-sampled first."""
    col = "shift_pts" if VERSION == 3 else "shift"
    return LADDER.sort_values("n", ascending=False)[["source", "target", "n", col]]


def _ladder_row(source, target, layer=None):
    """The measured shift for a direction, on the horizon being asked about.

    The ladder used to hold one figure per direction, averaged over both horizons, and
    it was quoted whichever horizon the caller wanted. "What is he worth there this
    year" and "what will he do there next year" are different questions, so since
    2026-09-22 the table carries a row per horizon and this picks the matching one.

    A thin direction can have too few moves on one horizon to say anything; those fall
    back to the `overall` row, and the basis string says so rather than hiding it.
    """
    def pick(frame, lay):
        r = frame[frame.layer == lay] if "layer" in frame.columns else frame
        return r if len(r) else None

    want = [layer, "overall"] if layer else ["overall"]
    fwd = LADDER[(LADDER.source == source) & (LADDER.target == target)]
    rev = LADDER[(LADDER.source == target) & (LADDER.target == source)]
    if not len(fwd) and not len(rev):
        raise ValueError(f"no measured moves between {source} and {target}")

    for lay in want:
        hit = pick(fwd, lay)
        if hit is not None and int(hit.iloc[0]["n"]) >= MIN_LADDER_OBS:
            note = "measured ladder" if lay == layer else \
                "measured ladder, both horizons pooled"
            return hit.iloc[0], note, 1.0
    for lay in want:
        hit = pick(rev, lay)
        if hit is not None and int(hit.iloc[0]["n"]) >= MIN_LADDER_OBS:
            note = (f"measured ladder, {COMP_NAME[target]} -> {COMP_NAME[source]} "
                    f"reversed")
            if lay != layer:
                note += ", both horizons pooled"
            return hit.iloc[0], note, -1.0
    # nothing clears the floor: take whatever exists rather than refusing, and say so
    any_fwd = pick(fwd, "overall")
    if any_fwd is not None:
        return any_fwd.iloc[0], "measured ladder, very few moves", 1.0
    return (pick(rev, "overall").iloc[0],
            f"measured ladder, {COMP_NAME[target]} -> {COMP_NAME[source]} reversed, "
            f"very few moves", -1.0)


def _forecast_note(source_score, ladder, forecast):
    """Why the forecast and the translation differ, described rather than explained.

    It used to say the gap is luck being handed back and "only the skill travels". The
    seventh review was right that this was asserted, not shown. What is observed is that
    ratings far from average have tended to come back toward the middle after a move.
    """
    gap = ladder - forecast
    if abs(gap) < 2.0:
        return ("The forecast and the translation land close together here, as they tend "
                "to for ratings near the middle.")
    n = round(abs(gap))
    return (f"The forecast sits {n} point{'' if n == 1 else 's'} "
            f"{'below' if gap > 0 else 'above'} the translation. Ratings far from average "
            f"have tended to come back toward the middle after a move; the forecast "
            f"builds that in and the translation does not.")


def _interpretation(shift_points, source, target):
    """What players making this move have done, not a claim about the leagues."""
    t = COMP_NAME[target]
    if shift_points <= -5:
        return f"Players making this move have rated clearly lower in {t} the next season."
    if shift_points < -1.5:
        return f"Players making this move have rated somewhat lower in {t} the next season."
    if shift_points <= 1.5:
        return f"Players making this move have rated about the same in {t} the next season."
    return f"Players making this move have rated higher in {t} the next season."


def translate(score, source, target, raw_position=None, position_group=None,
              age=None, minutes_pg=None, games=None, horizon="next_season",
              position=None):
    """Translate a 0-100 rating from `source` to `target`.

    `horizon` selects which fitted layer answers: "next_season" forecasts the season
    after the rating was earned, which is the recruitment question; "same_season" asks
    what he would have scored in the other competition at the same time.
    """
    if position is not None:
        raise TypeError(
            "translate() no longer takes `position=`, because it could not tell a raw "
            "Stats Perform label from a rating-engine group and silently dropped the "
            "latter. Pass raw_position='Half Back' or position_group='Halves'.")
    if source == target:
        return {"source": source, "target": target, "score_source": float(score),
                "score_target": float(score), "score_forecast": float(score),
                "score_ladder": float(score), "score_model": None,
                "shift_points": 0.0, "ladder_shift_points": 0.0,
                "regression_points": 0.0, "band_points": 0.0,
                "avg_band_points": 0.0, "n_obs": None, "basis": "same competition",
                "headline": "unchanged", "inputs_used": {}, "model_version": VERSION,
                "interpretation": "Same competition - nothing to translate.",
                "forecast_note": ""}
    if VERSION == 2:
        return _translate_v2(score, source, target, raw_position, position_group,
                             age, minutes_pg, games)

    layer = LAYER_NEXT_SEASON if horizon == "next_season" else LAYER_SAME_SEASON
    row, basis, sign = _ladder_row(source, target, layer)
    shift_pts = sign * float(row["shift_pts"])
    se = float(row["se_pts"])
    n_obs = int(row["n"])
    score_ladder = float(np.clip(score + shift_pts, 0.0, 100.0))

    lay = _PKL["layers"].get(layer) or next(iter(_PKL["layers"].values()))
    spec = tf.FeatureSpec.from_dict(lay["spec"])
    X, used = spec.transform(tf.frame(score, source, target,
                                      raw_position=raw_position,
                                      position_group=position_group,
                                      age=age, minutes_pg=minutes_pg, games=games))
    # the model was fitted on the families config selects, so predict on the same ones
    X = X[lay["features"]]
    _mult = np.asarray(lay.get("mult", np.ones(len(lay["features"]))), dtype=float)
    score_model = float(np.clip(
        lay["model"].predict(lay["scaler"].transform(X) * _mult)[0], 0.0, 100.0))
    rmse = float(lay["rmse"])
    used = used[0]

    line = _line(source, target, layer) if target in LINE_TARGETS else None
    if line is not None:
        forecast = float(np.clip(line["slope"] * float(score) + line["intercept"],
                                 0.0, 100.0))
        band_sd, method = line["resid_sd"], "straight line"
        note = (f"; forecast from a straight line fitted to {line['n']} earlier moves "
                + (f"from {COMP_NAME[source]} into {COMP_NAME[target]}"
                   if line["basis"] == "direction" else "between all competitions"))
    else:
        forecast, band_sd, method = score_model, rmse, "conditional model"
        note = "; forecast from the conditional model"

    # The forecast is the headline. The ladder stays beside it and the gap between them
    # is regression toward the mean — not a disagreement, which is how it reads unless
    # somebody says so, hence `regression_points` and the sentence that explains it.
    # `score_model` is always the conditional model's number, whichever one is shown.
    return {"source": source, "target": target, "score_source": float(score),
            "score_target": forecast,
            "score_forecast": forecast, "score_ladder": score_ladder,
            "score_model": score_model,
            "shift_points": float(forecast - score),
            "ladder_shift_points": float(shift_pts),
            "regression_points": float(score_ladder - forecast),
            # two uncertainties, kept apart on purpose: how well the AVERAGE shift for
            # this pair is known, and how well ONE player's outcome can be predicted
            "avg_band_points": float(1.96 * se),
            "band_points": float(1.96 * band_sd),
            "se_points": se, "rmse_points": band_sd, "n_obs": n_obs, "layer": layer,
            "basis": basis + note, "forecast_method": method, "line": line,
            "headline": "forecast", "inputs_used": used, "model_version": 3,
            "interpretation": _interpretation(shift_pts, source, target),
            "forecast_note": _forecast_note(score, score_ladder, forecast)}


# ─────────────────────────────────────────────────────────────────────────────
# v2, frozen. Reached only when translation_model_v3.pkl is absent. Kept so the
# sealed v1 holdout result can be reproduced; not maintained, not extended.
# ─────────────────────────────────────────────────────────────────────────────

def _layer_for_v2(source, target):
    if (source, target) in LAYER_PAIRS["A"]:
        return "A", True
    if (source, target) in LAYER_PAIRS["B"]:
        return "B", True
    if (target, source) in LAYER_PAIRS["A"]:
        return "A", False
    return "B", False


def _feature_row_v2(layer, z_source, source, target, group, age, minutes_pg, games):
    lay = _PKL["layers"][layer]
    feats, sc = lay["features"], lay["scaler"]
    rng = _PKL.get("_range", {})
    row = dict(zip(feats, sc.mean_))
    row["z_source"] = z_source
    clamped = []

    def _put(key, value):
        if value is None or not pd.notna(value):
            return
        v = float(value)
        lo, hi = rng.get(key, (None, None))
        if lo is not None and not (lo <= v <= hi):
            clamped.append(f"{key}={v:g} outside [{lo:g}, {hi:g}], clamped")
            v = min(max(v, lo), hi)
        row[key] = v

    _put("age", age)
    _put("mins_pg", minutes_pg)
    _put("games_src", games)
    if group:
        for f in feats:
            if f.startswith("grp_"):
                row[f] = 1.0 if f == "grp_" + group else 0.0
    if any(f.startswith("pair_") for f in feats):
        want = "pair_" + source + "->" + target
        for f in feats:
            if f.startswith("pair_"):
                row[f] = 1.0 if f == want else 0.0
    used = {"position": group, "age": age is not None and pd.notna(age),
            "minutes": minutes_pg is not None and pd.notna(minutes_pg),
            "games": games is not None and pd.notna(games), "clamped": clamped}
    return pd.DataFrame([row])[feats], used


def _translate_v2(score, source, target, raw_position, position_group, age,
                  minutes_pg, games):
    group, _ = tf.resolve_position(raw_position, position_group)
    row, basis, sign = _ladder_row(source, target)
    shift_z = sign * float(row["shift"])
    se, n_obs = float(row["se"]), int(row["n"])
    z_src = score_to_z(score)
    z_ladder = z_src + shift_z
    score_ladder = z_to_score(z_ladder)
    layer, fitted = _layer_for_v2(source, target)
    meta = _PKL.get("_meta", pd.DataFrame())
    rmse = (float(meta[meta.label.str.startswith("Layer " + layer)].rmse.iloc[0])
            if len(meta) else se)
    score_model, used = None, {}
    if fitted:
        X, used = _feature_row_v2(layer, z_src, source, target, group, age,
                                  minutes_pg, games)
        lay = _PKL["layers"][layer]
        score_model = z_to_score(
            float(lay["model"].predict(lay["scaler"].transform(X))[0]))
    z_tgt = score_to_z(score_ladder)
    return {"source": source, "target": target, "score_source": float(score),
            "score_target": float(score_ladder), "score_ladder": float(score_ladder),
            "score_model": None if score_model is None else float(score_model),
            "shift_points": float(score_ladder - score),
            "band_points": abs(z_to_score(z_tgt + 1.96 * rmse)
                               - z_to_score(z_tgt - 1.96 * rmse)) / 2,
            "avg_band_points": abs(z_to_score(z_ladder + 1.96 * se)
                                   - z_to_score(z_ladder - 1.96 * se)) / 2,
            "score_forecast": None if score_model is None else float(score_model),
            "ladder_shift_points": float(score_ladder - score),
            "regression_points": (0.0 if score_model is None
                                  else float(score_ladder - score_model)),
            "se_z": se, "rmse_z": rmse, "n_obs": n_obs, "layer": layer,
            # v2 is frozen and its headline stays the ladder: the evidence that the
            # forecast is the better headline was gathered on v3, and reopening a sealed
            # artefact to apply it would defeat the seal
            "basis": basis, "headline": "translation",
            "inputs_used": used, "model_version": 2, "forecast_note": "",
            "interpretation": _interpretation(score_ladder - score, source, target)}
