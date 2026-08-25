# -*- coding: utf-8 -*-
"""Competition translation prediction — v2.

Answers "this player rates X in competition A; what would he rate in competition B?"
for any ordered pair among NRL, NSW Cup, Queensland Cup and Super League.

Two estimates are produced for every request, and both are returned:

  the LADDER estimate — the measured average shift for that competition pair, which
  applies to any player and is the number to quote about a move in general;
  the MODEL estimate — the Ridge fit conditioned on this player's position, age,
  minutes per game and matches played, which is the number to quote about him.

Until 2026-08-25 the model was loaded here but never actually used, so a 20-year-old
prop with four games and a 31-year-old fullback with 150 received the identical
answer. It is wired in now, and `basis` reports which estimate the headline came from.

Two sources of truth, both built by fit_translation_v2.py:

  translation_ladder  the measured within-player level shift for each competition
                      pair, with its standard error and sample size. Used directly
                      when there is no player detail to condition on, and always
                      shown to the user, because it is a measurement rather than a
                      model output.
  translation_model_v2.pkl
                      Ridge fits that additionally condition on position, age and
                      minutes per game. Layer A covers feeder -> NRL, layer B covers
                      Australia <-> Super League.

Scale. The app shows a 0-100 benchmark where score = 100 * Phi(class_z), so the exact
inverse is z = Phi^-1(score/100). v1 approximated this with (score - 62) / 12, which
is wrong at both tails; use score_to_z / z_to_score below.

Caveat worth repeating to a client: the model is fitted on players who actually moved
(or played both competitions in one season). Those players are not a random sample —
someone called up to the NRL was picked for a reason — so a level shift measured on
them need not apply unchanged to a player nobody has promoted.
"""
import os
import pickle
import sqlite3
from statistics import NormalDist

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
MODEL_PATH = os.path.join(BASE, "translation_model_v2.pkl")

COMP_NAME = {"NRL": "NRL", "SL": "Super League", "NSW": "NSW Cup", "QLD": "Queensland Cup"}
_ND = NormalDist()


def score_to_z(score):
    """0-100 benchmark -> rating z. Exact inverse of the engine's mapping."""
    s = min(max(float(score), 0.01), 99.99) / 100.0
    return _ND.inv_cdf(s)


def z_to_score(z):
    """Rating z -> 0-100 benchmark."""
    return 100.0 * _ND.cdf(float(z))


def _load():
    with open(MODEL_PATH, "rb") as f:
        pkl = pickle.load(f)
    con = sqlite3.connect(DB)
    ladder = pd.read_sql("SELECT * FROM translation_ladder", con)
    meta = pd.read_sql("SELECT * FROM translation_model_meta", con)
    # the range each conditioning feature was actually fitted over, so the model can
    # refuse to extrapolate. These are PER-SEASON quantities: games_src tops out near
    # a full season, not a career.
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
    return pkl, ladder, meta, rng


_PKL, LADDER, META, FEATURE_RANGE = _load()


# Which fitted layer covers which ordered pair. A was fitted on feeder->NRL only;
# B on every pair involving Super League. NRL->feeder is in neither training set, so
# it falls back to the ladder and borrows layer A's error, being the same
# relationship reversed.
LAYER_PAIRS = {
    "A": {("NSW", "NRL"), ("QLD", "NRL")},
    "B": {("NRL", "SL"), ("NSW", "SL"), ("QLD", "SL"),
          ("SL", "NRL"), ("SL", "NSW"), ("SL", "QLD")},
}
POS_GROUP = _PKL.get("position_group", {})


def _layer_for(source, target):
    if (source, target) in LAYER_PAIRS["A"]:
        return "A", True
    if (source, target) in LAYER_PAIRS["B"]:
        return "B", True
    if (target, source) in LAYER_PAIRS["A"]:
        return "A", False
    return "B", False


def _feature_row(layer, z_source, source, target, position, age, minutes_pg, games):
    """One row in the exact column order the layer was fitted on.

    Anything the caller cannot supply is filled with that column position's training
    mean, so a missing age pulls the estimate toward the average player rather than
    toward zero. The dropped dummy is the reference group, so leaving every group
    dummy at zero IS that reference.
    """
    lay = _PKL["layers"][layer]
    feats, sc = lay["features"], lay["scaler"]
    row = dict(zip(feats, sc.mean_))
    row["z_source"] = z_source
    clamped = []

    def _put(key, value):
        if value is None or not pd.notna(value):
            return
        v = float(value)
        lo, hi = FEATURE_RANGE.get(key, (None, None))
        if lo is not None and not (lo <= v <= hi):
            clamped.append(f"{key}={v:g} outside the fitted range "
                           f"[{lo:g}, {hi:g}], clamped")
            v = min(max(v, lo), hi)
        row[key] = v

    _put("age", age)
    _put("mins_pg", minutes_pg)
    _put("games_src", games)
    grp = POS_GROUP.get(position) if position else None
    if grp:
        for f in feats:
            if f.startswith("grp_"):
                row[f] = 1.0 if f == "grp_" + grp else 0.0
    if any(f.startswith("pair_") for f in feats):
        want = "pair_" + source + "->" + target
        for f in feats:
            if f.startswith("pair_"):
                row[f] = 1.0 if f == want else 0.0
    used = {"position": grp,
            "age": age is not None and pd.notna(age),
            "minutes": minutes_pg is not None and pd.notna(minutes_pg),
            "games": games is not None and pd.notna(games),
            "clamped": clamped}
    return pd.DataFrame([row])[feats], used


def available_pairs():
    """Competition pairs with a measured shift, most-sampled first."""
    return LADDER.sort_values("n", ascending=False)[["source", "target", "n", "shift"]]


def translate(score, source, target, position=None, age=None, minutes_pg=None,
              games=None):
    """Translate a 0-100 rating from `source` competition to `target`.

    Returns two numbers, deliberately:

      score_target / score_ladder — the measured average shift for this pair applied
        to his rating. This is the headline and the one to quote.
      score_model — the Ridge fit conditioned on position, age, minutes per game and
        matches played in the source season. It regresses toward the middle on
        purpose, so it reads lower than the ladder for a strongly rated player; it is
        a forecast of his next season, not a restatement of his current level.

    inputs_used says which details were actually used and flags any that fell outside
    the fitted range and were clamped.
    """
    if source == target:
        return {"source": source, "target": target, "score_source": score,
                "score_target": score, "score_ladder": score, "score_model": None,
                "shift_z": 0.0, "shift_points": 0.0, "band_points": 0.0,
                "avg_band_points": 0.0, "n_obs": None, "basis": "same competition",
                "inputs_used": {}, "interpretation":
                    "Same competition - nothing to translate."}

    row = LADDER[(LADDER.source == source) & (LADDER.target == target)]
    if row.empty:
        rev = LADDER[(LADDER.source == target) & (LADDER.target == source)]
        if rev.empty:
            raise ValueError("no measured moves between " + source + " and " + target)
        shift_z = -float(rev["shift"].iloc[0])
        n_obs, se = int(rev.n.iloc[0]), float(rev.se.iloc[0])
        ladder_basis = ("measured ladder, " + COMP_NAME[target] + " -> "
                        + COMP_NAME[source] + " reversed")
    else:
        shift_z = float(row["shift"].iloc[0])
        n_obs, se = int(row.n.iloc[0]), float(row.se.iloc[0])
        ladder_basis = "measured ladder"

    # score_to_z inverts the engine's own mapping, so z is already on the composite
    # scale the ladder shift was measured on - it is added directly.
    z_src = score_to_z(score)
    z_ladder = z_src + shift_z
    score_ladder = z_to_score(z_ladder)

    layer, fitted = _layer_for(source, target)
    rmse = (float(META[META.label.str.startswith("Layer " + layer)].rmse.iloc[0])
            if len(META) else se)

    score_model, used = None, {}
    if fitted:
        X, used = _feature_row(layer, z_src, source, target, position, age,
                               minutes_pg, games)
        lay = _PKL["layers"][layer]
        score_model = z_to_score(float(lay["model"].predict(lay["scaler"].transform(X))[0]))

    # The headline stays the LADDER, and the model is reported beside it, because the
    # two answer different questions and the model is the more easily misread of the
    # two. Its slope on the source rating is attenuated by measurement error, so it
    # deliberately regresses a 70 toward the middle: as a forecast of the player's
    # NEXT season that is the lower-error answer (0.219 vs 0.277 z out of sample), but
    # quoted next to his current 70 it looks like the model disagrees with the ladder.
    # It does not - it is pricing in the part of that 70 which was luck.
    personalised = score_model is not None and any(
        [used.get("position"), used.get("age"), used.get("minutes"), used.get("games")])
    score_tgt = score_ladder
    basis = ladder_basis
    if personalised:
        basis += "; conditional model also available (layer " + layer + ")"
    shift_points = score_tgt - score

    # TWO different uncertainties, and conflating them would oversell the model:
    #   how well we know the AVERAGE shift for this pair -> standard error (tight)
    #   how well we can predict ONE player's new rating  -> out-of-sample RMSE (wide)
    z_tgt = score_to_z(score_tgt)
    band_points = abs(z_to_score(z_tgt + 1.96 * rmse)
                      - z_to_score(z_tgt - 1.96 * rmse)) / 2
    avg_band_points = abs(z_to_score(z_ladder + 1.96 * se)
                          - z_to_score(z_ladder - 1.96 * se)) / 2

    if shift_points <= -5:
        interp = "Expect a clear drop moving to " + COMP_NAME[target] + " - a stronger pool."
    elif shift_points < -1.5:
        interp = "Expect a modest drop in " + COMP_NAME[target] + "."
    elif shift_points <= 1.5:
        interp = ("Broadly like-for-like between " + COMP_NAME[source] + " and "
                  + COMP_NAME[target] + ".")
    else:
        interp = ("Expect a modest lift in " + COMP_NAME[target]
                  + " - a slightly weaker pool.")

    return {"source": source, "target": target, "score_source": float(score),
            "score_target": float(score_tgt), "score_ladder": float(score_ladder),
            "score_model": None if score_model is None else float(score_model),
            "shift_z": shift_z, "shift_points": float(shift_points),
            "band_points": float(band_points),
            "avg_band_points": float(avg_band_points),
            "se_z": se, "rmse_z": float(rmse), "n_obs": n_obs, "layer": layer,
            "basis": basis, "inputs_used": used,
            "position": position, "age": age, "interpretation": interp}


if __name__ == "__main__":
    print("measured competition pairs (within-player moves):")
    print(available_pairs().to_string(index=False))
    print()
    for src, tgt, sc in [("NRL", "SL", 70), ("NRL", "SL", 50), ("NSW", "NRL", 70),
                         ("QLD", "SL", 65), ("SL", "NRL", 60)]:
        r = translate(sc, src, tgt)
        print(f"{COMP_NAME[src]} {sc} -> {COMP_NAME[tgt]}: ladder {r['score_ladder']:.1f}"
              f" | n={r['n_obs']} | +/-{r['avg_band_points']:.1f} on the average, "
              f"+/-{r['band_points']:.1f} on the individual")
    print("\nsame rating, different players (this used to return one number):")
    for kw in [dict(position="Prop", age=20, minutes_pg=25, games=4),
               dict(position="Full Back", age=31, minutes_pg=78, games=24)]:
        r = translate(70, "NRL", "SL", **kw)
        print(f"  NRL 70 -> SL, {kw['age']}yo {kw['position']}, {kw['games']} games: "
              f"ladder {r['score_ladder']:.1f}, model {r['score_model']:.1f}"
              f"  [{r['basis']}]"
              + (f"  !! {'; '.join(r['inputs_used']['clamped'])}"
                 if r["inputs_used"].get("clamped") else ""))
    print()
    print("fitted range of each conditioning feature:",
          {k: (round(v[0], 1), round(v[1], 1)) for k, v in FEATURE_RANGE.items()})
