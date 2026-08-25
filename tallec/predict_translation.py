# -*- coding: utf-8 -*-
"""Competition translation prediction — v2.

Answers "this player rates X in competition A; what would he rate in competition B?"
for any ordered pair among NRL, NSW Cup, Queensland Cup and Super League.

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
    con.close()
    return pkl, ladder, meta


_PKL, LADDER, META = _load()


def available_pairs():
    """Competition pairs with a measured shift, most-sampled first."""
    return LADDER.sort_values("n", ascending=False)[["source", "target", "n", "shift"]]


def translate(score, source, target, position=None, age=None, minutes_pg=None,
              games=None):
    """Translate a 0-100 rating from `source` competition to `target`.

    Returns the predicted score, the shift in both z and points, a 95% band from
    the model's out-of-sample error, and the evidence behind it. `basis` says
    whether the number came from the conditional model or from the measured
    ladder alone, and `n_obs` how many observed player moves stand behind it.
    """
    if source == target:
        return {"source": source, "target": target, "score_source": score,
                "score_target": score, "shift_z": 0.0, "shift_points": 0.0,
                "band_points": 0.0, "basis": "same competition", "n_obs": None,
                "interpretation": "Same competition — nothing to translate."}

    row = LADDER[(LADDER.source == source) & (LADDER.target == target)]
    if row.empty:
        # fall back to the reverse pair with the sign flipped
        rev = LADDER[(LADDER.source == target) & (LADDER.target == source)]
        if rev.empty:
            raise ValueError(f"no measured moves between {source} and {target}")
        shift_z, n_obs, se = -float(rev["shift"].iloc[0]), int(rev.n.iloc[0]), float(rev.se.iloc[0])
        basis = f"measured ladder, {COMP_NAME[target]} -> {COMP_NAME[source]} reversed"
    else:
        shift_z, n_obs, se = float(row["shift"].iloc[0]), int(row.n.iloc[0]), float(row.se.iloc[0])
        basis = "measured ladder"

    # score_to_z inverts the engine's own mapping, so z is already on the composite
    # scale the ladder shift was measured on — it is added directly.
    z_src = score_to_z(score)
    z_tgt = z_src + shift_z
    score_tgt = z_to_score(z_tgt)
    shift_points = score_tgt - score

    layer = "A" if (target == "NRL" and source in ("NSW", "QLD")) else "B"
    rmse = (float(META[META.label.str.startswith(f"Layer {layer}")].rmse.iloc[0])
            if len(META) else se)
    # TWO different uncertainties, and conflating them would oversell the model:
    #   how well we know the AVERAGE shift for this pair   -> standard error (tight)
    #   how well we can predict ONE player's new rating    -> out-of-sample RMSE (wide)
    band_points = abs(z_to_score(z_tgt + 1.96 * rmse) - z_to_score(z_tgt - 1.96 * rmse)) / 2
    avg_band_points = abs(z_to_score(z_tgt + 1.96 * se) - z_to_score(z_tgt - 1.96 * se)) / 2

    if shift_points <= -5:
        interp = f"Expect a clear drop moving to {COMP_NAME[target]} — a stronger pool."
    elif shift_points < -1.5:
        interp = f"Expect a modest drop in {COMP_NAME[target]}."
    elif shift_points <= 1.5:
        interp = f"Broadly like-for-like between {COMP_NAME[source]} and {COMP_NAME[target]}."
    else:
        interp = f"Expect a modest lift in {COMP_NAME[target]} — a slightly weaker pool."

    return {"source": source, "target": target, "score_source": float(score),
            "score_target": float(score_tgt), "shift_z": shift_z,
            "shift_points": float(shift_points), "band_points": float(band_points),
            "avg_band_points": float(avg_band_points),
            "se_z": se, "rmse_z": float(rmse), "n_obs": n_obs, "layer": layer,
            "basis": basis,
            "position": position, "age": age, "interpretation": interp}


if __name__ == "__main__":
    print("measured competition pairs (within-player moves):")
    print(available_pairs().to_string(index=False))
    print()
    for src, tgt, sc in [("NRL", "SL", 70), ("NRL", "SL", 50), ("NSW", "NRL", 70),
                         ("QLD", "SL", 65), ("SL", "NRL", 60)]:
        r = translate(sc, src, tgt)
        print(f"{COMP_NAME[src]} {sc} -> {COMP_NAME[tgt]}: {r['score_target']:.1f} "
              f"({r['shift_points']:+.1f} pts) | average shift known to "
              f"+/-{r['avg_band_points']:.1f} | one player's 95% band "
              f"+/-{r['band_points']:.1f} | n={r['n_obs']}")
