# -*- coding: utf-8 -*-
"""How much a player's rating is expected to move next season, purely from his age.

Every established projection system carries an aging curve — it is one of the three
things ZiPS and Steamer have in common, alongside weighting recent seasons more heavily
and regressing toward the mean. This project had age only as a straight line inside the
translation model, which cannot represent a curve at all: a term that fits the decline
after 27 has to get the improvement before 22 wrong, and vice versa.

The curve here is measured from this project's own data rather than borrowed. Over 3,711
season-to-season pairs across the four competitions, taking each player's rating built
from one season alone against his rating built from the next alone:

      age      pairs   change in rating
      19        128        +1.43   clearly above zero
      20-22     880        +0.2 to +0.5
      23-26    1438        -0.07   flat, the plateau
      27-30     877        -0.76   clearly below zero
      31+       342        -0.75   clearly below zero

So: improvement into the early twenties, four flat years, then roughly three quarters of
a point a year of decline from 27. That shape is what a collision sport should produce
and it is not imported from baseball, where the plateau sits later.

**What it has not done is improve a prediction.** Added as a feature to the translation
model it moves the error by +0.01 points, which is nothing. That is a question of scale
rather than of the effect being imaginary: three quarters of a point a year against a
residual error near six points is not going to show. It is kept for three reasons — the
curve is measured and is a real fact about the data; age was previously only a straight
line in the model and a straight line cannot represent a curve at all; and a proper
next-season projection, which is the thing the literature builds on top of a
translation, needs an aging term. It earns nothing today and the config says so.

Two caveats worth keeping in view. Date of birth is known for 85% of players, and the
15% without one are not a random sample — they are more often fringe players with few
matches, so the curve is measured on the better-documented part of the population. And
the ages themselves were wrong until 2026-09-14: the column mixes two date formats and
a single inferred format silently dropped most of them, so what was previously in the
model as "age" was present for a fraction of players and varied between runs
(`sp_schema.parse_dob`).

Reads `aging.knots` from config.json; set it to null to switch the adjustment off.
"""
import numpy as np

import player_rating_engine as pre

KNOTS = (pre.CONFIG.get("aging", {}) or {}).get("knots")


def expected_delta(age):
    """Expected change in the 0-100 rating over the next season, given age now.

    Linear between the measured knots, flat outside them — an 18-year-old is treated
    like a 19-year-old rather than extrapolated into a figure the data never supported,
    and the same at the far end.
    """
    if KNOTS is None:
        return np.zeros_like(np.asarray(age, dtype=float))
    xs = np.array([k[0] for k in KNOTS], dtype=float)
    ys = np.array([k[1] for k in KNOTS], dtype=float)
    a = np.asarray(age, dtype=float)
    out = np.interp(a, xs, ys, left=ys[0], right=ys[-1])
    # an unknown age must not silently become "expected to hold his level" — that is a
    # claim, and the missing flag on the feature is what carries the uncertainty
    return np.where(np.isnan(a), np.nan, out)


def curve(lo=17, hi=38, step=1):
    """The curve as a table, for a report or a chart."""
    import pandas as pd
    ages = np.arange(lo, hi + step, step, dtype=float)
    return pd.DataFrame({"age": ages, "expected_change": expected_delta(ages)})
