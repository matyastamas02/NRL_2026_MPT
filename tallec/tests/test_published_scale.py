# -*- coding: utf-8 -*-
"""The published 0-100 scale, held to the sentence printed underneath it.

The external review of 2026-09-22 found the scale was not what the documentation said.
It applied a standard normal CDF to a composite whose between-player standard deviation
is about 0.19, so everything flattened toward the middle: a published 69 was the 99.5th
percentile of its pool, a 60 was the 92.6th, and 50 was the 56.7th rather than the
median. A recruiter reading 62 as "somewhat above average" was reading the top few per
cent.

These tests hold the two claims the app makes — 50 is the median of the peer group, and
the curve means what a normal curve means — and the one it must not make, that the score
is itself a percentile.
"""
import math
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import player_rating_engine as pre

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "tallec.db")


def engine(tau2=0.04, grand=0.0):
    e = pre.PlayerRatingEngine("NRL")
    e.tau2, e.sigma2, e.grand_mean = tau2, 0.19, grand
    return e


def test_the_bare_cdf_cannot_be_called_any_more():
    """It treated a 0.19-scale quantity as a standard normal and had to go."""
    with pytest.raises(RuntimeError, match="unit variance"):
        pre.PlayerRatingEngine._to_0_100(0.5)


def test_the_centre_scores_fifty():
    e = engine(grand=0.137)
    assert e.scale_0_100(0.137) == pytest.approx(50.0)
    assert e.scale_0_100(0.0, centre=0.0) == pytest.approx(50.0)


def test_one_tau_above_the_centre_is_one_normal_sd():
    e = engine(tau2=0.04)                      # tau = 0.2
    assert e.scale_0_100(0.2, centre=0.0) == pytest.approx(84.13, abs=0.05)
    assert e.scale_0_100(-0.2, centre=0.0) == pytest.approx(15.87, abs=0.05)


def test_the_scale_is_symmetric_about_its_centre():
    e = engine()
    for d in (0.05, 0.1, 0.3):
        assert (e.scale_0_100(d, centre=0.0)
                + e.scale_0_100(-d, centre=0.0)) == pytest.approx(100.0)


def test_a_degenerate_pool_scores_everyone_fifty_rather_than_dividing_by_zero():
    e = engine(tau2=0.0)
    assert e.scale_0_100(1.0) == 50.0


def test_calibration_is_idempotent():
    """It is run twice — once over everyone, once over the players published."""
    e = engine()
    df = pd.DataFrame({"player_id": list(range(40)),
                       "class_z": np.linspace(-0.4, 0.4, 40),
                       "form_z": np.linspace(-0.3, 0.3, 40),
                       "raw_composite": np.linspace(-0.5, 0.5, 40),
                       "group": ["Middles"] * 20 + ["Edge"] * 20})
    once = e.calibrate(df)
    twice = e.calibrate(once)
    assert np.allclose(once.class_score.values, twice.class_score.values)


def test_a_group_too_small_falls_back_to_the_competition_and_says_so():
    e = engine()
    df = pd.DataFrame({"player_id": list(range(22)),
                       "class_z": np.linspace(-0.3, 0.3, 22),
                       "form_z": np.linspace(-0.3, 0.3, 22),
                       "raw_composite": np.linspace(-0.3, 0.3, 22),
                       "group": ["Middles"] * 20 + ["Hooker"] * 2})
    out = e.calibrate(df)
    assert set(out.loc[out.group == "Hooker", "scale_basis"]) == {"competition"}
    assert set(out.loc[out.group == "Middles", "scale_basis"]) == {"position group"}


# ── against what is actually published ───────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(DB), reason="no database")
def test_every_peer_group_has_fifty_at_its_median():
    """The sentence under the number: 50 is the median of his position group."""
    con = sqlite3.connect(DB)
    r = pd.read_sql("SELECT competition, \"group\", class_score, scale_basis "
                    "FROM player_ratings", con)
    con.close()
    grouped = r[r.scale_basis == "position group"]
    med = grouped.groupby(["competition", "group"]).class_score.median()
    big = grouped.groupby(["competition", "group"]).size()
    for key, m in med[big >= pre.MIN_GROUP_FOR_CENTRE].items():
        assert abs(m - 50.0) < 1.0, f"{key} sits at {m:.2f}, not 50"


@pytest.mark.skipif(not os.path.exists(DB), reason="no database")
def test_the_score_is_not_sold_as_a_percentile_but_one_is_published():
    con = sqlite3.connect(DB)
    r = pd.read_sql("SELECT competition, \"group\", class_score, class_percentile "
                    "FROM player_ratings", con)
    con.close()
    assert r.class_percentile.between(0, 100).all()
    # they must not be the same number: the score keeps the shrinkage's compression,
    # the percentile does not
    assert not np.allclose(r.class_score, r.class_percentile, atol=1.0)

    # Within a peer pool the two must order players identically — that is what makes
    # publishing both safe. ACROSS pools they need not, and should not: the score is in
    # shared units of tau, so a group with a tighter spread of ability produces a
    # narrower band of scores while its percentiles still run 0 to 100. Asserting a
    # whole-table correlation instead would be asserting that the score is a percentile,
    # which is the thing this file exists to deny.
    for key, g in r.groupby(["competition", "group"]):
        if len(g) < 10:
            continue
        rho = g.class_score.corr(g.class_percentile, method="spearman")
        assert rho == pytest.approx(1.0, abs=1e-9), f"{key} disagrees: rho={rho}"


@pytest.mark.skipif(not os.path.exists(DB), reason="no database")
def test_the_published_scale_is_no_longer_crushed_into_the_middle():
    """The defect itself: before the fix the whole competition sat between 45 and 55."""
    con = sqlite3.connect(DB)
    r = pd.read_sql("SELECT class_score FROM player_ratings", con)
    con.close()
    assert r.class_score.std() > 15
    assert r.class_score.quantile(0.9) - r.class_score.quantile(0.1) > 40
