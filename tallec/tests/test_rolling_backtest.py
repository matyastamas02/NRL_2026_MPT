# -*- coding: utf-8 -*-
"""The rolling backtest claims nothing from the forecast season reaches the forecast.

That claim is the whole point of the module — the report it replaces failed exactly
here — so it is tested rather than asserted in a docstring.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rolling_backtest as rb
import translation_features as tf

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "tallec.db")
# Rebuilding the rating history for an origin takes a couple of minutes, which is
# the price of testing the thing that actually matters here. Deselect with
#   pytest -m "not slow"
pytestmark = [pytest.mark.skipif(not os.path.exists(DB), reason="no database"),
              pytest.mark.slow]


@pytest.fixture(scope="module")
def known():
    """What the project held at the end of 2023, used to forecast 2024."""
    con = sqlite3.connect(DB)
    out = rb.knowledge_at(con, 2023)
    con.close()
    return out


def test_training_pairs_stop_at_the_origin(known):
    pairs = known[0]
    assert len(pairs) > 100
    assert pairs.season_src.max() <= 2023
    assert pairs.season_tgt.max() <= 2023, "a move landing in 2024 is in the training set"


def test_the_source_ratings_stop_at_the_origin(known):
    cum = known[1]
    assert cum.season.max() <= 2023


def test_the_predicted_moves_are_scored_on_the_forecast_season(known):
    pairs, cum, ext, pos, dob = known
    con = sqlite3.connect(DB)
    pend = rb.moves_into(con, 2024, cum, ext, pos, dob)
    con.close()
    assert len(pend) > 50
    # The rating going in is from the season before, the outcome from the season itself.
    # These used to share one column called `season`, which held the source season and
    # read like the target one; asserting both separately is what the test meant.
    assert (pend.season_src == 2023).all()
    assert (pend.season_tgt == 2024).all()
    assert pend.class_source.between(0, 100).all()
    assert pend.class_target.between(0, 100).all()


def test_no_player_season_is_both_trained_on_and_predicted(known):
    """The sharpest form of the leak: the same observation on both sides."""
    pairs, cum, ext, pos, dob = known
    con = sqlite3.connect(DB)
    pend = rb.moves_into(con, 2024, cum, ext, pos, dob)
    con.close()
    trained = set(zip(pairs.player_id, pairs.season_tgt, pairs.target))
    predicted = set(zip(pend.player_id, [2024] * len(pend), pend.target))
    assert not (trained & predicted)


def test_a_fitted_origin_predicts_without_touching_the_target(known):
    pairs, cum, ext, pos, dob = known
    f = rb.fit(pairs, "B_next_season")
    assert f is not None and f["n"] == len(pairs[pairs.layer == "B_next_season"])
    con = sqlite3.connect(DB)
    pend = rb.moves_into(con, 2024, cum, ext, pos, dob)
    con.close()
    out = rb.predict(f, pend)
    # predictions must not simply echo the outcome, and must stay on the scale
    assert out.projected.between(0, 100).all()
    assert out.model.between(0, 100).all()
    assert not np.allclose(out.projected.values, out.class_target.values)


def test_the_feature_spec_used_here_is_the_shared_one(known):
    """One definition — the same builder the live model uses."""
    f = rb.fit(known[0], "B_next_season")
    spec = f["spec"]
    assert isinstance(spec, tf.FeatureSpec)
    assert all(f"grp_{g}" in spec.columns
               for g in tf.POSITION_GROUPS if g != tf.REFERENCE_GROUP)
    assert "pos_missing" in spec.columns


def test_score_reports_every_predictor_on_the_same_rows():
    d = pd.DataFrame({"class_target": [50.0, 60.0, 40.0],
                      "projected": [52.0, 58.0, 44.0],
                      "model": [51.0, 57.0, 45.0],
                      "class_source": [55.0, 65.0, 35.0]})
    s = rb.score(d)
    assert set(s.n) == {3}, "predictors compared on different row counts"
    assert len(s) == 4


# ── a model must beat the simplest thing that could work ─────────────────────

def test_a_straight_line_is_fitted_per_direction_and_pooled():
    import numpy as np
    import rolling_backtest as rb
    rng = np.random.default_rng(0)
    n = 400
    p = pd.DataFrame({
        "class_source": rng.normal(50, 20, n),
        "source": rng.choice(["NSW", "QLD"], n), "target": "NRL",
        "player_id": [f"p{i}" for i in range(n)]})
    # a direction that halves the source, and one that leaves it alone
    p["class_target"] = np.where(p.source == "NSW",
                                 50 + 0.5 * (p.class_source - 50),
                                 p.class_source) + rng.normal(0, 1, n)
    lines = rb.straight_lines(p)
    assert "__pooled__" in lines and "NSW->NRL" in lines and "QLD->NRL" in lines
    assert lines["NSW->NRL"][0] == pytest.approx(0.5, abs=0.05)
    assert lines["QLD->NRL"][0] == pytest.approx(1.0, abs=0.05)
    # and the per-direction fit must beat the pooled one on this fixture
    per = rb.apply_lines(lines, p, per_direction=True)
    pooled = rb.apply_lines(lines, p, per_direction=False)
    assert (p.class_target - per).abs().mean() < (p.class_target - pooled).abs().mean()


def test_a_thin_direction_borrows_the_pooled_line():
    import numpy as np
    import rolling_backtest as rb
    rng = np.random.default_rng(1)
    big = pd.DataFrame({"class_source": rng.normal(50, 15, 200),
                        "source": "NSW", "target": "NRL"})
    big["class_target"] = 50 + 0.4 * (big.class_source - 50)
    thin = pd.DataFrame({"class_source": [40.0, 60.0], "source": "SL", "target": "NRL"})
    thin["class_target"] = [45.0, 55.0]
    lines = rb.straight_lines(pd.concat([big, thin], ignore_index=True))
    assert "SL->NRL" not in lines, "two moves is not a direction-specific line"
    out = rb.apply_lines(lines, thin, per_direction=True)
    assert np.isfinite(out).all()


def test_the_comparison_names_a_winner_and_can_say_no():
    import numpy as np
    import rolling_backtest as rb
    n = 60
    truth = np.linspace(30, 70, n)
    # the source is a noisy, over-dispersed version of the truth, so no simple
    # predictor is already perfect — an earlier version of this fixture set the source
    # equal to the target, which made "no translation" unbeatable and the test wrong
    source = 50 + 2.0 * (truth - 50)
    base = dict(pair="NSW->NRL", class_target=truth, class_source=source,
                projected=source, line_pooled=50.0, flat50=50.0)
    good = pd.DataFrame({**base, "model": truth, "line_direction": 50.0})
    bad = pd.DataFrame({**base, "model": 50.0, "line_direction": truth})
    assert rb.beats_the_simple_thing(good)["model wins"].iloc[0] == "yes"
    assert rb.beats_the_simple_thing(bad)["model wins"].iloc[0] == "NO"


def test_a_direction_with_too_few_moves_is_left_out_rather_than_judged():
    import numpy as np
    import rolling_backtest as rb
    d = pd.DataFrame({"pair": ["A->B"] * 5, "class_target": np.arange(5.0),
                      "class_source": np.arange(5.0), "model": 2.0,
                      "line_direction": 2.0, "line_pooled": 2.0,
                      "projected": 2.0, "flat50": 50.0})
    assert rb.beats_the_simple_thing(d, min_moves=20).empty


def _direction(n, model_err, line_err, seed=0):
    """One direction where the model and the line miss by known amounts.

    The errors are drawn rather than fixed. An earlier version used a constant offset,
    which made every row's paired difference identical and collapsed the bootstrap
    interval to a single point — a fixture that cannot fail is not a test.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    truth = rng.normal(50, 20, n)
    return pd.DataFrame({
        "pair": "NSW->NRL", "player_id": [f"p{i}" for i in range(n)],
        "class_target": truth, "class_source": 50 + 2 * (truth - 50),
        "model": truth + rng.normal(0, model_err, n),
        "line_direction": truth + rng.normal(0, line_err, n),
        "line_pooled": 50.0, "projected": 50.0, "flat50": 50.0})


def test_the_gap_against_the_line_carries_an_interval():
    import rolling_backtest as rb
    t = rb.beats_the_simple_thing(_direction(200, 1.0, 4.0))
    r = t.iloc[0]
    # both errors are half-normal, so the expected gap is (4 - 1) * sqrt(2/pi)
    assert r["vs line"] == pytest.approx(3.0 * (2 / 3.141592653589793) ** 0.5, abs=0.4)
    assert r["ci low"] < r["vs line"] < r["ci high"]
    assert r["clear"] == "yes"


def test_a_tie_is_reported_as_not_clear():
    """The result that matters: on the client's directions the interval contains zero."""
    import rolling_backtest as rb
    t = rb.beats_the_simple_thing(_direction(200, 3.0, 3.0, seed=5))
    r = t.iloc[0]
    assert abs(r["vs line"]) < 1.0
    assert r["clear"] == "no", "equal errors must not read as a win"
    assert r["ci low"] < 0 < r["ci high"]


def test_the_interval_is_signed_so_a_losing_model_is_visible():
    import rolling_backtest as rb
    t = rb.beats_the_simple_thing(_direction(200, 5.0, 1.0))
    r = t.iloc[0]
    assert r["vs line"] < 0, "negative means the straight line is better"
    assert r["ci high"] < 0 and r["clear"] == "yes"
    assert r["model wins"] == "NO"


def test_the_interval_is_clustered_on_players_not_rows():
    """One man appearing many times is one piece of evidence, not many."""
    import numpy as np
    import rolling_backtest as rb
    base = _direction(240, 1.0, 4.0)
    # the same 240 observations spread over 240 players, 24 players, and 3
    wide = base.assign(player_id=[f"p{i}" for i in range(240)])
    few = base.assign(player_id=[f"p{i % 24}" for i in range(240)])
    widths = []
    for frame in (wide, few):
        r = rb.beats_the_simple_thing(frame).iloc[0]
        widths.append(r["ci high"] - r["ci low"])
    assert widths[1] > widths[0], (
        "240 rows from 24 players must give a wider interval than from 240 players")


def test_one_player_gets_no_interval_rather_than_a_perfect_one():
    """A clustered bootstrap over a single cluster resamples the same rows every time.

    It returned an interval of zero width — perfect certainty exactly where there is
    least of it. This test found that, not a review.
    """
    import numpy as np
    import rolling_backtest as rb
    frame = _direction(40, 1.0, 4.0).assign(player_id="p0")
    r = rb.beats_the_simple_thing(frame).iloc[0]
    assert np.isnan(r["ci low"]) and np.isnan(r["ci high"])
    assert r["clear"] == "no"
    assert not np.isnan(r["vs line"]), "the point estimate is still real"
