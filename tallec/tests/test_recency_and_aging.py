# -*- coding: utf-8 -*-
"""Season weighting and the aging curve — the two things every projection system has.

Both are tested against a stated mechanism rather than against whatever config.json
currently says, by setting the module constants directly. That way a change of
configuration cannot quietly turn a test green.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aging
import player_rating_engine as pre


# ── season weighting ─────────────────────────────────────────────────────────

@pytest.fixture
def weights(monkeypatch):
    monkeypatch.setattr(pre, "SEASON_WEIGHTS", [1.0, 0.5, 0.25])
    monkeypatch.setattr(pre, "SEASON_WEIGHT_FLOOR", 0.1)


def _player(pairs):
    """pairs: [(season, composite), ...] — one row per match."""
    return pd.DataFrame([{"season": s, "composite": c} for s, c in pairs])


def test_older_seasons_count_less(weights):
    """A good recent season should outweigh an equally long bad old one."""
    g = _player([(2023, -1.0), (2025, +1.0)])
    n, mean = pre._weighted_class(g, 2025)
    assert mean > 0, "the recent season must dominate"
    assert n == pytest.approx(1.25), "effective matches = 1.0 + 0.25"


def test_switching_it_off_restores_the_plain_average(monkeypatch):
    monkeypatch.setattr(pre, "SEASON_WEIGHTS", None)
    g = _player([(2021, 0.0), (2025, 2.0)])
    n, mean = pre._weighted_class(g, 2025)
    assert n == 2 and mean == pytest.approx(1.0)


def test_beyond_the_list_everything_gets_the_floor(weights):
    assert pre.season_weight(0) == 1.0
    assert pre.season_weight(2) == 0.25
    assert pre.season_weight(3) == 0.1
    assert pre.season_weight(12) == 0.1


def test_effective_matches_drive_the_shrinkage_not_the_row_count(weights):
    """Ten matches from five years ago are weaker evidence than ten from this year."""
    recent = _player([(2025, 1.0)] * 10)
    old = _player([(2019, 1.0)] * 10)
    n_recent, _ = pre._weighted_class(recent, 2025)
    n_old, _ = pre._weighted_class(old, 2025)
    assert n_recent > n_old
    eng = pre.PlayerRatingEngine("NRL")
    eng.sigma2, eng.tau2 = 0.18, 0.04
    _, b_recent = eng._shrink(1.0, n_recent)
    _, b_old = eng._shrink(1.0, n_old)
    assert b_recent > b_old, "the older record must be pulled harder toward the prior"


def test_the_reference_season_is_the_latest_in_the_data(weights):
    """A walk-forward snapshot weights against its own as-of season, not against today."""
    g = _player([(2021, 0.0), (2022, 2.0)])
    _, as_of_2022 = pre._weighted_class(g, 2022)
    _, as_of_2025 = pre._weighted_class(g, 2025)
    assert as_of_2022 > as_of_2025, "2022 is recent in one frame and old in the other"


# ── aging curve ──────────────────────────────────────────────────────────────

def test_the_curve_has_the_shape_the_data_showed():
    """Up into the early twenties, flat through the mid-twenties, down from 27."""
    if aging.KNOTS is None:
        pytest.skip("aging switched off in config")
    assert aging.expected_delta(20) > 0.2
    assert abs(float(aging.expected_delta(25))) < 0.2
    assert aging.expected_delta(29) < -0.5
    assert aging.expected_delta(33) < aging.expected_delta(29)


def test_it_is_monotonically_declining_after_the_peak():
    if aging.KNOTS is None:
        pytest.skip("aging switched off in config")
    ages = np.arange(25, 38, 0.5)
    d = aging.expected_delta(ages)
    assert np.all(np.diff(d) <= 1e-9), "no bumps after the peak"


def test_it_does_not_extrapolate_past_what_was_measured():
    """A 16-year-old is treated like the youngest age measured, not projected beyond."""
    if aging.KNOTS is None:
        pytest.skip("aging switched off in config")
    lo = aging.KNOTS[0][0]
    assert aging.expected_delta(lo - 5) == pytest.approx(aging.expected_delta(lo))
    hi = aging.KNOTS[-1][0]
    assert aging.expected_delta(hi + 5) == pytest.approx(aging.expected_delta(hi))


def test_an_unknown_age_gives_an_unknown_adjustment():
    """Not zero — 'expected to hold his level' is a claim, and we do not have it."""
    assert np.isnan(float(aging.expected_delta(np.nan)))


def test_switching_it_off_makes_it_a_no_op(monkeypatch):
    monkeypatch.setattr(aging, "KNOTS", None)
    assert float(aging.expected_delta(19)) == 0.0
    assert float(aging.expected_delta(34)) == 0.0


def test_the_feature_builder_derives_the_adjustment_itself():
    """A caller cannot supply an age and a contradictory aging adjustment."""
    import translation_features as tf
    f = tf.frame(60.0, "NSW", "NRL", position_group="Prop", age=30.0)
    assert f.age_delta.iloc[0] == pytest.approx(float(aging.expected_delta(30.0)))
    assert pd.isna(tf.frame(60.0, "NSW", "NRL", position_group="Prop").age_delta.iloc[0])


def test_an_unknown_date_of_birth_gives_nan_not_a_sentinel():
    """It gave about -2.5e16 years, and nothing downstream noticed.

    Subtracting a NaT and casting through timedelta64 produces the int64 sentinel, not a
    NaN. The translation feature builder then clamped that to its lower bound of 16 and
    left `age_missing` at zero, so the model was told the player was definitely sixteen
    rather than that his age was unknown. About 5% of the arrival cohort carried it.
    """
    import pandas as pd
    import sp_schema as sp
    dob = sp.parse_dob(pd.Series(["1995-03-10", None, "08/07/1991", "not a date"]))
    age = sp.age_at(dob, pd.Series([2025] * 4))
    assert pd.isna(age.iloc[1]) and pd.isna(age.iloc[3])
    assert age.iloc[0] == pytest.approx(30.31, abs=0.02)
    assert age.iloc[2] == pytest.approx(33.98, abs=0.02)
    # and nothing may come back as a plausible-looking number
    assert not ((age < 0) | (age > 60)).any()


def test_a_missing_age_reaches_the_model_as_missing():
    """The flag is the whole point: 'unknown' must not read as 'sixteen'."""
    import pandas as pd
    import sp_schema as sp
    import translation_features as tf
    dob = sp.parse_dob(pd.Series([None]))
    age = sp.age_at(dob, pd.Series([2025]))
    spec = tf.FeatureSpec.fit(pd.DataFrame(
        {"class_source": [40.0, 50.0, 60.0], "age": [22.0, 25.0, 28.0],
         "mins_pg": [30.0, 50.0, 70.0], "games_src": [5, 12, 20]}))
    X, used = spec.transform(tf.frame(60.0, "NSW", "NRL", position_group="Halves",
                                      age=float(age.iloc[0])))
    assert X.at[0, "age_missing"] == 1.0
    assert used[0]["age"] is False
