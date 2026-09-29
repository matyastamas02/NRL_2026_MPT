# -*- coding: utf-8 -*-
"""The feature builder is the thing that was silently wrong, so it is tested hardest.

Every position group is exercised by name. Three used to vanish silently — Halves,
Back Row and Bench — while the rest survived only because their group name happened to
equal their raw Stats Perform label. Back Row no longer exists as a group (the forwards
were regrouped into Middles and Edge on 19 September), but the parametrised test walks
whatever POSITION_GROUPS holds, so a renaming cannot quietly drop one from coverage.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sp_schema as sp
import translation_features as tf


def spec(pairs=()):
    train = pd.DataFrame({"class_source": [40.0, 50.0, 60.0], "age": [22.0, 25.0, 28.0],
                          "mins_pg": [30.0, 50.0, 70.0], "games_src": [5, 12, 20]})
    return tf.FeatureSpec.fit(train, pairs=pairs)


# ── position resolution ───────────────────────────────────────────────────────

@pytest.mark.parametrize("group", tf.POSITION_GROUPS)
def test_every_group_resolves_and_encodes(group):
    """No group may silently disappear — this is the regression that started it."""
    s = spec()
    X, used = s.transform(tf.frame(60.0, "NSW", "NRL", position_group=group))
    assert used[0]["position"] == group
    assert X.at[0, "pos_missing"] == 0.0
    if group == tf.REFERENCE_GROUP:
        assert all(X.at[0, c] == 0.0 for c in X.columns if c.startswith("grp_"))
    else:
        assert X.at[0, f"grp_{group}"] == 1.0
        assert sum(X.at[0, c] for c in X.columns if c.startswith("grp_")) == 1.0


@pytest.mark.parametrize("raw,group", sorted(sp.POSITION_GROUP.items()))
def test_raw_positions_map_to_their_group(raw, group):
    s = spec()
    _, used = s.transform(tf.frame(60.0, "NSW", "NRL", raw_position=raw))
    assert used[0]["position"] == group
    assert used[0]["position_from"] == "raw"


def test_group_passed_as_raw_is_an_error_not_a_silent_drop():
    """The original bug: 'Halves' handed to a raw-position mapper resolved to nothing."""
    with pytest.raises(ValueError, match="position_group"):
        tf.resolve_position(raw_position="Halves")


def test_unknown_position_is_not_the_reference_group():
    s = spec()
    X, used = s.transform(tf.frame(60.0, "NSW", "NRL"))
    assert used[0]["position"] is None
    assert X.at[0, "pos_missing"] == 1.0
    ref, _ = s.transform(tf.frame(60.0, "NSW", "NRL",
                                  position_group=tf.REFERENCE_GROUP))
    assert not X.iloc[0].equals(ref.iloc[0])


def test_contradictory_position_arguments_raise():
    with pytest.raises(ValueError, match="contradicts"):
        tf.resolve_position(raw_position="Prop", position_group="Halves")


def test_agreeing_position_arguments_are_accepted():
    grp, how = tf.resolve_position(raw_position="Half Back", position_group="Halves")
    assert grp == "Halves" and how == "both agreed"


# ── missing values ────────────────────────────────────────────────────────────

def test_missing_numeric_sets_a_flag_and_is_reported():
    s = spec()
    X, used = s.transform(tf.frame(60.0, "NSW", "NRL", position_group="Middles"))
    for c in ("age", "mins_pg", "games_src"):
        assert X.at[0, f"{c}_missing"] == 1.0
        assert used[0][c] is False
        assert X.at[0, c] == s.medians[c]


def test_supplied_numeric_clears_the_flag():
    s = spec()
    X, used = s.transform(tf.frame(60.0, "NSW", "NRL", position_group="Middles",
                                   age=24.0, minutes_pg=55.0, games=14))
    for c in ("age", "mins_pg", "games_src"):
        assert X.at[0, f"{c}_missing"] == 0.0
        assert used[0][c] is True
    assert X.at[0, "age"] == 24.0


def test_out_of_range_is_clamped_and_declared():
    s = spec()
    X, used = s.transform(tf.frame(60.0, "NSW", "NRL", position_group="Middles", age=99.0))
    assert X.at[0, "age"] == tf.FEATURE_RANGE["age"][1]
    assert any("age=99" in c for c in used[0]["clamped"])


# ── training and prediction agree ─────────────────────────────────────────────

def test_same_columns_whether_one_row_or_many():
    s = spec()
    import aging
    many = pd.DataFrame([
        {"class_source": 55.0, "source": "NSW", "target": "NRL",
         "position_group": "Halves", "age": 24.0,
         "age_delta": float(aging.expected_delta(24.0)),
         "mins_pg": 60.0, "games_src": 10},
        {"class_source": 45.0, "source": "QLD", "target": "NRL",
         "position_group": "Bench", "age": None, "age_delta": None,
         "mins_pg": None, "games_src": 6}])
    Xm, _ = s.transform(many)
    Xo, _ = s.transform(tf.frame(55.0, "NSW", "NRL", position_group="Halves",
                                 age=24.0, minutes_pg=60.0, games=10))
    assert list(Xm.columns) == list(Xo.columns) == s.columns
    assert Xm.iloc[0].tolist() == Xo.iloc[0].tolist()


def test_pair_dummies_only_when_the_layer_has_them():
    plain = spec()
    assert not any(c.startswith("pair_") for c in plain.columns)
    withpairs = spec(pairs=["NSW->SL", "SL->NRL"])
    X, used = withpairs.transform(tf.frame(60.0, "SL", "NRL", position_group="Middles"))
    assert X.at[0, "pair_SL->NRL"] == 1.0
    assert X.at[0, "pair_NSW->SL"] == 0.0
    assert used[0]["pair"] == "SL->NRL"


def test_spec_round_trips_through_a_dict():
    s = spec(pairs=["NSW->SL"])
    back = tf.FeatureSpec.from_dict(s.to_dict())
    assert back.columns == s.columns and back.medians == s.medians
    a, _ = s.transform(tf.frame(60.0, "NSW", "SL", position_group="Winger"))
    b, _ = back.transform(tf.frame(60.0, "NSW", "SL", position_group="Winger"))
    assert a.iloc[0].tolist() == b.iloc[0].tolist()


# ── partial pooling must actually pool ───────────────────────────────────────

def test_the_pooling_multiplier_marks_only_interactions():
    s = spec(pairs=["NSW->NRL", "SL->NRL"])
    import numpy as np
    m = tf.pooling_multipliers(s.columns)
    for col, mult in zip(s.columns, m):
        expect = tf.INTERACTION_SCALE if col.startswith("px_") else 1.0
        assert mult == expect, col
    assert (m < 1.0).any(), "there should be interaction columns to pool"


def test_changing_the_pooling_strength_changes_the_fit():
    """The regression this exists for.

    Until 2026-09-23 the pooling was applied inside `transform`, before the design was
    standardised — and `StandardScaler` divides each column by its own standard
    deviation, which undid it exactly. Fitted coefficients were bit-identical at 1.0,
    0.3 and 0.01, so a feature documented as carrying eleven times the shrinkage of the
    others carried the same as the others. The ablation that reported the interaction as
    the largest single improvement was measuring an unpooled one.

    Applied after standardisation it bites, and this test fails if it ever stops.
    """
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n = 300
    train = pd.DataFrame({
        "class_source": rng.normal(50, 10, n),
        "age": rng.normal(25, 3, n), "mins_pg": rng.normal(50, 10, n),
        "games_src": rng.integers(3, 25, n),
        "source": rng.choice(["NSW", "SL"], n), "target": "NRL",
        "position_group": rng.choice(tf.POSITION_GROUPS, n)})
    train["pair"] = train.source + "->" + train.target
    s = tf.FeatureSpec.fit(train, pairs=sorted(train.pair.unique()))
    X, _ = s.transform(train)
    y = rng.normal(50, 10, n)
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)

    def coefs(scale):
        old = tf.INTERACTION_SCALE
        try:
            tf.INTERACTION_SCALE = scale
            mult = tf.pooling_multipliers(list(X.columns))
            return Ridge(alpha=1.0).fit(Z * mult, y).coef_ * mult
        finally:
            tf.INTERACTION_SCALE = old

    loose, tight = coefs(1.0), coefs(0.02)
    ix = np.array([c.startswith("px_") for c in X.columns])
    assert not np.allclose(loose, tight), "the pooling strength does nothing"
    # and it has to pool the INTERACTIONS specifically, toward zero
    assert np.abs(tight[ix]).sum() < np.abs(loose[ix]).sum()
