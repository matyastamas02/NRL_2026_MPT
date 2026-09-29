# -*- coding: utf-8 -*-
"""The arrival model, and the diagnostic that first condemned it and then rescued it.

The model answers "does he get there at all", which every other evaluation in the project
takes for granted. Its pooled AUC is misleading on its own, because directions differ
enormously in how often anyone moves along them, so the within-direction figure is the one
that means something.

That diagnostic has now done both jobs. In September it showed the first version could not
beat a coin inside a direction, and the report concluded the data held no player-level
signal. The fourth external review refuted that in one line — source minutes alone rank
NSW Cup arrivals at 0.71 — and the fault was the specification: one slope per feature,
shared across pathways that do not work the same way. With direction-specific slopes the
within-direction AUC went from 0.53 to 0.68.

So these tests hold two things. That the metrics mean what they say, and that a model is
measured against its own inputs — which is the check whose absence let a model score 0.44
on a column that scores 0.71 by itself.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arrival_model as am

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "tallec.db")


# ── the metrics ──────────────────────────────────────────────────────────────

def test_auc_is_half_for_a_coin_and_one_for_a_perfect_ranking():
    y = [0, 0, 0, 1, 1, 1]
    assert am.auc(y, [1, 2, 3, 4, 5, 6]) == pytest.approx(1.0)
    assert am.auc(y, [6, 5, 4, 3, 2, 1]) == pytest.approx(0.0)
    assert am.auc(y, [1, 1, 1, 1, 1, 1]) == pytest.approx(0.5)


def test_auc_is_undefined_rather_than_wrong_when_nobody_arrives():
    assert np.isnan(am.auc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert np.isnan(am.auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_brier_rewards_being_right_and_confident():
    assert am.brier([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
    assert am.brier([1, 0], [0.5, 0.5]) == pytest.approx(0.25)
    assert am.brier([1, 0], [0.0, 1.0]) == pytest.approx(1.0)


def test_lift_is_one_when_the_ranking_is_random():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"rated": rng.random(4000) < 0.05,
                      "p_arrive": rng.random(4000)})
    t = am.lift(d, "rated", top=(0.25,))
    assert t.lift.iloc[0] == pytest.approx(1.0, abs=0.35)


def test_lift_is_high_when_the_ranking_is_perfect():
    d = pd.DataFrame({"rated": [True] * 50 + [False] * 950})
    d["p_arrive"] = np.r_[np.linspace(0.9, 1.0, 50), np.linspace(0.0, 0.1, 950)]
    t = am.lift(d, "rated", top=(0.05,))
    assert t.hit_rate.iloc[0] == pytest.approx(1.0)
    assert t.lift.iloc[0] > 15


# ── the diagnostic that mattered ─────────────────────────────────────────────

def test_a_pooled_auc_can_be_high_while_every_direction_is_a_coin():
    """The exact trap this diagnostic exists to catch, built deliberately.

    Two directions. Inside each one the score carries no information at all. But one
    direction has ten times the arrival rate of the other and its scores are shifted up,
    so pooling them produces an AUC well above a coin — skill at telling the busy
    direction from the quiet one, which no recruiter ever needs.
    """
    rng = np.random.default_rng(1)
    busy = pd.DataFrame({"pair": "A->B", "rated": rng.random(1000) < 0.30,
                         "p_arrive": rng.uniform(0.25, 0.35, 1000)})
    quiet = pd.DataFrame({"pair": "C->D", "rated": rng.random(1000) < 0.03,
                          "p_arrive": rng.uniform(0.0, 0.10, 1000)})
    d = pd.concat([busy, quiet], ignore_index=True)

    pooled = am.auc(d.rated, d.p_arrive)
    table, weighted = am.within_direction(d, "rated")
    assert pooled > 0.7, "the fixture should look good pooled"
    assert weighted == pytest.approx(0.5, abs=0.05), "and be a coin inside a direction"
    assert set(table.direction) == {"A->B", "C->D"}


def test_within_direction_skips_directions_with_too_few_arrivals():
    d = pd.DataFrame({"pair": ["A->B"] * 100 + ["C->D"] * 100,
                      "rated": [True] * 50 + [False] * 50 + [True] * 2 + [False] * 98,
                      "p_arrive": np.linspace(0, 1, 200)})
    table, _ = am.within_direction(d, "rated", min_arrivals=8)
    assert list(table.direction) == ["A->B"]


def test_the_probabilities_are_put_back_on_the_real_base_rate():
    """The fit is class-balanced, so its raw output describes a 50/50 world."""
    rng = np.random.default_rng(2)
    n = 3000
    train = pd.DataFrame({
        "class_source": rng.normal(50, 20, n), "source_matches": rng.integers(3, 25, n),
        "source_minutes": rng.integers(100, 1500, n), "mins_pg": rng.normal(50, 12, n),
        "age": rng.normal(25, 3, n),
        "pair": rng.choice(["A->B", "C->D"], n),
        "group": rng.choice(["Middles", "Edge"], n)})
    train["rated"] = rng.random(n) < 0.05
    f = am.fit(train, "rated")
    assert f is not None
    p = am.predict(f, train)
    assert 0.0 < p.mean() < 0.15, f"mean predicted {p.mean():.3f} is not near the 5% base"
    assert ((p >= 0) & (p <= 1)).all()


def test_fit_refuses_rather_than_pretending_on_too_few_arrivals():
    d = pd.DataFrame({"class_source": np.arange(100.0), "source_matches": 5,
                      "source_minutes": 400, "mins_pg": 50.0, "age": 25.0,
                      "pair": "A->B", "group": "Middles",
                      "rated": [True] * 3 + [False] * 97})
    assert am.fit(d, "rated") is None


# ── against the real data ────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(DB), reason="no database")
@pytest.mark.slow
def test_the_model_has_signal_inside_a_direction():
    """This test earned its keep by failing.

    Written on 2026-09-23 asserting the opposite — that within-direction AUC stayed below
    0.60 — because the model of the day could not beat a coin inside a direction and the
    report concluded the data held no signal. The fourth review showed the conclusion was
    wrong: source minutes alone rank NSW Cup arrivals at 0.71, and the pooled
    specification was throwing that away. Direction-specific slopes recovered it, this
    test failed as designed, and the verdict was rewritten.

    It now guards the other direction: if the signal disappears again, something has
    regressed.
    """
    d, _ = am.run([2024, 2025], "rated")
    _, weighted = am.within_direction(d, "rated")
    assert weighted > 0.60, (
        "within-direction AUC has collapsed; the coefficient compromise may be back")


@pytest.mark.skipif(not os.path.exists(DB), reason="no database")
@pytest.mark.slow
def test_the_model_beats_its_own_inputs_where_the_client_looks():
    """A model given a column must beat that column, and this one did not for months."""
    d, _ = am.run([2024, 2025], "rated")
    bl = am.baselines(d, "rated").set_index("direction")
    for direction in ("NSW->NRL", "QLD->NRL"):
        if direction not in bl.index:
            continue
        r = bl.loc[direction]
        assert r.model > r.best_single, (
            f"{direction}: the model scores {r.model:.3f} against {r.best_single:.3f} "
            f"for the best single raw column")
