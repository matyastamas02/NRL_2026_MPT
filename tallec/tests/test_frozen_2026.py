# -*- coding: utf-8 -*-
"""Rule 16 of FROZEN_2026_SPEC.md: the estimator, the timing and the decision, tested.

Synthetic frames except the last two tests, which read translation_pairs_v3 and the
match table and skip when no database is present.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import frozen_2026 as fz


def _pairs(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pair, n, a, b in (("NRL->SL", 60, 40.0, 0.4), ("NSW->SL", 30, 45.0, 0.3),
                          ("QLD->SL", 12, 35.0, 0.35)):
        x = rng.uniform(20, 90, n)
        rows.append(pd.DataFrame({"pair": pair, "class_source": x,
                                  "class_target": a + b * x + rng.normal(0, 15, n),
                                  "flag": rng.integers(0, 2, n).astype(float),
                                  "num": np.where(rng.random(n) < .2, np.nan, rng.normal(0, 1, n))}))
    return pd.concat(rows, ignore_index=True)


def test_without_extras_the_estimator_is_own_and_all_pairs_lines():
    p = _pairs()
    m = fz.fit_lines(p)
    assert set(m["own"]) == {"NRL->SL", "NSW->SL"}          # QLD has 12 < 25 rows
    for pair in m["own"]:
        g = p[p.pair == pair]
        b, a = np.polyfit(g.class_source, g.class_target, 1)
        assert m["own"][pair]["coef"] == pytest.approx([a, b])
    b, a = np.polyfit(p.class_source, p.class_target, 1)
    assert m["pooled"]["coef"] == pytest.approx([a, b])
    pred, basis = fz.predict_lines(m, p)
    q = p.pair == "QLD->SL"
    assert set(basis[q]) == {"pooled"} and set(basis[~q]) == {"direction"}
    assert pred[q] == pytest.approx(np.clip(a + b * p.class_source[q], 0, 100))


def test_a_constant_source_falls_back_to_the_pooled_line():
    p = _pairs()
    p.loc[p.pair == "NSW->SL", "class_source"] = 55.0
    assert "NSW->SL" not in fz.fit_lines(p)["own"]


def test_a_constant_feature_is_zeroed_and_changes_nothing():
    p = _pairs().assign(const=1.0)
    with_const = fz.fit_lines(p, [("const", "binary")])
    plain = fz.fit_lines(p)
    a, _ = fz.predict_lines(with_const, p)
    b, _ = fz.predict_lines(plain, p)
    assert a == pytest.approx(b)


def test_the_feature_is_standardised_within_each_fitting_sample():
    p = _pairs()
    m = fz.fit_lines(p, [("flag", "binary")])
    g = p[p.pair == "NRL->SL"].flag
    assert m["own"]["NRL->SL"]["stats"]["flag"] == pytest.approx((g.mean(), g.std(ddof=1)))
    assert m["pooled"]["stats"]["flag"] == pytest.approx((p.flag.mean(), p.flag.std(ddof=1)))


def test_numeric_extras_impute_and_carry_an_unstandardised_flag():
    p = _pairs()
    X, stats = fz._columns(p, [("num", "numeric")])
    assert X.shape[1] == 4
    assert set(np.unique(X[:, 3])) == {0.0, 1.0}
    assert np.allclose(X[p.num.isna().to_numpy(), 2], 0.0)


def test_the_outcomes_being_scored_cannot_move_the_fit():
    train, test = _pairs(1), _pairs(2)
    m = fz.fit_lines(train, [("flag", "binary")])
    before, _ = fz.predict_lines(m, test)
    test["class_target"] = test.class_target + 50
    after, _ = fz.predict_lines(fz.fit_lines(train, [("flag", "binary")]), test)
    assert before == pytest.approx(after)


def test_history_reads_only_the_season_before_the_source_season():
    rows = pd.DataFrame({"player_id": ["1", "2"], "source": ["NRL", "NRL"],
                         "season_src": [2024, 2024]})
    sea = pd.DataFrame({"player_id": ["1", "2", "2"], "comp": ["NRL"] * 3,
                        "season": [2023, 2024, 2025], "n_games": [5, 9, 9]})
    covered = {("NRL", 2023)}
    h = fz.add_history(rows, sea, covered)
    assert h.rated_source_prev_year.tolist() == [1.0, 0.0]
    # a later season, or the source season itself, cannot change it
    sea2 = pd.concat([sea, pd.DataFrame({"player_id": ["1"], "comp": ["NRL"],
                                         "season": [2026], "n_games": [3]})])
    assert fz.add_history(rows, sea2, covered).rated_source_prev_year.tolist() == [1.0, 0.0]
    assert h.prev_year_covered.tolist() == [True, True]


def test_the_decision_rule():
    assert fz.decide(1.2, 0.1, 2.0, 30, 6, 24) == "POSITIVE EXPLORATORY SIGNAL"
    assert fz.decide(0.9, 0.1, 2.0, 30, 6, 24) == "UNDECIDED"
    assert fz.decide(-1.0, -2.0, -0.1, 30, 6, 24) == "EVIDENCE OF HARM"
    assert fz.decide(0.2, -0.5, 0.8, 30, 6, 24) == "A ONE-POINT GAIN IS NOT SUPPORTED"
    assert fz.decide(3.0, 1.0, 5.0, 24, 6, 18).startswith("UNDECIDED")
    assert fz.decide(3.0, 1.0, 5.0, 30, 4, 26).startswith("UNDECIDED")


def test_the_bootstrap_does_not_depend_on_row_order():
    rng = np.random.default_rng(3)
    n = 40
    y, a, b = rng.normal(50, 20, n), rng.normal(50, 20, n), rng.normal(50, 20, n)
    ids = np.array([f"p{i:02d}" for i in range(n)])
    one = fz.paired_bootstrap(y, a, b, ids)
    order = rng.permutation(n)
    assert fz.paired_bootstrap(y[order], a[order], b[order], ids[order]) == pytest.approx(one)


DBS = [os.path.join(HERE, f) for f in ("tallec.db", "tallec_app.db")]
needs_db = pytest.mark.skipif(not any(os.path.exists(p) for p in DBS), reason="no database")


@needs_db
def test_s0_on_the_shipped_pairs_is_the_app_line():
    import predict_translation as pt
    db = next(p for p in DBS if os.path.exists(p))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    p = pd.read_sql("SELECT source, target, class_source, class_target FROM "
                    "translation_pairs_v3 WHERE layer='B_next_season'", con)
    con.close()
    p["pair"] = p.source + "->" + p.target
    m = fz.fit_lines(p)
    for src in ("NRL", "NSW", "QLD"):
        app = pt._line(src, "SL", "B_next_season")
        a, b = m["own"][f"{src}->SL"]["coef"]
        assert (a, b) == pytest.approx((app["intercept"], app["slope"]))
    a, b = m["pooled"]["coef"]
    app = pt._line("SL", "NSW", "B_next_season")
    assert app["basis"] == "pooled" and (a, b) == pytest.approx((app["intercept"], app["slope"]))


@pytest.mark.skipif(not os.path.exists(DBS[0]), reason="needs the full database")
def test_knowledge_through_the_freeze_never_reads_a_later_season():
    import rating_history as rh
    con = sqlite3.connect(f"file:{DBS[0]}?mode=ro", uri=True)
    for comp in ("NRL", "SL"):
        assert rh.load(con, comp, through=fz.FREEZE_SEASON).season.max() <= fz.FREEZE_SEASON
    con.close()
