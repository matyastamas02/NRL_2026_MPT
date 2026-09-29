# -*- coding: utf-8 -*-
"""The position metrics, held to the conventions their docstring claims.

The dangerous failure here is not a crash. It is a metric the feed never recorded
summing to zero and being ranked as though the player did none of it — which is what
post-contact metres did before the availability rules were applied, putting 81% of one
season at the floor and the remaining 19% at the top.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metric_spec as ms
import position_metrics as pm

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "tallec.db")


# ── the formulas ─────────────────────────────────────────────────────────────

def _season(**kw):
    base = {c: np.array([0.0]) for c in ms.FEED}
    base["matches"] = np.array([10.0])
    base["minutes"] = np.array([600.0])
    for k, v in kw.items():
        base[k] = np.array([float(v)])
    return base


def test_a_volume_is_per_match_not_a_season_total():
    """Otherwise the benchmark ranks availability rather than contribution."""
    s = _season(RM=1000, matches=10)
    assert ms.VOLUME["RM"](s)[0] == pytest.approx(100.0)


def test_a_rate_divides_season_totals_on_both_sides():
    s = _season(RM=1000, Runs=100)
    assert ms.RATE["MpR"](s)[0] == pytest.approx(10.0)


def test_a_percentage_is_rebuilt_from_its_components():
    """One tackle made and none missed is not a 100% season if he played twenty games."""
    s = _season(TacklesMade=90, TacklesMissed=10)
    assert ms.RATE["TE%"](s)[0] == pytest.approx(90.0)


def test_no_opportunities_gives_nothing_not_infinity():
    s = _season(RM=500, Runs=0)
    assert np.isnan(ms.RATE["MpR"](s)[0])


def test_every_metric_in_the_spec_has_a_formula():
    vol = {m["volume"] for p in ms.SPEC.values() for c in p.values() for m in c}
    rat = {m["rate"] for p in ms.SPEC.values() for c in p.values() for m in c}
    assert not vol - set(ms.VOLUME)
    assert not rat - set(ms.RATE)


def test_combined_infringements_survives_a_missing_set_restart():
    """Super League has no set-restart data; penalties alone still rank within it."""
    s = _season(Penalties=20, SetRestart=0, matches=10)
    assert ms.VOLUME["Combined Infringements"](s)[0] == pytest.approx(2.0)


# ── availability and degradation ─────────────────────────────────────────────

def test_a_stat_not_recorded_blocks_the_metric_outright():
    ok, deg = pm._available(["PCM", "Runs"], (), np.array(["NRL"] * 2),
                            np.array([2023, 2025]))
    assert list(ok) == [False, True]
    assert not deg.any()


def test_an_optional_input_degrades_instead_of_blocking():
    ok, deg = pm._available(["Penalties", "SetRestart"], ("SetRestart",),
                            np.array(["SL", "NRL"]), np.array([2025, 2025]))
    assert list(ok) == [True, True], "it must still be computed"
    assert list(deg) == [True, False], "and Super League must be flagged"


def test_lower_is_better_metrics_are_inverted():
    """A clean player scores high, so every figure on the page reads the same way."""
    v = pd.DataFrame({
        "comp": ["NRL"] * 20, "season": [2025] * 20, "position": ["Prop"] * 20,
        "metric": ["Errors"] * 20, "form": ["volume"] * 20,
        "value": np.arange(20, dtype=float)})
    out = pm.benchmark(v)
    worst = out[out.value == 19].score.iloc[0]
    best = out[out.value == 0].score.iloc[0]
    assert best > worst
    assert "Errors" in ms.LOWER_IS_BETTER


def test_more_is_better_metrics_are_not_inverted():
    v = pd.DataFrame({
        "comp": ["NRL"] * 20, "season": [2025] * 20, "position": ["Prop"] * 20,
        "metric": ["RM"] * 20, "form": ["volume"] * 20,
        "value": np.arange(20, dtype=float)})
    out = pm.benchmark(v)
    assert out[out.value == 19].score.iloc[0] > out[out.value == 0].score.iloc[0]


def test_a_pool_too_small_to_rank_is_left_unscored():
    v = pd.DataFrame({
        "comp": ["SL"] * 4, "season": [2025] * 4, "position": ["Lock"] * 4,
        "metric": ["RM"] * 4, "form": ["volume"] * 4, "value": [1.0, 2.0, 3.0, 4.0]})
    assert pm.benchmark(v).score.isna().all()


# ── against the real data ────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(DB), reason="no database")
def test_post_contact_metres_is_not_scored_before_it_was_recorded():
    con = sqlite3.connect(DB)
    d = pd.read_sql("SELECT season, COUNT(*) n, SUM(score IS NOT NULL) scored "
                    "FROM player_position_metrics WHERE metric='PCM' GROUP BY 1", con)
    con.close()
    if d.empty:
        pytest.skip("metrics not built yet")
    before = d[d.season < 2025]
    assert (before.scored == 0).all(), "a stat that was never recorded was ranked"
    assert d[d.season == 2025].scored.iloc[0] > 0


@pytest.mark.skipif(not os.path.exists(DB), reason="no database")
def test_every_scored_figure_sits_on_the_published_scale():
    con = sqlite3.connect(DB)
    d = pd.read_sql("SELECT MIN(score) lo, MAX(score) hi FROM player_position_metrics "
                    "WHERE score IS NOT NULL", con)
    con.close()
    assert 0 <= d.lo.iloc[0] and d.hi.iloc[0] <= 100


# ── what the external review of 2026-09-22 found ─────────────────────────────

def _pool(metric, n=12, values=None):
    v = values if values is not None else np.arange(n, dtype=float)
    return pd.DataFrame({
        "comp": ["NRL"] * len(v), "season": [2025] * len(v),
        "position": ["Middles"] * len(v), "metric": [metric] * len(v),
        "form": ["volume"] * len(v), "value": v})


def test_the_two_directions_share_one_scale():
    """Inverting used to cost the best player 8.33 points for facing the wrong way.

    `100 - percentile` is not symmetric: in a pool of twelve it ran 8.33 to 100 one way
    and 0 to 91.67 the other, so the cleanest tackler in a group could not reach what the
    best runner reached. The ranks are mirrored now instead of the scores.
    """
    hi = pm.benchmark(_pool("RM")).score
    lo = pm.benchmark(_pool("Errors")).score
    assert lo.min() == pytest.approx(hi.min())
    assert lo.max() == pytest.approx(hi.max())
    assert lo.mean() == pytest.approx(hi.mean())


def test_reversing_the_values_reverses_the_scores_exactly():
    v = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0, 8.0, 7.0, 6.0, 0.5, 2.0])
    hi = pm.benchmark(_pool("RM", values=v)).score.values
    lo = pm.benchmark(_pool("Errors", values=v)).score.values
    assert np.allclose(hi, 100.0 - lo)


def test_the_top_of_a_pool_is_not_branded_a_flat_hundred():
    s = pm.benchmark(_pool("RM")).score
    assert 0.0 < s.min() and s.max() < 100.0


def test_the_pool_counts_players_who_can_be_ranked_not_rows():
    """Twelve rows with three values is a field of three, whatever MIN_POOL reads."""
    v = np.full(12, np.nan)
    v[:3] = [1.0, 2.0, 3.0]
    out = pm.benchmark(_pool("RM", values=v))
    assert (out["pool"] == 3).all()
    assert out.score.isna().all()


def test_a_blocked_metric_is_not_computed_at_all():
    """Forced drop-outs used the dropout the player's own side took while defending."""
    blocked = [m for cats in ms.SPEC.values() for ml in cats.values() for m in ml
               if m.get("blocked")]
    assert blocked, "the FDO metrics should still be blocked until DATA-2 arrives"
    for m in blocked:
        assert "DATA-2" in m["blocked"]


def test_the_conversion_metric_no_longer_promises_a_column_it_lacks():
    names = {m["rate"] for cats in ms.SPEC.values() for ml in cats.values() for m in ml}
    assert not any("KTA" in n for n in names), (
        "a metric may not name an input the feed does not have")
    using = [m for cats in ms.SPEC.values() for ml in cats.values() for m in ml
             if m["rate"].startswith("CV (")]
    assert using
    for m in using:
        assert "KTA" in m["optional"], "the missing input has to flag the row degraded"


def test_the_missing_conversion_input_actually_degrades_a_row():
    ok, degraded = pm._available(["LB", "LBA", "KTA", "Receipts"], ("KTA",),
                                 np.array(["NRL", "SL"]), np.array([2025, 2025]))
    assert ok.all(), "a floor still orders its own pool"
    assert degraded.all(), "and every row of it has to say it is a floor"


@pytest.mark.skipif(not os.path.exists(DB), reason="no database")
def test_a_season_with_no_position_is_counted_rather_than_vanishing():
    con = sqlite3.connect(DB)
    pos = pm.positions(con, through=2025)
    con.close()
    assert set(pos.position_from) <= {"season", "career", "none"}
    assert (pos.position_from == "none").sum() > 0, (
        "there are player-seasons with no usable position; they must appear here "
        "rather than being dropped by the query that looks for them")
    assert pos[pos.position_from == "none"]["group"].isna().all()


@pytest.mark.skipif(not os.path.exists(DB), reason="no database")
def test_a_category_score_says_what_it_was_made_of():
    s = pm.build(through=2025)
    cat = pm.category_scores(s)
    for col in ("metrics", "of", "degraded"):
        assert col in cat.columns
    assert (cat.metrics <= cat["of"]).all()
    assert (cat.degraded <= cat.metrics).all()
    assert cat.degraded.sum() > 0, "the KTA gap should reach the category level"
