# -*- coding: utf-8 -*-
"""The path the match model actually uses, held to being leak-free end to end.

The third external review of 2026-09-23 was right that `PlayerRatingEngine.
compute_prematch` fits its pool statistics and variance components on the whole frame,
future matches included, while its docstring implied otherwise. That docstring is fixed.

What was left unproven is the claim made in its place: that the predictive path — the one
`teamlist_backtest.py` and `gigot_v2.py` run — does not use that function and is strictly
walk-forward. Reading the code says so. These tests measure it, because reading the code
is how three of this project's faults survived.

Two separable claims, tested separately:

  * the standardisation for a season is fitted on seasons strictly before it;
  * a match's own statistics never reach the features attached to that match, nor to any
    earlier one.

Synthetic data, so this runs in a second and does not depend on what the database happens
to hold.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gigot_v2 as g
import teamlist_backtest as tb


def _feed(seasons=(2022, 2023, 2024), players=40, rounds=8, seed=0):
    """A small multi-season competition with the columns the GIGOT path reads."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        # each season has its own level, so a standardisation fitted on the wrong
        # seasons would show up as a shifted composite
        level = 1.0 + 0.4 * (s - min(seasons))
        for pid in range(players):
            skill = rng.normal(0, 1)
            for rd in range(1, rounds + 1):
                rows.append({
                    "player_id": f"p{pid}", "player": f"Player {pid}",
                    "season": s, "round": rd,
                    "team": f"T{pid % 8}", "position": "Prop",
                    "minutes": int(rng.integers(40, 81)),
                    "all_run_metres": float(level * 100 * np.exp(0.2 * skill)
                                            + rng.normal(0, 8)),
                    "p_c_m": float(level * 40 + rng.normal(0, 5)),
                    "tackle_breaks": float(max(0, level * 2 + rng.normal(0, 1))),
                    "line_breaks": float(max(0, rng.normal(0.4, 0.4))),
                    "tackles": float(level * 25 + rng.normal(0, 4)),
                    "offloads": float(max(0, rng.normal(1.0, 0.8))),
                    "try_assists": float(max(0, rng.normal(0.2, 0.3))),
                    "tries": float(max(0, rng.normal(0.2, 0.4))),
                    "errors": float(max(0, rng.normal(1.0, 0.7))),
                })
    return pd.DataFrame(rows)


def _con(df):
    con = sqlite3.connect(":memory:")
    df.assign(competition="TEST").to_sql("player_match_stats", con, index=False)
    return con


def test_the_predictive_path_never_calls_the_function_that_is_not_walk_forward():
    """`compute_prematch` fits on everything; the match model must not reach it."""
    for mod in (g, tb):
        src = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), f"{mod.__name__}.py"), encoding="utf-8").read()
        assert "compute_prematch" not in src, (
            f"{mod.__name__} reaches compute_prematch, whose pool statistics are fitted "
            f"on the whole period including the future")
        assert "_fit_variance_components" not in src
        assert "_shrink" not in src


def test_a_seasons_standardisation_uses_only_earlier_seasons():
    """Change the last season only; every earlier row must be untouched."""
    df = _feed()
    base = g.prematch_players("TEST", _con(df), walk_forward=True)

    tampered = df.copy()
    late = tampered.season == tampered.season.max()
    tampered.loc[late, "all_run_metres"] *= 4.0
    after = g.prematch_players("TEST", _con(tampered), walk_forward=True)

    key = ["player_id", "season", "round"]
    b = base.set_index(key).sort_index()
    a = after.set_index(key).sort_index()
    early = b.index.get_level_values("season") < df.season.max()
    assert early.any()
    assert np.allclose(b.loc[early, "composite"], a.loc[early, "composite"]), (
        "a later season moved an earlier season's composite, so the standardisation "
        "is not fitted on the past alone")


def _tamper_one_match(df, season, player="p0", rd=1, factor=8.0):
    t = df.copy()
    row = (t.season == season) & (t.player_id == player) & (t["round"] == rd)
    assert row.sum() == 1
    t.loc[row, "all_run_metres"] *= factor
    return t, row


def test_one_match_does_not_move_its_own_seasons_other_matches():
    """The leak the review found in `compute_prematch`, tested on the path that ships.

    Pool statistics fitted on the season being rated make every match in it depend on
    every other: change one man's run metres and the z-score of everyone he is compared
    against shifts. Walk-forward fits them on earlier seasons, so it cannot happen.
    """
    df = _feed()
    season = int(df.season.max())
    tampered, row = _tamper_one_match(df, season)
    base = g.prematch_players("TEST", _con(df), walk_forward=True)
    after = g.prematch_players("TEST", _con(tampered), walk_forward=True)

    key = ["player_id", "season", "round"]
    b, a = base.set_index(key).sort_index(), after.set_index(key).sort_index()
    same_season = b.index.get_level_values("season") == season
    others = same_season & (b.index.get_level_values("player_id") != "p0")
    assert others.sum() > 100
    assert np.allclose(b.loc[others, "composite"], a.loc[others, "composite"]), (
        "one match moved another player's composite in the same season, so the pool "
        "statistics are being fitted on the season they describe")


def test_the_control_leaks_so_the_test_above_proves_something():
    """Without walk-forward each season is standardised on itself, and it does leak."""
    df = _feed()
    season = int(df.season.max())
    tampered, _ = _tamper_one_match(df, season)
    base = g.prematch_players("TEST", _con(df), walk_forward=False)
    after = g.prematch_players("TEST", _con(tampered), walk_forward=False)

    key = ["player_id", "season", "round"]
    b, a = base.set_index(key).sort_index(), after.set_index(key).sort_index()
    same_season = b.index.get_level_values("season") == season
    others = same_season & (b.index.get_level_values("player_id") != "p0")
    assert not np.allclose(b.loc[others, "composite"], a.loc[others, "composite"]), (
        "the control should leak; if it does not, the assertion above proves nothing")


def test_a_match_never_reaches_its_own_pre_match_features():
    """The leak that matters most: match M's statistics informing match M's rating."""
    df = _feed()
    con = _con(df)
    base = g.prematch_players("TEST", con, walk_forward=True)

    # scramble the performance column across rows. The pre-match features describe
    # EARLIER matches, so they may move — what must not happen is a row's own numbers
    # changing the features attached to that row. The engine's own permutation hook
    # does this on the composites directly.
    shuffled = g.prematch_players("TEST", _con(df), walk_forward=True,
                                  shuffle_outcome=True, seed=3)
    assert not np.allclose(base.prior_class.fillna(-9),
                           shuffled.prior_class.fillna(-9)), (
        "permuting performance changed nothing, so the fixture is not exercising "
        "the feature at all")

    # and the real assertion: within a player, the feature for his match k is built
    # from matches 1..k-1 and nothing else
    for pid, gp in base.sort_values(["season", "round"]).groupby("player_id"):
        comp = gp.composite.values
        want = pd.Series(comp).shift(1).expanding().mean().values
        assert np.allclose(gp.prior_class.values, want, equal_nan=True), pid
        break


def test_the_first_season_is_flagged_rather_than_quietly_fitted_on_itself():
    df = _feed()
    pm = g.prematch_players("TEST", _con(df), walk_forward=True)
    first = pm[pm.season == df.season.min()]
    later = pm[pm.season > df.season.min()]
    assert first.warmup.all(), "the season with nothing before it must be flagged"
    assert not later.warmup.any()


def test_the_team_list_backtest_takes_the_walk_forward_default():
    """It calls `prematch_players` without arguments, so the default decides."""
    import inspect
    sig = inspect.signature(g.prematch_players)
    assert sig.parameters["walk_forward"].default is True
    src = inspect.getsource(tb.build)
    assert "prematch_players(comp, con)" in src
    assert "walk_forward=False" not in src
