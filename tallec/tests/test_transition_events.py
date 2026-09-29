# -*- coding: utf-8 -*-
"""The cohort a recruitment question is asked about, held to being one.

Each test here is a defect the external review of 2026-09-22 found in the set that
`rolling_backtest` used to infer. That set joined every competition a player appeared in
last season to every competition he appeared in this one, which produced 1,140 rows from
531 players and turned a fringe forward shuttling between the NRL and the NSW Cup into
two opposite "moves" scored independently.
"""
import os
import sqlite3
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transition_events as te

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "tallec.db")
pytestmark = pytest.mark.skipif(not os.path.exists(DB), reason="no database")


@pytest.fixture(scope="module")
def events():
    con = sqlite3.connect(DB)
    ev = te.build(con, through=2025)
    con.close()
    return ev


def test_one_row_per_player_target_and_season(events):
    """The duplicate the old construction produced by joining both directions."""
    key = ["player_id", "target", "season"]
    assert not events.duplicated(key).any()


def test_a_player_has_one_source_per_season(events):
    """Two sources would put the same man in the cohort twice from different levels."""
    per = events.groupby(["player_id", "season"]).source.nunique()
    assert (per == 1).all()


def test_no_reciprocal_pair_for_the_same_player_and_season(events):
    """A->B and B->A in one season is shuttling described as two transfers."""
    e = events[events.arrived]
    forward = set(zip(e.player_id, e.season, e.source, e.target))
    backward = {(p, s, t, o) for p, s, o, t in forward}
    assert not (forward & backward)


def test_source_is_never_the_target(events):
    assert (events.source != events.target).all()


def test_the_type_is_decided_before_the_season_it_forecasts(events):
    """Nothing about season T may decide whether a row is allowed into the evaluation.

    The first version of this module called a player `dual_registered` when he turned up
    in both competitions during the season being forecast — the outcome choosing the
    sample. Every type is now a statement about season T-1 or earlier, which shows up as
    types being assigned to rows where the player never arrived at all.
    """
    never = events[~events.arrived]
    assert len(never) > 0
    assert set(never.transition_type) <= set(te.TYPES)
    assert (never.transition_type != "").all()


def test_a_player_who_never_arrives_is_still_a_row(events):
    """Survivorship: the old set existed only where he later played three matches."""
    assert (~events.arrived).sum() > 0
    assert events.arrived.mean() < 0.5


def test_arrived_and_rated_are_nested(events):
    assert not (events.rated & ~events.arrived).any()
    assert (events.loc[events.rated, "target_matches"]
            >= te.MIN_RATED_MATCHES).all()
    assert (events.loc[~events.arrived, "target_matches"] == 0).all()


def test_dual_registration_means_both_competitions_last_season(events):
    dual = events[events.transition_type == "dual_registered"]
    assert len(dual) > 0
    con = sqlite3.connect(DB)
    app = te.appearances(con, through=2025)
    con.close()
    had = set(zip(app.player_id, app.comp, app.season))
    for r in dual.head(200).itertuples(index=False):
        assert (r.player_id, r.target, r.season - 1) in had
        assert (r.player_id, r.source, r.season - 1) in had


def test_first_means_never_seen_in_that_competition_before(events):
    first = events[events.transition_type == "first"]
    assert len(first) > 0
    con = sqlite3.connect(DB)
    app = te.appearances(con, through=2025)
    con.close()
    earliest = app.groupby(["player_id", "comp"]).season.min()
    for r in first.head(300).itertuples(index=False):
        e = earliest.get((r.player_id, r.target))
        assert e is None or e >= r.season


def test_the_source_choice_does_not_depend_on_row_order(events):
    """A tie broken by row order makes the cohort depend on how SQLite felt."""
    con = sqlite3.connect(DB)
    app = te.appearances(con, through=2025)
    con.close()
    a = te.pick_source(app[app.season == 2024])
    b = te.pick_source(app[app.season == 2024].sample(frac=1.0, random_state=7))
    m = a.merge(b, on=["player_id", "season"], suffixes=("_a", "_b"))
    assert (m.source_a == m.source_b).all()


def test_entries_leave_out_the_men_who_never_moved(events):
    ent = te.entries(events)
    assert "dual_registered" not in set(ent.transition_type)
    assert len(ent) < len(events)
    # and the filter is a view, not a deletion: the table still holds them
    assert (events.transition_type == "dual_registered").sum() > 0
