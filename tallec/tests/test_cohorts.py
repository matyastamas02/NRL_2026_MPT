# -*- coding: utf-8 -*-
"""The cohort split is what an external review found missing; it is held to its meaning."""
import os
import sqlite3
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cohorts as ch

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "tallec.db")


def _db(rows):
    con = sqlite3.connect(":memory:")
    pd.DataFrame(rows).to_sql("player_match_stats", con, index=False)
    return con


def _rows(spec):
    """spec: {player: [(season, n_matches), ...]} in one competition."""
    out = []
    for pid, seasons in spec.items():
        for season, n in seasons:
            out += [dict(player_id=pid, competition="NRL", season=season, round=str(r))
                    for r in range(1, n + 1)]
    return out


def test_the_three_states_are_what_they_say():
    con = _db(_rows({
        "debut":     [(2026, 5)],                 # never there before
        "returner":  [(2024, 10), (2026, 5)],     # there before, not last season
        "continuer": [(2025, 10), (2026, 5)],     # there last season too
    }))
    d = ch.classify(con, "NRL", 2026).set_index("player_id").cohort
    assert d["debut"] == "first"
    assert d["returner"] == "returning"
    assert d["continuer"] == "continuing"
    con.close()


def test_someone_absent_this_season_is_not_classified_at_all():
    con = _db(_rows({"gone": [(2024, 10)], "here": [(2026, 3)]}))
    d = ch.classify(con, "NRL", 2026)
    assert set(d.player_id) == {"here"}
    con.close()


def test_history_can_be_capped_so_hindsight_is_excluded():
    """A walk-forward evaluation asks the question as it stood, not as it turned out."""
    con = _db(_rows({"p": [(2024, 5), (2025, 5), (2026, 5)]}))
    # asked at the time, with 2025 known, he is continuing
    assert ch.classify(con, "NRL", 2026, through=2026).iloc[0].cohort == "continuing"
    con.close()


def test_min_matches_decides_what_counts_as_having_been_there():
    """One bench cameo is not really a season in the competition."""
    con = _db(_rows({"cameo": [(2025, 1), (2026, 5)]}))
    assert ch.classify(con, "NRL", 2026).iloc[0].cohort == "continuing"
    assert ch.classify(con, "NRL", 2026, min_matches=2).iloc[0].cohort == "first"
    con.close()


def test_one_row_per_player_keeps_the_better_evidenced_side():
    d = pd.DataFrame([dict(player_id="p", source="NSW", g=4),
                      dict(player_id="p", source="QLD", g=11),
                      dict(player_id="q", source="NSW", g=7)])
    out = ch.one_row_per_player(d, prefer="g")
    assert len(out) == 2
    assert out[out.player_id == "p"].iloc[0].source == "QLD"


def test_summarise_always_lists_all_three_in_a_fixed_order():
    d = pd.DataFrame({"cohort": ["first", "first", "continuing"]})
    s = ch.summarise(d)
    assert list(s.cohort) == ch.LABELS
    assert list(s.n) == [2, 0, 1]


@pytest.mark.skipif(not os.path.exists(DB), reason="no database")
def test_the_real_nrl_seasons_are_mostly_continuing_players():
    """A competition where most of the squad is new every year would be a red flag."""
    con = sqlite3.connect(DB)
    for season in (2024, 2025, 2026):
        d = ch.classify(con, "NRL", season)
        share = (d.cohort == "continuing").mean()
        assert 0.70 < share < 0.95, (season, share)
    con.close()
