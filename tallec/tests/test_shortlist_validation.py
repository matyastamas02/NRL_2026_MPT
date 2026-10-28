# -*- coding: utf-8 -*-
"""The shortlist comparison: each season's top third, the same men for every rule.

Synthetic frames only; tallec.db is never opened.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import validate_shortlist as vs


def _entrants():
    return pd.DataFrame({
        "player_id": [f"p{i}" for i in range(9)],
        "season_tgt": [2024] * 6 + [2025] * 3,
        "score": [9, 8, 1, 2, 3, 4, 5, 5, 1],
        "top3": [1, 0, 0, 0, 1, 0, 0, 1, 1]})


def test_the_shortlist_is_the_top_third_of_each_season():
    h, k = vs.shortlist_hits(_entrants(), "score")
    assert k == 2 + 1          # a third of six, a third of three
    assert h == 1 + 0          # p0 hits, p1 misses; in 2025 p6 wins the tie and misses


def test_ties_are_broken_by_player_not_by_row_order():
    d = _entrants()
    a = vs.shortlist_hits(d, "score")
    b = vs.shortlist_hits(d.iloc[::-1].reset_index(drop=True), "score")
    assert a == b


def test_positions_split_into_forwards_and_backs():
    u = vs.unit(["Prop", "Hooker", "Second Row", "Lock", "Winger", "Half Back", None])
    assert list(u) == ["forwards", "forwards", "forwards", "forwards", "backs", "backs",
                       "unknown"]
