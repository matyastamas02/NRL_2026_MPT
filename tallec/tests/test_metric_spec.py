# -*- coding: utf-8 -*-
"""The position specification came from a spreadsheet; these hold it to its shape."""
import os
import sqlite3
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metric_spec as ms
import sp_schema as sp

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "tallec.db")


def test_every_position_has_twentyfour_figures():
    """Mike's sheet gives 12 metrics per position, each in a volume and a rate form."""
    for pos, n in ms.counts().items():
        assert n == 24, f"{pos} has {n}"


def test_every_position_has_all_four_categories():
    for pos, cats in ms.SPEC.items():
        assert list(cats) == ms.CATEGORIES, pos


def test_every_position_maps_to_a_real_engine_group():
    groups = set(sp.POSITION_GROUP.values())
    for pos, meta in ms.POSITIONS.items():
        assert meta["engine_group"] in groups, pos


def test_every_feed_code_names_a_column_that_exists():
    if not os.path.exists(DB):
        pytest.skip("no database")
    con = sqlite3.connect(DB)
    cols = set(pd.read_sql("SELECT * FROM player_match_raw LIMIT 1", con).columns)
    con.close()
    missing = {k: v for k, v in ms.FEED.items() if v not in cols}
    assert not missing, f"feed columns not in player_match_raw: {missing}"


def test_every_input_the_spec_uses_is_declared_somewhere():
    """An input is either a feed column or an absence we have written down.

    `KTA` is the second kind. Mike's conversion metric names kick try assists and this
    extract has no such column, so the metric is computed without it. Leaving KTA out of
    the spec's inputs kept this test green while the shortfall lived only in a prose
    note — which is exactly how the external review of 2026-09-22 came to find a metric
    promising an input it never had. It is now named as an input and declared missing in
    every competition, so the rows it affects are flagged degraded.

    The rule this enforces: a code may be absent from FEED only if AVAILABILITY says it
    is absent from every competition. Anything else is a typo.
    """
    comps = {c[0] for c in sp.COMPETITIONS}
    bad = []
    for code in ms.all_inputs():
        if code in ms.FEED:
            continue
        av = ms.AVAILABILITY.get(code) or {}
        if set(av.get("missing_in") or []) < comps:
            bad.append(code)
    assert not bad, (
        f"{bad} are neither feed columns nor declared missing everywhere")


def test_post_contact_metres_is_blocked_before_2025():
    """The known gap: it is absent before 2025 and most profiles depend on it."""
    r = ms.resolve("Middles", "NRL", 2023)
    pcm = [m for m in r["Yardage"] if m["volume"] == "PCM"][0]
    assert not pcm["computable"]
    assert any("2025" in b for b in pcm["blocked_by"])
    assert [m for m in ms.resolve("Middles", "NRL", 2025)["Yardage"]
            if m["volume"] == "PCM"][0]["computable"]


def test_set_restarts_block_super_league_only():
    aus = ms.resolve("Middles", "NRL", 2025)["Defence & Discipline"]
    sl = ms.resolve("Middles", "SL", 2025)["Defence & Discipline"]
    name = "Combined Infringements"
    assert [m for m in aus if m["volume"] == name][0]["computable"]
    assert not [m for m in sl if m["volume"] == name][0]["computable"]


def test_mikes_answers_are_recorded_with_the_outstanding_requests():
    """Six months on, nobody remembers which choices were decisions and which guesses."""
    assert set(ms.RESOLVED) == {"ninth block", "Back Row and Lock",
                                "Combined Infringements", "LKS", "Break Cause",
                                "PCM and LER"}
    assert set(ms.DATA_REQUESTS) == {"DATA-1", "DATA-2"}
    for src in (ms.RESOLVED, ms.DATA_REQUESTS):
        for k, v in src.items():
            assert len(v) > 80, k


def test_interchange_is_a_position_not_a_second_winger():
    assert "IT" in ms.SPEC
    assert ms.POSITIONS["IT"]["label"] == "Interchange"
    # its metric set is the middle-forward profile, which is what the sheet gave
    mid = [m["volume"] for c in ms.SPEC["Middles"].values() for m in c]
    it = [m["volume"] for c in ms.SPEC["IT"].values() for m in c]
    assert mid == it


def test_middles_and_edge_are_the_forward_groups():
    """Leeds, 19 September: props and locks together as Middles, second row as Edge.
    Recorded as a test because the forwards have now been grouped three different ways
    and the next change should have to be deliberate."""
    assert ms.POSITIONS["Middles"]["engine_group"] == "Middles"
    assert ms.POSITIONS["Edge"]["engine_group"] == "Edge"
    assert "Lock" not in ms.SPEC and "Prop" not in ms.SPEC
    # what the merge gave up is kept where it can be found again
    assert ms.LOCK_BLOCK_RETIRED["Attack"][0]["volume"] == "LBA"


def test_break_cause_is_used_as_the_count_it_is():
    """An earlier reading had it as a category code and refused to sum it. It is a
    count: 64,617 rows at nought decaying monotonically to twelve."""
    for pos in ("CT", "Halves"):
        m = [x for x in ms.SPEC[pos]["Defence & Discipline"]
             if x["volume"] == "Break Cause"]
        assert m and "BreakCause" in m[0]["inputs"], pos
