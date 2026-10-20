# -*- coding: utf-8 -*-
"""The Comparison page's calibrated ranges and decision outputs.

Reads translation_pairs_v3 and the tables build_decision_tables.py writes; skipped when
no database is present.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
DBS = [os.path.join(HERE, f) for f in ("tallec.db", "tallec_app.db")]
pytestmark = pytest.mark.skipif(not any(os.path.exists(p) for p in DBS),
                                reason="no database next to the code")

import predict_translation as pt


@pytest.mark.parametrize("source,target", [("NRL", "SL"), ("NSW", "SL"), ("QLD", "SL"),
                                           ("NRL", "NSW"), ("SL", "NRL")])
def test_the_ranges_nest_and_sit_on_the_scale(source, target):
    r = pt.translate(62, source, target)
    (a, b), (c, d) = r["range50"], r["range80"]
    assert 0 <= c <= a <= b <= d <= 100
    assert r["range_n"] > 0 and r["range_basis"] in ("direction", "pooled")


def test_the_line_range_is_the_forecast_plus_the_lines_residual_quantiles():
    db = next(p for p in DBS if os.path.exists(p))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    p = pd.read_sql("SELECT class_source, class_target FROM translation_pairs_v3 "
                    "WHERE layer='B_next_season' AND source='NRL' AND target='SL'", con)
    con.close()
    slope, icept = np.polyfit(p.class_source, p.class_target, 1)
    q = np.percentile(p.class_target - (slope * p.class_source + icept), [10, 25, 75, 90])
    r = pt.translate(55, "NRL", "SL")
    f = slope * 55 + icept
    assert r["range80"] == pytest.approx(tuple(np.clip([f + q[0], f + q[3]], 0, 100)))
    assert r["range50"] == pytest.approx(tuple(np.clip([f + q[1], f + q[2]], 0, 100)))


def test_the_top_third_probability_rises_with_the_forecast():
    lo, hi = pt.top_third(40), pt.top_third(75)
    if lo is None:
        pytest.skip("decision_calibration not built")
    assert 0 < lo["p"] < hi["p"] < 1


def test_the_calibration_uses_the_app_line():
    import build_decision_tables as bd
    db = next(p for p in DBS if os.path.exists(p))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    p = pd.read_sql("SELECT source, target, class_source, class_target FROM "
                    "translation_pairs_v3 WHERE layer='B_next_season'", con)
    con.close()
    p["pair"] = p.source + "->" + p.target
    fitted = bd.line_fitted(p)
    for src in ("NRL", "NSW", "QLD"):
        g = p[p.pair == f"{src}->SL"].head(5)
        line = pt._line(src, "SL", "B_next_season")
        want = np.clip(line["slope"] * g.class_source + line["intercept"], 0, 100)
        assert fitted[g.index] == pytest.approx(want.values)


def test_start_records_and_bench_evidence_are_well_formed():
    ev = pt.bench_evidence("SL")
    if ev is None:
        pytest.skip("bench_evidence not built")
    assert ev["n_bench"] > 0 and ev["n_start"] > 0
    assert 0 <= ev["starter_bench"] <= 1 and 0 <= ev["starter_start"] <= 1
    db = next(p for p in DBS if os.path.exists(p))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = pd.read_sql("SELECT * FROM player_season_starts LIMIT 1", con).iloc[0]
    con.close()
    rec = pt.start_record(row.player_id, row.comp, row.season)
    assert rec["starts"] <= rec["games"] and rec["starter"] in (0, 1)
