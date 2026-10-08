# -*- coding: utf-8 -*-
"""Moves into Super League are forecast by a straight line, everything else by the model.

Reads translation_pairs_v3 from whichever database is present; skipped when neither is.
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


def _independent_line(source):
    db = next(p for p in DBS if os.path.exists(p))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    p = pd.read_sql("SELECT class_source, class_target FROM translation_pairs_v3 "
                    "WHERE layer='B_next_season' AND source=? AND target='SL'", con,
                    params=(source,))
    con.close()
    return np.polyfit(p.class_source, p.class_target, 1)


@pytest.mark.parametrize("source", ["NRL", "NSW", "QLD"])
def test_a_move_into_super_league_is_forecast_by_its_direction_line(source):
    slope, icept = _independent_line(source)
    r = pt.translate(70, source, "SL")
    assert r["forecast_method"] == "straight line"
    assert r["score_target"] == pytest.approx(slope * 70 + icept)
    assert r["score_forecast"] == r["score_target"]
    assert r["band_points"] == pytest.approx(1.96 * r["line"]["resid_sd"])


def test_the_model_number_is_still_reported_beside_the_line():
    r = pt.translate(70, "NRL", "SL")
    assert r["score_model"] is not None and r["score_model"] != r["score_target"]


def test_other_directions_keep_the_conditional_model():
    r = pt.translate(70, "NRL", "NSW")
    assert r["forecast_method"] == "conditional model"
    assert r["score_target"] == r["score_model"]


def test_a_horizon_without_enough_pairs_falls_back_to_the_model():
    r = pt.translate(70, "NRL", "SL", horizon="same_season")
    assert r["forecast_method"] == "conditional model"
