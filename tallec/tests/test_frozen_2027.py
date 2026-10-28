# -*- coding: utf-8 -*-
"""The 2027 protocol (FROZEN_2027_SPEC.md, rule 21). Synthetic data; tallec.db is never
opened."""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import frozen_2026 as fz
import frozen_2027 as f7


def _pairs(seed=1):
    rng = np.random.default_rng(seed)
    big = pd.DataFrame({"pair": "NRL->SL", "class_source": rng.uniform(20, 90, 40)})
    big["class_target"] = 0.4 * big.class_source + 40 + rng.normal(0, 8, 40)
    small = pd.DataFrame({"pair": "QLD->SL", "class_source": rng.uniform(20, 90, 10)})
    small["class_target"] = 0.3 * small.class_source + 35 + rng.normal(0, 8, 10)
    return pd.concat([big, small], ignore_index=True)


def test_b0_is_the_mean_of_the_rows_its_line_is_fitted_on():
    p = _pairs()
    L = f7.lines(p)
    rows = pd.DataFrame({"pair": ["NRL->SL", "QLD->SL"], "class_source": [60.0, 60.0]})
    fc = f7.forecast(L, rows)
    assert fc.B0.iloc[0] == pytest.approx(p[p.pair == "NRL->SL"].class_target.mean())
    # ten pairs are too few for a line of its own: the pooled mean, like S0's fallback
    assert fc.B0.iloc[1] == pytest.approx(p.class_target.mean())
    assert list(fc.basis) == ["direction", "pooled"]


def test_s0_is_the_2026_estimator():
    p = _pairs()
    rows = pd.DataFrame({"pair": ["NRL->SL", "QLD->SL", "NRL->SL"],
                         "class_source": [15.0, 55.0, 99.0]})
    mine = f7.forecast(f7.lines(p), rows).S0.to_numpy()
    theirs, _ = fz.predict_lines(fz.fit_lines(p), rows)
    assert np.max(np.abs(mine - theirs)) < 1e-9


def test_the_ranges_nest_around_the_forecast():
    fc = f7.forecast(f7.lines(_pairs()), pd.DataFrame(
        {"pair": ["NRL->SL"] * 3, "class_source": [5.0, 50.0, 95.0]}))
    assert (fc.lo80 <= fc.lo50).all() and (fc.lo50 <= fc.S0).all()
    assert (fc.S0 <= fc.hi50).all() and (fc.hi50 <= fc.hi80).all()
    assert fc[["lo80", "hi80"]].stack().between(0, 100).all()


@pytest.mark.parametrize("delta,lo,hi,n,want", [
    (2.0, 0.5, 3.5, 40, 0), (-1.0, -2.5, -0.1, 40, 1), (0.1, -0.6, 0.9, 40, 2),
    (0.8, -0.4, 2.0, 40, 3), (3.0, 1.0, 5.0, 24, 4), (0.9, 0.1, 1.7, 40, 3)])
def test_the_labels(delta, lo, hi, n, want):
    assert f7.label(delta, lo, hi, n) == f7.LABELS[want]


def test_the_config_fingerprint_ignores_the_freeze_season_but_not_a_weight(tmp_path):
    cfg = {"evaluation": {"freeze_season": 2026, "description": "x"},
           "class_calculation": {"season_weights": [1.0, 0.7], "description": "y"}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    h0 = f7.config_hash(path)
    cfg["evaluation"]["freeze_season"] = 2027
    cfg["class_calculation"]["description"] = "reworded"
    path.write_text(json.dumps(cfg))
    assert f7.config_hash(path) == h0
    cfg["class_calculation"]["season_weights"] = [1.0, 0.6]
    path.write_text(json.dumps(cfg))
    assert f7.config_hash(path) != h0


def test_a_freeze_is_refused_once_a_later_season_is_loaded(tmp_path):
    con = sqlite3.connect(tmp_path / "t.db")
    con.execute("CREATE TABLE player_match_stats (season INT)")
    con.executemany("INSERT INTO player_match_stats VALUES (?)", [(2025,), (2026,)])
    assert f7.can_freeze(con)
    con.execute("INSERT INTO player_match_stats VALUES (2027)")
    assert not f7.can_freeze(con)


def test_the_cutoff_is_not_read_from_the_config():
    """The app may roll its ratings forward during 2027; the test must not move."""
    src = open(f7.__file__).read()
    assert "TRAIN_THROUGH = 2026" in src
    assert "FREEZE_SEASON" not in src.split('"""', 2)[2]


def _entrants(gap_carry, gap_mean, n=30, seed=2):
    """Entrants whose outcome sits near S0, with the two baselines further away."""
    rng = np.random.default_rng(seed)
    s0 = rng.uniform(40, 80, n)
    y = s0 + rng.normal(0, 2, n)
    return pd.DataFrame({"player_id": [f"p{i}" for i in range(n)], "S0": s0,
                         "class_target": y, "class_source": y + gap_carry,
                         "B0": y + gap_mean})


def test_h1b_is_tested_only_after_h1a_is_confirmed():
    both = f7.primary(_entrants(gap_carry=8.0, gap_mean=5.0))
    assert both["H1a"]["label"] == f7.LABELS[0]
    assert both["H1b"]["label"] == f7.LABELS[0]
    gate = f7.primary(_entrants(gap_carry=0.0, gap_mean=5.0))
    assert gate["H1a"]["label"] != f7.LABELS[0]
    assert gate["H1b"]["label"].startswith("NOT TESTED (H1a not confirmed)")
