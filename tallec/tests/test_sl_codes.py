# -*- coding: utf-8 -*-
"""xladder_pipeline.fix_sl_team_codes: the 2022-2025 Super League rows relabelled to
SL_NAMES' codes, Match IDs kept unique, ELO recomputed by the pipeline's own rule."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
xp = pytest.importorskip("xladder_pipeline")


def _mini():
    rows = [  # season, round, id, A, B, score A, score B
        (2023, 1, "2023-1-C-LS", "C", "LS", 20, 10),
        (2023, 1, "2023-1-HFC-CF", "CF", "HFC", 12, 30),    # id lists B first
        (2023, 30, "2023-30-WI-C", "WI", "C", 10, 2),
        (2026, 1, "2026-1-C-WI", "C", "WI", 16, 26),
        (2026, 1, "2026-1-LH-LS", "LH", "LS", 22, 18),
    ]
    d = pd.DataFrame(rows, columns=["Season", "Round", "Match ID", "A Team", "B Team",
                                    "A_Points Scored", "B_Points Scored"])
    d["ELO_A"] = d["ELO_B"] = 2000.0
    d["Diff ELO"] = 0.0
    return d


def test_old_seasons_are_relabelled_and_2026_left_alone():
    f = xp.fix_sl_team_codes(_mini())
    assert f.loc[2, ["A Team", "B Team"]].tolist() == ["WI", "CF"]
    assert f.loc[0, ["A Team", "B Team"]].tolist() == ["CF", "LH"]
    assert f.loc[3, ["A Team", "B Team"]].tolist() == ["C", "WI"]
    assert f.loc[4, ["A Team", "B Team"]].tolist() == ["LH", "LS"]


def test_match_ids_swap_in_place_and_stay_unique():
    f = xp.fix_sl_team_codes(_mini())
    assert f["Match ID"].tolist()[:3] == ["2023-1-CF-LH", "2023-1-HF-C", "2023-30-WI-CF"]
    assert f["Match ID"].is_unique


def test_the_fix_is_idempotent_and_detects_itself():
    f = xp.fix_sl_team_codes(_mini())
    assert not xp.sl_old_coding(f)
    assert xp.fix_sl_team_codes(f).equals(f)


def test_detection_survives_string_season_and_round():
    d = _mini().astype({"Season": str, "Round": str})
    assert xp.sl_old_coding(d)


def test_elo_carries_over_by_club_after_the_fix():
    f = xp.fix_sl_team_codes(_mini())
    # Catalans (old C, now CF) lost the 2023 final 10-2 to Wigan; in 2026 C is
    # Castleford, a club without a rated match in this frame, so it starts near 2000
    cf_after_2023 = f.loc[2, "ELO_B"]
    assert cf_after_2023 != 2000.0
    assert f.loc[3, "ELO_A"] == pytest.approx(2000.0 + (f.loc[0, "ELO_A"] + 27 * 0.5
                                                        - 2000.0) * 0.7, abs=30)


MASTER = os.path.join(ROOT, "SL_master.xlsx")


@pytest.mark.skipif(not os.path.exists(MASTER), reason="no SL master")
def test_on_the_master_2022_2025_elo_is_unchanged():
    raw = pd.read_excel(MASTER)
    f = xp.fix_sl_team_codes(raw)
    old = pd.to_numeric(raw.Season) <= 2025
    assert np.allclose(raw.loc[old, "ELO_A"], f.loc[old, "ELO_A"])
    assert np.allclose(raw.loc[old, "ELO_B"], f.loc[old, "ELO_B"])
    assert f["Match ID"].is_unique
    assert not xp.sl_old_coding(f)
