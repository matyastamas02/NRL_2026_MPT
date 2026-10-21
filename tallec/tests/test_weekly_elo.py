# -*- coding: utf-8 -*-
"""xladder_pipeline.get_current_elos: the weekly path (append_new_round) continues from
each club's POST-match rating. ELO_A/ELO_B store PRE-match ratings, so the latest
result has to be applied on top of them; returning the stored value froze ELO."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
xp = pytest.importorskip("xladder_pipeline")


def _rounds():
    rows = [  # season, round, id, A, B, score A, score B
        (2025, 1, "2025-1-X-Y", "X", "Y", 30, 6),
        (2025, 2, "2025-2-Y-Z", "Y", "Z", 12, 18),
        (2025, 3, "2025-3-Z-X", "Z", "X", 26, 4),
        (2026, 1, "2026-1-X-Z", "X", "Z", 10, 24),
        (2026, 2, "2026-2-Y-X", "Y", "X", 22, 8),
        (2026, 3, "2026-3-Z-Y", "Z", "Y", 16, 14),
    ]
    return pd.DataFrame(rows, columns=["Season", "Round", "Match ID", "A Team", "B Team",
                                       "A_Points Scored", "B_Points Scored"])


def _master(rows):
    """Rows rated from scratch, pre-match ratings stored, as the master keeps them."""
    d, post = xp.update_elos_for_new_matches(rows, {}, int(rows["Season"].min()))
    return d.reset_index(drop=True), post


def test_current_elos_are_post_match():
    m, post = _master(_rounds())
    cur = xp.get_current_elos(m)
    assert cur == pytest.approx(post)
    # and not the stored pre-match value of each club's last row (the old behaviour)
    last = m.iloc[-1]
    assert cur[last["A Team"]] != pytest.approx(last["ELO_A"])


def test_weekly_submits_match_one_batch_and_a_recompute():
    r = _rounds()
    base, _ = _master(r[r.Season == 2025])
    weekly = base
    for i in r.index[r.Season == 2026]:
        weekly = xp.append_new_round(weekly, r.loc[[i]].copy(), "NRL")
    batch = xp.append_new_round(base, r[r.Season == 2026].copy(), "NRL")
    full, _ = _master(r)
    for c in ("ELO_A", "ELO_B", "Diff ELO"):
        assert np.allclose(weekly[c], batch[c])
        assert np.allclose(weekly[c], full[c])


def test_no_frozen_step_and_season_regression_applies_to_post_match():
    r = _rounds()
    base, post25 = _master(r[r.Season == 2025])
    m = xp.append_new_round(base, r[r.Season == 2026].iloc[:1].copy(), "NRL")
    new = m.iloc[-1]                                    # 2026-1 X v Z
    for team, col in (("X", "ELO_A"), ("Z", "ELO_B")):
        want = post25[team] + (xp.ELO_MEAN - post25[team]) * xp.ELO_REGRESSION
        assert new[col] == pytest.approx(want)
    # the old code started 2026 from Z's pre-match rating of 2025-3, which Z won 26-4
    z_pre = base.iloc[-1]["ELO_A"]
    assert new["ELO_B"] != pytest.approx(z_pre + (xp.ELO_MEAN - z_pre) * xp.ELO_REGRESSION)


def test_stored_rows_are_left_alone_and_new_rows_hold_pre_match():
    r = _rounds()
    base, _ = _master(r[r.Season == 2025])
    cur = xp.get_current_elos(base)
    m = xp.append_new_round(base, r[r.Season == 2026].copy(), "NRL")
    assert np.allclose(m.iloc[:3]["ELO_A"], base["ELO_A"])
    assert np.allclose(m.iloc[:3]["ELO_B"], base["ELO_B"])
    x_first = m[(m.Season == 2026) & (m["A Team"] == "X")].iloc[0]["ELO_A"]
    assert x_first == pytest.approx(cur["X"] + (xp.ELO_MEAN - cur["X"]) * xp.ELO_REGRESSION)


def test_unplayed_fixture_keeps_its_stored_rating():
    m, post = _master(_rounds())
    fx = pd.DataFrame([{"Season": 2026, "Round": 4, "Match ID": "2026-4-X-Y",
                        "A Team": "X", "B Team": "Y", "A_Points Scored": np.nan,
                        "B_Points Scored": np.nan, "ELO_A": post["X"], "ELO_B": post["Y"]}])
    cur = xp.get_current_elos(pd.concat([m, fx], ignore_index=True))
    assert cur["X"] == pytest.approx(post["X"])
    assert cur["Y"] == pytest.approx(post["Y"])


MASTER = os.path.join(ROOT, "NRL_master.xlsx")


@pytest.mark.skipif(not os.path.exists(MASTER), reason="no NRL master")
def test_on_the_master_the_stored_2026_chain_continues_from_post_match():
    # NRL 2026 R1-R17 were rated in one batch, so their stored chain is the correct one:
    # the rating of every R17 side must equal get_current_elos of the master up to R16
    raw = pd.read_excel(MASTER)
    s, rd = raw["Season"], raw["Round"]
    cur = xp.get_current_elos(raw[(s < 2026) | ((s == 2026) & (rd <= 16))])
    r17 = raw[(s == 2026) & (rd == 17)]
    assert np.allclose(r17["ELO_A"], r17["A Team"].map(cur))
    assert np.allclose(r17["ELO_B"], r17["B Team"].map(cur))
