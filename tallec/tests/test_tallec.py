# -*- coding: utf-8 -*-
"""Tests for the parts of TALLEC that have actually broken.

Every test here corresponds to a bug that shipped at least once. They run against
synthetic data and never touch tallec.db, so they are safe to run any time:

    python -m pytest tests -q

The suite is deliberately about behaviour that is easy to get wrong and hard to
notice — identity keys, the position rule, the coverage gate, leakage, the rollback —
rather than about coverage percentages.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gigot_contribution as gc          # noqa: E402
import player_rating_engine as pre       # noqa: E402
import sp_schema as sp                   # noqa: E402


# ── identity ────────────────────────────────────────────────────────────────
def test_player_id_survives_a_float_column():
    """A blank row makes pandas read the id column as float, and "24528.0" does not
    join to "24528". This forked every player into a new identity once."""
    s = pd.Series([24528, None, 21722])
    out = sp.normalize_player_id(s)
    assert out.iloc[0] == "24528"
    assert out.iloc[2] == "21722"
    assert pd.isna(out.iloc[1]) or out.iloc[1] is None


def test_player_id_keeps_non_numeric_ids():
    out = sp.normalize_player_id(pd.Series(["aj-brimson", 24528]))
    assert out.iloc[0] == "aj-brimson"
    assert out.iloc[1] == "24528"


def test_clean_name_strips_the_asterisks():
    assert sp.clean_name("*Cameron *Scott") == "Cameron Scott"


# ── the position rule ───────────────────────────────────────────────────────
def test_interchange_is_a_role_not_a_position():
    """A prop who mostly comes off the bench is still a prop. Getting this wrong
    moved five bench forwards to 'Interchange' on the first weekly import."""
    s = pd.Series(["Interchange"] * 27 + ["Prop"] * 7 + ["Second Row"] * 2)
    assert sp.primary_position(s) == "Prop"


def test_interchange_wins_only_when_he_never_starts():
    assert sp.primary_position(pd.Series(["Interchange"] * 5)) == "Interchange"


def test_position_groups_merge_the_way_the_engine_rates():
    assert sp.POSITION_GROUP["Second Row"] == sp.POSITION_GROUP["Lock"] == "Back Row"
    assert sp.POSITION_GROUP["Half Back"] == sp.POSITION_GROUP["Five-Eighth"] == "Halves"


# ── synthetic match data ────────────────────────────────────────────────────
def _matches(n_players=40, n_rounds=10, seed=0, position="Prop", known=True):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        skill = rng.normal(0, 1)
        for r in range(1, n_rounds + 1):
            rows.append(dict(
                player_id=f"p{p}", player=f"Player {p}", season=2026, round=r,
                team=f"T{p % 4}", minutes=float(rng.integers(40, 81)),
                position=position if known else "Unknown",
                all_run_metres=80 + 12 * skill + rng.normal(0, 8),
                p_c_m=30 + 5 * skill + rng.normal(0, 4),
                tackle_breaks=max(0, 2 + skill + rng.normal(0, 1)),
                line_breaks=max(0, rng.normal(0.4, 0.4)),
                tackles=20 + 3 * skill + rng.normal(0, 4),
                offloads=max(0, rng.normal(1, 0.8)),
                try_assists=max(0, rng.normal(0.2, 0.3)),
                tries=max(0, rng.normal(0.2, 0.4)),
                errors=max(0, rng.normal(1, 0.7))))
    return pd.DataFrame(rows)


def test_coverage_gate_pools_the_competition_when_position_is_patchy():
    """Below the threshold the engine must NOT split by position: a partly known pool
    makes two identical performances score differently."""
    df = _matches(known=True)
    df.loc[df.index[: int(len(df) * 0.7)], "position"] = "Unknown"
    eng = pre.PlayerRatingEngine("TEST")
    eng._composite(df)
    assert eng.position_mode == "competition_relative"
    assert eng.position_coverage < pre.MIN_POS_COVERAGE


def test_coverage_gate_uses_position_when_it_is_known():
    eng = pre.PlayerRatingEngine("TEST")
    eng._composite(_matches(known=True))
    assert eng.position_mode == "position_relative"


def test_force_mode_overrides_a_fully_known_pool():
    eng = pre.PlayerRatingEngine("TEST", force_mode="competition_relative")
    eng._composite(_matches(known=True))
    assert eng.position_mode == "competition_relative"


# ── shrinkage ───────────────────────────────────────────────────────────────
def test_shrinkage_grows_with_evidence():
    """B must rise monotonically with matches played, or a one-game player can top a
    sorted table."""
    df = _matches(n_rounds=12)
    eng = pre.PlayerRatingEngine("TEST")
    pm = eng._composite(df)
    eng._fit_variance_components(pm)
    b = [eng._shrink(0.5, n)[1] for n in (1, 3, 10, 30)]
    assert b == sorted(b)
    assert b[0] < 0.5 < b[-1]


def test_one_match_is_pulled_most_of_the_way_to_average():
    df = _matches(n_rounds=12)
    eng = pre.PlayerRatingEngine("TEST")
    pm = eng._composite(df)
    eng._fit_variance_components(pm)
    z, b = eng._shrink(2.0, 1)
    assert b < 0.5
    assert abs(z - eng.grand_mean) < abs(2.0 - eng.grand_mean)


# ── leakage ─────────────────────────────────────────────────────────────────
def test_a_match_cannot_influence_its_own_prematch_rating():
    """The real leakage claim: with the standardization held fixed, a player's
    pre-match rating is built from his EARLIER matches only, so altering a match
    cannot change the rating that match is predicted with."""
    df = _matches(n_rounds=8).sort_values(["player_id", "round"]).reset_index(drop=True)
    eng = pre.PlayerRatingEngine("TEST")
    eng._fit_standardization(_matches(n_rounds=8, seed=99))    # a separate pool

    def prior(frame):
        pm = eng._transform(frame).sort_values(["player_id", "round"])
        return pm.groupby("player_id")["composite"].transform(
            lambda s: s.shift(1).expanding().mean()).reset_index(drop=True)

    base = prior(df)
    tampered = df.copy()
    last = tampered.groupby("player_id").tail(1).index
    tampered.loc[last, "all_run_metres"] *= 5
    assert np.allclose(base.fillna(-9), prior(tampered).fillna(-9), atol=1e-9)


def test_fitting_the_standardization_on_the_whole_pool_is_not_leak_free():
    """A known property, asserted so nobody mistakes it for the strict version.

    _composite fits the pool statistics on the frame it is given, so if that frame
    includes the match being predicted, altering that match moves every other row's
    z-score. It is a population descriptor rather than an outcome, and small, but it
    is not a walk-forward. gigot_v2.prematch_players(walk_forward=True) is.
    """
    df = _matches(n_rounds=8).sort_values(["player_id", "round"]).reset_index(drop=True)

    def composites(frame):
        return pre.PlayerRatingEngine("TEST")._composite(frame)["composite"].values

    base = composites(df)
    tampered = df.copy()
    tampered.loc[tampered.index[-1], "all_run_metres"] *= 20
    moved = ~np.isclose(base, composites(tampered), atol=1e-9)
    assert moved.sum() > 1, ("altering one match should move other rows under "
                             "whole-pool fitting; if this fails the engine changed "
                             "and the walk-forward caveat can be dropped")


# ── contribution ────────────────────────────────────────────────────────────
def test_contribution_shares_sum_to_one_within_a_team_match():
    df = _matches(n_players=34, n_rounds=3)
    df["opposition"] = "X"
    out = gc.compute_contribution(df, "TEST")
    tot = out.groupby(["season", "round", "team"])["attack_share"].sum()
    assert np.allclose(tot.values, 1.0, atol=1e-6)


def test_contribution_median_sits_at_fifty():
    df = _matches(n_players=34, n_rounds=4)
    df["opposition"] = "X"
    out = gc.compute_contribution(df, "TEST")
    assert 45 < out["contribution_rating"].median() < 55


def test_contribution_is_scaled_within_the_competition_passed_in():
    df = _matches(n_players=20, n_rounds=3)
    df["opposition"] = "X"
    out = gc.compute_contribution(df, "ONLY_ME")
    assert set(out["competition"]) == {"ONLY_ME"}


# ── the 0-100 scale and the translation guardrails ──────────────────────────
def test_score_to_z_round_trips():
    import predict_translation as pt
    for score in (10, 25, 50, 73.9, 90):
        assert abs(pt.z_to_score(pt.score_to_z(score)) - score) < 1e-6


def test_fifty_is_the_middle_of_the_scale():
    import predict_translation as pt
    assert abs(pt.score_to_z(50)) < 1e-9


def test_translation_conditions_on_the_player():
    """This is the bug the review found: the model was loaded but never used, so two
    very different players got the identical answer."""
    import predict_translation as pt
    a = pt.translate(70, "NRL", "SL", position="Prop", age=20, minutes_pg=25, games=4)
    b = pt.translate(70, "NRL", "SL", position="Full Back", age=31, minutes_pg=78, games=24)
    assert a["score_model"] is not None
    assert a["score_model"] != b["score_model"]
    # the ladder half is a property of the pair, so it must NOT move with the player
    assert a["score_ladder"] == b["score_ladder"]


def test_translation_refuses_to_extrapolate_silently():
    import predict_translation as pt
    r = pt.translate(70, "NRL", "SL", position="Prop", age=26, minutes_pg=60, games=150)
    assert r["inputs_used"]["clamped"], "an out-of-range season game count must be flagged"


def test_translating_to_the_same_competition_is_a_no_op():
    import predict_translation as pt
    r = pt.translate(63.5, "SL", "SL")
    assert r["score_target"] == 63.5 and r["shift_points"] == 0.0


# ── import validation ───────────────────────────────────────────────────────
def _feed(rows):
    return pd.DataFrame(rows)


def test_validation_catches_broken_scoring():
    import weekly_update as wu
    df = _feed([dict(**{"Full Name": "A", "Team": "T", "Round": 1, "Minutes": 80,
                        "Player ID": 1, "Points Scored": 6, "Try Scored - Total": 1,
                        "Conversion - Made": 0, "Penalty Goal - Made": 0,
                        "Field Goal - 1 Point Made": 0, "Field Goal - 2 Point Made": 0})])
    sev = dict((m[:20], s) for s, m in wu.validate(df, "t"))
    assert any(s == "error" for s in sev.values())


def test_validation_passes_correct_scoring():
    import weekly_update as wu
    df = _feed([dict(**{"Full Name": "A", "Team": "T", "Round": 1, "Minutes": 80,
                        "Player ID": 1, "Points Scored": 4, "Try Scored - Total": 1,
                        "Conversion - Made": 0, "Penalty Goal - Made": 0,
                        "Field Goal - 1 Point Made": 0, "Field Goal - 2 Point Made": 0})])
    assert not [m for s, m in wu.validate(df, "t") if s == "error"]


def test_one_player_at_two_clubs_in_a_round_is_a_warning_not_an_error():
    """Super League reschedules fixtures, so a mid-season transfer leaves the same
    round number recorded for two clubs. That is real data."""
    import weekly_update as wu
    base = dict(**{"Full Name": "A", "Minutes": 60, "Player ID": 7,
                   "Points Scored": 0, "Try Scored - Total": 0, "Conversion - Made": 0,
                   "Penalty Goal - Made": 0, "Field Goal - 1 Point Made": 0,
                   "Field Goal - 2 Point Made": 0})
    df = _feed([dict(base, Team="Leigh", Round=9),
                dict(base, Team="Huddersfield", Round=9)])
    out = wu.validate(df, "t")
    assert not [m for s, m in out if s == "error"]
    assert [m for s, m in out if s == "warn"]


def test_the_same_player_twice_for_one_club_is_an_error():
    import weekly_update as wu
    base = dict(**{"Full Name": "A", "Minutes": 80, "Player ID": 7, "Team": "Capras",
                   "Round": 20, "Points Scored": 0, "Try Scored - Total": 0,
                   "Conversion - Made": 0, "Penalty Goal - Made": 0,
                   "Field Goal - 1 Point Made": 0, "Field Goal - 2 Point Made": 0})
    out = wu.validate(_feed([base, dict(base)]), "t")
    assert [m for s, m in out if s == "error"]


# ── the rollback ────────────────────────────────────────────────────────────
def test_a_failed_write_is_rolled_back_and_recorded(tmp_path, monkeypatch):
    import runtime
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE player_match_stats (x INTEGER)")
    con.executemany("INSERT INTO player_match_stats VALUES (?)", [(i,) for i in range(10)])
    con.commit()
    con.close()

    monkeypatch.setattr(runtime, "DB", str(db))
    monkeypatch.setattr(runtime, "AUDIT_DB", str(tmp_path / "audit.db"))
    monkeypatch.setattr(runtime, "BACKUP_DIR", str(tmp_path / "b"))

    with pytest.raises(RuntimeError):
        with runtime.guarded_write("test"):
            c = sqlite3.connect(db)
            c.execute("DELETE FROM player_match_stats")
            c.commit()
            c.close()
            raise RuntimeError("boom")

    assert runtime.row_count() == 10, "the rows must come back"
    audit = pd.read_sql("SELECT status, rows_before, rows_after FROM data_imports",
                        runtime.audit_con())
    assert audit.status.iloc[0] == "rolled_back"
    assert audit.rows_before.iloc[0] == 10 and audit.rows_after.iloc[0] == 10


def test_the_audit_log_is_not_inside_the_database_it_audits():
    """It was, once — so a rollback erased the record of the failure it recovered from."""
    import runtime
    assert os.path.abspath(runtime.AUDIT_DB) != os.path.abspath(runtime.DB)
