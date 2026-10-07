# -*- coding: utf-8 -*-
"""The deployed copy of the database: what it leaves out and how a stale one is caught.

Synthetic databases in a temporary folder; tallec.db is never opened.
"""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_app_db as bad


@pytest.fixture
def dbs(tmp_path, monkeypatch):
    full, app = tmp_path / "tallec.db", tmp_path / "tallec_app.db"
    con = sqlite3.connect(full)
    con.execute("CREATE TABLE player_match_stats (player_id TEXT, season INT, tries INT)")
    con.executemany("INSERT INTO player_match_stats VALUES (?,?,?)",
                    [("1", 2025, 0), ("2", 2025, 2), ("3", 2026, 1)])
    con.execute("CREATE TABLE player_match_raw (player_id TEXT, blob TEXT)")
    con.execute("INSERT INTO player_match_raw VALUES ('1', 'x')")
    con.execute("CREATE INDEX ix_raw ON player_match_raw(player_id)")
    con.commit()
    con.close()
    monkeypatch.setattr(bad, "DB", str(full))
    monkeypatch.setattr(bad, "APP_DB", str(app))
    return full, app


def test_the_copy_leaves_out_the_raw_table_and_nothing_else(dbs):
    full, app = dbs
    bad.build()
    tables = {r[0] for r in sqlite3.connect(app).execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"player_match_stats"}
    assert bad.differences() == []


def test_a_missing_copy_is_reported(dbs):
    assert bad.differences() == ["tallec_app.db does not exist"]


def test_a_copy_left_behind_after_an_update_is_caught(dbs):
    full, app = dbs
    bad.build()
    con = sqlite3.connect(full)
    con.execute("UPDATE player_match_stats SET tries = 3 WHERE player_id = '2'")
    con.commit()
    con.close()
    assert bad.differences() == ["player_match_stats: 1 rows differ"]


def test_a_new_table_in_the_full_database_is_caught(dbs):
    full, app = dbs
    bad.build()
    con = sqlite3.connect(full)
    con.execute("CREATE TABLE player_ratings (player_id TEXT)")
    con.commit()
    con.close()
    assert bad.differences() == ["missing table player_ratings"]


def test_building_never_writes_to_the_full_database(dbs):
    full, app = dbs
    before = full.read_bytes()
    bad.build()
    assert full.read_bytes() == before
