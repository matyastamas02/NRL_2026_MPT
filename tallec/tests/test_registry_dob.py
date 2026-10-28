# -*- coding: utf-8 -*-
"""The registry's date of birth: one ISO date per player, placeholders treated as missing.

Synthetic database in a temporary folder; tallec.db is never opened.
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import weekly_update as wu


def _db(tmp_path, registry, feed):
    con = sqlite3.connect(tmp_path / "t.db")
    con.execute("CREATE TABLE players (player_id TEXT, dob TEXT)")
    con.executemany("INSERT INTO players VALUES (?, ?)", registry)
    con.execute('CREATE TABLE player_match_raw (player_id TEXT, "Date of Birth" TEXT)')
    con.executemany("INSERT INTO player_match_raw VALUES (?, ?)", feed)
    con.commit()
    return con


def test_a_placeholder_gives_way_to_the_feeds_real_date(tmp_path):
    con = _db(tmp_path, [("1", "0000-00-00")], [("1", "0000-00-00"), ("1", "2005-06-14")])
    assert wu.registry_dob(con).set_index("player_id").dob.to_dict() == {"1": "2005-06-14"}


def test_a_date_in_the_registry_is_kept_against_a_different_feed_date(tmp_path):
    """Corrections are made explicitly, not by an import (Salesi Foketi's two dates)."""
    con = _db(tmp_path, [("1", "2005-11-04")], [("1", "2004-04-11")])
    assert wu.registry_dob(con).set_index("player_id").dob.to_dict() == {"1": "2005-11-04"}


def test_every_date_is_written_day_last_whatever_the_feed_used(tmp_path):
    con = _db(tmp_path, [("1", "07/10/1999")], [("2", "24/04/2006")])
    assert wu.registry_dob(con).set_index("player_id").dob.to_dict() == {
        "1": "1999-10-07", "2": "2006-04-24"}


def test_a_player_without_any_real_date_has_none(tmp_path):
    con = _db(tmp_path, [("1", "0000-00-00")], [("1", "0000-00-00")])
    assert wu.registry_dob(con).empty
