# -*- coding: utf-8 -*-
"""Merge player identifiers that Stats Perform re-issued to the same player.

Found 2026-10-28, before the ratings were rolled forward to 2026. Each pair below is
one player under two identifiers: the same name, the same date of birth, and never on
a team sheet in the same match. Split like this, his Class is built from part of his
career, a move between competitions can disappear, and he can appear twice in a list.

Only that strict rule merges. Pairs with a date of birth missing on one side, or a
different spelling of the name, are left as they are and listed for the data provider
(see MERGE_CANDIDATES_FOR_MIKE below). The two `syn-` identifiers made by
`resolve_duplicates.py` are the provider's own ruling (two different players) and are
never merged, although they share a name and a date of birth.

The identifier kept is the one used in the player's latest season, since later feeds
will carry it; the moved rows take the kept identifier's name. Every move is recorded
in `player_id_map` (match_basis `reissued_same_name_dob`).

Source tables only: player_match_stats, player_match_raw, sl_teamsheets and
excluded_rows, then the `players` registry is rebuilt. The derived tables (ratings,
pairs, cohort, decision tables) are rebuilt by their own writers afterwards. The sealed
v2 translation tables are not touched.

Idempotent: re-running finds nothing to move.

    python merge_reissued_ids.py --dry-run
    python merge_reissued_ids.py
"""
import argparse
import os
import sqlite3
import sys

import pandas as pd

import runtime
import weekly_update as wu

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")

# (identifier that goes, identifier that stays)
MERGES = [
    ("24521", "67889"),   # Benji Kot, 1997-10-02, PNG Hunters 2021-23 / 2024-26
    ("22453", "51422"),   # Cameron Scott, 1999-10-07, Hull 2021-22 / Hull, Wakefield 2023-26
    ("22560", "52396"),   # Jack Brown, 2000-06-25, Hull 2021-23 / 2023-26
    ("22849", "64775"),   # Josh Simm, 2001-02-27, SL 2021-22 + 1 QLD game / QLD 2023, SL 2024-26
    ("67632", "23402"),   # Riley Dean, 2001-08-10, Mackay 2024 (6) / QLD + SL 2021-26
    ("22969", "64757"),   # Robson Stevens, 2002-04-04, Huddersfield 2021 / Mackay 2023
    ("63312", "67876"),   # Rodney Tuipulotu-Vea, 2005-05-13, Warriors NSW 2025 / 2024-26
    ("64639", "21503"),   # Tom Carr, 1995-05-12, Souths NSW 2023 (7) / 2021-23
]

# Probably the same player, but the evidence is not conclusive here: the provider decides.
MERGE_CANDIDATES_FOR_MIKE = [
    ("70057", "72468", "Floyd Aubrey: Souths Logan 2025 and 2026; no date of birth on "
                       "70057, listed as a winger in 2025 and a half in 2026"),
    ("67489", "55028", "Daniel O'Donnell: NSW Cup 2024-25 (Blacktown, Souths) and 2026 "
                       "(Bulldogs); no date of birth on 67489"),
    ("59678", "67862", "Chris / Christopher Slater-Raptis: same date of birth, NSW Cup 2023 "
                       "and 2024-25"),
    ("24524", "55673", "Pele / Pele Junior Vaigalu: same date of birth, Souths NSW Cup 2021"),
    ("62900", "22282", "Billy Magoulias: one 2022 Newtown game without a date of birth"),
    ("25757", "24369", "Joshua Coric: one 2022 St George game without a date of birth "
                       "(also 62643 / 62926 'Joshua CORIC', one or two games each)"),
]

SOURCE_TABLES = ("player_match_stats", "player_match_raw", "sl_teamsheets",
                 "excluded_rows")


def check(con, old, keep):
    """The rule, re-verified on the data before anything moves."""
    p = pd.read_sql("SELECT player_id, name, dob FROM players WHERE player_id IN (?, ?)",
                    con, params=(old, keep))
    if len(p) != 2:
        return "already merged" if (p.player_id == keep).any() else "not found"
    names = p.name.str.lower().str.replace(r"[^a-z]", "", regex=True)
    dobs = wu.sp.parse_dob(p.dob)
    if names.nunique() != 1:
        return f"names differ: {list(p.name)}"
    if dobs.isna().any() or dobs.nunique() != 1:
        return f"dates of birth differ or are missing: {list(p.dob)}"
    shared = con.execute(
        "SELECT count(*) FROM player_match_stats a JOIN player_match_stats b "
        "ON a.competition = b.competition AND a.season = b.season AND a.round = b.round "
        "WHERE a.player_id = ? AND b.player_id = ?", (old, keep)).fetchone()[0]
    if shared:
        return f"both appear in {shared} of the same rounds"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    todo = []
    for old, keep in MERGES:
        why = check(con, old, keep)
        name = con.execute("SELECT name FROM players WHERE player_id = ?",
                           (keep,)).fetchone()
        if why == "already merged":
            print(f"  {old} -> {keep}: already merged")
        elif why:
            sys.exit(f"ABORT: {old} -> {keep}: {why}")
        else:
            n = con.execute("SELECT count(*) FROM player_match_stats WHERE player_id = ?",
                            (old,)).fetchone()[0]
            print(f"  {old} -> {keep} {name[0]}: {n} match rows")
            todo.append((old, keep, name[0]))
    con.close()
    if not todo or a.dry_run:
        print("nothing to move" if not todo else "--dry-run: nothing written")
        return

    with runtime.guarded_write("merge_reissued_ids", note=f"{len(todo)} merges"):
        con = sqlite3.connect(DB)
        for old, keep, name in todo:
            moved = pd.read_sql("SELECT DISTINCT competition, season FROM player_match_stats "
                                "WHERE player_id = ?", con, params=(old,))
            con.execute("UPDATE player_match_stats SET player_id = ?, player = ? "
                        "WHERE player_id = ?", (keep, name, old))
            con.execute('UPDATE player_match_raw SET player_id = ?, "Player ID" = ? '
                        "WHERE player_id = ?", (keep, keep, old))
            for t in SOURCE_TABLES[2:]:
                con.execute(f"UPDATE {t} SET player_id = ? WHERE player_id = ?", (keep, old))
            con.executemany(
                "INSERT INTO player_id_map (old_player_id, new_player_id, player, "
                "competition, season, match_basis) VALUES (?, ?, ?, ?, ?, ?)",
                [(old, keep, name, r.competition, int(r.season), "reissued_same_name_dob")
                 for r in moved.itertuples()])
        con.commit()
        players = wu.refresh_registry(con)
        dup = con.execute(
            "SELECT count(*) FROM (SELECT player_id, competition, season, round, team, "
            "count(*) k FROM player_match_stats GROUP BY 1, 2, 3, 4, 5 HAVING k > 1)"
        ).fetchone()[0]
        con.close()
        if dup:
            raise RuntimeError(f"{dup} player-match keys now repeat")
    print(f"merged {len(todo)}; players registry {len(players):,}")
    print("\nleft for the data provider:")
    for old, keep, why in MERGE_CANDIDATES_FOR_MIKE:
        print(f"  {old} / {keep}  {why}")
    print("\nrun regenerate_full.py next")


if __name__ == "__main__":
    main()
