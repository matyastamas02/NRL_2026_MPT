# -*- coding: utf-8 -*-
"""Write the entry cohort the comparison card shows, as table `entry_cohort`.

The card first read `translation_pairs_v3`, the pairs the translation is fitted on. The
seventh review showed those are not new entrants: of the 184 next-season pairs into
Super League, 131 players, some repeated, some dual-registered, 15 already playing in
Super League the season before. A club asking "who made this move before" needs the men
who actually entered the competition, so the card now reads the explicit entry cohort --
first and returning entries, the source fixed from the season before, built exactly as the
frozen 2026 test builds it (`frozen_2026.entry_cohort`) -- for every landing season up to
the freeze season. Only rated entrants are in it: three or more matches in the new
competition. Players who moved and barely played, or never played, are not.

    python build_entry_cohort.py            # dry run: counts only
    python build_entry_cohort.py --write    # replace the table, through runtime.guarded_write
"""
import argparse
import sqlite3
import sys

import frozen_2026 as fz
import rolling_backtest as rb
import runtime

COLS = ["player_id", "name", "source", "target", "pair", "transition_type",
        "season_src", "season_tgt", "class_source", "class_target", "n_target",
        "raw_position"]


def build():
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    c = fz.entry_cohort(con, range(fz.FIRST_LANDING, fz.FREEZE_SEASON + 1))
    con.close()
    return c[COLS].sort_values(["pair", "season_tgt", "player_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    c = build()
    print(f"{len(c)} entries, {c.player_id.nunique()} players, landing "
          f"{c.season_tgt.min()}-{c.season_tgt.max()}")
    print(c.groupby("pair").agg(entries=("player_id", "size"),
                                players=("player_id", "nunique")).to_string())
    if not a.write:
        return 0
    with runtime.guarded_write("entry_cohort",
                               note=f"landing {fz.FIRST_LANDING}-{fz.FREEZE_SEASON}"):
        con = sqlite3.connect(rb.DB)
        c.to_sql("entry_cohort", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_entry_pair ON entry_cohort(source, target)")
        con.commit()
        con.close()
    print("wrote entry_cohort")
    return 0


if __name__ == "__main__":
    sys.exit(main())
