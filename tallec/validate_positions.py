# -*- coding: utf-8 -*-
"""How good is a position we did not read off a match sheet?

The 2026 NRL rows carry an estimated position: the player's career position taken from
somewhere else, because no NRL 2026 match sheet has been supplied. The open request to
Mike is to close that gap. Before asking, it is worth knowing what the gap costs — an
estimate that is right nine times in ten is a different problem from one that is right
six times in ten, and only one of them is worth anyone's afternoon.

The measurement is possible because three competitions do carry match-sheet positions,
so the estimate can be made where the answer is already known. For each player the
estimate is built from his match-sheet rows in *other* competitions and compared with
what the target competition's own sheets say — the same construction the ingest uses,
with the target held out, so nothing being predicted contributes to its own prediction.

Two accuracies are reported and they answer different questions:

  position   the exact Stats Perform label — Prop, Second Row, Lock, and so on.
  group      the peer group the rating engine actually benchmarks within. Second Row
             and Lock both land in Back Row, so confusing them costs nothing; calling a
             winger a prop costs a great deal. This is the number that matters.

Read-only.

    python validate_positions.py
"""
import sqlite3

import pandas as pd

import sp_schema as sp

DB = "tallec.db"
MIN_ROWS = 3          # a career of one appearance is not a career


def load():
    con = sqlite3.connect(DB)
    d = pd.read_sql("SELECT player_id, competition, season, position, position_source "
                    "FROM player_match_stats WHERE position IS NOT NULL "
                    "AND position <> 'Unknown'", con)
    con.close()
    return d


def career(rows):
    """Career position per player, over match-sheet rows only."""
    m = rows[rows.position_source == "match"]
    keep = m.groupby("player_id").size()
    keep = keep[keep >= MIN_ROWS].index
    m = m[m.player_id.isin(keep)]
    return m.groupby("player_id")["position"].agg(sp.primary_position)


def main():
    d = load()
    truth_pool = d[d.position_source == "match"]
    print("match-sheet rows available as ground truth:")
    t = truth_pool.groupby(["competition", "season"]).size().unstack(fill_value=0)
    print(t.to_string())

    print(f"\n\nholding out one competition at a time, estimating from the others\n"
          f"{'target':>26}{'players':>9}{'covered':>9}{'position':>11}{'group':>9}")
    rows = []
    for comp in sorted(truth_pool.competition.unique()):
        target = career(truth_pool[truth_pool.competition == comp])
        source = career(truth_pool[truth_pool.competition != comp])
        both = target.index.intersection(source.index)
        if not len(both):
            print(f"{comp:>26}{len(target):>9}{0:>9}{'-':>11}{'-':>9}")
            continue
        exact = (target[both] == source[both]).mean()
        grp = (target[both].map(sp.POSITION_GROUP)
               == source[both].map(sp.POSITION_GROUP)).mean()
        print(f"{comp:>26}{len(target):>9}{len(both) / len(target):>8.0%}"
              f"{exact:>11.1%}{grp:>9.1%}")
        rows.append((comp, target, source, both))

    # The 2026 NRL case specifically: an Australian career, used to guess a top-grade
    # position. NRL 2020 is the only NRL season with its own sheets, so it is the only
    # place that exact question can be asked and answered.
    print("\n\nthe case that matters — an Australian second-tier career used to guess "
          "a top-grade position")
    nrl = career(truth_pool[truth_pool.competition == "NRL"])
    aus = career(truth_pool[truth_pool.competition.isin(["NSW", "QLD"])])
    both = nrl.index.intersection(aus.index)
    if len(both):
        exact = (nrl[both] == aus[both]).mean()
        grp = (nrl[both].map(sp.POSITION_GROUP) == aus[both].map(sp.POSITION_GROUP)).mean()
        print(f"  {len(both)} players appear in both, with match sheets on each side")
        print(f"  exact position agrees   {exact:.1%}")
        print(f"  peer group agrees       {grp:.1%}")
        cm = pd.crosstab(nrl[both].map(sp.POSITION_GROUP).rename("NRL sheet"),
                         aus[both].map(sp.POSITION_GROUP).rename("estimated"))
        print("\n  where it goes wrong (rows: truth, columns: estimate)")
        print(cm.to_string())
    else:
        print("  no player has match sheets on both sides")

    # And the population the estimate is actually applied to, so the accuracy above can
    # be weighted by how much of 2026 it really covers.
    con = sqlite3.connect(DB)
    cov = pd.read_sql("SELECT position_source, COUNT(*) n FROM player_match_stats "
                      "WHERE competition='NRL' AND season=2026 GROUP BY 1", con)
    con.close()
    print("\n\nwhat NRL 2026 is actually made of:")
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
