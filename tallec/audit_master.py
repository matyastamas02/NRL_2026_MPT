# -*- coding: utf-8 -*-
"""Audit an xLadder match master against the TALLEC player data.

The player database is an independent record of the same matches, so it can answer
questions the master cannot answer about itself. Every club's scoring events sit in
`player_match_raw`; summing them per team per match reconstructs the scoreline without
consulting any column the master derives its own outcomes from.

Three things are checked, because three different things can be wrong and only the
third is visible in the numbers the app finally displays:

  fixtures  Does each master fixture exist in the player data, and in the round the
            master files it under? A pairing that exists in the season but not in that
            round means the `Round` column is wrong rather than the fixture. Round order
            is what Form rolls over and what ELO walks through, so a wrong round is a
            wrong chronology even when every other value on the row is right.

  scores    Do the master's two score columns agree with the reconstruction? Compared
            per club per season, so the answer does not depend on the round labels being
            right — which is the point, since they are one of the things being tested.

  alignment Do the master's margins look like the real ones? Two scores can each be
            correct and still be paired with the wrong opponent, which no per-column
            check can see. A season whose margins are a third of the size of reality has
            had its rows misaligned, whatever the column totals say.

A season still being played has rounds in the player feed that the master has not been
given yet. Counting those against it would report corruption where there is only a
season in progress, so the comparison is restricted to the fixtures the master actually
contains — matched by club pairing, not by round, for the reason above.

Read-only. Run it after any bulk rebuild of a master, the way position files are checked
before they are trusted.

    python audit_master.py            # both leagues
    python audit_master.py SL         # one
"""
import sqlite3
import sys
from collections import Counter

import pandas as pd

import team_map as tm

# Points are rebuilt from the scoring events rather than read from the aggregate
# column: `Points Scored` is known to undercount the real score by 2-4 points in about
# a tenth of matches, which is the same order as the discrepancies being looked for.
POINTS = """
SELECT s.season, s.round, s.team, s.opposition,
       SUM(4 * COALESCE(r."Try Scored - Total", 0)
         + 2 * COALESCE(r."Conversion - Made", 0)
         + 2 * COALESCE(r."Penalty Goal - Made", 0)
         + 1 * COALESCE(r."Field Goal - 1 Point Made", 0)
         + 2 * COALESCE(r."Field Goal - 2 Point Made", 0)) AS pts
FROM player_match_stats s
JOIN player_match_raw r ON r.player_id = s.player_id
 AND r.Competition = s.competition AND r.Season = s.season AND r."Round" = s.round
WHERE s.competition = ? GROUP BY 1, 2, 3, 4
"""

SCORE_COLS = ["Season", "Round", "A Team", "B Team", "A Score", "B Score",
              "A_Points Scored", "B_Points Scored"]

# how far a master may drift before it is called wrong: 3% of a season's points, and a
# mean margin within a sixth of the real one
SCORE_TOL = 0.03
MARGIN_TOL = (0.85, 1.18)


def player_fixtures(comp, db="tallec.db"):
    """One row per fixture in the player data: season, round, both clubs, both scores.

    A fixture is known from either side — each team row names its opposition — but it
    can only be *scored* when both sides' player rows are present. Both facts are
    returned, because they answer different questions: whether the master invented a
    fixture is asked of every fixture, whether its score is right only of the scorable
    ones. Conflating them reports a master as wrong for a gap in the player feed.
    """
    con = sqlite3.connect(db)
    p = pd.read_sql(POINTS, con, params=(comp,))
    con.close()
    p["round"] = pd.to_numeric(p["round"], errors="coerce")
    p = p.dropna(subset=["round"])
    pts = {(int(r.season), int(r["round"]), r.team): r.pts for _, r in p.iterrows()}
    seen, rows = set(), []
    for _, r in p.iterrows():
        season, rnd = int(r.season), int(r["round"])
        pair = tuple(sorted((r.team, r.opposition)))
        if (season, rnd, pair) in seen:
            continue
        seen.add((season, rnd, pair))
        a = pts.get((season, rnd, pair[0]))
        b = pts.get((season, rnd, pair[1]))
        rows.append(dict(season=season, round=rnd, pair=pair,
                         home=pair[0], away=pair[1], pts_home=a, pts_away=b,
                         scorable=a is not None and b is not None))
    return pd.DataFrame(rows)


def _master_pairs(m, mp):
    m = m.copy()
    m["pair"] = [tuple(sorted((mp.get(a, a), mp.get(b, b))))
                 for a, b in zip(m["A Team"], m["B Team"])]
    return m


def _matched(fx, m):
    """Pair each master row with the player-data fixture it describes.

    Clubs meet two or three times a season, so a pairing is not a key on its own; each
    master row claims one occurrence of its pairing, taken in round order. Rows that
    find no partner — a fixture the player feed has not got, or one side of it missing —
    are left out of both sides of the comparison, so a gap in the feed is never counted
    against the master.

    Returns (master rows, player fixtures) as two frames in the same order.
    """
    fx = fx[fx.scorable]
    mi, pi = [], []
    for season, g in m.groupby("Season"):
        pool = fx[fx.season == int(season)].sort_values("round")
        want = Counter(g["pair"])
        taken = {}
        for idx, r in pool.iterrows():
            if want.get(r["pair"], 0) > 0:
                want[r["pair"]] -= 1
                taken.setdefault(r["pair"], []).append(idx)
        for idx, r in g.iterrows():
            q = taken.get(r["pair"])
            if q:
                mi.append(idx)
                pi.append(q.pop(0))
    return m.loc[mi], fx.loc[pi]


def audit(comp, db="tallec.db", master=None):
    mp, _ = tm.solve(comp)
    if master:
        # the club mapping is still solved from the master the project ships, so a
        # candidate file is judged against the same yardstick as the original
        tm.MASTERS[comp] = master
    m = _master_pairs(tm.load_master(comp, SCORE_COLS), mp)
    fx = player_fixtures(comp, db)
    mm_, tt_ = _matched(fx, m)

    in_round = {(s, r): set(g["pair"]) for (s, r), g in fx.groupby(["season", "round"])}
    in_season = {s: set(g["pair"]) for s, g in fx.groupby("season")}

    print(f"\n{'=' * 78}\n{comp}: {len(m)} fixtures in the master, {len(fx)} in the "
          f"player data, {len(mm_)} matched and scorable\n{'=' * 78}")

    print(f"\nfixtures\n{'season':>8}{'n':>6}{'in that round':>16}{'in the season':>16}"
          f"{'verdict':>20}")
    for s, g in m.groupby("Season"):
        s, n = int(s), len(g)
        if s not in in_season:
            print(f"{s:>8}{n:>6}{'-':>16}{'-':>16}{'no player data':>20}")
            continue
        same = sum(p in in_round.get((s, int(r)), ())
                   for p, r in zip(g["pair"], g["Round"]))
        any_ = sum(p in in_season[s] for p in g["pair"])
        verdict = ("ok" if same == n else
                   "ROUND LABELS WRONG" if any_ / n > 0.95 else "FIXTURES DIFFER")
        print(f"{s:>8}{n:>6}{same / n:>15.1%}{any_ / n:>16.1%}{verdict:>20}")

    # Club totals only mean something over a complete set of a club's matches. Where
    # the feed is missing fixtures the master has, the two sides are summing different
    # games and the difference says nothing about either — so coverage is printed and
    # a season that is not almost fully matched gets no verdict.
    matched_n = mm_.groupby("Season").size().to_dict()
    total_n = m.groupby("Season").size().to_dict()

    print(f"\nscores, per club per season (matched on fixture, not on round)\n"
          f"{'season':>8}{'clubs':>7}{'matched':>10}{'A Score err':>14}"
          f"{'Pts Scored err':>16}{'verdict':>22}")
    master_long = pd.concat([
        mm_.rename(columns={"A Team": "code", "A Score": "sc",
                            "A_Points Scored": "ps"})[["Season", "code", "sc", "ps"]],
        mm_.rename(columns={"B Team": "code", "B Score": "sc",
                            "B_Points Scored": "ps"})[["Season", "code", "sc", "ps"]]])
    master_long["team"] = master_long.code.map(mp)
    agg = master_long.groupby(["Season", "team"], as_index=False)[["sc", "ps"]].sum()
    truth = pd.concat([
        tt_.rename(columns={"home": "team", "pts_home": "pts"})[["season", "team", "pts"]],
        tt_.rename(columns={"away": "team", "pts_away": "pts"})[["season", "team", "pts"]],
    ]).groupby(["season", "team"], as_index=False).pts.sum()
    j = agg.merge(truth, left_on=["Season", "team"], right_on=["season", "team"])
    for s, g in j.groupby("Season"):
        es = (g.sc - g.pts).abs().sum() / g.pts.sum()
        ep = (g.ps - g.pts).abs().sum() / g.pts.sum()
        cov = matched_n.get(s, 0) / total_n[s]
        # the pipeline derives its outcomes from Points Scored, so that is the column
        # that has to be right; A Score only has to be right to be worth displaying
        verdict = ("not fully matched" if cov < 0.98 else
                   "ok" if ep < SCORE_TOL else "POINTS SCORED WRONG")
        print(f"{int(s):>8}{len(g):>7}{cov:>9.0%}{es:>13.1%}{ep:>15.1%}{verdict:>22}")

    print(f"\nrow alignment, from the margins\n{'season':>8}{'n':>6}"
          f"{'master |margin|':>18}{'player data':>14}{'verdict':>22}")
    mmar = (mm_["A_Points Scored"] - mm_["B_Points Scored"]).abs()
    tmar = (tt_.pts_home - tt_.pts_away).abs()
    for s in sorted(mm_.Season.unique()):
        sel = (mm_.Season == s).values
        a, b = mmar[sel].dropna(), tmar[sel]
        if not len(a) or not len(b):
            continue
        ratio = a.mean() / b.mean()
        verdict = "ok" if MARGIN_TOL[0] <= ratio <= MARGIN_TOL[1] else "ROWS MISALIGNED"
        print(f"{int(s):>8}{len(a):>6}{a.mean():>10.1f}/{a.std():<7.1f}"
              f"{b.mean():>7.1f}/{b.std():<6.1f}{verdict:>22}")


if __name__ == "__main__":
    args = sys.argv[1:]
    master = None
    if "--file" in args:
        i = args.index("--file")
        master = args[i + 1]
        args = args[:i] + args[i + 2:]
    for c in (args or ["NRL", "SL"]):
        audit(c.upper(), master=master)
    print()
