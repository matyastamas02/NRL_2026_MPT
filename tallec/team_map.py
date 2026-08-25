# -*- coding: utf-8 -*-
"""Map xLadder match-master team identifiers to TALLEC player-data team names.

The NRL master uses full club names that already match the player data exactly. The
Super League master uses short codes (LS, WI, HKR ...), and the codes are not
documented reliably — the project notes say HF is Hull FC, but Huddersfield also
needs a code, so at least one entry would be a guess.

Rather than guess, the mapping is solved from the fixture list: a club plays in a
specific set of (season, round) slots, and that set is a near-unique fingerprint over
five seasons. Each code is matched to the player-data club whose fixture set overlaps
it most, and the result is asserted to be a clean one-to-one mapping before use.
"""
import os
import sqlite3

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.dirname(BASE)
DB = os.path.join(BASE, "tallec.db")
MASTERS = {"NRL": os.path.join(DL, "NRL_2026_MPT", "NRL_master.xlsx"),
           "SL": os.path.join(DL, "NRL_2026_MPT", "SL_master.xlsx")}


def load_master(comp, cols=None):
    keep = ["Match ID", "Season", "Round", "A Team", "B Team", "A Score", "B Score",
            "Margin", "Total", "Home Advantage", "Home_flag", "ELO_A", "ELO_B",
            "Diff ELO", "Played", "Margin_Pred_v1", "Margin_Pred_v2"]
    d = pd.read_excel(MASTERS[comp])
    have = [c for c in (cols or keep) if c in d.columns]
    return d[have].copy()


def _team_points(comp):
    """Points scored by each team in each (season, round), from the player rows."""
    con = sqlite3.connect(DB)
    d = pd.read_sql(
        'SELECT s.season, s.round, s.team, s.opposition, '
        '       sum(COALESCE(r."Points Scored", 0)) AS pts '
        'FROM player_match_stats s '
        'JOIN player_match_raw r ON r.player_id = s.player_id '
        '  AND r.Competition = s.competition AND r.Season = s.season AND r."Round" = s.round '
        'WHERE s.competition = ? GROUP BY 1, 2, 3, 4', con, params=(comp,))
    con.close()
    d["round"] = pd.to_numeric(d["round"], errors="coerce")
    return d.dropna(subset=["round"])


def solve(comp, tol=4, verbose=False):
    """Return {master identifier -> player-data team name}.

    Fixture slots alone do not identify a club — nearly every club plays in nearly
    every round, so the overlaps are all ~0.75 and ties break on noise. The final
    score does identify it: within one (season, round) the side that scored 26 in the
    master is the side that scored 26 in the player data, and the two sides must also
    be each other's opponent. Votes are collected over every fixture and the mapping
    is the highest-voted assignment, which makes a handful of feed-level scoring
    discrepancies harmless.
    """
    m = load_master(comp, ["Season", "Round", "A Team", "B Team", "A Score", "B Score"]).dropna()
    p = _team_points(comp)
    votes = {}
    matched = 0
    for (season, rnd), fx in m.groupby(["Season", "Round"]):
        pr = p[(p.season == season) & (p["round"] == rnd)]
        if pr.empty:
            continue
        pts = dict(zip(pr.team, pr.pts))
        opp = dict(zip(pr.team, pr.opposition))
        for _, f in fx.iterrows():
            for tA, ptsA in pts.items():
                if abs(ptsA - f["A Score"]) > tol:
                    continue
                tB = opp.get(tA)
                if tB is None or tB not in pts:
                    continue
                if abs(pts[tB] - f["B Score"]) > tol:
                    continue
                w = 2 if (ptsA == f["A Score"] and pts[tB] == f["B Score"]) else 1
                votes[(f["A Team"], tA)] = votes.get((f["A Team"], tA), 0) + w
                votes[(f["B Team"], tB)] = votes.get((f["B Team"], tB), 0) + w
                matched += 1
    # greedy one-to-one assignment, strongest evidence first
    mapping, taken = {}, set()
    for (code, team), v in sorted(votes.items(), key=lambda kv: -kv[1]):
        if code in mapping or team in taken:
            continue
        mapping[code] = team
        taken.add(team)
    codes = sorted(set(m["A Team"]) | set(m["B Team"]))
    runner_up = {}
    for code in codes:
        cand = sorted(((v, t) for (c, t), v in votes.items() if c == code), reverse=True)
        mapping.setdefault(code, None)
        runner_up[code] = (cand[0][0] if cand else 0,
                           cand[1][0] if len(cand) > 1 else 0)
    if verbose:
        print(f"  fixtures matched on score: {matched}")
        for code in codes:
            top, second = runner_up[code]
            print(f"  {code:5s} -> {str(mapping[code]):22s} votes {top:4d} "
                  f"(next best {second:4d})")
    assigned = [v for v in mapping.values() if v]
    dupes = {v for v in assigned if assigned.count(v) > 1}
    unmatched = [k for k, v in mapping.items() if v is None]
    thin = [(c, runner_up[c][0]) for c in codes if runner_up[c][0] < 10]
    return mapping, {"duplicates": sorted(dupes), "unmatched": unmatched,
                     "thin_evidence": thin, "fixtures_matched": matched}


if __name__ == "__main__":
    for comp in ("NRL", "SL"):
        print(f"=== {comp} ===")
        mp, issues = solve(comp, verbose=True)
        print(f"  duplicates: {issues['duplicates'] or 'none'} | "
              f"unmatched: {issues['unmatched'] or 'none'} | "
              f"thin evidence (<10 votes): {issues['thin_evidence'] or 'none'}")
        print()
