# -*- coding: utf-8 -*-
"""Role and rating for entrants into Super League: a two-part forecast, and how much of
the rating a role explains.

Mike's team sheets (`sl_teamsheets`) give, for every Super League match 2021-2026,
who wore 1-13, who was on the bench (14-17) and who was in the 21 but not the 17. That
is the role the club actually gave a man, match by match.

B1, a two-part (hurdle) forecast. A signing's season is two things: whether he gets the
shirt, and how well he plays when he has it. The rating forecast says nothing about the
first: in the post-2026 exploration its AUC for becoming a starter was 0.43-0.47. Here
  P(starter)  logistic on what was known the season before: share of matches started
              and minutes per match in the source, the line forecast; fitted on earlier
              landing seasons only
  E[rating]   the app's line forecast (S0)
and the season they produce, contribution = share of the club's matches started x
season rating, is ranked by S0 alone, by P(starter) alone and by their product.

A4, role as a mediator, in the form that can be read without strong assumptions.
  - how much of the rating the line misses is explained by the role he got (share of the
    club's matches started), and by his role in the club's first five matches, a proxy
    for the role the club planned that is fixed early
  - the matched shift (matching_moves.py) for entrants who became starters and for those
    who did not
A formal decomposition (natural direct and indirect effects) would need the mediator to
mean the same thing on both sides of the comparison; a start in the NSW Cup and a start
in Super League do not, so it is not attempted.

Uses the post-2026 exploration's entrant table (post2026_explore/entrants.csv: each
landing season's forecast built from the seasons before) and the matched movers
(matching/movers_matched.csv).

    python role_hurdle.py      # writes ROLE_HURDLE_REPORT.md
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import rolling_backtest as rb
import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BASE, "ROLE_HURDLE_REPORT.md")
H1 = {"NRL->SL", "NSW->SL", "QLD->SL"}
N_BOOT, SEED = 2000, 0


def sl_roles(con):
    ts = pd.read_sql("SELECT season, round, team, role, player_id FROM sl_teamsheets "
                     "WHERE player_id IS NOT NULL", con)
    ts["player_id"] = sp.normalize_player_id(ts.player_id)
    club = pd.read_sql("SELECT season, team, round FROM sl_match_lines", con)
    n_club = club.groupby(["season", "team"])["round"].nunique().rename("club_matches")
    first5 = (club.sort_values("round").groupby(["season", "team"])["round"]
              .apply(lambda r: set(r.head(5))))
    ts["early"] = [r in first5.get((s, t), set()) for s, t, r in
                   zip(ts.season, ts.team, ts["round"])]
    g = ts.groupby(["player_id", "season", "team"])
    r = pd.DataFrame({"starts": g.role.apply(lambda x: (x == "start").sum()),
                      "bench": g.role.apply(lambda x: (x == "bench").sum()),
                      "reserve": g.role.apply(lambda x: (x == "reserve").sum()),
                      "early_starts": g.apply(lambda d: ((d.role == "start") & d.early).sum(),
                                              include_groups=False)}).reset_index()
    r = r.merge(n_club.reset_index(), on=["season", "team"], how="left")
    # a man at two clubs in a season: keep the club he started most for
    r = r.sort_values("starts", ascending=False).drop_duplicates(["player_id", "season"])
    r["start_rate"] = r.starts / r.club_matches
    r["early_rate"] = r.early_starts / 5.0
    r["starter_ts"] = (r.starts >= 0.5 * r.club_matches).astype(float)
    return r


def boot(fn, ids, n=N_BOOT):
    ids = np.asarray(ids)
    u = np.array(sorted(pd.unique(ids)))
    at = {p: np.flatnonzero(ids == p) for p in u}
    rng = np.random.default_rng(SEED)
    v = [fn(np.concatenate([at[p] for p in rng.choice(u, u.size)])) for _ in range(n)]
    v = np.array([x for x in v if np.isfinite(x)])
    return fn(np.arange(ids.size)), np.percentile(v, 2.5), np.percentile(v, 97.5)


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = ~(np.isnan(y) | np.isnan(s))
    y, s = y[ok], s[ok]
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if not n1 or not n0:
        return np.nan
    r = pd.Series(s).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def run():
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    roles = sl_roles(con)
    con.close()
    ev = pd.read_csv(os.path.join(BASE, "post2026_explore", "entrants.csv"))
    ev["player_id"] = sp.normalize_player_id(ev.player_id.astype(str))
    sl = ev[ev.pair.isin(H1)].merge(
        roles.rename(columns={"season": "season_tgt"}), on=["player_id", "season_tgt"],
        how="left")
    sl = sl[sl.club_matches.notna()].reset_index(drop=True)
    # P(starter), fitted each season on the entrants who landed before it
    feats = ["src_start_share", "src_mins", "S0"]
    sl["p_starter"] = np.nan
    for t in sorted(sl.season_tgt.unique()):
        tr = sl[(sl.season_tgt < t)].dropna(subset=feats + ["starter_ts"])
        te = sl.season_tgt == t
        if len(tr) < 25 or tr.starter_ts.nunique() < 2:
            continue
        X = tr[feats].values
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1
        m = LogisticRegression(C=1.0).fit((X - mu) / sd, tr.starter_ts)
        Z = sl.loc[te, feats].values
        ok = ~np.isnan(Z).any(1)
        p = np.full(te.sum(), np.nan)
        p[ok] = m.predict_proba((Z[ok] - mu) / sd)[:, 1]
        sl.loc[te, "p_starter"] = p
    sl["contribution"] = sl.start_rate * sl.class_target
    sl["hurdle"] = sl.p_starter * sl.S0
    sl["resid"] = sl.class_target - sl.S0
    return sl


def report(sl):
    L = ["# Role and rating for entrants into Super League", "",
         "Generated by `role_hurdle.py`. Entrants into Super League from the NRL, NSW Cup "
         "and Queensland Cup with a team-sheet record in the landing season. Roles from "
         "Mike's team sheets. Intervals: 95% player-cluster bootstrap.", "",
         f"Entrants: {len(sl)} (landing {sl.season_tgt.min()}–{sl.season_tgt.max()}); "
         f"regular starters (half or more of the club's matches in 1-13): "
         f"{sl.starter_ts.mean():.0%}. Median share of the club's matches started: "
         f"{sl.start_rate.median():.0%}.", ""]
    o = sl[sl.p_starter.notna()]
    ids = o.player_id.values
    L += ["## B1. Who gets the shirt, and the season it adds up to", "",
          f"Scored on {len(o)} entrants whose P(starter) was fitted on earlier landing "
          "seasons only.", "",
          "| Score | AUC for becoming a regular starter [95%] | Spearman with contribution "
          "(share started x rating) [95%] |", "|---|---|---|"]
    for s, lab in (("S0", "Line forecast (rating only)"), ("src_start_share",
                   "Share of matches started in the source"), ("p_starter", "P(starter)"),
                   ("hurdle", "P(starter) x line forecast")):
        a = boot(lambda i: auc(o.starter_ts.values[i], o[s].values[i]), ids)
        r = boot(lambda i: spearman(o[s].values[i], o.contribution.values[i]), ids)
        L.append(f"| {lab} | {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}] | "
                 f"{r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] |")
    d = boot(lambda i: spearman(o.hurdle.values[i], o.contribution.values[i])
             - spearman(o.S0.values[i], o.contribution.values[i]), ids)
    base = np.mean(o.starter_ts)
    brier = np.mean((o.p_starter - o.starter_ts) ** 2)
    b0 = np.mean((base - o.starter_ts) ** 2)
    L += ["", f"Hurdle against the line on contribution: {d[0]:+.2f} [{d[1]:+.2f}, "
          f"{d[2]:+.2f}] in Spearman. P(starter) Brier {brier:.3f} against {b0:.3f} for the "
          f"base rate (skill {1 - brier / b0:+.3f}).", ""]

    L += ["## A4. How much of the rating the role explains", "",
          "Residual = season rating in Super League minus the line forecast. Slope per 10 "
          "percentage points of the club's matches started.", "",
          "| Role measure | Slope per 10 pts [95%] | R² |", "|---|---|---|"]
    ids = sl.player_id.values
    for col, lab in (("start_rate", "Share of the club's matches started, whole season"),
                     ("early_rate", "Starts in the club's first five matches (the early, "
                                    "planned role)")):
        ok = sl[col].notna() & sl.resid.notna()
        x, y = sl.loc[ok, col].values, sl.loc[ok, "resid"].values
        slope = lambda i: np.polyfit(x[i], y[i], 1)[0] / 10
        s = boot(slope, ids[ok.values])
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        L.append(f"| {lab} | {s[0]:+.2f} [{s[1]:+.2f}, {s[2]:+.2f}] | {r2:.3f} |")
    L.append("")

    mm_path = os.path.join(BASE, "matching", "movers_matched.csv")
    if os.path.exists(mm_path):
        mm = pd.read_csv(mm_path)
        mm["player_id"] = sp.normalize_player_id(mm.player_id.astype(str))
        mm = mm[mm.target == "SL"].merge(
            sl[["player_id", "season_tgt", "starter_ts", "start_rate"]],
            on=["player_id", "season_tgt"], how="inner")
        L += ["Matched shift (mover in Super League against matched stayers), by the role "
              "the mover got:", "", "| Role in the landing season | Movers | Matched shift "
              "[95%] |", "|---|---|---|"]
        for v, g in mm.groupby("starter_ts"):
            b = boot(lambda i: g.att_i.values[i].mean(), g.player_id.values)
            L.append(f"| {'regular starter' if v == 1 else 'not a regular starter'} | "
                     f"{len(g)} | {b[0]:+.1f} [{b[1]:+.1f}, {b[2]:+.1f}] |")
        L.append("")
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    report(run())
    sys.exit(0)
