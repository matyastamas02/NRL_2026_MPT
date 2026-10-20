# -*- coding: utf-8 -*-
"""A learned shortlist order, and how long entrants into Super League stay.

B6, learning to rank. The shortlist question is an ordering inside one landing season
and one target competition, not a rating. A pairwise ranker (RankNet with a linear
score, i.e. logistic regression on feature differences of two players from the same
season and target, label = which one rated higher) is trained on the next-season pairs
of the seasons before each landing season and scored on that season's entrants, against
ordering them by the app's line forecast alone. Features, all known the season before:
the line forecast, the source rating, age, minutes per match, rated matches and the
share of matches started.

B7, survival. Entrants into Super League landing 2021-2025, followed season by season
until they no longer appear in Super League (censored at 2026). Kaplan-Meier retention,
and a discrete-time hazard (logistic, one row per player and season at risk) for
leaving, from what was known at signing (line forecast, age, source start share) and
from the first season (season rating, share of the club's matches started, from Mike's
team sheets).

    python rank_survival.py      # writes RANK_SURVIVAL_REPORT.md
"""
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import frozen_2026 as fz
import rolling_backtest as rb
import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BASE, "RANK_SURVIVAL_REPORT.md")
CACHE = os.environ.get("POST2026_CACHE")
FEATS = ["line", "class_source", "age", "mins_pg", "games_src", "start_share"]
N_BOOT, SEED = 1000, 0


def starts(con):
    s = pd.read_sql("SELECT player_id, comp, season, start_share FROM player_season_starts",
                    con)
    s["player_id"] = sp.normalize_player_id(s.player_id)
    return s.set_index(["player_id", "comp", "season"]).start_share


def prep(f, line_model, st):
    f = f.copy()
    f["player_id"] = sp.normalize_player_id(f.player_id.astype(str))
    f["pair"] = f.source + "->" + f.target
    f["line"] = fz.predict_lines(line_model, f)[0]
    f["start_share"] = st.reindex(list(zip(f.player_id, f.source, f.season_src))).values
    return f


def pairwise(train, mu, sd, max_pairs=60000, seed=SEED):
    rng = np.random.default_rng(seed)
    Xd, yd = [], []
    for _, g in train.groupby(["season_tgt", "target"]):
        if len(g) < 2:
            continue
        X = (g[FEATS].fillna(pd.Series(mu, index=FEATS)).values - mu) / sd
        y = g.class_target.values
        i, j = np.triu_indices(len(g), 1)
        keep = y[i] != y[j]
        i, j = i[keep], j[keep]
        Xd.append(X[i] - X[j])
        yd.append((y[i] > y[j]).astype(int))
    Xd, yd = np.vstack(Xd), np.concatenate(yd)
    if len(yd) > max_pairs:
        k = rng.choice(len(yd), max_pairs, replace=False)
        Xd, yd = Xd[k], yd[k]
    # symmetrise so the model has no intercept to hide in
    Xd, yd = np.vstack([Xd, -Xd]), np.concatenate([yd, 1 - yd])
    return LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(Xd, yd)


def ndcg(rel, score, k):
    o = np.argsort(-score)[:k]
    ideal = np.sort(rel)[::-1][:k]
    disc = 1 / np.log2(np.arange(2, k + 2))
    return float((rel[o] * disc).sum() / (ideal * disc).sum())


def ranking():
    D = pickle.load(open(CACHE, "rb"))
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    st = starts(con)
    con.close()
    rows = []
    for t in (2023, 2024, 2025, 2026):
        tr = D["pairs"][t].copy()
        tr["pair"] = tr.source + "->" + tr.target
        lm = fz.fit_lines(tr)
        tr = prep(tr, lm, st)
        ev = prep(D["cohort"][D["cohort"].season_tgt == t], lm, st)
        mu, sd = tr[FEATS].mean().values, tr[FEATS].std().replace(0, 1).values
        r = pairwise(tr, mu, sd)
        ev["rank_score"] = ((ev[FEATS].fillna(pd.Series(mu, index=FEATS)).values - mu) / sd) @ r.coef_[0]
        rows.append(ev)
    return pd.concat(rows, ignore_index=True), r, FEATS


def survival():
    D = pickle.load(open(CACHE, "rb"))
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    pres = pd.read_sql("SELECT DISTINCT player_id, season FROM player_match_stats "
                       "WHERE competition='SL'", con)
    ts = pd.read_sql("SELECT season, team, round, role, player_id FROM sl_teamsheets "
                     "WHERE player_id IS NOT NULL", con)
    ml = pd.read_sql("SELECT season, team, round FROM sl_match_lines", con)
    st = starts(con)
    con.close()
    pres["player_id"] = sp.normalize_player_id(pres.player_id)
    inSL = set(zip(pres.player_id, pres.season))
    ts["player_id"] = sp.normalize_player_id(ts.player_id)
    club_n = ml.groupby(["season", "team"])["round"].nunique()
    sr = (ts[ts.role == "start"].groupby(["player_id", "season", "team"]).size()
          .rename("starts").reset_index())
    sr["rate"] = sr.starts / club_n.reindex(list(zip(sr.season, sr.team))).values
    sr = sr.sort_values("rate", ascending=False).drop_duplicates(["player_id", "season"])
    rate = sr.set_index(["player_id", "season"]).rate
    c = D["cohort"][(D["cohort"].target == "SL") & (D["cohort"].season_tgt <= 2025)].copy()
    c["player_id"] = sp.normalize_player_id(c.player_id.astype(str))
    c = c.sort_values("season_tgt").drop_duplicates("player_id")        # first entry
    lines = {}
    for t in range(2021, 2026):
        p = D["pairs"].get(t)
        if p is None:                       # landings before 2023: the 2023 lines
            p = D["pairs"][2023]
        p = p.copy()
        p["pair"] = p.source + "->" + p.target
        lines[t] = fz.fit_lines(p)
    c["pair"] = c.source + "->" + c.target
    c["line"] = [fz.predict_lines(lines[t], g)[0][0] for t, g in
                 ((int(r.season_tgt), c.loc[[i]]) for i, r in c.iterrows())]
    c["start_share"] = st.reindex(list(zip(c.player_id, c.source, c.season_src))).values
    c["first_rate"] = rate.reindex(list(zip(c.player_id, c.season_tgt))).values
    risk = []
    for r in c.itertuples():
        for k in range(1, 2027 - int(r.season_tgt)):
            left = (r.player_id, int(r.season_tgt) + k) not in inSL
            risk.append(dict(player_id=r.player_id, k=k, left=int(left), line=r.line,
                             age=r.age, start_share=r.start_share, rating1=r.class_target,
                             rate1=r.first_rate, landing=int(r.season_tgt),
                             source=r.source))
            if left:
                break
    return c, pd.DataFrame(risk)


def boot_or(R, cols, n=N_BOOT):
    R = R.dropna(subset=cols)
    X = R[cols].values
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    Zk = np.column_stack([Z, pd.get_dummies(R.k.clip(upper=3)).values.astype(float)])
    y = R.left.values
    m = LogisticRegression(C=10.0, max_iter=2000).fit(Zk, y)
    ids = R.player_id.values
    u = np.array(sorted(set(ids)))
    at = {p: np.flatnonzero(ids == p) for p in u}
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(n):
        i = np.concatenate([at[p] for p in rng.choice(u, u.size)])
        if y[i].min() == y[i].max():
            continue
        bs.append(LogisticRegression(C=10.0, max_iter=2000).fit(Zk[i], y[i]).coef_[0][:len(cols)])
    bs = np.exp(np.array(bs))
    return [(c, float(np.exp(m.coef_[0][j])), float(np.percentile(bs[:, j], 2.5)),
             float(np.percentile(bs[:, j], 97.5))) for j, c in enumerate(cols)], len(u)


def report():
    R, r, feats = ranking()
    L = ["# Learned shortlist order, and staying in Super League", "",
         "Generated by `rank_survival.py`; methods in its docstring.", "",
         "## B6. A pairwise ranker against ordering by the line forecast", "",
         "Each landing season and target competition is one list. Precision = share of "
         "the top third (by the score) who reached the top third of that list by actual "
         "rating; NDCG@10 on the actual ratings.", "",
         "| Entrants | Lists | Order by | Spearman | Precision, top third | NDCG@10 |",
         "|---|---|---|---|---|---|"]
    for lab, sub in (("into SL from NRL/NSW/QLD", R[R.pair.isin({"NRL->SL", "NSW->SL",
                                                                 "QLD->SL"})]),
                     ("all directions", R)):
        for col, name in (("line", "line forecast"), ("rank_score", "pairwise ranker")):
            sp_, pr, nd = [], [], []
            for _, g in sub.groupby(["season_tgt", "target"]):
                if len(g) < 6:
                    continue
                rel = g.class_target.values
                s = g[col].values
                sp_.append(pd.Series(s).corr(pd.Series(rel), method="spearman"))
                k = max(1, len(g) // 3)
                top_true = set(np.argsort(-rel)[:k])
                pr.append(len(top_true & set(np.argsort(-s)[:k])) / k)
                nd.append(ndcg(rel, s, min(10, len(g))))
            L.append(f"| {lab} | {len(sp_)} | {name} | {np.mean(sp_):+.2f} | "
                     f"{np.mean(pr):.0%} | {np.mean(nd):.3f} |")
    L += ["", "Ranker weights in the last season (standardised): "
          + ", ".join(f"{f} {w:+.2f}" for f, w in zip(feats, r.coef_[0])), ""]
    c, risk = survival()
    L += ["## B7. Staying in Super League", "",
          f"{len(c)} first entries into Super League, landing 2021–2025, followed to 2026.",
          "", "| Seasons after landing | Still in Super League (Kaplan–Meier) | At risk |",
          "|---|---|---|"]
    s = 1.0
    for k in range(1, 6):
        at = risk[risk.k == k]
        if at.empty:
            break
        s *= 1 - at.left.mean()
        L.append(f"| {k} | {s:.0%} | {len(at)} |")
    for cols, lab in ((["line", "age", "start_share"], "known at signing"),
                      (["rating1", "rate1", "age"], "after the first season")):
        ors, n = boot_or(risk, cols)
        L += ["", f"Hazard of leaving, {lab} ({n} players). Odds ratio per standard "
              "deviation; below 1 = stays longer.", "", "| Variable | Odds ratio [95%] |",
              "|---|---|"]
        for v, o, lo, hi in ors:
            L.append(f"| {v} | {o:.2f} [{lo:.2f}, {hi:.2f}] |")
    L.append("")
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    report()
    sys.exit(0)
