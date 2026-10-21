# -*- coding: utf-8 -*-
"""A gradient-boosting benchmark for the translation, and whether style fit matters.

B4. Everything known the season before, handed to a flexible learner: source rating,
age, minutes per match, rated matches, share started, the nine per-rate profile values
(each rate's player mean shrunk by its own reliability, from `explore_post2026.py`),
source and target competitions and the position group. Histogram gradient boosting,
shallow trees, early stopping on a validation split by player, five seeds averaged,
on the rating basis known at each cut-off, fitted on the next-season
pairs of the seasons before each landing season. If this does not beat the line, no
amount of feature mining on these data will.

B8. Style fit (the Transfer Portal idea). A player's style: his per-rate profile. A
club's style: the mean per-rate profile of its players in the target competition the
season before. Fit: the cosine similarity of the two, for the club he actually joined
(the first club he is seen at). Tested as an extra term on the app's line
(`frozen_2026.fit_lines` with a numeric extra, the same ridge as the frozen test) and
as a correlate of what the line misses.

    python gbm_style.py      # writes GBM_STYLE_REPORT.md
"""
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

import explore_post2026 as ep
import frozen_2026 as fz
import rolling_backtest as rb
import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BASE, "GBM_STYLE_REPORT.md")
CACHE = os.environ.get("POST2026_CACHE")
SZ = ["s_" + c for c in ep.ZC]
MZ = ["m_" + c for c in ep.ZC]
NUM = ["class_source", "age", "mins_pg", "games_src", "start_share"] + SZ


def club_tables(con):
    pm = pd.read_sql("SELECT player_id, competition comp, season, round, team "
                     "FROM player_match_stats", con)
    pm["player_id"] = sp.normalize_player_id(pm.player_id)
    first = pm.sort_values("round").groupby(["player_id", "comp", "season"]).team.first()
    st = pd.read_sql("SELECT player_id, comp, season, team, start_share FROM "
                     "player_season_starts", con)
    st["player_id"] = sp.normalize_player_id(st.player_id)
    return first, st


def club_style(one, st):
    j = one.merge(st[["player_id", "comp", "season", "team"]],
                  on=["player_id", "comp", "season"], how="inner")
    return j.groupby(["comp", "season", "team"])[MZ].mean()


def features(f, D, first, st, style, t):
    f = ep.attach(f, D.get("cum_by_t", {}).get(t, D["cum"]),
                  D.get("one_by_t", {}).get(t, D["one"]), False)
    f["pair"] = f.source + "->" + f.target
    s = st.set_index(["player_id", "comp", "season"]).start_share
    f["start_share"] = s.reindex(list(zip(f.player_id, f.source, f.season_src))).values
    f["club"] = first.reindex(list(zip(f.player_id, f.target, f.season_tgt))).values
    cs = style.reindex(list(zip(f.target, f.season_tgt - 1, f.club)))
    a, b = np.nan_to_num(f[SZ].values), np.nan_to_num(cs.values)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    f["fit"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return f


def design(f, cats):
    X = f[NUM].astype(float)
    for c, vals in cats.items():
        for v in vals:
            X[f"{c}={v}"] = (f[c] == v).astype(float)
    return X


def run():
    D = pickle.load(open(CACHE, "rb"))
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    first, st = club_tables(con)
    con.close()
    out = []
    for t in (2023, 2024, 2025, 2026):
        style = club_style(D.get("one_by_t", {}).get(t, D["one"]), st)
        tr = features(D["pairs"][t], D, first, st, style, t)
        ev = features(D["cohort"][D["cohort"].season_tgt == t], D, first, st, style, t)
        cats = {"source": sorted(tr.source.unique()), "target": sorted(tr.target.unique()),
                "group": sorted(tr.group.dropna().unique())}
        Xtr, Xev = design(tr, cats), design(ev, cats)
        # Early stopping on a validation set split by PLAYER (a random split let the
        # same player's other pairs into validation), and five seeds averaged: the
        # first version used one seed, which sat at the top of the seed distribution.
        preds = []
        for seed in range(5):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            fit_i, val_i = next(gss.split(Xtr, tr.class_target, tr.player_id))
            gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                max_iter=500, l2_regularization=1.0,
                                                early_stopping=True, random_state=seed)
            gbm.fit(Xtr.iloc[fit_i], tr.class_target.iloc[fit_i],
                    X_val=Xtr.iloc[val_i], y_val=tr.class_target.iloc[val_i])
            preds.append(gbm.predict(Xev))
        ev["S0"] = fz.predict_lines(fz.fit_lines(tr), ev)[0]
        ev["gbm"] = np.clip(np.mean(preds, axis=0), 0, 100)
        trf = tr.assign(fit=tr.fit.fillna(tr.fit.median()))
        evf = ev.assign(fit=ev.fit.fillna(tr.fit.median()))
        ev["S_fit"] = fz.predict_lines(fz.fit_lines(trf, [("fit", "numeric")]), evf)[0]
        out.append(ev)
    return pd.concat(out, ignore_index=True)


def report(R):
    L = ["# Gradient boosting and style fit", "",
         "Generated by `gbm_style.py`; methods in its docstring. Rolling, landing seasons "
         "2023–2026; Δ > 0 means the alternative is closer than the app's line.", "",
         "| Entrants | n | MAE, app line | Gradient boosting | Δ [95%] | Line + style fit | "
         "Δ [95%] |", "|---|---|---|---|---|---|---|"]
    for lab, g in (("into SL from NRL/NSW/QLD", R[R.pair.isin({"NRL->SL", "NSW->SL",
                                                               "QLD->SL"})]),
                   ("all directions", R)):
        y = g.class_target.values
        a = fz.paired_bootstrap(y, g.S0.values, g.gbm.values, g.player_id.values)
        b = fz.paired_bootstrap(y, g.S0.values, g.S_fit.values, g.player_id.values)
        L.append(f"| {lab} | {len(g)} | {np.abs(y - g.S0).mean():.2f} | "
                 f"{np.abs(y - g.gbm).mean():.2f} | {a[0]:+.2f} [{a[1]:+.2f}, {a[2]:+.2f}] | "
                 f"{np.abs(y - g.S_fit).mean():.2f} | {b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}] |")
    ok = R.fit.notna()
    r = (R.class_target - R.S0)[ok].corr(R.fit[ok])
    L += ["", f"Style fit available for {ok.mean():.0%} of entrants; correlation of fit "
          f"with what the line misses: {r:+.3f}.", ""]
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    report(run())
    sys.exit(0)
