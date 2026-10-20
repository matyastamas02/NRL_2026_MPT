# -*- coding: utf-8 -*-
"""Three robustness checks on the competition gaps: a doubly robust estimate, Lee
bounds tightened with a covariate, and how much hidden selection would undo them.

A1, doubly robust (AIPW for the effect on the movers). Matching uses each mover's few
nearest stayers. The doubly robust estimator uses every stayer, weighted by the odds
of moving, and corrects with an outcome model fitted on the stayers; it is consistent
if either the propensity or the outcome model is right.
    tau = [sum_movers (y - m0(x)) - sum_stayers e/(1-e) (y - m0(x))] / n_movers
e(x): logistic, movers a->b against stayers in a, on the season-before profile plus
season and position group; m0(x): linear on the same, stayers only. Propensities are
trimmed to [0.01, 0.99]. Intervals: 300 player-cluster resamples, refitting both
models each time.

A7, Lee bounds with a covariate (in the spirit of Semenova 2023): bounds computed
inside thirds of the mover's season-before rating, then averaged with the movers'
weights. Sharper wherever the chance of reaching three matches differs across thirds.

A8, sensitivity.
  Rosenbaum: matched differences (mover minus his stayers) tested with the signed-rank
    statistic; Gamma is how much more likely one of two players who look the same
    could be to move, for a hidden reason, before the shift loses significance at 5%.
  Robustness value (Cinelli and Hazlett 2020): from the regression of the rating on
    the move and the profile, the share of the residual variance of both the move and
    the rating an unobserved confounder would need to explain to bring the estimate to
    zero (RV) or to make it insignificant (RV_a).

    python robust_gaps.py        # writes ROBUST_GAPS_REPORT.md
"""
import os
import sqlite3
import sys
from math import erf, sqrt

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

import lee_bounds as lb
import matching_moves as mm
import rolling_backtest as rb

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BASE, "ROBUST_GAPS_REPORT.md")
N_BOOT, SEED = 300, 0
MIN_N = 15


def design(f, seasons, groups):
    X = f[mm.COVS].astype(float)
    X = X.fillna(X.median())
    for s in seasons[1:]:
        X[f"s{s}"] = (f.season_src == s).astype(float)
    for g in groups[1:]:
        X[f"g{g}"] = (f.group == g).astype(float)
    return X.values


def aipw(mv, st):
    seasons = sorted(set(mv.season_src) | set(st.season_src))
    groups = sorted(set(mv.group) | set(st.group))
    Xm, Xs = design(mv, seasons, groups), design(st, seasons, groups)
    X = np.vstack([Xm, Xs])
    t = np.r_[np.ones(len(Xm)), np.zeros(len(Xs))]
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    e = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, t).predict_proba(
        (X - mu) / sd)[:, 1].clip(0.01, 0.99)
    m0 = LinearRegression().fit(Xs, st.y.values)
    rm, rs = mv.y.values - m0.predict(Xm), st.y.values - m0.predict(Xs)
    es = e[len(Xm):]
    return float((rm.sum() - (es / (1 - es) * rs).sum()) / len(Xm))


def aipw_boot(mv, st):
    point = aipw(mv, st)
    rng = np.random.default_rng(SEED)
    ids_m, ids_s = mv.player_id.values, st.player_id.values
    um = np.array(sorted(set(ids_m)))
    at_m = {p: np.flatnonzero(ids_m == p) for p in um}
    us = np.array(sorted(set(ids_s)))
    at_s = {p: np.flatnonzero(ids_s == p) for p in us}
    v = []
    for _ in range(N_BOOT):
        im = np.concatenate([at_m[p] for p in rng.choice(um, um.size)])
        is_ = np.concatenate([at_s[p] for p in rng.choice(us, us.size)])
        try:
            v.append(aipw(mv.iloc[im], st.iloc[is_]))
        except ValueError:
            continue
    return point, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def rosenbaum_gamma(d, alpha=0.05):
    """Largest Gamma at which the signed-rank test of d > 0 (or < 0) stays significant."""
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) < 5:
        return np.nan
    sign = 1 if np.median(d) > 0 else -1
    r = pd.Series(np.abs(d)).rank().values
    T = r[(sign * d) > 0].sum()
    best = np.nan
    for g in np.arange(1.0, 6.01, 0.05):
        p = g / (1 + g)
        z = (T - p * r.sum()) / np.sqrt(p * (1 - p) * (r ** 2).sum())
        pval = 0.5 * (1 - erf(z / sqrt(2)))
        if pval < alpha:
            best = g
        else:
            break
    return best


def robustness_value(mv, st):
    """Cinelli-Hazlett RV and RV_alpha for the move coefficient in y ~ move + profile."""
    seasons = sorted(set(mv.season_src) | set(st.season_src))
    groups = sorted(set(mv.group) | set(st.group))
    X = np.vstack([design(mv, seasons, groups), design(st, seasons, groups)])
    t = np.r_[np.ones(len(mv)), np.zeros(len(st))]
    y = np.r_[mv.y.values, st.y.values]
    Z = np.column_stack([np.ones(len(y)), t, X])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    res = y - Z @ b
    df = len(y) - Z.shape[1]
    s2 = (res ** 2).sum() / df
    cov = s2 * np.linalg.pinv(Z.T @ Z)
    tval = b[1] / np.sqrt(cov[1, 1])
    f = abs(tval) / np.sqrt(df)
    rv = 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)
    fa = max(f - 1.96 / np.sqrt(df - 1), 0)
    rva = 0.5 * (np.sqrt(fa ** 4 + 4 * fa ** 2) - fa ** 2) if fa > 0 else 0.0
    return float(b[1]), float(tval), float(rv), float(rva)


def lee_with_cells(T_dir, S, J, M):
    """Lee bounds inside thirds of y_prev, averaged with the movers' weights."""
    rows = []
    for pair, g in M.groupby("pair"):
        if len(g) < 3 * 8:
            continue
        cuts = pd.qcut(g.y_prev, 3, labels=False, duplicates="drop")
        lo = hi = 0.0
        for c, gc in g.groupby(cuts):
            if len(gc) < 5 or gc.obs.sum() == 0:
                break
            _, l, h, _, _ = lb.direction(gc, S, J)
            lo += l * len(gc) / len(g)
            hi += h * len(gc) / len(g)
        else:
            rows.append(dict(pair=pair, lo_cells=lo, hi_cells=hi))
    return pd.DataFrame(rows)


def run():
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    D = mm.load(con)
    con.close()
    M = mm.movers(D)
    M["pair"] = M.source + "->" + M.target
    S = mm.stayers(D, M)
    # A1 and A8 per direction
    rows = []
    _, Mm, Sm, Jm, _ = mm.run(D=D)
    for pair, g in M.groupby("pair"):
        if len(g) < MIN_N:
            continue
        st = S[S.comp == g.source.iloc[0]]
        st = st[st.season_src.isin(g.season_src.unique())]
        a = aipw_boot(g, st)
        coef, tv, rv, rva = robustness_value(g, st)
        dm = Mm[Mm.pair == pair].att_i.values
        rows.append(dict(pair=pair, n=len(g), aipw=a[0], lo=a[1], hi=a[2],
                         matched=float(np.mean(dm)) if len(dm) else np.nan,
                         gamma=rosenbaum_gamma(dm), ols=coef, t=tv, rv=rv, rva=rva))
    A = pd.DataFrame(rows)
    # A7 on the main and one-match settings
    cells = {}
    for min_n, col, label in (lb.SETTINGS[0], lb.SETTINGS[1]):
        Mr = lb.movers_at_risk(D, min_n, col)
        Sr = lb.stayers_at_risk(D, Mr, min_n, col)
        Jr, _ = mm.match(Mr, Sr)
        keep = np.array([len(j) > 0 for j in Jr])
        Jr = [j for j, k in zip(Jr, keep) if k]
        Mr = Mr[keep].reset_index(drop=True)
        T = pd.DataFrame([dict(pair=p, **dict(zip(["att", "lo", "hi"],
                                                  lb.direction(g, Sr, Jr)[:3])))
                          for p, g in Mr.groupby("pair")])
        cells[label] = T.merge(lee_with_cells(T, Sr, Jr, Mr), on="pair", how="left")
    return A, cells


def pair_gap(T, a, b, lo="lo", hi="hi", point="att"):
    t = T.set_index("pair")
    f, r = f"{a}->{b}", f"{b}->{a}"
    if f not in t.index or r not in t.index:
        return None
    return ((t.loc[f, point] - t.loc[r, point]) / 2,
            (t.loc[f, lo] - t.loc[r, hi]) / 2, (t.loc[f, hi] - t.loc[r, lo]) / 2)


def report(A, cells):
    L = ["# Robustness of the competition gaps", "",
         "Generated by `robust_gaps.py`; methods in its docstring. Ratings and the "
         "season-before profile as in `MATCHING_REPORT.md`.", "",
         "## A1. Doubly robust shift and A8. sensitivity, by direction", "",
         "Shift = mover's rating in the new competition minus what comparable stayers "
         "rated in the old one. Γ: hidden selection, as an odds ratio of moving, that the "
         "matched shift survives at 5%. RV: share of residual variance a hidden confounder "
         "must explain in both the move and the rating to remove the estimate (RV_a: to "
         "make it insignificant).", "",
         "| Move | Movers | Matched | Doubly robust [95%] | Γ | RV | RV_a |",
         "|---|---|---|---|---|---|---|"]
    for r in A.sort_values("n", ascending=False).itertuples():
        g = "—" if not np.isfinite(r.gamma) else f"{r.gamma:.2f}"
        L.append(f"| {r.pair} | {r.n} | {r.matched:+.1f} | {r.aipw:+.1f} "
                 f"[{r.lo:+.1f}, {r.hi:+.1f}] | {g} | {r.rv:.2f} | {r.rva:.2f} |")
    L += ["", "Pair gaps (half the difference of the two directions):", "",
          "| Pair | Matched | Doubly robust [interval from the two directions] |",
          "|---|---|---|"]
    T = A.rename(columns={"aipw": "att"})
    for a, b in (("NRL", "NSW"), ("NRL", "QLD"), ("NRL", "SL"), ("NSW", "QLD"),
                 ("NSW", "SL"), ("QLD", "SL")):
        g = pair_gap(T, a, b)
        mt = pair_gap(A.rename(columns={"matched": "att"}).assign(lo=0, hi=0), a, b)
        if g:
            L.append(f"| {a}–{b} | {mt[0]:+.1f} | {g[0]:+.1f} [{g[1]:+.1f}, {g[2]:+.1f}] |")
    L += ["", "## A7. Lee bounds with a covariate", "",
          "Bounds inside thirds of the season-before rating, averaged; against the plain "
          "bounds of `LEE_BOUNDS_REPORT.md`.", "",
          "| Pair | Setting | Plain bounds | Bounds within thirds |", "|---|---|---|---|"]
    for label, T in cells.items():
        for a, b in (("NRL", "NSW"), ("NRL", "QLD"), ("NRL", "SL"), ("NSW", "SL"),
                     ("QLD", "SL"), ("NSW", "QLD")):
            g = pair_gap(T, a, b)
            c = pair_gap(T.dropna(subset=["lo_cells"]), a, b, "lo_cells", "hi_cells")
            if g:
                cs = f"[{c[1]:+.1f}, {c[2]:+.1f}]" if c else "too few movers per third"
                L.append(f"| {a}–{b} | {label} | [{g[1]:+.1f}, {g[2]:+.1f}] | {cs} |")
    L.append("")
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    report(*run())
    sys.exit(0)
