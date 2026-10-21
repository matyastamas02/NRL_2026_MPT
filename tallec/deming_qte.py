# -*- coding: utf-8 -*-
"""A structural exchange rate between competitions, and whether moving shifts every
part of the distribution alike.

B9, errors in variables. The app's line regresses the rating after a move on the rating
before it, which is the right thing for a forecast: it shrinks toward the middle
because the rating before is measured with error. For an exchange rate ("a 60 in the
NSW Cup is worth what in Super League?") the question is about true levels, and least
squares understates the slope. Deming regression corrects it, given the ratio of the
two error variances. The error variances come from reliabilities measured here:
split-half (odd against even matches) correlation of a player-season's mean composite,
stepped down to one match and up to the number of matches behind each rating by
Spearman-Brown, within the source and the target competition.

A9, quantile effects. The matched shift is a mean. Comparing the movers' ratings with
the matched stayers' (each mover's stayers weighted 1/K) quantile by quantile shows
whether the low end and the high end shift by the same amount: does moving into Super
League lift weak players more than strong ones?

    python deming_qte.py       # writes DEMING_QTE_REPORT.md
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

import kalman_ability as ka
import matching_moves as mm
import rolling_backtest as rb

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BASE, "DEMING_QTE_REPORT.md")
QS = [0.1, 0.25, 0.5, 0.75, 0.9]
N_BOOT, SEED = 1000, 0


def one_match_reliability(con, comp):
    pm = ka.composites(con, comp)
    pm["k"] = pm.groupby(["player_id", "season"]).cumcount()
    g = pm.groupby(["player_id", "season"])
    n = g.size()
    keep = n[n >= 8].index
    pm = pm.set_index(["player_id", "season"]).loc[keep].reset_index()
    h = pm.assign(half=pm.k % 2).groupby(["player_id", "season", "half"]).composite.mean().unstack()
    r_half = float(h[0].corr(h[1]))
    m = float((n[n >= 8] / 2).mean())
    rho1 = r_half / (m - (m - 1) * r_half)
    return rho1, r_half


def sb(rho1, n):
    return n * rho1 / (1 + (n - 1) * rho1)


def deming(x, y, lam):
    sxx, syy = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.cov(x, y)[0, 1]
    b = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
    return b, y.mean() - b * x.mean()


def wquantile(v, w, q):
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) / w.sum()
    return np.interp(q, c, v)


def run():
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    rel = {c: one_match_reliability(con, c) for c in ("NRL", "NSW", "QLD", "SL")}
    p = pd.read_sql("SELECT source, target, class_source, class_target, n_tgt FROM "
                    "translation_pairs_v3 WHERE layer='B_next_season'", con)
    nc = pd.read_sql("SELECT source, target, class_source FROM translation_pairs_v3 "
                     "WHERE layer='B_next_season'", con)
    D = mm.load(con)
    con.close()
    # effective matches behind the source Class: from the cumulative table
    cum = D["cum"]
    rows = []
    for (s, t), g in p.groupby(["source", "target"]):
        if len(g) < 25:
            continue
        x, y = g.class_source.values, g.class_target.values
        n_src = float(cum[cum.comp == s].n_games.median())
        n_tgt = float(g.n_tgt.median())
        rx, ry = sb(rel[s][0], n_src), sb(rel[t][0], n_tgt)
        lam = ((1 - ry) * np.var(y)) / ((1 - rx) * np.var(x))
        b_ols, a_ols = np.polyfit(x, y, 1)
        b_dem, a_dem = deming(x, y, lam)
        b_att = b_ols / rx                    # error in the source only
        rows.append(dict(source=s, target=t, n=len(g), rel_src=rx, rel_tgt=ry,
                         b_ols=b_ols, a_ols=a_ols, b_dem=b_dem, a_dem=a_dem,
                         b_att=b_att, a_att=y.mean() - b_att * x.mean()))
    T = pd.DataFrame(rows)
    # quantile effects from the matched movers
    _, M, S, J, _ = mm.run(D=D)
    Q = {}
    for lab, mask in (("into SL from NRL/NSW/QLD", (M.target == "SL") &
                       M.source.isin(["NRL", "NSW", "QLD"])),
                      ("NRL -> NSW Cup", (M.source == "NRL") & (M.target == "NSW"))):
        idx = np.flatnonzero(mask.values)
        def qte(ix):
            ym = M.y.values[ix]
            cj = np.concatenate([J[i] for i in ix])
            wc = np.concatenate([np.full(len(J[i]), 1 / len(J[i])) for i in ix])
            return np.array([np.quantile(ym, q) - wquantile(S.y.values[cj], wc, q)
                             for q in QS])
        point = qte(idx)
        ids = M.player_id.values[idx]
        u = np.array(sorted(set(ids)))
        at = {p_: idx[ids == p_] for p_ in u}
        rng = np.random.default_rng(SEED)
        bs = np.array([qte(np.concatenate([at[p_] for p_ in rng.choice(u, u.size)]))
                       for _ in range(N_BOOT)])
        Q[lab] = (len(idx), point, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0))
    return rel, T, Q


def report(rel, T, Q):
    L = ["# A structural exchange rate, and quantile effects of moving", "",
         "Generated by `deming_qte.py`; methods in its docstring.", "",
         "## Reliability of a rating", "",
         "| Competition | Split-half correlation | One match | 10 matches | 25 matches |",
         "|---|---|---|---|---|"]
    for c, (r1, rh) in rel.items():
        L.append(f"| {c} | {rh:.2f} | {r1:.3f} | {sb(r1, 10):.2f} | {sb(r1, 25):.2f} |")
    L += ["", "## B9. Least squares (the forecast) against Deming (the exchange rate)", "",
          "| Move | Pairs | Reliability, source / target | Forecast slope | Source error "
          "removed | Deming | 40 → | 50 → | 60 → | 70 → | 80 → |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in T.sort_values("n", ascending=False).itertuples():
        eq = " | ".join(f"{np.clip(r.a_ols + r.b_ols * v, 0, 100):.0f} / "
                        f"{np.clip(r.a_att + r.b_att * v, 0, 100):.0f}"
                        for v in (40, 50, 60, 70, 80))
        L.append(f"| {r.source}->{r.target} | {r.n} | {r.rel_src:.2f} / {r.rel_tgt:.2f} | "
                 f"{r.b_ols:.2f} | {r.b_att:.2f} | {r.b_dem:.2f} | {eq} |")
    L += ["", "Each cell: forecast / the 'source error removed' equivalent. **Neither "
          "correction is valid here, and no exchange rate is claimed.** The source rating "
          "is Class, which is already an empirical-Bayes posterior mean: shrinkage has "
          "removed most of the attenuation (the next-season slope on the shrunk rating is "
          "1.16-1.27 times the slope on the raw one), so dividing by a reliability corrects "
          "a second time, and Deming's error variances are computed for a shrunk score and "
          "assume the true level does not change between seasons (an independent review). "
          "The forecast slopes stand as forecasts.", "",
          "## A9. Quantile effects of moving", "",
          "Mover's rating minus matched stayers' rating, quantile by quantile. Differences "
          "between quantiles are not established (into SL, q50 minus q90 is +7.2 "
          "[-2.0, +18.5]), and a constant shift on the latent scale pushed through the "
          "bounded 0-100 scale predicts the same pattern.", "",
          "| Moves | Movers | " + " | ".join(f"q{int(q * 100)}" for q in QS) + " |",
          "|---|---|" + "---|" * len(QS)]
    for lab, (n, pt, lo, hi) in Q.items():
        L.append(f"| {lab} | {n} | " + " | ".join(
            f"{a:+.1f} [{b:+.1f}, {c:+.1f}]" for a, b, c in zip(pt, lo, hi)) + " |")
    L.append("")
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    report(*run())
    sys.exit(0)
