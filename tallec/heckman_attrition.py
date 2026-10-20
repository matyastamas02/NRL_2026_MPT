# -*- coding: utf-8 -*-
"""Heckman selection for the arrivals who never reach three matches.

Lee bounds (`lee_bounds.py`) assume nothing about who drops out and are wide where many
do. Heckman's two-step model assumes more and returns a point: a probit for reaching
three matches in the new competition, and the rating equation on those who did, with
the inverse Mills ratio from the probit as the correction. Its credibility rests on an
exclusion restriction, something that moves the chance of getting games without moving
how well a man plays when he gets them. Used here: the vacancy at his new club in his
position group, the share of the club's minutes there the season before that were
played by men who have since left (1 for a club that did not exist the season before,
such as the Dolphins in 2023). A vacancy gives a newcomer games; it should not make him
better per minute.

Reported per direction: the probit coefficient on the vacancy (is the restriction
strong?), the Mills-ratio coefficient (the sign and size of the selection), and the
selection bias in the movers' mean rating: the mean among those observed minus the
mean the outcome equation implies for every arrival.

    python heckman_attrition.py     # writes HECKMAN_REPORT.md
"""
import os
import sqlite3
import sys

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import lee_bounds as lb
import matching_moves as mm
import rolling_backtest as rb
import sp_schema as sp

warnings.filterwarnings("ignore", category=RuntimeWarning)
BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BASE, "HECKMAN_REPORT.md")
X_COLS = ["y_prev", "cls", "age", "mins_pg", "games", "start_share"]
N_BOOT, SEED = 300, 0


def vacancy(con):
    pm = pd.read_sql("SELECT player_id, competition comp, season, round, team, position, "
                     "minutes FROM player_match_stats", con)
    pm["player_id"] = sp.normalize_player_id(pm.player_id)
    pm["group"] = pm.position.map(sp.POSITION_GROUP).fillna("Bench")
    first = (pm.sort_values("round").groupby(["player_id", "comp", "season"]).team.first())
    mins = pm.groupby(["comp", "season", "team", "group", "player_id"]).minutes.sum()
    at = set(zip(pm.comp, pm.season, pm.team, pm.player_id))
    vac = {}
    for (c, s, t, g), x in mins.groupby(level=[0, 1, 2, 3]):
        pids = x.index.get_level_values(4)
        left = np.array([(c, s + 1, t, p) not in at for p in pids])
        vac[(c, s + 1, t, g)] = float(x.values[left].sum() / x.values.sum())
    clubs_prev = set(mins.index.droplevel([3, 4]).unique())
    return first, vac, clubs_prev


def probit_fit(X, d):
    def nll(b):
        z = X @ b
        return -(d * norm.logcdf(z) + (1 - d) * norm.logcdf(-z)).sum()
    b0 = np.zeros(X.shape[1])
    r = minimize(nll, b0, method="BFGS")
    return r.x, r.hess_inv


def two_step(g):
    Xs = g[X_COLS].astype(float).fillna(g[X_COLS].median()).values
    mu, sd = Xs.mean(0), Xs.std(0)
    sd[sd == 0] = 1
    Xs = (Xs - mu) / sd
    Z = np.column_stack([np.ones(len(g)), Xs, g.vacancy.values])
    gam, cov = probit_fit(Z, g.obs.values.astype(float))
    xb = Z @ gam
    imr = norm.pdf(xb) / norm.cdf(xb)
    o = g.obs.values
    Xo = np.column_stack([np.ones(o.sum()), Xs[o], imr[o]])
    beta, *_ = np.linalg.lstsq(Xo, g.y.values[o], rcond=None)
    pred_all = np.column_stack([np.ones(len(g)), Xs]) @ beta[:-1]
    bias = float(g.y.values[o].mean() - pred_all.mean())
    return dict(gamma_vac=float(gam[-1]), z_vac=float(gam[-1] / np.sqrt(cov[-1, -1])),
                rho_sigma=float(beta[-1]), bias=bias, obs_rate=float(o.mean()),
                n=len(g))


def run():
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    D = mm.load(con)
    first, vac, clubs_prev = vacancy(con)
    con.close()
    M = lb.movers_at_risk(D, 3, "class_score")
    M["club"] = first.reindex(list(zip(M.player_id, M.target, M.season_tgt))).values
    M["vacancy"] = [vac.get((c, s, t, g), 1.0 if (c, s - 1, t) not in clubs_prev else 0.0)
                    if isinstance(t, str) else np.nan
                    for c, s, t, g in zip(M.target, M.season_tgt, M.club, M.group)]
    M = M[M.vacancy.notna()].reset_index(drop=True)
    rows = []
    for pair, g in M.groupby("pair"):
        if len(g) < 40 or g.obs.all():
            continue
        r = two_step(g)
        rng = np.random.default_rng(SEED)
        bs = []
        for _ in range(N_BOOT):
            gb = g.iloc[rng.integers(0, len(g), len(g))]
            if gb.obs.all() or not gb.obs.any():
                continue
            try:
                bs.append(two_step(gb))
            except Exception:
                continue
        bs = pd.DataFrame(bs)
        r.update(pair=pair, rho_lo=bs.rho_sigma.quantile(.025), rho_hi=bs.rho_sigma.quantile(.975),
                 bias_lo=bs.bias.quantile(.025), bias_hi=bs.bias.quantile(.975))
        rows.append(r)
    return pd.DataFrame(rows), M


def report(T, M):
    L = ["# Heckman selection for arrivals who never reach three matches", "",
         "Generated by `heckman_attrition.py`; model in its docstring. Directions with 40+ "
         "arrivals and some dropout. Intervals: 300 resamples.", "",
         "| Move | Arrivals | Reached 3 matches | Vacancy in the probit (z) | Mills-ratio "
         "coefficient [95%] | Selection bias in the movers' mean [95%] |",
         "|---|---|---|---|---|---|"]
    for r in T.sort_values("n", ascending=False).itertuples():
        L.append(f"| {r.pair} | {r.n} | {r.obs_rate:.0%} | {r.gamma_vac:+.2f} "
                 f"({r.z_vac:+.1f}) | {r.rho_sigma:+.1f} [{r.rho_lo:+.1f}, {r.rho_hi:+.1f}] | "
                 f"{r.bias:+.1f} [{r.bias_lo:+.1f}, {r.bias_hi:+.1f}] |")
    L += ["", "A vacancy z below about 2 means the exclusion restriction is weak and the "
          "correction rests on the probit's functional form alone. A positive bias means "
          "the movers who got games rate higher than all arrivals would have, so the "
          "matched shift for that direction is too high by about that much.", ""]
    strong = T[T.z_vac.abs() >= 2]
    L += [f"**Verdict.** The vacancy is a usable exclusion restriction in "
          f"{len(strong)} of {len(T)} directions"
          + (f" ({', '.join(strong.pair)})" if len(strong) else "")
          + ". Elsewhere it does not move the chance of reaching three matches, so the "
          "correction is identified only by the probit's shape and its intervals are "
          "uninformative. Heckman does not narrow what the Lee bounds say; the bounds "
          "remain the honest statement.", ""]
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    report(*run())
    sys.exit(0)
