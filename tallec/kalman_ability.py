# -*- coding: utf-8 -*-
"""A state-space model of a player's level, against the recency-weighted Class.

Class is a weighted average of a player's match composites (this season 1.0, last 0.7,
and so on) shrunk toward the pool once. The rating engine's own docstring names the
better model and says it was not built: a level that moves, a player-season random
effect at minimum, a state-space formulation properly. This is that formulation.

    level(k)   = level(k-1) + drift,     drift ~ N(0, q)  between matches
                                          + N(0, Q) more at a season break
    composite(k) = level(k) + noise,     noise ~ N(0, sigma2)
    level(0) ~ N(pool mean, tau2)

filtered match by match (Kalman), within each competition, on the same per-season
standardised composites the engine uses. sigma2 and tau2 are moment estimates and q, Q
are chosen by a grid search, all on seasons up to 2022 only, so nothing tuned sees the
seasons it is scored on. The output is the filtered level at the end of each season and
its variance: a rating with its own uncertainty.

Scored two ways, each season predicted from the seasons before it:
  persistence  players who stayed in a competition: next season's rating from this
               season's level (Kalman) or from Class, through a line fitted on earlier
               seasons
  translation  the app's line with the source Class replaced by the Kalman level

    python kalman_ability.py      # writes KALMAN_REPORT.md
"""
import itertools
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd

import frozen_2026 as fz
import rating_history as rh
import rolling_backtest as rb
import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BASE, "KALMAN_REPORT.md")
COMPS = ["NRL", "NSW", "QLD", "SL"]
TUNE_THROUGH = 2022
CACHE = os.environ.get("POST2026_CACHE")


def composites(con, comp):
    hist = rh.load(con, comp)
    force, _ = rh._mode(hist)
    pm = rh.composites(hist, comp, force)
    pm = pm[pm.ratable].copy()
    pm["player_id"] = sp.normalize_player_id(pm.player_id)
    return pm.sort_values(["player_id", "season", "round"])[
        ["player_id", "season", "round", "composite"]]


def moments(pm):
    """sigma2: within player-season variance; tau2: between player-season means."""
    g = pm.groupby(["player_id", "season"]).composite
    n, mu, var = g.size(), g.mean(), g.var()
    sigma2 = float(np.average(var[n > 2], weights=n[n > 2] - 1))
    tau2 = float(max(np.var(mu[n >= 5]) - sigma2 / np.mean(n[n >= 5]), 1e-6))
    return sigma2, tau2, float(pm.composite.mean())


def kalman(pm, sigma2, tau2, m0, q, Q):
    """Filtered level and variance at the end of each player-season."""
    out = []
    for pid, g in pm.groupby("player_id", sort=False):
        m, P, last = m0, tau2, None
        for s, gs in g.groupby("season", sort=True):
            if last is not None:
                P += Q
            for y in gs.composite.values:
                P += q
                K = P / (P + sigma2)
                m += K * (y - m)
                P *= (1 - K)
            out.append((pid, int(s), m, P, len(gs)))
            last = s
    return pd.DataFrame(out, columns=["player_id", "season", "level", "level_var", "n"])


def season_means(pm):
    return (pm.groupby(["player_id", "season"]).composite.agg(["mean", "size"])
              .reset_index().rename(columns={"mean": "y_mean", "size": "n"}))


def tune(pm, sigma2, tau2, m0):
    """q, Q by one-season-ahead squared error of the season mean, seasons <= 2022."""
    sm = season_means(pm)
    train = pm[pm.season <= TUNE_THROUGH]
    best = None
    for qf, Qf in itertools.product([0, 0.002, 0.005, 0.01, 0.02],
                                    [0, 0.1, 0.25, 0.5, 1.0]):
        k = kalman(train, sigma2, tau2, m0, qf * sigma2, Qf * tau2)
        j = k.merge(sm.assign(season=sm.season - 1), on=["player_id", "season"],
                    suffixes=("", "_next"))
        j = j[(j.season + 1 <= TUNE_THROUGH) & (j.n_next >= 5)]
        err = float(np.mean((j.y_mean - j.level) ** 2))
        if best is None or err < best[0]:
            best = (err, qf, Qf)
    return best


def run():
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    levels, params = [], {}
    for c in COMPS:
        pm = composites(con, c)
        s2, t2, m0 = moments(pm[pm.season <= TUNE_THROUGH])
        err, qf, Qf = tune(pm, s2, t2, m0)
        k = kalman(pm, s2, t2, m0, qf * s2, Qf * t2)
        k["comp"] = c
        levels.append(k)
        params[c] = dict(sigma2=s2, tau2=t2, q_share=qf, Q_share=Qf)
    con.close()
    K = pd.concat(levels, ignore_index=True)
    _, sea, _ = rh.all_competitions(rb.DB, through=2026)
    cum, sea = (d.assign(player_id=sp.normalize_player_id(d.player_id))
                for d in rh.all_competitions(rb.DB, through=2026)[:2])
    return K, params, cum, sea


def persistence(K, cum, sea):
    """Stayers: next season's rating from this season's level or Class."""
    s = sea[sea.n_games >= rb.MIN_GAMES][["player_id", "comp", "season", "class_score"]]
    c = cum[["player_id", "comp", "season", "class_score"]].rename(
        columns={"class_score": "cls"})
    j = (K.merge(c, on=["player_id", "comp", "season"])
          .merge(s.assign(season=s.season - 1).rename(columns={"class_score": "y_next"}),
                 on=["player_id", "comp", "season"]))
    rows = []
    for t in range(2023, 2027):
        tr, te = j[j.season + 1 < t], j[j.season + 1 == t]
        if te.empty:
            continue
        for col in ("cls", "level"):
            for comp, g in te.groupby("comp"):
                tc = tr[tr.comp == comp]
                b = np.polyfit(tc[col], tc.y_next, 1)
                te.loc[g.index, f"p_{col}"] = np.polyval(b, g[col])
        rows.append(te)
    return pd.concat(rows)


def translation(K):
    D = pickle.load(open(CACHE, "rb"))
    lev = K.rename(columns={"comp": "source", "season": "season_src"})[
        ["player_id", "source", "season_src", "level"]]
    out = []
    for t in (2023, 2024, 2025, 2026):
        tr = D["pairs"][t].copy()
        tr["player_id"] = sp.normalize_player_id(tr.player_id.astype(str))
        tr["pair"] = tr.source + "->" + tr.target
        ev = D["cohort"][D["cohort"].season_tgt == t].copy()
        tr = tr.merge(lev, on=["player_id", "source", "season_src"], how="left")
        ev = ev.merge(lev, on=["player_id", "source", "season_src"], how="left")
        tr = tr[tr.level.notna()]
        ev["S0"] = fz.predict_lines(fz.fit_lines(tr), ev)[0]
        ok = ev.level.notna()
        ev["S_kal"] = ev.S0
        ev.loc[ok, "S_kal"] = fz.predict_lines(
            fz.fit_lines(tr.assign(class_source=tr.level)),
            ev[ok].assign(class_source=ev.level[ok]))[0]
        out.append(ev)
    return pd.concat(out, ignore_index=True)


def report(K, params, cum, sea):
    L = ["# A state-space model of player level", "",
         "Generated by `kalman_ability.py`; model and tuning in its docstring.", "",
         "## Parameters (fitted on seasons up to 2022)", "",
         "| Competition | Match noise σ² | Between-player τ² | Drift per match (×σ²) | "
         "Extra at a season break (×τ²) |", "|---|---|---|---|---|"]
    for c, p in params.items():
        L.append(f"| {c} | {p['sigma2']:.3f} | {p['tau2']:.4f} | {p['q_share']:.3f} | "
                 f"{p['Q_share']:.2f} |")
    P = persistence(K, cum, sea)
    L += ["", "## Persistence: stayers' next season, rolling 2023–2026", "",
          "| Competition | Player-seasons | MAE from Class | MAE from the Kalman level | "
          "Δ [95%] |", "|---|---|---|---|---|"]
    for comp, g in list(P.groupby("comp")) + [("all", P)]:
        d, lo, hi = fz.paired_bootstrap(g.y_next.values, g.p_cls.values, g.p_level.values,
                                        g.player_id.values)
        L.append(f"| {comp} | {len(g)} | {np.abs(g.y_next - g.p_cls).mean():.2f} | "
                 f"{np.abs(g.y_next - g.p_level).mean():.2f} | {d:+.2f} [{lo:+.2f}, "
                 f"{hi:+.2f}] |")
    if CACHE and os.path.exists(CACHE):
        T = translation(K)
        L += ["", "## Translation: the app's line on the Kalman level", "",
              "| Entrants | n | MAE, Class | MAE, Kalman level | Δ [95%] |",
              "|---|---|---|---|---|"]
        for lab, g in (("into SL from NRL/NSW/QLD",
                        T[T.pair.isin({"NRL->SL", "NSW->SL", "QLD->SL"})]),
                       ("all directions", T)):
            d, lo, hi = fz.paired_bootstrap(g.class_target.values, g.S0.values,
                                            g.S_kal.values, g.player_id.values)
            L.append(f"| {lab} | {len(g)} | {np.abs(g.class_target - g.S0).mean():.2f} | "
                     f"{np.abs(g.class_target - g.S_kal).mean():.2f} | {d:+.2f} "
                     f"[{lo:+.2f}, {hi:+.2f}] |")
    L += ["", "Δ > 0 means the Kalman level is closer.", ""]
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    report(*run())
    sys.exit(0)
