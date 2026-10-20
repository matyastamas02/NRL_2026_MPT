# -*- coding: utf-8 -*-
"""Player and club effects together: the AKM design from labour economics.

Abowd, Kramarz and Margolis separated worker and firm effects in wages using the
workers who change firms. The same structure fits here. A season rating is
    y(player, season) = theta(player) + psi(club) + f(age) + noise
where theta is the player's own level, psi(club) what playing at that club does to a
rating, and clubs belong to competitions. Players who change club, within a competition
or across, identify psi. Because every competition's ratings are centred on its own
pool, the average psi of a competition's clubs measures how much higher the same player
rates there: the strength gap, from every move and every dual registration at once,
not only from the entry cohort.

Two things to know before reading it.
  - Limited-mobility bias (Andrews et al. 2008): with few movers per club the spread
    of the estimated club effects overstates the true spread. It is measured here by
    split samples, players split at random into halves, each half fitted alone; the
    covariance of the two halves' club effects is an unbiased estimate of the true
    variance, and their correlation says how reproducible a club's effect is.
  - It assumes moves are as good as random given the player's own level (no sorting on
    the noise). Clubs do sign players after good seasons; the model absorbs that only
    as far as theta captures it.

Data: season-only ratings (3+ matches) from one table built through 2026, the club the
player started most for that season (player_season_starts), age from date of birth.

    python akm_clubs.py        # writes AKM_REPORT.md and matching/akm_clubs.csv
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

import rating_history as rh
import rolling_backtest as rb
import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BASE, "AKM_REPORT.md")
OUT = os.path.join(BASE, "matching")
COMPS = ["NRL", "NSW", "QLD", "SL"]
N_BOOT, N_SPLIT, SEED = 200, 50, 0


def panel(con):
    _, sea, _ = rh.all_competitions(rb.DB, through=2026)
    sea["player_id"] = sp.normalize_player_id(sea.player_id)
    sea = sea[sea.n_games >= rb.MIN_GAMES]
    ps = pd.read_sql("SELECT player_id, comp, season, team FROM player_season_starts", con)
    ps["player_id"] = sp.normalize_player_id(ps.player_id)
    d = sea.merge(ps, on=["player_id", "comp", "season"], how="inner")
    dob = pd.read_sql("SELECT player_id, dob FROM players", con)
    dob["player_id"] = sp.normalize_player_id(dob.player_id)
    d["age"] = sp.age_at(sp.parse_dob(d.player_id.map(dob.set_index("player_id").dob)),
                         pd.Series(d.season.values, index=d.index)).values
    d["age"] = d.age.fillna(d.age.median())
    d["club"] = d.comp + "|" + d.team
    return d[["player_id", "comp", "season", "club", "class_score", "n_games", "age"]]


def connected(d):
    """Keep the largest set of players and clubs joined by moves."""
    pi = pd.factorize(d.player_id)[0]
    ci = pd.factorize(d.club)[0]
    P, C = pi.max() + 1, ci.max() + 1
    A = csr_matrix((np.ones(len(d)), (pi, P + ci)), shape=(P + C, P + C))
    _, lab = connected_components(A + A.T, directed=False)
    big = np.bincount(lab[pi]).argmax()
    return d[lab[pi] == big].reset_index(drop=True)


def fit(d, weight=False):
    """theta, psi and age terms by sparse least squares; psi demeaned within the NRL."""
    pi, players = pd.factorize(d.player_id)
    ci, clubs = pd.factorize(d.club)
    n = len(d)
    Xp = csr_matrix((np.ones(n), (np.arange(n), pi)), shape=(n, len(players)))
    keep = np.arange(1, len(clubs))                       # club 0 is the reference
    Xc = csr_matrix((np.ones(n), (np.arange(n), ci)), shape=(n, len(clubs)))[:, keep]
    a = (d.age.values - 27.0) / 5.0
    Xa = csr_matrix(np.column_stack([a ** 2, a ** 3]))   # the level goes into theta
    X = hstack([Xp, Xc, Xa]).tocsr()
    y = d.class_score.values.astype(float)
    if weight:
        w = np.sqrt(d.n_games.values)
        X = X.multiply(w[:, None]).tocsr()
        y = y * w
    b = lsqr(X, y, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    psi = np.concatenate([[0.0], b[len(players):len(players) + len(keep)]])
    psi = pd.Series(psi, index=clubs)
    comp = psi.index.str.split("|").str[0]
    psi = psi - psi[comp == "NRL"].mean()
    theta = pd.Series(b[:len(players)], index=players)
    resid = y - X @ b
    return psi, theta, resid


def league_gaps(psi, d):
    """Player-season-weighted mean club effect per competition, NRL = 0."""
    n = d.club.value_counts()
    comp = psi.index.str.split("|").str[0]
    out = {}
    for c in COMPS:
        m = comp == c
        out[c] = float(np.average(psi[m], weights=n.reindex(psi.index[m]).fillna(0)))
    return {c: v - out["NRL"] for c, v in out.items()}


def primary_only(d, con):
    """One row per player-season: his competition with the most minutes. Without dual
    registrations, the gaps are identified only by moves between seasons."""
    pm = pd.read_sql("SELECT player_id, competition comp, season, sum(minutes) m FROM "
                     "player_match_stats GROUP BY 1, 2, 3", con)
    pm["player_id"] = sp.normalize_player_id(pm.player_id)
    top = (pm.sort_values("m", ascending=False).drop_duplicates(["player_id", "season"])
           [["player_id", "season", "comp"]])
    return connected(d.merge(top, on=["player_id", "season", "comp"], how="inner"))


SHRINK = None      # per-competition reliability of a club effect, set in run()


def rolling_adjustment(d, cache):
    """Does subtracting the source club's effect help the translation? Each landing
    season t: club effects fitted on seasons up to t-1, the source rating of every
    training pair and entrant adjusted by its source club's effect (shrunk by the
    competition's reliability), and the app's line refitted on the adjusted ratings."""
    import pickle
    import frozen_2026 as fz
    D = pickle.load(open(cache, "rb"))
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    ps = pd.read_sql("SELECT player_id, comp, season, team FROM player_season_starts", con)
    con.close()
    ps["player_id"] = sp.normalize_player_id(ps.player_id)
    club = dict(zip(zip(ps.player_id, ps.comp, ps.season), ps.comp + "|" + ps.team))
    out = []
    for t in (2023, 2024, 2025, 2026):
        dt = connected(d[d.season <= t - 1])
        psi = fit(dt)[0]
        comp = psi.index.str.split("|").str[0]
        psi = psi - psi.groupby(comp).transform("mean")          # within-competition
        psi = psi * comp.map(SHRINK).fillna(0).values

        def adj(f):
            f = f.copy()
            f["player_id"] = sp.normalize_player_id(f.player_id.astype(str))
            k = [club.get((p, c, y)) for p, c, y in zip(f.player_id, f.source, f.season_src)]
            f["psi_src"] = [psi.get(x, 0.0) if x else 0.0 for x in k]
            return f
        tr = adj(D["pairs"][t])
        tr["pair"] = tr.source + "->" + tr.target
        ev = adj(D["cohort"][D["cohort"].season_tgt == t])
        s0 = fz.predict_lines(fz.fit_lines(tr), ev)[0]
        tra = tr.assign(class_source=tr.class_source - tr.psi_src)
        s1 = fz.predict_lines(fz.fit_lines(tra), ev.assign(
            class_source=ev.class_source - ev.psi_src))[0]
        out.append(ev.assign(S0=s0, S_club=s1))
    return pd.concat(out, ignore_index=True)


def run():
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    d = panel(con)
    con.close()
    d0 = len(d)
    d = connected(d)
    psi, theta, resid = fit(d)
    psi_w, _, _ = fit(d, weight=True)
    gaps, gaps_w = league_gaps(psi, d), league_gaps(psi_w, d)
    # bootstrap over players for the gaps
    rng = np.random.default_rng(SEED)
    players = d.player_id.unique()
    at = d.groupby("player_id").indices
    bs = []
    for _ in range(N_BOOT):
        pick = rng.choice(players, players.size)
        idx = np.concatenate([at[p] for p in pick])
        db = d.iloc[idx].copy()
        # resampled players become distinct people
        db["player_id"] = np.repeat(np.arange(pick.size), [len(at[p]) for p in pick])
        db = connected(db)
        bs.append(league_gaps(fit(db)[0], db))
    bs = pd.DataFrame(bs)
    # split samples for the club effects
    sp_rows = []
    for k in range(N_SPLIT):
        r = np.random.default_rng(SEED + k + 1)
        half = set(r.choice(players, players.size // 2, replace=False))
        a = d[d.player_id.isin(half)]
        b = d[~d.player_id.isin(half)]
        a, b = connected(a), connected(b)
        pa, pb = fit(a)[0], fit(b)[0]
        common = pa.index.intersection(pb.index)
        for c in COMPS:
            m = common[common.str.startswith(c + "|")]
            if len(m) < 5:
                continue
            xa, xb = pa[m] - pa[m].mean(), pb[m] - pb[m].mean()
            sp_rows.append(dict(comp=c, cov=float(np.mean(xa * xb)),
                                var_a=float(np.var(xa)), corr=float(np.corrcoef(xa, xb)[0, 1])))
    S = pd.DataFrame(sp_rows)
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    dp = primary_only(d, con)
    con.close()
    gaps_p = league_gaps(fit(dp)[0], dp)
    global SHRINK
    r = S.groupby("comp")["corr"].mean().clip(lower=0)
    SHRINK = (2 * r / (1 + r)).to_dict()           # Spearman-Brown: whole-sample reliability
    cache = os.environ.get("POST2026_CACHE")
    roll = rolling_adjustment(d, cache) if cache and os.path.exists(cache) else None
    return d, d0, psi, psi_w, theta, resid, gaps, gaps_w, bs, S, gaps_p, dp, roll


def report(d, d0, psi, psi_w, theta, resid, gaps, gaps_w, bs, S, gaps_p, dp, roll):
    os.makedirs(OUT, exist_ok=True)
    n = d.club.value_counts()
    moves = (d.sort_values("season").groupby("player_id").club.apply(
        lambda s: int((s != s.shift()).sum() - 1)))
    tab = pd.DataFrame({"psi": psi, "psi_weighted": psi_w,
                        "player_seasons": n.reindex(psi.index)}).sort_index()
    tab.to_csv(os.path.join(OUT, "akm_clubs.csv"))
    var_y = np.var(d.class_score)
    th = d.player_id.map(theta)
    ps = d.club.map(psi)
    L = ["# Player and club effects (AKM)", "",
         "Generated by `akm_clubs.py`; see its docstring for the model and its "
         "assumptions. Season-only peer scores, 3+ matches, 2020/21–2026.", "",
         f"Player-seasons: {d0:,}; in the largest connected set: {len(d):,} "
         f"({d.player_id.nunique():,} players, {d.club.nunique()} clubs). Players who "
         f"changed club at least once: {int((moves > 0).sum()):,}.", "",
         "## How much higher the same player rates, by competition", "",
         "Mean club effect of the competition's clubs (player-season weighted), NRL = 0. "
         "Positive = the same player rates higher there, i.e. a weaker field.", "",
         "| Competition | Unweighted fit [95%] | Weighted by matches | Without dual "
         "registrations |", "|---|---|---|---|"]
    for c in COMPS:
        lo, hi = (bs[c].quantile(.025), bs[c].quantile(.975)) if c != "NRL" else (0, 0)
        L.append(f"| {c} | {gaps[c]:+.1f} [{lo:+.1f}, {hi:+.1f}] | {gaps_w[c]:+.1f} | "
                 f"{gaps_p[c]:+.1f} |")
    L += ["", f"Without dual registrations: {len(dp):,} player-seasons, each player only in "
          "the competition he played most minutes in that season, so only moves between "
          "seasons identify the gaps."]
    L += ["", "For comparison, the strength from matched movers (`MATCHING_REPORT.md`) was "
          "NSW −3.2 to −3.7, QLD −6.8 to −8.4, SL −9.0 to −12.0 (the sign there is "
          "strength; here it is the rating gain, so the signs flip).", ""]
    # variance decomposition
    L += ["## What explains a season rating", "",
          f"- variance of season ratings: {var_y:.0f}",
          f"- of the player effect: {np.var(th):.0f}; of the club effect: {np.var(ps):.0f}; "
          f"covariance x2: {2 * np.cov(th, ps)[0, 1]:+.0f}; residual: {np.var(resid):.0f}", ""]
    L += ["## Are club effects real or noise?", "",
          "Players split at random into halves, each half fitted alone, 50 times. Within a "
          "competition, the correlation of the two halves' club effects is how "
          "reproducible a club's effect is; the covariance is the unbiased estimate of the "
          "true variance (the split-sample correction for limited mobility).", "",
          "| Competition | Split-half correlation | True SD of club effects | Naive SD |",
          "|---|---|---|---|"]
    for c, g in S.groupby("comp"):
        L.append(f"| {c} | {g['corr'].mean():+.2f} | {np.sqrt(max(g['cov'].mean(), 0)):.1f} "
                 f"| {np.sqrt(g.var_a.mean()):.1f} |")
    if roll is not None:
        import frozen_2026 as fz
        L += ["", "## Does a club-adjusted source rating translate better?", "",
              "Rolling, each landing season from club effects fitted on the seasons before; "
              "the source rating minus the source club's effect, shrunk by the "
              "competition's reliability (" + ", ".join(f"{k} {v:.2f}" for k, v in
                                                       sorted(SHRINK.items())) + "). "
              "Δ > 0 means the adjusted line is closer.", "",
              "| Entrants | n | MAE, app line | MAE, club-adjusted | Δ [95%] |",
              "|---|---|---|---|---|"]
        for lab, g in (("into SL from NRL/NSW/QLD",
                        roll[roll.pair.isin({"NRL->SL", "NSW->SL", "QLD->SL"})]),
                       ("all directions", roll)):
            dl, lo, hi = fz.paired_bootstrap(g.class_target.values, g.S0.values,
                                             g.S_club.values, g.player_id.values)
            L.append(f"| {lab} | {len(g)} | {np.abs(g.class_target - g.S0).mean():.2f} | "
                     f"{np.abs(g.class_target - g.S_club).mean():.2f} | {dl:+.2f} "
                     f"[{lo:+.2f}, {hi:+.2f}] |")
    L += ["", "## Club effects, Super League", "",
          "Within Super League, demeaned; positive = players rate higher at this club than "
          "their own level predicts.", "", "| Club | Effect | Player-seasons |",
          "|---|---|---|"]
    t = tab[tab.index.str.startswith("SL|")]
    t = t.assign(e=t.psi - t.psi.mean()).sort_values("e", ascending=False)
    for k, r in t.iterrows():
        L.append(f"| {k.split('|')[1]} | {r.e:+.1f} | {int(r.player_seasons)} |")
    L.append("")
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    report(*run())
    sys.exit(0)
