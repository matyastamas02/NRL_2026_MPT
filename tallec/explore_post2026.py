# -*- coding: utf-8 -*-
"""Three things tried after the 2026 test. Exploratory: none of it can change the 2026
result, and anything worth keeping goes into the 2027 specification before 2027 is seen.

1. Component translation. The line translates one number, the Class composite. Here the
   nine per-minute rates the composite is built from are translated one by one: each
   rate's player mean is shrunk by its OWN reliability (tries per minute is far noisier
   than tackles per minute, the composite shrinks them as one), each rate gets its own
   persistence slope across a move, and the composite weights are applied after.
     C-shrink   per-rate shrinkage only, composite weights unchanged
     C-persist  per-rate shrinkage and per-rate persistence slopes
     C-ridge    the nine weighted rates with a ridge fit, alpha by grouped CV
   Each index then goes through exactly the line the app uses (own line per direction
   with 25+ pairs, else the line over all pairs), so the only difference from the shipped
   line is what is fed into it.

3. Partial pooling and the ranges. The direction lines are pulled toward the line over
   all pairs by empirical Bayes instead of the hard 25-pair switch. And the ranges are
   checked against what happened: the app's reference band (forecast +- 1.96 residual
   SD, nominally 95%), an empirical-quantile band from the training residuals (50% and
   80%), and the earlier-entrants card's middle half and 10-90% range.

4. A decision target. Instead of the rating itself: did the entrant finish in the top
   third of his new competition that season (season-only rating against every rated
   player there), and did he become a regular starter (started half or more of his
   club's matches). Scored by AUC, by a shortlist hit rate, and by the Brier score of a
   probability calibrated on the training pairs.

Every forecast for landing season t uses only what was known at the end of t-1: the
pairs, ratings and positions of `frozen_2026.knowledge(t-1)` and the entry cohort built
the way the frozen test builds it. The per-rate z-scores are standardised inside each
competition-season pool, exactly as the rating engine does, so no season sees another.
Nothing in frozen_2026.FROZEN_FILES is changed; this only reads them.

    python explore_post2026.py        # writes POST2026_EXPLORE.md and post2026_explore/
"""
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import frozen_2026 as fz
import player_rating_engine as pre
import rating_history as rh
import rolling_backtest as rb
import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "post2026_explore")
REPORT = os.path.join(BASE, "POST2026_EXPLORE.md")
SEASONS = [2023, 2024, 2025, 2026]
LANDINGS = list(range(2021, 2027))
H1 = {"NRL->SL", "NSW->SL", "QLD->SL"}
RATES = pre.RATE_ORDER
ZC = [f"z_{r}" for r in RATES]
MIN_DIR = 5                 # pairs a direction needs before empirical Bayes uses it
ALPHAS = [0.3, 1, 3, 10, 30, 100, 300]
N_BOOT, SEED = 2000, 0
CACHE = os.environ.get("POST2026_CACHE")


# ── per-rate z-scores, one row per player-match ──────────────────────────────────
def rate_rows(con, modes=None):
    """`modes`: {competition: rating basis} as decided at a cut-off. Without it the
    basis is decided from the whole history, which for the Queensland Cup is not the
    basis known before 2025 (an independent review found the first version did this)."""
    out, worst = [], 0.0
    for comp in ("NRL", "SL", "NSW", "QLD"):
        hist = rh.load(con, comp)
        force = modes[comp] if modes is not None else rh._mode(hist)[0]
        for s in sorted(hist.season.dropna().unique()):
            part = hist[hist.season == s]
            if len(part) < rh.MIN_POOL:
                continue
            eng = pre.PlayerRatingEngine(comp, force_mode=force)
            eng._fit_standardization(part)
            df = eng._transform(part)
            z = np.full((len(df), len(RATES)), np.nan)
            grp = df["group"].values
            for g in pd.unique(grp):
                m = grp == g
                params = eng.norm.get(g, eng.norm.get("Bench"))
                for j, r in enumerate(RATES):
                    mu, sd = params[r]
                    if r in eng.available and np.isfinite(sd) and sd > 0:
                        z[m, j] = (df.loc[m, r].values - mu) / sd
            # the composite rebuilt from the kept z-scores must be the engine's
            wmap = {g: eng._weights(g) for g in pd.unique(grp)}
            w = np.vstack([wmap[g] for g in grp])
            worst = max(worst, float(np.abs((np.nan_to_num(z) * w).sum(1)
                                            - df["composite"].values).max()))
            f = pd.DataFrame(z, columns=ZC)
            f["player_id"] = sp.normalize_player_id(df.player_id).values
            f["comp"], f["season"] = comp, int(s)
            f["group"], f["ratable"] = grp, df["ratable"].values
            out.append(f)
    return pd.concat(out, ignore_index=True), worst


def _anova(x, ids):
    """sigma^2, tau^2, grand mean of one rate, the engine's one-way ANOVA."""
    d = pd.DataFrame({"x": x, "id": ids}).dropna()
    g = d.groupby("id").x
    n, mu = g.size(), g.mean()
    N, k = len(d), len(n)
    if k < 3 or N <= k:
        return np.nan, 0.0, float(d.x.mean()) if N else 0.0
    grand = d.x.mean()
    msw = ((d.x - d.id.map(mu)) ** 2).sum() / (N - k)
    msb = (n * (mu - grand) ** 2).sum() / (k - 1)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    return float(msw), float(max(0.0, (msb - msw) / n0)), float(grand)


def aggregate(rows, comp, S, cumulative):
    """Per player: each rate's mean (season-weighted when cumulative), the same mean
    shrunk by that rate's own reliability, and the modal peer group."""
    r = rows[(rows.comp == comp) & rows.ratable
             & ((rows.season <= S) if cumulative else (rows.season == S))]
    if r.empty:
        return None
    wt = ((S - r.season).map(pre.season_weight).values if cumulative
          else np.ones(len(r)))
    Z = r[ZC].values
    have = ~np.isnan(Z)
    pid = r.player_id.values
    num = pd.DataFrame(np.where(have, Z, 0) * wt[:, None], columns=ZC).groupby(pid).sum()
    den = pd.DataFrame(have * wt[:, None], columns=ZC).groupby(pid).sum()
    mean = num / den.where(den > 0)
    res = pd.DataFrame(index=mean.index)
    for c in ZC:
        sig2, tau2, gm = _anova(r[c].values, pid)
        b = tau2 / (tau2 + sig2 / den[c].where(den[c] > 0)) if np.isfinite(sig2) else 0.0
        res["m_" + c] = mean[c]
        res["s_" + c] = gm + b * (mean[c] - gm)
    res["group"] = r.groupby("player_id").group.agg(lambda s: s.mode().iloc[0])
    res["comp"], res["season"] = comp, int(S)
    return res.rename_axis("player_id").reset_index()


def rate_features(rows):
    cum, one = [], []
    for comp in ("NRL", "SL", "NSW", "QLD"):
        for S in sorted(rows[rows.comp == comp].season.unique()):
            a = aggregate(rows, comp, S, True)
            b = aggregate(rows, comp, S, False)
            if a is not None:
                cum.append(a)
            if b is not None:
                one.append(b)
    return pd.concat(cum, ignore_index=True), pd.concat(one, ignore_index=True)


def attach(frame, cum, one, with_target):
    f = frame.drop(columns=[c for c in ("group",) if c in frame.columns])
    f["player_id"] = sp.normalize_player_id(f.player_id)
    src = cum.rename(columns={"comp": "source", "season": "season_src"})
    f = f.merge(src[["player_id", "source", "season_src", "group"]
                    + ["s_" + c for c in ZC]], on=["player_id", "source", "season_src"],
                how="left")
    if with_target:
        tgt = one.rename(columns={"comp": "target", "season": "season_tgt",
                                  "group": "group_tgt"})
        f = f.merge(tgt[["player_id", "target", "season_tgt"] + ["m_" + c for c in ZC]],
                    on=["player_id", "target", "season_tgt"], how="left")
    return f


def _weights(f):
    """Composite weights of each row's peer group, renormalised over the rates it has."""
    raw = np.vstack([pre.POSITION_WEIGHTS.get(g, pre.POSITION_WEIGHTS["Bench"])
                     for g in f.group.fillna("Bench")])
    S = f[["s_" + c for c in ZC]].values
    w = np.where(np.isnan(S), 0.0, raw)
    tot = w.sum(1, keepdims=True)
    return np.where(tot > 0, w / np.where(tot > 0, tot, 1), 0.0), np.nan_to_num(S)


def persistence(train, default="median"):
    """Each rate's slope from its shrunk source mean to its raw target mean, all pairs.

    A rate no training pair records on both sides (post-contact metres before 2025) has
    no slope. The first run gave it 1.0, more weight than any measured rate; `median`
    gives it the median of the measured slopes instead. Both are reported."""
    b = np.full(len(ZC), np.nan)
    for j, c in enumerate(ZC):
        x, y = train["s_" + c], train["m_" + c]
        ok = x.notna() & y.notna()
        if ok.sum() >= 30 and x[ok].std() > 0:
            b[j] = np.polyfit(x[ok], y[ok], 1)[0]
    if default == "drop":
        return b                       # NaN: the rate is left out on both sides
    fill = np.nanmedian(b) if default == "median" else 1.0
    return np.where(np.isnan(b), fill, b)


def index(f, b=None):
    f = f.copy()
    if b is not None and np.isnan(b).any():
        # a rate without a slope is dropped from training pairs and entrants alike, so
        # the line is not fitted on an index without it and applied to one with it
        for j in np.flatnonzero(np.isnan(b)):
            f["s_" + ZC[j]] = np.nan
        b = np.nan_to_num(b)
    w, S = _weights(f)
    return (w * S * (1.0 if b is None else b)).sum(1)


def ridge_fit(train):
    w, S = _weights(train)
    X = np.column_stack([w * S, pd.get_dummies(train.pair).values.astype(float)])
    y = train.class_target.values
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Xs = (X - mu) / sd
    gk = GroupKFold(n_splits=5)
    best = None
    for a in ALPHAS:
        err = []
        for tr, te in gk.split(Xs, y, train.player_id.values):
            m = Ridge(alpha=a).fit(Xs[tr], y[tr])
            err.append(np.abs(m.predict(Xs[te]) - y[te]).mean())
        if best is None or np.mean(err) < best[0]:
            best = (np.mean(err), a)
    cols = sorted(train.pair.unique())
    return dict(model=Ridge(alpha=best[1]).fit(Xs, y), mu=mu, sd=sd, cols=cols,
                alpha=best[1])


def ridge_predict(r, f):
    w, S = _weights(f)
    D = np.column_stack([(f.pair == c).values.astype(float) for c in r["cols"]])
    X = (np.column_stack([w * S, D]) - r["mu"]) / r["sd"]
    return np.clip(r["model"].predict(X), 0, 100)


# ── 3: empirical-Bayes direction lines, residual bands, the card ────────────────
def eb_lines(train):
    x, y = train.class_source.values, train.class_target.values
    m = x.mean()
    B0, A0 = np.polyfit(x - m, y, 1)
    rows = {}
    for d, g in train.groupby("pair"):
        if len(g) < MIN_DIR or g.class_source.std() == 0:
            continue
        xc, yy = g.class_source.values - m, g.class_target.values
        b, a = np.polyfit(xc, yy, 1)
        s2 = ((yy - a - b * xc) ** 2).sum() / max(len(g) - 2, 1)
        sxx = ((xc - xc.mean()) ** 2).sum()
        rows[d] = dict(a=a, b=b, va=s2 * (1 / len(g) + xc.mean() ** 2 / sxx), vb=s2 / sxx)
    t = pd.DataFrame(rows).T
    tau_a = max(0.0, t.a.var() - t.va.mean()) if len(t) > 2 else 0.0
    tau_b = max(0.0, t.b.var() - t.vb.mean()) if len(t) > 2 else 0.0
    t["a_eb"] = A0 + tau_a / (tau_a + t.va) * (t.a - A0)
    t["b_eb"] = B0 + tau_b / (tau_b + t.vb) * (t.b - B0)
    return dict(m=m, A0=A0, B0=B0, t=t)


def eb_predict(e, f):
    a = f.pair.map(e["t"].a_eb).fillna(e["A0"]).values
    b = f.pair.map(e["t"].b_eb).fillna(e["B0"]).values
    return np.clip(a + b * (f.class_source.values - e["m"]), 0, 100)


def residual_bands(train, model):
    """For each line the app would use: residual SD and quantiles on its training rows."""
    pred, basis = fz.predict_lines(model, train)
    res = train.class_target.values - pred
    key = np.where(basis == "direction", train.pair.values, "_pooled")
    pooled_res = train.class_target.values - fz.predict_lines(
        dict(own={}, pooled=model["pooled"]), train)[0]
    out = {"_pooled": pooled_res}
    for k in set(key) - {"_pooled"}:
        out[k] = res[key == k]
    return {k: dict(sd=float(np.std(v, ddof=2)),
                    q=np.percentile(v, [10, 25, 75, 90])) for k, v in out.items()}


def card(cohort, row):
    """predict_translation.comparables, on the cohort that landed before this season."""
    p = cohort[(cohort.pair == row.pair) & (cohort.player_id != row.player_id)]
    for window in (7.5, 15.0):
        near = p[(p.class_source - row.class_source).abs() <= window].assign(
            gap=lambda d: (d.class_source - row.class_source).abs())
        near = near.sort_values(["gap", "season_tgt", "player_id"]).drop_duplicates(
            "player_id")
        same = near[near.group_pos == row.group_pos]
        rows = same if len(same) >= 8 else near if len(near) >= 8 else None
        if rows is not None:
            q = rows.class_target.quantile([.1, .25, .75, .9]).values
            return len(rows), q
    return 0, None


# ── 4: outcomes ─────────────────────────────────────────────────────────────────
def starts(con):
    pms = pd.read_sql("SELECT player_id, competition comp, season, round, team, position, "
                      "position_source, minutes FROM player_match_stats", con)
    raw = pd.read_sql('SELECT player_id, "Competition" comp, "Season" season, '
                      '"Round" round, "Team" team, "Interchange In - All" i_in, '
                      '"Interchange Out - All" i_out FROM player_match_raw', con)
    for d in (pms, raw):
        d["player_id"] = sp.normalize_player_id(d.player_id)
    pm = pms.merge(raw, on=["player_id", "comp", "season", "round", "team"], how="left")
    START = {(0, 0), (0, 1), (1, 2), (2, 3)}
    BENCH = {(1, 0), (2, 1), (3, 2), (4, 3)}
    key = zip(pm.i_in.fillna(-1).astype(int), pm.i_out.fillna(-1).astype(int))
    rule = np.array([1.0 if k in START else 0.0 if k in BENCH else np.nan for k in key])
    rule = np.where(np.isnan(rule), (pm.minutes >= 40).astype(float), rule)
    pm["started"] = np.where(pm.position_source == "match",
                             (pm.position != "Interchange").astype(float), rule)
    team_n = pm.groupby(["comp", "season", "team"])["round"].nunique()
    ps = pm.groupby(["player_id", "comp", "season"]).agg(
        team=("team", lambda s: s.mode().iloc[0]), starts=("started", "sum"),
        games=("round", "size"), mins=("minutes", "mean"))
    ps["team_games"] = team_n.reindex(list(zip(ps.index.get_level_values(1),
                                               ps.index.get_level_values(2), ps.team))).values
    ps["starter"] = (ps.starts >= 0.5 * ps.team_games).astype(float)
    ps["start_share"] = ps.starts / ps.games
    return ps


def label(f, ps, thr):
    f = f.copy()
    t = ps.reindex(list(zip(f.player_id, f.target, f.season_tgt)))
    s = ps.reindex(list(zip(f.player_id, f.source, f.season_src)))
    f["starter"] = t.starter.values
    f["src_start_share"], f["src_mins"] = s.start_share.values, s.mins.values
    f["top3"] = (f.class_target.values >= np.array(
        [thr.get((c, y), np.nan) for c, y in zip(f.target, f.season_tgt)])).astype(float)
    return f


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = ~(np.isnan(y) | np.isnan(s))
    y, s = y[ok], s[ok]
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if not n1 or not n0:
        return np.nan
    r = pd.Series(s).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit_fit(X, y):
    from sklearn.linear_model import LogisticRegression
    ok = ~np.isnan(X).any(1) & ~np.isnan(y)
    mu, sd = X[ok].mean(0), X[ok].std(0)
    sd[sd == 0] = 1
    m = LogisticRegression(C=1.0).fit((X[ok] - mu) / sd, y[ok])

    def predict(Z):
        Z = np.asarray(Z, float)
        out = np.full(len(Z), np.nan)
        good = ~np.isnan(Z).any(1)
        if good.any():
            out[good] = m.predict_proba((Z[good] - mu) / sd)[:, 1]
        return out
    return predict


def boot(ids, fn, n=N_BOOT):
    """Point value and 95% player-cluster interval of fn(index array)."""
    ids = np.asarray(ids)
    u = np.array(sorted(pd.unique(ids)))
    at = {p: np.flatnonzero(ids == p) for p in u}
    rng = np.random.default_rng(SEED)
    v = [fn(np.concatenate([at[p] for p in rng.choice(u, u.size)])) for _ in range(n)]
    v = np.array([x for x in v if np.isfinite(x)])
    return fn(np.arange(ids.size)), np.percentile(v, 2.5), np.percentile(v, 97.5)


# ── the run ─────────────────────────────────────────────────────────────────────
def build(con):
    # per-rate features for each landing season, on the rating basis known at its cut-off
    feats, by_t = {}, {}
    for t in SEASONS:
        modes = {c: rh._mode(rh.load(con, c, through=t - 1))[0]
                 for c in ("NRL", "SL", "NSW", "QLD")}
        key = tuple(sorted((k, str(v)) for k, v in modes.items()))
        if key not in feats:
            rows, worst = rate_rows(con, modes)
            print(f"per-rate z-scores for {dict(key)}: {len(rows):,} player-matches; "
                  f"composite rebuilt differs from the engine's by at most {worst:.2e}")
            feats[key] = rate_features(rows)
        by_t[t] = feats[key]
    cum, one = by_t[max(SEASONS)]
    cohort = fz.entry_cohort(con, LANDINGS)
    know = {t: fz.knowledge(con, t - 1) for t in SEASONS}
    pairs = {t: know[t]["pairs"][know[t]["pairs"].layer == "B_next_season"].copy()
             for t in SEASONS}
    _, sea_all, _ = rh.all_competitions(rb.DB, through=max(SEASONS))
    # season-only tables as held at the end of each season: a competition's rating basis
    # (position- or competition-relative) is decided from the data held, and the
    # Queensland Cup changes basis in 2025, so a threshold has to come from the same
    # table as the rating it is compared with
    sea_at = {t - 1: know[t]["sea"] for t in SEASONS}
    sea_at[max(SEASONS)] = sea_all
    return dict(cum=cum, one=one, cum_by_t={t: v[0] for t, v in by_t.items()},
                one_by_t={t: v[1] for t, v in by_t.items()}, cohort=cohort, pairs=pairs,
                sea_at=sea_at, ps=starts(con))


def run():
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    if CACHE and os.path.exists(CACHE):
        D = pickle.load(open(CACHE, "rb"))
    else:
        D = build(con)
        if CACHE:
            pickle.dump(D, open(CACHE, "wb"))
    con.close()
    cum, one, ps = D["cum"], D["one"], D["ps"]
    thr_at, rated_at = {}, {}
    for t, sea in D["sea_at"].items():
        sea = sea.copy()
        sea["player_id"] = sp.normalize_player_id(sea.player_id)
        rated_at[t] = sea[sea.n_games >= rb.MIN_GAMES]
        thr_at[t] = rated_at[t].groupby(["comp", "season"]).class_score.quantile(2 / 3).to_dict()
    cohort = D["cohort"].copy()
    cohort["group_pos"] = cohort.raw_position.map(sp.POSITION_GROUP)
    worst = 0.0
    for t in SEASONS:
        chk = cohort[cohort.season_tgt == t].merge(
            rated_at[t].rename(columns={"comp": "target", "season": "season_tgt"}),
            on=["player_id", "target", "season_tgt"], how="left")
        worst = max(worst, float((chk.class_target - chk.class_score).abs().max()))
        assert chk.class_score.notna().all()
    print(f"entrant targets against the table held at the end of their season: largest "
          f"difference {worst:.2e}")

    tests, notes = [], []
    for t in SEASONS:
        cum = D.get("cum_by_t", {}).get(t, D["cum"])
        one = D.get("one_by_t", {}).get(t, D["one"])
        train = attach(D["pairs"][t], cum, one, True)
        train["pair"] = train.source + "->" + train.target
        ev = attach(cohort[cohort.season_tgt == t], cum, one, False)
        miss = int(ev[["s_" + c for c in ZC]].isna().all(1).sum())
        # the shipped line
        m0 = fz.fit_lines(train)
        ev["S0"], ev["S0_basis"] = fz.predict_lines(m0, ev)
        train["S0_fit"] = fz.predict_lines(m0, train)[0]
        # 1: components
        b = persistence(train, default="drop")
        b1 = persistence(train, default="median")
        for name, bb in (("C-shrink", None), ("C-persist", b), ("C-persist (pcm median)", b1)):
            tr = train.assign(class_source=index(train, bb))
            mm = fz.fit_lines(tr)
            ev[name] = fz.predict_lines(mm, ev.assign(class_source=index(ev, bb)))[0]
            train[name + "_fit"] = fz.predict_lines(mm, tr)[0]
        r = ridge_fit(train)
        ev["C-ridge"] = ridge_predict(r, ev)
        notes.append(dict(season=t, train_pairs=len(train), entrants=len(ev),
                          missing_rates=miss, ridge_alpha=r["alpha"],
                          corr_shrink_vs_class=float(np.corrcoef(
                              index(ev), ev.class_source)[0, 1]),
                          **{f"b_{x}": round(v, 3) for x, v in zip(RATES, b)}))
        # 3: empirical Bayes and the ranges
        ev["S0-EB"] = eb_predict(eb_lines(train), ev)
        bands = residual_bands(train, m0)
        key = np.where(ev.S0_basis == "direction", ev.pair, "_pooled")
        for i, k in enumerate(key):
            bd = bands.get(k, bands["_pooled"])
            ev.loc[ev.index[i], "sd"] = bd["sd"]
            for q, v in zip(("e10", "e25", "e75", "e90"), bd["q"]):
                ev.loc[ev.index[i], q] = v
        earlier = cohort[cohort.season_tgt < t]
        cards = [card(earlier, row) for row in ev.itertuples()]
        ev["card_n"] = [c[0] for c in cards]
        for j, q in enumerate(("c10", "c25", "c75", "c90")):
            ev[q] = [c[1][j] if c[1] is not None else np.nan for c in cards]
        # 4: outcomes and calibrated probabilities from the training pairs
        ev = label(ev, ps, thr_at[t])
        trl = label(train, ps, thr_at[t - 1])
        for score in ("S0", "C-persist"):
            col = score + "_fit"
            f = logit_fit(trl[[col]].values, trl.top3.values)
            ev[f"p_top3_{score}"] = f(ev[[score]].values)
        f = logit_fit(trl[["S0_fit", "src_start_share", "src_mins"]].values,
                      trl.starter.values)
        ev["p_starter"] = f(ev[["S0", "src_start_share", "src_mins"]].values)
        ev["base_top3"] = np.nanmean(trl.top3)
        # the same base rate inside each target competition, for subsets by target
        ev["base_top3_tgt"] = ev.target.map(trl.groupby("target").top3.mean())
        ev["base_starter"] = np.nanmean(trl.starter)
        tests.append(ev)
    return pd.concat(tests, ignore_index=True), pd.DataFrame(notes)


# ── reporting ───────────────────────────────────────────────────────────────────
def mae_rows(d, models, ref="S0"):
    y = d.class_target.values
    out = []
    for m in models:
        delta, lo, hi = fz.paired_bootstrap(y, d[ref].values, d[m].values,
                                            d.player_id.values)
        out.append(f"| {m} | {np.abs(y - d[m]).mean():.2f} | {delta:+.2f} "
                   f"[{lo:+.2f}, {hi:+.2f}] |")
    return out


def coverage(d):
    y = d.class_target
    S0 = d.S0
    rows = []
    def cov(lo, hi, nominal, label, sub):
        ok = sub & lo.notna()
        c = ((y >= lo) & (y <= hi))[ok]
        rows.append(f"| {label} | {nominal} | {int(ok.sum())} | {100 * c.mean():.0f}% | "
                    f"{(hi - lo)[ok].mean():.1f} |")
    allr = pd.Series(True, index=d.index)
    cov(S0 - 1.96 * d.sd, S0 + 1.96 * d.sd, "95%", "App reference band (±1.96 SD)", allr)
    cov(S0 + d.e10, S0 + d.e90, "80%", "Residual quantiles 10–90", allr)
    cov(S0 + d.e25, S0 + d.e75, "50%", "Residual quantiles 25–75", allr)
    cov(d.c25, d.c75, "50%", "Card middle half", d.card_n >= 8)
    cov(d.c10, d.c90, "80%", "Card 10–90 (shown at 20+)", d.card_n >= 20)
    return rows


def decision_rows(d, sub_label):
    out = []
    ids = d.player_id.values
    for target, scores in (("top3", ["class_source", "S0", "C-persist"]),
                           ("starter", ["class_source", "S0", "src_start_share",
                                        "p_starter"])):
        y = d[target].values
        base = np.nanmean(y)
        for s in scores:
            a, lo, hi = boot(ids, lambda i: auc(y[i], d[s].values[i]))
            out.append(f"| {sub_label} | {target} | {s} | {base:.2f} | {a:.2f} "
                       f"[{lo:.2f}, {hi:.2f}] |")
    return out


def shortlist(d):
    out = []
    for t, g in d.groupby("season_tgt"):
        k = max(1, int(round(len(g) / 3)))
        top = g.sort_values("S0", ascending=False).head(k)
        out.append((t, len(g), k, g.top3.mean(), top.top3.mean(), g.starter.mean(),
                    top.starter.mean()))
    return out


def brier(d, p, y, base):
    ok = d[p].notna() & d[y].notna()
    b = ((d[p] - d[y]) ** 2)[ok].mean()
    b0 = ((d[base] - d[y]) ** 2)[ok].mean()
    return b, b0, 1 - b / b0


def report(ev, notes):
    os.makedirs(OUT, exist_ok=True)
    keep = [c for c in ev.columns if not c.startswith(("s_z", "m_z")) and c not in ("name", "dob")]
    ev[keep].to_csv(os.path.join(OUT, "entrants.csv"), index=False)
    notes.to_csv(os.path.join(OUT, "fits.csv"), index=False)
    sl = ev[ev.pair.isin(H1)]
    L = ["# After the 2026 test: components, ranges, decisions", "",
         "Exploratory. Generated by `explore_post2026.py`; every forecast for a landing "
         "season uses only what was known at the end of the season before. Entrants are "
         "the explicit entry cohort (rated: three or more matches in the new competition). "
         "Intervals are 95% player-cluster bootstrap. Positive Δ means the alternative is "
         "closer than the app's line (S0).", ""]
    for lab, d in (("Into Super League from NRL / NSW Cup / Queensland Cup", sl),
                   ("All directions", ev)):
        L += [f"## 1 and 3. Point forecasts — {lab}", "",
              f"{len(d)} entrants, {d.player_id.nunique()} players, landing 2023–2026.", "",
              "| Model | MAE | Δ vs S0 |", "|---|---|---|",
              f"| S0 (app line) | {np.abs(d.class_target - d.S0).mean():.2f} | — |"]
        L += mae_rows(d, ["C-shrink", "C-persist", "C-persist (pcm median)", "C-ridge",
                          "S0-EB"]) + [""]
        if d is sl:
            L += ["By season, C-persist against S0:", "", "| Season | n | S0 | C-persist | Δ |",
                  "|---|---|---|---|---|"]
            for t, g in d.groupby("season_tgt"):
                dl, lo, hi = fz.paired_bootstrap(g.class_target.values, g.S0.values,
                                                 g["C-persist"].values, g.player_id.values)
                L.append(f"| {t} | {len(g)} | {np.abs(g.class_target - g.S0).mean():.2f} | "
                         f"{np.abs(g.class_target - g['C-persist']).mean():.2f} | "
                         f"{dl:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
            L.append("")
        L += [f"## 3. Do the ranges hold — {lab}", "",
              "| Range | Nominal | Entrants | Covered | Mean width |", "|---|---|---|---|---|"]
        L += coverage(d) + [""]
    L += ["## 4. Decision targets", "",
          "top3 = season-only rating in the top third of every rated player in the new "
          "competition that season. starter = started at least half of his club's matches "
          "that season. Base = share of entrants with the outcome. p_starter is a "
          "logistic fit on the training pairs (line forecast, source start share, source "
          "minutes per match).", "",
          "| Entrants | Outcome | Score | Base | AUC [95%] |", "|---|---|---|---|---|"]
    L += decision_rows(sl, "into SL") + decision_rows(ev, "all") + [""]
    L += ["Shortlist, into Super League: the third of each season's entrants with the "
          "highest line forecast, against all entrants.", "",
          "| Season | Entrants | Shortlist | Top third: all | Top third: shortlist | "
          "Starter: all | Starter: shortlist |", "|---|---|---|---|---|---|---|"]
    for t, n, k, a, b, c, e in shortlist(sl):
        L.append(f"| {t} | {n} | {k} | {a:.0%} | {b:.0%} | {c:.0%} | {e:.0%} |")
    L.append("")
    L += ["Calibrated probabilities against the training base rate (Brier; skill > 0 "
          "means better than the base rate):", "", "| Entrants | Probability | Brier | "
          "Base-rate Brier | Skill |", "|---|---|---|---|---|"]
    for lab, d in (("into SL", sl), ("all", ev)):
        tb = "base_top3_tgt" if lab == "into SL" else "base_top3"
        for p, y, base in (("p_top3_S0", "top3", tb),
                           ("p_top3_C-persist", "top3", tb),
                           ("p_starter", "starter", "base_starter")):
            b, b0, sk = brier(d, p, y, base)
            L.append(f"| {lab} | {p} | {b:.3f} | {b0:.3f} | {sk:+.3f} |")
    L += ["", "## Fits", "", "```", notes.T.to_string(), "```", ""]
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ev, notes = run()
    report(ev, notes)
    sys.exit(0)
