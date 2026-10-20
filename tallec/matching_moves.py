# -*- coding: utf-8 -*-
"""Players who changed competition against similar players who stayed.

The measured ladder is a raw shift: the rating after a move minus the rating before it.
That shift carries more than the difference between two competitions. Clubs sign men
after good seasons, so part of any rating falls back next year wherever he plays; he is
a year older; and his role may change. Matching each mover to players who looked the
same the season before and did NOT move separates that out: whatever the stayers did
next season is what the mover would have been expected to do anyway, on his old
competition's scale. The mover's rating in the new competition minus that is the
matched shift.

What the matched shift is, read carefully. Ratings are peer scores inside each
competition, so the shift is mostly the difference in strength between the two fields
plus any cost or gain of moving itself (adapting, a new role). With both directions of a
pair measured, the two separate: a pure strength difference changes sign with the
direction, a cost of moving does not. `joint` fits one strength per competition and one
common effect of moving across every direction at once. What matching cannot remove is
selection on what the data do not record: a club's scouting, a contract, a family.

Design, fixed before any result was seen:
  ratings     one season-only table and one cumulative table built through the last
              season, so every season of a competition is on the same basis (the
              Queensland Cup changes basis with the data cut-off; see HANDOVER)
  movers      the explicit entry cohort: first and returning entries, rated (3+
              matches) in the season before in the source and in the landing season in
              the target
  stayers     primary competition the season before = the mover's source (most
              minutes), rated there, and the next season rated there and seen in no
              other competition
  exact on    source competition, season, position group
  distance    Mahalanobis on cumulative Class, season-only rating, age, minutes per
              match, rated matches, share of matches started, all the season before
  matches     K nearest stayers with replacement, within a caliper; a mover with none
              inside it is dropped and counted
  checks      standardised differences before and after; a placebo on the season two
              before (not matched on); stayers given a pretend move, which must give 0

    python matching_moves.py        # writes MATCHING_REPORT.md and matching/
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

import rating_history as rh
import rolling_backtest as rb
import sp_schema as sp
import transition_events as te

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "matching")
REPORT = os.path.join(BASE, "MATCHING_REPORT.md")
COMPS = ["NRL", "NSW", "QLD", "SL"]
THROUGH = 2026
K = 5
CALIPER = 1.5            # Mahalanobis units
COVS = ["cls", "y_prev", "age", "mins_pg", "games", "start_share"]
N_BOOT, SEED = 2000, 0
MIN_DIR = 15             # movers a direction needs to be reported on its own


# ── data ────────────────────────────────────────────────────────────────────────
def load(con):
    cum, sea, ext = rh.all_competitions(rb.DB, through=THROUGH)
    for d in (cum, sea, ext):
        d["player_id"] = sp.normalize_player_id(d.player_id)
    sea = sea[sea.n_games >= rb.MIN_GAMES]
    pm = pd.read_sql("SELECT player_id, competition comp, season, minutes "
                     "FROM player_match_stats", con)
    pm["player_id"] = sp.normalize_player_id(pm.player_id)
    mins = pm.groupby(["player_id", "season", "comp"]).minutes.sum().reset_index()
    primary = (mins.sort_values(["player_id", "season", "minutes", "comp"],
                                ascending=[True, True, False, True])
               .drop_duplicates(["player_id", "season"])
               .rename(columns={"comp": "primary"})[["player_id", "season", "primary"]])
    comps_in = mins.groupby(["player_id", "season"]).comp.agg(set)
    starts = pd.read_sql("SELECT player_id, comp, season, start_share FROM "
                         "player_season_starts", con)
    starts["player_id"] = sp.normalize_player_id(starts.player_id)
    dob = pd.read_sql("SELECT player_id, dob FROM players", con)
    dob["player_id"] = sp.normalize_player_id(dob.player_id)
    dob = dob.set_index("player_id").dob
    ev = te.build(con, through=THROUGH)
    ev["player_id"] = sp.normalize_player_id(ev.player_id)
    ev = ev[ev.arrived.astype(bool) & ev.transition_type.isin(rb.ENTRY_TYPES)]
    return dict(cum=cum, sea=sea, ext=ext, primary=primary, comps_in=comps_in,
                starts=starts, dob=dob, ev=ev)


def covariates(D, rows):
    """The season-before profile of (player_id, comp, season_src) rows."""
    r = rows.copy()
    k = ["player_id", "comp", "season"]
    r = r.merge(D["cum"][k + ["class_score"]].rename(
        columns={"class_score": "cls", "season": "season_src"}),
        on=["player_id", "comp", "season_src"], how="left")
    r = r.merge(D["sea"][k + ["class_score"]].rename(
        columns={"class_score": "y_prev", "season": "season_src"}),
        on=["player_id", "comp", "season_src"], how="left")
    r = r.merge(D["sea"][k + ["class_score"]].rename(
        columns={"class_score": "y_prev2"}).assign(season_src=lambda d: d.season + 1)
        .drop(columns="season"), on=["player_id", "comp", "season_src"], how="left")
    r = r.merge(D["ext"][k + ["mins_pg", "games_src", "position"]].rename(
        columns={"season": "season_src", "games_src": "games"}),
        on=["player_id", "comp", "season_src"], how="left")
    r = r.merge(D["starts"].rename(columns={"season": "season_src"}),
                on=["player_id", "comp", "season_src"], how="left")
    r["group"] = r.position.map(sp.POSITION_GROUP).fillna("Bench")
    r["age"] = sp.age_at(sp.parse_dob(r.player_id.map(D["dob"])),
                         pd.Series(r.season_src.values, index=r.index)).values
    return r


def outcome(D, rows, comp_col, season_col, name):
    s = D["sea"][["player_id", "comp", "season", "class_score"]].rename(
        columns={"comp": comp_col, "season": season_col, "class_score": name})
    return rows.merge(s, on=["player_id", comp_col, season_col], how="left")


def movers(D):
    m = D["ev"][["player_id", "source", "target", "season", "transition_type"]].rename(
        columns={"season": "season_tgt"})
    m["season_src"] = m.season_tgt - 1
    m["comp"] = m.source
    m = covariates(D, m)
    m = outcome(D, m, "target", "season_tgt", "y")
    keep = m.y_prev.notna() & m.y.notna() & m.cls.notna()
    return m[keep].reset_index(drop=True)


def stayers(D, exclude):
    p = D["primary"].rename(columns={"season": "season_src", "primary": "comp"})
    p = covariates(D, p)
    p = p[p.y_prev.notna() & p.cls.notna()].copy()
    p["season_tgt"] = p.season_src + 1
    nxt = D["comps_in"].reindex(list(zip(p.player_id, p.season_tgt)))
    p["only_same"] = [isinstance(s, set) and s == {c} for s, c in zip(nxt, p.comp)]
    p = outcome(D, p.assign(target=p.comp), "target", "season_tgt", "y")
    p = p[p.only_same & p.y.notna()]
    bad = set(zip(exclude.player_id, exclude.season_tgt))
    p = p[[k not in bad for k in zip(p.player_id, p.season_tgt)]]
    return p.reset_index(drop=True)


# ── matching ────────────────────────────────────────────────────────────────────
def _fill(X, med):
    return np.where(np.isnan(X), med, X)


def match(treated, pool, k=K, caliper=CALIPER):
    """For each treated row: up to k pool row indices, exact on comp, season, group."""
    Xp = pool[COVS].to_numpy(float)
    med = np.nanmedian(Xp, axis=0)
    Xp = _fill(Xp, med)
    VI = np.linalg.pinv(np.cov(Xp, rowvar=False))
    Xt = _fill(treated[COVS].to_numpy(float), med)
    keys = pool.groupby(["comp", "season_src", "group"]).indices
    out, dist = [], []
    for i, row in enumerate(treated.itertuples()):
        idx = keys.get((row.comp, row.season_src, row.group))
        if idx is None or not len(idx):
            out.append(np.array([], int))
            dist.append(np.nan)
            continue
        d = Xp[idx] - Xt[i]
        dd = np.sqrt(np.einsum("ij,jk,ik->i", d, VI, d))
        order = np.argsort(dd)[:k]
        ok = order[dd[order] <= caliper]
        out.append(idx[ok])
        dist.append(float(dd[order[0]]))
    return out, np.array(dist)


def smd(a, b):
    v = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2)
    return (np.nanmean(a) - np.nanmean(b)) / v if v > 0 else 0.0


def boot_mean(x, ids, n=N_BOOT):
    x, ids = np.asarray(x, float), np.asarray(ids)
    ok = ~np.isnan(x)
    x, ids = x[ok], ids[ok]
    if not len(x):
        return np.nan, np.nan, np.nan
    u = np.array(sorted(pd.unique(ids)))
    at = {p: np.flatnonzero(ids == p) for p in u}
    rng = np.random.default_rng(SEED)
    s = [x[np.concatenate([at[p] for p in rng.choice(u, u.size)])].mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ── joint strengths and the effect of moving ────────────────────────────────────
def joint(table, comps=COMPS):
    """ATT(a->b) = S_a - S_b + M, S_NRL = 0, weighted by 1/var. Returns S and M."""
    t = table.dropna(subset=["att", "se"])
    X, y, w = [], [], []
    for r in t.itertuples():
        x = np.zeros(len(comps))           # S_NSW, S_QLD, S_SL, M
        if r.source != "NRL":
            x[comps.index(r.source) - 1] += 1
        if r.target != "NRL":
            x[comps.index(r.target) - 1] -= 1
        x[-1] = 1
        X.append(x), y.append(r.att), w.append(1 / r.se ** 2)
    X, y, w = np.array(X), np.array(y), np.array(w)
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    return dict(zip(comps[1:] + ["move"], beta)), X @ beta


# ── the run ─────────────────────────────────────────────────────────────────────
def pairwise(T):
    """Both directions of a pair: strength gap = (a->b - b->a)/2, pair's own effect of
    moving = (a->b + b->a)/2. No assumption that moving costs the same everywhere."""
    t = T.set_index("pair")
    out = []
    for i, a in enumerate(COMPS):
        for b in COMPS[i + 1:]:
            f, r = f"{a}->{b}", f"{b}->{a}"
            if f in t.index and r in t.index:
                out.append(dict(a=a, b=b, n_ab=int(t.n[f]), n_ba=int(t.n[r]),
                                gap=(t.att[f] - t.att[r]) / 2,
                                move=(t.att[f] + t.att[r]) / 2,
                                gap_naive=(t.naive[f] - t.naive[r]) / 2,
                                se=float(np.sqrt(t.se[f] ** 2 + t.se[r] ** 2) / 2)))
    return pd.DataFrame(out)


def network(P, col="gap"):
    """One strength per competition from the pair gaps, weighted by precision."""
    X, y, w = [], [], []
    for r in P.itertuples():
        x = np.zeros(len(COMPS) - 1)
        if r.a != "NRL":
            x[COMPS.index(r.a) - 1] += 1
        if r.b != "NRL":
            x[COMPS.index(r.b) - 1] -= 1
        X.append(x), y.append(getattr(r, col)), w.append(1 / r.se ** 2)
    X, y, w = np.array(X), np.array(y), np.array(w)
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    # gap = S_a - S_b with S_NRL = 0, so a weaker competition comes out negative
    return dict(zip(COMPS[1:], beta))


def directions(M):
    rows = []
    for pair, g in M.groupby("pair"):
        a, lo, hi = boot_mean(g.att_i, g.player_id)
        rows.append(dict(pair=pair, source=g.source.iloc[0], target=g.target.iloc[0],
                         n=len(g), naive=g.naive_i.mean(), att=a, lo=lo, hi=hi,
                         stay_change=g.stay_change_i.mean(), se=(hi - lo) / 3.92))
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def run(caliper=CALIPER, D=None):
    if D is None:
        con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
        D = load(con)
        con.close()
    M = movers(D)
    S = stayers(D, M)
    J, dist = match(M, S, caliper=caliper)
    M["n_matched"] = [len(j) for j in J]
    M["nn_dist"] = dist
    dropped = M[M.n_matched == 0]
    M = M[M.n_matched > 0].reset_index(drop=True)
    J = [j for j in J if len(j)]
    M["y0"] = [S.y.values[j].mean() for j in J]                 # the counterfactual
    M["att_i"] = M.y - M.y0
    M["naive_i"] = M.y - M.cls
    M["stay_change_i"] = M.y0 - [S.cls.values[j].mean() for j in J]
    # placebo on the season two before, which was not matched on
    pre_s = [np.nanmean(S.y_prev.values[j] - S.y_prev2.values[j])
             if np.isfinite(S.y_prev2.values[j]).any() else np.nan for j in J]
    M["pre_i"] = (M.y_prev - M.y_prev2) - np.array(pre_s)
    M["pair"] = M.source + "->" + M.target
    return D, M, S, J, dropped


def pseudo(S, n=600):
    """Stayers given a pretend move and matched to the other stayers: should be ~0."""
    rng = np.random.default_rng(SEED)
    out = []
    for c in COMPS:
        pool = S[S.comp == c]
        if len(pool) < 100:
            continue
        pick = pool.sample(min(n, len(pool) // 4), random_state=SEED)
        rest = pool.drop(pick.index)
        J, _ = match(pick, rest)
        d = [y - rest.y.values[j].mean() for y, j in zip(pick.y, J) if len(j)]
        out.append((c, len(d), float(np.mean(d)), float(np.std(d) / np.sqrt(len(d)))))
    return out


def event_study(D, M, S, J, target="SL"):
    """Mean rating of entrants into `target` and of their matched stayers, t-2 .. t+2."""
    sea = D["sea"].set_index(["player_id", "comp", "season"]).class_score
    rows = []
    sub = M[M.target == target]
    for i in sub.index:
        r = M.loc[i]
        js = S.iloc[J[i]]
        for k in (-2, -1, 0, 1, 2):
            yr = int(r.season_tgt + k)
            comp_m = r.source if k < 0 else target
            vm = sea.get((r.player_id, comp_m, yr), np.nan)
            vs = np.array([sea.get((p, r.source, yr), np.nan) for p in js.player_id])
            rows.append(dict(i=i, player_id=r.player_id, k=k, mover=vm,
                             stayers=np.nanmean(vs) if np.isfinite(vs).any() else np.nan))
    return pd.DataFrame(rows)


def report(D, M, S, J, dropped):
    os.makedirs(OUT, exist_ok=True)
    M.drop(columns=[c for c in ("position",) if c in M]).to_csv(
        os.path.join(OUT, "movers_matched.csv"), index=False)
    L = ["# Movers against matched stayers", "",
         "Exploratory. Generated by `matching_moves.py`. Ratings are season-only peer "
         "scores (50 = position-group median of that competition and season) from one "
         "table built through 2026. Intervals: 95% player-cluster bootstrap over movers, "
         "with matches held fixed (they ignore the variance of reusing stayers, so they "
         "are somewhat too narrow).", "",
         f"Movers rated on both sides: {len(M) + len(dropped)}; matched: {len(M)}; "
         f"dropped, no stayer within the caliper: {len(dropped)}. Stayer pool: "
         f"{len(S):,} player-seasons. Median matches per mover: "
         f"{int(np.median(M.n_matched))} of {K}.", ""]

    # who is in the comparison at all
    sv = (D["ev"].groupby(["source", "target"])
          .agg(arrived=("rated", "size"), rated=("rated", "mean"),
               matches=("target_matches", "median")).reset_index())
    L += ["## Who reaches three matches", "",
          "A mover enters the comparison only if he plays three or more matches in the "
          "new competition. Where many arrivals do not, the ones who do were selected on "
          "how it went, which biases the matched shift upward in that direction. Into "
          "Super League almost everyone does; between the NRL and the two Cups about 40% "
          "do not, in both directions.", "",
          "| Move | Arrivals | Reached 3 matches | Median matches |", "|---|---|---|---|"]
    for r in sv.itertuples():
        L.append(f"| {r.source}->{r.target} | {r.arrived} | {r.rated:.0%} | "
                 f"{r.matches:.0f} |")
    L.append("")

    # balance
    L += ["## Balance (standardised mean difference, movers vs stayers)", "",
          "| Covariate | Before | After |", "|---|---|---|"]
    for c in COVS + ["y_prev2"]:
        after_ctrl = np.concatenate([S[c].values[j] for j in J])
        L.append(f"| {c} | {smd(M[c], S[c]):+.2f} | {smd(M[c], after_ctrl):+.2f} |")
    L += ["", "|SMD| under 0.1 is the usual standard. `y_prev2` (the season two before) "
          "was not matched on.", ""]

    # placebos
    L += ["## Placebos", ""]
    pre = boot_mean(M.pre_i, M.player_id)
    L += [f"- **Season two before, not matched on**: movers' change from t−2 to t−1 "
          f"minus their stayers' change, {M.pre_i.notna().sum()} movers with both "
          f"seasons: {pre[0]:+.2f} [{pre[1]:+.2f}, {pre[2]:+.2f}]. Zero means movers were "
          f"not on a different trajectory from stayers who looked the same at t−1."]
    for c, n, m, se in pseudo(S):
        L.append(f"- **Pretend move inside {c}**: {n} stayers, matched to other "
                 f"stayers: {m:+.2f} (se {se:.2f}). Should be zero.")
    L.append("")

    # per direction
    T = directions(M)
    T.to_csv(os.path.join(OUT, "directions.csv"), index=False)
    L += ["## The shift, raw and matched", "",
          "naive = rating after minus cumulative Class before (the ladder's quantity). "
          "matched = rating after minus the matched stayers' rating that season. "
          "stayers' change = what the matched stayers did relative to their own Class: "
          "the part of the raw shift that would have happened without a move.", "",
          "| Move | Movers | Naive | Matched [95%] | Stayers' change |",
          "|---|---|---|---|---|"]
    for r in T.itertuples():
        flag = "" if r.n >= MIN_DIR else " (few)"
        L.append(f"| {r.pair}{flag} | {r.n} | {r.naive:+.1f} | {r.att:+.1f} "
                 f"[{r.lo:+.1f}, {r.hi:+.1f}] | {r.stay_change:+.1f} |")
    L.append("")

    # joint
    Tj = T[T.n >= MIN_DIR]
    est, fit = joint(Tj)
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(500):
        rr = []
        for pair, g in M[M.pair.isin(Tj.pair)].groupby("pair"):
            s = g.att_i.values[rng.integers(0, len(g), len(g))]
            rr.append(dict(source=g.source.iloc[0], target=g.target.iloc[0], att=s.mean(),
                           se=float(Tj.set_index("pair").se[pair])))
        bs.append(joint(pd.DataFrame(rr))[0])
    bs = pd.DataFrame(bs)
    L += ["## One strength per competition, and the effect of moving", "",
          f"Fitted on the {len(Tj)} directions with {MIN_DIR}+ movers, weighted by "
          "precision: matched shift(a→b) = S_a − S_b + M. S_NRL = 0. M is the common "
          "effect of changing competition at all; a pure strength gap changes sign with "
          "direction and M does not. Intervals from 500 resamples of the movers.", "",
          "| | Estimate [95%] |", "|---|---|", "| NRL | 0 (reference) |"]
    for c in ["NSW", "QLD", "SL", "move"]:
        lab = {"move": "**M, effect of moving**", "NSW": "NSW Cup", "QLD": "Queensland Cup",
               "SL": "Super League"}[c]
        L.append(f"| {lab} | {est[c]:+.1f} [{bs[c].quantile(.025):+.1f}, "
                 f"{bs[c].quantile(.975):+.1f}] |")
    L += ["", "| Move | Matched | Joint fit |", "|---|---|---|"]
    for r, f in zip(Tj.itertuples(), fit):
        L.append(f"| {r.pair} | {r.att:+.1f} | {f:+.1f} |")
    L.append("")

    # pairwise, with each pair's own effect of moving
    P = pairwise(T)
    P.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    na, ma = network(P, "gap_naive"), network(P, "gap")
    L += ["## Pair by pair, each pair with its own effect of moving", "",
          "With both directions measured, the strength gap is half the difference of the "
          "two matched shifts and the pair's effect of moving is half their sum. If the "
          "effect of moving differs a lot between pairs, the single-M fit above is "
          "misspecified and this is the one to read.", "",
          "| Pair | Movers a→b / b→a | Gap, naive | Gap, matched | Effect of moving |",
          "|---|---|---|---|---|"]
    for r in P.itertuples():
        L.append(f"| {r.a}–{r.b} | {r.n_ab} / {r.n_ba} | {r.gap_naive:+.1f} | "
                 f"{r.gap:+.1f} ± {1.96 * r.se:.1f} | {r.move:+.1f} |")
    L += ["", "Strength against the NRL from the pair gaps (precision-weighted):", "",
          "| | Naive, same movers | Matched |", "|---|---|---|", "| NRL | 0 | 0 |"]
    for c in ("NSW", "QLD", "SL"):
        L.append(f"| {c} | {na[c]:+.1f} | {ma[c]:+.1f} |")
    L.append("")
    # sensitivity to the caliper
    L += ["## Sensitivity to the caliper", "",
          "| Caliper | Movers matched | NSW | QLD | SL | NRL–SL gap | Effect of moving, "
          "NRL–NSW |", "|---|---|---|---|---|---|---|"]
    for cal in (CALIPER, 3.0, np.inf):
        _, Mc, _, _, _ = run(cal, D)
        Pc = pairwise(directions(Mc))
        nc = network(Pc, "gap")
        g = Pc.set_index(["a", "b"])
        L.append(f"| {cal:g} | {len(Mc)} | {nc['NSW']:+.1f} | {nc['QLD']:+.1f} | "
                 f"{nc['SL']:+.1f} | {g.gap[('NRL', 'SL')]:+.1f} | "
                 f"{g.move[('NRL', 'NSW')]:+.1f} |")
    L.append("")

    # event study
    E = event_study(D, M, S, J)
    E.to_csv(os.path.join(OUT, "event_study_sl.csv"), index=False)
    L += ["## Entrants into Super League, two seasons either side", "",
          "Seasons before the move are in the source competition, from the move on in "
          "Super League; the matched stayers stay in the source competition throughout. "
          "Each row uses the movers who are rated that season and have a rated matched "
          "stayer, so the set of players changes from row to row.", "",
          "| Season vs move | Movers | Mover mean | Stayers mean | Difference [95%] |",
          "|---|---|---|---|---|"]
    for k, g in E.groupby("k"):
        g = g.dropna(subset=["mover", "stayers"])
        d = boot_mean(g.mover - g.stayers, g.player_id)
        L.append(f"| {k:+d} | {len(g)} | {g.mover.mean():.1f} | {g.stayers.mean():.1f} | "
                 f"{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] |")
    w = E.pivot_table(index="i", columns="k", values=["mover", "stayers"])
    bal = w.dropna(subset=[("mover", 0), ("mover", 1), ("stayers", 0), ("stayers", 1)])
    if len(bal):
        ch = ((bal[("mover", 1)] - bal[("mover", 0)])
              - (bal[("stayers", 1)] - bal[("stayers", 0)]))
        ids = M.loc[bal.index, "player_id"].values
        d = boot_mean(ch.values, ids)
        L += ["", f"**Second season**, the same {len(bal)} movers rated in Super League in "
              f"both their first and second season, against their stayers over the same "
              f"two seasons: change from season one to two, mover minus stayers, "
              f"{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]. Positive means they gained on "
              f"their stayers in year two. These are the movers who kept a place, so "
              f"they are a selected group."]
    L.append("")

    # heterogeneity, into SL, net of each direction's own matched shift
    sl = M[M.target == "SL"].copy()
    sl["rel"] = sl.att_i - sl.pair.map(T.set_index("pair").att)
    sl["role"] = np.where(sl.start_share < 0.5, "mostly bench", "mostly starting")
    sl["age_band"] = pd.cut(sl.age, [0, 24.99, 29.99, 99], labels=["under 25", "25-29",
                                                                    "30+"])
    L += ["## Who does better than his direction's shift, into Super League", "",
          "Each mover's matched shift minus the matched shift of his direction, so the "
          "difference between source competitions is taken out. Small groups; read the "
          "intervals.", "", "| Group | Movers | Relative [95%] |", "|---|---|---|"]
    for col in ("role", "age_band", "group", "source"):
        for v, g in sl.groupby(col, observed=True):
            if len(g) < 8:
                continue
            d = boot_mean(g.rel, g.player_id)
            L.append(f"| {col}: {v} | {len(g)} | {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] |")
    L.append("")
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    report(*run())
    sys.exit(0)
