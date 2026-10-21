# -*- coding: utf-8 -*-
"""Lee bounds on the matched shifts, for the players the comparison cannot see.

`matching_moves.py` compares a mover's rating in his new competition with matched
stayers' ratings in the old one. Both ratings exist only for players with three or more
matches that season, and the share who get there differs by direction: about 60% of
moves between the NRL and the two Cups, almost all moves into Super League, and some
other share of the stayers. Whoever does not reach three matches is missing, and he is
missing because of how it went. The matched shift on the players who remain can then
be biased either way.

Lee (2009) bounds the effect for the players who would have been observed whichever
group they were in. Under one assumption (monotonicity: being in the group with the
lower observation rate never makes a player observable who otherwise would not be), the
group observed more often contains the always-observed plus some marginal players. They
are removed by trimming that group's observed ratings from the top (giving the lowest
the effect can be) or from the bottom (the highest). Nothing is assumed about who the
marginal players are, which is why the bounds can be wide.

Design: the at-risk populations are everyone, rated or not, at the outcome season.
  movers   every arrival in the explicit entry cohort with a rated source season; a
           mover is observed if rated (3+ matches) in the target that season
  stayers  primary competition the season before = the source, rated there, and the
           next season seen in the source and in no other competition; observed if
           rated in the source that season
Matching is `matching_moves.match`, exact on competition, season and position group,
Mahalanobis on the season-before profile, K nearest with replacement inside the
caliper; each mover's stayers carry weight 1/K_i. Intervals: 500 resamples of movers
per direction, the 2.5th percentile of the lower bound to the 97.5th of the upper.

Outcome settings. The main run is the matching report's outcome: the shrunk season
rating, observed at 3+ ratable matches. Lowering the bar to 1+ match makes most of the
missing players observable, but a shrunk rating on one or two matches is pulled hard
toward the middle of the pool, which would drag every shift toward zero; so the 1+ runs
are also repeated on the unshrunk rating (`raw_score`, same scale, same centre), which is
noisier and not pulled. All four are reported.

    python lee_bounds.py      # writes LEE_BOUNDS_REPORT.md and matching/lee_*.csv
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

import matching_moves as mm
import rolling_backtest as rb

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BASE, "LEE_BOUNDS_REPORT.md")
N_BOOT, SEED = 500, 0
LAST = 2026


SETTINGS = [(3, "class_score", "3+ matches, shrunk (main)"),
            (1, "class_score", "1+ match, shrunk"),
            (3, "raw_score", "3+ matches, unshrunk"),
            (1, "raw_score", "1+ match, unshrunk")]


def outcome(D, rows, min_n, col):
    s = D["sea_any"]
    s = s[s.n_games >= min_n][["player_id", "comp", "season", col]].rename(
        columns={"comp": "target", "season": "season_tgt", col: "y"})
    return rows.merge(s, on=["player_id", "target", "season_tgt"], how="left")


def movers_at_risk(D, min_n=3, col="class_score"):
    m = D["ev"][["player_id", "source", "target", "season", "transition_type"]].rename(
        columns={"season": "season_tgt"})
    m["season_src"] = m.season_tgt - 1
    m["comp"] = m.source
    m = mm.covariates(D, m)
    m = outcome(D, m, min_n, col)
    m = m[m.y_prev.notna() & m.cls.notna()].reset_index(drop=True)
    m["obs"] = m.y.notna()
    m["pair"] = m.source + "->" + m.target
    return m


def stayers_at_risk(D, movers, min_n=3, col="class_score"):
    p = D["primary"].rename(columns={"season": "season_src", "primary": "comp"})
    p = p[p.season_src < LAST]
    p = mm.covariates(D, p)
    p = p[p.y_prev.notna() & p.cls.notna()].copy()
    p["season_tgt"] = p.season_src + 1
    nxt = D["comps_in"].reindex(list(zip(p.player_id, p.season_tgt)))
    # stayed: seen in his competition next season and in no other. A player seen
    # nowhere is not at risk, just as an arrival must have played a match to be one;
    # counting the stayers who left the game as unobserved (the first version) made the
    # two groups' observation rates incomparable and the bounds too narrow.
    p["stayed"] = [isinstance(s, set) and s == {c} for s, c in zip(nxt, p.comp)]
    p = p[p.stayed]
    p = outcome(D, p.assign(target=p.comp), min_n, col)
    bad = set(zip(movers.player_id, movers.season_tgt))
    p = p[[k not in bad for k in zip(p.player_id, p.season_tgt)]].reset_index(drop=True)
    p["obs"] = p.y.notna()
    return p


def trimmed(v, w, keep, top):
    """Weighted mean of the `keep` share of weight from the top or bottom of v."""
    o = np.argsort(v)
    v, w = v[o], w[o]
    if top:
        v, w = v[::-1], w[::-1]
    W = w.sum() * keep
    c = np.cumsum(w)
    take = np.minimum(w, np.maximum(0.0, W - (c - w)))
    return float((v * take).sum() / take.sum())


def bounds(yt, ot, yc, oc, wc):
    """Observed-only shift, Lee lower and upper bound, and the two observation rates."""
    qt = ot.mean()
    qc = (wc * oc).sum() / wc.sum()
    vt, vc, wco = yt[ot], yc[oc], wc[oc]
    mt, mc = vt.mean(), (vc * wco).sum() / wco.sum()
    if qt <= qc:              # stayers observed more often: trim the stayers
        keep = qt / qc
        lo = mt - trimmed(vc, wco, keep, top=True)
        hi = mt - trimmed(vc, wco, keep, top=False)
    else:                     # movers observed more often: trim the movers
        keep = qc / qt
        w1 = np.ones(len(vt))
        lo = trimmed(vt, w1, keep, top=False) - mc
        hi = trimmed(vt, w1, keep, top=True) - mc
    return mt - mc, lo, hi, qt, qc


def direction(g, S, J, rng=None):
    idx = np.arange(len(g)) if rng is None else rng.integers(0, len(g), len(g))
    yt = g.y.values[idx]
    ot = g.obs.values[idx]
    js = [J[i] for i in g.index.values[idx]]
    cj = np.concatenate(js)
    wc = np.concatenate([np.full(len(j), 1.0 / len(j)) for j in js])
    return bounds(yt, ot, S.y.values[cj], S.obs.values[cj], wc)


def run(caliper=mm.CALIPER, min_n=3, col="class_score", D=None):
    if D is None:
        con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
        D = mm.load(con)
        con.close()
    M = movers_at_risk(D, min_n, col)
    S = stayers_at_risk(D, M, min_n, col)
    J, _ = mm.match(M, S, caliper=caliper)
    keep = np.array([len(j) > 0 for j in J])
    dropped = int((~keep).sum())
    J = [j for j, k in zip(J, keep) if k]
    M = M[keep].reset_index(drop=True)
    rows, draws = [], {}
    for pair, g in M.groupby("pair"):
        att, lo, hi, qt, qc = direction(g, S, J)
        rng = np.random.default_rng(SEED)
        b = np.array([direction(g, S, J, rng)[:3] for _ in range(N_BOOT)])
        draws[pair] = b
        rows.append(dict(pair=pair, source=g.source.iloc[0], target=g.target.iloc[0],
                         at_risk=len(g), observed=int(g.obs.sum()), q_movers=qt,
                         q_stayers=qc, att=att, lee_lo=lo, lee_hi=hi,
                         ci_lo=float(np.percentile(b[:, 1], 2.5)),
                         ci_hi=float(np.percentile(b[:, 2], 97.5))))
    T = pd.DataFrame(rows).sort_values("at_risk", ascending=False)
    P = []
    for i, a in enumerate(mm.COMPS):
        for c in mm.COMPS[i + 1:]:
            f, r = f"{a}->{c}", f"{c}->{a}"
            if f not in draws or r not in draws:
                continue
            tf, tr = T.set_index("pair").loc[f], T.set_index("pair").loc[r]
            bf, br = draws[f], draws[r]
            P.append(dict(a=a, b=c, gap=(tf.att - tr.att) / 2,
                          lo=(tf.lee_lo - tr.lee_hi) / 2, hi=(tf.lee_hi - tr.lee_lo) / 2,
                          ci_lo=float(np.percentile((bf[:, 1] - br[:, 2]) / 2, 2.5)),
                          ci_hi=float(np.percentile((bf[:, 2] - br[:, 1]) / 2, 97.5)),
                          n=f"{int(tf.at_risk)} / {int(tr.at_risk)}"))
    return T, pd.DataFrame(P), M, S, dropped


def settings_section(results):
    """Pair gaps and observation rates under every outcome setting."""
    L = ["## The threshold and the shrinkage", "",
         "Observation rates (movers / matched stayers) and the gap bounds for each outcome "
         "setting. Same matching throughout (caliper 1.5).", "",
         "| Pair | Setting | Observed, a→b | Observed, b→a | Gap, observed only | "
         "Lee bounds | 95% for the bounds |", "|---|---|---|---|---|---|---|"]
    for pair in ("NRL–NSW", "NRL–QLD", "NRL–SL", "NSW–QLD", "NSW–SL", "QLD–SL"):
        a, b = pair.split("–")
        for label, (T, P) in results.items():
            t = T.set_index("pair")
            p = P.set_index(["a", "b"])
            if (a, b) not in p.index:
                continue
            r = p.loc[(a, b)]
            f, rv = t.loc[f"{a}->{b}"], t.loc[f"{b}->{a}"]
            L.append(f"| {pair} | {label} | {f.q_movers:.0%} / {f.q_stayers:.0%} | "
                     f"{rv.q_movers:.0%} / {rv.q_stayers:.0%} | {r.gap:+.1f} | "
                     f"[{r.lo:+.1f}, {r.hi:+.1f}] | [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}] |")
    L.append("")
    return L


def report(T, P, M, S, dropped, T3, P3, results=None):
    os.makedirs(mm.OUT, exist_ok=True)
    T.to_csv(os.path.join(mm.OUT, "lee_directions.csv"), index=False)
    P.to_csv(os.path.join(mm.OUT, "lee_pairs.csv"), index=False)
    L = ["# Lee bounds on the matched shifts", "",
         "Generated by `lee_bounds.py`; see its docstring for the design. Shifts are in "
         "season-only peer-score points: the mover's rating in the new competition minus "
         "what matched stayers rated in the old one. Bounds hold for the players who "
         "would have played three or more matches whichever group they were in, under "
         "monotonicity. Arrivals who never played a match in the new competition are not "
         "in the data at all and are outside every number here.", "",
         f"At-risk movers matched: {len(M)} (dropped, no stayer inside the caliper: "
         f"{dropped}); at-risk stayer pool: {len(S):,}, of whom {S.obs.mean():.0%} were "
         "rated the next season.", "",
         "## By direction", "",
         "| Move | At risk | Observed: movers / matched stayers | Shift, observed only | "
         "Lee bounds | 95% for the bounds |", "|---|---|---|---|---|---|"]
    for r in T.itertuples():
        L.append(f"| {r.pair} | {r.at_risk} | {r.q_movers:.0%} / {r.q_stayers:.0%} | "
                 f"{r.att:+.1f} | [{r.lee_lo:+.1f}, {r.lee_hi:+.1f}] | "
                 f"[{r.ci_lo:+.1f}, {r.ci_hi:+.1f}] |")
    L += ["", "## Strength gaps between two competitions", "",
          "Gap = half the difference of the two directions' shifts, as in "
          "`MATCHING_REPORT.md`; its bounds combine the two directions' bounds the "
          "unfavourable way round.", "",
          "| Pair | At risk a→b / b→a | Gap, observed only | Lee bounds | 95% for the "
          "bounds | Bounds, caliper 3 |", "|---|---|---|---|---|---|"]
    p3 = P3.set_index(["a", "b"])
    for r in P.itertuples():
        q = p3.loc[(r.a, r.b)]
        L.append(f"| {r.a}–{r.b} | {r.n} | {r.gap:+.1f} | [{r.lo:+.1f}, {r.hi:+.1f}] | "
                 f"[{r.ci_lo:+.1f}, {r.ci_hi:+.1f}] | [{q.lo:+.1f}, {q.hi:+.1f}] |")
    L.append("")
    if results:
        L += settings_section(results)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    D = mm.load(con)
    con.close()
    T, P, M, S, dropped = run(D=D)
    T3, P3, *_ = run(3.0, D=D)
    results = {"3+ matches, shrunk (main)": (T, P)}
    for min_n, col, label in SETTINGS[1:]:
        Tx, Px, *_ = run(min_n=min_n, col=col, D=D)
        results[label] = (Tx, Px)
    report(T, P, M, S, dropped, T3, P3, results)
    sys.exit(0)
