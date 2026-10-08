# -*- coding: utf-8 -*-
"""Recompute the claims under review from the exported CSVs alone (C1-C9, R1-R4).

No database, no project imports - pandas and numpy (and the standard library). Each
section prints the figure the status note or the reports state, recomputed. If a number
here disagrees with docs/what_is_live.html, docs/ROLLING_REPORT.md or results/*.txt,
that is a finding in its own right: report it before anything else.

The bootstrap is the project's: paired difference in absolute error, resampling PLAYERS
(not rows), 4,000 draws, seed 0 - so the intervals should match to the second decimal.

    python reproduce.py
"""
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
pd.set_option("display.width", 140)


def rule(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def boot(d, a, b, target="class_target", n=4000, seed=0):
    """MAE(a) - MAE(b): positive means b is closer. Player-clustered, as in the project."""
    g = d[d[a].notna() & d[b].notna() & d[target].notna()]
    diff = ((g[target] - g[a]).abs() - (g[target] - g[b]).abs()).values
    players = g.player_id.values
    uniq = pd.unique(players)
    at = {p: np.where(players == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    s = np.empty(n)
    for i in range(n):
        pick = rng.choice(uniq, uniq.size, replace=True)
        s[i] = np.concatenate([diff[at[p]] for p in pick]).mean()
    return float(diff.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def fmt(b):
    clear = b[1] > 0 or b[2] < 0
    return f"{b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}]" + ("  clear" if clear else "")


def mae(d, col, target="class_target"):
    return float((d[target] - d[col]).abs().mean())


def main():
    files = sorted(glob.glob(os.path.join(DATA, "rolling_eval_*.csv")))
    if not files:
        sys.exit("no data/rolling_eval_*.csv - run this from the package root")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["flat50"] = 50.0
    sl = d[d.target == "SL"]

    rule("0. HEADLINE BY ORIGIN - compare with docs/ROLLING_REPORT.md")
    preds = [("conditional model", "model"), ("straight line, per direction", "line_direction"),
             ("translation (ladder)", "projected"), ("no translation", "class_source"),
             ("competition average (50)", "flat50")]
    rows = [dict(origin=o, predictor=lab, n=len(g), mae=mae(g, c))
            for o, g in d.groupby("origin") for lab, c in preds]
    print(pd.DataFrame(rows).pivot(index="predictor", columns="origin", values="mae")
          .round(2).to_string())
    print(f"\nall {len(d)} moves: model {mae(d, 'model'):.2f}, line {mae(d, 'line_direction'):.2f}")

    rule("C1. MOVES INTO SUPER LEAGUE: the model against the straight line")
    print(f"n = {len(sl)} moves, {sl.player_id.nunique()} players")
    print(f"MAE model {mae(sl, 'model'):.2f}   line {mae(sl, 'line_direction'):.2f}")
    print(f"line - model (positive = model closer): {fmt(boot(sl, 'line_direction', 'model'))}")
    print("note claims: 17.98 / 17.72, difference -1.56 to +1.06")

    rule("C2. PER DIRECTION (20+ moves): the model against the straight line")
    rows = []
    for pair, g in d.groupby("pair"):
        if len(g) < 20:
            continue
        b = boot(g, "line_direction", "model")
        rows.append(dict(direction=pair, n=len(g), model=round(mae(g, "model"), 2),
                         line=round(mae(g, "line_direction"), 2),
                         model_closer_by=round(b[0], 2), lo=round(b[1], 2), hi=round(b[2], 2),
                         clear="yes" if (b[1] > 0 or b[2] < 0) else "no"))
    print(pd.DataFrame(rows).to_string(index=False))
    print("note claims: clearly closer for NRL->QLD (+1.96, +0.75 to +3.29) and NSW->QLD\n"
          "(+1.47, +0.58 to +2.42); clearly further off for NRL->NSW, QLD->NSW, QLD->SL")

    rule("C3. THE LINES THE APP NOW USES INTO SUPER LEAGUE")
    tp = pd.read_csv(os.path.join(DATA, "translation_pairs_v3.csv"))
    tp = tp[(tp.layer == "B_next_season") & (tp.target == "SL")]
    for src, g in tp.groupby("source"):
        if len(g) < 25:
            continue
        k, c = np.polyfit(g.class_source, g.class_target, 1)
        res = g.class_target - (k * g.class_source + c)
        sd = math.sqrt(float((res ** 2).sum()) / (len(g) - 2))
        print(f"{src}->SL: SL = {k:.3f} x source + {c:.2f}   ({len(g)} pairs, residual SD {sd:.2f},"
              f" individual band +/-{1.96 * sd:.1f})")
    print("note claims: from the NRL, 0.37 x rating + 43")

    nf = os.path.join(DATA, "noise_floor")
    if os.path.isdir(nf):
        m = pd.read_csv(os.path.join(nf, "movers_floor.csv"))
        rule("C4. NOISE FLOOR OF THE TARGET, 90 moves into Super League")
        erf = np.vectorize(math.erf)
        g = lambda z, c, tau: 100.0 * 0.5 * (1.0 + erf((z - c) / tau / math.sqrt(2.0)))
        rng = np.random.default_rng(0)
        fl, nv = [], []
        for r in m.itertuples():
            tau = math.sqrt(r.tau2)
            e = rng.normal(0.0, math.sqrt(r.sigma2 / r.n_tgt), 20000)
            y0 = g(np.array([r.mu + r.shrinkage_B * (r.class_z - r.mu)]), r.scale_centre, tau)[0]
            ys = g(r.mu + r.shrinkage_B * (r.class_z + e - r.mu), r.scale_centre, tau)
            fl.append(float(np.abs(ys - y0).mean()))
            nv.append(float(((ys - y0) ** 2).mean()))
        m["floor_re"], m["nv_re"] = fl, nv
        print(f"floor on the published target: {np.mean(fl):.2f}   (note: about 10)")
        print(f"matches the exported per-mover floor to {float((m.floor_re - m.floor_published).abs().max()):.4f}")
        print(f"target matches: median {m.n_tgt.median():.0f}; shrinkage B median {m.shrinkage_B.median():.2f}")

        rule("C5. THE VERDICT ON LESS NOISY TARGETS")
        for lab, sub in (("all 90", m), ("16+ target matches", m[m.n_tgt >= 16]),
                         ("20+ target matches", m[m.n_tgt >= 20])):
            print(f"{lab:20s} n={len(sub):3d}  floor {sub.floor_re.mean():5.2f}  "
                  f"model {mae(sub, 'model'):.2f}  line {mae(sub, 'line_direction'):.2f}  "
                  f"line - model {fmt(boot(sub, 'line_direction', 'model'))}")
        se_m = (m.class_target - m.model) ** 2
        se_l = (m.class_target - m.line_direction) ** 2
        noise = float(m.nv_re.mean())
        true = lambda se: math.sqrt(max(float(se.mean()) - noise, 0.0))
        pids = m.player_id.values
        uniq = pd.unique(pids)
        at = {q: np.where(pids == q)[0] for q in uniq}
        rng = np.random.default_rng(0)
        bs = np.empty(4000)
        for i in range(4000):
            idx = np.concatenate([at[q] for q in rng.choice(uniq, uniq.size, replace=True)])
            bs[i] = true(se_l.iloc[idx]) - true(se_m.iloc[idx])
        print(f"noise subtracted (an assumption-dependent sensitivity figure, not an error "
              f"against observed true ability): model {true(se_m):.2f}, "
              f"line {true(se_l):.2f}, line - model {true(se_l) - true(se_m):+.2f} "
              f"[{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")

        S = pd.read_csv(os.path.join(nf, "split_half.csv"))
        rule("C6. IS sigma^2/n THE RIGHT NOISE? split-half check")
        print(f"{len(S)} SL player-seasons with 6+ matches: observed / expected squared "
              f"half-difference {S.diff2.mean() / S.expected.mean():.3f} (1.0 = right size)")
        print(S.groupby("season").apply(lambda x: round(x.diff2.mean() / x.expected.mean(), 3),
                                        include_groups=False).to_string())

        st = pd.read_csv(os.path.join(nf, "stayers.csv"))
        rule("C7. REFERENCE: SL players who stayed, from their own previous SL season")
        q75 = m.n_tgt.quantile(.75)
        small = st[st.n_games <= q75]
        print(f"{len(st)} player-seasons, MAE {mae(st, 'pred', 'class_score'):.2f}; with no more "
              f"than {q75:.0f} target matches ({len(small)}): {mae(small, 'pred', 'class_score'):.2f}")
        print("note claims: about 16 points")

    tt = os.path.join(DATA, "team_role_trend")
    if os.path.isdir(tt):
        rule("C8. TEAM CONTEXT, EXPECTED ROLE AND TREND as first tested - superseded by R1")
        ev = pd.read_csv(os.path.join(tt, "eval.csv"))
        base = {"trend": ["trend"], "source starts share": ["start_share"],
                "old team margin": ["old_margin"], "new club margin": ["new_margin"],
                "old team xLadder": ["old_xl"], "new club xLadder": ["new_xl"],
                "role (a) incumbents": ["incumbents"],
                "role (b) vacated [look-ahead]": ["vacated_LOOKAHEAD"]}
        feats = dict(base)
        feats["all, no look-ahead"] = [c for k, v in base.items() if "look-ahead" not in k for c in v]
        feats["all, with look-ahead"] = [c for v in base.values() for c in v]

        def design(df, cols, stats=None):
            X, st_ = [], stats or {}
            for c in cols:
                v = df[c].astype(float)
                mu, sd = st_.get(c, (v.mean(), v.std() or 1.0))
                st_[c] = (mu, sd)
                X += [((v - mu) / sd).fillna(0.0).values, v.isna().astype(float).values]
            X.append(np.ones(len(df)))
            return np.column_stack(X), st_

        worst = 0.0
        for name, cols in feats.items():
            pred = pd.Series(np.nan, index=ev.index)
            for o in sorted(ev.origin.unique()):
                tr = pd.read_csv(os.path.join(tt, f"train_{o}.csv"))
                X, st_ = design(tr, cols)
                coef, *_ = np.linalg.lstsq(X, (tr.class_target - tr.line).values, rcond=None)
                sel = ev.origin == o
                Xt, _ = design(ev[sel], cols, st_)
                pred[sel] = np.clip(ev.line_direction[sel].values + Xt @ coef, 0, 100)
            ev[f"re::{name}"] = pred
            worst = max(worst, float((pred - ev[f"ext::{name}"]).abs().max()))
        print(f"refitted corrections match the exported ones to {worst:.6f}\n")
        print("line + correction against the line alone, positive = the correction helps:")
        for lab, sub in (("into Super League", ev[ev.target == "SL"]), ("all moves", ev)):
            print(f"  {lab} (n={len(sub)}), line MAE {mae(sub, 'line_direction'):.2f}")
            for name in feats:
                print(f"    {name:32s} {fmt(boot(sub, 'line_direction', f're::{name}'))}")
        cov = ev[[c for v in base.values() for c in v]].notna().groupby(ev.origin).mean().round(2)
        print("\nshare of evaluated moves with each feature present, by origin:")
        print(cov.T.to_string())

        rc = os.path.join(tt, "starts_rule_check.csv.gz")
        if os.path.exists(rc):
            rule("C9. STARTS REBUILT FROM INTERCHANGE COUNTS, checked on match-sheet rows")
            r = pd.read_csv(rc)
            dec = r.rule.notna()
            print(f"{len(r):,} match-sheet rows; the rule decides {dec.mean():.1%}, "
                  f"agreeing with the sheet on {(r.rule[dec] == r.truth[dec]).mean():.4%}")
            amb = r[~dec]
            print(f"undecided rows, 'started' if 40+ minutes: {((amb.minutes >= 40) == amb.truth).mean():.3f}")
            print(r[dec].groupby("comp").apply(lambda x: round(float((x.rule == x.truth).mean()), 3),
                                               include_groups=False).to_string())
            print(f"decided rows that disagree with the sheet: {int((r.rule[dec] != r.truth[dec]).sum())} "
                  f"of {int(dec.sum()):,}")

    rt = os.path.join(DATA, "retest_r6")
    if os.path.isdir(rt):
        rule("R1. THE RE-TEST: one cohort, history baseline, joint fit (retest_r6.py)")
        C = pd.read_csv(os.path.join(rt, "retest_cohort.csv"))
        sc = pd.read_csv(os.path.join(rt, "retest_scored.csv"))
        cands = {"trend": "trend", "old team margin": "old_margin",
                 "new club margin": "new_margin", "source starts share": "start_share",
                 "incumbents at the new club": "incumbents"}
        specs = {"line": [], "line + has_history": ["has_history"]}
        for k, v in cands.items():
            specs[f"+ {k}"] = ["has_history", v]
        specs["+ all candidates"] = ["has_history"] + list(cands.values())

        def line_terms(train, frame):
            big = [p for p, n in train.pair.value_counts().items() if n >= 25]
            cols = [np.ones(len(frame)), frame.class_source.values]
            for p in big:
                on = (frame.pair == p).astype(float).values
                cols += [on, on * frame.class_source.values]
            return np.column_stack(cols)

        def fit_predict(train, test, extra, alpha=1.0):
            Xl, Tl = line_terms(train, train), line_terms(train, test)
            Xe, Te = [], []
            for c in extra:
                v, t = train[c].astype(float), test[c].astype(float)
                mu, sd = v.mean(), (v.std() or 1.0)
                if np.isnan(mu):
                    continue
                Xe.append(((v - mu) / sd).fillna(0.0).values)
                Te.append(((t - mu) / sd).fillna(0.0).values)
                if c != "has_history":
                    Xe.append(v.isna().astype(float).values)
                    Te.append(t.isna().astype(float).values)
            X = np.column_stack([Xl] + Xe) if Xe else Xl
            T = np.column_stack([Tl] + Te) if Te else Tl
            k = X.shape[1] - Xl.shape[1]
            A = np.vstack([X, np.hstack([np.zeros((k, Xl.shape[1])), math.sqrt(alpha) * np.eye(k)])])
            coef, *_ = np.linalg.lstsq(A, np.concatenate([train.class_target.values, np.zeros(k)]),
                                       rcond=None)
            return np.clip(T @ coef, 0, 100)

        re_ = []
        for T_ in (2023, 2024, 2025):
            tr, te = C[C.season_tgt < T_], C[C.season_tgt == T_].copy()
            for name, extra in specs.items():
                te[f"re::{name}"] = fit_predict(tr, te, extra)
            re_.append(te)
        re_ = pd.concat(re_, ignore_index=True)
        key_ = ["player_id", "source", "target", "season_tgt"]
        j = re_.merge(sc[key_ + [f"p::{k}" for k in specs]], on=key_)
        worst = max(float((j[f"re::{k}"] - j[f"p::{k}"]).abs().max()) for k in specs)
        print(f"refitted from retest_cohort.csv; matches the exported forecasts to {worst:.6f}")
        for lab, g, ref in (("into Super League (primary)", j[j.target == "SL"], "17.72"),
                            ("all moves", j, "20.22")):
            print(f"\n{lab}, n={len(g)}")
            print(f"  line MAE {mae(g, 're::line'):.2f}   (shipped-construction line on the same "
                  f"moves in the main backtest: {ref})")
            print(f"  line + has_history vs line: {fmt(boot(g, 're::line', 're::line + has_history'))}")
            for k in [k for k in specs if k.startswith("+ ")]:
                print(f"  {k:32s} vs line + has_history: "
                      f"{fmt(boot(g, 're::line + has_history', f're::{k}'))}")
        print("\nnote claims, into SL: has_history +0.85 (+0.20 to +1.54); incumbents +0.18, all "
              "moves +0.53 (+0.12 to +0.92)")

    rule("R2. THE APP'S LINE RULE: own line from 25 pairs, else pooled over the layer")
    tp_all = pd.read_csv(os.path.join(DATA, "translation_pairs_v3.csv"))
    tp_all = tp_all[tp_all.layer == "B_next_season"]
    for src_, tgt_ in (("NRL", "SL"), ("NSW", "SL"), ("QLD", "SL"), ("SL", "NSW")):
        own = tp_all[(tp_all.source == src_) & (tp_all.target == tgt_)]
        use, basis = (own, "direction") if len(own) >= 25 else (tp_all, "pooled")
        k, c = np.polyfit(use.class_source, use.class_target, 1)
        print(f"{src_}->{tgt_}: {basis:9s} n={len(use):5d}  slope {k:.4f}  intercept {c:.3f}")
    print("app (predict_translation._line): NRL->SL direction 107 0.3704 42.958; NSW->SL 45 "
          "0.2806 47.662;\nQLD->SL 32 0.3223 37.371; SL->NSW pooled 1512 0.2624 40.578")

    rule("R3. THE COMPARISON CARD'S RULE (predict_translation.comparables)")
    group = {"Full Back": "Fullback", "Winger": "Winger", "Centre": "Centre",
             "Five-Eighth": "Halves", "Half Back": "Halves", "Hooker": "Hooker",
             "Prop": "Middles", "Lock": "Middles", "Second Row": "Edge", "Interchange": "Bench"}

    def comparables(score, src_, tgt_, grp=None, window=7.5, minimum=8):
        p = tp_all[(tp_all.source == src_) & (tp_all.target == tgt_)].copy()
        p["group"] = p.raw_position.map(group)
        for w in (window, 2 * window):
            near = p[(p.class_source - score).abs() <= w]
            same = near[near.group == grp] if grp else near.iloc[0:0]
            rows, basis = ((same, f"same position group ({grp})") if len(same) >= minimum
                           else (near, "any position") if len(near) >= minimum else (None, None))
            if rows is not None:
                return len(rows), basis, w, rows.class_target.quantile([.1, .25, .5, .75, .9]).round(2).tolist()
        return 0, "not enough data", None, []

    for probe in ((60, "NRL", "SL", "Middles"), (75, "QLD", "SL", "Halves"),
                  (2, "QLD", "SL", None), (60, "NRL", "NSW", "Centre")):
        print(probe, "->", comparables(*probe))
    print("app: (60 NRL SL Middles) 21, any position, 7.5, median 66.59; (75 QLD SL Halves) 13,\n"
          "any position, 15.0, median 51.32; (2 QLD SL) not enough data; (60 NRL NSW Centre) 9,\n"
          "same position group, 7.5, median 75.21")

    rule("R4. THE LADDER'S SIGN CHECK shown on the translation page")
    lad = pd.read_csv(os.path.join(DATA, "translation_ladder_v3.csv"))
    lad = lad[lad.layer == "B_next_season"].set_index(["source", "target"]).shift_pts
    pairs_ = [(a_, b_) for (a_, b_) in lad.index if (b_, a_) in lad.index and a_ < b_]
    odd = [f"{a_}<->{b_}" for a_, b_ in pairs_ if lad[(a_, b_)] * lad[(b_, a_)] >= 0]
    print(f"{len(pairs_) - len(odd)} of {len(pairs_)} direction pairs have opposite signs; "
          f"not: {odd}   (app: 5 of 6; Queensland Cup <-> Super League does not)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
