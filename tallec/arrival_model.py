# -*- coding: utf-8 -*-
"""Does he get there at all — the question the rating model does not answer.

Every evaluation in this project so far has been conditional on arrival: a player is
scored only once he has played enough matches in the target competition to carry a
rating. The three external reviews all said the same thing about that, and the third put
it plainly — a club needs two models, not one.

    1. will this player reach a usable role in that competition?
    2. how good will he be once he does?

`rolling_backtest.py` answers the second. This answers the first, on the same explicit
cohort (`transition_events.py`) and the same rolling origins, so the two can be read
together.

**What the number means, exactly.** The population is everyone who played at least three
matches in a source competition last season, set against every other competition he
could have entered. `rated` is whether he then played at least three matches there. So a
probability here is:

    of players at this level in this competition, what share turn up in that one

and NOT

    of the players a club signed, what share worked out

We have no signing or registration data, so the second is unmeasurable here — a man who
was never wanted and a man who was signed and never picked are the same row. The model
therefore mixes club interest with player quality and cannot separate them. That is a
limit of the data, not of the fit, and it is the single most important sentence on this
page: it means a low probability is *not* evidence against a player a club has already
decided to sign.

**Why a probability and not a classifier.** The base rate is about 5%, so a model that
predicts "no" every time is right 95% of the time and useless. Nothing here reports
accuracy. What is reported is discrimination (does it rank arrivals above non-arrivals),
calibration (when it says 20%, do a fifth of them arrive), and lift over the base rate,
which is what a shortlist actually consumes.

    python arrival_model.py
    python arrival_model.py --origins 2024,2025 --outcome arrived
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import rolling_backtest as rb
import sp_schema as sp
import transition_events as te

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
OUT = os.path.join(BASE, "ARRIVAL_REPORT.md")
ENTRY_TYPES = ("first", "returning")
# The competition the client recruits INTO. Leeds Rhinos plays in Super League.
CLIENT_TARGET = "SL"
CLIENT_SOURCES = ("NRL", "NSW", "QLD")
# the Australian second tier, which feeds the NRL rather than Super League
FEEDERS = ("NSW", "QLD")
# a direction needs this many arrivals in the training window before it gets its own term
MIN_PAIR_ARRIVALS = 10
C = 1.0


def at_risk(con, through, cum, ext, pos, dob, types=ENTRY_TYPES):
    """Every candidate entry up to `through`, with what was known before each season.

    One row per player, target competition and season, whether or not he turned up. The
    source rating is the cumulative Class score as it stood at the end of the season
    before, which is what a recruiter would have been looking at.
    """
    ev = te.build(con, through=through)
    ev = ev[ev.transition_type.isin(types)].copy()
    if ev.empty:
        return ev

    src = cum.merge(ext, on=["player_id", "season", "comp"], how="left")
    src = src.rename(columns={"class_score": "class_source", "comp": "source",
                              "season": "season_src"})
    keep = ["player_id", "source", "season_src", "class_source", "n_games",
            "games_src", "mins_pg"]
    src = src[[c for c in keep if c in src.columns]]

    d = ev.merge(src, left_on=["player_id", "source", "forecast_from_season"],
                 right_on=["player_id", "source", "season_src"], how="inner")
    if d.empty:
        return d
    d["raw_position"] = [pos.get((p, c)) for p, c in zip(d.player_id, d.source)]
    d["group"] = d.raw_position.map(sp.POSITION_GROUP)
    d["dob"] = sp.parse_dob(d.player_id.map(dob))
    d["age"] = sp.age_at(d["dob"], d.forecast_from_season)
    d["pair"] = d.source + "->" + d.target
    return d


NUMERIC = ["class_source", "source_matches", "source_minutes", "mins_pg", "age"]
# How hard a direction's own slope is pooled toward the common one. Applied after
# standardisation, for the reason `translation_features.pooling_multipliers` records:
# before it, the scaler divides it away.
SLOPE_POOLING = 0.35


def design(d, pairs, groups, medians=None):
    """Numeric columns, direction and position dummies, and a direction-specific slope.

    The last part is the fix for what the fourth external review found. The first version
    gave every direction its own intercept and forced them all to share one set of
    slopes. That is a coefficient compromise, and on this data it was ruinous: minutes
    played in the NSW Cup predict an NRL call-up on their own at AUC 0.71, and the fitted
    model managed 0.44 — worse than the single raw feature it had been given. A pathway
    out of the Queensland Cup and a pathway out of Super League are not the same
    mechanism and cannot share a coefficient.

    Each direction now gets its own slope on every numeric feature, centred so it reads
    as a deviation from the common slope, and pooled toward it so a thin direction cannot
    invent its own story.
    """
    X = pd.DataFrame(index=range(len(d)))
    med = {} if medians is None else dict(medians)
    for c in NUMERIC:
        v = pd.to_numeric(d[c], errors="coerce").reset_index(drop=True) \
            if c in d.columns else pd.Series(np.nan, index=range(len(d)))
        if medians is None:
            med[c] = float(v.median()) if v.notna().any() else 0.0
        X[c] = v.fillna(med[c]).values
        X[f"{c}_missing"] = v.isna().astype(float).values
    pair_col = d.pair.reset_index(drop=True)
    for p in pairs:
        ind = (pair_col == p).astype(float).values
        X[f"pair_{p}"] = ind
        for c in NUMERIC:
            X[f"px_{p}__{c}"] = ind * (X[c].values - med[c])
    for g in groups:
        X[f"grp_{g}"] = (d["group"].reset_index(drop=True) == g).astype(float).values
    return X, med


def pooling_multipliers(columns):
    """Applied AFTER standardisation; see `design` and the note in
    `translation_features.pooling_multipliers` about why the order matters."""
    return np.array([SLOPE_POOLING if c.startswith("px_") else 1.0 for c in columns],
                    dtype=float)


def fit(train, outcome):
    y = train[outcome].astype(int).values
    if y.sum() < 20 or y.sum() == len(y):
        return None
    arrivals = train[train[outcome].astype(bool)].pair.value_counts()
    pairs = sorted(arrivals[arrivals >= MIN_PAIR_ARRIVALS].index)
    groups = sorted(g for g in train["group"].dropna().unique())
    X, med = design(train, pairs, groups)
    sc = StandardScaler().fit(X)
    mult = pooling_multipliers(list(X.columns))
    # balanced weights so the fit is not dominated by the 95% who never move; the
    # probabilities are recalibrated to the true base rate afterwards
    m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(
        sc.transform(X) * mult, y)
    return dict(model=m, scaler=sc, pairs=pairs, groups=groups, medians=med, mult=mult,
                base=float(y.mean()), n=len(y), arrivals=int(y.sum()))


def predict(f, rows):
    X, _ = design(rows, f["pairs"], f["groups"], medians=f["medians"])
    p = f["model"].predict_proba(f["scaler"].transform(X) * f["mult"])[:, 1]
    # undo the balancing: the fit saw a 50/50 world, the competition is not one
    w = f["base"] / (1.0 - f["base"])
    odds = p / np.clip(1.0 - p, 1e-9, None)
    return odds * w / (1.0 + odds * w)


def auc(y, p):
    """Rank-based, so it is unaffected by the calibration step above."""
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(p).rank().values
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p):
    return float(np.mean((np.asarray(y, dtype=float) - np.asarray(p)) ** 2))


def calibration(d, outcome, bins=(0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01)):
    rows = []
    cut = pd.cut(d.p_arrive, bins=list(bins), right=False)
    for b, g in d.groupby(cut, observed=True):
        rows.append(dict(predicted=f"{b.left:.0%}-{b.right:.0%}", n=len(g),
                         mean_predicted=float(g.p_arrive.mean()),
                         actual=float(g[outcome].mean()),
                         arrivals=int(g[outcome].sum())))
    return pd.DataFrame(rows)


def lift(d, outcome, top=(0.05, 0.10, 0.25)):
    """What a shortlist actually consumes: take the top N%, how many arrivals are in it."""
    rows = []
    base = float(d[outcome].mean())
    for frac in top:
        k = max(1, int(round(len(d) * frac)))
        sel = d.nlargest(k, "p_arrive")
        rows.append(dict(shortlist=f"top {frac:.0%}", players=k,
                         arrivals_caught=int(sel[outcome].sum()),
                         of_all_arrivals=float(sel[outcome].sum() / max(d[outcome].sum(), 1)),
                         hit_rate=float(sel[outcome].mean()),
                         lift=float(sel[outcome].mean() / base) if base else np.nan))
    return pd.DataFrame(rows)


def auc_ci(y, p, players=None, n=2000, seed=0):
    """AUC with a bootstrap interval, clustered on players where they are given.

    The first version of this report quoted bare AUCs and then declared two directions
    "inverted" on the strength of 0.44 and 0.39. The fourth external review pointed out
    that no interval had been put on them, which is exactly the objection this project
    has raised about other people's numbers. Below 0.5 on 34 arrivals is not obviously
    distinguishable from a coin.
    """
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    if y.all() or not y.any():
        return np.nan, np.nan, np.nan
    point = auc(y, p)
    rng = np.random.default_rng(seed)
    idx = (np.arange(len(y)) if players is None
           else None)
    if players is None:
        draws = [rng.integers(0, len(y), len(y)) for _ in range(n)]
    else:
        players = np.asarray(players)
        uniq = pd.unique(players)
        at = {q: np.where(players == q)[0] for q in uniq}
        draws = [np.concatenate([at[q] for q in rng.choice(uniq, uniq.size, True)])
                 for _ in range(n)]
    vals = []
    for take in draws:
        a = auc(y[take], p[take])
        if not np.isnan(a):
            vals.append(a)
    if len(vals) < 50:
        return point, np.nan, np.nan
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


BASELINE = "source_minutes"      # the single column a new model must beat, fixed in advance


def paired_auc_ci(g, outcome, model_col, base_col, n=2000, seed=0):
    """Bootstrap the DIFFERENCE between two AUCs on the same rows, clustered on players.

    Resampling the same players for both predictors keeps the comparison paired, which
    is the only way the difference can carry an interval worth reading: the two AUCs are
    computed on identical data and move together.
    """
    y = np.asarray(g[outcome], dtype=bool)
    if y.all() or not y.any():
        return np.nan, np.nan, np.nan
    m = np.asarray(g[model_col], dtype=float)
    v = pd.to_numeric(g[base_col], errors="coerce")
    b = np.asarray(v.fillna(v.median()), dtype=float)
    point = auc(y, m) - auc(y, b)
    players = np.asarray(g.player_id) if "player_id" in g else np.arange(len(g))
    uniq = pd.unique(players)
    if len(uniq) < 8:
        return point, np.nan, np.nan
    at = {q: np.where(players == q)[0] for q in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        take = np.concatenate([at[q] for q in rng.choice(uniq, uniq.size, True)])
        a1, a2_ = auc(y[take], m[take]), auc(y[take], b[take])
        if not (np.isnan(a1) or np.isnan(a2_)):
            vals.append(a1 - a2_)
    if len(vals) < 50:
        return point, np.nan, np.nan
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def baselines(d, outcome, min_arrivals=8):
    """What a single raw column achieves inside each direction, with no model at all.

    This is the bar the fourth review said was missing, and it was the finding: minutes
    played in the source competition rank NSW Cup arrivals at AUC 0.71 on their own,
    while the first version of this model — which had been given that column — managed
    0.44. A model that cannot beat one of its own inputs is not measuring something
    subtle, it is broken, and nothing in the old report would have shown that.
    """
    rows = []
    for pair, g in d.groupby("pair"):
        if g[outcome].sum() < min_arrivals:
            continue
        r = dict(direction=pair, n=len(g), arrivals=int(g[outcome].sum()))
        for c in NUMERIC:
            v = pd.to_numeric(g[c], errors="coerce")
            r[c] = auc(g[outcome], v.fillna(v.median()))
        r["model"] = auc(g[outcome], g.p_arrive)
        r["best_single"] = max(r[c] for c in NUMERIC if not np.isnan(r[c]))
        r["model_beats_best"] = "yes" if r["model"] > r["best_single"] else "NO"
        # The paired difference, with an interval. The fifth external review pointed out
        # that comparing a model's point estimate against the MAXIMUM of five observed
        # AUCs and calling it a win is the same fault this project had just corrected
        # elsewhere: a selected maximum is biased upward and neither figure carried an
        # interval. The comparison is therefore made against a baseline fixed in advance
        # — source minutes, the strongest single column and the one the review named —
        # and bootstrapped over players.
        pt, lo, hi = paired_auc_ci(g, outcome, "p_arrive", BASELINE)
        r["vs " + BASELINE] = pt
        r["ci_low"], r["ci_high"] = lo, hi
        r["clear"] = "yes" if (lo > 0 or hi < 0) else "no"
        rows.append(r)
    return pd.DataFrame(rows).sort_values("direction")


def within_direction(d, outcome, min_arrivals=8):
    """AUC inside each direction, which is the only situation a club is ever in.

    The pooled figure can look respectable purely because directions differ in how often
    anyone moves along them: ranking every NRL-to-NSW-Cup candidate above every
    Super-League-to-NRL one scores well and tells a recruiter nothing, because he is
    never choosing between those two men. He has a target competition and a list of
    feeder players, and the question is which of THOSE to sign.
    """
    rows = []
    for pair, g in d.groupby("pair"):
        if g[outcome].sum() < min_arrivals:
            continue
        rows.append(dict(direction=pair, n=len(g), arrivals=int(g[outcome].sum()),
                         base_rate=float(g[outcome].mean()),
                         auc=auc(g[outcome], g.p_arrive)))
    t = pd.DataFrame(rows).sort_values("auc", ascending=False)
    weighted = (float((t.auc * t.arrivals).sum() / t.arrivals.sum())
                if len(t) and t.arrivals.sum() else np.nan)
    return t, weighted


def md(df, fmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, rule] + body)


def run(origins, outcome):
    con = sqlite3.connect(DB)
    got, fits = [], []
    for origin in origins:
        print(f"origin {origin}: what was known at the end of {origin - 1} ...",
              flush=True)
        _, cum, ext, pos, dob = rb.knowledge_at(con, origin - 1)
        train = at_risk(con, origin - 1, cum, ext, pos, dob)
        train = train[train.season < origin]
        f = fit(train, outcome)
        if f is None:
            print("  too few arrivals to fit")
            continue
        now = at_risk(con, origin, cum, ext, pos, dob)
        now = now[now.season == origin].copy()
        if now.empty:
            continue
        now["p_arrive"] = predict(f, now)
        now["origin"] = origin
        got.append(now)
        fits.append(dict(origin=origin, train_rows=f["n"], train_arrivals=f["arrivals"],
                         train_base=f["base"], directions=len(f["pairs"]),
                         evaluated=len(now), evaluated_arrivals=int(now[outcome].sum())))
        print(f"  trained on {f['n']:,} candidates / {f['arrivals']} arrivals, "
              f"scored {len(now):,}")
    con.close()
    return (pd.concat(got, ignore_index=True) if got else pd.DataFrame(),
            pd.DataFrame(fits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", default="2023,2024,2025")
    ap.add_argument("--outcome", default="rated", choices=["rated", "arrived"],
                    help="'rated' is three or more matches there; 'arrived' is any")
    a = ap.parse_args()
    origins = [int(x) for x in a.origins.split(",")]
    d, fits = run(origins, a.outcome)
    if d.empty:
        print("nothing to report")
        return 1

    base = float(d[a.outcome].mean())
    W = ["# Does he get there at all\n",
         "Every other evaluation in this project is conditional on arrival — a player is "
         "scored once he has played enough in the target competition to carry a rating. "
         "This is the step before that, on the same cohort and the same rolling origins: "
         "for a player at a given level, what is the chance he reaches a usable role in "
         "another competition next season.\n",
         "\n**Read the population carefully.** It is everyone who played three or more "
         "matches in a source competition, set against every competition he could have "
         "entered. So this measures *of players at this level, what share turn up*, and "
         "not *of the players a club signed, what share worked out*. There is no signing "
         "or registration data here, so a man nobody wanted and a man who was signed and "
         "never picked are the same row. A low probability is therefore not evidence "
         "against a player a club has already decided to sign — it is a statement about "
         "how often men like him appear.\n",
         f"\nOutcome: **{a.outcome}** "
         + ("(three or more matches in the target competition)" if a.outcome == "rated"
            else "(any appearance in the target competition)")
         + f". Base rate {base:.2%} over {len(d):,} candidate entries.\n",
         "\n## What was fitted\n", md(fits, "{:.4f}"),
         "\n\n## Does it rank arrivals above the rest\n"]

    rows = []
    for origin, g in d.groupby("origin"):
        rows.append(dict(origin=origin, n=len(g), arrivals=int(g[a.outcome].sum()),
                         base_rate=float(g[a.outcome].mean()),
                         auc=auc(g[a.outcome], g.p_arrive),
                         brier=brier(g[a.outcome], g.p_arrive),
                         brier_base=brier(g[a.outcome],
                                          np.full(len(g), g[a.outcome].mean()))))
    per = pd.DataFrame(rows)
    W.append(md(per, "{:.4f}"))
    W.append("\n\nAUC is the chance a randomly chosen arrival is ranked above a randomly "
             "chosen non-arrival; 0.5 is a coin. `brier_base` is what predicting the "
             "base rate for everyone would score, so the model has to beat it to be "
             "worth anything.\n")
    a_mean = float(per.auc.mean())
    beat = int((per.brier < per.brier_base).sum())
    W.append(f"\nAcross {len(per)} origins the mean AUC is **{a_mean:.3f}**, and the "
             f"model beats the base rate on Brier score in {beat} of {len(per)}.\n")

    # ── the diagnostic that decides whether any of this is usable ────────────
    W.append("\n## Inside a direction, which is the only place a club stands\n")
    W.append("The pooled figure above can look respectable for a reason that helps "
             "nobody: directions differ enormously in how often anyone moves along "
             "them, and ranking every NRL-to-NSW-Cup candidate above every "
             "Super-League-to-NRL one scores well while telling a recruiter nothing. He "
             "is never choosing between those two men. He has one target competition and "
             "a list of feeder players, and the question is which of THOSE to sign. So "
             "the same model is scored again inside each direction.\n")
    wd, weighted = within_direction(d, a.outcome)
    W.append(md(wd, "{:.4f}"))
    W.append("\n\n### Against the simplest thing that could work\n")
    W.append("A model given a column has to beat that column. The first version of this "
             "model did not, and nothing in the old report would have shown it: minutes "
             "played in the source competition rank NSW Cup arrivals at 0.71 on their "
             "own, while the fitted model managed 0.44. That is what a coefficient "
             "compromise looks like — one slope per feature, shared across pathways that "
             "do not work the same way.\n")
    W.append(md(baselines(d, a.outcome), "{:.3f}"))
    if not np.isnan(weighted):
        W.append(f"\n\nArrival-weighted mean AUC inside a direction: **{weighted:.3f}**, "
                 f"against {a_mean:.3f} pooled.\n")
        if weighted < 0.55:
            W.append("**That is a coin, and it is the finding.** Nearly all the apparent "
                     "skill in the pooled number is the model learning which directions "
                     "are busy, not which players move.\n")
        else:
            W.append(f"**There is player-level signal inside a direction**, which the "
                     f"first version of this report denied. It concluded that the data "
                     f"held nothing usable, and the fourth external review refuted that "
                     f"in one line: source minutes alone rank NSW Cup arrivals at 0.71. "
                     f"The failure was the specification — one slope per feature shared "
                     f"across every pathway — and not the data.\n")
        fn_rows = wd[wd.direction.isin([f"{s}->NRL" for s in FEEDERS])]
        if len(fn_rows):
            bits = []
            for r in fn_rows.itertuples():
                g = d[d.pair == r.direction]
                pt, lo, hi = auc_ci(g[a.outcome], g.p_arrive, g.player_id)
                bits.append(f"{r.direction} {pt:.3f} [{lo:.3f}, {hi:.3f}]")
            W.append("\nThe two feeder-to-NRL directions, now with intervals — which the "
                     "previous version quoted without, calling them the client's "
                     "and inverted on the strength of a bare 0.44 and 0.39. They "
                     "are not the client's: Leeds recruits into Super League, "
                     "measured below. "
                     + "; ".join(bits) + ".\n")

    W.append("\n## Is it calibrated\n")
    W.append("When it says twenty per cent, do a fifth of them arrive? This matters more "
             "than discrimination for a recruiter, because the number is read as a "
             "probability rather than as a rank.\n")
    W.append(md(calibration(d, a.outcome), "{:.4f}"))

    W.append("\n\n## What a shortlist gets\n")
    W.append("The practical form of the question. Rank every candidate by the model, take "
             "the top slice, and see how many of the season's actual arrivals are in it.\n")
    W.append(md(lift(d, a.outcome), "{:.3f}"))
    W.append(f"\n`lift` is the hit rate in the slice divided by the {base:.2%} base "
             f"rate.\n")

    # The direction the client actually asks about. Leeds Rhinos is a Super League club,
    # so its question is who arrives in Super League. Until the fifth external review on
    # 2026-09-29 this section reported feeder-to-NRL under that heading, which is a
    # pathway into a competition the client does not recruit into.
    for label, frame, note in (
            ("Into Super League — the client's direction",
             d[d.source.isin(CLIENT_SOURCES) & (d.target == CLIENT_TARGET)],
             "Leeds recruits into Super League. This is who arrives there."),
            ("Feeder to NRL — not the client's direction",
             d[d.source.isin(FEEDERS) & (d.target == "NRL")],
             "Reported because an Australian club would ask it, and because this "
             "pathway carries the most arrivals.")):
        if len(frame) <= 50:
            continue
        W.append(f"\n\n## {label}\n")
        W.append(note + "\n")
        W.append(md(pd.DataFrame([dict(
            n=len(frame), arrivals=int(frame[a.outcome].sum()),
            base_rate=float(frame[a.outcome].mean()),
            auc=auc(frame[a.outcome], frame.p_arrive),
            brier=brier(frame[a.outcome], frame.p_arrive),
            brier_base=brier(frame[a.outcome],
                             np.full(len(frame), frame[a.outcome].mean())))]), "{:.4f}"))
        pt, lo, hi = paired_auc_ci(frame, a.outcome, "p_arrive", BASELINE)
        if not np.isnan(lo):
            clear = lo > 0 or hi < 0
            W.append(f"\nAgainst `{BASELINE}` alone: {pt:+.3f} AUC [{lo:+.3f}, "
                     f"{hi:+.3f}]"
                     + (", clear of zero.\n" if clear else ", which contains zero.\n"))
        W.append(md(lift(frame, a.outcome), "{:.3f}"))

    W.append("\n\n## Verdict\n")
    if not np.isnan(weighted) and weighted < 0.55:
        W.append("**Not usable as a per-player probability, and not shippable at all for "
                 "the client's own direction.** What the exercise does establish is the "
                 "base rates, which are worth having and were written down nowhere "
                 "before: of players at a given level in a feeder competition, a few per "
                 "cent reach a usable role in the NRL in a given season. That is a "
                 "corrective to any conversation that starts from a rating alone. What "
                 "it does not establish is which of them.\n")
        W.append("\nThe honest next step is not a better classifier on this data. It is "
                 "the data the third review named — squad membership, contracts, "
                 "registrations — anything separating 'nobody wanted him' from 'he was "
                 "signed and not picked'. Until then the outcome is half club decision "
                 "and half player quality, and no fit can pull those apart.\n")
    else:
        bl = baselines(d, a.outcome)
        beats = int((bl.model_beats_best == "yes").sum())
        clear_beats = int((bl["clear"] == "yes").sum()) if "clear" in bl else 0
        W.append(f"**There is usable signal, and the model finds it — in some "
                 f"directions.** It beats the best single raw column in {beats} of "
                 f"{len(bl)} directions on point estimates. Measured properly, against "
                 f"`{BASELINE}` fixed in advance with a paired interval, the advantage "
                 f"is clear of zero in {clear_beats} of {len(bl)}. The fifth external "
                 f"review was right that comparing a model against the maximum of five "
                 f"observed AUCs, with no interval on either, is the same fault this "
                 f"project had just corrected elsewhere — a selected maximum is biased "
                 f"upward and neither number said how sure it was.\n")
        W.append("\nIt is still not a probability to quote at a player. The directions "
                 "where it loses to a single column are the thin ones, the calibration "
                 "above over-predicts in the upper bins, and the population is everyone "
                 "at a level rather than everyone a club wanted. Use it to order a "
                 "shortlist, not to put a number beside a name.\n")
        W.append("\nWhat has NOT changed is the limit on what the outcome means. Without "
                 "squad, contract or registration data, a man nobody wanted and a man "
                 "signed and never picked are the same row, so this ranks pathway "
                 "arrival and not recruitment success. The third review asked for that "
                 "data and the request stands.\n")

    W.append("\n\n## What this is not\n")
    W.append("- Not a probability that a signing works out. See the population note "
             "above; nothing here observes a contract.\n")
    W.append("- Not independent of the rating model. Both are fed the same source Class "
             "score, so a player rated highly will tend to score highly on both, and the "
             "two numbers should be read as one picture rather than as agreement between "
             "two witnesses.\n")
    W.append("- Not a statement about a particular club's intentions. A Super League club "
             "that has already decided to sign a man should read the conditional rating "
             "and ignore this page.\n")

    open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(W) + "\n")
    print("\n" + per.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n" + lift(d, a.outcome).to_string(index=False,
                                              float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
