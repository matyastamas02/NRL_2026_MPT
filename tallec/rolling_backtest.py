# -*- coding: utf-8 -*-
"""Rolling-origin backtest of the competition translation model.

What this replaces. `VALIDATION_REPORT.md` section 6 measured the translation model by
applying the *current* model — fitted on everything through 2025 — to historical moves
that were part of its own training set. The external review of 2026-08-28 was right that
this establishes nothing: a model asked about data it has already learned will do well,
and the directional effect it appeared to find was exploratory rather than evidence.

What this does instead. For each target season T in turn:

  * the rating history is rebuilt using matches from T-1 and earlier and nothing later,
    so a player's source rating is what BOSC would have shown at the time;
  * translation pairs are built only from moves that had already *landed* by T-1, and
    the ladder and the model are fitted on those alone;
  * that model then predicts moves landing in T, and is scored against what those men
    actually did in T.

Nothing available only after the forecast is used to make it. The inner split for the
model is `GroupKFold` by player, so no player appears on both sides of a fold, and the
outer split is strictly chronological.

Who is evaluated. The set used to be inferred: every competition a player appeared in
last season joined to every competition he appeared in this one. The external review of
2026-09-22 showed that is not a transfer list. 15.5% of player-seasons span two
competitions, so a fringe forward shuttling between the NRL and the NSW Cup became two
opposite moves and was scored on both; 1,140 rows came from 531 players; and a row
existed only where the man later held a place, so everyone who moved and failed was
invisible.

`transition_events.py` now builds the cohort explicitly — one row per player, target and
season, one source fixed in advance, and a type read entirely from seasons before the one
being forecast. This scores the men genuinely entering a competition, first-timers and
returners, which is 441 entries over three origins rather than 1,140 inferred moves. It
is a smaller and much weaker sample, and the conclusions below say so where it matters.

    python rolling_backtest.py                 # writes ROLLING_REPORT.md
    python rolling_backtest.py --origins 2024,2025
"""
import argparse
import os
import sqlite3
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import aging
import fit_translation_v3 as f3
import rating_history as rh
import sp_schema as sp
import transition_events as te
import translation_features as tf

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
OUT = os.path.join(BASE, "ROLLING_REPORT.md")
MIN_GAMES = f3.MIN_GAMES
ALPHA = f3.ALPHA
# the cohort a recruitment question is about: men entering a competition,
# not men already shuttling across it
ENTRY_TYPES = ("first", "returning")
# The competition the client recruits INTO. Leeds Rhinos is a Super League club and the
# question the project exists to answer is "this NRL or second-tier Australian player —
# what would he do in Super League". From 2026-09-23 until the fifth external review on
# 2026-09-29 this file called feeder-to-NRL "the question Leeds asks" and reported it as
# the headline cohort. That is a pathway into the NRL, which Leeds does not recruit into.
# Three report regenerations carried the error.
CLIENT_TARGET = "SL"
CLIENT_SOURCES = ("NRL", "NSW", "QLD")
# the second-tier Australian competitions, which feed the NRL rather than Super League
FEEDERS = ("NSW", "QLD")
# how each type reads in a sentence
LONG = {"first": "first season in the competition",
        "returning": "returning after a season away",
        "dual_registered": "played in both competitions last season"}


def knowledge_at(con, through):
    """Everything the project would have held at the end of season `through`."""
    cum, sea, ext = rh.all_competitions(DB, through=through)
    pos = f3.career_position(con, through)
    dob = pd.read_sql("SELECT player_id, dob FROM players WHERE dob IS NOT NULL",
                      con).set_index("player_id").dob
    return f3.build_pairs(cum, sea, ext, pos, dob), cum, ext, pos, dob


def fit(pairs, layer):
    """Ladder and Ridge on one layer's pairs. Nothing here sees the target season."""
    p = pairs[pairs.layer == layer]
    if len(p) < 40:
        return None
    spec = tf.FeatureSpec.fit(p, pairs=sorted(p.pair.unique()))
    X, _ = spec.transform(p)
    # the backtest has to score the model that actually ships, so it honours the same
    # feature selection; otherwise it would keep reporting on a specification the
    # ablation retired
    keep = tf.select_columns(list(X.columns), f3.MODEL_FEATURES)
    X = X[keep]
    y = p.class_target.values
    sc = StandardScaler().fit(X)
    # pooling is applied AFTER standardisation; before it, the scaler divides it away
    mult = tf.pooling_multipliers(keep)
    model = Ridge(alpha=ALPHA).fit(sc.transform(X) * mult, y)
    # the inner split exists to show the fit is not memorising, not to tune anything
    n_splits = min(5, p.player_id.nunique())
    inner = np.nan
    if n_splits >= 2:
        pred = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, p.player_id.values):
            s2 = StandardScaler().fit(X.iloc[tr])
            pred[te] = Ridge(alpha=ALPHA).fit(
                s2.transform(X.iloc[tr]) * mult, y[tr]).predict(
                    s2.transform(X.iloc[te]) * mult)
        inner = float(np.sqrt(np.nanmean((y - pred) ** 2)))
    lad = (p.assign(d=p.class_target - p.class_source)
            .groupby(["source", "target"])["d"]
            .agg(["mean", "size"]).rename(columns={"mean": "shift", "size": "n"}))
    return dict(spec=spec, model=model, scaler=sc, ladder=lad, n=len(p), mult=mult,
                features=keep, players=int(p.player_id.nunique()), inner_rmse=inner,
                lines=straight_lines(p))


MIN_LINE_PAIRS = 25      # below this a direction borrows the pooled line
MIN_CLUSTERS = 8         # fewer distinct players than this and an interval is a fiction


def straight_lines(p):
    """Least-squares `target ~ source`, pooled and once per direction.

    The bar the model has to clear, and the one that was missing. Twice now a model in
    this project has been compared only against constants — a flat 50 and the untouched
    source rating — and twice a conclusion drawn from that turned out to be about the
    model rather than about the data. The arrival model scored 0.44 on a column that
    scores 0.71 by itself and nobody noticed, because nothing put the two side by side.

    A straight line from source to target is the least a conditional model can be asked
    to beat: it shrinks by exactly the right amount for the direction it is fitted on,
    and it costs two parameters. Fitted on the training window only, so it is a real
    baseline and not an oracle.
    """
    lines = {}
    fit_one = lambda g: np.polyfit(g.class_source.values, g.class_target.values, 1) \
        if len(g) >= 2 and g.class_source.std() > 0 else None
    lines["__pooled__"] = fit_one(p)
    for (s_, t_), g in p.groupby(["source", "target"]):
        if len(g) >= MIN_LINE_PAIRS:
            co = fit_one(g)
            if co is not None:
                lines[f"{s_}->{t_}"] = co
    return lines


def apply_lines(lines, rows, per_direction=True):
    """Predictions from `straight_lines`, falling back to the pooled fit."""
    out = np.full(len(rows), np.nan)
    pooled = lines.get("__pooled__")
    for i, (s_, t_, x) in enumerate(zip(rows.source, rows.target, rows.class_source)):
        co = lines.get(f"{s_}->{t_}") if per_direction else None
        co = co if co is not None else pooled
        if co is not None:
            out[i] = np.clip(co[0] * float(x) + co[1], 0, 100)
    return out


def predict(fitted, rows):
    """Ladder and model projections for a frame of pending moves."""
    lad = fitted["ladder"]
    shift = []
    for s_, t_ in zip(rows.source, rows.target):
        if (s_, t_) in lad.index:
            shift.append(float(lad.loc[(s_, t_), "shift"]))
        elif (t_, s_) in lad.index:
            shift.append(-float(lad.loc[(t_, s_), "shift"]))
        else:
            shift.append(np.nan)
    out = rows.copy()
    out["projected"] = np.clip(out.class_source + np.array(shift), 0, 100)
    X, _ = fitted["spec"].transform(rows)
    X = X[fitted["features"]]
    out["model"] = np.clip(
        fitted["model"].predict(fitted["scaler"].transform(X) * fitted["mult"]),
        0, 100)
    # the two straight lines the model has to beat, fitted on the training window
    out["line_pooled"] = apply_lines(fitted["lines"], rows, per_direction=False)
    out["line_direction"] = apply_lines(fitted["lines"], rows, per_direction=True)
    return out


def moves_into(con, origin, cum_prior, ext_prior, pos, dob, types=ENTRY_TYPES):
    """Players entering a competition in `origin`, from the explicit cohort.

    This used to infer moves by joining every competition a player appeared in last
    season to every competition he appeared in this one. The external review of
    2026-09-22 was right that the result is not a transfer list: 15.5% of player-seasons
    span two competitions, and each such man became two opposite moves, so a fringe
    forward shuttling between the NRL and the NSW Cup was scored as having moved both
    ways. 1,140 rows came from 531 players.

    `transition_events.py` now builds the cohort once: one row per player, target and
    season, a single source fixed in advance, and a type decided entirely from seasons
    before the one being forecast. `types` says which of those belong in a recruitment
    question — by default the men actually entering a competition, not the ones already
    shuttling across it.

    The source rating still comes from the history built through origin-1, and the
    target side is still a measurement of the season being forecast and is never fitted
    on.
    """
    ev = te.build(con, through=origin)
    ev = ev[(ev.season == origin) & ev.transition_type.isin(types) & ev.rated]
    if ev.empty:
        return ev

    _, sea_now, _ = rh.all_competitions(DB, through=origin)
    tgt = (sea_now[(sea_now.season == origin) & (sea_now.n_games >= MIN_GAMES)]
           .rename(columns={"class_score": "class_target", "comp": "target",
                            "n_games": "n_target", "raw_score": "class_target_raw"}))

    src = cum_prior[cum_prior.season == origin - 1].merge(
        ext_prior, on=["player_id", "season", "comp"], how="left")
    src = src[src.games_src >= MIN_GAMES].rename(
        columns={"class_score": "class_source", "comp": "source"})

    j = (ev[["player_id", "source", "target", "transition_type",
             "target_matches", "target_minutes"]]
         .merge(src.drop(columns=["season"], errors="ignore"),
                on=["player_id", "source"], how="inner")
         .merge(tgt[["player_id", "target", "class_target", "class_target_raw",
                     "n_target"]],
                on=["player_id", "target"], how="inner"))
    if j.empty:
        return j
    # Named apart on purpose. The column used to be a bare `season` holding the SOURCE
    # season, which reads as the season being forecast and is the opposite of what it
    # was — a confusion worth removing while the cohort is being rebuilt anyway.
    j["season_src"] = origin - 1
    j["season_tgt"] = origin
    j["pair"] = j.source + "->" + j.target
    j["layer"] = "B_next_season"
    j["raw_position"] = [pos.get((p, c)) for p, c in zip(j.player_id, j.source)]
    j["dob"] = sp.parse_dob(j.player_id.map(dob))
    j["age"] = sp.age_at(j["dob"], pd.Series(origin - 1, index=j.index))
    j["age_delta"] = aging.expected_delta(j["age"])
    return j


def beats_the_simple_thing(d, target="class_target", min_moves=20):
    """Per direction: the model against the simplest predictors, and whether it wins.

    Every predictor here is available to anyone with the training data and ten minutes.
    A conditional model that cannot beat a straight line fitted on the same direction is
    not adding conditioning, it is adding parameters — and until 2026-09-24 nothing in
    this project checked that. The arrival model spent a month scoring 0.44 on a feature
    worth 0.71 alone because the comparison did not exist.

    `line_direction` is the bar that matters. `line_pooled` is shown beside it so a
    reader can see how much of any advantage is simply knowing which direction it is.
    """
    cols = [("model", "conditional model"), ("line_direction", "straight line, this direction"),
            ("line_pooled", "straight line, pooled"), ("projected", "ladder"),
            ("class_source", "no translation"), ("flat50", "flat 50")]
    rows = []
    for pair, g in d.groupby("pair"):
        if len(g) < min_moves or g[target].isna().all():
            continue
        r = dict(direction=pair, n=len(g))
        for col, label in cols:
            if col not in g:
                continue
            e = (g[target] - g[col]).abs()
            r[label] = float(e.mean())
        simple = [r[l] for c, l in cols[1:] if l in r and not np.isnan(r[l])]
        if not simple or "conditional model" not in r:
            continue
        r["best simple"] = float(min(simple))
        r["model wins"] = "yes" if r["conditional model"] < r["best simple"] else "NO"

        # The interval, against the straight line chosen in advance as the bar rather
        # than against the best baseline picked afterwards — the second is a
        # post-hoc minimum over six candidates and its interval would mean nothing.
        # Positive is the model ahead, in points of MAE.
        if "line_direction" in g and "player_id" in g:
            b = boot(g, "line_direction", "model", target=target)
            if b is not None:
                r["vs line"], r["ci low"], r["ci high"] = b
                r["clear"] = "yes" if (b[1] > 0 or b[2] < 0) else "no"
        rows.append(r)
    if not rows:
        # an empty frame with no columns blows up on sort_values, and "no direction had
        # enough moves" is a legitimate answer rather than a crash
        return pd.DataFrame(columns=["direction", "n", "conditional model",
                                     "best simple", "model wins"])
    return pd.DataFrame(rows).sort_values("direction")


def score(d, target="class_target"):
    """Error of each predictor against one definition of the outcome.

    `target` exists because the choice is contested and should be reported rather than
    assumed. `class_target` is the shrunk season rating the app would publish; it is
    pulled toward 50, which is the number one of the baselines predicts, so a shrunk
    predictor is flattered by it. `class_target_raw` is the same season's unshrunk mean
    on the same scale — noisier, and not what anyone would publish, but not drawn toward
    any predictor. A finding that holds on both is a finding.
    """
    rows = []
    y = d[target]
    for label, p in [("translation (ladder)", d.projected),
                     ("conditional model", d.model),
                     ("no translation", d.class_source),
                     ("competition average (50)", pd.Series(50.0, index=d.index))]:
        ok = p.notna() & y.notna()
        if not ok.any():
            continue
        e = y[ok] - p[ok]
        rows.append(dict(predictor=label, n=int(ok.sum()),
                         mae=float(e.abs().mean()),
                         rmse=float((e ** 2).mean() ** 0.5),
                         bias=float(e.mean())))
    return pd.DataFrame(rows)


def boot(d, a, b, n=4000, seed=0, target="class_target"):
    """Bootstrap the paired error difference, resampling PLAYERS not rows.

    It was called a player-level bootstrap while resampling rows, which the external
    review of 2026-09-22 caught. The distinction matters here because the same man
    appears in several moves — 1,140 rows come from 531 players — and treating those
    rows as independent makes the interval narrower than the evidence supports.
    """
    ok = d[a].notna() & d[b].notna() & d[target].notna()
    g = d[ok]
    if len(g) < 8:
        return None
    diff = ((g[target] - g[a]).abs() - (g[target] - g[b]).abs()).values
    players = g.player_id.values
    uniq = pd.unique(players)
    if len(uniq) < MIN_CLUSTERS:
        # A clustered bootstrap over one or two players resamples the same rows every
        # time and returns an interval of zero width — perfect certainty exactly where
        # there is least. Say nothing instead.
        return float(diff.mean()), np.nan, np.nan
    at = {p: np.where(players == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    s = np.empty(n)
    for i in range(n):
        pick = rng.choice(uniq, uniq.size, replace=True)
        s[i] = np.concatenate([diff[at[p]] for p in pick]).mean()
    return float(diff.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def md(df, fmt="{:.2f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, rule] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", default="2023,2024,2025")
    a = ap.parse_args()
    origins = [int(x) for x in a.origins.split(",")]
    t0 = time.time()

    con = sqlite3.connect(DB)
    got, fits = [], []
    for origin in origins:
        print(f"origin {origin}: fitting on everything through {origin - 1} ...",
              flush=True)
        pairs, cum, ext, pos, dob = knowledge_at(con, origin - 1)
        f = fit(pairs, "B_next_season")
        if f is None:
            print(f"  too few pairs ({len(pairs)}), skipped")
            continue
        pend = moves_into(con, origin, cum, ext, pos, dob)
        if pend.empty:
            print("  no moves landing in this season")
            continue
        out = predict(f, pend)
        out["origin"] = origin
        # the cohort now comes with the row. `cohorts.classify` asked the database the
        # same question a second time and from the target season's side, which was one
        # more place for hindsight to get in; `transition_events` decides it from
        # seasons before the forecast and hands it over already attached.
        out["cohort"] = out["transition_type"]
        got.append(out)
        fits.append(dict(origin=origin, train_pairs=f["n"], train_players=f["players"],
                         inner_rmse=f["inner_rmse"], evaluated=len(out)))
        print(f"  trained on {f['n']} pairs / {f['players']} players, "
              f"predicted {len(out)} moves")
    con.close()

    if not got:
        print("nothing to report")
        return
    d = pd.concat(got, ignore_index=True)
    d["flat50"] = 50.0

    W, A = [], None
    A = W.append
    A("# Competition translation — rolling-origin backtest\n")
    A("Each season is forecast using only what was known before it. The rating history "
      "is rebuilt from matches up to the previous season, the ladder and the model are "
      "fitted on moves that had already landed by then, and the result is scored "
      "against what the players actually did in the season being forecast. The inner "
      "split is `GroupKFold` by player; the outer split is chronological.\n")
    A("This replaces section 6 of `VALIDATION_REPORT.md`, which applied the current "
      "model to historical moves it had been fitted on and therefore measured nothing. "
      "Errors below are in points of the 0-100 rating.\n")

    A("\n## What was fitted, and on what\n")
    A(md(pd.DataFrame(fits), "{:.2f}"))
    A("\n`inner_rmse` is the out-of-sample error inside the training window under "
      "GroupKFold — a check that the fit is not memorising players, not a forecast.\n")

    A("\n## The forecast, by origin\n")
    rows = []
    for origin, g in d.groupby("origin"):
        s = score(g)
        for _, r in s.iterrows():
            rows.append(dict(origin=origin, **r.to_dict()))
    A(md(pd.DataFrame(rows), "{:.2f}"))

    # ── is the model worth its own complexity? ────────────────────────────────
    A("\n\n## Against the simplest thing that could work\n")
    A("A conditional model has to beat a straight line. `target ~ source` fitted on the "
      "same training window, once per direction, costs two parameters and shrinks by "
      "exactly the right amount for that pathway — so anything the model adds has to be "
      "conditioning rather than flexibility. This comparison did not exist in this "
      "project until 2026-09-24, and its absence let an arrival model score 0.44 on a "
      "feature worth 0.71 by itself for a month.\n")
    A("\nThe line is a least-squares fit while the comparison is on MAE. An earlier "
      "version of this section called that a mismatch in the line's favour; the fifth "
      "external review refitted an MAE-optimal line and found it changes almost nothing "
      "— 20.20 against the least-squares line's 20.60 overall, with the model at 20.12 "
      "and still not separable on the client's direction. The remark was wrong and is "
      "withdrawn.\n")
    simple = beats_the_simple_thing(d)
    A(md(simple, "{:.2f}"))
    A("\n\n`vs line` is the model's mean absolute error subtracted from the "
      "direction-specific line's, so **positive means the model is ahead**, with a "
      "player-clustered interval. It is measured against the line chosen in advance as "
      "the bar rather than against `best simple`, which is a minimum over six "
      "candidates picked after the fact and would carry an interval that means "
      "nothing.\n")
    if len(simple) and "clear" in simple:
        won = int((simple["model wins"] == "yes").sum())
        ahead = simple[(simple["clear"] == "yes") & (simple["vs line"] > 0)]
        behind = simple[(simple["clear"] == "yes") & (simple["vs line"] < 0)]
        A(f"\nOn point estimates the model wins in **{won} of {len(simple)}** "
          f"directions. On intervals the picture is harder: it is clearly ahead in "
          f"{len(ahead)} and clearly **behind** in {len(behind)}"
          + (" — " + ", ".join(f"{r['direction']} by {-r['vs line']:.2f}"
                               for _, r in behind.iterrows()) if len(behind) else "")
          + ".\n")
        cl = d[d.source.isin(CLIENT_SOURCES) & (d.target == CLIENT_TARGET)]
        b = boot(cl, "line_direction", "model") if len(cl) > 20 else None
        if b is not None:
            clear = b[1] > 0 or b[2] < 0
            A(f"\n**On the client's own direction the model is not distinguishable from "
              f"a straight line.** Everything entering Super League, {len(cl)} moves, "
              f"{b[0]:+.2f} points with an interval of [{b[1]:+.2f}, {b[2]:+.2f}]"
              + (", clear of zero.\n" if clear else ", which contains zero.\n"))
            A("\nThat is the honest description of what the conditional model is worth "
              "where it is sold. It is not an argument for deleting it: a line cannot "
              "use position, cannot carry a missing-value flag, and cannot be quoted "
              "for a direction with too few moves to fit one. It is an argument against "
              "presenting the conditioning as the thing that makes the product work. "
              "What makes it work is the shrinkage — and `target ~ source` does that "
              "with two parameters.\n")

    # ── does the answer depend on what we call the outcome? ───────────────────
    A("\n\n## The same question against a different outcome\n")
    A("The outcome above is the shrunk season rating the app would publish, and it is "
      "pulled toward 50 — which is the number one of the baselines predicts. The review "
      "of 2026-09-22 argued that this flatters shrunk predictors, and it does. So every "
      "comparison is also run against the same season's **unshrunk** mean on the same "
      "scale: noisier, not something anyone would publish, but not drawn toward any "
      "predictor. A finding that holds on both is a finding; one that holds on only the "
      "first is a property of the scoring.\n")
    cmp_rows = []
    for tgt, name in (("class_target", "shrunk (published)"),
                      ("class_target_raw", "unshrunk season mean")):
        if tgt not in d.columns or d[tgt].isna().all():
            continue
        # `boot(x, y)` returns mean(|error of x| − |error of y|), so a NEGATIVE value
        # means the FIRST argument has less error. An earlier version of this table said
        # the opposite in its legend while the narrative underneath read it correctly,
        # which the third review caught. The columns are now named for what they hold
        # rather than relying on a sentence: `cost` is how much worse the predictor is
        # than its baseline, so positive is bad for the predictor.
        for pred, base, plabel, blabel in (
                ("projected", "flat50", "translation (ladder)", "a flat 50"),
                ("model", "flat50", "conditional model", "a flat 50"),
                ("projected", "class_source", "translation (ladder)",
                 "leaving the rating alone")):
            bb = boot(d, base, pred, target=tgt)
            if bb is None:
                continue
            cmp_rows.append(dict(outcome=name, predictor=plabel, against=blabel,
                                 cost=-bb[0], ci_low=-bb[2], ci_high=-bb[1],
                                 clear=("yes" if (bb[2] < 0 or bb[1] > 0) else "no")))
    cmp = pd.DataFrame(cmp_rows)
    A(md(cmp, "{:+.3f}"))
    A("\n`cost` is the predictor's mean absolute error minus the baseline's, so a "
      "**positive cost means the predictor is worse** than the baseline it is set "
      "against. `clear` is whether the player-clustered interval excludes zero.\n")
    if len(cmp):
        def verdict(outcome, predictor):
            r = cmp[(cmp.outcome == outcome) & (cmp.predictor == predictor)
                    & (cmp["against"] == "a flat 50")]
            return None if r.empty else r.iloc[0]
        a1 = verdict("shrunk (published)", "translation (ladder)")
        a2 = verdict("unshrunk season mean", "translation (ladder)")
        m1 = verdict("shrunk (published)", "conditional model")
        m2 = verdict("unshrunk season mean", "conditional model")
        if a1 is not None and a2 is not None and a1.clear == "yes" and a2.clear == "no":
            A("**The ladder's defeat does not survive the change of outcome.** Against "
              "the published rating it is beaten by a constant with an interval clear "
              "of zero; against the unshrunk mean the same comparison contains zero. "
              "The review was right that part of that result was the scoring rather "
              "than the ladder, and the claim is narrowed accordingly: the ladder is "
              "beaten by a constant at predicting *the number this system publishes*, "
              "which is a real and useful statement about the product, and is not "
              "established as a statement about the player.\n")
        if (m1 is not None and m2 is not None
                and m1.clear == "yes" and m2.clear == "yes"):
            A("**The decision that actually ships does survive it.** The conditional "
              "model beats a flat 50 on both outcomes, and by more on the unshrunk one "
              f"({-m2.cost:+.2f} points against {-m1.cost:+.2f}). Showing the model "
              "rather than the ladder is therefore not an artefact of how the outcome "
              "was defined, which is the one thing here a club depends on.\n")

    # ── the one cohort the client actually asks about ────────────────────────
    A("\n\n## The headline use case, on its own\n")
    A("Leeds Rhinos is a Super League club, so the question the project exists to answer "
      "is: this man is playing in the NRL, the NSW Cup or the Queensland Cup — what "
      "would he do in Super League. Every figure above pools that with moves in other "
      "directions. This is that cohort alone.\n")
    A("\nIt is also a correction. From 23 September until the fifth external review on "
      "the 29th this section reported feeder-to-NRL as the client's cohort and called it "
      "the question Leeds asks. That is a pathway *into* the NRL, which Leeds does not "
      "recruit into, and the mistake survived three regenerations of this report. What "
      "it said was true of feeder-to-NRL and wrong about the client.\n")
    cl = d[d.source.isin(CLIENT_SOURCES) & (d.target == CLIENT_TARGET)]
    if len(cl) >= 20:
        rows = []
        for tgt, oname in (("class_target", "shrunk (published)"),
                           ("class_target_raw", "unshrunk season mean")):
            if tgt not in cl.columns or cl[tgt].isna().all():
                continue
            s_ = score(cl, tgt)
            for _, r in s_.iterrows():
                rows.append(dict(outcome=oname, **r.to_dict()))
        A(md(pd.DataFrame(rows), "{:.2f}"))
        first = int((cl.transition_type == "first").sum())
        A(f"\n{len(cl)} moves over {cl.origin.nunique()} origins, "
          f"{cl.player_id.nunique()} players, {first} of them entering Super League for "
          f"the first time.\n")

        def verdict(b, what):
            if b is None:
                return
            clear = b[1] > 0 or b[2] < 0
            A(f"\n**{what}** {b[0]:+.2f} points [{b[1]:+.2f}, {b[2]:+.2f}]"
              + (", clear of zero.\n" if clear else ", which contains zero.\n"))
        verdict(boot(cl, "class_source", "model"),
                "Against carrying his Australian rating across unchanged:")
        verdict(boot(cl, "flat50", "model"),
                "Against assuming every arrival is average:")
        verdict(boot(cl, "line_direction", "model"),
                "Against a two-parameter straight line for the same direction:")
        A("\nRead together, and this is the sentence to hand the client. The correction "
          "is large and certain — carrying an Australian number into Super League "
          "unchanged is the worst thing that can be done with it. The model also beats "
          "assuming every arrival is average, which is more than could be shown on the "
          "feeder-to-NRL pathway this section used to report. What it still cannot show "
          "is that the conditional apparatus beats a straight line, so the honest claim "
          "is a calibrated correction rather than a recruit ranking.\n")
    else:
        A(f"Too few — {len(cl)} — to say anything.\n")

    A("\n\n## The other pathway: feeder to NRL\n")
    A("Not the client's direction. Kept because it is where the arrival model has its "
      "data, and because an NRL club would ask it.\n")
    fn = d[d.source.isin(FEEDERS) & (d.target == "NRL")
           & (d.transition_type == "first")]
    if len(fn) >= 20:
        A(md(score(fn), "{:.2f}"))
        b50 = boot(fn, "flat50", "model")
        bnone = boot(fn, "class_source", "model")
        if b50 and bnone:
            A(f"\n{len(fn)} players. Against a flat 50: {b50[0]:+.2f} "
              f"[{b50[1]:+.2f}, {b50[2]:+.2f}]. Against leaving the rating alone: "
              f"{bnone[0]:+.2f} [{bnone[1]:+.2f}, {bnone[2]:+.2f}].\n")
    else:
        A(f"Too few — {len(fn)}.\n")

    # Written from the numbers. Three things fall out of the table above and all three
    # are uncomfortable, which is what a rolling test is for.
    A("\n\n## What this says\n")
    piv = d.groupby("origin").apply(
        lambda g: pd.Series({k: float((g.class_target - v).abs().mean())
                             for k, v in [("ladder", g.projected),
                                          ("model", g.model),
                                          ("none", g.class_source),
                                          ("flat50", pd.Series(50.0, index=g.index))]}),
        include_groups=False)
    beaten = int((piv.flat50 < piv.ladder).sum())

    # Every claim below answers to its own interval rather than to the sentence that was
    # true when it was written. The cohort changed under these conclusions once already:
    # before `transition_events.py` this frame held 1,140 rows of which most were men
    # shuttling between competitions rather than moving, and the prose asserting the
    # model "wins everywhere" would have survived that correction untouched, because
    # nothing made it answer to a number.
    lad = boot(d, "flat50", "projected")
    mod = boot(d, "flat50", "model")

    A(f"**The ladder is beaten by assuming everyone is average.** In {beaten} of "
      f"{len(piv)} origins, predicting 50 for every player produced less error than "
      f"carrying his rating across with the measured shift.")
    if lad:
        clear = lad[2] < 0 or lad[1] > 0
        A(f"Across all {len(d)} entries the ladder costs {-lad[0]:+.2f} points against "
          f"a flat 50, player-clustered 95% interval [{-lad[2]:+.2f}, {-lad[1]:+.2f}]"
          + (" — clear of zero.\n" if clear else
             " — which contains zero, so the direction is suggestive and not "
             "established.\n"))
    A("The cause is spread. The ladder inherits the full range of the source rating "
      "when the range that can actually be predicted in the target competition is much "
      "narrower. It answers 'what does this level become', which is a real quantity, "
      "but as a forecast of one man it does not shrink and it should.\n")

    if mod:
        clear = mod[2] < 0 or mod[1] > 0
        A(f"**The conditional model shrinks, and is the better forecast — by a little.** "
          f"It beats a flat 50 by {mod[0]:.2f} points [{mod[1]:+.2f}, {mod[2]:+.2f}]"
          + (", an interval clear of zero.\n" if clear
             else ", an interval containing zero.\n"))
        A(f"That margin is worth stating plainly rather than dressing up. On a scale "
          f"whose standard deviation is about 26, beating a constant by {mod[0]:.2f} "
          f"points is a small edge — and the model's own error is still "
          f"{float((d.class_target - d.model).abs().mean()):.1f}. The model is the "
          f"right number to show because it is the only one of the four not beaten by "
          f"a constant, not because it is accurate.\n")

    for label in ("first", "returning"):
        sub = d[d.transition_type == label]
        if len(sub) < 20:
            continue
        b = boot(sub, "class_source", "projected")
        if b is None:
            continue
        clear = b[2] < 0 or b[1] > 0
        if label == "first":
            A(f"**For a player with no record in the competition, the translation still "
              f"adds nothing.** Over {len(sub)} such entries the ladder saves "
              f"{b[0]:+.2f} points against leaving the rating untouched, 95% interval "
              f"[{b[1]:+.2f}, {b[2]:+.2f}]. This is the group Leeds asks about, and it "
              f"is the group where the headline number earns least.\n")
        else:
            A(f"**For a player returning to a competition, it does.** Over {len(sub)} "
              f"such entries the ladder saves {b[0]:+.2f} points against leaving the "
              f"rating untouched, 95% interval [{b[1]:+.2f}, {b[2]:+.2f}]"
              + (" — clear of zero.\n" if clear else " — containing zero.\n"))
            if clear:
                A("This one is new, and it only became visible once the cohort stopped "
                  "being dominated by men who had not moved at all. A returning player "
                  "left at a known level and comes back to a competition whose standing "
                  "relative to his last one is exactly what the ladder measures. A "
                  "first-timer has no such anchor.\n")

    A("\n\n## By what the player already was\n")
    A("The distinction the review insisted on. A translation has real work to do only "
      "for a player with no record in the competition he is moving to; for anyone else "
      "his own record there is the better evidence.\n")
    for label in te.TYPES:
        sub = d[d.cohort == label]
        if len(sub) < 5:
            continue
        A(f"\n**{LONG[label].capitalize()}** — {len(sub)} moves, "
          f"{sub.player_id.nunique()} players\n")
        A(md(score(sub)))
        b = boot(sub, "class_source", "projected")
        if b:
            m, lo, hi = b
            A(f"\nTranslation against leaving the rating alone: **{m:+.2f}** points of "
              f"error saved [95% CI {lo:+.2f}, {hi:+.2f}], "
              f"{'significant' if lo > 0 else 'not significant'}. Bootstrapped over "
              f"players.\n")

    A("\n\n## Into the NRL specifically\n")
    A("The Leeds question in its own right — a feeder player moving up.\n")
    up = d[(d.target == "NRL") & (d.source.isin(["NSW", "QLD"]))]
    if len(up):
        rows = []
        for label in te.TYPES:
            sub = up[up.cohort == label]
            if len(sub) < 5:
                continue
            s = score(sub)
            for _, r in s.iterrows():
                rows.append(dict(cohort=label, n_players=sub.player_id.nunique(),
                                 **r.to_dict()))
        if rows:
            A(md(pd.DataFrame(rows), "{:.2f}"))

    A("\n\n## The direction-of-move hypothesis\n")
    A("An earlier report claimed the ladder under-corrects for a promoted player by a "
      "specific number of points and proposed fitting that correction. The claim was "
      "measured on a mixed cohort and is withdrawn. What can be said here, on a "
      "properly rolling basis, is the bias of each cohort — the average of actual minus "
      "projected, so a positive figure means the projection was too low.\n")
    rows = []
    for label in te.TYPES:
        sub = d[d.cohort == label]
        if len(sub) < 5:
            continue
        e = sub.class_target - sub.projected
        rng = np.random.default_rng(1)
        s = np.array([rng.choice(e.values, e.size, replace=True).mean()
                      for _ in range(4000)])
        rows.append(dict(cohort=label, moves=len(sub), bias=float(e.mean()),
                         lo=float(np.percentile(s, 2.5)),
                         hi=float(np.percentile(s, 97.5))))
    A(md(pd.DataFrame(rows), "{:.2f}"))
    A("\nA bias whose interval excludes zero is a real, repeatable offset and worth "
      "modelling. One that straddles zero is not, however tempting the point estimate.\n")
    A("\n**The hypothesis does not survive.** Every cohort's interval contains zero, so "
      "there is no repeatable direction-of-move offset to fit. The earlier figure of "
      "3.8 points came from a single season, on a cohort that was mostly established "
      "players, and measured with a model that had already seen the season it was "
      "scored on. Three of those four problems are fixed here and the effect "
      "disappears. It should not be modelled, and the app should not carry a correction "
      "for it.\n")

    A("\n## Limitations\n")
    A("- The number of origins is small. Each one needs a full season of moves to "
      "predict and enough prior moves to fit on, which the data supports only a few "
      "times over.")
    A("- Super League directions rest on few moves at every origin, so their share of "
      "these figures is thin. The ladder borrows strength across directions through the "
      "common scale, which is an assumption the transitivity check supports but does "
      "not prove.")
    A("- The target is a season-only rating and therefore noisier than a cumulative "
      "one, which puts a floor under every error here. An earlier version of this line "
      "went on to claim the noise 'does not favour any predictor'. That was wrong, and "
      "the section above measures how wrong: the published target is shrunk toward 50, "
      "which is exactly what one of the baselines predicts, and the ladder's defeat "
      "holds against it while vanishing against the unshrunk target. Independent noise "
      "would indeed be even-handed; a systematic pull toward one predictor's answer is "
      "not noise.")
    A("- Post-contact metres are absent before 2025, so the composites behind the "
      "earlier origins rest on a slightly different stat set than the later ones.")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(W) + "\n")
    print(f"\nwrote {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
