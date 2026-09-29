# -*- coding: utf-8 -*-
"""Historical validation of the player ratings, written to be read by a stranger.

The question is not whether the ratings look sensible. It is whether a rating computed
at the end of one season tells you anything about the next one, how much of it survives
a year, how many matches it takes before it means anything, whether it works as well for
a winger as for a prop, and where it is wrong.

Method, stated so it can be attacked:

  * Ratings are rebuilt WALK-FORWARD. For every season S the snapshot uses matches from
    S and earlier and nothing later, so a rating being tested against season S+1 was
    computed without any knowledge of it. This is not the shipped snapshot re-read from
    the database — that one uses the whole history and would grade its own homework.

  * The thing being predicted is the player's mean composite in S+1, standardized
    within the S+1 pool. Same units as the rating, so the two are directly comparable.

  * Two baselines, because a correlation on its own means little:
        "no information"  predict the competition average for everyone
        "last season raw" predict his unshrunk mean from season S alone
    The rating has to beat both, and beating the second is the harder test: it is what
    shows the shrinkage and the multi-season Class are earning their place.

  * Nothing after `evaluation.freeze_season` is touched, so the held-out season stays
    held out.

What this does NOT establish. Both the rating and the target are built from the same
statistics, so this measures whether performance persists as the rating says it should
— internal predictive validity. Whether the rating helps predict MATCH RESULTS is a
different question, answered separately by gigot_v2.py and teamlist_backtest.py.
A correlation here is also attenuated: the target is a noisy measure of a player's true
level, so the ceiling is well below 1.0 and the numbers should be read against the
baselines rather than against perfection.

    python validate_ratings.py            # writes VALIDATION_REPORT.md
    python validate_ratings.py --print    # also dumps the tables to the terminal
"""
import argparse
import os
import sqlite3
import warnings

import numpy as np
import pandas as pd

import player_rating_engine as pre
import sp_schema as sp

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
OUT = os.path.join(BASE, "VALIDATION_REPORT.md")
COMPS = ["NRL", "SL", "NSW", "QLD"]
COLS = ["player_id", "player", "season", "round", "team", "position", "minutes",
        "all_run_metres", "p_c_m", "tackle_breaks", "line_breaks", "tackles",
        "offloads", "try_assists", "tries", "errors"]
MIN_GAMES_TARGET = 5      # a next season worth grading


def load(con, comp):
    q = (f"SELECT {', '.join(COLS)} FROM player_match_stats WHERE competition=?")
    p = [comp]
    if pre.FREEZE_SEASON is not None:
        q += " AND season<=?"
        p.append(pre.FREEZE_SEASON)
    return pd.read_sql(q, con, params=p)


def season_composites(hist, comp):
    """Composites season by season, each standardized in its own season's pool."""
    known = hist["position"].notna() & (hist["position"] != "Unknown")
    force = None if float(known.mean()) >= pre.MIN_POS_COVERAGE else "competition_relative"
    parts = []
    for s in sorted(hist.season.dropna().unique()):
        part = hist[hist.season == s]
        if len(part) < 100:
            continue
        parts.append(pre.PlayerRatingEngine(comp, force_mode=force)._composite(part))
    return pd.concat(parts, ignore_index=True), force


def walk_forward(pm, comp, force):
    """The rating each player held at the end of each season, from that season back.

    Restricted to the men who actually played that season, which is what the app
    publishes. It used to keep everyone who had ever appeared, so a player who retired in
    2022 still carried a 2025 rating built from nothing new — and a rating that cannot
    move because no matches were added is perfectly stable. The external review of
    2026-09-22 was right that this flattered the operational stability figure, which is
    the one quoted as "how much does the number a recruiter sees move from year to year".
    """
    out = []
    seasons = sorted(pm.season.dropna().unique())
    for s in seasons:
        upto = pm[pm.season <= s]
        eng = pre.PlayerRatingEngine(comp, force_mode=force)
        snap = eng.compute_snapshot(pm=upto)
        active = set(upto.loc[(upto.season == s) & upto["ratable"], "player_id"])
        snap = eng.calibrate(snap[snap.player_id.isin(active)].copy())
        snap["as_of"] = int(s)
        # matches played in this season alone, which is what a reader means by
        # "how much did we see of him this year"
        g = upto[(upto.season == s) & upto["ratable"]].groupby("player_id").size()
        snap["games_this_season"] = snap.player_id.map(g).fillna(0).astype(int)
        raw_s = (upto[(upto.season == s) & upto["ratable"]]
                 .groupby("player_id")["composite"].mean())
        snap["raw_this_season"] = snap.player_id.map(raw_s)
        out.append(snap)
    return pd.concat(out, ignore_index=True)


def outcomes(pm):
    """What each player actually did in each season — the thing being predicted."""
    r = pm[pm["ratable"]]
    return (r.groupby(["player_id", "season"])
             .agg(actual=("composite", "mean"), games=("composite", "size"))
             .reset_index())


def season_only(pm, comp, force):
    """Each player's rating from ONE season's matches, on the published 0-100 scale.

    Needed for the cross-competition section. A player's ordinary rating is cumulative,
    so an established NRL player who spends a year in the NSW Cup and returns still
    carries his whole NRL past — comparing that with a projection of the move measures
    his career, not the move. What the move has to be judged against is what he did in
    the target competition in the season after it, and nothing else.
    """
    out = []
    for s in sorted(pm.season.dropna().unique()):
        part = pm[pm.season == s]
        if part["ratable"].sum() < 50:
            continue
        eng = pre.PlayerRatingEngine(comp, force_mode=force)
        snap = eng.compute_snapshot(pm=part)
        snap["season"] = int(s)
        snap["comp"] = comp
        out.append(snap[["player_id", "name", "season", "comp", "class_score",
                         "class_z", "n_games"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def pairs_for(comp, con):
    hist = load(con, comp)
    if hist.empty:
        return None
    pm, force = season_composites(hist, comp)
    wf = walk_forward(pm, comp, force)
    act = outcomes(pm)
    act["prev"] = act.season - 1
    d = wf.merge(act, left_on=["player_id", "as_of"], right_on=["player_id", "prev"],
                 suffixes=("", "_next"))
    d = d[d.games >= MIN_GAMES_TARGET].copy()
    d["comp"] = comp
    # career position group, for the positional split
    posmap = (hist[hist.position.notna() & (hist.position != "Unknown")]
              .groupby("player_id")["position"].agg(sp.primary_position))
    d["grp"] = d.player_id.map(posmap).map(sp.POSITION_GROUP)
    return d, season_only(pm, comp, force), posmap


def scoreboard(d):
    """How well each predictor does against what actually happened.

    Every predictor is scored on the SAME rows. An earlier version claimed this and did
    not do it: the unshrunk baseline is missing wherever a player had no matches in the
    season it is built from, so it ran on 1,706 NRL pairs against the rating's 1,752 and
    the two columns were not comparable. The comparison is now restricted to the rows
    where all three can be computed, and the count of rows dropped is reported, because
    a baseline that is unavailable on the hard cases would otherwise look strong by
    being absent from them.
    """
    preds = {"no information (competition average)": pd.Series(0.0, index=d.index),
             "last season, unshrunk": d.raw_this_season,
             "the rating (shrunk Class)": d.class_z}
    ok = d.actual.notna()
    for p in preds.values():
        ok &= p.notna()
    rows = []
    for label, pred in preds.items():
        e = d.actual[ok] - pred[ok]
        rows.append(dict(predictor=label, n=int(ok.sum()),
                         r=float(np.corrcoef(pred[ok], d.actual[ok])[0, 1])
                         if pred[ok].std() > 0 else np.nan,
                         mae=float(e.abs().mean()), rmse=float((e ** 2).mean() ** .5)))
    base = rows[0]["rmse"]
    for r in rows:
        r["vs_naive"] = (base - r["rmse"]) / base
        r["dropped"] = int((~ok).sum())
    return pd.DataFrame(rows)


def boot_ci(values, n=4000, seed=0):
    """Bootstrap interval for a mean, resampling the players themselves."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 8:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    s = np.array([rng.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def stability(frames, per_season):
    """Two different questions that an earlier version ran together.

    `independent` correlates a rating built from season S alone with one built from S+1
    alone. The two share no match, so the correlation is a genuine measure of how much
    of a player's level persists.

    `operational` correlates the cumulative ratings the app actually displays. It is
    much higher — around 0.89 — but partly by construction, because the rating at the
    end of S+1 contains every match behind the rating at the end of S. It answers "how
    much does the number a recruiter sees move", which is worth knowing and is not the
    same claim.
    """
    rows = []
    for comp, d in frames.items():
        s = per_season[per_season.comp == comp][["player_id", "season", "class_z",
                                                 "class_score", "n_games"]]
        s = s[s.n_games >= MIN_GAMES_TARGET]
        ind = s.merge(s.assign(season=s.season - 1), on=["player_id", "season"],
                      suffixes=("", "_next"))
        w = d[["player_id", "as_of", "class_z", "class_score"]]
        cum = w.merge(w.assign(as_of=w.as_of - 1), on=["player_id", "as_of"],
                      suffixes=("", "_next"))
        row = dict(competition=comp)
        for name, j in (("independent", ind), ("operational", cum)):
            if len(j) < 20:
                row[f"{name}_n"] = len(j)
                continue
            move = (j.class_score_next - j.class_score).abs()
            lo, hi = boot_ci(move)
            row[f"{name}_n"] = len(j)
            row[f"{name}_r"] = float(np.corrcoef(j.class_z, j.class_z_next)[0, 1])
            row[f"{name}_move"] = float(move.mean())
            row[f"{name}_ci"] = f"[{lo:.1f}, {hi:.1f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(df, floatfmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, rule] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", dest="show")
    a = ap.parse_args()

    con = sqlite3.connect(DB)
    res = {c: pairs_for(c, con) for c in COMPS}
    res = {k: v for k, v in res.items() if v is not None and len(v[0])}
    frames = {k: v[0] for k, v in res.items()}
    per_season = pd.concat([v[1] for v in res.values() if len(v[1])],
                           ignore_index=True)
    posmap = pd.concat([v[2] for v in res.values()])
    posmap = posmap[~posmap.index.duplicated()]
    allp = pd.concat(frames.values(), ignore_index=True)
    print(f"season-to-season pairs: {len(allp)} "
          f"({', '.join(f'{k} {len(v)}' for k, v in frames.items())})")

    md = []
    W = md.append
    W("# Player ratings — historical validation\n")
    W(f"Generated by `validate_ratings.py`. Everything below is computed walk-forward: "
      f"a rating tested against season S+1 was built from season S and earlier only. "
      f"Nothing after {pre.FREEZE_SEASON} is used, so the held-out season stays held "
      f"out.\n")
    W("**What is measured.** For each player and each season S, the rating he held at "
      "the end of S is compared with what he actually did in S+1 — his mean "
      "performance composite, standardized within the S+1 pool, so predictor and "
      "target share a scale. A season counts as gradable at "
      f"{MIN_GAMES_TARGET} matches or more.\n")
    W("**What is not measured.** The rating and the target are built from the same "
      "statistics, so this establishes that performance persists as the rating claims "
      "— internal predictive validity. Whether the ratings help predict match results "
      "is a separate question, answered by `gigot_v2.py` and `teamlist_backtest.py`. "
      "Both sides are also noisy measures of a player's true level, so correlations "
      "here are attenuated and should be read against the baselines, not against 1.0.\n")

    W("\n## 1. Predictive validity\n")
    W("Three predictors of next season, scored on the same rows. Lower error is better; "
      "`vs_naive` is the reduction in RMSE against knowing nothing. `dropped` counts "
      "the pairs excluded because the unshrunk baseline cannot be computed for them — "
      "a player with no matches in the season it is built from. An earlier version left "
      "those pairs in for the other two predictors, so the columns were not comparable "
      "and the baseline was flattered by being absent from the cases it would have "
      "found hardest.\n")
    for comp, d in frames.items():
        W(f"\n**{comp}** — {len(d)} player-season pairs, "
          f"{d.as_of.min()}→{d.as_of.max() + 1}\n")
        sb = scoreboard(d)
        W(md_table(sb))
        if a.show:
            print(f"\n{comp}\n", sb.to_string(index=False))
    W(f"\n**All competitions pooled** — {len(allp)} pairs\n")
    W(md_table(scoreboard(allp)))

    W("\n\n## 2. Stability\n")
    W("Two different questions, which an earlier version of this report ran together "
      "and reported as one.\n")
    W("**Independent** correlates a rating built from season S alone with one built "
      "from S+1 alone. The two share no match, so it measures how much of a player's "
      "level genuinely persists. **Operational** correlates the cumulative ratings the "
      "app displays. It is much higher, but partly by construction: the later rating "
      "contains every match behind the earlier one. It answers \"how much does the "
      "number a recruiter sees move from year to year\" — worth knowing, and a "
      "different claim. Movement is the mean absolute change on the 0-100 scale, with a "
      "95% interval bootstrapped over players.\n")
    W(md_table(stability(frames, per_season)))
    W("\nThe gap between the two is the answer to the review's objection. Where "
      "`operational_r` sits far above `independent_r`, the steadiness of the published "
      "rating is mostly the shared history inside it rather than the player being that "
      "consistent.\n")

    W("\n\n## 3. Sample-size effects\n")
    W("Validity by how many matches the rating was built on. If the shrinkage is doing "
      "its job, players with few matches should be pulled toward the average and "
      "should not be confidently wrong.\n")
    bins = [0, 5, 10, 20, 40, 10_000]
    labels = ["1-5", "6-10", "11-20", "21-40", "40+"]
    allp["bucket"] = pd.cut(allp.n_games, bins=bins, labels=labels, right=True)
    rows = []
    for b, g in allp.groupby("bucket", observed=True):
        if len(g) < 20:
            continue
        rows.append(dict(matches=b, n=len(g),
                         r=float(np.corrcoef(g.class_z, g.actual)[0, 1]),
                         mae=float((g.actual - g.class_z).abs().mean()),
                         mean_shrinkage=float(g.shrinkage_B.mean()),
                         rating_spread=float(g.class_score.std())))
    W(md_table(pd.DataFrame(rows)))
    W("\n`mean_shrinkage` runs 0 (the rating is entirely the prior) to 1 (entirely his "
      "own record). `rating_spread` is the standard deviation of the published 0-100 "
      "score in the bucket: a thin record should produce a narrow spread, which is the "
      "shrinkage refusing to make claims it cannot support.\n")

    W("\n## 4. Positional effects\n")
    W("The same validity measure split by the player's career position group.\n")
    rows = []
    for grp, g in allp.dropna(subset=["grp"]).groupby("grp"):
        if len(g) < 30:
            continue
        rows.append(dict(position=grp, n=len(g),
                         r=float(np.corrcoef(g.class_z, g.actual)[0, 1]),
                         mae=float((g.actual - g.class_z).abs().mean()),
                         naive_mae=float(g.actual.abs().mean())))
    pr = pd.DataFrame(rows).sort_values("r", ascending=False)
    pr["beats_naive"] = (pr.naive_mae - pr.mae) / pr.naive_mae
    W(md_table(pr))

    W("\n\n## 5. Hits and misses\n")
    W("The pairs where the rating was most and least right, as a check that the "
      "failures are the kind you would expect rather than a systematic fault. "
      "`rating` is the 0-100 score going into the season; `actual` is what he did, on "
      "the composite scale where 0 is the competition average.\n")
    allp["residual"] = allp.actual - allp.class_z
    show = ["name", "comp", "as_of", "grp", "n_games", "class_score", "actual", "residual"]
    W("\n**Most under-rated** — did far better than the rating implied\n")
    W(md_table(allp.nlargest(8, "residual")[show].rename(
        columns={"as_of": "rated_after", "class_score": "rating"}), "{:.2f}"))
    W("\n**Most over-rated** — fell furthest short\n")
    W(md_table(allp.nsmallest(8, "residual")[show].rename(
        columns={"as_of": "rated_after", "class_score": "rating"}), "{:.2f}"))

    W("\n\n## 6. Across competitions — moved to its own report\n")
    W("This section used to apply the *current* translation model to historical moves "
      "that were part of its own training set, and reported the result as validation. "
      "It was not: a model asked about data it has already learned will do well, and "
      "the directional effect it appeared to find did not survive a proper test. It is "
      "removed rather than patched.\n")
    W("The replacement is `ROLLING_REPORT.md`, from `rolling_backtest.py`, which "
      "forecasts each season using only what was known before it and reports the "
      "result separately for players entering a competition for the first time, "
      "returning to it, and continuing in it. Its findings are materially different "
      "from what this section used to claim — on a rolling basis the ladder does not "
      "beat assuming every player is average, and for a player with no record in the "
      "competition he is moving to it saves nothing at all.\n")

    W("\n\n## Limitations\n")
    W("- Both predictor and target are noisy measures of the same underlying quantity, "
      "so the correlations are attenuated and the achievable ceiling is well under 1.0.")
    W("- A player only appears when he played enough in consecutive seasons, so the "
      "sample leans toward established players; whoever fell out of the competition "
      "is absent. That flatters stability, and section 2 now measures how much: the "
      "published rating's year-to-year correlation runs 0.77-0.91, while ratings "
      "built from single seasons that share no match correlate at 0.56-0.67. The "
      "difference is shared history, not consistency.")
    W("- Position groups come from the career mode of a player's starting positions. "
      "Where the position itself is estimated rather than read off a match sheet it is "
      "right about four times in five, and only about half the time for wingers and "
      "fullbacks (`validate_positions.py`), so the positional split is softer than it "
      "looks for the outside backs.")
    W("- Cross-competition results are no longer here at all. They were measured with "
      "the model applied to its own training data; `ROLLING_REPORT.md` replaces them "
      "with a rolling-origin test whose conclusions are different.")
    W(f"- Everything is fitted through {pre.FREEZE_SEASON}. What these projections say "
      f"about a later season can only be tested on a season nobody has looked at, which "
      f"2026 no longer is: the models were revised after a review had seen it. The one "
      f"sealed out-of-sample figure is in `v1_holdout_record.json`, and the next clean "
      f"one is 2027.")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    con.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
