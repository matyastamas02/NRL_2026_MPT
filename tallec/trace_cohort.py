# -*- coding: utf-8 -*-
"""Follow a player from a feeder competition into the NRL, and grade the projection.

This is the question Mike put in one sentence: *"This guy was a 66 Class player in NSW
Cup in 2022; based on historical transitions we projected that profile as XX in the NRL;
by 2025 he was actually XX."* The machinery for it exists — a rating, a translation
ladder, a held-out season — and this puts the three together for a named player and for
the whole cohort at once.

**This is a post-hoc evaluation, not a holdout, and the difference matters.** The ratings
and the translation model are fitted through 2025 and nothing later
(`evaluation.freeze_season`), so no 2026 match contributed to any number in the
`projected` column. But the *design* of the current model was shaped by what 2026
showed: an external review used these results, and the model was revised in response.
A season you have already looked at cannot test the model you built after looking at it.

The project's one clean out-of-sample result is sealed in `v1_holdout_record.json` —
the model as it stood before any of that, with the hashes of the artefacts that produced
it. Read that for the honest holdout figure. Read this for how the current model behaves,
knowing it has had the advantage. The first genuinely clean confirmation is 2027.

One sign of why the distinction is not pedantic: on first-season players the earlier
model was worse than using the feeder rating untouched, and the current one is better.
That is the kind of reversal a post-hoc evaluation is prone to produce.

  rated       his rating in the feeder competition at the end of 2025 — the number BOSC
              would have shown a recruiter looking at him that summer.
  projected   that rating carried across by the measured competition ladder.
  model       the Ridge fit, which conditions on position, age, minutes and matches
              played, and deliberately regresses toward the middle.
  actual      what he rated in the NRL in 2026, computed from his 2026 matches alone
              and standardized within the 2026 NRL pool.

Three caveats, all of which make the test harder rather than flattering it:

  * The 2026 season is incomplete — twenty rounds of it. A player with five matches in
    it is being measured on a thin sample, which is why the tables are cut by matches
    played.
  * The 2026 NRL positions are estimated, not read from match sheets, and are right
    about four times in five overall but only about half the time for wingers and
    fullbacks (`validate_positions.py`). That noise sits in `actual`, not in the
    projection.
  * Nobody who failed to get an NRL game appears at all. The cohort is players the
    clubs chose to select, so this measures the projection among those given the
    chance, not among everyone BOSC rated highly.
  * **Most of these men are not being promoted.** An external review found that of the
    graded players, the large majority had already played in the NRL and most were still
    NRL players the season before — established men having a spell in reserve grade, not
    prospects making the step up. Every figure is therefore reported split by what the
    player already was (`cohorts.py`), and the aggregate is given last, because it is
    dominated by the group the question was not about.

    python trace_cohort.py                     # writes TRACE_REPORT.md
    python trace_cohort.py --player "Surname"   # one player's story
"""
import argparse
import os
import sqlite3

import numpy as np
import pandas as pd

import cohorts as ch
import player_rating_engine as pre
import predict_translation as pt
import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
OUT = os.path.join(BASE, "TRACE_REPORT.md")
COLS = ["player_id", "player", "season", "round", "team", "position", "minutes",
        "all_run_metres", "p_c_m", "tackle_breaks", "line_breaks", "tackles",
        "offloads", "try_assists", "tries", "errors"]
FEEDERS = ["NSW", "QLD"]
TARGET, TARGET_SEASON = "NRL", 2026
MIN_SOURCE_GAMES = 5


def season_rating(con, comp, season):
    """Rating from one season's matches alone, on the published 0-100 scale."""
    df = pd.read_sql(f"SELECT {', '.join(COLS)} FROM player_match_stats "
                     f"WHERE competition=? AND season=?", con, params=(comp, season))
    if len(df) < 100:
        return pd.DataFrame()
    known = df["position"].notna() & (df["position"] != "Unknown")
    force = None if float(known.mean()) >= pre.MIN_POS_COVERAGE else "competition_relative"
    eng = pre.PlayerRatingEngine(comp, force_mode=force)
    snap = eng.compute_snapshot(pm=eng._composite(df))
    return snap[["player_id", "name", "class_score", "class_z", "n_games"]]


def cohort(con):
    """Feeder players of 2025 who appeared in the NRL in 2026, with all three numbers."""
    src = pd.read_sql(
        "SELECT r.player_id, r.competition source, r.class_score rated, r.n_games, "
        "       r.confidence "
        "FROM player_ratings r WHERE r.season=2025 AND r.competition IN ('NSW','QLD')",
        con)
    games25 = pd.read_sql(
        "SELECT player_id, competition, COUNT(*) g FROM player_match_stats "
        "WHERE season=2025 AND competition IN ('NSW','QLD') GROUP BY 1,2", con)
    src = src.merge(games25, left_on=["player_id", "source"],
                    right_on=["player_id", "competition"])
    src = src[src.g >= MIN_SOURCE_GAMES]

    pos = pd.read_sql(
        "SELECT player_id, position FROM player_match_stats "
        "WHERE competition IN ('NSW','QLD') AND season<=2025 "
        "AND position IS NOT NULL AND position<>'Unknown'", con)
    career = pos.groupby("player_id")["position"].agg(sp.primary_position)
    src["grp"] = src.player_id.map(career).map(sp.POSITION_GROUP)

    # A man who played in both feeder competitions appears once per competition; left
    # alone he is counted twice and weighted twice in every average. Keep the
    # better-evidenced side.
    src = ch.one_row_per_player(src, prefer="g")

    tgt = season_rating(con, TARGET, TARGET_SEASON)
    d = src.merge(tgt.rename(columns={"class_score": "actual", "n_games": "nrl_games"}),
                  on="player_id")
    # what he already was in the competition he is moving into, before this season
    d = d.merge(ch.classify(con, TARGET, TARGET_SEASON)[["player_id", "cohort"]],
                on="player_id", how="left")

    rows = []
    for _, r in d.iterrows():
        try:
            tr = pt.translate(float(r.rated), r.source, TARGET,
                              position_group=r.grp, games=int(r.g))
        except Exception:
            continue
        rows.append(dict(player=r["name"], source=r.source, feeder_games=int(r.g),
                         grp=r.grp, cohort=r.cohort, rated=r.rated,
                         projected=tr["score_ladder"], model=tr["score_model"],
                         actual=r.actual, nrl_games=int(r.nrl_games),
                         confidence=r.confidence))
    out = pd.DataFrame(rows)
    out["error"] = out.actual - out.projected
    return out.sort_values("projected", ascending=False)


def md(df, fmt="{:.2f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, rule] + body)


def accuracy(d):
    rows = []
    for label, p in [("the projection (ladder)", d.projected),
                     ("the Ridge model", d.model),
                     ("no translation (his feeder rating)", d.rated),
                     ("no information (competition average, 50)",
                      pd.Series(50.0, index=d.index))]:
        ok = p.notna()
        e = d.actual[ok] - p[ok]
        rows.append(dict(predictor=label, n=int(ok.sum()), mae=float(e.abs().mean()),
                         bias=float(e.mean()),
                         r=float(np.corrcoef(p[ok], d.actual[ok])[0, 1])
                         if p[ok].std() > 0 else np.nan))
    return pd.DataFrame(rows)


def sentence(r):
    return (f"**{r.player}** — a {r.rated:.0f} in the {r.source} Cup in 2025 over "
            f"{r.feeder_games} matches. The ladder projected that profile as "
            f"{r.projected:.0f} in the NRL"
            + (f", the model as {r.model:.0f}" if pd.notna(r.model) else "")
            + f". Across {r.nrl_games} NRL matches in 2026 he has rated "
              f"{r.actual:.0f}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default=None)
    ap.add_argument("--min-games", type=int, default=5,
                    help="minimum NRL 2026 matches to be graded")
    a = ap.parse_args()

    con = sqlite3.connect(DB)
    d = cohort(con)
    con.close()
    if d.empty:
        print("no cohort")
        return

    if a.player:
        hit = d[d.player.str.contains(a.player, case=False, na=False)]
        if hit.empty:
            print(f"no player matching {a.player!r} in the cohort")
        for _, r in hit.iterrows():
            print(sentence(r).replace("**", ""))
        return

    g = d[d.nrl_games >= a.min_games].copy()
    W, A = [], None
    A = W.append
    A("# From the feeder competitions into the NRL\n")
    A("> **Post-hoc evaluation, not a holdout.** Nothing here is fitted on 2026 — the "
      "models stop at 2025 — but the current model was *designed* after an external "
      "review had already seen these results, and a season you have looked at cannot "
      "test the model you built afterwards. The project's one clean out-of-sample "
      "figure is sealed in `v1_holdout_record.json`, with the hashes of the artefacts "
      "that produced it. The first genuinely clean confirmation is 2027.\n")
    A(f"Every player below was rated in the NSW Cup or the Queensland Cup in 2025 over "
      f"at least {MIN_SOURCE_GAMES} matches, and has since played in the NRL in 2026. "
      f"No 2026 match contributed to any figure in the `projected` column — the freeze "
      f"at 2025 sees to that — so it is a forecast in the mechanical sense. What it is "
      f"not is a test of a model that has had sight of the answer.\n")
    A(f"{len(d)} players in the cohort, {len(g)} of them with at least {a.min_games} "
      f"NRL matches in 2026 — the 2026 season is twenty rounds old, so a thin NRL "
      f"sample is the main source of noise in `actual` and the tables below are cut by "
      f"it.\n")

    A("\n## Who these players actually are\n")
    A("This is not a group of prospects making the step up, and reading it as one is "
      "how an earlier version of this report reached a conclusion it could not "
      "support. Split by what each man already was:\n")
    A(md(ch.summarise(g), "{:.0f}"))

    A("\n\n## How good was the projection\n")
    A(f"Graded on the {len(g)} players with {a.min_games}+ NRL matches in 2026. `bias` "
      f"is the average of actual minus predicted, so a positive number means the "
      f"projection was too low. Errors are in points on the 0-100 rating scale.\n")
    A("`no translation` is the baseline a club would otherwise use — take his feeder "
      "rating at face value. `no information` assumes every selected player is average. "
      "A translation earns its place only by beating the first of those.\n")

    for label in ch.LABELS:
        sub = g[g.cohort == label]
        if sub.empty:
            continue
        A(f"\n**{ch.LONG[label].capitalize()}** — {len(sub)} players\n")
        A(md(accuracy(sub)))
        if len(sub) < 20:
            A(f"\n*{len(sub)} players. Indicative only — one outlier moves these "
              f"figures by more than the differences between them.*\n")

    A("\n\n**All of them together**, dominated by the third group and so read last\n")
    A(md(accuracy(g)))
    # Stated from the numbers rather than asserted, because the answer has already
    # flipped once between model versions and a hardcoded sentence would have gone on
    # claiming the old one.
    fs = g[g.cohort == "first"]
    if len(fs):
        lad = float((fs.actual - fs.projected).abs().mean())
        raw = float((fs.actual - fs.rated).abs().mean())
        verdict = ("beats" if lad < raw else "does not beat")
        A(f"\n**What the split shows.** The group Leeds actually asks about — men with "
          f"no top-grade record, where the feeder rating is the only evidence there is "
          f"— is the smallest one here, {len(fs)} players. On them the translation "
          f"{verdict} taking the rating at face value: {lad:.2f} against {raw:.2f} "
          f"points of error. For players who already have an NRL record the comparison "
          f"is less interesting either way, because a recruiter would use that record "
          f"rather than a translation out of reserve grade.\n")
        A(f"\n{len(fs)} players is not a verdict. It is the right question, and the "
          f"rolling-origin backtest is where it gets a sample large enough to answer.\n")
    A("\nAn earlier version of this report turned these biases into a recommended "
      "correction of a specific size. That is withdrawn: it was measured on the mixed "
      "cohort, and fitting anything to 2026 would consume the only clean holdout the "
      "project has. The correction is a hypothesis for the historical backtest to "
      "test, not a result.\n")

    A("\n### By how much of him we had seen\n")
    rows = []
    for lo, hi, lab in [(a.min_games, 9, f"{a.min_games}-9"), (10, 14, "10-14"),
                        (15, 99, "15+")]:
        s = g[(g.nrl_games >= lo) & (g.nrl_games <= hi)]
        if len(s) < 5:
            continue
        rows.append(dict(nrl_matches=lab, n=len(s),
                         mae=float((s.actual - s.projected).abs().mean()),
                         bias=float((s.actual - s.projected).mean()),
                         r=float(np.corrcoef(s.projected, s.actual)[0, 1])))
    if rows:
        A(md(pd.DataFrame(rows)))
        A("\nA player with a handful of NRL matches has a noisy `actual`, so the error "
          "should fall as the sample grows. If it does not, the projection is wrong "
          "rather than the measurement being noisy.\n")

    A("\n## Does it pick the right men?\n")
    A("Exact points matter less to a recruiter than order: if the list is sorted by "
      "projection, do the ones near the top turn out better than the ones near the "
      "bottom? Spearman rank correlation between projection and outcome, and the "
      "average outcome by projected quartile.\n")
    if len(g) >= 20:
        rho = g[["projected", "actual"]].corr(method="spearman").iloc[0, 1]
        A(f"\nSpearman rank correlation: **{rho:+.2f}** over {len(g)} players.\n")
        q = g.assign(quartile=pd.qcut(g.projected, 4,
                                      labels=["lowest", "3rd", "2nd", "highest"]))
        qt = (q.groupby("quartile", observed=True)
               .agg(n=("actual", "size"), mean_projected=("projected", "mean"),
                    mean_actual=("actual", "mean")).reset_index())
        A(md(qt))

    A("\n## The stories\n")
    A("Ten men the projection rated highest, in order, with what has happened since.\n")
    for _, r in g.nlargest(10, "projected").iterrows():
        A(f"- {sentence(r)}")
    A("\nAnd where it was furthest wrong in each direction.\n")
    A("\n**Under-projected**\n")
    for _, r in g.nlargest(4, "error").iterrows():
        A(f"- {sentence(r)}")
    A("\n**Over-projected**\n")
    for _, r in g.nsmallest(4, "error").iterrows():
        A(f"- {sentence(r)}")

    A("\n\n## The full cohort\n")
    A(md(g[["player", "source", "grp", "feeder_games", "rated", "projected", "model",
            "nrl_games", "actual", "error"]].sort_values("projected", ascending=False),
         "{:.1f}"))

    A("\n\n## What this does not establish\n")
    A("- **The 2026 season is not finished.** Twenty rounds. A rating from five matches "
      "moves a long way with a sixth; treat the individual lines as illustrative and "
      "the aggregate as the result.")
    A("- **Only the promoted appear.** Clubs chose these men. A player BOSC rated at 70 "
      "who never got a game is invisible here, so this measures the projection among "
      "those given the chance, not its accuracy over everyone.")
    A("- **The 2026 positions are estimated**, so the pool a player is standardized "
      "against in `actual` is itself uncertain — most for wingers and fullbacks. This "
      "resolves when the match-sheet positions arrive with the completed season.")
    A("- **The feeder rating is cumulative** and the NRL rating is from one season, "
      "which is the right pairing for the question but means the two sides carry "
      "different amounts of evidence.")
    A("- Nothing here is fitted. If it were re-run after adding 2026 to the fit, it "
      "would stop being a test.")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(W) + "\n")
    print(f"cohort {len(d)}, graded {len(g)} | wrote {OUT}")


if __name__ == "__main__":
    main()
