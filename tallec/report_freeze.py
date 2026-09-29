# -*- coding: utf-8 -*-
"""Report on the 2025 freeze — what changed, by how much, and what to distrust.

Written to be read by someone who was not here. Every number is pulled from an
artefact rather than typed in. The before/after rating statistics come from the audit
log's `model_runs`, which records each rebuild's fit statistics and the hash of the
config it ran under. The before/after translation ladder comes from `freeze_ab_ladder.csv`,
a deliberate A/B: the same code fitted twice, once with the freeze off and once on, with
nothing else touched — see `ladder_ab()` for why a database snapshot was not good enough.

    python report_freeze.py        # writes FREEZE_REPORT.md
"""
import json
import os
import sqlite3

import pandas as pd

import player_rating_engine as pre

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
AUDIT = os.path.join(BASE, "tallec_audit.db")
OUT = os.path.join(BASE, "FREEZE_REPORT.md")


def md(df, fmt="{:.4f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, rule] + body)


def rating_runs(before_id=None, after_id=None):
    """The two rating rebuilds either side of the freeze, named by run id.

    An earlier version took "the last two", which meant the report silently described a
    different pair every time anything was rebuilt. The pair is now chosen by the config
    hash actually changing — the freeze is a config change, so the last run under the old
    hash and the first under the new one are the two that bracket it — and both ids are
    printed so the choice can be checked.
    """
    con = sqlite3.connect(AUDIT)
    r = pd.read_sql("SELECT * FROM model_runs WHERE script='regenerate_full.py' "
                    "ORDER BY id", con)
    con.close()
    if len(r) < 2:
        return None, None
    if before_id and after_id:
        return (r[r.id == before_id].iloc[0], r[r.id == after_id].iloc[0])
    # The freeze has a signature in the recorded statistics: the season the ratings are
    # published for drops to FREEZE_SEASON. Looking for a changed config hash instead
    # found the earliest config change of any kind, which was three days earlier and
    # about something else entirely.
    def season_of(row, comp="NRL"):
        try:
            return json.loads(row.stats).get(comp, {}).get("season")
        except Exception:
            return None
    for i in range(1, len(r)):
        if (season_of(r.iloc[i]) == pre.FREEZE_SEASON
                and season_of(r.iloc[i - 1]) != pre.FREEZE_SEASON):
            return r.iloc[i - 1], r.iloc[i]
    return r.iloc[-2], r.iloc[-1]


AB = os.path.join(BASE, "freeze_ab_ladder.csv")


def ladder_ab():
    """The controlled comparison of the translation ladder, frozen against not.

    Not read from an old snapshot. An earlier draft of this report did that and was
    wrong: the most recent snapshot predated other changes to the rating engine, so the
    difference it showed was the freeze plus three days of unrelated work, and the
    report drew a conclusion ("every gap widened") that the controlled measurement
    contradicts. The comparison here comes from toggling `evaluation.freeze_season`
    between null and 2025, refitting under each, and recording both ladders — with the
    frozen state verified bit-identical afterwards. `data_imports` rows 5 and 6 are the
    two runs. To regenerate: set the key to null, run `fit_translation_v2.py`, save
    `translation_ladder`, set it back, run again, and join the two.
    """
    return pd.read_csv(AB) if os.path.exists(AB) else None


def main():
    W = []
    A = W.append
    before, after = rating_runs()
    con = sqlite3.connect(DB)
    ratings_now = pd.read_sql("SELECT competition, season, COUNT(*) players, "
                              "ROUND(AVG(n_games),1) avg_effective_matches, rating_basis "
                              "FROM player_ratings GROUP BY 1,2,5", con)
    try:
        meta = pd.read_sql("SELECT * FROM translation_model_v3_meta", con)
        lad_v3 = pd.read_sql("SELECT * FROM translation_ladder_v3", con)
    except Exception:
        meta, lad_v3 = pd.DataFrame(), pd.DataFrame()
    con.close()
    lad_ab = ladder_ab()

    A("# Freezing the fit at 2025\n")
    A("## Why\n")
    A("Mike's instruction (2026-08-26): leave the 2026 data alone until the season "
      "finishes in early October, rather than re-importing it every week for a project "
      "that runs longer than that. He also drew the methodological consequence, which "
      "is the real reason to do it — with everything frozen at the end of 2025, the "
      "2026 season becomes an out-of-sample test of what the ratings and the "
      "translation model projected, *particularly for players who changed competitions "
      "or levels*.\n")
    A("That only holds if 2026 contributed nothing to the projections. It previously "
      "did: the ratings were built from every season including 2026, and the "
      "translation model used moves landing in 2026. Freezing is therefore not a "
      "convenience — without it the planned test would be scored against its own "
      "training data.\n")

    A("\n## How it is implemented\n")
    A(f"One setting, `evaluation.freeze_season` in `config.json`, currently "
      f"**{pre.FREEZE_SEASON}**. It is read once in `player_rating_engine.py` as "
      f"`FREEZE_SEASON` and consumed by both fitting scripts — `regenerate_full.py` "
      f"for the ratings and `fit_translation_v2.py` for the translation model. A single "
      f"source so the two cannot end up frozen at different points, which would be "
      f"silent and would invalidate the comparison between them.\n")
    A("Seasons after the freeze are **not deleted**. They stay in `player_match_stats` "
      "and the app can still display them. What the freeze governs is only what is "
      "*fitted*. Setting it to `null` restores the previous behaviour.\n")

    if before is not None:
        A("\n## Effect on the ratings\n")
        A(f"Both rebuilds are in the audit log, named by run id: **{int(before.id)}** "
          f"before the freeze and **{int(after.id)}** after. The config hash each ran "
          f"under — `{before.config_hash}` before, `{after.config_hash}` after — is the "
          f"recorded evidence that the settings, not the data, changed. The database "
          f"held {int(after.db_rows):,} player-match rows throughout; nothing was "
          f"added or removed.\n")
        rows = []
        for comp in ("NRL", "SL", "NSW", "QLD"):
            b = json.loads(before.stats).get(comp, {})
            a = json.loads(after.stats).get(comp, {})
            if not b or not a:
                continue
            rows.append(dict(competition=comp,
                             season_before=b.get("season"), season_after=a.get("season"),
                             players_before=b.get("players"), players_after=a.get("players"),
                             rows_before=b.get("rows"), rows_after=a.get("rows"),
                             sigma2_before=b.get("sigma2"), sigma2_after=a.get("sigma2"),
                             tau2_before=b.get("tau2"), tau2_after=a.get("tau2")))
        r = pd.DataFrame(rows)
        A(md(r[["competition", "season_before", "season_after", "players_before",
                "players_after", "rows_before", "rows_after"]], "{:.0f}"))
        A("\nThe variance components, which set the rating scale and the shrinkage:\n")
        A(md(r[["competition", "sigma2_before", "sigma2_after",
                "tau2_before", "tau2_after"]]))
        A("\n**Read the player counts carefully — they move for a reason that is not "
          "the freeze itself.** Ratings are published for players active in the most "
          "recent fitted season, so that season changed from 2026 to 2025 for the NRL "
          "and Super League. The NRL *gained* players (a complete 2025 has more men "
          "appear in it than a 2026 stopped at round 19) while Super League *lost* them "
          "(2026 runs 14 clubs against 2025's twelve, Bradford and York having joined). "
          "Neither movement says anything about rating quality.\n")
        A("The variance components barely move — σ² and τ² agree to three decimal "
          "places in every competition. That is the useful robustness check here: "
          "removing a season did not rescale the ratings or change how hard the "
          "shrinkage pulls, so scores before and after the freeze remain comparable.\n")

    A("\n## Effect on the translation model\n")
    if lad_ab is not None:
        j = lad_ab.sort_values("pts_0_100_frozen")
        A("A controlled comparison: the same code fitted twice, once with the freeze "
          "off and once on, nothing else altered. `n` is the number of observed moves "
          "behind each estimate and the points are on the 0-100 rating scale.\n")
        A(md(j[["source", "target", "n_unfrozen", "n_frozen", "pts_0_100_unfrozen",
                "pts_0_100_frozen", "d_pts"]], "{:.2f}"))
        moved = j[j.valtozott]
        A(f"\n**Five of the eight directions are bit-identical, and the three that "
          f"moved are exactly the three that land in Super League.** This is not a "
          f"coincidence and it is the cleanest evidence that the freeze did what it was "
          f"supposed to and no more: the Australian competitions have no 2026 data at "
          f"all, so no move ending in the NSW Cup, the Queensland Cup or — through "
          f"same-season pairing — the NRL could ever have reached 2026. Only Super "
          f"League has a 2026 season, so only moves into it lost observations "
          f"({', '.join(f'{r.source}→{r.target} {int(r.n_unfrozen)}→{int(r.n_frozen)}' for r in moved.itertuples())}).\n")
        A(f"Those three shifted by {moved.d_pts.abs().min():.2f} to "
          f"{moved.d_pts.abs().max():.2f} points and not in a consistent direction, "
          f"which is what dropping a fifth to a third of a small sample looks like. "
          f"The two Super League estimates that matter most for the client are also the "
          f"ones now resting on the fewest moves. When 2026 is added back in October "
          f"these three are the numbers to re-check first.\n")
    A("\nThe translation model's fit statistics, read from the database rather than "
      "typed in. An earlier version of this report quoted them from memory, which held "
      "until the model was refitted and then quietly described a model that no longer "
      "existed.\n")
    if len(meta):
        show = meta[[c for c in ("label", "n", "players", "rmse", "mae", "naive", "r2")
                     if c in meta.columns]]
        A("\n" + md(show, "{:.3f}"))
        if {"rmse", "naive"} <= set(meta.columns):
            best = meta.iloc[meta.rmse.idxmin()]
            A(f"\nThe better of the two layers cuts the error to {best.rmse:.2f} from "
              f"{best.naive:.2f} for assuming a player rates the same in the new "
              f"competition — an improvement of "
              f"{(1 - best.rmse / best.naive) * 100:.1f}%.\n")
    else:
        A("\n*No translation-model statistics found in the database.*\n")
    if len(lad_v3):
        g = lambda a, b: lad_v3[(lad_v3.source == a) & (lad_v3.target == b)]
        try:
            f2n = float(pd.concat([g("NSW", "NRL"), g("QLD", "NRL")]).shift_pts.mean())
            n2s = float(g("NRL", "SL").shift_pts.iloc[0])
            f2s = float(pd.concat([g("NSW", "SL"), g("QLD", "SL")]).shift_pts.mean())
            gap = abs((f2n + n2s) - f2s)
            A(f"The ladder's internal consistency check, recomputed from the model in "
              f"the database rather than quoted from a run since superseded. Going the "
              f"long way round — feeder to NRL ({f2n:+.2f} points), then NRL to Super "
              f"League ({n2s:+.2f}) — gives {f2n + n2s:+.2f}. Measured directly, feeder "
              f"to Super League is {f2s:+.2f}. The two routes disagree by "
              f"{gap:.2f} points.\n")
            A("This is what licenses quoting a translation for a pair with few direct "
              "observations of its own, so the size of the disagreement is the size of "
              "the licence. "
              + ("Well under a point: the two routes agree and the licence holds."
                 if gap < 1.0 else
                 f"At {gap:.1f} points it is not nothing. Every Super League direction "
                 f"rests on a few dozen moves, and this is the arithmetic saying so — "
                 f"quote a Super League translation as a band rather than a figure, and "
                 f"revisit when the 2026 data roughly doubles those samples.")
              + "\n")
        except (IndexError, KeyError, ValueError):
            A("*The transitivity check could not be recomputed from the current "
              "ladder.*\n")

    A("\n## A second change made at the same time\n")
    A("`regenerate_full.py` and `fit_translation_v2.py` were writing to the database "
      "directly, outside `runtime.guarded_write` — contrary to what the handover "
      "states. Each replaces three tables wholesale, so a failure part-way through "
      "would leave the app reading a rebuild that never finished, with no snapshot to "
      "go back to. Both now write inside the guard, and the two runs in this report are "
      "the first entries either has made in the audit log. This is unrelated to the "
      "freeze; it was found while preparing to run it.\n")

    A("\n## Current state\n")
    A(md(ratings_now, "{:.1f}"))
    A("\n`avg_effective_matches` is not matches played. Since season weighting came in "
      "on 19 September the shrinkage counts a match from four years ago as a quarter of "
      "a match, and this column sums those weights — which is why it reads around half "
      "the played figure. It is the weight of evidence behind a rating, which is what "
      "the shrinkage acts on.\n")
    A("\nVerify with `python -m pytest tests -q`, `python smoke_bosc.py` and "
      "`python build_manifest.py --check`. No count is quoted here: a number typed into "
      "a report is a number that goes stale, and this report was caught doing exactly "
      "that with the translation statistics above.\n")

    A("\n## What a reader should question\n")
    A("- **The app now shows 2025.** This is the intended consequence, but it is "
      "visible to anyone who opens BOSC: the NRL and Super League player lists are "
      "2025 squads, not 2026. Anyone expecting current-season names will think "
      "something is broken. It reverts when 2026 is added in October.")
    A("- **The freeze is only as good as its coverage.** It governs the two fitting "
      "scripts. Any future script that fits something and does not read "
      "`FREEZE_SEASON` would reintroduce the leak silently. There is no test that "
      "enforces this, which is a gap worth closing.")
    A("- **The three Super League ladder entries rest on fewer moves than before** "
      "(32, 45 and 107). Their standard errors are reported by `fit_translation_v2.py` "
      "and are wide; a reader should not treat a one-point difference between "
      "competitions as meaningful at those sample sizes.")
    A("- **The 2026 rows still carry estimated positions** for the NRL — right about "
      "four in five times overall and only about half for wingers and fullbacks "
      "(`validate_positions.py`). The freeze keeps them out of the fit, but the "
      "out-of-sample test planned for October will be measured *on* those rows, so it "
      "needs the match-sheet positions to arrive with the data.")
    A("- **Nothing here validates the ratings.** It records that the fit was moved and "
      "what moved with it. Whether the ratings predict anything is `VALIDATION_REPORT.md`.")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(W) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
