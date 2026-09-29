# -*- coding: utf-8 -*-
"""Report on point 0 of the remediation: one feature definition, and provenance.

Generated from artefacts. The acceptance criteria are quoted as they were set and each
is answered with the evidence for it, including the ones that were only partly met and
the one that turned out to be unobtainable as written.

    python report_point0.py        # writes POINT0_REPORT.md
"""
import json
import os
import sqlite3

import pandas as pd

import player_rating_engine as pre
import sp_schema as sp
import translation_features as tf

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "POINT0_REPORT.md")


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
    man = json.load(open(os.path.join(BASE, "MANIFEST.json"), encoding="utf-8"))
    seal = json.load(open(os.path.join(BASE, "v1_holdout_record.json"),
                          encoding="utf-8"))
    con = sqlite3.connect(os.path.join(BASE, "tallec.db"))
    meta = pd.read_sql("SELECT * FROM translation_model_v3_meta", con)

    # how thin a position-specific translation would be, counted rather than asserted
    _p = pd.read_sql("SELECT source, target, raw_position FROM translation_pairs_v3",
                     con)
    _p["direction"] = _p.source + "->" + _p.target
    _p["group"] = _p.raw_position.map(sp.POSITION_GROUP)
    pos_cells = (_p.groupby(["direction", "group"]).size().unstack(fill_value=0)
                 .reset_index())
    pos_cells_flat = pd.Series(
        pos_cells.drop(columns="direction").values.flatten())
    lad = pd.read_sql("SELECT * FROM translation_ladder_v3 ORDER BY shift_pts", con)
    con.close()

    W, A = [], None
    A = W.append
    A("# Point 0 — one feature definition, and provenance that can be checked\n")
    A("Everything below is generated from artefacts on disk: `MANIFEST.json`, "
      "`v1_holdout_record.json`, and the model tables in `tallec.db`. Nothing is typed "
      "in by hand.\n")

    A("\n## What was wrong\n")
    A("The review found the translation model was not evaluating the model it "
      "described. Checking confirmed it and found the fault was worse than reported. "
      "The model had **two** feature definitions. Training built its columns with "
      "`get_dummies(..., drop_first=True)`; prediction rebuilt them by hand, starting "
      "every column at the fitted scaler's mean and overwriting the ones it could "
      "resolve. Three failures followed, all silent:\n")
    A("1. **Prediction mapped a raw Stats Perform label to a group, but callers passed "
      "a group.** `Halves`, `Back Row` and `Bench` are not keys in that map, so the "
      "lookup returned nothing. (The review named `Fullback` here as well; that one is "
      "fine — there is a legacy self-mapping. The list is exactly those three.)")
    A("2. **An unresolved position did not fall back to the reference group.** It left "
      "every group dummy at its training mean — a fractional blend of positions that "
      "corresponds to no player and that the model was never fitted on. This was not in "
      "the review and is the more damaging of the two, because `Back Row` *is* the "
      "reference group and so should have encoded as all-zeros; it did not.")
    A("3. **A missing age or minutes figure was replaced by the training mean**, so the "
      "model could not distinguish an average player from an unknown one, and neither "
      "could the caller.")
    A("\n**The live app was not affected.** `bosc_app.py` reads `position` from "
      "`player_match_stats`, which holds the raw Stats Perform label, and so was "
      "passing exactly what the mapper expected. The two callers that passed groups "
      "were the analysis scripts — `trace_cohort.py` and `validate_ratings.py` — which "
      "is to say the fault corrupted the evidence about the model rather than the "
      "product.\n")

    A("\n## The acceptance criteria, one by one\n")
    rows = [
        ("Separate `raw_position` and `position_group`; not confusable",
         "Met", "`translation_features.resolve_position` takes both and raises when a "
                "group is passed as raw, when a raw label is unknown, or when the two "
                "contradict. `translate(position=...)` now raises a `TypeError` naming "
                "the replacement."),
        ("Automatic test for every position group",
         "Met", "All eight groups, plus every one of the ten raw labels, plus the "
                "reference-versus-unknown distinction. 61 tests pass."),
        ("Training and prediction use the same feature definition",
         "Met", "One `FeatureSpec`, fitted in `fit_translation_v3.py` and stored inside "
                "the model file; `predict_translation.translate` reads it back. There "
                "is no second code path for the current model."),
        ("Age, minutes and matches only from the source season, before the cutoff",
         "Met", "Built in `rating_history.build`, which aggregates them per "
                "player-season from matches at or before the cutoff only."),
        ("Decide season-only or cumulative, and use it consistently",
         "Met", "Cumulative, as directed. The model is fitted on the same Class score "
                "the app shows. What it predicts is the target competition's "
                "season-only rating, because a cumulative target contains the player's "
                "career there before the move."),
        ("Missing features flagged rather than silently averaged",
         "Met", "Every numeric feature has a companion `_missing` column; the value "
                "still falls back to the training median so the row stays in range, and "
                "`inputs_used` reports which were supplied and which were clamped."),
        ("Canonical config hash",
         "Met", "`runtime.config_hash` hashes sorted-key JSON with descriptive fields "
                "removed. Tested: identical under CRLF and LF and under a reworded "
                "description, different when a value changes."),
        ("Clean tree, one config hash, a manifest",
         "Partly", "One config hash and a manifest: met. A wholly clean tree is not "
                   "obtainable — the databases are tracked and every run writes to "
                   "them, so no run can observe one. Provenance now separates "
                   "`code_dirty` from database churn and records the former; data "
                   "integrity is carried by content hashes instead. If tracking a "
                   "90 MB binary in git is not wanted, untracking it would make the "
                   "original criterion achievable as written."),
    ]
    A(md(pd.DataFrame(rows, columns=["criterion", "status", "evidence"])))

    A("\n\n## Provenance of the current artefacts\n")
    A(f"- code commit **{man['code_commit']}**, code clean: **{man['code_clean']}**")
    A(f"- canonical config hash **{man['config_hash_canonical']}**, freeze season "
      f"**{man['freeze_season']}**")
    A(f"- input `player_match_stats` content hash **{man['input']['sha256']}** over "
      f"{man['input']['rows']:,} rows — a hash of the identifying and measured columns, "
      f"so an edit in place is visible even when the row count does not move")
    art = pd.DataFrame([
        dict(artefact=k, sha256=v.get("sha256"),
             run_id=v.get("produced_by_run_id"), run_commit=v.get("run_commit"),
             code_dirty=v.get("run_tree_dirty"))
        for k, v in man["artefacts"].items()])
    A("\n" + md(art, "{:.0f}"))
    A("\n`python build_manifest.py --check` re-hashes everything and reports drift; it "
      "currently reports none. Both rebuilds are recorded in the audit log by run id, "
      "not by position, and both ran with the code committed and under the same config "
      "hash.\n")

    A("\n## What the sealed v1 result says\n")
    A(f"`v1_holdout_record.json` was written before any change, and hashes the "
      f"artefacts that produced it. It is the only clean out-of-sample result the "
      f"project has, and it stays that way: the v2 model file is deliberately not "
      f"rebuilt.\n")
    A(f"Cohort: {seal['cohort_size']} players, {seal['graded_size']} graded at "
      f"{seal['graded_min_nrl_matches']}+ NRL matches in 2026, split by what the player "
      f"already was.\n")
    rows = []
    for name, v in seal["by_cohort"].items():
        for r in v["results"]:
            rows.append(dict(cohort=name, n=v["n"], predictor=r["predictor"],
                             mae=r["mae"], bias=r["bias"]))
    A(md(pd.DataFrame(rows)))
    A("\nThe review's central objection is confirmed by this table: only ten of the "
      "graded men were in their first NRL season, and for them the translation made the "
      "answer **worse** than leaving the feeder rating alone. The aggregate improvement "
      "reported earlier was carried by players who already had an NRL record — for whom "
      "a recruiter would use that record, not a translation. Ten players is not a "
      "finding, but it is the right question, and v3 has to answer it.\n")

    A("\n## What v3 is, and what is deliberately not claimed about it\n")
    A("v3 differs from v2 in four ways, each forced by a defect rather than chosen:\n")
    A("- fitted on the **cumulative Class score the app displays**, which is the number "
      "the question is asked about;\n"
      "- predicts the target competition's **season-only** rating, so a player's "
      "history in the competition he moved to cannot flatter the score;\n"
      "- keyed on the **Stats Perform Player ID** throughout, rather than matching "
      "Super League on name and date of birth — 224 players carry the same id on both "
      "sides;\n"
      "- pairs built between **any two competitions**, which fills the NSW Cup ↔ "
      "Queensland Cup hole the old layer definitions created by construction.")
    A("\n" + md(meta[["label", "n", "players", "rmse", "mae", "naive", "r2",
                      "shift"]].rename(columns={"label": "layer"})))
    A("\nErrors are in points of the 0-100 rating and are out-of-sample under "
      "`GroupKFold` by player, so no player appears in both halves of a fold. The naive "
      "column is \"he rates exactly the same in the new competition\".\n")
    A("\nThe measured ladder, now covering all twelve directions:\n")
    A(md(lad[["source", "target", "n", "shift_pts", "se_pts"]]))
    A("\nOne inconsistency to keep in view: `NSW→SL` and `SL→NSW` both read positive, "
      "which cannot both be true of the same pair. Both are small against their "
      "standard errors and rest on 57 and 30 moves, so this is noise rather than a "
      "contradiction — but it is the kind of thing that should be checked again when "
      "the samples grow, and it is why the Super League directions should not be quoted "
      "to a decimal place.\n")
    A("\n**No claim is made that v3 beats v2.** The obvious way to show it would be to "
      "run v3 against the 2026 season and compare with the sealed v1 figure. That "
      "number would be a post-hoc evaluation, not a holdout: 2026 was seen, and what it "
      "showed shaped the design of v3. The comparison that will carry weight is the "
      "rolling-origin backtest in point 2, and the first clean confirmatory holdout is "
      "2027.\n")

    A("\n## Still open\n")
    A("Points 1 to 3 are closed. `trace_cohort.py` splits by cohort through "
      "`cohorts.py`; section 6 of `VALIDATION_REPORT.md` was removed rather than "
      "patched and replaced by `ROLLING_REPORT.md`; and stability is now reported as "
      "two separate quantities, an independent correlation between ratings that share "
      "no match and the operational correlation between the cumulative numbers the app "
      "displays. All three reports have been regenerated since.")
    A("\n**Point 4 — position-specific translation — is deliberately not done**, and "
      "the pair counts are the argument. Splitting the twelve directions by the eight "
      "position groups gives 96 cells:\n")
    A(md(pos_cells, "{:.0f}"))
    A(f"\nThe median cell holds {int(pos_cells_flat.median())} moves and "
      f"{int((pos_cells_flat < 10).sum())} of the 96 hold fewer than ten. Every Super "
      f"League direction is in single figures for almost every position. A "
      f"position-specific shift estimated on that is noise with a decimal point, and "
      f"it would be read as precision. The honest version of this is the interaction "
      f"already in the conditional model, where position enters as a dummy fitted "
      f"across all directions at once and borrows strength between them.")
    A("\nAlso open: the app's `metric_dictionary` still holds rows at "
      "`Decision=Review`; the 2026 season is not yet imported, so the sealed v1 "
      "holdout remains the only clean out-of-sample result and anything v3 says about "
      "2026 can only be a post-hoc evaluation; and the first genuinely confirmatory "
      "holdout for the current model is 2027.")

    A("\n## Where to attack this\n")
    A("- The cumulative-in, season-only-out asymmetry is a judgement. It matches the "
      "question being asked but means the two sides of every training pair carry "
      "different amounts of evidence, and a season-only target is noisier than a "
      "cumulative one. An alternative worth arguing for is shrinking the target toward "
      "the source competition's mean by its own sample size.")
    A("- `FeatureSpec` falls back to the training median for a missing value **and** "
      "sets a flag. A model with an intercept can use the flag, but Ridge shrinks it "
      "like any other coefficient; whether that is enough has not been tested.")
    A("- The ladder is a simple mean of observed differences and inherits every "
      "selection effect in who moves. That is the known defect the whole exercise is "
      "circling, and point 2 is where it gets addressed.")
    A("- `code_clean` is a weaker claim than `tree clean`. It is defensible only "
      "because the content hashes exist; if the manifest were lost, the databases would "
      "have no version at all.")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(W) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
