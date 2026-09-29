# Point 0 — one feature definition, and provenance that can be checked

Everything below is generated from artefacts on disk: `MANIFEST.json`, `v1_holdout_record.json`, and the model tables in `tallec.db`. Nothing is typed in by hand.


## What was wrong

The review found the translation model was not evaluating the model it described. Checking confirmed it and found the fault was worse than reported. The model had **two** feature definitions. Training built its columns with `get_dummies(..., drop_first=True)`; prediction rebuilt them by hand, starting every column at the fitted scaler's mean and overwriting the ones it could resolve. Three failures followed, all silent:

1. **Prediction mapped a raw Stats Perform label to a group, but callers passed a group.** `Halves`, `Back Row` and `Bench` are not keys in that map, so the lookup returned nothing. (The review named `Fullback` here as well; that one is fine — there is a legacy self-mapping. The list is exactly those three.)
2. **An unresolved position did not fall back to the reference group.** It left every group dummy at its training mean — a fractional blend of positions that corresponds to no player and that the model was never fitted on. This was not in the review and is the more damaging of the two, because `Back Row` *is* the reference group and so should have encoded as all-zeros; it did not.
3. **A missing age or minutes figure was replaced by the training mean**, so the model could not distinguish an average player from an unknown one, and neither could the caller.

**The live app was not affected.** `bosc_app.py` reads `position` from `player_match_stats`, which holds the raw Stats Perform label, and so was passing exactly what the mapper expected. The two callers that passed groups were the analysis scripts — `trace_cohort.py` and `validate_ratings.py` — which is to say the fault corrupted the evidence about the model rather than the product.


## The acceptance criteria, one by one

| criterion | status | evidence |
| --- | --- | --- |
| Separate `raw_position` and `position_group`; not confusable | Met | `translation_features.resolve_position` takes both and raises when a group is passed as raw, when a raw label is unknown, or when the two contradict. `translate(position=...)` now raises a `TypeError` naming the replacement. |
| Automatic test for every position group | Met | All eight groups, plus every one of the ten raw labels, plus the reference-versus-unknown distinction. 61 tests pass. |
| Training and prediction use the same feature definition | Met | One `FeatureSpec`, fitted in `fit_translation_v3.py` and stored inside the model file; `predict_translation.translate` reads it back. There is no second code path for the current model. |
| Age, minutes and matches only from the source season, before the cutoff | Met | Built in `rating_history.build`, which aggregates them per player-season from matches at or before the cutoff only. |
| Decide season-only or cumulative, and use it consistently | Met | Cumulative, as directed. The model is fitted on the same Class score the app shows. What it predicts is the target competition's season-only rating, because a cumulative target contains the player's career there before the move. |
| Missing features flagged rather than silently averaged | Met | Every numeric feature has a companion `_missing` column; the value still falls back to the training median so the row stays in range, and `inputs_used` reports which were supplied and which were clamped. |
| Canonical config hash | Met | `runtime.config_hash` hashes sorted-key JSON with descriptive fields removed. Tested: identical under CRLF and LF and under a reworded description, different when a value changes. |
| Clean tree, one config hash, a manifest | Partly | One config hash and a manifest: met. A wholly clean tree is not obtainable — the databases are tracked and every run writes to them, so no run can observe one. Provenance now separates `code_dirty` from database churn and records the former; data integrity is carried by content hashes instead. If tracking a 90 MB binary in git is not wanted, untracking it would make the original criterion achievable as written. |


## Provenance of the current artefacts

- code commit **73f0772**, code clean: **False**
- canonical config hash **ab97bd7b51fb**, freeze season **2025**
- input `player_match_stats` content hash **6a3cf3a71b2870d1** over 122,359 rows — a hash of the identifying and measured columns, so an edit in place is visible even when the row count does not move

| artefact | sha256 | run_id | run_commit | code_dirty |
| --- | --- | --- | --- | --- |
| translation_model_v3.pkl | 31eb60e92a298bf2 | 31 | cc94a46 | True |
| translation_model_v2.pkl | ef528cf680e7b102 |  | None | None |
| config.json | 7620bf4ed0c58327 |  | None | None |
| v1_holdout_record.json | 82998d427aa99a7e |  | None | None |

`python build_manifest.py --check` re-hashes everything and reports drift; it currently reports none. Both rebuilds are recorded in the audit log by run id, not by position, and both ran with the code committed and under the same config hash.


## What the sealed v1 result says

`v1_holdout_record.json` was written before any change, and hashes the artefacts that produced it. It is the only clean out-of-sample result the project has, and it stays that way: the v2 model file is deliberately not rebuilt.

Cohort: 134 players, 79 graded at 5+ NRL matches in 2026, split by what the player already was.

| cohort | n | predictor | mae | bias |
| --- | --- | --- | --- | --- |
| NRL regular in 2025 as well | 60 | ladder | 7.55 | 2.86 |
| NRL regular in 2025 as well | 60 | ridge | 6.84 | 3.01 |
| NRL regular in 2025 as well | 60 | no translation | 7.94 | -3.85 |
| first NRL season | 10 | ladder | 5.36 | 4.81 |
| first NRL season | 10 | ridge | 5.63 | 1.99 |
| first NRL season | 10 | no translation | 4.31 | -2.23 |
| returning after a season away | 9 | ladder | 5.86 | 1.29 |
| returning after a season away | 9 | ridge | 5.83 | -0.22 |
| returning after a season away | 9 | no translation | 8.18 | -5.48 |

The review's central objection is confirmed by this table: only ten of the graded men were in their first NRL season, and for them the translation made the answer **worse** than leaving the feeder rating alone. The aggregate improvement reported earlier was carried by players who already had an NRL record — for whom a recruiter would use that record, not a translation. Ten players is not a finding, but it is the right question, and v3 has to answer it.


## What v3 is, and what is deliberately not claimed about it

v3 differs from v2 in four ways, each forced by a defect rather than chosen:

- fitted on the **cumulative Class score the app displays**, which is the number the question is asked about;
- predicts the target competition's **season-only** rating, so a player's history in the competition he moved to cannot flatter the score;
- keyed on the **Stats Perform Player ID** throughout, rather than matching Super League on name and date of birth — 224 players carry the same id on both sides;
- pairs built between **any two competitions**, which fills the NSW Cup ↔ Queensland Cup hole the old layer definitions created by construction.

| layer | n | players | rmse | mae | naive | r2 | shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_same_season | 1436 | 439 | 23.47 | 19.56 | 31.27 | 0.17 | 0.18 |
| B_next_season | 1512 | 623 | 23.83 | 19.94 | 30.83 | 0.12 | 1.58 |

Errors are in points of the 0-100 rating and are out-of-sample under `GroupKFold` by player, so no player appears in both halves of a fold. The naive column is "he rates exactly the same in the new competition".


The measured ladder, now covering all twelve directions:

| source | target | n | shift_pts | se_pts |
| --- | --- | --- | --- | --- |
| SL | NRL | 12 | -22.48 | 9.06 |
| SL | NRL | 35 | -19.08 | 4.51 |
| QLD | NRL | 190 | -17.58 | 2.05 |
| SL | NRL | 23 | -17.30 | 5.10 |
| QLD | NRL | 356 | -16.00 | 1.52 |
| QLD | NRL | 166 | -14.20 | 2.27 |
| QLD | SL | 13 | -13.88 | 5.12 |
| NSW | NRL | 464 | -11.78 | 1.31 |
| QLD | SL | 45 | -11.35 | 3.75 |
| NSW | NRL | 857 | -10.40 | 0.96 |
| QLD | SL | 32 | -10.32 | 4.89 |
| NSW | NRL | 393 | -8.77 | 1.41 |
| QLD | NSW | 71 | -5.02 | 3.46 |
| SL | NSW | 12 | -4.43 | 5.57 |
| QLD | NSW | 98 | -3.86 | 2.74 |
| SL | NSW | 30 | -2.88 | 3.92 |
| SL | QLD | 19 | -2.31 | 5.54 |
| SL | NSW | 18 | -1.85 | 5.49 |
| QLD | NSW | 27 | -0.81 | 4.04 |
| NSW | QLD | 27 | 1.34 | 4.16 |
| SL | QLD | 32 | 2.31 | 3.79 |
| NSW | SL | 12 | 2.91 | 3.93 |
| NSW | QLD | 135 | 3.96 | 2.36 |
| NSW | QLD | 108 | 4.62 | 2.76 |
| NSW | SL | 57 | 5.99 | 3.15 |
| NSW | SL | 45 | 6.82 | 3.86 |
| SL | QLD | 13 | 9.06 | 4.19 |
| NRL | NSW | 364 | 11.44 | 1.61 |
| NRL | NSW | 828 | 12.02 | 1.03 |
| NRL | NSW | 464 | 12.47 | 1.34 |
| NRL | SL | 107 | 13.38 | 2.62 |
| NRL | SL | 119 | 13.59 | 2.54 |
| NRL | SL | 12 | 15.45 | 9.89 |
| NRL | QLD | 166 | 17.82 | 2.24 |
| NRL | QLD | 356 | 17.96 | 1.56 |
| NRL | QLD | 190 | 18.09 | 2.17 |

One inconsistency to keep in view: `NSW→SL` and `SL→NSW` both read positive, which cannot both be true of the same pair. Both are small against their standard errors and rest on 57 and 30 moves, so this is noise rather than a contradiction — but it is the kind of thing that should be checked again when the samples grow, and it is why the Super League directions should not be quoted to a decimal place.


**No claim is made that v3 beats v2.** The obvious way to show it would be to run v3 against the 2026 season and compare with the sealed v1 figure. That number would be a post-hoc evaluation, not a holdout: 2026 was seen, and what it showed shaped the design of v3. The comparison that will carry weight is the rolling-origin backtest in point 2, and the first clean confirmatory holdout is 2027.


## Still open

Points 1 to 3 are closed. `trace_cohort.py` splits by cohort through `cohorts.py`; section 6 of `VALIDATION_REPORT.md` was removed rather than patched and replaced by `ROLLING_REPORT.md`; and stability is now reported as two separate quantities, an independent correlation between ratings that share no match and the operational correlation between the cumulative numbers the app displays. All three reports have been regenerated since.

**Point 4 — position-specific translation — is deliberately not done**, and the pair counts are the argument. Splitting the twelve directions by the eight position groups gives 96 cells:

| direction | Bench | Centre | Edge | Fullback | Halves | Hooker | Middles | Winger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NRL->NSW | 0 | 147 | 109 | 59 | 133 | 62 | 227 | 91 |
| NRL->QLD | 1 | 80 | 38 | 19 | 56 | 26 | 85 | 43 |
| NRL->SL | 5 | 16 | 21 | 9 | 21 | 5 | 33 | 8 |
| NSW->NRL | 0 | 168 | 117 | 70 | 133 | 61 | 238 | 70 |
| NSW->QLD | 5 | 22 | 24 | 7 | 21 | 8 | 28 | 20 |
| NSW->SL | 0 | 7 | 10 | 3 | 9 | 4 | 19 | 5 |
| QLD->NRL | 0 | 74 | 50 | 23 | 59 | 25 | 74 | 41 |
| QLD->NSW | 2 | 17 | 15 | 6 | 17 | 8 | 18 | 15 |
| QLD->SL | 2 | 6 | 5 | 1 | 2 | 7 | 19 | 2 |
| SL->NRL | 0 | 3 | 7 | 2 | 12 | 5 | 5 | 1 |
| SL->NSW | 2 | 3 | 5 | 1 | 7 | 2 | 9 | 1 |
| SL->QLD | 1 | 7 | 4 | 0 | 5 | 3 | 11 | 1 |

The median cell holds 9 moves and 49 of the 96 hold fewer than ten. Every Super League direction is in single figures for almost every position. A position-specific shift estimated on that is noise with a decimal point, and it would be read as precision. The honest version of this is the interaction already in the conditional model, where position enters as a dummy fitted across all directions at once and borrows strength between them.

Also open: the app's `metric_dictionary` still holds rows at `Decision=Review`; the 2026 season is not yet imported, so the sealed v1 holdout remains the only clean out-of-sample result and anything v3 says about 2026 can only be a post-hoc evaluation; and the first genuinely confirmatory holdout for the current model is 2027.

## Where to attack this

- The cumulative-in, season-only-out asymmetry is a judgement. It matches the question being asked but means the two sides of every training pair carry different amounts of evidence, and a season-only target is noisier than a cumulative one. An alternative worth arguing for is shrinking the target toward the source competition's mean by its own sample size.
- `FeatureSpec` falls back to the training median for a missing value **and** sets a flag. A model with an intercept can use the flag, but Ridge shrinks it like any other coefficient; whether that is enough has not been tested.
- The ladder is a simple mean of observed differences and inherits every selection effect in who moves. That is the known defect the whole exercise is circling, and point 2 is where it gets addressed.
- `code_clean` is a weaker claim than `tree clean`. It is defensible only because the content hashes exist; if the manifest were lost, the databases would have no version at all.
