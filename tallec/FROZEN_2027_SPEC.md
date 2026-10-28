# Frozen 2027 evaluation specification

Status: **adopted 2026-10-28**, implemented in `frozen_2027.py`, release fingerprinted in
`frozen_2027_manifest.json`. Written under rule 17 of `FROZEN_2026_SPEC.md`: the complete
2027 protocol and its training cutoff (through 2026) are fixed before any 2027 outcome
is loaded. 2027 is the first confirmatory season. No 2027 result may select a model,
threshold, feature, cohort or reporting rule.

## What is tested, and why

The 2026 test found no support for changing the line (`FROZEN_2026_RESULT.md`), and
the exploration after it found no model that beats the line (`METHODS_SUMMARY.md`,
`SHORTLIST_VALIDATION.md`). So 2027 tests the shipped product itself, not a challenger,
on the two claims made for it: that a rating cannot simply be read across into Super
League (a translation is needed), and that the translated forecast says more about a
player than what is known about arrivals in general. No challenger model is under test.
A model change found later needs its own frozen specification and a later season.

## Data and prediction time

1. **Training cutoff 2026.** Every forecast is computed from matches through 2026, with
   the rating algorithm, position mapping and translation code of the frozen commit. The
   cutoff is a constant of `frozen_2027.py` (`TRAIN_THROUGH`), not `config.json`'s
   `evaluation.freeze_season`, so the app may roll its own ratings forward during 2027
   without touching this test.
2. **What is forecast.** The end-of-2026 conditional forecast of a player's 2027
   season-only Super League peer score, conditional on his entering Super League and
   playing at least three ratable matches there. Not signing success, not arrival
   probability, not a forecast made on an arbitrary signing date.
3. **Evaluation rows.** One player per landing in 2027: first or returning entries into
   Super League from the NRL, NSW Cup or Queensland Cup under the frozen
   `transition_events` rule (`rolling_backtest.moves_into`): source chosen from 2026 by
   most minutes, then matches, then competition code; three or more source appearances
   and ratable source matches; players already in Super League in 2026 excluded; three
   or more ratable target matches. Exclusion counts are published. No exclusion on the
   outcome beyond the rating threshold.
4. **Outcome.** The frozen rating algorithm on 2027 Super League matches, season only.
   *Top third*: at or above the two-thirds quantile of the season-only ratings of every
   rated Super League player in 2027 (three or more ratable matches).
5. **Fingerprint.** The frozen files (`frozen_2027.FROZEN_FILES`, this one included);
   `config.json` without its `evaluation` block and without description text; every
   column of `player_match_stats` through 2026; the constants; the fitted artefacts
   (rule 12); the environment versions, reported but not enforced. Dates of birth are
   not used by any 2027 model and are not fingerprinted, so a 2027 feed filling in a
   missing date cannot block the test.
6. **Re-freezing.** Allowed only while the database holds no row after 2026, to take in
   corrections to the through-2026 data (for example the provider's decisions on
   possibly duplicated player IDs). Every freeze is kept in the manifest's history. Once
   a 2027 row is loaded, corrections to through-2026 data wait until this test has run,
   and the run refuses while the inputs differ from the manifest.
7. **Running later.** The manifest records the commit. If the code has moved on, the
   test is run from a checkout of that commit against the live database
   (`git worktree add ../frozen2027 <commit>`); file hashes are checked in the checkout
   that runs.

## Models fixed in advance

8. **S0, the shipped line.** Next-season translation pairs (`B_next_season`) whose
   target season is 2026 or earlier. One pooled OLS line, target on source rating, over
   all pairs, and an own line for each direction with 25 or more pairs and a source
   rating that varies; otherwise the pooled line. Clipped to [0, 100]. This is rule 6 of
   the 2026 specification with the cutoff moved.
9. **B0, nothing known about the player.** The same rows and the same fallback, with
   the slope fixed at zero: the direction's mean target where S0 has an own line, the
   pooled mean otherwise. B0 is the forecast if the source rating told us nothing.
10. **Ranges.** The 10th, 25th, 75th and 90th percentiles of the residuals of the line
    that produced the forecast, on that line's own training rows, added to the forecast
    and clipped to [0, 100] (the app's construction): an 80% and a 50% range.
11. **Top-third probability.** The logistic in `decision_calibration` built through 2026
    (`build_decision_tables.py`), applied to S0, and Super League's base rate from
    `top_third_base`. Both are copied into the manifest at the freeze.
12. **Frozen artefacts.** Each line's coefficients, row count, mean target and residual
    quantiles, and the top-third coefficients and base rate, are stored in the manifest.
    The run recomputes the lines from the fingerprinted inputs and stops if any
    coefficient differs by more than 1e-9.

## Primary endpoints and decision (a fixed sequence)

13. Over the 2027 entrants into Super League from the NRL, NSW Cup and Queensland Cup,
    two hypotheses are tested in a fixed order:
    - **H1a, tested first:** Δa = mean(|Y − C| − |Y − S0|), where C is the source rating
      carried across untranslated. The claim: a translation is needed.
    - **H1b, tested only if H1a is confirmed:** Δb = mean(|Y − B0| − |Y − S0|). The
      claim: the forecast carries information beyond the average arrival.

    Positive favours the forecast in both. A fixed sequence keeps the chance of any false
    confirmation at 5% without splitting it, so each interval is a 95% interval. If H1a
    is not confirmed, H1b is reported as descriptive only. For each: both MAEs, Δ and a
    paired player-cluster percentile bootstrap interval: 4,000 draws, seed 0, sorted
    canonical player IDs, every row of a drawn player together, 2.5th and 97.5th
    percentiles. Models are not refitted inside the bootstrap.
14. **Labels**, for each hypothesis. Fewer than 25 distinct entrants: *UNDECIDED (sample
    below the minimum)*. Otherwise: Δ ≥ 1.0 and lower bound > 0: *CONFIRMED*; upper bound
    < 0: *EVIDENCE OF HARM*; upper bound < 1.0: *A ONE-POINT GAIN IS NOT SUPPORTED*;
    otherwise *UNDECIDED*. The one-point threshold is a declared practical rule, as in
    2026, not an effect size established by earlier data.
15. **2027 alone.** The decision uses 2027 only. 2023–2026, which the hypotheses were
    formed on, are shown separately as context (`frozen_2027.py --context`,
    `FROZEN_2027_CONTEXT.md`): there H1a held in every season with enough entrants (2023
    fell below the minimum) and H1b in none, so with a cohort the size of 2026's an
    undecided H1b is the likely outcome. Pooling the earlier seasons must not turn an
    undecided 2027 into a confirmation.

## Secondary, descriptive

No multiplicity control; none of these can rescue or overturn H1a or H1b.

16. **Ranges.** Share of entrants inside the 50% and the 80% range, each with a 95%
    Wilson interval. Read as consistent with the nominal level when the nominal value
    lies inside the interval.
17. **Shortlist.** The top third of 2027 entrants ranked by S0, by the untranslated
    source rating, by source appearances, source starts, source start share and source
    minutes per match (all from 2026): hits of the 2027 top third for each, and each
    rule's AUC over all entrants.
18. **Top-third probability.** Brier score of rule 11's probability against the stored
    base rate, and the skill score 1 − Brier / Brier(base).
19. **Breakdowns.** Each source competition's MAE for S0 and B0, and S0 against B0 over
    every direction's 2027 entrants. Outside Super League the app forecasts with the
    conditional model, not the line, so the second is context for the line only.

## Release

20. **Tests** (`tests/test_frozen_2027.py`): B0 is the mean of the rows its S0 line is
    fitted on, with the same fallback; S0 equals the 2026 estimator; the decision labels
    and the fixed sequence; the configuration fingerprint ignores the `evaluation` block
    but not a rating weight; a freeze is refused once a season after 2026 is loaded.
21. **Export.** `--out` writes the row-level forecasts, ranges, outcomes, eligibility and
    the fingerprint for independent review. A missing forecast stops the run as an
    implementation failure; rows are never silently dropped.
