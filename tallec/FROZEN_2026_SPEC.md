# Frozen 2026 evaluation specification

Status: **adopted 2026-10-08**, implemented in `frozen_2026.py`, release fingerprinted in
`frozen_2026_manifest.json`. Everything below the rule is the seventh external review's
proposal, unchanged. Where the text leaves a choice open, the implementation makes it as
follows:

- *Rule 3.* Evaluation rows are `rolling_backtest.moves_into`, built on
  `transition_events.py`: source by most minutes, then matches, then competition code;
  three or more source appearances; prior-season dual participants excluded; three or
  more ratable matches on both sides.
- *Rule 1.* The models are refitted at run time from the fingerprinted inputs by
  deterministic least squares, so there are no separately stored fitted artefacts. The
  fingerprint covers the files in `frozen_2026.FROZEN_FILES` (this one included), the
  match rows through 2025, the dates of birth of the players in them, the environment
  and the constants. Editing any of them after the freeze makes a 2026 run refuse.
- *Rule 9.* H2 is fitted on the entry-cohort rows with the same estimator as C1.
- *Rule 10.* shrinkage_B is the source rating's B from the cumulative snapshot.
- *Rule 16.* `tests/test_frozen_2026.py`. S0 reproduces the backtest's line on all 441
  evaluated moves of 2023–25 to 1e-13.

---

2026 is exploratory; 2027 is the first confirmatory season. No result from 2027 may select a model, threshold, feature, cohort or reporting rule.

## Data and prediction time

1. Freeze the code, configuration, position mapping, environment versions, through-2025 input snapshot and fitted artefacts with SHA-256 hashes before processing additional 2026 outcomes. Keep the original R6/R7 results unchanged. This freeze does not erase prior exposure to partial 2026 data.
2. Primary use case: an end-of-2025 conditional forecast of the player's 2026 season-only SL peer score, conditional on entering SL and attaining at least three ratable matches. This is not signing success, unconditional arrival probability or a forecast made on an arbitrary signing date.
3. Evaluation rows: one player × target × landing season; first/returning entries under the frozen transition_events rule; source selected from 2025 by most minutes, then matches, then competition code, with at least three source appearances and three ratable source matches. Exclude prior-season dual participants in that target. Apply the existing rating eligibility rules unchanged and publish the exclusion counts. No outcome-based exclusions beyond the declared rating threshold.
4. Source Class, source position and all covariates use information through 2025 only. `rated_source_prev_year` means a nonmissing season-only source rating with n_games >= 3 in 2024. Zero means no qualifying rating in this dataset, not necessarily no playing experience. Preserve missing/uncovered-season status separately for audit.
5. Outcome: the frozen rating algorithm applied to 2026 season-only matches. NRL 2026 match-sheet positions are joined only to 2026 outcome rows (including secondary NRL-target evaluation), never to 2025 source features, historical training rows or career-position lookups. Canonical mapping and existing fallback rules are unchanged; reject duplicate/ambiguous joins and report unmatched coverage. Do not tune quality thresholds or rating weights to obtain favourable errors. An unresolved ingestion defect suspends the affected analysis.

## Models fixed in advance

6. S0 is the shipped-construction straight line: next-season translation pairs ending by 2025; OLS target ~ intercept + source score. Fit one pooled model on ALL pairs and a separate model for each direction with at least 25 rows and nonconstant source score. Otherwise use the pooled model. Clip predictions to [0,100]. Persist the exact training rows and coefficients.
7. S1 adds `rated_source_prev_year` to S0 on the SAME pair rows, with each historical feature calculated relative to that row's source year. Jointly fit unpenalised intercept/source terms and the standardised history feature with sum-of-squared-errors + 1.0 × squared feature coefficient. Fit pooled and qualifying direction models separately; do not implement the fallback as the reference category of a saturated direction-interaction design. Standardise using each fitting sample only, sample SD (ddof=1); a constant feature is zeroed and SD set to 1. Clip to [0,100]. This explicitly replaces the R7 shared-coefficient specification; its historical +0.85 result is not a result for S1.
8. C0/C1 repeat exactly the S0/S1 fitting and fallback rules on the entry-cohort training rows landing in 2021–25. These are secondary specification diagnostics, not alternative primary models selected after seeing 2026.
9. H2 is a separately labelled retrospective secondary analysis on all entry directions: compare C1 with a joint C1 + prior-season, minutes-weighted same-position club-player rating model. The player is excluded from the club average. Numeric covariates use training-only mean/SD, mean imputation and an unstandardised missing flag; penalise each added coefficient by 1.0. First-observed destination club remains outcome-derived, so this analysis cannot establish pre-signing forecasting benefit. Do not describe last season's players as confirmed retained incumbents. A genuinely prospective destination-club experiment requires a separately frozen, timestamped destination rule before its evaluation.
10. One additional descriptive diagnostic is permitted: S0 plus the source rating's already-computed shrinkage_B, using the same joint-fit rules. It tests whether history is acting as an evidence-strength proxy. It cannot replace S1 or create a success claim after H1 fails. No other feature search is part of this evaluation.

## Endpoints, decisions and multiplicity

11. H1 is the sole primary comparison: Δ = mean(|Y−S0| − |Y−S1|) on 2026 SL entries from NRL/NSW/QLD. Positive favours S1. Report both MAEs, Δ and a paired player-cluster percentile bootstrap interval: 4,000 draws, seed 0, stable sorted canonical player IDs, resampling all rows of a selected player together, 2.5th/97.5th percentiles. Models are not refitted in this test-set bootstrap; its interval is conditional on the frozen fit and this season.
12. A positive exploratory H1 signal requires Δ >= 1.0 rating point AND the interval's lower bound > 0. The one-point threshold is a predeclared practical decision rule, not an effect size established by existing evidence. Minimum sample: 25 distinct eligible SL entrants, with at least five in each history category. Below either threshold, the result is UNDECIDED regardless of the interval. These are feasibility safeguards, not a power guarantee.
13. With sufficient sample, report separately: positive signal under rule 12; evidence of harm if the interval's upper bound < 0; a one-point gain not supported if the upper bound < 1.0; otherwise undecided. These labels do not imply model equivalence or signing utility. No automatic production promotion follows a 2026 result.
14. Score 2026 ALONE for the primary decision. Show 2023–25 and any combined historical summary separately as exploratory context; pooling the discovery sample must not turn an undecided 2026 result into success. Do not describe an expected ~30 new entries or ~13% standard-error reduction as guaranteed.
15. H2, C0/C1, shrinkage_B and direction/position breakdowns are descriptive secondary analyses; their intervals are not multiplicity-controlled discovery claims and cannot rescue H1. There is one primary hypothesis, not a menu of whichever comparison clears zero.

## Release and 2027

16. Before the 2026 run, unit-test that the no-extra estimator equals independently fitted own/all-pairs lines, that held-out outcomes/positions cannot change source features or training artefacts, and that app and evaluator use the same frozen release. Export row-level predictions, eligibility, feature timestamps/missingness, baseline identity and hashes locally for independent review. Do not silently drop rows where a forecast is missing; report an implementation failure and the affected count.
17. Before any 2027 outcomes are inspected, freeze the complete 2027 protocol and training cutoff (through 2026), including any changes motivated by exploratory 2026. New covariates require a new predeclared specification. No tuning or model selection on 2027; an underpowered 2027 result stays undecided rather than being rescued by pooling development years.
