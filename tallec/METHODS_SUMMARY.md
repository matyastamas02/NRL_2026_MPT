# What every method tried after the 2026 test found (after independent review)

All of it is exploratory. Five independent reviewers re-ran or re-implemented each
analysis, audited the code and checked every number in the first version of this file.
This version carries their corrections. Bugs they found were fixed and the analyses
re-run. Where a claim did not survive, it is withdrawn here.

Conventions:
- **Rolling.** Each landing season is forecast from the seasons before it.
- **"Into SL"** means entries into Super League from the NRL, NSW Cup and Queensland Cup,
  landing 2023–2026 (132 entrants).
- **Intervals** are 95%: player-cluster bootstraps for the translation work; bootstraps
  over matches for GIGOT, market and weather; and, for matching, with the matches held
  fixed, which makes them 30–50% too narrow.
- **Multiplicity.** About 30 alternatives were scored on the same entrants and fixtures.
  An unadjusted interval that excludes zero is not, on its own, evidence.
- **Look-ahead.** 2023–25 were used to build the line, and 2026 aggregates had been seen
  before these methods were run. 2027 is the first season that tests anything here.

## New data: Mike's Super League history workbooks

`ingest_sl_history.py` writes `sl_teamsheets` and `sl_match_lines`
(see `SL_HISTORY_REPORT.md`). The workbooks cover 2021–2026, 2,021 club-matches.

- Alignment: 98.8% of club-matches matched to a Stats Perform round, and the aligned
  opposition agrees 99.9% of the time.
- Names: 97.6% resolved. The surname fallback produced 3 false matches, all among the
  reserves.
- Presence in Stats Perform: shirts 1–13 appear 99.8% of the time, the bench 98.2%, the
  18th–21st men 0.9%.
- Lines: the two clubs' handicaps mirror each other 97.3% of the time. Handicap signs
  check out: correlation with the margin is −0.60.

Unknown: which bookmaker the line comes from and when it was taken; what the numbers in
the week headers mean; and what the bracketed second total in 2022–23 is. The sheets
were completed after the match, because tries are marked on them.

## Can the translation be made more accurate? No

Δ is the MAE gain over the app's line. Δ > 0 means the alternative is closer.

| Method | Into SL | All directions |
|---|---|---|
| Component translation, rating basis as known at each cut-off, pcm dropped (`explore_post2026.py`) | +0.27 [−0.58, +1.11] | +0.17 [−0.15, +0.49] |
| Gradient boosting: player-grouped early stopping, five seeds (`gbm_style.py`) | −0.05 [−1.08, +0.98] | +0.40 [−0.05, +0.84] |
| Kalman level instead of Class (`kalman_ability.py`) | −0.02 [−0.61, +0.58] | −0.02 [−0.24, +0.21] |
| Hierarchical Bayes, priors scaled to the coefficients (`bayes_translation.py`) | −0.07 [−0.60, +0.46] | +0.06 [−0.12, +0.23] |
| Empirical-Bayes pooling of the direction lines | −0.25 [−0.91, +0.41] | −0.13 [−0.41, +0.17] |
| Club-adjusted source rating, AKM (`akm_clubs.py`) | −0.11 [−0.32, +0.12] | −0.00 |
| Style fit, player against destination club (`gbm_style.py`) | −0.13 [−0.43, +0.15] | −0.03 |
| Pairwise shortlist ranker (`rank_survival.py`) | Spearman 0.28 vs 0.28 | 0.29 vs 0.28 |

The first version reported larger gains for two of these, and both came from bugs:
- **Component translation (+0.55)** chose the Queensland Cup's rating basis from the
  whole history. That basis was not the one known before 2025.
- **Gradient boosting (+0.54)** was one random seed that happened to sit at the top of
  the seed distribution. Its early stopping also split at random, so the same player's
  other pairs leaked into validation.

**Reference point, not a ceiling.** Super League players who stayed are forecast from
their own previous season with an MAE of 16.0 (Class) or 15.7 (Kalman level) over
2023–26, against 17.4 for entrants. The Kalman gain for SL stayers, +0.33
[+0.02, +0.66], is one of five comparisons and goes the other way for the Queensland Cup
(−0.17). It is not established.

## Ranges and probabilities

**Forecast ranges.** In the exploratory backtest, the app's residual-quantile ranges held
50% and 83% into SL, and 48% and 80% across all directions.
- On the unseen 2026 entrants the ranges of the conditional-model directions held 79%
  (80% range) and 45% (50% range). For moves into the Queensland Cup the 50% range held
  only 36%.
- Into SL, the 80% coverage falls season by season: 91, 88, 80, 76%.
- The old ±1.96 SD band covered 100% but was 87 points wide (80 when clipped to 0–100).
- Conformal ranges do not do better.
- The comparables card's middle half held 37% into SL and 41% across all directions.

**P(top third).** The logistic is now fitted on out-of-sample line forecasts, each made
from earlier seasons only. The slope is 0.069 per point, and the predicted and observed
shares agree within 0.05 in each fifth. Each target competition has its own base rate:
NRL 31%, NSW Cup 39%, Queensland Cup 41%, Super League 46%. Into SL, the top third by
forecast reached the top third 29 times in 45 (64%) against a 48% base. In 2023 the
shortlist did worse than the base. Brier skill is +0.06 against the SL base rate.

## Role

- **Who starts.** The line forecast does not predict who gets a starting shirt: AUC 0.38
  [0.25, 0.51] on the 77 entrants of 2025–26, 0.47 on all 131. The source start share
  does slightly better: 0.59 [0.46, 0.71] here and 0.64 in the exploration. Both
  intervals are close to 0.5.
- **Two-part forecast.** It ranks starts × rating better than the line, by +0.29
  [+0.06, +0.54] in Spearman correlation, but only because the line ranks that outcome
  backwards. It is no better than the source start share alone, and its probabilities are
  no better calibrated than the base rate.
- **Role and the rating.** Role explains almost none of what the line misses (R² 0.004).
  Entrants who became regular starters show a larger matched shift (+15.4 against +4.3).
  Starting is itself an outcome of the landing season, so this is descriptive.
- **Bench flag.** Into SL, 39% of 31 bench entrants became regular starters against 52%
  of 100 starters (p = 0.22). The app now says the gap could be chance. Only for the
  Queensland Cup is it significant (p = 0.03).
- **Staying in SL.** Of 130 first entries landing 2021–25, 73% were still in SL after
  one season and 43% after two. A good first season keeps a player there: odds ratio
  0.64 [0.45, 0.85] per SD. The age effect does not survive multiplicity. "Left" also
  counts seasons missed through injury.

## How far apart are the competitions?

In peer points; positive means the first competition is stronger.

**NRL–SL: 11–17 points, robust.**

| Method | NRL–SL gap |
|---|---|
| Matched | 12.9 |
| Doubly robust | 14.0 (13.9 on the matched movers) |
| AKM | 15.7 (17.1 with a linear age term) |
| Coarsened exact matching and OLS cross-checks | 11.6–14.8 and 13.7 |
| Symmetric Lee bounds at 3+ matches | [+10.9, +18.5], 95% [+5.5, +26.6] |

It survives hidden selection up to Γ 1.65.

**NRL–Cups: matching understates them.** Promoted Cup players rated +17.1 [+10.0, +23.8]
above their matched stayers in the same season. So promotions are selected on something
the season-before profile does not record, and that biases the matched NRL–Cup gaps
toward zero.

The AKM is the more credible estimator here:

| Pair | AKM | Same-season dual registrations | Best reading |
|---|---|---|---|
| NRL–NSW Cup | 11.1 [8.9, 12.9] | +12.6 | about 9–12 |
| NRL–Queensland Cup | 15.3 | +18.1 | about 10–16 |

The true difference-in-differences gives about 10 and 15, but its pre-trends fail. The
3+ match Lee bounds are uninformative for these pairs: [−13.6, +25.0] and
[−14.7, +23.7].

**SL against the Cups.**
- **SL against the Queensland Cup: not resolved.** Lee 3+: [−2.6, +1.8].
- **The NSW Cup is probably stronger than SL.** Lee 3+: [+6.0, +11.2], 95% [−3.9, +20.4].

**Withdrawn:**
- the 1+ match Lee bounds of the first version, because stayers who left the game had
  been counted as at risk;
- the Heckman correction, because the vacancy is a usable exclusion restriction only for
  NSW Cup → NRL, and even there the result is uninformative;
- the Dolphins natural experiment (n = 2).

## The effect of moving into Super League

- **About +10 in the landing season.** Matched, doubly robust, and the DR-learner median
  over 50 splits (+10.3 to +10.9) all agree.
- **The players were already rising.** The true DiD placebo shows movers into SL on a
  rising path before the move: +5.1 [+0.7, +9.5]. Part of the shift is that path
  continuing.
- **No further gain in season two:** +1.8 [−2.9, +6.7].
- **Withdrawn: heterogeneity.** Under 0.05–0.95 trimming, β2 is significant in only 9 of
  50 splits on the 0–100 scale and 19 of 50 on the probit scale. What does show is weaker
  persistence across a move, which the line already encodes. It is not evidence that some
  players benefit from moving more than others.
- **Withdrawn: quantile differences.** The 0–100 ceiling alone predicts the pattern.
- **Withdrawn: the measurement-error "exchange rate".** Class is already shrunk, so
  dividing by a reliability corrects twice, and Deming's assumptions fail.

## Club effects

Club effects reproduce in the NRL (split-half 0.65, true SD 5.1 points) but barely in
Super League (0.25, SD 2.5). Adjusting the source rating for them does not help the
translation.

## The match model and the market (Super League)

Inputs: the master read with the team codes fixed and ELO recomputed, the 36 fixtures
whose pairing disagrees with Stats Perform dropped, and only mirrored handicaps used.

**The player layer.** Gain over ELO + home:

| Input | Gain | Status |
|---|---|---|
| 21-man squad, named before the match | **+0.74 [+0.27, +1.22]**, 586 fixtures 2023–26 | the pre-match variant |
| Named 17 | +0.89 | close to hindsight |
| Actual line-up | +0.92 | hindsight |

By season, the named-21 gain is +0.51, +1.13, +1.36 and **−0.31 in 2026**. The gain did
not show in the latest season.

**Against the betting line.**
- The line itself (MAE 14.62) is more accurate than every model.
- The model carries information beyond the line: coefficient +0.25 [+0.01, +0.47].
- Line + named 21 covered 55.5% [51.3, 59.4] of 577 matches. That is the best of 16
  predictor and threshold combinations, so after multiple testing it is not significant
  against the 52.4% break-even. If the stored line is an opening line, team news could
  explain it. **It is not a demonstrated edge.**
- The shipped xLadder prediction covered 51.6% in 2025.

**Market rating.** A team rating built from earlier lines is no better than ELO. The
player layer adds +0.82 [+0.30, +1.34] on top of it.

**Weather.** The totals-under-the-line effect disappears once each season's own level is
taken out: damp −2.8 [−5.7, +0.2], wet −1.6 [−3.6, +0.5]; for 2024–26 only, damp −1.2,
wet −0.7. Errors per 80 minutes do not change with weather.

## Data issues found by the review

- **The SL master's team codes.** The 2022–25 rows used eight codes the other way round.
  This is fixed in `xladder_pipeline.fix_sl_team_codes` (see CLAUDE.md). The repo xlsx
  and the Google Sheet are not yet rewritten.
- **Wrong pairings in the SL master.** 36 rows have pairings that match neither Stats
  Perform nor Mike's sheets: 26 in 2023 (LS/WA, shifted a round) and 10 in 2025. For the
  xLadder owner.
- **Reversed scores.** In about half of the 2022–24 master rows, `A Score` and `B Score`
  are reversed against the codes. `Margin` and `A_Points Scored` are right, so the GIGOT
  scripts use those.
- **Mike's sheets.**
  - "(1)"-style try markers are not parsed.
  - 22 team-weeks list a player twice.

## Fixed in this round, and still open

**Fixed:**
- the top-third calibration, now out of sample with a base rate per target;
- the bench-flag caveat and the "exploratory backtest" wording in the app and the client
  note;
- the GIGOT inputs;
- the symmetric Lee at-risk sets;
- the true DiD;
- the DR-learner, now run over many splits;
- the Queensland Cup rating basis per cut-off in the component, GBM and Kalman tests;
- GBM grouped early stopping and seed bagging;
- the Bayes priors;
- pcm dropped on both sides;
- the target-specific Brier base;
- the Deming and quantile claims withdrawn;
- the matching caveats.

**Open (small):**
- the AKM linear age term (SL 15.7 → 17.1) and the label on its naive SD;
- the Heckman perfect-prediction guard, and a vacancy defined from appearances after the
  fact;
- one player at two clubs who flips the role label;
- the survival definition of "left";
- the in-sample lines used for the 2021–22 landings in the survival model;
- the conformal `min(n, k)` edge case.
