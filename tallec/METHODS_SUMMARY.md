# What every method tried after the 2026 test found

Exploratory throughout. Each line points to the script and its report. Unless stated,
forecasts are rolling (each landing season from the seasons before it) and intervals are
95% player-cluster bootstrap. "Into SL" means entries from the NRL, NSW Cup and
Queensland Cup into Super League, 2023–2026 (132 entrants).

## New data: Mike's Super League history workbooks

`ingest_sl_history.py` → `sl_teamsheets`, `sl_match_lines` (`SL_HISTORY_REPORT.md`).
2021–2026, 2,021 club-matches, 98.8% aligned to Stats Perform rounds, 97.6% of names
resolved. Shirts 1–13 are in Stats Perform 99.8% of the time, the bench 98.2%, the
18th–21st men 0.9%: the four extra names are the unused part of the 21-man squad.
Handicaps mirror between the two clubs 97.3%, scores agree 98.8%. Unknown: which
bookmaker and when the line was taken, the meaning of the numbers in the week headers.

## Is the translation forecast improvable?

| Method | Into SL, Δ MAE vs the line | All directions | Verdict |
|---|---|---|---|
| Component translation (`explore_post2026.py`) | +0.55 [−0.30, +1.41] | +0.30 [−0.03, +0.64] | not established |
| Gradient boosting, full profile (`gbm_style.py`) | +0.49 [−0.50, +1.48] | **+0.54 [+0.04, +1.02]** | small gain across directions; a 2027 candidate |
| Kalman level instead of Class (`kalman_ability.py`) | +0.25 [−0.38, +0.91] | +0.06 [−0.17, +0.31] | not established |
| Hierarchical Bayes (`bayes_translation.py`) | −0.18 [−0.68, +0.33] | +0.04 [−0.18, +0.26] | no gain |
| Empirical-Bayes pooling of directions (`explore_post2026.py`) | −0.25 [−0.91, +0.41] | −0.13 | worse |
| Club-adjusted source rating, AKM (`akm_clubs.py`) | −0.11 [−0.32, +0.12] | −0.00 | no gain |
| Style fit, player vs club (`gbm_style.py`) | −0.12 [−0.42, +0.16] | −0.03 | no gain |
| Pairwise shortlist ranker (`rank_survival.py`) | Spearman 0.28 vs 0.28 | 0.29 vs 0.28 | no gain |

The persistence ceiling explains why: players who stayed in Super League are predicted
from their own previous season with about the same error (16.4) as entrants are
translated (16.6). The Kalman level predicts stayers slightly better in Super League
(+0.33 [+0.02, +0.66]).

## Ranges and probabilities

- The old ±1.96 SD band covered 100% into SL at 87 points wide; the residual-quantile
  ranges now in the app covered 50% and 83% (`explore_post2026.py`). Split, normalised
  and Mondrian conformal ranges do not beat them (`conformal_ranges.py`).
- P(top third) is informative (top-third shortlist 64% against a 48% base; Brier skill
  +0.10); the Bayesian predictive gives the same Brier (`bayes_translation.py`).

## Role

- The rating does not predict who gets a starting shirt (AUC 0.38 into SL); the source
  start share does, weakly (0.59). A two-part forecast does not beat the line on
  starts × rating, and role explains almost none of what the line misses (R² 0.004)
  (`role_hurdle.py`).
- Entrants who became regular starters show a larger matched shift (+15.4 against +4.3).
- 73% of entrants into SL are still there after one season, 43% after two; a good first
  season (odds ratio 0.64 per SD) and youth keep them (`rank_survival.py`).

## How far apart are the competitions?

Gap in season-only peer points (positive = the first competition is stronger):

| Pair | Matched | Doubly robust | Staggered DiD | AKM | Lee bounds, 1+ match |
|---|---|---|---|---|---|
| NRL–NSW Cup | +4.1 | +7.7 | ~+7.5 | +11.1 | [−0.2, +11.2] |
| NRL–QLD Cup | +5.6 | +10.8 | ~+10.9 | +15.3 | [+2.1, +9.5] |
| NRL–Super League | +12.9 | +14.0 | — | +15.7 | [+7.8, +18.8] |
| NSW–QLD | +5.2 | +6.2 | — | +4.2 | [−8.1, +15.4] |
| NSW Cup–Super League | +8.0 | +5.1 | — | +4.6 | [+1.4, +17.7] |
| QLD Cup–Super League | +1.2 | −3.1 | — | +0.4 | [−6.8, +14.7] |

(`matching_moves.py`, `robust_gaps.py`, `did_dolphins.py`, `akm_clubs.py`, `lee_bounds.py`.)
NRL > NSW Cup > QLD Cup ≈ Super League. The NRL–Super League gap, 13–16, holds across
methods and survives hidden selection up to Γ 1.65–1.70 (RV 0.16). The Cup gaps depend
on the identification by a factor of about two. About 45% of moves between the NRL and
the Cups never reach three matches; Heckman with the club vacancy as exclusion
restriction is usable only for NSW Cup → NRL (`heckman_attrition.py`), so the bounds
are the honest statement. The Dolphins' 2023 entry yields two matched promotions and
says nothing (`did_dolphins.py`).

## The effect of moving into Super League

- About +10 to +11 points in the landing season (matched, DiD, DR-learner +10.5
  [+7.2, +13.8]); no further gain in season two (+1.0 [−3.8, +6.0]).
- Heterogeneous (DR-learner BLP β2 +0.40 [+0.12, +0.67]): largest for lower-rated
  regular starters (+20.5 in the top third of predicted effect, source rating ~39,
  starting 84%), smallest for high-rated players (+7.8, rating ~71) (`dr_learner.py`).
  Lower quantiles shift more than the top (`deming_qte.py`), partly the 0–100 ceiling.
- Measurement-error-corrected exchange-rate slopes are 0.43–0.63 against the forecast's
  0.23–0.39; Deming's slopes above 1 are its no-change assumption failing.

## Club effects

Reproducible in the NRL (split-half 0.65, true SD 5.1 points), barely in Super League
(0.25, SD 2.5) (`akm_clubs.py`).

## The match model and the market (Super League)

- The player layer survives without hindsight: named 17 +0.93 [+0.48, +1.38], named 21
  +0.73 [+0.31, +1.20], against +0.93 with the actual line-up (`gigot_sl_named.py`).
- The line (MAE 14.51) is more accurate than every model (15.0–16.1). The model adds
  information beyond the line (coefficient +0.30 [+0.07, +0.52]); line + named 21 covers
  55.3% [51.4, 59.2] of 586, not clear of the 52.4% break-even. The shipped xLadder
  prediction covered 51.9% in 2025.
- A team rating from the market's earlier lines is no better than result ELO; the player
  layer adds +0.75 [+0.31, +1.23] on top of it (`market_weather.py`).
- Totals come in under the line in damp (−4.0 [−7.0, −1.0], 59 matches) and wet weather
  (−2.4 [−4.5, −0.3], 75); errors per 80 do not change with weather. Favourites covered
  47% of 854 dry-weather matches.

## Found on the way

- **The SL master's 2026 rows use eight team codes for other clubs than 2022–25 did**
  (C/CF, HF/HFC, LS/LH, WA/WFT swap). The ELO in the master continues by code across
  the season break, so those eight clubs started 2026 from the paired club's 2025 rating.
  xLadder issue, not changed here; `gigot_sl_named.season_maps` maps codes per season.
- The Queensland Cup's rating basis is competition-relative through 2024 and
  position-relative from 2025, depending on the data cut-off.
