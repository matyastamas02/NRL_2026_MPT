# TALLEC / BOSC — package for the sixth independent review

This round has a narrow job. The project now makes a specific set of claims to its
client in a status note (`docs/what_is_live.html`). They are listed below as C1–C14,
each with the code that computes it and the section of `reproduce.py` that recomputes it
from `data/` alone. The review is of those claims, judged against what the project is
for. Read this file first; it saves re-deriving context from sixty files.

---

## 1. What the project is for

From the client's scope of work (quoted, not paraphrased, where it matters):

- **A prototype.** "Build a working prototype of TALLEC … The priority is to prove the
  concept rather than build a finished commercial product." It supports two separate
  applications: **BOSC**, "recruitment and player intelligence platform (for
  demonstration to Leeds Rhinos)", and **GIGOT v2**, an internal match model "completely
  separate from the Leeds project".
- **BOSC, the part the claims are about:**
  - "benchmarking by competition, season and position, based on score of 0-100 when 50
    is average";
  - "Class Rating (since start of database), Form Rating (3 & 5 game rolling averages)
    and Divergence Rating (How far is Form from Class as % over or under)";
  - "Develop Competition Translation models to estimate player performance across
    competitions. **This is the key modelling task for Leeds as they want to estimate
    how a player in NRL or NSW Cup will go in SL.**";
  - "Support multiple positional ratings for the same player (e.g. centre and wing)";
  - a simple Streamlit prototype with search, profile and ratings.
- **GIGOT v2:** team form and class plus player form and class, a Contribution Rating
  ("Player stats as a % of Team Stats, so we can track Expected Contribution based on
  Team List"), back-tested, with winner probabilities, margins and confidence.
- **Technical principles:** raw data stored permanently, derived metrics reproducible
  from raw, configurable rating formulae, every calculation version-controlled.

Who reads the claims: the client (Mike), who will demonstrate BOSC to Leeds Rhinos, a
Super League club that recruits into Super League from the NRL, the NSW Cup and the
Queensland Cup. The note is decision support for that demo, not a research paper.

## 2. The system in brief

- **Ratings** (`player_rating_engine.py`). A per-match composite of rate statistics,
  z-scored within competition × season × position group, averaged per player, shrunk by
  B = τ²/(τ² + σ²/n), published as a **peer score**: 100·Φ((z − group median)/τ), so 50
  is the median of his position group in that competition and season. Not a percentile,
  and not comparable across competitions without translation.
- **Translation.** Three answers to "what will he do in competition T next season":
  - the **ladder**, the mean shift observed for movers in each direction
    (`translation_ladder_v3`);
  - the **conditional model**, a Ridge regression on source rating, position group and
    direction (`fit_translation_v3.py`);
  - the **straight line** `target ~ source`, fitted per direction on the same pairs,
    which is the bar the model has to clear (`rolling_backtest.straight_lines`).
  **Since this round, the app forecasts every move into Super League with that line**
  (`predict_translation.LINE_TARGETS`); other directions still use the model.
- **Arrival** (`arrival_model.py`): of the players at a level, who turns up in another
  competition next season.
- **GIGOT player layer** (`gigot_v2.py`): pre-match player ratings aggregated over the
  line-up, added to the match model.

## 3. What changed since the fifth review

Methodology:

1. Every report now treats the client's direction as **into Super League**.
2. The conditional model was found no better than the per-direction line into Super
   League, so **the app now uses the line there**.
3. **Noise floor** (`noise_floor.py`): how much of the 18-point error is measurement
   noise in a single season's target rating.
4. **Less noisy targets**: the model against the line on players with 16+/20+ target
   matches, and with the simulated noise removed from the squared errors.
5. **Team context, expected role and trend** (`team_role_trend.py`), each tested as a
   correction to the line, including **starts rebuilt from interchange counts** for NRL
   seasons whose export has no match-sheet positions.
6. The status note was rewritten. In particular, it no longer leads with the model
   beating two baselines nobody would use (carrying a rating across unchanged, and
   predicting 50 for everyone); it states the comparison with the line.

Engineering only, no effect on any figure: the deployed app reads a copy of the database
without `player_match_raw`, and a cache bug that kept a replaced database file open was
fixed.

## 4. The claims under review

C1–C9 can be recomputed from `data/` with `python reproduce.py`. C10–C11 are
inferences from them. C12–C13 need the database or the xLadder match masters, which are
not in the package; their code and reports are.

| | Claim, as the note states it | Computed in | Recompute |
| --- | --- | --- | --- |
| C1 | For the 90 moves into Super League the conditional model is no more accurate than a straight line fitted to earlier moves into Super League: both miss by about 18 points (17.98 and 17.72 MAE; difference −1.56 to +1.06) | `rolling_backtest.py` | §C1 |
| C2 | Of the nine directions with enough moves, the model is clearly closer than the line in two, NRL→QLD (+1.96, +0.75 to +3.29) and NSW→QLD (+1.47, +0.58 to +2.42); clearly further off in three, NRL→NSW, QLD→NSW and QLD→SL; indistinguishable in four | `rolling_backtest.beats_the_simple_thing` | §C2 |
| C3 | The app forecasts moves into Super League with one line per source competition; it is as accurate as the model and is explained by two numbers (from the NRL, 0.37 × rating + 43) | `predict_translation.py` | §C3 |
| C4 | A season's rating carries measurement noise that puts a floor of about 10 points under any forecast, so roughly 8 of the 18 points are real differences between a player's record and what he then did | `noise_floor.py` | §C4 |
| C5 | The verdict does not depend on that noise: on players with 16+ Super League matches, or with the noise removed, model and line remain indistinguishable | `noise_floor.py` | §C5 |
| C6 | The noise model, σ²/n with σ² from a within-season variance decomposition, is about the right size (split-half ratio 0.92) | `noise_floor.py` | §C6 |
| C7 | Super League players who stayed are forecast from their own previous season at about 16 points, so even a player's own record in the league leaves most of those 8 points unexplained: "what changes between two seasons is not in the ratings" | `noise_floor.py` | §C7 |
| C8 | None of trend, starts share, old/new team points margin, old/new team xLadder expected points, or two proxies for the role at the new club clearly improves the line; the closest is the trend into Super League, +0.50 (−0.03 to +1.03) | `team_role_trend.py` | §C8 |
| C9 | Starts for NRL players are rebuilt from interchange records; the rule is exact wherever it decides, about three rows in four, and right 84% of the time on the rest | `team_role_trend.py` | §C9 |
| C10 | A better translation needs new information, above all the role a player is expected to have at his new club; the versions of it that can be built from appearances did not capture it | inference from C4–C8 | — |
| C11 | The 2026 season adds roughly one more year of moves into Super League, about 30, which narrows the uncertainty only moderately | 90 moves over three origins | — |
| C12 | The arrival model beats minutes played alone clearly in four of twelve directions; into Super League by +0.168 AUC (+0.093 to +0.247), with 31% of actual arrivals in the top tenth of its list | `arrival_model.py`, `docs/ARRIVAL_REPORT.md` | not exported |
| C13 | The player layer cuts match-model margin error by 0.24 points in the NRL (+0.05 to +0.44, 752 fixtures) and 1.13 in Super League (+0.62 to +1.63, 497), an upper bound because it uses the line-up that actually played | `gigot_v2.py` | needs masters |
| C14 | Good enough for an internal or beta demo; not yet a validated recruitment ranking | judgement | — |

The printed output each analysis produced for this package is in `results/`.

## 5. What is in the package

```
00_START_HERE.md      this file
PROMPT.md             the review brief
reproduce.py          recomputes C1-C9 from data/ alone - run this first
MANIFEST.md           every file with row counts and sha256

code/                 the files that carry the methodology
code_context/         ingest, reporting, the app - read only if a trail leads there
docs/                 reports, specs, the handover, and what_is_live.html (the claims)
tests/                the current test suite
results/              printed output of noise_floor.py and team_role_trend.py
data/                 everything needed to recompute without the database
```

| file | use it to |
| --- | --- |
| `data/rolling_eval_{2023,2024,2025}.csv` | C1, C2: one row per move landing in that season, with the source rating as known beforehand, every prediction (`model`, `line_direction`, `projected`, …), the outcome `class_target` and the cohort |
| `data/rolling_train_{2023,2024,2025}.csv` | refit the model and the lines on exactly what they were trained on |
| `data/translation_pairs_v3.csv` | C3: the pairs the shipped model and the app's lines are fitted on |
| `data/noise_floor/movers_floor.csv` | C4, C5: the 90 moves with each target's n, B, σ², τ², centre and posterior level |
| `data/noise_floor/split_half.csv` | C6: odd/even half-season means for every SL player-season with 6+ matches |
| `data/noise_floor/stayers.csv` | C7: SL players who stayed, with their walk-forward line forecast |
| `data/team_role_trend/train_{origin}.csv`, `eval.csv` | C8: the features for every training pair and evaluated move, the line, and each correction's forecast |
| `data/team_role_trend/team_margin.csv`, `xladder_eppg.csv` | the two team-strength measures by competition, season and team |
| `data/team_role_trend/starts_rule_check.csv.gz` | C9: every match-sheet row with the interchange counts, the rule's call and the sheet's truth |
| `data/player_match_stats.csv.gz`, `data/player_season_ratings.csv`, `data/players.csv` … | rebuild ratings from the feed |

The data are Stats Perform rows supplied under a client agreement, including dates of
birth. They are here for this review only.

## 6. Known weak points, stated up front

New this round:

- **Samples.** 90 moves into Super League over three origins; 441 entry moves overall
  from 362 players. With 90, an improvement smaller than about one point cannot be shown
  either way.
- **Two different constructions.** The model and lines are fitted on
  `translation_pairs_v3`-style pairs (`fit_translation_v3.build_pairs`); they are scored
  on the explicit entry cohort (`transition_events.py`). The two do not select players
  the same way.
- **The line's band** in the app is 1.96 × its in-sample residual SD on 32–107 pairs per
  direction, not an out-of-sample error.
- **Noise floor assumptions.** σ² comes from a one-way variance decomposition within the
  target season, so within-season form swings count as noise and the "true level" is a
  season average. The noise is simulated as normal on the composite scale and pushed
  through the engine's nonlinear map, with the true level placed at the shrunk posterior
  mean (at the unshrunk mean instead it reads 9.5). The noise-removed comparison is in
  RMSE, not MAE, and assumes the noise is independent of the predictions.
- **The stayers reference is not a ceiling.** It is a walk-forward straight line on the
  previous season, on a different cohort. The project owner has already pointed out that
  it lacks information about how a player develops between seasons; judge whether C7
  still says too much.
- **The feature tests.** Each correction is a linear model of the line's residuals with
  coefficients shared across all directions, fitted on the training pairs of each
  origin, with standardised values, a missing flag per feature and an intercept. Twenty
  comparisons are reported. Role is proxied from appearances: (a) the minutes-weighted
  rating of the new club's incumbents in his position group the season before, (b) the
  share of their minutes played by men who do not appear for the club in the target
  season, which is **look-ahead**. The new club itself is read from the target season,
  on the argument that it is known at signing. xLadder figures exist only for the NRL and
  Super League from 2022 (33–50% of moves), and their stored win probabilities may be
  in-sample for the seasons the xLadder model was trained on. Team margins are derived
  from player points; against the xLadder masters (not in the package) they agree at
  r = 0.997 for the NRL and within 1.1 points per game for Super League 2022–25.
- **The starts rule** is validated on competitions with match sheets and then applied
  to NRL 2021–26, where there is none.

Carried over, see `docs/HANDOVER.md`: selection into the ladder; cumulative source
against season-only target; a constant-ability variance decomposition; no squad,
contract or registration data; 2026 is exploratory and 2027 the first confirmatory
holdout; the fit is frozen at 2025.

## 7. Conventions

- **Freeze.** `config.json → evaluation.freeze_season = 2025`. Nothing is fitted past it.
- **Position groups.** Fullback, Winger, Centre, Halves, Hooker, Middles (prop + lock),
  Edge (second row), Bench. `sp_schema.POSITION_GROUP` is the only mapping.
- **Two outcomes.** `class_target` is the shrunk season rating the app would publish;
  `class_target_raw` the same season unshrunk on the same scale.
- **Sign of a comparison.** "A − B" in absolute error, so a positive number means B was
  closer. In `reproduce.py`, `boot(d, "line_direction", "model")` positive = model closer.
