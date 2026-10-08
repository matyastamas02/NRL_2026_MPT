# TALLEC / BOSC — package for the seventh independent review

The sixth review checked fourteen claims the project made to its client and found ten
of them overstated, unsupported or wrong. Everything it found was checked and acted on.
This round asks for three narrower things:

1. **Did the response fix what the sixth review found?** `RESPONSE_TO_R6.md` lists every
   finding with what was done and where.
2. **Is the 2026 test ready to freeze?** `code/retest_r6.py` is the specification the
   project proposes to run unchanged on the 2026 season. Once 2026 data arrives, any
   change to it is tuning, so this is the last point at which it can be corrected
   cleanly.
3. **Is the comparison card a sound demo deliverable?** It is what the sixth review
   proposed showing a club instead of, or beside, a point forecast, and it is now built.

The status note's current claims (N1–N13 below) are also open to the same four-level
verdict as last round. Read this file first.

---

## 1. What the project is for

From the client's scope of work, quoted where it matters:

- **A prototype.** "Build a working prototype of TALLEC … The priority is to prove the
  concept rather than build a finished commercial product." Two separate applications:
  **BOSC**, "recruitment and player intelligence platform (for demonstration to Leeds
  Rhinos)", and **GIGOT v2**, an internal match model "completely separate from the Leeds
  project".
- **BOSC:** "benchmarking by competition, season and position, based on score of 0-100
  when 50 is average"; "Class Rating (since start of database), Form Rating (3 & 5 game
  rolling averages) and Divergence Rating (How far is Form from Class as % over or
  under)"; "Develop Competition Translation models to estimate player performance across
  competitions. **This is the key modelling task for Leeds as they want to estimate how
  a player in NRL or NSW Cup will go in SL.**"; "Support multiple positional ratings for
  the same player"; a simple Streamlit prototype.
- **Technical principles:** raw data stored permanently, derived metrics reproducible,
  configurable formulae, every calculation version-controlled.

The reader of the status note is the client, who will demonstrate BOSC to Leeds Rhinos, a
Super League club recruiting from the NRL, the NSW Cup and the Queensland Cup.

## 2. The system, as it stands now

- **Ratings** (`player_rating_engine.py`): a per-match composite of rate statistics,
  shrunk by B = τ²/(τ² + σ²/n), published as a **peer score** 100·Φ((z − group
  median)/τ). 50 is the median of his position group in that competition and season.
- **Translation into Super League** (`predict_translation.py`): since last round, a
  **straight line** `target ~ source` per source competition, fitted on the next-season
  pairs in `translation_pairs_v3`, as a provisional default. Below 25 pairs a direction
  uses the line fitted on every next-season pair, the rule the backtest scored. Other
  directions and horizons use the conditional model (Ridge on source rating, direction and
  position group).
- **The comparison card** (`predict_translation.comparables`, shown on the app's
  translation page): earlier movers in the same direction, next season, source rating
  within 7.5 points; the same position group if that leaves at least eight, otherwise any
  position, and the card says which; the window doubled once if there are still fewer than
  eight; otherwise "not enough data". The player himself is excluded. It shows the count,
  the median, the middle half, the 10–90% range and the list. Every pair is a mover who
  played at least three matches in the new competition, and the card says so.
- **Arrival** (`arrival_model.py`) and the **GIGOT player layer** (`gigot_v2.py`) are
  unchanged since last round; only their wording in the note changed.

## 3. What changed since the sixth review

All of it is in `RESPONSE_TO_R6.md`. In brief: the note was reworded throughout; the
app's captions now state the model's actual inputs and call the band a historical range;
the app's line fallback matches the backtest's; the translation page reads the current
ladder (it was reading a superseded one); the candidates were re-tested on one cohort with
a history baseline and a joint fit (`retest_r6.py`); the comparison card was built; and
the app reloads its own modules when a push replaces them.

## 4. The status note's current claims

| | Claim, as the note now states it | Computed in | Recompute |
| --- | --- | --- | --- |
| N1 | On 90 moves into Super League no clear accuracy advantage for the conditional model over a straight line (17.98 and 17.72 MAE; difference −1.56 to +1.06); "that does not show the two are equally good"; 38 of the 90 used the pooled line; the app's own Queensland Cup line has not been tested | `rolling_backtest.py` | §C1, §R2 |
| N2 | In exploratory, unadjusted comparisons across nine directions, two favour the model (NRL→QLD +1.96, NSW→QLD +1.47), three favour the line, four cannot be told apart | `rolling_backtest.py` | §C2 |
| N3 | Under a simplified repeat-measurement model the target rating varies by about 10 points; "not a floor that no forecast can beat"; among players with 16+ SL matches still no clear difference; stayers' own-previous-season line misses by about 16, as a reference | `noise_floor.py` | §C4, §C5, §C7 |
| N4 | Re-test (specified before running): whether the player had a rated season in the source the year before improves forecasts into SL, +0.85 (+0.20 to +1.54), in each of three seasons; nothing clear on top of it into SL; across all moves the strength of the new club's incumbents in his position helps, +0.53 (+0.12 to +0.92); the re-test's line is weaker than the shipped one (18.77 against 17.72) | `retest_r6.py` | §R1 |
| N5 | Starts rebuilt from interchange records agree with match sheets 99.98% where the rule decides, 84% on the rest; not checked on NRL after 2020 | `team_role_trend.py` | §C9 |
| N6 | The app uses the line into SL as a provisional default, one per source competition, with the pooled fallback; from the NRL, 0.37 × rating + 43 | `predict_translation.py` | §C3, §R2 |
| N7 | Every forecast comes with the players who made the same move before, showing how widely outcomes spread | `predict_translation.comparables` | §R3 |
| N8 | Dated squad and intended-role information is the most promising next test; not established as necessary or sufficient | inference | — |
| N9 | The 2026 season adds about 30 moves into SL, narrowing the uncertainty by roughly 13% | √(90/120) | — |
| N10 | Arrival: against a fixed minutes ranking, clearly better in four of twelve directions; into SL, pooled 2023–25, +0.168 AUC (+0.093 to +0.247); top tenth holds 28 of the 90 rated arrivals; single-season shortlist untested | `arrival_model.py`, `docs/ARRIVAL_REPORT.md` | not exported |
| N11 | GIGOT: a retrospective test with actual participants and minutes improved an ELO+home baseline by 0.24 (NRL, 752 fixtures) and 1.13 (SL, 497); pre-kick-off gains untested | `gigot_v2.py` | needs masters |
| N12 | Good enough for an internal or beta demo; not a validated recruitment ranking | judgement | — |
| N13 | On the translation page: five of six direction pairs in the next-season ladder carry opposite signs; Queensland Cup ↔ Super League does not | `bosc_app.py`, ladder | §R4 |

## 5. The 2026 test as proposed

`code/retest_r6.py`, unchanged, run with `LANDING` extended to 2026 and `ORIGINS` to
include 2026:

- **Cohort.** The explicit entry cohort (`transition_events.py` through
  `rolling_backtest.moves_into`), each landing season built from what was known before
  it. Fit on cohort moves landing before the forecast season, score on those landing in
  it.
- **Baseline.** A straight line, own line per direction from 25 training moves, else the
  line over all training moves, plus `has_history` (rated in the same source competition
  the season before, n ≥ 3).
- **Hypotheses.** H1, moves into Super League: line + `has_history` beats the line.
  H2, all moves: adding the strength of the new club's incumbents in his position group
  (minutes-weighted rating the season before) beats line + `has_history`.
- **Fit.** OLS on the line terms; ridge α = 1.0 on each standardised candidate and its
  missing flag. The new club is the first club he is seen playing for in the target
  season.
- **Measure.** Paired difference in MAE, bootstrap over players, 4,000 draws.

What is **not** yet fixed, and is part of what this round should advise on: what counts
as success (a threshold, an interval rule, or both), whether 2026 alone or 2026 pooled
with 2023–25 is the test, what to do if the 2026 cohort into Super League is much smaller
than 30, and how NRL 2026 positions (expected from the client with match sheets) enter.

## 6. What is in the package

```
00_START_HERE.md      this file
RESPONSE_TO_R6.md     every sixth-review finding, what was done, where, what is open
PROMPT.md             the review brief
reproduce.py          recomputes C1-C9 and R1-R4 from data/ alone - run this first
MANIFEST.md           every file with row counts and sha256

code/                 the methodology, including retest_r6.py
code_context/         ingest, reporting, the app
docs/                 reports, the handover, what_is_live.html (the status note)
tests/                the current test suite
results/              printed output of noise_floor.py, team_role_trend.py, retest_r6.py
data/                 everything needed to recompute without the database
```

New this round: `data/retest_r6/retest_cohort.csv` (the cohort with every candidate,
names and dates of birth removed) and `retest_scored.csv` (the forecasts of every
specification), and `results/retest_r6.txt`. The rest is as last round, and the C1–C9
sections of `reproduce.py` still recompute the figures N1–N6 rest on.

The data are Stats Perform rows supplied under a client agreement; other files still
carry dates of birth. They are here for this review only.

## 7. Known weak points, stated up front

- **The history effect was found on the same 90 moves it is reported on**, in an analysis
  run because the sixth review pointed at it, with fourteen comparisons in the re-test.
  It is a hypothesis for 2026, not a finding.
- **The re-test's line is weaker than the shipped line** (18.77 against 17.72 into Super
  League), because the entry cohort it trains on is smaller than the pair set. Gains in
  the re-test are against that weaker line.
- **The app still trains its line on the pair set**, not the cohort, so the object being
  shipped and the object being re-tested differ.
- **The pooled fallback** mixes every direction into one line, and the app's own
  Queensland Cup line was never tested at 25+ pairs.
- **Ridge α = 1.0** was fixed, not tuned. `has_history` depends on the n ≥ 3 rating rule.
- **The comparison card** shows only movers with three or more matches in the new
  competition, so it cannot show how often a move failed; the 7.5-point window and the
  eight-player minimum were chosen, not derived; at the edges of the rating range the
  window doubles and the comparables are less similar.
- Carried over: small samples, three origins, selection into who moves, no squad,
  contract or registration data, 2026 exploratory and 2027 confirmatory, fit frozen at
  2025.

## 8. Conventions

As last round: the freeze at 2025; eight position groups with `sp_schema.POSITION_GROUP`
the only mapping; `class_target` the published (shrunk) season rating; a comparison "A −
B" in absolute error, so positive means B was closer.
