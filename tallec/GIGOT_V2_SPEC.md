# GIGOT v2 — Team-List-Aware Match Prediction

GIGOT v2 extends the xLadder match model with player-level signals from TALLEC.
Five inputs per team per fixture:

| # | Input | Source | Status |
| --- | --- | --- | --- |
| 1 | Team Form | xLadder pipeline (rolling DTS form) | **DONE** — live in `xladder_pipeline.py` |
| 2 | Team Class | xLadder ELO (K=27, stored in masters) | **DONE** — live |
| 3 | Player Form | pre-match rolling mean of the last 5 composites | **DONE — measured, adds nothing** |
| 4 | Player Class | pre-match expanding mean of all prior composites | **DONE — measured, small real gain** |
| 5 | Contribution Rating | `gigot_contribution.py` | **DONE — real data**, per competition |

**Measured 2026-08-20** by `gigot_v2.py`. The headline: on the NRL the player layer
beats the shipped xLadder margin prediction, but by about 3%. That is enough to
justify wiring it in and not enough to build the pitch around.

## The result

953 NRL fixtures joined across 2022–2026, held out on the 140 played 2026 matches:

| | MAE | RMSE |
| --- | --- | --- |
| xLadder `Margin_Pred_v2` as shipped | 16.41 | 20.77 |
| same, recalibrated on the training seasons | 16.41 | 20.76 |
| recalibrated + player layer | **15.95** | 20.67 |

Paired bootstrap over 4,000 resamples of the held-out season:

- all three player features: **+0.46 MAE [95% CI +0.04, +0.89]**
- `d_class` alone: **+0.22 MAE [95% CI +0.01, +0.42]**

Both intervals exclude zero, so the gain is real. It is also small.

## What each input is actually worth

Effects on predicted margin, per standard deviation of the feature, from the fit on
2022–2025:

| Feature | Coefficient | 1 SD is worth |
| --- | --- | --- |
| `d_green` — difference in share of players with fewer than 3 prior matches | −32.7 | **−2.10 points** |
| `d_class` — difference in minutes-weighted pre-match player Class | +17.7 | **+1.43 points** |
| `d_form` — same for the 5-match Form window | +2.0 | +0.23 points |

**The useful signal is inexperience, not form.** A team's ELO already knows how good
the team is; what it does not know is that three of the seventeen have barely played.
Form adds nothing measurable on top of team-level form, which is the result to expect
once you notice that team form is itself an aggregate of the same players.

Sanity check on the raw data, no model involved: fixtures where one side held the
top-5% line-up advantage finished at an average margin of +11.2; the bottom 5%
averaged −8.2. The signal exists — but `d_class` correlates +0.275 with the margin and
only +0.074 with what the existing model gets *wrong*, which is why the incremental
gain is modest.

## Method, and the leakage discipline

1. Every player-match gets a composite score from the rating engine, standardized
   within its competition-season pool.
2. Pre-match features use only that player's **earlier** matches: Class is the
   expanding mean of prior composites, Form the mean of the last five. Match M's own
   performance never enters its own features.
3. Aggregated over the players who actually took the field, weighted by minutes.
4. Differenced between the two sides and joined to the master fixture.

The leakage claim is tested, not asserted: permuting the composite column across rows
and rebuilding drops the correlation between real and permuted `d_class` to +0.06
(NRL) and +0.01 (SL). The features carry information about players; destroying the
performance data destroys the features without touching the row keys.

**Known limits.** The standardization means and standard deviations are fitted over
the whole period, as the engine ships — a population descriptor rather than an
outcome, but not a strict walk-forward. And using the actual line-up is an *upper
bound* on what a team-list feed could deliver: on Friday you know the named 17, not
who finished the match.

## A second baseline, one we control

The comparison above rests on the master's stored prediction. Because that turned out
to be unusable for Super League (next section), the same test was run against a
baseline built here and evaluated walk-forward: each season predicted by a Ridge fitted
on the seasons before it, using only what is known before kick-off — the stored
pre-match ELO difference and home advantage.

| | Out-of-sample fixtures | Baseline MAE | + player layer | Gain |
| --- | --- | --- | --- | --- |
| NRL 2023–2026 | 752 | 14.66 | 14.39 | **+0.27 [+0.08, +0.48]** |
| Super League 2023–2025 | 494 | 13.53 | 13.37 | +0.16 [−0.33, +0.61] |

Both point estimates land in the same place — around a quarter of a point of margin
error. The NRL result clears significance on 752 fixtures; Super League, on 494,
points the same way without getting there. Taken together the effect looks real and
small, which is the same conclusion the stored-prediction test reached for the NRL.

## Super League's stored prediction cannot be used

The SL master's stored prediction has a training-set MAE of 15.69 and a held-out 2025
MAE of 7.63. Per season:

| Season | Matches | MAE |
| --- | --- | --- |
| 2022 | 167 | 14.93 |
| 2023 | 166 | 16.11 |
| 2024 | 167 | 16.04 |
| **2025** | 164 | **7.65** |

A model does not predict an unseen season twice as well as the data it was fitted on.
`Margin_Pred_v2` for SL 2025 is in-sample, so it is not a baseline. **This is an
xLadder issue, not a TALLEC one** — the SL app reports accuracy from that column, and
regenerating it out-of-sample matters for the app regardless of this project. For the
purposes of measuring the player layer it is no longer blocking: the walk-forward
baseline above does that job without touching the live model.

## Joining the two datasets

The masters and the player data name teams differently: the NRL master uses full club
names that match exactly (17 of 17), the SL master uses short codes. `team_map.py`
solves the codes from the data rather than from documentation — within a round, the
side that scored 26 in the master is the side that scored 26 in the player data, and
the two must be each other's opponent. Votes over every fixture, then a one-to-one
assignment.

That turned up a documentation error worth keeping in mind: **`HF` is Huddersfield and
`HFC` is Hull FC**, where the project notes say `HF = Hull FC`. Hand-coding the mapping
from the notes would have silently corrupted two clubs' results.

## Contribution Rating (input #5)

Per player per match: share of own team's output, blended across stat groups (attack
45%, defence 35%, points 20%), then percentile-ranked to 0–100 within his own
competition. Team totals are the sum of teammates' stats in the same match, so the
metric needs no external join.

Written **only** by `regenerate_full.py`, per competition; the formula lives in
`gigot_contribution.py`, which is import-only, and the weights in `config.json`.
`expected_contribution(con, team, competition, available)` returns a line-up's summed
rating and the delta against the full squad — the Player Availability signal.

## Next

1. **A team-list feed.** Everything above is measured on line-ups known after the
   fact. Whether the gain survives on Friday-morning team sheets is untested, and it
   is the only version a client can act on.
2. **Regenerate the SL master's stored predictions out-of-sample** — no longer needed
   to measure the player layer, but the Super League app reports its accuracy from that
   column, so the number it shows is currently not an out-of-sample number.
3. **Wire `d_green` in first.** It is the strongest per-SD signal, it is cheap to
   compute, and it is the one thing the team model demonstrably does not know.
4. ~~Feeder-competition ratings~~ — **done**. NSW Cup and Queensland Cup players are
   now rated in their own pools (496 and 492 players), so inputs 3–4 cover the players a
   club is actually choosing between, and both competitions are selectable in the app.
