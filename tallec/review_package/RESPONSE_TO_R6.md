# Response to the sixth review

Every finding of the sixth review was checked against the data or the code before
anything was changed, and every one held. This file says what became of each: what was
done, where to see it, and what is still open. Commits are in the project repository;
the files named are in this package.

Status: **fixed** — changed as the review asked; **reworded** — the claim was narrowed in
the status note and handover, nothing computed changed; **re-tested** — a new analysis
replaces the old one; **open** — not done, and why.

## The fourteen claims

| R6 | Review verdict | What was done | Status | Where |
| --- | --- | --- | --- | --- |
| C1 | Supported but overstated: no equivalence; 38 of 90 forecasts used the pooled line | Note says "no clear accuracy advantage … that does not show the two are equally good" and names the 38 pooled forecasts. The app now uses the same pooled fallback below 25 pairs | fixed | `docs/what_is_live.html`, `code/predict_translation.py` (`_line`), commit 59c87a3 |
| C2 | Overstated: unadjusted, re-examined many times | "Exploratory comparisons … not adjusted for making nine of them" | reworded | note |
| C3 | Overstated: "as accurate"; the app's own QLD→SL line was never tested | "Provisional default … whether it is better or worse than the model remains uncertain"; the untested QLD→SL line is stated | reworded | note |
| C4 | Wrong: MAEs do not decompose as 18 = 10 + 8; not a universal floor | The decomposition is gone. "Under a simplified repeat-measurement model the target rating varies by about 10 points … not a floor that no forecast can beat, and the rest of the error cannot be found by subtracting it" | fixed | note, `docs/HANDOVER.md` §2 |
| C5 | Overstated; the noise-removed comparison leaves the MSE difference unchanged | The noise-removed figure is no longer claimed; the 16+ subgroup is reported as "still no clear difference" | reworded | note |
| C6 | Overstated: split-half checks scale, not an irreducible floor | No longer presented as validation in the note | reworded | note |
| C7 | Not supported | "For reference, a straight line from a Super League player's own previous season … also misses by about 16 points" — reference, no diagnosis | reworded | note |
| C8 | Overstated: +0.43 of the trend's +0.50 came from a missingness indicator | Re-tested with that indicator (`has_history`) as its own baseline; see below | re-tested | `code/retest_r6.py`, commit 912b3ca |
| C9 | Overstated: 99.98%, not 100%; NRL checked only on 2020 | "99.98% … 84% on the rest. It has not been checked on NRL seasons after 2020" | reworded | note |
| C10 | Not supported | "Dated squad and intended-role information is the most promising next test. We have not established that it is necessary or sufficient" | reworded | note |
| C11 | Supported | Precision added: about 30 more moves narrow the uncertainty by roughly 13% | reworded | note |
| C12 | Overstated: pooled, fixed raw ranking, no single-season shortlist | "Pooled over 2023–25 … against a fixed ranking by minutes played … how it does on a single season's shortlist has not been tested" | reworded | note |
| C13 | Overstated: retrospective, actual participants and minutes | "A retrospective test using the players who actually took part, and their actual minutes … gains using only what is known before kick-off … have not been tested yet" | reworded | note |
| C14 | Supported | Unchanged | — | note |

## The ranked problems

| R6 problem | What was done | Status |
| --- | --- | --- |
| P1 the 10-point floor does not leave an 8-point gap | As C4. The measurement-error model the review pointed to is listed as a next step, not built | fixed / open |
| P1 the trend's half point is mostly its missingness | `retest_r6.py` scores every candidate against line + `has_history`. Into Super League the indicator itself gains +0.85 [+0.20, +1.54]; trend on top of it +0.09 [−0.41, +0.59] | re-tested |
| P1 38 of 90 forecasts used the pooled line | Documented in the note; the app's fallback now matches the backtest | fixed |
| P1 the new club was the modal club of the target season | The re-test uses the first club he is seen playing for; still read from the target season, labelled retrospective. Dated signing records are now asked for | partly fixed |
| P2 residual correction is not a joint fit | The re-test fits source rating, direction and candidate together (OLS on line terms, ridge 1.0 on the standardised candidate and its flag) | re-tested |
| P2 xLadder absent from the first origin's training data (0 of 372) | xLadder left out of the re-test | fixed |
| P2 "90 moves cannot show less than a point"; "the negative results are noise fitting" | Both sentences are gone | fixed |
| P2 role proxies are not the planned role | Note: "Role is measured from who played, not from what the club planned" | reworded |
| P2 starts rule not exact, unvalidated on NRL 2021–26 | As C9. A hand-checked NRL sample, stratified by season and position, is not done | reworded / open |
| P2 the 95% band | The app calls it a "historical range" and says it is not a calibrated 95% promise. A proper prediction interval (t multiplier, leverage) is not implemented | reworded / open |
| P2 the app claimed age, minutes and matches | Both captions now state the model's actual inputs | fixed |
| P2 arrival baseline should learn the minutes direction | Listed in the handover as the next arrival test | open |
| P2 C12/C13 cannot be reproduced from the package | Not exported this round | open |

## Scope and demo

| R6 item | What was done | Status |
| --- | --- | --- |
| Form over 3 and 5 matches; Divergence as a percentage; several positional ratings | Listed in the note as "three scope points to agree" with the client | open, for the client |
| Form's centre is Class's, so Form's median need not be 50; Class is recency-weighted | Not addressed in the note | open |
| Provenance: the model manifest records a dirty tree; not every input is fingerprinted | Not addressed | open |
| Demo deliverable: a card of comparable historical moves | Built: `predict_translation.comparables` and the translation page; see the brief | fixed |
| The brief said "90 moves into Super League a year" | Wrong: about 30 a year, 90 over three origins. Corrected in this round's brief | fixed |

## Found while fixing, not in the review

- **The translation page read the v2 ladder.** It offered eight directions — nothing but
  Super League from the NRL — and printed the measured ladder on the scale retired in
  September. It now reads the v3 next-season ladder. Its caption claimed every pair of
  directions has opposite signs; five of six do, and the sentence is now computed (commit
  59c87a3).
- **A push broke the live app.** Streamlit Cloud swapped the files under the running
  process, the new `bosc_app.py` imported `comparables`, and the `predict_translation`
  module in memory was the old one. The app now reloads its own modules when their files
  change (commit e01a413). The database file had the same problem one round earlier.
