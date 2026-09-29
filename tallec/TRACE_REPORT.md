# From the feeder competitions into the NRL

> **Post-hoc evaluation, not a holdout.** Nothing here is fitted on 2026 — the models stop at 2025 — but the current model was *designed* after an external review had already seen these results, and a season you have looked at cannot test the model you built afterwards. The project's one clean out-of-sample figure is sealed in `v1_holdout_record.json`, with the hashes of the artefacts that produced it. The first genuinely clean confirmation is 2027.

Every player below was rated in the NSW Cup or the Queensland Cup in 2025 over at least 5 matches, and has since played in the NRL in 2026. No 2026 match contributed to any figure in the `projected` column — the freeze at 2025 sees to that — so it is a forecast in the mechanical sense. What it is not is a test of a model that has had sight of the answer.

134 players in the cohort, 79 of them with at least 5 NRL matches in 2026 — the 2026 season is twenty rounds old, so a thin NRL sample is the main source of noise in `actual` and the tables below are cut by it.


## Who these players actually are

This is not a group of prospects making the step up, and reading it as one is how an earlier version of this report reached a conclusion it could not support. Split by what each man already was:

| cohort | n | meaning |
| --- | --- | --- |
| first | 10 | first season in the competition |
| returning | 9 | returning after a season away |
| continuing | 60 | played there the season before as well |


## How good was the projection

Graded on the 79 players with 5+ NRL matches in 2026. `bias` is the average of actual minus predicted, so a positive number means the projection was too low. Errors are in points on the 0-100 rating scale.

`no translation` is the baseline a club would otherwise use — take his feeder rating at face value. `no information` assumes every selected player is average. A translation earns its place only by beating the first of those.


**First season in the competition** — 10 players

| predictor | n | mae | bias | r |
| --- | --- | --- | --- | --- |
| the projection (ladder) | 10 | 11.22 | 2.73 | 0.80 |
| the Ridge model | 10 | 17.76 | -0.66 | 0.74 |
| no translation (his feeder rating) | 10 | 14.86 | -8.21 | 0.80 |
| no information (competition average, 50) | 10 | 22.81 | -2.77 |  |

*10 players. Indicative only — one outlier moves these figures by more than the differences between them.*


**Returning after a season away** — 9 players

| predictor | n | mae | bias | r |
| --- | --- | --- | --- | --- |
| the projection (ladder) | 9 | 17.31 | -3.49 | 0.67 |
| the Ridge model | 9 | 16.37 | -7.90 | 0.71 |
| no translation (his feeder rating) | 9 | 20.73 | -13.47 | 0.69 |
| no information (competition average, 50) | 9 | 20.38 | -8.99 |  |

*9 players. Indicative only — one outlier moves these figures by more than the differences between them.*


**Played there the season before as well** — 60 players

| predictor | n | mae | bias | r |
| --- | --- | --- | --- | --- |
| the projection (ladder) | 60 | 23.56 | -2.79 | 0.37 |
| the Ridge model | 60 | 22.41 | 2.11 | 0.33 |
| no translation (his feeder rating) | 60 | 25.69 | -12.64 | 0.38 |
| no information (competition average, 50) | 60 | 24.39 | 4.68 |  |


**All of them together**, dominated by the third group and so read last

| predictor | n | mae | bias | r |
| --- | --- | --- | --- | --- |
| the projection (ladder) | 79 | 21.29 | -2.17 | 0.47 |
| the Ridge model | 79 | 21.13 | 0.62 | 0.43 |
| no translation (his feeder rating) | 79 | 23.75 | -12.18 | 0.48 |
| no information (competition average, 50) | 79 | 23.73 | 2.18 |  |

**What the split shows.** The group Leeds actually asks about — men with no top-grade record, where the feeder rating is the only evidence there is — is the smallest one here, 10 players. On them the translation beats taking the rating at face value: 11.22 against 14.86 points of error. For players who already have an NRL record the comparison is less interesting either way, because a recruiter would use that record rather than a translation out of reserve grade.


10 players is not a verdict. It is the right question, and the rolling-origin backtest is where it gets a sample large enough to answer.


An earlier version of this report turned these biases into a recommended correction of a specific size. That is withdrawn: it was measured on the mixed cohort, and fitting anything to 2026 would consume the only clean holdout the project has. The correction is a hypothesis for the historical backtest to test, not a result.


### By how much of him we had seen

| nrl_matches | n | mae | bias | r |
| --- | --- | --- | --- | --- |
| 5-9 | 38 | 17.88 | 0.29 | 0.54 |
| 10-14 | 30 | 26.52 | -3.68 | 0.34 |
| 15+ | 11 | 18.77 | -6.53 | 0.58 |

A player with a handful of NRL matches has a noisy `actual`, so the error should fall as the sample grows. If it does not, the projection is wrong rather than the measurement being noisy.


## Does it pick the right men?

Exact points matter less to a recruiter than order: if the list is sorted by projection, do the ones near the top turn out better than the ones near the bottom? Spearman rank correlation between projection and outcome, and the average outcome by projected quartile.


Spearman rank correlation: **+0.46** over 79 players.

| quartile | n | mean_projected | mean_actual |
| --- | --- | --- | --- |
| lowest | 20 | 19.06 | 33.58 |
| 3rd | 20 | 44.54 | 48.38 |
| 2nd | 19 | 69.99 | 60.48 |
| highest | 20 | 84.57 | 66.68 |

## The stories

Ten men the projection rated highest, in order, with what has happened since.

- **Te Maire Martin** — a 100 in the NSW Cup in 2025 over 6 matches. The ladder projected that profile as 91 in the NRL, the model as 63. Across 8 NRL matches in 2026 he has rated 82.
- **Sam Healey** — a 99 in the NSW Cup in 2025 over 15 matches. The ladder projected that profile as 90 in the NRL, the model as 59. Across 8 NRL matches in 2026 he has rated 93.
- **Joey Walsh** — a 99 in the NSW Cup in 2025 over 15 matches. The ladder projected that profile as 90 in the NRL, the model as 62. Across 5 NRL matches in 2026 he has rated 57.
- **Jonathan Sua** — a 98 in the NSW Cup in 2025 over 21 matches. The ladder projected that profile as 89 in the NRL, the model as 63. Across 5 NRL matches in 2026 he has rated 55.
- **Daine Laurie** — a 98 in the NSW Cup in 2025 over 11 matches. The ladder projected that profile as 89 in the NRL, the model as 67. Across 10 NRL matches in 2026 he has rated 43.
- **Matthew Lodge** — a 97 in the NSW Cup in 2025 over 5 matches. The ladder projected that profile as 88 in the NRL, the model as 64. Across 11 NRL matches in 2026 he has rated 90.
- **Clayton Faulalo** — a 97 in the NSW Cup in 2025 over 13 matches. The ladder projected that profile as 88 in the NRL, the model as 66. Across 10 NRL matches in 2026 he has rated 72.
- **Trai Fuller** — a 99 in the QLD Cup in 2025 over 6 matches. The ladder projected that profile as 85 in the NRL, the model as 63. Across 6 NRL matches in 2026 he has rated 82.
- **Paul Alamoti** — a 93 in the NSW Cup in 2025 over 6 matches. The ladder projected that profile as 84 in the NRL, the model as 61. Across 17 NRL matches in 2026 he has rated 61.
- **Oryn Keeley** — a 98 in the QLD Cup in 2025 over 5 matches. The ladder projected that profile as 84 in the NRL, the model as 59. Across 8 NRL matches in 2026 he has rated 83.

And where it was furthest wrong in each direction.


**Under-projected**

- **Royce Hunt** — a 18 in the NSW Cup in 2025 over 7 matches. The ladder projected that profile as 9 in the NRL, the model as 38. Across 8 NRL matches in 2026 he has rated 96.
- **Luke Laulilii** — a 24 in the NSW Cup in 2025 over 10 matches. The ladder projected that profile as 15 in the NRL, the model as 43. Across 8 NRL matches in 2026 he has rated 74.
- **Harrison Graham** — a 56 in the QLD Cup in 2025 over 11 matches. The ladder projected that profile as 42 in the NRL, the model as 41. Across 13 NRL matches in 2026 he has rated 98.
- **Bunty Afoa** — a 29 in the NSW Cup in 2025 over 16 matches. The ladder projected that profile as 20 in the NRL, the model as 41. Across 6 NRL matches in 2026 he has rated 74.

**Over-projected**

- **Ali Leiataua** — a 82 in the NSW Cup in 2025 over 6 matches. The ladder projected that profile as 73 in the NRL, the model as 58. Across 14 NRL matches in 2026 he has rated 21.
- **Samuel Stonestreet** — a 81 in the NSW Cup in 2025 over 8 matches. The ladder projected that profile as 73 in the NRL, the model as 58. Across 12 NRL matches in 2026 he has rated 22.
- **Manaia Waitere** — a 81 in the NSW Cup in 2025 over 24 matches. The ladder projected that profile as 72 in the NRL, the model as 57. Across 6 NRL matches in 2026 he has rated 23.
- **Mawene Hiroti** — a 82 in the NSW Cup in 2025 over 6 matches. The ladder projected that profile as 74 in the NRL, the model as 58. Across 7 NRL matches in 2026 he has rated 24.


## The full cohort

| player | source | grp | feeder_games | rated | projected | model | nrl_games | actual | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Te Maire Martin | NSW | Halves | 6 | 99.7 | 91.0 | 62.8 | 8 | 82.2 | -8.7 |
| Sam Healey | NSW | Hooker | 15 | 99.0 | 90.2 | 58.6 | 8 | 92.6 | 2.3 |
| Joey Walsh | NSW | Halves | 15 | 98.6 | 89.8 | 62.4 | 5 | 56.8 | -33.0 |
| Jonathan Sua | NSW | Winger | 21 | 97.7 | 88.9 | 63.4 | 5 | 54.7 | -34.2 |
| Daine Laurie | NSW | Fullback | 11 | 97.6 | 88.8 | 66.7 | 10 | 42.9 | -45.9 |
| Matthew Lodge | NSW | Middles | 5 | 97.0 | 88.2 | 63.5 | 11 | 90.2 | 2.0 |
| Clayton Faulalo | NSW | Fullback | 13 | 96.8 | 88.0 | 66.4 | 10 | 72.2 | -15.8 |
| Trai Fuller | QLD | Fullback | 6 | 99.2 | 85.0 | 63.3 | 6 | 82.3 | -2.7 |
| Paul Alamoti | NSW | Centre | 6 | 92.9 | 84.1 | 61.2 | 17 | 60.9 | -23.2 |
| Oryn Keeley | QLD | Edge | 5 | 98.1 | 83.9 | 59.4 | 8 | 82.9 | -1.1 |
| Taine Tuaupiki | NSW | Fullback | 9 | 92.2 | 83.5 | 65.0 | 15 | 75.6 | -7.9 |
| Josh Patston | QLD | Edge | 14 | 97.5 | 83.3 | 59.2 | 6 | 95.7 | 12.3 |
| Jordan Samrani | NSW | Centre | 10 | 91.8 | 83.0 | 60.8 | 10 | 51.2 | -31.8 |
| Deine Mariner | QLD | Centre | 6 | 97.1 | 82.9 | 58.7 | 11 | 33.5 | -49.4 |
| Setu Tu | NSW | Winger | 14 | 90.3 | 81.5 | 61.0 | 15 | 56.2 | -25.4 |
| Josiah Karapani | QLD | Centre | 9 | 94.8 | 80.6 | 57.9 | 15 | 36.3 | -44.3 |
| Oliver Pascoe | QLD | Hooker | 22 | 94.7 | 80.5 | 53.3 | 13 | 85.6 | 5.1 |
| Samuel Hughes | NSW | Middles | 6 | 89.1 | 80.3 | 61.0 | 6 | 83.5 | 3.2 |
| Kelma Tuilagi | NSW | Edge | 5 | 87.7 | 78.9 | 59.9 | 14 | 49.9 | -29.0 |
| Trent Toelau | NSW | Halves | 12 | 87.4 | 78.7 | 58.8 | 6 | 48.3 | -30.4 |
| Jethro Rinakama | NSW | Winger | 14 | 86.5 | 77.8 | 59.8 | 5 | 78.0 | 0.2 |
| Tevita Naufahu | QLD | Winger | 9 | 91.3 | 77.1 | 57.5 | 5 | 96.5 | 19.4 |
| Cooper Bai | QLD | Middles | 7 | 90.5 | 76.3 | 57.5 | 13 | 70.2 | -6.2 |
| Blake Lawrie | NSW | Middles | 7 | 84.6 | 75.9 | 59.5 | 9 | 82.8 | 7.0 |
| Mawene Hiroti | NSW | Centre | 6 | 82.4 | 73.6 | 57.8 | 7 | 24.1 | -49.5 |
| Bailey Simonsson | NSW | Centre | 7 | 82.3 | 73.5 | 57.7 | 5 | 82.7 | 9.2 |
| Ali Leiataua | NSW | Centre | 6 | 82.3 | 73.5 | 57.7 | 14 | 21.5 | -52.0 |
| Noah Martin | NSW | Edge | 19 | 81.6 | 72.9 | 57.9 | 10 | 43.1 | -29.8 |
| Samuel Stonestreet | NSW | Winger | 8 | 81.4 | 72.6 | 58.1 | 12 | 22.4 | -50.2 |
| Manaia Waitere | NSW | Centre | 24 | 81.2 | 72.4 | 57.4 | 6 | 22.6 | -49.8 |
| Eddie Ieremia-Toeava | NSW | Edge | 17 | 80.2 | 71.4 | 57.4 | 5 | 76.4 | 5.0 |
| Va'a Semu | QLD | Middles | 15 | 84.7 | 70.5 | 55.7 | 5 | 63.8 | -6.8 |
| Sualauvi Faalogo | QLD | Fullback | 5 | 83.8 | 69.6 | 58.4 | 18 | 96.5 | 26.9 |
| Mathew Feagai | NSW | Fullback | 5 | 75.7 | 66.9 | 59.6 | 12 | 26.1 | -40.8 |
| Tanner Stowers-Smith | NSW | Middles | 11 | 71.6 | 62.8 | 55.3 | 12 | 94.2 | 31.4 |
| Ativalu Lisati | NSW | Edge | 7 | 70.9 | 62.1 | 54.4 | 10 | 46.3 | -15.8 |
| Trey Mooney | NSW | Middles | 17 | 70.5 | 61.7 | 54.9 | 16 | 57.2 | -4.5 |
| Hame Sele | NSW | Middles | 10 | 69.8 | 61.0 | 54.7 | 5 | 76.0 | 14.9 |
| Latrell Siegwalt | QLD | Fullback | 19 | 72.3 | 58.1 | 54.6 | 6 | 68.8 | 10.7 |
| Loko Jnr Pasifiki Tonga | NSW | Middles | 13 | 65.8 | 57.0 | 53.4 | 11 | 93.3 | 36.3 |
| Edward Kosi | NSW | Winger | 20 | 65.5 | 56.8 | 53.0 | 6 | 20.8 | -36.0 |
| Cody Ramsey | NSW | Fullback | 24 | 62.9 | 54.1 | 55.4 | 7 | 47.9 | -6.2 |
| Tanah Boyd | NSW | Halves | 16 | 61.8 | 53.1 | 50.5 | 9 | 63.2 | 10.2 |
| Selwyn Cobbo | QLD | Centre | 5 | 67.3 | 53.1 | 49.0 | 12 | 84.8 | 31.7 |
| Billy Burns | NSW | Edge | 5 | 60.9 | 52.2 | 51.2 | 17 | 46.1 | -6.1 |
| Ethan Sanders | NSW | Halves | 19 | 59.5 | 50.7 | 49.8 | 18 | 30.8 | -19.9 |
| Ben Talty | NSW | Middles | 13 | 56.2 | 47.4 | 50.3 | 17 | 70.9 | 23.5 |
| Hohepa Puru | NSW | Middles | 16 | 53.0 | 44.2 | 49.3 | 7 | 2.3 | -41.9 |
| Ronald Volkman | NSW | Halves | 22 | 51.7 | 42.9 | 47.3 | 11 | 62.3 | 19.4 |
| Saxon Pryke | NSW | Middles | 17 | 51.7 | 42.9 | 48.8 | 5 | 36.2 | -6.7 |
| Harrison Graham | QLD | Hooker | 11 | 56.2 | 42.0 | 40.8 | 13 | 98.3 | 56.3 |
| Jock Madden | QLD | Halves | 14 | 55.3 | 41.1 | 44.5 | 11 | 23.3 | -17.7 |
| Enari Tuala | NSW | Centre | 5 | 48.5 | 39.7 | 46.8 | 12 | 63.5 | 23.7 |
| Thomas Duffy | QLD | Halves | 8 | 53.6 | 39.4 | 44.0 | 7 | 46.0 | 6.6 |
| Lachlan Ilias | NSW | Halves | 20 | 45.6 | 36.8 | 45.3 | 7 | 18.1 | -18.8 |
| Zac Laybutt | QLD | Centre | 6 | 50.5 | 36.3 | 43.6 | 13 | 11.3 | -25.0 |
| Pasami Saulo | NSW | Middles | 22 | 43.2 | 34.5 | 46.1 | 12 | 31.4 | -3.0 |
| Moses Leo | NSW | Centre | 5 | 42.7 | 33.9 | 44.9 | 10 | 75.9 | 42.0 |
| Tristan Hope | NSW | Hooker | 16 | 41.6 | 32.8 | 40.0 | 5 | 41.1 | 8.3 |
| Zane Harrison | QLD | Halves | 11 | 46.4 | 32.2 | 41.7 | 8 | 28.5 | -3.7 |
| Tony Sukkar | NSW | Edge | 8 | 40.7 | 31.9 | 44.6 | 5 | 34.0 | 2.1 |
| Bronson Garlick | NSW | Edge | 8 | 37.7 | 28.9 | 43.7 | 10 | 12.3 | -16.6 |
| Joash Papalii | NSW | Fullback | 9 | 37.5 | 28.8 | 47.3 | 12 | 13.2 | -15.5 |
| Arama Hau | QLD | Edge | 13 | 42.9 | 28.7 | 41.5 | 16 | 20.8 | -7.9 |
| Billy Phillips | NSW | Middles | 21 | 34.6 | 25.8 | 43.3 | 13 | 52.2 | 26.4 |
| Ashton Ward | NSW | Halves | 17 | 32.6 | 23.8 | 41.1 | 10 | 39.1 | 15.2 |
| Preston Riki | NSW | Middles | 17 | 32.3 | 23.5 | 42.6 | 8 | 39.7 | 16.2 |
| Jack Underhill | NSW | Middles | 20 | 31.7 | 22.9 | 42.4 | 6 | 30.8 | 7.9 |
| Jensen Taumoepeau | NSW | Winger | 26 | 29.3 | 20.6 | 41.3 | 6 | 24.0 | 3.4 |
| Bunty Afoa | NSW | Middles | 16 | 28.9 | 20.2 | 41.5 | 6 | 73.5 | 53.3 |
| Josh Feledy | NSW | Centre | 16 | 27.1 | 18.3 | 39.9 | 6 | 24.8 | 6.5 |
| Freddy Lussick | NSW | Hooker | 18 | 25.8 | 17.0 | 34.9 | 13 | 41.3 | 24.3 |
| Luke Laulilii | NSW | Fullback | 10 | 23.7 | 15.0 | 42.8 | 8 | 73.6 | 58.7 |
| Jed Stuart | NSW | Winger | 10 | 22.8 | 14.0 | 39.2 | 10 | 5.7 | -8.3 |
| Kurtis Morrin | NSW | Middles | 16 | 18.0 | 9.3 | 38.0 | 16 | 26.3 | 17.0 |
| Royce Hunt | NSW | Middles | 7 | 17.8 | 9.0 | 37.9 | 8 | 96.1 | 87.1 |
| Charlie Guymer | NSW | Middles | 7 | 14.9 | 6.2 | 37.0 | 8 | 1.3 | -4.9 |
| Brandon Wakeham | NSW | Halves | 24 | 11.9 | 3.1 | 34.4 | 12 | 31.9 | 28.8 |
| Siulagi Tuimalatu-Brown | NSW | Winger | 14 | 10.7 | 1.9 | 35.2 | 5 | 2.6 | 0.7 |


## What this does not establish

- **The 2026 season is not finished.** Twenty rounds. A rating from five matches moves a long way with a sixth; treat the individual lines as illustrative and the aggregate as the result.
- **Only the promoted appear.** Clubs chose these men. A player BOSC rated at 70 who never got a game is invisible here, so this measures the projection among those given the chance, not its accuracy over everyone.
- **The 2026 positions are estimated**, so the pool a player is standardized against in `actual` is itself uncertain — most for wingers and fullbacks. This resolves when the match-sheet positions arrive with the completed season.
- **The feeder rating is cumulative** and the NRL rating is from one season, which is the right pairing for the question but means the two sides carry different amounts of evidence.
- Nothing here is fitted. If it were re-run after adding 2026 to the fit, it would stop being a test.
