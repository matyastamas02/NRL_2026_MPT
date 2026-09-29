# Freezing the fit at 2025

## Why

Mike's instruction (2026-08-26): leave the 2026 data alone until the season finishes in early October, rather than re-importing it every week for a project that runs longer than that. He also drew the methodological consequence, which is the real reason to do it — with everything frozen at the end of 2025, the 2026 season becomes an out-of-sample test of what the ratings and the translation model projected, *particularly for players who changed competitions or levels*.

That only holds if 2026 contributed nothing to the projections. It previously did: the ratings were built from every season including 2026, and the translation model used moves landing in 2026. Freezing is therefore not a convenience — without it the planned test would be scored against its own training data.


## How it is implemented

One setting, `evaluation.freeze_season` in `config.json`, currently **2025**. It is read once in `player_rating_engine.py` as `FREEZE_SEASON` and consumed by both fitting scripts — `regenerate_full.py` for the ratings and `fit_translation_v2.py` for the translation model. A single source so the two cannot end up frozen at different points, which would be silent and would invalidate the comparison between them.

Seasons after the freeze are **not deleted**. They stay in `player_match_stats` and the app can still display them. What the freeze governs is only what is *fitted*. Setting it to `null` restores the previous behaviour.


## Effect on the ratings

Both rebuilds are in the audit log, named by run id: **2** before the freeze and **3** after. The config hash each ran under — `e3cb138d8ec0` before, `233a71856591` after — is the recorded evidence that the settings, not the data, changed. The database held 122,359 player-match rows throughout; nothing was added or removed.

| competition | season_before | season_after | players_before | players_after | rows_before | rows_after |
| --- | --- | --- | --- | --- | --- | --- |
| NRL | 2026 | 2025 | 466 | 472 | 44961 | 40040 |
| SL | 2026 | 2025 | 417 | 373 | 31763 | 27287 |
| NSW | 2025 | 2025 | 496 | 496 | 21427 | 21427 |
| QLD | 2025 | 2025 | 494 | 494 | 24208 | 24208 |

The variance components, which set the rating scale and the shrinkage:

| competition | sigma2_before | sigma2_after | tau2_before | tau2_after |
| --- | --- | --- | --- | --- |
| NRL | 0.1817 | 0.1818 | 0.0389 | 0.0383 |
| SL | 0.1934 | 0.1918 | 0.0377 | 0.0377 |
| NSW | 0.2002 | 0.2002 | 0.0403 | 0.0403 |
| QLD | 0.1780 | 0.1780 | 0.0427 | 0.0427 |

**Read the player counts carefully — they move for a reason that is not the freeze itself.** Ratings are published for players active in the most recent fitted season, so that season changed from 2026 to 2025 for the NRL and Super League. The NRL *gained* players (a complete 2025 has more men appear in it than a 2026 stopped at round 19) while Super League *lost* them (2026 runs 14 clubs against 2025's twelve, Bradford and York having joined). Neither movement says anything about rating quality.

The variance components barely move — σ² and τ² agree to three decimal places in every competition. That is the useful robustness check here: removing a season did not rescale the ratings or change how hard the shrinkage pulls, so scores before and after the freeze remain comparable.


## Effect on the translation model

A controlled comparison: the same code fitted twice, once with the freeze off and once on, nothing else altered. `n` is the number of observed moves behind each estimate and the points are on the 0-100 rating scale.

| source | target | n_unfrozen | n_frozen | pts_0_100_unfrozen | pts_0_100_frozen | d_pts |
| --- | --- | --- | --- | --- | --- | --- |
| QLD | NRL | 190 | 190 | -7.92 | -7.92 | 0.00 |
| NSW | NRL | 464 | 464 | -6.23 | -6.23 | 0.00 |
| QLD | SL | 49 | 32 | -4.53 | -5.72 | -1.19 |
| SL | NRL | 23 | 23 | -4.63 | -4.63 | 0.00 |
| NSW | SL | 62 | 45 | -0.66 | 0.22 | 0.88 |
| SL | NSW | 18 | 18 | 2.16 | 2.16 | 0.00 |
| SL | QLD | 21 | 21 | 2.93 | 2.93 | 0.00 |
| NRL | SL | 126 | 107 | 4.58 | 4.25 | -0.33 |

**Five of the eight directions are bit-identical, and the three that moved are exactly the three that land in Super League.** This is not a coincidence and it is the cleanest evidence that the freeze did what it was supposed to and no more: the Australian competitions have no 2026 data at all, so no move ending in the NSW Cup, the Queensland Cup or — through same-season pairing — the NRL could ever have reached 2026. Only Super League has a 2026 season, so only moves into it lost observations (QLD→SL 49→32, NSW→SL 62→45, NRL→SL 126→107).

Those three shifted by 0.33 to 1.19 points and not in a consistent direction, which is what dropping a fifth to a third of a small sample looks like. The two Super League estimates that matter most for the client are also the ones now resting on the fewest moves. When 2026 is added back in October these three are the numbers to re-check first.


The translation model's fit statistics, read from the database rather than typed in. An earlier version of this report quoted them from memory, which held until the model was refitted and then quietly described a model that no longer existed.


| label | n | players | rmse | mae | naive | r2 |
| --- | --- | --- | --- | --- | --- | --- |
| A_same_season | 1436 | 439 | 23.468 | 19.564 | 31.270 | 0.166 |
| B_next_season | 1512 | 623 | 23.826 | 19.940 | 30.833 | 0.116 |

The better of the two layers cuts the error to 23.47 from 31.27 for assuming a player rates the same in the new competition — an improvement of 24.9%.

The ladder's internal consistency check, recomputed from the model in the database rather than quoted from a run since superseded. Going the long way round — feeder to NRL (-13.12 points), then NRL to Super League (+15.45) — gives +2.33. Measured directly, feeder to Super League is -3.31. The two routes disagree by 5.64 points.

This is what licenses quoting a translation for a pair with few direct observations of its own, so the size of the disagreement is the size of the licence. At 5.6 points it is not nothing. Every Super League direction rests on a few dozen moves, and this is the arithmetic saying so — quote a Super League translation as a band rather than a figure, and revisit when the 2026 data roughly doubles those samples.


## A second change made at the same time

`regenerate_full.py` and `fit_translation_v2.py` were writing to the database directly, outside `runtime.guarded_write` — contrary to what the handover states. Each replaces three tables wholesale, so a failure part-way through would leave the app reading a rebuild that never finished, with no snapshot to go back to. Both now write inside the guard, and the two runs in this report are the first entries either has made in the audit log. This is unrelated to the freeze; it was found while preparing to run it.


## Current state

| competition | season | players | avg_effective_matches | rating_basis |
| --- | --- | --- | --- | --- |
| NRL | 2025 | 472 | 33.9 | position_relative |
| NSW | 2025 | 496 | 17.6 | position_relative |
| QLD | 2025 | 494 | 19.9 | position_relative |
| SL | 2025 | 373 | 31.7 | position_relative |

`avg_effective_matches` is not matches played. Since season weighting came in on 19 September the shrinkage counts a match from four years ago as a quarter of a match, and this column sums those weights — which is why it reads around half the played figure. It is the weight of evidence behind a rating, which is what the shrinkage acts on.


Verify with `python -m pytest tests -q`, `python smoke_bosc.py` and `python build_manifest.py --check`. No count is quoted here: a number typed into a report is a number that goes stale, and this report was caught doing exactly that with the translation statistics above.


## What a reader should question

- **The app now shows 2025.** This is the intended consequence, but it is visible to anyone who opens BOSC: the NRL and Super League player lists are 2025 squads, not 2026. Anyone expecting current-season names will think something is broken. It reverts when 2026 is added in October.
- **The freeze is only as good as its coverage.** It governs the two fitting scripts. Any future script that fits something and does not read `FREEZE_SEASON` would reintroduce the leak silently. There is no test that enforces this, which is a gap worth closing.
- **The three Super League ladder entries rest on fewer moves than before** (32, 45 and 107). Their standard errors are reported by `fit_translation_v2.py` and are wide; a reader should not treat a one-point difference between competitions as meaningful at those sample sizes.
- **The 2026 rows still carry estimated positions** for the NRL — right about four in five times overall and only about half for wingers and fullbacks (`validate_positions.py`). The freeze keeps them out of the fit, but the out-of-sample test planned for October will be measured *on* those rows, so it needs the match-sheet positions to arrive with the data.
- **Nothing here validates the ratings.** It records that the fit was moved and what moved with it. Whether the ratings predict anything is `VALIDATION_REPORT.md`.
