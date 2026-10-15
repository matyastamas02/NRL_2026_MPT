# Frozen 2026 test — result

Run once, unchanged, with `python frozen_2026.py --eval-season 2026 --out frozen_2026_run`,
after the 2026 season of all four competitions was imported (data_imports ids 40–43) and
`check_frozen` confirmed the release still matched `frozen_2026_manifest.json`. The
fingerprint the run saw is in `frozen_2026_run/frozen_run.json`; the row-level export is
`frozen_2026_run/frozen_eval_2026.csv`.

**H1, entries into SL from NRL/NSW/QLD, 2026 alone:** 42 rows, 42 players, 9 without and 33
with a rated previous source season (both groups clear the minimum of five; 42 clears 25).
MAE S0 16.59, S1 16.73, Δ −0.14, 95% player-cluster bootstrap [−1.34, +0.99].
**Label: A ONE-POINT GAIN IS NOT SUPPORTED** (upper bound < 1.0). All 42 S0 forecasts used
their own direction's line. No forecast was missing.

Secondary, descriptive only (rule 13; cannot rescue H1):

| Comparison | MAE | Δ (95%) |
|---|---|---|
| S0 + shrinkage_B vs S0, into SL | 16.59 → 16.89 | −0.31 [−0.99, +0.32] |
| C1 vs C0, into SL | 17.45 → 17.31 | +0.14 [−0.60, +0.86] |
| H2 vs C1, all entries (retrospective club) | 20.23 → 20.15 | +0.08 [−0.51, +0.66] |

Consequence: the shipped straight line S0 stays the app's →SL forecast; the history
indicator is not added. 2026 is exploratory (rule 1: prior exposure to partial 2026 data).
2027 needs its own specification, trained through 2026 and frozen before any 2027 outcome
is seen (rule 17).

After the run, one through-2025 row was removed at the provider's instruction (Ben Talty,
QLD 2025 round 20, Capras; data_imports id 44). The manifest therefore no longer matches
the database and a second `--eval-season 2026` run refuses, which is intended: the run
above is the record.
