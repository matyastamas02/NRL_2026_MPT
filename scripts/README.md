# Data tooling

## `rebuild_2026_from_restated.py`

One-off rebuild of all season-2026 rows in both masters from Mike's full-season
**"Model Stats Restarted"** CSV exports (July 2026). Reusable pattern if a restated
full-season export arrives again.

What it does:

1. Base = a backup of the current Google Sheet state (`sheet_backup_YYYYMMDD/` CSVs),
   so weekly rounds already in the sheet are preserved for other seasons.
2. Pairs the two per-team rows of each match in the export (Round + Team/Opposition).
3. Joins to fixture records (`nrl_fixtures.csv` / `sl_fixtures.csv`, parsed from the
   Wikipedia season-results pages) to get **home team, venue, date and the official
   final score** — the exports have no venue column and their `Points Scored` aggregate
   undercounts the real score by 2–4 pts in ~10% of matches.
4. Builds master-format rows (`A_`/`B_`/`Diff_` columns, Match ID
   `{Season}-{Round}-{CodeA}-{CodeB}`, A = home), derives `Pre-Contact Metres`
   (= Metres Carried − Post Contact Metres), carries over the client's existing
   `Home Advantage` neutral flags, and replaces every 2026 row.
5. Writes `updated_masters/{NRL,SL}_master_updated.csv` + a validation report
   (missing pairs, score fixes, home-side conflicts) to stdout.

Upload to the Google Sheet and the repo xlsx files is done separately (gspread,
chunked `ws.update`) — always take a full sheet backup first.

## `measure_weekly_elo_fix.py`

Measures the 2026-10-21 `get_current_elos` fix (see CLAUDE.md §9, "Weekly submits
froze ELO"). Replays season 2026 through `append_new_round` one round per submit,
with the old and the fixed function, then compares each with the stored master:
M3/M3+ winner accuracy and margin MAE (train 2022–2025, via `run_pipeline`), with
a paired bootstrap over matches (10,000 draws). It also checks that the old code
reproduces the stored NRL R18–R19 ELO, and that a weekly replay under the fix equals
a one-batch recompute. Read-only: it writes only `elo_fix_measure.json` to the
output folder.

    python scripts/measure_weekly_elo_fix.py <repo_dir> <out_dir>
