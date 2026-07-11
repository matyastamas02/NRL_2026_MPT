# TALLEC — Player Intelligence Platform for Rugby League Recruitment

**Status:** Phase 0 (infrastructure & schema) complete. Phase 1 (BOSC UI) in design → build (Aug 15–31). Phase 2 (Competition Translation) model design + transfer history collection (Jul 15–Aug 15).

Proof-of-concept ingest of **Gerard NRL player-stats exports** into SQLite (`tallec.db`), ready to extend with Stats Perform weekly imports. Vertical slice: Leeds Rhinos recruitment analyst dashboard (Form/Class/Benchmark player ratings + NRL ↔ SL translation model).

## What's in the database

- `player_match_stats` — one row per player per match, 64 raw stats normalised to
  numeric (percentages stripped, `3.34s` → 3.34) + per-minute derived metrics.
  Currently: **NRL 2026 rounds 11–12, 494 player-match rows.**
- `players` — provisional registry (player_id = name slug), teams/positions seen,
  first/last round, matches, total minutes.

Run `python ingest_gerard.py` to rebuild — it picks up every
`nrl_2026_player_stats_rounds_*.csv` in Downloads and `gerard_round12/`.

## Data source inventory (as of 2026-07-09)

| Source | Coverage | Where |
| --- | --- | --- |
| Gerard player stats (nrl.com scrape) | NRL 2026 R11, R12 only — **full season available on request** ("if you need any other rounds just let me know") | Mike's fwds: "Files Attached" (May 19), "Round 12 Files" (May 27) |
| Gerard extras | tryscorers with halves (R11-12), set restarts (R11-12), play-by-play (R11 only) | same emails |
| SL player data | **exists on Mike's side** ("we still have it for Super League") — never sent to us | — |
| Stats Perform player+team (NRL, SL, NSW Cup, Q Cup) | promised in TALLEC scope, samples requested in quote reply | pending |

## Caveats

- Gerard exports leave zero-valued stats **blank** — for count stats (errors,
  tackle_breaks, tries…) NaN usually means 0, not missing. Decide fill policy
  per stat before modelling.
- `player_id` is a name slug — fine for a two-round seed, but the real Phase 1
  needs collision handling (same name, different player) and cross-competition
  identity matching.
- Positions include bench roles (Interchange, Reserve) — positional benchmarks
  will need the *played* position, likely from stint minutes + number.
