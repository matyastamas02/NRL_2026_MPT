# TALLEC — Player Intelligence Platform for Rugby League Recruitment

Player-level ratings, positional benchmarks and a measured competition-translation
model, built on Stats Perform player-match data and served through the BOSC
recruitment app. `AUS_DATA.md` documents the Australian source file in detail;
`GIGOT_V2_SPEC.md` covers the match-model integration.

## What is in the database

`tallec.db` — SQLite, ~85 MB, **122,359 player-match rows** across four competitions
and 23 league-seasons.

| Competition | Seasons | Rows | Players | Position known | Rated against |
| --- | --- | --- | --- | --- | --- |
| NRL | 2020–2026 | 44,961 | 927 | 98% | its own position groups |
| Super League | 2021–2026 | 31,763 | 834 | 100% | its own position groups |
| NSW Cup | 2021–2025 | 21,425 | 1,234 | 100% | its own position groups |
| Queensland Cup | 2021–2025 | 24,194 | 1,188 | 91% | its own position groups |

All four are rated position-relative and selectable in the app — 466 NRL, 417 Super
League, 496 NSW Cup and 494 Queensland Cup players. **Competitions do not share a current season**: the
Australian feeder data stops at 2025 while the NRL and Super League run into 2026, so
every competition-scoped query resolves its own (`season_of()` in the app,
per-competition in `regenerate_full.py`).

Key tables:

- `player_match_stats` — engine view: one row per player-match, lowercase stat columns
  the rating engine and app query, plus `position` and `position_source`.
- `player_match_raw` — the full canonical Stats Perform field set (342 columns), so the
  metric dictionary lines up 1:1 with the source.
- `players` — registry keyed on the permanent Stats Perform Player ID (name-slug only
  where no ID has ever been seen), with date of birth.
- `player_ratings` — Form, Class, Divergence, shrinkage and `rating_basis` per player.
  **Class spans every season the player has**, each match standardized against its own
  season's pool; Form is his last five matches. One standardization mode per
  competition for the whole history, chosen from overall position coverage — mixing
  position-relative and competition-relative seasons into one Class would be
  meaningless.
- `player_contribution`, `player_contribution_rating` — GIGOT input #5.
- `translation_ladder`, `translation_pairs`, `translation_model_meta` — the measured
  competition level differences and the observations behind them.
- `metric_dictionary`, `rate_rules`, `overlap_rules` — Mike's Volume + Rate spec as
  config. **All 341 rows are still `Decision=Review`** and need sign-off.
- `player_id_map` — every name-slug → permanent-ID re-key, for audit.
- `excluded_rows` — the 16 rows dropped for breaking the player-match key.

## Which script writes what

Each table has exactly one writer. Running the wrong script is how data gets lost, so
the boundaries are enforced in code, not just documented.

| Script | Writes | When |
| --- | --- | --- |
| `weekly_update.py` | one round of `player_match_stats` / `player_match_raw`, `players` | **every week in season** |
| `ingest_full_season.py` | rebuilds NRL 2026 + SL from their season files | rarely; refuses to run while other competitions are present |
| `ingest_aus_history.py` | appends the Australian history file | once, done 2026-08-19 |
| `regenerate_full.py` | `player_ratings`, `player_contribution*` (all four competitions) | after any data change |
| `unify_sl_ids.py` | re-keys SL history onto permanent ids | superseded — the 2021-2026 master carries real ids |
| `resolve_duplicates.py` | splits identifiers covering two players, restores the 16 held-out rows | once, done 2026-08-25 |
| `gigot_v2.py` | nothing — evaluation only, writes `gigot_v2_results.csv` | when the question is asked again |
| `fit_translation_v2.py` | translation tables + `translation_model_v2.pkl` | when a season completes |
| `gigot_contribution.py` | **nothing** — import-only module holding the formula | — |
| `runtime.py` | the audit log in `tallec_audit.db`; snapshots into `_backups/` | imported by every writer |

Shared definitions live in `sp_schema.py` (field mapping, position groups, the
career-position rule) and `config.json` (rating weights, contribution weights, the
position-coverage threshold). Change them there, not in a copy.

## Safety, and how to see what happened

Every script that writes to the database does it inside `runtime.guarded_write`. That
takes a snapshot first, **restores it automatically if anything raises**, and records
the attempt either way. Eight snapshots are kept in `_backups/` (git-ignored — they are
full copies of an 85 MB file).

The audit log lives in `tallec_audit.db`, a separate file on purpose: it used to live
inside `tallec.db`, which meant a rollback restored a snapshot taken before the failure
was recorded, so the recovery erased the record of what it was recovering from.

```
python runtime.py            # provenance and audit row counts
python runtime.py backups    # what snapshots exist
python runtime.py restore    # put the most recent one back
```

`data_imports` has one row per write: what ran, with which arguments, which commit,
rows before and after, which snapshot it took, and whether it succeeded or rolled back.
`model_runs` has one row per rating or model rebuild with the fit statistics, so two
numbers taken a week apart can be traced to different runs. The app footer shows the
commit, the config hash and when the ratings were last rebuilt.

## Tests

```
python -m pytest tests -q
```

27 tests, synthetic data only, never touches `tallec.db`. Every one corresponds to a
bug that shipped at least once: the float player-id that forked every identity, the
Interchange rule, the position-coverage gate, shrinkage rising with evidence, the
leakage claim, the scoring validation, the cross-club round repeat, and the rollback.
One test asserts a *known* weakness — that fitting the standardization on the whole
pool is not a strict walk-forward — so that nobody mistakes it for the strict version.

## Weekly update — the procedure

1. **Save the new file** from Mike anywhere; the loader looks in the repo and its
   parent directory.
2. **Dry-run it.** Nothing is written, and you see the validation result, how many
   players resolved to a permanent ID, and which rounds would be replaced:

   ```
   python weekly_update.py --file "SL26 Players.csv" --competition SL --season 2026 --dry-run
   ```

3. **Read the validation block.** `[ERROR]` blocks the write. The scoring check is the
   one that matters: `Points Scored` must equal 4·tries + 2·conversions + 2·penalty
   goals + field goals. A Super League feed once failed this badly enough to flip which
   team won a match. Squad size, minutes and duplicate keys are warnings.
4. **Run it for real** — drop `--dry-run`. Ratings and contribution are rebuilt
   automatically afterwards.
5. **Check the summary line**: rows now held for that competition and season, the last
   round, and the registry size.

Useful flags:

- `--rounds 21 22` — only those rounds, when a file carries the whole season to date.
- `--sheet NRL26` — required for a multi-sheet workbook.
- `--no-ratings` — skip the rating rebuild when loading several files; run
  `python regenerate_full.py` once at the end.
- `--force` — write despite validation errors. Say so in the write-up if you use it.

**Re-running is safe.** Rows are keyed on competition + season + round, and the file
replaces whatever the database holds for those keys. Importing the same file twice
leaves the same row count, verified bit-for-bit against a backup.

## Conventions that matter

- **Position is a per-match jersey.** The Australian history file and the Super League
  2021-2026 master carry it directly (`position_source='match'`); the 2026 NRL feed does
  not, so those rows are backfilled through the permanent Player ID and marked
  `position_source='career_aus'`. Interchange is a role, not a position: a player's
  career position is the mode of his *starting* positions (`sp_schema.primary_position`).
- **Ratings go position-relative only above 90% position coverage** for that
  competition (`config.json: min_position_coverage`). Below it, splitting a partly-known
  pool would make players incomparable, so the whole competition is the peer group.
  `player_ratings.rating_basis` records which applied — the app displays it.
- **Sample size is not decoration.** Ratings are shrunk toward the peer average by
  `B = τ²/(τ² + σ²/n)`; a one-match player keeps under a third of his own number. Tables
  in the app also apply a minimum-matches floor, because shrinkage alone still lets a
  freak single game top a sorted list.
- **Two uncertainties, never merged.** The average translation shift between two
  competitions is known to about ±1 rating point; an individual player's outcome to
  about ±12. The first sets expectations, the second forbids ranking recruits on it.

## Running the app

```
python -m streamlit run bosc_app.py
```

Deployed from the `NRL_2026_MPT` repo (`tallec/bosc_app.py`) to
https://bosc-tallec.streamlit.app — it reads the bundled `tallec.db`, so a data change
needs the database copied across and pushed. Streamlit Cloud installs from the
`requirements.txt` nearest the main module.

Before pushing an app change, run the headless page check:

```
python smoke_bosc.py
```

It drives all six pages for each of the four competitions — 24 combinations — and fails
on any exception. It is what caught a helper that made two pages take longer than ninety
seconds to render.

## Open items

1. **NRL 2026 positions.** The only remaining position gap. Its 2026 rows are estimated —
   3,033 from a metadata file, 1,754 from players' Australian careers, 134 unknown. Ask for
   the same treatment the Super League master got. On arrival, check: average minutes per
   position (props ~46, wingers ~78), the spread of passes per minute (~30x, hookers
   highest), positions per player (~1.5), and that the column has no more values than the
   data has rows.
2. **Metric dictionary sign-off.** All 341 rows sit at `Decision=Review`; ten are low
   confidence and one (`Charge Down`) points at a denominator that is not in the data.
3. **A team-list feed.** The player layer helps the match model by about a quarter of a
   point of margin MAE (`GIGOT_V2_SPEC.md`), measured on line-ups known after the fact.
   Friday-morning team sheets are the version a client can act on.
4. **2026 for the feeder competitions** — the Australian history file stops at 2025.
5. **The SL master's stored margin predictions are in-sample for 2025** — an xLadder issue
   rather than a TALLEC one, but the Super League app reports its accuracy from that column.
