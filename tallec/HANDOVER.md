# TALLEC — handover

Rewritten 2026-10-07, after the move to a new machine, for whoever picks this up next
(including a fresh assistant session). It replaces the 2026-08-25 version, which still
described the Windows laptop, the old rating scale, a database committed to the repo, and
listed `fit_translation_v2.py` as a routine command. `README.md` is the operating manual;
`GIGOT_V2_SPEC.md` and `AUS_DATA.md` are the method and data documents;
`DATA_NOT_IN_GIT.md` says which data is in git and why. This file is the state of play,
the open items, and the traps.

---

## 1. Where things are

| What | Where |
| --- | --- |
| Deploy repo (canonical) | `~/Downloads/NRL_2026_MPT/tallec/` — `github.com/matyastamas02/NRL_2026_MPT`, branch `main`, **public** |
| Working copy | `~/Downloads/TALLEC/` — its own git, **no remote**; kept identical to the deploy folder by hand |
| Live app | https://bosc-tallec.streamlit.app — redeploys on push to `main`, reads `tallec_app.db` |
| Full database | `tallec.db` (110 MiB), in both folders, **not in git**; audit log `tallec_audit.db` and `_backups/` beside it |
| Source data | `~/Downloads`: `TALLEC all Aus Data.xlsx`, `Player Level Stats NRL.xlsx`, `TALLEC_Super_League_Master_2021_2026_positions_complete.xlsx`, `Metadata.csv`, `TALLEC SL26 All Players.csv`, `BOSC_Full_Metric_Rate_Review_v2.xlsx`; `~/Downloads/sl21_players/` |
| Match masters | `NRL_2026_MPT/NRL_master.xlsx` and `SL_master.xlsx` — xLadder's, read by TALLEC, **never written by it** |
| Python | `NRL_2026_MPT/.venv` (3.14, scikit-learn 1.7.2): run as `../.venv/bin/python script.py` from `tallec/` |

Work in either folder, but run the checks and commit in `NRL_2026_MPT/tallec/`, then copy
the changed files back so `~/Downloads/TALLEC/` stays identical. Making the working copy
a clone of the deploy repo would remove that step; it has not been done.

Client documents (private artifacts; shared from each page's own Share menu):

- **What Is Live** — status note, rewritten 2026-10-07: https://claude.ai/code/artifact/3ddd7f4c-014a-4819-8008-e5ddfb0d6655
- **Reading the Numbers** — August, old scale, marked superseded: https://claude.ai/code/artifact/769e147b-3aaa-475a-8e1b-418f5c5f9e8b
- **Four Leagues, One Scale** — August workings, marked superseded: https://claude.ai/code/artifact/346b0b8e-27b1-460e-984c-8c446ed0bc63

Their sources are `what_is_live.html`, `reading_the_numbers.html` and
`session_writeup.html` in this folder. They are in the public repo, together with
`transfer_dataset.csv`; the owner has said that is fine.

---

## 2. State as of this handover

Counts are generated, not written here: the README's data-state block (`datastate.py`)
and the footer of the live app say what the database holds. The competition ladder lives
in `translation_ladder_v3`. The published scale was recalibrated on 2026-09-22 to a
**peer score** — 50 is the median of the player's position group in that competition and
season, not a percentile, and not comparable across competitions without Translation.
**Every figure from before 2026-09-22 is on the old scale and must not be quoted**,
including the −4.3 / −5.8 / −7.6 ladder, the 0.77–0.91 stability, the 3.8-point
correction and the TRACE_REPORT conclusion.

The client is Leeds, a Super League club, so the client's direction is **into Super
League** (`CLIENT_TARGET = "SL"`, sources NRL, NSW Cup, Queensland Cup). From 23 to 29
September the reports called feeder-to-NRL the client's cohort; the fifth external
review caught it, and the last two sentences still saying so were corrected on
2026-10-07.

Where the results stand (source and date with each, so they can be re-derived):

- **Translation into Super League**, 90 moves (`ROLLING_REPORT.md`): beats carrying the
  Australian rating across by +7.79 [+4.21, +11.40] and a flat 50 by +2.66
  [+0.16, +5.12]; against a two-parameter straight line, −0.26 [−1.56, +1.06] — not
  distinguishable. A calibrated correction, not a recruit ranking.
- **Arrival into Super League** (`ARRIVAL_REPORT.md`): +0.168 AUC [+0.093, +0.247]
  against source minutes alone; across all twelve directions the model is clearly ahead
  in 4.
- **Player layer in the match model** (`gigot_v2.py`, re-run 2026-10-07 after the
  2026-09-22 rebuild): +0.24 MAE [+0.05, +0.44] on the NRL over 752 walk-forward
  fixtures, +1.13 [+0.62, +1.63] on Super League over 497. Measured on the actual
  line-up, so an upper bound until the teams named before each round exist.
- Verdict: good for an internal or beta demo; not yet a validated recruitment ranking.
- **Moves into Super League are forecast by a straight line** since 2026-10-08
  (`predict_translation.LINE_TARGETS`), as a provisional default: `target ~ source` per
  direction from 25 pairs, otherwise the line over every next-season pair -- the rule the
  backtest scored, where 38 of the 90 forecasts into SL used the pooled line. Fitted on
  the shipped `translation_pairs_v3`. No clear difference from the model (17.72 against
  17.98, −1.56 to +1.06), which is not equivalence. Other directions and horizons keep
  the model; `score_model` is always reported.
- **Noise** (`noise_floor.py`): under a simplified repeat-measurement model the SL target
  rating varies by about 10 points. A sensitivity figure, **not** a floor under any
  forecast, and MAEs do not subtract: the earlier "10 noise + 8 real" reading was wrong
  (the sixth review; independent normal parts would make the other part about 15). The
  "noise removed" RMSE comparison proves nothing, since it subtracts the same amount from
  both sides. The SL stayers' 16.4 is a reference, not a ceiling.
- **The 2026 test** (`frozen_2026.py`, `FROZEN_2026_SPEC.md`, `frozen_2026_manifest.json`):
  H1 only -- the shipped line (S0) against the same line plus `rated_source_prev_year`
  (S1), fitted on the same next-season pairs with the correct fallback, scored on 2026
  entrants into SL alone; positive only if delta >= 1.0 and the lower bound > 0, with 25
  entrants and five per history group, otherwise undecided. A season after 2025 is
  refused unless the manifest's hashes match. Historical context (discovery data, not a
  test): 2023 +1.86 [-1.46, +4.93] on 23 entrants (below the minimum), 2024 -0.03,
  2025 -0.05. `retest_r6.py`'s +0.85 came largely from its fallback, which was fitted on
  the small directions only (the seventh review); it is superseded. The indicator
  partly measures evidence behind the source rating and data coverage
  (`prev_year_covered`), not experience.
- **Comparison card** (`predict_translation.comparables`, table `entry_cohort` from
  `build_entry_cohort.py`): earlier first/returning entrants in the same direction,
  one case per player, eight distinct players minimum, 10-90% only from twenty, above the
  line forecast. Rated entrants only. It read `translation_pairs_v3` until the seventh
  review showed those are not entrants.
- **The translation page read the v2 ladder** until 2026-10-08: eight directions and
  shifts on the retired scale. It now reads the v3 next-season ladder. Five of its six
  direction pairs carry opposite signs; Queensland Cup and Super League do not.
- **Starts rule**: exact 99.98% where it decides (10 of 54,808 match-sheet rows differ),
  84% fallback; validated on NRL 2020 only, unvalidated on NRL 2021-26.

**After the 2026 test** (`explore_post2026.py`, `POST2026_EXPLORE.md`, exploratory,
rolling 2023–2026):
- **Component translation** (each of the nine rates shrunk and translated on its own) is
  not established: +0.55 [−0.30, +1.41] into SL, +0.03 on 2026. A 2027 candidate at
  most. **Empirical-Bayes pooling** of direction lines is worse (−0.25); the hard
  25-pair switch stays.
- **Ranges.** The old ±1.96 SD band covered 100% into SL at 87 points wide. The
  Comparison page now shows the line's residual-quantile ranges, 50% and 80%. In the
  test they covered 50% and 83% into SL, and 48% and 80% across all directions
  (`predict_translation.RANGE_Q`). The conditional model's directions use the model's
  own in-sample residuals; that variant was not backtested. The card's middle half held
  only 37% (41% across all directions), and its caption now says so.
- **Decision outputs** (`build_decision_tables.py` → `player_season_starts`,
  `decision_calibration`, `top_third_thresholds`, `bench_evidence`; rerun it after any
  data change, then `build_app_db.py`).
  - P(top third of the target competition | forecast) is a logistic fit on the 1,512
    next-season pairs. In the test, a top-third-by-forecast shortlist hit 64% against a
    48% base.
  - The rating does not predict who starts (AUC 0.43–0.47 into SL); the source start
    share does (0.64).
  - The page shows the player's starts and, when he started fewer than half his
    matches, the bench evidence for the target: into SL, 39% of 31 bench entrants
    became starters against 52% of 100, landing 2021–25.
- **Trap: the Queensland Cup's rating basis depends on the data cut-off.** Through
  2022–2024 its position coverage is under 90%, so the ratings are
  competition-relative; from 2025 they are position-relative. Anything that compares a
  QLD rating with a threshold must take both from the same table.

**The methods programme after the 2026 test** (`METHODS_SUMMARY.md` has every result):
- **Mike's SL history workbooks** (2021–26; `ingest_sl_history.py --write` →
  `sl_teamsheets`, `sl_match_lines`): the 21-man squads, the matchday 17 by shirt, the
  handicap, the total line and the weather for every match. Rerun it when Mike sends a
  new season.
- **Nothing on these data beats the line by much.** Gradient boosting comes closest,
  +0.54 [+0.04, +1.02] across directions, and is a 2027 candidate. What does hold up:
  - the competition gaps, with NRL–SL at 13–16 points by every method;
  - the effect of moving into SL, about +10, heterogeneous by profile;
  - the named-team GIGOT gain, +0.93 with the named 17;
  - the model carrying information beyond the betting line.
- Most scripts read the exploration's cache: set `POST2026_CACHE` to a pickle path, and
  `explore_post2026.py` writes it on its first run.
- **Trap: the SL master's 2026 rows reuse eight team codes for other clubs** (C/CF,
  HF/HFC, LS/LH, WA/WFT). `team_map.solve` maps codes once for all seasons and gets 2026
  wrong; `gigot_sl_named.season_maps` maps them per season. The master's ELO continues by
  code, so those eight clubs began 2026 on the paired club's 2025 rating. This is an
  xLadder issue and has not been fixed (the xLadder files are read-only here); the owner
  should decide.
- **`tallec_app.db` is now 56 MB**, above GitHub's 50 MB warning; the hard limit is 100 MB.

One item from the August handover looks closed by the Super League 2025 repair of
2026-09-20: the master's stored margin predictions for 2025 had an error of 7.63 against
15.69 in training, which suggested an in-sample column. The same column now scores
17.17 for 2025 against 14.9–16.1 for 2022–2024 (checked 2026-10-07 from
`SL_master.xlsx`, read only). It is an xLadder matter in any case.

Verification, in order, from `tallec/`: `pytest tests -q`, `smoke_bosc.py`,
`build_manifest.py --check`, `datastate.py --check`, `build_app_db.py --check`, and
`export_review_package.py --check` before sending the review package anywhere (it looks
for the package one folder up, so run it from `~/Downloads/TALLEC/` or pass `--out`).

---

## 3. Open items

### Waiting on Mike

1. **Done: 2026 data.** `2026_Player_Data_Tamas.xlsx` (sheets NRL, SL, Q Cup, NSW Cup),
   every round through the grand finals, with a match-sheet `Position` on all but a few
   dozen rows. Imported with `weekly_update.py` (data_imports 40–43); it replaced the
   earlier partial NRL R1–20 and SL R1–19 rows, whose new players had slug IDs. Position
   checks pass (props 45–49 min, wingers 78, ten values). The starts rule on NRL 2026
   match sheets: decides 74% of rows at 100%, 40+ minutes gets 87% of the rest.
   `weekly_update.py` now keeps a registry date of birth once set (the string `min` let a
   reformatted or `0000-00-00` date replace it, which would have broken the freeze).
2. **Super League set restarts** — present on the 2026 SL rows, still NULL on SL
   2021–25 (DATA-1). Mike: everyone should have it, so the earlier SL export lacks it.
3. **`Try Assist - Kick` and `Kick - Forced Dropout`** — not in the 2026 export either
   (DATA-2). `Try Assists − Pass Outcome - Try Assist` is not a substitute (negative on
   some rows). Mike may re-export them for all players; ask for SL set restarts
   2021–25 in the same pull.
4. **Done: Middles use hit-up metres** — Mike confirmed ("ignore assists"), which is
   what `metric_spec` already does. "Edge" was not commented on; it stays.
5. **Named teams and Super League squad numbers.** Mike: SL does not publish team
   sheets (may start next year); he knows someone who collects them, else a slower
   workaround. Every match's actual line-up is
   already in the data, and so are mid-season club changes within the four
   competitions (5.5% of SL player-seasons show two clubs) and players moving between
   the NRL and a Cup. What is missing is what was known beforehand: the team named
   before each round (the match model needs it during the week; the backtest uses the
   actual line-up and is an upper bound), and each SL club's squad and squad numbers at
   the start of the season, the closest record of the role a club planned for a
   signing. Players on the books who never appear (injured, dropped, or loaned outside
   the four competitions, e.g. to the Championship) are invisible, which is why the
   look-ahead "vacated minutes" proxy cannot tell an injury from a departure. Squad
   numbers are public; 12 clubs × ~30 players × 3 seasons could be typed by hand.
6. **Metric dictionary** — all 341 rows still at `Decision=Review`; Mike's position
   specification may have superseded it, which Mike should confirm.
7. **Done: the Ben Talty row** — Mike: he played only for Burleigh. The QLD 2025 R20
   Capras row is removed (data_imports 44, recorded in `excluded_rows`); Burleigh played
   Redcliffe that round and the Capras had 18 rows. Ratings rebuilt; only QLD moved.

### Ours, not started

8. **Done: the 2026 test** — `FROZEN_2026_RESULT.md`. 42 SL entrants, S0 16.59 vs S1
   16.73, Δ −0.14 [−1.34, +0.99]: a one-point gain is not supported; the line stays.
   Next: freeze the 2027 specification (trained through 2026) before any 2027 result.
   Before that freeze, clean what the 2026 freeze kept fixed: mixed DOB formats in
   `players` (dd/mm/yyyy from the old SL 2026 feed), a few `0000-00-00` and one
   disagreeing DOB (Salesi Foketi), and two likely re-issued IDs (Floyd Aubrey QLD
   70057/72468, Daniel O'Donnell NSW 67489/55028).
9. **A season-by-season shortlist test** for the arrival model, against a minutes
   baseline whose direction is learned from the training window (the sixth review showed
   reversing raw minutes would beat the model on NRL→QLD).
10. **Established-role output and a Super-League-target shortlist validation.**
11. **Rating reliability and a measurement-error model.**
12. **Sensitivity of the arrival model's 0.35 pooling.**
13. **The named-team backtest** — needs item 5; the single question that would most
    change what can be claimed about the match model. **Expected role from squad
    numbers** — also needs item 5; the one untested candidate for closing the
    translation's gap above the noise floor.
14. **Refresh *Reading the Numbers*** on the peer-score scale, or retire it. The status
    note currently points readers to the app's own explanation instead.
15. **Position-specific ratings.** A player who covers several positions gets one rating
    against his most common one; the scope asks for "multiple positional ratings for the
    same player (e.g. centre and wing)". Not built.
16. **One scope definition to settle with Mike.** Form: settled, five matches only (Mike:
    three is too noisy), as `config.json` already has. Divergence is Form
    minus Class on the composite scale; the scope defines it as "% over or under".

### Commercial (the owner's, not the code's)

- One-page IP and revenue-share agreement: the LICENSE names joint ownership with no
  percentages.
- The investor hosting quote (2026-09-03: 70–105 h build, €80–150/month running,
  4–8 h/month support) is still open.

---

## 4. Traps — read before touching anything

**Do not run or import `fit_translation_v2.py`.** It overwrites
`translation_model_v2.pkl`, which is sealed (hash `ef528cf680e7b102`):
`v1_holdout_record.json` records an out-of-sample result produced by that exact file, the
only untouched holdout the project has. Do not touch `v1_holdout_record.json` either. The
current ladder and model come from `fit_translation_v3.py`. Every new fit reads
`FREEZE_SEASON` (2025) from `config.json`.

**The live app reads `tallec_app.db`, not `tallec.db`.** A weekly update changes nothing
on the live site until `python build_app_db.py` has run and `tallec_app.db` is committed
and pushed; `--check` says when it is behind. Nothing else may write to that file.

**Do not write derived numbers into prose by hand.** It has gone stale four times. The
reports are generated by their scripts; if a sentence in one is wrong, fix the generator
and the report together.

**Validate any position column before loading it.** The first Super League position file
was misaligned and would have silently corrupted every rating. Five checks separate real
from misaligned, and the broken file failed all of them:

| Check | Real match-sheet data | The broken file |
| --- | --- | --- |
| Position values vs data rows | equal | 983 spare |
| Mean minutes by position | props ~46, wingers ~78 | all 62–66 |
| Tackles per minute, spread | ~9–10× | 1.3× |
| Passes per minute, spread | ~30× (hookers highest) | 1.8× |
| Distinct positions per player | ~1.5 | 1.92 |

Also cross-check against the independent Australian career record: the corrected file
agreed on 74% of players, the broken one on 12%.

**`ingest_full_season.py` rebuilds its tables.** It refuses to run while competitions it
does not manage are present, but that guard only fires *because* the Australian data is
loaded. It deleted the positions once when the database held only NRL and SL. Prefer
`weekly_update.py`.

**Three legacy scripts refuse to run without `--i-know-this-overwrites`**:
`seed_mock_ratings.py` (writes *mock* ratings), `add_positions.py`, and
`player_rating_engine.py`'s own `__main__`. They predate the one-writer rule.

**Every write goes through `runtime.guarded_write`** — snapshot, automatic restore on
exception, recorded either way. The audit log is in `tallec_audit.db`, a **separate file
on purpose**: it lived inside `tallec.db` first, so a rollback erased the record of the
failure it was recovering from. `python runtime.py restore` puts the last snapshot back.

**A stat a season never recorded must not be scored as average.** Post-contact metres only
exist from 2025 and carried 24% of a prop's weight in every earlier season. The engine
drops rates below `min_rate_coverage` and renormalises. If a new feed adds or removes a
column, check `eng.dropped` per season before trusting cross-season numbers.

**Position coverage gates the rating mode.** A competition is rated within position groups
only above 90% coverage (`min_position_coverage`); below it, the whole competition is the
peer group. `player_ratings.rating_basis` records which applied and the app states it.

**`HF` is Huddersfield and `HFC` is Hull FC.** The xLadder notes say `HF = Hull FC`.
`team_map.py` solves the codes from scores and fixtures rather than from documentation.

**Competitions do not share a current season.** NSW Cup and Queensland Cup stop at 2025;
NRL and Super League run to 2026. Use `season_of(comp)` in the app, per-competition in
`regenerate_full.py`.

**The Stats Perform Player ID is global**, but pandas reads it as float when a file has
blank rows, which turns `24528` into `24528.0` and forks every identity. Always go through
`sp_schema.normalize_player_id`.

**Interchange is a role, not a position.** A player's career position is the mode of his
*starting* positions (`sp_schema.primary_position`).

**The history was rewritten on 2026-09-29** to remove `tallec.db`. Any clone from before
then still holds it and has a diverged `main`; re-clone rather than pull.

---

## 5. Running things

```bash
cd ~/Downloads/NRL_2026_MPT/tallec
../.venv/bin/python -m pytest tests -q     # synthetic data, safe any time
../.venv/bin/python smoke_bosc.py          # every app page, all four competitions
../.venv/bin/python -m streamlit run bosc_app.py --server.port 8503

../.venv/bin/python weekly_update.py --file "SL26 Players.csv" --competition SL --season 2026 --dry-run
../.venv/bin/python weekly_update.py --file ... --competition SL --season 2026   # for real
../.venv/bin/python regenerate_full.py     # ratings + contribution, all competitions
../.venv/bin/python build_decision_tables.py --write   # starts, top-third calibration
../.venv/bin/python build_app_db.py        # refresh the copy the live app reads
../.venv/bin/python fit_translation_v3.py  # the ladder and the translation model
../.venv/bin/python gigot_v2.py            # the match-model evaluation (read-only)
../.venv/bin/python runtime.py             # provenance and audit counts
../.venv/bin/python runtime.py restore     # undo the last guarded write
```

Deploy: run the checks in §2, `git add` the changed files (including `tallec_app.db` if
the data moved), commit, `git push origin main`. Streamlit redeploys itself; a cold start
takes a few minutes. Then copy the changed files into `~/Downloads/TALLEC/`.

---

## 6. How the client conversation stands

On 2026-10-07 the status note **What Is Live** was rewritten for that day and an update
email to Mike drafted around it. The two August documents carry a banner saying their
numbers are on the old scale. Whatever was sent between 2 and 23 September is not
reconstructed here and does not need to be: the new note supersedes it. Invoicing for the
first checkpoint is done.
