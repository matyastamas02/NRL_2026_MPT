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
- **Candidates** (`retest_r6.py`, specified before running; supersedes the
  `team_role_trend.py` table): one entry cohort for fitting and scoring, the app's line
  rule, a joint fit, the first club seen in the target season, and `has_history` as its
  own baseline because most of the trend's earlier +0.50 came from whether a trend
  existed. Into SL: has_history +0.85 [+0.20, +1.54], all three origins; nothing clear
  on top of it. All moves: incumbents at the new club +0.53 [+0.12, +0.92]. The
  cohort-trained line is weaker than the shipped one (18.77 against 17.72), so these
  figures compare only with each other. Exploratory: found on the same 90 moves.
- **Comparison card** (`predict_translation.comparables`): every translation shows
  earlier movers in the same direction within 7.5 rating points (same position group if
  eight remain, window doubled once, else "not enough data").
- **The translation page read the v2 ladder** until 2026-10-08: eight directions and
  shifts on the retired scale. It now reads the v3 next-season ladder. Five of its six
  direction pairs carry opposite signs; Queensland Cup and Super League do not.
- **Starts rule**: exact 99.98% where it decides (10 of 54,808 match-sheet rows differ),
  84% fallback; validated on NRL 2020 only, unvalidated on NRL 2021-26.

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

1. **2026 data** — NSW Cup and Queensland Cup 2026, and NRL 2026 with match-sheet
   positions. Promised for early October. Run the position checks in §4 on arrival.
2. **Super League set restarts** — NULL on every row (`metric_spec.DATA_REQUESTS`
   DATA-1). Without it discipline cannot be compared across hemispheres.
3. **`Try Assist - Kick` and `Kick - Forced Dropout`** — named in Mike's spec, absent from
   the extract (DATA-2).
4. **Middles: hit-up metres or line-break assists**, and confirmation of the name
   "Edge". Folding Lock into Middles dropped the lock block's LBA-per-receipt slot
   (`metric_spec.LOCK_BLOCK_RETIRED`).
5. **Named teams and Super League squad numbers.** Every match's actual line-up is
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
7. **The Ben Talty row** — reassigned as instructed, which gives Ben Talty a round-20
   Capras appearance while every other Queensland Cup 2025 row for the player is Burleigh
   Bears.

### Ours, not started

8. **The 2026 test, fixed in advance** (`retest_r6.py` as it stands): has_history for
   moves into SL and incumbents across all directions, beside the line and the model,
   same cohort and line rule. Do not retune on 2026; 2027 stays the confirmatory holdout.
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
16. **Two scope definitions to settle with Mike.** Form uses a five-match window
    (`config.json`); the scope asks for "3 & 5 game rolling averages". Divergence is Form
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
