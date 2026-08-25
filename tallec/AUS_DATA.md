# Australian Player Data - `TALLEC all Aus Data.xlsx`

Source file supplied by Mike, 100 MB, received 2026-08-19. Stats Perform
player-match exports for the three Australian competitions. This document records what is
in the file, what is verified, what is broken, and what it unlocks for TALLEC. Every
figure below was computed from the file itself; nothing is estimated.

## 1. What arrived

**85,675 player-match rows** across **16 sheets** - 87,425 player-hours,
2,374 distinct players, 5,049 team-match records. One row =
one player in one match. 340 columns per sheet.

| Sheet | Competition | Season | Rows | Players | Rounds | Teams | `Position` column |
|---|---|---|---|---|---|---|---|
| NRL20 | NRL | 2020 | 5,204 | 464 | 1-24 | 16 | yes |
| NRL21 | NRL | 2021 | 6,796 | 493 | 1-29 | 16 | **no** |
| NRL22 | NRL | 2022 | 6,826 | 476 | 1-29 | 16 | **no** |
| NRL23 | NRL | 2023 | 7,224 | 502 | 1-31 | 17 | **no** |
| NRL24 | NRL | 2024 | 7,224 | 508 | 1-31 | 17 | **no** |
| NRL25 | NRL | 2025 | 6,766 | 479 | 1-31 | 16 | **no** |
| NSW21 | NSW Cup | 2021 | 2,207 | 425 | 1-15 | 11 | yes |
| NSW22 | NSW Cup | 2022 | 3,102 | 380 | 1-28 | 9 | yes |
| NSW23 | NSW Cup | 2023 | 5,506 | 561 | 1-30 | 13 | yes |
| NSW24 | NSW Cup | 2024 | 5,518 | 570 | 1-30 | 13 | yes |
| NSW25 | NSW Cup | 2025 | 5,094 | 515 | 1-30 | 12 | yes |
| QLD21 | Queensland Cup | 2021 | 3,806 | 495 | 1-22 | 14 | **no** |
| QLD22 | Queensland Cup | 2022 | 4,260 | 474 | 1-24 | 14 | **no** |
| QLD23 | Queensland Cup | 2023 | 5,356 | 509 | 1-26 | 15 | yes |
| QLD24 | Queensland Cup | 2024 | 5,399 | 516 | 1-27 | 15 | yes |
| QLD25 | Queensland Cup | 2025 | 5,387 | 497 | 1-27 | 15 | yes |

NRL 2020 is a short season (24 rounds, COVID); NSW Cup 2021 was cut to 15 rounds. Round
numbers run past the regular season, so finals are included.

**This is history, not an update.** The file covers NRL **2020-2025** and contains no 2026
data. `tallec.db` already holds NRL 2026 and SL 2021-2026, so the two are
complementary - together they make NRL a continuous 2020-2026 record.

Currently in `tallec.db`:

| Competition | Season | Rows |
|---|---|---|
| NRL | 2020 | 5,204 |
| NRL | 2021 | 6,796 |
| NRL | 2022 | 6,826 |
| NRL | 2023 | 7,224 |
| NRL | 2024 | 7,224 |
| NRL | 2025 | 6,766 |
| NRL | 2026 | 4,921 |
| NSW | 2021 | 2,207 |
| NSW | 2022 | 3,100 |
| NSW | 2023 | 5,506 |
| NSW | 2024 | 5,518 |
| NSW | 2025 | 5,094 |
| QLD | 2021 | 3,806 |
| QLD | 2022 | 4,260 |
| QLD | 2023 | 5,356 |
| QLD | 2024 | 5,399 |
| QLD | 2025 | 5,373 |
| SL | 2021 | 4,673 |
| SL | 2022 | 5,663 |
| SL | 2023 | 5,652 |
| SL | 2024 | 5,667 |
| SL | 2025 | 5,632 |
| SL | 2026 | 4,476 |

## 2. Schema

**339 of the 340 columns are identical across all 16 sheets.** The only variation
is a single column: sheets that carry `Position` do not carry `Opta Live`,
and vice versa. There is no other structural difference between competitions or seasons -
unusually clean for a multi-season, multi-competition export.

Fit against the existing TALLEC configuration:

- **Metric dictionary:** 335 of 340 dictionary rows
  (99%) match a column in this file by exact name. The
  5 that do not are the BOSC-derived metrics, which by definition have no
  source column: `Creativity Value`, `Earned Run Metres`, `Involvement Volume`, `Metres per Carry`, `Off-ball Volume`.
- **`player_match_raw`:** 341 columns in common. The file adds
  three the database does not have: `Age`, `Games Played`, `Position`.

So the metric dictionary - including the Volume + Rate design and its 341 rows awaiting
sign-off - applies to this data as-is. No re-mapping needed.

Columns present in the file but absent from the dictionary: `Age`, `Ball Run Metres / Kick Return`, `Date of Birth`, `Games Played`, `Player ID`, `Position`. These are
identity and metadata fields rather than metrics, so the omission is correct, but
`Position` and `Age` now need a place in the rating engine.

## 3. Player identity - solved

`Player ID` is populated on **100% of rows in every sheet** and is stable across
competitions and seasons, so a player can be tracked from Queensland Cup to NRL without
name matching. `Date of Birth` is present on 96.1% of rows.

This closes the identity problem that forced name-slug keys in the earlier ingest. Names
alone were never safe: several sheets carry more distinct `Player ID` values than
distinct names, i.e. two different players share a name.

## 4. Position - the gap is closed for Australia

Position was the standing blocker: the earlier feed had no position column, so positional
benchmarks fell back to competition-relative, and the NSW-Cup metadata join reached only
about 62% of NRL rows by name.

Two findings change that.

**Position here is a per-match jersey assignment, not a static player attribute.** The
totals prove it - exactly two Wingers, Centres, Props and Second Rows per team-match
against exactly one Full Back, Half Back, Five-Eighth, Hooker and Lock:

| Position | Player-match rows |
|---|---|
| Interchange | 9,984 |
| Centre | 5,040 |
| Winger | 5,040 |
| Prop | 5,039 |
| Second Row | 5,035 |
| Half Back | 2,521 |
| Lock | 2,520 |
| Full Back | 2,520 |
| Five-Eighth | 2,519 |
| Hooker | 2,519 |

This is better than a primary position: a player who covers hooker one week and comes off
the bench the next is recorded correctly in each match.

**Because `Player ID` is universal, position backfills across sheets.** Taking each
player's most common starting position (Interchange used only when he never starts) lifts
coverage from **50% to 97% of all rows**:

| Competition | Season | Own `Position` % | After Player-ID backfill % |
|---|---|---|---|
| NRL | 2020 | 100 | 100 |
| NRL | 2021 | 0 | 99 |
| NRL | 2022 | 0 | 98 |
| NRL | 2023 | 0 | 99 |
| NRL | 2024 | 0 | 98 |
| NRL | 2025 | 0 | 98 |
| NSW Cup | 2021 | 100 | 100 |
| NSW Cup | 2022 | 100 | 100 |
| NSW Cup | 2023 | 100 | 100 |
| NSW Cup | 2024 | 100 | 100 |
| NSW Cup | 2025 | 100 | 100 |
| Queensland Cup | 2021 | 0 | 63 |
| Queensland Cup | 2022 | 0 | 83 |
| Queensland Cup | 2023 | 100 | 100 |
| Queensland Cup | 2024 | 100 | 100 |
| Queensland Cup | 2025 | 100 | 100 |

NRL 2021-2025 carries no `Position` column at all, yet ends up 98-99% covered
because the same players appear with positions in NRL 2020, NSW Cup or Queensland Cup. Only
Queensland Cup 2021 (63%) and 2022 (83%) stay partial - those players largely never appear
in a sheet that has positions.

Note the two-tier quality this creates: on the 9 sheets with a `Position` column the
value is the true per-match assignment, while on the other 7 it is a career-mode estimate.
Any position-relative rating should record which of the two it used.

## 5. Data quality

Verified clean:

- **Scoring is internally consistent on 100.00% of rows.** `Points Scored`
  equals 4*tries + 2*conversions + 2*penalty goals + 1*one-point field goals + 2*two-point
  field goals for every row in the file. This is the arithmetic the SL 2026 feed failed.
- **Squad sizes and minutes are plausible.** Mean 17.0 players per team-match
  (rugby league names 17) and mean 1039 team-minutes against a theoretical
  1,040 (13 players x 80 minutes).
- **No zero-minute rows.** Every row is a player who actually took the field.

Flagged, with exact scope:

- **16 rows (0.02%) break the
  `Player ID + competition + season + round` key** - 3 player IDs
  (23487, 25090, 58001) appear for two different clubs in the same round, including 80
  minutes for both, which is impossible. One pair is an exact duplicate row. These must be
  resolved or dropped before the key can be enforced; either the source duplicated them, or
  one ID covers two people.
- **8 rows carry an implausible date of birth.** One player is
  recorded as born 1974, i.e. 49 years old in NSW Cup. (A 38-year-old in Queensland Cup
  checks out - Adam Cuthbertson genuinely played that late.)
- **`Date of Birth` is missing on 3,375 rows
  (3.9%)**, almost all in NSW Cup and Queensland Cup. Any age-curve
  work is NRL-solid and feeder-partial.
- **8.6% of team-matches name other than 17 players** (mostly 16 or 18). Some of
  this is legitimate - the 18th-man rule, sin-bins - but it is worth a spot check before team
  totals are used as denominators, because the Rate design divides by them.
- **840 rows exceed 80 minutes** (max 94). Extra time and golden
  point, so per-minute rates are correct, but any "share of 80 minutes" logic is not.

Not verified: **scores against official results.** The file is internally consistent, which
is not the same as correct. The known Stats Perform undercount bit this project before - a
2026 SL match had its winner flipped - so before any of these team totals feed a match model,
they need a pass against official results.

## 6. What the data unlocks

**A real competition-translation training set.** The model currently ships fitted on 16
NRL-SL transfers. This file contains **726 player-seasons where the same player
played at least 3 matches in both a feeder competition and the NRL in the same season**
(437 distinct players), plus 405 feeder-season to
next-season-NRL step-ups with 5+ matches on each side. Same player, same season, two
competitions is the cleanest identification available for a level difference - it holds the
player fixed instead of comparing populations.

The naive within-player read on that sample:

| Metric | Feeder (per min) | NRL (per min) | NRL / feeder | Within-player correlation |
|---|---|---|---|---|
| Run metres / min | 1.3181 | 1.3157 | **0.998** | 0.78 |
| Post-contact metres / min | 0.0999 | 0.1031 | **1.032** | 0.95 |
| Tackles made / min | 0.2858 | 0.3692 | **1.292** | 0.93 |
| Tackle breaks / min | 0.0347 | 0.0286 | **0.823** | 0.58 |
| Line breaks / min | 0.0050 | 0.0041 | **0.813** | 0.35 |
| Errors / min | 0.0113 | 0.0102 | **0.906** | 0.30 |

Two effects are already visible and they point in opposite directions: a player makes
**29% more tackles per minute** in the NRL than in the feeder
competitions, while his **tackle breaks and line breaks fall by roughly
18%**. Defensive workload rises, attacking cut-through drops. Run
metres and post-contact metres travel almost unchanged (ratios 0.998 and
1.032), and post-contact metres are the most stable signal of the six
(within-player correlation 0.95) - i.e. the metric that survives a level change
best.

Treat these as measured descriptive ratios, not as the model. They are unadjusted for
minutes, position, age or opposition strength, and a player picked for NRL duty is not a
random draw from the feeder pool. The point is that the sample is now large enough to adjust
for those things.

**Player Class with real history.** Six NRL seasons
(40,040 rows) replace a single partial season, so Class
ratings can build over years rather than weeks, and a genuine train/test split becomes
possible.

**Age curves.** Date of birth on 96.1% of rows, mean age
25.3, makes peak-age and development-curve work feasible for the first time -
the core recruitment question of whether a player is still improving.

**Position-relative benchmarks for Australia**, per section 4.

**A feeder-league scouting layer.** 1,234 NSW Cup and 1,188
Queensland Cup players, 668 of whom also appear
in the NRL - the population a club actually recruits from.

## 7. What it does not unlock

- **No Super League data.** Leeds is a Super League club and the demo runs on Super League.
  This file improves the Australian side of every cross-league question; the SL side still
  rests on `tallec.db`'s SL 2021-2026.
- **No SL positions.** Section 4 fixes Australia only. Super League rows stay
  `Unknown` until a Super League position source arrives, so SL positional benchmarks
  remain competition-relative.
- **No 2026.** Weekly in-season updates still need the 2026 feed.
- **No team-level match data**, no venue and no result column - home advantage and match
  outcomes still come from the xLadder masters.

## 8. Open questions for Mike

1. The 3 duplicated player-rounds (section 5) - source error, or one ID covering
   two people?
2. Is a **Super League** equivalent of this file available, with `Position` and
   `Player ID`? That single file would remove the last structural gap in the Leeds
   demo.
3. Queensland Cup 2021-2022 carry no `Position` column and backfill only to 63% /
   83%. Does a position source exist for those two seasons?
4. `Opta Live` appears on exactly the 7 sheets that lack `Position`. Is it a
   flag we should read, or an artefact of how the export was assembled?

## 9. Load status

Loaded into `tallec.db` by `ingest_aus_history.py` on 2026-08-19 —
85,659 of the 85,675 rows, the 16 key-breaking rows excluded and kept in the
`excluded_rows` table. The database now holds **122,343 player-match rows**
and is 78 MB.

| Competition | Seasons | Player-match rows | Distinct players |
|---|---|---|---|
| NRL | 2020-2026 | 44,961 | 927 |
| NSW Cup | 2021-2025 | 21,425 | 1,234 |
| Queensland Cup | 2021-2025 | 24,194 | 1,188 |
| SL | 2021-2026 | 31,763 | 1,103 |

The Stats Perform Player ID turned out to be **global**: of the 137 SL 2026 players who
also appear in this file, 133 carry an identical ID. Earlier loads had keyed some rows by
name-slug because their source files carried no ID column, so the load re-keys those to the
permanent ID - 513 keys, 67 of them confirmed by matching date of birth, and
13 candidate matches rejected because the dates of birth disagreed. Every change is
recorded in the `player_id_map` table. NRL 2025 and NRL 2026 now share
364 players under one ID, where before the re-key they shared none.

Position coverage after the backfill, and the standardization mode each competition
therefore gets:

| Competition | Position known | Standardization |
|---|---|---|
| NRL | 98% | position-relative |
| NSW Cup | 100% | not rated |
| Queensland Cup | 91% | not rated |
| SL | 8% | competition-relative |

Super League stays competition-relative: only its players who have also played in
Australia carry a position, and standardizing within position group on a partially known
pool would make the composite incomparable across players. The rating engine gates on
`min_position_coverage` in `config.json` (0.90) and records the outcome in
`player_ratings.rating_basis`.
