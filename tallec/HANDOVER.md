# TALLEC — handover

Written 2026-08-25 at the end of a working session, for whoever picks this up next
(including a fresh assistant session). `README.md` is the operating manual;
`GIGOT_V2_SPEC.md` and `AUS_DATA.md` are the method and data documents. This file is
the state of play, the open items, and the traps.

---

## 1. Where things are

| What | Where |
| --- | --- |
| Working copy | `C:\Users\matyas-peter.tamas\Downloads\TALLEC\` — **no git remote**, 1 stale commit |
| Deployed copy | `Downloads\NRL_2026_MPT\tallec\` — this is the versioned, canonical one |
| GitHub | `github.com/matyastamas02/NRL_2026_MPT`, branch `main`, **public** |
| Live app | https://bosc-tallec.streamlit.app — redeploys automatically on push to main |
| Source data | `Downloads\TALLEC all Aus Data.xlsx`, `TALLEC_Super_League_Master_2021_2026_positions_complete.xlsx`, `Player Level Stats NRL.xlsx`, `sl21_players\` |
| Match masters | `Downloads\NRL_2026_MPT\NRL_master.xlsx` and `SL_master.xlsx` (xLadder, not TALLEC) |

**The two copies are synchronised by hand.** Work in `TALLEC\`, copy the changed files
into `NRL_2026_MPT\tallec\`, run the checks *there*, then commit and push. Every deploy
this session needed 10–26 files copied manually; making `TALLEC\` a clone of the deploy
repo would remove the step and the risk.

Client documents (private artifacts, shared from their own share menu):

- **What Is Live** — status note: https://claude.ai/code/artifact/3ddd7f4c-014a-4819-8008-e5ddfb0d6655
- **Reading the Numbers** — how to read a rating: https://claude.ai/code/artifact/769e147b-3aaa-475a-8e1b-418f5c5f9e8b
- **Four Leagues, One Scale** — the workings: https://claude.ai/code/artifact/346b0b8e-27b1-460e-984c-8c446ed0bc63

Their sources are `what_is_live.html`, `reading_the_numbers.html`, `session_writeup.html`
in `TALLEC\`. They are **not** in the repo, deliberately: it is public and they contain a
recruitment shortlist with named players and a frank assessment of the xLadder Super
League model.

---

## 2. State as of this handover

`tallec.db` — 89 MB, **122,359 player-match rows**, four competitions, 23 league-seasons,
**1,873 rated players**, all four rated within position group.

| Competition | Seasons | Rows | Rated | Position source |
| --- | --- | --- | --- | --- |
| NRL | 2020–2026 | 44,961 | 466 (2026) | match sheet 2020 only; 2021–2026 estimated |
| Super League | 2021–2026 | 31,763 | 417 (2026) | match sheet, every season |
| NSW Cup | 2021–2025 | 21,427 | 496 (2025) | match sheet |
| Queensland Cup | 2021–2025 | 24,208 | 494 (2025) | match sheet (91%) |

The measured competition ladder, relative to the NRL: **Super League −4.3 points, NSW Cup
−5.8, Queensland Cup −7.6**, fitted from 953 within-player moves, χ²/dof 0.15.

Match-model result: the player layer is worth **+0.27 MAE [+0.08, +0.46]** on the NRL over
752 out-of-sample fixtures (significant), +0.21 [−0.27, +0.67] on Super League (not).
The useful signal is squad inexperience, not form.

Verification: `python -m pytest tests -q` → 31 pass. `python smoke_bosc.py` → all six
pages for all four competitions.

Commits this session: `f14fe29` (four competitions), `4aea993` (external-review fixes),
`e811a80` (guarded writes, audit, provenance, tests), `9f203ef` (the post-contact-metres
availability bug).

---

## 3. Open items

### Waiting on Mike

1. **NRL 2026 positions**, joined row by row the way the Super League master was. The last
   position gap: 2026 NRL positions are estimated (3,033 rows from a metadata file, 1,754
   from players' Australian careers, 134 unknown). **Run the four acceptance checks on
   arrival** — see §4.
2. **Metric dictionary sign-off.** All 341 rows sit at `Decision=Review`; ten are low
   confidence and one (`Charge Down`) divides by something not in the data. Nothing in the
   Volume + Rate layer can be finalised without it. This is a rugby judgement.
3. **Team lists**, even manually typed — a list of names per club per round. Everything
   about availability is currently measured on the seventeen who actually played, which is
   only known afterwards.
4. **2026 for NSW Cup and Queensland Cup** — the Australian history file stops at 2025, so
   those ratings describe last season.
5. **Confirm one row**: the Ben Talty reassignment gives him a round-20 Capras appearance,
   while his other Queensland Cup 2025 rows are Burleigh Bears. Done as instructed; worth
   a sentence back.

### Ours, not started

6. **The real team-list backtest** (P2). Everything measured so far uses the actual line-up
   and post-match minutes, which is an upper bound. Whether the +0.27 survives on
   Friday-morning team sheets is the single question that would most change what can be
   claimed. Needs item 3.
7. **Position-specific ratings** (P3) — a player who covers hooker and bench gets one
   rating against his most common position. Needs item 1 first.
8. **Analyst workflow and shortlist export** (P3) — the flow Leeds would actually use:
   filter, shortlist, export.
9. **Generated data-state block in the READMEs** (P3), so they cannot go stale again. Both
   went stale twice this session.
10. **Regenerate the Super League master's stored margin predictions out-of-sample.** An
    xLadder issue rather than a TALLEC one, but the Super League app reports its accuracy
    from that column and it is in-sample for 2025 (training MAE 15.69, held-out 7.63).
11. **The database is 89 MB in a public repo.** Under GitHub's hard limit, over its
    recommendation, and it is licensed Stats Perform data including dates of birth. The
    user has decided to leave it public for now. Options if that changes: drop
    `player_match_raw` from the deployed copy (the app does not read it), or make the repo
    private and grant the Streamlit GitHub App access.

---

## 4. Traps — read before touching anything

**Validate any position column before loading it.** The first Super League position file
was misaligned and would have silently corrupted every rating. Four checks separate real
from misaligned, and the broken file failed all four:

| Check | Real match-sheet data | The broken file |
| --- | --- | --- |
| Position values vs data rows | equal | 983 spare |
| Mean minutes by position | props ~46, wingers ~78 | all 62–66 |
| Tackles per minute, spread | ~9–10× | 1.3× |
| Passes per minute, spread | ~30× (hookers highest) | 1.8× |
| Distinct positions per player | ~1.5 | 1.92 |

Also worth a cross-check against the independent Australian career record: the corrected
file agreed on 74% of players, the broken one on 12%.

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
`_backups/` holds eight snapshots, 450 MB, git-ignored.

**A stat a season never recorded must not be scored as average.** Post-contact metres only
exist from 2025 and carried 24% of a prop's weight in every earlier season. The engine now
drops rates below `min_rate_coverage` and renormalises. If a new feed adds or removes a
column, check `eng.dropped` per season before trusting cross-season numbers.

**Position coverage gates the rating mode.** A competition is rated within position groups
only above 90% coverage (`min_position_coverage`); below it, the whole competition is the
peer group, because a partly-known pool makes two identical performances score differently.
`player_ratings.rating_basis` records which applied and the app states it in words.

**`HF` is Huddersfield and `HFC` is Hull FC.** The project notes say `HF = Hull FC`.
`team_map.py` solves the codes from scores and fixtures rather than from documentation, and
validates the master's columns — an earlier version searched by filename and silently found
a four-month-old master in the parent folder.

**Competitions do not share a current season.** NSW Cup and Queensland Cup stop at 2025;
NRL and Super League run to 2026. Use `season_of(comp)` in the app,
per-competition in `regenerate_full.py`.

**The Stats Perform Player ID is global** across competitions — verified on 169 shared ids
with 100% name and 98% date-of-birth agreement. But feeds vary: some carry it, some do not,
and pandas reads it as float when the file has blank rows, which turns `24528` into
`24528.0` and forks every identity. Always go through `sp_schema.normalize_player_id`.

**Interchange is a role, not a position.** A player's career position is the mode of his
*starting* positions (`sp_schema.primary_position`). Getting this wrong moved five bench
forwards to "Interchange" on the first weekly import.

---

## 5. Running things

```bash
python -m pytest tests -q                 # 31 tests, synthetic data, safe any time
python smoke_bosc.py                      # all six app pages, all four competitions
python -m streamlit run bosc_app.py       # the app locally

python weekly_update.py --file "SL26 Players.csv" --competition SL --season 2026 --dry-run
python weekly_update.py --file ... --competition SL --season 2026     # for real
python regenerate_full.py                 # ratings + contribution, all competitions
python fit_translation_v2.py              # the ladder and the translation model
python gigot_v2.py                        # the match-model evaluation
python runtime.py                         # provenance and audit counts
python runtime.py restore                 # undo the last guarded write
```

Deploy: copy the changed files into `NRL_2026_MPT\tallec\`, run the two checks there,
`git add -A && git commit && git push origin main`. Streamlit redeploys itself; a cold
start with an 89 MB database takes a few minutes.

---

## 6. How the client conversation stands

Mike has been sent nothing yet from this session — the three artifacts are private until
shared. The status note (**What Is Live**) is written to be the first thing he opens: it
names the app, what changed, what the code review found, and the three asks. Sending that
plus the other two links, or the three HTML files as attachments, is the next client-facing
step.

The commercial item is still open and unrelated to the code: the repo LICENSE names joint
ownership with no written revenue share. One page — who owns what, what percentage, what
happens on a split — is an easy conversation now and a hard one after a club signs.
