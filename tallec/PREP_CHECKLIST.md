# TALLEC Aug 1–31 Prep Checklist — Data-Independent Build

**Goal:** When Stats Perform player CSV arrives, everything is ready. Just plug in data and iterate ratings.

---

## P0: Database & Config (DONE ✓)

- [x] `tallec.db` schema extended with `competitions`, `positions`, `player_ratings`, `player_comparisons` tables.
- [x] `config.json` written with all rating formula weights, thresholds, update schedule.
- [x] `ingest_gerard.py` updated to create all tables on run.
- [x] Sample data (NRL R11–12, 356 players) seeded with mock ratings (ready to replace with real Stats Perform).

**Deliverable:** `tallec.db` + `config.json` — ready for import pipeline.

---

## P1: BOSC UI Skeleton (Aug 10–31)

### Spec & Design (Aug 10–14)
- [x] `P1_BOSC_UI_SPEC.md` written (4 tabs: Search, Benchmarks, Comparison, Trends).
- [x] Mockup layout sketched (Streamlit architecture, component list).
- [ ] Design system (CSS colors, spacing) — integrate with xLadder palette.
- [ ] Decide: Streamlit vs React. (Recommend Streamlit for speed + TALLEC data tables; React if need advanced viz.)

### Build Phase (Aug 15–28)
- [ ] `bosc_app.py` — Streamlit entry point + navigation.
- [ ] `pages/search.py` — Player search + profile tab (filters, stats table, ratings bars).
- [ ] `pages/benchmarks.py` — Positional benchmark tab (top 10 leaderboards, scatter plot, peer group stats).
- [ ] `pages/comparison.py` — NRL ↔ SL comparison tab (split view, translation callout, export CSV).
- [ ] `pages/trends.py` — Ratings trend tab (line charts, bench time indicator).
- [ ] `helpers/db.py` — Query functions (`get_player_profile`, `get_peer_group`, `get_comparables`).
- [ ] `helpers/ratings.py` — Mock rating generators (z-score logic, random stable distribution for R11–12 seed).
- [ ] `helpers/export.py` — CSV export utilities (comparison sheets, player profiles).
- [ ] `styles.css` — Dark theme config (embed in Streamlit `st.set_page_config()`).
- [ ] Test with mock data (356 NRL players, form/class 0–100 random, see if UI works).

### Integration with Real Data (Aug 29–31)
- [ ] Swap mock ratings → real Stats Perform calculations (weekly import pipeline).
- [ ] Test with first week of live imports (Stats Perform → `player_match_stats` → Form/Class calc).
- [ ] Screenshot + demo for Leeds (show comparison tab especially).

**Deliverable:** `bosc_app.py` + pages + helpers — ready to demo to Leeds Rhinos analyst.

---

## P2: Competition Translation Model (Aug 1–31 parallel)

### Transfer History Collection (Aug 1–15)
- [ ] Google Sheets: list all documented NRL ↔ SL transfers, 2015–2026.
  - Columns: Player, Position, Old Team (NRL), New Team (SL), Year, Form before, Form after, Notes.
  - Minimum target: 20 transfers with before/after stats.
  - Sources:
    - [Wikipedia: List of NRL players in Super League](https://en.wikipedia.org/wiki/)
    - [SuperLeague.com](https://www.superleague.co.uk/) archive + transfer news.
    - Stats Perform historical data (once Mike provides samples).
    - Personal interviews with Leeds analyst (recent signings they remember).

### Model Design (Aug 15–20)
- [ ] `transfer_dataset.csv` — finalized transfer list with pre/post form z-scores, position, age, injury rate.
- [ ] `models/competition_translation_fit.py` — Ridge regression fit code (input: form_z_pre, class_z_pre, position; output: translation_factor).
- [ ] `models/competition_translation_predict.py` — Real-time prediction code (+ confidence via kNN residuals).
- [ ] Retrospective validation: predict on held-out 5 transfers, check ranking.

### Integration into BOSC (Aug 20–31)
- [ ] Hook translation model into `pages/comparison.py`.
- [ ] Show prediction + confidence band on NRL ↔ SL comparison tab.
- [ ] Test with mock transfers (sample NRL → SL predictions for top 10 players).

**Deliverable:** `transfer_dataset.csv` + `models/*` code — ready to validate against real Leeds signings (Sep onwards).

---

## Last-Minute Tasks (Aug 25–31)

- [ ] Write integration guide: "How to add a new Stats Perform import" (for whoever maintains it after launch).
- [ ] Create Streamlit secrets template (expected CSV columns, database path, config location).
- [ ] Document position codes & benchmark groups (Fullback / Wing / Centre / Halves / Props / Back Row / Bench).
- [ ] Screenshot dashboard (4 tabs, sample players) for Leeds meeting.
- [ ] One-page summary: "BOSC Workflow for a Recruitment Analyst" (e.g., "Search NRL player → See form/class → Check translation model → Export comparison → Share with coaching staff").

---

## Data Waiting List (Blocked until Mike/Stats Perform provides)

| Blocker | Needed By | From | Unblocks |
| --- | --- | --- | --- |
| Stats Perform CSV format sample | Aug 5 | Mike / Stats Perform | Full ingest pipeline testing |
| Historical NRL ↔ SL transfer list | Aug 10 | Mike / Wikipedia / Leeds analyst | Competition translation model training |
| Player photos / team branding | Aug 20 (optional) | Stats Perform / SL / NRL | Player cards aesthetics in BOSC UI |
| Leeds analyst "pain points" list | Aug 15 | Mike / Leeds meeting | Feature prioritization for second iteration |

---

## Success Criteria (by Aug 31)

- [ ] **BOSC app is live** (Streamlit Cloud or localhost) with mock data, 4 fully functional tabs.
- [ ] **Competition translation model** is trained on ≥15 historical transfers, validated on ≥5 held-out transfers.
- [ ] **Weekly import pipeline skeleton** is written (awaits Stats Perform CSV to finalize).
- [ ] **Leeds analyst can use BOSC** to:
  1. Search for an NRL player.
  2. See Form/Class/Benchmark ratings.
  3. Get SL equivalent estimate + confidence.
  4. Export a comparison CSV.
- [ ] **Documentation is complete** (P1_BOSC_UI_SPEC.md, COMPETITION_TRANSLATION_SPEC.md, integration guide).

---

## Schedule (Rough)

```
Now (Jul 11)   │
               ├─── Jul 15: Collection of transfer history starts
               ├─── Jul 25: Transfer dataset frozen (20 transfers)
               ├─── Aug 1: Model fit + validation complete
Aug 10         │
               ├─── Aug 10: P1 design final
               ├─── Aug 15: P1 build starts
Aug 20         │
               ├─── Aug 20: P1 80% done (mockup working)
               ├─── Aug 25: Stats Perform sample CSV arrives (ideally)
               ├─── Aug 28: P1 final + live demo version
Aug 31         │
               └─── Sep 1: Leeds demo (BOSC + translation model)
```

---

## Owner / Contacts

- **Database & Config (P0):** Done.
- **BOSC UI (P1):** You (Tamas) — estimated 3–5 build days + 2 iteration days.
- **Competition Translation (P2):** You + Mike (transfer history collection needs Mike's knowledge).
- **Stats Perform Integration (P3, Sep onwards):** Depends on CSV format from Mike.
- **Leeds Analyst Feedback Loop (Sep onwards):** Mike as liaison, you iterate.

---

## Notes

- **Iterate quickly:** UI can change based on Leeds feedback (Sep). Model coefficients can shift as real data arrives. The *structure* (schema, API, config) is what matters now — data is fungible.
- **Hedge against delays:** If Stats Perform CSV is late, BOSC still launches with mock data Sep 1. Translation model still trains on Wikipedia transfers alone. Good enough to show.
- **Quality bar:** For a prototype to a professional club, *clear, honest about limitations, easy to iterate*. Not pixel-perfect, but functional.

---

Good luck! 🏉
