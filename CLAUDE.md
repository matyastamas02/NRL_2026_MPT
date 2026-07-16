# CLAUDE.md — xLadder / xLadder Pro

> Project onboarding for Claude Code. This file summarizes the whole xLadder rugby-league
> analytics project so a fresh Claude Code session understands the goal, the domain, the
> data, the models, the conventions, and the current state. Read it top to bottom before
> touching code. When you change something structural (a model, a data format, storage),
> update the relevant section here.

---

## 1. High-level overview

**xLadder / xLadder Pro** is a professional sports-analytics and betting-intelligence
platform for **rugby league** — primarily the **NRL** (National Rugby League, Australia)
and the **Super League (SL)**. It is a client-facing, commercial project built with a
partner (**Tamas**), and it extends an existing pipeline rather than being a greenfield
system.

The core idea (Moneyball for rugby league): raw results (win/loss, margin) are noisy — a
team can win on luck or lose while dominating. xLadder uses 50+ match-level team stats to
estimate a team's **underlying performance**, separates it from results, and points the
edge at the **betting market** instead of at roster construction.

Two core outputs per fixture:

- **Expected Winner (EW)** — which team the model predicts to win, with confidence.
- **Expected Margin (EM)** — by how many points the favoured team is expected to win.

The betting **edge** is defined as:

```
Edge = EM_model − EM_bookmaker_closing_line
```

A positive edge in the **underdog** direction is the highest-value signal the model can
produce. Clients combine model outputs with their own domain knowledge (player
availability, team news) — this is decision support, not a black box.

**Target performance:**

| Metric | Target | Benchmark |
| --- | --- | --- |
| Expected Winner accuracy | > 62% | Current human baseline: 62% |
| Expected Margin vs closing line | > 52.5% | Break-even threshold |

---

## 2. Domain glossary

Learn these terms — they appear constantly in code, chat, and docs.

- **DTS (Differential Team Score / Differential Team Stats)** — per-match difference of a
  stat between the two teams: `DTS_s = Stat_A,s − Stat_B,s`. The single most important
  transformation. Regressions run on DTS, **not** raw team stats. For "lower is better"
  stats (errors, penalties conceded), the sign is inverted so a positive DTS always means
  an advantage for Team A.
- **ATS (Advanced Team Stats)** — derived metrics (ratios, efficiencies) built on top of
  raw stats, e.g. tackle efficiency, kick-return index, errors per set.
- **Form** — short-term performance, rolling averages (3-game and 5-game windows) of the
  top-ranked DTS stats, blended.
- **Class** — long-term structural team strength via a **performance-weighted ELO**
  system (updates on *how* a team performed vs expectation using DTS, not just the
  scoreline). **K = 27.**
- **ELO ratings** — the Class rating mechanism.
- **ElasticNet** — regularised logistic regression used for the win/loss model
  (interpretable, feature-selecting).
- **Ridge regression** — used for margin prediction.
- **EW / EM** — Expected Winner / Expected Margin (see above).
- **xLadder** — model-based league standings ranked by underlying performance (vs the
  actual competition ladder). The project's namesake output.
- **Gerard** —the **Gerard scraper** pulls foxsports NRL data. The
  "Gerard format" is a distinct CSV format the pipeline parses. 
- **GIGOT** — a team-list system (GIGOT v3 team-list integration is out of MVP scope for
  the recruitment workstream).
- **PCM** — Post-Contact Metres.
- **z-score** — standardization used in feature/rating computation.
- **PCM, run metres, line breaks, tackle breaks, missed tackles** 
- **Contribution score / positional benchmark / competition translation factor** — terms
  from the newer Recruitment Intelligence workstream.
- **NRL / SL** — National Rugby League / Super League.

---

## 3. Pipeline architecture

One master table is the single source of truth: **one row = one match**
(`Team_A vs Team_B` + home/away/neutral flag). No downstream step reads any other source.

Stages (each has a domain-expert check-in before proceeding):

1. **Data cleaning** → build the master table. *(Master table confirmed.)*
2. **Feature engineering** → ATS (derived stats) + DTS (differentials). *(Variables agreed.)*
3. **Regression** → win/loss (ElasticNet logistic) + margin (Ridge/ElasticNet linear) on
   the DTS matrix. Output: a **ranked, named-coefficient** stat list. **Check-in #1.**
4. **Model build** → combine **Form** (3/5-game rolling) + **Class** (performance-weighted
   ELO). **Check-in #2.**
5. **Backtesting** → train 2023–2024, test 2025; expanding-window walk-forward within
   train. Compare EM vs bookmaker closing line. **Check-in #3.**
6. **Externalities & weighting** → home advantage, player-change toggle, State-of-Origin
   toggle; iterative stat-weight optimisation. Final model signed off.
7. **Outputs & documentation** → xLadder table, rolling rating charts, feature-importance
   (ElasticNet coefficients + SHAP), model-vs-actual margin scatter; codebase + how-to
   guide + plain-language write-up.

Combined model output:

```
EM = w_form · ΔForm + w_class · ΔClass + w_home · HomeAdv + ε_ext
```

Weights `w_form`, `w_class`, `w_home` are optimised on the training set to maximise EM
accuracy vs closing line, then applied unchanged to the 2025 test set.

---

## 4. Models

The live pipeline (`xladder_pipeline.py`, v3.0 in repo `NRL_2026_MPT`) supports variants:

- **M3 (original)** — 4 NRL stats / 10 SL stats (`NRL_STATS` / `SL_STATS`).
- **M3+ (enhanced)** — 12 NRL stats / 14 SL stats (`NRL_STATS_V2` / `SL_STATS_V2`),
  better regularisation.
- **Total** — Ridge regression for total points (`TOTAL_STATS`).

(An earlier **M4** variant using stats existed in previous iterations;
it is **not** in the current live pipeline.)

Methods in the live code: **LogisticRegression** (win/loss), **Ridge** (margin/total),
**performance-weighted ELO** (Class, K=27). Form uses rolling 5-match (and 3-match)
averages with `shift(1)` leakage prevention. ElasticNet was used in the research phase
for feature selection (that's how the model stat lists were chosen); the deployed app
runs plain logistic + Ridge on the selected stats.

**Outcome source:** the pipeline derives Win/Margin/Total from `A_Points Scored` /
`B_Points Scored` (falls back to renaming `A Score`/`B Score` only if those are absent) —
so those columns must always hold the true final scores.

---

## 5. Data sources & formats

- **NRL match statistics** — 50+ team-level stats per match, seasons 2022–2025 (source
  raw: 2023/2024/2025 season + fixtures xlsx; see `NRL.md` notebook for the cleaning /
  master-table build logic — team-code mapping, opponent join via Match ID, differential
  columns, home-advantage flag from stadium mapping).
- **Historical betting data** — 3,421 matches, 2009–2025, opening + closing odds and line.
  (`aussportsbetting.com`.)
- **Gerard pipeline** — foxsports-scraped NRL data; canonical source. Pipeline has
  dedicated parsers `parse_gerard_round` and `parse_gerard_odds`. Accepts **both** master
  format and Gerard format CSVs.
- **The Odds API** — live odds; sport key `rugbyleague_nrl`, 11 Australian bookmakers.
- **External ELO init** — `aussportstipping.com/sports/nrl/elo_ratings/`, `pythagonrl.com`.

**Master format (actual):** one row = one match; columns `A_{stat}`, `B_{stat}`,
`Diff_{stat}` (= A − B) for ~400 stats, plus `Match ID`, `Season`, `Round`, `A Team`,
`B Team`, `A Score`, `B Score`, `Home Advantage` (`A`/`B`/`neutral`). NRL uses full team
names in `A Team`/`B Team`; SL uses short codes (BD, C, CF, HF=Hull FC, HFC, HKR, LH,
LS, SH, TL, WA, WFT, WI, YK).

**Match ID convention (actual):** `{Season}-{Round}-{CodeA}-{CodeB}` (e.g.
`2026-1-CR-MWSE`). Codes are **not** alphabetically sorted; since the July 2026 rebuild,
Team A = the home team for all 2026 rows (Home Advantage `A`, or `neutral` for
Vegas-type games).

**Super League 2026 note:** SL 2026 required a full R1–R7 rebuild because two new teams —
**Bradford Bulls (BD)** and **York RLFC Knights (YK)** — are absent from historical master
data. Possession % (always sums to 100% per match pair) is a reliable signal for pairing
team stats when reconstructing historical rounds.

**European decimal separators** appear in some source files — handle commas as decimals.

---

## 6. Storage & stack

- **Languages/frameworks:** Python, Streamlit (8-tab app: Dashboard, xLadder, Team
  Stats, Betting, Bet History, Model, Weekly Input, Players), Plotly, Matplotlib.
- **ML libs:** scikit-learn (LogisticRegression, Ridge), pandas.
- **Repo:** `github.com/matyastamas02/NRL_2026_MPT` — `app.py`, `xladder_pipeline.py`,
  `NRL_master.xlsx`, `SL_master.xlsx`, `requirements.txt`.
- **Storage (primary):** **Google Sheets** via `gspread` service account, read/written
  by the app at runtime (credentials in Streamlit secrets: `gcp_service_account` +
  `SHEET_ID`/`NRL_SHEET_ID`/`SL_SHEET_ID`).
  - Spreadsheet "xladder data", ID: `1afnVBY5ZSMPCwuClUOafD6q6L6vfKquBKRVqhhvELtg`
  - Tabs: `NRL_master`, `SL_master`, `bets`
  - Service account: `xladder-app@light-rhythm-494915-e0.iam.gserviceaccount.com`
    (key JSON lives locally in Downloads — never commit it)
- **Storage (fallback):** the two master xlsx files in the repo — used only when Sheets
  is unreachable. Keep them roughly in sync with the sheet after big updates.
- **Deployment:** Streamlit Cloud (read-only filesystem — see gotchas). Live app:
  `cgbkzxztsv7hu7geytzavc.streamlit.app`.
- **Weekly flow:** client enters/uploads a round in the app's Weekly Input tab → app
  runs the pipeline → writes the updated master back to the Sheet (no GitHub upload
  needed, contrary to the old README description).
- **Sheet write helpers:** after writing to Sheets, cache must be invalidated (see
  gotchas).

---

## 7. Critical conventions & rules

- **⚠ Leakage rule (highest-risk failure mode):** every feature value for match *M* must
  be computed **only** from matches 1…*M*−1. Never use match *M*'s own data in its
  prediction.
  - Rolling Form: compute on the full log, then `.shift(1)` **before** the join.
  - Class/ELO: attach the **pre-match** rating; run the ELO update only after the result
    is recorded, attach the updated value to *M*+1 onward.
  - Sanity check: randomly permuting the outcome column must leave all feature values
    unchanged. If any feature changes, there is leakage.
- **Train/test split:** train 2023–2024, test 2025. **Strict — no bleed.** 2025 is a
  sealed envelope until the model is finalised.
- **Regress on DTS, not raw stats.** Invert sign for "lower is better" stats.
- **Design around Gerard-available stats** where possible .
- **Data availability is the primary constraint** on new features — not coding
  complexity. Check data first before scoping.

---

## 8. Current state & workstreams

**xLadder Pro (live):** multi-tab Streamlit app, Google Sheets persistence, gspread
service-account backend. M3/M3+/Total in the core pipeline. Odds API integration live.

**Data state (as of 2026-07-09):** both masters rebuilt from Mike's full-season
"Model Stats Restarted" CSVs (July 2026) — NRL 2026 complete through **Round 17**
(128 matches), SL 2026 through **Round 16** (111 matches, Bradford BD and York YK now
properly included). All 2026 scores validated against official results; home/away and
neutral flags set from fixture records. Sheet backup taken before the rewrite
(`Downloads/sheet_backup_20260709/`).

**TALLEC (current client brief, July 2026):** prototype of "Total All Expected
Contributions" — the player-level intelligence engine. Three phases:
1. **TALLEC Database** — Stats Perform player+team data from NRL, SL, NSW Cup,
   Queensland Cup; permanent player IDs; raw storage; derived metrics; weekly imports.
2. **BOSC** — recruitment/player-intelligence prototype for a **Leeds Rhinos** demo:
   0–100 positional benchmarks (50 = average), player Class/Form/Divergence ratings,
   **competition-translation model** (the key deliverable: how would an NRL / NSW Cup
   player go in SL), Streamlit search+profile UI.
3. **GIGOT v2** — TALLEC ratings feed the internal predictive model: 5 inputs
   (Team Form, Team Class, Player Form, Player Class, Contribution Rating =
   player stats as % of team stats → Expected Contribution from team lists).

Scope doc: `TALLEC_Scope_of_Work_for_Tamas_v1.docx` (Mike, 2026-07-03). Quote drafted
2026-07-09 (75–90h across phases, SQLite storage, config-driven rating formulas,
vertical slice on SL first). Mike meets the Leeds analyst Friday evening AUS time to
define the framework. The old "Recruitment Intelligence Platform" brief has been
superseded by/absorbed into TALLEC. Data dependency: Mike provides Stats Perform
exports — request sample files per competition before building.

---

## 9. Gotchas / previously resolved issues

Watch for these — they've bitten the project before:

- **pandas 2.x:** `errors="ignore"` was removed — don't use it.
- **European decimal separators** in source data.
- **Streamlit duplicate element keys** — every widget needs a unique key.
- **Streamlit Cloud read-only filesystem** conflicts — don't write to local disk in
  deployment; use Google Sheets.
- **Cache invalidation after Google Sheets writes** — fixed via `session_state`
  `force_reload` + `st.cache_data.clear()`.
- **Plotly axis dict conflicts** — fixed via a merged-dict helper `at()`.
- **"Points Scored" in Stats-Perform-style exports undercounts** the real final score
  by 2–4 points in ~10% of matches (penalty goals / field goals missing from the
  aggregate). Always validate scores against official results before writing them into
  `A_Points Scored`/`A Score` — one 2026 SL match (R13 WA–HKR) had its **winner**
  flipped by this.
- **Exports have no venue column** — home/away must come from fixture records; row
  order in the CSVs does not encode the home team.
- **ELO is STORED DATA in the master** (`ELO_A`/`ELO_B`/`Diff ELO` = pre-match
  ratings). `run_pipeline` does **not** recompute it for loaded rows — only the weekly
  `append_new_round` path extends it. Any bulk rebuild of rows MUST fill these columns
  (via `update_elos_for_new_matches`), otherwise `Diff ELO` — the model's dominant
  feature (coef ≈ 0.54) — silently becomes 0 and accuracy collapses toward the
  intercept (~52%). This happened on 2026-07-09 and was fixed same day.
- **Making the repo PRIVATE breaks Streamlit Cloud deploy.** Streamlit Cloud clones
  the repo on every startup; if the repo is private and the Streamlit GitHub App/OAuth
  connection lacks access, it fails with `🐙 Failed to download the sources ... Make sure
  the repository and the branch exist and you have write access` and the app shows
  "Oh no. Error running app." This is NOT a code/deps/secrets issue — no push or
  reboot helps until access is restored. Fix: github.com/settings/installations →
  Streamlit → grant Repository access to `NRL_2026_MPT` (or make repo public), then
  Manage app → Reboot. (Happened 2026-07-12 after the repo was made private for IP
  protection; the actual cause was only visible in the Manage-app logs.)
- **SL historical ELO columns were decimal-corrupted** (values like `1906483` =
  1906.483 with the separator lost, mixed ×100/×1000) — the SL model effectively ran
  without ELO until 2026-07-09, when the full SL ELO history was recomputed from
  results (start 2000 flat in 2022, K=27, season regression 0.30). SL 2026 accuracy
  jumped from ~54% to ~71% once train + predict both had clean ELO.

---

## 10. Security

- The client's **Odds API key was shared in chat and should be regenerated** at
  the-odds-api.com. Never commit API keys or the gspread service-account JSON to the repo;
  use environment variables / Streamlit secrets.

---

## 11. How MPT likes to work

- **Concise, direct, realistic scoping.** Push back on over-engineering and inflated
  estimates. Time estimates should reflect AI-assisted development velocity.
- **Iterative and fast-moving** — features built and refined within single sessions.
- Build dedicated parsers for new data formats (e.g. Gerard) into the existing pipeline
  rather than restructuring the core.
- Betting edge = model-implied probability vs market-implied probability; surface the
  strongest signals as underdog H2H value bets.
- The model is one input layer; clients layer in external knowledge. Keep outputs
  interpretable and auditable.

---

## 12. Final deliverables (definition of done)

- Full, documented, modular Python codebase: data cleaning → feature engineering →
  regression → model → backtest.
- How-to guide for weekly updates using the same data format.
- Plain-language methodology write-up: what the model found, what drives predictions, and
  where it is / isn't reliable.
