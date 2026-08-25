# xLadder Pro — NRL / Super League analytics

> © 2026 Tamás Mátyás-Péter, Michael Robert Wood & Ádám Asztalos. All rights reserved — proprietary
> software, see [LICENSE](LICENSE). Not open source; no use or redistribution
> without written permission.

Rugby-league analytics and betting-intelligence platform. Estimates each team's
underlying performance from ~400 match-level stats, separates it from raw results, and
produces per-fixture **Expected Winner** and **Expected Margin** plus a model-based
league table (the *xLadder*).

**Live app:** Streamlit Cloud · **Data:** Google Sheets (primary) with the xlsx files
in this repo as fallback.

## Repo layout

| Path | What it is |
| --- | --- |
| `app.py` | Streamlit app (8 tabs: Dashboard, xLadder, Team Stats, Betting, Bet History, Model, Weekly Input, Players) |
| `xladder_pipeline.py` | Core pipeline v3.0 — models M3 / M3+ / Total, ELO (K=27), Form, backtests |
| `NRL_master.xlsx`, `SL_master.xlsx` | Master match tables (one row = one match), local fallback for the Sheets data |
| `docs/` | Project documentation and client briefs (TALLEC scope of work) |
| `tallec/` | **TALLEC / BOSC** — player-level ratings, positional benchmarks and the measured competition-translation ladder across NRL, Super League, NSW Cup and Queensland Cup. Its own Streamlit app (`tallec/bosc_app.py`, deployed at bosc-tallec.streamlit.app) reading the bundled `tallec/tallec.db`. See `tallec/README.md` for the data state, the weekly import procedure and which script writes what. |
| `scripts/` | One-off data tooling (2026 season rebuild from restated stat exports) |
| `CLAUDE.md` | Full project onboarding: domain glossary, conventions, data formats, gotchas |

## How data flows

1. **Weekly:** client enters/uploads the round in the app's *Weekly Input* tab → the
   app runs the pipeline and writes the updated master back to Google Sheets. No repo
   upload needed.
2. **Fallback:** if Sheets is unreachable the app reads the xlsx files in this repo —
   keep them roughly in sync after bulk updates.
3. **Bulk rebuilds** (e.g. a restated full-season export): see `scripts/`.

## Credentials

The app needs Streamlit secrets: `gcp_service_account` (service-account JSON fields) and
`SHEET_ID` (or `NRL_SHEET_ID`/`SL_SHEET_ID`), plus `ODDS_API_KEY` for live odds.
**Never commit any of these** — `.gitignore` blocks JSON key files by default.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Without Sheets secrets it falls back to the local xlsx masters automatically.

## Data state

2026 season rebuilt on 2026-07-09 from the client's full-season "Model Stats Restarted"
exports: NRL complete through Round 17, Super League through Round 16 (Bradford Bulls
and York Knights included). Scores validated against official results.
