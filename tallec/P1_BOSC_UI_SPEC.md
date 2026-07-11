# P1: BOSC UI Skeleton — Player Intelligence Dashboard

**Scope:** Streamlit app (or React + Claude Artifact) MVP for Leeds Rhinos recruitment analyst. Vertical slice: NRL → SL player comparison.

**Timeline:** ~1 week (3–5 build days, 2 iteration days with mock data).

**Goal:** When Stats Perform data arrives, UI is done, just plug in the ratings.

---

## 1. Layout Overview

```
┌─────────────────────────────────────────┐
│ BOSC — Player Intelligence              │
│ (Recruitment analytics for Super League)│
└─────────────────────────────────────────┘

[Search player...] [Filter: Position] [Filter: Competition]

┌──────────────────────────────────────────────────────────┐
│ PLAYER PROFILE: Clayton Faulalo (Manly, Fullback)       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ Stats (5-match)        Ratings              Profile     │
│ ────────────────       ───────              ────────    │
│ Run m/min: 3.26        Form: 72 ⭐⭐⭐      Position:    │
│ Tackles/min: 2.1       Class: 65 ⭐⭐⭐     Fullback    │
│ Tackle %: 89%          Benchmark: 78       Age: 26     │
│ Avg touches: 22        Divergence: +0.4    Matches: 42 │
│                                                          │
│ ┌─ Competition Translation ─────────────────┐           │
│ │ If Fullback moves NRL → SL:               │           │
│ │ Expected Form drop: 0.6 z-scores          │           │
│ │ Translation factor: -0.5 (moderately      │           │
│ │ transfer-loss for this position type)     │           │
│ └───────────────────────────────────────────┘           │
│                                                          │
│ 5-game trend: [line chart Form/Class over last games]  │
│ Peer comparison: [scatter: all Fullbacks in comp]      │
└──────────────────────────────────────────────────────────┘

[← Prev player] [Save comparison] [Export profile] [→ Next player]
```

---

## 2. Key Screens (Streamlit tabs)

### Tab 1: **Player Search & Profile**
- **Input:** Search bar (name / player_id), position filter (select), competition filter (NRL / SL / both).
- **Output:** 
  - Player card (photo placeholder, name, team, position, matches played).
  - Raw stats table (touchdowns, run metres, tackles, fantasy points, stints, etc.) — scrollable, sortable by column.
  - Ratings card (Form, Class, Positional Benchmark, Divergence) — each with a 0–100 bar + z-score note.
  - Competition translation callout (if player has NRL/SL crossover or translation model applied).

### Tab 2: **Positional Benchmarks** (Fullback, Wing, Centre, Halves, Props, Back Row, Bench)
- Each position has:
  - **Top 10 by Form** (leaderboard: name, team, form score, trend).
  - **Top 10 by Class** (lifetime strength, sorted).
  - **Scatter plot:** Form (x) vs Class (y), with bubble size = minutes played. Click a bubble to jump to profile.
  - **Statistical summary:** median form/class, percentile bands (10th/50th/90th).

### Tab 3: **NRL ↔ SL Comparison** (Leeds recruitment focus)
- **Left side (NRL):** Player search in NRL database, display profile.
- **Right side (SL):** Show 3–5 "most similar" SL players (by position, form, class, stat profile).
- **Central panel:** Translation model output.
  - "Clayton Faulalo (Manly) would rate as **SL Fullback: 62** (today's Form 72 → SL 62, -10pt expectation)."
  - Confidence interval: "60–65 (high confidence, 8 similar transfers in history)."
- **Export button:** "Save comparison sheet" → CSV (name, position, NRL form/class, SL predicted form/class, notes).

### Tab 4: **Ratings Over Time** (once data accumulates)
- **Line chart:** Player's Form, Class over weeks (x=round, y=score).
- **Stat sparklines:** Run metres, tackles, errors trend mini-graphs.
- **Bench time indicator:** Is the player getting minutes or sitting out?

---

## 3. Components & Styling

### Color Scheme (matches xLadder)
- **Dark mode (default):** bg `#0E1526`, panel `#18213A`, accent blue `#4C8DFF`, secondary gold `#F5A524`.
- **Form score:** blue gradient (low 30 → high 90).
- **Class score:** purple tint (long-term strength signal).
- **Benchmark (positional):** gold bars (relative to position median).

### Typography
- **Player name:** Large, bold (Fullback / SL secondary).
- **Stat labels:** Monospace for numbers (tackles: `42`, run m/min: `3.26`).
- **Callouts:** Smaller grey text for "z-score: +1.2" context.

### Interactivity
- **Click a player card:** Focus that player, show full profile, show comparables.
- **Hover over a stat:** Tooltip with definition ("Run metres per minute = total run metres / minutes played").
- **Hover over a rating bar:** Show z-score, percentile, peer group context.
- **Export buttons:** Save as CSV, or copy a comparison link.

---

## 4. Data Dependencies (awaiting Stats Perform CSV)

Once weekly imports run, these will populate:

| Table | Source | Purpose |
| --- | --- | --- |
| `player_ratings` | `compute_ratings.py` (Form, Class calculations) | Stats on player profile card |
| `player_comparisons` | `translate_competition.py` (Ridge model on historical transfers) | SL prediction in comparison tab |
| `player_match_stats` | Gerard CSV (already have NRL R11–12) | Raw stats in player profile |

**Placeholder strategy:** Use mock ratings (random 0–100 scores) until real data lands. UI stays identical, just data swaps.

---

## 5. Mock Data Structure (for iteration before Stats Perform arrives)

```python
# In Python, generate mock ratings for display:
def mock_ratings(player_id, season=2026):
    form = np.random.normal(65, 12)  # mean 65, std 12
    class_ = np.random.normal(62, 8)  # class more stable
    bench = np.random.uniform(0, 1)   # % bench time
    return {
        "form_score": np.clip(form, 0, 100),
        "class_score": np.clip(class_, 0, 100),
        "divergence": np.random.normal(0, 0.3),
        "bench_time_pct": bench,
        "form_z": (form - 65) / 12,
        "class_z": (class_ - 62) / 8,
    }
```

Use this to populate `player_ratings` table for the R11–12 dataset, iterate UI.

---

## 6. Integration Path (after Stats Perform CSV arrives)

1. **Parse Stats Perform CSV** → `player_match_stats` (ingest_gerard.py pattern).
2. **Run form calculation** → SQL: `SELECT player_id, AVG(run_metres) FROM player_match_stats WHERE round >= current_round - 5 GROUP BY player_id`.
3. **Run class update** → ELO logic (similar to xLadder `update_elos_for_new_matches`).
4. **Run translation model** → Ridge predict on `(form_z, class_z, position) → SL_rating`.
5. **Refresh Streamlit app** → New ratings flow into UI, no code changes.

---

## 7. Questions for Mike / Leeds (Friday meeting)

1. **Player photos:** Are photos available from your system? Or just names + stats?
2. **Existing transfer list:** Which NRL players have actually moved to SL? (Builds historical training data for translation model.)
3. **Key metrics:** Beyond Form/Class, what single stat should be highlighted for a recruitment analyst? (e.g., "error rate" for discipline, "offload %" for ball movement.)
4. **Update frequency:** Weekly, or after every round?
5. **Export format:** CSV, or do you want links/reports?

---

## 8. Build Checklist

- [ ] Streamlit project structure (`pages/` for tabs).
- [ ] Player search + filter UI (dropdowns, search input).
- [ ] Player profile card template (name, team, position, photo placeholder).
- [ ] Stats table (raw match data, scrollable).
- [ ] Ratings visualization (bars, z-scores, sparklines).
- [ ] Positional benchmark tab (leaderboard + scatter).
- [ ] NRL ↔ SL comparison tab (split view + translation callout).
- [ ] Export buttons (CSV, comparison link).
- [ ] Dark mode CSS (use xLadder palette).
- [ ] Mock data seeding (`mock_ratings` function in `player_ratings` table).
- [ ] Test with R11–12 dataset (356 players, mock ratings).
- [ ] Docstring each major function (so Stats Perform integrator understands data flow).

---

## 9. Files to Create

- `bosc_app.py` — Main Streamlit entry point.
- `pages/search.py` — Tab 1.
- `pages/benchmarks.py` — Tab 2.
- `pages/comparison.py` — Tab 3.
- `pages/trends.py` — Tab 4 (draft).
- `helpers/ratings.py` — Mock rating generators, z-score logic.
- `helpers/db.py` — Query helpers (get_player_profile, get_peer_group, etc.).
- `helpers/export.py` — CSV export utilities.
- `styles.css` — Dark mode colors + spacing (embed in Streamlit theme config).

---

This is the skeleton. Build it, seed with mock data, and the real pipeline just fills the tables. Ship to Leeds by 2026-08-31.
