# Competition Translation Model — NRL ↔ SL Player Equivalence

**Objective:** Given an NRL player's recent form and class, estimate their expected performance if they moved to Super League (or vice versa).

**Output:** A *translation factor* (–2.0 to +2.0 z-score multiplier) + confidence band.

**Example:** Clayton Faulalo (NRL Fullback, Form +1.2σ) → *SL prediction: +0.7σ* (–0.5σ drop due to league difference, position-specific learning curve).

---

## 1. Problem Statement

Rugby league is the same sport across NRL and SL, but:
- **Pace & physicality:** NRL slightly faster, more direct.
- **Play style:** SL more structured, less space.
- **Positions differ slightly:** NRL Props are heavier; SL Halves more kicking-focused in some clubs.
- **Individual variance:** Some players thrive on the move (→ NRL boost), others in tight structure (→ SL boost).

**Historical fact:** Not all great NRL players translate to SL equally (e.g., wingers often thrive, halves struggle with 6-again rules).

---

## 2. Data Foundation: Transfer History

**First task:** Collect actual NRL ↔ SL transfers, 2015–2026.

| Transfer | Position | NRL Form (z) | SL Form (z) | Δ Form | Translation Factor | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| player_X | Wing | +1.5 | +1.1 | –0.4 | –0.3 | Good fit, light learning curve |
| player_Y | Halfback | +1.0 | –0.2 | –1.2 | –1.0 | Struggled with SL defensive demands |
| player_Z | Fullback | +0.8 | +0.9 | +0.1 | +0.1 | Slight boost (more space in SL?) |

**Data sources:** 
- Wikipedia transfer pages (searchable by club: Leeds, Wigan, St Helens, etc.).
- NRL official player profiles.
- SuperLeague.com archive.
- Stats Perform historical records (when available).

**Minimum viable dataset:** 15–20 well-documented transfers with before/after stats.

---

## 3. Model Architecture

### Input Features (at transfer time)

For each player, compute at time *T* (last season before transfer):

| Feature | Source | Calculation |
| --- | --- | --- |
| `form_z_pre` | player_ratings | 5-game form, z-scored to position peer group |
| `class_z_pre` | player_ratings | ELO-based class rating, z-scored |
| `position_group` | positions | Categorical: Fullback, Wing, Centre, Halves, Props, Back Row, Bench |
| `age` | player_match_stats | Years (approx) |
| `games_per_season` | player_match_stats | Average games played in pre-transfer season |
| `injury_rate` | player_match_stats | % of fixtures missed (inactive weeks / total weeks) |
| `minutes_per_game` | player_match_stats | Average minutes (bench vs starting indicator) |
| `tackle_efficiency` | player_match_stats | Tackles made % (or tackling vs evasion balance) |

### Target Variable

`form_z_post` — Form z-score in first season of new league, **capped to 3 games** (to isolate "transfer effect" before full adaptation).

`translation_factor = form_z_post - form_z_pre` (adjusted for opponent strength, but simplified for now).

### Model: Ridge Regression

```python
from sklearn.linear_model import Ridge

X_train = transfers_df[["form_z_pre", "class_z_pre", "age", "games_per_season", "injury_rate"]]
# Position as one-hot: Fullback_1, Wing_1, etc.
X_train_encoded = pd.get_dummies(X_train, columns=["position_group"], drop_first=True)

y_train = transfers_df["translation_factor"]

model = Ridge(alpha=1.0)
model.fit(X_train_encoded, y_train)

# Prediction for a new player:
# form_z_post_predicted = form_z_pre + model.predict(X_new)
```

**Why Ridge?**
- Interpretable coefficients (e.g., "Halves have a –0.3 translation penalty").
- Regularized to avoid overfitting on small dataset.
- Fast inference (needed for real-time BOSC queries).

---

## 4. Confidence Scoring

For each prediction, return a **confidence interval** (±0.3 to ±0.8 z-scores).

**Method:** Distance-weighted kNN on training set. For a new player (e.g., NRL Fullback, form +1.1):
1. Find 3–5 nearest training transfers (by position, form, class).
2. Compute residuals in that neighborhood.
3. Report: *prediction ± std(residuals)*.

**Example output:**
- *"SL Form estimate: +0.7σ (confidence: ±0.4σ, high confidence based on 5 similar Fullback transfers)"*
- vs.
- *"SL Form estimate: +0.4σ (confidence: ±1.0σ, low confidence, only 1 Halfback transfer in history)"*

---

## 5. Validation & Iteration

### Phase 1: Retrospective Validation (Aug 2026)
1. Collect 20+ historical transfers.
2. Fit model on 15 transfers.
3. Predict on held-out 5 transfers.
4. Check: does model ordering match reality? (e.g., player who actually succeeded in SL gets a high prediction).

### Phase 2: Prospective Validation (Sep–Dec 2026)
- When Leeds makes an actual signing from NRL, use model to predict first-season form.
- After 10–15 games, compare prediction vs actual performance.
- Refit model quarterly with real outcomes.

---

## 6. Position-Specific Notes

**Fullbacks:** Generally translate well (–0.2 to 0.0 factor). Rely on speed and decision-making, which transfer.

**Wingers:** Usually thrive in SL (maybe +0.3 boost?). Space-dependent; SL less congested.

**Centres:** Mixed (–0.1 to +0.2). Depends on defensive reads vs offload skill.

**Halves (Five-Eighth / Halfback):** Highest variance (–1.0 to +0.5). SL defensive rules (ruck speed, 6-again) disrupt NRL play-caller patterns. High-risk position for NRL→SL moves.

**Props:** May struggle with SL pace (–0.3 to –0.1), or dominate if strong ball-carrier (–0.1 to +0.2). Very position-specific.

**Back Row (2nd Row / Lock):** Moderate drop expected (–0.2 to 0.0). Good athletes transfer OK.

---

## 7. Failure Modes & Caveats

**Known limitations:**
- **Sample size:** Until we have 20+ documented transfers, confidence is low.
- **Unmodeled confounds:** Coaching, team structure, family/lifestyle factors affect performance but aren't in raw stats.
- **Stat availability:** Stats Perform may measure things differently than Gerard; future model will need re-calibration.
- **Injury risk:** Model doesn't predict injury; focus on "if healthy, what's the expected adaptation?"

**Mitigations:**
- Always report confidence interval.
- Pair model output with narrative: *"Model says +0.7σ, but this player is known for quick adaptation / slow starts / family reasons."*
- Monthly retraining once real data lands.

---

## 8. Implementation Timeline

| Phase | Timeline | Task |
| --- | --- | --- |
| **P1** | Now–Aug 15 | Collect 20 transfers manually from Wikipedia + sports news. Estimate pre/post form from archive stats. |
| **P2** | Aug 15–20 | Fit Ridge model on 15 transfers, validate on 5. |
| **P3** | Aug 20–31 | Integrate into BOSC UI (comparison tab). Show predictions with confidence bands. |
| **P4** | Sep–Dec | Real-time validation: track Leeds signings, compare prediction vs actual. Refit monthly. |

---

## 9. Questions for Stats Perform / Leeds Analyst (Friday meeting)

1. **Historical player transfers:** Do you have a list of NRL players who moved to SL (or vice versa)? Helps bootstrap the training set.
2. **Performance metrics they trust:** Beyond our Form/Class, what do you use to evaluate a new signing after 5 games?
3. **Position-specific concerns:** Are there positions you're particularly concerned about (e.g., "our Halves are struggling against SL defensive structure")?
4. **Peer transfer outcomes:** Are there 2–3 recent transfers (good or bad) you'd like the model to retroactively explain?

---

## 10. File Structure (to build)

```
TALLEC/
├── models/
│   ├── competition_translation_fit.py     # Fit Ridge on transfer history
│   ├── competition_translation_predict.py # Real-time prediction + confidence
│   └── transfer_dataset.csv               # 20+ historical transfers + outcomes
├── COMPETITION_TRANSLATION_SPEC.md        # This file
└── (integrated into BOSC comparison tab)
```

---

**Summary:** This model is the jewel of BOSC for Leeds — it lets a recruitment analyst ask "should we sign this NRL star?" and get a data-backed answer: "Model says he'll rate +0.7σ in SL (confidence ±0.4σ) — that's competitive with our current Fullback pool, so risky unless he's a culture fit."
