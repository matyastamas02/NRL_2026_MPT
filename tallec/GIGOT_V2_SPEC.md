# GIGOT v2 — Team-List-Aware Match Prediction

GIGOT v2 extends the xLadder match model with player-level signals from TALLEC.
Five inputs per team per fixture:

| # | Input | Source | Status |
| --- | --- | --- | --- |
| 1 | Team Form | xLadder pipeline (rolling DTS form) | **DONE** — live in `xladder_pipeline.py` |
| 2 | Team Class | xLadder ELO (K=27, stored in masters) | **DONE** — live |
| 3 | Player Form | TALLEC `player_ratings.form_score` | schema ready, real values need full-season player data |
| 4 | Player Class | TALLEC `player_ratings.class_score` | schema ready, same dependency |
| 5 | Contribution Rating | `gigot_contribution.py` | **DONE — real data** (R11–12), percentile-scaled |

## Contribution Rating (input #5) — implemented

Per player per match: share of own team's output, blended across stat groups
(attack 45% = run metres/PCM/tackle breaks/line breaks, defence 35% = tackles,
points 20% = tries/try assists), then percentile-ranked to 0–100 (median = 50).
Team totals are computed as the sum of teammates' stats in the same match, so
the metric is fully self-contained — no external team-stat join needed.

Sanity check on R11–12: top of the table is Munster, Reynolds, Holmes,
Farnworth, Olakau'atu — the metric finds the stars from raw shares alone.

Tables: `player_contribution` (per match), `player_contribution_rating`
(per player, rolling mean; switch to 5-match window once more rounds land).

## Expected Contribution from team lists

`expected_contribution(team, available_players)` in `gigot_contribution.py`
sums the lineup's contribution ratings and returns the delta vs the full squad.

Demo (R11–12): Manly without Olakau'atu → expected contribution −97
(≈ 8.5% of squad total). This delta is the **Player Availability** signal.

## Integration sketch (once inputs 3–4 are real)

```
EM_gigot = w_tf·ΔTeamForm + w_tc·ΔTeamClass
         + w_pf·ΔLineupForm + w_pc·ΔLineupClass
         + w_av·ΔExpectedContribution + w_home·HomeAdv
```

- Lineup aggregates = sum over the named 17 (from team lists) of each player's
  Form/Class weighted by his contribution share.
- Weights optimised exactly like the current xLadder weight fit (train
  2023–24 style split once player history is deep enough; until then, hold
  team weights at xLadder values and fit only the player terms).
- Baseline requirement: GIGOT v2 must beat plain xLadder EM accuracy on the
  same fixtures, else the player layer isn't earning its complexity.

## Blocked on

1. **Team lists feed** — who names the 17? (Mike said GIGOT v3 team-list
   integration is out of MVP scope; v2 can start with manual lists or
   nrl.com team announcements via Gerard.)
2. **Full-season player data** — for real Form/Class and stable contribution
   windows (Gerard has it, one email away).
3. **try_assists sparsity** — R11–12 Gerard exports rarely fill it; points
   group currently leans on tries alone. Revisit weights with more data.
