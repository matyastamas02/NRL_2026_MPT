# Weight profiles for Middles and Edge

The rating engine kept a position map of its own — Prop against Back Row (second row and lock together) — while `sp_schema` had already moved to Middles (prop and lock) and Edge (second row alone) at Leeds's request. Unifying them leaves the two new groups needing weight vectors, and the retired Back Row profile does not transfer cleanly, because it was fitted to a pool that included the locks now leaving it.


Each candidate is scored by the **independent year-to-year correlation** of its composite: a player's season mean against his next season's, sharing no match, before any shrinkage. It measures how much of what the weighting picks up is a trait the player repeats rather than noise.


Computed on 36,989 ratable matches, seasons through 2025, minimum 5 matches in each season of a pair.


## Candidates

| candidate | Edge | Middles | mean_r |
| --- | --- | --- | --- |
| Middles = Prop, Edge = 0.5/0.5 | 0.580 | 0.634 | 0.607 |
| inherit (Middles=Prop, Edge=Back Row) | 0.576 | 0.637 | 0.606 |
| both = Prop | 0.579 | 0.630 | 0.604 |
| Middles = 0.7 Prop + 0.3 Back Row | 0.578 | 0.629 | 0.603 |
| both = Back Row | 0.580 | 0.604 | 0.592 |
| uniform (control) | 0.541 | 0.522 | 0.532 |


Sample sizes are the same for every candidate: Edge 510 season pairs over 211 players, Middles 858 season pairs over 341 players.


## The two retired profiles, for reference

| rate | Prop (old) | Back Row (old) |
| --- | --- | --- |
| run_pm | 0.29 | 0.19 |
| pcm_pm | 0.24 | 0.14 |
| tb_pm | 0.05 | 0.10 |
| lb_pm | 0.04 | 0.09 |
| tck_pm | 0.24 | 0.24 |
| off_pm | 0.05 | 0.14 |
| ta_pm | 0.00 | 0.00 |
| tries_pm | 0.04 | 0.05 |
| err_pm | 0.05 | 0.05 |


## Reading this

Differences of a few thousandths are not a decision. Take the simplest candidate whose correlation is not meaningfully below the best, and record the choice as a calibration decision rather than a discovery — reliability says a weighting is measuring *something* consistently, not that it is measuring the right thing. The component weights above are the part a coach can disagree with, and that disagreement should win.

