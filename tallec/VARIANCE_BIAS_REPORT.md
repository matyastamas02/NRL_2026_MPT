# What the constant-ability assumption costs

The rating engine estimates its variance components with one intercept per player, held constant over his whole career. Everything that actually moves — ageing, a change of position, a season off the bench — is then booked as game-to-game noise. This compares that against a decomposition that gives each player-SEASON its own intercept, so a level that changes between seasons is no longer mistaken for randomness within them.


Neither is a state-space model. The point is the size and direction of the gap, not that the alternative is right.


## Components

| competition | sigma2_career | sigma2_season | sigma2_change | tau2_career | tau2_season | tau2_change |
| --- | --- | --- | --- | --- | --- | --- |
| NRL | 0.1847 | 0.1715 | -7.1223 | 0.0376 | 0.0507 | 34.8687 |
| SL | 0.1905 | 0.1848 | -2.9713 | 0.0396 | 0.0452 | 14.1249 |
| NSW | 0.2018 | 0.1944 | -3.6318 | 0.0368 | 0.0441 | 19.8178 |
| QLD | 0.1794 | 0.1671 | -6.8517 | 0.0416 | 0.0539 | 29.4510 |


`sigma2_change` and `tau2_change` are percentages.


## What it does to the shrinkage

B is the share of a player's own numbers the rating keeps. `B_now` is today's, `B_alt` the same figure under the player-season decomposition.

| competition | matches | B_now | B_alt | change_pct |
| --- | --- | --- | --- | --- |
| NRL | 1 | 0.169 | 0.228 | 34.904 |
| NRL | 5 | 0.504 | 0.596 | 18.256 |
| NRL | 10 | 0.670 | 0.747 | 11.437 |
| NRL | 25 | 0.836 | 0.881 | 5.393 |
| SL | 1 | 0.172 | 0.196 | 14.160 |
| SL | 5 | 0.509 | 0.550 | 7.932 |
| SL | 10 | 0.675 | 0.710 | 5.118 |
| SL | 25 | 0.839 | 0.859 | 2.479 |
| NSW | 1 | 0.154 | 0.185 | 19.834 |
| NSW | 5 | 0.477 | 0.531 | 11.402 |
| NSW | 10 | 0.646 | 0.694 | 7.445 |
| NSW | 25 | 0.820 | 0.850 | 3.648 |
| QLD | 1 | 0.188 | 0.244 | 29.475 |
| QLD | 5 | 0.537 | 0.617 | 14.925 |
| QLD | 10 | 0.699 | 0.763 | 9.230 |
| QLD | 25 | 0.853 | 0.890 | 4.303 |


## And what it does to the published list

The components above are a diagnosis; this is the prognosis. Each player's rating is rebuilt with the alternative components and nothing else changed, then compared with what is published today, within his own peer group.

| competition | reversed_pairs | median_rank_move | p95_rank_move | median_score_move | p90_score_move |
| --- | --- | --- | --- | --- | --- |
| NRL | 0.012 | 0.000 | 4.000 | 1.600 | 2.965 |
| SL | 0.006 | 0.000 | 2.000 | 0.709 | 1.268 |
| NSW | 0.008 | 0.000 | 2.450 | 0.604 | 1.548 |
| QLD | 0.012 | 0.000 | 4.000 | 1.093 | 2.353 |


`reversed_pairs` is the share of player pairs whose order flips; rank moves are places within a peer group; score moves are points on the published 0-100.



## What this says

Letting a player's level move between seasons raises tau-squared by 25% on average and lowers sigma-squared by 5%. Both point the same way: the current model books real change as noise, so it holds the true spread between players too low and the match-to-match randomness too high.

The consequence is heaviest where evidence is thinnest: at one match the shrinkage factor would be 25% higher, at ten matches 8%. 'At least conservative' was too comfortable a description of that and has been withdrawn.


**But the size of the consequence is not what an earlier version of this page implied.** It said the correction reorders players rather than merely compressing them, and stopped. Measured, the reordering is small: at most 1.2% of pairs within a peer group change places and the median player does not move a single rank. The fourth external review was right to ask for the number, and right that the wording suggested more product risk than the data supports.


Where it does bite is the distance rather than the order. tau divides the published 0-100, so a player's score moves by a median of up to 1.6 points and more in the tails. That is the honest statement: **the ranking is robust to this bias, the displayed gaps between players are not.** A peer score is usable for ordering a shortlist and should not be read as a calibrated distance.


The fix is a model that lets ability move — a player-season random effect at minimum, a state-space formulation properly. Not done.

