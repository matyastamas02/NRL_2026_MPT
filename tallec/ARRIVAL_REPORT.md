# Does he get there at all

Every other evaluation in this project is conditional on arrival — a player is scored once he has played enough in the target competition to carry a rating. This is the step before that, on the same cohort and the same rolling origins: for a player at a given level, what is the chance he reaches a usable role in another competition next season.


**Read the population carefully.** It is everyone who played three or more matches in a source competition, set against every competition he could have entered. So this measures *of players at this level, what share turn up*, and not *of the players a club signed, what share worked out*. There is no signing or registration data here, so a man nobody wanted and a man who was signed and never picked are the same row. A low probability is therefore not evidence against a player a club has already decided to sign — it is a statement about how often men like him appear.


Outcome: **rated** (three or more matches in the target competition). Base rate 3.93% over 11,895 candidate entries.


## What was fitted

| origin | train_rows | train_arrivals | train_base | directions | evaluated | evaluated_arrivals |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | 4931 | 200 | 0.0406 | 6 | 3734 | 160 |
| 2024 | 8665 | 360 | 0.0415 | 9 | 4038 | 140 |
| 2025 | 12703 | 500 | 0.0394 | 12 | 4123 | 167 |


## Does it rank arrivals above the rest

| origin | n | arrivals | base_rate | auc | brier | brier_base |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | 3734 | 160 | 0.0428 | 0.7346 | 0.0438 | 0.0410 |
| 2024 | 4038 | 140 | 0.0347 | 0.7351 | 0.0343 | 0.0335 |
| 2025 | 4123 | 167 | 0.0405 | 0.7742 | 0.0379 | 0.0389 |


AUC is the chance a randomly chosen arrival is ranked above a randomly chosen non-arrival; 0.5 is a coin. `brier_base` is what predicting the base rate for everyone would score, so the model has to beat it to be worth anything.


Across 3 origins the mean AUC is **0.748**, and the model beats the base rate on Brier score in 1 of 3.


## Inside a direction, which is the only place a club stands

The pooled figure above can look respectable for a reason that helps nobody: directions differ enormously in how often anyone moves along them, and ranking every NRL-to-NSW-Cup candidate above every Super-League-to-NRL one scores well while telling a recruiter nothing. He is never choosing between those two men. He has one target competition and a list of feeder players, and the question is which of THOSE to sign. So the same model is scored again inside each direction.

| direction | n | arrivals | base_rate | auc |
| --- | --- | --- | --- | --- |
| NSW->NRL | 715 | 75 | 0.1049 | 0.7583 |
| NRL->SL | 1035 | 42 | 0.0406 | 0.7457 |
| NRL->NSW | 800 | 71 | 0.0887 | 0.7343 |
| SL->NRL | 984 | 14 | 0.0142 | 0.6957 |
| QLD->NSW | 1179 | 48 | 0.0407 | 0.6935 |
| QLD->SL | 1186 | 22 | 0.0185 | 0.6650 |
| NSW->SL | 1009 | 26 | 0.0258 | 0.6617 |
| QLD->NRL | 1070 | 34 | 0.0318 | 0.6556 |
| NRL->QLD | 941 | 29 | 0.0308 | 0.6085 |
| NSW->QLD | 1007 | 79 | 0.0785 | 0.5888 |
| SL->QLD | 986 | 15 | 0.0152 | 0.5587 |
| SL->NSW | 983 | 12 | 0.0122 | 0.4888 |


### Against the simplest thing that could work

A model given a column has to beat that column. The first version of this model did not, and nothing in the old report would have shown it: minutes played in the source competition rank NSW Cup arrivals at 0.71 on their own, while the fitted model managed 0.44. That is what a coefficient compromise looks like — one slope per feature, shared across pathways that do not work the same way.

| direction | n | arrivals | class_source | source_matches | source_minutes | mins_pg | age | model | best_single | model_beats_best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NRL->NSW | 800 | 71 | 0.349 | 0.337 | 0.326 | 0.452 | 0.347 | 0.734 | 0.452 | yes |
| NRL->QLD | 941 | 29 | 0.504 | 0.317 | 0.353 | 0.508 | 0.490 | 0.608 | 0.508 | yes |
| NRL->SL | 1035 | 42 | 0.415 | 0.389 | 0.376 | 0.439 | 0.728 | 0.746 | 0.728 | yes |
| NSW->NRL | 715 | 75 | 0.621 | 0.682 | 0.706 | 0.593 | 0.339 | 0.758 | 0.706 | yes |
| NSW->QLD | 1007 | 79 | 0.429 | 0.612 | 0.579 | 0.515 | 0.538 | 0.589 | 0.612 | NO |
| NSW->SL | 1009 | 26 | 0.609 | 0.595 | 0.624 | 0.575 | 0.717 | 0.662 | 0.717 | NO |
| QLD->NRL | 1070 | 34 | 0.621 | 0.618 | 0.647 | 0.581 | 0.275 | 0.656 | 0.647 | yes |
| QLD->NSW | 1179 | 48 | 0.523 | 0.623 | 0.672 | 0.607 | 0.373 | 0.694 | 0.672 | yes |
| QLD->SL | 1186 | 22 | 0.687 | 0.691 | 0.657 | 0.516 | 0.515 | 0.665 | 0.691 | NO |
| SL->NRL | 984 | 14 | 0.557 | 0.655 | 0.698 | 0.599 | 0.375 | 0.696 | 0.698 | NO |
| SL->NSW | 983 | 12 | 0.483 | 0.434 | 0.462 | 0.522 | 0.464 | 0.489 | 0.522 | NO |
| SL->QLD | 986 | 15 | 0.532 | 0.427 | 0.471 | 0.507 | 0.449 | 0.559 | 0.532 | yes |


Arrival-weighted mean AUC inside a direction: **0.676**, against 0.748 pooled.

**There is player-level signal inside a direction**, which the first version of this report denied. It concluded that the data held nothing usable, and the fourth external review refuted that in one line: source minutes alone rank NSW Cup arrivals at 0.71. The failure was the specification — one slope per feature shared across every pathway — and not the data.


The two directions the client asks about, now with intervals — which the previous version quoted without, having called them inverted on the strength of a bare 0.44 and 0.39: NSW->NRL 0.758 [0.707, 0.806]; QLD->NRL 0.656 [0.571, 0.744].


## Is it calibrated

When it says twenty per cent, do a fifth of them arrive? This matters more than discrimination for a recruiter, because the number is read as a probability rather than as a rank.

| predicted | n | mean_predicted | actual | arrivals |
| --- | --- | --- | --- | --- |
| 0%-2% | 6631 | 0.0088 | 0.0157 | 104 |
| 2%-5% | 2816 | 0.0313 | 0.0398 | 112 |
| 5%-10% | 1372 | 0.0688 | 0.0736 | 101 |
| 10%-20% | 641 | 0.1393 | 0.1170 | 75 |
| 20%-40% | 295 | 0.2741 | 0.1525 | 45 |
| 40%-101% | 140 | 0.5846 | 0.2143 | 30 |


## What a shortlist gets

The practical form of the question. Rank every candidate by the model, take the top slice, and see how many of the season's actual arrivals are in it.

| shortlist | players | arrivals_caught | of_all_arrivals | hit_rate | lift |
| --- | --- | --- | --- | --- | --- |
| top 5% | 595 | 93 | 0.199 | 0.156 | 3.981 |
| top 10% | 1190 | 159 | 0.340 | 0.134 | 3.403 |
| top 25% | 2974 | 280 | 0.600 | 0.094 | 2.398 |

`lift` is the hit rate in the slice divided by the 3.93% base rate.



## Feeder to NRL, on its own

The question Leeds asks, and the only direction where the answer is quoted to anyone.

| n | arrivals | base_rate | auc | brier | brier_base |
| --- | --- | --- | --- | --- | --- |
| 1785 | 109 | 0.0611 | 0.7751 | 0.0643 | 0.0573 |


| shortlist | players | arrivals_caught | of_all_arrivals | hit_rate | lift |
| --- | --- | --- | --- | --- | --- |
| top 5% | 89 | 24 | 0.220 | 0.270 | 4.416 |
| top 10% | 178 | 40 | 0.367 | 0.225 | 3.680 |
| top 25% | 446 | 74 | 0.679 | 0.166 | 2.717 |


## Verdict

**There is usable signal, and the model now finds it.** It separates arrivals from the rest inside a direction as well as across them, and it beats the best single raw column in 7 of 12 directions — including both of the client's. That is the condition for showing it at all, and the previous version failed it without anyone noticing, because nothing compared the model with its own inputs.


It is still not a probability to quote at a player. The directions where it loses to a single column are the thin ones, the calibration above over-predicts in the upper bins, and the population is everyone at a level rather than everyone a club wanted. Use it to order a shortlist, not to put a number beside a name.


What has NOT changed is the limit on what the outcome means. Without squad, contract or registration data, a man nobody wanted and a man signed and never picked are the same row, so this ranks pathway arrival and not recruitment success. The third review asked for that data and the request stands.



## What this is not

- Not a probability that a signing works out. See the population note above; nothing here observes a contract.

- Not independent of the rating model. Both are fed the same source Class score, so a player rated highly will tend to score highly on both, and the two numbers should be read as one picture rather than as agreement between two witnesses.

- Not a statement about a particular club's intentions. A Super League club that has already decided to sign a man should read the conditional rating and ignore this page.

