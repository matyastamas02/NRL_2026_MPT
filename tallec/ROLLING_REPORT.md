# Competition translation — rolling-origin backtest

Each season is forecast using only what was known before it. The rating history is rebuilt from matches up to the previous season, the ladder and the model are fitted on moves that had already landed by then, and the result is scored against what the players actually did in the season being forecast. The inner split is `GroupKFold` by player; the outer split is chronological.

This replaces section 6 of `VALIDATION_REPORT.md`, which applied the current model to historical moves it had been fitted on and therefore measured nothing. Errors below are in points of the 0-100 rating.


## What was fitted, and on what

| origin | train_pairs | train_players | inner_rmse | evaluated |
| --- | --- | --- | --- | --- |
| 2023 | 372 | 255 | 24.62 | 152 |
| 2024 | 705 | 370 | 23.80 | 131 |
| 2025 | 1113 | 495 | 24.15 | 158 |

`inner_rmse` is the out-of-sample error inside the training window under GroupKFold — a check that the fit is not memorising players, not a forecast.


## The forecast, by origin

| origin | predictor | n | mae | rmse | bias |
| --- | --- | --- | --- | --- | --- |
| 2023 | translation (ladder) | 152 | 22.14 | 27.80 | -0.04 |
| 2023 | conditional model | 152 | 18.86 | 22.56 | -1.41 |
| 2023 | no translation | 152 | 24.28 | 29.48 | 4.03 |
| 2023 | competition average (50) | 152 | 19.53 | 23.13 | 3.51 |
| 2024 | translation (ladder) | 131 | 25.11 | 31.80 | 3.24 |
| 2024 | conditional model | 131 | 21.75 | 25.38 | 0.32 |
| 2024 | no translation | 131 | 25.66 | 32.13 | 2.54 |
| 2024 | competition average (50) | 131 | 22.51 | 26.78 | 3.13 |
| 2025 | translation (ladder) | 158 | 24.29 | 30.45 | 1.88 |
| 2025 | conditional model | 158 | 19.99 | 24.11 | 1.51 |
| 2025 | no translation | 158 | 25.23 | 31.91 | 1.32 |
| 2025 | competition average (50) | 158 | 21.95 | 26.02 | 4.73 |


## Against the simplest thing that could work

A conditional model has to beat a straight line. `target ~ source` fitted on the same training window, once per direction, costs two parameters and shrinks by exactly the right amount for that pathway — so anything the model adds has to be conditioning rather than flexibility. This comparison did not exist in this project until 2026-09-24, and its absence let an arrival model score 0.44 on a feature worth 0.71 by itself for a month.

| direction | n | conditional model | straight line, this direction | straight line, pooled | ladder | no translation | flat 50 | best simple | model wins | vs line | ci low | ci high | clear |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NRL->NSW | 67 | 20.76 | 20.03 | 21.00 | 23.83 | 29.81 | 21.07 | 20.03 | NO | -0.73 | -1.41 | -0.06 | yes |
| NRL->QLD | 29 | 21.42 | 23.38 | 22.98 | 26.07 | 25.58 | 23.48 | 22.98 | yes | 1.96 | 0.75 | 3.29 | yes |
| NRL->SL | 42 | 17.84 | 18.05 | 18.71 | 21.66 | 24.87 | 20.57 | 18.05 | yes | 0.20 | -1.40 | 1.82 | no |
| NSW->NRL | 65 | 19.55 | 19.74 | 19.85 | 21.18 | 21.70 | 19.98 | 19.74 | yes | 0.19 | -0.68 | 1.07 | no |
| NSW->QLD | 76 | 19.68 | 21.15 | 21.04 | 25.70 | 24.94 | 21.83 | 21.04 | yes | 1.47 | 0.58 | 2.42 | yes |
| NSW->SL | 26 | 14.07 | 14.67 | 16.10 | 21.29 | 23.17 | 19.97 | 14.67 | yes | 0.60 | -2.57 | 3.69 | no |
| QLD->NRL | 25 | 16.82 | 16.98 | 18.31 | 22.56 | 23.76 | 17.80 | 16.98 | yes | 0.16 | -1.37 | 1.59 | no |
| QLD->NSW | 48 | 24.40 | 22.85 | 23.51 | 25.57 | 25.55 | 22.58 | 22.58 | NO | -1.55 | -2.93 | -0.33 | yes |
| QLD->SL | 22 | 22.84 | 20.69 | 20.69 | 31.03 | 30.55 | 21.56 | 20.69 | NO | -2.15 | -4.19 | -0.18 | yes |


`vs line` is the model's mean absolute error subtracted from the direction-specific line's, so **positive means the model is ahead**, with a player-clustered interval. It is measured against the line chosen in advance as the bar rather than against `best simple`, which is a minimum over six candidates picked after the fact and would carry an interval that means nothing.


On point estimates the model wins in **6 of 9** directions. On intervals the picture is harder: it is clearly ahead in 2 and clearly **behind** in 3 — NRL->NSW by 0.73, QLD->NSW by 1.55, QLD->SL by 2.15.


**On the client's own pathways the model is not distinguishable from a straight line.** Taking the two feeder-to-NRL directions together, 90 moves, it is ahead by +0.18 points with an interval of [-0.62, +0.92], which contains zero.


That is the honest description of what the conditional model is worth where it is sold. It is not an argument for deleting it: a line cannot use position, cannot carry a missing-value flag, and cannot be quoted for a direction with too few moves to fit one. It is an argument against presenting the conditioning as the thing that makes the product work. What makes it work is the shrinkage — and `target ~ source` does that with two parameters.



## The same question against a different outcome

The outcome above is the shrunk season rating the app would publish, and it is pulled toward 50 — which is the number one of the baselines predicts. The review of 2026-09-22 argued that this flatters shrunk predictors, and it does. So every comparison is also run against the same season's **unshrunk** mean on the same scale: noisier, not something anyone would publish, but not drawn toward any predictor. A finding that holds on both is a finding; one that holds on only the first is a property of the scoring.

| outcome | predictor | against | cost | ci_low | ci_high | clear |
| --- | --- | --- | --- | --- | --- | --- |
| shrunk (published) | translation (ladder) | a flat 50 | +2.513 | +0.432 | +4.630 | yes |
| shrunk (published) | conditional model | a flat 50 | -1.159 | -2.196 | -0.134 | yes |
| shrunk (published) | translation (ladder) | leaving the rating alone | -1.235 | -2.145 | -0.275 | yes |
| unshrunk season mean | translation (ladder) | a flat 50 | +0.397 | -1.779 | +2.611 | no |
| unshrunk season mean | conditional model | a flat 50 | -1.691 | -2.730 | -0.676 | yes |
| unshrunk season mean | translation (ladder) | leaving the rating alone | -0.845 | -1.800 | +0.137 | no |

`cost` is the predictor's mean absolute error minus the baseline's, so a **positive cost means the predictor is worse** than the baseline it is set against. `clear` is whether the player-clustered interval excludes zero.

**The ladder's defeat does not survive the change of outcome.** Against the published rating it is beaten by a constant with an interval clear of zero; against the unshrunk mean the same comparison contains zero. The review was right that part of that result was the scoring rather than the ladder, and the claim is narrowed accordingly: the ladder is beaten by a constant at predicting *the number this system publishes*, which is a real and useful statement about the product, and is not established as a statement about the player.

**The decision that actually ships does survive it.** The conditional model beats a flat 50 on both outcomes, and by more on the unshrunk one (+1.69 points against +1.16). Showing the model rather than the ladder is therefore not an artefact of how the outcome was defined, which is the one thing here a club depends on.



## The headline use case, on its own

Leeds asks one question: a man is playing in the NSW Cup or the Queensland Cup and has never played in the NRL — what would he do there? Every figure above pools that with returners and with moves in other directions. This is that cohort alone, and it is the number to quote when anyone asks whether the system works.

| outcome | predictor | n | mae | rmse | bias |
| --- | --- | --- | --- | --- | --- |
| shrunk (published) | translation (ladder) | 72 | 22.90 | 28.76 | 5.91 |
| shrunk (published) | conditional model | 72 | 19.16 | 22.79 | 2.06 |
| shrunk (published) | no translation | 72 | 23.28 | 28.24 | -3.15 |
| shrunk (published) | competition average (50) | 72 | 19.26 | 23.32 | 0.87 |
| unshrunk season mean | translation (ladder) | 72 | 27.36 | 33.18 | 5.37 |
| unshrunk season mean | conditional model | 72 | 24.96 | 28.89 | 1.52 |
| unshrunk season mean | no translation | 72 | 27.41 | 32.86 | -3.69 |
| unshrunk season mean | competition average (50) | 72 | 25.60 | 29.76 | 0.33 |

72 players over 3 origins.

**Against a flat 50 the model is not distinguishable here.** It is ahead by 0.10 points with a player-clustered interval of [-1.56, +1.80] — which contains zero. On the cohort the product exists to serve, at this sample size, we cannot show the model beats assuming every arrival is average.

**Against carrying his feeder rating across unchanged it clearly is.** 4.12 points [+0.98, +7.45], clear of zero.

Read together: what the model reliably does is stop a feeder rating being taken at face value. What it has not yet been shown to do is rank one arrival above another. Those are different products, and only the first is evidenced.



## What this says

**The ladder is beaten by assuming everyone is average.** In 3 of 3 origins, predicting 50 for every player produced less error than carrying his rating across with the measured shift.
Across all 441 entries the ladder costs +2.51 points against a flat 50, player-clustered 95% interval [+0.43, +4.63] — clear of zero.

The cause is spread. The ladder inherits the full range of the source rating when the range that can actually be predicted in the target competition is much narrower. It answers 'what does this level become', which is a real quantity, but as a forecast of one man it does not shrink and it should.

**The conditional model shrinks, and is the better forecast — by a little.** It beats a flat 50 by 1.16 points [+0.13, +2.20], an interval clear of zero.

That margin is worth stating plainly rather than dressing up. On a scale whose standard deviation is about 26, beating a constant by 1.16 points is a small edge — and the model's own error is still 20.1. The model is the right number to show because it is the only one of the four not beaten by a constant, not because it is accurate.

**For a player with no record in the competition, the translation still adds nothing.** Over 320 such entries the ladder saves +0.99 points against leaving the rating untouched, 95% interval [-0.14, +2.07]. This is the group Leeds asks about, and it is the group where the headline number earns least.

**For a player returning to a competition, it does.** Over 121 such entries the ladder saves +1.90 points against leaving the rating untouched, 95% interval [+0.29, +3.48] — clear of zero.

This one is new, and it only became visible once the cohort stopped being dominated by men who had not moved at all. A returning player left at a known level and comes back to a competition whose standing relative to his last one is exactly what the ladder measures. A first-timer has no such anchor.



## By what the player already was

The distinction the review insisted on. A translation has real work to do only for a player with no record in the competition he is moving to; for anyone else his own record there is the better evidence.


**First season in the competition** — 320 moves, 289 players

| predictor | n | mae | rmse | bias |
| --- | --- | --- | --- | --- |
| translation (ladder) | 320 | 23.62 | 29.64 | 2.05 |
| conditional model | 320 | 20.08 | 23.88 | -0.19 |
| no translation | 320 | 24.61 | 30.67 | 2.10 |
| competition average (50) | 320 | 21.30 | 25.31 | 3.52 |

Translation against leaving the rating alone: **+0.99** points of error saved [95% CI -0.14, +2.07], not significant. Bootstrapped over players.


**Returning after a season away** — 121 moves, 114 players

| predictor | n | mae | rmse | bias |
| --- | --- | --- | --- | --- |
| translation (ladder) | 121 | 24.25 | 30.89 | 0.49 |
| conditional model | 121 | 20.24 | 24.25 | 1.04 |
| no translation | 121 | 26.15 | 32.42 | 3.99 |
| competition average (50) | 121 | 21.23 | 25.26 | 4.66 |

Translation against leaving the rating alone: **+1.90** points of error saved [95% CI +0.29, +3.48], significant. Bootstrapped over players.



## Into the NRL specifically

The Leeds question in its own right — a feeder player moving up.

| cohort | n_players | predictor | n | mae | rmse | bias |
| --- | --- | --- | --- | --- | --- | --- |
| first | 72 | translation (ladder) | 72 | 22.90 | 28.76 | 5.91 |
| first | 72 | conditional model | 72 | 19.16 | 22.79 | 2.06 |
| first | 72 | no translation | 72 | 23.28 | 28.24 | -3.15 |
| first | 72 | competition average (50) | 72 | 19.26 | 23.32 | 0.87 |
| returning | 18 | translation (ladder) | 18 | 16.22 | 22.31 | -1.42 |
| returning | 18 | conditional model | 18 | 17.33 | 20.78 | -0.21 |
| returning | 18 | no translation | 18 | 18.25 | 24.66 | -10.53 |
| returning | 18 | competition average (50) | 18 | 19.85 | 23.91 | -0.81 |


## The direction-of-move hypothesis

An earlier report claimed the ladder under-corrects for a promoted player by a specific number of points and proposed fitting that correction. The claim was measured on a mixed cohort and is withdrawn. What can be said here, on a properly rolling basis, is the bias of each cohort — the average of actual minus projected, so a positive figure means the projection was too low.

| cohort | moves | bias | lo | hi |
| --- | --- | --- | --- | --- |
| first | 320 | 2.05 | -1.09 | 5.29 |
| returning | 121 | 0.49 | -4.97 | 6.20 |

A bias whose interval excludes zero is a real, repeatable offset and worth modelling. One that straddles zero is not, however tempting the point estimate.


**The hypothesis does not survive.** Every cohort's interval contains zero, so there is no repeatable direction-of-move offset to fit. The earlier figure of 3.8 points came from a single season, on a cohort that was mostly established players, and measured with a model that had already seen the season it was scored on. Three of those four problems are fixed here and the effect disappears. It should not be modelled, and the app should not carry a correction for it.


## Limitations

- The number of origins is small. Each one needs a full season of moves to predict and enough prior moves to fit on, which the data supports only a few times over.
- Super League directions rest on few moves at every origin, so their share of these figures is thin. The ladder borrows strength across directions through the common scale, which is an assumption the transitivity check supports but does not prove.
- The target is a season-only rating and therefore noisier than a cumulative one, which puts a floor under every error here. An earlier version of this line went on to claim the noise 'does not favour any predictor'. That was wrong, and the section above measures how wrong: the published target is shrunk toward 50, which is exactly what one of the baselines predicts, and the ladder's defeat holds against it while vanishing against the unshrunk target. Independent noise would indeed be even-handed; a systematic pull toward one predictor's answer is not noise.
- Post-contact metres are absent before 2025, so the composites behind the earlier origins rest on a slightly different stat set than the later ones.
