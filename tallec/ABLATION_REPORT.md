# Which translation features earn their place

Every specification is fitted and scored exactly as `rolling_backtest.py` fits and scores the live model, over the same rolling origins. Features outside a specification are zeroed after the design matrix is built, so the comparison is between identical designs rather than between differently shaped ones.


441 moves over origins [2023, 2024, 2025].


## Error by specification

| specification | n | mae | rmse |
| --- | --- | --- | --- |
| + position x pair (pooled) | 441 | 19.448 | 23.899 |
| everything incl. interaction | 441 | 19.561 | 24.152 |
| source + pair + position | 441 | 20.121 | 23.979 |
| + load (mins, games) | 441 | 20.209 | 24.148 |
| + age | 441 | 20.241 | 24.125 |
| source + pair | 441 | 20.242 | 23.989 |
| full model | 441 | 20.272 | 24.250 |
| source + position | 441 | 20.500 | 24.255 |
| source only | 441 | 20.629 | 24.306 |


## Against the full model

Negative means the reduced specification is **better**. The interval is a bootstrap over players, not rows, because the same man appears in several moves and a row bootstrap would be too narrow.

| specification | mae_gap_vs_full | ci_low | ci_high |
| --- | --- | --- | --- |
| source only | 0.357 | -0.304 | 1.031 |
| source + pair | -0.030 | -0.428 | 0.357 |
| source + position | 0.228 | -0.342 | 0.812 |
| source + pair + position | -0.151 | -0.411 | 0.111 |
| + load (mins, games) | -0.063 | -0.216 | 0.079 |
| + age | -0.031 | -0.252 | 0.195 |
| + position x pair (pooled) | -0.823 | -1.965 | 0.294 |
| everything incl. interaction | -0.711 | -1.837 | 0.408 |


## The same comparison, origin by origin

Positive means the specification beats the full model in that season. A mean over three origins can hide something that has stopped working, and for the interaction term it does.

| specification | 2023 | 2024 | 2025 |
| --- | --- | --- | --- |
| source only | +0.347 | -0.447 | -0.959 |
| source + pair | +0.488 | -0.353 | -0.093 |
| source + position | +0.513 | -0.344 | -0.846 |
| source + pair + position | +0.664 | -0.194 | -0.057 |
| + load (mins, games) | +0.180 | -0.023 | +0.021 |
| + age | +0.349 | -0.182 | -0.099 |
| + position x pair (pooled) | +1.597 | +1.899 | -0.813 |
| full model | +0.000 | +0.000 | +0.000 |
| everything incl. interaction | +1.138 | +2.032 | -0.796 |


## What this says

The best specification is **+ position x pair (pooled)** at 19.448 MAE, against the full model's 20.272.
No reduced specification beats the full model with an interval clear of zero.


## Cohort: first

| specification | n | mae |
| --- | --- | --- |
| everything incl. interaction | 320 | 19.602 |
| + position x pair (pooled) | 320 | 19.609 |
| + load (mins, games) | 320 | 20.015 |
| full model | 320 | 20.071 |
| source + pair + position | 320 | 20.077 |
| + age | 320 | 20.214 |
| source + pair | 320 | 20.239 |
| source + position | 320 | 20.307 |
| source only | 320 | 20.483 |


## Cohort: returning

| specification | n | mae |
| --- | --- | --- |
| + position x pair (pooled) | 121 | 19.024 |
| everything incl. interaction | 121 | 19.454 |
| source + pair + position | 121 | 20.239 |
| source + pair | 121 | 20.249 |
| + age | 121 | 20.312 |
| + load (mins, games) | 121 | 20.724 |
| full model | 121 | 20.802 |
| source + position | 121 | 21.012 |
| source only | 121 | 21.013 |
