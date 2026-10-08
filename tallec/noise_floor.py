# -*- coding: utf-8 -*-
"""How much of the translation error is measurement noise in the target itself.

The backtest scores a forecast against the player's season-only rating in the
competition he moved to. That rating is a mean over a finite number of matches, so it
carries noise of its own: even a forecaster who knew the player's true level for that
season would miss by that noise. Its expected size is a floor under any method's error.

For the 90 moves into Super League this rebuilds the backtest's cohort, forecasts and
target exactly (and says so), then for each mover simulates the target's noise through
the engine's own formula: a season mean with variance sigma^2/n, shrunk by B, mapped by
100 * Phi((z - centre) / tau). A split-half check on every Super League player with six
or more matches in 2023-2025 tests whether sigma^2/n is the right size for that noise.
As a reference rather than a floor, it also scores Super League players who stayed,
forecast from their own previous Super League season through a straight line fitted
walk-forward. Last, it compares the model with the straight line on less noisy targets:
movers with 16 or 20+ target matches, and all 90 with the simulated noise variance
subtracted from each squared error.

Read-only: opens tallec.db read-only and writes nothing unless given an output folder.

    python noise_floor.py              # prints the result
    python noise_floor.py <folder>     # also writes the per-mover and split-half CSVs
"""
import math
import os
import sqlite3
import sys

import numpy as np
import pandas as pd


import player_rating_engine as pre
import rating_history as rh
import rolling_backtest as rb

ORIGINS = (2023, 2024, 2025)
RNG = np.random.default_rng(0)
DRAWS = 20000
OUT = sys.argv[1] if len(sys.argv) > 1 else None


def g(z, c, tau):
    """The engine's published map: 100 * Phi((z - centre) / tau)."""
    return 100.0 * 0.5 * (1.0 + np.vectorize(math.erf)((z - c) / tau / math.sqrt(2.0)))


con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)

# ── 1. the 90 moves, exactly as the backtest builds and scores them ───────────
rows = []
for origin in ORIGINS:
    pairs, cum, ext, pos, dob = rb.knowledge_at(con, origin - 1)
    f = rb.fit(pairs, "B_next_season")
    pend = rb.moves_into(con, origin, cum, ext, pos, dob)
    out = rb.predict(f, pend)
    out["origin"] = origin
    rows.append(out)
d = pd.concat(rows, ignore_index=True)
sl = d[d.target == "SL"].copy()
mae = lambda col: float((sl.class_target - sl[col]).abs().mean())
print(f"moves into SL: {len(sl)}  (report: 90)")
print(f"MAE model {mae('model'):.2f} (report 17.98) | line {mae('line_direction'):.2f} | "
      f"flat50 {float((sl.class_target - 50).abs().mean()):.2f} (report 20.64) | "
      f"carry-over {mae('class_source'):.2f} (report 25.77)")

# ── 2. the target-season engine for each SL season, to get sigma, tau, B, centre ──
eng_rows, split_rows = [], []
for season in ORIGINS:
    hist = rh.load(con, "SL", through=season)
    force, _ = rh._mode(hist)
    pm = rh.composites(hist, "SL", force)
    this = pm[pm.season == season]
    eng = pre.PlayerRatingEngine("SL", force_mode=force)
    snap = eng.compute_snapshot(pm=this)
    snap["season"] = season
    snap["sigma2"], snap["tau2"], snap["mu"] = eng.sigma2, eng.tau2, eng.grand_mean
    eng_rows.append(snap)

    # split-half check of the noise model: alternate matches in round order
    r = this[this.ratable].sort_values(["player_id", "round"])
    for pid, gp in r.groupby("player_id"):
        if len(gp) < 6:
            continue
        c = gp.composite.values
        a, b = c[0::2], c[1::2]
        split_rows.append(dict(season=season, player_id=pid, n_a=len(a), n_b=len(b),
                               diff2=(a.mean() - b.mean()) ** 2,
                               expected=eng.sigma2 * (1 / len(a) + 1 / len(b))))
E = pd.concat(eng_rows, ignore_index=True)
S = pd.DataFrame(split_rows)

E = E.rename(columns={"n_games": "n_tgt"})
m = sl.drop(columns=[c for c in ("n_games", "shrinkage_B", "class_z", "raw_composite",
                                  "scale_centre", "class_score", "raw_score")
                     if c in sl.columns]).merge(E[["player_id", "season", "n_tgt", "shrinkage_B", "class_z",
                "raw_composite", "scale_centre", "class_score", "raw_score",
                "sigma2", "tau2", "mu"]],
             left_on=["player_id", "season_tgt"], right_on=["player_id", "season"],
             how="left")
miss = m.class_score.isna().sum()
gap = float((m.class_score - m.class_target).abs().max())
print(f"n_tgt equals backtest n_target: {bool((m.n_tgt == m.n_target).all())}")
print(f"target reproduced for {len(m) - miss}/{len(m)} movers, max |diff| {gap:.6f}")

# ── 3. Monte Carlo floor per mover ─────────────────────────────────────────────
fl_pub, fl_raw, nv_pub = [], [], []
for r in m.itertuples():
    tau = math.sqrt(r.tau2)
    e = RNG.normal(0.0, math.sqrt(r.sigma2 / r.n_tgt), DRAWS)
    T = r.class_z                       # posterior mean of his true season level
    B = r.shrinkage_B
    y0 = g(np.array([r.mu + B * (T - r.mu)]), r.scale_centre, tau)[0]
    ys = g(r.mu + B * (T + e - r.mu), r.scale_centre, tau)
    fl_pub.append(float(np.abs(ys - y0).mean()))
    nv_pub.append(float(((ys - y0) ** 2).mean()))
    yr0 = g(np.array([T]), r.scale_centre, tau)[0]
    yrs = g(T + e, r.scale_centre, tau)
    fl_raw.append(float(np.abs(yrs - yr0).mean()))
m["floor_published"], m["floor_raw"], m["noise_var"] = fl_pub, fl_raw, nv_pub

# sensitivity: locate the true level at the unshrunk mean instead
fl_alt = []
for r in m.itertuples():
    tau = math.sqrt(r.tau2)
    e = RNG.normal(0.0, math.sqrt(r.sigma2 / r.n_tgt), DRAWS)
    T, B = r.raw_composite, r.shrinkage_B
    y0 = g(np.array([r.mu + B * (T - r.mu)]), r.scale_centre, tau)[0]
    fl_alt.append(float(np.abs(g(r.mu + B * (T + e - r.mu), r.scale_centre, tau) - y0).mean()))
m["floor_published_alt"] = fl_alt

print("\nnoise floor on the published target (mean over the 90):")
print(f"  {m.floor_published.mean():.2f}  (true level at the shrunk estimate)")
print(f"  {m.floor_published_alt.mean():.2f}  (true level at the unshrunk mean)")
print(f"noise floor on the unshrunk target: {m.floor_raw.mean():.2f}")
print(f"target matches: median {m.n_tgt.median():.0f}, quartiles "
      f"{m.n_tgt.quantile(.25):.0f}-{m.n_tgt.quantile(.75):.0f}; "
      f"B median {m.shrinkage_B.median():.2f}")
print(f"sigma2 by season: " + ", ".join(f"{s}: {v:.4f}" for s, v in
      E.groupby('season').sigma2.first().items()))
print(f"tau2 by season:   " + ", ".join(f"{s}: {v:.4f}" for s, v in
      E.groupby('season').tau2.first().items()))

print("\nsplit-half check (SL 2023-25, players with >= 6 matches):")
print(f"  players {len(S)}; observed / expected squared half-difference "
      f"{S.diff2.mean() / S.expected.mean():.3f}  (1.0 = sigma^2/n is the right noise)")
for s, gs in S.groupby("season"):
    print(f"    {s}: {gs.diff2.mean() / gs.expected.mean():.3f}  (n={len(gs)})")

# by match count
m["n_band"] = pd.cut(m.n_tgt, [2, 5, 10, 15, 40], labels=["3-5", "6-10", "11-15", "16+"])
print("\nby target matches:")
print(m.groupby("n_band", observed=True).agg(
    moves=("player_id", "size"),
    floor=("floor_published", "mean"),
    model_mae=("model", lambda s_: float((m.loc[s_.index, "class_target"] - s_).abs().mean())),
    line_mae=("line_direction", lambda s_: float((m.loc[s_.index, "class_target"] - s_).abs().mean())),
).round(2).to_string())

# ── 4. second benchmark: SL players who stayed, own previous SL season ──────────
# walk-forward line fitted on (S-2 -> S-1), applied to (S-1 -> S), season-only ratings
_, sea_all, _ = rh.all_competitions(rb.DB, comps=("SL",), through=max(ORIGINS))
sea_all = sea_all[sea_all.n_games >= rb.MIN_GAMES]
movers = set(zip(sl.player_id, sl.season_tgt))
stay = []
for season in ORIGINS:
    prev = sea_all[sea_all.season == season - 1][["player_id", "class_score"]]
    now = sea_all[sea_all.season == season][["player_id", "class_score", "n_games"]]
    tr_x = sea_all[sea_all.season == season - 2][["player_id", "class_score"]].merge(
        prev, on="player_id", suffixes=("_x", "_y"))
    if len(tr_x) < 30:
        continue
    k, c0 = np.polyfit(tr_x.class_score_x, tr_x.class_score_y, 1)
    j = prev.merge(now, on="player_id", suffixes=("_prev", ""))
    j = j[[(p, season) not in movers for p in j.player_id]]
    j["pred"] = np.clip(k * j.class_score_prev + c0, 0, 100)
    j["season"] = season
    stay.append(j)
st = pd.concat(stay, ignore_index=True)
st_small = st[st.n_games <= m.n_tgt.quantile(.75)]
print(f"\nSL stayers, own previous SL season through a straight line: "
      f"{len(st)} player-seasons, MAE {float((st.class_score - st.pred).abs().mean()):.2f}; "
      f"with no more target matches than the movers' upper quartile ({len(st_small)}): "
      f"{float((st_small.class_score - st_small.pred).abs().mean()):.2f}")

# ── 5. the model against the line on less noisy targets ───────────────────────
# Noise in the target is the same for both predictors, so it does not move their
# difference in expectation; it widens the interval. Two ways to take it out, neither of
# which touches 2026: score only movers with many target matches, and subtract the
# simulated noise variance from each predictor's squared error on all 90.
def mae_line_model(g):
    b = rb.boot(g, "line_direction", "model")
    e = lambda c: float((g.class_target - g[c]).abs().mean())
    return e("model"), e("line_direction"), b


print("\nmodel against the straight line, positive = model closer:")
for label, sub in (("all 90, season-only target", m),
                   ("16+ target matches", m[m.n_tgt >= 16]),
                   ("20+ target matches", m[m.n_tgt >= 20])):
    mm, ml, b = mae_line_model(sub)
    print(f"  {label:28s} n={len(sub):3d}  floor {sub.floor_published.mean():5.2f}  "
          f"MAE model {mm:5.2f}  line {ml:5.2f}  diff {b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}]")

se_m = (m.class_target - m.model) ** 2
se_l = (m.class_target - m.line_direction) ** 2
nv = float(m.noise_var.mean())
true_rmse = lambda se: math.sqrt(max(float(se.mean()) - nv, 0.0))
diff = (se_l - se_m).values
pids = m.player_id.values
uniq = pd.unique(pids)
at = {q: np.where(pids == q)[0] for q in uniq}
bs = np.empty(4000)
rng = np.random.default_rng(0)
for i in range(4000):
    pick = rng.choice(uniq, uniq.size, replace=True)
    idx = np.concatenate([at[q] for q in pick])
    bs[i] = true_rmse(se_l.iloc[idx]) - true_rmse(se_m.iloc[idx])
print(f"  noise removed, all 90 (RMSE against the true season level): "
      f"model {true_rmse(se_m):.2f}  line {true_rmse(se_l):.2f}  "
      f"diff {true_rmse(se_l) - true_rmse(se_m):+.2f} "
      f"[{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]  "
      f"(observed RMSE model {math.sqrt(se_m.mean()):.2f}, line {math.sqrt(se_l.mean()):.2f}; "
      f"noise SD {math.sqrt(nv):.2f})")

if OUT:
    os.makedirs(OUT, exist_ok=True)
    m.to_csv(os.path.join(OUT, "movers_floor.csv"), index=False)
    S.to_csv(os.path.join(OUT, "split_half.csv"), index=False)
con.close()
