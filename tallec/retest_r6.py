# -*- coding: utf-8 -*-
"""The narrow re-test the sixth review asked for, specified before it was run.

What it fixes in the earlier tests (`team_role_trend.py`):

  * One cohort. Training and evaluation both use the explicit entry cohort built by
    `transition_events.py` (`rolling_backtest.moves_into`). Origin T is fitted on cohort
    moves that landed before T and scored on those landing in T. The earlier tests
    trained on `fit_translation_v3.build_pairs` pairs and scored on the cohort.
  * One baseline rule, the app's: a direction's own line from MIN_LINE_PAIRS training
    moves, otherwise the line over all training moves.
  * A history baseline. The review found that +0.43 of the trend's +0.50 came from
    whether a trend could be computed at all. `has_history` -- rated in the same source
    competition the season before -- is therefore a baseline of its own, and every
    candidate is scored against line + has_history.
  * A joint fit. Source rating, direction and each candidate are fitted together
    (ordinary least squares on the line terms, a ridge penalty of RIDGE_ALPHA on the
    standardised candidate and its missing flag), instead of correcting the residuals of
    a line fitted first.
  * A dated club. The new club is the first club he is seen playing for in the target
    season, not the club he played for most -- still read from the target season, so
    still retrospective, and labelled so.
  * Fewer candidates. Trend, the old team's points margin, the new club's points margin,
    the share of source matches started, and the incumbents at the new club. The
    look-ahead vacancy measure and the xLadder figures (absent from the first origin's
    training data) are left out.

Fixed before running: the candidates, the baselines, the fallback, RIDGE_ALPHA, the
primary comparison (each candidate against line + has_history on the 90 moves into
Super League, player-clustered bootstrap) and the secondary one (all moves).

Read-only against tallec.db.

    python retest_r6.py              # prints the result
    python retest_r6.py <folder>     # also writes the scored cohort
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

import rating_history as rh
import rolling_backtest as rb
import sp_schema as sp

LANDING = (2021, 2022, 2023, 2024, 2025)     # cohort seasons built
ORIGINS = (2023, 2024, 2025)                 # seasons forecast
MIN_LINE_PAIRS = 25
RIDGE_ALPHA = 1.0
CANDIDATES = {"trend": "trend", "old team margin": "old_margin",
              "new club margin": "new_margin", "source starts share": "start_share",
              "incumbents at the new club": "incumbents"}
OUT = sys.argv[1] if len(sys.argv) > 1 else None

con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)

# ── per-match facts (as in team_role_trend.py) ──────────────────────────────────
pms = pd.read_sql("SELECT player_id, competition comp, season, round, team, position, "
                  "position_source, minutes FROM player_match_stats", con)
raw = pd.read_sql('SELECT player_id, "Competition" comp, "Season" season, "Round" round, '
                  '"Team" team, "Opposition" opp, "Points Scored" pts, '
                  '"Interchange In - All" i_in, "Interchange Out - All" i_out '
                  'FROM player_match_raw', con)
for d_ in (pms, raw):
    d_["player_id"] = sp.normalize_player_id(d_.player_id)
raw["pts"] = pd.to_numeric(raw.pts, errors="coerce")
pm = pms.merge(raw[["player_id", "comp", "season", "round", "team", "i_in", "i_out"]],
               on=["player_id", "comp", "season", "round", "team"], how="left")
START = {(0, 0), (0, 1), (1, 2), (2, 3)}
BENCH = {(1, 0), (2, 1), (3, 2), (4, 3)}
key = list(zip(pm.i_in.fillna(-1).astype(int), pm.i_out.fillna(-1).astype(int)))
rule = np.array([1.0 if k in START else 0.0 if k in BENCH else np.nan for k in key])
rule = np.where(np.isnan(rule), (pm.minutes >= 40).astype(float), rule)
pm["started"] = np.where(pm.position_source == "match",
                         (pm.position != "Interchange").astype(float), rule)

ps = (pm.groupby(["player_id", "comp", "season"])
        .agg(team_mode=("team", lambda s: s.mode().iloc[0]), start_share=("started", "mean"))
        .reset_index())
first_club = (pm.sort_values("round").groupby(["player_id", "comp", "season"]).team.first()
                .rename("team_first"))

tm = raw.groupby(["comp", "season", "round", "team", "opp"], as_index=False).pts.sum()
both = tm.merge(tm, left_on=["comp", "season", "round", "team", "opp"],
                right_on=["comp", "season", "round", "opp", "team"], suffixes=("", "_o"))
both["margin"] = both.pts - both.pts_o
margin = both.groupby(["comp", "season", "team"]).margin.mean()


def features(r, sea, pos):
    r = r.copy()
    r["player_id"] = sp.normalize_player_id(r.player_id)
    s = sea[sea.n_games >= rb.MIN_GAMES].copy()
    s["player_id"] = sp.normalize_player_id(s.player_id)
    s = s.set_index(["player_id", "comp", "season"]).class_score
    now = np.array([s.get((p, c, y), np.nan) for p, c, y in zip(r.player_id, r.source, r.season_src)])
    before = np.array([s.get((p, c, y - 1), np.nan) for p, c, y in zip(r.player_id, r.source, r.season_src)])
    r["has_history"] = (~np.isnan(before)).astype(float)
    r["trend"] = now - before
    P = ps.set_index(["player_id", "comp", "season"])
    src = P.reindex(list(zip(r.player_id, r.source, r.season_src)))
    r["start_share"] = src.start_share.values
    r["old_margin"] = margin.reindex(list(zip(r.source, r.season_src, src.team_mode.values))).values
    r["new_team"] = first_club.reindex(list(zip(r.player_id, r.target, r.season_tgt))).values
    r["new_margin"] = margin.reindex(list(zip(r.target, r.season_tgt - 1, r.new_team))).values
    grp = lambda pid, comp: sp.POSITION_GROUP.get(pos.get((pid, comp)))
    r["group"] = [grp(p, c) for p, c in zip(r.player_id, r.source)]
    pg = pm.assign(grp=[grp(p, c) for p, c in zip(pm.player_id, pm.comp)])
    mins = pg.groupby(["comp", "season", "team", "grp", "player_id"]).minutes.sum()
    have = set(mins.index.droplevel(4))
    inc = []
    for row in r.itertuples():
        k = (row.target, row.season_tgt - 1, row.new_team, row.group)
        if pd.isna(row.new_team) or row.group is None or k not in have:
            inc.append(np.nan)
            continue
        w = mins.loc[k]
        w = w[w.index != row.player_id]
        rat = np.array([s.get((q, row.target, row.season_tgt - 1), np.nan) for q in w.index])
        ok = ~np.isnan(rat) & (w.values > 0)
        inc.append(float((rat[ok] * w.values[ok]).sum() / w.values[ok].sum()) if ok.any() else np.nan)
    r["incumbents"] = inc
    return r


# ── the cohort, every landing season, each built from what was known before it ──
rows = []
for o in LANDING:
    print(f"cohort landing in {o} ...", flush=True)
    pairs, cum, ext, pos, dob = rb.knowledge_at(con, o - 1)
    pend = rb.moves_into(con, o, cum, ext, pos, dob)
    if pend.empty:
        continue
    _, sea, _ = rh.all_competitions(rb.DB, through=o - 1)
    f = rb.fit(pairs, "B_next_season")
    if f is not None:                      # the shipped construction, for reference
        pend = rb.predict(f, pend)
    rows.append(features(pend, sea, pos))
C = pd.concat(rows, ignore_index=True)
C["pair"] = C.source + "->" + C.target


def line_terms(train, frame):
    """Columns for a per-direction line with the app's fallback, fitted on `train`."""
    big = [p for p, n in train.pair.value_counts().items() if n >= MIN_LINE_PAIRS]
    cols = [np.ones(len(frame)), frame.class_source.values]
    for p in big:
        on = (frame.pair == p).astype(float).values
        cols += [on, on * frame.class_source.values]
    return np.column_stack(cols), big


def fit_predict(train, test, extra):
    """OLS on the line terms, ridge RIDGE_ALPHA on standardised extras and flags."""
    Xl, big = line_terms(train, train)
    Tl, _ = line_terms(train, test)
    Xe, Te = [], []
    for c in extra:
        v, t = train[c].astype(float), test[c].astype(float)
        mu, sd = v.mean(), (v.std() or 1.0)
        if np.isnan(mu):
            continue
        Xe += [((v - mu) / sd).fillna(0.0).values]
        Te += [((t - mu) / sd).fillna(0.0).values]
        if c != "has_history":
            Xe.append(v.isna().astype(float).values)
            Te.append(t.isna().astype(float).values)
    X = np.column_stack([Xl] + Xe) if Xe else Xl
    T = np.column_stack([Tl] + Te) if Te else Tl
    k = X.shape[1] - Xl.shape[1]
    A = np.vstack([X, np.hstack([np.zeros((k, Xl.shape[1])), np.sqrt(RIDGE_ALPHA) * np.eye(k)])])
    b = np.concatenate([train.class_target.values, np.zeros(k)])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.clip(T @ coef, 0, 100), big


SPECS = {"line": [], "line + has_history": ["has_history"]}
for name, col in CANDIDATES.items():
    SPECS[f"+ {name}"] = ["has_history", col]
SPECS["+ all candidates"] = ["has_history"] + list(CANDIDATES.values())

scored, fits = [], []
for T in ORIGINS:
    tr, te = C[C.season_tgt < T], C[C.season_tgt == T].copy()
    for name, extra in SPECS.items():
        te[f"p::{name}"], big = fit_predict(tr, te, extra)
    te["line_basis"] = np.where(te.pair.isin(big), "direction", "pooled")
    fits.append(dict(origin=T, train=len(tr), test=len(te), sl_test=int((te.target == "SL").sum()),
                     sl_on_pooled=int(((te.target == "SL") & (te.line_basis == "pooled")).sum()),
                     directions_with_own_line=len(big)))
    scored.append(te)
d = pd.concat(scored, ignore_index=True)

print("\nfolds:")
print(pd.DataFrame(fits).to_string(index=False))
mae = lambda g, c: float((g.class_target - g[c]).abs().mean())
for label, g in (("into Super League (primary)", d[d.target == "SL"]), ("all moves", d)):
    print(f"\n{label}, n={len(g)}, {g.player_id.nunique()} players")
    print(f"  line (cohort-trained, app fallback)  MAE {mae(g, 'p::line'):.2f}")
    if "line_direction" in g:
        print(f"  reference: shipped-construction line {mae(g, 'line_direction'):.2f}, "
              f"conditional model {mae(g, 'model'):.2f}")
    b = rb.boot(g, "p::line", "p::line + has_history")
    print(f"  line + has_history vs line           MAE {mae(g, 'p::line + has_history'):.2f}  "
          f"gain {b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}]")
    for name in [k for k in SPECS if k.startswith("+ ")]:
        b = rb.boot(g, "p::line + has_history", f"p::{name}")
        print(f"  {name:36s} MAE {mae(g, f'p::{name}'):.2f}  gain over history baseline "
              f"{b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}]" + ("  clear" if b[1] > 0 or b[2] < 0 else ""))
    print("  by origin, MAE:")
    cols = [c for c in g if c.startswith("p::")]
    tab = pd.DataFrame({c[3:]: g.groupby("season_tgt").apply(
        lambda x, c=c: float((x.class_target - x[c]).abs().mean()), include_groups=False)
        for c in cols}).round(2)
    print(tab.T.to_string())

cov = d[["has_history"] + list(CANDIDATES.values())].notna().mean().round(2)
cov["has_history (share with history)"] = round(float(d.has_history.mean()), 2)
print("\nshare of evaluated moves with each candidate present:")
print(cov.to_string())
if OUT:
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "retest_scored.csv"), index=False)
    C.to_csv(os.path.join(OUT, "retest_cohort.csv"), index=False)
con.close()
