# -*- coding: utf-8 -*-
"""Does team context, expected role or trend add anything to the straight line?

Rolling origins 2023-2025, the same cohort and lines as rolling_backtest.py. For each
origin the line is fitted on the training pairs, then a linear model of the line's
residuals on the new features is fitted on the same training pairs, and the forecast is
line + that correction. Scored against the line, bootstrapped over players.

Features, all from data already held:
  trend               change in his season-only rating over his last two source seasons
  source starts share share of source matches he started: the match sheet where there
                      is one, else the interchange counts (exact wherever they decide,
                      100% on every match-sheet row), else 40+ minutes (84% on the rest)
  old/new margin      points margin per game of his source team, and of his new club in
                      the season before he joined, from the player rows (NRL r=0.997
                      against the xLadder master; SL 2022-25 within 1.1 points)
  old/new xLadder     the xLadder's expected points per game (2 x win probability),
                      NRL and SL masters only
  role (a)            minutes-weighted rating of the new club's incumbents in his
                      position group, the season before he joined
  role (b)            share of those minutes played by men who do not appear for the
                      club in the target season. LOOK-AHEAD: it uses the season being
                      forecast, and an injured incumbent looks like a departure.
The new club itself is read from the target season, which is known at signing.

Read-only against tallec.db and the xLadder masters.

    python team_role_trend.py            # prints the comparison
    python team_role_trend.py <file>     # also writes the scored moves to a CSV
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd


import fit_translation_v3 as f3
import rating_history as rh
import rolling_backtest as rb
import sp_schema as sp
import team_map as tmap

ORIGINS = (2023, 2024, 2025)
con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)

# ── per-match facts ────────────────────────────────────────────────────────────
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

# started: match sheet where it exists; else the interchange counts (exact where they
# decide, 100% on match-sheet rows); else 40+ minutes (84% on the ambiguous rows)
START = {(0, 0), (0, 1), (1, 2), (2, 3)}
BENCH = {(1, 0), (2, 1), (3, 2), (4, 3)}
key = list(zip(pm.i_in.fillna(-1).astype(int), pm.i_out.fillna(-1).astype(int)))
rule = np.array([1.0 if k in START else 0.0 if k in BENCH else np.nan for k in key])
rule = np.where(np.isnan(rule), (pm.minutes >= 40).astype(float), rule)
pm["started"] = np.where(pm.position_source == "match",
                         (pm.position != "Interchange").astype(float), rule)

# player-season facts: modal team, starts share, minutes per game
ps = (pm.groupby(["player_id", "comp", "season"])
        .agg(team=("team", lambda s: s.mode().iloc[0]), games=("round", "size"),
             start_share=("started", "mean"), mins_pg=("minutes", "mean"))
        .reset_index())

# team strength 1: points margin per game, from the player rows (validated against the
# xLadder masters: NRL r=0.997, SL 2022-25 within 1.1 points)
tm = raw.groupby(["comp", "season", "round", "team", "opp"], as_index=False).pts.sum()
both = tm.merge(tm, left_on=["comp", "season", "round", "team", "opp"],
                right_on=["comp", "season", "round", "opp", "team"], suffixes=("", "_o"))
both["margin"] = both.pts - both.pts_o
margin = both.groupby(["comp", "season", "team"]).margin.mean().rename("margin_pg")

# team strength 2: xLadder expected points per game (NRL and SL masters only)
xl = []
for comp, cols in (("NRL", ["Season", "A Team", "B Team", "WL_Prob_A_v2", "A_Points Scored"]),
                   ("SL", ["Season", "A Team", "B Team", "WL_Prob_A_v2", "A_Points Scored"])):
    mp, _ = tmap.solve(comp)
    m = tmap.load_master(comp, cols)
    m = m[m["A_Points Scored"].notna() & m.WL_Prob_A_v2.notna()]
    a = pd.DataFrame({"season": m.Season, "team": m["A Team"].map(mp), "e": 2 * m.WL_Prob_A_v2})
    b = pd.DataFrame({"season": m.Season, "team": m["B Team"].map(mp), "e": 2 * (1 - m.WL_Prob_A_v2)})
    g = pd.concat([a, b]).groupby(["season", "team"]).e.mean().reset_index()
    g["comp"] = comp
    xl.append(g)
xladder = pd.concat(xl).set_index(["comp", "season", "team"]).e.rename("xl_eppg")


def features(rows, sea, pos):
    """Add the new features to rows with player_id, source, target, season_src/tgt."""
    r = rows.copy()
    r["player_id"] = sp.normalize_player_id(r.player_id)
    # trend: season-only rating in the source, last two seasons
    s = sea[sea.n_games >= rb.MIN_GAMES].set_index(["player_id", "comp", "season"]).class_score
    s.index = s.index.set_levels([sp.normalize_player_id(pd.Series(s.index.levels[0])).values,
                                  s.index.levels[1], s.index.levels[2]])
    now = [s.get((p, c, y), np.nan) for p, c, y in zip(r.player_id, r.source, r.season_src)]
    before = [s.get((p, c, y - 1), np.nan) for p, c, y in zip(r.player_id, r.source, r.season_src)]
    r["trend"] = np.array(now) - np.array(before)
    # source role and old team
    P = ps.set_index(["player_id", "comp", "season"])
    src = P.reindex(list(zip(r.player_id, r.source, r.season_src)))
    r["start_share"] = src.start_share.values
    r["old_team"] = src.team.values
    r["old_margin"] = margin.reindex(list(zip(r.source, r.season_src, r.old_team))).values
    r["old_xl"] = xladder.reindex(list(zip(r.source, r.season_src, r.old_team))).values
    # new club: known at signing, taken from the target season's appearances
    tgt = P.reindex(list(zip(r.player_id, r.target, r.season_tgt)))
    r["new_team"] = tgt.team.values
    prev = r.season_tgt - 1
    r["new_margin"] = margin.reindex(list(zip(r.target, prev, r.new_team))).values
    r["new_xl"] = xladder.reindex(list(zip(r.target, prev, r.new_team))).values
    # role (a): minutes-weighted rating of the new club's incumbents in his position
    # group, previous season. role (b): share of those minutes played by men who no
    # longer appear for the club in the target season -- LOOK-AHEAD, flagged.
    grp = lambda pid, comp: sp.POSITION_GROUP.get(pos.get((pid, comp)))
    r["group"] = [grp(p, c) for p, c in zip(r.player_id, r.source)]
    pg = pm.assign(grp=[grp(p, c) for p, c in zip(pm.player_id, pm.comp)])
    mins = pg.groupby(["comp", "season", "team", "grp", "player_id"]).minutes.sum()
    roster = pg.groupby(["comp", "season", "team"]).player_id.agg(set)
    inc_a, inc_b = [], []
    for row in r.itertuples():
        k = (row.target, row.season_tgt - 1, row.new_team, row.group)
        if pd.isna(row.new_team) or row.group is None or k not in mins.index.droplevel(4):
            inc_a.append(np.nan); inc_b.append(np.nan); continue
        w = mins.loc[k]
        w = w[w.index != row.player_id]
        if w.empty or w.sum() <= 0:
            inc_a.append(np.nan); inc_b.append(np.nan); continue
        rat = np.array([s.get((p, row.target, row.season_tgt - 1), np.nan) for p in w.index])
        ok = ~np.isnan(rat)
        inc_a.append(float((rat[ok] * w.values[ok]).sum() / w.values[ok].sum()) if ok.any() else np.nan)
        stay = roster.get((row.target, row.season_tgt, row.new_team), set())
        inc_b.append(float(w[[q not in stay for q in w.index]].sum() / w.sum()))
    r["incumbents"] = inc_a
    r["vacated_LOOKAHEAD"] = inc_b
    return r


FEATS = {"trend": ["trend"], "source starts share": ["start_share"],
         "old team margin": ["old_margin"], "new club margin": ["new_margin"],
         "old team xLadder": ["old_xl"], "new club xLadder": ["new_xl"],
         "role (a) incumbents": ["incumbents"],
         "role (b) vacated [look-ahead]": ["vacated_LOOKAHEAD"]}
FEATS["all, no look-ahead"] = sum([v for k, v in FEATS.items() if "look-ahead" not in k], [])
FEATS["all, with look-ahead"] = sum([v for k, v in FEATS.items() if k.startswith(("trend", "source", "old", "new", "role"))], [])


def design(df, cols, stats=None):
    X = []
    st = stats or {}
    for c in cols:
        v = df[c].astype(float)
        mu, sd = st.get(c, (v.mean(), v.std() or 1.0))
        st[c] = (mu, sd)
        X.append(((v - mu) / sd).fillna(0.0).values)
        X.append(v.isna().astype(float).values)            # missing flag
    X.append(np.ones(len(df)))
    return np.column_stack(X), st


scored, cover = [], []
for origin in ORIGINS:
    print(f"origin {origin} ...", flush=True)
    pairs, cum, ext, pos, dob = rb.knowledge_at(con, origin - 1)
    _, sea, _ = rh.all_competitions(rb.DB, through=origin - 1)
    f = rb.fit(pairs, "B_next_season")
    pend = rb.predict(f, rb.moves_into(con, origin, cum, ext, pos, dob))
    tr = pairs[pairs.layer == "B_next_season"].copy()
    tr["line"] = rb.apply_lines(f["lines"], tr, per_direction=True)
    tr = features(tr, sea, pos)
    te_ = features(pend, sea, pos)
    te_["origin"] = origin
    cover.append(te_[list(dict.fromkeys(sum(FEATS.values(), [])))].notna().mean().rename(origin))
    for name, cols in FEATS.items():
        cols = list(dict.fromkeys(cols))
        X, st = design(tr, cols)
        coef, *_ = np.linalg.lstsq(X, (tr.class_target - tr.line).values, rcond=None)
        Xt, _ = design(te_, cols, st)
        te_[f"ext::{name}"] = np.clip(te_.line_direction + Xt @ coef, 0, 100)
    scored.append(te_)

d = pd.concat(scored, ignore_index=True)
print("\nfeature coverage on the evaluated moves (share non-missing), by origin:")
print(pd.concat(cover, axis=1).round(2).to_string())

print("\nline + feature against the line alone, positive = the feature helps (MAE points):")
for label, sub in (("all moves", d), ("into Super League", d[d.target == "SL"])):
    print(f"  {label} (n={len(sub)}), line MAE {float((sub.class_target - sub.line_direction).abs().mean()):.2f}")
    for name in FEATS:
        b = rb.boot(sub, "line_direction", f"ext::{name}")
        mae = float((sub.class_target - sub[f'ext::{name}']).abs().mean())
        print(f"    {name:32s} MAE {mae:6.2f}  gain {b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}]"
              + ("  clear" if (b[1] > 0 or b[2] < 0) else ""))
if len(sys.argv) > 1:
    d.to_csv(sys.argv[1], index=False)
con.close()
