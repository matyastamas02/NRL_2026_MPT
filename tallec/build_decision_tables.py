# -*- coding: utf-8 -*-
"""The small tables behind the Comparison page's decision outputs.

The post-2026 exploration (`explore_post2026.py`, `POST2026_EXPLORE.md`) found two things
the rating forecast alone does not say. Whether an entrant finished in the top third of
his new competition is predictable from the line forecast: the third with the highest
forecast reached it 64% of the time against 48% for all entrants into Super League. Who
becomes a regular starter is not: the rating's AUC was below 0.5, while whether he had
been starting in the source competition reached 0.64. Players who came mostly off the
bench carried higher per-minute ratings and became starters less often. Both need data
the deployed app does not hold (every competition-season's full rating table, and the
interchange columns of the raw export), so they are computed here and stored.

  player_season_starts  one row per player, competition and season: matches, starts,
                        his club's matches, and starter = started at least half of them.
                        A start is the match-sheet position where one exists, else the
                        interchange counts (exact where they decide), else 40+ minutes.
  decision_calibration  P(top third | line forecast): a logistic fit on every
                        next-season pair the app's lines are fitted on, the label being
                        a season rating in the top third of every rated player in the
                        target competition that season. Through the freeze season.
  bench_evidence        per target competition, from the entry cohort the comparison
                        card reads: how many entrants had started at least half their
                        source-season matches, how many had not, and what share of each
                        became regular starters in the new competition.

    python build_decision_tables.py            # dry run: print what would be written
    python build_decision_tables.py --write    # replace the three tables, guarded
"""
import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import player_rating_engine as pre
import rating_history as rh
import rolling_backtest as rb
import runtime
import sp_schema as sp

START = {(0, 0), (0, 1), (1, 2), (2, 3)}
BENCH = {(1, 0), (2, 1), (3, 2), (4, 3)}
MIN_LINE_PAIRS = 25
LAYER = "B_next_season"


def season_starts(con):
    pms = pd.read_sql("SELECT player_id, competition comp, season, round, team, position, "
                      "position_source, minutes FROM player_match_stats", con)
    raw = pd.read_sql('SELECT player_id, "Competition" comp, "Season" season, '
                      '"Round" round, "Team" team, "Interchange In - All" i_in, '
                      '"Interchange Out - All" i_out FROM player_match_raw', con)
    for d in (pms, raw):
        d["player_id"] = sp.normalize_player_id(d.player_id)
    pm = pms.merge(raw, on=["player_id", "comp", "season", "round", "team"], how="left")
    key = zip(pm.i_in.fillna(-1).astype(int), pm.i_out.fillna(-1).astype(int))
    rule = np.array([1.0 if k in START else 0.0 if k in BENCH else np.nan for k in key])
    rule = np.where(np.isnan(rule), (pm.minutes >= 40).astype(float), rule)
    pm["started"] = np.where(pm.position_source == "match",
                             (pm.position != "Interchange").astype(float), rule)
    team_n = pm.groupby(["comp", "season", "team"])["round"].nunique()
    ps = (pm.groupby(["player_id", "comp", "season"])
            .agg(team=("team", lambda s: s.mode().iloc[0]), games=("round", "size"),
                 starts=("started", "sum"))
            .reset_index())
    ps["team_games"] = team_n.reindex(list(zip(ps.comp, ps.season, ps.team))).values
    ps["starts"] = ps.starts.astype(int)
    ps["start_share"] = ps.starts / ps.games
    ps["starter"] = (ps.starts >= 0.5 * ps.team_games).astype(int)
    return ps


def line_fitted(p):
    """The app's rule: own line with 25+ pairs and a varying source, else all pairs."""
    out = np.empty(len(p))
    pooled = np.polyfit(p.class_source, p.class_target, 1)
    for pair, idx in p.groupby("pair").indices.items():
        g = p.iloc[idx]
        coef = (np.polyfit(g.class_source, g.class_target, 1)
                if len(g) >= MIN_LINE_PAIRS and g.class_source.std() > 0 else pooled)
        out[idx] = np.clip(np.polyval(coef, g.class_source), 0, 100)
    return out


def rolling_line(p, min_train=100):
    """Each pair's line forecast from the pairs of earlier target seasons only."""
    out = np.full(len(p), np.nan)
    for s in sorted(p.season_tgt.unique()):
        tr = p[p.season_tgt < s]
        if len(tr) < min_train:
            continue
        te = np.flatnonzero((p.season_tgt == s).values)
        pooled = np.polyfit(tr.class_source, tr.class_target, 1)
        for pair in p.pair.iloc[te].unique():
            g = tr[tr.pair == pair]
            coef = (np.polyfit(g.class_source, g.class_target, 1)
                    if len(g) >= MIN_LINE_PAIRS and g.class_source.std() > 0 else pooled)
            idx = te[(p.pair.iloc[te] == pair).values]
            out[idx] = np.clip(np.polyval(coef, p.class_source.iloc[idx]), 0, 100)
    return out


def calibration(con):
    freeze = pre.FREEZE_SEASON
    p = pd.read_sql("SELECT player_id, source, target, season_tgt, class_source, "
                    "class_target FROM translation_pairs_v3 WHERE layer = ?", con,
                    params=(LAYER,))
    p["player_id"] = sp.normalize_player_id(p.player_id)
    p["pair"] = p.source + "->" + p.target
    _, sea, _ = rh.all_competitions(rb.DB, through=freeze)
    sea["player_id"] = sp.normalize_player_id(sea.player_id)
    rated = sea[sea.n_games >= rb.MIN_GAMES]
    thr = rated.groupby(["comp", "season"]).class_score.quantile(2 / 3)
    # The label is the target rating as the season tables hold it now. The pairs were
    # built with the frozen release, before the Ben Talty row was removed, and the
    # Queensland Cup 2025 pool moved with it (up to 2.8 points); every other target
    # must match exactly, or something else has changed.
    chk = p.merge(rated.rename(columns={"comp": "target", "season": "season_tgt"}),
                  on=["player_id", "target", "season_tgt"], how="left")
    d = (chk.class_target - chk.class_score).abs()
    moved = d > 1e-6
    known = (chk.target == "QLD") & (chk.season_tgt == 2025)
    if chk.class_score.isna().any() or (moved & ~known).any() or d.max() > 3:
        sys.exit(f"ABORT: pair targets differ from the season tables beyond the known "
                 f"Queensland Cup 2025 drift (max {d.max():.2f})")
    print(f"pair targets: {int(moved.sum())} Queensland Cup 2025 rows moved by up to "
          f"{d.max():.2f} since the pairs were built; labelled on the current ratings")
    p["top3"] = (chk.class_score.values
                 >= thr.reindex(list(zip(p.target, p.season_tgt))).values).astype(int)
    # The probability is fitted on forecasts made out of sample: each pair's line from
    # the pairs of earlier target seasons only, as the app would have forecast it. The
    # first version used each pair's own in-sample line fit, which overstated the top
    # end (an independent review measured 0.64 predicted against 0.57 observed in the
    # top fifth).
    x = rolling_line(p)
    ok = ~np.isnan(x)
    xo, yo = x[ok], p.top3.values[ok]
    mu, sd = xo.mean(), xo.std()
    m = LogisticRegression(C=1.0).fit(((xo - mu) / sd).reshape(-1, 1), yo)
    c0, c1 = float(m.intercept_[0]), float(m.coef_[0, 0])
    cal = pd.DataFrame([dict(kind="top_third", intercept=c0 - c1 * mu / sd, slope=c1 / sd,
                             n_pairs=int(ok.sum()), base_rate=float(yo.mean()),
                             fitted_through=int(freeze),
                             predictor="line forecast, out of sample (earlier seasons only)",
                             label="season rating in the top third of every rated "
                                   "player in the target competition that season")])
    base = (p.assign(top3=p.top3).groupby("target").top3.agg(["mean", "size"])
              .rename(columns={"mean": "base_rate", "size": "n_pairs"}).reset_index())
    thresholds = thr.rename("threshold").reset_index()
    return cal, thresholds, base


def evidence(con, ps):
    c = pd.read_sql("SELECT player_id, source, target, season_src, season_tgt, "
                    "class_source FROM entry_cohort", con)
    c["player_id"] = sp.normalize_player_id(c.player_id)
    k = ps.set_index(["player_id", "comp", "season"])
    c["src_share"] = k.start_share.reindex(list(zip(c.player_id, c.source, c.season_src))).values
    c["became"] = k.starter.reindex(list(zip(c.player_id, c.target, c.season_tgt))).values
    c = c.dropna(subset=["src_share", "became"])
    c["bench"] = c.src_share < 0.5
    rows = []
    for tgt, g in c.groupby("target"):
        b, s = g[g.bench], g[~g.bench]
        from scipy.stats import fisher_exact
        p_val = float(fisher_exact([[int(b.became.sum()), len(b) - int(b.became.sum())],
                                    [int(s.became.sum()), len(s) - int(s.became.sum())]])[1])
        rows.append(dict(target=tgt, n_bench=len(b), starter_bench=float(b.became.mean()),
                         p_value=p_val,
                         rating_bench=float(b.class_source.median()), n_start=len(s),
                         starter_start=float(s.became.mean()),
                         rating_start=float(s.class_source.median()),
                         landing_from=int(g.season_tgt.min()),
                         landing_to=int(g.season_tgt.max())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    ps = season_starts(con)
    cal, thresholds, base = calibration(con)
    ev = evidence(con, ps)
    con.close()
    print(f"player_season_starts: {len(ps):,} rows; starter share "
          f"{ps.starter.mean():.2f}")
    print(cal.T.to_string())
    print(base.round(3).to_string(index=False))
    print(ev.round(2).to_string(index=False))
    if not a.write:
        return 0
    with runtime.guarded_write("decision_tables",
                               note="player_season_starts, decision_calibration, "
                                    "bench_evidence, top_third_thresholds, top_third_base"):
        w = sqlite3.connect(rb.DB)
        ps.to_sql("player_season_starts", w, if_exists="replace", index=False)
        w.execute("CREATE INDEX IF NOT EXISTS ix_pss ON player_season_starts"
                  "(player_id, comp, season)")
        cal.to_sql("decision_calibration", w, if_exists="replace", index=False)
        base.to_sql("top_third_base", w, if_exists="replace", index=False)
        thresholds.to_sql("top_third_thresholds", w, if_exists="replace", index=False)
        ev.to_sql("bench_evidence", w, if_exists="replace", index=False)
        w.commit()
        w.close()
    print("wrote player_season_starts, decision_calibration, top_third_thresholds, "
          "bench_evidence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
