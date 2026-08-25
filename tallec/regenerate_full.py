# -*- coding: utf-8 -*-
"""Regenerate ratings + contribution, PER COMPETITION, over the full history.

THIS SCRIPT IS THE SINGLE WRITER of player_ratings, player_contribution and
player_contribution_rating. The contribution formula lives in gigot_contribution.py
(import-only) so the two files cannot drift apart.

Two design decisions worth knowing before reading the numbers.

**Each competition is its own pool.** A Super League forward is judged against Super
League, never against the NRL. Cross-competition comparison is the translation
model's job (`predict_translation.py`), not the rating's.

**Class spans every season the player has; Form is his last five matches.** Each
match is standardized against ITS OWN season's pool, so a strong 2023 is measured
against 2023 — then Class averages across them. That is why composites are built
season by season rather than in one pass over the whole history.

**One standardization mode per competition, for the whole history.** The mode is
chosen from the competition's overall position coverage, not per season. Otherwise a
player's 2026 matches could be position-relative while his 2024 matches were
competition-relative, and averaging them into one Class would be meaningless. Super
League is therefore competition-relative throughout (17% coverage overall, despite
2026 being complete), and the Australian competitions are position-relative.

Ratings are published for players active in each competition's most recent season —
that is who a recruiter is looking at — but built from everything they have played.
"""
import os
import sqlite3

import numpy as np
import pandas as pd

import time

import player_rating_engine as pre
import runtime
import gigot_contribution as gc

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
COMPS = ["NRL", "SL", "NSW", "QLD"]
COLS = ["player_id", "player", "season", "round", "team", "position", "minutes",
        "all_run_metres", "p_c_m", "tackle_breaks", "line_breaks", "tackles",
        "offloads", "try_assists", "tries", "errors"]


def load(con, comp, season=None):
    q = f"SELECT {', '.join(COLS)} FROM player_match_stats WHERE competition=?"
    p = [comp]
    if season is not None:
        q += " AND season=?"
        p.append(season)
    return pd.read_sql(q, con, params=p)


def competition_mode(df):
    """One mode for the whole history, from overall coverage."""
    known = df["position"].notna() & (df["position"] != "Unknown")
    cov = float(known.mean()) if len(df) else 0.0
    return (None if cov >= pre.MIN_POS_COVERAGE else "competition_relative"), cov


_t0 = time.time()
con = sqlite3.connect(DB)
rating_frames, contrib_frames, match_frames = [], [], []
_stats = {}

for comp in COMPS:
    hist = load(con, comp)
    if hist.empty:
        continue
    seasons = sorted(hist.season.dropna().unique())
    current = int(seasons[-1])
    force, cov = competition_mode(hist)

    # composites season by season, each against its own season's pool
    pm_parts = []
    for s in seasons:
        part = hist[hist.season == s]
        if len(part) < 100:
            continue
        eng_s = pre.PlayerRatingEngine(comp, force_mode=force)
        pm_parts.append(eng_s._composite(part))
    pm = pd.concat(pm_parts, ignore_index=True).sort_values(["player_id", "season", "round"])

    eng = pre.PlayerRatingEngine(comp, force_mode=force)
    eng.position_mode = ("competition_relative" if force else "position_relative")
    snap = eng.compute_snapshot(pm=pm)

    # publish the players active in the most recent season
    active = set(load(con, comp, current).player_id)
    snap = snap[snap.player_id.isin(active)].copy()
    snap["competition"] = comp
    snap["comp_code"] = comp
    snap["season"] = current
    snap["positional_benchmark"] = snap["class_score"]
    snap["competition_translation_factor"] = 0.0
    snap["updated_at"] = "2026-08-20"
    rating_frames.append(snap)
    _stats[comp] = {"players": int(len(snap)), "season": current,
                    "rows": int(len(pm)), "mode": eng.position_mode,
                    "coverage": round(cov, 3), "sigma2": round(eng.sigma2, 4),
                    "tau2": round(eng.tau2, 4), "median_games": float(snap.n_games.median())}
    print(f"{comp}: rated {len(snap)} players active in {current}, from "
          f"{len(pm):,} player-matches across {int(seasons[0])}-{current} "
          f"| {eng.position_mode} (coverage {cov*100:.0f}%) "
          f"| sigma^2={eng.sigma2:.3f} tau^2={eng.tau2:.3f} "
          f"reliability={eng.tau2/(eng.tau2+eng.sigma2):.2f} "
          f"| median games {snap.n_games.median():.0f}")

    # contribution: current season only — it is a share of this season's team output
    df = pd.read_sql(f"SELECT {', '.join(gc.INPUT_COLS)} FROM player_match_stats "
                     f"WHERE competition=? AND season=?", con, params=(comp, current))
    if not df.empty:
        per_match = gc.compute_contribution(df, comp)
        match_frames.append(per_match)
        contrib_frames.append(gc.aggregate_contribution(per_match, comp))

ratings = pd.concat(rating_frames, ignore_index=True)
cols = ["player_id", "season", "comp_code", "competition", "form_score", "form_z",
        "class_score", "class_z", "divergence", "positional_benchmark",
        "competition_translation_factor", "updated_at", "shrinkage_B",
        "n_games", "confidence", "rating_basis"]
ratings[cols].to_sql("player_ratings", con, if_exists="replace", index=False)
con.execute("CREATE INDEX IF NOT EXISTS ix_ratings_comp ON player_ratings(competition, season)")

per_match_all = pd.concat(match_frames, ignore_index=True)
per_match_all.to_sql("player_contribution", con, if_exists="replace", index=False)
contrib = pd.concat(contrib_frames, ignore_index=True).sort_values(
    "contribution_rating", ascending=False)
contrib.to_sql("player_contribution_rating", con, if_exists="replace", index=False)
con.commit()

print(f"\nplayer_ratings: {len(ratings)} rows | "
      f"player_contribution: {len(per_match_all)} rows | "
      f"player_contribution_rating: {len(contrib)} rows")
runtime.record_model_run("player_ratings+contribution", _stats, time.time() - _t0,
                         script="regenerate_full.py")
print(pd.read_sql("SELECT competition, season, count(*) players, "
                  "round(avg(n_games),1) avg_games, rating_basis "
                  "FROM player_ratings GROUP BY 1,2,5 ORDER BY 1", con).to_string(index=False))
con.close()
