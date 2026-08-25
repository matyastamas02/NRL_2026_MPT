# -*- coding: utf-8 -*-
"""GIGOT v2 input #5: Contribution Rating — computation module.

Contribution Rating = player's share of his team's output in a match, computed
per stat group, then blended. This is REAL data (no mocks) — team totals are the
sum of teammates' stats in the same match.

IMPORT-ONLY: this module never writes to the database. `regenerate_full.py` is
the single writer of both contribution tables and calls the functions below once
per competition. (Until 2026-08-14 the formula was duplicated here and in
regenerate_full.py, and both scripts wrote `player_contribution_rating` — this
module pooled NRL+SL into one percentile ranking, regenerate_full.py kept them
separate, so whichever ran last silently decided the numbers.)

Stat groups (per TALLEC scope: "player stats as % of team stats"):
  attack:  all_run_metres, p_c_m, tackle_breaks, line_breaks
  defence: tackles
  points:  points involvement (tries + try_assists)

Tables written by regenerate_full.py from these functions:
  player_contribution        - one row per player per match (shares)
  player_contribution_rating - per player, mean of his per-match ratings (0-100,
                               median contributor = 50 within his competition)
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DB = BASE / "tallec.db"

# config.json is optional — fall back to the shipped defaults so the module never
# hard-fails if the file is absent (same pattern as player_rating_engine.py).
_DEFAULT_CONFIG = {
    "contribution_rating": {
        "attack_stats": ["all_run_metres", "p_c_m", "tackle_breaks", "line_breaks"],
        "defence_stats": ["tackles"],
        "points_stats": ["tries", "try_assists"],
        "attack_weight": 0.45,
        "defence_weight": 0.35,
        "points_weight": 0.20,
    }
}
try:
    _CFG = json.loads((BASE / "config.json").read_text())["contribution_rating"]
except (FileNotFoundError, ValueError, KeyError):
    _CFG = _DEFAULT_CONFIG["contribution_rating"]

ATTACK = _CFG["attack_stats"]
DEFENCE = _CFG["defence_stats"]
POINTS = _CFG["points_stats"]
ALL_STATS = ATTACK + DEFENCE + POINTS
W_ATT = _CFG["attack_weight"]
W_DEF = _CFG["defence_weight"]
W_PTS = _CFG["points_weight"]

# Columns regenerate_full.py must SELECT for compute_contribution()
INPUT_COLS = (["player_id", "player", "team", "opposition", "season", "round", "minutes"]
              + ALL_STATS)

PER_MATCH_COLS = ["player_id", "player", "competition", "season", "round", "team",
                  "opposition", "minutes", "attack_share", "defence_share",
                  "points_share", "contribution_share", "contribution_rating"]


def compute_contribution(df, competition):
    """Per-player-per-match contribution shares for ONE competition.

    The percentile scaling is relative to the frame passed in, so the caller
    controls the comparison pool — pass one competition at a time (a Super
    League forward is judged against Super League, not against the NRL).
    """
    df = df.copy()
    # zero-valued count stats arrive blank in the feeds -> treat NaN as 0 for shares
    for c in ALL_STATS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0)

    # Team totals per match (sum over teammates). The frame is already a single
    # competition, so (season, round, team) identifies one team-performance.
    team_tot = df.groupby(["season", "round", "team"])[ALL_STATS].transform("sum")

    # Player share per stat (guard against zero team totals)
    for c in ALL_STATS:
        df[f"share_{c}"] = np.where(team_tot[c] > 0, df[c] / team_tot[c], 0.0)

    df["attack_share"] = df[[f"share_{c}" for c in ATTACK]].mean(axis=1)
    df["defence_share"] = df[[f"share_{c}" for c in DEFENCE]].mean(axis=1)
    df["points_share"] = df[[f"share_{c}" for c in POINTS]].mean(axis=1)
    df["contribution_share"] = (W_ATT * df["attack_share"]
                                + W_DEF * df["defence_share"]
                                + W_PTS * df["points_share"])

    # Rating = percentile rank of the blended share across this competition's
    # player-matches (0-100, median contributor = 50; the TALLEC benchmark
    # convention). Raw x17x50 scaling clips at 100 for try-scorers because points
    # shares are spiky — percentiles keep the scale honest.
    df["contribution_rating"] = df["contribution_share"].rank(pct=True) * 100
    df["competition"] = competition
    if "opposition" not in df.columns:
        df["opposition"] = None
    return df[PER_MATCH_COLS]


def aggregate_contribution(per_match, competition):
    """Per-player rating: mean of his per-match ratings (later: 5-match window)."""
    agg = (per_match.groupby("player_id")
           .agg(name=("player", "first"), team=("team", "first"),
                matches=("round", "count"), minutes=("minutes", "sum"),
                attack_share=("attack_share", "mean"),
                defence_share=("defence_share", "mean"),
                points_share=("points_share", "mean"),
                contribution_rating=("contribution_rating", "mean"))
           .reset_index())
    agg["competition"] = competition
    return agg.sort_values("contribution_rating", ascending=False)


def load_ratings(con, competition):
    """Read the stored per-player contribution ratings for one competition."""
    return pd.read_sql(
        "SELECT * FROM player_contribution_rating WHERE competition = ? "
        "ORDER BY contribution_rating DESC", con, params=(competition,))


def expected_contribution(con, team, competition, available_players=None):
    """GIGOT team-list hook: expected total contribution of a lineup.

    available_players: list of player names, or None for the full known squad.
    Returns the summed contribution ratings and the delta vs the full squad —
    this is the 'player availability' signal that feeds GIGOT v2.
    """
    ratings = load_ratings(con, competition)
    squad = ratings[ratings["team"] == team]
    full = squad["contribution_rating"].sum()
    if available_players is None:
        return {"team": team, "competition": competition,
                "expected": full, "delta_vs_full": 0.0}
    avail = squad[squad["name"].isin(available_players)]
    exp = avail["contribution_rating"].sum()
    return {"team": team, "competition": competition, "expected": exp,
            "delta_vs_full": exp - full,
            "missing": sorted(set(squad["name"]) - set(available_players))}


if __name__ == "__main__":
    # READ-ONLY report. Regeneration lives in regenerate_full.py — running this
    # file cannot overwrite the tables any more.
    con = sqlite3.connect(DB)
    print("gigot_contribution.py is import-only; this is a read-only report.")
    print("To (re)compute the contribution tables, run: python regenerate_full.py\n")
    for comp in ["NRL", "SL"]:
        try:
            rating = load_ratings(con, comp)
        except Exception as e:
            print(f"{comp}: no contribution data ({e}) — run regenerate_full.py")
            continue
        if rating.empty:
            print(f"{comp}: no contribution data — run regenerate_full.py")
            continue
        print(f"-- {comp}: {len(rating)} players | top 5 contributors --")
        print(rating.head(5)[["name", "team", "matches", "contribution_rating"]]
              .round(1).to_string(index=False))
        # team-list demo: what does losing the top contributor cost his club?
        top = rating.iloc[0]
        squad = rating[rating["team"] == top["team"]]
        without_top = [n for n in squad["name"] if n != top["name"]]
        reduced = expected_contribution(con, top["team"], comp, without_top)
        full = expected_contribution(con, top["team"], comp)
        print(f"   team-list scenario — {top['team']} full squad: "
              f"{full['expected']:.0f}; without {top['name']}: "
              f"{reduced['expected']:.0f} ({reduced['delta_vs_full']:+.0f})\n")
    con.close()
