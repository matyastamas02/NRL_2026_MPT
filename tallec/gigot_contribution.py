# -*- coding: utf-8 -*-
"""GIGOT v2 input #5: Contribution Rating.

Contribution Rating = player's share of his team's output in a match,
computed per stat group, then blended. This is REAL data (no mocks) -
team totals are the sum of teammates' stats in the same match.

Stat groups (per TALLEC scope: "player stats as % of team stats"):
  attack:  all_run_metres, p_c_m, tackle_breaks, line_breaks
  defence: tackles
  points:  points involvement (tries + try_assists)

Output tables:
  player_contribution        - one row per player per match (shares)
  player_contribution_rating - rolling blended rating per player (0-100,
                               where 100/17 * 17 = full team output; scaled
                               so that an average starter ~= 50)
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

DB = Path(__file__).parent / "tallec.db"
con = sqlite3.connect(DB)

ATTACK = ["all_run_metres", "p_c_m", "tackle_breaks", "line_breaks"]
DEFENCE = ["tackles"]
POINTS = ["tries", "try_assists"]
ALL_STATS = ATTACK + DEFENCE + POINTS

df = pd.read_sql(
    f"SELECT player_id, player, season, round, team, opposition, minutes, "
    f"{', '.join(ALL_STATS)} FROM player_match_stats", con)

# Gerard leaves zero-valued count stats blank -> treat NaN as 0 for shares
for c in ALL_STATS:
    df[c] = df[c].fillna(0)

# Team totals per match (sum over teammates)
team_tot = df.groupby(["season", "round", "team"])[ALL_STATS].transform("sum")

# Player share per stat (guard against zero team totals)
for c in ALL_STATS:
    df[f"share_{c}"] = np.where(team_tot[c] > 0, df[c] / team_tot[c], 0.0)

# Blend into group shares
df["attack_share"] = df[[f"share_{c}" for c in ATTACK]].mean(axis=1)
df["defence_share"] = df[[f"share_{c}" for c in DEFENCE]].mean(axis=1)
df["points_share"] = df[[f"share_{c}" for c in POINTS]].mean(axis=1)

# Overall contribution share: attack 45%, defence 35%, points 20%
W_ATT, W_DEF, W_PTS = 0.45, 0.35, 0.20
df["contribution_share"] = (W_ATT * df["attack_share"]
                            + W_DEF * df["defence_share"]
                            + W_PTS * df["points_share"])

# Rating = percentile rank of the blended share across all player-matches
# (0-100, median contributor = 50; matches the TALLEC benchmark convention).
# Raw x17x50 scaling clips at 100 for try-scorers because points shares are
# spiky - percentiles keep the scale honest.
df["contribution_rating"] = df["contribution_share"].rank(pct=True) * 100

per_match = df[["player_id", "player", "season", "round", "team", "opposition",
                "minutes", "attack_share", "defence_share", "points_share",
                "contribution_share", "contribution_rating"]]
per_match.to_sql("player_contribution", con, if_exists="replace", index=False)

# Rolling rating per player (mean over available matches; later: 5-match window)
rating = (per_match.groupby("player_id")
          .agg(name=("player", "first"), team=("team", "first"),
               matches=("round", "count"), minutes=("minutes", "sum"),
               attack_share=("attack_share", "mean"),
               defence_share=("defence_share", "mean"),
               contribution_rating=("contribution_rating", "mean"))
          .reset_index()
          .sort_values("contribution_rating", ascending=False))
rating.to_sql("player_contribution_rating", con, if_exists="replace", index=False)
con.commit()

print(f"player_contribution: {len(per_match)} rows | ratings: {len(rating)} players")
print("\n-- Top 10 contributors (share of own team's output) --")
print(rating.head(10)[["name", "team", "matches", "contribution_rating"]]
      .to_string(index=False))


def expected_contribution(team, available_players=None):
    """GIGOT team-list hook: expected total contribution of a lineup.

    available_players: list of player names, or None for full known squad.
    Returns the summed contribution ratings and the delta vs full squad -
    this is the 'player availability' signal that feeds GIGOT v2.
    """
    squad = rating[rating["team"] == team]
    full = squad["contribution_rating"].sum()
    if available_players is None:
        return {"team": team, "expected": full, "delta_vs_full": 0.0}
    avail = squad[squad["name"].isin(available_players)]
    exp = avail["contribution_rating"].sum()
    return {"team": team, "expected": exp, "delta_vs_full": exp - full,
            "missing": sorted(set(squad["name"]) - set(available_players))}


if __name__ == "__main__":
    # Demo: what does losing the top contributor cost Manly?
    team = "Manly Sea Eagles"
    squad = rating[rating["team"] == team].sort_values(
        "contribution_rating", ascending=False)
    top = squad.iloc[0]["name"]
    without_top = [n for n in squad["name"] if n != top]
    full = expected_contribution(team)
    reduced = expected_contribution(team, without_top)
    print(f"\n-- GIGOT demo: {team} team-list scenario --")
    print(f"  Full squad expected contribution: {full['expected']:.0f}")
    print(f"  Without {top}: {reduced['expected']:.0f} "
          f"(delta {reduced['delta_vs_full']:+.0f})")
    print(f"  -> this delta is the Player Availability input for GIGOT v2")

con.close()
