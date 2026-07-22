# -*- coding: utf-8 -*-
"""Regenerate ratings + contribution on the full-season data, PER COMPETITION.

NRL and SL are rated in separate pools (a Super League winger is judged against
Super League, not against the NRL). Ratings/contribution are computed for the
current season (2026); earlier SL seasons stay in the DB for history/trends.

Position is not in the Stats Perform feed, so standardization is currently
competition-relative, not position-relative — flagged until a position source
arrives.
"""
import sqlite3, os
import pandas as pd, numpy as np
import player_rating_engine as pre

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
SEASON = 2026


def load_comp_season(con, comp, season):
    cols = ["player_id", "player", "season", "round", "team", "position", "minutes",
            "all_run_metres", "p_c_m", "tackle_breaks", "line_breaks", "tackles",
            "offloads", "try_assists", "tries", "errors"]
    q = (f"SELECT {', '.join(cols)} FROM player_match_stats "
         f"WHERE competition=? AND season=?")
    return pd.read_sql(q, con, params=(comp, season))


con = sqlite3.connect(DB)

# ── Ratings per competition ────────────────────────────────────────────────
rating_frames = []
for comp in ["NRL", "SL"]:
    raw = load_comp_season(con, comp, SEASON)
    if raw.empty:
        continue
    eng = pre.PlayerRatingEngine(comp)
    snap = eng.compute_snapshot(raw)
    snap["competition"] = comp
    snap["season"] = SEASON
    snap["comp_code"] = comp
    snap["positional_benchmark"] = snap["class_score"]
    snap["competition_translation_factor"] = 0.0
    snap["updated_at"] = "2026-07-22"
    rating_frames.append(snap)
    print(f"{comp} {SEASON}: rated {len(snap)} players "
          f"(sigma^2={eng.sigma2:.3f}, tau^2={eng.tau2:.3f}, "
          f"reliability={eng.tau2/(eng.tau2+eng.sigma2):.2f})")

ratings = pd.concat(rating_frames, ignore_index=True)
cols = ["player_id", "season", "comp_code", "competition", "form_score", "form_z",
        "class_score", "class_z", "divergence", "positional_benchmark",
        "competition_translation_factor", "updated_at", "shrinkage_B",
        "n_games", "confidence"]
ratings[cols].to_sql("player_ratings", con, if_exists="replace", index=False)

# ── Contribution per competition (share of own team's output) ───────────────
ATT = ["all_run_metres", "p_c_m", "tackle_breaks", "line_breaks"]
DEF = ["tackles"]
PTS = ["tries", "try_assists"]
W = {"att": 0.45, "def": 0.35, "pts": 0.20}

contrib_frames = []
for comp in ["NRL", "SL"]:
    df = pd.read_sql(
        "SELECT player_id, player, team, season, round, minutes, "
        + ", ".join(ATT + DEF + PTS)
        + " FROM player_match_stats WHERE competition=? AND season=?",
        con, params=(comp, SEASON))
    if df.empty:
        continue
    for c in ATT + DEF + PTS:
        df[c] = df[c].fillna(0)
    tot = df.groupby(["season", "round", "team"])[ATT + DEF + PTS].transform("sum")
    for c in ATT + DEF + PTS:
        df[f"s_{c}"] = np.where(tot[c] > 0, df[c] / tot[c], 0.0)
    df["att"] = df[[f"s_{c}" for c in ATT]].mean(axis=1)
    df["dfe"] = df[[f"s_{c}" for c in DEF]].mean(axis=1)
    df["pts"] = df[[f"s_{c}" for c in PTS]].mean(axis=1)
    df["share"] = W["att"]*df["att"] + W["def"]*df["dfe"] + W["pts"]*df["pts"]
    df["contribution_rating"] = df["share"].rank(pct=True) * 100
    agg = (df.groupby("player_id").agg(
                name=("player", "first"), team=("team", "first"),
                matches=("round", "count"), minutes=("minutes", "sum"),
                attack_share=("att", "mean"), defence_share=("dfe", "mean"),
                contribution_rating=("contribution_rating", "mean"))
           .reset_index())
    agg["competition"] = comp
    contrib_frames.append(agg)
    print(f"{comp} contribution: {len(agg)} players")

contrib = pd.concat(contrib_frames, ignore_index=True).sort_values(
    "contribution_rating", ascending=False)
contrib.to_sql("player_contribution_rating", con, if_exists="replace", index=False)

# ── patch players table with a positions placeholder + competition ──────────
players = pd.read_sql("SELECT * FROM players", con)
if "positions" not in players.columns:
    players["positions"] = "Unknown"
    players.to_sql("players", con, if_exists="replace", index=False)

con.commit()
print(f"\nplayer_ratings: {len(ratings)} rows | player_contribution_rating: {len(contrib)} rows")
print("Top 5 SL contributors:")
print(contrib[contrib.competition == "SL"].head(5)[
    ["name", "team", "contribution_rating"]].round(1).to_string(index=False))
con.close()
