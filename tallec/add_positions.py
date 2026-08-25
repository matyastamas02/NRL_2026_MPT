# -*- coding: utf-8 -*-
"""Join Mike's Metadata.csv (Position, Position Group, permanent Player ID, DoB)
onto player_match_stats, so ratings/benchmarks become position-relative.

Metadata is NRL/NSW-Cup-centric (SL name-match only ~25%), so NRL gets real
positions; unmatched players (much of SL) keep 'Unknown' and fall back to
competition-relative — flagged until SL position metadata arrives.

Primary position per player = most common STARTING position (Interchange is a
role, not a position, so it's only used when a player never starts).
"""
import sys as _sys

# This is a legacy entry point. regenerate_full.py is the only writer of
# player_ratings and player_contribution*, and ingest_aus_history.py /
# weekly_update.py the only writers of the match tables — see README, "which script
# writes what". Running this would overwrite a live table from a narrower or older
# code path, so it refuses unless the caller is explicit.
if __name__ == "__main__" and "--i-know-this-overwrites" not in _sys.argv:
    _sys.exit(__doc__.strip().splitlines()[0] + "\n\n"
              "REFUSED: this script overwrites a table that another script owns.\n"
              "Use regenerate_full.py (ratings) or weekly_update.py (match data).\n"
              "Pass --i-know-this-overwrites to run it anyway.")

import sqlite3, os, re
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.dirname(BASE)
DB = os.path.join(BASE, "tallec.db")

# metadata specific Position -> rating-engine position group key
POS_TO_GROUP = {
    "Full Back": "Fullback", "Winger": "Winger", "Centre": "Centre",
    "Five-Eighth": "Halves", "Half Back": "Halves", "Hooker": "Hooker",
    "Prop": "Prop", "Second Row": "Back Row", "Lock": "Back Row",
    "Interchange": "Bench",
}

def clean(s):
    return re.sub(r"\*", "", str(s)).strip()

m = pd.read_csv(os.path.join(DL, "Metadata.csv"))
m["nm"] = m["Full Name"].map(clean)

# primary position: mode of Position excluding Interchange (fall back to Interchange)
def primary_pos(s):
    starts = s[s != "Interchange"]
    pool = starts if len(starts) else s
    return pool.mode().iloc[0] if len(pool.mode()) else "Interchange"

prim = (m.groupby("nm")
        .agg(position=("Position", primary_pos),
             player_id_perm=("Player ID", "first"),
             dob=("Date of Birth", "first"))
        .reset_index())
prim["pos_group"] = prim["position"].map(POS_TO_GROUP).fillna("Bench")
print(f"metadata: {len(prim)} players with a primary position")
print(prim["position"].value_counts().to_string())

con = sqlite3.connect(DB)
pms = pd.read_sql("SELECT rowid, * FROM player_match_stats", con)
pms["nm"] = pms["player"].map(clean)
pos_map = dict(zip(prim["nm"], prim["position"]))
# The metadata is an NSW Cup (Australian feeder) database. Matching by bare name
# is only safe inside the Australian player universe -> apply to NRL rows only.
# SL (English) name-matches to NSW Cup are unreliable (same-name / different person)
# so SL stays 'Unknown' until a Super League position source arrives.
pms["position_new"] = pms.apply(
    lambda r: pos_map.get(r["nm"]) if r["competition"] == "NRL" else None, axis=1)

# coverage per competition
cov = pms.groupby("competition")["position_new"].apply(lambda s: s.notna().mean())
print("\nposition coverage by competition:")
print((cov*100).round(0).astype(int).astype(str).add("%").to_string())

# write back: set position where matched, else keep 'Unknown'
pms["position"] = pms["position_new"].fillna("Unknown")
pms = pms.drop(columns=["nm", "position_new", "rowid"])
pms.to_sql("player_match_stats", con, if_exists="replace", index=False)

# store primary-position lookup for the app / players registry
prim[["nm", "position", "pos_group", "player_id_perm", "dob"]].to_sql(
    "player_positions", con, if_exists="replace", index=False)

# patch players table with primary position where known
players = pd.read_sql("SELECT * FROM players", con)
players["nm"] = players["name"].map(clean)
players["positions"] = players["nm"].map(pos_map).fillna("Unknown")
players = players.drop(columns=["nm"])
players.to_sql("players", con, if_exists="replace", index=False)
con.commit(); con.close()
print("\nplayer_match_stats.position updated; player_positions + players patched.")
