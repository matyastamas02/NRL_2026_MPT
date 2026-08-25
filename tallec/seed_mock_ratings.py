# -*- coding: utf-8 -*-
"""Seed player_ratings table with mock data for BOSC UI testing."""
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

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent / "tallec.db"
con = sqlite3.connect(DB_PATH)

# Get all players
players = pd.read_sql("SELECT player_id FROM players", con)
print(f"Seeding mock ratings for {len(players)} players...")

# Generate mock ratings (realistic: form more volatile, class more stable)
np.random.seed(42)
ratings = []
for _, row in players.iterrows():
    player_id = row["player_id"]
    # Form: mean 62, std 12 (range ~25-99)
    form = np.clip(np.random.normal(62, 12), 0, 100)
    form_z = (form - 62) / 12

    # Class: mean 60, std 8 (range ~30-90, more stable)
    class_ = np.clip(np.random.normal(60, 8), 0, 100)
    class_z = (class_ - 60) / 8

    # Benchmark (positional): mean 50, std 10
    bench = np.clip(np.random.normal(50, 10), 0, 100)

    # Divergence: how much recent differs from long-term
    divergence = form_z - class_z

    ratings.append({
        "player_id": player_id,
        "season": 2026,
        "round": 12,
        "comp_code": "NRL",
        "form_score": form,
        "form_z": form_z,
        "class_score": class_,
        "class_z": class_z,
        "divergence": divergence,
        "positional_benchmark": bench,
        "competition_translation_factor": 0.0,
        "updated_at": "2026-07-11",
    })

ratings_df = pd.DataFrame(ratings)
ratings_df.to_sql("player_ratings", con, if_exists="replace", index=False)

print(f"OK: Seeded {len(ratings_df)} mock ratings")
print(f"  Form: {ratings_df['form_score'].describe()}")
print(f"  Class: {ratings_df['class_score'].describe()}")
print(f"  Benchmark: {ratings_df['positional_benchmark'].describe()}")

con.close()
