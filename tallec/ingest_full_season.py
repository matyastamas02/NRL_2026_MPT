# -*- coding: utf-8 -*-
"""Full-season ingest — Stats Perform NRL (2026) + Super League (2021-2026).

Replaces the R11-12 Gerard seed. Writes tallec.db with:
  player_match_stats  — one row per player per match. Keeps the CANONICAL Stats
                        Perform column names (so the metric dictionary lines up
                        1:1) AND a set of lowercase engine-compat columns the
                        existing rating engine / contribution / app already use.
  players             — registry (permanent Player ID where available, else slug)
  competitions        — NRL / SL
  metric_dictionary, rate_rules, overlap_rules — Mike's Rate spec, as config.

Data reality (flagged to the client):
  * No position column in either feed — positional benchmarks currently fall
    back to competition-relative until a position source is supplied.
  * SL has permanent Player ID only from 2026; earlier seasons keyed by name.
  * NRL 337 cols vs SL 269 cols share 269 — cross-league work uses that core.
"""
import sqlite3, glob, os, re, sys, unicodedata
import pandas as pd, numpy as np

import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.dirname(BASE)
DB = os.path.join(BASE, "tallec.db")
NRL_FILE = os.path.join(DL, "Player Level Stats NRL.xlsx")
SL_GLOB = os.path.join(DL, "sl21_players", "SL2*.csv")
DICT_FILE = os.path.join(DL, "BOSC_Full_Metric_Rate_Review_v2.xlsx")

# canonical Stats Perform field -> lowercase engine-compat key. Defined once in
# sp_schema so this script and ingest_aus_history.py cannot drift apart.
ENGINE_MAP = sp.ENGINE_MAP
slugify = sp.slugify

# ── guard: this script REBUILDS its tables with if_exists="replace", so it would
# delete any competition it does not load itself (the Australian history added by
# ingest_aus_history.py, 85k rows). Refuse to run in that state unless forced.
MANAGES = {"NRL", "SL"}
if os.path.exists(DB):
    _con = sqlite3.connect(DB)
    try:
        present = {r[0] for r in _con.execute(
            "SELECT DISTINCT competition FROM player_match_stats")}
    except sqlite3.OperationalError:
        present = set()
    _con.close()
    foreign = present - MANAGES
    if foreign and "--force" not in sys.argv:
        sys.exit(f"ABORT: {DB} also holds {sorted(foreign)}, which this script does not "
                 f"load and would delete.\nRun ingest_aus_history.py after this one to "
                 f"restore them, or pass --force if that is what you want.")

def load_one(df, comp, season):
    df = df.copy()
    df["Competition"] = comp
    df["Season"] = season
    # stable player_id: permanent Player ID if present else name-slug
    if "Player ID" in df.columns and df["Player ID"].notna().any():
        df["player_id"] = df["Player ID"].astype(str).where(
            df["Player ID"].notna(), df["Full Name"].map(slugify))
    else:
        df["player_id"] = df["Full Name"].map(slugify)
    # engine-compat aliases + meta
    df["player"] = df["Full Name"]
    df["team"] = df["Team"]
    df["opposition"] = df.get("Opposition")
    df["round"] = pd.to_numeric(df["Round"], errors="coerce")
    df["minutes"] = pd.to_numeric(df["Minutes"], errors="coerce").fillna(0)
    df["position"] = "Unknown"          # no position field in these two feeds
    df["position_source"] = "unknown"   # filled later from the Australian file
    for src, dst in ENGINE_MAP.items():
        df[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan
    # fantasy proxy (no official fantasy col): standard-ish attacking+defensive blend
    df["fantasy"] = (df.get("tries", 0).fillna(0)*4 + df.get("try_assists", 0).fillna(0)*2
                     + df.get("line_breaks", 0).fillna(0) + df.get("tackle_breaks", 0).fillna(0)
                     + df.get("all_run_metres", 0).fillna(0)/10
                     + df.get("tackles", 0).fillna(0)*0.5 - df.get("errors", 0).fillna(0))
    return df

frames = []
# NRL — single 2026 file
nrl = pd.ExcelFile(NRL_FILE).parse("NRL26")
frames.append(load_one(nrl, "NRL", 2026))
print(f"NRL 2026: {len(nrl)} rows, {nrl.shape[1]} cols")
# SL — six seasons, year from filename SL{YY} Players.csv
for f in sorted(glob.glob(SL_GLOB)):
    yy = re.search(r"SL(\d{2})", os.path.basename(f)).group(1)
    season = 2000 + int(yy)
    sldf = pd.read_csv(f)
    frames.append(load_one(sldf, "SL", season))
    print(f"SL {season}: {len(sldf)} rows, {sldf.shape[1]} cols")

# union of all columns; align
raw = pd.concat(frames, ignore_index=True, sort=False)
raw = raw.drop_duplicates(subset=["Competition", "Season", "round", "team", "player"])
print(f"\ncombined: {len(raw)} rows, {raw.shape[1]} cols")

con = sqlite3.connect(DB)

# ── engine table: only lowercase engine-compat columns (what the rating engine,
# gigot_contribution and bosc_app query). SQLite is case-insensitive on column
# names, so keep this strictly disjoint from the canonical Title-Case set.
ENGINE_COLS = (["player_id", "player", "team", "opposition", "Competition", "Season",
                "round", "minutes", "position", "position_source", "fantasy"]
               + list(ENGINE_MAP.values()))
eng = raw[ENGINE_COLS].rename(columns={"Competition": "competition", "Season": "season"})
eng.to_sql("player_match_stats", con, if_exists="replace", index=False)

# ── raw table: full canonical Stats Perform fields (for the metric dictionary /
# Rate section). Join key = player_id + Competition + Season + round.
_drop_from_canon = sp.LOWER_ALIASES
canon = [c for c in raw.columns if c not in _drop_from_canon]
raw[canon].to_sql("player_match_raw", con, if_exists="replace", index=False)
print(f"player_match_stats (engine): {len(eng)} rows, {eng.shape[1]} cols")
print(f"player_match_raw (canonical): {len(raw)} rows, {len(canon)} cols")

# players registry
players = (raw.groupby("player_id")
    .agg(name=("player", "first"),
         comps=("Competition", lambda x: "; ".join(sorted(set(x)))),
         teams=("team", lambda x: "; ".join(sorted(set(map(str, x))))),
         seasons=("Season", lambda x: f"{int(min(x))}-{int(max(x))}"),
         matches=("player", "count"), total_minutes=("minutes", "sum"))
    .reset_index())
players.to_sql("players", con, if_exists="replace", index=False)

pd.DataFrame({"comp_code": ["NRL", "SL"],
              "comp_name": ["National Rugby League", "Super League"],
              "country": ["Australia", "England"]}
             ).to_sql("competitions", con, if_exists="replace", index=False)

# metric dictionary + rules as config tables
xl = pd.ExcelFile(DICT_FILE)
xl.parse("Metric Dictionary").to_sql("metric_dictionary", con, if_exists="replace", index=False)
xl.parse("Rate Rules").to_sql("rate_rules", con, if_exists="replace", index=False)
xl.parse("Overlap Rules").to_sql("overlap_rules", con, if_exists="replace", index=False)
con.commit()

print(f"\nplayers: {len(players)} | "
      f"NRL players: {(players['comps']=='NRL').sum()} | "
      f"SL players: {players['comps'].str.contains('SL').sum()}")
print("\n-- per competition/season coverage --")
print(raw.groupby(["Competition","Season"]).size().to_string())
print("\n-- metric_dictionary loaded:", len(xl.parse("Metric Dictionary")), "rows --")
print("-- players with permanent Player ID (SL 2026):",
      raw[raw["player_id"].str.match(r"^\d", na=False)]["player_id"].nunique(), "--")
con.close()
