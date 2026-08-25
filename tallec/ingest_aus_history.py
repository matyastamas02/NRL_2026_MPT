# -*- coding: utf-8 -*-
"""Load the Australian history file into tallec.db — NRL 2020-2025, NSW Cup and
Queensland Cup 2021-2025 (85,675 player-match rows, 340 columns).

APPENDS to the existing tables; it does not rebuild them, so the NRL 2026 and
SL 2021-2026 rows already loaded by ingest_full_season.py survive. See AUS_DATA.md
for the profile of the source file and the reasoning behind each step below.

What it does, in order:
  1. Drops the 16 rows that break the player-match key (documented and sent to the
     data provider in TALLEC_Aus_Data_Queries.xlsx). Excluded, not corrected — the
     fix depends on an answer we do not have yet.
  2. Appends engine rows (player_match_stats) and canonical rows (player_match_raw).
  3. Unifies the player_id space. The Stats Perform Player ID is GLOBAL — 133 of the
     137 SL 2026 players who also appear in the Australian file carry an identical
     ID — but earlier loads keyed some rows by name-slug because their source files
     had no ID column. Slug-keyed rows are re-keyed to the permanent ID, guarded by
     date of birth wherever both sides have it. The mapping is written to
     player_id_map so every change is auditable.
  4. Backfills position. Position in this feed is a per-match jersey assignment and
     is present on 9 of the 16 sheets; because the ID is global it backfills to the
     other 7, and onward to NRL 2026 and to the SL players who have played in
     Australia. position_source records which tier each value came from:
       match          - the row's own per-match Position value
       career_aus     - player's most common starting position in the Aus file
       career_metadata- pre-existing value from Mike's NSW-Cup Metadata.csv
       unknown        - no position source; the engine treats these as Bench
  5. Rebuilds the players registry and the competitions table.

Run regenerate_full.py afterwards to recompute ratings and contribution on the
widened data.
"""
import os
import sqlite3

import numpy as np
import pandas as pd

import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.dirname(BASE)
DB = os.path.join(BASE, "tallec.db")
SRC = os.path.join(DL, "TALLEC all Aus Data.xlsx")

SHEET_COMP = {"NRL": "NRL", "NSW": "NSW", "QLD": "QLD"}


def parse_sheet_name(s):
    comp = SHEET_COMP["".join(ch for ch in s if ch.isalpha())]
    season = 2000 + int("".join(ch for ch in s if ch.isdigit()))
    return comp, season


# ── 1. read every sheet, drop key-breaking rows ────────────────────────────────
xl = pd.ExcelFile(SRC)
frames, dropped = [], []
for sheet in xl.sheet_names:
    comp, season = parse_sheet_name(sheet)
    df = xl.parse(sheet)
    df["Competition"], df["Season"], df["_sheet"] = comp, season, sheet
    df["player_id"] = sp.normalize_player_id(df["Player ID"])
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
    if "Position" not in df.columns:
        df["Position"] = None
    dup = df.duplicated(subset=["player_id", "Round"], keep=False)
    if dup.any():
        dropped.append(df[dup][["_sheet", "player_id", "Full Name", "Team", "Round"]])
    frames.append(df[~dup])
    print(f"  {sheet}: {len(df)} rows, dropped {int(dup.sum())}", flush=True)

raw = pd.concat(frames, ignore_index=True, sort=False)
dropped = pd.concat(dropped, ignore_index=True) if dropped else pd.DataFrame()
print(f"\nloaded {len(raw):,} rows | excluded {len(dropped)} key-breaking rows")

# ── 2. position: per-match value, plus a career map from the ID ────────────────
known = raw[raw["Position"].notna()]


career_pos = known.groupby("player_id")["Position"].agg(sp.primary_position)
print(f"career position map: {len(career_pos):,} player IDs "
      f"(from {len(known):,} rows that carry a Position)")

raw["position"] = raw["Position"]
raw["position_source"] = np.where(raw["Position"].notna(), "match", None)
fill = raw["position"].isna() & raw["player_id"].map(career_pos).notna()
raw.loc[fill, "position"] = raw.loc[fill, "player_id"].map(career_pos)
raw.loc[fill, "position_source"] = "career_aus"
raw["position"] = raw["position"].fillna("Unknown")
raw["position_source"] = raw["position_source"].fillna("unknown")
print("position_source:", raw["position_source"].value_counts().to_dict())

# ── 3. engine + raw frames ─────────────────────────────────────────────────────
raw["player"] = raw["Full Name"].map(sp.clean_name)
raw["team"] = raw["Team"]
raw["opposition"] = raw.get("Opposition")
raw["round"] = raw["Round"]
raw["minutes"] = pd.to_numeric(raw["Minutes"], errors="coerce").fillna(0)
for src, dst in sp.ENGINE_MAP.items():
    raw[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else np.nan
raw["fantasy"] = sp.fantasy_proxy(raw)

con = sqlite3.connect(DB)
eng_cols = [r[1] for r in con.execute("PRAGMA table_info(player_match_stats)")]
raw_cols = [r[1] for r in con.execute("PRAGMA table_info(player_match_raw)")]

# widen the existing tables for the columns this file adds
for col in ["position_source"]:
    if col not in eng_cols:
        con.execute(f'ALTER TABLE player_match_stats ADD COLUMN "{col}" TEXT')
        eng_cols.append(col)
        print(f"player_match_stats: added column {col}")
for col in ["Position", "Age", "Games Played"]:
    if col not in raw_cols:
        con.execute(f'ALTER TABLE player_match_raw ADD COLUMN "{col}"')
        raw_cols.append(col)
        print(f"player_match_raw: added column {col}")
con.commit()

eng = raw.rename(columns={"Competition": "competition", "Season": "season"})
eng = eng.reindex(columns=eng_cols)
eng.to_sql("player_match_stats", con, if_exists="append", index=False)
print(f"\nplayer_match_stats: appended {len(eng):,} rows")

canon = raw.reindex(columns=raw_cols)
CHUNK = 5000
for i in range(0, len(canon), CHUNK):
    canon.iloc[i:i + CHUNK].to_sql("player_match_raw", con, if_exists="append", index=False)
print(f"player_match_raw: appended {len(canon):,} rows")

# ── 4. unify the player_id space ───────────────────────────────────────────────
# name -> permanent ID, taken from this file (plus DoB for the guard)
aus_id = (raw.assign(slug=raw["player"].map(sp.slugify),
                     dob=pd.to_datetime(raw["Date of Birth"], errors="coerce"))
          .groupby("slug").agg(pid=("player_id", "first"),
                               n_pid=("player_id", "nunique"),
                               dob=("dob", "first")).reset_index())
aus_id = aus_id[aus_id.n_pid == 1]          # ambiguous names are never mapped

old = pd.read_sql(
    'SELECT DISTINCT player_id, player, competition, season FROM player_match_stats '
    "WHERE player_id NOT GLOB '[0-9]*'", con)
old_dob = pd.read_sql(
    'SELECT DISTINCT player_id, "Date of Birth" AS dob FROM player_match_raw '
    "WHERE player_id NOT GLOB '[0-9]*' AND \"Date of Birth\" IS NOT NULL", con)
old_dob["dob"] = pd.to_datetime(old_dob["dob"], errors="coerce")
old_dob = old_dob.groupby("player_id")["dob"].first()

cand = old.assign(slug=old["player"].map(sp.clean_name).map(sp.slugify)).merge(
    aus_id, on="slug", how="inner")
cand["old_dob"] = cand["player_id"].map(old_dob)
both_dob = cand.old_dob.notna() & cand.dob.notna()
cand["match_basis"] = np.where(both_dob, "name+dob", "name_only")
# reject a name match whose dates of birth disagree
bad = both_dob & (cand.old_dob.dt.date != cand.dob.dt.date)
rejected = cand[bad]
cand = cand[~bad]
# one permanent ID must not absorb two different old keys
cand = cand[~cand.duplicated(subset=["pid"], keep=False)]

print(f"\nid unification: {len(cand)} slug keys -> permanent ID "
      f"({(cand.match_basis == 'name+dob').sum()} confirmed by date of birth, "
      f"{(cand.match_basis == 'name_only').sum()} by name only); "
      f"{len(rejected)} rejected on a date-of-birth mismatch")
print(cand.groupby(["competition", "match_basis"]).size().to_string())

audit = cand[["player_id", "pid", "player", "competition", "season", "match_basis"]].rename(
    columns={"player_id": "old_player_id", "pid": "new_player_id"})
audit.to_sql("player_id_map", con, if_exists="replace", index=False)
for _, r in cand.iterrows():
    con.execute("UPDATE player_match_stats SET player_id=? WHERE player_id=?",
                (r["pid"], r["player_id"]))
    con.execute('UPDATE player_match_raw SET player_id=?, "Player ID"=? WHERE player_id=?',
                (r["pid"], r["pid"], r["player_id"]))
con.commit()
print(f"player_id_map: {len(audit)} rows written for audit")

# ── 5. backfill position onto the previously loaded rows ───────────────────────
cp = career_pos.reset_index().rename(columns={"Position": "pos"})
before = pd.read_sql("SELECT competition, season, position, count(*) n FROM player_match_stats "
                     "WHERE competition IN ('NRL','SL') AND season=2026 GROUP BY 1,2,3", con)
con.execute("UPDATE player_match_stats SET position_source='career_metadata' "
            "WHERE position_source IS NULL AND position IS NOT NULL AND position<>'Unknown'")
con.execute("UPDATE player_match_stats SET position_source='unknown' WHERE position_source IS NULL")
for _, r in cp.iterrows():
    con.execute("UPDATE player_match_stats SET position=?, position_source='career_aus' "
                "WHERE player_id=? AND (position IS NULL OR position='Unknown')",
                (r["pos"], r["player_id"]))
con.commit()

# ── 6. registry + competitions ────────────────────────────────────────────────
pms = pd.read_sql("SELECT player_id, player, team, competition, season, minutes, position "
                  "FROM player_match_stats", con)
# date of birth belongs on this small registry, not on the 342-column raw table —
# reading it from there per player is slow enough to hang the app
dob_reg = pd.read_sql('SELECT player_id, min("Date of Birth") dob FROM player_match_raw '
                      'WHERE "Date of Birth" IS NOT NULL GROUP BY player_id', con)
players = (pms.groupby("player_id").agg(
    name=("player", "first"),
    comps=("competition", lambda x: "; ".join(sorted(set(x)))),
    teams=("team", lambda x: "; ".join(sorted(set(map(str, x))))),
    seasons=("season", lambda x: f"{int(min(x))}-{int(max(x))}"),
    matches=("player", "count"), total_minutes=("minutes", "sum"),
    positions=("position", lambda x: x.mode().iloc[0] if len(x.mode()) else "Unknown"))
    .reset_index()
    .merge(dob_reg, on="player_id", how="left"))
players.to_sql("players", con, if_exists="replace", index=False)
for ddl in ["CREATE INDEX IF NOT EXISTS ix_pms_comp_season ON player_match_stats(competition, season)",
            "CREATE INDEX IF NOT EXISTS ix_pms_player ON player_match_stats(player_id)",
            "CREATE INDEX IF NOT EXISTS ix_pmr_player ON player_match_raw(player_id)",
            "CREATE INDEX IF NOT EXISTS ix_ratings_comp ON player_ratings(competition, season)",
            "CREATE INDEX IF NOT EXISTS ix_players_id ON players(player_id)"]:
    con.execute(ddl)
pd.DataFrame(sp.COMPETITIONS, columns=["comp_code", "comp_name", "country"]).to_sql(
    "competitions", con, if_exists="replace", index=False)

if len(dropped):
    dropped.to_sql("excluded_rows", con, if_exists="replace", index=False)

con.commit()

# ── report ────────────────────────────────────────────────────────────────────
print("\n== player_match_stats coverage ==")
print(pd.read_sql("SELECT competition, season, count(*) rows, count(DISTINCT player_id) players, "
                  "min(round) r_min, max(round) r_max FROM player_match_stats "
                  "GROUP BY 1,2 ORDER BY 1,2", con).to_string(index=False))
print("\n== position source ==")
print(pd.read_sql("SELECT competition, position_source, count(*) n FROM player_match_stats "
                  "GROUP BY 1,2 ORDER BY 1,3 DESC", con).to_string(index=False))
print("\n== position known, by competition ==")
print(pd.read_sql("SELECT competition, round(100.0*sum(CASE WHEN position<>'Unknown' THEN 1 END)"
                  "/count(*),1) pct_known, count(*) rows FROM player_match_stats "
                  "GROUP BY 1", con).to_string(index=False))
print(f"\nplayers registry: {len(players):,} | db size "
      f"{os.path.getsize(DB)/1048576:.1f} MB")
con.close()
