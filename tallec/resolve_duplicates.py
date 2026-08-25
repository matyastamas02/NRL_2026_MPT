# -*- coding: utf-8 -*-
"""Resolve the 16 excluded Australian rows, per the data provider's ruling.

Those rows broke the player-match key and were held out of the load pending an
answer. The answer (Mike, 2026-08-25):

  * "There are two players called Blake Moore. We need to generate a second player
    ID for the Falcons one, as the Capras one is a much more prominent player."
  * "Same deal for James Walsh. The Redcliffe one is less important so we can create
    a random number for him."
  * "I can't find DOB for Blake #2 and James #2 so just leave [it]."
  * "I have added Ben Talty as the correct player in one of the duplicates — I think
    this is a data entry error from SP in their teamsheet."

So two of the three identifiers were each covering two different people, and one
same-club duplicate pair was a mis-typed name on a team sheet.

The split covers EVERY row of the affected identifiers, not only the 16 excluded
ones: the database already held Falcons rows under Blake Moore's identifier and
Redcliffe rows under James Walsh's, and leaving those behind would keep the two
careers merged.

New identifiers are deliberately NOT numeric. A made-up number could one day collide
with a real Stats Perform ID; `syn-<name>-2` cannot, and it says what it is on sight.

Rua Ngatikaura's pair is left as one player. It is a cross-club repeat of a round
number — a mid-season move — which the loader now treats as legitimate rather than as
a duplicate, and the provider did not flag it.

Idempotent: re-running finds nothing to move.
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

# (identifier, club that keeps it) -> the other club's rows move to a new identifier
SPLITS = [
    {"pid": "23487", "name": "Blake Moore", "move_team": "Sunshine Coast Falcons",
     "new_pid": "syn-blake-moore-2"},
    {"pid": "58001", "name": "James Walsh", "move_team": "Redcliffe Dolphins",
     "new_pid": "syn-james-walsh-2"},
]
TALTY = "23574"      # the correct player in the same-club duplicate pair

con = sqlite3.connect(DB)

# ── 1. split the two identifiers that cover two people ───────────────────────
print("splitting identifiers that cover two players")
for s in SPLITS:
    n = con.execute("SELECT count(*) FROM player_match_stats WHERE player_id=? AND team=?",
                    (s["pid"], s["move_team"])).fetchone()[0]
    con.execute("UPDATE player_match_stats SET player_id=?, player=? WHERE player_id=? AND team=?",
                (s["new_pid"], s["name"], s["pid"], s["move_team"]))
    con.execute('UPDATE player_match_raw SET player_id=?, "Player ID"=? '
                "WHERE player_id=? AND Team=?",
                (s["new_pid"], s["new_pid"], s["pid"], s["move_team"]))
    print(f"  {s['name']}: {n} {s['move_team']} rows -> {s['new_pid']} "
          f"(the {s['pid']} identity keeps the other club)")

# ── 2. bring the 16 held-out rows in, with the ruling applied ────────────────
try:
    excl = pd.read_sql("SELECT * FROM excluded_rows", con)
except Exception:
    excl = pd.DataFrame()

if excl.empty:
    print("\nno held-out rows to restore")
else:
    sheets = sorted(excl["_sheet"].unique())
    eng_cols = [r[1] for r in con.execute("PRAGMA table_info(player_match_stats)")]
    raw_cols = [r[1] for r in con.execute("PRAGMA table_info(player_match_raw)")]
    xl = pd.ExcelFile(SRC)
    added, notes = 0, []
    for sheet in sheets:
        comp = {"NRL": "NRL", "NSW": "NSW", "QLD": "QLD"}["".join(
            ch for ch in sheet if ch.isalpha())]
        season = 2000 + int("".join(ch for ch in sheet if ch.isdigit()))
        want = excl[excl["_sheet"] == sheet]
        df = xl.parse(sheet)
        df["player_id"] = sp.normalize_player_id(df["Player ID"])
        df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
        keys = set(zip(want.player_id.astype(str), want["Round"].astype(float)))
        rows = df[[(p, r) in keys for p, r in
                   zip(df.player_id, df["Round"].astype(float))]].copy()
        if rows.empty:
            continue

        # apply the ruling row by row
        new_ids, new_names = [], []
        talty_used = False
        for _, r in rows.iterrows():
            pid, nm, team = r.player_id, sp.clean_name(r["Full Name"]), r["Team"]
            hit = next((s for s in SPLITS if s["pid"] == pid and s["move_team"] == team), None)
            same_club = ((rows.player_id == pid) & (rows.Team == team)
                         & (rows.Round == r["Round"])).sum() > 1
            if hit:
                new_ids.append(hit["new_pid"]); new_names.append(hit["name"])
            elif same_club and not talty_used:
                # the mis-typed team-sheet slot: one of the pair is Ben Talty
                new_ids.append(TALTY); new_names.append("Ben Talty")
                talty_used = True
                notes.append(f"{sheet} R{int(r['Round'])} {team}: one of two identical "
                             f"'{nm}' rows reassigned to Ben Talty ({TALTY})")
            else:
                new_ids.append(pid); new_names.append(nm)
        rows["player_id"] = new_ids
        rows["player"] = new_names

        # shape exactly as the loader does
        rows["Competition"], rows["Season"] = comp, season
        rows["competition"], rows["season"] = comp, season
        rows["team"] = rows["Team"]
        rows["opposition"] = rows.get("Opposition")
        rows["round"] = rows["Round"]
        rows["minutes"] = pd.to_numeric(rows["Minutes"], errors="coerce").fillna(0)
        if "Position" not in rows.columns:
            rows["Position"] = None
        rows["position"] = rows["Position"].fillna("Unknown")
        rows["position_source"] = np.where(rows["Position"].notna(), "match", "unknown")
        for src, dst in sp.ENGINE_MAP.items():
            rows[dst] = (pd.to_numeric(rows[src], errors="coerce")
                         if src in rows.columns else np.nan)
        rows["fantasy"] = sp.fantasy_proxy(rows)

        rows.reindex(columns=eng_cols).to_sql("player_match_stats", con,
                                              if_exists="append", index=False)
        rows.reindex(columns=raw_cols).to_sql("player_match_raw", con,
                                              if_exists="append", index=False)
        added += len(rows)
        print(f"\n  {sheet}: restored {len(rows)} rows")
    for n in notes:
        print(f"    note — {n}")

    excl["resolution"] = "restored 2026-08-25 per provider ruling"
    excl.to_sql("excluded_rows", con, if_exists="replace", index=False)
    print(f"\nrestored {added} of the 16 held-out rows")

# ── 3. registry ─────────────────────────────────────────────────────────────
pms = pd.read_sql("SELECT player_id, player, team, competition, season, minutes, "
                  "position FROM player_match_stats", con)
dob = pd.read_sql('SELECT player_id, min("Date of Birth") dob FROM player_match_raw '
                  'WHERE "Date of Birth" IS NOT NULL GROUP BY player_id', con)
players = (pms.groupby("player_id").agg(
    name=("player", "first"),
    comps=("competition", lambda x: "; ".join(sorted(set(x)))),
    teams=("team", lambda x: "; ".join(sorted(set(map(str, x))))),
    seasons=("season", lambda x: f"{int(min(x))}-{int(max(x))}"),
    matches=("player", "count"), total_minutes=("minutes", "sum"),
    positions=("position", lambda x: x.mode().iloc[0] if len(x.mode()) else "Unknown"))
    .reset_index().merge(dob, on="player_id", how="left"))
players.to_sql("players", con, if_exists="replace", index=False)
con.execute("CREATE INDEX IF NOT EXISTS ix_players_id ON players(player_id)")
con.commit()

bad = pd.read_sql("SELECT count(*) n FROM (SELECT player_id, competition, season, round, "
                  "team, count(*) k FROM player_match_stats GROUP BY 1,2,3,4,5 HAVING k>1)",
                  con)
print(f"\nplayers registry: {len(players):,} | rows: "
      f"{pd.read_sql('SELECT count(*) n FROM player_match_stats', con).n[0]:,}")
print(f"same-player-same-club-same-round duplicates remaining: {int(bad.n[0])}")
con.close()
print("\nrun regenerate_full.py next")
