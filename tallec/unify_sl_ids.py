# -*- coding: utf-8 -*-
"""Join Super League's own history to its 2026 identities.

SL 2021-2025 was loaded from files with no Player ID column, so those rows are keyed
by name-slug. SL 2026 carries the permanent Stats Perform ID. The consequence is that
a player does not meet his own past: the multi-season Class of an SL player was built
from his 2026 matches alone, a median of 11, while an NRL player had 58.

Both sides carry a full name and a date of birth, within one competition, so the link
is a name+date-of-birth match — strong enough to use, and much stronger than name
alone. Only pairs that are unambiguous on BOTH sides are taken: a name+date that maps
to two old keys, or two new ids, is left alone rather than guessed.

Idempotent: re-running finds nothing left to re-key. Every change is appended to
player_id_map with basis 'sl_name+dob', so the migration is auditable and reversible.

Run regenerate_full.py afterwards.
"""
import os
import re
import sqlite3
import unicodedata

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def parse_dob(s):
    iso = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")
    dmy = pd.to_datetime(s, errors="coerce", format="%d/%m/%Y")
    return iso.fillna(dmy)


con = sqlite3.connect(DB)
d = pd.read_sql(
    'SELECT s.player_id, r."Full Name" nm, r."Date of Birth" dob '
    'FROM player_match_stats s JOIN player_match_raw r '
    '  ON r.player_id = s.player_id AND r.Competition = s.competition '
    '  AND r.Season = s.season AND r."Round" = s.round '
    "WHERE s.competition = 'SL'", con)
d["key"] = d.nm.map(norm)
d["dt"] = parse_dob(d.dob).dt.date
d = d.dropna(subset=["dt"])
d["numeric"] = d.player_id.str.match(r"^\d+$", na=False)

old = d[~d.numeric].drop_duplicates(["player_id", "key", "dt"])
new = d[d.numeric].drop_duplicates(["player_id", "key", "dt"])
print(f"slug-keyed SL players: {old.player_id.nunique()} | "
      f"permanent-id SL players: {new.player_id.nunique()}")

pairs = old.merge(new, on=["key", "dt"], suffixes=("_old", "_new"))
before = len(pairs)
pairs = pairs[~pairs.duplicated("player_id_old", keep=False)
              & ~pairs.duplicated("player_id_new", keep=False)]
dropped = before - len(pairs)
print(f"unambiguous name+date-of-birth pairs: {len(pairs)}"
      + (f" ({dropped} ambiguous pairs left alone)" if dropped else ""))

if pairs.empty:
    print("nothing to do.")
else:
    n_rows = 0
    for _, r in pairs.iterrows():
        cur = con.execute("UPDATE player_match_stats SET player_id=? WHERE player_id=?",
                          (r.player_id_new, r.player_id_old))
        n_rows += cur.rowcount
        con.execute('UPDATE player_match_raw SET player_id=?, "Player ID"=? '
                    "WHERE player_id=?",
                    (r.player_id_new, r.player_id_new, r.player_id_old))
    # player_id_map was created by ingest_aus_history.py; match its column names
    # exactly or the append fails and rolls the whole re-key back
    existing = [r[1] for r in con.execute("PRAGMA table_info(player_id_map)")]
    audit = pairs[["player_id_old", "player_id_new", "key"]].rename(
        columns={"player_id_old": "old_player_id", "player_id_new": "new_player_id",
                 "key": "player"})
    audit["competition"] = "SL"
    audit["season"] = None
    audit["match_basis"] = "sl_name+dob"
    audit = audit.reindex(columns=existing) if existing else audit
    audit.to_sql("player_id_map", con, if_exists="append", index=False)
    con.commit()
    print(f"re-keyed {n_rows:,} rows | player_id_map appended")

    # rebuild the registry so names/histories collapse onto the merged ids
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
    print(f"players registry: {len(players):,}")

    hist = pd.read_sql(
        "SELECT count(DISTINCT season) seasons, count(*) n FROM player_match_stats "
        "WHERE competition='SL' AND player_id GLOB '[0-9]*'", con)
    print(f"SL rows now under a permanent id: {int(hist.n[0]):,} "
          f"across {int(hist.seasons[0])} seasons")
con.close()
