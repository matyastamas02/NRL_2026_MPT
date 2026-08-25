# -*- coding: utf-8 -*-
"""Weekly Stats Perform import — the in-season path.

The bulk loaders (ingest_full_season.py, ingest_aus_history.py) rebuild or append
whole seasons. Neither is safe to run every Friday: the first one replaces its
tables outright, which now means discarding 122k rows to add one round, and both
re-derive things a weekly file cannot supply on its own — the permanent player-ID
mapping and the position backfill.

This script does one round (or a handful) at a time and is safe to re-run:

  * IDEMPOTENT. Rows are keyed on competition + season + round. Whatever is in the
    file for those keys replaces whatever is in the database for those keys, so
    importing the same file twice leaves the same 4,476 rows, not 8,952.
  * PRESERVES the ID space. Weekly feeds vary: the Super League CSV carries a
    permanent Player ID, the NRL workbook does not. Rows without one are resolved
    through the same name -> permanent ID map the history load built, so a player
    does not fork into two identities mid-season.
  * PRESERVES position. The current-season feeds have no position column at all, so
    position comes from the player's other seasons via his permanent ID, recorded as
    position_source='career_aus'. A file that does carry Position wins over that.
  * VALIDATES BEFORE WRITING. The scoring arithmetic and squad sizes are checked
    first — the same checks that caught a Super League feed undercounting points
    badly enough to flip a result. Failures block the write unless --force.

Usage
    python weekly_update.py --file "SL26 Players.csv" --competition SL --season 2026
    python weekly_update.py --file "Player Level Stats NRL.xlsx" --sheet NRL26 \\
        --competition NRL --season 2026 --rounds 21
    python weekly_update.py ... --dry-run        # validate and report, write nothing
    python weekly_update.py ... --no-ratings     # skip the rating rebuild

After writing it re-runs regenerate_full.py so ratings, contribution and the app all
reflect the new round. Pass --no-ratings to batch several files and rebuild once.
"""
import argparse
import os
import subprocess
import sqlite3
import sys

import numpy as np
import pandas as pd

import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")


# ── loading ───────────────────────────────────────────────────────────────────
def read_source(path, sheet=None):
    if not os.path.exists(path):
        alt = os.path.join(os.path.dirname(BASE), path)
        if os.path.exists(alt):
            path = alt
        else:
            sys.exit(f"ABORT: file not found: {path}")
    if path.lower().endswith((".xlsx", ".xlsm")):
        xl = pd.ExcelFile(path)
        if sheet is None:
            if len(xl.sheet_names) == 1:
                sheet = xl.sheet_names[0]
            else:
                sys.exit(f"ABORT: {os.path.basename(path)} has sheets "
                         f"{xl.sheet_names} — name one with --sheet")
        return xl.parse(sheet), f"{os.path.basename(path)}::{sheet}"
    return pd.read_csv(path), os.path.basename(path)


def resolve_player_ids(df, con):
    """Permanent Player ID where the file has one, else the name -> ID map."""
    have_id = "Player ID" in df.columns and df["Player ID"].notna().any()
    names = df["Full Name"].map(sp.clean_name)
    slug = names.map(sp.slugify)
    reg = pd.read_sql("SELECT player_id, name FROM players", con)
    reg["slug"] = reg.name.map(sp.clean_name).map(sp.slugify)
    numeric = reg[reg.player_id.str.match(r"^\d+$", na=False)]
    dupes = set(numeric.slug[numeric.slug.duplicated()])
    name_map = dict(zip(numeric[~numeric.slug.isin(dupes)].slug,
                        numeric[~numeric.slug.isin(dupes)].player_id))
    if have_id:
        pid = sp.normalize_player_id(df["Player ID"])
        src = pd.Series("file", index=df.index)
        missing = df["Player ID"].isna()
        pid[missing] = slug[missing].map(name_map).fillna(slug[missing])
        src[missing] = np.where(slug[missing].isin(name_map), "name_map", "new_slug")
    else:
        pid = slug.map(name_map)
        src = pd.Series(np.where(pid.notna(), "name_map", "new_slug"), index=df.index)
        pid = pid.fillna(slug)
    return pid, src, names


def backfill_position(df, con, pid):
    """Position from the file if present, else the player's career position."""
    # Built from match-sheet rows only, with sp.primary_position, so the weekly path
    # assigns exactly what the history load would have. Deriving it from every row
    # instead — including positions that were themselves backfilled — moved five
    # bench forwards from their real position to "Interchange" on the first test.
    counts = pd.read_sql(
        "SELECT player_id, position, count(*) n FROM player_match_stats "
        "WHERE position_source='match' AND position<>'Unknown' GROUP BY 1, 2", con)
    career = (counts.loc[counts.index.repeat(counts.n)]
              .groupby("player_id")["position"].agg(sp.primary_position)
              if len(counts) else pd.Series(dtype=object))
    if "Position" in df.columns and df["Position"].notna().any():
        pos = df["Position"]
        src = pd.Series(np.where(pos.notna(), "match", None), index=df.index)
        fill = pos.isna()
        pos = pos.where(~fill, pid.map(career))
        src = src.where(~(fill & pid.map(career).notna()), "career_aus")
    else:
        pos = pid.map(career)
        src = pd.Series(np.where(pos.notna(), "career_aus", None), index=df.index)
    return pos.fillna("Unknown"), src.fillna("unknown")


# ── validation ────────────────────────────────────────────────────────────────
def validate(df, label):
    """The checks that have caught real feed errors before. Returns a list of
    (severity, message); 'error' blocks the write."""
    out = []
    n = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df.columns else 0

    calc = (4 * n("Try Scored - Total") + 2 * n("Conversion - Made")
            + 2 * n("Penalty Goal - Made") + 1 * n("Field Goal - 1 Point Made")
            + 2 * n("Field Goal - 2 Point Made"))
    if "Points Scored" in df.columns:
        bad = int((n("Points Scored") != calc).sum())
        if bad:
            out.append(("error", f"{bad} of {len(df)} rows: Points Scored does not equal "
                                 f"4*tries + 2*conversions + 2*penalty goals + field goals. "
                                 f"This is the failure that once flipped a match result."))
        else:
            out.append(("ok", f"scoring arithmetic consistent on all {len(df)} rows"))
    else:
        out.append(("warn", "no Points Scored column — scoring could not be checked"))

    if {"Team", "Round"} <= set(df.columns):
        sq = df.groupby(["Round", "Team"]).size()
        odd = sq[(sq < 15) | (sq > 18)]
        if len(odd):
            out.append(("warn", f"{len(odd)} team-matches name fewer than 15 or more than "
                                f"18 players: {dict(list(odd.items())[:4])}"))
        else:
            out.append(("ok", f"{len(sq)} team-matches, all 15-18 players "
                              f"(mean {sq.mean():.1f})"))

    if "Minutes" in df.columns:
        mins = pd.to_numeric(df["Minutes"], errors="coerce")
        if int((mins.fillna(0) <= 0).sum()):
            out.append(("warn", f"{int((mins.fillna(0) <= 0).sum())} rows with no minutes"))
        if int((mins > 100).sum()):
            out.append(("warn", f"{int((mins > 100).sum())} rows over 100 minutes"))

    # A player CAN legitimately appear twice in one round: Super League reschedules
    # fixtures (2021 especially) and a mid-season transfer then leaves the same round
    # number recorded for two different clubs. That is real data. The genuine error is
    # the same player twice for the SAME club in the same round — which is how the
    # Australian file's 16 bad rows looked (80 minutes at two clubs, one exact
    # duplicate). So the key includes team, and a cross-club repeat is only a warning.
    idc = "Player ID" if "Player ID" in df.columns else "Full Name"
    if "Team" in df.columns:
        same_club = int(df.duplicated(subset=[idc, "Round", "Team"], keep=False).sum())
        cross_club = int(df.duplicated(subset=[idc, "Round"], keep=False).sum()) - same_club
        if same_club:
            out.append(("error", f"{same_club} rows share {idc} + Round + Team — the same "
                                 f"player twice for the same club in one round"))
        if cross_club:
            out.append(("warn", f"{cross_club} rows have one player at two different clubs "
                                f"in the same round: a mid-season move with a repeated "
                                f"round number. Kept — it is real."))
    else:
        dup = int(df.duplicated(subset=[idc, "Round"], keep=False).sum())
        if dup:
            out.append(("error", f"{dup} rows share {idc} + Round and there is no Team "
                                 f"column to tell a transfer from a duplicate"))
    return out


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Weekly Stats Perform import")
    ap.add_argument("--file", required=True)
    ap.add_argument("--competition", required=True, choices=["NRL", "SL", "NSW", "QLD"])
    ap.add_argument("--season", required=True, type=int)
    ap.add_argument("--sheet")
    ap.add_argument("--rounds", type=int, nargs="*",
                    help="only these rounds; default is every round in the file")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="write despite validation errors")
    ap.add_argument("--no-ratings", action="store_true")
    a = ap.parse_args()

    df, label = read_source(a.file, a.sheet)
    print(f"source: {label} — {len(df):,} rows, {df.shape[1]} columns")
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
    df = df[df["Round"].notna()]
    if a.rounds:
        df = df[df["Round"].isin(a.rounds)]
        if df.empty:
            sys.exit(f"ABORT: no rows for round(s) {a.rounds} in {label}")
    rounds = sorted(int(r) for r in df["Round"].unique())
    print(f"target: {a.competition} {a.season}, round(s) {rounds} — {len(df):,} rows")

    print("\nvalidation")
    checks = validate(df, label)
    for sev, msg in checks:
        print(f"  [{sev.upper():5s}] {msg}")
    errors = [m for s, m in checks if s == "error"]
    if errors and not a.force:
        sys.exit(f"\nABORT: {len(errors)} validation error(s). Fix the feed, or pass "
                 f"--force to write anyway (and say so in the write-up).")

    con = sqlite3.connect(DB)
    existing = pd.read_sql(
        "SELECT round, count(*) n FROM player_match_stats "
        "WHERE competition=? AND season=? GROUP BY 1", con,
        params=(a.competition, a.season)).set_index("round")["n"].to_dict()
    overlap = {r: existing[r] for r in rounds if r in existing}
    if overlap:
        print(f"\nround(s) already present and will be REPLACED: {overlap}")
    new_rounds = [r for r in rounds if r not in existing]
    if new_rounds:
        print(f"new round(s): {new_rounds}")

    # ── shape the frame ──────────────────────────────────────────────────────
    pid, pid_src, names = resolve_player_ids(df, con)
    pos, pos_src = backfill_position(df, con, pid)
    df = df.copy()
    df["player_id"], df["player"] = pid, names
    if "Player ID" in df.columns:      # keep the canonical column clean as well,
        df["Player ID"] = pid          # or the raw table stores "24528.0"
    df["competition"], df["season"] = a.competition, a.season
    df["Competition"], df["Season"] = a.competition, a.season
    df["team"] = df["Team"]
    df["opposition"] = df.get("Opposition")
    df["round"] = df["Round"]
    df["minutes"] = pd.to_numeric(df["Minutes"], errors="coerce").fillna(0)
    df["position"], df["position_source"] = pos, pos_src
    for src, dst in sp.ENGINE_MAP.items():
        df[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan
    df["fantasy"] = sp.fantasy_proxy(df)

    print(f"\nplayer identity: {dict(pid_src.value_counts())}")
    print(f"position source: {dict(pos_src.value_counts())}")
    unseen = pd.read_sql("SELECT player_id FROM players", con).player_id
    fresh = sorted(set(df.player_id) - set(unseen))
    print(f"players not seen before: {len(fresh)}"
          + (f" (e.g. {', '.join(df[df.player_id.isin(fresh[:3])].player.unique()[:3])})"
             if fresh else ""))

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        con.close()
        return

    # ── write: delete the target keys, then insert ───────────────────────────
    eng_cols = [r[1] for r in con.execute("PRAGMA table_info(player_match_stats)")]
    raw_cols = [r[1] for r in con.execute("PRAGMA table_info(player_match_raw)")]
    q = ",".join("?" * len(rounds))
    for tbl in ("player_match_stats", "player_match_raw"):
        comp_col = "competition" if tbl == "player_match_stats" else "Competition"
        season_col = "season" if tbl == "player_match_stats" else "Season"
        rnd_col = "round" if tbl == "player_match_stats" else '"Round"'
        con.execute(f'DELETE FROM {tbl} WHERE {comp_col}=? AND {season_col}=? '
                    f'AND {rnd_col} IN ({q})', [a.competition, a.season, *rounds])
    df.reindex(columns=eng_cols).to_sql("player_match_stats", con, if_exists="append",
                                        index=False)
    df.reindex(columns=raw_cols).to_sql("player_match_raw", con, if_exists="append",
                                        index=False)
    con.commit()

    # ── registry refresh (small table; rebuilt rather than patched) ──────────
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

    after = pd.read_sql("SELECT count(*) n, max(round) r FROM player_match_stats "
                        "WHERE competition=? AND season=?", con,
                        params=(a.competition, a.season))
    print(f"\nwritten. {a.competition} {a.season}: {int(after.n[0]):,} rows, "
          f"through round {int(after.r[0])} | players registry {len(players):,}")
    con.close()

    if a.no_ratings:
        print("--no-ratings: run regenerate_full.py when the batch is done.")
        return
    print("\nrebuilding ratings and contribution ...")
    r = subprocess.run([sys.executable, os.path.join(BASE, "regenerate_full.py")],
                       cwd=BASE, capture_output=True, text=True)
    print(r.stdout.strip()[-900:] or r.stderr.strip()[-900:])
    if r.returncode:
        sys.exit("ABORT: regenerate_full.py failed — ratings are stale, data is loaded.")


if __name__ == "__main__":
    main()
