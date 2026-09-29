# -*- coding: utf-8 -*-
"""An explicit cohort of players entering a competition, built once and stored.

What it replaces. `rolling_backtest.moves_into` inferred moves by joining every
competition a player appeared in last season to every competition he appeared in this
one. That is not a transfer list. Three things went wrong with it, all found by the
external review of 2026-09-22 and all confirmed here:

  * **Reciprocal phantoms.** 15.5% of player-seasons span two competitions — a fringe
    NRL player shuttles to the NSW Cup and back within one year. The join turned each
    such man into two opposite "moves", NRL->NSW *and* NSW->NRL, and scored the model on
    both. He did not move twice; he did not move at all.
  * **Duplicates.** 1,140 rows came from 531 players, with 263 rows sharing a
    player and origin. A bootstrap over rows then treated one man's several appearances
    as several independent pieces of evidence.
  * **Survivors only.** A row existed only if the player later played at least three
    matches in the target. Everyone who moved and did not hold a place was invisible,
    so the set answered "how well do arrivals who established themselves perform" while
    being read as "how well do arrivals perform".

What this builds instead. One row per player, target competition and season, with a
single source decided in advance — the competition where he played the most minutes the
season before. A player who genuinely appeared in two competitions gets a row for each,
because he genuinely did; what he does not get is a second row describing the same fact
backwards.

Every row carries what actually happened afterwards, including nothing. `arrived` is
whether he played in the target at all and `rated` whether he played enough to be given
a rating, so the two questions a club actually asks can be answered separately:

    1. does a player at this level reach a usable role in that competition?
    2. how good is he once he does?

**The honest limit.** We have no signing or registration data. The population here is
everyone who played in the source competition, not everyone a club signed, so an arrival
rate answers "of the players at this level, what share appeared" and not "of the players
we recruited, what share worked out". A signing who never played a minute is invisible
to us and stays invisible; a signing who played once or twice is now visible, where
before he was dropped. Whether that gap matters depends on how it is read, which is why
the column is named `arrived` rather than `signed`.

    python transition_events.py            # report only
    python transition_events.py --write    # replace the transition_events table
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
MIN_SOURCE_MATCHES = 3     # below this a source season says too little to forecast from
MIN_RATED_MATCHES = 3      # below this the target season cannot carry a rating

TYPES = ["first", "returning", "dual_registered"]


def appearances(con, through=None):
    """Matches and minutes per player, competition and season."""
    q = ("SELECT player_id, competition AS comp, season, COUNT(*) AS matches, "
         "SUM(COALESCE(minutes, 0)) AS minutes FROM player_match_stats ")
    if through:
        q += "WHERE season <= ? "
    q += "GROUP BY 1, 2, 3"
    return pd.read_sql(q, con, params=(through,) if through else None)


def pick_source(prior):
    """One source competition per player and season, decided before any outcome is seen.

    Most minutes wins, then most matches, then the competition code alphabetically. The
    last is arbitrary but it has to be *something* fixed: a tie broken by row order makes
    the cohort depend on how the database happened to return rows, and two runs would
    then disagree about who moved from where.
    """
    p = prior[prior.matches >= MIN_SOURCE_MATCHES]
    p = p.sort_values(["player_id", "season", "minutes", "matches", "comp"],
                      ascending=[True, True, False, False, True])
    return (p.groupby(["player_id", "season"], as_index=False).first()
             .rename(columns={"comp": "source", "matches": "source_matches",
                              "minutes": "source_minutes"}))


def build(con, through=None):
    """One row per player, target competition and season."""
    app = appearances(con, through)
    seasons = sorted(app.season.unique())
    comps = sorted(app.comp.unique())
    ever = app.groupby(["player_id", "comp"]).season.min().rename("first_season")

    rows = []
    for season in seasons:
        prior = app[app.season == season - 1]
        if prior.empty:
            continue
        src = pick_source(prior)
        if src.empty:
            continue
        now = app[app.season == season]

        # the at-risk set: every source player against every OTHER competition, so a
        # player who did not turn up in the target is a row with zeros rather than an
        # absence. This is the population that can be observed, not the population that
        # was signed — see the module docstring.
        grid = src.assign(key=1).merge(
            pd.DataFrame({"target": comps, "key": 1}), on="key").drop(columns="key")
        grid = grid[grid.target != grid.source].copy()
        grid["season"] = season

        got = now.rename(columns={"comp": "target", "matches": "target_matches",
                                  "minutes": "target_minutes"})
        d = grid.merge(got[["player_id", "target", "target_matches", "target_minutes"]],
                       on=["player_id", "target"], how="left")
        d["target_matches"] = d.target_matches.fillna(0).astype(int)
        d["target_minutes"] = d.target_minutes.fillna(0).astype(int)

        # Everything that types a row is read from season-1 or earlier. An earlier
        # version of this file decided `dual_registered` from whether the player turned
        # up in both competitions during the season being forecast, which is the outcome
        # deciding who is allowed into the evaluation — the hindsight the review of
        # 2026-09-22 asked to be kept out.
        prev_t = prior.rename(columns={"comp": "target"})[["player_id", "target"]]
        prev_t["in_target_last"] = True
        d = d.merge(prev_t, on=["player_id", "target"], how="left")
        d["in_target_last"] = d.in_target_last.notna()

        fs = ever.reset_index().rename(columns={"comp": "target"})
        d = d.merge(fs, on=["player_id", "target"], how="left")
        d["seen_before"] = (d.first_season.notna() & (d.first_season < season))

        # A man who played in BOTH his source and this target last season is not moving
        # between them, he is shuttling across them — a fringe NRL forward who takes NSW
        # Cup minutes when he is not picked. Since the source is by construction his
        # biggest competition of last season, being in the target last season as well IS
        # dual registration, which is why there is no separate "continuing" type here:
        # with one source per season the two cannot be told apart, and calling him a
        # continuing player would imply a move that never happened.
        d["transition_type"] = np.select(
            [d.in_target_last, d.seen_before],
            ["dual_registered", "returning"],
            default="first")
        rows.append(d)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["arrived"] = out.target_matches > 0
    out["rated"] = out.target_matches >= MIN_RATED_MATCHES
    out["forecast_from_season"] = out.season - 1
    return out[["player_id", "source", "target", "forecast_from_season", "season",
                "source_matches", "source_minutes", "target_matches", "target_minutes",
                "transition_type", "arrived", "rated"]].sort_values(
        ["season", "target", "player_id"]).reset_index(drop=True)


def entries(events, types=("first", "returning")):
    """The rows a recruitment question is actually about.

    `dual_registered` is excluded because he never left: he was in both competitions
    last season and will be in one or both again, and a club reading him has his record
    in its own competition already. He stays in the table, so the exclusion is a filter
    a reader can lift rather than a decision baked into the data.
    """
    return events[events.transition_type.isin(types)]


def summarise(events):
    by_type = (events.groupby("transition_type")
               .agg(rows=("player_id", "size"), players=("player_id", "nunique"),
                    arrived=("arrived", "sum"), rated=("rated", "sum"))
               .reindex(TYPES).fillna(0).astype(int).reset_index())
    by_type["arrival_rate"] = (by_type.arrived / by_type.rows).round(4)

    ent = entries(events)
    by_pair = (ent.groupby(["source", "target"])
               .agg(at_risk=("player_id", "size"), arrived=("arrived", "sum"),
                    rated=("rated", "sum")).reset_index())
    by_pair["arrival_rate"] = (by_pair.arrived / by_pair.at_risk).round(4)
    by_pair = by_pair[by_pair.at_risk >= 20].sort_values("arrival_rate",
                                                         ascending=False)
    return by_type, by_pair


def duplicates(events):
    """What the old construction double-counted, measured against the new one."""
    d = events[events.arrived]
    return dict(rows=len(d), players=int(d.player_id.nunique()),
                per_player_target_season=int(
                    len(d) - len(d.drop_duplicates(["player_id", "target", "season"]))))


def write(db=DB, through=None):
    import runtime
    con = sqlite3.connect(db)
    ev = build(con, through)
    with runtime.guarded_write("transition_events", note=f"through={through}"):
        ev.to_sql("transition_events", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_te_player "
                    "ON transition_events(player_id, target, season)")
        con.commit()
    con.close()
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--through", type=int, default=None)
    a = ap.parse_args()
    import player_rating_engine as pre
    through = a.through or pre.FREEZE_SEASON

    if a.write:
        ev = write(through=through)
        print(f"wrote transition_events: {len(ev):,} rows")
    else:
        con = sqlite3.connect(DB)
        ev = build(con, through)
        con.close()

    by_type, by_pair = summarise(ev)
    print(f"\n{len(ev):,} player-target-season rows through {through}\n")
    print(by_type.to_string(index=False))
    print("\nA row is one player against one competition he could have entered. "
          "`arrived` is\nany appearance at all; `rated` is enough matches to carry a "
          "rating.")
    print("\nArrival rate by direction, entries only (first and returning):")
    print(by_pair.to_string(index=False))
    d = duplicates(ev)
    print(f"\nOf the rows where he did arrive: {d['rows']:,} rows over "
          f"{d['players']:,} players, {d['per_player_target_season']} duplicate "
          f"player-target-season combinations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
