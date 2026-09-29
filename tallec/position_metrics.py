# -*- coding: utf-8 -*-
"""Mike's position metrics, computed and benchmarked.

`metric_spec.py` says which twelve metrics belong to each position and how each is
calculated. This computes them from the feed and turns each one into a 0-100 figure
against the players who do the same job — a prop's run metres against other props, in
the same competition and the same season, never against a winger's.

Three decisions worth arguing with:

**A volume is a per-match average.** A season total would rank availability: a man with
twenty matches would beat an equal player with eight for no footballing reason. Rates
are computed from season totals on both sides of the division, so a quiet game does not
count as heavily as a busy one.

**The peer pool is competition, season and position.** That is narrow on purpose — the
0-100 is meant to answer "how does he compare with the men doing his job right now" —
and it is why the thin pools matter: Super League Lock has about forty-five players, so
a single place in the order is worth more than two points there.

**A position comes from the season, not the career.** A man who moved from centre to
back row is judged this season as a back-rower, which is what he was. Where a season has
no starting appearance the career position stands in, and where neither exists he is not
benchmarked at all rather than being guessed into a pool.

Lower-is-better metrics — errors, infringements, breaks conceded — are inverted, so a
clean player scores high on every figure rather than having to be read backwards on
three of them.
"""
import os
import sqlite3

import numpy as np
import pandas as pd

import metric_spec as ms
import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
MIN_MATCHES = 5        # below this a season is not benchmarked
MIN_POOL = 12          # a peer pool smaller than this is not a benchmark

# engine position group -> Mike's block key
GROUP_TO_BLOCK = {v["engine_group"]: k for k, v in ms.POSITIONS.items()}

# What `compute` refused to score and what it discarded, filled on each run so the
# report can say so. A metric that quietly disappears and a metric that was never
# defined look identical from the output alone, which is how a wrong one survived.
BLOCKED = []
DROPPED = []


def season_totals(con, through=None):
    """Per player, competition and season: matches, minutes and every feed total."""
    sums = ", ".join(f'SUM(COALESCE(r."{col}", 0)) AS "{code}"'
                     for code, col in ms.FEED.items() if code != "Minutes")
    q = f"""SELECT s.player_id, s.competition AS comp, s.season,
                   COUNT(*) AS matches, SUM(COALESCE(s.minutes, 0)) AS minutes, {sums}
            FROM player_match_stats s
            JOIN player_match_raw r ON r.player_id = s.player_id
             AND r.Competition = s.competition AND r.Season = s.season
             AND r."Round" = s.round
            {"WHERE s.season <= ?" if through else ""}
            GROUP BY 1, 2, 3"""
    return pd.read_sql(q, con, params=(through,) if through else None)


def positions(con, through=None):
    """(player, competition, season) -> position group, from the season then the career.

    `sp.primary_position` takes the mode of a man's STARTING positions and only returns
    Interchange for someone who never started, so a bench forward is judged as the
    forward he is rather than as a role.
    """
    where = "WHERE position IS NOT NULL AND position <> 'Unknown'"
    season_cut = " AND season <= ?" if through else ""
    args = (through,) if through else None

    # Every player-season that exists, whether or not it has a usable position. The
    # career fallback used to be written as `pos_season.fillna(pos_career)` over a frame
    # built FROM the positioned rows only, so pos_season was never null and the fallback
    # never fired — it read as a safety net while catching nothing. The seasons it was
    # meant to catch were missing from the frame entirely and were then dropped by the
    # inner join downstream, silently.
    allseasons = pd.read_sql(
        f"SELECT DISTINCT player_id, competition AS comp, season "
        f"FROM player_match_stats {'WHERE season <= ?' if through else ''}",
        con, params=args)

    d = pd.read_sql(f"SELECT player_id, competition AS comp, season, position "
                    f"FROM player_match_stats {where}{season_cut}", con, params=args)
    by_season = (d.groupby(["player_id", "comp", "season"])["position"]
                  .agg(sp.primary_position).rename("pos_season").reset_index())
    by_career = (d.groupby(["player_id", "comp"])["position"]
                  .agg(sp.primary_position).rename("pos_career").reset_index())

    out = (allseasons.merge(by_season, on=["player_id", "comp", "season"], how="left")
                     .merge(by_career, on=["player_id", "comp"], how="left"))
    out["position"] = out.pos_season.fillna(out.pos_career)
    out["position_from"] = np.where(out.pos_season.notna(), "season",
                                    np.where(out.pos_career.notna(), "career", "none"))
    out["group"] = out.position.map(sp.POSITION_GROUP)
    return out[["player_id", "comp", "season", "position", "position_from", "group"]]


def compute(totals, pos):
    """Every figure the spec defines for each player's position, one row per figure.

    Also records what fell out on the way. A player-season reaches a benchmark only if
    it has enough matches, a position that resolves, and a block to belong to; each of
    those is a defensible rule and each was previously an inner join that discarded rows
    without saying so.
    """
    BLOCKED.clear()
    DROPPED.clear()
    n_all = len(totals)
    d = totals.merge(pos, on=["player_id", "comp", "season"], how="left")
    DROPPED.append(dict(reason="no position, season or career",
                        rows=int(d["group"].isna().sum())))
    d = d[d["group"].notna()]
    DROPPED.append(dict(reason=f"fewer than {MIN_MATCHES} matches",
                        rows=int((d.matches < MIN_MATCHES).sum())))
    d = d[d.matches >= MIN_MATCHES]
    d["block"] = d["group"].map(GROUP_TO_BLOCK)
    DROPPED.append(dict(reason="position group has no metric block",
                        rows=int(d.block.isna().sum())))
    d = d[d.block.notna()]
    DROPPED.append(dict(reason="benchmarked", rows=len(d)))
    DROPPED.append(dict(reason="player-seasons in", rows=n_all))

    rows = []
    for block, g in d.groupby("block"):
        s = {c: g[c].values for c in ms.FEED if c in g.columns}
        s["matches"] = g.matches.values
        s["minutes"] = g.minutes.values
        for category, metrics in ms.SPEC[block].items():
            for m in metrics:
                # A blocked metric is not a weaker version of itself, it is a different
                # quantity wearing the right name, and ranking players on it would be
                # worse than leaving the slot empty. It is skipped and reported.
                if m.get("blocked"):
                    BLOCKED.append(dict(block=block, category=category,
                                        volume=m["volume"], rate=m["rate"],
                                        reason=m["blocked"]))
                    continue
                # A stat the feed never recorded sums to zero, not to nothing, so a
                # metric that depends on it reads as "he did none of it" and ranks the
                # whole competition-season at the floor — except for the handful of
                # players who happen to carry a stray value, who then rank at the top.
                # Post-contact metres did exactly that: in 2024, 81% of players scored
                # a flat zero and the other 19% came out at 100. The availability rules
                # in metric_spec exist for this, and are applied before anything is
                # ranked.
                ok, degraded = _available(
                    m["inputs"], m.get("optional", ()),
                    g.comp.values, g.season.values)
                for form, label, fn in (("volume", m["volume"], ms.VOLUME[m["volume"]]),
                                        ("rate", m["rate"], ms.RATE[m["rate"]])):
                    rows.append(pd.DataFrame({
                        "player_id": g.player_id.values, "comp": g.comp.values,
                        "season": g.season.values, "position": g["group"].values,
                        "block": block, "category": category, "form": form,
                        "metric": label,
                        "value": np.where(ok, fn(s), np.nan),
                        "available": ok, "degraded": degraded,
                        "matches": g.matches.values}))
    return pd.concat(rows, ignore_index=True)


def _available(inputs, optional, comps, seasons):
    """Per row: can this metric be computed, and was anything left out to do it?

    An input the metric cannot do without blocks it entirely. An optional one that is
    missing degrades it — still right within the competition, no longer comparable
    across competitions — and that is recorded rather than hidden.
    """
    ok = np.ones(len(comps), dtype=bool)
    degraded = np.zeros(len(comps), dtype=bool)
    for code in inputs:
        av = ms.AVAILABILITY.get(code)
        if not av:
            continue
        gap = np.zeros(len(comps), dtype=bool)
        if av.get("usable_from"):
            gap |= np.asarray(seasons) < av["usable_from"]
        for miss in (av.get("missing_in") or []):
            gap |= np.asarray(comps) == miss
        if code in optional:
            degraded |= gap
        else:
            ok &= ~gap
    return ok, degraded


def benchmark(vals):
    """Turn each raw figure into 0-100 against the same job, competition and season.

    A percentile rather than a z-score: these distributions are skewed and bounded in
    ways a normal curve does not describe — most halves kick, almost no wingers do — and
    a percentile says something true about both.
    """
    vals = vals.copy()
    vals["lower_is_better"] = vals.metric.isin(ms.LOWER_IS_BETTER)
    key = ["comp", "season", "position", "metric", "form"]

    # The pool is how many players can actually be ranked, which is not how many rows
    # exist. `size` counted the missing ones too, so a metric with three real values
    # among twelve players cleared a floor meant to require twelve — and then ranked
    # those three against each other as if the field were full.
    vals["pool"] = vals.groupby(key)["value"].transform("count")

    # Rank the signed value rather than flipping the percentile afterwards. Subtracting
    # from 100 is not symmetric: in a pool of twelve, more-is-better ran 8.33 to 100 and
    # lower-is-better ran 0 to 91.67, so the best tackler in a group could not score what
    # the best runner scored. Negating the value mirrors the ranks exactly instead.
    #
    # The percentile is (rank - 0.5) / n, which also stops the top of every pool being
    # branded a flat 100 and the bottom a flat 0 — a twelfth place out of twelve is a
    # position in a small field, not the absence of the skill.
    signed = np.where(vals.lower_is_better, -vals["value"], vals["value"])
    vals["_signed"] = signed
    r = vals.groupby(key)["_signed"].rank(method="average")
    vals["score"] = (r - 0.5) / vals["pool"] * 100.0

    # a pool too small to rank against is left unscored rather than scored badly
    vals.loc[vals["pool"] < MIN_POOL, "score"] = np.nan
    vals.loc[vals.value.isna(), "score"] = np.nan
    return vals.drop(columns=["lower_is_better", "_signed"])


def category_scores(scored):
    """One figure per category per player, averaging the metrics that could be computed.

    It carries how it was made. A category averaged over six clean metrics and one
    averaged over two, of which both were computed without an input the name promises,
    are not the same number, and until now they were indistinguishable once averaged.
    `metrics` is how many went in, `of` how many the spec defines for that category, and
    `degraded` how many were computed without an input they name.
    """
    full = (scored.groupby(["player_id", "comp", "season", "position", "category"])
                  .agg(of=("score", "size")).reset_index())
    got = (scored.dropna(subset=["score"])
                 .groupby(["player_id", "comp", "season", "position", "category"])
                 .agg(score=("score", "mean"), metrics=("score", "size"),
                      degraded=("degraded", "sum"))
                 .reset_index())
    out = got.merge(full, on=["player_id", "comp", "season", "position", "category"],
                    how="left")
    out["degraded"] = out["degraded"].astype(int)
    return out


def build(db=DB, through=None):
    con = sqlite3.connect(db)
    tot = season_totals(con, through)
    pos = positions(con, through)
    con.close()
    vals = compute(tot, pos)
    return benchmark(vals)


STORE_COLS = ["player_id", "comp", "season", "position", "block", "category", "form",
              "metric", "value", "score", "available", "degraded", "pool", "matches"]


def write(db=DB, through=None):
    """Replace the stored metrics. The only writer of `player_position_metrics`."""
    import runtime
    s = build(db, through)
    cat = category_scores(s)
    con = sqlite3.connect(db)
    with runtime.guarded_write("position_metrics", note=f"through={through}"):
        s[STORE_COLS].to_sql("player_position_metrics", con, if_exists="replace",
                             index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_ppm_player "
                    "ON player_position_metrics(player_id, comp, season)")
        cat.to_sql("player_position_category", con, if_exists="replace", index=False)
        con.commit()
    con.close()
    return s, cat


if __name__ == "__main__":
    import sys

    import player_rating_engine as pre
    if "--write" in sys.argv:
        s, cat = write(through=pre.FREEZE_SEASON)
        print(f"wrote player_position_metrics: {len(s):,} rows | "
              f"player_position_category: {len(cat):,} rows")
        raise SystemExit(0)
    s = build(through=pre.FREEZE_SEASON)
    print(f"{len(s):,} figures for {s.player_id.nunique():,} players")
    print(f"scored: {s.score.notna().mean():.0%}")
    print("\nunscored, by reason:")
    print(f"  the stat is not recorded there: {(~s.available).sum():,}")
    print(f"  no opportunity to divide by   : "
          f"{(s.available & s.value.isna()).sum():,}")
    print(f"  peer pool under {MIN_POOL}          : "
          f"{((s['pool'] < MIN_POOL) & s.value.notna()).sum():,}")
    print(f"  computed without a named input : {int(s.degraded.sum()):,} "
          f"(degraded, still ranked)")
    if BLOCKED:
        print(f"\nnot computed at all ({len(BLOCKED)} metric pairs):")
        for b in BLOCKED:
            print(f"  {b['block']:8s} {b['category']:22s} {b['volume']} / {b['rate']}")
            print(f"           {b['reason']}")
    if DROPPED:
        print("\nplayer-seasons:")
        for row in DROPPED:
            print(f"  {row['reason']:38s} {row['rows']:>7,}")
    print("\nper position (2025):")
    g = s[s.season == 2025]
    print(g.groupby("position").agg(players=("player_id", "nunique"),
                                    scored=("score", lambda v: f"{v.notna().mean():.0%}")
                                    ).to_string())
