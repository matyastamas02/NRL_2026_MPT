# -*- coding: utf-8 -*-
"""Player ratings as they stood at each point in time, computed once for everyone.

Three scripts needed this and each grew its own copy, which is how two of them ended up
measuring subtly different things. It is one definition now.

Two quantities, and the difference between them matters:

  cumulative   what the app shows: a player's Class built from every match he has
               played in that competition up to and including season S. This is what a
               recruiter is looking at, so it is what a projection has to start from.

  season_only  a rating built from season S alone. This is what a projection has to be
               judged against, because a cumulative rating in the competition a player
               moved *to* contains his career there before the move and would grade the
               model on a past it never predicted.

Both are shrunk by the same engine and live on the same 0-100 scale, so they can be
compared with each other. Every match is standardized within its own competition-season
pool before either is computed, so a strong 2023 is measured against 2023.
"""
import sqlite3

import pandas as pd

import player_rating_engine as pre

COLS = ["player_id", "player", "season", "round", "team", "position", "minutes",
        "all_run_metres", "p_c_m", "tackle_breaks", "line_breaks", "tackles",
        "offloads", "try_assists", "tries", "errors"]
MIN_POOL = 100          # a season smaller than this cannot define a pool


def load(con, comp, through=None):
    q = f"SELECT {', '.join(COLS)} FROM player_match_stats WHERE competition=?"
    p = [comp]
    if through is not None:
        q += " AND season<=?"
        p.append(through)
    return pd.read_sql(q, con, params=p)


def _mode(hist):
    """Position-relative or competition-relative, decided once for the whole history."""
    known = hist["position"].notna() & (hist["position"] != "Unknown")
    cov = float(known.mean()) if len(hist) else 0.0
    return (None if cov >= pre.MIN_POS_COVERAGE else "competition_relative"), cov


def composites(hist, comp, force):
    """Per-match composites, each standardized inside its own season's pool."""
    parts = []
    for s in sorted(hist.season.dropna().unique()):
        part = hist[hist.season == s]
        if len(part) < MIN_POOL:
            continue
        parts.append(pre.PlayerRatingEngine(comp, force_mode=force)._composite(part))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _snap(pm, comp, force, season, label):
    eng = pre.PlayerRatingEngine(comp, force_mode=force)
    snap = eng.compute_snapshot(pm=pm)
    keep = ["player_id", "name", "class_score", "class_z", "n_games",
            "shrinkage_B", "confidence"]
    if "raw_score" in snap.columns:
        keep.append("raw_score")
    snap = snap[keep].copy()
    snap["season"] = int(season)
    snap["comp"] = comp
    snap["basis"] = label
    return snap


def build(con, comp, through=None):
    """Return (cumulative, season_only, extras) for one competition.

    `extras` carries the per-player-season minutes and matches used by the translation
    features, taken from the source season alone so nothing later can leak in.
    """
    hist = load(con, comp, through)
    if hist.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    force, _ = _mode(hist)
    pm = composites(hist, comp, force)
    if pm.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cum, sea = [], []
    for s in sorted(pm.season.dropna().unique()):
        upto = pm[pm.season <= s]
        if upto["ratable"].sum() >= 20:
            cum.append(_snap(upto, comp, force, s, "cumulative"))
        this = pm[pm.season == s]
        if this["ratable"].sum() >= 20:
            sea.append(_snap(this, comp, force, s, "season_only"))

    r = pm[pm["ratable"]]
    extras = (r.groupby(["player_id", "season"])
               .agg(mins_pg=("minutes", "mean"), games_src=("minutes", "size"),
                    position=("position", lambda s_: s_.mode().iloc[0]
                              if len(s_.mode()) else None))
               .reset_index())
    extras["comp"] = comp
    return (pd.concat(cum, ignore_index=True) if cum else pd.DataFrame(),
            pd.concat(sea, ignore_index=True) if sea else pd.DataFrame(),
            extras)


def all_competitions(db, comps=("NRL", "SL", "NSW", "QLD"), through=None):
    con = sqlite3.connect(db)
    cum, sea, ext = [], [], []
    for c in comps:
        a, b, e = build(con, c, through)
        for dst, src in ((cum, a), (sea, b), (ext, e)):
            if len(src):
                dst.append(src)
    con.close()
    return (pd.concat(cum, ignore_index=True), pd.concat(sea, ignore_index=True),
            pd.concat(ext, ignore_index=True))
