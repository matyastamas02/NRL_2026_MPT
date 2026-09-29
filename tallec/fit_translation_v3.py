# -*- coding: utf-8 -*-
"""Competition translation, v3 — one feature definition, and the app's own number in.

What changed from v2, and why each change was forced rather than chosen:

  * **The model is given the cumulative Class score the app displays.** v2 trained on a
    season's mean composite z but was *called* with a cumulative, shrunk Class score, so
    the number it was asked about was not the kind of number it had learned on. Mike's
    question is "BOSC says 66 in the NSW Cup — what is that in the NRL?", and 66 is the
    cumulative figure, so that is what goes in on both sides now.

  * **What it predicts is a single season, not a career.** The target is the player's
    rating in the new competition built from the target season alone. A cumulative
    rating there would contain everything he did in that competition before the move,
    which is a past no projection ever predicted; grading against it flattered v2's
    numbers for anyone with a history in the target.

  * **Features come from `translation_features.FeatureSpec`**, fitted here and stored in
    the model file, so prediction cannot build its columns differently. v2 had two
    definitions and they disagreed: a position group handed to a raw-position mapper
    resolved to nothing, an unresolved position silently took the training *mean* of
    each dummy — a fractional blend of positions no player has — and a missing age was
    indistinguishable from an average one.

  * **Players are keyed on the Stats Perform Player ID throughout.** v2 matched Super
    League to Australia on name and date of birth because the ids were thought not to
    span competitions. They do: 224 players carry the same id on both sides.

  * **Pairs are built between any two competitions**, not only feeder→NRL and
    Australia↔Super League. The old layer definitions excluded NSW Cup ↔ Queensland Cup
    by construction, which is why the ladder had a hole in it.

The layer split is kept because it is a real distinction: a same-season pair holds the
player fixed, while an adjacent-season pair has a year of ageing and form drift inside
it. Nothing is fitted past `evaluation.freeze_season`.

This writes `translation_model_v3.pkl` and leaves v2 alone — `v1_holdout_record.json`
hashes the v2 artefacts, and overwriting them would break the only sealed out-of-sample
result the project has.

    python fit_translation_v3.py [--dry-run]
"""
import argparse
import json
import os
import pickle
import sqlite3
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import aging
import player_rating_engine as pre
import rating_history as rh
import runtime
import sp_schema as sp
import translation_features as tf

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
MODEL = os.path.join(BASE, "translation_model_v3.pkl")
MIN_GAMES = 3          # per side of a pair
ALPHA = 1.0
# feature families the model may use; see config.json -> translation.model_features and
# ABLATION_REPORT.md for the measurement behind the default
MODEL_FEATURES = json.load(
    open(os.path.join(BASE, "config.json"), encoding="utf-8")
)["translation"]["model_features"]


def career_position(con, through):
    """Raw career position per player per competition, from matches up to the cutoff.

    Interchange is a role; `sp.primary_position` takes the mode of a player's STARTING
    positions and only falls back to Interchange for a man who has never started.
    """
    d = pd.read_sql(
        "SELECT player_id, competition, season, position FROM player_match_stats "
        "WHERE position IS NOT NULL AND position<>'Unknown' AND season<=?",
        con, params=(through,))
    return d.groupby(["player_id", "competition"])["position"].agg(sp.primary_position)


def build_pairs(cum, sea, ext, pos, dob):
    """Every observed competition change, as (what we knew) -> (what happened)."""
    src = cum.merge(ext, left_on=["player_id", "season", "comp"],
                    right_on=["player_id", "season", "comp"], how="left")
    src = src.rename(columns={"class_score": "class_source", "n_games": "n_cum",
                              "comp": "source", "season": "season_src"})
    tgt = sea.rename(columns={"class_score": "class_target", "comp": "target",
                              "season": "season_tgt", "n_games": "n_tgt"})
    src = src[src.games_src >= MIN_GAMES]
    tgt = tgt[tgt.n_tgt >= MIN_GAMES]

    j = src.merge(tgt[["player_id", "target", "season_tgt", "class_target", "n_tgt"]],
                  on="player_id")
    j = j[(j.source != j.target)
          & (j.season_tgt - j.season_src).isin([0, 1])].copy()
    j["gap"] = j.season_tgt - j.season_src
    j["layer"] = np.where(j.gap == 0, "A_same_season", "B_next_season")
    j["pair"] = j.source + "->" + j.target
    j["raw_position"] = [pos.get((p, c)) for p, c in zip(j.player_id, j.source)]
    # sp.parse_dob rather than pd.to_datetime: the column mixes ISO and day-first
    # dates, and a single inferred format silently drops whichever is in the minority
    j["dob"] = sp.parse_dob(j.player_id.map(dob))
    j["age"] = sp.age_at(j["dob"], j.season_src)
    j["age_delta"] = aging.expected_delta(j["age"])
    return j.reset_index(drop=True)


def fit_layer(p, label, use_pairs):
    pairs = sorted(p.pair.unique()) if use_pairs else []
    spec = tf.FeatureSpec.fit(p, pairs=pairs)
    X, used = spec.transform(p)
    # only the families config selects reach the fit; the rest are still computed and
    # reported, so the app can tell a recruiter what was known about a player even where
    # the model does not use it
    keep = tf.select_columns(list(X.columns), MODEL_FEATURES)
    X = X[keep]
    # applied after standardisation on every path, or the scaler undoes it
    mult = tf.pooling_multipliers(keep)
    y = p.class_target.values
    groups = p.player_id.values
    naive = float(np.sqrt(np.mean((y - p.class_source.values) ** 2)))

    n_splits = min(5, p.player_id.nunique())
    preds = np.full(len(y), np.nan)
    if n_splits >= 2:
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            sc = StandardScaler().fit(X.iloc[tr])
            m = Ridge(alpha=ALPHA).fit(sc.transform(X.iloc[tr]) * mult, y[tr])
            preds[te] = m.predict(sc.transform(X.iloc[te]) * mult)
    rmse = float(np.sqrt(np.nanmean((y - preds) ** 2)))
    mae = float(np.nanmean(np.abs(y - preds)))
    r2 = float(1 - np.nansum((y - preds) ** 2) / np.nansum((y - y.mean()) ** 2))

    sc = StandardScaler().fit(X)
    full = Ridge(alpha=ALPHA).fit(sc.transform(X) * mult, y)
    mean_row = X.mean().to_frame().T
    shift = float(full.predict(sc.transform(mean_row) * mult)[0]
                  - X.class_source.mean())

    print(f"\n=== {label} ===")
    print(f"  observations {len(y)} | players {p.player_id.nunique()} | "
          f"directions {p.pair.nunique()}")
    print(f"  out-of-sample (GroupKFold by player)  RMSE {rmse:5.2f}  MAE {mae:5.2f}  "
          f"points on the 0-100 scale")
    print(f"  naive 'he rates the same'             RMSE {naive:5.2f}"
          f"   -> improvement {(1 - rmse / naive) * 100:+.1f}%   grouped R^2 {r2:+.3f}")
    print(f"  level shift for an average player     {shift:+.2f} points")
    # the columns the model was actually fitted on, not the full design — prediction
    # must select the same ones or the coefficients land on the wrong features
    return dict(label=label, model=full, scaler=sc, spec=spec.to_dict(),
                features=keep, mult=mult.tolist(), n=len(y),
                players=int(p.player_id.nunique()), rmse=rmse, mae=mae, naive=naive,
                r2=r2, shift=shift, pairs=pairs)


def ladder(p):
    """Mean observed shift per direction, kept separate by horizon.

    These are two different questions and the answer to one is not the answer to the
    other: `A_same_season` is what a player was worth in the other competition in the
    same year, `B_next_season` is what he went on to do the year after. Until
    2026-09-22 both were averaged into a single figure that was then quoted for
    whichever horizon the caller asked about — a `layer_mix` column recorded that the
    mixing had happened without preventing it. The external review of that date was
    right that this is two estimands wearing one number.

    An `overall` row is still produced per direction, because a thin pair sometimes has
    no usable estimate on one horizon alone, but it is now labelled rather than implied.
    """
    rows = []
    for (s, t, layer), g in p.groupby(["source", "target", "layer"]):
        d = g.class_target - g.class_source
        rows.append(dict(source=s, target=t, layer=layer, n=len(g),
                         shift_pts=float(d.mean()),
                         se_pts=float(d.std() / np.sqrt(len(g)))))
    for (s, t), g in p.groupby(["source", "target"]):
        d = g.class_target - g.class_source
        rows.append(dict(source=s, target=t, layer="overall", n=len(g),
                         shift_pts=float(d.mean()),
                         se_pts=float(d.std() / np.sqrt(len(g)))))
    return pd.DataFrame(rows).sort_values(["layer", "shift_pts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    print(f"building rating history through {pre.FREEZE_SEASON} ...")
    cum, sea, ext = rh.all_competitions(DB, through=pre.FREEZE_SEASON)
    con = sqlite3.connect(DB)
    pos = career_position(con, pre.FREEZE_SEASON)
    dob = pd.read_sql("SELECT player_id, dob FROM players WHERE dob IS NOT NULL",
                      con).set_index("player_id").dob
    con.close()
    print(f"  cumulative {len(cum):,} | season-only {len(sea):,}")

    p = build_pairs(cum, sea, ext, pos, dob)
    print(f"\npairs: {len(p)} over {p.player_id.nunique()} players, "
          f"{p.pair.nunique()} directions")
    print(p.groupby(["layer", "pair"]).size().unstack(fill_value=0).to_string())
    miss = p.raw_position.isna().mean()
    print(f"\nposition known for {(1 - miss) * 100:.0f}% of pairs, age for "
          f"{p.age.notna().mean() * 100:.0f}%")

    out = {}
    for layer, g in p.groupby("layer"):
        if len(g) < 40:
            print(f"\n{layer}: only {len(g)} pairs, not fitted")
            continue
        out[layer] = fit_layer(g.reset_index(drop=True), layer, use_pairs=True)

    lad = ladder(p)
    print("\n=== ladder — mean observed change, in points of the 0-100 rating ===")
    print(lad.round(2).to_string(index=False))
    print("  (negative = he rates LOWER in the target, i.e. the target is the stronger "
          "pool)")

    if a.dry_run:
        print(f"\ndry run, nothing written ({time.time() - t0:.0f}s)")
        return

    keep = ["player_id", "name", "layer", "source", "target", "season_src",
            "season_tgt", "class_source", "class_target", "raw_position", "age",
            "mins_pg", "games_src", "n_tgt"]
    con = sqlite3.connect(DB)
    with runtime.guarded_write("fit_translation_v3",
                               note=f"freeze_season={pre.FREEZE_SEASON}"):
        lad.to_sql("translation_ladder_v3", con, if_exists="replace", index=False)
        p[[c for c in keep if c in p.columns]].to_sql(
            "translation_pairs_v3", con, if_exists="replace", index=False)
        pd.DataFrame([{k: v for k, v in m.items()
                       if k not in ("model", "scaler", "spec", "features", "pairs")}
                      for m in out.values()]).to_sql(
            "translation_model_v3_meta", con, if_exists="replace", index=False)
        runtime.record_model_run(
            "translation_model_v3",
            {k: {kk: vv for kk, vv in m.items()
                 if kk not in ("model", "scaler", "spec", "features", "pairs")}
             for k, m in out.items()},
            time.time() - t0, script="fit_translation_v3.py")
        with open(MODEL, "wb") as f:
            pickle.dump({"version": 3, "layers": out,
                         "ladder": lad.to_dict("records"),
                         "freeze_season": pre.FREEZE_SEASON,
                         "min_games": MIN_GAMES}, f)
        con.commit()
    con.close()
    print(f"\nwrote {os.path.basename(MODEL)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
