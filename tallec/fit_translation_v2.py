# -*- coding: utf-8 -*-
"""Competition translation model, v2 — fitted on measured within-player moves.

v1 was fitted on 16 hand-collected NRL<->SL transfers with a hand-assigned target.
The Australian history file replaces that with observed moves:

  Layer A (core)  feeder -> NRL, SAME season, keyed on the permanent Player ID.
                  The same player, in the same year, rated in two competitions —
                  the cleanest read on a level difference there is, because the
                  player is held fixed. ~700 player-seasons.

  Layer B (Leeds) Australia <-> Super League, ADJACENT seasons, matched on name and
                  date of birth. Noisier (a year of ageing and form drift sits
                  inside each pair) but it is the move Leeds actually asks about.
                  Both sides are position-relative now that Super League carries
                  match-sheet positions for every season.

Method. Every player-match gets a composite performance z-score from the rating
engine, standardized WITHIN its own competition-season pool, so a score means "how
far above this competition's average". A pair therefore gives (z_source, z_target)
for one player, and the model predicts z_target.

  z_target ~ z_source + position + age + minutes per game (+ direction, gap for B)

Read the coefficients carefully. Both z's are noisy measures of the same underlying
ability, so the slope on z_source is attenuated toward zero by measurement error
(regression to the mean) — a slope below 1 is NOT evidence that form fails to carry
over. The interpretable quantity is the INTERCEPT: the expected level shift for an
average player. Every reported error is out-of-sample, grouped by player so no
player appears in both train and test, and compared against the naive baseline
"the player rates exactly the same in the new competition".

Writes: translation_model_v2.pkl, and tables translation_pairs (every observation
used) + translation_model_meta (fit statistics) into tallec.db.
"""
import os
import pickle
import sqlite3
import unicodedata
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import player_rating_engine as pre

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
MIN_GAMES = 3          # per side of a pair
# yardsticks measured from player_ratings, so effect sizes can be read in the units
# the app actually shows rather than in raw composite z
CLASS_Z_SD = 0.250      # spread of the shrunk player rating (class_z)
BENCH_PTS_PER_SD = 9.6  # points of the 0-100 benchmark per class_z SD
FEEDERS = ["NSW", "QLD"]
POS_GROUP = pre.POSITION_GROUP


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


# ── 1. composites, standardized within each competition-season pool ───────────
con = sqlite3.connect(DB)
COLS = ["player_id", "player", "season", "round", "team", "position", "minutes",
        "all_run_metres", "p_c_m", "tackle_breaks", "line_breaks", "tackles",
        "offloads", "try_assists", "tries", "errors"]


def composites(comp, season, force_mode):
    df = pd.read_sql(f"SELECT {', '.join(COLS)} FROM player_match_stats "
                     f"WHERE competition=? AND season=?", con, params=(comp, season))
    if len(df) < 100:
        return None
    eng = pre.PlayerRatingEngine(comp, force_mode=force_mode)
    pm = eng._composite(df)
    pm = pm[pm["ratable"]]
    out = (pm.groupby("player_id").agg(
        player=("player", "first"), z=("composite", "mean"), g=("composite", "size"),
        mins=("minutes", "mean"), pos=("position", lambda s: s.mode().iloc[0]))
        .reset_index())
    out["comp"], out["season"] = comp, season
    return out


def build(force_mode, comps):
    frames = []
    for comp in comps:
        seasons = pd.read_sql("SELECT DISTINCT season FROM player_match_stats "
                              "WHERE competition=? ORDER BY season", con,
                              params=(comp,)).season.tolist()
        for s in seasons:
            c = composites(comp, s, force_mode)
            if c is not None:
                frames.append(c)
    return pd.concat(frames, ignore_index=True)


# Layer A: NRL + feeders, all position-relative (coverage allows it on every pool)
print("building composites — layer A (position-relative)")
A = build(None, ["NRL"] + FEEDERS)
# Layer B: Australia + SL. Both sides used to be forced competition-relative because
# Super League had no position source; since 2026-08-25 it has match-sheet positions
# for all six seasons, so every competition now standardizes within position group and
# the two sides are comparable without giving that up.
print("building composites — layer B (position-relative, both sides)")
B = build(None, ["NRL"] + FEEDERS + ["SL"])
print(f"  layer A pool: {len(A):,} player-seasons | layer B pool: {len(B):,}")

# ── 2. identity: date of birth per player, for the layer-B name match ─────────
dob = pd.read_sql('SELECT player_id, "Full Name" nm, "Date of Birth" dob, Competition comp '
                  'FROM player_match_raw WHERE "Date of Birth" IS NOT NULL', con)
dob["dob"] = pd.to_datetime(dob.dob, errors="coerce").dt.date
dob["key"] = dob.nm.map(norm_name)
ident = dob.dropna(subset=["dob"]).groupby("player_id").agg(
    key=("key", "first"), dob=("dob", "first")).reset_index()

# ── 3. layer A pairs: same player, same season, feeder + NRL ─────────────────
nrl = A[A.comp == "NRL"].rename(columns={"z": "z_nrl", "g": "g_nrl", "mins": "m_nrl"})
fee = A[A.comp.isin(FEEDERS)].rename(columns={"z": "z_feed", "g": "g_feed", "mins": "m_feed"})
pa = fee.merge(nrl[["player_id", "season", "z_nrl", "g_nrl", "m_nrl"]],
               on=["player_id", "season"])
pa = pa[(pa.g_feed >= MIN_GAMES) & (pa.g_nrl >= MIN_GAMES)].copy()
pa["z_source"], pa["z_target"] = pa.z_feed, pa.z_nrl
pa["source"], pa["target"] = pa.comp, "NRL"
pa["gap"] = 0
pa["layer"] = "A_feeder_to_NRL"
print(f"\nlayer A pairs: {len(pa)} player-seasons, {pa.player_id.nunique()} players")

# ── 4. layer B pairs: Australia <-> SL, adjacent seasons, name + DoB ─────────
Bi = B.merge(ident, on="player_id", how="left")
Bi["side"] = np.where(Bi.comp == "SL", "SL", "AUS")
Bi = Bi[Bi.key.notna()]
aus, sl = Bi[Bi.side == "AUS"], Bi[Bi.side == "SL"]
# a name must identify exactly one date of birth on each side, else it is dropped
ok = lambda d: d.groupby("key").dob.nunique().eq(1)
aus = aus[aus.key.isin(ok(aus)[ok(aus)].index)]
sl = sl[sl.key.isin(ok(sl)[ok(sl)].index)]
pb = aus.merge(sl, on="key", suffixes=("_a", "_s"))
pb = pb[pb.dob_a == pb.dob_s]                       # date of birth must agree
pb = pb[(pb.g_a >= MIN_GAMES) & (pb.g_s >= MIN_GAMES)]
pb = pb[(pb.season_s - pb.season_a).abs() == 1]     # adjacent seasons only
rows = []
for _, r in pb.iterrows():
    a2s = r.season_s > r.season_a                   # Australia -> Super League
    rows.append(dict(
        player_id=r.player_id_a, player=r.player_a, key=r.key,
        z_source=r.z_a if a2s else r.z_s, z_target=r.z_s if a2s else r.z_a,
        g_feed=r.g_a if a2s else r.g_s, g_nrl=r.g_s if a2s else r.g_a,
        m_feed=r.mins_a if a2s else r.mins_s, pos=r.pos_a,
        season=min(r.season_a, r.season_s), gap=1,
        source=(r.comp_a if a2s else "SL"), target=("SL" if a2s else r.comp_a),
        layer="B_aus_to_SL" if a2s else "B_SL_to_aus"))
pb = pd.DataFrame(rows)
print(f"layer B pairs: {len(pb)} ({pb.layer.value_counts().to_dict() if len(pb) else ''}), "
      f"{pb.player_id.nunique() if len(pb) else 0} players")

# ── 5. features + fit ────────────────────────────────────────────────────────
age = pd.read_sql('SELECT player_id, min("Date of Birth") dob FROM player_match_raw '
                  'WHERE "Date of Birth" IS NOT NULL GROUP BY 1', con)
age["dob"] = pd.to_datetime(age.dob, errors="coerce")


def featurise(p):
    p = p.merge(age, on="player_id", how="left")
    p["age"] = (pd.to_datetime(p.season.astype(str) + "-06-30") - p.dob).dt.days / 365.25
    p["age"] = p.age.fillna(p.age.median())
    p["grp"] = p.pos.map(POS_GROUP).fillna("Bench")
    p["mins_pg"] = p.m_feed.fillna(p.m_feed.median())
    p["games_src"] = p.g_feed
    return p


def fit(p, label, extra=()):
    p = featurise(p).reset_index(drop=True)
    X = pd.get_dummies(p[["z_source", "age", "mins_pg", "games_src", "grp"] + list(extra)],
                       columns=["grp"], drop_first=True).astype(float)
    y = p.z_target.values
    groups = p.player_id.values
    naive = np.sqrt(np.mean((y - p.z_source.values) ** 2))
    gkf = GroupKFold(n_splits=min(5, p.player_id.nunique()))
    preds = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X.iloc[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X.iloc[tr]), y[tr])
        preds[te] = m.predict(sc.transform(X.iloc[te]))
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    ss = 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    sc = StandardScaler().fit(X)
    full = Ridge(alpha=1.0).fit(sc.transform(X), y)
    # level shift for an average player = prediction at the pool's mean features
    mean_row = X.mean().to_frame().T
    shift = float(full.predict(sc.transform(mean_row))[0] - X.z_source.mean())
    slope = float(np.polyfit(p.z_source, y, 1)[0])
    print(f"\n=== {label} ===")
    print(f"  observations {len(y)} | players {p.player_id.nunique()}")
    print(f"  out-of-sample RMSE      {rmse:.3f} z   (naive 'no change': {naive:.3f})")
    print(f"  improvement over naive  {(1-rmse/naive)*100:+.1f}%   | grouped R^2 {ss:+.3f}")
    # the composite is a weighted blend of correlated z-scores, so its own spread —
    # not 1.0 — is the yardstick. CLASS_Z_SD is the spread of the shrunk player
    # ratings the app shows, and BENCH_PTS_PER_SD converts to its 0-100 scale.
    print(f"  level shift, average player   {shift:+.3f} z"
          f"  = {shift/CLASS_Z_SD:+.2f} player-rating SD"
          f"  = {shift/CLASS_Z_SD*BENCH_PTS_PER_SD:+.1f} pts on the 0-100 scale")
    print(f"  raw slope z_target~z_source   {slope:.3f}  (attenuated by measurement error)")
    return dict(label=label, model=full, scaler=sc, features=list(X.columns),
                n=len(y), players=int(p.player_id.nunique()), rmse=float(rmse),
                naive=float(naive), r2=float(ss), shift=shift, slope=slope), p


mA, pAf = fit(pa, "Layer A — feeder to NRL, same season")
out = {"A": mA}
pairs = [pAf.assign(layer=pa.layer.values)]
if len(pb) >= 40:
    # the Australian side is three different competitions and layer A shows they sit
    # at different levels, so the model has to know which pair it is looking at
    pb["pair"] = pb.source + "->" + pb.target
    pb = pd.get_dummies(pb, columns=["pair"], drop_first=True)
    pair_cols = [c for c in pb.columns if c.startswith("pair_")]
    mB, pBf = fit(pb, "Layer B — Australia <-> Super League, adjacent seasons",
                  extra=tuple(pair_cols))
    out["B"] = mB
    pairs.append(pBf)

# ── raw pairwise ladder: no model, just the mean within-player difference ─────
print("\n=== competition ladder — raw within-player level shifts ===")
lad = []
for p_, lab in [(pa, "A"), (pb, "B")]:
    if not len(p_):
        continue
    for (src, tgt), g in p_.groupby(["source", "target"]):
        d = g.z_target - g.z_source
        lad.append(dict(source=src, target=tgt, n=len(g), shift=d.mean(),
                        se=d.std() / np.sqrt(len(g))))
lad = pd.DataFrame(lad).sort_values("shift")
lad["pts_0_100"] = lad["shift"] / CLASS_Z_SD * BENCH_PTS_PER_SD
print(lad.assign(shift=lad["shift"].round(3), se=lad.se.round(3),
                 pts_0_100=lad.pts_0_100.round(1))[
    ["source", "target", "n", "shift", "se", "pts_0_100"]].to_string(index=False))
print("  (positive = the player rates HIGHER in the target competition, i.e. the")
print("   target is the weaker pool. Both directions of a pair should have opposite signs.)")
g = lambda s_, t_: lad[(lad.source == s_) & (lad.target == t_)]["shift"]
try:
    f2n = pd.concat([g("NSW", "NRL"), g("QLD", "NRL")]).mean()
    n2s = float(g("NRL", "SL").iloc[0])
    f2s = pd.concat([g("NSW", "SL"), g("QLD", "SL")]).mean()
    print(f"\n  transitivity check: feeder->NRL ({f2n:+.3f}) + NRL->SL ({n2s:+.3f}) "
          f"= {f2n+n2s:+.3f}, measured feeder->SL = {f2s:+.3f}")
except Exception as e:
    print("  transitivity check unavailable:", e)

lad.to_sql("translation_ladder", con, if_exists="replace", index=False)

allp = pd.concat(pairs, ignore_index=True)
keep = ["player_id", "player", "layer", "source", "target", "season", "z_source",
        "z_target", "g_feed", "g_nrl", "age", "grp", "mins_pg"]
allp[keep].to_sql("translation_pairs", con, if_exists="replace", index=False)
pd.DataFrame([{k: v for k, v in m.items() if k not in ("model", "scaler", "features")}
              for m in out.values()]).to_sql("translation_model_meta", con,
                                             if_exists="replace", index=False)
with open(os.path.join(BASE, "translation_model_v2.pkl"), "wb") as f:
    pickle.dump({"layers": out, "position_group": POS_GROUP,
                 "built": "2026-08-19", "min_games": MIN_GAMES}, f)
con.commit()
print(f"\nwrote translation_model_v2.pkl | translation_pairs: {len(allp)} rows")
con.close()
