# -*- coding: utf-8 -*-
"""Measure the get_current_elos fix on 2026 (train 2022-2025, via run_pipeline).

Scenarios per league (nothing is written to the masters or the Sheet):
  A  master as stored (what the app runs on today)
  A2 stored R1-17, R18+R19 re-appended as they were submitted (one batch) with the FIX
     (NRL only: isolates the results the master actually lost)
  B  2026 replayed through append_new_round one round per submit, FIXED function
  C  same replay with the OLD function (what a season of weekly submits produces)
Usage: python scripts/measure_weekly_elo_fix.py <repo_dir> <out_dir>
"""
import os
import sys
import json
import tempfile
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
import xladder_pipeline as xp  # noqa: E402

FIXED = xp.get_current_elos


def old_get_current_elos(df):
    elos = {}
    for _, r in df.sort_values(["Season", "Round", "Match ID"]).iterrows():
        elos[r["A Team"]] = float(r["ELO_A"])
        elos[r["B Team"]] = float(r["ELO_B"])
    return elos


def loaded(raw, league):
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as t:
        raw.to_excel(t.name, index=False)
        p = t.name
    try:
        return xp.load_data(p, league)
    finally:
        os.unlink(p)


def replay(base, rows_by_batch, league, fn):
    xp.get_current_elos = fn
    try:
        df = base.copy()
        for rows in rows_by_batch:
            df = xp.append_new_round(df, rows.copy(), league)
        return df
    finally:
        xp.get_current_elos = FIXED


def predict(df, league):
    keep = [c for c in df.columns if c not in
            ("League", "Home_flag", "Played", "Margin", "Total", "A_Win", "A_Name", "B_Name")]
    out = xp.run_pipeline(None, league, target_season=2026, raw_df=df[keep])
    d = out["df"]
    d = d[(d["Season"] == 2026) & d["Played"]].set_index("Match ID").sort_index()
    res = pd.DataFrame({"Diff ELO": d["Diff ELO"]})
    for v in ("v1", "v2"):
        res[f"hit_{v}"] = (d[f"WL_Pred_{v}"] == d["A_Win"]).astype(int)
        res[f"ae_{v}"] = (d[f"Margin_Pred_{v}"] - d["Margin"]).abs()
        res[f"winner_{v}"] = d[f"WL_Pred_{v}"]
    return res


def boot(new, old, n=10000, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(new), size=(n, len(new)))
    out = {}
    for v, name in (("v1", "M3"), ("v2", "M3+")):
        dh = (new[f"hit_{v}"] - old[f"hit_{v}"]).values
        da = (new[f"ae_{v}"] - old[f"ae_{v}"]).values
        bh = dh[idx].mean(1) * 100
        ba = da[idx].mean(1)
        out[name] = {
            "acc_old": round(old[f"hit_{v}"].mean() * 100, 1),
            "acc_new": round(new[f"hit_{v}"].mean() * 100, 1),
            "d_acc_pp": round(dh.mean() * 100, 1),
            "d_acc_ci": [round(x, 1) for x in np.percentile(bh, [2.5, 97.5])],
            "mae_old": round(old[f"ae_{v}"].mean(), 2),
            "mae_new": round(new[f"ae_{v}"].mean(), 2),
            "d_mae": round(da.mean(), 2),
            "d_mae_ci": [round(x, 2) for x in np.percentile(ba, [2.5, 97.5])],
            "winner_flips": int((new[f"winner_{v}"] != old[f"winner_{v}"]).sum()),
        }
    de = (new["Diff ELO"] - old["Diff ELO"]).abs()
    out["diff_elo_change"] = {"mean_abs": round(de.mean(), 1), "max_abs": round(de.max(), 1),
                              "matches_changed": int((de > 1e-6).sum()), "n": len(de)}
    return out


def frozen_steps(df):
    """Club-steps inside 2026 whose next pre-match ELO equals the previous pre-match ELO."""
    a = df[["Season", "Round", "Match ID", "A Team", "ELO_A"]].set_axis(["S", "R", "M", "T", "E"], axis=1)
    b = df[["Season", "Round", "Match ID", "B Team", "ELO_B"]].set_axis(["S", "R", "M", "T", "E"], axis=1)
    L = pd.concat([a, b]).sort_values(["T", "S", "R", "M"])
    L["nE"] = L.groupby("T")["E"].shift(-1)
    L["nS"] = L.groupby("T")["S"].shift(-1)
    s = L[(L.S == 2026) & (L.nS == 2026)]
    return int(np.isclose(s.E, s.nE, atol=1e-9).sum()), len(s)


report = {}
for league, fname in (("NRL", "NRL_master.xlsx"), ("SL", "SL_master.xlsx")):
    raw = pd.read_excel(os.path.join(REPO, fname))
    full = loaded(raw, league)                       # SL: code fix + ELO recompute
    s = full["Season"]
    base = full[s <= 2025].copy()
    rnds = sorted(full.loc[s == 2026, "Round"].unique())
    per_round = [full[(s == 2026) & (full["Round"] == r)].copy() for r in rnds]

    B = replay(base, per_round, league, FIXED)
    C = replay(base, per_round, league, old_get_current_elos)
    B_one = replay(base, [full[s == 2026].copy()], league, FIXED)
    r = {"rounds_2026": [int(rnds[0]), int(rnds[-1])],
         "frozen_steps": {"A": frozen_steps(full), "B": frozen_steps(B), "C": frozen_steps(C)},
         "fixed_batch_invariant_maxdiff": float(np.abs(
             B.set_index("Match ID")["Diff ELO"] - B_one.set_index("Match ID")["Diff ELO"]).max())}

    # 2026 chain under the fix = recompute of 2026 seeded from post-match 2025-end ratings
    seed = {t: e + (xp.ELO_MEAN - e) * xp.ELO_REGRESSION for t, e in FIXED(base).items()}
    chk, _ = xp.update_elos_for_new_matches(full[s == 2026], seed, 2026)
    r["B_equals_seeded_recompute_maxdiff"] = float(np.abs(
        chk.set_index("Match ID")["Diff ELO"] - B.set_index("Match ID")["Diff ELO"]).max())

    pA, pB, pC = predict(full, league), predict(B, league), predict(C, league)
    r["B_vs_A"] = boot(pB, pA)
    r["B_vs_C"] = boot(pB, pC)

    if league == "NRL":
        upto17 = full[(s <= 2025) | ((s == 2026) & (full["Round"] <= 17))].copy()
        late = full[(s == 2026) & (full["Round"] >= 18)].copy()
        repro = replay(upto17, [late], league, old_get_current_elos)
        stored = full.set_index("Match ID").loc[late["Match ID"], ["ELO_A", "ELO_B"]]
        got = repro.set_index("Match ID").loc[late["Match ID"], ["ELO_A", "ELO_B"]]
        r["old_code_reproduces_stored_R18_R19_maxdiff"] = float(np.abs(stored.values - got.values).max())
        A2 = replay(upto17, [late], league, FIXED)
        r["frozen_steps"]["A2"] = frozen_steps(A2)
        r["A2_vs_A"] = boot(predict(A2, league), pA)
    report[league] = r
    print(league, json.dumps(r, indent=1), flush=True)

with open(os.path.join(OUT, "elo_fix_measure.json"), "w") as f:
    json.dump(report, f, indent=1)
