# -*- coding: utf-8 -*-
"""The 2027 test of the Into Super League forecast, as fixed in FROZEN_2027_SPEC.md.

The spec says what is tested and why; this maps its rules to the code.

  rule 1     TRAIN_THROUGH is a constant here, never config.json's freeze_season
  rule 3-4   evaluation rows and outcome: `frozen_2026.entry_cohort` for the landing
             season, built from what was known at the end of the season before
  rule 5-7   `fingerprint`, `can_freeze`, `check_frozen`; the manifest keeps every freeze
  rule 8-10  `lines`: S0, B0 and each line's residual quantiles, one function
  rule 11-12 `artefacts`: the lines plus the top-third logistic, stored at the freeze and
             compared at the run
  rule 13-15 `primary`: H1a, then H1b only if H1a is confirmed; the bootstrap and the
             labels; one season, never pooled
  rule 16-19 `secondary`
  rule 21    `--out`

    python frozen_2027.py --context              # 2023-2026, one at a time, as context
    python frozen_2027.py --freeze               # write frozen_2027_manifest.json
    python frozen_2027.py --eval --out dir       # 2027, only against the manifest
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sqlite3
import sys

import numpy as np
import pandas as pd

import explore_post2026 as ex
import frozen_2026 as fz
import rolling_backtest as rb
import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(BASE, "frozen_2027_manifest.json")
CONTEXT = os.path.join(BASE, "FROZEN_2027_CONTEXT.md")
CONFIG = os.path.join(BASE, "config.json")

TRAIN_THROUGH = 2026
EVAL_SEASON = 2027
LAYER = "B_next_season"
MIN_PAIRS = 25
DELTA_MIN = 1.0
MIN_ENTRANTS = 25
N_BOOT = 4000
SEED = 0
RANGE_Q = [10, 25, 75, 90]
SL_SOURCES = ("NRL", "NSW", "QLD")
TOLERANCE = 1e-9

FROZEN_FILES = ("frozen_2027.py", "FROZEN_2027_SPEC.md", "frozen_2026.py",
                "explore_post2026.py", "rolling_backtest.py", "transition_events.py",
                "rating_history.py", "player_rating_engine.py", "sp_schema.py",
                "fit_translation_v3.py", "translation_features.py", "aging.py")

LABELS = ("CONFIRMED", "EVIDENCE OF HARM", "A ONE-POINT GAIN IS NOT SUPPORTED",
          "UNDECIDED", "UNDECIDED (sample below the minimum)")
CLAIMS = {"H1a": "a translation is needed: the forecast beats carrying the rating across",
          "H1b": "the forecast carries information beyond the average arrival"}


# ── rules 8-10: the lines ────────────────────────────────────────────────────────
def lines(pairs):
    """S0's lines with their mean target (B0) and residual quantiles, keyed by direction;
    `_pooled` is the line over every pair, used where a direction has no line of its own."""
    def one(p, basis):
        slope, icept = np.polyfit(p.class_source, p.class_target, 1)
        res = p.class_target - (slope * p.class_source + icept)
        return dict(basis=basis, n=int(len(p)), slope=float(slope), intercept=float(icept),
                    mean=float(p.class_target.mean()),
                    resid_q=[float(v) for v in np.percentile(res, RANGE_Q)])
    out = {"_pooled": one(pairs, "pooled")}
    for pair, g in pairs.groupby("pair"):
        if len(g) >= MIN_PAIRS and g.class_source.std() > 0:
            out[pair] = one(g, "direction")
    return out


def forecast(L, rows):
    """S0, B0, the line's basis and the 50% / 80% ranges for each row."""
    out = []
    for pair, x in zip(rows.pair, rows.class_source.astype(float)):
        ln = L.get(pair, L["_pooled"])
        s0 = float(np.clip(ln["slope"] * x + ln["intercept"], 0, 100))
        q = ln["resid_q"]
        out.append(dict(S0=s0, B0=ln["mean"], basis=ln["basis"],
                        lo80=float(np.clip(s0 + q[0], 0, 100)),
                        lo50=float(np.clip(s0 + q[1], 0, 100)),
                        hi50=float(np.clip(s0 + q[2], 0, 100)),
                        hi80=float(np.clip(s0 + q[3], 0, 100))))
    return pd.DataFrame(out, index=rows.index)


def training_pairs(con, through):
    k = fz.knowledge(con, through)
    return k["pairs"][k["pairs"].layer == LAYER].copy()


# ── rules 11-12: what the freeze stores ──────────────────────────────────────────
def artefacts(con):
    L = lines(training_pairs(con, TRAIN_THROUGH))
    cal = pd.read_sql("SELECT * FROM decision_calibration WHERE kind = 'top_third'", con)
    base = pd.read_sql("SELECT * FROM top_third_base WHERE target = 'SL'", con)
    if cal.empty or base.empty or int(cal.fitted_through.iloc[0]) != TRAIN_THROUGH:
        sys.exit(f"decision_calibration must be built through {TRAIN_THROUGH} "
                 f"(python build_decision_tables.py --write) before the freeze")
    return dict(lines=L, top_third=dict(intercept=float(cal.intercept.iloc[0]),
                                        slope=float(cal.slope.iloc[0]),
                                        n_pairs=int(cal.n_pairs.iloc[0])),
                sl_base_rate=float(base.base_rate.iloc[0]))


def same_lines(a, b):
    """The differences between two sets of lines, as readable strings; empty if none."""
    bad = [f"line {k} missing" for k in set(a) ^ set(b)]
    for k in set(a) & set(b):
        for f in ("n", "slope", "intercept", "mean"):
            if abs(a[k][f] - b[k][f]) > TOLERANCE:
                bad.append(f"{k}.{f}: {a[k][f]} vs {b[k][f]}")
        if np.max(np.abs(np.subtract(a[k]["resid_q"], b[k]["resid_q"]))) > TOLERANCE:
            bad.append(f"{k}.resid_q")
    return bad


# ── rules 5-7: fingerprint and freeze ────────────────────────────────────────────
def config_hash(path=CONFIG):
    """config.json without its evaluation block and without description text."""
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items()
                    if k != "evaluation" and k != "description" and not k.startswith("_")
                    and not k.endswith("_note")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    canon = json.dumps(strip(json.load(open(path))), sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()


def can_freeze(con):
    """A freeze can be taken or retaken only while no season after the cutoff is held."""
    last = con.execute("SELECT max(season) FROM player_match_stats").fetchone()[0]
    return last is not None and int(last) <= TRAIN_THROUGH


def fingerprint(con):
    files = {f: fz._sha(os.path.join(BASE, f)) for f in FROZEN_FILES
             if os.path.exists(os.path.join(BASE, f))}
    pms = pd.read_sql("SELECT * FROM player_match_stats WHERE season <= ?", con,
                      params=(TRAIN_THROUGH,))
    import sklearn
    return dict(files=files, config=config_hash(),
                inputs={"player_match_stats_through": TRAIN_THROUGH,
                        "player_match_stats": fz._frame_sha(pms.astype(str)),
                        "rows": int(len(pms))},
                environment={"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__},
                constants=dict(TRAIN_THROUGH=TRAIN_THROUGH, EVAL_SEASON=EVAL_SEASON,
                               LAYER=LAYER, MIN_PAIRS=MIN_PAIRS, DELTA_MIN=DELTA_MIN,
                               MIN_ENTRANTS=MIN_ENTRANTS, N_BOOT=N_BOOT, SEED=SEED,
                               RANGE_Q=RANGE_Q))


def _commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True,
                             text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", *FROZEN_FILES],
                                    cwd=BASE, capture_output=True, text=True).stdout.strip())
        return sha, dirty
    except OSError:
        return None, None


def freeze(con):
    if not can_freeze(con):
        sys.exit(f"the database holds a season after {TRAIN_THROUGH}: the 2027 freeze can "
                 f"no longer be taken or retaken (FROZEN_2027_SPEC.md, rule 6)")
    sha, dirty = _commit()
    if dirty:
        sys.exit("commit the frozen files first, so the manifest names the commit they "
                 "are in")
    prev = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    fp, art = fingerprint(con), artefacts(con)
    entry = dict(frozen_at=datetime.datetime.now(datetime.timezone.utc)
                 .isoformat(timespec="seconds"), commit=sha,
                 inputs=fp["inputs"]["player_match_stats"])
    json.dump(dict(spec="FROZEN_2027_SPEC.md", commit=sha, fingerprint=fp, artefacts=art,
                   history=prev.get("history", []) + [entry]),
              open(MANIFEST, "w"), indent=1, sort_keys=True)
    print(f"wrote {os.path.basename(MANIFEST)} (commit {sha[:7] if sha else '?'}); "
          f"commit it")


def check_frozen(con):
    if not os.path.exists(MANIFEST):
        sys.exit("frozen_2027_manifest.json is missing: 2027 is scored only against a "
                 "frozen release")
    m = json.load(open(MANIFEST))
    want, have = m["fingerprint"], fingerprint(con)
    bad = [f for f in want["files"] if want["files"][f] != have["files"].get(f)]
    if want["config"] != have["config"]:
        bad.append("config.json (outside the evaluation block)")
    if want["inputs"] != have["inputs"]:
        bad.append("player_match_stats through 2026")
    if want["constants"] != have["constants"]:
        bad.append("constants")
    if bad:
        sys.exit("the release no longer matches the frozen manifest: " + ", ".join(bad)
                 + f". Run from the frozen commit {m['commit']} (spec rule 7).")
    if want["environment"] != have["environment"]:
        print(f"note: environment differs from the freeze: {want['environment']} -> "
              f"{have['environment']}")
    return m


# ── rules 13-15: the primary endpoint ────────────────────────────────────────────
def label(delta, lo, hi, n_players):
    if n_players < MIN_ENTRANTS:
        return LABELS[4]
    if delta >= DELTA_MIN and lo > 0:
        return LABELS[0]
    if hi < 0:
        return LABELS[1]
    if hi < DELTA_MIN:
        return LABELS[2]
    return LABELS[3]


def primary(sl):
    """H1a (S0 against the rating carried across), then H1b (S0 against B0) only if H1a
    is confirmed. Δ = mean(|Y-baseline| - |Y-S0|), positive favouring the forecast."""
    y, ids, n = sl.class_target.to_numpy(float), sl.player_id.to_numpy(), sl.player_id.nunique()
    out = dict(entrants=int(len(sl)), players=int(n),
               mae_S0=float((sl.class_target - sl.S0).abs().mean()))
    for h, base in (("H1a", "class_source"), ("H1b", "B0")):
        d, lo, hi = fz.paired_bootstrap(y, sl[base].to_numpy(float), sl.S0.to_numpy(), ids)
        out[h] = dict(baseline=base, mae_baseline=float((sl.class_target - sl[base]).abs().mean()),
                      delta=d, lo=lo, hi=hi, label=label(d, lo, hi, n))
    if out["H1a"]["label"] != LABELS[0]:
        out["H1b"]["label"] = f"NOT TESTED (H1a not confirmed); descriptive: {out['H1b']['label']}"
    return out


# ── rules 16-20: secondary, descriptive ──────────────────────────────────────────
def wilson(k, n, z=1.96):
    if not n:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def secondary(sl, ev, art=None):
    y = sl.class_target
    out = {}
    for lab, a, b, nom in (("50%", "lo50", "hi50", 0.5), ("80%", "lo80", "hi80", 0.8)):
        k = int(((y >= sl[a]) & (y <= sl[b])).sum())
        lo, hi = wilson(k, len(sl))
        out[f"range_{lab}"] = dict(inside=k, n=int(len(sl)), share=k / len(sl),
                                   ci=[lo, hi], consistent=bool(lo <= nom <= hi))
    rules = [("forecast", "S0"), ("rating", "class_source"), ("appearances", "games_src"),
             ("starts", "starts_src"), ("start_share", "start_share_src"),
             ("minutes", "mins_src")]
    k = max(1, int(round(len(sl) / 3)))
    out["shortlist"] = {}
    for name, col in rules:
        s = sl[col].fillna(-np.inf)
        top = sl.assign(_s=s).sort_values(["_s", "player_id"], ascending=[False, True]).head(k)
        out["shortlist"][name] = dict(shortlisted=k, hits=int(top.top3.sum()),
                                      auc=ex.auc(sl.top3.values, sl[col].values))
    out["base_top3"] = float(sl.top3.mean())
    if art is not None:
        t = art["top_third"]
        p = 1 / (1 + np.exp(-(t["intercept"] + t["slope"] * sl.S0.to_numpy())))
        brier = float(np.mean((p - sl.top3) ** 2))
        b0 = float(np.mean((art["sl_base_rate"] - sl.top3) ** 2))
        out["top_third_probability"] = dict(brier=brier, brier_base=b0,
                                            skill=1 - brier / b0 if b0 else np.nan)
    out["by_source"] = {}
    for src, g in sl.groupby("source"):
        out["by_source"][src] = dict(n=int(len(g)),
                                     mae_S0=float((g.class_target - g.S0).abs().mean()),
                                     mae_B0=float((g.class_target - g.B0).abs().mean()))
    d, lo, hi = fz.paired_bootstrap(ev.class_target.to_numpy(float), ev.B0.to_numpy(),
                                    ev.S0.to_numpy(), ev.player_id.to_numpy())
    out["all_directions"] = dict(n=int(len(ev)), delta=d, lo=lo, hi=hi)
    return out


# ── one season ───────────────────────────────────────────────────────────────────
def evaluate(con, season, L, ps):
    ev = fz.entry_cohort(con, [season])
    fc = forecast(L, ev)
    if fc[["S0", "B0"]].isna().any(axis=None):
        raise RuntimeError("an evaluation row has no forecast: implementation failure")
    ev = pd.concat([ev, fc], axis=1)
    # the cross-check of rule 12's recomputation against the 2026 estimator
    s0_2026, _ = fz.predict_lines(fz.fit_lines(training_pairs(con, season - 1)), ev)
    if np.max(np.abs(s0_2026 - ev.S0.to_numpy())) > 1e-6:
        raise RuntimeError("S0 differs from the 2026 estimator: implementation failure")
    sea = fz.knowledge(con, season)["sea"]
    rated = sea[(sea.n_games >= rb.MIN_GAMES) & (sea.comp == "SL") & (sea.season == season)]
    thr = float(rated.class_score.quantile(2 / 3))
    ev["top3"] = (ev.class_target >= thr).astype(int)
    src = ps.reindex(list(zip(ev.player_id, ev.source, ev.season_src)))
    ev["games_src"], ev["starts_src"] = src.games.values, src.starts.values
    ev["start_share_src"], ev["mins_src"] = src.start_share.values, src.mins.values
    sl = ev[(ev.target == "SL") & ev.source.isin(SL_SOURCES)].copy()
    return ev, sl, thr


def _fmt(season, p, s, thr, context):
    tag = " (context: data the hypotheses were formed on, not a test)" if context else ""
    L = [f"### {season}{tag}", "",
         f"{p['entrants']} entrants into Super League, {p['players']} players; forecast "
         f"MAE {p['mae_S0']:.2f}.", ""]
    for h, name in (("H1a", "the rating carried across"), ("H1b", "the average arrival")):
        r = p[h]
        L.append(f"- {h}, against {name} (MAE {r['mae_baseline']:.2f}): Δ "
                 f"{r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] — **{r['label']}**")
    L += ["",
         f"Ranges: 50% held {s['range_50%']['inside']} of {s['range_50%']['n']} "
         f"({100 * s['range_50%']['share']:.0f}%), 80% held {s['range_80%']['inside']} "
         f"({100 * s['range_80%']['share']:.0f}%). Top-third line {thr:.1f}; "
         f"{100 * s['base_top3']:.0f}% of entrants reached it.", "",
         "| Shortlist by | Hits | AUC |", "|---|---|---|"]
    for name, r in s["shortlist"].items():
        L.append(f"| {name} | {r['hits']} of {r['shortlisted']} | {r['auc']:.2f} |")
    if "top_third_probability" in s:
        t = s["top_third_probability"]
        L += ["", f"Top-third probability: Brier {t['brier']:.3f} against "
                  f"{t['brier_base']:.3f} for the base rate (skill {t['skill']:+.2f})."]
    a = s["all_directions"]
    L += ["", f"Every direction ({a['n']} entrants), forecast against the average "
              f"arrival: Δ {a['delta']:+.2f} [{a['lo']:+.2f}, {a['hi']:+.2f}].", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--context", action="store_true")
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--eval", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    if a.freeze:
        freeze(con)
        return 0

    ps = ex.starts(con)
    if a.context:
        L = ["# The 2027 protocol on the seasons it was formed on", "",
             "Generated by `frozen_2027.py --context`; do not edit by hand. Each season is "
             "forecast from the seasons before it, exactly as 2027 will be. These are the "
             "data the hypotheses were formed on: context, never a test.", ""]
        for season in (2023, 2024, 2025, 2026):
            ev, sl, thr = evaluate(con, season, lines(training_pairs(con, season - 1)), ps)
            L += _fmt(season, primary(sl), secondary(sl, ev), thr, context=True)
        open(CONTEXT, "w").write("\n".join(L))
        print("\n".join(L))
        return 0

    m = check_frozen(con)
    L_now = lines(training_pairs(con, TRAIN_THROUGH))
    bad = same_lines(m["artefacts"]["lines"], L_now)
    if bad:
        sys.exit("the recomputed lines differ from the frozen artefacts: " + "; ".join(bad))
    ev, sl, thr = evaluate(con, EVAL_SEASON, m["artefacts"]["lines"], ps)
    p, s = primary(sl), secondary(sl, ev, m["artefacts"])
    print("\n".join(_fmt(EVAL_SEASON, p, s, thr, context=False)))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        keep = [c for c in ev.columns if c not in ("dob",)]
        ev[keep].to_csv(os.path.join(a.out, f"frozen_2027_rows.csv"), index=False)
        json.dump(dict(primary=p, secondary=s, top_third_line=thr,
                       fingerprint=fingerprint(con), commit=m["commit"]),
                  open(os.path.join(a.out, "frozen_2027_run.json"), "w"), indent=1,
                  default=str)
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
