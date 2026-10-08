# -*- coding: utf-8 -*-
"""The 2026 translation test, as fixed in FROZEN_2026_SPEC.md (adopted 2026-10-08).

The specification was proposed by the seventh external review after it found that the
first re-test (`retest_r6.py`) did not do what it said: its "line over all training
moves" fallback was identified by the small directions alone. This module implements the
specification and nothing else. Read the spec for the reasons; this docstring maps its
rules to the code.

  rule 2-5   prediction time and rows: `knowledge(season)` holds only what was known at the
             end of a season; evaluation rows are the explicit entry cohort
             (`rolling_backtest.moves_into`, first/returning entries, source fixed from the
             season before, three or more source and target appearances); every feature
             is computed relative to each row's own source season.
  rule 6     S0, the shipped construction: next-season pairs (`build_pairs`) ending by the
             training cutoff; one pooled OLS line on ALL pairs and an own line for each
             direction with 25+ pairs and a source that varies; otherwise pooled.
  rule 7     S1: S0 plus `rated_source_prev_year`, fitted jointly on the same rows, pooled
             and own models fitted separately, the feature standardised within each fitting
             sample (ddof=1, a constant feature zeroed), penalty 1.0 on its coefficient.
  rule 8     C0/C1: the same two models fitted on entry-cohort rows instead of pairs.
  rule 9     H2: C1 plus the prior-season minutes-weighted rating of the destination club's
             players in his position group, with mean imputation and an unstandardised
             missing flag, both penalised. Retrospective: the club is the first one he is
             seen playing for in the target season.
  rule 10    one diagnostic: S0 plus the source rating's shrinkage_B.
  rule 11-15 H1 is the only primary comparison: delta = mean(|Y-S0| - |Y-S1|) over the
             evaluation season's entries into Super League from NRL/NSW/QLD; a paired
             player-cluster percentile bootstrap (4,000 draws, seed 0, sorted canonical
             ids, models not refitted); the decision labels of rule 12-13; one season at a
             time, never pooled.
  rule 1,16  `--freeze` records the hashes of the code, config, environment and the
             through-2025 inputs; an evaluation of a season after the freeze season refuses
             to run unless those hashes still match. Missing forecasts stop the run.

Seasons up to the freeze season can be scored as historical context. They are the data
the hypotheses were found on, so their results are context, never a test.

    python frozen_2026.py --context                  # 2023, 2024, 2025, one at a time
    python frozen_2026.py --eval-season 2025 --out d # one season, with the row export
    python frozen_2026.py --freeze                   # write frozen_2026_manifest.json
    python frozen_2026.py --eval-season 2026         # only once frozen, hashes matching
"""
import argparse
import hashlib
import json
import os
import platform
import sqlite3
import sys

import numpy as np
import pandas as pd

import player_rating_engine as pre
import rating_history as rh
import rolling_backtest as rb
import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(BASE, "frozen_2026_manifest.json")
FREEZE_SEASON = pre.FREEZE_SEASON or 2025

MIN_PAIRS = 25
ALPHA = 1.0
FIRST_LANDING = 2021
SL_SOURCES = ("NRL", "NSW", "QLD")
# rule 12
DELTA_MIN = 1.0
MIN_ENTRANTS = 25
MIN_PER_HISTORY_GROUP = 5
N_BOOT = 4000
SEED = 0

FROZEN_FILES = ("frozen_2026.py", "rolling_backtest.py", "transition_events.py",
                "rating_history.py", "player_rating_engine.py", "sp_schema.py",
                "fit_translation_v3.py", "translation_features.py", "aging.py",
                "config.json", "FROZEN_2026_SPEC.md")


# ── the estimator (rules 6-10) ───────────────────────────────────────────────────
def _columns(rows, extras, stats=None):
    """Design for one fitting sample: intercept, source, then each extra.

    `extras` is a list of (column, kind): "binary" is standardised (ddof=1) within the
    fitting sample, a constant one zeroed; "numeric" is mean-imputed and standardised with
    an unstandardised missing flag beside it. `stats` carries a fitted sample's own means
    and SDs to a new frame.
    """
    fit = stats is None
    stats = {} if fit else stats
    X = [np.ones(len(rows)), rows["class_source"].to_numpy(float)]
    for col, kind in extras:
        v = rows[col].astype(float)
        if fit:
            mu = float(v.mean()) if v.notna().any() else 0.0
            sd = float(v.std(ddof=1)) if v.notna().sum() > 1 else 0.0
            stats[col] = (mu, sd if np.isfinite(sd) and sd > 0 else None)
        mu, sd = stats[col]
        z = ((v.fillna(mu) - mu) / sd).to_numpy() if sd else np.zeros(len(rows))
        X.append(z)
        if kind == "numeric":
            X.append(v.isna().to_numpy(float))
    return np.column_stack(X), stats


def fit_one(rows, extras=()):
    """Least squares, intercept and source unpenalised, ALPHA on every added coefficient."""
    X, stats = _columns(rows, extras)
    k = X.shape[1] - 2
    y = rows["class_target"].to_numpy(float)
    if k:
        X = np.vstack([X, np.hstack([np.zeros((k, 2)), np.sqrt(ALPHA) * np.eye(k)])])
        y = np.concatenate([y, np.zeros(k)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(coef=coef, stats=stats, extras=tuple(extras), n=len(rows))


def fit_lines(train, extras=()):
    """One pooled model on every row, and an own model per qualifying direction."""
    own = {p: fit_one(g, extras) for p, g in train.groupby("pair")
           if len(g) >= MIN_PAIRS and g["class_source"].std() > 0}
    return dict(pooled=fit_one(train, extras), own=own)


def predict_lines(model, rows):
    out = np.full(len(rows), np.nan)
    basis = np.empty(len(rows), dtype=object)
    for pair, idx in rows.groupby("pair").indices.items():
        m, b = (model["own"][pair], "direction") if pair in model["own"] else (model["pooled"], "pooled")
        X, _ = _columns(rows.iloc[idx], m["extras"], m["stats"])
        out[idx] = np.clip(X @ m["coef"], 0, 100)
        basis[idx] = b
    return out, basis


# ── what was known when (rules 2-5) ──────────────────────────────────────────────
_KNOWN = {}


def knowledge(con, season):
    """Pairs, ratings and positions as held at the end of `season`, cached per run."""
    if season not in _KNOWN:
        pairs, cum, ext, pos, dob = rb.knowledge_at(con, season)
        _, sea, _ = rh.all_competitions(rb.DB, through=season)
        sea = sea.copy()
        sea["player_id"] = sp.normalize_player_id(sea.player_id)
        _KNOWN[season] = dict(pairs=pairs, cum=cum, ext=ext, pos=pos, dob=dob, sea=sea)
    return _KNOWN[season]


def _covered(con):
    rows = con.execute("SELECT DISTINCT competition, season FROM player_match_stats").fetchall()
    return {(c, int(s)) for c, s in rows}


def add_history(rows, sea, covered):
    """rated_source_prev_year: a season-only source rating with n >= 3 the season before
    the row's own source season. prev_year_covered says whether the data hold that season
    at all, so a 0 caused by the start of the data can be told from a 0 earned."""
    r = rows.copy()
    r["player_id"] = sp.normalize_player_id(r.player_id)
    s = sea[sea.n_games >= rb.MIN_GAMES]
    have = set(zip(s.player_id, s.comp, s.season))
    prev = r.season_src - 1
    r["rated_source_prev_year"] = [float((p, c, y) in have)
                                   for p, c, y in zip(r.player_id, r.source, prev)]
    r["prev_year_covered"] = [(c, int(y)) in covered for c, y in zip(r.source, prev)]
    return r


def add_incumbents(rows, sea, pos, pm):
    """H2's covariate: minutes-weighted season-only rating, the season before the target
    season, of the destination club's players in his position group, him excluded. The
    destination is the first club he is seen playing for in the target season."""
    r = rows.copy()
    grp = lambda pid, comp: sp.POSITION_GROUP.get(pos.get((pid, comp)))
    first = (pm.sort_values("round").groupby(["player_id", "comp", "season"]).team.first())
    r["new_club_first_seen"] = first.reindex(list(zip(r.player_id, r.target, r.season_tgt))).values
    r["group"] = [grp(p, c) for p, c in zip(r.player_id, r.source)]
    g = pm.assign(grp=[grp(p, c) for p, c in zip(pm.player_id, pm.comp)])
    mins = g.groupby(["comp", "season", "team", "grp", "player_id"]).minutes.sum()
    keys = set(mins.index.droplevel(4))
    s = sea[sea.n_games >= rb.MIN_GAMES].set_index(["player_id", "comp", "season"]).class_score
    vals = []
    for row in r.itertuples():
        k = (row.target, row.season_tgt - 1, row.new_club_first_seen, row.group)
        if pd.isna(row.new_club_first_seen) or row.group is None or k not in keys:
            vals.append(np.nan)
            continue
        w = mins.loc[k]
        w = w[(w.index != row.player_id) & (w.values > 0)]
        rat = np.array([s.get((q, row.target, row.season_tgt - 1), np.nan) for q in w.index])
        ok = ~np.isnan(rat)
        vals.append(float((rat[ok] * w.values[ok]).sum() / w.values[ok].sum()) if ok.any() else np.nan)
    r["club_position_prev_rating"] = vals
    return r


def entry_cohort(con, landings):
    """The explicit entry cohort, each landing season built from what was known before it."""
    out = []
    for o in landings:
        k = knowledge(con, o - 1)
        e = rb.moves_into(con, o, k["cum"], k["ext"], k["pos"], k["dob"])
        if not e.empty:
            out.append(e)
    c = pd.concat(out, ignore_index=True)
    c["player_id"] = sp.normalize_player_id(c.player_id)
    c["pair"] = c.source + "->" + c.target
    return c


# ── the endpoint (rules 11-15) ───────────────────────────────────────────────────
def paired_bootstrap(y, a, b, players):
    """mean(|y-a| - |y-b|) and its 95% player-cluster percentile interval."""
    diff = np.abs(y - a) - np.abs(y - b)
    ids = np.array(sorted(pd.unique(players)))
    at = {p: np.flatnonzero(players == p) for p in ids}
    rng = np.random.default_rng(SEED)
    s = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s[i] = np.concatenate([diff[at[p]] for p in rng.choice(ids, ids.size, replace=True)]).mean()
    return float(diff.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def decide(delta, lo, hi, n_players, n_without, n_with):
    if n_players < MIN_ENTRANTS or min(n_without, n_with) < MIN_PER_HISTORY_GROUP:
        return "UNDECIDED (sample below the minimum)"
    if delta >= DELTA_MIN and lo > 0:
        return "POSITIVE EXPLORATORY SIGNAL"
    if hi < 0:
        return "EVIDENCE OF HARM"
    if hi < DELTA_MIN:
        return "A ONE-POINT GAIN IS NOT SUPPORTED"
    return "UNDECIDED"


# ── provenance (rules 1, 16) ─────────────────────────────────────────────────────
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _frame_sha(df):
    df = df.sort_values(list(df.columns)).reset_index(drop=True)
    return hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def fingerprint(con, through=FREEZE_SEASON):
    files = {f: _sha(os.path.join(BASE, f)) for f in FROZEN_FILES if os.path.exists(os.path.join(BASE, f))}
    pms = pd.read_sql("SELECT * FROM player_match_stats WHERE season <= ?", con, params=(through,))
    players = pd.read_sql("SELECT player_id, dob FROM players", con)
    import sklearn
    return dict(files=files,
                inputs={"player_match_stats_through": through,
                        "player_match_stats": _frame_sha(pms.astype(str)),
                        "players_dob": _frame_sha(players.astype(str))},
                environment={"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__},
                constants=dict(MIN_PAIRS=MIN_PAIRS, ALPHA=ALPHA, DELTA_MIN=DELTA_MIN,
                               MIN_ENTRANTS=MIN_ENTRANTS,
                               MIN_PER_HISTORY_GROUP=MIN_PER_HISTORY_GROUP,
                               N_BOOT=N_BOOT, SEED=SEED, FREEZE_SEASON=FREEZE_SEASON))


def check_frozen(con):
    if not os.path.exists(MANIFEST):
        sys.exit(f"{os.path.basename(MANIFEST)} is missing: a season after {FREEZE_SEASON} is "
                 f"only scored against a frozen release (python frozen_2026.py --freeze, then "
                 f"commit the manifest)")
    want = json.load(open(MANIFEST))["fingerprint"]
    have = fingerprint(con)
    bad = [k for k in want["files"] if want["files"][k] != have["files"].get(k)]
    bad += [k for k in want["inputs"] if want["inputs"][k] != have["inputs"].get(k)]
    if want["constants"] != have["constants"]:
        bad.append("constants")
    if bad:
        sys.exit("the release no longer matches the frozen manifest: " + ", ".join(bad))


# ── one evaluation season ────────────────────────────────────────────────────────
def evaluate(con, eval_season, pm, covered):
    train_through = eval_season - 1
    k = knowledge(con, train_through)
    pairs = k["pairs"][k["pairs"].layer == "B_next_season"].copy()
    pairs = add_history(pairs, k["sea"], covered)
    cohort_train = add_history(entry_cohort(con, range(FIRST_LANDING, eval_season)),
                               k["sea"], covered)
    cohort_train = add_incumbents(cohort_train, k["sea"], k["pos"], pm)
    ev = entry_cohort(con, [eval_season])
    ev = add_incumbents(add_history(ev, k["sea"], covered), k["sea"], k["pos"], pm)

    hist = [("rated_source_prev_year", "binary")]
    models = {
        "S0": (pairs, []),
        "S1": (pairs, hist),
        "C0": (cohort_train, []),
        "C1": (cohort_train, hist),
        "H2": (cohort_train, hist + [("club_position_prev_rating", "numeric")]),
        "S0+B": (pairs, [("shrinkage_B", "numeric")]),
    }
    for name, (train, extras) in models.items():
        ev[name], ev[f"{name}_basis"] = predict_lines(fit_lines(train, extras), ev)
    missing = int(ev[list(models)].isna().any(axis=1).sum())
    if missing:
        raise RuntimeError(f"{missing} evaluation rows have no forecast: implementation failure")
    return ev, len(pairs), len(cohort_train)


def report(ev, eval_season, n_pairs, n_cohort, context):
    sl = ev[(ev.target == "SL") & ev.source.isin(SL_SOURCES)]
    y, ids = sl.class_target.to_numpy(float), sl.player_id.to_numpy()
    n_with = int((sl.rated_source_prev_year == 1).sum())
    n_without = int((sl.rated_source_prev_year == 0).sum())
    d, lo, hi = paired_bootstrap(y, sl.S0.to_numpy(), sl.S1.to_numpy(), ids)
    label = decide(d, lo, hi, sl.player_id.nunique(), n_without, n_with)
    mae = lambda g, c: float((g.class_target - g[c]).abs().mean())
    print(f"\n=== evaluation season {eval_season} (fitted through {eval_season - 1}) "
          f"{'- HISTORICAL CONTEXT, discovery data, not a test' if context else ''}")
    print(f"training: {n_pairs} next-season pairs (S0/S1), {n_cohort} entry-cohort rows (C0/C1/H2)")
    print(f"H1, entries into SL from NRL/NSW/QLD: {len(sl)} rows, {sl.player_id.nunique()} players, "
          f"{n_without} without and {n_with} with a rated previous source season "
          f"({int((~sl.prev_year_covered).sum())} of the 'without' are seasons the data do not cover)")
    print(f"  MAE S0 {mae(sl, 'S0'):.2f}   S1 {mae(sl, 'S1'):.2f}   "
          f"delta {d:+.2f} [{lo:+.2f}, {hi:+.2f}]   ->  {label}"
          + ("   (context only)" if context else ""))
    print(f"  S0 basis: {dict(pd.Series(sl.S0_basis).value_counts())}")
    print("secondary, descriptive (no multiplicity control, cannot rescue H1):")
    for a, b, rows, lab in (("S0", "S0+B", sl, "S0 + shrinkage_B vs S0, into SL"),
                            ("C0", "C1", sl, "C1 vs C0, into SL"),
                            ("C1", "H2", ev, "H2 vs C1, all entries (retrospective club)")):
        dd = paired_bootstrap(rows.class_target.to_numpy(float), rows[a].to_numpy(),
                              rows[b].to_numpy(), rows.player_id.to_numpy())
        print(f"  {lab:46s} MAE {mae(rows, a):.2f} -> {mae(rows, b):.2f}  "
              f"delta {dd[0]:+.2f} [{dd[1]:+.2f}, {dd[2]:+.2f}]")
    return dict(season=eval_season, rows=len(sl), players=int(sl.player_id.nunique()),
                without=n_without, with_=n_with, mae_S0=mae(sl, "S0"), mae_S1=mae(sl, "S1"),
                delta=d, lo=lo, hi=hi, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-season", type=int)
    ap.add_argument("--context", action="store_true",
                    help="score 2023, 2024 and 2025 one at a time, as historical context")
    ap.add_argument("--freeze", action="store_true", help="write the frozen manifest")
    ap.add_argument("--out", help="folder for the row-level export")
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)

    if a.freeze:
        json.dump(dict(spec="FROZEN_2026_SPEC.md", fingerprint=fingerprint(con)),
                  open(MANIFEST, "w"), indent=1, sort_keys=True)
        print(f"wrote {os.path.basename(MANIFEST)}; commit it with the code it fingerprints")
        return 0
    seasons = [2023, 2024, 2025] if a.context else [a.eval_season]
    if not seasons or seasons == [None]:
        ap.error("give --eval-season or --context")
    if max(seasons) > FREEZE_SEASON:
        check_frozen(con)

    pms = pd.read_sql("SELECT player_id, competition comp, season, round, team, minutes "
                      "FROM player_match_stats", con)
    pms["player_id"] = sp.normalize_player_id(pms.player_id)
    covered = _covered(con)
    out = []
    for season in seasons:
        pm = pms[pms.season <= season]
        ev, n_pairs, n_cohort = evaluate(con, season, pm, covered)
        out.append(report(ev, season, n_pairs, n_cohort, context=season <= FREEZE_SEASON))
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            keep = [c for c in ev.columns if c not in ("name", "dob")]
            ev[keep].to_csv(os.path.join(a.out, f"frozen_eval_{season}.csv"), index=False)
    if a.out:
        json.dump(dict(results=out, fingerprint=fingerprint(con)),
                  open(os.path.join(a.out, "frozen_run.json"), "w"), indent=1, default=str)
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
