# -*- coding: utf-8 -*-
"""Build a self-contained package for an outside methodological review.

A reviewer who can only read prose can only argue with prose. The first external review
of this project (2026-08-28) found five real faults, and every one of them was a fault in
the *code* that the reports described incorrectly — a feature builder that silently
dropped three position groups, a validation section that scored a model on its own
training data, hardcoded statistics presented as if read from artefacts. None of those
were visible in the write-up. They were visible in the files.

So this exports the files, and with them enough data to recompute the central claims
from scratch without the 89 MB database: the per-origin training and evaluation frames
of the rolling backtest, the season-level ratings, and the full player-match feed
gzipped. A reviewer can refit, rescore, and disagree with a number rather than with a
sentence about a number.

Nothing here is a summary. Every CSV is dumped from the same functions that produced the
reports, so if the export and the report disagree, that is itself a finding.

    python export_review_package.py
    python export_review_package.py --out ../TALLEC_review_package --origins 2024,2025
"""
import argparse
import gzip
import hashlib
import io
import os
import shutil
import sqlite3
import subprocess
import sys
import time

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")

# The files that carry the methodology. A reviewer asked to find errors needs the code
# that computes, not the code that ingests or draws — ingest and UI are listed separately
# so the package says what it is leaving out rather than leaving it out silently.
CODE_CORE = [
    "player_rating_engine.py",     # Class/Form, shrinkage, season weighting
    "translation_features.py",     # the feature builder that was silently wrong
    "fit_translation_v3.py",       # the conditional model now in front
    "fit_translation_v2.py",       # the sealed model the 2026 holdout belongs to
    "rolling_backtest.py",         # the central negative result
    "transition_events.py",        # the explicit entry cohort it is scored on
    "arrival_model.py",            # does he get there at all - and why it is not shipped
    "ablate_translation.py",       # which features earn their place, measured
    "weights_for_middles_edge.py", # how the Middles and Edge weightings were chosen
    "variance_bias.py",            # what the constant-ability assumption costs, measured
    "cohorts.py",                  # retired from the backtest; still used by the trace
    "aging.py",
    "sp_schema.py",                # position groups, date-of-birth parsing
    "predict_translation.py",      # what the app actually calls: the line into SL
    "noise_floor.py",              # how much of the error is noise in the target
    "team_role_trend.py",          # team context, expected role and trend, tested
    "retest_r6.py",                # the narrow re-test on one cohort, history baseline
    "position_metrics.py",         # the client's per-position metric set
    "metric_spec.py",
    "runtime.py",                  # guarded writes, config hashing
    "rating_history.py",
    "record_v1_holdout.py",
    "build_manifest.py",
    "config.json",
]
CODE_CONTEXT = [
    "regenerate_full.py", "validate_ratings.py", "report_point0.py", "report_freeze.py",
    "trace_cohort.py", "teamlist_backtest.py", "validate_positions.py", "gigot_v2.py",
    "gigot_contribution.py", "datastate.py", "smoke_bosc.py",
    "bosc_app.py", "build_app_db.py", "team_map.py",
]
DOCS = [
    "README.md", "HANDOVER.md", "POINT0_REPORT.md", "ROLLING_REPORT.md",
    "VALIDATION_REPORT.md", "FREEZE_REPORT.md", "TRACE_REPORT.md",
    "COMPETITION_TRANSLATION_SPEC.md", "GIGOT_V2_SPEC.md", "AUS_DATA.md",
    "ABLATION_REPORT.md",          # the feature decision, with its intervals
    "ARRIVAL_REPORT.md",           # the arrival model and its negative result
    "VARIANCE_BIAS_REPORT.md",     # what the constant-ability assumption costs
    "WEIGHTS_REPORT.md",           # the Middles/Edge weight comparison
    "MANIFEST.json", "v1_holdout_record.json",
    "what_is_live.html",           # the status note whose claims are under review
]
TESTS_DIR = "tests"
# the hand-written files at the package root live here, so they are versioned with the
# code they describe instead of only inside a zip
ROOT_DIR = os.path.join(BASE, "review_package")
ROOT_FILES = ("00_START_HERE.md", "PROMPT.md", "reproduce.py")
# analyses re-run for the package: script, the data folder it writes, the results file
ANALYSES = (("noise_floor.py", "data/noise_floor", "results/noise_floor.txt"),
            ("team_role_trend.py", "data/team_role_trend", "results/team_role_trend.txt"),
            ("retest_r6.py", "data/retest_r6", "results/retest_r6.txt"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def dump(df, path, gz=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if gz:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8")
    return len(df)


def rolling_frames(out, origins):
    """Re-run the rolling backtest, keeping the frames instead of the write-up.

    Imported rather than reimplemented: if these functions change, the export changes
    with them, and a reviewer comparing this against ROLLING_REPORT.md is comparing two
    outputs of the same code rather than two transcriptions.
    """
    import rolling_backtest as rb

    con = sqlite3.connect(DB)
    rows = []
    for origin in origins:
        print(f"  origin {origin}: rebuilding knowledge as of {origin - 1} ...",
              flush=True)
        pairs, cum, ext, pos, dob = rb.knowledge_at(con, origin - 1)
        f = rb.fit(pairs, "B_next_season")
        if f is None:
            print(f"    too few pairs ({len(pairs)}) — skipped")
            continue
        train = pairs[pairs.layer == "B_next_season"] if "layer" in pairs else pairs
        n_tr = dump(train, os.path.join(out, "data", f"rolling_train_{origin}.csv"))

        pend = rb.moves_into(con, origin, cum, ext, pos, dob)
        if pend.empty:
            print("    no moves landing in this season")
            continue
        ev = rb.predict(f, pend)
        ev["origin"] = origin
        # the cohort travels with the row now; asking cohorts.classify again would be
        # the export disagreeing with the backtest about who is in which group
        ev["cohort"] = ev["transition_type"]
        n_ev = dump(ev, os.path.join(out, "data", f"rolling_eval_{origin}.csv"))
        rows.append(dict(origin=origin, train_rows=n_tr, train_pairs=f["n"],
                         train_players=f["players"], inner_rmse=round(f["inner_rmse"], 4),
                         eval_rows=n_ev))
        print(f"    train {n_tr} rows, eval {n_ev} rows")
    con.close()
    return pd.DataFrame(rows)


def tables(out):
    """Everything else a reviewer needs, straight out of the database."""
    con = sqlite3.connect(DB)
    got = []
    small = {
        "player_ratings": "SELECT * FROM player_ratings",
        "players": "SELECT * FROM players",
        "translation_ladder_v3": "SELECT * FROM translation_ladder_v3",
        "translation_pairs_v3": "SELECT * FROM translation_pairs_v3",
        "translation_model_v3_meta": "SELECT * FROM translation_model_v3_meta",
        # the explicit entry cohort, including the men who never arrived
        "transition_events": "SELECT * FROM transition_events",
        "player_position_category": "SELECT * FROM player_position_category",
    }
    for name, q in small.items():
        try:
            n = dump(pd.read_sql(q, con), os.path.join(out, "data", f"{name}.csv"))
            got.append((f"data/{name}.csv", n))
        except Exception as e:
            print(f"  {name}: skipped ({e})")

    # the raw feed, gzipped — this is what lets a reviewer rebuild the ratings from
    # nothing and check whether the engine does what the engine says it does
    s = pd.read_sql("SELECT * FROM player_match_stats", con)
    n = dump(s, os.path.join(out, "data", "player_match_stats.csv.gz"), gz=True)
    got.append(("data/player_match_stats.csv.gz", n))

    # season-level ratings as the backtest sees them, not as the app publishes them
    try:
        import rating_history as rh
        _, sea, _ = rh.all_competitions(DB, through=2025)
        n = dump(sea, os.path.join(out, "data", "player_season_ratings.csv"))
        got.append(("data/player_season_ratings.csv", n))
    except Exception as e:
        print(f"  season ratings: skipped ({e})")
    con.close()
    return got


def copy_code(out):
    got = []
    for n in ROOT_FILES:
        src = os.path.join(ROOT_DIR, n)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, n))
        else:
            print(f"  {n}: missing from review_package/, not copied")
    for group, names in (("code", CODE_CORE), ("code_context", CODE_CONTEXT),
                         ("docs", DOCS)):
        for n in names:
            src = os.path.join(BASE, n)
            if not os.path.exists(src):
                print(f"  {n}: missing, not copied")
                continue
            dst = os.path.join(out, group, n)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            got.append((f"{group}/{n}", os.path.getsize(src)))
    t = os.path.join(BASE, TESTS_DIR)
    if os.path.isdir(t):
        dst = os.path.join(out, "tests")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(t, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for f in sorted(os.listdir(dst)):
            if f.endswith(".py"):
                got.append((f"tests/{f}", os.path.getsize(os.path.join(dst, f))))
    return got


def check(out):
    """Report which packaged files no longer match the repo they were copied from.

    The package has gone stale four times, always the same way: work lands after the
    export and nobody re-runs it, so an outside reviewer spends their time on code that
    has since been fixed. Twice that produced findings we had already addressed.

    This makes it a command rather than a habit. It compares only the copied files —
    the data extracts are regenerated wholesale and have no repo counterpart — and says
    nothing about whether the export is *complete*, which is a separate way to be wrong.
    """
    if not os.path.isdir(out):
        print(f"no package at {out}; run the exporter first")
        return 1
    stale, missing, checked = [], [], 0
    for n in ROOT_FILES:
        src, dst = os.path.join(ROOT_DIR, n), os.path.join(out, n)
        if not os.path.exists(src):
            continue
        if not os.path.exists(dst):
            missing.append(n)
            continue
        checked += 1
        if sha(src) != sha(dst):
            stale.append(n)
    for group, names in (("code", CODE_CORE), ("code_context", CODE_CONTEXT),
                         ("docs", DOCS)):
        for n in names:
            src, dst = os.path.join(BASE, n), os.path.join(out, group, n)
            if not os.path.exists(src):
                continue
            if not os.path.exists(dst):
                missing.append(f"{group}/{n}")
                continue
            checked += 1
            if sha(src) != sha(dst):
                stale.append(f"{group}/{n}")
    t = os.path.join(BASE, TESTS_DIR)
    if os.path.isdir(t):
        for n in sorted(os.listdir(t)):
            if not n.endswith(".py"):
                continue
            dst = os.path.join(out, "tests", n)
            if not os.path.exists(dst):
                missing.append(f"tests/{n}")
                continue
            checked += 1
            if sha(os.path.join(t, n)) != sha(dst):
                stale.append(f"tests/{n}")

    print(f"{checked} packaged files compared against the repo")
    for f in stale:
        print(f"  STALE   {f}")
    for f in missing:
        print(f"  MISSING {f}")
    if not stale and not missing:
        print("  the package matches this checkout")
        return 0
    print("\nRun `python export_review_package.py` before sending it to anyone.")
    return 1


def analyses(out):
    """Re-run the analyses behind the status note, keeping their frames and output.

    Each writes the frames it computed from into its data folder, so `reproduce.py`
    can recompute the claims from them without the database, and its printed result
    goes to results/ as the reference those recomputations are checked against.
    """
    got = []
    for script, data_dir, result in ANALYSES:
        print(f"  {script} ...", flush=True)
        r = subprocess.run([sys.executable, os.path.join(BASE, script),
                            os.path.join(out, data_dir)],
                           capture_output=True, text=True, cwd=BASE)
        if r.returncode != 0:
            raise RuntimeError(f"{script} failed:\n{r.stderr[-2000:]}")
        dst = os.path.join(out, result)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        io.open(dst, "w", encoding="utf-8", newline="\n").write(r.stdout)
        got.append((result, os.path.getsize(dst)))
        for f in sorted(os.listdir(os.path.join(out, data_dir))):
            rel = f"{data_dir}/{f}"
            if f.endswith(".csv"):
                n = sum(1 for _ in io.open(os.path.join(out, rel), encoding="utf-8")) - 1
            else:
                n = os.path.getsize(os.path.join(out, rel))
            got.append((rel, n))
    return got


def manifest(out, fits, data, code):
    lines = ["# Package manifest",
             "",
             "Generated by `export_review_package.py`. Sizes are bytes; `rows` is the "
             "row count of the frame as exported.",
             "",
             "## Rolling backtest frames", ""]
    if len(fits):
        lines += ["| " + " | ".join(fits.columns) + " |",
                  "| " + " | ".join("---" for _ in fits.columns) + " |"]
        lines += ["| " + " | ".join(str(v) for v in r) + " |"
                  for r in fits.itertuples(index=False)]
    else:
        lines.append("Not re-run on this export (`--skip-rolling`); the frames listed "
                     "below are the ones already present, and their hashes are still "
                     "computed fresh.")
    # the hand-written root files are hashed too, so the manifest covers the whole
    # package rather than only the generated parts
    root = [(f, os.path.getsize(os.path.join(out, f)))
            for f in ("00_START_HERE.md", "PROMPT.md", "reproduce.py")
            if os.path.exists(os.path.join(out, f))]
    lines += ["", "## Files", "", "| file | rows/bytes | sha256 (16) |",
              "| --- | --- | --- |"]
    for rel, n in root + data + code:
        p = os.path.join(out, rel)
        lines.append(f"| `{rel}` | {n:,} | `{sha(p)}` |")
    lines.append("")
    io.open(os.path.join(out, "MANIFEST.md"), "w", encoding="utf-8",
            newline="\n").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BASE, "..", "TALLEC_review_package"))
    ap.add_argument("--origins", default="2023,2024,2025")
    ap.add_argument("--skip-rolling", action="store_true",
                    help="reuse the frames already exported (the backtest is slow)")
    ap.add_argument("--skip-analyses", action="store_true",
                    help="do not re-run noise_floor.py and team_role_trend.py")
    ap.add_argument("--check", action="store_true",
                    help="report which packaged files no longer match the repo, "
                         "write nothing, exit 1 if any do not")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if a.check:
        return check(out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    print("code and documents ...")
    code = copy_code(out)
    print("database tables ...")
    data = tables(out)
    fits = pd.DataFrame()
    if a.skip_rolling:
        print("rolling backtest frames: reusing what is already exported")
    else:
        print("rolling backtest frames (slow) ...")
        fits = rolling_frames(out, [int(x) for x in a.origins.split(",")])
    if a.skip_analyses:
        print("analyses: not re-run; listing what is already exported")
        for _, data_dir, result in ANALYSES:
            for rel in [result] + [f"{data_dir}/{f}" for f in sorted(
                    os.listdir(os.path.join(out, data_dir)))
                    if os.path.isdir(os.path.join(out, data_dir))]:
                path = os.path.join(out, rel)
                if os.path.exists(path):
                    data.append((rel, sum(1 for _ in io.open(path, encoding="utf-8")) - 1
                                 if rel.endswith(".csv") else os.path.getsize(path)))
    else:
        print("analyses behind the status note ...")
        data += analyses(out)
    # listed either way, so a --skip-rolling run still produces a complete manifest
    for o in a.origins.split(","):
        for k in ("rolling_train", "rolling_eval"):
            rel = f"data/{k}_{o}.csv"
            if os.path.exists(os.path.join(out, rel)):
                data.append((rel, sum(1 for _ in io.open(
                    os.path.join(out, rel), encoding="utf-8")) - 1))
    manifest(out, fits, data, code)
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
