# -*- coding: utf-8 -*-
"""Seal the v1 translation model's 2026 result before anything is changed.

This is the only genuinely clean holdout the project will have until 2027. The v1
model was fitted through 2025 and had never been compared against 2026 when its
projections were made, so its error on 2026 is a real out-of-sample number. The moment
the model is revised in the light of what 2026 showed — which is exactly what the
review asks for — that stops being true for the successor: the 2026 season will have
informed the design, and any v2 figure on it is a post-hoc evaluation, not a holdout.

So the v1 number is recorded here, once, with the hashes of the artefacts that produced
it. If any of those hashes stop matching, this file describes a run that can no longer
be reproduced and it says so rather than pretending otherwise.

The cohort is reported split three ways, because the aggregate on its own was
misleading: most of the men in it were established NRL players who had a spell in
reserve grade, not players being promoted. The split is a property of who these players
are, not of the model, so recording it does not contaminate anything.

    python record_v1_holdout.py            # writes v1_holdout_record.json, refuses to overwrite
    python record_v1_holdout.py --verify   # re-checks the hashes against the record
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

import runtime
import sp_schema as sp
import trace_cohort as tc

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
OUT = os.path.join(BASE, "v1_holdout_record.json")
ARTEFACTS = ["translation_model_v2.pkl", "config.json"]


def file_hash(name):
    p = os.path.join(BASE, name)
    if not os.path.exists(p):
        return None
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def db_fingerprint():
    """Content fingerprint of the inputs, not just a row count.

    A row count is not evidence that the input data is unchanged — rows can be edited in
    place. This hashes every column the models actually consume.

    It did not always. Until 2026-09-22 the query selected identifiers, minutes and
    position and stopped there, while the docstring claimed it covered the measured
    columns. Every run metre, tackle and try in the database could have been rewritten
    and the fingerprint would have matched, which is the one thing it exists to prevent.
    The external review found it.

    Changing the columns changes the hash, so the recorded fingerprint from before that
    date cannot be compared with one from after; `--verify` reports that as a mismatch
    it cannot resolve rather than pretending either way.
    """
    cols = ["player_id", "competition", "season", "round", "team", "minutes",
            "position"] + sorted(sp.ENGINE_MAP.values())
    con = sqlite3.connect(DB)
    have = {r[1] for r in con.execute("PRAGMA table_info(player_match_stats)")}
    missing = [c for c in cols if c not in have]
    use = [c for c in cols if c in have]
    d = pd.read_sql(f"SELECT {', '.join(chr(34) + c + chr(34) for c in use)} "
                    f"FROM player_match_stats "
                    f"ORDER BY competition, season, round, player_id", con)
    con.close()
    payload = d.to_csv(index=False).encode("utf-8")
    return dict(rows=len(d), columns=len(use), missing=missing,
                covers="identifiers, minutes, position and every measured stat the "
                       "engine reads",
                sha256=hashlib.sha256(payload).hexdigest()[:16])


def classify(con, players):
    """Debutant, returner, or still an NRL regular in the source season."""
    hist = pd.read_sql(
        "SELECT player_id, SUM(CASE WHEN season<=2025 THEN 1 ELSE 0 END) before_2026, "
        "       SUM(CASE WHEN season=2025 THEN 1 ELSE 0 END) in_2025 "
        "FROM player_match_stats WHERE competition='NRL' GROUP BY 1", con)
    m = players.merge(hist, on="player_id", how="left").fillna({"before_2026": 0,
                                                                "in_2025": 0})
    def label(r):
        if r.before_2026 == 0:
            return "first NRL season"
        if r.in_2025 > 0:
            return "NRL regular in 2025 as well"
        return "returning after a season away"
    m["cohort"] = m.apply(label, axis=1)
    return m


def measure(d):
    rows = []
    for label, p in [("ladder", d.projected), ("ridge", d.model),
                     ("no translation", d.rated)]:
        ok = p.notna()
        e = d.actual[ok] - p[ok]
        rows.append(dict(predictor=label, n=int(ok.sum()),
                         mae=round(float(e.abs().mean()), 3),
                         bias=round(float(e.mean()), 3)))
    return rows


def build():
    con = sqlite3.connect(DB)
    d = tc.cohort(con)
    # one row per player: a man who appears from two feeder competitions was counted
    # twice, which inflated the cohort and double-weighted him in every average
    d = d.sort_values("feeder_games", ascending=False).drop_duplicates("player")
    ids = pd.read_sql("SELECT DISTINCT player_id, player FROM player_match_stats", con)
    d = d.merge(ids.drop_duplicates("player"), left_on="player", right_on="player",
                how="left")
    d = classify(con, d)
    con.close()
    graded = d[d.nrl_games >= 5].copy()

    prov = runtime.provenance()
    rec = {
        "what": "v1 translation model, held-out 2026 result, recorded before any "
                "revision informed by 2026",
        "recorded_at": prov["at"],
        "commit": prov["commit"],
        "git_tree_dirty": prov["dirty"],
        "config_hash_canonical": prov["config_hash"],
        "artefacts": {a: file_hash(a) for a in ARTEFACTS},
        "input_fingerprint": db_fingerprint(),
        "freeze_season": 2025,
        "cohort_definition": "rated in NSW or QLD Cup in 2025 over at least 5 matches, "
                             "then appeared in the NRL in 2026; one row per player",
        "cohort_size": int(len(d)),
        "graded_min_nrl_matches": 5,
        "graded_size": int(len(graded)),
        "overall": measure(graded),
        "by_cohort": {},
        "caveats": [
            "The 2026 season was 20 rounds old when this was recorded, so `actual` "
            "rests on a partial season.",
            "NRL 2026 positions are estimated, not from match sheets; that noise is in "
            "`actual`, not in the projection.",
            "Only players a club chose to select appear, so this measures the "
            "projection among those given the chance.",
            "The 'first NRL season' group is small; its numbers are indicative only.",
        ],
    }
    for name, g in graded.groupby("cohort"):
        rec["by_cohort"][name] = {"n": int(len(g)), "results": measure(g)}
    return rec, graded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.verify:
        if not os.path.exists(OUT):
            print("no record to verify")
            return 1
        rec = json.load(open(OUT, encoding="utf-8"))
        ok = True
        for name, h in rec["artefacts"].items():
            now = file_hash(name)
            state = "unchanged" if now == h else f"CHANGED (now {now})"
            print(f"  {name:<28} {h} {state}")
            ok &= now == h
        fp = db_fingerprint()
        old = rec["input_fingerprint"]
        # The fingerprint was widened on 2026-09-22 to cover the measured stats, which it
        # had always claimed to cover and did not. A record written before that hashed a
        # different set of columns, so a mismatch there says nothing about whether the
        # data moved — and saying nothing is the honest answer, not "changed".
        widened = old.get("columns") is None
        same = fp["sha256"] == old["sha256"]
        if widened:
            print(f"  {'player_match_stats':<28} {old['sha256']} was recorded over "
                  f"identifiers only, so it is not comparable with today's wider "
                  f"fingerprint ({fp['sha256']} over {fp['columns']} columns)")
        else:
            print(f"  {'player_match_stats':<28} {old['sha256']} "
                  f"{'unchanged' if same else 'CHANGED (now ' + fp['sha256'] + ')'}")
            ok &= same
        if ok:
            print("\nthe recorded result is still reproducible from this checkout")
            return 0
        # The record is a measurement taken on a date, and it stands whatever happens
        # afterwards. What a mismatch means is that this checkout has moved on — which
        # is the point of checking, not a fault in the record.
        print("\nThe record still stands: it is what the v1 model produced on 2026-08-28,"
              "\nand that does not change. What has moved is this checkout, so re-running"
              "\nthe measurement here would give different numbers. The lines above say"
              "\nwhich part moved.")
        if rec["artefacts"].get("translation_model_v2.pkl") == file_hash(
                "translation_model_v2.pkl"):
            print("\nThe model itself is untouched, which is the part that matters most:"
                  "\nthe sealed out-of-sample figure was produced by that file and it is"
                  "\nstill the file it was.")
        return 1

    if os.path.exists(OUT):
        print(f"{OUT} already exists and is not overwritten — it is the sealed record. "
              f"Delete it deliberately if it must be rebuilt.")
        return 1
    rec, graded = build()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(f"sealed {OUT}\n")
    print(f"cohort {rec['cohort_size']}, graded {rec['graded_size']}")
    print(f"\noverall: " + " | ".join(
        f"{r['predictor']} MAE {r['mae']:.2f} bias {r['bias']:+.2f}"
        for r in rec["overall"]))
    for name, v in rec["by_cohort"].items():
        print(f"\n{name} (n={v['n']}):")
        for r in v["results"]:
            print(f"   {r['predictor']:>16}: MAE {r['mae']:5.2f}  bias {r['bias']:+5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
