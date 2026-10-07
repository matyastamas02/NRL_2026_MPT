# -*- coding: utf-8 -*-
"""Record what produced the current artefacts, precisely enough to challenge.

The review's objection to the earlier provenance was fair: an unchanged row count is
not evidence that the input data is unchanged, "the last two runs" is not a reference to
a specific run, and a report that hardcodes half its numbers cannot claim they all come
from artefacts. This writes a manifest instead — one file naming, for every artefact:

  * the commit the code was at, and whether the working tree was clean;
  * the canonical config hash, which now ignores line endings and comment wording but
    not a single changed value;
  * a content hash of the input table, so an edit in place is visible even when the row
    count does not move;
  * the cutoff everything was fitted to;
  * the audit-log run id that produced it, by id rather than by position;
  * the hash of the artefact itself.

The manifest is written after the artefacts and committed after that, so the commit it
names is the one the code was at when they were built, not the one containing the
manifest. That is the normal ordering and is stated here rather than papered over.

    python build_manifest.py            # writes MANIFEST.json
    python build_manifest.py --check    # re-hash everything and report drift
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys

import pandas as pd

import player_rating_engine as pre
import runtime

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
AUDIT = os.path.join(BASE, "tallec_audit.db")
OUT = os.path.join(BASE, "MANIFEST.json")

ARTEFACTS = {
    "translation_model_v3.pkl": "translation_model_v3",
    "translation_model_v2.pkl": None,          # sealed, not rebuilt
    "config.json": None,
    "v1_holdout_record.json": None,
}
TABLES = ["player_ratings", "player_contribution", "player_contribution_rating",
          "translation_ladder_v3", "translation_pairs_v3", "translation_model_v3_meta"]


# Line endings must not reach a hash. to_csv() ends rows with os.linesep, so the same
# table hashed to one value on Windows and another on macOS or Linux (Streamlit Cloud),
# and a git checkout writes text files with whatever endings its autocrlf setting says.
# Both made --check report drift on identical data after the move off Windows.
#  * Tables: rows end in CRLF on every platform — what every manifest so far was
#    written with, so those recorded hashes stay valid.
#  * Text artefacts: hashed with CRLF folded to LF. A manifest recorded before this
#    change hashed the raw bytes of a Windows checkout, so --check also accepts the
#    CRLF form of the same content. Binary artefacts (.pkl) are hashed byte for byte.
CSV_EOL = "\r\n"
TEXT_SUFFIXES = (".json",)


def _sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def file_hash(name, crlf=False):
    p = os.path.join(BASE, name)
    if not os.path.exists(p):
        return None
    b = open(p, "rb").read()
    if name.endswith(TEXT_SUFFIXES):
        b = b.replace(b"\r\n", b"\n")
        if crlf:
            b = b.replace(b"\n", b"\r\n")
    return _sha(b)


def frame_hash(d):
    return _sha(d.to_csv(index=False, lineterminator=CSV_EOL).encode())


def table_hash(con, table):
    """Content hash of a table, ordered so it does not depend on storage order."""
    try:
        d = pd.read_sql(f"SELECT * FROM {table}", con)
    except Exception:
        return None
    d = d.sort_values(list(d.columns)).reset_index(drop=True)
    return {"rows": len(d), "sha256": frame_hash(d)}


def input_hash(con):
    d = pd.read_sql(
        "SELECT player_id, competition, season, round, team, minutes, position, "
        "       all_run_metres, tackles, tries FROM player_match_stats "
        "ORDER BY competition, season, round, player_id", con)
    return {"table": "player_match_stats", "rows": len(d), "sha256": frame_hash(d)}


def last_run(target):
    con = sqlite3.connect(AUDIT)
    try:
        d = pd.read_sql("SELECT id, run_at, commit_sha, tree_dirty, config_hash "
                        "FROM model_runs WHERE target=? ORDER BY id DESC LIMIT 1",
                        con, params=(target,))
    finally:
        con.close()
    return None if d.empty else d.iloc[0].to_dict()


def build():
    prov = runtime.provenance()
    con = sqlite3.connect(DB)
    man = {
        "generated_at": prov["at"],
        "code_commit": prov["commit"],
        # the databases are tracked and every run writes to them, so a wholly clean
        # tree is unobtainable; what a reviewer needs to know is whether the CODE was
        # committed, with the data's integrity carried by the content hashes below
        "code_clean": not prov["code_dirty"],
        "uncommitted_paths": prov["dirty_paths"],
        "config_hash_canonical": prov["config_hash"],
        "freeze_season": pre.FREEZE_SEASON,
        "input": input_hash(con),
        "artefacts": {},
        "tables": {t: table_hash(con, t) for t in TABLES},
        "note": "The commit named here is the one the code was at when the artefacts "
                "were built. The manifest itself is committed afterwards, so the "
                "commit containing this file is one later.",
    }
    for name, target in ARTEFACTS.items():
        entry = {"sha256": file_hash(name)}
        if target:
            r = last_run(target)
            if r:
                entry["produced_by_run_id"] = int(r["id"])
                entry["run_at"] = r["run_at"]
                entry["run_commit"] = r["commit_sha"]
                entry["run_tree_dirty"] = bool(r["tree_dirty"])
                entry["run_config_hash"] = r["config_hash"]
        if name == "translation_model_v2.pkl":
            entry["status"] = ("sealed — the v1 out-of-sample result in "
                               "v1_holdout_record.json was produced with this file and "
                               "it must not be rebuilt")
        man["artefacts"][name] = entry
    con.close()
    return man


def check():
    if not os.path.exists(OUT):
        print("no manifest")
        return 1
    old = json.load(open(OUT, encoding="utf-8"))
    new = build()
    ok = True
    print(f"{'item':<38}{'recorded':<20}{'now':<20}")
    rows = [("config (canonical)", old["config_hash_canonical"],
             new["config_hash_canonical"]),
            ("input player_match_stats", old["input"]["sha256"],
             new["input"]["sha256"])]
    for name in old["artefacts"]:
        a = old["artefacts"][name].get("sha256")
        b = new["artefacts"].get(name, {}).get("sha256")
        # a pre-normalisation manifest may hold the CRLF form of identical content
        if a != b and a is not None and a == file_hash(name, crlf=True):
            b = a
        rows.append((name, a, b))
    for t in old["tables"]:
        a = (old["tables"][t] or {}).get("sha256")
        b = (new["tables"].get(t) or {}).get("sha256")
        rows.append((f"table {t}", a, b))
    for name, a, b in rows:
        same = a == b
        ok &= same
        print(f"{name:<38}{str(a):<20}{str(b):<20}{'' if same else '  DRIFTED'}")
    print("\neverything matches the manifest" if ok else
          "\nSOMETHING HAS DRIFTED — the manifest no longer describes this checkout")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    man = build()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    print(f"wrote {OUT}")
    print(f"  commit {man['code_commit']}  code clean: {man['code_clean']}"
          + (f"  (uncommitted: {', '.join(man['uncommitted_paths'])})"
             if man["uncommitted_paths"] else ""))
    print(f"  config {man['config_hash_canonical']}  freeze {man['freeze_season']}")
    print(f"  input  {man['input']['sha256']} over {man['input']['rows']:,} rows")
    for k, v in man["artefacts"].items():
        print(f"  {k:<28} {v['sha256']}"
              + (f"  run {v['produced_by_run_id']}" if "produced_by_run_id" in v else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
