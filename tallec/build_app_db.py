# -*- coding: utf-8 -*-
"""Build the copy of the database the deployed app reads.

`tallec.db` cannot be committed: at 110 MiB it is over GitHub's 100 MiB file limit, so a
repository containing it cannot be pushed at all. When it was taken out of the history on
2026-09-29 nothing took its place, and the live app has opened an empty file ever since —
`sqlite3.connect` creates one rather than failing, so the first query reported a missing
table instead of a missing database.

Most of that size is `player_match_raw`: every column of every Stats Perform export, kept
so a stat can be re-derived without the source files. The app never reads it. Everything
it does read is in the other tables, which come to under half the file. This writes
`tallec_app.db`, the same database without that one table, and that is the file that
goes to git.

Two rules keep the copy honest:

  * nothing writes to it except this script — every ingest, rebuild and guarded write
    still targets `tallec.db`, and the copy is rebuilt from it afterwards;
  * `--check` compares every table in the copy with the same table in `tallec.db`, row
    for row, and exits 1 on any difference, so a copy left behind after a weekly update
    is caught before it is pushed.

    python build_app_db.py            # rebuild tallec_app.db if it is behind
    python build_app_db.py --check    # exit 1 if it differs from tallec.db
"""
import argparse
import os
import sqlite3
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
APP_DB = os.path.join(BASE, "tallec_app.db")

# Tables the app does not read and the copy leaves out. Nothing else is dropped.
LEFT_OUT = ("player_match_raw",)


def _tables(con, schema="main"):
    return [r[0] for r in con.execute(
        f"SELECT name FROM {schema}.sqlite_master WHERE type='table' "
        f"AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def differences():
    """Every way the copy differs from tallec.db, as readable lines; empty if none."""
    if not os.path.exists(APP_DB):
        return [f"{os.path.basename(APP_DB)} does not exist"]
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.execute("ATTACH DATABASE ? AS app", (f"file:{APP_DB}?mode=ro",))
    want = [t for t in _tables(con) if t not in LEFT_OUT]
    have = _tables(con, "app")
    out = [f"missing table {t}" for t in want if t not in have]
    out += [f"table {t} should not be in the copy" for t in have if t not in want]
    for t in (t for t in want if t in have):
        n_main = con.execute(f'SELECT count(*) FROM main."{t}"').fetchone()[0]
        n_app = con.execute(f'SELECT count(*) FROM app."{t}"').fetchone()[0]
        if n_main != n_app:
            out.append(f"{t}: {n_app:,} rows in the copy, {n_main:,} in tallec.db")
            continue
        try:
            only = con.execute(
                f'SELECT count(*) FROM (SELECT * FROM main."{t}" '
                f'EXCEPT SELECT * FROM app."{t}")').fetchone()[0]
        except sqlite3.OperationalError as e:      # the columns no longer line up
            out.append(f"{t}: {e}")
            continue
        if only:
            out.append(f"{t}: {only:,} rows differ")
    con.close()
    return out


def build():
    tmp = APP_DB + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    src = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    dst = sqlite3.connect(tmp)
    src.backup(dst)                     # a consistent snapshot, even mid-write
    src.close()
    for t in LEFT_OUT:
        dst.execute(f'DROP TABLE IF EXISTS "{t}"')
    dst.commit()
    dst.execute("PRAGMA journal_mode=DELETE")
    dst.execute("VACUUM")
    dst.close()
    os.replace(tmp, APP_DB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report whether tallec_app.db differs from tallec.db, write nothing")
    a = ap.parse_args()
    if not os.path.exists(DB):
        print("tallec.db is not here; the copy can only be built or checked next to it "
              "(see DATA_NOT_IN_GIT.md)")
        return 1

    diff = differences()
    if not diff:
        mb = os.path.getsize(APP_DB) / 1e6
        print(f"  tallec_app.db: already current ({mb:.1f} MB)")
        return 0
    if a.check:
        print("  tallec_app.db: OUT OF DATE")
        for line in diff[:20]:
            print(f"    {line}")
        print("\nRun `python build_app_db.py` and commit tallec_app.db.")
        return 1

    build()
    left = differences()
    if left:
        print("  tallec_app.db: rebuilt but still differs — " + "; ".join(left[:5]))
        return 1
    mb = os.path.getsize(APP_DB) / 1e6
    print(f"  tallec_app.db: rebuilt ({mb:.1f} MB, without {', '.join(LEFT_OUT)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
