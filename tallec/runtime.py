# -*- coding: utf-8 -*-
"""Safety and provenance for anything that writes to tallec.db.

Two jobs, both learned the hard way.

**Guarded writes.** Every import and rebuild in this project has been preceded by a
manual `cp tallec.db backup.db`, six times in one day, and on the one occasion a
script died half-way the recovery was manual too. `guarded_write` takes the snapshot,
restores it if the block raises, and records what happened either way.

**Provenance.** A number is only checkable if you can say which code and which
configuration produced it. `provenance()` returns the commit, whether the tree was
dirty, a hash of config.json and the row count, and every guarded write and model run
records it. The app shows it in the footer instead of a hardcoded date.

Tables it maintains:

  data_imports — one row per guarded write: what ran, with what arguments, rows before
                 and after, the backup it took, and whether it succeeded.
  model_runs   — one row per rating or model rebuild, with the fit statistics, so two
                 numbers taken a week apart can be traced to different runs.
"""
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
# The audit log lives in its own file. It used to live in tallec.db, which meant a
# rollback restored a snapshot taken BEFORE the failure was recorded — the recovery
# erased the record of what it was recovering from. An audit trail cannot sit inside
# the thing it audits.
AUDIT_DB = os.path.join(BASE, "tallec_audit.db")
BACKUP_DIR = os.path.join(BASE, "_backups")
KEEP_BACKUPS = 8

DDL = {
    "data_imports": """
        CREATE TABLE IF NOT EXISTS data_imports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            script TEXT NOT NULL,
            label TEXT,
            argv TEXT,
            commit_sha TEXT,
            tree_dirty INTEGER,
            config_hash TEXT,
            rows_before INTEGER,
            rows_after INTEGER,
            backup_path TEXT,
            status TEXT NOT NULL,
            note TEXT
        )""",
    "model_runs": """
        CREATE TABLE IF NOT EXISTS model_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT NOT NULL,
            script TEXT NOT NULL,
            target TEXT NOT NULL,
            commit_sha TEXT,
            tree_dirty INTEGER,
            config_hash TEXT,
            db_rows INTEGER,
            seconds REAL,
            stats TEXT
        )""",
}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def audit_con():
    con = sqlite3.connect(AUDIT_DB)
    for ddl in DDL.values():
        con.execute(ddl)
    con.commit()
    return con


def ensure_tables(con=None):
    """Kept for callers that passed a connection; the audit tables are their own file."""
    audit_con().close()


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=BASE, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def config_hash():
    p = os.path.join(BASE, "config.json")
    if not os.path.exists(p):
        return ""
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]


def row_count(con=None):
    own = con is None
    con = con or sqlite3.connect(DB)
    try:
        return con.execute("SELECT count(*) FROM player_match_stats").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        if own:
            con.close()


def provenance(con=None):
    """Everything needed to say which run produced a number."""
    sha = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return {"commit": sha or "unknown", "dirty": dirty,
            "config_hash": config_hash(), "at": _now(),
            "db_rows": row_count(con),
            "db_mb": round(os.path.getsize(DB) / 1048576, 1) if os.path.exists(DB) else 0}


def snapshot(label="manual"):
    """Copy the database aside and prune old copies. Returns the path."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
    path = os.path.join(BACKUP_DIR, f"tallec_{stamp}_{safe}.db")
    shutil.copy2(DB, path)
    kept = sorted(f for f in os.listdir(BACKUP_DIR) if f.endswith(".db"))
    for old in kept[:-KEEP_BACKUPS]:
        os.remove(os.path.join(BACKUP_DIR, old))
    return path


@contextmanager
def guarded_write(label, note=None, dry_run=False):
    """Snapshot, run the block, restore on failure, record either way.

    A dry run records nothing and takes no snapshot, so --dry-run stays free.
    """
    if dry_run:
        yield None
        return

    prov = provenance()
    before = row_count()
    backup = snapshot(label)
    con = audit_con()
    cur = con.execute(
        "INSERT INTO data_imports (started_at, script, label, argv, commit_sha, "
        "tree_dirty, config_hash, rows_before, backup_path, status, note) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (prov["at"], os.path.basename(sys.argv[0]) or "python", label,
         " ".join(sys.argv[1:]), prov["commit"], int(prov["dirty"]),
         prov["config_hash"], before, backup, "running", note))
    run_id = cur.lastrowid
    con.commit()
    con.close()

    t0 = time.time()
    try:
        yield run_id
    except BaseException as exc:
        shutil.copy2(backup, DB)
        con = audit_con()
        con.execute("UPDATE data_imports SET finished_at=?, rows_after=?, status=?, "
                    "note=? WHERE id=?",
                    (_now(), row_count(), "rolled_back",
                     f"{type(exc).__name__}: {exc}"[:400], run_id))
        con.commit()
        con.close()
        print(f"\nROLLED BACK: {type(exc).__name__}: {exc}")
        print(f"the database was restored from {os.path.basename(backup)}; "
              f"data_imports row {run_id} records it")
        raise
    else:
        after = row_count()
        con = audit_con()
        con.execute("UPDATE data_imports SET finished_at=?, rows_after=?, status=? "
                    "WHERE id=?", (_now(), after, "ok", run_id))
        con.commit()
        con.close()
        print(f"guarded write ok: {before:,} -> {after:,} rows "
              f"({after - before:+,}) | backup {os.path.basename(backup)} | "
              f"data_imports id {run_id}")


def record_model_run(target, stats, seconds, script=None):
    """One row per rating or model rebuild."""
    prov = provenance()
    con = audit_con()
    con.execute(
        "INSERT INTO model_runs (run_at, script, target, commit_sha, tree_dirty, "
        "config_hash, db_rows, seconds, stats) VALUES (?,?,?,?,?,?,?,?,?)",
        (prov["at"], script or (os.path.basename(sys.argv[0]) or "python"), target,
         prov["commit"], int(prov["dirty"]), prov["config_hash"], prov["db_rows"],
         round(seconds, 1), json.dumps(stats, default=str)))
    con.commit()
    con.close()


def restore(which=-1):
    """Put a backup back. Default is the most recent."""
    files = sorted(f for f in os.listdir(BACKUP_DIR) if f.endswith(".db"))
    if not files:
        raise SystemExit("no backups in " + BACKUP_DIR)
    pick = files[which]
    shutil.copy2(os.path.join(BACKUP_DIR, pick), DB)
    print(f"restored {pick} | rows now {row_count():,}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore()
    elif len(sys.argv) > 1 and sys.argv[1] == "backups":
        os.makedirs(BACKUP_DIR, exist_ok=True)
        for f in sorted(os.listdir(BACKUP_DIR)):
            p = os.path.join(BACKUP_DIR, f)
            print(f"  {f}  {os.path.getsize(p)/1048576:.1f} MB")
    else:
        print(json.dumps(provenance(), indent=1))
        con = audit_con()
        for t in ("data_imports", "model_runs"):
            n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            print(f"{t}: {n} rows")
        con.close()
