# What data is in this repository, and what is not

Everything here runs against `tallec.db`, which is not in git and was removed from its
history on 2026-09-29. One reason is enough on its own:

- **Size.** It is 110 MiB. GitHub refuses any file over 100 MiB, so a repository
  containing it cannot be pushed at all.

Two more were the reason for the 2026-09-29 rewrite, and still describe the data:

- **Licence.** It holds Stats Perform data supplied under a client agreement.
- **Personal data.** `players.dob` is a date of birth for roughly 3,000 named athletes.

## The one database that is committed

Taking `tallec.db` out left the live app with nothing to open, and it ran on an empty
file from 2026-09-29 until 2026-10-07. Since then the repository carries
**`tallec_app.db`** — `tallec.db` without `player_match_raw`, the full Stats Perform
export that the app never reads — plus the audit log's `model_runs` table as
`audit_model_runs`, so the app can say when the ratings were last rebuilt. It is about
48 MB and is built by `build_app_db.py`;
`python build_app_db.py --check` exits 1 when it has fallen behind `tallec.db`.

It still contains Stats Perform rows and the dates of birth, in a public repository.
The project owner decided this knowingly on 2026-10-07, with the two reasons above
stated, as the way to keep the app running without a private repository or a separate
data host. If that decision is reversed, either make the repository private
(and grant the Streamlit GitHub App access, or both apps stop deploying), or host
`tallec_app.db` somewhere private and fetch it at startup.

Nothing writes to `tallec_app.db` except `build_app_db.py`. Every ingest, rebuild and
guarded write targets `tallec.db`, and the copy is rebuilt from it afterwards.

## What is missing

| file | size | what it is |
| --- | --- | --- |
| `tallec.db` | 110 MiB | everything — 122,359 player-match rows across four competitions, ratings, translation tables, position metrics |
| `tallec_audit.db` | 52 KB | the guarded-write log: every rebuild, its config hash and row counts |
| `tallec_seed_backup.db` | 328 KB | the original seeded database, kept for provenance |

## Where to get it

It lives with the project owner, alongside the Stats Perform exports it was built from —
the Australian history file, the Super League master, and the per-season CSVs. Ask before
assuming any copy you find is current: the database has been rebuilt many times and
`MANIFEST.json` in this repository records the hashes of the tables the committed results
were produced from. If a copy does not match the manifest, the reports here do not
describe it.

## Rebuilding from nothing

Possible but not quick, and it needs the source exports rather than the database:

    python ingest_aus_history.py      # the Australian competitions
    python ingest_full_season.py      # a season's exports
    python regenerate_full.py         # ratings and contribution
    python fit_translation_v3.py      # the translation model
    python position_metrics.py --write

Then `python build_manifest.py --check` will tell you whether what you have matches what
the committed reports were written from. Expect it not to on the first attempt.

**Do not run `fit_translation_v2.py`.** It overwrites `translation_model_v2.pkl`, which is
sealed: `v1_holdout_record.json` records an out-of-sample result produced by that exact
file, and it is the only untouched holdout the project has. The script refuses to be
imported for the same reason.
