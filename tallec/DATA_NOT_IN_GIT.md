# The data is not in this repository, and cannot be

Everything here runs against `tallec.db`, which is not in git and was removed from its
history on 2026-09-29. Three reasons, any one of which is sufficient:

- **Size.** It is 111 MB. GitHub refuses any file over 100 MB, so a repository containing
  it cannot be pushed at all.
- **Licence.** It holds Stats Perform data supplied under a client agreement. It is not
  ours to publish.
- **Personal data.** `players.dob` is a date of birth for roughly 3,000 named athletes.

## What is missing

| file | size | what it is |
| --- | --- | --- |
| `tallec.db` | 111 MB | everything — 122,359 player-match rows across four competitions, ratings, translation tables, position metrics |
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
