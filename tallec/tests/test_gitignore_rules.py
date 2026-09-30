# -*- coding: utf-8 -*-
"""The repository's ignore rules, held to blocking secrets and nothing else.

This file exists because the previous rule failed four times in the same way. It was
`*.json` with an allow-list underneath: a rule about a file format standing in for a rule
about secrets. Every ordinary JSON the project needed had to be noticed and named back
in, and four were — `config.json`, `MANIFEST.json`, `v1_holdout_record.json`,
`dash_data.json`. The last was missed for a day, which put `tallec_dashboard.html` on a
public GitHub repository without the data file it reads, with nothing able to regenerate
it. Only a filename-level comparison against the working tree found it; content checks
could not, because a file missing from one side is not a difference between two versions
of it.

The rule now names what a secret looks like. That is the safer failure mode for this
project — a new ordinary file works, and a new credential is caught — but it is only as
good as the list, so the list is tested rather than trusted.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GITIGNORE = os.path.join(REPO, ".gitignore")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(REPO, ".git")) or not os.path.exists(GITIGNORE),
    reason="not inside the repository checkout")


def ignored(path):
    """Whether git would refuse to add this path."""
    return subprocess.run(["git", "-C", REPO, "check-ignore", "-q", path]).returncode == 0


# Names a credential plausibly takes in this project: a Google service-account key for
# the gspread account, Streamlit secrets, an Odds API key, and the usual suspects.
SECRETS = [
    "light-rhythm-494915-e0-a1b2c3.json",   # the actual GCP project behind the key
    "gcp_service_account.json",
    "my-serviceaccount-key.json",
    "secrets.json",
    "client_secret_123.json",
    ".streamlit/secrets.toml",
    ".env",
    ".env.production",
    "aws_credentials",
    "odds_api_key.txt",
    "apikey.json",
    "server-key.json",
    "deploy_key.pem",
    "cert.p12",
    "id_rsa",
    "id_ed25519.pub",
    "tallec/secrets/db_credentials.yaml",
]

# Files the project needs. The first four are the ones the old rule actually swallowed.
ORDINARY = [
    "tallec/config.json",
    "tallec/MANIFEST.json",
    "tallec/v1_holdout_record.json",
    "tallec/dash_data.json",
    "tallec/.claude/launch.json",
    ".devcontainer/devcontainer.json",
    "tallec/a_file_nobody_has_written_yet.json",
    "package.json",
    "app.py",
    "tallec/README.md",
]


@pytest.mark.parametrize("path", SECRETS)
def test_a_credential_is_blocked(path):
    assert ignored(path), f"{path} would be committed"


@pytest.mark.parametrize("path", ORDINARY)
def test_an_ordinary_file_is_not_blocked(path):
    assert not ignored(path), (
        f"{path} is blocked; the rule is catching ordinary files again, which is how "
        f"the dashboard lost its data file")


def test_the_databases_stay_local():
    """Licensed Stats Perform rows and three thousand dates of birth, public repo."""
    for path in ("tallec/tallec.db", "tallec/tallec_audit.db", "anything.db",
                 "tallec/_backups/tallec_20260101.db"):
        assert ignored(path), path


def test_nothing_currently_tracked_has_become_ignored():
    """The check that makes tightening the rules safe to do at all."""
    tracked = subprocess.run(["git", "-C", REPO, "ls-files"],
                             capture_output=True, text=True).stdout.split("\n")
    lost = [f for f in tracked if f and ignored(f)]
    assert not lost, f"these are committed but would now be ignored: {lost[:10]}"


def test_the_rule_does_not_block_a_whole_file_format():
    """The shape of the old mistake, so it cannot come back unnoticed.

    A bare `*.json` or `*.csv` line means every future file of that kind has to be
    argued back in one at a time. Name the secret instead.
    """
    lines = [ln.strip() for ln in open(GITIGNORE, encoding="utf-8")
             if ln.strip() and not ln.strip().startswith("#")]
    blanket = {"*.json", "*.csv", "*.yaml", "*.yml", "*.toml", "*.txt", "*.md"}
    found = blanket & set(lines)
    assert not found, (
        f"{found} blocks a file format rather than a secret; that rule swallowed four "
        f"files the project needed before it was replaced")
