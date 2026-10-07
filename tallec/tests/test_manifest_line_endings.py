# -*- coding: utf-8 -*-
"""The manifest's hashes do not depend on the platform that computes them.

They did. `to_csv()` ends rows with os.linesep, and a git checkout writes text files with
whatever endings its autocrlf setting chooses, so after the project moved from Windows to
macOS `build_manifest.py --check` reported every table, the input and the sealed holdout
record as drifted — on a database byte-identical to the one the manifest was written
from. A provenance check that fails on unchanged data is worse than none: the first real
drift would be read as the same false alarm.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_manifest as bm


def test_frame_hash_ignores_os_linesep(monkeypatch):
    d = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    h = bm.frame_hash(d)
    for sep in ("\n", "\r\n"):
        monkeypatch.setattr(os, "linesep", sep)
        assert bm.frame_hash(d) == h


def test_frame_hash_keeps_the_windows_form_existing_manifests_hold():
    d = pd.DataFrame({"a": [1, 2]})
    expected = bm._sha(d.to_csv(index=False, lineterminator="\r\n").encode())
    assert bm.frame_hash(d) == expected


def test_text_artefact_hash_ignores_line_endings(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "BASE", str(tmp_path))
    (tmp_path / "lf.json").write_bytes(b'{\n  "a": 1\n}\n')
    (tmp_path / "crlf.json").write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    assert bm.file_hash("lf.json") == bm.file_hash("crlf.json")
    # the CRLF form is what a manifest recorded on a Windows checkout holds
    assert bm.file_hash("lf.json", crlf=True) == bm._sha(b'{\r\n  "a": 1\r\n}\r\n')


def test_binary_artefact_is_hashed_byte_for_byte(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "BASE", str(tmp_path))
    raw = b"\x80\x04\r\n\x00\n"
    (tmp_path / "m.pkl").write_bytes(raw)
    assert bm.file_hash("m.pkl") == bm._sha(raw)
    assert bm.file_hash("m.pkl", crlf=True) == bm._sha(raw)


def test_a_real_change_still_drifts(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "BASE", str(tmp_path))
    (tmp_path / "a.json").write_bytes(b'{"a": 1}\n')
    (tmp_path / "b.json").write_bytes(b'{"a": 2}\n')
    assert bm.file_hash("a.json") != bm.file_hash("b.json")
    assert bm.file_hash("a.json", crlf=True) != bm.file_hash("b.json", crlf=True)
