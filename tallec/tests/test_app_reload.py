# -*- coding: utf-8 -*-
"""A push that replaces a module's source or a file it caches from must reload it."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_reload


def _toy(tmp_path, body):
    (tmp_path / "toy_cache_mod.py").write_text(body)
    sys.path.insert(0, str(tmp_path))


def _bump(path):
    t = time.time() + 5
    os.utime(path, (t, t))


def test_a_changed_data_file_empties_the_module_cache(tmp_path):
    data = tmp_path / "data.db"
    data.write_text("one")
    _toy(tmp_path, "CACHE = {}\ndef read(p):\n    return CACHE.setdefault('v', open(p).read())\n")
    sys.modules.pop("toy_cache_mod", None)
    import toy_cache_mod
    deps = {"toy_cache_mod": ("data.db",)}
    app_reload.refresh(tmp_path, ("toy_cache_mod",), deps)          # stamps it
    assert toy_cache_mod.read(data) == "one"
    data.write_text("two")
    assert toy_cache_mod.read(data) == "one"                        # stale, as before
    _bump(data)
    assert app_reload.refresh(tmp_path, ("toy_cache_mod",), deps) == ["toy_cache_mod"]
    assert sys.modules["toy_cache_mod"].read(data) == "two"
    assert app_reload.refresh(tmp_path, ("toy_cache_mod",), deps) == []


def test_a_changed_source_file_is_reloaded(tmp_path):
    _toy(tmp_path, "NAME = 'old'\n")
    sys.modules.pop("toy_cache_mod", None)
    import toy_cache_mod
    app_reload.refresh(tmp_path, ("toy_cache_mod",), {})
    (tmp_path / "toy_cache_mod.py").write_text("NAME = 'new'\n")
    _bump(tmp_path / "toy_cache_mod.py")
    assert app_reload.refresh(tmp_path, ("toy_cache_mod",), {}) == ["toy_cache_mod"]
    assert sys.modules["toy_cache_mod"].NAME == "new"


def test_modules_from_elsewhere_are_left_alone(tmp_path):
    assert app_reload.refresh(tmp_path, ("os", "json"), {}) == []
