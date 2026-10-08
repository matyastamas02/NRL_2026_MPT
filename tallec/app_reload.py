# -*- coding: utf-8 -*-
"""Keep the app's own modules in step with the files a push replaces.

Streamlit Cloud applies a push by swapping files under the running process, and Python
keeps every module it has already imported. Twice that left the live app answering from
the past: once with a new bosc_app.py importing a name the old predict_translation in
memory did not have, and -- found by the seventh review -- with predict_translation's
own caches (the fitted lines, the model pickle, the ladder) surviving a change to the
database or the pickle, because only a change to the module's source was being watched.

A module is reloaded when its source file, or any file it reads at import or caches from,
has changed since it was loaded. Dependencies are listed first so a reloaded module
imports reloaded ones.
"""
import importlib
import os
import sys

MODULES = ("sp_schema", "metric_spec", "translation_features", "runtime",
           "player_rating_engine", "predict_translation")
# files a module reads at import time or caches from, relative to the app folder
DEPENDS = {
    "translation_features": ("config.json",),
    "player_rating_engine": ("config.json",),
    "predict_translation": ("config.json", "translation_model_v3.pkl",
                            "tallec.db", "tallec_app.db"),
}


def _stamp(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def refresh(app_dir, modules=MODULES, depends=DEPENDS):
    """Reload every loaded app module whose files changed; return the names reloaded."""
    app_dir = os.path.abspath(app_dir)
    reloaded = []
    for name in modules:
        mod = sys.modules.get(name)
        src = getattr(mod, "__file__", None)
        if not src or os.path.dirname(os.path.abspath(src)) != app_dir:
            continue
        stamp = (_stamp(src),) + tuple(_stamp(os.path.join(app_dir, f))
                                       for f in depends.get(name, ()))
        if getattr(mod, "_loaded_stamp", None) != stamp:
            mod = importlib.reload(mod)
            reloaded.append(name)
        mod._loaded_stamp = stamp
    return reloaded
