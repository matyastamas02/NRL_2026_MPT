# -*- coding: utf-8 -*-
"""There must be exactly one position mapping, and this test exists because there wasn't.

On 2026-09-20 Leeds's regrouping was applied to `sp_schema` and not to the rating engine,
which kept its own older split of Prop against Back Row. For two days ratings were built
in one set of peer pools and described by another: a lock was measured against
second-rowers and then labelled a middle. The published medians were 54.9 for locks and
45.6 for second-rowers where both should have sat near 50 inside their own group.

The existing tests did not catch it because they checked `sp_schema` against itself. These
check the modules against each other.

The one deliberate exception is `fit_translation_v2`, which is sealed and keeps the map
that was live when it was fitted — that is asserted too, so it cannot be tidied away.
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import player_rating_engine as pre
import sp_schema as sp
import translation_features as tf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def v2_source():
    """Read `fit_translation_v2.py` rather than import it.

    Importing it refits the model and overwrites the sealed artefact, because its work
    happens at module level. An earlier version of this very test did exactly that and
    broke the seal four times before the manifest's hash check noticed. The script now
    refuses to be imported; this reads the text instead.
    """
    with open(os.path.join(BASE, "fit_translation_v2.py"), encoding="utf-8") as f:
        return f.read()


def test_engine_and_schema_agree_on_every_canonical_position():
    """The regression itself: no raw label may resolve to two different groups."""
    for raw, group in sp.POSITION_GROUP.items():
        assert pre.POSITION_GROUP[raw] == group, (
            f"{raw!r} is {group!r} to sp_schema but "
            f"{pre.POSITION_GROUP[raw]!r} to the rating engine")


def test_engine_has_a_weight_profile_for_every_group_it_can_produce():
    """A group with no weights silently falls back to Bench's, which is a rating bug."""
    for group in set(pre.POSITION_GROUP.values()):
        assert group in pre.POSITION_WEIGHTS, f"no weight profile for {group!r}"


def test_no_retired_group_survives_anywhere():
    retired = {"Prop", "Back Row", "Lock"}
    assert not retired & set(pre.POSITION_WEIGHTS)
    assert not retired & set(pre.POSITION_GROUP.values())
    assert not retired & set(tf.POSITION_GROUPS)


def test_weight_vectors_are_normalised():
    for group, w in pre.POSITION_WEIGHTS.items():
        assert abs(sum(w) - 1.0) < 1e-9, f"{group} weights sum to {sum(w)}"
        assert len(w) == len(pre.RATE_ORDER)


def test_engine_adds_legacy_strings_and_nothing_else():
    extra = set(pre.POSITION_GROUP) - set(sp.POSITION_GROUP)
    assert extra == set(sp.LEGACY_POSITIONS), (
        "the engine's map should be exactly the canonical labels plus the declared "
        "legacy ones")


def test_translation_features_reject_the_legacy_strings():
    """Legacy labels are a storage concern, not a modelling one.

    `Unknown` maps to Bench for the engine so old rows can still be pooled. If that
    leaked into the feature builder, a player of unrecorded position would be silently
    modelled as a bench forward instead of carrying a missing-position flag.
    """
    for raw in sp.LEGACY_POSITIONS:
        if raw in sp.POSITION_GROUP:
            continue
        with pytest.raises(ValueError):
            tf.resolve_position(raw_position=raw)


def test_groups_match_the_feature_builder():
    assert set(tf.POSITION_GROUPS) == set(sp.POSITION_GROUP.values())


def test_the_sealed_v2_keeps_the_map_it_was_fitted_with():
    """v2's result is the only untouched out-of-sample record the project has.

    It must keep the retired map, written out in its own source rather than imported
    from the engine, or re-running it would quietly produce a different model while
    still being called the sealed one.
    """
    src = v2_source()
    assert '"Prop": "Prop"' in src
    assert '"Lock": "Back Row"' in src
    assert '"Second Row": "Back Row"' in src
    assert "pre.POSITION_GROUP" not in src


def test_the_sealed_script_cannot_be_imported():
    """Importing it refits the sealed artefact; that has to fail loudly."""
    src = v2_source()
    assert 'if __name__ != "__main__":' in src
    with pytest.raises(RuntimeError, match="sealed"):
        importlib.import_module("fit_translation_v2")


# ── what the client actually sees ────────────────────────────────────────────

def displayed_strings(path):
    """Every string literal in a module except docstrings.

    Comments never reach the AST and docstrings are excluded explicitly, so what is left
    is roughly what a user can be shown. Grepping the file instead would flag the
    explanations of why a name was retired, which is the opposite of what we want: the
    history should be written down, the label should not.
    """
    import ast
    tree = ast.parse(open(path, encoding="utf-8").read())
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and                     isinstance(body[0].value, ast.Constant) and                     isinstance(body[0].value.value, str):
                docs.add(id(body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


def test_the_app_never_shows_a_retired_position_group():
    """A Benchmarks caption said Second Row and Lock were both Back Row.

    It sat there for four days after Leeds asked for that grouping to be retired, in
    front of the client, because it was a hand-written sentence rather than a lookup. It
    is generated from the schema now, and this stops the next one being typed in.
    """
    retired = ("Back Row", "Prop")
    bad = [t for t in displayed_strings(os.path.join(BASE, "bosc_app.py"))
           if any(r in t for r in retired)]
    assert not bad, f"the app can display a retired position group: {bad}"


def test_the_grouping_note_is_derived_rather_than_written():
    """Importing the app needs Streamlit, so the source is checked instead."""
    app = open(os.path.join(BASE, "bosc_app.py"), encoding="utf-8").read()
    assert "def _grouping_note()" in app
    assert "sp.POSITION_GROUP.items()" in app
    assert "GROUPING_NOTE" in app
