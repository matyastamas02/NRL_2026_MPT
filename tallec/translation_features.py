# -*- coding: utf-8 -*-
"""The one definition of the translation model's features, used to fit and to predict.

There were two definitions before, and they disagreed. Training built its columns with
`pd.get_dummies(..., drop_first=True)` over a position group; prediction rebuilt them by
hand, starting every column at the training mean and overwriting the ones it could
resolve. Three consequences, all silent:

  * Prediction mapped a *raw* Stats Perform position ("Half Back") to a group. Callers
    were passing a group ("Halves"), which is not a key in that map, so the lookup
    failed and the position was dropped. `Halves`, `Back Row` and `Bench` all behaved
    this way; `Fullback`, `Prop`, `Centre`, `Winger` and `Hooker` survived only because
    their group name happens to equal their raw name.
  * A position that failed to resolve did not fall back to the reference group. It left
    every group dummy at its training mean — a fractional blend of positions that the
    model was never fitted on and that corresponds to no player.
  * A missing age or minutes figure was quietly replaced by the training mean, so the
    model could not tell "average" from "unknown" and reported neither.

This module fixes all three by construction: one `FeatureSpec`, fitted once on the
training frame and stored inside the model, which both sides then call. If the two ever
diverge again it is a code change, not an accident.

Design decisions, stated because they are choices rather than facts:

  * **The model takes the cumulative Class score the app shows** — not a season-only
    composite. Mike's question is "BOSC says 66 in the NSW Cup, what is that in the
    NRL?", and 66 is the cumulative number. Training and prediction therefore both use
    it, which is what makes the answer mean what the question asked.
  * **Unknown is not the reference group.** `Middles` is the reference and encodes as
    all-zero dummies; an unknown position sets `pos_missing` instead, so the model can
    price ignorance separately from being a middle forward.
  * **Every numeric feature carries a missing flag.** The value still falls back to the
    training median so the row stays in range, but the flag says so, and `inputs_used`
    reports it to the caller.
  * **Source-season features only.** Age, minutes per game and matches played all
    describe the player in the season the projection is made from, never later.
"""
import numpy as np
import pandas as pd

import sp_schema as sp

# every group the rating engine benchmarks within
POSITION_GROUPS = ["Bench", "Centre", "Edge", "Fullback", "Halves", "Hooker",
                   "Middles", "Winger"]
# encoded as all-zero dummies; a group has to be the reference or the dummies are
# collinear with the intercept
REFERENCE_GROUP = "Middles"

# age_delta is the aging curve's expected change for this player's age. Age itself
# stays in as a level term; a straight line cannot represent a curve, which is why the
# two are separate features rather than one.
NUMERIC = ["class_source", "age", "age_delta", "mins_pg", "games_src"]

# Which family each design column belongs to, so a model can be fitted on a subset
# without the feature builder changing shape. Everything is still COMPUTED and reported
# back to the caller — the app shows a recruiter which inputs were available for a player
# whether or not the model uses them — but only the selected families reach the fit.
#
# The selection lives in config.json -> translation.model_features and is measured by
# ablate_translation.py rather than argued.
#
# How hard the position-by-direction interactions are pooled toward the common direction
# effect. Lower means more pooling; 1.0 would be none at all. See the note in `transform`
# for why this is done with the penalty rather than with a hierarchical prior.
INTERACTION_SCALE = 0.3

FAMILY = {"class_source": "class_source", "age": "age", "age_delta": "age",
          "age_missing": "age", "age_delta_missing": "age",
          "mins_pg": "load", "games_src": "load",
          "mins_pg_missing": "load", "games_src_missing": "load",
          "pos_missing": "position"}


def family_of(column):
    """Which feature family a design column belongs to."""
    if column.startswith("px_"):
        return "interaction"
    if column.startswith("pair_"):
        return "pair"
    if column.startswith("grp_"):
        return "position"
    return FAMILY.get(column, "other")


def pooling_multipliers(columns):
    """Per-column multipliers to apply AFTER standardisation, for partial pooling.

    Ridge penalises every coefficient by the same alpha. Scaling a column down by s
    makes its effective penalty alpha/s^2, which is how the position-by-direction
    interactions are pooled toward the common direction effect instead of each cell of
    nine moves inventing its own shift.

    The order matters and got it wrong once. Until 2026-09-23 the scaling was applied
    inside `FeatureSpec.transform`, before `StandardScaler`, which divides every column
    by its own standard deviation and therefore undid it exactly — the fitted
    coefficients were identical at 1.0, 0.3 and 0.01, and the ablation that reported the
    interaction as the largest single improvement was measuring an UNPOOLED one. The
    third external review found it. Applied here, after standardisation, nothing
    renormalises it away.

    Callers must use this on both sides: `Ridge.fit(scaler.transform(X) * mult, y)` and
    `model.predict(scaler.transform(X) * mult)`.
    """
    return np.array([INTERACTION_SCALE if family_of(c) == "interaction" else 1.0
                     for c in columns], dtype=float)


def select_columns(columns, families):
    """Design columns to fit on. `other` is never dropped — it is the escape hatch for
    anything added later that nobody has classified yet, and silently discarding such a
    column would be the same class of bug as the position map going unnoticed."""
    keep = set(families) | {"class_source", "other"}
    return [c for c in columns if family_of(c) in keep]


# plausible ranges, outside which a value is clamped and the caller told
FEATURE_RANGE = {"class_source": (0.0, 100.0), "age": (16.0, 42.0),
                 "age_delta": (-3.0, 3.0), "mins_pg": (1.0, 80.0),
                 "games_src": (1.0, 40.0)}


def resolve_position(raw_position=None, position_group=None):
    """Return (group, how) from either a raw Stats Perform label or a group.

    The two are separate arguments on purpose. Passing a group where a raw position was
    expected is the bug this module exists to prevent, and it cannot be expressed here:
    a raw label that is not in the schema's map raises, rather than silently becoming
    nothing.
    """
    if raw_position is not None and position_group is not None:
        mapped = sp.POSITION_GROUP.get(raw_position)
        if mapped != position_group:
            raise ValueError(
                f"raw_position={raw_position!r} maps to {mapped!r}, which contradicts "
                f"position_group={position_group!r}")
        return position_group, "both agreed"
    if position_group is not None:
        if position_group not in POSITION_GROUPS:
            raise ValueError(f"unknown position group {position_group!r}; "
                             f"expected one of {POSITION_GROUPS}")
        return position_group, "group"
    if raw_position is not None:
        mapped = sp.POSITION_GROUP.get(raw_position)
        if mapped is None:
            raise ValueError(
                f"unknown raw position {raw_position!r}. If this is already a position "
                f"group, pass it as position_group= instead.")
        return mapped, "raw"
    return None, "missing"


class FeatureSpec:
    """The column layout and the fallbacks, fitted once and carried in the model."""

    def __init__(self, medians, pairs, columns):
        self.medians = medians
        self.pairs = list(pairs)
        self.columns = list(columns)

    @classmethod
    def fit(cls, rows, pairs=()):
        med = {}
        for c in NUMERIC:
            v = pd.to_numeric(rows[c], errors="coerce") if c in rows else pd.Series(dtype=float)
            med[c] = float(v.median()) if v.notna().any() else 0.0
        cols = list(NUMERIC)
        cols += [f"{c}_missing" for c in NUMERIC]
        cols += [f"grp_{g}" for g in POSITION_GROUPS if g != REFERENCE_GROUP]
        cols += ["pos_missing"]
        cols += [f"pair_{p}" for p in pairs]
        cols += [f"px_{p}__{g}" for p in pairs for g in POSITION_GROUPS
                 if g != REFERENCE_GROUP]
        return cls(med, pairs, cols)

    def transform(self, rows):
        """Feature matrix in the fitted column order, plus what was actually used."""
        rows = rows.reset_index(drop=True)
        n = len(rows)
        out = pd.DataFrame(0.0, index=range(n), columns=self.columns)
        used = []
        for i in range(n):
            r = rows.iloc[i]
            note = {"clamped": []}
            for c in NUMERIC:
                v = r.get(c) if c in rows.columns else None
                have = v is not None and pd.notna(v)
                if have:
                    v = float(v)
                    lo, hi = FEATURE_RANGE.get(c, (None, None))
                    if lo is not None and not (lo <= v <= hi):
                        note["clamped"].append(f"{c}={v:g} outside [{lo:g}, {hi:g}]")
                        v = min(max(v, lo), hi)
                else:
                    v = self.medians[c]
                out.at[i, c] = v
                out.at[i, f"{c}_missing"] = 0.0 if have else 1.0
                note[c] = bool(have)
            grp, how = resolve_position(r.get("raw_position"), r.get("position_group"))
            if grp is None:
                out.at[i, "pos_missing"] = 1.0
            elif grp != REFERENCE_GROUP:
                out.at[i, f"grp_{grp}"] = 1.0
            note["position"] = grp
            note["position_from"] = how
            if self.pairs:
                want = f"pair_{r.get('source')}->{r.get('target')}"
                if want in self.columns:
                    out.at[i, want] = 1.0
                note["pair"] = want[5:] if want in self.columns else None
                # A plain indicator. The pooling is applied by `pooling_multipliers`
                # AFTER standardisation — see the note there for why it cannot be done
                # here.
                if grp is not None and grp != REFERENCE_GROUP:
                    ix = f"px_{want[5:]}__{grp}"
                    if ix in self.columns:
                        out.at[i, ix] = 1.0
            used.append(note)
        return out.astype(float), used

    def to_dict(self):
        return {"medians": self.medians, "pairs": self.pairs, "columns": self.columns}

    @classmethod
    def from_dict(cls, d):
        return cls(d["medians"], d["pairs"], d["columns"])


def frame(class_source, source, target, raw_position=None, position_group=None,
          age=None, minutes_pg=None, games=None):
    """A one-row input frame, so callers never assemble column names themselves.

    `age_delta` is derived here rather than asked for, so a caller cannot supply an age
    and an inconsistent aging adjustment.
    """
    import aging
    delta = None if age is None or pd.isna(age) else float(aging.expected_delta(age))
    return pd.DataFrame([{
        "class_source": class_source, "source": source, "target": target,
        "raw_position": raw_position, "position_group": position_group,
        "age": age, "age_delta": delta,
        "mins_pg": minutes_pg, "games_src": games}])
