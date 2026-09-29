# -*- coding: utf-8 -*-
"""Which stat weighting should Middles and Edge use — measured, not inherited.

Leeds asked for props and locks to be one group (Middles) and the second row to stand
alone (Edge). `sp_schema` was changed on 2026-09-20; the rating engine was not, and kept
its own older split of Prop against Back Row (second row *and* lock). The external review
of 2026-09-22 found this. Unifying the map means the two new groups need weight vectors,
and the old ones do not map onto them cleanly: the retired Back Row profile was fitted to
a pool that contained the locks now leaving it.

Rather than inherit and hope, each candidate is scored on how much persistent signal its
composite carries. The criterion is the **independent year-to-year correlation**: a
player's season-S composite against his season-S+1 composite, the two sharing no match.
A weighting that loads on what a player reliably repeats scores high; one that loads on
noise scores low. This is the same quantity `validate_ratings.py` reports as
`independent_r`, computed here before shrinkage so the comparison is about the weights
alone.

Reliability is necessary, not sufficient — a weighting could be stable and still measure
the wrong thing, which is why the component weights are printed alongside for a human to
object to. But between candidates that all encode a defensible view of the job, the one
that repeats is the better instrument.

    python weights_for_middles_edge.py
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

import player_rating_engine as pre
import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
OUT = os.path.join(BASE, "WEIGHTS_REPORT.md")
MIN_MATCHES = 5          # per season, to keep a season mean meaningful
RATE = pre.RATE_ORDER
PROP = np.array(pre.POSITION_WEIGHTS["Prop"], dtype=float)
BACKROW = np.array(pre.POSITION_WEIGHTS["Back Row"], dtype=float)

CANDIDATES = {
    "inherit (Middles=Prop, Edge=Back Row)": {"Middles": PROP, "Edge": BACKROW},
    "both = Prop": {"Middles": PROP, "Edge": PROP},
    "both = Back Row": {"Middles": BACKROW, "Edge": BACKROW},
    "Middles = 0.7 Prop + 0.3 Back Row": {"Middles": 0.7 * PROP + 0.3 * BACKROW,
                                          "Edge": BACKROW},
    "Middles = Prop, Edge = 0.5/0.5": {"Middles": PROP,
                                       "Edge": 0.5 * PROP + 0.5 * BACKROW},
    "uniform (control)": {"Middles": np.full(len(RATE), 1 / len(RATE)),
                          "Edge": np.full(len(RATE), 1 / len(RATE))},
}


def rates(df):
    mins = df["minutes"].clip(lower=1)
    out = pd.DataFrame(index=df.index)
    for raw, (rate, sign) in pre.RATE_STATS.items():
        out[rate] = sign * (df[raw].fillna(0.0) if raw in df else 0.0) / mins
    return out[RATE]


def available_mask(df):
    """Rates this pool actually recorded, mirroring the engine's coverage test."""
    m = []
    for raw, (rate, _) in pre.RATE_STATS.items():
        share = float((df[raw].fillna(0) != 0).mean()) if raw in df else 0.0
        m.append(share > pre.MIN_RATE_COVERAGE)
    order = [pre.RATE_STATS[r][0] for r in pre.RATE_STATS]
    return np.array([m[order.index(r)] for r in RATE], dtype=float)


def renorm(w, mask):
    w = np.asarray(w, dtype=float) * mask
    t = w.sum()
    return w / t if t > 0 else mask / max(mask.sum(), 1)


def season_composites(df, w_by_group):
    """Per-player season mean composite, standardising within competition x group."""
    out = []
    for (comp, season), g in df.groupby(["competition", "season"]):
        mask = available_mask(g)
        r = rates(g)
        for grp, gg in g.groupby("group"):
            if grp not in w_by_group:
                continue
            sub = r.loc[gg.index]
            z = (sub - sub.mean()) / sub.std().replace(0, np.nan)
            z = z.fillna(0.0)
            comp_val = z.values @ renorm(w_by_group[grp], mask)
            out.append(pd.DataFrame({
                "player_id": gg.player_id.values, "competition": comp,
                "season": season, "group": grp, "composite": comp_val}))
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    agg = (d.groupby(["player_id", "competition", "season", "group"])
             .composite.agg(["mean", "size"]).reset_index()
             .rename(columns={"mean": "composite", "size": "n"}))
    return agg[agg.n >= MIN_MATCHES]


def independent_r(agg):
    """Correlate season S with season S+1 for the same player, same competition."""
    a = agg.copy()
    a["next"] = a.season + 1
    j = a.merge(a, left_on=["player_id", "competition", "next"],
                right_on=["player_id", "competition", "season"],
                suffixes=("", "_b"))
    rows = []
    for grp, g in j.groupby("group"):
        if len(g) < 30:
            continue
        rows.append(dict(group=grp, pairs=len(g),
                         players=int(g.player_id.nunique()),
                         r=float(np.corrcoef(g.composite, g.composite_b)[0, 1])))
    return pd.DataFrame(rows)


def md(df, fmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, rule] + body)


def main():
    con = sqlite3.connect(DB)
    cols = ["player_id", "competition", "season", "minutes", "position"] + \
           list(pre.RATE_STATS)
    df = pd.read_sql(f"SELECT {', '.join(cols)} FROM player_match_stats "
                     f"WHERE season <= 2025", con)
    con.close()
    df = df[df.minutes >= pre.MIN_MINUTES].copy()
    df["group"] = df.position.map(sp.POSITION_GROUP)
    df = df[df.group.isin(["Middles", "Edge"])]
    print(f"{len(df):,} ratable matches for Middles and Edge "
          f"({df.player_id.nunique():,} players)\n")

    results = []
    for name, w in CANDIDATES.items():
        agg = season_composites(df, w)
        r = independent_r(agg)
        for row in r.itertuples(index=False):
            results.append(dict(candidate=name, group=row.group, pairs=row.pairs,
                                players=row.players, independent_r=row.r))
        print(f"{name:40s} " + "  ".join(f"{x.group} r={x.r:.3f}"
                                         for x in r.itertuples(index=False)))
    res = pd.DataFrame(results)
    wide = res.pivot(index="candidate", columns="group",
                     values="independent_r").reset_index()
    wide["mean_r"] = wide[["Edge", "Middles"]].mean(axis=1)
    wide = wide.sort_values("mean_r", ascending=False)

    counts = res.groupby("group")[["pairs", "players"]].max().reset_index()

    wt = pd.DataFrame({"rate": RATE, "Prop (old)": PROP, "Back Row (old)": BACKROW})

    W = ["# Weight profiles for Middles and Edge\n",
         "The rating engine kept a position map of its own — Prop against Back Row "
         "(second row and lock together) — while `sp_schema` had already moved to "
         "Middles (prop and lock) and Edge (second row alone) at Leeds's request. "
         "Unifying them leaves the two new groups needing weight vectors, and the "
         "retired Back Row profile does not transfer cleanly, because it was fitted to "
         "a pool that included the locks now leaving it.\n",
         "\nEach candidate is scored by the **independent year-to-year correlation** of "
         "its composite: a player's season mean against his next season's, sharing no "
         "match, before any shrinkage. It measures how much of what the weighting picks "
         "up is a trait the player repeats rather than noise.\n",
         f"\nComputed on {len(df):,} ratable matches, seasons through 2025, "
         f"minimum {MIN_MATCHES} matches in each season of a pair.\n",
         "\n## Candidates\n", md(wide),
         "\n\nSample sizes are the same for every candidate: "
         + ", ".join(f"{r.group} {int(r.pairs)} season pairs over {int(r.players)} "
                     f"players" for r in counts.itertuples(index=False)) + ".\n",
         "\n## The two retired profiles, for reference\n", md(wt, "{:.2f}"),
         "\n\n## Reading this\n",
         "Differences of a few thousandths are not a decision. Take the simplest "
         "candidate whose correlation is not meaningfully below the best, and record "
         "the choice as a calibration decision rather than a discovery — reliability "
         "says a weighting is measuring *something* consistently, not that it is "
         "measuring the right thing. The component weights above are the part a coach "
         "can disagree with, and that disagreement should win.\n"]
    open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(W) + "\n")
    print("\n" + wide.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
