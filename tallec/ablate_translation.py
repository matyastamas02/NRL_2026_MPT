# -*- coding: utf-8 -*-
"""Which features of the conditional translation model actually earn their place.

The 2026-09-22 external review reported that age, minutes per game and games played add
nothing on a rolling backtest and slightly hurt, and that the model's strength is
regression to the mean plus competition pair and position. That is a claim about what
should ship, so it is checked here rather than accepted.

Each specification is fitted and scored exactly as `rolling_backtest.py` does — same
FeatureSpec, same Ridge alpha, same standardisation, same rolling origins — with columns
zeroed out after the spec is built rather than rebuilt, so every specification sees an
identical design apart from the features under test. Differences are reported against the
full model with a player-clustered bootstrap, because the same player appears in several
rows and a row bootstrap would understate the interval.

    python ablate_translation.py
    python ablate_translation.py --origins 2024,2025
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import rolling_backtest as rb
import translation_features as tf

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
OUT = os.path.join(BASE, "ABLATION_REPORT.md")

AGE = ["age", "age_delta", "age_missing", "age_delta_missing"]
LOAD = ["mins_pg", "games_src", "mins_pg_missing", "games_src_missing"]

# name -> the feature families it is allowed to use, beyond class_source
SPECS = {
    "source only": [],
    "source + pair": ["pair"],
    "source + position": ["pos"],
    "source + pair + position": ["pair", "pos"],
    "+ load (mins, games)": ["pair", "pos", "load"],
    "+ age": ["pair", "pos", "age"],
    "+ position x pair (pooled)": ["pair", "pos", "interaction"],
    "full model": ["pair", "pos", "load", "age"],
    "everything incl. interaction": ["pair", "pos", "load", "age", "interaction"],
}


def keep_mask(columns, families):
    """Columns to retain. Anything outside the allowed families is zeroed, not dropped,
    so the design matrix keeps its shape and the alpha penalises a comparable space."""
    keep = []
    for c in columns:
        if c == "class_source":
            keep.append(True)
        elif c.startswith("px_"):
            keep.append("interaction" in families)
        elif c.startswith("pair_"):
            keep.append("pair" in families)
        elif c.startswith("grp_") or c == "pos_missing":
            keep.append("pos" in families)
        elif c in AGE:
            keep.append("age" in families)
        elif c in LOAD:
            keep.append("load" in families)
        else:
            keep.append(True)          # anything unrecognised stays in every spec
    return np.array(keep)


def fit_spec(pairs, layer, families):
    p = pairs[pairs.layer == layer]
    if len(p) < 40:
        return None
    spec = tf.FeatureSpec.fit(p, pairs=sorted(p.pair.unique()))
    X, _ = spec.transform(p)
    m = keep_mask(list(X.columns), families)
    Xm = X.copy()
    Xm.loc[:, ~m] = 0.0
    sc = StandardScaler().fit(Xm)
    mult = tf.pooling_multipliers(list(X.columns))
    model = Ridge(alpha=rb.ALPHA).fit(sc.transform(Xm) * mult, p.class_target.values)
    return dict(spec=spec, model=model, scaler=sc, mask=m, mult=mult)


def predict_spec(f, rows):
    X, _ = f["spec"].transform(rows)
    Xm = X.copy()
    Xm.loc[:, ~f["mask"]] = 0.0
    return np.clip(f["model"].predict(f["scaler"].transform(Xm) * f["mult"]), 0, 100)


def cluster_boot(d, col_a, col_b, n=4000, seed=0):
    """Bootstrap the paired difference in absolute error, resampling PLAYERS."""
    ok = d[col_a].notna() & d[col_b].notna() & d.class_target.notna()
    g = d[ok]
    diff = ((g.class_target - g[col_a]).abs() - (g.class_target - g[col_b]).abs()).values
    players = g.player_id.values
    uniq = pd.unique(players)
    idx = {p: np.where(players == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    s = np.empty(n)
    for i in range(n):
        pick = rng.choice(uniq, uniq.size, replace=True)
        s[i] = np.concatenate([diff[idx[p]] for p in pick]).mean()
    return float(diff.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", default="2023,2024,2025")
    a = ap.parse_args()
    origins = [int(x) for x in a.origins.split(",")]

    con = sqlite3.connect(DB)
    got = []
    for origin in origins:
        print(f"origin {origin} ...", flush=True)
        pairs, cum, ext, pos, dob = rb.knowledge_at(con, origin - 1)
        pend = rb.moves_into(con, origin, cum, ext, pos, dob)
        if pend.empty:
            continue
        out = pend.copy()
        out["origin"] = origin
        out["cohort"] = out["transition_type"]
        for name, fam in SPECS.items():
            f = fit_spec(pairs, "B_next_season", fam)
            if f is None:
                continue
            out[name] = predict_spec(f, pend)
        got.append(out)
    con.close()
    d = pd.concat(got, ignore_index=True)

    names = [n for n in SPECS if n in d.columns]
    rows = []
    for n in names:
        e = d.class_target - d[n]
        rows.append(dict(specification=n, n=int(e.notna().sum()),
                         mae=e.abs().mean(), rmse=float((e ** 2).mean() ** 0.5)))
    table = pd.DataFrame(rows).sort_values("mae")

    diffs = []
    for n in names:
        if n == "full model":
            continue
        b = cluster_boot(d, n, "full model")
        diffs.append(dict(specification=n, mae_gap_vs_full=b[0],
                          ci_low=b[1], ci_high=b[2]))
    diffs = pd.DataFrame(diffs)

    W = ["# Which translation features earn their place\n",
         "Every specification is fitted and scored exactly as `rolling_backtest.py` "
         "fits and scores the live model, over the same rolling origins. Features "
         "outside a specification are zeroed after the design matrix is built, so the "
         "comparison is between identical designs rather than between differently "
         "shaped ones.\n",
         f"\n{len(d):,} moves over origins {origins}.\n",
         "\n## Error by specification\n", md(table),
         "\n\n## Against the full model\n",
         "Negative means the reduced specification is **better**. The interval is a "
         "bootstrap over players, not rows, because the same man appears in several "
         "moves and a row bootstrap would be too narrow.\n", md(diffs)]

    # Per origin, because a mean over three seasons can hide a specification that has
    # stopped working. The fourth external review made this the stronger argument for
    # keeping the interaction switched off: it helps in the two older origins and hurts
    # in the most recent one, which is not what a real effect looks like.
    per = []
    for n in names:
        row = dict(specification=n)
        for origin, g in d.groupby("origin"):
            e_n = (g.class_target - g[n]).abs().mean()
            e_f = (g.class_target - g["full model"]).abs().mean()
            row[str(origin)] = e_f - e_n          # positive = better than the full model
        per.append(row)
    W.append("\n\n## The same comparison, origin by origin\n")
    W.append("Positive means the specification beats the full model in that season. A "
             "mean over three origins can hide something that has stopped working, and "
             "for the interaction term it does.\n")
    W.append(md(pd.DataFrame(per), "{:+.3f}"))

    best = table.iloc[0]
    full = table[table.specification == "full model"].iloc[0]
    W.append(f"\n\n## What this says\n")
    if best.specification != "full model":
        W.append(f"The best specification is **{best.specification}** at "
                 f"{best.mae:.3f} MAE, against the full model's {full.mae:.3f}.")
    sig = diffs[(diffs.ci_high < 0)]
    if len(sig):
        W.append("Specifications beating the full model with an interval clear of zero: "
                 + ", ".join(f"**{r.specification}** ({r.mae_gap_vs_full:+.3f} "
                             f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}])"
                             for r in sig.itertuples()) + ".")
    else:
        W.append("No reduced specification beats the full model with an interval clear "
                 "of zero.")

    for cohort in ["first", "returning", "continuing"]:
        g = d[d.cohort == cohort]
        if len(g) < 30:
            continue
        r = []
        for n in names:
            e = g.class_target - g[n]
            r.append(dict(specification=n, n=int(e.notna().sum()),
                          mae=e.abs().mean()))
        W.append(f"\n\n## Cohort: {cohort}\n")
        W.append(md(pd.DataFrame(r).sort_values("mae")))

    open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(W) + "\n")
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + diffs.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
