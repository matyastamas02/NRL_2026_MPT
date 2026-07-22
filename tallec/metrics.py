# -*- coding: utf-8 -*-
"""Metric config engine — the 'Rate section' Mike specified.

Every stat has TWO end forms:
  * Volume — the raw count (how many of the thing the player did)
  * Rate   — Volume / denominator (how many per x), denominator chosen per the
             metric dictionary's rate family.

This module reads the dictionary Mike supplied (loaded into tallec.db as
`metric_dictionary`, `rate_rules`, `overlap_rules`) and turns it into callable
config: given a canonical Stats Perform field, return its volume + rate recipe.
When Mike approves/adjusts a denominator, it's a DATA edit in that table — no
code change. Overlap rules are surfaced so we never sum double-counted metrics.
"""
import sqlite3, os
import pandas as pd

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tallec.db")


def _con():
    c = sqlite3.connect(DB)
    return c


def load_dictionary():
    con = _con()
    d = pd.read_sql("SELECT * FROM metric_dictionary", con)
    con.close()
    return d


def rate_recipe(metric_field):
    """Return {volume, rate_name, formula, denominator, rate_type, confidence,
    overlap_risk} for a canonical field, or None if not in the dictionary."""
    d = load_dictionary()
    row = d[d["Metric / Raw Field"] == metric_field]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "metric": metric_field,
        "volume": r["Volume Definition"],
        "rate_name": r["Suggested Rate Name"],
        "formula": r["Rate Formula"],
        "denominator": r["Denominator"],
        "rate_type": r["Rate Type"],
        "confidence": r["Confidence"],
        "overlap_risk": r["Overlap / Double-count Risk"],
    }


def compute_volume_and_rate(df, metric_field):
    """Given a player_match_raw DataFrame, return (volume_series, rate_series)
    for one metric, using the dictionary's denominator. Rate is NaN where the
    metric has no rate (identifiers/exposure) or the denominator is 0/missing."""
    rec = rate_recipe(metric_field)
    if rec is None or metric_field not in df.columns:
        return None, None
    vol = pd.to_numeric(df[metric_field], errors="coerce")
    denom_col = rec["denominator"]
    if not denom_col or pd.isna(denom_col) or denom_col not in df.columns:
        return vol, None
    denom = pd.to_numeric(df[denom_col], errors="coerce")
    rate = vol.where(denom > 0) / denom.where(denom > 0)
    return vol, rate


def overlap_pairs():
    """Metrics the dictionary says must NOT be summed together."""
    con = _con()
    o = pd.read_sql("SELECT * FROM overlap_rules", con)
    con.close()
    return o


def rate_families():
    con = _con()
    r = pd.read_sql("SELECT * FROM rate_rules", con)
    con.close()
    return r


def review_status():
    """Summary of what still needs domain sign-off (all rows start as Review)."""
    d = load_dictionary()
    return {
        "total": len(d),
        "by_decision": d["Decision"].value_counts().to_dict(),
        "by_confidence": d["Confidence"].value_counts().to_dict(),
        "low_confidence": d[d["Confidence"] == "Low"]["Metric / Raw Field"].tolist(),
    }


if __name__ == "__main__":
    print("-- review status --")
    st = review_status()
    print(f"  {st['total']} metrics | confidence: {st['by_confidence']}")
    print(f"  low-confidence (need most review): {st['low_confidence']}")

    print("\n-- example recipes --")
    for m in ["Tackle Break", "Line Break", "Try Assists", "Ball Runs - Metres Gained"]:
        r = rate_recipe(m)
        if r:
            print(f"  {m}: Volume='{r['volume']}' | Rate='{r['rate_name']}' "
                  f"= {r['formula']} ({r['rate_type']})")

    print("\n-- overlap traps (do not sum) --")
    for _, o in overlap_pairs().iterrows():
        print(f"  {o['Overlap family']}: avoid '{o['Do not calculate']}'")

    # demo compute on real data
    con = _con()
    raw = pd.read_sql("SELECT * FROM player_match_raw WHERE Competition='SL' AND Season=2026", con)
    con.close()
    vol, rate = compute_volume_and_rate(raw, "Tackle Break")
    print(f"\n-- SL 2026 'Tackle Break': volume mean {vol.mean():.2f}, "
          f"rate (per run) mean {rate.mean():.3f} --")
