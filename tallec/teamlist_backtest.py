# -*- coding: utf-8 -*-
"""Does the player layer still earn its keep when you only know the squad?

Every number the project has produced about the player layer was measured on the
players who actually took the field, with the minutes they actually played. That is an
upper bound and has always been labelled one: on a Friday you know a named squad, not
who finished the game, and not who limped off at twenty minutes.

The honest test needs a team-list feed nobody has. But the question can still be
answered from the other side. Instead of the seventeen who played round R, take the
players who played for that club in the rounds *before* R, weighted by the minutes they
were getting. That is knowable on the Friday — it is roughly what a coach's squad
announcement contains, minus the coach's private information. It misses debutants and
includes players who turn out to be dropped or injured, which is precisely the error a
real team sheet would carry.

So the two runs bracket the truth:

  actual line-up    what has been reported so far — an upper bound
  recent squad      this — a lower bound, since a real team sheet is better information
                    than "whoever played last week"

If the gain survives the lower bound, it survives a team-list feed, and the request to
Mike stops being a precondition for claiming anything.

    python teamlist_backtest.py                 # both competitions, window of 3
    python teamlist_backtest.py --window 4
    python teamlist_backtest.py --sl-master <path to a repaired SL master>
"""
import argparse
import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import gigot_v2 as g
import team_map as tm

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
FEATS = g.FEATS


def with_cumulative(pm):
    """Each player's rating as it stands *after* each of his matches.

    `prior_class` in gigot_v2 is the value going into a match, which is what that
    evaluation needs. Here the squad is assembled from earlier rounds, so what is wanted
    is the value coming out of the most recent one — everything known by the Friday.
    """
    pm = pm.sort_values(["player_id", "season", "round"]).copy()
    grp = pm.groupby("player_id")["composite"]
    pm["cum_class"] = grp.transform(lambda s: s.expanding().mean())
    pm["cum_form"] = grp.transform(
        lambda s: s.rolling(g.FORM_WINDOW, min_periods=1).mean())
    pm["n_played"] = grp.transform(lambda s: s.notna().cumsum())
    return pm


def squad_rows(pm, window):
    """Per club per round, the squad implied by the previous `window` rounds.

    A player enters through his most recent appearance inside the window, carrying the
    rating he had after it; his weight is the minutes he was getting, which stands in
    for the minutes he is about to get. Nothing from round R itself is read.
    """
    pm = pm.dropna(subset=["round"]).copy()
    pm["round"] = pm["round"].astype(int)
    out = []
    for (season, team), gg in pm.groupby(["season", "team"]):
        rounds = sorted(gg["round"].unique())
        for r in rounds:
            prev = gg[(gg["round"] >= r - window) & (gg["round"] < r)]
            if prev.empty:
                continue
            # one row per player: his latest appearance in the window
            last = prev.sort_values("round").groupby("player_id").tail(1)
            mins = prev.groupby("player_id")["minutes"].mean()
            w = last.player_id.map(mins).clip(lower=1).values
            out.append(dict(
                season=season, round=r, team=team,
                lineup_class=np.average(last.cum_class.fillna(0), weights=w),
                lineup_form=np.average(last.cum_form.fillna(0), weights=w),
                green_share=float((last.n_played.fillna(0) < 3).mean()),
                warmup=bool(last.get("warmup", pd.Series([False])).any()),
                n_players=len(last)))
    return pd.DataFrame(out)


def build(comp, con, window, actual=False):
    """Fixture-level differences, from either the real line-up or the recent squad."""
    pm = g.prematch_players(comp, con)
    tr = g.team_rows(pm) if actual else squad_rows(with_cumulative(pm), window)
    mp, issues = tm.solve(comp)
    assert not issues["duplicates"] and not issues["unmatched"], issues
    m = tm.load_master(comp).dropna(subset=["Margin"])
    m["team_a"] = m["A Team"].map(mp)
    m["team_b"] = m["B Team"].map(mp)
    cols = ["lineup_class", "lineup_form", "green_share", "n_players", "warmup"]
    a = tr.rename(columns={c: c + "_a" for c in cols})
    b = tr.rename(columns={c: c + "_b" for c in cols})
    d = (m.merge(a, left_on=["Season", "Round", "team_a"],
                 right_on=["season", "round", "team"], how="inner")
          .merge(b, left_on=["Season", "Round", "team_b"],
                 right_on=["season", "round", "team"], how="inner", suffixes=("", "_y")))
    d["d_class"] = d.lineup_class_a - d.lineup_class_b
    d["d_form"] = d.lineup_form_a - d.lineup_form_b
    d["d_green"] = d.green_share_a - d.green_share_b
    warm = (d.get("warmup_a", pd.Series(False, index=d.index)).fillna(False)
            | d.get("warmup_b", pd.Series(False, index=d.index)).fillna(False))
    return d[~warm].copy()


def evaluate(d, label, n_boot=4000, seed=0):
    """Walk-forward against a baseline of pre-match ELO and home advantage."""
    d = d.copy()
    d["home"] = np.where(d["Home Advantage"] == "A", 1.0,
                         np.where(d["Home Advantage"] == "B", -1.0, 0.0))
    BF = ["Diff ELO", "home"]
    rows, per_season = [], []
    for t in sorted(d.Season.unique())[1:]:
        tr, te = d[d.Season < t], d[d.Season == t]
        if len(tr) < 100 or te.empty:
            continue
        b = Ridge(alpha=1.0).fit(tr[BF], tr.Margin)
        gg = Ridge(alpha=1.0).fit(tr[BF + FEATS], tr.Margin)
        te = te.assign(pred_base=b.predict(te[BF]),
                       pred_gigot=gg.predict(te[BF + FEATS]))
        rows.append(te)
        eb = (te.Margin - te.pred_base).abs()
        eg = (te.Margin - te.pred_gigot).abs()
        per_season.append((int(t), len(te), eb.mean(), eg.mean(), eb.mean() - eg.mean()))
    if not rows:
        print(f"  {label}: nothing to evaluate")
        return None
    oos = pd.concat(rows, ignore_index=True)
    diff = ((oos.Margin - oos.pred_base).abs() - (oos.Margin - oos.pred_gigot).abs()).values
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  {label}")
    print(f"    {len(oos)} out-of-sample fixtures, "
          f"mean squad size {oos.n_players_a.mean():.1f}")
    print(f"    baseline MAE {(oos.Margin - oos.pred_base).abs().mean():5.2f}   "
          f"+ player layer {(oos.Margin - oos.pred_gigot).abs().mean():5.2f}")
    print(f"    earns {diff.mean():+5.2f} [95% CI {lo:+.2f}, {hi:+.2f}]  "
          f"{'SIGNIFICANT' if lo > 0 else 'not significant'}")
    for t, n, a_, c_, gn in per_season:
        print(f"       {t}  n={n:<4} baseline {a_:5.2f}  + layer {c_:5.2f}  {gn:+5.2f}")
    return dict(label=label, n=len(oos), gain=diff.mean(), lo=lo, hi=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=3,
                    help="how many previous rounds make up the known squad")
    ap.add_argument("--sl-master", default=None,
                    help="use a different Super League master (e.g. the repaired one)")
    ap.add_argument("--comps", default="NRL,SL")
    a = ap.parse_args()

    if a.sl_master:
        tm.MASTERS["SL"] = a.sl_master
        print(f"Super League master: {a.sl_master}")
    con = sqlite3.connect(DB)
    out = []
    for comp in a.comps.split(","):
        comp = comp.strip()
        print(f"\n{'=' * 74}\n{comp}: what the player layer earns, "
              f"two ways of knowing the side\n{'=' * 74}")
        out.append(evaluate(build(comp, con, a.window, actual=True),
                            "the line-up that actually played  (upper bound)"))
        out.append(evaluate(build(comp, con, a.window),
                            f"the squad from the previous {a.window} rounds  (lower bound)"))
    con.close()


if __name__ == "__main__":
    main()
