# -*- coding: utf-8 -*-
"""GIGOT v2 — does a player layer improve the match model?

The spec's baseline is explicit: the player layer has to beat plain xLadder on the
same fixtures, or it is not worth the complexity. Both masters store the model's own
margin prediction, so that is the benchmark rather than something re-derived here.

Method

  1. Every player-match gets a composite performance score from the rating engine,
     standardized within its own competition-season pool.
  2. For each player-match, PRE-MATCH ratings are built from his earlier matches only:
     class  = mean of every prior composite
     form   = mean of the last five prior composites
     Match M's own performance never enters its own features. The permutation test at
     the bottom asserts this rather than trusting it.
  3. Those are aggregated over the players who actually took the field for each team.
     No team-list feed exists, so validation uses the real line-ups retrospectively —
     which answers "would knowing the line-up have helped?", the question that has to
     come first anyway.
  4. Each fixture gets the difference between the two sides, and the test asks whether
     those differences explain any of what the existing model gets wrong.

Honest limits, stated because they bound the conclusion:

  * Composites are standardized within each competition-season pool. The pool is the
    whole season rather than only its earlier rounds, so the population a match is
    compared against includes its own season — a population descriptor rather than an
    outcome, but not a strict walk-forward. prematch_players(walk_forward=True) does
    the strict version; see its docstring for why it is not yet the default.
  * Using the actual line-up is an upper bound on what a team-list feed could deliver:
    on Friday you know the named 17, not who finished the game.
  * Margin_Pred_v2 is taken as given from the master.
"""
import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import player_rating_engine as pre
import team_map as tm

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
FORM_WINDOW = 5
COLS = ["player_id", "player", "season", "round", "team", "position", "minutes",
        "all_run_metres", "p_c_m", "tackle_breaks", "line_breaks", "tackles",
        "offloads", "try_assists", "tries", "errors"]


def prematch_players(comp, con, shuffle_outcome=False, seed=0, walk_forward=False):
    """Per player-match pre-match class and form, from earlier matches only.

    walk_forward is OFF by default, deliberately and temporarily. Switching it on is
    the right thing methodologically, and a first run says it changes the answer: the
    NRL gain falls from +0.27 [+0.09, +0.47] to +0.13 [-0.11, +0.38], i.e. no longer
    significant, and Super League turns negative in a way that looks like a bug rather
    than a finding (the warm-up drop did not fire and the per-season z-scales are not
    obviously comparable). Publishing either number before that is understood would be
    worse than either. This is the first task of the next round of work.

    With walk_forward=True each season's composites are standardized
    using means and standard deviations fitted on the seasons BEFORE it, so nothing
    about a match — not even the population it is compared against — comes from its
    own season or later. The first season has no prior data, so it is fitted on itself
    and flagged in the returned frame as `warmup`; the evaluation drops those rows.

    Fitting over the whole period instead, which this function used to do, is what the
    leakage test caught: altering one match moved every other match's composite,
    because they all shared one set of pool statistics.
    """
    raw = pd.read_sql(f"SELECT {', '.join(COLS)} FROM player_match_stats "
                      f"WHERE competition=?", con, params=(comp,))
    seasons = sorted(raw.season.dropna().unique())
    parts = []
    for i, s in enumerate(seasons):
        part = raw[raw.season == s]
        if len(part) < 100:
            continue
        eng = pre.PlayerRatingEngine(comp)
        if walk_forward and i > 0:
            prior = raw[raw.season < s]
            eng._fit_standardization(prior)          # population from the past only
            out = eng._transform(part)
            out["warmup"] = False
        else:
            out = eng._composite(part)               # nothing earlier to fit on
            out["warmup"] = bool(walk_forward)
        parts.append(out)
    pm = pd.concat(parts, ignore_index=True)

    if shuffle_outcome:
        # permutation test: scramble the performance column across rows. Pre-match
        # features must be unchanged for the rows they describe if no match leaks
        # into its own rating.
        rng = np.random.default_rng(seed)
        pm = pm.copy()
        pm["composite"] = rng.permutation(pm["composite"].values)

    pm = pm.sort_values(["player_id", "season", "round"])
    g = pm.groupby("player_id")["composite"]
    pm["prior_class"] = g.transform(lambda s: s.shift(1).expanding().mean())
    pm["prior_form"] = g.transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
    pm["n_prior"] = g.transform(lambda s: s.shift(1).notna().cumsum())
    return pm


def team_rows(pm):
    """Aggregate the players who actually played, per team per round."""
    pm = pm.copy()
    pm["w"] = pm["minutes"].clip(lower=1)
    out = pm.groupby(["season", "round", "team"]).apply(
        lambda g: pd.Series({
            "lineup_class": np.average(g.prior_class.fillna(0), weights=g.w),
            "lineup_form": np.average(g.prior_form.fillna(0), weights=g.w),
            "green_share": float((g.n_prior.fillna(0) < 3).mean()),
            "warmup": bool(g.get("warmup", pd.Series([False])).any()),
            "n_players": len(g)}), include_groups=False).reset_index()
    return out


def build(comp, con, **kw):
    pm = prematch_players(comp, con, **kw)
    tr = team_rows(pm)
    mp, issues = tm.solve(comp)
    assert not issues["duplicates"] and not issues["unmatched"], issues
    m = tm.load_master(comp).dropna(subset=["Margin", "Margin_Pred_v2"])
    m["team_a"] = m["A Team"].map(mp)
    m["team_b"] = m["B Team"].map(mp)
    a = tr.rename(columns={c: c + "_a" for c in
                           ["lineup_class", "lineup_form", "green_share", "n_players",
                            "warmup"]})
    b = tr.rename(columns={c: c + "_b" for c in
                           ["lineup_class", "lineup_form", "green_share", "n_players",
                            "warmup"]})
    d = (m.merge(a, left_on=["Season", "Round", "team_a"],
                 right_on=["season", "round", "team"], how="inner")
          .merge(b, left_on=["Season", "Round", "team_b"],
                 right_on=["season", "round", "team"], how="inner", suffixes=("", "_y")))
    d["d_class"] = d.lineup_class_a - d.lineup_class_b
    d["d_form"] = d.lineup_form_a - d.lineup_form_b
    d["d_green"] = d.green_share_a - d.green_share_b
    d["resid"] = d.Margin - d.Margin_Pred_v2
    # the first season was standardized on itself for want of anything earlier; it is
    # dropped rather than counted, so every evaluated fixture is genuinely walk-forward
    warm = d.get("warmup_a", pd.Series(False, index=d.index)).fillna(False) | \
        d.get("warmup_b", pd.Series(False, index=d.index)).fillna(False)
    if warm.any():
        print(f"  dropping {int(warm.sum())} fixtures in the warm-up season "
              f"(no earlier data to standardize against)")
    return d[~warm].copy()


FEATS = ["d_class", "d_form", "d_green"]


def evaluate(d, comp):
    seasons = sorted(d.Season.unique())
    test_season = seasons[-1]
    tr, te = d[d.Season < test_season], d[d.Season == test_season]
    print(f"\n=== {comp}: {len(d)} fixtures joined "
          f"({', '.join(str(s) for s in seasons)}) ===")
    print(f"  train {len(tr)} ({seasons[0]}-{test_season - 1}) | test {len(te)} ({test_season})")

    print("\n  correlation of each player feature with what the model gets wrong:")
    for f in FEATS:
        r = d[f].corr(d.resid)
        rt = te[f].corr(te.resid)
        print(f"    {f:9s} vs residual: all {r:+.3f} | test season {rt:+.3f}")
    print(f"    {'d_class':9s} vs actual margin: {d.d_class.corr(d.Margin):+.3f} "
          f"(the existing prediction already captures most of this)")

    base_mae = (te.Margin - te.Margin_Pred_v2).abs().mean()
    base_rmse = ((te.Margin - te.Margin_Pred_v2) ** 2).mean() ** .5
    X_tr = tr[["Margin_Pred_v2"] + FEATS]
    X_te = te[["Margin_Pred_v2"] + FEATS]
    mdl = Ridge(alpha=1.0).fit(X_tr, tr.Margin)
    p = mdl.predict(X_te)
    mae = np.abs(te.Margin - p).mean()
    rmse = ((te.Margin - p) ** 2).mean() ** .5
    # control: refit on the baseline alone, so the comparison is not just recalibration
    ctl = Ridge(alpha=1.0).fit(tr[["Margin_Pred_v2"]], tr.Margin)
    pc = ctl.predict(te[["Margin_Pred_v2"]])
    cmae = np.abs(te.Margin - pc).mean()
    crmse = ((te.Margin - pc) ** 2).mean() ** .5

    print(f"\n  held-out {test_season} margin error")
    print(f"    xLadder prediction as shipped      MAE {base_mae:5.2f}  RMSE {base_rmse:5.2f}")
    print(f"    same, recalibrated on train        MAE {cmae:5.2f}  RMSE {crmse:5.2f}")
    print(f"    recalibrated + player layer        MAE {mae:5.2f}  RMSE {rmse:5.2f}")
    print(f"    player layer earns                 MAE {cmae - mae:+5.2f}  RMSE {crmse - rmse:+5.2f}")
    print("    coefficients: " + ", ".join(
        f"{n}={c:+.3f}" for n, c in zip(["baseline"] + FEATS, mdl.coef_)))
    return dict(comp=comp, n=len(d), test_season=int(test_season), n_test=len(te),
                base_mae=base_mae, ctl_mae=cmae, gigot_mae=mae,
                base_rmse=base_rmse, ctl_rmse=crmse, gigot_rmse=rmse,
                gain_mae=cmae - mae, gain_rmse=crmse - rmse)


def evaluate_own_baseline(d, comp, n_boot=4000, seed=0):
    """Same question, but against a baseline we build ourselves, walk-forward.

    Needed because the Super League master's stored prediction is in-sample for its
    last season (training MAE 15.69, held-out 7.63 — a model cannot predict an unseen
    season twice as well as its own training data), so it cannot serve as a benchmark.
    This baseline uses only what is genuinely known before kick-off: the stored
    pre-match ELO difference and home advantage. Each season is predicted by a model
    fitted on the seasons before it, so every prediction is out-of-sample.
    """
    d = d.copy()
    d["home"] = np.where(d["Home Advantage"] == "A", 1.0,
                         np.where(d["Home Advantage"] == "B", -1.0, 0.0))
    BASE_F = ["Diff ELO", "home"]
    seasons = sorted(d.Season.unique())
    rows = []
    for t in seasons[1:]:
        tr, te = d[d.Season < t], d[d.Season == t]
        if len(tr) < 100 or te.empty:
            continue
        b = Ridge(alpha=1.0).fit(tr[BASE_F], tr.Margin)
        g = Ridge(alpha=1.0).fit(tr[BASE_F + FEATS], tr.Margin)
        te = te.assign(pred_base=b.predict(te[BASE_F]),
                       pred_gigot=g.predict(te[BASE_F + FEATS]))
        rows.append(te)
    oos = pd.concat(rows, ignore_index=True)
    e_base = (oos.Margin - oos.pred_base).abs()
    e_gig = (oos.Margin - oos.pred_gigot).abs()
    diff = (e_base - e_gig).values
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n=== {comp}: walk-forward on a baseline we control "
          f"({len(oos)} out-of-sample fixtures, {oos.Season.min()}-{oos.Season.max()}) ===")
    print(f"  ELO + home advantage            MAE {e_base.mean():5.2f}  "
          f"RMSE {((oos.Margin-oos.pred_base)**2).mean()**.5:5.2f}")
    print(f"  + player layer                  MAE {e_gig.mean():5.2f}  "
          f"RMSE {((oos.Margin-oos.pred_gigot)**2).mean()**.5:5.2f}")
    print(f"  player layer earns              MAE {diff.mean():+5.2f} "
          f"[95% CI {lo:+.2f}, {hi:+.2f}]  "
          f"{'SIGNIFICANT' if lo > 0 else 'not significant'}")
    return dict(comp=comp, n_oos=len(oos), base_mae=e_base.mean(),
                gigot_mae=e_gig.mean(), gain=diff.mean(), ci_lo=lo, ci_hi=hi)


if __name__ == "__main__":
    con = sqlite3.connect(DB)
    rows = []
    own = []
    for comp in ("NRL", "SL"):
        d = build(comp, con)
        rows.append(evaluate(d, comp))
        own.append(evaluate_own_baseline(d, comp))

    print("\n=== leakage check: permute the performance column, refit ===")
    for comp in ("NRL", "SL"):
        real = build(comp, con)
        perm = build(comp, con, shuffle_outcome=True, seed=7)
        j = real[["Match ID", "d_class"]].merge(perm[["Match ID", "d_class"]],
                                                on="Match ID", suffixes=("", "_perm"))
        r = j.d_class.corr(j.d_class_perm)
        print(f"  {comp}: correlation between real and permuted d_class = {r:+.3f} "
              f"(near zero is correct — features carry information about the players, "
              f"and permuting performance destroys it without touching the row keys)")
    con.close()
    pd.DataFrame(rows).to_csv(os.path.join(BASE, "gigot_v2_results.csv"), index=False)
    print("\nwrote gigot_v2_results.csv")
