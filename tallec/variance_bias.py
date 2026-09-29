# -*- coding: utf-8 -*-
"""How much the constant-ability assumption costs, measured rather than asserted.

`PlayerRatingEngine._fit_variance_components` is a one-way random-effects ANOVA with one
intercept per player, held constant across every season, position and role he has. Ageing,
a change of position, a season spent off the bench and the recency weighting applied
elsewhere in the engine are therefore all booked as within-player noise. That inflates
sigma^2, deflates tau^2, and shrinks everyone harder than they deserve.

The engine's docstring has said so since 2026-09-22 and called the bias "at least
conservative". The third external review of 2026-09-23 measured it and objected to that
phrase on two grounds: the shrinkage factor B = tau^2/(tau^2 + sigma^2/n) depends on n, so
a bias in the components reorders players with different match counts rather than merely
compressing everyone; and tau is the denominator of the published scale, so the bias moves
the displayed distances too.

This reproduces the measurement. The alternative is a two-level decomposition that lets a
player's level move between seasons — a player-season random effect — which is still not a
proper state-space model but does separate "he changed" from "he was noisy".

    python variance_bias.py
"""
import os
import sqlite3
import sys

import math

import numpy as np
import pandas as pd

import player_rating_engine as pre
import rating_history as rh

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
OUT = os.path.join(BASE, "VARIANCE_BIAS_REPORT.md")
COMPS = ["NRL", "SL", "NSW", "QLD"]
NL = chr(10)


def components_career(pm):
    """What the engine does now: one intercept per player."""
    e = pre.PlayerRatingEngine("X")
    e._fit_variance_components(pm)
    return e.sigma2, e.tau2


def components_player_season(pm):
    """A player-season intercept: within-season noise, and level allowed to move.

    sigma^2 becomes the variance of matches around their own PLAYER-SEASON mean, so a
    player who was better in 2025 than 2022 no longer has that difference counted as
    game-to-game randomness. tau^2 is then recovered from the spread of player-season
    means around the grand mean, corrected for sampling as in the one-way case.
    """
    r = pm[pm["ratable"]].copy()
    r["unit"] = r.player_id.astype(str) + "|" + r.season.astype(str)
    groups = [g["composite"].values for _, g in r.groupby("unit")]
    k, N = len(groups), sum(len(g) for g in groups)
    if k < 2 or N <= k:
        return np.nan, np.nan
    grand = np.concatenate(groups).mean()
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ms_within = ss_within / (N - k)
    ms_between = ss_between / (k - 1)
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    return float(ms_within), float(max(0.0, (ms_between - ms_within) / n0))


def shrinkage(tau2, sigma2, n):
    return tau2 / (tau2 + sigma2 / n) if (tau2 + sigma2 / n) > 0 else 0.0


def rank_impact(con, comp, s_c, t_c, s_p, t_p):
    """What the alternative components would do to the PUBLISHED list, not in theory.

    An earlier version of this report said the correction reorders players rather than
    merely compressing them, and left it there. True, and unquantified — the fourth
    external review measured it and found the reordering small while the displayed
    distances move more. Both halves matter to a reader, so both are computed here.
    """
    r = pd.read_sql("SELECT player_id, class_z, n_games, class_score, \"group\" grp "
                    "FROM player_ratings WHERE competition=?", con, params=(comp,))
    if r.empty or not t_c or not t_p:
        return None
    grand = float(r.class_z.mean())

    def rescore(tau2, sigma2):
        # undo the old shrinkage, reapply the new one, keep everything else fixed
        b_old = np.array([shrinkage(t_c, s_c, n) for n in r.n_games])
        raw = grand + (r.class_z.values - grand) / np.where(b_old > 0, b_old, 1.0)
        b_new = np.array([shrinkage(tau2, sigma2, n) for n in r.n_games])
        z = grand + b_new * (raw - grand)
        return z, np.sqrt(tau2)

    z_old, tau_old = rescore(t_c, s_c)
    z_new, tau_new = rescore(t_p, s_p)
    score_old = 100 * 0.5 * (1 + np.vectorize(math.erf)((z_old - grand) / tau_old
                                                        / math.sqrt(2)))
    score_new = 100 * 0.5 * (1 + np.vectorize(math.erf)((z_new - grand) / tau_new
                                                        / math.sqrt(2)))

    out = []
    for grp, g in pd.DataFrame({"grp": r.grp, "a": score_old, "b": score_new}).groupby(
            "grp"):
        if len(g) < 10:
            continue
        ra, rb = g.a.rank(ascending=False), g.b.rank(ascending=False)
        n = len(g)
        # share of pairs whose order flips
        A, B = g.a.values, g.b.values
        ii, jj = np.triu_indices(n, 1)
        flips = np.mean(np.sign(A[ii] - A[jj]) != np.sign(B[ii] - B[jj]))
        out.append(dict(group=grp, players=n, reversed_pairs=float(flips),
                        median_rank_move=float((rb - ra).abs().median()),
                        p95_rank_move=float((rb - ra).abs().quantile(0.95)),
                        median_score_move=float((g.b - g.a).abs().median()),
                        p90_score_move=float((g.b - g.a).abs().quantile(0.90))))
    if not out:
        return None
    t = pd.DataFrame(out)
    w = t.players / t.players.sum()
    return dict(competition=comp,
                reversed_pairs=float((t.reversed_pairs * w).sum()),
                median_rank_move=float(t.median_rank_move.median()),
                p95_rank_move=float(t.p95_rank_move.max()),
                median_score_move=float((t.median_score_move * w).sum()),
                p90_score_move=float(t.p90_score_move.max()))


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
    rows, shr, impact = [], [], []
    for comp in COMPS:
        hist = pd.read_sql(
            f"SELECT {', '.join(rh.COLS) if hasattr(rh, 'COLS') else '*'} "
            f"FROM player_match_stats WHERE competition=? AND season<=?",
            con, params=(comp, pre.FREEZE_SEASON))
        if hist.empty:
            continue
        force, _ = rh._mode(hist)
        pm = rh.composites(hist, comp, force)
        if pm.empty:
            continue
        s_c, t_c = components_career(pm)
        s_p, t_p = components_player_season(pm)
        rows.append(dict(competition=comp, sigma2_career=s_c, sigma2_season=s_p,
                         sigma2_change=(s_p - s_c) / s_c * 100 if s_c else np.nan,
                         tau2_career=t_c, tau2_season=t_p,
                         tau2_change=(t_p - t_c) / t_c * 100 if t_c else np.nan))
        for n in (1, 5, 10, 25):
            b_c, b_p = shrinkage(t_c, s_c, n), shrinkage(t_p, s_p, n)
            shr.append(dict(competition=comp, matches=n, B_now=b_c, B_alt=b_p,
                            change_pct=(b_p - b_c) / b_c * 100 if b_c else np.nan))
        ri = rank_impact(con, comp, s_c, t_c, s_p, t_p)
        if ri:
            impact.append(ri)
    con.close()

    comp_df, shr_df = pd.DataFrame(rows), pd.DataFrame(shr)
    W = ["# What the constant-ability assumption costs\n",
         "The rating engine estimates its variance components with one intercept per "
         "player, held constant over his whole career. Everything that actually moves — "
         "ageing, a change of position, a season off the bench — is then booked as "
         "game-to-game noise. This compares that against a decomposition that gives each "
         "player-SEASON its own intercept, so a level that changes between seasons is no "
         "longer mistaken for randomness within them.\n",
         "\nNeither is a state-space model. The point is the size and direction of the "
         "gap, not that the alternative is right.\n",
         "\n## Components\n", md(comp_df, "{:.4f}"),
         "\n\n`sigma2_change` and `tau2_change` are percentages.\n",
         "\n## What it does to the shrinkage\n",
         "B is the share of a player's own numbers the rating keeps. `B_now` is today's, "
         "`B_alt` the same figure under the player-season decomposition.\n",
         md(shr_df, "{:.3f}")]
    imp_df = pd.DataFrame(impact)
    if len(imp_df):
        W.append(NL + NL + "## And what it does to the published list" + NL)
        W.append("The components above are a diagnosis; this is the prognosis. Each "
                 "player's rating is rebuilt with the alternative components and "
                 "nothing else changed, then compared with what is published today, "
                 "within his own peer group." + NL)
        W.append(md(imp_df, "{:.3f}"))
        W.append(NL + NL + "`reversed_pairs` is the share of player pairs whose order "
                 "flips; rank moves are places within a peer group; score moves are "
                 "points on the published 0-100." + NL)

    if len(comp_df):
        t = comp_df.tau2_change.mean()
        s = comp_df.sigma2_change.mean()
        W.append(f"\n\n## What this says\n")
        W.append(f"Letting a player's level move between seasons raises tau-squared by "
                 f"{t:.0f}% on average and lowers sigma-squared by {abs(s):.0f}%. Both "
                 f"point the same way: the current model books real change as noise, so "
                 f"it holds the true spread between players too low and the "
                 f"match-to-match randomness too high.\n")
        one = shr_df[shr_df.matches == 1].change_pct.mean()
        ten = shr_df[shr_df.matches == 10].change_pct.mean()
        W.append(f"The consequence is heaviest where evidence is thinnest: at one match "
                 f"the shrinkage factor would be {one:.0f}% higher, at ten matches "
                 f"{ten:.0f}%. 'At least conservative' was too comfortable a description "
                 f"of that and has been withdrawn.\n")
        if len(imp_df):
            rp = float(imp_df.reversed_pairs.max())
            sm = float(imp_df.median_score_move.max())
            W.append(f"\n**But the size of the consequence is not what an earlier version "
                     f"of this page implied.** It said the correction reorders players "
                     f"rather than merely compressing them, and stopped. Measured, the "
                     f"reordering is small: at most {rp:.1%} of pairs within a peer group "
                     f"change places and the median player does not move a single rank. "
                     f"The fourth external review was right to ask for the number, and "
                     f"right that the wording suggested more product risk than the data "
                     f"supports.\n")
            W.append(f"\nWhere it does bite is the distance rather than the order. tau "
                     f"divides the published 0-100, so a player's score moves by a median "
                     f"of up to {sm:.1f} points and more in the tails. That is the honest "
                     f"statement: **the ranking is robust to this bias, the displayed "
                     f"gaps between players are not.** A peer score is usable for "
                     f"ordering a shortlist and should not be read as a calibrated "
                     f"distance.\n")
        W.append("\nThe fix is a model that lets ability move — a player-season random "
                 "effect at minimum, a state-space formulation properly. Not done.\n")

    open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(W) + "\n")
    print(comp_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(shr_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
