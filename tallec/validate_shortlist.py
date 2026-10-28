# -*- coding: utf-8 -*-
"""Is the Into Super League forecast a better shortlist than the simple rules?

Mike's question (October 2026): the third of entrants with the highest forecast reached
Super League's top third more often than entrants as a whole, but how much better is that
than picking players on their Australian rating, their appearances or their starts?

The entrants and the forecasts are the exploration's (`explore_post2026.run`): every
first or returning entrant into Super League from the NRL, the NSW Cup or the Queensland
Cup who landed in 2023-2026 and played three or more matches there, each forecast by the
app's line fitted only on the seasons before he landed. The outcome is the same: a
season rating in the top third of every rated Super League player that season.

Every rule ranks the same entrants, using only what was known before the move (his
source season):

  forecast       the app's line forecast (the translation)
  rating         his rating in the source competition, not translated
  appearances    matches played in the source season
  starts         matches started in the source season
  start share    starts as a share of his matches
  minutes        minutes per match in the source season

Each season's shortlist is the top third of that season's entrants by the rule; the hit
rate is pooled over the four seasons. Differences against the forecast have 95% player-
cluster bootstrap intervals. AUC over all entrants is reported alongside, because a
shortlist cut at a third throws away most of the ranking.

Two more answers come out of the same entrants:

  * the track record (table `entrant_backtest`): every entrant, what the app's line
    forecast him at before he moved, its 50% and 80% ranges, and what he then rated, so
    a club can see the hits and the misses and whether the ranges held;
  * whether the translation should differ by position. Each direction's line is refitted
    separately for forwards (middles, edges, hookers) and backs wherever a direction has
    25 pairs of that kind, otherwise the direction's own line stays. Scored like the
    frozen 2026 test: a gain counts only if it is at least one point of MAE and its
    interval excludes zero.

This is an exploratory backtest: the 2023-2026 entrants have been looked at before.
Set POST2026_CACHE as for `explore_post2026.py`; the position test reads that cache.

    python validate_shortlist.py            # report only
    python validate_shortlist.py --write    # also replace shortlist_validation and
                                            # entrant_backtest
"""
import argparse
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd

import explore_post2026 as ex
import frozen_2026 as fz
import rolling_backtest as rb
import runtime
import sp_schema as sp

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BASE, "SHORTLIST_VALIDATION.md")
N_BOOT = 5000
SEED = 20261028

RULES = [("forecast", "S0", "Forecast (translated)"),
         ("rating", "class_source", "Australian rating"),
         ("appearances", "games_src", "Appearances"),
         ("starts", "starts_src", "Starts"),
         ("start_share", "src_start_share", "Start share"),
         ("minutes", "src_mins", "Minutes per match")]


def entrants():
    ev, _ = ex.run()
    d = ev[ev.pair.isin(ex.H1)].copy()
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    ps = ex.starts(con)
    con.close()
    s = ps.reindex(list(zip(d.player_id, d.source, d.season_src)))
    d["games_src"], d["starts_src"] = s.games.values, s.starts.values
    return d.reset_index(drop=True)


def shortlist_hits(d, col):
    """Hits and shortlist size, the top third of each landing season by `col`."""
    hits = k_all = 0
    for _, g in d.groupby("season_tgt"):
        k = max(1, int(round(len(g) / 3)))
        # ties broken by player id, so the answer does not depend on row order
        top = g.sort_values([col, "player_id"], ascending=[False, True]).head(k)
        hits += int(top.top3.sum())
        k_all += k
    return hits, k_all


def auc(y, s):
    return ex.auc(y, s)


def bootstrap(d, fn):
    """95% player-cluster percentile interval of fn(resampled frame)."""
    ids = np.array(sorted(d.player_id.unique()))
    at = {p: np.flatnonzero(d.player_id.values == p) for p in ids}
    rng = np.random.default_rng(SEED)
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        rows = np.concatenate([at[p] for p in rng.choice(ids, ids.size, replace=True)])
        out[i] = fn(d.iloc[rows])
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def rate(d, col):
    h, k = shortlist_hits(d, col)
    return h / k


FORWARDS = {"Middles", "Edge", "Hooker"}


def unit(raw_position):
    g = pd.Series(raw_position).map(sp.POSITION_GROUP)
    return np.where(g.isin(FORWARDS), "forwards", np.where(g.notna(), "backs", "unknown"))


def position_split():
    """Entrants into SL forecast by the direction line (S0) and by a forwards / backs
    line where one has 25 pairs, each landing season from earlier seasons only."""
    if not (ex.CACHE and os.path.exists(ex.CACHE)):
        return None, []
    D = pickle.load(open(ex.CACHE, "rb"))
    out, fitted = [], []
    for t in ex.SEASONS:
        cum = D.get("cum_by_t", {}).get(t, D["cum"])
        one = D.get("one_by_t", {}).get(t, D["one"])
        train = ex.attach(D["pairs"][t], cum, one, True)
        train["pair"] = train.source + "->" + train.target
        ev = ex.attach(D["cohort"][D["cohort"].season_tgt == t], cum, one, False)
        ev = ev[ev.pair.isin(ex.H1)].reset_index(drop=True)
        s0, _ = fz.predict_lines(fz.fit_lines(train), ev)
        m1 = fz.fit_lines(train.assign(pair=train.pair + "|" + unit(train.raw_position)))
        key = ev.pair + "|" + unit(ev.raw_position)
        s1 = s0.copy()
        for k, idx in key.groupby(key).indices.items():
            if k in m1["own"]:
                X, _ = fz._columns(ev.iloc[idx], m1["own"][k]["extras"], m1["own"][k]["stats"])
                s1[idx] = np.clip(X @ m1["own"][k]["coef"], 0, 100)
        fitted.append((t, sorted(m1["own"])))
        out.append(ev.assign(S0=s0, S_pos=s1, unit=unit(ev.raw_position), season=t))
    return pd.concat(out, ignore_index=True), fitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    d = entrants()
    d = d[d.top3.notna()]
    for _, col, _ in RULES:
        if d[col].isna().any():
            print(f"{col}: {int(d[col].isna().sum())} entrants without a source figure")
            # a missing source figure ranks last rather than dropping the entrant, so
            # every rule is scored on the same men
            d[col] = d[col].fillna(-np.inf)
    base = float(d.top3.mean())
    rows = []
    for key, col, label in RULES:
        h, k = shortlist_hits(d, col)
        a_ = auc(d.top3.values, d[col].replace(-np.inf, np.nan).values)
        row = dict(rule=key, label=label, entrants=len(d), players=d.player_id.nunique(),
                   shortlisted=k, hits=h, hit_rate=h / k, base_rate=base, auc=a_)
        if key == "forecast":
            row["auc_lo"], row["auc_hi"] = bootstrap(
                d, lambda x: auc(x.top3.values, x.S0.values))
        else:
            dlt = rate(d, "S0") - h / k
            lo, hi = bootstrap(d, lambda x: rate(x, "S0") - rate(x, col))
            dauc = auc(d.top3.values, d.S0.values) - a_
            alo, ahi = bootstrap(d, lambda x: auc(x.top3.values, x.S0.values)
                                 - auc(x.top3.values, x[col].replace(-np.inf, np.nan).values))
            row.update(forecast_minus=dlt, lo=lo, hi=hi,
                       auc_forecast_minus=dauc, auc_lo=alo, auc_hi=ahi)
        rows.append(row)
    t = pd.DataFrame(rows)
    t["seasons"] = f"{int(d.season_tgt.min())}-{int(d.season_tgt.max())}"

    L = ["# Into Super League: the forecast against simpler shortlists", "",
         "Generated by `validate_shortlist.py`; do not edit by hand. Exploratory backtest.",
         "",
         f"{len(d)} entrants into Super League from the NRL, the NSW Cup and the "
         f"Queensland Cup, {d.player_id.nunique()} players, landing "
         f"{t.seasons.iloc[0]}. Each was ranked on what was known before he moved. A hit "
         f"is a season rating in the top third of Super League that season; "
         f"{100 * base:.0f}% of all entrants managed it.", "",
         "| Rule | Shortlisted | Hits | Hit rate | Forecast minus rule | AUC | "
         "Forecast minus rule (AUC) |", "|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        if r.rule == "forecast":
            L.append(f"| {r.label} | {r.shortlisted} | {r.hits} | {100 * r.hit_rate:.0f}% "
                     f"| — | {r.auc:.2f} [{r.auc_lo:.2f}, {r.auc_hi:.2f}] | — |")
        else:
            L.append(f"| {r.label} | {r.shortlisted} | {r.hits} | {100 * r.hit_rate:.0f}% "
                     f"| {100 * r.forecast_minus:+.0f} pts [{100 * r.lo:+.0f}, "
                     f"{100 * r.hi:+.0f}] | {r.auc:.2f} | {r.auc_forecast_minus:+.2f} "
                     f"[{r.auc_lo:+.2f}, {r.auc_hi:+.2f}] |")
    L += ["", "By season (forecast shortlist):", "",
          "| Season | Entrants | Shortlisted | Hits | All entrants hitting |",
          "|---|---|---|---|---|"]
    for s, g in d.groupby("season_tgt"):
        h, k = shortlist_hits(g, "S0")
        L.append(f"| {s} | {len(g)} | {k} | {h} | {100 * g.top3.mean():.0f}% |")
    L += ["", "Within one source competition the forecast is a straight line of the "
          "rating, so it ranks those players exactly as the rating does; any difference "
          "between the two rules comes from ranking players of different competitions "
          "against each other.", ""]
    # ── the track record and its ranges ──────────────────────────────────────
    tr = pd.DataFrame({
        "player_id": d.player_id, "name": d.name, "source": d.source,
        "season_src": d.season_src, "season_tgt": d.season_tgt,
        "position": d.raw_position, "rating_source": d.class_source,
        "forecast": d.S0,
        "lo50": (d.S0 + d.e25).clip(0, 100), "hi50": (d.S0 + d.e75).clip(0, 100),
        "lo80": (d.S0 + d.e10).clip(0, 100), "hi80": (d.S0 + d.e90).clip(0, 100),
        "actual": d.class_target, "matches_sl": d.n_target, "top3": d.top3,
        "starter": d.starter})
    tr["in50"] = (tr.actual >= tr.lo50) & (tr.actual <= tr.hi50)
    tr["in80"] = (tr.actual >= tr.lo80) & (tr.actual <= tr.hi80)
    L += ["## Track record and ranges", "",
          f"Every one of the {len(tr)} entrants is in the table `entrant_backtest`: the "
          f"forecast made before he moved, its ranges, and what he rated. The 50% range "
          f"held {100 * tr.in50.mean():.0f}% of them and the 80% range "
          f"{100 * tr.in80.mean():.0f}%. Mean absolute error "
          f"{(tr.actual - tr.forecast).abs().mean():.1f} points.", "",
          "| Season | Entrants | In the 50% range | In the 80% range | MAE |",
          "|---|---|---|---|---|"]
    for s_, g in tr.groupby("season_tgt"):
        L.append(f"| {s_} | {len(g)} | {100 * g.in50.mean():.0f}% | "
                 f"{100 * g.in80.mean():.0f}% | "
                 f"{(g.actual - g.forecast).abs().mean():.1f} |")

    # ── does the translation differ by position? ─────────────────────────────
    ps_, fitted = position_split()
    L += ["", "## A separate line for forwards and backs?", ""]
    if ps_ is None:
        L.append("Not run: set POST2026_CACHE.")
    else:
        y = ps_.class_target.values
        dl, lo, hi = fz.paired_bootstrap(y, ps_.S0.values, ps_.S_pos.values,
                                         ps_.player_id.values)
        verdict = ("a gain of a point or more is supported" if dl >= 1 and lo > 0 else
                   "worse than one line per direction" if hi < 0 else
                   "a one-point gain is not supported" if hi < 1 else "undecided")
        L += [f"{len(ps_)} entrants ({(ps_.unit == 'forwards').sum()} forwards, "
              f"{(ps_.unit == 'backs').sum()} backs). Positive means the forwards / backs "
              f"lines are closer.", "",
              f"| Forecast | MAE |", "|---|---|",
              f"| One line per direction (the app) | {np.abs(y - ps_.S0).mean():.2f} |",
              f"| Forwards / backs where 25 pairs allow | {np.abs(y - ps_.S_pos).mean():.2f} |",
              "", f"Δ {dl:+.2f} [{lo:+.2f}, {hi:+.2f}]: {verdict}.", "",
              "Lines that had enough pairs of their own, by landing season: "
              + "; ".join(f"{t_}: {', '.join(k) or 'none'}" for t_, k in fitted) + ".",
              "", "| Unit | Entrants | One line | Forwards / backs |", "|---|---|---|---|"]
        for u, g in ps_.groupby("unit"):
            L.append(f"| {u} | {len(g)} | {np.abs(g.class_target - g.S0).mean():.2f} | "
                     f"{np.abs(g.class_target - g.S_pos).mean():.2f} |")
    L.append("")

    open(REPORT, "w").write("\n".join(L))
    print("\n".join(L))

    if a.write:
        with runtime.guarded_write("shortlist_validation", note=f"{len(d)} entrants"):
            con = sqlite3.connect(rb.DB)
            t.to_sql("shortlist_validation", con, if_exists="replace", index=False)
            tr.to_sql("entrant_backtest", con, if_exists="replace", index=False)
            con.commit()
            con.close()
        print("wrote shortlist_validation, entrant_backtest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
