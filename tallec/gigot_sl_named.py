# -*- coding: utf-8 -*-
"""GIGOT on Super League with what is known before kick-off, and against the market.

`gigot_v2.py` showed that the player layer improves an ELO-plus-home baseline when it is
fed the players who actually took the field, weighted by the minutes they actually
played: an upper bound, since neither is known before the match. Mike's history
workbooks hold, for 2021-2026, the 21-man squad (named about two days out), the
matchday seventeen and the club's handicap and total line for each match.
(`ingest_sl_history.py` -> sl_teamsheets, sl_match_lines.)

Three versions of the player layer, each a minutes-weighted mean of the players' pre-
match class and form, from their earlier matches only:
  actual   who played, weighted by the minutes they played (gigot_v2, the upper bound)
  named17  the matchday seventeen, weighted by each man's average minutes so far
  named21  the 21-man squad, the same weights; four of them will not play
and the betting line two ways: as a benchmark (is the model closer to the margin than
the line? does it pick the side that covers?) and as information (does the model add
anything once the line is known?).

Every model is fitted walk-forward: season t is predicted from seasons before t. The
shipped xLadder prediction (Margin_Pred_v2) is scored only on 2025, its held-out
season, and 2026, where it has been live.

    python gigot_sl_named.py        # writes GIGOT_SL_NAMED.md and gigot_sl_named.csv
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

import gigot_v2 as gv
import sp_schema as sp
import team_map as tm

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(BASE, "GIGOT_SL_NAMED.md")
N_BOOT, SEED = 4000, 0
VARIANTS = ("actual", "named17", "named21")
BASE_F = ["Diff ELO", "home"]


def player_history(pm):
    """After each played match: class (all matches so far), form (last five), minutes."""
    h = pm.sort_values(["player_id", "season", "round"]).copy()
    g = h.groupby("player_id")
    h["post_class"] = g.composite.transform(lambda s: s.expanding().mean())
    h["post_form"] = g.composite.transform(lambda s: s.rolling(gv.FORM_WINDOW, 1).mean())
    h["post_mins"] = g.minutes.transform(lambda s: s.expanding().mean())
    h["post_n"] = g.cumcount() + 1
    h["t"] = (h.season * 100 + h["round"]).astype(float)
    h["player_id"] = sp.normalize_player_id(h.player_id)
    return h[["player_id", "t", "post_class", "post_form", "post_mins", "post_n"]]


def named_rows(con, hist, roles, default_mins):
    ts = pd.read_sql("SELECT season, round, team, slot, role, player_id FROM sl_teamsheets",
                     con)
    ts = ts[ts.role.isin(roles)].copy()
    ts["player_id"] = sp.normalize_player_id(ts.player_id.where(ts.player_id.notna(), ""))
    ts["t"] = (ts.season * 100 + ts["round"] - 0.5).astype(float)   # strictly before
    ts = ts.sort_values("t")
    hh = hist.sort_values("t")
    j = pd.merge_asof(ts, hh, on="t", by="player_id", direction="backward")
    j["w"] = j.post_mins.fillna(j.role.map(default_mins))
    j["c"] = j.post_class.fillna(0.0)
    j["f"] = j.post_form.fillna(0.0)
    j["green"] = (j.post_n.fillna(0) < 3).astype(float)
    out = j.groupby(["season", "round", "team"]).apply(lambda g: pd.Series({
        "lineup_class": np.average(g.c, weights=g.w), "lineup_form": np.average(g.f, weights=g.w),
        "green_share": g.green.mean(), "n_players": len(g)}), include_groups=False)
    return out.reset_index()


def season_maps(comp="SL", tol=4, m=None):
    """{(season, code) -> team}: team_map.solve's score-vote, one season at a time.

    One mapping for every season is wrong for Super League: the 2026 rows of the master
    use the codes differently (C is Castleford there, Catalans before). Votes are the
    same as team_map.solve's: within a round, the side whose score matches, and whose
    opponent's score matches too."""
    if m is None:
        m = tm.load_master(comp, ["Season", "Round", "A Team", "B Team", "A Score",
                                  "B Score"])
    m = m[["Season", "Round", "A Team", "B Team", "A Score", "B Score"]].dropna()
    p = tm._team_points(comp)
    out = {}
    for season, ms in m.groupby("Season"):
        votes = {}
        for rnd, fx in ms.groupby("Round"):
            pr = p[(p.season == season) & (p["round"] == rnd)]
            pts, opp = dict(zip(pr.team, pr.pts)), dict(zip(pr.team, pr.opposition))
            for _, f in fx.iterrows():
                for ta, pa in pts.items():
                    tb = opp.get(ta)
                    if abs(pa - f["A Score"]) > tol or tb not in pts \
                            or abs(pts[tb] - f["B Score"]) > tol:
                        continue
                    w = 2 if (pa == f["A Score"] and pts[tb] == f["B Score"]) else 1
                    votes[(f["A Team"], ta)] = votes.get((f["A Team"], ta), 0) + w
                    votes[(f["B Team"], tb)] = votes.get((f["B Team"], tb), 0) + w
        taken, done = set(), set()
        for (code, team), v in sorted(votes.items(), key=lambda kv: -kv[1]):
            if code in done or team in taken:
                continue
            out[(int(season), code)] = team
            done.add(code)
            taken.add(team)
    return out


def fixtures(con):
    pm = gv.prematch_players("SL", con)
    hist = player_history(pm)
    feats = {"actual": gv.team_rows(pm).drop(columns=["warmup"]),
             "named17": named_rows(con, hist, ("start", "bench"),
                                   {"start": 60.0, "bench": 25.0}),
             "named21": named_rows(con, hist, ("start", "bench", "reserve"),
                                   {"start": 60.0, "bench": 25.0, "reserve": 25.0})}
    # The master as the xLadder pipeline now reads it: the 2022-2025 team codes
    # relabelled and ELO recomputed (xladder_pipeline.fix_sl_team_codes). The margin
    # comes from A_/B_Points Scored, which are right where A Score / B Score are
    # reversed against the codes (about half of the 2022-24 rows). Read only.
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(BASE))
    import xladder_pipeline as xp
    cols = ["Match ID", "Season", "Round", "A Team", "B Team", "A Score", "B Score",
            "A_Points Scored", "B_Points Scored", "Margin", "Home Advantage",
            "ELO_A", "ELO_B", "Diff ELO", "Margin_Pred_v2"]
    m = xp.fix_sl_team_codes(tm.load_master("SL", cols))
    m["Margin"] = m["A_Points Scored"] - m["B_Points Scored"]
    smap = season_maps(m=m.assign(**{"A Score": m["A_Points Scored"],
                                     "B Score": m["B_Points Scored"]}))
    m = m.dropna(subset=["Margin"])
    m["team_a"] = [smap.get((int(y), c)) for y, c in zip(m.Season, m["A Team"])]
    m["team_b"] = [smap.get((int(y), c)) for y, c in zip(m.Season, m["B Team"])]
    m["home"] = np.where(m["Home Advantage"] == "A", 1.0,
                         np.where(m["Home Advantage"] == "B", -1.0, 0.0))
    for v, f in feats.items():
        a = f.rename(columns=lambda c: c if c in ("season", "round", "team") else f"{c}_a")
        b = f.rename(columns=lambda c: c if c in ("season", "round", "team") else f"{c}_b")
        m = (m.merge(a, left_on=["Season", "Round", "team_a"],
                     right_on=["season", "round", "team"], how="left")
              .drop(columns=["season", "round", "team"])
              .merge(b, left_on=["Season", "Round", "team_b"],
                     right_on=["season", "round", "team"], how="left")
              .drop(columns=["season", "round", "team"]))
        m[f"{v}_d_class"] = m.lineup_class_a - m.lineup_class_b
        m[f"{v}_d_form"] = m.lineup_form_a - m.lineup_form_b
        m[f"{v}_d_green"] = m.green_share_a - m.green_share_b
        m = m.drop(columns=[c for c in m.columns if c.startswith(
            ("lineup_class_", "lineup_form_", "green_share_", "n_players_"))])
    lines = pd.read_sql("SELECT season, round, team, opposition, handicap, total_line, "
                        "score_for, score_against FROM sl_match_lines", con)
    m = m.merge(lines.rename(columns={"season": "Season", "round": "Round",
                                      "team": "team_a", "opposition": "opp_line"}),
                on=["Season", "Round", "team_a"], how="left")
    m["market"] = -m.handicap                  # the line's expected margin for team A
    m.loc[m.opp_line != m.team_b, "market"] = np.nan
    # the two clubs' handicaps must mirror each other
    hb = lines.set_index(["season", "round", "team"]).handicap
    hb = hb[~hb.index.duplicated()]
    other = hb.reindex(list(zip(m.Season, m.Round, m.team_b))).values
    m.loc[~(np.abs(m.handicap.values + other) < 0.01), "market"] = np.nan
    # fixtures whose pairing does not match Stats Perform (36 master rows, mostly 2023
    # LS/WA shifted a round) are dropped: their features belong to other matches
    sp_opp = (pd.read_sql("SELECT DISTINCT season, round, team, opposition FROM "
                          "player_match_stats WHERE competition='SL'", con)
              .drop_duplicates(["season", "round", "team"])
              .set_index(["season", "round", "team"]).opposition)
    m["sp_opp"] = sp_opp.reindex(list(zip(m.Season, m.Round, m.team_a))).values
    m["pairing_ok"] = m.sp_opp == m.team_b
    return m[m.pairing_ok].copy()


def walk_forward(d, cols):
    out = pd.Series(np.nan, index=d.index)
    for t in sorted(d.Season.unique())[1:]:
        tr, te = d[d.Season < t], d[d.Season == t]
        if len(tr) < 100 or te.empty:
            continue
        out[te.index] = Ridge(alpha=1.0).fit(tr[cols], tr.Margin).predict(te[cols])
    return out


def boot(fn, n, ids=None):
    rng = np.random.default_rng(SEED)
    idx = np.arange(n)
    v = np.array([fn(rng.choice(idx, n)) for _ in range(N_BOOT)])
    return fn(idx), np.percentile(v, 2.5), np.percentile(v, 97.5)


def ats(margin, market, pred, min_edge=0.0):
    """Share of bets that cover: back the side the model prefers to the line."""
    edge = pred - market
    res = margin - market
    keep = (np.abs(edge) > min_edge) & (res != 0) & np.isfinite(edge) & np.isfinite(res)
    win = (np.sign(edge) == np.sign(res))[keep]
    return win, int(keep.sum())


def run():
    con = sqlite3.connect(f"file:{gv.DB}?mode=ro", uri=True)
    d = fixtures(con)
    con.close()
    feats = {v: [f"{v}_d_class", f"{v}_d_form", f"{v}_d_green"] for v in VARIANTS}
    d = d.dropna(subset=BASE_F + sum(feats.values(), [])).reset_index(drop=True)
    d["p_base"] = walk_forward(d, BASE_F)
    for v in VARIANTS:
        d[f"p_{v}"] = walk_forward(d, BASE_F + feats[v])
    hm = d.market.notna()
    d.loc[hm, "p_market_named21"] = walk_forward(d[hm], ["market"] + feats["named21"]).values
    d.loc[hm, "p_market"] = walk_forward(d[hm], ["market"]).values
    return d


def report(d):
    d.to_csv(os.path.join(BASE, "gigot_sl_named.csv"), index=False)
    o = d[d.p_base.notna()]
    L = ["# GIGOT on Super League: named teams and the betting line", "",
         "Generated by `gigot_sl_named.py`. Walk-forward by season (each season predicted "
         "from the seasons before it). Intervals: 95% bootstrap over matches. The master "
         "is read with the 2022-2025 team codes fixed and ELO recomputed, fixtures whose "
         "pairing does not match Stats Perform are dropped, and a line counts only when "
         "the two clubs' handicaps mirror.", "",
         f"Fixtures with every feature: {len(d)} ({d.Season.min()}–{d.Season.max()}); "
         f"scored out of sample: {len(o)} ({o.Season.min()}–{o.Season.max()}); with a "
         f"betting line: {int(o.market.notna().sum())}.", "",
         "## Does the player layer survive without hindsight?", "",
         "| Model | MAE | Gain over ELO + home [95%] |", "|---|---|---|"]
    eb = (o.Margin - o.p_base).abs().values
    L.append(f"| ELO + home | {eb.mean():.2f} | — |")
    for v in VARIANTS:
        e = (o.Margin - o[f"p_{v}"]).abs().values
        g = boot(lambda i: (eb[i] - e[i]).mean(), len(e))
        lab = {"actual": "+ players who played, actual minutes (upper bound)",
               "named17": "+ named 17, expected minutes",
               "named21": "+ named 21, expected minutes"}[v]
        L.append(f"| {lab} | {e.mean():.2f} | {g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}] |")
    L += ["", "By season (gain over ELO + home):", "", "| Season | Fixtures | Actual | "
          "Named 17 | Named 21 |", "|---|---|---|---|---|"]
    for yr, g in o.groupby("Season"):
        e0 = (g.Margin - g.p_base).abs()
        L.append(f"| {yr} | {len(g)} | " + " | ".join(
            f"{(e0 - (g.Margin - g[f'p_{v}']).abs()).mean():+.2f}" for v in VARIANTS) + " |")
    L += ["", "The named 17 is close to hindsight: the sheets were completed after the "
          "match (tries are marked), and 99.8% of the starters and 98.2% of the bench "
          "appear in Stats Perform. The 21-man squad, named about two days before, is the "
          "pre-match variant.", ""]

    m = o[o.market.notna()]
    y, mk = m.Margin.values, m.market.values
    L += ["## Against the betting line", "",
          f"{len(m)} matches with the club's handicap. The line's expected margin is minus "
          "the handicap. Break-even at standard -110 pricing is 52.4%.", "",
          "| Prediction | MAE | Covers when it disagrees with the line [95%] | Bets | "
          "Covers, edge > 4 pts | Bets |", "|---|---|---|---|---|---|"]
    L.append(f"| The line itself | {np.abs(y - mk).mean():.2f} | — | — | — | — |")
    for col, lab in (("p_base", "ELO + home"), ("p_named21", "+ named 21"),
                     ("p_actual", "+ actual line-up (hindsight)"),
                     ("p_market_named21", "line + named 21")):
        p = m[col].values
        w, n = ats(y, mk, p)
        w4, n4 = ats(y, mk, p, 4.0)
        b = boot(lambda i: w[i].mean(), len(w))
        L.append(f"| {lab} | {np.abs(y - p).mean():.2f} | {100 * b[0]:.1f}% "
                 f"[{100 * b[1]:.1f}, {100 * b[2]:.1f}] | {n} | "
                 f"{100 * w4.mean():.1f}% | {n4} |")
    L.append("")
    # does the model add information once the line is known?
    X = np.column_stack([mk, m.p_named21.values])
    lr = LinearRegression().fit(X, y)
    bb = []
    rng = np.random.default_rng(SEED)
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        bb.append(LinearRegression().fit(X[i], y[i]).coef_)
    bb = np.array(bb)
    L += ["Regression of the margin on the line and the named-21 model together, out-of-"
          "sample predictions only:", "",
          f"- line coefficient {lr.coef_[0]:+.2f} [{np.percentile(bb[:, 0], 2.5):+.2f}, "
          f"{np.percentile(bb[:, 0], 97.5):+.2f}]",
          f"- model coefficient {lr.coef_[1]:+.2f} [{np.percentile(bb[:, 1], 2.5):+.2f}, "
          f"{np.percentile(bb[:, 1], 97.5):+.2f}]",
          "", "A model coefficient whose interval excludes zero means the model carries "
          "information the line does not.", ""]
    # the shipped xLadder prediction, held-out and live seasons only
    s = d[d.Season.isin([2025, 2026]) & d.market.notna() & d.Margin_Pred_v2.notna()]
    if len(s):
        w, n = ats(s.Margin.values, s.market.values, s.Margin_Pred_v2.values)
        w4, n4 = ats(s.Margin.values, s.market.values, s.Margin_Pred_v2.values, 4.0)
        b = boot(lambda i: w[i].mean(), len(w))
        L += ["## The shipped xLadder prediction (Margin_Pred_v2)", "",
              "Scored on 2025, its stated test season; the master holds no 2026 "
              "prediction, and whether the stored 2025 column was truly out of sample "
              "cannot be checked from the master.", "",
              f"- matches: {len(s)}; MAE {np.abs(s.Margin - s.Margin_Pred_v2).mean():.2f} "
              f"against the line's {np.abs(s.Margin - s.market).mean():.2f}",
              f"- covers when it disagrees with the line: {100 * b[0]:.1f}% "
              f"[{100 * b[1]:.1f}, {100 * b[2]:.1f}] of {n}; with an edge over 4 points "
              f"{100 * w4.mean():.1f}% of {n4}", ""]
        for yr, g in s.groupby("Season"):
            w, n = ats(g.Margin.values, g.market.values, g.Margin_Pred_v2.values)
            L.append(f"  - {yr}: {100 * w.mean():.1f}% of {n}")
        L.append("")
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    report(run())
    sys.exit(0)
