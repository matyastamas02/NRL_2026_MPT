# -*- coding: utf-8 -*-
"""Mike's Super League history workbooks: team sheets, betting lines, weather.

One workbook per season (2021-2026), one sheet per club, one column per week. Each
column holds the H/A flag, the opposition, the club's handicap, the result, the weather,
a "spin points" line (read as the match total line; to be confirmed with Mike) and the
named players: shirts 1-13, four interchange, then up to four more names. Checked
against Stats Perform on Leeds 2025: the seventeen are the matchday squad, including an
unused substitute Stats Perform does not list, and the four extra names did not play,
which is what the 18th-21st men of a 21-man squad look like. Tries are marked after a
name as "[n]". Which bookmaker and which moment the handicap is from is not stated.

Writes, through runtime.guarded_write:
  sl_teamsheets   season, round, team, opposition, slot 1-21, role (start 1-13 /
                  bench 14-17 / reserve 18-21), the name as written, tries marked, and
                  the Stats Perform player_id where the name resolves
  sl_match_lines  season, round, team, opposition, venue, handicap (the team's own),
                  total line, score for and against, weather

Weeks are aligned to Stats Perform rounds club by club: the next unused fixture with
the same opposition, and among several, the one whose player list overlaps the named
seventeen most. Names resolve inside the club-season roster, then the season's whole
competition, exact after normalisation and then by close spelling.

    python ingest_sl_history.py            # dry run: alignment and validation report
    python ingest_sl_history.py --write    # replace the two tables
"""
import argparse
import difflib
import glob
import os
import re
import sqlite3
import sys

import numpy as np
import openpyxl
import pandas as pd

import rolling_backtest as rb
import runtime
import sp_schema as sp

HERE = os.path.dirname(os.path.abspath(__file__))
# the workbooks sit in Downloads, two folders up from the repository's tallec/ and one up
# from the TALLEC working copy
FILES = sorted(set(sum((glob.glob(os.path.join(d, "Super League History *.xlsx"))
                        for d in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE)))),
                       [])), key=os.path.basename)
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SL_HISTORY_REPORT.md")
NOT_CLUBS = {"Patterns", "Compatibility Report", "Vs H'cap", "PCM", "Sheet1", "Errors"}
SHEET = {"Castleford": "Castleford", "Catalans": "Catalans", "Hudds": "Huddersfield",
         "Hull": "Hull", "KR": "Hull KR", "Leeds": "Leeds", "Leigh": "Leigh",
         "Saints": "St Helens", "Salford": "Salford Red Devils", "Wakefield": "Wakefield",
         "Wakey": "Wakefield", "Wigan": "Wigan", "Wire": "Warrington",
         "London": "London Broncos", "Toulouse": "Toulouse", "Bradford": "Bradford",
         "York": "York RLFC Knights"}
OPP = {"bradford": "Bradford", "caatleford": "Castleford", "castelford": "Castleford",
       "castleford": "Castleford", "cass": "Castleford", "catalan": "Catalans",
       "catalans": "Catalans", "catlans": "Catalans", "hudderfield": "Huddersfield",
       "huddersfield": "Huddersfield", "hudds": "Huddersfield", "hull": "Hull",
       "hull fc": "Hull", "hull kr": "Hull KR", "kr": "Hull KR", "leeds": "Leeds",
       "leigh": "Leigh", "london": "London Broncos", "saints": "St Helens",
       "salford": "Salford Red Devils", "st helens": "St Helens", "toulouse": "Toulouse",
       "wakefield": "Wakefield", "wakey": "Wakefield", "walefield": "Wakefield",
       "warrington": "Warrington", "wire": "Warrington", "wgan": "Wigan",
       "wigan": "Wigan", "york": "York RLFC Knights"}
NAME_ROWS = (list(range(11, 18)) + list(range(19, 25)) + list(range(26, 30))
             + list(range(31, 35)))                    # slots 1-21
TRY = re.compile(r"\[(\d+)\]")


def _num(v):
    if v is None:
        return np.nan
    s = str(v).strip().replace("−", "-").lower()
    if s in ("", "pk", "pick", "0"):
        return 0.0 if s in ("pk", "pick", "0") else np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _two(v):
    """'35.5(34.5)' or '34.5[32.5]' -> (35.5, 34.5); a plain number -> (x, nan)."""
    s = str(v or "").strip()
    m = re.match(r"\s*([+-]?\d+(?:\.\d+)?)\s*(?:[\(\[]\s*([+-]?\d+(?:\.\d+)?)\s*[\)\]])?", s)
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2)) if m.group(2) else np.nan


def _score(v):
    m = re.match(r"\s*(\d+)\s*-\s*(\d+)", str(v or ""))
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)


def read_workbooks():
    sheets, lines = [], []
    for f in FILES:
        season = int(re.search(r"(20\d\d)", os.path.basename(f)).group(1))
        wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
        for name in wb.sheetnames:
            if name in NOT_CLUBS:
                continue
            team = SHEET[name]
            rows = list(wb[name].iter_rows(min_row=1, max_row=35, values_only=True))
            seq = 0
            for col in range(3, len(rows[0]), 2):
                head = rows[0][col]
                if not isinstance(head, str) or not head.strip():
                    continue
                cell = lambda r: rows[r][col] if col < len(rows[r]) else None
                opp = OPP.get(str(cell(2) or "").strip().lower())
                ha = str(cell(1) or "").strip().lower()
                if opp is None or ha in ("bye", "rest"):
                    continue
                seq += 1
                sf, sa = _score(cell(5))
                tl, tl2 = _two(cell(9))
                hc, hc2 = _two(cell(4))
                lines.append(dict(season=season, team=team, opposition=opp, seq=seq,
                                  week=head.strip(),
                                  venue="neutral" if "neutral" in ha or "[" in ha
                                  else ha, handicap=hc, handicap_alt=hc2,
                                  total_line=tl, total_line_alt=tl2,
                                  score_for=sf, score_against=sa,
                                  weather=str(cell(6) or "").strip().lower() or None))
                for slot, r in enumerate(NAME_ROWS, start=1):
                    raw = cell(r)
                    if raw is None or not str(raw).strip():
                        continue
                    raw = str(raw).strip()
                    t = TRY.search(raw)
                    sheets.append(dict(season=season, team=team, opposition=opp, seq=seq,
                                       slot=slot, name_raw=raw,
                                       name=sp.clean_name(TRY.sub("", raw).strip()),
                                       tries_marked=int(t.group(1)) if t else 0))
    S, L = pd.DataFrame(sheets), pd.DataFrame(lines)
    S["role"] = np.select([S.slot <= 13, S.slot <= 17], ["start", "bench"], "reserve")
    return S, L


def stats_perform(con):
    pm = pd.read_sql("SELECT player_id, player, team, opposition, season, round, position, "
                     "minutes, tries FROM player_match_stats WHERE competition='SL'", con)
    pm["player_id"] = sp.normalize_player_id(pm.player_id)
    pm["slug"] = pm.player.map(sp.clean_name).map(sp.slugify)
    return pm


def resolve(S, pm):
    """player_id per named player: club-season roster, then the season, exact then close."""
    club = {k: dict(zip(g.slug, g.player_id)) for k, g in
            pm.drop_duplicates(["team", "season", "slug"]).groupby(["team", "season"])}
    season = {k: dict(zip(g.slug, g.player_id)) for k, g in
              pm.drop_duplicates(["season", "slug"]).groupby("season")}
    def surname(book, s):
        """Unique surname in the roster (with the first initial when one is given):
        'Matt Dufty' -> 'matthew-dufty', 'thompson' -> the only Thompson."""
        parts = s.split("-")
        last, first = parts[-1], (parts[0][0] if len(parts) > 1 else None)
        hits = [k for k in book if k.split("-")[-1] == last
                and (first is None or k[0] == first)]
        return hits[0] if len(set(book[h] for h in hits)) == 1 and hits else None

    ids, how = [], []
    for name, team, yr in zip(S.name, S.team, S.season):
        s = sp.slugify(name)
        found = None
        for label, book in (("club", club.get((team, yr), {})),
                            ("season", season.get(yr, {}))):
            if s in book:
                found = (book[s], label)
                break
            close = difflib.get_close_matches(s, list(book), n=1, cutoff=0.88)
            if close:
                found = (book[close[0]], label + "_close")
                break
            sur = surname(book, s)
            if sur:
                found = (book[sur], label + "_surname")
                break
        ids.append(found[0] if found else None)
        how.append(found[1] if found else "unresolved")
    return S.assign(player_id=ids, id_method=how)


def align(S, L, pm):
    """Map each club's sequence of weeks to Stats Perform rounds."""
    fx = (pm.groupby(["season", "team", "opposition", "round"]).player_id.agg(set)
            .reset_index().sort_values(["season", "team", "round"]))
    named = S[S.role != "reserve"].groupby(["season", "team", "seq"]).player_id.agg(
        lambda x: set(v for v in x if v))
    # One assignment per club-season: weeks to fixtures, maximising player overlap, with
    # a bonus for the same opposition and a small one for keeping the order. Overlap is
    # the strong signal: 2021 was rescheduled so heavily that order alone misplaces weeks.
    from scipy.optimize import linear_sum_assignment
    out = []
    for (yr, team), g in L.groupby(["season", "team"]):
        cand = fx[(fx.season == yr) & (fx.team == team)].reset_index(drop=True)
        g = g.sort_values("seq").reset_index(drop=True)
        if cand.empty:
            out += [(yr, team, q, np.nan, 0.0) for q in g.seq]
            continue
        ov = np.zeros((len(g), len(cand)))
        for i, r in g.iterrows():
            mine = named.get((yr, team, r.seq), set())
            ov[i] = [len(s & mine) / max(len(s), 1) for s in cand.player_id]
        same = (g.opposition.values[:, None] == cand.opposition.values[None, :])
        rank = np.abs(np.arange(len(g))[:, None] / max(len(g) - 1, 1)
                      - np.arange(len(cand))[None, :] / max(len(cand) - 1, 1))
        score = ov + 0.3 * same - 0.1 * rank
        ri, ci = linear_sum_assignment(-score)
        got = dict(zip(ri, ci))
        for i, r in g.iterrows():
            j = got.get(i)
            # accept a fixture that shares the players, or the same opposition when the
            # week names nobody we could resolve
            if j is not None and (ov[i, j] >= 0.5 or (same[i, j] and ov[i, j] == 0
                                                       and not named.get((yr, team, r.seq)))):
                out.append((yr, team, r.seq, int(cand.loc[j, "round"]), float(ov[i, j])))
            else:
                out.append((yr, team, r.seq, np.nan, float(ov[i, j]) if j is not None else 0.0))
    A = pd.DataFrame(out, columns=["season", "team", "seq", "round", "overlap"])
    return A


def validate(S, L, A, pm):
    L = L.merge(A, on=["season", "team", "seq"], how="left")
    S = S.merge(A, on=["season", "team", "seq"], how="left")
    rep = []
    ok = L["round"].notna()
    rep.append(f"- Weeks: {len(L):,} club-matches read; aligned to a Stats Perform round: "
               f"{ok.mean():.1%}; median share of Stats Perform's players found among the "
               f"named 17: {L.overlap[ok].median():.0%} (below 70%: "
               f"{int((L.overlap[ok] < .7).sum())}).")
    played = set(zip(pm.player_id, pm.season, pm["round"], pm.team))
    S["in_sp"] = [(p, y, r, t) in played if p and pd.notna(r) else np.nan
                  for p, y, r, t in zip(S.player_id, S.season, S["round"], S.team)]
    for role in ("start", "bench", "reserve"):
        x = S[(S.role == role) & S.player_id.notna() & S["round"].notna()]
        rep.append(f"- {role}: {len(x):,} names resolved; in Stats Perform for that match: "
                   f"{x.in_sp.mean():.1%}")
    rep.append(f"- Names resolved to a player: {S.player_id.notna().mean():.1%} "
               f"({dict(S.id_method.value_counts())})")
    # tries marked against Stats Perform's tries
    t = S[S.player_id.notna() & S["round"].notna() & (S.role != "reserve")].merge(
        pm[["player_id", "season", "round", "team", "tries"]],
        on=["player_id", "season", "round", "team"], how="inner")
    rep.append(f"- Tries marked equal Stats Perform's tries on {(t.tries_marked == t.tries).mean():.1%} "
               f"of {len(t):,} player-matches.")
    # both sides' lines
    both = L[ok].merge(L[ok], left_on=["season", "round", "team", "opposition"],
                       right_on=["season", "round", "opposition", "team"],
                       suffixes=("", "_o"))
    h = both.dropna(subset=["handicap", "handicap_o"])
    rep.append(f"- Matches with both clubs' sheets: {len(both) // 2:,}; handicaps that are "
               f"mirror images (sum to 0): {(np.abs(h.handicap + h.handicap_o) < 0.01).mean():.1%}; "
               f"totals equal: {(np.abs(both.total_line - both.total_line_o) < 0.01).mean():.1%}; "
               f"scores consistent: {((both.score_for == both.score_against_o)).mean():.1%}.")
    return S, L, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{rb.DB}?mode=ro", uri=True)
    pm = stats_perform(con)
    con.close()
    S, L = read_workbooks()
    S = resolve(S, pm)
    A = align(S, L, pm)
    S, L, rep = validate(S, L, A, pm)
    by = L.groupby("season").agg(club_matches=("seq", "size"),
                                 aligned=("round", lambda r: r.notna().mean()),
                                 handicap=("handicap", lambda h: h.notna().mean()),
                                 total=("total_line", lambda h: h.notna().mean()))
    text = ["# Super League history workbooks", "",
            f"Files: {', '.join(os.path.basename(f) for f in FILES)}", "",
            "## Validation", ""] + rep + ["", "## By season", "", "```",
                                          by.round(3).to_string(), "```", ""]
    open(REPORT, "w", encoding="utf-8").write("\n".join(text))
    print("\n".join(text))
    if not a.write:
        return 0
    S = S[S["round"].notna()].drop(columns=["seq"])
    L = L[L["round"].notna()].drop(columns=["seq"])
    S["round"], L["round"] = S["round"].astype(int), L["round"].astype(int)
    with runtime.guarded_write("sl_history", note=f"{len(L)} club-matches, {len(S)} names"):
        w = sqlite3.connect(rb.DB)
        S.to_sql("sl_teamsheets", w, if_exists="replace", index=False)
        L.to_sql("sl_match_lines", w, if_exists="replace", index=False)
        w.execute("CREATE INDEX IF NOT EXISTS ix_slts ON sl_teamsheets(season, round, team)")
        w.commit()
        w.close()
    print("wrote sl_teamsheets and sl_match_lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
