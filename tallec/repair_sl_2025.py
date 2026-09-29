# -*- coding: utf-8 -*-
"""Repair the Super League master's 2025 season, which is shuffled three ways.

What is wrong, established by `audit_master.py` and by checks that need nothing but
the file itself:

  * The `Round` column is scrambled. Rows labelled round 1 carry dates from February to
    July, and 74 times a club is booked to play twice in the same round. Against the
    dates, round order correlates at +0.22 in 2025 where the clean 2024 gives +0.98.
  * The `A_*` / `B_*` statistic blocks are attached to the wrong rows. Each block is a
    genuine, complete team-match vector — every club's season totals come out right —
    but the A side and the B side of a row come from two different matches, which is
    why the mean absolute margin reads 7.0 where the real one is 19.9.

What is intact, and is what makes the repair possible rather than a rebuild:

  * `A Team`, `B Team` and `Home Advantage` are correct and agree with each other. With
    them the home side wins 58-65% of the time in the clean seasons, which is the rate
    a real league produces; a misaligned venue column would give 50%.
  * `A Score` / `B Score` (with `Home Score` / `Away Score` and `Date`) hold the row's
    own real scoreline — they match the pairing's actual result 92.1% of the time, the
    same rate as the clean seasons. Their A/B order follows the `Match ID` rather than
    `A Team`, which is a stale orientation dating from when rows were turned around to
    put the home side first; the statistic blocks were turned and these were not.

So each row knows which fixture it is and where it was played, and the player database
knows which round that fixture belongs to and what each side scored. The repair puts
them back together:

  1. Each row's fixture is identified by its pairing and its scoreline, which gives the
     true round.
  2. Each statistic block is identified by a fingerprint of six count statistics summed
     from the players who played — 327 of 328 resolve to exactly one team-match, with no
     two blocks pointing at the same one, so the reattachment is a permutation and not a
     choice.
  3. Blocks are reattached to their own fixture, oriented to `A Team`, and everything
     derived from them is recomputed: the differentials, the score columns, the outcome,
     and the ELO history.

Rows whose fixture or blocks cannot be resolved are left exactly as they were and
reported, never guessed at.

Nothing is written to `SL_master.xlsx`. The output is a separate file, and
`audit_master.py` pointed at it is the acceptance test.

    python repair_sl_2025.py --dry-run     # resolve and report, write nothing
    python repair_sl_2025.py               # write SL_master_2025_repaired.xlsx
    python repair_sl_2025.py --self-test   # run the same logic against clean 2024
"""
import argparse
import os
import sqlite3
import sys
from collections import Counter

import numpy as np
import pandas as pd

import team_map as tm

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "tallec.db")
SEASON = 2025

# Six counting statistics, summed over the players who took the field. Counts rather
# than rates or percentages, because a rate is rounded in the master and would not
# compare exactly. Six is more than enough: across every Super League season this key
# produces no collisions at all.
FINGERPRINT = ["Ball Runs - Total", "Ball Runs - Metres Carried", "Line Break",
               "Tackle Break", "Try Scored - Total", "Ball Runs - Post Contact Metres"]

# columns that describe the fixture rather than the performance, and are kept as they are
IDENTITY = ["Season", "A Team", "B Team", "Home Advantage"]

# derived from the statistic blocks, so recomputed rather than carried over
DERIVED = ["Margin", "Total", "A_Win", "Played", "Home_flag",
           "ELO_A", "ELO_B", "Diff ELO", "ELO_Sum", "ELO_Diff_Abs"]

# the pipeline reuses rolling-form columns when it finds them, so leaving stale ones in
# place would quietly survive the repair; dropping them forces a recomputation
FORM_PREFIXES = ("A_Form_", "B_Form_", "Diff_Form_", "Sum_Form_")


def _xladder_dir():
    here = BASE
    for _ in range(4):
        if os.path.basename(here) == "NRL_2026_MPT":
            return here
        cand = os.path.join(here, "NRL_2026_MPT")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    raise FileNotFoundError("NRL_2026_MPT checkout not found; needed for the ELO helper")


def player_side(season, db=DB):
    """Per team per round: the score, and the fingerprint of the statistic block."""
    sel = ", ".join(f'SUM(COALESCE(r."{c}", 0)) AS "{c}"' for c in FINGERPRINT)
    con = sqlite3.connect(db)
    p = pd.read_sql(f'''
        SELECT s.round, s.team, s.opposition, COUNT(*) AS n_players,
               SUM(4 * COALESCE(r."Try Scored - Total", 0)
                 + 2 * COALESCE(r."Conversion - Made", 0)
                 + 2 * COALESCE(r."Penalty Goal - Made", 0)
                 + 1 * COALESCE(r."Field Goal - 1 Point Made", 0)
                 + 2 * COALESCE(r."Field Goal - 2 Point Made", 0)) AS pts, {sel}
        FROM player_match_stats s
        JOIN player_match_raw r ON r.player_id = s.player_id
         AND r.Competition = s.competition AND r.Season = s.season
         AND r."Round" = s.round
        WHERE s.competition = 'SL' AND s.season = ? GROUP BY 1, 2, 3''',
                    con, params=(season,))
    con.close()
    p["round"] = pd.to_numeric(p["round"], errors="coerce")
    return p.dropna(subset=["round"]).astype({"round": int})


def real_fixtures(p):
    """(round, X, Y) -> (X's score, Y's score), from both sides of the player data."""
    pts = {(r["round"], r.team): r.pts for _, r in p.iterrows()}
    out = {}
    for _, r in p.iterrows():
        other = pts.get((r["round"], r.opposition))
        if other is not None:
            out[(r["round"], r.team, r.opposition)] = (r.pts, other)
    return out


def resolve_rows(m, fixtures, mp):
    """Which real fixture is each master row? Matched on pairing plus scoreline.

    The scoreline is compared as an unordered pair: the row holds the right two numbers
    but in the `Match ID` order rather than the `A Team` order, and which of the two it
    is cannot be assumed. Matching the pair and then reading the orientation off the
    player data settles both questions at once.
    """
    by_pair = {}
    for (rnd, x, y), (sx, sy) in fixtures.items():
        by_pair.setdefault(tuple(sorted((x, y))), []).append((rnd, x, y, sx, sy))
    out, why = {}, Counter()
    for i, r in m.iterrows():
        a, b = mp.get(r["A Team"]), mp.get(r["B Team"])
        if a is None or b is None:
            why["unmapped club"] += 1
            continue
        want = {r["A Score"], r["B Score"]}
        cand = [c for c in by_pair.get(tuple(sorted((a, b))), [])
                if c[1] == a and {c[3], c[4]} == want]
        if len(cand) == 1:
            out[i] = cand[0]
            why["resolved on the scoreline"] += 1
        elif not cand:
            why["no fixture with that scoreline"] += 1
        else:
            why["scoreline occurs more than once"] += 1

    # Some scorelines cannot be matched because the player feed is missing players from
    # that match, so the score rebuilt from it is short. Those rows are still pinned
    # down when everything around them is taken: within one pairing, if a single row and
    # a single fixture are left over, there is nowhere else for either to go. This is
    # elimination, not inference — an ambiguous remainder is left alone.
    for pair, cands in by_pair.items():
        rows = [i for i, r in m.iterrows()
                if i not in out
                and tuple(sorted((mp.get(r["A Team"]), mp.get(r["B Team"])))) == pair]
        if len(rows) != 1:
            continue
        i = rows[0]
        a = mp.get(m.at[i, "A Team"])
        taken = set(out.values())
        free = [c for c in cands if c[1] == a and c not in taken]
        if len(free) == 1:
            out[i] = free[0]
            why["resolved by elimination"] += 1
            why["(of which had failed above)"] += 1
    return out, why


def resolve_blocks(m, p, mp):
    """Which team-match is each statistic block? Matched on the fingerprint."""
    index = {}
    for _, r in p.iterrows():
        index.setdefault((r.team,) + tuple(float(r[c]) for c in FINGERPRINT),
                         []).append((r["round"], r.team))
    out, why = {}, Counter()
    for i, r in m.iterrows():
        for side in ("A", "B"):
            team = mp.get(r[f"{side} Team"])
            hit = index.get((team,) + tuple(float(r[f"{side}_{c}"])
                                            for c in FINGERPRINT), [])
            if len(hit) == 1:
                out[(i, side)] = hit[0]
                why["resolved"] += 1
            else:
                why["no match" if not hit else "ambiguous"] += 1
    seen = Counter(out.values())
    why["two blocks on one team-match"] = sum(1 for v in seen.values() if v > 1)
    return out, why


def repair(m, stats, row_fx, blocks, mp):
    """Rebuild each resolvable row in place. Returns the frame and a per-row status."""
    where = {v: k for k, v in blocks.items()}          # (round, team) -> (row, side)
    out = m.copy()
    status = pd.Series("unrepaired", index=m.index)
    for i, (rnd, a, b, sa, sb) in row_fx.items():
        src_a, src_b = where.get((rnd, a)), where.get((rnd, b))
        if src_a is None or src_b is None:
            status[i] = "block missing"
            continue
        (ra, sida), (rb, sidb) = src_a, src_b
        for stat in stats:
            out.at[i, f"A_{stat}"] = m.at[ra, f"{sida}_{stat}"]
            out.at[i, f"B_{stat}"] = m.at[rb, f"{sidb}_{stat}"]
        out.at[i, "Round"] = rnd
        out.at[i, "Match ID"] = (f"{m.at[i, 'Season']:.0f}-{rnd}-"
                                 f"{m.at[i, 'A Team']}-{m.at[i, 'B Team']}")
        # The scores are taken from the reattached blocks, not from the player data:
        # `A_Points Scored` is the master's own figure for that team-match, whereas a
        # score rebuilt from the players is short whenever the feed is missing one.
        # The score columns are then written in the `A Team` orientation, which is what
        # every other column uses — the stale `Match ID` order caused half of this.
        pa = out.at[i, "A_Points Scored"]
        pb = out.at[i, "B_Points Scored"]
        out.at[i, "A Score"] = pa
        out.at[i, "B Score"] = pb
        ha = m.at[i, "Home Advantage"]
        out.at[i, "Home Score"] = pa if ha == "A" else pb
        out.at[i, "Away Score"] = pb if ha == "A" else pa
        # kept so the caller can check the reattachment against the row's own original
        # scoreline, which came from a different group of columns entirely
        status[i] = ("repaired" if {pa, pb} == {m.at[i, "A Score"], m.at[i, "B Score"]}
                     else "repaired, scoreline differs")
        _ = (sa, sb)
    return out, status


def recompute(df, stats, touched):
    """Differentials, outcomes and the ELO history, from the repaired blocks.

    Only the repaired rows are recomputed. Recomputing the differentials everywhere
    looked harmless and was not: where a statistic is blank on one side the shipped file
    still carries a differential, so `A - B` replaces a real number with a blank. That
    is 408 cells in 2022 alone, in seasons this repair has no business touching.
    """
    at = df.index.isin(touched)
    for stat in stats:
        d = f"Diff_{stat}"
        if d in df.columns:
            df.loc[at, d] = (pd.to_numeric(df.loc[at, f"A_{stat}"], errors="coerce")
                             - pd.to_numeric(df.loc[at, f"B_{stat}"], errors="coerce"))
    played = at & df["A_Points Scored"].notna()
    df.loc[at, "Played"] = df.loc[at, "A_Points Scored"].notna()
    df.loc[played, "Margin"] = df.loc[played, "A_Points Scored"] - df.loc[played, "B_Points Scored"]
    df.loc[played, "Total"] = df.loc[played, "A_Points Scored"] + df.loc[played, "B_Points Scored"]
    df.loc[played, "A_Win"] = (df.loc[played, "Margin"] > 0).astype(int)
    df.loc[at, "Home_flag"] = df.loc[at, "Home Advantage"].map(
        {"A": 1, "B": -1, "neutral": 0}).fillna(0)

    # ELO is stored data, and it is cumulative: a changed 2025 result changes every
    # rating after it. The whole history is walked again from a flat start rather than
    # patched, which also reproduces the untouched seasons and so checks itself.
    sys.path.insert(0, _xladder_dir())
    from xladder_pipeline import update_elos_for_new_matches, ELO_MEAN
    order = df.sort_values(["Season", "Round", "Match ID"]).index
    walked, _ = update_elos_for_new_matches(
        df.loc[order].copy(), {}, int(df["Season"].min()) - 1)
    for c in ("ELO_A", "ELO_B", "Diff ELO"):
        df.loc[walked.index, c] = walked[c]
    df["ELO_Sum"] = df["ELO_A"] + df["ELO_B"]
    df["ELO_Diff_Abs"] = (df["ELO_A"] - df["ELO_B"]).abs()
    _ = ELO_MEAN
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="resolve and report only")
    ap.add_argument("--self-test", action="store_true",
                    help="run against clean 2024 and check it reproduces the original")
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--out", default=os.path.join(
        _xladder_dir(), "SL_master_2025_repaired.xlsx"))
    a = ap.parse_args()
    season = 2024 if a.self_test else a.season

    src = tm.MASTERS["SL"]
    print(f"reading {src}")
    full = pd.read_excel(src)
    full["Match ID"] = full["Match ID"].astype(str)
    stats = [c[2:] for c in full.columns
             if c.startswith("A_") and f"B_{c[2:]}" in full.columns]
    print(f"  {len(full)} rows, {len(stats)} paired statistics")

    mp, _ = tm.solve("SL")
    m = full[full.Season == season]
    p = player_side(season)
    fixtures = real_fixtures(p)
    print(f"\nseason {season}: {len(m)} master rows, {len(fixtures)//2} real fixtures "
          f"in the player data")

    row_fx, why_rows = resolve_rows(m, fixtures, mp)
    print("\nwhich fixture is each row?")
    for k, v in why_rows.most_common():
        print(f"   {v:>4}  {k}")
    blocks, why_blocks = resolve_blocks(m, p, mp)
    print("\nwhich team-match is each statistic block?")
    for k, v in why_blocks.most_common():
        print(f"   {v:>4}  {k}")

    repaired, status = repair(m, stats, row_fx, blocks, mp)
    print("\nrow outcome:")
    for k, v in status.value_counts().items():
        print(f"   {v:>4}  {k}")

    done = status.str.startswith("repaired")
    if done.any():
        chk = repaired[done]
        agree = (status[done] == "repaired").mean()
        mar = (chk["A_Points Scored"] - chk["B_Points Scored"]).abs()
        print(f"\ncheck: reattached block score equals the row's own scoreline "
              f"in {agree:.1%} of repaired rows")
        print(f"check: mean absolute margin now {mar.mean():.1f} "
              f"(was {(m['A_Points Scored'] - m['B_Points Scored']).abs().mean():.1f}, "
              f"real {np.mean([abs(v[0]-v[1]) for v in fixtures.values()]):.1f})")
        dbl = repaired[done].groupby(["Round", "A Team"]).size()
        dbl2 = repaired[done].groupby(["Round", "B Team"]).size()
        print(f"check: clubs booked twice in a round: "
              f"{int((dbl > 1).sum() + (dbl2 > 1).sum())}")

    if a.self_test:
        same = 0
        for i in m.index[done[m.index]]:
            if (repaired.at[i, "Round"] == m.at[i, "Round"]
                    and repaired.at[i, "A_Points Scored"] == m.at[i, "A_Points Scored"]):
                same += 1
        print(f"\nSELF-TEST on {season}: {same}/{int(done.sum())} repaired rows come "
              f"back identical to the original "
              f"({'PASS' if same == int(done.sum()) else 'FAIL'})")
        return

    if a.dry_run:
        print("\ndry run — nothing written")
        return

    repaired = repaired.assign(_repaired=done)
    keep = full[full.Season != season].assign(_repaired=False)
    out = pd.concat([keep, repaired]).sort_values(
        ["Season", "Round", "Match ID"]).reset_index(drop=True)
    touched = out.index[out["_repaired"]]
    out = out.drop(columns=["_repaired"])
    dropped = [c for c in out.columns if c.startswith(FORM_PREFIXES)]
    out = out.drop(columns=dropped)
    print(f"\ndropped {len(dropped)} stale rolling-form columns so they are recomputed")
    out = recompute(out, stats, touched)
    print(f"recomputed differentials and outcomes for {len(touched)} repaired rows only; "
          f"ELO walked over all {len(out)}")
    out.to_excel(a.out, index=False)
    print(f"wrote {a.out}  ({len(out)} rows, {len(out.columns)} columns)")
    print("\nnow run:  python audit_master.py SL   against it before trusting anything")


if __name__ == "__main__":
    main()
