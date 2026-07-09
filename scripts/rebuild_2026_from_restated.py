# -*- coding: utf-8 -*-
"""Rebuild season 2026 rows for NRL_master and SL_master from Mike's
'Model Stats Restarted' CSVs + Wikipedia fixtures (home/away/venue/date).
Base = current Google Sheet state (backed up in sheet_backup_20260709/).
Output: updated_masters/{NRL,SL}_master_updated.csv + validation report.
"""
import pandas as pd, numpy as np, re, os

DL = r"C:/Users/matyas-peter.tamas/Downloads"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DL, "updated_masters"); os.makedirs(OUT, exist_ok=True)

# ---------- name maps ----------
NRL_WIKI2CSV = {
    "Dolphins (NRL)": "Dolphins",
    "Manly Warringah Sea Eagles": "Manly-Warringah Sea Eagles",
}
SL_WIKI2CODE = {
    "York Knights":"YK","Hull F.C.":"HF","St Helens R.F.C.":"SH","Bradford Bulls":"BD",
    "Castleford Tigers":"C","Catalans Dragons":"CF","Huddersfield Giants":"HFC",
    "Hull KR":"HKR","Hull Kingston Rovers":"HKR","Leeds Rhinos":"LH","Leigh Leopards":"LS",
    "Toulouse Olympique":"TL","Wakefield Trinity":"WA","Warrington Wolves":"WFT",
    "Wigan Warriors":"WI",
}
SL_CSV2CODE = {
    "Bradford":"BD","Castleford":"C","Catalans":"CF","Huddersfield":"HFC","Hull":"HF",
    "Hull KR":"HKR","Leeds":"LH","Leigh":"LS","St Helens":"SH","Toulouse":"TL",
    "Wakefield":"WA","Warrington":"WFT","Wigan":"WI","York RLFC Knights":"YK",
}

def clean_num(v):
    """CSV value -> clean string suitable for the sheet ('' if missing)."""
    if pd.isna(v): return ""
    s = str(v).strip()
    if s.endswith("%"): s = s[:-1].strip()
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s): s = s.replace(",", "")
    if re.fullmatch(r"\d{1,2}:\d{2}", s): return s  # time-like, kept verbatim
    return s

def as_float(v):
    s = clean_num(v)
    if s == "" or ":" in s: return np.nan
    try: return float(s)
    except ValueError: return np.nan

def fmt(x):
    if pd.isna(x): return ""
    if float(x) == int(x): return str(int(x))
    return f"{float(x):.2f}".rstrip("0").rstrip(".")

def build(league):
    is_nrl = league == "NRL"
    base = pd.read_csv(os.path.join(DL, f"sheet_backup_20260709/{league}_master.csv"), dtype=str).fillna("")
    csv = pd.read_csv(os.path.join(DL, f"{league} Model Stats Restarted.csv"))
    fx = pd.read_csv(os.path.join(SCRATCH, f"{league.lower()}_fixtures.csv"))
    csv["Round"] = pd.to_numeric(csv["Round"], errors="coerce").astype("Int64")
    max_round = 17 if is_nrl else 16
    fx = fx[fx["Round"] <= max_round].copy()

    # canonical team key per source
    if is_nrl:
        fx["h"] = fx["Home"].map(lambda t: NRL_WIKI2CSV.get(t, t))
        fx["a"] = fx["Away"].map(lambda t: NRL_WIKI2CSV.get(t, t))
        csv["team"] = csv["Team"]; csv["opp"] = csv["Opposition"]
        # code map from existing Match IDs: {season}-{round}-{codeA}-{codeB}
        code = {}
        for _, r in base.iterrows():
            m = re.fullmatch(r"\d{4}-\d+-([A-Z]+)-([A-Z]+)", str(r["Match ID"]))
            if m:
                code[r["A Team"]] = m.group(1); code[r["B Team"]] = m.group(2)
    else:
        fx["h"] = fx["Home"].map(SL_WIKI2CODE); fx["a"] = fx["Away"].map(SL_WIKI2CODE)
        csv["team"] = csv["Team"].map(SL_CSV2CODE); csv["opp"] = csv["Opposition"].map(SL_CSV2CODE)
        code = None
    assert fx["h"].notna().all() and fx["a"].notna().all(), "unmapped wiki team"
    assert csv["team"].notna().all() and csv["opp"].notna().all(), "unmapped csv team"

    # Home-advantage flags: carry over the client's existing flags from the sheet
    # (keyed by round + team pair); new rounds default to 'A' (wiki home team = A).
    season_col = pd.to_numeric(base["Season"], errors="coerce")
    old26 = base[season_col == 2026]
    ha_lookup, ha_conflicts = {}, []
    for _, r in old26.iterrows():
        key = (int(float(r["Round"])), frozenset([r["A Team"], r["B Team"]]))
        flag = r["Home Advantage"]
        home = None if flag == "neutral" else (r["A Team"] if flag == "A" else r["B Team"])
        ha_lookup[key] = (flag, home)

    stat_cols = [c for c in csv.columns if c not in
                 ("Team","Competition","Opposition","Round","Opta Live","team","opp")]
    base_stats = {c[2:] for c in base.columns if c.startswith("A_")}
    used = [s for s in stat_cols if s in base_stats]
    derive_precm = "Pre-Contact Metres" in base_stats and "Pre-Contact Metres" not in stat_cols

    csv_idx = csv.set_index(["Round","team"])
    problems, rows = [], []
    for _, f in fx.iterrows():
        rnd, h, a = int(f["Round"]), f["h"], f["a"]
        try:
            rh, ra = csv_idx.loc[(rnd, h)], csv_idx.loc[(rnd, a)]
        except KeyError:
            problems.append(f"MISSING in csv: R{rnd} {h} vs {a}"); continue
        if isinstance(rh, pd.DataFrame) or isinstance(ra, pd.DataFrame):
            problems.append(f"DUPLICATE csv rows: R{rnd} {h}/{a}"); continue
        if rh["opp"] != a or ra["opp"] != h:
            problems.append(f"OPPONENT mismatch: R{rnd} {h} vs {a} (csv says {rh['opp']}/{ra['opp']})"); continue
        # scores: Wikipedia results are authoritative (csv 'Points Scored' undercounts
        # by 2-4 pts in ~10% of matches); csv value only reported for transparency
        hs, as_ = float(f["HS"]), float(f["AS"])
        chs, cas = as_float(rh["Points Scored"]), as_float(ra["Points Scored"])
        if chs != hs or cas != as_:
            problems.append(f"score fixed R{rnd} {h}-{a}: csv {chs}-{cas} -> wiki {hs}-{as_}")
        ha = "A"
        if is_nrl:
            got = ha_lookup.get((rnd, frozenset([h, a])))
            if got:
                flag, home = got
                if flag == "neutral": ha = "neutral"
                elif home != h:
                    ha_conflicts.append(f"R{rnd}: sheet said {home} home, wiki says {h} (using wiki)")
        ch = code[h] if is_nrl else h
        ca = code[a] if is_nrl else a
        row = {"Match ID": f"2026-{rnd}-{ch}-{ca}", "Season":"2026", "Round":str(rnd),
               "A Team": h, "B Team": a,
               "A Score": fmt(hs), "B Score": fmt(as_),
               "Home Advantage": ha}
        if is_nrl: row["League"] = "NRL"
        else:
            row["Home Score"] = fmt(hs); row["Away Score"] = fmt(as_)
            d = str(f.get("Date") or "")
            m = re.search(r"(\d{1,2}) (\w+) (\d{4})", d)
            if m:
                try: row["Date"] = pd.to_datetime(f"{m.group(1)} {m.group(2)} {m.group(3)}").strftime("%Y-%m-%d")
                except Exception: pass
        for s in used:
            if s == "Points Scored":
                va, vb = hs, as_  # actual final scores, not the csv aggregate
            else:
                va, vb = as_float(rh[s]), as_float(ra[s])
            row[f"A_{s}"] = fmt(va) if s == "Points Scored" else clean_num(rh[s])
            row[f"B_{s}"] = fmt(vb) if s == "Points Scored" else clean_num(ra[s])
            dc = f"Diff_{s}"
            if dc in base.columns:
                row[dc] = fmt(va - vb) if (va==va and vb==vb) else ""
        if derive_precm:
            for pref, src in (("A", rh), ("B", ra)):
                mc, pcm = as_float(src["Ball Runs - Metres Carried"]), as_float(src["Ball Runs - Post Contact Metres"])
                row[f"{pref}_Pre-Contact Metres"] = fmt(mc - pcm) if (mc==mc and pcm==pcm) else ""
            pa, pb = as_float(row["A_Pre-Contact Metres"]), as_float(row["B_Pre-Contact Metres"])
            if "Diff_Pre-Contact Metres" in base.columns and pa==pa and pb==pb:
                row["Diff_Pre-Contact Metres"] = fmt(pa - pb)
        rows.append(row)

    new26 = pd.DataFrame(rows).reindex(columns=base.columns).fillna("")
    season = pd.to_numeric(base["Season"], errors="coerce")
    kept = base[season != 2026]
    out = pd.concat([kept, new26], ignore_index=True)
    out["_s"] = pd.to_numeric(out["Season"], errors="coerce")
    out["_r"] = pd.to_numeric(out["Round"], errors="coerce")
    out = out.sort_values(["_s","_r","Match ID"]).drop(columns=["_s","_r"]).reset_index(drop=True)
    out.to_csv(os.path.join(OUT, f"{league}_master_updated.csv"), index=False)

    print(f"=== {league}: kept {len(kept)} rows (2022-25), new 2026 rows: {len(new26)} ===")
    print("2026 rounds:", sorted(pd.to_numeric(new26["Round"]).unique()))
    print("HomeAdv:", new26["Home Advantage"].value_counts().to_dict())
    print("stats filled per row:", len(used), "+ PreCM derived" if derive_precm else "")
    print("problems:", len(problems))
    for p in problems: print("  ", p)
    if is_nrl:
        print("home conflicts sheet-vs-wiki:", len(ha_conflicts))
        for c in ha_conflicts: print("  ", c)
    return out

nrl = build("NRL")
sl  = build("SL")
