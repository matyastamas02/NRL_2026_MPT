"""
xLadder Weekly Pipeline  v2.1
==============================
Two modes:
  A) run_pipeline(master_path, league)
     Uses the master xlsx as-is (ELO already computed).
  B) run_pipeline_with_new_round(master_path, league, new_matches_df)
     Takes the master + a DataFrame of new matches (from Weekly Input),
     computes ELO forward, appends, then runs the full pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

# ── Constants ─────────────────────────────────────────────────────────────────
ELO_K          = 27      # K-factor (derived from 2026 data)
ELO_MEAN       = 2000    # Mean ELO for new teams
ELO_REGRESSION = 0.30    # Season-start regression toward mean

NRL_STATS = [
    "PTB - Strong Tackle",
    "Kick Chase - Good Chase",
    "Receipt - Falcon",
    "Kick - Crossfield",
]
SL_STATS = [
    "PTB - Strong Tackle",
    "Kick Chase - Good Chase",
    "Receipt - Falcon",
    "Kick - Crossfield",
    "Set Complete - Total",
    "Tackle Break",
    "PTB - Won",
    "Ball Run - Restart Return",
    "Line Break",
    "Pre-Contact Metres",
]

SL_NAMES = {
    "C":   "Castleford Tigers",
    "CF":  "Catalans Dragons",
    "HF":  "Halifax Panthers",
    "HFC": "Huddersfield Giants",
    "HKR": "Hull KR",
    "LB":  "London Broncos",
    "LH":  "Leeds Rhinos",
    "LS":  "Leigh Leopards",
    "SF":  "Salford Red Devils",
    "SH":  "St Helens",
    "TL":  "Toulouse Olympique",
    "WA":  "Wakefield Trinity",
    "WFT": "Warrington Wolves",
    "WI":  "Wigan Warriors",
}

NRL_SHORT = {
    "Brisbane Broncos":               "Brisbane",
    "Canberra Raiders":               "Canberra",
    "Canterbury-Bankstown Bulldogs":  "Canterbury",
    "Cronulla-Sutherland Sharks":     "Cronulla",
    "Dolphins":                       "Dolphins",
    "Gold Coast Titans":              "Gold Coast",
    "Manly-Warringah Sea Eagles":     "Manly",
    "Melbourne Storm":                "Melbourne",
    "New Zealand Warriors":           "NZ Warriors",
    "Newcastle Knights":              "Newcastle",
    "North Queensland Cowboys":       "NQ Cowboys",
    "Parramatta Eels":                "Parramatta",
    "Penrith Panthers":               "Penrith",
    "South Sydney Rabbitohs":         "South Sydney",
    "St. George Illawarra Dragons":   "St George",
    "Sydney Roosters":                "Sydney",
    "Wests Tigers":                   "Wests Tigers",
}

BRAND = {
    "bg":       "#0F172A",
    "panel":    "#1E293B",
    "accent":   "#3B82F6",
    "accent2":  "#8B5CF6",
    "positive": "#10B981",
    "negative": "#EF4444",
    "neutral":  "#64748B",
    "text":     "#F1F5F9",
    "text_dim": "#94A3B8",
    "gold":     "#F59E0B",
    "nrl_blue": "#003087",
    "sl_red":   "#CC0000",
}


# ═══════════════════════════════════════════════════════════════════════════════
# ELO  —  forward-only computation from current state
# ═══════════════════════════════════════════════════════════════════════════════

def get_current_elos(master_df: pd.DataFrame) -> dict:
    """Extract latest ELO for every team from the master DataFrame."""
    df = master_df.sort_values(["Season","Round","Match ID"])
    elos = {}
    for _, row in df.iterrows():
        elos[row["A Team"]] = float(row["ELO_A"])
        elos[row["B Team"]] = float(row["ELO_B"])
    return elos


def update_elos_for_new_matches(
    new_matches: pd.DataFrame,
    current_elos: dict,
    prev_season: int,
    K: float = ELO_K,
) -> pd.DataFrame:
    """
    Given a DataFrame of new played matches (must have A Team, B Team,
    Season, A_Points Scored, B_Points Scored), compute and attach ELO columns.
    Returns the DataFrame with ELO_A, ELO_B, Diff ELO filled in.
    """
    df = new_matches.sort_values(["Season","Round","Match ID"]).copy()
    elos = dict(current_elos)
    elo_a_list = []; elo_b_list = []
    last_season = prev_season

    for _, row in df.iterrows():
        season = int(row["Season"])

        # Season regression
        if season != last_season:
            for t in elos:
                elos[t] = elos[t] + (ELO_MEAN - elos[t]) * ELO_REGRESSION
            last_season = season

        a_team = row["A Team"]
        b_team = row["B Team"]
        elo_a = elos.get(a_team, ELO_MEAN)
        elo_b = elos.get(b_team, ELO_MEAN)

        elo_a_list.append(elo_a)
        elo_b_list.append(elo_b)

        score_a = row["A_Points Scored"]
        score_b = row["B_Points Scored"]
        result_a = 1 if score_a > score_b else 0.5 if score_a == score_b else 0
        exp_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

        elos[a_team] = elo_a + K * (result_a - exp_a)
        elos[b_team] = elo_b + K * ((1 - result_a) - (1 - exp_a))

    df["ELO_A"]    = elo_a_list
    df["ELO_B"]    = elo_b_list
    df["Diff ELO"] = df["ELO_A"] - df["ELO_B"]
    return df, elos


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: str, league: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["League"] = league.upper()

    if "A_Points Scored" not in df.columns and "A Score" in df.columns:
        df = df.rename(columns={"A Score":"A_Points Scored","B Score":"B_Points Scored"})

    if "Home Advantage" in df.columns:
        df["Home_flag"] = df["Home Advantage"].map({"A":1,"B":-1,"neutral":0}).fillna(0)
    else:
        df["Home_flag"] = 1

    df["Played"] = df["A_Points Scored"].notna()

    played = df["Played"]
    df.loc[played,"Margin"] = (
        df.loc[played,"A_Points Scored"] - df.loc[played,"B_Points Scored"]
    )
    df.loc[played,"A_Win"] = (df.loc[played,"Margin"] > 0).astype(int)

    if league.upper() == "SL":
        df["A_Name"] = df["A Team"].map(SL_NAMES).fillna(df["A Team"])
        df["B_Name"] = df["B Team"].map(SL_NAMES).fillna(df["B Team"])
    else:
        df["A_Name"] = df["A Team"].map(NRL_SHORT).fillna(df["A Team"])
        df["B_Name"] = df["B Team"].map(NRL_SHORT).fillna(df["B Team"])

    return df.sort_values(["Season","Round","Match ID"]).reset_index(drop=True)


def append_new_round(master_df: pd.DataFrame, new_rows: pd.DataFrame,
                     league: str) -> pd.DataFrame:
    """
    Merge new round data into the master, compute ELO forward, return full DataFrame.
    new_rows must have at minimum:
      Season, Round, A Team, B Team, Home Advantage,
      A_Points Scored, B_Points Scored,
      + the stat columns for the league.
    """
    # Get current ELO state and last season
    current_elos = get_current_elos(master_df)
    prev_season  = int(master_df["Season"].max())

    # Compute ELO for new rows
    new_rows, _ = update_elos_for_new_matches(new_rows, current_elos, prev_season)

    # Build Match IDs for new rows if missing
    if "Match ID" not in new_rows.columns:
        # Generate unique IDs
        max_id = master_df["Match ID"].max() if "Match ID" in master_df.columns else 0
        new_rows = new_rows.reset_index(drop=True)
        new_rows.insert(0,"Match ID", range(int(max_id)+1, int(max_id)+1+len(new_rows)))

    # Align columns: add missing stat columns as NaN
    for col in master_df.columns:
        if col not in new_rows.columns:
            new_rows[col] = np.nan

    # Append
    combined = pd.concat([master_df, new_rows[master_df.columns]], ignore_index=True)
    combined = combined.sort_values(["Season","Round","Match ID"]).reset_index(drop=True)

    # Re-apply name/flag columns
    if league.upper() == "SL":
        combined["A_Name"] = combined["A Team"].map(SL_NAMES).fillna(combined["A Team"])
        combined["B_Name"] = combined["B Team"].map(SL_NAMES).fillna(combined["B Team"])
    else:
        combined["A_Name"] = combined["A Team"].map(NRL_SHORT).fillna(combined["A Team"])
        combined["B_Name"] = combined["B Team"].map(NRL_SHORT).fillna(combined["B Team"])

    if "Home Advantage" in combined.columns:
        combined["Home_flag"] = combined["Home Advantage"].map(
            {"A":1,"B":-1,"neutral":0}).fillna(0)
    else:
        combined["Home_flag"] = 1

    combined["Played"] = combined["A_Points Scored"].notna()
    played = combined["Played"]
    combined.loc[played,"Margin"] = (
        combined.loc[played,"A_Points Scored"] - combined.loc[played,"B_Points Scored"]
    )
    combined.loc[played,"A_Win"] = (combined.loc[played,"Margin"] > 0).astype(int)

    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# FORM FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def build_form_features(df: pd.DataFrame, stats: list) -> tuple:
    df = df.copy()
    form_cols = []
    for stat in stats:
        a_col = f"A_{stat}"; b_col = f"B_{stat}"
        if a_col not in df.columns:
            continue
        ra = df[["Season","Round","Match ID","A Team",a_col]].copy()
        ra.columns = ["Season","Round","Match ID","Team","Val"]
        rb = df[["Season","Round","Match ID","B Team",b_col]].copy()
        rb.columns = ["Season","Round","Match ID","Team","Val"]
        long = pd.concat([ra,rb],ignore_index=True).sort_values(
            ["Team","Season","Round","Match ID"])
        long["Val"] = pd.to_numeric(long["Val"], errors="coerce")
        fc = f"Form_{stat}"
        long[fc] = long.groupby("Team")["Val"].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1))
        fa = long[["Match ID","Team",fc]].copy()
        fa.columns = ["Match ID","A Team",f"A_{fc}"]
        fb = long[["Match ID","Team",fc]].copy()
        fb.columns = ["Match ID","B Team",f"B_{fc}"]
        df = df.merge(fa, on=["Match ID","A Team"], how="left")
        df = df.merge(fb, on=["Match ID","B Team"], how="left")
        dc = f"Diff_{fc}"
        df[dc] = (pd.to_numeric(df[f"A_{fc}"], errors="coerce")
                  - pd.to_numeric(df[f"B_{fc}"], errors="coerce"))
        form_cols.append(dc)
    return df, list(dict.fromkeys(form_cols))


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_predict(df: pd.DataFrame, form_cols: list, train_seasons: list) -> pd.DataFrame:
    F = ["Diff ELO","Home_flag"] + form_cols
    played = df["Played"]
    train  = df[df["Season"].isin(train_seasons) & played]
    sc = StandardScaler(); sc.fit(train[F].fillna(0))
    wl = LogisticRegression(C=0.05, l1_ratio=0.7, penalty="elasticnet",
                             solver="saga", max_iter=2000, random_state=42)
    wl.fit(sc.transform(train[F].fillna(0)), train["A_Win"])
    mg = Ridge(alpha=1.0)
    mg.fit(sc.transform(train[F].fillna(0)), train["Margin"])
    X_all = sc.transform(df[F].fillna(0))
    df = df.copy()
    df["WL_Pred"]     = wl.predict(X_all)
    df["WL_Prob_A"]   = wl.predict_proba(X_all)[:,1]
    df["Margin_Pred"] = mg.predict(X_all)
    df["Pred_Winner"] = df.apply(
        lambda r: r["A_Name"] if r["WL_Pred"]==1 else r["B_Name"], axis=1)
    df["Pred_Margin_Abs"] = df["Margin_Pred"].abs().round(1)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# xLADDER  (PPG)
# ═══════════════════════════════════════════════════════════════════════════════

def build_xladder(df: pd.DataFrame, season: int) -> pd.DataFrame:
    played = df[(df["Season"]==season) & df["Played"]].copy()
    rows = []
    for team in sorted(set(played["A_Name"]) | set(played["B_Name"])):
        home = played[played["A_Name"]==team]
        away = played[played["B_Name"]==team]
        gp   = len(home) + len(away)
        if gp == 0: continue
        act_pts = 0
        for _,r in home.iterrows():
            m=r["Margin"]; act_pts += 2 if m>0 else 1 if m==0 else 0
        for _,r in away.iterrows():
            m=-r["Margin"]; act_pts += 2 if m>0 else 1 if m==0 else 0
        exp_pts = sum(r["WL_Prob_A"]*2 for _,r in home.iterrows()) + \
                  sum((1-r["WL_Prob_A"])*2 for _,r in away.iterrows())
        rows.append({"Team":team,"GP":gp,
                     "Actual_Pts":act_pts,"Expected_Pts":round(exp_pts,2),
                     "Actual_PPG":round(act_pts/gp,4),
                     "Expected_PPG":round(exp_pts/gp,4)})
    xl = pd.DataFrame(rows)
    xl["PPG_Diff"] = (xl["Actual_PPG"] - xl["Expected_PPG"]).round(4)
    xl = xl.sort_values("Expected_PPG", ascending=False).reset_index(drop=True)
    xl.index = xl.index + 1
    return xl


# ═══════════════════════════════════════════════════════════════════════════════
# MARGIN TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def build_margin_table(df: pd.DataFrame, season: int) -> pd.DataFrame:
    played = df[(df["Season"]==season) & df["Played"]].copy()
    rows = []
    for team in sorted(set(played["A_Name"]) | set(played["B_Name"])):
        home = played[played["A_Name"]==team]
        away = played[played["B_Name"]==team]
        gp   = len(home) + len(away)
        if gp == 0: continue
        act  = sum(r["Margin"] for _,r in home.iterrows()) + \
               sum(-r["Margin"] for _,r in away.iterrows())
        pred = sum(r["Margin_Pred"] for _,r in home.iterrows()) + \
               sum(-r["Margin_Pred"] for _,r in away.iterrows())
        rows.append({"Team":team,"GP":gp,
                     "Actual_Margin":round(act,1),"Expected_Margin":round(pred,1),
                     "Margin_Diff":round(act-pred,1),
                     "Avg_Act":round(act/gp,2),"Avg_Exp":round(pred/gp,2),
                     "Avg_Diff":round((act-pred)/gp,2)})
    return pd.DataFrame(rows).sort_values("Margin_Diff",ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# NEXT ROUND PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_next_round_predictions(df: pd.DataFrame, season: int) -> pd.DataFrame:
    season_df = df[df["Season"]==season].copy()
    played_rounds   = set(season_df[season_df["Played"]]["Round"].unique())
    all_rounds      = set(season_df["Round"].unique())
    unplayed_rounds = sorted(all_rounds - played_rounds)
    if not unplayed_rounds:
        return pd.DataFrame()
    next_r  = min(unplayed_rounds)
    next_df = season_df[season_df["Round"]==next_r].copy()
    cols    = ["Season","Round","A_Name","B_Name","Home_flag","Diff ELO",
               "WL_Pred","WL_Prob_A","Margin_Pred","Pred_Winner","Pred_Margin_Abs"]
    avail   = [c for c in cols if c in next_df.columns]
    out = next_df[avail].copy()
    out["Line_Open"]  = ""
    out["Line_Close"] = ""
    out["Model_Edge_Open"]  = ""
    out["Model_Edge_Close"] = ""
    return out.reset_index(drop=True)


def compute_edge(predictions_df: pd.DataFrame) -> pd.DataFrame:
    df = predictions_df.copy()
    for col in ["Line_Open","Line_Close"]:
        line = pd.to_numeric(df[col], errors="coerce")
        df[col.replace("Line","Model_Edge")] = df["Margin_Pred"] - (-line)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_brand(fig, ax_list=None):
    fig.patch.set_facecolor(BRAND["bg"])
    if ax_list:
        for ax in ax_list:
            ax.set_facecolor(BRAND["panel"])
            ax.tick_params(colors=BRAND["text_dim"], labelsize=10)
            ax.xaxis.label.set_color(BRAND["text_dim"])
            ax.yaxis.label.set_color(BRAND["text_dim"])
            for spine in ax.spines.values():
                spine.set_edgecolor(BRAND["neutral"])
                spine.set_linewidth(0.5)


def chart_xladder(xl: pd.DataFrame, league: str, season: int) -> plt.Figure:
    n = len(xl)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.55)))
    _apply_brand(fig, [ax])
    y     = np.arange(n)
    teams = xl["Team"].tolist()[::-1]
    exp   = xl["Expected_PPG"].tolist()[::-1]
    act   = xl["Actual_PPG"].tolist()[::-1]
    diff  = xl["PPG_Diff"].tolist()[::-1]
    ranks = xl.index.tolist()[::-1]
    bar_h = 0.35
    ax.barh(y + bar_h/2, exp, bar_h, color=BRAND["accent"], alpha=0.85,
            label="Expected PPG (model)")
    ax.barh(y - bar_h/2, act, bar_h,
            color=[BRAND["positive"] if d>=0 else BRAND["negative"] for d in diff],
            alpha=0.85, label="Actual PPG")
    for i, (e,a,d,rk) in enumerate(zip(exp,act,diff,ranks)):
        col  = BRAND["positive"] if d>=0 else BRAND["negative"]
        sign = "+" if d>=0 else ""
        ax.text(max(e,a)+0.02, i, f"{sign}{d:.2f}",
                va="center", ha="left", color=col, fontsize=9, fontweight="bold")
        ax.text(0.01, i+bar_h/2, f"#{rk}", va="center", ha="left",
                color=BRAND["bg"], fontsize=8, fontweight="bold", zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(teams, color=BRAND["text"], fontsize=10)
    ax.yaxis.set_tick_params(pad=8)
    ax.set_xlabel("Points Per Game", color=BRAND["text_dim"])
    ax.set_xlim(0, max(exp+act)*1.25)
    ax.set_title(f"{league.upper()} {season}  —  xLadder (PPG)",
                 color=BRAND["text"], fontsize=14, fontweight="bold", pad=16)
    ax.legend(loc="lower right", facecolor=BRAND["panel"],
              edgecolor=BRAND["neutral"], labelcolor=BRAND["text"], fontsize=9)
    fig.tight_layout(pad=1.5)
    return fig


def chart_margin_vs_expected(mt: pd.DataFrame, league: str, season: int) -> plt.Figure:
    mt_s = mt.sort_values("Margin_Diff", ascending=True).reset_index(drop=True)
    n = len(mt_s)
    fig, ax = plt.subplots(figsize=(12, max(6, n*0.52)))
    _apply_brand(fig, [ax])
    y      = np.arange(n)
    diffs  = mt_s["Margin_Diff"].values
    colors = [BRAND["positive"] if d>=0 else BRAND["negative"] for d in diffs]
    bars   = ax.barh(y, diffs, color=colors, alpha=0.88, height=0.65)
    for i,(d,bar) in enumerate(zip(diffs,bars)):
        sign = "+" if d>=0 else ""
        x_pos = d+(0.8 if d>=0 else -0.8); ha = "left" if d>=0 else "right"
        ax.text(x_pos, i, f"{sign}{d:.0f}", va="center", ha=ha,
                color=BRAND["text"], fontsize=9, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(mt_s["Team"].tolist(), color=BRAND["text"], fontsize=10)
    ax.axvline(0, color=BRAND["neutral"], linewidth=1.2, linestyle="--")
    for i,row in mt_s.iterrows():
        ax.text(ax.get_xlim()[1]*0.98, i, f"GP {row['GP']}",
                va="center", ha="right", color=BRAND["text_dim"], fontsize=8)
    ax.set_title(
        f"{league.upper()} {season}  —  Actual vs Expected Margin\n"
        f"(+) = outperforming model  |  (−) = underperforming model",
        color=BRAND["text"], fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Cumulative Margin Difference  (Actual − Expected)",
                  color=BRAND["text_dim"])
    pos_p = mpatches.Patch(color=BRAND["positive"], label="Outperforming model")
    neg_p = mpatches.Patch(color=BRAND["negative"], label="Underperforming model")
    ax.legend(handles=[pos_p,neg_p], loc="lower right", facecolor=BRAND["panel"],
              edgecolor=BRAND["neutral"], labelcolor=BRAND["text"], fontsize=9)
    fig.tight_layout(pad=1.5)
    return fig


def chart_next_round(preds: pd.DataFrame, league: str, season: int, round_n) -> plt.Figure:
    n = len(preds)
    if n == 0:
        fig,ax = plt.subplots(figsize=(10,3)); _apply_brand(fig,[ax]); ax.axis("off")
        ax.text(0.5,0.5,"No upcoming fixtures found",transform=ax.transAxes,
                ha="center",va="center",color=BRAND["text"],fontsize=14)
        return fig
    fig_h = max(4, n*1.2+1.5)
    fig,ax = plt.subplots(figsize=(11,fig_h)); _apply_brand(fig,[ax]); ax.axis("off")
    row_h = 1.0/(n+1)
    for i,(_,r) in enumerate(preds.iterrows()):
        y_pos  = 1-(i+1)*row_h-row_h*0.1
        a_name = r.get("A_Name",r.get("A Team",""))
        b_name = r.get("B_Name",r.get("B Team",""))
        prob_a = r.get("WL_Prob_A",0.5)
        marg   = r.get("Margin_Pred",0)
        fav    = a_name if prob_a>=0.5 else b_name
        home_f = r.get("Home_flag",0)
        hl_a   = " 🏠" if home_f==1 else (" ✈" if home_f==-1 else "")
        hl_b   = " 🏠" if home_f==-1 else (" ✈" if home_f==1 else "")
        bg_rect = mpatches.FancyBboxPatch(
            (0.02,y_pos),0.96,row_h*0.82,boxstyle="round,pad=0.01",
            linewidth=0,facecolor=BRAND["panel"],transform=ax.transAxes,zorder=1)
        ax.add_patch(bg_rect)
        a_col = BRAND["accent"] if prob_a>=0.5 else BRAND["text_dim"]
        b_col = BRAND["accent2"] if prob_a<0.5 else BRAND["text_dim"]
        ax.text(0.05,y_pos+row_h*0.5,f"{a_name}{hl_a}",transform=ax.transAxes,
                va="center",ha="left",color=a_col,fontsize=11,
                fontweight="bold" if prob_a>=0.5 else "normal")
        ax.text(0.05,y_pos+row_h*0.15,f"{prob_a:.0%}",transform=ax.transAxes,
                va="center",ha="left",color=a_col,fontsize=10)
        ax.text(0.5,y_pos+row_h*0.5,"vs",transform=ax.transAxes,
                va="center",ha="center",color=BRAND["text_dim"],fontsize=10)
        ax.text(0.95,y_pos+row_h*0.5,f"{b_name}{hl_b}",transform=ax.transAxes,
                va="center",ha="right",color=b_col,fontsize=11,
                fontweight="bold" if prob_a<0.5 else "normal")
        ax.text(0.95,y_pos+row_h*0.15,f"{1-prob_a:.0%}",transform=ax.transAxes,
                va="center",ha="right",color=b_col,fontsize=10)
        ax.text(0.5,y_pos+row_h*0.15,f"Model: {fav} by {abs(marg):.1f}",
                transform=ax.transAxes,va="center",ha="center",
                color=BRAND["gold"],fontsize=9)
    ax.set_title(f"{league.upper()} {season}  —  Round {round_n} Predictions",
                 color=BRAND["text"],fontsize=14,fontweight="bold",pad=14,y=1.0)
    fig.tight_layout(pad=1.5)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER RUN
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(path: str, league: str, target_season: int = None,
                 train_seasons: list = None, new_matches_df: pd.DataFrame = None,
                 raw_df: pd.DataFrame = None) -> dict:
    stats = NRL_STATS if league.upper()=="NRL" else SL_STATS
    if raw_df is not None:
        import tempfile as _tmpfile, os as _os
        with _tmpfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as _tmp:
            raw_df.to_excel(_tmp.name, index=False); _p = _tmp.name
        try:
            df = load_data(_p, league)
        finally:
            _os.unlink(_p)
    else:
        df = load_data(path, league)

    # Optionally append new matches
    if new_matches_df is not None and len(new_matches_df) > 0:
        df = append_new_round(df, new_matches_df, league)

    if target_season is None:
        target_season = int(df["Season"].max())
    if train_seasons is None:
        train_seasons = sorted(df[df["Season"] < target_season]["Season"].unique().tolist())
        if not train_seasons:
            train_seasons = [target_season]

    df, form_cols = build_form_features(df, stats)
    df = train_and_predict(df, form_cols, train_seasons)

    xl  = build_xladder(df, target_season)
    mt  = build_margin_table(df, target_season)
    nxt = get_next_round_predictions(df, target_season)
    next_round = int(nxt["Round"].iloc[0]) if len(nxt) else None

    fig_xl  = chart_xladder(xl, league, target_season)
    fig_mt  = chart_margin_vs_expected(mt, league, target_season)
    fig_nxt = chart_next_round(nxt, league, target_season, next_round)

    return {"df":df, "xladder":xl, "margin_table":mt, "next_round":nxt,
            "next_round_n":next_round, "fig_xladder":fig_xl,
            "fig_margin":fig_mt, "fig_nextround":fig_nxt,
            "season":target_season, "league":league.upper(),
            "current_elos": get_current_elos(df)}


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/charts", exist_ok=True)
    for league, path in [
        ("NRL","/mnt/user-data/uploads/NRL_22_26_all_matches__2_.xlsx"),
        ("SL", "/mnt/user-data/uploads/SL_22_26_all_matches__4_.xlsx"),
    ]:
        print(f"\n{league}...")
        out = run_pipeline(path, league)
        print(f"  xLadder top 3: {out['xladder'].head(3)['Team'].tolist()}")
        print(f"  Next round: {out['next_round_n']}")
        out["fig_xladder"].savefig(f"/home/claude/charts/{league}_xl.png",
            dpi=150, bbox_inches="tight", facecolor=BRAND["bg"])
        plt.close("all")
    print("\nDone.")
