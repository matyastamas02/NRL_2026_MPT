# Copyright (c) 2026 Tamas Matyas-Peter and Adam Asztalos. All rights reserved.
# Proprietary software - see LICENSE. No use or redistribution without written permission.
"""
xLadder Pipeline  v3.0
======================
Models:
  M3    — original (4 NRL / 10 SL stats)
  M3+   — enhanced (12 NRL stats, better regularisation)
  Total — ridge regression for total points
Outputs:
  xladder, margin_table, margin_bands, underdog_flags,
  total_points_pred, next_round, hot_cold
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

# ── Team name maps ─────────────────────────────────────────────────────────────
SL_NAMES = {
    "BD":"Bradford Bulls","C":"Castleford Tigers","CF":"Catalans Dragons",
    "HF":"Hull FC","HFC":"Huddersfield Giants","HKR":"Hull KR",
    "LB":"London Broncos","LH":"Leeds Rhinos","LS":"Leigh Leopards",
    "SF":"Salford Red Devils","SH":"St Helens","TL":"Toulouse Olympique",
    "WA":"Wakefield Trinity","WFT":"Warrington Wolves","WI":"Wigan Warriors",
    "YK":"York Knights",
}
NRL_SHORT = {
    "Brisbane Broncos":"Brisbane","Canberra Raiders":"Canberra",
    "Canterbury-Bankstown Bulldogs":"Canterbury","Cronulla-Sutherland Sharks":"Cronulla",
    "Dolphins":"Dolphins","Gold Coast Titans":"Gold Coast",
    "Manly-Warringah Sea Eagles":"Manly","Melbourne Storm":"Melbourne",
    "New Zealand Warriors":"NZ Warriors","Newcastle Knights":"Newcastle",
    "North Queensland Cowboys":"NQ Cowboys","Parramatta Eels":"Parramatta",
    "Penrith Panthers":"Penrith","South Sydney Rabbitohs":"South Sydney",
    "St. George Illawarra Dragons":"St George","Sydney Roosters":"Sydney",
    "Wests Tigers":"Wests Tigers",
}

# ── Model stats ────────────────────────────────────────────────────────────────
NRL_STATS = ["PTB - Strong Tackle","Kick Chase - Good Chase","Receipt - Falcon","Kick - Crossfield"]
SL_STATS  = ["PTB - Strong Tackle","Kick Chase - Good Chase","Receipt - Falcon","Kick - Crossfield",
              "Set Complete - Total","Tackle Break","PTB - Won","Ball Run - Restart Return",
              "Line Break","Pre-Contact Metres"]
NRL_STATS_V2 = ["PTB - Strong Tackle","Kick Chase - Good Chase","Receipt - Falcon","Kick - Crossfield",
                "Set Complete - Total","Tackle Break","Line Break","All Possessions - Positive",
                "Ball Runs - Metres Gained","Errors","Penalty - Total","Made Tackle %"]
SL_STATS_V2  = SL_STATS + ["All Possessions - Positive","Ball Runs - Metres Gained",
                             "Errors","Penalty - Total"]

TOTAL_STATS = ["All Possessions - Positive","Errors","Penalty - Total",
               "Set Complete - Total","Ball Runs - Metres Gained","Try Scored - Total"]

MARGIN_BANDS = ["1-6","7-12","13-18","19-24","25-30","30+"]

BRAND = {
    "bg":"#0F172A","panel":"#1E293B","accent":"#3B82F6","accent2":"#8B5CF6",
    "positive":"#10B981","negative":"#EF4444","neutral":"#64748B",
    "text":"#F1F5F9","text_dim":"#94A3B8","gold":"#F59E0B",
    "nrl_blue":"#003087","sl_red":"#CC0000",
}

ELO_K = 27; ELO_MEAN = 2000; ELO_REGRESSION = 0.30


# ═══════════════════════════════ ELO ══════════════════════════════════════════

def get_current_elos(df):
    elos = {}
    for _,r in df.sort_values(["Season","Round","Match ID"]).iterrows():
        elos[r["A Team"]] = float(r["ELO_A"])
        elos[r["B Team"]] = float(r["ELO_B"])
    return elos

def update_elos_for_new_matches(new_matches, current_elos, prev_season, K=ELO_K):
    df = new_matches.sort_values(["Season","Round","Match ID"]).copy()
    elos = dict(current_elos); elo_a=[]; elo_b=[]; last_s=prev_season
    for _,row in df.iterrows():
        s=int(row["Season"])
        if s!=last_s:
            for t in elos: elos[t]+=( ELO_MEAN-elos[t])*ELO_REGRESSION
            last_s=s
        ea=elos.get(row["A Team"],ELO_MEAN); eb=elos.get(row["B Team"],ELO_MEAN)
        elo_a.append(ea); elo_b.append(eb)
        sa=row["A_Points Scored"]; sb=row["B_Points Scored"]
        res=1 if sa>sb else 0.5 if sa==sb else 0
        exp=1/(1+10**((eb-ea)/400))
        elos[row["A Team"]]=ea+K*(res-exp)
        elos[row["B Team"]]=eb+K*((1-res)-(1-exp))
    df["ELO_A"]=elo_a; df["ELO_B"]=elo_b; df["Diff ELO"]=df["ELO_A"]-df["ELO_B"]
    return df, elos


# ═══════════════════════════ DATA LOADING ════════════════════════════════════

def load_data(path, league):
    df = pd.read_excel(path)
    df["League"] = league.upper()
    if "A_Points Scored" not in df.columns and "A Score" in df.columns:
        df=df.rename(columns={"A Score":"A_Points Scored","B Score":"B_Points Scored"})
    if "Home Advantage" in df.columns:
        df["Home_flag"]=df["Home Advantage"].map({"A":1,"B":-1,"neutral":0}).fillna(0)
    else:
        df["Home_flag"]=1
    df["Played"]=df["A_Points Scored"].notna()
    played=df["Played"]
    df.loc[played,"Margin"]=df.loc[played,"A_Points Scored"]-df.loc[played,"B_Points Scored"]
    df.loc[played,"Total"] =df.loc[played,"A_Points Scored"]+df.loc[played,"B_Points Scored"]
    df.loc[played,"A_Win"] =(df.loc[played,"Margin"]>0).astype(int)
    nm=NRL_SHORT if league.upper()=="NRL" else SL_NAMES
    df["A_Name"]=df["A Team"].map(nm).fillna(df["A Team"])
    df["B_Name"]=df["B Team"].map(nm).fillna(df["B Team"])
    return df.sort_values(["Season","Round","Match ID"]).reset_index(drop=True)

def append_new_round(master_df, new_rows, league):
    current_elos=get_current_elos(master_df); prev_season=int(master_df["Season"].max())
    new_rows,_=update_elos_for_new_matches(new_rows,current_elos,prev_season)
    if "Match ID" not in new_rows.columns:
        mx=master_df["Match ID"].max() if "Match ID" in master_df.columns else 0
        new_rows=new_rows.reset_index(drop=True)
        new_rows.insert(0,"Match ID",range(int(mx)+1,int(mx)+1+len(new_rows)))
    for col in master_df.columns:
        if col not in new_rows.columns: new_rows[col]=np.nan
    combined=pd.concat([master_df,new_rows[master_df.columns]],ignore_index=True)
    combined=combined.sort_values(["Season","Round","Match ID"]).reset_index(drop=True)
    nm=NRL_SHORT if league.upper()=="NRL" else SL_NAMES
    combined["A_Name"]=combined["A Team"].map(nm).fillna(combined["A Team"])
    combined["B_Name"]=combined["B Team"].map(nm).fillna(combined["B Team"])
    if "Home Advantage" in combined.columns:
        combined["Home_flag"]=combined["Home Advantage"].map({"A":1,"B":-1,"neutral":0}).fillna(0)
    else:
        combined["Home_flag"]=1
    combined["Played"]=combined["A_Points Scored"].notna()
    played=combined["Played"]
    combined.loc[played,"Margin"]=combined.loc[played,"A_Points Scored"]-combined.loc[played,"B_Points Scored"]
    combined.loc[played,"Total"] =combined.loc[played,"A_Points Scored"]+combined.loc[played,"B_Points Scored"]
    combined.loc[played,"A_Win"]=(combined.loc[played,"Margin"]>0).astype(int)
    return combined


# ═══════════════════════ FORM FEATURES ════════════════════════════════════════

def build_form_features(df, stats, window=5):
    df=df.copy(); cols=[]
    for stat in stats:
        a_col=f"A_{stat}"; b_col=f"B_{stat}"
        if a_col not in df.columns: continue
        fc=f"Form_{stat}"; dc=f"Diff_{fc}"
        # Skip if already computed
        if dc in df.columns:
            cols.append(dc); continue
        ra=df[["Season","Round","Match ID","A Team",a_col]].copy(); ra.columns=["Season","Round","Match ID","Team","Val"]
        rb=df[["Season","Round","Match ID","B Team",b_col]].copy(); rb.columns=["Season","Round","Match ID","Team","Val"]
        long=pd.concat([ra,rb],ignore_index=True).sort_values(["Team","Season","Round","Match ID"])
        long["Val"]=pd.to_numeric(long["Val"],errors="coerce")
        long[fc]=long.groupby("Team")["Val"].transform(lambda x:x.rolling(window,min_periods=1).mean().shift(1))
        fa=long[["Match ID","Team",fc]].copy(); fa.columns=["Match ID","A Team",f"A_{fc}"]
        fb=long[["Match ID","Team",fc]].copy(); fb.columns=["Match ID","B Team",f"B_{fc}"]
        if f"A_{fc}" not in df.columns:
            df=df.merge(fa,on=["Match ID","A Team"],how="left")
        if f"B_{fc}" not in df.columns:
            df=df.merge(fb,on=["Match ID","B Team"],how="left")
        df[dc]=pd.to_numeric(df[f"A_{fc}"],errors="coerce")-pd.to_numeric(df[f"B_{fc}"],errors="coerce")
        cols.append(dc)
    return df, list(dict.fromkeys(cols))


# ═══════════════════════ MODEL TRAINING ═══════════════════════════════════════

def train_and_predict(df, form_cols, train_seasons, version="v1"):
    F=["Diff ELO","Home_flag"]+form_cols
    F=[c for c in F if c in df.columns]
    played=df["Played"]
    train=df[df["Season"].isin(train_seasons)&played]
    sc=StandardScaler(); sc.fit(train[F].fillna(0))
    wl=LogisticRegression(C=0.05,l1_ratio=0.7,penalty="elasticnet",
                          solver="saga",max_iter=2000,random_state=42)
    wl.fit(sc.transform(train[F].fillna(0)),train["A_Win"])
    mg=Ridge(alpha=1.0); mg.fit(sc.transform(train[F].fillna(0)),train["Margin"])
    X=sc.transform(df[F].fillna(0))
    df=df.copy()
    suffix=f"_{version}"
    df[f"WL_Pred{suffix}"]=wl.predict(X)
    df[f"WL_Prob_A{suffix}"]=wl.predict_proba(X)[:,1]
    df[f"Margin_Pred{suffix}"]=mg.predict(X)
    df[f"Pred_Winner{suffix}"]=df.apply(
        lambda r: r["A_Name"] if r[f"WL_Pred{suffix}"]==1 else r["B_Name"],axis=1)
    df[f"Pred_Margin_Abs{suffix}"]=df[f"Margin_Pred{suffix}"].abs().round(1)
    return df, sc, wl, mg, F

def train_total_model(df, form_cols, train_seasons):
    """Ridge model for total points prediction."""
    played=df["Played"]
    train=df[df["Season"].isin(train_seasons)&played]
    # Sum features (both teams combined)
    sum_feats=[]
    for fc in form_cols:
        stat=fc.replace("Diff_Form_","")
        a_fc=f"A_Form_{stat}"; b_fc=f"B_Form_{stat}"
        if a_fc in df.columns and b_fc in df.columns:
            col=f"Sum_Form_{stat}"
            df[col]=pd.to_numeric(df[a_fc],errors="coerce")+pd.to_numeric(df[b_fc],errors="coerce")
            sum_feats.append(col)
    df["ELO_Sum"]=df["ELO_A"]+df["ELO_B"]
    df["ELO_Diff_Abs"]=(df["ELO_A"]-df["ELO_B"]).abs()
    F_t=["ELO_Sum","ELO_Diff_Abs"]+sum_feats
    F_t=[c for c in F_t if c in df.columns]
    train_t=df[df["Season"].isin(train_seasons)&played]
    if train_t["Total"].notna().sum()<10:
        df["Total_Pred"]=df["Total"].mean() if played.any() else 45.0
        return df, None, None, F_t
    sc_t=StandardScaler(); sc_t.fit(train_t[F_t].fillna(0))
    mg_t=Ridge(alpha=1.0); mg_t.fit(sc_t.transform(train_t[F_t].fillna(0)),train_t["Total"])
    df["Total_Pred"]=mg_t.predict(sc_t.transform(df[F_t].fillna(0)))
    return df, sc_t, mg_t, F_t


# ═══════════════════════ MARGIN BANDS ════════════════════════════════════════

def assign_margin_band(margin):
    a=abs(margin)
    if a<=6:    return "1-6"
    elif a<=12: return "7-12"
    elif a<=18: return "13-18"
    elif a<=24: return "19-24"
    elif a<=30: return "25-30"
    else:       return "30+"

def build_margin_bands(df, season, version="v1"):
    """Add margin band predictions and probabilities."""
    played=df[(df["Season"]==season)&df["Played"]].copy()
    mp_col=f"Margin_Pred_{version}"
    if mp_col not in df.columns: return pd.DataFrame()
    df_out=df.copy()
    df_out[f"Pred_Band_{version}"]=df_out[mp_col].apply(assign_margin_band)
    df_out["Actual_Band"]=df_out["Margin"].apply(
        lambda x: assign_margin_band(x) if not pd.isna(x) else None)
    # Probability for over/under relative to predicted margin
    # P(home wins) already computed; band distribution is ±1 band around prediction
    return df_out


# ═══════════════════════ xLADDER ═════════════════════════════════════════════

def build_xladder(df, season, version="v1"):
    played=df[(df["Season"]==season)&df["Played"]].copy()
    prob_col=f"WL_Prob_A_{version}"
    if prob_col not in df.columns: prob_col="WL_Prob_A_v1"
    rows=[]
    for team in sorted(set(played["A_Name"])|set(played["B_Name"])):
        home=played[played["A_Name"]==team]; away=played[played["B_Name"]==team]
        gp=len(home)+len(away)
        if gp==0: continue
        act_pts=sum(2 if r["Margin"]>0 else 1 if r["Margin"]==0 else 0 for _,r in home.iterrows())+\
                sum(2 if -r["Margin"]>0 else 1 if r["Margin"]==0 else 0 for _,r in away.iterrows())
        exp_pts=sum(r[prob_col]*2 for _,r in home.iterrows())+\
                sum((1-r[prob_col])*2 for _,r in away.iterrows())
        rows.append({"Team":team,"GP":gp,"Actual_Pts":act_pts,
                     "Expected_Pts":round(exp_pts,2),
                     "Actual_PPG":round(act_pts/gp,4),
                     "Expected_PPG":round(exp_pts/gp,4)})
    xl=pd.DataFrame(rows)
    if len(xl)==0: return xl
    xl["PPG_Diff"]=(xl["Actual_PPG"]-xl["Expected_PPG"]).round(4)
    return xl.sort_values("Expected_PPG",ascending=False).reset_index(drop=True).pipe(lambda d: d.assign(**{d.columns[0]:d[d.columns[0]]}))


# ═══════════════════════ MARGIN TABLE ════════════════════════════════════════

def build_margin_table(df, season, version="v1"):
    played=df[(df["Season"]==season)&df["Played"]].copy()
    mp_col=f"Margin_Pred_{version}"
    if mp_col not in df.columns: mp_col="Margin_Pred_v1"
    rows=[]
    for team in sorted(set(played["A_Name"])|set(played["B_Name"])):
        home=played[played["A_Name"]==team]; away=played[played["B_Name"]==team]
        gp=len(home)+len(away)
        if gp==0: continue
        act=sum(r["Margin"] for _,r in home.iterrows())+sum(-r["Margin"] for _,r in away.iterrows())
        pred=sum(r[mp_col] for _,r in home.iterrows())+sum(-r[mp_col] for _,r in away.iterrows())
        rows.append({"Team":team,"GP":gp,"Actual_Margin":round(act,1),
                     "Expected_Margin":round(pred,1),"Margin_Diff":round(act-pred,1),
                     "Avg_Act":round(act/gp,2),"Avg_Exp":round(pred/gp,2),
                     "Avg_Diff":round((act-pred)/gp,2)})
    return pd.DataFrame(rows).sort_values("Margin_Diff",ascending=False).reset_index(drop=True)


# ═══════════════════════ HOT/COLD ════════════════════════════════════════════

def build_hot_cold(df, season, version="v1", window=3):
    played=df[(df["Season"]==season)&df["Played"]].copy()
    mp_col=f"Margin_Pred_{version}"
    if mp_col not in df.columns: mp_col="Margin_Pred_v1"
    rows=[]
    for _,r in played.iterrows():
        rows.append({"team":r["A_Name"],"round":r["Round"],
                     "over":float(r["Margin"]-r[mp_col])})
        rows.append({"team":r["B_Name"],"round":r["Round"],
                     "over":float(-r["Margin"]-(-r[mp_col]))})
    if not rows: return pd.DataFrame()
    long=pd.DataFrame(rows).sort_values(["team","round"]).reset_index(drop=True)
    out=[]
    for team,grp in long.groupby("team"):
        grp=grp.sort_values("round").reset_index(drop=True)
        vals=[]
        for i in range(len(grp)):
            start=max(0,i-window+1); v=grp["over"][start:i+1].values
            n=len(v); w=np.arange(1,n+1,dtype=float); w/=w.sum()
            vals.append(float(np.dot(v,w)))
        grp["roll_over"]=vals; grp["cum_over"]=grp["over"].cumsum()
        out.append(grp)
    return pd.concat(out,ignore_index=True)


# ═══════════════════════ UNDERDOG FLAGS ═══════════════════════════════════════

def build_underdog_flags(df, season, version="v1", threshold=0.42):
    """Flag matches where model sees value on the underdog."""
    played=df[(df["Season"]==season)&df["Played"]].copy()
    prob_col=f"WL_Prob_A_{version}"
    if prob_col not in df.columns: prob_col="WL_Prob_A_v1"
    mp_col=f"Margin_Pred_{version}"
    if mp_col not in df.columns: mp_col="Margin_Pred_v1"
    rows=[]
    for _,r in played.iterrows():
        p=r[prob_col]; m=r[mp_col]
        is_ug_a=(p<=threshold); is_ug_b=((1-p)<=threshold)
        if is_ug_a or is_ug_b:
            underdog=r["A_Name"] if is_ug_a else r["B_Name"]
            ug_prob=p if is_ug_a else 1-p
            ug_margin=-m if is_ug_a else m  # positive = underdog expected to win
            actual_win=(r["A_Win"]==1 and is_ug_a) or (r["A_Win"]==0 and is_ug_b)
            rows.append({"Round":r["Round"],"Underdog":underdog,
                         "Favourite":r["B_Name"] if is_ug_a else r["A_Name"],
                         "Model_Prob":round(ug_prob,3),"Model_Margin":round(ug_margin,1),
                         "Actual_Win":int(actual_win),"Margin":r["Margin"]*(1 if is_ug_b else -1)})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ═══════════════════════ NEXT ROUND ══════════════════════════════════════════

def get_next_round_predictions(df, season, version="v1"):
    sdf=df[df["Season"]==season].copy()
    played_r=set(sdf[sdf["Played"]]["Round"].unique())
    all_r=set(sdf["Round"].unique())
    unplayed=sorted(all_r-played_r)
    if not unplayed: return pd.DataFrame()
    nr=min(unplayed); ndf=sdf[sdf["Round"]==nr].copy()
    cols=["Season","Round","A_Name","B_Name","Home_flag","Diff ELO",
          f"WL_Pred_{version}",f"WL_Prob_A_{version}",f"Margin_Pred_{version}",
          f"Pred_Winner_{version}",f"Pred_Margin_Abs_{version}"]
    avail=[c for c in cols if c in ndf.columns]
    out=ndf[avail].copy()
    out["Line_Open"]=""; out["Line_Close"]=""; out["H2H_Home"]=""; out["H2H_Away"]=""
    out["Model_Edge_Open"]=""; out["Model_Edge_Close"]=""
    return out.reset_index(drop=True)


# ═══════════════════════ TOTAL POINTS ════════════════════════════════════════

def build_team_total_tendencies(df):
    """Team-level historical total points tendency."""
    played=df[df["Played"]].copy()
    if "Total" not in played.columns:
        played["Total"]=played["A_Points Scored"]+played["B_Points Scored"]
    totals={}
    for team in set(played["A_Name"])|set(played["B_Name"]):
        ht=played[played["A_Name"]==team]["Total"]
        at=played[played["B_Name"]==team]["Total"]
        all_t=pd.concat([ht,at])
        if len(all_t)>=3:
            totals[team]={"mean":all_t.mean(),"std":all_t.std(),"n":len(all_t)}
    return totals

def predict_total(team_a, team_b, tendencies, league_avg=45.4):
    ta=tendencies.get(team_a,{}).get("mean",league_avg)
    tb=tendencies.get(team_b,{}).get("mean",league_avg)
    return (ta+tb)/2


# ═══════════════════════ CHARTS ═══════════════════════════════════════════════

def _brand_ax(fig, ax_list=None):
    fig.patch.set_facecolor(BRAND["bg"])
    if ax_list:
        for ax in ax_list:
            ax.set_facecolor(BRAND["panel"])
            ax.tick_params(colors=BRAND["text_dim"],labelsize=10)
            ax.xaxis.label.set_color(BRAND["text_dim"])
            ax.yaxis.label.set_color(BRAND["text_dim"])
            for sp in ax.spines.values(): sp.set_edgecolor(BRAND["neutral"]); sp.set_linewidth(0.5)

def chart_xladder(xl, league, season):
    n=len(xl); y=np.arange(n)
    teams=xl["Team"].tolist()[::-1]; exp=xl["Expected_PPG"].tolist()[::-1]
    act=xl["Actual_PPG"].tolist()[::-1]; diff=xl["PPG_Diff"].tolist()[::-1]
    ranks=list(range(n,0,-1))
    fig,ax=plt.subplots(figsize=(12,max(6,n*0.55))); _brand_ax(fig,[ax])
    bh=0.35
    ax.barh(y+bh/2,exp,bh,color=BRAND["accent"],alpha=0.85,label="Expected PPG (model)")
    ax.barh(y-bh/2,act,bh,color=[BRAND["positive"] if d>=0 else BRAND["negative"] for d in diff],
            alpha=0.85,label="Actual PPG")
    for i,(e,a,d,rk) in enumerate(zip(exp,act,diff,ranks)):
        col=BRAND["positive"] if d>=0 else BRAND["negative"]
        sign="+" if d>=0 else ""
        ax.text(max(e,a)+0.02,i,f"{sign}{d:.2f}",va="center",ha="left",color=col,fontsize=9,fontweight="bold")
        ax.text(0.01,i+bh/2,f"#{rk}",va="center",ha="left",color=BRAND["bg"],fontsize=8,fontweight="bold",zorder=5)
    ax.set_yticks(y); ax.set_yticklabels(teams,color=BRAND["text"],fontsize=10); ax.yaxis.set_tick_params(pad=8)
    ax.set_xlabel("Points Per Game",color=BRAND["text_dim"]); ax.set_xlim(0,max(exp+act)*1.25)
    ax.set_title(f"{league.upper()} {season}  —  xLadder (PPG)",color=BRAND["text"],fontsize=14,fontweight="bold",pad=16)
    ax.legend(loc="lower right",facecolor=BRAND["panel"],edgecolor=BRAND["neutral"],labelcolor=BRAND["text"],fontsize=9)
    fig.tight_layout(pad=1.5); return fig

def chart_margin_vs_expected(mt, league, season):
    mt_s=mt.sort_values("Margin_Diff",ascending=True).reset_index(drop=True)
    n=len(mt_s)
    fig,ax=plt.subplots(figsize=(12,max(6,n*0.52))); _brand_ax(fig,[ax])
    y=np.arange(n); diffs=mt_s["Margin_Diff"].values
    colors=[BRAND["positive"] if d>=0 else BRAND["negative"] for d in diffs]
    bars=ax.barh(y,diffs,color=colors,alpha=0.88,height=0.65)
    for i,(d,bar) in enumerate(zip(diffs,bars)):
        sign="+" if d>=0 else ""; x_pos=d+(0.8 if d>=0 else -0.8); ha="left" if d>=0 else "right"
        ax.text(x_pos,i,f"{sign}{d:.0f}",va="center",ha=ha,color=BRAND["text"],fontsize=9,fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(mt_s["Team"].tolist(),color=BRAND["text"],fontsize=10)
    ax.axvline(0,color=BRAND["neutral"],linewidth=1.2,linestyle="--")
    for i,row in mt_s.iterrows():
        ax.text(ax.get_xlim()[1]*0.98,i,f"GP {row['GP']}",va="center",ha="right",color=BRAND["text_dim"],fontsize=8)
    ax.set_title(f"{league.upper()} {season}  —  Actual vs Expected Margin\n(+) outperforming  |  (−) underperforming",
                 color=BRAND["text"],fontsize=13,fontweight="bold",pad=14)
    ax.set_xlabel("Cumulative Margin Difference  (Actual − Expected)",color=BRAND["text_dim"])
    pos_p=mpatches.Patch(color=BRAND["positive"],label="Outperforming")
    neg_p=mpatches.Patch(color=BRAND["negative"],label="Underperforming")
    ax.legend(handles=[pos_p,neg_p],loc="lower right",facecolor=BRAND["panel"],
              edgecolor=BRAND["neutral"],labelcolor=BRAND["text"],fontsize=9)
    fig.tight_layout(pad=1.5); return fig


# ═══════════════════════ MASTER PIPELINE ══════════════════════════════════════

def run_pipeline(path, league, target_season=None, train_seasons=None,
                 new_matches_df=None, raw_df=None):
    stats_v1  = NRL_STATS   if league.upper()=="NRL" else SL_STATS
    stats_v2  = NRL_STATS_V2 if league.upper()=="NRL" else SL_STATS_V2

    # Load data
    if raw_df is not None:
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".xlsx",delete=False) as tmp:
            raw_df.to_excel(tmp.name,index=False); _p=tmp.name
        try: df=load_data(_p,league)
        finally: _os.unlink(_p)
    else:
        df=load_data(path,league)

    if new_matches_df is not None and len(new_matches_df)>0:
        df=append_new_round(df,new_matches_df,league)

    if target_season is None: target_season=int(df["Season"].max())
    if train_seasons is None:
        train_seasons=sorted(df[df["Season"]<target_season]["Season"].unique().tolist())
        if not train_seasons: train_seasons=[target_season]

    # Build form features for both model versions
    df,form_v1=build_form_features(df,stats_v1)
    df,form_v2=build_form_features(df,stats_v2)

    # Train both models
    df,sc_v1,wl_v1,mg_v1,F_v1=train_and_predict(df,form_v1,train_seasons,"v1")
    df,sc_v2,wl_v2,mg_v2,F_v2=train_and_predict(df,form_v2,train_seasons,"v2")

    # Total points model
    df,sc_t,mg_t,F_t=train_total_model(df,form_v1,train_seasons)

    # Build outputs
    xl_v1   = build_xladder(df,target_season,"v1")
    xl_v2   = build_xladder(df,target_season,"v2")
    mt_v1   = build_margin_table(df,target_season,"v1")
    mt_v2   = build_margin_table(df,target_season,"v2")
    hc      = build_hot_cold(df,target_season,"v1")
    nxt     = get_next_round_predictions(df,target_season,"v1")
    nr_n    = int(nxt["Round"].iloc[0]) if len(nxt) else None
    ug      = build_underdog_flags(df,target_season,"v1")
    totals  = build_team_total_tendencies(df)

    fig_xl  = chart_xladder(xl_v1,league,target_season)
    fig_mt  = chart_margin_vs_expected(mt_v1,league,target_season)

    return {
        "df":df,"xladder":xl_v1,"xladder_v2":xl_v2,
        "margin_table":mt_v1,"margin_table_v2":mt_v2,
        "hot_cold":hc,"next_round":nxt,"next_round_n":nr_n,
        "underdog_flags":ug,"team_totals":totals,
        "fig_xladder":fig_xl,"fig_margin":fig_mt,
        "season":target_season,"league":league.upper(),
        "current_elos":get_current_elos(df),
        "sc_v1":sc_v1,"wl_v1":wl_v1,"mg_v1":mg_v1,"F_v1":F_v1,
        "sc_v2":sc_v2,"wl_v2":wl_v2,"mg_v2":mg_v2,"F_v2":F_v2,
        "form_cols_v1":form_v1,"form_cols_v2":form_v2,
    }


if __name__=="__main__":
    import os
    os.makedirs("/home/claude/charts",exist_ok=True)
    for league,path in [
        ("NRL","/mnt/user-data/uploads/NRL_22_26_all_matches__2_.xlsx"),
        ("SL", "/mnt/user-data/uploads/SL_22_26_all_matches__4_.xlsx"),
    ]:
        print(f"\n{league}...")
        out=run_pipeline(path,league)
        xl=out["xladder"]; xl_v2=out["xladder_v2"]
        print(f"  xLadder top3 v1: {xl.head(3)['Team'].tolist()}")
        print(f"  xLadder top3 v2: {xl_v2.head(3)['Team'].tolist()}")
        print(f"  Underdogs: {len(out['underdog_flags'])}")
        print(f"  Hot/Cold rows: {len(out['hot_cold'])}")
        out["fig_xladder"].savefig(f"/home/claude/charts/{league}_xl.png",dpi=150,
            bbox_inches="tight",facecolor=BRAND["bg"])
        plt.close("all")
    print("\nDone.")


# ═══════════════════════ GERARD FORMAT PARSER ═════════════════════════════════

FOXSPORTS_TO_MASTER = {
    'Brisbane':'Brisbane Broncos','Canterbury':'Canterbury-Bankstown Bulldogs',
    'Cronulla':'Cronulla-Sutherland Sharks','Dolphins':'Dolphins',
    'Gold Coast Titans':'Gold Coast Titans','Gold Coast':'Gold Coast Titans',
    'Manly':'Manly-Warringah Sea Eagles','Melbourne':'Melbourne Storm',
    'North Queensland':'North Queensland Cowboys',
    'North Queensland Cowboys':'North Queensland Cowboys',
    'Newcastle':'Newcastle Knights','Parramatta':'Parramatta Eels',
    'Penrith':'Penrith Panthers','South Sydney':'South Sydney Rabbitohs',
    'St George Illawarra':'St. George Illawarra Dragons',
    'Sydney Roosters':'Sydney Roosters','Warriors':'New Zealand Warriors',
    'New Zealand Warriors':'New Zealand Warriors','Wests Tigers':'Wests Tigers',
}

MATCHSTATS_TO_MASTER = {
    'Brisbane Broncos':'Brisbane Broncos',
    'Canterbury Bulldogs':'Canterbury-Bankstown Bulldogs',
    'Cronulla Sharks':'Cronulla-Sutherland Sharks','Dolphins':'Dolphins',
    'Gold Coast Titans':'Gold Coast Titans','Manly Sea Eagles':'Manly-Warringah Sea Eagles',
    'Melbourne Storm':'Melbourne Storm','New Zealand Warriors':'New Zealand Warriors',
    'Newcastle Knights':'Newcastle Knights',
    'North Queensland Cowboys':'North Queensland Cowboys',
    'Parramatta Eels':'Parramatta Eels','Penrith Panthers':'Penrith Panthers',
    'South Sydney Rabbitohs':'South Sydney Rabbitohs',
    'St George Illawarra Dragons':'St. George Illawarra Dragons',
    'Sydney Roosters':'Sydney Roosters','Wests Tigers':'Wests Tigers',
}

ODDS_TO_MASTER = {
    'Canberra Raiders':'Canberra Raiders',
    'Canterbury Bulldogs':'Canterbury-Bankstown Bulldogs',
    'St George Illawarra Dragons':'St. George Illawarra Dragons',
    'Manly Sea Eagles':'Manly-Warringah Sea Eagles',
    'Nth Queensland Cowboys':'North Queensland Cowboys',
    **MATCHSTATS_TO_MASTER,
}

GERARD_COL_MAP = {
    'runs':'Ball Runs - Total','run_metres':'Ball Runs - Metres Gained',
    'dummy_half_runs':'Ball Run - Dummy Half Run','tackle_busts':'Tackle Break',
    'post_contact_metres':'Ball Runs - Post Contact Metres',
    'offloads':'Offloads Per Set','linebreaks':'Line Break',
    '20m_restarts':'Set Restart Won','completion_rate':'Completed Sets %',
    'tackled_in_opp_20':'Time In Possession Opp 20',
    'set_restarts':'Set Restart Conceded','tackles':'Tackle - Total Made',
    'missed_tackles':'Tackle - Total Missed','errors':'Errors',
    'penalties_conceded':'Penalty - Total','attacking_kicks':'Kick Chase - Total',
}

GERARD_STATS_M4 = [
    'Ball Runs - Total','Ball Runs - Metres Gained','Tackle Break',
    'Ball Runs - Post Contact Metres','Line Break','Set Restart Won',
    'Tackle - Total Made','Tackle - Total Missed','Errors','Penalty - Total',
]

def _parse_completion(s):
    try:
        if '/' in str(s):
            a,b = str(s).split('/'); return float(a)/float(b)*100 if float(b)>0 else np.nan
        return float(s)
    except: return np.nan

def parse_gerard_round(foxsports_df, match_stats_df, season=2026, base_mid=90200):
    """Convert Gerard foxsports + match_stats CSVs to master-format DataFrame."""
    fs = foxsports_df.copy(); ms = match_stats_df.copy()
    for side in ['home','away']:
        cr = f'completion_rate_{side}'
        if cr in fs.columns: fs[cr] = fs[cr].apply(_parse_completion)
    rows = []
    for i,frow in fs.iterrows():
        ht = FOXSPORTS_TO_MASTER.get(frow['home_team'], frow['home_team'])
        at = FOXSPORTS_TO_MASTER.get(frow['away_team'], frow['away_team'])
        rnd = int(frow['round'])
        score_h = score_a = np.nan
        for _,mrow in ms.iterrows():
            mht = MATCHSTATS_TO_MASTER.get(mrow['home_team'], mrow['home_team'])
            mat = MATCHSTATS_TO_MASTER.get(mrow['away_team'], mrow['away_team'])
            if mht==ht and mat==at:
                score_h = mrow['tries_home']*4 + mrow['conversions_home']*2 + mrow.get('penalty_goals_home',0)*2
                score_a = mrow['tries_away']*4 + mrow['conversions_away']*2 + mrow.get('penalty_goals_away',0)*2
                break
        row = {'Season':season,'Round':rnd,'Match ID':base_mid+i,
               'A Team':ht,'B Team':at,'Home Advantage':'A',
               'A_Points Scored':score_h,'B_Points Scored':score_a}
        for gc,mc in GERARD_COL_MAP.items():
            if mc is None: continue
            if f'{gc}_home' in frow.index: row[f'A_{mc}'] = frow[f'{gc}_home']
            if f'{gc}_away' in frow.index: row[f'B_{mc}'] = frow[f'{gc}_away']
        rows.append(row)
    return pd.DataFrame(rows)

def parse_gerard_odds(odds_df):
    """Convert Gerard odds CSV to fixture list for Betting tab."""
    fixtures = []
    for _,r in odds_df.iterrows():
        ht = ODDS_TO_MASTER.get(r['home_team'], r['home_team'])
        at = ODDS_TO_MASTER.get(r['away_team'], r['away_team'])
        hd = NRL_SHORT.get(ht, ht); ad = NRL_SHORT.get(at, at)
        fixtures.append({
            'hd':hd,'ad':ad,'hc':ht,'ac':at,'ha':'A',
            'h2h_h':float(r['home_odds']) if pd.notna(r.get('home_odds')) else 1.91,
            'h2h_a':float(r['away_odds']) if pd.notna(r.get('away_odds')) else 1.91,
            'line':float(r['home_line']) if pd.notna(r.get('home_line')) else 0.0,
            'lo_h':float(r['home_line_odds']) if pd.notna(r.get('home_line_odds')) else 1.91,
            'lo_a':float(r['away_line_odds']) if pd.notna(r.get('away_line_odds')) else 1.91,
            'total_line':float(r['ou_total_points']) if pd.notna(r.get('ou_total_points')) else 45.5,
            'tot_odds':float(r['total_over_odds']) if pd.notna(r.get('total_over_odds')) else 1.91,
            'date':'',
        })
    return fixtures


# ═══════════════════════ GERARD EXTENDED DATA ════════════════════════════════

PLAYER_STAT_COLS = [
    'all_runs','all_run_metres','kick_return_metres','p_c_m',
    'line_breaks','lba','try_assists','tackle_breaks','hit_ups',
    'offloads','passes','tackles','missed_tackles','ineffective_tackles',
    'kicks','kicking_metres','errors','handling_errors','penalties',
    'forced_drop_outs','tries','points','fantasy','minutes',
]

PLAYER_DISPLAY = {
    'all_run_metres':'Run Metres','p_c_m':'Post Contact M',
    'line_breaks':'Line Breaks','tackle_breaks':'Tackle Breaks',
    'tries':'Tries','tackles':'Tackles','missed_tackles':'Missed Tackles',
    'errors':'Errors','offloads':'Offloads','kicks':'Kicks',
    'kicking_metres':'Kick Metres','fantasy':'Fantasy Pts',
    'all_runs':'Runs','hit_ups':'Hit Ups',
}

def aggregate_player_to_team(player_df):
    """Aggregate player-level stats to team level per round."""
    numeric = [c for c in PLAYER_STAT_COLS if c in player_df.columns]
    for col in numeric:
        player_df[col] = pd.to_numeric(player_df[col], errors='coerce')
    agg = player_df.groupby(['round','team'])[numeric].sum().reset_index()
    return agg

def build_halfline_table(tryscorer_df):
    """Build 1H vs 2H scoring table from tryscorer data."""
    rows = []
    for _,r in tryscorer_df.iterrows():
        total = (r.get('home_1H_tries',0)+r.get('home_2H_tries',0)+
                 r.get('away_1H_tries',0)+r.get('away_2H_tries',0))
        if total == 0: continue
        # Home team
        rows.append({'team':r['home_team'],'round':r['round'],
                     '1H':r.get('home_1H_tries',0),'2H':r.get('home_2H_tries',0),
                     'total':r.get('home_1H_tries',0)+r.get('home_2H_tries',0)})
        # Away team
        rows.append({'team':r['away_team'],'round':r['round'],
                     '1H':r.get('away_1H_tries',0),'2H':r.get('away_2H_tries',0),
                     'total':r.get('away_1H_tries',0)+r.get('away_2H_tries',0)})
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def get_top_players(player_df, stat='all_run_metres', n=20, rounds=None):
    """Get top n players for a given stat, optionally filtered to recent rounds."""
    df = player_df.copy()
    if rounds:
        df = df[df['round'].isin(rounds)]
    df[stat] = pd.to_numeric(df[stat], errors='coerce')
    grp = df.groupby(['team','player'])[stat].sum().reset_index()
    return grp.sort_values(stat, ascending=False).head(n)

def get_form_players(player_df, stat='all_run_metres', window=3):
    """Get players trending up in recent rounds."""
    df = player_df.copy()
    df[stat] = pd.to_numeric(df[stat], errors='coerce')
    all_rounds = sorted(df['round'].unique())
    recent = all_rounds[-window:]
    older  = all_rounds[:-window]
    if not older: return pd.DataFrame()
    rec  = df[df['round'].isin(recent)].groupby(['team','player'])[stat].mean().reset_index()
    rec.columns  = ['team','player','recent_avg']
    old  = df[df['round'].isin(older)].groupby(['team','player'])[stat].mean().reset_index()
    old.columns  = ['team','player','prev_avg']
    merged = rec.merge(old, on=['team','player'], how='inner')
    merged['trend'] = merged['recent_avg'] - merged['prev_avg']
    merged['trend_pct'] = merged['trend'] / merged['prev_avg'].replace(0, np.nan) * 100
    return merged.sort_values('trend', ascending=False)

def parse_play_by_play_momentum(pbp_df, match_url=None):
    """Extract score momentum: lead changes, comeback data."""
    df = pbp_df.copy()
    if match_url:
        df = df[df['match_url']==match_url]
    df['score_diff'] = df['home_score'] - df['away_score']
    df['lead_changes'] = (df['score_diff'].apply(np.sign).diff().fillna(0) != 0).sum()
    return df
