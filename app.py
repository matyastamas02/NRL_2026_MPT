"""
xLadder Pro  v3.0
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, os, sys, tempfile
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from xladder_pipeline import (
    run_pipeline, get_current_elos, update_elos_for_new_matches,
    NRL_STATS, SL_STATS, BRAND, SL_NAMES, NRL_SHORT,
)

BASE_DIR   = Path(__file__).parent
NRL_MASTER = BASE_DIR / "NRL_master.xlsx"
SL_MASTER  = BASE_DIR / "SL_master.xlsx"
ELO_K      = 27

STAT_CATS = {
    "Attack":    ["Ball Runs - Total","Ball Runs - Metres Gained","Ball Run - Run",
                  "Ball Run - Run Metres","Line Break","Kick Line Break","Tackle Break",
                  "Try Scored - Total","Offloads Per Set","Ball Runs - Post Contact Metres",
                  "Pre-Contact Metres","All Possessions - Positive","Good Ball Sets",
                  "Yardage Sets","Supports - General"],
    "Defence":   ["Tackle - Total Made","Tackle - Total Missed","Tackle - Total Ineffective",
                  "Tackles - Total Atempted","Made Tackle %","Set Restart Conceded","Set Restart Won"],
    "Kicking":   ["Kick Chase - Good Chase","Kick Chase - Total","Kick - Crossfield",
                  "Kick - Grubber","Kick - Bomb","Receipt - Falcon","Receipt - Total",
                  "Ball Run - Kick Return Metres"],
    "Possession":["Possession %","Territory %","Time In Possession (Seconds)",
                  "Time in Possession Opp Half (Seconds)","Time In Possession Opp 20",
                  "Passes Per Set","Completed Sets %","Ball Run Metres per Set",
                  "PTB - Won","PTB - Strong Tackle","Set Complete - Total",
                  "Set Incomplete - Total","All Possessions","All Possessions - Positive"],
    "Errors":    ["Errors","Errors per Set","Errors - Own Half","Errors - Opposition Half",
                  "Errors - Handling Errors","Penalty - Total","Penalty - Defence","Penalty - Offence"],
}

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1E293B",
    font=dict(family="Arial", color="#F1F5F9", size=12),
    xaxis=dict(gridcolor="#334155", linecolor="#334155", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#334155", linecolor="#334155", tickfont=dict(size=11)),
    margin=dict(l=60,r=20,t=40,b=40),
    hoverlabel=dict(bgcolor="#1E293B",bordercolor="#334155",font=dict(color="#F1F5F9")),
)

def apply_theme(fig, **kw):
    # Merge axis dicts instead of overwriting
    merged = dict(PLOTLY_THEME)
    for k, v in kw.items():
        if k in ('xaxis','yaxis') and k in merged and isinstance(v, dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    fig.update_layout(**merged)
    return fig

def team_color(team):
    palette=["#F78166","#79C0FF","#3FB950","#E3B341","#D2A8FF","#FF7B72","#56D364",
             "#58A6FF","#FFA657","#FF8585","#63E6BE","#C9D1D9","#A8DADC","#F4A261",
             "#E76F51","#2EC4B6","#FFD166"]
    teams_sorted=sorted(st.session_state.get("all_teams",[team]))
    idx=teams_sorted.index(team) if team in teams_sorted else 0
    return palette[idx % len(palette)]

def pct_rank(series, val):
    if series.std()==0: return 50
    return int(round((series < val).mean()*100))

def hot_cold(df_in, window=3):
    rows=[]
    for _,r in df_in.iterrows():
        rows.append({"team":r["A_Name"],"round":r["Round"],"over":float(r["Margin"]-r["Margin_Pred"])})
        rows.append({"team":r["B_Name"],"round":r["Round"],"over":float(-r["Margin"]-(-r["Margin_Pred"]))})
    long=pd.DataFrame(rows).sort_values(["team","round"]).reset_index(drop=True)
    out=[]
    for team,grp in long.groupby("team"):
        grp=grp.sort_values("round").reset_index(drop=True)
        vals=[]; cum_vals=[]
        for i in range(len(grp)):
            start=max(0,i-window+1)
            v=grp["over"][start:i+1].values
            n=len(v); w=np.arange(1,n+1,dtype=float); w/=w.sum()
            vals.append(float(np.dot(v,w)))
        grp["roll_over"]=vals
        grp["cum_over"]=grp["over"].cumsum()
        out.append(grp)
    return pd.concat(out,ignore_index=True)

st.set_page_config(page_title="xLadder Pro",page_icon="🏉",layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
body,.stApp{background-color:#0F172A;color:#F1F5F9}
section[data-testid="stSidebar"]{background:#0D1117;border-right:.5px solid #1E293B}
.stTabs [role="tab"]{color:#64748B;font-weight:600;font-size:13px;padding:8px 16px}
.stTabs [aria-selected="true"]{color:#F1F5F9;border-bottom:2px solid #3B82F6}
.stTabs [role="tablist"]{background:#0D1117;border-bottom:.5px solid #1E293B;gap:0}
h1,h2,h3{color:#F1F5F9 !important}
.kpi{background:#1E293B;border-radius:10px;padding:16px 18px;border:.5px solid #334155}
.kpi-label{color:#64748B;font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}
.kpi-val{font-size:26px;font-weight:700;color:#F1F5F9;line-height:1}
.kpi-sub{font-size:11px;color:#475569;margin-top:4px}
.kpi-pos{border-left:3px solid #10B981}
.kpi-neg{border-left:3px solid #EF4444}
.kpi-neu{border-left:3px solid #3B82F6}
.shr{font-size:13px;font-weight:600;color:#94A3B8;text-transform:uppercase;
  letter-spacing:.06em;margin:1.5rem 0 .75rem;padding-bottom:6px;border-bottom:.5px solid #1E293B}
.info{background:#1E3A5F;border-radius:8px;padding:10px 14px;border-left:3px solid #3B82F6;
  color:#93C5FD;font-size:12px;margin:8px 0}
.ok{background:#052e16;border-radius:8px;padding:10px 14px;border-left:3px solid #10B981;
  color:#6EE7B7;font-size:12px;margin:8px 0}
.warn{background:#292108;border-radius:8px;padding:10px 14px;border-left:3px solid #F59E0B;
  color:#FCD34D;font-size:12px;margin:8px 0}
div[data-testid="stDownloadButton"] button{background:#1D4ED8;color:white;border-radius:6px;
  padding:6px 16px;font-weight:600;border:none;font-size:13px}
div[data-testid="stDownloadButton"] button:hover{background:#2563EB}
</style>
""", unsafe_allow_html=True)

# ── Sidebar
with st.sidebar:
    st.markdown("## 🏉 xLadder Pro")
    st.markdown("---")
    league = st.selectbox("League", ["NRL","SL"])
    season = st.number_input("Season", min_value=2022, max_value=2030, value=2026, step=1)
    master_path = NRL_MASTER if league=="NRL" else SL_MASTER
    st.markdown("---")
    if master_path.exists():
        kb=master_path.stat().st_size//1024
        st.markdown(f'<div class="ok">Loaded · {master_path.name} · {kb}KB</div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn">Missing: {master_path.name}</div>',unsafe_allow_html=True)
    st.markdown("---")
    edge_thresh = st.slider("Edge threshold (pts)",1,10,3)
    kelly_frac  = st.slider("Kelly fraction",0.1,1.0,0.25,0.05)
    bankroll    = st.number_input("Bankroll ($)",value=1000,step=100)
    st.markdown("---")
    st.caption("xLadder Pro v3.0  |  M3 model")

if not master_path.exists():
    st.error(f"Upload `{master_path.name}` to the repo.")
    st.stop()

@st.cache_data(show_spinner="Running model...",ttl=180)
def get_result(path_str,league,season):
    return run_pipeline(path_str,league,target_season=int(season))

try:
    R=get_result(str(master_path),league,int(season))
except Exception as e:
    st.error(f"Pipeline error: {e}"); st.exception(e); st.stop()

df=R["df"]; xl=R["xladder"]; mt=R["margin_table"]
current_elos=R["current_elos"]
form_cols=[c for c in df.columns if c.startswith("Diff_Form_")]
played=df[(df["Season"]==int(season))&df["Played"]].copy()
all_pl=df[df["Played"]].copy()
n_pl=len(played)
last_rnd=int(played["Round"].max()) if n_pl>0 else 0
teams=sorted(set(played["A_Name"])|set(played["B_Name"]))
st.session_state["all_teams"]=teams
stats_model=NRL_STATS if league=="NRL" else SL_STATS
name_map=NRL_SHORT if league=="NRL" else SL_NAMES
raw_teams=sorted(set(df["A Team"])|set(df["B Team"]))
rev_map={name_map.get(t,t):t for t in raw_teams}
disp_teams=[name_map.get(t,t) for t in raw_teams]
hc_df=hot_cold(played) if n_pl>0 else pd.DataFrame()
model_flag=set(stats_model)

# ── Retrain model for betting tab (shared)
F=["Diff ELO","Home_flag"]+form_cols
F=[c for c in F if c in df.columns]
train_seasons=sorted(df[df["Season"]<int(season)]["Season"].unique().tolist())
if not train_seasons: train_seasons=[int(season)]
train_data=df[df["Season"].isin(train_seasons)&df["Played"]]
sc_m=StandardScaler(); sc_m.fit(train_data[F].fillna(0))
wl_m=LogisticRegression(C=0.05,l1_ratio=0.7,penalty="elasticnet",solver="saga",max_iter=2000,random_state=42)
wl_m.fit(sc_m.transform(train_data[F].fillna(0)),train_data["A_Win"])
mg_m=Ridge(alpha=1.0); mg_m.fit(sc_m.transform(train_data[F].fillna(0)),train_data["Margin"])

team_elos_m={}
for _,r in df.sort_values(["Season","Round"]).iterrows():
    team_elos_m[r["A Team"]]=float(r["ELO_A"]); team_elos_m[r["B Team"]]=float(r["ELO_B"])
team_form_m={}
for fc in form_cols:
    stat=fc.replace("Diff_Form_",""); a_fc=f"A_Form_{stat}"; b_fc=f"B_Form_{stat}"
    for _,r in df.sort_values(["Season","Round"]).iterrows():
        if a_fc in df.columns and pd.notna(r.get(a_fc)): team_form_m[(r["A Team"],fc)]=float(r[a_fc])
        if b_fc in df.columns and pd.notna(r.get(b_fc)): team_form_m[(r["B Team"],fc)]=float(r[b_fc])

def predict_fx(h_code,a_code,ha):
    elo_h=team_elos_m.get(h_code,2000); elo_a=team_elos_m.get(a_code,2000)
    ha_map={"A":1,"B":-1,"neutral":0}
    feat={"Diff ELO":elo_h-elo_a,"Home_flag":ha_map.get(ha,0)}
    for fc in form_cols: feat[fc]=team_form_m.get((h_code,fc),0)-team_form_m.get((a_code,fc),0)
    X=np.array([[feat.get(f,0) for f in F]])
    return float(wl_m.predict_proba(sc_m.transform(X))[0,1]), float(mg_m.predict(sc_m.transform(X))[0])

# ═══════════ TABS ═══════════════════════════════════════════════════════════
st.markdown(f"# 🏉 xLadder Pro — {league} {season}")
tab1,tab2,tab3,tab4,tab5=st.tabs(["📊 Dashboard","🏆 xLadder","📈 Team Stats","🎯 Betting","📥 Weekly Input"])

# ════════ TAB 1: DASHBOARD ════════════════════════════════════════════════
with tab1:
    if n_pl>0:
        acc=accuracy_score(played["A_Win"],played["WL_Pred"])
        try: auc=roc_auc_score(played["A_Win"],played["WL_Prob_A"])
        except: auc=float("nan")
        top_over=hc_df.groupby("team")["roll_over"].last().idxmax() if len(hc_df) else "—"
        top_under=hc_df.groupby("team")["roll_over"].last().idxmin() if len(hc_df) else "—"
    else:
        acc=auc=0; top_over=top_under="—"

    k1,k2,k3,k4=st.columns(4)
    for col,lab,val,sub,klass in [
        (k1,"Season",f"{league} {season}",f"Round {last_rnd} completed","kpi-neu"),
        (k2,"WL accuracy",f"{acc:.1%}",f"AUC {auc:.3f}","kpi-pos" if acc>=0.6 else "kpi-neg"),
        (k3,"Hottest team",top_over,"Rolling 3-game overperf","kpi-pos"),
        (k4,"Coldest team",top_under,"Rolling 3-game underperf","kpi-neg"),
    ]:
        col.markdown(f'<div class="kpi {klass}"><div class="kpi-label">{lab}</div>'
                     f'<div class="kpi-val">{val}</div><div class="kpi-sub">{sub}</div></div>',
                     unsafe_allow_html=True)
    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="shr">Hot / Cold — rolling 3-game vs model</div>',unsafe_allow_html=True)
        if len(hc_df):
            last_hc=hc_df.groupby("team")["roll_over"].last().sort_values(ascending=True)
            fig_hc=go.Figure(go.Bar(x=last_hc.values,y=last_hc.index,orientation="h",
                marker_color=["#10B981" if v>=0 else "#EF4444" for v in last_hc.values],
                text=[f"{v:+.1f}" for v in last_hc.values],textposition="outside",
                textfont=dict(size=10,color="#F1F5F9"),
                hovertemplate="%{y}: %{x:+.1f}<extra></extra>"))
            fig_hc.add_vline(x=0,line_dash="dash",line_color="#64748B")
            apply_theme(fig_hc,height=420,xaxis_title="Overperformance vs model (pts)")
            st.plotly_chart(fig_hc,use_container_width=True)
    with c2:
        st.markdown('<div class="shr">Model accuracy by season</div>',unsafe_allow_html=True)
        acc_rows=[]
        for s in sorted(all_pl["Season"].unique()):
            sub=all_pl[all_pl["Season"]==s]
            if len(sub)<5: continue
            a=accuracy_score(sub["A_Win"],sub["WL_Pred"])
            try: au=roc_auc_score(sub["A_Win"],sub["WL_Prob_A"])
            except: au=float("nan")
            acc_rows.append({"Season":int(s),"Accuracy":a,"AUC":au,"n":len(sub)})
        if acc_rows:
            adf=pd.DataFrame(acc_rows)
            fig_acc=go.Figure()
            fig_acc.add_trace(go.Bar(x=adf["Season"].astype(str),y=adf["Accuracy"],
                marker_color="#3B82F6",
                text=[f"{v:.1%}" for v in adf["Accuracy"]],textposition="outside",
                hovertemplate="%{x}: %{y:.1%} (n=%{customdata})<extra></extra>",
                customdata=adf["n"]))
            fig_acc.add_hline(y=0.524,line_dash="dot",line_color="#F59E0B",annotation_text="BEP 52.4%")
            fig_acc.add_hline(y=0.62,line_dash="dot",line_color="#10B981",annotation_text="Target 62%")
            apply_theme(fig_acc,height=420,yaxis=dict(tickformat=".0%",range=[0.45,0.75],gridcolor="#334155"))
            st.plotly_chart(fig_acc,use_container_width=True)

    st.markdown('<div class="shr">Current ELO ratings</div>',unsafe_allow_html=True)
    elo_s=sorted(current_elos.items(),key=lambda x:-x[1])
    elo_names=[name_map.get(t,t) for t,_ in elo_s]; elo_vals=[v for _,v in elo_s]
    fig_elo=go.Figure(go.Bar(x=elo_names,y=elo_vals,
        marker_color=["#10B981" if v>=2000 else "#EF4444" for v in elo_vals],
        text=[f"{v:.0f}" for v in elo_vals],textposition="outside",
        hovertemplate="%{x}: %{y:.1f}<extra></extra>"))
    fig_elo.add_hline(y=2000,line_dash="dash",line_color="#64748B",annotation_text="Mean 2000")
    apply_theme(fig_elo,height=300,yaxis=dict(range=[1700,2300],gridcolor="#334155"))
    st.plotly_chart(fig_elo,use_container_width=True)

# ════════ TAB 2: xLADDER ═════════════════════════════════════════════════
with tab2:
    c1,c2=st.columns([1,1.4])
    with c1:
        st.markdown('<div class="shr">Standings — PPG</div>',unsafe_allow_html=True)
        disp=xl[["Team","GP","Expected_PPG","Actual_PPG","PPG_Diff"]].copy()
        disp.columns=["Team","GP","xPPG","Actual PPG","Δ PPG"]
        for c in ["xPPG","Actual PPG","Δ PPG"]: disp[c]=disp[c].round(3)
        def hl(v):
            if isinstance(v,float):
                return f"color:{'#10B981' if v>0 else '#EF4444' if v<0 else '#64748B'};font-weight:bold"
            return ""
        st.dataframe(disp.style.map(hl,subset=["Δ PPG"]),use_container_width=True,height=480)
        st.download_button("Download CSV",disp.to_csv(index=True).encode(),f"{league}_{season}_xladder.csv","text/csv")
    with c2:
        st.markdown('<div class="shr">Expected vs Actual PPG</div>',unsafe_allow_html=True)
        fig_xl=go.Figure()
        fig_xl.add_trace(go.Bar(y=xl["Team"],x=xl["Expected_PPG"],orientation="h",name="xPPG (model)",
            marker_color="#3B82F6",opacity=0.85,hovertemplate="%{y}: xPPG %{x:.3f}<extra></extra>"))
        fig_xl.add_trace(go.Bar(y=xl["Team"],x=xl["Actual_PPG"],orientation="h",name="Actual PPG",
            marker_color=["#10B981" if d>=0 else "#EF4444" for d in xl["PPG_Diff"]],
            opacity=0.85,hovertemplate="%{y}: Actual %{x:.3f}<extra></extra>"))
        for _,row in xl.iterrows():
            sign="+" if row["PPG_Diff"]>=0 else ""
            col="#10B981" if row["PPG_Diff"]>=0 else "#EF4444"
            fig_xl.add_annotation(x=max(row["Expected_PPG"],row["Actual_PPG"])+0.04,y=row["Team"],
                text=f"{sign}{row['PPG_Diff']:.2f}",showarrow=False,
                font=dict(size=10,color=col),xanchor="left")
        fig_xl.update_layout(barmode="overlay",**PLOTLY_THEME,height=500,
            xaxis_title="Points Per Game",
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig_xl,use_container_width=True)

    st.markdown('<div class="shr">ELO trend — all seasons</div>',unsafe_allow_html=True)
    sel_elo=st.multiselect("Select teams",teams,default=teams[:6] if len(teams)>=6 else teams,key="elo_sel")
    if sel_elo and len(all_pl):
        fig_elo_t=go.Figure()
        for team in sel_elo:
            t_orig={v:k for k,v in name_map.items()}.get(team,team)
            sa=all_pl[all_pl["A Team"]==t_orig][["Season","Round","ELO_A"]].copy(); sa.columns=["Season","Round","ELO"]
            sb=all_pl[all_pl["B Team"]==t_orig][["Season","Round","ELO_B"]].copy(); sb.columns=["Season","Round","ELO"]
            sub=pd.concat([sa,sb]).sort_values(["Season","Round"])
            sub["Label"]=sub["Season"].astype(str)+" R"+sub["Round"].astype(str)
            col=team_color(team)
            fig_elo_t.add_trace(go.Scatter(x=list(range(len(sub))),y=sub["ELO"],
                mode="lines+markers",name=team,line=dict(color=col,width=2),marker=dict(size=4,color=col),
                hovertemplate=f"{team} — %{{customdata}}: ELO %{{y:.1f}}<extra></extra>",customdata=sub["Label"]))
        fig_elo_t.add_hline(y=2000,line_dash="dot",line_color="#334155")
        apply_theme(fig_elo_t,height=380,yaxis_title="ELO",xaxis_title="Match (chronological)")
        st.plotly_chart(fig_elo_t,use_container_width=True)

    if n_pl>0:
        st.markdown('<div class="shr">Rank movement — xLadder vs official ladder</div>',unsafe_allow_html=True)
        actual_pts={}
        for _,r in played.iterrows():
            m=r["Margin"]
            for team,margin in [(r["A_Name"],m),(r["B_Name"],-m)]:
                actual_pts.setdefault(team,0)
                actual_pts[team]+=(2 if margin>0 else 1 if margin==0 else 0)
        al_rank={t:i+1 for i,(t,_) in enumerate(sorted(actual_pts.items(),key=lambda x:-x[1]))}
        xl_rank={row["Team"]:i+1 for i,(_,row) in enumerate(xl.iterrows())}
        rrows=[{"Team":t,"xLadder Rank":xl_rank.get(t,"—"),"Actual Rank":al_rank.get(t,"—"),
                "Move":al_rank.get(t,0)-xl_rank.get(t,0) if isinstance(xl_rank.get(t,"—"),int) else 0}
               for t in teams]
        rdf=pd.DataFrame(rrows).sort_values("xLadder Rank")
        rdf["Move"]=rdf["Move"].apply(lambda x:f"▲{abs(x)}" if x>1 else(f"▼{abs(x)}" if x<-1 else "—"))
        def hl_mv(v):
            if isinstance(v,str):
                if v.startswith("▲"): return "color:#10B981;font-weight:bold"
                if v.startswith("▼"): return "color:#EF4444;font-weight:bold"
            return ""
        st.dataframe(rdf.style.map(hl_mv,subset=["Move"]),use_container_width=True,height=300)

# ════════ TAB 3: TEAM STATS ═══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="shr">Team stats explorer</div>',unsafe_allow_html=True)
    ctrl1,ctrl2,ctrl3=st.columns([1.5,1.5,1])
    sel_team=ctrl1.selectbox("Team",teams,key="st_team")
    sel_cat=ctrl2.selectbox("Category",list(STAT_CATS.keys()),key="st_cat")
    roll_w=ctrl3.number_input("Rolling window",1,10,5,key="st_roll")

    cat_stats=[s for s in STAT_CATS[sel_cat] if f"A_{s}" in df.columns or f"B_{s}" in df.columns]
    t_orig={v:k for k,v in name_map.items()}.get(sel_team,sel_team)

    if cat_stats and n_pl>0:
        stat_data={}
        for stat in cat_stats:
            vals=[]
            for _,r in played.sort_values(["Season","Round"]).iterrows():
                if r["A_Name"]==sel_team:
                    v=pd.to_numeric(r.get(f"A_{stat}",np.nan),errors="coerce")
                    if not pd.isna(v): vals.append({"round":r["Round"],"season":r["Season"],"val":v,"vs":r["B_Name"]})
                elif r["B_Name"]==sel_team:
                    v=pd.to_numeric(r.get(f"B_{stat}",np.nan),errors="coerce")
                    if not pd.isna(v): vals.append({"round":r["Round"],"season":r["Season"],"val":v,"vs":r["A_Name"]})
            if vals: stat_data[stat]=pd.DataFrame(vals)

        # Percentile cards
        n_show=min(8,len(stat_data))
        cols_perc=st.columns(min(4,n_show))
        for i,(stat,sdf) in enumerate(list(stat_data.items())[:n_show]):
            team_avg=sdf["val"].mean()
            all_vals=[]
            for _,r in played.iterrows():
                for col in [f"A_{stat}",f"B_{stat}"]:
                    v=pd.to_numeric(r.get(col,np.nan),errors="coerce")
                    if not pd.isna(v): all_vals.append(v)
            if not all_vals: continue
            pct=pct_rank(pd.Series(all_vals),team_avg)
            is_m=stat in model_flag
            badge="🔷 " if is_m else ""
            klass="kpi-pos" if pct>=60 else "kpi-neg" if pct<=40 else "kpi-neu"
            cols_perc[i%4].markdown(
                f'<div class="kpi {klass}" style="margin-bottom:8px">'
                f'<div class="kpi-label">{badge}{stat[:26]}</div>'
                f'<div class="kpi-val" style="font-size:18px">{team_avg:.1f}</div>'
                f'<div class="kpi-sub">{pct}th pct</div></div>',unsafe_allow_html=True)

        # Trend chart
        st.markdown('<div class="shr">Rolling trend</div>',unsafe_allow_html=True)
        sel_stat=st.selectbox("Stat",list(stat_data.keys()),key="trend_s")
        if sel_stat in stat_data:
            sdf=stat_data[sel_stat].sort_values(["season","round"]).reset_index(drop=True)
            sdf["rolling"]=sdf["val"].rolling(int(roll_w),min_periods=1).mean()
            all_vals2=[]
            for _,r in played.iterrows():
                for col in [f"A_{sel_stat}",f"B_{sel_stat}"]:
                    v=pd.to_numeric(r.get(col,np.nan),errors="coerce")
                    if not pd.isna(v): all_vals2.append(v)
            l_avg=np.mean(all_vals2) if all_vals2 else 0
            col=team_color(sel_team)
            fig_t=go.Figure()
            fig_t.add_trace(go.Scatter(x=list(range(len(sdf))),y=sdf["val"],
                mode="markers",name="Per game",marker=dict(size=7,color=col,opacity=0.5),
                hovertemplate="R%{customdata[0]} vs %{customdata[1]}: %{y:.1f}<extra></extra>",
                customdata=list(zip(sdf["round"],sdf["vs"]))))
            fig_t.add_trace(go.Scatter(x=list(range(len(sdf))),y=sdf["rolling"],
                mode="lines",name=f"{roll_w}-game avg",line=dict(color=col,width=2.5)))
            fig_t.add_hline(y=l_avg,line_dash="dot",line_color="#64748B",
                annotation_text=f"League avg {l_avg:.1f}")
            apply_theme(fig_t,height=340,yaxis_title=sel_stat,
                title_text=f"{'🔷 ' if sel_stat in model_flag else ''}{sel_team} — {sel_stat}")
            st.plotly_chart(fig_t,use_container_width=True)

    # Comparison
    st.markdown('<div class="shr">Team comparison</div>',unsafe_allow_html=True)
    ca,cb,cc=st.columns(3)
    ta=ca.selectbox("Team A",teams,key="cmp_a")
    tb=cb.selectbox("Team B",teams,index=1 if len(teams)>1 else 0,key="cmp_b")
    ccat=cc.selectbox("Category",list(STAT_CATS.keys()),key="cmp_cat")
    if ta and tb and ta!=tb:
        cstats=[s for s in STAT_CATS[ccat] if f"A_{s}" in df.columns][:10]
        avgs_a=[]; avgs_b=[]; labels=[]
        for stat in cstats:
            def gavg(team):
                v=[]
                for _,r in played.iterrows():
                    if r["A_Name"]==team: vv=pd.to_numeric(r.get(f"A_{stat}",np.nan),errors="coerce")
                    elif r["B_Name"]==team: vv=pd.to_numeric(r.get(f"B_{stat}",np.nan),errors="coerce")
                    else: continue
                    if not pd.isna(vv): v.append(vv)
                return np.mean(v) if v else 0
            aa=gavg(ta); ab=gavg(tb)
            if aa>0 or ab>0:
                avgs_a.append(aa); avgs_b.append(ab)
                labels.append(stat[:22]+"…" if len(stat)>22 else stat)
        if labels:
            fig_cmp=go.Figure()
            fig_cmp.add_trace(go.Bar(y=labels,x=avgs_a,orientation="h",name=ta,
                marker_color=team_color(ta),opacity=0.85))
            fig_cmp.add_trace(go.Bar(y=labels,x=avgs_b,orientation="h",name=tb,
                marker_color=team_color(tb),opacity=0.85))
            fig_cmp.update_layout(barmode="group",**PLOTLY_THEME,height=420,
                legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            st.plotly_chart(fig_cmp,use_container_width=True)

# ════════ TAB 4: BETTING ═════════════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="info">Edge threshold: ±{edge_thresh} pts · Kelly: {kelly_frac:.0%} · Bankroll: ${bankroll:,}</div>',unsafe_allow_html=True)
    next_rnd=last_rnd+1
    st.markdown(f'<div class="shr">Round {next_rnd} — fixtures & lines</div>',unsafe_allow_html=True)
    n_fix=st.number_input("Fixtures",1,16,8 if league=="NRL" else 6,key="bet_nf")
    fixtures=[]
    for i in range(int(n_fix)):
        with st.expander(f"Fixture {i+1}",expanded=(i<4)):
            fc1,fc2,fc3=st.columns([2,2,1])
            hd=fc1.selectbox("Home",disp_teams,key=f"bh_{i}",index=min(i*2,len(disp_teams)-1))
            ad=fc2.selectbox("Away",disp_teams,key=f"ba_{i}",index=min(i*2+1,len(disp_teams)-1))
            ven=fc3.selectbox("Venue",["A (home)","B (away)","neutral"],key=f"bv_{i}")
            ha=ven.split()[0]; hc=rev_map.get(hd,hd); ac=rev_map.get(ad,ad)
            oc1,oc2,oc3,oc4,oc5=st.columns(5)
            h2h_h=oc1.number_input("H2H Home",1.01,20.0,1.80,0.05,key=f"h2hh_{i}",format="%.2f")
            h2h_a=oc2.number_input("H2H Away",1.01,20.0,2.05,0.05,key=f"h2ha_{i}",format="%.2f")
            line=oc3.number_input("Line",  -50.0,50.0,-6.5,0.5,key=f"line_{i}",format="%.1f")
            lo_h=oc4.number_input("Line H",1.01,5.0,1.91,0.01,key=f"loh_{i}",format="%.2f")
            lo_a=oc5.number_input("Line A",1.01,5.0,1.91,0.01,key=f"loa_{i}",format="%.2f")
        fixtures.append(dict(hd=hd,ad=ad,hc=hc,ac=ac,ha=ha,h2h_h=h2h_h,h2h_a=h2h_a,line=line,lo_h=lo_h,lo_a=lo_a))

    if st.button("Calculate all edges",type="primary",use_container_width=True):
        results=[]
        for fx in fixtures:
            ph,mg=predict_fx(fx["hc"],fx["ac"],fx["ha"])
            imp_h=(1/fx["h2h_h"])/(1/fx["h2h_h"]+1/fx["h2h_a"])
            le=mg-(-fx["line"]); he=ph-imp_h
            lb="Home covers" if le>0 else "Away covers"
            lo=fx["lo_h"] if le>0 else fx["lo_a"]
            hb="Home" if he>0 else "Away"
            ho=fx["h2h_h"] if he>0 else fx["h2h_a"]
            b=lo-1; p=ph if le>0 else 1-ph; q=1-p
            kelly=max(0,(p*b-q)/b)*kelly_frac
            results.append(dict(Home=fx["hd"],Away=fx["ad"],ProbHome=f"{ph:.0%}",
                ModelMargin=f"{mg:+.1f}",BkLine=f"{fx['line']:+.1f}",
                LineEdge=round(le,2),LineBet=lb,LineOdds=lo,
                H2HEdge=round(he,3),H2HBet=hb,H2HOdds=ho,
                KellyStake=round(bankroll*kelly,1)))

        # Edge chart
        le_vals=[r["LineEdge"] for r in results]
        labels=[f"{r['Home'][:10]} v {r['Away'][:10]}" for r in results]
        fig_e=go.Figure(go.Bar(x=le_vals,y=labels,orientation="h",
            marker_color=["#10B981" if v>=0 else "#EF4444" for v in le_vals],
            text=[f"{v:+.1f}" for v in le_vals],textposition="outside",
            textfont=dict(size=11,color="#F1F5F9"),
            hovertemplate="%{y}: %{x:+.1f} pts<extra></extra>"))
        fig_e.add_vline(x=0,line_dash="dash",line_color="#64748B")
        fig_e.add_vline(x=edge_thresh,line_dash="dot",line_color="#10B981")
        fig_e.add_vline(x=-edge_thresh,line_dash="dot",line_color="#EF4444")
        apply_theme(fig_e,height=max(300,len(results)*45),xaxis_title="Line edge (pts)")
        st.plotly_chart(fig_e,use_container_width=True)

        # Cards + Kelly
        for r in results:
            le=r["LineEdge"]; he=r["H2HEdge"]
            cc1,cc2,cc3,cc4=st.columns([2.5,1.5,1.5,1.5])
            win_t=r["Home"] if float(r["ProbHome"].replace("%",""))/100>=0.5 else r["Away"]
            cc1.markdown(f"**{r['Home']}** vs **{r['Away']}**")
            cc1.caption(f"Model: {win_t} {r['ProbHome']} · {r['ModelMargin']} pts · Bk: {r['BkLine']}")
            lc="#10B981" if le>0 else "#EF4444"
            cc2.markdown(f"**Line edge:** <span style='color:{lc};font-weight:bold'>{le:+.1f}{'  ⭐' if abs(le)>=edge_thresh else ''}</span>",unsafe_allow_html=True)
            cc2.caption(f"{r['LineBet']} @ {r['LineOdds']:.2f}")
            hc2="#10B981" if he>0 else "#EF4444"
            cc3.markdown(f"**H2H edge:** <span style='color:{hc2};font-weight:bold'>{he:+.3f}{'  ⭐' if abs(he)>=0.08 else ''}</span>",unsafe_allow_html=True)
            cc3.caption(f"{r['H2HBet']} @ {r['H2HOdds']:.2f}")
            if r["KellyStake"]>0:
                cc4.markdown(f"**Kelly: ${r['KellyStake']:.0f}**")
                cc4.caption(f"{r['KellyStake']/bankroll:.1%} of bankroll")
            else:
                cc4.caption("No edge — skip")
            st.divider()

        st.download_button("Download predictions CSV",
            pd.DataFrame(results).to_csv(index=False).encode(),
            f"{league}_{season}_R{next_rnd}_bets.csv","text/csv")

    # Per-round accuracy
    if n_pl>0:
        st.markdown('<div class="shr">Round-by-round model accuracy</div>',unsafe_allow_html=True)
        racc=[]
        for rnd in sorted(played["Round"].unique()):
            sub=played[played["Round"]==rnd]
            if len(sub)<2: continue
            racc.append({"Round":int(rnd),"Acc":accuracy_score(sub["A_Win"],sub["WL_Pred"]),"n":len(sub)})
        if racc:
            radf=pd.DataFrame(racc)
            fig_ra=go.Figure(go.Bar(x=radf["Round"].astype(str),y=radf["Acc"],
                marker_color=["#10B981" if v>=0.524 else "#EF4444" for v in radf["Acc"]],
                text=[f"{v:.0%}" for v in radf["Acc"]],textposition="outside",
                hovertemplate="R%{x}: %{y:.0%}<extra></extra>"))
            fig_ra.add_hline(y=0.524,line_dash="dot",line_color="#F59E0B",annotation_text="BEP")
            apply_theme(fig_ra,height=280,yaxis=dict(tickformat=".0%",range=[0,1],gridcolor="#334155"))
            st.plotly_chart(fig_ra,use_container_width=True)

# ════════ TAB 5: WEEKLY INPUT ═════════════════════════════════════════════
with tab5:
    st.markdown("## Add completed round")
    st.markdown(f'<div class="info">ELO auto-computes (K={ELO_K}). Download Updated Master xlsx and replace in repo.</div>',unsafe_allow_html=True)
    inp_rnd=last_rnd+1
    st.markdown(f"### {league} {season} — Round {inp_rnd}")
    n_m=st.number_input("Matches",1,16,8 if league=="NRL" else 6,key="inp_nm")
    match_rows=[]
    for i in range(int(n_m)):
        with st.expander(f"Match {i+1}",expanded=(i<3)):
            mc1,mc2,mc3=st.columns([2,2,1])
            hd=mc1.selectbox("Home",disp_teams,key=f"ih_{i}",index=min(i*2,len(disp_teams)-1))
            ad=mc2.selectbox("Away",disp_teams,key=f"ia_{i}",index=min(i*2+1,len(disp_teams)-1))
            ven=mc3.selectbox("Venue",["A (home)","B (away)","neutral"],key=f"iv_{i}")
            hc_i=rev_map.get(hd,hd); ac_i=rev_map.get(ad,ad); ha_i=ven.split()[0]
            sc1,sc2=st.columns(2)
            sa=sc1.number_input(f"Score {hd[:20]}",0,200,0,key=f"isa_{i}")
            sb=sc2.number_input(f"Score {ad[:20]}",0,200,0,key=f"isb_{i}")
            st.markdown("**Stats — A left / B right:**")
            cg=st.columns(4); sv_a={}; sv_b={}
            for j,stat in enumerate(stats_model):
                short=(stat.replace("PTB - ","").replace("Kick Chase - ","")
                       .replace("Ball Run - ","").replace("Receipt - ","")[:18])
                idx=(j%2)*2
                sv_a[stat]=cg[idx].number_input(f"A:{short}",0.0,9999.0,0.0,1.0,key=f"isxa_{i}_{j}")
                sv_b[stat]=cg[idx+1].number_input(f"B:{short}",0.0,9999.0,0.0,1.0,key=f"isxb_{i}_{j}")
        row={"Season":int(season),"Round":int(inp_rnd),"Match ID":99000+i,
             "A Team":hc_i,"B Team":ac_i,"Home Advantage":ha_i,
             "A_Points Scored":float(sa),"B_Points Scored":float(sb)}
        for stat in stats_model:
            row[f"A_{stat}"]=sv_a[stat]; row[f"B_{stat}"]=sv_b[stat]
        match_rows.append(row)

    cp,cs=st.columns(2)
    with cp:
        if st.button("Preview ELO"):
            ndp=pd.DataFrame(match_rows)
            _,ne=update_elos_for_new_matches(ndp,current_elos,int(season))
            ep=[{"Team":name_map.get(t,t),"Before":round(current_elos.get(t,2000),1),
                 "After":round(ne[t],1),"Change":round(ne[t]-current_elos.get(t,2000),1)}
                for t in sorted(ne)]
            epdf=pd.DataFrame(ep).sort_values("After",ascending=False)
            def hle(v):
                if isinstance(v,float):
                    return f"color:{'#10B981' if v>0 else '#EF4444' if v<0 else '#94A3B8'}"
                return ""
            st.dataframe(epdf.style.map(hle,subset=["Change"]),use_container_width=True,height=360)
    with cs:
        st.markdown("&nbsp;")
        sub_btn=st.button("Submit & Regenerate",type="primary",use_container_width=True)

    if sub_btn:
        with st.spinner("Updating..."):
            ndf=pd.DataFrame(match_rows)
            Rn=run_pipeline(str(master_path),league,target_season=int(season),new_matches_df=ndf)
        st.success(f"Round {inp_rnd} added!")
        st.dataframe(Rn["xladder"][["Team","GP","Expected_PPG","Actual_PPG","PPG_Diff"]].round(3),
                     use_container_width=True,height=340)
        d1,d2,d3=st.columns(3)
        b1=io.BytesIO(); Rn["fig_xladder"].savefig(b1,format="png",dpi=150,bbox_inches="tight",facecolor="#0F172A"); b1.seek(0)
        d1.download_button("xLadder PNG",b1.getvalue(),f"{league}_{season}_R{inp_rnd}_xladder.png","image/png")
        b2=io.BytesIO(); Rn["fig_margin"].savefig(b2,format="png",dpi=150,bbox_inches="tight",facecolor="#0F172A"); b2.seek(0)
        d2.download_button("Margin PNG",b2.getvalue(),f"{league}_{season}_R{inp_rnd}_margin.png","image/png")
        upd=Rn["df"]
        keep=[c for c in upd.columns if not c.startswith("Diff_Form") and not c.startswith("A_Form") and not c.startswith("B_Form")]
        bm=io.BytesIO(); upd[keep].to_excel(bm,index=False); bm.seek(0)
        d3.download_button("Updated Master xlsx",bm.getvalue(),master_path.name,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        plt.close("all")
