"""
xLadder Pro  v3.1
=================
Tabs:
  1. Dashboard   — KPIs, hot/cold, ELO snapshot, accuracy
  2. xLadder     — PPG standings, ELO trend, rank movement, lucky/unlucky
  3. Team Stats  — 50+ stats explorer, percentile, comparison, radar
  4. Betting     — M3/M3+ predictions, margin bands, underdog radar,
                   total pts, H2H+line edge, Kelly, bet logger
  5. Bet History — P&L tracker, CLV, per-config breakdown
  6. Model       — M3 vs M3+ comparison, feature importances
  7. Weekly Input— CSV upload OR manual entry, ELO auto-update
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, os, sys, tempfile, json
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_OK = True
except ImportError:
    GSPREAD_OK = False
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from xladder_pipeline import (
    run_pipeline, get_current_elos, update_elos_for_new_matches,
    build_form_features, assign_margin_band,
    predict_total, build_team_total_tendencies,
    NRL_STATS, SL_STATS, NRL_STATS_V2, SL_STATS_V2,
    BRAND, SL_NAMES, NRL_SHORT, MARGIN_BANDS
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
NRL_MASTER = BASE_DIR / "NRL_master.xlsx"
SL_MASTER  = BASE_DIR / "SL_master.xlsx"
BETS_FILE  = BASE_DIR / "bets_history.json"

# ── Google Sheets helpers ─────────────────────────────────────────────────────
def _gs_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    info = {k: v for k, v in st.secrets["gcp_service_account"].items()}
    # Ensure private key has real newlines not escaped \n
    pk = info.get("private_key", "")
    if pk and chr(92) + "n" in pk:
        info["private_key"] = pk.replace(chr(92) + "n", chr(10))
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

def read_master_from_sheets(league):
    """Read master DataFrame from Google Sheets."""
    # Support both per-league IDs and single SHEET_ID
    if league=="NRL":
        sheet_key = "NRL_SHEET_ID" if "NRL_SHEET_ID" in st.secrets else "SHEET_ID"
    else:
        sheet_key = "SL_SHEET_ID" if "SL_SHEET_ID" in st.secrets else "SHEET_ID"
    ws_name = "NRL_master" if league=="NRL" else "SL_master"
    try:
        gc  = _gs_client()
        sh  = gc.open_by_key(st.secrets[sheet_key])
        ws  = sh.worksheet(ws_name)
        data = ws.get_all_records(numericise_ignore=["all"])
        df  = pd.DataFrame(data)
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            # Only replace if conversion didn't turn everything to NaN
            if converted.notna().sum() >= df[col].notna().sum() * 0.5:
                df[col] = converted
        return df, True
    except Exception as e:
        st.warning(f"Google Sheets unavailable ({e}) — using local file.")
        return None, False

def write_master_to_sheets(df, league):
    """Write updated master DataFrame back to Google Sheets."""
    sheet_key = "NRL_SHEET_ID" if league=="NRL" else "SL_SHEET_ID"
    ws_name   = "NRL_master"   if league=="NRL" else "SL_master"
    try:
        gc = _gs_client()
        sh = gc.open_by_key(st.secrets[sheet_key])
        try:    ws = sh.worksheet(ws_name)
        except: ws = sh.add_worksheet(title=ws_name, rows=5000, cols=500)
        df_clean = df.fillna("").astype(str)
        ws.clear()
        ws.update([df_clean.columns.tolist()] + df_clean.values.tolist())
        return True
    except Exception as e:
        st.warning(f"Could not save to Google Sheets: {e}")
        return False

@st.cache_data(ttl=3600)
def _test_sheets_connection():
    """Returns (ok: bool, error: str)"""
    if not GSPREAD_OK: return False, "gspread not installed"
    if "gcp_service_account" not in st.secrets: return False, "no gcp_service_account in secrets"
    sheet_key = "NRL_SHEET_ID" if "NRL_SHEET_ID" in st.secrets else "SHEET_ID" if "SHEET_ID" in st.secrets else None
    if not sheet_key: return False, "no SHEET_ID in secrets"
    try:
        gc = _gs_client()
        gc.open_by_key(st.secrets[sheet_key])
        return True, ""
    except Exception as e:
        return False, str(e)

def has_sheets_config():
    if not (GSPREAD_OK
            and "gcp_service_account" in st.secrets
            and ("NRL_SHEET_ID" in st.secrets or "SHEET_ID" in st.secrets)):
        return False
    ok, _ = _test_sheets_connection()
    return ok

def get_sheets_status():
    if not GSPREAD_OK: return "gspread not installed"
    if "gcp_service_account" not in st.secrets: return "no credentials in secrets"
    ok, err = _test_sheets_connection()
    return "" if ok else err
ELO_K      = 27

STAT_CATS = {
    "Attack":    ["Ball Runs - Total","Ball Runs - Metres Gained","Ball Run - Run",
                  "Ball Run - Run Metres","Line Break","Kick Line Break","Tackle Break",
                  "Try Scored - Total","Offloads Per Set","Ball Runs - Post Contact Metres",
                  "Pre-Contact Metres","All Possessions - Positive","Good Ball Sets","Yardage Sets"],
    "Defence":   ["Tackle - Total Made","Tackle - Total Missed","Tackle - Total Ineffective",
                  "Tackles - Total Atempted","Made Tackle %","Set Restart Conceded","Set Restart Won"],
    "Kicking":   ["Kick Chase - Good Chase","Kick Chase - Total","Kick - Crossfield",
                  "Kick - Grubber","Kick - Bomb","Receipt - Falcon","Receipt - Total",
                  "Ball Run - Kick Return Metres"],
    "Possession":["Possession %","Territory %","Time In Possession (Seconds)",
                  "Time in Possession Opp Half (Seconds)","Time In Possession Opp 20",
                  "Passes Per Set","Completed Sets %","Ball Run Metres per Set",
                  "PTB - Won","PTB - Strong Tackle","Set Complete - Total","Set Incomplete - Total"],
    "Errors":    ["Errors","Errors per Set","Errors - Own Half","Errors - Opposition Half",
                  "Errors - Handling Errors","Penalty - Total","Penalty - Defence","Penalty - Offence"],
}

PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1E293B",
    font=dict(family="Arial",color="#F1F5F9",size=12),
    xaxis=dict(gridcolor="#334155",linecolor="#334155",tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#334155",linecolor="#334155",tickfont=dict(size=11)),
    margin=dict(l=60,r=20,t=40,b=40),
    hoverlabel=dict(bgcolor="#1E293B",bordercolor="#334155",font=dict(color="#F1F5F9")),
)

def at(fig,**kw):
    merged=dict(PT)
    for k,v in kw.items():
        if k in ("xaxis","yaxis") and k in merged and isinstance(v,dict):
            merged[k]={**merged[k],**v}
        else: merged[k]=v
    fig.update_layout(**merged); return fig

PALETTE=["#F78166","#79C0FF","#3FB950","#E3B341","#D2A8FF","#FF7B72","#56D364",
         "#58A6FF","#FFA657","#FF8585","#63E6BE","#C9D1D9","#A8DADC","#F4A261",
         "#E76F51","#2EC4B6","#FFD166"]

def tc(team):
    tl=sorted(st.session_state.get("all_teams",[team]))
    return PALETTE[(tl.index(team) if team in tl else 0)%len(PALETTE)]

def pct_rank(series,val):
    if series.std()==0: return 50
    return int(round((series<val).mean()*100))

# ── Bet history helpers ────────────────────────────────────────────────────────
def load_bets():
    if BETS_FILE.exists():
        try: return json.loads(BETS_FILE.read_text())
        except: pass
    return []

def save_bets(bets):
    BETS_FILE.write_text(json.dumps(bets,indent=2))

def add_bet(bet_dict):
    bets=load_bets(); bets.append(bet_dict); save_bets(bets)

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="xLadder Pro",page_icon="🏉",layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
body,.stApp{background-color:#0F172A;color:#F1F5F9}
section[data-testid="stSidebar"]{background:#0D1117;border-right:.5px solid #1E293B}
.stTabs [role="tab"]{color:#64748B;font-weight:600;font-size:12px;padding:7px 14px}
.stTabs [aria-selected="true"]{color:#F1F5F9;border-bottom:2px solid #3B82F6}
.stTabs [role="tablist"]{background:#0D1117;border-bottom:.5px solid #1E293B;gap:0}
h1,h2,h3{color:#F1F5F9 !important}
.kpi{background:#1E293B;border-radius:10px;padding:14px 16px;border:.5px solid #334155;margin-bottom:8px}
.kpi-label{color:#64748B;font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px}
.kpi-val{font-size:24px;font-weight:700;color:#F1F5F9;line-height:1}
.kpi-sub{font-size:11px;color:#475569;margin-top:3px}
.kpi-pos{border-left:3px solid #10B981}.kpi-neg{border-left:3px solid #EF4444}.kpi-neu{border-left:3px solid #3B82F6}
.shr{font-size:12px;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:.06em;
  margin:1.2rem 0 .6rem;padding-bottom:5px;border-bottom:.5px solid #1E293B}
.info{background:#1E3A5F;border-radius:6px;padding:8px 12px;border-left:3px solid #3B82F6;
  color:#93C5FD;font-size:12px;margin:6px 0}
.ok{background:#052e16;border-radius:6px;padding:8px 12px;border-left:3px solid #10B981;
  color:#6EE7B7;font-size:12px;margin:6px 0}
.warn{background:#292108;border-radius:6px;padding:8px 12px;border-left:3px solid #F59E0B;
  color:#FCD34D;font-size:12px;margin:6px 0}
.bet-card{background:#1E293B;border-radius:8px;padding:12px 16px;border:.5px solid #334155;
  margin-bottom:8px}
.bet-val{border-left:3px solid #10B981}.bet-skip{border-left:3px solid #EF4444}
div[data-testid="stDownloadButton"] button{background:#1D4ED8;color:white;border-radius:6px;
  padding:5px 14px;font-weight:600;border:none;font-size:12px}
</style>
""",unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏉 xLadder Pro")
    st.markdown("---")
    league=st.selectbox("League",["NRL","SL"])
    season=st.number_input("Season",min_value=2022,max_value=2030,value=2026,step=1)
    model_ver=st.selectbox("Model version",["M3 (original)","M3+ (enhanced)"])
    mver="v1" if "original" in model_ver else "v2"
    master_path=NRL_MASTER if league=="NRL" else SL_MASTER
    st.markdown("---")
    if GSPREAD_OK and "gcp_service_account" in st.secrets:
        ok, err = _test_sheets_connection()
        if ok:
            st.markdown('<div class="ok">Google Sheets connected — auto-save on</div>',unsafe_allow_html=True)
        else:
            short_err = err[:60] if err else "unknown"
            st.markdown(f'<div class="warn">Sheets error: {short_err}</div>',unsafe_allow_html=True)
            if master_path.exists():
                kb=master_path.stat().st_size//1024
                st.markdown(f'<div class="ok">Fallback: local file ({kb}KB)</div>',unsafe_allow_html=True)
    elif master_path.exists():
        kb=master_path.stat().st_size//1024
        st.markdown(f'<div class="ok">Local file · {master_path.name} · {kb}KB</div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn">Missing: {master_path.name}</div>',unsafe_allow_html=True)
    st.markdown("---")
    edge_thresh=st.slider("Edge threshold (pts)",1,10,3)
    kelly_frac =st.slider("Kelly fraction",0.1,1.0,0.25,0.05)
    bankroll   =st.number_input("Bankroll ($)",value=1000,step=100)
    ug_thresh  =st.slider("Underdog threshold",0.25,0.48,0.42,0.01)
    st.markdown("---")
    st.caption("xLadder Pro v3.1  |  M3 / M3+ models")

# ── Load master — Google Sheets first, local file fallback ──────────────────
_use_sheets = has_sheets_config()

@st.cache_data(show_spinner="Loading data...", ttl=180)
def get_result_sheets(league, season):
    df_raw, ok = read_master_from_sheets(league)
    if not ok or df_raw is None: return None
    import tempfile as _tf, os as _os
    with _tf.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        df_raw.to_excel(tmp.name, index=False); tp=tmp.name
    try:    return run_pipeline(tp, league, target_season=int(season))
    finally: _os.unlink(tp)

@st.cache_data(show_spinner="Running model...", ttl=180)
def get_result_local(path_str, league, season):
    return run_pipeline(path_str, league, target_season=int(season))

R = None
if _use_sheets:
    try:
        R = get_result_sheets(league, int(season))
    except Exception as e:
        st.warning(f"Sheets load error: {e} — trying local file.")

if R is None:
    if not master_path.exists():
        st.error(f"No Google Sheets config and no local `{master_path.name}` found."); st.stop()
    try: R = get_result_local(str(master_path), league, int(season))
    except Exception as e: st.error(f"Pipeline error: {e}"); st.exception(e); st.stop()

df=R["df"]; xl=R[f"xladder{'_v2' if mver=='v2' else ''}"]
mt=R[f"margin_table{'_v2' if mver=='v2' else ''}"]
hc_df=R["hot_cold"]; ug_df=R["underdog_flags"]
current_elos=R["current_elos"]; team_totals=R["team_totals"]
sc_m=R[f"sc_{mver}"]; wl_m=R[f"wl_{mver}"]; mg_m=R[f"mg_{mver}"]; F_m=R[f"F_{mver}"]
form_cols=R[f"form_cols_{mver}"]
played=df[(df["Season"]==int(season))&df["Played"]].copy()
all_pl=df[df["Played"]].copy()
n_pl=len(played); last_rnd=int(played["Round"].max()) if n_pl>0 else 0
teams=sorted(set(played["A_Name"])|set(played["B_Name"]))
st.session_state["all_teams"]=teams
name_map=NRL_SHORT if league=="NRL" else SL_NAMES
raw_teams=sorted(set(df["A Team"])|set(df["B Team"]))
rev_map={name_map.get(t,t):t for t in raw_teams}
disp_teams=[name_map.get(t,t) for t in raw_teams]
stats_model=NRL_STATS if league=="NRL" else SL_STATS
stats_v2=NRL_STATS_V2 if league=="NRL" else SL_STATS_V2
model_flag=set(stats_model)|set(stats_v2)

# ELO + form lookups for betting predictions
team_elos_m={}
for _,r in df.sort_values(["Season","Round"]).iterrows():
    team_elos_m[r["A Team"]]=float(r["ELO_A"]); team_elos_m[r["B Team"]]=float(r["ELO_B"])
team_form_m={}
for fc in form_cols:
    stat=fc.replace("Diff_Form_",""); a_fc=f"A_Form_{stat}"; b_fc=f"B_Form_{stat}"
    for _,r in df.sort_values(["Season","Round"]).iterrows():
        if a_fc in df.columns and pd.notna(r.get(a_fc)): team_form_m[(r["A Team"],fc)]=float(r[a_fc])
        if b_fc in df.columns and pd.notna(r.get(b_fc)): team_form_m[(r["B Team"],fc)]=float(r[b_fc])

def predict_fx(hc,ac,ha):
    elo_h=team_elos_m.get(hc,2000); elo_a=team_elos_m.get(ac,2000)
    ha_map={"A":1,"B":-1,"neutral":0}
    feat={"Diff ELO":elo_h-elo_a,"Home_flag":ha_map.get(ha,0)}
    for fc in form_cols: feat[fc]=team_form_m.get((hc,fc),0)-team_form_m.get((ac,fc),0)
    X=np.array([[feat.get(f,0) for f in F_m]])
    ph=float(wl_m.predict_proba(sc_m.transform(X))[0,1])
    mg=float(mg_m.predict(sc_m.transform(X))[0])
    return ph,mg

# ── TABS ──────────────────────────────────────────────────────────────────────
st.markdown(f"# 🏉 xLadder Pro — {league} {season}  <span style='font-size:14px;color:#64748B;font-weight:400'>({model_ver})</span>",unsafe_allow_html=True)
tab_names=["📊 Dashboard","🏆 xLadder","📈 Team Stats","🎯 Betting","📒 Bet History","🔬 Model","📥 Weekly Input"]
tabs=st.tabs(tab_names)

# ════════════════════ TAB 1: DASHBOARD ═══════════════════════════════════════
with tabs[0]:
    if n_pl>0:
        prob_col=f"WL_Prob_A_{mver}"
        acc=accuracy_score(played["A_Win"],played[f"WL_Pred_{mver}"])
        try: auc=roc_auc_score(played["A_Win"],played[prob_col])
        except: auc=float("nan")
        top_over=hc_df.groupby("team")["roll_over"].last().idxmax() if len(hc_df) else "—"
        top_under=hc_df.groupby("team")["roll_over"].last().idxmin() if len(hc_df) else "—"
    else:
        acc=auc=0; top_over=top_under="—"

    k1,k2,k3,k4=st.columns(4)
    for col,lab,val,sub,kls in [
        (k1,"Season",f"{league} {season}",f"Round {last_rnd} completed","kpi-neu"),
        (k2,"WL Accuracy",f"{acc:.1%}",f"AUC {auc:.3f}","kpi-pos" if acc>=0.60 else "kpi-neg"),
        (k3,"Hottest",top_over,"Rolling 3-game overperf","kpi-pos"),
        (k4,"Coldest",top_under,"Rolling 3-game underperf","kpi-neg"),
    ]:
        col.markdown(f'<div class="kpi {kls}"><div class="kpi-label">{lab}</div>'
                     f'<div class="kpi-val">{val}</div><div class="kpi-sub">{sub}</div></div>',
                     unsafe_allow_html=True)
    st.markdown("---")
    c1,c2=st.columns(2)

    with c1:
        st.markdown('<div class="shr">Hot / Cold — rolling 3-game vs model</div>',unsafe_allow_html=True)
        if len(hc_df):
            lhc=hc_df.groupby("team")["roll_over"].last().sort_values(ascending=True)
            fig=go.Figure(go.Bar(x=lhc.values,y=lhc.index,orientation="h",
                marker_color=["#10B981" if v>=0 else "#EF4444" for v in lhc.values],
                text=[f"{v:+.1f}" for v in lhc.values],textposition="outside",
                textfont=dict(size=10,color="#F1F5F9"),
                hovertemplate="%{y}: %{x:+.1f}<extra></extra>"))
            fig.add_vline(x=0,line_dash="dash",line_color="#64748B")
            at(fig,height=420,xaxis_title="Overperformance vs model (pts)")
            st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown('<div class="shr">Model accuracy by season</div>',unsafe_allow_html=True)
        acc_rows=[]
        for s in sorted(all_pl["Season"].unique()):
            sub=all_pl[all_pl["Season"]==s]
            if len(sub)<5: continue
            pc=f"WL_Prob_A_{mver}"; wc=f"WL_Pred_{mver}"
            if pc not in sub.columns or wc not in sub.columns: continue
            a=accuracy_score(sub["A_Win"],sub[wc])
            try: au=roc_auc_score(sub["A_Win"],sub[pc])
            except: au=float("nan")
            acc_rows.append({"Season":int(s),"Accuracy":a,"n":len(sub)})
        if acc_rows:
            adf=pd.DataFrame(acc_rows)
            fig2=go.Figure(go.Bar(x=adf["Season"].astype(str),y=adf["Accuracy"],
                marker_color=["#10B981" if v>=0.6 else "#3B82F6" if v>=0.524 else "#EF4444" for v in adf["Accuracy"]],
                text=[f"{v:.1%}" for v in adf["Accuracy"]],textposition="outside",
                hovertemplate="%{x}: %{y:.1%} (n=%{customdata})<extra></extra>",customdata=adf["n"]))
            fig2.add_hline(y=0.524,line_dash="dot",line_color="#F59E0B",annotation_text="BEP 52.4%")
            fig2.add_hline(y=0.62,line_dash="dot",line_color="#10B981",annotation_text="Target 62%")
            at(fig2,height=420,yaxis=dict(tickformat=".0%",range=[0.45,0.75],gridcolor="#334155"))
            st.plotly_chart(fig2,use_container_width=True)

    # ELO snapshot
    st.markdown('<div class="shr">Current ELO ratings</div>',unsafe_allow_html=True)
    es=sorted(current_elos.items(),key=lambda x:-x[1])
    en=[name_map.get(t,t) for t,_ in es]; ev=[v for _,v in es]
    fig3=go.Figure(go.Bar(x=en,y=ev,
        marker_color=["#10B981" if v>=2000 else "#EF4444" for v in ev],
        text=[f"{v:.0f}" for v in ev],textposition="outside",
        hovertemplate="%{x}: ELO %{y:.1f}<extra></extra>"))
    fig3.add_hline(y=2000,line_dash="dash",line_color="#64748B",annotation_text="Mean")
    at(fig3,height=280,yaxis=dict(range=[1700,2300],gridcolor="#334155"))
    st.plotly_chart(fig3,use_container_width=True)

# ════════════════════ TAB 2: xLADDER ═════════════════════════════════════════
with tabs[1]:
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
        st.download_button("CSV",disp.to_csv(index=True).encode(),f"{league}_{season}_xladder.csv","text/csv")
    with c2:
        st.markdown('<div class="shr">Expected vs Actual PPG</div>',unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Bar(y=xl["Team"],x=xl["Expected_PPG"],orientation="h",name="xPPG (model)",
            marker_color="#3B82F6",opacity=0.85))
        fig.add_trace(go.Bar(y=xl["Team"],x=xl["Actual_PPG"],orientation="h",name="Actual PPG",
            marker_color=["#10B981" if d>=0 else "#EF4444" for d in xl["PPG_Diff"]],opacity=0.85))
        for _,row in xl.iterrows():
            sign="+" if row["PPG_Diff"]>=0 else ""
            fig.add_annotation(x=max(row["Expected_PPG"],row["Actual_PPG"])+0.04,y=row["Team"],
                text=f"{sign}{row['PPG_Diff']:.2f}",showarrow=False,
                font=dict(size=10,color="#10B981" if row["PPG_Diff"]>=0 else "#EF4444"),xanchor="left")
        fig.update_layout(barmode="overlay",**PT,height=500,
            xaxis_title="Points Per Game",
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig,use_container_width=True)

    # Lucky/Unlucky chart
    st.markdown('<div class="shr">Who is getting lucky? — Expected Wins vs Actual Wins</div>',unsafe_allow_html=True)
    if n_pl>0:
        prob_col=f"WL_Prob_A_{mver}"
        exp_wins={}; act_wins={}; games={}
        for _,r in played.iterrows():
            for team,p,win in [(r["A_Name"],r[prob_col],r["A_Win"]),(r["B_Name"],1-r[prob_col],1-r["A_Win"])]:
                exp_wins.setdefault(team,0); act_wins.setdefault(team,0); games.setdefault(team,0)
                exp_wins[team]+=p; act_wins[team]+=win; games[team]+=1
        lk_rows=[{"Team":t,"xWins":round(exp_wins[t],1),"ActWins":act_wins[t],
                  "Diff":round(act_wins[t]-exp_wins[t],1)} for t in sorted(exp_wins)]
        lk_df=pd.DataFrame(lk_rows).sort_values("Diff",ascending=True)
        fig_lk=go.Figure()
        fig_lk.add_trace(go.Bar(y=lk_df["Team"],x=lk_df["xWins"],orientation="h",
            name="Expected Wins",marker_color="#3B82F6",opacity=0.8))
        fig_lk.add_trace(go.Bar(y=lk_df["Team"],x=lk_df["ActWins"],orientation="h",
            name="Actual Wins",marker_color=["#10B981" if d>=0 else "#EF4444" for d in lk_df["Diff"]],opacity=0.8))
        for _,row in lk_df.iterrows():
            sign="+" if row["Diff"]>=0 else ""
            fig_lk.add_annotation(x=max(row["xWins"],row["ActWins"])+0.1,y=row["Team"],
                text=f"{sign}{row['Diff']:.1f}",showarrow=False,
                font=dict(size=10,color="#10B981" if row["Diff"]>=0 else "#EF4444"),xanchor="left")
        fig_lk.update_layout(barmode="overlay",**PT,height=500,xaxis_title="Wins",
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig_lk,use_container_width=True)

    # ELO trend
    st.markdown('<div class="shr">ELO trend — all seasons</div>',unsafe_allow_html=True)
    sel_elo=st.multiselect("Teams",teams,default=teams[:6] if len(teams)>=6 else teams,key="elo_sel")
    if sel_elo and len(all_pl):
        fig_e=go.Figure()
        for team in sel_elo:
            t_orig={v:k for k,v in name_map.items()}.get(team,team)
            sa=all_pl[all_pl["A Team"]==t_orig][["Season","Round","ELO_A"]].copy(); sa.columns=["Season","Round","ELO"]
            sb=all_pl[all_pl["B Team"]==t_orig][["Season","Round","ELO_B"]].copy(); sb.columns=["Season","Round","ELO"]
            sub=pd.concat([sa,sb]).sort_values(["Season","Round"])
            sub["Label"]=sub["Season"].astype(str)+" R"+sub["Round"].astype(str)
            col=tc(team)
            fig_e.add_trace(go.Scatter(x=list(range(len(sub))),y=sub["ELO"],
                mode="lines+markers",name=team,line=dict(color=col,width=2),marker=dict(size=4,color=col),
                hovertemplate=f"{team} — %{{customdata}}: ELO %{{y:.1f}}<extra></extra>",customdata=sub["Label"]))
        fig_e.add_hline(y=2000,line_dash="dot",line_color="#334155")
        at(fig_e,height=380,yaxis_title="ELO",xaxis_title="Match #")
        st.plotly_chart(fig_e,use_container_width=True)

    # Rank movement
    if n_pl>0:
        st.markdown('<div class="shr">Rank movement — xLadder vs official ladder</div>',unsafe_allow_html=True)
        act_pts={}
        for _,r in played.iterrows():
            m=r["Margin"]
            for team,margin in [(r["A_Name"],m),(r["B_Name"],-m)]:
                act_pts.setdefault(team,0)
                act_pts[team]+=(2 if margin>0 else 1 if margin==0 else 0)
        al_rank={t:i+1 for i,(t,_) in enumerate(sorted(act_pts.items(),key=lambda x:-x[1]))}
        xl_rank={row["Team"]:i+1 for i,(_,row) in enumerate(xl.iterrows())}
        rrows=[{"Team":t,"xLadder Rank":xl_rank.get(t,"—"),"Official Rank":al_rank.get(t,"—"),
                "Move":al_rank.get(t,0)-xl_rank.get(t,0) if isinstance(xl_rank.get(t,"—"),int) else 0}
               for t in teams]
        rdf=pd.DataFrame(rrows).sort_values("xLadder Rank")
        rdf["Δ"]=rdf["Move"].apply(lambda x:f"▲{abs(x)}" if x>1 else(f"▼{abs(x)}" if x<-1 else "—"))
        def hl_mv(v):
            if isinstance(v,str):
                if v.startswith("▲"): return "color:#10B981;font-weight:bold"
                if v.startswith("▼"): return "color:#EF4444;font-weight:bold"
            return ""
        st.dataframe(rdf[["Team","xLadder Rank","Official Rank","Δ"]].style.map(hl_mv,subset=["Δ"]),
                     use_container_width=True,height=300)

    # PNG export
    if st.button("Export xLadder PNG"):
        buf=io.BytesIO(); R["fig_xladder"].savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor="#0F172A"); buf.seek(0)
        st.download_button("Download",buf.getvalue(),f"{league}_{season}_xladder.png","image/png",key="dl_xl")

# ════════════════════ TAB 3: TEAM STATS ══════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="shr">Team stats explorer</div>',unsafe_allow_html=True)
    ctrl1,ctrl2,ctrl3=st.columns([1.5,1.5,1])
    sel_team=ctrl1.selectbox("Team",teams,key="st_team")
    sel_cat=ctrl2.selectbox("Category",list(STAT_CATS.keys()),key="st_cat")
    roll_w=ctrl3.number_input("Rolling window",1,10,5,key="st_roll")

    cat_stats=[s for s in STAT_CATS[sel_cat] if f"A_{s}" in df.columns or f"B_{s}" in df.columns]

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
        n_show=min(8,len(stat_data)); cols_p=st.columns(min(4,n_show))
        for i,(stat,sdf) in enumerate(list(stat_data.items())[:n_show]):
            team_avg=sdf["val"].mean()
            all_vals=[pd.to_numeric(r.get(c,np.nan),errors="coerce")
                      for _,r in played.iterrows() for c in [f"A_{stat}",f"B_{stat}"]
                      if not pd.isna(pd.to_numeric(r.get(c,np.nan),errors="coerce"))]
            if not all_vals: continue
            pct=pct_rank(pd.Series(all_vals),team_avg)
            is_m=stat in model_flag; badge="🔷 " if is_m else ""
            kls="kpi-pos" if pct>=60 else "kpi-neg" if pct<=40 else "kpi-neu"
            cols_p[i%4].markdown(
                f'<div class="kpi {kls}" style="margin-bottom:6px">'
                f'<div class="kpi-label">{badge}{stat[:26]}</div>'
                f'<div class="kpi-val" style="font-size:18px">{team_avg:.1f}</div>'
                f'<div class="kpi-sub">{pct}th pct</div></div>',unsafe_allow_html=True)

        # Trend chart
        st.markdown('<div class="shr">Rolling trend</div>',unsafe_allow_html=True)
        sel_stat=st.selectbox("Stat",list(stat_data.keys()),key="trend_s")
        if sel_stat in stat_data:
            sdf=stat_data[sel_stat].sort_values(["season","round"]).reset_index(drop=True)
            sdf["rolling"]=sdf["val"].rolling(int(roll_w),min_periods=1).mean()
            all_v2=[pd.to_numeric(r.get(c,np.nan),errors="coerce")
                    for _,r in played.iterrows() for c in [f"A_{sel_stat}",f"B_{sel_stat}"]
                    if not pd.isna(pd.to_numeric(r.get(c,np.nan),errors="coerce"))]
            l_avg=np.mean(all_v2) if all_v2 else 0
            col_t=tc(sel_team)
            fig_t=go.Figure()
            fig_t.add_trace(go.Scatter(x=list(range(len(sdf))),y=sdf["val"],mode="markers",
                name="Per game",marker=dict(size=7,color=col_t,opacity=0.5),
                hovertemplate="R%{customdata[0]} vs %{customdata[1]}: %{y:.1f}<extra></extra>",
                customdata=list(zip(sdf["round"],sdf["vs"]))))
            fig_t.add_trace(go.Scatter(x=list(range(len(sdf))),y=sdf["rolling"],mode="lines",
                name=f"{roll_w}-game avg",line=dict(color=col_t,width=2.5)))
            fig_t.add_hline(y=l_avg,line_dash="dot",line_color="#64748B",
                annotation_text=f"League avg {l_avg:.1f}")
            at(fig_t,height=320,yaxis_title=sel_stat,
                title_text=f"{'🔷 ' if sel_stat in model_flag else ''}{sel_team} — {sel_stat}")
            st.plotly_chart(fig_t,use_container_width=True)

    # Radar chart
    st.markdown('<div class="shr">Team profile — radar</div>',unsafe_allow_html=True)
    radar_team=st.selectbox("Team for radar",teams,key="radar_t")
    radar_stats=["Ball Runs - Metres Gained","Line Break","Tackle Break",
                 "Kick Chase - Good Chase","Made Tackle %","Errors","PTB - Strong Tackle",
                 "Set Complete - Total","All Possessions - Positive","Penalty - Total"]
    radar_stats=[s for s in radar_stats if f"A_{s}" in df.columns]
    if radar_stats and n_pl>0:
        pcts=[]
        for stat in radar_stats:
            vals_t=[]; vals_all=[]
            for _,r in played.iterrows():
                for team_,col_ in [(r["A_Name"],f"A_{stat}"),(r["B_Name"],f"B_{stat}")]:
                    v=pd.to_numeric(r.get(col_,np.nan),errors="coerce")
                    if not pd.isna(v):
                        vals_all.append(v)
                        if team_==radar_team: vals_t.append(v)
            if vals_t and vals_all:
                p=pct_rank(pd.Series(vals_all),np.mean(vals_t))
                # Invert for "lower is better" stats
                if stat in ["Errors","Penalty - Total"]: p=100-p
                pcts.append(p)
            else: pcts.append(50)
        labels=[s[:20] for s in radar_stats]
        fig_r=go.Figure(go.Scatterpolar(r=pcts+[pcts[0]],theta=labels+[labels[0]],
            fill="toself",name=radar_team,
            line=dict(color=tc(radar_team),width=2),fillcolor=tc(radar_team),opacity=0.25))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100],
            gridcolor="#334155",linecolor="#334155",tickfont=dict(color="#64748B",size=9))),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F1F5F9"),height=400,
            title=dict(text=f"{radar_team} — stat profile (percentile rank)",font=dict(size=13)))
        st.plotly_chart(fig_r,use_container_width=True)

    # Comparison
    st.markdown('<div class="shr">Team comparison</div>',unsafe_allow_html=True)
    ca,cb,cc=st.columns(3)
    ta=ca.selectbox("Team A",teams,key="cmp_a")
    tb=cb.selectbox("Team B",teams,index=1 if len(teams)>1 else 0,key="cmp_b")
    ccat=cc.selectbox("Category",list(STAT_CATS.keys()),key="cmp_cat")
    if ta and tb and ta!=tb and n_pl>0:
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
            if aa>0 or ab>0: avgs_a.append(aa); avgs_b.append(ab); labels.append(stat[:22])
        if labels:
            fig_cmp=go.Figure()
            fig_cmp.add_trace(go.Bar(y=labels,x=avgs_a,orientation="h",name=ta,
                marker_color=tc(ta),opacity=0.85))
            fig_cmp.add_trace(go.Bar(y=labels,x=avgs_b,orientation="h",name=tb,
                marker_color=tc(tb),opacity=0.85))
            fig_cmp.update_layout(barmode="group",**PT,height=400,
                legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            st.plotly_chart(fig_cmp,use_container_width=True)

# ════════════════════ TAB 4: BETTING ═════════════════════════════════════════
with tabs[3]:
    st.markdown(f'<div class="info">Model: {model_ver} · Edge ≥ {edge_thresh} pts · Kelly {kelly_frac:.0%} · Bankroll ${bankroll:,}</div>',unsafe_allow_html=True)
    next_rnd=last_rnd+1
    st.markdown(f'<div class="shr">Round {next_rnd} — enter fixtures & lines</div>',unsafe_allow_html=True)

    n_fix=st.number_input("Fixtures",1,16,8 if league=="NRL" else 6,key="bet_nf")
    fixtures=[]
    for i in range(int(n_fix)):
        with st.expander(f"Fixture {i+1}",expanded=(i<4)):
            fc1,fc2,fc3=st.columns([2,2,1])
            hd=fc1.selectbox("Home",disp_teams,key=f"bh_{i}",index=min(i*2,len(disp_teams)-1))
            ad=fc2.selectbox("Away",disp_teams,key=f"ba_{i}",index=min(i*2+1,len(disp_teams)-1))
            ven=fc3.selectbox("Venue",["A (home)","B (away)","neutral"],key=f"bv_{i}")
            ha=ven.split()[0]; hc_=rev_map.get(hd,hd); ac_=rev_map.get(ad,ad)
            oc1,oc2,oc3,oc4,oc5=st.columns(5)
            h2h_h=oc1.number_input("H2H Home",1.01,20.0,1.80,0.05,key=f"h2hh_{i}",format="%.2f")
            h2h_a=oc2.number_input("H2H Away",1.01,20.0,2.05,0.05,key=f"h2ha_{i}",format="%.2f")
            line=oc3.number_input("Line",-50.0,50.0,-6.5,0.5,key=f"line_{i}",format="%.1f")
            lo_h=oc4.number_input("Line H",1.01,5.0,1.91,0.01,key=f"loh_{i}",format="%.2f")
            lo_a=oc5.number_input("Line A",1.01,5.0,1.91,0.01,key=f"loa_{i}",format="%.2f")
            # Total over/under
            oc6,oc7=st.columns(2)
            total_line=oc6.number_input("Total O/U",20.0,100.0,45.5,0.5,key=f"tot_{i}",format="%.1f")
            tot_odds=oc7.number_input("Total odds",1.01,5.0,1.91,0.01,key=f"todo_{i}",format="%.2f")
        fixtures.append(dict(hd=hd,ad=ad,hc=hc_,ac=ac_,ha=ha,
                             h2h_h=h2h_h,h2h_a=h2h_a,line=line,lo_h=lo_h,lo_a=lo_a,
                             total_line=total_line,tot_odds=tot_odds))

    if st.button("Generate predictions & edges",type="primary",use_container_width=True):
        results=[]
        for fx in fixtures:
            ph,mg=predict_fx(fx["hc"],fx["ac"],fx["ha"])
            elo_h=team_elos_m.get(fx["hc"],2000); elo_a=team_elos_m.get(fx["ac"],2000)
            imp_h=(1/fx["h2h_h"])/(1/fx["h2h_h"]+1/fx["h2h_a"])
            le=mg-(-fx["line"]); he=ph-imp_h
            lb="Home covers" if le>0 else "Away covers"
            lo=fx["lo_h"] if le>0 else fx["lo_a"]
            hb="Home" if he>0 else "Away"; ho=fx["h2h_h"] if he>0 else fx["h2h_a"]
            b_l=lo-1; p_l=ph if le>0 else 1-ph; q_l=1-p_l
            kelly_l=max(0,(p_l*b_l-q_l)/b_l)*kelly_frac
            b_h=ho-1; p_h=ph if he>0 else 1-ph; q_h=1-p_h
            kelly_h=max(0,(p_h*b_h-q_h)/b_h)*kelly_frac
            pred_band=assign_margin_band(mg)
            pred_total=predict_total(fx["hd"],fx["ad"],team_totals)
            total_edge=pred_total-fx["total_line"]
            is_ug=(ph<=float(ug_thresh)) or ((1-ph)<=float(ug_thresh))
            ug_team=fx["hd"] if ph<=float(ug_thresh) else (fx["ad"] if (1-ph)<=float(ug_thresh) else None)
            ug_prob=min(ph,1-ph) if is_ug else None
            results.append(dict(
                Home=fx["hd"],Away=fx["ad"],
                ProbHome=ph,ProbAway=1-ph,
                ModelMargin=round(mg,1),PredBand=pred_band,
                BkLine=fx["line"],LineEdge=round(le,2),LineBet=lb,LineOdds=lo,
                H2HEdge=round(he,3),H2HBet=hb,H2HOdds=ho,
                PredTotal=round(pred_total,1),TotalLine=fx["total_line"],
                TotalEdge=round(total_edge,1),TotOdds=fx["tot_odds"],
                KellyLine=round(bankroll*kelly_l,1),
                KellyH2H=round(bankroll*kelly_h,1),
                IsUnderdog=is_ug,UnderdogTeam=ug_team,UnderdogProb=ug_prob,
                EloHome=round(elo_h,0),EloAway=round(elo_a,0),
            ))

        # Summary table
        st.markdown("---")
        sum_df=pd.DataFrame([{
            "Home":r["Home"],"Away":r["Away"],
            "P(Home)":f"{r['ProbHome']:.0%}","Model Margin":f"{r['ModelMargin']:+.1f}",
            "Pred Band":r["PredBand"],"Bk Line":f"{r['BkLine']:+.1f}",
            "Line Edge":f"{r['LineEdge']:+.1f}","H2H Edge":f"{r['H2HEdge']:+.3f}",
            "Total Pred":r["PredTotal"],"Total Line":r["TotalLine"],
            "Total Edge":f"{r['TotalEdge']:+.1f}",
        } for r in results])
        def hl_e(v):
            try:
                n=float(str(v).replace("+",""))
                if n>=edge_thresh or n>=0.08: return "color:#10B981;font-weight:bold"
                if n<=-edge_thresh or n<=-0.08: return "color:#EF4444;font-weight:bold"
            except: pass
            return "color:#94A3B8"
        st.dataframe(sum_df.style.map(hl_e,subset=["Line Edge","H2H Edge","Total Edge"]),
                     use_container_width=True,height=300)

        # Edge chart
        le_vals=[r["LineEdge"] for r in results]
        labels=[f"{r['Home'][:10]} v {r['Away'][:10]}" for r in results]
        fig_e=go.Figure(go.Bar(x=le_vals,y=labels,orientation="h",
            marker_color=["#10B981" if v>=0 else "#EF4444" for v in le_vals],
            text=[f"{v:+.1f}" for v in le_vals],textposition="outside",
            textfont=dict(size=11,color="#F1F5F9"),
            hovertemplate="%{y}: line edge %{x:+.1f} pts<extra></extra>"))
        fig_e.add_vline(x=0,line_dash="dash",line_color="#64748B")
        fig_e.add_vline(x=edge_thresh,line_dash="dot",line_color="#10B981")
        fig_e.add_vline(x=-edge_thresh,line_dash="dot",line_color="#EF4444")
        at(fig_e,height=max(280,len(results)*45),xaxis_title="Line edge (pts)")
        st.plotly_chart(fig_e,use_container_width=True)

        # Detailed cards + bet logger
        st.markdown('<div class="shr">Detailed view + log bets</div>',unsafe_allow_html=True)
        for r in results:
            le=r["LineEdge"]; he=r["H2HEdge"]; te=r["TotalEdge"]
            ph=r["ProbHome"]; mg=r["ModelMargin"]
            win_t=r["Home"] if ph>=0.5 else r["Away"]
            card_cls="bet-val" if abs(le)>=edge_thresh else "bet-skip"
            with st.container():
                cc1,cc2,cc3,cc4=st.columns([2.5,1.5,1.5,1.5])
                cc1.markdown(f"**{r['Home']}** vs **{r['Away']}**")
                cc1.caption(f"Model: {win_t} {ph:.0%} · {mg:+.1f} pts · Band: {r['PredBand']} · ELO {r['EloHome']:.0f} v {r['EloAway']:.0f}")
                if r["IsUnderdog"] and r["UnderdogTeam"]:
                    cc1.markdown(f"🐉 **Underdog value:** {r['UnderdogTeam']} @ {r['UnderdogProb']:.0%}",unsafe_allow_html=True)
                lc="#10B981" if le>0 else "#EF4444"
                cc2.markdown(f"**Line:** <span style='color:{lc};font-weight:bold'>{le:+.1f}{'⭐' if abs(le)>=edge_thresh else ''}</span>",unsafe_allow_html=True)
                cc2.caption(f"{r['LineBet']} @ {r['LineOdds']:.2f} · Kelly ${r['KellyLine']:.0f}")
                hc2="#10B981" if he>0 else "#EF4444"
                cc3.markdown(f"**H2H:** <span style='color:{hc2};font-weight:bold'>{he:+.3f}{'⭐' if abs(he)>=0.08 else ''}</span>",unsafe_allow_html=True)
                cc3.caption(f"{r['H2HBet']} @ {r['H2HOdds']:.2f} · Kelly ${r['KellyH2H']:.0f}")
                tc2="#10B981" if te>3 else "#EF4444" if te<-3 else "#64748B"
                cc4.markdown(f"**Total:** <span style='color:{tc2};font-weight:bold'>{r['PredTotal']:.0f} vs {r['TotalLine']:.0f} ({te:+.1f})</span>",unsafe_allow_html=True)
                cc4.caption(f"{'Over' if te>0 else 'Under'} @ {r['TotOdds']:.2f}")

                # Log bet button
                with st.expander("📝 Log a bet for this match"):
                    bl1,bl2,bl3,bl4=st.columns(4)
                    bet_type=bl1.selectbox("Bet type",["Line","H2H","Total"],key=f"btype_{r['Home']}")
                    bet_side=bl2.text_input("Bet side",value=r["LineBet"] if bet_type=="Line" else r["H2HBet"],key=f"bside_{r['Home']}")
                    bet_odds=bl3.number_input("Odds",1.01,20.0,r["LineOdds"] if bet_type=="Line" else r["H2HOdds"],key=f"bodds_{r['Home']}",format="%.2f")
                    bet_stake=bl4.number_input("Stake ($)",0.0,10000.0,float(r["KellyLine"] if bet_type=="Line" else r["KellyH2H"]),key=f"bstake_{r['Home']}")
                    if st.button("Log bet",key=f"blog_{r['Home']}"):
                        add_bet({
                            "season":int(season),"round":int(next_rnd),
                            "home":r["Home"],"away":r["Away"],
                            "bet_type":bet_type,"bet_side":bet_side,
                            "line":r["BkLine"],"model_edge":le,"model_margin":mg,
                            "odds":bet_odds,"stake":bet_stake,
                            "result":None,"pnl":None,
                            "logged_at":str(pd.Timestamp.now())[:19]
                        })
                        st.success("Bet logged!")
                st.divider()

        st.download_button("Download predictions CSV",
            pd.DataFrame(results).to_csv(index=False).encode(),
            f"{league}_{season}_R{next_rnd}_predictions.csv","text/csv")

    # Underdog history
    if len(ug_df)>0:
        st.markdown('<div class="shr">Underdog tracker — season history</div>',unsafe_allow_html=True)
        ug_acc=ug_df["Actual_Win"].mean() if len(ug_df)>0 else 0
        st.markdown(f'<div class="info">Underdog win rate this season: {ug_acc:.1%} ({len(ug_df)} matches)</div>',unsafe_allow_html=True)
        st.dataframe(ug_df[["Round","Underdog","Favourite","Model_Prob","Model_Margin","Actual_Win"]].assign(
            Model_Prob=ug_df["Model_Prob"].apply(lambda x:f"{x:.0%}"),
            Model_Margin=ug_df["Model_Margin"].apply(lambda x:f"{x:+.1f}"),
            Result=ug_df["Actual_Win"].map({1:"✓ WIN",0:"✗ LOSS"})
        ).rename(columns={"Actual_Win":"W/L"}),use_container_width=True,height=280)

    # Round accuracy
    if n_pl>0:
        st.markdown('<div class="shr">Round accuracy</div>',unsafe_allow_html=True)
        racc=[]
        for rnd in sorted(played["Round"].unique()):
            sub=played[played["Round"]==rnd]; wc=f"WL_Pred_{mver}"
            if len(sub)<2 or wc not in sub.columns: continue
            racc.append({"Round":int(rnd),"Acc":accuracy_score(sub["A_Win"],sub[wc]),"n":len(sub)})
        if racc:
            radf=pd.DataFrame(racc)
            fig_ra=go.Figure(go.Bar(x=radf["Round"].astype(str),y=radf["Acc"],
                marker_color=["#10B981" if v>=0.524 else "#EF4444" for v in radf["Acc"]],
                text=[f"{v:.0%}" for v in radf["Acc"]],textposition="outside",
                hovertemplate="R%{x}: %{y:.0%}<extra></extra>"))
            fig_ra.add_hline(y=0.524,line_dash="dot",line_color="#F59E0B",annotation_text="BEP")
            at(fig_ra,height=260,yaxis=dict(tickformat=".0%",range=[0,1],gridcolor="#334155"))
            st.plotly_chart(fig_ra,use_container_width=True)

# ════════════════════ TAB 5: BET HISTORY ════════════════════════════════════
with tabs[4]:
    st.markdown("## Bet History")
    bets=load_bets()
    if not bets:
        st.info("No bets logged yet. Use the Betting tab to log your bets after each round.")
    else:
        bdf=pd.DataFrame(bets)
        # Filter to current season/league
        if "season" in bdf.columns:
            bdf_s=bdf[bdf["season"]==int(season)].copy()
        else:
            bdf_s=bdf.copy()

        # Settle bets — show unresolved
        unresolved=bdf_s[bdf_s["result"].isna()] if "result" in bdf_s.columns else bdf_s
        if len(unresolved)>0:
            st.markdown('<div class="shr">Unsettled bets — enter results</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="warn">{len(unresolved)} bet(s) awaiting result</div>',unsafe_allow_html=True)
            for idx,row in unresolved.iterrows():
                uc1,uc2,uc3=st.columns([3,1,1])
                uc1.markdown(f"R{row.get('round','?')} · {row.get('home','?')} v {row.get('away','?')} · **{row.get('bet_side','?')}** @ {row.get('odds','?')} · ${row.get('stake','?')}")
                res=uc2.selectbox("Result",["—","WIN","LOSS"],key=f"res_{idx}")
                if uc3.button("Settle",key=f"settle_{idx}") and res!="—":
                    all_bets=load_bets()
                    # Find matching bet
                    for b in all_bets:
                        if (b.get("home")==row.get("home") and b.get("round")==row.get("round")
                            and b.get("result") is None):
                            b["result"]=res
                            stake=float(b.get("stake",0)); odds=float(b.get("odds",1))
                            b["pnl"]=round(stake*(odds-1),2) if res=="WIN" else round(-stake,2)
                            break
                    save_bets(all_bets)
                    st.rerun()

        # P&L summary
        settled=bdf_s[bdf_s["pnl"].notna()] if "pnl" in bdf_s.columns else pd.DataFrame()
        if len(settled)>0:
            total_pnl=settled["pnl"].sum()
            total_staked=settled["stake"].sum()
            roi=total_pnl/total_staked*100 if total_staked>0 else 0
            wins=(settled["pnl"]>0).sum(); n=len(settled)

            k1,k2,k3,k4=st.columns(4)
            for col,lab,val,sub,kls in [
                (k1,"Total P&L",f"${total_pnl:+.0f}",f"{n} settled bets","kpi-pos" if total_pnl>=0 else "kpi-neg"),
                (k2,"ROI",f"{roi:+.1f}%",f"${total_staked:.0f} staked","kpi-pos" if roi>=0 else "kpi-neg"),
                (k3,"Win Rate",f"{wins/n:.0%}",f"{wins}W / {n-wins}L","kpi-neu"),
                (k4,"Avg P&L/bet",f"${total_pnl/n:+.1f}","per settled bet","kpi-pos" if total_pnl>=0 else "kpi-neg"),
            ]:
                col.markdown(f'<div class="kpi {kls}"><div class="kpi-label">{lab}</div>'
                             f'<div class="kpi-val">{val}</div><div class="kpi-sub">{sub}</div></div>',unsafe_allow_html=True)

            # P&L curve
            st.markdown('<div class="shr">P&L curve</div>',unsafe_allow_html=True)
            settled_s=settled.sort_values("round") if "round" in settled.columns else settled
            settled_s=settled_s.reset_index(drop=True)
            settled_s["cum_pnl"]=settled_s["pnl"].cumsum()
            settled_s["cum_bank"]=bankroll+settled_s["cum_pnl"]
            fig_pnl=go.Figure()
            fig_pnl.add_trace(go.Scatter(y=settled_s["cum_bank"],mode="lines+markers",
                name="Running bankroll",line=dict(color="#3B82F6",width=2.5),marker=dict(size=5)))
            fig_pnl.add_hline(y=bankroll,line_dash="dash",line_color="#64748B",
                annotation_text=f"Start ${bankroll}")
            at(fig_pnl,height=280,yaxis_title="Bankroll ($)",xaxis_title="Bet #")
            st.plotly_chart(fig_pnl,use_container_width=True)

            # By bet type
            st.markdown('<div class="shr">Breakdown by bet type</div>',unsafe_allow_html=True)
            if "bet_type" in settled.columns:
                bt_grp=settled.groupby("bet_type").agg(
                    n=("pnl","count"),wins=("pnl",lambda x:(x>0).sum()),
                    pnl=("pnl","sum"),staked=("stake","sum")).reset_index()
                bt_grp["win_rate"]=(bt_grp["wins"]/bt_grp["n"]).apply(lambda x:f"{x:.0%}")
                bt_grp["roi"]=(bt_grp["pnl"]/bt_grp["staked"]*100).apply(lambda x:f"{x:+.1f}%")
                bt_grp["pnl"]=bt_grp["pnl"].apply(lambda x:f"${x:+.0f}")
                st.dataframe(bt_grp[["bet_type","n","win_rate","pnl","roi"]],
                             use_container_width=True,height=160)

            # Full log
            st.markdown('<div class="shr">Full bet log</div>',unsafe_allow_html=True)
            log_disp=settled[["round","home","away","bet_type","bet_side","odds","stake","pnl","result"]].copy() if all(c in settled.columns for c in ["round","pnl","result"]) else settled
            def hl_pnl(v):
                if isinstance(v,(int,float)): return f"color:{'#10B981' if v>=0 else '#EF4444'};font-weight:bold"
                return ""
            if "pnl" in log_disp.columns:
                st.dataframe(log_disp.style.map(hl_pnl,subset=["pnl"]),use_container_width=True,height=300)
            st.download_button("Download bet history CSV",
                bdf_s.to_csv(index=False).encode(),f"{league}_{season}_bets.csv","text/csv")

# ════════════════════ TAB 6: MODEL ═══════════════════════════════════════════
with tabs[5]:
    st.markdown("## Model Comparison — M3 vs M3+")
    if n_pl>0:
        model_rows=[]
        for ver,lab in [("v1","M3 (original)"),("v2","M3+ (enhanced)")]:
            wc=f"WL_Pred_{ver}"; pc=f"WL_Prob_A_{ver}"
            if wc not in played.columns: continue
            for seas in sorted(played["Season"].unique()):
                sub=played[played["Season"]==seas]
                if len(sub)<5: continue
                a=accuracy_score(sub["A_Win"],sub[wc])
                try: au=roc_auc_score(sub["A_Win"],sub[pc])
                except: au=float("nan")
                model_rows.append({"Model":lab,"Season":int(seas),"Accuracy":a,"AUC":au,"n":len(sub)})
        if model_rows:
            mdf=pd.DataFrame(model_rows)
            fig_mc=go.Figure()
            for ver,lab,col in [("M3 (original)","M3","#3B82F6"),("M3+ (enhanced)","M3+","#8B5CF6")]:
                sub=mdf[mdf["Model"]==ver]
                if len(sub)==0: continue
                fig_mc.add_trace(go.Scatter(x=sub["Season"].astype(str),y=sub["Accuracy"],
                    mode="lines+markers+text",name=lab,line=dict(color=col,width=2),
                    marker=dict(size=8,color=col),text=[f"{v:.1%}" for v in sub["Accuracy"]],
                    textposition="top center",textfont=dict(size=10,color=col)))
            fig_mc.add_hline(y=0.524,line_dash="dot",line_color="#F59E0B",annotation_text="BEP")
            fig_mc.add_hline(y=0.62,line_dash="dot",line_color="#10B981",annotation_text="Target")
            at(fig_mc,height=350,yaxis=dict(tickformat=".0%",range=[0.45,0.75],gridcolor="#334155"),
               title_text="WL Accuracy: M3 vs M3+ by Season")
            st.plotly_chart(fig_mc,use_container_width=True)

    # Feature importance
    st.markdown('<div class="shr">Feature importance — ElasticNet coefficients</div>',unsafe_allow_html=True)
    for ver,lab,col_bar in [("v1","M3","#3B82F6"),("v2","M3+","#8B5CF6")]:
        wl_v=R.get(f"wl_{ver}"); F_v=R.get(f"F_{ver}")
        if wl_v is None or F_v is None: continue
        coefs=dict(zip(F_v,wl_v.coef_[0]))
        coefs_nonzero={k:v for k,v in coefs.items() if v!=0}
        if not coefs_nonzero: continue
        sorted_c=sorted(coefs_nonzero.items(),key=lambda x:abs(x[1]),reverse=True)
        names=[k.replace("Diff_Form_","").replace("Diff_","") for k,_ in sorted_c]
        vals=[v for _,v in sorted_c]
        fig_fi=go.Figure(go.Bar(x=vals,y=names,orientation="h",
            marker_color=["#10B981" if v>0 else "#EF4444" for v in vals],
            text=[f"{v:+.4f}" for v in vals],textposition="outside",
            textfont=dict(size=10,color="#F1F5F9"),
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>"))
        at(fig_fi,height=max(250,len(sorted_c)*35),xaxis_title="Coefficient",
           title_text=f"{lab} — retained features ({len(coefs_nonzero)}/{len(F_v)})")
        st.plotly_chart(fig_fi,use_container_width=True)

    # Betting value map
    if n_pl>0:
        st.markdown('<div class="shr">Betting value map — model confidence vs result</div>',unsafe_allow_html=True)
        pc=f"WL_Prob_A_{mver}"
        if pc in played.columns:
            played_plot=played[[pc,"A_Win","Margin","A_Name","B_Name","Round"]].copy()
            played_plot["jitter"]=np.random.uniform(-0.03,0.03,len(played_plot))
            played_plot["Result"]=played_plot["A_Win"].map({1:"Home Win",0:"Away Win"})
            played_plot["correct"]=(
                ((played_plot[pc]>=0.5)&(played_plot["A_Win"]==1))|
                ((played_plot[pc]<0.5)&(played_plot["A_Win"]==0))
            )
            fig_vm=go.Figure()
            for correct,col_,name in [(True,"#10B981","Correct"),(False,"#EF4444","Incorrect")]:
                sub=played_plot[played_plot["correct"]==correct]
                fig_vm.add_trace(go.Scatter(
                    x=sub[pc],y=sub["A_Win"]+sub["jitter"],mode="markers",name=name,
                    marker=dict(size=7,color=col_,opacity=0.7),
                    hovertemplate="R%{customdata[2]}: %{customdata[0]} v %{customdata[1]}<br>P(Home)=%{x:.0%}<extra></extra>",
                    customdata=list(zip(sub["A_Name"],sub["B_Name"],sub["Round"]))))
            fig_vm.add_vline(x=0.5,line_dash="dash",line_color="#64748B")
            at(fig_vm,height=320,xaxis=dict(tickformat=".0%",title="Model P(Home Win)",gridcolor="#334155"),
               yaxis=dict(tickvals=[0,1],ticktext=["Away Win","Home Win"],gridcolor="#334155"))
            st.plotly_chart(fig_vm,use_container_width=True)
        st.markdown("""<div class="info">
<b>Reading this chart:</b> Top-right and bottom-left = correct predictions.
Top-left = model said Away but Home won (underdog wins).
Bottom-right = model said Home but Away won.
Clusters in the "incorrect" quadrants show where the model systematically struggles.
</div>""",unsafe_allow_html=True)

# ════════════════════ TAB 7: WEEKLY INPUT ════════════════════════════════════
with tabs[6]:
    st.markdown("## Weekly Input")
    inp_mode=st.radio("Input method",["CSV upload (from stats programme)","Manual entry"],horizontal=True)
    inp_rnd=last_rnd+1
    st.markdown(f"### {league} {season} — Round {inp_rnd}")

    new_data=None

    if inp_mode=="CSV upload (from stats programme)":
        st.markdown("""<div class="info">
Upload a CSV with one row per team per match. Required columns:<br>
<code>Season, Round, Match ID, A Team, B Team, Home Advantage,
A_Points Scored, B_Points Scored</code> + stat columns (A_StatName, B_StatName)
</div>""",unsafe_allow_html=True)
        csv_file=st.file_uploader("Upload CSV",type=["csv"])
        if csv_file:
            try:
                new_data=pd.read_csv(csv_file)
                st.success(f"Loaded {len(new_data)} rows, {len(new_data.columns)} columns")
                st.dataframe(new_data.head(5),use_container_width=True)
            except Exception as e:
                st.error(f"CSV error: {e}")
    else:
        n_m=st.number_input("Matches",1,16,8 if league=="NRL" else 6,key="inp_nm")
        stats_inp=NRL_STATS if league=="NRL" else SL_STATS
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
                for j,stat in enumerate(stats_inp):
                    short=(stat.replace("PTB - ","").replace("Kick Chase - ","")
                           .replace("Ball Run - ","").replace("Receipt - ","")[:18])
                    idx=(j%2)*2
                    sv_a[stat]=cg[idx].number_input(f"A:{short}",0.0,9999.0,0.0,1.0,key=f"isxa_{i}_{j}")
                    sv_b[stat]=cg[idx+1].number_input(f"B:{short}",0.0,9999.0,0.0,1.0,key=f"isxb_{i}_{j}")
            row={"Season":int(season),"Round":int(inp_rnd),"Match ID":99000+i,
                 "A Team":hc_i,"B Team":ac_i,"Home Advantage":ha_i,
                 "A_Points Scored":float(sa),"B_Points Scored":float(sb)}
            for stat in stats_inp:
                row[f"A_{stat}"]=sv_a[stat]; row[f"B_{stat}"]=sv_b[stat]
            match_rows.append(row)
        new_data=pd.DataFrame(match_rows)

    cp,cs=st.columns(2)
    with cp:
        if st.button("Preview ELO updates") and new_data is not None:
            _,ne=update_elos_for_new_matches(new_data,current_elos,int(season))
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
        sub_btn=st.button("Submit & Regenerate",type="primary",use_container_width=True,
                          disabled=(new_data is None or len(new_data)==0))

    if sub_btn and new_data is not None and len(new_data)>0:
        st.cache_data.clear()  # Force full reload after new data
        with st.spinner("Updating..."):
            Rn=run_pipeline(str(master_path),league,target_season=int(season),new_matches_df=new_data)
        # Save back to Google Sheets (primary) or offer xlsx download (fallback)
        if has_sheets_config():
            upd = Rn["df"]
            keep = [c for c in upd.columns if not c.startswith("Diff_Form")
                    and not c.startswith("A_Form") and not c.startswith("B_Form")]
            saved = write_master_to_sheets(upd[keep], league)
            if saved:
                st.cache_data.clear()
                st.success(f"Round {inp_rnd} saved to Google Sheets automatically! Dashboard will refresh.")
                st.rerun()
            else:
                st.cache_data.clear()
                st.warning("Could not save to Sheets — download the xlsx below and upload to GitHub.")
        else:
            st.cache_data.clear()
            st.success(f"Round {inp_rnd} added! Download the Updated Master xlsx and upload to GitHub.")
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
