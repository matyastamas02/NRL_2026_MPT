"""
xLadder Weekly App  v2.3
========================
Master data lives in the GitHub repo as NRL_master.xlsx / SL_master.xlsx.
No file uploads needed during weekly use.

Weekly workflow:
  1. Weekly Input tab  -> enter completed round stats -> Submit
  2. Next Round tab    -> enter upcoming fixtures + odds/lines -> Get Edges
  3. Download PNGs for socials
  4. Download Updated Master xlsx -> upload to GitHub once
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from xladder_pipeline import (
    run_pipeline, append_new_round,
    get_current_elos, update_elos_for_new_matches,
    NRL_STATS, SL_STATS, BRAND,
    build_form_features, train_and_predict
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="xLadder Weekly",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
body,.stApp{background-color:#0F172A;color:#F1F5F9}
.stTabs [role="tab"]{color:#94A3B8;font-weight:600;font-size:14px}
.stTabs [aria-selected="true"]{color:#3B82F6;border-bottom:2px solid #3B82F6}
h1,h2,h3{color:#F1F5F9 !important}
.metric-box{background:#1E293B;border-radius:10px;padding:16px 20px;
  border-left:4px solid #3B82F6;margin-bottom:10px}
.metric-label{color:#94A3B8;font-size:12px;text-transform:uppercase;letter-spacing:1px}
.metric-value{color:#F1F5F9;font-size:28px;font-weight:700;margin-top:4px}
.metric-sub{color:#64748B;font-size:12px;margin-top:2px}
.info-box{background:#1E3A5F;border-radius:8px;padding:12px 16px;
  border-left:3px solid #3B82F6;margin:8px 0;color:#93C5FD;font-size:13px}
.success-box{background:#052e16;border-radius:8px;padding:12px 16px;
  border-left:3px solid #10B981;margin:8px 0;color:#6ee7b7;font-size:13px}
.edge-pos{background:#052e16;border-radius:8px;padding:14px 18px;
  border-left:4px solid #10B981;margin:6px 0}
.edge-neg{background:#1c0a0a;border-radius:8px;padding:14px 18px;
  border-left:4px solid #EF4444;margin:6px 0}
.edge-neut{background:#1E293B;border-radius:8px;padding:14px 18px;
  border-left:4px solid #64748B;margin:6px 0}
div[data-testid="stDownloadButton"] button{
  background:#1D4ED8;color:white;border-radius:8px;
  padding:8px 20px;font-weight:600;border:none}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
NRL_MASTER = BASE_DIR / "NRL_master.xlsx"
SL_MASTER  = BASE_DIR / "SL_master.xlsx"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## xLadder Weekly")
    st.markdown("---")
    league = st.selectbox("League", ["NRL","SL"], index=0)
    season = st.number_input("Season", min_value=2022, max_value=2030, value=2026, step=1)
    master_path = NRL_MASTER if league=="NRL" else SL_MASTER
    st.markdown("---")
    if master_path.exists():
        size_kb = master_path.stat().st_size // 1024
        st.markdown(f"""<div class="success-box">
        Master loaded from repo<br><b>{master_path.name}</b> ({size_kb} KB)
        </div>""", unsafe_allow_html=True)
    else:
        st.error(f"Missing: {master_path.name}")
    st.markdown("---")
    st.markdown("""
**Weekly steps:**
1. **Weekly Input** → add round results
2. **Next Round** → add fixtures + odds
3. Download PNGs for socials
4. Download Updated Master → upload to GitHub
""")
    st.caption("xLadder v2.3  |  M3 model")

# ── Check master exists ───────────────────────────────────────────────────────
st.markdown(f"# xLadder Weekly — {league} {season}")

if not master_path.exists():
    st.error(f"Upload **{master_path.name}** to the GitHub repo to get started.")
    st.stop()

# ── Load pipeline ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading model...", ttl=300)
def cached_pipeline(path_str, league, season):
    return run_pipeline(path_str, league, target_season=int(season))

try:
    result = cached_pipeline(str(master_path), league, int(season))
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.exception(e)
    st.stop()

xl           = result["xladder"]
mt           = result["margin_table"]
df           = result["df"]
current_elos = result["current_elos"]

played_df = df[(df["Season"]==int(season)) & df["Played"]]
n_played  = len(played_df)
last_rnd  = int(played_df["Round"].max()) if n_played > 0 else 0
top_team  = xl.iloc[0]["Team"] if len(xl) else "—"

# ── Metrics ───────────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
for col,label,val,sub in zip([c1,c2,c3,c4],
    ["Matches Played","Teams","Last Round","xLadder Leader"],
    [n_played, len(xl), last_rnd, top_team],
    ["this season","in standings",f"of {season}","by Expected PPG"]):
    col.markdown(f"""<div class="metric-box">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{val}</div>
      <div class="metric-sub">{sub}</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📊 xLadder (PPG)",
    "📈 Expected vs Actual Margin",
    "🎯 Next Round — Fixtures & Betting",
    "📥 Weekly Input — Add completed round",
    "📋 Full Match Log",
])

# ════════════ TAB 1: xLADDER ════════════════════════════════════════════════
with tab1:
    ca,cb = st.columns([1,1.6])
    with ca:
        st.markdown("#### Standings Table")
        disp = xl[["Team","GP","Expected_PPG","Actual_PPG","PPG_Diff"]].copy()
        disp.columns = ["Team","GP","xPPG (model)","Actual PPG","Diff PPG"]
        for c in ["xPPG (model)","Actual PPG","Diff PPG"]:
            disp[c] = disp[c].round(3)
        def hl_ppg(v):
            if isinstance(v,float):
                return f"color:{'#10B981' if v>0 else '#EF4444' if v<0 else '#64748B'};font-weight:bold"
            return ""
        st.dataframe(disp.style.map(hl_ppg,subset=["Diff PPG"]),
                     use_container_width=True,height=500)
        st.download_button("Download xLadder CSV",
            disp.to_csv(index=True).encode(),
            f"{league}_{season}_xladder.csv","text/csv")
    with cb:
        st.markdown("#### Chart")
        buf=io.BytesIO()
        result["fig_xladder"].savefig(buf,format="png",dpi=150,
            bbox_inches="tight",facecolor=BRAND["bg"]); buf.seek(0)
        st.image(buf,use_container_width=True)
        st.download_button("Download xLadder PNG",buf.getvalue(),
            f"{league}_{season}_xladder.png","image/png")
    plt.close("all")

# ════════════ TAB 2: MARGIN CHART ════════════════════════════════════════════
with tab2:
    ca,cb = st.columns([1,1.6])
    with ca:
        st.markdown("#### Margin Table")
        disp2=mt[["Team","GP","Actual_Margin","Expected_Margin","Margin_Diff","Avg_Diff"]].copy()
        disp2.columns=["Team","GP","Actual","Expected","Total Diff","Avg Diff"]
        def hl_mg(v):
            if isinstance(v,(int,float)):
                return f"color:{'#10B981' if v>0 else '#EF4444' if v<0 else '#64748B'};font-weight:bold"
            return ""
        st.dataframe(disp2.style.map(hl_mg,subset=["Total Diff","Avg Diff"]),
                     use_container_width=True,height=500)
        st.download_button("Download Margin CSV",
            disp2.to_csv(index=False).encode(),
            f"{league}_{season}_margins.csv","text/csv")
        st.markdown("""<div class="info-box">
<b>(+)</b> = outperforming model  |  <b>(−)</b> = underperforming model
</div>""",unsafe_allow_html=True)
    with cb:
        st.markdown("#### Chart  (shareable)")
        buf2=io.BytesIO()
        result["fig_margin"].savefig(buf2,format="png",dpi=150,
            bbox_inches="tight",facecolor=BRAND["bg"]); buf2.seek(0)
        st.image(buf2,use_container_width=True)
        st.download_button("Download Margin PNG",buf2.getvalue(),
            f"{league}_{season}_margin_vs_expected.png","image/png")
    plt.close("all")

# ════════════ TAB 3: NEXT ROUND — FIXTURES & BETTING ════════════════════════
with tab3:
    st.markdown("## Next Round — Fixtures & Betting Lines")
    st.markdown("""<div class="info-box">
Enter next round's fixtures and bookmaker odds/lines below.
The model will generate predictions and calculate the edge for each match.
<b>Edge = Model Margin − Bookmaker Line.</b>
A large positive or negative edge = potential value bet.
</div>""", unsafe_allow_html=True)

    stats     = NRL_STATS if league=="NRL" else SL_STATS
    all_teams = sorted(set(df["A Team"]) | set(df["B Team"]))
    next_rnd  = last_rnd + 1

    st.markdown(f"### {league} {season} — Round {next_rnd}")

    n_fix = st.number_input("Number of fixtures",
        min_value=1, max_value=16,
        value=8 if league=="NRL" else 6, step=1)

    fixture_rows = []
    for i in range(int(n_fix)):
        with st.expander(f"Fixture {i+1}", expanded=True):
            fc1,fc2,fc3 = st.columns([2,2,1])
            home = fc1.selectbox("Home team", all_teams,
                key=f"fh_{i}", index=min(i*2,   len(all_teams)-1))
            away = fc2.selectbox("Away team", all_teams,
                key=f"fa_{i}", index=min(i*2+1, len(all_teams)-1))
            venue = fc3.selectbox("Venue",
                ["A (home)","B (away)","neutral"], key=f"fv_{i}")
            ha = venue.split(" ")[0]

            st.markdown("**Bookmaker odds & line:**")
            oc1,oc2,oc3,oc4,oc5 = st.columns(5)
            h2h_home  = oc1.number_input("H2H Home",  min_value=1.01, value=1.80,
                step=0.05, key=f"h2h_h_{i}", format="%.2f")
            h2h_away  = oc2.number_input("H2H Away",  min_value=1.01, value=2.05,
                step=0.05, key=f"h2h_a_{i}", format="%.2f")
            line      = oc3.number_input("Line (Home)", min_value=-50.0, value=-6.5,
                step=0.5, key=f"line_{i}", format="%.1f",
                help="Negative = home favourite. e.g. −6.5 means home gives 6.5")
            line_odds_h = oc4.number_input("Line Odds H", min_value=1.01, value=1.91,
                step=0.01, key=f"lo_h_{i}", format="%.2f")
            line_odds_a = oc5.number_input("Line Odds A", min_value=1.01, value=1.91,
                step=0.01, key=f"lo_a_{i}", format="%.2f")

        fixture_rows.append({
            "Home": home, "Away": away, "HA": ha,
            "H2H_Home": h2h_home, "H2H_Away": h2h_away,
            "Line": line, "LineOdds_H": line_odds_h, "LineOdds_A": line_odds_a,
        })

    if st.button("Generate Predictions & Edges", type="primary",
                  use_container_width=True):

        # Build prediction rows using current model state
        stats_list = NRL_STATS if league=="NRL" else SL_STATS
        df_pred, form_cols = build_form_features(df, stats_list)
        train_seasons = sorted(df_pred[df_pred["Season"]<int(season)]["Season"].unique())
        if not train_seasons: train_seasons = [int(season)]
        df_pred = train_and_predict(df_pred, form_cols, train_seasons)

        # Get latest ELO and Form for each team
        team_elos = {}
        for _, row in df_pred.sort_values(["Season","Round"]).iterrows():
            team_elos[row["A Team"]] = float(row["ELO_A"])
            team_elos[row["B Team"]] = float(row["ELO_B"])

        # Get rolling form for each team (last known value)
        team_form = {}
        for fc in form_cols:
            stat_name = fc.replace("Diff_Form_","")
            a_fc = f"A_Form_{stat_name}"
            b_fc = f"B_Form_{stat_name}"
            for _, row in df_pred.sort_values(["Season","Round"]).iterrows():
                if pd.notna(row.get(a_fc)):
                    team_form[f"{row['A Team']}_{fc}"] = row[a_fc]
                if pd.notna(row.get(b_fc)):
                    team_form[f"{row['B Team']}_{fc}"] = row[b_fc]

        # Score each fixture
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import StandardScaler

        F = ["Diff ELO","Home_flag"] + form_cols
        train_data = df_pred[df_pred["Season"].isin(train_seasons) & df_pred["Played"]]
        sc = StandardScaler(); sc.fit(train_data[F].fillna(0))
        wl = LogisticRegression(C=0.05,l1_ratio=0.7,penalty="elasticnet",
                                 solver="saga",max_iter=2000,random_state=42)
        wl.fit(sc.transform(train_data[F].fillna(0)), train_data["A_Win"])
        mg = Ridge(alpha=1.0)
        mg.fit(sc.transform(train_data[F].fillna(0)), train_data["Margin"])

        results_list = []
        for fx in fixture_rows:
            home_team = fx["Home"]; away_team = fx["Away"]
            ha_map    = {"A":1,"B":-1,"neutral":0}
            home_flag = ha_map.get(fx["HA"], 0)

            elo_h = team_elos.get(home_team, 2000)
            elo_a = team_elos.get(away_team, 2000)
            diff_elo = elo_h - elo_a

            feat_row = {"Diff ELO": diff_elo, "Home_flag": home_flag}
            for fc in form_cols:
                stat_name = fc.replace("Diff_Form_","")
                fh = team_form.get(f"{home_team}_Diff_Form_{stat_name}", 0) or 0
                fa = team_form.get(f"{away_team}_Diff_Form_{stat_name}", 0) or 0
                feat_row[fc] = fh - fa

            X = np.array([[feat_row.get(f,0) for f in F]])
            X_s = sc.transform(X)
            prob_home = float(wl.predict_proba(X_s)[0,1])
            margin    = float(mg.predict(X_s)[0])

            # Market implied probs (vig-removed)
            imp_h_raw = 1/fx["H2H_Home"]; imp_a_raw = 1/fx["H2H_Away"]
            vig       = imp_h_raw + imp_a_raw
            imp_home  = imp_h_raw / vig
            imp_away  = imp_a_raw / vig

            # Edges
            h2h_edge  = prob_home - imp_home          # model vs market prob
            line_edge  = margin - (-fx["Line"])        # model margin vs line

            # Bet recommendations
            # H2H: bet home if model prob > market implied
            h2h_bet    = "Home" if h2h_edge > 0 else "Away"
            h2h_bet_odds = fx["H2H_Home"] if h2h_edge > 0 else fx["H2H_Away"]

            # Line: bet home covers if line_edge > 0, away covers if < 0
            line_bet   = "Home covers" if line_edge > 0 else "Away covers"
            line_bet_odds = fx["LineOdds_H"] if line_edge > 0 else fx["LineOdds_A"]

            # Confidence flags
            h2h_conf  = abs(h2h_edge)
            line_conf = abs(line_edge)

            results_list.append({
                "Home": home_team, "Away": away_team,
                "ELO Home": round(elo_h,0), "ELO Away": round(elo_a,0),
                "Prob Home": round(prob_home,3),
                "Prob Away": round(1-prob_home,3),
                "Model Margin": round(margin,1),
                "Bookmaker Line": fx["Line"],
                "Line Edge": round(line_edge,2),
                "Line Bet": line_bet,
                "Line Odds": line_bet_odds,
                "H2H Edge": round(h2h_edge,3),
                "H2H Bet": h2h_bet,
                "H2H Odds": h2h_bet_odds,
                "H2H Conf": h2h_conf,
                "Line Conf": line_conf,
            })

        st.markdown("---")
        st.markdown("### Predictions & Edges")

        # Summary table
        summary_df = pd.DataFrame(results_list)[[
            "Home","Away","Prob Home","Model Margin",
            "Bookmaker Line","Line Edge","Line Bet",
            "H2H Edge","H2H Bet"
        ]]
        summary_df["Prob Home"] = summary_df["Prob Home"].apply(lambda x: f"{x:.0%}")
        summary_df["Model Margin"]    = summary_df["Model Margin"].apply(lambda x: f"{x:+.1f}")
        summary_df["Bookmaker Line"]  = summary_df["Bookmaker Line"].apply(lambda x: f"{x:+.1f}")
        summary_df["Line Edge"]       = summary_df["Line Edge"].apply(lambda x: f"{x:+.2f}")
        summary_df["H2H Edge"]        = summary_df["H2H Edge"].apply(lambda x: f"{x:+.3f}")

        def hl_edge(v):
            try:
                n = float(str(v).replace("+",""))
                if n > 0.5 or n > 3: return "color:#10B981;font-weight:bold"
                if n < -0.5 or n < -3: return "color:#EF4444;font-weight:bold"
            except: pass
            return "color:#94A3B8"

        st.dataframe(summary_df.style.map(hl_edge, subset=["Line Edge","H2H Edge"]),
                     use_container_width=True, height=380)

        # Per-fixture cards
        st.markdown("### Detailed View")
        EDGE_THRESH = 3.0   # pts edge to flag as value bet

        for r in results_list:
            line_sig = abs(r["Line Edge"]) >= EDGE_THRESH
            h2h_sig  = abs(r["H2H Edge"])  >= 0.08

            box_class = "edge-pos" if (line_sig and r["Line Edge"]>0) or \
                        (h2h_sig and r["H2H Edge"]>0) else \
                        "edge-neg" if (line_sig and r["Line Edge"]<0) or \
                        (h2h_sig and r["H2H Edge"]<0) else "edge-neut"

            with st.container():
                cc1,cc2,cc3,cc4 = st.columns([2.5,1.5,1.5,1.5])

                with cc1:
                    win_team = r["Home"] if r["Prob Home"]>0.5 else r["Away"]
                    win_prob = max(r["Prob Home"], r["Prob Away"])
                    st.markdown(f"**{r['Home']}** vs **{r['Away']}**")
                    st.caption(f"Model tips: **{win_team}** ({win_prob:.0%}) | "
                               f"by {abs(r['Model Margin']):.1f} pts | "
                               f"ELO: {r['ELO Home']:.0f} vs {r['ELO Away']:.0f}")

                with cc2:
                    le = r["Line Edge"]
                    le_col = "#10B981" if le>0 else "#EF4444"
                    flag   = " ⭐ VALUE" if abs(le)>=EDGE_THRESH else ""
                    st.markdown(f"**Line Edge:** "
                                f"<span style='color:{le_col};font-weight:bold'>"
                                f"{le:+.1f}{flag}</span>",
                                unsafe_allow_html=True)
                    st.caption(f"Bet: **{r['Line Bet']}** @ {r['Line Odds']:.2f}")

                with cc3:
                    he = r["H2H Edge"]
                    he_col = "#10B981" if he>0 else "#EF4444"
                    flag2  = " ⭐" if abs(he)>=0.08 else ""
                    st.markdown(f"**H2H Edge:** "
                                f"<span style='color:{he_col};font-weight:bold'>"
                                f"{he:+.3f}{flag2}</span>",
                                unsafe_allow_html=True)
                    st.caption(f"Bet: **{r['H2H Bet']}** @ {r['H2H Odds']:.2f}")

                with cc4:
                    if abs(le) >= EDGE_THRESH:
                        st.success(f"Line edge: {le:+.1f} pts")
                    elif abs(he) >= 0.08:
                        st.info(f"H2H edge: {he:+.3f}")
                    else:
                        st.caption("No significant edge")

                st.divider()

        # Download full predictions CSV
        full_csv = pd.DataFrame(results_list)
        st.download_button(
            "Download Full Predictions CSV",
            full_csv.to_csv(index=False).encode(),
            f"{league}_{season}_R{next_rnd}_predictions.csv",
            "text/csv"
        )

        # Chart
        _fig, _ax = plt.subplots(figsize=(10, max(4, len(results_list)*0.7)))
        _fig.patch.set_facecolor(BRAND["bg"])
        _ax.set_facecolor(BRAND["panel"])
        _y = np.arange(len(results_list))
        _le = [r["Line Edge"] for r in results_list]
        _labels = [f"{r['Home'][:10]} v {r['Away'][:10]}" for r in results_list]
        _colors = [BRAND["positive"] if v>0 else BRAND["negative"] for v in _le]
        _ax.barh(_y, _le, color=_colors, alpha=0.85)
        _ax.axvline(0, color=BRAND["neutral"], linewidth=1, linestyle="--")
        _ax.axvline( EDGE_THRESH, color=BRAND["positive"], linewidth=0.8,
                     linestyle=":", alpha=0.6)
        _ax.axvline(-EDGE_THRESH, color=BRAND["negative"], linewidth=0.8,
                     linestyle=":", alpha=0.6)
        for i, (v, lbl) in enumerate(zip(_le, _labels)):
            sign = "+" if v >= 0 else ""
            _ax.text(v + (0.1 if v>=0 else -0.1), i,
                     f"{sign}{v:.1f}", va="center",
                     ha="left" if v>=0 else "right",
                     color=BRAND["text"], fontsize=9, fontweight="bold")
        _ax.set_yticks(_y); _ax.set_yticklabels(_labels, color=BRAND["text"], fontsize=10)
        _ax.set_xlabel("Line Edge (Model Margin − Bookmaker Line)", color=BRAND["text_dim"])
        _ax.set_title(f"{league} {season} R{next_rnd} — Line Edges",
                      color=BRAND["text"], fontsize=13, fontweight="bold")
        _ax.tick_params(colors=BRAND["text_dim"])
        for sp in _ax.spines.values(): sp.set_edgecolor(BRAND["neutral"])
        _fig.tight_layout(pad=1.5)

        buf_bet = io.BytesIO()
        _fig.savefig(buf_bet, format="png", dpi=150,
                     bbox_inches="tight", facecolor=BRAND["bg"])
        buf_bet.seek(0)
        st.image(buf_bet, use_container_width=True)
        st.download_button("Download Edge Chart PNG", buf_bet.getvalue(),
            f"{league}_{season}_R{next_rnd}_edges.png", "image/png")
        plt.close("all")

# ════════════ TAB 4: WEEKLY INPUT ════════════════════════════════════════════
with tab4:
    st.markdown("## Add Completed Round Results")
    st.markdown(f"""<div class="info-box">
Enter the stats for each match from the completed round, then click <b>Submit Round</b>.<br>
ELO updates automatically (K=27). After submitting, download the
<b>Updated Master xlsx</b> and upload it to GitHub as <code>{master_path.name}</code>.
</div>""", unsafe_allow_html=True)

    stats     = NRL_STATS if league=="NRL" else SL_STATS
    all_teams = sorted(set(df["A Team"]) | set(df["B Team"]))
    next_rnd_input = last_rnd + 1

    st.markdown(f"### {league} {season} — Round {next_rnd_input}")

    n_matches = st.number_input("Number of matches",
        min_value=1, max_value=16,
        value=8 if league=="NRL" else 6, step=1)

    match_rows = []
    for i in range(int(n_matches)):
        with st.expander(f"Match {i+1}", expanded=(i<3)):
            mc1,mc2,mc3 = st.columns([2,2,1])
            a_team = mc1.selectbox("Home team", all_teams, key=f"at_{i}",
                                    index=min(i*2,   len(all_teams)-1))
            b_team = mc2.selectbox("Away team", all_teams, key=f"bt_{i}",
                                    index=min(i*2+1, len(all_teams)-1))
            venue  = mc3.selectbox("Venue",["A (home)","B (away)","neutral"],
                                    key=f"v_{i}")
            ha = venue.split(" ")[0]

            sc1,sc2 = st.columns(2)
            score_a = sc1.number_input(f"Score — {a_team[:22]}",
                min_value=0,max_value=200,value=0,key=f"sca_{i}")
            score_b = sc2.number_input(f"Score — {b_team[:22]}",
                min_value=0,max_value=200,value=0,key=f"scb_{i}")

            st.markdown("**Stats — A team (left) / B team (right):**")
            cols_g = st.columns(4)
            sv_a={}; sv_b={}
            for j,stat in enumerate(stats):
                short=(stat.replace("PTB - ","").replace("Kick Chase - ","")
                          .replace("Ball Run - ","").replace("Receipt - ","")
                          .replace("Set Complete - ","Sets ")[:18])
                idx=(j%2)*2
                sv_a[stat]=cols_g[idx  ].number_input(
                    f"A: {short}",min_value=0.0,value=0.0,step=1.0,key=f"a_{i}_{j}")
                sv_b[stat]=cols_g[idx+1].number_input(
                    f"B: {short}",min_value=0.0,value=0.0,step=1.0,key=f"b_{i}_{j}")

        row={"Season":int(season),"Round":int(next_rnd_input),
             "Match ID":99000+i,"A Team":a_team,"B Team":b_team,
             "Home Advantage":ha,"A_Points Scored":float(score_a),
             "B_Points Scored":float(score_b)}
        for stat in stats:
            row[f"A_{stat}"]=sv_a[stat]; row[f"B_{stat}"]=sv_b[stat]
        match_rows.append(row)

    col_prev,col_sub = st.columns([1,1])
    with col_prev:
        if st.button("Preview ELO updates"):
            _new=pd.DataFrame(match_rows)
            _,new_elos=update_elos_for_new_matches(_new,current_elos,int(season))
            _rows=[{"Team":t,"Before":round(current_elos.get(t,2000),1),
                    "After":round(new_elos[t],1),
                    "Change":round(new_elos[t]-current_elos.get(t,2000),1)}
                   for t in sorted(new_elos)]
            _edf=pd.DataFrame(_rows).sort_values("After",ascending=False)
            def hl_e(v):
                if isinstance(v,float):
                    return f"color:{'#10B981' if v>0 else '#EF4444' if v<0 else '#94A3B8'}"
                return ""
            st.dataframe(_edf.style.map(hl_e,subset=["Change"]),
                         use_container_width=True,height=400)
    with col_sub:
        st.markdown("&nbsp;")
        submit=st.button("Submit Round & Regenerate",type="primary",
                          use_container_width=True)

    if submit:
        with st.spinner("Updating..."):
            new_df=pd.DataFrame(match_rows)
            result_new=run_pipeline(str(master_path),league,
                target_season=int(season),new_matches_df=new_df)
        st.success(f"Round {next_rnd_input} added!")
        st.markdown("#### Updated xLadder")
        st.dataframe(result_new["xladder"][
            ["Team","GP","Expected_PPG","Actual_PPG","PPG_Diff"]].round(3),
            use_container_width=True,height=360)

        dl1,dl2,dl3,dl4=st.columns(4)
        b1=io.BytesIO()
        result_new["fig_xladder"].savefig(b1,format="png",dpi=150,
            bbox_inches="tight",facecolor=BRAND["bg"]); b1.seek(0)
        dl1.download_button("xLadder PNG",b1.getvalue(),
            f"{league}_{season}_R{next_rnd_input}_xladder.png","image/png")
        b2=io.BytesIO()
        result_new["fig_margin"].savefig(b2,format="png",dpi=150,
            bbox_inches="tight",facecolor=BRAND["bg"]); b2.seek(0)
        dl2.download_button("Margin PNG",b2.getvalue(),
            f"{league}_{season}_R{next_rnd_input}_margin.png","image/png")

        upd=result_new["df"]
        keep=[c for c in upd.columns
              if not c.startswith("Diff_Form") and not c.startswith("A_Form")
              and not c.startswith("B_Form")]
        bm=io.BytesIO(); upd[keep].to_excel(bm,index=False); bm.seek(0)
        dl4.download_button("Updated Master xlsx",bm.getvalue(),
            master_path.name,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.markdown(f"""<div class="info-box">
<b>To save permanently:</b> Download Updated Master xlsx → upload to GitHub as
<code>{master_path.name}</code> → app auto-reloads.
</div>""",unsafe_allow_html=True)
        plt.close("all")

# ════════════ TAB 5: MATCH LOG ═══════════════════════════════════════════════
with tab5:
    st.markdown("#### All Played Matches")
    if n_played==0:
        st.info("No played matches found.")
    else:
        log=played_df[["Season","Round","A_Name","B_Name",
            "A_Points Scored","B_Points Scored","Margin",
            "WL_Pred","WL_Prob_A","Margin_Pred"]].copy()
        log.columns=["Season","Round","Home","Away","H Pts","A Pts",
                     "Actual Margin","WL Pred","P(Home)","Pred Margin"]
        log["P(Home)"]      =log["P(Home)"].apply(lambda x:f"{x:.0%}")
        log["Pred Margin"]  =log["Pred Margin"].apply(lambda x:f"{x:+.1f}")
        log["Actual Margin"]=log["Actual Margin"].apply(lambda x:f"{x:+.0f}")
        log["Correct?"]=(
            (log["WL Pred"]==1)==
            (log["Actual Margin"].str.replace("+","",regex=False).astype(float)>0)
        ).map({True:"Yes",False:"No"})
        st.dataframe(log,use_container_width=True,height=550)
        correct=(log["Correct?"]=="Yes").sum()
        st.markdown(f"**Model accuracy:** {correct}/{len(log)} = {correct/len(log):.1%}")
        st.download_button("Download CSV",log.to_csv(index=False).encode(),
            f"{league}_{season}_match_log.csv","text/csv")
