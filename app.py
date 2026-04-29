"""
xLadder Weekly App  v2.2
========================
NO FILE UPLOAD NEEDED.
Master data lives in the GitHub repo.
Weekly workflow:
  1. Enter new round stats in the Weekly Input tab
  2. Submit -> all charts update
  3. Download updated master xlsx
  4. Upload the xlsx to GitHub (replaces the old one)
  5. Done. Next week the app auto-loads the new file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from xladder_pipeline import (
    run_pipeline, compute_edge, append_new_round,
    get_current_elos, update_elos_for_new_matches,
    NRL_STATS, SL_STATS, BRAND
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
.stTabs [role="tab"]{color:#94A3B8;font-weight:600}
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
div[data-testid="stDownloadButton"] button{
  background:#1D4ED8;color:white;border-radius:8px;
  padding:8px 20px;font-weight:600;border:none}
</style>
""", unsafe_allow_html=True)

# ── Master file paths (in the repo) ──────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
NRL_MASTER = BASE_DIR / "NRL_master.xlsx"
SL_MASTER  = BASE_DIR / "SL_master.xlsx"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## xLadder Weekly")
    st.markdown("---")
    league = st.selectbox("League", ["NRL", "SL"], index=0)
    season = st.number_input("Season", min_value=2022, max_value=2030, value=2026, step=1)
    st.markdown("---")
    master_path = NRL_MASTER if league == "NRL" else SL_MASTER
    if master_path.exists():
        size_kb = master_path.stat().st_size // 1024
        st.markdown(f"""<div class="success-box">
        Master file loaded from repo<br>
        <b>{master_path.name}</b> ({size_kb} KB)
        </div>""", unsafe_allow_html=True)
    else:
        st.error(f"Master file not found: {master_path.name}")
        st.markdown("""
Upload one of these to the GitHub repo:
- `NRL_master.xlsx` for NRL
- `SL_master.xlsx` for SL
""")
    st.markdown("---")
    st.markdown("""
**Weekly workflow:**
1. Go to **Weekly Input** tab
2. Enter round stats → Submit
3. Download **Updated Master xlsx**
4. Upload it to GitHub (replaces old file)
5. App auto-loads next week
""")
    st.markdown("---")
    st.caption("xLadder v2.2  |  M3 model")

# ── Load master ───────────────────────────────────────────────────────────────
st.markdown(f"# xLadder Weekly — {league} {season}")

if not master_path.exists():
    st.error(f"""
### Master file not found

The app expects **`{master_path.name}`** in the GitHub repository.

**One-time setup:**
1. Rename your master xlsx to `{master_path.name}`
2. Upload it to the GitHub repo
3. Redeploy — after that you never touch GitHub again

The app will update the file automatically every week.
""")
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading model...", ttl=300)
def cached_pipeline(path_str, league, season):
    return run_pipeline(path_str, league, target_season=int(season))

try:
    result = cached_pipeline(str(master_path), league, int(season))
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.exception(e)
    st.stop()

xl   = result["xladder"]
mt   = result["margin_table"]
nxt  = result["next_round"]
nr   = result["next_round_n"]
df   = result["df"]
current_elos = result["current_elos"]

played_df = df[(df["Season"] == int(season)) & df["Played"]]
n_played  = len(played_df)
last_rnd  = int(played_df["Round"].max()) if n_played > 0 else 0
top_team  = xl.iloc[0]["Team"] if len(xl) else "—"

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, label, val, sub in zip(
    [c1, c2, c3, c4],
    ["Matches Played", "Teams", "Last Round", "xLadder Leader"],
    [n_played, len(xl), last_rnd, top_team],
    ["this season", "in standings", f"of {season}", "by Expected PPG"]
):
    col.markdown(f"""<div class="metric-box">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{val}</div>
      <div class="metric-sub">{sub}</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 xLadder (PPG)",
    "📈 Expected vs Actual Margin",
    "🔮 Next Round Predictions",
    "📥 Weekly Input — Add new round",
    "📋 Full Match Log",
])

# ════════════════ TAB 1: xLADDER ═════════════════════════════════════════════
with tab1:
    ca, cb = st.columns([1, 1.6])
    with ca:
        st.markdown("#### Standings Table")
        disp = xl[["Team","GP","Expected_PPG","Actual_PPG","PPG_Diff"]].copy()
        disp.columns = ["Team","GP","xPPG (model)","Actual PPG","Diff PPG"]
        for c in ["xPPG (model)","Actual PPG","Diff PPG"]:
            disp[c] = disp[c].round(3)
        def hl_ppg(v):
            if isinstance(v, float):
                c = "#10B981" if v > 0 else "#EF4444" if v < 0 else "#64748B"
                return f"color:{c};font-weight:bold"
            return ""
        st.dataframe(disp.style.map(hl_ppg, subset=["Diff PPG"]),
                     use_container_width=True, height=500)
        st.download_button("Download xLadder CSV",
            disp.to_csv(index=True).encode(),
            f"{league}_{season}_xladder.csv", "text/csv")
    with cb:
        st.markdown("#### Chart")
        buf = io.BytesIO()
        result["fig_xladder"].savefig(buf, format="png", dpi=150,
            bbox_inches="tight", facecolor=BRAND["bg"])
        buf.seek(0)
        st.image(buf, use_container_width=True)
        st.download_button("Download xLadder PNG", buf.getvalue(),
            f"{league}_{season}_xladder.png", "image/png")
    plt.close("all")

# ════════════════ TAB 2: MARGIN CHART ════════════════════════════════════════
with tab2:
    ca, cb = st.columns([1, 1.6])
    with ca:
        st.markdown("#### Margin Table")
        disp2 = mt[["Team","GP","Actual_Margin","Expected_Margin",
                    "Margin_Diff","Avg_Diff"]].copy()
        disp2.columns = ["Team","GP","Actual","Expected","Total Diff","Avg Diff"]
        def hl_mg(v):
            if isinstance(v, (int, float)):
                c = "#10B981" if v > 0 else "#EF4444" if v < 0 else "#64748B"
                return f"color:{c};font-weight:bold"
            return ""
        st.dataframe(disp2.style.map(hl_mg, subset=["Total Diff","Avg Diff"]),
                     use_container_width=True, height=500)
        st.download_button("Download Margin CSV",
            disp2.to_csv(index=False).encode(),
            f"{league}_{season}_margins.csv", "text/csv")
        st.markdown("""<div class="info-box">
<b>(+)</b> = outperforming model expectations<br>
<b>(−)</b> = underperforming model expectations
</div>""", unsafe_allow_html=True)
    with cb:
        st.markdown("#### Chart  (shareable on socials)")
        buf2 = io.BytesIO()
        result["fig_margin"].savefig(buf2, format="png", dpi=150,
            bbox_inches="tight", facecolor=BRAND["bg"])
        buf2.seek(0)
        st.image(buf2, use_container_width=True)
        st.download_button("Download Margin PNG", buf2.getvalue(),
            f"{league}_{season}_margin_vs_expected.png", "image/png")
    plt.close("all")

# ════════════════ TAB 3: NEXT ROUND ══════════════════════════════════════════
with tab3:
    if nr is None or len(nxt) == 0:
        st.info(
            "No upcoming fixtures in the master file yet. "
            "After adding the new round in Weekly Input, "
            "come back here to enter betting lines."
        )
    else:
        st.markdown(f"#### Round {nr} — Model Predictions")
        st.markdown("""<div class="info-box">
Fill in <b>Line Open</b> and <b>Line Close</b>, then click Calculate Edge.
</div>""", unsafe_allow_html=True)

        edit_cols = ["A_Name","B_Name","WL_Prob_A","Margin_Pred",
                     "Pred_Winner","Pred_Margin_Abs","Line_Open","Line_Close"]
        avail = [c for c in edit_cols if c in nxt.columns]
        edit_df = nxt[avail].copy()
        rename = {
            "A_Name":"Home","B_Name":"Away",
            "WL_Prob_A":"P(Home Win)","Margin_Pred":"Model Margin",
            "Pred_Winner":"Predicted Winner","Pred_Margin_Abs":"Win by",
            "Line_Open":"Line Open","Line_Close":"Line Close",
        }
        edit_df = edit_df.rename(columns={k:v for k,v in rename.items() if k in edit_df})
        if "P(Home Win)" in edit_df.columns:
            edit_df["P(Home Win)"] = edit_df["P(Home Win)"].apply(lambda x: f"{x:.0%}")
        if "Model Margin" in edit_df.columns:
            edit_df["Model Margin"] = edit_df["Model Margin"].apply(lambda x: f"{x:+.1f}")

        edited = st.data_editor(edit_df, use_container_width=True,
                                 num_rows="fixed", key="preds_ed")

        if st.button("Calculate Edge", type="primary"):
            for i, row in edited.iterrows():
                nxt.loc[i,"Line_Open"]  = row.get("Line Open","")
                nxt.loc[i,"Line_Close"] = row.get("Line Close","")
            nxt_e = compute_edge(nxt)
            st.markdown("#### Edge Results")
            for _, r in nxt_e.iterrows():
                a=r.get("A_Name",""); b=r.get("B_Name","")
                eo=r.get("Model_Edge_Open",np.nan)
                ec=r.get("Model_Edge_Close",np.nan)
                prob=r.get("WL_Prob_A",0.5); marg=r.get("Margin_Pred",0)
                cc1,cc2,cc3,cc4 = st.columns([2,2,1.5,1.5])
                cc1.metric(f"{a} vs {b}",
                    f"{'Home' if prob>=0.5 else 'Away'} {max(prob,1-prob):.0%}",
                    f"Model: {marg:+.1f}")
                cc2.metric("Tip", r.get("Pred_Winner","—"),
                    f"by {r.get('Pred_Margin_Abs','—')} pts")
                cc3.metric("Edge vs Open",
                    f"{eo:+.1f}" if not pd.isna(eo) else "—",
                    "Bet A" if not pd.isna(eo) and eo>0 else "Bet B" if not pd.isna(eo) else "")
                cc4.metric("Edge vs Close",
                    f"{ec:+.1f}" if not pd.isna(ec) else "—",
                    "Bet A" if not pd.isna(ec) and ec>0 else "Bet B" if not pd.isna(ec) else "")
                st.divider()
            st.download_button("Download Predictions CSV",
                nxt_e.to_csv(index=False).encode(),
                f"{league}_{season}_R{nr}_predictions.csv","text/csv")

        buf3 = io.BytesIO()
        result["fig_nextround"].savefig(buf3, format="png", dpi=150,
            bbox_inches="tight", facecolor=BRAND["bg"])
        buf3.seek(0)
        st.image(buf3, use_container_width=True)
        st.download_button("Download Round Preview PNG", buf3.getvalue(),
            f"{league}_{season}_R{nr}_preview.png","image/png")
        plt.close("all")

# ════════════════ TAB 4: WEEKLY INPUT ════════════════════════════════════════
with tab4:
    st.markdown("## Add New Round Results")
    st.markdown(f"""<div class="info-box">
<b>How this works:</b><br>
1. Enter the stats for each match from the completed round.<br>
2. Click <b>Submit Round</b>.<br>
3. ELO is computed automatically (K=27, carried forward from the master).<br>
4. Download the <b>Updated Master xlsx</b>.<br>
5. Upload it to GitHub as <code>{master_path.name}</code> — replaces the old file.<br>
6. That's it. The app auto-loads it next time.
</div>""", unsafe_allow_html=True)

    stats = NRL_STATS if league == "NRL" else SL_STATS
    teams = sorted(set(df["A Team"]) | set(df["B Team"]))
    next_round_num = last_rnd + 1

    st.markdown(f"### {league} {season} — Round {next_round_num}")

    n_matches = st.number_input(
        "Number of matches this round",
        min_value=1, max_value=16,
        value=8 if league == "NRL" else 6,
        step=1
    )

    match_rows = []
    for i in range(int(n_matches)):
        with st.expander(f"Match {i+1}", expanded=(i < 3)):
            mc1, mc2, mc3 = st.columns([2,2,1])
            a_team = mc1.selectbox("Home team", teams, key=f"at_{i}",
                                    index=min(i*2, len(teams)-1))
            b_team = mc2.selectbox("Away team", teams, key=f"bt_{i}",
                                    index=min(i*2+1, len(teams)-1))
            venue  = mc3.selectbox("Venue", ["A (home)","B (away)","neutral"],
                                    key=f"v_{i}")
            ha = venue.split(" ")[0]

            sc1, sc2 = st.columns(2)
            score_a = sc1.number_input(f"Score — {a_team[:22]}",
                min_value=0, max_value=200, value=0, key=f"sca_{i}")
            score_b = sc2.number_input(f"Score — {b_team[:22]}",
                min_value=0, max_value=200, value=0, key=f"scb_{i}")

            st.markdown("**Stats — A (left) and B (right) for each:**")
            n_cols = 2
            cols_grid = st.columns(n_cols * 2)
            sv_a = {}; sv_b = {}
            for j, stat in enumerate(stats):
                short = (stat.replace("PTB - ","").replace("Kick Chase - ","")
                            .replace("Ball Run - ","").replace("Receipt - ","")
                            .replace("Set Complete - ","Sets ")[:20])
                idx = (j % n_cols) * 2
                sv_a[stat] = cols_grid[idx].number_input(
                    f"A: {short}", min_value=0.0, value=0.0, step=1.0, key=f"a_{i}_{j}")
                sv_b[stat] = cols_grid[idx+1].number_input(
                    f"B: {short}", min_value=0.0, value=0.0, step=1.0, key=f"b_{i}_{j}")

        row = {
            "Season": int(season), "Round": int(next_round_num),
            "Match ID": 99000 + i,
            "A Team": a_team, "B Team": b_team,
            "Home Advantage": ha,
            "A_Points Scored": float(score_a),
            "B_Points Scored": float(score_b),
        }
        for stat in stats:
            row[f"A_{stat}"] = sv_a[stat]
            row[f"B_{stat}"] = sv_b[stat]
        match_rows.append(row)

    # ── Buttons ───────────────────────────────────────────────────────────
    col_prev, col_submit = st.columns([1,1])

    with col_prev:
        if st.button("Preview ELO updates"):
            new_df_prev = pd.DataFrame(match_rows)
            _, new_elos = update_elos_for_new_matches(
                new_df_prev, current_elos, prev_season=int(season))
            elo_rows = [{"Team": t,
                         "Before": round(current_elos.get(t, 2000), 1),
                         "After":  round(new_elos[t], 1),
                         "Change": round(new_elos[t] - current_elos.get(t, 2000), 1)}
                        for t in sorted(new_elos.keys())]
            elo_df = pd.DataFrame(elo_rows).sort_values("After", ascending=False)
            def hl_elo(v):
                if isinstance(v, float):
                    return f"color:{'#10B981' if v>0 else '#EF4444' if v<0 else '#94A3B8'}"
                return ""
            st.dataframe(elo_df.style.map(hl_elo, subset=["Change"]),
                         use_container_width=True, height=420)

    with col_submit:
        st.markdown("&nbsp;")
        submit = st.button("Submit Round & Regenerate",
                            type="primary", use_container_width=True)

    if submit:
        with st.spinner(f"Adding Round {next_round_num}, recomputing..."):
            new_df = pd.DataFrame(match_rows)
            result_new = run_pipeline(
                str(master_path), league,
                target_season=int(season),
                new_matches_df=new_df
            )

        st.success(f"Round {next_round_num} added successfully!")
        st.markdown("""<div class="success-box">
<b>Next step:</b> Download the Updated Master xlsx below,
then upload it to GitHub as <code>{}</code> to replace the current file.
The app will auto-load it from now on — no more manual uploads needed.
</div>""".format(master_path.name), unsafe_allow_html=True)

        # Updated xLadder preview
        st.markdown("#### Updated xLadder")
        xl_new = result_new["xladder"]
        st.dataframe(
            xl_new[["Team","GP","Expected_PPG","Actual_PPG","PPG_Diff"]].round(3),
            use_container_width=True, height=380
        )

        # Download buttons
        dl1, dl2, dl3, dl4 = st.columns(4)

        buf_xl2 = io.BytesIO()
        result_new["fig_xladder"].savefig(buf_xl2, format="png", dpi=150,
            bbox_inches="tight", facecolor=BRAND["bg"]); buf_xl2.seek(0)
        dl1.download_button("xLadder PNG", buf_xl2.getvalue(),
            f"{league}_{season}_R{next_round_num}_xladder.png","image/png")

        buf_mg2 = io.BytesIO()
        result_new["fig_margin"].savefig(buf_mg2, format="png", dpi=150,
            bbox_inches="tight", facecolor=BRAND["bg"]); buf_mg2.seek(0)
        dl2.download_button("Margin PNG", buf_mg2.getvalue(),
            f"{league}_{season}_R{next_round_num}_margin.png","image/png")

        nxt2 = result_new["next_round"]
        if len(nxt2) > 0:
            dl3.download_button("Next Round CSV",
                nxt2.to_csv(index=False).encode(),
                f"{league}_{season}_R{result_new['next_round_n']}_preds.csv","text/csv")

        # ── Updated master xlsx ────────────────────────────────────────
        updated_df = result_new["df"]
        keep_cols  = [c for c in updated_df.columns
                      if not c.startswith("Diff_Form")
                      and not c.startswith("A_Form")
                      and not c.startswith("B_Form")]
        buf_master = io.BytesIO()
        updated_df[keep_cols].to_excel(buf_master, index=False)
        buf_master.seek(0)

        dl4.download_button(
            f"Updated Master xlsx",
            buf_master.getvalue(),
            master_path.name,   # same filename -> easy GitHub replace
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.markdown(f"""<div class="info-box">
<b>To update the app permanently:</b><br>
1. Download the <b>Updated Master xlsx</b> above<br>
2. Go to your GitHub repo<br>
3. Click on <code>{master_path.name}</code> → Edit → Upload file<br>
4. Replace with the downloaded file → Commit<br>
5. Done — the app reloads automatically within 1 minute
</div>""", unsafe_allow_html=True)
        plt.close("all")

# ════════════════ TAB 5: MATCH LOG ═══════════════════════════════════════════
with tab5:
    st.markdown("#### All Played Matches with Predictions")
    if n_played == 0:
        st.info("No played matches found for this season.")
    else:
        log = played_df[[
            "Season","Round","A_Name","B_Name",
            "A_Points Scored","B_Points Scored","Margin",
            "WL_Pred","WL_Prob_A","Margin_Pred"
        ]].copy()
        log.columns = ["Season","Round","Home","Away","H Pts","A Pts",
                       "Actual Margin","WL Pred","P(Home)","Pred Margin"]
        log["P(Home)"]       = log["P(Home)"].apply(lambda x: f"{x:.0%}")
        log["Pred Margin"]   = log["Pred Margin"].apply(lambda x: f"{x:+.1f}")
        log["Actual Margin"] = log["Actual Margin"].apply(lambda x: f"{x:+.0f}")
        log["Correct?"]      = (
            (log["WL Pred"] == 1) ==
            (log["Actual Margin"].str.replace("+","",regex=False).astype(float) > 0)
        ).map({True:"Yes", False:"No"})

        st.dataframe(log, use_container_width=True, height=550)
        correct = (log["Correct?"] == "Yes").sum()
        st.markdown(f"**Model accuracy:** {correct}/{len(log)} = {correct/len(log):.1%}")
        st.download_button("Download Full Log CSV",
            log.to_csv(index=False).encode(),
            f"{league}_{season}_match_log.csv","text/csv")
