# -*- coding: utf-8 -*-
"""
BOSC — Player Intelligence Dashboard for Rugby League Recruitment
MVP: Search, Benchmarks, Comparison, Trends tabs.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from predict_translation import predict_translation

st.set_page_config(
    page_title="BOSC — Player Intelligence",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Styling ───────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --bg: #0E1526;
  --panel: #18213A;
  --accent: #4C8DFF;
  --text: #EDF1F9;
  --text-dim: #9AA8C7;
}
[data-testid="stAppViewContainer"], .stApp { background-color: var(--bg); }
/* force readable text everywhere (Cloud default base is light -> dark text on
   our dark bg was invisible) */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stMarkdown, h1, h2, h3, h4, h5, h6 { color: var(--text); }
.stCaption, [data-testid="stCaptionContainer"], small { color: var(--text-dim) !important; }
.stMetric { background-color: var(--panel); border-radius: 8px; padding: 12px 14px; }
[data-testid="stMetricValue"] { color: var(--text) !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: var(--text-dim) !important; }
[data-testid="stMetricDelta"] { color: var(--text-dim) !important; }
/* dataframes / tables */
.stDataFrame, .stDataFrame * { color: var(--text); }
.stTabs [data-baseweb="tab"] { color: var(--text-dim); }
.stTabs [data-baseweb="tab-list"] { border-color: var(--panel); }
</style>
""", unsafe_allow_html=True)

# ─── Database & Cache ──────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "tallec.db"

@st.cache_resource
def get_db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

con = get_db()

SEASON = 2026

@st.cache_data
def load_all_players(comp):
    """Players active in `comp` this season, joined to their ratings."""
    query = """
    SELECT r.player_id, p.name, p.teams, p.total_minutes,
           r.form_score, r.class_score,
           r.positional_benchmark as benchmark_score,
           r.divergence, r.confidence, r.shrinkage_B, r.n_games
    FROM player_ratings r
    JOIN players p ON p.player_id = r.player_id
    WHERE r.competition = ? AND r.season = ?
    ORDER BY p.name
    """
    df = pd.read_sql(query, con, params=(comp, SEASON))
    df["positions"] = "Unknown"   # no position field in the Stats Perform feed yet
    return df

@st.cache_data
def get_player_stats(player_id, comp):
    """Get raw match stats for a player in a competition (all seasons held)."""
    query = """
    SELECT season, round, team, opposition, minutes, all_run_metres, tackles,
           tackle_breaks, tries, fantasy, p_c_m, errors
    FROM player_match_stats
    WHERE player_id = ? AND competition = ?
    ORDER BY season DESC, round DESC
    """
    return pd.read_sql(query, con, params=(player_id, comp))

# Radar axes: (label, per-what) — computed as percentile ranks across qualified players
RADAR_METRICS = [
    ("Run m / min",      "run_pm"),
    ("PCM / min",        "pcm_pm"),
    ("Tackles / min",    "tck_pm"),
    ("Tackle breaks /g", "tb_pg"),
    ("Attack punch /g",  "punch_pg"),   # line breaks + tries per game
    ("Fantasy / min",    "fant_pm"),
]

@st.cache_data
def load_player_metrics(comp, min_minutes=40):
    """Aggregate per-player rate metrics + their percentile ranks (0-100)."""
    q = """
    SELECT player_id, player as name, team, position, minutes,
           all_run_metres, p_c_m, tackles, tackle_breaks, line_breaks, tries, fantasy
    FROM player_match_stats
    WHERE competition = ? AND season = ?
    """
    pm = pd.read_sql(q, con, params=(comp, SEASON))
    for c in ["all_run_metres","p_c_m","tackles","tackle_breaks","line_breaks","tries","fantasy"]:
        pm[c] = pm[c].fillna(0)
    agg = pm.groupby("player_id").agg(
        name=("name","first"), team=("team","first"), position=("position","first"),
        mins=("minutes","sum"), games=("player_id","count"),
        run=("all_run_metres","sum"), pcm=("p_c_m","sum"), tck=("tackles","sum"),
        tb=("tackle_breaks","sum"), lb=("line_breaks","sum"),
        tries=("tries","sum"), fant=("fantasy","sum")).reset_index()
    agg = agg[agg["mins"] >= min_minutes].copy()
    agg["run_pm"]   = agg["run"]  / agg["mins"]
    agg["pcm_pm"]   = agg["pcm"]  / agg["mins"]
    agg["tck_pm"]   = agg["tck"]  / agg["mins"]
    agg["tb_pg"]    = agg["tb"]   / agg["games"]
    agg["punch_pg"] = (agg["lb"] + agg["tries"]) / agg["games"]
    agg["fant_pm"]  = agg["fant"] / agg["mins"]
    for _, key in RADAR_METRICS:
        agg[f"pct_{key}"] = agg[key].rank(pct=True) * 100
    return agg

@st.cache_data
def load_contribution(comp):
    """GIGOT contribution ratings (player share of own team's output)."""
    try:
        return pd.read_sql(
            "SELECT * FROM player_contribution_rating WHERE competition = ? "
            "ORDER BY contribution_rating DESC", con, params=(comp,))
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_round_composites(comp):
    """Per-match composite performance scores from the rating engine, this season."""
    import player_rating_engine as pre
    cols = ["player_id", "player", "season", "round", "team", "position", "minutes",
            "all_run_metres", "p_c_m", "tackle_breaks", "line_breaks", "tackles",
            "offloads", "try_assists", "tries", "errors"]
    raw = pd.read_sql(
        f"SELECT {', '.join(cols)} FROM player_match_stats "
        f"WHERE competition = ? AND season = ?", con, params=(comp, SEASON))
    eng = pre.PlayerRatingEngine(comp)
    pm = eng._composite(raw)
    return pm[["player_id","player","team","position","season","round",
               "minutes","composite","ratable"]]

if "shortlist" not in st.session_state:
    st.session_state["shortlist"] = []

# ─── Header ────────────────────────────────────────────────────────────
st.markdown("# 🏉 BOSC")
st.markdown("*Player Intelligence — Recruitment Analytics for Super League*")
st.divider()

# ─── League + Navigation ───────────────────────────────────────────────
nav1, nav2 = st.columns([1, 3])
with nav1:
    comp = st.selectbox("League:", ["SL", "NRL"],
                        format_func=lambda c: {"SL": "🇬🇧 Super League",
                                               "NRL": "🇦🇺 NRL"}[c])
with nav2:
    page = st.selectbox(
        "Select section:",
        ["🔍 Search", "⚖️ Compare", "📊 Benchmarks", "🔄 Comparison",
         "🏉 Squad (GIGOT)", "📈 Trends"],
        label_visibility="collapsed")
st.caption(f"{'Super League' if comp=='SL' else 'NRL'} · 2026 season · ratings are "
           f"competition-relative (position data pending from provider).")

# ─── PAGE 1: Search & Profile ──────────────────────────────────────────
if page == "🔍 Search":
    col1, col2 = st.columns([2, 1])

    all_players = load_all_players(comp)

    with col1:
        search_term = st.text_input("Search player name...")
    with col2:
        team_filter = st.selectbox(
            "Team:", ["All"] + sorted(
                {t.split(";")[0].strip() for t in all_players["teams"].dropna()}))

    # Filter
    filtered = all_players.copy()
    if search_term:
        filtered = filtered[filtered["name"].str.contains(search_term, case=False, na=False)]
    if team_filter != "All":
        filtered = filtered[filtered["teams"].str.contains(team_filter, na=False)]

    if not filtered.empty:
        # Player picker
        selected_name = st.selectbox("Select player:", filtered["name"].values)
        player = filtered[filtered["name"] == selected_name].iloc[0]
        player_id = player["player_id"]

        # Profile card
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("Player", player["name"])
            st.metric("Team", player["teams"].split(";")[0].strip() if player["teams"] else "—")
        with col2:
            st.metric("Games rated", int(player["n_games"]))
        with col3:
            st.metric("Minutes Played", int(player["total_minutes"]))

        st.divider()

        # Ratings
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Form Score", f"{player['form_score']:.0f}/100")
        with col2:
            st.metric("Class Score", f"{player['class_score']:.0f}/100")
        with col3:
            st.metric("Divergence", f"{player['divergence']:+.2f}",
                      help="Form minus Class (z-scores). Positive = recent form above structural level.")
        with col4:
            conf = str(player["confidence"]).upper()
            icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(conf, "⚪")
            st.metric("Confidence", f"{icon} {conf}",
                      help="How much the raw stats are trusted vs the positional average.")

        # Honesty note: shrinkage on small samples
        B = float(player["shrinkage_B"])
        ng = int(player["n_games"])
        if ng > 0:
            st.caption(
                f"Rated on **{ng} game{'s' if ng != 1 else ''}**. "
                f"Bayesian shrinkage keeps **{B*100:.0f}%** of the raw signal and "
                f"pulls the rest toward the positional average — with this little "
                f"data the model is deliberately cautious. Ratings sharpen as more "
                f"rounds arrive.")
        else:
            st.caption("No ratable minutes yet — rating defaults to the positional average (50).")

        st.divider()

        # Raw stats table
        st.subheader("Match Statistics (5-Match & Career)")
        stats = get_player_stats(player_id, comp)
        if not stats.empty:
            # Recent 5-game summary
            recent = stats.head(5)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_min = recent["minutes"].mean()
                st.metric("Avg Minutes (5g)", f"{avg_min:.1f}")
            with col2:
                avg_run = recent["all_run_metres"].mean()
                st.metric("Avg Run Metres (5g)", f"{avg_run:.1f}")
            with col3:
                avg_tck = recent["tackles"].mean()
                st.metric("Avg Tackles (5g)", f"{avg_tck:.1f}")
            with col4:
                total_tries = stats["tries"].sum()
                st.metric("Career Tries", int(total_tries))

            st.write("**All Matches**")
            display_stats = stats[["season", "round", "team", "opposition", "minutes", "all_run_metres", "tackles", "tries"]].copy()
            for c in ["minutes", "all_run_metres", "tackles", "tries"]:
                display_stats[c] = display_stats[c].fillna(0).astype(int)
            st.dataframe(display_stats, use_container_width=True)
        else:
            st.info("No match stats available for this player.")

        # Shortlist + export
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            if selected_name in st.session_state["shortlist"]:
                if st.button("➖ Remove from shortlist"):
                    st.session_state["shortlist"].remove(selected_name); st.rerun()
            else:
                if st.button("➕ Add to shortlist"):
                    st.session_state["shortlist"].append(selected_name); st.rerun()
        with c2:
            if not stats.empty:
                st.download_button("⬇️ Export matches (CSV)",
                                   stats.to_csv(index=False).encode(),
                                   file_name=f"{player_id}_matches.csv", mime="text/csv")

        if st.session_state["shortlist"]:
            with st.expander(f"📋 Shortlist ({len(st.session_state['shortlist'])})", expanded=False):
                sl = all_players[all_players["name"].isin(st.session_state["shortlist"])].copy()
                sl["team"] = sl["teams"].str.split(";").str[0].str.strip()
                sl_view = sl[["name","team","n_games","form_score",
                              "class_score","confidence"]].copy()
                sl_view.columns = ["Player","Team","GP","Form","Class","Conf"]
                st.dataframe(sl_view.round(0), use_container_width=True, hide_index=True)
                st.download_button("⬇️ Export shortlist (CSV)",
                                   sl_view.to_csv(index=False).encode(),
                                   file_name="bosc_shortlist.csv", mime="text/csv")
    else:
        st.warning("No players found. Try a different search.")

# ─── PAGE: Head-to-Head Compare ────────────────────────────────────────
elif page == "⚖️ Compare":
    st.subheader("Head-to-Head Comparison")
    st.write("Two players side by side. Radar axes are **percentile ranks** across "
             "all players with ≥ 40 minutes — 100 = best in the database.")

    metrics = load_player_metrics(comp)
    all_players = load_all_players(comp)
    names = metrics.sort_values("name")["name"].tolist()

    default_b = 1 if len(names) > 1 else 0
    c1, c2 = st.columns(2)
    with c1:
        name_a = st.selectbox("Player A", names, index=0)
    with c2:
        name_b = st.selectbox("Player B", names, index=default_b)

    ma = metrics[metrics["name"] == name_a].iloc[0]
    mb = metrics[metrics["name"] == name_b].iloc[0]

    # rating cards
    ra = all_players[all_players["name"] == name_a]
    rb = all_players[all_players["name"] == name_b]
    c1, c2 = st.columns(2)
    for col, m, r in [(c1, ma, ra), (c2, mb, rb)]:
        with col:
            st.markdown(f"**{m['name']}** — {m['team']} · {m['position']}")
            k1, k2, k3 = st.columns(3)
            form = float(r["form_score"].iloc[0]) if len(r) else 50
            cls  = float(r["class_score"].iloc[0]) if len(r) else 50
            k1.metric("Form", f"{form:.0f}")
            k2.metric("Class", f"{cls:.0f}")
            k3.metric("Minutes", int(m["mins"]))

    # radar
    import plotly.graph_objects as go
    labels = [lab for lab, _ in RADAR_METRICS]
    va = [float(ma[f"pct_{k}"]) for _, k in RADAR_METRICS]
    vb = [float(mb[f"pct_{k}"]) for _, k in RADAR_METRICS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=va + va[:1], theta=labels + labels[:1],
                                  fill="toself", name=name_a,
                                  line=dict(color="#4C8DFF")))
    fig.add_trace(go.Scatterpolar(r=vb + vb[:1], theta=labels + labels[:1],
                                  fill="toself", name=name_b,
                                  line=dict(color="#F5A524")))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(range=[0, 100], showticklabels=True,
                                   gridcolor="#2A3554", tickfont=dict(color="#9AA8C7")),
                   angularaxis=dict(gridcolor="#2A3554", tickfont=dict(color="#EDF1F9"))),
        paper_bgcolor="rgba(0,0,0,0)", legend=dict(font=dict(color="#EDF1F9")),
        height=420, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # numeric table
    rows = []
    for lab, k in RADAR_METRICS:
        rows.append({"Metric": lab,
                     name_a: round(float(ma[k]), 2),
                     f"{name_a} pct": round(float(ma[f'pct_{k}'])),
                     name_b: round(float(mb[k]), 2),
                     f"{name_b} pct": round(float(mb[f'pct_{k}']))})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Rates are raw per-minute / per-game values; 'pct' columns are the "
               "percentile ranks plotted on the radar. Same small-sample caveat as "
               "everywhere: 2 rounds of data.")

# ─── PAGE 2: Positional Benchmarks ─────────────────────────────────────
elif page == "📊 Benchmarks":
    st.subheader("Benchmarks")
    st.info("Position data isn't in the current provider feed, so these are "
            "**competition-wide** benchmarks (0–100, 50 = league median). "
            "Once a position source is supplied these split into positional "
            "benchmarks — the intended view.")

    all_players = load_all_players(comp)
    all_players["team"] = all_players["teams"].str.split(";").str[0].str.strip()

    tab_lead, tab_team, tab_dist = st.tabs(
        ["League leaders", "By team", "Form vs Class"])

    with tab_lead:
        st.write("**Top 15 by Form**")
        lead = all_players.nlargest(15, "form_score")[
            ["name", "team", "form_score", "class_score", "confidence"]].copy()
        lead.columns = ["Player", "Team", "Form", "Class", "Conf"]
        st.dataframe(lead.round(0), use_container_width=True, hide_index=True)

    with tab_team:
        team = st.selectbox("Team:", sorted(all_players["team"].dropna().unique()))
        sq = all_players[all_players["team"] == team].nlargest(20, "form_score")[
            ["name", "form_score", "class_score", "divergence", "confidence"]].copy()
        sq.columns = ["Player", "Form", "Class", "Diverg", "Conf"]
        st.dataframe(sq.round(2), use_container_width=True, hide_index=True)

    with tab_dist:
        st.write("**Form vs Class** — bubble size = minutes played")
        sc = all_players.copy()
        sc["Size"] = sc["total_minutes"].fillna(0) / 10
        st.scatter_chart(sc, x="form_score", y="class_score", size="Size", height=380)

# ─── PAGE 3: NRL ↔ SL Comparison ───────────────────────────────────────
elif page == "🔄 Comparison":
    st.subheader("NRL → SL Competition Translation")
    st.write("Estimate how an NRL player would perform in Super League. "
             "(Independent of the league selector — source is always NRL, target SL.)")

    nrl_players = load_all_players("NRL")
    sl_players = load_all_players("SL")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**NRL Player**")
        nrl_player_name = st.selectbox("Select NRL player:", nrl_players["name"].values, key="nrl_search")
        nrl_player = nrl_players[nrl_players["name"] == nrl_player_name].iloc[0]

        st.metric("Current Form", f"{nrl_player['form_score']:.0f}")
        st.metric("Class (Strength)", f"{nrl_player['class_score']:.0f}")

    with col2:
        st.write("**SL Equivalent (Model Estimate)**")
        position = "Fullback"   # position source pending; model uses a neutral prior

        # Real translation model prediction
        # Estimate form_z and class_z from 0-100 scores
        form_z = (nrl_player["form_score"] - 62) / 12
        class_z = (nrl_player["class_score"] - 60) / 8

        prediction = predict_translation(
            player_name=nrl_player_name,
            position=position,
            form_z=form_z,
            class_z=class_z,
            age=26,  # Mock (could extract from player data)
            games_per_season=17,
            injury_rate=0.1
        )

        # Convert z-scores back to 0-100 scale
        predicted_form_score = 62 + prediction["predicted_form_z"] * 12
        predicted_form_score = max(0, min(100, predicted_form_score))

        translation_delta = prediction["translation_factor"] * 12  # Scale to 0-100

        st.metric("Predicted Form (SL)", f"{predicted_form_score:.0f}", delta=f"{translation_delta:.0f}")
        st.metric("Confidence Band", f"±{prediction['confidence_band'] * 12:.0f} pts")

    st.divider()
    st.info(f"**Model Estimate:** {nrl_player_name} would rate **{predicted_form_score:.0f}/100** in SL.\n\n"
            f"{prediction['interpretation']}")

    # Closest SL players by rating level
    st.subheader("SL players at a comparable level")
    sl_players["gap"] = (sl_players["form_score"] - predicted_form_score).abs()
    similar = sl_players.nsmallest(5, "gap")[["name", "teams", "form_score", "class_score"]].copy()
    similar["teams"] = similar["teams"].str.split(";").str[0].str.strip()
    similar.columns = ["Player", "Team", "Form", "Class"]
    st.dataframe(similar.round(0), use_container_width=True, hide_index=True)

# ─── PAGE: Squad & Contribution (GIGOT) ────────────────────────────────
elif page == "🏉 Squad (GIGOT)":
    st.subheader("Squad Contribution — GIGOT input #5")
    st.write("**Contribution Rating** = a player's share of his own team's output "
             "(attack 45% · defence 35% · points 20%), percentile-scaled so the "
             "median contributor = 50. This is real data, not a mock.")

    contrib = load_contribution(comp)
    if contrib.empty:
        st.warning("No contribution data — run gigot_contribution.py first.")
    else:
        teams = sorted(contrib["team"].dropna().unique())
        team = st.selectbox("Team:", teams)
        squad = contrib[contrib["team"] == team].sort_values(
            "contribution_rating", ascending=False).reset_index(drop=True)

        # what-if team list simulator
        st.markdown("**Team-list simulator** — untick players to see the expected "
                    "contribution loss (the *Player Availability* signal that feeds "
                    "the GIGOT match model).")
        available = st.multiselect("Available players:", squad["name"].tolist(),
                                   default=squad["name"].tolist())
        full = squad["contribution_rating"].sum()
        got = squad[squad["name"].isin(available)]["contribution_rating"].sum()
        missing = squad[~squad["name"].isin(available)]

        k1, k2, k3 = st.columns(3)
        k1.metric("Full-squad expected contribution", f"{full:.0f}")
        k2.metric("Selected line-up", f"{got:.0f}", delta=f"{got-full:+.0f}")
        pct = (got - full) / full * 100 if full else 0
        k3.metric("Impact", f"{pct:+.1f}%")
        if len(missing):
            st.caption("Missing: " + ", ".join(
                f"{r['name']} ({r['contribution_rating']:.0f})"
                for _, r in missing.iterrows()))

        st.divider()
        view = squad[["name","matches","minutes","attack_share","defence_share",
                      "contribution_rating"]].copy()
        view["attack_share"] = (view["attack_share"]*100).round(1)
        view["defence_share"] = (view["defence_share"]*100).round(1)
        view["contribution_rating"] = view["contribution_rating"].round(0)
        view.columns = ["Player","GP","Min","Attack %","Defence %","Contribution"]
        st.dataframe(view, use_container_width=True, hide_index=True)

        st.divider()
        st.write("**League-wide top contributors**")
        top = contrib.head(10)[["name","team","matches","contribution_rating"]].copy()
        top["contribution_rating"] = top["contribution_rating"].round(1)
        top.columns = ["Player","Team","GP","Contribution"]
        st.dataframe(top, use_container_width=True, hide_index=True)

# ─── PAGE 4: Trends ────────────────────────────────────────────────────
elif page == "📈 Trends":
    st.subheader("Round-to-Round Movement")
    st.write("Composite performance score per match (position-relative, from the "
             "rating engine) — who moved up or down between the two rounds we have.")

    pm = load_round_composites(comp)
    rated = pm[pm["ratable"]]
    rounds = sorted(rated["round"].unique())
    if len(rounds) < 2:
        st.info("Need at least two rounds of player data for movement.")
    else:
        r_prev, r_last = rounds[-2], rounds[-1]
        piv = rated.pivot_table(index=["player_id","player","team"],
                                columns="round", values="composite").reset_index()
        both = piv.dropna(subset=[r_prev, r_last]).copy()
        both["delta"] = both[r_last] - both[r_prev]
        both = both.sort_values("delta", ascending=False)

        st.caption(f"{len(both)} players featured in both R{r_prev} and R{r_last} "
                   f"with ≥ 20 minutes. One-round movement is noisy — treat as a "
                   f"watch-list, not a verdict; with a full season this becomes a "
                   f"proper Form trend line.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📈 Biggest risers**")
            up = both.head(10)[["player","team",r_prev,r_last,"delta"]].copy()
            up.columns = ["Player","Team",f"R{r_prev}",f"R{r_last}","Δ"]
            st.dataframe(up.round(2), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**📉 Biggest fallers**")
            down = both.tail(10).iloc[::-1][["player","team",r_prev,r_last,"delta"]].copy()
            down.columns = ["Player","Team",f"R{r_prev}",f"R{r_last}","Δ"]
            st.dataframe(down.round(2), use_container_width=True, hide_index=True)

        # distribution of deltas
        st.write("**Movement distribution** (composite z-score change)")
        hist = np.histogram(both["delta"], bins=20)
        hist_df = pd.DataFrame({"count": hist[0]},
                               index=[f"{e:.1f}" for e in hist[1][:-1]])
        st.bar_chart(hist_df)

st.divider()
st.caption("BOSC MVP — Phase 1 prototype. Real ratings available after Stats Perform data ingestion.")
