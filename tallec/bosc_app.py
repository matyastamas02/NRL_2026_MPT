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
[data-testid="stAppViewContainer"] { background-color: var(--bg); }
.stMetric { background-color: var(--panel); border-radius: 8px; padding: 12px; }
.stTabs [data-baseweb="tab"] { color: var(--text-dim); }
.stTabs [data-baseweb="tab-list"] { border-color: var(--panel); }
h1 { color: var(--text); }
h2 { color: var(--text); margin-top: 24px; }
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

@st.cache_data
def load_all_players():
    """Load all players from DB with their ratings."""
    query = """
    SELECT p.player_id, p.name, p.teams, p.positions, p.matches, p.total_minutes,
           COALESCE(r.form_score, 50) as form_score,
           COALESCE(r.class_score, 50) as class_score,
           COALESCE(r.positional_benchmark, 50) as benchmark_score,
           COALESCE(r.divergence, 0) as divergence,
           COALESCE(r.confidence, 'low') as confidence,
           COALESCE(r.shrinkage_B, 0) as shrinkage_B,
           COALESCE(r.n_games, 0) as n_games
    FROM players p
    LEFT JOIN player_ratings r ON p.player_id = r.player_id
    ORDER BY p.name
    """
    return pd.read_sql(query, con)

@st.cache_data
def get_player_stats(player_id):
    """Get raw stats for a player (all matches)."""
    query = """
    SELECT season, round, team, opposition, minutes, all_run_metres, tackles,
           tackle_breaks, tries, fantasy, p_c_m, errors
    FROM player_match_stats
    WHERE player_id = ?
    ORDER BY season DESC, round DESC
    """
    return pd.read_sql(query, con, params=(player_id,))

@st.cache_data
def get_position_peers(position, limit=10):
    """Get top 10 peers in same position by form score."""
    query = """
    SELECT p.player_id, p.name, p.teams,
           COALESCE(r.form_score, 50) as form_score,
           COALESCE(r.class_score, 50) as class_score,
           p.total_minutes
    FROM players p
    LEFT JOIN player_ratings r ON p.player_id = r.player_id
    WHERE p.positions LIKE ?
    ORDER BY COALESCE(r.form_score, 50) DESC
    LIMIT ?
    """
    return pd.read_sql(query, con, params=(f"%{position}%", limit))

# ─── Header ────────────────────────────────────────────────────────────
st.markdown("# 🏉 BOSC")
st.markdown("*Player Intelligence — Recruitment Analytics for Super League*")
st.divider()

# ─── Navigation ────────────────────────────────────────────────────────
page = st.selectbox(
    "Select section:",
    ["🔍 Search", "📊 Benchmarks", "🔄 Comparison", "📈 Trends"],
    label_visibility="collapsed"
)

# ─── PAGE 1: Search & Profile ──────────────────────────────────────────
if page == "🔍 Search":
    col1, col2 = st.columns([2, 1])

    all_players = load_all_players()

    with col1:
        search_term = st.text_input("Search player name...")
    with col2:
        position_filter = st.selectbox("Position:", ["All"] + sorted(all_players["positions"].dropna().unique().tolist()))

    # Filter
    filtered = all_players.copy()
    if search_term:
        filtered = filtered[filtered["name"].str.contains(search_term, case=False, na=False)]
    if position_filter != "All":
        filtered = filtered[filtered["positions"].str.contains(position_filter, na=False)]

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
            st.metric("Position", player["positions"].split(";")[0].strip() if player["positions"] else "—")
            st.metric("Matches", int(player["matches"]))
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
        stats = get_player_stats(player_id)
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
    else:
        st.warning("No players found. Try a different search.")

# ─── PAGE 2: Positional Benchmarks ─────────────────────────────────────
elif page == "📊 Benchmarks":
    st.subheader("Positional Benchmarks")
    st.write("Top performers by position, ranked by Form score.")

    all_players = load_all_players()

    # Get unique positions
    positions_list = []
    for pos_str in all_players["positions"].dropna():
        for p in pos_str.split(";"):
            p_clean = p.strip()
            if p_clean and p_clean not in positions_list:
                positions_list.append(p_clean)

    position_tabs = st.tabs(sorted(positions_list)[:6])  # First 6 positions

    for idx, position in enumerate(sorted(positions_list)[:6]):
        with position_tabs[idx]:
            peers = get_position_peers(position, limit=10)
            if not peers.empty:
                st.write(f"**Top 10 {position}s by Form**")
                # Rename for display
                display_df = peers[["name", "teams", "form_score", "class_score", "total_minutes"]].copy()
                display_df.columns = ["Player", "Team", "Form", "Class", "Minutes"]
                display_df["Team"] = display_df["Team"].str.split(";").str[0].str.strip()
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Scatter: Form vs Class
                st.write(f"**Form vs Class Distribution**")
                scatter_data = peers.copy()
                scatter_data["Size"] = scatter_data["total_minutes"] / 10  # Scale for bubble size
                st.scatter_chart(scatter_data, x="form_score", y="class_score", size="Size", height=300)
            else:
                st.info(f"No players found for {position}")

# ─── PAGE 3: NRL ↔ SL Comparison ───────────────────────────────────────
elif page == "🔄 Comparison":
    st.subheader("NRL ↔ SL Competition Translation")
    st.write("Estimate how an NRL player would perform in Super League.")

    all_players = load_all_players()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**NRL Player**")
        nrl_player_name = st.selectbox("Select NRL player:", all_players["name"].values, key="nrl_search")
        nrl_player = all_players[all_players["name"] == nrl_player_name].iloc[0]

        st.metric("Current Form", f"{nrl_player['form_score']:.0f}")
        st.metric("Class (Strength)", f"{nrl_player['class_score']:.0f}")
        st.metric("Position", nrl_player["positions"].split(";")[0].strip() if nrl_player["positions"] else "—")

    with col2:
        st.write("**SL Equivalent (Model Estimate)**")
        position = nrl_player["positions"].split(";")[0].strip() if nrl_player["positions"] else "Fullback"

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
        st.metric("Position", position)
        st.metric("Confidence Band", f"±{prediction['confidence_band'] * 12:.0f} pts")

    st.divider()
    st.info(f"**Model Estimate:** {nrl_player_name} would rate **{predicted_form_score:.0f}/100** as {position} in SL.\n\n"
            f"{prediction['interpretation']}")

    # Similar SL players
    st.subheader("Most Similar SL Players (Current Comparable Level)")
    similar = all_players[
        (all_players["positions"].str.contains(position, na=False))
    ].nlargest(5, "form_score")[["name", "teams", "form_score", "class_score"]]
    st.dataframe(similar, use_container_width=True, hide_index=True)

# ─── PAGE 4: Trends ────────────────────────────────────────────────────
elif page == "📈 Trends":
    st.subheader("Form & Class Trends (Coming Soon)")
    st.info("This section shows how player ratings change over time. Data available after first full week of Stats Perform imports.")

st.divider()
st.caption("BOSC MVP — Phase 1 prototype. Real ratings available after Stats Perform data ingestion.")
