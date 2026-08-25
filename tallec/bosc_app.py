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
from predict_translation import translate, available_pairs, COMP_NAME
from sp_schema import POSITION_GROUP
import runtime

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

# The competitions do not share a current season — NSW Cup and Queensland Cup data
# stops at 2025 while NRL and Super League run to 2026 — so every competition-scoped
# query resolves its own. SEASON stays as the default for anything unscoped.
SEASON = 2026


@st.cache_data
def season_of(comp):
    r = pd.read_sql("SELECT max(season) s FROM player_ratings WHERE competition = ?",
                    con, params=(comp,))
    if len(r) and pd.notna(r.s[0]):
        return int(r.s[0])
    r = pd.read_sql("SELECT max(season) s FROM player_match_stats WHERE competition = ?",
                    con, params=(comp,))
    return int(r.s[0]) if len(r) and pd.notna(r.s[0]) else SEASON

@st.cache_data
def load_all_players(comp):
    """Players active in `comp` this season, joined to their ratings."""
    query = """
    SELECT r.player_id, p.name, p.teams, p.total_minutes, p.positions,
           r.form_score, r.class_score,
           r.positional_benchmark as benchmark_score,
           r.divergence, r.confidence, r.shrinkage_B, r.n_games, r.rating_basis
    FROM player_ratings r
    JOIN players p ON p.player_id = r.player_id
    WHERE r.competition = ? AND r.season = ?
    ORDER BY p.name
    """
    return pd.read_sql(query, con, params=(comp, season_of(comp)))

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
    pm = pd.read_sql(q, con, params=(comp, season_of(comp)))
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
def load_player_meta(comp):
    """Age (from date of birth) and most-played position, per player this season."""
    # date of birth is read from the small players registry. It used to come from a
    # correlated subquery over player_match_raw — 122k rows across 342 columns — which
    # made this page take minutes rather than seconds.
    q = """
    SELECT s.player_id, s.position, s.position_source, count(*) AS n_at_pos,
           sum(s.minutes) AS mins, p.dob AS dob
    FROM player_match_stats s
    JOIN players p ON p.player_id = s.player_id
    WHERE s.competition = ? AND s.season = ?
    GROUP BY s.player_id, s.position, s.position_source, p.dob
    """
    season = season_of(comp)
    d = pd.read_sql(q, con, params=(comp, season))
    if d.empty:
        return pd.DataFrame(columns=["player_id", "position", "pos_group",
                                     "position_source", "mins_pg", "games", "age"])
    tot = d.groupby("player_id").agg(games=("n_at_pos", "sum"), mins=("mins", "sum"),
                                    dob=("dob", "first")).reset_index()
    top = d.sort_values("n_at_pos", ascending=False).drop_duplicates("player_id")
    out = tot.merge(top[["player_id", "position", "position_source"]], on="player_id")
    out["mins_pg"] = out.mins / out.games
    dob = pd.to_datetime(out["dob"], errors="coerce")
    out["age"] = ((pd.Timestamp(f"{season}-06-30") - dob).dt.days / 365.25).round(1)
    out["pos_group"] = out["position"].map(POSITION_GROUP).fillna("Unknown")
    return out[["player_id", "position", "pos_group", "position_source",
                "mins_pg", "games", "age"]]


@st.cache_data
def load_ladder():
    return pd.read_sql("SELECT * FROM translation_ladder ORDER BY n DESC", con)

@st.cache_data
def load_round_composites(comp):
    """Per-match composite performance scores from the rating engine, this season."""
    import player_rating_engine as pre
    cols = ["player_id", "player", "season", "round", "team", "position", "minutes",
            "all_run_metres", "p_c_m", "tackle_breaks", "line_breaks", "tackles",
            "offloads", "try_assists", "tries", "errors"]
    raw = pd.read_sql(
        f"SELECT {', '.join(cols)} FROM player_match_stats "
        f"WHERE competition = ? AND season = ?", con, params=(comp, season_of(comp)))
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
    comp = st.selectbox("League:", ["SL", "NRL", "NSW", "QLD"],
                        format_func=lambda c: {"SL": "🇬🇧 Super League",
                                               "NRL": "🇦🇺 NRL",
                                               "NSW": "🇦🇺 NSW Cup",
                                               "QLD": "🇦🇺 Queensland Cup"}[c])
with nav2:
    page = st.selectbox(
        "Select section:",
        ["🔍 Search", "⚖️ Compare", "📊 Benchmarks", "🔄 Comparison",
         "🏉 Squad (GIGOT)", "📈 Trends"],
        label_visibility="collapsed")
_basis = pd.read_sql("SELECT rating_basis, count(*) n FROM player_ratings "
                     "WHERE competition = ? AND season = ? GROUP BY 1 "
                     "ORDER BY n DESC LIMIT 1", con, params=(comp, season_of(comp)))
_b = _basis.rating_basis[0] if len(_basis) and pd.notna(_basis.rating_basis[0]) else "unknown"
st.caption(f"{COMP_NAME.get(comp, comp)} · {season_of(comp)} season · ratings are "
           + ("**position-relative** — each player is measured against his own "
              "position group." if _b == "position_relative" else
              "**competition-relative** — each player is measured against the whole "
              "competition, because position is known for too little of this pool."))

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
            st.metric("Position", player.get("positions", "Unknown") or "Unknown")
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
                      help="How much of the player's own numbers the rating keeps, "
                           "versus falling back on the average for his peer group.")

        # Honesty note: shrinkage on small samples
        B = float(player["shrinkage_B"])
        ng = int(player["n_games"])
        if ng > 0:
            st.caption(
                f"Rated on **{ng} game{'s' if ng != 1 else ''}**. "
                f"Bayesian shrinkage keeps **{B*100:.0f}%** of the raw signal and "
                f"pulls the rest toward the "
                f"{'positional' if _b == 'position_relative' else 'competition'} "
                f"average — with this little data the model is deliberately cautious. "
                f"Ratings sharpen as more rounds arrive.")
        else:
            st.caption(f"No ratable minutes yet — rating defaults to the "
                       f"{'positional' if _b == 'position_relative' else 'competition'} "
                       f"average (50).")

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
            st.dataframe(display_stats, width="stretch")
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
                st.dataframe(sl_view.round(0), width="stretch", hide_index=True)
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
    st.plotly_chart(fig, width="stretch")

    # numeric table
    rows = []
    for lab, k in RADAR_METRICS:
        rows.append({"Metric": lab,
                     name_a: round(float(ma[k]), 2),
                     f"{name_a} pct": round(float(ma[f'pct_{k}'])),
                     name_b: round(float(mb[k]), 2),
                     f"{name_b} pct": round(float(mb[f'pct_{k}']))})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Rates are raw per-minute / per-game values; 'pct' columns are the "
               "percentile ranks plotted on the radar. Same small-sample caveat as "
               "everywhere: 2 rounds of data.")

# ─── PAGE 2: Positional Benchmarks ─────────────────────────────────────
elif page == "📊 Benchmarks":
    st.subheader("Benchmarks")

    all_players = load_all_players(comp)
    if all_players.empty:
        st.warning(f"No {COMP_NAME.get(comp, comp)} ratings held for {season_of(comp)}.")
    else:
        all_players["team"] = all_players["teams"].str.split(";").str[0].str.strip()
        meta = load_player_meta(comp)
        pool_all = all_players.merge(
            meta[["player_id", "position", "pos_group", "position_source", "age"]],
            on="player_id", how="left")
        pool_all["pos_group"] = pool_all["pos_group"].fillna("Unknown")

        basis = (all_players["rating_basis"].dropna().iloc[0]
                 if all_players["rating_basis"].notna().any() else "unknown")
        known = (pool_all["pos_group"] != "Unknown").mean()
        real_pos = (pool_all["position_source"] == "match").mean()

        # ── how the ratings on this page were actually built ─────────────────
        if basis == "position_relative":
            st.success(f"**Rated within position group.** Position is known for "
                       f"{known:.0%} of the rated {COMP_NAME.get(comp, comp)} players, so each "
                       f"player is measured against his own position — a prop against "
                       f"props, a hooker against hookers. 50 means average *for that "
                       f"position*.")
        else:
            st.warning(f"**Rated against the whole competition, not by position.** "
                       f"Position is known for only {known:.0%} of the rated "
                       f"{COMP_NAME.get(comp, comp)} players. Splitting a partly-known pool "
                       f"into position groups would make the ratings incomparable "
                       f"between players, so 50 means average *for the competition*. "
                       f"The position tab below groups players descriptively; it does "
                       f"not change how they were rated.")
        if real_pos == 0:
            st.caption("None of these positions come from a match sheet — the "
                       f"{season_of(comp)} feed has no position column. Each is the player's most "
                       "common starting position elsewhere in the data, matched on his "
                       "permanent player ID.")
        elif real_pos < 1:
            st.caption(f"Position comes from the match sheet for {real_pos:.0%} of these "
                       f"players; for the rest it is their most common starting position "
                       f"elsewhere in the data.")

        # ── sample-size floor, applied to every table on the page ───────────
        f1, f2 = st.columns([1, 2])
        with f1:
            min_games = st.slider("Minimum matches", 1, 15, 5, key="bm_min_games")
        pool = pool_all[pool_all["n_games"] >= min_games].copy()
        dropped = len(pool_all) - len(pool)
        with f2:
            st.caption(f"Showing **{len(pool)}** of {len(pool_all)} rated players. "
                       f"{dropped} are held back for having fewer than {min_games} "
                       f"matches — a rating on one or two games is mostly luck, and the "
                       f"engine already shrinks it hard toward average, but it can still "
                       f"top a table sorted by score.")
        if pool.empty:
            st.info("No players clear that threshold — lower it to see the pool.")
            st.stop()

        SHOW = ["name", "team", "position", "form_score", "class_score",
                "divergence", "n_games", "confidence"]
        HEAD = ["Player", "Team", "Position", "Form", "Class", "Diverg", "GP", "Conf"]

        def table(df, n=15, sort="form_score"):
            v = df.nlargest(n, sort)[SHOW].copy()
            v.columns = HEAD
            for c in ["Form", "Class", "Diverg"]:
                v[c] = v[c].round(1)
            return v

        tab_pos, tab_lead, tab_team, tab_dist = st.tabs(
            ["By position", "League leaders", "By team", "Form vs Class"])

        with tab_pos:
            groups = sorted(g for g in pool["pos_group"].unique() if g != "Unknown")
            if not groups:
                st.info(f"No position data held for {COMP_NAME.get(comp, comp)} yet.")
            else:
                gsel = st.selectbox("Position group:", groups, key="bm_group")
                grp = pool[pool["pos_group"] == gsel]
                if basis == "position_relative":
                    st.write(f"**Top {gsel} by Form** — measured against other {gsel} "
                             f"players ({len(grp)} in the pool at {min_games}+ matches)")
                else:
                    st.write(f"**Top {gsel} by Form** — {len(grp)} players at "
                             f"{min_games}+ matches. Scores are competition-relative, "
                             f"so these are the best-rated players who happen to play "
                             f"{gsel}, not a within-position ranking.")
                st.dataframe(table(grp), width="stretch", hide_index=True)
                st.caption("Position groups follow the rating engine: Second Row and "
                           "Lock are both Back Row, Half Back and Five-Eighth are both "
                           "Halves, and bench players are grouped separately.")

        with tab_lead:
            c1, c2 = st.columns([1, 1])
            with c1:
                metric = st.radio("Rank by:", ["Form", "Class"], horizontal=True,
                                  key="bm_metric")
            key = "form_score" if metric == "Form" else "class_score"
            st.write(f"**Top 15 by {metric}**")
            st.dataframe(table(pool, 15, key), width="stretch", hide_index=True)
            st.caption("Form is the recent window, Class the full season, both shrunk "
                       "toward average by sample size. Divergence is Form minus Class: "
                       "positive means playing above his own level right now.")

        with tab_team:
            teams = sorted(pool["team"].dropna().unique())
            team = st.selectbox("Team:", teams, key="bm_team")
            sq = pool[pool["team"] == team]
            st.dataframe(table(sq, 25), width="stretch", hide_index=True)
            st.caption(f"{len(sq)} players at {min_games}+ matches.")

        with tab_dist:
            st.write("**Form vs Class** — where a player sits now against his own level")
            sc = pool[["name", "form_score", "class_score", "total_minutes"]].copy()
            sc["Minutes"] = sc["total_minutes"].fillna(0)
            st.scatter_chart(sc, x="class_score", y="form_score", size="Minutes",
                             height=400)
            st.caption("Above the diagonal means current form is running ahead of "
                       "season-long class; below it means the opposite. Both axes are "
                       "0-100 with 50 as average.")

# ─── PAGE 3: Competition Translation ───────────────────────────────────
elif page == "🔄 Comparison":
    st.subheader("Competition Translation")
    st.write("How a player's rating carries across competitions. The shift for each "
             "pair is **measured**, not assumed: it comes from players who played both "
             "competitions in the same season (NSW/QLD Cup ↔ NRL) or in adjacent "
             "seasons (Australia ↔ Super League), so the player is held fixed.")

    ladder = load_ladder()
    pairs = [(r.source, r.target) for r in ladder.itertuples()]
    sources = sorted({p[0] for p in pairs})

    c1, c2 = st.columns(2)
    with c1:
        src = st.selectbox("From competition:", sources,
                           format_func=lambda c: COMP_NAME[c],
                           index=sources.index("NRL") if "NRL" in sources else 0,
                           key="tr_src")
    targets = sorted({t for s_, t in pairs if s_ == src})
    with c2:
        tgt = st.selectbox("To competition:", targets,
                           format_func=lambda c: COMP_NAME[c],
                           index=targets.index("SL") if "SL" in targets else 0,
                           key="tr_tgt")

    src_players = load_all_players(src)
    meta = load_player_meta(src)
    if src_players.empty:
        st.warning(f"No ratings held for {COMP_NAME[src]} — ratings are computed for "
                   f"the current season only.")
    else:
        pool = src_players.merge(meta, on="player_id", how="left")
        name = st.selectbox(f"{COMP_NAME[src]} player:", pool["name"].values, key="tr_player")
        p = pool[pool["name"] == name].iloc[0]

        res = translate(p["class_score"], src, tgt,
                        position=p.get("position"), age=p.get("age"),
                        minutes_pg=p.get("mins_pg"), games=p.get("games"))

        k1, k2, k3 = st.columns(3)
        k1.metric(f"Rating in {COMP_NAME[src]}", f"{p['class_score']:.0f}")
        k2.metric(f"Expected in {COMP_NAME[tgt]}", f"{res['score_target']:.0f}",
                  delta=f"{res['shift_points']:+.1f}")
        k3.metric("Player detail",
                  f"{p.get('position') or 'Unknown'}"
                  + (f", {p['age']:.0f}y" if pd.notna(p.get("age")) else ""))

        st.caption(f"{res['interpretation']} Based on **{res['n_obs']} observed player "
                   f"moves** between these two competitions.")

        st.markdown("**Two different uncertainties — worth keeping apart**")
        u1, u2 = st.columns(2)
        u1.metric("Average shift for this pair", f"{res['shift_points']:+.1f} pts",
                  delta=f"±{res['avg_band_points']:.1f} at 95%", delta_color="off")
        u2.metric("This individual player", f"{res['score_target']:.0f} pts",
                  delta=f"±{res['band_points']:.1f} at 95%", delta_color="off")
        st.caption("The average level difference between two competitions is measured "
                   "tightly. **One player's** outcome is not: the individual band is "
                   "about as wide as the whole spread of player ratings, because how a "
                   "specific player adapts is mostly not predictable from his numbers. "
                   "Use the shift to set expectations, not to rank recruits.")

        st.divider()
        st.write(f"**{COMP_NAME[tgt]} players at a comparable level**")
        tgt_players = load_all_players(tgt)
        if tgt_players.empty:
            st.info(f"No {COMP_NAME[tgt]} ratings held for {season_of(tgt)}.")
        else:
            tgt_players["gap"] = (tgt_players["class_score"] - res["score_target"]).abs()
            sim = tgt_players.nsmallest(6, "gap")[
                ["name", "teams", "class_score", "form_score", "n_games"]].copy()
            sim["teams"] = sim["teams"].str.split(";").str[0].str.strip()
            sim.columns = ["Player", "Team", "Class", "Form", "GP"]
            st.dataframe(sim.round(0), width="stretch", hide_index=True)

    st.divider()
    st.write("**The measured ladder** — mean within-player shift, in rating points")
    lad = ladder.copy()
    lad["Move"] = lad.source.map(COMP_NAME) + " → " + lad.target.map(COMP_NAME)
    lad["Shift (pts)"] = lad.pts_0_100.round(1)
    lad["± 95%"] = (1.96 * lad.se / 0.25 * 9.6).round(1)
    view = lad[["Move", "n", "Shift (pts)", "± 95%"]].rename(columns={"n": "Moves observed"})
    st.dataframe(view, width="stretch", hide_index=True)
    st.caption("A positive shift means the player rates **higher** in the destination, "
               "i.e. it is the weaker pool. Both directions of a pair are listed "
               "separately and should carry opposite signs — they do, which is the "
               "main internal check on the whole ladder.")

# ─── PAGE: Squad & Contribution (GIGOT) ────────────────────────────────
elif page == "🏉 Squad (GIGOT)":
    st.subheader("Squad Contribution — GIGOT input #5")
    st.write("**Contribution Rating** = a player's share of his own team's output "
             "(attack 45% · defence 35% · points 20%), percentile-scaled so the "
             "median contributor = 50. This is real data, not a mock.")

    contrib = load_contribution(comp)
    if contrib.empty:
        st.warning("No contribution data — run regenerate_full.py first.")
    else:
        teams = sorted(contrib["team"].dropna().unique())
        team = st.selectbox("Team:", teams)
        squad = contrib[contrib["team"] == team].sort_values(
            "contribution_rating", ascending=False).reset_index(drop=True)

        # what-if team list simulator — compared against the club's strongest
        # SEVENTEEN, not against everyone who has appeared this season. Comparing a
        # 17-man line-up with a 30-plus-man season squad mostly measures squad size.
        LINEUP = 17
        st.markdown("**Team-list simulator** — untick players to see what the line-up "
                    "gives up against this club's strongest available seventeen. That "
                    "gap is the *Player Availability* signal the GIGOT match model uses.")
        available = st.multiselect("Available players:", squad["name"].tolist(),
                                   default=squad["name"].tolist())
        best17 = squad.nlargest(LINEUP, "contribution_rating")
        best = best17["contribution_rating"].sum()
        picked = (squad[squad["name"].isin(available)]
                  .nlargest(LINEUP, "contribution_rating"))
        got = picked["contribution_rating"].sum()
        missing = best17[~best17["name"].isin(available)]

        k1, k2, k3 = st.columns(3)
        k1.metric(f"Strongest {LINEUP}", f"{best:.0f}")
        k2.metric(f"Best {LINEUP} from those available", f"{got:.0f}",
                  delta=f"{got-best:+.0f}")
        pct = (got - best) / best * 100 if best else 0
        k3.metric("Line-up strength lost", f"{pct:+.1f}%")
        st.caption(f"Both figures are a seventeen. {len(squad)} players have appeared "
                   f"for this club this season, so the squad list below is deeper than "
                   f"any line-up.")
        if len(missing):
            st.caption("Out of the strongest seventeen: " + ", ".join(
                f"{r['name']} ({r['contribution_rating']:.0f})"
                for _, r in missing.iterrows()))

        st.divider()
        view = squad[["name","matches","minutes","attack_share","defence_share",
                      "contribution_rating"]].copy()
        view["attack_share"] = (view["attack_share"]*100).round(1)
        view["defence_share"] = (view["defence_share"]*100).round(1)
        view["contribution_rating"] = view["contribution_rating"].round(0)
        view.columns = ["Player","GP","Min","Attack %","Defence %","Contribution"]
        st.dataframe(view, width="stretch", hide_index=True)

        st.divider()
        st.write("**League-wide top contributors**")
        top = contrib.head(10)[["name","team","matches","contribution_rating"]].copy()
        top["contribution_rating"] = top["contribution_rating"].round(1)
        top.columns = ["Player","Team","GP","Contribution"]
        st.dataframe(top, width="stretch", hide_index=True)

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
            st.dataframe(up.round(2), width="stretch", hide_index=True)
        with c2:
            st.markdown("**📉 Biggest fallers**")
            down = both.tail(10).iloc[::-1][["player","team",r_prev,r_last,"delta"]].copy()
            down.columns = ["Player","Team",f"R{r_prev}",f"R{r_last}","Δ"]
            st.dataframe(down.round(2), width="stretch", hide_index=True)

        # distribution of deltas
        st.write("**Movement distribution** (composite z-score change)")
        hist = np.histogram(both["delta"], bins=20)
        hist_df = pd.DataFrame({"count": hist[0]},
                               index=[f"{e:.1f}" for e in hist[1][:-1]])
        st.bar_chart(hist_df)

st.divider()
_cov = pd.read_sql("SELECT competition, min(season) s0, max(season) s1, count(*) n "
                   "FROM player_match_stats GROUP BY 1 ORDER BY 1", con)


@st.cache_data
def _build_info():
    """Which code and which run produced the numbers on screen."""
    prov = runtime.provenance()
    try:
        last = pd.read_sql("SELECT run_at, target FROM model_runs "
                           "ORDER BY id DESC LIMIT 1", runtime.audit_con())
        ran = f"{last.run_at[0][:16].replace('T', ' ')} UTC" if len(last) else "unknown"
    except Exception:
        ran = "unknown"
    return prov, ran


_prov, _ran = _build_info()
st.caption(
    "BOSC — built on " + f"{_cov.n.sum():,}" + " Stats Perform player-match records: "
    + " · ".join(f"{COMP_NAME.get(r.competition, r.competition)} {r.s0}-{r.s1}"
                 for r in _cov.itertuples())
    + f". Ratings last rebuilt {_ran}."
)
st.caption(
    f"build {_prov['commit']}"
    + (" (uncommitted changes)" if _prov["dirty"] else "")
    + f" · config {_prov['config_hash']} · database {_prov['db_mb']} MB, "
      f"{_prov['db_rows']:,} rows. Every figure here can be traced to the run that "
      f"produced it — see the model_runs and data_imports tables."
)
