# -*- coding: utf-8 -*-
"""
BOSC — Player Intelligence Dashboard for Rugby League Recruitment
MVP: Search, Benchmarks, Comparison, Trends tabs.
"""
# A push swaps the files under a running app and Python keeps every module it has
# imported, so the app's own modules are reloaded whenever their source -- or a file
# they cache from, such as the database or the model pickle -- has changed since they
# were loaded. See app_reload.py; it reloads itself first when it has changed.
import importlib as _importlib
import os as _os

_APP_DIR = _os.path.dirname(_os.path.abspath(__file__))
import app_reload as _app_reload
if getattr(_app_reload, "_loaded_stamp", None) != _os.path.getmtime(_app_reload.__file__):
    _app_reload = _importlib.reload(_app_reload)
    _app_reload._loaded_stamp = _os.path.getmtime(_app_reload.__file__)
_app_reload.refresh(_APP_DIR)

import metric_spec as ms
import sp_schema as sp
import streamlit as st
import pandas as pd
import numpy as np
import re
import sqlite3
import unicodedata
from pathlib import Path
from predict_translation import (translate, available_pairs, comparables, COMP_NAME,
                                 top_third, start_record, bench_evidence, retention,
                                 line_forecasts, league_strength)
from sp_schema import POSITION_GROUP
import player_rating_engine as pre
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
# The full tallec.db never goes to git (DATA_NOT_IN_GIT.md). Where it is present — the
# owner's machine — the app reads it; the deployed app reads tallec_app.db, the same
# tables without player_match_raw, built by build_app_db.py. Opened read-only: a plain
# connect on a missing path creates an empty file, which is how the live app spent a
# week reporting "no such table" instead of "no database".
_HERE = Path(__file__).parent
DB_PATH = next((p for p in (_HERE / "tallec.db", _HERE / "tallec_app.db")
                if p.exists()), None)
if DB_PATH is None:
    st.error("No database next to the app: neither tallec.db nor tallec_app.db is "
             "present. Run `python build_app_db.py` beside tallec.db and commit "
             "tallec_app.db.")
    st.stop()

# A push replaces the database file under a running app: Streamlit Cloud swaps the files
# but keeps the process, and with it every cache. The cached connection then kept reading
# the replaced file and every cached query its old contents, until someone rebooted the
# app — seen on 2026-10-07, when a new tallec_app.db was live on disk and the app still
# answered from the old one. So the connection is keyed on the file itself, and the
# query caches are emptied the first time a run sees a different file.
_st = DB_PATH.stat()
DB_VERSION = (_st.st_ino, _st.st_mtime_ns, _st.st_size)


@st.cache_resource
def _db_version_seen():
    return {"version": None}


_seen = _db_version_seen()
if _seen["version"] != DB_VERSION:
    st.cache_data.clear()
    _seen["version"] = DB_VERSION


@st.cache_resource
def get_db(version):
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

con = get_db(DB_VERSION)

# The competitions need not share a current season, so every competition-scoped query
# resolves its own (`season_of`). SEASON stays as the default for anything unscoped.
SEASON = 2026

# ─── What the 0-100 number is, said once ───────────────────────────────
# Three external reviews in a row found the same misreading, and the third asked for the
# scale to be renamed rather than explained. A Class of 60 is a standing among the men
# doing that job in that competition — it is not a percentile, and it is not comparable
# with a 60 somewhere else. Every competition has its own centre and its own tau, so an
# NRL 60 and a Super League 60 are the same place in two different fields.
#
# The wording lives here because it appeared in four places and had already drifted in
# two of them.
PEER_SCALE = "peer score"
PEER_SCALE_HELP = (
    "A standing among the players doing the same job in the same competition and season. "
    "50 is the median of that group. It is NOT a percentile, and NOT comparable with the "
    "same number in another competition — use Translation to move a rating between "
    "competitions.")


def _grouping_note():
    """How the raw positions are pooled, read from the schema rather than described.

    This caption used to say "Second Row and Lock are both Back Row" — the grouping
    Leeds asked us to retire on 19 September. It sat in the app for four days after the
    change, in front of the client, because it was a sentence rather than a lookup.
    """
    by = {}
    for raw, grp in sorted(sp.POSITION_GROUP.items()):
        by.setdefault(grp, []).append(raw)
    parts = [f"**{g}** ({', '.join(v)})" for g, v in sorted(by.items())
             if len(v) > 1 or g != v[0]]
    return "Position groups follow the rating engine: " + "; ".join(parts) + "."


GROUPING_NOTE = _grouping_note()


def peer_note(comp_code, group=None, extra=""):
    """One sentence saying which pool a 0-100 figure is measured against."""
    where = COMP_NAME.get(comp_code, comp_code)
    who = f"{where} {group}" if group else f"{where} players in the same position"
    return (f"Scored against {who} — 50 is the median of that group. Not comparable "
            f"with the same number in another competition.{(' ' + extra) if extra else ''}")


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
def load_position_metrics(comp, season):
    """Mike's twelve metrics for a position, each benchmarked 0-100 against that job."""
    return pd.read_sql(
        "SELECT m.*, p.name FROM player_position_metrics m "
        "JOIN players p ON p.player_id = m.player_id "
        "WHERE m.comp = ? AND m.season = ?", con, params=(comp, season))


@st.cache_data
def load_position_categories(comp, season):
    return pd.read_sql(
        "SELECT c.*, p.name FROM player_position_category c "
        "JOIN players p ON p.player_id = c.player_id "
        "WHERE c.comp = ? AND c.season = ?", con, params=(comp, season))


@st.cache_data
def player_home():
    """Where each player is now: his most recent rated season, and in it the competition
    he played the most minutes in. Current Players shows every player once, there, so an
    NRL player with three NSW Cup games is not also listed as a NSW Cup player, and a
    Queensland Cup career that ended before this season is not shown as current."""
    d = pd.read_sql("SELECT player_id, competition AS comp, season, count(*) AS games, "
                    "sum(minutes) AS mins FROM player_match_stats "
                    "WHERE season <= (SELECT max(season) FROM player_ratings) "
                    "GROUP BY 1, 2, 3", con)
    d = d[d.season == d.groupby("player_id").season.transform("max")]
    d = d.sort_values(["mins", "games"], ascending=False)
    home = d.drop_duplicates("player_id").rename(
        columns={"comp": "home_comp", "season": "home_season"})
    rest = d[d.duplicated("player_id")]
    also = (rest.assign(txt=[f"{COMP_NAME.get(c, c)} ({n})"
                             for c, n in zip(rest.comp, rest.games)])
            .groupby("player_id").txt.agg(", ".join).rename("also_played").reset_index())
    return home[["player_id", "home_comp", "home_season"]].merge(
        also, on="player_id", how="left")


def current_ids(comp):
    """The players whose competition this is, this season (see `player_home`)."""
    h = player_home()
    return set(h[(h.home_comp == comp) & (h.home_season == season_of(comp))].player_id)


def load_all_players(comp):
    """The rated players of `comp`: with Current Players on, only those whose main
    competition this is in their most recent season."""
    d = _rated_players(comp)
    if CURRENT_ONLY:
        d = d[d.player_id.isin(current_ids(comp))]
    return d.reset_index(drop=True)


@st.cache_data
def shortlist_table():
    """`validate_shortlist.py`'s comparison of the forecast with simpler shortlists."""
    try:
        return pd.read_sql("SELECT * FROM shortlist_validation", con)
    except Exception:
        return pd.DataFrame()


def retention_sentence(tgt):
    """Who stayed in the target competition, and where the others went the next season."""
    ret = retention(tgt)
    if not ret or ret["n1"] < 20:
        return ""
    out = (f"Of {int(ret['n1'])} earlier entrants into {COMP_NAME[tgt]} (landing "
           f"{int(ret['landing_from'])}–{int(ret['landing_to'])}), "
           f"{100 * ret['stay1']:.0f}% were still playing there the season after")
    gone = {c: int(ret.get(f"left_to_{c}") or 0) for c in ("NRL", "NSW", "QLD", "SL")}
    gone = [f"{n} to the {COMP_NAME[c]}" for c, n in gone.items() if n and c != tgt]
    if "left_to_elsewhere" in ret:
        gone.append(f"{int(ret['left_to_elsewhere'])} to none of the four competitions "
                    "(retired, injured, or a league the data do not cover)")
        out += "; of those who left, " + ", ".join(gone)
    return out + (f", and {100 * ret['stay2']:.0f}% of {int(ret['n2'])} with two seasons "
                  f"to look at were there in both.")


def shortlist_sentence():
    """One sentence on the forecast shortlist's backtest record, or an empty string."""
    v = shortlist_table()
    if v.empty or not (v.rule == "forecast").any():
        return ""
    f = v[v.rule == "forecast"].iloc[0]
    return (f"In an exploratory backtest on {f.seasons} entrants, the third of entrants "
            f"into Super League with the highest forecast reached the top third "
            f"{int(f.hits)} times in {int(f.shortlisted)} ({100 * f.hit_rate:.0f}%), "
            f"against {100 * f.base_rate:.0f}% of all entrants. ")


@st.cache_data
def load_season_ratings(comp):
    """Every player-season of `comp`, rated on that season alone, with where the player
    is now (his main competition and club in his most recent season)."""
    d = pd.read_sql("SELECT s.*, p.dob FROM player_season_ratings s "
                    "JOIN players p ON p.player_id = s.player_id "
                    "WHERE s.competition = ?", con, params=(comp,))
    if d.empty:
        return d
    d = d.merge(player_home(), on="player_id", how="left")
    now = pd.read_sql("SELECT player_id, competition AS home_comp, season AS home_season, "
                      "team AS home_team FROM player_season_ratings", con)
    d = d.merge(now, on=["player_id", "home_comp", "home_season"], how="left")
    d["age"] = ((pd.to_datetime(d.season.astype(int).astype(str) + "-06-30")
                 - sp.parse_dob(d.dob)).dt.days / 365.25).round(1)
    return d


@st.cache_data
def _rated_players(comp):
    """Players active in `comp` this season, joined to their ratings."""
    query = """
    SELECT r.player_id, p.name, p.teams, p.total_minutes, p.positions,
           r.form_score, r.class_score,
           r.positional_benchmark as benchmark_score,
           r.divergence, r.confidence, r.shrinkage_B, r.n_games, r.rating_basis,
           r.class_percentile, r."group" AS peer_group,
           (SELECT COUNT(*) FROM player_match_stats s
             WHERE s.player_id = r.player_id AND s.competition = r.competition
               AND s.season <= r.season) AS n_matches,
           -- minutes on the same rows as n_matches: the registry's total_minutes spans
           -- every competition and season, 2026 included, so the two would not agree
           (SELECT COALESCE(SUM(s.minutes), 0) FROM player_match_stats s
             WHERE s.player_id = r.player_id AND s.competition = r.competition
               AND s.season <= r.season) AS n_minutes
    FROM player_ratings r
    JOIN players p ON p.player_id = r.player_id
    WHERE r.competition = ? AND r.season = ?
    ORDER BY p.name
    """
    d = pd.read_sql(query, con, params=(comp, season_of(comp)))
    # The registry lists every club a player has had, in any competition and season,
    # alphabetically, and the pages showed its first entry -- an old club as often as
    # not (Keano Kini, a Gold Coast player, showed as Burleigh Bears). `teams` now holds
    # his club in this competition and season: the one he played the most matches for.
    club = pd.read_sql("SELECT player_id, team, count(*) n FROM player_match_stats "
                       "WHERE competition = ? AND season = ? GROUP BY 1, 2", con,
                       params=(comp, season_of(comp)))
    club = club.sort_values("n", ascending=False).drop_duplicates("player_id")
    d = d.merge(club[["player_id", "team"]].rename(columns={"team": "club"}),
                on="player_id", how="left")
    d["teams"] = d.club.fillna(d.teams)
    return d.drop(columns="club").merge(player_home(), on="player_id", how="left")

@st.cache_data
def get_player_stats(player_id, comp):
    """Get raw match stats for a player in a competition (all seasons held)."""
    query = """
    SELECT season, round, team, opposition, minutes, all_run_metres, tackles,
           tackle_breaks, tries, p_c_m, errors
    FROM player_match_stats
    WHERE player_id = ? AND competition = ?
    ORDER BY season DESC, round DESC
    """
    return pd.read_sql(query, con, params=(player_id, comp))

# The radar's axes are the four categories of Mike's positional framework. Until
# 2026-10-28 they were six raw rates from the first prototype, two of them home-made
# (attack punch = line breaks + tries per game; a fantasy-style points blend) and none
# scored against the player's own position. Every axis now has a rugby league rationale
# Mike signed off, and a prop is measured against props, not against wingers.
RADAR_CATEGORIES = ["Yardage", "Attack", "Involvement", "Defence & Discipline"]


@st.cache_data
def load_player_metrics(comp, min_minutes=40):
    """Per player this season: club, most-played position and minutes, for the pickers."""
    q = """
    SELECT player_id, player as name, team, position, minutes
    FROM player_match_stats
    WHERE competition = ? AND season = ?
    """
    pm = pd.read_sql(q, con, params=(comp, season_of(comp)))
    club = (pm.groupby(["player_id", "team"]).size().rename("n").reset_index()
            .sort_values("n", ascending=False).drop_duplicates("player_id"))
    pos = (pm.groupby(["player_id", "position"]).size().rename("n").reset_index()
           .sort_values("n", ascending=False).drop_duplicates("player_id"))
    agg = pm.groupby("player_id").agg(name=("name", "first"), mins=("minutes", "sum"),
                                     games=("player_id", "count")).reset_index()
    agg = (agg.merge(club[["player_id", "team"]], on="player_id")
              .merge(pos[["player_id", "position"]], on="player_id"))
    return agg[agg["mins"] >= min_minutes].copy()


def category_profile(player_id, comp, season):
    """The four category scores at the position he played most in that season, with
    how many of each category's metrics the data could score."""
    cats = load_position_categories(comp, season)
    mets = load_position_metrics(comp, season)
    mine = mets[mets.player_id.astype(str) == str(player_id)]
    if mine.empty:
        return None, pd.DataFrame()
    main = (mine[["position", "matches"]].drop_duplicates()
            .sort_values("matches", ascending=False).position.iloc[0])
    c = cats[(cats.player_id.astype(str) == str(player_id)) & (cats.position == main)]
    return main, c.set_index("category").reindex(RADAR_CATEGORIES)

@st.cache_data
def load_contribution(comp):
    """Player Contribution ratings (a player's share of his own team's output)."""
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
    dob = sp.parse_dob(out["dob"])
    out["age"] = ((pd.Timestamp(f"{season}-06-30") - dob).dt.days / 365.25).round(1)
    out["pos_group"] = out["position"].map(POSITION_GROUP).fillna("Unknown")
    return out[["player_id", "position", "pos_group", "position_source",
                "mins_pg", "games", "age"]]


@st.cache_data
def load_ladder():
    """The ladder the forecasts use: v3, next-season horizon, on the peer-score scale.

    This read the v2 `translation_ladder` table until the sixth review round, which
    offered only eight directions (nothing but Super League from the NRL) and printed
    shifts on the scale retired in September.
    """
    from predict_translation import LADDER, LAYER_NEXT_SEASON
    lad = LADDER[LADDER.layer == LAYER_NEXT_SEASON]
    return lad.sort_values("n", ascending=False).reset_index(drop=True)

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

def _ladder_table(ladder):
    """The measured ladder: mean within-player shift per direction, with the sign check
    computed rather than asserted."""
    lad = ladder.copy()
    lad["Move"] = lad.source.map(COMP_NAME) + " → " + lad.target.map(COMP_NAME)
    lad["Shift (pts)"] = lad.shift_pts.round(1)
    lad["± 95%"] = (1.96 * lad.se_pts).round(1)
    view = lad[["Move", "n", "Shift (pts)", "± 95%"]].rename(columns={"n": "Moves observed"})
    st.dataframe(view, width="stretch", hide_index=True)
    by = {(r.source, r.target): r for r in ladder.itertuples()}
    both_ways = [(a, b) for (a, b) in by if (b, a) in by and a < b]
    odd = [f"{COMP_NAME[a]} ↔ {COMP_NAME[b]}" for a, b in both_ways
           if by[(a, b)].shift_pts * by[(b, a)].shift_pts >= 0]
    st.caption("A positive shift means players who made this move rated higher in the "
               "destination the next season than in the source. The two directions of a "
               "pair come from different players and seasons, so they need not mirror "
               f"each other; in this ladder {len(both_ways) - len(odd)} of "
               f"{len(both_ways)} pairs have opposite signs"
               + (f", and {', '.join(odd)} does not." if odd else "."))


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
        ["🔍 Search", "⚖️ Compare", "📊 Benchmarks", "🎯 Position", "🔄 Comparison",
         "🌐 Into Super League", "📚 Historical Seasons", "👥 Player Contribution",
         "📈 Trends"],
        label_visibility="collapsed")
CURRENT_ONLY = st.toggle(
    "Current players only", value=True, key="current_only",
    help="Each player once, in the competition he played the most minutes in during "
         "his most recent season. Turn off to list everyone with a rating in this "
         "competition, including players whose main competition is now another one. "
         "Earlier seasons are on the Historical Seasons page.")
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
            st.metric("Matches played", int(player["n_matches"]))
        with col3:
            st.metric("Minutes played", int(player["n_minutes"]))
        if isinstance(player.get("also_played"), str):
            st.caption(f"Also played in {season_of(comp)}: {player['also_played']} "
                       f"matches. His main competition that season was "
                       f"{COMP_NAME.get(player['home_comp'], player['home_comp'])}.")

        st.divider()

        # Ratings
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"Form {PEER_SCALE}", f"{player['form_score']:.0f}/100",
                      help=PEER_SCALE_HELP)
        with col2:
            # The 0-100 is a calibrated scale, not a percentile, and it was being read
            # as one — a published 69 used to be the 99.5th percentile of its pool. The
            # percentile now sits underneath it so nobody has to guess which it is.
            pct = player.get("class_percentile")
            grp = player.get("peer_group", "peers")
            st.metric(f"Class {PEER_SCALE}", f"{player['class_score']:.0f}/100",
                      delta=(None if pd.isna(pct) else
                             f"{pct:.0f}th percentile of {COMP_NAME.get(comp, comp)} "
                             f"{grp}"),
                      delta_color="off",
                      help=f"Where he stands among {COMP_NAME.get(comp, comp)} {grp}. "
                           f"50 is the median of that group. It is NOT comparable with "
                           f"the same number in another competition — an NRL 60 and a "
                           f"Super League 60 are the same standing in two different "
                           f"fields, not the same player. Use the Translation page to "
                           f"move a rating between competitions.")
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
        nm = int(player["n_matches"])
        ng = float(player["n_games"])
        if ng > 0:
            st.caption(
                f"Rated on **{nm} match{'es' if nm != 1 else ''}**, which count as "
                f"**{ng:.0f}** once recent seasons are weighted above old ones. "
                f"Bayesian shrinkage keeps **{B*100:.0f}%** of the raw signal and "
                f"pulls the rest toward the "
                f"{'positional' if _b == 'position_relative' else 'competition'} "
                f"average — with this little data the model is deliberately cautious. "
                f"Ratings sharpen as more rounds arrive.")
        else:
            st.caption(f"No ratable minutes yet — rating defaults to the "
                       f"{'positional' if _b == 'position_relative' else 'competition'} "
                       f"average (50).")

        # the club's own selections, from Mike's Super League team sheets: how often
        # he was in the 21, and in which part of it
        if comp == "SL":
            ts = pd.read_sql(
                "SELECT t.season, t.team, sum(t.role = 'start') AS starts, "
                "sum(t.role = 'bench') AS bench, sum(t.role = 'reserve') AS reserve, "
                "(SELECT count(DISTINCT c.round) FROM sl_teamsheets c "
                " WHERE c.season = t.season AND c.team = t.team) AS club_matches "
                "FROM sl_teamsheets t WHERE t.player_id = ? GROUP BY 1, 2 "
                "ORDER BY 1 DESC", con, params=(str(player_id),)) \
                if "sl_teamsheets" in set(pd.read_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'", con).name) \
                else pd.DataFrame()
            if len(ts):
                st.markdown("**Team selections** — from the club's team sheets")
                view = ts.head(4).rename(columns={
                    "season": "Season", "team": "Club", "starts": "Started (1–13)",
                    "bench": "Bench (14–17)", "reserve": "In the 21, not the 17",
                    "club_matches": "Club matches"})
                st.dataframe(view, width="stretch", hide_index=True)
                st.caption("Counted from Mike's Super League team sheets, 2021–2026: "
                           "shirt 1–13 is a start, 14–17 the bench, and the further four "
                           "names the rest of the 21-man squad.")

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
                sl_view = sl[["name","team","n_matches","form_score",
                              "class_score","confidence"]].copy()
                sl_view.columns = ["Player","Team","GP","Form","Class","Conf"]
                st.dataframe(sl_view.round(0), width="stretch", hide_index=True)
                st.download_button("⬇️ Export shortlist (CSV)",
                                   sl_view.to_csv(index=False).encode(),
                                   file_name="bosc_shortlist.csv", mime="text/csv")
    else:
        st.warning("No players found. Try a different search"
                   + (", or turn off *Current players only* to include players whose "
                      "main competition this season is another one." if CURRENT_ONLY
                      else "."))

# ─── PAGE: Head-to-Head Compare ────────────────────────────────────────
elif page == "⚖️ Compare":
    st.subheader("Head-to-Head Comparison")
    st.write("Two players side by side, on the four categories of the positional "
             "framework. Each axis is scored 0–100 against players **in the same "
             "position**, competition and season; 50 is the middle of that group.")

    metrics = load_player_metrics(comp)
    all_players = load_all_players(comp)
    if CURRENT_ONLY:
        metrics = metrics[metrics.player_id.isin(current_ids(comp))]
    names = metrics.sort_values("name")["name"].tolist()

    default_b = 1 if len(names) > 1 else 0
    c1, c2 = st.columns(2)
    with c1:
        name_a = st.selectbox("Player A", names, index=0)
    with c2:
        name_b = st.selectbox("Player B", names, index=default_b)

    ma = metrics[metrics["name"] == name_a].iloc[0]
    mb = metrics[metrics["name"] == name_b].iloc[0]
    season = season_of(comp)
    pos_a, prof_a = category_profile(ma["player_id"], comp, season)
    pos_b, prof_b = category_profile(mb["player_id"], comp, season)

    # rating cards
    ra = all_players[all_players["name"] == name_a]
    rb = all_players[all_players["name"] == name_b]
    c1, c2 = st.columns(2)
    for col, m, r, pos in [(c1, ma, ra, pos_a), (c2, mb, rb, pos_b)]:
        with col:
            st.markdown(f"**{m['name']}** — {m['team']} · {pos or m['position']}")
            k1, k2, k3 = st.columns(3)
            form = float(r["form_score"].iloc[0]) if len(r) else 50
            cls  = float(r["class_score"].iloc[0]) if len(r) else 50
            k1.metric("Form", f"{form:.0f}", help=PEER_SCALE_HELP)
            k2.metric("Class", f"{cls:.0f}", help=PEER_SCALE_HELP)
            k3.metric("Minutes", int(m["mins"]))

    if prof_a.empty or prof_b.empty:
        st.info("Category scores need five matches or more at one position this "
                "season; " + " and ".join(n for n, pr in [(name_a, prof_a), (name_b, prof_b)]
                                         if pr.empty) + " has fewer.")
    else:
        import plotly.graph_objects as go
        labels = RADAR_CATEGORIES
        va = [float(x) if pd.notna(x) else 0.0 for x in prof_a.score]
        vb = [float(x) if pd.notna(x) else 0.0 for x in prof_b.score]
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

        def _made_of(pr, cat):
            r = pr.loc[cat]
            if pd.isna(r.score):
                return "—"
            return f"{r.score:.0f}" + (f" ({int(r.metrics)} of {int(r['of'])})"
                                       if r.metrics < r["of"] else "")
        st.dataframe(pd.DataFrame({
            "Category": RADAR_CATEGORIES,
            f"{name_a} ({pos_a})": [_made_of(prof_a, c) for c in RADAR_CATEGORIES],
            f"{name_b} ({pos_b})": [_made_of(prof_b, c) for c in RADAR_CATEGORIES]}),
            width="stretch", hide_index=True)
        st.caption("Each category is the average of its metrics in the positional "
                   "framework (volume and rate), each scored against the same position. "
                   "\"3 of 6\" means only three of the six could be scored from the data. "
                   "Players in different positions are each measured against their own "
                   "position, so the shapes compare how good each is at his job. The "
                   "Position page shows every metric behind these numbers.")

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
        pool = pool_all[pool_all["n_matches"] >= min_games].copy()
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
                "divergence", "n_matches", "confidence"]
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
                st.caption(GROUPING_NOTE)

        with tab_lead:
            c1, c2 = st.columns([1, 1])
            with c1:
                metric = st.radio("Rank by:", ["Form", "Class"], horizontal=True,
                                  key="bm_metric")
            key = "form_score" if metric == "Form" else "class_score"
            st.write(f"**Top 15 by {metric}**")
            st.dataframe(table(pool, 15, key), width="stretch", hide_index=True)
            st.caption("Form is his last five matches; Class combines his seasons in "
                       "this competition, the latest weighted most. Both are shrunk "
                       "toward average by sample size. Divergence is Form minus Class: "
                       "positive means playing above his own level right now. "
                       + peer_note(comp))

        with tab_team:
            teams = sorted(pool["team"].dropna().unique())
            team = st.selectbox("Team:", teams, key="bm_team")
            sq = pool[pool["team"] == team]
            st.dataframe(table(sq, 25), width="stretch", hide_index=True)
            st.caption(f"{len(sq)} players at {min_games}+ matches.")

        with tab_dist:
            st.write("**Form vs Class** — where a player sits now against his own level")
            sc = pool[["name", "form_score", "class_score", "n_minutes"]].copy()
            sc["Minutes"] = sc["n_minutes"].fillna(0)
            st.scatter_chart(sc, x="class_score", y="form_score", size="Minutes",
                             height=400)
            st.caption("Above the diagonal means current form is running ahead of "
                       "season-long class; below it means the opposite. "
                       + peer_note(comp))

# ─── PAGE 3: Competition Translation ───────────────────────────────────
elif page == "🎯 Position":
    st.subheader("Position Profile")
    st.write("The twelve metrics that matter for each position, from Mike's "
             "specification, every one given as a **volume** — how much he contributes "
             "per match — and a **rate** — how efficiently he does it when he gets the "
             "chance. Each is scored 0–100 against the players doing the same job, in "
             "the same competition and the same season. 50 is the middle of that group.")

    season = season_of(comp)
    mets = load_position_metrics(comp, season)
    cats = load_position_categories(comp, season)
    if mets.empty:
        st.info(f"No position metrics held for {COMP_NAME.get(comp, comp)} {season}. "
                f"Run `python position_metrics.py --write`.")
    else:
        groups = sorted(mets.position.dropna().unique())
        c1, c2 = st.columns([1, 2])
        with c1:
            grp = st.selectbox("Position:", groups, key="pos_group")
        # every score is ranked against the whole pool of the position; the toggle only
        # decides whose profile can be opened and who is on the board below
        ranked = mets[mets.position == grp].player_id.nunique()
        shown = mets[mets.player_id.astype(str).isin(current_ids(comp))] \
            if CURRENT_ONLY else mets
        pool = (shown[shown.position == grp][["player_id", "name", "matches"]]
                .drop_duplicates().sort_values("name"))
        with c2:
            who = st.selectbox(f"{grp} ({len(pool)} players):", pool.name.values,
                               key="pos_player")
        pid = pool[pool.name == who].player_id.iloc[0]
        mine = mets[(mets.player_id == pid) & (mets.position == grp)]
        overall = mine.score.dropna()

        k = st.columns(5)
        k[0].metric("Matches", int(mine.matches.iloc[0]))
        for i, cat in enumerate(["Yardage", "Attack", "Involvement",
                                 "Defence & Discipline"]):
            row = cats[(cats.player_id == pid) & (cats.category == cat)]
            # how many metrics went into the average, because a category built from two
            # of six and one built from six of six are not the same number
            made_of = ""
            if len(row) and {"metrics", "of"} <= set(row.columns):
                got, tot = int(row.metrics.iloc[0]), int(row["of"].iloc[0])
                if got < tot:
                    made_of = f"{got} of {tot} metrics"
            k[i + 1].metric(cat.replace(" & Discipline", " & Disc."),
                            f"{row.score.iloc[0]:.0f}" if len(row) else "—",
                            delta=made_of or None, delta_color="off")
        st.caption(
            f"Ranked against the {ranked} {grp}s in "
            f"{COMP_NAME.get(comp, comp)} {season} with five matches or more."
            + ("  ⚠ A pool this small makes each place in the order worth several "
               "points — read these as broad bands, not as precise scores."
               if ranked < 40 else ""))

        for cat in ["Yardage", "Attack", "Involvement", "Defence & Discipline"]:
            g = mine[mine.category == cat]
            if g.empty:
                continue
            with st.expander(f"**{cat}**", expanded=(cat == "Yardage")):
                view = g[["form", "metric", "value", "score", "degraded"]].copy()
                view["value"] = view.value.round(2)
                view["score"] = view.score.round(0)
                view["note"] = np.where(
                    view.score.isna(), "not recorded in this competition or season",
                    np.where(view.degraded, "narrower definition here — see below", ""))
                view = view.drop(columns=["degraded"])
                view.columns = ["Form", "Metric", "Value", "0–100", "Note"]
                st.dataframe(view, width="stretch", hide_index=True)
                if cat == "Defence & Discipline":
                    st.caption("Errors, infringements and breaks conceded are inverted: "
                               "a high score means a clean player, so every figure on "
                               "this page reads the same way round.")

        if mine.degraded.any():
            st.caption("⚠ Where a metric is marked with a narrower definition, one of "
                       "its inputs is not in the data. Set restarts are absent from the "
                       "Super League extract before 2026, so discipline there counts "
                       "penalties alone; kick try assists are absent everywhere, so break "
                       "conversion is a floor rather than the full figure. The ranking "
                       "within this competition is still right; the number is not "
                       "comparable with another competition's.")

        # A metric the spec defines but the data cannot support has to be visible as an
        # absence. Forced drop-outs would otherwise just be one fewer row in Yardage,
        # indistinguishable from a metric nobody ever asked for.
        block_key = {v['engine_group']: k for k, v in ms.POSITIONS.items()}.get(grp)
        blocked = [m for cat in ms.SPEC.get(block_key, {}).values()
                   for m in cat if m.get("blocked")]
        if blocked:
            st.caption("**Not shown, pending data.** "
                       + " ".join(f"*{m['volume']}* — {m['blocked']}" for m in blocked))

        st.divider()
        st.write(f"**Strongest {grp}s in {COMP_NAME.get(comp, comp)} {season}**")
        st.caption("Average of the four category scores. A blunt summary — the profile "
                   "above is where the useful information is.")
        board_cats = cats[cats.player_id.astype(str).isin(set(pool.player_id.astype(str)))]
        board = (board_cats[board_cats.position == grp]
                 .pivot_table(index="name", columns="category", values="score")
                 .round(0))
        if len(board):
            board["Overall"] = board.mean(axis=1).round(0)
            st.dataframe(board.sort_values("Overall", ascending=False).head(15),
                         width="stretch")

elif page == "🔄 Comparison":
    st.subheader("Competition Translation")
    st.write("How players' ratings have carried from one competition to another, "
             "measured on earlier players who made the move, one season to the next.")

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

        # the app holds the raw Stats Perform label, which is what raw_position
        # takes; handing a rating-engine group to the old `position=` argument is
        # what silently dropped the position in the analysis scripts
        res = translate(p["class_score"], src, tgt,
                        raw_position=p.get("position"), age=p.get("age"),
                        minutes_pg=p.get("mins_pg"), games=p.get("games"))

        # ── the comparison card first: what earlier entrants like him actually did,
        # chosen by the rule fixed in predict_translation.comparables
        st.markdown(f"**Players who entered {COMP_NAME[tgt]} from {COMP_NAME[src]} "
                    f"before**")
        st.caption("**Rated entrants only (three or more matches in the new "
                   "competition). Signing success and signings who played less are not "
                   "measured.**")
        cmp_ = comparables(p["class_score"], src, tgt,
                           position_group=sp.POSITION_GROUP.get(p.get("position")),
                           exclude_player=p.get("player_id"))
        if cmp_["n"] == 0:
            st.info(f"Not enough earlier entrants from {COMP_NAME[src]} into "
                    f"{COMP_NAME[tgt]} near a rating of {p['class_score']:.0f} to compare "
                    f"with ({cmp_['n_direction']} rated entrants in this direction in all).")
        else:
            cols = st.columns(4 if cmp_["wide"] else 3)
            cols[0].metric("Comparable players", f"{cmp_['n']}")
            cols[1].metric(f"Their median in {COMP_NAME[tgt]}", f"{cmp_['median']:.0f}")
            cols[2].metric("Middle half of them", f"{cmp_['q25']:.0f}–{cmp_['q75']:.0f}")
            if cmp_["wide"]:
                cols[3].metric("Most of them (10–90%)",
                               f"{cmp_['q10']:.0f}–{cmp_['q90']:.0f}")
            st.caption(
                f"Players rated within ±{cmp_['window']:g} of {p['class_score']:.0f} in "
                f"{COMP_NAME[src]} when they entered {COMP_NAME[tgt]} the next season, "
                f"{cmp_['basis']}; one case per player, the closest rating. "
                f"{cmp_['n_direction']} rated entrants in this direction in all. The "
                f"rating before the move is his Class in {COMP_NAME[src]}; the rating "
                f"after it is his rating for the season he moved. These ranges describe "
                f"what these players did, not a forecast interval for this one: tested on "
                f"the entrants who came after them, the middle half held 37% of them into "
                f"Super League and 41% across all directions, not half. The forecast "
                f"ranges below are the calibrated ones.")
            show = cmp_["rows"][["name", "season_src", "season_tgt", "raw_position",
                                 "class_source", "class_target", "n_target"]].copy()
            show.columns = ["Player", f"Season in {src}", f"Season in {tgt}", "Position",
                            f"Class in {src} before", f"Season rating in {tgt}",
                            f"Matches in {tgt}"]
            st.dataframe(show.round(0), width="stretch", hide_index=True)

        # ── then the line forecast, as a secondary reference point
        st.divider()
        st.markdown("**A single-number forecast, for reference**")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Rating in {COMP_NAME[src]}", f"{p['class_score']:.0f}")
        k2.metric(f"Forecast in {COMP_NAME[tgt]}", f"{res['score_target']:.0f}",
                  delta=f"{res['shift_points']:+.1f}")
        k3.metric("His level translated", f"{res['score_ladder']:.0f}",
                  delta=f"{res['ladder_shift_points']:+.1f}", delta_color="off")
        k4.metric("Player detail",
                  f"{p.get('position') or 'Unknown'}"
                  + (f", {p['age']:.0f}y" if pd.notna(p.get("age")) else ""))

        # what the conditional model actually uses: config.json -> translation.
        # model_features (rating, direction, position group); age, minutes and matches
        # were removed after an ablation and must not be claimed here
        if res.get("forecast_method") == "straight line":
            fitted_on = (f"{res['line']['n']} earlier moves from {COMP_NAME[src]}"
                         if res["line"]["basis"] == "direction" else
                         f"{res['line']['n']} earlier moves between all competitions, "
                         f"because too few have gone from {COMP_NAME[src]}")
            how = (f"For moves into {COMP_NAME[tgt]} it is a straight line fitted to "
                   f"{fitted_on}, used as a simple default: in the 2023–2025 backtest the "
                   f"conditional model, which also uses the player's position group, "
                   f"showed no clear advantage over it.")
        else:
            how = ("It comes from the conditional model, which uses his rating, the "
                   "direction of the move and his position group.")
        st.caption(
            f"The forecast is one number for what he might rate there next season. {how} "
            "**His level translated** applies the average shift seen for this pair of "
            "competitions; it describes the pair, not one player.")
        if res.get("forecast_note"):
            st.caption(res["forecast_note"])
        st.caption(f"{res['interpretation']} Based on **{res['n_obs']} observed player "
                   f"moves** between these two competitions.")

        # ranges from the quantiles of past errors (predict_translation.RANGE_Q); the
        # +-1.96 SD band they replace covered 100% of entrants at 87 points wide
        r50, r80 = res["range50"], res["range80"]
        u1, u2, u3 = st.columns(3)
        u1.metric("Likely range (50%)", f"{r50[0]:.0f}–{r50[1]:.0f}")
        u2.metric("Wider range (80%)", f"{r80[0]:.0f}–{r80[1]:.0f}")
        u3.metric("Average shift for this pair",
                  f"{res['ladder_shift_points']:+.1f} pts",
                  delta=f"±{res['avg_band_points']:.1f} at 95%", delta_color="off")
        errs = (f"the line's past errors on its {res['range_n']} moves"
                if res.get("forecast_method") == "straight line" else
                f"the conditional model's past errors on the {res['range_n']} moves it was "
                f"fitted on" + ("" if res["range_basis"] == "direction" else
                                ", all directions together"))
        st.caption(f"The forecast plus {errs}. In an exploratory backtest on entrants from "
                   "2023 to 2026, each season scored on what was known before it, the "
                   "straight line's 50% range held 50% of entrants into Super League and "
                   "its 80% range 83% (48% and 80% across all directions); 2027 is the "
                   "first season none of this has seen. The conditional model's ranges "
                   "were not tested separately. The average shift for the pair is a "
                   "different quantity: how well the typical move is known, not where one "
                   "player will land.")

        # ── decision outputs (build_decision_tables.py)
        tt = top_third(res["score_target"], tgt)
        rec = start_record(p.get("player_id"), src, season_of(src))
        ev_b = bench_evidence(tgt)
        if tt or rec:
            d1, d2 = st.columns(2)
            if tt:
                d1.metric(f"Chance of a top-third season in {COMP_NAME[tgt]}",
                          f"{100 * tt['p']:.0f}%",
                          delta=f"{100 * tt['base_rate']:.0f}% of earlier moves into "
                                f"{COMP_NAME[tgt]}",
                          delta_color="off")
            if rec:
                d2.metric(f"Started in {COMP_NAME[src]} {season_of(src)}",
                          f"{int(rec['starts'])} of {int(rec['games'])}",
                          delta=f"club played {int(rec['team_games'])}", delta_color="off")
            if tt:
                st.caption(f"Top third: a season rating among the top third of every rated "
                           f"player in {COMP_NAME[tgt]} that season. Calibrated on forecasts "
                           f"made out of sample for {tt['n_pairs']:,} earlier next-season "
                           f"moves in all directions, through {tt['fitted_through']}. "
                           + shortlist_sentence()
                           + "The rating does not tell who will start: that was no better "
                           "than chance.")
            if rec and rec["games"] and rec["starts"] < 0.5 * rec["games"]:
                msg = (f"**Mostly off the bench in {COMP_NAME[src]}**: {int(rec['starts'])} "
                       f"starts in {int(rec['games'])} matches. The rating is per minute, "
                       f"so short stints can carry it.")
                if ev_b and ev_b["n_bench"] >= 10:
                    msg += (f" Of {int(ev_b['n_bench'])} earlier entrants into "
                            f"{COMP_NAME[tgt]} who had mostly come off the bench, "
                            f"{100 * ev_b['starter_bench']:.0f}% started half or more of "
                            f"their new club's matches, against "
                            f"{100 * ev_b['starter_start']:.0f}% of "
                            f"{int(ev_b['n_start'])} who had been starting "
                            f"(landing {int(ev_b['landing_from'])}–"
                            f"{int(ev_b['landing_to'])})"
                            + (f"; a gap this size on {int(ev_b['n_bench'])} players could "
                               f"be chance (p = {ev_b['p_value']:.2f})."
                               if ev_b.get("p_value", 0) >= 0.05 else "."))
                st.warning(msg)
            if retention_sentence(tgt):
                st.caption(retention_sentence(tgt))

        st.divider()
        st.write(f"**{COMP_NAME[tgt]} players at a comparable level**")
        tgt_players = load_all_players(tgt)
        if tgt_players.empty:
            st.info(f"No {COMP_NAME[tgt]} ratings held for {season_of(tgt)}.")
        else:
            tgt_players["gap"] = (tgt_players["class_score"] - res["score_target"]).abs()
            sim = tgt_players.nsmallest(6, "gap")[
                ["name", "teams", "class_score", "form_score", "n_matches"]].copy()
            sim["teams"] = sim["teams"].str.split(";").str[0].str.strip()
            sim.columns = ["Player", "Team", "Class", "Form", "GP"]
            st.dataframe(sim.round(0), width="stretch", hide_index=True)

    st.divider()
    st.write("**How strong each competition is** — how much higher the same player "
             "rates there than in the NRL, in peer points")
    ls = league_strength()
    ls["Competition"] = ls.comp.map(COMP_NAME)
    ls["Same player rates higher by"] = [
        "reference" if r.low == r.high == 0 else f"{r.low}–{r.high} pts"
        for r in ls.itertuples()]
    ls["Estimated from"] = ls.basis
    st.dataframe(ls[["Competition", "Same player rates higher by", "Estimated from"]],
                 width="stretch", hide_index=True)
    st.caption("A player who is a 50 in the NRL would be about a 61–67 in Super League, "
               "59–62 in the NSW Cup and 60–66 in the Queensland Cup. Super League and "
               "the Queensland Cup cannot be told apart; the NSW Cup is probably the "
               "stronger of the three. Ranges across the methods that held up under "
               "independent review, on 2021–2026 moves; exploratory, since 2027 is the "
               "first season none of it has seen. These describe the fields, not one "
               "player: for him, use the forecast above.")
    with st.expander("Raw shifts by direction (the measured ladder)"):
        st.caption("Mean within-player shift of every next-season move. Each direction "
                   "carries its own regression to the middle and its own selection (who "
                   "gets promoted or released), so directions can contradict each other "
                   "and the strengths above are the better summary.")
        _ladder_table(ladder)


# ─── PAGE: Into Super League — every rated NRL / Cup player on the SL scale ─────
elif page == "🌐 Into Super League":
    st.subheader("Players from the NRL and the Cups, on the Super League scale")
    st.write("Every rated player in the NRL, the NSW Cup and the Queensland Cup, with "
             "what he would be expected to rate in his first Super League season if he "
             "moved, the range that forecast has held, and the chance of a season in "
             "Super League's top third. One list across the three competitions, ordered "
             "by that chance.")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        srcs = st.multiselect("From:", ["NRL", "NSW", "QLD"], default=["NRL", "NSW", "QLD"],
                              format_func=lambda c: COMP_NAME[c], key="isl_src")
    with f2:
        grp_pick = st.selectbox("Position group:", ["All", "Fullback", "Winger", "Centre",
                                                    "Halves", "Hooker", "Middles", "Edge"],
                                key="isl_grp")
    with f3:
        min_games = st.slider("Matches last season, at least:", 0, 25, 10, key="isl_min")
    with f4:
        max_age = st.slider("Age, at most:", 20, 38, 33, key="isl_age")

    # The four recruitment views. Proposed thresholds, shown on the page so a club can
    # see exactly what each view means and ask for different ones.
    sl_thr = pd.read_sql("SELECT threshold FROM top_third_thresholds WHERE comp = 'SL' "
                         "ORDER BY season DESC LIMIT 1", con)
    sl_top = float(sl_thr.threshold[0]) if len(sl_thr) else 65.0
    sl_base = top_third(50, "SL")
    sl_base = sl_base["base_rate"] if sl_base else 0.45
    VIEWS = {
        "All players": ("", lambda t: np.ones(len(t), bool)),
        "Immediate starters": (
            "started at least 75% of his matches last season, averaging 60 minutes or "
            "more, with a Super League forecast of 50 or above (at least the middle of "
            "Super League players in his position)",
            lambda t: (t.start_pct >= 75) & (t.mins_pg >= 60) & (t.forecast >= 50)),
        "Young prospects": (
            f"23 or younger, with a top-third chance at least the {100 * sl_base:.0f}% "
            f"that new arrivals reach on average; no condition on starts",
            lambda t: (t.age <= 23) & (t.p_top3 >= sl_base)),
        "High upside": (
            f"26 or younger, playing above his own level right now (Form at least 5 "
            f"points over Class), and the top of his 80% range reaching Super League's "
            f"top third ({sl_top:.0f})",
            lambda t: (t.age <= 26) & (t.form_score - t.class_score >= 5)
            & (t.hi80 >= sl_top)),
        "Reliable squad options": (
            "a rating built on two or more seasons and 30 or more matches in that "
            "competition, 12 or more matches last season, and a Super League forecast of "
            "40 or above",
            lambda t: (t.n_seasons >= 2) & (t.n_matches >= 30) & (t.games >= 12)
            & (t.forecast >= 40)),
    }
    v1, v2 = st.columns([1, 2])
    with v1:
        view_pick = st.selectbox("View:", list(VIEWS), key="isl_view")
    with v2:
        names_in = st.text_area(
            "Only these players (paste names, one per line, e.g. an off-contract list):",
            key="isl_names", height=68)
    if VIEWS[view_pick][0]:
        st.caption(f"**{view_pick}** — {VIEWS[view_pick][0]}. Suggested definition; "
                   "the thresholds can be changed.")

    @st.cache_data
    def _isl_table(src, current):
        pl = _rated_players(src)
        if current:
            pl = pl[pl.player_id.isin(current_ids(src))].reset_index(drop=True)
        meta = load_player_meta(src)
        if pl.empty:
            return pd.DataFrame()
        d = pl.merge(meta, on="player_id", how="left")
        fc = line_forecasts(d.class_score.values, src, "SL")
        if fc is None:
            return pd.DataFrame()
        d = pd.concat([d.reset_index(drop=True), fc], axis=1)
        st_ = pd.read_sql("SELECT player_id, starts, games AS games_all FROM "
                          "player_season_starts WHERE comp = ? AND season = ?", con,
                          params=(src, season_of(src)))
        d = d.merge(st_, on="player_id", how="left")
        # how much evidence the rating rests on: seasons and matches in this competition
        ev = pd.read_sql("SELECT player_id, count(DISTINCT season) n_seasons "
                         "FROM player_match_stats WHERE competition = ? AND season <= ? "
                         "GROUP BY 1", con, params=(src, season_of(src)))
        d = d.merge(ev, on="player_id", how="left")
        d["team"] = d.teams
        d["source"] = src
        return d

    tabs = [_isl_table(c, CURRENT_ONLY) for c in srcs]
    tab = pd.concat([t for t in tabs if len(t)], ignore_index=True) if any(
        len(t) for t in tabs) else pd.DataFrame()
    if tab.empty:
        st.info("No rated players in the chosen competitions.")
    else:
        tab["start_pct"] = 100 * tab.starts / tab.games_all
        tab["mins_total"] = tab.mins_pg * tab.games
        m = (tab.games.fillna(0) >= min_games) & (tab.age.fillna(0) <= max_age)
        if grp_pick != "All":
            m &= tab.pos_group == grp_pick
        m &= np.asarray(VIEWS[view_pick][1](tab.fillna({"age": 99})), bool)
        missing = []
        if names_in.strip():
            def _key(s):
                s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
                return re.sub(r"[^a-z]", "", s.lower())
            want = {_key(n): n.strip() for n in names_in.splitlines() if n.strip()}
            keys = tab.name.map(_key)
            m &= keys.isin(want)
            missing = [n for k, n in want.items() if k not in set(keys)]
        v = tab[m].sort_values("p_top3", ascending=False).copy()
        v["From"] = v.source.map(COMP_NAME)
        v["Team"] = v.team.fillna("—")
        v["Started"] = [f"{100 * a / b:.0f}% ({int(a)} of {int(b)})"
                        if pd.notna(a) and pd.notna(b) and b else "—"
                        for a, b in zip(v.starts, v.games_all)]
        v["Mostly bench"] = np.where(v.starts.notna() & (v.starts < 0.5 * v.games_all),
                                     "yes", "")
        v["Sample"] = [f"{int(s)} season{'s' if s != 1 else ''} · {int(n)} matches"
                       if pd.notna(s) else "—" for s, n in zip(v.n_seasons, v.n_matches)]
        v["Likely range (50%)"] = [f"{a:.0f}–{b:.0f}" for a, b in zip(v.lo50, v.hi50)]
        v["Top-third chance"] = (100 * v.p_top3).round(0)
        show = v[["name", "From", "Team", "position", "age", "class_score", "forecast",
                  "Likely range (50%)", "Top-third chance", "games", "mins_pg",
                  "mins_total", "Started", "Sample", "Mostly bench"]]
        show.columns = ["Player", "From", "Team", "Position", "Age", "Rating there",
                        "SL forecast", "Likely range (50%)", "Top-third chance (%)",
                        f"Matches {season_of('NRL')}", "Avg min", "Total min",
                        "Started", "Rating built on", "Mostly bench"]
        st.caption(f"{len(v):,} players match the filters."
                   + (f" Not found among them: {', '.join(missing)}." if missing else ""))
        st.dataframe(show.head(150).round(1), width="stretch", hide_index=True)
        st.download_button("⬇️ Export this list (CSV)", show.to_csv(index=False),
                           "into_super_league.csv", "text/csv", key="isl_csv")
        val = shortlist_table()
        st.caption(
            "**How to read it.** The forecast is the app's straight line for each "
            "direction, fitted on earlier moves into Super League; the range is that "
            "line's past errors; the chance is calibrated on forecasts made out of sample. "
            + shortlist_sentence()
            + "Most of these players will never move, and the list says nothing about who "
            "is available or affordable; it ranks how a move would be expected to go on "
            "the numbers. A high rating built mostly off the bench is flagged, because "
            "short stints can carry a per-minute rating. *Rating built on* is the "
            "evidence behind the rating: a high rating from five matches is less certain "
            "than the same rating over two seasons, and the rating is already pulled "
            "toward average the less evidence there is."
            + (" " + retention_sentence("SL") if retention_sentence("SL") else ""))
        if len(val):
            with st.expander("Is the forecast a better shortlist than simpler rules?"):
                tv = val.copy()
                tv["Hit rate"] = [f"{100 * r:.0f}% ({int(h)} of {int(k)})"
                                  for r, h, k in zip(tv.hit_rate, tv.hits, tv.shortlisted)]
                tv["Forecast minus rule"] = [
                    "—" if pd.isna(d_) else f"{100 * d_:+.0f} pts [{100 * lo:+.0f}, {100 * hi:+.0f}]"
                    for d_, lo, hi in zip(tv.forecast_minus, tv.lo, tv.hi)]
                st.dataframe(tv[["label", "Hit rate", "Forecast minus rule", "auc"]]
                             .rename(columns={"label": "Shortlist by", "auc": "AUC"})
                             .round(2), width="stretch", hide_index=True)
                st.caption(
                    f"The same {int(val.entrants.iloc[0])} entrants into Super League, each "
                    "season's top third by each rule, using only what was known before "
                    "the move. A hit is a season in Super League's top third. Brackets "
                    "are 95% intervals; an interval that includes 0 means the backtest "
                    "cannot tell the two rules apart. Within one source competition the "
                    "forecast ranks players exactly as their rating does, so the two "
                    "differ only in how they rank players from different competitions. "
                    "Exploratory: these entrants have been studied before.")

        # ── the meeting shortlist ────────────────────────────────────────────
        st.divider()
        st.markdown("**Shortlist for a meeting** — up to five players from the list "
                    "above, with what to expect and how sure the numbers are")
        picks = st.multiselect("Players:", v.name.tolist(),
                               default=v.name.head(5).tolist(), max_selections=5,
                               key="isl_picks")
        for nm in picks:
            r = v[v.name == nm].iloc[0]
            cmp_ = comparables(r.class_score, r.source, "SL",
                               position_group=r.pos_group, exclude_player=r.player_id)
            with st.container(border=True):
                st.markdown(f"**{r['name']}** — {r.Team}, {COMP_NAME[r.source]} · "
                            f"{r.position}" + (f" · {r.age:.0f}" if pd.notna(r.age) else ""))
                k = st.columns(4)
                k[0].metric("SL forecast", f"{r.forecast:.0f}",
                            help="Expected season rating in Super League, on the peer "
                                 "scale (50 = middle of his position group).")
                k[1].metric("Likely range", f"{r.lo50:.0f}–{r.hi50:.0f}",
                            delta=f"80%: {r.lo80:.0f}–{r.hi80:.0f}", delta_color="off")
                k[2].metric("Top-third chance", f"{100 * r.p_top3:.0f}%",
                            delta=f"average arrival {100 * sl_base:.0f}%", delta_color="off")
                k[3].metric("Started", r.Started.split(" (")[0],
                            delta=(f"{r.mins_pg:.0f} min a match" if pd.notna(r.mins_pg)
                                   else None), delta_color="off")
                note = [f"Rating built on {r.Sample}."]
                if r["Mostly bench"]:
                    note.append("Mostly off the bench last season, so the rating may "
                                "flatter him over 80 minutes.")
                if cmp_["n"]:
                    note.append(f"{cmp_['n']} earlier entrants from "
                                f"{COMP_NAME[r.source]} at a similar rating "
                                f"({cmp_['basis']}) went on to rate "
                                f"{cmp_['q25']:.0f}–{cmp_['q75']:.0f} in Super League "
                                f"(middle half; median {cmp_['median']:.0f}).")
                st.caption(" ".join(note))
        if picks:
            st.download_button(
                "⬇️ Export the shortlist (CSV)",
                show[show.Player.isin(picks)].to_csv(index=False),
                "bosc_shortlist_into_sl.csv", "text/csv", key="isl_picks_csv")

    # ── the track record ─────────────────────────────────────────────────────
    try:
        bt = pd.read_sql("SELECT * FROM entrant_backtest", con)
    except Exception:
        bt = pd.DataFrame()
    if len(bt):
        st.divider()
        st.markdown("**Track record** — every player who moved into Super League from "
                    f"these competitions, {int(bt.season_tgt.min())}–"
                    f"{int(bt.season_tgt.max())}: what the forecast said before he moved, "
                    "and what happened")
        t1, t2 = st.columns(2)
        with t1:
            bt_src = st.multiselect("From:", sorted(bt.source.unique()),
                                    default=sorted(bt.source.unique()),
                                    format_func=lambda c: COMP_NAME[c], key="bt_src")
        with t2:
            bt_seas = st.multiselect("Season in Super League:",
                                     sorted(bt.season_tgt.unique(), reverse=True),
                                     default=sorted(bt.season_tgt.unique(), reverse=True),
                                     key="bt_seas")
        b_ = bt[bt.source.isin(bt_src) & bt.season_tgt.isin(bt_seas)].copy()
        b_["Moved"] = [f"{COMP_NAME[c]} {int(a)} → SL {int(t)}"
                       for c, a, t in zip(b_.source, b_.season_src, b_.season_tgt)]
        b_["Range (50%)"] = [f"{a:.0f}–{b:.0f}" for a, b in zip(b_.lo50, b_.hi50)]
        b_["Range (80%)"] = [f"{a:.0f}–{b:.0f}" for a, b in zip(b_.lo80, b_.hi80)]
        b_["Outcome"] = np.where(b_.in50, "inside 50%",
                                 np.where(b_.in80, "inside 80%", "outside"))
        b_["Top third"] = np.where(b_.top3 == 1, "yes", "")
        view = b_.sort_values(["season_tgt", "forecast"], ascending=False)[
            ["name", "Moved", "position", "rating_source", "forecast", "Range (50%)",
             "Range (80%)", "actual", "Outcome", "Top third", "matches_sl"]]
        view.columns = ["Player", "Moved", "Position", "Rating before", "Forecast",
                        "Range (50%)", "Range (80%)", "Rated in SL", "Outcome",
                        "SL top third", "SL matches"]
        st.dataframe(view.round(0), width="stretch", hide_index=True)
        st.caption(
            f"{len(b_)} moves shown. Each forecast uses only the seasons before the move. "
            f"The 50% range held {100 * b_.in50.mean():.0f}% of them and the 80% range "
            f"{100 * b_.in80.mean():.0f}%; the forecast missed by "
            f"{(b_.actual - b_.forecast).abs().mean():.0f} points on average. All moves "
            "are listed, the misses with the hits. Exploratory backtest.")

# ─── PAGE: Historical Seasons ──────────────────────────────────────────
elif page == "📚 Historical Seasons":
    st.subheader("Historical Seasons")
    st.write("Every player-season in this competition, each rated on that season alone "
             "and shown with its year and club. A strong season here is not a current "
             "rating: **Now** says where the player played most in his latest season. "
             "The other pages show current players only.")
    hs = load_season_ratings(comp)
    if hs.empty:
        st.info(f"No season ratings held for {COMP_NAME.get(comp, comp)}.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        seasons = sorted(hs.season.unique(), reverse=True)
        with f1:
            pick = st.multiselect("Seasons:", seasons, default=seasons, key="hs_seasons")
        with f2:
            grp_hs = st.selectbox("Position group:", ["All"] + sorted(
                hs.group.dropna().unique()), key="hs_grp")
        with f3:
            min_hs = st.slider("Matches that season, at least:", 1, 25, 8, key="hs_min")
        with f4:
            who_hs = st.text_input("Player name contains:", key="hs_name")
        v = hs[hs.season.isin(pick) & (hs.matches >= min_hs)]
        if grp_hs != "All":
            v = v[v.group == grp_hs]
        if who_hs:
            v = v[v.name.str.contains(who_hs, case=False, na=False)]
        v = v.sort_values("season_score", ascending=False).copy()
        v["Now"] = [
            f"{COMP_NAME.get(c, c)} {int(y)}" + (f" · {t}" if isinstance(t, str) else "")
            if isinstance(c, str) else "—"
            for c, y, t in zip(v.home_comp, v.home_season, v.home_team)]
        show = v[["name", "season", "team", "position", "age", "matches", "minutes",
                  "season_score", "Now"]].copy()
        show.columns = ["Player", "Season", "Club", "Position", "Age", "Matches",
                        "Minutes", "Season rating", "Now"]
        st.caption(f"{len(v):,} player-seasons match the filters.")
        st.dataframe(show.head(200).round(1), width="stretch", hide_index=True)
        st.download_button("⬇️ Export this list (CSV)", show.to_csv(index=False),
                           f"historical_seasons_{comp}.csv", "text/csv", key="hs_csv")
        w = pre.SEASON_WEIGHTS or []
        st.caption(
            "**Season rating** is the same 0–100 peer score as Class, built from that "
            "season's matches alone and pulled toward 50 the fewer there were. 50 is the "
            "median of the position group in that competition and season, so a season "
            "rating is not comparable with one in another competition. **Class**, on the "
            "other pages, combines a player's seasons in a competition: the latest counts "
            + ", the one before ".join(["1.0"] + [f"{x:g}" for x in w[1:2]])
            + (", then " + ", ".join(f"{x:g}" for x in w[2:]) if len(w) > 2 else "")
            + f", and anything older {pre.SEASON_WEIGHT_FLOOR:g}. Form is his last "
            "five matches.")

# ─── PAGE: Player Contribution ─────────────────────────────────────────
elif page == "👥 Player Contribution":
    st.subheader("Player Contribution")
    st.write("**Contribution Rating** = a player's share of his own team's output "
             "(attack 45% · defence 35% · points 20%), percentile-scaled so the "
             "median contributor = 50.")

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
        st.markdown("**Squad depth** — untick players to see what the line-up gives up "
                    "against this club's strongest seventeen: how much the club leans "
                    "on a few players, and what a departure would leave to fill.")
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
    prov = runtime.provenance(con, DB_PATH)
    ran = "unknown"
    # The audit log stays local; the deployed copy carries its run table as
    # audit_model_runs (build_app_db.py), so fall back to that when the log is absent.
    for table, source in (("model_runs", runtime.audit_con), ("audit_model_runs", lambda: con)):
        try:
            last = pd.read_sql(f"SELECT run_at, target FROM {table} "
                               "ORDER BY id DESC LIMIT 1", source())
        except Exception:
            continue
        if len(last):
            ran = f"{last.run_at[0][:16].replace('T', ' ')} UTC"
            break
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
    + f" · config {_prov['config_hash']} · database {DB_PATH.name} {_prov['db_mb']} MB, "
      f"{_prov['db_rows']:,} rows. Every figure here can be traced to the run that "
      f"produced it — see the model_runs and data_imports tables."
)
