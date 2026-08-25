# -*- coding: utf-8 -*-
"""Stats Perform feed schema — shared by every ingest path.

Single source of truth for the canonical-field -> engine-column mapping, so the
full-season rebuild (ingest_full_season.py) and the Australian history load
(ingest_aus_history.py) cannot drift apart.
"""
import re
import unicodedata

# canonical Stats Perform field -> lowercase engine-compat key (what the rating
# engine, gigot_contribution and bosc_app query).
ENGINE_MAP = {
    "Ball Runs - Metres Gained":        "all_run_metres",
    "Ball Runs - Post Contact Metres":  "p_c_m",
    "Tackle Break":                     "tackle_breaks",
    "Line Break":                       "line_breaks",
    "Tackle - Total Made":              "tackles",
    "Offload - Successful":             "offloads",
    "Try Assists":                      "try_assists",
    "Try Scored - Total":               "tries",
    "Errors":                           "errors",
    "Receipts":                         "receipts",
    "Ball Runs - Total":                "ball_runs_total",
    "Pass - Attempted":                 "passes",
}

# meta columns of the engine table, in order, ahead of the ENGINE_MAP values
ENGINE_META = ["player_id", "player", "team", "opposition", "competition", "season",
               "round", "minutes", "position", "position_source", "fantasy"]

# canonical fields that must NOT be duplicated into the raw table (SQLite compares
# column names case-insensitively, so the raw set has to stay disjoint from these)
LOWER_ALIASES = set(ENGINE_MAP.values()) | {
    "player", "team", "opposition", "minutes", "position", "position_source",
    "fantasy", "round"}

# Stats Perform position string -> rating-engine position group
POSITION_GROUP = {
    "Full Back": "Fullback", "Winger": "Winger", "Centre": "Centre",
    "Five-Eighth": "Halves", "Half Back": "Halves", "Hooker": "Hooker",
    "Prop": "Prop", "Second Row": "Back Row", "Lock": "Back Row",
    "Interchange": "Bench",
}

COMPETITIONS = [
    ("NRL", "National Rugby League", "Australia"),
    ("SL", "Super League", "England"),
    ("NSW", "NSW Cup", "Australia"),
    ("QLD", "Queensland Cup", "Australia"),
]


def slugify(name):
    """Fallback player key when no permanent Player ID is available."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def clean_name(name):
    """Strip the asterisks some Stats Perform exports prefix to names."""
    return re.sub(r"\*", "", str(name)).strip()


def normalize_player_id(series):
    """Stats Perform Player ID as a clean string.

    pandas reads the column as float whenever the file has blank rows, so a plain
    astype(str) yields "24528.0" where the database holds "24528" — which forks every
    player into a second identity. Numeric ids therefore go through int, and anything
    genuinely non-numeric keeps its literal form.
    """
    import pandas as _pd
    num = _pd.to_numeric(series, errors="coerce")
    out = _pd.Series(index=series.index, dtype=object)
    ok = num.notna()
    out[ok] = num[ok].astype("int64").astype(str)
    lit = series.notna() & ~ok
    out[lit] = series[lit].astype(str)
    return out


def primary_position(series):
    """A player's career position from his per-match assignments.

    Interchange is a ROLE, not a position: a prop who mostly comes off the bench is
    still a prop, and for a positional benchmark he belongs against props. So the
    mode is taken over his STARTING positions, and Interchange is used only when he
    has never started. Every ingest path must use this, or the position a player
    gets will depend on which script last touched his rows.
    """
    starts = series[series != "Interchange"]
    pool = starts if len(starts) else series
    m = pool.mode()
    return m.iloc[0] if len(m) else "Interchange"


def fantasy_proxy(df):
    """No official fantasy column in the feed — standard attacking/defensive blend."""
    g = lambda c: df[c].fillna(0) if c in df.columns else 0
    return (g("tries") * 4 + g("try_assists") * 2 + g("line_breaks") + g("tackle_breaks")
            + g("all_run_metres") / 10 + g("tackles") * 0.5 - g("errors"))
