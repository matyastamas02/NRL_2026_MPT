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

# Stats Perform position string -> rating-engine position group.
#
# Props and locks share one group, "Middles", and second-rowers stand alone as "Edge".
# Leeds asked for this on 2026-09-19 and it is their competition to describe; Mike
# passed it on saying he does not agree with it himself but that it does make the edge
# forwards a more distinct category.
#
# Worth recording that it also repairs something. Lock had been its own group since
# 31 August, which left Super League Lock at 45 players — the thinnest peer pool in the
# system, where a single place in the order was worth more than two points. Folded into
# Middles that pool becomes 179, and the NRL's 225.
#
# The two profiles differed in exactly one metric: props were measured on hit-up metres
# and locks on line-break assists. Middles takes hit-up metres, because hit-ups are what
# every middle forward does, while line-break assists describe the ball-playing lock —
# which is the distinction Leeds is asking us to stop drawing.
POSITION_GROUP = {
    "Full Back": "Fullback", "Winger": "Winger", "Centre": "Centre",
    "Five-Eighth": "Halves", "Half Back": "Halves", "Hooker": "Hooker",
    "Prop": "Middles", "Lock": "Middles", "Second Row": "Edge",
    "Interchange": "Bench",
}

# Older position strings from the Gerard-seeded data, kept so historical rows still
# resolve. They are deliberately NOT in POSITION_GROUP: that map is the canonical set of
# Stats Perform labels, and the translation feature builder treats anything outside it as
# an unknown position rather than guessing. The rating engine, which has to read every
# row ever loaded, uses POSITION_GROUP | LEGACY_POSITIONS instead.
#
# "Unknown" resolves to Bench here for the engine's benefit only. That is a fallback for
# pooling, not a claim that the player is a bench forward, and it must never leak into
# the translation features, where an unknown position has its own flag.
LEGACY_POSITIONS = {
    "Fullback": "Fullback", "Halfback": "Halves", "2nd Row": "Edge",
    "Reserve": "Bench", "Unknown": "Bench",
}

# every string the engine can see, canonical labels winning over legacy ones
ALL_POSITIONS = {**LEGACY_POSITIONS, **POSITION_GROUP}

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


def parse_dob(series):
    """Dates of birth, which arrive in two formats in the same column.

    2,699 of them are ISO (`1988-03-10`) and 287 are day-first with slashes
    (`08/07/1991`). A plain `pd.to_datetime` infers ONE format from the first non-null
    value and then fails every row in the other one — and because the inference depends
    on which row happens to come first, the same data parsed in a different order gives
    a different answer. That is how the age feature came to be present on 6% of players
    in one script and 87% in another, with nobody noticing either number was wrong.

    Each format is therefore matched and parsed explicitly. The slash form is
    unambiguously day-first: its first field reaches 31 and its second never exceeds 12.
    """
    import pandas as _pd
    v = series.astype(str)
    out = _pd.Series(_pd.NaT, index=series.index, dtype="datetime64[ns]")
    iso = v.str.match(r"^\d{4}-\d{2}-\d{2}")
    out[iso] = _pd.to_datetime(v[iso], format="%Y-%m-%d", errors="coerce")
    slash = v.str.match(r"^\d{1,2}/\d{1,2}/\d{4}")
    out[slash] = _pd.to_datetime(v[slash], format="%d/%m/%Y", errors="coerce")
    return out


def age_at(dob, season):
    """Age in years at the midpoint of a season, or NaN where the date is unknown.

    The NaN matters and was missing. Subtracting a NaT gives a NaT, and casting that
    through `timedelta64[D]` to float yields the int64 sentinel rather than a NaN — so a
    player with no recorded date of birth came out at about -2.5e16 years old. Roughly
    5% of the arrival cohort carried that value.

    Downstream it did not look like an error, which is why it survived. The translation
    feature builder clamps age to [16, 42], so -2.5e16 became a confident 16 with its
    `age_missing` flag at zero: the model was told the man was definitely sixteen rather
    than that his age was unknown. In the arrival model the raw value went into a
    standard scaler and took the column over.

    Found on 2026-09-24 while checking a claim in the fourth external review — not by
    the review itself, and not by any test.
    """
    import numpy as _np
    import pandas as _pd
    mid = _pd.to_datetime(_pd.Series(season).astype(int).astype(str) + "-06-30")
    mid.index = dob.index if hasattr(dob, "index") else None
    days = (_pd.Series(mid.values) - _pd.Series(_pd.to_datetime(dob).values))
    out = days.dt.days.astype(float) / 365.25
    out[_pd.isna(dob).values] = _np.nan
    out.index = dob.index if hasattr(dob, "index") else out.index
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
