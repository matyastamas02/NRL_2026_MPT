# -*- coding: utf-8 -*-
"""What a player already was when he arrived — the one definition, used everywhere.

The external review of 2026-08-28 found the trace report describing 79 men as promoted
players when 69 of them had already played in the NRL and 60 were still NRL players the
season before. The aggregate it reported was carried by established players having a
spell in reserve grade, which is a different question from the one Leeds asks.

The distinction is not a detail. For a genuine first-timer the feeder rating is the only
evidence there is, so a translation has real work to do. For a man who played eighteen
NRL games last year, his NRL record is the better evidence and a translation from the
NSW Cup is answering a question nobody asked. Mixing them hides both results.

Three states, relative to a competition and a season:

  first     never appeared in this competition before this season
  returning appeared before, but not in the season immediately prior
  continuing appeared in the season immediately prior as well

Every consumer reads it from here, so a change of definition moves every number that
depends on it at once rather than in whichever script is edited first.
"""
import pandas as pd

LABELS = ["first", "returning", "continuing"]
LONG = {
    "first": "first season in the competition",
    "returning": "returning after a season away",
    "continuing": "played there the season before as well",
}


def appearances(con, competition, through=None):
    """Matches per player per season in one competition."""
    q = ("SELECT player_id, season, COUNT(*) n FROM player_match_stats "
         "WHERE competition=?")
    p = [competition]
    if through is not None:
        q += " AND season<=?"
        p.append(through)
    return pd.read_sql(q + " GROUP BY 1,2", con, params=p)


def classify(con, competition, season, through=None, min_matches=1):
    """For everyone appearing in `competition` in `season`, what he already was.

    `through` caps the history considered, so a walk-forward evaluation can ask the
    question as it stood at the time rather than with hindsight. `min_matches` sets how
    much of a previous season counts as having been there — one appearance is the
    literal reading, but a single bench cameo is not really a season in the competition.
    """
    app = appearances(con, competition, through=through)
    app = app[app.n >= min_matches]
    now = app[app.season == season][["player_id"]].drop_duplicates()
    before = app[app.season < season]
    prior = set(before.player_id)
    last = set(before[before.season == season - 1].player_id)

    def label(pid):
        if pid not in prior:
            return "first"
        return "continuing" if pid in last else "returning"

    now = now.copy()
    now["cohort"] = now.player_id.map(label)
    now["competition"] = competition
    now["season"] = season
    return now


def summarise(d):
    """Counts by cohort, in a fixed order so two runs can be compared by eye."""
    n = d.cohort.value_counts()
    return pd.DataFrame({"cohort": LABELS,
                         "n": [int(n.get(c, 0)) for c in LABELS],
                         "meaning": [LONG[c] for c in LABELS]})


def one_row_per_player(d, prefer):
    """Collapse a frame that can hold a player twice, keeping the better-evidenced row.

    A man who played in both the NSW Cup and the Queensland Cup in the same season
    appears once per feeder competition. Left alone he is counted twice and weighted
    twice in every average — the review found exactly one such case, which was small
    enough to hide and large enough to be wrong.
    """
    return (d.sort_values(prefer, ascending=False)
             .drop_duplicates("player_id", keep="first")
             .reset_index(drop=True))
