# -*- coding: utf-8 -*-
"""TALLEC player rating engine — Form, Class, Divergence.

Replaces the mock ratings with defensible statistics. Three ideas do the work:

0. Availability-aware blending. A stat that a season never recorded must not be
   scored as if every player was average at it: post-contact metres only appear from
   2025, and spending 24% of a prop's weight on a dead input compressed every earlier
   composite. Each pool decides which rates it recorded and the position weights are
   renormalised over those.

1. Position-relative standardization. A prop's 90 run metres and a fullback's
   150 are not comparable; every per-minute stat is z-scored WITHIN its
   (position group, competition) peer pool, then blended by a position-specific
   emphasis vector into one composite performance score per player-match.

2. Empirical-Bayes shrinkage (one-way random-effects model). Rating a player on
   2 games is mostly noise. We estimate the game-to-game noise (sigma^2) and the
   true spread between players (tau^2), then pull each player's observed mean
   toward the positional prior by B_i = tau^2 / (tau^2 + sigma^2 / n_i). Few
   games or noisy position => heavy shrinkage. This is the honest answer to
   "you can't rate a player on 2 games" — the engine says so numerically.

3. Leakage discipline. The snapshot rating (for BOSC scouting) uses all of a
   player's games — correct, you want the best current estimate. The pre-match
   rating (for GIGOT prediction) for match M uses only matches < M, and a
   permutation test asserts match M's own stats never leak into its rating.

Form = short recent window (shrunk). Class = full history (shrunk). Divergence
= Form - Class (short-term over/under-performance vs structural level). With
only R11-12 the two windows nearly coincide; the machinery is built to separate
them cleanly once more rounds land.
"""
import sqlite3
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import sp_schema as sp

BASE = Path(__file__).parent
DB = BASE / "tallec.db"

# config.json is optional — fall back to the same defaults it ships with so the
# engine never hard-fails if the file is absent (it is *.json-gitignored).
_DEFAULT_CONFIG = {
    "form_calculation": {"window_matches": 5},
    "data_quality_thresholds": {"min_minutes_for_form": 20},
}
try:
    CONFIG = json.loads((BASE / "config.json").read_text())
except (FileNotFoundError, ValueError):
    CONFIG = _DEFAULT_CONFIG

# ── Core per-minute stats and position emphasis ────────────────────────────
# Each raw stat -> per-minute rate name. "lower is better" stats (errors) get
# their sign flipped so a higher composite is always better.
RATE_STATS = {
    "all_run_metres": ("run_pm", +1),
    "p_c_m": ("pcm_pm", +1),
    "tackle_breaks": ("tb_pm", +1),
    "line_breaks": ("lb_pm", +1),
    "tackles": ("tck_pm", +1),
    "offloads": ("off_pm", +1),
    "try_assists": ("ta_pm", +1),
    "tries": ("tries_pm", +1),
    "errors": ("err_pm", -1),
}
RATE_ORDER = ["run_pm", "pcm_pm", "tb_pm", "lb_pm", "tck_pm",
              "off_pm", "ta_pm", "tries_pm", "err_pm"]

# Position emphasis weights over RATE_ORDER (each row sums to 1). Config-tunable.
POSITION_WEIGHTS = {
    "Fullback":  [0.20, 0.08, 0.18, 0.20, 0.05, 0.05, 0.09, 0.10, 0.05],
    "Winger":    [0.24, 0.08, 0.16, 0.19, 0.05, 0.02, 0.04, 0.17, 0.05],
    "Centre":    [0.19, 0.09, 0.19, 0.11, 0.10, 0.10, 0.08, 0.09, 0.05],
    "Halves":    [0.10, 0.05, 0.10, 0.14, 0.10, 0.09, 0.27, 0.08, 0.07],
    "Hooker":    [0.14, 0.09, 0.09, 0.05, 0.29, 0.19, 0.09, 0.01, 0.05],
    # Middles inherits the retired Prop profile and Edge the retired Back Row one. The
    # six candidates compared in WEIGHTS_REPORT.md sit within 0.004 of each other on
    # year-to-year reliability (0.6035 to 0.6067) while a uniform control falls to
    # 0.5316 — so the measurement can tell a good weighting from a bad one and finds no
    # difference between the sensible ones. Inheriting is therefore the least-change
    # choice, and it is a calibration decision rather than a finding: a coach who wants
    # a middle judged differently should say so and win.
    "Middles":   [0.29, 0.24, 0.05, 0.04, 0.24, 0.05, 0.00, 0.04, 0.05],
    "Edge":      [0.19, 0.14, 0.10, 0.09, 0.24, 0.14, 0.00, 0.05, 0.05],
    "Bench":     [0.15, 0.12, 0.12, 0.10, 0.20, 0.12, 0.06, 0.08, 0.05],
}

# The engine used to carry a position map of its own, splitting Prop from Back Row
# (second row *and* lock). `sp_schema` moved to Middles and Edge on 2026-09-20 at Leeds's
# request and the engine was not changed with it, so for two days ratings were computed
# in one set of peer pools while the translation model labelled players by another. A
# lock was measured against second-rowers and then described as a middle. The external
# review of 2026-09-22 found it; the damage was a median of 54.9 for locks against 45.6
# for second-rowers, when both should sit near 50 within their own group.
#
# There is now one map. The engine adds the legacy strings because it reads every row
# ever loaded; nothing else should.
POSITION_GROUP = sp.ALL_POSITIONS

MIN_MINUTES = CONFIG["data_quality_thresholds"]["min_minutes_for_form"]  # 20
FORM_WINDOW = CONFIG["form_calculation"]["window_matches"]                # 5
# share of the pool that must have a known position before ratings are computed
# within position group rather than across the whole competition
MIN_POS_COVERAGE = CONFIG.get("positional_benchmark", {}).get("min_position_coverage", 0.90)
# A rate counts as recorded in a pool when more than this share of rows carry a
# non-zero value. Post-contact metres are absent before 2025 (0-6% of rows) and present
# after (96-98%); try assists are sparse but real all along (13-17%). Anything below
# the threshold is dropped from the blend and its weight redistributed, so a season
# that never recorded a stat is not silently scored as if every player was average at
# it — which compressed every pre-2025 composite and broke cross-season comparison.
MIN_RATE_COVERAGE = CONFIG.get("data_quality_thresholds", {}).get("min_rate_coverage", 0.10)
# The last season anything is FITTED on. Seasons after it are held out so they can test
# what was projected before they happened — which is only a test if they contributed
# nothing to the projection. None fits on everything. Both the rating rebuild and the
# translation fit read it from here so the two cannot be frozen at different points.
FREEZE_SEASON = CONFIG.get("evaluation", {}).get("freeze_season")
# How much an older season counts toward Class. Indexed by seasons ago; anything beyond
# the list gets the floor. None means every season counts the same, which is what this
# project did until 2026-09-14 and which no established projection system does — a
# player is not the average of his career, he is mostly what he is now.
SEASON_WEIGHTS = CONFIG.get("class_calculation", {}).get("season_weights")
SEASON_WEIGHT_FLOOR = CONFIG.get("class_calculation", {}).get("season_weight_floor", 0.15)


def season_weight(seasons_ago):
    """Weight for a match that many seasons before the one being rated."""
    if SEASON_WEIGHTS is None:
        return 1.0
    i = int(max(0, seasons_ago))
    return float(SEASON_WEIGHTS[i]) if i < len(SEASON_WEIGHTS) else SEASON_WEIGHT_FLOOR


MIN_GROUP_FOR_CENTRE = 15   # below this a group has no stable centre of its own


def _modal_group(g):
    """The peer pool a player belongs to: the group most of his rated matches sat in.

    Taken from the matches themselves rather than from a career position lookup, so it
    is by construction the pool his composite was standardised against.
    """
    if "group" not in g.columns or g["group"].isna().all():
        return "Bench"
    m = g["group"].mode()
    return m.iloc[0] if len(m) else "Bench"


def _weighted_class(g, ref_season):
    """(effective matches, weighted mean composite) for one player.

    The effective count is the SUM of the weights rather than the row count, so the
    shrinkage treats a career of old seasons as the weaker evidence it is. With weights
    switched off both fall back to exactly the old behaviour.

    Sum-of-weights is a choice, not the textbook quantity. The usual effective sample
    size of a weighted mean is Kish's (sum w)^2 / sum w^2, which is always the larger of
    the two here and would shrink long old careers less. Using the sum instead is a
    power-prior style decision: it says an eight-year-old season is worth a fraction of
    a match as *evidence about this season*, not merely a fraction of a match as data.
    The external review of 2026-09-22 was right that it should be labelled that way and
    not presented as the mathematically correct n. Which of the two forecasts better has
    not been measured; it is a live question, and `ablate_translation.py` is the shape
    the measurement would take.
    """
    if SEASON_WEIGHTS is None or ref_season is None:
        return len(g), float(g["composite"].mean())
    w = (ref_season - g["season"]).map(season_weight).astype(float).values
    c = g["composite"].astype(float).values
    total = float(w.sum())
    if total <= 0:
        return len(g), float(c.mean())
    return total, float((w * c).sum() / total)


class PlayerRatingEngine:
    def __init__(self, comp_code="NRL", force_mode=None):
        self.comp = comp_code
        # force_mode="competition_relative" makes the engine ignore position even
        # when coverage allows it. Cross-competition work needs BOTH sides
        # standardized the same way, and Super League has no position source.
        self.force_mode = force_mode
        self.sigma2 = None   # within-player (game-to-game) variance
        self.tau2 = None     # between-player (true talent) variance
        self.grand_mean = 0.0
        self.position_mode = "competition_relative"
        self.position_coverage = 0.0
        self.available = None   # rates this pool recorded; set by _fit_standardization
        self.dropped = []

    # ── 1. Standardization (fit / transform split) ────────────────────────
    # Standardization params are POPULATION descriptors ("what's an average
    # prop"), not outcome data — fitting them on the full pool is a design
    # choice, and separating fit from transform lets us fit on train and apply
    # to test unchanged once real multi-season data arrives.
    # Standardizing within position group only works if we know the position of
    # (nearly) the whole pool. With partial coverage the composite stops being
    # comparable across players — the ones whose position is unknown all land in
    # one bucket whose mean/std is a blend of every position — and the "50 = pool
    # average" reading of the benchmark breaks. Below the coverage threshold we
    # therefore pool the whole competition, which is what the engine did before
    # any position source existed. `position_mode` records which applied.
    def _position_groups(self, df):
        known = df["position"].notna() & (df["position"] != "Unknown")
        self.position_coverage = float(known.mean()) if len(df) else 0.0
        if self.force_mode == "competition_relative":
            self.position_mode = "competition_relative"
            return pd.Series("Bench", index=df.index)
        if self.position_coverage >= MIN_POS_COVERAGE:
            self.position_mode = "position_relative"
            return df["position"].map(POSITION_GROUP).fillna("Bench")
        self.position_mode = "competition_relative"
        return pd.Series("Bench", index=df.index)

    def _fit_standardization(self, df):
        df = df.copy()
        df["group"] = self._position_groups(df)
        mins = df["minutes"].clip(lower=1)
        self.norm = {}  # group -> rate -> (mean, std)
        for raw, (rate, sign) in RATE_STATS.items():
            df[rate] = sign * (df[raw].fillna(0.0) if raw in df else 0.0) / mins
        # which rates this pool actually recorded
        self.available = []
        self.dropped = []
        for raw, (rate, _) in RATE_STATS.items():
            share = float((df[raw].fillna(0) != 0).mean()) if raw in df else 0.0
            (self.available if share > MIN_RATE_COVERAGE else self.dropped).append(rate)
        for grp, g in df.groupby("group"):
            self.norm[grp] = {r: (g[r].mean(), g[r].std() or np.nan)
                              for r in RATE_ORDER}
        return self

    def _weights(self, grp):
        """Position weights renormalised over the rates this pool recorded."""
        w = np.array(POSITION_WEIGHTS.get(grp, POSITION_WEIGHTS["Bench"]), dtype=float)
        if self.available is not None and self.dropped:
            mask = np.array([r in self.available for r in RATE_ORDER], dtype=float)
            w = w * mask
            total = w.sum()
            # a position whose whole weight vector is unavailable falls back to an
            # equal blend of what is left rather than to all zeros
            w = w / total if total > 0 else mask / max(mask.sum(), 1)
        return w

    def _transform(self, df):
        """Apply fitted standardization -> per-match composite z-score."""
        df = df.copy()
        df["group"] = (df["position"].map(POSITION_GROUP).fillna("Bench")
                       if self.position_mode == "position_relative"
                       else pd.Series("Bench", index=df.index))
        mins = df["minutes"].clip(lower=1)
        for raw, (rate, sign) in RATE_STATS.items():
            df[rate] = sign * (df[raw].fillna(0.0) if raw in df else 0.0) / mins
        comp = np.zeros(len(df))
        for grp in df["group"].unique():
            mask = (df["group"] == grp).values
            params = self.norm.get(grp, self.norm.get("Bench"))
            w = self._weights(grp)
            zmat = np.zeros((mask.sum(), len(RATE_ORDER)))
            for j, r in enumerate(RATE_ORDER):
                mu, sd = params[r]
                col = df.loc[mask, r].values
                zmat[:, j] = 0.0 if (sd is np.nan or np.isnan(sd)) else (col - mu) / sd
            comp[mask] = zmat @ w
        df["composite"] = np.nan_to_num(comp)
        df["ratable"] = df["minutes"] >= MIN_MINUTES
        return df

    def _composite(self, df):
        """Fit standardization on df, then transform it (snapshot use)."""
        self._fit_standardization(df)
        return self._transform(df)

    # ── 2. Empirical-Bayes shrinkage ───────────────────────────────────────
    def _fit_variance_components(self, pm):
        """One-way random-effects ANOVA on composite scores -> sigma^2, tau^2.

        The assumption worth knowing about, raised by the review of 2026-09-22: this is
        a constant random intercept per player. It treats a man's ability as one number
        across every season, position and role in the data, so everything that actually
        moves — ageing, a change of position, a season spent off the bench, the recency
        weighting applied elsewhere in this very file — is booked as within-player noise
        and inflates sigma^2. An inflated sigma^2 shrinks everyone harder than they
        should be shrunk.

        It has been measured rather than left as a worry — `variance_bias.py` and
        `VARIANCE_BIAS_REPORT.md`. Letting a player's level move between seasons raises
        tau-squared by 14 to 35% depending on the competition and lowers sigma-squared by
        3 to 7%. Both point the same way: real change is being booked as noise.

        An earlier version of this note called the bias "at least conservative" on the
        grounds that it only pulls ratings toward the middle. The third external review
        of 2026-09-23 was right to object and the phrase is withdrawn. B depends on n, so
        the correction is worth about a third more shrinkage at one match and a twentieth
        at twenty-five — it therefore REORDERS players who differ in how much they have
        played, rather than compressing everyone equally. And tau divides the published
        0-100 scale, so an understated tau stretches the displayed distance between men.

        Usable for ranking within a peer pool; not a calibrated distance. The fix is a
        model that lets ability move — a player-season random effect at minimum, a
        state-space formulation properly. Not done.
        """
        pm = pm[pm["ratable"]]
        groups = [g["composite"].values for _, g in pm.groupby("player_id")
                  if len(g) >= 1]
        k = len(groups)
        N = sum(len(g) for g in groups)
        grand = np.concatenate(groups).mean()
        self.grand_mean = float(grand)

        ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        df_within = N - k
        df_between = k - 1

        ms_within = ss_within / df_within if df_within > 0 else 0.0
        ms_between = ss_between / df_between if df_between > 0 else 0.0

        # average group-size adjustment n0 (unequal n)
        sum_n2 = sum(len(g) ** 2 for g in groups)
        n0 = (N - sum_n2 / N) / df_between if df_between > 0 else 1.0

        self.sigma2 = float(ms_within)
        self.tau2 = float(max(0.0, (ms_between - ms_within) / n0)) if n0 else 0.0
        return self.sigma2, self.tau2

    def _shrink(self, player_mean, n):
        """Pull observed mean toward grand mean by B = tau2/(tau2 + sigma2/n)."""
        if self.tau2 is None:
            raise RuntimeError("fit variance components first")
        denom = self.tau2 + (self.sigma2 / n if n > 0 else np.inf)
        B = self.tau2 / denom if denom > 0 else 0.0
        return self.grand_mean + B * (player_mean - self.grand_mean), B

    @staticmethod
    def _to_0_100(z):
        """Retired. Kept only so an old pickle or notebook fails loudly, not quietly."""
        raise RuntimeError("use the instance method scale_0_100; the bare CDF treated "
                           "the composite as if it had unit variance")

    def scale_0_100(self, z, centre=None):
        """Put a shrunk composite on the published scale.

        This used to be the plain normal CDF of `z`, which assumes the composite has
        unit variance. It does not. The composite is a weighted blend of correlated
        rate z-scores, and its between-player standard deviation — tau, from the variance
        decomposition — comes out near 0.19. Feeding a quantity of that size into a
        curve expecting 1 flattened everything toward the middle, and the flattening was
        not cosmetic: a published 69 was the 99.5th percentile of its pool, a 60 was the
        92.6th, and 50 was the 56.7th rather than the median. The external review of
        2026-09-22 was right that the number could not be called a percentile, and a
        recruiter reading 62 as "somewhat above average" was reading the top few per
        cent.

        So the score is centred on the pool's own mean and expressed in units of tau:
        50 is the middle of the competition, and a point on the curve means what the
        normal curve says it means.

        What this deliberately does NOT do is rescale to the spread of the published
        estimates themselves. Shrunk estimates are less spread out than true ability,
        correctly so — that is the whole point of shrinking — and stretching them back
        out would reintroduce exactly the over-dispersion that makes the translation
        ladder lose to a constant. Dividing by tau is a change of units; dividing by the
        observed spread of the estimates would be a change of claim.

        **What the number is not.** Every competition gets its own centre and its own
        tau, and every position group its own centre, so the score says where a man sits
        among the people doing his job in his league. It is a standardised peer score.
        An NRL 60 and an NSW Cup 60 are the same standing in two different fields, not
        the same footballer — there is no common absolute scale here, and converting
        between leagues is the translation layer's job precisely because this scale
        cannot do it. The third external review of 2026-09-23 asked for that stated
        outright rather than left to be inferred, and it is right: a reader who takes
        these as comparable across competitions will make exactly the mistake the
        product exists to prevent.
        """
        tau = math.sqrt(self.tau2) if self.tau2 else 0.0
        if not tau or not np.isfinite(tau):
            return 50.0
        c = self.grand_mean if centre is None else float(centre)
        t = (float(z) - c) / tau
        return 100.0 * 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))

    # ── 3a. Snapshot ratings (all games) — for BOSC scouting ───────────────
    def compute_snapshot(self, raw=None, pm=None):
        """Snapshot ratings. Pass `pm` to supply composites computed elsewhere —
        multi-season Class needs each match standardized against ITS OWN season's
        pool, which a single _composite() call over the whole history cannot do."""
        if pm is None:
            pm = self._composite(raw)
        self._fit_variance_components(pm)
        rated = pm[pm["ratable"]]
        # "how many seasons ago" is counted from the most recent season in the data
        # being rated, so a walk-forward snapshot weights relative to its own as-of
        # season rather than to today
        ref_season = (int(rated["season"].max())
                      if len(rated) and rated["season"].notna().any() else None)

        rows = []
        for pid, g in rated.groupby("player_id"):
            g = g.sort_values(["season", "round"])
            n, class_raw = _weighted_class(g, ref_season)
            form_raw = g["composite"].tail(FORM_WINDOW).mean()
            class_z, B = self._shrink(class_raw, n)
            # form uses the recent window's own n for its shrinkage
            form_z, _ = self._shrink(form_raw, min(n, FORM_WINDOW))
            rows.append({
                "player_id": pid,
                "name": g["player"].iloc[0],
                "n_games": n,
                "raw_composite": class_raw,
                "class_z": class_z,
                "form_z": form_z,
                "divergence": form_z - class_z,
                "shrinkage_B": B,           # 0=all prior, 1=trust the data
                "group": _modal_group(g),
                "confidence": "high" if B > 0.5 else "medium" if B > 0.2 else "low",
                "rating_basis": self.position_mode,
            })
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return self.calibrate(out).sort_values("class_z", ascending=False)

    def calibrate(self, out):
        """Put a set of rated players on the published 0-100 scale.

        Separate from `compute_snapshot` because the population that defines the middle
        is the population a reader sees. The snapshot rates everyone who has ever played;
        the app publishes the men active in the latest season. Centring on the first and
        publishing the second left the middles' median at 56 while the sentence under the
        number said 50, so this is called again after the active filter.

        Idempotent: everything is recomputed from `class_z`, never from a previous score.
        """
        out = out.copy()

        # The published scale is centred on the player's own PEER GROUP, not on the
        # competition. "50 means the median of his position group" is what the app says
        # and what a recruiter reads, and centring on the competition instead left the
        # middles sitting at 58.6 while every other group sat between 40 and 45 —
        # an artefact of where bench minutes get standardised, read as middles being
        # better footballers than everyone else.
        #
        # A group too small to have a stable centre falls back to the competition, and
        # says so, rather than being centred on six players.
        # the MEDIAN, because "50 is the median of his position group" is the sentence
        # the app prints; centring on the mean leaves the median a point or two off it
        # whenever the group is skewed, which every one of them is
        centre = out.groupby("group").class_z.transform(
            lambda s: s.median() if len(s) >= MIN_GROUP_FOR_CENTRE else np.nan)
        out["scale_centre"] = centre.fillna(self.grand_mean)
        out["scale_basis"] = np.where(centre.notna(), "position group", "competition")
        for src, dst in (("class_z", "class_score"), ("form_z", "form_score"),
                         # the UNSHRUNK mean, on the same scale. It exists so an
                         # evaluation can be scored against a target that is not itself
                         # pulled toward the middle: the shrunk score is drawn toward 50,
                         # which is exactly the value one of the baselines predicts, and
                         # the external review of 2026-09-22 was right that comparing
                         # predictors against it quietly favours the shrunk ones.
                         ("raw_composite", "raw_score")):
            if src not in out.columns:
                continue
            out[dst] = [self.scale_0_100(z, c)
                        for z, c in zip(out[src], out["scale_centre"])]
        out["positional_benchmark"] = out["class_score"]

        # A genuine percentile, alongside rather than instead. The 0-100 score is a
        # monotone map that keeps the shrinkage's compression; this says outright where
        # a man sits among his peers, which is the question the score was being misread
        # as answering.
        out["class_percentile"] = (out.groupby("group").class_z
                                      .rank(pct=True, method="average") * 100.0).round(1)
        return out

    # ── 3b. Pre-match rolling ratings — leakage-safe, for GIGOT ────────────
    def compute_prematch(self, raw, pool=None):
        """Rating attached to each match, from that player's EARLIER games only.

        What is walk-forward and what is not, because the difference matters and the
        one-line version of this docstring hid it until the third external review of
        2026-09-23 said so.

        **Walk-forward:** the player history. A match's rating is built from his earlier
        matches and nothing else, and `leakage_test` permutes outcomes to prove it.

        **Not walk-forward:** the pool statistics. `_composite` fits the standardisation
        means and deviations, and `_fit_variance_components` fits sigma and tau, on
        whatever frame they are given — by default the whole of `raw`, future matches
        included. Those are population descriptors rather than outcomes, and altering
        one match moves every other row's z-score a little, so this is a small leak
        rather than a large one. It is still a leak, and calling the function
        leakage-safe without qualification was wrong.

        `pool` fixes it for a caller who needs it: pass the frame the pool statistics
        should be fitted on — typically seasons strictly before the one being rated —
        and the history walk still runs over `raw`.

        The predictive model does not come through here at all. `gigot_v2.
        prematch_players(walk_forward=True)`, which is what `teamlist_backtest.py` runs,
        fits each season's standardisation on earlier seasons and uses a plain shifted
        expanding mean with no shrinkage, so it touches neither this function nor the
        variance components. That was asserted rather than shown until
        `tests/test_gigot_walk_forward.py`, which tampers with one match and checks that
        no other match in its season moves — with the non-walk-forward branch as a
        control that does leak, so the assertion proves something.
        """
        pm = self._composite(raw if pool is None else pool)
        self._fit_variance_components(pm)
        if pool is not None:
            pm = self._transform(raw)
        pm = pm.sort_values(["player_id", "season", "round"])

        out = []
        for pid, g in pm.groupby("player_id"):
            hist = []
            for _, row in g.iterrows():
                if hist:
                    mean_prev = float(np.mean(hist))
                    z, B = self._shrink(mean_prev, len(hist))
                else:
                    z, B = self.grand_mean, 0.0  # no prior -> sit at league mean
                out.append({
                    "player_id": pid, "season": row["season"], "round": row["round"],
                    "prematch_class_z": z, "prematch_B": B, "n_prior": len(hist),
                })
                if row["ratable"]:
                    hist.append(row["composite"])
        return pd.DataFrame(out)

    # ── Leakage test ───────────────────────────────────────────────────────
    def leakage_test(self, raw, seed=0):
        """Two checks, both on FIXED standardization (so the only thing that can
        move a rating is genuine time-leakage, not a shifted normalization pool):

        (a) Definitional: each match's pre-match rating must equal the shrink of
            the mean of that player's STRICTLY EARLIER composites. Independent
            recomputation — proves the exact leak-free formula.
        (b) Future-invariance: scrambling a player's LAST game's stats must not
            change ANY earlier match's pre-match rating (the past cannot see the
            future). The last game sits in no prior window, so a leak-free engine
            is perfectly invariant.
        """
        self._fit_standardization(raw)                     # fix params once
        pm = self._transform(raw).sort_values(["player_id", "season", "round"])
        eng_out = self.compute_prematch(raw).set_index(
            ["player_id", "season", "round"])["prematch_class_z"]

        # (a) independent recomputation
        max_def_delta = 0.0
        for pid, g in pm.groupby("player_id"):
            hist = []
            for _, row in g.iterrows():
                expect = (self._shrink(float(np.mean(hist)), len(hist))[0]
                          if hist else self.grand_mean)
                got = eng_out.loc[(pid, row["season"], row["round"])]
                max_def_delta = max(max_def_delta, abs(expect - got))
                if row["ratable"]:
                    hist.append(row["composite"])

        # (b) future-invariance: scramble each player's last game, refit-free
        rng = np.random.default_rng(seed)
        scrambled = raw.copy().sort_values(["player_id", "season", "round"])
        last_idx = scrambled.groupby("player_id").tail(1).index
        for c in RATE_STATS:
            if c in scrambled:
                vals = scrambled.loc[last_idx, c].values
                scrambled.loc[last_idx, c] = rng.permutation(vals)
        pm2 = self._transform(scrambled).sort_values(
            ["player_id", "season", "round"])
        out2 = self._prematch_from_composites(pm2).set_index(
            ["player_id", "season", "round"])["prematch_class_z"]
        # every rating must be invariant: a match's own (scrambled) stats never
        # feed its rating, and last games sit in no other match's prior window.
        a, b = eng_out.align(out2, join="inner")
        max_future_delta = float((a - b).abs().max()) if len(a) else 0.0
        return max_def_delta, max_future_delta

    def _prematch_from_composites(self, pm):
        """Rolling pre-match rating given already-computed composites."""
        pm = pm.sort_values(["player_id", "season", "round"])
        out = []
        for pid, g in pm.groupby("player_id"):
            hist = []
            for _, row in g.iterrows():
                z = (self._shrink(float(np.mean(hist)), len(hist))[0]
                     if hist else self.grand_mean)
                out.append({"player_id": pid, "season": row["season"],
                            "round": row["round"], "prematch_class_z": z})
                if row["ratable"]:
                    hist.append(row["composite"])
        return pd.DataFrame(out)


def write_ratings_to_db(snapshot, comp_code="NRL", season=2026, rnd=12):
    """Persist real snapshot ratings into player_ratings (replaces mocks)."""
    con = sqlite3.connect(DB)
    out = snapshot.copy()
    out["season"] = season
    out["round"] = rnd
    out["comp_code"] = comp_code
    out["competition_translation_factor"] = 0.0
    out["updated_at"] = "2026-07-12"
    cols = ["player_id", "season", "round", "comp_code", "form_score", "form_z",
            "class_score", "class_z", "divergence", "positional_benchmark",
            "competition_translation_factor", "updated_at",
            "shrinkage_B", "n_games", "confidence"]
    out[cols].to_sql("player_ratings", con, if_exists="replace", index=False)
    con.commit()
    con.close()


def load_player_matches(comp="NRL", season=None):
    """Rows for ONE competition. The comp argument used to be accepted and ignored,
    which returned all four competitions pooled into a single rating pool."""
    con = sqlite3.connect(DB)
    cols = ["player_id", "player", "season", "round", "team", "position", "minutes"]
    cols += list(RATE_STATS.keys())
    have = pd.read_sql("PRAGMA table_info(player_match_stats)", con)["name"].tolist()
    cols = [c for c in cols if c in have]
    q = f"SELECT {', '.join(cols)} FROM player_match_stats WHERE competition = ?"
    params = [comp]
    if season is not None:
        q += " AND season = ?"
        params.append(season)
    df = pd.read_sql(q, con, params=params)
    con.close()
    return df


if __name__ == "__main__":
    # regenerate_full.py owns player_ratings. This block predates it and writes the
    # table from a single competition's pool with replace semantics.
    import sys
    if "--i-know-this-overwrites" not in sys.argv:
        sys.exit("REFUSED: regenerate_full.py is the writer of player_ratings.\n"
                 "Pass --i-know-this-overwrites to run this legacy path anyway.")
    raw = load_player_matches()
    eng = PlayerRatingEngine("NRL")
    snap = eng.compute_snapshot(raw)

    print(f"Variance components: sigma^2 (game noise) = {eng.sigma2:.3f}, "
          f"tau^2 (true talent spread) = {eng.tau2:.3f}")
    reliability = eng.tau2 / (eng.tau2 + eng.sigma2) if (eng.tau2 + eng.sigma2) else 0
    print(f"Single-game reliability tau^2/(tau^2+sigma^2) = {reliability:.2f}  "
          f"(how much of one game's signal is real talent)")
    print(f"Rated players: {len(snap)}\n")

    print("-- Shrinkage in action: biggest raw scores pulled toward the mean --")
    top_raw = snap.reindex(snap["raw_composite"].sort_values(ascending=False).index).head(8)
    show = top_raw[["name", "n_games", "raw_composite", "class_z",
                    "shrinkage_B", "class_score", "confidence"]].copy()
    show.columns = ["player", "games", "raw_z", "shrunk_z", "B", "score/100", "conf"]
    print(show.round(2).to_string(index=False))

    print("\n-- Top 10 by shrunk Class score --")
    t = snap.head(10)[["name", "n_games", "class_score", "form_score",
                       "divergence", "confidence"]].copy()
    t.columns = ["player", "games", "class", "form", "diverg", "conf"]
    print(t.round(1).to_string(index=False))

    print("\n-- Leakage test (pre-match ratings, fixed standardization) --")
    def_delta, fut_delta = eng.leakage_test(raw)
    print(f"  (a) definitional recomputation max error: {def_delta:.2e}")
    print(f"  (b) future-scramble invariance max change: {fut_delta:.2e}")
    ok = def_delta < 1e-9 and fut_delta < 1e-9
    print("  PASS (no time-leakage)" if ok else "  FAIL - leakage detected")

    write_ratings_to_db(snap, "NRL")
    print(f"\nWrote {len(snap)} real ratings to player_ratings "
          f"(replaced mock values).")
