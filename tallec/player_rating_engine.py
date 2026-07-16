# -*- coding: utf-8 -*-
"""TALLEC player rating engine — Form, Class, Divergence.

Replaces the mock ratings with defensible statistics. Three ideas do the work:

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
    "Prop":      [0.29, 0.24, 0.05, 0.04, 0.24, 0.05, 0.00, 0.04, 0.05],
    "Back Row":  [0.19, 0.14, 0.10, 0.09, 0.24, 0.14, 0.00, 0.05, 0.05],
    "Bench":     [0.15, 0.12, 0.12, 0.10, 0.20, 0.12, 0.06, 0.08, 0.05],
}

# Gerard position string -> benchmark group
POSITION_GROUP = {
    "Fullback": "Fullback", "Winger": "Winger", "Centre": "Centre",
    "Five-Eighth": "Halves", "Halfback": "Halves", "Hooker": "Hooker",
    "Prop": "Prop", "2nd Row": "Back Row", "Lock": "Back Row",
    "Interchange": "Bench", "Reserve": "Bench",
}

MIN_MINUTES = CONFIG["data_quality_thresholds"]["min_minutes_for_form"]  # 20
FORM_WINDOW = CONFIG["form_calculation"]["window_matches"]                # 5


class PlayerRatingEngine:
    def __init__(self, comp_code="NRL"):
        self.comp = comp_code
        self.sigma2 = None   # within-player (game-to-game) variance
        self.tau2 = None     # between-player (true talent) variance
        self.grand_mean = 0.0

    # ── 1. Standardization (fit / transform split) ────────────────────────
    # Standardization params are POPULATION descriptors ("what's an average
    # prop"), not outcome data — fitting them on the full pool is a design
    # choice, and separating fit from transform lets us fit on train and apply
    # to test unchanged once real multi-season data arrives.
    def _fit_standardization(self, df):
        df = df.copy()
        df["group"] = df["position"].map(POSITION_GROUP).fillna("Bench")
        mins = df["minutes"].clip(lower=1)
        self.norm = {}  # group -> rate -> (mean, std)
        for raw, (rate, sign) in RATE_STATS.items():
            df[rate] = sign * (df[raw].fillna(0.0) if raw in df else 0.0) / mins
        for grp, g in df.groupby("group"):
            self.norm[grp] = {r: (g[r].mean(), g[r].std() or np.nan)
                              for r in RATE_ORDER}
        return self

    def _transform(self, df):
        """Apply fitted standardization -> per-match composite z-score."""
        df = df.copy()
        df["group"] = df["position"].map(POSITION_GROUP).fillna("Bench")
        mins = df["minutes"].clip(lower=1)
        for raw, (rate, sign) in RATE_STATS.items():
            df[rate] = sign * (df[raw].fillna(0.0) if raw in df else 0.0) / mins
        comp = np.zeros(len(df))
        for grp in df["group"].unique():
            mask = (df["group"] == grp).values
            params = self.norm.get(grp, self.norm.get("Bench"))
            w = np.array(POSITION_WEIGHTS.get(grp, POSITION_WEIGHTS["Bench"]))
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
        """One-way random-effects ANOVA on composite scores -> sigma^2, tau^2."""
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
        """Normal CDF -> 0..100 benchmark scale (median z=0 -> 50)."""
        return 100.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # ── 3a. Snapshot ratings (all games) — for BOSC scouting ───────────────
    def compute_snapshot(self, raw):
        pm = self._composite(raw)
        self._fit_variance_components(pm)
        rated = pm[pm["ratable"]]

        rows = []
        for pid, g in rated.groupby("player_id"):
            g = g.sort_values(["season", "round"])
            n = len(g)
            class_raw = g["composite"].mean()
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
                "class_score": self._to_0_100(class_z),
                "form_score": self._to_0_100(form_z),
                "positional_benchmark": self._to_0_100(class_z),
                "confidence": "high" if B > 0.5 else "medium" if B > 0.2 else "low",
            })
        return pd.DataFrame(rows).sort_values("class_z", ascending=False)

    # ── 3b. Pre-match rolling ratings — leakage-safe, for GIGOT ────────────
    def compute_prematch(self, raw):
        """Rating attached to each match uses only that player's EARLIER games."""
        pm = self._composite(raw)
        self._fit_variance_components(pm)
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


def load_player_matches(comp="NRL"):
    con = sqlite3.connect(DB)
    cols = ["player_id", "player", "season", "round", "team", "position", "minutes"]
    cols += list(RATE_STATS.keys())
    have = pd.read_sql("PRAGMA table_info(player_match_stats)", con)["name"].tolist()
    cols = [c for c in cols if c in have]
    df = pd.read_sql(f"SELECT {', '.join(cols)} FROM player_match_stats", con)
    con.close()
    return df


if __name__ == "__main__":
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
