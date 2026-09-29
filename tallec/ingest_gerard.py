# -*- coding: utf-8 -*-
"""TALLEC Phase-1 seed: ingest Gerard NRL player-stats CSVs into SQLite.

Tables:
  players            — registry with provisional player_id (name slug + first team),
                       positions seen, teams seen, first/last round
  player_match_stats — one row per player per match, raw stats normalised to numeric
                       (percentages stripped, '3.34s' -> 3.34), plus per-minute derived
Reruns are idempotent (ingest key = season/round/team/player).
"""
import pandas as pd, numpy as np, sqlite3, re, glob, os, unicodedata

BASE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.dirname(BASE)
DB = os.path.join(BASE, "tallec.db")

SOURCES = (glob.glob(os.path.join(DL, "nrl_2026_player_stats_rounds_*.csv"))
           + glob.glob(os.path.join(DL, "gerard_round12", "nrl_2026_player_stats_rounds_*.csv")))
# de-dup " (1).csv" copies
SOURCES = [f for f in SOURCES if "(1)" not in f]

def slugify(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def to_num(v):
    if pd.isna(v): return None
    s = str(v).strip()
    if s.endswith("%"): s = s[:-1]
    if s.endswith("s") and re.fullmatch(r"\d+(\.\d+)?s", s): s = s[:-1]
    s = s.replace(",", "")
    try: return float(s)
    except ValueError: return None

META = ["season","round","team","opposition","player","number","position","match_url"]

frames = []
for f in sorted(SOURCES):
    df = pd.read_csv(f)
    frames.append(df)
    print(f"read {os.path.basename(f)}: {len(df)} rows")
raw = pd.concat(frames, ignore_index=True)
raw = raw.drop_duplicates(subset=["season","round","team","player"])

stat_cols = [c for c in raw.columns if c not in META]
for c in stat_cols:
    raw[c] = raw[c].map(to_num)

raw["player_id"] = raw["player"].map(slugify)
raw["minutes"] = raw["minutes"].fillna(0)
# per-minute derived metrics (reproducible from raw — TALLEC technical principle)
for src, out in [("all_run_metres","run_metres_per_min"), ("tackles","tackles_per_min"),
                 ("p_c_m","pcm_per_min")]:
    raw[out] = np.where(raw["minutes"] > 0, raw[src] / raw["minutes"], None)

con = sqlite3.connect(DB)
raw.to_sql("player_match_stats", con, if_exists="replace", index=False)

players = (raw.groupby("player_id")
    .agg(name=("player","first"),
         teams=("team", lambda x: "; ".join(sorted(set(x)))),
         positions=("position", lambda x: "; ".join(sorted(set(str(v) for v in x if pd.notna(v))))),
         first_round=("round","min"), last_round=("round","max"),
         matches=("player","count"), total_minutes=("minutes","sum"))
    .reset_index())
players.to_sql("players", con, if_exists="replace", index=False)

# Create metadata tables (empty, ready for weekly updates)
competitions_df = pd.DataFrame({
    "comp_code": ["NRL", "SL", "NSW_Cup", "QCup"],
    "comp_name": ["National Rugby League", "Super League", "NSW Cup", "Queensland Cup"],
    "country": ["Australia", "England", "Australia", "Australia"]
})
competitions_df.to_sql("competitions", con, if_exists="replace", index=False)

# Standard positions across competitions
positions_df = pd.DataFrame({
    "position_code": ["FB", "W", "C", "H5", "HB", "P", "2R", "LK", "E", "I"],
    "position_name": ["Fullback", "Winger", "Centre", "Five-Eighth", "Halfback", "Prop", "2nd Row", "Lock", "Interchange", "Reserve"],
    "benchmark_group": ["Fullback", "Winger", "Centre", "Halves", "Halves", "Prop", "Back Row", "Back Row", "Bench", "Bench"],
    "stat_emphasis": [
        "Speed, reads, kick return",
        "Speed, line break, evasion",
        "Speed, tackle break, offload",
        "Game control, vision, kicking",
        "Game control, playmaking",
        "Strength, tackle, set-start",
        "Strength, lineout, tackle",
        "Lineout, strength, tackle",
        "Versatility",
        "Specialist cover"
    ]
})
positions_df.to_sql("positions", con, if_exists="replace", index=False)

# Player ratings (empty, weekly updates will fill this)
con.execute("""
CREATE TABLE IF NOT EXISTS player_ratings (
  player_id TEXT PRIMARY KEY,
  season INTEGER,
  round INTEGER,
  comp_code TEXT,
  form_score REAL,
  form_z REAL,
  class_score REAL,
  class_z REAL,
  divergence REAL,
  positional_benchmark REAL,
  competition_translation_factor REAL,
  updated_at TEXT,
  FOREIGN KEY(comp_code) REFERENCES competitions(comp_code)
)
""")

# Player comparisons (NRL vs SL translation matrix)
con.execute("""
CREATE TABLE IF NOT EXISTS player_comparisons (
  player_a TEXT,
  comp_a TEXT,
  player_b TEXT,
  comp_b TEXT,
  similarity_score REAL,
  translation_confidence REAL,
  notes TEXT,
  PRIMARY KEY(player_a, player_b)
)
""")

con.commit()

print(f"\ntallec.db written: {len(raw)} player-match rows, {len(players)} players")
print("\n-- data quality: fill rate of key stats --")
for c in ["minutes","all_run_metres","p_c_m","tackles","tackle_breaks","errors","fantasy"]:
    print(f"  {c:18s} {raw[c].notna().mean():5.1%}")
print("\n-- demo: top 8 by run metres per minute (min 100 total minutes) --")
q = """
SELECT p.name, p.positions, p.matches, ROUND(SUM(s.all_run_metres),0) AS run_m,
       ROUND(SUM(s.all_run_metres)/SUM(s.minutes),2) AS m_per_min
FROM player_match_stats s JOIN players p USING(player_id)
GROUP BY player_id HAVING SUM(s.minutes) >= 100
ORDER BY m_per_min DESC LIMIT 8"""
print(pd.read_sql(q, con).to_string(index=False))
print("\n-- demo: multi-position players --")
q2 = "SELECT name, positions, matches FROM players WHERE positions LIKE '%;%' LIMIT 8"
print(pd.read_sql(q2, con).to_string(index=False))
con.close()
