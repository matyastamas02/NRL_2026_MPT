# Review brief — paste this as your prompt

---

You are the sixth independent reviewer of a working sports-analytics prototype. Five
reviews have run before you and the faults they found have been fixed; this round is
different. The project now makes a specific set of claims to its client, and I want them
checked against the evidence and judged against what the project is for.

**Read `00_START_HERE.md` first.** Section 1 is the project's goals, quoted from the
client's scope of work. Section 4 lists the claims, C1–C14, with the code behind each.
Section 6 lists the weaknesses we already know about; confirming one with a sharper
argument is useful, presenting it as a discovery is not.

Then run `python reproduce.py` (pandas and numpy only, reads `data/`). It recomputes
C1–C9. If any number disagrees with `docs/what_is_live.html` or `results/`, report that
first.

---

## Part A — is each claim true?

For every claim C1–C14, give one verdict:

- **Supported** — the evidence shows it, and the wording does not go beyond it;
- **Supported but overstated** — true in substance, but the wording claims more than the
  evidence carries; give the wording you would use;
- **Not supported** — the evidence does not show it;
- **Wrong** — the evidence shows something else.

Say why in a sentence or two, and how I can check you. Look hardest at these, because
they are new and nobody outside the project has examined them:

1. **The noise floor** (`noise_floor.py`, C4–C6). Is simulating σ²/n noise through the
   engine's shrinkage and Φ map, with the true level at the posterior mean, a valid way
   to bound the error of any forecast? Is σ² from a within-season one-way decomposition
   the right noise, given that it counts within-season form as noise? Does the
   split-half ratio of 0.92 actually validate it?
2. **The less noisy comparisons** (C5). Is subtracting the simulated noise variance from
   each squared error a valid estimate of error against the true level? Does
   restricting to players with many matches introduce a selection that matters here?
3. **The stayers reference** (C7). The project owner has already objected that a
   forecast from a player's own previous season also lacks information about how he
   develops between seasons, so it is not a ceiling. Does the note's sentence "what
   changes between two seasons is not in the ratings" survive that objection?
4. **The feature tests** (`team_role_trend.py`, C8–C10). Is a shared-coefficient linear
   correction to the line's residuals a fair test of team context, role and trend, or
   could the null result be an artefact of the design: coefficients shared across
   directions, training pairs built differently from the evaluation cohort, missing-value
   handling, twenty comparisons, power? Is any feature leaking information from the
   season being forecast beyond the one labelled look-ahead? Are the role proxies a
   reasonable reading of "expected role", or a different quantity?
5. **The starts rule** (C9). Validated on match-sheet competitions, applied to NRL
   2021–26 where no sheet exists. Is that transfer safe?
6. **The switch to the line** (C3). Given C1 and C2, is forecasting moves into Super
   League with a two-parameter line the right call? Is its band — 1.96 × the in-sample
   residual SD, about ±40 points — honest, and is that how it should be shown?

## Part B — do the claims serve the project's goals?

Answer as an advisor who has read the evidence, not only as an auditor.

1. **Does the note represent the evidence fairly for its reader?** The client will use
   it to demonstrate the product to a Super League club. What is overstated,
   understated, or missing? What would a club analyst reject on first reading?
2. **The scope calls translation "the key modelling task for Leeds".** The best
   evidenced translation into Super League is now a two-parameter line whose individual
   band is about as wide as the whole spread of ratings. What is a defensible deliverable
   for the demo: a band, a tier, a probability of reaching a regular role, a comparison
   set of similar past movers, something else? Argue for one.
3. **What would most plausibly close the roughly 8 points above the noise floor**, using
   data the client could realistically obtain: the teams named before each round, Super
   League squads and squad numbers at the start of the season (the data hold every
   actual line-up but nothing known beforehand), signing and contract dates,
   Championship data, scouts' priors. How would you
   test it with about 90 moves into Super League a year, and what should be fixed in
   advance for 2026 (exploratory) and 2027 (confirmatory)?
4. **Scope items that are not met or are at risk.** Check the build against section 1
   of `00_START_HERE.md`. Examples to verify, not conclusions: the scope asks for Form
   over "3 & 5 game rolling averages" and the engine uses five; it defines Divergence as
   "% over or under" and the engine reports Form minus Class on the composite scale; it
   asks for "multiple positional ratings for the same player", which the handover lists
   as not built. Say which gaps matter for a demo and which do not.

---

## Constraints to respect

- **2027 is the first clean confirmatory holdout.** Everything fitted now is exploratory
  until then; say where a recommendation needs a specification fixed in advance.
- **Samples.** 90 moves into Super League over three origins; 441 entry moves in all.
- **No squad, contract or registration data** today; the client may be able to get some.
- **A prototype on a commercial timetable**, not a research programme. If something
  takes months, say so plainly.
- The data are licensed Stats Perform rows supplied under a client agreement; use them
  only for this review.

## What I want back

1. **A table of C1–C14** with your verdict, a one-line reason, and for anything not
   "Supported" the wording you would put in the note instead.
2. **Problems ranked by how much they change a decision**, each with file, function or
   line, and how I can check you are right.
3. **Recommended next steps**, in order, with what each buys and what it costs. Be
   willing to recommend stopping or descoping.
4. **What you would tell the client this week**, in two or three sentences a club would
   understand, using only what is evidenced.
5. **What you could not assess**, and what you would need to.

Where a judgement is arguable, give both sides and then commit to one. Do not pad: if
the right answer to a question is short, a short answer is the one I want.
