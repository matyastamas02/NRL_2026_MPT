# Review brief — paste this as your prompt

---

You are the seventh independent reviewer of a sports-analytics prototype. The sixth
review found that ten of the fourteen claims the project made to its client went beyond
the evidence; every finding was checked and acted on. This round is narrower and has a
deadline built into it: a test specification has to be frozen before the next season's
data arrives.

**Read `00_START_HERE.md` first**, then `RESPONSE_TO_R6.md`. Run `python reproduce.py`
(pandas and numpy only, reads `data/`); it recomputes C1–C9 from last round and R1–R4
from this one, and prints the figure each is checked against. Report any disagreement
before anything else.

---

## Part A — did the response fix what the sixth review found?

For every row of `RESPONSE_TO_R6.md`, say whether the action is **adequate**,
**partial** (say what is missing) or **inadequate**, and whether anything the response
says it did is not actually true in the code or the note. Look in particular at:

- the note's new wording against the sixth review's proposed wording, where they differ;
- `predict_translation._line`: does the app's fallback now match what the backtest
  scored, and is anything left inconsistent between them;
- the rows marked open: is leaving each one open acceptable for a client demo?

## Part B — is the 2026 test ready to freeze?

Section 5 of `00_START_HERE.md` and `code/retest_r6.py`. Treat it as a pre-registration
and review it as one:

1. **Design.** Is anything in the re-test leaking information from the season being
   forecast — `has_history`, the first club seen in the target season, the incumbents'
   ratings, the cohort itself? Is a joint fit with OLS line terms and a fixed ridge on the
   candidate the right specification? Is `has_history` measuring what its name says, or
   something else (for example the amount of evidence behind the source rating)?
2. **The two hypotheses.** H1 (`has_history`, into Super League) was found on the same 90
   moves it is reported on. H2 (incumbents, all moves) is a secondary result. Are these
   the right things to carry forward, and is anything that should be tested missing?
3. **What to fix before the data arrives.** Propose, concretely, the frozen version:
   the primary endpoint, what counts as success (a threshold, an interval rule, or both),
   whether 2026 is tested alone or pooled, the minimum cohort size below which the result
   is declared undecided, how multiplicity is handled, and how NRL 2026 match-sheet
   positions, expected from the client, enter without changing the specification. A few
   lines that could be committed as-is are the most useful answer.
4. **The weaker line.** The re-test's cohort-trained line is less accurate than the
   shipped pair-trained line (18.77 against 17.72 MAE into Super League). Should the
   frozen test use the shipped construction, the cohort construction, or both, and what
   does each choice mean for interpreting a gain?

## Part C — is the comparison card a sound demo deliverable?

`predict_translation.comparables` and the translation page in `code_context/bosc_app.py`.
`reproduce.py` §R3 reimplements its rule. Judge:

1. **The rule:** same direction, next season, within 7.5 rating points, same position
   group when eight remain, window doubled once, otherwise "not enough data", the player
   excluded. Is it defensible, and what would you change before a club sees it?
2. **What it hides.** Only movers with three or more matches in the new competition are
   included. The card says so in a sentence. Is that enough, or does the card need a
   denominator — how many moved and did not reach three matches — before it can be shown?
3. **Presentation.** Median, middle half and 10–90% range with the list of names. What
   would a club analyst misread? Should the line forecast stay beside it, below it, or go?

## Part D — the note's current claims

For N1–N13 in `00_START_HERE.md` §4, give the same verdicts as last round:
**Supported**, **Supported but overstated** (with the wording you would use), **Not
supported**, or **Wrong**, with a one-line reason and how to check.

---

## Constraints

- **2026 is exploratory, 2027 confirmatory.** Nothing may be tuned on 2027.
- **Samples:** about 30 moves into Super League a season; 90 over 2023–25; 441 entry
  moves in all.
- **No squad, contract or registration data yet**; the client has been asked for named
  teams, squads with squad numbers, signing dates and, if possible, a role band recorded
  at signing.
- **A prototype on a commercial timetable.** Say plainly if something takes months.
- The data are licensed Stats Perform rows; use them only for this review.

## What I want back

1. **Part A table:** each row of `RESPONSE_TO_R6.md` with adequate / partial /
   inadequate and a one-line reason.
2. **The frozen 2026 specification**, as text that could be committed, plus anything in
   `retest_r6.py` that must change before freezing.
3. **The card:** keep, change (how), or hold back from the demo.
4. **Part D table** of N1–N13.
5. **Anything that should stop the client seeing the note** in its current form.
6. **What you could not assess.**

Where a judgement is arguable, give both sides and commit to one. If the right answer is
short, give the short answer.
