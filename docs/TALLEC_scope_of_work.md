TALLEC\
Technical Scope of Work (Prototype)\
Prepared for: Tamas
====================================

**\
Project Goal\**
Build a working prototype of TALLEC (Total All Expected Contributions),
the underlying rugby league intelligence engine. The priority is to
prove the concept rather than build a finished commercial product.\
\
The prototype will support two independent applications:\
\
• BOSC – recruitment and player intelligence platform (for demonstration
to Leeds Rhinos).\
• GIGOT v2 – internal predictive model (completely separate from the
Leeds project).

This prototype is intended to demonstrate the concept to Leeds Rhinos
and validate the underlying modelling. If successful, the platform will
be redeveloped later into a production application with additional
funding and development resources.

The priority is therefore to build a technically sound MVP that proves
the concept, rather than a finished commercial product.

## Phase 1 – TALLEC Database

- Import all available Stats Perform player and team data from NRL,
  Super League, NSW Cup and Queensland Cup (Mike to provide)

- Create permanent Player IDs across clubs and competitions.

- Store every raw statistic and metadata field.

- Track age, minutes, games played, competition history and positional
  history.

- Create derived metrics (per minute, per run, percentages, rolling
  averages and career averages) (Mike to provide framework)

- Create entry point for weekly data imports and updates.

## Phase 2 – BOSC Prototype

- Create benchmarking by competition, season and position, based on
  score of 0-100 when 50 is average.

- Build Class Rating (since start of database), Form Rating (3 & 5 game
  rolling averages) and Divergence Rating (How far is Form from Class as
  % over or under).

- Develop Competition Translation models to estimate player performance
  across competitions. **This is the key modelling task for Leeds as
  they want to estimate how a player in NRL or NSW Cup will go in SL.**

- Support multiple positional ratings for the same player (e.g. centre
  and wing).

- Build a simple Streamlit prototype with player search, player profile
  and ratings.

- Focus on a usable proof of concept rather than polished UI.

## Phase 3 – GIGOT v2

- Integrate TALLEC ratings into the existing GIGOT workflow.

- Combine team-level metrics with player-level ratings, giving us 5
  inputs – Team Form, Team Class (which we already have), plus Player
  Form, Player Class (combined to create a collective rating based on
  team strength that week)

- Integrate a new varial, Contribution Rating, which takes Player stats
  as a % of Team Stats, so we can track Expected Contribution based on
  Team List.

- Back-test against historical results.

- Output winner probabilities, margin predictions and confidence.

- Allow updates to TALLEC to feed through to this every week

- If you want to do even more regression, we now have double the match
  data, as this now includes more seasons of SL&NRL plus the equivalent
  seasons of second grade

## Technical Principles

- Store all raw data permanently.

- Derived metrics must always be reproducible from raw data.

- Keep architecture modular and scalable.

- Rating formulae should be configurable without changing application
  code.

- Design the prototype so it can later be rebuilt into a production
  platform.

- Every calculation and rating should be version-controlled so
  methodology changes do not overwrite historical results.

## Quote Request

- Estimate cost and development time for each phase separately.

- Recommend the best technical architecture.

- Suggest where the MVP can be simplified while protecting the long-term
  vision.

- Identify technical risks or dependencies.

  Please challenge any assumptions that you think could be improved. I
  am looking for both a development partner and technical advice on the
  best architecture.
