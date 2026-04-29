xLadder Weekly App  v2.1
========================

FILES:
  app.py                 Streamlit web interface
  xladder_pipeline.py    Core model (import this)
  requirements.txt       Python dependencies

HOW TO RUN:
  pip install -r requirements.txt
  streamlit run app.py
  -> opens at http://localhost:8501

DEPLOY FREE ONLINE (Streamlit Community Cloud):
  1. Put all 3 files in a GitHub repo
  2. share.streamlit.io -> New app -> connect repo
  3. Share the URL with the client

WEEKLY WORKFLOW FOR CLIENT:
  1. Open the app
  2. Upload the master xlsx (from previous week)
  3. Go to "Weekly Input" tab
  4. Enter match results + stats for each game
  5. Click "Submit Round"
  6. Download:
     - xLadder PNG       -> post on socials
     - Margin PNG        -> post on socials
     - Updated Master xlsx -> use this next week as upload file
  7. Go to "Next Round Predictions" tab
  8. Enter betting lines -> click Calculate Edge

WHAT STATS TO ENTER (from your stats programme):
  NRL (4 stats per team):
    PTB - Strong Tackle
    Kick Chase - Good Chase
    Receipt - Falcon
    Kick - Crossfield

  SL (10 stats per team):
    PTB - Strong Tackle
    Kick Chase - Good Chase
    Receipt - Falcon
    Kick - Crossfield
    Set Complete - Total
    Tackle Break
    PTB - Won
    Ball Run - Restart Return
    Line Break
    Pre-Contact Metres

ELO IS AUTOMATIC:
  The app carries ELO forward from the master file
  using K=27 (derived from 2022-2026 data).
  No manual ELO input needed.
