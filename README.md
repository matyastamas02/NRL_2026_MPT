xLadder Weekly App  v2.2
========================
NO WEEKLY FILE UPLOAD NEEDED after initial setup.

ONE-TIME SETUP (5 minutes):
  1. Rename your NRL master xlsx to:   NRL_master.xlsx
     Rename your SL master xlsx to:    SL_master.xlsx
  2. Upload BOTH to the GitHub repo (alongside app.py)
  3. Deploy on share.streamlit.io -> done

WEEKLY WORKFLOW (client):
  1. Open the app URL in browser
  2. Select league (NRL or SL) in sidebar
  3. Go to "Weekly Input" tab
  4. Enter match results + stats for the round -> Submit
  5. Download:
       - xLadder PNG      -> post on socials
       - Margin PNG       -> post on socials
       - Updated Master xlsx
  6. Upload the Updated Master xlsx to GitHub
     (click the filename -> Edit -> Upload -> Commit)
  7. App auto-reloads within 1 minute. Done.

STATS TO ENTER PER MATCH:
  NRL (4 stats per team):
    PTB - Strong Tackle | Kick Chase - Good Chase
    Receipt - Falcon | Kick - Crossfield

  SL (10 stats per team):
    PTB - Strong Tackle | Kick Chase - Good Chase
    Receipt - Falcon | Kick - Crossfield
    Set Complete - Total | Tackle Break
    PTB - Won | Ball Run - Restart Return
    Line Break | Pre-Contact Metres

ELO: computed automatically (K=27). No manual input.

LOCAL RUN:
  pip install -r requirements.txt
  streamlit run app.py
