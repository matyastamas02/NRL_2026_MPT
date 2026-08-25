# -*- coding: utf-8 -*-
"""Headless page check for the BOSC app.

Drives all six pages for both leagues through Streamlit's own test harness and fails
on any exception. Run before pushing an app change — this is what caught a helper
that made two pages take longer than ninety seconds to render.

    python smoke_bosc.py
"""
import os
import sys

APPDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APPDIR)          # the app imports its own sibling modules
os.chdir(APPDIR)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from streamlit.testing.v1 import AppTest

APP = os.path.join(APPDIR, "bosc_app.py")
PAGES = [("🔍 Search", "Search"), ("⚖️ Compare", "Compare"),
         ("📊 Benchmarks", "Benchmarks"), ("🔄 Comparison", "Translation"),
         ("🏉 Squad (GIGOT)", "Squad"), ("📈 Trends", "Trends")]
LEAGUES = ["SL", "NRL", "NSW", "QLD"]

fails = 0
for league in LEAGUES:
    for page, label in PAGES:
        at = AppTest.from_file(APP, default_timeout=90)
        at.run()
        try:
            at.selectbox[0].select(league)          # League:
            at.selectbox[1].select(page)            # section
            at.run()
        except Exception as e:
            print(f"FAIL  {league:3s} {label:12s} driving widgets: {type(e).__name__}: {e}")
            fails += 1
            continue
        if at.exception:
            for ex in at.exception:
                print(f"FAIL  {league:3s} {label:12s} {ex.type}: {str(ex.message)[:220]}")
            fails += len(at.exception)
        else:
            n_tables = len(at.dataframe)
            n_metrics = len(at.metric)
            warn = len(at.warning) + len(at.error)
            print(f"ok    {league:3s} {label:12s} tables={n_tables} metrics={n_metrics} "
                  f"warnings={warn}")
print(f"\n{'ALL PAGES OK' if not fails else str(fails) + ' FAILURE(S)'}")
sys.exit(1 if fails else 0)
