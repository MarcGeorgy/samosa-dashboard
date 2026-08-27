# MSY Listing Survey - Data Quality Dashboard

Deployable copy of the pipeline_scripts from the "RST 2026 - Real Time Survey
Automation - Innovation Fair" project (kept as your main working copy in
Google Drive). This folder is a plain git repo so it can be pushed to GitHub
and deployed on Streamlit Community Cloud - git and Drive's live file sync
don't get along well, so this is intentionally a separate copy, not a Drive
folder.

**Keeping the two in sync:** whenever pipeline_scripts changes in the Drive
folder, re-copy the changed .py files here and commit/push. There is no
automatic sync between the two locations.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (private repo recommended - see below).
2. Go to https://share.streamlit.io, sign in with GitHub, click "New app".
3. Pick this repo/branch, and set **Main file path** to
   `pipeline_scripts/11_debrief_app.py`.
4. Under **Advanced settings > Secrets**, add any of these you want live
   (never commit them to the repo):
   ```
   ANTHROPIC_API_KEY = "..."
   SCTO_SERVER = "jpalmena"
   SCTO_FORM_ID = "msy_listing_survey_test"
   SCTO_USERNAME = "..."
   SCTO_PASSWORD = "..."
   ```
   Without these, the app runs in its offline demo mode (stub LLM analysis,
   simulated data) - it still works, just not against real SurveyCTO data or
   real Claude analysis.
5. Deploy. You'll get a permanent URL like `https://<something>.streamlit.app`.

**Storage note:** Streamlit Cloud's disk is ephemeral - it can reset on
sleep/wake or redeploy. The app bootstraps its own baseline demo data
automatically on a cold start, so it never just errors out, but anything
written only to local disk (corrections log, alerts log, LLM cache, any
locally-simulated test submissions not pulled from a real SurveyCTO server)
can be lost on a reset. Real SurveyCTO submissions are safe either way -
they live on your SurveyCTO server, not on Streamlit's disk, and get
re-pulled on the next refresh.

## Local development

```
pip install -r requirements.txt
python pipeline_scripts/00_metadata.py
python pipeline_scripts/04_generate_synthetic_data.py
python pipeline_scripts/05_monitoring_pipeline.py
streamlit run pipeline_scripts/11_debrief_app.py
```
