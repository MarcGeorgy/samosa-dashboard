"""
SurveyCTO REST API connector.

Before this can pull real data:
  1. In your SurveyCTO console (https://<server>.surveycto.com/console),
     upload and deploy 01_msy_listing_survey.xlsx as a NEW, clearly-labeled
     TEST form - e.g. title "S.A.M.O.S.A Listing Survey - TEST" and form ID
     "msy_listing_survey_test" - so it can never be confused with a real
     production deployment. The form ID you choose becomes SCTO_FORM_ID below.
  2. Create/use a SurveyCTO login with API access (Console > Users) for
     SCTO_USERNAME / SCTO_PASSWORD.
  3. Set these as environment variables in your own shell - never paste
     credentials into a script or into chat with an assistant:
       SCTO_SERVER=<the part before .surveycto.com>
       SCTO_FORM_ID=msy_listing_survey_test
       SCTO_USERNAME=<your API-enabled login>
       SCTO_PASSWORD=<its password>

Once those are set, swap `load_data(IN_PATH)` in 05_monitoring_pipeline.py
for `fetch_submissions_df(...)` below (or just use 09_realtime_watcher.py,
which does this automatically when SCTO_* env vars are present - see its
docstring) and the rest of the pipeline (fatigue, duplicates, outliers,
quality score, comments) runs unchanged - it only depends on getting a
pandas DataFrame with the same column names.

Docs: https://docs.surveycto.com/05-exporting-and-publishing-data/
      02-api-access/01.api-access.html
"""
import io
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

SERVER_NAME = os.environ.get("SCTO_SERVER", "")   # https://{SERVER_NAME}.surveycto.com
FORM_ID = os.environ.get("SCTO_FORM_ID", "msy_listing_survey_test")
USERNAME = os.environ.get("SCTO_USERNAME", "")   # SurveyCTO login (server dataset/API user recommended)
PASSWORD = os.environ.get("SCTO_PASSWORD", "")   # set as an env var in your own shell, never hard-code

BASE = Path(tempfile.gettempdir()) / "mvsy_monitoring"


def _require_credentials():
    missing = [name for name, val in [("SCTO_SERVER", SERVER_NAME), ("SCTO_USERNAME", USERNAME),
                                       ("SCTO_PASSWORD", PASSWORD)] if not val]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s): {', '.join(missing)}. Set them in your own shell "
            f"before running this - see this file's module docstring for what each one is."
        )


def fetch_submissions_df(since: datetime | None = None) -> pd.DataFrame:
    """Pull all (or incremental, via `since`) submissions for FORM_ID as a
    wide-format CSV and return it as a DataFrame with the same columns the
    rest of the pipeline expects.

    For true real-time monitoring, run this on a schedule (e.g. every
    5-15 minutes via cron, GitHub Actions, or a small always-on worker) and
    persist `since` = the max SubmissionDate seen on the previous run, so
    each pull only fetches new submissions. SurveyCTO also supports
    server-side webhooks (Data > Publish) that can push new submissions
    directly to a listening endpoint if you'd rather not poll.
    """
    _require_credentials()
    url = f"https://{SERVER_NAME}.surveycto.com/api/v2/forms/data/csv/{FORM_ID}"
    params = {}
    if since is not None:
        # SurveyCTO's API accepts a `date` filter (submissions on/after this date);
        # for finer-grained incremental pulls, filter client-side on SubmissionDate
        # after fetching, or track the max KEY/instanceID already processed.
        params["date"] = since.strftime("%b %d, %Y %I:%M:%S %p")

    resp = requests.get(url, params=params, auth=(USERNAME, PASSWORD), timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), dtype=str, keep_default_na=False)
    return df


def run_incremental(state_path=str(BASE / "output" / "_last_pull.txt")):
    """Example of the incremental polling loop a scheduled job would run."""
    last_pull = None
    if os.path.exists(state_path):
        with open(state_path) as f:
            last_pull = datetime.fromisoformat(f.read().strip())

    df_new = fetch_submissions_df(since=last_pull)
    if len(df_new) == 0:
        print("No new submissions since last pull.")
        return

    # append to the running raw export the rest of the pipeline reads from
    out_path = str(BASE / "output" / "msy_listing_raw_export.csv")
    if os.path.exists(out_path) and last_pull is not None:
        df_existing = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        df_all = pd.concat([df_existing, df_new]).drop_duplicates(subset=["KEY"], keep="last")
    else:
        df_all = df_new
    df_all.to_csv(out_path, index=False)

    with open(state_path, "w") as f:
        f.write(datetime.now().isoformat())

    print(f"Pulled {len(df_new)} new submission(s); {len(df_all)} total on file.")


if __name__ == "__main__":
    print(__doc__)
    try:
        _require_credentials()
        print(f"Credentials found for server '{SERVER_NAME}', form '{FORM_ID}'. Running one pull...")
        run_incremental()
    except RuntimeError as e:
        print(f"\nNot configured yet: {e}")
