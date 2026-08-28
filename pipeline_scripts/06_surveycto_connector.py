"""
SurveyCTO REST API connector.

Built on the `pysurveycto` library (by IDinsight) rather than hand-rolled
HTTP calls. SurveyCTO's actual API is easy to get subtly wrong by hand -
the CSV export path is `/api/v1/forms/data/wide/csv/{form_id}?r=...`
(v1, not v2; requires a review-status query param), authentication falls
back from Basic to Digest depending on server version, and a few other
details that a maintained library gets right more reliably than a first
pass at the docs would. See https://github.com/IDinsight/surveycto-python

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

Once those are set, 09_realtime_watcher.py and 11_debrief_app.py both pick
this up automatically (see their docstrings) - the rest of the pipeline
(fatigue, duplicates, outliers, quality score, comments) runs unchanged,
since it only depends on getting a wide-format pandas DataFrame with the
same column names the synthetic data already uses.

Note on "incremental" pulls: SurveyCTO's CSV export does not support a
server-side date filter (only its JSON export does, and only in a
different, non-wide shape) - so every pull below fetches the form's FULL
current submission history, not just new rows. run_incremental() merges
that against the local raw export by 'KEY' (the submission ID) client
side, so no submission is ever lost or duplicated either way - it's just
a less efficient round trip than a true incremental pull would be for a
very large form. Fine at this survey's scale; worth knowing if the form
grows into the tens of thousands of submissions.
"""
import os
import tempfile
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

try:
    from pysurveycto import SurveyCTOObject
except ImportError:
    SurveyCTOObject = None

SERVER_NAME = os.environ.get("SCTO_SERVER", "")   # the part before .surveycto.com
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
    if SurveyCTOObject is None:
        raise RuntimeError("The 'pysurveycto' package is required - pip install pysurveycto")


def _client() -> "SurveyCTOObject":
    return SurveyCTOObject(SERVER_NAME, USERNAME, PASSWORD)


def fetch_submissions_df(since: datetime | None = None) -> pd.DataFrame:
    """Pull the form's current full set of submissions as a wide-format
    DataFrame with the same columns the rest of the pipeline expects.
    `since` is accepted for interface compatibility with callers but not
    applied server-side - see the module docstring for why.
    """
    _require_credentials()
    client = _client()
    try:
        csv_text = client.get_form_data(FORM_ID, format="csv", shape="wide")
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status == 401:
            raise RuntimeError(
                f"SurveyCTO rejected the login for '{USERNAME}' on server '{SERVER_NAME}' "
                f"(401 Unauthorized). Double-check SCTO_USERNAME/SCTO_PASSWORD, and that this "
                f"login has 'Allow server API access' enabled (Console > Users)."
            ) from e
        if status == 404:
            raise RuntimeError(
                f"SurveyCTO couldn't find form '{FORM_ID}' on server '{SERVER_NAME}' "
                f"(404 Not Found). Double-check SCTO_FORM_ID matches the deployed form's ID exactly."
            ) from e
        raise
    df = pd.read_csv(StringIO(csv_text), dtype=str, keep_default_na=False)
    return df


def run_incremental(state_path=str(BASE / "output" / "_last_pull.txt")):
    """Pull the current full submission set and merge it into the running
    raw export by 'KEY', so re-running never duplicates or drops a row."""
    df_new = fetch_submissions_df()
    if len(df_new) == 0:
        print("No submissions returned from the server.")
        return

    out_path = str(BASE / "output" / "msy_listing_raw_export.csv")
    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        existing_keys = set(df_existing["KEY"]) if "KEY" in df_existing.columns else set()
        n_new = int((~df_new["KEY"].isin(existing_keys)).sum()) if "KEY" in df_new.columns else len(df_new)
        df_all = pd.concat([df_existing, df_new]).drop_duplicates(subset=["KEY"], keep="last")
    else:
        n_new = len(df_new)
        df_all = df_new
    df_all.to_csv(out_path, index=False)

    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        f.write(datetime.now().isoformat())

    print(f"Fetched {len(df_new)} submission(s) from the server ({n_new} new); {len(df_all)} total on file.")


if __name__ == "__main__":
    print(__doc__)
    try:
        _require_credentials()
        print(f"Credentials found for server '{SERVER_NAME}', form '{FORM_ID}'. Running one pull...")
        run_incremental()
    except RuntimeError as e:
        print(f"\nNot configured yet: {e}")
