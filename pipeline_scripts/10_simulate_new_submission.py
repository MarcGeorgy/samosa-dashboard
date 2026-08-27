"""
Demo helper: appends N new rows to output/msy_listing_raw_export.csv to
simulate SurveyCTO submissions arriving live, so 09_realtime_watcher.py has
something new to detect. Not part of the production pipeline - there is no
real server to poll for this exercise (see 06_surveycto_connector.py).

Each new row is a mutated copy of an existing submission (same village /
enumerator pool, so it stays consistent with villages.json/enumerators.json)
with a fresh KEY and a later timestamp. Pass --inject-issue to force one of
the pipeline's checks to fire, so the watcher's alerting has something to
catch on demand:

    python 10_simulate_new_submission.py --n 3 --inject-issue duplicate
"""
import argparse
import random
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

fake = Faker("en_IN")

OUT_DIR = Path(tempfile.gettempdir()) / "mvsy_monitoring" / "output"
RAW_EXPORT_PATH = OUT_DIR / "msy_listing_raw_export.csv"

DT_FMT = "%b %d, %Y %I:%M:%S %p"

ISSUE_CHOICES = ["none", "duplicate", "short_duration", "missing_gps", "outlier_income",
                  "off_hours", "super_fast", "stale_version"]
STALE_FORM_VERSION = "2027020100"   # an older version string than the current "2027021500"


def _fmt(dt):
    return dt.strftime(DT_FMT)


def _jitter_latlon(lat, lon, meters):
    deg = meters / 111_000
    return lat + random.uniform(-deg, deg), lon + random.uniform(-deg, deg)


def make_row(template: pd.Series, issue: str) -> dict:
    row = template.to_dict()
    row["KEY"] = f"uuid:{uuid.uuid4()}"

    orig_end = datetime.strptime(template["endtime"], DT_FMT)
    new_start = orig_end + timedelta(minutes=random.uniform(20, 90))
    duration = random.uniform(9.0, 15.0)
    new_end = new_start + timedelta(minutes=duration)
    new_submission = new_end + timedelta(minutes=random.uniform(2, 30))

    row["starttime"] = _fmt(new_start)
    row["endtime"] = _fmt(new_end)
    row["today"] = new_start.strftime("%b %d, %Y")
    row["SubmissionDate"] = _fmt(new_submission)
    row["hh_outcome"] = "completed"
    row["hh_number"] = str(int(template.get("hh_number") or 1) + random.randint(100, 999))
    row["hh_id"] = f"{row['village']}-{row['hh_number']}"
    row["enumerator_comments"] = ""

    if issue != "duplicate":
        # every other issue type represents a genuinely NEW household - give it
        # its own name and a GPS point well outside the duplicate-matcher's
        # 60m radius, so it doesn't accidentally also register as a fuzzy
        # duplicate of whatever template it was cloned from
        if row.get("hh_head_name"):
            row["hh_head_name"] = fake.name_female() if random.random() < 0.15 else fake.name_male()
        try:
            lat, lon = float(row["gps_location_Latitude"]), float(row["gps_location_Longitude"])
            lat, lon = _jitter_latlon(lat, lon, random.uniform(300, 3000))
            row["gps_location_Latitude"] = round(lat, 6)
            row["gps_location_Longitude"] = round(lon, 6)
        except (ValueError, TypeError):
            pass  # template had no GPS to jitter from

    if issue == "duplicate":
        # exact duplicate of the template household - keep hh_id identical
        row["hh_number"] = template["hh_number"]
        row["hh_id"] = template["hh_id"]
        row["enumerator_comments"] = "Re-visited this household, wanted to confirm details from earlier."
    elif issue == "short_duration":
        new_end = new_start + timedelta(minutes=random.uniform(1.5, 3.5))
        row["endtime"] = _fmt(new_end)
    elif issue == "missing_gps":
        row["gps_location_Latitude"] = ""
        row["gps_location_Longitude"] = ""
    elif issue == "outlier_income":
        row["monthly_income"] = str(random.choice([185000, 199000, 210000]))
    elif issue == "off_hours":
        # push the start time to somewhere between 8pm and 5am, same calendar day logic kept simple
        off_hour = random.choice(list(range(20, 24)) + list(range(0, 6)))
        new_start = new_start.replace(hour=off_hour, minute=random.randint(0, 59))
        new_end = new_start + timedelta(minutes=random.uniform(9.0, 15.0))
        row["starttime"] = _fmt(new_start)
        row["endtime"] = _fmt(new_end)
        row["today"] = new_start.strftime("%b %d, %Y")
    elif issue == "super_fast":
        new_end = new_start + timedelta(minutes=random.uniform(0.4, 1.8))
        row["endtime"] = _fmt(new_end)
    elif issue == "stale_version":
        row["formdef_version"] = STALE_FORM_VERSION

    return row


def simulate(n: int = 2, issue: str = "none", seed: int | None = None, all_rows: bool = False):
    if seed is not None:
        random.seed(seed)
    if not RAW_EXPORT_PATH.exists():
        raise SystemExit(f"{RAW_EXPORT_PATH} not found - run 04_generate_synthetic_data.py first.")

    df = pd.read_csv(RAW_EXPORT_PATH, dtype=str, keep_default_na=False)
    completed = df[df["hh_outcome"] == "completed"]
    if len(completed) == 0:
        raise SystemExit("No completed submissions to use as templates.")

    new_rows = []
    for i in range(n):
        template = completed.sample(1).iloc[0]
        # by default only the first new row gets the forced issue, so a batch
        # isn't 100% duplicates/outliers when --inject-issue is used with -n>1;
        # --all-rows applies it to every row instead, for denser test coverage
        apply_issue = all_rows or i == 0
        new_rows.append(make_row(template, issue if apply_issue else "none"))

    updated = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    updated.to_csv(RAW_EXPORT_PATH, index=False)
    print(f"Appended {n} new submission(s) to {RAW_EXPORT_PATH} "
          f"({'issue: ' + issue + (' (all rows)' if all_rows else ' (first row)') if issue != 'none' else 'no forced issue'}).")
    print("Run 09_realtime_watcher.py to pick them up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2, help="number of new submissions to simulate")
    parser.add_argument("--inject-issue", choices=ISSUE_CHOICES, default="none",
                         help="force a row to trigger a specific check")
    parser.add_argument("--all-rows", action="store_true",
                         help="apply --inject-issue to every new row instead of just the first")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    simulate(n=args.n, issue=args.inject_issue, seed=args.seed, all_rows=args.all_rows)
