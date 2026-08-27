"""
Real-time trigger: react whenever new submissions land in the raw export.

Two modes, chosen automatically:

  LIVE   - if SCTO_SERVER / SCTO_USERNAME / SCTO_PASSWORD are all set as
           environment variables (see 06_surveycto_connector.py's docstring
           for how to get these), each check pulls real submissions from
           your SurveyCTO server via `run_incremental()` and merges them
           into output/msy_listing_raw_export.csv before looking for new
           rows.
  SIMULATED - otherwise, "new submission arrives" is simulated by watching
           that same CSV for rows this process hasn't seen before -
           10_simulate_new_submission.py appends rows to it to trigger this
           for a demo without a real server.

Either way, once new rows are in the CSV, `check_once()`:
  1. reruns the monitoring pipeline over the full current dataset (it's
     idempotent by design - see 05_monitoring_pipeline.py's docstring)
  2. runs comment triage on the (now-current) flagged data - cached by
     submission KEY, so only genuinely new comments cost an API call
  3. alerts on any of the NEW rows that landed in band C/D or came back
     flagged as a duplicate, appending to output/alerts_log.csv
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = Path(tempfile.gettempdir()) / "mvsy_monitoring" / "output"
RAW_EXPORT_PATH = OUT_DIR / "msy_listing_raw_export.csv"
SEEN_KEYS_PATH = OUT_DIR / "_watcher_seen_keys.json"
ALERTS_PATH = OUT_DIR / "alerts_log.csv"

ALERT_BANDS = {"C - needs review", "D - flag for supervisor callback"}


def _import(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pipeline = _import("monitoring_pipeline", "05_monitoring_pipeline.py")
comment_analysis = _import("comment_analysis", "07_comment_analysis.py")
scto = _import("surveycto_connector", "06_surveycto_connector.py")

LIVE_MODE = bool(os.environ.get("SCTO_SERVER") and os.environ.get("SCTO_USERNAME")
                  and os.environ.get("SCTO_PASSWORD"))


def _sync_from_surveycto(verbose: bool = True):
    """Pull the latest submissions from the real SurveyCTO server into
    RAW_EXPORT_PATH - the same file the simulated flow appends to, so
    everything downstream is unaware of which source populated it."""
    if verbose:
        print(f"  pulling from live server '{scto.SERVER_NAME}', form '{scto.FORM_ID}'...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scto.run_incremental()


def _load_seen_keys() -> set:
    if SEEN_KEYS_PATH.exists():
        return set(json.loads(SEEN_KEYS_PATH.read_text()))
    return set()


def _save_seen_keys(keys: set):
    SEEN_KEYS_PATH.write_text(json.dumps(sorted(keys)))


def _read_new_rows(seen_keys: set) -> pd.DataFrame:
    if not RAW_EXPORT_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(RAW_EXPORT_PATH, dtype=str, keep_default_na=False)
    return df[~df["KEY"].isin(seen_keys)]


def _append_alerts(rows: list[dict]):
    if not rows:
        return
    alerts_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = not ALERTS_PATH.exists()
    alerts_df.to_csv(ALERTS_PATH, mode="a", header=header, index=False)


def check_once(verbose: bool = True) -> dict:
    """Single poll: detect new rows, rerun pipeline + comment triage, alert. Returns a status dict."""
    if LIVE_MODE:
        _sync_from_surveycto(verbose=verbose)

    seen_keys = _load_seen_keys()
    new_rows = _read_new_rows(seen_keys)

    if len(new_rows) == 0:
        if verbose:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] no new submissions.")
        return {"new_submissions": 0}

    if verbose:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {len(new_rows)} new submission(s) detected "
              f"- rerunning pipeline...")

    df, enum_summary, daily, comments, summary = pipeline.run_pipeline()
    flagged = pd.read_csv(OUT_DIR / "msy_listing_flagged.csv", dtype=str, keep_default_na=False)
    enriched = comment_analysis.analyze_comments_batch(flagged)
    enriched.to_csv(OUT_DIR / "msy_listing_flagged_analyzed.csv", index=False)

    new_keys = set(new_rows["KEY"])
    new_flagged = enriched[enriched["KEY"].isin(new_keys)]

    alert_rows = []
    for _, row in new_flagged.iterrows():
        reasons = []
        if row.get("quality_band") in ALERT_BANDS:
            reasons.append(f"band {row['quality_band']}")
        if str(row.get("flag_any_duplicate")) == "True":
            reasons.append("duplicate")
        if reasons:
            alert_rows.append({
                "detected_at": datetime.now().isoformat(timespec="seconds"),
                "KEY": row["KEY"],
                "enumerator_id": row["enumerator_id"],
                "village": row["village"],
                "hh_id": row["hh_id"],
                "data_quality_score": row["data_quality_score"],
                "reasons": "; ".join(reasons),
            })

    _append_alerts(alert_rows)
    seen_keys |= new_keys
    _save_seen_keys(seen_keys)

    if verbose:
        print(f"  pipeline rerun complete: {summary['n_submissions']} total submissions, "
              f"avg score {summary['avg_quality_score']}.")
        if alert_rows:
            print(f"  ALERTS: {len(alert_rows)} new submission(s) need attention:")
            for a in alert_rows:
                print(f"    - {a['hh_id']} ({a['enumerator_id']}): {a['reasons']} "
                      f"[score {a['data_quality_score']}]")
        else:
            print("  no alerts from this batch.")

    return {"new_submissions": len(new_rows), "alerts": alert_rows, "summary": summary}


def watch(interval_sec: int = 30, max_iterations: int | None = None):
    source = f"live SurveyCTO server '{scto.SERVER_NAME}' (form '{scto.FORM_ID}')" if LIVE_MODE \
        else f"local simulation ({RAW_EXPORT_PATH})"
    print(f"Watching {source} every {interval_sec}s "
          f"(Ctrl+C to stop; analysis mode: {'LIVE' if comment_analysis.ANALYSIS_LIVE else 'OFFLINE'})")
    i = 0
    while max_iterations is None or i < max_iterations:
        check_once()
        i += 1
        if max_iterations is None or i < max_iterations:
            time.sleep(interval_sec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="poll continuously instead of checking once")
    parser.add_argument("--interval", type=int, default=30, help="poll interval in seconds (--watch mode)")
    args = parser.parse_args()

    if args.watch:
        watch(interval_sec=args.interval)
    else:
        check_once()
