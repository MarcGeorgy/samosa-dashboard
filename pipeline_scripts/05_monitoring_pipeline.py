"""
Real-time data quality monitoring pipeline for the S.A.M.O.S.A listing survey.

Designed to run repeatedly (cron / SurveyCTO webhook / manual re-run) against
whatever the latest SurveyCTO export looks like. Each run is idempotent: it
recomputes flags over the full current dataset (simplest, most robust design
for a monitoring job at this scale; see NOTE at the bottom for an incremental
variant once volumes are much larger).

Produces:
  1. output/msy_listing_flagged.csv   - one row per submission + all flags/score
  2. output/enumerator_summary.csv    - one row per enumerator, aggregated QA metrics
  3. output/daily_summary.csv         - one row per day, aggregated QA metrics
  4. output/comments_feed.csv         - enumerator free-text comments, triaged
  5. output/monitoring_summary.json   - headline numbers for the dashboard
"""
import json
import math
import tempfile
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(tempfile.gettempdir()) / "mvsy_monitoring"
IN_PATH = str(BASE / "output" / "msy_listing_raw_export.csv")
OUT_DIR = BASE / "output"

DT_FMT = "%b %d, %Y %I:%M:%S %p"

# ------------------------------------------------------------------ config
# Thresholds are the kind of thing a field manager should be able to tune
# per survey instrument; kept as named constants rather than buried in logic.
MIN_PLAUSIBLE_DURATION_MIN = 5.0      # below this, a "completed" listing interview is suspect
MAX_PLAUSIBLE_DURATION_MIN = 60.0     # above this, the device was probably left open
FATIGUE_RUSH_RATIO = 0.70             # duration < 70% of the enumerator's own day-median
FATIGUE_LATE_DAY_FRAC = 0.5           # only look at the back half of the work-day
DUPLICATE_GPS_METERS = 60.0
DUPLICATE_NAME_SIMILARITY = 0.82
OUTLIER_IQR_MULT = 1.5

WORK_HOURS_START = 7                  # interviews starting before 7am are suspect (24h clock)
WORK_HOURS_END = 19                   # ...or at/after 7pm - outside typical field hours
FAKE_ENTRY_MAX_MINUTES = 2.0          # a full household listing interview cannot genuinely finish
                                       # this fast - distinct from MIN_PLAUSIBLE_DURATION_MIN (5 min),
                                       # which is a softer "unusually quick" read; this one is a much
                                       # stronger "this was probably fabricated, not conducted" signal


def parse_dt(s):
    if pd.isna(s) or s == "":
        return pd.NaT
    return datetime.strptime(s, DT_FMT)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def name_similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio()


def load_data(path=IN_PATH):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["starttime_dt"] = df["starttime"].apply(parse_dt)
    df["endtime_dt"] = df["endtime"].apply(parse_dt)
    df["submission_dt"] = df["SubmissionDate"].apply(parse_dt)
    df["duration_min"] = (df["endtime_dt"] - df["starttime_dt"]).dt.total_seconds() / 60.0
    df["work_date"] = df["starttime_dt"].dt.date
    for col in ["hh_size", "num_adult_women", "num_eligible_women", "monthly_income",
                "land_owned_acres", "gps_location_Latitude", "gps_location_Longitude",
                "hh_number"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["lag_flag"] = ((df["submission_dt"] - df["endtime_dt"]).dt.total_seconds() / 3600.0)
    return df


# ------------------------------------------------------------- 1. fatigue
def compute_fatigue(df):
    df = df.sort_values(["enumerator_id", "work_date", "starttime_dt"]).copy()
    df["day_seq"] = df.groupby(["enumerator_id", "work_date"]).cumcount() + 1
    df["day_total"] = df.groupby(["enumerator_id", "work_date"])["day_seq"].transform("max")
    df["day_frac"] = df["day_seq"] / df["day_total"]

    completed = df[df["hh_outcome"].isin(["completed", "partial_interrupted"])]
    enum_median_dur = completed.groupby("enumerator_id")["duration_min"].median().rename("enum_median_duration")
    df = df.merge(enum_median_dur, on="enumerator_id", how="left")

    df["flag_duration_too_short"] = (
        df["hh_outcome"].isin(["completed", "partial_interrupted"])
        & (df["duration_min"] < MIN_PLAUSIBLE_DURATION_MIN)
    )
    df["flag_duration_too_long"] = df["duration_min"] > MAX_PLAUSIBLE_DURATION_MIN

    df["flag_enumerator_fatigue"] = (
        df["hh_outcome"].isin(["completed", "partial_interrupted"])
        & (df["day_frac"] >= FATIGUE_LATE_DAY_FRAC)
        & (df["duration_min"] < FATIGUE_RUSH_RATIO * df["enum_median_duration"])
    )

    # respondent fatigue proxy: short completed interview where the LAST module
    # (scheme awareness) looks skipped/defaulted rather than genuinely answered
    df["flag_respondent_fatigue"] = (
        (df["hh_outcome"] == "completed")
        & (df["duration_min"] < df["enum_median_duration"] * 0.75)
        & (df["aware_of_scheme"] == "0")
    )
    return df


# ------------------------------------------------------------- 2. outliers
def compute_outliers(df):
    df = df.copy()
    df["flag_outlier_hh_size"] = False
    df["flag_outlier_income"] = False
    df["flag_outlier_land"] = False
    completed_mask = df["hh_outcome"].isin(["completed", "partial_interrupted"])

    # monthly_income and land_owned_acres are naturally right-skewed (lognormal-
    # like); running raw IQR on the untransformed scale flags a large share of
    # the legitimate upper tail as "outliers". Log-transform first so the IQR
    # rule targets genuine anomalies rather than ordinary skew. hh_size is
    # roughly symmetric already, so it's left on the raw scale.
    log_cols = {"monthly_income", "land_owned_acres"}
    MIN_GROUP_N = 8
    for col, flagcol in [("hh_size", "flag_outlier_hh_size"),
                          ("monthly_income", "flag_outlier_income"),
                          ("land_owned_acres", "flag_outlier_land")]:
        district_groups = {d: s[col].dropna() for d, s in df[completed_mask].groupby("district")}
        # districts with too few submissions of their own (early in data
        # collection, or a small pilot batch) borrow the pooled reference
        # distribution instead of being skipped outright.
        pooled = df.loc[completed_mask, col].dropna()
        for dist, raw_vals in district_groups.items():
            ref = pooled if len(raw_vals) < MIN_GROUP_N else raw_vals
            if len(ref) < MIN_GROUP_N:
                continue
            vals = np.log1p(ref) if col in log_cols else ref
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - OUTLIER_IQR_MULT * iqr, q3 + OUTLIER_IQR_MULT * iqr
            series = np.log1p(df[col]) if col in log_cols else df[col]
            mask = (df["district"] == dist) & completed_mask & ((series < lo) | (series > hi))
            df.loc[mask, flagcol] = True

    df["n_outlier_flags"] = df[["flag_outlier_hh_size", "flag_outlier_income", "flag_outlier_land"]].sum(axis=1)
    return df


# ------------------------------------------------------------ 3. duplicates
def compute_duplicates(df):
    df = df.copy()
    df["flag_exact_duplicate"] = df.duplicated(subset=["hh_id"], keep=False) & (df["hh_id"] != "")
    df["duplicate_group_id"] = ""
    df["flag_fuzzy_duplicate"] = False

    grp = 0
    flagged_pairs = []
    for village, sub in df.groupby("village"):
        sub = sub[sub["hh_outcome"].isin(["completed", "partial_interrupted"])]
        idx = sub.index.tolist()
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = df.loc[idx[i]], df.loc[idx[j]]
                if pd.isna(a["gps_location_Latitude"]) or pd.isna(b["gps_location_Latitude"]):
                    continue
                dist_m = haversine_m(a["gps_location_Latitude"], a["gps_location_Longitude"],
                                      b["gps_location_Latitude"], b["gps_location_Longitude"])
                if dist_m > DUPLICATE_GPS_METERS:
                    continue
                sim = name_similarity(a["hh_head_name"], b["hh_head_name"])
                if sim >= DUPLICATE_NAME_SIMILARITY:
                    flagged_pairs.append((idx[i], idx[j], dist_m, sim))

    for i, j, dist_m, sim in flagged_pairs:
        grp += 1
        df.loc[i, "flag_fuzzy_duplicate"] = True
        df.loc[j, "flag_fuzzy_duplicate"] = True
        gid = f"dup_{grp:04d}"
        df.loc[i, "duplicate_group_id"] = (df.loc[i, "duplicate_group_id"] + ";" + gid).strip(";")
        df.loc[j, "duplicate_group_id"] = (df.loc[j, "duplicate_group_id"] + ";" + gid).strip(";")

    df["flag_any_duplicate"] = df["flag_exact_duplicate"] | df["flag_fuzzy_duplicate"]
    return df


# ------------------------------------------------------ 4. submission integrity
def compute_submission_integrity(df):
    """Signals aimed at fabricated-data risk rather than field-quality risk:
    activity outside plausible working hours, entries too fast to have been
    genuinely conducted, and enumerators still submitting on an outdated form
    version (a compliance nudge - not a fraud signal, but visible here so a
    supervisor can catch it early rather than discover it as bad data).
    """
    df = df.copy()

    start_hour = df["starttime_dt"].dt.hour
    df["flag_off_hours"] = df["starttime_dt"].notna() & (
        (start_hour < WORK_HOURS_START) | (start_hour >= WORK_HOURS_END)
    )

    df["flag_possible_fake_fast_entry"] = (
        df["hh_outcome"].isin(["completed", "partial_interrupted"])
        & (df["duration_min"] < FAKE_ENTRY_MAX_MINUTES)
    )

    # "latest" is whatever version is most common in THIS batch, not a config
    # value - so this only fires once a newer version has actually started
    # appearing in submissions, exactly the moment a supervisor needs to know
    # who hasn't updated yet.
    versions = df.loc[df["formdef_version"] != "", "formdef_version"]
    if len(versions) and versions.nunique() > 1:
        latest_version = versions.max()
        df["flag_stale_form_version"] = (df["formdef_version"] != "") & (df["formdef_version"] != latest_version)
    else:
        df["flag_stale_form_version"] = False

    return df


# ------------------------------------------------------- 5. enumerator notes
ISSUE_KEYWORDS = {
    "fatigue": ["rushed", "tired", "behind schedule", "long day", "pace picked up", "no break"],
    "respondent_fatigue": ["impatient", "distracted", "one-word", "disengaged", "brief"],
    "duplicate": ["already", "duplicate", "same household", "repeat visit"],
    "data_quality": ["uncertain", "rough estimate", "rough guess", "could not verify", "could not independently"],
    "logistics": ["rain", "gps", "translator", "dialect", "dog", "away for work"],
}


def triage_comment(text):
    if not text:
        return ""
    low = text.lower()
    for tag, kws in ISSUE_KEYWORDS.items():
        if any(kw in low for kw in kws):
            return tag
    return "other"


def compute_comments(df):
    df = df.copy()
    df["comment_tag"] = df["enumerator_comments"].apply(triage_comment)
    return df


# ------------------------------------------------------- 6. quality score
def compute_quality_score(df):
    df = df.copy()
    score = pd.Series(100.0, index=df.index)

    score -= df["flag_duration_too_short"].astype(int) * 25
    score -= df["flag_duration_too_long"].astype(int) * 10
    score -= df["flag_enumerator_fatigue"].astype(int) * 15
    score -= df["flag_respondent_fatigue"].astype(int) * 10
    score -= df["n_outlier_flags"] * 12
    score -= df["flag_exact_duplicate"].astype(int) * 30
    score -= (df["flag_fuzzy_duplicate"] & ~df["flag_exact_duplicate"]).astype(int) * 20
    score -= df["comment_tag"].isin(["fatigue", "respondent_fatigue", "data_quality"]).astype(int) * 8
    # missing GPS on a completed interview is a real field-protocol miss
    score -= (df["hh_outcome"].isin(["completed", "partial_interrupted"])
              & df["gps_location_Latitude"].isna()).astype(int) * 10
    score -= df["flag_off_hours"].astype(int) * 15
    score -= df["flag_possible_fake_fast_entry"].astype(int) * 20
    score -= df["flag_stale_form_version"].astype(int) * 5

    df["data_quality_score"] = score.clip(lower=0, upper=100).round(1)

    def band(s):
        if s >= 90:
            return "A - clean"
        if s >= 75:
            return "B - minor issues"
        if s >= 55:
            return "C - needs review"
        return "D - flag for supervisor callback"
    df["quality_band"] = df["data_quality_score"].apply(band)
    return df


# --------------------------------------------------------------- run all
def run_pipeline(in_path=IN_PATH, out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(in_path)
    df = compute_fatigue(df)
    df = compute_outliers(df)
    df = compute_duplicates(df)
    df = compute_submission_integrity(df)
    df = compute_comments(df)
    df = compute_quality_score(df)

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["n_flags"] = df[flag_cols].sum(axis=1)

    keep_cols = [
        "KEY", "SubmissionDate", "starttime", "endtime", "duration_min", "formdef_version",
        "enumerator_id", "district", "village", "hh_number", "hh_id", "hh_outcome",
        "hh_head_name", "respondent_name", "day_seq", "day_total", "day_frac",
        "enum_median_duration",
    ] + flag_cols + [
        "n_outlier_flags", "duplicate_group_id", "enumerator_comments", "comment_tag",
        "n_flags", "data_quality_score", "quality_band",
    ]
    flagged = df[keep_cols].sort_values("SubmissionDate")
    flagged.to_csv(out_dir / "msy_listing_flagged.csv", index=False)

    # ---- enumerator-level summary ----
    enum_summary = df.groupby("enumerator_id").agg(
        n_submissions=("KEY", "count"),
        n_completed=("hh_outcome", lambda s: (s == "completed").sum()),
        pct_completed=("hh_outcome", lambda s: round(100 * (s == "completed").mean(), 1)),
        median_duration_min=("duration_min", "median"),
        pct_too_short=("flag_duration_too_short", lambda s: round(100 * s.mean(), 1)),
        pct_fatigue_flagged=("flag_enumerator_fatigue", lambda s: round(100 * s.mean(), 1)),
        pct_respondent_fatigue=("flag_respondent_fatigue", lambda s: round(100 * s.mean(), 1)),
        n_duplicates=("flag_any_duplicate", "sum"),
        n_outliers=("n_outlier_flags", lambda s: (s > 0).sum()),
        pct_off_hours=("flag_off_hours", lambda s: round(100 * s.mean(), 1)),
        n_possible_fake_entries=("flag_possible_fake_fast_entry", "sum"),
        n_stale_form_version=("flag_stale_form_version", "sum"),
        n_comments=("enumerator_comments", lambda s: (s != "").sum()),
        avg_quality_score=("data_quality_score", "mean"),
    ).round(1).reset_index().sort_values("avg_quality_score")
    enum_summary.to_csv(out_dir / "enumerator_summary.csv", index=False)

    # ---- daily summary ----
    daily = df.groupby("work_date").agg(
        n_submissions=("KEY", "count"),
        n_completed=("hh_outcome", lambda s: (s == "completed").sum()),
        avg_quality_score=("data_quality_score", "mean"),
        n_duplicates=("flag_any_duplicate", "sum"),
        n_outliers=("n_outlier_flags", lambda s: (s > 0).sum()),
        n_fatigue_flags=("flag_enumerator_fatigue", "sum"),
        n_off_hours=("flag_off_hours", "sum"),
        n_possible_fake_entries=("flag_possible_fake_fast_entry", "sum"),
    ).round(1).reset_index().sort_values("work_date")
    daily.to_csv(out_dir / "daily_summary.csv", index=False)

    # ---- comments feed ----
    comments = df[df["enumerator_comments"] != ""][
        ["SubmissionDate", "enumerator_id", "village", "hh_id", "hh_outcome",
         "comment_tag", "enumerator_comments", "data_quality_score"]
    ].sort_values("SubmissionDate", ascending=False)
    comments.to_csv(out_dir / "comments_feed.csv", index=False)

    # ---- headline JSON for the dashboard ----
    summary = {
        "generated_at": datetime.now().isoformat(),
        "n_submissions": int(len(df)),
        "n_completed": int((df["hh_outcome"] == "completed").sum()),
        "n_enumerators": int(df["enumerator_id"].nunique()),
        "n_villages": int(df["village"].nunique()),
        "avg_quality_score": round(float(df["data_quality_score"].mean()), 1),
        "pct_flagged_any": round(100 * float((df["n_flags"] > 0).mean()), 1),
        "n_exact_duplicates": int(df["flag_exact_duplicate"].sum()),
        "n_fuzzy_duplicates": int((df["flag_fuzzy_duplicate"] & ~df["flag_exact_duplicate"]).sum()),
        "n_outlier_rows": int((df["n_outlier_flags"] > 0).sum()),
        "n_enumerator_fatigue_flags": int(df["flag_enumerator_fatigue"].sum()),
        "n_respondent_fatigue_flags": int(df["flag_respondent_fatigue"].sum()),
        "n_off_hours_flags": int(df["flag_off_hours"].sum()),
        "n_possible_fake_entries": int(df["flag_possible_fake_fast_entry"].sum()),
        "n_stale_form_version": int(df["flag_stale_form_version"].sum()),
        "n_comments": int((df["enumerator_comments"] != "").sum()),
        "quality_band_counts": df["quality_band"].value_counts().to_dict(),
        "worst_enumerators": enum_summary.head(5)[["enumerator_id", "avg_quality_score", "pct_fatigue_flagged"]].to_dict("records"),
    }
    with open(out_dir / "monitoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return df, enum_summary, daily, comments, summary


if __name__ == "__main__":
    df, enum_summary, daily, comments, summary = run_pipeline()
    print(json.dumps(summary, indent=2, default=str))
