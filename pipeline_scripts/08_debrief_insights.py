"""
Data-driven inputs for the enumerator debrief call:

  - enumerator_recommendations()   who to flag for a check-in vs. commend,
                                    computed from the same enum_summary.csv
                                    05_monitoring_pipeline.py already produces
  - todays_flagged_records()       today's outliers/duplicates/fatigue flags,
                                    the set a supervisor can correct live
  - generate_debrief_agenda()      pulls both of the above plus the
                                    comment-theme summary (07) into one
                                    agenda, phrased by Claude (or a stub)

All the *selection* logic (who gets flagged, which records need correction)
is plain pandas over the pipeline's own output - deterministic and
inspectable. Claude (see 07_comment_analysis.py for the stub/live
split) is only used to turn that structured selection into call-ready prose;
swapping stub<->live never changes who gets flagged.
"""
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = Path(tempfile.gettempdir()) / "mvsy_monitoring" / "output"
MODEL = "claude-opus-5"


def _import(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


comment_analysis = _import("comment_analysis", "07_comment_analysis.py")

FATIGUE_PCT_FLAG_THRESHOLD = 10.0   # % of an enumerator's interviews fatigue-flagged
Z_SCORE_FLAG = -1.0                 # avg_quality_score this far below the group mean
Z_SCORE_COMMEND = 1.0                # ...or this far above, with a clean record
DUPLICATES_FLAG_THRESHOLD = 2       # a single duplicate can be a one-off (e.g. a different
                                     # enumerator re-visiting the same household); 2+ suggests a pattern


# --------------------------------------------------------- who to flag/commend
def enumerator_recommendations(enum_summary: pd.DataFrame) -> pd.DataFrame:
    df = enum_summary.copy()
    numeric_cols = ["avg_quality_score", "pct_fatigue_flagged", "pct_respondent_fatigue", "n_duplicates"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    mean_score = df["avg_quality_score"].mean()
    std_score = df["avg_quality_score"].std(ddof=0)
    df["z_score"] = 0.0 if not std_score else (df["avg_quality_score"] - mean_score) / std_score

    def classify(row):
        reasons = []
        if row["z_score"] <= Z_SCORE_FLAG:
            reasons.append(f"avg quality score {row['avg_quality_score']} is well below the team mean ({mean_score:.1f})")
        if row["pct_fatigue_flagged"] >= FATIGUE_PCT_FLAG_THRESHOLD:
            reasons.append(f"{row['pct_fatigue_flagged']}% of interviews fatigue-flagged")
        if row["pct_respondent_fatigue"] >= FATIGUE_PCT_FLAG_THRESHOLD:
            reasons.append(f"{row['pct_respondent_fatigue']}% respondent-fatigue flagged")
        if row["n_duplicates"] >= DUPLICATES_FLAG_THRESHOLD:
            reasons.append(f"{int(row['n_duplicates'])} duplicate submissions - a repeated pattern, not a one-off")
        if reasons:
            return "flag_for_checkin", "; ".join(reasons)

        clean = (row["pct_fatigue_flagged"] == 0 and row["pct_respondent_fatigue"] == 0
                 and row["n_duplicates"] == 0)
        if row["z_score"] >= Z_SCORE_COMMEND and clean:
            return "commend", f"avg quality score {row['avg_quality_score']} with zero fatigue/duplicate flags"
        return "monitor", "within normal range, no action needed"

    results = df.apply(classify, axis=1, result_type="expand")
    df["recommendation"], df["rationale"] = results[0], results[1]
    return df.sort_values(["recommendation", "avg_quality_score"])


# ----------------------------------------------------------- today's outliers
def _parse_dt_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%b %d, %Y %I:%M:%S %p", errors="coerce")


def todays_flagged_records(flagged: pd.DataFrame, work_date=None) -> pd.DataFrame:
    """Records with >=1 flag, restricted to one work day (default: the most
    recent day present) - the set a supervisor could correct live on a call.
    """
    df = flagged.copy()
    df["_work_date"] = _parse_dt_series(df["starttime"]).dt.date
    df["n_flags"] = pd.to_numeric(df["n_flags"], errors="coerce")
    df["data_quality_score"] = pd.to_numeric(df["data_quality_score"], errors="coerce")
    if work_date is None:
        work_date = df["_work_date"].max()
    day_df = df[df["_work_date"] == work_date]
    flag_cols = [c for c in day_df.columns if c.startswith("flag_")]
    correctable = day_df[day_df["n_flags"] > 0].copy()
    correctable["flags_triggered"] = correctable[flag_cols].apply(
        lambda r: ", ".join(c.replace("flag_", "") for c, v in r.items() if str(v) in ("True", "1")), axis=1
    )
    return correctable.sort_values("data_quality_score")


# ------------------------------------------------------------------- agenda
def _agenda_payload(flagged, enum_summary, summary_json, work_date=None) -> dict:
    recs = enumerator_recommendations(enum_summary)
    correctable = todays_flagged_records(flagged, work_date)
    themes = comment_analysis.summarize_comment_themes(flagged)

    resolved_date = work_date or (correctable["_work_date"].iloc[0] if len(correctable) else "latest available day")
    return {
        "date": str(resolved_date),
        "headline": {
            "n_submissions": summary_json.get("n_submissions"),
            "avg_quality_score": summary_json.get("avg_quality_score"),
            "pct_flagged_any": summary_json.get("pct_flagged_any"),
            "quality_band_counts": summary_json.get("quality_band_counts"),
        },
        "flag_for_checkin": recs[recs["recommendation"] == "flag_for_checkin"][
            ["enumerator_id", "avg_quality_score", "rationale"]].to_dict("records"),
        "commend": recs[recs["recommendation"] == "commend"][
            ["enumerator_id", "avg_quality_score", "rationale"]].to_dict("records"),
        "correction_items": correctable[
            ["KEY", "enumerator_id", "village", "hh_id", "flags_triggered", "data_quality_score"]
        ].head(15).to_dict("records"),
        "comment_themes": themes,
    }


def _stub_phrase_agenda(payload: dict) -> str:
    lines = [f"# Enumerator Debrief Agenda - {payload['date']}", ""]
    h = payload["headline"]
    lines.append(f"**Overview:** {h['n_submissions']} submissions, avg quality score {h['avg_quality_score']}, "
                 f"{h['pct_flagged_any']}% carrying at least one flag.")
    lines.append("")

    if payload["flag_for_checkin"]:
        lines.append("## 1. Enumerators to check in with")
        for r in payload["flag_for_checkin"]:
            lines.append(f"- **{r['enumerator_id']}** (avg {r['avg_quality_score']}): {r['rationale']}")
        lines.append("")

    if payload["commend"]:
        lines.append("## 2. Enumerators to commend")
        for r in payload["commend"]:
            lines.append(f"- **{r['enumerator_id']}** (avg {r['avg_quality_score']}): {r['rationale']}")
        lines.append("")

    if payload["correction_items"]:
        lines.append("## 3. Records to review / correct on the call")
        for c in payload["correction_items"]:
            lines.append(f"- `{c['hh_id']}` ({c['enumerator_id']}, {c['village']}): "
                         f"{c['flags_triggered']} - score {c['data_quality_score']}")
        lines.append("")

    themes = payload["comment_themes"]
    lines.append("## 4. Field comment themes")
    lines.append(themes.get("overall_summary", ""))
    for t in themes.get("themes", []):
        lines.append(f"- **{t['theme']}** ({t['n_mentions']}x, {', '.join(t['affected_enumerators'])}): "
                     f"\"{t['example_quote']}\"")
    lines.append("")
    lines.append("## 5. Discussion prompts")
    lines.append("- Walk through each correction item above and confirm/update the value on record.")
    if payload["flag_for_checkin"]:
        names = ", ".join(r["enumerator_id"] for r in payload["flag_for_checkin"])
        lines.append(f"- Ask {names} about pacing late in the day - is workload realistic?")
    lines.append("- Share any logistics blockers (transport, translation, weather) raised in comments.")
    return "\n".join(lines)


def _live_phrase_agenda(payload: dict) -> str:
    client = comment_analysis._client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        output_config={"effort": "medium"},
        system=(
            "You write a concise, well-organized markdown agenda for a field supervisor's weekly "
            "enumerator debrief call, from structured monitoring data. Sections: overview, "
            "enumerators to check in with (with reasons), enumerators to commend, specific records "
            "to review/correct live on the call, field comment themes, and 3-5 discussion prompts. "
            "Be concrete and cite the actual numbers given - do not invent data."
        ),
        messages=[{"role": "user", "content": json.dumps(payload, indent=2, default=str)}],
    )
    return next(b.text for b in response.content if b.type == "text")


def generate_debrief_agenda(flagged: pd.DataFrame, enum_summary: pd.DataFrame,
                             summary_json: dict, work_date=None) -> str:
    payload = _agenda_payload(flagged, enum_summary, summary_json, work_date)
    if comment_analysis.ANALYSIS_LIVE:
        return _live_phrase_agenda(payload)
    return _stub_phrase_agenda(payload)


if __name__ == "__main__":
    flagged = pd.read_csv(OUT_DIR / "msy_listing_flagged.csv", dtype=str, keep_default_na=False)
    for col in ["data_quality_score", "n_flags"]:
        flagged[col] = pd.to_numeric(flagged[col], errors="coerce")
    enum_summary = pd.read_csv(OUT_DIR / "enumerator_summary.csv")
    summary_json = json.loads((OUT_DIR / "monitoring_summary.json").read_text())

    print(f"mode: {'LIVE Claude API' if comment_analysis.ANALYSIS_LIVE else 'OFFLINE (set ANTHROPIC_API_KEY for live analysis)'}")
    print(generate_debrief_agenda(flagged, enum_summary, summary_json))
