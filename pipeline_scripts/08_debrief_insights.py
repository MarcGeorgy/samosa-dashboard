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
        reasons, codes = [], []
        if row["z_score"] <= Z_SCORE_FLAG:
            reasons.append(f"avg quality score {row['avg_quality_score']} is well below the team mean ({mean_score:.1f})")
            codes.append("low_score")
        if row["pct_fatigue_flagged"] >= FATIGUE_PCT_FLAG_THRESHOLD:
            reasons.append(f"{row['pct_fatigue_flagged']}% of interviews fatigue-flagged")
            codes.append("fatigue")
        if row["pct_respondent_fatigue"] >= FATIGUE_PCT_FLAG_THRESHOLD:
            reasons.append(f"{row['pct_respondent_fatigue']}% respondent-fatigue flagged")
            codes.append("respondent_fatigue")
        if row["n_duplicates"] >= DUPLICATES_FLAG_THRESHOLD:
            reasons.append(f"{int(row['n_duplicates'])} duplicate submissions - a repeated pattern, not a one-off")
            codes.append("duplicates")
        if reasons:
            return "flag_for_checkin", "; ".join(reasons), codes

        clean = (row["pct_fatigue_flagged"] == 0 and row["pct_respondent_fatigue"] == 0
                 and row["n_duplicates"] == 0)
        if row["z_score"] >= Z_SCORE_COMMEND and clean:
            return "commend", f"avg quality score {row['avg_quality_score']} with zero fatigue/duplicate flags", []
        return "monitor", "within normal range, no action needed", []

    results = df.apply(classify, axis=1, result_type="expand")
    df["recommendation"], df["rationale"], df["reason_codes"] = results[0], results[1], results[2]
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
# Ready-to-read questions for the RA to ask the enumerator directly, keyed by
# the same reason_codes enumerator_recommendations() attaches to each
# flag_for_checkin row - one question per reason, not a vague prompt.
ASK_TEMPLATES = {
    "low_score": "Is anything making interviews harder for you right now - transport, translation, "
                 "household availability, workload? What would help?",
    "fatigue": "Do you find it harder to keep the same pace by the end of the day? Would a different "
               "household order, or an extra break, help?",
    "respondent_fatigue": "When a respondent seems tired or distracted partway through, what do you "
                          "currently do - push through, or come back another time?",
    "duplicates": "Walk me through how you check whether a household has already been listed before "
                  "you start an interview.",
}


def _questions_to_ask(recs: pd.DataFrame) -> list:
    out = []
    for _, row in recs[recs["recommendation"] == "flag_for_checkin"].iterrows():
        for code in row["reason_codes"]:
            question = ASK_TEMPLATES.get(code)
            if question:
                out.append({"enumerator_id": row["enumerator_id"], "question": question})
    return out


def _things_to_monitor(summary_json: dict, correctable_count: int) -> list:
    """Reminders for the RA to personally check - distinct from the questions
    above, these are things to verify yourself, not to ask the enumerator."""
    items = []
    n_off = summary_json.get("n_off_hours_flags", 0)
    if n_off:
        items.append(f"Spot-check the {n_off} off-hours submission(s) - confirm the device timestamp "
                      f"is accurate and this wasn't backfilled later.")
    n_fake = summary_json.get("n_possible_fake_entries", 0)
    if n_fake:
        items.append(f"Personally review the {n_fake} implausibly-fast submission(s) line by line "
                      f"before accepting them as genuine.")
    n_stale = summary_json.get("n_stale_form_version", 0)
    if n_stale:
        items.append(f"Confirm the {n_stale} submission(s) on an outdated form version means those "
                      f"enumerators have actually re-synced their app since this was flagged.")
    dup_total = summary_json.get("n_exact_duplicates", 0) + summary_json.get("n_fuzzy_duplicates", 0)
    if dup_total:
        items.append(f"Cross-check the {dup_total} flagged duplicate household(s) against the master "
                      f"listing before the next round goes out.")
    if correctable_count:
        items.append(f"Walk through each of the {correctable_count} flagged record(s) below with the "
                      f"team and correct the value on record during the call.")
    if not items:
        items.append("No specific integrity concerns this period - a routine spot-check of a few "
                      "random submissions is still good practice.")
    return items


def build_agenda_payload(flagged, enum_summary, summary_json, work_date=None) -> dict:
    """Structured data behind the debrief agenda - used both to phrase the
    on-screen/markdown agenda (stub template or Claude) and, directly, to
    build the Word export (13_agenda_docx_export.py), so the two can never
    drift apart from each other."""
    recs = enumerator_recommendations(enum_summary)
    correctable = todays_flagged_records(flagged, work_date)
    themes = comment_analysis.summarize_comment_themes(flagged)

    resolved_date = work_date or (correctable["_work_date"].iloc[0] if len(correctable) else "latest available day")
    return {
        "date": str(resolved_date),
        "questions_to_ask": _questions_to_ask(recs),
        "things_to_monitor": _things_to_monitor(summary_json, len(correctable)),
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
    lines = [f"# S.A.M.O.S.A Debrief Agenda - {payload['date']}", ""]

    lines.append("## Questions to Ask Your Enumerators")
    if payload["questions_to_ask"]:
        by_enum = {}
        for q in payload["questions_to_ask"]:
            by_enum.setdefault(q["enumerator_id"], []).append(q["question"])
        for e, qs in by_enum.items():
            lines.append(f"**{e}:**")
            for q in qs:
                lines.append(f"- {q}")
        lines.append("")
    else:
        lines.append("- No enumerator meets the check-in threshold this period - no targeted questions needed.")
        lines.append("")

    lines.append("## Things to Monitor Yourself")
    for m in payload["things_to_monitor"]:
        lines.append(f"- {m}")
    lines.append("")

    h = payload["headline"]
    lines.append("## Overview")
    lines.append(f"{h['n_submissions']} submissions, avg quality score {h['avg_quality_score']}, "
                 f"{h['pct_flagged_any']}% carrying at least one flag.")
    lines.append("")

    if payload["flag_for_checkin"]:
        lines.append("## Enumerators to Check In With")
        for r in payload["flag_for_checkin"]:
            lines.append(f"- **{r['enumerator_id']}** (avg {r['avg_quality_score']}): {r['rationale']}")
        lines.append("")

    if payload["commend"]:
        lines.append("## Enumerators to Commend")
        for r in payload["commend"]:
            lines.append(f"- **{r['enumerator_id']}** (avg {r['avg_quality_score']}): {r['rationale']}")
        lines.append("")

    if payload["correction_items"]:
        lines.append("## Records to Review / Correct on the Call")
        for c in payload["correction_items"]:
            lines.append(f"- `{c['hh_id']}` ({c['enumerator_id']}, {c['village']}): "
                         f"{c['flags_triggered']} - score {c['data_quality_score']}")
        lines.append("")

    themes = payload["comment_themes"]
    lines.append("## Field Comment Themes")
    lines.append(themes.get("overall_summary", ""))
    for t in themes.get("themes", []):
        lines.append(f"- **{t['theme']}** ({t['n_mentions']}x, {', '.join(t['affected_enumerators'])}): "
                     f"\"{t['example_quote']}\"")
    lines.append("")

    lines.append("## RA Comments")
    lines.append("*(Space for your own notes after the call - not filled in automatically. "
                 "Download the Word version for a printable notes section.)*")
    return "\n".join(lines)


def _live_phrase_agenda(payload: dict) -> str:
    client = comment_analysis._client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        output_config={"effort": "medium"},
        system=(
            "You write a concise, well-organized markdown agenda for a field supervisor's (Research "
            "Associate's) weekly enumerator debrief call, from structured monitoring data. Use exactly "
            "this section order: "
            "(1) 'Questions to Ask Your Enumerators' - use the exact items in questions_to_ask, grouped "
            "by enumerator, phrased as direct, ready-to-read questions - these are things to literally "
            "ask out loud on the call; "
            "(2) 'Things to Monitor Yourself' - use the exact items in things_to_monitor; these are for "
            "the RA to personally verify, NOT to ask the enumerator; "
            "(3) Overview; (4) Enumerators to Check In With; (5) Enumerators to Commend; "
            "(6) Records to Review/Correct on the Call; (7) Field Comment Themes; "
            "(8) 'RA Comments' - end with just this heading and a one-line italic note that this space "
            "is for the RA's own notes after the call. "
            "Be concrete and cite the actual numbers given - do not invent data or add questions beyond "
            "what's given in questions_to_ask/things_to_monitor."
        ),
        messages=[{"role": "user", "content": json.dumps(payload, indent=2, default=str)}],
    )
    return next(b.text for b in response.content if b.type == "text")


def phrase_agenda_markdown(payload: dict) -> str:
    if comment_analysis.ANALYSIS_LIVE:
        return _live_phrase_agenda(payload)
    return _stub_phrase_agenda(payload)


def generate_debrief_agenda(flagged: pd.DataFrame, enum_summary: pd.DataFrame,
                             summary_json: dict, work_date=None) -> str:
    payload = build_agenda_payload(flagged, enum_summary, summary_json, work_date)
    return phrase_agenda_markdown(payload)


if __name__ == "__main__":
    flagged = pd.read_csv(OUT_DIR / "msy_listing_flagged.csv", dtype=str, keep_default_na=False)
    for col in ["data_quality_score", "n_flags"]:
        flagged[col] = pd.to_numeric(flagged[col], errors="coerce")
    enum_summary = pd.read_csv(OUT_DIR / "enumerator_summary.csv")
    summary_json = json.loads((OUT_DIR / "monitoring_summary.json").read_text())

    print(f"mode: {'LIVE Claude API' if comment_analysis.ANALYSIS_LIVE else 'OFFLINE (set ANTHROPIC_API_KEY for live analysis)'}")
    print(generate_debrief_agenda(flagged, enum_summary, summary_json))
