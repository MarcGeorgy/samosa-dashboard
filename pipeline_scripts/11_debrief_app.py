"""
Interactive assistant for the enumerator debrief call.

Run with:
    streamlit run pipeline_scripts/11_debrief_app.py

Requires output/msy_listing_raw_export.csv to already exist (run
04_generate_synthetic_data.py first, or use 09_realtime_watcher.py /
10_simulate_new_submission.py to feed it live).

Tabs:
  - Overview            headline QA numbers for the current dataset
  - Debrief Agenda       full call agenda generated from the data below
  - Correct Outliers     today's flagged records, editable in place -
                          "corrected on the spot" during the call
  - Comment Themes       auto-summarized enumerator free-text comments
  - Enumerator Callouts  who to flag for a check-in vs. commend, and why

Corrections write straight to output/msy_listing_raw_export.csv (matched by
KEY) and are logged to output/corrections_log.csv for audit, then the
pipeline is rerun so scores/flags reflect the fix immediately.
"""
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = Path(tempfile.gettempdir()) / "mvsy_monitoring" / "output"
RAW_EXPORT_PATH = OUT_DIR / "msy_listing_raw_export.csv"
CORRECTIONS_LOG_PATH = OUT_DIR / "corrections_log.csv"

EDITABLE_COLS = ["hh_size", "monthly_income", "land_owned_acres",
                  "gps_location_Latitude", "gps_location_Longitude", "hh_head_name", "starttime"]


def _import(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pipeline = _import("monitoring_pipeline", "05_monitoring_pipeline.py")
comment_analysis = _import("comment_analysis", "07_comment_analysis.py")
insights = _import("debrief_insights", "08_debrief_insights.py")
scto = _import("surveycto_connector", "06_surveycto_connector.py")
excel_export = _import("excel_export", "12_excel_export.py")
agenda_docx = _import("agenda_docx_export", "13_agenda_docx_export.py")

DT_FMT = "%b %d, %Y %I:%M:%S %p"

LIVE_MODE = bool(os.environ.get("SCTO_SERVER") and os.environ.get("SCTO_USERNAME")
                  and os.environ.get("SCTO_PASSWORD"))

# J-PAL brand colors (Abdul Latif Jameel Poverty Action Lab visual identity)
JPAL_ORANGE = "#E35925"
JPAL_TEAL = "#2FAA9F"
JPAL_GREEN = "#61B77F"
JPAL_YELLOW = "#F2C200"
JPAL_BLUE = "#2D616E"

LOGO_SVG_PATH = SCRIPT_DIR / "assets" / "jpal_logo_official.svg"

st.set_page_config(page_title="S.A.M.O.S.A Debrief Assistant", page_icon="💧", layout="wide")

# Narrowly scoped: only html/body get the brand font, so Streamlit's own
# icon fonts (which set their own font-family with higher specificity)
# aren't affected.
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap" '
    'rel="stylesheet"><style>html, body { font-family: \'Open Sans\', sans-serif; }</style>',
    unsafe_allow_html=True,
)

header_logo, header_text = st.columns([2, 6], vertical_alignment="center")
with header_logo:
    if LOGO_SVG_PATH.exists():
        # official J-PAL logo (vector, from povertyactionlab.org) - embedded
        # inline so the browser renders it natively, pixel-perfect at any
        # size, instead of a rasterized (and softer) PNG copy
        svg_markup = LOGO_SVG_PATH.read_text(encoding="utf-8")
        st.markdown(f'<div style="max-width:260px">{svg_markup}</div>', unsafe_allow_html=True)
with header_text:
    st.markdown(f"<span style='color:{JPAL_TEAL}; font-size:16px;'>S.A.M.O.S.A Debrief Assistant</span>",
                unsafe_allow_html=True)


# --------------------------------------------------------------- data load
def load_flagged():
    return pd.read_csv(OUT_DIR / "msy_listing_flagged.csv", dtype=str, keep_default_na=False)


def load_enum_summary():
    return pd.read_csv(OUT_DIR / "enumerator_summary.csv")


def load_summary_json():
    return json.loads((OUT_DIR / "monitoring_summary.json").read_text())


def load_daily():
    return pd.read_csv(OUT_DIR / "daily_summary.csv")


def load_comments_feed():
    return pd.read_csv(OUT_DIR / "comments_feed.csv")


def build_narrative(summary_json: dict, enum_summary: pd.DataFrame) -> str:
    """Plain-language paragraph summarizing the current dataset - always
    available (templated, no API call needed) so the Overview tab reads as a
    briefing rather than a bare metrics dump."""
    s = summary_json
    band = s.get("quality_band_counts", {})
    n = s.get("n_submissions", 0)
    worst = enum_summary.sort_values("avg_quality_score").iloc[0] if len(enum_summary) else None
    best = enum_summary.sort_values("avg_quality_score").iloc[-1] if len(enum_summary) else None

    parts = [
        f"This dataset covers **{n} submissions** from **{s.get('n_enumerators')} enumerators** "
        f"across **{s.get('n_villages')} villages**, averaging a quality score of "
        f"**{s.get('avg_quality_score')}/100**. "
        f"**{s.get('pct_flagged_any')}%** of submissions carry at least one quality flag "
        f"({band.get('C - needs review', 0)} need review, "
        f"{band.get('D - flag for supervisor callback', 0)} need a supervisor callback)."
    ]

    integrity_bits = []
    if s.get("n_off_hours_flags"):
        integrity_bits.append(f"**{s['n_off_hours_flags']}** submissions happened outside the "
                               f"{pipeline.WORK_HOURS_START}am-{pipeline.WORK_HOURS_END - 12}pm working window")
    if s.get("n_possible_fake_entries"):
        integrity_bits.append(f"**{s['n_possible_fake_entries']}** were completed implausibly fast "
                               f"(under {pipeline.FAKE_ENTRY_MAX_MINUTES} minutes) and may be fabricated")
    if s.get("n_stale_form_version"):
        integrity_bits.append(f"**{s['n_stale_form_version']}** came in on an outdated form version")
    if integrity_bits:
        parts.append("Integrity checks: " + "; ".join(integrity_bits) + ".")

    dup_total = s.get("n_exact_duplicates", 0) + s.get("n_fuzzy_duplicates", 0)
    if dup_total:
        parts.append(f"**{dup_total}** likely duplicate household(s) were caught "
                     f"({s.get('n_exact_duplicates', 0)} exact ID match, {s.get('n_fuzzy_duplicates', 0)} "
                     f"only by name+GPS proximity).")

    if worst is not None and best is not None and worst["enumerator_id"] != best["enumerator_id"]:
        parts.append(f"**{worst['enumerator_id']}** has the lowest average score "
                     f"({worst['avg_quality_score']}) and is worth a closer look; "
                     f"**{best['enumerator_id']}** has the highest ({best['avg_quality_score']}).")

    return " ".join(parts)


def rerun_pipeline():
    """Refresh everything: pull new SurveyCTO submissions (if credentials are
    set) into the local raw export, then rescore the full dataset. Without
    the sync step, new SurveyCTO submissions never show up here even after
    a "refresh" - this is the same sync 09_realtime_watcher.py does."""
    if LIVE_MODE:
        with st.spinner(f"Pulling new submissions from '{scto.SERVER_NAME}'..."):
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            scto.run_incremental()
    with st.spinner("Rerunning monitoring pipeline..."):
        pipeline.run_pipeline()
    st.cache_data.clear()


def bootstrap_demo_data():
    """Generate a baseline synthetic dataset from scratch. Needed because
    hosted storage (e.g. Streamlit Community Cloud) is ephemeral - the app's
    working files can vanish on a cold start/redeploy, so it must be able to
    rebuild its own demo data rather than just erroring out."""
    metadata = _import("metadata", "00_metadata.py")
    synth = _import("generate_synthetic_data", "04_generate_synthetic_data.py")


if LIVE_MODE and not RAW_EXPORT_PATH.exists():
    rerun_pipeline()

if not LIVE_MODE and not RAW_EXPORT_PATH.exists():
    with st.spinner("First run - generating baseline demo data..."):
        bootstrap_demo_data()
        rerun_pipeline()

if not RAW_EXPORT_PATH.exists():
    st.error(f"{RAW_EXPORT_PATH} not found. Run 04_generate_synthetic_data.py first "
             f"(or set SCTO_SERVER/SCTO_USERNAME/SCTO_PASSWORD to pull from your live SurveyCTO server).")
    st.stop()

if not (OUT_DIR / "msy_listing_flagged.csv").exists():
    rerun_pipeline()

flagged = load_flagged()
enum_summary = load_enum_summary()
summary_json = load_summary_json()
daily = load_daily()

# --------------------------------------------------------------- sidebar

if st.sidebar.button("🔄 Refresh from latest data", width="stretch",
                       help="Pulls new SurveyCTO submissions (if live mode is on) and rescores everything."):
    rerun_pipeline()
    st.rerun()

available_dates = sorted(pd.to_datetime(flagged["starttime"], format="%b %d, %Y %I:%M:%S %p",
                                          errors="coerce").dt.date.dropna().unique())
selected_date = st.sidebar.selectbox("Debrief for work day:", options=available_dates,
                                       index=len(available_dates) - 1 if available_dates else 0)
corrected_by = st.sidebar.text_input("Your name/email (for correction log):",
                                       value="marcnabil123@gmail.com")

st.sidebar.metric("Submissions", summary_json["n_submissions"])
st.sidebar.metric("Avg quality score", summary_json["avg_quality_score"])
st.sidebar.metric("% flagged", f"{summary_json['pct_flagged_any']}%")

comments_feed = load_comments_feed()
xlsx_bytes = excel_export.build_workbook(flagged, enum_summary, daily, comments_feed, summary_json)
st.sidebar.download_button(
    "📥 Export dashboard to Excel", xlsx_bytes,
    file_name=f"samosa_dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    width="stretch",
)

tab_overview, tab_agenda, tab_correct, tab_comments, tab_enum = st.tabs(
    ["Overview", "Debrief Agenda", "Correct Outliers", "Comment Themes", "Enumerator Callouts"]
)

# --------------------------------------------------------------- overview
with tab_overview:
    st.header("Data quality overview")
    st.markdown(build_narrative(summary_json, enum_summary))

    cols = st.columns(4)
    cols[0].metric("Submissions", summary_json["n_submissions"])
    cols[1].metric("Enumerators", summary_json["n_enumerators"])
    cols[2].metric("Avg quality score", summary_json["avg_quality_score"])
    cols[3].metric("Exact + fuzzy duplicates",
                    summary_json["n_exact_duplicates"] + summary_json["n_fuzzy_duplicates"])
    cols2 = st.columns(4)
    cols2[0].metric("Off-hours submissions", summary_json.get("n_off_hours_flags", 0),
                     help=f"Started before {pipeline.WORK_HOURS_START}am or at/after "
                          f"{pipeline.WORK_HOURS_END - 12}pm")
    cols2[1].metric("Possible fake entries", summary_json.get("n_possible_fake_entries", 0),
                     help=f"Completed in under {pipeline.FAKE_ENTRY_MAX_MINUTES} minutes - "
                          f"too fast to be a genuine interview")
    cols2[2].metric("Stale form version", summary_json.get("n_stale_form_version", 0),
                     help="Submitted on an older form version than the current one")
    cols2[3].metric("Comments received", summary_json["n_comments"])

    st.subheader("Quality band distribution")
    band_df = pd.DataFrame(list(summary_json["quality_band_counts"].items()), columns=["Band", "Count"])
    st.bar_chart(band_df.set_index("Band"))

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Daily trend")
        if len(daily):
            st.line_chart(daily.set_index("work_date")[["avg_quality_score"]])
            st.caption("Average quality score by work day.")
        else:
            st.caption("No daily data yet.")

    with chart_col2:
        st.subheader("Flags by type")
        flag_cols = [c for c in flagged.columns if c.startswith("flag_")]
        flag_counts = flagged[flag_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(bool).sum()
        flag_counts = flag_counts[flag_counts > 0].sort_values(ascending=False)
        if len(flag_counts):
            flag_counts.index = [c.replace("flag_", "").replace("_", " ") for c in flag_counts.index]
            st.bar_chart(flag_counts)
        else:
            st.caption("No flags in this dataset.")

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        st.subheader("Quality score distribution")
        scores = pd.to_numeric(flagged["data_quality_score"], errors="coerce").dropna()
        if len(scores):
            bins = pd.cut(scores, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                          include_lowest=True)
            hist = bins.value_counts().sort_index()
            hist.index = [f"{int(iv.left)}-{int(iv.right)}" for iv in hist.index]
            st.bar_chart(hist)
        else:
            st.caption("No scores yet.")

    with chart_col4:
        st.subheader("Submissions per enumerator")
        if len(enum_summary):
            st.bar_chart(enum_summary.set_index("enumerator_id")[["n_submissions"]])
        else:
            st.caption("No enumerator data yet.")

    st.subheader("All flagged records")
    st.dataframe(flagged[flagged["n_flags"].astype(float) > 0][
        ["KEY", "enumerator_id", "village", "hh_id", "quality_band", "data_quality_score"] + flag_cols
    ], width="stretch")

# --------------------------------------------------------- correct outliers
with tab_correct:
    st.header(f"Records to review - {selected_date}")
    correctable = insights.todays_flagged_records(flagged, work_date=selected_date)

    if len(correctable) == 0:
        st.success("No flagged records for this work day.")
    else:
        raw = pd.read_csv(RAW_EXPORT_PATH, dtype=str, keep_default_na=False)
        editable = correctable[["KEY", "enumerator_id", "village", "hh_id", "flags_triggered",
                                  "data_quality_score", "quality_band"]].merge(
            raw[["KEY"] + EDITABLE_COLS], on="KEY", how="left"
        )
        st.caption("Edit the underlying values below and click Save to correct the record on the spot "
                   "- the pipeline rescoring runs immediately after. `starttime` must stay in the exact "
                   f"format `{DT_FMT.replace('%b', 'Mon').replace('%d', 'DD').replace('%Y', 'YYYY').replace('%I', 'HH').replace('%M', 'MM').replace('%S', 'SS').replace('%p', 'AM/PM')}` "
                   "e.g. `Feb 20, 2027 09:15:00 AM` - an unparseable value is skipped rather than saved.")
        edited = st.data_editor(
            editable,
            width="stretch",
            disabled=["KEY", "enumerator_id", "village", "hh_id", "flags_triggered",
                      "data_quality_score", "quality_band"],
            key=f"editor_{selected_date}",
        )

        if st.button("💾 Save corrections", type="primary"):
            changes, rejected = [], []
            for _, new_row in edited.iterrows():
                key = new_row["KEY"]
                old_row = editable[editable["KEY"] == key].iloc[0]
                for col in EDITABLE_COLS:
                    old_val, new_val = str(old_row[col]), str(new_row[col])
                    if old_val == new_val:
                        continue
                    if col == "starttime":
                        try:
                            datetime.strptime(new_val, DT_FMT)
                        except ValueError:
                            rejected.append(f"{new_row['hh_id']}: '{new_val}' doesn't match the required format")
                            continue
                    raw.loc[raw["KEY"] == key, col] = new_val
                    changes.append({
                        "corrected_at": datetime.now().isoformat(timespec="seconds"),
                        "corrected_by": corrected_by,
                        "KEY": key,
                        "hh_id": new_row["hh_id"],
                        "field": col,
                        "old_value": old_val,
                        "new_value": new_val,
                    })

            if rejected:
                st.error("Some changes were not saved:\n" + "\n".join(f"- {r}" for r in rejected))

            if changes:
                raw.to_csv(RAW_EXPORT_PATH, index=False)
                changes_df = pd.DataFrame(changes)
                header = not CORRECTIONS_LOG_PATH.exists()
                changes_df.to_csv(CORRECTIONS_LOG_PATH, mode="a", header=header, index=False)
                st.success(f"Saved {len(changes)} correction(s) and rescored the dataset.")
                rerun_pipeline()
                st.rerun()
            elif not rejected:
                st.info("No changes to save.")

    if CORRECTIONS_LOG_PATH.exists():
        with st.expander("Correction history"):
            st.dataframe(pd.read_csv(CORRECTIONS_LOG_PATH), width="stretch")

# ------------------------------------------------------------- comments
with tab_comments:
    st.header("Field comment analysis")
    with st.spinner("Analyzing comments..."):
        enriched = comment_analysis.analyze_comments_batch(flagged)
        themes = comment_analysis.summarize_comment_themes(enriched)

    st.info(themes.get("overall_summary", ""))
    for t in themes.get("themes", []):
        with st.container(border=True):
            st.markdown(f"**{t['theme']}** - {t['n_mentions']} mention(s), "
                        f"affecting: {', '.join(t['affected_enumerators'])}")
            st.caption(f"“{t['example_quote']}”")

    st.subheader("All comments (analyzed)")
    with_comments = enriched[enriched["enumerator_comments"] != ""]
    st.dataframe(with_comments[[
        "enumerator_id", "hh_id", "enumerator_comments", "comment_tag",
        "detail_tags", "detail_severity", "keyword_mismatch", "recommended_action"
    ]], width="stretch")
    st.caption("`comment_tag` is the pipeline's single-keyword tag; `detail_tags`/`detail_severity` are the "
               "richer analyzed read, and `keyword_mismatch` flags comments where the keyword tag "
               "likely got it wrong (e.g. negated fatigue language).")

# --------------------------------------------------------- enumerator callouts
with tab_enum:
    st.header("Enumerator performance callouts")
    recs = insights.enumerator_recommendations(enum_summary)

    st.subheader("🚩 Flag for check-in")
    flag_df = recs[recs["recommendation"] == "flag_for_checkin"]
    if len(flag_df):
        st.dataframe(flag_df[["enumerator_id", "avg_quality_score", "pct_fatigue_flagged",
                               "pct_respondent_fatigue", "n_duplicates", "rationale"]],
                     width="stretch")
    else:
        st.caption("No enumerators meet the check-in threshold this period.")

    st.subheader("⭐ Commend for consistency")
    commend_df = recs[recs["recommendation"] == "commend"]
    if len(commend_df):
        st.dataframe(commend_df[["enumerator_id", "avg_quality_score", "rationale"]],
                     width="stretch")
    else:
        st.caption("No enumerators meet the commendation bar this period.")

    with st.expander("Everyone (including 'monitor' - no action needed)"):
        st.dataframe(recs[["enumerator_id", "avg_quality_score", "z_score", "recommendation", "rationale"]],
                     width="stretch")

# --------------------------------------------------------------- agenda
with tab_agenda:
    st.header("Debrief call agenda")
    st.caption("Starts with what to ask your enumerators and what to check yourself, then the "
               "supporting data, then a blank space for your own notes.")
    if st.button("Generate agenda"):
        with st.spinner("Generating agenda..."):
            agenda_payload = insights.build_agenda_payload(flagged, enum_summary, summary_json,
                                                              work_date=selected_date)
            agenda_md = insights.phrase_agenda_markdown(agenda_payload)
            agenda_docx_bytes = agenda_docx.build_agenda_docx(agenda_payload)
        st.session_state["agenda_md"] = agenda_md
        st.session_state["agenda_docx_bytes"] = agenda_docx_bytes

    if "agenda_md" in st.session_state:
        st.markdown(st.session_state["agenda_md"])
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button("📄 Download agenda (.docx)", st.session_state["agenda_docx_bytes"],
                                file_name=f"debrief_agenda_{selected_date}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                width="stretch")
        with dl_col2:
            st.download_button("📝 Download agenda (.md)", st.session_state["agenda_md"],
                                file_name=f"debrief_agenda_{selected_date}.md", width="stretch")
