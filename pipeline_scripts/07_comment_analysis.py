"""
Claude-based triage for enumerator free-text comments.

05_monitoring_pipeline.py's `triage_comment()` is a single keyword match -
fast and fully deterministic, but it single-tags each comment and can't
tell "not rushed at all today" from "very tired after 14 interviews", or
notice that one comment raises two separate issues at once. This module
adds a second pass on top of that keyword tag using Claude, producing:

  - one or more tags per comment (not just the first keyword hit)
  - a severity read (does this actually need supervisor attention?)
  - a flag for cases where the keyword tag looks like a false positive
  - a one-line summary and a suggested action

It's designed to run per new submission (see 09_realtime_watcher.py), so
results are cached by submission KEY - a rerun only calls the API for
comments it hasn't seen before.

STUBBED BY DEFAULT: without ANTHROPIC_API_KEY set, `analyze_comment()` and
`summarize_comment_themes()` fall back to `_stub_*` heuristics so the rest
of the pipeline (and the debrief app) can be exercised offline. The call
sites and data shapes are identical either way - set ANTHROPIC_API_KEY to
switch to real Claude analysis with no other code changes.
"""
import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd

MODEL = "claude-opus-5"
OUT_DIR = Path(tempfile.gettempdir()) / "mvsy_monitoring" / "output"
CACHE_PATH = OUT_DIR / "comment_analysis_cache.json"

ANALYSIS_LIVE = bool(os.environ.get("ANTHROPIC_API_KEY"))

TAGS = ["fatigue", "respondent_fatigue", "duplicate", "data_quality", "logistics", "positive", "other"]
SEVERITIES = ["none", "low", "medium", "high"]

COMMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {"type": "string", "enum": TAGS},
            "description": "All issue categories genuinely raised by the comment; use ['positive'] or ['other'] when nothing else applies.",
        },
        "severity": {
            "type": "string",
            "enum": SEVERITIES,
            "description": "How much this comment, on its own, should worry a field supervisor.",
        },
        "keyword_tag_is_false_positive": {
            "type": "boolean",
            "description": "True if a naive keyword match on this text would likely mis-tag it (e.g. negated fatigue language, sarcasm, an unrelated use of a trigger word).",
        },
        "summary": {"type": "string", "description": "One sentence, plain language, for a supervisor skimming a list."},
        "recommended_action": {"type": "string", "description": "One short, concrete next step, or 'none' if no action is needed."},
    },
    "required": ["tags", "severity", "keyword_tag_is_false_positive", "summary", "recommended_action"],
    "additionalProperties": False,
}

THEMES_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "n_mentions": {"type": "integer"},
                    "example_quote": {"type": "string"},
                    "affected_enumerators": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["theme", "n_mentions", "example_quote", "affected_enumerators"],
                "additionalProperties": False,
            },
        },
        "overall_summary": {"type": "string", "description": "2-3 sentences a supervisor could read aloud to open a debrief call."},
    },
    "required": ["themes", "overall_summary"],
    "additionalProperties": False,
}


def _client():
    import anthropic
    return anthropic.Anthropic()


def _load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


# ------------------------------------------------------------- stub mode
_NEGATORS = re.compile(r"\b(not|no|n't|didn't|despite|without)\b", re.IGNORECASE)
_KEYWORDS = {
    "fatigue": ["rushed", "tired", "behind schedule", "long day", "pace picked up", "no break", "exhausted"],
    "respondent_fatigue": ["impatient", "distracted", "one-word", "disengaged", "brief", "whatever you think"],
    "duplicate": ["already", "duplicate", "same household", "repeat visit", "familiar", "spelling of the name"],
    "data_quality": ["uncertain", "rough estimate", "rough guess", "could not verify", "could not independently", "too round", "not sure"],
    "logistics": ["rain", "gps", "translator", "dialect", "dog", "away for work"],
    "positive": ["smooth", "no issues", "cooperative", "confidently", "happy to participate", "not rushed", "did not feel rushed"],
}


def _stub_analyze(text: str, context: dict) -> dict:
    """Heuristic stand-in for the Claude call: same output shape, cruder logic.

    Unlike the pipeline's single-tag `triage_comment()`, this collects every
    matching category and treats a keyword hit inside a negated clause
    ("not rushed", "didn't feel rushed") as a positive/false-positive signal
    instead of the issue it names - the specific gap a real Claude read closes.
    """
    low = text.lower()
    hits = []
    for tag, kws in _KEYWORDS.items():
        if any(kw in low for kw in kws):
            hits.append(tag)

    negated = bool(_NEGATORS.search(low))
    fp = False
    if negated and ("fatigue" in hits or "respondent_fatigue" in hits):
        # e.g. "Not rushed at all today..." - drop the fatigue read, it's a false alarm
        hits = [h for h in hits if h not in ("fatigue", "respondent_fatigue")]
        if "positive" not in hits:
            hits.append("positive")
        fp = True

    if not hits:
        hits = ["other"]

    real_issues = [h for h in hits if h not in ("positive", "other")]
    if "positive" in hits and not real_issues:
        severity = "none"
    elif len(real_issues) >= 2:
        severity = "medium"
    elif real_issues:
        severity = "low"
    else:
        severity = "none"

    action = "none"
    if "duplicate" in real_issues:
        action = "Cross-check household ID / GPS against nearby submissions before next visit."
    elif "fatigue" in real_issues:
        action = "Discuss workload pacing with enumerator at next debrief."
    elif "data_quality" in real_issues:
        action = "Flag figure for supervisor verification on next visit."

    return {
        "tags": hits,
        "severity": severity,
        "keyword_tag_is_false_positive": fp,
        "summary": text if len(text) <= 120 else text[:117] + "...",
        "recommended_action": action,
        "_source": "stub",
    }


def _stub_summarize_themes(comments: list[dict]) -> dict:
    from collections import Counter
    tag_counts = Counter()
    examples = {}
    enum_by_tag = {}
    for c in comments:
        for t in c["analysis"]["tags"]:
            if t in ("positive",):
                continue
            tag_counts[t] += 1
            examples.setdefault(t, c["enumerator_comments"])
            enum_by_tag.setdefault(t, set()).add(c["enumerator_id"])

    themes = [
        {
            "theme": tag.replace("_", " ").title(),
            "n_mentions": n,
            "example_quote": examples[tag],
            "affected_enumerators": sorted(enum_by_tag[tag]),
        }
        for tag, n in tag_counts.most_common()
    ]
    top = themes[0]["theme"] if themes else "no significant issues"
    overall = (
        f"{len(comments)} comments reviewed this period; the most common theme is {top.lower()} "
        f"({themes[0]['n_mentions']} mention(s))." if themes
        else f"{len(comments)} comments reviewed this period; no recurring issues stood out."
    )
    return {"themes": themes, "overall_summary": overall, "_source": "stub"}


# ---------------------------------------------------------------- live call
def _live_analyze(text: str, context: dict) -> dict:
    client = _client()
    ctx_str = ", ".join(f"{k}={v}" for k, v in context.items() if v not in (None, ""))
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": COMMENT_SCHEMA}},
        system=(
            "You triage free-text field notes from household-listing survey enumerators for a "
            "field supervisor. Read each comment carefully in context - watch for negation "
            "('not rushed', 'didn't feel rushed') and for comments that raise more than one "
            "issue at once. Be terse and practical; supervisors skim these."
        ),
        messages=[{"role": "user", "content": f"Comment: \"{text}\"\nContext: {ctx_str}"}],
    )
    text_block = next(b.text for b in response.content if b.type == "text")
    result = json.loads(text_block)
    result["_source"] = "claude"
    return result


def _live_summarize_themes(comments: list[dict]) -> dict:
    client = _client()
    lines = [f"- [{c['enumerator_id']}] {c['enumerator_comments']}" for c in comments]
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": THEMES_SCHEMA}},
        system=(
            "You are preparing talking points for a field supervisor's enumerator debrief call. "
            "Given a batch of raw enumerator comments, group them into a small number of concrete "
            "themes (not one theme per comment), and write an opening summary a supervisor could "
            "read aloud."
        ),
        messages=[{"role": "user", "content": "Comments:\n" + "\n".join(lines)}],
    )
    text_block = next(b.text for b in response.content if b.type == "text")
    result = json.loads(text_block)
    result["_source"] = "claude"
    return result


# ------------------------------------------------------------------ API
def analyze_comment(text: str, context: dict | None = None) -> dict:
    context = context or {}
    if not text:
        return {"tags": [], "severity": "none", "keyword_tag_is_false_positive": False,
                "summary": "", "recommended_action": "none", "_source": "none"}
    if ANALYSIS_LIVE:
        return _live_analyze(text, context)
    return _stub_analyze(text, context)


def analyze_comments_batch(df: pd.DataFrame, key_col: str = "KEY") -> pd.DataFrame:
    """Enrich `df` (must have 'enumerator_comments' + key_col) with analysis columns.

    Cached by key_col in CACHE_PATH so re-running the pipeline (e.g. from
    the real-time watcher) only pays for genuinely new comments.
    """
    cache = _load_cache()
    df = df.copy()
    detail_tags, detail_severity, keyword_mismatch, comment_summary, recommended_action = [], [], [], [], []

    for _, row in df.iterrows():
        text = row.get("enumerator_comments", "") or ""
        key = str(row.get(key_col, ""))
        if not text:
            result = {"tags": [], "severity": "none", "keyword_tag_is_false_positive": False,
                       "summary": "", "recommended_action": "none"}
        elif key in cache:
            result = cache[key]
        else:
            context = {
                "hh_outcome": row.get("hh_outcome"),
                "duration_min": row.get("duration_min"),
                "existing_keyword_tag": row.get("comment_tag"),
            }
            result = analyze_comment(text, context)
            cache[key] = result

        detail_tags.append(", ".join(result.get("tags", [])))
        detail_severity.append(result.get("severity", "none"))
        keyword_mismatch.append(result.get("keyword_tag_is_false_positive", False))
        comment_summary.append(result.get("summary", ""))
        recommended_action.append(result.get("recommended_action", "none"))

    _save_cache(cache)
    df["detail_tags"] = detail_tags
    df["detail_severity"] = detail_severity
    df["keyword_mismatch"] = keyword_mismatch
    df["comment_summary"] = comment_summary
    df["recommended_action"] = recommended_action
    return df


def summarize_comment_themes(df: pd.DataFrame, max_comments: int = 80) -> dict:
    """Thematic narrative over a batch of comments, for the debrief agenda."""
    enriched = analyze_comments_batch(df) if "detail_tags" not in df.columns else df
    with_comments = enriched[enriched["enumerator_comments"] != ""]
    if len(with_comments) == 0:
        return {"themes": [], "overall_summary": "No enumerator comments in this period.", "_source": "none"}

    sample = with_comments.tail(max_comments)
    comments = []
    for _, row in sample.iterrows():
        comments.append({
            "enumerator_id": row["enumerator_id"],
            "enumerator_comments": row["enumerator_comments"],
            "analysis": {"tags": [t.strip() for t in row["detail_tags"].split(",") if t.strip()]},
        })

    if ANALYSIS_LIVE:
        return _live_summarize_themes(comments)
    return _stub_summarize_themes(comments)


if __name__ == "__main__":
    IN_PATH = OUT_DIR / "msy_listing_flagged.csv"
    if not IN_PATH.exists():
        print(f"Run 05_monitoring_pipeline.py first - {IN_PATH} not found.")
    else:
        df = pd.read_csv(IN_PATH, dtype=str, keep_default_na=False)
        enriched = analyze_comments_batch(df)
        enriched.to_csv(OUT_DIR / "msy_listing_flagged_analyzed.csv", index=False)
        themes = summarize_comment_themes(enriched)
        print(f"mode: {'LIVE Claude API' if ANALYSIS_LIVE else 'OFFLINE (set ANTHROPIC_API_KEY for live analysis)'}")
        print(json.dumps(themes, indent=2))
