"""
Synthetic SurveyCTO-style export for the S.A.M.O.S.A listing survey.

Simulates ~120 villages x ~25 attempted households, run by 16 female
enumerators over Feb-Apr 2027 (per the scope of work timeline), with
realistic data-quality problems deliberately injected so the monitoring
pipeline (script 05) has something real to catch:

  - enumerator fatigue: interviews late in an enumerator's work-day run
    faster and lower-quality for a subset of "rushing" enumerators
  - duplicate submissions: same household captured twice
  - statistical outliers: implausible household size / income / land
  - enumerator free-text comments, correlated with the underlying issues
"""
import json
import random
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from faker import Faker

random.seed(20260825)
np.random.seed(20260825)
fake = Faker("en_IN")
Faker.seed(20260825)

BASE = Path(tempfile.gettempdir()) / "mvsy_monitoring"
DATA = BASE / "data"
villages = json.load(open(DATA / "villages.json"))
enumerators = json.load(open(DATA / "enumerators.json"))

village_by_id = {v["village_id"]: v for v in villages}

# ---------------------------------------------------------------- village
# SMALL EXERCISE DATASET: the full sampling frame (120 villages, 16
# enumerators, data/villages.json + enumerators.json) is kept as the real
# reference frame, but this run only *simulates* a handful of villages so
# the resulting export is ~100 rows - small enough to inspect by eye and to
# extend later with real SurveyCTO submissions when testing live collection.
TARGET_TOTAL_ROWS = 100
N_VILLAGES_SAMPLE = 6
N_ENUMERATORS_SAMPLE = 5

random.shuffle(villages)
sample_villages = villages[:N_VILLAGES_SAMPLE]
enum_ids = [e["enumerator_id"] for e in random.sample(enumerators, N_ENUMERATORS_SAMPLE)]
enum_ids = [e["enumerator_id"] for e in enumerators if e["enumerator_id"] in enum_ids]  # keep canonical order

# distribute the sampled villages across the sampled enumerators, weighted so
# a couple of enumerators carry a heavier load (realistic uneven staffing)
loads = np.random.dirichlet(np.ones(len(enum_ids)) * 3) * len(sample_villages)
loads = np.maximum(0, np.round(loads)).astype(int)
while loads.sum() > len(sample_villages):
    loads[np.argmax(loads)] -= 1
while loads.sum() < len(sample_villages):
    loads[np.argmin(loads)] += 1

village_queue = sample_villages[:]
enum_villages = {}
idx = 0
for eid, n in zip(enum_ids, loads):
    enum_villages[eid] = village_queue[idx: idx + n]
    idx += n

# "rush propensity": how much an enumerator's pace/quality degrades across
# a work-day. Most enumerators are fine; a handful show real fatigue.
rush_propensity = {}
quality_baseline = {}
for e in enumerators:
    eid = e["enumerator_id"]
    rush_propensity[eid] = float(np.clip(np.random.beta(1.5, 4), 0, 1))
    quality_baseline[eid] = float(np.clip(np.random.normal(0.9, 0.08), 0.55, 1.0))
# force a couple of the sampled enumerators to be clear "fatigue" / "low
# quality" cases so the small exercise dataset still has something to catch
for eid in random.sample(enum_ids, min(2, len(enum_ids))):
    rush_propensity[eid] = float(np.random.uniform(0.7, 0.95))
for eid in random.sample(enum_ids, min(1, len(enum_ids))):
    quality_baseline[eid] = float(np.random.uniform(0.55, 0.68))

DEVICE = {e["enumerator_id"]: f"35{random.randint(10**12, 10**13-1)}" for e in enumerators}

OUTCOME_WEIGHTS = {
    "completed": 0.80, "partial_interrupted": 0.05, "no_one_home": 0.06,
    "refused": 0.05, "vacant_not_found": 0.04,
}
RELIGIONS = ["hindu"] * 6 + ["muslim"] * 3 + ["christian", "sikh", "other"]
CASTES = ["obc"] * 4 + ["sc"] * 3 + ["st"] * 2 + ["general"] * 3 + ["other"]
RATION = ["bpl"] * 4 + ["aay"] * 2 + ["apl"] * 3 + ["none"]
RELATIONS = ["self"] * 5 + ["spouse"] * 3 + ["daughter_in_law", "daughter", "mother"]
INFO_SOURCES = ["govt_office", "anganwadi", "ngo", "neighbor", "media", "other"]

COMMENT_BANK = {
    "fatigue": [
        "Long day, several households left in this village for tomorrow - rushed the last couple of interviews.",
        "Running behind schedule by late afternoon, kept answers brief to finish before dark.",
        "Very tired after {n} interviews today, respondent may not have gotten full attention.",
        "Back-to-back interviews with no break, pace picked up a lot toward the end of the day.",
        "Lost an hour to a detour around a flooded path; compressed the remaining interviews to catch up.",
        "Team started late after a vehicle issue, so the afternoon households were rushed to hit the day's target.",
    ],
    "respondent_fatigue": [
        "Respondent seemed impatient and wanted to finish quickly, especially toward the end.",
        "Respondent was distracted by children/cooking for the second half of the interview.",
        "Respondent gave short one-word answers by the later questions, seemed disengaged.",
        "Had to pause interview twice for respondent's chores; answers got briefer afterward.",
        "Respondent kept glancing at the stove and rushed through the last few questions to get back to it.",
        "By the scheme-awareness questions the respondent just said 'whatever you think is fine' a few times.",
    ],
    "duplicate": [
        "This household may already have been listed - head's name sounded familiar from earlier in the week.",
        "Neighbor mentioned another team member visited this house yesterday - flagging possible duplicate.",
        "Same household surveyed on a repeat visit after the first attempt was incomplete.",
        "Household confirmed a different enumerator already came by two days ago; relisting to be safe.",
    ],
    "outlier": [
        "Respondent was uncertain about exact household size, gave a rough estimate.",
        "Income figure is a rough guess - household did not want to share exact amount.",
        "Family reported unusually large landholding, could not independently verify.",
        "Household includes several extended-family members staying temporarily; size may look unusually high.",
        "Respondent hesitated a lot on the income question and eventually rounded to a number that felt too clean.",
    ],
    "other": [
        "Translator needed for parts of the interview (respondent spoke only local dialect).",
        "Heavy rain interrupted the interview; completed under a shared shelter.",
        "GPS signal weak indoors, location recorded from just outside the house.",
        "Household head was away for work; interviewed his mother-in-law instead.",
        "Dog in the compound made the visit difficult; interview conducted at the gate.",
        "",  # majority of interviews have no comment
    ],
    # ---- qualitative stress-test categories -------------------------------
    # These are NOT tied to a quantitative flag and are designed to expose
    # the limits of naive keyword tagging (05_monitoring_pipeline.py's
    # triage_comment): they reuse the same trigger words ("tired", "rushed",
    # "duplicate", "rough estimate"...) but in contexts where a keyword match
    # gets the wrong answer - fatigue language that means the opposite,
    # multi-issue comments a single tag can't capture, genuine ambiguity, and
    # plain good-quality notes. This is the material the Claude triage step
    # (07_comment_analysis.py) is meant to handle better.
    "false_positive_fatigue": [
        "Not rushed at all today - had plenty of time and the respondent gave full, thoughtful answers throughout.",
        "Despite being the last household of a long day, this interview did not feel rushed - respondent was talkative and engaged.",
        "Respondent mentioned she was tired from farm work, but that didn't affect the interview - answers were clear and complete.",
        "Team joked about being 'exhausted' after a good lunch break, but the pace and quality of this interview were normal.",
        "Long list of households today but this one went smoothly - took the full time needed, no shortcuts.",
    ],
    "mixed_signal": [
        "Running low on daylight and the household size figure is a rough estimate - respondent wasn't fully sure how many people usually stay there.",
        "Rushed the last part of the interview because of the hour, and I suspect this may be a repeat visit - the head's name matched someone from earlier in the week.",
        "Respondent grew impatient near the end and the income figure given seems too round to be exact - worth double-checking both.",
        "Long day meant I moved quickly through this one, and the reported landholding sounded unusually large for the area.",
        "Translator was needed and it slowed things down, so later answers may be less detailed than usual.",
    ],
    "ambiguous_duplicate": [
        "Name and general description sound familiar, but this is a common name locally - not confident enough to call it a duplicate.",
        "Could be the household visited on the first day of listing here, but I did not personally confirm - flagging just in case.",
        "Not sure if this is a fresh household or one already counted under a slightly different spelling of the name.",
    ],
    "neutral_positive": [
        "Smooth interview, respondent was cooperative and engaged the whole way through.",
        "No issues - clear answers, easy to locate the household, good GPS signal.",
        "Respondent was well-informed about the household details and answered confidently.",
        "Straightforward visit, household head available and happy to participate.",
        "Quick but complete - respondent had all figures ready and answered without hesitation.",
    ],
}

# categories that CAN be selected as a "no quantitative flag" free comment
# (weighted so most such comments are still mundane/logistics, matching
# real field patterns, while giving the stress-test categories real
# representation in the dataset)
QUALITATIVE_COMMENT_RATE = 0.22
QUALITATIVE_COMMENT_WEIGHTS = {
    "other": 4,
    "neutral_positive": 3,
    "false_positive_fatigue": 2,
    "mixed_signal": 2,
    "ambiguous_duplicate": 1.5,
}

def pick_comment(kind):
    bank = COMMENT_BANK[kind]
    c = random.choice(bank)
    return c.replace("{n}", str(random.randint(9, 14)))


def jitter_latlon(base_lat, base_lon, meters):
    deg = meters / 111_000
    return base_lat + random.uniform(-deg, deg), base_lon + random.uniform(-deg, deg)


# village centroids (fictitious state, roughly Gujarat-Rajasthan bounding box)
village_centroid = {}
for v in villages:
    lat = 21.5 + (hash(v["district_id"]) % 100) / 40 + random.uniform(-0.15, 0.15)
    lon = 71.0 + (hash(v["village_id"]) % 100) / 40 + random.uniform(-0.15, 0.15)
    village_centroid[v["village_id"]] = (lat, lon)

WORK_START_HOUR = 9
WORK_END_HOUR = 17
rows = []
duplicate_log = []

for eid, vlist in enum_villages.items():
    device = DEVICE[eid]
    rush = rush_propensity[eid]
    quality = quality_baseline[eid]
    # spread this enumerator's villages over consecutive working days starting mid-Feb 2027
    day_cursor = datetime(2027, 2, 15) + timedelta(days=random.randint(0, 5))
    for v in vlist:
        vid = v["village_id"]
        base_lat, base_lon = village_centroid[vid]
        n_attempts = random.randint(15, 19)  # ~100 rows total across N_VILLAGES_SAMPLE villages
        # this village's households get worked across 1-3 consecutive days
        n_days = 1 if n_attempts <= 12 else (2 if n_attempts <= 22 else 3)
        per_day = np.array_split(range(1, n_attempts + 1), n_days)

        for day_hh_numbers in per_day:
            if day_cursor.weekday() == 6:  # skip Sundays
                day_cursor += timedelta(days=1)
            n_today = len(day_hh_numbers)
            cur_time = datetime(day_cursor.year, day_cursor.month, day_cursor.day,
                                 WORK_START_HOUR, random.randint(0, 30))
            for seq, hh_number in enumerate(day_hh_numbers, start=1):
                fatigue_factor = seq / n_today  # 0 -> start of day, 1 -> end of day

                outcome = random.choices(list(OUTCOME_WEIGHTS.keys()),
                                          weights=list(OUTCOME_WEIGHTS.values()))[0]

                # ---- duration model ----
                if outcome == "completed":
                    base_dur = np.random.normal(13.0, 2.5)
                elif outcome == "partial_interrupted":
                    base_dur = np.random.normal(7.0, 2.0)
                elif outcome == "refused":
                    base_dur = np.random.normal(3.0, 1.0)
                else:  # no_one_home / vacant_not_found
                    base_dur = np.random.normal(1.5, 0.5)
                base_dur = max(0.7, base_dur)

                rush_cut = 0.0
                is_fatigued_interview = False
                if outcome in ("completed", "partial_interrupted") and rush > 0.55:
                    rush_cut = 0.45 * rush * fatigue_factor
                    if fatigue_factor > 0.7 and random.random() < rush:
                        is_fatigued_interview = True
                duration_min = max(1.2, base_dur * (1 - rush_cut))

                start_dt = cur_time
                end_dt = start_dt + timedelta(minutes=float(duration_min))
                submission_dt = end_dt + timedelta(minutes=random.uniform(1, 240))
                cur_time = end_dt + timedelta(minutes=random.uniform(1.0, 6.0))
                if cur_time.hour >= WORK_END_HOUR:
                    cur_time = datetime(day_cursor.year, day_cursor.month, day_cursor.day,
                                         WORK_END_HOUR, 0)

                lat, lon = jitter_latlon(base_lat, base_lon, random.uniform(30, 400))

                hh_id = f"{vid}-{hh_number}"
                key = f"uuid:{uuid.uuid4()}"

                row = {
                    "KEY": key,
                    "SubmissionDate": submission_dt.strftime("%b %d, %Y %I:%M:%S %p"),
                    "starttime": start_dt.strftime("%b %d, %Y %I:%M:%S %p"),
                    "endtime": end_dt.strftime("%b %d, %Y %I:%M:%S %p"),
                    "today": start_dt.strftime("%b %d, %Y"),
                    "deviceid": device,
                    "formdef_version": "2027021500",
                    "enumerator_id": eid,
                    "district": v["district_id"],
                    "village": vid,
                    "gps_location_Latitude": round(lat, 6),
                    "gps_location_Longitude": round(lon, 6),
                    "hh_number": hh_number,
                    "hh_id": hh_id,
                    "hh_outcome": outcome,
                    "consent_obtained": "", "hh_head_name": "", "respondent_name": "",
                    "respondent_relation": "", "respondent_phone": "",
                    "hh_size": "", "num_adult_women": "", "num_eligible_women": "",
                    "religion": "", "caste_category": "", "ration_card_type": "",
                    "monthly_income": "", "land_owned_acres": "",
                    "govt_employee_in_hh": "", "already_enrolled_msy": "",
                    "eligible_msy": "", "aware_of_scheme": "", "info_source": "",
                    "enumerator_comments": "",
                }

                comment_parts = []

                if outcome in ("completed", "partial_interrupted"):
                    row["consent_obtained"] = "1"
                    female_headed = random.random() < 0.15
                    head_name = fake.name_female() if female_headed else fake.name_male()
                    row["hh_head_name"] = head_name
                    # respondent is always an adult female; she can only be "self"
                    # (i.e. the head) when the household is female-headed
                    relation_choices = RELATIONS if female_headed else [r for r in RELATIONS if r != "self"]
                    relation = random.choice(relation_choices)
                    row["respondent_relation"] = relation
                    row["respondent_name"] = head_name if relation == "self" else fake.name_female()
                    row["respondent_phone"] = (f"{random.choice('6789')}{random.randint(10**8,10**9-1)}"
                                                if random.random() > 0.12 else "")

                    hh_size = max(1, int(np.random.normal(5.2, 1.8)))
                    num_adult_women = min(hh_size, max(0, int(np.random.normal(1.4, 0.7))))
                    num_eligible = min(num_adult_women, max(0, int(np.random.normal(0.9, 0.6))))

                    # ---- outlier injection (~2.5% of completed interviews) ----
                    is_outlier = outcome == "completed" and random.random() < 0.05  # small dataset: keep a few visible
                    if is_outlier:
                        pick = random.choice(["hh_size", "income", "land"])
                        if pick == "hh_size":
                            hh_size = random.randint(19, 27)
                        comment_parts.append(("outlier", pick))

                    row["hh_size"] = hh_size
                    row["num_adult_women"] = num_adult_women
                    row["num_eligible_women"] = num_eligible
                    row["religion"] = random.choice(RELIGIONS)
                    row["caste_category"] = random.choice(CASTES)
                    ration = random.choice(RATION)
                    row["ration_card_type"] = ration

                    income = max(2000, np.random.lognormal(9.0, 0.5))
                    if is_outlier and comment_parts and comment_parts[-1][1] == "income":
                        income = random.choice([180000, 195000, 199000])
                    row["monthly_income"] = int(income)

                    land = max(0.0, np.random.exponential(0.6))
                    if is_outlier and comment_parts and comment_parts[-1][1] == "land":
                        land = round(random.uniform(40, 80), 1)
                    row["land_owned_acres"] = round(land, 2)

                    row["govt_employee_in_hh"] = "1" if random.random() < 0.08 else "0"
                    row["already_enrolled_msy"] = "1" if random.random() < 0.18 else "0"
                    row["eligible_msy"] = "1" if (ration != "apl" and row["govt_employee_in_hh"] == "0"
                                                   and row["monthly_income"] < 15000
                                                   and num_eligible > 0) else "0"

                    # respondent fatigue: near end of a long interview, awareness
                    # module gets rushed / defaulted for a subset of interviews
                    resp_fatigue = (duration_min < 8.5 and outcome == "completed"
                                     and random.random() < (0.15 + 0.25 * fatigue_factor))
                    aware = "1" if random.random() < 0.55 else "0"
                    if resp_fatigue:
                        aware = "0"  # rushed straight past awareness questions
                        comment_parts.append(("respondent_fatigue", None))
                    row["aware_of_scheme"] = aware
                    if aware == "1":
                        n_src = 1 if resp_fatigue else random.choice([1, 1, 2, 2, 3])
                        row["info_source"] = " ".join(random.sample(INFO_SOURCES, n_src))

                    # enumerator-fatigue quality degradation: round-number heaping
                    if is_fatigued_interview and random.random() < 0.5:
                        row["hh_size"] = random.choice([4, 5, 6])
                        row["monthly_income"] = random.choice([10000, 12000, 15000, 20000])
                        comment_parts.append(("fatigue", None))

                    if random.random() > quality:
                        comment_parts.append(("fatigue", None))

                # ---- comments ----
                kinds = [k for k, _ in comment_parts]
                if kinds:
                    kind = kinds[0]
                    row["enumerator_comments"] = pick_comment(kind)
                elif random.random() < QUALITATIVE_COMMENT_RATE:
                    kind = random.choices(
                        list(QUALITATIVE_COMMENT_WEIGHTS.keys()),
                        weights=list(QUALITATIVE_COMMENT_WEIGHTS.values()),
                    )[0]
                    row["enumerator_comments"] = pick_comment(kind)

                rows.append(row)

        day_cursor += timedelta(days=1)

# ---------------------------------------------------------------- duplicates
# ~2% of completed households get a second, near-duplicate submission
completed_idx = [i for i, r in enumerate(rows) if r["hh_outcome"] == "completed"]
n_dupes = max(3, int(len(completed_idx) * 0.04))  # small dataset: force enough to show both detectors
dupe_sample = random.sample(completed_idx, n_dupes)
for k, i in enumerate(dupe_sample):
    src = rows[i]
    dup = dict(src)
    dup["KEY"] = f"uuid:{uuid.uuid4()}"
    # ~1/3 of injected duplicates are re-listed under a fresh household number
    # (data-entry/relisting error) so hh_id does NOT match the original -
    # these can only be caught by the fuzzy name+GPS matcher, not an exact-key check.
    is_fuzzy_only = k % 3 == 0
    # duplicate submitted a bit later, sometimes by a different enumerator covering the same village
    same_enum = random.random() < 0.6
    orig_start = datetime.strptime(src["starttime"], "%b %d, %Y %I:%M:%S %p")
    delay = timedelta(hours=random.uniform(1, 48))
    new_start = orig_start + delay
    new_end = new_start + timedelta(minutes=random.uniform(6, 15))
    dup["starttime"] = new_start.strftime("%b %d, %Y %I:%M:%S %p")
    dup["endtime"] = new_end.strftime("%b %d, %Y %I:%M:%S %p")
    dup["today"] = new_start.strftime("%b %d, %Y")
    dup["SubmissionDate"] = (new_end + timedelta(minutes=random.uniform(1, 120))).strftime("%b %d, %Y %I:%M:%S %p")
    if not same_enum:
        # a different enumerator working the same village re-lists the household
        same_village_enums = [e for e, vs in enum_villages.items()
                               if any(v["village_id"] == src["village"] for v in vs)]
        dup["enumerator_id"] = random.choice(same_village_enums) if same_village_enums else src["enumerator_id"]
        dup["deviceid"] = DEVICE.get(dup["enumerator_id"], src["deviceid"])
    lat, lon = jitter_latlon(src["gps_location_Latitude"], src["gps_location_Longitude"], random.uniform(3, 25))
    dup["gps_location_Latitude"] = round(lat, 6)
    dup["gps_location_Longitude"] = round(lon, 6)
    if is_fuzzy_only:
        new_hh_number = min(60, int(src["hh_number"]) + 30)
        dup["hh_number"] = new_hh_number
        dup["hh_id"] = f"{src['village']}-{new_hh_number}"
    # minor re-typed name variation sometimes (simulates a second enumerator
    # spelling the same name slightly differently), otherwise identical
    if random.random() < 0.4 and dup["hh_head_name"]:
        parts = dup["hh_head_name"].split()
        if len(parts) > 1:
            typo_word = random.choice(parts)
            if len(typo_word) > 3:
                pos = random.randint(1, len(typo_word) - 2)
                typo = typo_word[:pos] + typo_word[pos + 1] + typo_word[pos] + typo_word[pos + 2:]
                dup["hh_head_name"] = dup["hh_head_name"].replace(typo_word, typo, 1)
    if not dup["enumerator_comments"]:
        dup["enumerator_comments"] = pick_comment("duplicate")
    rows.append(dup)

random.shuffle(rows)  # interleave as a server export would (submission order, not enumerator-grouped)
rows.sort(key=lambda r: datetime.strptime(r["SubmissionDate"], "%b %d, %Y %I:%M:%S %p"))

import csv
out_dir = BASE / "output"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "msy_listing_raw_export.csv"
fieldnames = list(rows[0].keys())
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"wrote {len(rows)} rows -> {out_path}")
print(f"completed: {sum(1 for r in rows if r['hh_outcome']=='completed')}")
print(f"injected duplicates: {n_dupes}")
