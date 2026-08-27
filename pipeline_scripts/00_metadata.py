"""
Shared reference metadata for the MSY listing-survey exercise:
villages (120, across 4 districts, 60 treatment / 60 control) and
enumerators (female, per the scope of work). Both the XLSForm builder
and the synthetic data generator import this module so IDs stay consistent.
"""
import json
import random
import tempfile
from pathlib import Path

random.seed(42)

OUT_DIR = Path(tempfile.gettempdir()) / "mvsy_monitoring" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DISTRICTS = [
    {"district_id": "D1", "district_name": "Anandpur"},
    {"district_id": "D2", "district_name": "Bhavnagar Rural"},
    {"district_id": "D3", "district_name": "Chandragiri"},
    {"district_id": "D4", "district_name": "Devgarh"},
]

# 120 villages, 30 per district, randomly assigned to treatment/control
# (60/60 as per the scope of work), block-randomized within district.
villages = []
vid = 1
for d in DISTRICTS:
    village_ids_in_district = list(range(1, 31))
    arm_pool = ["treatment"] * 15 + ["control"] * 15
    random.shuffle(arm_pool)
    for i, local_idx in enumerate(village_ids_in_district):
        villages.append({
            "village_id": f"V{vid:03d}",
            "village_name": f"{d['district_name'].split()[0]}_{local_idx:02d}",
            "district_id": d["district_id"],
            "district_name": d["district_name"],
            "arm": arm_pool[i],
        })
        vid += 1

with open(OUT_DIR / "villages.json", "w") as f:
    json.dump(villages, f, indent=2)

# 16 female listing enumerators, unevenly loaded (as happens in the field),
# each working a block of villages within 1-2 districts.
FIRST_NAMES = ["Anita","Bhavna","Chitra","Deepa","Esha","Farida","Geeta","Hina",
               "Indira","Jyoti","Kavita","Lakshmi","Meera","Nisha","Omisha","Priya"]
enumerators = []
for i, name in enumerate(FIRST_NAMES, start=1):
    enumerators.append({
        "enumerator_id": f"E{i:02d}",
        "enumerator_name": name,
        # each enumerator is anchored to one district to reflect realistic team assignment
        "home_district": DISTRICTS[(i - 1) % 4]["district_id"],
    })

with open(OUT_DIR / "enumerators.json", "w") as f:
    json.dump(enumerators, f, indent=2)

print(f"{len(villages)} villages -> {OUT_DIR/'villages.json'}")
print(f"{len(enumerators)} enumerators -> {OUT_DIR/'enumerators.json'}")
