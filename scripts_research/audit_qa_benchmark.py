"""Audit and correct allergen conflicts and citations in data/qa_benchmark_merged_review.csv."""

import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(PROJECT_ROOT, "data", "qa_benchmark_merged_review.csv")
sys.stdout.reconfigure(encoding="utf-8")

with open(csv_path, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print(f"Total rows in benchmark: {len(rows)}")

# Known persona allergies mapping:
# P_ALLERGY_01 -> USER_CLINICAL_006: allergies = gluten, milk, egg, soy, peanut
# P_CKD_01 -> USER_CLINICAL_005: chronic kidney disease stage 3 (phosphorus, sodium, protein limit)
# P_PREG_01 -> USER_CLINICAL_004: pregnancy 1st/2nd trimester (avoid chemical preservatives, unpasteurized)
# P_DIAB_HYPER_01 -> USER_CLINICAL_002: hypertension & diabetes (sodium, sugar limit)
# P_CHILD_01 -> USER_CLINICAL_003: child 5yo (avoid azo colors INS 102, INS 110)
# P_OBESE_GOUT_01 -> USER_CLINICAL_007: healthy adult / gout

allergy_contradictions = []
for idx, r in enumerate(rows):
    pid = r.get("persona_id")
    pname = r.get("product_name", "")
    ing = r.get("product_ingredients", "").lower()
    gold_safety = r.get("gold_safety_label", "").upper().strip()
    domain = r.get("domain", "")

    if pid == "P_ALLERGY_01":
        allergens_found = []
        if any(t in ing for t in ["đậu nành"]):
            allergens_found.append("đậu nành (soy)")
        if any(t in ing for t in ["bột mì", "lúa mì"]):
            allergens_found.append("bột mì (gluten)")
        if any(t in ing for t in ["sữa", "bơ lạt", "bột sữa"]):
            allergens_found.append("sữa (milk protein)")
        if any(t in ing for t in ["trứng"]):
            allergens_found.append("trứng (egg)")
        if any(t in ing for t in ["hạnh nhân"]):
            allergens_found.append("hạnh nhân (tree nut - almond)")

        if allergens_found:
            allergy_contradictions.append({
                "row_index": idx,
                "id": r.get("id"),
                "product": pname,
                "domain": domain,
                "allergens": allergens_found,
                "current_gold": gold_safety,
                "current_binary": r.get("binary_answer"),
                "question": r.get("question"),
            })

print(f"Found {len(allergy_contradictions)} rows for P_ALLERGY_01 containing allergens!")
sample_counts = {}
for c in allergy_contradictions:
    key = (c["product"], tuple(c["allergens"]), c["current_gold"])
    sample_counts[key] = sample_counts.get(key, 0) + 1

for k, count in sample_counts.items():
    print(f"Product: {k[0]} | Allergens: {k[1]} | Current Gold: {k[2]} -> Count: {count}")
