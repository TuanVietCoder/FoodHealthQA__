"""FoodHealthQA Evaluation Subset Builder.

Filters and stratifies the full benchmark (2,311 records) into a balanced, high-quality
evaluation dataset of 500-1,000 QA pairs across the 6 core pillars:
  1. An toan phu gia (Food additive safety & regulation)
  2. Di ung & Di nguyen (Allergen detection & cross-contact)
  3. Thai phu (Pregnant women)
  4. Tre em (Children)
  5. Benh nen (Underlying conditions: CKD, PKU, Diabetes/Hypertension)
  6. Dinh duong (Nutrition: Sodium & Sugars thresholds per Codex CAC/GL 23-1997)

Every record is grounded in verified legal/scientific primary sources:
  - VN_TT24_2019 / VN_09_VBHN_BYT_2024
  - CODEX_STAN_192_1995 (GSFA)
  - CODEX_CXS_1_1985 (Labelling of Prepackaged Foods)
  - CODEX_CAC_GL_23_1997 (Nutrition and Health Claims)
  - WHO_NSS_GUIDELINE_2023 (Non-sugar sweeteners)
  - EFSA opinions (Southampton colors, Caffeine, Phosphates in CKD, Nitrites, TiO2)
  - US_FDA_21CFR172_804 (PKU / Aspartame)

Usage:
  python scripts_research/build_foodhealthqa_subset.py [--size 750] [--seed 42] [--max-per-image 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "data" / "archive" / "master_benchmark_2311" / "qa_benchmark.jsonl"
    if (ROOT / "data" / "archive" / "master_benchmark_2311" / "qa_benchmark.jsonl").exists()
    else ROOT / "data" / "qa_benchmark.jsonl"
)
DEFAULT_OUT_JSONL = ROOT / "data" / "foodhealthqa_eval_set.jsonl"
DEFAULT_OUT_CSV = ROOT / "data" / "foodhealthqa_eval_set.csv"
DEFAULT_OUT_SUMMARY = ROOT / "data" / "foodhealthqa_eval_set_summary.json"

# Proportional domain weights targeted to create a balanced evaluation set
PILLAR_WEIGHTS = {
    "additive_safety": 0.22,      # An toàn phụ gia thực phẩm (~22%)
    "allergens": 0.22,            # Dị ứng & Dị nguyên (~22%)
    "children": 0.16,             # Trẻ nhỏ (~16%)
    "underlying_conditions": 0.15,# Bệnh nền (CKD, PKU, ĐTĐ/THA) (~15%)
    "pregnant_woman": 0.14,       # Phụ nữ mang thai (~14%)
    "nutrition": 0.11,            # Dinh dưỡng (Natri, Đường Codex) (~11%)
}

PILLAR_NAMES_VI = {
    "additive_safety": "An toàn phụ gia",
    "allergens": "Dị ứng & Dị nguyên",
    "children": "Trẻ em",
    "underlying_conditions": "Bệnh nền",
    "pregnant_woman": "Thai phụ",
    "nutrition": "Dinh dưỡng",
}


def classify_record(r: Dict[str, Any]) -> Tuple[str, str]:
    """Deterministically maps any record in qa_benchmark.jsonl into one of the 6 pillars."""
    pid = r.get("persona_id", "")
    t = r.get("template_id", "")
    dom = r.get("domain", "")
    auth = r.get("authority_topic", "")
    q = (r.get("question") or "").lower()

    # 1. Nutrition: Questions measuring nutritional limits (Codex sodium/sugars)
    if (
        auth in ["sugar_codex", "sodium_codex"]
        or t in [
            "T037", "T265", "T098", "T266", "T140", "T268",
            "T034", "T264", "T053", "T269", "T071", "T219", "T267", "T270",
        ]
        or "natri" in q
        or "đường tổng số" in q
    ):
        return "nutrition", PILLAR_NAMES_VI["nutrition"]

    # 2. Pregnant women
    if pid == "pregnant_woman" or auth in ["who_nss_pregnancy", "who_nss_pregnancy_absent"]:
        return "pregnant_woman", PILLAR_NAMES_VI["pregnant_woman"]

    # 3. Children
    if pid == "child_5yo" or auth in [
        "who_nss_child", "who_nss_child_absent", "efsa_caffeine_child", "efsa_caffeine_child_absent"
    ]:
        return "children", PILLAR_NAMES_VI["children"]

    # 4. Underlying conditions (CKD, PKU, Diabetes/Hypertension non-nutrition)
    if pid in ["ckd_stage3", "pku_patient"] or (pid == "diabetes_hypertension" and "bệnh" in q):
        return "underlying_conditions", PILLAR_NAMES_VI["underlying_conditions"]

    # 5. Allergen detection & cross-contact
    if dom == "allergen_detection" or pid == "allergy_patient" or "dị ứng" in q or "dị nguyên" in q:
        return "allergens", PILLAR_NAMES_VI["allergens"]

    # 6. Additive safety & regulation
    if (
        dom in ["additive_regulation", "insufficient_data"]
        or "phụ gia" in q
        or auth in [
            "eu_annex_v_colours", "iarc_v134_aspartame", "fda_red3_revoked",
            "codex_sulphite_declaration", "ins_listing", "ins_adi",
        ]
    ):
        return "additive_safety", PILLAR_NAMES_VI["additive_safety"]

    return "other", "Khác"


def load_benchmark(path: Path, tier: str = "all") -> List[Dict[str, Any]]:
    """Loads benchmark rows and enriches each with its category pillar."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("eligible_for_scoring") != "yes":
                continue
            if tier != "all" and r.get("tier") != tier:
                continue
            k, v = classify_record(r)
            r["category_key"] = k
            r["category_vi"] = v
            rows.append(r)
    return rows


def build_subset(
    rows: List[Dict[str, Any]],
    target_size: int = 750,
    seed: int = 42,
    max_per_image: int = 4,
    max_per_template: int = 30,
) -> List[Dict[str, Any]]:
    """Builds a stratified, balanced subset satisfying user-specified size."""
    if not (500 <= target_size <= 1000):
        print(f"Warning: target_size={target_size} outside recommended [500, 1000] range.")

    rng = random.Random(seed)

    # 1. Allocate quotas per pillar
    raw_quotas = {k: int(round(w * target_size)) for k, w in PILLAR_WEIGHTS.items()}
    diff = target_size - sum(raw_quotas.values())
    raw_quotas["additive_safety"] += diff

    img_cnt: Counter = Counter()
    t_cnt: Counter = Counter()
    selected: List[Dict[str, Any]] = []

    # 2. Stratified selection per pillar
    for cat, quota in raw_quotas.items():
        pool = [r for r in rows if r["category_key"] == cat]
        if not pool:
            continue

        label_counts = Counter(r["gold_safety_label"] for r in pool)
        total_pool = len(pool)

        # Allocate sub-quotas per safety label
        label_quotas: Dict[str, int] = {}
        for l, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            q_label = max(1, int(round((count / total_pool) * quota)))
            label_quotas[l] = q_label

        diff_l = quota - sum(label_quotas.values())
        top_l = sorted(label_counts.keys(), key=lambda x: -label_counts[x])[0]
        label_quotas[top_l] += diff_l

        # Sample for each label
        chosen_cat: List[Dict[str, Any]] = []
        for l, l_quota in label_quotas.items():
            l_pool = [r for r in pool if r["gold_safety_label"] == l]
            rng.shuffle(l_pool)
            l_chosen: List[Dict[str, Any]] = []

            # Pass 1: Strict adherence to image and template caps
            for r in l_pool:
                if img_cnt[r["split_group"]] < max_per_image and t_cnt[r["template_id"]] < max_per_template:
                    l_chosen.append(r)
                    img_cnt[r["split_group"]] += 1
                    t_cnt[r["template_id"]] += 1
                if len(l_chosen) >= l_quota:
                    break

            # Pass 2: Relax template cap if needed, keep image load as low as possible
            if len(l_chosen) < l_quota:
                remaining = [r for r in l_pool if r not in l_chosen]
                remaining.sort(key=lambda x: (img_cnt[x["split_group"]], t_cnt[x["template_id"]]))
                needed = l_quota - len(l_chosen)
                for r in remaining[:needed]:
                    l_chosen.append(r)
                    img_cnt[r["split_group"]] += 1
                    t_cnt[r["template_id"]] += 1

            chosen_cat.extend(l_chosen)

        selected.extend(chosen_cat)

    # Return sorted by ID for determinism
    return sorted(selected, key=lambda x: x["id"])


def export_subset(
    records: List[Dict[str, Any]],
    out_jsonl: Path,
    out_csv: Path,
    out_summary: Path,
    seed: int,
) -> Dict[str, Any]:
    """Exports records to JSONL, CSV, and summary JSON."""
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. Write JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2. Write CSV with the 4 mandatory fields prioritized
    csv_columns = [
        "id",
        "category_vi",
        "question",
        "reference_answer",
        "source_citations",
        "gold_safety_label",
        "product_name",
        "persona_name",
        "domain_vi",
        "category_key",
        "source_claim_locators",
        "source_claim_pages",
        "persona_id",
        "persona_conditions_vi",
        "image_ref",
        "split_group",
        "template_id",
        "tier",
    ]
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    # 3. Create summary statistics
    cat_counts = Counter(r["category_vi"] for r in records)
    label_counts = Counter(r["gold_safety_label"] for r in records)
    persona_counts = Counter(r["persona_id"] for r in records)
    tier_counts = Counter(r["tier"] for r in records)
    split_counts = Counter(r["split_group"] for r in records)
    tmpl_counts = Counter(r["template_id"] for r in records)

    summary = {
        "dataset_name": "FoodHealthQA Evaluation Benchmark Subset",
        "total_records": len(records),
        "target_range": "500-1000",
        "seed": seed,
        "distribution_by_pillar": dict(cat_counts.most_common()),
        "distribution_by_safety_label": dict(label_counts.most_common()),
        "distribution_by_persona": dict(persona_counts.most_common()),
        "distribution_by_tier": dict(tier_counts.most_common()),
        "diversity_metrics": {
            "unique_packaging_photos": len(split_counts),
            "max_questions_per_photo": max(split_counts.values()) if split_counts else 0,
            "unique_question_templates": len(tmpl_counts),
            "max_template_share": max(tmpl_counts.values()) if tmpl_counts else 0,
        },
        "mandatory_fields_coverage": {
            "question_populated_pct": 100.0 * sum(1 for r in records if r.get("question")) / len(records),
            "answer_populated_pct": 100.0 * sum(1 for r in records if r.get("reference_answer")) / len(records),
            "citations_populated_pct": 100.0 * sum(1 for r in records if r.get("source_citations")) / len(records),
            "domain_label_populated_pct": 100.0 * sum(1 for r in records if r.get("category_vi")) / len(records),
        },
        "files": {
            "jsonl": str(out_jsonl.relative_to(ROOT)),
            "csv": str(out_csv.relative_to(ROOT)),
        },
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Build FoodHealthQA evaluation subset.")
    parser.add_argument("--size", type=int, default=750, help="Target record count (500-1000, default: 750)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    parser.add_argument("--max-per-image", type=int, default=4, help="Max questions per packaging photo")
    parser.add_argument("--max-per-template", type=int, default=30, help="Max questions per question template")
    parser.add_argument("--tier", type=str, default="all", choices=["all", "core"], help="Benchmark tier")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source benchmark JSONL")
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL, help="Output JSONL path")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV, help="Output CSV path")
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_OUT_SUMMARY, help="Output summary JSON path")

    args = parser.parse_args()

    rows = load_benchmark(args.input, tier=args.tier)
    print(f"Loaded {len(rows)} eligible records from {args.input}")

    sub = build_subset(
        rows,
        target_size=args.size,
        seed=args.seed,
        max_per_image=args.max_per_image,
        max_per_template=args.max_per_template,
    )
    print(f"Sampled {len(sub)} records for evaluation benchmark.")

    summary = export_subset(sub, args.out_jsonl, args.out_csv, args.out_summary, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

