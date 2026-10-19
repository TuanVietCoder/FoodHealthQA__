"""
Tổng hợp 2 bộ QA benchmark thành file review CSV thống nhất:
  1. qa-benchmark-600-samples.json (600 câu tình huống persona + sản phẩm)
  2. qa_benchmark_600_source_grounded.jsonl (600 câu tra cứu kiến thức — lấy 1 variant/group)

Xuất:
  - qa_benchmark_merged_review.csv — CSV để review/kiểm chứng
  - qa_benchmark_merged_draft.jsonl — JSONL đầy đủ

Chạy: python scripts_research/merge_qa_benchmarks.py
"""

import json
import os
import csv
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_samples_json(path):
    """Nạp bộ qa-benchmark-600-samples.json và chuẩn hóa schema."""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for r in raw:
        gt = r.get("ground_truth", {})
        persona = r.get("user_persona", {})
        product = r.get("food_product", {})
        citations = r.get("source_citations", [])

        conds_str = ", ".join(persona.get("health_conditions", []))
        conds_vi_map = {
            "pregnancy, gestational_diabetes": "Mang thai (thai kỳ 24 tuần), đái tháo đường thai kỳ",
            "childhood_hyperactivity, sensitive_stomach": "Trẻ nhỏ 5 tuổi hiếu động (nguy cơ ADHD), dạ dày nhạy cảm",
            "diabetes_type_2, hypertension": "Đái tháo đường týp 2, tăng huyết áp mạn tính",
            "chronic_kidney_disease, hypertension": "Bệnh thận mạn tính (CKD giai đoạn 3), tăng huyết áp",
            "atopic_dermatitis": "Cơ địa dị ứng thực phẩm cấp tính, viêm da cơ địa",
            "obesity, gout": "Béo phì độ 2, bệnh gút (tăng acid uric máu)",
            "milk_allergy, atopic_asthma": "Dị ứng đạm sữa bò cấp tính, hen suyễn cơ địa",
            "lactose_intolerance": "Không dung nạp đường lactose (thiếu hụt enzyme lactase)",
            "soy_allergy, atopic_dermatitis": "Dị ứng đậu nành cấp tính, viêm da cơ địa",
            "peanut_allergy": "Dị ứng đậu phộng (lạc) cấp tính (nguy cơ sốc phản vệ)",
        }
        records.append({
            "id": r["id"],
            "source_file": "qa-benchmark-600-samples.json",
            "domain": r.get("domain_label", ""),
            "domain_vi": r.get("domain_name_vi", ""),
            "question_level": r.get("question_level", ""),
            "question": r.get("question", ""),
            "reference_answer": gt.get("explanation", ""),
            "binary_answer": gt.get("binary_answer", ""),
            "conflict_tags": ", ".join(gt.get("conflict_tags", [])),
            "persona_id": persona.get("persona_id", ""),
            "persona_name": persona.get("name", ""),
            "persona_conditions": conds_str,
            "persona_conditions_vi": conds_vi_map.get(conds_str, conds_str),
            "product_name": product.get("name", ""),
            "product_ingredients": ", ".join(product.get("ingredients", [])[:5]),
            "source_citations": "; ".join(
                c.get("source", "") for c in citations
            ) if citations else "",
            "gold_safety_label": "UNSAFE" if gt.get("binary_answer") == "No" else "SAFE",
            "review_status": "draft_needs_review",
        })
    return records


def load_source_grounded(path):
    """Nạp bộ source_grounded.jsonl, lấy 1 variant mỗi group."""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    domain_vi_map = {
        "additive_safety": "An toàn Phụ gia & Ngưỡng Pháp lý",
        "allergen_safety": "An toàn Dị ứng & Sốc phản vệ",
        "children_safety": "An toàn Trẻ em & Phẩm màu Azo",
        "condition_safety": "Chống chỉ định Bệnh nền Lâm sàng",
        "nutrition_facts": "Định lượng Dinh dưỡng & Calo FNDDS",
        "pregnancy_safety": "Tư vấn Thai phụ & Dinh dưỡng Thai kỳ",
    }

    seen_groups = set()
    records = []
    for r in raw:
        gid = r.get("question_group_id", r["id"])
        if gid in seen_groups:
            continue
        seen_groups.add(gid)

        domain = r.get("domain", "")
        records.append({
            "id": r["id"],
            "source_file": "qa_benchmark_600_source_grounded.jsonl",
            "domain": domain,
            "domain_vi": domain_vi_map.get(domain, ""),
            "question_level": r.get("question_variant", ""),
            "question": r.get("question", ""),
            "reference_answer": r.get("reference_answer", ""),
            "binary_answer": "",
            "conflict_tags": "",
            "persona_id": "",
            "persona_name": "",
            "persona_conditions": "",
            "persona_conditions_vi": "Tra cứu quy chuẩn chung (Không áp dụng persona)",
            "product_name": "",
            "product_ingredients": "",
            "source_citations": ", ".join(r.get("source_ids", [])),
            "gold_safety_label": r.get("gold_safety_label", "N/A"),
            "review_status": "draft_needs_review",
        })
    return records


def main():
    samples_path = os.path.join(DATA_DIR, "qa-benchmark-600-samples.json")
    grounded_path = os.path.join(DATA_DIR, "qa_benchmark_600_source_grounded.jsonl")
    merged_jsonl = os.path.join(DATA_DIR, "qa_benchmark_merged_draft.jsonl")
    merged_csv_path = os.path.join(DATA_DIR, "qa_benchmark_merged_review.csv")

    samples = load_samples_json(samples_path)
    grounded = load_source_grounded(grounded_path)

    print(f"Bộ samples (persona + sản phẩm): {len(samples)} câu")
    print(f"Bộ source_grounded (tra cứu, deduplicated): {len(grounded)} câu")

    merged = samples + grounded

    # Kiểm tra ID trùng → prefix nếu cần
    ids = [r["id"] for r in merged]
    if len(ids) != len(set(ids)):
        id_counts = Counter(ids)
        dupes = {k for k, v in id_counts.items() if v > 1}
        for r in grounded:
            if r["id"] in dupes:
                r["id"] = "SG-" + r["id"]
        print(f"  → Đã prefix {len(dupes)} ID trùng trong bộ source_grounded")

    # Xuất JSONL
    with open(merged_jsonl, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Xuất CSV review
    csv_fields = [
        "id", "source_file", "domain", "domain_vi", "question_level",
        "question", "reference_answer", "binary_answer", "conflict_tags",
        "persona_id", "persona_name", "persona_conditions", "persona_conditions_vi",
        "product_name", "product_ingredients", "source_citations",
        "gold_safety_label",
        # Cột dành cho reviewer điền
        "reviewer_id", "answer_correct", "citation_correct",
        "safety_label_correct", "corrected_answer", "corrected_safety_label",
        "review_notes", "review_date", "review_status",
    ]
    with open(merged_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in merged:
            row = {k: r.get(k, "") for k in csv_fields}
            writer.writerow(row)

    # Thống kê
    domain_counts = Counter(r.get("domain", "unknown") for r in merged)
    label_counts = Counter(r.get("gold_safety_label", "N/A") for r in merged)
    source_counts = Counter(r.get("source_file", "") for r in merged)

    print(f"\n{'='*60}")
    print(f"BỘ MERGED TỔNG HỢP: {len(merged)} câu")
    print(f"{'='*60}")
    print(f"\nTheo nguồn:")
    for s, c in source_counts.most_common():
        print(f"  {s}: {c} câu")
    print(f"\nTheo Domain:")
    for d, c in sorted(domain_counts.items()):
        print(f"  {d:30s}: {c:4d} câu")
    print(f"\nTheo Safety Label:")
    for l, c in sorted(label_counts.items()):
        print(f"  {l:20s}: {c:4d} câu")
    print(f"\n✅ Xuất JSONL: {merged_jsonl}")
    print(f"✅ Xuất CSV:   {merged_csv_path}")


if __name__ == "__main__":
    main()
