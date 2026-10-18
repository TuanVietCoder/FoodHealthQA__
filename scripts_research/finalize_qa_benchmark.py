"""
Script chuyển bộ QA benchmark từ draft → gold sau khi review.
Đọc file CSV đã review, merge với JSONL draft, xuất bộ gold cuối cùng.

Chạy: python scripts/finalize_qa_benchmark.py
Input:  data/qa_benchmark_generated_review.csv (đã điền review)
        data/qa_benchmark_generated_draft.jsonl
Output: data/qa_benchmark_gold.jsonl
        data/qa_benchmark_gold_stats.json
"""

import json
import os
import csv
import sys
from collections import Counter
from datetime import date

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_reviews(csv_path):
    """Nạp file CSV review, trả về dict id → review_data."""
    reviews = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id", "").strip()
            if rid:
                reviews[rid] = row
    return reviews


def load_draft(jsonl_path):
    """Nạp JSONL draft."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def merge_and_finalize(draft_records, reviews):
    """Merge review vào draft, phân loại validated/rejected/pending."""
    gold = []
    rejected = []
    pending = []

    for rec in draft_records:
        rid = rec["id"]
        rev = reviews.get(rid, {})

        review_status = rev.get("review_status", "").strip().lower()
        answer_ok = rev.get("answer_supported_by_source", "").strip().lower()
        citation_ok = rev.get("citation_correct", "").strip().lower()
        safety_ok = rev.get("safety_label_correct", "").strip().lower()

        # Nếu reviewer đã đánh dấu approved/validated
        if review_status in ("approved", "validated", "gold"):
            rec["review_status"] = "validated"
            rec["is_expert_validated_gold"] = True
            rec["human_review_required"] = False
            rec["reviewer_id"] = rev.get("reviewer_id", "")
            rec["review_date"] = rev.get("review_date", date.today().isoformat())

            # Dùng corrected answer nếu có
            corrected = rev.get("corrected_answer", "").strip()
            if corrected:
                rec["reference_answer"] = corrected

            corrected_label = rev.get("corrected_safety_label", "").strip()
            if corrected_label:
                rec["gold_safety_label"] = corrected_label

            rec["review_notes"] = rev.get("review_notes", "")
            gold.append(rec)

        elif review_status in ("rejected", "invalid"):
            rec["review_status"] = "rejected"
            rec["review_notes"] = rev.get("review_notes", "")
            rejected.append(rec)

        elif answer_ok == "yes" and citation_ok == "yes" and safety_ok == "yes":
            # Tất cả đều yes nhưng chưa đánh approved → coi như validated
            rec["review_status"] = "validated"
            rec["is_expert_validated_gold"] = True
            rec["human_review_required"] = False
            rec["reviewer_id"] = rev.get("reviewer_id", "")
            rec["review_date"] = rev.get("review_date", date.today().isoformat())
            rec["review_notes"] = rev.get("review_notes", "")
            gold.append(rec)

        else:
            rec["review_status"] = "pending"
            pending.append(rec)

    return gold, rejected, pending


def main():
    draft_path = os.path.join(DATA_DIR, "qa_benchmark_generated_draft.jsonl")
    review_path = os.path.join(DATA_DIR, "qa_benchmark_generated_review.csv")
    gold_path = os.path.join(DATA_DIR, "qa_benchmark_gold.jsonl")
    stats_path = os.path.join(DATA_DIR, "qa_benchmark_gold_stats.json")

    if not os.path.isfile(draft_path):
        print("❌ Chưa có file draft. Chạy generate_qa_benchmark.py trước.")
        return
    if not os.path.isfile(review_path):
        print("❌ Chưa có file review CSV. Mở qa_benchmark_generated_review.csv và điền review.")
        return

    draft = load_draft(draft_path)
    reviews = load_reviews(review_path)
    gold, rejected, pending = merge_and_finalize(draft, reviews)

    # Xuất gold JSONL
    with open(gold_path, "w", encoding="utf-8") as f:
        for r in gold:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Thống kê
    domain_counts = Counter(r["domain"] for r in gold)
    label_counts = Counter(r["gold_safety_label"] for r in gold)
    stats = {
        "total_draft": len(draft),
        "total_reviewed": len(reviews),
        "gold_validated": len(gold),
        "rejected": len(rejected),
        "pending": len(pending),
        "domain_distribution": dict(sorted(domain_counts.items())),
        "safety_label_distribution": dict(sorted(label_counts.items())),
        "finalized_date": date.today().isoformat(),
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"KẾT QUẢ FINALIZE:")
    print(f"{'='*60}")
    print(f"  Tổng draft:      {len(draft)}")
    print(f"  Đã review:       {len(reviews)}")
    print(f"  ✅ Gold validated: {len(gold)}")
    print(f"  ❌ Rejected:      {len(rejected)}")
    print(f"  ⏳ Pending:       {len(pending)}")
    print(f"\nPhân bổ Gold theo Domain:")
    for d, c in sorted(domain_counts.items()):
        print(f"  {d:25s}: {c:4d}")
    print(f"\nPhân bổ Gold theo Safety Label:")
    for l, c in sorted(label_counts.items()):
        print(f"  {l:20s}: {c:4d}")
    print(f"\n✅ Xuất gold: {gold_path}")
    print(f"✅ Xuất stats: {stats_path}")


if __name__ == "__main__":
    main()
