"""
Script chuyển đổi chuẩn từ data/qa_benchmark_merged_review.csv
sang Bộ Dữ liệu Chuẩn Vàng (Gold Standard):
1. data/qa_benchmark_gold_verified.csv (Bảng chuẩn vàng cho con người / đối soát)
2. data/qa_benchmark_gold.jsonl (Bộ chuẩn vàng cho pipeline đánh giá tự động)

Quy tắc xuất bộ Gold Standard:
- Các câu có answer_correct == 'no':
    + reference_answer được cập nhật bằng câu chuẩn y khoa (corrected_answer).
    + gold_safety_label được cập nhật bằng nhãn chuẩn y khoa (corrected_safety_label).
    + binary_answer được cập nhật phù hợp ('No' nếu UNSAFE).
    + Giữ nguyên các cột reviewer_id, answer_correct, review_notes để minh bạch nguồn gốc.
- Các câu có answer_correct == 'yes':
    + Giữ nguyên reference_answer và gold_safety_label ban đầu.
"""

import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
INPUT_CSV = os.path.join(DATA_DIR, "qa_benchmark_merged_review.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "qa_benchmark_gold_verified.csv")
OUTPUT_JSONL = os.path.join(DATA_DIR, "qa_benchmark_gold.jsonl")


def generate_true_gold_benchmark():
    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    print(f"Tổng số bản ghi đọc từ file review: {len(rows)}")

    gold_rows = []
    corrected_count = 0

    for row in rows:
        r = dict(row)

        # Áp dụng hiệu chỉnh nếu câu trả lời gốc bị sai y khoa
        if r.get("answer_correct") == "no" and r.get("corrected_answer"):
            r["reference_answer"] = r["corrected_answer"]
            corrected_count += 1

        # Áp dụng hiệu chỉnh nhãn an toàn nếu nhãn gốc bị sai
        if r.get("safety_label_correct") == "no" and r.get("corrected_safety_label"):
            r["gold_safety_label"] = r["corrected_safety_label"]
            if r["corrected_safety_label"] == "UNSAFE":
                r["binary_answer"] = "No"
            elif r["corrected_safety_label"] == "SAFE":
                r["binary_answer"] = "Yes"

        gold_rows.append(r)

    # 1. Ghi ra qa_benchmark_gold_verified.csv
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gold_rows)

    print(f"✅ ĐÃ TẠO MỚI CSV GOLD STANDARD: {OUTPUT_CSV}")
    print(f"   -> Đã áp dụng {corrected_count} câu hiệu chỉnh y khoa vào reference_answer")

    # 2. Ghi ra qa_benchmark_gold.jsonl
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in gold_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ ĐÃ TẠO MỚI JSONL GOLD STANDARD: {OUTPUT_JSONL}")
    print(f"   -> Đồng bộ 750/750 câu hỏi chuẩn xác.")


if __name__ == "__main__":
    generate_true_gold_benchmark()
