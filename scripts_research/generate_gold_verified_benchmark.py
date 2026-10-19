"""
Script xuất đồng bộ bộ dữ liệu QA Benchmark từ data/qa_benchmark_merged_review.csv
sang:
1. data/qa_benchmark_gold_verified.csv (CSV đối soát chuẩn, khớp 100% nội dung thẩm định)
2. data/qa_benchmark_gold.jsonl (Bộ Gold cho pipeline đánh giá tự động)

Quy tắc:
- Khớp 100% nội dung từ qa_benchmark_merged_review.csv.
- Không thay đổi câu hỏi, câu trả lời, nhãn an toàn, hoặc ghi chú thẩm định so với file review.
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


def sync_gold_benchmark():
    # 1. Đọc dữ liệu chuẩn từ qa_benchmark_merged_review.csv
    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    print(f"Tổng số bản ghi đọc từ file review: {len(rows)}")

    # 2. Ghi ra qa_benchmark_gold_verified.csv (Khớp 100% fieldnames và dữ liệu)
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ ĐÃ TẠO MỚI CSV HOÀN TOÀN TRÙNG KHỚP: {OUTPUT_CSV}")

    # 3. Ghi ra qa_benchmark_gold.jsonl (Đồng bộ từng bản ghi dạng JSONL)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            # Ghi toàn bộ các trường y hệt từ hàng CSV vào JSON line
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ ĐÃ TẠO MỚI JSONL HOÀN TOÀN TRÙNG KHỚP: {OUTPUT_JSONL}")
    print(f"Tổng số câu hỏi được đồng bộ: {len(rows)} / {len(rows)}")


if __name__ == "__main__":
    sync_gold_benchmark()
