"""Thêm cột persona_conditions_vi vào data/qa_benchmark_merged_review.csv.

Mục tiêu:
1. Bổ sung cột 'persona_conditions_vi' ngay sau 'persona_conditions' giải thích rõ
   bệnh lý / cơ địa bằng tiếng Việt có dấu.
2. Với các câu hỏi tra cứu luật/quy chuẩn chung (145 câu không gán persona),
   ghi rõ 'Tra cứu quy chuẩn chung (Không áp dụng persona)'.
3. Điền bổ sung 'domain_vi' cho các câu còn thiếu để tiện tra cứu/lọc dữ liệu.
4. Bảo toàn 100% dữ liệu đã review (150 approved rows, reviewer_id, review_notes).
"""

import csv
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT_DIR, "data", "qa_benchmark_merged_review.csv")
BACKUP_PATH = os.path.join(ROOT_DIR, "data", "qa_benchmark_merged_review.csv.bak")

PERSONA_CONDITIONS_VI_MAP = {
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

DOMAIN_VI_MAP = {
    "additive_safety": "An toàn Phụ gia & Ngưỡng Pháp lý",
    "allergen_safety": "An toàn Dị ứng & Sốc phản vệ",
    "children_safety": "An toàn Trẻ em & Phẩm màu Azo",
    "condition_safety": "Chống chỉ định Bệnh nền Lâm sàng",
    "nutrition_facts": "Định lượng Dinh dưỡng & Calo FNDDS",
    "pregnancy_safety": "Tư vấn Thai phụ & Dinh dưỡng Thai kỳ",
}


def main():
    if not os.path.isfile(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    # Backup trước khi thao tác
    shutil.copyfile(CSV_PATH, BACKUP_PATH)
    print(f"Đã tạo bản sao lưu tại: {BACKUP_PATH}")

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    print(f"Tổng số dòng hiện tại: {len(rows)}")
    print(f"Số cột hiện tại: {len(fieldnames)}")

    # Thêm cột persona_conditions_vi ngay sau persona_conditions
    if "persona_conditions_vi" not in fieldnames:
        idx = fieldnames.index("persona_conditions")
        fieldnames.insert(idx + 1, "persona_conditions_vi")
        print(f"Đã chèn cột 'persona_conditions_vi' vào vị trí {idx + 1}")

    # Xử lý từng dòng
    updated_count = 0
    for row in rows:
        cond_raw = (row.get("persona_conditions") or "").strip()
        persona_id = (row.get("persona_id") or "").strip()

        if cond_raw in PERSONA_CONDITIONS_VI_MAP:
            row["persona_conditions_vi"] = PERSONA_CONDITIONS_VI_MAP[cond_raw]
        elif not persona_id:
            row["persona_conditions_vi"] = "Tra cứu quy chuẩn chung (Không áp dụng persona)"
        else:
            row["persona_conditions_vi"] = cond_raw

        # Bổ sung domain_vi nếu còn trống
        domain = (row.get("domain") or "").strip()
        if not (row.get("domain_vi") or "").strip() and domain in DOMAIN_VI_MAP:
            row["domain_vi"] = DOMAIN_VI_MAP[domain]

        updated_count += 1

    # Ghi lại file
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã cập nhật {updated_count} dòng vào {CSV_PATH}")
    print(f"Số cột mới: {len(fieldnames)}")

    # Thống kê nhanh
    from collections import Counter
    vi_counts = Counter(r["persona_conditions_vi"] for r in rows)
    print("\nThống kê phân bổ cột 'persona_conditions_vi':")
    for k, v in vi_counts.most_common():
        print(f"  - {k}: {v} dòng")


if __name__ == "__main__":
    main()
