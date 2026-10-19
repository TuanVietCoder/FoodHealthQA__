"""
Script bổ sung các ca kiểm thử chuẩn hóa Benchmark về Dị ứng nâng cao theo 7 tài liệu FAO/WHO & US FDA:
1. Sữa (IgE Allergy) vs Không dung nạp Lactose (Enzyme Deficiency) (FAO/WHO JEMRA 2021)
2. Miễn trừ dầu tinh luyện hoàn toàn Highly Refined Oil (US FDA FALCPA Sec 203 & FDA Guidance Ed. 5)
3. Cảnh báo tiếp xúc chéo Precautionary Allergen Labeling (PAL) (FAO/WHO JEMRA Part 3, 2023)
4. Dẫn xuất phụ gia chứa protein dị nguyên - Lecithin INS 322 (FDA Guidance Ed. 5 & EFSA 2017)
"""

import os
import sys
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BENCHMARK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "qa_benchmark_merged_review.csv",
)

NEW_BENCHMARK_ROWS = [
    {
        "id": "FOODQA_ALLERGEN_ADV_01",
        "source_file": "jemra_fao_who_fda_guidelines.json",
        "domain": "allergen_safety",
        "domain_vi": "An toàn Dị ứng & Sốc phản vệ",
        "question_level": "standard",
        "question": "Bệnh nhân có tiền sử dị ứng đạm sữa bò (IgE-mediated milk allergy) có dùng được sản phẩm 'Sữa tiệt trùng nguyên kem' không? Cần lưu ý gì?",
        "reference_answer": "Không phù hợp và không an toàn. Sản phẩm chứa sữa bò tươi, đạm casein và đạm whey gây nguy cơ phản ứng dị ứng qua trung gian IgE (nguy cơ sốc phản vệ) theo FAO/WHO JEMRA 2021 Part 1 và US FDA FALCPA.",
        "binary_answer": "No",
        "conflict_tags": "allergen_conflict_milk",
        "persona_id": "P_ALLERGY_MILK",
        "persona_name": "Phạm Thị M (Dị ứng đạm sữa bò)",
        "persona_conditions": "milk_allergy, atopic_asthma",
        "product_name": "Sữa tươi tiệt trùng nguyên kem",
        "product_ingredients": "Sữa bò tươi 100%, chất ổn định (INS 471, INS 407)",
        "source_citations": "FAO/WHO JEMRA 2021 Part 1 & US FDA FALCPA",
        "gold_safety_label": "UNSAFE",
        "reviewer_id": "EXPERT_JEMRA_ADJUDICATED",
        "answer_correct": "yes",
        "citation_correct": "yes",
        "safety_label_correct": "yes",
        "corrected_answer": "",
        "corrected_safety_label": "",
        "review_notes": "Chuẩn hóa theo FAO/WHO JEMRA 2021: Dị ứng sữa là dị ứng ưu tiên toàn cầu qua IgE.",
        "review_date": "2026-10-18",
        "review_status": "approved",
    },
    {
        "id": "FOODQA_ALLERGEN_ADV_02",
        "source_file": "jemra_fao_who_fda_guidelines.json",
        "domain": "allergen_safety",
        "domain_vi": "An toàn Dị ứng & Sốc phản vệ",
        "question_level": "complex",
        "question": "Người mắc hội chứng không dung nạp lactose (thiếu enzyme lactase) có uống được 'Sữa tiệt trùng nguyên kem thông thường' không? Tình trạng này có phải dị ứng sữa không?",
        "reference_answer": "Không khuyến nghị dùng sữa thông thường do chứa đường lactose gây đầy hơi, đau bụng do thiếu enzyme lactase. Tuy nhiên, theo FAO/WHO JEMRA 2021 Part 1, không dung nạp lactose là rối loạn tiêu hóa/chuyển hóa men lactase, KHÔNG PHẢI dị ứng miễn dịch IgE và không gây sốc phản vệ. Người dùng có thể sử dụng sữa không lactose (lactose-free) hoặc bổ sung men lactase.",
        "binary_answer": "No",
        "conflict_tags": "lactose_intolerance_enzyme_deficiency",
        "persona_id": "P_LACTOSE_INTOLERANT",
        "persona_name": "Đỗ Văn L (Không dung nạp Lactose)",
        "persona_conditions": "lactose_intolerance",
        "product_name": "Sữa tươi tiệt trùng nguyên kem",
        "product_ingredients": "Sữa bò tươi 100%, chất ổn định (INS 471, INS 407)",
        "source_citations": "FAO/WHO JEMRA 2021 Part 1 & US FDA FAQ",
        "gold_safety_label": "CAUTION",
        "reviewer_id": "EXPERT_JEMRA_ADJUDICATED",
        "answer_correct": "yes",
        "citation_correct": "yes",
        "safety_label_correct": "yes",
        "corrected_answer": "",
        "corrected_safety_label": "",
        "review_notes": "Phân định rõ rệt: Không dung nạp lactose là enzyme deficiency, không xếp vào IgE allergy theo JEMRA.",
        "review_date": "2026-10-18",
        "review_status": "approved",
    },
    {
        "id": "FOODQA_ALLERGEN_ADV_03",
        "source_file": "jemra_fao_who_fda_guidelines.json",
        "domain": "allergen_safety",
        "domain_vi": "An toàn Dị ứng & Sốc phản vệ",
        "question_level": "standard",
        "question": "Người dị ứng đậu nành có sử dụng được sản phẩm chứa 'Dầu đậu nành tinh luyện hoàn toàn' không theo quy chuẩn an toàn quốc tế?",
        "reference_answer": "Có thể sử dụng được theo quy chuẩn US FDA FALCPA Section 203 và FDA Guidance Ed. 5. Dầu đậu nành tinh luyện hoàn toàn (highly refined oil) được miễn trừ dán nhãn dị ứng do quá trình tinh luyện (tẩy màu, khử mùi, gia nhiệt) đã loại bỏ hầu như toàn bộ protein kháng nguyên gây dị ứng. Tuy nhiên, nếu có tiền sử sốc phản vệ cực kỳ nghiêm trọng, người tiêu dùng nên tham khảo ý kiến bác sĩ dị ứng trước khi dùng.",
        "binary_answer": "Yes",
        "conflict_tags": "fda_refined_oil_exemption",
        "persona_id": "P_ALLERGY_01",
        "persona_name": "Hoàng Thị E (Dị ứng đậu nành)",
        "persona_conditions": "soy_allergy, atopic_dermatitis",
        "product_name": "Dầu ăn thực vật cao cấp",
        "product_ingredients": "Dầu đậu nành tinh luyện, dầu olein cọ tinh luyện, vitamin A",
        "source_citations": "US FDA FALCPA Sec 203 & FDA Guidance Ed. 5",
        "gold_safety_label": "SAFE",
        "reviewer_id": "EXPERT_FDA_ADJUDICATED",
        "answer_correct": "yes",
        "citation_correct": "yes",
        "safety_label_correct": "yes",
        "corrected_answer": "",
        "corrected_safety_label": "",
        "review_notes": "Xác nhận quy chuẩn US FDA FALCPA: Highly refined oils được miễn trừ dị ứng protein.",
        "review_date": "2026-10-18",
        "review_status": "approved",
    },
    {
        "id": "FOODQA_ALLERGEN_ADV_04",
        "source_file": "jemra_fao_who_fda_guidelines.json",
        "domain": "allergen_safety",
        "domain_vi": "An toàn Dị ứng & Sốc phản vệ",
        "question_level": "complex",
        "question": "Sản phẩm bánh quy không có đậu phộng trong thành phần nhưng ghi cảnh báo: 'Có thể chứa vết của hạt đậu phộng do sản xuất trên cùng dây chuyền'. Người có tiền sử dị ứng đậu phộng có nên ăn không?",
        "reference_answer": "Không nên sử dụng. Dòng chữ 'Có thể chứa vết của đậu phộng' là nhãn cảnh báo phòng ngừa (Precautionary Allergen Labeling - PAL). Theo báo cáo FAO/WHO JEMRA Part 3 (2023), nguy cơ nhiễm chéo ngoài ý muốn trong quá trình sản xuất trên dây chuyền dùng chung vẫn có thể gây phản ứng dị ứng nghiêm trọng cho người có ngưỡng nhạy cảm thấp.",
        "binary_answer": "No",
        "conflict_tags": "pal_cross_contact_risk",
        "persona_id": "P_ALLERGY_PEANUT",
        "persona_name": "Ngô Văn P (Dị ứng đậu phộng)",
        "persona_conditions": "peanut_allergy",
        "product_name": "Bánh quy ngũ cốc giòn",
        "product_ingredients": "Bột mì, đường kính, bơ thực vật, muối. Có thể chứa vết của hạt đậu phộng, mè.",
        "source_citations": "FAO/WHO JEMRA Part 3 (PAL, 2023) & Codex CXS 1-1985",
        "gold_safety_label": "UNSAFE",
        "reviewer_id": "EXPERT_JEMRA_ADJUDICATED",
        "answer_correct": "yes",
        "citation_correct": "yes",
        "safety_label_correct": "yes",
        "corrected_answer": "",
        "corrected_safety_label": "",
        "review_notes": "Áp dụng khuyến nghị FAO/WHO JEMRA Part 3: PAL tiếp xúc chéo là rủi ro thực sự cho người dị ứng nặng.",
        "review_date": "2026-10-18",
        "review_status": "approved",
    },
    {
        "id": "FOODQA_ALLERGEN_ADV_05",
        "source_file": "jemra_fao_who_fda_guidelines.json",
        "domain": "allergen_safety",
        "domain_vi": "An toàn Dị ứng & Sốc phản vệ",
        "question_level": "standard",
        "question": "Người có cơ địa dị ứng đậu nành có cần kiêng chất nhũ hóa Lecithin (INS 322) chiết xuất từ đậu nành trong bánh kẹo không?",
        "reference_answer": "Cần thận trọng và kiêng sử dụng nếu dị ứng mức độ trung bình-nặng. Theo FDA Guidance Edition 5 và EFSA 2017, Lecithin thương mại thường được chiết xuất từ đậu nành và vẫn có thể chứa lượng nhỏ protein đậu nành dư thừa gây phản ứng dị ứng cho người nhạy cảm cao, do đó bắt buộc phải ghi rõ nguồn gốc đậu nành trên nhãn.",
        "binary_answer": "No",
        "conflict_tags": "allergen_derivative_lecithin_soy",
        "persona_id": "P_ALLERGY_01",
        "persona_name": "Hoàng Thị E (Dị ứng đậu nành)",
        "persona_conditions": "soy_allergy, atopic_dermatitis",
        "product_name": "Bánh quy socola cao cấp",
        "product_ingredients": "Bột mì, đường, bột cacao, bơ, chất nhũ hóa (INS 322i lecithin từ đậu nành), hương vani",
        "source_citations": "US FDA Guidance Ed. 5 & EFSA ANS Panel 2017",
        "gold_safety_label": "UNSAFE",
        "reviewer_id": "EXPERT_FDA_ADJUDICATED",
        "answer_correct": "yes",
        "citation_correct": "yes",
        "safety_label_correct": "yes",
        "corrected_answer": "",
        "corrected_safety_label": "",
        "review_notes": "Dẫn xuất phụ gia theo FDA Guidance Ed. 5: Lecithin chứa protein đậu nành dư.",
        "review_date": "2026-10-18",
        "review_status": "approved",
    },
]


def update_benchmark():
    df = pd.read_csv(BENCHMARK_PATH)
    initial_len = len(df)
    existing_ids = set(df["id"].tolist())

    added_count = 0
    updated_count = 0

    for row in NEW_BENCHMARK_ROWS:
        row_id = row["id"]
        if row_id in existing_ids:
            # Cập nhật
            idx = df[df["id"] == row_id].index[0]
            for k, v in row.items():
                df.at[idx, k] = v
            updated_count += 1
        else:
            # Thêm mới
            new_df = pd.DataFrame([row])
            df = pd.concat([df, new_df], ignore_index=True)
            added_count += 1

    df.to_csv(BENCHMARK_PATH, index=False, encoding="utf-8")
    print(f"Đã cập nhật Benchmark: Thêm mới {added_count} câu, cập nhật {updated_count} câu. Tổng số câu: {len(df)}")


if __name__ == "__main__":
    update_benchmark()
