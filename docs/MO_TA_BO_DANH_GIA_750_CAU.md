# Mô tả bộ đánh giá FoodHealthQA (750 câu)

Bộ `data/foodhealthqa_eval_set.jsonl` (kèm `.csv` và `_summary.json`) là **bộ đánh giá duy nhất** của hệ thống: mọi script chấm điểm đều dùng bộ này. Giao diện hỏi đáp **không** dùng bộ này làm tri thức hay ví dụ; chế độ "Báo cáo nghiên cứu" của giao diện chỉ đọc bộ này để hiển thị kết quả đã chấm. Mọi số liệu dưới đây được tính trực tiếp từ file dữ liệu.

## 1. Tổng quan

| Mục | Giá trị |
|---|---|
| Số câu hỏi–đáp | **750** |
| Số ảnh nhãn bao bì | **376** (mỗi ảnh có 1–4 câu; 142 ảnh 1 câu, 138 ảnh 2 câu, 52 ảnh 3 câu, 44 ảnh 4 câu) |
| Số mẫu câu hỏi (`template_id`) | 150 |
| Số hồ sơ người dùng (persona) | 7 |
| Số mã INS khác nhau xuất hiện | 63 (trong 547 câu có mã INS) |
| Số nguồn trích dẫn khác nhau | 18 mã nguồn |
| Trạng thái | Silver kiểm chứng bằng quy tắc, **không có chuyên gia** (`expert_validated = false` ở cả 750 bản ghi) |
| Phiên bản | Tập con của bản nguồn v5 (2.311 bản ghi); cùng mã câu hỏi và cùng đáp án với bản nguồn |

## 2. Cách dựng

Bộ 750 được chọn từ bản nguồn v5 (2.311 bản ghi) bằng `scripts_research/build_foodhealthqa_subset.py` (mặc định `--size 750 --seed 42 --max-per-image 4`), lấy mẫu phân tầng theo 6 trụ cột chuyên môn và theo nhãn an toàn, có trần số câu mỗi ảnh và mỗi mẫu câu hỏi. Dựng lại: `python scripts_research/build_foodhealthqa_subset.py` (cùng bản nguồn và seed cho ra cùng bộ 750).

Các bước chọn chi tiết, bảng bản nguồn → được chọn theo từng trụ cột, và toàn bộ quy trình sinh và kiểm chứng câu hỏi–đáp án trước đó xem [BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md](BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md), mục 1.

## 3. Cấu trúc một bản ghi

Bốn trường bắt buộc của FoodHealthQA (đều đầy đủ 100%):

| Trường | Ý nghĩa |
|---|---|
| `question` | Câu hỏi tiếng Việt (trung bình khoảng 155 ký tự) |
| `reference_answer` | Câu trả lời chuẩn, mở đầu bằng kết luận, nêu giới hạn thông tin trên nhãn, kèm thẻ nguồn `[MÃ_NGUỒN]` (trung bình khoảng 644 ký tự) |
| `source_citations` | Danh sách mã nguồn (1–4 mã mỗi câu) |
| `category_vi` | Nhãn lĩnh vực: một trong 6 trụ cột ở mục 4.1 |

Các trường quan trọng khác:

| Trường | Ý nghĩa |
|---|---|
| `gold_safety_label` | Nhãn an toàn chuẩn: SAFE, CAUTION, UNSAFE, CRITICAL_ALERT, ABSTAIN. Nhãn mịn của bản v5 nằm ở `label_v5` (thêm PERMITTED_LISTED, NOT_DECLARED) |
| `domain` / `domain_vi` | Nhãn miền thứ hai gồm 4 giá trị (mục 4.4); khác với `category_vi` |
| `source_claim_locators`, `source_claim_pages` | Vị trí cụ thể trong tài liệu nguồn (có ở 750/750 câu; số trang có ở 688 câu) |
| `ins_codes_matched` | Mã INS liên quan (547 câu có giá trị) |
| `persona_id`, `persona_name`, `persona_conditions_vi` | Hồ sơ người dùng gắn với câu hỏi |
| `image_ref`, `label_transcription_path`, `label_locator` | Ảnh nhãn và bản phiên âm chuẩn của nhãn |
| `split_group` | Mã nhóm: một ảnh là một nhóm; khi chia dữ liệu hoặc lấy mẫu lại phải chia theo trường này |
| `template_id` | Mẫu câu hỏi; nên báo cáo kết quả theo mẫu |
| `tier` | `core` hoặc `silver` (mục 4.5) |
| `abstain_reason` | Lý do cần từ chối với câu ABSTAIN |
| `review_status`, `source_verification_level` | Mức kiểm chứng (mục 6) |

## 4. Phân bố

### 4.1. Theo trụ cột chuyên môn (`category_vi`)

| Trụ cột | Số câu | Tỷ lệ |
|---|---:|---:|
| An toàn phụ gia | 166 | 22,1% |
| Dị ứng & Dị nguyên | 165 | 22,0% |
| Trẻ em | 120 | 16,0% |
| Bệnh nền | 112 | 14,9% |
| Thai phụ | 105 | 14,0% |
| Dinh dưỡng | 82 | 10,9% |

### 4.2. Theo nhãn an toàn (`gold_safety_label`)

| Nhãn | Số câu | Tỷ lệ |
|---|---:|---:|
| SAFE | 285 | 38,0% |
| CAUTION | 277 | 36,9% |
| UNSAFE | 85 | 11,3% |
| ABSTAIN | 72 | 9,6% |
| CRITICAL_ALERT | 31 | 4,1% |

Nhãn ABSTAIN có hai lý do: nhãn thiếu thông tin (46 câu, `LABEL_MISSING_INFO`) và không có câu cảnh báo nhiễm chéo trên nhãn (26 câu, `PAL_NOT_ON_LABEL`).

### 4.3. Nhãn an toàn theo trụ cột

| Trụ cột | SAFE | CAUTION | UNSAFE | CRITICAL_ALERT | ABSTAIN |
|---|---:|---:|---:|---:|---:|
| An toàn phụ gia | 100 | 20 | 0 | 0 | 46 |
| Dị ứng & Dị nguyên | 0 | 35 | 85 | 19 | 26 |
| Trẻ em | 63 | 57 | 0 | 0 | 0 |
| Bệnh nền | 60 | 40 | 0 | 12 | 0 |
| Thai phụ | 47 | 58 | 0 | 0 | 0 |
| Dinh dưỡng | 15 | 67 | 0 | 0 | 0 |

### 4.4. Theo miền (`domain`) và hồ sơ người dùng

| Miền | Số câu |
|---|---:|
| `vulnerable_groups` (Nhóm dễ tổn thương) | 419 |
| `allergen_detection` (Phát hiện dị nguyên) | 138 |
| `additive_regulation` (Đánh giá phụ gia) | 126 |
| `insufficient_data` (Thiếu dữ liệu) | 67 |

| Persona | Số câu |
|---|---:|
| healthy_adult | 166 |
| allergy_patient | 165 |
| child_5yo | 120 |
| pregnant_woman | 105 |
| diabetes_hypertension | 82 |
| ckd_stage3 | 81 |
| pku_patient | 31 |

### 4.5. Theo tầng (`tier`)

| Tầng | Số câu | Ý nghĩa sử dụng |
|---|---:|---|
| `core` | 339 | Dùng cho so sánh chính |
| `silver` | 411 | Báo cáo riêng; gồm toàn bộ câu Trẻ em (120), Bệnh nền (112), Thai phụ (105) và 42 câu Dinh dưỡng, 22 câu An toàn phụ gia, 10 câu Dị ứng |

## 5. Nguồn dẫn chứng

18 mã nguồn, số câu trích dẫn mỗi mã (một câu có thể trích nhiều mã):

| Mã nguồn | Số câu |
|---|---:|
| CODEX_CXS_1_1985 | 168 |
| US_FDA_FALCPA | 164 |
| WHO_JEMRA_PRIORITY_ALLERGENS | 147 |
| WHO_NSS_GUIDELINE_2023 | 141 |
| EFSA_OPINIONS | 137 |
| VN_TT24_2019 | 125 |
| CODEX_STAN_192_1995 | 88 |
| FAO_WHO_JEMRA_PART_3 | 82 |
| CODEX_CAC_GL_23_1997 | 82 |
| US_FDA_21CFR172_804 | 31 |
| JECFA_EVALUATIONS | 22 |
| EFSA_CAFFEINE_2015 | 20 |
| CODEX_CXG_36_1989 | 17 |
| EU_REG_1333_2008_ANNEX_V | 10 |
| EFSA_2021_TITANIUM_DIOXIDE_E171 | 9 |
| IARC_MONOGRAPHS_V134 | 4 |
| US_FDA_RED3_REVOCATION | 2 |
| US_FDA_ALLERGEN_QA_ED5 | 1 |

Tên tài liệu, mục/điều khoản và URL của từng mã nằm trong `knowledge_base/sources/catalog/source_citation_catalog.json`.

## 6. Mức kiểm chứng

Cả 750 bản ghi có `review_status = approved_rule_verified` (kiểm bằng quy tắc, không có người thẩm định); 741 bản đạt `label_and_primary_source_text_verified` và 9 bản ở mức `label_only_no_regulatory_assertion`. Các cổng kiểm tra, bảng khẳng định nguồn và kết quả kiểm nhãn với ảnh xem [BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md](BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md), mục 2–6.

## 7. Giới hạn cần nêu khi dùng số liệu

Các giới hạn về cấu trúc bộ nằm ở đây. Giới hạn về độ tin cậy của nhãn và kiểm chứng (không có chuyên gia, vòng tròn quy tắc ↔ guardrail, kiểm nhãn do một người thực hiện, chưa đối chiếu pháp quy Việt Nam về dị nguyên) xem [BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md](BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md), mục 9.

1. **Nhãn an toàn gần như phụ thuộc vào trụ cột** (bảng 4.3). Ví dụ Trẻ em, Thai phụ, Dinh dưỡng và Bệnh nền không có câu UNSAFE hay ABSTAIN; UNSAFE chỉ có ở nhóm Dị ứng. Một baseline chỉ đoán nhãn đa số của từng trụ cột đã đạt khoảng 57,7% (433/750), so với 38,0% (285/750) nếu chỉ đoán SAFE. Vì vậy nên báo cáo kết quả theo trụ cột và so sánh với các baseline này, không chỉ độ chính xác tổng.
2. **Nhóm CRITICAL_ALERT nhỏ** (31 câu) và nhãn ABSTAIN chỉ có hai lý do; cần báo cáo kèm khoảng tin cậy.
3. **Mẫu câu hỏi chưa thật đều.** Một mẫu (`T256`) chiếm 85 câu (11,3%), vượt mức trần 30 câu/mẫu vì vòng 2 của thuật toán nới giới hạn khi chưa đủ hạn ngạch. Nên báo cáo kết quả theo `template_id`.
4. **Tầng silver chiếm 54,8% (411/750)** và toàn bộ ba trụ cột nhóm dễ tổn thương nằm trong tầng này; kết quả hai tầng cần báo cáo riêng.
5. **Chỉ một hồ sơ người dùng cho mỗi câu** và 7 hồ sơ cố định; chưa phủ tổ hợp bệnh nền hay dị ứng khác.
6. Nhãn nhóm dễ tổn thương và nhóm natri là **cờ theo văn bản của cơ quan được trích**, không phải chẩn đoán hay khuyến cáo lâm sàng.

## 8. Cách dùng trong hệ thống

| Mục đích | Lệnh / vị trí |
|---|---|
| Sinh dự đoán A/B/C | `python scripts_research/generate_predictions.py --llm-provider gemini --resume` |
| Chấm điểm | `python scripts_research/run_evaluation.py --run-dir data/runs/gemini` |
| So sánh hai mô hình | `python scripts_research/compare_providers.py` |
| Đo truy xuất đoạn bằng chứng | `python scripts_research/eval_passage_retrieval.py` |
| Báo cáo cân bằng nhãn, baseline, khoảng tin cậy | `python scripts_research/benchmark_report.py --eval-750` |

Mọi script mặc định đọc bộ 750 câu; nếu file bị thiếu, script báo lỗi thay vì dùng bộ khác. Cách tạo bộ 750 và cách kiểm chứng đáp án, nguồn xem [BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md](BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md). Bản nguồn 2.311 bản ghi chỉ dùng để dựng bộ 750, không dùng để đánh giá.
