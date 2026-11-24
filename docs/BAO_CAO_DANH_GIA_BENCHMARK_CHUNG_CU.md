# Báo cáo Cách Tạo và Kiểm chứng Bộ Đánh giá FoodHealthQA (750 câu)

Báo cáo này trả lời hai câu hỏi: **bộ 750 câu được tạo ra như thế nào** (mục 1) và **đáp án, nhãn của nó được kiểm chứng ra sao, dựa trên nguồn nào, còn giới hạn gì** (mục 2 trở đi). Cấu trúc bản ghi và các bảng phân bố nằm ở [MO_TA_BO_DANH_GIA_750_CAU.md](MO_TA_BO_DANH_GIA_750_CAU.md); báo cáo này không lặp lại các bảng đó.

| Thuộc tính | Giá trị |
|---|---|
| Bộ được kiểm chứng | `data/foodhealthqa_eval_set.{jsonl,csv}`: **750 câu**, **376 ảnh nhãn** (bộ đánh giá duy nhất của hệ thống) |
| Xuất xứ | Tập con của bản nguồn v5 (2.311 bản ghi, `data/qa_benchmark.*`); cùng mã câu hỏi và cùng đáp án với bản nguồn, nên thừa hưởng toàn bộ kiểm chứng của bản nguồn |
| Trạng thái | **Silver kiểm chứng bằng quy tắc, không có chuyên gia** (`expert_validated = false`; không đủ điều kiện gold) |
| Phương pháp kiểm chứng | `rule_based_gates_G1_G6_plus_label_text_adjudication` (750/750 bản ghi) |
| Cập nhật lần cuối | 2026-11-13 |
| Mục đích sử dụng | Phát triển hệ thống và đánh giá độ trung thực/an toàn; **chưa thay thế tư vấn y khoa lâm sàng** |

> **Nguyên tắc nền:** một bản ghi không có nguồn tra cứu kiểm chứng được thì **không** đưa vào bộ QA, vì bộ QA là chuẩn để chấm hệ thống. Hệ quả là bộ có phạm vi hẹp hơn nhu cầu thực tế (mục 9), nhưng mọi khẳng định trong đáp án đều truy được tới văn bản gốc.

---

## 1. Cách tạo bộ 750 câu

Bộ 750 không được viết riêng từ đầu: nó là **tập con chọn ra từ bản nguồn v5** (2.311 bản ghi, `data/qa_benchmark.*`), và bản nguồn này đã qua toàn bộ quy trình sinh và kiểm chứng. Vì vậy mỗi câu trong bộ 750 có cùng mã `id` và cùng đáp án với bản ghi tương ứng của bản nguồn (đã đối chiếu: 750/750 mã có trong bản nguồn và 750/750 đáp án giống hệt).

| Giai đoạn | Việc làm | Kết quả | Script / tệp truy vết |
|---|---|---|---|
| 1. Dữ liệu nền | Ảnh bao bì thực phẩm bán tại Việt Nam và bản chép nhãn (thành phần, phụ gia, cảnh báo, dinh dưỡng); đối chiếu bản chép với ảnh gốc | 493 ảnh, 493 tệp nhãn | `assets/images/`, `assets/labels/`, `data/label_audit_*.csv` |
| 2. Sinh câu hỏi và đáp án | Sinh câu hỏi theo mẫu từ nội dung nhãn; đáp án dẫn xuất bằng quy tắc, mở đầu bằng kết luận, nêu giới hạn thông tin trên nhãn và kết thúc bằng thẻ nguồn | Bản nguồn 2.311 bản ghi, 489 ảnh, 278 mẫu câu hỏi (`template_id`) | `create_benchmark_v5.py`, `add_authority_records_v5.py` |
| 3. Kiểm chứng và loại bỏ | Đối chiếu nguyên văn khẳng định nguồn; chạy 6 cổng G1–G6; bản ghi không có nguồn kiểm chứng được thì loại | 2.311/2.311 qua cổng; 303 bản ghi bị loại ở các bước trước (mục 8) | `verify_external_claims.py`, `verify_v5_answers.py`, `finalize_benchmark_v5.py` |
| 4. Phân tầng | Gán tầng `core` (dùng cho so sánh chính) hoặc `silver` (báo cáo riêng) | 1.498 `core`, 813 `silver` | `finalize_benchmark_v5.py` |
| 5. Chọn 750 câu | Lấy mẫu phân tầng theo trụ cột, nhãn an toàn, ảnh và mẫu câu hỏi | 750 bản ghi, 376 ảnh, 150 mẫu câu hỏi | `build_foodhealthqa_subset.py` |

### 1.1. Giai đoạn 2: các nhóm câu hỏi và cách sinh

Câu hỏi được sinh theo mẫu; mỗi bản ghi mang `domain` (4 miền: dị nguyên, phụ gia, thiếu dữ liệu, nhóm dễ tổn thương) và `template_id`. Ngoài nhóm cơ bản về dị nguyên và phụ gia, các nhóm sau được bổ sung từng bước, mỗi nhóm gắn với đúng một nguồn thẩm quyền và chỉ khẳng định điều nguồn đó nói. Số câu của từng nhóm trong bộ 750 xem bảng `label_scope` ở mục 4.

- **Dị nguyên và nhiễm chéo:** nhãn dị nguyên đối chiếu với bộ tái suy luận riêng (từ điển dị nguyên viết tách khỏi bộ máy v4, nhưng cùng một người xây dựng nên không độc lập tuyệt đối); câu ABSTAIN dùng cho nhãn thiếu thông tin hoặc không có câu cảnh báo nhiễm chéo (PAL).
- **Phụ gia theo danh mục và ADI (`ins_listing`, `ins_adi`):** mỗi câu hỏi **đúng một mã INS**, chọn các mã in trên nhãn nằm trong danh mục TT 24 và đúng khoảng INS 100–1599; đáp án do các hàm dựng sẵn sinh từ nguồn đã kiểm (`permitted_answer`, `codex192_sentence`, `jecfa_status_answer`), không viết tay và không dùng nguồn mới.
- **Cờ theo cơ quan có thẩm quyền:** EU Phụ lục V (phẩm màu), IARC Vol. 134 (aspartame), FDA Red No. 3, Codex mục 4.2.1.7 (sulphite), 21 CFR 172.804 (PKU). Mỗi câu chỉ nói điều văn bản nói và ghi giới hạn ngay trong đáp án (ví dụ Phụ lục V là quy định ghi nhãn của EU, không suy ra yêu cầu ở Việt Nam).
- **Nhóm dễ tổn thương (EFSA, WHO):** photphat với người bệnh thận mạn (EFSA 2019), phẩm màu McCann và E211 với trẻ em (EFSA 2008), nitrit và TiO2 với thai phụ (EFSA 2017, 2021). Nhãn ở nhóm này nghĩa là "nhãn có/không có chất mà văn bản nêu ra", không phải khuyến cáo lâm sàng.
- **Chất ngọt không đường và caffeine (WHO 2023, EFSA 2015):** mỗi chủ đề có **câu cờ** (nhãn có chất đó) và **câu đối chứng** (nhãn không có chất đó). Cả hai dùng chung một câu hỏi trung tính nên mẫu câu hỏi không tiết lộ nhãn; câu đối chứng chọn ngẫu nhiên với seed cố định, số lượng bằng số câu cờ, từ các nhãn không có bất kỳ dấu hiệu nào của chất đó.
- **Ngưỡng dinh dưỡng (Codex CAC/GL 23-1997):** natri với ngưỡng "natri thấp" (≤ 0,12 g/100 g) và đường với ngưỡng "không đường" (≤ 0,5 g/100 g). Chỉ dùng nhãn có xác định được cơ sở của bảng dinh dưỡng (`sodium_basis`: 100 g, 100 ml hay khẩu phần) rồi quy đổi về 100 g hoặc 100 ml; nhãn chỉ ghi khoảng giá trị hoặc hai sản phẩm trong một hộp bị loại.

### 1.2. Giai đoạn 5: chọn 750 câu từ bản nguồn

`build_foodhealthqa_subset.py` (mặc định `--size 750 --seed 42 --max-per-image 4 --max-per-template 30`) làm theo các bước, đều xác định được và lặp lại được:

1. **Đầu vào:** chỉ bản ghi `eligible_for_scoring = yes` (2.311/2.311).
2. **Gán trụ cột chuyên môn** bằng quy tắc xác định, kiểm theo thứ tự ưu tiên: Dinh dưỡng, Thai phụ, Trẻ em, Bệnh nền, Dị ứng & Dị nguyên, An toàn phụ gia. Bản ghi vào trụ cột đầu tiên mà nó thỏa.
3. **Chia hạn ngạch theo trụ cột** theo tỷ lệ mục tiêu 22% / 22% / 16% / 15% / 14% / 11% (An toàn phụ gia, Dị ứng, Trẻ em, Bệnh nền, Thai phụ, Dinh dưỡng); phần dư làm tròn dồn vào An toàn phụ gia.
4. **Phân tầng theo nhãn an toàn** trong từng trụ cột: hạn ngạch mỗi nhãn tỷ lệ với phân bố của nhóm trong bản nguồn, tối thiểu 1 câu mỗi nhãn.
5. **Lấy mẫu ngẫu nhiên có seed (42).** Vòng 1 áp hai trần: tối đa 4 câu mỗi ảnh (`split_group`) và 30 câu mỗi mẫu câu hỏi (`template_id`). Vòng 2 chỉ chạy khi chưa đủ hạn ngạch: nới trần theo mẫu câu hỏi và ưu tiên bản ghi có ảnh và mẫu ít bị dùng nhất.
6. **Xuất** JSONL, CSV (4 trường bắt buộc đứng đầu) và `foodhealthqa_eval_set_summary.json`, sắp theo `id`.

Kết quả theo trụ cột (số bản ghi trong bản nguồn và số được chọn):

| Trụ cột | Bản nguồn | Được chọn | Tỷ lệ chọn |
|---|---:|---:|---:|
| An toàn phụ gia | 846 | 166 | 19,6% |
| Dị ứng & Dị nguyên | 753 | 165 | 21,9% |
| Trẻ em | 254 | 120 | 47,2% |
| Bệnh nền | 198 | 112 | 56,6% |
| Thai phụ | 157 | 105 | 66,9% |
| Dinh dưỡng | 103 | 82 | 79,6% |
| **Tổng** | **2.311** | **750** | **32,5%** |

Bộ 750 chọn tỷ lệ cao ở các trụ cột nhỏ (Dinh dưỡng, Thai phụ, Bệnh nền, Trẻ em) và thấp ở trụ cột lớn, nên cân bằng hơn bản nguồn, nhưng **không còn đại diện cho phân bố của bản nguồn**. Bộ 750 phủ 376 trên 489 ảnh và 150 trên 278 mẫu câu hỏi của bản nguồn.

## 2. Kết quả kiểm chứng trên 750 bản ghi

Các số dưới đây đếm trực tiếp từ `data/foodhealthqa_eval_set.jsonl` và `data/build/qa_verification.csv`.

| Kiểm tra | Kết quả |
|---|---|
| Trạng thái duyệt (`review_status`) | `approved_rule_verified`: 750/750 |
| Bản ghi trượt cổng G1–G6 (`verify_failed_gates`) | **0/750**; cả 750 id đều có trong `qa_verification.csv` với `eligible = yes` và không có cổng nào trượt |
| Mức kiểm chứng nguồn (`source_verification_level`) | 741 bản `label_and_primary_source_text_verified` (nhãn và văn bản nguồn gốc đều được kiểm); 9 bản `label_only_no_regulatory_assertion` (chỉ đối chiếu nhãn, không khẳng định về quy định) |
| Khẳng định nguồn được dùng (`external_claims`) | **57 khẳng định khác nhau**, tất cả đều đạt trong `external_claim_verification.json` (tệp này có 159 khẳng định, 159 đạt) |
| Bản ghi không có khẳng định nguồn nào | 9 (chính là 9 bản chỉ khẳng định về nhãn) |
| Khoảng trống nguồn (`source_gap`) | rỗng ở 750/750 |
| Ảnh nhãn được đối chiếu thủ công với bản chép | 376/376 (mục 6) |

Lưu ý: các trường `answer_correct`, `citation_correct`, `safety_label_correct` (đều `yes` ở 750 bản) là **kết quả của bộ kiểm tra quy tắc**, không phải đánh giá của người.

## 3. Các cổng kiểm tra từng câu trả lời (G1–G6)

Mỗi bản ghi phải qua đủ 6 cổng (`scripts_research/verify_v5_answers.py`):

| Cổng | Kiểm tra |
|---|---|
| G1 Kết luận | Mở đầu đáp án khớp nhãn; không có từ ngữ phủ định nhãn dương và ngược lại |
| G2 Đúng câu hỏi | Nhắc đúng sản phẩm; nêu đúng dị nguyên được hỏi; mọi mã INS trong câu hỏi có trong đáp án; đúng chủ đề |
| G3 Đúng nhãn | Mọi mã INS trong đáp án có trên nhãn; nhãn dị nguyên khớp bộ tái suy luận độc lập; số liệu dinh dưỡng khớp bảng nhãn |
| G4 Trích dẫn | Thẻ nguồn thuộc danh mục được công nhận; nguồn phù hợp chủ đề; không có ngoặc vuông lạ bị parser đọc nhầm; câu trả lời natri không được dẫn USDA |
| G5 Định vị | Claim locator có trong `knowledge_base/sources/catalog/source_citation_catalog.json`; trích đoạn nhãn có nguyên văn trong trường được trỏ tới |
| G6 Đúng nguồn | Mọi thẻ nguồn phải có một khẳng định đã kiểm chứng nguyên văn tương ứng (ví dụ thẻ JECFA cần khẳng định `JECFA_ADI_*`, thẻ Codex GL 23 cần `CODEX_GL23_*`); mọi mã `PERMITTED_LISTED` khớp văn bản gốc TT 24/2019 |

**Độ nhạy của bộ kiểm tra** được thử bằng cách cố ý chèn lỗi (đổi nhãn, xóa tên sản phẩm, đổi mã INS, thẻ nguồn bịa, đáp án lạc đề, mã ngoài TT 24): theo `tests/test_benchmark_v5.py`, bộ kiểm tra bắt được từ 95% trở lên ở cả 6 loại lỗi. Đây là thử nghiệm trên bản nguồn; không riêng cho tập con 750.

## 4. Nhãn an toàn và phạm vi khẳng định


**Ánh xạ nhãn.** `gold_safety_label` dùng 5 nhãn của ứng dụng; nhãn mịn của v5 nằm ở `label_v5`. Với 750 câu:

| Nhãn mịn `label_v5` | Số câu | Nhãn ứng dụng `gold_safety_label` | Số câu |
|---|---:|---|---:|
| `PERMITTED_LISTED` | 88 | `SAFE` (= PERMITTED_LISTED + SAFE) | 285 |
| `SAFE` | 197 | | |
| `NOT_DECLARED` | 35 | `CAUTION` (= NOT_DECLARED + CAUTION) | 277 |
| `CAUTION` | 242 | | |
| `UNSAFE` | 85 | `UNSAFE` | 85 |
| `ABSTAIN` | 72 | `ABSTAIN` | 72 |
| `CRITICAL_ALERT` | 31 | `CRITICAL_ALERT` | 31 |

Các câu `PERMITTED_LISTED` hỏi đúng thứ nguồn trả lời được (phụ gia có nằm trong danh mục TT 24/2019 hay không), không hỏi "có an toàn không". Các câu `NOT_DECLARED` là ca "nhãn không có câu cảnh báo nhiễm chéo (PAL)": theo Codex CXS 1-1985 và FAO/WHO JEMRA Part 3, không có PAL không bảo đảm sản phẩm không có dị nguyên, nên nhãn là `CAUTION`, không phải `SAFE`.

**Phạm vi khẳng định (`label_scope`).** Mỗi câu chỉ khẳng định đúng điều nguồn nói; phạm vi này được ghi ngay trong bản ghi:

| `label_scope` | Số câu | Ý nghĩa |
|---|---:|---|
| `full_record` | 167 | Kết luận đầy đủ theo nhãn và nguồn |
| `authority_flag:who_nss_pregnancy` | 91 | Cờ theo WHO 2023 về chất ngọt không đường ở thai phụ (gồm câu đối chứng không có chất đó) |
| `listed_in_positive_list_only` | 88 | Chỉ khẳng định phụ gia có tên trong Phụ lục 1 TT 24/2019 |
| `authority_flag:efsa_2019_phosphate_group` | 81 | Cờ theo EFSA 2019 về photphat (người bệnh thận mạn) |
| `authority_flag:efsa_2008_mccann_substances` | 50 | Cờ theo EFSA 2008 về nhóm phẩm màu McCann và E211 (trẻ em) |
| `authority_flag:who_nss_child` | 50 | Cờ theo WHO 2023 về chất ngọt không đường ở trẻ em |
| `single_factor:natri_vs_codex_low_threshold_per_100g` | 42 | Chỉ so natri với ngưỡng "natri thấp" của Codex CAC/GL 23 trên 100 g |
| `sugar_free_codex_gl23_per_100g` | 40 | Chỉ so đường với ngưỡng "không đường" của Codex CAC/GL 23 trên 100 g |
| `declared_ingredients_only` | 35 | Chỉ dựa trên thành phần nhãn đã khai |
| `authority_flag:cfr_172_804_aspartame` | 31 | Cờ theo 21 CFR 172.804 (người phenylketon niệu và aspartame) |
| `adi_status_per_jecfa_only` | 22 | Chỉ nêu trạng thái ADI theo trang JECFA |
| `authority_flag:efsa_caffeine_child` | 20 | Cờ theo EFSA 2015 về caffeine ở trẻ em |
| `authority_flag:eu_annex_v_colours` | 10 | Cờ theo Phụ lục V Quy định EU 1333/2008 |
| `authority_flag:efsa_2021_e171` | 8 | Cờ theo EFSA 2021 về E171 |
| `authority_flag:efsa_2017_nitrite` | 6 | Cờ theo EFSA 2017 về nitrit |
| `authority_flag:iarc_v134_aspartame` | 4 | Cờ theo IARC Vol. 134 (aspartame nhóm 2B) |
| `authority_flag:codex_sulphite_declaration` | 3 | Cờ theo Codex CXS 1 mục 4.2.1.7 (sulphite) |
| `authority_flag:fda_red3_revoked` | 2 | Cờ theo lệnh bãi bỏ FD&C Red No. 3 của FDA |

Nhãn ở nhóm `authority_flag` nghĩa là *"nhãn có/không có chất mà văn bản của cơ quan này nêu ra"*, **không phải lời khuyên y khoa**. Toàn bộ 419 câu miền `vulnerable_groups` thuộc dạng này.

## 5. Đối chiếu văn bản nguồn gốc

Tài liệu gốc nằm ở `knowledge_base/sources/raw/` (băm sha256 trong `catalog/external_sources_manifest.json`); mỗi khẳng định được đối chiếu nguyên văn bằng `scripts_research/verify_external_claims.py`. Bảng liệt kê các khẳng định nguồn mà 750 câu dựa vào (số câu: một câu có thể dùng nhiều khẳng định).

**Dị nguyên**

| Khẳng định (`external_claims`) | Số câu | Nguồn đã đọc |
|---|---:|---|
| `FALCPA_321qq_NINE_ALLERGENS` | 165 | 21 U.S.C. 321(qq): 9 "major food allergen"; không có động vật thân mềm |
| `JEMRA1_PRIORITY_LIST` | 148 | FAO/WHO JEMRA 2021 Part 1: danh sách ưu tiên toàn cầu (ngũ cốc chứa gluten, giáp xác, trứng, cá, sữa, đậu phộng, mè, 6 loại hạt cây) |
| `CXS1_4214_MANDATORY_ALLERGEN_LIST` | 148 | Codex CXS 1-1985 mục 4.2.1.4: danh sách phải khai báo; không có đậu nành, không có nhuyễn thể |
| `JEMRA1_SOY_AND_OATS_NOT_GLOBAL` | 54 | JEMRA 2021 Part 1: đậu nành và yến mạch không thuộc danh sách toàn cầu |
| `CXS1_4215_OPTIONAL_LIST_SOY_OATS` | 54 | Codex CXS 1-1985 mục 4.2.1.5: đậu nành, yến mạch "có thể phải khai báo" tùy quốc gia |
| `JEMRA3_PAL_REPORT` | 82 | FAO/WHO JEMRA Part 3 (2023): hệ thống PAL hiện chưa thống nhất và chưa đủ thông tin (tr. 31, 104) |
| `CXS1_PAL_ANNEX_DEFINITION` | 82 | Codex CXS 1-1985, Phụ lục (2026): định nghĩa PAL |
| `CFR_172_804_PKU_STATEMENT` | 31 | 21 CFR 172.804: câu "PHENYLKETONURICS: CONTAINS PHENYLALANINE" |
| `CXS1_4217_SULPHITE_DECLARATION` | 3 | Codex CXS 1-1985 mục 4.2.1.7: sulphite từ 10 mg/kg phải khai báo bằng tên |
| `FDA_QA5_REFINED_OIL_EXEMPT` | 1 | FDA Q&A về dị nguyên, Edition 5 (2025), mục D.9: dầu tinh luyện cao được miễn khai báo |

**Phụ gia**

| Khẳng định | Số câu | Nguồn đã đọc |
|---|---:|---|
| `TT24_PHULUC1_LISTS_INS_AND_NAME` | 106 | TT 24/2019/TT-BYT, Phụ lục 1: danh mục phụ gia theo mã INS và tên |
| `TT24_ML_IN_MG_PER_KG` | 10 | TT 24/2019/TT-BYT, Phụ lục 2: mức tối đa (ML) tính theo mg/kg |
| `CODEX192_TABLE1_TABLE3_PARSED` | 88 | Codex CXS 192-1995 (bản 2026): mã INS ở Table One (có điều kiện) và Table Three (GMP) |
| `CODEX_CXG36_INS_AND_CLASS_NAMES` | 17 | Codex CXG 36-1989: hệ thống số INS và nhóm chức năng |
| `JECFA_ADI_*` (30 khẳng định, mỗi mã INS một khẳng định) | 22 bản ghi | Trang cơ sở dữ liệu JECFA của WHO; chỉ nhận khi **số INS in trên trang khớp đúng** mã đang kiểm |
| `IARC_V134_ASPARTAME_2B` | 4 | IARC Monographs Vol. 134 (2023): aspartame nhóm 2B |
| `EU_ANNEXV_COLOURS_WARNING` | 10 | Quy định EU 1333/2008, Phụ lục V: cảnh báo cho 6 phẩm màu |
| `FDA_RED3_REVOKED_FOOD` | 2 | FDA, FR Doc. 2025-00830: bãi bỏ FD&C Red No. 3 trong thực phẩm |

**Nhóm dễ tổn thương và ngưỡng dinh dưỡng**

| Khẳng định | Số câu | Nguồn đã đọc |
|---|---:|---|
| `WHO_NSS_2023_PREGNANCY` | 91 | WHO (2023) *Use of non-sugar sweeteners*: khuyến cáo có điều kiện; bằng chứng quan sát về sinh non có độ chắc chắn thấp |
| `WHO_NSS_2023_CHILDREN` | 50 | Cùng tài liệu, phần trẻ em; không phải ngưỡng an toàn |
| `EFSA_OPINION_2019_PHOSPHATES` | 81 | EFSA 2019;17(6):5674: ADI nhóm 40 mg/kg tt/ngày không áp dụng cho người giảm chức năng thận |
| `EFSA_OPINION_2008_MCCANN` | 50 | EFSA 2008;6(3):660: thử nghiệm McCann, bằng chứng hạn chế, không quy được cho từng chất |
| `EFSA_CAFFEINE_2015_CHILDREN` | 20 | EFSA 2015;13(5):4102: chưa đủ thông tin để xác định mức caffeine an toàn cho trẻ em |
| `EFSA_OPINION_2017_NITRITE` | 6 | EFSA 2017;15(6):4786: ADI nitrit 0,07 mg/kg tt/ngày |
| `EFSA_2021_TIO2_FULLTEXT`, `EFSA_2021_TIO2_NOT_SAFE` | 8 + 1 | EFSA 2021;19(5):6585: không loại trừ độc tính gen; E171 không còn được coi là an toàn |
| `CODEX_GL23_SODIUM_PER_100G` | 42 | Codex CAC/GL 23-1997: natri thấp ≤ 0,12 g/100 g |
| `CODEX_GL23_SUGARS_FREE_PER_100G` | 40 | Codex CAC/GL 23-1997: không đường ≤ 0,5 g/100 g (hoặc 100 ml) |

Ba điểm về cách đọc nguồn, rút từ quá trình kiểm chứng:
- **JECFA:** trang JECFA chỉ được nhận khi số INS in trên trang khớp đúng mã đang kiểm (không đoán theo tên). Một câu hỏi chứa nhiều mã INS chỉ được giữ khi **mọi** mã đều có trang JECFA đã kiểm; thiếu một mã là loại cả câu.
- **Natri và đường** dùng ngưỡng Codex CAC/GL 23 tính trên 100 g, không dùng ngưỡng FDA theo khẩu phần tham chiếu (RACC) vì bảng dinh dưỡng trên nhãn ghi theo 100 g. Các nhãn chỉ ghi trên khẩu phần được quy đổi về 100 g hoặc 100 ml; đồ uống coi 100 ml tương đương 100 g.
- **WHO NSS và EFSA caffeine:** đáp án chỉ nói điều văn bản nói (bằng chứng hạn chế, không chứng minh nhân quả) và **không nêu bất kỳ giới hạn an toàn nào cho trẻ em hoặc thai phụ**. Polyol (sorbitol, maltitol, erythritol, xylitol) không bị gắn cờ vì WHO nói rõ chúng không phải chất ngọt không đường thuộc phạm vi khuyến cáo.

## 6. Kiểm tra nhãn với ảnh gốc

Toàn bộ ảnh nhãn đã được đối chiếu thủ công với bản chép trong `assets/labels/*.json`. **Cả 376 ảnh của bộ 750 đều nằm trong đợt kiểm này.** Kết quả ghi ở `data/label_audit_sample.csv`, `label_audit_high_risk.csv`, `label_audit_remaining.csv`:

| Nhóm kiểm | Ảnh thuộc bộ 750 |
|---|---:|
| Danh sách rủi ro cao (nhiều bản ghi phụ thuộc) | 245 |
| Mẫu ngẫu nhiên | 78 |
| Phần còn lại | 53 |
| **Tổng** | **376** |

Trong 376 ảnh, **103 ảnh (27,4%)** bị đánh dấu có ít nhất một sai lệch giữa ảnh và bản chép ở thời điểm kiểm: 49 ảnh sai mã phụ gia, 40 sai danh sách thành phần, 30 sai câu cảnh báo dị nguyên, 4 sai tên sản phẩm. Các tệp nhãn có sai lệch đã được sửa và câu hỏi được sinh lại từ nhãn đã sửa trước khi dựng bản nguồn hiện hành (theo ghi nhận của lần kiểm này, hơn 100 tệp nhãn được sửa). Cột đánh dấu là cờ cho mọi mức sai lệch, nên số 103 không phải số lỗi nặng; số lỗi nặng (sai/thiếu câu dị nguyên, sai mã INS, sai sản phẩm) chỉ là một phần của số này.

**Hạn chế của lần kiểm:** do một người thực hiện, không phải chuyên gia độc lập; chỉ xem một mặt bao bì của mỗi sản phẩm nên câu cảnh báo ở mặt khác không phát hiện được; một số ảnh bị cắt dở hoặc mờ.

**Nhận diện phụ gia chỉ ghi bằng tên.** Mô-đun `modules/additive_names.py` nhận phosphat, phẩm màu McCann, nitrit, TiO2, aspartame khi nhãn ghi tên thay vì mã INS, loại trừ mẫu phủ định ("không chứa natri benzoat") và các chất không thuộc nhóm EFSA. Tên đối chiếu từ Phụ lục 1 TT 24/2019 (`knowledge_base/sources/derived/tt24_additive_names.json`).

## 7. Lỗi đã phát hiện và sửa nhờ kiểm chứng

Các lỗi dưới đây có ở phiên bản trước (v4) và đã được sửa ở v5; bộ 750 thừa hưởng bản đã sửa. Chúng cho thấy cổng kiểm tra bắt được lỗi thật:
- **Sai nhãn và sai đáp án:** 11 bản dị nguyên gán nhãn sai; 3 bản ABSTAIN khẳng định "nhãn không kê khai" trong khi nhãn có khai; 22 đáp án thai phụ không nhắc tên sản phẩm; 5 câu hỏi về tăng động bị trả lời về nitrit; 21 đáp án PKU trích sai thẻ nguồn; 22 locator không có trong catalog.
- **Nhãn quá mạnh:** 374 nhãn `SAFE` về phụ gia đổi thành `PERMITTED_LISTED` (chỉ khẳng định tư cách có trong danh mục); 157 ca "không có PAL" đổi `CAUTION` thành `NOT_DECLARED`.
- **Mã INS giả hoặc cũ** (ví dụ `INS025` sinh từ "0,025 %", `INS3220` thực ra là 322(i)): 8 bản bị xóa.
- **Trích nguồn sai:** trích JEMRA "ưu tiên" cho đậu nành và nhuyễn thể; trích FALCPA cho nhuyễn thể; locator `Table3_GMP` áp cho mọi phụ gia trong khi 183/374 bản "được phép" có ít nhất một mã chỉ nằm ở Table One (ví dụ INS 211, 202, 951); URL JEMRA Part 3 trả 404; ID hóa chất JECFA trỏ sai.
- **Ngưỡng không có nguồn:** ngưỡng "480 mg ≥ 20% DV" và "15 g đường/khẩu phần" không có trong văn bản nguồn; WHO 2015 khuyến nghị về đường tự do, không phải đường tổng số trên nhãn; 2 bản benzen về bánh không được nguồn bao trùm (nguồn chỉ nói về đồ uống).
- **Lỗi nhận diện dị nguyên:** "Sữa chiết từ đậu nành" từng bị coi là sữa bò; nhiều nhãn "Có chứa" khai thiếu dị nguyên (nhiều nhất là mè), nên hệ thống nhận diện dị nguyên cả từ danh sách thành phần chứ không chỉ từ phần "Có chứa".

## 8. Bản ghi bị loại ở bản nguồn

Nguyên tắc "không nguồn thì không đưa vào" đã loại 303 bản ghi của v4/v5 cũ, nên bộ 750 chỉ chứa những gì có nguồn kiểm chứng:

| Lý do | Số bản ghi |
|---|---:|
| Nhãn 0429 bị xóa (theo yêu cầu) | 3 |
| Nhãn 0585 bị xóa (trùng 0467) | 3 |
| Natri chỉ ghi dạng khoảng hoặc hai sản phẩm trong một hộp | 6 |
| Câu hỏi chứa mã INS giả/cũ, đáp án không còn khớp | 8 |
| Chất ngoài KB 50 chất, còn thiếu trang JECFA cho ít nhất một mã | 171 |
| Ngưỡng đường (không có nguồn ngưỡng) | 62 |
| Nguồn chưa lấy được (JECFA/Codex về thai kỳ, INS 471/412…) | 31 |
| Câu hỏi lâm sàng về thai kỳ (không có nguồn trực tiếp) | 17 |
| Benzen ngoài đồ uống (nguồn chỉ bao trùm đồ uống) | 2 |
| **Tổng** | **303** |

Danh sách ID chi tiết của 303 bản ghi này đã không còn lưu, chỉ còn số lượng theo lý do.

## 9. Giới hạn cần nêu khi công bố

Các giới hạn về cấu trúc bộ (nhãn phụ thuộc trụ cột, mẫu câu hỏi `T256` chiếm 11,3%, tầng silver 54,8%, một persona cho mỗi câu) xem [MO_TA_BO_DANH_GIA_750_CAU.md](MO_TA_BO_DANH_GIA_750_CAU.md), mục 7. Các giới hạn riêng về kiểm chứng:

**Bộ này không đo được:**
- Phụ gia không-SAFE ngoài các chủ đề đã có nguồn (màu azo/EU, Red No. 3, aspartame, sulphite, TiO2, photphat, nitrit, nhóm McCann) vẫn thưa. Thai kỳ và trẻ em chỉ được đo qua cờ cảnh báo của vài nguồn, không qua đánh giá phụ gia riêng cho từng nhóm.
- Đường chỉ được đo theo ngưỡng "không đường" của Codex trên 100 g; không đo ngưỡng WHO 10%/5% năng lượng vì ngưỡng đó tính theo cả ngày, không so được với một sản phẩm.
- Mức caffeine an toàn cho trẻ em: EFSA 2015 kết luận không đủ thông tin, nên bộ không có và không tạo ra giới hạn nào cho trẻ.
- Tăng huyết áp chỉ đo qua natri trên 100 g, không đo huyết áp hay tổng lượng natri trong ngày.
- Benzen trong thực phẩm không phải đồ uống; các mã phụ gia chưa có ADI trên JECFA (nhiều mã nhóm phụ như phosphate 339(ii), 450(ii), 1100(i), 960d, 640, 965).

**Về phương pháp:**
- **Không có chuyên gia.** Điểm đo trên bộ này là số liệu theo nhãn kiểm chứng bằng quy tắc (exploratory), không phải hiệu năng gold standard.
- **Vòng tròn quy tắc ↔ guardrail.** Đáp án chuẩn được dẫn xuất bằng quy tắc, và guardrail của hệ thống dùng quy tắc/kho tri thức tương tự, nên điểm của cấu hình B/C có thể cao một phần do trùng logic; không diễn giải điểm B/C là độ an toàn thực tế.
- **Bộ tái suy luận dị nguyên và cờ nhóm dễ tổn thương** do cùng một người xây dựng nên không độc lập tuyệt đối.
- **Kiểm tra nhãn với ảnh** do một người thực hiện, một mặt bao bì mỗi sản phẩm (mục 6).
- **Ô nhỏ:** các ô nhãn có ít câu (xem mục 7 của MO_TA_BO_DANH_GIA_750_CAU.md) cần báo cáo kèm khoảng tin cậy; `benchmark_report.py` báo số lượng và khoảng tin cậy Wilson cho các ô này, không báo F1.
- **Hướng dẫn FDA Q&A** là tài liệu không bắt buộc của Mỹ, chỉ dùng để giải thích cách hiểu "dẫn xuất từ dị nguyên" (1 câu trong bộ 750). Nhãn trong dữ liệu là hàng bán tại Việt Nam, nhưng chưa đối chiếu văn bản pháp quy Việt Nam về dị nguyên.
- **Mã sinh dữ liệu từng bị mất** khỏi thư mục làm việc và đã được dựng lại từ nhật ký phiên làm việc; chạy lại chuỗi cho ra dữ liệu giống hệt dữ liệu trước đó. Nên giữ mọi thay đổi trong git để tránh lặp lại.

**Về thực nghiệm (chưa có kết quả):**
- Chưa chạy 3 cấu hình A (LLM không truy xuất), B (RAG trên văn bản chuẩn), C (ảnh → OCR → RAG) trên 750 câu, chưa đo CER/WER của OCR và chưa chạy ablation toàn diện. Vì vậy chưa có số liệu EM, F1, citation accuracy, groundedness, faithfulness hay độ an toàn của hệ thống. Mọi con số trong báo cáo này chỉ mô tả bộ dữ liệu và nguồn, không mô tả hiệu năng hệ thống.
- Đã đo chất lượng truy xuất đoạn bằng chứng trên 750 câu (không gọi LLM); kết quả ở `docs/QUY_TRINH_3_CAU_HINH_HE_THONG.md`, mục 4.6.
- Khi báo kết quả cần đi kèm: macro-F1 và baseline đa số theo trụ cột, báo cáo theo `template_id`, khoảng tin cậy lấy mẫu lại theo ảnh (`split_group`), và ghi rõ các ô nhỏ ở trên. Bộ chỉ là silver, không gọi là gold.

## 10. Truy vết và dựng lại

| Thành phần | Vị trí |
|---|---|
| Bộ đánh giá 750 câu | `data/foodhealthqa_eval_set.{jsonl,csv,_summary.json}`, dựng bằng `scripts_research/build_foodhealthqa_subset.py` (seed 42) |
| Bản nguồn v5 (2.311 bản ghi) | `data/qa_benchmark.{jsonl,csv}`, chỉ để dựng bộ 750 |
| Kiểm tra từng bản ghi | `data/build/qa_verification.csv`; kèm `qa_fix_list.csv`, `qa_activation_report.json`, `qa_build_full.jsonl`, `qa_input_source_verified.*` |
| Khẳng định nguồn đã kiểm | `knowledge_base/sources/derived/external_claim_verification.json`, `jecfa_adi_verification*.json`, `codex_192_ins_codes.json`, `tt24_*.json` |
| Danh mục nguồn (tên tài liệu, mục, URL) | `knowledge_base/sources/catalog/source_citation_catalog.json` |
| Tài liệu gốc và băm sha256 | `knowledge_base/sources/raw/`, `catalog/external_sources_manifest.json` |
| Kiểm tra nhãn với ảnh | `data/label_audit_*.csv` |

Dựng lại bản nguồn: `verify_external_claims.py` → `add_authority_records_v5.py` → `create_benchmark_v5.py` → `verify_v5_answers.py` → `finalize_benchmark_v5.py`; sau đó dựng lại bộ 750 bằng `build_foodhealthqa_subset.py`. Khi chia train/validation/test hoặc lấy mẫu lại **bắt buộc theo `split_group`** (một ảnh là một nhóm), không chia ngẫu nhiên theo dòng.
