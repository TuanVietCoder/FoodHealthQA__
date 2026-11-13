# Báo cáo Kiểm chứng Nguồn và Trạng thái Benchmark FoodHealthQA

| Thuộc tính | Giá trị |
|---|---|
| Phiên bản active | **v5** (phân tầng, kiểm chứng câu trả lời bằng quy tắc, chỉ giữ bản ghi có nguồn kiểm chứng) |
| Cập nhật lần cuối | 2026-11-09 |
| Trạng thái | **Silver kiểm chứng bằng quy tắc, không có chuyên gia** (`expert_validated = false`, không đủ điều kiện gold) |
| Số bản ghi | **2.311**, trong đó **2.311 tính điểm** (không còn bản ghi "giữ để kiểm toán") |
| Số nhãn bao bì | 493 tệp `assets/labels/*.json` ứng với 493 ảnh `assets/images/*.jpeg`; bộ benchmark dùng 489 ảnh |
| Tệp chuẩn duy nhất | [qa_benchmark.csv](../data/qa_benchmark.csv) / [qa_benchmark.jsonl](../data/qa_benchmark.jsonl) |
| Tệp kiểm toán (trong `data/build/`) | [bản dựng đầy đủ](../data/build/qa_build_full.jsonl), [fix list](../data/build/qa_fix_list.csv), [kiểm tra từng bản ghi](../data/build/qa_verification.csv), [activation report](../data/build/qa_activation_report.json) |
| Mục đích sử dụng | Phát triển hệ thống và đánh giá độ trung thực/an toàn; **chưa thay thế tư vấn y khoa lâm sàng** |
| Kiểm thử | 244 test tự động đều qua (cập nhật 2026-11-12); 0/2.311 bản ghi trượt các cổng G1–G6 |

> **Nguyên tắc từ phiên này:** một bản ghi không có nguồn tra cứu kiểm chứng được thì **không** đưa vào bộ QA, vì bộ QA là chuẩn để chấm hệ thống. Vì vậy 303 bản ghi của v4/v5 cũ đã bị loại (mục 6).

Mục 1–4 mô tả v4 (lịch sử, không còn active). Mục 5 trở đi mô tả v5 hiện hành.

**Một bộ chuẩn duy nhất.** Chỉ có `data/qa_benchmark.csv` / `.jsonl` được dùng để đánh giá. Cấu trúc thư mục dữ liệu:

| Vị trí | Nội dung | Dùng để đánh giá? |
|---|---|---|
| `data/qa_benchmark.{csv,jsonl}` | Bộ chuẩn (v5, 2.311 bản ghi) | **Có** |
| `data/benchmark_manifest.json` | Chỉ ghi bộ chuẩn hiện hành | — |
| `data/build/` | Tệp trung gian của quy trình dựng: đầu vào đã kiểm nguồn, bản dựng đầy đủ, danh sách sửa, kiểm tra từng bản ghi, báo cáo kích hoạt | Không |

Chạy lại quy trình: `create_benchmark_v5.py` → `verify_v5_answers.py` → `finalize_benchmark_v5.py` (từ 5.10: chạy thêm `add_authority_records_v5.py` trước `create_benchmark_v5.py`; đọc `data/build/qa_input_source_verified.*`, ghi `data/build/*` và `data/qa_benchmark.*`).

> **Cập nhật 2026-11-11 (persona dị ứng).** Hàm dựng persona trước đó suy ra danh sách dị ứng bằng cách quét từ khóa trong câu hỏi vì bản ghi không có trường `persona_conditions`; kết quả là 143/753 bản ghi dị ứng có danh sách lệch tên persona (141 thừa, 3 thiếu, 1 rơi về danh sách chung). Bản benchmark hiện hành đã thêm hai cột `persona_conditions` và `persona_conditions_vi` cho 753 bản ghi dị ứng (tạo bằng `scripts_research/add_persona_conditions.py`, các trường gốc không đổi); bản cũ lưu ở `data/_backup_pre_persona_conditions/`. Chỉ số nhóm dị ứng tính trước thay đổi này cần chạy lại.

---

# Phần A — v4 (lịch sử)

## 1. Tóm tắt kiểm chứng nguồn của v4

Phiên bản v4 (2.081 cặp QA) được đối soát với nhãn bao bì đã gắn nhãn thủ công và các quy chuẩn `VN_TT24_2019`, `CODEX_STAN_192_1995`, `US_FDA_FALCPA`, `FAO_WHO_JEMRA`. Các chỉ số "100%" của v4 (độ phủ nhãn, độ phủ locator, `approved_source_verified`) chỉ nói rằng mỗi bản ghi có nhãn và locator đi kèm; **không** chứng minh từng câu trả lời đúng nguồn. Việc đó được làm ở v5 (mục 5.1) và đã phát hiện nhiều lỗi của v4 (mục 5.2, 5.4, 5.6, 5.6c).

## 2. Phân bố v4 (để tham chiếu)

| Domain | Số bản ghi v4 |
|---|---:|
| `allergen_detection` | 632 |
| `insufficient_data` | 568 |
| `additive_regulation` | 453 |
| `vulnerable_groups` | 428 |

## 3. Quy tắc chuẩn hóa của v4 vẫn còn hiệu lực
1. "Không có PAL" không phải bảo đảm phi dị nguyên (Codex CXS 1-1985 4.2.1.4, FAO/WHO JEMRA Part 3): v4 chuyển `SAFE` → `CAUTION`; v5 mịn hơn thành `NOT_DECLARED` (xem 5.3).
2. Người PKU và aspartame (INS 951): `CRITICAL_ALERT` theo 21 CFR 172.804.
3. Phụ gia có trong danh mục cho phép nhưng nhãn không ghi hàm lượng: chỉ khẳng định tư cách có trong danh mục, không kết luận tuân thủ ML.
4. Mỗi bản ghi có hai lớp locator: lớp nhãn (`assets/labels/{id}.json#/…`) và lớp văn bản pháp quy (`modules/claim_locator_resolver.py`).

## 4. Hướng dẫn sử dụng tập benchmark
* Dùng [qa_benchmark.csv](../data/qa_benchmark.csv) (hoặc `.jsonl`) làm benchmark đánh giá RAG, LLM và bộ máy quy tắc. Các cột `v4_*` đã được bỏ khỏi file; bộ chỉ còn nội dung v5.
* **Chia tập Train/Validation/Test bắt buộc theo `image_ref`**, không chia ngẫu nhiên theo dòng.
* Định danh khi công bố: **Silver — kiểm chứng bằng quy tắc, không có chuyên gia**; không gọi là gold.

---

# Phần B — v5 (ACTIVE)

## 5. Cấu trúc và kết quả

| Thuộc tính | Giá trị |
|---|---|
| Scripts | `create_benchmark_v5.py` → `verify_v5_answers.py` → `finalize_benchmark_v5.py`; nguồn dùng `verify_external_claims.py` (khẳng định kiểm chứng nguyên văn) và `source_grounding_v5.py` |
| Mức xác minh | **1.744** bản có khẳng định pháp lý đã đối chiếu nguyên văn; **34** bản chỉ khẳng định về nhãn |
| Khẳng định nguồn đã kiểm | 151/151 đạt (`knowledge_base/sources/derived/external_claim_verification.json`); catalog nguồn có 204 mục |

### Phân bố sau kiểm chứng
| Domain | Số bản ghi |
|---|---:|
| `allergen_detection` | 629 |
| `additive_regulation` | 440 |
| `insufficient_data` | 359 |
| `vulnerable_groups` | 350 |

| Nhãn mịn `label_v5` | Số bản ghi | Nhãn app `gold_safety_label` | Số bản ghi |
|---|---:|---|---:|
| `UNSAFE` | 387 | `UNSAFE` | 387 |
| `PERMITTED_LISTED` | 372 | `SAFE` (gồm PERMITTED_LISTED + SAFE) | 613 |
| `SAFE` | 241 | | |
| `ABSTAIN` | 345 | `ABSTAIN` | 345 |
| `NOT_DECLARED` | 156 | `CAUTION` (gồm NOT_DECLARED + CAUTION) | 324 |
| `CAUTION` | 168 | | |
| `CRITICAL_ALERT` | 109 | `CRITICAL_ALERT` | 109 |

| Nhóm người dùng | Số bản ghi |
|---|---:|
| Dị ứng | 753 |
| Người trưởng thành khỏe mạnh | 675 |
| Bệnh thận mạn giai đoạn 3 | 141 |
| Trẻ 5 tuổi | 84 |
| Phenylketon niệu | 57 |
| Tăng huyết áp (chỉ natri) | 53 |
| Thai phụ (chỉ câu hỏi về cảnh báo trên nhãn) | 15 |

### 5.1 Các cổng kiểm tra từng câu trả lời (G1–G6)
| Cổng | Kiểm tra |
|---|---|
| G1 Kết luận | Mở đầu đáp án khớp nhãn; không có từ ngữ phủ định nhãn dương và ngược lại |
| G2 Đúng câu hỏi | Nhắc đúng sản phẩm; nêu đúng dị nguyên được hỏi; mọi mã INS trong câu hỏi có trong đáp án; đúng chủ đề |
| G3 Đúng nhãn | Mọi mã INS trong đáp án có trên nhãn; nhãn dị nguyên khớp bộ tái suy luận độc lập; số liệu dinh dưỡng khớp bảng nhãn |
| G4 Trích dẫn | Thẻ nguồn được công nhận; nguồn phù hợp chủ đề; không có ngoặc vuông lạ bị parser đọc nhầm; câu trả lời natri không được dẫn USDA |
| G5 Định vị | Claim locator có trong `source_citation_catalog.json`; trích đoạn nhãn có nguyên văn trong trường được trỏ tới |
| G6 Đúng nguồn | Mọi thẻ nguồn phải có một khẳng định đã kiểm chứng nguyên văn tương ứng (ví dụ thẻ JECFA cần claim `JECFA_ADI_*`, thẻ Codex GL 23 cần `CODEX_GL23_*`, thẻ FDA Q&A cần `FDA_QA5_*`); mọi mã `PERMITTED_LISTED` khớp văn bản gốc TT 24/2019 |

Độ nhạy của bộ kiểm tra được kiểm bằng cách cố ý chèn lỗi (đổi nhãn, xóa tên sản phẩm, đổi mã INS, thẻ nguồn bịa, đáp án lạc đề, mã ngoài TT24): đạt ≥95% ở cả 6 loại (`tests/test_benchmark_v5.py`).

### 5.2 Lỗi của v4 được phát hiện và sửa nhờ các cổng này
- 11 bản dị nguyên gán nhãn sai (ví dụ `FOODQA_G3_2058`); 3 bản ABSTAIN khẳng định "nhãn không kê khai" trong khi nhãn có khai (`FOODQA_G3_0257`, `1413`, `1705`).
- 22 đáp án thai phụ không nhắc tên sản phẩm; 5 câu hỏi về tăng động bị trả lời về nitrit; 21 đáp án PKU trích sai thẻ nguồn; 22 locator không có trong catalog.
- 374 nhãn `SAFE` về phụ gia đổi thành `PERMITTED_LISTED`; 157 ca "không có PAL" đổi `CAUTION` → `NOT_DECLARED`.
- Các bản có mã INS giả hoặc cũ (ví dụ `INS025` sinh từ "0,025 %", `INS3220` thật ra là 322(i), `INS422` thật ra là 1422): 8 bản bị xóa (mục 6).

### 5.3 Nhãn đánh giá cho consumer
`gold_safety_label` dùng 5 nhãn của app (`PERMITTED_LISTED → SAFE`, `NOT_DECLARED → CAUTION`); nhãn mịn nằm ở `label_v5`. Các câu hỏi `PERMITTED_LISTED` đã được viết lại để hỏi đúng thứ nguồn trả lời (phụ gia có nằm trong danh mục TT 24/2019 hay không), không hỏi "có an toàn không".

### 5.4 Đối chiếu văn bản nguồn gốc
Tài liệu gốc ở `knowledge_base/sources/raw/external/` (sha256 trong `external_sources_manifest.json`), đối chiếu bằng `verify_external_claims.py`: 151/151 khẳng định đạt.

| Khẳng định được phép dùng | Nguồn đã đọc |
|---|---|
| 9 "major food allergen"; **không có động vật thân mềm** | 21 U.S.C. 321(qq) |
| Danh sách ưu tiên toàn cầu: ngũ cốc chứa gluten, giáp xác, trứng, cá, sữa, đậu phộng, mè, 6 loại hạt cây; **đậu nành và yến mạch không thuộc danh sách** | FAO/WHO JEMRA 2021 Part 1 |
| Hệ thống PAL chưa thống nhất và chưa đủ thông tin | FAO/WHO JEMRA Part 3 (2023, toàn văn báo cáo 104 trang `JEMRA_PART3.pdf`, p.31, 104) |
| EFSA 2021: E171 không còn được coi là an toàn | Trang thông cáo EFSA |
| Câu "PHENYLKETONURICS: CONTAINS PHENYLALANINE" | 21 CFR 172.804(d)(2) |
| Phụ lục 1 liệt kê phụ gia theo mã INS và tên (400/400 dòng); Phụ lục 2 quy định ML theo mg/kg | PDF TT 24/2019 |
| **Mới:** natri thấp ≤ 0,12 g/100 g, rất thấp ≤ 0,04 g, không natri ≤ 0,005 g | Codex CAC/GL 23-1997 (bảng điều kiện công bố) |
| **Mới:** dầu tinh luyện hoàn toàn từ dị nguyên chính được miễn quy định khai báo dị nguyên; dầu chưa tinh luyện phải khai nguồn | FDA Q&A về dị nguyên, Edition 5 (1/2025), mục D.9 |

Lỗi nguồn của v4 mà việc đối chiếu làm lộ ra (đã sửa trong v5): trích JEMRA "ưu tiên" cho đậu nành và nhuyễn thể; trích FALCPA cho nhuyễn thể; URL JEMRA Part 3 trả 404; thẻ 21 CFR 172.804 mang nhầm `citation_id`; ID hóa chất JECFA trỏ sai; đáp án "SAFE theo Codex STAN 192-1995" không kiểm chứng được nên chỉ giữ khẳng định tư cách có trong danh mục TT24.

### 5.5 Đối chiếu JECFA (cơ sở dữ liệu WHO)
`fetch_jecfa.py` và `fetch_jecfa_more.py` chỉ nhận một bản ghi khi **số INS in trên trang JECFA khớp đúng** mã đang kiểm, không đoán theo tên. Trang gốc lưu ở `knowledge_base/sources/raw/external/jecfa/` kèm sha256.

**Đợt 1 (14 mã)**, ví dụ: INS 300 (not specified), 322(i) (not limited), 330 (not limited), 401, 407, 440, 500(ii) (not limited), 551, 627, 415. Không tìm được bản ghi cho INS 471.

**Đợt 2 (mở rộng, 186 mã INS xuất hiện trong các câu hỏi ABSTAIN):**
- Tìm được trang JECFA cho **130** mã, trong đó **117** mã có giá trị ADI. **69 mã chưa có ADI**: 56 mã không tìm thấy trang và 13 mã có trang nhưng không có ADI (nhiều mã nhóm phụ như phosphate 339(ii), 340(i), 450(ii), 452(i); 1100(i); 960d; 640; 965…).
- Hai lỗi đọc trang đã được sửa giữa chừng: (1) trang có thêm số COE/FEMA giữa trường INS và phần sau nên mã bị đọc sai; (2) mục đánh giá mới nhất nhiều trang chỉ ghi "chỉ xét tiêu chuẩn kỹ thuật" và không có ADI, ADI nằm ở mục trước.
- Với mỗi bản ghi mà **mọi** mã INS trong câu hỏi đều có trang JECFA, đáp án được viết lại: nếu tất cả mã có ADI "not specified/not limited" thì `SAFE`; ngược lại `ABSTAIN` có dẫn ADI của từng chất và nêu lý do thiếu hàm lượng trên nhãn để tính mức phơi nhiễm. **153 bản ghi** thuộc dạng này (69 SAFE, 84 ABSTAIN), `label_scope = adi_status_per_jecfa_only`.
- Chỉ một mã thiếu nguồn là đủ để bản ghi bị loại; đó là lý do 171 bản ghi nhóm này không còn trong bộ (mục 6).

### 5.6 Đối chiếu Codex (3 tệp PDF do người dùng tải tay, bản sửa đổi 2026) và Codex CAC/GL 23-1997
| Khẳng định | Kết quả đọc từ văn bản |
|---|---|
| CXS 192-1995 Table One / Table Three | 322 mã INS ở Table One (cho phép có điều kiện) và 192 mã ở Table Three (GMP); lưu `knowledge_base/sources/derived/codex_192_ins_codes.json` |
| CXS 1-1985 mục 4.2.1.4 | Ngũ cốc chứa gluten (lúa mì, lúa mạch đen, đại mạch), giáp xác, trứng, cá, đậu phộng, sữa, mè, 6 loại hạt cây. **Không có đậu nành, không có nhuyễn thể** |
| CXS 1-1985 mục 4.2.1.5 | Kiều mạch, cần tây, **yến mạch**, lupin, mù tạt, đậu nành, 3 loại hạt (Brazil, macadamia, hạt thông) và một số thực phẩm khác: có thể khai báo tùy quốc gia. Chú thích: yến mạch nhiều người celiac dung nạp được nhưng không phải tất cả vì nhiễm chéo với lúa mì, đại mạch, lúa mạch đen |
| CXS 1-1985 Phụ lục PAL (2026) | PAL là tuyên bố nguy cơ hiện diện ngoài ý muốn do tiếp xúc chéo, xác định bằng đánh giá rủi ro |
| CXG 36-1989 | Hệ thống số INS và tên nhóm chức năng |
| **Mới:** CAC/GL 23-1997 | Bảng điều kiện công bố natri tính trên 100 g (xem 5.7) |

Lỗi nguồn của v4: locator `Table3_GMP` áp cho mọi phụ gia, nhưng 183/374 bản "được phép" có ít nhất một mã chỉ nằm ở Table One (ví dụ INS 211, 202, 951, 102, 133).

### 5.6b Ngưỡng dinh dưỡng và benzen
| Khẳng định đã đọc | Hệ quả |
|---|---|
| 21 CFR 101.61: natri "free" <5 mg, "very low" ≤35 mg, "low" ≤140 mg, **mỗi khẩu phần tham chiếu (RACC)** | Không áp được lên bảng dinh dưỡng ghi trên 100 g |
| 21 CFR 101.54: "high" = từ 20% DV trở lên mỗi RACC | Số "480 mg ≥ 20% DV" của v4 không có trong văn bản |
| FDA Q&A: benzen có thể hình thành ở mức ppb trong **một số đồ uống** | 2 bản benzen của v4 là bánh, nguồn không bao trùm → **đã loại** |
| WHO 2015: khuyến nghị về **đường tự do**, không phải đường tổng số trên nhãn | Ngưỡng "15 g/khẩu phần" của v4 không có nguồn → **62 bản đường đã loại** |

### 5.6c Đối chiếu toàn văn 4 ý kiến EFSA và nhóm dễ tổn thương
| Ý kiến | Khẳng định được phép dùng |
|---|---|
| EFSA 2019;17(6):5674 (photphat) | ADI nhóm 40 mg/kg tt/ngày tính theo phospho bảo vệ người lớn khỏe mạnh nhưng **không áp dụng cho người giảm chức năng thận mức vừa đến nặng** |
| EFSA 2008;6(3):660 (McCann 2007) | Thử 6 phẩm màu và natri benzoat E211 ở trẻ 3 và 8–9 tuổi; "bằng chứng hạn chế", không quy được cho từng chất |
| EFSA 2017;15(6):4786 (nitrit) | ADI 0,07 mg ion nitrit/kg tt/ngày; báo cáo trích Hội đồng Bộ trưởng Bắc Âu xếp thai phụ vào nhóm nhạy cảm hơn |
| EFSA 2021;19(5):6585 (TiO2) | Không loại trừ được mối lo ngại độc tính gen; E171 không còn được coi là an toàn |

Nhãn CAUTION/SAFE ở nhóm này nghĩa là *"nhãn có/không có chất mà văn bản của cơ quan này nêu ra"*, không phải lời khuyên y khoa. Nhóm này gồm 297 bản (CKD, trẻ em, PKU, thai phụ với TiO2/nitrit). Lỗi v4 đã sửa: bộ nối mã INS bỏ sót mã trong ngoặc vuông (`FOODQA_G3_1998`); quy tắc trẻ em bỏ sót E211; các câu "KDOQI 2020", "sinh khả dụng 90–100%" không có trong văn bản; DOI EFSA 2008 trong catalog trỏ nhầm.

### 5.7 Natri cho người tăng huyết áp — dựng lại theo ngưỡng Codex trên 100 g
- 59 bản natri của bản cũ trích USDA FDC (không phải nguồn ngưỡng) và trích nhầm Codex CXS 1 mục 4.2.1.4 (mục về dị nguyên). Ngưỡng FDA tính theo RACC nên không dùng được cho bảng ghi trên 100 g.
- Tôi mở cả 59 ảnh để xác định số natri tính trên gì: 33 nhãn ghi trên 100 g, 7 nhãn trên 100 ml, 13 nhãn chỉ ghi trên khẩu phần/gói/ly (quy đổi sang mg/100 g); 6 nhãn chỉ ghi khoảng giá trị hoặc gồm hai sản phẩm trong một hộp (0178, 0238, 0446, 0479, 0522, 0570) nên bị xóa. Trường `sodium_basis` được thêm vào 53 nhãn.
- **53 bản ghi còn lại** hỏi "natri có đạt điều kiện natri thấp (≤120 mg/100 g) theo Codex CAC/GL 23-1997 không": 17 `SAFE`, 36 `CAUTION`. Đáp án nói rõ đây là so sánh với ngưỡng công bố dinh dưỡng, không phải kết luận y khoa. Đồ uống: coi 100 ml tương đương 100 g.

### 5.8 Kiểm tra nhãn với ảnh gốc
Toàn bộ ảnh nhãn đã được đối chiếu thủ công với bản chép trong `assets/labels/*.json` (người kiểm: một người, không phải chuyên gia độc lập; chỉ xem một mặt bao bì mỗi sản phẩm):

| Nhóm kiểm | Số ảnh | Lỗi nặng (sai/thiếu câu dị nguyên, sai mã INS, sai sản phẩm) |
|---|---:|---:|
| Mẫu ngẫu nhiên | 100 | 11 |
| Danh sách rủi ro cao (nhiều bản ghi phụ thuộc) | 321 | 14 (4,4%) |
| Phần còn lại | 74 | 1 |

Các kết quả ở `data/label_audit_sample.csv`, `label_audit_high_risk.csv`, `label_audit_remaining.csv`. Hậu quả: hơn 100 tệp nhãn được sửa; câu hỏi được sinh lại từ nhãn đã sửa; nhãn 0429 bị xóa theo yêu cầu, nhãn 0585 bị xóa vì trùng sản phẩm 0467 (khác thể tích; 0467 ghi `net_weight` 180 ml và câu cảnh báo đúng "sữa và gluten từ ngũ cốc (lúa mạch)").

**Nhận diện phụ gia chỉ ghi bằng tên.** Mô-đun `additive_names.py` nhận phosphat, phẩm màu McCann, nitrit, TiO2, aspartame khi nhãn ghi tên thay vì mã INS, loại trừ mẫu phủ định ("không chứa natri benzoat") và các chất không thuộc nhóm EFSA (iron pyrophosphate, distarch phosphate…). Tên đối chiếu từ Phụ lục 1 TT 24/2019 (`knowledge_base/sources/derived/tt24_additive_names.json`, 392 mã).

### 5.9 Áp dụng hướng dẫn FDA Q&A (Edition 5) và JEMRA Part 1 vào hệ thống
Đã làm:
- **Dầu tinh luyện (D.9):** dầu ghi rõ "tinh luyện" không còn tính là bằng chứng dị nguyên; dầu không ghi tinh luyện vẫn tính là có khai báo. Ảnh hưởng 1 bản ghi (`FOODQA_G3_1240`, đáp án thêm câu giải thích).
- **Nhãn "Có chứa" thiếu dị nguyên (D.5, D.6):** script `check_contains_consistency.py` → `data/label_contains_consistency.csv`, 41 cờ trên 30 nhãn. Sau khi mở ảnh: 22 trường hợp nhà sản xuất khai thiếu (nhiều nhất **mè: 11**), 19 trường hợp dùng từ chung ("hải sản", "các loại hạt"), 1 báo nhầm. Không có lỗi chép nhãn. Kết luận: phải nhận diện dị nguyên cả từ danh sách thành phần, không chỉ từ phần "Có chứa".
- **Lỗi thật phát hiện:** "Sữa chiết từ đậu nành" bị coi là sữa bò; đã sửa (đáp án `FOODQA_G3_1802` được chỉnh).
- **Hạt cây theo Bảng 1 (C.7, C.8)** và **tuyên bố "không sữa"** (D.13): quy tắc benchmark và app đều đã cập nhật; sữa thực vật và "không sữa" không còn là bằng chứng sữa; bột kem không sữa vẫn cảnh báo cho người dị ứng sữa (có chủ ý).
- **Danh mục nhóm dị nguyên (`allergen_groups.json`):** sửa nhuyễn thể và thịt bò (bản tóm tắt JEMRA không kết luận cho hai nhóm này); thêm kiều mạch; tách yến mạch khỏi nhóm gluten (nhóm gluten chỉ còn cảnh báo yến mạch ở mức nhiễm chéo); thêm danh sách theo dõi (đậu các loại, côn trùng, kiwi) ở trạng thái tắt; mô tả mức nghiêm trọng của cần tây, mù tạt, lupin, đậu nành, kiều mạch, yến mạch đổi thành "không nêu trong tài liệu"; cảnh báo của app nay hiện mức ưu tiên toàn cầu/khu vực (không đổi mức nghiêm trọng).

Chưa làm: mô tả mức nghiêm trọng của 10 nhóm còn lại (gluten, sữa, trứng, đậu phộng, hạt cây, giáp xác, cá, mè, sulfit, lactose) còn chi tiết không có trong bản tóm tắt JEMRA; hướng dẫn xử trí khi phản ứng, lưu ý giai đoạn chuyển tiếp của mè, và lưu bao bì/báo cáo sự cố (từ trang "Food Allergies: What You Need to Know" của FDA) chưa được thêm vào app.

### 5.10 Câu phụ gia bổ sung từ ba nguồn mới và Codex mục 4.2.1.7 (1.778 → 1.870)
Mục đích: phần phụ gia có 87% nhãn SAFE nên không đo được khả năng nhận diện phụ gia cần thận trọng. Thêm **92 câu** (nhãn CAUTION, tier `silver`), mỗi câu chỉ khẳng định đúng điều văn bản nguồn nói và đều qua cổng G1–G6:

| Chủ đề (`authority_topic`) | Số câu | Nguồn (tệp trong `knowledge_base/sources/raw/external/`) | Claim kiểm chứng |
|---|---:|---|---|
| `eu_annex_v_colours` | 52 | EU Reg. 1333/2008 Phụ lục V (`EU_REG_1333_2008_ANNEX_V.pdf`): E102, 104, 110, 122, 124, 129 phải kèm cảnh báo về trẻ em; ngoại lệ đánh dấu thịt/vỏ trứng và đồ uống trên 1,2% cồn | `EU_ANNEXV_COLOURS_WARNING` |
| `iarc_v134_aspartame` | 22 | IARC Monographs Vol. 134 (`IARC_V134_ASPARTAME.pdf`): aspartame Nhóm 2B | `IARC_V134_ASPARTAME_2B` |
| `fda_red3_revoked` | 7 | FDA, FR Doc. 2025-00830 (`FR_RED3_REVOCATION.pdf`): bãi bỏ FD&C Red No. 3 trong thực phẩm, hiệu lực 15/01/2027 | `FDA_RED3_REVOKED_FOOD` |
| `codex_sulphite_declaration` | 11 | Codex CXS 1-1985 mục 4.2.1.7 (`cxs_001e.pdf`): sulphite từ 10 mg/kg phải khai báo bằng tên; chỉ lấy nhãn ghi mã E220–228 mà không có chữ sulphite/sulfite | `CXS1_4217_SULPHITE_DECLARATION` |

Giới hạn ghi ngay trong đáp án: Phụ lục V là quy định ghi nhãn của EU, không suy ra yêu cầu ở Việt Nam; lệnh của FDA dựa trên điều khoản Delaney và dữ liệu ở chuột đực, không phải đánh giá nguy cơ ở người; IARC mô tả mức độ bằng chứng chứ không phải rủi ro ở một mức tiêu thụ; nhãn không có hàm lượng nên không kết luận được vi phạm. Không câu nào nhắc JECFA hay EFSA.

Quy trình: `add_authority_records_v5.py` (thêm dòng đầu vào, cùng ảnh với bản ghi cũ, id từ `FOODQA_G3_2101`) → `create_benchmark_v5.py` → `verify_v5_answers.py` → `finalize_benchmark_v5.py --activate`. Dùng Python hệ thống (venv thiếu PyMuPDF). Số claim nguồn: 155, tất cả đạt.

Kết quả: phụ gia từ 440 lên 532 câu (SAFE 382, CAUTION 100, ABSTAIN 50); nhãn SAFE còn khoảng 72%, baseline "luôn đoán SAFE" của phụ gia khoảng 72%.

### 5.11 Trường nhóm, dạng câu hỏi và báo cáo kết quả
Thêm vào mỗi bản ghi hai trường sinh trong `finalize_benchmark_v5.py`: `split_group` (một ảnh là một nhóm, 489 nhóm) và `template_id` (dạng câu hỏi trong từng domain, 255 dạng). `scripts_research/benchmark_report.py` báo cáo theo domain và tier, accuracy, macro-F1, accuracy cân bằng theo dạng câu hỏi, baseline "luôn đoán nhãn đa số" và khoảng tin cậy lấy mẫu lại theo ảnh. Mỗi claim nguồn PDF nay có `evidence_pages` (16/19 claim PDF định vị được; 3 claim bảng/định nghĩa dạng phân tích chưa định vị) và mỗi bản ghi có `source_claim_pages` (1.594/1.870). Chưa làm vùng ảnh (bounding box) cho đoạn trích nhãn; việc này chỉ giúp tra cứu, không ảnh hưởng điểm số. Trên tier `core`, baseline nhãn đa số đạt 74,9% accuracy nhưng macro-F1 chỉ khoảng 0,49, nên khi chấm hệ thống phải so với các con số này.

### 5.12 Từ chối hai loại trong hệ thống và độ phủ trên bộ chuẩn

Hệ thống từ chối theo hai loại (`insufficient_data` → chụp lại nhãn / bổ sung thông tin; `out_of_scope` → hỏi chuyên gia), mô tả ở `docs/QUY_TRINH_3_CAU_HINH_HE_THONG.md`. Đo bằng `scripts_research/measure_abstention.py` (tầng luật, Cấu hình B, OCR = 1,0): từ chối đúng 330/345 câu gold ABSTAIN (nhóm `PAL_NOT_ON_LABEL` 118/119, `LABEL_MISSING_INFO` 212/226) và từ chối nhầm 77/1525 câu có đáp án khác. Bộ chuẩn chưa có câu gold thuộc loại vượt phạm vi, và điểm số chỉ so nhãn ABSTAIN chứ không chấm loại từ chối hay nội dung lời khuyên; nên loại `out_of_scope` hiện chỉ được kiểm bằng test đơn vị (`tests/test_abstention.py`).

### 5.11 Câu hỏi thai phụ từ hướng dẫn WHO 2023 về chất ngọt không đường (1.870 → 2.012)

Trước đó nhóm thai phụ chỉ có 15 câu, đều dựa trên TiO2 và nitrit. Nguồn bổ sung là *Use of non-sugar sweeteners: WHO guideline* (2023), tệp `Bookshelf_NBK592258.pdf` (90 trang, bạn tải thủ công; hai tệp HTML cùng đợt chỉ là trang CAPTCHA nên không dùng). Thêm **71 câu cờ cảnh báo** (`authority_topic = who_nss_pregnancy`, persona `pregnant_woman`, nhãn CAUTION, tier `silver`), mỗi câu ứng với một nhãn sản phẩm có kê khai INS 950, 951, 952, 954, 955, 960, 961 hoặc 969 (thực tế gặp 950: 32, 955: 24, 951: 22, 960: 8, 952: 1).

Claim `WHO_NSS_2023_PREGNANCY` đối chiếu nguyên văn 9 đoạn trong PDF và ghi số trang: khuyến cáo có điều kiện (tr. 34), áp dụng cho cả phụ nữ mang thai và cho con bú, trừ người đã có đái tháo đường (tr. 13, 37), phân tích gộp ba nghiên cứu đoàn hệ về sinh non (OR 1,25; KTC 95% 1,07-1,46; độ chắc chắn thấp, tr. 26, 36), không thấy liên quan với đái tháo đường thai kỳ, lưu ý nhiễu và quan hệ nhân quả ngược (tr. 78), danh sách NSS và việc loại polyol (tr. 37).

Phạm vi và giới hạn:
- Câu trả lời chỉ nói những gì WHO nói: bằng chứng hạn chế, quan sát, không chứng minh nhân quả, không có ngưỡng an toàn riêng cho thai phụ; không viết "gây sinh non" hay "không an toàn".
- Polyol (sorbitol 420, maltitol 965, erythritol 968, xylitol 967) không bị gắn cờ vì WHO nói rõ chúng không phải NSS. Thaumatin (957) và hỗn hợp aspartame-acesulfame (962) không có trong danh sách WHO nên không được đưa vào.
- Chỉ phát hiện NSS theo mã INS trên nhãn (và tên aspartame); nhãn chỉ ghi tên chất ngọt khác mà không có mã sẽ không được gắn cờ.
- **Câu đối chứng (`who_nss_pregnancy_absent`, 71 câu, nhãn SAFE, tier `silver`):** cùng một câu hỏi trung tính với 71 câu cờ cảnh báo ("nhãn có chứa chất ngọt không đường ... không?"), nên mẫu câu hỏi không tiết lộ nhãn. Chọn ngẫu nhiên (seed cố định 2026) từ các nhãn mà **không có bất kỳ dấu hiệu chất ngọt nào**: không mã 950-969, không tên chất ngọt, không chữ "ngọt"/"sweet" ở bất cứ trường nào của bản phiên âm, danh sách thành phần đủ dài (từ 30 ký tự). Quy tắc cố ý rộng để nhãn nghi ngờ bị loại hơn là gắn nhầm SAFE; cổng G3 quét lại độc lập toàn bộ bản phiên âm. "SAFE" ở đây chỉ có nghĩa "không kê khai NSS": câu trả lời nói rõ kết luận chỉ về NSS, không đánh giá các phụ gia khác, và polyol không phải NSS.
- Nhóm thai phụ hiện có **157 câu** (86 CAUTION, 71 SAFE). Vẫn chỉ có hai nhãn và chỉ một yếu tố (NSS cùng TiO2/nitrit), nên chỉ dùng để mô tả và đo báo động giả, không so sánh thống kê giữa các cấu hình về lâm sàng.
- Phát hiện chất ngọt chỉ dựa trên mã INS (và tên aspartame) khi gắn cờ; nhãn chỉ ghi tên chất ngọt khác mà không có mã sẽ không được gắn cờ (đã được loại khỏi nhóm đối chứng).
- Nguồn dùng cho kiểm tra này không phải hướng dẫn riêng về thai kỳ: đây là hướng dẫn chung của WHO có một đoạn về thai phụ.

### 5.12 Mở rộng phủ mã INS: câu danh mục TT24 và câu ADI JECFA (2.012 → 2.091)

Trước đó 69 mã INS (cơ sở số) được hỏi trong khi nhãn in ra 150 mã. Script `add_authority_records_v5.py` (hàm `ins_coverage_rows`) chọn các mã in trên nhãn, **chưa được bản ghi nào hỏi**, nằm trong danh mục TT24 và đúng khoảng INS (100-1599). Mỗi câu hỏi **đúng một mã**, đáp án do các hàm dựng sẵn sinh từ nguồn đã kiểm (`permitted_answer`, `codex192_sentence`, `jecfa_status_answer`); không viết tay, không nguồn mới:

| Loại (`authority_topic`) | Số câu | Nội dung | Nhãn |
|---|---:|---|---|
| `ins_listing` | 61 | Có tên trong Phụ lục 1 TT24 (kèm Codex 192 nếu có); tối đa 2 câu mỗi mã; không kết luận an toàn | PERMITTED_LISTED (SAFE) |
| `ins_adi` | 18 | Trạng thái ADI theo JECFA; chỉ khi mã có bản ghi JECFA đã kiểm; tối đa 1 câu mỗi mã | 12 SAFE (ADI không dạng số), 6 ABSTAIN (ADI dạng số, thiếu hàm lượng) |

Kết quả: 36 mã mới, số mã INS khác nhau trong bộ từ 69 lên 105. Cả 79 câu thuộc tier `core` (là tra cứu xác định, giống các câu cùng loại đã có) và qua G1-G6. Số mã mới ít hơn ước tính ban đầu (82) vì phép đếm ban đầu tính theo bản đã sinh lại; theo dữ liệu đầu vào thực tế, 109 mã đã được hỏi, phần còn lại là mã đọc sai (900, 109, 130, 137...) hoặc không có trong TT24 nên bị loại theo quy tắc "không nguồn thì không đưa vào". Không thêm câu "mã không có trong danh mục": không có mã thật nào nằm trên nhãn mà thiếu trong TT24. Hệ quả cần biết: domain `additive_regulation` core thiên SAFE hơn (cận đa số 0,879 → 0,895), nên vẫn chỉ dùng domain này để kiểm tra tính đúng, không để so sánh cấu hình.

### 5.13 Câu về đường theo Codex CAC/GL 23-1997 (2.091 → 2.141)

Nguồn WHO "Guideline: sugars intake for adults and children" (`WHO_SUGARS_2015_PAGE.html`) **không dùng được**: tệp chỉ là trang điều hướng, không có nội dung ngưỡng 10% / 5% năng lượng. Hơn nữa ngưỡng đó áp dụng cho tổng năng lượng cả ngày, không so được với một sản phẩm. Thay vào đó dùng bảng công bố dinh dưỡng trong `cxg_023e.pdf` (đã có): **"sugars free" = không quá 0,5 g/100 g (rắn) hoặc 100 ml (lỏng)**, claim `CODEX_GL23_SUGARS_FREE_PER_100G` (đối chiếu nguyên văn, trang 3).

Thêm **50 câu** (`sugar_codex`, persona đái tháo đường & tăng huyết áp, tier `core`): cùng quy tắc với natri, chỉ dùng nhãn có `sodium_basis` (cơ sở của bảng dinh dưỡng, dùng chung cho cả bảng) và một dòng "đường tổng số" dạng số đơn thuần; quy đổi về /100 g hoặc /100 ml rồi so với 0,5 g. Cổng G3 tự tính lại độc lập từ nhãn. Giới hạn: **48 CAUTION, 2 SAFE** (hầu hết sản phẩm có đường vượt 0,5 g/100 g; 2 câu SAFE là kẹo gum xylitol và trân châu có "0 g"); đây là so sánh với ngưỡng công bố dinh dưỡng, không phải kết luận y khoa về đường huyết. Đường bổ sung/lactose/glucose... không được tính là đường tổng.

### 5.14 Câu hỏi của phụ huynh trẻ 5 tuổi: chất ngọt không đường (WHO 2023) và caffeine (EFSA 2015) (2.141 → 2.311)

Nhóm `child_5yo` trước đó chỉ có 84 câu, đều theo hai dạng câu về phẩm màu/tăng động (nguồn EFSA 2008). Thêm **170 câu** (persona `child_5yo`, domain `vulnerable_groups`, tier `silver`), nâng nhóm này lên 254 câu (120 CAUTION, 134 SAFE).

| Chủ đề | Cờ cảnh báo (CAUTION) | Đối chứng không có chất đó (SAFE) | Claim đã đối chiếu nguyên văn |
|---|---:|---:|---|
| `who_nss_child` / `who_nss_child_absent` | 71 | 71 | `WHO_NSS_2023_CHILDREN` (PDF NBK592258; trang 9–37, gồm trang 26) |
| `efsa_caffeine_child` / `efsa_caffeine_child_absent` | 14 | 14 | `EFSA_CAFFEINE_2015_CHILDREN` (EFSA Journal 2015;13(5):4102, 120 trang, bạn tải thủ công; trang 1 và 20) |

**WHO NSS, trẻ em.** Khuyến cáo có điều kiện không dùng NSS để kiểm soát cân nặng hoặc giảm nguy cơ bệnh không lây áp dụng cho trẻ em như người lớn (trừ người đã có đái tháo đường; trang 13). Bằng chứng trực tiếp ở trẻ em hạn chế: một thử nghiệm ngẫu nhiên có đối chứng ghi nhận giảm một số chỉ số mỡ cơ thể, nhưng khi gộp với thử nghiệm thứ hai thì không có tác động lên BMI z-score; các nghiên cứu quan sát không thấy liên quan có ý nghĩa; độ chắc chắn chung được đánh giá là thấp (trang 12, 26, 35). WHO nói rõ hướng dẫn này dựa trên mức dùng đã được coi là an toàn và không thay thế các mức an toàn hiện hành (trang 16). Vì vậy đáp án chỉ là **cờ cảnh báo**, không nói chất này gây hại cho trẻ. Quy tắc phát hiện nhãn dùng chung với chủ đề thai phụ (mã INS 950, 951, 952, 954, 955, 960, 961, 969; polyol không bị gắn cờ).

**EFSA caffeine.** Điều EFSA viết cho trẻ em và thanh thiếu niên là: *thông tin hiện có không đủ để xác định mức caffeine an toàn* (trang 1). EFSA chỉ nói mức không đáng ngại đã xác định cho người lớn khi dùng một lần (3 mg/kg thể trọng) **có thể làm cơ sở** để suy ra mức cho các nhóm này. Do đó đáp án **không nêu bất kỳ giới hạn an toàn nào cho trẻ**, và không viết mức 3 mg/kg là mức hằng ngày hay mức dành cho trẻ (test kiểm tra điều này). Cờ chỉ gắn khi danh sách thành phần có cà phê hoặc caffeine (EFSA nêu cà phê, trà, đồ uống mềm có caffeine và sô cô la là các nguồn caffeine chính trong khẩu phần; trang 20); không gắn khi chỉ có "hương cà phê", hay nhãn ghi không caffeine/khử caffeine; không gắn cho trà (có thể là trà thảo mộc) và sô cô la (ít caffeine), nhưng các nhãn đó cũng **không** được dùng làm đối chứng. Đối chứng là nhãn mà toàn bộ bản phiên âm không có dấu hiệu của bất kỳ nguồn nào EFSA nêu (cà phê, trà, ca cao/sô cô la, cola, guarana, nước tăng lực, taurine...).

Cách xây dựng giống nhóm thai phụ: một câu hỏi trung tính duy nhất cho cả câu cờ và câu đối chứng (test kiểm tra mỗi chủ đề chỉ có một dạng câu hỏi), đối chứng lấy mẫu ngẫu nhiên với seed cố định và bằng đúng số câu cờ, cổng G3 quét lại toàn bộ bản phiên âm bằng mẫu regex riêng. Các chủ đề mới được thêm **sau** mọi chủ đề cũ nên không có id nào của 2.311 bản ghi trước đó thay đổi (đã kiểm tra: 0 bản ghi cũ bị đổi).

Giới hạn: chỉ 14 nhãn có caffeine/cà phê và hầu hết nhãn không ghi hàm lượng, nên không thể ước tính mức tiếp xúc theo cân nặng; bằng chứng WHO cho trẻ em có độ chắc chắn thấp; mọi câu là cờ theo văn bản cơ quan, không phải khuyến cáo lâm sàng.

## 6. Bản ghi bị loại (2.081 → 1.778, sau đó thêm 92 câu ở 5.10, 142 câu ở 5.11, 79 câu ở 5.12, 50 câu ở 5.13 và 170 câu ở 5.14)

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

Số lượng theo từng lý do ở bảng trên. Danh sách ID chi tiết của 303 bản ghi bị loại từng nằm trong thư mục sao lưu cũ và đã bị xóa cùng thư mục đó, nên không còn tệp liệt kê từng ID.

## 7. Giới hạn cần nêu khi công bố

**Bộ chuẩn này không đo được:**
- Phụ gia không-SAFE ngoài các chủ đề ở 5.10 (màu azo/EU, Red No. 3, aspartame, sulphite, TiO2, photphat, nitrit, nhóm McCann) vẫn thưa. Thai kỳ và trẻ em chỉ được đo qua cờ cảnh báo của vài nguồn (chất ngọt không đường WHO 2023, TiO2, nitrit, phẩm màu; riêng trẻ em thêm caffeine EFSA 2015), không đo qua đánh giá phụ gia riêng cho từng nhóm.
- Đường chỉ được đo theo ngưỡng "không đường" của Codex trên 100 g (50 câu, 48 CAUTION và 2 SAFE); không đo ngưỡng WHO 10%/5% năng lượng vì ngưỡng đó tính theo cả ngày, không so được với một sản phẩm.
- Mức caffeine an toàn cho trẻ em: EFSA 2015 kết luận không đủ thông tin để xác định, nên bộ chuẩn không có và không được tạo ra bất kỳ giới hạn nào cho trẻ. Chỉ 14 nhãn có cà phê/caffeine và hầu hết không ghi hàm lượng, nên không ước tính được mức tiếp xúc theo cân nặng.
- Benzen trong thực phẩm không phải đồ uống.
- 69 mã phụ gia chưa có ADI trên JECFA (phosphat nhóm phụ, 1100(i), 960d, 640, 965…), nên các câu hỏi chứa những mã này bị loại.
- Tăng huyết áp chỉ đo qua natri (53 bản, ngưỡng Codex trên 100 g); không đo huyết áp hay tổng lượng natri trong ngày.

**Về phương pháp:**
- **Ô nhỏ.** Ba ô có dưới 30 bản ghi: `insufficient_data` × UNSAFE (3), `insufficient_data` × CRITICAL_ALERT (2) và `vulnerable_groups` × CRITICAL_ALERT (21, toàn bộ là PKU với aspartame, đã dùng hết các nhãn có aspartame). Không thêm bản ghi vì không có nguồn đã kiểm chứng; `benchmark_report.py` chỉ báo số lượng và khoảng tin cậy Wilson cho các ô này (không báo F1), và đọc domain `insufficient_data` như quyết định nhị phân từ chối/trả lời (301 ABSTAIN, 76 còn lại).
- **Tập đánh giá chuẩn FoodHealthQA 750 câu** `data/foodhealthqa_eval_set.jsonl` (tạo bằng `scripts_research/build_foodhealthqa_subset.py`): phân tầng cân bằng theo đúng 6 trụ cột chuyên môn (An toàn phụ gia 166, Dị ứng 165, Trẻ em 120, Bệnh nền 112, Thai phụ 105, Dinh dưỡng 82), khống chế $\le 4$ câu/ảnh, bao phủ 376 ảnh bao bì thực tế và 150 template khác nhau, có đầy đủ 5 nhãn an toàn (`SAFE` 285, `CAUTION` 277, `UNSAFE` 85, `ABSTAIN` 72, `CRITICAL_ALERT` 31), 100% đầy đủ 4 trường bắt buộc (*câu hỏi, câu trả lời, nguồn tham khảo, nhãn lĩnh vực*) phục vụ đánh giá so sánh các cấu hình A/B/C.
- **Tập đánh giá cân bằng** `data/qa_eval_balanced.jsonl` (tạo bằng `scripts_research/build_eval_subset.py`, không sửa bộ chuẩn): mỗi template tối đa 30 câu, mỗi ảnh tối đa 4 câu, `insufficient_data` giảm ABSTAIN về 1× nhãn lớn kế tiếp (đa số còn khoảng 55%), các domain khác 2×. Đây là góc nhìn dẫn xuất để chạy thử và giảm chi phí; kết quả chính vẫn báo trên bộ đầy đủ theo tier.
- **Nhóm trẻ em và PKU còn nhỏ** (child_5yo 254, pku_patient 57): chỉ nêu như quan sát. PKU bị giới hạn bởi số nhãn có aspartame; thêm bản ghi cần ảnh nhãn mới (cần cả bản ghi nền và phiên âm đã kiểm).
- **Vòng tròn quy tắc ↔ guardrail.** Đáp án chuẩn được dẫn xuất bằng quy tắc, và guardrail của hệ thống dùng quy tắc/kho tri thức tương tự, nên điểm của cấu hình B/C có thể cao một phần do trùng logic; mức cải thiện so với cấu hình A (không truy xuất) vẫn có ý nghĩa, nhưng không diễn giải điểm B/C là độ an toàn thực tế.
- Nhãn nhóm dễ tổn thương và nhóm natri là **cờ theo văn bản của cơ quan được trích**, không phải chẩn đoán hay khuyến cáo lâm sàng; chưa có chuyên gia đánh giá.
- Kiểm tra nhãn với ảnh do một người thực hiện, chỉ một mặt bao bì mỗi sản phẩm; câu cảnh báo nằm ở mặt khác thì không phát hiện được. Một số ảnh bị cắt dở hoặc mờ (ví dụ 0472, dùng danh sách thành phần do người dùng cung cấp).
- Bộ tái suy luận dị nguyên và cờ nhóm dễ tổn thương do cùng một người xây dựng nên không độc lập tuyệt đối.
- 34 bản ghi (khoảng 1,5%) chỉ khẳng định về nhãn, không khẳng định pháp lý.
- Hướng dẫn FDA Q&A là tài liệu không bắt buộc của Mỹ, chỉ dùng để giải thích cách hiểu "dẫn xuất từ dị nguyên"; nhãn trong dữ liệu là hàng bán tại Việt Nam, chưa đối chiếu văn bản pháp quy Việt Nam về dị nguyên (trong danh mục nhóm dị nguyên, mục Việt Nam đang ở trạng thái chờ kiểm tra văn bản chính thức).
- Các script và test của quy trình v3/v4 đã bị xóa; bộ QA chỉ còn v5 và dựng lại bằng chuỗi script v5 (`verify_external_claims` → `add_authority_records_v5` → `create_benchmark_v5` → `finalize_benchmark_v5 --activate`). v3 vẫn có thể lấy từ lịch sử git (commit `d1612d35`). Thư mục `data/archive/backups_2026-11/` chỉ giữ các bản sao lưu trước mỗi lần mở rộng.
- Mã sinh và kiểm các chủ đề mới (WHO NSS, INS listing/ADI, đường Codex, trẻ em) từng bị mất khỏi thư mục làm việc và đã được dựng lại từ nhật ký phiên làm việc; chạy lại chuỗi cho ra đúng 2.141 bản ghi giống hệt dữ liệu trước đó, nhưng nên giữ mọi thay đổi trong git để tránh lặp lại.

**Về thực nghiệm (chưa có kết quả):**
- Chưa chạy 3 cấu hình A (LLM không truy xuất), B (RAG trên văn bản chuẩn) và C (ảnh → OCR → RAG) trên bộ này, chưa đo CER/WER của OCR, chưa chạy ablation (retriever, top-k, embedding, prompt, nguồn tri thức), nên chưa có số liệu EM, F1, citation accuracy, groundedness, faithfulness hay độ an toàn. Mọi con số trong báo cáo này chỉ mô tả bộ dữ liệu và nguồn, không mô tả hiệu năng hệ thống.
- Khi báo kết quả cần đi kèm: macro-F1 và baseline đa số theo domain, baseline theo template (cross-validated), khoảng tin cậy lấy mẫu lại theo ảnh, và ghi rõ các ô nhỏ ở trên. Bộ chỉ là silver (`expert_validated = false`), không gọi là gold.
