# Báo cáo công việc đã thực hiện và kết quả đạt được — FoodHealthQA

Ngày lập: 2026-11-19 · Nhánh: `vtbly_add_source_new`

> Nguồn số liệu: file kết quả trong repo (`data/analysis/`, `data/calibration/`, `data/runs/`, `data/ocr_regression_baseline.json`), lịch sử git và ghi chú dự án. Số nào chưa đo lại sau lần thay đổi gần nhất đều được ghi rõ. Benchmark là **silver, kiểm chứng bằng luật, chưa có chuyên gia** (`expert_validated=false`), không được mô tả là gold.

---

## 1. Tổng quan

FoodHealthQA là hệ thống hỏi–đáp nhãn thực phẩm tiếng Việt về **phụ gia và dị nguyên**. Luồng xử lý: ảnh → OCR → chuẩn hóa/liên kết thực thể → truy hồi → luật an toàn tất định → LLM diễn đạt có trích dẫn, hoặc từ chối trả lời có nguyên tắc. **Luật quyết định nhãn an toàn, LLM chỉ diễn đạt.**

Công việc trong giai đoạn này chia thành 6 mảng:

1. Xây dựng và kiểm chứng bộ benchmark (v5).
2. Rà soát thủ công toàn bộ dữ liệu nhãn.
3. Mở rộng cơ sở tri thức (KB) và nguồn trích dẫn.
4. Cải thiện OCR và trích xuất thành phần từ ảnh thật.
5. Lớp tin cậy: từ chối trả lời (abstention), hiệu chỉnh ngưỡng, NLI kiểm chứng.
6. Giao diện, hạ tầng đánh giá, tài liệu.

---

## 2. Benchmark và bộ đánh giá

### 2.1 Benchmark v5
- Quy trình: `create_benchmark_v5.py` → `verify_v5_answers.py` (các cổng G1–G6) → `finalize_benchmark_v5.py`.
- Danh sách dương TT24/2019 (400 mã phụ gia) trích tự động từ PDF gốc.
- **Quy mô qua các mốc** (số bản ghi thay đổi vì mỗi bước đều có lý do, không phải mất dữ liệu):
  - 2.081 → 2.069: bỏ 0585 (trùng 0467) và 6 bản ghi natri chỉ có khoảng giá trị/hai sản phẩm.
  - 2.069 → **1.778**: áp nguyên tắc "không có nguồn thì không vào bộ chuẩn", loại 283 bản ghi không tra cứu được nguồn (ABSTAIN phụ thuộc KB thiếu JECFA, ngưỡng đường, thai kỳ, benzen).
  - 1.778 → 2.141 (2026-11-10): thêm 9 chủ đề "authority" có nguồn (WHO NSS thai kỳ, ins_listing, ins_adi, sugar_codex…) sau khi mất script và dựng lại.
  - 2.141 → **2.311** (hiện tại): thêm 170 bản ghi silver cho trẻ 5 tuổi (WHO NSS 2023 và EFSA caffeine 2015), không đổi bản ghi cũ. Phân bố: core 1.498, silver 813.
  - **Con số 1.778 trong bản báo cáo trước là số cũ (từ ghi chú ngày 2026-11-09), không còn đúng. Số hiện hành là 2.311** (`data/benchmark_manifest.json`, `data/build/qa_build_summary.json`; 489 ảnh).
  - 2.311 bản ghi là kho nguồn đã lưu trữ ở `data/archive/master_benchmark_2311/`. Bộ dùng để chấm là 750 câu trích phân tầng từ kho này.
- Ngưỡng natri làm lại theo Codex CAC/GL 23-1997 (≤ 0,12 g/100 g); 53 bản ghi tăng huyết áp được chấm điểm.
- Bản ghi nhóm nhạy cảm (297) chấm theo ý kiến EFSA đã đọc trực tiếp; kết luận là "nhãn có/không có chất mà văn bản nêu", không phải tư vấn lâm sàng.

### 2.2 Bộ đánh giá 750 câu
- `data/foodhealthqa_eval_set.jsonl`: **750 câu hỏi–đáp, 376 ảnh**, là bộ duy nhất dùng để chấm/demo. Chia theo `split_group` (1 ảnh = 1 nhóm) thành dev 387 / test 363, tránh rò rỉ.
- Benchmark không bao giờ được dùng làm KB truy hồi hay hiển thị trên UI.

### 2.3 Kết quả kiểm chứng nguồn
- Văn bản gốc lấy bằng curl (21 USC 321(qq), 21 CFR 172.804, JEMRA 2021, WHO, EFSA, TT24): **8/8 claim đạt**.
- Phát hiện và sửa sai sót ở bản cũ: v4 trích JEMRA/FALCPA cho đậu nành và nhuyễn thể nhưng nguồn nói ngược lại; catalog sai URL/ID (DOI EFSA 2008.852 là xylitol → đúng là 2008.660); locator Codex Table 3 sai ở 183/374 bản ghi; có ngưỡng đường "WHO/FDA" bịa ở nhóm dễ tổn thương.
- JECFA: tra được **130/186 mã** có ADI (`fetch_jecfa_more.py`).

---

## 3. Rà soát thủ công dữ liệu nhãn

Xem ảnh đối chiếu bản chép nhãn (một người rà, chưa có chuyên gia):

| Đợt | Phạm vi | Lỗi nặng |
|---|---|---|
| Mẫu ngẫu nhiên | 100 ảnh | 11/100 |
| High-risk lô 1 | 60 | 7/60 |
| lô 2 | 60 | 2/60 |
| lô 3–7 | 201 | 0/30, 4/76, 1/50, 0/45 |
| Phần còn lại | 74 | 1 nặng (0287 thiếu câu đậu phộng) |

- Tổng: **toàn bộ ảnh đã được rà** (321 ảnh high-risk + mẫu 100 + 74 còn lại); tỉ lệ lỗi nặng giảm từ ~11% xuống ~0 ở các lô cuối.
- Lỗi hệ thống: đọc sai mã con INS (160c(i) ↔ (ii)), thiếu câu "sản xuất cùng sản phẩm có trứng/đậu phộng", thiếu dòng "Thông tin dị ứng".
- Đã sửa hàng chục nhãn JSON, chạy lại benchmark sau mỗi đợt (có backup). Xóa ảnh 0429 (sai sản phẩm) và 0585 (trùng).
- Câu hỏi mang mã cũ được sinh lại từ nhãn đã sửa (`refresh_from_corrected_label`): số bản ghi chấm điểm tăng 1.534 → 1.583.
- Thêm nhận diện phụ gia chỉ viết tên (phosphate, màu McCann, nitrit, TiO2, aspartame) — `additive_names.py`.
- Kiểm tra nhất quán "Contains": 31 nhãn bị cờ; 18 nhãn nhà sản xuất bỏ sót (mè nhiều nhất, 10 nhãn). Kết luận áp dụng: **phát hiện dị nguyên từ cả danh sách thành phần, không chỉ từ câu "Contains"**; lỗi regex sữa đậu nành được sửa.

---

## 4. Cơ sở tri thức và luật

- Bổ sung nguồn: Codex CXS 192/CXG 36/CXS 1/CXS 72/CXS 74, FDA Allergen Q&A Ed.5, QCVN 11-1:2012/BYT, Nghị định 100/2014, JEMRA Part 1. Bảng INS Codex đã parse: 322 mã (Table One) + 192 mã (Table Three).
- Sửa `allergen_groups.json`: bỏ nhãn "ưu tiên JEMRA" không có căn cứ ở nhuyễn thể/thịt bò; thêm `buckwheat`, nhóm `oats` (khu vực, CXS 1-1985 4.2.1.5); yến mạch tách khỏi từ khóa gluten (cảnh báo lây nhiễm chéo).
- Luật dị nguyên trong app: bỏ qua sữa thực vật và tuyên bố "không sữa", nhưng **cố ý giữ cờ** với bột kem (nguy cơ caseinate). Hạt cây theo bảng FDA Table 1.
- Thêm `infant_rules`, `infant_food_additives.json`, mô hình ràng buộc theo persona.
- **Còn tồn đọng:** câu mô tả `clinical_severity` của cần tây/mù tạt/lupin/đậu nành chưa có nguồn hỗ trợ; chưa có nguồn ngưỡng đường, phụ gia thai kỳ; benzen chỉ có cho đồ uống.

---

## 5. OCR và trích xuất thành phần từ ảnh thật

Đo trên 376 ảnh thật bằng `ocr_regression.py` (baseline `data/ocr_regression_baseline.json`, 2.407 mã vàng):

| Mốc | Code recall | Code precision | Allergen recall | Fake allergens | Oversized |
|---|---|---|---|---|---|
| Trước khi cải tiến | 47,1% | 93,7% | 67,7% | 28 | 28 |
| Bước 1–3 (parser/linker) | 54,9% | – | – | 28 | 28 |
| Bước 4 (pp_v5) | 62,9% | 94,3% | 72,1% | 28 | 28 |
| **pp_v7 (hiện tại)** | **66,5%** | – | **75,6%** | 31 | 24 |

Số mã bỏ sót giảm từ **1.264 xuống ~920**.

**Việc đã làm**
- Phân tích nguyên nhân (`analyze_code_misses.py`): nguyên nhân chính là *thứ tự đọc trộn cột làm cụm từ dừng của khối bên cạnh cắt danh sách*, không phải lỗi ký tự của VietOCR.
- Sửa parser: dừng mềm (chọn theo đo đạc), sửa nhầm ký tự theo ngữ cảnh mã, sửa hậu tố La Mã, tiêu đề mờ.
- Sửa linker: không chèn dấu cách vào hậu tố chữ (lỗi "INS 150a" → "INS 150 a"), hậu tố chưa đóng, mã trần có ngữ cảnh, lọc đơn vị, mã mồ côi cạnh mã đã gắn INS.
- Sửa chính tả theo cụm (`ocr_lexicon`, pp_v6): lexicon từ registry + từ khóa dị nguyên + cụm vàng của tập dev; trên tập test allergen recall 70,3 → 70,9%, **132/141 sửa đúng**. Quy tắc sửa từ Latin edit-1 đã thử và loại bỏ (Carbohydrate → Carbohydrase).
- Fine-tune VietOCR (3.240 ảnh cắt dòng, train 2.813 / val 386): đọc chữ tốt hơn nhưng làm mất tiêu đề "Thành phần:" khiến parser trả 0 mục → thêm fallback không tiêu đề (pp_v7).
- Thử bộ phát hiện PP-OCRv6 trên 30 ảnh: không tốt hơn, không dùng.

**So sánh VietOCR fine-tune và gốc (181 ảnh test, `vietocr_ft_vs_base.json`)**

| Chỉ số | Gốc | Fine-tune |
|---|---|---|
| Token recall | 87,1% | 90,3% |
| Item recall | 80,6% | 85,9% |
| Code precision | 94,5% | 96,4% |
| Code recall | 65,4% | 66,8% |
| Allergen recall | 74,2% | 73,8% |

→ **Quyết định: chưa áp dụng bản fine-tune**, vì allergen recall (chỉ số ưu tiên an toàn) không tăng.

**Nguyên nhân còn lại (920 mã bỏ sót):** parser bị cắt 313, mã không có trong OCR 230, đọc sai chữ số 155, linker từ chối 55, sai hậu tố ~92. Hướng khắc phục thực sự là đọc theo cột (cần chạy lại OCR 376 ảnh vì chưa lưu hộp văn bản).

---

## 6. Lớp tin cậy

### 6.1 Hiệu chỉnh từ chối (`calibrate_confidence.py`, pipeline OCR `ocr_vietocr_prob_v2`, tập test 363 câu)

| Cấu hình | Accuracy | Coverage | False-safe | Danger recall | Abstain recall |
|---|---|---|---|---|---|
| B (nhãn chữ chuẩn) | 70,2% | 87,6% | 14,5% | 85,5% | 97,2% |
| C (ảnh → OCR) | 59,5% | 75,5% | 18,2% | 72,7% | 100% |

- Chênh B–C (~10,7 điểm accuracy) là phần đóng góp của lỗi OCR.
- Ngưỡng chọn trên dev là **0,0**, tức ngưỡng 0,65 hiện tại **không làm thay đổi kết quả**; việc từ chối thực tế đến từ quy tắc độ phủ KB và quy tắc mã không có trong KB lẫn nhãn.
- Điểm tin cậy OCR tương quan nghịch yếu với lỗi (Spearman −0,27 với token miss, −0,19 với item miss) — chưa đủ mạnh để dùng làm tín hiệu từ chối đáng tin.
- Liên kết INS: **recall** so với mã được hỏi là 99,6% (B) và 66,1% (C). Con số precision 34–37% ở bản trước là **lỗi cách tính, không phải lỗi bộ liên kết**: "mã chuẩn" chỉ gồm mã được nêu trong câu hỏi, còn mã liên kết gồm mọi mã trên nhãn, nên mã đúng nhưng không được hỏi bị đếm là FP. Đã sửa: precision không áp dụng cho so sánh này; ở cấu hình C so với mã liên kết từ nhãn tham chiếu (B) thì **precision 93,0%, recall 66,2%, F1 77,4%** (TP 3.631, FP 274, FN 1.851), tức OCR chủ yếu làm *mất* mã chứ ít *bịa* mã.
- **Lưu ý:** báo cáo hiệu chỉnh được tạo trước khi chuyển lên pp_v7; theo quy ước cần **chạy lại `calibrate_confidence.py`** trước khi dùng số này làm kết quả cuối.

### 6.2 NLI kiểm chứng câu trả lời (`nli_verifier.py`, `eval_nli_faithfulness.py`)
Trên mẫu **20 bản ghi / 89 claim**: faithfulness 51,7%, mâu thuẫn **0%**, không được chứng minh 48,3%. Theo trụ: An toàn phụ gia 100% (5 claim), Dị ứng 33,3% (9), Bệnh nền 0% (5), Chung 54,3% (70). Mẫu quá nhỏ để kết luận; điểm đáng chú ý là không có mâu thuẫn nhưng nhiều phát biểu chưa có bằng chứng đi kèm.

---

## 7. Giao diện, hạ tầng và tài liệu

- `app.py`: viết lại phần lớn giao diện Streamlit (tab, widget NLI, nhập liệu qua `chat_intake.py` ~800 dòng), chỉ cấu hình C hiển thị cho người dùng.
- Mô-đun mới/cập nhật: `nli_verifier`, `chat_intake`, `research_report`, `hybrid_retriever`, `evidence`, `abstention`, `constraint_model`, `evaluation`, `ablation`.
- Kiểm thử: bổ sung test cho NLI, chat intake, abstention, evaluation; hồi quy OCR chạy trên 376 ảnh. (Lần chạy đầy đủ gần nhất ghi nhận 142 test đạt; **chưa chạy lại trong phiên này**.)
- Tài liệu: `MO_TA_HE_THONG.md`, `QUY_TRINH_3_CAU_HINH_HE_THONG.md`, `MO_TA_BO_DANH_GIA_750_CAU.md`, `BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md`, `TIEU_CHI_LABEL_GOLD_STANDARD.md`, README, `CLAUDE.md`.
- Khôi phục script bị mất sau rewind (2026-11-10) từ transcript và đã kiểm chứng.
- Truy hồi đoạn văn: BM25 + TF-IDF, tùy chọn bge-m3 dense (đã có `passage_retrieval_eval*.json`).

---

## 8. Hạn chế và việc cần làm tiếp

1. Benchmark là silver, rà soát bởi một người, **chưa có chuyên gia xác nhận**; không nêu là gold.
2. Chạy lại `calibrate_confidence.py` sau pp_v7, rồi chạy lại `run_evaluation.py` / `compare_providers.py` để có số cuối cho A/B/C (thư mục `data/runs/gemini` hiện rỗng).
3. Cấu hình C còn kém B ~10 điểm accuracy do OCR: ưu tiên đọc theo cột để giảm 313 ca parser bị cắt.
4. Chưa áp dụng VietOCR fine-tune; cần allergen recall tăng cùng code recall.
5. Hoàn thiện nguồn cho `clinical_severity`, ngưỡng đường, thai kỳ; mở rộng mẫu NLI vượt 20 bản ghi.
6. (Đã xử lý) precision liên kết INS: sửa cách tính trong `calibrate_confidence.py`; còn 8 mã được hỏi mà cấu hình B không liên kết được (FN 8/2.058) chưa điều tra.
7. Giải quyết 31 fake allergens (INS 223 → sulfites, "Lạc" trong địa chỉ).
8. Chạy lại toàn bộ test trước khi commit; nhiều file đang chưa commit (30 file sửa, nhiều dữ liệu mới).
