# TỔNG HỢP KIẾN TRÚC VÀ QUY TRÌNH HOẠT ĐỘNG HỆ THỐNG FOODHEALTHQA

> **Tài liệu Báo cáo Kỹ thuật Toàn diện**  
> *Dự án: Hệ thống Hỏi đáp Tư vấn An toàn Thực phẩm & Dinh dưỡng Cá nhân hóa (FoodHealthQA)*  
> *Ngày cập nhật: 13/11/2026*  
> *Đối tượng phục vụ: Hội đồng chuyên môn, Giảng viên hướng dẫn, Nhóm nghiên cứu & Phát triển.*

---

## 1. TỔNG QUAN HỆ THỐNG & TRIẾT LÝ THIẾT KẾ

### 1.1. FoodHealthQA là gì?
**FoodHealthQA** là hệ thống AI hỏi đáp tư vấn an toàn thực phẩm cá nhân hóa đầu tiên tại Việt Nam kết hợp:
1. **Thị giác máy tính (OCR đa tầng)** đọc nhãn bao bì tiếng Việt.
2. **Chuẩn hóa thực thể 4-Agent (FoodOntoRAG)** kết hợp đồ thị bản thể học **FoodOn** (29.255 nút).
3. **Cơ sở tri thức pháp quy & khoa học đa nguồn (Data Hub 16 bảng tra cứu)**: Thông tư 24/2019/TT-BYT, Codex, JECFA, USDA FNDDS, Open Food Facts, FAO/WHO JEMRA.
4. **Bộ máy luật chốt chặn y tế tiền định (Deterministic Medical Guardrail)** dựa trên hồ sơ bệnh lý cá nhân hóa và các chỉ số sinh hóa **NHANES**.
5. **Mô hình ngôn ngữ lớn (LLM)** đa nền tảng (Google Gemini API & Qwen2.5 chạy cục bộ qua Ollama) đóng vai trò diễn giải ngôn ngữ tự nhiên có trích dẫn nguồn kiểm chứng.

### 1.2. Bảy nguyên tắc thiết kế cốt lõi (Core Principles)
* **Nguyên tắc 1: Luật quyết định an toàn — LLM chỉ diễn đạt ngôn ngữ (Decoupled Generation):** Mức độ an toàn của sản phẩm (vượt ADI, dị ứng, chống chỉ định bệnh lý) do máy luật y tế quyết định khách quan. LLM không bao giờ được tự bịa ra nhãn an toàn.
* **Nguyên tắc 2: RAG có căn cứ và trích dẫn bắt buộc (Source-Grounded):** Mọi nhận định lâm sàng phải đi kèm mã nguồn chính thức trong ngoặc vuông (ví dụ: `[VN_TT24_2019]`, `[JECFA_EVALUATIONS]`, `[CODEX_STAN_192_1995]`).
* **Nguyên tắc 3: Từ chối có nguyên tắc (Principled Abstention):** Khi ảnh quá mờ, độ tin cậy thấp ($S_{\text{total}} < 0.65$), hoặc thiếu dữ liệu y tế quan trọng, hệ thống từ chối trả lời (`ABSTAIN`) kèm giải thích rõ ràng, tránh gây hại cho người dùng.
* **Nguyên tắc 4: Hỏi gì tra nấy (Dynamic QA):** Câu hỏi của người dùng quyết định thông tin cần tra cứu; hệ thống xuất vết tra cứu minh bạch (Traceability) cho biết nguồn nào tìm thấy, nguồn nào thiếu.
* **Nguyên tắc 5: Nghiên cứu thực nghiệm đối chứng 3 Cấu hình (Ablation Study):** Tách bạch rõ rệt giá trị của RAG (so sánh A và B) và ảnh hưởng của sai số thị giác OCR (so sánh B và C).
* **Nguyên tắc 6: Bộ kiểm chuẩn tách biệt với kho tri thức (No Data Leakage):** Bộ QA 750 câu hỏi là bài thi độc lập, không dùng làm tài liệu huấn luyện hay kho tri thức truy xuất.
* **Nguyên tắc 7: Bền vững trước trôi dạt Ontology (Ontology Drift Resilience):** Ứng dụng mô hình FoodOntoRAG (IEEE BigData 2025) để liên kết thực thể chính xác mà không bị phụ thuộc vào việc tinh chỉnh lại mô hình khi ontology tiến hóa.

---

## 2. KIẾN TRÚC TỔNG THỂ HỆ THỐNG (END-TO-END PIPELINE)

```
                            NGƯỜI DÙNG
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. Ảnh bao bì (hoặc văn bản nhãn tham chiếu)                │
   │ 2. Câu hỏi tư vấn tự gõ (ngôn ngữ tự nhiên tiếng Việt)      │
   │ 3. Hồ sơ sức khỏe cá nhân (Persona + Biomarkers NHANES)     │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    ▼                             ▼                             ▼
【CẤU HÌNH A】              【CẤU HÌNH B】              【CẤU HÌNH C】
LLM không truy xuất         Văn bản nhãn chuẩn + RAG    Ảnh bao bì + OCR + RAG
(No-RAG Baseline)           (Gold Text Reference)       (Full Optical Pipeline)
    │                             │                             │
    │                             │                     ┌───────┴───────┐
    │                             │                     │    TẦNG 1:    │
    │                             │                     │  PERCEPTION   │
    │                             │                     │(PaddleOCR+    │
    │                             │                     │ Phục hồi lỗi) │
    │                             │                     └───────┬───────┘
    │                             │       Văn bản OCR           │
    │                             └──────────────┬──────────────┘
    │                                            ▼
    │                           ┌─────────────────────────────────┐
    │                           │            TẦNG 2:              │
    │                           │          NORMALIZATION          │
    │                           │   (FoodOntoRAG 4-Agent NEL      │
    │                           │    + Đồ thị FoodOn 29.255 nút)  │
    │                           └────────────────┬────────────────┘
    │                                            ▼
    │                           ┌─────────────────────────────────┐
    │                           │            TẦNG 3:              │
    │                           │     HYBRID RETRIEVAL & QA       │
    │                           │ (QA Động + 16 Bảng Data Hub:    │
    │                           │  TT 24, JECFA, Codex, FNDDS,    │
    │                           │  OFF, Food Hazards, Dị nguyên)  │
    │                           └────────────────┬────────────────┘
    │                                            ▼
    │                           ┌─────────────────────────────────┐
    │                           │            TẦNG 4:              │
    │                           │      GUARDRAIL & EVALUATION     │
    │                           │   (Toán WSDM '21 + Rule Engine: │
    │                           │    Phân loại Negation Constraint│
    │                           │    Tính S_total & Nhãn an toàn) │
    │                           └────────────────┬────────────────┘
    ▼                                            ▼
LLM trả lời trực tiếp                 Gói bằng chứng có kiểm soát
(Dễ bị ảo giác y khoa)                + Ép nhãn an toàn máy luật
    │                                            │
    └─────────────────────┬──────────────────────┘
                          ▼
             HẬU KIỂM VÀ LỌC TRÍCH DẪN
        (clean_llm_citations_and_label)
                          ▼
            KẾT QUẢ HIỂN THỊ STREAMLIT
    (Nhãn an toàn · Câu trả lời · Trích dẫn · Vết tra cứu)
```

---

## 3. CHI TIẾT 4 TẦNG XỬ LÝ KỸ THUẬT

### 3.1. TẦNG 1: PERCEPTION (Thị giác máy tính & OCR quang học)
*Mã nguồn thực thi:* `modules/ocr_extractor.py`

* **Bước 1: Tiền xử lý ảnh thích ứng (Adaptive Preprocessing):**
  - Khử góc nghiêng tự động: Dùng thuật toán phân ngưỡng nhị phân Otsu kết hợp `cv2.minAreaRect` để xoay ảnh về phương ngang chuẩn xác.
  - Cân bằng tương phản cục bộ CLAHE (*Contrast Limited Adaptive Histogram Equalization*) với lưới tile 8×8 để xử lý bao bì bị lóa đèn hoặc tối góc.
  - Làm nét cạnh chữ (Unsharp Masking) phục vụ quét nhãn chữ siêu nhỏ.
* **Bước 2: Nhận diện chữ đa tầng (Multi-engine OCR):**
  - Khung phát hiện vùng văn bản (Detection): Sử dụng **PaddleOCR DBNet** chuyên trị văn bản tự nhiên, uốn lượn trên chai lọ, hộp bánh.
  - Khung nhận diện ký tự (Recognition): Sử dụng **VietOCR (Transformer)** chuyên biệt cho dấu tiếng Việt phức tạp.
  - Tự động chuyển đổi chế độ dự phòng nhiều lượt (multi-pass fallback) khi phát hiện ảnh nhiễu.
* **Bước 3: Phục hồi chính tả sau OCR (Domain-Specific Lexical Post-correction):**
  - Dùng từ điển thuật ngữ phụ gia thực phẩm chuẩn hóa (không dùng LLM để đảm bảo tính tất định, tránh ảo giác).
  - Sửa lỗi quét quang học kinh điển: `INS 50O` $\rightarrow$ `INS 500`, `E21l` $\rightarrow$ `E211`, `Natri benzoat` bị mất dấu...
  - Chỉ gắn tiền tố "INS" cho số trong ngoặc khi số đó **có trong danh mục phụ gia** và cùng thành phần có từ chỉ nhóm phụ gia ("chất …", "phẩm màu", "hương liệu"…). (Trước đây mọi số 3–4 chữ số trong ngoặc đều bị gắn, vd "(7,4 g/l)" OCR thành "(749)" rồi thành mã giả "INS 749".)
  - Chỉ bỏ dấu ")" **thừa thật sự**; giữ nguyên ngoặc lồng hợp lệ như "(100(i))" (trước đây quy tắc "))"→")" làm lệch ngoặc và dính nhiều thành phần thành một cụm).
  - Phiên bản hậu xử lý: `OCR_POSTPROCESS_VERSION`. Bộ đệm OCR lưu chữ thô nên đổi phiên bản chỉ cần tính lại hậu xử lý, không phải OCR lại.
* **Bước 4: Phân đoạn thành phần bảo toàn ngữ cảnh (Regex Segmentation):**
  - Tách các thành phần độc lập bằng biểu thức chính quy bảo toàn dấu ngoặc đơn: Cụm `"Chất làm dày (INS 412, INS 415)"` không bị chẻ vụn mà được nhận dạng nguyên khối. Các mã trong cụm được tách ra ở tầng chuẩn hóa (mục 3.2).
* **Cập nhật 16/11/2026 — cải tiến hậu xử lý và trích xuất SAU OCR (không phải cải tiến mô hình OCR):**
  - *Phạm vi:* khâu dò vùng chữ (PaddleOCR) và đọc chữ (VietOCR) **không đổi** — độ phủ từ của chữ OCR thô giữ nguyên 87,8% trên 376 ảnh. Các thay đổi dưới đây nằm ở hậu xử lý, tách thành phần và liên kết mã (mục 3.2), nên cũng có lợi cho Cấu hình B (văn bản chuẩn, không qua OCR).
  - *Bộ kiểm tra hồi quy trên ảnh thật* (`scripts_research/ocr_regression.py`, mốc `data/ocr_regression_baseline.json`, chạy trong `tests/test_ocr_regression.py`): tính lại hậu xử lý → tách thành phần → dò mã/dị nguyên từ chữ OCR thô của 376 ảnh và so với danh sách thành phần chuẩn. Báo lỗi khi một chỉ số tụt quá ngưỡng cho phép (vd dị nguyên giả tăng > 2). Mọi thay đổi dưới đây chỉ được giữ khi qua bộ kiểm tra này.
  - *Phân tích mã bị sót* (`scripts_research/analyze_code_misses.py`): trước khi sửa, 1.264/2.388 mã (52,9%) bị sót. Nguyên nhân lớn nhất **không phải VietOCR đọc nhầm ký tự** (≈25% số mã sót) mà là **thứ tự đọc trộn cột**: các khung chữ được xếp theo hàng ngang trên toàn ảnh nên chữ của khối bên cạnh ("từ ngày sản xuất", "giá trị dinh dưỡng"…) chen vào giữa danh sách thành phần; bộ tách gặp cụm đó thì dừng → 521 mã (41%) bị cắt dù chữ OCR có đúng mã.
  - *Bộ tách không dừng sai chỗ:* gặp cụm điểm dừng mà ngay sau đó vẫn là danh sách (≥3 dấu phẩy, có mã phụ gia, ≤1 số kèm đơn vị) thì đi tiếp. Hai cụm luôn dừng (`HARD_STOP_PATTERNS`: "hướng dẫn sử dụng", "ingredients") được chọn bằng số đo: cho đi tiếp qua "hướng dẫn sử dụng" sinh +6 dị nguyên giả ("chấm cá, hải sản"), qua "ingredients" sinh +2 cụm quá dài.
  - *Sửa ký tự VietOCR đọc nhầm, chỉ trong ngữ cảnh mã* (`_fix_code_confusions`): hậu tố La Mã `500(11)`→`500(ii)`, `420(l)`→`420(i)`, `331 (ili)`→`331(iii)`; chữ giống số `16oc`→`160c`, `42o`→`420`, `inss51`→`INS551`; tiền tố `1NS`/`IN5`/`ivs`→`INS`. Chuỗi thường (`(7,4 g/l)`, `ISO 22000`, `Vitamin B1 (1)`) không bị đụng tới.
  - *Nhận tiêu đề bị đọc sai* ("THÀNH PHẦN" → "Hành PHÂN:", "NGREDIENTS:"): chỉ dùng khi không có tiêu đề đúng và phải có dấu ":" ngay sau.
  - *Sửa lỗi sẵn có:* quy tắc chuẩn hóa `INS 150a` chèn nhầm dấu cách thành `INS 150 a` (và `307b`, `472e`…) làm khâu liên kết không nhận mã.
  - *Kết quả trên 376 ảnh* (mốc cũ → sau bước 1–3 → sau khâu liên kết, mục 3.2; sửa chính tả theo cụm thêm bắt dị nguyên 72,1% → 73,2%, xem dưới):

    | Chỉ số | Trước | Sau tách + sửa ký tự | Sau liên kết |
    |---|---|---|---|
    | Bắt mã phụ gia (code recall) | 47,1% | 54,9% | **62,9%** |
    | Độ chính xác mã | 93,7% | 93,6% | 94,3% |
    | Bắt dị nguyên | 67,7% | 72,1% | 72,1% |
    | Dị nguyên giả / cụm quá dài | 28 / 27 | 28 / 28 | 28 / 28 |

  - *Lưu ý độ tin cậy của số đo:* các luật được viết sau khi soi lỗi trên chính 376 ảnh này (kể cả việc chọn `HARD_STOP_PATTERNS`), nên con số **có thể lạc quan**. Cần xác nhận trên ảnh nhãn mới chưa từng dùng để chỉnh luật; từ nay chỉ soi lỗi trên phần ảnh "phát triển", báo số trên phần "kiểm tra". Tổng số mã chuẩn đổi nhẹ (2.388 → 2.407) vì mã chuẩn cũng được tính bằng khâu liên kết.
  - *Còn lại (~920 mã sót):* ~310 vẫn bị bộ tách cắt (gốc rễ là thứ tự đọc), ~230 không có trong chữ OCR, ~155 VietOCR đọc sai chữ số. Hai nhóm sau chỉ cải thiện được bằng mô hình đọc chữ tốt hơn (xem mục 10 của `QUY_TRINH_3_CAU_HINH_HE_THONG.md`).
  - *Thử mô hình dò vùng chữ PP-OCRv6* (PaddleOCR 3.7, môi trường riêng `venv_ocr3`, `scripts_research/ocr_candidate.py`, so bằng `ocr_regression.py --cache`): trên 30 ảnh, bắt mã 47,5% → 46,3%, bắt dị nguyên 64,5% → 67,3%, chậm hơn (14 giây/ảnh dò vùng trên CPU) → **giữ mô hình hiện tại**. Mô hình đọc chữ PP-OCRv6 không dùng được cho tiếng Việt (bộ ký tự thiếu 90/136 chữ có dấu).
  - *Sửa chính tả theo cụm* (`modules/ocr_lexicon.py`, áp trong từng cụm thành phần của `parse_ingredients_from_ocr`; chỗ sửa lưu ở `mention["ocr_corrections"]`): chữ OCR sai dấu tiếng Việt thường vẫn là một từ đúng chính tả ("sửa", "cả", "khoảng"), nên **không sửa từng từ** (bộ sửa từng từ trước đây làm độ phủ từ 87,4% → 82,7%, sinh dị nguyên giả "trung"→"trứng"). Cụm 5 → 2 từ liền nhau, so ở dạng bỏ dấu, khớp đúng MỘT tên trong danh mục → thay bằng cách viết có dấu ("Khoảng chất"→"Khoáng chất", "Hượng liệu"→"Hương liệu", "lòng đó trứng gà"→"lòng đỏ trứng gà"). Không sửa khi chữ gốc đã là tên hợp lệ, khi dạng bỏ dấu ứng với ≥2 cách viết, khi chỉ khác chỗ đặt dấu ("hoá"/"hóa"), hoặc khi tên trong danh mục không dấu mà chữ gốc có dấu. Luật "từ Latin dài sai một ký tự" đã thử và **bỏ** vì đổi từ đúng thành từ khác ("Carbohydrate"→"Carbohydrase", "tocopheryl"→"tocopherol").
    - *Danh mục:* tên phụ gia trong `additive_registry.json`, từ khóa dị nguyên, 48 cụm viết tay cũ, và 397 cụm thành phần xuất hiện ở ≥2 ảnh thuộc phần **phát triển** (`knowledge_base/ocr_ingredient_lexicon.json`, tạo bằng `scripts_research/build_ocr_lexicon.py`; ảnh kiểm tra không được dùng).
    - *Đo riêng phần kiểm tra (181 ảnh chưa dùng để xây danh mục):* độ phủ từ trong cụm thành phần 58,5% → 59,1%, bắt dị nguyên 70,3% → 70,9%; mã, độ chính xác mã, dị nguyên giả, cụm quá dài không đổi. Đối chiếu từng chỗ sửa với danh sách chuẩn: 132 đúng, 1 "sai" (thực ra đúng — danh sách chuẩn ghi nhầm "tinh bột sản"), 8 không xác định được (phần lớn đúng, vd "Sửa Đậu Nành"→"Sữa Đậu Nành"). Trên cả 376 ảnh: bắt dị nguyên 72,1% → 73,2%.
    - *Giới hạn:* chỉ sửa được cụm có trong danh mục — vd "Dầu cả", "Kêm Sulphat", "Đóng Sulphat" trên nhãn ColosBaby chưa sửa được vì danh mục chưa có "dầu cá", "kẽm sulphat", "đồng sulphat". Không tự thêm tên theo ví dụ đang xem (tránh chỉnh theo dữ liệu kiểm tra); mở rộng danh mục nên lấy từ nguồn tổng quát (danh mục thành phần/khoáng chất chuẩn). Chữ của khối khác chen vào ("lê mạnh nhất", "khỏi các tác") và con số đọc sai ("1410 tuổi" thay vì 1–10) không sửa được bằng chính tả.
    - Giao diện (phần chi tiết, tab bóc tách nhãn): ô "Văn bản đọc từ nhãn (sau hậu xử lý)" và mục "Chữ OCR thô và các chỗ đã sửa" liệt kê từng chỗ sửa theo cụm và từng chỗ hậu xử lý chung đổi so với chữ OCR thô (trước đây ô "Sau từ điển thực phẩm" trùng nội dung với ô phía trên). LLM được dặn dùng tên đã chuẩn hóa khi nhắc tới thành phần, vì tên đọc từ ảnh có thể còn sai.
  - Phiên bản hậu xử lý hiện tại: `pp_v6_phrase_lexicon_correction`.
* **Điểm tin cậy thị giác:**
  $$S_{\text{perception}} = \frac{1}{|L|} \sum_{\ell \in L} \bar p(\ell) \in [0, 1]$$
  với $\bar p(\ell)$ là **xác suất trung bình các ký tự** mà VietOCR trả về cho dòng chữ $\ell$ (chỉ tính tới ký tự kết thúc của dòng). Trước đây PaddleOCR 2.9 khi chỉ dò vùng không trả điểm nên mọi dòng bị gán cứng 0,85 — con số đó không phản ánh chất lượng OCR. Độ tin cậy được kiểm chứng với lỗi OCR thật bằng `scripts_research/calibrate_confidence.py`.
  *(Với Cấu hình B, do dùng trực tiếp văn bản tham chiếu nên $S_{\text{perception}} = 1.0$).*
* **Tốc độ:** VietOCR đọc **theo lô** các vùng chữ cùng bề rộng (kết quả giống hệt đọc từng vùng) và torch được đặt lại ~6 luồng (paddle nạp trước làm torch tụt về 1 luồng; đổi bằng `FH_TORCH_THREADS`). OCR khởi tạo một lần ở luồng nền khi trang được mở lần đầu, dùng chung cho mọi phiên.

---

### 3.2. TẦNG 2: NORMALIZATION (Chuẩn hóa Thực thể với FoodOntoRAG 4-Agent & Đồ thị FoodOn)
*Mã nguồn thực thi:* `modules/additive_linker.py`, `modules/hybrid_retriever.py`, `modules/foodon.py`

Kế thừa công trình công bố tại **IEEE BigData 2025** (*Beyond Fine-Tuning: Robust Food Entity Linking under Ontology Drift with FoodOntoRAG*):

#### A. Kiến trúc 4 Tác tử FoodOntoRAG:
1. **Agent 1: Hybrid Retriever (Truy xuất lai):**
   - *Nhánh từ vựng (BM25):* Tra cứu trên nhãn, từ đồng nghĩa, định nghĩa.
   - *Nhánh ngữ nghĩa (SBERT):* Nhúng vector 384 chiều bằng `all-MiniLM-L6-v2` kết hợp đo khoảng cách Cosine trên FAISS.
   - Hợp nhất và xếp hạng: Khử trùng lặp, ưu tiên exact-match lên đầu, lấy Top 30 ứng viên.
2. **Agent 2: Selector Agent (Tác tử chọn lựa):**
   - Hoạt động dựa trên 2 quy tắc cứng:
     - *Ưu tiên khớp chính xác (Exact-match Preference):* Khớp hoàn toàn nhãn hoặc từ đồng nghĩa được ưu tiên cao nhất.
     - *Quy tắc cụ thể (Specificity Rule):* Chọn khái niệm hẹp cụ thể hơn thay vì khái niệm cha quá chung chung.
3. **Agent 3: Rule match scorer (điểm khớp theo luật — KHÔNG gọi LLM; hàm `rule_match_score`, tên cũ `llm_scorer`):**
   - Gán $\tau$ theo luật: khớp mã INS 0,97 · khớp tên chuẩn 0,94 · khớp tên đồng nghĩa 0,91 · còn lại 0,60 + phần điểm truy xuất (+0,02 nếu có FoodOn CURIE). $\tau$ **không phải xác suất đã hiệu chỉnh**; giao diện ghi là "Mức khớp (theo luật)".
   - Nếu $\tau \ge 0.70 \rightarrow$ chấp nhận; liên kết không khớp định danh (mã/tên/tên đồng nghĩa) bị loại.
4. **Agent 4: Synonym Generator & Feedback Loop (Sinh từ đồng nghĩa & Vòng phản hồi):**
   - Khi $\tau < 0.70$, đọc lý do lỗi (`failure_reason` từ Scorer) để sinh ra 5 cách diễn đạt thay thế (tên khoa học, biến thể viết đảo ngữ).
   - Kích hoạt vòng truy xuất lượt 2 (vòng lặp 1 bước kiểm soát chống trôi nghĩa).

#### A2. Một cụm nhiều mã & mã không đọc chắc (`modules/pipeline.link_all_mentions`):
- Cụm như "chất tạo ngọt tổng hợp (950, 955)" được tách thành **một liên kết cho mỗi mã**, chỉ dùng bộ dò định danh (`additive_linker.link_additive_mentions`), không đoán thêm mã. Trước đây mỗi cụm chỉ giữ một mã: trên văn bản nhãn chuẩn của bộ 750 câu, **1.150/3.032 mã (37,9%) bị bỏ**. Sau khi sửa: recall mã trên văn bản chuẩn 67,2% → 99,6%; nhãn an toàn tầng luật khớp nhãn chuẩn 60,5% → 62,9% (18 câu sai→đúng, 0 câu đúng→sai).
- Mã OCR **không có trong danh mục** (vd "481(i)" đọc thành "4310") được ghi là một mục "phụ gia không liên kết được" (`category = additive_uncertain_code`) để $S_{\text{retrieval}}$ giảm đúng thực tế; hệ thống **không** tự sửa sang mã "gần nhất" vì dễ thành một chất khác.
- **Cập nhật 16/11/2026 — khâu liên kết nhận mã đã đọc đúng số** (`modules/additive_linker.py`, đo bằng bộ kiểm tra OCR ở mục 3.1; bắt mã 54,9% → 62,9%, độ chính xác 93,6% → 94,3%):
  - Số 3–4 chữ số trần chỉ là mã khi cụm có ngữ cảnh phụ gia. Ngữ cảnh nay gồm thêm: cụm đã có mã INS/E ghi rõ (`INS 331(iii), 296, 334`); từ chỉ nhóm so ở dạng bỏ dấu, chịu được OCR sai dấu ("chật diều chính độ aciơ (260)", "Mầu thực phầm (129)") và tên chất phổ biến (glutamat, citrat…); dãy ≥2 số trong cùng ngoặc đều có trong danh mục ("(1442, 1440, 415)" khi tên nhóm bị đọc nát).
  - Số đi kèm đơn vị ("100 g", "250 ml", "%") **không** được coi là mã — riêng điều này tăng độ chính xác mã 93,6% → 94,5% mà không mất mã nào.
  - Chấp nhận hậu tố có ngoặc chưa đóng ("450(i" ở cuối cụm/trước dấu phẩy).
  - Cụm chỉ là một số mã trần (bị tách khỏi tên nhóm vì OCR mất ngoặc) đứng cạnh một cụm có mã → gắn "INS" (`ocr_extractor._tag_orphan_codes`); số phải có trong danh mục, năm "2023" hay số lô không tính. Đây là thay đổi lợi nhất (+4,7 điểm bắt mã).
  - **Không** dùng khớp tên gần đúng (vd "tinh bột biến tính aipha"): chỉ ~50 mã, dễ nhận nhầm sang chất khác — giữ nguyên nguyên tắc không dùng khớp mờ cho liên kết quy chuẩn.
  - *Ảnh hưởng lên bộ QA 750 câu* (tầng luật, không gọi LLM, `calibrate_confidence.py`, so với lần chạy ngay trước): Cấu hình B 69,3% → 69,3% (không đổi); Cấu hình C 56,4% → 58,7% (toàn bộ), phần kiểm tra 54,0% → 56,5%, độ phủ 73,0% → 76,0%, tỉ lệ trả lời "an toàn" sai giữ 20,0%. Chỉ số "liên kết INS so với mã chuẩn" của C: recall 53,8% → 62,2%, precision 38,1% → 34,9% — precision này so với các mã **câu hỏi nhắc tới** (cả B với văn bản chuẩn cũng chỉ 37,4%), nên mã đúng trên nhãn mà câu hỏi không nhắc bị tính là sai; thước đo nhận nhầm mã là độ chính xác mã so với toàn bộ nhãn trong bộ kiểm tra OCR (93,7% → 94,3%).

#### B. Ứng dụng Đồ thị Bản thể học FoodOn (29.255 thực thể DAG):
- Xây dựng chỉ mục quan hệ đồ thị hai chiều (`self.parents` và `self.children`).
- **Phép suy diễn kế thừa dị nguyên (Hierarchical Allergen Subsumption):**
  - Khi người dùng dị ứng một nhóm dị nguyên gốc (ví dụ: `FOODON:03420115` - Đậu phộng / Peanut), hệ thống duyệt đồ thị con tìm toàn bộ con cháu (Descendants): bơ đậu phộng, dầu đậu phộng ép lạnh, bột lạc...
  - Ngược lại, khi nhãn xuất hiện một thành phần cụ thể, hệ thống truy vết chuỗi phả hệ ngược lên gốc (Ancestors / Lineage) để phát hiện dị nguyên cha.
  - Sử dụng thuật toán tìm tổ tiên chung gần nhất (Lowest Common Ancestor - LCA) để xác định mức độ quan hệ họ hàng giữa các thành phần thực phẩm.

---

### 3.3. TẦNG 3: HYBRID RETRIEVAL & DATA HUB (16 Bảng dữ liệu tra cứu & QA Động)
*Mã nguồn thực thi:* `modules/dynamic_qa.py`, `modules/passage_retriever.py`, `knowledge_base/`, `data/`

#### A. Data Hub 16 Bảng dữ liệu tra cứu:
1. `additives_standard.json`: 50 phụ gia trọng yếu kèm ADI, nhóm quy định Thông tư 24, Codex, chống chỉ định lâm sàng.
2. `additive_registry.json`: 400 phụ gia định danh xác định (tên tiếng Việt, tiếng Anh, bí danh, mã INS/E).
3. `allergen_groups.json`: 21 nhóm dị nguyên theo phân loại quốc tế FALCPA & WHO/JEMRA Part 3.
4. `foodon_entities.json`: 29.255 nút bản thể học thực phẩm quốc tế FoodOn.
5. `ingredients_standard.json`: Danh mục thành phần thực phẩm thông dụng liên kết FoodOn.
6. `fndds_nutrition.csv`: Bảng thành phần dinh dưỡng chuẩn USDA FNDDS (năng lượng, đường, muối, chất béo bão hòa...).
7. `openfoodfacts_vietnam.csv`: Cơ sở dữ liệu sản phẩm đóng gói thực tế trên thị trường Việt Nam.
8. `incidents_train.csv`: CSDL cảnh báo mối nguy & thu hồi thực phẩm (Food Hazard Detector - SemEval 2025).
9–16. **Tập dữ liệu NHANES Hoa Kỳ**: Hồ sơ dịch tễ học và chỉ số cận lâm sàng đại diện dân số:
   - `P_DEMO`: Nhân khẩu học, giới tính, độ tuổi.
   - `P_BMX`: Thể chất (chiều cao, cân nặng, chỉ số BMI).
   - `P_BPXO`: Huyết áp tâm thu/tâm trương.
   - `P_GLU`: Đường huyết lúc đói (Fasting Glucose).
   - `P_GHB`: Chỉ số Glycohemoglobin (HbA1c).
   - `P_BIOPRO`: Sinh hóa máu toàn diện (Creatinine, Acid Uric, Men gan, eGFR).
   - `P_DBQ`: Thói quen dinh dưỡng và hành vi tiêu thụ.
   - `nhanes_personas.csv`: 7 nhóm hồ sơ sức khỏe lâm sàng chuẩn hóa.

#### B. Cơ chế QA Động (Dynamic QA - Hỏi gì tra nấy):
- Không tra cứu thụ động: Hệ thống dùng phân tích cú pháp quy tắc để hiểu ý định câu hỏi:
  - Hỏi về **an toàn phụ gia** $\rightarrow$ Kích hoạt tra cứu ADI JECFA & TT 24.
  - Hỏi về **dị ứng / nhiễm chéo** $\rightarrow$ Kích hoạt rà soát JEMRA Part 3, PAL, FoodOn DAG.
  - Hỏi về **dinh dưỡng / bệnh nền** $\rightarrow$ Kích hoạt so sánh định lượng với FNDDS.
  - **Số liệu dinh dưỡng ghi trên nhãn** (`pipeline.extract_label_nutrition`, cập nhật 16/11/2026): đọc natri (mg) và đường tổng số (g) từ câu hỏi hoặc bảng dinh dưỡng trên nhãn, quy về /100 g (theo 100 g/ml, theo khẩu phần "… trên 1 khẩu phần 25 g" quy đổi ×100/25, hoặc bảng "tính trên 100 ml"). Không đoán khi thiếu con số hay cơ sở tính; cụm nêu NGƯỠNG ("không quá 120 mg/100 g", "natri thấp", "không đường") bị bỏ qua để không hiểu nhầm là số của sản phẩm. Có số trên nhãn thì kiểm tra điều kiện Codex CAC/GL 23-1997: "natri thấp" ≤ 120 mg/100 g (hồ sơ tăng huyết áp/tim mạch), "không đường" ≤ 0,5 g/100 g (hồ sơ đái tháo đường), ghi nguồn `CODEX_CAC_GL_23_1997`. Kiểm tra trên 82 câu Codex của bộ 750 câu: đọc đúng 82/82; tầng luật Cấu hình B đúng 87,8% số câu này.
  - **Ước tính FNDDS không thay nhãn:** khi nhãn không có số liệu, giá trị ước tính từ món tương tự trong FNDDS **không** được dùng để đổi nhãn an toàn và không đưa vào câu trả lời như số của sản phẩm (`NUTRITION_ESTIMATE_CHANGES_LABEL = False`). Trước đây một ước tính sai món ("Muscle Milk") đã làm nhãn sữa trẻ em bị đánh "cẩn trọng".
  - Câu hỏi **ngoài phạm vi** (hỏi đơn thuốc, chẩn đoán bệnh) $\rightarrow$ Lập tức kích hoạt cờ từ chối có nguyên tắc (`out_of_scope`).
- **Minh bạch hóa vết tra cứu tri thức (Traceability & Transparent Wording):**
  - Hệ thống hiển thị bảng tra cứu trực quan: Nguồn tri thức, Thực thể truy vấn, Trạng thái và Chi tiết kết quả.
  - **Diễn đạt khoa học, trấn an người dùng (Cách 1):** Với các phụ gia phổ thông đã định danh hợp lệ và không nằm trong diện cảnh báo độc tính/ngưỡng ML đặc biệt (ví dụ: Lecithin INS 322(i), Caramen nhóm I INS 150a, Tocopherol INS 307c), hệ thống ghi rõ: `Đã định danh hợp lệ (phụ gia thông dụng, không có cảnh báo nguy cơ ADI/ML đặc biệt)`, tránh dùng câu cụt ngủn dễ gây hiểu nhầm hoặc lo lắng không cần thiết.
  - **Chuẩn hóa 100% tiếng Việt có dấu:** Toàn bộ dữ liệu tra cứu trong `knowledge_base/additives_standard.json` (kể cả các mã hỗn hợp điều vị tổ hợp như `INS 621+631+627` MSG + I+G) đều được chuẩn hóa chính tả tiếng Việt có dấu đầy đủ và thống nhất.
- **Truy xuất đoạn bằng chứng (Passage Retrieval):** Xếp hạng các đoạn trích quy định có trọng số, gắn thẻ nguồn duy nhất (`[VN_TT24_2019]`, `[JECFA_EVALUATIONS]`, `[CODEX_STAN_192_1995]`).

#### C. Hệ thống Hồ sơ Sức khỏe Cá nhân hóa & Cơ chế Tích hợp Dịch tễ học NHANES:
*Mã nguồn thực thi:* `modules/persona_builder.py`, `modules/pipeline.py`, `data/nhanes_personas.csv`
Hệ thống chuyển đổi giao diện chính sang mô hình **Tự chỉnh Hồ sơ Sức khỏe Người dùng (Custom Persona Builder)** làm trọng tâm, đặt ngay phía trước ô nhập câu hỏi để thiết lập ngữ cảnh y khoa cá nhân hóa:
1. **Kiến trúc Giao diện Cân đối & Chống lỗi:**
   - **Bố cục 2 hàng đối xứng:** Hàng 1 chia 2 cột đều 50/50 cho **Tuổi** (có viền đồng bộ với giao diện) và **Giới tính**; Hàng 2 chia 2 cột đều 50/50 cho **Thai kỳ** (`🤰 Đang mang thai`) và **Nuôi con nhỏ** (`🤱 Đang cho con bú`).
   - **Đa chọn bệnh nền lâm sàng (Multi-select dropdown):** Cho phép kết hợp đồng thời nhiều bệnh nền phức tạp: Đái tháo đường, Tăng huyết áp, Bệnh thận mạn (CKD), Gout, Viêm loét dạ dày, Rối loạn dung nạp Lactose, PKU, Béo phì, Hen phế quản, Trẻ em ADHD... Hệ thống tự động ánh xạ sang các tập luật ràng buộc y khoa tương ứng.
   - **Chọn nhanh 9 nhóm dị nguyên ưu tiên:** Sữa bò, Trứng, Đậu phộng, Đậu nành, Lúa mì (Gluten), Động vật giáp xác (Tôm, Cua), Cá, Hạt cây (Hạt điều, Óc chó), Mè (Vừng) - khớp mã FoodOn CURIE chuẩn.
   - **Ô nhập tự do dị nguyên riêng (Free-text Allergen Tags) & Phòng vệ 3 lớp chống nhập sai:**
     - *Lớp 1 (Từ phủ định):* Nhận diện "không", "không có", "chưa", "bình thường", "none" $\rightarrow$ Tự động bỏ qua không báo lỗi sai, giữ an toàn hệ thống.
     - *Lớp 2 (Triệu chứng cơ thể):* Bắt các từ phản ứng lâm sàng ("đau bụng", "ngứa ngáy", "nôn", "khó thở", "sốc phản vệ") $\rightarrow$ Cảnh báo nhẹ nhàng rằng ô này dành cho tên thực phẩm gây dị ứng, không phải triệu chứng.
     - *Lớp 3 (Thuốc & Dược phẩm):* Bắt tên thuốc ("paracetamol", "panadol", "aspirin", "amoxicillin", "kháng sinh") $\rightarrow$ Cảnh báo hệ thống chỉ phân tích thành phần thực phẩm, hướng dẫn tư vấn bác sĩ chuyên khoa nếu dị ứng thuốc.
   - **Tận dụng Tầng 2 (FoodOntoRAG) chuẩn hóa thực thể:**
     - Tên dị ứng người dùng gõ được đối chiếu ngay với Đồ thị Bản thể học FoodOn (29.255 nút) để tìm mã `CURIE` và cây phả hệ (Lineage / Ancestors / Descendants).
     - Giúp hệ thống tự động suy diễn quan hệ bắc cầu: Người dùng khai "dị ứng hạt óc chó" $\rightarrow$ Hệ thống tự động cảnh báo mọi sản phẩm chứa dầu óc chó, bơ hạt cây hoặc chiết xuất óc chó.
2. **Cơ chế Làm giàu Dữ liệu Tự động qua NHANES (`enrich_persona_with_nhanes`):**
   - Không bị giới hạn trong 7 hồ sơ mẫu tĩnh: Hàm `enrich_persona_with_nhanes()` trực tiếp phân tích danh sách bệnh lý (`conditions`) và nhân khẩu học (`demographics`: tuổi, giới tính, thai kỳ) từ hồ sơ tự chỉnh của người dùng.
   - Tự động so khớp với hàng nghìn đối tượng khảo sát dịch tễ học thực nghiệm trong các tệp NHANES:
     - `P_DEMO`: Lọc nhóm tuổi, giới tính, phụ nữ mang thai thực tế.
     - `P_GHB`: Đối chiếu chỉ số HbA1c ($\ge 6.5\%$ với Đái tháo đường).
     - `P_BPXO`: Đối chiếu Huyết áp tâm thu/tâm trương ($\ge 140/90\text{ mmHg}$ với Tăng huyết áp).
     - `P_BIOPRO`: Đối chiếu nồng độ BUN ($> 7.1\text{ mmol/L}$) và Creatinine với Bệnh thận mạn (CKD).
     - `P_DBQ`: Bổ sung dữ liệu thói quen tiêu thụ (tần suất ăn đồ ăn nhanh, dùng bữa sẵn đóng gói, thói quen đọc nhãn dinh dưỡng).
   - Nhờ đó, bất kỳ hồ sơ tự tạo nào của người dùng cũng được tiếp sức bởi dữ liệu sinh hóa thực tế từ NHANES để đánh giá nguy cơ dinh dưỡng định lượng.
   - **Giới hạn (cập nhật 16/11/2026):** NHANES chỉ là **tham chiếu** (`persona["nhanes_reference"]`), **không** thêm bệnh nền hay chế độ ăn vào hồ sơ người dùng. Trước đây đối tượng NHANES khớp tuổi có thể kéo theo "bệnh thận mạn" vào hồ sơ một em bé 1 tuổi không khai bệnh, làm kết luận "cẩn trọng" sai.

---

### 3.4. TẦNG 4: GUARDRAIL AN TOÀN Y TẾ & GROUNDED GENERATION
*Mã nguồn thực thi:* `modules/constraint_model.py`, `modules/abstention.py`, `modules/pipeline.py`

#### A. Khung toán học Mô hình hóa Ràng buộc Phủ định (Negation Constraint Modeling - Kế thừa WSDM '21):
Áp dụng nền tảng toán học từ bài báo WSDM '21 (*Personalized Food Recommendation as Constrained Question Answering*):
1. **Hàm chỉ thị điều kiện sức khỏe $I_g(v)$:**
   $$I_g(v) = \begin{cases} 1 & \text{nếu thành phần/dinh dưỡng } v \text{ thỏa mãn điều kiện y tế } g \\ 0 & \text{nếu vi phạm chống chỉ định y tế } g \end{cases}$$
2. **Không gian gán nhãn ràng buộc $M = \{+1, -1, 0\}$:**
   - $+1$ (`PERMITTED`): Thành phần an toàn, phù hợp với hồ sơ.
   - $-1$ (`CONTRAINDICATED`): Vi phạm chống chỉ định y khoa (ví dụ: phenylalanin với bệnh nhân PKU; muối photphat với suy thận mạn CKD3).
   - $0$ (`RESTRICTED` / `CAUTION`): Cần thận trọng về lượng ăn, vượt ngưỡng khuyến nghị (ví dụ: đường đối với tiểu đường, muối với tăng huyết áp).
3. **Mô hình xếp hạng liên hợp (Joint Preference & Safety Ranking):**
   $$S = (1 - \lambda) S_{\text{eqa}} + \lambda S_{\text{esim}}$$
   Trong đó $S_{\text{eqa}}$ là điểm khớp chính xác quy chuẩn an toàn, $S_{\text{esim}}$ là độ tương đồng ngữ cảnh, $\lambda \in [0, 1]$ là trọng số cân bằng.

#### B. Quy chuẩn Nhãn An toàn Chuẩn hóa & Hệ thống Phán quyết Đa cấp:
Hệ thống phân định rạch ròi giữa **Trạng thái Kỹ thuật Nội bộ của Máy luật** và **Không gian Nhãn Chuẩn hóa 4 Cấp độ**:
1. **Trạng thái Kỹ thuật Nội bộ của Guardrail:**
   - `SCREENED_NO_ALERT`: *"⚪ CHƯA PHÁT HIỆN CẢNH BÁO TRONG DỮ LIỆU ĐÃ SÀNG LỌC"* — Đã sàng lọc toàn bộ thành phần đọc được qua 16 bảng tra cứu mà không phát hiện nguy cơ. Tránh cam đoan tuyệt đối nếu bao bì còn phần bị mờ.
   - `CAUTION`: Thành phần vượt ngưỡng khuyến nghị dinh dưỡng hoặc phụ gia có mức ADI chặt chẽ.
   - `WARNING`: Có thành phần chống chỉ định với bệnh nền hồ sơ hoặc vi phạm quy định giới hạn ML.
   - `CRITICAL_ALERT`: Dị ứng cấp tính đe dọa tính mạng (theo FoodOn/JEMRA) hoặc phụ gia bị thu hồi khẩn cấp. Mức cảnh báo này không bao giờ bị hạ cấp.
   - `ABSTAIN`: Từ chối kết luận do điểm tin cậy tổng dưới ngưỡng (`CONFIDENCE_ABSTAIN_THRESHOLD`, mục C), thiếu dữ liệu trọng yếu trên nhãn hoặc câu hỏi ngoài phạm vi y khoa.
2. **Không gian Nhãn Chuẩn hóa 5 nhãn** (khớp `guardrail_status_to_safety_label` và nhãn chuẩn của bộ QA):
   - 🟢 **`SAFE`**: ánh xạ từ `SCREENED_NO_ALERT` — nghĩa là **chưa thấy cảnh báo trong dữ liệu đã sàng lọc**, không phải cam đoan an toàn. Khi chấm điểm, số câu SAFE thuộc loại này được báo riêng (`safe_from_screened_no_alert`).
   - 🟡 **`CAUTION`**: ánh xạ từ `CAUTION`.
   - 🟠 **`UNSAFE`**: ánh xạ từ `WARNING`.
   - 🔴 **`CRITICAL_ALERT`**: giữ nguyên `CRITICAL_ALERT` (cần xác minh với chuyên gia).
   - ⚪ **`ABSTAIN`**: ánh xạ từ `ABSTAIN`.

#### C. Điểm tin cậy kép & Cơ chế từ chối (Confidence Scoring & Abstention):
- $S_{\text{perception}}$: xác suất ký tự trung bình của VietOCR (mục 3.1); B = 1,0.
- $S_{\text{retrieval}}$ = **tỉ lệ cụm phụ gia trên nhãn liên kết được với cơ sở tri thức** bằng khớp định danh (mã INS/E, tên chuẩn, tên đồng nghĩa); mã OCR không đọc chắc tính là không liên kết được. Nhãn **không có cụm phụ gia nào** → $S_{\text{retrieval}}$ = **không áp dụng**. (Trước đây dùng điểm BM25/ngữ nghĩa chia cho điểm cao nhất của lượt tìm — ứng viên đầu luôn ≈1 — và gán 1,0 cho nhãn không có phụ gia.) Hạn chế: chỉ số chỉ thấy các cụm OCR còn đọc ra; mã OCR làm mất hẳn không được tính — phần này đo khi chấm bằng `scripts_research/analyze_normalization_errors.py`.
- Điểm tin cậy tổng thể:
  $$S_{\text{total}} = \begin{cases} 0.5 \cdot S_{\text{perception}} + 0.5 \cdot S_{\text{retrieval}} & \text{nếu có cụm phụ gia} \\ S_{\text{perception}} & \text{nếu không} \end{cases}$$
- Nếu $S_{\text{total}} <$ `CONFIDENCE_ABSTAIN_THRESHOLD` (hiện 0,65, `modules/pipeline.py`) **và nhãn không có cảnh báo nào** $\rightarrow$ `ABSTAIN` loại `insufficient_data`. Ngưỡng phải được chọn trên phần "phát triển" của bộ đánh giá (chia theo ảnh) bằng `scripts_research/calibrate_confidence.py` và báo cáo trên phần "kiểm tra".
- Thêm một luật: hơn 50% cụm phụ gia không liên kết được $\rightarrow$ `ABSTAIN` (ngưỡng 50% là lựa chọn thiết kế, được báo cáo riêng khi hiệu chỉnh).
- **Kết quả hiệu chỉnh ngưỡng** (`calibrate_confidence.py`, tầng luật, không gọi LLM): ở Cấu hình B, ngưỡng 0,00–0,65 cho kết quả như nhau; ở Cấu hình C, không dùng ngưỡng cho độ chính xác cao nhất trên phần phát triển (58,7% so với 58,1% ở 0,65), còn ngưỡng 0,65 giảm tỉ lệ trả lời "an toàn" sai 23,0% → 21,3%. Ngưỡng 0,65 được **giữ vì lý do thận trọng** (câu hỏi sức khỏe), không phải vì tối ưu độ chính xác. $S_{\text{perception}}$ dự báo lỗi OCR ở mức yếu (Spearman −0,26 với tỉ lệ thiếu từ).
- **Sửa lỗi (16/11/2026)** trong `modules/abstention.py`: biểu thức nhận "INS 211"/"E211" trong khoảng trống thông tin `ADDITIVE_IDENTITY` bị lưu thành ký tự backspace thay cho `\b`, nên hai nhánh này không bao giờ khớp — cụm có ghi mã nhưng chưa liên kết được vẫn bị coi là "nhãn không khai báo mã INS".
- Nếu câu hỏi yêu cầu chẩn đoán/kê đơn $\rightarrow$ Kích hoạt `ABSTAIN` loại `out_of_scope` (hướng dẫn người dùng đến gặp bác sĩ chuyên khoa).
- **Luật phụ gia cho trẻ dưới 12 tháng tuổi (16/11/2026)** — `modules/infant_rules.py`.
  - *Căn cứ, trích tự động từ văn bản gốc đã lưu* (`scripts_research/extract_infant_additive_rules.py` → `knowledge_base/sources/derived/infant_food_additives.json`; chạy bằng python có PyMuPDF):
    - TT 24/2019/TT-BYT, bản hợp nhất 09/VBHN-BYT 2024, Phụ lục 2A: các dòng nhóm 13.1.1 (công thức cho trẻ đến 12 tháng), 13.1.2 (công thức ăn bổ sung 6–36 tháng), 13.1.3 (công thức y tế đặc biệt đến 12 tháng), 13.2 (thực phẩm bổ sung cho trẻ đến 36 tháng). Trích được lần lượt 26/28/27/56 dòng, đúng bằng số lần mã nhóm xuất hiện trong Phụ lục 2A. Mỗi dòng có INS, tên, ML, ghi chú và trang.
    - TT24 Phụ lục 3 mục 2 ("Đối tượng thực phẩm sử dụng theo GMP"): **không** có nhóm 13.1, 13.2.
    - Codex GSFA CXS 192-1995: Bảng Hai nhóm 13.1.1/13.1.2/13.1.3/13.2 (28/34/30/57 dòng, PDF tr. 425–428). Annex to Table Three (tr. 493) loại 13.1, 13.2 khỏi quy tắc GMP chung; Preamble §3.3 (tr. 5) không chấp nhận chất mang phụ gia cho hai nhóm này trừ khi Bảng 1–2 ghi riêng.
    - → Với thực phẩm cho trẻ nhỏ, chỉ phụ gia được ghi riêng cho nhóm này mới được dùng.
  - *Khi nào áp dụng:* hồ sơ có số tháng tuổi < 12 (chat lưu `age_months` khi người dùng khai "8 tháng tuổi"; trước đây bị làm tròn thành 1 tuổi), hoặc câu hỏi nhắc "sơ sinh", "nhũ nhi", "dưới 12 tháng", "N tháng tuổi" với N < 12. Tên sản phẩm trong dấu nháy được bỏ qua. Số tháng ghi cùng câu "đang cho con bú" được hiểu là tuổi của con.
  - *Luật:* mỗi phụ gia có mã INS trên nhãn được đối chiếu với hợp của bốn nhóm theo TT24 và theo Codex.
    - Không có trong cả hai danh mục → cảnh báo **CAUTION** "Có phụ gia không thuộc danh mục cho trẻ dưới 12 tháng", kèm căn cứ.
    - Có trong danh mục → ghi mức tối đa và vị trí (nhãn không ghi hàm lượng nên không so được với ML).
    - Nhãn không có phụ gia mang mã → không từ chối, chỉ có dòng lưu ý.
  - *Dòng lưu ý luôn hiện:* luật chỉ đối chiếu phụ gia, không đánh giá sản phẩm có phù hợp dinh dưỡng cho trẻ hay không (vd sữa tươi, thực phẩm người lớn) — nên hỏi bác sĩ nhi.
  - *Đo:* bộ QA 750 câu không có câu nào cho trẻ dưới 12 tháng (0/750 kích hoạt), nên nhãn và số đo không đổi (từ chối đúng 69/72, từ chối nhầm 18/678); **chưa có số đo lợi ích**. Dòng lưu ý người cao tuổi hiện ở 81 câu của hồ sơ `ckd_stage3` (62 tuổi). Dòng này không đổi nhãn nhưng được đưa vào lời dặn LLM ở Cấu hình B/C.
  - *Sửa 3 lỗi (16/11/2026)*, phát hiện khi thử chat cho trẻ dưới 12 tháng trên ảnh 0600–0611 và ảnh cũ (không lập bộ QA riêng, nên **không có số độ chính xác cho nhóm này**; chỉ có test và số đo trên bộ 750 câu):
    1. *Câu giới hạn tuổi trên nhãn* (`infant_rules.label_age_alerts`): đọc "không dùng cho trẻ dưới N tuổi/tháng", "dùng/phù hợp cho trẻ từ/trên N…", "N+ tháng tuổi"; bé nhỏ hơn tuổi tối thiểu → **UNSAFE** ("Nhãn ghi không dùng cho độ tuổi của bé"), căn cứ Codex CXS 74-1981 mục 8.6.4. Không đổi thành ABSTAIN khi nhãn còn thiếu dữ liệu khác (giống dị nguyên trực tiếp). Chỉ biết "dưới 12 tháng" (vd "sơ sinh") thì chỉ chặn khi nhãn ghi từ 12 tháng. Câu không có số ("không thích hợp cho trẻ sơ sinh…") không suy tuổi. Chỉ chạy cho trẻ dưới 12 tháng.
    2. *Nguồn khoáng/vitamin không phải phụ gia* (`infant_rules.is_nutrient_source`): dòng thành phần ghi "nguồn bổ sung khoáng", "khoáng chất", "(Vitamin …)" và không có nhóm phụ gia trong cùng dòng thì không đối chiếu với danh mục phụ gia (QCVN 11-1:2012/BYT mục 1.3.4; CXS 72-1981 mục A3.4). Trước đây kali clorid (508), beta-caroten (160a) trong sữa công thức Similac/Nutramigen bị CAUTION nhầm.
    3. *Từ dị nguyên trong câu hỏi* (`pipeline._question_allergies`): chỉ nhận khi đứng ngay sau dấu hiệu dị ứng ("dị ứng", "mẫn cảm", "không dung nạp", "tránh", "kiêng", "nhiễm chéo", "chứa"), trong cùng cụm. Trước đây mọi từ đều được nhận: "bé ăn sữa chua 'X'" → dị ứng sữa; "đạt **điều** kiện 'natri thấp'" → từ khóa "điều" (hạt điều) → dị ứng hạt; "sulphite" trong câu hỏi về khai báo → dị ứng sulphite.
    - Thêm: chat và luật nhận "bé (nhà tôi) 8 tháng" không cần chữ "tuổi".
    - **Đo trên bộ 750 câu, tầng luật đầy đủ (`calibrate_confidence.py --stage rules`, ngưỡng 0,65), trước → sau:** Cấu hình B **520 → 530/750**, Cấu hình C **437 → 449/750**. Không câu nào đang đúng bị sai. Đổi kết quả: B 13 câu (10 sai → đúng: 0349, 0531, 1291, 1469, 1684, 2424, 2433, 2434, 2447, 2459; 3 vẫn sai UNSAFE → ABSTAIN: 2184, 2185, 2192 hỏi khai báo sulphite, đáp án CAUTION); C 16 câu (12 sai → đúng, 4 vẫn sai: ba câu sulphite và 0108). Lỗi 1 và 2 không kích hoạt trên bộ 750 câu. `measure_abstention.py`: từ chối đúng giữ 69/72, từ chối nhầm 18 → 20/678 (hai câu sulphite). 364 test qua.
  - *Nguồn bổ sung cho trẻ nhỏ (16/11/2026)* — người dùng tải tay, ghi SHA-256 vào manifest, 12 khóa trích dẫn mới; vị trí đã đối chiếu với chữ trích từ tệp: WHO 2023 *complementary feeding* (Khuyến nghị 2 tr. 12: sữa có đường/hương không phù hợp cho trẻ 6–23 tháng; Khuyến nghị 5 tr. 14: không dùng chất tạo ngọt), Codex CXS 74-1981 mục 8.6.4 (nhãn phải ghi tuổi bắt đầu dùng, không dưới 6 tháng), CXS 72-1981 mục A3.4, QCVN 11-1..11-4:2012 (mục phụ gia đều dẫn về quy định Bộ Y tế, tức TT24). NĐ 100/2014 là bản quét không có lớp chữ nên chưa dùng. Ghi bằng `scripts_research/register_infant_sources.py`; luật hiện chưa dùng các nguồn này.
  - *Đã gỡ các luật chỉnh theo lỗi của bộ QA (17/11/2026).* Có một đợt viết luật sau khi xem danh sách câu sai trên bộ 750 câu (luật cơ quan chuyên môn theo chủ đề, luật dị nguyên không kê khai, nhiễm chéo → CRITICAL_ALERT, PKU + aspartame, nhận câu nhiễm chéo "sản xuất trên thiết bị…", số nhiều tiếng Anh, "bơ ca cao"). Vì bộ QA chỉ dùng để kiểm thử/đánh giá, không dùng để chỉnh hệ thống (tránh thiên lệch), **toàn bộ code đó đã bị xóa** (`authority_rules.py`, `qa_tuning.py` và các nhánh trong `pipeline.py`, `ocr_extractor.py`, `constraint_model.py`, `abstention.py`, giao diện). Các số 530 → 699 (B) và 449 → 580 (C) từng ghi ở đây là số **chỉnh theo tập kiểm thử**, không còn là kết quả của hệ thống. **Số chính thức** là trạng thái trước đợt chỉnh đó: sau sửa 3 lỗi trẻ nhỏ ở trên, B 530/750 và C 449/750 (lưu ở `data/runs/cal_official_base_16_11`); kiểm tra lại sau khi gỡ trên 30 câu đầu cho kết quả giống hệt (0 câu đổi). Nếu cần cải tiến (cấu hình D), phải rút luật từ một tập phát triển tách khỏi 750 câu rồi mới đo một lần trên 750 câu.
- **Các loại thiếu dữ liệu dù OCR đọc đúng — đã áp dụng đến đâu (16/11/2026).** Mọi loại thiếu dữ liệu đều trả lời theo đúng hai hướng: *thiếu dữ liệu* thì đề nghị chụp lại hoặc bổ sung thông tin; *vượt phạm vi* thì đề nghị trao đổi với chuyên gia phù hợp.

  | Loại thiếu | Luật | Trạng thái |
  |---|---|---|
  | Hàm lượng / lượng tiêu thụ | câu hỏi về liều/ADI mà nhãn chỉ có mã, không có hàm lượng | có |
  | Dị nguyên / nhiễm chéo | `CROSS_CONTACT_STATEMENT`: hỏi nhiễm chéo mà nhãn không có câu "có thể chứa" nêu dị nguyên đó | có |
  | Bằng chứng cho nhóm người dùng | Trẻ dưới 12 tháng: **luật riêng** `modules/infant_rules.py` (xem dưới). Người cao tuổi, phụ nữ cho con bú: KB không có quy định riêng → **chỉ thêm dòng lưu ý** (`abstention.population_notes`), kết luận theo luật người lớn và bệnh nền đã khai, không từ chối | có |
  | Thực thể trong KB | hơn 50% phụ gia không liên kết được; `ADDITIVE_NOT_IN_KB`; `ADDITIVE_IDENTITY` | có |
  | Nguồn đủ rõ | chỉ có cho mã không có nguồn quy chuẩn (`ADDITIVE_NOT_IN_KB`). Khi không có đoạn bằng chứng phù hợp, hệ thống chỉ ghi lý do, không từ chối | một phần |

  **Đo trước khi giữ** (tầng luật, cấu hình B, bộ 750 câu; bộ QA chỉ có 72 câu ABSTAIN, đều thuộc loại "nhãn thiếu thông tin" hoặc "không có câu 'có thể chứa'"):
  - Bản đầu (16/11/2026) từ chối cả ba nhóm (`POPULATION_EVIDENCE`, 0/750 câu) đã được **thay** bằng luật trẻ dưới 12 tháng và dòng lưu ý, vì từ chối người cao tuổi/cho con bú làm mất cả cảnh báo đang có (vd natri với người cao tuổi tăng huyết áp).
  - Ba phương án cho "nguồn đủ rõ" đều bị **loại**:
    - "không truy xuất được đoạn nào": 0 lần, vì bộ truy xuất luôn trả về top-k;
    - "không đoạn nào nhắc đúng mã được hỏi": 139 câu, chỉ 5 câu có đáp án ABSTAIN, làm sai 76 câu đang đúng là SAFE;
    - "hỏi dinh dưỡng mà không có bảng dinh dưỡng của sản phẩm": 150 câu, không câu nào có đáp án ABSTAIN.
  - Phương án coi phần đối chiếu Codex còn "pending" (đúng với cả 50 phụ gia trong KB) là "nguồn chưa rõ" cũng bị loại, vì sẽ từ chối 88 câu hỏi về mức tối đa/danh mục mà đáp án chuẩn coi là trả lời được.

#### D. Sinh lời khuyên & Rào chắn hậu kiểm (Generation & Guardrail Enforcement):
- Prompt ép buộc cấu trúc: Đưa vào toàn bộ bằng chứng tra cứu được, cấm LLM tự bịa kiến thức ngoài.
- Dòng cuối cùng của prompt bắt buộc LLM xuất ra: `SAFETY_LABEL: {canonical_safety_status}` (lấy đúng nhãn từ máy luật).
- **Bộ lọc hậu kiểm & Bằng chứng minh bạch (`modules/evidence.py`):**
  - Rà soát từng mã trích dẫn trong câu trả lời; loại bỏ mọi trích dẫn "ma" không nằm trong tài liệu tra cứu.
  - Chuyển đổi mã nguồn kỹ thuật `[MÃ_NGUỒN]` thành các số trích dẫn tương tác `[1]`, `[2]`... liên kết trực tiếp tới thẻ tài liệu gốc ở danh sách nguồn.
  - **Cơ chế Fallback Passage đảm bảo tính nguyên bản:** Với các cơ sở dữ liệu quốc tế như EFSA Journal (Ý kiến khoa học), FAO/WHO JEMRA hay USDA FoodData Central, hệ thống tự động đối chiếu đoạn trích thẩm định nguyên bản hoặc phạm vi phê duyệt, triệt tiêu hoàn toàn lỗi thiếu trích dẫn gốc.
  - **Ẩn dòng kỹ thuật nội bộ:** Dòng `SAFETY_LABEL: ...` do LLM sinh ra được tách lọc tự động khỏi văn bản hiển thị và chuyển hóa thành Thẻ tiêu đề đồ họa nổi bật ở đầu trang để bảo đảm tính thẩm mỹ giao diện.
  - Đảm bảo nhãn an toàn cuối cùng khớp 100% với phán quyết của máy luật y tế.

---

## 4. BA CẤU HÌNH THỰC NGHIỆM ĐỐI CHỨNG (ABLATION STUDY)

Hệ thống thiết kế 3 cấu hình chạy song song để cô lập và định lượng chính xác giá trị của từng thành phần:

| Tiêu chí | Cấu hình A (No-RAG Baseline) | Cấu hình B (Gold Text + RAG) | Cấu hình C (OCR + RAG - Thực tế) |
| :--- | :--- | :--- | :--- |
| **Đầu vào thành phần** | Văn bản nhãn tham chiếu | Văn bản nhãn tham chiếu | Ảnh bao bì thực phẩm |
| **Thị giác (OCR)** | Bỏ qua | Bỏ qua | **PaddleOCR + VietOCR + Sửa lỗi** |
| **Chuẩn hóa thực thể** | Bỏ qua | **FoodOntoRAG 4-Agent + FoodOn** | **FoodOntoRAG 4-Agent + FoodOn** |
| **Tra cứu tri thức (RAG)** | Không (LLM tự nhớ) | **Có (16 bảng Data Hub + QA Động)**| **Có (16 bảng Data Hub + QA Động)**|
| **Chốt chặn y tế** | Không (LLM tự quyết) | **Rule Engine Guardrail** | **Rule Engine Guardrail** |
| **Nhãn an toàn do** | LLM tự gán (dễ ảo giác) | Máy luật quyết định | Máy luật quyết định |
| **Mục đích khoa học** | Đo mức độ rủi ro & ảo giác | Đo lường giá trị tối đa của RAG | Đo ảnh hưởng của sai số OCR thực tế |

*Lưu ý khoa học:* Cấu hình B và C hoàn toàn chia sẻ chung mã nguồn từ Tầng 2 trở đi (`run_grounded_pipeline`), đảm bảo tính công bằng và nhất quán tuyệt đối trong các bài báo cáo khoa học.

---

## 5. BỘ DỮ LIỆU ĐÁNH GIÁ BENCHMARK (750 BẢN GHI ĐÁNH GIÁ CHUẨN)
*Mã nguồn thực thi:* `modules/benchmark_manifest.py`, `data/foodhealthqa_eval_set.jsonl`

Hệ thống được kiểm chứng trên tập benchmark **750 câu hỏi chuẩn hóa**:
- Gắn với **376 ảnh bao bì sản phẩm độc lập** thu thập tại thị trường Việt Nam (không quá 4 câu hỏi/sản phẩm để tránh lệch mẫu).
- **7 nhóm Persona lâm sàng:** Người khỏe mạnh, Trẻ em < 6 tuổi, Thai phụ, Người cao tuổi tăng huyết áp, Bệnh nhân đái tháo đường Type 2, Bệnh nhân suy thận mạn tính (CKD giai đoạn 3b), Bệnh nhân Phenylketonuria (PKU).
- **6 trụ cột chuyên môn sâu rộng:**
  1. *An toàn phụ gia (166 câu):* TT 24/2019/TT-BYT, Codex, JECFA ADI, phẩm màu Azo, chất bảo quản.
  2. *Dị nguyên & Dị ứng (165 câu):* 9 nhóm dị nguyên ưu tiên, cảnh báo tiếp xúc chéo PAL, FoodOn hierarchy.
  3. *Trẻ nhỏ (120 câu):* Phẩm màu hiếu động Southampton, chất tạo ngọt không đường NSS, caffeine.
  4. *Bệnh nền mạn tính (112 câu):* Muối vô cơ photphat/kali trong suy thận mạn, aspartam trong PKU.
  5. *Thai kỳ & Phụ nữ mang thai (105 câu):* Phụ gia nhạy cảm tích lũy (TiO2 INS 171, nitrit INS 250).
  6. *Dinh dưỡng định lượng (82 câu):* Ngưỡng natri thấp ($\le 120\text{ mg}/100\text{ g}$), không đường ($\le 0.5\text{ g}/100\text{ g}$).

---

## 6. GIAO DIỆN STREAMLIT & CƠ CHẾ VẬN HÀNH (`app.py`)

> **Cập nhật 16/11/2026 — câu trả lời dạng gọn:** mọi câu trả lời trong chat (kể cả câu mới nhất) mặc định chỉ hiện: một dòng sản phẩm + hồ sơ (không lặp lại câu hỏi), kết luận một dòng có màu, một dòng lý do khi chưa đủ bằng chứng, câu trả lời, và một dòng "📜 Nguồn: [1] … · [2] …" (bấm để mở danh sách nguồn đầy đủ). Nút "🔎 Xem chi tiết phân tích" mở 4 tab; tab đầu "📊 Đánh giá & chỉ số" chứa câu hỏi, nhãn kỹ thuật 5 mức, lĩnh vực, ô đối chiếu từ khóa (trùng từ với bằng chứng, không phải kiểm tra ngữ nghĩa), cảnh báo/sự cố và 3 chỉ số tin cậy. Lời dặn độ dài câu trả lời `pipeline.ANSWER_LENGTH_RULE` (1 câu kết luận + tối đa 4 ý, ~120–150 từ) dùng **chung** cho Cấu hình A và B/C để phép so sánh trong bộ QA không lệch.
>
> **Cập nhật 16/11/2026 — nhãn của bộ luật viết để đọc là hiểu** (`app.verdict_view`, chỉ đổi chữ hiển thị; trạng thái và nhãn chấm bộ QA giữ nguyên). Câu kết luận của LLM vẫn là phần chính; phía trên là một khung ngắn gồm **tên nhãn + ý nghĩa cụ thể** ghép từ đúng dữ liệu đã kích hoạt luật:
>
> | Trạng thái → nhãn QA | Hiển thị (ví dụ) |
> |---|---|
> | CRITICAL_ALERT → CRITICAL_ALERT | 🔴 **Chống chỉ định với bệnh của bạn** — Nhãn có Aspartame (INS 951), được quy chuẩn ghi là chống chỉ định với bệnh bạn đã khai. |
> | WARNING → UNSAFE (dị nguyên trực tiếp) | 🟠 **Có dị nguyên bạn bị dị ứng** — Nhãn có sữa, trùng với dị ứng bạn đã khai. |
> | WARNING → UNSAFE (chỉ "có thể chứa") | 🟠 **Có thể nhiễm chéo dị nguyên** — Nhãn ghi "có thể chứa" đậu phộng… (không có trong thành phần, nhưng có thể lẫn trong quá trình sản xuất). |
> | CAUTION → CAUTION | 🟡 **Nên hạn chế** — Có Natri benzoat (INS 211) cần hạn chế với hồ sơ của bạn. / 🟡 **Vượt ngưỡng dinh dưỡng** — (số trên nhãn so với ngưỡng Codex). |
> | ABSTAIN → ABSTAIN (thiếu dữ liệu) | ⚪ **Chưa đủ thông tin để kết luận** — Nhãn hoặc dữ liệu chưa đủ… Đề nghị chụp lại nhãn rõ hơn hoặc bổ sung thông tin (ghi cụ thể phần cần chụp nếu luật thiếu dữ liệu nêu được). |
> | ABSTAIN → ABSTAIN (vượt phạm vi) | ⚪ **Ngoài phạm vi của hệ thống** — Hệ thống chỉ sàng lọc nhãn thực phẩm… Đề nghị trao đổi với bác sĩ, dược sĩ hoặc chuyên gia dinh dưỡng phù hợp. (Có cả hai loại → "Thiếu thông tin và ngoài phạm vi", ghi cả hai đề nghị.) |
> | SCREENED_NO_ALERT → SAFE | 🟢 **Không thấy điểm cần lưu ý** — Trong phần chữ đọc được từ nhãn, không có thành phần nào khớp với dị ứng hay bệnh bạn đã khai. Đây là kết quả sàng lọc, chưa phải xác nhận an toàn. |
>
> Tên dị nguyên lấy theo dị ứng người dùng đã khai (không lấy cả cụm chữ OCR). Khung không thêm dòng so sánh với câu trả lời của LLM. Ở phần chi tiết, dòng kỹ thuật ghi "Nhãn dùng khi chấm bộ QA: … — (ý nghĩa)", kèm nhãn LLM tự ghi và trạng thái nội bộ. Tiêu đề cũ `evaluation["title"]` vẫn giữ trong dữ liệu cho chế độ so sánh và báo cáo.
>
> **Cập nhật 16/11/2026 — sửa thông tin nhãn ngay trong chat** (`chat_intake.detect_label_correction`, `apply_label_correction`): giao diện không hiện danh sách thành phần và không bắt người dùng xác nhận; người dùng chỉ thấy câu trả lời. Nếu câu trả lời nói sai về nhãn, người dùng gõ tiếp như nói chuyện: "không có E621", "hệ thống đọc sai E621", "sản phẩm không có sữa và đậu nành" (bỏ), "mã là E330 chứ không phải E33" / "sửa E33 thành E330" (đổi mã), "Thành phần đúng là: bột mì, đường, muối" (thay danh sách). Hệ thống áp lên thành phần đã đọc, tính lại liên kết phụ gia (không OCR lại) và trả lời lại câu hỏi gần nhất; bản sửa được giữ cho các câu hỏi sau về cùng ảnh. Câu hỏi ("không có sữa phải không?", "E621 là gì?") và câu khai báo bản thân ("tôi không có dị ứng") không bị hiểu là câu sửa. Mục không có trong phần đọc được thì hệ thống nói rõ "không có gì để sửa".

Giao diện được thiết kế hiện đại, tinh gọn theo phong cách dashboard y tế cao cấp với các tiêu chuẩn công thái học (Ergonomics) và chuẩn tiếp cận WCAG AAA:
1. **Luồng Trải nghiệm Người dùng Tự nhiên & Trực quan (Clinical Workflow):**
   - **Hồ sơ sức khỏe đặt trước câu hỏi:** Người dùng thiết lập bối cảnh lâm sàng (Tuổi, Giới tính, Thai kỳ, Bệnh nền, Dị ứng) trước khi đặt câu hỏi, giúp hệ thống cá nhân hóa ngay từ khâu tiếp nhận thông tin.
   - **Bố cục cân đối 50/50:** Hàng 1 chia đôi cân đối cho Tuổi và Giới tính (có viền đồng bộ); Hàng 2 bố trí 2 ô tích chọn ngang cho Mang thai và Nuôi con nhỏ.
   - **Độ tương phản cao (High Contrast Typography):** Toàn bộ các dòng chú thích, hướng dẫn và metadata được nâng cấp lên màu Slate 700 (`#334155`, độ đậm 500), bảo đảm khả năng đọc rõ ràng trên mọi màn hình và điều kiện ánh sáng.
   - **Danh sách cảnh báo sự cố nổi bật:** Các sự cố thực phẩm tương tự sản phẩm được định dạng Markdown rõ nét (`- ⚠️ [CATEGORY] ...`) thay vì chữ mờ.
   - **Cấu trúc Kết quả 4 Tab Chuyên biệt (Tránh cuộn trang dài):** Thay vì dồn toàn bộ ảnh, bảng thành phần và văn bản kéo dài hàng nghìn pixel, cột kết quả được tổ chức thành 4 Tab khoa học:
     - *Tab 1: 💡 Kết quả & Dẫn chứng:* Trình bày ngay lập tức phán quyết an toàn (Banner), Đánh giá an toàn tự động, Câu trả lời có số trích dẫn `[1], [2]...`, Cảnh báo sức khỏe trọng tâm, 3 thẻ chỉ số tin cậy và Danh sách nguồn chính thức. Người dùng xem được ngay kết luận mà không cần cuộn trang.
     - *Tab 2: 🏷️ Bóc tách & Chuẩn hóa nhãn:* Ảnh nhãn bounding box OCR, Văn bản đọc được từ bao bì (kèm công cụ so sánh phục hồi chính tả) và Bảng thành phần chuẩn hóa INS/FoodOn.
     - *Tab 3: 📚 Tra cứu Tri thức & Quy chuẩn:* Vết tra cứu QA động (Dynamic QA Trace), Bảng quy chuẩn TT24/Codex/JECFA, Ý kiến khoa học EFSA, Ràng buộc dinh dưỡng bệnh lý (WSDM '21) và Sự cố thu hồi thực phẩm liên quan.
     - *Tab 4: 🔬 Chi tiết Kỹ thuật:* Bảng 4-Agent FoodOntoRAG đầy đủ (FoodOn CURIE, phả hệ DAG lineage, điểm $\tau$), Từng dòng OCR kèm độ tin cậy và Danh sách đoạn trích bằng chứng thô trong CSDL.
   - **Đánh giá an toàn tự động 5 nhãn:** Hiển thị nhãn chuẩn mực (`SAFE`, `CAUTION`, `UNSAFE`, `CRITICAL_ALERT`, `ABSTAIN`) kèm giải thích minh bạch, phân định rõ ràng giữa suy diễn động cho câu hỏi tự nhập và bộ dữ liệu kiểm chuẩn.
2. **Chế độ Nghiên cứu So sánh (Comparison Mode):**
   - Đặt câu hỏi và so sánh trực tiếp **Cấu hình A vs B vs C** hoặc **Google Gemini vs Qwen2.5 (Ollama)** trên cùng một màn hình chia cột.
   - Bảng so sánh nhanh đối chiếu nhãn an toàn, số nguồn trích dẫn, số đoạn văn bản và thời gian phản hồi.
   - Bộ công cụ đối chiếu sai số OCR (Diff Highlighter) giữa văn bản chuẩn và kết quả OCR đọc được.
3. **Cơ chế chống spam lệnh & Khóa nút bấm (Button Debounce/Disablement):**
   - Hệ thống tự động tạo mã băm chữ ký đầu vào `current_input_sig` (ảnh + câu hỏi + persona + mô hình).
   - Khi đang phân tích: Khóa nút, chuyển sang `⏳ Đang phân tích...`
   - Khi đã có kết quả cho đúng dữ liệu đó: Khóa nút, chuyển sang `✅ Đã phân tích (Đổi lựa chọn để chạy lại)`.
   - Ngay khi người dùng thay đổi bất kỳ tùy chọn nào (gõ chữ khác, đổi hồ sơ, đổi ảnh), nút lập tức mở lại trạng thái sẵn sàng màu xanh `🚀 Phân tích Nhãn & Đánh giá An toàn Sức khỏe`.

---

## 7. CƠ CHẾ ĐO LƯỜNG & ĐÁNH GIÁ ĐỘ TIN CẬY CỦA HỆ THỐNG (RELIABILITY FRAMEWORK)

Trong các bài toán AI Y tế & An toàn Thực phẩm, việc đo lường độ tin cậy của hệ thống và bộ dữ liệu đánh giá khi **chưa có điều kiện thuê hội đồng bác sĩ/chuyên gia lâm sàng gán nhãn thủ công từng câu** là một thách thức lớn. Hệ thống FoodHealthQA giải quyết vấn đề này bằng phương pháp **Giám sát yếu & Kiểm định bằng chương trình (Weak Supervision & Programmatic Verification)** đạt chuẩn khoa học quốc tế.

### 7.1. Định danh Chuẩn Bạc (Silver Standard via Programmatic Adjudication)
* **Đạo đức nghiên cứu & Tính trung thực khoa học:** Hệ thống định danh bộ 750 câu hỏi là **Silver Standard** (`expert_validated = false`), không mạo nhận là Gold Standard của bác sĩ lâm sàng khi chưa có thẩm duyệt người thật.
* **Nguyên tắc loại trừ:** Một bản ghi nếu không có văn bản pháp lý/y văn chính thống đối chiếu thì **không bao giờ được đưa vào bộ kiểm chuẩn**. Mọi câu trả lời đều có thể truy vết về văn bản gốc của WHO, EFSA, US FDA hoặc Bộ Y tế Việt Nam.

### 7.2. Bốn Thuật toán Đánh giá Độ tin cậy Bộ Benchmark (Thay thế Bác sĩ)
1. **Thuật toán Cổng Kiểm định Logic 6 Lớp (Automated G1–G6 Verification Gates):**
   *Mã nguồn:* `scripts_research/verify_v5_answers.py`
   - **Cổng G1 (Tính nhất quán kết luận):** Kiểm tra đối sánh ngữ nghĩa xem câu trả lời có mâu thuẫn với nhãn an toàn không (ví dụ: nhãn `SAFE` nhưng đáp án chứa từ ngữ cảnh báo nguy hiểm).
   - **Cổng G2 (Bao phủ thực thể):** Đối soát xem mọi mã INS, dị nguyên và tên sản phẩm trong câu hỏi có xuất hiện đầy đủ trong đáp án không.
   - **Cổng G3 (Khớp nhãn bao bì):** Đối chiếu chéo độc lập: các mã INS và số liệu dinh dưỡng trong đáp án phải thực sự in trên ảnh bao bì gốc.
   - **Cổng G4 (Hợp lệ danh mục nguồn):** Thẻ nguồn `[MÃ_NGUỒN]` bắt buộc thuộc danh mục 18 nguồn quy chuẩn được hệ thống công nhận.
   - **Cổng G5 (Định vị mục chính xác):** Mọi trích dẫn phải liên kết với một `claim_locator` cụ thể trong `source_citation_catalog.json`.
   - **Cổng G6 (Xác thực văn bản gốc):** 100% khẳng định quy chuẩn phải khớp nguyên văn với 57 khẳng định gốc đã băm mã SHA-256 trong `external_claim_verification.json`.
2. **Thuật toán Thử nghiệm Đột biến Đối kháng (Adversarial Perturbation / Mutation Testing):**
   *Mã nguồn:* `tests/test_benchmark_v5.py`
   - Để chứng minh bộ lọc G1–G6 không phải là "bù nhìn", thuật toán cố ý tạo ra các lỗi đột biến giả (chèn chất dị ứng, đổi ngược nhãn an toàn, sửa mã INS, bịa thẻ nguồn).
   - Đo lường chỉ số **Mutation Score**:
     $$\text{Mutation Score} = \frac{\text{Số ca đột biến bị bộ lọc phát hiện và chặn lại}}{\text{Tổng số ca đột biến được chèn vào}} \times 100\%$$
   - **Kết quả:** Hệ thống đạt **$\text{Mutation Score} \ge 95\%$**, chứng minh bộ kiểm định có độ nhạy (Sensitivity/Recall) rất cao.
3. **Thuật toán Kiểm định Rò rỉ Nhãn qua Câu hỏi (Shortcut Learning Detection):**
   - Để tránh tình trạng mô hình "đoán mò" nhãn an toàn chỉ từ câu hỏi mà không cần đọc nhãn ảnh, hệ thống áp dụng kỹ thuật **Absence Controls**: Dùng cùng một mẫu câu hỏi trung tính cho cả sản phẩm có chất cảnh báo và sản phẩm không có chất cảnh báo.
4. **Thuật toán Phân cụm Lấy mẫu theo Ảnh (Cluster Bootstrap per `split_group`):**
   - Thay vì lấy mẫu ngẫu nhiên từng dòng đơn lẻ (dễ gây over-optimism do các câu hỏi chung 1 ảnh bao bì có tính tương quan), thuật toán gom cụm theo từng ảnh bao bì (`split_group = 1 ảnh`) khi tính toán khoảng tin cậy 95% CI.

### 7.3. Đối chiếu nhận định với bằng chứng & Faithfulness
*Mã nguồn:* `modules/nli_verifier.py`, `scripts_research/judge_faithfulness.py`, `scripts_research/eval_nli_faithfulness.py`

Câu trả lời được tách thành các nhận định; mỗi nhận định được đối chiếu với **bằng chứng thật mà mô hình đã đọc**: văn bản nhãn đầu vào (B: nhãn chuẩn, C: chữ OCR) + các đoạn nguồn đã truy xuất (`modules/evaluation.prediction_evidence`). Có hai mức, phải gọi đúng tên:

1. **Đối chiếu từ khóa** (giao diện và `eval_nli_faithfulness.py`, không dùng LLM): đếm từ trùng giữa nhận định và bằng chứng, cộng vài luật mâu thuẫn "cấm / cho phép". Đây **không phải** suy luận ngôn ngữ (NLI) và không có độ tin cậy; huy hiệu ghi "Đối chiếu từ khóa — chưa phải xác thực ngữ nghĩa". (Trước đây huy hiệu ghi "ĐÃ XÁC THỰC NGỮ NGHĨA", dùng lời khuyên của chính guardrail làm "bằng chứng", gán độ tin cậy cố định 0,85–0,92, và script chấm dùng đáp án tham chiếu làm bằng chứng — con số 51,69% trong `data/runs/nli_faithfulness_report.json` thuộc cách cũ, không dùng được.)
2. **Faithfulness chính thức — LLM làm giám khảo** (`judge_faithfulness.py`): giám khảo gán SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO cho từng nhận định và phải trích nguyên văn câu bằng chứng; code kiểm tra trích dẫn có thật trong bằng chứng, không có thì hạ xuống NOT_ENOUGH_INFO. Nhiệt độ 0, prompt cố định (`PROMPT_VERSION`), nên dùng model khác model sinh câu trả lời. Kết quả chỉ được báo cáo kèm **mức đồng thuận với người chấm** (Cohen's kappa trên ~50 câu trả lời chấm tay, xuất bằng `export-annotation`).

### 7.4. Các Chỉ số Đo lường Khoa học Cốt lõi
* **Faithfulness** (giám khảo, mục 7.3): $\dfrac{\#\text{SUPPORTED}}{\#\text{nhận định}}$ mỗi câu trả lời, lấy trung bình theo cấu hình; báo kèm Cohen's kappa với người chấm. Câu trả lời không có nhận định nào → không áp dụng.
* **Tỷ lệ mâu thuẫn (dấu hiệu hallucination):** $\#\text{CONTRADICTED} / \#\text{nhận định}$; **tỷ lệ thiếu căn cứ:** $\#\text{NOT\_ENOUGH\_INFO} / \#\text{nhận định}$.
* **Groundedness (trùng từ):** tỉ lệ từ của câu trả lời có trong bằng chứng thật (nhãn đầu vào + đoạn nguồn). Là ước lượng bề mặt, báo cáo đúng tên này.
* **An toàn:** safety accuracy (5 nhãn; nhãn dự đoán không hợp lệ tính là sai và đếm riêng), unsafe recall, **false-safe rate** (nhãn chuẩn nguy hiểm mà hệ thống nói SAFE). Không còn điểm gộp 0,6×accuracy + 0,4×recall (trọng số tự chọn). Chỉ số có mẫu số bằng 0 ghi **không áp dụng**, không ghi 0.
* **Tỷ lệ Tuân thủ Trích dẫn (Citation Compliance):** Tỷ lệ các lượt chạy mà câu trả lời có chứa ít nhất 1 mã trích dẫn hợp lệ được công nhận trong catalog.
* **Tỷ lệ Nguồn Lạ (Unrecognized Citation Rate):** Tỷ lệ câu trả lời xuất hiện mã nguồn bịa đặt không nằm trong danh mục pháp lý.

---

## 8. BẢNG TỔNG KẾT ĐỐI CHIẾU MÃ NGUỒN CÁC MÔ-ĐUN

| Mô-đun / Tệp | Vị trí | Trách nhiệm chính |
| :--- | :--- | :--- |
| `app.py` | Gốc | Giao diện Streamlit, quản lý trạng thái session, điều phối luồng người dùng và CSS công thái học. |
| `modules/ocr_extractor.py` | Tầng 1 | PaddleOCR, VietOCR, tiền xử lý ảnh, phục hồi chính tả, tách thành phần. |
| `modules/foodon.py` | Tầng 2 | Quản lý đồ thị bản thể 29.255 nút FoodOn DAG, duyệt cha-con, LCA, lineage. |
| `modules/additive_linker.py` | Tầng 2 | Định danh tiền định phụ gia tiếng Việt $\rightarrow$ mã INS quốc tế. |
| `modules/hybrid_retriever.py`| Tầng 2 | Kiến trúc 4-Agent FoodOntoRAG (Hybrid Retrieval + Selector + Scorer + Feedback Loop).|
| `modules/persona_builder.py` | Tầng 2-4 | Giao diện Tự chỉnh Hồ sơ Sức khỏe, phòng vệ 3 lớp chống nhập sai, chuẩn hóa dị nguyên qua FoodOn. |
| `modules/dynamic_qa.py` | Tầng 3 | Nhận diện ý định câu hỏi, điều phối tra cứu CSDL theo đúng nội dung được hỏi, xuất vết tra cứu minh bạch. |
| `modules/passage_retriever.py`| Tầng 3 | Truy xuất và xếp hạng đoạn trích bằng chứng (BM25 + TF-IDF + Dense Embedding). |
| `modules/evidence.py` | Tầng 3-4 | Xử lý bằng chứng trích dẫn, phân cấp 3 Tier tin cậy nguồn, cơ chế Fallback Passage và bóc tách lý do thiếu dữ liệu. |
| `modules/constraint_model.py`| Tầng 4 | Mô hình hóa ràng buộc WSDM '21, rà soát dị nguyên, kiểm tra chống chỉ định y tế. |
| `modules/abstention.py` | Tầng 4 | Tính điểm tin cậy kép $S_{\text{total}}$, ra quyết định từ chối trả lời (`ABSTAIN`). |
| `modules/nli_verifier.py` | Tầng 4 / Hậu kiểm | Bộ thẩm định khẳng định ngữ nghĩa NLI (Atomic Claim Extraction, Entailment/Contradiction classification, Trigger-based Fast NLI). |
| `modules/pipeline.py` | Tầng 4 | Nhạc trưởng kết nối `run_grounded_pipeline`, làm giàu NHANES (`enrich_persona_with_nhanes`), prompt engineering và gọi LLM. |
| `modules/compare_runner.py` | Nghiên cứu | Bộ điều phối chạy song song so sánh A/B/C và Gemini với Qwen2.5. |
| `modules/compare_ui.py` | Nghiên cứu | Hiển thị bảng kết quả so sánh, grid chỉ số mini, đối chiếu sai số OCR. |
| `modules/benchmark_manifest.py`| Đánh giá | Quản lý bộ kiểm chuẩn 750 câu hỏi, kiểm soát tính hợp lệ và chống rò rỉ dữ liệu. |
| `scripts_research/eval_nli_faithfulness.py` | Nghiên cứu | Script đánh giá tự động độ trung thực ngữ nghĩa NLI Faithfulness và tỷ lệ ảo giác cho báo cáo/khóa luận. |
