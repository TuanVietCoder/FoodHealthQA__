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
* **Bước 4: Phân đoạn thành phần bảo toàn ngữ cảnh (Regex Segmentation):**
  - Tách các thành phần độc lập bằng biểu thức chính quy bảo toàn dấu ngoặc đơn: Cụm `"Chất làm dày (INS 412, INS 415)"` không bị chẻ vụn mà được nhận dạng nguyên khối.
* **Điểm tin cậy thị giác:**
  $$S_{\text{perception}} = \frac{1}{|M|} \sum_{m \in M} \text{conf}(m) \in [0, 1]$$
  *(Với Cấu hình B, do dùng trực tiếp văn bản tham chiếu nên $S_{\text{perception}} = 1.0$).*

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
3. **Agent 3: LLM Scorer (Tác tử thẩm định độc lập):**
   - Chấm điểm tin cậy liên tục $\tau \in [0, 1]$, phạt nặng sai lệch bản chất hóa học.
   - Nếu $\tau \ge 0.70 \rightarrow$ Phê duyệt kết quả (PASS).
4. **Agent 4: Synonym Generator & Feedback Loop (Sinh từ đồng nghĩa & Vòng phản hồi):**
   - Khi $\tau < 0.70$, đọc lý do lỗi (`failure_reason` từ Scorer) để sinh ra 5 cách diễn đạt thay thế (tên khoa học, biến thể viết đảo ngữ).
   - Kích hoạt vòng truy xuất lượt 2 (vòng lặp 1 bước kiểm soát chống trôi nghĩa).

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
   - `ABSTAIN`: Từ chối kết luận do dữ liệu OCR mờ ($S_{\text{total}} < 0.65$), thiếu bảng thành phần trọng yếu hoặc câu hỏi ngoài phạm vi y khoa.
2. **Không gian Nhãn Chuẩn hóa (Taxonomy) 4 Cấp độ:**
   Khi chấm điểm benchmark hoặc hiển thị đánh giá an toàn cho người dùng (kể cả câu hỏi tự nhập), mọi kết luận kỹ thuật đều được ánh xạ về tập 4 nhãn chuẩn mực:
   - 🟢 **`SAFE`** (An toàn / Chưa thấy cảnh báo trong dữ liệu đã lọc): Ánh xạ từ `SCREENED_NO_ALERT`.
   - 🟡 **`CAUTION`** (Thận trọng / Lưu ý liều lượng): Ánh xạ từ `CAUTION`.
   - 🔴 **`UNSAFE`** (Nguy hiểm / Cảnh báo nguy cơ cao): Ánh xạ từ `WARNING` và `CRITICAL_ALERT`.
   - ⚪ **`ABSTAIN`** (Từ chối tư vấn có nguyên tắc): Ánh xạ từ `ABSTAIN`.
   *Trình bày UI:* Dưới tiêu đề trực quan, hệ thống ghi rõ: `Đánh giá an toàn tự động: [NHÃN] (Mức an toàn theo chuẩn kiểm định 4 cấp độ)`.

#### C. Điểm tin cậy kép & Cơ chế từ chối (Confidence Scoring & Abstention):
- Điểm tin cậy tổng thể:
  $$S_{\text{total}} = 0.5 \cdot S_{\text{perception}} + 0.5 \cdot S_{\text{retrieval}}$$
- Nếu $S_{\text{total}} < 0.65 \rightarrow$ Kích hoạt `ABSTAIN` loại `insufficient_data` (yêu cầu người dùng chụp lại ảnh rõ hơn hoặc bổ sung thông tin).
- Nếu câu hỏi yêu cầu chẩn đoán/kê đơn $\rightarrow$ Kích hoạt `ABSTAIN` loại `out_of_scope` (hướng dẫn người dùng đến gặp bác sĩ chuyên khoa).

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
   - **Đánh giá an toàn tự động 4 cấp độ:** Hiển thị nhãn chuẩn mực (`SAFE`, `CAUTION`, `UNSAFE`, `ABSTAIN`) kèm giải thích minh bạch, phân định rõ ràng giữa suy diễn động cho câu hỏi tự nhập và bộ dữ liệu kiểm chuẩn.
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

## 7. BẢNG TỔNG KẾT ĐỐI CHIẾU MÃ NGUỒN CÁC MÔ-ĐUN

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
| `modules/pipeline.py` | Tầng 4 | Nhạc trưởng kết nối `run_grounded_pipeline`, làm giàu NHANES (`enrich_persona_with_nhanes`), prompt engineering và gọi LLM. |
| `modules/compare_runner.py` | Nghiên cứu | Bộ điều phối chạy song song so sánh A/B/C và Gemini với Qwen2.5. |
| `modules/compare_ui.py` | Nghiên cứu | Hiển thị bảng kết quả so sánh, grid chỉ số mini, đối chiếu sai số OCR. |
| `modules/benchmark_manifest.py`| Đánh giá | Quản lý bộ kiểm chuẩn 750 câu hỏi, kiểm soát tính hợp lệ và chống rò rỉ dữ liệu. |
