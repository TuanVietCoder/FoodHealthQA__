# FoodHealthQA: Hệ Thống Hỏi Đáp Tư Vấn Sức Khỏe Từ Nhãn Bao Bì Thực Phẩm

Ứng dụng web tương tác hỗ trợ tra cứu và sàng lọc phụ gia, thành phần, dị nguyên tiềm năng và thông tin dinh dưỡng từ ảnh bao bì thực phẩm. Hệ thống kết hợp mạng nơ-ron quang học, bản thể học quốc tế, kho quy chuẩn pháp lý và mô hình ngôn ngữ lớn (Google Gemini hoặc Qwen2.5 chạy cục bộ qua Ollama) với cơ chế chốt chặn an toàn y tế (Guardrail & Principled Abstention).

---

## 🔬 Kế Thừa Nghiên Cứu & Dữ Liệu Nền Tảng

1. **PaddleOCR + VietOCR Adaptive Perception Pipeline (Chỉ dùng OCR)**:
   - Hệ thống **chỉ sử dụng pipeline OCR chuyên sâu** (không dùng mô hình Student).
   - Tầng tiền xử lý quang học thích ứng: Khử góc nghiêng ảnh (*Deskewing* via `cv2.minAreaRect`), Cân bằng tương phản cục bộ (*Adaptive CLAHE* $8 \times 8$), và Làm nét ký tự nhỏ (*Unsharp Masking* via Gaussian blur).
   - PaddleOCR DBNet phát hiện vùng chữ; VietOCR VGG-Transformer nhận dạng tiếng Việt. PaddleOCR multi-pass/SVTR là nhánh dự phòng khi VietOCR không khả dụng.
   - Trích xuất thành phần bằng Lookahead Regex bảo toàn trọn vẹn cụm phụ gia trong ngoặc đơn.
2. **Post-OCR Lexical Normalizer**:
   - Phục hồi từ vựng tiếng Việt miền thực phẩm bằng Domain Lexicon/rule-based và chuẩn hóa mã INS/E-number (ví dụ: `INS. 322(i` $\rightarrow$ `INS 322i`). Gemini không tham gia sửa chính tả OCR trong thí nghiệm chính.
3. **FoodOntoRAG (2025)**:
   - Chuẩn hóa thực thể đa tác tử 4-agent (*Retriever, Selector, Scorer, Synonym Generator*) liên kết mã **FoodOn CURIE**.
4. **Food Hazard Detector (SemEval-2025 `incidents_train.csv`)**:
   - Đối soát tự động 5.082 báo cáo sự cố thực phẩm, phát hiện nguy cơ nhiễm khuẩn sinh học, hóa chất độc hại và dị nguyên ẩn chưa công bố.
5. **USDA FNDDS (`fndds_nutrition.csv`)**:
   - Định lượng dinh dưỡng (calo, natri, đường, chất béo) và so sánh ngưỡng bệnh lý theo kỹ thuật mô hình hóa ràng buộc số học pFoodReQ.
6. **NHANES Personas (`nhanes_personas.csv`)**:
   - Hồ sơ lâm sàng người dùng thực tế từ khảo sát CDC Hoa Kỳ (chỉ số BMI, huyết áp tâm thu/tâm trương, glucose, HbA1c, BUN, 54 thói quen ăn uống).
7. **Open Food Facts VN (`openfoodfacts_vietnam.csv`)**:
   - Kho sản phẩm đóng gói thị trường Việt Nam hỗ trợ tra cứu barcode và đối soát nhãn.

---

## 🛠️ Hướng Dẫn Cài Đặt & Khởi Chạy Nhanh

### Bước 1: Kích hoạt Môi trường Python (Khuyên dùng Python 3.10 hoặc 3.11)
- **Trên Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Trên Windows (Command Prompt - CMD):**
  ```cmd
  venv\Scripts\activate
  ```
- **Trên Linux / macOS / Git Bash:**
  ```bash
  source venv/Scripts/activate  # hoặc source venv/bin/activate
  ```

### Bước 2: Cài Đặt Thư Viện
```bash
pip install -r requirements.txt
pip install paddlepaddle paddleocr
```
*(Lưu ý: Nếu chưa cài đặt PaddleOCR hoặc chạy trên máy không có GPU, hệ thống tự động chạy ở CPU mode và hỗ trợ chế độ dùng văn bản nhãn tham chiếu).*

### Bước 3: Chạy Bộ Kiểm Thử Tự Động
```powershell
.\venv\Scripts\python.exe -m unittest discover tests
```
Số lượng test có thể thay đổi theo phiên bản; chỉ coi lần chạy hiện tại là hợp lệ khi lệnh kết thúc thành công.

### Bước 4: Khởi Chạy Giao Diện Web Streamlit
```powershell
streamlit run app.py
```
Trình duyệt sẽ tự động mở ứng dụng tại `http://localhost:8501`. Ứng dụng chỉ có một màn hình **hỏi đáp phụ gia và dị nguyên từ ảnh nhãn** (pipeline cấu hình C: ảnh → OCR → normalization → retrieval → answer). Kết quả hiển thị trên một trang, theo thứ tự:
1. **Ảnh nhãn và vùng chữ OCR đọc được** (khung xanh quanh từng dòng chữ).
2. **Văn bản đã đọc**, kèm so sánh OCR thô và sau phục hồi chính tả.
3. **Bảng chuẩn hóa gọn**: tên trên nhãn → thực thể chuẩn, mã INS/E, FoodOn CURIE, độ tin cậy.
4. **Câu trả lời kèm trích dẫn đánh số `[1]`, `[2]`** bấm được, nhãn lĩnh vực của câu hỏi, và danh sách nguồn (tên tài liệu, mục/điều khoản, đoạn trích, URL, mức tin cậy của nguồn). Khi thiếu bằng chứng, hộp cảnh báo nêu rõ lý do (ảnh khó đọc, không nhận diện được thành phần, không truy xuất được đoạn bằng chứng, điểm tin cậy dưới 0,65).

Quy chuẩn ADI, sự cố thu hồi và cảnh báo chỉ hiện khi khớp phụ gia/dị nguyên được hỏi. Phần chi tiết kỹ thuật (bảng 4-Agent FoodOntoRAG đầy đủ, phả hệ FoodOn, từng dòng OCR, các đoạn bằng chứng đã truy xuất) nằm trong mục **🔧 Chi tiết kỹ thuật**. Cấu hình A và B chỉ dùng để đánh giá, chạy bằng `scripts_research/` và không có trong giao diện.

### 🧭 Cách hỏi đáp: QA động

Hệ thống đi theo hướng **QA động**: câu hỏi của người dùng quyết định hệ thống tra cứu gì trong cơ sở tri thức (KB).

- Ở trang demo, người dùng **tự gõ câu hỏi** (có gợi ý câu hỏi phổ biến để chọn nhanh) và **chọn hồ sơ sức khỏe (persona)**. Câu hỏi và đáp án của bộ QA benchmark **không** đi vào giao diện.
- Với Config B và C, tầng `modules/dynamic_qa.py` hiểu ý định câu hỏi, nhận diện phụ gia/dị nguyên/chất dinh dưỡng được hỏi, tra KB theo đó và ghi vết nguồn đã tra (khối **"🔎 Tri thức đã tra cứu theo câu hỏi"** trong kết quả). Khi câu hỏi nêu một mã phụ gia mà KB hoàn toàn không có và nhãn cũng không có, hệ thống từ chối kèm lý do thay vì để LLM suy đoán.
- Config A (LLM không truy xuất) giữ nguyên làm baseline.
- Bộ QA benchmark chỉ là **bộ chấm** chạy ngoài giao diện (`generate_predictions.py`, `run_evaluation.py`). Chi tiết: mục 4.5 và 7.4 của [docs/QUY_TRINH_3_CAU_HINH_HE_THONG.md](docs/QUY_TRINH_3_CAU_HINH_HE_THONG.md).

---

## 🗺️ Bản Đồ Dữ Liệu

| Nhóm | Vị trí | Vai trò |
| :--- | :--- | :--- |
| **Cơ sở tri thức (KB) chuẩn hóa** | `knowledge_base/` | Phụ gia, thành phần, nhóm dị nguyên, thẻ nguồn (`sources/`). Pipeline nạp trực tiếp khi truy xuất. |
| **Bảng tra cứu nguồn ngoài (Data Hub)** | `data/` (`P_*.csv` NHANES, `food*.csv`/`nutrient.csv` USDA, `incidents_train.csv`, `openfoodfacts_vietnam.csv`, `foodon_entities.json`, `nhanes_personas.csv`) | Dữ liệu tham chiếu của các tầng chuẩn hóa và đối soát; không có màn hình riêng trong giao diện. |
| **Bộ QA** | `data/foodhealthqa_eval_set.*` (bộ đánh giá duy nhất, 750 câu); `data/qa_benchmark.*` (bản nguồn 2.311, chỉ để dựng bộ 750) | Câu hỏi, đáp án, nguồn, nhãn lĩnh vực. |
| **Kết quả thí nghiệm** | `data/predictions_config_*.jsonl`, `data/evaluation_results.json`, `data/runs/`, `data/cache/` | Sinh ra bởi `scripts_research/`; có thể tạo lại. |

Lưu ý: `data/additives_standard.csv` (10 dòng) chỉ là bảng đối soát nhỏ, **không** thay thế `knowledge_base/additives_standard.json` (50 mục, là nguồn KB chính mà pipeline dùng).

---

## 📊 Bộ QA Silver/Pre-Gold & Đánh Giá 3 Cấu Hình

Bộ **đánh giá duy nhất** của hệ thống là **`data/foodhealthqa_eval_set.jsonl`** (bản CSV: `foodhealthqa_eval_set.csv`): **750 câu hỏi–đáp**, gắn với **376 ảnh nhãn** và 7 nhóm persona. Bộ này là tập con phân tầng của bản nguồn `data/qa_benchmark.jsonl` (2.311 câu, 489 ảnh), cùng mã và cùng đáp án; bản nguồn **chỉ dùng để dựng bộ 750** (`scripts_research/build_foodhealthqa_subset.py`), không dùng để chấm điểm hay demo (xem `primary_evaluation_set` trong `data/benchmark_manifest.json`). Đây là bộ **silver kiểm chứng bằng quy tắc, không có chuyên gia** (`expert_validated = false`): không dùng để tuyên bố hiệu năng trên gold standard. Mô tả đầy đủ (cách dựng, phân bố, nguồn dẫn chứng, giới hạn): [docs/MO_TA_BO_DANH_GIA_750_CAU.md](docs/MO_TA_BO_DANH_GIA_750_CAU.md).
- Mô tả cấu trúc và phân bố: [docs/MO_TA_BO_DANH_GIA_750_CAU.md](docs/MO_TA_BO_DANH_GIA_750_CAU.md). Cách tạo và kiểm chứng: [docs/BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md](docs/BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md).

- Mọi bản ghi đều có nguồn tra cứu đã kiểm chứng nguyên văn; bản ghi không có nguồn đã bị loại khỏi bộ (xem báo cáo).
- Mọi bản ghi tính điểm (`eligible_for_scoring = yes`) và đã qua 6 cổng kiểm tra G1–G6; các thẻ nguồn nằm trong `knowledge_base/sources/catalog/source_citation_catalog.json`.
- Cột `gold_safety_label` dùng 5 nhãn của app; nhãn mịn nằm ở `label_v5`.
- Mỗi bản ghi có `split_group` (một ảnh = một nhóm; chia train/test và lấy mẫu lại theo trường này) và `template_id` (dạng câu hỏi). Tier `core` (1.498) dùng để so sánh chính, `silver` (813) báo riêng. Báo cáo kết quả bằng `python scripts_research/benchmark_report.py [--predictions file.jsonl]` (theo domain/tier, macro-F1, baseline nhãn đa số, khoảng tin cậy theo ảnh).
- 753 bản ghi dị ứng có thêm cột `persona_conditions` / `persona_conditions_vi` để dựng danh sách dị ứng đúng với tên persona (trước đó 143/753 bản ghi bị lệch). Các chỉ số của nhóm dị ứng tính trước thay đổi này không còn giá trị và cần chạy lại.

- **Phân bổ 4 miền chuyên môn:**
  - `vulnerable_groups`: 712 câu
  - `allergen_detection`: 629 câu
  - `additive_regulation`: 593 câu
  - `insufficient_data`: 377 câu

### Ma trận Hiệu Năng 3 Cấu hình Thực Nghiệm:

| Chỉ số Đánh giá | Cấu hình A (LLM-Only) | Cấu hình B (Reference Text RAG) | Cấu hình C (Full OCR + RAG) |
| :--- | :---: | :---: | :---: |
| **Bản chất kỹ thuật** | Closed-book (Không RAG) | Open-book (Văn bản sạch) | End-to-End từ ảnh bao bì |
| **Đầu vào** | Văn bản nhãn tham chiếu | Văn bản nhãn tham chiếu | Ảnh nhãn thực tế |
| **OCR** | Không | Không | PaddleOCR DBNet + VietOCR |
| **Normalization/Retrieval/Guardrail** | Không | Có | Có — cùng mã xử lý với B |
| **Nhãn đánh giá** | Do LLM sinh | Do guardrail sinh | Do guardrail sinh |

> Mọi script (`generate_predictions.py`, `run_evaluation.py`, `compare_providers.py`, `eval_passage_retrieval.py`...) mặc định dùng bộ 750 câu `data/foodhealthqa_eval_set.jsonl`; nếu file thiếu, script báo lỗi thay vì chuyển sang bộ khác. Các chỉ số là số liệu theo nhãn kiểm chứng bằng quy tắc, không phải hiệu năng gold-standard.

**Đầu vào của từng cấu hình khi chấm.** A và B nhận văn bản thành phần lấy từ annotation của ảnh mẫu (không OCR); C nhận chính ảnh và chạy OCR. Chi tiết: mục 3.3 của `docs/QUY_TRINH_3_CAU_HINH_HE_THONG.md`.

**Cấu hình mô hình và truy xuất (biến môi trường / `.env`):**

| Biến | Ý nghĩa |
| :--- | :--- |
| `LLM_PROVIDER` | `gemini` (mặc định) hoặc `ollama` (Qwen2.5 chạy cục bộ; đổi model bằng `OLLAMA_MODEL`) |
| `GEMINI_API_KEYS` hoặc tệp `gemini_keys.txt` | Danh sách nhiều khóa Gemini, dùng lần lượt đến khi hết hạn mức ngày (mẫu: `gemini_keys.example.txt`) |
| `PASSAGE_EMBEDDING` | `bge-m3` để bật truy xuất dense (cần `ollama pull bge-m3`); không đặt thì dùng BM25 + TF-IDF |

---

## 🚀 Kịch Bản Nghiên Cứu & Tái Tạo Thực Nghiệm

Toàn bộ script nghiên cứu khoa học được lưu tại thư mục `scripts_research/`:

```powershell
# 0. (Tùy chọn) Dựng lại bộ 750 câu từ bản nguồn v5
.\venv\Scripts\python.exe .\scripts_research\build_foodhealthqa_subset.py

# 1. Sinh dự đoán A/B/C. Chọn mô hình: --llm-provider gemini (mặc định) hoặc ollama (Qwen2.5 cục bộ).
#    Kết quả ghi vào data/runs/<provider>/. --resume chạy tiếp (chạy lại bản ghi lỗi); dừng sạch khi hết hạn mức Gemini.
.\venv\Scripts\python.exe .\scripts_research\generate_predictions.py --llm-provider ollama --resume
.\venv\Scripts\python.exe .\scripts_research\generate_predictions.py --llm-provider gemini --resume
.\venv\Scripts\python.exe .\scripts_research\generate_predictions.py --llm-provider gemini --status

# 2. Chấm đủ A/B/C cho từng mô hình, rồi so sánh Gemini với Qwen2.5 trên cùng tập câu hỏi
.\venv\Scripts\python.exe .\scripts_research\run_evaluation.py --run-dir data/runs/ollama
.\venv\Scripts\python.exe .\scripts_research\compare_providers.py

# 3. Đo sai số nhận diện OCR (CER/WER) và độ chính xác bóc tách
.\venv\Scripts\python.exe .\scripts_research\compute_ocr_metrics.py

# 4. Đo chất lượng truy xuất đoạn bằng chứng (bm25/tfidf/dense/hybrid...; dense cần `ollama pull bge-m3`)
.\venv\Scripts\python.exe .\scripts_research\eval_passage_retrieval.py

# 5. Đo tỷ lệ tuân thủ trích dẫn khi chạy lặp (B; số lượt gọi = --questions x --repeats)
.\venv\Scripts\python.exe .\scripts_research\measure_citation_compliance.py --llm-provider ollama --questions 10 --repeats 3

# 6. Củng cố metadata cơ sở tri thức
.\venv\Scripts\python.exe .\scripts_research\harden_knowledge_base.py
```

---

## 🎯 Cấu Trúc Thư Mục Hệ Thống

```text
d:\FoodHealthQA__\
├── app.py                         # Giao diện Streamlit: demo hỏi đáp phụ gia/dị nguyên từ ảnh nhãn (cấu hình C)
├── .streamlit/config.toml         # Giao diện sáng, bảng màu teal
├── requirements.txt               # Khai báo thư viện (Streamlit, PaddleOCR, Pandas, Pillow, OpenCV...)
├── README.md                      # Tài liệu kỹ thuật và hướng dẫn vận hành
├── tinh_nang.txt                  # Đặc tả tính năng và yêu cầu nghiệp vụ
├── assets/                        # Kho ảnh bao bì & nhãn thực tế
│   ├── images/                    # Kho ảnh; 489 ảnh đang được benchmark tham chiếu
│   └── labels/                    # Nhãn bóc tách ground-truth tương ứng
├── data/                          # Kho dữ liệu tích hợp đa nguồn
│   ├── benchmark_manifest.json    # Ghi bộ đánh giá duy nhất (750 câu) và bản nguồn v5
│   ├── foodhealthqa_eval_set.jsonl # BỘ ĐÁNH GIÁ DUY NHẤT: 750 câu QA (kèm .csv và _summary.json); mặc định của mọi script
│   ├── qa_benchmark.jsonl         # Bản nguồn 2.311 câu, chỉ dùng để dựng bộ 750, không dùng để đánh giá
│   ├── qa_benchmark.csv           # Cùng nội dung bản đầy đủ, định dạng CSV
│   ├── build/                     # Tệp trung gian của quy trình dựng (đầu vào, bản đầy đủ, kiểm tra từng bản ghi)
│   ├── runs/                      # Kết quả theo mô hình: gemini/, ollama/ (predictions_config_*.jsonl, run_log.jsonl), comparison.md|json
│   ├── cache/                     # Bộ nhớ đệm embedding bge-m3 (embeddings_bge-m3.npz)
│   ├── archive/retired_eval_sets/ # Các tập đã ngừng dùng (cân bằng 1.203, pilot 150) và script/test dựng chúng, chỉ lưu để tham khảo
│   ├── evaluation_results.json    # Kết quả đo 3 cấu hình A, B, C (CHƯA CÓ: tạo bằng generate_predictions.py rồi run_evaluation.py; kết quả theo mô hình nằm ở data/runs/<provider>/)
│   ├── _backup_pre_persona_conditions/  # Bản benchmark trước khi thêm persona_conditions (kèm SHA256.txt)
│   ├── foodon_entities.json       # Danh mục bản thể học FoodOn CURIE
│   ├── incidents_train.csv        # 5.082 báo cáo sự cố Food Hazard Detector
│   ├── fndds_nutrition.csv        # Dinh dưỡng định lượng USDA FNDDS
│   ├── nhanes_personas.csv        # Hồ sơ lâm sàng NHANES CDC Hoa Kỳ
│   ├── openfoodfacts_vietnam.csv  # Nhãn sản phẩm thực phẩm thị trường Việt Nam
│   └── additives_standard.csv     # Bảng quy chuẩn phụ gia đối soát
├── knowledge_base/                # Cơ sở tri thức chuẩn hóa (Knowledge Base)
│   ├── additives_standard.json    # Quy chuẩn phụ gia, mã INS, ADI JECFA, TT24/2019, Codex STAN 192
│   ├── ingredients_standard.json  # 51 thành phần thông dụng liên kết FoodOn CURIE
│   └── allergen_groups.json       # 21 nhóm dị nguyên ưu tiên theo US FDA FALCPA & WHO JEMRA
├── modules/                       # Mã nguồn kiến trúc 4 tầng
│   ├── ocr_extractor.py           # Tầng 1: Tiền xử lý quang học, PaddleOCR, Phục hồi chính tả, Bounding Box
│   ├── foodon.py                  # Tầng 2: Tra cứu, xác thực và liên kết FoodOn CURIE
│   ├── hybrid_retriever.py        # Tầng 2-3: Bộ truy xuất lai đa tác tử (BM25 + TF-IDF/SBERT)
│   ├── passage_retriever.py       # Truy xuất đoạn bằng chứng: BM25 + TF-IDF + dense, RRF, slot khớp INS
│   ├── ollama_embedder.py         # Embedding đa ngôn ngữ (bge-m3) qua Ollama, có cache đĩa
│   ├── llm_keys.py                # Danh sách khóa Gemini, xoay vòng theo hạn mức ngày, chế độ dừng sạch khi chạy lô
│   ├── compare_runner.py          # Chạy cùng đầu vào qua A/B/C và một hoặc hai mô hình (Gemini, Qwen2.5)
│   ├── compare_ui.py              # Hiển thị so sánh A/B/C cạnh nhau trong giao diện
│   ├── constraint_model.py        # Tầng 3-4: Ràng buộc số học (pFoodReQ) & Guardrail an toàn y tế
│   ├── pipeline.py                # Tích hợp End-to-End toàn bộ luồng xử lý
│   ├── dynamic_qa.py              # Tầng QA động: hiểu câu hỏi, tra KB theo câu hỏi, vết tra cứu, khoảng trống tri thức
│   ├── benchmark.py               # Đo lường khoa học: Exact Match, F1, Citation Precision/Recall
│   └── ablation.py                # Kịch bản Ablation Study thành phần lõi
├── scripts_research/              # Các công cụ phục vụ nghiên cứu & thực nghiệm
│   ├── add_authority_records_v5.py / create_benchmark_v5.py / verify_v5_answers.py / finalize_benchmark_v5.py  # Chuỗi dựng và kiểm bộ QA v5
│   ├── fetch_external_sources.py / verify_external_claims.py  # Tải nguồn gốc và kiểm chứng claim
│   ├── add_persona_conditions.py  # Sinh cột persona_conditions / persona_conditions_vi cho bản ghi dị ứng
│   ├── generate_predictions.py    # Sinh câu trả lời bằng Gemini hoặc Qwen2.5 (Ollama); chạy theo lô, resume, --status
│   ├── run_evaluation.py          # Đánh giá tự động 3 cấu hình và xuất ma trận nhầm lẫn (--run-dir cho từng mô hình)
│   ├── compare_providers.py       # So sánh Gemini và Qwen2.5 trên cùng tập câu hỏi (bootstrap theo cặp)
│   ├── eval_passage_retrieval.py  # Đo truy xuất đoạn: INS hit, locator hit, citation recall
│   ├── measure_citation_compliance.py  # Đo tuân thủ trích dẫn và thẻ nguồn lạ khi chạy lặp
│   ├── compute_ocr_metrics.py     # Đo lường sai số OCR CER/WER
│   └── benchmark_report.py        # Báo cáo cân bằng nhãn, baseline, macro-F1, khoảng tin cậy
├── docs/                          # Báo cáo chuyên sâu
│   ├── MO_TA_BO_DANH_GIA_750_CAU.md           # Mô tả bộ đánh giá 750 câu
│   ├── QUY_TRINH_3_CAU_HINH_HE_THONG.md       # Báo cáo quy trình chi tiết 3 cấu hình
│   └── BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md # Cách tạo và kiểm chứng bộ 750 câu
└── tests/                         # Bộ kiểm thử tự động (282 test đạt tại 2026-11-13, gồm test_dynamic_qa.py, test_passage_retriever.py, test_dense_retrieval.py, test_llm_provider.py, test_llm_keys.py, test_compare_runner.py, test_citation_compliance.py)
```
