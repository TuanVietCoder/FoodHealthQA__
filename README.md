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
Trình duyệt sẽ tự động mở ứng dụng tại `http://localhost:8501`. Thanh bên trái có hai không gian làm việc; ứng dụng mở mặc định vào **Trải Nghiệm Demo**.

**1. 🔍 Trải Nghiệm Demo (Hỏi Đáp & Quét Nhãn)** — sản phẩm hỏi đáp từ ảnh bao bì. Mặc định chạy **Cấu hình C** (ảnh → OCR → normalization → retrieval → answer); có thể chọn A hoặc B, hoặc bật *Chế độ nghiên cứu* để chạy A/B/C cạnh nhau trên cùng một đầu vào. Các tab kết quả:
- **🛡️ Đánh giá An toàn & Guardrail**: Kết luận cảnh báo y khoa, điểm tin cậy kép $S_{total}$, lời khuyên và câu trả lời LLM có trích dẫn (ở Cấu hình A, tab này là câu trả lời LLM không truy xuất).
- **🧬 Chuẩn hóa Thực thể (FoodOntoRAG)** *(B, C)*: Bảng liên kết FoodOn CURIE, mã INS, chức năng và rationale.
- **📝 Nhận diện OCR & Phục hồi Chính tả** *(chỉ C hiển thị OCR; A/B hiển thị văn bản đầu vào)*: Ảnh kèm Bounding Box màu xanh lá quanh từng dòng chữ, bảng đối chiếu OCR thô và văn bản sau sửa lỗi.

**2. 🔬 Báo Cáo Nghiên Cứu (Ablation Study 3 Cấu Hình)** — dành cho phần nghiên cứu, gồm:
- Dashboard thực nghiệm 3 cấu hình A/B/C, phân tích 4 miền chuyên môn, ma trận nhầm lẫn (Confusion Matrix) và độ lan truyền sai số (Error Propagation).
- **📁 Tra cứu Đa nguồn Dữ liệu (Data Hub)**: Trình duyệt trực quan các tệp dữ liệu CSV/JSON (đặt dưới dashboard, dùng chung cho mọi cấu hình).

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
| **Bảng tra cứu nguồn ngoài (Data Hub)** | `data/` (`P_*.csv` NHANES, `food*.csv`/`nutrient.csv` USDA, `incidents_train.csv`, `openfoodfacts_vietnam.csv`, `foodon_entities.json`, `nhanes_personas.csv`) | Dữ liệu tham chiếu, xem trong không gian "Báo Cáo Nghiên Cứu". |
| **Bộ QA** | `data/foodhealthqa_eval_set.*` (chính, 750 câu); `data/qa_benchmark.*` (đầy đủ 2.311, dự phòng) | Câu hỏi, đáp án, nguồn, nhãn lĩnh vực. |
| **Kết quả thí nghiệm** | `data/predictions_config_*.jsonl`, `data/evaluation_results.json`, `data/runs/`, `data/cache/` | Sinh ra bởi `scripts_research/`; có thể tạo lại. |

Lưu ý: `data/additives_standard.csv` (10 dòng) chỉ là bảng đối soát nhỏ, **không** thay thế `knowledge_base/additives_standard.json` (50 mục, là nguồn KB chính mà pipeline dùng).

---

## 📊 Bộ QA Silver/Pre-Gold & Đánh Giá 3 Cấu Hình

Bộ dùng để **đánh giá và demo** là **`data/foodhealthqa_eval_set.jsonl`** (bản CSV: `foodhealthqa_eval_set.csv`): **750 câu hỏi–đáp**, gắn với **376 ảnh nhãn** và 7 nhóm persona. Bộ này là tập con phân tầng của bộ đầy đủ **`data/qa_benchmark.jsonl`** (2.311 câu, 489 ảnh); cả 750 câu có cùng mã và cùng đáp án với bản đầy đủ. **Bộ 2.311 câu chỉ lưu dự phòng và làm nguồn dựng bộ con**, không dùng để chấm điểm (xem `primary_evaluation_set` trong `data/benchmark_manifest.json`). Cả hai đều là bộ **silver kiểm chứng bằng quy tắc, không có chuyên gia** (`expert_validated = false`): không dùng để tuyên bố hiệu năng trên gold standard.

- Mọi bản ghi đều có nguồn tra cứu đã kiểm chứng nguyên văn; bản ghi không có nguồn đã bị loại khỏi bộ (xem báo cáo).
- Mọi bản ghi tính điểm (`eligible_for_scoring = yes`) và đã qua 6 cổng kiểm tra G1–G6; các thẻ nguồn nằm trong `knowledge_base/sources/catalog/source_citation_catalog.json`.
- Cột `gold_safety_label` dùng 5 nhãn của app; nhãn mịn nằm ở `label_v5`.
- Mỗi bản ghi có `split_group` (một ảnh = một nhóm; chia train/test và lấy mẫu lại theo trường này) và `template_id` (dạng câu hỏi). Tier `core` (1.498) dùng để so sánh chính, `silver` (813) báo riêng. Báo cáo kết quả bằng `python scripts_research/benchmark_report.py [--predictions file.jsonl]` (theo domain/tier, macro-F1, baseline nhãn đa số, khoảng tin cậy theo ảnh).
- 753 bản ghi dị ứng có thêm cột `persona_conditions` / `persona_conditions_vi` để dựng danh sách dị ứng đúng với tên persona (trước đó 143/753 bản ghi bị lệch). Các chỉ số của nhóm dị ứng tính trước thay đổi này không còn giá trị và cần chạy lại.
- Chi tiết, giới hạn và lịch sử: [docs/BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md](docs/BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md).

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

> Bộ chấm mặc định là bộ 750 câu `data/foodhealthqa_eval_set.jsonl` (cả `generate_predictions.py` và `run_evaluation.py`). Các chỉ số là số liệu theo nhãn kiểm chứng bằng quy tắc, không phải hiệu năng gold-standard. **Mọi lệnh phải dùng cùng một tập** (`--benchmark` khi sinh dự đoán, `--gold` khi chấm hoặc so sánh); chỉ truyền `data/qa_benchmark.jsonl` khi cố ý chạy trên bản đầy đủ.

**Nhãn mẫu và đầu vào tự nhập.** Trên giao diện, A và B nhận văn bản thành phần (chọn nhãn mẫu hoặc tự nhập, không hiển thị ảnh), C nhận ảnh (ảnh mẫu hoặc tải lên). Chỉ nhãn mẫu mới có ảnh đối chiếu cho phép so sánh A–B–C; đầu vào tự nhập dùng để thử nghiệm. Chi tiết: mục 3.3 của `docs/QUY_TRINH_3_CAU_HINH_HE_THONG.md`.

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
# 0. Tạo tập pilot phân tầng 150 câu (từ qa_eval_balanced.jsonl; seed cố định)
.\venv\Scripts\python.exe .\scripts_research\build_pilot_subset.py

# 1. Sinh dự đoán A/B/C. Chọn mô hình: --llm-provider gemini (mặc định) hoặc ollama (Qwen2.5 cục bộ).
#    Kết quả ghi vào data/runs/<provider>/. --resume chạy tiếp (chạy lại bản ghi lỗi); dừng sạch khi hết hạn mức Gemini.
.\venv\Scripts\python.exe .\scripts_research\generate_predictions.py --benchmark data/qa_eval_pilot.jsonl --llm-provider ollama --resume
.\venv\Scripts\python.exe .\scripts_research\generate_predictions.py --benchmark data/qa_eval_pilot.jsonl --llm-provider gemini --resume
.\venv\Scripts\python.exe .\scripts_research\generate_predictions.py --benchmark data/qa_eval_pilot.jsonl --llm-provider gemini --status

# 2. Chấm đủ A/B/C cho từng mô hình, rồi so sánh Gemini với Qwen2.5 trên cùng tập câu hỏi
.\venv\Scripts\python.exe .\scripts_research\run_evaluation.py --run-dir data/runs/ollama --gold data/qa_eval_pilot.jsonl
.\venv\Scripts\python.exe .\scripts_research\compare_providers.py --gold data/qa_eval_pilot.jsonl

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
├── app.py                         # Giao diện chính Streamlit (2 không gian: Demo hỏi đáp từ ảnh & Báo cáo Nghiên cứu gồm Ablation, Data Hub)
├── .streamlit/config.toml         # Giao diện sáng, bảng màu teal
├── requirements.txt               # Khai báo thư viện (Streamlit, PaddleOCR, Pandas, Pillow, OpenCV...)
├── README.md                      # Tài liệu kỹ thuật và hướng dẫn vận hành
├── tinh_nang.txt                  # Đặc tả tính năng và yêu cầu nghiệp vụ
├── assets/                        # Kho ảnh bao bì & nhãn thực tế
│   ├── images/                    # Kho ảnh; 489 ảnh đang được benchmark tham chiếu
│   └── labels/                    # Nhãn bóc tách ground-truth tương ứng
├── data/                          # Kho dữ liệu tích hợp đa nguồn
│   ├── benchmark_manifest.json    # Ghi bộ đánh giá chính (750 câu) và bản đầy đủ dự phòng (v5)
│   ├── foodhealthqa_eval_set.jsonl # BỘ ĐÁNH GIÁ CHÍNH: 750 câu QA (kèm .csv và _summary.json)
│   ├── qa_benchmark.jsonl         # Bản đầy đủ 2.311 câu, chỉ lưu dự phòng / nguồn dựng bộ con
│   ├── qa_benchmark.csv           # Cùng nội dung bản đầy đủ, định dạng CSV
│   ├── build/                     # Tệp trung gian của quy trình dựng (đầu vào, bản đầy đủ, kiểm tra từng bản ghi)
│   ├── runs/                      # Kết quả theo mô hình: gemini/, ollama/ (predictions_config_*.jsonl, run_log.jsonl), comparison.md|json
│   ├── cache/                     # Bộ nhớ đệm embedding bge-m3 (embeddings_bge-m3.npz)
│   ├── qa_eval_balanced.jsonl     # Tập cân bằng 1.203 câu; qa_eval_pilot.jsonl: tập pilot 150 câu
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
│   ├── build_pilot_subset.py      # Tạo tập pilot phân tầng
│   ├── eval_passage_retrieval.py  # Đo truy xuất đoạn: INS hit, locator hit, citation recall
│   ├── measure_citation_compliance.py  # Đo tuân thủ trích dẫn và thẻ nguồn lạ khi chạy lặp
│   ├── compute_ocr_metrics.py     # Đo lường sai số OCR CER/WER
│   └── benchmark_report.py        # Báo cáo cân bằng nhãn, baseline, macro-F1, khoảng tin cậy
├── docs/                          # Báo cáo chuyên sâu
│   ├── QUY_TRINH_3_CAU_HINH_HE_THONG.md       # Báo cáo quy trình chi tiết 3 cấu hình
│   └── BAO_CAO_DANH_GIA_BENCHMARK_CHUNG_CU.md # Báo cáo kiểm chứng nguồn và trạng thái benchmark
└── tests/                         # Bộ kiểm thử tự động (244 test đạt tại 2026-11-12, gồm test_dynamic_qa.py, test_passage_retriever.py, test_dense_retrieval.py, test_llm_provider.py, test_llm_keys.py, test_compare_runner.py, test_citation_compliance.py)
```
