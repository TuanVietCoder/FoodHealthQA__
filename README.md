# FoodHealthQA: Hệ Thống Hỏi Đáp Tư Vấn Sức Khỏe Từ Nhãn Bao Bì Thực Phẩm (v7 - Comprehensive CSV & Multi-Data Integration)

Ứng dụng web demo tương tác hỗ trợ tra cứu và sàng lọc phụ gia, thành phần, dị nguyên tiềm năng và thông tin dinh dưỡng từ ảnh bao bì. Kết quả OCR/keyword không chứng minh sản phẩm an toàn, không xác nhận dị ứng cá nhân và không thay thế tư vấn chuyên môn.

Kế thừa các công trình nghiên cứu và dữ liệu nền tảng:
- **PaddleOCR & Adaptive Perception Pipeline**: Tiền xử lý quang học (Deskewing, Adaptive CLAHE, Unsharp Masking) kết hợp Tinh chỉnh tham số nhận diện DBNet (`det_db_unclip_ratio=2.0`, `det_db_thresh=0.25`).
- **Post-OCR Lexical Normalizer**: Tầng phục hồi chính tả tiếng Việt miền thực phẩm dựa trên vocabulary nội bộ và FoodOn; tự động chuẩn hóa mã phụ gia INS/E-number. Vocabulary này không xác nhận tuân thủ Thông tư 24/2019/TT-BYT.
- **FoodOntoRAG (2025)**: Chuẩn hóa thực thể phụ gia 4-agent (Retriever, Selector, Scorer, Synonym Generator) với mã **FoodOn CURIE**.
- **Food Hazard Detector (SemEval-2025 `incidents_train.csv`)**: Tự động đối soát 5.082 báo cáo sự cố thực phẩm, phát hiện nguy cơ nhiễm khuẩn sinh học, hóa chất độc hại và dị nguyên ẩn chưa công bố.
- **USDA FNDDS (`fndds_nutrition.csv`)**: Phân tích định lượng dinh dưỡng (calo, natri, đường, chất béo) và so sánh ngưỡng bệnh lý.
- **NHANES Personas (`nhanes_personas.csv`)**: Hồ sơ lâm sàng người dùng thực tế từ khảo sát CDC Hoa Kỳ (chỉ số BMI, huyết áp tâm thu/tâm trương, glucose, HbA1c, BUN, 54 thói quen ăn uống).
- **Open Food Facts VN (`openfoodfacts_vietnam.csv`)**: Kho sản phẩm đóng gói thị trường Việt Nam hỗ trợ tra cứu mã vạch barcode và đối soát nhãn.

---

## 🛠️ Hướng Dẫn Cài Đặt & Khởi Chạy Nhanh

### Bước 1: Kích hoạt Môi trường Python (Khuyên dùng Python 3.10 hoặc 3.11)
- **Trên Windows (Git Bash):**
  ```bash
  source venv/Scripts/activate
  ```
- **Trên Windows (Command Prompt - CMD):**
  ```cmd
  venv\Scripts\activate
  ```
- **Trên Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

### Bước 2: Cài Đặt Thư Viện
```bash
pip install -r requirements.txt
pip install paddlepaddle paddleocr
```
*(Lưu ý: Nếu chưa cài đặt PaddleOCR hoặc đang gặp sự cố môi trường, hệ thống vẫn khởi chạy trơn tru 100% nhờ cơ chế Nhãn chuẩn Ground Truth Benchmark tích hợp sẵn).*

### Bước 3: Chạy Kiểm Thử Tự Động Toàn Diện
Kiểm tra khả năng tích hợp và liên kết của các tệp dữ liệu CSV:
```bash
python test_csv_integration.py
```
*(Màn hình sẽ in ra kết quả phân tích kiểm thử đối soát thành công 100%).*

### Bước 4: Khởi Chạy Giao Diện Web Streamlit
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở ứng dụng tại `http://localhost:8501`.

### Tra cứu và sử dụng FoodOn
Đặt danh mục FoodOn dạng JSON tại `data/foodon_entities.json`. Ứng dụng nạp và lập chỉ mục toàn bộ thuật ngữ (cache đến khi tệp thay đổi). Trong tab **Tra cứu Đa nguồn Dữ liệu (Data Hub)**, tìm theo nhãn tiếng Anh hoặc CURIE để xem các kết quả phù hợp.

Trong bảng chuẩn hóa, ứng dụng chỉ gán CURIE khi nhãn FoodOn khớp duy nhất sau khi chuẩn hóa chữ hoa/thường, dấu và dấu câu; mã không khớp, mơ hồ hoặc obsolete sẽ không được gán. Tệp JSON hiện có các trường `curie`, `label` và `parents`, không có đủ từ đồng nghĩa để tự động ánh xạ mọi tên tiếng Việt hoặc phụ gia. Trường hợp không chứng minh được kết quả khớp sẽ hiển thị `FOODON:00000000`; không tự suy đoán mã.

### Phạm vi và trạng thái knowledge base

`knowledge_base/additives_standard.json` hiện có 11 phụ gia; `knowledge_base/ingredients_standard.json` có 14 nhóm dị nguyên để sàng lọc và 36 thành phần thông dụng được nối với FoodOn bằng nhãn khớp duy nhất. Phạm vi nhóm có nguồn được tách khỏi các alias sàng lọc: alias hiện là vocabulary tuyển chọn, đang chờ chuyên gia rà soát và không bao quát đầy đủ mọi dị nguyên. Đây là bước mở rộng có provenance, không phải danh mục đầy đủ mọi nguyên liệu, phụ gia hay thành phần của sản phẩm. Tên/alias tiếng Việt của các mục FoodOn mới cũng cần rà soát trước khi dùng để sàng lọc tự động.

Thông tin Codex, EU và FDA được ghi riêng theo thẩm quyền và nguồn; ghi chú Việt Nam, điều kiện sử dụng phụ gia và giới hạn theo nhóm thực phẩm vẫn ở trạng thái chờ xác minh văn bản chính thức. Vì vậy runtime không đưa ra kết luận tuân thủ pháp luật Việt Nam từ các claim này. E621 có ADI nhóm EFSA 30 mg/kg thể trọng/ngày cho E620-E625, biểu thị theo glutamic acid; hệ thống không tính nó như giới hạn số riêng cho E621. Claim sức khỏe cũ được giữ trong hàng đợi rà soát nhưng không dùng làm cảnh báo cá nhân. Khớp từ khóa dị nguyên chỉ là cảnh báo kiểm tra nhãn, không phải chẩn đoán hay kết luận chống chỉ định. Trạng thái `SCREENED_NO_ALERT` chỉ có nghĩa là không tìm thấy cảnh báo trong phần dữ liệu đã nhận diện; nó không phải kết luận an toàn hay tuân thủ.

Tạo lại metadata và hàng đợi review sau khi cập nhật KB:

```powershell
.\venv\Scripts\python.exe .\scripts\harden_knowledge_base.py
.\venv\Scripts\python.exe .\scripts\create_review_templates.py
```

### Nhận diện ảnh và chuẩn hóa thành phần
Với ảnh tải lên, ứng dụng chạy OCR PaddleOCR trên ảnh thật, sau đó chuẩn hóa văn bản, tách các thành phần và liên kết từng mã INS/E với knowledge base. Nhiều mã phụ gia xuất hiện trong cùng ngoặc được tách thành các mention riêng để tránh gộp nhầm thực thể. Nếu OCR không khả dụng hoặc không nhận diện được văn bản, ứng dụng dừng phân tích và không thay bằng nhãn mẫu. Ground-truth chỉ dùng khi người dùng chọn ảnh mẫu ở chế độ demo. Yêu cầu nhận diện có thể được đáp ứng bằng OCR; Student model là một phương án thay thế tùy chọn, không bắt buộc trong phạm vi hiện tại.

### OCR cục bộ và QA từ ảnh

Không còn duy trì metric benchmark riêng cho CER/WER, ingredient normalization F1 hoặc additive/allergen linking trên ảnh nhãn. Các biểu mẫu ground truth và kết quả pilot cho đánh giá pipeline đó đã được gỡ bỏ. OCR trong ứng dụng vẫn hoạt động; script OCR cục bộ có thể tạo đầu vào ảnh cho cấu hình OCR+RAG, không cần nhãn ground truth:

```powershell
.\venv\Scripts\python.exe .\scripts\run_image_ocr.py --limit 5
```

Lệnh này chỉ ghi dự đoán OCR cục bộ vào `data/image_ocr_predictions.csv`; không tính độ chính xác. `data/image_benchmark_mapping_template.csv` vẫn dùng riêng để nối một câu QA đã được duyệt với ảnh sản phẩm tương ứng cho so sánh ba cấu hình.

### Bộ QA 600 câu có liên kết nguồn

`data/qa_benchmark_600_source_grounded.jsonl` là bộ riêng gồm 600 câu nháp, chia đều 100 câu cho sáu miền. Facts về phụ gia dùng WHO/FAO JECFA và EFSA; dị nguyên đối chiếu bản Codex do FAO lưu; thai kỳ có giới hạn rõ rằng đánh giá phụ gia không xác định mức phơi nhiễm hay độ an toàn cá nhân; câu trẻ em và chế độ ăn dùng hướng dẫn WHO; câu dinh dưỡng truy vấn các bản ghi USDA FNDDS trong CSV cục bộ. Miền bệnh nền hiện dùng khuyến nghị chế độ ăn lành mạnh chung của WHO, không phải hướng dẫn điều trị theo từng bệnh. Metadata, URL nguồn và giới hạn của bộ nằm trong `data/qa_benchmark_600_source_metadata.json`.

Đã đối chiếu PDF Codex được liên kết: trang đầu ghi sửa đổi đến năm 2010; danh sách thực phẩm gây quá mẫn nằm ở mục 4.2.1.4 (không phải Annex II). Mục này quy định sulfite từ 10 mg/kg trở lên; mục 6 cho phép miễn một số yêu cầu ghi nhãn mục 4.2 đối với đơn vị nhỏ có diện tích bề mặt lớn nhất dưới 10 cm², ngoại trừ gia vị và thảo mộc. QA hiện nêu phạm vi đó và cảnh báo kiểm tra sửa đổi sau năm 2010 cùng quy định Việt Nam trước khi áp dụng. Đây là xác minh nội dung của đúng PDF được dẫn, **không phải xác nhận rằng PDF đó là bản Codex mới nhất hoặc quy định hiện hành tại Việt Nam**.

Các câu được tạo thành nhóm biến thể từ một tập facts nhỏ, không phải 600 fact độc lập. Giữ nguyên `question_group_id`/`split_group_id` khi chia train/dev/test để tránh rò rỉ. Dùng `data/qa_benchmark_600_source_review.csv` để chuyên gia rà soát; tất cả mục hiện ở trạng thái `pending`. Bộ này **chưa được chuyên gia xác thực, không phải gold benchmark**; đặc biệt cần kiểm tra nguyên văn/ngoại lệ của Codex, citation đến tài liệu gốc và phạm vi từng khuyến nghị y tế trước khi dùng đánh giá chính thức. Không diễn giải câu dinh dưỡng FNDDS là giá trị của mọi sản phẩm cùng tên.

Tạo lại JSONL, biểu mẫu review và metadata từ các nguồn/dữ liệu hiện có:

```powershell
.\venv\Scripts\python.exe .\scripts\generate_source_grounded_qa.py
```

Đây là bộ QA duy nhất đang được dùng trong dự án. Tạo lại biểu mẫu review và các biểu mẫu đánh giá liên quan:

```powershell
.\venv\Scripts\python.exe .\scripts\create_review_templates.py
```

Chuyên gia cập nhật `data/qa_benchmark_600_source_review.csv`. Script so sánh ba cấu hình đọc biểu mẫu này và chỉ sử dụng các hàng đã được đánh dấu `approved`.

### Đánh giá groundedness và faithfulness

`data/answer_adjudication_template.csv` là form rỗng để người đánh giá ghi câu hỏi, câu trả lời, bằng chứng đã đưa cho mô hình, citation, groundedness, faithfulness, citation entailment, safety, hallucination, nguy cơ bị bỏ sót và việc có cần chuyển chuyên gia hay không. Chấm từng mục 0–2: 0 = sai/không có căn cứ; 1 = đúng một phần hoặc bằng chứng/citation yếu; 2 = được bằng chứng hỗ trợ đầy đủ. Đánh giá một phần câu trả lời độc lập trước, rồi adjudicate bất đồng; câu cần chuyên môn y tế hoặc nhãn mơ hồ phải chuyển chuyên gia, không suy đoán thay.

Script `scripts/run_three_configs.py` chạy LLM-only, text-RAG và OCR+RAG nhưng từ chối chạy nếu chưa có câu QA nguồn được chuyên gia duyệt và mapping ảnh–câu hỏi. Mỗi câu được chạy qua cả ba cấu hình; đầu ra JSONL cần được chấm thủ công bằng `data/answer_adjudication_template.csv`. Chưa có mapping/gold được duyệt nên chưa có số liệu so sánh hợp lệ.

---

## 🎯 Cấu Trúc Thư Mục Dữ Liệu CSV & Mã Nguồn

```
foodhealthqa_demo/
├── app.py                         # Giao diện chính Streamlit (4 tabs phân tích & Data Hub CSV)
├── requirements.txt               # Danh sách thư viện cần thiết (Streamlit, Pandas, Pillow, OpenCV...)
├── test_all.py                    # Kịch bản kiểm thử toàn diện 5 tầng pipeline
├── test_perception.py             # Kiểm thử OCR ảnh tải lên, tách mã và chuẩn hóa INS
├── test_images.py                 # Kịch bản kiểm tra quét 3 ảnh mẫu thực nghiệm
├── test_csv_integration.py        # Kịch bản kiểm tra đọc và phân tích dữ liệu CSV
├── README.md                      # Tài liệu hướng dẫn sử dụng và phương pháp luận
├── data/                          # Kho dữ liệu CSV & JSON tích hợp
│   ├── incidents_train.csv        # Báo cáo sự cố thu hồi & mối nguy từ Food Hazard Detector
│   ├── fndds_nutrition.csv        # Thành phần dinh dưỡng định lượng USDA FNDDS
│   ├── openfoodfacts_vietnam.csv  # Nhãn sản phẩm & mã vạch Open Food Facts VN
│   ├── nhanes_personas.csv        # Hồ sơ lâm sàng người dùng từ NHANES CDC Hoa Kỳ
│   ├── foodon_entities.json       # Danh mục thuật ngữ FoodOn (CURIE, nhãn, quan hệ cha)
│   ├── additives_standard.csv     # Cơ sở tri thức phụ gia chuẩn hóa kèm mã FoodOn CURIE
│   ├── additives_mini.json        # Dữ liệu dự phòng JSON
│   └── personas.json              # Dữ liệu dự phòng JSON
├── assets/                        # Kho ảnh bao bì thực tế (Mì ăn liền, Lon nước ngọt cong, Bánh quy hộp)
│   ├── instant_noodles_ingredient_label.png
│   ├── beverage_can_ingredient_label.png
│   └── cookie_box_allergen_label.png
└── modules/
    ├── ocr_extractor.py           # Tầng 1: Tiền xử lý quang học, PaddleOCR, Phục hồi chính tả, Bounding Box
    ├── foodon.py                  # Nạp, tra cứu và kiểm chứng CURIE FoodOn
    └── pipeline.py                # Tầng 2, 3, 4: FoodOntoRAG Entity Linking, CSV Data Hub và Guardrail y tế
```
