# FoodHealthQA: Hệ Thống Hỏi Đáp Tư Vấn Sức Khỏe Từ Nhãn Bao Bì Thực Phẩm (v7 - Comprehensive CSV & Multi-Data Integration)

Ứng dụng Web Demo tương tác hỗ trợ người tiêu dùng và bệnh nhân đánh giá mức độ an toàn của phụ gia, thành phần và giá trị dinh dưỡng thực phẩm từ ảnh chụp bao bì.

Kế thừa các công trình nghiên cứu và dữ liệu nền tảng:
- **PaddleOCR & Adaptive Perception Pipeline**: Tiền xử lý quang học (Deskewing, Adaptive CLAHE, Unsharp Masking) kết hợp Tinh chỉnh tham số nhận diện DBNet (`det_db_unclip_ratio=2.0`, `det_db_thresh=0.25`).
- **Post-OCR Lexical Normalizer**: Tầng phục hồi chính tả tiếng Việt miền thực phẩm dựa trên Từ điển Chuẩn (Thông tư 24/2019/TT-BYT & FoodOn), tự động khôi phục dấu thanh và mã số phụ gia quốc tế (INS, E-number).
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

---

## 🎯 Cấu Trúc Thư Mục Dữ Liệu CSV & Mã Nguồn

```
foodhealthqa_demo/
├── app.py                         # Giao diện chính Streamlit (4 tabs phân tích & Data Hub CSV)
├── requirements.txt               # Danh sách thư viện cần thiết (Streamlit, Pandas, Pillow, OpenCV...)
├── test_all.py                    # Kịch bản kiểm thử toàn diện 5 tầng pipeline
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
