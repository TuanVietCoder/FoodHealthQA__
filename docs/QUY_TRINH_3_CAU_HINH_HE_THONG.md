# BÁO CÁO MÔ TẢ QUY TRÌNH THỰC HIỆN 3 CẤU HÌNH TRONG HỆ THỐNG FOODHEALTHQA

> **Tài liệu Kỹ thuật & Thiết kế Kiến trúc Hệ thống**  
> **Đề tài:** Hệ thống hỏi đáp tư vấn an toàn thực phẩm cá nhân hóa từ thành phần bao bì  
> **Cơ chế suy luận & tổng hợp:** Mô hình Ngôn ngữ Lớn (Google Gemini 3.8 Flash) kết hợp RAG Đa tầng  

---

## MỤC LỤC
1. [Tổng quan Đề tài & Bản chất của 3 Cấu hình Thử nghiệm](#1-tổng-quan-đề-tài--bản-chất-của-3-cấu-hình-thử-nghiệm)
2. [Vai trò của Mô hình Ngôn ngữ Lớn (LLM) trong Hệ thống](#2-vai-trò-của-mô-hình-ngôn-ngữ-lớn-llm-trong-hệ-thống)
3. [Bảng Ma trận So sánh 3 Cấu hình](#3-bảng-ma-trận-so-sánh-3-cấu-hình)
4. [Mô tả Chi tiết Quy trình Vận hành của Từng Cấu hình](#4-mô-tả-chi-tiết-quy-trình-vận-hành-của-từng-cấu-hình)
   - [4.1. Cấu hình A: LLM-Only (Mức sàn Đối chứng - Lower Bound)](#41-cấu-hình-a-llm-only-mức-sàn-đối-chứng---lower-bound)
   - [4.2. Cấu hình B: Ground Truth + RAG (Mức trần Tiềm năng - Upper Bound)](#42-cấu-hình-b-ground-truth--rag-mức-trần-tiềm-năng---upper-bound)
   - [4.3. Cấu hình C: OCR + RAG + LLM (Chu trình Ứng dụng Thực tế)](#43-cấu-hình-c-ocr--rag--llm-chu-trình-ứng-dụng-thực-tế)
5. [Ý nghĩa của Khâu Đóng gói Chỉ thị (Prompting) trong Đề tài](#5-ý-nghĩa-của-khâu-đóng-gói-chỉ-thị-prompting-trong-đề-tài)
6. [Hệ thống Đo lường & Các Chỉ số Đánh giá Khoa học](#6-hệ-thống-đo-lường--các-chỉ-số-đánh-giá-khoa-học)

---

## 1. Tổng quan Đề tài & Bản chất của 3 Cấu hình Thử nghiệm

Hệ thống **FoodHealthQA** giải quyết bài toán: Nhận diện thành phần thực phẩm từ ảnh chụp bao bì, chuẩn hóa thực thể và đối chiếu với kho tri thức y tế/pháp lý (mã INS/E-number, ngưỡng an toàn độc học JECFA ADI, quy định Thông tư 24/2019/TT-BYT, tiêu chuẩn quốc tế Codex STAN 192-1995, cảnh báo dị ứng, hồ sơ dinh dưỡng định lượng USDA FNDDS, và lịch sử sự cố an toàn thực phẩm). Cuối cùng, hệ thống kích hoạt chốt chặn an toàn (Guardrail) và điều khiển mô hình ngôn ngữ lớn (LLM) sinh câu trả lời tư vấn cá nhân hóa có trích dẫn nguồn xác thực.

Để đánh giá khoa học và khách quan tính hiệu quả của từng mắt xích trong chuỗi xử lý, hệ thống thiết lập **3 cấu hình thử nghiệm đối chứng (Ablation Study)**:

* **Cấu hình A (LLM-Only - Mô hình Ngôn ngữ thuần túy):** Đóng vai trò là **Mức sàn (Lower Bound)**. Mô hình trả lời hoàn toàn dựa vào tri thức nén sẵn có bên trong mạng nơ-ron của nó mà không được cấp bất kỳ tài liệu tra cứu nào. Cấu hình này giúp đo lường mức độ rủi ro, nguy cơ bịa đặt thông tin (hallucination) và khả năng phán đoán tự thân của AI khi không có hệ thống hỗ trợ.
* **Cấu hình B (Ground Truth + RAG - Tri thức Chuẩn hóa):** Đóng vai trò là **Mức trần (Upper Bound)**. Hệ thống nhận đầu vào là chuỗi thành phần đã được chuyên gia hiệu đính chính xác 100%, bỏ qua các lỗi quang học từ camera, sau đó kích hoạt toàn bộ cơ chế tra cứu và luật an toàn y tế. Cấu hình này phản ánh năng lực tối đa mà kiến trúc RAG và kho tri thức có thể đạt được trong điều kiện lý tưởng.
* **Cấu hình C (OCR + RAG + LLM - Pipeline Đầy đủ Thực tế):** Đại diện cho **Hệ thống Triển khai Thực tế**. Dữ liệu bắt đầu từ ảnh chụp nhãn mác (vốn có thể bị mờ nhòe, chói sáng, cong vênh), đi qua quá trình nhận diện quang học, sửa lỗi chính tả, liên kết thực thể đa tác tử, tra cứu tri thức và chốt chặn an toàn. So sánh Cấu hình C với Cấu hình B giúp đo lường chính xác **sự lan truyền sai số (Error Propagation)** từ khâu đọc ảnh đến chất lượng tư vấn cuối cùng.

---

## 2. Vai trò của Mô hình Ngôn ngữ Lớn (LLM) trong Hệ thống

Trong toàn bộ hệ thống, mô hình ngôn ngữ lớn được tích hợp là **Google Gemini 3.8 Flash**. Thay vì để mô hình đóng vai trò quyết định toàn bộ, hệ thống phân định rõ ràng ranh giới chức năng:

1. **Khâu Thẩm định & Logic An toàn (Do Hệ thống RAG & Guardrail đảm nhiệm):** 
   Việc kiểm tra xem một chất có vượt ngưỡng ADI hay không, có gây sốc phản vệ cho người dị ứng hay không, hoặc có vi phạm quy định của Bộ Y tế hay không hoàn toàn do các thuật toán đối soát dữ liệu và mô hình ràng buộc y khoa thực hiện.
2. **Khâu Tổng hợp & Giao tiếp (Do LLM đảm nhiệm):** 
   LLM đóng vai trò như một **"Bác sĩ tư vấn & Người phát ngôn"**:
   * Tiếp nhận các kết luận y khoa, chứng cứ pháp lý và dữ liệu dinh dưỡng do hệ thống cung cấp.
   * Chuyển hóa các bảng mã số phức tạp (như mã INS, nồng độ mg/kg, mã sự cố thu hồi) thành ngôn ngữ tự nhiên tiếng Việt, mạch lạc, dễ hiểu và thân thiện với người tiêu dùng.
   * Gắn đúng mã trích dẫn nguồn đối chứng ở từng nhận định, tuyệt đối không suy diễn thêm các tuyên bố y khoa ngoài bằng chứng.

---

## 3. Bảng Ma trận So sánh 3 Cấu hình

| Tiêu chí | Cấu hình A (LLM-Only) | Cấu hình B (GT + RAG) | Cấu hình C (OCR + RAG + LLM) |
| :--- | :--- | :--- | :--- |
| **Bản chất kỹ thuật** | Closed-book QA (Không RAG) | Open-book RAG (Văn bản sạch) | Full End-to-End Pipeline (Ảnh/OCR) |
| **Dữ liệu đầu vào** | Câu hỏi + Hồ sơ bệnh lý Persona | Text thành phần chuẩn (Ground Truth) | Ảnh nhãn qua PaddleOCR / Text có nhiễu |
| **Độ tự tin quang học ($S_{ocr}$)** | Không áp dụng ($N/A$) | Tuyệt đối ($1.0$) | Bị suy giảm ($0.80 - 0.95$) |
| **Hậu xử lý chính tả** | Bỏ qua | Không cần thiết | Có (`post_process_ocr_spelling`) |
| **Entity Linking** | Bỏ qua | Khớp chính xác 1-1 với KB | HybridRetriever 4 tác tử (BM25 + TF-IDF) |
| **Tra cứu Đa nguồn Tri thức** | Bỏ qua hoàn toàn | Đầy đủ (KB, FNDDS, Hazard, OFF) | Đầy đủ (Dựa trên thực thể linked từ OCR) |
| **Kiểm định Guardrail** | Không có | Chạy luật với độ tin cậy tuyệt đối | Chạy luật + Rủi ro sai số OCR |
| **Cơ chế từ chối (Abstention)**| Không có | Không kích hoạt (Đầy đủ dữ liệu) | Tự động kích hoạt khi $S_{total} < \tau$ |
| **Cấu trúc Prompt** | Zero-evidence (Chỉ hỏi + Persona) | Strict Grounded (Evidence + Nguồn KB) | Strict Grounded (Evidence + Nguồn KB) |
| **Xác thực Trích dẫn** | Không có mã trích dẫn | Bắt buộc kiểm tra `[KB:...]` | Bắt buộc kiểm tra `[KB:...]` |
| **Ý nghĩa Nghiên cứu** | **Mức sàn (Lower Bound)** | **Mức trần (Upper Bound)** | **Đo lường sai số thực tế (Real-world)** |

---

## 4. Mô tả Chi tiết Quy trình Vận hành của Từng Cấu hình

```
[Ảnh nhãn / Text thành phần + Câu hỏi + Hồ sơ lâm sàng Persona]
                              │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
 [ CẤU HÌNH A ]      [ CẤU HÌNH B ]      [ CẤU HÌNH C ]
 (LLM-Only / Sàn)     (GT + RAG / Trần)   (OCR + RAG / Thực tế)
       │                   │                   │
  Bỏ qua OCR/KB     Nạp Text chuẩn GT     Quét ảnh qua PaddleOCR
       │            (Sạch 100%, conf=1.0) (Có nhiễu ảnh, conf thấp)
       │                   │                   │
  Bỏ qua Linking    Entity Linking chuẩn  Hậu xử lý chính tả Post-OCR
       │                   │                   │
  Bỏ qua Guardrail  Tra cứu KB + Luật     HybridRetriever (BM25+TF-IDF)
       │                   │                   │
       │            Guardrail An toàn     Tra cứu KB + Guardrail An toàn
       │                   │              (Tính rủi ro sai số OCR)
       │                   │                   │
 Prompt Không Evidence  Prompt Đầy đủ Evidence Prompt Đầy đủ Evidence
  (Chỉ hỏi + Persona) (Evidence + Nguồn KB) (Evidence + Nguồn KB)
       │                   │                   │
       └───────────────────┼───────────────────┘
                              ▼
               [ Gọi LLM: Gemini 3.8 Flash ]
                              │
                              ▼
             [ Hậu kiểm Trích dẫn & Tách nhãn an toàn ]
                              │
                              ▼
          [ Xuất file kết quả đánh giá: predictions_config_X.jsonl ]
```

---

### 4.1. Cấu hình A: LLM-Only (Mức sàn Đối chứng - Lower Bound)

* **Ý nghĩa:** Đo lường xem nếu chỉ dựa vào dữ liệu học sẵn của mô hình ngôn ngữ thì câu trả lời có đủ an toàn cho người bệnh hay không.

#### Quy trình 4 bước thực hiện:
1. **Bước 1: Tiếp nhận Yêu cầu & Nhận diện Hồ sơ Bệnh lý**
   * Hệ thống tiếp nhận câu hỏi của người dùng và bối cảnh sức khỏe.
   * Quá trình phân tích ngữ cảnh sẽ trích xuất hồ sơ bệnh lý đặc thù của người dùng (ví dụ: người mắc bệnh suy thận mạn tính, đái tháo đường týp 2, phụ nữ trong thai kỳ, trẻ nhỏ 5 tuổi, hoặc người có tiền sử dị ứng đậu nành).
2. **Bước 2: Cô lập Tri thức (Bỏ qua hoàn toàn khâu RAG)**
   * Hệ thống cố tình không kết nối với bất kỳ cơ sở dữ liệu nào: không đọc ảnh bao bì, không liên kết thực thể và không tra cứu các tiêu chuẩn an toàn thực phẩm.
3. **Bước 3: Đóng gói Yêu cầu Suy đoán Tự do**
   * Hệ thống thiết lập một bản chỉ thị mang tính "đóng sách":
     * Thông báo cho LLM biết đây là tình huống tư vấn an toàn thực phẩm nhưng **hoàn toàn không cung cấp cơ sở dữ liệu tham chiếu**.
     * Cung cấp thông tin bệnh lý của người dùng cùng câu hỏi cần giải đáp.
     * Buộc mô hình phải tự huy động trí nhớ tiền huấn luyện để đưa ra lời khuyên và tự gán nhãn mức độ an toàn vào dòng cuối.
4. **Bước 4: Sinh Câu trả lời & Tách Nhãn An toàn**
   * Mô hình Gemini xử lý với nhiệt độ thấp để giảm tính ngẫu nhiên.
   * Hệ thống bóc tách nhãn mức độ an toàn được mô hình tự kết luận (như An toàn, Cảnh báo, hoặc Nguy hiểm) và lưu vào tệp dữ liệu kết quả để phục vụ chấm điểm.
* **Hạn chế:** Mô hình thường trả lời chung chung mang tính ngoại giao, không có căn cứ pháp lý, không trích dẫn được điều luật và có xu hướng đoán nhãn "An toàn" một cách chủ quan, gây rủi ro lớn trong y tế.

---

### 4.2. Cấu hình B: Ground Truth + RAG (Mức trần Tiềm năng - Upper Bound)

* **Ý nghĩa:** Đánh giá năng lực của toàn bộ hệ thống tri thức và luật an toàn khi loại bỏ hoàn toàn các lỗi sai sót do việc quét chữ từ bao bì gây ra.

#### Quy trình 6 bước thực hiện:
1. **Bước 1: Tiếp nhận Danh mục Thành phần Chuẩn xác 100%**
   * Thay vì dùng camera đọc ảnh, hệ thống nạp trực tiếp danh sách thành phần đã được kiểm định chính xác tuyệt đối từ bộ dữ liệu chuẩn (văn bản sạch, không mất dấu, không lỗi chính tả).
2. **Bước 2: Phân tách & Ánh xạ Thực thể Chuẩn tắc**
   * Hệ thống phân rã đoạn văn bản thành các thành phần và phụ gia đơn lẻ.
   * Tiến hành đối chiếu trực tiếp từng chất với Cơ sở Tri thức Phụ gia chuẩn để định danh chính xác mã số quốc tế (INS/E-number) và tên khoa học.
3. **Bước 3: Tra cứu Đa nguồn Tri thức Chuyên sâu**
   * Hệ thống đồng thời trích xuất các bằng chứng khoa học vững chắc:
     * **Liều lượng chấp nhận:** Ngưỡng nạp hàng ngày an toàn từ Ủy ban Chuyên gia hỗn hợp FAO/WHO về Phụ gia Thực phẩm (JECFA ADI).
     * **Quy chuẩn pháp lý:** Bảng danh mục cho phép và giới hạn tối đa theo Thông tư 24/2019/TT-BYT và tiêu chuẩn Codex Quốc tế.
     * **Cảnh báo nguy cơ:** Các phát hiện khoa học độc tính mới nhất từ Cơ quan An toàn Thực phẩm Châu Âu (EFSA).
4. **Bước 4: Đánh giá Ràng buộc Phủ định & Chống chỉ định Y khoa**
   * Chạy mô hình kiểm định an toàn đối chiếu giữa các chất đã nhận diện với hồ sơ bệnh lý người dùng:
     * Phát hiện các xung đột y khoa (ví dụ: người bệnh Phenylketon niệu không được dùng chất tạo ngọt Aspartame; người tăng huyết áp cần kiểm soát lượng muối có trong bột ngọt và bột nở).
     * Rà soát 9 nhóm dị nguyên phổ biến (sữa, trứng, đậu nành, lúa mì, hải sản...) và các dẫn xuất chuyển hóa.
5. **Bước 5: Đóng gói Ngữ cảnh Bằng chứng Nghiêm ngặt**
   * Thiết lập bản chỉ thị cho LLM theo nguyên tắc "mở sách":
     * Đưa toàn bộ các đoạn bằng chứng đã xác thực ở Bước 3 và Bước 4 vào làm tài liệu học tập duy nhất cho câu trả lời.
     * Cung cấp danh sách các mã định danh tài liệu được phép trích dẫn.
     * Ra quy định bắt buộc: Mọi câu khẳng định y khoa phải đi kèm trích dẫn nguồn trong ngoặc vuông; không được thêm thắt thông tin nằm ngoài bằng chứng.
6. **Bước 6: Khởi tạo Lời tư vấn & Xác thực Trích dẫn**
   * Mô hình Gemini tiếp nhận khối tri thức, tổng hợp thành lời khuyên mạch lạc.
   * Hệ thống kích hoạt thuật toán hậu kiểm trích dẫn để xác minh: Mọi trích dẫn trong ngoặc vuông có thực sự tồn tại trong danh mục nguồn được phép hay không. Sau đó lưu lại câu trả lời và nhãn an toàn đã thẩm định.

---

### 4.3. Cấu hình C: OCR + RAG + LLM (Chu trình Ứng dụng Thực tế)

* **Ý nghĩa:** Đây là mô hình hoạt động đầy đủ trong đời thực của ứng dụng khi người dùng chụp ảnh sản phẩm từ điện thoại di động. Cấu hình này giúp đo lường mức độ ảnh hưởng của lỗi đọc chữ đối với chất lượng tư vấn y khoa.

#### Quy trình 7 bước thực hiện:
1. **Bước 1: Thu nhận Quang học & Trích xuất Ký tự từ Ảnh Bao bì**
   * Mô hình nhận dạng chữ quang học (PaddleOCR) tiếp nhận ảnh chụp nhãn mác thực tế (lon nước, gói bánh, hộp sữa).
   * Do đặc thù bề mặt bao bì bị bóng, cong hoặc chữ quá nhỏ, hệ thống thu nhận chuỗi văn bản thô kèm theo điểm số độ tin cậy nhận dạng (đo lường mức độ rõ nét của ảnh).
2. **Bước 2: Phục hồi Ngữ nghĩa & Chuẩn hóa Chính tả Sau OCR**
   * Chuỗi chữ thô được đưa qua thuật toán phục hồi chính tả chuyên ngành thực phẩm:
     * Khôi phục các thanh dấu tiếng Việt bị mờ nhạt (ví dụ: phân biệt "dầu" và "đậu", "bơ" và "bò").
     * Hiệu chỉnh các lỗi nhầm lẫn quang học kinh điển trong mã số phụ gia (như chữ `O` bị quét nhầm thành số `0`, hoặc chuỗi `INS 50O` được tự động chuyển thành `INS 500`).
3. **Bước 3: Chuẩn hóa Thực thể Đa tác tử Dưới điều kiện Bất định**
   * Vì chữ quét được có thể vẫn còn khiếm khuyết, hệ thống kích hoạt cơ chế 4 tác tử thông minh:
     * **Tác tử Truy xuất lai:** Tìm kiếm kết hợp theo cả 2 hướng: trùng khớp từ khóa bề mặt (BM25) và tương đồng ngữ nghĩa mờ trong không gian vector (TF-IDF/SBERT) để không bỏ sót các dạng viết tắt hoặc tên thương mại.
     * **Tác tử Chọn lọc:** Ưu tiên mã số chính thức và các khái niệm cụ thể.
     * **Tác tử Đánh giá:** Chấm điểm mức độ tương đồng bản chất hóa học giữa từ quét được và thực thể chuẩn.
     * **Tác tử Sinh từ đồng nghĩa:** Nếu độ tương đồng còn thấp, tác tử sẽ tự động mở rộng các dạng danh pháp khác dựa trên cây phả hệ Bản thể học Thực phẩm (FoodOn) để tìm kiếm lượt hai.
4. **Bước 4: Tra cứu Đa nguồn & Tính toán Ràng buộc Số học**
   * Đối soát thực thể với kho quy chuẩn an toàn, bảng liều lượng chấp nhận ADI và hệ thống cảnh báo sự cố thực phẩm toàn cầu (phát hiện các lô hàng từng bị thu hồi do nhiễm khuẩn hoặc dị nguyên ẩn).
   * Tích hợp dữ liệu dinh dưỡng định lượng USDA FNDDS: Chuyển đổi các giá trị dinh dưỡng thành hàm chỉ thị khoảng để kiểm tra xem lượng Natri, Calo, Đường trên khẩu phần có vượt quá ngưỡng cho phép của bệnh nhân hay không.
5. **Bước 5: Chốt chặn An toàn & Kích hoạt Cơ chế Từ chối Thông minh**
   * Hệ thống tổng hợp điểm tin cậy kép:
     $$\text{Độ tin cậy Tổng thể} = \text{Độ rõ nét của Ảnh OCR} \times \text{Độ chắc chắn khi Liên kết Thực thể}$$
   * **Cơ chế Từ chối có Nguyên tắc (Principled Abstention):** Nếu điểm tin cậy tổng thể rơi xuống dưới ngưỡng an toàn (do ảnh chụp quá mờ, bao bì rách góc, hoặc không nhận diện được các chất nghi vấn), hệ thống chủ động kích hoạt trạng thái **Từ chối kết luận (ABSTAIN)**. Ứng dụng sẽ từ chối đưa ra nhận định vội vàng và cảnh báo người dùng chụp lại ảnh rõ nét hơn hoặc tham vấn chuyên gia y tế.
6. **Bước 6: Đóng gói Ngữ cảnh Tri thức Thực tế & Khởi tạo Lời tư vấn**
   * Khi dữ liệu đạt chuẩn an toàn, toàn bộ chứng cứ thu thập được cùng các cảnh báo lâm sàng được đưa vào bản chỉ thị nghiêm ngặt gửi tới mô hình Gemini 3.8 Flash.
   * LLM tổng hợp thành văn bản tư vấn tiếng Việt hoàn chỉnh, dẫn chứng từng mã nguồn đối soát.
7. **Bước 7: Phân tích Sự Lan truyền Sai số (Error Propagation)**
   * Lưu trữ kết quả và đối chiếu chéo giữa Cấu hình C và Cấu hình B:
     * Xác định xem các lỗi đọc sai chữ của OCR đã làm giảm bao nhiêu phần trăm độ chính xác.
     * Đo lường chỉ số **Tỷ lệ An toàn Giả mạo (False Safe Rate):** Xác định có bao nhiêu trường hợp sản phẩm thực tế có chứa chất cấm đối với bệnh nhân nhưng do khâu OCR đọc sót nên hệ thống lại kết luận nhầm là "An toàn". Đây là chỉ số quan trọng nhất về mặt an toàn tính mạng trong nghiên cứu y tế.

---

## 5. Ý nghĩa của Khâu Đóng gói Chỉ thị (Prompting) trong Đề tài

Khâu đóng gói chỉ thị trước khi gửi sang cho LLM không đơn thuần là việc ghép nối câu chữ, mà đóng vai trò trọng tâm trong việc hiện thực hóa các yêu cầu nghiên cứu của đề tài:

1. **Là ranh giới phân tách 3 cấu hình thử nghiệm:**
   Nếu không có khâu này, hệ thống không thể tạo ra sự khác biệt giữa "Mô hình ngôn ngữ tự đoán mò" (Cấu hình A) và "Mô hình ngôn ngữ có bằng chứng bảo chứng" (Cấu hình B và C).
2. **Hiện thực hóa Nghiên cứu Khảo sát Mẫu Chỉ thị (Prompt Template Ablation):**
   Đề tài chứng minh được rằng việc áp dụng các quy tắc ràng buộc nghiêm ngặt (buộc trích dẫn nguồn, cấm suy diễn ngoài tài liệu, yêu cầu định dạng dòng nhãn an toàn riêng biệt) giúp giảm thiểu hiện tượng bịa đặt thông tin của AI tới mức tối đa.
3. **Hiện thực hóa Kỹ thuật Sinh văn bản có Căn cứ (Grounded Generation):**
   Đảm bảo rằng toàn bộ câu chữ mà người bệnh đọc được đều có nguồn gốc xuất phát từ các văn bản quy chuẩn chính thống của Bộ Y tế, FAO/WHO và EFSA, biến LLM từ một công cụ sinh chữ ngẫu nhiên thành một trợ lý y tế có kỷ luật.

---

## 6. Hệ thống Đo lường & Các Chỉ số Đánh giá Khoa học

Hệ thống tích hợp bộ công cụ đo lường toàn diện nhằm đánh giá kết quả của 3 cấu hình qua 3 góc độ:

### 1. Góc độ Chất lượng Ngôn ngữ & Độ Khớp Văn bản:
* **Độ khớp Tuyệt đối (Exact Match - EM):** Tỷ lệ câu trả lời của hệ thống giống chính xác 100% từng từ với câu trả lời chuẩn do chuyên gia y tế biên soạn.
* **Chỉ số Token F1:** Đo lường sự hài hòa giữa độ chuẩn xác (Precision) và độ bao phủ (Recall) ở cấp độ từ vựng tiếng Việt, đánh giá xem câu trả lời có truyền tải đủ ý cốt lõi hay không.

### 2. Góc độ Trung thực & Trích dẫn Nguồn Khoa học:
* **Độ chuẩn xác Trích dẫn (Citation Precision):** Tỷ lệ các nguồn tham khảo được mô hình nêu ra nằm đúng trong danh mục tài liệu đối chứng của chuyên gia.
* **Độ bao phủ Trích dẫn (Citation Recall):** Tỷ lệ các nguồn tài liệu quy chuẩn quan trọng được mô hình phát hiện và viện dẫn đầy đủ.
* **Tính Trung thực (Faithfulness) & Tính Bám sát (Groundedness):** Đo lường tỷ lệ các câu khẳng định trong bài viết được bảo chứng trực tiếp bởi đoạn trích bằng chứng, chứng minh hệ thống không tự ý sáng tác kiến thức.

### 3. Góc độ An toàn Lâm sàng Y tế:
* **Độ chính xác Nhãn An toàn (Safety Label Accuracy):** Đánh giá mức độ phân loại chuẩn xác 5 trạng thái an toàn: An toàn (SAFE), Cần lưu ý (CAUTION), Cảnh báo nguy cơ (WARNING), Báo động đỏ chống chỉ định (CRITICAL_ALERT), và Từ chối trả lời (ABSTAIN).
* **Tỷ lệ Bỏ sót Nguy cơ (False Safe Rate):** Đo lường tỷ lệ bệnh nhân gặp nguy hiểm khi sản phẩm thực tế độc hại/chống chỉ định nhưng hệ thống lại phán đoán là an toàn. Mục tiêu tối thượng của đề tài là ép chỉ số này về mức $0\%$.
* **Ma trận Nhầm lẫn (Confusion Matrix):** Bảng thống kê chi tiết sự dịch chuyển giữa các mức độ cảnh báo, giúp chỉ ra rõ hệ thống hay nhầm lẫn giữa mức độ nào với mức độ nào để tiếp tục tinh chỉnh thuật toán.
