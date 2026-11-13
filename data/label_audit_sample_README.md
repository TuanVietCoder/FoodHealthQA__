# Danh sách mẫu kiểm tra phiên âm nhãn

File: `data/label_audit_sample.csv` (100 ảnh, chọn ngẫu nhiên có phân tầng, seed 20261108).
Mẫu này chi phối **309** bản ghi tính điểm trên tổng 1588; 70 ảnh ưu tiên nhãn rủi ro cao
(UNSAFE / CRITICAL_ALERT / CAUTION), phần còn lại là ảnh ngẫu nhiên.

## Cách làm cho mỗi dòng
1. Mở ảnh ở cột `image_file` (thư mục `assets/`).
2. So với các cột `transcribed_*`. Chú ý cột `what_to_check`.
3. Điền Y/N vào 4 cột `*_ok`. Nếu N: ghi vào `what_is_wrong` và chép đoạn đúng vào `corrected_text`.

## Cách đọc kết quả
- Mỗi lỗi ở `ingredient` hoặc `warning` có thể làm sai nhãn dị nguyên của các `record_ids` đi kèm.
- Mỗi mã INS bị chép sai hoặc sót làm sai nhãn phụ gia và nhãn nhóm dễ tổn thương.
- Nếu tỷ lệ ảnh có lỗi vượt khoảng 5%, nên kiểm tra thêm toàn bộ các ảnh có bản ghi UNSAFE/CRITICAL_ALERT.
