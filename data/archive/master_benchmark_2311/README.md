# Kho dữ liệu gốc QA Benchmark (Master Pool - 2.311 bản ghi)

Thư mục này chứa tập dữ liệu gốc 2.311 câu hỏi được sinh ra từ 489 ảnh bao bì sản phẩm (phiên bản v5).

## Mục đích & Vai trò
- **Không dùng trực tiếp để đánh giá hay demo**: Hệ thống FoodHealthQA hiện tại chỉ sử dụng duy nhất **Bộ đánh giá chuẩn 750 câu** tại `data/foodhealthqa_eval_set.jsonl`.
- **Vai trò lưu trữ & tái lập (Master Pool / Full Backup)**: Toàn bộ 750 câu hỏi của `foodhealthqa_eval_set.jsonl` được trích xuất phân tầng từ kho 2.311 câu này thông qua script `scripts_research/build_foodhealthqa_subset.py`.
- Tệp này được bảo lưu tại đây nhằm phục vụ việc kiểm tra tính bất biến (invariants) và tái lập mẫu khi cần thiết.

## Danh sách tệp
- `qa_benchmark.jsonl`: Bản nguồn đầy đủ 2.311 bản ghi định dạng JSON Lines.
- `qa_benchmark.csv`: Bản nguồn định dạng CSV.
- `qa_benchmark_persona_conditions.csv`: Bảng mapping điều kiện lâm sàng / persona trung gian trong quá trình xây dựng v5.

