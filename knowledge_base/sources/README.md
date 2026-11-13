# Nguồn tham chiếu (knowledge_base/sources)

| Thư mục | Nội dung | Quy tắc |
|---|---|---|
| `raw/vn/` | Văn bản gốc của Việt Nam (TT 24/2019/TT-BYT, 09/VBHN-BYT) | Không sửa tay |
| `raw/external/` | Tài liệu gốc quốc tế tải về (Codex, EFSA, FDA, IARC, EU, WHO...) và `jecfa/` (167 trang JECFA) | Do `fetch_external_sources.py`, `fetch_jecfa*.py` tải; không sửa tay |
| `derived/` | Dữ liệu trích/kiểm chứng từ `raw/` (`external_claim_verification.json`, `jecfa_adi_verification*.json`, `codex_192_ins_codes.json`, `tt24_*.json`) | Sinh lại bằng script `verify_external_claims.py`, `extract_tt24_*.py`, `fetch_jecfa*.py` |
| `catalog/` | `source_citation_catalog.json` (danh mục khóa trích dẫn, chỉ 1 bản) và `external_sources_manifest.json` (URL, thời điểm tải, SHA-256) | `build_source_citation_catalog.py` chỉ thêm khóa còn thiếu, không ghi đè |

Thêm nguồn mới: đặt tệp vào `raw/`, thêm vào `fetch_external_sources.py`, thêm claim ở `verify_external_claims.py`,
rồi chạy lại chuỗi `add_authority_records_v5 -> create_benchmark_v5 -> verify_v5_answers -> finalize_benchmark_v5`.
