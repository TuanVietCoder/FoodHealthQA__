# Báo cáo Trạng thái Benchmark FoodHealthQA

## 1. Tệp và vai trò hiện hành

| Tệp | Định dạng | Số bản ghi | Vai trò |
| :--- | :---: | ---: | :--- |
| `data/qa_benchmark_gold_v3.jsonl` | JSON Lines | 1.868 | **Đầu vào đánh giá mặc định** |
| `data/qa_benchmark_gold_v3.csv` | CSV (UTF-8 BOM) | 1.868 | Bản tham chiếu bảng tính |
| `scripts_research/rule_engine_audit_final.jsonl` | JSON Lines | 1.868 | Kết quả kiểm chứng rule engine |

### Phân bổ theo miền (Domains):

| Domain | Mô tả | Số bản ghi |
|--------|--------|-----------|
| `additive_regulation` | Đánh giá phụ gia INS/E-number | 437 |
| `allergen_detection` | Phát hiện dị nguyên & nhiễm chéo | 386 |
| `insufficient_data` | Từ chối có trách nhiệm (Abstain) | 960 |
| `vulnerable_groups` | Nhóm dễ tổn thương - trẻ em, thai kỳ | 85 |

### Phân bổ theo nhãn an toàn (Safety Labels):

| Nhãn | Số bản ghi | Tỷ lệ |
|------|-----------|-------|
| ABSTAIN | 960 | 51.4% |
| SAFE | 473 | 25.3% |
| UNSAFE | 384 | 20.6% |
| CAUTION | 49 | 2.6% |
| CRITICAL_ALERT | 2 | 0.1% |

---

## 2. Phạm vi nguồn dữ liệu

Benchmark sử dụng duy nhất các nguồn thẩm quyền quốc tế và Việt Nam sau:

**Phụ gia thực phẩm:**
- `CODEX_STAN_192_1995` — Codex General Standard for Food Additives
- `VN_TT24_2019` — Thông tư 24/2019/TT-BYT (Bộ Y tế VN)
- `EFSA_OPINIONS` — European Food Safety Authority
- `JECFA_EVALUATIONS` — Joint FAO/WHO Expert Committee on Food Additives

**Dị nguyên thực phẩm:**
- `WHO_JEMRA_PRIORITY_ALLERGENS` — WHO/FAO JEMRA Priority Allergens
- `FAO_WHO_JEMRA_PART_3` — FAO/WHO JEMRA Part 3 (PAL guidelines)
- `US_FDA_FALCPA` — FDA Food Allergen Labeling and Consumer Protection Act

**Phạm vi KHÔNG bao gồm:** Hướng dẫn lâm sàng dinh dưỡng (ADA, KDOQI, KDIGO, WHO Healthy Diet) — đã loại bỏ khỏi v3 do nằm ngoài phạm vi KB phụ gia/dị nguyên.

---

## 3. Phương pháp xây dựng và kiểm chứng

### 3.1 Quy trình label

Labels được sinh TỪ knowledge base (KB) theo rule engine — không phải từ domain hay template:

1. **Định nghĩa tiêu chí trước** (xem `docs/TIEU_CHI_LABEL_GOLD_STANDARD.md`)
2. **KB từ nguồn thẩm quyền** — INS code → ADI → health_considerations (EFSA/JECFA/VN)
3. **Rule engine sinh label** — deterministic, từ KB:
   - `additive_regulation`: KB concern `general_population`? → CAUTION : SAFE
   - `allergen_detection`: allergen trực tiếp → UNSAFE; nhiễm chéo → CRITICAL_ALERT
   - `insufficient_data`: thiếu hàm lượng định lượng → ABSTAIN
   - `vulnerable_groups`: KB concern theo persona → CAUTION : SAFE
4. **QA content** viết theo label — không phải ngược lại

### 3.2 Kết quả kiểm chứng rule engine

Tệp: `scripts_research/rule_engine_audit_final.jsonl`

| Trạng thái | Số bản ghi | Tỷ lệ |
|-----------|-----------|-------|
| MATCH | 907 | 99.9% (non-ABSTAIN) |
| MISMATCH_ALLERGEN_EDGE | 1 | 0.1% |
| CANNOT_DETERMINE (ABSTAIN) | 960 | 51.4% tổng |

**1 edge case còn lại (documented):** FOODQA_G3_0468 — sô-cô-la đen có "có thể chứa sữa" nằm trong ngoặc của sub-ingredient, trong khi hạt hạnh nhân 15% xuất hiện sau dấu đóng ngoặc. Rule engine bị split sai tại cross-contamination keyword này.

### 3.3 Các lỗi đã sửa

| Loại | Số bản ghi | Mô tả |
|------|-----------|-------|
| Scope reduction | 241 | Bỏ ADA/KDOQI/KDIGO/WHO_HEALTHY_DIET (ngoài phạm vi KB) |
| CAUTION → SAFE | 46 | Template hallucination: answer cite sai INS Southampton không có trong sản phẩm |
| CRITICAL_ALERT → UNSAFE | 101 | Allergen trực tiếp trong thành phần bị nhầm là nhiễm chéo |
| INS 171 SAFE → CAUTION | 8 | EFSA 2021 không xác lập ADI được, EU ban 2022 |
| INS 211+300 SAFE → CAUTION | 2 | Nguy cơ benzene khi đồng tồn tại (FDA 2006) |

---

## 4. Cách chạy

```powershell
# Chạy rule engine audit
python scripts_research/rule_engine_audit_final.jsonl

# Đánh giá ablation 3 cấu hình
.\venv\Scripts\python.exe .\scripts_research\run_evaluation.py --baseline
```
