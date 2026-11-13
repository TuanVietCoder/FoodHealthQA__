# Quy tắc Gán Nhãn Thử nghiệm — FoodHealthQA

**Phiên bản:** 1.0  
**Ngày:** 2026-11-01  
**Mục đích:** Mô tả heuristic gán nhãn để xây dựng bộ dữ liệu silver/pre-gold. Tài liệu này chưa được chuyên gia độc lập xác nhận và không chứng minh nhãn, đáp án hay kết luận sức khỏe là đúng.

> **Giới hạn sử dụng:** Không gọi các bản ghi tạo theo quy tắc này là expert-validated gold. `SAFE`, `UNSAFE`, `CAUTION`, `CRITICAL_ALERT` là nhãn đầu ra của tác vụ, không phải chẩn đoán hay đảm bảo sản phẩm an toàn. Chỉ kết luận điều quan sát được trên nhãn; nếu thiếu dữ liệu thì dùng `ABSTAIN`. Kết quả đánh giá trên bộ này chỉ là exploratory weak-label metrics.

---

## 1. Nguyên tắc nền tảng

### 1.1 Label phản ánh nội dung, không phản ánh domain

Label KHÔNG được suy ra từ loại câu hỏi (domain). Label phải được suy ra từ **bằng chứng cụ thể** cho **sản phẩm cụ thể** với **persona cụ thể**.

```
SAI:  domain = additive_regulation → label = SAFE  (hardcode)
ĐÚNG: INS 171 + healthy adult + EFSA cannot establish ADI → label = CAUTION
```

### 1.2 Thứ bậc bằng chứng (Evidence Tier)

Khi các nguồn xung đột nhau, ưu tiên theo thứ tự:

| Tier | Nguồn | Ví dụ | Trọng số |
|------|--------|--------|----------|
| T1 | Cơ quan đánh giá rủi ro quốc gia/quốc tế có thẩm quyền nhất | EFSA, JECFA/WHO, US FDA | Cao nhất |
| T2 | Cơ quan phân loại carcinogen | IARC (Group 1, 2A, 2B) | Cao |
| T3 | Quy chuẩn pháp lý VN | TT24/2019/TT-BYT | Trung bình |
| T4 | Quy chuẩn pháp lý khác | Codex, EU, US | Tham chiếu |
| T5 | Hướng dẫn lâm sàng | KDOQI, ADA, EFSA clinical | Cao với nhóm bệnh lý |

**Nguyên tắc xung đột pháp lý:** Nếu VN (T3) cho phép nhưng T1 đưa ra kết luận bảo vệ sức khỏe mạnh (ví dụ: EFSA "cannot establish ADI", EU cấm dựa trên T1), ưu tiên T1 cho label. Ghi rõ trong `note`.

### 1.3 Thứ tự ưu tiên label

```
CRITICAL_ALERT > UNSAFE > CAUTION > SAFE > ABSTAIN
```

Ngoại lệ: ABSTAIN ghi đè SAFE và CAUTION khi **thiếu dữ liệu cần thiết** để kết luận.

---

## 2. Định nghĩa từng label

### SAFE
**Điều kiện:** Tất cả các điều kiện sau đều đúng:
1. Mọi phụ gia trong sản phẩm đều nằm trong danh mục cho phép (Codex STAN 192-1995 và/hoặc TT24/2019)
2. Không có kết luận T1/T2 về rủi ro đáng kể **với nhóm dân số của persona được hỏi**
3. Không phát hiện allergen liên quan đến persona được hỏi

**Chú ý:** "Được phép theo luật" là điều kiện cần nhưng KHÔNG đủ nếu có T1/T2 concern.

`SAFE` trong bộ dữ liệu chỉ có thể có nghĩa là không phát hiện yếu tố kích hoạt theo quy tắc và thông tin nhãn đang có. Không được diễn giải thành bảo đảm không có dị nguyên ẩn, không có nhiễm chéo, phù hợp với mọi cá nhân hoặc tuân thủ mức sử dụng pháp lý.

---

### CAUTION
**Điều kiện:** Ít nhất một trong các điều kiện sau:

**C1 — Phụ gia có health concern từ T1/T2 với nhóm dân số của persona:**
- T1 (EFSA/JECFA) kết luận "cannot establish ADI" hoặc "không thể xác định ngưỡng an toàn" → bất kể persona
- T1 ban/hạn chế nghiêm ngặt dựa trên bằng chứng mới (ví dụ: EU cấm do genotoxicity) → bất kể VN còn cho phép
- IARC Group 2A hoặc 2B với bằng chứng đủ liên quan đến mức tiêu thụ thực phẩm bình thường

**C2 — Nhóm dễ tổn thương + phụ gia có mối quan hệ lâm sàng đã được ghi nhận:**
- CKD + phosphate (INS 338, 339, 341, 450, 452): KDOQI 2020
- Trẻ dưới 6 tuổi + azo dyes (INS 102, 104, 110, 122, 124, 129): EFSA 2008/EU Annex V
- Hen suyễn + sulfite (INS 220, 221): EFSA 2016/2022
- Trẻ dưới 6 tuổi + nitrite (INS 249, 250): EFSA 2017 (methemoglobin)
- PKU + aspartame (INS 951): FDA/EFSA PKU labeling requirement

**C3 — Rủi ro tương tác hóa học có bằng chứng:**
- INS 211 (sodium benzoate) + INS 300 (ascorbic acid) đồng thời trong sản phẩm → nguy cơ tạo benzene (FDA 2006). Chỉ áp dụng khi sản phẩm có CẢ HAI INS này.

---

### UNSAFE
**Điều kiện:**
- Allergen được **khai báo trực tiếp** trong danh sách thành phần của sản phẩm (không phải trong cảnh báo nhiễm chéo)
- Allergen phù hợp với nhóm dị ứng của persona được hỏi
- Nguồn: WHO JEMRA, FAO JEMRA, US FDA FALCPA

Đây là phân loại dựa trên thành phần được khai báo, không phải đánh giá ngưỡng phản ứng dị ứng của cá nhân. Không suy diễn rằng không thấy khai báo nghĩa là không có dị nguyên.

**Phân biệt với CRITICAL_ALERT:** Allergen xuất hiện trong thành phần chính (ingredient list) → UNSAFE. Allergen chỉ xuất hiện trong câu cảnh báo nhiễm chéo → CRITICAL_ALERT.

---

### CRITICAL_ALERT
**Điều kiện:**
- Allergen của persona **chỉ** xuất hiện trong cảnh báo nhiễm chéo (cross-contamination warning), không có trong thành phần chính
- Cảnh báo nhiễm chéo bao gồm các cụm từ: "sản xuất chung dây chuyền", "trên cùng thiết bị", "có thể chứa", "may contain", v.v.

**Lưu ý:** Nếu sản phẩm vừa có allergen trực tiếp vừa có cảnh báo nhiễm chéo cho **cùng một allergen** → UNSAFE (trực tiếp > nhiễm chéo).

---

### ABSTAIN
**Điều kiện:** Ít nhất một trong các điều kiện sau:
- Nhãn không ghi hàm lượng (ML) mà safety assessment đòi hỏi phải có hàm lượng
- Câu hỏi liên quan đến dinh dưỡng (đường, natri, calo) — nằm ngoài phạm vi KB phụ gia
- Không đủ thông tin từ nhãn sản phẩm để kết luận

---

## 3. Áp dụng vào 5 phụ gia đang tranh luận

### INS 171 (Titanium Dioxide — TiO₂)

| Yếu tố | Nội dung |
|--------|----------|
| Regulatory VN | TT24/2019: được phép |
| EFSA 2021 | "cannot establish ADI" do genotoxicity concern |
| EU Regulation 2022/63 | Cấm hoàn toàn trong thực phẩm |
| Tier áp dụng | T1 (EFSA) > T3 (VN) |
| **Label với healthy adult** | **CAUTION** |
| Lý do | T1 không thể xác định ngưỡng an toàn; EU cấm dựa trên T1. "Được phép ở VN" không đủ để gán SAFE khi T1 chưa có ADI. |

→ **Benchmark hiện tại SAI** cho 8 records này. Cần sửa → CAUTION.

---

### INS 211 (Sodium Benzoate)

| Yếu tố | Nội dung |
|--------|----------|
| Regulatory VN | TT24/2019: được phép |
| FDA 2006 | Benzene formation khi kết hợp với ascorbic acid (INS 300) |
| Điều kiện | Chỉ có rủi ro khi sản phẩm đồng thời chứa INS 300 |
| **Label khi CÓ INS 300** | **CAUTION** (quy tắc C3) |
| **Label khi KHÔNG có INS 300** | **SAFE** |

→ Benchmark cần kiểm tra từng record: sản phẩm có INS 300 không?

---

### INS 320 (BHA — Butylated Hydroxyanisole)

| Yếu tố | Nội dung |
|--------|----------|
| Regulatory VN | TT24/2019: được phép |
| IARC | Group 2B (possibly carcinogenic — bằng chứng hạn chế) |
| EFSA 2011 | Đánh giá lại, TĂNG ADI từ 0.5 lên 1.0 mg/kg bw/day → EFSA xác nhận an toàn trong giới hạn |
| Tier áp dụng | T1 (EFSA) xác nhận an toàn trong ADI; T2 (IARC 2B) có nhưng bằng chứng yếu hơn |
| **Label với healthy adult** | **SAFE** với lưu ý; không đủ cơ sở CAUTION vì EFSA 2011 xác nhận an toàn |

→ **KB cần sửa**: xóa `condition=general_population` cho INS 320. Benchmark SAFE là **đúng**. IARC 2B không đủ để override EFSA ADI còn hiệu lực.

---

### INS 150d (Caramel IV — chứa 4-MEI)

| Yếu tố | Nội dung |
|--------|----------|
| Regulatory VN | TT24/2019: được phép |
| IARC | 4-MEI Group 2B (primarily NTP rodent studies) |
| EFSA 2011 | "Current dietary exposure to caramel colours is not a safety concern" |
| Tier áp dụng | T1 (EFSA) xác nhận không lo ngại; T2 (IARC 2B) bằng chứng từ động vật, không ngoại suy trực tiếp |
| **Label với healthy adult** | **SAFE** |

→ **KB cần sửa**: xóa `condition=general_population` cho INS 150d. Benchmark SAFE là **đúng**.

---

### INS 250 (Sodium Nitrite)

| Yếu tố | Nội dung |
|--------|----------|
| Regulatory VN | TT24/2019: được phép |
| EFSA 2017 | Rủi ro methemoglobin chủ yếu ở trẻ nhỏ/sơ sinh, không phải người lớn |
| Người lớn khỏe mạnh | Nitrite → nitric oxide (sinh lý bình thường); rủi ro nitrosamine liên quan đến thịt chế biến, không phải INS 250 đơn lẻ |
| Tier áp dụng | T1 (EFSA 2017) concern chính cho trẻ nhỏ, không cho healthy adult |
| **Label với healthy adult** | **SAFE** |
| **Label với trẻ dưới 6 tuổi** | **CAUTION** (đã đúng trong KB) |

→ **KB cần sửa**: xóa `condition=general_population` cho INS 250, chỉ giữ `condition=infants_young_children`. Benchmark SAFE là **đúng** cho healthy adult.

---

## 4. Tóm tắt hành động

| Phụ gia | KB sửa | Benchmark sửa | Kết quả |
|---------|--------|---------------|---------|
| INS 171 | Giữ `general_population` | **Sửa 8 records SAFE → CAUTION** | Rule engine MATCH |
| INS 211 | Sửa rule: chỉ CAUTION khi CÓ INS 300 | Sửa records có INS 211 + INS 300 → CAUTION | Rule engine MATCH |
| INS 320 | **Xóa `general_population`** | Không cần sửa (SAFE đúng) | Rule engine MATCH |
| INS 150d | **Xóa `general_population`** | Không cần sửa (SAFE đúng) | Rule engine MATCH |
| INS 250 | **Xóa `general_population`** | Không cần sửa (SAFE đúng) | Rule engine MATCH |

---

## 5. Quy trình tạo QA record thử nghiệm

```
1. Chọn sản phẩm + đọc thành phần
2. Xác định: INS codes, allergens, persona
3. Chạy rule engine → nhận derived_label
4. Nếu derived_label = CANNOT_DETERMINE → domain = insufficient_data, label = ABSTAIN
5. Viết câu hỏi phù hợp persona
6. Viết reference_answer DỰA TRÊN derived_label:
   - SAFE: "phụ gia X được phép theo Codex/TT24, trong giới hạn ADI, an toàn"
   - CAUTION: "phụ gia X được phép NHƯNG [source] ghi nhận [concern] — cần lưu ý"
   - UNSAFE: "sản phẩm chứa [allergen] trực tiếp trong thành phần"
   - CRITICAL_ALERT: "cảnh báo nhiễm chéo [allergen] trong dây chuyền sản xuất"
7. Gán nhãn theo quy tắc, nhưng ghi trạng thái `machine_screened_pre_gold`, không gọi là gold
8. source_citations = sources đã dùng trong rule engine
9. Ghi rõ đây là sàng lọc tự động; không gán `approved`, `answer_correct=yes` hoặc `expert_*` nếu chưa có người thực sự kiểm tra
10. Khi chưa có chuyên gia, chỉ dùng cho kiểm thử nội bộ / exploratory weak-label evaluation. Với câu hỏi cần đánh giá lâm sàng hoặc pháp lý, thu hẹp thành câu hỏi kiểm tra nhãn hoặc trả `ABSTAIN`; không phát hành kết luận như dữ kiện gold.
```

---

## 6. Điều không được làm

- Không gán label từ domain mà không kiểm tra nội dung sản phẩm
- Không dùng "có trong TT24 = SAFE" khi T1 có kết luận bảo vệ mạnh hơn
- Không thêm `condition=general_population` cho health concern vốn chỉ liên quan đến nhóm đặc thù (ví dụ: nitrite → trẻ nhỏ, phosphate → CKD)
- Không để `note` rỗng khi có health concern — phải ghi rõ cơ chế và nguồn
- Không ghi trạng thái tự động hoặc kế thừa là chuyên gia đã duyệt
- Không dùng rule-engine match như bằng chứng độc lập cho tính đúng của rule hoặc nhãn
- Không gọi vắng dị nguyên trên nhãn là bằng chứng sản phẩm không có nguy cơ dị nguyên/nhiễm chéo
- Không kết luận sản phẩm tuân thủ ML/ADI khi thiếu loại thực phẩm, hàm lượng và phơi nhiễm cần thiết
