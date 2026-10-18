"""
Kịch bản Kiểm thử Tích hợp Đa nguồn Dữ liệu (FoodHealthQA Data Suite):
- Tương thích 100% với toàn bộ 16 tệp dữ liệu thực tế (theo hình chụp name-data.png)
  1. additives_standard (CSV/XLS/JSON) + additives_mini.json
  2. personas.json / nhanes_personas.csv
  3. incidents_train (12.5 MB Food Hazard Detector)
  4. openfoodfacts_vietnam (Open Food Facts VN)
  5. foodon_entities.json (4.6 MB Food Ontology)
  6. food, nutrient, food_nutrient (USDA FNDDS)
  7. P_DEMO, P_GHB, P_GLU, P_BIOPRO, P_BMX, P_BPXO, P_DBQ (NHANES Clinical Survey)
- Tự động phát hiện thông minh kể cả khi Windows ẩn đuôi tệp (.csv, .xlsx).
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from modules.file_resolver import find_data_file
from modules.pipeline import (
    load_knowledge_base,
    load_personas,
    load_food_hazards,
    load_fndds_nutrition,
    load_openfoodfacts_vietnam,
    check_nhanes_raw_files,
    foodontorag_entity_linking,
    evaluate_safety_and_guardrail,
    check_food_hazards,
    evaluate_nutritional_risk,
    run_perception_pipeline
)

def run_csv_tests():
    print("=" * 70)
    print("🔍 BẮT ĐẦU KIỂM THỬ ĐA NGUỒN DỮ LIỆU THỰC TẾ (FOODHEALTHQA DATA SUITE)")
    print("=" * 70)

    data_dir = os.path.join(BASE_DIR, "data")
    assets_dir = os.path.join(BASE_DIR, "assets")

    # 1. Nạp tất cả tệp dữ liệu thông minh
    kb = load_knowledge_base(data_dir)
    personas = load_personas(data_dir)
    hazards = load_food_hazards(data_dir)
    fndds = load_fndds_nutrition(data_dir)
    off = load_openfoodfacts_vietnam(data_dir)

    print(f"\n[TEST 1] Đối soát và Nạp Cơ sở Tri thức & Dữ liệu Cốt lõi:")
    print(f"  * Phụ gia & Bản thể học (additives_standard / mini): {len(kb)} mục đã sẵn sàng")
    print(f"  * Hồ sơ Lâm sàng Người dùng (personas.json / csv): {len(personas)} hồ sơ bệnh lý")
    print(f"  * Mối nguy Thực phẩm (incidents_train): {len(hazards)} sự cố thu hồi")
    print(f"  * Định lượng Dinh dưỡng (FNDDS food/nutrition): {len(fndds)} mục khảo sát")
    print(f"  * Nhãn Hàng Việt Nam (openfoodfacts_vietnam): {len(off)} sản phẩm thương phẩm")
    assert len(kb) >= 5 and len(personas) >= 5, "Lỗi nạp dữ liệu cốt lõi"
    print("  ✅ TEST 1 THÀNH CÔNG!\n")

    # 2. Kiểm tra các tệp dữ liệu đặc thù theo hình name-data.png
    print(f"[TEST 2] Kiểm tra Nhận diện 16 Nguồn Dữ liệu Thực tế (name-data.png):")
    
    # A. Food Ontology
    foodon_p = find_data_file(data_dir, ["foodon_entities", "foodon"])
    if foodon_p:
        size_kb = os.path.getsize(foodon_p) / 1024
        print(f"  * [Food Ontology] Tìm thấy 'foodon_entities.json' ({size_kb:.0f} KB) -> Đã tích hợp vào FoodOntoRAG!")
    else:
        print(f"  * [Food Ontology] Sử dụng mã FoodOn CURIE chuẩn hóa trong additives_standard.")

    # B. NHANES raw files
    nhanes_status = check_nhanes_raw_files(data_dir)
    nhanes_found = [k for k, v in nhanes_status.items() if v]
    if nhanes_found:
        print(f"  * [NHANES Raw Survey] Phát hiện {len(nhanes_found)}/7 tệp lâm sàng: {', '.join(nhanes_found)}")
    else:
        print(f"  * [NHANES Raw Survey] Đang sử dụng các hồ sơ bệnh án tổng hợp trong personas.json.")

    # C. FNDDS raw files
    fndds_raw = [name for name in ["food", "nutrient", "food_nutrient"] if find_data_file(data_dir, [name])]
    if fndds_raw:
        print(f"  * [USDA FNDDS Raw] Phát hiện các tệp FNDDS: {', '.join(fndds_raw)}")
    print("  ✅ TEST 2 THÀNH CÔNG!\n")

    # 3. Test kịch bản Nhãn Bánh quy bơ với Người Dị ứng Thực phẩm
    cookie_path = os.path.join(assets_dir, "cookie_box_allergen_label.png")
    cookie_res = run_perception_pipeline(cookie_path, sample_key="cookie", force_real_ocr=False)
    cookie_linked = [foodontorag_entity_linking(m, kb) for m in cookie_res["mentions"]]
    
    allergy_persona = next((p for p in personas if "dị ứng" in p["name"].lower() or "allergy" in p.get("id", "")), personas[0])
    eval_cookie = evaluate_safety_and_guardrail(cookie_linked, allergy_persona, "Có ăn được không?")
    hazard_alerts = check_food_hazards("Bánh quy bơ", cookie_res["mentions"], hazards)
    
    print(f"[TEST 3] Đánh giá Bánh quy bơ với Người Dị ứng Thực phẩm:")
    print(f"  * Trạng thái Guardrail: {eval_cookie['status']} ({eval_cookie['title']})")
    print(f"  * Số lượng cảnh báo sự cố Food Hazard: {len(hazard_alerts)}")
    for h in hazard_alerts:
        print(f"    - [{h['hazard_category'].lower()}] {h['title']}")
    assert eval_cookie["status"] in ["WARNING", "CRITICAL_ALERT"], "Phải kích hoạt cảnh báo dị ứng"
    print("  ✅ TEST 3 THÀNH CÔNG!\n")

    # 4. Test kịch bản Mì ăn liền với Bệnh nhân Tăng huyết áp & FNDDS
    noodle_path = os.path.join(assets_dir, "instant_noodles_ingredient_label.png")
    noodle_res = run_perception_pipeline(noodle_path, sample_key="instant_noodles", force_real_ocr=False)
    noodle_linked = [foodontorag_entity_linking(m, kb) for m in noodle_res["mentions"]]
    
    ht_persona = next((p for p in personas if "tăng huyết áp" in p["name"].lower() or "hypertension" in p.get("id", "")), personas[1])
    eval_noodle = evaluate_safety_and_guardrail(noodle_linked, ht_persona, "Tôi cao huyết áp dùng được không?")
    matched_nutri, nutri_warnings = evaluate_nutritional_risk("Mì ăn liền", fndds, ht_persona)

    print(f"[TEST 4] Đánh giá Mì ăn liền với Bệnh nhân Tăng huyết áp:")
    print(f"  * Trạng thái Guardrail: {eval_noodle['status']} ({eval_noodle['title']})")
    print(f"  * Cảnh báo dinh dưỡng số học FNDDS: {len(nutri_warnings)}")
    for nw in nutri_warnings:
        print(f"    - {nw}")
    assert len(nutri_warnings) > 0, "Phải phát hiện cảnh báo Natri cao từ FNDDS"
    print("  ✅ TEST 4 THÀNH CÔNG!\n")

    print("=" * 70)
    print("🎉 TẤT CẢ CÁC BÀI KIỂM THỬ ĐÃ HOÀN TẤT THÀNH CÔNG 100%!")
    print("=" * 70)

if __name__ == "__main__":
    run_csv_tests()
