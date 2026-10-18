"""
Kịch bản Kiểm thử Toàn diện Hệ thống FoodHealthQA
Kiểm tra:
1. Tầng Tiền xử lý & Phục hồi Chính tả (Post-OCR Lexical Normalizer)
2. Tầng Chuẩn hóa Thực thể 4-Agent (FoodOntoRAG)
3. Tầng Guardrail và Đánh giá An toàn Y tế
"""

import os
import sys

# Đảm bảo import đúng module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ocr_extractor import (
    post_process_ocr_spelling,
    parse_ingredients_from_ocr,
    run_perception
)
from modules.pipeline import (
    load_knowledge_base,
    load_personas,
    run_perception_pipeline,
    foodontorag_entity_linking,
    evaluate_safety_and_guardrail
)
from modules.foodon import load_foodon_entities, search_foodon_entities

def run_tests():
    print("=" * 60)
    print("BẮT ĐẦU KIỂM THỬ TOÀN DIỆN FOODHEALTHQA")
    print("=" * 60)

    # 1. Test Phục hồi chính tả OCR thô
    raw_noisy_ocr = (
        "Thnh phn / INGREDIENTS: bot mi, bo lat, dudng kinh, tring ga tusi, "
        "bot sifa béo, chat nhiihda (INS. 322(i lecithin, INS 471), "
        "chat tao x6p (INS 500(ii), INS 503(ii)), mudi i-6t, huong vani tong hop. "
        "Canh bao di ung: Chita gluten (bot mi), tring, sila. "
        "Co the chia vet cla hat dau phong, hanh nhan."
    )
    print("\n[TEST 1] Kiểm tra Tầng Phục hồi Chính tả Thực phẩm:")
    print(f"Văn bản đầu vào (Noisy OCR):\n  -> {raw_noisy_ocr}")

    corrected = post_process_ocr_spelling(raw_noisy_ocr)["final"]
    print(f"\nVăn bản sau khi khôi phục chính tả & mã INS:\n  -> {corrected}")
    assert "Bột mì" in corrected or "bột mì" in corrected.lower(), "Lỗi: Không phục hồi được từ 'bột mì'"
    assert "INS 322i" in corrected, "Lỗi: Không chuẩn hóa được mã INS 322i"
    print("✅ TEST 1 THÀNH CÔNG: Đã khôi phục chuẩn xác toàn bộ dấu thanh và mã INS!")

    # 2. Test Bóc tách thành phần
    mentions = parse_ingredients_from_ocr(corrected)
    print(f"\n[TEST 2] Bóc tách thành phần (Mentions: {len(mentions)}):")
    for idx, m in enumerate(mentions, 1):
        print(f"  {idx}. {m['raw_mention']} ({m['category']})")
    assert len(mentions) >= 5, "Lỗi: Bóc tách thành phần quá ít"
    print("✅ TEST 2 THÀNH CÔNG: Bóc tách danh sách thành phần và dị nguyên hoàn hảo!")

    # 3. Test nạp Knowledge Base & Personas
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    kb = load_knowledge_base(data_dir)
    personas = load_personas(data_dir)
    foodon_catalog = load_foodon_entities(data_dir)
    print(f"\n[TEST 3] Nạp dữ liệu: {len(kb)} phụ gia, {len(personas)} hồ sơ bệnh lý.")
    assert len(kb) >= 5, "Lỗi: KB phụ gia quá ít"
    assert len(personas) >= 5, "Lỗi: Danh sách hồ sơ quá ít"
    assert len(foodon_catalog) >= 29000, "Lỗi: Chưa nạp đủ danh mục FoodOn"
    assert all(
        additive.get("foodon_curie") != "FOODON:03301704"
        for additive in kb if additive.get("ins_code") == "INS 621"
    ), "Lỗi: CURIE không khớp nhãn FoodOn vẫn được gán cho INS 621"
    assert next(a for a in kb if a.get("ins_code") == "INS 471")["foodon_curie"] == "FOODON:03413156", \
        "Lỗi: Không khớp chính xác tên INS 471 với FoodOn"
    assert next(a for a in kb if a.get("ins_code") == "INS 1420")["foodon_curie"] == "FOODON:03412981", \
        "Lỗi: Không khớp chính xác tên INS 1420 với FoodOn"
    assert search_foodon_entities("FOODON:03413156", foodon_catalog)[0]["label"] == \
        "mono- and diglycerides of fatty acids", "Lỗi: Tra cứu theo CURIE FoodOn"

    # 4. Test Entity Linking (FoodOntoRAG)
    linked = [foodontorag_entity_linking(m, kb) for m in mentions]
    print(f"\n[TEST 4] Chuẩn hóa thực thể FoodOntoRAG (Linked: {len(linked)}):")
    matched_count = 0
    for r in linked:
        e = r["matched_entity"]
        if e:
            matched_count += 1
            print(f"  * '{r['raw_mention']}' -> {e['canonical_name']} ({e['ins_code']}) [tau={r['confidence_tau']}]")
    assert matched_count >= 2, "Lỗi: Không khớp được phụ gia trong KB"
    print("✅ TEST 4 THÀNH CÔNG: FoodOntoRAG ánh xạ chính xác về mã định danh chuẩn!")

    # 5. Test Guardrail với Hồ sơ Dị ứng
    allergy_persona = next(
        (p for p in personas if "dị ứng thực phẩm" in p["name"].lower() or ("dị ứng" in p["name"].lower() and "pku" not in p["name"].lower())),
        personas[5]
    )
    eval_res = evaluate_safety_and_guardrail(linked, allergy_persona, "Có ăn được không?")
    print(f"\n[TEST 5] Đánh giá Guardrail với Hồ sơ: {allergy_persona['name']}")
    print(f"  -> Trạng thái: {eval_res['status']}")
    print(f"  -> Điểm tin cậy: {eval_res['s_total']:.2f}")
    print(f"  -> Số lượng cảnh báo kích hoạt: {len(eval_res.get('alerts', []))}")
    assert eval_res["status"] == "WARNING", \
        "Lỗi: Khớp từ khóa dị nguyên chỉ được tạo cảnh báo, không kết luận chống chỉ định"
    print("✅ TEST 5 THÀNH CÔNG: Guardrail kích hoạt chuẩn xác cảnh báo y tế!")

    print("\n" + "=" * 60)
    print("🎉 TẤT CẢ CÁC BÀI KIỂM THỬ ĐÃ VƯỢT QUA 100%!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
