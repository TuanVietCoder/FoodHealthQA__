import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import paddle
    paddle.set_flags({'FLAGS_use_mkldnn': False})
except Exception:
    pass

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.pipeline import (
    load_knowledge_base,
    load_personas,
    run_perception_pipeline,
    foodontorag_entity_linking,
    evaluate_safety_and_guardrail
)

kb = load_knowledge_base(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "additives_mini.json"))
personas = load_personas(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "personas.json"))
samples = [
    ("cookie", os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "cookie_box_allergen_label.png")),
    ("instant_noodles", os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "instant_noodles_ingredient_label.png")),
    ("beverage", os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "beverage_can_ingredient_label.png"))
]

print("=== KIỂM THỬ CHẾ ĐỘ NHÃN CHUẨN (GROUND TRUTH) ===")
for key, path in samples:
    res = run_perception_pipeline(path, sample_key=key, force_real_ocr=False)
    print(f"[{key}] Full text: {res['full_text'][:70]}...")
    print(f"  -> Mentions ({len(res['mentions'])}): {[m['raw_mention'] for m in res['mentions'][:3]]}")
    linked = [foodontorag_entity_linking(m, kb) for m in res['mentions']]
    eval_res = evaluate_safety_and_guardrail(linked, personas[0], "An toàn không?")
    print(f"  -> Evaluation: {eval_res['status']} (S_total={eval_res['s_total']:.2f})\n")

print("=== KIỂM THỬ CHẾ ĐỘ PADDLEOCR / THỰC TẾ ===")
for key, path in samples:
    res = run_perception_pipeline(path, sample_key=key, force_real_ocr=True)
    print(f"[{key}] Engine: {res['engine_name']}")
    print(f"  -> Full text: {res['full_text'][:70]}...")
    print(f"  -> Mentions ({len(res['mentions'])}): {[m['raw_mention'] for m in res['mentions'][:3]]}\n")
print("Tất cả bài kiểm tra hoàn tất thành công!")
