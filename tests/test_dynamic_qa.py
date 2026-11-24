"""Kiểm tra tầng QA động (modules/dynamic_qa.py) và việc tích hợp vào run_grounded_pipeline."""
import inspect
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules import dynamic_qa
from modules.ocr_extractor import parse_ingredients_from_ocr
from modules.pipeline import (
    load_knowledge_base,
    load_personas,
    run_grounded_pipeline,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
_KB = load_knowledge_base(DATA_DIR)
_HEALTHY = {"id": "healthy_adult", "name": "Người trưởng thành khỏe mạnh", "conditions": ["healthy"], "allergies": []}


def _run(label_text, question, persona=None, enabled=True):
    mentions = parse_ingredients_from_ocr(label_text, include_allergen_statements=True)
    return run_grounded_pipeline(
        mentions=mentions, avg_ocr_conf=1.0, product_name="Sản phẩm thử", persona=persona or _HEALTHY,
        question=question, kb_additives=_KB, hazards_list=[], fndds_list=[], off_products=[],
        generate_answer=False, dynamic_qa_enabled=enabled,
    )


def test_understand_extracts_intent_and_entities():
    plan = dynamic_qa.understand_question("Aspartame có an toàn cho trẻ em khi dùng hàng ngày không?")
    codes = {i["code"] for i in plan["intents"]}
    assert "intake_limit" in codes
    assert [a["ins_code"] for a in plan["additives"]] == ["INS 951"]
    assert [p["code"] for p in plan["populations"]] == ["child"]


def test_product_name_in_quotes_is_not_an_entity():
    plan = dynamic_qa.understand_question("Nếu dùng 'Sữa đậu nành đường' thường xuyên thì sao?")
    assert plan["allergens"] == []
    assert plan["nutrients"] == []


def test_out_of_scope_question_is_flagged():
    plan = dynamic_qa.understand_question("Tôi có phải bị tiểu đường không, chữa bằng sản phẩm này được không?")
    assert plan["out_of_scope"]


def test_additive_in_kb_but_not_on_label_gives_no_gap_and_adds_evidence():
    result = _run("Thành phần: Nước, đường, hương liệu", "INS 951 có an toàn không?")
    qa = result["qa"]
    assert qa["knowledge_gaps"] == []
    assert any(t["status"] == "found" and "INS 951" in t["query"] for t in qa["trace"])
    assert "JECFA_EVALUATIONS" in result["evaluation"]["citations"]


def test_identity_only_additive_is_reported_not_found_but_does_not_abstain():
    # INS 100(i) (Curcumin) chỉ có định danh trong registry, chưa có dữ liệu quy chuẩn trong KB.
    # Câu hỏi thường nêu phụ gia như ví dụ của một nhóm chất, nên chỉ ghi vào vết tra cứu, không từ chối.
    text, q = "Thành phần: Nước, đường", "Curcumin INS 100(i) có vượt ADI không?"
    result = _run(text, q)
    assert result["qa"]["knowledge_gaps"] == []
    assert any(t["status"] == "not_found" and "100(i)" in t["query"] for t in result["qa"]["trace"])
    assert result["safety_label"] == _run(text, q, enabled=False)["safety_label"]
    assert "KHÔNG TÌM THẤY" in dynamic_qa.render_context(result["qa"] and dynamic_qa.understand_question(q), result["qa"]["trace"])


def test_suffix_code_on_label_is_not_a_gap():
    # 'INS160A(I)' trong câu hỏi không được cắt thành '160A' rồi coi là mã lạ khi nhãn có phụ gia đó
    result = _run("Thành phần: Nước, chất tạo màu (INS 160a(i))", "Phụ gia INS160A(I) trong sản phẩm có ảnh hưởng sức khỏe không?")
    assert result["qa"]["knowledge_gaps"] == []
    assert result["qa"]["entities"]["additives"] == [] or "INS 160A" not in result["qa"]["entities"]["additives"]


def test_class_examples_in_question_do_not_force_abstain():
    q = "Tôi đang mang thai — nhãn có chứa chất ngọt không đường (như aspartame, acesulfame K) không?"
    text = "Thành phần: Nước, đường, hương liệu"
    assert _run(text, q)["safety_label"] == _run(text, q, enabled=False)["safety_label"]


def test_unknown_ins_code_is_not_silently_ignored():
    result = _run("Thành phần: Nước, đường", "INS 9999 có an toàn không?")
    assert result["safety_label"] == "ABSTAIN"
    assert result["qa"]["entities"]["additives"] == ["INS 9999"]


def test_additive_on_label_is_left_to_existing_guardrail():
    result = _run("Thành phần: Nước, chất tạo ngọt (INS 951)", "INS 951 có an toàn không?")
    assert result["qa"]["knowledge_gaps"] == []


def test_dynamic_layer_does_not_change_label_for_plain_label_question():
    q = "Sản phẩm này có phụ gia nào cần lưu ý không?"
    text = "Thành phần: Nước, đường, chất bảo quản (INS 211), hương liệu"
    on, off = _run(text, q, enabled=True), _run(text, q, enabled=False)
    assert on["safety_label"] == off["safety_label"]
    assert off["qa"] is None


def test_b_and_c_share_same_dynamic_layer_for_same_text():
    text = "Thành phần: Nước, đường, chất bảo quản (INS 211)"
    q = "INS 951 có vượt ADI không?"
    assert _run(text, q)["qa"] == _run(text, q)["qa"]


def test_dynamic_qa_never_reads_benchmark():
    source = inspect.getsource(dynamic_qa).lower()
    assert "qa_benchmark" not in source
    assert "reference_answer" not in source
    assert "gold_safety_label" not in source


def _intent_and_nutrients(question):
    plan = dynamic_qa.understand_question(question)
    return [i["code"] for i in plan["intents"]], [n["code"] for n in plan["nutrients"]], [a["ins_code"] for a in plan["additives"]]


def test_sodium_inside_an_additive_name_is_not_a_nutrition_question():
    for question, code in (("Tôi muốn biết về natri benzoat", "INS 211"), ("Natri nitrit có an toàn không?", "INS 250")):
        intents, nutrients, additives = _intent_and_nutrients(question)
        assert "nutrition" not in intents and nutrients == [], question
        assert code in additives, question


def test_sweetener_class_phrase_is_not_a_sugar_question():
    intents, nutrients, additives = _intent_and_nutrients("Chất tạo ngọt aspartame có sao không?")
    assert nutrients == [] and "additive_safety" in intents and "INS 951" in additives


def test_real_nutrition_questions_are_still_recognised():
    assert _intent_and_nutrients("Sản phẩm có nhiều natri không?")[:2] == (["nutrition"], ["sodium"])
    assert _intent_and_nutrients("Lượng đường trong bánh này có cao không?")[:2] == (["nutrition"], ["sugar"])


def test_additive_and_nutrient_in_the_same_question_are_both_kept():
    intents, nutrients, additives = _intent_and_nutrients("Natri benzoat và lượng muối trong sản phẩm này thế nào?")
    assert "nutrition" in intents and nutrients == ["sodium"] and additives == ["INS 211"]
