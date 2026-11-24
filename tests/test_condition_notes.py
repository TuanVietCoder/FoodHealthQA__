"""Bệnh nền KB chưa có lưu ý phụ gia: chỉ thêm dòng lưu ý, không từ chối và không đổi nhãn."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import abstention, pipeline  # noqa: E402

COVERED = {"asthma", "ckd", "pku", "children", "infants_young_children", "general_population"}
WITH_ADDITIVE = [{"raw_mention": "Natri benzoat (211)", "matched_entity": {"ins_code": "INS 211"}}]
NO_ADDITIVE = [{"raw_mention": "đường", "matched_entity": None}]


def _codes(conditions, linked=WITH_ADDITIVE, covered=COVERED):
    notes = abstention.population_notes("q", {"conditions": conditions}, linked_results=linked, covered_conditions=covered)
    return [n["group"] for n in notes if n["code"] == "CONDITION_NO_ADDITIVE_EVIDENCE"]


class ConditionNotesTests(unittest.TestCase):
    def test_condition_without_any_checked_additive_note_gets_a_note(self):
        self.assertEqual(_codes(["gout"]), ["bệnh gút"])
        self.assertEqual(_codes(["pregnancy"]), ["thai kỳ"])

    def test_conditions_the_kb_does_cover_get_no_note_in_any_spelling(self):
        for spelling in ("ckd", "chronic_kidney_disease_stage_3", "Chronic_Kidney_Disease_Stage_3", "asthma", "phenylketonuria_pku", "PKU"):
            self.assertEqual(_codes([spelling]), [], spelling)

    def test_healthy_age_groups_and_allergies_are_not_medical_conditions_for_this_note(self):
        for cond in ("healthy", "Healthy_Adult", "children_adhd", "Childhood_ADHD_Risk", "Children_Under_6", "soy_allergy",
                     "Severe_Food_Allergies", "lactose_intolerance", "gluten_intolerance", "elderly"):
            self.assertEqual(_codes([cond]), [], cond)

    def test_no_note_when_the_label_has_no_identified_additive(self):
        self.assertEqual(_codes(["gout"], linked=NO_ADDITIVE), [])

    def test_nutrient_scope_is_stated_for_conditions_with_a_codex_threshold(self):
        note = abstention.condition_notes({"conditions": ["diabetes"]}, WITH_ADDITIVE, COVERED)[0]
        self.assertIn("lượng đường", note["text"])
        self.assertNotIn("natri", note["text"])
        note = abstention.condition_notes({"conditions": ["hypertension"]}, WITH_ADDITIVE, COVERED)[0]
        self.assertIn("lượng natri", note["text"])

    def test_old_callers_without_the_new_arguments_are_unchanged(self):
        self.assertEqual(abstention.population_notes("q", {"conditions": ["gout"]}), [])

    def test_each_condition_is_noted_once(self):
        self.assertEqual(_codes(["gout", "Gout", "diabetes"]), ["bệnh gút", "đái tháo đường"])


class ConditionNotesThroughThePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = pipeline.load_knowledge_base(str(Path(__file__).resolve().parents[1] / "data"))

    def _evaluate(self, conditions, mention="Disodium 5'-inosinate (631)"):
        mentions = [{"raw_mention": mention, "category": "additive"}]
        linked = pipeline.link_all_mentions(mentions, self.kb)
        return pipeline.evaluate_safety_and_guardrail(linked, {"conditions": conditions, "allergies": []}, "q",
                                                      avg_ocr_conf=0.9, all_mentions=mentions)

    def test_gout_gets_a_note_but_the_label_and_status_are_untouched(self):
        result = self._evaluate(["gout"])
        notes = [n for n in result["population_notes"] if n["code"] == "CONDITION_NO_ADDITIVE_EVIDENCE"]
        self.assertEqual(len(notes), 1)
        self.assertEqual((result["safety_label"], result["status"]), ("SAFE", "SCREENED_NO_ALERT"))
        self.assertEqual(result["data_gaps"], [])  # không biến thành từ chối

    def test_ckd_persona_is_covered_and_still_alerts(self):
        result = self._evaluate(["ckd"], "Phosphoric acid (338)")
        self.assertEqual(result["safety_label"], "CAUTION")
        self.assertEqual([n for n in result["population_notes"] if n["code"] == "CONDITION_NO_ADDITIVE_EVIDENCE"], [])


class ElderlyFromWordsTests(unittest.TestCase):
    """Người ăn nói bằng lời (không nêu số tuổi) vẫn được nhận ra là người cao tuổi; không nhận nhầm 'bà bầu'."""

    def _codes(self, question):
        return [n["code"] for n in abstention.population_notes(question, {"conditions": ["healthy"], "demographics": {"age": 30}})]

    def test_kinship_words_for_older_relatives_trigger_the_elderly_note(self):
        for q in ("Bà nội tôi ăn sản phẩm này được không?", "Ông ngoại ăn được không?", "Ông bà tôi lớn tuổi, ăn được không?",
                  "Cụ nhà tôi ăn được không?", "Người cao tuổi ăn được không?", "Ông tôi ăn được không?"):
            self.assertEqual(self._codes(q), ["ELDERLY_NO_SPECIFIC_EVIDENCE"], q)

    def test_unrelated_words_do_not(self):
        for q in ("Bà bầu ăn được không?", "Ông chủ cửa hàng bán sản phẩm này", "Sản phẩm này ăn được không?", "Con tôi 5 tuổi ăn được không?"):
            self.assertEqual(self._codes(q), [], q)

    def test_a_product_name_in_quotes_is_ignored(self):
        self.assertEqual(self._codes("Sản phẩm 'Kẹo Ông Bà Tôi' ăn được không?"), [])


class PatientFacingWordingTests(unittest.TestCase):
    """Khi KB không có dữ liệu cho bệnh nền, người dùng không được hiểu 'không có cảnh báo' là 'an toàn'."""

    def test_the_note_says_plainly_that_no_warning_is_not_a_safety_statement(self):
        note = abstention.condition_notes({"conditions": ["gout"]}, WITH_ADDITIVE, COVERED)[0]
        self.assertIn("Chưa có thông tin", note["text"])
        self.assertIn("KHÔNG có nghĩa là sản phẩm an toàn", note["text"])
        self.assertIn("bác sĩ", note["text"])

    def test_condition_gap_groups_lists_only_the_conditions_without_data(self):
        from modules import evidence
        evaluation = {"population_notes": [{"code": "CONDITION_NO_ADDITIVE_EVIDENCE", "group": "bệnh gút"},
                                           {"code": "ELDERLY_NO_SPECIFIC_EVIDENCE", "group": "người cao tuổi"}]}
        self.assertEqual(evidence.condition_gap_groups(evaluation), ["bệnh gút"])
        self.assertEqual(evidence.condition_gap_groups({}), [])

    def test_the_green_no_issue_box_is_replaced_when_a_declared_condition_has_no_data(self):
        import re
        from modules import evidence
        source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        body = re.search(r"(def verdict_view.*?)(?=\ndef _word_edits)", source, re.S).group(1)
        scope = {"m_evidence": evidence, "_join_vi": lambda items: ", ".join(items), "_allergy_hits": lambda *a: [],
                 "_allergen_vi": lambda k: k}
        exec(body, scope)  # noqa: S102 - chỉ chạy hàm hiển thị của chính dự án
        plain = scope["verdict_view"]({"status": "SCREENED_NO_ALERT", "population_notes": []})
        gap = scope["verdict_view"]({"status": "SCREENED_NO_ALERT", "population_notes": [
            {"code": "CONDITION_NO_ADDITIVE_EVIDENCE", "group": "bệnh gút"}]})
        self.assertEqual(plain[3], "success")
        self.assertEqual((gap[1], gap[3]), ("Chưa có thông tin về bệnh bạn đã khai", "warning"))
        self.assertIn("KHÔNG có nghĩa là bạn dùng được", gap[2])

    def test_the_llm_is_told_never_to_call_the_product_safe_for_that_condition(self):
        from unittest import mock
        evaluation = {"title": "t", "advice": "a", "safety_label": "SAFE", "alerts": [], "regulatory": [], "citations": ["JECFA_EVALUATIONS"],
                      "population_notes": [{"code": "CONDITION_NO_ADDITIVE_EVIDENCE", "group": "bệnh gút", "text": "CHƯA CÓ DỮ LIỆU cho bệnh gút"}]}
        seen = []
        with mock.patch.object(pipeline, "_llm_ready", return_value=True), mock.patch.object(pipeline, "_get_gemini_api_key", return_value="k"),                 mock.patch.object(pipeline, "get_llm_provider", return_value="gemini"),                 mock.patch.object(pipeline, "_call_llm", side_effect=lambda prompt, **k: seen.append(prompt) or "x [JECFA_EVALUATIONS]"):
            pipeline.generate_llm_answer(evaluation, [], {"conditions": [], "allergies": []}, "q")
        self.assertIn("TUYỆT ĐỐI không viết rằng sản phẩm 'an toàn'", seen[0])


if __name__ == "__main__":
    unittest.main()
