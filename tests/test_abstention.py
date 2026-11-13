"""Two-type abstention: missing data -> re-photograph / supply information; out of scope -> refer to a professional."""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import abstention  # noqa: E402
from modules import pipeline  # noqa: E402
from modules.ocr_extractor import parse_ingredients_from_ocr  # noqa: E402


def _persona(allergies=(), conditions=("healthy", "healthy_adult")):
    return {"id": "t", "conditions": list(conditions), "allergies": list(allergies), "weight_kg": 60.0}


class AbstentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = pipeline.load_knowledge_base(str(ROOT / "data"))

    def run_guardrail(self, label_text, question, persona=None):
        mentions = parse_ingredients_from_ocr(label_text, include_allergen_statements=True)
        linked = [pipeline.foodontorag_entity_linking(m, self.kb) for m in mentions]
        # avg_ocr_conf=1.0: the Config B situation, where the OCR-confidence trigger can never fire
        return pipeline.evaluate_safety_and_guardrail(linked, persona or _persona(), question, avg_ocr_conf=1.0, all_mentions=mentions)

    # ---- missing data --------------------------------------------------------------------------------
    def test_cross_contact_question_without_pal_statement_asks_for_more_data(self):
        r = self.run_guardrail("Thành phần: Bột mì, đường, muối.", "'Bánh' có nguy cơ nhiễm chéo dị nguyên không? Tôi bị dị ứng nặng đậu phộng.",
                               _persona(allergies=["peanut"]))
        self.assertEqual(r["status"], "ABSTAIN")
        self.assertEqual(r["abstention_type"], "insufficient_data")
        self.assertIn("chụp lại", r["message"])
        self.assertNotIn("bác sĩ", r["message"])
        self.assertEqual([g["code"] for g in r["data_gaps"]], ["CROSS_CONTACT_STATEMENT"])

    def test_cross_contact_question_with_pal_statement_is_answered(self):
        r = self.run_guardrail("Thành phần: Bột mì, đường. Có thể chứa đậu phộng.", "'Bánh' có nguy cơ nhiễm chéo dị nguyên không? Tôi bị dị ứng nặng đậu phộng.",
                               _persona(allergies=["peanut"]))
        self.assertNotEqual(r["status"], "ABSTAIN")
        self.assertEqual(r["data_gaps"], [])

    def test_additive_question_with_no_identifiable_additive_asks_for_more_data(self):
        r = self.run_guardrail("Thành phần: Hoa chuối, muối, dầu thực vật.", "Từ danh sách thành phần của 'Hoa chuối sấy', có thể kết luận gì về an toàn phụ gia khi dùng thường xuyên?")
        self.assertEqual(r["status"], "ABSTAIN")
        self.assertEqual(r["abstention_type"], "insufficient_data")
        self.assertEqual([g["code"] for g in r["data_gaps"]], ["ADDITIVE_IDENTITY"])

    def test_permission_question_is_not_treated_as_an_exposure_question(self):
        r = self.run_guardrail("Thành phần: Nước, đường, chất điều chỉnh độ acid (330).",
                               "Tôi ăn 'Trà' thường xuyên — phụ gia INS330 trong sản phẩm này có được phép sử dụng theo Thông tư 24/2019/TT-BYT không?")
        self.assertNotEqual(r["status"], "ABSTAIN")

    def test_allergen_named_in_question_ignores_the_quoted_product_name(self):
        self.assertEqual(pipeline._question_allergies("'Sữa NuVi socola' có nguy cơ nhiễm chéo không? Tôi bị dị ứng nặng tôm."), ["shellfish"])

    def test_pal_statement_about_another_allergen_does_not_answer_the_question(self):
        r = self.run_guardrail("Thành phần: Bột mì, đường. Được sản xuất chung dây chuyền với sản phẩm có chứa trứng.",
                               "'Bánh' có nguy cơ nhiễm chéo dị nguyên không? Tôi bị dị ứng nặng mè/vừng.")
        self.assertEqual(r["status"], "ABSTAIN")
        self.assertIn("nêu dị nguyên bạn hỏi", r["message"])

    def test_flavour_enhancer_wording_is_not_a_cashew_match(self):
        r = self.run_guardrail("Thành phần: Nước, đường, chất điều vị (621), chất điều chỉnh độ acid (330).",
                               "Tôi cần tránh tuyệt đối hạt cây — 'Nước' có an toàn cho tôi không?", _persona(allergies=["nuts"]))
        self.assertFalse(any("điều" in a.get("additive", "").lower() and a.get("allergen_key") == "nuts" for a in r.get("alerts", [])))

    # ---- out of scope --------------------------------------------------------------------------------
    def test_treatment_question_is_out_of_scope_and_refers_to_a_professional(self):
        r = self.run_guardrail("Thành phần: Nước, đường.", "Tôi bị cao huyết áp, ăn 'Nước' này có thể ngừng uống thuốc huyết áp được không?")
        self.assertEqual(r["status"], "ABSTAIN")
        self.assertEqual(r["abstention_type"], "out_of_scope")
        self.assertIn("bác sĩ", r["message"])
        self.assertNotIn("chụp lại", r["message"])

    def test_both_types_give_both_pieces_of_advice(self):
        r = self.run_guardrail("Thành phần: Bột mì, đường.", "'Bánh' có nhiễm chéo dị nguyên không? Tôi bị dị ứng đậu phộng, có thể thay thuốc dị ứng bằng ăn kiêng không?",
                               _persona(allergies=["peanut"]))
        self.assertEqual(r["status"], "ABSTAIN")
        self.assertEqual(set(r["abstention_types"]), {"insufficient_data", "out_of_scope"})
        self.assertIn("chụp lại", r["message"])
        self.assertIn("bác sĩ", r["message"])

    def test_in_scope_question_has_no_scope_issue(self):
        self.assertEqual(abstention.detect_out_of_scope("Phụ gia INS330 trong 'Trà' có được phép sử dụng không?"), [])

    # ---- safety floor --------------------------------------------------------------------------------
    def test_critical_alert_is_never_downgraded_to_abstain(self):
        critical = {"status": "CRITICAL_ALERT", "safety_label": "CRITICAL_ALERT", "alerts": [{"additive": "X", "regulation": "VN_TT24_2019"}],
                    "advice": "Có quy tắc mức cao.", "title": "t"}
        with mock.patch.object(pipeline, "_evaluate_safety_core", return_value=dict(critical)):
            r = pipeline.evaluate_safety_and_guardrail([], _persona(), "Có thể ngừng uống thuốc nếu ăn sản phẩm này không?")
        self.assertEqual(r["status"], "CRITICAL_ALERT")
        self.assertIn("out_of_scope", r["abstention_types"])
        self.assertIn("bác sĩ", r["advice"])

    def test_direct_allergen_hit_keeps_its_warning_when_data_is_also_missing(self):
        r = self.run_guardrail("Thành phần: Bột mì, đậu phộng rang.", "'Bánh' có nguy cơ nhiễm chéo dị nguyên không? Tôi bị dị ứng nặng đậu phộng.",
                               _persona(allergies=["peanut"]))
        self.assertIn(r["status"], ("WARNING", "CRITICAL_ALERT"))
        self.assertTrue(r["data_gaps"])


if __name__ == "__main__":
    unittest.main()
