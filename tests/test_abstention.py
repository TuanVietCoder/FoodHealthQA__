"""Two-type abstention: missing data -> re-photograph / supply information; out of scope -> refer to a professional."""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import abstention  # noqa: E402
from modules import infant_rules  # noqa: E402
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

    # ---- nhóm người dùng: trẻ dưới 12 tháng có luật riêng; người cao tuổi / cho con bú chỉ có dòng lưu ý ----------
    LABEL_INFANT = ("Thành phần: Sữa bột, đường, chất nhũ hóa (lecithin 322), chất điều chỉnh độ acid (330), "
                    "chất bảo quản (211).")

    def test_infant_flags_additive_not_in_infant_food_list(self):
        for q in ("Bé 8 tháng tuổi uống được không?", "Trẻ sơ sinh dùng được không?"):
            r = self.run_guardrail(self.LABEL_INFANT, q)
            self.assertEqual(r["status"], "CAUTION", q)
            flagged = [a for a in r["alerts"] if a.get("infant_rule")]
            self.assertEqual([a["ins_code"] for a in flagged], ["INS 211"], q)
            self.assertIn("Phụ lục 2A", flagged[0]["message"])
            ok = {p["ins_code"] for p in r["infant_check"]["permitted"]}
            self.assertTrue({"INS 322(i)", "INS 330"} <= ok or {"INS 322", "INS 330"} & ok, ok)
            self.assertIn("VN_TT24_2019", r["citations"])
            self.assertEqual([n["code"] for n in r["population_notes"]], ["INFANT_SCOPE"])

    def test_infant_rule_from_profile_months_and_not_for_older_children(self):
        infant = {**_persona(), "demographics": {"age": 1, "age_months": 8}}
        self.assertEqual(self.run_guardrail(self.LABEL_INFANT, "Uống được không?", infant)["status"], "CAUTION")
        for q in ("Bé 18 tháng tuổi dùng được không?", "'Sữa cho trẻ sơ sinh' có chất bảo quản không?"):
            r = self.run_guardrail(self.LABEL_INFANT, q)
            self.assertFalse([a for a in r["alerts"] if a.get("infant_rule")], q)
            self.assertNotIn("infant_check", r, q)

    def test_infant_label_without_additives_is_not_refused(self):
        r = self.run_guardrail("Thành phần: Sữa tươi, đường.", "Bé 8 tháng tuổi uống được không?")
        self.assertEqual(r["status"], "SCREENED_NO_ALERT")
        self.assertEqual(r["infant_check"]["checked"], 0)
        self.assertEqual([n["code"] for n in r["population_notes"]], ["INFANT_SCOPE"])

    # ---- giới hạn tuổi trên nhãn; nguồn khoáng/vitamin; từ dị nguyên trong câu hỏi ------------------------------
    def test_label_age_statement_blocks_younger_infant_only(self):
        cases = {  # (nhãn, tuổi bé tháng) -> có cảnh báo không
            ("Thành phần: Sữa tươi, đường. Không dùng cho trẻ dưới 1 tuổi.", 9): True,
            ("Thành phần: Sữa, đường. Không dùng cho trẻ dưới 2 tuổi.", 11): True,
            ("Thành phần: Bột gạo, bí đỏ. 7+ tháng tuổi.", 5): True,
            ("Thành phần: Bột gạo, bí đỏ. 7+ tháng tuổi.", 8): False,
            ("Thành phần: Gạo, thịt heo. Sản phẩm này phù hợp cho trẻ từ 7 tháng tuổi trở lên và người lớn.", 6): True,
            ("Thành phần: Gạo, thịt heo. Sản phẩm này phù hợp cho trẻ từ 6 tháng tuổi trở lên và người lớn.", 7): False,
            ("Thành phần: Sữa bò. Không thích hợp cho trẻ sơ sinh trừ khi có chỉ định của bác sỹ.", 10): False,
            ("Thành phần: Kẹo. Trẻ em dưới 3 tuổi phải có sự giám sát của người lớn khi ăn.", 10): False,
        }
        for (label, months), blocked in cases.items():
            r = self.run_guardrail(label, f"Bé {months} tháng tuổi dùng được không?")
            age = [a for a in r["alerts"] if a.get("label_age_rule")]
            self.assertEqual(bool(age), blocked, (label, months))
            if blocked:
                self.assertEqual(r["status"], "WARNING", label)
                self.assertIn("CODEX_CXS_74_1981", r["citations"])
                self.assertEqual(r["infant_check"]["label_age"], age)
        # chỉ biết "dưới 12 tháng": chỉ chặn khi nhãn ghi từ 12 tháng trở lên
        self.assertEqual(self.run_guardrail("Thành phần: Bột gạo. 7+ tháng tuổi.", "Trẻ sơ sinh ăn được không?")["status"],
                         "SCREENED_NO_ALERT")
        self.assertEqual(self.run_guardrail("Thành phần: Sữa. Không dùng cho trẻ dưới 1 tuổi.", "Trẻ sơ sinh uống được không?")["status"],
                         "WARNING")
        # người lớn: không đọc câu tuổi
        self.assertEqual(self.run_guardrail("Thành phần: Sữa. Không dùng cho trẻ dưới 1 tuổi.", "Uống được không?")["status"],
                         "SCREENED_NO_ALERT")

    def test_nutrient_sources_in_formula_are_not_checked_as_additives(self):
        label = ("Thành phần: Đạm đậu nành, Kali clorid (nguồn bổ sung khoáng), Beta caroten (tiền Vitamin A), "
                 "Acid ascorbic (Vitamin C), Chất nhũ hóa: Lecithin đậu nành, Chất chống oxy hóa: Ascorbyl palmitat.")
        r = self.run_guardrail(label, "Bé 3 tháng tuổi uống được không?")
        self.assertEqual(r["status"], "SCREENED_NO_ALERT")
        self.assertIn("INS 508", {n["ins_code"] for n in r["infant_check"]["nutrients"]})
        self.assertIn("CODEX_CXS_72_1981", r["citations"])
        # cùng dòng mà có nhóm phụ gia thì vẫn xét: "vitamin (D3, A). chất ổn định (466)"
        self.assertTrue(infant_rules.is_nutrient_source("Kali clorid (nguồn bổ sung khoáng)"))
        self.assertFalse(infant_rules.is_nutrient_source("vitamin (D3, A). chất ổn định (440, 466)"))
        self.assertFalse(infant_rules.is_nutrient_source("Chất chống oxy hóa: Ascorbyl palmitat"))

    def test_allergen_word_in_question_needs_an_allergy_cue(self):
        q_allergy = pipeline._question_allergies
        self.assertEqual(q_allergy("Bé 9 tháng tuổi ăn sữa chua 'TH true YOGURT' được không?"), [])
        self.assertEqual(q_allergy("Tôi ăn sữa chua 'X' được không?"), [])
        self.assertEqual(q_allergy("Bé dị ứng trứng có uống sữa chua 'X' được không?"), ["egg"])
        self.assertIn("milk", q_allergy("Người dị ứng sữa, lúa mì (gluten) có nên tránh 'X' không?"))
        self.assertEqual(q_allergy("'X' có chứa sữa không?"), ["milk"])
        self.assertIn("peanut", q_allergy("'X' có nguy cơ nhiễm chéo dị nguyên không? Tôi bị dị ứng nặng đậu phộng."))
        self.assertEqual(q_allergy("Mức này có đạt điều kiện 'natri thấp' không?"), [])  # "điều" không phải hạt điều
        r = self.run_guardrail("Thành phần: Sữa bò tươi, chất ổn định (471). Sản phẩm có chứa sữa.",
                               "Bé 9 tháng tuổi ăn sữa chua 'TH true YOGURT' được không?")
        self.assertEqual(r["status"], "SCREENED_NO_ALERT")

    def test_infant_months_without_the_word_tuoi(self):
        self.assertTrue(infant_rules.is_infant({}, "Bé nhà tôi 8 tháng ăn được không?"))
        self.assertEqual(infant_rules.infant_age_months({}, "Bé nhà tôi 8 tháng ăn được không?"), 8)
        self.assertFalse(infant_rules.is_infant({}, "Tôi đã dùng sản phẩm này 8 tháng, có sao không?"))

    def test_elderly_and_lactation_get_a_note_not_a_refusal(self):
        cases = {"Sản phẩm có hợp với người cao tuổi không?": "ELDERLY_NO_SPECIFIC_EVIDENCE",
                 "Tôi đang cho con bú, ăn được không?": "LACTATION_NO_SPECIFIC_EVIDENCE"}
        for q, code in cases.items():
            r = self.run_guardrail("Thành phần: Bột mì, đường, muối.", q)
            self.assertEqual(r["status"], "SCREENED_NO_ALERT", q)
            self.assertEqual([n["code"] for n in r["population_notes"]], [code], q)
        old = {**_persona(), "demographics": {"age": 70}}
        self.assertEqual([n["code"] for n in abstention.population_notes("Ăn được không?", old)],
                         ["ELDERLY_NO_SPECIFIC_EVIDENCE"])
        self.assertEqual(abstention.population_notes("'Sữa cho người cao tuổi' có chất bảo quản không?"), [])
        self.assertEqual(abstention.population_notes("Con tôi 5 tuổi có dùng được không?"), [])

if __name__ == "__main__":
    unittest.main()

