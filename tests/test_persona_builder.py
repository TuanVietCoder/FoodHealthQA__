"""Unit tests for modules/persona_builder.py."""

import unittest
from modules import persona_builder as pb
from modules import foodon as m_foodon


class TestPersonaBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foodon_cat = m_foodon.load_foodon_entities("data")

    def test_negation_words_ignored(self):
        for word in ["không", "khong co", "none", "no", "chưa phát hiện"]:
            res = pb.resolve_custom_allergen_term(word, self.foodon_cat)
            self.assertEqual(res["status"], "ignored", f"Word '{word}' should be ignored")

    def test_symptoms_warned(self):
        for sym in ["đau đầu", "mẩn ngứa", "nổi mề đay", "buồn nôn", "chóng mặt"]:
            res = pb.resolve_custom_allergen_term(sym, self.foodon_cat)
            self.assertEqual(res["status"], "symptom_warning", f"Symptom '{sym}' should trigger warning")
            self.assertIn("triệu chứng", res["warning"].lower())

    def test_drugs_warned(self):
        for drug in ["penicillin", "aspirin", "paracetamol", "kháng sinh", "panadol"]:
            res = pb.resolve_custom_allergen_term(drug, self.foodon_cat)
            self.assertEqual(res["status"], "drug_warning", f"Drug '{drug}' should trigger warning")
            self.assertTrue("dược phẩm" in res["warning"].lower() or "kháng sinh" in res["warning"].lower())

    def test_vietnamese_allergen_resolved(self):
        res = pb.resolve_custom_allergen_term("đậu phộng", self.foodon_cat)
        self.assertEqual(res["status"], "valid")
        self.assertEqual(res["key"], "peanut")
        self.assertEqual(res["curie"], "FOODON:03420115")

        res_tom = pb.resolve_custom_allergen_term("tôm", self.foodon_cat)
        self.assertEqual(res_tom["status"], "valid")
        self.assertEqual(res_tom["key"], "shellfish")

    def test_foodon_entity_resolved(self):
        res = pb.resolve_custom_allergen_term("kiwi", self.foodon_cat)
        self.assertEqual(res["status"], "valid")
        self.assertTrue(res["curie"].startswith("FOODON:"))

    def test_build_custom_persona_combination(self):
        conditions = [
            "Đái tháo đường Típ 2 (Cần kiểm soát Đường & Calo)",
            "Tăng huyết áp (Cần kiểm soát Natri / Muối)",
        ]
        allergens_common = ["Đậu phộng (Lạc)"]
        custom_terms = [
            pb.resolve_custom_allergen_term("tôm", self.foodon_cat),
            pb.resolve_custom_allergen_term("đau đầu", self.foodon_cat),  # should be skipped by caller or builder
        ]

        p = pb.build_custom_persona(
            selected_conditions=conditions,
            selected_common_allergens=allergens_common,
            custom_allergen_terms=custom_terms,
            age=52,
            gender="Nữ",
            is_pregnant=False,
        )

        self.assertIn("diabetes", p["conditions"])
        self.assertIn("hypertension", p["conditions"])
        self.assertIn("peanut", p["allergies"])
        self.assertIn("shellfish", p["allergies"])
        self.assertNotIn("đau đầu", p["allergies"])
        self.assertEqual(p["demographics"]["age"], 52)
        self.assertEqual(p["demographics"]["gender"], "nữ")
        self.assertTrue(p["is_custom"])


if __name__ == "__main__":
    unittest.main()

