"""
Unit tests for Advanced Allergen Features:
- 3-block label parsing (Ingredients vs Contains vs PAL)
- US FDA FALCPA Highly Refined Oil exemption
- FDA Guidance Ed. 5 derivative allergen mapping (lecithin, casein, lysozyme)
- Precautionary Allergen Labeling (PAL) guardrail handling (FAO/WHO JEMRA Part 3)
- Milk protein allergy (IgE) vs Lactose intolerance distinction (FAO/WHO JEMRA 2021)
"""

import unittest
from modules.ocr_extractor import parse_allergen_statements, parse_ingredients_from_ocr
from modules.constraint_model import (
    classify_constraint_type,
    is_exempt_refined_oil,
    FDA_BIG_9_ALLERGENS,
    is_fda_major_allergen,
    get_fda_allergen_info,
    CONSTRAINT_PERMITTED,
    CONSTRAINT_ALLERGEN_RISK,
    CONSTRAINT_PRECAUTIONARY_RISK,
)
from modules.pipeline import evaluate_safety_and_guardrail


class AdvancedAllergenTests(unittest.TestCase):
    def test_three_block_allergen_statement_parsing(self):
        sample_label = (
            "Thành phần: Bột mì, đường, bơ thực vật, muối. "
            "Thông tin dị ứng: Chứa bột mì, sữa và trứng. "
            "Có thể chứa vết của hạt đậu phộng, mè."
        )
        statements = parse_allergen_statements(sample_label)
        self.assertTrue(statements["has_contains"])
        self.assertTrue(statements["has_pal"])
        self.assertIn("bột mì", [x.lower() for x in statements["contains_allergens"]])
        self.assertIn("sữa", [x.lower() for x in statements["contains_allergens"]])
        self.assertTrue(any("đậu phộng" in x.lower() for x in statements["pal_allergens"]))
        self.assertTrue(any("mè" in x.lower() for x in statements["pal_allergens"]))

    def test_refined_oil_falcpa_exemption(self):
        # Refined oils must be exempt from allergen risk
        self.assertTrue(is_exempt_refined_oil("dầu đậu nành tinh luyện"))
        self.assertTrue(is_exempt_refined_oil("dầu phộng tinh luyện hoàn toàn"))
        self.assertTrue(is_exempt_refined_oil("highly refined soybean oil"))
        self.assertFalse(is_exempt_refined_oil("đạm đậu nành"))
        self.assertFalse(is_exempt_refined_oil("bột đậu phộng"))

        # Constraint classification for soy allergy
        soy_persona = {"conditions": [], "allergies": ["soy"]}
        res_exempt = classify_constraint_type("Dầu đậu nành tinh luyện", None, soy_persona)
        self.assertEqual(res_exempt["constraint_type"], CONSTRAINT_PERMITTED)
        self.assertTrue(res_exempt["is_refined_oil_exempt"])

        # Unrefined soy must still trigger ALLERGEN_RISK
        res_unrefined = classify_constraint_type("Đạm đậu nành cô đặc", None, soy_persona)
        self.assertEqual(res_unrefined["constraint_type"], CONSTRAINT_ALLERGEN_RISK)

    def test_fda_guidance_ed5_derivatives(self):
        # Lecithin -> Soy allergy
        soy_persona = {"conditions": [], "allergies": ["soy"]}
        res_lecithin = classify_constraint_type("Lecithin INS 322", None, soy_persona)
        self.assertEqual(res_lecithin["constraint_type"], CONSTRAINT_ALLERGEN_RISK)

        # Casein / Whey -> Milk allergy
        milk_persona = {"conditions": [], "allergies": ["milk"]}
        res_casein = classify_constraint_type("Natri caseinat", None, milk_persona)
        self.assertEqual(res_casein["constraint_type"], CONSTRAINT_ALLERGEN_RISK)

        # Lysozyme -> Egg allergy
        egg_persona = {"conditions": [], "allergies": ["egg"]}
        res_lysozyme = classify_constraint_type("Lysozyme (INS 1105)", None, egg_persona)
        self.assertEqual(res_lysozyme["constraint_type"], CONSTRAINT_ALLERGEN_RISK)

    def test_pal_precautionary_vs_intentional_ingredient(self):
        peanut_persona = {"conditions": [], "allergies": ["peanut"]}

        # Intentional peanut
        res_direct = classify_constraint_type("Đậu phộng rang muối", None, peanut_persona, is_precautionary=False)
        self.assertEqual(res_direct["constraint_type"], CONSTRAINT_ALLERGEN_RISK)

        # Precautionary cross-contact (PAL)
        res_pal = classify_constraint_type("Có thể chứa vết đậu phộng", None, peanut_persona, is_precautionary=True)
        self.assertEqual(res_pal["constraint_type"], CONSTRAINT_PRECAUTIONARY_RISK)

    def test_milk_allergy_vs_lactose_intolerance_distinction(self):
        # 1. Person with IgE Milk Allergy
        milk_allergy_persona = {
            "name": "Bệnh nhân dị ứng đạm sữa",
            "conditions": [],
            "allergies": ["milk"],
            "weight_kg": 60.0
        }
        linked_milk = [{
            "raw_mention": "sữa bột béo",
            "retrieval_confidence": 0.95,
            "matched_entity": None
        }]
        eval_allergy = evaluate_safety_and_guardrail(linked_milk, milk_allergy_persona, "Uống được không?")
        self.assertEqual(eval_allergy["status"], "WARNING")
        self.assertTrue(any("dị nguyên" in a["message"].lower() for a in eval_allergy["alerts"]))

        # 2. Person with Lactose Intolerance (Enzyme deficiency, no IgE allergy)
        lactose_intolerant_persona = {
            "name": "Người không dung nạp lactose",
            "conditions": ["lactose_intolerance"],
            "allergies": [],
            "weight_kg": 60.0
        }
        eval_lactose = evaluate_safety_and_guardrail(linked_milk, lactose_intolerant_persona, "Uống được không?")
        self.assertEqual(eval_lactose["status"], "CAUTION")
        self.assertTrue(any("không dung nạp" in c["message"].lower() for c in eval_lactose["alerts"]))

    def test_fda_big_9_allergens_registry_and_rules(self):
        # Must have exactly 9 major FDA allergen categories
        self.assertEqual(len(FDA_BIG_9_ALLERGENS), 9)
        required_groups = {
            "milk", "egg", "peanut", "tree_nuts", "wheat",
            "fish", "crustacean_shellfish", "soybeans", "sesame"
        }
        self.assertEqual(set(FDA_BIG_9_ALLERGENS.keys()), required_groups)

        # Sesame must cite FASTER Act 2023
        sesame_info = FDA_BIG_9_ALLERGENS["sesame"]
        self.assertEqual(sesame_info["citation"], "US_FDA_FASTER_ACT_2023")
        self.assertEqual(sesame_info["statute"], "FASTER_ACT_2021")

        # FALCPA 2004 citations for classical groups
        self.assertEqual(FDA_BIG_9_ALLERGENS["milk"]["citation"], "US_FDA_FALCPA")
        self.assertEqual(FDA_BIG_9_ALLERGENS["soybeans"]["citation"], "US_FDA_FALCPA")

        # Helper detection
        self.assertTrue(is_fda_major_allergen("soybeans"))
        self.assertTrue(is_fda_major_allergen("đậu nành"))
        self.assertTrue(is_fda_major_allergen("sesame"))
        self.assertTrue(is_fda_major_allergen("vừng"))
        self.assertTrue(is_fda_major_allergen("crustacean shellfish"))

        grp, info = get_fda_allergen_info("sesame")
        self.assertEqual(grp, "sesame")
        self.assertEqual(info["citation"], "US_FDA_FASTER_ACT_2023")

    def test_refined_soybean_oil_specific_exemption(self):
        soy_persona = {"conditions": [], "allergies": ["soybeans"]}
        res1 = classify_constraint_type("dầu đậu nành tinh luyện hoàn toàn", None, soy_persona)
        self.assertEqual(res1["constraint_type"], CONSTRAINT_PERMITTED)
        self.assertTrue(res1["is_refined_oil_exempt"])

        res2 = classify_constraint_type("refined soybean oil", None, soy_persona)
        self.assertEqual(res2["constraint_type"], CONSTRAINT_PERMITTED)
        self.assertTrue(res2["is_refined_oil_exempt"])

    def test_parse_allergen_statements_standalone_contains(self):
        # Case with Contains at very start of text
        text_start = "Contains: Milk, Wheat. May contain: Peanuts."
        res_start = parse_allergen_statements(text_start)
        self.assertTrue(res_start["has_contains"])
        self.assertTrue(res_start["has_may_contain"])
        self.assertIn("Milk", res_start["contains_allergens"])
        self.assertIn("Wheat", res_start["contains_allergens"])
        self.assertIn("Peanuts", res_start["may_contain_allergens"])

        # Case with Contains preceded by newline
        text_nl = "Ingredients: Sugar, Flour.\nContains: Soybeans.\nMay contain: Sesame."
        res_nl = parse_allergen_statements(text_nl)
        self.assertTrue(res_nl["has_contains"])
        self.assertTrue(res_nl["has_may_contain"])
        self.assertIn("Soybeans", res_nl["contains_allergens"])
        self.assertIn("Sesame", res_nl["may_contain_allergens"])


class FdaQa5ScopeTests(unittest.TestCase):
    """FDA Q&A Ed.5: tree-nut list (Table 1), plant milks and voluntary 'milk-free' claims."""

    def _risk(self, text, allergy):
        persona = {"allergies": [allergy], "conditions": []}
        return classify_constraint_type(text, None, persona)["constraint_type"]

    def test_plant_milk_and_milk_free_claims_are_not_milk_evidence(self):
        self.assertEqual(self._risk("Sữa chiết từ đậu nành nguyên hạt", "milk"), CONSTRAINT_PERMITTED)
        self.assertEqual(self._risk("Nước cốt dừa", "milk"), CONSTRAINT_PERMITTED)
        self.assertEqual(self._risk("Natri caseinat", "milk"), CONSTRAINT_ALLERGEN_RISK)
        self.assertEqual(self._risk("Sữa bột gầy", "milk"), CONSTRAINT_ALLERGEN_RISK)
        # a non-dairy creamer stays flagged: such creamers commonly contain caseinate
        self.assertEqual(self._risk("Bột kem không sữa", "milk"), CONSTRAINT_ALLERGEN_RISK)

    def test_tree_nuts_follow_fda_table_1(self):
        for nut in ("hạt dẻ cười", "hạt thông", "pecan", "hạt hạnh nhân", "hạt điều"):
            self.assertEqual(self._risk(nut, "nuts"), CONSTRAINT_ALLERGEN_RISK, nut)
        self.assertEqual(self._risk("dừa", "nuts"), CONSTRAINT_PERMITTED)

    def test_benchmark_rules_use_table_1_only(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_research"))
        import create_benchmark_v5 as b
        for nut in ("hạt dẻ cười", "hạt thông", "pecan", "mắc ca", "hạt điều"):
            self.assertTrue(b.has_term(nut, "hạt cây"), nut)
        for other in ("hạt dẻ", "chestnut", "dừa", "coconut", "hạt bí xanh"):
            self.assertFalse(b.has_term(other, "hạt cây"), other)
        self.assertFalse(b.has_term("bột kem không sữa", "sữa"))
        self.assertFalse(b.has_term("sữa chiết từ đậu nành", "sữa"))
        self.assertTrue(b.has_term("sữa bột gầy", "sữa"))


class JemraPart1CatalogTests(unittest.TestCase):
    """The allergen catalog must not claim more than the JEMRA 2021 Part 1 summary says."""

    def setUp(self):
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "knowledge_base" / "allergen_groups.json"
        self.groups = {g["id"]: g for g in json.loads(path.read_text(encoding="utf-8"))}

    def test_buckwheat_is_a_regional_not_global_priority_group(self):
        bw = self.groups["buckwheat"]
        self.assertEqual(bw["jemra_priority"], "REGIONAL_PRIORITY")
        self.assertTrue(bw["enabled_for_allergen_screening"])
        self.assertIn("kiều mạch", bw["keywords"])
        self.assertNotIn("kiều mạch", self.groups["gluten"]["keywords"])

    def test_groups_the_summary_does_not_conclude_on_are_not_called_regional_priority(self):
        for gid in ("molluscs", "beef"):
            self.assertNotEqual(self.groups[gid]["jemra_priority"], "REGIONAL_PRIORITY", gid)
            self.assertNotIn("regional priority allergen", self.groups[gid]["note"].lower(), gid)
            self.assertNotIn("REGIONAL_PRIORITY", self.groups[gid]["regulatory_claims"]["fao_who_jemra"]["claim"], gid)
        for gid in ("celery", "mustard", "lupin", "soy", "buckwheat"):
            self.assertEqual(self.groups[gid]["jemra_priority"], "REGIONAL_PRIORITY", gid)

    def test_regional_groups_carry_no_severity_or_prevalence_claim_the_summary_does_not_make(self):
        for gid in ("celery", "mustard", "lupin", "soy", "buckwheat", "oats"):
            g = self.groups[gid]
            self.assertTrue(g["clinical_severity"].startswith("Not stated"), gid)
            text = (g["note"] + g["regulatory_claims"]["fao_who_jemra"]["claim"]).lower()
            for banned in ("central europe", "cross-reactivity", "lower severity", "lower prevalence"):
                self.assertNotIn(banned, text, gid)


class JemraTierAndOatsTests(unittest.TestCase):
    """JEMRA 2021 Part 1: oats are not gluten cereals; priority tier is shown to the user; watch list is not an alert."""

    def _alerts(self, mention, allergy):
        persona = {"name": "t", "conditions": [], "allergies": [allergy], "weight_kg": 60.0}
        linked = [{"raw_mention": mention, "retrieval_confidence": 0.95, "matched_entity": None}]
        return evaluate_safety_and_guardrail(linked, persona, "Dùng được không?")["alerts"]

    def test_oats_are_a_precaution_for_gluten_and_a_direct_hit_for_oat_allergy(self):
        self.assertEqual(classify_constraint_type("Yến mạch cán dẹt", None, {"allergies": ["gluten"], "conditions": []})["constraint_type"],
                         CONSTRAINT_PRECAUTIONARY_RISK)
        self.assertEqual(classify_constraint_type("Lúa mạch", None, {"allergies": ["gluten"], "conditions": []})["constraint_type"],
                         CONSTRAINT_ALLERGEN_RISK)
        self.assertEqual(classify_constraint_type("Yến mạch cán dẹt", None, {"allergies": ["oats"], "conditions": []})["constraint_type"],
                         CONSTRAINT_ALLERGEN_RISK)

    def test_priority_tier_is_reported_but_does_not_change_severity(self):
        peanut = classify_constraint_type("Đậu phộng rang", None, {"allergies": ["peanut"], "conditions": []})
        celery = classify_constraint_type("Cần tây", None, {"allergies": ["celery"], "conditions": []})
        self.assertEqual(peanut["priority_tier"], "GLOBAL_PRIORITY")
        self.assertEqual(celery["priority_tier"], "REGIONAL_PRIORITY")
        self.assertEqual(peanut["constraint_type"], celery["constraint_type"])

    def test_pipeline_alert_carries_the_tier_text(self):
        alerts = self._alerts("Đậu phộng rang muối", "peanut")
        self.assertTrue(any("ưu tiên toàn cầu" in a["message"] for a in alerts))
        alerts = self._alerts("Bột kiều mạch", "buckwheat")
        self.assertTrue(any("khu vực" in a["message"] for a in alerts))

    def test_oats_for_gluten_user_raise_a_cross_contact_alert(self):
        alerts = self._alerts("Yến mạch cán dẹt", "gluten")
        self.assertTrue(alerts)
        self.assertTrue(any(a.get("regulation") == "FAO_WHO_JEMRA_PART_3" for a in alerts))

    def test_watch_list_groups_are_not_used_for_alerts(self):
        from modules.constraint_model import ALLERGEN_KEYWORDS
        for gid in ("watch_pulses", "watch_insects", "watch_kiwi"):
            self.assertNotIn(gid, ALLERGEN_KEYWORDS)


if __name__ == "__main__":
    unittest.main()
