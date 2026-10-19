import json
import os
import unittest

from modules.constraint_model import (
    CONSTRAINT_ALLERGEN_RISK,
    classify_constraint_type,
    evaluate_adi_compliance,
    evaluate_regulatory_compliance,
)
from modules.foodon import load_foodon_entities
from modules.pipeline import evaluate_safety_and_guardrail


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(ROOT, "knowledge_base")
DATA_DIR = os.path.join(ROOT, "data")


def _read_json(path):
    with open(path, encoding="utf-8") as source:
        return json.load(source)


class KnowledgeBaseHardeningTests(unittest.TestCase):
    def test_foodon_ingredient_catalog_is_broader_but_not_overclaimed(self):
        ingredients = _read_json(os.path.join(KB_DIR, "ingredients_standard.json"))
        catalog = load_foodon_entities(DATA_DIR)
        common = [
            item for item in ingredients
            if item["ingredient_type"] == "common_food_ingredient"
        ]
        self.assertGreaterEqual(len(common), 30)
        for item in common:
            entity = catalog.by_curie[item["foodon_curie"]]
            self.assertEqual(entity["label"], item["foodon_label"])
            self.assertEqual(item["verification_status"], "source_checked_for_ontology_identity_only")
            self.assertEqual(item["alias_verification_status"], "pending_label_review")
            self.assertFalse(item["use_for_allergen_screening"])

        beef = next(item for item in ingredients if item["id"] == "beef")
        self.assertFalse(beef["use_for_allergen_screening"])
        self.assertIsNone(beef["allergen_group_id"])

    def test_regulatory_claims_are_jurisdiction_scoped_and_pending_for_vietnam(self):
        groups = _read_json(os.path.join(KB_DIR, "allergen_groups.json"))
        for group in groups:
            self.assertNotIn("regulation", group)
            vietnam = group["regulatory_claims"]["vietnam"]
            self.assertEqual(
                vietnam["verification_status"],
                "pending_official_text_and_amendment_check",
            )
            self.assertFalse(vietnam["use_for_compliance_decision"])
            for claim in group["regulatory_claims"].values():
                self.assertFalse(claim["use_for_compliance_decision"])
            if group["enabled_for_allergen_screening"]:
                self.assertEqual(group["severity"], "warning")
                self.assertEqual(
                    group["keyword_vocabulary_status"],
                    "curated_aliases_pending_expert_review",
                )
                self.assertTrue(
                    all(value == "ALLERGEN_RISK" for value in group["negation_rules"].values())
                )

        additives = _read_json(os.path.join(KB_DIR, "additives_standard.json"))
        for additive in additives:
            if additive["adi_assessment"]["verification_status"] != "source_checked":
                self.assertIsNone(additive["adi_upper"])
                self.assertIsNone(
                    additive["adi_assessment"]["numeric_upper_mg_per_kg_bw_day"]
                )
        result = evaluate_regulatory_compliance(additives[0])
        self.assertEqual(result["vn_regulation"]["status"], "SOURCE_REVIEW_REQUIRED")
        self.assertFalse(result["vn_regulation"]["use_for_compliance_decision"])
        self.assertIn("pending", additives[0]["vn_regulation"].lower())
        self.assertFalse(
            additives[0]["unverified_legacy_regulatory_claims"]["use_for_compliance_decision"]
        )

    def test_group_adi_is_not_calculated_as_an_individual_additive_limit(self):
        additives = _read_json(os.path.join(KB_DIR, "additives_standard.json"))
        glutamate = next(item for item in additives if item["ins_code"] == "INS 621")
        self.assertEqual(glutamate["adi_assessment"]["group_adi_mg_per_kg_bw_day"], 30.0)
        self.assertIsNone(glutamate["adi_assessment"]["numeric_upper_mg_per_kg_bw_day"])
        self.assertIsNone(glutamate["adi_upper"])
        result = evaluate_adi_compliance(glutamate)
        self.assertEqual(result["compliance_status"], "NOT_NUMERICALLY_APPLICABLE")
        self.assertIsNone(result["max_daily_intake_mg"])

        unverified = next(item for item in additives if item["ins_code"] == "INS 211")
        pending_result = evaluate_adi_compliance(unverified)
        self.assertEqual(pending_result["compliance_status"], "SOURCE_REVIEW_REQUIRED")
        self.assertIsNone(pending_result["adi_parsed"])

        legacy_result = evaluate_adi_compliance({
            "ins_code": "INS 211",
            "canonical_name": "Sodium benzoate",
            "jecfa_adi": "0-5 mg/kg bw/day",
        })
        self.assertEqual(legacy_result["compliance_status"], "SOURCE_REVIEW_REQUIRED")
        self.assertIsNone(legacy_result["adi_parsed"])

        modified_starch = next(item for item in additives if item["ins_code"] == "INS 1420")
        no_numeric_adi = evaluate_adi_compliance(modified_starch)
        self.assertEqual(no_numeric_adi["compliance_status"], "NO_NUMERIC_ADI")

    def test_allergen_keyword_match_is_screening_warning_not_contraindication(self):
        result = classify_constraint_type(
            "peanut",
            None,
            {"conditions": [], "allergies": ["peanut"]},
        )
        self.assertEqual(result["constraint_type"], CONSTRAINT_ALLERGEN_RISK)
        false_positive = classify_constraint_type(
            "mì chính",
            None,
            {"conditions": [], "allergies": ["gluten"]},
        )
        self.assertNotEqual(false_positive["constraint_type"], CONSTRAINT_ALLERGEN_RISK)

    def test_unverified_legacy_contraindications_do_not_trigger_critical_alerts(self):
        evaluation = evaluate_safety_and_guardrail(
            [{
                "raw_mention": "Sodium benzoate",
                "retrieval_confidence": 0.9,
                "matched_entity": {
                    "canonical_name": "Sodium benzoate",
                    "ins_code": "INS 211",
                    "contraindications": [{
                        "condition": "hypertension",
                        "severity": "CRITICAL",
                        "message": "unverified legacy claim",
                    }],
                },
            }],
            {"conditions": ["hypertension"], "allergies": []},
            "Can I eat this?",
        )
        self.assertNotEqual(evaluation["status"], "CRITICAL_ALERT")
        self.assertFalse(evaluation["alerts"])


if __name__ == "__main__":
    unittest.main()
