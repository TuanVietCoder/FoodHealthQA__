import csv
import json
import os
import unittest
from collections import Counter


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MERGED_CSV_PATH = os.path.join(DATA_DIR, "qa_benchmark_merged_review.csv")


class SourceGroundedQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(MERGED_CSV_PATH, encoding="utf-8-sig", newline="") as source:
            cls.records = list(csv.DictReader(source))

    def test_merged_benchmark_has_expected_records_and_unique_ids(self):
        self.assertGreaterEqual(len(self.records), 750)
        ids = [row["id"] for row in self.records]
        self.assertEqual(len(ids), len(set(ids)))

    def test_domains_and_safety_labels_are_covered(self):
        domains = {row["domain"] for row in self.records if row.get("domain")}
        self.assertIn("allergen_safety", domains)
        self.assertIn("additive_safety", domains)
        self.assertIn("pregnancy_safety", domains)

        safety_labels = {row["gold_safety_label"] for row in self.records if row.get("gold_safety_label")}
        self.assertIn("SAFE", safety_labels)
        self.assertIn("UNSAFE", safety_labels)

    def test_persona_and_product_scenarios_have_required_fields(self):
        scenario_records = [r for r in self.records if r["id"].startswith("FOODQA_")]
        self.assertGreaterEqual(len(scenario_records), 600)
        for row in scenario_records:
            self.assertTrue(bool(row["question"]))
            self.assertTrue(bool(row["reference_answer"]))
            self.assertTrue(bool(row["persona_id"]))
            self.assertTrue(bool(row["product_name"]))
            self.assertTrue(bool(row["product_ingredients"]))
            self.assertIn(row["gold_safety_label"], {"SAFE", "UNSAFE", "CAUTION"})

    def test_advanced_allergen_scenarios_are_approved(self):
        adv_records = [r for r in self.records if "ADV" in r["id"]]
        self.assertEqual(len(adv_records), 5)
        for r in adv_records:
            self.assertEqual(r["review_status"], "approved")
            self.assertTrue(bool(r["source_citations"]))

    def test_source_citations_contain_recognized_authorities(self):
        citations_text = " ".join(r.get("source_citations", "") for r in self.records)
        self.assertTrue(any(auth in citations_text for auth in ["WHO", "FAO", "FDA", "Codex", "EFSA"]))


if __name__ == "__main__":
    unittest.main()
