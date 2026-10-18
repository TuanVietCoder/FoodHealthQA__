import csv
import json
import os
import unittest
from collections import Counter

from modules.pipeline import load_fndds_nutrition


ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
QA_PATH = os.path.join(DATA_DIR, "qa_benchmark_600_source_grounded.jsonl")
REVIEW_PATH = os.path.join(DATA_DIR, "qa_benchmark_600_source_review.csv")
METADATA_PATH = os.path.join(DATA_DIR, "qa_benchmark_600_source_metadata.json")


class SourceGroundedQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(QA_PATH, encoding="utf-8") as source:
            cls.records = [json.loads(line) for line in source if line.strip()]
        with open(METADATA_PATH, encoding="utf-8") as source:
            cls.metadata = json.load(source)
        with open(REVIEW_PATH, encoding="utf-8-sig", newline="") as source:
            cls.review_rows = list(csv.DictReader(source))

    def test_dataset_has_600_unique_questions_balanced_by_domain(self):
        self.assertEqual(len(self.records), 600)
        self.assertEqual(len({row["id"] for row in self.records}), 600)
        self.assertEqual(len({row["question"] for row in self.records}), 600)
        self.assertEqual(
            Counter(row["domain"] for row in self.records),
            {
                "additive_safety": 100,
                "allergen_safety": 100,
                "pregnancy_safety": 100,
                "children_safety": 100,
                "condition_safety": 100,
                "nutrition_facts": 100,
            },
        )

    def test_records_link_to_known_sources_and_retain_grouped_splits(self):
        source_ids = set(self.metadata["sources"])
        for row in self.records:
            self.assertTrue(row["source_ids"], row["id"])
            self.assertTrue(row["source_urls"], row["id"])
            self.assertTrue(set(row["source_ids"]).issubset(source_ids), row["id"])
            self.assertEqual(len(row["source_ids"]), len(row["source_urls"]), row["id"])
            for source_id in row["source_ids"]:
                self.assertIn(
                    self.metadata["sources"][source_id]["url"],
                    row["source_urls"],
                    row["id"],
                )
            self.assertTrue(row["evidence_excerpt"], row["id"])
            self.assertEqual(
                row["review_status"],
                "draft_requires_source_and_expert_review",
            )
            self.assertFalse(row["is_expert_validated_gold"])
            self.assertEqual(row["split_group_id"], row["question_group_id"])

    def test_health_domains_do_not_claim_personalized_medical_safety(self):
        for row in self.records:
            if row["domain"] in {
                "pregnancy_safety",
                "children_safety",
                "condition_safety",
                "allergen_safety",
            }:
                self.assertTrue(row["not_individualized_medical_advice"], row["id"])
        pregnancy = [row for row in self.records if row["domain"] == "pregnancy_safety"]
        self.assertTrue(all(row["requires_clinician_referral"] for row in pregnancy))

    def test_codex_answers_match_linked_text_and_state_its_limitations(self):
        codex = self.metadata["sources"]["CODEX_CXS_1"]
        self.assertIn("amended through 2010", codex["title"])
        self.assertIn("4.2.1.4", codex["locator"])
        self.assertIn("6", codex["locator"])
        self.assertIn("Verify any later Codex amendment", codex["scope_warning"])

        allergen_rows = [
            row for row in self.records if row["domain"] == "allergen_safety"
        ]
        allergen_list_rows = [
            row for row in allergen_rows
            if row["question_group_id"].startswith("allergen:codex_allergen_")
        ]
        self.assertTrue(all("mục 4.2.1.4" in row["reference_answer"] for row in allergen_list_rows))
        self.assertTrue(all("Mục 6" in row["reference_answer"] for row in allergen_list_rows))
        self.assertFalse(any("Annex II" in row["reference_answer"] for row in allergen_list_rows))
        sulfite_rows = [
            row for row in allergen_rows
            if row["question_group_id"] == "allergen:codex_allergen_09"
        ]
        self.assertTrue(all("10 mg/kg trở lên" in row["evidence_excerpt"] for row in sulfite_rows))
        self.assertTrue(all("10 mg/kg trở lên" in row["reference_answer"] for row in sulfite_rows))

    def test_child_and_condition_questions_are_natural_and_source_specific(self):
        child_rows = [row for row in self.records if row["domain"] == "children_safety"]
        self.assertFalse(any("WHO, WHO " in row["question"] for row in child_rows))
        self.assertFalse(
            any("Khuyến nghị WHO nào trả lời câu hỏi:" in row["question"] for row in child_rows)
        )
        complementary = [
            row for row in child_rows
            if row["question_group_id"] == "children:complementary_start"
        ]
        self.assertEqual(
            {source_id for row in complementary for source_id in row["source_ids"]},
            {"WHO_COMPLEMENTARY_FEEDING"},
        )

        condition_rows = [
            row for row in self.records if row["domain"] == "condition_safety"
        ]
        self.assertFalse(any("Khuyến nghị WHO nêu điều gì về" in row["question"] for row in condition_rows))
        self.assertFalse(any("Mức hoặc nguyên tắc WHO được nêu là gì?" in row["question"] for row in condition_rows))
        self.assertFalse(any("WHO, WHO " in row["question"] for row in condition_rows))

    def test_nutrition_values_match_local_fndds_rows(self):
        fndds = {
            str(row["food_code"]): row
            for row in load_fndds_nutrition(DATA_DIR)
        }
        nutrition = [row for row in self.records if row["domain"] == "nutrition_facts"]
        for record in nutrition:
            food = fndds[record["food_code"]]
            self.assertEqual(
                record["value_per_100g"],
                food[record["nutrient_field"]],
                record["id"],
            )
            for source_ref in record["source_refs"]:
                self.assertTrue(os.path.isfile(os.path.join(ROOT, source_ref)))

    def test_review_form_contains_all_records_as_pending(self):
        self.assertEqual(len(self.review_rows), 600)
        self.assertTrue(all(row["review_status"] == "pending" for row in self.review_rows))
        self.assertTrue(all(not row["reviewer_id"] for row in self.review_rows))
        self.assertEqual(
            {row["id"] for row in self.review_rows},
            {row["id"] for row in self.records},
        )
        self.assertFalse(self.metadata["is_expert_validated_gold"])
        self.assertTrue(self.metadata["human_review_required"])


if __name__ == "__main__":
    unittest.main()
