import csv
import json
import os
import tempfile
import unittest

from scripts.create_review_templates import _write_csv


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _read_csv(name):
    with open(
        os.path.join(DATA_DIR, name),
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as source:
        return list(csv.DictReader(source))


class ReviewTemplateTests(unittest.TestCase):
    def test_source_grounded_qa_review_queue_preserves_draft_status(self):
        rows = _read_csv("qa_benchmark_600_source_review.csv")
        self.assertEqual(len(rows), 600)
        self.assertTrue(all(row["review_status"] == "pending" for row in rows))
        self.assertTrue(all(not row["final_answer"] for row in rows))

    def test_source_review_queue_marks_claims_pending(self):
        rows = _read_csv("knowledge_base_source_review_template.csv")
        self.assertEqual(len(rows), 100)
        pending = [
            row for row in rows
            if row["verification_status"] == "pending_primary_source_check"
        ]
        reviewed = [
            row for row in rows
            if row["verification_status"] != "pending_primary_source_check"
        ]
        source_checked = [
            row for row in rows
            if row["verification_status"] in {
                "source_checked",
                "source_checked_partial",
                "source_updated",
            }
        ]
        self.assertTrue(reviewed)
        self.assertGreater(len(pending), 40)
        self.assertTrue(all(row["source_url"] for row in source_checked))
        self.assertTrue(all(not row["reviewer_id"] for row in reviewed))
        by_id = {row["claim_id"]: row for row in rows}
        self.assertEqual(
            by_id["INS 621:adi_assessment"]["runtime_use"],
            "display_only_non_numeric_or_group_claim",
        )
        self.assertIn("INS 1420:efsa_opinion", by_id)
        self.assertIn(
            "diabetes warning was removed",
            by_id["INS 1420:efsa_opinion"]["review_notes"].lower(),
        )
        self.assertEqual(
            by_id["INS 220:efsa_opinion"]["verification_status"],
            "source_checked",
        )

    def test_active_knowledge_base_corrects_outdated_or_mismatched_claims(self):
        kb_path = os.path.join(
            os.path.dirname(__file__),
            "knowledge_base",
            "additives_standard.json",
        )
        with open(kb_path, encoding="utf-8") as source:
            additives = {
                item["ins_code"]: item
                for item in json.load(source)
            }

        modified_starch = additives["INS 1420"]
        self.assertEqual(
            modified_starch["efsa_opinion"]["reference"],
            "EFSA Journal 2017;15(10):4911",
        )
        self.assertFalse(modified_starch["contraindications"])
        self.assertNotIn("diabetes", modified_starch["efsa_opinion"]["risk_groups"])

        sulfites = additives["INS 220"]
        self.assertEqual(sulfites["efsa_opinion"]["year"], 2022)
        self.assertIsNone(sulfites["efsa_opinion"]["adi_efsa"])
        self.assertEqual(
            sulfites["efsa_opinion"]["reference"],
            "EFSA Journal 2022;20(11):7594",
        )
        self.assertNotIn(
            "asthma",
            [item["condition"] for item in sulfites["contraindications"]],
        )
        self.assertNotIn(
            "children_under_6",
            [item["condition"] for item in sulfites["contraindications"]],
        )
        self.assertFalse(sulfites["contraindications"])
        self.assertFalse(additives["INS 951"]["contraindications"])
        self.assertEqual(
            additives["INS 621"]["adi_assessment"]["verification_status"],
            "source_checked",
        )
        self.assertFalse(
            additives["INS 621"]["adi_assessment"]["use_for_numeric_compliance"]
        )

    def test_template_regeneration_preserves_manual_review_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "review.csv")
            fields = ["id", "draft", "review_status", "final_answer"]
            _write_csv(path, fields, [{
                "id": "case-1",
                "draft": "old draft",
                "review_status": "approved",
                "final_answer": "reviewed answer",
            }])

            _write_csv(
                path,
                fields,
                [{
                    "id": "case-1",
                    "draft": "updated draft",
                    "review_status": "pending",
                    "final_answer": "",
                }],
                key_field="id",
                preserve_fields=["review_status", "final_answer"],
            )

            with open(path, encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]["draft"], "updated draft")
            self.assertEqual(rows[0]["review_status"], "approved")
            self.assertEqual(rows[0]["final_answer"], "reviewed answer")


if __name__ == "__main__":
    unittest.main()
