import unittest

try:
    from scripts_research.run_three_configs import _is_approved
except ImportError:
    from scripts.run_three_configs import _is_approved


class ThreeConfigRunnerTests(unittest.TestCase):
    def test_generated_draft_is_not_eligible_for_model_comparison(self):
        row = {
            "review_status": "pending",
            "answer_supported_by_source": "",
            "citation_correct": "",
            "final_answer": "",
        }
        self.assertFalse(_is_approved(row))

    def test_only_fully_reviewed_rows_are_eligible(self):
        row = {
            "review_status": "approved",
            "answer_supported_by_source": "yes",
            "citation_correct": "yes",
            "final_answer": "Đáp án đã được chuyên gia rà soát.",
        }
        self.assertTrue(_is_approved(row))

    def test_approval_requires_source_and_citation_checks(self):
        row = {
            "review_status": "approved",
            "answer_supported_by_source": "yes",
            "citation_correct": "no",
            "final_answer": "Draft",
        }
        self.assertFalse(_is_approved(row))


if __name__ == "__main__":
    unittest.main()
