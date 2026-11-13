import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_research"))

import build_foodhealthqa_subset as builder  # noqa: E402


class FoodHealthQASubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = builder.load_benchmark(builder.DEFAULT_INPUT)
        cls.sub = builder.build_subset(cls.rows, target_size=750, seed=42, max_per_image=4)

    def test_record_count_within_500_to_1000(self):
        self.assertGreaterEqual(len(self.sub), 500)
        self.assertLessEqual(len(self.sub), 1000)
        self.assertEqual(len(self.sub), 750)

    def test_mandatory_fields_present_and_non_empty(self):
        for r in self.sub:
            # 1. Câu hỏi
            self.assertTrue(r.get("question"), f"Empty question in {r.get('id')}")
            # 2. Câu trả lời
            self.assertTrue(r.get("reference_answer"), f"Empty reference_answer in {r.get('id')}")
            # 3. Nguồn tham khảo
            self.assertTrue(r.get("source_citations"), f"Empty source_citations in {r.get('id')}")
            # 4. Nhãn lĩnh vực
            self.assertTrue(r.get("category_vi"), f"Empty category_vi in {r.get('id')}")
            self.assertIn(r.get("category_key"), builder.PILLAR_WEIGHTS)

    def test_all_six_pillars_adequately_represented(self):
        counts = Counter(r["category_key"] for r in self.sub)
        self.assertEqual(len(counts), 6, "Must contain all 6 pillars")
        # Every pillar must have at least 50 questions
        for key in builder.PILLAR_WEIGHTS:
            self.assertGreaterEqual(
                counts[key], 50, f"Pillar '{key}' has fewer than 50 questions ({counts[key]})"
            )

    def test_all_safety_labels_represented(self):
        labels = Counter(r["gold_safety_label"] for r in self.sub)
        expected = {"SAFE", "CAUTION", "UNSAFE", "ABSTAIN", "CRITICAL_ALERT"}
        self.assertEqual(set(labels.keys()), expected)
        for lbl in expected:
            self.assertGreaterEqual(labels[lbl], 20, f"Label {lbl} too scarce")

    def test_packaging_photo_diversity(self):
        img_counts = Counter(r["split_group"] for r in self.sub)
        # Should cover at least 300 distinct packaging photos
        self.assertGreaterEqual(len(img_counts), 300)
        # Max per image cap respected
        self.assertLessEqual(max(img_counts.values()), 4)

    def test_deterministic_reproducibility(self):
        again = builder.build_subset(self.rows, target_size=750, seed=42, max_per_image=4)
        self.assertEqual([r["id"] for r in again], [r["id"] for r in self.sub])


if __name__ == "__main__":
    unittest.main()

