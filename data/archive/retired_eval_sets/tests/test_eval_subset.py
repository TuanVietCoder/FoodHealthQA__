import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_research"))

import build_eval_subset as bs  # noqa: E402

SRC = bs.load()


class EvalSubsetTests(unittest.TestCase):
    def setUp(self):
        self.sub = bs.build(SRC, max_per_template=30, max_per_image=4, majority_ratio=2.0, seed=42)

    def test_subset_records_are_real_benchmark_records(self):
        ids = {r["id"] for r in SRC}
        self.assertTrue(all(r["id"] in ids for r in self.sub))
        self.assertEqual(len({r["id"] for r in self.sub}), len(self.sub))

    def test_caps_respected(self):
        self.assertLessEqual(max(Counter(r["template_id"] for r in self.sub).values()), 30)
        self.assertLessEqual(max(Counter(r["split_group"] for r in self.sub).values()), 4)

    def test_majority_baseline_does_not_get_worse(self):
        def maj(rows, d):
            c = Counter(r["gold_safety_label"] for r in rows if r["domain"] == d)
            return max(c.values()) / sum(c.values())

        for d in {r["domain"] for r in SRC}:
            # a domain that is already balanced may move by sampling noise; it must not get materially worse
            self.assertLessEqual(maj(self.sub, d), maj(SRC, d) + 0.03, d)
        # the strongly skewed domain must clearly improve
        self.assertLess(maj(self.sub, "additive_regulation"), maj(SRC, "additive_regulation") - 0.1)

    def test_deterministic(self):
        again = bs.build(SRC, 30, 4, 2.0, 42)
        self.assertEqual([r["id"] for r in again], [r["id"] for r in self.sub])


if __name__ == "__main__":
    unittest.main()
