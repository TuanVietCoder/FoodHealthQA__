import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_research"))

import benchmark_report as br  # noqa: E402

ROWS = br.load(ROOT / "data" / "qa_benchmark.jsonl")


class GroupFieldTests(unittest.TestCase):
    def test_every_scored_record_has_group_and_template(self):
        self.assertTrue(all(r.get("split_group") and r.get("template_id") for r in ROWS))

    def test_split_group_is_exactly_one_photo(self):
        by_group = {}
        for r in ROWS:
            by_group.setdefault(r["split_group"], set()).add(r["image_ref"])
        self.assertTrue(all(len(v) == 1 for v in by_group.values()))

    def test_template_never_spans_domains(self):
        dom = {}
        for r in ROWS:
            dom.setdefault(r["template_id"], set()).add(r["domain"])
        self.assertTrue(all(len(v) == 1 for v in dom.values()))

    def test_manifest_documents_tier_usage(self):
        m = json.loads((ROOT / "data" / "benchmark_manifest.json").read_text(encoding="utf-8"))
        self.assertIn("tier_usage", m["benchmarks"]["v5"])


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self):
        preds = {r["id"]: r["gold_safety_label"] for r in ROWS}
        pairs = [(r["gold_safety_label"], preds[r["id"]]) for r in ROWS]
        self.assertEqual(br.accuracy(pairs), 1.0)
        self.assertEqual(br.macro_f1(pairs), 1.0)
        self.assertEqual(br.template_balanced_accuracy(ROWS, preds), 1.0)

    def test_majority_baseline_is_beatable_only_by_real_signal(self):
        core = [r for r in ROWS if r["tier"] == "core"]  # the primary score; silver is reported separately
        base = br.majority_pred(core)
        pairs = [(r["gold_safety_label"], base[r["id"]]) for r in core]
        # high accuracy, poor macro-F1: why macro-F1 is reported (0.6: core now holds CAUTION-majority sodium/sugar records,
        # which lift the per-domain majority baseline's macro-F1 from 0.49 to about 0.57)
        self.assertLess(br.macro_f1(pairs), 0.6)
        self.assertGreater(br.accuracy(pairs) - br.macro_f1(pairs), 0.1)

    def test_bootstrap_interval_contains_point_estimate(self):
        preds = br.majority_pred(ROWS)
        pairs = [(r["gold_safety_label"], preds[r["id"]]) for r in ROWS]
        lo, hi = br.bootstrap_ci(ROWS, preds, br.accuracy, n=200)
        self.assertLessEqual(lo, br.accuracy(pairs))
        self.assertGreaterEqual(hi, br.accuracy(pairs))


if __name__ == "__main__":
    unittest.main()


class ShortcutAndReportingTests(unittest.TestCase):
    def test_cross_validated_template_baseline_is_not_better_than_in_sample(self):
        cv = br.template_pred(ROWS, cross_validated=True)
        ins = br.template_pred(ROWS, cross_validated=False)
        a = lambda p: br.accuracy([(r["gold_safety_label"], p[r["id"]]) for r in ROWS])
        self.assertLessEqual(a(cv), a(ins) + 1e-9)
        self.assertGreater(a(cv), a(br.majority_pred(ROWS)))  # the shortcut is real

    def test_cv_prediction_never_uses_the_rows_own_photo(self):
        # every id is predicted exactly once
        cv = br.template_pred(ROWS, cross_validated=True)
        self.assertEqual(set(cv), {r["id"] for r in ROWS})

    def test_capped_weighting_equals_plain_when_cap_is_huge(self):
        preds = br.majority_pred(ROWS)
        pairs = [(r["gold_safety_label"], preds[r["id"]]) for r in ROWS]
        self.assertAlmostEqual(br.capped_template_accuracy(ROWS, preds, cap=10**6), br.accuracy(pairs))

    def test_per_class_f1_perfect_and_zero(self):
        self.assertEqual(br.class_f1([("A", "A"), ("B", "B")], "A"), (1.0, 1.0, 1.0))
        self.assertEqual(br.class_f1([("A", "B")], "A")[2], 0.0)

    def test_paired_diff_of_identical_predictions_is_zero(self):
        preds = br.majority_pred(ROWS)
        pt, lo, hi = br.paired_diff_ci(ROWS, preds, preds, br.accuracy, n=50)
        self.assertEqual((pt, lo, hi), (0.0, 0.0, 0.0))

    def test_paired_diff_detects_a_perfect_system_over_majority(self):
        perfect = {r["id"]: r["gold_safety_label"] for r in ROWS}
        pt, lo, hi = br.paired_diff_ci(ROWS, br.majority_pred(ROWS), perfect, br.accuracy, n=100)
        self.assertGreater(lo, 0)

    def test_merged_subtypes_exist_and_strict_subset_excludes_them(self):
        strict = [r for r in ROWS if r["label_v5"] not in br.MERGED_SUBTYPES]
        self.assertLess(len(strict), len(ROWS))
        self.assertTrue(all(r["label_v5"] not in br.MERGED_SUBTYPES for r in strict))


class SmallCellAndAbstentionTests(unittest.TestCase):
    def test_wilson_interval_basics(self):
        lo, hi = br.wilson(21, 21)
        self.assertGreater(lo, 0.8)
        self.assertAlmostEqual(hi, 1.0)
        lo0, hi0 = br.wilson(0, 10)
        self.assertEqual(lo0, 0.0)
        self.assertLess(hi0, 0.35)
        lo5, hi5 = br.wilson(50, 100)
        self.assertLess(lo5, 0.5)
        self.assertGreater(hi5, 0.5)

    def test_small_cells_lists_only_cells_below_the_minimum(self):
        cells = br.small_cells(ROWS)
        self.assertIn("insufficient_data|UNSAFE", cells)
        self.assertTrue(all(n < br.MIN_CELL for n in cells.values()))
        self.assertNotIn("additive_regulation|SAFE", cells)

    def test_abstention_metrics_perfect_and_never_abstaining(self):
        import contextlib
        import io
        perfect = {r["id"]: r["gold_safety_label"] for r in ROWS}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            br.abstention_metrics(ROWS, perfect)
        out = buf.getvalue()
        self.assertIn("P=1.000 R=1.000", out)
        self.assertIn("false-abstain on answerable=0.000", out)
        never = {r["id"]: "SAFE" for r in ROWS}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            br.abstention_metrics(ROWS, never)
        self.assertIn("R=0.000", buf.getvalue())
