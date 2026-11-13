"""Tests for modules/evaluation.py — the canonical evaluation pipeline."""

import unittest

from modules.evaluation import (
    compute_cer,
    compute_wer,
    compute_confusion_matrix,
    format_confusion_matrix,
    evaluate_single,
    aggregate_metrics,
    evaluate_config,
    format_results_table,
    _heuristic_groundedness,
    _edit_distance,
)


class TestEditDistance(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(_edit_distance(list("abc"), list("abc")), 0)

    def test_insertion(self):
        self.assertEqual(_edit_distance(list("abc"), list("abcd")), 1)

    def test_deletion(self):
        self.assertEqual(_edit_distance(list("abcd"), list("abc")), 1)

    def test_substitution(self):
        self.assertEqual(_edit_distance(list("abc"), list("axc")), 1)

    def test_empty(self):
        self.assertEqual(_edit_distance([], list("abc")), 3)
        self.assertEqual(_edit_distance(list("abc"), []), 3)
        self.assertEqual(_edit_distance([], []), 0)


class TestCERWER(unittest.TestCase):
    def test_cer_identical(self):
        self.assertAlmostEqual(compute_cer("hello", "hello"), 0.0)

    def test_cer_one_char_diff(self):
        self.assertAlmostEqual(compute_cer("hello", "hallo"), 1 / 5)

    def test_wer_identical(self):
        self.assertAlmostEqual(compute_wer("hello world", "hello world"), 0.0)

    def test_wer_one_word_diff(self):
        self.assertAlmostEqual(compute_wer("hello world", "hello earth"), 1 / 2)

    def test_cer_empty_reference(self):
        self.assertAlmostEqual(compute_cer("", "abc"), 3.0)

    def test_wer_empty_reference(self):
        self.assertAlmostEqual(compute_wer("", "hello world"), 2.0)


class TestConfusionMatrix(unittest.TestCase):
    def test_perfect_predictions(self):
        gold = ["SAFE", "WARNING", "CAUTION"]
        pred = ["SAFE", "WARNING", "CAUTION"]
        cm = compute_confusion_matrix(gold, pred)
        self.assertEqual(cm["SAFE"]["SAFE"], 1)
        self.assertEqual(cm["WARNING"]["WARNING"], 1)
        self.assertEqual(cm["CAUTION"]["CAUTION"], 1)
        self.assertEqual(cm["SAFE"]["WARNING"], 0)

    def test_misclassification(self):
        gold = ["WARNING", "WARNING"]
        pred = ["SAFE", "WARNING"]
        cm = compute_confusion_matrix(gold, pred)
        self.assertEqual(cm["WARNING"]["WARNING"], 1)
        self.assertEqual(cm["WARNING"]["SAFE"], 1)

    def test_format_has_all_labels(self):
        cm = compute_confusion_matrix(["SAFE"], ["SAFE"])
        text = format_confusion_matrix(cm)
        for label in ["SAFE", "CAUTION", "WARNING", "CRITICAL_ALERT", "ABSTAIN"]:
            self.assertIn(label, text)


class TestHeuristicGroundedness(unittest.TestCase):
    def test_full_overlap(self):
        self.assertAlmostEqual(
            _heuristic_groundedness("hello world", "hello world"), 1.0
        )

    def test_partial_overlap(self):
        score = _heuristic_groundedness("hello world foo", "hello world bar")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_no_overlap(self):
        self.assertAlmostEqual(
            _heuristic_groundedness("abc", "xyz"), 0.0
        )

    def test_empty(self):
        self.assertAlmostEqual(_heuristic_groundedness("", "evidence"), 0.0)
        self.assertAlmostEqual(_heuristic_groundedness("answer", ""), 0.0)


class TestEvaluateSingle(unittest.TestCase):
    def _make_ref(self, **overrides):
        base = {
            "id": "q1",
            "question": "test?",
            "reference_answer": "Monosodium glutamate",
            "source_ids": ["codex stan 192 1995"],
            "domain": "additive",
            "gold_safety_label": "SAFE",
        }
        base.update(overrides)
        return base

    def _make_pred(self, **overrides):
        base = {
            "id": "q1",
            "answer": "Monosodium glutamate [Codex STAN 192-1995]",
            "safety_label": "SAFE",
        }
        base.update(overrides)
        return base

    def test_exact_match_with_citation(self):
        ref = self._make_ref()
        pred = self._make_pred()
        result = evaluate_single(pred, ref)
        self.assertFalse(result["exact_match"])
        self.assertGreaterEqual(result["token_f1"], 0.5)

    def test_perfect_match(self):
        ref = self._make_ref()
        pred = self._make_pred(answer="Monosodium glutamate")
        result = evaluate_single(pred, ref)
        self.assertTrue(result["exact_match"])
        self.assertAlmostEqual(result["token_f1"], 1.0)

    def test_safety_correct(self):
        ref = self._make_ref(gold_safety_label="WARNING")
        pred = self._make_pred(safety_label="WARNING")
        result = evaluate_single(pred, ref)
        self.assertTrue(result["safety_correct"])

    def test_safety_incorrect(self):
        ref = self._make_ref(gold_safety_label="WARNING")
        pred = self._make_pred(safety_label="SAFE")
        result = evaluate_single(pred, ref)
        self.assertFalse(result["safety_correct"])

    def test_groundedness_human(self):
        ref = self._make_ref()
        pred = self._make_pred(grounded=True)
        result = evaluate_single(pred, ref)
        self.assertTrue(result["groundedness"])

    def test_groundedness_heuristic(self):
        ref = self._make_ref(evidence_excerpt="Monosodium glutamate is a flavor enhancer")
        pred = self._make_pred()
        result = evaluate_single(pred, ref)
        self.assertIsInstance(result["groundedness"], float)
        self.assertGreater(result["groundedness"], 0.0)


class TestAggregateMetrics(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(aggregate_metrics([])["count"], 0)

    def test_single_item(self):
        items = [{
            "exact_match": True,
            "token_f1": 0.8,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "safety_correct": True,
            "gold_safety_label": "WARNING",
            "predicted_safety_label": "WARNING",
            "groundedness": 0.9,
            "faithfulness": True,
        }]
        result = aggregate_metrics(items)
        self.assertEqual(result["count"], 1)
        self.assertAlmostEqual(result["exact_match"], 1.0)
        self.assertAlmostEqual(result["token_f1"], 0.8)
        self.assertAlmostEqual(result["safety_accuracy"], 1.0)
        self.assertAlmostEqual(result["unsafe_recall"], 1.0)
        self.assertIn("safety_score", result)

    def test_abstention_rate(self):
        items = [
            {
                "exact_match": True, "token_f1": 1.0,
                "citation_precision": 1.0, "citation_recall": 1.0,
                "safety_correct": None, "gold_safety_label": "ABSTAIN",
                "predicted_safety_label": "ABSTAIN",
                "groundedness": None, "faithfulness": None,
            },
            {
                "exact_match": True, "token_f1": 1.0,
                "citation_precision": 1.0, "citation_recall": 1.0,
                "safety_correct": True, "gold_safety_label": "SAFE",
                "predicted_safety_label": "SAFE",
                "groundedness": None, "faithfulness": None,
            },
        ]
        result = aggregate_metrics(items)
        self.assertAlmostEqual(result["abstention_rate_predicted"], 0.5)
        self.assertAlmostEqual(result["abstention_rate_gold"], 0.5)


class TestEvaluateConfig(unittest.TestCase):
    def _make_data(self):
        refs = [
            {
                "id": "q1", "question": "?",
                "reference_answer": "Answer one",
                "source_ids": ["src1"], "domain": "d1",
                "gold_safety_label": "SAFE",
            },
            {
                "id": "q2", "question": "?",
                "reference_answer": "Answer two",
                "source_ids": ["src2"], "domain": "d2",
                "gold_safety_label": "WARNING",
            },
        ]
        preds = [
            {"id": "q1", "answer": "Answer one", "safety_label": "SAFE"},
            {"id": "q2", "answer": "Wrong answer", "safety_label": "WARNING"},
        ]
        return refs, preds

    def test_structure(self):
        refs, preds = self._make_data()
        result = evaluate_config(refs, preds, "test_config")
        self.assertEqual(result["config"], "test_config")
        self.assertIn("overall", result)
        self.assertIn("per_domain", result)
        self.assertIn("confusion_matrix", result)
        self.assertIn("item_results", result)
        self.assertEqual(len(result["item_results"]), 2)

    def test_missing_predictions(self):
        refs, preds = self._make_data()
        preds_partial = [preds[0]]
        result = evaluate_config(refs, preds_partial, "partial")
        self.assertEqual(result["missing_predictions"], ["q2"])

    def test_per_domain(self):
        refs, preds = self._make_data()
        result = evaluate_config(refs, preds, "test")
        self.assertIn("d1", result["per_domain"])
        self.assertIn("d2", result["per_domain"])
        self.assertEqual(result["per_domain"]["d1"]["count"], 1)


class TestFormatResultsTable(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(format_results_table([]), "No results.")

    def test_single_config(self):
        refs = [{
            "id": "q1", "question": "?",
            "reference_answer": "Answer",
            "source_ids": ["s"], "domain": "d",
            "gold_safety_label": "SAFE",
        }]
        preds = [{"id": "q1", "answer": "Answer", "safety_label": "SAFE"}]
        result = evaluate_config(refs, preds, "A")
        table = format_results_table([result])
        self.assertIn("EM", table)
        self.assertIn("Token F1", table)
        self.assertIn("A", table)


class TestScriptsResearchReExport(unittest.TestCase):
    """Verify scripts_research/evaluation.py re-exports work correctly."""

    def test_import_from_wrapper(self):
        from scripts_research.evaluation import (
            compute_cer,
            compute_wer,
            evaluate_config,
            format_results_table,
            format_confusion_matrix,
        )
        self.assertAlmostEqual(compute_cer("abc", "abc"), 0.0)
        self.assertAlmostEqual(compute_wer("a b", "a b"), 0.0)
        self.assertTrue(callable(evaluate_config))
        self.assertTrue(callable(format_results_table))
        self.assertTrue(callable(format_confusion_matrix))


if __name__ == "__main__":
    unittest.main()
