import unittest

from modules.ablation import run_retrieval_ablation
from modules.benchmark import evaluate_predictions, validate_citations


class BenchmarkMetricsTests(unittest.TestCase):
    def test_metrics_score_exact_text_citations_and_safety(self):
        references = [{
            "id": "case-1",
            "question": "What is INS 621?",
            "reference_answer": "Monosodium glutamate",
            "source_ids": ["Codex STAN 192-1995"],
            "domain": "additive_identity",
            "safety_label": "WARNING",
        }]
        predictions = [{
            "id": "case-1",
            "answer": "Monosodium glutamate [Codex STAN 192-1995]",
            "safety_label": "WARNING",
            "grounded": True,
            "faithful": True,
        }]

        result = evaluate_predictions(references, predictions)

        self.assertEqual(result["exact_match"], 0)
        self.assertGreaterEqual(result["token_f1"], 0.5)
        self.assertEqual(result["citation_accuracy"], 1)
        self.assertEqual(result["citation_coverage"], 1)
        self.assertEqual(result["safety_accuracy"], 1)
        self.assertEqual(result["unsafe_recall"], 1)
        self.assertEqual(result["groundedness_human_adjudicated"], 1)
        self.assertEqual(result["faithfulness_human_adjudicated"], 1)

    def test_unknown_or_missing_citations_are_rejected(self):
        self.assertFalse(
            validate_citations("Answer [invented source]", ["Codex"])["valid"]
        )
        self.assertTrue(
            validate_citations("Answer [Codex]", ["Codex"])["valid"]
        )
        self.assertTrue(
            validate_citations("Answer", ["Codex"])["missing_citations"]
        )
        self.assertFalse(
            validate_citations(
                "First sentence [Codex]. Second unsupported sentence.",
                ["Codex"],
                require_each_sentence=True,
            )["valid"]
        )

    def test_retrieval_ablation_returns_metrics_for_each_top_k(self):
        additives = [{
            "ins_code": "INS 621",
            "e_number": "E621",
            "canonical_name": "Monosodium glutamate",
            "canonical_name_en": "Monosodium glutamate",
            "function": "Flavor enhancer",
            "common_names": ["MSG", "glutamate"],
        }]
        records = [{"question": "INS 621 monosodium glutamate", "expected_ins_code": "INS 621"}]

        results = run_retrieval_ablation(records, additives, top_k_values=(1, 3))

        self.assertEqual(len(results), 6)
        self.assertTrue(all(row["hit_rate"] == 1 for row in results))
        self.assertEqual({row["top_k"] for row in results}, {1, 3})

    def test_retrieval_ablation_rejects_unlabeled_records(self):
        with self.assertRaises(ValueError):
            run_retrieval_ablation([{"question": "MSG"}], [{"ins_code": "INS 621"}])


if __name__ == "__main__":
    unittest.main()
