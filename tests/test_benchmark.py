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

    def test_recognized_citation_sources_falcpa_faster_act_jemra(self):
        from modules.benchmark import (
            RECOGNIZED_CITATION_SOURCES,
            is_recognized_citation,
            canonicalize_citation,
        )
        # 1. Ensure identifiers are in RECOGNIZED_CITATION_SOURCES
        self.assertIn("US_FDA_FALCPA", RECOGNIZED_CITATION_SOURCES)
        self.assertIn("US_FDA_FASTER_ACT_2023", RECOGNIZED_CITATION_SOURCES)
        self.assertIn("FAO_WHO_JEMRA_PART_3", RECOGNIZED_CITATION_SOURCES)

        # 2. Test is_recognized_citation
        self.assertTrue(is_recognized_citation("US_FDA_FALCPA"))
        self.assertTrue(is_recognized_citation("US FDA FALCPA"))
        self.assertTrue(is_recognized_citation("US_FDA_FASTER_ACT_2023"))
        self.assertTrue(is_recognized_citation("FASTER Act 2023"))
        self.assertTrue(is_recognized_citation("FAO_WHO_JEMRA_PART_3"))
        self.assertTrue(is_recognized_citation("FAO/WHO JEMRA Part 3"))

        # 3. Test canonicalize_citation
        self.assertEqual(canonicalize_citation("US FDA FALCPA"), "US_FDA_FALCPA")
        self.assertEqual(canonicalize_citation("FASTER Act 2023"), "US_FDA_FASTER_ACT_2023")
        self.assertEqual(canonicalize_citation("FAO/WHO JEMRA Part 3 (PAL, 2023)"), "FAO_WHO_JEMRA_PART_3")

        # 4. Test evaluate_predictions citation_accuracy
        references = [
            {
                "id": "c-falcpa",
                "question": "Dị ứng đậu nành?",
                "reference_answer": "Sản phẩm chứa đậu nành.",
                "source_ids": ["US_FDA_FALCPA"],
                "domain": "allergen",
                "safety_label": "UNSAFE",
            },
            {
                "id": "c-faster",
                "question": "Dị ứng mè?",
                "reference_answer": "Mè là dị nguyên thứ 9.",
                "source_ids": ["US_FDA_FASTER_ACT_2023"],
                "domain": "allergen",
                "safety_label": "UNSAFE",
            },
            {
                "id": "c-jemra3",
                "question": "Cảnh báo tiếp xúc chéo?",
                "reference_answer": "Có thể chứa vết mè.",
                "source_ids": ["FAO_WHO_JEMRA_PART_3"],
                "domain": "allergen",
                "safety_label": "WARNING",
            },
        ]
        predictions = [
            {
                "id": "c-falcpa",
                "answer": "Sản phẩm chứa đậu nành [US FDA FALCPA].",
                "safety_label": "UNSAFE",
            },
            {
                "id": "c-faster",
                "answer": "Mè là dị nguyên lớn theo đạo luật [FASTER Act 2023].",
                "safety_label": "UNSAFE",
            },
            {
                "id": "c-jemra3",
                "answer": "Có nguy cơ nhiễm chéo [FAO/WHO JEMRA Part 3].",
                "safety_label": "WARNING",
            },
        ]
        metrics = evaluate_predictions(references, predictions)
        self.assertEqual(metrics["citation_accuracy"], 1.0)
        self.assertEqual(metrics["citation_coverage"], 1.0)

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

    def test_embedding_and_topk_ablation_methods(self):
        from modules.ablation import run_embedding_ablation, run_topk_ablation
        additives = [{
            "ins_code": "INS 621",
            "e_number": "E621",
            "canonical_name": "Monosodium glutamate",
            "canonical_name_en": "Monosodium glutamate",
            "function": "Flavor enhancer",
            "common_names": ["MSG", "glutamate"],
        }]
        records = [{"question": "INS 621", "expected_ins_code": "INS 621"}]

        emb_results = run_embedding_ablation(records, additives, top_k=3)
        self.assertEqual(len(emb_results), 3)
        self.assertTrue(all(row["hit_rate"] == 1.0 for row in emb_results))

        topk_results = run_topk_ablation(records, additives, top_k_values=(1, 3, 5))
        self.assertEqual(len(topk_results), 3)
        self.assertEqual([r["top_k"] for r in topk_results], [1, 3, 5])

    def test_knowledge_source_and_error_propagation_study(self):
        from modules.ablation import run_knowledge_source_ablation, run_error_propagation_study
        additives = [{
            "ins_code": "INS 322",
            "canonical_name": "Lecithin",
            "synonyms": ["INS 322i"],
        }]
        persona = {
            "id": "P_ALLERGY_01",
            "conditions": [],
            "allergies": ["soy"],
        }
        personas_dict = {"P_ALLERGY_01": persona}
        row = {
            "id": "FOODQA_TEST_01",
            "persona_id": "P_ALLERGY_01",
            "product_name": "Test Snack",
            "product_ingredients": "Bột mì, đậu nành, dầu ăn",
            "gold_safety_label": "UNSAFE",
            "question": "Ăn được không?",
            "domain": "allergen_safety",
        }

        # Knowledge Source Ablation: Leave-One-Out
        ks_res = run_knowledge_source_ablation(
            [row], additives, personas_dict, hazards_list=[], fndds_list=[],
            off_products=[], excluded_sources=["additives_standard"],
        )
        self.assertEqual(ks_res["total_evaluated"], 1)

        # Error Propagation Study: Config B vs Config C
        ep_res = run_error_propagation_study(
            [row], additives, personas_dict, hazards_list=[], fndds_list=[],
            off_products=[], noise_level=0.1,
        )
        self.assertIn("delta_safety_acc", ep_res)
        self.assertIn("false_safe_rate_in_config_c", ep_res)


if __name__ == "__main__":
    unittest.main()
