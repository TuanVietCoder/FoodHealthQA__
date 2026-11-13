import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.benchmark import (  # noqa: E402
    analyze_answer_citations,
    evaluate_predictions,
    valid_predicted_sources,
)
from modules.evaluation import aggregate_metrics, evaluate_config, evaluate_single  # noqa: E402


def ref(rid="Q1", sources="VN_TT24_2019, JECFA_EVALUATIONS", label="SAFE"):
    return {"id": rid, "reference_answer": "đáp án chuẩn", "source_citations": sources,
            "gold_safety_label": label, "domain": "additive_regulation"}


def pred(rid="Q1", answer="", label="SAFE"):
    return {"id": rid, "answer": answer, "safety_label": label}


class AnalyzeCitationTests(unittest.TestCase):
    def test_splits_recognized_and_unrecognized_without_duplicates(self):
        out = analyze_answer_citations(
            "Được phép [VN_TT24_2019]. Không rõ [khong tim thay]. Lặp lại [VN_TT24_2019] [JECFA_EVALUATIONS] [khong tim thay]."
        )
        self.assertEqual(out["recognized"], ["VN_TT24_2019", "JECFA_EVALUATIONS"])
        self.assertEqual(out["unrecognized"], ["khong tim thay"])
        self.assertTrue(out["has_citation"])

    def test_only_unrecognized_means_not_cited(self):
        out = analyze_answer_citations("Không có nguồn [khong tim thay].")
        self.assertFalse(out["has_citation"])
        self.assertEqual(analyze_answer_citations("Không có thẻ nào.")["recognized"], [])

    def test_stray_tag_that_matches_gold_is_kept(self):
        analysis = {"recognized": ["VN_TT24_2019"], "unrecognized": ["us fda benzene qa", "khong tim thay"]}
        valid, stray = valid_predicted_sources(analysis, {"us fda benzene qa"})
        self.assertEqual(valid, {"VN_TT24_2019", "us fda benzene qa"})
        self.assertEqual(stray, ["khong tim thay"])


class MetricTests(unittest.TestCase):
    def test_stray_tag_does_not_lower_precision_but_is_reported(self):
        clean = evaluate_single(pred(answer="OK [VN_TT24_2019]"), ref())
        noisy = evaluate_single(pred(answer="OK [VN_TT24_2019] [khong tim thay]"), ref())
        self.assertEqual(clean["citation_precision"], 1.0)
        self.assertEqual(noisy["citation_precision"], 1.0)
        self.assertEqual(noisy["unrecognized_citations"], ["khong tim thay"])
        self.assertEqual(clean["unrecognized_citations"], [])

    def test_wrong_recognized_source_still_lowers_precision(self):
        r = evaluate_single(pred(answer="OK [VN_TT24_2019] [EFSA_OPINIONS]"), ref())
        self.assertEqual(r["citation_precision"], 0.5)

    def test_aggregate_compliance_and_fake_tag_rate(self):
        items = [
            evaluate_single(pred("Q1", "đúng [VN_TT24_2019]"), ref("Q1")),
            evaluate_single(pred("Q2", "không nguồn"), ref("Q2")),
            evaluate_single(pred("Q3", "bịa [khong tim thay]"), ref("Q3")),
            evaluate_single(pred("Q4", "[VN_TT24_2019] [x y z] [a b]"), ref("Q4")),
        ]
        m = aggregate_metrics(items)
        self.assertEqual(m["citation_compliance"], 0.5)          # Q1, Q4 có thẻ được công nhận
        self.assertEqual(m["unrecognized_citation_rate"], 0.5)   # Q3, Q4 có thẻ lạ
        self.assertEqual(m["unrecognized_tags_per_answer"], 0.75)  # (0+0+1+2)/4

    def test_evaluate_config_exposes_new_metrics(self):
        result = evaluate_config([ref("Q1"), ref("Q2")],
                                 [pred("Q1", "đúng [VN_TT24_2019]"), pred("Q2", "không nguồn")], "B")
        self.assertEqual(result["overall"]["citation_compliance"], 0.5)
        self.assertEqual(result["overall"]["unrecognized_citation_rate"], 0.0)

    def test_benchmark_evaluate_predictions_has_same_semantics(self):
        refs = [ref("Q1"), ref("Q2")]
        preds = [pred("Q1", "đúng [VN_TT24_2019] [khong tim thay]"), pred("Q2", "không nguồn")]
        m = evaluate_predictions(refs, preds)
        group = m["retrieval_and_answer"]
        self.assertEqual(group["citation_compliance"], 0.5)
        self.assertEqual(group["unrecognized_citation_rate"], 0.5)
        self.assertEqual(group["citation_precision"], 0.5)  # Q1: 1.0 (thẻ lạ bị loại), Q2: 0.0
        self.assertEqual(m["citation_compliance"], 0.5)  # bản tổng phẳng


if __name__ == "__main__":
    unittest.main()
