"""Kiểm thử các phần không gọi LLM của scripts phân tích (giám khảo faithfulness, phân tích ca)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_research"))

import analyze_cases as ac  # noqa: E402
import judge_faithfulness as jf  # noqa: E402


class JudgeHelpersTests(unittest.TestCase):
    def test_parse_judge_output_complete_only(self):
        raw = 'Kết quả: [{"i": 1, "label": "SUPPORTED", "quote": "a b c d e"}, {"i": 2, "label": "not_enough_info"}]'
        parsed = jf.parse_judge_output(raw, 2)
        self.assertEqual(parsed[1]["label"], "SUPPORTED")
        self.assertEqual(parsed[2]["label"], "NOT_ENOUGH_INFO")
        self.assertIsNone(jf.parse_judge_output(raw, 3))           # thiếu nhận định → không chấp nhận
        self.assertIsNone(jf.parse_judge_output("không có json", 1))

    def test_quote_must_exist_in_evidence(self):
        evidence = "Thông tư 24/2019/TT-BYT cho phép sử dụng Aspartame (INS 951) trong đồ uống."
        self.assertTrue(jf.quote_in_evidence("cho phép sử dụng Aspartame (INS 951)", evidence))
        self.assertFalse(jf.quote_in_evidence("cấm sử dụng Aspartame trong mọi thực phẩm", evidence))
        self.assertFalse(jf.quote_in_evidence("cho phép", evidence))   # quá ngắn

    def test_unverified_quote_downgraded(self):
        pred = {"id": "q", "config": "B", "answer": "Aspartame (INS 951) bị cấm hoàn toàn trong đồ uống ở Việt Nam.",
                "ground_truth_text": "Thành phần: nước, aspartame (INS 951)",
                "retrieved_passages": [{"text": "Thông tư 24 cho phép sử dụng aspartame trong đồ uống."}]}
        original = jf.pl._call_llm
        jf.pl._call_llm = lambda *a, **k: '[{"i": 1, "label": "CONTRADICTED", "quote": "đoạn này không hề có trong bằng chứng"}]'
        try:
            res = jf.judge_one(pred, {"product_ingredients": ""})
        finally:
            jf.pl._call_llm = original
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["claims"][0]["judge_label"], "CONTRADICTED")
        self.assertEqual(res["claims"][0]["label"], "NOT_ENOUGH_INFO")   # trích dẫn bịa → hạ nhãn
        self.assertFalse(res["claims"][0]["quote_verified"])

    def test_cohen_kappa(self):
        self.assertAlmostEqual(jf.cohen_kappa([("S", "S"), ("N", "N"), ("C", "C")]), 1.0)
        self.assertLess(jf.cohen_kappa([("S", "N"), ("N", "S"), ("S", "N"), ("N", "S")]), 0)
        self.assertIsNone(jf.cohen_kappa([]))


class CaseHelpersTests(unittest.TestCase):
    def test_codes_in_requires_prefix(self):
        self.assertEqual(ac.codes_in("dùng INS 951 và E621, 7,4 g/l"), {"951", "621"})

    def test_codes_on_label_include_names(self):
        codes = ac.codes_on_label("Thành phần: chất tạo ngọt (950, 955), aspartame")
        self.assertTrue({"950", "955"} <= codes)


if __name__ == "__main__":
    unittest.main()
