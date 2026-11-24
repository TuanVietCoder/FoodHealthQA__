"""Mã INS ghi trên nhãn chỉ được coi là khớp một mục KB khi đúng mã (hoặc mã gốc/mã con), không phải khi chỉ chứa chung vài chữ số."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import hybrid_retriever as hr  # noqa: E402
from modules import pipeline  # noqa: E402


def _match(query, entity_code):
    return any(hr._code_matches_entity(c, entity_code.lower()) for c in hr._explicit_codes(query.lower()))


class ExplicitCodeMatchTests(unittest.TestCase):
    def test_same_code_parent_and_child_match(self):
        for query, entity in (("ins 322", "INS 322(i)"), ("INS 160a", "INS 160a(i)"), ("(211)", "INS 211"), ("e330", "INS 330"),
                              ("INS 450(i)", "INS 450(i)")):
            self.assertTrue(_match(query, entity), (query, entity))

    def test_a_different_code_sharing_digits_does_not_match(self):
        # "160(i)" không tồn tại; "160a(ii)" là carotene. 407 (carrageenan) khác 407a. 450(iii) khác 450(i).
        for query, entity in (("INS 160(i)", "INS 160a(ii)"), ("INS 407", "INS 407a"), ("INS 450(iii)", "INS 450(i)")):
            self.assertFalse(_match(query, entity), (query, entity))

    def test_a_unit_letter_after_a_number_is_not_a_code_suffix(self):
        self.assertEqual(hr._explicit_codes("100g"), [])
        self.assertEqual(hr._explicit_codes("ins 100a"), ["100a"])


class MisreadCodeStaysUnlinkedTests(unittest.TestCase):
    def test_a_code_that_does_not_exist_is_not_silently_turned_into_a_carotene(self):
        kb = pipeline.load_knowledge_base(str(Path(__file__).resolve().parents[1] / "data"))
        linked = pipeline.link_all_mentions([{"raw_mention": "màu thực phẩm tổng hợp INS 160(i)", "category": "additive"}], kb)
        self.assertTrue(all(item["matched_entity"] is None for item in linked))
        self.assertIn("unlinked", [item["match_method"] for item in linked])
        self.assertIn("uncertain_code", [item["match_method"] for item in linked])  # mã lạ vẫn được ghi nhận để S_retrieval giảm đúng


if __name__ == "__main__":
    unittest.main()
