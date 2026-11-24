"""Đánh giá dị nguyên của mọi phụ gia trong KB: một phương pháp, trích dẫn nguyên văn danh mục, không gán nhóm cứng khi chưa biết nguồn."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_legacy_health_review import _page_text, _squash  # noqa: E402

KB = json.loads((ROOT / "knowledge_base" / "additives_standard.json").read_text(encoding="utf-8"))
BY = {x["ins_code"]: x["allergen_assessment"] for x in KB}
IDENTITY = "assessed_by_identity_against_allergen_lists"


class AllergenAssessmentTests(unittest.TestCase):
    def test_every_additive_is_classified_and_none_is_left_unassessed(self):
        classes = {"listed_substance", "not_from_listed_allergen", "source_dependent", "not_determinable"}
        for code, a in BY.items():
            self.assertIn(a.get("classification"), classes, code)
            self.assertNotEqual(a["verification_status"], "not_assessed", code)

    def test_only_substances_named_in_the_lists_are_source_checked_and_carry_a_group(self):
        for code, a in BY.items():
            if a["classification"] == "listed_substance":
                self.assertEqual((a["verification_status"], a["groups"]), ("source_checked", ["sulfit"]), code)
            else:
                self.assertEqual(a["verification_status"], IDENTITY, code)
                self.assertEqual(a["groups"], [], code)  # không gán nhóm cứng khi nhãn chưa ghi nguồn
        self.assertEqual(sorted(c for c, a in BY.items() if a["classification"] == "listed_substance"), ["INS 220", "INS 221"])

    def test_source_dependent_entries_name_the_possible_groups(self):
        for code, a in BY.items():
            if a["classification"] == "source_dependent":
                self.assertTrue(a["possible_groups_if_source_is_allergenic"], code)
            else:
                self.assertEqual(a["possible_groups_if_source_is_allergenic"], [], code)

    def test_key_decisions(self):
        for code in ("INS 1400", "INS 1412", "INS 1414", "INS 1420", "INS 1422", "INS 1440", "INS 1442"):
            self.assertEqual(BY[code]["possible_groups_if_source_is_allergenic"], ["gluten (lúa mì)"], code)  # tinh bột lúa mì không được miễn
        for code in ("INS 471", "INS 470(i)", "INS 477"):
            self.assertEqual(BY[code]["possible_groups_if_source_is_allergenic"], ["mù tạt"], code)  # Phụ lục II nêu đích danh E 470a/E 471/E 477
        for code in ("INS 422", "INS 475", "INS 491", "INS 492"):
            self.assertEqual(BY[code]["classification"], "not_from_listed_allergen", code)  # dẫn xuất dầu đậu nành tinh luyện: chú thích (1)
        self.assertEqual(sorted(BY["INS 322(i)"]["possible_groups_if_source_is_allergenic"]), ["trứng", "đậu nành"])
        self.assertEqual(BY["INS 635"]["classification"], "not_determinable")

    def test_old_hand_written_notes_are_kept_for_traceability(self):
        kept = [c for c, a in BY.items() if a.get("note_before_review")]
        self.assertGreaterEqual(len(kept), 50)
        self.assertIn("EFSA GRAS", BY["INS 415"]["note_before_review"])  # nhận định cũ không có nguồn đã bị thay
        self.assertNotIn("GRAS", BY["INS 415"]["note"])

    def test_every_list_quote_is_verbatim_on_its_page(self):
        checked = 0
        for code, a in BY.items():
            self.assertTrue(a.get("evidence"), code)
            for e in a["evidence"]:
                self.assertIn(_squash(e["quote"]), _page_text(e["source_file"], e["page"]), (code, e["page"]))
                checked += 1
        self.assertGreater(checked, 150)


if __name__ == "__main__":
    unittest.main()
