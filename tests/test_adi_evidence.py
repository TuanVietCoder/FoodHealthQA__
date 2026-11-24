"""Mọi giá trị ADI trong KB phải có câu nguyên văn trong file nguồn đã lưu (kèm trang), và câu đó phải chứa đúng giá trị."""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_legacy_health_review import _page_text, _squash  # noqa: E402

KB = json.loads((ROOT / "knowledge_base" / "additives_standard.json").read_text(encoding="utf-8"))
BY = {x["ins_code"]: x["adi_assessment"] for x in KB}


class AdiEvidenceTests(unittest.TestCase):
    def test_every_adi_has_verbatim_evidence_on_the_cited_page(self):
        missing, checked = [], 0
        for code, adi in BY.items():
            if not adi.get("evidence"):
                missing.append(code)
                continue
            for e in adi["evidence"]:
                self.assertIn(_squash(e["quote"]), _page_text(e["source_file"], e["page"]), (code, e["source_file"], e["page"]))
                checked += 1
        self.assertEqual(missing, [])
        self.assertGreaterEqual(checked, 88)

    def test_numeric_adi_appears_in_one_of_its_quotes(self):
        for code, adi in BY.items():
            upper = adi.get("numeric_upper_mg_per_kg_bw_day")
            if upper is None:
                continue
            num = f"{float(upper):g}"
            pattern = r"(?<![\d.,])" + re.escape(num).replace(r"\.", r"[.,]") + r"(?!\d)"
            self.assertTrue(any(re.search(pattern, e["quote"]) for e in adi["evidence"]), (code, num))

    def test_not_limited_is_not_reported_as_not_specified(self):
        # JECFA ghi "NOT LIMITED" (thuật ngữ cũ) cho các mục này; KB từng ghi nhầm thành "Not Specified"
        for code in ("INS 322(i)", "INS 330", "INS 500(ii)", "INS 262(i)", "INS 471"):
            self.assertIn("Not Limited", BY[code]["display_text"], code)
            self.assertIn("Not Specified", BY[code]["display_text_before_review"], code)
        self.assertTrue(BY["INS 471"]["source_url"].endswith("/Home/Chemical/917"))
        self.assertIn("Not Limited", BY["INS 260"]["display_text"])  # ADI nhóm của acid acetic, không chỉ câu về chất tạo hương


if __name__ == "__main__":
    unittest.main()
