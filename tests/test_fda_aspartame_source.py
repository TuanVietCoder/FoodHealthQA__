"""Trang FDA 'Aspartame and Other Sweeteners in Food' (nguồn cho câu 'người PKU nên tránh hoặc hạn chế aspartame')
phải được đăng ký đủ: manifest (sha256 khớp file), danh mục trích dẫn, mã nguồn được công nhận, file cục bộ,
và nhận định đã kiểm có câu nguyên văn trong file."""
import hashlib
import html
import json
import re
import unittest
from pathlib import Path

from modules import evidence
from modules.benchmark import is_recognized_citation

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "knowledge_base" / "sources" / "raw" / "external" / "FDA_SWEETENERS_IN_FOOD.html"
CAT = ROOT / "knowledge_base" / "sources" / "catalog"
CLAIMS = ROOT / "knowledge_base" / "sources" / "derived" / "external_claim_verification.json"
QUOTE = ("people with difficulty metabolizing phenylalanine because of a rare genetic disorder called "
         "phenylketonuria (PKU) should avoid or restrict aspartame")


def page_text() -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", "", FILE.read_text(encoding="utf-8", errors="ignore"))
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))


class TestFdaAspartameSource(unittest.TestCase):
    def test_quote_in_saved_page(self):
        self.assertIn(QUOTE, page_text())

    def test_manifest_hash(self):
        manifest = json.loads((CAT / "external_sources_manifest.json").read_text(encoding="utf-8"))
        entry = next(x for x in manifest["manually_downloaded"] if x["file"] == FILE.name)
        self.assertEqual(entry["sha256"], hashlib.sha256(FILE.read_bytes()).hexdigest())

    def test_registered_everywhere(self):
        catalog = json.loads((CAT / "source_citation_catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(catalog["US_FDA_ASPARTAME_SWEETENERS#PKU_AvoidOrRestrict"]["citation_id"], "US_FDA_ASPARTAME_SWEETENERS")
        self.assertTrue(is_recognized_citation("US_FDA_ASPARTAME_SWEETENERS"))
        self.assertTrue(evidence.local_source_file("US_FDA_ASPARTAME_SWEETENERS", "").endswith(FILE.name))

    def test_claim_verified(self):
        claim = json.loads(CLAIMS.read_text(encoding="utf-8"))["FDA_ASPARTAME_PKU_AVOID_OR_RESTRICT"]
        self.assertTrue(claim["passed"])
        self.assertEqual(claim["sha256"], hashlib.sha256(FILE.read_bytes()).hexdigest())
        self.assertIn("should avoid or restrict aspartame", claim["evidence"]["pku"])


if __name__ == "__main__":
    unittest.main()
