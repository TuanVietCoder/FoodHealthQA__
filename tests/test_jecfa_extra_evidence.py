"""Nhận định JECFA dùng trong đáp án nhiều hơn con số ADI phải có câu nguyên văn tương ứng từ trang JECFA đã lưu
(câu "không có bằng chứng thuyết phục … tác động bất lợi" và "giữ nguyên ADI" của JECFA 2023)."""
import hashlib
import html
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "knowledge_base" / "sources" / "raw" / "external"
CLAIMS = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "external_claim_verification.json").read_text(encoding="utf-8"))


def page_text(rel: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", "", (EXT / rel).read_text(encoding="utf-8", errors="ignore"))
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))


class TestJecfaExtraEvidence(unittest.TestCase):
    def test_aspartame_statements_verbatim(self):
        c = CLAIMS["JECFA_ADI_951"]
        self.assertTrue(c["passed"])
        self.assertEqual(c["sha256"], hashlib.sha256((EXT / c["source_file"]).read_bytes()).hexdigest())
        text = page_text(c["source_file"])
        for key in ("no_adverse", "reaffirmed"):
            self.assertIn(c["evidence"][key].strip(), text, key)
        self.assertIn("no convincing evidence", c["evidence"]["no_adverse"])
        self.assertIn("reaffirmed the ADI of 0–40 mg/kg bw", c["evidence"]["reaffirmed"])


if __name__ == "__main__":
    unittest.main()
