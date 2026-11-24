"""Bảng Phụ lục 2A/2B TT24 đã cấu trúc hoá phải khớp nguyên văn PDF đã lưu."""
import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "knowledge_base" / "sources" / "derived" / "tt24_annex2_ml.json"


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


class TestTT24Annex2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fitz

        cls.data = json.loads(DATA.read_text(encoding="utf-8"))
        pdf = ROOT / cls.data["source"]
        cls.sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
        doc = fitz.open(pdf)
        # bỏ dải đầu trang (số trang) để nội dung nối liền khi một ô kéo sang trang sau
        cls.pages = {i + 1: squash(doc[i].get_text(clip=fitz.Rect(0, 55, doc[i].rect.width, doc[i].rect.height)))
                     for i in range(45, 409)}

    def entries(self, annex=None):
        for g in self.data["groups"]:
            if annex is None or g["annex"] == annex:
                for e in g["entries"]:
                    yield g, e

    def test_source_hash(self):
        self.assertEqual(self.data["sha256"], self.sha)

    def test_no_parse_warnings(self):
        self.assertEqual(self.data["warnings"], [])

    def test_every_row_verbatim_on_its_pages(self):
        bad = []
        for g, e in self.entries():
            text = "".join(self.pages[p] for p in e.get("pages", [e["page"]]))
            row = squash(e["food_category_no"] + e["food_category"])
            if row not in text or squash(e["ml"]) not in text:
                bad.append((g["group_name"], e["food_category_no"], e["food_category"][:40], e["page"]))
        self.assertEqual(bad, [])

    def test_members_verbatim(self):
        for g in self.data["groups"]:
            for m in g["members"]:
                self.assertIn(squash(m["ins"] + m["name_vi"]), self.pages[m["page"]], (g["group_name"], m))

    def test_notes_verbatim_and_resolved(self):
        notes = self.data["notes"]
        for key, n in notes.items():
            text = "".join(self.pages[p] for p in n.get("pages", [n["page"]]))
            self.assertIn(squash(n["text"]), text, key)
        self.assertEqual(notes["52"]["text"], "Trừ sữa socola")
        self.assertIn("CO2", notes["60"]["text"])  # chỉ số dưới phải được giữ
        used = {k for _, e in self.entries() for k in e["note_keys"]}
        self.assertEqual(sorted(used - set(notes)), [])
        # bảng giải thích ghi chú không được dính vào hàng cuối của Phụ lục 2B
        self.assertFalse(any("GIẢI THÍCH" in e["food_category"] for _, e in self.entries()))

    def test_known_rows(self):
        def find(annex, ins, cat=None, name=None):
            for g, e in self.entries(annex):
                if any(m["ins"] == ins for m in g["members"]):
                    if (cat and e["food_category_no"] == cat) or (name and e["food_category"] == name):
                        return e
            return None

        e = find("2A", "100(i)", cat="01.1.4")
        self.assertEqual((e["ml"], e["notes"], e["page"]), ("150", "52, 402", 46))
        self.assertEqual(find("2A", "100(i)", cat="12.5")["ml"], "50")
        self.assertEqual(find("2A", "220", cat="11.1.3")["ml"], "20")  # khác Codex (70): quy định riêng của VN
        self.assertEqual(find("2B", "100(i)", name="Kẹo cao su")["ml"], "300")
        self.assertEqual(find("2B", "100(i)", name="Kẹo cao su")["notes"], "554")

    def test_footnotes_on_2b_groups(self):
        foot = [g for g in self.data["groups"] if "footnote" in g]
        self.assertEqual(len(foot), 18)
        self.assertTrue(all(g["annex"] == "2B" and "17/2023/TT" in g["footnote"]["text"] for g in foot))

    def test_codex_agreement_reported(self):
        stats = self.data["codex_comparison"]["stats"]
        self.assertGreater(stats["same"] / stats["pairs"], 0.95)


if __name__ == "__main__":
    unittest.main()
