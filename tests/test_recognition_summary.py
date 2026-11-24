"""Khung "máy đọc được gì trên nhãn" và tính đúng của các liên kết trang JECFA (id trang phải thuộc đúng phụ gia)."""
import glob
import html
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import claim_locator_resolver, evidence, pipeline  # noqa: E402

JECFA_DIR = ROOT / "knowledge_base" / "sources" / "raw" / "external" / "jecfa"
KB_PATH = ROOT / "knowledge_base" / "additives_standard.json"
CATALOG_PATH = ROOT / "knowledge_base" / "sources" / "catalog" / "source_citation_catalog.json"


def _norm(code):
    return re.sub(r"[()\s]", "", re.sub(r"^(ins|e)\s*", "", str(code).strip(), flags=re.I)).lower()


def _page_codes(chemical_id):
    path = JECFA_DIR / f"Chemical_{chemical_id}.html"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<(script|style).*?</\1>", "", text))))
    m = re.search(r"INS matches (.*?) Evaluations", text) or re.search(r"\bINS (.*?) Functional Class", text)
    return [_norm(x) for x in re.split(r"[,;/]| and ", m.group(1))] if m else []


def _matches(code, page_codes):
    return any(c == code or c.startswith(code) or code.startswith(c) for c in page_codes if c)


class RecognitionSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = pipeline.load_knowledge_base(str(ROOT / "data"))
        mentions = [{"raw_mention": "Natri benzoat (211)", "category": "additive"},
                    {"raw_mention": "Tartrazine", "category": "additive"},
                    {"raw_mention": "INS 999", "category": "additive"},
                    {"raw_mention": "đường", "category": "ingredient"}]
        cls.rows = evidence.build_recognition_summary(pipeline.link_all_mentions(mentions, cls.kb))

    def test_one_row_per_additive_and_plain_ingredients_are_left_out(self):
        self.assertEqual([r["raw"] for r in self.rows], ["Natri benzoat (211)", "Tartrazine", "INS 999"])
        self.assertEqual([r["status"] for r in self.rows], ["linked", "linked", "unlinked"])

    def test_row_reports_what_was_read_and_how_it_was_identified(self):
        benzoate, tartrazine = self.rows[0], self.rows[1]
        self.assertEqual(benzoate["ins_code"], "INS 211")
        self.assertIn("ghi trên nhãn", benzoate["how"])
        self.assertIn("suy ra từ tên", tartrazine["how"])

    def test_unlinked_additive_is_shown_and_not_guessed(self):
        unlinked = self.rows[2]
        self.assertEqual((unlinked["ins_code"], unlinked["name"], unlinked["sources"]), ("", "", []))

    def test_adi_source_is_the_one_behind_the_displayed_number(self):
        benzoate, tartrazine = self.rows[0], self.rows[1]
        adi_source = next(s for s in benzoate["sources"] if s["role"] == "ADI")
        self.assertTrue(adi_source["local_file"].endswith("Chemical_1098.html"))  # trang JECFA đúng là Sodium benzoate
        # Tartrazine: KB ghi ADI 0–7,5 của EFSA 2009 → nguồn phải là bài EFSA đó, không phải trang JECFA (ghi 0–10)
        self.assertIn("EFSA", tartrazine["adi_text"])
        self.assertTrue(next(s for s in tartrazine["sources"] if s["role"] == "ADI")["local_file"].endswith("efsa_2009_1331.pdf"))

    def test_every_source_file_exists_on_disk(self):
        for row in self.rows:
            for s in row["sources"]:
                self.assertTrue(s["local_file"] and (ROOT / s["local_file"]).is_file(), (row["raw"], s["title"]))


class ThreeSectionAnswerFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = pipeline.load_knowledge_base(str(ROOT / "data"))

    def _rows(self, mentions):
        return evidence.build_recognition_summary(pipeline.link_all_mentions(mentions, self.kb))

    def test_headline_lists_codes_keywords_and_unrecognised_text(self):
        mentions = [{"raw_mention": "Natri benzoat (211)", "category": "additive"},
                    {"raw_mention": "INS 999", "category": "additive"},
                    {"raw_mention": "đường", "category": "ingredient"}]
        headline = evidence.recognition_headline(mentions, self._rows(mentions), "Nước ngọt X")
        self.assertIn("Sản phẩm: Nước ngọt X", headline)
        self.assertIn("INS 211 (Natri benzoat)", headline)
        self.assertIn("Chưa nhận diện được: “INS 999”", headline)
        self.assertIn("Từ khóa khác: đường", headline)

    def test_headline_says_not_clear_enough_when_nothing_was_read(self):
        self.assertEqual(evidence.recognition_headline([], []), "Không đủ dữ liệu rõ nét")

    def test_references_come_only_from_sources_actually_used(self):
        mentions = [{"raw_mention": "Natri benzoat (211)", "category": "additive"}]
        rows = self._rows(mentions)
        refs = evidence.reference_lines([], rows)
        self.assertTrue(refs)
        self.assertTrue(all(r["title"] and r["from"] == "tra cứu theo phụ gia" for r in refs))
        self.assertIn("ADI của INS 211", refs[0]["where"])
        self.assertEqual(evidence.reference_lines([], []), [])  # không có nguồn thì không bịa

    def test_cited_sources_are_listed_first_and_not_duplicated(self):
        rows = self._rows([{"raw_mention": "Natri benzoat (211)", "category": "additive"}])
        cited = [{"n": 1, "title": rows[0]["sources"][0]["title"], "scope": rows[0]["sources"][0]["scope"], "locator": ""}]
        refs = evidence.reference_lines(cited, rows)
        self.assertEqual(refs[0]["n"], 1)
        self.assertEqual(sum(1 for r in refs if r["title"] == cited[0]["title"] and r["where"] == (cited[0]["scope"] or "")), 1)


class JecfaLinksPointToTheRightChemicalTests(unittest.TestCase):
    """Từng có 44 id trang JECFA bịa (vd Tartrazine → trang 'PROPYL 2-METHYL-3-FURYL DISULFIDE'): id phải thuộc đúng mã INS."""

    def _check(self, label, code, chemical_id, bad):
        codes = _page_codes(chemical_id)
        if codes is None:
            bad.append((label, code, chemical_id, "không có file trang"))
        elif not _matches(_norm(code), codes):
            bad.append((label, code, chemical_id, f"trang là INS {codes}"))

    def test_resolver_map_ids_match_their_ins_code(self):
        bad = []
        for ins, entry in claim_locator_resolver.JECFA_CHEMICAL_MAP.items():
            m = re.search(r"Home/Chemical/(\d+)", entry["url"])
            if m:
                self._check("MAP", ins, m.group(1), bad)
        self.assertEqual(bad, [])

    def test_catalog_ids_match_the_ins_code_in_their_title(self):
        bad = []
        for key, entry in json.loads(CATALOG_PATH.read_text(encoding="utf-8")).items():
            m = re.search(r"Home/Chemical/(\d+)", entry.get("url", ""))
            codes = re.findall(r"INS\s*([0-9]{3,4}[a-z]?(?:\([ivxIVX]+\))?)", entry.get("title", ""))
            if m and codes:
                self._check("CATALOG", codes[0], m.group(1), bad)
        self.assertEqual(bad, [])

    def test_kb_entries_link_only_to_pages_of_their_own_additive(self):
        bad = []
        for item in json.loads(KB_PATH.read_text(encoding="utf-8")):
            code = _norm(item["ins_code"].split("+")[0])
            for chemical_id in set(re.findall(r"Home/Chemical/(\d+)", json.dumps(item, ensure_ascii=False))):
                self._check("KB", item["ins_code"], chemical_id, bad)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
