import unittest

from modules import evidence


PASSAGES = [
    {"citation_id": "VN_TT24_2019", "locator": "VN_TT24_2019#PhuLuc1_Bang1",
     "text": "Phụ lục 1 TT24/2019: 334. 951 Aspartam Aspartame Chất điều vị, chất tạo ngọt", "ins_codes": ["951"]},
    {"citation_id": "JECFA_EVALUATIONS", "locator": "JECFA_EVALUATIONS#951",
     "text": "INS 951 Aspartame. ADI: 0-40 mg/kg thể trọng/ngày.", "ins_codes": ["951"]},
]


class CitedAnswerTests(unittest.TestCase):
    def test_numbers_follow_first_appearance_and_reuse(self):
        answer = "ADI 0-40 [JECFA_EVALUATIONS]. Được phép [VN_TT24_2019]. Lặp lại [JECFA_EVALUATIONS]."
        out = evidence.build_cited_answer(answer, PASSAGES, ["951"])
        self.assertEqual(out["text"], "ADI 0-40 [1]. Được phép [2]. Lặp lại [1].")
        self.assertEqual([s["citation_id"] for s in out["sources"]], ["JECFA_EVALUATIONS", "VN_TT24_2019"])

    def test_multi_citation_tag_becomes_adjacent_numbers(self):
        out = evidence.build_cited_answer("Được phép [VN_TT24_2019, CODEX_STAN_192_1995].", PASSAGES, ["951"])
        self.assertEqual(out["text"], "Được phép [1][2].")

    def test_unrecognized_tag_is_left_alone_and_gets_no_source(self):
        out = evidence.build_cited_answer("Bịa [nguon_gia] ở đây.", PASSAGES, ["951"])
        self.assertEqual(out["text"], "Bịa [nguon_gia] ở đây.")
        self.assertEqual(out["sources"], [])

    def test_source_has_title_scope_quote_url_and_tier(self):
        out = evidence.build_cited_answer("Được phép [VN_TT24_2019].", PASSAGES, ["951"])
        src = out["sources"][0]
        self.assertEqual(src["n"], 1)
        self.assertIn("24/2019", src["title"])
        self.assertTrue(src["scope"])
        self.assertTrue(src["url"].startswith("http"))
        self.assertIn("Aspartam", src["quote"])
        self.assertEqual(src["tier"], 1)

    def test_empty_answer(self):
        self.assertEqual(evidence.build_cited_answer(None), {"text": "", "sources": []})

    def test_tiers(self):
        self.assertEqual(evidence.source_tier("VN_TT24_2019")[0], 1)
        self.assertEqual(evidence.source_tier("JECFA_EVALUATIONS")[0], 2)
        self.assertEqual(evidence.source_tier("USDA_FDC")[0], 3)
        self.assertEqual(evidence.source_tier("KHONG_RO")[0], 4)


class DomainTests(unittest.TestCase):
    def test_labels_match_dataset(self):
        self.assertEqual(set(evidence.DOMAIN_LABELS), {
            "additive_regulation", "allergen_detection", "vulnerable_groups", "insufficient_data"})

    def test_priority(self):
        self.assertEqual(evidence.classify_domain({}, {"status": "ABSTAIN"}, {})[0], "insufficient_data")
        self.assertEqual(evidence.classify_domain({"entities": {"allergens": ["soy"]}}, {}, {})[0], "allergen_detection")
        self.assertEqual(evidence.classify_domain({}, {}, {"conditions": ["suy thận"]})[0], "vulnerable_groups")
        self.assertEqual(evidence.classify_domain({"entities": {"additives": ["INS 951"]}}, {}, {})[0], "additive_regulation")


class InsufficiencyTests(unittest.TestCase):
    def test_no_reason_when_evidence_is_sufficient(self):
        evaluation = {"s_perception": 0.9, "s_total": 0.9}
        linked = [{"matched_entity": {"ins_code": "951"}}]
        self.assertEqual(evidence.insufficiency_reasons(evaluation, {"full_text": "x"}, linked, PASSAGES, None), [])

    def test_blurry_image_missing_ingredients_and_low_score(self):
        evaluation = {"s_perception": 0.4, "s_total": 0.5}
        reasons = evidence.insufficiency_reasons(evaluation, {"full_text": "chữ mờ"}, [], [], None)
        text = " ".join(reasons)
        self.assertIn("OCR", text)
        self.assertIn("Không nhận diện được phụ gia", text)
        self.assertIn("đoạn bằng chứng", text)
        self.assertIn("0.65", text)

    def test_empty_ocr_text(self):
        reasons = evidence.insufficiency_reasons({"s_perception": 0.0, "s_total": 0.0}, {"full_text": ""}, [], PASSAGES, None)
        self.assertTrue(any("Không đọc được chữ" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()


class LocalSourceTests(unittest.TestCase):
    def test_efsa_opinions_resolve_to_files_whose_first_page_carries_the_same_doi(self):
        import fitz
        cases = {"https://www.efsa.europa.eu/en/efsajournal/pub/7594": "2022.7594",
                 "https://www.efsa.europa.eu/en/efsajournal/pub/1327": "2009.1327",
                 "https://pmc.ncbi.nlm.nih.gov/articles/PMC7009848/": "2017.4910",
                 "https://pmc.ncbi.nlm.nih.gov/articles/PMC7009865/": "2017.4911",
                 "https://doi.org/10.2903/j.efsa.2025.9317": "2025.9317",
                 "https://doi.org/10.2903/j.efsa.2017.4786": "2017.4786"}
        for url, doi in cases.items():
            lf = evidence.local_source_file("X", url)
            if not lf:
                self.skipTest(f"chưa có file cục bộ cho {url}")
            with fitz.open(evidence._ROOT / lf) as doc:
                self.assertIn(doi, "".join(doc[0].get_text().split()).replace("j.efsa.", ""), url)

    def test_local_file_resolves_for_catalog_source(self):
        out = evidence.build_cited_answer("Được phép [VN_TT24_2019].", PASSAGES, ["951"])
        self.assertTrue(out["sources"][0]["local_file"].endswith("09_VBHN_BYT_2024.pdf"))
        self.assertEqual(out["sources"][0]["ins_codes"], ["951"])

    def test_locate_pages_finds_aspartame_row_in_tt24(self):
        lf = evidence.local_source_file("VN_TT24_2019", "")
        if not lf:
            self.skipTest("file TT24 cục bộ không có")
        pages = evidence.locate_pages(lf, "951")
        self.assertTrue(pages and pages[0][0] == 40)
        self.assertTrue(evidence.render_pdf_row(lf, pages[0][0], "951").startswith(b"\x89PNG"))

    def test_local_text_for_html_and_doc(self):
        html = evidence.local_source_file("US_FDA_21CFR172_804", "")
        doc = evidence.local_source_file("VN_QCVN_11_4_2012", "")
        if html:
            self.assertIn("Aspartame", evidence.local_text(html))
        if doc:  # .doc đã được chuyển sang PDF để xem theo trang
            self.assertTrue(doc.endswith(".pdf"))


class SourceResolutionTests(unittest.TestCase):
    TT24 = "knowledge_base/sources/raw/vn/09_VBHN_BYT_2024.pdf"
    CODEX = "knowledge_base/sources/raw/external/cxs_192e.pdf"

    def _need(self, rel):
        if not (evidence._ROOT / rel).is_file():
            self.skipTest(f"thiếu {rel}")

    def test_tt24_scope_follows_the_passage_locator(self):
        scopes = {sec: evidence.resolve_source("VN_TT24_2019", f"VN_TT24_2019#PhuLuc{sec}")["scope"]
                  for sec in ("2A", "2B")}
        self.assertIn("2A", scopes["2A"])
        self.assertIn("2B", scopes["2B"])
        self.assertIn("Bảng 1", evidence.resolve_source("VN_TT24_2019", "VN_TT24_2019#PhuLuc1_Bang1")["scope"])

    def test_tt24_locate_is_restricted_to_the_locator_section(self):
        self._need(self.TT24)
        p1 = evidence.locate_pages(self.TT24, "951", 4, "VN_TT24_2019#PhuLuc1_Bang1")
        p2a = evidence.locate_pages(self.TT24, "951", 4, "VN_TT24_2019#PhuLuc2A")
        p2b = evidence.locate_pages(self.TT24, "951", 4, "VN_TT24_2019#PhuLuc2B")
        ranges = evidence._section_ranges(self.TT24)
        self.assertTrue(p1 and all(ranges["1"][0] <= pg <= ranges["1"][1] for pg, _ in p1))
        self.assertTrue(p2a and all(ranges["2A"][0] <= pg <= ranges["2A"][1] for pg, _ in p2a))
        self.assertTrue(p2b and all(ranges["2B"][0] <= pg <= ranges["2B"][1] for pg, _ in p2b))

    def test_codex_group_code_is_found_without_registry_name(self):
        self._need(self.CODEX)
        pages = evidence.locate_pages(self.CODEX, "100i")
        self.assertTrue(pages and "100(i)" in pages[0][1])
        self.assertTrue(evidence.render_pdf_row(self.CODEX, pages[0][0], "100i").startswith(b"\x89PNG"))

    def test_code_inside_a_sentence_is_not_a_row(self):
        self._need(self.CODEX)
        for _, row in evidence.locate_pages(self.CODEX, "110"):
            self.assertNotIn("For use of", row)

    def test_jecfa_passage_resolves_to_its_own_additive_not_aspartame(self):
        p = {"citation_id": "JECFA_EVALUATIONS", "locator": "JECFA_EVALUATIONS#102", "text": "INS 102", "ins_codes": ["102"]}
        src = evidence.build_cited_answer("x [JECFA_EVALUATIONS]", [p], ["102"])["sources"][0]
        self.assertIn("Tartrazine", src["title"])
        self.assertNotIn("/Chemical/62", src["url"])
        p0 = {"citation_id": "JECFA_EVALUATIONS", "locator": "JECFA_EVALUATIONS#110", "text": "INS 110", "ins_codes": ["110"]}
        src0 = evidence.build_cited_answer("x [JECFA_EVALUATIONS]", [p0], ["110"])["sources"][0]
        self.assertEqual(src0["url"], "")  # không có trang JECFA riêng → không gắn link sai
