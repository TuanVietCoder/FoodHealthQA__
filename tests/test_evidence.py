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
        # người dùng khỏe mạnh (conditions chỉ có "healthy") không phải "nhóm dễ tổn thương"
        for conds in (["healthy"], ["healthy", "healthy_adult"]):
            self.assertEqual(evidence.classify_domain({"entities": {"additives": ["INS 951"]}}, {}, {"conditions": conds})[0],
                             "additive_regulation")


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


class BottomLineTests(unittest.TestCase):
    ROWS = [{"status": "linked", "name": "Natri benzoat (Sodium benzoate)", "ins_code": "INS 211"},
            {"status": "linked", "name": "Aspartam (Aspartame)", "ins_code": "INS 951"}]

    def bl(self, evaluation, rows=None, declared=True, profile="40 tuổi, Nữ, tăng huyết áp", allergens=None):
        return evidence.build_bottom_line(evaluation, self.ROWS if rows is None else rows, profile, declared, allergens)

    def test_each_status_gives_a_concrete_recommendation(self):
        crit = self.bl({"status": "CRITICAL_ALERT", "alerts": [{"additive": "Aspartame (INS 951)"}]})
        self.assertIn("không nên dùng", crit)
        self.assertIn("Aspartame (INS 951)", crit)
        self.assertIn("40 tuổi, Nữ, tăng huyết áp", crit)
        allergy = self.bl({"status": "WARNING", "alerts": [{"additive": "bột sữa, đường"}]}, allergens=lambda g: "sữa")
        self.assertIn("không nên dùng", allergy)
        self.assertIn("trên nhãn có sữa", allergy)
        pal = self.bl({"status": "WARNING", "alerts": [{"additive": "x", "regulation": "FAO_WHO_JEMRA_PART_3"}]},
                      allergens=lambda g: "đậu phộng")
        self.assertIn("có thể chứa", pal)
        self.assertIn("đậu phộng", pal)
        caution = self.bl({"status": "CAUTION", "alerts": [{"additive": "Natri benzoat (INS 211)"}]})
        self.assertIn("nên hạn chế", caution)
        self.assertIn("Natri benzoat (INS 211)", caution)
        self.assertIn("nên dùng ít", self.bl({"status": "CAUTION", "alerts": []}))
        self.assertIn("không nên cho bé", self.bl({"status": "WARNING", "alerts": [{"additive": "x", "label_age_rule": True}]}))
        self.assertIn("dưới 12 tháng", self.bl({"status": "CAUTION", "alerts": [{"additive": "INS 211", "infant_rule": True}]}))

    def test_safe_names_what_was_read_and_is_not_a_guarantee(self):
        text = self.bl({"status": "SCREENED_NO_ALERT", "population_notes": []})
        self.assertIn("2 phụ gia", text)
        self.assertIn("Natri benzoat", text)
        self.assertIn("chưa thấy chất nào cần tránh", text)
        self.assertIn("chưa phải bảo đảm an toàn", text)

    def test_safe_with_unlinked_text_or_nothing_read_stays_cautious(self):
        rows = self.ROWS + [{"status": "unlinked", "raw": "INS 33O"}]
        self.assertIn("1 cụm chữ máy chưa nhận diện được", self.bl({"status": "SCREENED_NO_ALERT"}, rows=rows))
        self.assertIn("chưa chắc chắn", self.bl({"status": "SCREENED_NO_ALERT"}, rows=[]))

    def test_condition_without_data_never_sounds_like_a_pass(self):
        ev = {"status": "SCREENED_NO_ALERT", "population_notes": [
            {"code": "CONDITION_NO_ADDITIVE_EVIDENCE", "group": "tăng huyết áp", "text": "x"}]}
        text = self.bl(ev)
        self.assertIn("Chưa thể nói", text)
        self.assertIn("tăng huyết áp", text)
        self.assertNotIn("chưa thấy chất nào cần tránh", text)

    def test_abstain_says_what_to_do_and_undeclared_profile_is_not_quoted(self):
        text = self.bl({"status": "ABSTAIN", "data_gaps": [{"need": "chụp lại phần thành phần"}]}, declared=False)
        self.assertIn("Chưa thể kết luận", text)
        self.assertIn("chụp lại phần thành phần", text)
        self.assertIn("Chưa thể trả lời", self.bl({"status": "ABSTAIN", "scope_issues": ["y khoa"]}))
        self.assertIn("Với thông tin hiện có", self.bl({"status": "CAUTION", "alerts": [{"additive": "INS 211"}]}, declared=False))


class FriendlyRefusalTests(unittest.TestCase):
    def test_technical_reasons_become_plain_sentences(self):
        for reason, key in [("nhãn do mô hình ghi (CAUTION) khác nhãn của bộ luật (SAFE)", "chưa khớp với kết luận"),
                            ("câu trả lời dẫn nguồn ngoài danh sách bằng chứng đã truy xuất: X", "không nằm trong số tài liệu"),
                            ("câu trả lời không dẫn nguồn nào", "không dẫn được tài liệu nào"),
                            ("lý do lạ", "chưa chắc")]:
            msg = evidence.friendly_refusal_reason(reason)
            self.assertIn(key, msg)
            for jargon in ("CAUTION", "SAFE", "bộ luật", "truy xuất"):
                self.assertNotIn(jargon, msg)


class JecfaSourceDisplayTests(unittest.TestCase):
    def test_unverified_jecfa_record_shows_no_question_marks(self):
        info = evidence.resolve_source("JECFA_EVALUATIONS", None, ["307c"])
        self.assertFalse(info["verified"])
        self.assertNotIn("?", info["title"] + info["scope"])
        self.assertIn("Chưa có", info["scope"])

    def test_verified_jecfa_record_shows_year_and_adi(self):
        info = evidence.resolve_source("JECFA_EVALUATIONS", None, ["300"])
        self.assertTrue(info["verified"])
        self.assertIn("ADI:", info["scope"])
        self.assertNotIn("?", info["scope"])

    def test_verified_record_missing_fields_shows_only_what_exists(self):
        info = evidence.resolve_source("JECFA_EVALUATIONS", None, ["450(i)"])
        self.assertNotIn("?", info["title"] + info["scope"])


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
