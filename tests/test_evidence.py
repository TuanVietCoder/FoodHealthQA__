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
