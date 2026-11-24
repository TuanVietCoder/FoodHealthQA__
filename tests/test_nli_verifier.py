import unittest
from modules.nli_verifier import (
    split_into_claims,
    classify_claim_nli,
    verify_answer_faithfulness,
)


class NLIVerifierTests(unittest.TestCase):
    def test_split_into_claims_strips_metadata(self):
        text = (
            "Sản phẩm này có chứa chất bảo quản Natri benzoat (INS 211). "
            "Trẻ em dưới 6 tuổi cần thận trọng khi sử dụng [EFSA_OPINIONS].\n"
            "SAFETY_LABEL: CAUTION"
        )
        claims = split_into_claims(text)
        self.assertEqual(len(claims), 2)
        self.assertTrue(claims[0]["text"].startswith("Sản phẩm này có chứa"))
        self.assertIn("EFSA_OPINIONS", claims[1]["citations"])
        self.assertTrue(claims[1]["is_high_risk"])

    def test_classify_claim_heuristic_contradiction(self):
        evidence = "Thông tư 24/2019/TT-BYT cho phép sử dụng INS 211 trong danh mục phụ gia thực phẩm."
        claim = "Chất INS 211 bị cấm hoàn toàn theo luật Việt Nam."
        res = classify_claim_nli(claim, evidence)
        self.assertEqual(res["label"], "CONTRADICTION")

    def test_classify_claim_heuristic_entailment(self):
        evidence = "JECFA thiết lập mức tiêu thụ hàng ngày chấp nhận được ADI từ 0 đến 5 mg/kg thể trọng đối với Natri benzoat."
        claim = "Natri benzoat có ngưỡng an toàn ADI từ 0 đến 5 mg/kg thể trọng theo đánh giá của JECFA."
        res = classify_claim_nli(claim, evidence)
        self.assertEqual(res["label"], "ENTAILMENT")

    def test_classify_claim_with_mock_llm(self):
        evidence = "EFSA kết luận không thể thiết lập mức ADI an toàn cho Titanium dioxide (E171)."
        claim = "Titanium dioxide E171 không còn được coi là an toàn do lo ngại về độc tính gen."

        def mock_llm(prompt):
            return '{"label": "ENTAILMENT", "reason": "Bằng chứng EFSA xác nhận không an toàn"}'

        res = classify_claim_nli(claim, evidence, call_llm_fn=mock_llm)
        self.assertEqual(res["label"], "ENTAILMENT")
        self.assertEqual(res["source"], "llm_nli")

    def test_verify_answer_faithfulness_clean_and_flagged(self):
        evidence = "Thông tư 24/2019/TT-BYT cho phép sử dụng Aspartame (INS 951). JECFA ADI là 40 mg/kg."
        clean_answer = (
            "Sản phẩm có chứa Aspartame (INS 951) [VN_TT24_2019]. "
            "Chất này được phép sử dụng theo Thông tư 24 với liều dùng cho phép."
        )
        res_clean = verify_answer_faithfulness(clean_answer, evidence, trigger_only=False)
        self.assertIn(res_clean["status"], {"ALL_SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"})
        # không có LLM → chỉ là đối chiếu từ khóa, không được gọi là "xác thực"
        self.assertEqual(res_clean["method"], "keyword_overlap")
        self.assertIn("Đối chiếu từ khóa", res_clean["badge_html"])
        self.assertNotIn("XÁC THỰC", res_clean["badge_html"])
        self.assertTrue(all(c["confidence"] is None for c in res_clean["claims"]))

        contradict_answer = (
            "Chất Aspartame INS 951 bị cấm hoàn toàn theo luật Việt Nam và không được phép dùng [VN_TT24_2019]."
        )
        res_contra = verify_answer_faithfulness(contradict_answer, evidence, trigger_only=False)
        self.assertEqual(res_contra["status"], "CONTRADICTION_DETECTED")
        self.assertIn("mâu thuẫn", res_contra["badge_html"])

    def test_no_claims_is_not_applicable(self):
        res = verify_answer_faithfulness("SAFETY_LABEL: SAFE", "bằng chứng", trigger_only=False)
        self.assertEqual(res["status"], "NO_CLAIMS")
        self.assertIsNone(res["faithfulness_score"])


if __name__ == "__main__":
    unittest.main()

