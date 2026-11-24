"""Cổng nguồn: câu trả lời của LLM chỉ được hiển thị khi bám đúng danh sách bằng chứng đã truy xuất."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import pipeline  # noqa: E402

ALLOWED = ["JECFA_EVALUATIONS", "VN_TT24_2019"]


class CheckAnswerSourcesTests(unittest.TestCase):
    def test_answer_citing_only_allowed_sources_passes(self):
        ok, _ = pipeline.check_answer_sources("Natri benzoat có ADI 0–20 mg/kg [JECFA_EVALUATIONS] [VN_TT24_2019].", ALLOWED, "CAUTION")
        self.assertTrue(ok)

    def test_source_outside_the_retrieved_list_is_rejected_even_if_it_is_a_real_source(self):
        ok, reason = pipeline.check_answer_sources("Có ADI [JECFA_EVALUATIONS]. Theo FDA [US_FDA_FALCPA].", ALLOWED, "CAUTION")
        self.assertFalse(ok)
        self.assertIn("US_FDA_FALCPA", reason)

    def test_invented_source_is_rejected(self):
        ok, reason = pipeline.check_answer_sources("Đã được chứng minh [Nguồn: Viện X 2031].", ALLOWED, "SAFE")
        self.assertFalse(ok)
        self.assertIn("ngoài danh sách", reason)

    def test_answer_without_any_source_is_rejected(self):
        ok, reason = pipeline.check_answer_sources("Chất này an toàn, bạn cứ dùng.", ALLOWED, "SAFE")
        self.assertEqual((ok, reason), (False, "câu trả lời không dẫn nguồn nào"))

    def test_no_retrieved_sources_at_all_gives_a_distinct_reason(self):
        ok, reason = pipeline.check_answer_sources("Chất này an toàn.", [], "SAFE")
        self.assertEqual((ok, reason), (False, "không truy xuất được nguồn nào cho câu hỏi này"))

    def test_an_abstention_answer_needs_no_source_but_may_not_invent_one(self):
        self.assertTrue(pipeline.check_answer_sources("Câu hỏi nằm ngoài phạm vi của hệ thống.", [], "ABSTAIN")[0])
        self.assertFalse(pipeline.check_answer_sources("Ngoài phạm vi [BIA_RA_NGUON].", [], "ABSTAIN")[0])

    def test_ordinary_brackets_are_not_mistaken_for_sources(self):
        ok, _ = pipeline.check_answer_sources("Liều [mg/kg] và [ADI] xem [1] [JECFA_EVALUATIONS].", ALLOWED, "SAFE")
        self.assertTrue(ok)

    def test_several_codes_in_one_bracket_are_each_checked(self):
        self.assertTrue(pipeline.check_answer_sources("OK [JECFA_EVALUATIONS, VN_TT24_2019].", ALLOWED, "SAFE")[0])
        self.assertFalse(pipeline.check_answer_sources("OK [JECFA_EVALUATIONS, EFSA_OPINIONS].", ALLOWED, "SAFE")[0])


class GenerateLlmAnswerGateTests(unittest.TestCase):
    """generate_llm_answer: thử lại đúng một lần rồi từ chối (trả None, ghi lý do), không trả câu trả lời không nguồn."""

    def _run(self, replies):
        evaluation = {"title": "t", "advice": "a", "safety_label": "CAUTION", "alerts": [], "regulatory": [],
                      "citations": ["JECFA_EVALUATIONS"]}
        linked = [{"raw_mention": "Natri benzoat", "matched_entity": {"canonical_name": "Natri benzoat", "ins_code": "INS 211",
                                                                      "jecfa_adi": "0-20"}}]
        calls = []

        def fake_llm(prompt, **kwargs):
            calls.append(prompt)
            return replies[min(len(calls) - 1, len(replies) - 1)]

        with mock.patch.object(pipeline, "_llm_ready", return_value=True), \
                mock.patch.object(pipeline, "_get_gemini_api_key", return_value="k"), \
                mock.patch.object(pipeline, "get_llm_provider", return_value="gemini"), \
                mock.patch.object(pipeline, "_call_llm", side_effect=fake_llm):
            answer = pipeline.generate_llm_answer(evaluation, linked, {"conditions": [], "allergies": []}, "An toàn không?")
        return answer, evaluation, calls

    def test_sourced_answer_is_returned_without_retry(self):
        answer, evaluation, calls = self._run(["Natri benzoat có ADI 0–20 mg/kg [JECFA_EVALUATIONS]. SAFETY_LABEL: CAUTION"])
        self.assertIn("[JECFA_EVALUATIONS]", answer)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("answer_refusal", evaluation)

    def test_unsourced_answer_is_retried_once_and_the_sourced_retry_is_used(self):
        answer, _, calls = self._run(["An toàn. SAFETY_LABEL: SAFE", "Có ADI 0–20 [JECFA_EVALUATIONS]. SAFETY_LABEL: CAUTION"])
        self.assertEqual(len(calls), 2)
        self.assertIn("NHẮC LẠI BẮT BUỘC", calls[1])
        self.assertIn("[JECFA_EVALUATIONS]", answer)

    def test_answer_that_stays_unsourced_is_refused(self):
        answer, evaluation, calls = self._run(["Cứ dùng thoải mái. SAFETY_LABEL: SAFE"])
        self.assertIsNone(answer)
        self.assertEqual(len(calls), 2)  # một lần đầu + đúng một lần thử lại
        self.assertEqual(evaluation["answer_refusal"], "câu trả lời không dẫn nguồn nào")

    def test_answer_with_foreign_source_is_refused(self):
        answer, evaluation, _ = self._run(["Theo FDA [US_FDA_FALCPA] thì ổn. SAFETY_LABEL: SAFE"])
        self.assertIsNone(answer)
        self.assertIn("US_FDA_FALCPA", evaluation["answer_refusal"])

    def test_gate_can_be_switched_off_for_comparison_with_old_runs(self):
        with mock.patch.object(pipeline, "STRICT_SOURCE_GATE", False):
            answer, evaluation, _ = self._run(["Cứ dùng thoải mái. SAFETY_LABEL: SAFE"])
        self.assertIsNotNone(answer)
        self.assertNotIn("answer_refusal", evaluation)


if __name__ == "__main__":
    unittest.main()
