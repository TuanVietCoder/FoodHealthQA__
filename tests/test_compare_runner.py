import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import compare_runner as cr  # noqa: E402
from modules import pipeline as P  # noqa: E402

RES = {"kb_additives": [], "hazards": [], "fndds": [], "off_products": []}
PERSONA = {"conditions": [], "allergies": []}
LABEL = "Nước, đường, chất điều vị (621)"


def grounded(label="UNSAFE", answer="Có nguy cơ [VN_TT24_2019] [JECFA_EVALUATIONS]. SAFETY_LABEL: UNSAFE"):
    return {
        "llm_answer": answer, "safety_label": label, "evaluation": {"status": label},
        "retrieved_passages": [{"citation_id": "VN_TT24_2019", "locator": "VN_TT24_2019#PhuLuc1_Bang1", "kind": "x"}],
    }


class RunConfigTests(unittest.TestCase):
    def setUp(self):
        patches = [
            mock.patch.object(cr, "provider_ready", return_value=(True, "ok")),
            mock.patch.object(P, "generate_llm_only_answer", return_value="Tôi nghĩ ổn. SAFETY_LABEL: CAUTION"),
            mock.patch.object(P, "run_grounded_pipeline", return_value=grounded()),
            mock.patch.object(P, "get_last_llm_model", return_value="fake-model"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(P.set_llm_provider, None)
        P.set_llm_provider("gemini")

    def kwargs(self, **over):
        base = dict(persona=PERSONA, question="An toàn không?", label_text=LABEL, product_name="Mì",
                    ocr_result={"full_text": "Thành phần: Nước, dường, chat điều vị (621)", "avg_confidence": 0.9},
                    resources=RES)
        base.update(over)
        return base

    def test_a_has_no_retrieval_and_no_citations(self):
        r = cr.run_config("A", "ollama", **self.kwargs())
        self.assertIsNone(r["error"])
        self.assertEqual(r["safety_label"], "CAUTION")
        self.assertEqual(r["passages"], [])
        self.assertEqual(r["citations"], [])
        self.assertEqual(r["model"], "fake-model")

    def test_b_and_c_report_label_citations_and_passages(self):
        for config in ("B", "C"):
            r = cr.run_config(config, "ollama", **self.kwargs())
            self.assertIsNone(r["error"], config)
            self.assertEqual(r["safety_label"], "UNSAFE")
            self.assertEqual(r["citations"], ["VN_TT24_2019", "JECFA_EVALUATIONS"])
            self.assertEqual(len(r["passages"]), 1)
            self.assertGreaterEqual(r["mentions"], 1)

    def test_c_without_ocr_reports_error_not_exception(self):
        r = cr.run_config("C", "ollama", **self.kwargs(ocr_result=None))
        self.assertIn("Cấu hình C cần ảnh", r["error"])
        r = cr.run_config("C", "ollama", **self.kwargs(ocr_result={"is_ocr_unavailable": True, "ocr_error": "paddle lỗi"}))
        self.assertEqual(r["error"], "paddle lỗi")

    def test_empty_answer_is_an_error(self):
        with mock.patch.object(P, "generate_llm_only_answer", return_value=None):
            r = cr.run_config("A", "ollama", **self.kwargs())
        self.assertIn("không trả về câu trả lời", r["error"])

    def test_provider_not_ready_and_provider_restored(self):
        with mock.patch.object(cr, "provider_ready", return_value=(False, "Ollama chưa chạy")):
            r = cr.run_config("A", "ollama", **self.kwargs())
        self.assertEqual(r["error"], "Ollama chưa chạy")
        self.assertEqual(P.get_llm_provider(), "gemini")
        cr.run_config("A", "ollama", **self.kwargs())
        self.assertEqual(P.get_llm_provider(), "gemini")  # đã khôi phục sau khi chạy

    def test_pipeline_exception_becomes_error(self):
        with mock.patch.object(P, "run_grounded_pipeline", side_effect=RuntimeError("boom")):
            r = cr.run_config("B", "ollama", **self.kwargs())
        self.assertEqual(r["error"], "RuntimeError: boom")


class RunComparisonTests(unittest.TestCase):
    def test_runs_each_provider_and_config_once_with_shared_ocr_and_progress(self):
        calls, progress = [], []
        ocr = {"full_text": "Thành phần: Nước, dường, chat điều vị (621)", "avg_confidence": 0.9}

        def fake_run_config(config, provider, **kw):
            calls.append((provider, config))
            return {"config": config, "provider": provider, "safety_label": "SAFE", "citations": [],
                    "passages": [], "seconds": 1.0, "error": None, "answer": "x", "model": "m"}

        with mock.patch.object(cr, "run_ocr_once", return_value=ocr) as ocr_mock, \
                mock.patch.object(cr, "run_config", side_effect=fake_run_config):
            out = cr.run_comparison(
                ["gemini", "ollama"], ["A", "B", "C"], persona=PERSONA, question="q", label_text=LABEL,
                product_name="p", image_path="img.jpg", sample_key="0001.jpeg", resources=RES,
                progress_cb=lambda r, done, total, ocr_info: progress.append((done, total)),
            )
        ocr_mock.assert_called_once()  # OCR dùng chung cho cả hai nhà cung cấp
        self.assertEqual(calls, [("gemini", "A"), ("gemini", "B"), ("gemini", "C"),
                                 ("ollama", "A"), ("ollama", "B"), ("ollama", "C")])
        self.assertEqual(progress[-1], (6, 6))
        self.assertLess(out["ocr"]["match_ratio"], 1.0)  # OCR có lỗi so với văn bản chuẩn
        self.assertGreater(out["ocr"]["match_ratio"], 0.8)
        self.assertEqual(out["ocr"]["text"], ocr["full_text"])

    def test_no_image_marks_c_unavailable_and_skips_ocr(self):
        with mock.patch.object(cr, "run_ocr_once") as ocr_mock, mock.patch.object(cr, "run_config") as run:
            run.return_value = {"config": "A", "provider": "ollama", "error": None, "seconds": 0, "answer": "x",
                                "safety_label": "SAFE", "citations": [], "passages": []}
            out = cr.run_comparison(["ollama"], ["A", "B", "C"], persona=PERSONA, question="q",
                                    label_text=LABEL, product_name="p", image_path=None,
                                    sample_key="custom", resources=RES)
        ocr_mock.assert_not_called()
        self.assertTrue(out["ocr"]["unavailable"])


class SummaryTests(unittest.TestCase):
    @staticmethod
    def res(provider, config, label, cites=0, passages=0, seconds=1.0, error=None):
        return {"provider": provider, "config": config, "safety_label": label, "seconds": seconds,
                "citations": ["X"] * cites, "passages": [{}] * passages, "error": error}

    def test_summary_links_results_to_research_questions(self):
        results = [self.res("ollama", "A", "CAUTION"), self.res("ollama", "B", "UNSAFE", 3, 5),
                   self.res("ollama", "C", "UNSAFE", 2, 5)]
        text = "\n".join(cr.summarize(results, ["ollama"], {"match_ratio": 0.96}))
        self.assertIn("A→B", text)
        self.assertIn("CAUTION → UNSAFE", text)
        self.assertIn("B→C", text)
        self.assertIn("96.0%", text)

    def test_summary_cross_provider_and_errors(self):
        results = [self.res("gemini", "A", "SAFE"), self.res("ollama", "A", "CAUTION"),
                   self.res("gemini", "B", "SAFE", error="hết hạn mức")]
        text = "\n".join(cr.summarize(results, ["gemini", "ollama"]))
        self.assertIn("KHÁC nhau", text)
        self.assertIn("hết hạn mức", text)

    def test_ingredient_section_ignores_text_outside_ingredient_list(self):
        ref = "Thành phần: Nước, đường"
        ocr = "Bánh xốp Jumbo 100g. Thành phần: Nước, dường. Hạn sử dụng 12 tháng kể từ ngày sản xuất, bảo quản nơi khô ráo"
        section = cr.ingredient_section(ocr, ref)
        self.assertTrue(section.startswith("Thành phần"))
        self.assertLessEqual(len(section), int(len(ref) * 1.3) + 20)
        self.assertNotIn("Bánh xốp", section)
        self.assertEqual(cr.match_ratio(ref, ref), 1.0)
        self.assertLess(cr.match_ratio(ref, section), 1.0)
        self.assertGreater(cr.match_ratio(ref, section), 0.85)
        self.assertEqual(cr.match_ratio("", "x"), 0.0)

    def test_ingredient_section_picks_the_ingredient_list_not_the_nutrition_header(self):
        ref = "Thành phần: Bột mì (34%), Đường, Muối"
        ocr = ("Bánh quy 100g. Thành phần dinh dưỡng trên một phần ăn: Năng lượng 70 kcal; Chất đạm 0 g. "
               "Thành phần: Bột mì (34%), Dường, Muối. Hạn sử dụng 12 tháng")
        section = cr.ingredient_section(ocr, ref)
        self.assertTrue(section.startswith("Thành phần: Bột mì"))
        self.assertGreater(cr.match_ratio(ref, section), 0.9)

    def test_diff_html_marks_errors_and_escapes(self):
        out = cr.diff_html("chất điều vị <621>", "chat điều vị <621>")
        self.assertIn("background:#fecaca", out)
        self.assertIn("&lt;621&gt;", out)
        self.assertNotIn("<621>", out)
        self.assertEqual(cr.diff_html("abc", "abc"), "abc")


if __name__ == "__main__":
    unittest.main()
