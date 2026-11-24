import unittest

from modules.passage_retriever import (
    PassageRetriever,
    build_corpus,
    extract_ins_codes,
    normalize_ins,
    render_passages,
)


class PassageRetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.passages = build_corpus()
        cls.retriever = PassageRetriever(cls.passages)

    def test_normalize_ins_variants(self):
        for raw, want in [("INS330", "330"), ("INS 331(III)", "331iii"), ("E331iii", "331iii"),
                          ("322(i)", "322i"), ("621", "621")]:
            self.assertEqual(normalize_ins(raw), want)

    def test_extract_ins_from_text(self):
        self.assertEqual(extract_ins_codes("chứa INS330, INS331(III) và E211"), ["330", "331iii", "211"])
        self.assertEqual(extract_ins_codes("không có phụ gia"), [])

    def test_corpus_passages_have_citation_and_locator(self):
        self.assertGreater(len(self.passages), 500)
        for p in self.passages:
            self.assertTrue(p["citation_id"])
            self.assertTrue(p["locator"])
            self.assertEqual(len({p["pid"] for p in self.passages}), len(self.passages))

    def test_exact_ins_lookup_returns_matching_passage(self):
        hits = self.retriever.retrieve("INS 621 có an toàn không?", top_k=5, mode="hybrid_exact",
                                       ins_codes=["621"])
        self.assertTrue(any("621" in p["ins_codes"] for p in hits))

    def test_exact_mode_keeps_topical_slots(self):
        hits = self.retriever.retrieve("sản phẩm có chứa sữa gây dị ứng không", top_k=4,
                                       mode="hybrid_exact", ins_codes=["621"])
        self.assertEqual(len(hits), 4)
        self.assertLessEqual(sum("621" in p["ins_codes"] for p in hits), 2)

    def test_all_modes_return_ranked_results(self):
        for mode in PassageRetriever.MODES:
            if mode in PassageRetriever.DENSE_MODES:  # cần encoder: xem test_dense_retrieval.py
                continue
            hits = self.retriever.retrieve("allergen peanut", top_k=3, mode=mode)
            self.assertLessEqual(len(hits), 3)
            self.assertEqual([h["rank"] for h in hits], list(range(1, len(hits) + 1)))

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve("x", mode="dense")

    def test_render_passages_is_citable(self):
        hits = self.retriever.retrieve("tartrazine", top_k=2, mode="hybrid")
        text = render_passages(hits)
        self.assertIn(f"[{hits[0]['citation_id']}]", text)
        self.assertEqual(render_passages([]), "")

    def test_jecfa_passages_are_only_built_from_verified_records_with_adi(self):
        """Bản ghi JECFA chưa tìm được trang (passed=false) không được thành đoạn bằng chứng rỗng "…gần nhất ?, ADI ?"."""
        jecfa = [p for p in self.passages if p["kind"] == "jecfa_adi"]
        self.assertTrue(jecfa)
        for p in jecfa:
            self.assertNotIn("?", p["text"], p["locator"])
            self.assertIn("ADI ", p["text"], p["locator"])

    def test_tocopherol_307_has_no_jecfa_passage(self):
        """INS 307a/b/c: JECFA không có trang tương ứng → không có đoạn ADI của JECFA cho các mã này."""
        for p in self.passages:
            if p["kind"] == "jecfa_adi":
                self.assertFalse({"307a", "307b", "307c"} & set(p.get("ins_codes", [])), p["locator"])

    def test_unrelated_jecfa_passages_are_dropped_before_answer_generation(self):
        """Truy xuất lấp chỗ trống bằng đoạn JECFA của chất khác → phải bị lọc, kẻo JECFA thành nguồn hợp lệ cho 307c."""
        from modules.pipeline import drop_unrelated_jecfa_passages
        hits = self.retriever.retrieve("INS 307c có an toàn không", top_k=5, mode="hybrid_exact", ins_codes=["307c"])
        self.assertIn("JECFA_EVALUATIONS", [h["citation_id"] for h in hits])  # tiền đề: đoạn lạc có thật
        linked = [{"matched_entity": {"ins_code": "INS 307c"}}]
        kept = drop_unrelated_jecfa_passages(hits, linked)
        self.assertNotIn("JECFA_EVALUATIONS", [p["citation_id"] for p in kept])
        self.assertIn("CODEX_STAN_192_1995", [p["citation_id"] for p in kept])  # nguồn khác giữ nguyên
        # chất CÓ trên nhãn và có dữ liệu JECFA thì đoạn JECFA được giữ
        own = [p for p in self.passages if p["kind"] == "jecfa_adi" and "300" in p["ins_codes"]]
        self.assertTrue(own)
        self.assertEqual(drop_unrelated_jecfa_passages(own, [{"matched_entity": {"ins_code": "INS 300"}}]), own)


if __name__ == "__main__":
    unittest.main()
