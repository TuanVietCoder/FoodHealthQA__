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


if __name__ == "__main__":
    unittest.main()
