import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.ollama_embedder import OllamaEmbedder  # noqa: E402
from modules.passage_retriever import PassageRetriever  # noqa: E402

PASSAGES = [
    {"pid": "P0", "kind": "additive_identity", "text": "aspartame sweetener ADI 40", "citation_id": "JECFA_EVALUATIONS",
     "locator": "JECFA#951", "ins_codes": ["951"]},
    {"pid": "P1", "kind": "allergen_group", "text": "milk allergen declaration", "citation_id": "CODEX_CXS_1_1985",
     "locator": "CXS1#4.2.1.4", "ins_codes": []},
    {"pid": "P2", "kind": "external_claim", "text": "pregnancy guidance for non-sugar sweeteners",
     "citation_id": "WHO_NSS_GUIDELINE_2023", "locator": "WHO#Pregnancy", "ins_codes": []},
]


def fake_encoder(texts):
    """Từ khóa nghĩa: 'bà bầu' ~ pregnancy; 'sữa' ~ milk. Trả vector one-hot chuẩn hóa."""
    vocab = {"pregnancy": 0, "bà bầu": 0, "milk": 1, "sữa": 1, "aspartame": 2, "sweetener": 2}
    out = np.zeros((len(texts), 3), dtype=np.float32)
    for r, t in enumerate(texts):
        for w, c in vocab.items():
            if w in t.lower():
                out[r, c] = 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return out / norms


class DenseModesTests(unittest.TestCase):
    def test_dense_mode_requires_encoder(self):
        sparse_only = PassageRetriever(PASSAGES)
        self.assertFalse(sparse_only.has_dense)
        with self.assertRaises(ValueError):
            sparse_only.retrieve("bà bầu", mode="dense")
        with self.assertRaises(ValueError):
            sparse_only.retrieve("bà bầu", mode="hybrid_dense")
        self.assertTrue(sparse_only.retrieve("aspartame", mode="hybrid"))  # sparse vẫn chạy

    def test_dense_bridges_vocabulary_gap_that_sparse_misses(self):
        retr = PassageRetriever(PASSAGES, dense_encoder=fake_encoder)
        query = "bà bầu dùng được không"  # không có từ nào trùng với đoạn tiếng Anh
        self.assertEqual(retr.retrieve(query, top_k=3, mode="bm25"), [])
        top = retr.retrieve(query, top_k=1, mode="dense")
        self.assertEqual(top[0]["locator"], "WHO#Pregnancy")
        hybrid = retr.retrieve(query, top_k=1, mode="hybrid_dense")
        self.assertEqual(hybrid[0]["locator"], "WHO#Pregnancy")

    def test_dense_exact_keeps_ins_slot(self):
        retr = PassageRetriever(PASSAGES, dense_encoder=fake_encoder)
        hits = retr.retrieve("bà bầu INS 951", top_k=2, mode="hybrid_dense_exact", ins_codes=["951"])
        locators = [h["locator"] for h in hits]
        self.assertIn("JECFA#951", locators)
        self.assertIn("WHO#Pregnancy", locators)

    def test_all_declared_modes_work_with_encoder(self):
        retr = PassageRetriever(PASSAGES, dense_encoder=fake_encoder)
        for mode in PassageRetriever.MODES:
            hits = retr.retrieve("sữa milk aspartame", top_k=3, mode=mode)
            self.assertLessEqual(len(hits), 3, mode)
            self.assertEqual([h["rank"] for h in hits], list(range(1, len(hits) + 1)), mode)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class OllamaEmbedderTests(unittest.TestCase):
    def test_batches_normalises_and_caches_on_disk(self):
        calls = []

        def fake_urlopen(req, timeout=None):
            body = json.loads(req.data)
            calls.append(body["input"])
            return _FakeResponse({"embeddings": [[3.0, 4.0, 0.0] for _ in body["input"]]})

        with tempfile.TemporaryDirectory() as tmp, mock.patch("urllib.request.urlopen", fake_urlopen):
            enc = OllamaEmbedder("bge-m3", batch_size=2, cache_dir=Path(tmp))
            vecs = enc(["a", "b", "c", "a"])  # 'a' lặp: chỉ embed 3 văn bản khác nhau
            self.assertEqual(sum(len(c) for c in calls), 3)
            self.assertEqual([len(c) for c in calls], [2, 1])  # chia lô theo batch_size
            np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)
            enc.save_cache()

            calls.clear()
            again = OllamaEmbedder("bge-m3", cache_dir=Path(tmp))  # đọc lại từ đĩa
            again(["a", "b", "c"])
            self.assertEqual(calls, [])  # trúng cache, không gọi Ollama

    def test_unreachable_ollama_raises(self):
        import urllib.error
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with self.assertRaises(urllib.error.URLError):
                OllamaEmbedder("bge-m3", cache_dir=Path(tmp))(["x"])


if __name__ == "__main__":
    unittest.main()
