import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import pipeline  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class LLMProviderTests(unittest.TestCase):
    def setUp(self):
        pipeline.set_llm_provider(None)
        self._env = dict(os.environ)

    def tearDown(self):
        pipeline.set_llm_provider(None)
        os.environ.clear()
        os.environ.update(self._env)

    def test_default_provider_is_gemini(self):
        os.environ.pop("LLM_PROVIDER", None)
        self.assertEqual(pipeline.get_llm_provider(), "gemini")

    def test_env_and_runtime_override(self):
        os.environ["LLM_PROVIDER"] = "ollama"
        self.assertEqual(pipeline.get_llm_provider(), "ollama")
        pipeline.set_llm_provider("gemini")
        self.assertEqual(pipeline.get_llm_provider(), "gemini")

    def test_invalid_provider_rejected(self):
        with self.assertRaises(ValueError):
            pipeline.set_llm_provider("openai")

    def test_ollama_routes_to_local_chat_api_without_gemini_key(self):
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)
        pipeline.set_llm_provider("ollama")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data)
            return _FakeResponse({"message": {"content": "Câu trả lời thử nghiệm đủ dài. SAFETY_LABEL: SAFE"}})

        with mock.patch("urllib.request.urlopen", fake_urlopen), \
                mock.patch.object(pipeline, "_call_gemini_with_fallback") as gemini:
            out = pipeline.generate_llm_only_answer("Nước, muối", {"conditions": [], "allergies": []}, "An toàn không?")
        gemini.assert_not_called()
        self.assertIn("SAFETY_LABEL", out)
        self.assertTrue(captured["url"].endswith("/api/chat"))
        self.assertEqual(captured["body"]["model"], pipeline.get_ollama_model())
        self.assertFalse(captured["body"]["stream"])

    def test_gemini_provider_still_requires_key(self):
        pipeline.set_llm_provider("gemini")
        with mock.patch.object(pipeline.llm_keys, "load_keys", return_value=[]):
            self.assertIsNone(
                pipeline.generate_llm_only_answer("x", {"conditions": [], "allergies": []}, "q")
            )

    def test_ollama_failure_returns_none(self):
        pipeline.set_llm_provider("ollama")
        with mock.patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            self.assertIsNone(pipeline._call_ollama("xin chào"))


if __name__ == "__main__":
    unittest.main()
