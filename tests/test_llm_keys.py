import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import llm_keys  # noqa: E402
from modules import pipeline  # noqa: E402
from scripts_research import generate_predictions as gp  # noqa: E402


class ErrorClassificationTests(unittest.TestCase):
    def test_daily_rate_and_other(self):
        daily = Exception("429 RESOURCE_EXHAUSTED: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        rate = Exception("429 RESOURCE_EXHAUSTED: GenerateRequestsPerMinutePerProjectPerModel retry in 12.5s")
        other = Exception("503 UNAVAILABLE: model overloaded")
        self.assertEqual(llm_keys.classify_error(daily), "daily")
        self.assertEqual(llm_keys.classify_error(rate), "rate")
        self.assertEqual(llm_keys.classify_error(other), "other")
        self.assertAlmostEqual(llm_keys.retry_delay_seconds(rate), 13.5)


class KeyLoadingAndPoolTests(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        for var in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_DAILY_LIMIT_PER_KEY"):
            os.environ.pop(var, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        llm_keys.reset_pool()
        self.addCleanup(llm_keys.reset_pool)
        self.addCleanup(llm_keys.set_strict, False)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_load_keys_merges_sources_without_duplicates(self):
        keys_file = Path(self.tmp.name) / "keys.txt"
        keys_file.write_text("# comment\nkeyB  # trailing\n\nkeyC\nkeyA\n", encoding="utf-8")
        os.environ["GEMINI_API_KEYS"] = "keyA, keyB;keyX"
        os.environ["GEMINI_API_KEY"] = "keyC"
        with mock.patch.dict(sys.modules, {"dotenv": None}):
            keys = llm_keys.load_keys(keys_file)
        self.assertEqual(keys, ["keyA", "keyB", "keyX", "keyC"])

    def test_pool_is_sticky_and_skips_exhausted(self):
        pool = llm_keys.KeyPool(["k1", "k2"], usage_file=Path(self.tmp.name) / "u.json")
        self.assertEqual(pool.acquire(), "k1")
        self.assertEqual(pool.acquire(), "k1")
        pool.mark_exhausted("k1")
        self.assertEqual(pool.acquire(), "k2")
        pool.mark_exhausted("k2")
        self.assertIsNone(pool.acquire())

    def test_daily_limit_and_usage_persistence_store_no_raw_key(self):
        usage = Path(self.tmp.name) / "u.json"
        pool = llm_keys.KeyPool(["secret-key-1", "secret-key-2"], daily_limit=2, usage_file=usage)
        pool.record_request("secret-key-1")
        pool.record_request("secret-key-1")
        self.assertEqual(pool.acquire(), "secret-key-2")
        reloaded = llm_keys.KeyPool(["secret-key-1", "secret-key-2"], daily_limit=2, usage_file=usage)
        self.assertEqual(reloaded.used_today("secret-key-1"), 2)
        self.assertNotIn("secret-key", usage.read_text(encoding="utf-8"))


def _fake_genai(behaviour):
    """behaviour(key, model) -> text or raises. Returns a fake `google` package."""
    class Models:
        def __init__(self, key):
            self.key = key

        def generate_content(self, model, contents, config):
            return types.SimpleNamespace(text=behaviour(self.key, model))

    class Client:
        def __init__(self, api_key):
            self.models = Models(api_key)

    genai = types.SimpleNamespace(Client=Client)
    google = types.SimpleNamespace(genai=genai)
    return {"google": google, "google.genai": genai}


class GeminiRotationTests(unittest.TestCase):
    def setUp(self):
        llm_keys.reset_pool()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(llm_keys.reset_pool)
        self.addCleanup(llm_keys.set_strict, False)
        self.addCleanup(pipeline.set_llm_provider, None)

    def _install_pool(self, keys):
        llm_keys._POOL = llm_keys.KeyPool(keys, usage_file=Path(self.tmp.name) / "u.json")

    def test_rotates_to_next_key_when_daily_quota_hit(self):
        calls = []

        def behaviour(key, model):
            calls.append(key)
            if key == "k1":
                raise Exception("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel")
            return "Câu trả lời hợp lệ từ khóa thứ hai. SAFETY_LABEL: SAFE"

        self._install_pool(["k1", "k2"])
        with mock.patch.dict(sys.modules, _fake_genai(behaviour)):
            out = pipeline._call_gemini_with_fallback("prompt")
        self.assertIn("khóa thứ hai", out)
        self.assertIn("k2", calls)
        self.assertEqual(llm_keys.get_pool().acquire(), "k2")  # k1 đã bị đánh dấu hết hạn mức

    def test_all_keys_exhausted_raises_only_in_strict_mode(self):
        def behaviour(key, model):
            raise Exception("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel")

        self._install_pool(["k1"])
        with mock.patch.dict(sys.modules, _fake_genai(behaviour)):
            self.assertIsNone(pipeline._call_gemini_with_fallback("p"))  # không strict: trả None như trước
        self._install_pool(["k1"])
        llm_keys.set_strict(True)
        with mock.patch.dict(sys.modules, _fake_genai(behaviour)):
            with self.assertRaises(llm_keys.LLMQuotaExhausted):
                pipeline._call_gemini_with_fallback("p")

    def test_non_quota_error_does_not_burn_other_keys(self):
        calls = []

        def behaviour(key, model):
            calls.append(key)
            raise Exception("503 UNAVAILABLE overloaded")

        self._install_pool(["k1", "k2"])
        with mock.patch.dict(sys.modules, _fake_genai(behaviour)):
            self.assertIsNone(pipeline._call_gemini_with_fallback("p"))
        self.assertEqual(set(calls), {"k1"})

    def test_ollama_unreachable_raises_in_strict_mode(self):
        pipeline.set_llm_provider("ollama")
        llm_keys.set_strict(True)
        import urllib.error
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with self.assertRaises(llm_keys.LLMUnavailable):
                pipeline._call_ollama("xin chào")


class ResumeFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(llm_keys.set_strict, False)
        self.addCleanup(pipeline.set_llm_provider, None)
        self.bench = Path(self.tmp.name) / "bench.jsonl"
        self.bench.write_text(
            "\n".join(json.dumps({"id": f"Q{i}", "question": "q"}) for i in range(1, 6)), encoding="utf-8"
        )
        self.out = Path(self.tmp.name) / "run"

    def _run(self, extra, predict):
        argv = ["generate_predictions.py", "--benchmark", str(self.bench), "--output-dir", str(self.out),
                "--llm-provider", "ollama", "--configs", "A", "B", *extra]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(gp, "_predict_one", side_effect=predict), \
                mock.patch.object(gp, "_load_resources", return_value={}), \
                mock.patch.object(gp, "is_evaluation_eligible", return_value=True), \
                mock.patch.object(gp, "is_paddleocr_available", return_value=True), \
                mock.patch("modules.pipeline.ollama_status", return_value=(True, "ok")):
            gp.main()

    @staticmethod
    def _ok(config, record, *_):
        return {"id": record["id"], "config": config, "answer": "Câu trả lời hợp lệ đủ dài", "error": None,
                "safety_label": "SAFE"}

    def test_stops_cleanly_on_quota_then_resumes_and_retries_failures(self):
        calls = {"n": 0}

        def first_day(config, record, *_):
            calls["n"] += 1
            if calls["n"] == 5:
                raise llm_keys.LLMQuotaExhausted("hết hạn mức")
            if calls["n"] == 3:  # một bản ghi lỗi thường: phải được chạy lại ở lần resume
                return {"id": record["id"], "config": config, "answer": "", "error": "llm_generation_failed",
                        "safety_label": "ABSTAIN"}
            return self._ok(config, record)

        self._run([], first_day)
        status_a = (self.out / "predictions_config_a.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(calls["n"], 5)
        self.assertLess(len(status_a), 5)  # dừng giữa chừng, không ghi bản ghi cho lần gọi bị hết hạn mức

        # Chạy lại không --resume khi đã có dữ liệu: bị chặn để không ghi đè
        with self.assertRaises(SystemExit):
            self._run([], lambda c, r, *_: self._ok(c, r))

        self._run(["--resume"], lambda c, r, *_: self._ok(c, r))
        for config in ("a", "b"):
            lines = (self.out / f"predictions_config_{config}.jsonl").read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines]
            self.assertEqual(sorted(r["id"] for r in records), [f"Q{i}" for i in range(1, 6)])  # đủ, không trùng
            self.assertTrue(all(not r["error"] for r in records))
            self.assertTrue(all(r["llm_provider"] == "ollama" for r in records))
        log = (self.out / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(log), 2)
        self.assertIn("LLMQuotaExhausted", json.loads(log[0])["stop_reason"])
        self.assertEqual(json.loads(log[1])["stop_reason"], "completed")

    def test_max_new_limits_a_single_run(self):
        self._run(["--max-new", "3"], lambda c, r, *_: self._ok(c, r))
        total = sum(
            len((self.out / f"predictions_config_{c}.jsonl").read_text(encoding="utf-8").splitlines())
            for c in ("a", "b")
        )
        self.assertEqual(total, 3)


if __name__ == "__main__":
    unittest.main()
