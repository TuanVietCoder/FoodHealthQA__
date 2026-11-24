"""harden_knowledge_base.py không được âm thầm ghi đè dữ liệu đã kiểm nguồn của additives_standard.json."""
import contextlib
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_research"))

import harden_knowledge_base as harden  # noqa: E402

KB_PATH = ROOT / "knowledge_base" / "additives_standard.json"


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HardenGuardTests(unittest.TestCase):
    def test_current_kb_is_recognised_as_verified_data(self):
        reasons = harden.verified_data_reasons(json.loads(KB_PATH.read_text(encoding="utf-8")))
        self.assertTrue(any("adi_assessment" in r for r in reasons))
        self.assertTrue(any("health_considerations" in r for r in reasons))

    def test_a_pending_only_kb_is_not_blocked(self):
        pending = [{"ins_code": "INS 1", "health_considerations": [], "adi_assessment": {"verification_status": "pending"},
                    "codex_claim": {"verification_status": "pending_gsfa"}, "regulatory_claim": {"verification_status": "pending"}}]
        self.assertEqual(harden.verified_data_reasons(pending), [])

    def test_running_without_force_refuses_and_touches_nothing(self):
        before = _digest(KB_PATH)
        err = io.StringIO()
        with mock.patch.object(harden, "enrich_additives", side_effect=AssertionError("không được ghi đè")) as enrich, \
                contextlib.redirect_stderr(err):
            code = harden.main([])
        self.assertEqual(code, 2)
        enrich.assert_not_called()
        self.assertIn("--force", err.getvalue())
        self.assertEqual(_digest(KB_PATH), before)

    def test_force_flag_is_the_only_way_through(self):
        with mock.patch.object(harden, "enrich_additives", return_value=[]) as enrich, \
                mock.patch.object(harden, "enrich_allergens", return_value=[]), \
                mock.patch.object(harden, "write_ingredient_catalog", return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()):
            harden.main(["--force"])
        enrich.assert_called_once()


if __name__ == "__main__":
    unittest.main()
