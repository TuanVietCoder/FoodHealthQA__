"""Các nguồn lâm sàng được nhắc trong tài liệu nhưng CHƯA tích hợp: kiểm bằng dữ liệu rằng chat và bộ QA không dùng chúng.
Nếu sau này tích hợp một nguồn, hãy xoá nó khỏi danh sách này và sửa tài liệu (QUY_TRINH_3_CAU_HINH_HE_THONG.md, bảng nguồn)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_research"))

import audit_source_usage as audit  # noqa: E402

NOT_INTEGRATED = ("KDIGO 2024", "ACR 2020 (gút)", "ACOG PB 190", "ADA Standards of Care", "WHO Fact Sheet 394")
WHERE = ("KB", "catalog", "mã công nhận", "đoạn", "bộ QA", "mã chạy")


class NotIntegratedSourcesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = audit.usage()

    def test_listed_sources_are_absent_from_everything_the_chat_and_the_benchmark_read(self):
        for name in NOT_INTEGRATED:
            used_in = {c: self.table[name][c] for c in WHERE if self.table[name][c]}
            self.assertEqual(used_in, {}, name)

    def test_the_audit_does_detect_a_source_that_is_used(self):
        row = self.table["KDOQI 2020"]
        self.assertGreater(row["KB"], 0)  # KDOQI được dùng làm bằng chứng bổ sung cho phosphat + bệnh thận mạn


if __name__ == "__main__":
    unittest.main()
