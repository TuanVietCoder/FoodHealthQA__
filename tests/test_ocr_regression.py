"""Hồi quy chất lượng OCR trên ẢNH THẬT: hậu xử lý → tách thành phần → dò mã/dị nguyên được tính lại từ chữ OCR thô
của các ảnh trong bộ đánh giá và so với mốc chuẩn (data/ocr_regression_baseline.json).

Báo lỗi khi một chỉ số tụt quá mức cho phép — ví dụ một thay đổi ở từ điển sửa chính tả hay bộ tách thành phần làm
chữ tệ hơn, sinh dị nguyên giả, hoặc làm cả phần hướng dẫn sử dụng bị nuốt vào một cụm thành phần.
Bỏ qua (skip) khi máy chưa có chữ OCR của các ảnh: chạy calibrate_confidence.py --stage ocr để tạo.
Cập nhật mốc chỉ khi thay đổi đã được duyệt: python scripts_research/ocr_regression.py --update-baseline
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts_research"))

import ocr_regression as reg  # noqa: E402


@unittest.skipUnless(reg.CACHE.is_file() and reg.BASELINE.is_file(), "chưa có chữ OCR của ảnh thật hoặc mốc chuẩn")
class OCRRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads(reg.BASELINE.read_text(encoding="utf-8"))
        images = reg.load_images()
        if len(images) < 0.9 * cls.baseline.get("images", 0):
            raise unittest.SkipTest(f"bộ đệm OCR chỉ có {len(images)}/{cls.baseline.get('images')} ảnh")
        cls.summary, cls.rows = reg.evaluate(images)

    def test_no_metric_regresses_beyond_tolerance(self):
        problems = reg.compare(self.summary, self.baseline)
        self.assertEqual(problems, [], "Chất lượng OCR tụt so với mốc chuẩn:\n" + "\n".join(problems))

    def test_postprocessing_does_not_make_text_much_worse_than_raw_ocr(self):
        # hậu xử lý không được làm chữ tệ hơn chữ OCR thô quá 1 điểm phần trăm (từ điển cũ: −5,1 điểm)
        self.assertGreaterEqual(self.summary["postprocess_gain"], -0.01)


if __name__ == "__main__":
    unittest.main()
