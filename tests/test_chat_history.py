"""Lưu / mở lại lịch sử hỏi đáp (JSON, không đưa lên git)."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import chat_history as ch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FakeImage:
    """Giả ảnh PIL (module bắt đầu bằng 'PIL') — không được ghi ra JSON."""


FakeImage.__module__ = "PIL.Image"


def sample_chat():
    return [
        {"role": "user", "text": "Sản phẩm này có an toàn cho tôi không?", "thumbs": [b"\xff\xd8jpeg-bytes", None]},
        {"role": "assistant", "blocks": [("success", "Đã nhận ảnh #1")], "choice": False,
         "result": {"image_thumb": b"\x00\x01\x02", "evaluation": {"status": "CAUTION", "s_total": 0.82, "alerts": [{"additive": "INS 211"}]},
                    "linked_results": [{"matched_entity": {"ins_code": "INS 211"}}], "avg_ocr_conf": 0.9}},
    ]


def sample_ctx():
    return {"image": FakeImage(), "annotated_images": {1: FakeImage()}, "ocr_result": {"full_text": "Thành phần: INS 211", "lines": []},
            "mentions": [{"raw_mention": "INS 211"}], "profile": {"conditions": ["hypertension"], "declared": True},
            "linked_cache": (1, [{"raw_mention": "INS 211"}]), "image_no": 1, "display_name": "Bánh quy"}


class ChatHistoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip_keeps_messages_bytes_and_context(self):
        cid = ch.new_id()
        ch.save_conversation(cid, sample_chat(), sample_ctx(), self.dir)
        chat, ctx = ch.load_conversation(cid, self.dir)
        self.assertEqual(chat[0]["text"], "Sản phẩm này có an toàn cho tôi không?")
        self.assertEqual(chat[0]["thumbs"], [b"\xff\xd8jpeg-bytes", None])          # bytes trở lại đúng bytes
        self.assertEqual(chat[1]["result"]["image_thumb"], b"\x00\x01\x02")
        self.assertEqual(chat[1]["result"]["evaluation"]["status"], "CAUTION")
        for kind, body in chat[1]["blocks"]:                                         # tuple → list vẫn bung được
            self.assertEqual((kind, body), ("success", "Đã nhận ảnh #1"))
        self.assertEqual(ctx["mentions"], [{"raw_mention": "INS 211"}])
        self.assertEqual(ctx["profile"]["conditions"], ["hypertension"])
        self.assertEqual(ctx["linked_cache"][0], 1)                                  # còn dùng được như cache[0]/cache[1]
        self.assertNotIn("image", ctx)                                               # ảnh PIL gốc không lưu
        self.assertNotIn("annotated_images", ctx)

    def test_listing_is_newest_first_with_title_and_message_count(self):
        a, b = "20261124-100000-aaaaaa", "20261124-110000-bbbbbb"
        ch.save_conversation(a, sample_chat(), sample_ctx(), self.dir)
        ch.save_conversation(b, [{"role": "user", "text": "Có chứa dị nguyên nào không?", "thumbs": []}], {}, self.dir)
        items = ch.list_conversations(directory=self.dir)
        self.assertEqual({i["id"] for i in items}, {a, b})
        by_id = {i["id"]: i for i in items}
        self.assertEqual(by_id[b]["title"], "Có chứa dị nguyên nào không?")
        self.assertEqual(by_id[a]["n_messages"], 2)
        # cập nhật lại cuộc cũ thì nó lên đầu danh sách và giữ ngày tạo
        created = ch.json.loads((self.dir / f"{a}.json").read_text(encoding="utf-8"))["created"]
        ch.save_conversation(a, sample_chat() + [{"role": "user", "text": "hỏi thêm", "thumbs": []}], sample_ctx(), self.dir)
        data = ch.json.loads((self.dir / f"{a}.json").read_text(encoding="utf-8"))
        self.assertEqual(data["created"], created)
        self.assertEqual(len(data["messages"]), 3)

    def test_title_skips_attachment_only_messages_and_truncates(self):
        chat = [{"role": "user", "text": "📎 anh.jpg", "thumbs": []},
                {"role": "user", "text": "x" * 100, "thumbs": []}]
        title = ch.make_title(chat)
        self.assertEqual(len(title), 45)
        self.assertTrue(title.endswith("…"))
        self.assertEqual(ch.make_title([], {"display_name": "Sữa chua"}), "Sữa chua")
        self.assertEqual(ch.make_title([]), "Cuộc trò chuyện mới")

    def test_ids_are_validated_so_paths_cannot_escape_the_folder(self):
        for bad in ("../secret", "a/b", "", None, "20261124-100000-ZZZZZZ", "abc"):
            self.assertFalse(ch.is_valid_id(bad), bad)
            self.assertIsNone(ch.load_conversation(bad, self.dir))
            self.assertFalse(ch.delete_conversation(bad, self.dir))
            with self.assertRaises(ValueError):
                ch.save_conversation(bad, [], {}, self.dir)

    def test_corrupt_or_missing_files_are_ignored_not_fatal(self):
        cid = "20261124-100000-cccccc"
        (self.dir / f"{cid}.json").write_text("{không phải json", encoding="utf-8")
        self.assertIsNone(ch.load_conversation(cid, self.dir))
        self.assertEqual(ch.list_conversations(directory=self.dir), [])
        self.assertIsNone(ch.load_conversation("20261124-100000-dddddd", self.dir))

    def test_delete_removes_the_file(self):
        cid = ch.new_id()
        ch.save_conversation(cid, sample_chat(), {}, self.dir)
        self.assertTrue(ch.delete_conversation(cid, self.dir))
        self.assertEqual(ch.list_conversations(directory=self.dir), [])
        self.assertFalse(ch.delete_conversation(cid, self.dir))

    def test_unserialisable_values_never_crash_the_save(self):
        class Odd:
            def __str__(self):
                return "odd"
        cid = ch.new_id()
        ch.save_conversation(cid, [{"role": "user", "text": "hi", "thumbs": [], "x": Odd(), "f": float("nan")}], {"s": {1, 2}}, self.dir)
        chat, ctx = ch.load_conversation(cid, self.dir)
        self.assertEqual(chat[0]["x"], "odd")
        self.assertIsNone(chat[0]["f"])
        self.assertEqual(sorted(ctx["s"]), [1, 2])

    def test_history_folder_is_ignored_by_git(self):
        result = subprocess.run(["git", "check-ignore", "data/chat_history/x.json"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, "data/chat_history/ phải nằm trong .gitignore")


if __name__ == "__main__":
    unittest.main()
