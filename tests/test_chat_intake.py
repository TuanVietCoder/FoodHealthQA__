"""Unit tests for modules/chat_intake.py (giao diện hỏi đáp dạng chat)."""

import io
import unittest

from PIL import Image

from modules import chat_intake as ci
from modules import foodon as m_foodon


def _png(size=(400, 300), noise=True):
    img = Image.new("RGB", size, "white")
    if noise:
        for x in range(0, size[0], 7):
            for y in range(0, size[1], 5):
                img.putpixel((x, y), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestChatIntake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = m_foodon.load_foodon_entities("data")

    def parse(self, text):
        return ci.parse_message(text, self.cat)

    def test_full_message_split_into_question_and_profile(self):
        r = self.parse("Tôi 35 tuổi, bị tăng huyết áp, dị ứng tôm và đậu phộng. Sản phẩm này có an toàn với tôi không?")
        self.assertEqual(r["kind"], "message")
        self.assertEqual(r["question"], "Sản phẩm này có an toàn với tôi không?")
        p, _ = ci.merge_profile(ci.empty_profile(), r["profile"]["update"])
        self.assertEqual(p["age"], 35)
        self.assertIn("hypertension", p["conditions"])
        self.assertEqual({a["key"] for a in p["allergens"]}, {"shellfish", "peanut"})

    def test_no_diacritics_input(self):
        r = self.parse("toi bi tieu duong, di ung tom, cua. san pham nay an duoc khong")
        self.assertEqual(r["question"], "san pham nay an duoc khong")
        p, _ = ci.merge_profile(ci.empty_profile(), r["profile"]["update"])
        self.assertIn("diabetes", p["conditions"])
        self.assertEqual(len(p["allergens"]), 2)

    def test_gibberish_rejected(self):
        for t in ["asdfgh jkl", "jjjjjjj", "???!!!", "xzqv brrrr"]:
            r = self.parse(t)
            self.assertEqual(r["kind"], "invalid", t)
            self.assertTrue(r["errors"], t)

    def test_off_topic_rejected(self):
        for t in ["Thời tiết hôm nay thế nào?", "1+1 bằng mấy?"]:
            r = self.parse(t)
            self.assertEqual(r["kind"], "invalid", t)
            self.assertEqual(r["question"], "")

    def test_contradictions_rejected(self):
        self.assertEqual(self.parse("Tôi là nam, đang mang thai")["kind"], "invalid")
        self.assertEqual(self.parse("tôi 150 tuổi")["kind"], "invalid")
        self.assertEqual(self.parse("Tôi dị ứng asdkjqwz")["kind"], "invalid")

    def test_allergy_question_is_not_a_declaration(self):
        r = self.parse("Sản phẩm này có gây dị ứng không?")
        self.assertEqual(r["question"], "Sản phẩm này có gây dị ứng không?")
        self.assertNotIn("allergens_add", r["profile"]["update"])

    def test_symptom_as_allergen_is_warned_not_added(self):
        r = self.parse("Tôi dị ứng đau đầu")
        self.assertTrue(r["warnings"])
        self.assertNotIn("allergens_add", r["profile"]["update"])

    def test_year_is_not_gender(self):
        r = self.parse("Năm nay tôi 30 tuổi")
        self.assertNotIn("gender", r["profile"]["update"])
        self.assertEqual(r["profile"]["update"]["age"], 30)

    def test_commands(self):
        self.assertEqual(self.parse("làm mới")["kind"], "reset")
        self.assertEqual(self.parse("xóa hồ sơ")["kind"], "clear_profile")
        self.assertEqual(self.parse("xin chào")["kind"], "greeting")

    def test_next_action(self):
        self.assertEqual(ci.next_action(True, "Có chứa dị nguyên không?"), "run")
        self.assertEqual(ci.next_action(False, "Có chứa dị nguyên không?"), "need_image")
        self.assertEqual(ci.next_action(True, ""), "need_question")
        q = "Tôi có nên ngừng uống thuốc huyết áp không?"
        self.assertEqual(ci.next_action(False, q, ci.abstention.detect_out_of_scope(q)), "out_of_scope_no_image")
        self.assertIn("VƯỢT PHẠM VI", ci.out_of_scope_reply(q))

    def test_child_age_adds_child_condition(self):
        p, _ = ci.merge_profile(ci.empty_profile(), self.parse("con tôi 4 tuổi")["profile"]["update"])
        self.assertIn("children_adhd", p["conditions"])
        persona = ci.profile_to_persona(p)
        self.assertEqual(persona["demographics"]["age"], 4)

    def test_undeclared_profile_is_healthy_persona(self):
        persona = ci.profile_to_persona(ci.empty_profile())
        self.assertEqual(persona["conditions"], ["healthy"])
        self.assertFalse(persona["is_declared"])

    def test_validate_image(self):
        img, err = ci.validate_image(_png(), "ok.png")
        self.assertIsNotNone(img)
        self.assertIsNone(err)
        self.assertIsNotNone(ci.validate_image(b"not an image", "x.jpg")[1])
        self.assertIsNotNone(ci.validate_image(_png((60, 60)), "tiny.png")[1])
        self.assertIsNotNone(ci.validate_image(_png(noise=False), "blank.png")[1])
        self.assertIsNotNone(ci.validate_image(b"", "empty.png")[1])


if __name__ == "__main__":
    unittest.main()
