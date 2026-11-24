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
        # chưa khai báo hồ sơ → hỏi hồ sơ, không tự gán mặc định
        self.assertEqual(ci.next_action(True, "Có chứa dị nguyên không?", has_profile=False), "need_profile")
        self.assertEqual(ci.next_action(False, "Có chứa dị nguyên không?", has_profile=False), "need_image_profile")
        q = "Tôi có nên ngừng uống thuốc huyết áp không?"
        self.assertEqual(ci.next_action(False, q, ci.abstention.detect_out_of_scope(q)), "out_of_scope_no_image")
        self.assertIn("VƯỢT PHẠM VI", ci.out_of_scope_reply(q))

    def test_child_age_adds_child_condition(self):
        p, _ = ci.merge_profile(ci.empty_profile(), self.parse("con tôi 4 tuổi")["profile"]["update"])
        self.assertIn("children_adhd", p["conditions"])
        persona = ci.profile_to_persona(p)
        self.assertEqual(persona["demographics"]["age"], 4)

    def test_child_weight_18kg_only_for_ages_3_to_5(self):
        for age, expected in ((2, None), (3, 18.0), (5, 18.0), (6, None), (30, None)):
            p, _ = ci.merge_profile(ci.empty_profile(), {"age": age})
            self.assertEqual(ci.profile_to_persona(p).get("weight_kg"), expected, age)
        p, _ = ci.merge_profile(ci.empty_profile(), {"age": 1, "age_months": 8})
        self.assertNotIn("weight_kg", ci.profile_to_persona(p))

    def test_undeclared_profile_is_healthy_persona(self):
        persona = ci.profile_to_persona(ci.empty_profile())
        self.assertEqual(persona["conditions"], ["healthy", "healthy_adult"])
        self.assertFalse(persona["is_declared"])

    def test_chat_persona_matches_benchmark_archetypes(self):
        """Cùng một người, hai đường dựng persona (chat / hồ sơ mẫu bộ QA) phải ra cùng điều kiện và dị ứng."""
        from modules.pipeline import build_persona_from_benchmark_record as bench
        cases = {
            "healthy_adult": "Tôi 30 tuổi, không có bệnh nền",
            "child_5yo": "Con tôi 5 tuổi",
            "pku_patient": "Tôi 30 tuổi, bị PKU",
            "ckd_stage3": "Tôi 50 tuổi, bị suy thận",
            "diabetes_hypertension": "Tôi 50 tuổi, bị tiểu đường, tăng huyết áp",
            "pregnant_woman": "Tôi 30 tuổi, đang mang thai",
        }
        # khóa chỉ có ở hồ sơ mẫu và không có nơi nào trong code dùng đến
        unused = {"pregnancy_second_trimester", "impaired_renal_filtration", "cardiovascular_risk"}
        for pid, text in cases.items():
            p, _ = ci.merge_profile(ci.empty_profile(), self.parse(text)["profile"]["update"])
            if pid == "healthy_adult":
                p["healthy"] = True
            chat_p = ci.profile_to_persona(p)
            ref = bench({"persona_id": pid})
            # chat có thể thêm khóa alias (vd children, ckd), nhưng không được thiếu khóa nào của hồ sơ mẫu
            self.assertLessEqual(set(ref["conditions"]) - unused, set(chat_p["conditions"]), pid)
            self.assertEqual(set(chat_p["allergies"]), set(ref["allergies"]), pid)

    def test_skip_profile_reply(self):
        for t in ["không có", "bỏ qua", "khỏe mạnh"]:
            self.assertEqual(self.parse(t)["kind"], "skip_profile", t)

    def test_profile_additions_are_merged(self):
        p, _ = ci.merge_profile(ci.empty_profile(), self.parse("Tôi 40 tuổi, bị tiểu đường")["profile"]["update"])
        p, notes = ci.merge_profile(p, self.parse("Tôi dị ứng tôm")["profile"]["update"])
        self.assertEqual(notes, [])
        self.assertIn("diabetes", p["conditions"])
        self.assertEqual(p["age"], 40)
        self.assertEqual(p["allergens"][0]["key"], "shellfish")

    def test_conflicting_profile_replaces_old_one(self):
        p, _ = ci.merge_profile(ci.empty_profile(), self.parse("Tôi 40 tuổi, nam, bị tiểu đường")["profile"]["update"])
        p, notes = ci.merge_profile(p, self.parse("Con tôi 4 tuổi dị ứng sữa, ăn được không?")["profile"]["update"])
        self.assertTrue(notes and "thay mới" in notes[0])
        self.assertEqual(p["age"], 4)
        self.assertNotIn("diabetes", p["conditions"])   # không gộp bệnh của người trước
        self.assertIsNone(p["gender"])
        self.assertEqual([a["key"] for a in p["allergens"]], ["milk"])

    def test_pronoun_after_allergen_list_is_not_an_allergen(self):
        upd = self.parse("Con tôi 4 tuổi dị ứng đậu phộng, bé ăn được không?")["profile"]["update"]
        self.assertEqual([a["key"] for a in upd["allergens_add"]], ["peanut"])

    def test_age_group_counts_as_profile(self):
        r = self.parse("sản phẩm này có an toàn cho người trưởng thành")
        self.assertEqual(r["question"], "sản phẩm này có an toàn cho người trưởng thành")
        p, _ = ci.merge_profile(ci.empty_profile(), r["profile"]["update"])
        self.assertTrue(p["declared"])
        self.assertEqual(p["age_group"], "adult")
        self.assertIn("người trưởng thành", ci.profile_summary(p))
        self.assertEqual(ci.profile_to_persona(p)["demographics"]["age"], 30)
        p2, _ = ci.merge_profile(ci.empty_profile(), self.parse("Bố tôi là người cao tuổi")["profile"]["update"])
        self.assertEqual(ci.profile_to_persona(p2)["demographics"]["age"], 65)
        self.assertEqual(self.parse("người trưởng thành 5 tuổi")["kind"], "invalid")
        # đổi nhóm tuổi → hồ sơ thay mới
        p3, notes = ci.merge_profile(p, self.parse("Còn người cao tuổi thì sao?")["profile"]["update"])
        self.assertEqual(p3["age_group"], "elderly")
        self.assertTrue(notes)

    def test_followup_uses_previous_focus(self):
        last = ["INS 950", "INS 955", "sữa"]
        for q in ["Chất đó có an toàn cho trẻ không?", "Nó có gây hại cho thai phụ không?",
                  "Còn với người tiểu đường thì sao?", "Các chất trên có được phép theo TT24 không?"]:
            q2, terms = ci.resolve_followup(q, last)
            self.assertEqual(terms, last, q)
            self.assertIn("INS 950", q2)

    def test_self_contained_question_not_rewritten(self):
        last = ["INS 950"]
        for q in ["Aspartame (INS 951) có an toàn không?", "Sản phẩm này có chứa dị nguyên nào không?"]:
            q2, terms = ci.resolve_followup(q, last)
            self.assertEqual((q2, terms), (q, []), q)
        self.assertEqual(ci.resolve_followup("Chất đó có an toàn không?", []), ("Chất đó có an toàn không?", []))

    def test_focus_from_turn_uses_class_then_answer(self):
        linked = [{"matched_entity": {"ins_code": "INS 950"}, "category": "sweetener"},
                  {"matched_entity": {"ins_code": "INS 955"}, "category": "sweetener"},
                  {"matched_entity": {"ins_code": "INS 471"}, "category": "emulsifier"}]
        self.assertEqual(ci.focus_from_turn({}, "Có chất tạo ngọt tổng hợp nào không?", linked),
                         ["INS 950", "INS 955"])
        self.assertEqual(ci.focus_from_turn({}, "Sản phẩm có an toàn không?", linked, "Có INS 471 và E330."),
                         ["INS 471", "INS 330"])
        self.assertEqual(ci.focus_from_turn({"entities": {"additives": ["INS 951"]}}, "x", linked), ["INS 951"])

    def test_focus_from_qa(self):
        qa = {"entities": {"additives": ["INS 951"], "allergens": ["milk"]}}
        self.assertEqual(ci.focus_from_qa(qa), ["INS 951", "sữa"])
        self.assertEqual(ci.focus_from_qa(None), [])

    def test_too_long_message_rejected(self):
        r = self.parse("Sản phẩm này có an toàn không? " * 60)
        self.assertEqual(r["kind"], "invalid")
        self.assertIn("quá dài", r["errors"][0])

    def test_use_old_image_command(self):
        for t in ["dùng ảnh cũ", "ảnh cũ", "dùng lại ảnh trước", "dung anh cu"]:
            self.assertEqual(self.parse(t)["kind"], "use_old_image", t)

    def test_guess_product_name(self):
        text = "BÁNH QUY BƠ DANISA\nKhối lượng tịnh: 200g\nThành phần: bột mì, bơ, đường"
        self.assertEqual(ci.guess_product_name(text), "BÁNH QUY BƠ DANISA")
        self.assertEqual(ci.guess_product_name("Thành phần: bột mì, đường"), "")
        one_line = ("Convertive Nở với 1 Bánh xốp ong Deka Jumbo cá phê trắng White Coffee Tổng số phần ăn: 10; "
                    "Thành phần: Bột mì (34,242%), Đường")
        self.assertEqual(ci.guess_product_name(one_line), "Bánh xốp ong Deka Jumbo cá phê trắng")

    def test_name_guess_rejects_usage_text_and_falls_back_to_filename(self):
        self.assertEqual(ci.guess_product_name("Tương SỬ DỤNG Dành cho trẻ từ 3 tuổi. Thành phần: đậu nành"), "")
        self.assertEqual(ci.guess_product_name("Sữa tươi tiệt trùng Vinamilk có đường HƯỚNG DẪN SỬ DỤNG lắc đều. "
                                               "Thành phần: sữa bò"), "Sữa tươi tiệt trùng Vinamilk có đường")
        self.assertEqual(ci.display_name_from_file("nuoc_tuong_maggi.png"), "nuoc_tuong_maggi")
        long_name = ci.display_name_from_file("1794675545761_2204332108467412409_g8131696245559082487_01c5db.jpg")
        self.assertLessEqual(len(long_name), 40)
        self.assertIn("…", long_name)

    def test_validate_image(self):
        self.assertIn("không được hỗ trợ", ci.validate_image(_png(), "label.gif")[1])
        self.assertIn("không được hỗ trợ", ci.validate_image(b"%PDF", "label.pdf")[1])
        self.assertIsNone(ci.validate_image(_png(), "label.webp")[1])  # webp: tên đuôi hợp lệ, nội dung PNG vẫn mở được
        img, err = ci.validate_image(_png(), "ok.png")
        self.assertIsNotNone(img)
        self.assertIsNone(err)
        self.assertIsNotNone(ci.validate_image(b"not an image", "x.jpg")[1])
        self.assertIsNotNone(ci.validate_image(_png((60, 60)), "tiny.png")[1])
        self.assertIsNotNone(ci.validate_image(_png(noise=False), "blank.png")[1])
        self.assertIsNotNone(ci.validate_image(b"", "empty.png")[1])


class TestLabelCorrection(unittest.TestCase):
    """Người dùng sửa thông tin nhãn ngay trong chat — không hiện danh sách thành phần, không bắt xác nhận."""

    LABEL = "Thành phần: đường, chất điều chỉnh độ acid (33), chất điều vị (E621, E627), E1422, sữa bột. Có chứa sữa, đậu nành."

    def mentions(self):
        from modules.ocr_extractor import parse_ingredients_from_ocr
        return parse_ingredients_from_ocr(self.LABEL, include_allergen_statements=True)

    def test_detects_corrections(self):
        cases = {
            "không có E621": {"remove": ["E621"], "replace": []},
            "Sản phẩm không có sữa và đậu nành.": {"remove": ["sữa", "đậu nành"], "replace": []},
            "Hệ thống đọc sai E621.": {"remove": ["E621"], "replace": []},
            "mã là E330 chứ không phải E33": {"remove": [], "replace": [("E33", "E330")]},
            "sửa E33 thành E330": {"remove": [], "replace": [("E33", "E330")]},
            "không phải E33 mà là E330": {"remove": [], "replace": [("E33", "E330")]},
        }
        for text, want in cases.items():
            parsed = ci.parse_message(text)
            self.assertEqual(parsed["kind"], "correction", text)
            got = parsed["correction"]
            self.assertEqual((got["remove"], got["replace"]), (want["remove"], want["replace"]), text)
        listed = ci.parse_message("Thành phần đúng là: bột mì, đường, muối")
        self.assertEqual(listed["correction"]["ingredients"], "bột mì, đường, muối")

    def test_questions_and_profile_are_not_corrections(self):
        for text in ("không có sữa phải không?", "có chứa sữa không", "E621 là gì?", "E33 có an toàn không?",
                     "sản phẩm này không có chất bảo quản à?", "Tôi không có dị ứng"):
            self.assertIsNone(ci.parse_message(text).get("correction"), text)

    def test_correction_with_question_keeps_question(self):
        parsed = ci.parse_message("nhãn không có E621. Vậy có an toàn cho trẻ không?")
        self.assertEqual(parsed["correction"]["remove"], ["E621"])
        self.assertEqual(parsed["question"], "Vậy có an toàn cho trẻ không?")

    def test_apply_remove_replace_and_list(self):
        from modules.ocr_extractor import parse_ingredients_from_ocr
        names = lambda ms: [m["raw_mention"] for m in ms]
        new, done, missing = ci.apply_label_correction(self.mentions(), {"remove": ["E621"]})
        self.assertFalse(any("E621" in n for n in names(new)))
        self.assertTrue(any("E627" in n for n in names(new)))     # mã khác cùng nhóm vẫn giữ
        self.assertEqual(missing, [])
        new, _, _ = ci.apply_label_correction(self.mentions(), {"remove": ["sữa"]})
        self.assertFalse(any("sữa" in n.lower() for n in names(new)))
        self.assertIn("đậu nành", names(new))
        new, done, _ = ci.apply_label_correction(self.mentions(), {"replace": [("E33", "E330")]})
        self.assertIn("chất điều chỉnh độ acid (E330)", names(new))
        _, done, missing = ci.apply_label_correction(self.mentions(), {"remove": ["E142"]})
        self.assertEqual((done, missing), ([], ["E142"]))          # E142 không khớp nhầm E1422
        new, done, _ = ci.apply_label_correction(self.mentions(), {"ingredients": "bột mì, đường, muối"},
                                                 parse_ingredients_from_ocr)
        self.assertEqual(names(new), ["bột mì", "đường", "muối"])


class TestAgeInMonths(unittest.TestCase):
    def test_months_are_kept_for_infant_rules(self):
        p, _ = ci.merge_profile(ci.empty_profile(), ci.extract_profile("Bé nhà tôi 8 tháng tuổi")["update"])
        self.assertEqual(p["age_months"], 8)
        self.assertEqual(ci.profile_to_persona(p)["demographics"]["age_months"], 8)
        self.assertIn("8 tháng tuổi", ci.profile_summary(p))
        p2, _ = ci.merge_profile(p, ci.extract_profile("bé 2 tuổi")["update"])
        self.assertIsNone(p2["age_months"])
        self.assertNotIn("age_months", ci.profile_to_persona(p2)["demographics"])

    def test_months_without_the_word_tuoi(self):
        self.assertEqual(ci.extract_profile("Bé nhà tôi 8 tháng, dị ứng trứng")["update"].get("age_months"), 8)
        self.assertNotIn("age_months", ci.extract_profile("Tôi đã dùng sản phẩm này 8 tháng")["update"])

    def test_months_of_the_baby_are_not_the_mothers_age(self):
        upd = ci.extract_profile("Tôi đang cho con bú, bé 3 tháng tuổi")["update"]
        self.assertTrue(upd["lactating"])
        self.assertNotIn("age_months", upd)
        self.assertNotIn("age", upd)


if __name__ == "__main__":
    unittest.main()
