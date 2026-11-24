"""Hồi quy cho các lỗi tìm thấy khi thử nhãn sữa bột trẻ em (ColosBaby Gold Pedia)."""

import unittest

from modules import pipeline as pl
from modules.ocr_extractor import (
    parse_allergen_statements,
    parse_ingredients_from_ocr,
    post_process_ocr_spelling,
)


class SpellingTests(unittest.TestCase):
    def test_correct_words_with_diacritics_are_not_changed(self):
        text = "Dành cho trẻ từ 1-10 tuổi, có miễn dịch kém. Chất béo chuỗi trung bình (MCT). Phòng ngừa. Hướng dẫn của cán bộ y tế"
        out = post_process_ocr_spelling(text)["final"]
        for word in ("tuổi", "có", "trung", "Phòng", "của"):
            self.assertIn(word, out, word)
        # không sinh dị nguyên giả
        for fake in ("trứng", "phộng", "cua "):
            self.assertNotIn(fake, out.lower(), fake)


class IngredientParsingTests(unittest.TestCase):
    def test_stops_at_usage_section(self):
        text = ("Thành phần: Sữa bột, Dầu thực vật, Hương liệu tổng hợp HƯỚNG DẪN SỬ DỤNG Pha 5 muỗng gạt với 170 ml nước. "
                "BẢO QUẢN nơi khô ráo.")
        items = [m["raw_mention"] for m in parse_ingredients_from_ocr(text)]
        self.assertEqual(items[-1], "Hương liệu tổng hợp")
        self.assertFalse(any("muỗng" in i for i in items))

    def test_preservative_class_is_not_a_section_stop(self):
        items = [m["raw_mention"] for m in parse_ingredients_from_ocr("Thành phần: đường, chất bảo quản (211), muối")]
        self.assertEqual(items, ["đường", "chất bảo quản (211)", "muối"])

    def test_unclosed_parenthesis_does_not_swallow_the_rest(self):
        text = "Thành phần: Khoáng chất (Canxi Carbonat, Kali Iodid, Natri Selenit, Taurine, Biotin"
        items = [m["raw_mention"] for m in parse_ingredients_from_ocr(text)]
        self.assertGreaterEqual(len(items), 4)
        self.assertIn("Taurine", items)

    def test_sentence_contains_statement_without_colon(self):
        res = parse_allergen_statements(full_text="Cho trẻ ăn bằng cốc. Có chứa sữa, đậu nành, dầu cá.")
        self.assertEqual(res["contains_allergens"], ["sữa", "đậu nành", "dầu cá"])
        res2 = parse_allergen_statements(full_text="Thành phần: đường. Sản phẩm có chứa đường.")
        self.assertEqual(res2["contains_allergens"], [])   # "có chứa" giữa câu không phải lời cảnh báo


class CodeRecallRegressionTests(unittest.TestCase):
    """Ba nguyên nhân mã phụ gia bị sót trên ảnh thật (scripts_research/analyze_code_misses.py)."""

    def test_vietocr_code_confusions_are_fixed_in_code_context(self):
        from modules.ocr_extractor import _fix_code_confusions as fix
        self.assertEqual(fix("Chất Tạo Xốp (500(11), 508 (II), đường"), "Chất Tạo Xốp (500(ii), 508(ii), đường")
        self.assertEqual(fix("chất nhũ hóa (ins16oc(i), chiết"), "chất nhũ hóa (INS160c(i), chiết")
        self.assertEqual(fix("chất làm ẩm (42o(i), 422)"), "chất làm ẩm (420(i), 422)")
        self.assertEqual(fix("chất chống đông vón inss51, màu"), "chất chống đông vón INS551, màu")
        self.assertEqual(fix("E33O, muối"), "E330, muối")
        for untouched in ("lá trà (7,4 g/l)", "Vitamin B1 (1), bước 1", "ISO 22000, HACCP", "Energy 487 kcal"):
            self.assertEqual(fix(untouched), untouched)

    def test_misread_heading_still_finds_ingredients(self):
        items = [m["raw_mention"] for m in parse_ingredients_from_ocr(
            "Banh gao Hành PHÂN: Gạo japonica 51%, dầu thực vật, chất điều vị (INS 621)")]
        self.assertIn("chất điều vị (INS 621)", items)
        self.assertEqual(parse_ingredients_from_ocr("hôm nay hành phần tốt, đường"), [])

    def test_interleaved_stop_phrase_does_not_cut_the_list(self):
        # chữ của cột bên cạnh ("từ ngày sản xuất") chen vào giữa danh sách thành phần
        text = ("Thành Phần: Bột Mì, Gia Vị Cà từ ngày sản xuất Chua, Khoai Tây Miếng, Bột Cà Chua (3%), Đường, Muối, "
                "Chất Điều Vị (INS 621); Chất Nhũ Hóa (INS 471). Hạn sử dụng: 6 tháng kể từ ngày sản xuất.")
        items = " | ".join(m["raw_mention"] for m in parse_ingredients_from_ocr(text))
        self.assertIn("INS 621", items)
        self.assertIn("INS 471", items)
        # điểm dừng thật (sau đó là bảng số liệu) vẫn dừng
        text2 = "Thành phần: đường, muối. Giá trị dinh dưỡng trong 100 g: Năng lượng 400 kcal, Chất béo 5 g, Natri 200 mg"
        self.assertEqual([m["raw_mention"] for m in parse_ingredients_from_ocr(text2)], ["đường", "muối"])


class LinkerContextTests(unittest.TestCase):
    """Bước 4: mã đọc đúng số nhưng khâu liên kết từ chối."""

    def codes(self, text):
        from modules.additive_linker import link_additive_mentions, normalize_ins_key
        return sorted(normalize_ins_key(h["ins_code"]) for h in link_additive_mentions(text))

    def test_letter_suffix_is_not_split_by_postprocessing(self):
        out = post_process_ocr_spelling("Chất chống oxy hóa (INS 307b, INS 307c), màu (INS 150 a)")["final"]
        self.assertIn("INS 307b", out)
        self.assertIn("INS 150a", out)

    def test_unclosed_roman_suffix(self):
        self.assertEqual(self.codes("Chất tạo xốp (450(i"), ["450(i)"])

    def test_bare_codes_with_explicit_code_or_misread_class_word(self):
        self.assertEqual(self.codes("INS 331(iii), 296, 334)"), ["296", "331(iii)", "334"])
        self.assertEqual(self.codes("chật diều chính độ aciơ (260)"), ["260"])
        self.assertEqual(self.codes("sool ĐỊNH (1442, 1440, 415)"), ["1440", "1442", "415"])

    def test_quantities_are_not_codes(self):
        self.assertEqual(self.codes("chất đạm 100 g"), [])
        self.assertEqual(self.codes("bột mì (100, 200 g)"), [])
        self.assertEqual(self.codes("Năng lượng (2000, 1500)"), [])

    def test_orphan_code_next_to_code_list(self):
        items = [m["raw_mention"] for m in parse_ingredients_from_ocr(
            "Thành phần: đường, chất ổn định 1442, 296, 334), muối, 2023, 500")]
        self.assertIn("INS 296", items)
        self.assertIn("INS 334", items)
        self.assertIn("2023", items)        # năm không phải mã
        self.assertIn("500", items)         # cạnh năm, không cạnh cụm có mã → giữ nguyên
        self.assertEqual([m["raw_mention"] for m in parse_ingredients_from_ocr("Thành phần: đường, muối, 296, bột mì")],
                         ["đường", "muối", "296", "bột mì"])

    def test_roman_suffix_with_l(self):
        from modules.ocr_extractor import _fix_code_confusions as fix
        self.assertEqual(fix("chất điều chỉnh (331 (ili), 330)"), "chất điều chỉnh (331(iii), 330)")
        self.assertEqual(fix("INS 170li), muối"), "INS 170(i), muối")


class PhraseLexiconTests(unittest.TestCase):
    """Sửa chính tả theo cụm (modules/ocr_lexicon.py) — chỉ trong cụm thành phần, không sửa từ đơn."""

    def fix(self, text):
        from modules.ocr_lexicon import correct_phrases
        return correct_phrases(text)[0]

    def test_misread_phrases_are_corrected(self):
        self.assertEqual(self.fix("Sửa bột (30-45%)"), "Sữa bột (30-45%)")
        self.assertEqual(self.fix("Khoảng chất (Canxi Carbonat"), "Khoáng chất (Canxi Carbonat")
        self.assertEqual(self.fix("Hượng liệu tổng hợp"), "Hương liệu tổng hợp")
        self.assertEqual(self.fix("CHÁT BÉO THỰC VẬT"), "CHẤT BÉO THỰC VẬT")   # giữ chữ hoa

    def test_valid_or_ambiguous_text_is_untouched(self):
        for text in ("dầu nành", "đậu nành", "Chất điều vị", "chất chống oxy hoá",   # hoá/hóa chỉ khác chỗ đặt dấu
                     "xirô fructoza", "Carbohydrate", "Hydrolyzed", "tocopheryl"):  # không sửa từ đơn/Latin
            self.assertEqual(self.fix(text), text, text)

    def test_parser_records_corrections(self):
        ms = parse_ingredients_from_ocr("Thành phần: Sửa bột, đường, Khoảng chất (kẽm oxyd)")
        self.assertEqual(ms[0]["raw_mention"], "Sữa bột")
        self.assertEqual(ms[0]["ocr_corrections"], [("Sửa bột", "Sữa bột")])
        self.assertNotIn("ocr_corrections", ms[1])


class PersonaAndNutritionTests(unittest.TestCase):
    def test_nhanes_does_not_add_conditions(self):
        persona = {"id": "custom", "conditions": ["children_adhd"], "allergies": [], "demographics": {"age": 1}}
        enriched = pl.enrich_persona_with_nhanes(persona, "data", biomarkers_cache=pl.load_nhanes_biomarkers("data"))
        self.assertEqual(enriched["conditions"], ["children_adhd"])

    def test_estimated_nutrition_does_not_change_label(self):
        self.assertFalse(pl.NUTRITION_ESTIMATE_CHANGES_LABEL)

    def test_label_sodium_extraction(self):
        e = pl.extract_label_nutrition
        self.assertEqual(e("nhãn ghi natri 853 mg trên 100 g")["sodium_mg"], 853.0)
        self.assertEqual(e("ghi natri 28,8 mg trên 1 khẩu phần 50 g")["sodium_mg"], 57.6)
        self.assertEqual(e("THÀNH PHẦN DINH DƯỠNG TÍNH TRÊN 100 ml Năng lượng 34 kcal Natri 31 mg")["sodium_mg"], 31.0)
        self.assertIsNone(e("natri có cao không?"))           # không có số / cơ sở tính → không đoán
        # con số NGƯỠNG trong câu hỏi không được hiểu nhầm là số của sản phẩm
        q = ("Tôi bị tăng huyết áp — nhãn 'Mì ly A-One vị thịt xào' ghi natri 1523 mg trên 1 ly 65 g (quy đổi khoảng "
             "2343,1 mg/100 g). Mức này có đạt điều kiện 'natri thấp' (không quá 120 mg/100 g) theo Codex CAC/GL 23-1997?")
        self.assertAlmostEqual(e(q)["sodium_mg"], 2343.1, places=1)
        self.assertIsNone(e("Mức natri thấp là không quá 120 mg/100 g phải không?"))
        # "1" của "170 ml" không bị nuốt như "1 hộp"
        self.assertAlmostEqual(e("natri 59,5 mg trên 170 ml (1 hộp)")["sodium_mg"], 35.0, places=1)

    def test_label_sugar_extraction(self):
        e = pl.extract_label_nutrition
        q = ("Tôi bị đái tháo đường — nhãn ghi đường tổng số 16,0 g trên 1 khẩu phần 25 g (quy đổi khoảng 64 g/100 g). "
             "Mức này có đạt điều kiện 'không đường' (không quá 0,5 g/100 g) theo Codex CAC/GL 23-1997 không?")
        self.assertAlmostEqual(e(q)["sugars_g"], 64.0, places=1)
        self.assertIsNone(e("Sản phẩm không đường nghĩa là không quá 0,5 g/100 g?"))

    def test_codex_sugar_free_for_diabetes(self):
        persona = {"conditions": ["diabetes"]}
        _, warn_hi, _ = pl.evaluate_nutritional_risk("x", [], persona, label_nutrition={"sugars_g": 64.0})
        self.assertTrue(any("không đường" in w for w in warn_hi))
        _, warn_lo, details = pl.evaluate_nutritional_risk("x", [], persona, label_nutrition={"sugars_g": 0.3})
        self.assertFalse(any("không đường" in w for w in warn_lo))
        self.assertTrue(any(d.get("citation") == "CODEX_CAC_GL_23_1997" and d["I_g"] == 1 for d in details))

    def test_codex_low_sodium_for_hypertension(self):
        persona = {"conditions": ["hypertension"]}
        _, warn_hi, _ = pl.evaluate_nutritional_risk("x", [], persona, label_nutrition={"sodium_mg": 853.0})
        self.assertTrue(any("natri thấp" in w for w in warn_hi))
        _, warn_lo, details = pl.evaluate_nutritional_risk("x", [], persona, label_nutrition={"sodium_mg": 57.6})
        self.assertFalse(any("natri thấp" in w for w in warn_lo))
        self.assertTrue(any(d.get("citation") == "CODEX_CAC_GL_23_1997" and d["I_g"] == 1 for d in details))


if __name__ == "__main__":
    unittest.main()
