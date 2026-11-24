"""Tên phụ gia ghi trên nhãn Việt Nam: biến thể chính tả thường gặp phải nhận ra đúng mã INS (và không nhận nhầm)."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import additive_linker as al  # noqa: E402

REGISTRY = Path(__file__).resolve().parents[1] / "knowledge_base" / "additive_registry.json"


def _codes(text):
    return [item.get("ins_code") for item in al.link_additive_mentions(text)]


class LabelNameVariantTests(unittest.TestCase):
    def test_monosodium_glutamate_is_recognised_without_the_L_prefix(self):
        for name in ("Mononatri glutamat", "mononatri glutamat", "Mononatri L-glutamat", "Monosodium glutamate",
                     "Monosodium L-glutamate", "natri glutamat", "bột ngọt", "mì chính", "MSG"):
            self.assertEqual(_codes(name), ["INS 621"], name)

    def test_axit_and_acid_are_the_same_word(self):
        for name, code in (("axit citric", "INS 330"), ("Axit citric (330)", "INS 330"), ("axit ascorbic", "INS 300"),
                           ("axit sorbic", "INS 200"), ("axit benzoic", "INS 210"), ("axit phosphoric", "INS 338"),
                           ("axit lactic", "INS 270"), ("axit axetic", "INS 260"), ("acid citric", "INS 330")):
            self.assertEqual(_codes(name), [code], name)

    def test_named_L_forms_resolve_through_explicit_aliases(self):
        for name, code in (("Axit L-ascorbic", "INS 300"), ("L-ascorbic acid", "INS 300"), ("Vitamin C (Natri-L-ascorbat)", "INS 301")):
            self.assertIn(code, _codes(name), name)

    def test_stereo_prefixes_are_never_ignored_because_D_DL_and_L_are_different_substances(self):
        # dl-alpha-tocopherol là 307c (tổng hợp), KHÔNG phải d-alpha-tocopherol 307a (tự nhiên)
        self.assertEqual(_codes("dl-alpha Tocopherol"), ["INS 307c"])
        # tên hoá học của acid folic chứa "L-glutamic acid" không có nghĩa là phụ gia 620
        self.assertNotIn("INS 620", _codes("Folic Acid (N-Pteroyl-L-glutamic Acid)"))

    def test_common_short_names_resolve(self):
        for name, code in (("TBHQ", "INS 319"), ("Acesulfame K", "INS 950"), ("Azorubine", "INS 122"), ("Azorubin", "INS 122"),
                           ("Steviol glycoside", "INS 960a"), ("Caramen nhóm III", "INS 150c"), ("Sunset yellow", "INS 110"),
                           ("Titan dioxit", "INS 171"), ("Dinatri inosinat", "INS 631"), ("Dinatri guanylat", "INS 627")):
            self.assertEqual(_codes(name), [code], name)

    def test_the_relaxed_matching_does_not_create_false_positives(self):
        for name in ("vitamin D", "Vitamin D3", "L-cystein", "muối glutamat", "axit béo", "Chất điều chỉnh độ axit", "đường", "muối"):
            self.assertEqual(_codes(name), [], name)

    def test_every_registry_name_still_links_to_its_own_code(self):
        al.clear_registry_cache()
        failures = []
        for record in json.loads(REGISTRY.read_text(encoding="utf-8")):
            key = al.normalize_ins_key(record["ins_code"])
            for name in {record.get("canonical_name_vi"), record.get("canonical_name_en"), *(record.get("aliases") or [])}:
                if name and key not in {al.normalize_ins_key(c) for c in _codes(name)}:
                    failures.append((record["ins_code"], name))
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
