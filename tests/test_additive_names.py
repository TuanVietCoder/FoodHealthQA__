import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_research"))

from additive_names import name_hits  # noqa: E402
import source_grounding_v5 as sg  # noqa: E402


def lab(ing=(), add=(), warn=()):
    return {"ingredient": list(ing), "additive": list(add), "warning": list(warn)}


class AdditiveNameTests(unittest.TestCase):
    def groups(self, label):
        return set(name_hits(label))

    def test_phosphate_names_are_detected(self):
        for name in ("Dinatri phosphat", "Tricalci phosphat", "Dicalci phosphat", "Canxi Photphat",
                     "Natri polyphosphat", "Disodium Dihydrogen Pyrophosphate", "Dikali hydro phosphat"):
            self.assertEqual(self.groups(lab(ing=[f"khoáng chất ({name})"])), {"phosphate"}, name)

    def test_non_group_substances_are_not_flagged(self):
        for name in ("Sắt (III) Pyrophosphat", "Hydroxypropyl Distarch Phosphate 1442", "Trisodium citrate (INS 331(iii))",
                     "Canxi carbonat", "Thiamin mononitrat", "Natri bicarbonat"):
            self.assertEqual(self.groups(lab(ing=[name])), set(), name)

    def test_negated_mentions_are_ignored(self):
        label = lab(warn=["Không chứa Natri Benzoat và Kali Sorbat.", "Không phẩm màu tổng hợp Sunset yellow FCF và Ponceau 4R"])
        self.assertEqual(self.groups(label), set())

    def test_colours_nitrite_tio2_aspartame(self):
        self.assertEqual(self.groups(lab(add=["Allura red AC", "Tartrazine"])), {"mccann"})
        self.assertEqual(self.groups(lab(add=["Natri nitrit"])), {"nitrite"})
        self.assertEqual(self.groups(lab(add=["Titan dioxyd"])), {"tio2"})
        self.assertEqual(self.groups(lab(add=["Aspartam"])), {"aspartame"})

    def test_name_only_phosphate_now_reaches_the_ckd_flag(self):
        label = lab(ing=["các khoáng chất (dinatri phosphat, magnesi oxyd)"])
        codes = sg.merge_declared_codes([], label)
        self.assertEqual(sg.ckd_answer("X", codes)[0], "CAUTION")
        self.assertEqual(sg.ckd_answer("X", sg.merge_declared_codes([], lab(ing=["canxi carbonat"])))[0], "SAFE")


if __name__ == "__main__":
    unittest.main()
