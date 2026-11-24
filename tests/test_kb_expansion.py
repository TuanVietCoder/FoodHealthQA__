"""Bất biến của KB phụ gia sau khi đối chiếu Codex GSFA Table One và mở rộng theo tần suất trên nhãn."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = json.loads((ROOT / "knowledge_base" / "additives_standard.json").read_text(encoding="utf-8"))
DERIVED = ROOT / "knowledge_base" / "sources" / "derived"
TABLE_ONE_FILE = DERIVED / "codex_192_gsfa_table_one.json"
MANIFEST = json.loads((ROOT / "knowledge_base" / "sources" / "catalog" / "external_sources_manifest.json").read_text(encoding="utf-8"))


def _norm(code):
    import re
    return re.sub(r"[()\s]", "", re.sub(r"^(ins|e)\s*", "", str(code), flags=re.I)).lower()


class KbInvariantTests(unittest.TestCase):
    def test_codes_are_unique(self):
        codes = [_norm(x["ins_code"]) for x in KB]
        self.assertEqual(len(codes), len(set(codes)))

    def test_codex_claims_point_to_the_manifest_file_and_extracted_codes(self):
        table = json.loads(TABLE_ONE_FILE.read_text(encoding="utf-8"))
        manifest_sha = next(x["sha256"] for x in MANIFEST["manually_downloaded"] if x["file"] == "cxs_192e.pdf")
        self.assertEqual(table["sha256"], manifest_sha)
        checked = [x for x in KB if x["codex_claim"]["verification_status"] == "source_checked_gsfa_table_one"]
        self.assertGreaterEqual(len(checked), 75)
        for item in checked:
            claim = item["codex_claim"]
            self.assertEqual(claim["sha256"], manifest_sha, item["ins_code"])
            self.assertGreater(claim["n_food_category_provisions"], 0, item["ins_code"])
            self.assertTrue(all(_norm(c) in table["additives"] for c in claim["codex_codes"]), item["ins_code"])
            self.assertFalse(claim["use_for_compliance_decision"])

    def test_codex_extract_matches_text_read_from_the_pdf(self):
        table = json.loads(TABLE_ONE_FILE.read_text(encoding="utf-8"))["additives"]
        aspartame = next(e for e in table["951"]["entries"] if e["food_category_no"] == "01.1.4")
        self.assertEqual((aspartame["max_level"], aspartame["year_adopted"]), ("600 mg/kg", "2024"))
        self.assertEqual(table["211"]["group_members"], ["210", "211", "212", "213"])  # nhóm benzoat dùng chung các dòng
        self.assertEqual(table["338"]["entries"], table["450vii"]["entries"])  # nhóm phosphat trải qua nhiều trang

    def test_added_entries_claim_nothing_beyond_their_sources(self):
        added = [x for x in KB if x["unverified_legacy_regulatory_claims"].get("verification_status") == "not_applicable_added_from_primary_sources"]
        self.assertGreaterEqual(len(added), 30)
        for item in added:
            # lưu ý sức khỏe chỉ được có khi đã kiểm nguồn VÀ kèm trích dẫn nguyên văn (vd 450(iii) cùng nhóm phosphat EFSA 2019)
            for note in item["health_considerations"]:
                self.assertEqual(note["verification_status"], "source_checked", item["ins_code"])
                self.assertTrue(note.get("evidence"), item["ins_code"])
            self.assertEqual(item["allergen_assessment"]["groups"], [], item["ins_code"])
            self.assertEqual(item["adi_assessment"]["verification_status"], "source_checked", item["ins_code"])
            self.assertIn("Phụ lục 1", item["vn_regulation"])


if __name__ == "__main__":
    unittest.main()


class KbMatchesRecognitionRegistryTests(unittest.TestCase):
    def test_every_kb_code_is_a_key_the_registry_can_produce_so_it_can_attach_to_labels(self):
        from modules.additive_linker import normalize_ins_key
        registry = json.loads((ROOT / "knowledge_base" / "additive_registry.json").read_text(encoding="utf-8"))
        keys = {normalize_ins_key(r["ins_code"]) for r in registry}
        orphans = [x["ins_code"] for x in KB for part in x["ins_code"].split("+") if normalize_ins_key(part) not in keys]
        # INS 420 và 160a từng mồ côi (sổ chỉ có 420(i)/(ii), 160a(i)-(iii)) nên không bao giờ gắn được vào nhãn
        self.assertEqual(orphans, [])

    def test_sorbitol_variants_carry_the_verified_laxative_note_and_a_jecfa_adi(self):
        for code in ("INS 420(i)", "INS 420(ii)"):
            item = next(x for x in KB if x["ins_code"] == code)
            self.assertEqual(item["adi_assessment"]["verification_status"], "source_checked", code)
            self.assertIn("general_population", [h["condition"] for h in item["health_considerations"]], code)

    def test_withdrawn_beta_carotene_adi_applies_only_to_the_forms_jecfa_withdrew(self):
        adi = {x["ins_code"]: x["adi_assessment"]["display_text"] for x in KB if x["ins_code"].startswith("INS 160a(")}
        self.assertIn("WITHDRAWN", adi["INS 160a(i)"])
        self.assertIn("WITHDRAWN", adi["INS 160a(iii)"])
        self.assertNotIn("WITHDRAWN", adi["INS 160a(ii)"])  # carotene thực vật (hỗn hợp): JECFA 1993 "Not Specified"

    def test_phosphate_mtdi_is_not_turned_into_a_numeric_adi(self):
        item = next(x for x in KB if x["ins_code"] == "INS 450(iii)")
        self.assertIn("MTDI 70 mg/kg bw (as P)", item["adi_assessment"]["display_text"])
        self.assertIsNone(item["adi_assessment"]["numeric_upper_mg_per_kg_bw_day"])
