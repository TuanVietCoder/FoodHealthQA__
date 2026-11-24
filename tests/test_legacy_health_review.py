"""Tuyên bố sức khỏe trong KB: không còn tuyên bố cũ chưa kiểm; mọi trích dẫn 'evidence' là nguyên văn của đúng trang nguồn."""
import json
import re
import unittest
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
KB = json.loads((ROOT / "knowledge_base" / "additives_standard.json").read_text(encoding="utf-8"))
REVIEW = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "legacy_health_claims_review.json").read_text(encoding="utf-8"))


def _squash(text):
    return " ".join(text.split())


def _page_text(rel_path, page):
    path = ROOT / rel_path
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as doc:
            return _squash(doc[page - 1].get_text())
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", path.read_text(encoding="utf-8", errors="ignore"))
    return _squash(re.sub(r"<[^>]+>", " ", raw))


class LegacyHealthClaimsReviewTests(unittest.TestCase):
    def test_no_unreviewed_legacy_health_claim_is_left_in_the_kb(self):
        self.assertEqual([x["ins_code"] for x in KB if x.get("unverified_legacy_health_claims")], [])

    def test_every_one_of_the_72_old_claims_has_a_recorded_decision_and_reason(self):
        claims = REVIEW["claims"]
        self.assertEqual(len(claims), 72)
        self.assertEqual([c["index"] for c in claims], list(range(1, 73)))
        self.assertEqual({c["decision"] for c in claims}, {"covered", "promoted", "removed"})
        self.assertTrue(all(c["reason"].strip() for c in claims))
        self.assertEqual(sum(REVIEW["counts"].values()), 72)

    def test_every_evidence_quote_is_verbatim_on_the_cited_page(self):
        checked = 0
        for item in KB:
            for h in item["health_considerations"]:
                for e in h.get("evidence", []):
                    self.assertIn(_squash(e["quote"]), _page_text(e["source_file"], e["page"]), (item["ins_code"], e["source_file"], e["page"]))
                    checked += 1
        self.assertGreaterEqual(checked, 35)

    def test_rewritten_messages_no_longer_carry_numbers_the_sources_do_not_have(self):
        sulfite = next(x for x in KB if x["ins_code"] == "INS 220")["health_considerations"][0]
        self.assertIn("3–10%", sulfite["message"])
        self.assertNotIn("5–10%", sulfite["message"])
        self.assertIn("5–10%", sulfite["message_before_review"])  # giữ lại câu cũ để truy vết
        nitrite = next(x for x in KB if x["ins_code"] == "INS 250")["health_considerations"][0]
        self.assertNotIn("6 tháng", nitrite["message"])
        self.assertIn("3 tháng", nitrite["message"])

    def test_review_does_not_change_which_conditions_can_raise_alerts(self):
        """constraint_model chỉ lấy health_considerations 'source_checked' theo điều kiện: điều kiện của các mục đã có giữ nguyên,
        mục mới chỉ dùng 'general_population' (không khớp hồ sơ bệnh lý nào)."""
        before = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "legacy_health_claims_review.json").read_text(encoding="utf-8"))
        self.assertIn("general_population", {h["condition"] for x in KB if x["ins_code"] == "INS 320" for h in x["health_considerations"]})
        conditions = {(x["ins_code"], h["condition"]) for x in KB for h in x["health_considerations"]}
        for code in ("INS 220", "INS 221"):
            self.assertIn((code, "asthma"), conditions)
        for code in ("INS 338", "INS 339(i)", "INS 450(i)", "INS 452(i)", "INS 341(i)"):
            self.assertIn((code, "ckd"), conditions)
        self.assertTrue(before["claims"])

    def test_every_health_consideration_is_source_checked(self):
        for item in KB:
            for h in item["health_considerations"]:
                self.assertEqual(h["verification_status"], "source_checked", item["ins_code"])


class NewSourcesEvidenceTests(unittest.TestCase):
    """Nguồn tải thêm 2026-11-23: mọi trích dẫn (cả của adi_assessment) là nguyên văn trang nguồn; mô tả file đúng nội dung."""

    def test_adi_assessment_evidence_is_verbatim(self):
        checked = 0
        for item in KB:
            for e in item["adi_assessment"].get("evidence", []):
                self.assertIn(_squash(e["quote"]), _page_text(e["source_file"], e["page"]), (item["ins_code"], e["source_file"], e["page"]))
                checked += 1
        self.assertGreaterEqual(checked, 4)  # 160a(i) và 160a(iii), mỗi mục 2 trích dẫn

    def test_withdrawn_beta_carotene_is_backed_by_jecfa_87_and_names_the_smokers_group(self):
        for code in ("INS 160a(i)", "INS 160a(iii)"):
            item = next(x for x in KB if x["ins_code"] == code)
            self.assertIn("WITHDRAWN", item["adi_assessment"]["display_text"])
            smokers = [h for h in item["health_considerations"] if h["condition"] == "smokers"]
            self.assertEqual(len(smokers), 1, code)
            self.assertTrue(all(e["source_file"].endswith("jecfa87_2019_summary.pdf") for e in smokers[0]["evidence"]))

    def test_sorbitol_laxative_note_now_cites_the_consolidated_1169_2011_text(self):
        for code in ("INS 420(i)", "INS 420(ii)"):
            item = next(x for x in KB if x["ins_code"] == code)
            note = next(h for h in item["health_considerations"] if h["condition"] == "general_population")
            self.assertEqual(note["source_id"], "EU_REG_1169_2011")
            self.assertIn("10%", note["message"])
            self.assertTrue(any(e["source_file"].endswith("CELEX_02011R1169_2025.pdf") for e in note["evidence"]))

    def test_all_phosphates_with_a_ckd_note_cite_kdoqi_as_extra_evidence_and_keep_the_ckd_condition(self):
        for code in ("INS 338", "INS 339(i)", "INS 341(i)", "INS 450(i)", "INS 450(iii)", "INS 452(i)"):
            item = next(x for x in KB if x["ins_code"] == code)
            ckd = [h for h in item["health_considerations"] if h["condition"] == "ckd"]
            self.assertEqual(len(ckd), 1, code)
            files = {e["source_file"].rsplit("/", 1)[-1] for e in ckd[0]["evidence"]}
            self.assertIn("efsa_2019_5674.pdf", files, code)
            self.assertIn("KDOQI_Nutrition_2020.pdf", files, code)

    def test_every_new_file_is_registered_with_the_hash_of_what_is_on_disk(self):
        manifest = json.loads((ROOT / "knowledge_base" / "sources" / "catalog" / "external_sources_manifest.json").read_text(encoding="utf-8"))
        registered = {x["file"]: x for x in manifest["manually_downloaded"]}
        import hashlib
        for name in ("jecfa87_2019_summary.pdf", "CELEX_02011R1169_2025.pdf", "KDOQI_Nutrition_2020.pdf"):
            path = ROOT / "knowledge_base" / "sources" / "raw" / "external" / name
            self.assertEqual(registered[name]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest(), name)
        # KDIGO 2024 và ADA 2023 rút gọn đã dời vào _archive_unused/sources/ (không có câu nào dùng được cho KB)
        for archived in ("KDIGO_2024_CKD.pdf", "ada_standards_2023_abridged.pdf"):
            self.assertNotIn(archived, registered)
            self.assertFalse((ROOT / "knowledge_base" / "sources" / "raw" / "external" / archived).exists(), archived)


if __name__ == "__main__":
    unittest.main()
