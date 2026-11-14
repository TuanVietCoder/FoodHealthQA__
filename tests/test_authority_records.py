import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_qa_bench_path = (
    ROOT / "data" / "archive" / "master_benchmark_2311" / "qa_benchmark.jsonl"
    if (ROOT / "data" / "archive" / "master_benchmark_2311" / "qa_benchmark.jsonl").exists()
    else ROOT / "data" / "qa_benchmark.jsonl"
)
ROWS = [json.loads(l) for l in _qa_bench_path.read_text(encoding="utf-8").splitlines() if l.strip()]
NEW = [r for r in ROWS if r.get("authority_topic")]
CLAIMS = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "external_claim_verification.json").read_text(encoding="utf-8"))
BASES = {"codex_sulphite_declaration": {"220", "221", "222", "223", "224", "225", "226", "227", "228"}, "eu_annex_v_colours": {"102", "104", "110", "122", "124", "129"}, "fda_red3_revoked": {"127"},
         "iarc_v134_aspartame": {"951"},
         "who_nss_pregnancy": {"950", "951", "952", "954", "955", "960", "961", "969"}}


# topics whose records are not "flag" answers: absence controls (SAFE), listing/ADI lookups and the Codex sugar threshold
OTHER_TOPICS = {"who_nss_pregnancy_absent", "ins_listing", "ins_adi", "sugar_codex",
                "who_nss_child", "who_nss_child_absent", "efsa_caffeine_child", "efsa_caffeine_child_absent"}
FLAG = [r for r in NEW if r["authority_topic"] in BASES]
LOOKUP_TOPICS = ("ins_listing", "ins_adi", "sugar_codex")  # INS coverage lookups, tested separately in InsCoverageTests


class AuthorityRecordTests(unittest.TestCase):
    def test_all_topics_present_and_scored(self):
        self.assertEqual({r["authority_topic"] for r in NEW}, set(BASES) | OTHER_TOPICS)
        self.assertTrue(all(r["eligible_for_scoring"] == "yes" for r in NEW))
        self.assertTrue(all(r["gold_safety_label"] == "CAUTION" for r in FLAG))
        for r in NEW:
            if r["authority_topic"] == "who_nss_pregnancy_absent":
                self.assertEqual((r["gold_safety_label"], r["tier"]), ("SAFE", "silver"), r["id"])

    def test_claims_were_checked_against_downloaded_text(self):
        for cid in ("EU_ANNEXV_COLOURS_WARNING", "FDA_RED3_REVOKED_FOOD", "IARC_V134_ASPARTAME_2B"):
            self.assertTrue(CLAIMS[cid]["passed"], cid)

    def test_each_record_names_only_codes_of_its_group_and_cites_its_source(self):
        want = {"codex_sulphite_declaration": "CODEX_CXS_1_1985", "eu_annex_v_colours": "EU_REG_1333_2008_ANNEX_V", "fda_red3_revoked": "US_FDA_RED3_REVOCATION",
                "iarc_v134_aspartame": "IARC_MONOGRAPHS_V134", "who_nss_pregnancy": "WHO_NSS_GUIDELINE_2023"}
        for r in FLAG:
            codes = {c for c in re.findall(r"INS(\d{3,4})", r["reference_answer"])}
            self.assertTrue(codes and codes <= BASES[r["authority_topic"]], r["id"])
            self.assertEqual(r["source_citations"], want[r["authority_topic"]], r["id"])

    def test_answers_do_not_overclaim(self):
        for r in NEW:
            a = r["reference_answer"]
            if r["authority_topic"] == "iarc_v134_aspartame":
                self.assertIn("không phải rủi ro", a)
                self.assertNotIn("JECFA", a)
            if r["authority_topic"] == "eu_annex_v_colours":
                self.assertIn("không suy ra yêu cầu tương ứng ở Việt Nam", a)
            if r["authority_topic"] == "fda_red3_revoked":
                self.assertIn("15/01/2027", a)




class SulphiteRecordTests(unittest.TestCase):
    def test_sulphite_records_only_for_labels_without_the_word_sulphite(self):
        recs = [r for r in NEW if r["authority_topic"] == "codex_sulphite_declaration"]
        self.assertTrue(recs)
        for r in recs:
            label = json.loads((ROOT / r["label_transcription_path"]).read_text(encoding="utf-8"))
            text = json.dumps(label, ensure_ascii=False).lower()
            self.assertNotRegex(text, r"sul[fp]it|sulphite|sulfite|lưu huỳnh")
            self.assertEqual(r["source_claim_locators"], "CODEX_CXS_1_1985#Sec4.2.1.7")
            self.assertIn("không thể kết luận", r["reference_answer"])
        self.assertTrue(CLAIMS["CXS1_4217_SULPHITE_DECLARATION"]["passed"])


class ClaimPageTests(unittest.TestCase):
    def test_pdf_claims_have_pages_and_records_expose_them(self):
        for cid, page in (("CXS1_4217_SULPHITE_DECLARATION", 7), ("EU_ANNEXV_COLOURS_WARNING", 366)):
            self.assertIn(page, set(CLAIMS[cid]["evidence_pages"].values()))
        recs = [r for r in ROWS if r.get("authority_topic") == "codex_sulphite_declaration"]
        self.assertTrue(all("CXS1_4217_SULPHITE_DECLARATION: p.7" in r["source_claim_pages"] for r in recs))
        self.assertGreater(sum(1 for r in ROWS if r.get("source_claim_pages")), 1500)


class WhoNssPregnancyTests(unittest.TestCase):
    """Pregnancy records backed by the WHO 2023 non-sugar-sweetener guideline (read from the downloaded PDF)."""
    RECS = [r for r in ROWS if r.get("authority_topic") == "who_nss_pregnancy"]

    def test_records_are_pregnancy_persona_silver_caution(self):
        self.assertGreaterEqual(len(self.RECS), 50)
        for r in self.RECS:
            self.assertEqual((r["persona_id"], r["domain"], r["tier"], r["gold_safety_label"]),
                             ("pregnant_woman", "vulnerable_groups", "silver", "CAUTION"), r["id"])
            self.assertEqual(r["source_claim_locators"], "WHO_NSS_GUIDELINE_2023#Pregnancy")

    def test_claim_passed_and_cites_pages(self):
        c = CLAIMS["WHO_NSS_2023_PREGNANCY"]
        self.assertTrue(c["passed"])
        pages = {p for v in c["evidence_pages"].values() for p in (v if isinstance(v, list) else [v])}
        self.assertIn(26, pages)  # the OR 1.25 preterm-birth sentence
        self.assertTrue(all("WHO_NSS_2023_PREGNANCY: p." in r["source_claim_pages"] for r in self.RECS))

    def test_polyols_are_never_flagged(self):
        # WHO states sugar alcohols (sorbitol 420, maltitol 965, erythritol 968, xylitol 967...) are not NSS
        for r in self.RECS:
            for code in re.findall(r"INS(\d{3})", r["ins_codes_matched"]):
                self.assertNotIn(code, {"420", "421", "965", "967", "968"}, r["id"])

    def test_answers_state_limits_and_do_not_overclaim(self):
        for r in self.RECS:
            a = r["reference_answer"]
            for must in ("OR 1,25", "độ chắc chắn thấp", "không chứng minh nhân quả", "không đặt ngưỡng an toàn riêng",
                         "nhãn không ghi lượng NSS", "hỏi ý kiến bác sĩ"):
                self.assertIn(must, a, r["id"])
            self.assertNotRegex(a, r"(?i)gây sinh non|không an toàn cho thai|cấm", r["id"])
            self.assertNotIn("JECFA", a)  # the ADI is a separate source and is not part of this claim

    def test_every_flagged_code_is_declared_on_the_label(self):
        for r in self.RECS:
            label = json.loads((ROOT / r["label_transcription_path"]).read_text(encoding="utf-8"))
            text = json.dumps(label, ensure_ascii=False).lower()
            for code in re.findall(r"INS(\d{3})", r["ins_codes_matched"]):
                self.assertTrue(code in text or {"951": "aspartam"}.get(code, "@@") in text, (r["id"], code))


class SugarCodexTests(unittest.TestCase):
    """Total sugars on the label vs the Codex CAC/GL 23-1997 'sugars free' threshold (0.5 g per 100 g/ml)."""
    RECS = [r for r in ROWS if r.get("authority_topic") == "sugar_codex"]

    def test_claim_passed_and_cited_with_the_sugar_locator(self):
        self.assertTrue(CLAIMS["CODEX_GL23_SUGARS_FREE_PER_100G"]["passed"])
        self.assertGreaterEqual(len(self.RECS), 30)
        for r in self.RECS:
            self.assertEqual(r["source_claim_locators"], "CODEX_CAC_GL_23_1997#Table_Sugars", r["id"])
            self.assertEqual((r["persona_id"], r["domain"], r["tier"]), ("diabetes_hypertension", "vulnerable_groups", "core"))

    def test_verdict_matches_an_independent_recomputation_from_the_label(self):
        for r in self.RECS:
            label = json.loads((ROOT / r["label_transcription_path"]).read_text(encoding="utf-8"))
            entry = [e for e in label["nutrition"] if "đường" in e["name"].lower()
                     and not re.search(r"bổ sung|thêm|added|lactose|glucose|fructose|sucrose|huyết", e["name"].lower())]
            self.assertEqual(len(entry), 1, r["id"])
            grams = float(re.match(r"\s*(\d+(?:[.,]\d+)?)", entry[0]["value"]).group(1).replace(",", "."))
            per100 = grams / float(label["sodium_basis"]["amount"]) * 100
            self.assertEqual(r["label_v5"], "SAFE" if per100 <= 0.5 else "CAUTION", r["id"])

    def test_answers_do_not_claim_clinical_safety(self):
        for r in self.RECS:
            a = r["reference_answer"]
            self.assertIn("Codex CAC/GL 23-1997", a)
            if r["label_v5"] == "CAUTION":
                self.assertIn("không phải kết luận y khoa", a)
            else:
                self.assertIn("không thay thế chỉ dẫn điều trị", a)


class InsCoverageTests(unittest.TestCase):
    """Questions on INS codes no other record asks about (TT24 listing status and JECFA ADI status)."""
    COV = [r for r in ROWS if r.get("authority_topic") in ("ins_listing", "ins_adi")]
    TT24 = set(json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "tt24_positive_list_codes.json").read_text(encoding="utf-8"))["codes"])

    def test_records_exist_scored_core_and_one_code_each(self):
        self.assertGreaterEqual(len(self.COV), 50)
        for r in self.COV:
            self.assertEqual((r["tier"], r["eligible_for_scoring"]), ("core", "yes"), r["id"])
            self.assertEqual(len(re.findall(r"INS", r["ins_codes_matched"])), 1, r["id"])
            self.assertNotIn("INSINS", r["question"] + r["reference_answer"], r["id"])

    def test_new_codes_do_not_repeat_codes_asked_elsewhere(self):
        base = lambda c: re.search(r"\d+", c).group(0)
        elsewhere = {base(c) for r in ROWS if r.get("authority_topic") not in LOOKUP_TOPICS
                     for c in re.findall(r"INS\s?\d{3,4}", r.get("ins_codes_matched", ""))}
        for r in self.COV:
            self.assertNotIn(base(r["ins_codes_matched"]), elsewhere, r["id"])

    def test_listing_records_are_in_tt24_and_do_not_claim_safety(self):
        recs = [r for r in self.COV if r["authority_topic"] == "ins_listing"]
        self.assertTrue(recs)
        for r in recs:
            code = r["ins_codes_matched"].lower().replace("ins", "")
            self.assertTrue(code in self.TT24 or any(t.startswith(code + "(") for t in self.TT24), r["id"])
            self.assertEqual(r["label_v5"], "PERMITTED_LISTED", r["id"])
            self.assertIn("không đủ để kết luận mức an toàn tuyệt đối", r["reference_answer"])
            self.assertIn("VN_TT24_2019", r["source_citations"])

    def test_adi_records_state_the_jecfa_adi_and_cite_a_passed_claim(self):
        recs = [r for r in self.COV if r["authority_topic"] == "ins_adi"]
        self.assertTrue(recs)
        for r in recs:
            self.assertIn("ADI", r["reference_answer"])
            self.assertIn("JECFA_EVALUATIONS", r["source_citations"])
            ids = [c for c in r["external_claims"].split(";") if c.startswith("JECFA_ADI_")]
            self.assertTrue(ids and all(CLAIMS[c]["passed"] for c in ids), r["id"])
            self.assertIn(r["label_v5"], ("SAFE", "ABSTAIN"))

    def test_code_is_really_printed_on_the_label(self):
        import sys
        sys.path.insert(0, str(ROOT / "scripts_research"))
        import source_grounding_v5 as sg
        for r in self.COV:
            label = json.loads((ROOT / r["label_transcription_path"]).read_text(encoding="utf-8"))
            self.assertTrue(sg.code_in_label_tokens(r["ins_codes_matched"], label), r["id"])


if __name__ == "__main__":
    unittest.main()


class SharedHelperTests(unittest.TestCase):
    def test_field_text_copy_matches_create_benchmark_v5(self):
        import sys
        sys.path.insert(0, str(ROOT / "scripts_research"))
        import create_benchmark_v5 as b5
        import source_grounding_v5 as sg
        for p in sorted((ROOT / "assets" / "labels").glob("*.json")):
            label = json.loads(p.read_text(encoding="utf-8"))
            for k in ("ingredient", "additive", "warning", "nutrition"):
                self.assertEqual(sg._field_text(label.get(k)), b5.field_text(label.get(k)), f"{p.name}:{k}")


class ChildNssTests(unittest.TestCase):
    """WHO 2023 NSS guideline asked by the parent of a 5-year-old: flagged labels (CAUTION) + balanced no-NSS controls (SAFE)."""
    FLAG_RECS = [r for r in ROWS if r.get("authority_topic") == "who_nss_child"]
    CTRL = [r for r in ROWS if r.get("authority_topic") == "who_nss_child_absent"]

    def test_claim_passed_with_child_evidence_and_pages(self):
        c = CLAIMS["WHO_NSS_2023_CHILDREN"]
        self.assertTrue(c["passed"], c)
        pages = {p for v in c["evidence_pages"].values() for p in (v if isinstance(v, list) else [v])}
        self.assertIn(26, pages)  # overall certainty for children was low
        for r in self.FLAG_RECS + self.CTRL:
            self.assertIn("WHO_NSS_2023_CHILDREN: p.", r["source_claim_pages"], r["id"])

    def test_shape_and_balance(self):
        self.assertGreaterEqual(len(self.FLAG_RECS), 50)
        self.assertEqual(len(self.FLAG_RECS), len(self.CTRL))
        for r in self.FLAG_RECS + self.CTRL:
            self.assertEqual((r["persona_id"], r["domain"], r["tier"], r["eligible_for_scoring"]),
                             ("child_5yo", "vulnerable_groups", "silver", "yes"), r["id"])
            self.assertEqual(r["source_claim_locators"], "WHO_NSS_GUIDELINE_2023#Children", r["id"])
            self.assertEqual(r["source_citations"], "WHO_NSS_GUIDELINE_2023", r["id"])
        self.assertTrue(all(r["gold_safety_label"] == "CAUTION" for r in self.FLAG_RECS))
        self.assertTrue(all(r["gold_safety_label"] == "SAFE" for r in self.CTRL))

    def test_flagged_and_control_share_one_question_wording(self):
        shape = lambda r: re.sub(r"nhãn '.*' có", "nhãn 'X' có", r["question"])  # product names may contain apostrophes
        self.assertEqual(len({shape(r) for r in self.FLAG_RECS + self.CTRL}), 1)

    def test_polyols_never_flagged_and_codes_are_nss(self):
        for r in self.FLAG_RECS:
            codes = set(re.findall(r"INS(\d{3})", r["ins_codes_matched"]))
            self.assertTrue(codes and codes <= BASES["who_nss_pregnancy"], r["id"])

    def test_answers_state_limits_and_do_not_overclaim(self):
        for r in self.FLAG_RECS:
            a = r["reference_answer"]
            for must in ("bằng chứng trực tiếp còn hạn chế", "độ chắc chắn chung của bằng chứng cho trẻ em được đánh giá là thấp",
                         "không phải kết luận rằng chất này gây hại cho trẻ", "nhãn không ghi lượng NSS"):
                self.assertIn(must, a, r["id"])
            self.assertNotRegex(a, r"(?i)gây hại cho trẻ nhỏ|không an toàn cho trẻ|\bcấm\b|JECFA", r["id"])


class CaffeineChildTests(unittest.TestCase):
    """EFSA 2015 caffeine opinion: for children the information is insufficient to derive a safe intake (no safe level exists)."""
    FLAG_RECS = [r for r in ROWS if r.get("authority_topic") == "efsa_caffeine_child"]
    CTRL = [r for r in ROWS if r.get("authority_topic") == "efsa_caffeine_child_absent"]
    HINT = re.compile(r"caff?ein|cà phê|ca phe|coffee|\bcafe\b|\btrà\b|\btea\b|ca ?cao|cocoa|chocolate|socola|sô cô la|matcha|"
                      r"guarana|\bcola\b|côla|\bkola\b|tăng lực|energy drink|taurin|\bmate\b|yerba", re.I)

    def test_claim_passed_and_pages(self):
        c = CLAIMS["EFSA_CAFFEINE_2015_CHILDREN"]
        self.assertTrue(c["passed"], c)
        pages = {p for v in c["evidence_pages"].values() for p in (v if isinstance(v, list) else [v])}
        self.assertIn(1, pages)  # the summary sentence about children and adolescents
        for r in self.FLAG_RECS + self.CTRL:
            self.assertIn("EFSA_CAFFEINE_2015_CHILDREN: p.", r["source_claim_pages"], r["id"])

    def test_shape_and_balance(self):
        self.assertGreaterEqual(len(self.FLAG_RECS), 10)
        self.assertEqual(len(self.FLAG_RECS), len(self.CTRL))
        for r in self.FLAG_RECS + self.CTRL:
            self.assertEqual((r["persona_id"], r["domain"], r["tier"], r["eligible_for_scoring"]),
                             ("child_5yo", "vulnerable_groups", "silver", "yes"), r["id"])
            self.assertEqual(r["source_claim_locators"], "EFSA_CAFFEINE_2015#Children", r["id"])
            self.assertEqual(r["source_citations"], "EFSA_CAFFEINE_2015", r["id"])
        self.assertTrue(all(r["gold_safety_label"] == "CAUTION" for r in self.FLAG_RECS))
        self.assertTrue(all(r["gold_safety_label"] == "SAFE" for r in self.CTRL))

    def test_flagged_labels_really_declare_caffeine_or_coffee_and_controls_have_no_hint(self):
        for r in self.FLAG_RECS:
            label = json.loads((ROOT / r["label_transcription_path"]).read_text(encoding="utf-8"))
            ing = json.dumps(label.get("ingredient"), ensure_ascii=False).lower()
            other = json.dumps([label.get("additive"), label.get("warning")], ensure_ascii=False).lower()
            self.assertTrue(re.search(r"caff?ein|cà phê|coffee", ing) or re.search(r"caff?ein", other), r["id"])
            self.assertNotRegex(ing + other, r"decaf|khử caf", r["id"])
        for r in self.CTRL:
            label = json.loads((ROOT / r["label_transcription_path"]).read_text(encoding="utf-8"))
            self.assertIsNone(self.HINT.search(json.dumps(label, ensure_ascii=False)), r["id"])
            self.assertGreaterEqual(len(json.dumps(label.get("ingredient"), ensure_ascii=False)), 30, r["id"])

    def test_answers_follow_the_opinion_and_never_invent_a_child_safe_level(self):
        for r in self.FLAG_RECS:
            a = r["reference_answer"]
            self.assertIn("không đủ để xác định mức caffeine an toàn", a, r["id"])
            self.assertIn("không phải kết luận về mức an toàn cho một trẻ cụ thể", a, r["id"])
            # the 3 mg/kg value is the ADULT single-dose level; it must never be stated as a daily or child limit
            self.assertNotIn("mg/kg thể trọng/ngày", a, r["id"])
            self.assertNotRegex(a, r"(?i)mức an toàn (là|cho trẻ em là)|an toàn cho trẻ|\bcấm\b", r["id"])
