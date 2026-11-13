"""Invariants for the v5 benchmark and sensitivity of its answer verifier (unittest, no pytest needed)."""

import copy
import json
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_research"))

V5_FULL = ROOT / "data" / "build" / "qa_build_full.jsonl"
V5_ACTIVE = ROOT / "data" / "qa_benchmark.jsonl"
V4 = ROOT / "data" / "build" / "qa_input_source_verified.jsonl"


def _load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@unittest.skipUnless(V5_ACTIVE.exists(), "v5 not built")
class BenchmarkV5InvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = _load(V5_FULL)
        cls.active = _load(V5_ACTIVE)
        cls.scored = [r for r in cls.active if r["eligible_for_scoring"] == "yes"]

    def test_ids_preserved_from_v4(self):
        self.assertEqual([r["id"] for r in self.active], [r["id"] for r in _load(V4)])

    def test_scored_records_pass_every_gate_and_are_not_expert_claims(self):
        for r in self.scored:
            self.assertEqual(r["verify_failed_gates"], "", r["id"])
            self.assertEqual(r["review_status"], "approved_rule_verified", r["id"])
            self.assertEqual(r["expert_validated"], "false", r["id"])

    def test_scored_labels_use_consumer_label_set(self):
        allowed = {"SAFE", "CAUTION", "UNSAFE", "CRITICAL_ALERT", "ABSTAIN"}
        for r in self.scored:
            self.assertIn(r["gold_safety_label"], allowed, r["id"])

    def test_not_scored_records_are_not_eligible_for_evaluation(self):
        from modules.benchmark_manifest import is_evaluation_eligible
        for r in self.active:
            self.assertEqual(is_evaluation_eligible(r), r["eligible_for_scoring"] == "yes", r["id"])

    def test_permitted_listed_answer_not_contradictory(self):
        for r in self.scored:
            if r["label_v5"] == "PERMITTED_LISTED":
                self.assertIn("an toàn tuyệt đối", r["reference_answer"])
                self.assertNotIn("AN TOÀN", r["reference_answer"].split("—")[0])

    def test_every_answer_has_citation_tag(self):
        for r in self.scored:
            self.assertTrue(re.search(r"\[[A-Za-z0-9_.]+\]", r["reference_answer"]), r["id"])

    def test_known_v4_defects_are_corrected(self):
        by_id = {r["id"]: r for r in self.active}
        expected = {"FOODQA_G3_2058": "UNSAFE", "FOODQA_G3_0256": "CRITICAL_ALERT", "FOODQA_G3_1613": "UNSAFE",
                    "FOODQA_G3_0042": "UNSAFE", "FOODQA_G3_0351": "CRITICAL_ALERT",
                    "FOODQA_G3_0257": "UNSAFE", "FOODQA_G3_1413": "UNSAFE"}
        for rid, label in expected.items():
            self.assertEqual(by_id[rid]["label_v5"], label, rid)
        for rid in ("FOODQA_G3_0861", "FOODQA_G3_1493"):
            self.assertEqual(by_id[rid]["label_v5"], "CAUTION", rid)
        # fake INS code derived from "0,025 %" (and other stale codes) must never be scored: such records were removed
        for rid in ("FOODQA_G3_1585", "FOODQA_G3_0114", "FOODQA_G3_1650", "FOODQA_G3_1742"):
            self.assertNotIn(rid, by_id)
        self.assertEqual(by_id["FOODQA_G3_1147"]["label_v5"], "UNSAFE")

    def test_external_claims_were_checked_against_primary_text(self):
        claims = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "external_claim_verification.json").read_text(encoding="utf-8"))
        for cid, c in claims.items():
            self.assertTrue(c["passed"], cid)
            self.assertTrue((ROOT / "knowledge_base" / "sources").rglob(c["source_file"]), cid)

    def test_scored_records_cite_only_tags_with_verified_sources(self):
        import source_grounding_v5 as sg
        for r in self.scored:
            tags = set(re.findall(r"\[([A-Za-z0-9_.]+)\]", r["reference_answer"]))
            self.assertTrue(tags, r["id"])
            self.assertLessEqual(tags, sg.VERIFIED_TAGS, r["id"])

    def test_no_answer_claims_soy_or_molluscs_as_jemra_priority_or_molluscs_under_falcpa(self):
        for r in self.scored:
            m = re.search(r"JEMRA \(2021, Part 1\) khuyến nghị (.*?) thuộc danh sách", r["reference_answer"])
            if m:
                self.assertNotRegex(m.group(1), r"đậu nành|động vật có vỏ|thân mềm", r["id"])
            f = re.search(r"321\(qq\) \(FALCPA\), (.*?) thuộc nhóm dị nguyên chính", r["reference_answer"])
            if f:
                self.assertNotRegex(f.group(1), r"động vật có vỏ|thân mềm", r["id"])

    def test_records_that_depend_on_unobtainable_documents_are_not_scored(self):
        for r in self.active:
            if r["tier"] == "excluded_source_not_verifiable":
                self.assertEqual(r["eligible_for_scoring"], "no", r["id"])
        for r in self.scored:
            self.assertNotRegex(r["reference_answer"], r"USDA_FDC", r["id"])
            if "EFSA_OPINIONS" in r["reference_answer"]:  # allowed only with a claim read from the EFSA full text
                self.assertIn("EFSA_OPINION_", r["external_claims"], r["id"])
            if "JECFA_EVALUATIONS" in r["reference_answer"]:  # allowed only with a verified JECFA record behind it
                self.assertIn("JECFA_ADI_", r["external_claims"], r["id"])

    def test_permitted_codes_exist_in_tt24_pdf_extract(self):
        codes = set(json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "tt24_positive_list_codes.json").read_text(encoding="utf-8"))["codes"])
        self.assertGreaterEqual(len(codes), 400)
        for r in self.scored:
            if r["label_v5"] == "PERMITTED_LISTED":
                for c in [x.strip().lower().replace("ins", "").replace(" ", "") for x in r["ins_codes_matched"].split(",") if x.strip()]:
                    self.assertTrue(c in codes or any(t.startswith(c + "(") for t in codes), (r["id"], c))

    def test_jecfa_answers_match_the_fetched_jecfa_records(self):
        jecfa = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "jecfa_adi_verification.json").read_text(encoding="utf-8"))
        more = ROOT / "knowledge_base" / "sources" / "derived" / "jecfa_adi_verification_more.json"
        if more.exists():
            jecfa.update(json.loads(more.read_text(encoding="utf-8")))
        n = 0
        for r in self.scored:
            if "JECFA_EVALUATIONS" in r["reference_answer"]:
                n += 1
                ids = [c for c in r["external_claims"].split(";") if c.startswith("JECFA_ADI_")]
                self.assertTrue(ids, r["id"])
                for cid in ids:
                    rec = jecfa[cid.replace("JECFA_ADI_", "")]
                    self.assertTrue(rec["passed"], cid)
                    adi = re.split(r"\s+(?:Intake|Comments?:)", rec["latest_adi"].strip())[0].strip()
                    self.assertTrue(f"ADI '{adi}'" in r["reference_answer"] or f"ADI '{adi.capitalize()}'" in r["reference_answer"], r["id"])
        self.assertGreater(n, 0)

    def test_codex_statements_are_backed_by_the_parsed_codex_documents(self):
        codex = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "codex_192_ins_codes.json").read_text(encoding="utf-8"))
        t1, t3 = set(codex["table_one"]), set(codex["table_three"])
        self.assertGreater(len(t1), 250)
        self.assertGreater(len(t3), 150)
        n = 0
        for r in self.scored:
            self.assertNotIn("USDA", r["reference_answer"], r["id"])
            if "CODEX_STAN_192_1995" in r["reference_answer"]:
                n += 1
                self.assertIn("CODEX192_TABLE1_TABLE3_PARSED", r["external_claims"], r["id"])
                m = re.search(r"INS ([^;]*?) nằm trong Table Three", r["reference_answer"])
                if m:
                    for c in [x.strip() for x in m.group(1).split(",")]:
                        self.assertTrue(c in t3 or any(t.startswith(c + "(") for t in t3), (r["id"], c))
            if "CODEX_CXS_1_1985" in r["reference_answer"]:
                self.assertIn("CXS1_", r["external_claims"], r["id"])
        self.assertGreater(n, 300)

    def test_soy_and_molluscs_are_never_called_mandatory_allergens_by_codex(self):
        for r in self.scored:
            m = re.search(r"CXS 1-1985 \(mục 4\.2\.1\.4\) quy định luôn phải khai báo (.*?) là thực phẩm", r["reference_answer"])
            if m:
                self.assertNotRegex(m.group(1), r"đậu nành|động vật có vỏ|thân mềm", r["id"])

    def test_v4_table_three_claim_was_wrong_for_codes_only_in_table_one(self):
        """Codes such as 211, 202, 951 are in CXS 192 Table One only, so the v4 'Table3_GMP' locator was wrong for them."""
        codex = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "codex_192_ins_codes.json").read_text(encoding="utf-8"))
        for c in ("211", "202", "951"):
            self.assertIn(c, codex["table_one"])
            self.assertNotIn(c, codex["table_three"])

    def test_nutrient_threshold_and_benzene_records_are_excluded_for_the_documented_reason(self):
        # records without a verifiable source were removed from the benchmark: nothing unscored may remain
        self.assertEqual([r["id"] for r in self.active if r["eligible_for_scoring"] == "no"], [])
        for r in self.active:
            self.assertNotIn("USDA", r["reference_answer_v5"], r["id"])
        for r in self.active:
            if r["persona_id"] == "diabetes_hypertension":
                if r["label_scope"].startswith(("single_factor:natri_vs_codex_low_threshold_per_100g",
                                                "sugar_free_codex_gl23_per_100g")):
                    # sodium / sugars records are scored only against the Codex per-100 g threshold, on a label basis that was read
                    self.assertEqual(r["eligible_for_scoring"], "yes", r["id"])
                    self.assertIn("[CODEX_CAC_GL_23_1997]", r["reference_answer_v5"], r["id"])
                    self.assertNotIn("USDA", r["reference_answer_v5"], r["id"])
                else:
                    self.assertEqual(r["eligible_for_scoring"], "no", r["id"])
        claims = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "external_claim_verification.json").read_text(encoding="utf-8"))
        for cid in ("CFR_101_61_SODIUM_PER_RACC", "CFR_101_54_HIGH_20PCT_DV", "FDA_BENZENE_BEVERAGES_ONLY",
                    "CODEX_GL23_SODIUM_PER_100G"):
            self.assertTrue(claims[cid]["passed"], cid)

    def test_most_labels_do_not_state_the_basis_of_the_nutrition_table(self):
        import glob
        none = 0
        files = glob.glob(str(ROOT / "assets" / "labels" / "*.json"))
        for f in files:
            t = json.dumps(json.load(open(f, encoding="utf-8")).get("nutrition"), ensure_ascii=False).lower()
            if not re.search(r"100\s?(g|ml)|khẩu phần|serving|per serv|trên 1|mỗi", t):
                none += 1
        self.assertGreater(none / len(files), 0.9)

    def test_kb_dependent_and_unsourced_clinical_records_are_not_scored(self):
        for r in self.active:
            if r["abstain_reason"] == "KB_OUT_OF_SCOPE":
                self.assertEqual(r["eligible_for_scoring"], "no", r["id"])
            if r["domain"] == "vulnerable_groups" and r["eligible_for_scoring"] == "yes":
                # a clinical-group record is scored only when its answer is restricted to a verified authority text
                if r["label_scope"].startswith(("single_factor:natri_vs_codex_low_threshold_per_100g",
                                                "sugar_free_codex_gl23_per_100g")):
                    self.assertEqual(r["persona_id"], "diabetes_hypertension", r["id"])
                    continue
                self.assertTrue(r["label_scope"].startswith("authority_flag:"), r["id"])
                self.assertIn(r["persona_id"], ("ckd_stage3", "child_5yo", "pku_patient", "pregnant_woman"), r["id"])

    def test_authority_flag_answers_quote_only_verified_claims(self):
        claims = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "external_claim_verification.json").read_text(encoding="utf-8"))
        for cid in ("EFSA_OPINION_2019_PHOSPHATES", "EFSA_OPINION_2008_MCCANN", "EFSA_OPINION_2017_NITRITE",
                    "EFSA_2021_TIO2_FULLTEXT", "CFR_172_804_PKU_STATEMENT"):
            self.assertTrue(claims[cid]["passed"], cid)
        n = 0
        for r in self.scored:
            if r["label_scope"].startswith("authority_flag:"):
                n += 1
                self.assertNotRegex(r["reference_answer"], r"KDOQI|KDIGO|90.100%", r["id"])
                self.assertTrue(r["external_claims"], r["id"])
        self.assertGreater(n, 250)

    def test_known_v4_clinical_errors_are_corrected(self):
        by_id = {r["id"]: r for r in self.active}
        # label declares a dipotassium phosphate written as [340(ii)] that the v4 linker missed
        self.assertEqual(by_id["FOODQA_G3_1998"]["label_v5"], "CAUTION")
        self.assertIn("340", by_id["FOODQA_G3_1998"]["reference_answer"])
        # sodium benzoate (E211) was one of the substances tested by McCann et al., so it must flag the child record
        self.assertEqual(by_id["FOODQA_G3_0037"]["label_v5"], "CAUTION")
        # "450 mg / 100 ml" is a quantity, not INS 450
        self.assertEqual(by_id["FOODQA_G3_0393"]["label_v5"], "SAFE")


@unittest.skipUnless(V5_FULL.exists(), "v5 not built")
class AnswerVerifierSensitivityTests(unittest.TestCase):
    """The verifier must reject deliberately corrupted records, otherwise 'verified' means nothing."""

    @classmethod
    def setUpClass(cls):
        import verify_v5_answers as ver
        cls.ver = ver
        cls.ctx = ver.Context()
        rows = _load(V5_FULL)
        cls.clean = [r for r in rows if r["tier"] in ("core", "silver") and not ver.check(r, cls.ctx)]
        random.seed(11)

    def _rate(self, pred, mutate, n=80):
        pool = [r for r in self.clean if pred(r)]
        sample = random.sample(pool, min(n, len(pool)))
        hits = 0
        for r in sample:
            m = copy.deepcopy(r)
            mutate(m)
            hits += bool(self.ver.check(m, self.ctx))
        return hits / len(sample)

    def test_label_flip_is_detected(self):
        flip = {"UNSAFE": "SAFE", "CRITICAL_ALERT": "SAFE", "SAFE": "UNSAFE", "PERMITTED_LISTED": "UNSAFE",
                "NOT_DECLARED": "UNSAFE", "ABSTAIN": "SAFE", "CAUTION": "SAFE"}
        self.assertGreaterEqual(self._rate(lambda r: True, lambda m: m.update(label_v5=flip[m["label_v5"]])), 0.95)

    def test_missing_product_name_is_detected(self):
        self.assertGreaterEqual(self._rate(
            lambda r: True,
            lambda m: m.update(reference_answer_v5=m["reference_answer_v5"].replace(m["product_name"], "sản phẩm"))), 0.95)

    def test_fabricated_ins_code_is_detected(self):
        self.assertGreaterEqual(self._rate(
            lambda r: re.search(r"INS\s?\d{3}", r["reference_answer_v5"]) and r["domain"] != "insufficient_data",
            lambda m: m.update(reference_answer_v5=re.sub(r"INS\s?\d{3,4}[a-z]?(\([ivx]+\))?", "INS8888",
                                                          m["reference_answer_v5"], count=1, flags=re.I))), 0.95)

    def test_unknown_citation_is_detected(self):
        self.assertGreaterEqual(self._rate(
            lambda r: True,
            lambda m: m.update(reference_answer_v5=re.sub(r"\[[A-Za-z0-9_.]+\]", "[MADE_UP_SOURCE]",
                                                          m["reference_answer_v5"], count=1))), 0.95)

    def test_off_topic_answer_is_detected(self):
        self.assertGreaterEqual(self._rate(
            lambda r: r["domain"] == "allergen_detection",
            lambda m: m.update(reference_answer_v5="CẢNH BÁO — Phụ gia INS171 không phù hợp. [EFSA_OPINIONS]")), 0.95)

    def test_code_outside_tt24_positive_list_is_detected(self):
        self.assertGreaterEqual(self._rate(
            lambda r: r["label_v5"] == "PERMITTED_LISTED",
            lambda m: m.update(ins_codes_matched="INS88889, " + m["ins_codes_matched"])), 0.95)

    def test_wrong_codex_table_three_membership_is_detected(self):
        def mutate(m):  # claim that a Table-One-only code (211) is in Table Three
            m["reference_answer_v5"] = m["reference_answer_v5"].replace("nằm trong Table Three", "nằm trong Table Three", 1)
            m["reference_answer_v5"] = re.sub(r"INS ([0-9a-z(),\s]+?) nằm trong Table Three", r"INS , 211 nằm trong Table Three",
                                              m["reference_answer_v5"], count=1)
        self.assertGreaterEqual(self._rate(lambda r: "nằm trong Table Three" in r["reference_answer_v5"], mutate), 0.95)

    def test_soy_called_codex_mandatory_is_detected(self):
        def mutate(m):
            m["reference_answer_v5"] = re.sub(r"quy định luôn phải khai báo ", "quy định luôn phải khai báo đậu nành, ",
                                              m["reference_answer_v5"], count=1)
        self.assertGreaterEqual(self._rate(lambda r: "quy định luôn phải khai báo" in r["reference_answer_v5"], mutate), 0.95)

    def test_codex_tag_without_claim_is_detected(self):
        self.assertGreaterEqual(self._rate(
            lambda r: "CODEX_CXS_1_1985" in r["reference_answer_v5"],
            lambda m: m.update(external_claims=";".join(c for c in m["external_claims"].split(";") if not c.startswith("CXS1_")))), 0.95)

    def test_flag_template_swapped_against_the_label_is_detected(self):
        """Give a record the answer template of the opposite flag: the label text must contradict it."""
        by_scope = {}
        for r in self.clean:
            if r["label_scope"].startswith("authority_flag:efsa_2019_phosphate_group"):
                by_scope.setdefault(r["label_v5"], r)
        safe, caution = by_scope["SAFE"], by_scope["CAUTION"]

        def mutate(m):
            donor = safe if m["label_v5"] == "CAUTION" else caution
            m["reference_answer_v5"] = donor["reference_answer_v5"].replace(donor["product_name"], m["product_name"])
            m["label_v5"] = donor["label_v5"]
        self.assertGreaterEqual(self._rate(
            lambda r: r["label_scope"].startswith("authority_flag:efsa_2019_phosphate_group"), mutate), 0.95)


if __name__ == "__main__":
    unittest.main()
