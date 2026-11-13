"""Verify every v5 answer against (1) its question, (2) the label text, (3) the source documents.

No experts are used. Each record is checked by deterministic gates:

  G1 verdict      answer verdict matches label_v5 and does not contradict it
  G2 q_a_align    answer addresses what the question asked (product, allergens, INS, topic)
  G3 label_facts  every fact in the answer (INS codes, numbers, allergens) is in the label text
  G4 citations    tags are recognised, relevant to the topic, and match source_citations_v5
  G5 locators     claim locators exist in the source catalog; label excerpt is verbatim
  G6 source_facts claims about the regulation are checked against the TT24 text / allergen KB

Output: data/build/qa_verification.csv (one row per record, failed gates listed)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_research"))

import create_benchmark_v5 as b5  # noqa: E402
import source_grounding_v5 as sg  # noqa: E402
import additive_names as an  # noqa: E402
from modules.benchmark import RECOGNIZED_CITATION_SOURCES  # noqa: E402

V5 = ROOT / "data" / "build" / "qa_build_full.jsonl"
CATALOG = ROOT / "knowledge_base" / "sources" / "catalog" / "source_citation_catalog.json"
TT24_CODES = ROOT / "knowledge_base" / "sources" / "derived" / "tt24_positive_list_codes.json"
KB_ADDITIVES = ROOT / "knowledge_base" / "additives_standard.json"
OUT_CSV = ROOT / "data" / "build" / "qa_verification.csv"

CITE = re.compile(r"\[([A-Za-z0-9_.]+)\]")
INS_RE = re.compile(r"INS\s?(\d{3,4}[a-z]?(?:\([ivx]+\))?)", re.I)

HEAD_OK = {
    "UNSAFE": ("KHÔNG AN TOÀN", "NGUY HIỂM"),
    "CRITICAL_ALERT": ("CẢNH BÁO KHẨN CẤP", "CẢNH BÁO NGUY HIỂM"),
    "CAUTION": ("THẬN TRỌNG", "CẢNH BÁO", "PHỤ GIA"),
    "SAFE": ("AN TOÀN", "PHỤ GIA", "KHÔNG PHÁT HIỆN"),
    "PERMITTED_LISTED": ("CÓ TRONG DANH MỤC CHO PHÉP",),
    "NOT_DECLARED": ("KHÔNG KÊ KHAI",),
    "ABSTAIN": ("KHÔNG ĐỦ DỮ LIỆU", "CHƯA ĐỦ DỮ LIỆU"),
}
NEGATIVE_WORDS = ("KHÔNG AN TOÀN", "NGUY HIỂM", "TUYỆT ĐỐI KHÔNG", "CHỐNG CHỈ ĐỊNH")
POSITIVE_HEADS = ("AN TOÀN VỀ", "AN TOÀN CHO")

# question topic -> (question regex, answer keywords of which >=1 must appear)
TOPICS = {
    "natri": (r"natri|huyết áp|muối", ("natri",)),
    "đường": (r"đường huyết|đái tháo đường|tiểu đường|lượng đường|đường tổng", ("đường",)),
    "photphat": (r"photphat|phốt pho|phospho|suy thận", ("photphat", "phốt pho", "phospho")),
    "phẩm màu": (r"phẩm màu|tăng động|hiếu động", ("phẩm màu", "southampton", "màu")),
    "pku": (r"phenylketon|pku|phenylalanin", ("phenylalanin", "aspartame", "pku")),
    "thai": (r"mang thai|thai kỳ|thai phụ|bà bầu", ("thai",)),
}


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def norm_code(code: str) -> str:
    return re.sub(r"\s+", "", code.lower().replace("ins", ""))


def label_blob(label: dict) -> str:
    return " ".join(b5.field_text(label.get(k)) for k in ("ingredient", "additive", "warning", "nutrition")).lower()


def codes_in_text(text: str) -> set[str]:
    return {norm_code(m.group(1)) for m in INS_RE.finditer(text)}


_LABEL_CODE = re.compile(r"(?<![\d.])(\d{3,4})\s*([a-z])?\s*((?:\([ivx]+\))*)", re.I)


def squash(code: str) -> str:
    return re.sub(r"[\s()]", "", code.lower())


def label_code_set(label: dict) -> set[str]:
    text = " ".join(b5.field_text(label.get(k)) for k in ("ingredient", "additive", "warning")).lower()
    text = re.sub(r"(?<!\d)\d{1,2}[.,]\d+", " ", text)   # drop decimals such as 0,025 / 3,2 before splitting lists
    text = text.replace(",", " ").replace(";", " ")
    out = set()
    for m in _LABEL_CODE.finditer(text):
        out.add(squash("".join(g or "" for g in m.groups())))
        out.add(m.group(1))  # base number
    return out


def code_in_label(code: str, label_codes: set[str]) -> bool:
    c = squash(code)
    base = re.match(r"\d+", c).group(0)
    return c in label_codes or any(x.startswith(base) and (x.startswith(c) or c.startswith(x)) for x in label_codes)


def head_of(answer: str) -> str:
    return re.split(r"\s[—-]\s", answer, maxsplit=1)[0].strip()


class Context:
    def __init__(self):
        self.catalog = load_json(CATALOG)
        self.catalog_keys = set(self.catalog)
        self.recognized = set()
        for key, meta in RECOGNIZED_CITATION_SOURCES.items():
            self.recognized.add(key)
            self.recognized.update(meta.get("aliases", []))
        self.tt24 = set(load_json(TT24_CODES)["codes"]) if TT24_CODES.exists() else None
        self.kb_codes = {norm_code(x["ins_code"]) for x in load_json(KB_ADDITIVES)}
        self.labels: dict = {}
        self.external = load_json(ROOT / "knowledge_base" / "sources" / "derived" / "external_claim_verification.json")


ALLERGEN_SOURCES = {"US_FDA_FALCPA", "WHO_JEMRA_PRIORITY_ALLERGENS", "FAO_WHO_JEMRA_PART_3",
                    "FAO_WHO_JEMRA_2021", "US_FDA_FASTER_ACT_2023", "US_FDA_ALLERGEN_QA_ED5"}


def check(row: dict, ctx: Context) -> list[str]:
    fails: list[str] = []
    lab = row["label_v5"]
    ans = row["reference_answer_v5"]
    q = row["question_v5"]
    dom = row["domain"]
    name = row["product_name"]
    label = b5.load_label(row["label_transcription_path"], ctx.labels)
    blob = label_blob(label)
    label_codes = label_code_set(label)
    # an additive named without its INS number still counts as declared (see additive_names.py)
    label_codes |= {b for hits in an.name_hits(label).values() for b, _ in hits}
    ans_l = ans.lower()

    # ---------------- G1 verdict ----------------
    if lab not in HEAD_OK:
        fails.append("G1:unknown_label")
        return fails
    head = head_of(ans).upper()
    if not any(head.startswith(h) for h in HEAD_OK[lab]):
        fails.append("G1:head_label_mismatch")
    if lab in ("SAFE", "PERMITTED_LISTED") and any(w in ans.upper() for w in NEGATIVE_WORDS):
        fails.append("G1:positive_label_negative_text")
    if lab in ("UNSAFE", "CRITICAL_ALERT") and any(h in ans.upper()[:80] for h in POSITIVE_HEADS):
        fails.append("G1:negative_label_positive_text")
    if lab == "PERMITTED_LISTED" and "không thể xác nhận" in ans_l and "an toàn tuyệt đối" not in ans_l:
        fails.append("G1:permitted_overclaims")

    # ---------------- G2 question-answer alignment ----------------
    if b5.sanitize_brackets(name).lower() not in ans_l:
        fails.append("G2:product_not_in_answer")
    q_clean = re.sub(r"'[^']*'", " ", q.lower())
    for cls, pat in b5.QUESTION_MAP:
        mm = re.search(pat, q_clean)
        if dom in ("allergen_detection", "insufficient_data") and mm and mm.group(0).strip() not in ans_l:
            # the answer must name what the user asked about (accept class synonyms)
            if not any(re.search(t, ans_l) for t in b5.ALLERGEN_TERMS[cls]):
                fails.append(f"G2:allergen_not_addressed:{cls}")
    q_codes = codes_in_text(q)
    if q_codes and dom != "vulnerable_groups":
        missing = [c for c in q_codes if c not in codes_in_text(ans)]
        if missing:
            fails.append("G2:question_ins_not_in_answer")
    for topic, (qpat, akw) in TOPICS.items():
        if dom == "vulnerable_groups" and re.search(qpat, q_clean):
            if not any(k in ans_l for k in akw):
                fails.append(f"G2:topic_not_addressed:{topic}")

    # ---------------- G3 label facts ----------------
    for c in codes_in_text(ans):
        if dom == "insufficient_data" and "ngoài phạm vi" in ans_l:
            pass
        if not code_in_label(c, label_codes):
            # codes may legitimately be listed as *absent* (Southampton colours); only check asserted presence
            negated = bool(re.search(r"không (chứa|kê khai|ghi nhận)[^.]*" + re.escape(c), ans_l))
            exempt = "southampton" in ans_l or "photphat vô cơ (như" in ans_l or "(như ins" in ans_l
            if negated or exempt:
                # a code may be named as ABSENT from the label, but it must still be a real INS code
                known = ctx.tt24 is not None and (c in ctx.tt24 or any(t.startswith(c + "(") for t in ctx.tt24))
                if not known:
                    fails.append("G3:absent_code_is_not_a_real_ins_code")
                    break
            else:
                fails.append("G3:ins_not_in_label")
                break
    if dom == "allergen_detection":
        ind = b5.independent_allergen_label(row, label)
        ok = (lab == ind) or (lab == "NOT_DECLARED" and ind == "NOT_DECLARED") or \
             (lab == "CAUTION" and ind in ("NOT_DECLARED", "AMBIGUOUS"))
        if not ok:
            fails.append(f"G3:allergen_label_vs_text:{ind}")
    m = re.search(r"(natri|đường)[^.\d]{0,40}?([\d]+[.,]?\d*) ?(mg|g)\b", ans_l)
    if dom == "vulnerable_groups" and m and ("natri" in ans_l[:80] or "đường" in ans_l[:80]):
        nutrition = b5.field_text(label.get("nutrition")).lower().replace(",", ".")
        if m.group(2).replace(",", ".") not in nutrition:
            fails.append("G3:number_not_in_nutrition")

    # ---------------- G4 citations ----------------
    tags = CITE.findall(ans)
    if len(re.findall(r"\[[^\[\]]*\]", ans)) != len(tags):
        fails.append("G4:foreign_bracket_parsed_as_citation")
    if not tags:
        fails.append("G4:no_citation")
    for t in tags:
        if t not in ctx.recognized:
            fails.append(f"G4:unrecognised_tag:{t}")
    tset = set(tags)
    if dom in ("allergen_detection",) or lab == "NOT_DECLARED":
        if not tset & ALLERGEN_SOURCES:
            fails.append("G4:allergen_answer_without_allergen_source")
    elif tset & {"US_FDA_FALCPA", "WHO_JEMRA_PRIORITY_ALLERGENS"} and "pku" not in row.get("persona_id", ""):
        if dom in ("additive_regulation", "vulnerable_groups"):
            fails.append("G4:allergen_source_on_non_allergen_answer")
    if row.get("persona_id") == "pku_patient" and lab == "CRITICAL_ALERT":
        if "US_FDA_21CFR172_804" not in tset:
            fails.append("G4:pku_missing_21cfr172_804")
        if tset & {"US_FDA_FALCPA", "WHO_JEMRA_PRIORITY_ALLERGENS"}:
            fails.append("G4:pku_cites_allergen_law")
    if lab == "PERMITTED_LISTED" and "VN_TT24_2019" not in tset:
        fails.append("G4:permitted_without_tt24")
    if "natri" in head_of(ans).lower() and "USDA_FDC" in tset:
        fails.append("G4:sodium_threshold_cites_usda_fdc")

    # ---------------- G5 locators ----------------
    for loc in [x.strip() for x in (row.get("source_claim_locators_v5") or "").split(";") if x.strip()]:
        if loc not in ctx.catalog_keys:
            fails.append("G5:locator_not_in_catalog")
            break
    ex = b5.excerpt_ok(row, label)
    if ex != "ok" and not (dom == "insufficient_data" and ex == "generic_locator"):
        fails.append(f"G5:label_excerpt:{ex}")

    # ---------------- G6 source facts ----------------
    if lab == "PERMITTED_LISTED":
        if ctx.tt24 is None:
            fails.append("G6:tt24_not_loaded")
        else:
            codes = [norm_code(c) for c in re.split(r"[,\s]+", row.get("ins_codes_matched", "")) if c]
            codes = [c for c in codes if c]
            absent = [c for c in codes if not (c in ctx.tt24 or any(t.startswith(c + "(") for t in ctx.tt24))]
            if absent:
                fails.append("G6:code_not_in_tt24_positive_list")
    if dom == "allergen_detection" and lab in ("UNSAFE", "CRITICAL_ALERT") and "US_FDA_FALCPA" in tset:
        # FALCPA covers only the major allergens; molluscs / generic seafood are outside it
        asked = b5.asked_allergens(q)
        if asked == ["động vật có vỏ"] and "WHO_JEMRA_PRIORITY_ALLERGENS" not in tset:
            fails.append("G6:falcpa_does_not_cover_molluscs")
    # authority-flag records: re-derive the flag straight from the label text, independent of the v4 linker output
    scope = row.get("label_scope", "")
    if scope.startswith("authority_flag:"):
        groups = {"efsa_2019_phosphate_group": sg.PHOS_BASES, "efsa_2008_mccann_substances": sg.MCCANN_BASES,
                  "cfr_172_804_aspartame": sg.ASPARTAME_BASES, "eu_annex_v_colours": sg.ANNEXV_BASES,
                  "fda_red3_revoked": sg.RED3_BASES, "iarc_v134_aspartame": sg.IARC_ASPARTAME_BASES,
                  "codex_sulphite_declaration": sg.SULPHITE_BASES, "who_nss_pregnancy": sg.NSS_BASES}
        key = scope.split(":", 1)[1]
        if key in groups:
            raw_txt = re.sub(r"(?<!\d)\d{1,2}[.,]\d+", " ", " ".join(b5.field_text(label.get(k)) for k in ("ingredient", "additive", "warning")).lower())
            raw_txt = raw_txt.replace(",", " ").replace(";", " ")
            decl = {m.group(1) for m in re.finditer(r"(?<![\d.])(\d{3,4})(?![\d])(?!\s?(?:mg|g|ml|kg|l|kcal|kj|%|ppm)\b)", raw_txt)}
            name_group = {"efsa_2019_phosphate_group": "phosphate", "efsa_2008_mccann_substances": "mccann",
                          "cfr_172_804_aspartame": "aspartame", "eu_annex_v_colours": "mccann",
                          "fda_red3_revoked": "red3", "iarc_v134_aspartame": "aspartame",
                          "codex_sulphite_declaration": "sulphite", "who_nss_pregnancy": "aspartame"}[key]
            decl |= {b for b, _ in an.name_hits(label).get(name_group, [])}
            hit = bool(decl & groups[key])
            should_flag = row["label_v5"] in ("CAUTION", "CRITICAL_ALERT")
            if key == "who_nss_pregnancy" and not should_flag:
                # a "no NSS" control: independently re-scan the WHOLE transcription for any sweetener hint
                whole = json.dumps(label, ensure_ascii=False).lower()
                if re.search(r"aspartam|acesulfam|sucralos|sacchar|sacar|cyclamat|neotam|advantam|stevi|rebaudi|ngọt|sweet|"
                             r"thaumatin|(?<![\d.,])9(?:5[0-5]|57|6[0-2]|69)(?!\d)", whole):
                    fails.append("G3:nss_control_but_sweetener_hint_on_label")
                if len(b5.field_text(label.get("ingredient")).strip()) < 30:
                    fails.append("G3:nss_control_without_ingredient_list")
            if hit != should_flag:
                fails.append(f"G3:authority_flag_vs_label_text:{key}")
    if scope == "sugar_free_codex_gl23_per_100g":
        # independent re-derivation of the sugars verdict straight from the label (own parser, not source_grounding's)
        basis = label.get("sodium_basis") or {}
        names = [x for x in (label.get("nutrition") or []) if isinstance(x, dict)
                 and re.search(r"đường|sugar", x.get("name", ""), re.I)
                 and not re.search(r"bổ sung|thêm|added|huyết|lactose|glucose|fructose|sucrose", x.get("name", ""), re.I)]
        num = re.match(r"\s*(\d+(?:[.,]\d+)?)", str(names[0].get("value", ""))) if len(names) == 1 else None
        if not (basis and num):
            fails.append("G3:sugar_basis_or_value_missing")
        else:
            per100 = float(num.group(1).replace(",", ".")) / float(basis["amount"]) * 100
            if (lab == "SAFE") != (per100 <= 0.5):
                fails.append("G3:sugar_verdict_vs_label_value")
            if lab not in ("SAFE", "CAUTION"):
                fails.append("G3:sugar_label_not_safe_or_caution")
        if "CODEX_CAC_GL_23_1997" not in tset:
            fails.append("G6:sugar_without_codex_gl23_tag")
    if re.search(r"KDOQI|KDIGO|ADA 20|Southampton study", ans):
        fails.append("G6:unverified_guideline_named")
    # every tag on a scored record must be backed by a claim that was checked against its primary text
    if row["tier"] in ("core", "silver"):
        for t in tset:
            if t not in sg.VERIFIED_TAGS:
                fails.append(f"G6:tag_without_verified_source:{t}")
        for cid in [c for c in (row.get("external_claims") or "").split(";") if c]:
            if cid not in ctx.external or not ctx.external[cid]["passed"]:
                fails.append(f"G6:external_claim_failed:{cid}")
    if row["tier"] in ("core", "silver"):
        ext_ids = [c for c in (row.get("external_claims") or "").split(";") if c]
        needs = {"JECFA_EVALUATIONS": "JECFA_ADI_", "CODEX_STAN_192_1995": "CODEX192_",
                 "CODEX_CXS_1_1985": "CXS1_", "CODEX_CXG_36_1989": "CODEX_CXG36_",
                 "EFSA_OPINIONS": "EFSA_OPINION_", "US_FDA_21CFR172_804": "CFR_172_804",
                 "EFSA_2021_TITANIUM_DIOXIDE_E171": "EFSA_2021_TIO2",
                 "US_FDA_ALLERGEN_QA_ED5": "FDA_QA5_", "CODEX_CAC_GL_23_1997": "CODEX_GL23_",
                 **sg.AUTHORITY_TAG_PREFIX}
        for tag, prefix in needs.items():
            if tag in tset and not any(c.startswith(prefix) for c in ext_ids):
                fails.append(f"G6:{tag}_tag_without_matching_claim")
        if re.search(r"USDA", ans):
            fails.append("G6:usda_named_in_answer")
        if re.search(r"Codex|CXS|CXG", ans) and not any(t.startswith("CODEX") for t in tset):
            fails.append("G6:codex_named_without_tag")
        if re.search(r"JECFA", ans) and not any(c.startswith("JECFA_ADI_") for c in ext_ids):
            fails.append("G6:jecfa_named_without_claim")
        if re.search(r"EFSA", ans) and not any(c.startswith(("EFSA_2021_TIO2", "EFSA_OPINION_")) for c in ext_ids):
            fails.append("G6:efsa_named_without_claim")
        # independent re-check of Codex CXS 192 membership straight from the parsed standard
        mc = re.search(r"Trong Codex CXS 192-1995 \(GSFA, bản 2026\): (.*?)\. \[CODEX_STAN_192_1995\]", ans)
        if mc:
            listed = re.findall(r"INS ([0-9a-z(),\s]+?) (?:nằm trong Table Three|có trong Table One)", mc.group(1))
            t3 = set(json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "codex_192_ins_codes.json").read_text(encoding="utf-8"))["table_three"])
            t1 = set(json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "codex_192_ins_codes.json").read_text(encoding="utf-8"))["table_one"])
            part3 = re.search(r"INS ([^;]*?) nằm trong Table Three", mc.group(1))
            part1 = re.search(r"INS ([^;]*?) có trong Table One", mc.group(1))
            def member(c, tab):
                return c in tab or any(x.startswith(c + "(") for x in tab)
            for grp, tab in ((part3, t3), (part1, t1 | t3)):
                if grp:
                    for c in [x.strip().replace(" ", "") for x in grp.group(1).split(",") if x.strip()]:
                        if not member(c, tab):
                            fails.append("G6:codex192_membership_false")
            if not listed:
                fails.append("G6:codex192_sentence_unparsed")
    mcx = re.search(r"Codex CXS 1-1985 \(mục 4\.2\.1\.4\) quy định luôn phải khai báo (.*?) là thực phẩm", ans)
    if mcx and re.search(r"đậu nành|động vật có vỏ|thân mềm", mcx.group(1)):
        fails.append("G6:codex_cxs1_mandatory_overclaim")
    m_j = re.search(r"JEMRA \(2021, Part 1\) khuyến nghị (.*?) thuộc danh sách", ans)
    if m_j and re.search(r"đậu nành|động vật có vỏ|thân mềm", m_j.group(1)):
        fails.append("G6:jemra_priority_overclaim")
    m_f = re.search(r"321\(qq\) \(FALCPA\), (.*?) thuộc nhóm dị nguyên chính", ans)
    if m_f and re.search(r"động vật có vỏ|thân mềm", m_f.group(1)):
        fails.append("G6:falcpa_overclaim")
    if row["abstain_reason"] == "KB_OUT_OF_SCOPE":
        codes = {c for c in codes_in_text(ans)}
        if codes and all(c in ctx.kb_codes for c in codes):
            fails.append("G6:abstain_kb_but_codes_in_kb")
    if row["abstain_reason"] == "PAL_NOT_ON_LABEL":
        ind = b5.independent_allergen_label(row, label)
        if ind not in ("NOT_DECLARED", None):
            fails.append("G6:abstain_pal_but_allergen_present")
    return fails


def main():
    rows = [json.loads(l) for l in V5.read_text(encoding="utf-8").splitlines() if l.strip()]
    ctx = Context()
    results = []
    for r in rows:
        f = check(r, ctx)
        results.append({"id": r["id"], "tier": r["tier"], "label_v5": r["label_v5"], "domain": r["domain"],
                        "failed_gates": ";".join(f)})
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    fail_rows = [r for r in results if r["failed_gates"]]
    print(f"records={len(results)} failing={len(fail_rows)}")
    c = Counter(g.split(":")[0] + ":" + g.split(":")[1] for r in fail_rows for g in r["failed_gates"].split(";"))
    for k, v in c.most_common():
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main()
