"""Append the additive records backed by EU Reg 1333/2008 Annex V, the FDA Red No. 3 order and IARC Vol. 134.

For each topic, every label that really declares one of the group's INS codes gets one new record (row cloned from
an existing record of the same photo, so image/label fields are consistent). The question, answer, verdict and
claims are written later by create_benchmark_v5.build() (rows carry `authority_topic`); verify_v5_answers gates them.

Idempotent: earlier authority rows are removed from the input before the new ones are added.
Run order: this script -> create_benchmark_v5.py -> verify_v5_answers.py -> finalize_benchmark_v5.py --activate
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_research"))

import create_benchmark_v5 as b5  # noqa: E402
import source_grounding_v5 as sg  # noqa: E402

SRC = b5.SRC
TOPICS = {"eu_annex_v_colours": sg.ANNEXV_BASES, "fda_red3_revoked": sg.RED3_BASES,
          "iarc_v134_aspartame": sg.IARC_ASPARTAME_BASES, "codex_sulphite_declaration": sg.SULPHITE_BASES,
          "who_nss_pregnancy": sg.NSS_BASES,
          "who_nss_pregnancy_absent": sg.NSS_BASES}  # keep new topics LAST so earlier record ids stay stable
# Topics whose records belong to a vulnerable group instead of the default healthy adult.
_PREG = ("vulnerable_groups", "Nhóm dễ tổn thương", "pregnant_woman", "Phụ nữ mang thai")
TOPIC_PERSONA = {"who_nss_pregnancy": _PREG, "who_nss_pregnancy_absent": _PREG}
CONTROL_SEED = 2026  # fixed seed: the control sample is reproducible
FIRST_NEW_ID = 2101
COVERAGE_SEED = 2027
LISTING_PER_CODE = 2   # at most this many listing questions per previously-unasked INS code
ADI_PER_CODE = 1       # at most this many JECFA-ADI questions per code (only if the code has a verified JECFA record)
TT24_CODES = ROOT / "knowledge_base" / "sources" / "derived" / "tt24_positive_list_codes.json"


def _base_of(code: str) -> str:
    return sg.code_base(code)


def ins_coverage_rows(rows, by_label, cache, n):
    """Questions about INS codes that no existing record asks about, one code per question.

    Candidate = a code token printed on a label (strict_label_codes), canonicalised, whose base number is not asked by any
    existing record, inside the real INS range and listed in the TT24 positive list. The question/answer/label are built by
    create_benchmark_v5 from verified sources only; here we only choose which (label, code) pairs to ask about.
    """
    tt = set(json.loads(TT24_CODES.read_text(encoding="utf-8"))["codes"])
    asked = set()
    for r in rows:
        for c in (r.get("ins_codes_matched") or "").split(","):
            if c.strip():
                asked.add(_base_of(c))
        for c in b5._Q_CODE_LIST.findall(r.get("question") or ""):
            asked.update(_base_of(x) for x in c.split(","))
    cand: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(by_label):
        label = b5.load_label(path, cache)
        for tok in dict.fromkeys(filter(None, (sg.canon_code(t) for t in sg.strict_label_codes(label)))):
            base = _base_of(tok)
            if base in asked or not (100 <= int(base) <= 1599) or not sg.codex_member(tok, tt):
                continue
            cand.setdefault(base, []).append((path, tok))
    dom_vi = {r["domain"]: r["domain_vi"] for r in rows}
    out = []
    rng = random.Random(COVERAGE_SEED)
    for base in sorted(cand, key=int):
        pairs = list(cand[base])
        rng.shuffle(pairs)
        listing = pairs[:LISTING_PER_CODE]
        adi = [(p, t) for p, t in pairs[LISTING_PER_CODE:] + pairs[:LISTING_PER_CODE]
               if sg.jecfa_status_answer("x", ["INS" + t])][:ADI_PER_CODE]
        for topic, chosen in (("ins_listing", listing), ("ins_adi", adi)):
            for path, tok in chosen:
                label = b5.load_label(path, cache)
                pointer = next((f for f in ("additive", "ingredient") if b5.field_text(label.get(f)).strip()
                                and any(sg.canon_code(t) == tok for t in sg.strict_label_codes({f: label.get(f)}))),
                               "additive")
                dom = "additive_regulation" if topic == "ins_listing" else "insufficient_data"
                base_row = dict(by_label[path])
                base_row.update({
                    "id": f"FOODQA_G3_{n}", "authority_topic": topic, "domain": dom,
                    "domain_vi": dom_vi.get(dom, base_row.get("domain_vi", "")),
                    "persona_id": "healthy_adult", "persona_name": "Người trưởng thành khỏe mạnh",
                    "ins_codes_matched": "INS" + tok, "question": "", "reference_answer": "",
                    "gold_safety_label": "SAFE", "binary_answer": "Yes",
                    "label_locator": f"{path}#/{pointer}", "label_excerpt": b5.field_text(label.get(pointer)),
                    "review_status": "approved_source_verified", "source_citations": "", "source_claim_locators": "",
                    "review_notes": f"Added for {topic}: INS coverage; question and answer are generated and gated in v5."})
                out.append(base_row)
                n += 1
    return out


CHILD_ORDER = ["who_nss_child", "who_nss_child_absent", "efsa_caffeine_child", "efsa_caffeine_child_absent"]


def child_rows(by_label, cache, n):
    """Questions from a 5-year-old's parent: WHO 2023 NSS guideline (children) and EFSA 2015 caffeine opinion. Every
    label that declares the substance gets one flagged record; each flag topic gets as many no-substance controls
    (seeded sample), so the two classes are balanced. Whether a label qualifies is decided by sg.child_authority_record,
    the same function create_benchmark_v5 uses to write the answer."""
    out, flagged = [], {}
    for topic in CHILD_ORDER:
        control = topic.endswith("_absent")
        qualifying = [p for p in sorted(by_label)
                      if sg.child_authority_record(topic, "x", b5.load_label(p, cache))]
        if control:
            paths = sorted(random.Random(CONTROL_SEED).sample(qualifying, min(flagged[topic[:-7]], len(qualifying))))
        else:
            paths = qualifying
            flagged[topic] = len(paths)
        for path in paths:
            label = b5.load_label(path, cache)
            name = by_label[path]["product_name"]
            question, _answer, _claims, _scope, verdict, codes = sg.child_authority_record(topic, name, label)
            if control:
                pointer = "ingredient"
            elif topic == "who_nss_child":
                pointer = next((f for f in ("additive", "ingredient") if b5.field_text(label.get(f)).strip()
                                and sg.declared_bases({f: label.get(f)}, sg.NSS_BASES)), "additive")
            else:
                pointer = next((f for f in ("ingredient", "additive", "warning")
                                if sg.caffeine_terms({f: label.get(f)})), "ingredient")
            dom, dom_vi, pid, pname = sg.CHILD_PERSONA
            base = dict(by_label[path])
            base.update({
                "id": f"FOODQA_G3_{n}", "authority_topic": topic, "domain": dom, "domain_vi": dom_vi,
                "persona_id": pid, "persona_name": pname, "ins_codes_matched": codes,
                "question": "", "reference_answer": "", "gold_safety_label": verdict,
                "binary_answer": "Yes" if verdict == "SAFE" else "No",
                "label_locator": f"{path}#/{pointer}", "label_excerpt": b5.field_text(label.get(pointer)),
                "review_status": "approved_source_verified", "source_citations": "", "source_claim_locators": "",
                "review_notes": f"Added for {topic}; question and answer are generated and gated in v5."})
            out.append(base)
            n += 1
    return out


def sugar_rows(by_label, cache, n):
    """Diabetes-persona questions on the Codex 'sugars free' threshold, only for labels whose nutrition table states its
    basis (sodium_basis, shared by the whole table) and has a plain total-sugars value."""
    out = []
    for path in sorted(by_label):
        label = b5.load_label(path, cache)
        entry = sg.sugar_entry(label)
        if not sg.sugar_codex_answer("x", label):
            continue
        base_row = dict(by_label[path])
        base_row.update({
            "id": f"FOODQA_G3_{n}", "authority_topic": "sugar_codex", "domain": "vulnerable_groups",
            "domain_vi": "Nhóm dễ tổn thương", "persona_id": "diabetes_hypertension",
            "persona_name": "Bệnh nhân đái tháo đường & tăng huyết áp", "ins_codes_matched": "",
            "question": "", "reference_answer": "", "gold_safety_label": "CAUTION", "binary_answer": "No",
            "label_locator": f"{path}#/nutrition",
            "label_excerpt": f"{entry['name']}: {entry['value']}",
            "review_status": "approved_source_verified", "source_citations": "", "source_claim_locators": "",
            "review_notes": "Added for sugar_codex: Codex CAC/GL 23 sugars-free threshold per 100 g/ml; generated and gated in v5."})
        out.append(base_row)
        n += 1
    return out


def main():
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("authority_topic")]
    by_label: dict[str, dict] = {}
    for r in rows:
        by_label.setdefault(r["label_transcription_path"], r)
    cache: dict = {}
    new, n = [], FIRST_NEW_ID
    n_flagged = 0
    for topic, bases in TOPICS.items():
        control = topic.endswith("_absent")
        paths = sorted(by_label)
        if control:  # NSS-free labels, as many as there are flagged NSS records, so the two classes are balanced
            clean = [p for p in paths if sg.nss_absent(b5.load_label(p, cache))]
            paths = sorted(random.Random(CONTROL_SEED).sample(clean, min(n_flagged, len(clean))))
        for path in paths:
            label = b5.load_label(path, cache)
            found = [] if control else sg.declared_bases(label, bases)
            if not found and not control:
                continue
            if topic == "codex_sulphite_declaration" and sg.SULPHITE_NAME.search(
                    " ".join(b5.field_text(label.get(k)) for k in ("ingredient", "additive", "warning"))):
                continue  # the label already uses the word sulphite/sulfite: nothing to flag
            base = dict(by_label[path])
            pointer = "ingredient" if control else next(
                (f for f in ("additive", "ingredient") if b5.field_text(label.get(f)).strip()
                 and sg.declared_bases({f: label.get(f)}, bases)), "additive")
            dom, dom_vi, pid, pname = TOPIC_PERSONA.get(
                topic, ("additive_regulation", "Đánh giá phụ gia", "healthy_adult", "Người trưởng thành khỏe mạnh"))
            base.update({
                "id": f"FOODQA_G3_{n}", "authority_topic": topic, "domain": dom,
                "domain_vi": dom_vi, "persona_id": pid,
                "persona_name": pname,
                "ins_codes_matched": ", ".join("INS" + b for b in found),
                "question": "", "reference_answer": "",
                "gold_safety_label": "SAFE" if control else "CAUTION", "binary_answer": "Yes" if control else "No",
                "label_locator": f"{path}#/{pointer}",
                "label_excerpt": b5.field_text(label.get(pointer)),
                "review_status": "approved_source_verified",
                "source_citations": "", "source_claim_locators": "",
                "review_notes": f"Added for {topic}; question and answer are generated and gated in v5."})
            new.append(base)
            n += 1
            if topic == "who_nss_pregnancy":
                n_flagged += 1
    new += ins_coverage_rows(rows, by_label, cache, n)
    new += sugar_rows(by_label, cache, FIRST_NEW_ID + len(new))
    new += child_rows(by_label, cache, FIRST_NEW_ID + len(new))  # LAST: earlier record ids must not move
    SRC.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows + new) + "\n", encoding="utf-8")
    from collections import Counter
    print(f"input now {len(rows) + len(new)} rows (+{len(new)}):", dict(Counter(r['authority_topic'] for r in new)))


if __name__ == "__main__":
    main()
