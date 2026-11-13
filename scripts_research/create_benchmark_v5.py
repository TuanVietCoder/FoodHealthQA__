"""Create benchmark v5 (tiered, rule-verified) from v4 without overwriting v4.

No domain experts are assumed. Instead:
  * every record gets an automatic check (label-excerpt provenance, independent
    allergen re-derivation, label/answer consistency),
  * records are split into tiers: core / silver / excluded,
  * known defects found in the v4 audit are fixed deterministically,
  * every change is written to a fix list for manual spot-checking.

Original v4 columns are preserved; v5 adds columns and `*_v5` fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_grounding_v5 as sg  # noqa: E402
SRC = ROOT / "data" / "build" / "qa_input_source_verified.jsonl"
OUT_JSONL = ROOT / "data" / "build" / "qa_build_full.jsonl"
OUT_CSV = ROOT / "data" / "build" / "qa_build_full.csv"
OUT_FIX = ROOT / "data" / "build" / "qa_fix_list.csv"
OUT_SUMMARY = ROOT / "data" / "build" / "qa_build_summary.json"

# --- independent allergen dictionary (written separately from the v4 engine) ---
ALLERGEN_TERMS = {
    "sữa": [r"sữa", r"whey", r"casein", r"lactose", r"phô mai", r"pho mai", r"cheese",
            r"milk", r"chất béo sữa", r"bơ sữa", r"kem sữa"],
    "trứng": [r"trứng", r"egg", r"lòng đỏ", r"lòng trắng", r"mayonnaise", r"(?<!\w)mayo(?!\w)"],
    "đậu phộng": [r"đậu phộng", r"(?<!\w)lạc(?!\w)", r"peanut"],
    "đậu nành": [r"đậu nành", r"(?<!\w)soy", r"đậu tương", r"nước tương", r"xì dầu", r"miso", r"tempeh",
                r"đậu phụ", r"tofu"],
    "lúa mì": [r"lúa mì", r"bột mì", r"bột mỳ", r"wheat", r"gluten", r"semolina"],
    "cá": [r"(?<!\w)cá(?!\w)", r"fish", r"nước mắm", r"cá hồi", r"cá thu"],
    "giáp xác": [r"(?<!\w)tôm(?!\w)", r"(?<!\w)cua(?!\w)", r"(?<!\w)ghẹ(?!\w)",
                 r"giáp xác", r"(?<!\w)tép(?!\w)", r"shrimp", r"prawn", r"crab", r"lobster", r"crustacean"],
    "động vật có vỏ": [r"(?<!\w)tôm(?!\w)", r"(?<!\w)cua(?!\w)", r"(?<!\w)ghẹ(?!\w)", r"giáp xác",
                       r"động vật có vỏ", r"(?<!\w)tép(?!\w)", r"thân mềm", r"(?<!\w)mực(?!\w)", r"bạch tuộc", r"(?<!\w)hàu(?!\w)",
                       r"(?<!\w)hào(?!\w)", r"(?<!\w)sò(?!\w)", r"(?<!\w)ốc(?!\w)", r"nghêu", r"(?<!\w)ngao(?!\w)",
                       r"vẹm", r"bào ngư", r"shrimp", r"crab", r"oyster", r"clam", r"squid", r"mussel", r"scallop"],
    # FDA Q&A Ed.5 Table 1 (C.7/C.8): almond, black/California/English walnut, Brazil nut, cashew, filbert/hazelnut,
    # heartnut, macadamia, pecan, pine nut, pistachio. Chestnut and coconut are NOT on the list ("hạt dẻ" alone is chestnut).
    "hạt cây": [r"hạt điều", r"hạnh nhân", r"óc chó", r"hạt phỉ", r"mac ?ca", r"mắc ?ca", r"hạt dẻ cười", r"hồ đào",
                r"hạt thông", r"hạt brazil", r"hạt hạnh", r"hạt cây", r"almond", r"cashew", r"walnut", r"hazelnut",
                r"filbert", r"pecan", r"pistachio", r"pine nut", r"brazil nut", r"macadamia", r"heartnut"],
    "mè": [r"(?<!\w)mè(?!\w)", r"vừng", r"sesame"],
}
# words in the question -> allergen classes
QUESTION_MAP = [
    ("sữa", r"(?<!\w)sữa(?!\w)(?! đậu nành)"),
    ("trứng", r"trứng"),
    ("đậu phộng", r"đậu phộng"),
    ("đậu nành", r"đậu nành"),
    ("lúa mì", r"lúa mì|gluten"),
    ("cá", r"(?<!\w)cá(?!\w)|hải sản"),
    ("giáp xác", r"(?<!\w)tôm(?!\w)|(?<!\w)cua(?!\w)|(?<!\w)ghẹ(?!\w)|giáp xác"),
    ("động vật có vỏ", r"động vật có vỏ|hải sản"),
    ("hạt cây", r"hạt cây"),
    ("mè", r"(?<!\w)mè(?!\w)|vừng"),
]
# FDA Q&A D.13: "free" claims are voluntary and unregulated, so a "không sữa"/"non-dairy" claim is not evidence of milk.
NEG_MILK_CLAIM = re.compile(
    r"(không|ko|non|free)[\s-]*(có\s+|chứa\s+|thêm\s+)?(sữa|dairy|milk)|(sữa|milk|dairy)[\s-]*free", re.I)
SOY_MILK = re.compile(r"sữa (chiết (xuất )?từ )?(đậu nành|dừa|hạnh nhân|gạo|yến mạch|hạt)")
GENERIC_SEAFOOD = re.compile(r"hải sản|seafood")
PAL_SENTENCE = re.compile(
    r"có thể (có )?chứa|dây chuyền|thiết bị|hệ thống sản xuất|nhà máy|cơ sở sản xuất|"
    r"sản xuất (trên|tại|cùng|chung)|may (also )?contain|manufactured on|processed in|also process|"
    r"equipment|facility|shared", re.I)
STRICT_WHEAT = re.compile(r"lúa mì|bột mì|bột mỳ|wheat|gluten|semolina", re.I)
OTHER_GLUTEN_GRAIN = re.compile(r"lúa mạch|mạch nha|yến mạch|barley|malt|oat|rye", re.I)

KB_SCOPE_MARK = "50 chất"


def load_label(path: str, cache: dict) -> dict:
    if path not in cache:
        p = ROOT / path
        cache[path] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return cache[path]


def field_text(value) -> str:
    if isinstance(value, list):
        parts = []
        for v in value:
            if isinstance(v, dict):
                parts.append(f"{v.get('name', '')}: {v.get('value', '')}" if "name" in v else json.dumps(v, ensure_ascii=False))
            else:
                parts.append(str(v))
        return ", ".join(parts)
    return str(value or "")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace(",", ".")).strip()


def resolve_pointer(label: dict, pointer: str):
    if "#" not in pointer:
        return None
    cur = label
    for part in [x for x in pointer.split("#", 1)[1].split("/") if x]:
        try:
            cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        except (KeyError, IndexError, ValueError, TypeError):
            return None
    return cur


def excerpt_ok(row: dict, label: dict) -> str:
    """ok | generic_locator | field_missing | excerpt_not_in_field"""
    if "#" not in row["label_locator"]:
        return "generic_locator"
    field = resolve_pointer(label, row["label_locator"])
    if field is None:
        return "field_missing"
    exc = norm(row.get("label_excerpt", ""))
    if not field_text(field).strip() and re.match(r"(nhãn|bao bì) không ghi", exc):
        return "ok"  # documented absence: the pointed field is genuinely empty
    text = norm(field_text(field))
    absent = re.match(r"không ghi nhận (.+?) trên nhãn", exc)
    if absent:  # absence claim: every named token must really be missing from the field
        tokens = [t for t in re.split(r"[/,]| hoặc | và ", absent.group(1)) if t.strip()]
        return "ok" if all(t.strip().replace("ins ", "") not in text for t in tokens) else "excerpt_not_in_field"
    parts = [x.strip() for x in exc.split(" | ")]
    return "ok" if exc and all(x in text for x in parts) else "excerpt_not_in_field"


_Q_CODE = r"INS\d{3,4}[A-Z]?(?:\([IVX]+\))*"
_Q_CODE_LIST = re.compile(_Q_CODE + r"(?:,\s*" + _Q_CODE + r")*")
_LABEL_CODE = re.compile(r"(?<![\d.])(\d{3,4})\s*([a-z])?\s*((?:\([ivx]+\))*)", re.I)


def _label_codes(label: dict) -> set[str]:
    text = " ".join(field_text(label.get(k)) for k in ("ingredient", "additive", "warning")).lower()
    text = re.sub(r"(?<!\d)\d{1,2}[.,]\d+", " ", text).replace(",", " ").replace(";", " ")
    out = set()
    for m in _LABEL_CODE.finditer(text):
        out.add(re.sub(r"[\s()]", "", "".join(g or "" for g in m.groups())))
        out.add(m.group(1))
    return out


def _code_on_label(code: str, codes: set[str]) -> bool:
    c = re.sub(r"[\s()]", "", code.lower().replace("ins", ""))
    base = re.match(r"\d+", c).group(0)
    return c in codes or any(x.startswith(base) and (x.startswith(c) or c.startswith(x)) for x in codes)


def refresh_from_corrected_label(r: dict, label: dict):
    """v4 questions/excerpts were written from the first (uncorrected) transcription. After the manual image audit,
    re-derive what depends on the label text: (1) a locator excerpt that is no longer in the field becomes the
    verbatim corrected field; (2) INS codes named in the question or in ins_codes_matched that are no longer on the
    label are dropped from the question text (codes still on the label are kept)."""
    r, flags = dict(r), []
    if excerpt_ok(r, label) == "excerpt_not_in_field":
        field = resolve_pointer(label, r["label_locator"])
        if field is not None and field_text(field).strip():
            r["label_excerpt"] = field_text(field)
            flags.append("excerpt_refreshed_from_corrected_label")
    if r.get("domain") in ("additive_regulation", "insufficient_data"):
        present = _label_codes(label)
        m = _Q_CODE_LIST.search(r["question"])
        if m:
            qcodes = re.findall(_Q_CODE, m.group(0))
            keep = [c for c in qcodes if _code_on_label(c, present)]
            if keep and len(keep) < len(qcodes):
                r["question"] = r["question"][:m.start()] + ", ".join(keep) + r["question"][m.end():]
                r["ins_codes_matched"] = ", ".join(c for c in (x.strip() for x in (r.get("ins_codes_matched") or "").split(","))
                                                  if c and _code_on_label(c, present))
                flags.append("question_codes_regenerated_from_corrected_label")
    return r, flags


_LISTING_TEMPLATES = [
    (re.compile(r"^(?P<p>.+?) chứa phụ gia (?P<c>INS.+?), ăn hàng ngày có an toàn cho người lớn khỏe mạnh không\?$"),
     "{p} chứa phụ gia {c} — những phụ gia này có nằm trong danh mục phụ gia thực phẩm được phép sử dụng theo quy định không?"),
    (re.compile(r"^Tôi ăn (?P<p>.+?) thường xuyên — phụ gia (?P<c>INS.+?) trong sản phẩm này có ảnh hưởng sức khỏe không\?$"),
     "Tôi ăn {p} thường xuyên — phụ gia {c} trong sản phẩm này có được phép sử dụng theo Thông tư 24/2019/TT-BYT không?"),
    (re.compile(r"^Tôi lo ngại về phụ gia (?P<c>INS.+?) trong (?P<p>.+?) — những chất này có gây hại gì không\?$"),
     "Tôi lo ngại về phụ gia {c} trong {p} — những chất này có thuộc danh mục phụ gia được phép sử dụng không?"),
]


def listing_question(question: str) -> str:
    """PERMITTED_LISTED answers only establish the legal listing status (the label gives no dose), so the question must
    ask about that status rather than about safety or harm, which the answer deliberately declines to conclude."""
    for rx, tpl in _LISTING_TEMPLATES:
        m = rx.match(question)
        if m:
            return tpl.format(p=m.group("p"), c=m.group("c"))
    return question


def asked_allergens(question: str) -> list[str]:
    # drop the quoted product name so it is not mistaken for the asked allergen
    q = re.sub(r"'[^']*'", " ", question.lower())
    return [name for name, pat in QUESTION_MAP if re.search(pat, q)]


def has_term(text: str, allergen: str) -> bool:
    text = NEG_MILK_CLAIM.sub(" ", SOY_MILK.sub(" ", text.lower())) if allergen == "sữa" else text.lower()
    return any(re.search(p, text) for p in ALLERGEN_TERMS[allergen])


PAL_MARKER = re.compile(
    r"và có thể (có )?chứa|có thể (có )?chứa|may (also )?contain|dây chuyền|thiết bị|hệ thống sản xuất|"
    r"nhà máy|cơ sở sản xuất|sản xuất (trên|tại|cùng|chung)|manufactured on|processed in|also process|"
    r"equipment|facility|shared", re.I)


def split_declared_pal(label: dict) -> tuple[str, str]:
    """Return (declared_text, pal_text). Warning sentences are split at the first PAL marker."""
    declared = field_text(label.get("ingredient")) + " " + field_text(label.get("additive"))
    pal = ""
    warnings = label.get("warning") or []
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
    for item in warnings:
        for sentence in re.split(r"(?<=[.;])\s+|\n", str(item)):
            m = PAL_MARKER.search(sentence)
            if m:
                declared += " " + sentence[:m.start()]
                pal += " " + sentence[m.start():]
            else:
                declared += " " + sentence
    # "có thể chứa" inside the ingredient list is a PAL statement too
    for chunk in re.findall(r"có thể chứa[^)\];.]*", declared, flags=re.I):
        pal += " " + chunk
        declared = declared.replace(chunk, " ")
    return declared, pal


# FDA Q&A (Ed. 5) D.9: highly refined (refined, bleached, deodorized) oils from a major allergen are exempt from the allergen
# labelling requirement; an oil that is not stated to be refined must still be declared, so only explicit wording is stripped.
REFINED_OIL = re.compile(
    r"(dầu|oil|olein)\s+(đậu nành|đậu phộng|lạc|mè|vừng|soy\w*|peanut|sesame)\s*(đã\s+)?(tinh luyện|refined)"
    r"|refined\s+(soy\w*|peanut|sesame)\s+oil", re.I)


def refined_oil_only(row: dict, label: dict) -> list[str]:
    """Asked allergens whose only evidence in the declared text is an explicitly refined oil."""
    declared, _ = split_declared_pal(label)
    stripped = REFINED_OIL.sub(" ", declared)
    return [a for a in asked_allergens(row["question"]) if has_term(declared, a) and not has_term(stripped, a)]


def allergen_hits(row: dict, label: dict):
    allergens = asked_allergens(row["question"])
    declared, pal = split_declared_pal(label)
    declared_x = REFINED_OIL.sub(" ", declared)
    return ([a for a in allergens if has_term(declared_x, a)],
            [a for a in allergens if has_term(pal, a)], allergens, declared, pal)


def independent_allergen_label(row: dict, label: dict):
    """Independent re-derivation: UNSAFE / CRITICAL_ALERT / NOT_DECLARED / AMBIGUOUS / None."""
    in_decl, in_pal, allergens, declared, pal = allergen_hits(row, label)
    if not allergens:
        return None
    if in_decl:
        return "UNSAFE"
    if in_pal:
        return "CRITICAL_ALERT"
    asked_generic = "hải sản" in re.sub(r"'[^']*'", " ", row["question"].lower())
    if (not asked_generic and any(a in ("giáp xác", "động vật có vỏ", "cá") for a in allergens)
            and GENERIC_SEAFOOD.search((declared + " " + pal).lower())):
        return "AMBIGUOUS"
    return "NOT_DECLARED"


# Decisions taken by reading the full label text of each quarantined record (no expert).
ADJUDICATED = {
    "FOODQA_G3_1147": "UNSAFE",          # allergen info: "lecithin (có nguồn gốc từ đậu nành)"
    "FOODQA_G3_0256": "CRITICAL_ALERT",  # PAL: "equipments that also process ... peanuts"
    "FOODQA_G3_0273": "CAUTION",         # "bột gia vị hải sản" has unspecified species
    "FOODQA_G3_0351": "CRITICAL_ALERT",  # "có thể chứa gluten" is PAL, not a direct ingredient
    "FOODQA_G3_0847": "CRITICAL_ALERT",  # PAL: equipment shared with milk, egg, peanut
    "FOODQA_G3_1276": "UNSAFE",          # shrimp is a direct ingredient (bột tôm)
    "FOODQA_G3_1613": "UNSAFE",          # "Eggs ≥30%" is a direct ingredient
    "FOODQA_G3_2058": "UNSAFE",          # "Wheat Flour" is a direct ingredient
    "FOODQA_G3_0042": "UNSAFE",          # soy declared directly, peanut only via PAL
    "FOODQA_G3_0967": "UNSAFE",          # same pattern
    "FOODQA_G3_1050": "UNSAFE",          # same pattern
    "FOODQA_G3_1074": "UNSAFE",          # soy declared directly, peanut only via PAL
}


def adjudicated_answer(row: dict, label: dict, final: str) -> str:
    in_decl, in_pal, allergens, _, _ = allergen_hits(row, label)
    name = row["product_name"]
    rest = [a for a in allergens if a not in in_decl and a not in in_pal]
    rest_note = f" Không ghi nhận trên nhãn: {', '.join(rest)}." if rest else ""
    if final == "UNSAFE":
        what = ", ".join(in_decl)
        extra = [a for a in in_pal if a not in in_decl]
        pal_note = f" Ngoài ra nhãn cảnh báo có thể nhiễm chéo {', '.join(extra)}." if extra else ""
        return (f"KHÔNG AN TOÀN — Nhãn '{name}' kê khai trực tiếp {what} trong thành phần hoặc phần cảnh báo.{pal_note} "
                f"Người dị ứng {what} KHÔNG ĐƯỢC dùng sản phẩm này.{rest_note} [US_FDA_FALCPA] [WHO_JEMRA_PRIORITY_ALLERGENS]")
    if final == "CRITICAL_ALERT":
        what = ", ".join(in_pal)
        return (f"CẢNH BÁO KHẨN CẤP (NHIỄM CHÉO) — Thành phần của '{name}' không kê khai trực tiếp {what}, "
                f"nhưng nhãn cảnh báo sản phẩm có thể chứa/nhiễm chéo {what} từ dây chuyền hoặc thiết bị sản xuất. "
                f"Người dị ứng nặng cần tránh.{rest_note} [FAO_WHO_JEMRA_PART_3] [US_FDA_FALCPA]")
    return (f"THẬN TRỌNG — Thành phần '{name}' có 'bột gia vị hải sản' không nêu rõ loài, và phần cảnh báo "
            f"chỉ liệt kê đậu nành, lúa mì, cá ngừ, cá mòi, không nhắc giáp xác. Không thể loại trừ giáp xác; "
            f"người dị ứng {', '.join(allergens)} nên xác nhận với nhà sản xuất. [US_FDA_FALCPA] [WHO_JEMRA_PRIORITY_ALLERGENS]")


def v4_to_comparable(row: dict) -> str:
    lab = row["gold_safety_label"]
    return "NOT_DECLARED" if lab == "CAUTION" else lab


def cites(row: dict) -> str:
    return " ".join(f"[{c.strip()}]" for c in (row.get("source_citations") or "").split(",") if c.strip())


def sanitize_brackets(text: str) -> str:
    """Square brackets are reserved for citation tags; turn any other [..] from label text into (..)."""
    return re.sub(r"\[([^\[\]]*)\]",
                  lambda m: m.group(0) if re.fullmatch(r"[A-Z][A-Z0-9_.]+", m.group(1)) else f"({m.group(1)})", text)


CATALOG_KEYS = set(json.loads((ROOT / "knowledge_base" / "sources" / "catalog" / "source_citation_catalog.json").read_text(encoding="utf-8")))


def build(rows: list[dict]):
    cache: dict = {}
    out, fixes = [], []

    def fix(row, code, detail):
        fixes.append({"id": row["id"], "fix_code": code, "detail": detail,
                      "v4_label": row["gold_safety_label"]})

    for r in rows:
        label = load_label(r["label_transcription_path"], cache)
        r, refresh_flags = refresh_from_corrected_label(r, label)
        v = dict(r)
        flags: list[str] = list(refresh_flags)
        v["label_v5"] = r["gold_safety_label"]
        v["label_scope"] = "full_record"
        v["abstain_reason"] = ""
        v["question_v5"] = r["question"]
        v["reference_answer_v5"] = r["reference_answer"]
        tier = "core"

        ex = excerpt_ok(r, label)
        if ex != "ok":
            flags.append(f"locator:{ex}")

        dom, lab, persona = r["domain"], r["gold_safety_label"], r["persona_id"]
        name = r["product_name"]

        # ---- additive records added for EU Annex V / FDA Red No. 3 / IARC aspartame (see add_authority_records_v5.py) ----
        if r.get("authority_topic") in sg.LOOKUP_TOPICS:
            pass  # INS coverage lookups (listing / JECFA ADI) are deterministic lookups: they keep tier core
        elif r.get("authority_topic"):
            tier = "silver"
            flags.append("authority_flag_scope")
        # ---- allergen domain: independent re-derivation + targeted fixes ----
        elif dom == "allergen_detection":
            ind = independent_allergen_label(r, label)
            allergens = asked_allergens(r["question"])
            if ind is None:
                flags.append("allergen_unparsed")
            elif ind != v4_to_comparable(r):
                # wheat/gluten special-case: only other gluten grains / oats
                if (lab == "UNSAFE" and allergens == ["lúa mì"] and ind == "NOT_DECLARED"):
                    blob = " ".join(field_text(label.get(k)) for k in ("ingredient", "additive", "warning"))
                    grain = OTHER_GLUTEN_GRAIN.findall(blob)
                    v["label_v5"] = "CAUTION"
                    v["reference_answer_v5"] = (
                        f"THẬN TRỌNG — Nhãn '{name}' không kê khai lúa mì/bột mì hay gluten từ lúa mì. "
                        f"Nhãn có ngũ cốc khác ({', '.join(sorted({g.lower() for g in grain}))}) có thể chứa gluten "
                        "hoặc nhiễm chéo; người dị ứng lúa mì cần cân nhắc/xác nhận với nhà sản xuất. "
                        "[US_FDA_FALCPA] [WHO_JEMRA_PRIORITY_ALLERGENS]")
                    fix(r, "WHEAT_GLUTEN_UNSAFE_TO_CAUTION", f"grains={grain}")
                    flags.append("wheat_gluten_fixed")
                    tier = "silver"
                elif refined_oil_only(r, label) and ind in ("NOT_DECLARED", "CRITICAL_ALERT"):
                    v["label_v5"] = ind
                    v["label_v5_proposed"] = ind
                    fix(r, "REFINED_OIL_EXEMPT_PER_FDA_D9", f"v4={lab}: evidence is only an explicitly refined oil -> {ind}")
                    flags.append("refined_oil_exempt")
                    tier = "silver"
                elif r["id"] in ADJUDICATED:
                    final = ADJUDICATED[r["id"]]
                    v["label_v5"] = final
                    v["reference_answer_v5"] = adjudicated_answer(r, label, final)
                    v["label_v5_proposed"] = final
                    fix(r, "V4_ERROR_ADJUDICATED", f"v4={lab} -> {final} (read full label text)")
                    flags.append("adjudicated_by_label_text")
                    tier = "silver"
                else:
                    flags.append(f"indep_disagree:{ind}")
                    fix(r, "INDEPENDENT_DISAGREE", f"v4={lab} indep={ind}")
                    v["label_v5"] = "NEEDS_REVIEW"
                    v["label_v5_proposed"] = ind
                    tier = "quarantine"
            if lab == "CAUTION" and v["label_v5"] == "CAUTION" and r["id"] not in ADJUDICATED:
                # "no PAL" is absence of information, not a hazard signal
                v["label_v5"] = "NOT_DECLARED"
                v["label_scope"] = "declared_ingredients_only"
                a = ", ".join(allergens) or "dị nguyên được hỏi"
                v["reference_answer_v5"] = (
                    f"KHÔNG KÊ KHAI — Danh sách thành phần và cảnh báo trên nhãn '{name}' không ghi nhận {a}. "
                    "Nhãn cũng không cung cấp thông tin về nhiễm chéo dây chuyền (PAL), nên không thể "
                    "kết luận sản phẩm không có nguy cơ nhiễm chéo. [US_FDA_FALCPA] [WHO_JEMRA_PRIORITY_ALLERGENS]")
                fix(r, "CAUTION_TO_NOT_DECLARED", "no-PAL is not a hazard signal")
            # marginal case known from v3 audit
            if re.search(r"có thể chứa[^)]*", field_text(label.get("ingredient")), re.I) and lab == "CRITICAL_ALERT":
                flags.append("pal_inside_ingredient_paren")
                tier = "silver"

        # ---- additive domain ----
        elif dom == "additive_regulation":
            if lab == "SAFE" and "không thể xác nhận" in r["reference_answer"] and r["reference_answer"].startswith("Phụ gia"):
                codes = r.get("ins_codes_matched", "")
                v["label_v5"] = "PERMITTED_LISTED"
                v["label_scope"] = "listed_in_positive_list_only"
                v["reference_answer_v5"] = (
                    f"CÓ TRONG DANH MỤC CHO PHÉP — Phụ gia {codes} trong '{name}' nằm trong danh mục phụ gia được "
                    "phép theo Thông tư 24/2019/TT-BYT và Codex STAN 192-1995. Phạm vi kết luận chỉ là tư cách "
                    "được phép sử dụng: nhãn không ghi hàm lượng (mg/kg) và mã nhóm thực phẩm nên không đánh giá "
                    f"được tuân thủ mức tối đa (ML) và không đủ để kết luận mức an toàn tuyệt đối. {cites(r)}")
                fix(r, "SAFE_RELABELED_PERMITTED_LISTED", "answer previously contradicted SAFE label")
            if lab == "ABSTAIN":
                v["abstain_reason"] = "KB_OUT_OF_SCOPE" if KB_SCOPE_MARK in r["reference_answer"] else "LABEL_MISSING_INFO"
            if lab == "CAUTION":
                flags.append("judgment_caution")
                tier = "silver"

        # ---- vulnerable groups: single-factor, judgement-dependent ----
        elif dom == "vulnerable_groups":
            head = r["reference_answer"].split("—")[0].strip()
            v["label_scope"] = "single_factor:" + head.lower()[:60]
            if persona == "pregnant_woman" and lab == "UNSAFE":
                v["question_v5"] = (f"Tôi đang mang thai — nhãn '{name}' có cảnh báo gì về việc dùng cho "
                                    "phụ nữ mang thai không?")
                fix(r, "PREGNANT_QUESTION_REALIGNED", "question asked about INS, answer was label warning")
                flags.append("question_realigned")
            if "NATRI" in head.upper() and "[USDA_FDC]" in r["reference_answer"]:
                flags.append("citation_threshold_misattributed(USDA_FDC->FDA 21CFR101.61)")
                fix(r, "SODIUM_CITATION_REVIEW", "low-sodium threshold is an FDA labelling rule")
            tier = "excluded_clinical_judgment"

        # ---- insufficient data ----
        elif dom == "insufficient_data":
            reason = "KB_OUT_OF_SCOPE" if KB_SCOPE_MARK in r["reference_answer"] else (
                "PAL_NOT_ON_LABEL" if "PAL" in r["reference_answer"] else "LABEL_MISSING_INFO")
            v["abstain_reason"] = reason
            if reason == "KB_OUT_OF_SCOPE":
                tier = "excluded_kb_dependent"
            if reason == "PAL_NOT_ON_LABEL":
                # the v4 answer claims the allergen is not declared and no PAL exists; verify that claim
                ind = independent_allergen_label(r, label)
                if ind in ("UNSAFE", "CRITICAL_ALERT"):
                    v["label_v5"] = ind
                    v["abstain_reason"] = ""
                    v["reference_answer_v5"] = adjudicated_answer(r, label, ind)
                    v["label_scope"] = "full_record"
                    fix(r, "V4_FALSE_NOT_DECLARED_ABSTAIN", f"v4 ABSTAIN said allergen undeclared; label text gives {ind}")
                    flags.append("v4_factual_error_fixed")
            # generic whole-file locator is expected for absence-of-information
            if flags and flags[0] == "locator:generic_locator":
                flags.clear()

        # ---- question/answer/citation alignment repairs found by verify_v5_answers.py ----
        if name.lower() not in v["reference_answer_v5"].lower() and " — " in v["reference_answer_v5"]:
            head, tail = v["reference_answer_v5"].split(" — ", 1)
            v["reference_answer_v5"] = f"{head} — Với sản phẩm '{name}': {tail}"
            fix(r, "ANSWER_PRODUCT_NAME_ADDED", "answer did not mention the product asked about")
        if (persona == "child_5yo" and re.search(r"tăng động|hiếu động", v["question_v5"].lower())
                and not re.search(r"southampton|hiếu động|tăng động|phẩm màu", v["reference_answer_v5"].lower())):
            v["question_v5"] = (f"Xét riêng về phụ gia trong '{name}', sản phẩm có nguy cơ gì đáng lưu ý "
                                "cho trẻ nhỏ 5 tuổi không?")
            fix(r, "CHILD_QUESTION_REALIGNED", "question asked about hyperactivity, answer addressed another hazard")
            flags.append("question_realigned")
        if persona == "pku_patient":
            if "172.804" in v["reference_answer_v5"]:
                v["reference_answer_v5"] = v["reference_answer_v5"].replace("[US_FDA_FALCPA]", "[US_FDA_21CFR172_804]")
                fix(r, "PKU_CITATION_TAG_FIXED", "FALCPA is an allergen law; the PKU claim is 21 CFR 172.804")
        # ---- source grounding: keep only statements that were checked against the primary texts ----
        grounded_claims: list[str] = []
        jecfa_loc: list[str] = []
        source_gap = ""
        old_answer = v["reference_answer_v5"]
        final_label = v["label_v5"]
        allergen_labels = ("UNSAFE", "CRITICAL_ALERT", "NOT_DECLARED")
        is_pal_abstain = dom == "insufficient_data" and v["abstain_reason"] == "PAL_NOT_ON_LABEL"
        is_allergen_caution = dom == "allergen_detection" and final_label == "CAUTION"
        if r.get("authority_topic") in sg.LOOKUP_TOPICS:
            lk_codes = [c.strip() for c in (r.get("ins_codes_matched") or "").split(",") if c.strip()]
            lk = sg.ins_lookup_record(r["authority_topic"], name, lk_codes, label)
            if lk and all(sg.code_in_label_tokens(c, label) for c in lk_codes):
                (v["question_v5"], v["reference_answer_v5"], v["label_v5"], grounded_claims, jecfa_loc,
                 v["label_scope"]) = lk
                final_label = v["label_v5"]
                v["abstain_reason"] = "LABEL_MISSING_INFO" if final_label == "ABSTAIN" else ""
            else:
                source_gap = "ins_lookup_source_or_label_code_missing"
        elif r.get("authority_topic") in sg.CHILD_TOPICS:
            rec = sg.child_authority_record(r["authority_topic"], name, label)
            if rec:
                (v["question_v5"], v["reference_answer_v5"], grounded_claims, v["label_scope"],
                 final_label, v["ins_codes_matched"]) = rec
                v["label_v5"] = final_label
            else:
                source_gap = "child_authority_label_does_not_qualify"
        elif r.get("authority_topic"):
            bases = sg.declared_bases(label, {"eu_annex_v_colours": sg.ANNEXV_BASES, "fda_red3_revoked": sg.RED3_BASES,
                                              "iarc_v134_aspartame": sg.IARC_ASPARTAME_BASES,
                                              "codex_sulphite_declaration": sg.SULPHITE_BASES,
                                              "who_nss_pregnancy": sg.NSS_BASES,
                                              "who_nss_pregnancy_absent": sg.NSS_BASES}[r["authority_topic"]])
            if r["authority_topic"] == "who_nss_pregnancy_absent":
                if not bases and sg.nss_absent(label):
                    v["question_v5"], v["reference_answer_v5"], grounded_claims, v["label_scope"] = \
                        sg.nss_absent_record(name)
                    v["ins_codes_matched"] = ""
                    v["label_v5"] = final_label = "SAFE"
                else:
                    source_gap = "nss_control_label_not_clean"
            elif bases:
                v["question_v5"], v["reference_answer_v5"], grounded_claims, v["label_scope"] = \
                    sg.authority_topic_record(r["authority_topic"], name, bases)
                v["ins_codes_matched"] = ", ".join("INS" + b for b in bases)
                v["label_v5"] = final_label = "CAUTION"
            else:
                source_gap = "authority_code_not_declared_on_label"
        elif (dom in ("allergen_detection", "insufficient_data") and final_label in allergen_labels)                 or is_pal_abstain or is_allergen_caution:
            in_decl, in_pal, asked_cls, declared_txt, pal_txt = allergen_hits(r, label)
            q_surface = re.sub(r"'[^']*'", " ", r["question"].lower())
            def cls_match(text, cls):
                return " ".join(m.group(0) for pat in ALLERGEN_TERMS[cls] for m in re.finditer(pat, text.lower()))
            base = {"UNSAFE": declared_txt, "CRITICAL_ALERT": pal_txt}.get(final_label, q_surface)
            matched = {c: cls_match(base, c) for c in asked_cls}
            extra_body = ""
            if is_allergen_caution:
                if "wheat_gluten_fixed" in flags:
                    blob = " ".join(field_text(label.get(k)) for k in ("ingredient", "additive", "warning"))
                    grain = sorted({g.lower() for g in OTHER_GLUTEN_GRAIN.findall(blob)})
                    extra_body = (f"THẬN TRỌNG — Nhãn '{name}' không kê khai lúa mì/bột mì hay gluten từ lúa mì. "
                                  f"Nhãn có ngũ cốc khác ({', '.join(grain)}) có thể chứa gluten hoặc nhiễm chéo; "
                                  "người dị ứng lúa mì nên xác nhận với nhà sản xuất.")
                else:  # unspecified seafood ingredient
                    extra_body = (f"THẬN TRỌNG — Thành phần '{name}' có 'bột gia vị hải sản' không nêu rõ loài, và phần "
                                  f"cảnh báo không liệt kê giáp xác. Không thể loại trừ giáp xác; người dị ứng "
                                  f"{sg.names(asked_cls)} nên xác nhận với nhà sản xuất.")
            v["reference_answer_v5"], grounded_claims = sg.allergen_answer(
                "ABSTAIN" if is_pal_abstain else final_label, name, asked_cls, in_decl, in_pal, matched,
                extra_body=extra_body, abstain_pal=is_pal_abstain)
            if "refined_oil_exempt" in flags and final_label in ("NOT_DECLARED", "CRITICAL_ALERT"):
                oils = REFINED_OIL.search(declared_txt)
                what = sg.names(refined_oil_only(r, label))
                lead = (f"KHÔNG KÊ KHAI — Nhãn '{name}' chỉ nêu {what} qua dầu tinh luyện ('{oils.group(0).strip()}'). "
                        if final_label == "NOT_DECLARED" else
                        f"CẢNH BÁO KHẨN CẤP (NHIỄM CHÉO) — Nhãn '{name}' chỉ nêu {what} qua dầu tinh luyện ('{oils.group(0).strip()}') "
                        f"và cảnh báo có thể nhiễm chéo {sg.names(in_pal)}. ")
                v["reference_answer_v5"] = (
                    lead + "Theo hướng dẫn của US FDA (Edition 5, mục D.9), dầu tinh luyện hoàn toàn (refined, bleached, "
                    "deodorized) có nguồn từ dị nguyên chính được miễn quy định khai báo dị nguyên của mục 403(w); nguồn dầu "
                    "vẫn phải ghi trong danh sách thành phần. Ngoài dầu này, thành phần và cảnh báo không kê khai "
                    f"{what}. Nhãn và hướng dẫn không xác nhận mức dung nạp của từng người, nên người dị ứng nặng cần hỏi "
                    "bác sĩ. [US_FDA_ALLERGEN_QA_ED5] [US_FDA_FALCPA]")
                grounded_claims = ["FDA_QA5_REFINED_OIL_EXEMPT", "FALCPA_321qq_NINE_ALLERGENS"]
            elif final_label in ("UNSAFE", "CRITICAL_ALERT") and refined_oil_only(r, label):
                oils = REFINED_OIL.search(declared_txt)
                v["reference_answer_v5"] += (
                    f" Riêng {sg.names(refined_oil_only(r, label))}: nhãn chỉ nêu qua dầu tinh luyện ('{oils.group(0).strip()}'), "
                    "loại dầu được miễn khai báo dị nguyên theo hướng dẫn của US FDA (Edition 5, mục D.9). "
                    "[US_FDA_ALLERGEN_QA_ED5]")
                grounded_claims = list(grounded_claims) + ["FDA_QA5_REFINED_OIL_EXEMPT"]
        elif (dom == "insufficient_data" and final_label == "ABSTAIN" and v["abstain_reason"] == "KB_OUT_OF_SCOPE"
              and sg.jecfa_status_answer(name, re.findall(r"INS\s?(\d{3,4}[A-Za-z]?(?:\([IVXivx]+\))?)", v["question_v5"]))):
            new_label, new_a, grounded_claims, jecfa_loc = sg.jecfa_status_answer(
                name, re.findall(r"INS\s?(\d{3,4}[A-Za-z]?(?:\([IVXivx]+\))?)", v["question_v5"]))
            v["label_v5"], final_label = new_label, new_label
            v["reference_answer_v5"] = new_a
            v["abstain_reason"] = "" if new_label == "SAFE" else "LABEL_MISSING_INFO"
            v["label_scope"] = "adi_status_per_jecfa_only"
            tier = "core"
            fix(r, "KB_ABSTAIN_REGROUNDED_ON_JECFA", f"all codes have a fetched JECFA record -> {new_label}")
        elif dom == "vulnerable_groups" and persona == "diabetes_hypertension" and                 "NATRI" in r["reference_answer"].split("—")[0].upper() and sg.sodium_codex_answer(name, label):
            new_label, new_q, new_a, grounded_claims = sg.sodium_codex_answer(name, label)
            v["label_v5"], final_label = new_label, new_label
            v["question_v5"], v["reference_answer_v5"] = new_q, new_a
            v["label_scope"] = "single_factor:natri_vs_codex_low_threshold_per_100g"
            tier = "core"
            fix(r, "SODIUM_RECOMPUTED_ON_CODEX_GL23", f"label per-100 value vs 120 mg/100 g -> {new_label}")
        elif dom == "vulnerable_groups" and tier == "excluded_clinical_judgment" and persona in (
                "ckd_stage3", "child_5yo", "pku_patient", "pregnant_woman"):
            vcodes = sg.merge_declared_codes([c.strip() for c in (r.get("ins_codes_matched") or "").split(",") if c.strip()], label)
            res = {"ckd_stage3": sg.ckd_answer, "child_5yo": sg.child_answer, "pku_patient": sg.pku_answer,
                   "pregnant_woman": sg.pregnancy_answer}[persona](name, vcodes)
            if res:
                new_label, new_text, grounded_claims, jecfa_loc, scope = res
                if new_label != r["gold_safety_label"]:
                    fix(r, "VULNERABLE_LABEL_RECOMPUTED",
                        f"v4={r['gold_safety_label']} -> {new_label} from declared INS codes {vcodes}")
                v["label_v5"], final_label = new_label, new_label
                v["reference_answer_v5"] = new_text
                if persona == "child_5yo" and v["question_v5"] != r["question"]:
                    # the answer now addresses the original hyperactivity question, so undo the earlier realignment
                    v["question_v5"] = r["question"]
                    fixes[:] = [f for f in fixes if not (f["id"] == r["id"] and f["fix_code"] == "CHILD_QUESTION_REALIGNED")]
                    flags[:] = [f for f in flags if f != "question_realigned"]
                v["label_scope"] = scope
                tier = "silver"
                flags.append("authority_flag_scope")
        elif final_label == "PERMITTED_LISTED":
            v["reference_answer_v5"], grounded_claims = sg.permitted_answer(name, r.get("ins_codes_matched", ""))
            cx_sentence, cx_claims = sg.codex192_sentence(r.get("ins_codes_matched", "").split(","))
            if cx_sentence:  # only when every code is really listed in CXS 192-1995
                v["reference_answer_v5"] += " " + cx_sentence
                grounded_claims = grounded_claims + cx_claims
        elif (dom in ("additive_regulation", "insufficient_data") and final_label == "ABSTAIN"
              and "không xác định duy nhất một phụ gia" in old_answer):
            had_cxg = "[CODEX_CXG_36_1989]" in old_answer
            v["reference_answer_v5"] = old_answer.replace(" [CODEX_CXG_36_1989]", "").replace(
                "[VN_TT24_2019]", "Phụ lục 1 của Thông tư 24/2019/TT-BYT liệt kê phụ gia theo mã INS và tên phụ gia, "
                "nên cần mã INS hoặc tên đúng danh mục để đối chiếu. [VN_TT24_2019]")
            grounded_claims = ["TT24_PHULUC1_LISTS_INS_AND_NAME"]
            if had_cxg:
                cg_sentence, cg_claims = sg.cxg36_sentence()
                if cg_sentence:
                    v["reference_answer_v5"] += " " + cg_sentence
                    grounded_claims += cg_claims
        elif dom == "insufficient_data" and final_label == "ABSTAIN" and "tìm ngưỡng ADI" in old_answer:
            codes = ", ".join(dict.fromkeys(re.findall(r"INS\d+[A-Za-z]?(?:\([IVXivx]+\))?", old_answer)))
            v["reference_answer_v5"], grounded_claims = sg.abstain_quantity_answer(name, codes)
        elif dom == "insufficient_data" and final_label == "ABSTAIN" and v["abstain_reason"] == "LABEL_MISSING_INFO":
            v["reference_answer_v5"] = old_answer.replace(" [CODEX_STAN_192_1995]", "")
        elif dom == "additive_regulation" and final_label == "SAFE" and old_answer.startswith("AN TOÀN VỀ NGƯỠNG ADI"):
            mm = re.search(r"nhóm phụ gia (.*?) trong '", old_answer)
            jcodes = [re.sub(r"^ins", "", c.strip().lower()) for c in (mm.group(1).split(",") if mm else [])]
            new_ans, used = sg.jecfa_adi_answer(name, jcodes)
            if new_ans:
                cx_sentence, cx_claims = sg.codex192_sentence(jcodes)
                v["reference_answer_v5"] = new_ans + (" " + cx_sentence if cx_sentence else "")
                grounded_claims = used + cx_claims
                v["label_scope"] = "adi_status_per_jecfa_only"
                jecfa_loc = sg.jecfa_locators(jcodes)
        elif dom == "additive_regulation" and final_label == "CAUTION" and "INS171" in old_answer.upper():
            codes = ", ".join(dict.fromkeys(re.findall(r"INS\d+[A-Za-z]?(?:\([IVXivx]+\))?", old_answer.split("Lưu ý phạm vi")[0])))
            v["reference_answer_v5"], grounded_claims = sg.tio2_answer(name, codes)
        if not tier.startswith("excluded"):
            answer_tags = sg.tags_of(v["reference_answer_v5"])
            bad_tags = [t for t in answer_tags if t not in sg.VERIFIED_TAGS]
            # a source tag is only valid when a claim checked against that very source backs it
            needs = {"JECFA_EVALUATIONS": "JECFA_ADI_", "CODEX_STAN_192_1995": "CODEX192_",
                     "CODEX_CXS_1_1985": "CXS1_", "CODEX_CXG_36_1989": "CODEX_CXG36_",
                     "EFSA_OPINIONS": "EFSA_OPINION_", "US_FDA_21CFR172_804": "CFR_172_804",
                     "EFSA_2021_TITANIUM_DIOXIDE_E171": "EFSA_2021_TIO2",
                     "CODEX_CAC_GL_23_1997": "CODEX_GL23_",
                     "US_FDA_ALLERGEN_QA_ED5": "FDA_QA5_", **sg.AUTHORITY_TAG_PREFIX}
            for tag, prefix in needs.items():
                if tag in answer_tags and not any(c.startswith(prefix) for c in grounded_claims):
                    bad_tags.append(tag)
            if re.search(r"USDA", v["reference_answer_v5"]):
                bad_tags.append("USDA_MENTIONED_WITHOUT_SOURCE")
            if re.search(r"Codex|CXS|CXG", v["reference_answer_v5"]) and not any(t.startswith("CODEX") for t in answer_tags):
                bad_tags.append("CODEX_NAMED_WITHOUT_TAG")
            if bad_tags:
                source_gap = "tags_not_verifiable:" + ",".join(dict.fromkeys(bad_tags))
            elif not answer_tags or (
                    dom == "additive_regulation" and final_label in ("SAFE", "CAUTION") and not grounded_claims):
                source_gap = "no_verifiable_source_claim"
        if source_gap and dom == "additive_regulation" and final_label == "CAUTION" and re.search(r"benzen", old_answer, re.I):
            source_gap = "benzene_source_covers_beverages_only;" + source_gap
        if source_gap and "USDA" in source_gap:
            source_gap = "nutrient_threshold_basis_not_stated_on_label;" + source_gap
        if v["reference_answer_v5"] != old_answer:
            fix(r, "ANSWER_REGROUNDED_TO_VERIFIED_SOURCES", ",".join(grounded_claims))
        v["external_claims"] = ";".join(grounded_claims)
        if source_gap:
            v["source_gap"] = source_gap
            fix(r, "SOURCE_NOT_VERIFIABLE", source_gap)
            tier = "excluded_source_not_verifiable"
        else:
            v["source_gap"] = ""

        cleaned = sanitize_brackets(v["reference_answer_v5"])
        if cleaned != v["reference_answer_v5"]:
            v["reference_answer_v5"] = cleaned
            fix(r, "FOREIGN_BRACKETS_SANITIZED", "label text in [..] would be parsed as a citation")
        v["question_v5"] = sanitize_brackets(v["question_v5"])
        locs = [x.strip() for x in (r.get("source_claim_locators") or "").split(";") if x.strip()]
        kept = [x for x in locs if x in CATALOG_KEYS]
        if kept != locs:
            fix(r, "LOCATOR_NOT_IN_CATALOG_DROPPED", "; ".join(x for x in locs if x not in CATALOG_KEYS))
        v["source_claim_locators_v5"] = "; ".join(
            [x for x in [sg.locators_for([t for t in sg.tags_of(v["reference_answer_v5"]) if t != "JECFA_EVALUATIONS"])] if x]
            + jecfa_loc) or "; ".join(kept)
        if r.get("authority_topic") == "codex_sulphite_declaration":
            v["source_claim_locators_v5"] = "CODEX_CXS_1_1985#Sec4.2.1.7"
        if r.get("authority_topic") == "sugar_codex":
            v["source_claim_locators_v5"] = "CODEX_CAC_GL_23_1997#Table_Sugars"  # not the sodium row of the same table
        if r.get("authority_topic") in sg.CHILD_TOPICS:
            v["source_claim_locators_v5"] = sg.CHILD_LOCATORS[r["authority_topic"]]  # children, not the pregnancy section
        v["source_citations_v5"] = ", ".join(dict.fromkeys(re.findall(r"\[([A-Za-z0-9_.]+)\]", v["reference_answer_v5"])))

        if tier == "core" and flags:
            tier = "silver"
        if (r["id"] in ADJUDICATED or "v4_factual_error_fixed" in flags) and tier == "core":
            tier = "silver"
        v.setdefault("label_v5_proposed", "")
        v["tier"] = tier
        v["fix_flags"] = ";".join(flags)
        if v["label_v5"] == "PERMITTED_LISTED":
            new_q = listing_question(v["question_v5"])
            if new_q != v["question_v5"]:
                v["question_v5"] = new_q
                v["fix_flags"] = ";".join([f for f in (v["fix_flags"], "question_rewritten_to_listing_status") if f])
        v["binary_answer_v5"] = {"SAFE": "Yes", "PERMITTED_LISTED": "Listed", "ABSTAIN": "Abstain",
                                 "NOT_DECLARED": "Not_declared", "NEEDS_REVIEW": ""}.get(v["label_v5"], "No")
        v["benchmark_version"] = "v5"
        v["dataset_status"] = "tiered_rule_verified_no_expert"
        v["expert_validated"] = "false"
        out.append(v)
    return out, fixes


def write_csv(path: Path, rows: list[dict]):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.src).read_text(encoding="utf-8").splitlines() if l.strip()]
    out, fixes = build(rows)
    OUT_JSONL.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    write_csv(OUT_CSV, out)
    write_csv(OUT_FIX, fixes or [{"id": "", "fix_code": "", "detail": "", "v4_label": ""}])
    summary = {
        "records": len(out),
        "tier": Counter(r["tier"] for r in out),
        "label_v5": Counter(r["label_v5"] for r in out),
        "fix_codes": Counter(f["fix_code"] for f in fixes),
        "fix_flags": Counter(f.split(":")[0] for r in out for f in r["fix_flags"].split(";") if f),
        "abstain_reason": Counter(r["abstain_reason"] for r in out if r["abstain_reason"]),
        "core_by_domain": Counter(r["domain"] for r in out if r["tier"] == "core"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
