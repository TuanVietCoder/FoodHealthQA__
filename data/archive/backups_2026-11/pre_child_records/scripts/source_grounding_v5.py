"""Rewrite v5 answers so that every regulatory statement is one that was checked against the primary text.

Verified claims (scripts_research/verify_external_claims.py):
  FALCPA_321qq_NINE_ALLERGENS     21 U.S.C. 321(qq): milk, egg, fish, crustacean shellfish, tree nuts, wheat,
                                  peanuts, soybeans, sesame (no molluscs)
  JEMRA1_PRIORITY_LIST            gluten cereals, crustacea, eggs, fish, milk, peanuts, sesame, six tree nuts
  JEMRA1_SOY_AND_OATS_NOT_GLOBAL  soybean / oats are not global priority allergens
  JEMRA3_PAL_REPORT               Part 3 is the PAL report; PAL systems neither uniform nor informative
  EFSA_2021_TIO2_NOT_SAFE         genotoxicity concern cannot be ruled out; E171 no longer considered safe
  TT24_PHULUC1_LISTS_INS_AND_NAME / TT24_ML_IN_MG_PER_KG   structure of TT 24/2019 Phụ lục 1 and 2
  (code membership of the positive list is checked per record in verify_v5_answers.py, gate G6)

Statements that rely on documents that cannot be fetched (Codex STAN 192 / CXG 36 / CXS 1, JECFA, EFSA full
opinions) are not made; a record that needs them is taken out of scoring (source_gap).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "external_claim_verification.json").read_text(encoding="utf-8"))

CRUSTACEAN = re.compile(r"(?<!\w)(tôm|cua|ghẹ|tép)(?!\w)|giáp xác|shrimp|prawn|crab|lobster|crustacean", re.I)
WHEAT = re.compile(r"lúa mì|bột mì|bột mỳ|wheat|semolina", re.I)
GLUTEN_CEREAL = re.compile(r"lúa mì|bột mì|bột mỳ|wheat|gluten|semolina|lúa mạch|mạch nha|barley|rye", re.I)
JEMRA_TREE_NUT = re.compile(r"hạt điều|hạnh nhân|óc chó|hạt phỉ|almond|cashew|walnut|hazelnut|pecan|pistachio", re.I)

NAME_VI = {"sữa": "sữa", "trứng": "trứng", "đậu phộng": "đậu phộng", "đậu nành": "đậu nành",
           "lúa mì": "lúa mì (gluten)", "cá": "cá", "giáp xác": "giáp xác", "động vật có vỏ": "động vật có vỏ",
           "hạt cây": "hạt cây", "mè": "mè/vừng"}

TT24_TAG = "[VN_TT24_2019]"
VERIFIED_TAGS = {"EFSA_OPINIONS", "US_FDA_21CFR172_804", "CODEX_STAN_192_1995", "CODEX_CXS_1_1985", "CODEX_CXG_36_1989", "JECFA_EVALUATIONS", "VN_TT24_2019", "US_FDA_FALCPA", "WHO_JEMRA_PRIORITY_ALLERGENS", "FAO_WHO_JEMRA_PART_3",
                 "EFSA_2021_TITANIUM_DIOXIDE_E171", "CODEX_CAC_GL_23_1997", "US_FDA_ALLERGEN_QA_ED5"}
LOC = {"US_FDA_ALLERGEN_QA_ED5": "US_FDA_ALLERGEN_QA_ED5#D9_RefinedOils", "CODEX_CAC_GL_23_1997": "CODEX_CAC_GL_23_1997#Table_Sodium",
       "US_FDA_FALCPA": "US_CODE_21_321#qq_major_food_allergen",
       "WHO_JEMRA_PRIORITY_ALLERGENS": "FAO_WHO_JEMRA_2021_PART1#PriorityAllergens",
       "FAO_WHO_JEMRA_PART_3": "FAO_WHO_JEMRA_2021_PART3#PAL_Guidance",
       "EFSA_2021_TITANIUM_DIOXIDE_E171": "EFSA_ANS_2021_6585#E171_NotSafe",
       "VN_TT24_2019": "VN_TT24_2019#PhuLuc1_Bang1",
       "CODEX_STAN_192_1995": "CODEX_STAN_192_1995#TableOne_TableThree",
       "CODEX_CXS_1_1985": "CODEX_CXS_1_1985#Sec4.2.1.4",
       "CODEX_CXG_36_1989": "CODEX_CXG_36_1989#ClassNames_INS"}


def claim_ok(*ids: str) -> bool:
    return all(CLAIMS.get(i, {}).get("passed") for i in ids)


def names(classes) -> str:
    return ", ".join(NAME_VI[c] for c in classes)


def coverage(cls: str, matched_text: str) -> tuple[bool, bool]:
    """(covered by 21 U.S.C. 321(qq), covered by JEMRA 2021 Part 1 priority list) for allergen class `cls`."""
    if cls in ("sữa", "trứng", "cá", "đậu phộng", "mè"):
        return True, True
    if cls == "đậu nành":
        return True, False
    if cls in ("giáp xác", "động vật có vỏ"):
        crust = bool(CRUSTACEAN.search(matched_text))
        return crust, crust
    if cls == "lúa mì":
        return bool(WHEAT.search(matched_text)), bool(GLUTEN_CEREAL.search(matched_text))
    if cls == "hạt cây":
        return True, bool(JEMRA_TREE_NUT.search(matched_text))
    return False, False


def coverage_codex(cls: str, matched_text: str) -> bool:
    """Covered by the mandatory allergen list of CXS 1-1985 s.4.2.1.4 (soy and molluscs are not)."""
    if cls in ("sữa", "trứng", "cá", "đậu phộng", "mè"):
        return True
    if cls in ("giáp xác", "động vật có vỏ"):
        return bool(CRUSTACEAN.search(matched_text))
    if cls == "lúa mì":
        return bool(GLUTEN_CEREAL.search(matched_text))
    if cls == "hạt cây":
        return bool(JEMRA_TREE_NUT.search(matched_text))
    return False


def source_sentences(classes, matched: dict, pal_topic: bool) -> tuple[str, list[str]]:
    """Return (sentence block, claim ids used) for the allergen classes the answer is about."""
    out, used = [], []
    falcpa = [c for c in classes if coverage(c, matched.get(c, ""))[0]]
    jemra = [c for c in classes if coverage(c, matched.get(c, ""))[1]]
    if falcpa and claim_ok("FALCPA_321qq_NINE_ALLERGENS"):
        shown = ["giáp xác" if c == "động vật có vỏ" else c for c in falcpa]
        out.append(f"Theo định nghĩa 'major food allergen' tại 21 U.S.C. 321(qq) (FALCPA), {names(dict.fromkeys(shown))} "
                   "thuộc nhóm dị nguyên chính. [US_FDA_FALCPA]")
        used.append("FALCPA_321qq_NINE_ALLERGENS")
    if jemra and claim_ok("JEMRA1_PRIORITY_LIST"):
        shown = ["giáp xác" if c == "động vật có vỏ" else c for c in jemra]
        out.append(f"FAO/WHO JEMRA (2021, Part 1) khuyến nghị {names(dict.fromkeys(shown))} thuộc danh sách dị nguyên "
                   "ưu tiên toàn cầu. [WHO_JEMRA_PRIORITY_ALLERGENS]")
        used.append("JEMRA1_PRIORITY_LIST")
    codex = [c for c in classes if coverage_codex(c, matched.get(c, ""))]
    if codex and claim_ok("CXS1_4214_MANDATORY_ALLERGEN_LIST"):
        shown = ["giáp xác" if c == "động vật có vỏ" else c for c in codex]
        out.append(f"Codex CXS 1-1985 (mục 4.2.1.4) quy định luôn phải khai báo {names(dict.fromkeys(shown))} là thực phẩm "
                   "gây dị ứng khi có chủ ý trong thực phẩm. [CODEX_CXS_1_1985]")
        used.append("CXS1_4214_MANDATORY_ALLERGEN_LIST")
    if "đậu nành" in classes and claim_ok("JEMRA1_SOY_AND_OATS_NOT_GLOBAL", "CXS1_4215_OPTIONAL_LIST_SOY_OATS"):
        out.append("Lưu ý: đậu nành không thuộc danh sách ưu tiên toàn cầu của JEMRA 2021 (Part 1) và không thuộc danh sách "
                   "bắt buộc ở CXS 1-1985 mục 4.2.1.4; CXS 1-1985 mục 4.2.1.5 xếp đậu nành vào danh sách có thể được yêu "
                   "cầu khai báo tùy dữ liệu đánh giá rủi ro của từng quần thể. [CODEX_CXS_1_1985]")
        used += ["JEMRA1_SOY_AND_OATS_NOT_GLOBAL", "CXS1_4215_OPTIONAL_LIST_SOY_OATS"]
    uncovered = [c for c in classes if not coverage(c, matched.get(c, ""))[0] and not coverage(c, matched.get(c, ""))[1]
                 and not coverage_codex(c, matched.get(c, ""))]
    if uncovered and claim_ok("FALCPA_321qq_NINE_ALLERGENS", "JEMRA1_PRIORITY_LIST", "CXS1_4214_MANDATORY_ALLERGEN_LIST"):
        out.append(f"Lưu ý: {names(uncovered)} không nằm trong định nghĩa 'major food allergen' của FALCPA "
                   "[US_FDA_FALCPA], danh sách ưu tiên JEMRA 2021 [WHO_JEMRA_PRIORITY_ALLERGENS] lẫn danh sách bắt buộc "
                   "của CXS 1-1985 mục 4.2.1.4 [CODEX_CXS_1_1985]; kết luận ở đây chỉ dựa trên thông tin kê khai trên nhãn.")
        used += ["FALCPA_321qq_NINE_ALLERGENS", "JEMRA1_PRIORITY_LIST", "CXS1_4214_MANDATORY_ALLERGEN_LIST"]
    if pal_topic and claim_ok("JEMRA3_PAL_REPORT"):
        out.append("Báo cáo FAO/WHO JEMRA Part 3 (2023) về nhãn cảnh báo dị nguyên phòng ngừa (PAL) nhận định các hệ "
                   "thống PAL hiện nay chưa thống nhất và chưa cung cấp đủ thông tin. [FAO_WHO_JEMRA_PART_3]")
        used.append("JEMRA3_PAL_REPORT")
    if pal_topic and claim_ok("CXS1_PAL_ANNEX_DEFINITION"):
        out.append("Phụ lục hướng dẫn PAL của Codex CXS 1-1985 (bản 2026) định nghĩa PAL là tuyên bố về nguy cơ hiện diện "
                   "ngoài ý muốn của chất gây dị ứng do tiếp xúc chéo, được xác định bằng đánh giá rủi ro. [CODEX_CXS_1_1985]")
        used.append("CXS1_PAL_ANNEX_DEFINITION")
    return " ".join(out), used


def allergen_answer(label_v5: str, name: str, asked: list[str], in_decl: list[str], in_pal: list[str],
                    matched: dict, extra_body: str = "", abstain_pal: bool = False):
    rest = [a for a in asked if a not in in_decl and a not in in_pal]
    rest_note = f" Không ghi nhận trên nhãn: {names(rest)}." if rest and label_v5 in ("UNSAFE", "CRITICAL_ALERT") else ""
    if label_v5 == "UNSAFE":
        extra = [a for a in in_pal if a not in in_decl]
        pal_note = f" Ngoài ra nhãn cảnh báo có thể nhiễm chéo {names(extra)}." if extra else ""
        body = (f"KHÔNG AN TOÀN — Nhãn '{name}' kê khai trực tiếp {names(in_decl)} trong thành phần hoặc phần cảnh báo."
                f"{pal_note} Người dị ứng {names(in_decl)} KHÔNG ĐƯỢC dùng sản phẩm này.{rest_note}")
        scope, pal_topic = in_decl, bool(extra)
    elif label_v5 == "CRITICAL_ALERT":
        body = (f"CẢNH BÁO KHẨN CẤP (NHIỄM CHÉO) — Thành phần của '{name}' không kê khai trực tiếp {names(in_pal)}, "
                f"nhưng nhãn cảnh báo sản phẩm có thể chứa/nhiễm chéo {names(in_pal)} từ dây chuyền hoặc thiết bị "
                f"sản xuất. Người dị ứng nặng cần tránh.{rest_note}")
        scope, pal_topic = in_pal, True
    elif label_v5 == "NOT_DECLARED":
        body = (f"KHÔNG KÊ KHAI — Danh sách thành phần và cảnh báo trên nhãn '{name}' không ghi nhận {names(asked)}. "
                "Nhãn cũng không cung cấp thông tin về nhiễm chéo dây chuyền (PAL), nên không thể kết luận sản phẩm "
                "không có nguy cơ nhiễm chéo.")
        scope, pal_topic = asked, True
    elif label_v5 == "ABSTAIN" and abstain_pal:
        body = (f"KHÔNG ĐỦ DỮ LIỆU — Danh sách thành phần của '{name}' không kê khai {names(asked)} và nhãn không ghi "
                f"cảnh báo nhiễm chéo (PAL) đối với chất này. Việc không có PAL không cho biết dây chuyền sản xuất có "
                f"tiếp xúc chéo với {names(asked)} hay không. Đề nghị liên hệ nhà sản xuất để xác nhận.")
        scope, pal_topic = asked, True
    else:  # CAUTION with a custom body (wheat/gluten grains, unspecified seafood)
        body, scope, pal_topic = extra_body, asked, False
    src, used = source_sentences(scope, matched, pal_topic)
    return f"{body} {src}".strip(), used


def permitted_answer(name: str, codes: str):
    return (f"CÓ TRONG DANH MỤC CHO PHÉP — Phụ gia {codes} trong '{name}' có tên trong Phụ lục 1 (Danh mục phụ gia "
            "được phép) của Thông tư 24/2019/TT-BYT. Phạm vi kết luận chỉ là tư cách được phép sử dụng: nhãn không "
            "ghi hàm lượng (mg/kg) và mã nhóm thực phẩm nên không đánh giá được tuân thủ mức tối đa (ML) và không đủ "
            "để kết luận mức an toàn tuyệt đối. [VN_TT24_2019]"), ["TT24_PHULUC1_LISTS_INS_AND_NAME"]


def abstain_quantity_answer(name: str, codes: str):
    return (f"KHÔNG ĐỦ DỮ LIỆU — Nhãn '{name}' liệt kê {codes} nhưng không ghi hàm lượng cụ thể (mg/kg) hay khối "
            "lượng khẩu phần, nên không thể tính liều tiếp nhận hằng ngày hoặc đối chiếu mức sử dụng tối đa (ML). "
            "Phụ lục 2 của Thông tư 24/2019/TT-BYT quy định ML theo mg/kg thực phẩm. Đề nghị bổ sung hàm lượng phụ gia "
            "hoặc liên hệ nhà sản xuất. [VN_TT24_2019]"), ["TT24_ML_IN_MG_PER_KG"]


def tio2_answer(name: str, codes: str):
    return (f"THẬN TRỌNG — Nhãn '{name}' có INS171 (titanium dioxide). Năm 2021 EFSA kết luận không thể loại trừ mối "
            "lo ngại về độc tính gen của hạt TiO2 và không còn coi E171 là an toàn khi dùng làm phụ gia thực phẩm. "
            "[EFSA_2021_TITANIUM_DIOXIDE_E171] Các phụ gia đã định danh trên nhãn "
            f"({codes}) có tên trong Phụ lục 1 của Thông tư 24/2019/TT-BYT; nhãn không ghi hàm lượng nên không đánh giá "
            "được tuân thủ mức tối đa (ML). [VN_TT24_2019]"), ["EFSA_2021_TIO2_NOT_SAFE", "TT24_PHULUC1_LISTS_INS_AND_NAME"]


def jecfa_adi_answer(name: str, codes: list[str]):
    """codes: normalised TT24-style codes such as '330' or '500(ii)'. Returns (answer, claim ids) or (None, [])."""
    items, used = [], []
    for c in codes:
        cl = CLAIMS.get(f"JECFA_ADI_{c}")
        if not cl or not cl["passed"]:
            return None, []
        adi = cl["adi"].capitalize()
        items.append(f"{cl['jecfa_name'].title()} (INS {c}): ADI '{adi}' (đánh giá {cl['year']})")
        used.append(f"JECFA_ADI_{c}")
    return (f"AN TOÀN VỀ NGƯỠNG ADI (THEO JECFA) — Theo cơ sở dữ liệu JECFA của WHO, các phụ gia đã định danh trên nhãn "
            f"'{name}' có ADI như sau: " + "; ".join(items) + ". JECFA không đặt ADI dạng số (mg/kg thể trọng/ngày) "
            "cho các chất này nên không có ngưỡng số để so sánh; nhãn cũng không ghi hàm lượng nên không đánh giá được "
            "mức phơi nhiễm thực tế. [JECFA_EVALUATIONS]"), used


def jecfa_locators(codes: list[str]) -> list[str]:
    out = []
    for c in codes:
        cl = CLAIMS.get(f"JECFA_ADI_{c}")
        if cl:
            out.append(f"WHO_FAO_JECFA#Chemical{cl['chemical_id']}_" + re.sub(r"[^A-Za-z0-9]+", "", cl["jecfa_name"].title()))
    return out


_CODEX192 = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "codex_192_ins_codes.json").read_text(encoding="utf-8"))     if (ROOT / "knowledge_base" / "sources" / "derived" / "codex_192_ins_codes.json").exists() else {"table_one": [], "table_three": []}
CODEX_T1, CODEX_T3 = set(_CODEX192["table_one"]), set(_CODEX192["table_three"])


def norm_code(c: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"(?i)^ins", "", c.strip().lower()))


def codex_member(code: str, table: set) -> bool:
    return code in table or any(x.startswith(code + "(") for x in table)


def codex192_sentence(codes: list[str]):
    """Sentence on CXS 192-1995 Table One / Table Three membership, only if every code is listed in the standard."""
    codes = [norm_code(c) for c in codes if c.strip()]
    if not codes or not claim_ok("CODEX192_TABLE1_TABLE3_PARSED"):
        return None, []
    if not all(codex_member(c, CODEX_T1) or codex_member(c, CODEX_T3) for c in codes):
        return None, []
    t3 = [c for c in codes if codex_member(c, CODEX_T3)]
    t1_only = [c for c in codes if c not in t3]
    parts = []
    if t3:
        parts.append("INS " + ", ".join(t3) + " nằm trong Table Three (được phép dùng theo GMP trong thực phẩm nói chung, "
                     "trừ khi có quy định khác)")
    if t1_only:
        parts.append("INS " + ", ".join(t1_only) + " có trong Table One (cho phép có điều kiện theo từng nhóm thực phẩm)")
    return ("Trong Codex CXS 192-1995 (GSFA, bản 2026): " + "; ".join(parts) + ". [CODEX_STAN_192_1995]",
            ["CODEX192_TABLE1_TABLE3_PARSED"])


def cxg36_sentence():
    if not claim_ok("CODEX_CXG36_INS_AND_CLASS_NAMES"):
        return None, []
    return ("Codex CXG 36-1989 cung cấp hệ thống số INS và tên nhóm chức năng để định danh phụ gia. "
            "[CODEX_CXG_36_1989]"), ["CODEX_CXG36_INS_AND_CLASS_NAMES"]


def tags_of(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[([A-Za-z0-9_.]+)\]", text)))


def locators_for(tags: list[str]) -> str:
    return "; ".join(LOC[t] for t in tags if t in LOC)


# ---------------------------------------------------------------------------
# Vulnerable-group answers restricted to what the primary texts say
# ---------------------------------------------------------------------------
PHOS_BASES = {"338", "339", "340", "341", "343", "450", "451", "452"}      # EFSA 2019 title: E 338-341, E 343, E 450-452
MCCANN_BASES = {"102", "104", "110", "122", "124", "129", "211"}          # EFSA 2008: colours + sodium benzoate tested
NITRITE_BASES = {"249", "250"}
TIO2_BASES = {"171"}
ASPARTAME_BASES = {"951"}

LOC_AUTHORITY = {
    "phosphate": "EFSA_ANS_2019_5674#Phosphates_ADI40",
    "mccann": "EFSA_ANS_2008_852#AzoDyes_Hyperactivity_Children",   # legacy key; the DOI behind it is now 2008.660
    "nitrite": "EFSA_ANS_2017_4786#Nitrite_Methemoglobin",
    "tio2": "EFSA_ANS_2021_6585#E171_NotSafe",
    "aspartame": "US_FDA_21CFR172.804#b",                             # legacy key; the statement is paragraph (d)(2)
}


def code_base(code: str) -> str:
    m = re.search(r"\d+", code)
    return m.group(0) if m else ""


def flagged(codes: list[str], bases: set) -> list[str]:
    return [c for c in dict.fromkeys(codes) if code_base(c) in bases]


def ckd_answer(name: str, codes: list[str]):
    fl = flagged(codes, PHOS_BASES)
    need = ["EFSA_OPINION_2019_PHOSPHATES"]
    if not claim_ok(*need):
        return None
    if fl:
        text = (f"THẬN TRỌNG — Nhãn '{name}' kê khai {', '.join(fl)} thuộc nhóm photphat (E 338–341, E 343, E 450–452) mà "
                "EFSA tái đánh giá năm 2019. EFSA nhận định ADI nhóm 40 mg/kg thể trọng/ngày (tính theo phospho) bảo vệ "
                "người lớn khỏe mạnh, nhưng ADI này không áp dụng cho người bị giảm chức năng thận mức vừa đến nặng. "
                "Vì vậy ADI không thể dùng làm căn cứ an toàn cho người suy thận mạn; nhãn không ghi hàm lượng nên cũng "
                "không ước tính được lượng phospho từ phụ gia. [EFSA_OPINIONS]")
        return "CAUTION", text, need, [LOC_AUTHORITY["phosphate"]], "authority_flag:efsa_2019_phosphate_group"
    text = (f"KHÔNG PHÁT HIỆN PHOTPHAT NHÓM EFSA — Nhãn '{name}' không kê khai phụ gia thuộc nhóm photphat (E 338–341, "
            "E 343, E 450–452) mà EFSA tái đánh giá năm 2019 (ADI nhóm 40 mg/kg thể trọng/ngày tính theo phospho, không "
            "áp dụng cho người giảm chức năng thận mức vừa đến nặng). Kết luận chỉ về nhóm phụ gia photphat này; không "
            "bao gồm phospho có sẵn trong thực phẩm hay phụ gia photphat nằm ngoài nhóm. [EFSA_OPINIONS]")
    return "SAFE", text, need, [LOC_AUTHORITY["phosphate"]], "authority_flag:efsa_2019_phosphate_group"


def child_answer(name: str, codes: list[str]):
    fl = flagged(codes, MCCANN_BASES)
    need = ["EFSA_OPINION_2008_MCCANN"]
    if not claim_ok(*need):
        return None
    study = ("tartrazine E102, quinoline yellow E104, sunset yellow E110, carmoisine E122, ponceau 4R E124, allura red "
             "E129 và natri benzoat E211")
    if fl:
        text = (f"THẬN TRỌNG — Nhãn '{name}' kê khai {', '.join(fl)}, thuộc nhóm phẩm màu/chất bảo quản được thử trong "
                f"nghiên cứu McCann và cộng sự (2007) về hành vi trẻ em ({study}). EFSA (2008) kết luận nghiên cứu này, "
                "thực hiện ở trẻ 3 tuổi và 8–9 tuổi, cung cấp bằng chứng hạn chế rằng hai hỗn hợp phẩm màu và natri "
                "benzoat có tác động nhỏ, có ý nghĩa thống kê lên hoạt động và sự chú ý của trẻ (tăng động); không thể "
                "quy cho từng chất riêng lẻ và không đủ làm cơ sở thay đổi ADI. Nghiên cứu không thử ở trẻ 5 tuổi. "
                "[EFSA_OPINIONS]")
        return "CAUTION", text, need, [LOC_AUTHORITY["mccann"]], "authority_flag:efsa_2008_mccann_substances"
    text = (f"KHÔNG PHÁT HIỆN CHẤT TRONG NGHIÊN CỨU McCANN — Nhãn '{name}' không kê khai các phẩm màu và chất bảo quản "
            f"được thử trong nghiên cứu McCann và cộng sự (2007) về hành vi trẻ em mà EFSA (2008) đánh giá ({study}). "
            "Kết luận chỉ về nhóm 7 chất này, không phải đánh giá toàn bộ phụ gia trong sản phẩm. [EFSA_OPINIONS]")
    return "SAFE", text, need, [LOC_AUTHORITY["mccann"]], "authority_flag:efsa_2008_mccann_substances"


def pku_answer(name: str, codes: list[str]):
    fl = flagged(codes, ASPARTAME_BASES)
    need = ["CFR_172_804_PKU_STATEMENT"]
    if not claim_ok(*need):
        return None
    if fl:
        text = (f"CẢNH BÁO KHẨN CẤP (PKU) — Nhãn '{name}' kê khai {', '.join(fl)} (aspartame). Theo 21 CFR 172.804(d)(2), "
                "thực phẩm chứa aspartame phải ghi 'PHENYLKETONURICS: CONTAINS PHENYLALANINE', tức sản phẩm chứa nguồn "
                "phenylalanine; câu cảnh báo này dành cho người bị phenylketon niệu (PKU). Người bị PKU nên tránh sản "
                "phẩm này hoặc hỏi ý kiến bác sĩ. [US_FDA_21CFR172_804]")
        return "CRITICAL_ALERT", text, need, [LOC_AUTHORITY["aspartame"]], "authority_flag:cfr_172_804_aspartame"
    text = (f"KHÔNG PHÁT HIỆN ASPARTAME — Nhãn '{name}' không kê khai aspartame (INS951), chất mà 21 CFR 172.804(d)(2) "
            "yêu cầu ghi câu 'PHENYLKETONURICS: CONTAINS PHENYLALANINE' cho người bị phenylketon niệu (PKU). Kết luận "
            "chỉ về aspartame; không loại trừ phenylalanine có sẵn trong protein hoặc các thành phần khác. "
            "[US_FDA_21CFR172_804]")
    return "SAFE", text, need, [LOC_AUTHORITY["aspartame"]], "authority_flag:cfr_172_804_aspartame"


def pregnancy_answer(name: str, codes: list[str]):
    tio2 = flagged(codes, TIO2_BASES)
    nit = flagged(codes, NITRITE_BASES)
    if tio2 and claim_ok("EFSA_2021_TIO2_FULLTEXT"):
        text = (f"THẬN TRỌNG — Nhãn '{name}' có INS171 (titanium dioxide). Năm 2021 EFSA kết luận không thể loại trừ mối "
                "lo ngại về độc tính gen của hạt TiO2 và E171 không còn được coi là an toàn khi dùng làm phụ gia thực "
                "phẩm. Kết luận này áp dụng cho E171 như một phụ gia thực phẩm nói chung, nên thai phụ cũng nằm trong "
                "phạm vi cần thận trọng; nhãn không ghi hàm lượng. [EFSA_2021_TITANIUM_DIOXIDE_E171]")
        return "CAUTION", text, ["EFSA_2021_TIO2_FULLTEXT"], [LOC_AUTHORITY["tio2"]], "authority_flag:efsa_2021_e171"
    if nit and claim_ok("EFSA_OPINION_2017_NITRITE"):
        text = (f"THẬN TRỌNG — Nhãn '{name}' kê khai {', '.join(nit)} (nitrit). EFSA (2017) đặt ADI 0,07 mg ion nitrit/kg "
                "thể trọng/ngày, dựa trên tác động làm tăng methaemoglobin. Báo cáo của EFSA trích dẫn Hội đồng Bộ "
                "trưởng Bắc Âu xếp phụ nữ mang thai vào nhóm nhạy cảm hơn với khả năng tạo methaemoglobin của nitrit. "
                "Nhãn không ghi hàm lượng nên không so được với ADI; thai phụ nên hỏi ý kiến bác sĩ. [EFSA_OPINIONS]")
        return "CAUTION", text, ["EFSA_OPINION_2017_NITRITE"], [LOC_AUTHORITY["nitrite"]], "authority_flag:efsa_2017_nitrite"
    return None


_UNIT_AFTER = r"(?!\s?(?:mg|g|ml|kg|l|kcal|kj|%|ppm|µg|mcg|iu|cfu)\b)"
_STRICT_CODE = re.compile(r"(?<![\d.,])(\d{3,4}[a-z]?(?:\s?\([ivx]+\))*)(?![\d])" + _UNIT_AFTER, re.I)


def strict_label_codes(label: dict) -> list[str]:
    """INS-like numbers on the label text, ignoring decimals and quantities (e.g. '450 mg / 100 ml' is not INS 450)."""
    parts = []
    for k in ("ingredient", "additive", "warning"):
        v = label.get(k)
        parts.append(" ".join(map(str, v)) if isinstance(v, list) else str(v or ""))
    text = re.sub(r"(?<!\d)\d{1,2}[.,]\d+", " ", " ".join(parts).lower())
    text = text.replace(",", " ").replace(";", " ")
    return list(dict.fromkeys(re.sub(r"\s+", "", m.group(1)).lower() for m in _STRICT_CODE.finditer(text)))


try:
    from additive_names import name_codes
except ImportError:  # imported as a package
    from scripts_research.additive_names import name_codes

ALL_GROUP_BASES = PHOS_BASES | MCCANN_BASES | NITRITE_BASES | TIO2_BASES | ASPARTAME_BASES


def merge_declared_codes(linker_codes: list[str], label: dict) -> list[str]:
    """Linker codes (v4) plus group-relevant codes the linker missed, e.g. a code written in square brackets."""
    have = {code_base(c) for c in linker_codes}
    extra = []
    for c in strict_label_codes(label):
        b = code_base(c)
        if b in ALL_GROUP_BASES and b not in have:
            extra.append("INS" + c.upper().replace("(", "(").replace(")", ")"))
            have.add(b)
    # additives written by name only (no INS code): see additive_names.py
    for c in name_codes(label):
        b = code_base(c)
        if b in ALL_GROUP_BASES and b not in have:
            extra.append(c)
            have.add(b)
    return list(linker_codes) + extra


SODIUM_LOW_MG_PER_100 = 120.0   # Codex CAC/GL 23-1997 table: sodium low = 0.12 g per 100 g
SODIUM_VERY_LOW_MG_PER_100 = 40.0


def _num(text: str):
    m = re.search(r"\d+(?:[.,]\d+)?", text or "")
    return float(m.group(0).replace(",", ".")) if m else None


def _vn(x: float) -> str:
    s = f"{x:.1f}".replace(".", ",")
    return s[:-2] if s.endswith(",0") else s


def sodium_codex_answer(name: str, label: dict):
    """Hypertension/sodium record rebuilt on a threshold that matches the label's own basis (per 100 g/ml).
    Returns (label_v5, question, answer, grounded_claims) or None when the label has no usable basis."""
    basis = label.get("sodium_basis")
    if not basis:
        return None
    entry = next((x for x in label.get("nutrition") or [] if re.search(r"natri|sodium", x.get("name", ""), re.I)), None)
    if not entry:
        return None
    raw = str(entry.get("value", "")).strip()
    raw = re.sub(r"\(\s*\d+(?:[.,]\d+)?\s*%\s*\)", "", raw)          # "853 mg (43 %)" -> %NRV column, not a quantity
    raw = re.sub(r"\(\s*mg\s*\)", "mg", raw, flags=re.I).strip()       # "5843,3 (mg)"
    val = _num(raw)
    if val is None or not re.fullmatch(r"\s*\d+(?:[.,]\d+)?\s*(?:mg)?\s*", raw, re.I):
        return None
    amount, unit, phrase = float(basis["amount"]), basis["unit"], basis["phrase"]
    per100 = val / amount * 100
    n = _vn(per100)
    shown = raw if re.search(r"mg", raw, re.I) else raw + " mg"
    low = per100 <= SODIUM_LOW_MG_PER_100
    liquid = " (thực phẩm lỏng: coi 100 ml tương đương 100 g)" if unit == "ml" else ""
    q = (f"Tôi bị tăng huyết áp — nhãn '{name}' ghi natri {shown} {phrase} (quy đổi khoảng {n} mg/100 {unit}). "
         "Mức này có đạt điều kiện 'natri thấp' (không quá 120 mg/100 g) theo Codex CAC/GL 23-1997 không?")
    if low:
        very = (" và cũng không vượt 40 mg/100 g nên đạt cả mức 'natri rất thấp'"
                if per100 <= SODIUM_VERY_LOW_MG_PER_100 else "")
        a = (f"AN TOÀN VỀ HÀM LƯỢNG NATRI (đạt ngưỡng natri thấp) — Nhãn '{name}' ghi natri {shown} {phrase}, quy đổi khoảng {n} mg/100 {unit}{liquid}. "
             f"Mức này không vượt 120 mg/100 g, ngưỡng 'natri thấp' của Codex CAC/GL 23-1997{very}. "
             "Kết luận chỉ xét riêng natri trên nhãn theo 100 g/100 ml, không thay thế việc tính tổng lượng natri "
             "trong ngày của người tăng huyết áp. [CODEX_CAC_GL_23_1997]")
        return "SAFE", q, a, ["CODEX_GL23_SODIUM_PER_100G"]
    a = (f"THẬN TRỌNG VỀ HÀM LƯỢNG NATRI (không đạt ngưỡng natri thấp) — Nhãn '{name}' ghi natri {shown} {phrase}, quy đổi khoảng {n} mg/100 {unit}{liquid}. "
         "Mức này vượt 120 mg/100 g nên không đạt điều kiện 'natri thấp' của Codex CAC/GL 23-1997. "
         "Đây là so sánh với ngưỡng công bố dinh dưỡng, không phải kết luận y khoa về mức an toàn; người tăng huyết áp "
         "cần đối chiếu với chỉ dẫn điều trị riêng và lượng dùng thực tế. [CODEX_CAC_GL_23_1997]")
    return "CAUTION", q, a, ["CODEX_GL23_SODIUM_PER_100G"]


def _jn(c: str) -> str:
    return re.sub(r"[\s()]+", "", re.sub(r"(?i)^ins", "", c.strip().lower()))


def _jecfa_by_norm() -> dict:
    return {_jn(k[len("JECFA_ADI_"):]): (k, v) for k, v in CLAIMS.items()
            if k.startswith("JECFA_ADI_") and v.get("passed") and v.get("adi")}


_NO_NUMERIC_ADI = re.compile(r"^(NOT SPECIFIED|NOT LIMITED)", re.I)


def jecfa_status_answer(name: str, raw_codes: list[str]):
    """Answer for a record whose codes were outside the 50-substance KB, built only from JECFA pages that were fetched.
    Returns (label, answer, claim ids, locators) or None unless EVERY code has a verified JECFA record."""
    table = _jecfa_by_norm()
    codes = list(dict.fromkeys(_jn(c) for c in raw_codes))
    if not codes or any(c not in table for c in codes):
        return None
    items, used, locs = [], [], []
    all_open = True
    for c in codes:
        key, cl = table[c]
        shown = raw_codes[[_jn(x) for x in raw_codes].index(c)].upper().replace("INS", "").strip()
        adi = re.split(r"\s+(?:Intake|Comments?:)", cl["adi"].strip())[0].strip()   # drop JECFA remarks that follow the ADI
        if not _NO_NUMERIC_ADI.match(adi):
            all_open = False
        items.append(f"{cl['jecfa_name'].title()} (INS {shown}): ADI '{adi}' (đánh giá {cl['year']})")
        used.append(key)
        locs.append(f"WHO_FAO_JECFA#Chemical{cl['chemical_id']}_" + re.sub(r"[^A-Za-z0-9]+", "", cl["jecfa_name"].title()))
    if all_open:
        return ("SAFE",
                f"AN TOÀN VỀ NGƯỠNG ADI (THEO JECFA) — Theo cơ sở dữ liệu JECFA của WHO, các phụ gia đã định danh trên nhãn "
                f"'{name}' có ADI như sau: " + "; ".join(items) + ". JECFA không đặt ADI dạng số (mg/kg thể trọng/ngày) "
                "cho các chất này nên không có ngưỡng số để so sánh; nhãn cũng không ghi hàm lượng nên không đánh giá được "
                "mức phơi nhiễm thực tế. [JECFA_EVALUATIONS]", used, locs)
    return ("ABSTAIN",
            f"KHÔNG ĐỦ DỮ LIỆU — Theo cơ sở dữ liệu JECFA của WHO, các phụ gia đã định danh trên nhãn '{name}' có ADI như sau: "
            + "; ".join(items) + ". Với chất có ADI dạng số (mg/kg thể trọng/ngày), cần hàm lượng phụ gia và khối lượng khẩu "
            "phần để tính mức phơi nhiễm, nhưng nhãn không ghi các thông tin này nên không thể kết luận mức tiếp nhận có "
            "vượt ADI hay không. [JECFA_EVALUATIONS]", used, locs)


# ---------------------------------------------------------------------------
# Additive records backed by three later-added primary sources (see verify_external_claims.new_authority_claims):
#   EU Reg 1333/2008 Annex V, FDA order FR 2025-00830 (FD&C Red No. 3), IARC Monographs Vol. 134 (aspartame)
# ---------------------------------------------------------------------------
ANNEXV_BASES = {"102", "104", "110", "122", "124", "129"}
RED3_BASES = {"127"}
IARC_ASPARTAME_BASES = {"951"}
SULPHITE_BASES = {"220", "221", "222", "223", "224", "225", "226", "227", "228"}
# WHO 2023 NSS guideline (p.23 remarks) names: acesulfame K, aspartame, advantame, cyclamates, neotame, saccharin,
# sucralose, stevia (+ derivatives). Polyols (sorbitol 420, maltitol 965, erythritol 968...) are explicitly NOT NSS.
NSS_NAMES = {"950": "acesulfame K", "951": "aspartame", "952": "cyclamate", "954": "saccharin", "955": "sucralose",
             "960": "steviol glycosides (stevia)", "961": "neotame", "969": "advantame"}
NSS_BASES = set(NSS_NAMES)
# A label may be used as a "no NSS" control only when NOTHING on it points to a sweetener. Deliberately broad: any
# sweetener name, the word 'ngọt'/'sweet', or a bare NSS number anywhere in the transcription (even in the
# nutrition table) drops the label, because a wrong SAFE is worse than a smaller control set.
_NSS_ANY = re.compile(r"aspartam|acesulfam|sucralos|sacchar|sacar|cyclamat|xyclamat|neotam|advantam|stevi|rebaudi|"
                      r"cỏ ngọt|ngọt|sweet|đường hóa học|thaumatin", re.I)
_NSS_NUMBER = re.compile(r"(?<![\d.,])(?:950|951|952|954|955|957|960|961|962|969)(?![\d])")


def nss_absent(label: dict) -> bool:
    """True when the ingredient list is present and nothing on the label suggests an NSS (see _NSS_ANY)."""
    ing = label.get("ingredient")
    ing_text = " ".join(map(str, ing)) if isinstance(ing, list) else str(ing or "")
    if len(ing_text.strip()) < 30:  # an empty / tiny ingredient list is not evidence of absence
        return False
    whole = json.dumps(label, ensure_ascii=False)
    return not (_NSS_ANY.search(whole) or _NSS_NUMBER.search(whole) or declared_bases(label, NSS_BASES))


def nss_question(name: str) -> str:
    """One neutral wording for flagged records AND no-NSS controls, so the question never reveals the label."""
    return (f"Tôi đang mang thai — nhãn '{name}' có chứa chất ngọt không đường (như aspartame, acesulfame K, sucralose, "
            "saccharin, stevia) mà hướng dẫn của WHO về chất ngọt không đường nói đến không?")


def nss_absent_record(name: str):
    """-> (question, answer, grounded_claims, scope). SAFE only in the narrow sense 'no NSS declared'."""
    return (nss_question(name),
            f"KHÔNG PHÁT HIỆN CHẤT NGỌT KHÔNG ĐƯỜNG — Nhãn '{name}' không kê khai chất ngọt không đường (NSS) nào. Hướng dẫn "
            "của WHO năm 2023 nêu các NSS thường gặp gồm acesulfame K, aspartame, advantame, cyclamate, neotame, "
            "saccharin, sucralose, stevia và dẫn xuất của stevia; vì vậy lưu ý của WHO về NSS khi mang thai (bằng chứng "
            "hạn chế về nguy cơ sinh non) không áp dụng cho sản phẩm này qua thành phần đã kê khai. Kết luận chỉ về NSS: "
            "không đánh giá các phụ gia hay yếu tố khác trên nhãn, và polyol (đường rượu như sorbitol, maltitol, "
            "erythritol, xylitol) không được WHO xếp vào NSS. [WHO_NSS_GUIDELINE_2023]",
            ["WHO_NSS_2023_PREGNANCY"], "authority_flag:who_nss_pregnancy")
SULPHITE_NAME = re.compile(r"sul[fp]it|sulphite|sulfite|lưu huỳnh", re.I)

VERIFIED_TAGS |= {"EU_REG_1333_2008_ANNEX_V", "US_FDA_RED3_REVOCATION", "IARC_MONOGRAPHS_V134", "WHO_NSS_GUIDELINE_2023"}
LOC.update({"EU_REG_1333_2008_ANNEX_V": "EU_REG_1333_2008#AnnexV_ColourWarning",
            "US_FDA_RED3_REVOCATION": "US_FR_2025_00830#Red3_Revocation",
            "IARC_MONOGRAPHS_V134": "IARC_MONOGRAPHS_V134#Aspartame_Group2B",
            "WHO_NSS_GUIDELINE_2023": "WHO_NSS_GUIDELINE_2023#Pregnancy"})
AUTHORITY_TAG_PREFIX = {"EU_REG_1333_2008_ANNEX_V": "EU_ANNEXV", "US_FDA_RED3_REVOCATION": "FDA_RED3",
                        "IARC_MONOGRAPHS_V134": "IARC_V134", "WHO_NSS_GUIDELINE_2023": "WHO_NSS_2023"}

_RED3_NAME = re.compile(r"erythrosin", re.I)


def declared_bases(label: dict, bases: set[str]) -> list[str]:
    """INS base numbers of the group that the label really declares (code or, where known, name), in label order."""
    found = [code_base(c) for c in strict_label_codes(label)]
    found += [code_base(c) for c in name_codes(label)]
    if bases & RED3_BASES:
        text = " ".join(str(label.get(k) or "") for k in ("ingredient", "additive"))
        if _RED3_NAME.search(text):
            found.append("127")
    return [b for b in dict.fromkeys(found) if b in bases]


def authority_topic_record(topic: str, name: str, bases: list[str]):
    """-> (question, answer, grounded_claims, scope). Verdict is always CAUTION: a regulatory/hazard flag, not a dose finding."""
    codes = ", ".join("INS" + b for b in bases)
    if topic == "eu_annex_v_colours":
        return (f"'{name}' chứa phụ gia {codes} — theo Quy định (EC) 1333/2008 của Liên minh châu Âu, thực phẩm chứa "
                "các màu này có phải ghi thêm cảnh báo về trẻ em không?",
                f"THẬN TRỌNG — Nhãn '{name}' có {codes}. Phụ lục V của Quy định (EC) 1333/2008 (EU) liệt kê sáu màu thực "
                "phẩm: Sunset yellow (E 110), Quinoline yellow (E 104), Carmoisine (E 122), Allura red (E 129), "
                "Tartrazine (E 102) và Ponceau 4R (E 124). Thực phẩm chứa một trong các màu này phải ghi thêm tên hoặc số E "
                "của màu kèm câu cảnh báo 'may have an adverse effect on activity and attention in children' (có thể ảnh "
                "hưởng xấu đến hoạt động và sự chú ý ở trẻ em). Ngoại lệ gồm thực phẩm dùng màu để đóng dấu hoặc đánh dấu "
                "trên sản phẩm thịt, trang trí vỏ trứng, và đồ uống có trên 1,2% thể tích cồn. Đây là quy định ghi nhãn "
                "của EU, không suy ra yêu cầu tương ứng ở Việt Nam; nhãn không ghi hàm lượng màu. "
                "[EU_REG_1333_2008_ANNEX_V]",
                ["EU_ANNEXV_COLOURS_WARNING"], "authority_flag:eu_annex_v_colours")
    if topic == "fda_red3_revoked":
        return (f"'{name}' chứa phụ gia {codes} (erythrosine, FD&C Red No. 3) — cơ quan FDA của Mỹ có bãi bỏ quy "
                "định cho phép dùng màu này trong thực phẩm không?",
                f"THẬN TRỌNG — Nhãn '{name}' có {codes} (erythrosine, FD&C Red No. 3). Trong lệnh đăng Federal Register "
                "ngày 16/01/2025 (FR Doc. 2025-00830), FDA đã bãi bỏ quy định cho phép dùng FD&C Red No. 3 trong thực phẩm "
                "(kể cả thực phẩm bổ sung) và trong thuốc uống, vì dữ liệu cho thấy chất này gây ung thư ở chuột đực; theo "
                "điều khoản Delaney, FDA không thể coi một màu là an toàn khi nó gây ung thư ở người hoặc động vật, bất kể "
                "xác suất hay mức rủi ro. Lệnh có hiệu lực từ 15/01/2027. Đây là quyết định quản lý của Mỹ, không phải "
                "đánh giá nguy cơ ung thư ở người ở một mức tiêu thụ cụ thể, và nhãn không ghi hàm lượng. "
                "[US_FDA_RED3_REVOCATION]",
                ["FDA_RED3_REVOKED_FOOD"], "authority_flag:fda_red3_revoked")
    if topic == "iarc_v134_aspartame":
        return (f"'{name}' chứa phụ gia {codes} (aspartame) — IARC (Chuyên khảo tập 134, 2023) xếp aspartame vào nhóm "
                "nguy cơ gây ung thư nào?",
                f"THẬN TRỌNG — Nhãn '{name}' có {codes} (aspartame). IARC Monographs tập 134 (2023) kết luận aspartame "
                "'possibly carcinogenic to humans' (Nhóm 2B), dựa trên bằng chứng hạn chế ở người (ung thư gan tế bào gan), "
                "bằng chứng hạn chế ở động vật thí nghiệm và bằng chứng hạn chế về cơ chế. Phân loại này mô tả mức độ "
                "bằng chứng về khả năng gây ung thư, không phải rủi ro ung thư ở một mức tiêu thụ cụ thể; nhãn không ghi "
                "lượng aspartame nên không thể ước tính mức phơi nhiễm. [IARC_MONOGRAPHS_V134]",
                ["IARC_V134_ASPARTAME_2B"], "authority_flag:iarc_v134_aspartame")
    if topic == "who_nss_pregnancy":
        named = ", ".join(f"INS{b} ({NSS_NAMES[b]})" for b in bases)
        return (nss_question(name),
                f"THẬN TRỌNG — Nhãn '{name}' có {named}, thuộc nhóm chất ngọt không đường (NSS) theo hướng dẫn của WHO "
                "năm 2023. WHO đưa ra khuyến cáo có điều kiện không dùng NSS để kiểm soát cân nặng hoặc giảm nguy cơ bệnh "
                "không lây, và nêu rõ khuyến cáo này áp dụng cho mọi người, kể cả phụ nữ mang thai và cho con bú (trừ người "
                "đã có đái tháo đường). Riêng với thai kỳ, bằng chứng còn hạn chế: phân tích gộp ba nghiên cứu đoàn hệ cho "
                "thấy dùng NSS nhiều hơn liên quan đến tăng nguy cơ sinh non (OR 1,25; KTC 95% 1,07-1,46; độ chắc chắn "
                "thấp), và không thấy liên quan với đái tháo đường thai kỳ. Đây là bằng chứng quan sát, WHO lưu ý khả "
                "năng nhiễu và quan hệ nhân quả ngược, không chứng minh nhân quả, và không đặt ngưỡng an toàn riêng cho "
                "thai phụ; nhãn không ghi lượng NSS nên không thể ước tính mức dùng. Thai phụ nên hỏi ý kiến bác sĩ. "
                "[WHO_NSS_GUIDELINE_2023]",
                ["WHO_NSS_2023_PREGNANCY"], "authority_flag:who_nss_pregnancy")
    if topic == "codex_sulphite_declaration":
        return (f"'{name}' ghi phụ gia {codes} (nhóm sulphite) — theo Codex CXS 1-1985, khi sulphite có từ 10 mg/kg trở "
                "lên thì nhãn có phải khai báo bằng tên 'sulphite' hoặc 'sulfite' không?",
                f"THẬN TRỌNG — Nhãn '{name}' ghi {codes}, thuộc nhóm sulphite, nhưng chỉ bằng mã số, không có chữ "
                "'sulphite' hay 'sulfite'. Codex CXS 1-1985 (mục 4.2.1.7) quy định sulphite có mặt từ 10 mg/kg trở lên "
                "(tính theo SO2) phải luôn được khai báo bằng tên 'sulphite' hoặc 'sulfite'. Nhãn không ghi hàm lượng "
                "SO2 nên không thể kết luận sản phẩm có vi phạm quy định này hay không; người nhạy cảm với sulphite "
                "nên lưu ý rằng nhãn chưa nêu tên gọi đó. [CODEX_CXS_1_1985]",
                ["CXS1_4217_SULPHITE_DECLARATION"], "authority_flag:codex_sulphite_declaration")
    raise ValueError(topic)


# ---------------------------------------------------------------------------
# INS coverage records: one code per question, answers built ONLY by the existing source-grounded builders
#   ins_listing -> TT24 Phụ lục 1 listing status (+ Codex 192 sentence when every code is in the standard)
#   ins_adi     -> JECFA ADI status (only when the code has a fetched, verified JECFA record)
# ---------------------------------------------------------------------------
LOOKUP_TOPICS = {"ins_listing", "ins_adi", "sugar_codex"}
_CANON = re.compile(r"(\d{3,4})([a-f]?)(?:\(?(i{1,3}|iv|v|vi{1,3})\)?)?")


def canon_code(token: str):
    """'322i' / '322(I)' -> '322(i)'; '160a(iv)' stays; '100g' (grams), '452x' -> None."""
    m = _CANON.fullmatch(re.sub(r"\s+", "", token.lower()))
    if not m:
        return None
    base, letter, roman = m.groups()
    return base + letter + (f"({roman})" if roman else "")


def ins_lookup_record(topic: str, name: str, codes: list[str], label: dict | None = None):
    """-> (question, answer, label_v5, claims, locators, scope) or None when a source for any code is missing."""
    shown = ", ".join("INS" + re.sub(r"(?i)^ins", "", c.strip()) for c in codes)
    if topic == "ins_listing":
        tt = set(json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "tt24_positive_list_codes.json").read_text(encoding="utf-8"))["codes"])
        if not codes or not all(codex_member(norm_code(c), tt) for c in codes):
            return None
        ans, claims = permitted_answer(name, shown)
        cx, cxc = codex192_sentence(codes)
        if cx:
            ans, claims = ans + " " + cx, claims + cxc
        return (f"'{name}' chứa phụ gia {shown} — những phụ gia này có nằm trong danh mục phụ gia thực phẩm được phép "
                "sử dụng theo quy định không?", ans, "PERMITTED_LISTED", claims, [], "listed_in_positive_list_only")
    if topic == "sugar_codex":
        res = sugar_codex_answer(name, label or {})
        if not res:
            return None
        lab, q, a, claims = res
        return (q, a, lab, claims, ["CODEX_CAC_GL_23_1997#Table_Sugars"], "sugar_free_codex_gl23_per_100g")
    if topic == "ins_adi":
        res = jecfa_status_answer(name, codes)
        if not res:
            return None
        label, ans, claims, locs = res
        return (f"Tôi lo ngại về phụ gia {shown} trong '{name}' — những chất này có gây hại gì không?", ans, label,
                claims, locs, "adi_status_per_jecfa_only")
    raise ValueError(topic)


def code_in_label_tokens(code: str, label: dict) -> bool:
    """The (canonical) code really appears as a token on the label text."""
    want = canon_code(norm_code(code))
    return bool(want) and any(canon_code(t) == want for t in strict_label_codes(label))


# ---------------------------------------------------------------------------
# Sugars: Codex CAC/GL 23-1997 "sugars free" = not more than 0.5 g per 100 g (solids) / 100 ml (liquids).
# Same basis rule as sodium: the label's nutrition table must state its basis (sodium_basis, shared by the whole table).
# ---------------------------------------------------------------------------
SUGAR_FREE_G_PER_100 = 0.5
_SUGAR_NAME = re.compile(r"đường|sugar", re.I)
_SUGAR_NOT_TOTAL = re.compile(r"bổ sung|thêm|added|lactose|glucose|fructose|sucrose|maltose|huyết|trans|alcohol|rượu|polyol", re.I)


def sugar_entry(label: dict):
    """The label's total-sugars entry (name, raw value) or None when absent/ambiguous."""
    found = [x for x in (label.get("nutrition") or []) if isinstance(x, dict)
             and _SUGAR_NAME.search(x.get("name", "")) and not _SUGAR_NOT_TOTAL.search(x.get("name", ""))]
    return found[0] if len(found) == 1 else None


def sugar_per100(label: dict):
    """-> (value_in_table, raw_shown, per100_g_or_ml, basis) or None. Only a plain 'N g' value on a stated basis."""
    basis, entry = label.get("sodium_basis"), sugar_entry(label)
    if not basis or not entry:
        return None
    raw = re.sub(r"\(\s*\d+(?:[.,]\d+)?\s*%\s*\)", "", str(entry.get("value", ""))).strip()
    if re.search(r"[<>≤≥~]", raw) or not re.fullmatch(r"\d+(?:[.,]\d+)?\s*g?", raw, re.I):
        return None
    if "g" not in raw.lower() and "(g)" not in entry.get("name", "").lower():
        return None
    val = _num(raw)
    amount = float(basis["amount"])
    if val is None or amount <= 0:
        return None
    return val, (raw if "g" in raw.lower() else raw + " g"), val / amount * 100, basis


def sugar_codex_answer(name: str, label: dict):
    """Diabetes-persona record on the Codex 'sugars free' threshold. -> (label_v5, question, answer, claims) or None."""
    got = sugar_per100(label)
    if not got:
        return None
    _, shown, per100, basis = got
    n = _vn(per100)
    unit, phrase = basis["unit"], basis["phrase"]
    liquid = " (thực phẩm lỏng: coi 100 ml tương đương 100 g)" if unit == "ml" else ""
    q = (f"Tôi bị đái tháo đường — nhãn '{name}' ghi đường tổng số {shown} {phrase} (quy đổi khoảng {n} g/100 {unit}). "
         "Mức này có đạt điều kiện 'không đường' (không quá 0,5 g/100 g) theo Codex CAC/GL 23-1997 không?")
    if per100 <= SUGAR_FREE_G_PER_100:
        a = (f"AN TOÀN VỀ HÀM LƯỢNG ĐƯỜNG (đạt ngưỡng không đường) — Nhãn '{name}' ghi đường tổng số {shown} {phrase}, "
             f"quy đổi khoảng {n} g/100 {unit}{liquid}. Mức này không vượt 0,5 g/100 g, ngưỡng 'không đường' (sugars free) "
             "của Codex CAC/GL 23-1997. Kết luận chỉ xét riêng lượng đường trên nhãn theo 100 g/100 ml; tinh bột và các "
             "carbohydrate khác vẫn ảnh hưởng đường huyết, nên đây không thay thế chỉ dẫn điều trị của người đái tháo "
             "đường. [CODEX_CAC_GL_23_1997]")
        return "SAFE", q, a, ["CODEX_GL23_SUGARS_FREE_PER_100G"]
    a = (f"THẬN TRỌNG VỀ HÀM LƯỢNG ĐƯỜNG (không đạt ngưỡng không đường) — Nhãn '{name}' ghi đường tổng số {shown} {phrase}, "
         f"quy đổi khoảng {n} g/100 {unit}{liquid}. Mức này vượt 0,5 g/100 g nên không đạt điều kiện 'không đường' (sugars "
         "free) của Codex CAC/GL 23-1997. Đây là so sánh với ngưỡng công bố dinh dưỡng, không phải kết luận y khoa về "
         "mức đường an toàn cho người đái tháo đường; cần đối chiếu với chỉ dẫn điều trị riêng và lượng dùng thực tế. "
         "[CODEX_CAC_GL_23_1997]")
    return "CAUTION", q, a, ["CODEX_GL23_SUGARS_FREE_PER_100G"]
