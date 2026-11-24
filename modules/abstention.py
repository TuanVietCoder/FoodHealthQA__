"""Two-type abstention: missing data vs. out of scope.

The guardrail refuses (or adds a referral) in exactly two situations, each with its own advice:

  insufficient_data  the label / user input lacks something the question needs
                     -> ask the user to re-photograph the label or supply the missing information
  out_of_scope       the question asks for something a label-screening system must not decide
                     (diagnosis, treatment, medication, personal medical decisions)
                     -> refer the user to a suitable professional

Everything here is a pure function of the question text and the label mentions, so it behaves the same
for Config B (reference text) and Config C (OCR text) and does not depend on OCR confidence.
"""

from __future__ import annotations

import re
import unicodedata

INSUFFICIENT_DATA = "insufficient_data"
OUT_OF_SCOPE = "out_of_scope"

# text on a label that declares possible cross-contact (precautionary allergen labelling)
PAL_INDICATORS = ("có thể chứa", "may contain", "vết của", "sản xuất trên", "chế biến trên",
                  "dây chuyền", "cùng cơ sở", "traces of", "produced in a facility", "manufactured on")

_CROSS_CONTACT_Q = re.compile(r"nhiễm chéo|tiếp xúc chéo|dây chuyền|cross.?contact|có thể chứa|vết của|dạng vết|may contain|traces?",
                              re.I)

# questions that ask for a medical decision, not a label screening
_OUT_OF_SCOPE_RULES = (
    (r"chẩn đoán|có phải (tôi )?(bị|mắc)|tôi có (bị|mắc) bệnh|triệu chứng của tôi|xét nghiệm",
     "câu hỏi yêu cầu chẩn đoán bệnh hoặc diễn giải triệu chứng/xét nghiệm cá nhân"),
    (r"điều trị|chữa (khỏi|bệnh|được)|trị (bệnh|dứt)|thay (thế )?thuốc|thay cho thuốc|phác đồ",
     "câu hỏi yêu cầu điều trị hoặc thay thế điều trị bằng thực phẩm"),
    (r"(uống|dùng|ngưng|ngừng|bỏ|giảm liều|tăng liều|đổi)\W+(\w+\W+){0,3}thuốc|tương tác (với )?thuốc|đơn thuốc|kê đơn|liều (dùng )?thuốc",
     "câu hỏi liên quan đến thuốc, liều thuốc hoặc tương tác thuốc"),
    (r"(mang thai|có thai|bầu|cho con bú)\W+(\w+\W+){0,6}(sảy|dị tật|thai nhi|an toàn cho thai)",
     "câu hỏi về nguy cơ thai kỳ cần đánh giá y khoa cá nhân"),
    (r"(khẩu phần|thực đơn|chế độ ăn)\W+(\w+\W+){0,4}(cho tôi|của tôi)|lập (thực đơn|chế độ ăn)",
     "câu hỏi yêu cầu xây dựng chế độ ăn/khẩu phần cá nhân hóa"),
)
_OUT_OF_SCOPE = tuple((re.compile(p, re.I), why) for p, why in _OUT_OF_SCOPE_RULES)

_QUANTITY = re.compile(r"\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|µg|ml|%|ppm)\b", re.I)
_GENERIC_ADDITIVE = re.compile(r"hương liệu|chất tạo (ngọt|màu|hương)|chất (điều chỉnh độ acid|điều vị|bảo quản|ổn định|nhũ hóa)|phụ gia thực phẩm",
                               re.I)
_ADDITIVE_Q = re.compile(r"phụ gia|\bins\s?\d|\be\d{3}\b|chất bảo quản|chất tạo|hương liệu", re.I)


# Nhóm người dùng mà cơ sở tri thức KHÔNG có luật/bằng chứng riêng: hệ thống vẫn trả lời theo luật cho người lớn
# (phụ gia, dị nguyên, bệnh nền đã khai) và chỉ thêm DÒNG LƯU Ý — không từ chối, không đổi kết luận.
# Trẻ dưới 12 tháng có luật riêng (modules/infant_rules.py), dòng lưu ý nêu phạm vi của luật đó.
# Người ăn được nói bằng lời mà không nêu số tuổi: "người cao tuổi", "ông bà", "bà nội tôi", "cụ nhà tôi" (không khớp "bà bầu")
_ELDERLY_Q = re.compile(r"người\s*(già|cao tuổi|lớn tuổi)|cao tuổi|\bông bà\b|\b(ông|bà|cụ)\s+(nội|ngoại)\b|\b(ông|bà|cụ)\s+(của\s+)?(tôi|mình|nhà)\b")
_LACTATING_Q = re.compile(r"cho con bú|nuôi con bằng sữa mẹ")
_QUOTED_NAME = re.compile(r"""(['"“‘]).*?(['"”’])""", re.S)
ELDERLY_AGE = 60


def _fold(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or "")).casefold()


def _mention_texts(linked_results, all_mentions) -> list[str]:
    texts, seen = [], set()
    for item in list(linked_results or []):
        t = str(item.get("raw_mention", "") or "")
        if t and t.lower() not in seen:
            texts.append(t); seen.add(t.lower())
    for m in list(all_mentions or []):
        t = str(m.get("raw_mention", m.get("text", "")) or "")
        if t and t.lower() not in seen:
            texts.append(t); seen.add(t.lower())
    return texts


def detect_out_of_scope(question: str) -> list[str]:
    """Reasons why the question is outside a label-screening system (empty list = in scope)."""
    q = _fold(question)
    return [why for rx, why in _OUT_OF_SCOPE if rx.search(q)]


def label_declares_cross_contact(linked_results, all_mentions, allergen_terms=None) -> bool:
    """True if the label has a cross-contact statement. With `allergen_terms` (the allergens the user asks about),
    the statement must name one of them: a warning about egg does not answer a question about sesame."""
    terms = [str(x).lower() for x in (allergen_terms or []) if str(x).strip()]
    for t in _mention_texts(linked_results, all_mentions):
        tl = t.lower()
        if any(p in tl for p in PAL_INDICATORS):
            if not terms or any(re.search(r"(?<!\w)" + re.escape(x) + r"(?!\w)", tl) for x in terms):
                return True
    if terms:
        return False
    return any(item.get("constraint_type") == "PRECAUTIONARY_RISK" for item in (linked_results or []))


def detect_data_gaps(question: str, linked_results, all_mentions=None, allergen_terms=None) -> list[dict]:
    """Missing inputs the question needs. Each gap: {code, reason, need}."""
    q = _fold(question)
    gaps: list[dict] = []
    texts = _mention_texts(linked_results, all_mentions)

    if _CROSS_CONTACT_Q.search(q) and not label_declares_cross_contact(linked_results, all_mentions, allergen_terms):
        gaps.append({
            "code": "CROSS_CONTACT_STATEMENT",
            "reason": ("Nhãn không có câu cảnh báo nguy cơ nhiễm chéo (PAL, ví dụ 'có thể chứa...') nào nêu dị nguyên bạn hỏi" if allergen_terms else
                       "Nhãn không có câu cảnh báo nguy cơ nhiễm chéo (PAL, ví dụ 'có thể chứa...')")
                      + ", nên không xác định được nguy cơ nhiễm chéo từ nhãn.",
            "need": "chụp lại phần cảnh báo dị nguyên/'có thể chứa' (thường ở mặt sau hoặc cạnh bảng thành phần) hoặc hỏi nhà sản xuất về dây chuyền sản xuất",
        })

    if _ADDITIVE_Q.search(q):
        def _is_additive(item):
            ent = item.get("matched_entity") or {}
            ins = str(ent.get("ins_code", "") or "")
            return bool(ins and not ins.upper().startswith("FOODON")) or bool(
                re.search(r"\bins\s?\d|\be\d{3}\b|\(\d{3,4}[a-z]?(?:\([ivx]+\))?\)", str(item.get("raw_mention", "")), re.I))
        if not any(_is_additive(i) for i in (linked_results or [])):
            generic = any(_GENERIC_ADDITIVE.search(t) for t in texts)
            gaps.append({
                "code": "ADDITIVE_IDENTITY",
                "reason": ("Nhãn chỉ ghi nhóm phụ gia/hương liệu chung, không nêu tên cụ thể hoặc mã INS" if generic else
                           "Nhãn không khai báo phụ gia nào có tên hoặc mã INS xác định (các tên ghi có thể chỉ là nguyên liệu thông thường)")
                          + ", nên không xác định được chất cần đánh giá.",
                "need": "chụp lại phần thành phần để thấy tên hoặc mã INS cụ thể của từng phụ gia, hoặc cung cấp thông tin từ nhà sản xuất",
            })
    return gaps


_CONDITION_NAMES_VI = {
    "gout": "bệnh gút", "hypertension": "tăng huyết áp", "diabetes": "đái tháo đường", "pregnancy": "thai kỳ",
    "obesity": "béo phì", "cardiovascular_risk": "nguy cơ tim mạch",
}
# Điều gì hệ thống VẪN xét cho bệnh đó dù chưa có lưu ý riêng về phụ gia (chỉ khi nhãn ghi số liệu định lượng)
_CONDITION_NUTRIENT_SCOPE = {
    "diabetes": "Hệ thống chỉ so lượng đường với điều kiện 'không đường' của Codex khi nhãn ghi số liệu đường.",
    "hypertension": "Hệ thống chỉ so lượng natri với điều kiện 'natri thấp' của Codex khi nhãn ghi số liệu natri.",
    "cardiovascular_risk": "Hệ thống chỉ so lượng natri với điều kiện 'natri thấp' của Codex khi nhãn ghi số liệu natri.",
}
# Không cần lưu ý này: không phải bệnh nền (khỏe mạnh), nhóm tuổi (có luật/lưu ý riêng), hoặc dị ứng/không dung nạp
# (xử lý bằng bộ luật dị nguyên, không bằng lưu ý phụ gia)
_NO_CONDITION_NOTE = re.compile(r"healthy|child|infant|adhd|elder|lactat|allerg|intoleran|celiac|gluten|^$")
# Cách ghi khác nhau của cùng một bệnh (hồ sơ benchmark, NHANES, chat) → khoá dùng trong KB
_CONDITION_CANON = ((re.compile(r"kidney|^ckd"), "ckd"), (re.compile(r"phenylketon|^pku"), "pku"), (re.compile(r"asthma"), "asthma"))


def _canon_condition(key: str) -> str:
    for rx, canon in _CONDITION_CANON:
        if rx.search(key):
            return canon
    return key


def _has_identified_additive(linked_results) -> bool:
    for item in linked_results or []:
        ins = str((item.get("matched_entity") or {}).get("ins_code", "") or "")
        if ins and not ins.upper().startswith("FOODON"):
            return True
    return False


def condition_notes(persona: dict | None, linked_results, covered_conditions) -> list[dict]:
    """Bệnh nền người dùng khai mà KB CHƯA có lưu ý phụ gia nào đã kiểm nguồn cho bệnh đó.

    Chỉ là dòng lưu ý (không từ chối, không đổi nhãn): phần ADI và quy định chung vẫn được tra cứu và giải thích; hệ thống
    nói rõ chưa có căn cứ riêng cho bệnh này thay vì để nhãn SAFE bị hiểu là 'ổn cho người bệnh'."""
    if not _has_identified_additive(linked_results):
        return []
    covered = {str(c).lower() for c in (covered_conditions or [])}
    notes, seen = [], set()
    for cond in (persona or {}).get("conditions", []):
        key = _canon_condition(str(cond).strip().lower().replace(" ", "_"))
        if key in seen or key in covered or _NO_CONDITION_NOTE.search(key):
            continue
        seen.add(key)
        name = _CONDITION_NAMES_VI.get(key, key.replace("_", " "))
        text = (f"CHƯA CÓ DỮ LIỆU cho {name}: hệ thống không có thông tin nào về ảnh hưởng của các phụ gia trên nhãn đối với người "
                f"{name}. Việc không có cảnh báo KHÔNG có nghĩa là sản phẩm an toàn hoặc không cần tránh đối với bạn — chỉ là hệ thống "
                "không có căn cứ để cảnh báo. Hãy hỏi bác sĩ hoặc chuyên gia dinh dưỡng trước khi dùng.")
        if key in _CONDITION_NUTRIENT_SCOPE:
            text += " " + _CONDITION_NUTRIENT_SCOPE[key]
        notes.append({"code": "CONDITION_NO_ADDITIVE_EVIDENCE", "group": name, "text": text})
    return notes


def population_notes(question: str, persona: dict | None = None, linked_results=None, covered_conditions=None) -> list[dict]:
    """Dòng lưu ý theo nhóm người dùng (câu hỏi hoặc hồ sơ). Không phải khoảng trống dữ liệu: kết luận giữ nguyên."""
    from modules import infant_rules  # tránh vòng import khi nạp module

    persona = persona or {}
    demo = persona.get("demographics") or {}
    conds = {str(c).lower() for c in persona.get("conditions", [])}
    q = _QUOTED_NAME.sub(" ", _fold(question))
    notes = []
    if infant_rules.is_infant(persona, question):
        notes.append({"code": "INFANT_SCOPE", "group": "trẻ dưới 12 tháng tuổi", "text": infant_rules.INFANT_SCOPE_NOTE})
        return notes  # các nhóm người lớn bên dưới không áp dụng cho trẻ
    try:
        age = int(demo.get("age")) if demo.get("age") is not None else None
    except (TypeError, ValueError):
        age = None
    if _ELDERLY_Q.search(q) or (age is not None and age >= ELDERLY_AGE):
        notes.append({"code": "ELDERLY_NO_SPECIFIC_EVIDENCE", "group": "người cao tuổi",
                      "text": "Cơ sở tri thức chưa có quy định riêng cho người cao tuổi; kết luận dựa trên quy định cho "
                              "người lớn và bệnh nền bạn đã khai — nên hỏi thêm bác sĩ nếu có nhiều bệnh hoặc đang dùng thuốc."})
    if _LACTATING_Q.search(q) or "lactating" in conds or demo.get("lactating"):
        notes.append({"code": "LACTATION_NO_SPECIFIC_EVIDENCE", "group": "phụ nữ cho con bú",
                      "text": "Cơ sở tri thức chưa có quy định riêng cho phụ nữ cho con bú; kết luận dựa trên quy định cho "
                              "người lớn — nên hỏi thêm bác sĩ hoặc chuyên gia dinh dưỡng."})
    if covered_conditions is not None:
        notes.extend(condition_notes(persona, linked_results, covered_conditions))
    return notes


def describe(types: list[str], data_gaps: list[dict], scope_reasons: list[str], extra_reasons: list[str] | None = None) -> dict:
    """Title, message and referral text for the given abstention types."""
    reasons_data = [g["reason"] for g in data_gaps] + list(extra_reasons or [])
    needs = [g["need"] for g in data_gaps]
    parts = []
    if INSUFFICIENT_DATA in types:
        parts.append(
            "THIẾU DỮ LIỆU: " + " ".join(reasons_data)
            + " Đề nghị chụp lại nhãn rõ hơn" + (f", cụ thể: {'; '.join(needs)}" if needs else "")
            + ", hoặc bổ sung thông tin còn thiếu (hàm lượng/khẩu phần, thông tin dị nguyên, bảng dinh dưỡng).")
    if OUT_OF_SCOPE in types:
        parts.append(
            "VƯỢT PHẠM VI HỆ THỐNG: " + "; ".join(scope_reasons).capitalize() + ". Hệ thống chỉ sàng lọc nhãn thực phẩm và không thay thế tư vấn y khoa. "
            "Đề nghị trao đổi với bác sĩ, dược sĩ hoặc chuyên gia dinh dưỡng phù hợp.")
    if set(types) == {INSUFFICIENT_DATA}:
        title = "⚪ TỪ CHỐI TƯ VẤN — THIẾU DỮ LIỆU"
    elif set(types) == {OUT_OF_SCOPE}:
        title = "⚪ TỪ CHỐI TƯ VẤN — VƯỢT PHẠM VI, HỎI CHUYÊN GIA"
    else:
        title = "⚪ TỪ CHỐI TƯ VẤN — THIẾU DỮ LIỆU VÀ VƯỢT PHẠM VI"
    return {"title": title, "message": " ".join(parts)}
