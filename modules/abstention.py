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
                re.search(r"ins\s?\d|e\d{3}|\(\d{3,4}[a-z]?(?:\([ivx]+\))?\)", str(item.get("raw_mention", "")), re.I))
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
