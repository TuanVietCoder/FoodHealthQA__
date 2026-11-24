"""Tầng QA động: câu hỏi của người dùng quyết định hệ thống tra cứu gì trong cơ sở tri thức (KB).

Luồng (dùng chung cho Config B và C; Config A là baseline không truy xuất nên không dùng):

    1. understand_question()     hiểu câu hỏi: ý định, phụ gia/dị nguyên/chất dinh dưỡng được nhắc tới
    2. lookup_question_additives()  tra KB theo các phụ gia mà CÂU HỎI nhắc tới (kể cả khi không có trên nhãn)
    3. build_trace()             ghi lại hệ thống đã tra nguồn nào, tìm thấy hay không
    4. render_context()          đoạn bằng chứng đưa cho LLM, nói rõ mục nào "không tìm thấy trong CSDL"

Nguyên tắc:
  * Chỉ dùng KB (phụ gia, dị nguyên, sự cố, dinh dưỡng, persona). KHÔNG đọc bộ QA benchmark / đáp án chuẩn.
  * Hàm thuần, xác định (không gọi LLM, không gọi mạng) nên B và C cho cùng kết quả khi cùng văn bản nhãn.
  * Khoảng trống tri thức "cứng" (hard gap) chỉ xuất hiện khi câu hỏi nêu đích danh một phụ gia KHÔNG có trên nhãn
    và KB không có dữ liệu quy chuẩn đã kiểm chứng cho phụ gia đó; khi đó guardrail từ chối thay vì để LLM suy đoán.
"""
from __future__ import annotations

import re
import unicodedata

from modules import abstention
from modules.additive_linker import (
    _EXPLICIT_CODE_RE,
    link_additive_mentions,
    normalize_ins_key,
)
from modules.constraint_model import ALLERGEN_KEYWORDS, contains_label_term

# (mã ý định, mô tả tiếng Việt, mẫu nhận diện)
INTENT_RULES = (
    ("additive_safety", "An toàn phụ gia",
     r"phụ gia|\bins\s?\d|\be\s?\d{3}|chất bảo quản|chất tạo|chất nhũ|chất ổn định|chất chống|hương liệu|phẩm màu|chất điều"),
    ("intake_limit", "Ngưỡng/liều dùng (ADI, mức tối đa)",
     r"\badi\b|liều|bao nhiêu|ngưỡng|tối đa|vượt|hàng ngày|mỗi ngày|thường xuyên|tích lũy|khẩu phần"),
    ("allergen", "Dị nguyên",
     r"dị ứng|dị nguyên|allerg|shock phản vệ|sốc phản vệ"),
    ("cross_contact", "Nhiễm chéo dị nguyên",
     r"nhiễm chéo|tiếp xúc chéo|có thể chứa|dạng vết|dây chuyền|cross.?contact|traces?"),
    ("nutrition", "Dinh dưỡng",
     r"dinh dưỡng|đường|muối|natri|sodium|calo|kcal|năng lượng|chất béo|béo|đạm|protein|cholesterol|tiểu đường|huyết áp"),
    ("regulation", "Quy định pháp lý",
     r"quy định|quy chuẩn|thông tư|\btt\s?24\b|codex|được phép|cho phép|hợp pháp|tiêu chuẩn"),
    ("hazard_history", "Sự cố / thu hồi",
     r"thu hồi|sự cố|ngộ độc|nhiễm khuẩn|salmonella|listeria|e\.? ?coli|độc tố|kim loại nặng"),
)
_INTENTS = tuple((c, d, re.compile(p, re.I)) for c, d, p in INTENT_RULES)

NUTRIENT_RULES = (
    ("sugar", "đường", r"đường|sugar|ngọt"),
    ("sodium", "natri/muối", r"muối|natri|sodium|salt|mặn"),
    ("fat", "chất béo", r"béo|fat|dầu mỡ"),
    ("energy", "năng lượng", r"calo|kcal|năng lượng|calorie"),
    ("protein", "đạm", r"đạm|protein"),
)
_NUTRIENTS = tuple((c, d, re.compile(p, re.I)) for c, d, p in NUTRIENT_RULES)

POPULATION_RULES = (
    ("child", "trẻ em", r"trẻ|bé\b|em bé|con tôi|nhi\b"),
    ("pregnancy", "thai kỳ / cho con bú", r"mang thai|có thai|bà bầu|cho con bú"),
    ("elderly", "người cao tuổi", r"người (già|cao tuổi)|cao tuổi"),
)
_POPULATIONS = tuple((c, d, re.compile(p, re.I)) for c, d, p in POPULATION_RULES)

_QUOTED = re.compile(r"""(['"“‘]).*?\1""", re.S)  # tên sản phẩm trong ngoặc kép/đơn không phải thực thể được hỏi

SOURCE_ADDITIVE = "Cơ sở tri thức phụ gia (TT24 · Codex · JECFA · EFSA)"
SOURCE_ALLERGEN = "Nhóm dị nguyên (FALCPA · WHO/JEMRA)"
SOURCE_NUTRITION = "Dinh dưỡng (nhãn · bảng TPTP · USDA FNDDS)"
SOURCE_HAZARD = "Sự cố thực phẩm (Food Hazard Detector)"


def _fold(text) -> str:
    return unicodedata.normalize("NFC", str(text or "")).casefold()


def strip_product_name(question: str) -> str:
    """Bỏ tên sản phẩm trong dấu nháy khỏi câu hỏi (tên sản phẩm có thể chứa từ như 'sữa', 'đường')."""
    return _QUOTED.sub(" ", str(question or ""))


_SWEETENER_CLASS = re.compile(r"chất\s+(?:tạo|làm)\s+ngọt(?:\s+(?:nhân tạo|tổng hợp|tự nhiên))?")


def _without_additive_names(folded_question: str, mentions) -> str:
    """Câu hỏi (đã _fold) sau khi bỏ các tên phụ gia đã nhận diện, để từ như 'natri' trong 'natri benzoat' không bị hiểu là
    chất dinh dưỡng. Chỉ bỏ đúng chuỗi alias mà bộ liên kết đã khớp."""
    text = _SWEETENER_CLASS.sub(" ", folded_question)  # "chất tạo ngọt" là nhóm phụ gia, không phải hỏi về đường
    aliases =sorted({_fold(m.get("matched_alias", "")).strip() for m in mentions or [] if m.get("matched_alias")}, key=len, reverse=True)
    for alias in aliases:
        if alias:
            text = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", " ", text)
    return text


# ---------------------------------------------------------------------------
# 1. Hiểu câu hỏi
# ---------------------------------------------------------------------------
def understand_question(question: str, persona: dict | None = None) -> dict:
    """Trả về kế hoạch: ý định, thực thể được hỏi, đối tượng, và cờ ngoài phạm vi."""
    q = _fold(strip_product_name(question))
    mentions = link_additive_mentions(strip_product_name(question))
    # Tên phụ gia chứa từ trùng chất dinh dưỡng ("natri" benzoat, "natri" nitrit, mononatri glutamat) không phải câu hỏi về
    # dinh dưỡng: bỏ tên phụ gia khỏi câu trước khi nhận diện ý định dinh dưỡng và chất dinh dưỡng được hỏi.
    q_nutrition = _without_additive_names(q, mentions)
    intents = [{"code": c, "label": d} for c, d, rx in _INTENTS if rx.search(q_nutrition if c == "nutrition" else q)]
    scope = abstention.detect_out_of_scope(question)
    if not intents and not scope:
        intents = [{"code": "general_safety", "label": "Đánh giá an toàn tổng quát theo nhãn"}]

    additives = []
    for item in mentions:
        ent = item.get("matched_entity") or {}
        additives.append({
            "ins_code": ent.get("ins_code", ""),
            "name": ent.get("canonical_name_vi") or ent.get("canonical_name") or ent.get("canonical_name_en", ""),
            "mention": item.get("matched_alias", ""),
        })
    # mã INS/E mà người dùng gõ nhưng registry không có (vẫn là thực thể được hỏi)
    known = {normalize_ins_key(a["ins_code"]) for a in additives}
    unknown_codes = []
    stripped = strip_product_name(question)
    for m in _EXPLICIT_CODE_RE.finditer(stripped):
        tail = re.match(r"\s*\(\s*[ivx]+\s*\)", stripped[m.end():], re.I)
        key = normalize_ins_key(m.group("code") + (tail.group(0) if tail else ""))
        if key and key not in known and not any(key == normalize_ins_key(a["ins_code"]) for a in additives):
            code_text = f"{m.group('prefix').upper()} {m.group('code').strip()}{tail.group(0).strip() if tail else ''}"
            if code_text not in unknown_codes and not _registry_has_key(key):
                unknown_codes.append(code_text)

    allergens = [
        a for a, kws in ALLERGEN_KEYWORDS.items()
        if any(contains_label_term(q, kw) for kw in kws)
    ]
    nutrients = [{"code": c, "label": d} for c, d, rx in _NUTRIENTS if rx.search(q_nutrition)]
    populations = [{"code": c, "label": d} for c, d, rx in _POPULATIONS if rx.search(q)]

    return {
        "question": str(question or ""),
        "intents": intents,
        "additives": additives,
        "unknown_additive_codes": unknown_codes,
        "allergens": allergens,
        "nutrients": nutrients,
        "populations": populations,
        "out_of_scope": scope,
        "persona_name": (persona or {}).get("name", ""),
    }


def _registry_has_key(key: str) -> bool:
    from modules.additive_linker import _build_indexes, DEFAULT_REGISTRY_PATH

    code_index, e_index, _alias, _aliases = _build_indexes(str(DEFAULT_REGISTRY_PATH.resolve()))
    if code_index.get(key) or e_index.get(key):
        return True
    # mã cha của mã con: 160a -> 160a(i), 100 -> 100(i)
    return any(k.startswith(f"{key}(") for k in list(code_index) + list(e_index))


# ---------------------------------------------------------------------------
# 2. Tra KB theo phụ gia mà câu hỏi nhắc tới
# ---------------------------------------------------------------------------
def _kb_entry(ins_code: str, kb_additives) -> dict | None:
    key = normalize_ins_key(ins_code)
    for entry in kb_additives or []:
        if normalize_ins_key(entry.get("ins_code", "")) == key:
            return entry
    return None


def _label_keys(linked_results) -> set[str]:
    keys = set()
    for item in linked_results or []:
        ent = item.get("matched_entity") or {}
        if ent.get("ins_code"):
            keys.add(normalize_ins_key(ent["ins_code"]))
    return keys


def _adi_text(entry: dict) -> str:
    return str(entry.get("adi_value") or entry.get("jecfa_adi") or "").strip()


def lookup_question_additives(plan: dict, kb_additives, linked_results) -> list[dict]:
    """Với mỗi phụ gia được hỏi: có trên nhãn không, KB có dữ liệu quy chuẩn không."""
    on_label = _label_keys(linked_results)
    rows = []
    for add in plan.get("additives", []):
        key = normalize_ins_key(add["ins_code"])
        entry = _kb_entry(add["ins_code"], kb_additives)
        has_data = bool(entry) and not entry.get("identity_only")
        rows.append({
            "ins_code": add["ins_code"],
            "name": add["name"] or (entry or {}).get("canonical_name_vi", ""),
            "mention": add["mention"],
            "on_label": key in on_label,
            "identity_known": True,
            "kb_entry": entry if has_data else None,
            "adi": _adi_text(entry) if has_data else "",
        })
    label_flat = " ".join(
        re.sub(r"[\s()]", "", _fold(item.get("raw_mention", ""))) for item in (linked_results or [])
    )
    for code in plan.get("unknown_additive_codes", []):
        flat = re.sub(r"[\s()]", "", _fold(code)).replace("ins", "").replace("e", "", 1) if code.upper().startswith("E ")             else re.sub(r"[\s()]", "", _fold(code)).replace("ins", "")
        rows.append({
            "ins_code": code, "name": "", "mention": code, "on_label": bool(flat) and flat in label_flat,
            "identity_known": False, "kb_entry": None, "adi": "",
        })
    return rows


def knowledge_gaps(question_additives: list[dict]) -> list[dict]:
    """Khoảng trống tri thức cứng: mã phụ gia được hỏi không có trong KB và không có trên nhãn.

    Phụ gia chỉ có định danh (chưa có ADI/ML) KHÔNG tạo gap: chỉ ghi "không tìm thấy" vào vết tra cứu.

    Trả về danh sách có cùng dạng với abstention.detect_data_gaps: {code, reason, need}.
    Phụ gia có trên nhãn do guardrail hiện có xử lý, không tạo gap ở đây.
    """
    gaps = []
    for row in question_additives:
        # Chỉ từ chối cứng khi mã người dùng gõ hoàn toàn không có trong cơ sở tri thức và không có trên nhãn.
        # Phụ gia chỉ có định danh (chưa có ADI/ML) chỉ được ghi vào vết tra cứu và báo cho LLM: câu hỏi thường nêu
        # chúng như ví dụ của một nhóm chất, từ chối vì chúng là quá tay.
        if row["on_label"] or row["kb_entry"] or row["identity_known"]:
            continue
        label = row["ins_code"]
        gaps.append({
            "code": "ADDITIVE_NOT_IN_KB",
            "reason": f"Cơ sở tri thức chưa có mã phụ gia {label} mà câu hỏi nhắc tới, và phụ gia này không có trên nhãn đang xét, nên không thể kết luận.",
            "need": f"cung cấp nguồn quy chuẩn cho {label} hoặc hỏi chuyên gia/nhà sản xuất",
        })
    return gaps


# ---------------------------------------------------------------------------
# 3. Truy vết: hệ thống đã tra gì, thấy gì
# ---------------------------------------------------------------------------
def build_trace(plan: dict, question_additives: list[dict], linked_results, evaluation: dict,
                detected_hazards, matched_fndds, nutri_warnings) -> list[dict]:
    codes = {i["code"] for i in plan["intents"]}
    trace: list[dict] = []

    def add(source, query, status, detail, origin):
        trace.append({"source": source, "query": query, "status": status, "detail": detail, "origin": origin})

    # a) phụ gia được hỏi
    for row in question_additives:
        q = row["ins_code"] + (f" ({row['name']})" if row["name"] else "")
        where = "Có trên nhãn" if row["on_label"] else "Không có trên nhãn"
        if row["kb_entry"]:
            adi = row["adi"] or "chưa có giá trị ADI trong KB"
            add(SOURCE_ADDITIVE, q, "found", f"{where} — Quy chuẩn: {adi}", "question")
        elif row["identity_known"]:
            add(SOURCE_ADDITIVE, q, "not_found", f"{where} — Đã định danh hợp lệ (phụ gia thông dụng, không có cảnh báo nguy cơ độc tính)", "question")
        else:
            add(SOURCE_ADDITIVE, q, "not_found", f"{where} — Mã phụ gia chưa có trong CSDL tri thức (cần tra cứu thêm từ nhà sản xuất)", "question")

    # b) phụ gia trên nhãn (khi câu hỏi về phụ gia / ngưỡng / quy định)
    if codes & {"additive_safety", "intake_limit", "regulation", "general_safety"}:
        asked = {normalize_ins_key(r["ins_code"]) for r in question_additives}
        shown = 0
        for item in linked_results or []:
            ent = item.get("matched_entity") or {}
            ins = ent.get("ins_code", "")
            if not ins or str(ins).upper().startswith("FOODON") or normalize_ins_key(ins) in asked:
                continue
            if shown >= 8:
                break
            shown += 1
            adi = _adi_text(ent)
            name = ent.get("canonical_name_vi") or ent.get("canonical_name") or ""
            if ent.get("identity_only"):
                add(SOURCE_ADDITIVE, f"{ins} {name}".strip(), "not_found",
                    "Có trên nhãn: Đã định danh hợp lệ (phụ gia thông dụng, không có cảnh báo nguy cơ ADI/ML đặc biệt)", "label")
            else:
                add(SOURCE_ADDITIVE, f"{ins} {name}".strip(), "found",
                    "Có trên nhãn: Quy chuẩn " + (adi or "đã đối chiếu, chưa ghi nhận giới hạn ADI riêng biệt"), "label")

    # c) dị nguyên
    if codes & {"allergen", "cross_contact"} or plan["allergens"]:
        stated = plan["allergens"]
        alerts = [a for a in evaluation.get("alerts", []) if a.get("regulation") == "US_FDA_FALCPA"]
        names = ", ".join(sorted({a.get("additive", "") for a in alerts if a.get("additive")}))
        query = ", ".join(stated) if stated else "dị nguyên theo hồ sơ người dùng"
        if alerts:
            add(SOURCE_ALLERGEN, query, "found", f"Cảnh báo dị nguyên: Phát hiện trên nhãn thành phần dị ứng ({names or 'có cảnh báo'})", "question")
        else:
            add(SOURCE_ALLERGEN, query, "found", "Đã đối chiếu nhãn với CSDL dị nguyên: Không phát hiện dị nguyên trùng khớp", "question")
        for gap in evaluation.get("data_gaps", []):
            if gap.get("code") == "CROSS_CONTACT_STATEMENT":
                add(SOURCE_ALLERGEN, "câu cảnh báo nhiễm chéo (PAL) trên nhãn", "not_found", gap.get("reason", ""), "label")

    # d) dinh dưỡng
    if "nutrition" in codes or plan["nutrients"]:
        query = ", ".join(n["label"] for n in plan["nutrients"]) or "chỉ số dinh dưỡng"
        if matched_fndds:
            src = {"product_label": "nhãn sản phẩm", "vcndd_reference": "bảng TPTP Việt Nam (ước tính)"}.get(
                matched_fndds.get("data_source_type"), "USDA FNDDS (ước tính)")
            extra = f"; {len(nutri_warnings)} cảnh báo vượt ngưỡng" if nutri_warnings else ""
            add(SOURCE_NUTRITION, query, "found", f"Dinh dưỡng: {matched_fndds.get('food_description', '')} (Nguồn: {src}){extra}", "question")
        else:
            add(SOURCE_NUTRITION, query, "not_found", "Chưa khớp được dữ liệu dinh dưỡng cho sản phẩm này", "question")

    # e) sự cố thực phẩm
    if "hazard_history" in codes:
        if detected_hazards:
            add(SOURCE_HAZARD, "sản phẩm / thành phần", "found", f"Phát hiện {len(detected_hazards)} sự cố cảnh báo thu hồi liên quan", "question")
        else:
            add(SOURCE_HAZARD, "sản phẩm / thành phần", "not_found", "Không có sự cố trùng khớp trong CSDL cảnh báo thu hồi", "question")
    return trace


# ---------------------------------------------------------------------------
# 4. Đoạn bằng chứng cho LLM
# ---------------------------------------------------------------------------
def render_context(plan: dict, trace: list[dict]) -> str:
    """Khối văn bản nối vào bằng chứng của LLM: nêu rõ cái gì CSDL có / không có."""
    if not trace:
        return ""
    lines = ["### Tri thức đã tra cứu theo câu hỏi:"]
    lines.append("Ý định nhận diện: " + "; ".join(i["label"] for i in plan["intents"]) + ".")
    for t in trace:
        mark = "TÌM THẤY" if t["status"] == "found" else "KHÔNG TÌM THẤY"
        lines.append(f"- [{mark}] {t['source']} | truy vấn: {t['query']} | {t['detail']}")
    lines.append(
        "Chỉ nêu kết luận dựa trên các mục TÌM THẤY. Với mục KHÔNG TÌM THẤY, hãy nói rõ cơ sở dữ liệu chưa có thông tin, "
        "không dùng kiến thức bên ngoài để lấp chỗ trống.")
    return "\n".join(lines)


def extra_citations(question_additives: list[dict]) -> list[str]:
    """Mã nguồn bổ sung cho phụ gia được hỏi mà không có trên nhãn nhưng KB có dữ liệu."""
    cits: list[str] = []
    for row in question_additives:
        ent = row.get("kb_entry")
        if not ent or row["on_label"]:
            continue
        if ent.get("jecfa_adi") and "JECFA_EVALUATIONS" not in cits:
            cits.append("JECFA_EVALUATIONS")
        if "24/2019" in str(ent.get("vn_regulation", "")) and "VN_TT24_2019" not in cits:
            cits.append("VN_TT24_2019")
    return cits


def summarize(plan: dict, trace: list[dict], gaps: list[dict]) -> dict:
    """Bản tóm tắt gọn cho UI và file dự đoán."""
    found = sum(1 for t in trace if t["status"] == "found")
    return {
        "intents": [i["label"] for i in plan["intents"]],
        "entities": {
            "additives": [a["ins_code"] for a in plan["additives"]] + list(plan["unknown_additive_codes"]),
            "allergens": plan["allergens"],
            "nutrients": [n["label"] for n in plan["nutrients"]],
            "populations": [p["label"] for p in plan["populations"]],
        },
        "trace": trace,
        "sources_found": found,
        "sources_missing": len(trace) - found,
        "knowledge_gaps": gaps,
        "sufficient": not gaps and not plan["out_of_scope"],
    }
