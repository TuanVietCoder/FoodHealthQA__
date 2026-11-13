"""Lớp bằng chứng cho giao diện hỏi đáp: đánh số trích dẫn, danh sách nguồn, nhãn lĩnh vực, lý do thiếu bằng chứng.

Không gọi LLM và không đọc ảnh; chỉ biến kết quả đã có của pipeline (câu trả lời có thẻ [MÃ_NGUỒN],
các đoạn đã truy xuất, kết quả guardrail) thành dữ liệu để hiển thị:

    numbered = build_cited_answer(answer, passages, ins_codes)
    numbered["text"]     # câu trả lời, thẻ [MÃ_NGUỒN] đã đổi thành [1], [2]...
    numbered["sources"]  # [{n, citation_id, title, scope, locator, quote, url, tier, tier_label, verified}]
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from modules.benchmark import RECOGNIZED_CITATION_SOURCES, canonicalize_citation, is_recognized_citation

_ROOT = Path(__file__).resolve().parent.parent
_CATALOG_PATH = _ROOT / "knowledge_base" / "sources" / "catalog" / "source_citation_catalog.json"

_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_PREFIX = re.compile(r"^(?:nguồn|nguon|source|ref)\s*:\s*", re.I)
QUOTE_MAX_CHARS = 420
ABSTAIN_THRESHOLD = 0.65  # cùng ngưỡng với tầng guardrail

# Mức tin cậy của nguồn theo bản chất tài liệu (không phải độ chắc chắn của từng nhận định).
_TIER_RULES = (
    (("VN_TT24", "CODEX_", "US_FDA", "US_CFR", "US_CODE", "EU_REG"), 1,
     "Cao: văn bản pháp quy / tiêu chuẩn chính thức"),
    (("JECFA", "WHO_FAO", "EFSA", "FAO_WHO", "WHO_JEMRA", "IARC", "WHO_"), 2,
     "Cao: đánh giá khoa học của cơ quan chuyên môn quốc tế"),
    (("USDA", "OFF_", "OPEN_FOOD"), 3,
     "Trung bình: cơ sở dữ liệu tham khảo (giá trị ước tính)"),
)


def source_tier(citation_id: str) -> tuple[int, str]:
    cid = str(citation_id or "").upper()
    for prefixes, tier, label in _TIER_RULES:
        if cid.startswith(prefixes):
            return tier, label
    return 4, "Chưa phân loại"


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    try:
        return json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def resolve_source(citation_id: str, locator: str | None = None) -> dict:
    """Tên tài liệu, mục/điều khoản, URL cho một mã nguồn; ưu tiên khớp đúng locator của đoạn trích."""
    catalog = load_catalog()
    entry = catalog.get(locator or "") if locator else None
    if not entry:
        entry = next((v for v in catalog.values() if v.get("citation_id") == citation_id), None)
    meta = RECOGNIZED_CITATION_SOURCES.get(citation_id, {})
    return {
        "title": (entry or {}).get("title") or meta.get("title") or citation_id,
        "scope": (entry or {}).get("scope", ""),
        "url": (entry or {}).get("url", ""),
        "verified": bool((entry or {}).get("verified_against")),
    }


def _clip(text: str, limit: int = QUOTE_MAX_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _fallback_passage(citation_id: str, ins_codes) -> dict | None:
    """Khi không có đoạn nào được truy xuất cho mã nguồn này, lấy một đoạn trong kho khớp mã nguồn và mã INS."""
    try:
        from modules.passage_retriever import get_passage_retriever, normalize_ins
        retriever = get_passage_retriever()
    except Exception:
        return None
    wanted = {normalize_ins(c) for c in (ins_codes or []) if c}
    candidates = [p for p in retriever.passages if p.get("citation_id") == citation_id]
    for p in candidates:
        if wanted & set(p.get("ins_codes", [])):
            return p
    return candidates[0] if candidates else None


def build_cited_answer(answer: str | None, passages=None, ins_codes=None) -> dict:
    """Đổi thẻ [MÃ_NGUỒN] trong câu trả lời thành số [n] và dựng danh sách nguồn tương ứng.

    Thẻ không thuộc danh mục nguồn được công nhận được giữ nguyên (không tự gán số cho nguồn bịa).
    """
    passages = list(passages or [])
    order: list[str] = []

    def number_for(cid: str) -> int:
        if cid not in order:
            order.append(cid)
        return order.index(cid) + 1

    def repl(match: re.Match) -> str:
        parts = [p.strip() for p in re.split(r"[,;&]", _PREFIX.sub("", match.group(1).strip())) if p.strip()]
        if not parts or not all(is_recognized_citation(p) for p in parts):
            return match.group(0)
        nums = sorted({number_for(canonicalize_citation(p)) for p in parts})
        return "".join(f"[{n}]" for n in nums)

    text = _BRACKET.sub(repl, str(answer or ""))

    sources = []
    for cid in order:
        passage = next((p for p in passages if canonicalize_citation(p.get("citation_id", "")) == cid), None)
        if passage is None:
            passage = _fallback_passage(cid, ins_codes)
        locator = (passage or {}).get("locator")
        info = resolve_source(cid, locator)
        tier, tier_label = source_tier(cid)
        quote_text = (passage or {}).get("text", "")
        if not quote_text and info.get("scope"):
            quote_text = f"Quy chuẩn / Phạm vi thẩm định: {info['scope']}"
        sources.append({
            "n": order.index(cid) + 1,
            "citation_id": cid,
            "title": info["title"],
            "scope": info["scope"],
            "locator": locator or "",
            "quote": _clip(quote_text),
            "url": info["url"],
            "tier": tier,
            "tier_label": tier_label,
            "verified": info["verified"],
        })
    return {"text": text, "sources": sources}


# ── Nhãn lĩnh vực: cùng bộ nhãn với dataset (domain / domain_vi) ──────────────────────────────

DOMAIN_LABELS = {
    "additive_regulation": "Đánh giá phụ gia",
    "allergen_detection": "Phát hiện dị nguyên",
    "vulnerable_groups": "Nhóm dễ tổn thương",
    "insufficient_data": "Thiếu dữ liệu",
}


def classify_domain(qa: dict | None, evaluation: dict | None, persona: dict | None) -> tuple[str, str]:
    """Gán một trong 4 nhãn lĩnh vực của dataset cho câu hỏi hiện tại.

    Thứ tự ưu tiên theo cách dataset được dựng: thiếu dữ liệu > dị nguyên > nhóm dễ tổn thương > phụ gia.
    """
    qa = qa or {}
    evaluation = evaluation or {}
    persona = persona or {}
    ents = qa.get("entities", {})
    intents = " ".join(qa.get("intents", [])).lower()

    if evaluation.get("status") == "ABSTAIN" or qa.get("knowledge_gaps"):
        code = "insufficient_data"
    elif ents.get("allergens") or "dị nguyên" in intents or persona.get("allergies"):
        code = "allergen_detection"
    elif ents.get("populations") or persona.get("conditions"):
        code = "vulnerable_groups"
    else:
        code = "additive_regulation"
    return code, DOMAIN_LABELS[code]


# ── Lý do "không đủ bằng chứng" ────────────────────────────────────────────────────────────────

def insufficiency_reasons(evaluation: dict, ocr_result: dict | None, linked_results, passages, qa: dict | None) -> list[str]:
    """Các lý do cụ thể khiến câu trả lời chưa đủ căn cứ (rỗng nếu đủ)."""
    reasons: list[str] = []
    ocr_result = ocr_result or {}
    s_perception = float(evaluation.get("s_perception", 1.0) or 0.0)
    s_total = float(evaluation.get("s_total", 1.0) or 0.0)

    if not str(ocr_result.get("full_text", "")).strip():
        reasons.append("Không đọc được chữ nào trên ảnh: ảnh có thể quá mờ, tối hoặc không chứa bảng thành phần.")
    elif s_perception < ABSTAIN_THRESHOLD:
        reasons.append(f"Ảnh khó đọc: độ tin cậy OCR {s_perception:.2f} thấp hơn ngưỡng {ABSTAIN_THRESHOLD:.2f}.")

    if not any(item.get("matched_entity") for item in (linked_results or [])):
        reasons.append("Không nhận diện được phụ gia hoặc dị nguyên nào trong thành phần đọc được.")

    if not passages:
        reasons.append("Không truy xuất được đoạn bằng chứng nào từ cơ sở tri thức cho câu hỏi này.")

    for reason in evaluation.get("abstention_reasons", []) or []:
        if reason not in reasons:
            reasons.append(str(reason))
    for gap in (qa or {}).get("knowledge_gaps", []) or []:
        text = gap.get("reason") if isinstance(gap, dict) else str(gap)
        if text and text not in reasons:
            reasons.append(text)

    if s_total < ABSTAIN_THRESHOLD:
        reasons.append(f"Điểm tin cậy tổng thể {s_total:.2f} thấp hơn ngưỡng {ABSTAIN_THRESHOLD:.2f}.")
    return reasons
