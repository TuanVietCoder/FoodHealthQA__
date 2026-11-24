"""Luật phụ gia cho THỰC PHẨM DÀNH CHO TRẺ DƯỚI 12 THÁNG TUỔI (nhóm thực phẩm 13.1 và 13.2).

Căn cứ (trích tự động từ văn bản gốc bởi scripts_research/extract_infant_additive_rules.py →
knowledge_base/sources/derived/infant_food_additives.json):
  - TT 24/2019/TT-BYT (bản hợp nhất 09/VBHN-BYT 2024), Phụ lục 2A: phụ gia và mức tối đa cho nhóm 13.1.1/13.1.2/13.1.3/13.2.
    Phụ lục 3 mục 2 (nhóm thực phẩm được dùng phụ gia theo GMP) KHÔNG có nhóm 13.1, 13.2.
  - Codex GSFA CXS 192-1995: Bảng Hai nhóm 13.1.x/13.2; Annex to Table Three loại 13.1, 13.2 khỏi quy tắc GMP chung.
→ Với thực phẩm cho trẻ nhỏ, chỉ phụ gia được ghi riêng cho các nhóm này mới được dùng.

Luật: người dùng là trẻ dưới 12 tháng (hồ sơ hoặc câu hỏi) → mỗi phụ gia có mã INS trên nhãn được đối chiếu với hợp của
các nhóm 13.1.1, 13.1.2, 13.1.3, 13.2 theo TT24 và theo Codex. Phụ gia KHÔNG có trong danh mục nào → cảnh báo mức CAUTION.
Phụ gia có trong danh mục → ghi mức tối đa làm thông tin (nhãn không ghi hàm lượng nên không so được với ML).
Luật không đánh giá sản phẩm có phù hợp dinh dưỡng cho trẻ hay không (ví dụ sữa tươi, thực phẩm của người lớn).
"""
from __future__ import annotations

import functools
import json
import re
import unicodedata
from pathlib import Path

from modules.additive_linker import normalize_ins_key

DATA_PATH = Path(__file__).resolve().parents[1] / "knowledge_base" / "sources" / "derived" / "infant_food_additives.json"
CATS = ("13.1.1", "13.1.2", "13.1.3", "13.2")

# trẻ dưới 12 tháng trong câu hỏi (tên sản phẩm trong dấu nháy bị bỏ qua trước khi gọi)
_BABY_MONTHS = r"(?:\b(?:bé|con|cháu|trẻ)(?:\s+(?:nhà|tôi|mình|em|của|được|mới|chỉ|tầm|khoảng|gần|đã))*\s+)"
INFANT_QUESTION = re.compile(r"sơ sinh|nhũ nhi|dưới\s*(1\s*tuổi|12\s*tháng)|(?<![\d.,])(?:[0-9]|1[01])\s*tháng\s*tuổi|"
                             + _BABY_MONTHS + r"(?:[0-9]|1[01])\s*tháng\b")
_QUOTED_NAME = re.compile(r"""(['"“‘]).*?(['"”’])""", re.S)

INFANT_SCOPE_NOTE = ("Với trẻ dưới 12 tháng, hệ thống chỉ đối chiếu phụ gia trên nhãn với danh mục phụ gia được phép cho "
                     "thực phẩm trẻ nhỏ (TT 24/2019 và Codex); không đánh giá sản phẩm có phù hợp dinh dưỡng cho trẻ hay "
                     "không — nên hỏi bác sĩ nhi trước khi cho trẻ dùng.")


def _fold(text) -> str:
    return unicodedata.normalize("NFC", str(text or "")).casefold()


@functools.lru_cache(maxsize=1)
def _data() -> dict:
    if not DATA_PATH.is_file():
        return {}
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _index() -> dict:
    """{nguồn: {mã INS chuẩn hóa: [dòng…]}} cho 'vn' và 'codex'."""
    idx = {"vn": {}, "codex": {}}
    d = _data()
    for src in idx:
        for cat, rows in ((d.get(src) or {}).get("categories") or {}).items():
            for r in rows:
                for code in r.get("ins_codes", []):
                    idx[src].setdefault(normalize_ins_key(code), []).append({**r, "category": cat})
    return idx


def _lookup(src: str, key: str) -> list[dict]:
    """Khớp mã trên nhãn với danh mục; mã cha trên nhãn ("322") khớp mã con trong danh mục ("322(i)") và ngược lại."""
    idx = _index()[src]
    if key in idx:
        return idx[key]
    out = []
    for k, rows in idx.items():
        if k.startswith(key + "(") or re.fullmatch(re.escape(key) + r"[a-z]", k) or key.startswith(k + "("):
            out += rows
    return out


def available() -> bool:
    return bool(_index()["vn"] or _index()["codex"])


def is_infant(persona: dict | None, question: str = "") -> bool:
    persona = persona or {}
    demo = persona.get("demographics") or {}
    months = demo.get("age_months", persona.get("age_months"))
    if months is not None:
        try:
            if int(months) < 12:
                return True
        except (TypeError, ValueError):
            pass
    if "infant_under_12m" in [str(c).lower() for c in persona.get("conditions", [])]:
        return True
    q = _QUOTED_NAME.sub(" ", _fold(question))
    return bool(INFANT_QUESTION.search(q))


def infant_age_months(persona: dict | None, question: str = "") -> int | None:
    """Tuổi tháng của bé (< 12) từ hồ sơ hoặc câu hỏi; None nếu chỉ biết là dưới 12 tháng (vd 'sơ sinh', 'dưới 1 tuổi')."""
    persona = persona or {}
    demo = persona.get("demographics") or {}
    months = demo.get("age_months", persona.get("age_months"))
    try:
        if months is not None and int(months) < 12:
            return int(months)
    except (TypeError, ValueError):
        pass
    q = _QUOTED_NAME.sub(" ", _fold(question))
    m = re.search(r"(?<![\d.,])([0-9]|1[01])\s*tháng\s*tuổi", q) or re.search(_BABY_MONTHS + r"([0-9]|1[01])\s*tháng\b", q)
    return int(m.group(1)) if m else None


# --- Lỗi 2: muối khoáng / vitamin ghi là nguồn dinh dưỡng không phải phụ gia -------------------------------------
# QCVN 11-1:2012/BYT mục 1.3.4 và Codex CXS 72-1981 mục A3.4: dạng vitamin, chất khoáng bổ sung chọn theo danh mục hợp
# chất dinh dưỡng (CAC/GL 10-1979), không theo danh mục phụ gia. Chỉ bỏ qua khi nhãn ghi rõ vai trò dinh dưỡng và thành
# phần đó không nằm trong một nhóm phụ gia (vd "vitamin (D3, A). chất ổn định (440, 466)" vẫn được xét).
_NUTRIENT_ROLE = re.compile(r"nguồn\s*(bổ\s*sung\s*)?(khoáng|vitamin)|khoáng\s*chất|\bvitamin\b|tiền\s*vitamin|\(khoáng")
_ADDITIVE_ROLE = re.compile(r"chất\s*(ổn\s*định|nhũ\s*hóa|nhũ\s*hoá|chống\s*oxy|điều|bảo\s*quản|tạo|làm|xử\s*lý|giữ)|"
                            r"phẩm\s*màu|màu\s*(thực\s*phẩm|tự\s*nhiên|tổng\s*hợp)|\bins\b|\(\s*\d{3,4}")
NUTRIENT_SOURCE_NOTE = ("Ghi trên nhãn là nguồn khoáng/vitamin: xét theo danh mục hợp chất dinh dưỡng (QCVN 11-1:2012/BYT "
                        "mục 1.3.4; Codex CXS 72-1981 mục A3.4), không xét như phụ gia.")


def is_nutrient_source(raw_mention: str) -> bool:
    t = _fold(raw_mention)
    return bool(_NUTRIENT_ROLE.search(t)) and not _ADDITIVE_ROLE.search(t)


# --- Lỗi 1: câu giới hạn tuổi in trên nhãn ---------------------------------------------------------------------------
# Codex CXS 74-1981 mục 8.6.4: nhãn phải ghi rõ tuổi bắt đầu dùng sản phẩm. Nhãn ghi tuổi tối thiểu lớn hơn tuổi của bé
# -> làm theo nhãn: không dùng. Câu không có số (vd "không thích hợp cho trẻ sơ sinh trừ khi có chỉ định") không suy tuổi.
_AGE_UNIT = r"(tuổi|tháng)"
_AGE_PATTERNS = (
    re.compile(r"không\s*(?:được\s*)?(?:dùng|sử\s*dụng|thích\s*hợp|phù\s*hợp)\s*(?:cho|với|đối\s*với)\s*trẻ\s*(?:em\s*)?"
               r"(?:nhỏ\s*)?dưới\s*(\d+)\s*" + _AGE_UNIT),
    re.compile(r"(?:dùng|sử\s*dụng|thích\s*hợp|phù\s*hợp|dành)\s*cho\s*trẻ\s*(?:em\s*)?(?:từ|trên)\s*(\d+)\s*" + _AGE_UNIT),
    re.compile(r"(?<![\d.,])(\d+)\s*\+\s*" + _AGE_UNIT),
)
LABEL_AGE_SOURCE = "Codex CXS 74-1981 mục 8.6.4 (nhãn phải ghi tuổi bắt đầu dùng sản phẩm)"


def label_min_ages(texts) -> list[tuple[int, str]]:
    """[(tuổi tối thiểu theo tháng, câu trên nhãn)] từ các câu chữ của nhãn."""
    out, seen = [], set()
    for raw in texts or []:
        t = _fold(raw)
        for pat in _AGE_PATTERNS:
            for m in pat.finditer(t):
                n, unit = int(m.group(1)), m.group(2)
                months = n * 12 if unit == "tuổi" else n
                sentence = re.split(r"(?<=[.;|])\s+", str(raw))
                excerpt = next((s for s in sentence if m.group(0) in _fold(s)), str(raw)).strip(" .;|")
                if (months, excerpt) not in seen and 0 < months <= 216:
                    seen.add((months, excerpt))
                    out.append((months, excerpt))
    return out


def label_age_alerts(persona: dict | None, question: str, texts) -> list[dict]:
    """Cảnh báo khi nhãn ghi tuổi tối thiểu lớn hơn tuổi của bé (< 12 tháng).

    Không biết tuổi chính xác (chỉ biết dưới 12 tháng) thì chỉ cảnh báo khi nhãn ghi từ 12 tháng / 1 tuổi trở lên."""
    age = infant_age_months(persona, question)
    alerts = []
    for months, excerpt in label_min_ages(texts):
        if (age is not None and age < months) or (age is None and months >= 12):
            who = f"bé {age} tháng tuổi" if age is not None else "trẻ dưới 12 tháng"
            alerts.append({
                "additive": f"Độ tuổi ghi trên nhãn: \"{excerpt}\"",
                "label_age_rule": True,
                "min_age_months": months,
                "label_excerpt": excerpt,
                "message": (f"Nhãn ghi \"{excerpt}\" (tuổi tối thiểu {months} tháng), nên sản phẩm không dành cho {who}. "
                            f"Hãy làm theo hướng dẫn độ tuổi của nhà sản xuất ({LABEL_AGE_SOURCE})."),
                "regulation": "CODEX_CXS_74_1981",
            })
    return alerts


def _loc_vn(r) -> str:
    page = f"tr. {r['printed_page']}" if r.get("printed_page") else f"PDF tr. {r['pdf_page']}"
    return f"TT 24/2019 (09/VBHN-BYT 2024) Phụ lục 2A, nhóm {r['category']}, {page}"


def _loc_codex(r) -> str:
    return f"Codex GSFA CXS 192-1995 Bảng Hai, nhóm {r['category']}, tr. {r['pdf_page']}"


def _ml(r) -> str:
    return "GMP" if r.get("max_level") == "GMP" else f"{r.get('max_level')} {r.get('unit') or 'mg/kg'}"


def evaluate(linked_results) -> dict:
    """Đối chiếu phụ gia trên nhãn. Trả về {alerts, permitted, nutrients, checked}."""
    alerts, permitted, nutrients, seen = [], [], [], set()
    if not available():
        return {"alerts": [], "permitted": [], "nutrients": [], "checked": 0}
    items = list(linked_results or [])
    # cùng mã có thể vừa ở dạng nguồn dinh dưỡng vừa ở mục phụ gia: xét các dòng phụ gia trước
    items.sort(key=lambda it: is_nutrient_source(it.get("raw_mention", "")))
    nutrient_keys = set()
    for item in items:
        ent = item.get("matched_entity") or {}
        ins = str(ent.get("ins_code") or "")
        if not ins or ins.upper().startswith("FOODON"):
            continue
        key = normalize_ins_key(ins)
        if not key or key in seen:
            continue
        name = ent.get("canonical_name_vi") or ent.get("canonical_name") or ins
        label = f"{name} ({ins})" if ins not in name else name
        if is_nutrient_source(item.get("raw_mention", "")):
            if key not in nutrient_keys:
                nutrient_keys.add(key)
                nutrients.append({"additive": label, "ins_code": ins, "raw_mention": item.get("raw_mention", ""),
                                  "note": NUTRIENT_SOURCE_NOTE})
            continue
        seen.add(key)
        vn, cx = _lookup("vn", key), _lookup("codex", key)
        if not vn and not cx:
            alerts.append({
                "additive": label,
                "ins_code": ins,
                "infant_rule": True,
                "message": (f"{label} không có trong danh mục phụ gia được phép cho thực phẩm dành cho trẻ đến 12 tháng "
                            "tuổi và trẻ nhỏ (nhóm 13.1, 13.2) của TT 24/2019/TT-BYT Phụ lục 2A, cũng không có trong "
                            "Codex GSFA Bảng Hai cho các nhóm này. Với thực phẩm cho trẻ nhỏ, chỉ phụ gia được ghi riêng "
                            "cho nhóm này mới được dùng (TT 24/2019 Phụ lục 3 mục 2; GSFA Annex to Table Three)."),
                "regulation": "VN_TT24_2019",
                "codex": "CODEX_STAN_192_1995",
            })
        else:
            first_vn = vn[0] if vn else None
            permitted.append({
                "additive": label, "ins_code": ins,
                "vn": [{"category": r["category"], "max_level": _ml(r), "locator": _loc_vn(r)} for r in vn],
                "codex": [{"category": r["category"], "max_level": _ml(r), "locator": _loc_codex(r)} for r in cx],
                "summary": (f"{label}: được phép cho thực phẩm trẻ nhỏ"
                            + (f" — TT24 nhóm {first_vn['category']} tối đa {_ml(first_vn)}" if first_vn else
                               " — chỉ có trong Codex GSFA (TT24 không ghi cho nhóm này)")),
            })
    return {"alerts": alerts, "permitted": permitted, "nutrients": nutrients, "checked": len(seen)}
