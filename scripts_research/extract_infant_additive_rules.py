"""Trích danh mục phụ gia được phép cho THỰC PHẨM DÀNH CHO TRẺ DƯỚI 12 THÁNG / TRẺ NHỎ từ văn bản gốc đã lưu.

Hai nguồn (đều đã có trong knowledge_base/sources/raw, không tải mạng):
  1. Codex GSFA CXS 192-1995 (cxs_192e.pdf)
     - Mục 3.3 (trang 5): chất mang phụ gia không được dùng cho nhóm 13.1 và 13.2 trừ khi Bảng 1–2 ghi riêng.
     - "Annex to Table Three" (trang 493): nhóm 13.1 và 13.2 KHÔNG áp dụng quy tắc chung "dùng theo GMP" của Bảng 3
       → chỉ phụ gia được ghi tên cho nhóm đó ở Bảng 1–2 mới được phép.
     - Bảng Hai, nhóm 13.1.1 / 13.1.2 / 13.1.3 / 13.2: từng phụ gia, mã INS, mức tối đa, ghi chú, năm.
  2. TT 24/2019/TT-BYT, bản hợp nhất 09/VBHN-BYT 2024 (09_VBHN_BYT_2024.pdf)
     - Phụ lục 2A: các dòng nhóm 13.1.1 / 13.1.2 / 13.1.3 / 13.2 dưới từng phụ gia (INS, tên, ML, ghi chú).
     - Phụ lục 3, mục 2 "Đối tượng thực phẩm sử dụng theo GMP": danh sách nhóm KHÔNG có 13.1, 13.2.

Cần PyMuPDF (python hệ thống có sẵn; venv dự án không có):
  python scripts_research/extract_infant_additive_rules.py
Đầu ra: knowledge_base/sources/derived/infant_food_additives.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # bản PyMuPDF cũ
    import fitz

ROOT = Path(__file__).resolve().parents[1]
CODEX_PDF = ROOT / "knowledge_base" / "sources" / "raw" / "external" / "cxs_192e.pdf"
VN_PDF = ROOT / "knowledge_base" / "sources" / "raw" / "vn" / "09_VBHN_BYT_2024.pdf"
OUT = ROOT / "knowledge_base" / "sources" / "derived" / "infant_food_additives.json"

CATS = ("13.1.1", "13.1.2", "13.1.3", "13.2")
ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
_YEAR = re.compile(r"^(19|20)\d{2}$")
_ML = re.compile(r"^(GMP|\d[\d.,]*)$")
_NOTES = re.compile(r"^(?:XS\d+|\d+)(?:\s*(?:&|,)\s*(?:XS\d+|\d+))*$")


# ---------------------------------------------------------------------------
# mở rộng mã INS: "307a, b, c" · "304, 305" · "339(i)-(iii)" · "450(i)-(iii),(v)-(vii), (ix)"
# ---------------------------------------------------------------------------
def _romans(suffix: str) -> list[str]:
    out = []
    for part in re.findall(r"\(([ivx]+)\)(?:\s*-\s*\(([ivx]+)\))?", suffix):
        a, b = part
        if b and a in ROMAN and b in ROMAN:
            out += [f"({r})" for r in ROMAN[ROMAN.index(a):ROMAN.index(b) + 1]]
        else:
            out.append(f"({a})")
    return out or [""]


def expand_ins(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    codes, base = [], ""
    for tok in re.split(r"[;,]", text):
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"^(\d{3,4}[a-z]?)(.*)$", tok)
        if m:
            base = m.group(1)
            codes += [base + s for s in _romans(m.group(2))]
        elif re.fullmatch(r"[a-z]", tok) and base:
            base = re.sub(r"[a-z]$", "", base) + tok
            codes.append(base)
        elif tok.startswith("(") and base:
            codes += [base + s for s in _romans(tok)]
    return list(dict.fromkeys(codes))


# ---------------------------------------------------------------------------
# Codex GSFA
# ---------------------------------------------------------------------------
def _find_page(doc, pattern, start=0) -> int:
    rx = re.compile(pattern, re.I | re.S)
    for i in range(start, len(doc)):
        if rx.search(doc[i].get_text().replace("\n", " ")):
            return i
    raise ValueError(pattern)


def codex_rules(doc) -> dict:
    p_carry = _find_page(doc, r"unacceptable for foods belonging to the\s+following food categories")
    p_annex = _find_page(doc, r"ANNEX TO TABLE THREE.{0,200}Excluded from the General Conditions")
    p_t2 = _find_page(doc, r"TABLE TWO\s+Food Categories or Individual Food Items")
    rows = {c: [] for c in CATS}
    cat, buf = None, []

    def flush():
        if cat in rows:
            rows[cat].extend(_codex_rows(buf))

    for pi in range(p_t2, len(doc)):
        text = doc[pi].get_text()
        if "TABLE THREE" in text and pi > p_t2:
            break
        for block in re.split(r"(?=Food Category No\.\s)", text):
            m = re.match(r"Food Category No\.\s*([\d.]+)", block)
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            if m:
                new = m.group(1)
                # bỏ khối tiêu đề "Food Category No. … Year Adopted"
                try:
                    lines = lines[lines.index("Year Adopted") + 1:]
                except ValueError:
                    lines = []
                if new != cat:
                    flush()
                    cat, buf = new, []
            else:
                lines = [l for l in lines if not re.fullmatch(r"\d{1,3}|CXS 192-1995|Table Two", l)]
            buf += [(l, pi + 1) for l in lines]
    flush()
    return {
        "source": "Codex GSFA CXS 192-1995 (cxs_192e.pdf)",
        "general_rules": [
            {"locator": f"CXS 192-1995, Preamble §3.3 (PDF trang {p_carry + 1})",
             "text": "Chất mang phụ gia không được chấp nhận cho nhóm 13.1 (sữa công thức cho trẻ sơ sinh, tiếp theo, "
                     "y tế đặc biệt cho trẻ sơ sinh) và 13.2 (thực phẩm bổ sung cho trẻ sơ sinh và trẻ nhỏ), "
                     "trừ khi Bảng 1 và 2 có quy định phụ gia cho nhóm đó."},
            {"locator": f"CXS 192-1995, Annex to Table Three (PDF trang {p_annex + 1})",
             "text": "Nhóm 13.1 và 13.2 nằm trong danh sách loại trừ khỏi điều kiện chung của Bảng 3 (dùng theo GMP): "
                     "việc dùng phụ gia cho các nhóm này chỉ theo quy định ở Bảng 1 và 2."},
        ],
        "categories": rows,
    }


def _codex_rows(lines) -> list[dict]:
    out, cur = [], []
    for l, p in lines:
        prev = cur[-1][0] if cur else ""
        cur.append((l, p))
        # năm thông qua đứng ngay sau "GMP" hoặc đơn vị; ML "2000" (trùng dạng năm) đứng trước ghi chú/đơn vị
        if _YEAR.match(l) and (prev == "GMP" or re.search(r"/(kg|L|l)$", prev)):
            r = _codex_row(cur)
            if r:
                out.append(r)
            cur = []
    return out


def _codex_row(chunk) -> dict | None:
    texts = [l for l, _ in chunk[:-1]]
    i = 0
    ins_parts = []
    while i < len(texts) and re.match(r"^[\d(]", texts[i]) and not re.search(r"[A-Z]{3}", texts[i]):
        ins_parts.append(texts[i]); i += 1
    name = []
    while i < len(texts) and re.search(r"[A-Z]{2}", texts[i]) and texts[i] != "GMP" and not _ML.match(texts[i]) \
            and not _NOTES.match(texts[i]):
        name.append(texts[i]); i += 1
    rest = texts[i:]
    if not ins_parts or not name:
        return None
    if "GMP" in rest:
        ml, unit = "GMP", ""
        notes = [x for x in rest[:rest.index("GMP")] if x]
    else:
        ml = rest[0] if rest else ""
        unit = next((x for x in rest if x.endswith("/kg") or x.endswith("/L") or x.endswith("/l")), "")
        notes = [x for x in rest[1:] if x != unit]
    ins_text = " ".join(ins_parts)
    return {"ins_text": ins_text, "ins_codes": expand_ins(ins_text), "name": " ".join(name),
            "max_level": ml, "unit": unit, "notes": " ".join(notes), "year": chunk[-1][0],
            "pdf_page": chunk[0][1]}


# ---------------------------------------------------------------------------
# TT 24/2019 (09/VBHN-BYT 2024)
# ---------------------------------------------------------------------------
def vn_rules(doc) -> dict:
    p2a = _find_page(doc, r"PHỤ LỤC 2A\s+\|?\s*MỨC SỬ DỤNG TỐI ĐA")
    p2b = _find_page(doc, r"PHỤ LỤC 2B", p2a + 1)
    p3 = _find_page(doc, r"PHỤ LỤC 3\s.{0,80}DANH MỤC PHỤ GIA THỰC PHẨM VÀ ĐỐI TƯỢNG", p2b)
    p3_2 = _find_page(doc, r"2\.\s*Đối tượng thực phẩm sử dụng theo GMP", p3)
    p4 = _find_page(doc, r"PHỤ LỤC 4", p3_2)
    gmp_cats = set()
    for pi in range(p3_2, p4):
        gmp_cats |= set(re.findall(r"(?m)^\s*(\d{2}\.\d+(?:\.\d+)*)\s*$", doc[pi].get_text()))

    lines = []
    for pi in range(p2a, p2b):
        printed = None
        for l in doc[pi].get_text().splitlines():
            l = l.strip()
            if not l:
                continue
            if printed is None and re.fullmatch(r"\d{1,3}", l):
                printed = l
                continue
            lines.append((l, pi + 1, printed))
    rows = {c: [] for c in CATS}
    add = None
    for i, (l, p, printed) in enumerate(lines):
        if l == "INS" and i + 3 < len(lines) and lines[i + 1][0] == "Tên phụ gia":
            name_en = re.sub(r"\d+$", "", lines[i - 1][0]).strip() if i else ""
            code = lines[i + 2][0]
            add = {"ins_text": code, "ins_codes": expand_ins(code), "name_en": name_en, "name_vi": lines[i + 3][0]}
            continue
        if l in CATS and add:
            j, name = i + 1, []
            while j < len(lines) and not _ML.match(lines[j][0]) and len(name) < 6:
                name.append(lines[j][0]); j += 1
            if j >= len(lines):
                continue
            ml = lines[j][0]
            notes = lines[j + 1][0] if j + 1 < len(lines) and _NOTES.match(lines[j + 1][0]) \
                and not re.fullmatch(r"\d{2}\.\d.*", lines[j + 1][0]) else ""
            rows[l].append({**add, "category_name_vi": " ".join(name), "max_level": ml,
                            "unit": "" if ml == "GMP" else "mg/kg", "notes": notes,
                            "pdf_page": p, "printed_page": printed})
    return {
        "source": "TT 24/2019/TT-BYT, bản hợp nhất 09/VBHN-BYT 2024 (09_VBHN_BYT_2024.pdf)",
        "general_rules": [
            {"locator": f"09/VBHN-BYT, Phụ lục 3 mục 2 'Đối tượng thực phẩm sử dụng theo GMP' (PDF trang {p3_2 + 1}–{p4})",
             "text": "Danh sách nhóm thực phẩm được dùng phụ gia theo GMP không có nhóm 13.1 và 13.2 "
                     "→ thực phẩm cho trẻ chỉ dùng phụ gia được ghi riêng cho nhóm này ở Phụ lục 2A/2B.",
             "check_13_1_or_13_2_listed": sorted(c for c in gmp_cats if c.startswith(("13.1", "13.2")))},
        ],
        "annex_2a_pdf_pages": [p2a + 1, p2b],
        "categories": rows,
    }


def main():
    codex = codex_rules(fitz.open(CODEX_PDF))
    vn = vn_rules(fitz.open(VN_PDF))
    out = {
        "description": "Phụ gia được phép cho thực phẩm dành cho trẻ sơ sinh và trẻ nhỏ (nhóm 13.1.x, 13.2). "
                       "Sinh bởi scripts_research/extract_infant_additive_rules.py — không sửa tay.",
        "category_names_vi": {
            "13.1.1": "Sản phẩm dinh dưỡng công thức dành cho trẻ đến 12 tháng tuổi",
            "13.1.2": "Sản phẩm dinh dưỡng công thức với mục đích ăn bổ sung cho trẻ từ 6 đến 36 tháng tuổi",
            "13.1.3": "Sản phẩm dinh dưỡng công thức với mục đích y tế đặc biệt cho trẻ đến 12 tháng tuổi",
            "13.2": "Thực phẩm bổ sung cho trẻ đến 36 tháng tuổi",
        },
        "codex": codex,
        "vn": vn,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for src in ("codex", "vn"):
        print(src, {c: len(v) for c, v in out[src]["categories"].items()})
    print("VN GMP list contains 13.1/13.2:", vn["general_rules"][0]["check_13_1_or_13_2_listed"])


if __name__ == "__main__":
    main()
