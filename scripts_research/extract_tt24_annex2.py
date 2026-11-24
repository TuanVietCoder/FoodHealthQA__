"""Cấu trúc hoá Phụ lục 2A và 2B của Thông tư 24/2019/TT-BYT (văn bản hợp nhất 09/VBHN-BYT 2024).

Phụ lục 2A (tr. 46–357): mức sử dụng tối đa theo mã nhóm thực phẩm (giống Codex GSFA Table One).
Phụ lục 2B (tr. 358–409): mức tối đa cho nhóm thực phẩm CHƯA quy định trong Codex STAN 192 (không có mã nhóm).

Đọc theo toạ độ cột của PyMuPDF (mỗi dòng PDF có x0, y0): cột mã nhóm / tên nhóm / ML / ghi chú.
Mỗi hàng giữ số trang PDF và văn bản gốc để kiểm nguyên văn.
Đầu ra: knowledge_base/sources/derived/tt24_annex2_ml.json
    {"source", "sha256", "annex_pages", "groups": [{"annex", "group_name", "members": [{"ins", "name_vi"}],
      "entries": [{"food_category_no", "food_category", "ml", "ml_unit", "notes", "page"}]}],
     "by_ins": {<mã chuẩn hoá>: [chỉ số nhóm]}, "warnings": [...]}
Với --codex: đối chiếu hàng 2A với Codex Table One (cùng mã INS + cùng mã nhóm) và in khác biệt.
KHÔNG thay đổi KB hay hành vi hệ thống.

Chạy: .\\venv\\Scripts\\python.exe scripts_research\\extract_tt24_annex2.py [--codex]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "knowledge_base" / "sources" / "raw" / "vn" / "09_VBHN_BYT_2024.pdf"
OUT = ROOT / "knowledge_base" / "sources" / "derived" / "tt24_annex2_ml.json"
CODEX = ROOT / "knowledge_base" / "sources" / "derived" / "codex_192_gsfa_table_one.json"
ANNEX_PAGES = {"2A": (46, 357), "2B": (358, 387)}
NOTES_PAGES = (388, 409)  # "GIẢI THÍCH GHI CHÚ TẠI PHỤ LỤC 2A VÀ PHỤ LỤC 2B"

CAT_RE = re.compile(r"^\d{2}\.\d(?:\.\d+)*$")
INS_RE = re.compile(r"^\d{3,4}[a-z]?(?:\s?\([ivx]+\))?$")
ML_RE = re.compile(r"^(GMP|\d[\d.,]*)(?:\s*(mg/kg|mg/l))?$")
HEADER = {"Mã nhóm", "thực phẩm", "Nhóm thực phẩm", "ML", "(mg/kg)", "(mg/l)", "Ghi chú", "Tên phụ gia", "INS"}


def norm(code: str) -> str:
    return re.sub(r"[()\s]", "", re.sub(r"^(ins|e)\s*", "", str(code or "").strip(), flags=re.I)).lower()


def page_lines(page, keep_small: bool = False) -> list[dict]:
    """Dòng chữ chính (cỡ ≥ 13pt) kèm toạ độ; bỏ số trang, chú thích chân trang và chỉ số chú thích nhỏ.
    keep_small=True giữ chữ nhỏ trong dòng (chỉ số dưới/trên như CO2, PO4, dm2 ở bảng giải thích ghi chú)."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if keep_small or s["size"] >= 10]  # bỏ chỉ số chú thích (8–9pt)
            text = "".join(s["text"] for s in spans).strip()
            if not text or max(s["size"] for s in spans) < 13:  # chú thích chân trang 12pt
                continue
            x0, y0 = line["bbox"][0], line["bbox"][1]
            if y0 < 55 and re.fullmatch(r"\d+", text):
                continue  # số trang
            out.append({"x": x0, "y": y0, "t": re.sub(r"\s+", " ", text)})
    return sorted(out, key=lambda d: (round(d["y"]), d["x"]))


class Parser:
    def __init__(self, annex: str):
        self.annex = annex
        self.groups: list[dict] = []
        self.warnings: list[str] = []
        self.mode = "start"
        self.pending_name: list[str] = []
        # ranh giới cột (cập nhật mỗi khi gặp dòng tiêu đề bảng)
        self.name_x = 150.0 if annex == "2A" else 85.0
        self.ml_x = 415.0 if annex == "2A" else 385.0
        self.note_x = 478.0 if annex == "2A" else 450.0

    def col(self, ln) -> str:
        if ln["x"] >= self.note_x:
            return "note"
        if ln["x"] >= self.ml_x:
            return "ml"
        if self.annex == "2A" and ln["x"] < self.name_x - 10:
            return "code"
        return "name"

    def feed_page(self, pno: int, lines: list[dict]) -> None:
        # cập nhật ranh giới cột theo tiêu đề bảng trên trang
        for ln in lines:
            if ln["t"] == "ML":
                self.ml_x = ln["x"] - 25
            elif ln["t"] == "Ghi chú":
                self.note_x = ln["x"] - 18
        i = 0
        rows_on_page: list[dict] = []
        while i < len(lines):
            ln = lines[i]
            t = ln["t"]
            if t.startswith(("PHỤ LỤC", "MỨC SỬ DỤNG", "CHƯA ĐƯỢC QUY ĐỊNH", "CODEX STAN", "(Ban hành kèm")):
                i += 1
                continue
            if t == "INS":
                name = " ".join(self.pending_name).strip()
                self.pending_name = []
                self.groups.append({"annex": self.annex, "group_name": name, "members": [], "entries": [], "page": pno})
                self.mode = "members"
                i += 1
                continue
            if t in HEADER:
                if t in ("(mg/kg)", "(mg/l)") and self.groups:
                    self.groups[-1]["unit"] = t.strip("()")
                if t in ("Mã nhóm", "Nhóm thực phẩm"):
                    self.mode = "table"
                i += 1
                continue
            if self.mode == "members":
                if INS_RE.match(t) and ln["x"] < self.name_x - 10 + (0 if self.annex == "2A" else 60):
                    same = [o for o in lines if o is not ln and abs(o["y"] - ln["y"]) < 4 and o["x"] > ln["x"] + 30]
                    self.groups[-1]["members"].append({"ins": t.replace(" ", ""), "name_vi": " ".join(o["t"] for o in same), "page": pno})
                    i += 1
                    continue
                if any(abs(o["y"] - ln["y"]) < 4 and INS_RE.match(o["t"]) for o in lines if o is not ln):
                    i += 1  # tên thành viên đã ghép với mã
                    continue
                # dòng tên thành viên kéo dài sang dòng thứ hai
                if self.groups[-1]["members"]:
                    self.groups[-1]["members"][-1]["name_vi"] += " " + t
                i += 1
                continue
            # chế độ bảng hoặc trang tiếp theo
            c = self.col(ln)
            letters = [ch for ch in t if ch.isalpha()]
            upper_ratio = sum(ch.isupper() for ch in letters) / len(letters) if letters else 0
            upper_group_name = c in ("code", "name") and upper_ratio >= 0.7 and re.search(r"[A-Z]{3}", t) and not CAT_RE.match(t)
            if upper_group_name and self.annex == "2B" and ln["x"] > 120:
                upper_group_name = False
            if upper_group_name:
                self.pending_name.append(t)
                self.mode = "await_ins"
                i += 1
                continue
            if self.mode not in ("table",):
                self.warnings.append(f"p.{pno} bỏ qua (ngoài bảng): {t[:60]}")
                i += 1
                continue
            rows_on_page.append({**ln, "c": c, "g": len(self.groups) - 1})
            i += 1
        for g in sorted({c["g"] for c in rows_on_page}):
            self._assign(pno, g, [c for c in rows_on_page if c["g"] == g])

    def _assign(self, pno: int, gidx: int, cells: list[dict]) -> None:
        if not cells or gidx < 0:
            self.warnings.extend(f"p.{pno} ô trước nhóm đầu tiên: {c['t']}" for c in cells)
            return
        group = self.groups[gidx]
        anchor_col = "code" if self.annex == "2A" else "ml"
        anchors = [c for c in cells if c["c"] == anchor_col and (CAT_RE.match(c["t"]) if anchor_col == "code" else ML_RE.match(c["t"]))]
        if self.annex == "2A":
            for c in cells:
                if c["c"] == "code" and not CAT_RE.match(c["t"]):
                    self.warnings.append(f"p.{pno} mã nhóm lạ: {c['t']}")
        # mỗi ô thuộc hàng có neo gần nhất phía trên (cho phép lệch 8pt vì ML có thể nằm cao hơn dòng tên đầu)
        new_rows = [{"anchor": a, "cells": []} for a in anchors]
        carry: list[dict] = []
        for c in cells:
            owner = None
            for r in new_rows:
                if c["y"] >= r["anchor"]["y"] - 8:
                    owner = r
            if owner is None:
                carry.append(c)  # phần tiếp của hàng ở trang trước
            else:
                owner["cells"].append(c)
        if carry:
            entries = group["entries"]
            if entries:
                last = entries[-1]
                for c in carry:
                    if c["c"] == "name":
                        last["food_category"] = (last["food_category"] + " " + c["t"]).strip()
                    elif c["c"] == "note":
                        last["notes"] = (last["notes"] + " " + c["t"]).strip()
                    else:
                        self.warnings.append(f"p.{pno} ô tiếp trang không rõ cột: {c['t']}")
                if pno not in last.setdefault("pages", [last["page"]]):
                    last["pages"].append(pno)
            else:
                self.warnings.append(f"p.{pno} ô không có hàng: {[c['t'] for c in carry]}")
        for r in new_rows:
            code = r["anchor"]["t"] if self.annex == "2A" else ""
            names = [c["t"] for c in r["cells"] if c["c"] == "name"]
            mls = [c["t"] for c in r["cells"] if c["c"] == "ml"]
            notes = [c["t"] for c in r["cells"] if c["c"] == "note"]
            extra_codes = [c["t"] for c in r["cells"] if c["c"] == "code" and c is not r["anchor"]]
            if extra_codes:
                self.warnings.append(f"p.{pno} hàng {code} có thêm ô cột mã: {extra_codes}")
            if len(mls) != 1 or not ML_RE.match(mls[0]):
                self.warnings.append(f"p.{pno} hàng {code or names[:1]} ML bất thường: {mls}")
            ml = mls[0] if mls else ""
            unit = group.get("unit", "mg/kg")
            mu = ML_RE.match(ml)
            if mu and mu.group(2):
                ml, unit = mu.group(1), mu.group(2)
            group["entries"].append({
                "food_category_no": code,
                "food_category": " ".join(names).strip(),
                "ml": ml,
                "ml_unit": "" if ml == "GMP" else unit,
                "notes": " ".join(notes).strip(),
                "page": pno,
            })


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(pdf: Path = PDF) -> dict:
    doc = fitz.open(pdf)
    groups, warnings = [], []
    for annex, (first, last) in ANNEX_PAGES.items():
        p = Parser(annex)
        for pno in range(first, last + 1):
            p.feed_page(pno, page_lines(doc[pno - 1]))
        groups += p.groups
        warnings += [f"{annex} {w}" for w in p.warnings]
    attach_footnotes(doc, groups, warnings)
    notes, note_warnings = extract_notes(doc)
    warnings += note_warnings
    for g in groups:
        for e in g["entries"]:
            e["note_keys"], free = note_keys(e["notes"])
            if free:
                e["note_text"] = free  # ô ghi chú viết bằng lời, không phải số
    used = {k for g in groups for e in g["entries"] for k in e["note_keys"]}
    missing = sorted(used - set(notes), key=lambda k: (len(k), k))
    if missing:
        warnings.append(f"số ghi chú dùng trong bảng nhưng không có trong bảng giải thích: {missing}")
    by_ins: dict[str, list[int]] = {}
    for idx, g in enumerate(groups):
        for m in g["members"]:
            by_ins.setdefault(norm(m["ins"]), []).append(idx)
        if not g["entries"]:
            warnings.append(f"{g['annex']} nhóm không có hàng: {g['group_name']} (p.{g['page']})")
    return {"source": "knowledge_base/sources/raw/vn/09_VBHN_BYT_2024.pdf", "sha256": sha(pdf),
            "annex_pages": ANNEX_PAGES, "notes_pages": NOTES_PAGES, "groups": groups, "by_ins": by_ins,
            "notes": notes, "warnings": warnings}


def attach_footnotes(doc, groups: list[dict], warnings: list[str]) -> None:
    """Chú thích chân trang của văn bản hợp nhất (vd. nhóm được bổ sung theo TT 17/2023) gắn vào nhóm mang số chú thích."""
    first, last = ANNEX_PAGES["2A"][0], ANNEX_PAGES["2B"][1]
    markers: dict[str, tuple[str, int]] = {}   # số chú thích -> (tên nhóm có chỉ số đó, trang)
    texts: dict[str, dict] = {}    # số chú thích -> nội dung chân trang
    for pno in range(first, last + 1):
        current = None
        for block in doc[pno - 1].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [sp for sp in line["spans"] if sp["text"].strip()]
                if not spans:
                    continue
                big = max(sp["size"] for sp in spans)
                small = [sp["text"].strip() for sp in spans if sp["size"] < 10]
                if big >= 13 and small:
                    name = " ".join(sp["text"].strip() for sp in spans if sp["size"] >= 10)
                    for m in small:
                        markers[m] = (name, pno)
                elif big < 13:  # chân trang 12pt; dòng mở đầu bằng số chú thích 8pt
                    if small and spans[0]["size"] < 10:
                        current = small[0]
                        texts[current] = {"text": "".join(sp["text"] for sp in spans[1:]).strip(), "page": pno}
                    elif current:
                        texts[current]["text"] += " " + "".join(sp["text"] for sp in spans).strip()
    for num, (name, page) in markers.items():
        g = next((g for g in groups if g["page"] == page
                  and re.sub(r"\s+", " ", g["group_name"]).endswith(re.sub(r"\s+", " ", name))), None)  # tên nhóm có thể 2 dòng
        if not g or num not in texts:
            warnings.append(f"chú thích chân trang {num} ({name}) không gắn được")
            continue
        g["footnote"] = {"number": num, **texts[num]}


NOTE_KEY_RE = re.compile(r"XS\s?\d+R?|\d+")


def note_keys(notes: str) -> tuple[list[str], str]:
    """'3, 53, XS 36 XS92 & 16' -> (['3', '53', 'XS36', 'XS92', '16'], ''); chữ tự do trả về ở phần thứ hai."""
    if not notes or not re.fullmatch(r"[\dXSR,&\s]+", notes):
        return [], notes
    return [k.replace(" ", "") for k in NOTE_KEY_RE.findall(notes)], ""


def extract_notes(doc) -> tuple[dict, list[str]]:
    """Bảng giải thích số ghi chú: {số: {"text", "page"}}. Ranh giới hàng lấy từ khung bảng (find_tables);
    chữ lấy từ page_lines vì find_tables tách sai dấu tiếng Việt."""
    notes: dict[str, dict] = {}
    warnings: list[str] = []
    last = None
    for pno in range(NOTES_PAGES[0], NOTES_PAGES[1] + 1):
        page = doc[pno - 1]
        lines = page_lines(page, keep_small=True)
        rows = [r for t in page.find_tables().tables for r in t.rows]
        taken: set[int] = set()  # khung bảng có thể tách một ô thành hai hàng chồng nhau: mỗi dòng chỉ thuộc một hàng
        for row in rows:
            y0, y1 = row.bbox[1], row.bbox[3]
            cells = [ln for k, ln in enumerate(lines) if k not in taken and y0 - 2 <= ln["y"] + 6 <= y1 + 2]
            taken.update(k for k, ln in enumerate(lines) if any(ln is c for c in cells))
            keys = [ln for ln in cells if ln["x"] < 150]
            text = " ".join(ln["t"] for ln in cells if ln["x"] >= 150).strip()
            if keys and keys[0]["t"] == "Số ghi chú":
                continue
            if not keys:
                if last and text:
                    notes[last]["text"] += " " + text  # phần tiếp của ghi chú ở trang trước
                    pages = notes[last].setdefault("pages", [notes[last]["page"]])
                    if pno not in pages:
                        pages.append(pno)
                continue
            key = keys[0]["t"]
            if len(keys) > 1 or not re.fullmatch(r"\d+|XS\d+R?", key):
                warnings.append(f"ghi chú p.{pno}: ô số lạ {[k['t'] for k in keys]}")
                continue
            if key in notes:
                warnings.append(f"ghi chú p.{pno}: số {key} lặp lại")
            notes[key] = {"text": text, "page": pno}
            last = key
    return notes, warnings


def codex_ml(value: str) -> str:
    """'150 mg/kg' -> '150'; 'GMP' -> 'GMP'; đơn vị khác giữ nguyên."""
    m = re.match(r"^(GMP|[\d.,]+)\s*(mg/kg)?", value or "")
    if not m:
        return value
    return m.group(1).replace(",", "") if m.group(2) or m.group(1) == "GMP" else value


def same_ml(tt24: str, codex: str) -> bool:
    """TT24 dùng dấu phẩy thập phân ('6,4'); Codex dùng dấu chấm và đã bỏ đơn vị mg/kg ở codex_ml."""
    if tt24 == "GMP" or codex == "GMP":
        return tt24 == codex
    try:
        return float(tt24.replace(",", ".")) == float(codex)
    except ValueError:
        return False


def compare_codex(data: dict) -> dict:
    codex = json.loads(CODEX.read_text(encoding="utf-8"))["additives"]
    stats = {"pairs": 0, "same": 0, "different": 0, "different_codex_revised_after_2018": 0, "only_tt24": 0}
    diffs = []
    for g in data["groups"]:
        if g["annex"] != "2A":
            continue
        for m in g["members"]:
            c = codex.get(norm(m["ins"]))
            if not c:
                continue
            cmap = {e["food_category_no"]: e for e in c["entries"]}
            for e in g["entries"]:
                ce = cmap.get(e["food_category_no"])
                if not ce:
                    stats["only_tt24"] += 1
                    continue
                stats["pairs"] += 1
                if same_ml(e["ml"], codex_ml(ce["max_level"])):
                    stats["same"] += 1
                else:
                    stats["different"] += 1
                    year = int(ce.get("year_adopted") or 0)
                    if year > 2018:  # TT24 theo Codex STAN 192-1995 (2018); Codex đã sửa sau đó
                        stats["different_codex_revised_after_2018"] += 1
                    diffs.append({"ins": m["ins"], "cat": e["food_category_no"], "tt24": e["ml"], "codex": ce["max_level"],
                                  "codex_year": year, "page": e["page"]})
    return {"stats": stats, "diffs": diffs}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--codex", action="store_true", help="đối chiếu 2A với Codex Table One")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    data = extract()
    n_entries = sum(len(g["entries"]) for g in data["groups"])
    for annex in ANNEX_PAGES:
        gs = [g for g in data["groups"] if g["annex"] == annex]
        print(f"Phụ lục {annex}: {len(gs)} nhóm phụ gia, {sum(len(g['members']) for g in gs)} mã INS, {sum(len(g['entries']) for g in gs)} hàng")
    print(f"Tổng {n_entries} hàng; cảnh báo: {len(data['warnings'])}")
    for w in data["warnings"][:40]:
        print("  ", w)
    if args.codex:
        cmp = compare_codex(data)
        data["codex_comparison"] = cmp
        print("Đối chiếu Codex:", cmp["stats"])
        for d in cmp["diffs"][:30]:
            print("   khác:", d)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Đã ghi", args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
