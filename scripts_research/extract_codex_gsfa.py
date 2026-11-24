"""Trích Codex CXS 192-1995 (GSFA) Table One theo từng phụ gia: nhóm thực phẩm, mức tối đa (ML/GMP), năm, chú thích, trang.

Đọc raw/external/cxs_192e.pdf (đúng bản trong manifest, kiểm sha256) bằng PyMuPDF.
Đầu ra: knowledge_base/sources/derived/codex_192_gsfa_table_one.json
    {"source", "sha256", "additives": {<mã chuẩn hoá>: {"codex_code", "name", "functional_class", "entries": [...]}}}
Với --apply, ghi trạng thái đã đối chiếu vào codex_claim của knowledge_base/additives_standard.json
(chỉ các mục có mã khớp Table One; mục khác giữ nguyên).

Chạy: .\venv\Scripts\python.exe scripts_research\extract_codex_gsfa.py [--apply]
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
PDF = ROOT / "knowledge_base" / "sources" / "raw" / "external" / "cxs_192e.pdf"
OUT = ROOT / "knowledge_base" / "sources" / "derived" / "codex_192_gsfa_table_one.json"
KB = ROOT / "knowledge_base" / "additives_standard.json"
MANIFEST = ROOT / "knowledge_base" / "sources" / "catalog" / "external_sources_manifest.json"

HEADER_TOKENS = {"FoodCatNo", "FoodCategory", "MaxLevel", "Notes", "Year Adopted", "Table One", "CXS 192-1995"}
CAT_RE = re.compile(r"^\d{2}\.\d(?:\.\d+)*$")
MAX_RE = re.compile(r"^(GMP|\d[\d,\.]*\s*(?:mg/kg|mg/L|g/kg|g/L|%)(?:\s*\S.*)?)$")
YEAR_RE = re.compile(r"^(19|20)\d\d$")
CODE_RE = re.compile(r"^\d{3,4}[a-z]?(?:\s?\([ivx]+\))*(?:\s*,\s*\d{3,4}[a-z]?(?:\s?\([ivx]+\))*)*$")


def norm(code: str) -> str:
    return re.sub(r"[()\s]", "", re.sub(r"^(ins|e)\s*", "", str(code or "").strip(), flags=re.I)).lower()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_page(lines: list[str]) -> list[str]:
    """Bỏ đầu trang lặp lại (TÊN / số trang / CXS 192-1995 / Table One / tiêu đề cột); trả về mã tên phụ gia ở đầu trang."""
    lines = [ln.strip() for ln in lines if ln.strip()]
    out, i = [], 0
    while i < len(lines):
        if lines[i] == "CXS 192-1995":
            # TÊN, số trang đứng ngay trước dòng này
            if out and re.fullmatch(r"\d+", out[-1]):
                out.pop()
                if out and out[-1].isupper():
                    out.pop()
            i += 1
            continue
        if lines[i] in HEADER_TOKENS:
            i += 1
            continue
        if re.fullmatch(r"[ivx]*\)", lines[i]) and out and re.fullmatch(r"\d{3,4}[a-z]?\([ivx]*", out[-1]):
            out[-1] += lines[i]  # mã bị ngắt dòng: "339(iii" + ")" hoặc "450(vi" + "i)"
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def parse() -> dict:
    doc = fitz.open(PDF)
    first = last = None
    for i, page in enumerate(doc):
        text = page.get_text()
        if first is None and "Table One" in text and "FoodCatNo" in text:
            first = i
        if first is not None and "TABLE TWO" in text:
            last = i - 1
            break
    last = last if last is not None else len(doc) - 1
    additives: dict[str, dict] = {}
    cur = None
    problems = []
    toks, pgs = [], []
    for pno in range(first, last + 1):
        page_toks = clean_page(doc[pno].get_text().split("\n"))
        toks += page_toks
        pgs += [pno + 1] * len(page_toks)
    k = 0
    while k < len(toks):
        # tiêu đề phụ gia / nhóm phụ gia: lặp (<mã>, "INS", <nhóm chức năng>..., "Functional Class:", <tên>)
        # cho từng thành viên; các dòng nhóm thực phẩm ngay sau đó áp dụng chung cho cả nhóm.
        if k + 1 < len(toks) and toks[k + 1] == "INS" and CODE_RE.match(toks[k]):
            shared: list = []
            members = []
            j = k
            while j + 1 < len(toks) and toks[j + 1] == "INS" and CODE_RE.match(toks[j]):
                code = toks[j]
                j += 2
                fclass = []
                while j < len(toks) and toks[j] != "Functional Class:" and not CAT_RE.match(toks[j])                             and not (j + 1 < len(toks) and toks[j + 1] == "INS"):
                    fclass.append(toks[j])
                    j += 1
                name = []
                if j < len(toks) and toks[j] == "Functional Class:":
                    j += 1
                    while j < len(toks) and not CAT_RE.match(toks[j]) and not (j + 1 < len(toks) and toks[j + 1] == "INS"):
                        name.append(toks[j])
                        j += 1
                members.append((code, " ".join(name), " ".join(fclass)))
            group = [c for c, _, _ in members]
            for code, name, fclass in members:
                for single in [c.strip() for c in code.split(",")]:
                    additives[norm(single)] = {"codex_code": single, "name": name, "functional_class": fclass,
                                               "group_members": group if len(group) > 1 else [], "entries": shared}
            cur = {"entries": shared, "codex_code": group[0]}
            k = j
            continue
        if cur is not None and CAT_RE.match(toks[k]):
            cat = toks[k]
            j = k + 1
            cname = []
            while j < len(toks) and not MAX_RE.match(toks[j]) and not CAT_RE.match(toks[j]) \
                    and not (j + 1 < len(toks) and toks[j + 1] == "INS"):
                cname.append(toks[j])
                j += 1
            if j >= len(toks) or not MAX_RE.match(toks[j]):
                problems.append((pgs[k], cur["codex_code"], cat))
                k = j
                continue
            maxlvl = toks[j]
            j += 1
            year = ""
            if j < len(toks) and YEAR_RE.match(toks[j]):
                year = toks[j]
                j += 1
            notes = []
            while j < len(toks) and not CAT_RE.match(toks[j]) and not (j + 1 < len(toks) and toks[j + 1] == "INS"):
                notes.append(toks[j])
                j += 1
            cur["entries"].append({"food_category_no": cat, "food_category": " ".join(cname), "max_level": maxlvl,
                                   "year_adopted": year, "notes": " ".join(notes), "page": pgs[k]})
            k = j
            continue
        k += 1
    return {"first_page": first + 1, "last_page": last + 1, "additives": additives, "problems": problems}


def codex_match(kb_code: str, additives: dict) -> list[str]:
    """Khoá Codex khớp mã KB: đúng mã, hoặc mã nhóm (160a → 160a(i)..), không lẫn số khác (100 ≠ 1000)."""
    n = norm(kb_code)
    return sorted(k for k in additives if k == n or (k.startswith(n) and re.fullmatch(r"[a-z]+", k[len(n):] or "0")))


def summarize(entries: list[dict]) -> str:
    """Tóm tắt đo từ dữ liệu trích (thay cho câu 'ML: 600 - 1000 mg/kg' viết sẵn, không khớp văn bản gốc)."""
    nums = []
    for e in entries:
        m = re.match(r"^(\d[\d,\.]*)\s*mg/kg", e["max_level"])
        if m:
            nums.append(float(m.group(1).replace(",", "")))
    gmp = sum(1 for e in entries if e["max_level"].startswith("GMP"))
    parts = [f"CXS 192-1995 Table One: {len(entries)} quy định theo nhóm thực phẩm"]
    if nums:
        parts.append(f"mức tối đa {min(nums):g}–{max(nums):g} mg/kg ({len(nums)} nhóm)")
    if gmp:
        parts.append(f"{gmp} nhóm theo GMP")
    return "; ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="ghi codex_claim vào additives_standard.json")
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expect = next((x["sha256"] for x in manifest.get("manually_downloaded", []) if x.get("file") == PDF.name), None)
    digest = sha(PDF)
    if expect and digest != expect:
        print(f"SHA-256 không khớp manifest ({digest[:12]} ≠ {expect[:12]}); dừng.", file=sys.stderr)
        return 2
    res = parse()
    OUT.write_text(json.dumps({"source": PDF.name, "sha256": digest, "pages": [res["first_page"], res["last_page"]],
                               "additives": res["additives"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    n_entries = sum(len(v["entries"]) for v in res["additives"].values())
    print(f"Table One: trang {res['first_page']}–{res['last_page']}, {len(res['additives'])} phụ gia, {n_entries} dòng; "
          f"dòng không đọc được mức tối đa: {len(res['problems'])}")
    if args.apply:
        kb = json.loads(KB.read_text(encoding="utf-8"))
        done = []
        for item in kb:
            matched = codex_match(item["ins_code"].replace("INS", "").strip(), res["additives"])
            if not matched:
                continue
            entries = [e for m in matched for e in res["additives"][m]["entries"]]
            item["codex_claim"] = {
                "claim": summarize(entries),
                **{k: v for k, v in item.get("codex_claim", {}).items() if k in ("source_id", "source_url")},
                "verification_status": "source_checked_gsfa_table_one",
                "source_file": f"knowledge_base/sources/raw/external/{PDF.name}",
                "sha256": digest,
                "extract_file": "knowledge_base/sources/derived/codex_192_gsfa_table_one.json",
                "codex_codes": [res["additives"][m]["codex_code"] for m in matched],
                "n_food_category_provisions": len(entries),
                "pages": sorted({e["page"] for e in entries}),
                "use_for_compliance_decision": False,
                "scope": "Đã đối chiếu mã INS với CXS 192-1995 Table One và trích nhóm thực phẩm + mức tối đa. "
                         "Chỉ dùng để tra mức tối đa khi đã xác định đúng mã nhóm thực phẩm của sản phẩm.",
            }
            done.append(item["ins_code"])
        KB.write_text(json.dumps(kb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Đã cập nhật codex_claim: {len(done)}/{len(kb)} mục; không có trong Table One: "
              f"{[i['ins_code'] for i in kb if i['ins_code'] not in done]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
