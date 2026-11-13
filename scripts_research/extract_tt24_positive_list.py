"""Extract the INS codes of TT 24/2019/TT-BYT Phụ lục 1 (positive list) from the source PDF.

Needs PyMuPDF (fitz). Output: knowledge_base/sources/derived/tt24_positive_list_codes.json
"""
import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "knowledge_base" / "sources" / "raw" / "vn" / "TT24_2019_TT-BYT_phu_luc_full.pdf"
OUT = ROOT / "knowledge_base" / "sources" / "derived" / "tt24_positive_list_codes.json"

doc = fitz.open(PDF)
text = "\n".join(p.get_text() for p in doc)
start = text.index("PHỤ LỤC 1")
end = text.find("PHỤ LỤC 2", start + 10)
section = text[start:end if end > 0 else len(text)]
lines = [l.strip() for l in section.splitlines()]

code_re = re.compile(r"^\d{3,4}[a-z]?(\([ivx]+\))*$", re.I)
codes, rows = set(), 0
for i, line in enumerate(lines[:-1]):
    if not re.fullmatch(r"\d{1,3}\.", line):
        continue
    nxt = lines[i + 1]
    if code_re.match(nxt.replace(" ", "")):
        code = nxt.replace(" ", "")
    else:  # code and name share a line, e.g. "160a(iii) Beta-Caroten," or "E914" (wax)
        m = re.match(r"^(E?\d{3,4}[a-z]?(?:\([ivx]+\))*)(?=\s|$)", nxt, re.I)
        if not m:
            continue
        code = m.group(1)
    codes.add(code.lower().lstrip("e") if code.lower().startswith("e9") else code.lower())
    rows += 1
last_stt = max(int(l[:-1]) for l in lines if re.fullmatch(r"\d{1,3}\.", l))
OUT.write_text(json.dumps({
    "source": "TT24_2019_TT-BYT_phu_luc_full.pdf, Phụ lục 1 (Danh mục phụ gia được phép)",
    "rows_parsed": rows, "highest_stt_seen": last_stt, "codes": sorted(codes)}, ensure_ascii=False, indent=1),
    encoding="utf-8")
print("rows", rows, "unique codes", len(codes), "highest STT", last_stt)
for c in ("330", "331(iii)", "621", "171", "211", "300", "951", "1442", "322(i)", "500(ii)"):
    print(c, c in codes)
