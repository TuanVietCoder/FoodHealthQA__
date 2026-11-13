"""Extract (INS code -> name block) from TT24/2019 Phụ lục 1 and cross-check scripts_research/additive_names.py.

Output: knowledge_base/sources/derived/tt24_additive_names.json
Each entry keeps the raw name block (Vietnamese + English names run together; the PDF wraps lines so they cannot be split reliably).
"""
import json
import re
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "knowledge_base" / "sources" / "raw" / "vn" / "TT24_2019_TT-BYT_phu_luc_full.pdf"
OUT = ROOT / "knowledge_base" / "sources" / "derived" / "tt24_additive_names.json"
FUNC = re.compile(r"^(Chất|Phẩm màu|Enzym|Chế phẩm|Hương|Khí)\b")
CODE = re.compile(r"^\d{3,4}[a-z]?(\([ivx]+\))*$", re.I)


def main():
    lines = []
    for page in fitz.open(PDF):
        lines += [l.strip() for l in page.get_text().splitlines() if l.strip()]
    entries, i = {}, 0
    while i < len(lines) - 2:
        if re.fullmatch(r"\d{1,3}\.", lines[i]) and CODE.match(lines[i + 1]):
            code = lines[i + 1].lower()
            j, block = i + 2, []
            while j < len(lines) and not FUNC.match(lines[j]) and not re.fullmatch(r"\d{1,3}\.", lines[j]):
                block.append(lines[j])
                j += 1
            entries.setdefault(code, " ".join(block))
            i = j
        else:
            i += 1
    OUT.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(entries), "codes ->", OUT)


if __name__ == "__main__":
    main()
