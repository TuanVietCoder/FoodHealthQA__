"""Kiểm một nguồn có THỰC SỰ được hệ thống dùng hay chỉ được nhắc trong tài liệu.

Với mỗi nguồn, đếm số lần xuất hiện ở các nơi mà câu trả lời của chat và bộ đánh giá lấy dữ liệu:
  file       có tệp nguồn trong knowledge_base/sources/raw
  KB         knowledge_base/additives_standard.json (lưu ý, ADI, nguồn của từng phụ gia)
  catalog    knowledge_base/sources/catalog/source_citation_catalog.json (mã trích dẫn để mở nguồn)
  mã công nhận  modules.benchmark.RECOGNIZED_CITATION_SOURCES (cổng nguồn chỉ cho phép các mã này)
  đoạn       kho đoạn bằng chứng mà chat truy xuất (passage_retriever.build_corpus)
  bộ QA      data/foodhealthqa_eval_set.csv
  mã chạy    app.py, modules/ (không tính tài liệu và test)
Nguồn mà cả bảy cột đều 0 (hoặc chỉ có "file") là nguồn không dùng.

Chạy: .\\venv\\Scripts\\python.exe scripts_research\\audit_source_usage.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = {
    "KDIGO 2024": [r"KDIGO"],
    "ACR 2020 (gút)": [r"\bACR[ _]?20\d\d", r"American College of Rheumatology", r"ACR_2020"],
    "ACOG PB 190": [r"ACOG"],
    "ADA Standards of Care": [r"\bADA[ _]?20\d\d", r"American Diabetes Association", r"Standards of Care in Diabetes"],
    "WHO Fact Sheet 394": [r"Fact Sheet(?: No\.?)? ?394", r"WHO_HEALTHY_DIET"],
    "KDOQI 2020": [r"KDOQI"],
}
RAW = ROOT / "knowledge_base" / "sources" / "raw"
FILE_HINTS = {"KDIGO 2024": "kdigo", "ACR 2020 (gút)": "acr", "ACOG PB 190": "acog", "ADA Standards of Care": "ada_standards",
              "WHO Fact Sheet 394": "fact", "KDOQI 2020": "kdoqi"}


def _count(patterns, text):
    return sum(len(re.findall(p, text, re.I)) for p in patterns)


def _code_text() -> str:
    parts = [(ROOT / "app.py").read_text(encoding="utf-8", errors="ignore")]
    parts += [p.read_text(encoding="utf-8", errors="ignore") for p in (ROOT / "modules").glob("*.py")]
    return "\n".join(parts)


def usage() -> dict[str, dict[str, int]]:
    from modules.benchmark import RECOGNIZED_CITATION_SOURCES
    from modules import passage_retriever

    kb = (ROOT / "knowledge_base" / "additives_standard.json").read_text(encoding="utf-8")
    catalog = (ROOT / "knowledge_base" / "sources" / "catalog" / "source_citation_catalog.json").read_text(encoding="utf-8")
    recognised = "\n".join(f"{k} {v}" for k, v in RECOGNIZED_CITATION_SOURCES.items())
    passages = "\n".join(f"{p.get('citation_id', '')} {p.get('locator', '')} {p.get('text', '')}" for p in passage_retriever.build_corpus())
    csv.field_size_limit(10 ** 9)
    eval_text = (ROOT / "data" / "foodhealthqa_eval_set.csv").read_text(encoding="utf-8-sig", errors="ignore")
    code = _code_text()
    out = {}
    for name, patterns in SOURCES.items():
        hint = FILE_HINTS[name]
        has_file = any(hint in p.name.lower() for p in RAW.rglob("*") if p.is_file())
        out[name] = {"file": int(has_file), "KB": _count(patterns, kb), "catalog": _count(patterns, catalog),
                     "mã công nhận": _count(patterns, recognised), "đoạn": _count(patterns, passages),
                     "bộ QA": _count(patterns, eval_text), "mã chạy": _count(patterns, code)}
    return out


def main() -> int:
    table = usage()
    cols = ["file", "KB", "catalog", "mã công nhận", "đoạn", "bộ QA", "mã chạy"]
    print(f"{'Nguồn':24}" + "".join(f"{c:>13}" for c in cols) + "  Kết luận")
    for name, row in table.items():
        used = any(row[c] for c in cols[1:])
        verdict = "ĐANG DÙNG" if used else ("có file, chưa dùng" if row["file"] else "không dùng, không có file")
        print(f"{name:24}" + "".join(f"{row[c]:>13}" for c in cols) + f"  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
