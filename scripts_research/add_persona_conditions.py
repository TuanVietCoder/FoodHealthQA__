"""Sinh bản benchmark mới có hai cột `persona_conditions` và `persona_conditions_vi`.

Nguồn: `persona_name` của từng bản ghi dị ứng ("Người dị ứng sữa, hải sản, tôm").
Không ghi đè tệp benchmark gốc; kết quả nằm ở tệp mới:
    data/qa_benchmark_persona_conditions.jsonl
    data/qa_benchmark_persona_conditions.csv

Chạy:  python scripts_research/add_persona_conditions.py
Bản ghi dị ứng có từ không ánh xạ được -> báo lỗi và không ghi tệp.
"""
import csv
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
SRC_DIR = (
    os.path.join(DATA, "archive", "master_benchmark_2311")
    if os.path.exists(os.path.join(DATA, "archive", "master_benchmark_2311", "qa_benchmark.jsonl"))
    else DATA
)
SRC_JSONL = os.path.join(SRC_DIR, "qa_benchmark.jsonl")
SRC_CSV = os.path.join(SRC_DIR, "qa_benchmark.csv")
OUT_JSONL = os.path.join(SRC_DIR, "qa_benchmark_persona_conditions.jsonl")
OUT_CSV = os.path.join(SRC_DIR, "qa_benchmark_persona_conditions.csv")

# (từ khóa trong tên persona) -> mã điều kiện; mã nằm trong condition_code_map của pipeline
KEYWORD_TO_CODE = [
    ("đậu nành", "soy_allergy"),
    ("sữa", "milk_allergy"),
    ("trứng", "egg_allergy"),
    ("lúa mì", "wheat_allergy"),
    ("gluten", "wheat_allergy"),
    ("đậu phộng", "peanut_allergy"),
    ("hạt cây", "tree_nut_allergy"),
    ("mè", "sesame_allergy"),
    ("vừng", "sesame_allergy"),
    ("cá", "fish_allergy"),
    ("tôm", "shellfish_allergy"),
    ("cua", "shellfish_allergy"),
    ("hải sản", "shellfish_allergy"),
    ("động vật có vỏ", "shellfish_allergy"),
]
PREFIX = re.compile(r"^Người dị ứng\s+", re.I)


def split_items(rest):
    """Tách theo dấu phẩy nhưng giữ nguyên nội dung trong ngoặc: 'lúa mì (gluten)'."""
    return [x.strip() for x in re.split(r",\s*(?![^()]*\))", rest) if x.strip()]


def derive(persona_name):
    rest = PREFIX.sub("", persona_name).strip()
    if rest == persona_name:
        raise ValueError(f"Tên persona dị ứng không có tiền tố 'Người dị ứng': {persona_name!r}")
    codes, unmapped = [], []
    for item in split_items(rest):
        low = item.lower()
        found = [code for kw, code in KEYWORD_TO_CODE if kw in low]
        if not found:
            unmapped.append(item)
        for code in found:
            if code not in codes:
                codes.append(code)
    if unmapped:
        raise ValueError(f"Không ánh xạ được {unmapped} trong {persona_name!r}")
    return ",".join(codes), f"dị ứng {rest}"


def enrich(record):
    rec = dict(record)
    if str(rec.get("persona_id", "")).strip().lower() == "allergy_patient":
        rec["persona_conditions"], rec["persona_conditions_vi"] = derive(str(rec.get("persona_name", "")))
    else:
        rec["persona_conditions"], rec["persona_conditions_vi"] = "", ""
    return rec


def main():
    errors, rows = [], []
    with open(SRC_JSONL, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(enrich(json.loads(line)))
            except ValueError as exc:
                errors.append(str(exc))
    if errors:
        print("DỪNG - có bản ghi không ánh xạ được:")
        for e in sorted(set(errors)):
            print("  -", e)
        sys.exit(1)

    with open(OUT_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(SRC_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        csv_rows = [enrich(r) for r in reader]
    for extra in ("persona_conditions", "persona_conditions_vi"):
        if extra not in fields:
            fields.append(extra)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)

    n_allergy = sum(1 for r in rows if r["persona_conditions"])
    print(f"Đã ghi {len(rows)} bản ghi ({n_allergy} bản ghi dị ứng có persona_conditions) vào:")
    print("  ", OUT_JSONL)
    print("  ", OUT_CSV)


if __name__ == "__main__":
    main()
