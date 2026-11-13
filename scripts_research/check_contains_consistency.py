"""FDA Q&A on food allergens (Edition 5) D.5/D.6: when a "Contains" statement is used it must name EVERY major allergen that is
in the product, including those already visible in the ingredient list. This script flags labels whose allergen statement
omits an allergen that the ingredient list names.

A flag means one of three things (decided by looking at the photo, see data/label_contains_consistency.csv):
  * manufacturer_omission      - the label itself is inconsistent (the transcription is right)
  * transcription_error        - the photo has the allergen in its statement and the transcription missed it
  * checker_false_positive     - the ingredient is not really an allergen source (e.g. soy milk matched as milk)

Output: data/label_contains_consistency.csv
"""

from __future__ import annotations

import csv
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import create_benchmark_v5 as b  # noqa: E402

CONTAINS = re.compile(r"có chứa|chứa|contains|thông tin dị ứng|thông tin gây dị ứng|chất gây dị ứng|thành phần gây dị ứng", re.I)
ALLERGENS = ("sữa", "trứng", "đậu phộng", "đậu nành", "lúa mì", "cá", "giáp xác", "mè", "hạt cây")
GENERIC_COVER = {"cá": re.compile(r"hải sản|thủy sản|seafood", re.I), "giáp xác": re.compile(r"hải sản|thủy sản|seafood", re.I),
                 "hạt cây": re.compile(r"các loại hạt|hạt", re.I)}


def contains_statement(label: dict) -> str:
    sentences = []
    for item in label.get("warning") or []:
        for s in re.split(r"(?<=[.;])\s+|\n", str(item)):
            if (CONTAINS.search(s) and not b.PAL_MARKER.search(s) and not re.search(r"^không sử dụng", s.strip(), re.I)
                    and not re.search(r"gói hút|hút ẩm|hút oxy|không chứa|không có", s, re.I)):
                sentences.append(s.strip())
    return " ".join(sentences)


def main() -> None:
    out = []
    path = ROOT / "data" / "label_contains_consistency.csv"
    kept = {}
    if path.exists():  # keep verdicts that were entered after looking at the photos
        for r in csv.DictReader(path.open(encoding="utf-8-sig")):
            kept[(r["label"], r["allergen_missing_from_contains"])] = (r["verdict_after_photo"], r["note"])
    for f in sorted(glob.glob(str(ROOT / "assets" / "labels" / "*.json"))):
        label = json.loads(Path(f).read_text(encoding="utf-8"))
        stmt = contains_statement(label)
        if not stmt:
            continue
        ing = " ; ".join(str(x) for x in (label.get("ingredient") or []))
        ing = b.REFINED_OIL.sub(" ", ing)
        add = " ; ".join(str(x) for x in (label.get("additive") or []))
        for a in ALLERGENS:
            if not b.has_term(ing, a):
                continue
            if b.has_term(stmt, a):
                continue
            via = next((m.group(0) for p in b.ALLERGEN_TERMS.get(a, []) for m in [re.search(p, ing.lower())] if m), "")
            generic = bool(GENERIC_COVER.get(a) and GENERIC_COVER[a].search(stmt))
            out.append({"label": Path(f).stem, "product": label.get("product_name", "")[:60], "allergen_missing_from_contains": a,
                        "ingredient_term": via, "contains_statement": stmt[:160],
                        "stated_only_generically": "yes" if generic else "no",
                        "verdict_after_photo": kept.get((Path(f).stem, a), ("", ""))[0],
                        "note": kept.get((Path(f).stem, a), ("", ""))[1]})
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(len(out), "flags on", len({o["label"] for o in out}), "labels ->", path)


if __name__ == "__main__":
    main()
