"""Tạo danh mục cụm tên thành phần cho bộ sửa chính tả theo cụm (modules/ocr_lexicon.py).

Nguồn: danh sách thành phần chuẩn (product_ingredients) của các ảnh thuộc phần PHÁT TRIỂN (dev) của bộ đánh giá —
chia theo ảnh bằng đúng hàm split_of của calibrate_confidence.py (dev_fraction 0,5). Ảnh KIỂM TRA không được dùng,
để số đo trên phần kiểm tra không bị "tự chấm bài mình".

Mỗi cụm 2–5 từ (bỏ số, %, ngoặc) xuất hiện ở ≥ MIN_IMAGES ảnh dev khác nhau được giữ.

Dùng: python scripts_research/build_ocr_lexicon.py  → knowledge_base/ocr_ingredient_lexicon.json
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_research"))

import calibrate_confidence as cc  # noqa: E402
import ocr_regression as reg  # noqa: E402

OUT = ROOT / "knowledge_base" / "ocr_ingredient_lexicon.json"
MIN_IMAGES = 2
DEV_FRACTION = 0.5
_WORD = re.compile(r"[^\W\d_]+")


def items_of(ingredients: str):
    for part in re.split(r"[,;:()\[\]/]|\s-\s|\.\s", unicodedata.normalize("NFC", ingredients)):
        words = _WORD.findall(part)
        if 2 <= len(words) <= 5:
            yield " ".join(w.casefold() for w in words)


def main():
    seen_img, images = set(), defaultdict(set)
    n_dev = 0
    for line in open(reg.BENCHMARK, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if cc.split_of(r, DEV_FRACTION) != "dev" or not str(r.get("product_ingredients") or "").strip():
            continue
        img = reg._resolve_image(r)
        key = str(img or r.get("split_group") or r["id"]).casefold()
        if key in seen_img:
            continue
        seen_img.add(key)
        n_dev += 1
        for phrase in set(items_of(r["product_ingredients"])):
            images[phrase].add(key)
    phrases = sorted((p, len(s)) for p, s in images.items() if len(s) >= MIN_IMAGES)
    OUT.write_text(json.dumps({
        "source": "product_ingredients của ảnh phần PHÁT TRIỂN (split_of, dev_fraction 0,5) trong data/foodhealthqa_eval_set.jsonl",
        "dev_images": n_dev, "min_images": MIN_IMAGES,
        "phrases": [{"phrase": p, "images": n} for p, n in phrases],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{n_dev} ảnh dev · {len(phrases)} cụm (≥{MIN_IMAGES} ảnh) → {OUT}")


if __name__ == "__main__":
    main()
