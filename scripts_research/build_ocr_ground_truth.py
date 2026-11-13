"""
Tao data/ocr_ground_truth.csv tu nhan da kiem tay (assets/labels/*.json).

Cot ground_truth_ingredients = danh sach "ingredient" cua nhan, noi bang ", ".
Cot ground_truth_text de TRONG: nhan JSON la du lieu co cau truc, khong phai ban
chep toan bo chu tren bao bi, nen khong dung de do CER/WER toan van.
compute_ocr_metrics.py bo qua CER toan van khi cot nay rong va chi do phan thanh phan.

Chay:
  venv\\Scripts\\python.exe scripts_research\\build_ocr_ground_truth.py
"""

import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_DIR = os.path.join(PROJECT_ROOT, "assets", "labels")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "ocr_ground_truth.csv")


def _ingredients_text(label):
    items = label.get("ingredient") or []
    if isinstance(items, str):
        return " ".join(items.split())
    parts = [" ".join(str(x).split()) for x in items if str(x).strip()]
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", default=LABEL_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = []
    skipped = []
    for name in sorted(os.listdir(args.labels_dir)):
        if not name.endswith(".json"):
            continue
        image_id = os.path.splitext(name)[0]
        with open(os.path.join(args.labels_dir, name), "r", encoding="utf-8") as f:
            label = json.load(f)
        ingredients = _ingredients_text(label)
        if not ingredients:
            skipped.append(image_id)
            continue
        rows.append(
            {
                "image_id": image_id,
                "ground_truth_text": "",
                "ground_truth_ingredients": ingredients,
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "ground_truth_text", "ground_truth_ingredients"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Da ghi {len(rows)} dong -> {args.output}")
    if skipped:
        print(f"Bo qua {len(skipped)} nhan khong co thanh phan: {', '.join(skipped[:20])}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
