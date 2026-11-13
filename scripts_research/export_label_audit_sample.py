"""Export a stratified sample of label images for a manual check of the label transcriptions.

Every v5 result rests on assets/labels/*.json (the transcription of the package). This script picks images whose
transcription the scored records depend on most and writes a CSV a person can fill in while looking at the photo.

Output: data/label_audit_sample.csv   (and data/label_audit_sample_README.md)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "data" / "foodhealthqa_eval_set.jsonl"
OUT = ROOT / "data" / "label_audit_sample.csv"
README = ROOT / "data" / "label_audit_sample_README.md"

HIGH_STAKES = {"UNSAFE", "CRITICAL_ALERT", "CAUTION"}


def join(v) -> str:
    if isinstance(v, list):
        return " ; ".join(str(x) for x in v)
    return str(v or "")


def what_to_check(recs: list[dict]) -> str:
    items = []
    if any(r["domain"] == "allergen_detection" for r in recs):
        items.append("DỊ NGUYÊN: đọc lại danh sách thành phần và câu 'Có chứa' / 'Có thể chứa' / 'sản xuất trên dây chuyền'")
    if any(r["domain"] == "additive_regulation" for r in recs):
        items.append("PHỤ GIA: mọi mã INS/E trong ngoặc hoặc ngoặc vuông có được chép đủ và đúng số không")
    scopes = sorted({r["label_scope"].split(":", 1)[1] for r in recs if r["label_scope"].startswith("authority_flag:")})
    if scopes:
        items.append("CỜ THEO CƠ QUAN: " + ", ".join(scopes) + " - kiểm tra mã liên quan có trên bao bì không")
    if any(r["domain"] == "insufficient_data" for r in recs):
        items.append("THIẾU DỮ LIỆU: bao bì có thật sự không ghi hàm lượng / không có câu cảnh báo dây chuyền không")
    return " | ".join(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261108)
    args = ap.parse_args()

    rows = [json.loads(l) for l in ACTIVE.read_text(encoding="utf-8").splitlines() if l.strip()]
    scored = [r for r in rows if r["eligible_for_scoring"] == "yes"]
    by_image = defaultdict(list)
    for r in scored:
        by_image[r["image_ref"]].append(r)

    high = [img for img, rs in by_image.items() if any(r["label_v5"] in HIGH_STAKES for r in rs)]
    low = [img for img in by_image if img not in set(high)]
    rng = random.Random(args.seed)
    n_high = round(args.size * 0.7)
    pick = rng.sample(sorted(high), min(n_high, len(high)))
    pick += rng.sample(sorted(low), min(args.size - len(pick), len(low)))
    rng.shuffle(pick)

    out_rows = []
    for k, img in enumerate(pick, 1):
        recs = by_image[img]
        r0 = recs[0]
        label_path = ROOT / r0["label_transcription_path"]
        label = json.loads(label_path.read_text(encoding="utf-8"))
        img_path = ROOT / "assets" / img
        counts = Counter(r["label_v5"] for r in recs)
        out_rows.append({
            "sample_no": k,
            "image_file": str(img_path.relative_to(ROOT)).replace("\\", "/"),
            "image_exists": "yes" if img_path.exists() else "NO",
            "label_json": r0["label_transcription_path"],
            "product_name": label.get("product_name", ""),
            "n_scored_records_depending": len(recs),
            "labels_depending": ", ".join(f"{k2}x{v}" for k2, v in sorted(counts.items())),
            "record_ids": " ".join(r["id"] for r in recs),
            "what_to_check": what_to_check(recs),
            "transcribed_ingredient": join(label.get("ingredient")),
            "transcribed_additive": join(label.get("additive")),
            "transcribed_warning": join(label.get("warning")),
            "transcribed_nutrition": join(
                [f"{n.get('name', '')}: {n.get('value', '')}" for n in (label.get("nutrition") or []) if isinstance(n, dict)]),
            # --- to be filled by the reviewer while looking at the photo ---
            "ingredient_ok (Y/N)": "",
            "additive_codes_ok (Y/N)": "",
            "warning_ok (Y/N)": "",
            "product_name_ok (Y/N)": "",
            "what_is_wrong": "",
            "corrected_text": "",
        })

    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    n_rec = sum(r["n_scored_records_depending"] for r in out_rows)
    README.write_text(f"""# Danh sách mẫu kiểm tra phiên âm nhãn

File: `data/label_audit_sample.csv` ({len(out_rows)} ảnh, chọn ngẫu nhiên có phân tầng, seed {args.seed}).
Mẫu này chi phối **{n_rec}** bản ghi tính điểm trên tổng {len(scored)}; {n_high} ảnh ưu tiên nhãn rủi ro cao
(UNSAFE / CRITICAL_ALERT / CAUTION), phần còn lại là ảnh ngẫu nhiên.

## Cách làm cho mỗi dòng
1. Mở ảnh ở cột `image_file` (thư mục `assets/`).
2. So với các cột `transcribed_*`. Chú ý cột `what_to_check`.
3. Điền Y/N vào 4 cột `*_ok`. Nếu N: ghi vào `what_is_wrong` và chép đoạn đúng vào `corrected_text`.

## Cách đọc kết quả
- Mỗi lỗi ở `ingredient` hoặc `warning` có thể làm sai nhãn dị nguyên của các `record_ids` đi kèm.
- Mỗi mã INS bị chép sai hoặc sót làm sai nhãn phụ gia và nhãn nhóm dễ tổn thương.
- Nếu tỷ lệ ảnh có lỗi vượt khoảng 5%, nên kiểm tra thêm toàn bộ các ảnh có bản ghi UNSAFE/CRITICAL_ALERT.
""", encoding="utf-8")
    print(f"sample: {len(out_rows)} images covering {n_rec} scored records ({len(scored)} total)")
    print("missing images:", [r["image_file"] for r in out_rows if r["image_exists"] == "NO"])
    print("labels:", Counter(l for r in out_rows for l in r["labels_depending"].split(", ")).most_common(8))


if __name__ == "__main__":
    main()
