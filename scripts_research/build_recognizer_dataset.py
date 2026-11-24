"""Dựng tập huấn luyện bộ nhận dạng VietOCR (crop dòng chữ, văn bản) từ ảnh + nhãn ngữ nghĩa assets/labels.

Nhãn trong assets/labels KHÔNG có bounding box hay chữ từng dòng, nên cặp (crop, text) được tạo bằng căn chỉnh:
  1. PaddleOCR DBNet dò vùng chữ trên ảnh, cắt từng vùng (đúng hàm cắt của pipeline).
  2. VietOCR hiện tại đọc từng vùng → chữ dự đoán (có thể sai dấu / sai mã INS).
  3. Chữ dự đoán được căn (rapidfuzz partial_ratio_alignment) vào văn bản chuẩn của chính ảnh đó
     (product_name, ingredient, additive, warning, origin, nutrition...). Đoạn chuẩn khớp ≥ --min-score
     (cắt theo ranh giới từ) trở thành nhãn của crop. Crop không khớp bị bỏ (không đoán nhãn).

CHỐNG RÒ RỈ: chỉ dùng ảnh KHÔNG thuộc phần kiểm tra (test) của bộ đánh giá 750 câu. Phần test được định nghĩa bằng đúng
split_of(dev_fraction=0.5) của calibrate_confidence.py. Ảnh ngoài bộ đánh giá luôn được dùng. Phần val (để chọn checkpoint)
được tách theo ảnh từ tập huấn luyện.

Dùng:  python scripts_research/build_recognizer_dataset.py [--limit N] [--min-score 90]
Ra:    data/ocr_train/{img/*.png, train.txt, val.txt, report.json}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_research"))

import cv2  # noqa: E402
from PIL import Image  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

import calibrate_confidence as cc  # noqa: E402
from modules import ocr_extractor as oe  # noqa: E402

ASSETS = ROOT / "assets"
EVAL_SET = ROOT / "data" / "foodhealthqa_eval_set.jsonl"
OUT = ROOT / "data" / "ocr_train"
DEV_FRACTION = 0.5
VAL_FRACTION = 0.1
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def nfc(s):
    return unicodedata.normalize("NFC", str(s or "")).strip()


def test_stems():
    """Mã ảnh thuộc phần KIỂM TRA của bộ đánh giá — tuyệt đối không dùng để huấn luyện."""
    out = set()
    for line in open(EVAL_SET, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if cc.split_of(r, DEV_FRACTION) == "test":
            out.add(Path(str(r.get("image_ref") or "")).stem)
    return out


def label_segments(label):
    """Các đoạn văn bản chuẩn của một ảnh (giữ nguyên hoa/thường, dấu)."""
    segs = [label.get("product_name")]
    ingredients = [nfc(x) for x in (label.get("ingredient") or []) if nfc(x)]
    segs += ingredients
    if ingredients:
        segs.append(", ".join(ingredients))  # nhãn thật thường in thành một đoạn liền, dấu phẩy
    for key in ("additive", "warning"):
        segs += [x for x in (label.get(key) or []) if isinstance(x, str)]
    for key in ("origin", "net_weight", "mfg_date", "expiry_date"):
        segs.append(label.get(key))
    for n in label.get("nutrition") or []:
        if isinstance(n, dict):
            segs += [n.get("name"), n.get("value"), f"{n.get('name', '')} {n.get('value', '')}"]
    segs = [nfc(s) for s in segs if s and nfc(s)]
    return list(dict.fromkeys(segs))


_WORDCH = re.compile(r"\w", re.UNICODE)


def snap(text, start, end):
    """Mở rộng [start, end) tới ranh giới từ để không cắt giữa chữ."""
    while start > 0 and _WORDCH.match(text[start - 1]) and _WORDCH.match(text[start]):
        start -= 1
    while end < len(text) and _WORDCH.match(text[end - 1]) and _WORDCH.match(text[end]):
        end += 1
    return start, end


def align(pred, segments, min_score):
    """(đoạn chuẩn, điểm) tốt nhất cho chữ dự đoán, hoặc (None, điểm)."""
    p = pred.lower()
    best, best_score = None, 0.0
    for seg in segments:
        s = seg.lower()
        if len(s) != len(seg):  # lower() đổi độ dài → bỏ qua đoạn này để chỉ số còn đúng
            continue
        a = fuzz.partial_ratio_alignment(p, s)
        if a is None or a.score < best_score:
            continue
        st, en = snap(seg, a.dest_start, a.dest_end)
        cand = seg[st:en].strip(" ,;")
        if not cand:
            continue
        ratio = len(cand) / max(len(pred), 1)
        if not 0.7 <= ratio <= 1.4:
            continue
        score = fuzz.ratio(p, cand.lower())  # điểm trên đoạn đã cắt, không phải trên đoạn dài
        if score > best_score:
            best, best_score = cand, score
    return (best, best_score) if best_score >= min_score else (None, best_score)


def is_val(stem):
    return int(hashlib.sha1(stem.encode()).hexdigest(), 16) % 1000 < VAL_FRACTION * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N ảnh đầu (thử nghiệm nhanh)")
    ap.add_argument("--min-score", type=float, default=90.0, help="điểm khớp tối thiểu (0-100)")
    ap.add_argument("--min-len", type=int, default=6, help="độ dài chữ dự đoán tối thiểu để căn")
    args = ap.parse_args()

    banned = test_stems()
    images = sorted(p for p in (ASSETS / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    images = [p for p in images if p.stem not in banned and (ASSETS / "labels" / f"{p.stem}.json").is_file()]
    if args.limit:
        images = images[: args.limit]
    print(f"Ảnh dùng huấn luyện: {len(images)} (loại {len(banned)} ảnh thuộc phần kiểm tra)")

    engine = oe.get_ocr_engine()
    predictor = oe.get_vietocr_predictor()
    if engine is None or predictor is None:
        sys.exit("Cần PaddleOCR + VietOCR để dựng dữ liệu (xem README).")

    (OUT / "img").mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "val": []}
    stats = Counter()
    for n, path in enumerate(images, 1):
        label = json.loads((ASSETS / "labels" / f"{path.stem}.json").read_text(encoding="utf-8"))
        segments = label_segments(label)
        img = oe.convert_to_cv2_image(str(path))
        boxes = oe._sort_boxes_reading_order(oe._detect_boxes_paddle_only(engine, img))
        crops, keep = [], []
        for box, _ in boxes:
            crop = oe._crop_polygon_region(img, box)
            if crop is None or crop.size == 0:
                continue
            crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        if not crops:
            continue
        texts = []
        for i in range(0, len(crops), oe.VIETOCR_BATCH_SIZE):
            t, _ = oe._vietocr_read_batch(predictor, crops[i : i + oe.VIETOCR_BATCH_SIZE])
            texts += t
        split = "val" if is_val(path.stem) else "train"
        for j, (crop, pred) in enumerate(zip(crops, texts)):
            pred = nfc(pred)
            stats["crops"] += 1
            if len(pred) < args.min_len:
                stats["too_short"] += 1
                continue
            target, score = align(pred, segments, args.min_score)
            if target is None:
                stats["unmatched"] += 1
                continue
            dist = Levenshtein.distance(pred, target)
            stats["matched"] += 1
            stats["changed" if dist else "identical"] += 1
            stats["edit_chars"] += dist
            stats["target_chars"] += len(target)
            rel = f"img/{path.stem}_{j:03d}.png"
            crop.save(OUT / rel)
            rows[split].append(f"{rel}\t{target}")
        if n % 20 == 0:
            print(f"  {n}/{len(images)} ảnh, {stats['matched']} mẫu khớp / {stats['crops']} crop")

    for split, lines in rows.items():
        (OUT / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "images": len(images), "train": len(rows["train"]), "val": len(rows["val"]),
        "min_score": args.min_score, "excluded_test_images": len(banned), **stats,
        "baseline_cer_vs_target": round(stats["edit_chars"] / max(stats["target_chars"], 1), 4),
        "note": "baseline_cer_vs_target = CER của VietOCR hiện tại trên các mẫu khớp (nhãn là bản căn chỉnh, không phải gold)",
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
