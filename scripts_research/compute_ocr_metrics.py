"""
Tính Character Error Rate (CER) và Word Error Rate (WER) từ kết quả OCR.

Đầu vào:
  - data/image_ocr_predictions.csv: kết quả OCR (cột raw_ocr_text, normalized_text)
  - data/ocr_ground_truth.csv (tuỳ chọn): ground truth thủ công
    (cột image_id, ground_truth_text, ground_truth_ingredients)

Nếu không có ground truth file, script dùng normalized_text làm "upper bound"
so với raw_ocr_text để đo mức cải thiện nhờ spell correction.

Đầu ra:
  - data/ocr_metrics.json: CER/WER per-image + trung bình
  - In bảng tổng hợp ra console

Chạy: python scripts/compute_ocr_metrics.py
       python scripts/compute_ocr_metrics.py --ground-truth data/ocr_ground_truth.csv
"""

import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.stdout.reconfigure(encoding="utf-8")

from modules.evaluation import compute_cer, compute_wer

DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def _read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_whitespace(text):
    return " ".join(str(text).split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default=os.path.join(DATA_DIR, "image_ocr_predictions.csv"),
        help="OCR predictions CSV",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Ground truth CSV (image_id, ground_truth_text, ground_truth_ingredients)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(DATA_DIR, "ocr_metrics.json"),
        help="Output JSON file",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.predictions):
        print(f"[ERROR] Không tìm thấy predictions: {args.predictions}")
        sys.exit(1)

    predictions = _read_csv(args.predictions)
    print(f"Loaded {len(predictions)} OCR predictions")

    gt_map = {}
    gt_ingredients_map = {}
    has_ground_truth = False

    if args.ground_truth and os.path.isfile(args.ground_truth):
        gt_rows = _read_csv(args.ground_truth)
        for row in gt_rows:
            img_id = row.get("image_id", "").strip()
            if img_id:
                gt_map[img_id] = row.get("ground_truth_text", "").strip()
                gt_ingredients_map[img_id] = row.get(
                    "ground_truth_ingredients", ""
                ).strip()
        has_ground_truth = True
        print(f"Loaded {len(gt_map)} ground truth records")

    results = []
    for pred in predictions:
        image_id = pred.get("image_id", "").strip()
        if not image_id:
            continue

        raw_ocr = _normalize_whitespace(pred.get("raw_ocr_text", ""))
        normalized = _normalize_whitespace(pred.get("normalized_text", ""))
        raw_ingredients = _normalize_whitespace(
            pred.get("raw_ingredients_text", "")
        )
        norm_ingredients = _normalize_whitespace(
            pred.get("normalized_ingredients_text", "")
        )
        confidence = pred.get("ocr_confidence", "")
        ocr_error = pred.get("ocr_error", "").strip()

        entry = {
            "image_id": image_id,
            "ocr_confidence": float(confidence) if confidence else None,
            "has_error": bool(ocr_error),
        }

        if ocr_error:
            entry["cer_raw"] = 1.0
            entry["wer_raw"] = 1.0
            entry["cer_normalized"] = 1.0
            entry["wer_normalized"] = 1.0
            results.append(entry)
            continue

        if has_ground_truth and image_id in gt_map:
            gt_text = gt_map[image_id]
            gt_ingr = gt_ingredients_map.get(image_id, "")

            if gt_text:
                entry["cer_raw"] = compute_cer(gt_text, raw_ocr)
                entry["wer_raw"] = compute_wer(gt_text, raw_ocr)
                entry["cer_normalized"] = compute_cer(gt_text, normalized)
                entry["wer_normalized"] = compute_wer(gt_text, normalized)

            if gt_ingr and raw_ingredients:
                entry["cer_ingredients_raw"] = compute_cer(
                    gt_ingr, raw_ingredients
                )
                entry["wer_ingredients_raw"] = compute_wer(
                    gt_ingr, raw_ingredients
                )
            if gt_ingr and norm_ingredients:
                entry["cer_ingredients_norm"] = compute_cer(
                    gt_ingr, norm_ingredients
                )
                entry["wer_ingredients_norm"] = compute_wer(
                    gt_ingr, norm_ingredients
                )
        else:
            if raw_ocr and normalized:
                entry["cer_spell_correction_diff"] = compute_cer(
                    raw_ocr, normalized
                )
                entry["wer_spell_correction_diff"] = compute_wer(
                    raw_ocr, normalized
                )
            if raw_ingredients and norm_ingredients:
                entry["cer_ingredients_spell_diff"] = compute_cer(
                    raw_ingredients, norm_ingredients
                )
                entry["wer_ingredients_spell_diff"] = compute_wer(
                    raw_ingredients, norm_ingredients
                )

        results.append(entry)

    valid = [r for r in results if not r.get("has_error")]
    errored = [r for r in results if r.get("has_error")]

    def _avg(items, key):
        vals = [r[key] for r in items if key in r]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "total_images": len(results),
        "successful_ocr": len(valid),
        "failed_ocr": len(errored),
        "has_ground_truth": has_ground_truth,
    }

    if has_ground_truth:
        summary["avg_cer_raw"] = _avg(valid, "cer_raw")
        summary["avg_wer_raw"] = _avg(valid, "wer_raw")
        summary["avg_cer_normalized"] = _avg(valid, "cer_normalized")
        summary["avg_wer_normalized"] = _avg(valid, "wer_normalized")
        summary["avg_cer_ingredients_raw"] = _avg(valid, "cer_ingredients_raw")
        summary["avg_wer_ingredients_raw"] = _avg(valid, "wer_ingredients_raw")
        summary["avg_cer_ingredients_norm"] = _avg(valid, "cer_ingredients_norm")
        summary["avg_wer_ingredients_norm"] = _avg(valid, "wer_ingredients_norm")
    else:
        summary["avg_cer_spell_correction_diff"] = _avg(
            valid, "cer_spell_correction_diff"
        )
        summary["avg_wer_spell_correction_diff"] = _avg(
            valid, "wer_spell_correction_diff"
        )
        summary["avg_cer_ingredients_spell_diff"] = _avg(
            valid, "cer_ingredients_spell_diff"
        )
        summary["avg_wer_ingredients_spell_diff"] = _avg(
            valid, "wer_ingredients_spell_diff"
        )

    avg_conf = _avg(valid, "ocr_confidence")
    if avg_conf is not None:
        summary["avg_ocr_confidence"] = avg_conf

    print(f"\n{'='*60}")
    print("OCR METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total images:     {summary['total_images']}")
    print(f"  Successful OCR:   {summary['successful_ocr']}")
    print(f"  Failed OCR:       {summary['failed_ocr']}")
    if avg_conf is not None:
        print(f"  Avg confidence:   {avg_conf:.4f}")

    if has_ground_truth:
        print(f"\n  With Ground Truth (vs manual transcription):")
        for metric in [
            "avg_cer_raw", "avg_wer_raw",
            "avg_cer_normalized", "avg_wer_normalized",
        ]:
            val = summary.get(metric)
            if val is not None:
                print(f"    {metric:>35s}: {val:.4f}")

        print(f"\n  Ingredients only:")
        for metric in [
            "avg_cer_ingredients_raw", "avg_wer_ingredients_raw",
            "avg_cer_ingredients_norm", "avg_wer_ingredients_norm",
        ]:
            val = summary.get(metric)
            if val is not None:
                print(f"    {metric:>35s}: {val:.4f}")

        cer_raw = summary.get("avg_cer_raw")
        cer_norm = summary.get("avg_cer_normalized")
        if cer_raw is not None and cer_norm is not None and cer_raw > 0:
            improvement = (cer_raw - cer_norm) / cer_raw * 100
            print(f"\n  Spell correction CER improvement: {improvement:+.1f}%")
    else:
        print(f"\n  Without Ground Truth (raw vs normalized diff):")
        for metric in [
            "avg_cer_spell_correction_diff",
            "avg_wer_spell_correction_diff",
            "avg_cer_ingredients_spell_diff",
            "avg_wer_ingredients_spell_diff",
        ]:
            val = summary.get(metric)
            if val is not None:
                print(f"    {metric:>40s}: {val:.4f}")
        print(
            "\n  [NOTE] Đây là diff giữa raw_ocr_text và normalized_text."
        )
        print(
            "  Để có CER/WER thực, cần tạo file ground truth:"
        )
        print(
            "  data/ocr_ground_truth.csv (image_id, ground_truth_text, ground_truth_ingredients)"
        )

    output_data = {"summary": summary, "per_image": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ OCR metrics đã lưu: {args.output}")


if __name__ == "__main__":
    main()
