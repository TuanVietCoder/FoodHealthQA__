"""Phân tích lỗi chuẩn hóa thành phần (yêu cầu 1: ảnh hưởng của lỗi OCR và lỗi chuẩn hóa).

Mã tham chiếu của mỗi câu = mọi mã phụ gia dò được trên VĂN BẢN NHÃN CHUẨN bằng bộ dò định danh
(modules.additive_linker.link_additive_mentions — mã INS/E hoặc tên/tên đồng nghĩa có nguyên văn trên nhãn).
Báo kèm so sánh với cột ins_codes_matched của bộ QA.

Với mỗi mã tham chiếu bị cấu hình C bỏ sót:
  - B tìm được mà C không   → lỗi do OCR / tách thành phần trên chữ OCR
  - B cũng bỏ sót           → lỗi do liên kết (xảy ra cả khi chữ nhãn đúng)
Đồng thời đo: nhãn an toàn của C có khác B nhiều hơn ở câu có lỗi OCR không; số mã OCR bị đánh dấu "không đọc chắc".

Đầu vào: data/predictions_config_{b,c}.jsonl (generate_predictions.py, có trường linked_entities).
Đầu ra:  data/analysis/normalization_errors.json + normalization_errors_by_question.csv

  python scripts_research/analyze_normalization_errors.py [--run-dir data]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.additive_linker import link_additive_mentions, normalize_ins_key  # noqa: E402
from modules.ocr_extractor import parse_ingredients_from_ocr  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "analysis"


def load_jsonl(path):
    path = Path(path)
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if path.is_file() else []


def label_text(record):
    text = str(record.get("product_ingredients") or "").strip()
    if not text:
        return ""
    return text if text.casefold().startswith(("thành phần", "ingredients")) else f"Thành phần: {text}"


def reference_codes(record):
    codes = set()
    for m in parse_ingredients_from_ocr(label_text(record), include_allergen_statements=True):
        for hit in link_additive_mentions(m["raw_mention"]):
            key = normalize_ins_key(hit["matched_entity"].get("ins_code", ""))
            if key:
                codes.add(key)
    return codes


def benchmark_codes(record):
    raw = record.get("ins_codes_matched")
    items = raw if isinstance(raw, list) else str(raw or "").replace(";", ",").split(",")
    return {normalize_ins_key(str(x).strip()) for x in items if str(x).strip()} - {""}


def predicted_codes(prediction):
    return {normalize_ins_key(e.get("ins_code") or "") for e in prediction.get("linked_entities") or []
            if e.get("ins_code")} - {""}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": (2 * p * r / (p + r)) if p and r else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(DATA_DIR))
    ap.add_argument("--benchmark", default=str(DATA_DIR / "foodhealthqa_eval_set.jsonl"))
    args = ap.parse_args()

    refs = {r["id"]: r for r in load_jsonl(args.benchmark)}
    preds = {c: {p["id"]: p for p in load_jsonl(Path(args.run_dir) / f"predictions_config_{c.lower()}.jsonl")
                 if p.get("linked_entities") is not None} for c in ("B", "C")}
    common = sorted(set(preds["B"]) & set(preds["C"]) & set(refs))
    if not common:
        print("Chưa có dự đoán B và C (có trường linked_entities) cho cùng câu hỏi. Chạy generate_predictions.py trước.")
        return

    ref_cache = {}
    totals = {c: Counter() for c in ("B", "C")}
    totals_bench = {c: Counter() for c in ("B", "C")}
    miss_cause = Counter()
    uncertain = Counter()
    label_change = Counter()
    rows = []
    for rid in common:
        rec = refs[rid]
        img = rec.get("image_ref")
        if img not in ref_cache:
            ref_cache[img] = reference_codes(rec)
        gold = ref_cache[img]
        bench = benchmark_codes(rec)
        got = {c: predicted_codes(preds[c][rid]) for c in ("B", "C")}
        for c in ("B", "C"):
            totals[c].update(tp=len(got[c] & gold), fp=len(got[c] - gold), fn=len(gold - got[c]))
            totals_bench[c].update(tp=len(got[c] & bench), fp=len(got[c] - bench), fn=len(bench - got[c]))
        missed_c = gold - got["C"]
        ocr_miss = missed_c & got["B"]
        link_miss = missed_c - got["B"]
        miss_cause.update(ocr=len(ocr_miss), linking=len(link_miss))
        n_unc = sum(1 for e in preds["C"][rid].get("linked_entities") or [] if e.get("category") == "additive_uncertain_code")
        uncertain.update(flags=n_unc, questions_with_flag=int(n_unc > 0))
        b_label, c_label = preds["B"][rid].get("safety_label"), preds["C"][rid].get("safety_label")
        has_ocr_error = bool(ocr_miss)
        label_change[("có lỗi OCR" if has_ocr_error else "không lỗi OCR", "nhãn C≠B" if b_label != c_label else "nhãn C=B")] += 1
        rows.append({
            "id": rid, "image_ref": img, "gold_label": rec.get("gold_safety_label"),
            "label_B": b_label, "label_C": c_label,
            "codes_on_label": " ".join(sorted(gold)), "linked_B": " ".join(sorted(got["B"])),
            "linked_C": " ".join(sorted(got["C"])),
            "missed_C_due_to_OCR": " ".join(sorted(ocr_miss)), "missed_due_to_linking": " ".join(sorted(link_miss)),
            "extra_C_not_on_label": " ".join(sorted(got["C"] - gold)), "uncertain_codes_flagged_C": n_unc,
        })

    def rate(change_key):
        n = label_change[(change_key, "nhãn C≠B")] + label_change[(change_key, "nhãn C=B")]
        return (label_change[(change_key, "nhãn C≠B")] / n) if n else None

    report = {
        "questions": len(common),
        "reference": "mã dò được trên văn bản nhãn chuẩn (bộ dò định danh)",
        "linking_vs_label_codes": {c: prf(**totals[c]) for c in ("B", "C")},
        "linking_vs_benchmark_ins_codes_matched": {c: prf(**totals_bench[c]) for c in ("B", "C")},
        "missed_codes_in_C_by_cause": dict(miss_cause),
        "uncertain_code_flags_in_C": dict(uncertain),
        "label_C_differs_from_B": {
            "khi có mã bị OCR làm mất": rate("có lỗi OCR"),
            "khi không có mã bị OCR làm mất": rate("không lỗi OCR"),
            "counts": {f"{a} / {b}": v for (a, b), v in sorted(label_change.items())},
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "normalization_errors.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(OUT_DIR / "normalization_errors_by_question.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n→ {OUT_DIR / 'normalization_errors.json'} và normalization_errors_by_question.csv")


if __name__ == "__main__":
    main()
