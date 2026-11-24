"""Kiểm tra hồi quy chất lượng OCR trên ẢNH THẬT của bộ đánh giá (không phải chuỗi chữ dựng sẵn).

Dữ liệu: chữ OCR thô của các ảnh trong bộ đánh giá (bộ đệm data/ocr_cache_config_c.json — tạo bằng
calibrate_confidence.py --stage ocr hoặc khi sinh dự đoán cấu hình C) + danh sách thành phần chuẩn của từng ảnh
(cột product_ingredients). Mỗi lần chạy TÍNH LẠI hậu xử lý → tách thành phần → dò mã phụ gia/dị nguyên từ chữ thô,
nên mọi thay đổi ở các bước đó đều được đo mà không phải OCR lại. Đổi mô hình OCR → bộ đệm tự OCR lại (phiên bản).

Chỉ số (trung bình theo ảnh, trừ khi ghi khác):
  token_recall            tỉ lệ từ của danh sách thành phần chuẩn có trong chữ OCR (sau hậu xử lý)
  item_recall             tỉ lệ thành phần chuẩn mà OCR đọc được ≥ 80% số từ
  raw_token_recall        như token_recall nhưng trên chữ OCR THÔ → postprocess_gain = token_recall − raw_token_recall
                          (âm nghĩa là hậu xử lý làm chữ TỆ hơn — lỗi đã từng xảy ra với từ điển sửa chính tả)
  code_precision/recall   mã phụ gia dò được trên chữ OCR so với mã trên danh sách thành phần chuẩn (tính gộp)
  allergen_precision/recall  nhóm dị nguyên dò được trên phần thành phần + câu cảnh báo so với danh sách chuẩn
  fake_allergens          tổng số nhóm dị nguyên dò được mà danh sách chuẩn không có (dị nguyên giả)
  oversized_items         tổng số "cụm thành phần" dài > 250 ký tự (dấu hiệu tách hỏng, vd mất ngoặc đóng)

Dùng:
  python scripts_research/ocr_regression.py                    # đo và so với mốc chuẩn, in bảng
  python scripts_research/ocr_regression.py --update-baseline  # ghi mốc chuẩn mới (chỉ khi thay đổi đã được duyệt)
  python scripts_research/ocr_regression.py --worst 15         # in 15 ảnh tệ nhất để soi lỗi
Kết quả từng ảnh: data/analysis/ocr_regression_by_image.csv. Mốc chuẩn: data/ocr_regression_baseline.json.
tests/test_ocr_regression.py chạy phép đo này và báo lỗi khi chỉ số tụt quá TOLERANCE so với mốc.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.additive_linker import link_additive_mentions, normalize_ins_key  # noqa: E402
from modules.constraint_model import ALLERGEN_KEYWORDS, contains_label_term  # noqa: E402
from modules.ocr_extractor import OCR_POSTPROCESS_VERSION, parse_ingredients_from_ocr, reprocess_ocr_text  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
CACHE = DATA_DIR / "ocr_cache_config_c.json"
BENCHMARK = DATA_DIR / "foodhealthqa_eval_set.jsonl"
BASELINE = DATA_DIR / "ocr_regression_baseline.json"
PER_IMAGE = DATA_DIR / "analysis" / "ocr_regression_by_image.csv"
OVERSIZED_CHARS = 250

# chỉ số càng CAO càng tốt → không được giảm quá; càng THẤP càng tốt → không được tăng quá
HIGHER_IS_BETTER = ("token_recall", "item_recall", "postprocess_gain", "code_precision", "code_recall",
                    "allergen_precision", "allergen_recall")
LOWER_IS_BETTER = ("fake_allergens", "oversized_items")
TOLERANCE = {  # mức tụt cho phép so với mốc chuẩn (dao động nhỏ do sắp xếp, làm tròn)
    "token_recall": 0.005, "item_recall": 0.005, "postprocess_gain": 0.005,
    "code_precision": 0.01, "code_recall": 0.01, "allergen_precision": 0.01, "allergen_recall": 0.01,
    "fake_allergens": 2, "oversized_items": 1,
}

_WORD = re.compile(r"[^\W_]+")


def _norm(text):
    return unicodedata.normalize("NFC", str(text or "")).casefold()


def _resolve_image(record):
    raw = str(record.get("image_ref") or "").strip().replace("\\", "/")
    if not raw:
        return None
    for candidate in (PROJECT_ROOT / "assets" / raw, PROJECT_ROOT / raw, PROJECT_ROOT / "assets" / "images" / Path(raw).name):
        if candidate.is_file():
            return candidate
    return None


def label_text(ingredients):
    text = str(ingredients or "").strip()
    return text if _norm(text).startswith(("thành phần", "ingredients")) else f"Thành phần: {text}"


def coverage(text, reference):
    ref_tokens = _WORD.findall(_norm(reference))
    if not ref_tokens:
        return None, None
    ocr_tokens = set(_WORD.findall(_norm(text)))
    token_recall = sum(t in ocr_tokens for t in ref_tokens) / len(ref_tokens)
    items = [i for i in re.split(r"[,;()\[\]]", _norm(reference)) if _WORD.findall(i)]
    found = sum(1 for it in items if sum(t in ocr_tokens for t in _WORD.findall(it)) / len(_WORD.findall(it)) >= 0.8)
    return token_recall, (found / len(items) if items else None)


def codes_from_mentions(mentions):
    codes = set()
    for m in mentions:
        for hit in link_additive_mentions(m.get("raw_mention", "")):
            key = normalize_ins_key(hit["matched_entity"].get("ins_code", ""))
            if key:
                codes.add(key)
    return codes


def allergens_in(text):
    t = _norm(text)
    return {a for a, kws in ALLERGEN_KEYWORDS.items() if any(contains_label_term(t, kw) for kw in kws)}


def load_images(cache_path=CACHE, benchmark_path=BENCHMARK):
    """[(tên ảnh, chữ OCR thô, danh sách thành phần chuẩn)] — mỗi ảnh một lần."""
    cache = json.loads(Path(cache_path).read_text(encoding="utf-8")).get("images", {}) if Path(cache_path).is_file() else {}
    seen, out = set(), []
    for line in open(benchmark_path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        img = _resolve_image(r)
        if img is None:
            continue
        key = str(img).casefold()
        if key in seen or key not in cache or not str(r.get("product_ingredients") or "").strip():
            continue
        seen.add(key)
        entry = cache[key]
        if entry.get("is_ocr_unavailable"):
            continue
        out.append((img.name, entry.get("raw_text_before_norm", ""), r["product_ingredients"]))
    return sorted(out)


def evaluate(images):
    rows = []
    tot = {"code_tp": 0, "code_fp": 0, "code_fn": 0, "al_tp": 0, "al_fp": 0, "al_fn": 0, "oversized": 0}
    for name, raw, ref in images:
        full, _, _ = reprocess_ocr_text(raw)
        mentions = parse_ingredients_from_ocr(full, include_allergen_statements=True)
        tok, item = coverage(full, ref)
        raw_tok, _ = coverage(raw, ref)
        ref_mentions = parse_ingredients_from_ocr(label_text(ref), include_allergen_statements=True)
        gold_codes, got_codes = codes_from_mentions(ref_mentions), codes_from_mentions(mentions)
        gold_al = allergens_in(ref)
        got_al = allergens_in(" ".join(m.get("raw_mention", "") for m in mentions))
        oversized = sum(len(m.get("raw_mention", "")) > OVERSIZED_CHARS for m in mentions)
        tot["code_tp"] += len(got_codes & gold_codes); tot["code_fp"] += len(got_codes - gold_codes)
        tot["code_fn"] += len(gold_codes - got_codes)
        tot["al_tp"] += len(got_al & gold_al); tot["al_fp"] += len(got_al - gold_al); tot["al_fn"] += len(gold_al - got_al)
        tot["oversized"] += oversized
        rows.append({
            "image": name, "token_recall": tok, "item_recall": item, "raw_token_recall": raw_tok,
            "codes_gold": " ".join(sorted(gold_codes)), "codes_missed": " ".join(sorted(gold_codes - got_codes)),
            "codes_extra": " ".join(sorted(got_codes - gold_codes)),
            "allergens_gold": " ".join(sorted(gold_al)), "allergens_fake": " ".join(sorted(got_al - gold_al)),
            "allergens_missed": " ".join(sorted(gold_al - got_al)), "mentions": len(mentions), "oversized_items": oversized,
        })

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    def ratio(a, b):
        return a / b if b else None

    summary = {
        "images": len(rows),
        "postprocess_version": OCR_POSTPROCESS_VERSION,
        "token_recall": mean("token_recall"),
        "item_recall": mean("item_recall"),
        "raw_token_recall": mean("raw_token_recall"),
        "code_precision": ratio(tot["code_tp"], tot["code_tp"] + tot["code_fp"]),
        "code_recall": ratio(tot["code_tp"], tot["code_tp"] + tot["code_fn"]),
        "allergen_precision": ratio(tot["al_tp"], tot["al_tp"] + tot["al_fp"]),
        "allergen_recall": ratio(tot["al_tp"], tot["al_tp"] + tot["al_fn"]),
        "fake_allergens": tot["al_fp"],
        "oversized_items": tot["oversized"],
    }
    if summary["token_recall"] is not None and summary["raw_token_recall"] is not None:
        summary["postprocess_gain"] = summary["token_recall"] - summary["raw_token_recall"]
    return summary, rows


def compare(summary, baseline):
    """Danh sách chỉ số tụt quá mức cho phép so với mốc chuẩn."""
    problems = []
    for key in HIGHER_IS_BETTER:
        cur, base = summary.get(key), baseline.get(key)
        if cur is not None and base is not None and cur < base - TOLERANCE[key]:
            problems.append(f"{key}: {cur:.4f} < mốc {base:.4f} (cho phép tụt {TOLERANCE[key]})")
    for key in LOWER_IS_BETTER:
        cur, base = summary.get(key), baseline.get(key)
        if cur is not None and base is not None and cur > base + TOLERANCE[key]:
            problems.append(f"{key}: {cur} > mốc {base} (cho phép tăng {TOLERANCE[key]})")
    return problems


def _fmt(v):
    return "—" if v is None else (f"{v * 100:.1f}%" if isinstance(v, float) and -1.5 <= v <= 1.5 else str(v))


KEY_METRICS = ("code_recall", "allergen_recall")   # hai chỉ số phải TĂNG RÕ mới đổi mô hình
MIN_GAIN = 0.02                                     # tối thiểu +2 điểm phần trăm


def compare_candidate(cache_path):
    """So mô hình OCR ứng viên với hệ thống hiện tại trên đúng các ảnh có ở cả hai bộ đệm.

    Đề xuất ĐỔI chỉ khi: ít nhất một chỉ số chính (KEY_METRICS) tăng ≥ MIN_GAIN, không chỉ số chính nào giảm,
    và không chỉ số nào tụt quá TOLERANCE (vd dị nguyên giả không được tăng quá mức cho phép). Trả về mã thoát
    0 = nên đổi, 1 = giữ nguyên.
    """
    candidate = {name: (raw, ref) for name, raw, ref in load_images(cache_path)}
    current = {name: (raw, ref) for name, raw, ref in load_images()}
    common = sorted(set(candidate) & set(current))
    if not common:
        print(f"Không có ảnh chung giữa {cache_path} và bộ đệm hiện tại.")
        return 1
    cur_sum, _ = evaluate([(n, *current[n]) for n in common])
    cand_sum, cand_rows = evaluate([(n, *candidate[n]) for n in common])
    out_csv = PER_IMAGE.with_name(f"ocr_candidate_{cache_path.stem}_by_image.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cand_rows[0]))
        w.writeheader()
        w.writerows(cand_rows)

    cache_meta = json.loads(cache_path.read_text(encoding="utf-8"))
    det_s = [v.get("det_seconds") for v in cache_meta.get("images", {}).values() if v.get("det_seconds") is not None]
    rec_s = [v.get("rec_seconds") for v in cache_meta.get("images", {}).values() if v.get("rec_seconds") is not None]
    print(f"So trên {len(common)} ảnh chung · ứng viên: {cache_meta.get('pipeline_version', cache_path.name)}")
    print(f"  {'chỉ số':<20} {'hiện tại':>9} {'ứng viên':>9} {'chênh':>8}")
    for key in ("token_recall", "raw_token_recall", "item_recall", "code_precision", "code_recall",
                "allergen_precision", "allergen_recall", "fake_allergens", "oversized_items"):
        a, b = cur_sum.get(key), cand_sum.get(key)
        diff = "" if a is None or b is None else (f"{b - a:+.1%}" if isinstance(a, float) else f"{b - a:+d}")
        print(f"  {key:<20} {_fmt(a):>9} {_fmt(b):>9} {diff:>8}")
    if det_s:
        print(f"  thời gian ứng viên: dò {sum(det_s) / len(det_s):.1f} s/ảnh · đọc {sum(rec_s) / max(len(rec_s), 1):.1f} s/ảnh")
    print(f"Chi tiết từng ảnh: {out_csv}")

    problems = [p for p in compare(cand_sum, cur_sum)]
    gains = {k: (cand_sum[k] or 0) - (cur_sum[k] or 0) for k in KEY_METRICS}
    reasons = []
    if not any(g >= MIN_GAIN for g in gains.values()):
        reasons.append(f"không chỉ số chính nào ({', '.join(KEY_METRICS)}) tăng ≥ {MIN_GAIN:.0%}")
    reasons += [f"{k} giảm {g:+.1%}" for k, g in gains.items() if g < 0]
    reasons += problems
    if reasons:
        print("\n➖ GIỮ NGUYÊN mô hình hiện tại:")
        for r in reasons:
            print("  - " + r)
        return 1
    print("\n✅ NÊN ĐỔI: chỉ số chính tăng rõ, không chỉ số nào tụt quá mức cho phép.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--worst", type=int, default=0, help="In N ảnh có độ phủ từ thấp nhất")
    ap.add_argument("--cache", help="bộ đệm OCR của mô hình ỨNG VIÊN (vd từ ocr_candidate.py) → so với hệ thống hiện tại "
                                    "trên cùng tập ảnh, không ghi mốc chuẩn")
    args = ap.parse_args()
    if args.cache:
        sys.exit(compare_candidate(Path(args.cache)))
    images = load_images()
    if not images:
        print("Chưa có chữ OCR cho ảnh nào — chạy: python scripts_research/calibrate_confidence.py --stage ocr")
        sys.exit(2)
    summary, rows = evaluate(images)
    PER_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    with open(PER_IMAGE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.is_file() else None
    print(f"Ảnh: {summary['images']} · phiên bản hậu xử lý: {summary['postprocess_version']}")
    for key in ("token_recall", "raw_token_recall", "postprocess_gain", "item_recall", "code_precision", "code_recall",
                "allergen_precision", "allergen_recall", "fake_allergens", "oversized_items"):
        base = f"  (mốc {_fmt(baseline.get(key))})" if baseline else ""
        print(f"  {key:<20} {_fmt(summary.get(key)):>8}{base}")
    if args.worst:
        print(f"\n{args.worst} ảnh có độ phủ từ thấp nhất:")
        for r in sorted((r for r in rows if r["token_recall"] is not None), key=lambda r: r["token_recall"])[:args.worst]:
            print(f"  {r['image']:<12} phủ từ {r['token_recall']:.0%} · mã bỏ sót [{r['codes_missed']}] · dị nguyên giả [{r['allergens_fake']}]")
    if args.update_baseline:
        BASELINE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nĐã ghi mốc chuẩn → {BASELINE}")
        return
    if baseline:
        problems = compare(summary, baseline)
        if problems:
            print("\n❌ CHẤT LƯỢNG OCR TỤT so với mốc chuẩn:")
            for p in problems:
                print("  - " + p)
            sys.exit(1)
        print("\n✅ Không chỉ số nào tụt quá mức cho phép so với mốc chuẩn.")


if __name__ == "__main__":
    main()
