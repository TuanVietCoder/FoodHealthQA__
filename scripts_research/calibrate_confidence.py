"""Bước 2 — Kiểm chứng điểm tin cậy và chọn ngưỡng từ chối bằng dữ liệu thật (không gọi LLM).

Ba giai đoạn (chạy lần lượt; mỗi giai đoạn chạy tiếp được nếu bị ngắt):

  ocr      OCR mọi ảnh của bộ đánh giá bằng OCR hiện tại, ghi vào ĐÚNG bộ đệm mà generate_predictions.py
           dùng cho cấu hình C (data/ocr_cache_config_c.json) → lần chạy chính không phải OCR lại.
  rules    Chạy tầng luật (guardrail) cho B (văn bản nhãn chuẩn) và C (văn bản OCR) trên mọi câu, mỗi câu hai lần:
           ngưỡng tắt và ngưỡng luôn kích hoạt. Ngưỡng S_total chỉ có tác dụng khi nhãn không có cảnh báo, nên mỗi
           câu chỉ có hai kết quả khả dĩ → tính được kết quả cho MỌI ngưỡng mà không phải chạy lại.
  analyze  (1) Độ tin cậy OCR (S_perception) có đi cùng lỗi OCR thật không: CER phần thành phần so với văn bản chuẩn.
           (2) Liên kết phụ gia: precision/recall mã INS so với ins_codes_matched, tách B và C.
           (3) Chọn ngưỡng từ chối trên phần PHÁT TRIỂN (chia theo ảnh – split_group), báo kết quả trên phần KIỂM TRA:
               ngưỡng chọn được so với 0,65 hiện tại và không dùng ngưỡng.

Tiêu chí chọn ngưỡng (cố định trước khi xem phần kiểm tra): độ chính xác 5 nhãn cao nhất trên phần phát triển;
hòa thì ưu tiên tỉ lệ bỏ sót nguy hiểm (false-safe) thấp hơn, rồi tỉ lệ trả lời (coverage) cao hơn.

Chạy:
  venv\\Scripts\\python.exe scripts_research\\calibrate_confidence.py --stage ocr        # ~1,5–3 giờ CPU, 376 ảnh
  venv\\Scripts\\python.exe scripts_research\\calibrate_confidence.py --stage rules      # vài phút
  venv\\Scripts\\python.exe scripts_research\\calibrate_confidence.py --stage analyze    # vài giây
  venv\\Scripts\\python.exe scripts_research\\calibrate_confidence.py --stage all

Đầu ra: data/calibration/rule_runs.jsonl, data/calibration/calibration_report.json (+ bảng in ra màn hình).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts_research"))
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import generate_predictions as gp  # noqa: E402  (dùng chung bộ đệm OCR, tài nguyên, văn bản chuẩn)
from modules import pipeline as pl  # noqa: E402
from modules.additive_linker import normalize_ins_key  # noqa: E402
from modules.compare_runner import ingredient_section  # noqa: E402
from modules.evaluation import compute_cer  # noqa: E402
from modules.ocr_extractor import parse_ingredients_from_ocr  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "calibration"
RULE_RUNS = OUT_DIR / "rule_runs.jsonl"
REPORT = OUT_DIR / "calibration_report.json"
LABELS = ["SAFE", "CAUTION", "UNSAFE", "CRITICAL_ALERT", "ABSTAIN"]
DANGER = {"UNSAFE", "CRITICAL_ALERT"}
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30 … 0.95
CURRENT_THRESHOLD = 0.65
# Đổi khi tầng luật/liên kết/hậu xử lý OCR thay đổi → các dòng rule_runs cũ bị bỏ qua và chạy lại.
RULES_VERSION = "rules_v4_label_sodium_sugar_codex_" + gp.OCR_POSTPROCESS_VERSION


# ---------------------------------------------------------------------------
# tiện ích
# ---------------------------------------------------------------------------
def load_benchmark(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def split_of(record, dev_fraction):
    """Chia theo ảnh (split_group): mọi câu của cùng một ảnh nằm cùng một phần, cố định giữa các lần chạy."""
    key = str(record.get("split_group") or record.get("image_ref") or record["id"])
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 1000
    return "dev" if bucket < dev_fraction * 1000 else "test"


def _norm_text(text):
    text = unicodedata.normalize("NFC", str(text or "")).casefold()
    return " ".join(text.split())


def ingredient_cer(ocr_text, reference):
    """CER của phần thành phần: cắt từ văn bản OCR đoạn khớp nhất với danh sách thành phần chuẩn rồi so ký tự."""
    ref = _norm_text(reference)
    if not ref or not str(ocr_text or "").strip():
        return None
    section = _norm_text(ingredient_section(ocr_text, reference))
    return compute_cer(ref, section)


_WORD = __import__("re").compile(r"[^\W_]+")


def ingredient_coverage(ocr_text, reference):
    """Chỉ số KHÔNG phụ thuộc thứ tự đọc (OCR hay đọc sai thứ tự bố cục nhãn, làm CER bị phạt nặng):
    - token_recall: tỉ lệ từ của danh sách thành phần chuẩn có trong chữ OCR;
    - item_recall: tỉ lệ thành phần (tách theo dấu phẩy/chấm phẩy) mà OCR đọc được ≥ 80% số từ."""
    ref_tokens = _WORD.findall(_norm_text(reference))
    if not ref_tokens or not str(ocr_text or "").strip():
        return None, None
    ocr_tokens = set(_WORD.findall(_norm_text(ocr_text)))
    token_recall = sum(t in ocr_tokens for t in ref_tokens) / len(ref_tokens)
    items = [i for i in __import__("re").split(r"[,;()\[\]]", _norm_text(reference)) if _WORD.findall(i)]
    found = 0
    for item in items:
        toks = _WORD.findall(item)
        if sum(t in ocr_tokens for t in toks) / len(toks) >= 0.8:
            found += 1
    return token_recall, (found / len(items) if items else None)


def ins_set(value):
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value or "").replace(";", ",").split(",")
    keys = {normalize_ins_key(str(x).strip()) for x in items if str(x).strip()}
    keys.discard("")
    return keys


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy) if vx and vy else None


# ---------------------------------------------------------------------------
# giai đoạn 1: OCR
# ---------------------------------------------------------------------------
def stage_ocr(records, cache_path):
    cache = gp._load_ocr_cache(cache_path)
    seen, todo = set(), []
    for r in records:
        img = gp._resolve_image(r)
        if img is None:
            continue
        key = str(img).casefold()
        if key in seen:
            continue
        seen.add(key)
        if key not in cache:
            todo.append(r)
    print(f"[ocr] ảnh trong bộ đánh giá: {len(seen)} · đã có trong bộ đệm: {len(seen) - len(todo)} · cần OCR: {len(todo)}")
    t0 = time.perf_counter()
    for i, r in enumerate(todo, 1):
        t = time.perf_counter()
        result, err = gp._ocr_input(r, cache, cache_path)  # ghi bộ đệm sau mỗi ảnh
        done = time.perf_counter() - t0
        eta = done / i * (len(todo) - i)
        conf = (result or {}).get("avg_confidence")
        print(f"[ocr] {i}/{len(todo)} {gp._resolve_image(r).name} conf={conf} lỗi={err or '-'} "
              f"({time.perf_counter() - t:.0f}s, còn ~{eta / 60:.0f} phút)", flush=True)


# ---------------------------------------------------------------------------
# giai đoạn 2: tầng luật, ngưỡng tắt / bật
# ---------------------------------------------------------------------------
def _run_rules(record, resources, mentions, conf, threshold, label_text=""):
    pl.CONFIDENCE_ABSTAIN_THRESHOLD = threshold
    persona = pl.build_persona_from_benchmark_record(record, resources["personas"])
    g = pl.run_grounded_pipeline(
        mentions, conf, str(record.get("product_name") or ""), persona, str(record.get("question") or ""),
        resources["kb_additives"], resources["hazards"], resources["fndds"], resources["off_products"],
        generate_answer=False, passage_retrieval_enabled=False, label_text=label_text,
    )
    return g


def stage_rules(records, cache_path, resources):
    cache = gp._load_ocr_cache(cache_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if RULE_RUNS.is_file():
        for l in open(RULE_RUNS, encoding="utf-8"):
            if l.strip():
                row = json.loads(l)
                if row.get("rules_version") == RULES_VERSION:
                    done_ids.add((row["id"], row["config"]))
    missing_ocr = 0
    with open(RULE_RUNS, "a", encoding="utf-8") as out:
        for i, r in enumerate(records, 1):
            gt_text = gp._ground_truth_text(r)
            for config in ("B", "C"):
                if (r["id"], config) in done_ids:
                    continue
                if config == "B":
                    if not gt_text:
                        continue
                    text, conf, cer, ocr_ver = gt_text, 1.0, 0.0, None
                    tok_rec, item_rec = 1.0, 1.0
                else:
                    img = gp._resolve_image(r)
                    entry = cache.get(str(img).casefold()) if img else None
                    if entry is not None:
                        gp.refresh_postprocess(entry)  # tính lại hậu xử lý theo phiên bản mới từ chữ OCR thô
                    if entry is None:
                        missing_ocr += 1
                        continue
                    if entry.get("is_ocr_unavailable"):
                        text, conf = "", 0.0
                    else:
                        text, conf = entry.get("full_text", ""), float(entry.get("avg_confidence") or 0.0)
                    cer = ingredient_cer(text, r.get("product_ingredients") or "")
                    tok_rec, item_rec = ingredient_coverage(text, r.get("product_ingredients") or "")
                    ocr_ver = gp.OCR_PIPELINE_VERSION
                mentions = parse_ingredients_from_ocr(text, include_allergen_statements=True)
                off = _run_rules(r, resources, mentions, conf, threshold=-1.0, label_text=text)  # ngưỡng không kích hoạt
                on = _run_rules(r, resources, mentions, conf, threshold=2.0, label_text=text)    # ngưỡng luôn kích hoạt
                ev = off["evaluation"]
                linked = {normalize_ins_key((it.get("matched_entity") or {}).get("ins_code", ""))
                          for it in off["linked_results"] if it.get("matched_entity")}
                linked.discard("")
                row = {
                    "id": r["id"], "config": config, "split": None, "ocr_pipeline_version": ocr_ver,
                    "rules_version": RULES_VERSION,
                    "image_ref": str(r.get("image_ref") or ""),
                    "gold_label": r.get("gold_safety_label"), "domain": r.get("domain"),
                    "abstain_reason_gold": r.get("abstain_reason") or "",
                    "s_perception": ev.get("s_perception"), "s_retrieval": ev.get("s_retrieval"),
                    "s_total": ev.get("s_total"),
                    "additive_total": ev.get("additive_mentions_total"), "additive_linked": ev.get("additive_mentions_linked"),
                    "label_threshold_off": off["safety_label"], "label_threshold_on": on["safety_label"],
                    "status_threshold_off": ev.get("status"),
                    "abstention_reasons_off": ev.get("abstention_reasons") or [],
                    "ingredient_cer": cer,
                    "ingredient_token_recall": tok_rec,
                    "ingredient_item_recall": item_rec,
                    "linked_ins": sorted(linked), "gold_ins": sorted(ins_set(r.get("ins_codes_matched"))),
                    "n_mentions": len(mentions),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            if i % 50 == 0:
                print(f"[rules] {i}/{len(records)}", flush=True)
    pl.CONFIDENCE_ABSTAIN_THRESHOLD = CURRENT_THRESHOLD
    if missing_ocr:
        print(f"[rules] {missing_ocr} câu C chưa có OCR trong bộ đệm → bỏ qua; chạy --stage ocr trước.")


# ---------------------------------------------------------------------------
# giai đoạn 3: phân tích
# ---------------------------------------------------------------------------
def label_at(row, t):
    return row["label_threshold_on"] if (row["s_total"] is not None and row["s_total"] < t) else row["label_threshold_off"]


def selective_metrics(rows, t):
    n = len(rows)
    if not n:
        return None
    preds = [label_at(r, t) for r in rows]
    golds = [r["gold_label"] for r in rows]
    correct = sum(p == g for p, g in zip(preds, golds))
    answered = [(p, g) for p, g in zip(preds, golds) if p != "ABSTAIN"]
    danger = [(p, g) for p, g in zip(preds, golds) if g in DANGER]
    gold_abs = [(p, g) for p, g in zip(preds, golds) if g == "ABSTAIN"]
    pred_abs = [(p, g) for p, g in zip(preds, golds) if p == "ABSTAIN"]
    return {
        "threshold": t,
        "n": n,
        "accuracy": correct / n,
        "coverage": len(answered) / n,
        "selective_accuracy": (sum(p == g for p, g in answered) / len(answered)) if answered else None,
        "false_safe_rate": (sum(p == "SAFE" for p, g in danger) / len(danger)) if danger else None,
        "danger_recall": (sum(p in DANGER for p, g in danger) / len(danger)) if danger else None,
        "abstain_recall": (sum(p == "ABSTAIN" for p, g in gold_abs) / len(gold_abs)) if gold_abs else None,
        "abstain_precision": (sum(g == "ABSTAIN" for p, g in pred_abs) / len(pred_abs)) if pred_abs else None,
        "abstain_count": len(pred_abs),
    }


def choose_threshold(dev_rows):
    """Độ chính xác cao nhất; hòa → false-safe thấp hơn; rồi coverage cao hơn. Thêm 'tắt ngưỡng' (t=0)."""
    cands = [0.0] + THRESHOLDS
    scored = [selective_metrics(dev_rows, t) for t in cands]
    best = max(scored, key=lambda m: (round(m["accuracy"], 6), -(m["false_safe_rate"] or 0.0), m["coverage"]))
    return best["threshold"], scored


def ocr_calibration(rows_c):
    by_img = {}  # mỗi ảnh một điểm (nhiều câu hỏi dùng chung một ảnh)
    for r in rows_c:
        if r.get("ingredient_token_recall") is not None:
            by_img.setdefault(r.get("image_ref") or r["id"], r)
    imgs = list(by_img.values())
    if not imgs:
        return None
    sp = [r["s_perception"] for r in imgs]
    metrics = {
        "token_miss": [1 - r["ingredient_token_recall"] for r in imgs],
        "item_miss": [1 - r["ingredient_item_recall"] if r.get("ingredient_item_recall") is not None else None for r in imgs],
        "cer_reading_order": [r["ingredient_cer"] for r in imgs],
    }
    bins = []
    edges = [0.0, 0.80, 0.85, 0.90, 0.95, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in imgs if lo <= r["s_perception"] < hi]
        if sel:
            def mean(key, inv=False):
                vals = [(1 - r[key]) if inv else r[key] for r in sel if r.get(key) is not None]
                return sum(vals) / len(vals) if vals else None
            bins.append({"s_perception": f"[{lo:.2f}, {min(hi, 1.0):.2f})", "images": len(sel),
                         "mean_token_miss": mean("ingredient_token_recall", inv=True),
                         "mean_item_miss": mean("ingredient_item_recall", inv=True),
                         "mean_cer_reading_order": mean("ingredient_cer")})
    return {
        "images": len(imgs),
        "spearman_sperception_vs": {k: spearman(sp, v) for k, v in metrics.items()},
        "bins": bins,
        "interpretation": "rho âm rõ (≤ -0,3): điểm OCR thấp đi cùng lỗi OCR cao → dùng được để từ chối; "
                          "gần 0: điểm không phản ánh lỗi. token/item_miss không phụ thuộc thứ tự đọc; "
                          "cer_reading_order bị phạt nặng khi OCR đọc sai thứ tự bố cục.",
    }


def linking_metrics(rows, ref_rows=None):
    """Recall so với mã trong câu hỏi (gold_ins ⊆ mã trên nhãn, nên recall hợp lệ).
    Precision KHÔNG tính theo gold_ins: gold chỉ gồm mã được hỏi, còn linked gồm mọi mã trên nhãn,
    nên mã đúng nhưng không được hỏi bị đếm nhầm là FP. Với cấu hình C, precision/recall/F1 được tính so với
    các mã liên kết từ nhãn tham chiếu (cấu hình B, cùng bộ liên kết) → phần chênh là lỗi do OCR."""
    tp = fn = 0
    for r in rows:
        gold, pred = set(r["gold_ins"]), set(r["linked_ins"])
        tp += len(gold & pred); fn += len(gold - pred)
    rec = tp / (tp + fn) if tp + fn else None
    out = {"tp": tp, "fn": fn, "recall": rec, "precision": None, "f1": None,
           "precision_note": "không áp dụng: gold_ins chỉ gồm mã được hỏi, không phải mọi mã trên nhãn",
           "records_with_gold_ins": sum(1 for r in rows if r["gold_ins"])}
    if ref_rows is not None:
        ref = {r["id"]: set(r["linked_ins"]) for r in ref_rows}
        t = f = m = 0
        for r in rows:
            if r["id"] not in ref:
                continue
            pred, gold = set(r["linked_ins"]), ref[r["id"]]
            t += len(pred & gold); f += len(pred - gold); m += len(gold - pred)
        p = t / (t + f) if t + f else None
        q = t / (t + m) if t + m else None
        out["vs_reference_label"] = {"tp": t, "fp": f, "fn": m, "precision": p, "recall": q,
                                     "f1": 2 * p * q / (p + q) if p and q else None}
    return out


def stage_analyze(records, dev_fraction):
    rows = [json.loads(l) for l in open(RULE_RUNS, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("rules_version") == RULES_VERSION]
    # mỗi (câu, cấu hình) chỉ giữ dòng mới nhất — các lần chạy bị ngắt rồi chạy tiếp có thể ghi trùng
    rows = list({(r["id"], r["config"]): r for r in rows}.values())
    split = {r["id"]: split_of(r, dev_fraction) for r in records}
    for r in rows:
        r["split"] = split.get(r["id"], "test")
    report = {"ocr_pipeline_version": gp.OCR_PIPELINE_VERSION, "dev_fraction": dev_fraction,
              "selection_rule": "max dev accuracy (5 nhãn); hòa → false-safe thấp hơn → coverage cao hơn",
              "current_threshold": CURRENT_THRESHOLD, "configs": {}}
    for config in ("B", "C"):
        cr = [r for r in rows if r["config"] == config]
        if not cr:
            continue
        dev = [r for r in cr if r["split"] == "dev"]
        test = [r for r in cr if r["split"] == "test"]
        chosen, dev_curve = choose_threshold(dev)
        kb_low = sum(any("không tìm thấy đủ thực thể" in x for x in r["abstention_reasons_off"]) for r in cr)
        report["configs"][config] = {
            "records": len(cr), "n_dev": len(dev), "n_test": len(test),
            "s_retrieval_not_applicable": sum(r["s_retrieval"] is None for r in cr),
            "s_total_below_current": sum((r["s_total"] or 0) < CURRENT_THRESHOLD for r in cr),
            "abstain_by_kb_coverage_rule_50pct": kb_low,
            "chosen_threshold": chosen,
            "dev_curve": dev_curve,
            "test": {
                "no_threshold": selective_metrics(test, 0.0),
                "current_0.65": selective_metrics(test, CURRENT_THRESHOLD),
                "chosen": selective_metrics(test, chosen),
            },
            "linking_vs_gold_ins": linking_metrics(cr, ref_rows=[r for r in rows if r["config"] == "B"] if config == "C" else None),
        }
        if config == "C":
            report["configs"][config]["ocr_confidence_vs_cer"] = ocr_calibration(cr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(report)
    print(f"\n→ {REPORT}")


def _f(x, pct=False):
    if x is None:
        return "  —  "
    return f"{x * 100:5.1f}%" if pct else f"{x:.3f}"


def _print_report(rep):
    for config, c in rep["configs"].items():
        print(f"\n==================== Cấu hình {config} ({c['records']} câu: dev {c['n_dev']} / test {c['n_test']}) ====================")
        print(f"S_retrieval không áp dụng: {c['s_retrieval_not_applicable']} · S_total < 0,65: {c['s_total_below_current']} "
              f"· từ chối do luật '>50% phụ gia không liên kết': {c['abstain_by_kb_coverage_rule_50pct']}")
        lk = c["linking_vs_gold_ins"]
        print(f"Liên kết INS so với mã được hỏi: recall {_f(lk['recall'], True)} (TP {lk['tp']}, FN {lk['fn']}); "
              f"precision không áp dụng (gold chỉ gồm mã được hỏi)")
        if lk.get("vs_reference_label"):
            v = lk["vs_reference_label"]
            print(f"   So với nhãn tham chiếu (B): precision {_f(v['precision'], True)} · recall {_f(v['recall'], True)} "
                  f"· F1 {_f(v['f1'], True)} (TP {v['tp']}, FP {v['fp']}, FN {v['fn']})")
        if c.get("ocr_confidence_vs_cer"):
            oc = c["ocr_confidence_vs_cer"]
            rho = oc["spearman_sperception_vs"]
            print(f"S_perception ↔ lỗi OCR ({oc['images']} ảnh), Spearman: thiếu từ {_f(rho['token_miss'])} · "
                  f"thiếu thành phần {_f(rho['item_miss'])} · CER theo thứ tự đọc {_f(rho['cer_reading_order'])}")
            for b in oc["bins"]:
                print(f"   S_perception {b['s_perception']}: {b['images']:3d} ảnh · thiếu từ {_f(b['mean_token_miss'], True)} · "
                      f"thiếu thành phần {_f(b['mean_item_miss'], True)} · CER {_f(b['mean_cer_reading_order'], True)}")
        print("Đường cong trên phần PHÁT TRIỂN (ngưỡng: accuracy · coverage · selective acc · false-safe · số từ chối):")
        for m in c["dev_curve"]:
            mark = " ← chọn" if m["threshold"] == c["chosen_threshold"] else (" ← hiện tại" if m["threshold"] == CURRENT_THRESHOLD else "")
            print(f"   t={m['threshold']:.2f}: {_f(m['accuracy'], True)} · {_f(m['coverage'], True)} · "
                  f"{_f(m['selective_accuracy'], True)} · {_f(m['false_safe_rate'], True)} · {m['abstain_count']}{mark}")
        print("Phần KIỂM TRA:")
        for name, m in c["test"].items():
            print(f"   {name:<13} t={m['threshold']:.2f}: accuracy {_f(m['accuracy'], True)} · coverage {_f(m['coverage'], True)} · "
                  f"selective {_f(m['selective_accuracy'], True)} · false-safe {_f(m['false_safe_rate'], True)} · "
                  f"abstain recall {_f(m['abstain_recall'], True)} · precision {_f(m['abstain_precision'], True)}")


def main():
    global OUT_DIR, RULE_RUNS, REPORT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["ocr", "rules", "analyze", "all"], default="all")
    ap.add_argument("--benchmark", default=str(DATA_DIR / "foodhealthqa_eval_set.jsonl"))
    ap.add_argument("--ocr-cache", default=str(DATA_DIR / "ocr_cache_config_c.json"))
    ap.add_argument("--limit", type=int, default=0, help="Chỉ lấy N câu đầu (thử nghiệm)")
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="Thư mục kết quả (mặc định data/calibration)")
    ap.add_argument("--dev-fraction", type=float, default=0.5, help="Tỉ lệ ảnh dùng để chọn ngưỡng")
    args = ap.parse_args()
    OUT_DIR = Path(args.out_dir)
    RULE_RUNS, REPORT = OUT_DIR / "rule_runs.jsonl", OUT_DIR / "calibration_report.json"
    records = load_benchmark(args.benchmark)
    if args.limit:
        records = records[: args.limit]
    cache_path = Path(args.ocr_cache)
    if args.stage in ("ocr", "all"):
        stage_ocr(records, cache_path)
    if args.stage in ("rules", "all"):
        stage_rules(records, cache_path, gp._load_resources())
    if args.stage in ("analyze", "all"):
        stage_analyze(records, args.dev_fraction)


if __name__ == "__main__":
    main()
