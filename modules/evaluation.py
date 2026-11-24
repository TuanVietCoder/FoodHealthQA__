"""
Module evaluation chính thức cho hệ thống FoodHealthQA.

Hợp nhất toàn bộ logic đánh giá:
  - Exact Match (EM) & Token F1
  - Citation Accuracy & Coverage
  - Groundedness (human-adjudicated hoặc heuristic)
  - Faithfulness (human-adjudicated)
  - Safety Score (accuracy + unsafe recall + abstention)
  - Confusion Matrix cho safety labels
  - Per-domain breakdown
  - OCR metrics: Character Error Rate (CER) & Word Error Rate (WER)
"""

import json
from collections import Counter, defaultdict

from modules.benchmark import (
    _normalize_text,
    _token_f1,
    analyze_answer_citations,
    canonicalize_citation,
    valid_predicted_sources,
    _SAFETY_LABELS,
)

SAFETY_LABEL_ORDER = ["SAFE", "CAUTION", "WARNING", "UNSAFE", "CRITICAL_ALERT", "ABSTAIN"]
# Nhãn dự đoán không thuộc tập nhãn hợp lệ được đếm riêng (trước đây bị quy thành SAFE trong ma trận nhầm lẫn).
INVALID_LABEL = "INVALID"
DANGER_LABELS = ("WARNING", "UNSAFE", "CRITICAL_ALERT")


# ---------------------------------------------------------------------------
# Edit-distance & OCR metrics (CER / WER)
# ---------------------------------------------------------------------------

def _edit_distance(ref, hyp):
    """Levenshtein edit distance."""
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def compute_cer(reference, hypothesis):
    """Character Error Rate bằng edit distance ở mức ký tự."""
    ref = list(reference)
    hyp = list(hypothesis)
    return _edit_distance(ref, hyp) / max(len(ref), 1)


def compute_wer(reference, hypothesis):
    """Word Error Rate bằng edit distance ở mức từ."""
    ref = reference.split()
    hyp = hypothesis.split()
    return _edit_distance(ref, hyp) / max(len(ref), 1)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix(gold_labels, predicted_labels):
    """
    Tạo confusion matrix cho safety labels.
    Trả về dict: {gold_label: {predicted_label: count}}.
    """
    labels = SAFETY_LABEL_ORDER
    matrix = {g: {p: 0 for p in labels + [INVALID_LABEL]} for g in labels}
    for gold, pred in zip(gold_labels, predicted_labels):
        if gold in labels:
            matrix[gold][pred if pred in labels else INVALID_LABEL] += 1
    return matrix


def format_confusion_matrix(matrix):
    """Định dạng confusion matrix thành bảng text."""
    labels = SAFETY_LABEL_ORDER
    cols = labels + ([INVALID_LABEL] if any(matrix.get(g, {}).get(INVALID_LABEL) for g in labels) else [])
    header = f"{'':>16s}" + "".join(f"{label:>16s}" for label in cols)
    rows = [header]
    for gold in labels:
        row = f"{gold:>16s}"
        for pred in cols:
            row += f"{matrix[gold].get(pred, 0):>16d}"
        rows.append(row)
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Heuristic groundedness
# ---------------------------------------------------------------------------

def _heuristic_groundedness(answer, evidence_excerpt):
    """
    Groundedness theo trùng từ (lexical): tỉ lệ từ của câu trả lời có mặt trong bằng chứng.
    Đây là ƯỚC LƯỢNG bề mặt, không phải người chấm hay mô hình NLI — báo cáo dưới tên "groundedness (trùng từ)".
    """
    if not answer or not evidence_excerpt:
        return 0.0
    answer_tokens = set(_normalize_text(answer).split())
    evidence_tokens = set(_normalize_text(evidence_excerpt).split())
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & evidence_tokens
    return len(overlap) / len(answer_tokens)


# ---------------------------------------------------------------------------
# Per-item evaluation
# ---------------------------------------------------------------------------

def prediction_evidence(prediction, reference):
    """Bằng chứng mà mô hình THỰC SỰ được đọc khi trả lời: văn bản nhãn đầu vào (B: nhãn chuẩn, C: chữ OCR,
    A: nhãn chuẩn) + các đoạn nguồn đã truy xuất (lưu trong dự đoán). Không dùng riêng danh sách thành phần làm
    "bằng chứng" cho các nhận định về quy định/sức khỏe nữa."""
    parts = []
    if reference.get("evidence_excerpt"):
        parts.append(str(reference["evidence_excerpt"]))
    config = str(prediction.get("config") or "")
    if config == "C":
        parts.append(str((prediction.get("ocr") or {}).get("full_text") or ""))
    else:
        parts.append(str(prediction.get("ground_truth_text") or reference.get("product_ingredients") or ""))
    for passage in prediction.get("retrieved_passages") or []:
        parts.append(str(passage.get("text") or ""))
    return "\n".join(part for part in parts if part.strip())


def evaluate_single(prediction, reference):
    """
    Đánh giá một cặp prediction/reference, trả về dict chỉ số chi tiết.
    """
    answer = str(prediction.get("answer", ""))
    gold_answer = str(reference["reference_answer"])

    em = _normalize_text(answer) == _normalize_text(gold_answer)
    f1 = _token_f1(answer, gold_answer)

    raw_sources = reference.get("source_ids") or [
        s.strip()
        for s in str(reference.get("source_citations", "")).split(",")
        if s.strip()
    ]
    gold_sources = {canonicalize_citation(s) for s in raw_sources}
    analysis = analyze_answer_citations(answer)
    predicted_sources, stray_citations = valid_predicted_sources(analysis, gold_sources)

    # Không áp dụng (None) khi mẫu số bằng 0: câu không trích nguồn nào → precision không xác định;
    # câu không có nguồn chuẩn → recall không xác định. Trước đây cả hai bị tính là 0.
    citation_precision = (
        len(predicted_sources & gold_sources) / len(predicted_sources)
        if predicted_sources else None
    )
    citation_recall = (
        len(predicted_sources & gold_sources) / len(gold_sources)
        if gold_sources else None
    )

    gold_safety = reference.get("gold_safety_label") or reference.get("safety_label", "")
    pred_safety = prediction.get("safety_label", "")
    invalid_label = pred_safety not in _SAFETY_LABELS
    # Nhãn dự đoán không hợp lệ là một câu trả lời SAI (không còn bị loại khỏi mẫu số như trước).
    safety_correct = (gold_safety == pred_safety) if gold_safety in _SAFETY_LABELS else None

    grounded = prediction.get("grounded")
    groundedness_method = "human" if isinstance(grounded, bool) else None
    if not isinstance(grounded, bool):
        evidence = prediction_evidence(prediction, reference)
        grounded = _heuristic_groundedness(answer, evidence) if (evidence and answer.strip()) else None
        groundedness_method = "lexical_overlap" if grounded is not None else None

    faithful = prediction.get("faithful")
    if not isinstance(faithful, bool):
        faithful = None

    return {
        "id": reference["id"],
        "domain": reference.get("domain", "unknown"),
        "exact_match": em,
        "token_f1": f1,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "cited": analysis["has_citation"],
        "unrecognized_citations": stray_citations,
        "safety_correct": safety_correct,
        "invalid_label": invalid_label,
        "gold_safety_label": gold_safety,
        "predicted_safety_label": pred_safety,
        "guardrail_status": prediction.get("guardrail_status") or (prediction.get("guardrail") or {}).get("status"),
        "groundedness": grounded,
        "groundedness_method": groundedness_method,
        "faithfulness": faithful,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_metrics(item_results):
    """Tổng hợp từ danh sách kết quả per-item thành metrics tổng thể."""
    n = len(item_results)
    if n == 0:
        return {"count": 0}

    em_scores = [r["exact_match"] for r in item_results]
    f1_scores = [r["token_f1"] for r in item_results]
    cit_prec = [r["citation_precision"] for r in item_results if r.get("citation_precision") is not None]
    cit_rec = [r["citation_recall"] for r in item_results if r.get("citation_recall") is not None]

    safety_items = [r for r in item_results if r["safety_correct"] is not None]
    safety_correct_count = sum(1 for r in safety_items if r["safety_correct"])

    unsafe_tp = sum(
        1 for r in safety_items
        if r["gold_safety_label"] in DANGER_LABELS
        and r["predicted_safety_label"] in DANGER_LABELS
    )
    unsafe_fn = sum(
        1 for r in safety_items
        if r["gold_safety_label"] in DANGER_LABELS
        and r["predicted_safety_label"] not in DANGER_LABELS
    )
    # Nguy hiểm nhất: nhãn chuẩn là nguy hiểm nhưng hệ thống nói SAFE
    false_safe = sum(
        1 for r in safety_items
        if r["gold_safety_label"] in DANGER_LABELS and r["predicted_safety_label"] == "SAFE"
    )

    abstain_pred = sum(
        1 for r in item_results
        if r.get("predicted_safety_label") == "ABSTAIN"
    )
    abstain_gold = sum(
        1 for r in item_results
        if r.get("gold_safety_label") == "ABSTAIN"
    )

    grounded_vals = [
        r["groundedness"] for r in item_results
        if r["groundedness"] is not None
    ]
    faithful_vals = [
        r["faithfulness"] for r in item_results
        if r["faithfulness"] is not None
    ]

    result = {
        "count": n,
        "exact_match": sum(em_scores) / n,
        "token_f1": sum(f1_scores) / n,
        # trung bình trên các câu xác định được (None = không áp dụng); kèm số câu dùng để tính
        "citation_precision": (sum(cit_prec) / len(cit_prec)) if cit_prec else None,
        "citation_precision_n": len(cit_prec),
        "citation_recall": (sum(cit_rec) / len(cit_rec)) if cit_rec else None,
        "citation_recall_n": len(cit_rec),
        # Tuân thủ trích dẫn: tỷ lệ câu trả lời có >=1 thẻ nguồn được công nhận (đáng chú ý ở B/C; A không truy xuất)
        "citation_compliance": sum(1 for r in item_results if r.get("cited")) / n,
        # Thẻ nguồn lạ do mô hình tự bịa: bị loại khỏi precision nhưng đếm riêng ở đây
        "unrecognized_citation_rate": sum(1 for r in item_results if r.get("unrecognized_citations")) / n,
        "unrecognized_tags_per_answer": sum(len(r.get("unrecognized_citations") or []) for r in item_results) / n,
    }

    if safety_items:
        result["safety_accuracy"] = safety_correct_count / len(safety_items)
        danger_n = unsafe_tp + unsafe_fn
        # không có câu nguy hiểm trong nhóm → không áp dụng (trước đây = 0)
        result["unsafe_recall"] = unsafe_tp / danger_n if danger_n else None
        result["false_safe_rate"] = false_safe / danger_n if danger_n else None
        result["danger_gold_n"] = danger_n
        # Điểm gộp 0,6×accuracy + 0,4×recall đã bỏ: trọng số tự chọn, không có căn cứ.
    result["invalid_label_count"] = sum(1 for r in item_results if r.get("invalid_label"))
    # "Chưa phát hiện cảnh báo" (SCREENED_NO_ALERT) được chấm như SAFE; báo riêng số SAFE thuộc loại này
    result["safe_from_screened_no_alert"] = sum(
        1 for r in item_results
        if r.get("predicted_safety_label") == "SAFE" and r.get("guardrail_status") == "SCREENED_NO_ALERT"
    )

    result["abstention_rate_predicted"] = abstain_pred / n
    result["abstention_rate_gold"] = abstain_gold / n

    if grounded_vals:
        result["groundedness"] = sum(
            (v if isinstance(v, (int, float)) else int(v))
            for v in grounded_vals
        ) / len(grounded_vals)
        result["groundedness_n"] = len(grounded_vals)
        methods = {r.get("groundedness_method") for r in item_results if r.get("groundedness") is not None}
        result["groundedness_method"] = "+".join(sorted(m for m in methods if m))

    if faithful_vals:
        result["faithfulness"] = sum(
            (v if isinstance(v, (int, float)) else int(v))
            for v in faithful_vals
        ) / len(faithful_vals)

    return result


# ---------------------------------------------------------------------------
# Config-level evaluation (A / B / C)
# ---------------------------------------------------------------------------

def evaluate_config(references, predictions, config_name="default"):
    """
    Đánh giá đầy đủ 1 cấu hình (A/B/C).
    Trả về: overall metrics, per-domain metrics, confusion matrix, item-level results.
    """
    ref_by_id = {r["id"]: r for r in references}
    pred_by_id = {r["id"]: r for r in predictions}

    common_ids = sorted(set(ref_by_id) & set(pred_by_id))
    missing = sorted(set(ref_by_id) - set(pred_by_id))
    extra = sorted(set(pred_by_id) - set(ref_by_id))

    item_results = []
    gold_labels = []
    pred_labels = []

    for rid in common_ids:
        result = evaluate_single(pred_by_id[rid], ref_by_id[rid])
        item_results.append(result)
        if result["gold_safety_label"] in _SAFETY_LABELS:
            gold_labels.append(result["gold_safety_label"])
            pred_labels.append(
                result["predicted_safety_label"]
                if result["predicted_safety_label"] in _SAFETY_LABELS
                else INVALID_LABEL
            )

    overall = aggregate_metrics(item_results)
    overall["config"] = config_name

    domain_groups = defaultdict(list)
    for r in item_results:
        domain_groups[r["domain"]].append(r)
    per_domain = {
        domain: aggregate_metrics(items)
        for domain, items in sorted(domain_groups.items())
    }

    cm = compute_confusion_matrix(gold_labels, pred_labels)

    return {
        "config": config_name,
        "overall": overall,
        "per_domain": per_domain,
        "confusion_matrix": cm,
        "missing_predictions": missing,
        "extra_predictions": extra,
        "item_results": item_results,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_results_table(config_results_list):
    """Định dạng bảng so sánh nhiều configs (A/B/C) thành text."""
    if not config_results_list:
        return "No results."

    metrics = [
        ("count", "N", "{:d}"),
        ("exact_match", "EM", "{:.3f}"),
        ("token_f1", "Token F1", "{:.3f}"),
        ("citation_precision", "Cit. Prec", "{:.3f}"),
        ("citation_recall", "Cit. Recall", "{:.3f}"),
        ("citation_compliance", "Cit. Comply", "{:.3f}"),
        ("unrecognized_citation_rate", "Fake-tag %", "{:.3f}"),
        ("safety_accuracy", "Safety Acc", "{:.3f}"),
        ("unsafe_recall", "Unsafe Rec", "{:.3f}"),
        ("false_safe_rate", "False-safe", "{:.3f}"),
        ("invalid_label_count", "Invalid lbl", "{:d}"),
        ("safe_from_screened_no_alert", "SAFE=no-alert", "{:d}"),
        ("groundedness", "Grounded(lex)", "{:.3f}"),
        ("abstention_rate_predicted", "Abstain %", "{:.3f}"),
    ]

    header = f"{'Metric':>16s}"
    for cr in config_results_list:
        header += f"  {cr['config']:>14s}"
    lines = [header, "-" * len(header)]

    for key, label, fmt in metrics:
        row = f"{label:>16s}"
        for cr in config_results_list:
            val = cr["overall"].get(key)
            if val is not None:
                if isinstance(val, int):
                    row += f"  {val:>14d}"
                else:
                    row += f"  {fmt.format(val):>14s}"
            else:
                row += f"  {'—':>14s}"
        lines.append(row)

    return "\n".join(lines)
