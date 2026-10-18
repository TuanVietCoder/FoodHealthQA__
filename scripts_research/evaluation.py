"""
Evaluation Pipeline cho hệ thống FoodHealthQA.
Tính toán đầy đủ các chỉ số đánh giá:
  - Exact Match (EM)
  - Token F1
  - Citation Accuracy & Coverage
  - Groundedness (human-adjudicated hoặc heuristic)
  - Faithfulness (human-adjudicated)
  - Safety Score (accuracy + unsafe recall + abstention)
  - Confusion Matrix cho safety labels
  - Per-domain breakdown
"""

import json
from collections import Counter, defaultdict

from modules.benchmark import (
    _normalize_text,
    _token_f1,
    extract_citations,
    _SAFETY_LABELS,
)

SAFETY_LABEL_ORDER = ["SAFE", "CAUTION", "WARNING", "CRITICAL_ALERT", "ABSTAIN"]


def compute_confusion_matrix(gold_labels, predicted_labels):
    """
    Tạo confusion matrix cho safety labels.
    Trả về dict: {gold_label: {predicted_label: count}}.
    """
    labels = SAFETY_LABEL_ORDER
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for gold, pred in zip(gold_labels, predicted_labels):
        if gold in labels and pred in labels:
            matrix[gold][pred] += 1
    return matrix


def format_confusion_matrix(matrix):
    """Định dạng confusion matrix thành bảng text."""
    labels = SAFETY_LABEL_ORDER
    header = f"{'':>16s}" + "".join(f"{l:>16s}" for l in labels)
    rows = [header]
    for gold in labels:
        row = f"{gold:>16s}"
        for pred in labels:
            row += f"{matrix[gold][pred]:>16d}"
        rows.append(row)
    return "\n".join(rows)


def _heuristic_groundedness(answer, evidence_excerpt):
    """
    Heuristic groundedness: đo tỷ lệ token trong answer có mặt trong evidence.
    Dùng khi không có human annotation.
    """
    if not answer or not evidence_excerpt:
        return 0.0
    answer_tokens = set(_normalize_text(answer).split())
    evidence_tokens = set(_normalize_text(evidence_excerpt).split())
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & evidence_tokens
    return len(overlap) / len(answer_tokens)


def evaluate_single(prediction, reference):
    """
    Đánh giá một cặp prediction/reference, trả về dict chỉ số chi tiết.
    """
    answer = str(prediction.get("answer", ""))
    gold_answer = str(reference["reference_answer"])

    em = _normalize_text(answer) == _normalize_text(gold_answer)
    f1 = _token_f1(answer, gold_answer)

    gold_sources = {_normalize_text(s) for s in reference.get("source_ids", [])}
    predicted_sources = set(extract_citations(answer))

    citation_precision = (
        len(predicted_sources & gold_sources) / len(predicted_sources)
        if predicted_sources else 0.0
    )
    citation_recall = (
        len(predicted_sources & gold_sources) / len(gold_sources)
        if gold_sources else 0.0
    )

    gold_safety = reference.get("gold_safety_label", "")
    pred_safety = prediction.get("safety_label", "")
    safety_correct = (
        gold_safety == pred_safety
        if gold_safety in _SAFETY_LABELS and pred_safety in _SAFETY_LABELS
        else None
    )

    grounded = prediction.get("grounded")
    if not isinstance(grounded, bool):
        evidence = str(reference.get("evidence_excerpt", ""))
        grounded = _heuristic_groundedness(answer, evidence) if evidence else None

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
        "safety_correct": safety_correct,
        "gold_safety_label": gold_safety,
        "predicted_safety_label": pred_safety,
        "groundedness": grounded,
        "faithfulness": faithful,
    }


def aggregate_metrics(item_results):
    """
    Tổng hợp từ danh sách kết quả per-item thành metrics tổng thể.
    """
    n = len(item_results)
    if n == 0:
        return {"count": 0}

    em_scores = [r["exact_match"] for r in item_results]
    f1_scores = [r["token_f1"] for r in item_results]
    cit_prec = [r["citation_precision"] for r in item_results]
    cit_rec = [r["citation_recall"] for r in item_results]

    safety_items = [r for r in item_results if r["safety_correct"] is not None]
    safety_correct_count = sum(1 for r in safety_items if r["safety_correct"])

    unsafe_tp = sum(
        1 for r in safety_items
        if r["gold_safety_label"] in ("WARNING", "CRITICAL_ALERT")
        and r["predicted_safety_label"] in ("WARNING", "CRITICAL_ALERT")
    )
    unsafe_fn = sum(
        1 for r in safety_items
        if r["gold_safety_label"] in ("WARNING", "CRITICAL_ALERT")
        and r["predicted_safety_label"] not in ("WARNING", "CRITICAL_ALERT")
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
        "citation_precision": sum(cit_prec) / n,
        "citation_recall": sum(cit_rec) / n,
    }

    if safety_items:
        safety_acc = safety_correct_count / len(safety_items)
        unsafe_recall = (
            unsafe_tp / (unsafe_tp + unsafe_fn)
            if (unsafe_tp + unsafe_fn) > 0 else 0.0
        )
        result["safety_accuracy"] = safety_acc
        result["unsafe_recall"] = unsafe_recall
        result["safety_score"] = 0.6 * safety_acc + 0.4 * unsafe_recall

    result["abstention_rate_predicted"] = abstain_pred / n
    result["abstention_rate_gold"] = abstain_gold / n

    if grounded_vals:
        result["groundedness"] = sum(
            (v if isinstance(v, (int, float)) else int(v))
            for v in grounded_vals
        ) / len(grounded_vals)

    if faithful_vals:
        result["faithfulness"] = sum(
            (v if isinstance(v, (int, float)) else int(v))
            for v in faithful_vals
        ) / len(faithful_vals)

    return result


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
                else "SAFE"
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


def format_results_table(config_results_list):
    """
    Định dạng bảng so sánh nhiều configs (A/B/C) thành text.
    """
    if not config_results_list:
        return "No results."

    metrics = [
        ("count", "N", "{:d}"),
        ("exact_match", "EM", "{:.3f}"),
        ("token_f1", "Token F1", "{:.3f}"),
        ("citation_precision", "Cit. Prec", "{:.3f}"),
        ("citation_recall", "Cit. Recall", "{:.3f}"),
        ("safety_accuracy", "Safety Acc", "{:.3f}"),
        ("unsafe_recall", "Unsafe Rec", "{:.3f}"),
        ("safety_score", "Safety Score", "{:.3f}"),
        ("groundedness", "Grounded", "{:.3f}"),
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
