"""Evaluation helpers for manually curated FoodHealthQA benchmark records."""

import json
import re
import unicodedata
from collections import Counter


_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_SAFETY_LABELS = {
    "SAFE",
    "CAUTION",
    "WARNING",
    "CRITICAL_ALERT",
    "ABSTAIN",
}


def _normalize_text(value):
    value = unicodedata.normalize("NFKD", str(value).casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _token_f1(prediction, reference):
    predicted_tokens = _normalize_text(prediction).split()
    reference_tokens = _normalize_text(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_citations(text):
    """Return normalized citation labels in the order they occur."""
    return [
        _normalize_text(citation)
        for citation in _CITATION_PATTERN.findall(str(text))
        if citation.strip()
    ]


def validate_citations(text, allowed_sources, require_each_sentence=False):
    """Check that citations are present and refer only to allowed evidence sources."""
    allowed = {_normalize_text(source) for source in allowed_sources if str(source).strip()}
    text = str(text)
    citations = extract_citations(text)
    unknown = sorted(set(citations) - allowed)
    uncited_sentences = []
    if require_each_sentence:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\r?\n+", text)
            if sentence.strip()
        ]
        uncited_sentences = [
            sentence for sentence in sentences
            if not _CITATION_PATTERN.search(sentence)
        ]
    return {
        "valid": bool(citations) and not unknown and not uncited_sentences,
        "citations": citations,
        "unknown_citations": unknown,
        "missing_citations": not citations,
        "uncited_sentences": uncited_sentences,
    }


def load_benchmark(path):
    """Load JSONL benchmark records and validate stable IDs and required fields."""
    records = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            required = {"id", "question", "reference_answer", "source_ids", "domain"}
            missing = required - record.keys()
            if missing:
                raise ValueError(
                    f"Benchmark line {line_number} is missing fields: {sorted(missing)}"
                )
            if record["id"] in seen_ids:
                raise ValueError(f"Duplicate benchmark id on line {line_number}: {record['id']}")
            if not isinstance(record["source_ids"], list) or not record["source_ids"]:
                raise ValueError(f"Benchmark line {line_number} needs at least one source_id")
            seen_ids.add(record["id"])
            records.append(record)
    return records


def evaluate_predictions(references, predictions):
    """
    Compute answer, citation, and safety metrics for aligned records.

    `grounded` and `faithful` are human-adjudicated booleans on prediction rows;
    those metrics are omitted unless such annotations are present.
    """
    reference_by_id = {record["id"]: record for record in references}
    prediction_by_id = {record["id"]: record for record in predictions}
    if reference_by_id.keys() != prediction_by_id.keys():
        missing = sorted(reference_by_id.keys() - prediction_by_id.keys())
        extra = sorted(prediction_by_id.keys() - reference_by_id.keys())
        raise ValueError(f"Prediction IDs must match references; missing={missing}, extra={extra}")

    exact_matches = []
    token_f1s = []
    citation_precisions = []
    citation_recalls = []
    safety_correct = []
    unsafe_true_positives = 0
    unsafe_false_negatives = 0
    grounded_labels = []
    faithful_labels = []

    for record_id, reference in reference_by_id.items():
        prediction = prediction_by_id[record_id]
        answer = str(prediction.get("answer", ""))
        gold_answer = str(reference["reference_answer"])
        exact_matches.append(_normalize_text(answer) == _normalize_text(gold_answer))
        token_f1s.append(_token_f1(answer, gold_answer))

        gold_sources = {_normalize_text(source) for source in reference["source_ids"]}
        predicted_sources = set(extract_citations(answer))
        citation_precisions.append(
            len(predicted_sources & gold_sources) / len(predicted_sources)
            if predicted_sources else 0.0
        )
        citation_recalls.append(
            len(predicted_sources & gold_sources) / len(gold_sources)
            if gold_sources else 0.0
        )

        gold_safety = reference.get("safety_label")
        predicted_safety = prediction.get("safety_label")
        if gold_safety in _SAFETY_LABELS and predicted_safety in _SAFETY_LABELS:
            correct = gold_safety == predicted_safety
            safety_correct.append(correct)
            if gold_safety in {"WARNING", "CRITICAL_ALERT"}:
                if predicted_safety in {"WARNING", "CRITICAL_ALERT"}:
                    unsafe_true_positives += 1
                else:
                    unsafe_false_negatives += 1

        if isinstance(prediction.get("grounded"), bool):
            grounded_labels.append(prediction["grounded"])
        if isinstance(prediction.get("faithful"), bool):
            faithful_labels.append(prediction["faithful"])

    result = {
        "count": len(reference_by_id),
        "exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
        "token_f1": sum(token_f1s) / len(token_f1s) if token_f1s else 0.0,
        "citation_accuracy": (
            sum(citation_precisions) / len(citation_precisions) if citation_precisions else 0.0
        ),
        "citation_coverage": (
            sum(citation_recalls) / len(citation_recalls) if citation_recalls else 0.0
        ),
    }
    if safety_correct:
        result["safety_accuracy"] = sum(safety_correct) / len(safety_correct)
        result["unsafe_recall"] = (
            unsafe_true_positives / (unsafe_true_positives + unsafe_false_negatives)
            if unsafe_true_positives + unsafe_false_negatives else 0.0
        )
    if grounded_labels:
        result["groundedness_human_adjudicated"] = (
            sum(grounded_labels) / len(grounded_labels)
        )
    if faithful_labels:
        result["faithfulness_human_adjudicated"] = (
            sum(faithful_labels) / len(faithful_labels)
        )
    return result
