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

# Danh mục trích dẫn quy chuẩn khoa học & pháp lý được công nhận chính thức
RECOGNIZED_CITATION_SOURCES = {
    "US_FDA_FALCPA": {
        "title": "US FDA Food Allergen Labeling and Consumer Protection Act (FALCPA of 2004)",
        "jurisdiction": "United States",
        "aliases": [
            "US_FDA_FALCPA", "US FDA FALCPA", "FALCPA", "FDA FALCPA",
            "US FDA FALCPA Sec 203", "FALCPA Section 203",
            "Food Allergen Labeling and Consumer Protection Act",
        ],
    },
    "US_FDA_FASTER_ACT_2023": {
        "title": "US FDA Food Allergy Safety, Treatment, Education, and Research Act (FASTER Act 2023)",
        "jurisdiction": "United States",
        "aliases": [
            "US_FDA_FASTER_ACT_2023", "US_FDA_FASTER_ACT", "US FDA FASTER Act 2023",
            "FASTER Act", "FASTER Act 2023", "US FDA FASTER Act",
            "Food Allergy Safety Treatment Education and Research Act",
        ],
    },
    "FAO_WHO_JEMRA_PART_3": {
        "title": "FAO/WHO JEMRA Part 3: Precautionary Allergen Labelling (PAL) Guidance (2023)",
        "jurisdiction": "International",
        "aliases": [
            "FAO_WHO_JEMRA_PART_3", "FAO/WHO JEMRA Part 3", "FAO WHO JEMRA Part 3",
            "FAO/WHO JEMRA Part 3 (PAL, 2023)", "FAO/WHO JEMRA Part 3 (PAL)",
            "FAO/WHO JEMRA Part 3 (Precautionary Allergen Labelling)",
            "FAO_WHO_JEMRA_PAL", "JEMRA Part 3", "JEMRA PAL",
        ],
    },
    "FAO_WHO_JEMRA_2021": {
        "title": "FAO/WHO JEMRA Part 1: Priority Allergen Summary Report (2021)",
        "jurisdiction": "International",
        "aliases": [
            "FAO_WHO_JEMRA_2021", "FAO_WHO_JEMRA_PART_1", "FAO/WHO JEMRA 2021 Part 1",
            "FAO/WHO JEMRA 2021", "FAO WHO JEMRA 2021", "JEMRA 2021", "JEMRA Part 1",
        ],
    },
    "CODEX_CXS_1_1985": {
        "title": "Codex General Standard for the Labelling of Prepackaged Foods (CXS 1-1985)",
        "jurisdiction": "Codex Alimentarius",
        "aliases": [
            "CODEX_CXS_1_1985", "CODEX_CXS_1", "Codex CXS 1-1985", "Codex CXS 1",
            "CXS 1-1985", "CXS 1", "Codex General Standard for the Labelling of Prepackaged Foods",
        ],
    },
    "CODEX_STAN_192_1995": {
        "title": "Codex General Standard for Food Additives (STAN 192-1995 / GSFA)",
        "jurisdiction": "Codex Alimentarius",
        "aliases": [
            "CODEX_STAN_192_1995", "Codex STAN 192-1995", "STAN 192-1995", "Codex GSFA", "GSFA",
        ],
    },
    "VN_TT24_2019": {
        "title": "Thông tư 24/2019/TT-BYT quy định về quản lý và sử dụng phụ gia thực phẩm",
        "jurisdiction": "Vietnam",
        "aliases": [
            "VN_TT24_2019", "Thông tư 24/2019/TT-BYT", "TT 24/2019/TT-BYT", "TT24/2019", "TT 24/2019",
        ],
    },
    "EFSA_OPINIONS": {
        "title": "European Food Safety Authority (EFSA) Scientific Opinions",
        "jurisdiction": "European Union",
        "aliases": [
            "EFSA", "EFSA_OPINIONS", "EFSA_GLUTAMATES", "EFSA_MODIFIED_STARCHES", "EFSA_SULFITES",
            "EFSA Journal", "EC Regulation No 1333/2008",
        ],
    },
    "JECFA_EVALUATIONS": {
        "title": "Joint FAO/WHO Expert Committee on Food Additives (JECFA)",
        "jurisdiction": "International",
        "aliases": [
            "JECFA", "JECFA_DB", "JECFA_ASPARTAME", "FAO_JECFA_SPECS", "JECFA ADI",
        ],
    },
    "USDA_FDC": {
        "title": "USDA FoodData Central / FNDDS Nutrient Database",
        "jurisdiction": "United States",
        "aliases": [
            "USDA_FDC", "USDA FDC", "USDA FNDDS", "FNDDS", "USDA",
        ],
    },
}


def _normalize_text(value):
    value = unicodedata.normalize("NFKD", str(value).casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


CANONICAL_CITATION_MAP = {}
for canonical_id, meta in RECOGNIZED_CITATION_SOURCES.items():
    norm_canon = _normalize_text(canonical_id)
    CANONICAL_CITATION_MAP[norm_canon] = canonical_id
    for alias in meta.get("aliases", []):
        norm_alias = _normalize_text(alias)
        if norm_alias:
            CANONICAL_CITATION_MAP[norm_alias] = canonical_id


def canonicalize_citation(citation):
    """
    Chuẩn hóa chuỗi trích dẫn về mã định danh chuẩn chính tắc (Canonical Citation ID).
    Hỗ trợ đối sánh linh hoạt các biến thể ký hiệu (VD: 'US FDA FALCPA' -> 'US_FDA_FALCPA').
    """
    norm = _normalize_text(citation)
    return CANONICAL_CITATION_MAP.get(norm, norm)


def is_recognized_citation(citation):
    """Kiểm tra xem trích dẫn có thuộc danh mục được công nhận chính thức hay không."""
    canon = canonicalize_citation(citation)
    return canon in RECOGNIZED_CITATION_SOURCES


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
    allowed = {canonicalize_citation(source) for source in allowed_sources if str(source).strip()}
    text = str(text)
    raw_citations = extract_citations(text)
    canonical_citations = [canonicalize_citation(c) for c in raw_citations]
    unknown = sorted(set(canonical_citations) - allowed)
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
        "valid": bool(canonical_citations) and not unknown and not uncited_sentences,
        "citations": canonical_citations,
        "unknown_citations": unknown,
        "missing_citations": not canonical_citations,
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

        gold_sources = {canonicalize_citation(source) for source in reference["source_ids"]}
        predicted_sources = {canonicalize_citation(source) for source in extract_citations(answer)}
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
