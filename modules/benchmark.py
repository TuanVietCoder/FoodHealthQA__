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
    "UNSAFE",
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
    "CODEX_CAC_GL_23_1997": {
        "title": "Codex CAC/GL 23-1997 - Guidelines for Use of Nutrition and Health Claims (sodium conditions)",
        "jurisdiction": "International",
        "aliases": [
            "CODEX_CAC_GL_23_1997", "CAC/GL 23-1997", "Codex CAC/GL 23-1997", "CXG 23-1997",
        ],
    },
    "EU_REG_1333_2008_ANNEX_V": {
        "title": "Regulation (EC) No 1333/2008 on food additives, Annex V (food colours requiring an additional label warning)",
        "jurisdiction": "European Union",
        "aliases": ["EU_REG_1333_2008_ANNEX_V", "EU Regulation 1333/2008 Annex V", "Regulation (EC) 1333/2008 Annex V"],
    },
    "US_FDA_RED3_REVOCATION": {
        "title": "US FDA order revoking the color additive listing of FD&C Red No. 3 in food and ingested drugs (FR Doc. 2025-00830)",
        "jurisdiction": "United States",
        "aliases": ["US_FDA_RED3_REVOCATION", "FR 2025-00830", "FDA FD&C Red No. 3 revocation"],
    },
    "IARC_MONOGRAPHS_V134": {
        "title": "IARC Monographs Volume 134: Aspartame, methyleugenol, and isoeugenol (2023 evaluation, published 2024)",
        "jurisdiction": "International",
        "aliases": ["IARC_MONOGRAPHS_V134", "IARC Monographs Volume 134", "IARC Aspartame Group 2B"],
    },
    "WHO_NSS_GUIDELINE_2023": {
        "title": "WHO guideline: Use of non-sugar sweeteners (Geneva: World Health Organization, 2023)",
        "jurisdiction": "International",
        "aliases": ["WHO_NSS_GUIDELINE_2023", "WHO non-sugar sweeteners guideline 2023", "WHO NSS guideline"],
    },
    "EFSA_CAFFEINE_2015": {
        "title": "EFSA NDA Panel (2015) Scientific Opinion on the safety of caffeine. EFSA Journal 13(5):4102",
        "jurisdiction": "European Union",
        "aliases": ["EFSA_CAFFEINE_2015", "EFSA caffeine opinion 2015", "EFSA Journal 2015;13(5):4102"],
    },
    "US_FDA_ALLERGEN_QA_ED5": {
        "title": "US FDA Q&A Regarding Food Allergens (Edition 5, 2025), non-binding guidance",
        "jurisdiction": "United States",
        "aliases": [
            "US_FDA_ALLERGEN_QA_ED5", "FDA Food Allergen Q&A Edition 5", "FDA Allergen Guidance Edition 5",
        ],
    },
    "US_FDA_ASPARTAME_SWEETENERS": {
        "title": "US FDA - Aspartame and Other Sweeteners in Food (2025): people with PKU should avoid or restrict aspartame",
        "jurisdiction": "United States",
        "aliases": ["US_FDA_ASPARTAME_SWEETENERS", "FDA Aspartame and Other Sweeteners in Food"],
    },
    "US_FDA_21CFR172_804": {
        "title": "US FDA 21 CFR 172.804 - Aspartame (label statement for phenylketonurics)",
        "jurisdiction": "United States",
        "aliases": [
            "US_FDA_21CFR172_804", "US_FDA_21CFR172.804", "21 CFR 172.804", "21 CFR § 172.804",
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
            "WHO_JEMRA_PRIORITY_ALLERGENS", "WHO JEMRA PRIORITY ALLERGENS",
        ],
    },
    "WHO_JEMRA_PRIORITY_ALLERGENS": {
        "title": "FAO/WHO JEMRA Part 1: Priority Allergen Summary Report (2021)",
        "jurisdiction": "International",
        "aliases": [
            "WHO_JEMRA_PRIORITY_ALLERGENS", "WHO JEMRA PRIORITY ALLERGENS",
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
            "Codex General Standard for the Labelling of Prepackaged Foods (CXS 1-1985)",
        ],
    },
    "CODEX_STAN_192_1995": {
        "title": "Codex General Standard for Food Additives (STAN 192-1995 / GSFA)",
        "jurisdiction": "Codex Alimentarius",
        "aliases": [
            "CODEX_STAN_192_1995", "Codex STAN 192-1995", "STAN 192-1995", "Codex GSFA", "GSFA",
        ],
    },
    "CODEX_CXS_72_1981": {
        "title": "Codex CXS 72-1981 Standard for Infant Formula and Formulas for Special Medical Purposes Intended for Infants",
        "jurisdiction": "International",
        "aliases": ["CODEX_CXS_72_1981", "CXS 72-1981", "Codex CXS 72-1981", "CODEX STAN 72-1981"],
    },
    "CODEX_CXS_74_1981": {
        "title": "Codex CXS 74-1981 Standard for Processed Cereal-Based Foods for Infants and Young Children",
        "jurisdiction": "International",
        "aliases": ["CODEX_CXS_74_1981", "CXS 74-1981", "Codex CXS 74-1981", "CODEX STAN 074-1981"],
    },
    "VN_QCVN_11_1_2012": {
        "title": "QCVN 11-1:2012/BYT - Sản phẩm dinh dưỡng công thức cho trẻ đến 12 tháng tuổi",
        "jurisdiction": "Vietnam",
        "aliases": ["VN_QCVN_11_1_2012", "QCVN 11-1:2012/BYT", "QCVN 11-1:2012"],
    },
    "WHO_COMPLEMENTARY_FEEDING_2023": {
        "title": "WHO Guideline for complementary feeding of infants and young children 6-23 months of age (2023)",
        "jurisdiction": "International",
        "aliases": ["WHO_COMPLEMENTARY_FEEDING_2023", "WHO complementary feeding guideline 2023"],
    },
    "CODEX_CXG_36_1989": {
        "title": "Codex Class Names and the International Numbering System for Food Additives (CXG 36-1989)",
        "jurisdiction": "Codex Alimentarius",
        "aliases": [
            "CODEX_CXG_36_1989", "Codex CXG 36-1989", "CXG 36-1989",
            "International Numbering System for Food Additives", "Codex INS",
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
            "EFSA Journal", "EC Regulation No 1333/2008", "EFSA_2021_TITANIUM_DIOXIDE_E171",
        ],
    },
    "EFSA_2021_TITANIUM_DIOXIDE_E171": {
        "title": "EFSA 2021 Scientific Opinion on Titanium Dioxide (E171)",
        "jurisdiction": "European Union",
        "aliases": [
            "EFSA_2021_TITANIUM_DIOXIDE_E171", "EFSA Titanium Dioxide E171", "E171 Ban",
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
    """Return normalized citation labels in the order they occur, stripping prefixes and splitting multi-citations."""
    raw = _CITATION_PATTERN.findall(str(text))
    citations = []
    for citation in raw:
        c_clean = re.sub(r"^(?:nguồn|nguon|source|ref)\s*:\s*", "", citation.strip(), flags=re.IGNORECASE)
        for sub_c in re.split(r"[,;&]", c_clean):
            sub_c = sub_c.strip()
            if sub_c:
                norm = _normalize_text(sub_c)
                if norm:
                    citations.append(norm)
    return citations


def analyze_answer_citations(answer):
    """
    Phân loại các thẻ nguồn `[...]` trong câu trả lời:
      - recognized: thuộc danh mục nguồn được công nhận (đã chuẩn hóa, không trùng lặp, đúng thứ tự xuất hiện)
      - unrecognized: thẻ không thuộc danh mục (vd mô hình tự bịa `[khong tim thay]`)
    `has_citation` chỉ tính thẻ được công nhận.
    """
    recognized, unrecognized = [], []
    for raw in extract_citations(answer):
        canon = canonicalize_citation(raw)
        bucket = recognized if canon in RECOGNIZED_CITATION_SOURCES else unrecognized
        if canon not in bucket:
            bucket.append(canon)
    return {"recognized": recognized, "unrecognized": unrecognized, "has_citation": bool(recognized)}


def valid_predicted_sources(analysis, gold_sources):
    """
    Nguồn dự đoán dùng để tính citation precision/recall: thẻ được công nhận, cộng thẻ khớp nguồn chuẩn của câu.
    Thẻ lạ (không công nhận, không khớp nguồn chuẩn) bị loại khỏi precision nhưng KHÔNG bị che giấu:
    chúng được đếm riêng ở `unrecognized_citation_rate`.
    """
    gold_sources = set(gold_sources)
    valid = set(analysis["recognized"]) | (set(analysis["unrecognized"]) & gold_sources)
    stray = [u for u in analysis["unrecognized"] if u not in gold_sources]
    return valid, stray


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
            if "source_ids" not in record and record.get("source_citations"):
                record["source_ids"] = [
                    s.strip()
                    for s in str(record["source_citations"]).split(",")
                    if s.strip()
                ]
            if "safety_label" not in record and record.get("gold_safety_label"):
                record["safety_label"] = record["gold_safety_label"]
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
    cited_flags = []
    stray_counts = []
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

        raw_sources = reference.get("source_ids") or [
            s.strip()
            for s in str(reference.get("source_citations", "")).split(",")
            if s.strip()
        ]
        gold_sources = {canonicalize_citation(source) for source in raw_sources}
        _analysis = analyze_answer_citations(answer)
        predicted_sources, _stray = valid_predicted_sources(_analysis, gold_sources)
        cited_flags.append(_analysis["has_citation"])
        stray_counts.append(len(_stray))
        citation_precisions.append(
            len(predicted_sources & gold_sources) / len(predicted_sources)
            if predicted_sources else 0.0
        )
        citation_recalls.append(
            len(predicted_sources & gold_sources) / len(gold_sources)
            if gold_sources else 0.0
        )

        gold_safety = reference.get("safety_label") or reference.get("gold_safety_label")
        predicted_safety = prediction.get("safety_label")
        if gold_safety in _SAFETY_LABELS and predicted_safety in _SAFETY_LABELS:
            _norm_map = {"WARNING": "UNSAFE", "SCREENED_NO_ALERT": "SAFE"}
            norm_gold = _norm_map.get(gold_safety, gold_safety)
            norm_pred = _norm_map.get(predicted_safety, predicted_safety)
            correct = (gold_safety == predicted_safety) or (norm_gold == norm_pred)
            safety_correct.append(correct)
            if gold_safety in {"WARNING", "CRITICAL_ALERT", "UNSAFE"}:
                if predicted_safety in {"WARNING", "CRITICAL_ALERT", "UNSAFE"}:
                    unsafe_true_positives += 1
                else:
                    unsafe_false_negatives += 1

        if isinstance(prediction.get("grounded"), bool):
            grounded_labels.append(prediction["grounded"])
        if isinstance(prediction.get("faithful"), bool):
            faithful_labels.append(prediction["faithful"])

    # --- Nhóm 1: Đầu vào (Input Quality) ---
    input_metrics = {}
    ocr_cer_values = [p.get("ocr_cer") for p in predictions if p.get("ocr_cer") is not None]
    ocr_wer_values = [p.get("ocr_wer") for p in predictions if p.get("ocr_wer") is not None]
    ingredient_precisions = [p.get("ingredient_precision") for p in predictions if p.get("ingredient_precision") is not None]
    ingredient_recalls = [p.get("ingredient_recall") for p in predictions if p.get("ingredient_recall") is not None]
    linking_accuracies = [p.get("entity_linking_accuracy") for p in predictions if p.get("entity_linking_accuracy") is not None]

    if ocr_cer_values:
        input_metrics["ocr_cer_mean"] = sum(ocr_cer_values) / len(ocr_cer_values)
    if ocr_wer_values:
        input_metrics["ocr_wer_mean"] = sum(ocr_wer_values) / len(ocr_wer_values)
    if ingredient_precisions:
        input_metrics["ingredient_precision"] = sum(ingredient_precisions) / len(ingredient_precisions)
    if ingredient_recalls:
        input_metrics["ingredient_recall"] = sum(ingredient_recalls) / len(ingredient_recalls)
        if ingredient_precisions:
            p = input_metrics.get("ingredient_precision", 0)
            r = input_metrics["ingredient_recall"]
            input_metrics["ingredient_f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    if linking_accuracies:
        input_metrics["entity_linking_accuracy"] = sum(linking_accuracies) / len(linking_accuracies)

    # --- Nhóm 2: Truy xuất và trả lời (Retrieval & Answer Quality) ---
    retrieval_answer_metrics = {
        "exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
        "token_f1": sum(token_f1s) / len(token_f1s) if token_f1s else 0.0,
        "citation_precision": (
            sum(citation_precisions) / len(citation_precisions) if citation_precisions else 0.0
        ),
        "citation_recall": (
            sum(citation_recalls) / len(citation_recalls) if citation_recalls else 0.0
        ),
        # Tuân thủ trích dẫn: tỷ lệ câu trả lời có >=1 thẻ nguồn được công nhận (đáng chú ý ở B/C; A không truy xuất)
        "citation_compliance": sum(cited_flags) / len(cited_flags) if cited_flags else 0.0,
        # Thẻ nguồn lạ (mô hình tự bịa): tỷ lệ câu trả lời có thẻ lạ, và số thẻ lạ trung bình mỗi câu
        "unrecognized_citation_rate": (
            sum(1 for c in stray_counts if c > 0) / len(stray_counts) if stray_counts else 0.0
        ),
        "unrecognized_tags_per_answer": sum(stray_counts) / len(stray_counts) if stray_counts else 0.0,
    }
    recall_at_k_values = [p.get("recall_at_k") for p in predictions if p.get("recall_at_k") is not None]
    if recall_at_k_values:
        retrieval_answer_metrics["recall_at_k"] = sum(recall_at_k_values) / len(recall_at_k_values)
    if grounded_labels:
        retrieval_answer_metrics["groundedness"] = sum(grounded_labels) / len(grounded_labels)
    if faithful_labels:
        retrieval_answer_metrics["faithfulness"] = sum(faithful_labels) / len(faithful_labels)

    # --- Nhóm 3: An toàn và từ chối (Safety & Abstention) ---
    safety_metrics = {}
    if safety_correct:
        safety_metrics["safety_label_accuracy"] = sum(safety_correct) / len(safety_correct)
        safety_metrics["false_safe_rate"] = (
            unsafe_false_negatives / (unsafe_true_positives + unsafe_false_negatives)
            if (unsafe_true_positives + unsafe_false_negatives) else 0.0
        )
        safety_metrics["unsafe_recall"] = (
            unsafe_true_positives / (unsafe_true_positives + unsafe_false_negatives)
            if unsafe_true_positives + unsafe_false_negatives else 0.0
        )

    # Coverage trên phần đã trả lời (non-abstain)
    non_abstain = [p for p in predictions if p.get("safety_label") != "ABSTAIN"]
    abstain_count = len(predictions) - len(non_abstain)
    safety_metrics["abstain_count"] = abstain_count
    safety_metrics["non_abstain_coverage"] = len(non_abstain) / len(predictions) if predictions else 0.0

    # Confusion matrix cho safety labels
    confusion = {}
    for record_id, reference in reference_by_id.items():
        prediction = prediction_by_id[record_id]
        gold = reference.get("safety_label", "")
        pred = prediction.get("safety_label", "")
        if gold in _SAFETY_LABELS and pred in _SAFETY_LABELS:
            confusion.setdefault(gold, Counter())[pred] += 1
    if confusion:
        safety_metrics["confusion_matrix"] = {k: dict(v) for k, v in confusion.items()}

    # Tổng hợp kết quả theo 3 nhóm
    result = {
        "count": len(reference_by_id),
        "input_quality": input_metrics,
        "retrieval_and_answer": retrieval_answer_metrics,
        "safety_and_abstention": safety_metrics,
        # Flat metrics giữ tương thích ngược
        "exact_match": retrieval_answer_metrics["exact_match"],
        "token_f1": retrieval_answer_metrics["token_f1"],
        "citation_accuracy": retrieval_answer_metrics["citation_precision"],
        "citation_coverage": retrieval_answer_metrics["citation_recall"],
        "citation_compliance": retrieval_answer_metrics["citation_compliance"],
        "unrecognized_citation_rate": retrieval_answer_metrics["unrecognized_citation_rate"],
    }
    if "safety_label_accuracy" in safety_metrics:
        result["safety_accuracy"] = safety_metrics["safety_label_accuracy"]
        result["unsafe_recall"] = safety_metrics["unsafe_recall"]
        result["false_safe_rate"] = safety_metrics["false_safe_rate"]
    if grounded_labels:
        result["groundedness_human_adjudicated"] = retrieval_answer_metrics["groundedness"]
    if faithful_labels:
        result["faithfulness_human_adjudicated"] = retrieval_answer_metrics["faithfulness"]
    return result


def resolve_friendly_domain(item):
    """
    Chuẩn hóa tên lĩnh vực chuyên môn thực phẩm chính xác.
    Khắc phục trường hợp 'insufficient_data' / 'Thiếu dữ liệu' (vốn là nhãn kỹ thuật
    của nhóm kiểm thử cơ chế từ chối Abstention trong benchmark).
    """
    domain = str(item.get("domain") or "").strip().lower()
    domain_vi = str(item.get("domain_vi") or "").strip()
    question = str(item.get("question") or "").lower()

    if domain == "insufficient_data" or domain_vi == "Thiếu dữ liệu":
        if any(w in question for w in [
            "dị nguyên", "dị ứng", "nhiễm chéo", "pal", "trứng", "sữa", "đậu phộng",
            "hạt cây", "gluten", "lúa mì", "tôm", "cua", "cá", "mè", "đậu nành", "hải sản"
        ]):
            return "Cảnh báo Dị ứng & Nhiễm chéo"
        if any(w in question for w in [
            "phụ gia", "ins", "chất bảo quản", "tạo màu", "tạo ngọt", "adi",
            "tích lũy", "hàm lượng", "ăn hàng ngày"
        ]):
            return "Phụ gia & Quy chuẩn TT24"
        if any(w in question for w in [
            "mang thai", "thai kỳ", "tiểu đường", "tim mạch", "trẻ em",
            "suy thận", "bệnh", "dinh dưỡng", "natri", "đường", "calo"
        ]):
            return "Dinh dưỡng & Bệnh lý lâm sàng"
        if any(w in question for w in ["không có mã ins", "chỉ ghi tên", "thành phần", "nhãn"]):
            return "Kiểm định thông tin nhãn"
        return "Kiểm định an toàn (Abstention)"

    if domain == "additive_regulation" or "phụ gia" in domain_vi.lower():
        return "Phụ gia & Quy chuẩn TT24"
    if domain == "allergen_detection" or "dị nguyên" in domain_vi.lower():
        return "Cảnh báo Dị ứng & Nhiễm chéo"
    if domain == "vulnerable_groups" or "tổn thương" in domain_vi.lower():
        return "Dinh dưỡng & Bệnh lý lâm sàng"

    return domain_vi or "Tư vấn an toàn"


def get_domain_badge(item):
    """Trả về badge ngắn gọn kèm biểu tượng emoji để hiển thị trong selectbox."""
    friendly = resolve_friendly_domain(item)
    if "Dị ứng" in friendly:
        return "⚠️ Dị ứng"
    if "Phụ gia" in friendly:
        return "🧪 Phụ gia"
    if "Lâm sàng" in friendly or "Bệnh lý" in friendly or "Dinh dưỡng" in friendly:
        return "🩺 Lâm sàng"
    if "nhãn" in friendly.lower():
        return "🔍 Nhãn thiếu INS"
    return "📋 Tư vấn"
