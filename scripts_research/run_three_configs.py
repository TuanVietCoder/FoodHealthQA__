"""Run a small Gemini comparison on expert-approved QA/image mappings only."""

import argparse
import csv
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import dotenv_values
from modules.hybrid_retriever import HybridRetriever
from modules.pipeline import load_knowledge_base


DEFAULT_REVIEW_CSV = os.path.join(
    PROJECT_ROOT, "data", "qa_benchmark_merged_review.csv"
)
DEFAULT_IMAGE_PREDICTIONS = os.path.join(
    PROJECT_ROOT, "data", "image_ocr_predictions.csv"
)
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "three_config_answers.jsonl")
MODEL_NAME = "gemini-3.8-flash"
APPROVED = {"yes", "true", "approved", "correct"}


def _read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _api_key():
    env_path = os.path.join(PROJECT_ROOT, ".env")
    file_values = dotenv_values(env_path) if os.path.isfile(env_path) else {}
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or file_values.get("GEMINI_API_KEY")
        or file_values.get("GOOGLE_API_KEY")
    )


def _is_approved(record):
    status = record.get("review_status", "").strip().casefold()
    if status != "approved":
        return False
    if "answer_supported_by_source" in record and record["answer_supported_by_source"]:
        if record["answer_supported_by_source"].strip().casefold() not in APPROVED:
            return False
    if "citation_correct" in record and record["citation_correct"]:
        if record["citation_correct"].strip().casefold() not in APPROVED:
            return False
    final = (
        record.get("final_answer", "")
        or record.get("reference_answer", "")
        or record.get("corrected_answer", "")
    )
    return bool(final.strip())


def _additive_evidence(additive, source_id):
    evidence = [
        f"INS/E code: {additive.get('ins_code', '')} / {additive.get('e_number', '')}",
        f"Name: {additive.get('canonical_name', '')}",
        f"Function: {additive.get('function', '')}",
        f"ADI in local KB: {additive.get('jecfa_adi', '')}",
        f"Vietnam regulation field: {additive.get('vn_regulation', '')}",
        f"Codex field: {additive.get('codex_standard', '')}",
    ]
    for caution in additive.get("contraindications", []):
        evidence.append(
            "Local KB caution (unverified): "
            f"{caution.get('condition', '')} / {caution.get('severity', '')}: "
            f"{caution.get('message', '')}"
        )
    return "\n".join(evidence), source_id


def _retrieve_text_evidence(question, retriever):
    evidence = []
    for additive, score, _method in retriever.retrieve(question, top_k=3):
        if score <= 0:
            continue
        source_id = f"KB:knowledge_base/additives_standard.json#{additive.get('ins_code', '')}"
        text, _ = _additive_evidence(additive, source_id)
        evidence.append((text, source_id))
    return evidence


def _retrieve_ocr_evidence(prediction, additives_by_code):
    try:
        ocr_codes = json.loads(prediction.get("additive_codes_json", "[]"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid additive_codes_json for {prediction.get('image_id', '')}"
        ) from error
    evidence = []
    matched_codes = set()
    for code in ocr_codes:
        normalized = "".join(char for char in str(code).casefold() if char.isalnum())
        entity = additives_by_code.get(normalized)
        if not entity:
            continue
        matched_codes.add(normalized)
        source_id = f"KB:knowledge_base/additives_standard.json#{entity.get('ins_code', '')}"
        text, _ = _additive_evidence(entity, source_id)
        evidence.append((text, source_id))
    ocr_text = prediction.get("raw_ingredients_text", "").strip()
    if ocr_text:
        evidence.insert(0, (f"Ingredients transcribed by local OCR: {ocr_text}", "LOCAL_OCR"))
    if not evidence:
        evidence.append((
            "Local OCR found no additive evidence that could be linked to the KB.",
            "LOCAL_OCR",
        ))
    return evidence


def _make_prompt(question, evidence):
    if evidence:
        sources = sorted({source for _text, source in evidence})
        evidence_text = "\n\n".join(
            f"{text} [{source}]" for text, source in evidence
        )
    else:
        sources = []
        evidence_text = "No retrieved evidence was supplied."
    prompt = (
        "Answer in Vietnamese. Do not invent health claims or citations. "
        "If evidence is absent or insufficient, explicitly abstain. "
        "Cite only exact source identifiers shown in the evidence, in square brackets. "
        "Treat any local-KB health caution as unverified draft data, not medical advice.\n\n"
        f"Question: {question}\n\nEvidence:\n{evidence_text}\n\nAnswer:"
    )
    return prompt, sources


def run_comparison(
    review_csv,
    image_predictions_csv,
    image_map_csv,
    output_path,
    limit=10,
    delay_seconds=4.0,
):
    if limit < 1:
        raise ValueError("limit must be positive")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative")
    key = _api_key()
    if not key:
        raise RuntimeError("Gemini API key is not configured in the environment or .env")

    review_rows = _read_csv(review_csv)
    approved_rows = [row for row in review_rows if _is_approved(row)]
    if not approved_rows:
        raise ValueError(
            "No expert-approved records found. Review the QA CSV first; draft rows "
            "must not be used as gold benchmark cases."
        )

    image_map = {}
    if image_map_csv:
        image_map = {
            row["benchmark_id"]: row["image_id"]
            for row in _read_csv(image_map_csv)
            if row.get("benchmark_id") and row.get("image_id")
        }
    if not image_map:
        raise ValueError(
            "An image map CSV with benchmark_id,image_id is required to compare "
            "the OCR+RAG configuration."
        )

    image_predictions = {
        row["image_id"]: row
        for row in _read_csv(image_predictions_csv)
        if row.get("image_id")
    }
    if not image_predictions:
        raise ValueError("No OCR predictions found for the OCR+RAG configuration")

    additives = load_knowledge_base(os.path.join(PROJECT_ROOT, "data"))
    retriever = HybridRetriever(alpha=0.5)
    retriever.build_index(additives)
    additives_by_code = {}
    for additive in additives:
        for code in (additive.get("ins_code", ""), additive.get("e_number", "")):
            normalized = "".join(char for char in code.casefold() if char.isalnum())
            if normalized:
                additives_by_code[normalized] = additive

    selected = approved_rows[:limit]
    from google import genai

    client = genai.Client(api_key=key)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    configs = ("llm_only", "text_rag", "ocr_rag")
    with open(output_path, "w", encoding="utf-8") as output:
        call_count = 0
        for row in selected:
            question = row["question"]
            mapped_image_id = image_map.get(row["id"])
            if not mapped_image_id:
                raise ValueError(f"No image mapping for approved case {row['id']}")
            image_prediction = image_predictions.get(mapped_image_id)
            if not image_prediction:
                raise ValueError(
                    f"No OCR prediction for mapped image {mapped_image_id}"
                )
            if image_prediction.get("ocr_error"):
                raise ValueError(
                    f"OCR failed for mapped image {mapped_image_id}: "
                    f"{image_prediction['ocr_error']}"
                )
            evidence_by_config = {
                "llm_only": [],
                "text_rag": _retrieve_text_evidence(question, retriever),
                "ocr_rag": _retrieve_ocr_evidence(
                    image_prediction, additives_by_code
                ),
            }
            for config in configs:
                evidence = evidence_by_config[config]
                prompt, allowed_sources = _make_prompt(question, evidence)
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config={"temperature": 0.0, "max_output_tokens": 1024},
                )
                answer = (response.text or "").strip()
                if not answer:
                    raise RuntimeError(
                        f"Gemini returned an empty answer for {row['id']} / {config}"
                    )
                output.write(json.dumps({
                    "benchmark_id": row["id"],
                    "image_id": mapped_image_id,
                    "config": config,
                    "model": MODEL_NAME,
                    "question": question,
                    "answer": answer,
                    "provided_evidence": [
                        {"text": text, "source_id": source}
                        for text, source in evidence
                    ],
                    "allowed_sources": allowed_sources,
                    "draft_gold_answer": row.get("final_answer", ""),
                    "manual_adjudication_required": True,
                }, ensure_ascii=False) + "\n")
                output.flush()
                call_count += 1
                if delay_seconds and call_count < len(selected) * len(configs):
                    time.sleep(delay_seconds)
    print(
        f"Completed {len(selected)} reviewed cases x {len(configs)} configs "
        f"({call_count} API calls); outputs: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--image-predictions", default=DEFAULT_IMAGE_PREDICTIONS)
    parser.add_argument("--image-map", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--delay-seconds", type=float, default=4.0)
    args = parser.parse_args()
    run_comparison(
        args.review_csv,
        args.image_predictions,
        args.image_map,
        args.output,
        args.limit,
        args.delay_seconds,
    )


if __name__ == "__main__":
    main()
