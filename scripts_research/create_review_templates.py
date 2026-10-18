"""Create ground-truth and human-review CSV templates without inventing labels."""

import csv
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.benchmark import load_benchmark


DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def _write_csv(
    path,
    fieldnames,
    rows,
    key_field=None,
    preserve_fields=None,
    key_normalizer=None,
    preserve_unmatched_rows=True,
):
    rows = [dict(row) for row in rows]
    if key_field and os.path.isfile(path):
        with open(path, "r", encoding="utf-8-sig", newline="") as source:
            existing_rows = list(csv.DictReader(source))
        normalize_key = key_normalizer or (lambda value: value)
        old_by_key = {
            normalize_key(row.get(key_field, "")): row
            for row in existing_rows
            if row.get(key_field, "")
        }
        new_keys = set()
        for row in rows:
            row_key = normalize_key(row.get(key_field, ""))
            new_keys.add(row_key)
            old_row = old_by_key.get(row_key)
            if old_row:
                fields_to_preserve = preserve_fields or fieldnames
                for field in fields_to_preserve:
                    if field in old_row:
                        row[field] = old_row[field]
        if preserve_unmatched_rows:
            rows.extend(
                row for row in existing_rows
                if row.get(key_field, "")
                and normalize_key(row[key_field]) not in new_keys
                and (
                    preserve_unmatched_rows is True
                    or (
                        preserve_unmatched_rows == "annotated"
                        and any(row.get(field) for field in (preserve_fields or []))
                    )
                )
            )
    with open(path, "w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def create_image_benchmark_mapping_template():
    fields = [
        "benchmark_id",
        "image_id",
        "product_group_id",
        "split",
        "mapping_review_status",
        "notes",
    ]
    target = os.path.join(DATA_DIR, "image_benchmark_mapping_template.csv")
    if not os.path.isfile(target):
        _write_csv(target, fields, [])
    return target


def create_qa_review_template():
    benchmark = load_benchmark(
        os.path.join(DATA_DIR, "qa_benchmark_600_source_grounded.jsonl")
    )
    fields = [
        "id",
        "domain",
        "question",
        "draft_answer",
        "source_ids_json",
        "source_urls_json",
        "evidence_excerpt",
        "question_group_id",
        "review_status",
        "reviewer_id",
        "reviewer_qualification",
        "answer_supported_by_source",
        "citation_correct",
        "safety_label_correct",
        "final_answer",
        "review_notes",
        "review_date",
    ]
    rows = []
    for record in benchmark:
        rows.append({
            "id": record["id"],
            "domain": record["domain"],
            "question": record["question"],
            "draft_answer": record["reference_answer"],
            "source_ids_json": json.dumps(record["source_ids"], ensure_ascii=False),
            "source_urls_json": json.dumps(record["source_urls"], ensure_ascii=False),
            "evidence_excerpt": record.get("evidence_excerpt", ""),
            "question_group_id": record.get("question_group_id", ""),
            "review_status": "pending",
        })
    target = os.path.join(DATA_DIR, "qa_benchmark_600_source_review.csv")
    _write_csv(
        target,
        fields,
        rows,
        key_field="id",
        preserve_fields=[
            "review_status",
            "reviewer_id",
            "reviewer_qualification",
            "answer_supported_by_source",
            "citation_correct",
            "safety_label_correct",
            "final_answer",
            "review_notes",
            "review_date",
        ],
    )
    return len(rows), target


def create_response_adjudication_template():
    fields = [
        "case_id",
        "benchmark_id",
        "system_config",
        "question",
        "model_answer",
        "provided_evidence_json",
        "citations_in_answer_json",
        "reviewer_id",
        "reviewer_qualification",
        "groundedness_0_2",
        "faithfulness_0_2",
        "citation_entailment_0_2",
        "safety_0_2",
        "abstention_appropriate",
        "expert_referral_required",
        "hallucinated_claims",
        "missed_risks",
        "review_notes",
        "review_date",
        "adjudication_status",
    ]
    target = os.path.join(DATA_DIR, "answer_adjudication_template.csv")
    if not os.path.isfile(target):
        _write_csv(target, fields, [])
    return target


def create_knowledge_base_review_template():
    kb_path = os.path.join(PROJECT_ROOT, "knowledge_base", "additives_standard.json")
    with open(kb_path, "r", encoding="utf-8") as source:
        additives = json.load(source)
    fields = [
        "claim_id",
        "entity_id",
        "entity_type",
        "claim_field",
        "claim_value",
        "source_id",
        "source_url",
        "source_locator",
        "verification_status",
        "runtime_use",
        "reviewer_id",
        "review_notes",
    ]
    rows = []
    for additive in additives:
        ins_code = additive.get("ins_code", "")
        claim_specs = [
            ("identity_and_function", {
                "name_vi": additive.get("canonical_name_vi", ""),
                "name_en": additive.get("canonical_name_en", ""),
                "function": additive.get("function", ""),
                "ins_code": ins_code,
                "e_number": additive.get("e_number", ""),
            }, "FOODON", "https://foodon.org/", "Identity and ontology mapping; validate individual CURIE."),
            ("adi_assessment", additive.get("adi_assessment", {}), "", "", ""),
            ("vietnam_regulation", additive.get("regulatory_claim", {}), "VN_TT24_2019", "", "Official consolidated text and additive/food category annex need verification."),
            ("codex_gsfa", additive.get("codex_claim", {}), "CODEX_GSFA", "https://www.fao.org/gsfaonline/", "Check exact INS, food category, technological function and ML/GMP."),
            ("allergen_assessment", additive.get("allergen_assessment", {}), "CODEX_CXS_1_1985", "https://www.fao.org/input/download/standards/32/CXS_001e.pdf", "An empty group list is not evidence of absence."),
            ("efsa_opinion", additive.get("efsa_opinion") or {}, "", "", ""),
        ]
        for claim_field, value, source_id, source_url, locator in claim_specs:
            source_id = value.get("source_id", source_id) if isinstance(value, dict) else source_id
            source_url = value.get("source_url", source_url) if isinstance(value, dict) else source_url
            locator = value.get("source_locator", locator) if isinstance(value, dict) else locator
            status = (
                value.get("verification_status", "pending_primary_source_check")
                if isinstance(value, dict)
                else "pending_primary_source_check"
            )
            runtime_use = (
                "limited_source_checked_claim"
                if status == "source_checked"
                else "disabled_until_primary_source_check"
            )
            if (
                claim_field == "adi_assessment"
                and isinstance(value, dict)
                and not value.get("use_for_numeric_compliance")
                and status == "source_checked"
            ):
                runtime_use = "display_only_non_numeric_or_group_claim"
            rows.append({
                "claim_id": f"{ins_code}:{claim_field}",
                "entity_id": ins_code,
                "entity_type": "additive",
                "claim_field": claim_field,
                "claim_value": json.dumps(value, ensure_ascii=False),
                "source_id": source_id,
                "source_url": source_url,
                "source_locator": locator,
                "verification_status": status,
                "runtime_use": runtime_use,
            })
        for index, claim in enumerate(additive.get("unverified_legacy_health_claims", []), 1):
            rows.append({
                "claim_id": f"{ins_code}:legacy_health_claim:{index}",
                "entity_id": ins_code,
                "entity_type": "additive",
                "claim_field": f"unverified_health_claim_{index}",
                "claim_value": json.dumps(claim, ensure_ascii=False),
                "source_id": "",
                "source_url": "",
                "source_locator": "",
                "verification_status": "pending_primary_source_check",
                "runtime_use": "disabled_until_primary_source_check",
            })
    allergen_path = os.path.join(PROJECT_ROOT, "knowledge_base", "allergen_groups.json")
    with open(allergen_path, "r", encoding="utf-8") as source:
        allergen_groups = json.load(source)
    for group in allergen_groups:
        rows.append({
            "claim_id": f"allergen:{group['id']}:classification",
            "entity_id": group["id"],
            "entity_type": "allergen_label_group",
            "claim_field": "classification_and_label_scope",
            "claim_value": json.dumps(group, ensure_ascii=False),
            "source_id": ",".join(group.get("source_ids", [])),
            "source_url": ",".join(group.get("source_urls", [])),
            "source_locator": "Codex CXS 1-1985 section 4.2.1.4; EU 1169/2011 Annex II; FDA source as applicable.",
            "verification_status": group.get("verification_status", "pending_primary_source_check"),
            "runtime_use": (
                "screening_only" if group.get("enabled_for_allergen_screening")
                else "disabled"
            ),
        })
    target = os.path.join(DATA_DIR, "knowledge_base_source_review_template.csv")
    _write_csv(
        target,
        fields,
        rows,
        key_field="claim_id",
        preserve_fields=[
            "reviewer_id",
            "review_notes",
        ],
        key_normalizer=_normalize_claim_review_id,
        preserve_unmatched_rows="annotated",
    )
    return len(rows), target


def _normalize_claim_review_id(claim_id):
    """Map old claim IDs to the current schema so existing notes follow the claim."""
    claim_id = str(claim_id or "")
    code, separator, suffix = claim_id.partition(":")
    if not separator:
        return claim_id.casefold()
    suffix_lower = suffix.casefold()
    if suffix_lower == "adi_jecfa_efsa":
        suffix = "adi_assessment"
    elif suffix_lower == "codex_standard":
        suffix = "codex_gsfa"
    elif suffix_lower == "efsa_opinion":
        suffix = "efsa_opinion"
    else:
        match = re.fullmatch(r"risk:(\d+)", suffix_lower)
        if match:
            suffix = f"legacy_health_claim:{match.group(1)}"
    return f"{code}:{suffix}".casefold()


def main():
    image_map_path = create_image_benchmark_mapping_template()
    qa_count, qa_path = create_qa_review_template()
    adjudication_path = create_response_adjudication_template()
    claim_count, source_review_path = create_knowledge_base_review_template()
    print(f"QA-to-image mapping template -> {image_map_path}")
    print(f"QA review rows: {qa_count} -> {qa_path}")
    print(f"Response adjudication template -> {adjudication_path}")
    print(f"Knowledge-base claims for review: {claim_count} -> {source_review_path}")


if __name__ == "__main__":
    main()
