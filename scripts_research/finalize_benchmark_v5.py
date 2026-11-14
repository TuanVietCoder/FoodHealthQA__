"""Finalize v5: run the answer verifier, assign eligibility, write the consumer-compatible active file.

Pipeline:  create_benchmark_v5.py -> verify gates (verify_v5_answers.check) -> this script.

Outputs
  data/qa_benchmark.jsonl / .csv   consumer-compatible (gold_safety_label in the v4 label set)
  data/build/qa_verification.csv      per-record failed gates
  data/build/qa_activation_report.json  dataset-level activation gates

Only records that pass every gate AND sit in tier core/silver get review_status
"approved_rule_verified". Everything else is kept for audit but is not scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_research"))

import verify_v5_answers as ver  # noqa: E402

SRC = ROOT / "data" / "build" / "qa_build_full.jsonl"
OUT_DIR = ROOT / "data" / "archive" / "master_benchmark_2311"
OUT_JSONL = OUT_DIR / "qa_benchmark.jsonl"
OUT_CSV = OUT_DIR / "qa_benchmark.csv"
REPORT = ROOT / "data" / "build" / "qa_activation_report.json"
MANIFEST = ROOT / "data" / "benchmark_manifest.json"

# v5 fine-grained label -> label set understood by app.py / modules/benchmark.py / evaluation.py
EVAL_LABEL = {
    "SAFE": "SAFE", "PERMITTED_LISTED": "SAFE", "CAUTION": "CAUTION", "NOT_DECLARED": "CAUTION",
    "UNSAFE": "UNSAFE", "CRITICAL_ALERT": "CRITICAL_ALERT", "ABSTAIN": "ABSTAIN",
}
EVAL_BINARY = {"SAFE": "Yes", "ABSTAIN": "Abstain"}  # everything else -> "No"
ELIGIBLE_TIERS = {"core", "silver"}


def verification_level(row: dict) -> str:
    claims = [c for c in (row.get("external_claims") or "").split(";") if c]
    if claims:
        return "label_and_primary_source_text_verified"
    if row["domain"] in ("allergen_detection", "insufficient_data", "additive_regulation"):
        return "label_only_no_regulatory_assertion"
    return "not_verified"


def exclusion_reason(row: dict, fails: list[str]) -> str:
    if row["tier"] == "excluded_kb_dependent":
        return "KB_DEPENDENT_ABSTAIN"
    if row["tier"] == "excluded_source_not_verifiable":
        gap = row.get("source_gap", "")
        if gap.startswith("benzene_source_covers_beverages_only"):
            return "BENZENE_SOURCE_COVERS_BEVERAGES_ONLY"
        if gap.startswith("nutrient_threshold_basis_not_stated_on_label"):
            return "NUTRIENT_THRESHOLD_BASIS_NOT_STATED_ON_LABEL"
        return "SOURCE_NOT_OBTAINABLE:" + gap
    if row["tier"] == "excluded_clinical_judgment":
        if row["persona_id"] == "diabetes_hypertension":
            return "NUTRIENT_THRESHOLD_BASIS_NOT_STATED_ON_LABEL"
        return "CLINICAL_CLAIM_NOT_VERIFIABLE_OFFLINE"
    if fails:
        return "FAILED_VERIFICATION:" + ",".join(f.split(":")[0] + ":" + f.split(":")[1] for f in fails)
    return ""


def question_template(question: str) -> str:
    """Question text with the product name and INS codes masked, so records asking the same thing share one id."""
    import re
    q = re.sub(r"'[^']*'", "'X'", question or "")
    q = re.sub(r"INS\w*(\([^)]*\))?(, ?INS\w*(\([^)]*\))?)*", "INS", q)
    q = re.sub(r"[\d]+([,\.]\d+)?", "N", q)
    return q.strip()


def add_group_fields(rows: list[dict]) -> None:
    """split_group: records of one photo share a group (split train/test and bootstrap by it).
    template_id: stable id of the question template inside its domain (report/weight by it)."""
    ids: dict[tuple, str] = {}
    for r in rows:
        img = Path(str(r.get("image_ref") or "")).stem
        r["split_group"] = img or r["id"]
        key = (r.get("domain", ""), question_template(r["question"]))
        if key not in ids:
            ids[key] = f"T{len(ids) + 1:03d}"
        r["template_id"] = ids[key]


def add_claim_pages(rows: list[dict]) -> None:
    """source_claim_pages: PDF page(s) (1-based page index of the downloaded file) where each external claim's
    evidence was found, e.g. 'CXS1_4217_SULPHITE_DECLARATION: p.7'. HTML sources and parsed-table claims have none."""
    claims = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "external_claim_verification.json").read_text(encoding="utf-8"))
    for r in rows:
        parts = []
        for cid in [c for c in (r.get("external_claims") or "").split(";") if c]:
            pages = sorted({p for v in (claims.get(cid, {}).get("evidence_pages") or {}).values()
                            for p in (v if isinstance(v, list) else [v])})
            if pages:
                parts.append(f"{cid}: p." + ",".join(map(str, pages)))
        r["source_claim_pages"] = "; ".join(parts)


def finalize(rows: list[dict]):
    ctx = ver.Context()
    out, ver_rows = [], []
    for r in rows:
        fails = ver.check(r, ctx)
        eligible = r["tier"] in ELIGIBLE_TIERS and not fails
        v = dict(r)
        # expose the verified v5 text under the names consumers read
        v["question"], v["reference_answer"] = r["question_v5"], r["reference_answer_v5"]
        v["gold_safety_label"] = EVAL_LABEL.get(r["label_v5"], "")
        v["binary_answer"] = EVAL_BINARY.get(v["gold_safety_label"], "No")
        v["source_citations"] = r["source_citations_v5"]
        v["source_claim_locators"] = r["source_claim_locators_v5"]
        v["verify_failed_gates"] = ";".join(fails)
        v["source_verification_level"] = verification_level(r)
        v["exclusion_reason"] = "" if eligible else exclusion_reason(r, fails)
        v["review_status"] = "approved_rule_verified" if eligible else "not_for_scoring"
        v["reviewer_id"] = "RULE_VERIFIER_V5_NO_EXPERT"
        v["review_method"] = "rule_based_gates_G1_G6_plus_label_text_adjudication"
        v["answer_correct"] = v["citation_correct"] = v["safety_label_correct"] = "yes" if eligible else ""
        v["eligible_for_scoring"] = "yes" if eligible else "no"
        out.append(v)
        ver_rows.append({"id": r["id"], "tier": r["tier"], "eligible": v["eligible_for_scoring"],
                         "failed_gates": v["verify_failed_gates"], "exclusion_reason": v["exclusion_reason"]})
    add_group_fields(out)
    add_claim_pages(out)
    return out, ver_rows


def activation_gates(out: list[dict]) -> dict:
    elig = [r for r in out if r["eligible_for_scoring"] == "yes"]
    gates = {}
    gates["D1_every_eligible_record_passes_all_gates"] = all(not r["verify_failed_gates"] for r in elig)
    gates["D2_no_unresolved_quarantine_or_needs_review"] = not any(
        r["tier"] == "quarantine" or r["label_v5"] == "NEEDS_REVIEW" for r in elig)
    gates["D3_ids_and_questions_unique"] = (len({r["id"] for r in out}) == len(out)
                                            and len({r["question"] for r in elig}) == len(elig))
    gates["D4_every_eligible_label_in_consumer_label_set"] = all(
        r["gold_safety_label"] in {"SAFE", "CAUTION", "UNSAFE", "CRITICAL_ALERT", "ABSTAIN"} for r in elig)
    counts = Counter(r["gold_safety_label"] for r in elig)
    gates["D5_each_label_has_min_30_eligible"] = all(counts[k] >= 30 for k in
                                                    ("SAFE", "CAUTION", "UNSAFE", "CRITICAL_ALERT", "ABSTAIN"))
    gates["D6_eligible_at_least_1000"] = len(elig) >= 1000
    ext = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "external_claim_verification.json").read_text(encoding="utf-8"))
    gates["D8_all_external_claims_passed"] = all(c["passed"] for c in ext.values())
    gates["D9_scored_records_cite_only_verified_tags"] = all(
        set(t for t in r["source_citations"].split(", ") if t) <= ver.sg.VERIFIED_TAGS for r in elig)
    gates["D7_no_expert_claim"] = all(r["expert_validated"] == "false" for r in out)
    return {"gates": gates, "all_pass": all(gates.values()), "eligible": len(elig),
            "eligible_by_label": dict(counts), "eligible_by_domain": dict(Counter(r["domain"] for r in elig)),
            "eligible_by_verification_level": dict(Counter(r["source_verification_level"] for r in elig)),
            "not_scored_by_reason": dict(Counter(r["exclusion_reason"].split(":")[0] for r in out
                                                 if r["eligible_for_scoring"] == "no"))}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activate", action="store_true",
                    help="set active_version=v5 in benchmark_manifest.json (only if every activation gate passes)")
    args = ap.parse_args()
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    out, ver_rows = finalize(rows)
    OUT_JSONL.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    write_csv(OUT_CSV, out)
    write_csv(ver.OUT_CSV, ver_rows)
    report = activation_gates(out)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    meta = manifest["benchmarks"]["v5"]
    meta.update({"csv": OUT_CSV.name, "jsonl": OUT_JSONL.name, "record_count": len(out),
                 "scored_record_count": report["eligible"], "status": "rule_verified_tiered_no_expert",
                 "activation_report": "build/" + REPORT.name, "full_audit_file": "build/" + SRC.name,
                 "fix_list": "build/qa_fix_list.csv", "expert_validated": False,
                 "tier_distribution": dict(Counter(r["tier"] for r in out)),
                 "label_v5_distribution": dict(Counter(r["label_v5"] for r in out)),
                 "gold_safety_label_distribution": dict(Counter(r["gold_safety_label"] for r in out)),
                 "split_group_field": "split_group (one photo = one group; split and bootstrap by it)",
                 "template_id_field": "template_id (question template within domain; report per template)",
                 "tier_usage": {"core": "primary comparison score",
                                "silver": "secondary, reported separately (authority-flag / threshold answers; includes all vulnerable_groups)"},
                 "report_script": "scripts_research/benchmark_report.py"})
    meta.pop("activation_note", None)
    # one standard dataset: v3/v4 were deleted (recoverable from git commit d1612d35); only v5 is listed
    manifest["benchmarks"] = {"v5": meta}
    if args.activate:
        if not report["all_pass"]:
            sys.exit("Activation refused: not every activation gate passed.")
        manifest["active_version"] = "v5"
        print("active_version -> v5")
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
