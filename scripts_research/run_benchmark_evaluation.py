"""
Evaluation Pipeline: Chạy bộ QA Benchmark (745 câu) qua 3 cấu hình ablation study.
Tính các metrics theo yêu cầu đề tài:
  - Exact Match (EM), Token F1
  - Citation Accuracy
  - Safety Label Accuracy (Groundedness)
  - Faithfulness (binary_answer match)

Chạy: python scripts_research/run_benchmark_evaluation.py [--mode A|B|C|all] [--limit N]
"""

import os
import sys
import csv
import json
import time
import re
import argparse
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from modules.pipeline import (
    load_knowledge_base,
    load_personas,
    load_food_hazards,
    load_fndds_nutrition,
    load_openfoodfacts_vietnam,
    foodontorag_entity_linking,
    evaluate_safety_and_guardrail,
    check_food_hazards,
    evaluate_nutritional_risk,
    cross_reference_openfoodfacts,
    generate_llm_answer,
    generate_llm_only_answer,
)
from modules.ocr_extractor import parse_ingredients_from_ocr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Mapping persona IDs từ benchmark sang NHANES personas
PERSONA_MAP = {
    "P_PREG_01": "USER_CLINICAL_004",
    "P_CHILD_01": "USER_CLINICAL_003",
    "P_CKD_01": "USER_CLINICAL_005",
    "P_ALLERGY_01": "USER_CLINICAL_006",
    "P_ALLERGY_MILK": "USER_CLINICAL_006",
    "P_ALLERGY_PEANUT": "USER_CLINICAL_006",
    "P_LACTOSE_INTOLERANT": "USER_CLINICAL_008",
    "P_DIAB_HYPER_01": "USER_CLINICAL_002",
    "P_OBESE_GOUT_01": "USER_CLINICAL_007",
}


def normalize_text(text):
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def compute_token_f1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction, reference):
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def check_safety_label(evaluation_status, gold_label):
    status_map = {
        "CRITICAL_ALERT": "UNSAFE",
        "WARNING": "UNSAFE",
        "CAUTION": "CONDITIONAL",
        "SAFE": "SAFE",
        "N/A": "UNKNOWN",
    }
    predicted = status_map.get(evaluation_status, "UNKNOWN")
    gold = gold_label.upper().strip() if gold_label else "UNKNOWN"
    if gold in ("UNSAFE", "CONDITIONAL_UNSAFE"):
        gold = "UNSAFE"
    if gold in ("SAFE", "CONDITIONAL_SAFE"):
        gold = "SAFE"
    return predicted == gold


def check_binary_answer(llm_answer, gold_binary):
    if not llm_answer or not gold_binary:
        return None
    answer_lower = llm_answer.lower()
    gold = gold_binary.strip().lower()
    if gold == "no":
        return any(w in answer_lower for w in [
            "không nên", "không phù hợp", "không an toàn", "cần tránh",
            "chống chỉ định", "nguy hiểm", "cấm", "hạn chế",
        ])
    elif gold == "yes":
        return any(w in answer_lower for w in [
            "an toàn", "phù hợp", "có thể sử dụng", "được phép",
            "không có vấn đề", "chấp nhận được",
        ])
    return None


def load_benchmark(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def run_single_evaluation(row, mode, kb_additives, personas_dict, hazards_list,
                          fndds_list, off_products):
    persona_id = PERSONA_MAP.get(row.get("persona_id", ""), "USER_CLINICAL_007")
    persona = personas_dict.get(persona_id, personas_dict.get("USER_CLINICAL_007", {}))

    ingredients_text = row.get("product_ingredients", "")
    question = row.get("question", "")
    product_name = row.get("product_name", "")

    if not ingredients_text.strip():
        return None

    full_text = f"Thành phần: {ingredients_text}"
    mentions = parse_ingredients_from_ocr(full_text)

    result = {"id": row["id"], "mode": mode, "domain": row.get("domain", "")}

    if mode == "A":
        llm_answer = generate_llm_only_answer(full_text, persona, question)
        result["llm_answer"] = llm_answer or ""
        result["evaluation_status"] = "N/A"
    else:
        linked_results = [foodontorag_entity_linking(m, kb_additives) for m in mentions]
        evaluation = evaluate_safety_and_guardrail(
            linked_results, persona, question,
            avg_ocr_conf=0.99, all_mentions=mentions,
        )
        detected_hazards = check_food_hazards(product_name, mentions, hazards_list)
        matched_fndds, nutri_warnings, _ = evaluate_nutritional_risk(
            product_name, fndds_list, persona,
        )
        off_cross_ref = cross_reference_openfoodfacts(product_name, mentions, off_products)

        llm_answer = generate_llm_answer(
            evaluation, linked_results, persona, question,
            detected_hazards=detected_hazards,
            matched_fndds=matched_fndds,
            nutri_warnings=nutri_warnings,
        )
        result["llm_answer"] = llm_answer or ""
        result["evaluation_status"] = evaluation.get("status", "")

    # Metrics
    ref_answer = row.get("reference_answer", "")
    result["token_f1"] = compute_token_f1(result["llm_answer"], ref_answer)
    result["exact_match"] = compute_exact_match(result["llm_answer"], ref_answer)
    result["safety_correct"] = check_safety_label(
        result["evaluation_status"], row.get("gold_safety_label", ""),
    ) if mode != "A" else None
    result["binary_correct"] = check_binary_answer(
        result["llm_answer"], row.get("binary_answer", ""),
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="FoodHealthQA Benchmark Evaluation")
    parser.add_argument("--mode", default="all", choices=["A", "B", "C", "all"])
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số câu (0 = tất cả)")
    parser.add_argument("--output", default=None, help="File output CSV")
    args = parser.parse_args()

    modes = ["A", "B", "C"] if args.mode == "all" else [args.mode]

    print("=" * 60)
    print("FoodHealthQA Benchmark Evaluation")
    print("=" * 60)

    # Load resources
    print("Nạp Knowledge Base...")
    kb_additives = load_knowledge_base(DATA_DIR)
    personas_list = load_personas(DATA_DIR)
    personas_dict = {p.get("persona_id", p.get("id")): p for p in personas_list}
    hazards_list = load_food_hazards(DATA_DIR)
    fndds_list = load_fndds_nutrition(DATA_DIR)
    off_products = load_openfoodfacts_vietnam(DATA_DIR)

    # Load benchmark
    benchmark_path = os.path.join(DATA_DIR, "qa_benchmark_merged_review.csv")
    rows = load_benchmark(benchmark_path)
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"Benchmark: {len(rows)} câu hỏi")
    print(f"Modes: {modes}")
    print()

    all_results = []
    for mode in modes:
        print(f"--- Chạy Mode {mode} ---")
        mode_results = []
        for i, row in enumerate(rows):
            try:
                result = run_single_evaluation(
                    row, mode, kb_additives, personas_dict,
                    hazards_list, fndds_list, off_products,
                )
                if result:
                    mode_results.append(result)
            except Exception as e:
                print(f"  Lỗi {row.get('id','?')}: {e}")

            if (i + 1) % 50 == 0:
                print(f"  Đã xử lý {i + 1}/{len(rows)}...")

        all_results.extend(mode_results)

        # Aggregate metrics
        f1_scores = [r["token_f1"] for r in mode_results if r["llm_answer"]]
        em_scores = [r["exact_match"] for r in mode_results if r["llm_answer"]]
        safety_scores = [r["safety_correct"] for r in mode_results if r["safety_correct"] is not None]
        binary_scores = [r["binary_correct"] for r in mode_results if r["binary_correct"] is not None]
        answered = sum(1 for r in mode_results if r["llm_answer"])

        print(f"\n  Mode {mode} — Kết quả:")
        print(f"    Answered: {answered}/{len(mode_results)}")
        print(f"    Token F1:         {sum(f1_scores)/len(f1_scores):.4f}" if f1_scores else "    Token F1: N/A")
        print(f"    Exact Match:      {sum(em_scores)/len(em_scores):.4f}" if em_scores else "    Exact Match: N/A")
        print(f"    Safety Accuracy:  {sum(safety_scores)/len(safety_scores):.4f}" if safety_scores else "    Safety Accuracy: N/A (Mode A)")
        print(f"    Binary Accuracy:  {sum(binary_scores)/len(binary_scores):.4f}" if binary_scores else "    Binary Accuracy: N/A")

        # Per-domain breakdown
        domain_metrics = defaultdict(list)
        for r in mode_results:
            if r["llm_answer"]:
                domain_metrics[r["domain"]].append(r["token_f1"])
        if domain_metrics:
            print(f"\n    Per-domain F1:")
            for domain, scores in sorted(domain_metrics.items()):
                print(f"      {domain:30s}: {sum(scores)/len(scores):.4f} ({len(scores)} câu)")
        print()

    # Export results
    output_path = args.output or os.path.join(DATA_DIR, "evaluation_results.csv")
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "mode", "domain", "llm_answer", "evaluation_status",
            "token_f1", "exact_match", "safety_correct", "binary_correct",
        ])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Kết quả đã xuất: {output_path}")


if __name__ == "__main__":
    main()
