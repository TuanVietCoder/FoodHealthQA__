"""
Comprehensive Ablation Study Suite for FoodHealthQA (Giai đoạn 3).

Supports:
1. Embedding Model Ablation: BM25 vs TF-IDF vs SBERT vs Hybrid
2. Top-k Tuning: k in (1, 3, 5, 10)
3. Knowledge Source Ablation: Leave-One-Out (Additives, FoodOn, Hazards, FNDDS, OFF)
4. Prompt Template Ablation: Standard, Zero-shot, Few-shot, CoT, Strict
5. Error Propagation Study: Config B (Ground Truth) vs Config C (OCR)
"""

import copy
import re
import unicodedata
from collections import Counter, defaultdict

from modules.hybrid_retriever import HybridRetriever


_RETRIEVER_ALPHAS = {
    "bm25": 1.0,
    "tfidf": 0.0,
    "hybrid": 0.5,
}


def run_retrieval_ablation(records, additives, top_k_values=(1, 3, 5, 10), models=None):
    """
    Measure entity retrieval hit rate and reciprocal rank across top-k/retriever settings.

    Records require `question` and `expected_ins_code`; output can be written as JSON
    by the CLI or consumed by tests and notebooks.
    """
    if not records:
        raise ValueError("At least one labeled benchmark record is required")
    if not additives:
        raise ValueError("The additive knowledge base is empty")
    if any(k < 1 for k in top_k_values):
        raise ValueError("top_k values must be positive integers")
    for index, record in enumerate(records):
        if not record.get("question") or not record.get("expected_ins_code"):
            raise ValueError(f"Record {index} needs question and expected_ins_code")

    target_models = models or _RETRIEVER_ALPHAS

    results = []
    for retriever_name, alpha in target_models.items():
        retriever = HybridRetriever(alpha=alpha)
        retriever.build_index(additives)
        for top_k in top_k_values:
            hits = 0
            reciprocal_ranks = []
            for record in records:
                candidates = retriever.retrieve(record["question"], top_k=top_k)
                expected = str(record["expected_ins_code"]).casefold()
                rank = next(
                    (
                        position
                        for position, (candidate, _, _) in enumerate(candidates, 1)
                        if str(candidate.get("ins_code", "")).casefold() == expected
                    ),
                    None,
                )
                hits += rank is not None
                reciprocal_ranks.append(1 / rank if rank is not None else 0.0)
            results.append({
                "retriever": retriever_name,
                "top_k": top_k,
                "queries": len(records),
                "hit_rate": hits / len(records),
                "mean_reciprocal_rank": sum(reciprocal_ranks) / len(records),
            })
    return results


def run_embedding_ablation(records, additives, top_k=5):
    """
    So sánh các mô hình biểu diễn truy xuất:
    - bm25_only (Lexical, alpha=1.0)
    - tfidf_only (Sparse Semantic, alpha=0.0)
    - hybrid (BM25 + TFIDF, alpha=0.5)
    - sbert / hybrid_sbert (Dense Semantic nếu SentenceTransformer khả dụng)
    """
    configs = {
        "bm25_only": 1.0,
        "tfidf_only": 0.0,
        "hybrid_bm25_tfidf": 0.5,
    }
    return run_retrieval_ablation(records, additives, top_k_values=(top_k,), models=configs)


def run_topk_ablation(records, additives, top_k_values=(1, 3, 5, 10), model_name="hybrid"):
    """
    Ablation khảo sát độ nhạy của top-k trên retriever được chỉ định.
    """
    alpha = _RETRIEVER_ALPHAS.get(model_name, 0.5)
    return run_retrieval_ablation(records, additives, top_k_values=top_k_values, models={model_name: alpha})


def run_knowledge_source_ablation(benchmark_rows, kb_additives, personas_dict,
                                  hazards_list, fndds_list, off_products,
                                  excluded_sources=None):
    """
    Ablation loại bỏ từng nguồn tri thức (Leave-One-Out).
    excluded_sources: tập hợp các nguồn muốn tắt, ví dụ: {'food_hazards', 'fndds_nutrition'}
    """
    from modules.pipeline import (
        foodontorag_entity_linking,
        evaluate_safety_and_guardrail,
        check_food_hazards,
        evaluate_nutritional_risk,
        cross_reference_openfoodfacts,
    )
    from modules.ocr_extractor import parse_ingredients_from_ocr

    excluded = set(excluded_sources or [])
    results = []

    active_additives = [] if "additives_standard" in excluded else kb_additives
    active_hazards = [] if "food_hazards" in excluded else hazards_list
    active_fndds = [] if "fndds_nutrition" in excluded else fndds_list
    active_off = [] if "openfoodfacts" in excluded else off_products

    for row in benchmark_rows:
        persona_id = row.get("persona_id", "USER_CLINICAL_007")
        persona = personas_dict.get(persona_id, personas_dict.get("USER_CLINICAL_007", {}))
        ingredients_text = row.get("product_ingredients", "")
        product_name = row.get("product_name", "")
        question = row.get("question", "")
        if not ingredients_text.strip():
            continue

        full_text = f"Thành phần: {ingredients_text}"
        mentions = parse_ingredients_from_ocr(full_text)

        linked_results = [foodontorag_entity_linking(m, active_additives) for m in mentions]
        evaluation = evaluate_safety_and_guardrail(
            linked_results, persona, question, avg_ocr_conf=0.99, all_mentions=mentions,
        )
        detected_hazards = check_food_hazards(product_name, mentions, active_hazards)
        matched_fndds, nutri_warnings, _ = evaluate_nutritional_risk(
            product_name, active_fndds, persona,
        )
        off_cross_ref = cross_reference_openfoodfacts(product_name, mentions, active_off)

        status = evaluation.get("status", "UNKNOWN")
        gold_safety = row.get("gold_safety_label", "").upper().strip()
        status_map = {
            "CRITICAL_ALERT": "UNSAFE",
            "WARNING": "UNSAFE",
            "CAUTION": "SAFE",
            "SAFE": "SAFE",
            "N/A": "UNKNOWN",
        }
        pred_label = status_map.get(status, "UNKNOWN")
        safety_match = (pred_label == gold_safety) if gold_safety in ("SAFE", "UNSAFE") else None

        results.append({
            "id": row.get("id"),
            "domain": row.get("domain"),
            "status": status,
            "safety_match": safety_match,
            "hazards_detected_count": len(detected_hazards),
            "nutrition_matched": bool(matched_fndds),
        })

    valid_matches = [r["safety_match"] for r in results if r["safety_match"] is not None]
    acc = sum(valid_matches) / len(valid_matches) if valid_matches else 0.0
    return {
        "excluded_sources": list(excluded),
        "total_evaluated": len(results),
        "safety_accuracy": acc,
        "detail_records": results,
    }


def run_error_propagation_study(benchmark_rows, kb_additives, personas_dict,
                                hazards_list, fndds_list, off_products,
                                noise_level=0.15):
    """
    Đo lường Lan truyền Sai số: So sánh Cấu hình B (Ground Truth) vs Cấu hình C (OCR).
    - Cấu hình B: đầu vào là Ground Truth `product_ingredients` chuẩn.
    - Cấu hình C: đầu vào có nhiễu OCR (ký tự bị méo, thiếu dấu, nhầm số INS).
    """
    import random
    from modules.pipeline import (
        foodontorag_entity_linking,
        evaluate_safety_and_guardrail,
    )
    from modules.ocr_extractor import parse_ingredients_from_ocr, post_process_ocr_spelling

    rng = random.Random(42)
    b_records = []
    c_records = []

    def _simulate_ocr_noise(text, p=0.15):
        tokens = text.split()
        noisy = []
        for token in tokens:
            if rng.random() < p:
                sub_choice = rng.choice(["drop_accent", "ins_confuse", "typo"])
                if sub_choice == "ins_confuse" and "500" in token:
                    noisy.append(token.replace("500", "50O"))
                elif sub_choice == "drop_accent":
                    noisy.append("".join(c for c in unicodedata.normalize("NFD", token) if not unicodedata.combining(c)))
                else:
                    noisy.append(token[:-1] if len(token) > 2 else token)
            else:
                noisy.append(token)
        return " ".join(noisy)

    for row in benchmark_rows:
        persona_id = row.get("persona_id", "USER_CLINICAL_007")
        persona = personas_dict.get(persona_id, personas_dict.get("USER_CLINICAL_007", {}))
        gt_text = row.get("product_ingredients", "")
        if not gt_text.strip():
            continue

        # Cấu hình B: Ground Truth
        b_mentions = parse_ingredients_from_ocr(f"Thành phần: {gt_text}")
        b_linked = [foodontorag_entity_linking(m, kb_additives) for m in b_mentions]
        b_eval = evaluate_safety_and_guardrail(b_linked, persona, row.get("question", ""), avg_ocr_conf=1.0, all_mentions=b_mentions)

        # Cấu hình C: Noisy OCR -> Post-OCR Correction -> Linking -> Guardrail
        ocr_noisy = _simulate_ocr_noise(gt_text, p=noise_level)
        corrected_ocr = post_process_ocr_spelling(ocr_noisy)
        c_mentions = parse_ingredients_from_ocr(f"Thành phần: {corrected_ocr}")
        c_linked = [foodontorag_entity_linking(m, kb_additives) for m in c_mentions]
        c_eval = evaluate_safety_and_guardrail(c_linked, persona, row.get("question", ""), avg_ocr_conf=0.88, all_mentions=c_mentions)

        gold_label = row.get("gold_safety_label", "SAFE").upper().strip()
        status_map = {"CRITICAL_ALERT": "UNSAFE", "WARNING": "UNSAFE", "CAUTION": "SAFE", "SAFE": "SAFE", "N/A": "UNKNOWN"}

        b_pred = status_map.get(b_eval.get("status"), "UNKNOWN")
        c_pred = status_map.get(c_eval.get("status"), "UNKNOWN")

        b_records.append({
            "id": row.get("id"),
            "mentions_count": len(b_mentions),
            "linked_count": sum(1 for x in b_linked if x.get("matched_entity")),
            "safety_correct": (b_pred == gold_label) if gold_label in ("SAFE", "UNSAFE") else None,
            "status": b_eval.get("status"),
        })
        c_records.append({
            "id": row.get("id"),
            "mentions_count": len(c_mentions),
            "linked_count": sum(1 for x in c_linked if x.get("matched_entity")),
            "safety_correct": (c_pred == gold_label) if gold_label in ("SAFE", "UNSAFE") else None,
            "status": c_eval.get("status"),
        })

    b_acc_list = [r["safety_correct"] for r in b_records if r["safety_correct"] is not None]
    c_acc_list = [r["safety_correct"] for r in c_records if r["safety_correct"] is not None]

    b_acc = sum(b_acc_list) / len(b_acc_list) if b_acc_list else 0.0
    c_acc = sum(c_acc_list) / len(c_acc_list) if c_acc_list else 0.0

    b_linked_avg = sum(r["linked_count"] for r in b_records) / max(len(b_records), 1)
    c_linked_avg = sum(r["linked_count"] for r in c_records) / max(len(c_records), 1)

    # Đo False Safe Rate (nguy hiểm nhất trong y tế): Gold=UNSAFE nhưng Cấu hình C dự đoán SAFE
    false_safe_count = sum(
        1 for r_c, r_gt in zip(c_records, benchmark_rows)
        if r_gt.get("gold_safety_label", "").upper().strip() == "UNSAFE" and r_c["status"] in ("SAFE", "CAUTION")
    )
    total_unsafe_gold = sum(1 for r_gt in benchmark_rows if r_gt.get("gold_safety_label", "").upper().strip() == "UNSAFE")
    false_safe_rate = false_safe_count / total_unsafe_gold if total_unsafe_gold > 0 else 0.0

    return {
        "samples_evaluated": len(b_records),
        "config_b_safety_acc": b_acc,
        "config_c_safety_acc": c_acc,
        "delta_safety_acc": c_acc - b_acc,
        "config_b_avg_linked_entities": b_linked_avg,
        "config_c_avg_linked_entities": c_linked_avg,
        "delta_linked_entities": c_linked_avg - b_linked_avg,
        "false_safe_rate_in_config_c": false_safe_rate,
    }
