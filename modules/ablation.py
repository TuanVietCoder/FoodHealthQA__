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


_DANGER = {"UNSAFE", "CRITICAL_ALERT", "WARNING"}


def _gold_label(row):
    return str(row.get("gold_safety_label") or "").upper().strip()


def _rule_label(linked_results, mentions, persona, question, perception_confidence):
    """Nhãn 5 lớp của tầng luật, quy đổi giống hệt khi chấm A/B/C (guardrail_status_to_safety_label)."""
    from modules.pipeline import evaluate_safety_and_guardrail, guardrail_status_to_safety_label
    evaluation = evaluate_safety_and_guardrail(
        linked_results, persona, question, avg_ocr_conf=perception_confidence, all_mentions=mentions,
    )
    return evaluation, guardrail_status_to_safety_label(evaluation.get("status"))


def run_knowledge_source_ablation(benchmark_rows, kb_additives, personas_dict,
                                  hazards_list, fndds_list, off_products,
                                  excluded_sources=None):
    """
    Ablation bỏ từng nguồn tri thức (Leave-One-Out) trên TẦNG LUẬT với văn bản nhãn chuẩn.

    Sửa so với bản cũ: độ tin cậy nhận diện = 1,0 (đầu vào là văn bản chuẩn, không có OCR — bản cũ gán cứng 0,99);
    chấm đủ 5 nhãn như đánh giá chính (bản cũ quy CAUTION → SAFE và chỉ chấm SAFE/UNSAFE); liên kết nhiều mã/cụm.
    Ablation có LLM (câu trả lời, trích dẫn) phải chạy bằng generate_predictions.py với nguồn tương ứng bị tắt.
    """
    from modules.pipeline import (
        check_food_hazards,
        evaluate_nutritional_risk,
        cross_reference_openfoodfacts,
        link_all_mentions,
    )
    from modules.ocr_extractor import parse_ingredients_from_ocr

    excluded = set(excluded_sources or [])
    results = []

    active_additives = [] if "additives_standard" in excluded else kb_additives
    active_hazards = [] if "food_hazards" in excluded else hazards_list
    active_fndds = [] if "fndds_nutrition" in excluded else fndds_list
    active_off = [] if "openfoodfacts" in excluded else off_products

    for row in benchmark_rows:
        persona = personas_dict.get(row.get("persona_id"), {}) or {}
        ingredients_text = row.get("product_ingredients", "")
        product_name = row.get("product_name", "")
        question = row.get("question", "")
        if not ingredients_text.strip():
            continue

        mentions = parse_ingredients_from_ocr(f"Thành phần: {ingredients_text}", include_allergen_statements=True)
        linked_results = link_all_mentions(mentions, active_additives)
        evaluation, pred_label = _rule_label(linked_results, mentions, persona, question, 1.0)
        detected_hazards = check_food_hazards(product_name, mentions, active_hazards)
        matched_fndds, nutri_warnings, _ = evaluate_nutritional_risk(product_name, active_fndds, persona)
        cross_reference_openfoodfacts(product_name, mentions, active_off)

        gold = _gold_label(row)
        results.append({
            "id": row.get("id"),
            "domain": row.get("domain"),
            "status": evaluation.get("status"),
            "predicted_label": pred_label,
            "gold_label": gold,
            "safety_match": (pred_label == gold) if gold else None,
            "hazards_detected_count": len(detected_hazards),
            "nutrition_matched": bool(matched_fndds),
        })

    valid_matches = [r["safety_match"] for r in results if r["safety_match"] is not None]
    return {
        "excluded_sources": sorted(excluded),
        "total_evaluated": len(results),
        "safety_accuracy": (sum(valid_matches) / len(valid_matches)) if valid_matches else None,
        "detail_records": results,
    }


def run_error_propagation_study(benchmark_rows, kb_additives, personas_dict,
                                hazards_list, fndds_list, off_products,
                                noise_level=0.15):
    """
    THỬ ĐỘ BỀN VỚI NHIỄU NHÂN TẠO (synthetic noise robustness) — KHÔNG phải lan truyền lỗi OCR thật.

    Văn bản nhãn chuẩn bị làm nhiễu ngẫu nhiên (bỏ dấu, cắt ký tự, "500"→"50O"), rồi so nhãn của tầng luật với bản
    không nhiễu. Lan truyền lỗi OCR THẬT = so cấu hình B và C trong đánh giá chính (generate_predictions.py →
    run_evaluation.py) và scripts_research/calibrate_confidence.py.

    Sửa so với bản cũ: cả hai nhánh dùng độ tin cậy 1,0 (nhiễu nhân tạo không có điểm OCR — bản cũ gán cứng 0,88 làm
    lẫn hiệu ứng ngưỡng vào kết quả); chấm 5 nhãn; false-safe tính trên đúng các dòng đã chấm (bản cũ ghép kết quả
    với danh sách gốc nên lệch dòng khi có dòng bị bỏ qua); liên kết nhiều mã/cụm.
    """
    import random
    from modules.pipeline import link_all_mentions
    from modules.ocr_extractor import parse_ingredients_from_ocr, post_process_ocr_spelling

    rng = random.Random(42)

    def _simulate_noise(text, p):
        noisy = []
        for token in text.split():
            if rng.random() < p:
                choice = rng.choice(["drop_accent", "ins_confuse", "typo"])
                if choice == "ins_confuse" and "500" in token:
                    noisy.append(token.replace("500", "50O"))
                elif choice == "drop_accent":
                    noisy.append("".join(c for c in unicodedata.normalize("NFD", token) if not unicodedata.combining(c)))
                else:
                    noisy.append(token[:-1] if len(token) > 2 else token)
            else:
                noisy.append(token)
        return " ".join(noisy)

    clean_records, noisy_records = [], []
    for row in benchmark_rows:
        persona = personas_dict.get(row.get("persona_id"), {}) or {}
        gt_text = row.get("product_ingredients", "")
        if not gt_text.strip():
            continue
        question = row.get("question", "")
        gold = _gold_label(row)

        b_mentions = parse_ingredients_from_ocr(f"Thành phần: {gt_text}", include_allergen_statements=True)
        b_linked = link_all_mentions(b_mentions, kb_additives)
        b_eval, b_pred = _rule_label(b_linked, b_mentions, persona, question, 1.0)

        corrected = post_process_ocr_spelling(_simulate_noise(gt_text, noise_level))
        c_text = corrected["final"] if isinstance(corrected, dict) else str(corrected)
        c_mentions = parse_ingredients_from_ocr(f"Thành phần: {c_text}", include_allergen_statements=True)
        c_linked = link_all_mentions(c_mentions, kb_additives)
        c_eval, c_pred = _rule_label(c_linked, c_mentions, persona, question, 1.0)

        for store, pred, mentions, linked, ev in (
            (clean_records, b_pred, b_mentions, b_linked, b_eval),
            (noisy_records, c_pred, c_mentions, c_linked, c_eval),
        ):
            store.append({
                "id": row.get("id"),
                "gold_label": gold,
                "predicted_label": pred,
                "mentions_count": len(mentions),
                "linked_count": sum(1 for x in linked if x.get("matched_entity")),
                "safety_correct": (pred == gold) if gold else None,
                "status": ev.get("status"),
            })

    def _acc(records):
        vals = [r["safety_correct"] for r in records if r["safety_correct"] is not None]
        return sum(vals) / len(vals) if vals else None

    def _false_safe(records):
        danger = [r for r in records if r["gold_label"] in _DANGER]
        return (sum(r["predicted_label"] == "SAFE" for r in danger) / len(danger)) if danger else None

    def _avg(records, key):
        return sum(r[key] for r in records) / len(records) if records else None

    b_acc, c_acc = _acc(clean_records), _acc(noisy_records)
    b_link, c_link = _avg(clean_records, "linked_count"), _avg(noisy_records, "linked_count")
    return {
        "study": "synthetic_noise_robustness",
        "note": "Nhiễu nhân tạo trên văn bản chuẩn; không phải lỗi OCR thật.",
        "noise_level": noise_level,
        "samples_evaluated": len(clean_records),
        "config_b_safety_acc": b_acc,
        "config_c_safety_acc": c_acc,
        "delta_safety_acc": (c_acc - b_acc) if (b_acc is not None and c_acc is not None) else None,
        "config_b_avg_linked_entities": b_link,
        "config_c_avg_linked_entities": c_link,
        "delta_linked_entities": (c_link - b_link) if (b_link is not None and c_link is not None) else None,
        "false_safe_rate_in_config_c": _false_safe(noisy_records),
        "false_safe_rate_in_config_b": _false_safe(clean_records),
    }
