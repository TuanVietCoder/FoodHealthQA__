"""
Chạy batch evaluation qua 3 cấu hình ablation:
  A: LLM-only (không RAG, không evidence)
  B: Ground Truth text + RAG (evidence từ KB, không qua OCR)
  C: OCR + RAG (ảnh bao bì -> OCR -> RAG)

Đầu vào:
  - Default benchmark: phiên bản active trong data/benchmark_manifest.json
  - Prediction files: data/predictions_config_A.jsonl, _B.jsonl, _C.jsonl
    Mỗi dòng: {"id": "...", "answer": "...", "safety_label": "...", ...}

Đầu ra:
  - data/evaluation_results.json (tất cả metrics)
  - In bảng so sánh + confusion matrix ra console

Chạy: python scripts/run_evaluation.py
       python scripts/run_evaluation.py --gold data/foodhealthqa_eval_set.jsonl
       python scripts/run_evaluation.py --pred-a answers_a.jsonl --pred-b answers_b.jsonl --pred-c answers_c.jsonl
"""

import argparse
from collections import Counter
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.stdout.reconfigure(encoding="utf-8")

from modules.benchmark_manifest import (
    get_active_benchmark_path,
    get_default_eval_benchmark_path,
    is_evaluation_eligible,
    load_benchmark_manifest,
)

try:
    from scripts_research.evaluation import (
        evaluate_config,
        format_results_table,
        format_confusion_matrix,
    )
except ImportError:
    from evaluation import (
        evaluate_config,
        format_results_table,
        format_confusion_matrix,
    )

DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def _load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _prediction_integrity(references, predictions, expected_config=None):
    gold_ids = {record["id"] for record in references}
    raw_prediction_ids = [record.get("id") for record in predictions]
    prediction_ids = set(raw_prediction_ids)
    duplicates = sorted(
        record_id for record_id, count in Counter(raw_prediction_ids).items()
        if record_id is not None and count > 1
    )
    missing = sorted(gold_ids - prediction_ids)
    extra = sorted(prediction_ids - gold_ids)
    failed = [record.get("id") for record in predictions if record.get("error")]
    empty = [record.get("id") for record in predictions if not str(record.get("answer", "")).strip()]
    non_real = [
        record.get("id") for record in predictions
        if record.get("evaluation_type") != "real"
    ]
    wrong_config = [
        record.get("id") for record in predictions
        if expected_config and record.get("config") != expected_config
    ]
    return {
        "missing": missing,
        "extra": extra,
        "duplicates": duplicates,
        "failed": failed,
        "empty": empty,
        "non_real": non_real,
        "wrong_config": wrong_config,
        "complete": not (
            missing or extra or duplicates or failed or empty or non_real or wrong_config
        ),
    }


def perception_summary(predictions, threshold=0.65, legacy_constant=0.85):
    """Thống kê độ tin cậy OCR (S_perception) và điểm tổng của các dự đoán — để báo cáo cấu hình C dùng điểm thật."""
    perc = [float(p["perception_confidence"]) for p in predictions if p.get("perception_confidence") is not None]
    if not perc:
        return None
    totals = [float(p["total_confidence"]) for p in predictions if p.get("total_confidence") is not None]
    srt = sorted(perc)
    return {
        "count": len(perc),
        "mean": sum(perc) / len(perc),
        "min": srt[0],
        "p10": srt[int(0.1 * (len(srt) - 1))],
        "median": srt[len(srt) // 2],
        "max": srt[-1],
        "below_threshold_total": sum(t < threshold for t in totals),
        "threshold": threshold,
        "legacy_constant_count": sum(abs(x - legacy_constant) < 1e-9 for x in perc),
        "ocr_pipeline_versions": sorted({
            str(p.get("ocr_pipeline_version") or ("legacy" if p.get("config") == "C" else "không dùng OCR"))
            for p in predictions if p.get("perception_confidence") is not None}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        default=get_default_eval_benchmark_path(DATA_DIR, "jsonl"),
        help="Reference benchmark JSONL (mặc định: data/foodhealthqa_eval_set.jsonl)",
    )
    parser.add_argument(
        "--eval-750",
        action="store_true",
        help="Sử dụng bộ đánh giá phân tầng chuẩn 750 câu (data/foodhealthqa_eval_set.jsonl)",
    )
    parser.add_argument("--pred-a", default=os.path.join(DATA_DIR, "predictions_config_a.jsonl"), help="Config A predictions JSONL")
    parser.add_argument("--pred-b", default=os.path.join(DATA_DIR, "predictions_config_b.jsonl"), help="Config B predictions JSONL")
    parser.add_argument("--pred-c", default=os.path.join(DATA_DIR, "predictions_config_c.jsonl"), help="Config C predictions JSONL")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file (mặc định: data/evaluation_results.json)",
    )
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Chỉ dùng để debug; cho phép đánh giá prediction thiếu/lỗi",
    )
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="Debug only: include records still waiting for manual verification",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Thư mục kết quả của một lần chạy (vd data/runs/gemini): tự lấy predictions_config_{a,b,c}.jsonl "
             "trong đó và ghi evaluation_results.json vào cùng thư mục",
    )
    args = parser.parse_args()

    if args.run_dir:
        args.pred_a = os.path.join(args.run_dir, "predictions_config_a.jsonl")
        args.pred_b = os.path.join(args.run_dir, "predictions_config_b.jsonl")
        args.pred_c = os.path.join(args.run_dir, "predictions_config_c.jsonl")
        if not args.output:
            args.output = os.path.join(args.run_dir, "evaluation_results.json")

    if args.eval_750:
        args.gold = os.path.join(DATA_DIR, "foodhealthqa_eval_set.jsonl")
    gold_path = args.gold
    if not os.path.isfile(gold_path):
        print(f"[ERROR] Không tìm thấy reference benchmark: {gold_path}")
        sys.exit(1)

    references = _load_jsonl(gold_path)
    if not args.include_pending:
        references = [record for record in references if is_evaluation_eligible(record)]
    manifest = load_benchmark_manifest(DATA_DIR)
    active_version = manifest.get("active_version", "")
    benchmark_metadata = manifest.get("benchmarks", {}).get(active_version, {})
    is_active_benchmark = os.path.abspath(gold_path) == os.path.abspath(
        get_default_eval_benchmark_path(DATA_DIR, "jsonl")
    )
    reference_status = (
        benchmark_metadata.get("status", "unspecified")
        if is_active_benchmark
        else "external_reference_status_unspecified"
    )
    gold_validated = bool(benchmark_metadata.get("expert_validated", False)) if is_active_benchmark else False
    print(
        f"Loaded {len(references)} reference benchmark records "
        f"({reference_status}; expert_validated={gold_validated}) from {gold_path}"
    )

    configs = {}
    integrity_by_config = {}
    config_files = {
        "A_LLM_Only": args.pred_a,
        "B_GT_RAG": args.pred_b,
        "C_OCR_RAG": args.pred_c,
    }

    for name, path in config_files.items():
        if path and os.path.isfile(path):
            configs[name] = _load_jsonl(path)
            print(f"Loaded {len(configs[name])} predictions cho {name}")
        else:
            print(f"[ERROR] Thiếu prediction file cho {name}: {path}")

    if len(configs) != 3:
        print("\n[ERROR] Đánh giá chính yêu cầu đủ cả A, B và C.")
        sys.exit(1)

    for name, predictions in configs.items():
        integrity = _prediction_integrity(references, predictions, expected_config=name[0])
        integrity_by_config[name] = integrity
        if not integrity["complete"]:
            print(
                f"[ERROR] {name} không đầy đủ: missing={len(integrity['missing'])}, "
                f"extra={len(integrity['extra'])}, duplicates={len(integrity['duplicates'])}, "
                f"failed={len(integrity['failed'])}, empty={len(integrity['empty'])}, "
                f"non_real={len(integrity['non_real'])}, "
                f"wrong_config={len(integrity['wrong_config'])}"
            )
    if not args.allow_incomplete and not all(
        item["complete"] for item in integrity_by_config.values()
    ):
        print("Dừng để không công bố số liệu một phần. Dùng --allow-incomplete chỉ khi debug.")
        sys.exit(2)

    all_results = []
    for config_name, predictions in configs.items():
        print(f"\n{'='*60}")
        print(f"Đánh giá: {config_name}")
        print(f"{'='*60}")

        result = evaluate_config(references, predictions, config_name)
        all_results.append(result)

        if result["missing_predictions"]:
            print(f"  [WARN] {len(result['missing_predictions'])} câu thiếu prediction")
        if result["extra_predictions"]:
            print(f"  [WARN] {len(result['extra_predictions'])} prediction dư")

        print(f"\n  Overall metrics:")
        for key, val in result["overall"].items():
            if key == "config":
                continue
            if isinstance(val, float):
                print(f"    {key:>30s}: {val:.4f}")
            else:
                print(f"    {key:>30s}: {val}")

        print(f"\n  Per-domain breakdown:")
        for domain, metrics in result["per_domain"].items():
            em = metrics.get("exact_match", 0)
            f1 = metrics.get("token_f1", 0)
            sa = metrics.get("safety_accuracy", "N/A")
            n = metrics.get("count", 0)
            sa_str = f"{sa:.3f}" if isinstance(sa, float) else sa
            print(f"    {domain:>25s}: N={n:3d}  EM={em:.3f}  F1={f1:.3f}  SafetyAcc={sa_str}")

        print(f"\n  Confusion Matrix (Safety Labels):")
        print(format_confusion_matrix(result["confusion_matrix"]))

    print(f"\n{'='*60}")
    print("BẢNG SO SÁNH TỔNG HỢP")
    print(f"{'='*60}")
    print(format_results_table(all_results))

    is_complete = all(item["complete"] for item in integrity_by_config.values())
    output_data = {
        "evaluation_type": "real",
        "complete": bool(is_complete),
        "reference_dataset": gold_path,
        "reference_dataset_status": reference_status,
        "gold_validated": gold_validated,
        "score_interpretation": (
            "Exploratory weak-label metrics; not gold-standard performance."
            if not gold_validated
            else "Gold-standard metrics."
        ),
        "reference_count": len(references),
        "prediction_integrity": integrity_by_config,
        "configs": {},
    }
    for result in all_results:
        perception = perception_summary(configs.get(result["config"], []))
        if perception:
            print(f"  [{result['config']}] S_perception: mean={perception['mean']:.3f} min={perception['min']:.3f} "
                  f"· S_total<{perception['threshold']}: {perception['below_threshold_total']} câu "
                  f"· còn hằng số 0,85 cũ: {perception['legacy_constant_count']}")
        serializable = {
            "overall": result["overall"],
            "per_domain": result["per_domain"],
            "confusion_matrix": result["confusion_matrix"],
            "missing_predictions": result["missing_predictions"],
            "extra_predictions": result["extra_predictions"],
        }
        if perception:
            serializable["perception"] = perception
        output_data["configs"][result["config"]] = serializable

    output_path = args.output or os.path.join(DATA_DIR, "evaluation_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Kết quả chi tiết đã lưu: {output_path}")


if __name__ == "__main__":
    main()
