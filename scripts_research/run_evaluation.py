"""
Chạy batch evaluation qua 3 cấu hình ablation:
  A: LLM-only (không RAG, không evidence)
  B: Ground Truth text + RAG (evidence từ KB, không qua OCR)
  C: OCR + RAG + LLM (full pipeline)

Đầu vào:
  - Gold benchmark: data/qa_benchmark_gold.jsonl (hoặc merged_draft.jsonl)
  - Prediction files: data/predictions_config_A.jsonl, _B.jsonl, _C.jsonl
    Mỗi dòng: {"id": "...", "answer": "...", "safety_label": "...", ...}

Đầu ra:
  - data/evaluation_results.json (tất cả metrics)
  - In bảng so sánh + confusion matrix ra console

Chạy: python scripts/run_evaluation.py
       python scripts/run_evaluation.py --gold data/qa_benchmark_gold.jsonl
       python scripts/run_evaluation.py --pred-a answers_a.jsonl --pred-b answers_b.jsonl --pred-c answers_c.jsonl
"""

import argparse
import json
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.stdout.reconfigure(encoding="utf-8")

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


def _generate_baseline_predictions(references, config, seed=42):
    """
    Tạo baseline predictions khi chưa có file prediction thật.
    Config A (LLM-only): trả lời generic, label mặc định SAFE (lower bound)
    Config B (GT+RAG):   dùng reference_answer làm answer (upper bound)
    Config C (OCR+RAG):  reference_answer + nhiễu OCR mô phỏng (~15% token flip, ~10% label noise)
    """
    rng = random.Random(seed)
    predictions = []
    for ref in references:
        pred = {"id": ref["id"]}

        if config == "A":
            pred["answer"] = (
                f"Thông tin về {ref.get('domain', 'chủ đề này')} "
                "cần được xác minh thêm từ nguồn chuyên môn."
            )
            pred["safety_label"] = "SAFE"
        elif config == "B":
            pred["answer"] = ref.get("reference_answer", "")
            pred["safety_label"] = ref.get("gold_safety_label", "SAFE")
        elif config == "C":
            answer = ref.get("reference_answer", "")
            words = answer.split()
            if len(words) > 5 and rng.random() < 0.15:
                idx = rng.randint(1, len(words) - 2)
                words[idx] = words[idx][::-1]
                answer = " ".join(words)
            pred["answer"] = answer
            label = ref.get("gold_safety_label", "SAFE")
            if rng.random() < 0.10:
                label = rng.choice(["SAFE", "CAUTION"])
            pred["safety_label"] = label
        else:
            pred["answer"] = ""
            pred["safety_label"] = "SAFE"

        predictions.append(pred)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        default=os.path.join(DATA_DIR, "qa_benchmark_gold.jsonl"),
        help="Gold benchmark JSONL (fallback: merged_draft.jsonl)",
    )
    parser.add_argument("--pred-a", default=None, help="Config A predictions JSONL")
    parser.add_argument("--pred-b", default=None, help="Config B predictions JSONL")
    parser.add_argument("--pred-c", default=None, help="Config C predictions JSONL")
    parser.add_argument(
        "--output",
        default=os.path.join(DATA_DIR, "evaluation_results.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Tạo baseline predictions giả lập nếu chưa có file prediction thật",
    )
    args = parser.parse_args()

    gold_path = args.gold
    if not os.path.isfile(gold_path):
        fallback = os.path.join(DATA_DIR, "qa_benchmark_merged_draft.jsonl")
        if os.path.isfile(fallback):
            print(f"[INFO] Gold file không tồn tại, dùng fallback: {fallback}")
            gold_path = fallback
        else:
            print(f"[ERROR] Không tìm thấy gold benchmark: {gold_path}")
            sys.exit(1)

    references = _load_jsonl(gold_path)
    print(f"Loaded {len(references)} gold benchmark records từ {gold_path}")

    configs = {}
    config_files = {
        "A_LLM_Only": args.pred_a,
        "B_GT_RAG": args.pred_b,
        "C_OCR_RAG_LLM": args.pred_c,
    }

    for name, path in config_files.items():
        if path and os.path.isfile(path):
            configs[name] = _load_jsonl(path)
            print(f"Loaded {len(configs[name])} predictions cho {name}")
        elif args.baseline:
            tag = name[0]
            configs[name] = _generate_baseline_predictions(references, tag)
            print(f"Generated {len(configs[name])} baseline predictions cho {name}")

    if not configs:
        print("\n[WARNING] Không có prediction files. Dùng --baseline để tạo baseline.")
        print("Ví dụ: python scripts/run_evaluation.py --baseline")
        print("\nHoặc chỉ định prediction files:")
        print("  python scripts/run_evaluation.py --pred-a pred_a.jsonl --pred-b pred_b.jsonl --pred-c pred_c.jsonl")
        sys.exit(1)

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

    output_data = {
        "gold_benchmark": gold_path,
        "gold_count": len(references),
        "configs": {},
    }
    for result in all_results:
        serializable = {
            "overall": result["overall"],
            "per_domain": result["per_domain"],
            "confusion_matrix": result["confusion_matrix"],
            "missing_predictions": result["missing_predictions"],
            "extra_predictions": result["extra_predictions"],
        }
        output_data["configs"][result["config"]] = serializable

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Kết quả chi tiết đã lưu: {args.output}")


if __name__ == "__main__":
    main()
