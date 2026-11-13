"""
So sánh kết quả đánh giá giữa hai nhà cung cấp LLM (Gemini và Ollama/Qwen2.5) trên CÙNG tập câu hỏi.

Vì chạy Gemini bị giới hạn hạn mức theo ngày nên có thể mới xong một phần; script chỉ so sánh trên
giao của các câu đã có dự đoán tốt (không lỗi, không rỗng) ở mọi (nhà cung cấp, cấu hình) đang so sánh,
để hai bên được so trên cùng một tập (so sánh theo cặp), và in rõ độ phủ.

Đầu vào mặc định: data/runs/gemini/ và data/runs/ollama/ (do generate_predictions.py tạo).
Đầu ra: data/runs/comparison.json và data/runs/comparison.md

Chạy:
  python scripts_research/compare_providers.py
  python scripts_research/compare_providers.py --configs A B --bootstrap 2000
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from modules.benchmark_manifest import (  # noqa: E402
    get_active_benchmark_path,
    get_default_eval_benchmark_path,
    is_evaluation_eligible,
)
from modules.evaluation import evaluate_config, format_results_table  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
PROVIDERS = ("gemini", "ollama")
METRICS = [
    ("safety_accuracy", "Safety Acc"),
    ("unsafe_recall", "Unsafe Recall"),
    ("safety_score", "Safety Score"),
    ("citation_precision", "Cit. Precision"),
    ("citation_recall", "Cit. Recall"),
    ("citation_compliance", "Citation compliance"),
    ("unrecognized_citation_rate", "Fake-tag rate"),
    ("token_f1", "Token F1"),
    ("groundedness", "Groundedness"),
    ("abstention_rate_predicted", "Abstain rate"),
]


def _load_jsonl(path):
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _good(record):
    return not record.get("error") and bool(str(record.get("answer", "")).strip())


def _paired_bootstrap(correct_a, correct_b, rounds, seed=13):
    """Khoảng tin cậy 95% của (acc_gemini - acc_ollama) theo cặp trên cùng các câu hỏi."""
    pairs = [(a, b) for a, b in zip(correct_a, correct_b) if a is not None and b is not None]
    if not pairs:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(rounds):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        diffs.append(sum(a for a, _ in sample) / len(sample) - sum(b for _, b in sample) / len(sample))
    diffs.sort()
    point = sum(a for a, _ in pairs) / len(pairs) - sum(b for _, b in pairs) / len(pairs)
    return {
        "n_pairs": len(pairs),
        "diff": round(point, 4),
        "ci95": [round(diffs[int(0.025 * rounds)], 4), round(diffs[int(0.975 * rounds) - 1], 4)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=get_default_eval_benchmark_path(str(DATA_DIR), "jsonl"))
    parser.add_argument("--gemini-dir", default=str(DATA_DIR / "runs" / "gemini"))
    parser.add_argument("--ollama-dir", default=str(DATA_DIR / "runs" / "ollama"))
    parser.add_argument("--configs", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", default=str(DATA_DIR / "runs"))
    args = parser.parse_args()

    references = [r for r in _load_jsonl(Path(args.gold)) if is_evaluation_eligible(r)]
    total = len(references)
    dirs = {"gemini": Path(args.gemini_dir), "ollama": Path(args.ollama_dir)}

    preds = {}  # (provider, config) -> {id: record}
    for provider, directory in dirs.items():
        for config in args.configs:
            records = _load_jsonl(directory / f"predictions_config_{config.lower()}.jsonl")
            preds[(provider, config)] = {r["id"]: r for r in records if _good(r)}

    present = {k: v for k, v in preds.items() if v}
    missing = [f"{p}/{c}" for (p, c) in preds if (p, c) not in present]
    if len({p for p, _ in present}) < 2:
        print("Cần có dự đoán của cả hai nhà cung cấp để so sánh. Thiếu:", ", ".join(missing) or "-")
        sys.exit(1)
    # chỉ so sánh những cấu hình có ở cả hai nhà cung cấp
    configs = [c for c in args.configs if ("gemini", c) in present and ("ollama", c) in present]
    if not configs:
        print("Không có cấu hình nào có dự đoán ở cả hai nhà cung cấp.")
        sys.exit(1)

    common = set.intersection(*(set(present[(p, c)]) for p in PROVIDERS for c in configs))
    refs = [r for r in references if r["id"] in common]
    print(f"Tổng câu eligible: {total} | đã có đủ dự đoán ở cả hai bên (cấu hình {','.join(configs)}): {len(refs)}")
    if not refs:
        print("Chưa có câu nào hoàn thành ở cả hai bên.")
        sys.exit(1)

    results, summary = {}, {}
    for provider in PROVIDERS:
        for config in configs:
            name = f"{provider}/{config}"
            result = evaluate_config(refs, list(present[(provider, config)].values()), name)
            results[name] = result
            models = {}
            for rec in present[(provider, config)].values():
                if rec["id"] in common:
                    models[rec.get("llm_model") or "?"] = models.get(rec.get("llm_model") or "?", 0) + 1
            summary[name] = {
                "overall": result["overall"],
                "per_domain": result["per_domain"],
                "confusion_matrix": result["confusion_matrix"],
                "models_used": models,
                "coverage_all_eligible": len(present[(provider, config)]),
            }

    paired = {}
    for config in configs:
        g = {i["id"]: i["safety_correct"] for i in results[f"gemini/{config}"]["item_results"]}
        o = {i["id"]: i["safety_correct"] for i in results[f"ollama/{config}"]["item_results"]}
        ids = sorted(set(g) & set(o))
        paired[config] = _paired_bootstrap(
            [None if g[i] is None else int(g[i]) for i in ids],
            [None if o[i] is None else int(o[i]) for i in ids],
            args.bootstrap,
        )

    table = format_results_table(list(results.values()))
    print("\n" + table)
    print("\nChênh lệch Safety Accuracy theo cặp (gemini − ollama), bootstrap 95% CI:")
    for config, stat in paired.items():
        if stat:
            print(f"  Config {config}: {stat['diff']:+.3f}  CI95 [{stat['ci95'][0]:+.3f}, {stat['ci95'][1]:+.3f}]  (n={stat['n_pairs']})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_dataset": args.gold,
        "n_eligible": total,
        "n_compared": len(refs),
        "coverage_pct": round(100 * len(refs) / max(total, 1), 1),
        "configs": configs,
        "not_available": missing,
        "results": summary,
        "paired_safety_accuracy_diff_gemini_minus_ollama": paired,
        "note": "Metric yếu-nhãn trên benchmark chưa có chuyên gia; mô hình Gemini dùng chuỗi fallback nên xem models_used.",
    }
    (out_dir / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# So sánh Gemini và Qwen2.5 (Ollama)",
        "",
        f"- Số câu so sánh: **{len(refs)}/{total}** ({payload['coverage_pct']}% tập eligible), cùng một tập câu cho mọi cột.",
        "- Benchmark chưa có chuyên gia kiểm định: đây là chỉ số yếu-nhãn, không phải gold-standard.",
        "",
        "| Metric | " + " | ".join(results) + " |",
        "|---|" + "---|" * len(results),
    ]
    for key, label in METRICS:
        row = [f"{summary[n]['overall'].get(key):.3f}" if isinstance(summary[n]["overall"].get(key), float) else "-" for n in results]
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    lines += ["", "## Chênh lệch Safety Accuracy theo cặp (gemini − ollama)", ""]
    for config, stat in paired.items():
        if stat:
            lines.append(f"- Config {config}: {stat['diff']:+.3f}, CI95 [{stat['ci95'][0]:+.3f}, {stat['ci95'][1]:+.3f}], n={stat['n_pairs']}")
    lines += ["", "## Model thực tế đã trả lời", ""]
    for name, s in summary.items():
        lines.append(f"- {name}: {s['models_used']}")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nĐã lưu: {out_dir / 'comparison.json'} và {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
