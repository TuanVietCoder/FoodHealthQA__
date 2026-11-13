"""Compatibility entry point for the canonical A/B/C benchmark workflow.

This script intentionally delegates prediction and scoring to the two canonical
programs so a second, divergent implementation of Config B/C cannot reappear.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts_research"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.benchmark_manifest import (
    get_active_benchmark_path,
    get_default_eval_benchmark_path,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="all", choices=["A", "B", "C", "all"])
    parser.add_argument(
        "--benchmark",
        default=get_default_eval_benchmark_path(DATA_DIR, "jsonl"),
        help="Canonical JSONL benchmark (mặc định: data/foodhealthqa_eval_set.jsonl)",
    )
    parser.add_argument(
        "--eval-750",
        action="store_true",
        help="Sử dụng bộ đánh giá phân tầng chuẩn 750 câu (data/foodhealthqa_eval_set.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    benchmark_target = str(DATA_DIR / "foodhealthqa_eval_set.jsonl") if args.eval_750 else args.benchmark

    configs = ["A", "B", "C"] if args.mode == "all" else [args.mode]
    generate_command = [
        sys.executable,
        str(SCRIPTS_DIR / "generate_predictions.py"),
        "--benchmark",
        benchmark_target,
        "--configs",
        *configs,
        "--delay",
        str(args.delay),
    ]
    if args.limit > 0:
        generate_command.extend(["--limit", str(args.limit)])
    if args.resume:
        generate_command.append("--resume")
    subprocess.run(generate_command, cwd=PROJECT_ROOT, check=True)

    if args.mode != "all":
        print("Đã sinh prediction cho cấu hình đã chọn; cần đủ A/B/C trước khi chấm chính thức.")
        return

    evaluate_command = [
        sys.executable,
        str(SCRIPTS_DIR / "run_evaluation.py"),
        "--gold",
        benchmark_target,
    ]
    if args.limit > 0:
        evaluate_command.extend(
            [
                "--allow-incomplete",
                "--output",
                str(DATA_DIR / "evaluation_results_debug.json"),
            ]
        )
    subprocess.run(evaluate_command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
