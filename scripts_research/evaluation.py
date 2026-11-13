"""
Evaluation Pipeline cho hệ thống FoodHealthQA.
Wrapper module — re-export từ modules/evaluation.py (module chính thức).

Giữ file này để scripts hiện tại không bị break import.
"""

import sys
from pathlib import Path

# Cho phép chạy trực tiếp từ bất kỳ thư mục nào (thêm thư mục gốc dự án vào sys.path).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from modules.evaluation import (  # noqa: E402
    SAFETY_LABEL_ORDER,
    compute_cer,
    compute_wer,
    compute_confusion_matrix,
    format_confusion_matrix,
    evaluate_single,
    aggregate_metrics,
    evaluate_config,
    format_results_table,
    _heuristic_groundedness,
    _edit_distance,
)

__all__ = [
    "SAFETY_LABEL_ORDER",
    "compute_cer",
    "compute_wer",
    "compute_confusion_matrix",
    "format_confusion_matrix",
    "evaluate_single",
    "aggregate_metrics",
    "evaluate_config",
    "format_results_table",
    "_heuristic_groundedness",
    "_edit_distance",
]
