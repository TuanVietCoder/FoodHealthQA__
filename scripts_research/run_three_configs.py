"""Backward-compatible launcher for the canonical A/B/C workflow.

The old implementation used different retrieval logic for B and C. Keeping a
second definition made the ablation invalid, so this entry point now delegates
to ``run_benchmark_evaluation.py``.
"""

import os
import sys

try:
    from .run_benchmark_evaluation import main
except ImportError:
    try:
        from run_benchmark_evaluation import main
    except ImportError:
        from scripts_research.run_benchmark_evaluation import main


def _is_approved(row):
    return (
        str(row.get("review_status", "")).strip().casefold() == "approved"
        and str(row.get("answer_supported_by_source", "")).strip().casefold() in ("yes", "true", "1")
        and str(row.get("citation_correct", "")).strip().casefold() in ("yes", "true", "1")
        and bool(str(row.get("final_answer", "")).strip())
    )


if __name__ == "__main__":
    main()
