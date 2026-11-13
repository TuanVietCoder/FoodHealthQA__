"""Build a balanced evaluation subset from data/qa_benchmark.jsonl (the benchmark itself is NOT modified).

Why: the full benchmark is skewed (additive_regulation 75% SAFE, insufficient_data 80% ABSTAIN) and a few templates
(T001, T256, T002) cover ~20% of it. For running the 3 configurations (A/B/C) a capped subset gives a cleaner signal
and fewer API calls. Caps applied while walking a seeded shuffle:
  * per template   <= --max-per-template
  * per photo      <= --max-per-image
  * per (domain, label): majority label of a domain <= --majority-ratio x the 2nd most frequent label of that domain
Every record stays a real, source-verified benchmark record; nothing is invented. Cells that are too small to report
(< --min-cell) are listed as "do not report per-label metrics".

Usage
  python scripts_research/build_eval_subset.py [--tier core|all] [--seed 42]
Writes data/qa_eval_balanced.jsonl and data/qa_eval_balanced_summary.json.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "qa_benchmark.jsonl"
OUT = ROOT / "data" / "qa_eval_balanced.jsonl"
SUMMARY = ROOT / "data" / "qa_eval_balanced_summary.json"


def load(path: Path = SRC) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("eligible_for_scoring") == "yes"]


def label_caps(rows: list[dict], ratio: float) -> dict:
    """(domain, label) -> max records. Only the most frequent label of a domain is capped."""
    caps = {}
    for d in {r["domain"] for r in rows}:
        c = Counter(r["gold_safety_label"] for r in rows if r["domain"] == d).most_common()
        second = c[1][1] if len(c) > 1 else c[0][1]
        for i, (lab, n) in enumerate(c):
            caps[(d, lab)] = min(n, int(ratio * second)) if i == 0 else n
    return caps


def build(rows: list[dict], max_per_template=30, max_per_image=4, majority_ratio=2.0, seed=42) -> list[dict]:
    caps = label_caps(rows, majority_ratio)
    rng = random.Random(seed)
    order = rows[:]
    rng.shuffle(order)
    t_n, i_n, l_n = Counter(), Counter(), Counter()
    out = []
    for r in order:
        key = (r["domain"], r["gold_safety_label"])
        if t_n[r["template_id"]] >= max_per_template or i_n[r["split_group"]] >= max_per_image:
            continue
        if l_n[key] >= caps[key]:
            continue
        t_n[r["template_id"]] += 1
        i_n[r["split_group"]] += 1
        l_n[key] += 1
        out.append(r)
    return sorted(out, key=lambda r: r["id"])


def summarize(src: list[dict], sub: list[dict], min_cell: int) -> dict:
    def cells(rows):
        return Counter(f'{r["domain"]}|{r["gold_safety_label"]}' for r in rows)

    def maj(rows):
        res = {}
        for d in sorted({r["domain"] for r in rows}):
            c = Counter(r["gold_safety_label"] for r in rows if r["domain"] == d)
            res[d] = round(max(c.values()) / sum(c.values()), 3)
        return res

    tmpl = Counter(r["template_id"] for r in sub)
    sc = cells(sub)
    return {
        "source_records": len(src),
        "subset_records": len(sub),
        "photos": len({r["split_group"] for r in sub}),
        "templates": len(tmpl),
        "max_template_share": round(max(tmpl.values()) / len(sub), 3),
        "by_label": dict(Counter(r["gold_safety_label"] for r in sub)),
        "by_domain": dict(Counter(r["domain"] for r in sub)),
        "by_persona": dict(Counter(r["persona_id"] for r in sub)),
        "majority_baseline_by_domain_before": maj(src),
        "majority_baseline_by_domain_after": maj(sub),
        "cells_too_small_for_per_label_metrics": {k: v for k, v in sorted(sc.items()) if v < min_cell},
        "note": "derived view of data/qa_benchmark.jsonl; silver, no expert validation",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["core", "all"], default="core")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-template", type=int, default=30)
    ap.add_argument("--max-per-image", type=int, default=4)
    ap.add_argument("--majority-ratio", type=float, default=2.0)
    ap.add_argument("--min-cell", type=int, default=30)
    a = ap.parse_args()
    src = [r for r in load() if a.tier == "all" or r.get("tier") == "core"]
    sub = build(src, a.max_per_template, a.max_per_image, a.majority_ratio, a.seed)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in sub) + "\n", encoding="utf-8")
    s = summarize(src, sub, a.min_cell)
    SUMMARY.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
