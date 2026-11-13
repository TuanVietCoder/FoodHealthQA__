"""How well does the rule guardrail (Config B: label text, OCR confidence 1.0) abstain where the benchmark says ABSTAIN?

Reports, over the scored records: recall of ABSTAIN on gold-ABSTAIN records (by abstain_reason), the share of
non-ABSTAIN records that are wrongly refused, and the abstention type given. Rule layer only, no LLM.
The label text is rebuilt from the transcription file, so this approximates (not replaces) the app's Config B run.

Usage: python scripts_research/measure_abstention.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.ocr_extractor import parse_ingredients_from_ocr  # noqa: E402
from modules.pipeline import (build_persona_from_benchmark_record, evaluate_safety_and_guardrail,  # noqa: E402
                              foodontorag_entity_linking, load_knowledge_base, load_personas)


def flat(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(flat(x) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{k}: {flat(x)}" for k, x in v.items())
    return str(v)


def main():
    data = str(ROOT / "data")
    kb, personas = load_knowledge_base(data), load_personas(data)
    rows = [json.loads(l) for l in (ROOT / "data" / "foodhealthqa_eval_set.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    res = []
    for r in rows:
        if r["eligible_for_scoring"] != "yes":
            continue
        lab = json.loads((ROOT / r["label_transcription_path"]).read_text(encoding="utf-8"))
        text = f"Thành phần: {flat(lab.get('ingredient'))}. {flat(lab.get('additive'))}. {flat(lab.get('warning'))}"
        mentions = parse_ingredients_from_ocr(text, include_allergen_statements=True)
        linked = [foodontorag_entity_linking(m, kb) for m in mentions]
        ev = evaluate_safety_and_guardrail(linked, build_persona_from_benchmark_record(r, personas), r["question"],
                                           avg_ocr_conf=1.0, all_mentions=mentions)
        res.append((r["gold_safety_label"], ev.get("status"), ev.get("abstention_type"), r.get("abstain_reason", "")))
    gold = [x for x in res if x[0] == "ABSTAIN"]
    rest = [x for x in res if x[0] != "ABSTAIN"]
    hit = sum(1 for x in gold if x[1] == "ABSTAIN")
    wrong = sum(1 for x in rest if x[1] == "ABSTAIN")
    print(f"gold ABSTAIN: {hit}/{len(gold)} refused ({hit / len(gold):.1%})")
    for reason in sorted({x[3] for x in gold}):
        s = [x for x in gold if x[3] == reason]
        print(f"  {reason}: {sum(1 for x in s if x[1] == 'ABSTAIN')}/{len(s)}")
    print(f"gold non-ABSTAIN wrongly refused: {wrong}/{len(rest)} ({wrong / len(rest):.1%})")
    print("abstention types given:", dict(collections.Counter(x[2] for x in res if x[1] == "ABSTAIN")))


if __name__ == "__main__":
    main()
