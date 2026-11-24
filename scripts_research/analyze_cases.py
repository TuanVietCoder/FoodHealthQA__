"""Phân tích ca (yêu cầu 5): hallucination, trả lời thiếu căn cứ, câu hỏi cần chuyển chuyên gia y tế.

Mỗi nhóm xuất DANH SÁCH CA CỤ THỂ (mã câu, cấu hình, câu hỏi, đoạn trả lời, lý do) để đưa vào báo cáo, kèm số đếm.
Chỉ dùng tín hiệu đo được — không gán điểm:

1. HALLUCINATION (dấu hiệu)
   - fake_citation: thẻ nguồn không thuộc danh mục và không phải nguồn chuẩn của câu (mô hình tự bịa);
   - ins_not_in_evidence: câu trả lời nêu mã INS/E không có trên nhãn đầu vào, trong câu hỏi hay trong đoạn nguồn đã truy xuất;
   - judge_contradicted: nhận định giám khảo chấm CONTRADICTED (khi đã chạy judge_faithfulness.py).
2. THIẾU CĂN CỨ
   - judge_not_enough_info: nhận định giám khảo chấm NOT_ENOUGH_INFO (khi đã chạy judge_faithfulness.py);
   - nếu chưa có giám khảo: low_lexical_grounding = tỉ lệ từ của câu trả lời có trong bằng chứng < 0,30
     (chỉ là SÀNG LỌC để đọc tay, ghi rõ là trùng từ).
3. CẦN CHUYÊN GIA Y TẾ
   - system_referred: guardrail xếp "vượt phạm vi" (chẩn đoán, thuốc, điều trị…) → đã chuyển chuyên gia;
   - question_needs_expert_not_referred: câu hỏi khớp luật vượt phạm vi nhưng hệ thống KHÔNG từ chối;
   - false_safe: nhãn chuẩn nguy hiểm (UNSAFE/CRITICAL_ALERT) nhưng hệ thống nói SAFE — ca rủi ro nhất;
   - critical_missed: nhãn chuẩn CRITICAL_ALERT (luật mức cao, cần xác minh với chuyên gia) mà hệ thống không gắn CRITICAL_ALERT.
   Lưu ý: bộ QA KHÔNG có nhãn chuẩn "cần chuyên gia", nên nhóm 3 là đối chiếu với luật và nhãn an toàn, không phải độ chính xác.

  python scripts_research/analyze_cases.py [--run-dir data] [--configs A B C]
Đầu ra: data/analysis/case_analysis.json + data/analysis/cases_<nhóm>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import abstention  # noqa: E402
from modules.additive_linker import _EXPLICIT_CODE_RE, link_additive_mentions, normalize_ins_key  # noqa: E402
from modules.benchmark import analyze_answer_citations, canonicalize_citation, valid_predicted_sources  # noqa: E402
from modules.evaluation import _heuristic_groundedness, prediction_evidence  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "analysis"
DANGER = {"UNSAFE", "CRITICAL_ALERT", "WARNING"}
LEXICAL_SCREEN = 0.30


def load_jsonl(path):
    path = Path(path)
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if path.is_file() else []


def codes_in(text):
    """Mã INS/E được nêu tường minh (có tiền tố INS hoặc E)."""
    return {normalize_ins_key(m.group("code")) for m in _EXPLICIT_CODE_RE.finditer(str(text or ""))} - {""}


def codes_on_label(text):
    """Mã trên nhãn: cả mã tường minh lẫn mã suy ra từ tên phụ gia có nguyên văn trên nhãn."""
    found = codes_in(text)
    for hit in link_additive_mentions(str(text or "")):
        found.add(normalize_ins_key(hit["matched_entity"].get("ins_code", "")))
    return found - {""}


def label_input(prediction, reference):
    if prediction.get("config") == "C":
        return str((prediction.get("ocr") or {}).get("full_text") or "")
    return str(prediction.get("ground_truth_text") or reference.get("product_ingredients") or "")


def clip(text, n=300):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n] + "…"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(DATA_DIR))
    ap.add_argument("--benchmark", default=str(DATA_DIR / "foodhealthqa_eval_set.jsonl"))
    ap.add_argument("--configs", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    args = ap.parse_args()

    refs = {r["id"]: r for r in load_jsonl(args.benchmark)}
    judgments = {}
    for j in load_jsonl(OUT_DIR / "faithfulness_judgments.jsonl"):
        if j.get("status") == "ok":
            judgments[(j["id"], j["config"])] = j
    cases = defaultdict(list)
    counts = defaultdict(Counter)
    answered = Counter()

    for cfg in args.configs:
        for p in load_jsonl(Path(args.run_dir) / f"predictions_config_{cfg.lower()}.jsonl"):
            p.setdefault("config", cfg)
            ref = refs.get(p["id"])
            answer = str(p.get("answer") or "")
            if not ref or p.get("error") or not answer.strip():
                continue
            answered[cfg] += 1
            base = {"id": p["id"], "config": cfg, "domain": ref.get("domain"), "question": ref.get("question"),
                    "gold_label": ref.get("gold_safety_label"), "predicted_label": p.get("safety_label"),
                    "answer_excerpt": clip(answer)}

            def add(group, reason, **extra):
                cases[group].append(dict(base, reason=reason, **extra))
                counts[cfg][group] += 1

            # 1. hallucination
            gold_sources = {canonicalize_citation(s.strip()) for s in str(ref.get("source_citations") or "").split(",") if s.strip()}
            _, stray = valid_predicted_sources(analyze_answer_citations(answer), gold_sources)
            if stray:
                add("fake_citation", "thẻ nguồn không thuộc danh mục: " + ", ".join(stray))
            # mã người dùng nêu trong câu hỏi cũng có nguồn gốc (câu trả lời nhắc lại không phải tự bịa)
            evidence_codes = codes_on_label(label_input(p, ref)) | codes_on_label(ref.get("question", ""))
            for passage in p.get("retrieved_passages") or []:
                evidence_codes |= codes_in(passage.get("text"))
            invented = codes_in(answer) - evidence_codes
            if invented:
                add("ins_not_in_evidence", "câu trả lời nêu mã không có trên nhãn đầu vào/đoạn nguồn: "
                    + ", ".join(sorted(invented)))
            j = judgments.get((p["id"], cfg))
            if j:
                for c in j["claims"]:
                    if c["label"] == "CONTRADICTED":
                        add("judge_contradicted", "giám khảo: trái bằng chứng", claim=c["text"], quote=c.get("quote", ""))
                    elif c["label"] == "NOT_ENOUGH_INFO":
                        add("judge_not_enough_info", "giám khảo: bằng chứng không nói tới", claim=c["text"])
            else:
                evidence = prediction_evidence(p, ref)
                g = _heuristic_groundedness(answer, evidence) if evidence else None
                if g is not None and g < LEXICAL_SCREEN:
                    add("low_lexical_grounding", f"trùng từ với bằng chứng {g:.2f} < {LEXICAL_SCREEN} (sàng lọc để đọc tay)")

            # 3. cần chuyên gia
            guard = p.get("guardrail") or {}
            referred = bool(guard.get("referred_to_expert")) or "out_of_scope" in (guard.get("abstention_types") or [])
            scope_reasons = abstention.detect_out_of_scope(ref.get("question", ""))
            if referred:
                add("system_referred", "; ".join(guard.get("abstention_reasons") or []) or "vượt phạm vi")
            elif scope_reasons and cfg != "A":
                add("question_needs_expert_not_referred", "câu hỏi khớp luật vượt phạm vi: " + "; ".join(scope_reasons))
            gold = str(ref.get("gold_safety_label") or "")
            pred = str(p.get("safety_label") or "")
            if gold in DANGER and pred == "SAFE":
                add("false_safe", f"nhãn chuẩn {gold} nhưng hệ thống trả SAFE")
            if gold == "CRITICAL_ALERT" and pred != "CRITICAL_ALERT":
                add("critical_missed", f"nhãn chuẩn CRITICAL_ALERT, hệ thống trả {pred}")

    report = {
        "answers_analyzed": dict(answered),
        "faithfulness_judge_available": bool(judgments),
        "counts": {cfg: dict(c) for cfg, c in counts.items()},
        "rates": {cfg: {g: n / answered[cfg] for g, n in c.items()} for cfg, c in counts.items() if answered[cfg]},
        "note": "Nhóm 'cần chuyên gia' đối chiếu với luật vượt phạm vi và nhãn an toàn; bộ QA không có nhãn chuẩn "
                "'cần chuyên gia'. 'low_lexical_grounding' chỉ là sàng lọc theo trùng từ, không phải kết luận.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "case_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for group, rows in cases.items():
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(OUT_DIR / f"cases_{group}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n→ {OUT_DIR / 'case_analysis.json'} và {len(cases)} file cases_*.csv")


if __name__ == "__main__":
    main()
