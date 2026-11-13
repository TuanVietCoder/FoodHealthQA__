"""
Đo tỷ lệ tuân thủ trích dẫn của mô hình ngôn ngữ khi chạy LẶP cùng một câu hỏi.

Vì sao cần: mô hình nhỏ (vd Qwen2.5 7B) đôi lúc bỏ sót thẻ nguồn `[MÃ_NGUỒN]` và kết quả đổi giữa các lần chạy
(nhiệt độ > 0), nên một lần chạy không đủ để nói mô hình "có trích dẫn". Script chạy cấu hình B (downstream giống C)
K lần trên N câu phân tầng và báo cáo:
  - citation_compliance: tỷ lệ lượt chạy có >=1 thẻ nguồn được công nhận
  - phân loại câu: luôn / đôi khi / không bao giờ trích dẫn (cần --repeats >= 2)
  - unrecognized_citation_rate: tỷ lệ lượt chạy có thẻ nguồn lạ (mô hình tự bịa)
  - số câu trả lời khác nhau trung bình mỗi câu (độ biến thiên) và độ ổn định của nhãn an toàn (do guardrail)

Chạy:
  python scripts_research/measure_citation_compliance.py --llm-provider ollama --questions 10 --repeats 3
  python scripts_research/measure_citation_compliance.py --llm-provider ollama --temperature 0   # thí nghiệm nhiệt độ
Ghi: data/runs/<provider>/citation_compliance[_t<temp>].json
Tốn N x K lượt gọi LLM. Gemini dừng sạch khi hết hạn mức ngày (kết quả đã đo vẫn được lưu).
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from modules import llm_keys  # noqa: E402
from modules import pipeline as P  # noqa: E402
from modules.benchmark import analyze_answer_citations  # noqa: E402
from modules.benchmark_manifest import is_evaluation_eligible  # noqa: E402
from scripts_research.generate_predictions import (  # noqa: E402
    DATA_DIR,
    _load_dotenv,
    _load_jsonl,
    _load_resources,
    _predict_one,
)


def stratified_sample(records, n, seed):
    """Lấy n bản ghi vòng tròn qua các ô (lĩnh vực, nhãn) để phủ đều; seed cố định."""
    rng = random.Random(seed)
    cells = defaultdict(list)
    for r in records:
        cells[(r.get("domain", "?"), r.get("gold_safety_label", "?"))].append(r)
    for items in cells.values():
        rng.shuffle(items)
    keys = sorted(cells)
    chosen = []
    while len(chosen) < n and any(cells[k] for k in keys):
        for k in keys:
            if cells[k] and len(chosen) < n:
                chosen.append(cells[k].pop())
    return chosen


def classify_question(cited_flags):
    if all(cited_flags):
        return "always"
    if not any(cited_flags):
        return "never"
    return "sometimes"


def summarize(per_question, repeats):
    runs = [run for q in per_question for run in q["runs"]]
    if not runs:
        return {}
    cls = Counter(q["class"] for q in per_question)
    stable_labels = sum(1 for q in per_question if len({run["label"] for run in q["runs"]}) == 1)
    return {
        "questions": len(per_question),
        "runs": len(runs),
        "citation_compliance": round(sum(1 for r in runs if r["cited"]) / len(runs), 4),
        "unrecognized_citation_rate": round(sum(1 for r in runs if r["unrecognized"]) / len(runs), 4),
        "unrecognized_tags_per_run": round(sum(len(r["unrecognized"]) for r in runs) / len(runs), 4),
        "mean_recognized_citations_per_run": round(sum(len(r["recognized"]) for r in runs) / len(runs), 3),
        "question_classes": {k: cls.get(k, 0) for k in ("always", "sometimes", "never")} if repeats >= 2 else None,
        "mean_distinct_answers_per_question": round(
            sum(len({r["answer_hash"] for r in q["runs"]}) for q in per_question) / len(per_question), 2),
        "label_stable_questions": f"{stable_labels}/{len(per_question)}",
        "failed_runs": sum(1 for r in runs if r["error"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=str(DATA_DIR / "foodhealthqa_eval_set.jsonl"))
    ap.add_argument("--llm-provider", choices=["gemini", "ollama"], default=None)
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=None,
                    help="Ép nhiệt độ sinh (thí nghiệm); mặc định dùng giá trị của pipeline (0,3)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    _load_dotenv()
    if args.llm_provider:
        P.set_llm_provider(args.llm_provider)
    provider = P.get_llm_provider()
    if provider == "ollama":
        ok, message = P.ollama_status()
        if not ok:
            ap.error(message)
    elif not llm_keys.get_pool().keys:
        ap.error("Chưa có khóa Gemini (GEMINI_API_KEYS / gemini_keys.txt)")

    records = [r for r in _load_jsonl(Path(args.benchmark)) if is_evaluation_eligible(r)]
    sample = stratified_sample(records, args.questions, args.seed)
    resources = _load_resources()

    if args.temperature is not None:
        original = P._call_llm
        forced = args.temperature
        P._call_llm = lambda prompt, **kw: original(prompt, **{**kw, "temperature": forced})

    llm_keys.set_strict(True)
    per_question, stop_reason = [], "completed"
    total = len(sample) * args.repeats
    done = 0
    try:
        for record in sample:
            runs = []
            for _ in range(args.repeats):
                prediction = _predict_one("B", record, resources, {}, None)
                answer = prediction.get("answer", "")
                analysis = analyze_answer_citations(answer)
                runs.append({
                    "cited": analysis["has_citation"], "recognized": analysis["recognized"],
                    "unrecognized": analysis["unrecognized"], "label": prediction.get("safety_label"),
                    "answer_hash": hashlib.md5(answer.encode("utf-8")).hexdigest()[:8],
                    "error": prediction.get("error"),
                })
                done += 1
                print(f"[{done}/{total}] {record['id']} cited={analysis['has_citation']} "
                      f"recognized={len(analysis['recognized'])} stray={analysis['unrecognized']} "
                      f"label={prediction.get('safety_label')}", flush=True)
            per_question.append({
                "id": record["id"], "domain": record.get("domain"), "gold_label": record.get("gold_safety_label"),
                "class": classify_question([r["cited"] for r in runs]), "runs": runs,
            })
    except (llm_keys.LLMQuotaExhausted, llm_keys.LLMUnavailable) as stop:
        stop_reason = f"{type(stop).__name__}: {stop}"
        print(f"\n⛔ Dừng sớm: {stop_reason} (kết quả đã đo vẫn được lưu)")
    except KeyboardInterrupt:
        stop_reason = "KeyboardInterrupt"
    finally:
        llm_keys.set_strict(False)

    summary = summarize(per_question, args.repeats)
    payload = {"provider": provider, "model": P.get_last_llm_model(), "temperature": args.temperature,
               "benchmark": args.benchmark, "seed": args.seed, "repeats": args.repeats,
               "stop_reason": stop_reason, "summary": summary, "per_question": per_question}
    suffix = f"_t{args.temperature}" if args.temperature is not None else ""
    out = Path(args.output) if args.output else DATA_DIR / "runs" / provider / f"citation_compliance{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Tóm tắt ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
