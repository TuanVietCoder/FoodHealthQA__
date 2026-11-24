"""
scripts_research/eval_nli_faithfulness.py: Đánh giá Độ trung thực Ngữ nghĩa (Faithfulness) và Ảo giác (Hallucination) qua NLI.

Script này phục vụ nghiên cứu khoa học, khóa luận và bài báo:
- Đọc các kết quả dự đoán của các Cấu hình (A, B, C) hoặc bộ benchmark 750 câu.
- Phân tích từng nhận định (atomic claims) và đo tỷ lệ Entailment / Contradiction / Neutral.
- Báo cáo các chỉ số khoa học:
  + Faithfulness Rate (Độ trung thực ngữ nghĩa)
  + Contradiction Rate (Tỷ lệ ảo giác mâu thuẫn)
  + Unsubstantiated Claim Rate (Tỷ lệ phát ngôn không có căn cứ)
  + Phân bố theo Trụ cột chuyên môn (Pillars)

Sử dụng:
  python scripts_research/eval_nli_faithfulness.py [--limit 50] [--input-file data/predictions_config_c.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.nli_verifier import verify_answer_faithfulness
from modules.evaluation import prediction_evidence


def load_jsonl(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_nli_evaluation(
    records: list[dict],
    limit: int = 0,
    verbose: bool = False
) -> dict:
    if limit > 0:
        records = records[:limit]

    total_records = len(records)
    total_claims = 0
    total_entailed = 0
    total_contradicted = 0
    total_neutral = 0

    per_pillar_stats = defaultdict(lambda: {"total": 0, "entailed": 0, "contradicted": 0, "neutral": 0})
    contradiction_examples = []

    print(f"Bắt đầu đánh giá NLI Faithfulness trên {total_records} bản ghi...")

    skipped_no_answer = 0
    gold_lookup = {}
    for p_cand in [
        ROOT / "data" / "foodhealthqa_eval_set.jsonl",
        ROOT / "data" / "archive" / "master_benchmark_2311" / "qa_benchmark.jsonl",
    ]:
        if p_cand.exists():
            for r_gold in load_jsonl(p_cand):
                if r_gold.get("id") and r_gold["id"] not in gold_lookup:
                    gold_lookup[r_gold["id"]] = r_gold

    for idx, r in enumerate(records, 1):
        rec_id = r.get("id")
        gold_rec = gold_lookup.get(rec_id, {})

        # Chỉ chấm CÂU TRẢ LỜI DO HỆ THỐNG SINH (file dự đoán). Trước đây khi không có "answer" script lấy luôn
        # đáp án tham chiếu làm câu trả lời rồi so với chính nó.
        answer = r.get("predicted_answer") or r.get("answer") or ""
        if not str(answer).strip():
            skipped_no_answer += 1
            continue
        # Bằng chứng = cái mô hình thực sự được đọc (văn bản nhãn đầu vào + đoạn nguồn đã truy xuất),
        # KHÔNG phải đáp án tham chiếu hay danh sách mã nguồn.
        evidence_text = prediction_evidence(r, gold_rec)
        pillar = gold_rec.get("category_vi") or r.get("category_vi") or r.get("domain") or "Chung"

        res = verify_answer_faithfulness(answer, evidence_text, trigger_only=False)

        total_claims += res["verified_count"]
        total_entailed += res["entailment_count"]
        total_contradicted += res["contradiction_count"]
        total_neutral += res["neutral_count"]

        per_pillar_stats[pillar]["total"] += res["verified_count"]
        per_pillar_stats[pillar]["entailed"] += res["entailment_count"]
        per_pillar_stats[pillar]["contradicted"] += res["contradiction_count"]
        per_pillar_stats[pillar]["neutral"] += res["neutral_count"]

        if res["contradiction_count"] > 0 and len(contradiction_examples) < 5:
            for c in res["claims"]:
                if c["label"] == "CONTRADICTION":
                    contradiction_examples.append({
                        "id": r.get("id", f"sample_{idx}"),
                        "claim": c["text"],
                        "reason": c["reason"]
                    })

        if verbose and idx % 10 == 0:
            print(f"  Đã xử lý {idx}/{total_records} bản ghi...")

    # không có nhận định nào → không áp dụng (trước đây = 100%)
    mean_faithfulness = (total_entailed / total_claims) * 100 if total_claims > 0 else None
    contradiction_rate = (total_contradicted / total_claims) * 100 if total_claims > 0 else None
    neutral_rate = (total_neutral / total_claims) * 100 if total_claims > 0 else None

    pillar_summary = {}
    for p, stats in per_pillar_stats.items():
        tot = stats["total"]
        pillar_summary[p] = {
            "claims_count": tot,
            "faithfulness_pct": round((stats["entailed"] / tot * 100), 2) if tot > 0 else None,
            "contradiction_pct": round((stats["contradicted"] / tot * 100), 2) if tot > 0 else None,
            "neutral_pct": round((stats["neutral"] / tot * 100), 2) if tot > 0 else None,
        }

    summary = {
        "method": "keyword_overlap",
        "method_note": ("Đối chiếu TỪ KHÓA giữa nhận định và bằng chứng (nhãn đầu vào + đoạn nguồn đã truy xuất); "
                        "không phải NLI/người chấm. Faithfulness chính thức dùng judge_faithfulness.py."),
        "skipped_no_answer": skipped_no_answer,
        "total_records_evaluated": total_records - skipped_no_answer,
        "total_claims_analyzed": total_claims,
        "overall_faithfulness_rate_pct": round(mean_faithfulness, 2) if mean_faithfulness is not None else None,
        "contradiction_rate_pct": round(contradiction_rate, 2) if contradiction_rate is not None else None,
        "unsubstantiated_rate_pct": round(neutral_rate, 2) if neutral_rate is not None else None,
        "pillar_breakdown": pillar_summary,
        "sample_contradictions": contradiction_examples
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Đánh giá NLI Faithfulness cho FoodHealthQA")
    parser.add_argument("--input-file", type=str, default="", help="Đường dẫn file predictions hoặc benchmark jsonl")
    parser.add_argument("--limit", type=int, default=50, help="Số lượng bản ghi tối đa để đánh giá")
    parser.add_argument("--out", type=str, default="data/runs/nli_faithfulness_report.json", help="Đường dẫn file output")
    parser.add_argument("--verbose", action="store_true", help="In tiến trình chi tiết")
    args = parser.parse_args()

    input_path = args.input_file
    if not input_path:
        # Mặc định thử predictions_config_c.jsonl hoặc foodhealthqa_eval_set.jsonl
        pred_c = ROOT / "data" / "predictions_config_c.jsonl"
        eval_set = ROOT / "data" / "foodhealthqa_eval_set.jsonl"
        input_path = str(pred_c if pred_c.exists() else eval_set)

    print(f"Nguồn dữ liệu đánh giá: {input_path}")
    records = load_jsonl(input_path)
    if not records:
        print(f"LỖI: Không đọc được dữ liệu từ {input_path}")
        return

    results = run_nli_evaluation(records, limit=args.limit, verbose=args.verbose)

    out_p = ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("           KẾT QUẢ ĐÁNH GIÁ NLI FAITHFULNESS           ")
    print("=" * 60)
    print(f"Số bản ghi đánh giá:            {results['total_records_evaluated']}")
    print(f"Tổng số nhận định (Claims):    {results['total_claims_analyzed']}")
    print(f"Độ trung thực (Faithfulness):   {results['overall_faithfulness_rate_pct']}% (Entailment)")
    print(f"Tỷ lệ mâu thuẫn (Ảo giác):      {results['contradiction_rate_pct']}% (Contradiction)")
    print(f"Tỷ lệ ngoài lề (Chưa chứng minh):{results['unsubstantiated_rate_pct']}% (Neutral)")
    print("-" * 60)
    print("Phân bố theo Trụ cột chuyên môn:")
    for pillar, st in results["pillar_breakdown"].items():
        print(f"  - {pillar:22s}: Faithfulness = {st['faithfulness_pct']}% (n={st['claims_count']})")
    print("=" * 60)
    print(f"Báo cáo chi tiết đã được ghi vào: {out_p}")


if __name__ == "__main__":
    main()
