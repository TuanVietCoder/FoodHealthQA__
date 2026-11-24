"""Faithfulness bằng LLM làm giám khảo + kiểm chứng bằng người chấm.

Mỗi câu trả lời được tách thành các nhận định (cùng cách tách với modules/nli_verifier.split_into_claims). Giám khảo
nhận BẰNG CHỨNG THẬT mà mô hình đã đọc khi trả lời (văn bản nhãn đầu vào + đoạn nguồn đã truy xuất, xem
modules/evaluation.prediction_evidence) và gán cho mỗi nhận định:
  SUPPORTED        bằng chứng ủng hộ trực tiếp
  CONTRADICTED     trái với bằng chứng                         → dấu hiệu hallucination
  NOT_ENOUGH_INFO  bằng chứng không nói tới                    → trả lời thiếu căn cứ
kèm TRÍCH NGUYÊN VĂN câu bằng chứng làm căn cứ. Code kiểm tra trích dẫn đó có thật trong bằng chứng không; không có
thì hạ xuống NOT_ENOUGH_INFO (giám khảo không được tự bịa căn cứ). Nhiệt độ 0, prompt cố định (PROMPT_VERSION).

Faithfulness của một câu trả lời = SUPPORTED / tổng nhận định. Số liệu chỉ được báo cáo kèm mức đồng thuận với người
chấm (Cohen's kappa) trên mẫu chấm tay.

Các bước:
  1) Chấm bằng giám khảo (chạy tiếp được, --max-new để chạy từng đợt theo hạn mức):
       python scripts_research/judge_faithfulness.py judge --configs A B C --provider gemini --judge-model <model>
     Nên dùng model KHÁC model sinh câu trả lời (giảm thiên vị tự chấm); ghi rõ trong báo cáo nếu dùng chung.
  2) Xuất mẫu chấm tay (giám khảo bị ẩn để người chấm không bị ảnh hưởng):
       python scripts_research/judge_faithfulness.py export-annotation --n 50
     → mở data/analysis/faithfulness_annotation.csv bằng Excel, điền cột human_label = S / C / N.
  3) Đo đồng thuận người – giám khảo:
       python scripts_research/judge_faithfulness.py agreement --annotation data/analysis/faithfulness_annotation.csv
  4) Tổng hợp:
       python scripts_research/judge_faithfulness.py summarize
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import pipeline as pl  # noqa: E402
from modules.evaluation import prediction_evidence  # noqa: E402
from modules.nli_verifier import split_into_claims  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "analysis"
JUDGMENTS = OUT_DIR / "faithfulness_judgments.jsonl"
SUMMARY = OUT_DIR / "faithfulness_summary.json"
ANNOTATION = OUT_DIR / "faithfulness_annotation.csv"
PROMPT_VERSION = "faithfulness_judge_v1"
LABELS = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")
SHORT = {"S": "SUPPORTED", "C": "CONTRADICTED", "N": "NOT_ENOUGH_INFO"}
EVIDENCE_MAX_CHARS = 9000

PROMPT = """Bạn là giám khảo kiểm chứng. Nhiệm vụ: với MỖI nhận định bên dưới, xác định BẰNG CHỨNG có ủng hộ nó không.
Chỉ dùng BẰNG CHỨNG được cung cấp, KHÔNG dùng kiến thức bên ngoài, kể cả khi bạn biết nhận định đúng.

Nhãn:
- SUPPORTED: bằng chứng nói trực tiếp hoặc suy ra hiển nhiên được điều nhận định nói.
- CONTRADICTED: bằng chứng nói điều trái ngược với nhận định.
- NOT_ENOUGH_INFO: bằng chứng không đề cập đủ để kết luận (kể cả khi nhận định nghe hợp lý).
Với SUPPORTED hoặc CONTRADICTED, trường "quote" PHẢI là một đoạn chép NGUYÊN VĂN (ít nhất 5 từ) từ BẰNG CHỨNG.
Với NOT_ENOUGH_INFO, để "quote" rỗng.

=== BẰNG CHỨNG ===
{evidence}

=== NHẬN ĐỊNH ===
{claims}

Trả lời DUY NHẤT một mảng JSON, mỗi phần tử: {{"i": <số thứ tự nhận định>, "label": "SUPPORTED|CONTRADICTED|NOT_ENOUGH_INFO", "quote": "..."}}"""


def load_jsonl(path):
    path = Path(path)
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if path.is_file() else []


def _norm(text):
    text = unicodedata.normalize("NFC", str(text or "")).casefold()
    return " ".join(re.findall(r"[^\W_]+", text))


def quote_in_evidence(quote, evidence, min_words=5):
    """Trích dẫn của giám khảo có thật trong bằng chứng không (so sau khi bỏ dấu câu, chữ thường)."""
    q = _norm(quote)
    return len(q.split()) >= min_words and q in _norm(evidence)


def parse_judge_output(raw, n_claims):
    """Đọc mảng JSON của giám khảo; nhận định thiếu → None (sẽ ghi lỗi để chấm lại)."""
    if not raw:
        return None
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return None
    try:
        items = json.loads(m.group(0))
    except ValueError:
        return None
    out = {}
    for it in items if isinstance(items, list) else []:
        try:
            i = int(it.get("i"))
        except (TypeError, ValueError, AttributeError):
            continue
        label = str(it.get("label", "")).strip().upper()
        if 1 <= i <= n_claims and label in LABELS:
            out[i] = {"label": label, "quote": str(it.get("quote") or "")}
    return out if len(out) == n_claims else None


def judge_one(prediction, reference):
    answer = str(prediction.get("answer") or "")
    claims = split_into_claims(answer)
    evidence = prediction_evidence(prediction, reference)
    base = {"id": prediction["id"], "config": prediction.get("config"), "prompt_version": PROMPT_VERSION}
    if not claims:
        return dict(base, claims=[], faithfulness=None, status="no_claims")
    if not evidence.strip():
        return dict(base, claims=[{"i": k, "text": c["text"], "label": "NOT_ENOUGH_INFO", "quote": "",
                                    "quote_verified": False, "judge_label": None} for k, c in enumerate(claims, 1)],
                    faithfulness=0.0, status="no_evidence")
    prompt = PROMPT.format(
        evidence=evidence[:EVIDENCE_MAX_CHARS],
        claims="\n".join(f"{k}. {c['text']}" for k, c in enumerate(claims, 1)),
    )
    raw = pl._call_llm(prompt, temperature=0.0, max_output_tokens=2048)
    parsed = parse_judge_output(raw, len(claims))
    if parsed is None:
        return dict(base, error="judge_output_unparseable", raw=str(raw or "")[:500])
    rows = []
    for k, c in enumerate(claims, 1):
        judged = parsed[k]
        label = judged["label"]
        verified = quote_in_evidence(judged["quote"], evidence) if label != "NOT_ENOUGH_INFO" else None
        final = label if (label == "NOT_ENOUGH_INFO" or verified) else "NOT_ENOUGH_INFO"
        rows.append({"i": k, "text": c["text"], "judge_label": label, "label": final,
                     "quote": judged["quote"], "quote_verified": verified})
    supported = sum(r["label"] == "SUPPORTED" for r in rows)
    return dict(base, judge_model=pl.get_last_llm_model(), claims=rows,
                faithfulness=supported / len(rows), status="ok")


def _predictions(run_dir, configs):
    for c in configs:
        for p in load_jsonl(Path(run_dir) / f"predictions_config_{c.lower()}.jsonl"):
            if not p.get("error") and str(p.get("answer") or "").strip():
                p.setdefault("config", c)
                yield p


def cmd_judge(args):
    if args.provider:
        pl.set_llm_provider(args.provider)
    if args.judge_model:
        os.environ["GEMINI_MODEL" if pl.get_llm_provider() == "gemini" else "OLLAMA_MODEL"] = args.judge_model
    refs = {r["id"]: r for r in load_jsonl(args.benchmark)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {(j["id"], j["config"]) for j in load_jsonl(JUDGMENTS)
            if j.get("status") in ("ok", "no_claims", "no_evidence") and j.get("prompt_version") == PROMPT_VERSION}
    new = 0
    with open(JUDGMENTS, "a", encoding="utf-8") as out:
        for pred in _predictions(args.run_dir, args.configs):
            key = (pred["id"], pred["config"])
            if key in done or pred["id"] not in refs:
                continue
            try:
                result = judge_one(pred, refs[pred["id"]])
            except Exception as exc:  # hết hạn mức → dừng, lần sau chạy tiếp
                print(f"[judge] dừng: {exc}")
                break
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            new += 1
            print(f"[judge] {pred['config']} {pred['id']} → {result.get('status')} "
                  f"faithfulness={result.get('faithfulness')}", flush=True)
            if args.max_new and new >= args.max_new:
                break
    print(f"[judge] chấm mới {new} câu trả lời → {JUDGMENTS}")


def _latest_judgments():
    latest = {}
    for j in load_jsonl(JUDGMENTS):
        if j.get("prompt_version") == PROMPT_VERSION and j.get("status") in ("ok", "no_claims", "no_evidence"):
            latest[(j["id"], j["config"])] = j
    return latest


def cmd_export(args):
    refs = {r["id"]: r for r in load_jsonl(args.benchmark)}
    preds = {(p["id"], p["config"]): p for p in _predictions(args.run_dir, args.configs)}
    candidates = [k for k, j in _latest_judgments().items() if j.get("claims") and k in preds]
    if not candidates:
        print("Chưa có câu trả lời nào đã được giám khảo chấm — chạy bước 'judge' trước.")
        return
    rng = random.Random(args.seed)
    by_stratum = defaultdict(list)  # phân tầng theo cấu hình × lĩnh vực
    for k in candidates:
        by_stratum[(k[1], refs.get(k[0], {}).get("domain", ""))].append(k)
    for v in by_stratum.values():
        rng.shuffle(v)
    picked, strata = [], sorted(by_stratum)
    while len(picked) < min(args.n, len(candidates)):
        for s in strata:
            if by_stratum[s] and len(picked) < args.n:
                picked.append(by_stratum[s].pop())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATION, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "id", "config", "claim_no", "claim", "evidence", "human_label (S/C/N)", "note"])
        judgments = _latest_judgments()
        for n, k in enumerate(picked, 1):
            evidence = prediction_evidence(preds[k], refs[k[0]])[:EVIDENCE_MAX_CHARS]
            for c in judgments[k]["claims"]:
                w.writerow([n, k[0], k[1], c["i"], c["text"], evidence if c["i"] == 1 else "(như dòng đầu của câu này)", "", ""])
    print(f"Đã xuất {len(picked)} câu trả lời ({sum(len(_latest_judgments()[k]['claims']) for k in picked)} nhận định) "
          f"→ {ANNOTATION}\nĐiền cột 'human_label (S/C/N)': S = có trong bằng chứng, C = trái bằng chứng, "
          f"N = bằng chứng không nói tới. Nhãn giám khảo được ẩn.")


def cohen_kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else None


def cmd_agreement(args):
    judgments = _latest_judgments()
    pairs, missing = [], 0
    with open(args.annotation, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            human = SHORT.get(str(row.get("human_label (S/C/N)", "")).strip().upper()[:1])
            j = judgments.get((row["id"], row["config"]))
            if not human or not j:
                missing += 1
                continue
            claim = next((c for c in j["claims"] if str(c["i"]) == str(row["claim_no"])), None)
            if claim:
                pairs.append((human, claim["label"]))
    if not pairs:
        print("Chưa có dòng nào được chấm tay.")
        return
    kappa = cohen_kappa(pairs)
    acc = sum(a == b for a, b in pairs) / len(pairs)
    cm = Counter(pairs)
    result = {"pairs": len(pairs), "unlabeled_or_unmatched": missing, "agreement": acc, "cohen_kappa": kappa,
              "confusion_human_vs_judge": {f"{a}->{b}": v for (a, b), v in sorted(cm.items())},
              "reading": "kappa ≥ 0,6: đồng thuận khá trở lên → dùng được số liệu giám khảo (ghi kèm kappa); "
                         "thấp hơn: sửa prompt rồi chấm lại, hoặc ghi rõ là hạn chế."}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "faithfulness_agreement.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_summarize(args):
    judgments = _latest_judgments()
    agreement_path = OUT_DIR / "faithfulness_agreement.json"
    agreement = json.loads(agreement_path.read_text(encoding="utf-8")) if agreement_path.is_file() else None
    per = defaultdict(lambda: Counter())
    faith = defaultdict(list)
    models = defaultdict(set)
    for (rid, cfg), j in judgments.items():
        if j.get("judge_model"):
            models[cfg].add(j["judge_model"])
        if j.get("faithfulness") is not None:
            faith[cfg].append(j["faithfulness"])
        for c in j.get("claims") or []:
            per[cfg][c["label"]] += 1
            per[cfg]["quote_rejected"] += int(c.get("judge_label") != c["label"])
    summary = {"prompt_version": PROMPT_VERSION, "human_agreement": agreement, "configs": {}}
    for cfg in sorted(set(per) | set(faith)):
        total = sum(per[cfg][l] for l in LABELS)
        summary["configs"][cfg] = {
            "answers": len(faith[cfg]),
            "claims": total,
            "faithfulness_mean": sum(faith[cfg]) / len(faith[cfg]) if faith[cfg] else None,
            "supported_rate": per[cfg]["SUPPORTED"] / total if total else None,
            "contradicted_rate": per[cfg]["CONTRADICTED"] / total if total else None,
            "not_enough_info_rate": per[cfg]["NOT_ENOUGH_INFO"] / total if total else None,
            "judge_quotes_rejected": per[cfg]["quote_rejected"],
            "judge_models": sorted(models[cfg]),
        }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not agreement:
        print("\n⚠️ Chưa có kết quả đồng thuận với người chấm — chưa nên báo cáo các số trên như kết quả chính thức.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("judge", "export-annotation", "agreement", "summarize"):
        sp = sub.add_parser(name)
        sp.add_argument("--run-dir", default=str(DATA_DIR))
        sp.add_argument("--benchmark", default=str(DATA_DIR / "foodhealthqa_eval_set.jsonl"))
        sp.add_argument("--configs", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
        if name == "judge":
            sp.add_argument("--provider", choices=["gemini", "ollama"], default=None)
            sp.add_argument("--judge-model", default=None, help="Model giám khảo (nên khác model sinh câu trả lời)")
            sp.add_argument("--max-new", type=int, default=0)
        if name == "export-annotation":
            sp.add_argument("--n", type=int, default=50)
            sp.add_argument("--seed", type=int, default=13)
        if name == "agreement":
            sp.add_argument("--annotation", default=str(ANNOTATION))
    args = ap.parse_args()
    {"judge": cmd_judge, "export-annotation": cmd_export, "agreement": cmd_agreement, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
