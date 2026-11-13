"""
Evaluate passage retrieval against benchmark gold evidence (no LLM needed).

For each scored benchmark record the query is the question text. Gold:
  - ins_codes_matched  -> INS hit@k (a retrieved passage mentions a gold INS)
  - source_citations   -> citation recall@k (gold citation ids covered)
  - source_claim_locators -> locator hit@k (a retrieved passage has a gold locator)

Usage:
  python -I scripts_research/eval_passage_retrieval.py [--data data/foodhealthqa_eval_set.jsonl]
         [--modes bm25 tfidf dense hybrid hybrid_exact hybrid_dense hybrid_dense_exact] [--dense-model bge-m3] [--ks 1 3 5 10] [--out data/passage_retrieval_eval.json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.passage_retriever import PassageRetriever, normalize_ins  # noqa: E402
from modules.ollama_embedder import OllamaEmbedder  # noqa: E402


def _split(s):
    return [x.strip() for x in str(s or "").replace(",", ";").split(";") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "foodhealthqa_eval_set.jsonl"))
    ap.add_argument("--modes", nargs="+", default=None,
                    help="Mặc định: tất cả chế độ; chế độ dense chỉ chạy khi có --dense-model")
    ap.add_argument("--dense-model", default="bge-m3",
                    help="Model embedding Ollama cho nhánh dense (vd bge-m3); 'none' để tắt")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--linking", choices=["oracle", "none"], default="oracle",
                    help="oracle: append gold-linked INS codes to the query (simulates entity linking and "
                         "isolates retrieval quality); none: question text only")
    ap.add_argument("--out", default=str(ROOT / "data" / "passage_retrieval_eval.json"))
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    encoder = None
    if args.dense_model and args.dense_model.lower() != "none":
        try:
            encoder = OllamaEmbedder(args.dense_model)
            print(f"dense: embedding {args.dense_model} qua Ollama (cache: {encoder.cache_path.name})")
        except Exception as exc:
            print(f"[WARN] không dùng được dense: {exc}")
    retr = PassageRetriever(dense_encoder=encoder)
    if args.modes is None:
        args.modes = [m for m in PassageRetriever.MODES if encoder is not None or m not in PassageRetriever.DENSE_MODES]
    kinds = {}
    for p in retr.passages:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(f"corpus: {len(retr.passages)} passages {kinds}")

    results = {"data": args.data, "linking": args.linking, "dense_model": args.dense_model if encoder is not None else None, "n_corpus": len(retr.passages), "corpus_kinds": kinds, "runs": {}}
    for mode in args.modes:
        agg = {k: {"ins": [0, 0], "cit": [0.0, 0], "loc": [0, 0]} for k in args.ks}
        for r in records:
            gold_ins = {normalize_ins(c) for c in _split(r.get("ins_codes_matched"))}
            gold_cit = set(_split(r.get("source_citations")))
            gold_loc = set(_split(r.get("source_claim_locators")))
            query = r.get("question", "")
            if args.linking == "oracle" and gold_ins:
                query += " " + " ".join(f"INS {c}" for c in _split(r.get("ins_codes_matched")))
            for k in args.ks:
                top = retr.retrieve(query, top_k=k, mode=mode)  # per-k call: ordering depends on k
                if gold_ins:
                    agg[k]["ins"][1] += 1
                    agg[k]["ins"][0] += any(gold_ins & set(p["ins_codes"]) for p in top)
                if gold_cit:
                    agg[k]["cit"][1] += 1
                    agg[k]["cit"][0] += len(gold_cit & {p["citation_id"] for p in top}) / len(gold_cit)
                if gold_loc:
                    agg[k]["loc"][1] += 1
                    agg[k]["loc"][0] += any(p["locator"] in gold_loc for p in top)
        results["runs"][mode] = {
            f"@{k}": {
                "ins_hit": round(v["ins"][0] / v["ins"][1], 4) if v["ins"][1] else None,
                "citation_recall": round(v["cit"][0] / v["cit"][1], 4) if v["cit"][1] else None,
                "locator_hit": round(v["loc"][0] / v["loc"][1], 4) if v["loc"][1] else None,
                "n_ins": v["ins"][1], "n_cit": v["cit"][1], "n_loc": v["loc"][1],
            } for k, v in agg.items()
        }
        print(mode, json.dumps({k: {m: x for m, x in v.items() if not m.startswith("n_")}
                                for k, v in results["runs"][mode].items()}))
    if encoder is not None:
        encoder.save_cache()
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
