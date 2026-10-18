"""Reproducible retrieval ablations over a labeled entity-linking benchmark."""

from modules.hybrid_retriever import HybridRetriever


_RETRIEVER_ALPHAS = {
    "bm25": 1.0,
    "tfidf": 0.0,
    "hybrid": 0.5,
}


def run_retrieval_ablation(records, additives, top_k_values=(1, 3, 5)):
    """
    Measure entity retrieval hit rate and reciprocal rank across top-k/retriever settings.

    Records require `question` and `expected_ins_code`; output can be written as JSON
    by the CLI or consumed by tests and notebooks.
    """
    if not records:
        raise ValueError("At least one labeled benchmark record is required")
    if not additives:
        raise ValueError("The additive knowledge base is empty")
    if any(k < 1 for k in top_k_values):
        raise ValueError("top_k values must be positive integers")
    for index, record in enumerate(records):
        if not record.get("question") or not record.get("expected_ins_code"):
            raise ValueError(f"Record {index} needs question and expected_ins_code")

    results = []
    for retriever_name, alpha in _RETRIEVER_ALPHAS.items():
        retriever = HybridRetriever(alpha=alpha)
        retriever.build_index(additives)
        for top_k in top_k_values:
            hits = 0
            reciprocal_ranks = []
            for record in records:
                candidates = retriever.retrieve(record["question"], top_k=top_k)
                expected = str(record["expected_ins_code"]).casefold()
                rank = next(
                    (
                        position
                        for position, (candidate, _, _) in enumerate(candidates, 1)
                        if str(candidate.get("ins_code", "")).casefold() == expected
                    ),
                    None,
                )
                hits += rank is not None
                reciprocal_ranks.append(1 / rank if rank is not None else 0.0)
            results.append({
                "retriever": retriever_name,
                "top_k": top_k,
                "queries": len(records),
                "hit_rate": hits / len(records),
                "mean_reciprocal_rank": sum(reciprocal_ranks) / len(records),
            })
    return results
