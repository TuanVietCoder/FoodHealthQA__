"""Report benchmark results the way the dataset's limits require.

Usage
  python scripts_research/benchmark_report.py                        # dataset profile + trivial baselines
  python scripts_research/benchmark_report.py --predictions P.jsonl   # P: one {"id", "safety_label"} per line
  python scripts_research/benchmark_report.py --predictions B.jsonl --compare A.jsonl
                                                                      # paired photo-level bootstrap of B - A

Why not one accuracy number:
  * labels are skewed inside each domain, so every result is shown next to the "always predict the majority
    label" baseline and as macro-F1;
  * the question template predicts the gold label very well (see shortcut baselines), so a system must beat the
    template baseline, not only the majority baseline. The cross-validated version (templates learned on other
    photos) is the fair one; the in-sample version is an upper bound;
  * questions are templated (template_id): plain accuracy, template-balanced accuracy and capped-template
    accuracy (each template counts at most TEMPLATE_CAP questions) are all reported;
  * the ~1-10 questions of one photo are not independent (split_group), so confidence intervals resample photos;
  * gold_safety_label merges PERMITTED_LISTED into SAFE and NOT_DECLARED into CAUTION. The sub-type breakdown
    (label_v5) and the "strict" rows without those two sub-types show what the merge hides;
  * EM here is label exact match (= accuracy over the 5 safety labels); free-text answers are not scored by EM.
    Per-class precision/recall/F1 are printed so rare labels (CRITICAL_ALERT) are not hidden by SAFE;
  * tier core is the primary score, tier silver (all vulnerable_groups) is reported separately.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "foodhealthqa_eval_set.jsonl"  # bộ đánh giá 750 câu, bộ duy nhất của hệ thống
LABELS = ["SAFE", "CAUTION", "UNSAFE", "CRITICAL_ALERT", "ABSTAIN"]
MERGED_SUBTYPES = ("PERMITTED_LISTED", "NOT_DECLARED")  # collapsed into SAFE / CAUTION in gold_safety_label
TEMPLATE_CAP = 30
CV_FOLDS = 5
MIN_CELL = 30  # a domain x label (or persona) cell below this is reported as counts + Wilson interval, never as F1


def load(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("eligible_for_scoring") == "yes"]


def load_predictions(path: Path) -> dict:
    preds = {}
    for l in path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            p = json.loads(l)
            preds[p["id"]] = p.get("safety_label", "ABSTAIN")
    return preds


def accuracy(pairs) -> float:
    return sum(g == p for g, p in pairs) / len(pairs) if pairs else float("nan")


def class_f1(pairs, lab) -> tuple[float, float, float]:
    tp = sum(g == lab and p == lab for g, p in pairs)
    fp = sum(g != lab and p == lab for g, p in pairs)
    fn = sum(g == lab and p != lab for g, p in pairs)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return prec, rec, f1


def macro_f1(pairs) -> float:
    labs = {g for g, _ in pairs}  # labels that exist in gold only
    return sum(class_f1(pairs, lab)[2] for lab in labs) / len(labs) if labs else float("nan")


def template_balanced_accuracy(rows, preds) -> float:
    by = defaultdict(list)
    for r in rows:
        by[r["template_id"]].append(r["gold_safety_label"] == preds[r["id"]])
    return sum(sum(v) / len(v) for v in by.values()) / len(by) if by else float("nan")


def capped_template_accuracy(rows, preds, cap: int = TEMPLATE_CAP) -> float:
    """Each row weighs min(n_t, cap)/n_t, so one template contributes at most `cap` questions."""
    n = Counter(r["template_id"] for r in rows)
    num = den = 0.0
    for r in rows:
        w = min(n[r["template_id"]], cap) / n[r["template_id"]]
        num += w * (r["gold_safety_label"] == preds[r["id"]])
        den += w
    return num / den if den else float("nan")


def majority_pred(rows) -> dict:
    """Majority label of each domain (the trivial baseline a system must beat)."""
    top = {}
    for d in {r["domain"] for r in rows}:
        top[d] = Counter(r["gold_safety_label"] for r in rows if r["domain"] == d).most_common(1)[0][0]
    return {r["id"]: top[r["domain"]] for r in rows}


def _majority_by(train, key, fallback):
    by = defaultdict(Counter)
    for r in train:
        by[key(r)][r["gold_safety_label"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in by.items()}, fallback


def template_pred(rows, cross_validated: bool = True, seed: int = 7) -> dict:
    """Predict the majority gold label of the question template (falls back to the domain majority).

    cross_validated=False learns on the same rows it predicts: an optimistic upper bound.
    cross_validated=True learns on other photos only (grouped k-fold by split_group): the fair shortcut baseline.
    """
    dom_top = majority_pred(rows)
    if not cross_validated:
        top, _ = _majority_by(rows, lambda r: r["template_id"], None)
        return {r["id"]: top.get(r["template_id"], dom_top[r["id"]]) for r in rows}
    groups = sorted({r["split_group"] for r in rows})
    random.Random(seed).shuffle(groups)
    fold_of = {g: i % CV_FOLDS for i, g in enumerate(groups)}
    out = {}
    for f in range(CV_FOLDS):
        train = [r for r in rows if fold_of[r["split_group"]] != f]
        top, _ = _majority_by(train, lambda r: r["template_id"], None)
        for r in rows:
            if fold_of[r["split_group"]] == f:
                out[r["id"]] = top.get(r["template_id"], dom_top[r["id"]])
    return out


def _cluster_bootstrap(rows, stat, n=1000, seed=7):
    groups = defaultdict(list)
    for r in rows:
        groups[r["split_group"]].append(r)
    keys = list(groups)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in groups[k]]
        vals.append(stat(sample))
    vals.sort()
    return vals[int(0.025 * n)], vals[min(int(0.975 * n), n - 1)]


def bootstrap_ci(rows, preds, metric, n=1000, seed=7):
    return _cluster_bootstrap(
        rows, lambda s: metric([(r["gold_safety_label"], preds[r["id"]]) for r in s]), n=n, seed=seed
    )


def paired_diff_ci(rows, preds_a, preds_b, metric, n=1000, seed=7):
    """95% interval of metric(B) - metric(A) on the same resampled photos (A, B scored on the same rows)."""
    def stat(s):
        return metric([(r["gold_safety_label"], preds_b[r["id"]]) for r in s]) - metric(
            [(r["gold_safety_label"], preds_a[r["id"]]) for r in s]
        )
    point = stat(rows)
    lo, hi = _cluster_bootstrap(rows, stat, n=n, seed=seed)
    return point, lo, hi


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (stable for small n and for k = 0 or k = n)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def small_cells(rows, min_cell: int = MIN_CELL) -> dict:
    """{'domain|label': n} and {'persona:<id>': n} for every cell that has fewer than min_cell scored records."""
    out = {}
    for (d, lab), n in sorted(Counter((r["domain"], r["gold_safety_label"]) for r in rows).items()):
        if n < min_cell:
            out[f"{d}|{lab}"] = n
    for pid, n in sorted(Counter(r["persona_id"] for r in rows).items()):
        if n < min_cell:
            out[f"persona:{pid}"] = n
    return out


def small_cell_report(rows):
    cells = small_cells(rows)
    print(f"-- cells with fewer than {MIN_CELL} records: report counts and a Wilson interval only, no per-label F1")
    if not cells:
        print("   none")
    for k, n in cells.items():
        lo, hi = wilson(n, n)
        print(f"   {k:<40} n={n:<3} if a system gets all {n} right, the 95% lower bound is {lo:.2f}")


def abstention_metrics(rows, preds):
    """ABSTAIN as a binary decision (answer vs refuse). This is how the insufficient_data domain should be read: its
    non-ABSTAIN labels are too few for per-label scores, but 'does the system know when to refuse' is well measured."""
    print("-- abstention as a binary decision (gold ABSTAIN vs answerable)")
    for title, part in (("all scored records", rows),
                        ("insufficient_data only", [r for r in rows if r["domain"] == "insufficient_data"])):
        gold_abs = [r for r in part if r["gold_safety_label"] == "ABSTAIN"]
        gold_ans = [r for r in part if r["gold_safety_label"] != "ABSTAIN"]
        if not gold_abs or not gold_ans:
            continue
        tp = sum(preds[r["id"]] == "ABSTAIN" for r in gold_abs)
        fp = sum(preds[r["id"]] == "ABSTAIN" for r in gold_ans)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / len(gold_abs)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rlo, rhi = wilson(tp, len(gold_abs))
        flo, fhi = wilson(fp, len(gold_ans))
        print(f"   {title:<24} gold_abstain={len(gold_abs):<4} answerable={len(gold_ans):<4} "
              f"abstain P={prec:.3f} R={rec:.3f} [{rlo:.3f},{rhi:.3f}] F1={f1:.3f} | "
              f"false-abstain on answerable={fp / len(gold_ans):.3f} [{flo:.3f},{fhi:.3f}]")


def block(title, rows, preds):
    if not rows:
        return
    pairs = [(r["gold_safety_label"], preds[r["id"]]) for r in rows]
    lo, hi = bootstrap_ci(rows, preds, accuracy)
    mlo, mhi = bootstrap_ci(rows, preds, macro_f1)
    base = majority_pred(rows)
    print(f"{title:<34} n={len(rows):<5} EM/acc={accuracy(pairs):.3f} [{lo:.3f},{hi:.3f}] "
          f"macroF1={macro_f1(pairs):.3f} [{mlo:.3f},{mhi:.3f}] "
          f"tmplBal={template_balanced_accuracy(rows, preds):.3f} cap{TEMPLATE_CAP}={capped_template_accuracy(rows, preds):.3f} "
          f"majority={accuracy([(r['gold_safety_label'], base[r['id']]) for r in rows]):.3f}")


def per_class_table(rows, preds):
    pairs = [(r["gold_safety_label"], preds[r["id"]]) for r in rows]
    print("-- per-class precision / recall / F1 (core+silver)")
    for lab in LABELS:
        n = sum(g == lab for g, _ in pairs)
        if n:
            p, rc, f = class_f1(pairs, lab)
            note = f"  (support < {MIN_CELL}: indicative only)" if n < MIN_CELL else ""
            print(f"{lab:<16} support={n:<5} P={p:.3f} R={rc:.3f} F1={f:.3f}{note}")


def subtype_breakdown(rows, preds):
    """What the system predicted for each label_v5 sub-type (PERMITTED_LISTED/NOT_DECLARED are merged in gold)."""
    print("-- predictions per label_v5 sub-type (hidden by the gold merge)")
    for sub in sorted({r["label_v5"] for r in rows}):
        part = [r for r in rows if r["label_v5"] == sub]
        dist = Counter(preds[r["id"]] for r in part)
        gold = part[0]["gold_safety_label"] if len({r["gold_safety_label"] for r in part}) == 1 else "mixed"
        hit = sum(preds[r["id"]] == r["gold_safety_label"] for r in part) / len(part)
        print(f"{sub:<17} n={len(part):<5} gold->{gold:<14} acc={hit:.3f} predicted={dict(dist.most_common())}")


def report(rows, preds):
    core = [r for r in rows if r["tier"] == "core"]
    block("ALL core (primary)", core, preds)
    block("ALL silver (secondary)", [r for r in rows if r["tier"] == "silver"], preds)
    block("core strict (no merged sub-types)", [r for r in core if r["label_v5"] not in MERGED_SUBTYPES], preds)
    print("-- per domain x tier")
    for d in sorted({r["domain"] for r in rows}):
        for t in ("core", "silver"):
            block(f"{d} / {t}", [r for r in rows if r["domain"] == d and r["tier"] == t], preds)
    per_class_table(rows, preds)
    abstention_metrics(rows, preds)
    subtype_breakdown(rows, preds)
    unsafe = [r for r in rows if r["gold_safety_label"] in ("UNSAFE", "CRITICAL_ALERT")]
    if unsafe:
        miss = sum(preds[r["id"]] == "SAFE" for r in unsafe) / len(unsafe)
        print(f"false-SAFE rate on UNSAFE/CRITICAL_ALERT: {miss:.3f}")


def baselines(rows):
    print("\n== baselines (what a system must beat)")
    print("-- A. majority label per domain")
    report(rows, majority_pred(rows))
    print("\n-- B. majority label per question template, learned on OTHER photos (shortcut baseline, fair)")
    cv = template_pred(rows, cross_validated=True)
    block("ALL core", [r for r in rows if r["tier"] == "core"], cv)
    block("ALL silver", [r for r in rows if r["tier"] == "silver"], cv)
    for d in sorted({r["domain"] for r in rows}):
        block(f"{d}", [r for r in rows if r["domain"] == d], cv)
    print("-- C. same, learned on the rows it predicts (optimistic upper bound)")
    block("ALL core", [r for r in rows if r["tier"] == "core"], template_pred(rows, cross_validated=False))


def profile(rows):
    print(f"scored records: {len(rows)}, photos: {len({r['split_group'] for r in rows})}, "
          f"templates: {len({r['template_id'] for r in rows})}")
    for d in sorted({r["domain"] for r in rows}):
        sub = [r for r in rows if r["domain"] == d]
        c = Counter(r["gold_safety_label"] for r in sub)
        t = Counter(r["template_id"] for r in sub)
        print(f"{d:<22} n={len(sub):<4} labels={dict(c)} majority={c.most_common(1)[0][1] / len(sub):.2f} "
              f"templates={len(t)} top3_template_share={sum(v for _, v in t.most_common(3)) / len(sub):.2f}")
    small_cell_report(rows)


def _complete(rows, preds):
    """Missing ids count as ABSTAIN so a partial run cannot look better than it is."""
    full = dict(preds)
    for r in rows:
        full.setdefault(r["id"], "ABSTAIN")
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument(
        "--eval-750",
        action="store_true",
        help="Sử dụng bộ đánh giá phân tầng chuẩn 750 câu (data/foodhealthqa_eval_set.jsonl)",
    )
    ap.add_argument("--predictions", type=Path)
    ap.add_argument("--compare", type=Path, help="reference predictions; prints a paired bootstrap of predictions - compare")
    args = ap.parse_args()
    data_path = (ROOT / "data" / "foodhealthqa_eval_set.jsonl") if args.eval_750 else args.data
    all_rows = load(data_path)
    profile(all_rows)
    baselines(all_rows)
    if args.predictions:
        preds = load_predictions(args.predictions)
        rows = [r for r in all_rows if r["id"] in preds]
        print(f"\n== system predictions ({len(rows)} matched, {len(preds) - len(rows)} ids not in the scored set)")
        full = _complete(rows, preds)
        report(rows, full)
        if args.compare:
            other = load_predictions(args.compare)
            both = [r for r in rows if r["id"] in other]
            print(f"\n== paired difference {args.predictions.name} - {args.compare.name} on {len(both)} shared ids")
            for name, rs in (("core", [r for r in both if r["tier"] == "core"]), ("silver", [r for r in both if r["tier"] == "silver"])):
                if not rs:
                    continue
                for mname, m in (("EM/acc", accuracy), ("macroF1", macro_f1)):
                    pt, lo, hi = paired_diff_ci(rs, other, full, m)
                    flag = "significant" if lo > 0 or hi < 0 else "not significant"
                    print(f"{name:<7} {mname:<8} diff={pt:+.3f} [{lo:+.3f},{hi:+.3f}] {flag}")


if __name__ == "__main__":
    main()
