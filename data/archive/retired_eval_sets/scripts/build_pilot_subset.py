"""Tạo tập pilot phân tầng từ data/qa_eval_balanced.jsonl (không sửa benchmark gốc).

Lấy đều theo từng ô (lĩnh vực, nhãn an toàn) với seed cố định để lặp lại được, mỗi ảnh tối đa
--max-per-image câu để pilot phủ nhiều ảnh/mẫu hỏi khác nhau. Mọi bản ghi vẫn là bản ghi thật của benchmark.

Chạy: python scripts_research/build_pilot_subset.py [--size 150] [--seed 42]
Ghi:  data/qa_eval_pilot.jsonl và data/qa_eval_pilot_summary.json
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "qa_eval_balanced.jsonl"
OUT = ROOT / "data" / "qa_eval_pilot.jsonl"
SUMMARY = ROOT / "data" / "qa_eval_pilot_summary.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-image", type=int, default=2)
    args = ap.parse_args()

    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(args.seed)
    cells = defaultdict(list)
    for row in rows:
        cells[(row.get("domain", "?"), row.get("gold_safety_label", "?"))].append(row)
    for items in cells.values():
        rng.shuffle(items)

    chosen, per_image = [], Counter()
    keys = sorted(cells)
    # vòng tròn qua các ô cho đến khi đủ số lượng: mọi (lĩnh vực, nhãn) đều có mặt
    while len(chosen) < args.size and any(cells[k] for k in keys):
        for key in keys:
            while cells[key]:
                row = cells[key].pop()
                if per_image[row.get("image_ref")] < args.max_per_image:
                    per_image[row.get("image_ref")] += 1
                    chosen.append(row)
                    break
            if len(chosen) >= args.size:
                break

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in chosen) + "\n", encoding="utf-8")
    summary = {
        "source": str(SRC.relative_to(ROOT)),
        "size": len(chosen),
        "seed": args.seed,
        "by_domain": dict(Counter(r["domain"] for r in chosen)),
        "by_label": dict(Counter(r["gold_safety_label"] for r in chosen)),
        "by_domain_label": {f"{d}|{l}": n for (d, l), n in sorted(Counter((r["domain"], r["gold_safety_label"]) for r in chosen).items())},
        "n_images": len({r.get("image_ref") for r in chosen}),
        "note": "Tập pilot để kiểm tra pipeline và so sánh sơ bộ; nhóm quá nhỏ không báo cáo metric theo nhãn.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("size", "n_images", "by_domain", "by_label")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
