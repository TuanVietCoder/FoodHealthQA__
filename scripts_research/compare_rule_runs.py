"""So sánh hai lần đo tầng luật (calibrate_confidence.py --stage rules) ở ngưỡng từ chối 0,65.

Dùng: python scripts_research/compare_rule_runs.py <thư mục trước> <thư mục sau>
In ra, cho cấu hình B và C: số câu đúng (tổng, phần phát triển, phần kiểm tra), số câu sửa đúng, số câu đang đúng bị
sai đi (quan trọng nhất), các câu đổi kết quả nhưng vẫn sai, và các nhóm lỗi còn lại. Không gọi LLM.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THRESHOLD = 0.65


def load(directory):
    path = Path(directory) / "rule_runs.jsonl"
    return {(r["id"], r["config"]): r for r in map(json.loads, open(path, encoding="utf-8") if path.is_file() else [])}


def pred(r):
    return r["label_threshold_on"] if r["s_total"] < THRESHOLD else r["label_threshold_off"]


def split_of(record):
    key = str(record.get("split_group") or record.get("image_ref") or record["id"])
    return "dev" if int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 1000 < 500 else "test"


def main(before_dir, after_dir):
    rows = {}
    for line in open(ROOT / "data" / "foodhealthqa_eval_set.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    a, b = load(before_dir), load(after_dir)
    keys = set(a) & set(b)
    print(f"Trước: {before_dir}  ({len(a)} dòng)\nSau:   {after_dir}  ({len(b)} dòng)\nCặp so sánh được: {len(keys)}")
    if len(keys) < 1500:
        print("CẢNH BÁO: chưa đủ 1500 cặp (750 câu x B và C); số dưới đây chỉ tính các câu đã có ở cả hai bên.")
    for cfg in "BC":
        ks = sorted(k for k in keys if k[1] == cfg)
        if not ks:
            continue
        print(f"\n===== Cấu hình {cfg} ({len(ks)} câu) =====")
        for name, sel in (("Tổng", None), ("Phát triển", "dev"), ("Kiểm tra", "test")):
            kk = [k for k in ks if sel is None or split_of(rows[k[0]]) == sel]
            ok = lambda d: sum(pred(d[k]) == d[k]["gold_label"] for k in kk)  # noqa: E731
            print(f"  {name:<11} {ok(a)} -> {ok(b)} / {len(kk)}")
        gold = lambda k: a[k]["gold_label"]  # noqa: E731
        fixed = [k for k in ks if pred(a[k]) != gold(k) and pred(b[k]) == gold(k)]
        broke = [k for k in ks if pred(a[k]) == gold(k) and pred(b[k]) != gold(k)]
        other = [k for k in ks if pred(a[k]) != pred(b[k]) and k not in fixed and k not in broke]
        print(f"  Sửa đúng: {len(fixed)}   ĐANG ĐÚNG BỊ SAI: {len(broke)}   đổi nhưng vẫn sai: {len(other)}")
        for k in broke + other:
            q = rows[k[0]]
            tag = "HỎNG" if k in broke else "đổi-vẫn-sai"
            print(f"    [{tag}] {k[0]} {q['persona_id']} {q['template_id']} đáp án={gold(k)} {pred(a[k])} -> {pred(b[k])} | {q['question'][:80]}")
        errs = collections.Counter((gold(k), pred(b[k])) for k in ks if pred(b[k]) != gold(k))
        print("  Lỗi còn lại (đáp án, hệ thống):", dict(errs.most_common()))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows: tiếng Việt không lỗi dù chưa đặt PYTHONIOENCODING
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
