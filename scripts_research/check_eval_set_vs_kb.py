"""Bộ QA 750 câu còn khớp với KB hiện tại không? (KHÔNG gọi LLM, KHÔNG ghi file dự án trừ --report)

Kiểm 1 – Nhãn: chạy Cấu hình B (văn bản nhãn tham chiếu → linker → bộ luật) cho mọi câu,
         so nhãn an toàn hệ thống với gold_safety_label đã gán. In ma trận nhầm lẫn và các câu lệch.
Kiểm 2 – Nội dung đáp án mẫu: với mỗi phụ gia KB đã bị SỬA theo nguồn (adi_assessment.display_text_before_review),
         tìm câu có mã đó mà reference_answer còn nhắc ADI → có thể đang ghi giá trị cũ; cùng vài chuỗi số đã sửa
         (sulfit 5–10%, nitrit dưới 6 tháng).

Chạy: .\\venv\\Scripts\\python.exe scripts_research\\check_eval_set_vs_kb.py [--limit N] [--report FILE.json] [--skip-labels]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EVAL = ROOT / "data" / "foodhealthqa_eval_set.jsonl"
KB = ROOT / "knowledge_base" / "additives_standard.json"
# Chuỗi giá trị cũ đã sửa theo nguồn khi rà lưu ý sức khỏe (2026-11-22)
OLD_VALUE_PATTERNS = {
    "sulfit/hen 5–10% (nguồn: 3–10%)": r"5\s*[-–]\s*10\s*%",
    "nitrit trẻ dưới 6 tháng (nguồn: dưới 3 tháng)": r"(?:nitrit|nitrite|INS ?25[01]).{0,200}d(?:ư|u)ới 6 tháng",
    "phosphat hấp thu 90–100% (đã bỏ)": r"90\s*[-–]\s*100\s*%",
}


def norm(code: str) -> str:
    return re.sub(r"[()\s]", "", re.sub(r"^(ins|e)\s*", "", str(code or "").strip(), flags=re.I)).lower()


def check_labels(records) -> dict:
    import generate_predictions as gp
    from modules import pipeline

    resources = gp._load_resources()
    pairs = {}
    with mock.patch.object(pipeline, "generate_llm_answer", return_value=None):
        for n, rec in enumerate(records, 1):
            if n == 1 or n % 50 == 0 or n == len(records):
                print(f"  nhãn {n}/{len(records)}", flush=True)
            pred = gp._predict_one("B", rec, resources, {}, None)
            pairs[rec["id"]] = (rec["gold_safety_label"], pred.get("safety_label"))
    return pairs


def check_reference_text(records) -> list[dict]:
    kb = json.loads(KB.read_text(encoding="utf-8"))
    corrected = {}
    for it in kb:
        adi = it.get("adi_assessment", {})
        if adi.get("display_text_before_review"):
            for c in it["ins_code"].replace("INS", "").split("+"):
                corrected[norm(c)] = (it["ins_code"], adi["display_text_before_review"], adi["display_text"])
    hits = []
    for rec in records:
        ref = rec.get("reference_answer") or ""
        codes = {norm(c) for c in re.split(r"[,;]", rec.get("ins_codes_matched") or "") if c.strip()}
        for c in codes & set(corrected):
            code, old, new = corrected[c]
            # đáp án nhắc ADI của đúng mã này (mã nằm cách chữ ADI không quá 250 ký tự)
            pat = rf"(?:INS\s*{re.escape(c)}\b.{{0,250}}\bADI\b|\bADI\b.{{0,250}}INS\s*{re.escape(c)}\b)"
            if re.search(pat, ref, re.I | re.S):
                hits.append({"id": rec["id"], "kind": "adi_corrected", "ins": code, "kb_old": old[:120], "kb_now": new[:120]})
        for label, pat in OLD_VALUE_PATTERNS.items():
            m = re.search(pat, ref, re.I | re.S)
            if m:
                hits.append({"id": rec["id"], "kind": "old_value", "what": label, "text": ref[max(0, m.start() - 60):m.end() + 40]})
    return hits


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-labels", action="store_true", help="chỉ kiểm nội dung đáp án mẫu (nhanh)")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args(argv)
    records = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]
    report = {}

    hits = check_reference_text(records)
    report["reference_text"] = hits
    print(f"\nKIỂM 2 – đáp án mẫu có thể chứa giá trị KB đã sửa: {len({h['id'] for h in hits})} câu")
    for k, v in collections.Counter(h.get("ins") or h.get("what") for h in hits).most_common():
        print(f"   {v:4}  {k}")

    if not args.skip_labels:
        pairs = check_labels(records)
        same = sum(g == p for g, p in pairs.values())
        print(f"\nKIỂM 1 – nhãn bộ luật (Cấu hình B) khớp nhãn gán: {same}/{len(pairs)} ({same / len(pairs):.1%})")
        cm = collections.Counter(pairs.values())
        labels = sorted({x for p in pairs.values() for x in p if x})
        print("   gold \\ hệ thống " + " ".join(f"{lab[:8]:>9}" for lab in labels))
        for g in labels:
            print(f"   {g[:16]:16} " + " ".join(f"{cm.get((g, p), 0):9}" for p in labels))
        diff = {k: v for k, v in pairs.items() if v[0] != v[1]}
        report["label_mismatches"] = diff
        for k, (g, p) in list(diff.items())[:25]:
            print(f"   lệch {k}: gán {g} → hệ thống {p}")
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print("Đã ghi", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
