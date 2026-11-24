"""So nhãn Cấu hình B (văn bản nhãn tham chiếu, KHÔNG gọi LLM) trên bộ đánh giá giữa hành vi CŨ và MỚI của hai thay đổi 2026-11-22.

  --notes   dòng lưu ý bệnh nền (abstention.condition_notes) + sửa nhận diện ý định (dynamic_qa._without_additive_names)
  --hybrid  khớp mã INS của bộ truy xuất lai: so cách cũ (chuỗi con '160' in '160a(ii)') với cách mới (đúng mã hoặc mã gốc/mã con)
  --old-kb FILE  so KB cũ (file JSON additives_standard sao lưu) với KB hiện tại, chỉ xét nhãn an toàn của bộ luật
  --linker  bộ liên kết tên phụ gia: "axit"=="acid" và các tên gọi thêm trong sổ (so với bản sổ ở git HEAD)

Chỉ in ra số câu đổi nhãn/trạng thái an toàn và (với --linker) số câu có tập mã INS nhận diện thay đổi. Không ghi file nào của dự án.
Cần: bản sổ git HEAD của knowledge_base/additive_registry.json (lệnh `git show` được gọi tự động). Chạy mất vài phút:
    .\venv\Scripts\python.exe scripts_research\compare_behaviour_b.py --notes
    .\venv\Scripts\python.exe scripts_research\compare_behaviour_b.py --linker
"""
from __future__ import annotations

import argparse
import collections
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_predictions as gp  # noqa: E402
from modules import abstention, additive_linker as al, dynamic_qa, pipeline  # noqa: E402


def _old_registry() -> Path:
    text = subprocess.run(["git", "show", "HEAD:knowledge_base/additive_registry.json"], cwd=ROOT, capture_output=True, check=True).stdout
    path = Path(tempfile.mkdtemp()) / "additive_registry_head.json"
    path.write_bytes(text)
    return path


def run(records, resources, mode, new, old_kb=None):
    """Trả về ({id: (nhãn, trạng thái)}, {id: thông tin phụ}) cho hành vi mới (new=True) hoặc cũ."""
    out, extra = {}, {}
    patches = [mock.patch.object(pipeline, "generate_llm_answer", return_value=None)]  # không gọi LLM
    if not new and mode == "notes":
        orig = abstention.population_notes
        patches += [mock.patch.object(abstention, "population_notes", lambda q, persona=None, linked_results=None, covered_conditions=None: orig(q, persona)),
                    mock.patch.object(dynamic_qa, "_without_additive_names", lambda q, m: q),
                    mock.patch.object(dynamic_qa, "_SWEETENER_CLASS", type("NoSub", (), {"sub": staticmethod(lambda r, t: t)})())]
    if not new and mode == "hybrid":
        import re as _re
        from modules import hybrid_retriever as hr
        patches += [mock.patch.object(hr, "_explicit_codes", lambda text: _re.findall(r"\d{3,4}", text or "")),
                    mock.patch.object(hr, "_code_matches_entity", lambda q, e: q in e)]
    if not new and mode == "linker":
        patches += [mock.patch.object(al, "_SPELLING_VARIANTS", {}), mock.patch.object(al, "DEFAULT_REGISTRY_PATH", _old_registry())]
    if mode == "kb" and not new:
        # nạp KB cũ bằng chính hàm nạp của hệ thống (giữ nguyên các bước chuẩn hoá trường), chỉ đổi file được chọn
        real_find = pipeline.find_data_file
        with mock.patch.object(pipeline, "find_data_file",
                               lambda folder, names, *a, **k: str(old_kb) if names and names[0] == "additives_standard" else real_find(folder, names, *a, **k)):
            resources = dict(resources, kb_additives=pipeline.load_knowledge_base(str(ROOT / "data")))
    for p in patches:
        p.start()
    al.clear_registry_cache()
    try:
        label = "MỚI" if new else "CŨ"
        for number, record in enumerate(records, 1):
            if number == 1 or number % 50 == 0 or number == len(records):
                print(f"  [{label}] {number}/{len(records)} câu", flush=True)
            pred = gp._predict_one("B", record, resources, {}, None)
            out[record["id"]] = (pred.get("safety_label"), pred.get("guardrail_status"))
            if mode in ("kb", "hybrid"):
                extra[record["id"]] = ()
            elif mode == "linker":
                text = gp._ground_truth_text(record)
                extra[record["id"]] = tuple(sorted({al.normalize_ins_key(x.get("ins_code")) for x in al.link_additive_mentions(text) if x.get("ins_code")}))
            else:
                plan = dynamic_qa.understand_question(record["question"])
                extra[record["id"]] = (tuple(i["code"] for i in plan["intents"]), tuple(n["code"] for n in plan["nutrients"]))
    finally:
        for p in patches:
            p.stop()
        al.clear_registry_cache()
    return out, extra


def main() -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--notes", action="store_true")
    group.add_argument("--linker", action="store_true")
    group.add_argument("--hybrid", action="store_true")
    group.add_argument("--old-kb", type=Path, help="file KB cũ để so với KB hiện tại")
    ap.add_argument("--benchmark", default=str(ROOT / "data" / "foodhealthqa_eval_set.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N câu đầu để thử nhanh (0 = toàn bộ)")
    args = ap.parse_args()
    mode = "notes" if args.notes else ("linker" if args.linker else ("hybrid" if args.hybrid else "kb"))
    print("Đang nạp dữ liệu (vài chục giây)…", flush=True)
    records = gp._load_jsonl(args.benchmark)
    if args.limit:
        records = records[: args.limit]
    resources = gp._load_resources()
    print("Lượt 1/2: hành vi CŨ", flush=True)
    before, extra_before = run(records, resources, mode, new=False, old_kb=args.old_kb)
    print("Lượt 2/2: hành vi MỚI", flush=True)
    after, extra_after = run(records, resources, mode, new=True)
    changed = [k for k in before if before[k] != after[k]]
    print(f"Số câu: {len(before)}")
    print(f"ĐỔI NHÃN/TRẠNG THÁI: {len(changed)}  " + str([(k, before[k], after[k]) for k in changed[:10]]))
    diff = [k for k in extra_before if extra_before[k] != extra_after[k]]
    if mode == "linker":
        gained = collections.Counter(c for k in diff for c in set(extra_after[k]) - set(extra_before[k]))
        lost = collections.Counter(c for k in diff for c in set(extra_before[k]) - set(extra_after[k]))
        print(f"Câu có tập mã INS nhận diện thay đổi: {len(diff)}")
        print("Mã được nhận thêm:", dict(gained.most_common(20)))
        print("Mã bị mất:", dict(lost))
    else:
        print(f"Câu đổi ý định/chất dinh dưỡng được nhận diện: {len(diff)}  {diff[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
