"""Tải các trang JECFA Chemical/<id> mà KB/catalog trích dẫn nhưng chưa có file trong raw/external/jecfa/.

Chỉ lưu một trang khi nội dung đúng là bản ghi JECFA (có "Functional Class" hoặc "Evaluation year" và tên chất);
trang lỗi/chặn bị bỏ qua và báo lại, không ghi đè file đã có.
Ghi sha256 + thời điểm tải vào knowledge_base/sources/derived/jecfa_pages_fetched.json.

Chạy: .\venv\Scripts\python.exe scripts_research\fetch_jecfa_missing_pages.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_jecfa as fj  # noqa: E402

ROOT = fj.ROOT
OUT_LOG = ROOT / "knowledge_base" / "sources" / "derived" / "jecfa_pages_fetched.json"
SKIP_DIRS = {"raw"}


def referenced_ids() -> set[int]:
    ids: set[int] = set()
    for p in (ROOT / "knowledge_base").rglob("*.json"):
        if SKIP_DIRS & set(p.relative_to(ROOT / "knowledge_base").parts):
            continue
        ids |= {int(x) for x in re.findall(r"Home/Chemical/(\d+)", p.read_text(encoding="utf-8", errors="ignore"))}
    return ids


def local_ids() -> set[int]:
    return {int(m.group(1)) for p in fj.OUT_DIR.glob("Chemical_*.html") if (m := re.match(r"Chemical_(\d+)", p.name))}


def looks_like_record(text: str) -> bool:
    return "Overview" in text and "Evaluations" in text and ("Functional Class" in text or "Evaluation year" in text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ids", type=int, nargs="*", default=[], help="tải thêm các id này, ngoài id KB/catalog đang trích dẫn")
    args = ap.parse_args()
    missing = sorted((referenced_ids() | set(args.ids)) - local_ids())
    print(f"Thiếu {len(missing)} trang: {missing}")
    if args.dry_run:
        return 0
    log = json.loads(OUT_LOG.read_text(encoding="utf-8")) if OUT_LOG.exists() else {}
    ok, bad = [], []
    for cid in missing:
        url = f"{fj.BASE}/Home/Chemical/{cid}"
        raw = fj.curl(url)
        text = fj.page_text(raw)
        if not looks_like_record(text):
            bad.append(cid)
            print(f"LỖI {cid}: không phải bản ghi JECFA ({len(raw)} byte)", flush=True)
            continue
        path = fj.OUT_DIR / f"Chemical_{cid}.html"
        path.write_bytes(raw)
        info = fj.parse_page(text)
        log[str(cid)] = {"url": url, "file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                         "ins_on_page": info["ins_on_page"], "latest_evaluation_year": info["latest_evaluation_year"],
                         "latest_adi": info["latest_adi"], "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        ok.append(cid)
        print(f"OK  {cid}: INS {info['ins_on_page']} ADI {info['latest_adi'] or '-'}", flush=True)
        OUT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(0.5)
    print(f"Xong: {len(ok)} đã tải, {len(bad)} lỗi {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
