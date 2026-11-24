"""Đồng bộ và kiểm tra mục "jecfa_pages" của knowledge_base/sources/catalog/external_sources_manifest.json với thư mục
knowledge_base/sources/raw/external/jecfa/.

  Không tham số : đăng ký các trang Chemical_<id>.html có trên đĩa nhưng chưa có trong manifest (kèm sha256, số byte).
  --check       : chỉ kiểm tra, không ghi; thoát mã 1 nếu có trang chưa đăng ký, bị mất, hoặc sha256/số byte lệch.

Không bao giờ sửa dòng đã có: dòng lệch được báo lỗi để người dùng quyết định (tải lại hay đăng ký lại), không tự ghi đè.
Thời điểm tải (retrieved_utc) lấy từ derived/jecfa_pages_fetched.json hoặc derived/jecfa_adi_verification*.json; không có
thì để rỗng chứ không bịa.

Chạy: .\venv\Scripts\python.exe scripts_research\sync_jecfa_manifest.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "knowledge_base" / "sources" / "catalog" / "external_sources_manifest.json"
JECFA_DIR = ROOT / "knowledge_base" / "sources" / "raw" / "external" / "jecfa"
DERIVED = ROOT / "knowledge_base" / "sources" / "derived"
BASE = "https://apps.who.int/food-additives-contaminants-jecfa-database"
PAGE = re.compile(r"^Chemical_(\d+)\.html$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def disk_pages() -> dict[str, Path]:
    return {p.name: p for p in JECFA_DIR.glob("Chemical_*.html") if PAGE.match(p.name)}


def retrieval_times() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("jecfa_adi_verification.json", "jecfa_adi_verification_more.json"):
        path = DERIVED / name
        if path.is_file():
            for rec in json.loads(path.read_text(encoding="utf-8")).values():
                if isinstance(rec, dict) and rec.get("file") and rec.get("retrieved_utc"):
                    out[rec["file"]] = rec["retrieved_utc"]
    log = DERIVED / "jecfa_pages_fetched.json"
    if log.is_file():
        for rec in json.loads(log.read_text(encoding="utf-8")).values():
            if rec.get("file") and rec.get("retrieved_utc"):
                out[rec["file"]] = rec["retrieved_utc"]
    return out


def audit(manifest: dict) -> dict[str, list[str]]:
    """Đối chiếu manifest với đĩa. Trả về các nhóm lỗi (rỗng = khớp)."""
    entries = manifest.get("jecfa_pages", [])
    on_disk = disk_pages()
    registered = {e["file"].split("/")[-1]: e for e in entries}
    problems: dict[str, list[str]] = {"chưa đăng ký": [], "mất file": [], "sha256 lệch": [], "số byte lệch": [], "trùng dòng": []}
    if len(registered) != len(entries):
        problems["trùng dòng"] = sorted({e["file"] for e in entries if sum(x["file"] == e["file"] for x in entries) > 1})
    for name in sorted(set(on_disk) - set(registered)):
        problems["chưa đăng ký"].append(name)
    for name, entry in sorted(registered.items()):
        path = on_disk.get(name)
        if path is None:
            problems["mất file"].append(name)
            continue
        if sha256(path) != entry["sha256"]:
            problems["sha256 lệch"].append(name)
        if path.stat().st_size != entry["bytes"]:
            problems["số byte lệch"].append(name)
    return {k: v for k, v in problems.items() if v}


def register_missing(manifest: dict) -> list[str]:
    entries = manifest.setdefault("jecfa_pages", [])
    have = {e["file"].split("/")[-1] for e in entries}
    times = retrieval_times()
    added = []
    for name, path in sorted(disk_pages().items(), key=lambda kv: int(PAGE.match(kv[0]).group(1))):
        if name in have:
            continue
        cid = int(PAGE.match(name).group(1))
        entries.append({"source_id": f"JECFA_CHEMICAL_{cid}", "url": f"{BASE}/Home/Chemical/{cid}",
                        "bytes": path.stat().st_size, "sha256": sha256(path), "ok": True,
                        "retrieved_utc": times.get(name, ""), "file": f"jecfa/{name}"})
        added.append(name)
    entries.sort(key=lambda e: int(e["source_id"].rsplit("_", 1)[1]))
    return added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="chỉ kiểm tra, không ghi manifest")
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not args.check:
        added = register_missing(manifest)
        if added:
            MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đăng ký thêm {len(added)} trang: {added}" if added else "Không có trang nào cần đăng ký.")
    problems = audit(manifest)
    n = len(manifest.get("jecfa_pages", []))
    if problems:
        for kind, names in problems.items():
            print(f"LỆCH — {kind} ({len(names)}): {names}")
        return 1
    print(f"Manifest khớp đĩa: {n} trang JECFA, sha256 và số byte đều khớp.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
