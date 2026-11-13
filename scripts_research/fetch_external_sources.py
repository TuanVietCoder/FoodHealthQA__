"""Download the primary regulatory documents cited by benchmark v5 into knowledge_base/sources/raw/external/.

Each file is stored with its URL, retrieval time, size and sha256 in external_sources_manifest.json,
so later verification runs offline against the exact bytes that were checked.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "knowledge_base" / "sources" / "raw" / "external"
MANIFEST = OUT.parent.parent / "catalog" / "external_sources_manifest.json"

SOURCES = {
    "US_CFR_21_172_804": ("https://www.law.cornell.edu/cfr/text/21/172.804", "html"),
    "US_CODE_21_321": ("https://www.law.cornell.edu/uscode/text/21/321", "html"),
    "JEMRA_2021_PART1_SUMMARY": (
        "https://cdn.who.int/media/docs/default-source/food-safety/jemra/1st-allergen-summary-report-10may2021.pdf",
        "pdf"),
    # NOTE: the catalog URL .../9789240076679 returns 404; the PAL report (Part 3) is item 9789240072510.
    "JEMRA_PART3_WHO_PAGE": ("https://www.who.int/publications/i/item/9789240072510", "html"),
    "EFSA_2021_TIO2_NEWS": (
        "https://www.efsa.europa.eu/en/news/titanium-dioxide-e171-no-longer-considered-safe-when-used-food-additive",
        "html"),
    "EFSA_2021_TIO2_PUBMED": ("https://pubmed.ncbi.nlm.nih.gov/33976718/", "html"),
    "US_CFR_21_101_54": ("https://www.law.cornell.edu/cfr/text/21/101.54", "html"),
    "US_CFR_21_101_61": ("https://www.law.cornell.edu/cfr/text/21/101.61", "html"),
    "FDA_BENZENE_QA": ("https://www.fda.gov/food/chemicals/questions-and-answers-occurrence-benzene-soft-drinks-and-other-beverages", "html"),
    "WHO_SUGARS_2015_PAGE": ("https://www.who.int/publications/i/item/9789241549028", "html"),
}

# Documents that the answers cite but that are blocked for automated download (HTTP 403/503 bot protection) or manual full PDFs.
UNOBTAINABLE = {
    "CODEX_CXS_192_1995": "fao.org returns 403 for automated clients",
    "CODEX_CXG_36_1989": "fao.org returns 403 for automated clients",
    "CODEX_CXS_1_1985": "fao.org returns 503 for automated clients",
    "EFSA_2008_660_2017_4786_2019_5674_2021_6585": "efsa.onlinelibrary.wiley.com returns 403 for scripts; download by hand",
    "JEMRA_PART3": "WHO/FAO Part 3 full meeting report (104-page PDF) downloaded manually to JEMRA_PART3.pdf",
}


def fetch(source_id: str, url: str, ext: str) -> dict:
    target = OUT / f"{source_id}.{ext}"
    cmd = ["curl", "-sL", "-m", "180", "-o", str(target), "-w", "%{http_code}|%{content_type}", url]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    code, _, ctype = (proc.stdout or "").partition("|")
    data = target.read_bytes() if target.exists() else b""
    ok = code == "200" and len(data) > 2000
    if ext == "pdf" and not data.startswith(b"%PDF"):
        ok = False
    return {"source_id": source_id, "url": url, "http_status": code, "content_type": ctype, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest() if data else "", "ok": ok,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "file": target.name if ok else ""}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for sid, (url, ext) in SOURCES.items():
        r = fetch(sid, url, ext)
        results.append(r)
        print(f"{'OK ' if r['ok'] else 'FAIL'} {sid}: HTTP {r['http_status']} {r['bytes']} bytes", flush=True)
        if not r["ok"] and (OUT / f"{sid}.{ext}").exists():
            (OUT / f"{sid}.{ext}").unlink()
    old = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    MANIFEST.write_text(json.dumps({"downloaded": results, "unobtainable": UNOBTAINABLE,
                                    "manually_downloaded": old.get("manually_downloaded", [])},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
