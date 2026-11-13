"""Fetch JECFA database records (WHO) for the additives whose ADI the benchmark mentions, and verify them.

Uses the public endpoints that the JECFA web site itself calls (api/SearchChemical/ByPartialName and
Home/Chemical/<id>). A candidate chemical is accepted only if the INS number printed on its JECFA page equals the
INS code being checked, so the mapping INS -> JECFA record is verified and not guessed from the name.

Output: knowledge_base/sources/raw/external/jecfa/  (raw API JSON + chemical pages)  and knowledge_base/sources/derived/jecfa_adi_verification.json
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "knowledge_base" / "sources" / "raw" / "external" / "jecfa"
OUT_JSON = ROOT / "knowledge_base" / "sources" / "derived" / "jecfa_adi_verification.json"
BASE = "https://apps.who.int/food-additives-contaminants-jecfa-database"

# INS code (lower case, as in TT24) -> search keywords for the JECFA name search
CODES = {
    "300": ["ascorbic acid"], "322(i)": ["lecithin"], "330": ["citric acid"], "401": ["sodium alginate", "alginate"],
    "407": ["carrageenan"], "440": ["pectin"], "471": ["mono- and diglycerides", "diglycerides", "di-glycerides", "mono- and di-glycerides", "glycerides", "monoglycerides", "glycerol", "fatty acid"],
    "500(ii)": ["sodium hydrogen carbonate", "bicarbonate"], "551": ["silicon dioxide"], "627": ["guanylate"],
    "1442": ["hydroxypropyl distarch phosphate"], "631": ["inosinate"], "412": ["guar gum", "guar"], "415": ["xanthan"], "450(i)": ["diphosphate"],
}


def curl(url: str) -> bytes:
    p = subprocess.run(["curl", "-sL", "-m", "90", url], capture_output=True)
    return p.stdout


def page_text(raw: bytes) -> str:
    t = raw.decode("utf-8", errors="ignore")
    t = re.sub(r"(?is)<(script|style).*?</\1>", "", t)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))


def norm(code: str) -> str:
    """JECFA prints 500ii where TT24 prints 500(ii): compare without spaces and parentheses."""
    return re.sub(r"[\s()]+", "", code.lower())


def parse_page(text: str) -> dict:
    ins = re.search(r"\bINS (.*?) Functional Class", text)
    ins_codes = [norm(x) for x in re.split(r"[,;/]| and ", ins.group(1))] if ins else []
    ev = re.search(r"Evaluations Evaluation year: (\d{4}) ADI: (.*?) (?:Comments:|Meeting:)", text)
    return {"ins_on_page": ins_codes, "latest_evaluation_year": ev.group(1) if ev else "",
            "latest_adi": ev.group(2).strip() if ev else ""}


def register_in_catalog(result: dict) -> None:
    cat_path = ROOT / "knowledge_base" / "sources" / "catalog" / "source_citation_catalog.json"
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    for code, r in result.items():
        if r.get("passed") and r.get("latest_adi"):
            key = f"WHO_FAO_JECFA#Chemical{r['chemical_id']}_" + re.sub(r"[^A-Za-z0-9]+", "", r["jecfa_name"].title())
            cat[key] = {"citation_id": "JECFA_EVALUATIONS",
                        "title": f"WHO JECFA database - {r['jecfa_name']} (INS {code})",
                        "url": r["url"],
                        "scope": f"Latest evaluation {r['latest_evaluation_year']}: ADI {r['latest_adi']}.",
                        "verified_against": "knowledge_base/sources/raw/external/jecfa/" + r["file"]}
    cat_path.write_text(json.dumps(cat, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {}
    for code, keywords in CODES.items():
        found = None
        tried = []
        for kw in keywords:
            raw = curl(f"{BASE}/api/SearchChemical/ByPartialName/{kw.replace(' ', '%20')}")
            (OUT_DIR / f"search_{re.sub(r'[^a-z0-9]+', '_', kw)}.json").write_bytes(raw)
            try:
                hits = [h for h in json.loads(raw) if h.get("Id")]
            except Exception:
                hits = []
            for h in hits[:60]:
                if h["Id"] in tried:
                    continue
                tried.append(h["Id"])
                page = curl(f"{BASE}/Home/Chemical/{h['Id']}")
                info = parse_page(page_text(page))
                if norm(code) in [norm(x) for x in info["ins_on_page"]]:
                    path = OUT_DIR / f"Chemical_{h['Id']}.html"
                    path.write_bytes(page)
                    found = {"chemical_id": h["Id"], "jecfa_name": h["Name"], "url": f"{BASE}/Home/Chemical/{h['Id']}",
                             "file": path.name, "sha256": hashlib.sha256(page).hexdigest(), **info,
                             "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                    break
            if found:
                break
        if found:
            found["passed"] = True
            found["adi_not_specified"] = found["latest_adi"].upper() == "NOT SPECIFIED"
        else:
            found = {"passed": False, "adi_not_specified": False, "candidates_tried": tried}
        result[code] = found
        print(("OK  " if found["passed"] else "MISS"), code, found.get("jecfa_name", ""), found.get("chemical_id", ""),
              "|", found.get("latest_evaluation_year", ""), found.get("latest_adi", ""), flush=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    register_in_catalog(result)


if __name__ == "__main__":
    main()
