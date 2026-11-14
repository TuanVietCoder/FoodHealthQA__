"""Extend the JECFA source set to every INS code that the KB-dependent ABSTAIN records mention.

Same acceptance rule as fetch_jecfa.py: a JECFA record is accepted only when the INS number printed on its page equals
the code being checked. Output goes to knowledge_base/sources/derived/jecfa_adi_verification_more.json (the original file is left untouched) and the
fetched pages to knowledge_base/sources/raw/external/jecfa/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_jecfa as fj  # noqa: E402

ROOT = fj.ROOT
OUT_MORE = ROOT / "knowledge_base" / "sources" / "derived" / "jecfa_adi_verification_more.json"
MAX_HITS = 40

# English JECFA-style names for codes whose TT24 name could not be searched directly (base code or full code, normalised)
EXTRA = {
    "101": ["riboflavin"], "120": ["carmines", "cochineal"], "141": ["chlorophyllin", "chlorophyll"], "142": ["green s"],
    "150a": ["caramel colour class i", "caramel colour"], "150c": ["caramel colour class iii", "caramel"],
    "150d": ["caramel colour class iv", "caramel"], "153": ["vegetable carbon"],
    "160a": ["beta-carotene", "carotenes"], "160b": ["annatto extracts", "annatto"], "160c": ["paprika oleoresin", "paprika"],
    "160d": ["lycopene"], "222": ["sodium hydrogen sulfite", "sulfite"], "260": ["acetic acid"], "270": ["lactic acid"],
    "296": ["malic acid"], "307": ["tocopherol"], "319": ["tertiary butylhydroquinone", "tbhq"], "322": ["lecithin"],
    "327": ["calcium lactate"], "331": ["sodium citrate", "citrate"], "332": ["potassium citrate", "citrate"],
    "334": ["tartaric acid"], "337": ["potassium sodium tartrate", "tartrate"], "339": ["sodium phosphate", "phosphate"],
    "340": ["potassium phosphate", "phosphate"], "341": ["calcium phosphate", "phosphate"], "364": ["sodium sulfite", "sulfite"],
    "385": ["calcium disodium ethylene diamine tetra acetate", "edta"], "405": ["propylene glycol alginate"],
    "406": ["agar"], "407a": ["processed eucheuma seaweed"], "414": ["acacia gum", "gum arabic"], "415": ["xanthan gum"],
    "418": ["gellan gum"], "422": ["glycerol"], "425": ["konjac"], "450": ["diphosphate", "phosphate"],
    "452": ["polyphosphate", "phosphate"], "460": ["cellulose"], "470": ["salts of fatty acids", "fatty acid"],
    "472a": ["acetic and fatty acid esters of glycerol"], "472e": ["diacetyltartaric and fatty acid esters of glycerol"],
    "473": ["sucrose esters of fatty acids"], "475": ["polyglycerol esters of fatty acids"],
    "477": ["propylene glycol esters of fatty acids"], "481": ["sodium stearoyl lactylate", "stearoyl lactylate"],
    "500": ["sodium carbonate", "carbonate"], "503": ["ammonium carbonate", "carbonate"], "514": ["sodium sulfate", "sulfate"],
    "575": ["glucono delta-lactone", "gluconolactone"], "620": ["glutamic acid"], "621": ["monosodium glutamate", "glutamate"],
    "630": ["inosinic acid"], "635": ["disodium 5'-ribonucleotides", "ribonucleotides"], "640": ["glycine"],
    "900": ["dimethylpolysiloxane", "polydimethylsiloxane"], "901": ["beeswax"], "904": ["shellac"], "941": ["nitrogen"],
    "953": ["isomalt"], "955": ["sucralose"], "960": ["steviol glycosides"], "964": ["polyglycitol syrup"],
    "965": ["maltitol"], "968": ["erythritol"], "1100": ["amylase"], "1101": ["protease"], "1104": ["lipase"],
    "1400": ["dextrin roasted starch", "dextrins"], "1401": ["acid-treated starch"], "1404": ["oxidized starch"],
    "1412": ["distarch phosphate"], "1414": ["acetylated distarch phosphate"], "1420": ["starch acetate"],
    "1422": ["acetylated distarch adipate"], "1440": ["hydroxypropyl starch"], "1442": ["hydroxypropyl distarch phosphate"],
    "1450": ["starch sodium octenyl succinate"], "1520": ["propylene glycol"],
}


def parse_page2(text: str) -> dict:
    """More tolerant than fetch_jecfa.parse_page: INS list from 'INS matches ...' (falls back to the INS field) and the
    most recent evaluation that actually carries an ADI (an earlier entry may only say 'considered for specification')."""
    m = re.search(r"INS matches (.*?) Evaluations", text)
    if m:
        raw = m.group(1)
    else:
        m = re.search(r"INS (.*?) (?:COE number|FEMA number|JECFA number|CAS number|Functional Class)", text)
        raw = m.group(1) if m else ""
    ins_codes = [fj.norm(x) for x in re.split(r"[,;/]| and ", raw) if x.strip()]
    evals = re.findall(r"Evaluation year: (\d{4}) (?:Comments: .*?)?(?:ADI: (.*?) )?(?=Comments:|Meeting:)", text)
    best = ("", "")
    for year, adi in re.findall(r"Evaluation year: (\d{4}) ADI: (.*?) (?:Comments:|Meeting:)", text):
        if year >= best[0]:
            best = (year, adi.strip())
    return {"ins_on_page": ins_codes, "latest_evaluation_year": best[0], "latest_adi": best[1]}


def wanted_codes() -> list[str]:
    _bench_csv = (
        ROOT / "data" / "archive" / "master_benchmark_2311" / "qa_benchmark.csv"
        if (ROOT / "data" / "archive" / "master_benchmark_2311" / "qa_benchmark.csv").exists()
        else ROOT / "data" / "qa_benchmark.csv"
    )
    rows = list(csv.DictReader(open(_bench_csv, encoding="utf-8-sig")))
    codes = set()
    for r in rows:
        if r["eligible_for_scoring"] == "no" and r["domain"] != "vulnerable_groups":
            for m in re.findall(r"INS\s?(\d{3,4}[A-Za-z]?(?:\([IVXivx]+\))?)", r["question_v5"]):
                codes.add(fj.norm(m))
    done = set(json.loads(fj.OUT_JSON.read_text(encoding="utf-8")))
    done = {fj.norm(c) for c in done}
    return sorted(c for c in codes if c not in done)


def keywords(code: str, names: dict) -> list[str]:
    key = next((k for k in names if fj.norm(k) == code), None)
    base = re.match(r"\d+[a-z]?", code).group(0)
    if key is None:
        key = next((k for k in names if fj.norm(k).startswith(base)), None)
    kws = list(EXTRA.get(code, [])) or list(EXTRA.get(re.match(r"\d+[a-z]?", code).group(0), []))
    if key:
        words = names[key].split()
        if len(words) % 2 == 0 and words[: len(words) // 2] == words[len(words) // 2:]:
            kws.append(" ".join(words[: len(words) // 2]))
        kws.append(names[key])
        kws += [" ".join(words[len(words) // 2:])]
    return list(dict.fromkeys(k for k in kws if k))


def main():
    names = json.loads((ROOT / "knowledge_base" / "sources" / "derived" / "tt24_additive_names.json").read_text(encoding="utf-8"))
    names = names.get("codes", names)
    out = json.loads(OUT_MORE.read_text(encoding="utf-8")) if OUT_MORE.exists() else {}
    for code in wanted_codes():
        if code in out and out[code].get("passed") and out[code].get("latest_adi"):
            continue
        found, tried = None, []
        for kw in keywords(code, names):
            raw = fj.curl(f"{fj.BASE}/api/SearchChemical/ByPartialName/{kw.replace(' ', '%20')}")
            try:
                hits = [h for h in json.loads(raw) if h.get("Id")]
            except Exception:
                hits = []
            for h in hits[:MAX_HITS]:
                if h["Id"] in tried:
                    continue
                tried.append(h["Id"])
                page = fj.curl(f"{fj.BASE}/Home/Chemical/{h['Id']}")
                info = parse_page2(fj.page_text(page))
                if code in [fj.norm(x) for x in info["ins_on_page"]]:
                    path = fj.OUT_DIR / f"Chemical_{h['Id']}.html"
                    path.write_bytes(page)
                    found = {"chemical_id": h["Id"], "jecfa_name": h["Name"], "url": f"{fj.BASE}/Home/Chemical/{h['Id']}",
                             "file": path.name, "sha256": hashlib.sha256(page).hexdigest(), **info,
                             "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                    break
            if found:
                break
        if found:
            found["passed"] = True
            found["adi_not_specified"] = found["latest_adi"].upper() == "NOT SPECIFIED"
        else:
            found = {"passed": False, "adi_not_specified": False, "candidates_tried": tried, "keywords": keywords(code, names)}
        out[code] = found
        print(("OK  " if found["passed"] else "MISS"), code, found.get("jecfa_name", ""), "|",
              found.get("latest_evaluation_year", ""), found.get("latest_adi", ""), flush=True)
        OUT_MORE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
