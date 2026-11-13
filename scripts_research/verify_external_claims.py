"""Check, against the downloaded primary documents, every regulatory claim that v5 answers are allowed to make.

Run scripts_research/fetch_external_sources.py first. Output: knowledge_base/sources/derived/external_claim_verification.json
Each claim records the source file, its sha256 and the evidence snippet that was matched, so the check is
reproducible offline against the exact bytes that were read.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "knowledge_base" / "sources" / "raw" / "external"
TT24_PDF = ROOT / "knowledge_base" / "sources" / "raw" / "vn" / "TT24_2019_TT-BYT_phu_luc_full.pdf"
OUT = ROOT / "knowledge_base" / "sources" / "derived" / "external_claim_verification.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def html_text(name: str) -> str:
    raw = (EXT / name).read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"(?is)<(script|style).*?</\1>", "", raw)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))


def pdf_text(path: Path) -> str:
    return re.sub(r"\s+", " ", "\n".join(p.get_text() for p in fitz.open(path)))


def find(text: str, pattern: str, width: int = 220) -> str | None:
    m = re.search(pattern, text, re.I)
    return text[max(0, m.start() - 40): m.start() + width].strip() if m else None


CODE_RE = r"\d{3,4}[a-z]?(?:\s?\([ivx]+\))*"


def squash(c: str) -> str:
    return re.sub(r"\s+", "", c.lower())


def raw_pdf_text(path: Path) -> str:
    return "\n".join(p.get_text() for p in fitz.open(path))



def fda_allergen_qa_claims(claims: dict) -> None:
    """FDA Q&A Regarding Food Allergens (Edition 5, Jan 2025), section D.9 on oils from major food allergens."""
    f = EXT / "FDA_ALLERGEN_QA_ED5.pdf"
    if not f.exists():
        return
    t = pdf_text(f)
    hits = {
        "exempt": find(t, r"Highly refined oils, which are intended to signify refined, bleached, deodorized oils, that are derived from major food allergens are exempt from section 403\(w\)", 400),
        "not_refined": find(t, r"If an oil that is derived from a major food allergen is not highly refined, then the source of the oil must be declared", 300),
        "edition": find(t, r"\(Edition 5\): Guidance for Industry"),
    }
    claims["FDA_QA5_REFINED_OIL_EXEMPT"] = {
        "description": "FDA Q&A on food allergens (Edition 5, 2025) D.9: highly refined oils derived from major food allergens are "
                       "exempt from section 403(w); an oil that is not highly refined must declare its source",
        "source_file": f.name, "sha256": sha(f), "passed": all(hits.values()), "evidence": hits}


def codex_gl23_claim(claims: dict) -> None:
    """CAC/GL 23-1997 (Guidelines for use of nutrition and health claims), Table: conditions for sodium claims."""
    f = EXT / "cxg_023e.pdf"
    if not f.exists():
        return
    t = pdf_text(f)
    hits = {
        "table": find(t, r"Sodium Low 0\.12 g per 100 g Very Low 0\.04 g per 100 g Free 0\.005 g per 100 g"),
        "title": find(t, r"Nutrition and Health Claims \(CAC/GL 23-1997\)"),
    }
    claims["CODEX_GL23_SODIUM_PER_100G"] = {
        "description": "Codex CAC/GL 23-1997 Table (nutrient content claims): sodium low = 0.12 g per 100 g, very low = "
                       "0.04 g per 100 g, free = 0.005 g per 100 g",
        "source_file": f.name, "sha256": sha(f), "passed": all(hits.values()), "evidence": hits}
    sugar = {"table": find(t, r"Sugars Free 0\.5 g per 100 g \(solids\) 0\.5 g per 100 ml \(liquids\)"),
             "title": hits["title"]}
    claims["CODEX_GL23_SUGARS_FREE_PER_100G"] = {
        "description": "Codex CAC/GL 23-1997 Table (nutrient content claims): sugars free = 0.5 g per 100 g (solids) / "
                       "0.5 g per 100 ml (liquids)",
        "source_file": f.name, "sha256": sha(f), "passed": all(sugar.values()), "evidence": sugar}


def codex_claims(claims: dict) -> None:
    """Claims from the manually downloaded Codex PDFs (CXS 192-1995, CXG 36-1989, CXS 1-1985; 2026 revisions)."""
    f192, f36, f1 = EXT / "cxs_192e.pdf", EXT / "cxg_036e.pdf", EXT / "cxs_001e.pdf"
    if not (f192.exists() and f36.exists() and f1.exists()):
        return
    # ---- CXS 192-1995: Table One (provisions by additive) and Table Three (GMP, food in general)
    t = raw_pdf_text(f192)
    t1s = t.find("Table One \n4-HEXYLRESORCINOL")
    t2s = next((m.start() for m in re.finditer(r"Table Two", t) if m.start() > t1s), -1)
    m3 = re.search(r"TABLE THREE\s*\nAdditives Permitted", t)
    t3s = m3.start() if m3 else -1
    seg3 = t[t3s:]
    e3 = seg3.find("Annex to Table 3")
    seg3 = seg3[: e3 if e3 > 0 else len(seg3)]
    t1 = sorted({squash(x) for x in re.findall(r"\n(" + CODE_RE + r")\s*\nINS\s*\n", t[t1s:t2s])})
    t3 = sorted({squash(x) for x in re.findall(r"\n(" + CODE_RE + r")\s*\n(?:19|20)\d\d\s*\n", seg3)})
    rev = re.search(r"Revision ([\d, ]+)", t[:600])
    sane = len(t1) > 250 and len(t3) > 150 and "300" in t3 and "300" in t1 and "211" in t1 and "211" not in t3
    claims["CODEX192_TABLE1_TABLE3_PARSED"] = {
        "description": "CXS 192-1995 (2026 revision): INS codes in Table One (provisions by food category) and in "
                       "Table Three (permitted in food in general under GMP)",
        "source_file": f192.name, "sha256": sha(f192), "passed": bool(sane and t1s > 0 and t3s > 0),
        "n_table_one": len(t1), "n_table_three": len(t3),
        "evidence": {"revisions": rev.group(1)[-30:] if rev else None, "300_in_table3": "300" in t3,
                     "211_only_table1": "211" in t1 and "211" not in t3}}
    (ROOT / "knowledge_base" / "sources" / "derived" / "codex_192_ins_codes.json").write_text(
        json.dumps({"source": f192.name, "sha256": sha(f192), "table_one": t1, "table_three": t3},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    # ---- CXG 36-1989: INS numbers with class names
    g = raw_pdf_text(f36)
    gm = re.search(r"\n300\s*\nAscorbic acid, L-\s*\n", g)
    claims["CODEX_CXG36_INS_AND_CLASS_NAMES"] = {
        "description": "CXG 36-1989 assigns INS numbers and technological class names (e.g. 300 = Ascorbic acid, L-: "
                       "acidity regulator, antioxidant, flour treatment agent, sequestrant)",
        "source_file": f36.name, "sha256": sha(f36),
        "passed": bool(gm) and "INTERNATIONAL NUMBERING SYSTEM" in g.upper(),
        "evidence": {"row": "300 / Ascorbic acid, L-" if gm else None}}
    # ---- CXS 1-1985: allergen lists and PAL annex
    c = re.sub(r"\s+", " ", raw_pdf_text(f1))
    m4 = re.search(r"4\.2\.1\.4 The following foods and ingredients are known to trigger food allergy or coeliac disease "
                   r"and shall always be declared.{0,1700}?Notes:", c)
    b4 = (m4.group(0) if m4 else "").lower()
    need = ["cereals containing gluten", "wheat", "rye", "barley", "crustacea", "eggs", "fish", "peanut", "milk",
            "sesame", "almond", "cashew", "hazelnut", "pecan", "pistachio", "walnut"]
    claims["CXS1_4214_MANDATORY_ALLERGEN_LIST"] = {
        "description": "CXS 1-1985 s.4.2.1.4: cereals containing gluten (wheat, rye, barley), crustacea, eggs, fish, "
                       "peanuts, milk, sesame, six specific tree nuts. Soy and molluscs are NOT in this list",
        "source_file": f1.name, "sha256": sha(f1),
        "passed": bool(m4) and all(n in b4 for n in need) and "soy" not in b4 and "mollus" not in b4,
        "evidence": {"list": b4[:400]}}
    m5 = re.search(r"4\.2\.1\.5 In addition to the foods and ingredients listed in Section 4\.2\.1\.4.{0,1400}?Note:", c)
    b5 = (m5.group(0) if m5 else "").lower()
    claims["CXS1_4215_OPTIONAL_LIST_SOY_OATS"] = {
        "description": "CXS 1-1985 s.4.2.1.5: soybean and oats are in the list of foods whose declaration 'may also be "
                       "required' based on risk assessment data for the respective population",
        "source_file": f1.name, "sha256": sha(f1),
        "passed": bool(m5) and "soybean" in b5 and "oats" in b5 and "may also be required" in b5,
        "evidence": {"list": b5[:300]}}
    pal = re.search(r"“Precautionary allergen labelling” is a statement made in the labelling of pre-packaged "
                    r"foods to indicate a risk from the unintended presence of a food allergen\(s\) due to cross-contact "
                    r"with an allergenic food\(s\) that has been identified by a risk assessment", c)
    restricted = re.search(r"The use of PAL shall be restricted to those situations in which the unintended presence of "
                           r"a food allergen\(s\) cannot be prevented or controlled", c)
    claims["CXS1_PAL_ANNEX_DEFINITION"] = {
        "description": "CXS 1-1985 Annex (2026): PAL definition (risk of unintended presence due to cross-contact, "
                       "identified by a risk assessment) and use restricted to what cannot be prevented or controlled",
        "source_file": f1.name, "sha256": sha(f1), "passed": bool(pal and restricted),
        "evidence": {"definition": bool(pal), "restricted": bool(restricted)}}


def new_authority_claims(claims: dict) -> None:
    """Three sources added to cover non-SAFE additive answers: EU Reg 1333/2008 Annex V, the FDA Red No. 3 order
    (FR 2025-00830) and IARC Monographs Vol. 134 (aspartame). Each claim is a regex over the downloaded text."""
    def text_of(name: str) -> tuple[Path, str]:
        p = EXT / name
        return p, re.sub(r"\s+", " ", pdf_text(p).replace("­", "")) if p.exists() else ""

    def add(cid, name, checks, desc):
        p, t = text_of(name)
        if not t:
            return
        hits = {k: find(t, pat, 500) for k, pat in checks.items()}
        claims[cid] = {"description": desc, "source_file": p.name, "sha256": sha(p),
                       "passed": all(hits.values()), "evidence": hits}

    add("EU_ANNEXV_COLOURS_WARNING", "EU_REG_1333_2008_ANNEX_V.pdf",
        {"list": r"ANNEX V List of the food colours referred to in Article 24.{0,200}Sunset yellow \(E 110\).{0,300}"
                 r"Quinoline yellow \(E 104\).{0,40}Carmoisine \(E 122\).{0,40}Allura red \(E 129\).{0,40}"
                 r"Tartrazine \(E 102\).{0,40}Ponceau 4R \(E 124\)",
         "wording": r"may have an adverse effect on activity and attention in children",
         "exception_a": r"health or other marking on meat products or for stamping or decorative colouring on eggshells",
         "exception_b": r"beverages containing more than 1,2 % by volume of alcohol"},
        "EU Reg 1333/2008 Annex V: foods containing E110, E104, E122, E129, E102 or E124 must carry the warning "
        "'may have an adverse effect on activity and attention in children'; exceptions for meat/egg marking and >1.2% alcohol drinks")
    add("FDA_RED3_REVOKED_FOOD", "FR_RED3_REVOCATION.pdf",
        {"order": r"granting a color additive petition .{0,200}repealing the color additive regulations that permit the "
                  r"use of FD&C Red No\. 3 in foods",
         "reason": r"induces cancer in male rats",
         "date": r"effective January 15, 2027",
         "delaney": r"Delaney Clause",
         "no_risk_weighing": r"regardless of the probability, or risk, of cancer"},
        "FDA final order (FR 2025-00830, 16 Jan 2025): authorisation of FD&C Red No. 3 in food repealed because it "
        "induces cancer in male rats (Delaney Clause); effective 15 Jan 2027")
    add("CXS1_4217_SULPHITE_DECLARATION", "cxs_001e.pdf",
        {"rule": r"4\.2\.1\.7 Sulphite when present in concentrations of 10 mg/kg or more in a food.{0,12}shall always be "
                 r"declared using the specified name .{0,3}sulphite.{0,3} or .{0,3}sulfite.{0,3}",
         "so2_basis": r"Sulphite measured on a sulphur dioxide \(SO2\) equivalents basis"},
        "Codex CXS 1-1985 s.4.2.1.7: sulphite at 10 mg/kg or more (as SO2) shall always be declared using the specified "
        "name 'sulphite' or 'sulfite'")
    add("IARC_V134_ASPARTAME_2B", "IARC_V134_ASPARTAME.pdf",
        {"overall": r"6\.4 Overall evaluation Aspartame is possibly carcinogenic to humans \(Group 2B\)",
         "humans": r"There is limited evidence in humans for the carcinogenicity of aspartame",
         "hazard_not_risk": r"not to the risk of cancer from particular exposures"},
        "IARC Monographs Vol. 134 (2023): aspartame is possibly carcinogenic to humans (Group 2B); limited evidence; "
        "the classification describes the strength of evidence, not the risk of cancer from particular exposures")
    add("WHO_NSS_2023_PREGNANCY", "Bookshelf_NBK592258.pdf",
        {"recommendation": r"WHO suggests that non-sugar sweeteners not be used as a means of achieving weight control or "
                           r"reducing the risk of noncommunicable diseases \(conditional recommendation\)",
         "applies_to_pregnant": r"this recommendation is relevant for everyone: children and adults of any age, including "
                                r"pregnant and lactating women",
         "diabetes_excluded": r"the recommendation does not apply to individuals with existing diabetes",
         "preterm": r"Higher intakes of NSS were associated with a 25% increase in risk of preterm birth, as assessed by "
                    r"meta-analyses of three prospective cohort studies \(odds ratio \[OR\] 1\.25; 95% CI: 1\.07 to 1\.46; "
                    r"129 009 pregnant women\)",
         "preterm_certainty": r"increased risk of preterm birth with higher NSS use during pregnancy \(low certainty evidence\)",
         "gdm": r"No associations were observed between NSS use and risk of gestational diabetes",
         "reverse_causation": r"reverse causation and confounding by body weight or other residual confounding "
                              r"contributes significantly to the associations observed in prospective cohort studies for "
                              r"adults and pregnant women",
         "definition": r"Common NSS include acesulfame K, aspartame, advantame, cyclamates, neotame, saccharin, sucralose, "
                       r"stevia and stevia derivatives",
         "polyols_excluded": r"low-calorie sugars and sugar alcohols \(polyols\) are sugars or sugar derivatives containing "
                             r"calories, they are not considered NSS"},
        "WHO guideline on non-sugar sweeteners (2023): conditional recommendation against using NSS for weight control or "
        "NCD risk reduction, relevant for everyone including pregnant women (not people with diabetes); limited evidence "
        "that higher NSS use in pregnancy is associated with preterm birth (OR 1.25, low certainty); no association with "
        "gestational diabetes; polyols are not NSS")
    add("WHO_NSS_2023_CHILDREN", "Bookshelf_NBK592258.pdf",
        {"recommendation": r"WHO suggests that non-sugar sweeteners not be used as a means of achieving weight control or "
                           r"reducing the risk of noncommunicable diseases \(conditional recommendation\)",
         "applies_to_children": r"this recommendation is relevant for everyone: children and adults of any age, including "
                                r"pregnant and lactating women",
         "diabetes_excluded": r"the recommendation does not apply to individuals with existing diabetes",
         "child_rct": r"One RCT conducted in children reported a reduction in several measures of body fatness when "
                      r"sugar-\s?sweetened beverages were replaced with those containing NSS \(.{0,6}\) \(moderate certainty evidence\)",
         "child_bmi": r"when results for BMI z-score.{0,30} were combined with those from a second trial .{0,6}, no effect was "
                      r"observed \(moderate certainty evidence\)",
         "child_observational": r"results from prospective observational studies did not suggest any significant "
                                r"associations between NSS use and measures of body fatness \(very low certainty evidence\)",
         "child_certainty": r"because both adult data and child data were considered for children, the overall certainty "
                            r"in the available evidence across both groups was assessed as low",
         "not_safety_limit": r"is based on evidence of health effects of NSS use at levels already considered safe by "
                             r"JECFA.{0,8}and is not intended to update or replace existing guidance on safe or maximal "
                             r"levels of intake",
         "definition": r"Common NSS include acesulfame K, aspartame, advantame, cyclamates, neotame, saccharin, sucralose, "
                       r"stevia and stevia derivatives",
         "polyols_excluded": r"low-calorie sugars and sugar alcohols \(polyols\) are sugars or sugar derivatives containing "
                             r"calories, they are not considered NSS"},
        "WHO guideline on non-sugar sweeteners (2023), children: conditional recommendation against NSS for weight control "
        "or NCD risk reduction applies to children (not people with diabetes); direct child evidence is limited (one RCT "
        "with reduced body fatness but no BMI z-score effect when pooled; observational studies no significant "
        "association; overall certainty low); the guideline does not replace existing safe-intake guidance")
    add("EFSA_CAFFEINE_2015_CHILDREN", "EFSA_Journal_2015_Scientific_Opinion_safety_caffeine.pdf",
        {"title": r"Scientific Opinion on the safety of caffeine",
         "doi": r"doi:10\.2903/j\.efsa\.2015\.4102",
         "child_insufficient": r"For children and adolescents, the information available is insufficient to derive a safe "
                               r"caffeine intake",
         "adult_basis": r"caffeine intakes of no concern derived for acute caffeine consumption by adults \(3 mg/kg bw per "
                        r"day\) may serve as a basis to derive single doses of caffeine and daily caffeine intakes of no "
                        r"concern for these population subgroups",
         "sources": r"The main sources of caffeine in the diet include coffee, tea, caffeinated soft drinks \(including "
                    r".{0,3}energy drinks.{0,3}\) and chocolate"},
        "EFSA NDA Panel (2015) opinion on the safety of caffeine: for children and adolescents the information is "
        "insufficient to derive a safe caffeine intake; the adult acute no-concern level (3 mg/kg bw) may serve as a "
        "basis for these subgroups; main dietary sources are coffee, tea, caffeinated soft drinks and chocolate")


def add_pages(claims: dict) -> None:
    """For claims backed by a PDF, record the 1-based page(s) where each evidence snippet was found.
    HTML sources have no pages. Evidence that is not a text snippet (parsed tables, booleans) is left unlocated."""
    def squash_all(t: str) -> str:
        return re.sub(r"\s+", "", t.replace("­", "")).lower()

    cache: dict[str, list[str]] = {}
    for cid, c in claims.items():
        name = c.get("source_file", "")
        if not name.lower().endswith(".pdf"):
            continue
        path = EXT / name if (EXT / name).exists() else (TT24_PDF if name == TT24_PDF.name else None)
        if path is None:
            continue
        if name not in cache:
            cache[name] = [squash_all(pg.get_text()) for pg in fitz.open(path)]
        pages = {}
        for key, snip in (c.get("evidence") or {}).items():
            if not isinstance(snip, str) or len(snip) < 60:
                continue
            body = snip[40:]                                 # find() prepends 40 characters of left context
            for probe in (squash_all(body)[:60], squash_all(body)[:30]):
                hits = [i + 1 for i, t in enumerate(cache[name]) if probe and probe in t]
                if hits:
                    pages[key] = hits[0] if len(hits) == 1 else hits
                    break
        c["evidence_pages"] = pages
        c["pages_located"] = bool(pages)


def build() -> dict:
    claims: dict = {}

    def add(cid, source, text, checks: dict[str, str], desc):
        hits = {k: find(text, p) for k, p in checks.items()}
        claims[cid] = {"description": desc, "source_file": source.name, "sha256": sha(source),
                       "passed": all(hits.values()), "evidence": hits}

    # 21 U.S.C. 321(qq): the nine "major food allergens"
    p = EXT / "US_CODE_21_321.html"
    t = html_text(p.name)
    m = re.search(r"\(qq\) The term .{0,3}major food allergen.{0,3} means any of the following: \(1\) (.{0,260})", t)
    body = (m.group(1) if m else "").lower()
    needed = ["milk", "egg", "fish", "crustacean shellfish", "tree nuts", "wheat", "peanuts", "soybeans", "sesame"]
    claims["FALCPA_321qq_NINE_ALLERGENS"] = {
        "description": "21 U.S.C. 321(qq)(1) defines major food allergen = milk, egg, fish, crustacean shellfish, "
                       "tree nuts, wheat, peanuts, soybeans, sesame (molluscs are NOT included)",
        "source_file": p.name, "sha256": sha(p),
        "passed": bool(m) and all(n in body for n in needed) and "mollus" not in body,
        "evidence": {"qq_1": body[:260]}}

    # 21 CFR 172.804(d)(2): phenylketonurics statement
    p = EXT / "US_CFR_21_172_804.html"
    add("CFR_172_804_PKU_STATEMENT", p, html_text(p.name),
        {"statement": r"PHENYLKETONURICS: CONTAINS PHENYLALANINE"},
        "21 CFR 172.804 requires 'PHENYLKETONURICS: CONTAINS PHENYLALANINE' on foods containing aspartame")

    # FAO/WHO JEMRA 2021 Part 1 summary: priority list and what it leaves out
    p = EXT / "JEMRA_2021_PART1_SUMMARY.pdf"
    t = pdf_text(p)
    m = re.search(r"should be listed as priority allergens: (.{0,330}?)\. Due to the lack of data", t)
    lst = (m.group(1) if m else "").lower()
    need = ["cereals containing gluten", "crustacea", "eggs", "fish", "milk", "peanuts", "sesame",
            "almond", "cashew", "hazelnut", "pecan", "pistachio", "walnut"]
    claims["JEMRA1_PRIORITY_LIST"] = {
        "description": "JEMRA 2021 Part 1 priority list = cereals containing gluten, crustacea, eggs, fish, milk, "
                       "peanuts, sesame, specific tree nuts (almond, cashew, hazelnut, pecan, pistachio, walnut)",
        "source_file": p.name, "sha256": sha(p),
        "passed": bool(m) and all(n in lst for n in need) and "soy" not in lst and "mollus" not in lst,
        "evidence": {"list": lst}}
    add("JEMRA1_SOY_AND_OATS_NOT_GLOBAL", p, t,
        {"not_global": r"buckwheat, celery, lupin, mustard, oats, soybean and tree nuts .{0,40}should not be listed as global priority allergens"},
        "JEMRA 2021 Part 1: soybean and oats are NOT global priority allergens (may be listed nationally)")

    # FAO/WHO JEMRA Part 3 (PAL report - full PDF meeting report, fallback to WHO webpage)
    pdf_p = EXT / "JEMRA_PART3.pdf"
    if pdf_p.exists():
        p = pdf_p
        t = pdf_text(p)
        finding_snip = find(t, r"current PAL systems used in many countries needed to be improved as they were neither uniform nor informative")
        claims["JEMRA3_PAL_REPORT"] = {
            "description": "FAO/WHO Part 3 is the PAL report; it found current PAL systems neither uniform nor informative",
            "source_file": p.name,
            "sha256": sha(p),
            "passed": bool(finding_snip) and "precautionary" in t.lower(),
            "evidence": {"finding": finding_snip},
        }
    else:
        p = EXT / "JEMRA_PART3_WHO_PAGE.html"
        add("JEMRA3_PAL_REPORT", p, html_text(p.name),
            {"title": r"part 3: review and establish precautionary labelling in foods of the priority allergens",
             "finding": r"current PAL systems used in many countries needed to be improved as they were neither uniform nor informative"},
            "FAO/WHO Part 3 is the PAL report; it found current PAL systems neither uniform nor informative")

    # EFSA 2021 titanium dioxide
    p = EXT / "EFSA_2021_TIO2_NEWS.html"
    add("EFSA_2021_TIO2_NOT_SAFE", p, html_text(p.name),
        {"cannot_rule_out": r"a concern for genotoxicity of TiO ?2 particles cannot be ruled out",
         "not_safe": r"no longer consider titanium dioxide safe when used as a food additive"},
        "EFSA 2021: genotoxicity concern for TiO2 cannot be ruled out; E171 no longer considered safe as an additive")

    # ---- FDA nutrient content claims and benzene (explain why the nutrient-threshold and benzene records cannot be scored)
    p = EXT / "US_CFR_21_101_61.html"
    t = html_text(p.name)
    add("CFR_101_61_SODIUM_PER_RACC", p, t,
        {"very_low": r"35 mg or less sodium per reference amount customarily consumed",
         "free": r"less than 5 milligrams \(mg\) of sodium per reference amount customarily consumed",
         "low": r"140 mg or less sodium per reference amount customarily consumed"},
        "21 CFR 101.61: sodium claim thresholds (free <5 mg, very low <=35 mg, low <=140 mg) are per reference amount "
        "customarily consumed (RACC), not per 100 g or per label serving")
    p = EXT / "US_CFR_21_101_54.html"
    add("CFR_101_54_HIGH_20PCT_DV", p, html_text(p.name),
        {"high": r"contains 20 percent or more of the (RDI|Reference Daily Intake|DRV|Daily Reference Value)"},
        "21 CFR 101.54: 'high' = 20 percent or more of the RDI/DRV per reference amount customarily consumed")
    p = EXT / "FDA_BENZENE_QA.html"
    add("FDA_BENZENE_BEVERAGES_ONLY", p, html_text(p.name),
        {"beverages": r"Benzene can form at very low levels \(ppb level\) in some beverages that contain both benzoate salts and ascorbic acid"},
        "FDA Q&A: benzene can form at ppb levels in some BEVERAGES containing benzoate + ascorbic acid; "
        "the page says nothing about bakery products")

    # ---- EFSA full-text opinions (downloaded by hand; Wiley blocks scripts)
    def efsa_text(name: str) -> str:
        raw = pdf_text(EXT / name)
        for a, b in (("ﬁ", "fi"), ("ﬂ", "fl"), ("–", "-"), ("‑", "-"), ("’", "'")):
            raw = raw.replace(a, b)
        return re.sub(r"\s+", " ", raw)

    p = EXT / "efsa_2019_5674.pdf"
    if p.exists():
        add("EFSA_OPINION_2019_PHOSPHATES", p, efsa_text(p.name),
            {"title": r"Re-evaluation of phosphoric acid-phosphates - di-, tri- and polyphosphates \(E 338-341, E 343, E 450-452\)",
             "adi": r"group ADI of 40 mg/kg bw per day, expressed as phosphorus, is protective for healthy adults",
             "ckd": r"this ADI does not apply to humans with moderate to severe reduction in renal function"},
            "EFSA 2019 (5674): group ADI 40 mg/kg bw/day as phosphorus protects healthy adults but does NOT apply to "
            "humans with moderate to severe reduction in renal function; group = E 338-341, E 343, E 450-452")
    p = EXT / "efsa_2008_660.pdf"
    if p.exists():
        add("EFSA_OPINION_2008_MCCANN", p, efsa_text(p.name),
            {"compounds": r"Tartrazine \(E102\), Quinoline Yellow \(E104\), Sunset Yellow FCF \(E110\), Ponceau 4R \(E124\), Allura Red AC \(E129\), Carmoisine \(E122\) and sodium benzoate \(E211\)",
             "ages": r"three 3-year old|one hundred and fifty three 3-year old and one hundred and forty four 8- to 9-year old children",
             "limited": r"provides limited evidence that the two different mixtures of synthetic colours and sodium benzoate tested had a small and statistically significant effect on activity and attention in children",
             "individual": r"it is not possible to ascribe the observed effects to any of the individual compounds",
             "adi": r"cannot be used as a basis for altering the ADI of the respective food colours or sodium benzoate"},
            "EFSA 2008 (660): McCann study tested mixtures of E102, E104, E110, E122, E124, E129 and sodium benzoate E211 "
            "in 3-year-old and 8-9-year-old children; limited evidence of a small effect on activity and attention, "
            "not attributable to individual compounds, not a basis for altering ADIs")
    p = EXT / "efsa_2017_4786.pdf"
    if p.exists():
        add("EFSA_OPINION_2017_NITRITE", p, efsa_text(p.name),
            {"adi": r"derived an ADI of 0\.07 mg nitrite ion/kg bw per day",
             "exposure": r"did not exceed this ADI for the general population, except for a slight exceedance in children at the highest percentile",
             "nordic": r"The Nordic Council identified these groups as including: pregnant women"},
            "EFSA 2017 (4786): ADI 0.07 mg nitrite ion/kg bw/day (methaemoglobin); slight exceedance in children at the "
            "highest percentile; quotes the Nordic Council listing pregnant women among more sensitive groups")
    p = EXT / "efsa_2021_6585.pdf"
    if p.exists():
        add("EFSA_2021_TIO2_FULLTEXT", p, efsa_text(p.name),
            {"conclusion": r"a concern for genotoxicity could not be ruled out, and given the many uncertainties, the Panel concluded that E 171 can no longer be considered as safe when used as a food additive"},
            "EFSA 2021 (6585) full text: genotoxicity concern could not be ruled out; E 171 can no longer be considered "
            "safe when used as a food additive")

    # TT 24/2019 full text (local PDF)
    t = pdf_text(TT24_PDF)
    add("TT24_PHULUC1_LISTS_INS_AND_NAME", TT24_PDF, t,
        {"title": r"DANH MỤC PHỤ GIA THỰC PHẨM ĐƯỢC PHÉP SỬ DỤNG TRONG THỰC PHẨM",
         "columns": r"STT INS Tên phụ gia Chức năng"},
        "TT24 Phụ lục 1 lists permitted additives by INS code and name")
    add("TT24_ML_IN_MG_PER_KG", TT24_PDF, t, {"ml_header": r"ML \(mg/ ?kg\)"},
        "TT24 Phụ lục 2 expresses maximum levels (ML) in mg/kg of food")
    for jecfa_path in (ROOT / "knowledge_base" / "sources" / "derived" / "jecfa_adi_verification.json", ROOT / "knowledge_base" / "sources" / "derived" / "jecfa_adi_verification_more.json"):
      if jecfa_path.exists():
        for code, r in json.loads(jecfa_path.read_text(encoding="utf-8")).items():
            if not (r.get("passed") and r.get("latest_adi")):
                continue
            f = EXT / "jecfa" / r["file"]
            ok = f.exists() and sha(f) == r["sha256"] and code.replace("(", "").replace(")", "") in                 [x.replace("(", "").replace(")", "") for x in r["ins_on_page"]]
            claims[f"JECFA_ADI_{code}"] = {
                "description": f"JECFA database: INS {code} = {r['jecfa_name']}, latest evaluation "
                               f"{r['latest_evaluation_year']}, ADI {r['latest_adi']}",
                "source_file": "jecfa/" + r["file"], "sha256": r["sha256"], "passed": bool(ok),
                "jecfa_name": r["jecfa_name"], "chemical_id": r["chemical_id"],
                "adi": r["latest_adi"], "year": r["latest_evaluation_year"],
                "evidence": {"ins_on_page": r["ins_on_page"], "adi": r["latest_adi"]}}
    codex_claims(claims)
    codex_gl23_claim(claims)
    fda_allergen_qa_claims(claims)
    new_authority_claims(claims)
    add_pages(claims)
    return claims


def main():
    claims = build()
    OUT.write_text(json.dumps(claims, ensure_ascii=False, indent=2), encoding="utf-8")
    for cid, c in claims.items():
        print(("PASS " if c["passed"] else "FAIL ") + cid)
    raise SystemExit(0 if all(c["passed"] for c in claims.values()) else 1)


if __name__ == "__main__":
    main()
