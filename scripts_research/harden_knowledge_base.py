"""Add provenance and conservative review states to the food knowledge base."""

import csv
import json
import os
import re
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.foodon import find_foodon_exact_label, load_foodon_entities


KB_DIR = os.path.join(ROOT, "knowledge_base")

SOURCES = {
    "JECFA_DB": {
        "title": "WHO/FAO JECFA Food Additives Database",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/",
        "scope": "Use the substance-specific evaluation record and linked monograph; the database landing page is not enough to validate an individual numeric claim.",
    },
    "EFSA_GLUTAMATES_2017": {
        "title": "EFSA Journal 2017;15(7):4910, glutamates",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7009848/",
        "scope": "Group ADI 30 mg/kg bw/day, expressed as glutamic acid, for E620-E625.",
    },
    "WHO_JECFA_ASPARTAME_2023": {
        "title": "WHO/FAO/IARC 2023 aspartame assessment summary",
        "url": "https://www.who.int/news/item/14-07-2023-aspartame-hazard-and-risk-assessment-results-released",
        "scope": "JECFA reaffirmed ADI 0-40 mg/kg bw/day; IARC hazard classification is distinct from a dose-specific risk assessment.",
    },
    "FDA_ASPARTAME": {
        "title": "US FDA, High-Intensity Sweeteners",
        "url": "https://www.fda.gov/food/food-additives-petitions/high-intensity-sweeteners",
        "scope": "FDA says people with PKU should avoid or restrict aspartame and labels must identify phenylalanine.",
    },
    "EFSA_MODIFIED_STARCH_2017": {
        "title": "EFSA Journal 2017;15(10):4911, modified starches",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7009865/",
        "scope": "Includes E1420; no numerical ADI needed and no safety concern at reported uses/use levels for the general population.",
    },
    "EFSA_SULFITES_2022": {
        "title": "EFSA Journal 2022;20(11):7594, sulfur dioxide and sulfites",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9685353/",
        "scope": "Available toxicity database was inadequate to derive an ADI; EFSA withdrew its temporary group ADI and used a margin-of-exposure approach.",
    },
    "CODEX_CXS_1_1985": {
        "title": "FAO-hosted Codex CXS 1-1985 (amended through 2010 in linked copy)",
        "url": "https://www.fao.org/input/download/standards/32/CXS_001e.pdf",
        "scope": "Section 4.2.1.4 list, section 4.2.2 and section 6 small-unit exemption. Verify later revisions and local law.",
    },
    "CODEX_GSFA": {
        "title": "Codex General Standard for Food Additives (CXS 192-1995) Online Database",
        "url": "https://www.fao.org/gsfaonline/",
        "scope": "Verify additive, INS number, food category, function and maximum level/GMP; the standard title alone is insufficient.",
    },
    "EU_1169_2011": {
        "title": "Regulation (EU) No 1169/2011, Annex II",
        "url": "https://eur-lex.europa.eu/eli/reg/2011/1169/oj",
        "scope": "EU ingredient-substance/product list for allergen labelling; not Vietnamese law.",
    },
    "FDA_MAJOR_ALLERGENS": {
        "title": "US FDA Food Allergies",
        "url": "https://www.fda.gov/food/nutrition-food-labeling-and-critical-foods/food-allergies",
        "scope": "US allergen compliance and food allergen controls; not Vietnamese law.",
    },
    "FOODON": {
        "title": "FoodOn ontology",
        "url": "https://foodon.org/",
        "scope": "Vocabulary/identity reference only; not a clinical or regulatory risk source.",
    },
    "VN_TT24_2019": {
        "title": "Vietnam Ministry of Health Circular 24/2019/TT-BYT",
        "url": "",
        "scope": "Official consolidated text, amendments, annex and food-category conditions have not yet been attached and checked; regulatory claims remain pending.",
    },
}

EFSA_VERIFIED = {
    "INS 621": ("EFSA_GLUTAMATES_2017", "Conclusion/abstract: group ADI and expression basis."),
    "INS 1420": ("EFSA_MODIFIED_STARCH_2017", "Conclusion/abstract: reported-use conclusion; not an individualized guarantee."),
    "INS 220": ("EFSA_SULFITES_2022", "Conclusion: withdrawal of temporary ADI and margin-of-exposure approach."),
}

ADDITIVE_REVIEW = {
    "INS 621": ("source_checked", "EFSA_GLUTAMATES_2017"),
    "INS 211": ("pending_primary_source_check", "JECFA_DB"),
    "INS 102": ("pending_primary_source_check", "JECFA_DB"),
    "INS 951": ("source_checked", "WHO_JECFA_ASPARTAME_2023"),
    "INS 330": ("pending_primary_source_check", "JECFA_DB"),
    "INS 322(i)": ("pending_primary_source_check", "JECFA_DB"),
    "INS 500(ii)": ("pending_primary_source_check", "JECFA_DB"),
    "INS 503(ii)": ("pending_primary_source_check", "JECFA_DB"),
    "INS 471": ("pending_primary_source_check", "JECFA_DB"),
    "INS 1420": ("source_checked", "EFSA_MODIFIED_STARCH_2017"),
    "INS 220": ("source_checked", "EFSA_SULFITES_2022"),
    "INS 150d": ("pending_primary_source_check", "JECFA_DB"),
    "INS 631": ("pending_primary_source_check", "JECFA_DB"),
    "INS 627": ("pending_primary_source_check", "JECFA_DB"),
    "INS 110": ("pending_primary_source_check", "JECFA_DB"),
    "INS 129": ("pending_primary_source_check", "JECFA_DB"),
    "INS 202": ("pending_primary_source_check", "JECFA_DB"),
    "INS 950": ("pending_primary_source_check", "JECFA_DB"),
    "INS 415": ("pending_primary_source_check", "JECFA_DB"),
    "INS 412": ("pending_primary_source_check", "JECFA_DB"),
    "INS 407": ("pending_primary_source_check", "JECFA_DB"),
    "INS 331(iii)": ("pending_primary_source_check", "JECFA_DB"),
    "INS 160a": ("pending_primary_source_check", "JECFA_DB"),
    "INS 300": ("pending_primary_source_check", "JECFA_DB"),
    "INS 551": ("pending_primary_source_check", "JECFA_DB"),
    "INS 420": ("pending_primary_source_check", "JECFA_DB"),
}


def _read_json(path):
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _load_preserved_legacy_claims():
    path = os.path.join(ROOT, "data", "knowledge_base_source_review_template.csv")
    if not os.path.isfile(path):
        return {}
    claims_by_code = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            claim_id = row.get("claim_id", "")
            if ":legacy_health_claim:" not in claim_id or not row.get("claim_value"):
                continue
            code = claim_id.split(":legacy_health_claim:", 1)[0]
            try:
                claim = json.loads(row["claim_value"])
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid legacy claim JSON in {path}: {claim_id}"
                ) from error
            claims_by_code.setdefault(code, []).append(claim)
    return claims_by_code


def enrich_additives():
    path = os.path.join(KB_DIR, "additives_standard.json")
    additives = _read_json(path)
    preserved_legacy_claims = _load_preserved_legacy_claims()
    for additive in additives:
        code = additive["ins_code"]
        legacy_regulatory_record = additive.get("unverified_legacy_regulatory_claims") or {}
        legacy_regulatory_claims = legacy_regulatory_record.get("claims") or {
            "vietnam": additive.get("vn_regulation", ""),
            "codex": additive.get("codex_standard", ""),
        }
        additive["unverified_legacy_regulatory_claims"] = {
            "claims": legacy_regulatory_claims,
            "verification_status": "pending_primary_source_check",
            "use_for_compliance_decision": False,
        }
        legacy_risks = (
            additive.get("unverified_legacy_health_claims")
            or additive.get("contraindications")
            or preserved_legacy_claims.get(code, [])
        )
        additive["unverified_legacy_health_claims"] = [
            {
                **claim,
                "verification_status": "pending_primary_source_check",
                "use_for_personalized_alerts": False,
            }
            for claim in legacy_risks
        ]
        additive["contraindications"] = []
        additive["health_considerations"] = []
        additive["health_claim_policy"] = (
            "Only source-checked, explicitly scoped considerations may be shown. "
            "Do not infer individual safety, diagnosis, or treatment from an ingredient match."
        )

        additive["regulatory_claim"] = {
            "jurisdiction": "Vietnam",
            "claim": legacy_regulatory_claims.get("vietnam", ""),
            "verification_status": "pending_official_text_and_food_category_check",
            "source_id": "VN_TT24_2019",
            "source_url": SOURCES["VN_TT24_2019"]["url"],
            "use_for_compliance_decision": False,
        }
        additive["codex_claim"] = {
            "claim": legacy_regulatory_claims.get("codex", ""),
            "verification_status": "pending_gsfa_food_category_and_maximum_level_check",
            "source_id": "CODEX_GSFA",
            "source_url": SOURCES["CODEX_GSFA"]["url"],
            "use_for_compliance_decision": False,
        }

        review = ADDITIVE_REVIEW.get(code)
        if not review:
            norm = re.sub(r'(\d)([ivx]+)$', r'\1(\2)', code)
            review = ADDITIVE_REVIEW.get(norm)
        if not review:
            norm2 = re.sub(r'(\d)([a-d])$', r'\1(\2)', code)
            review = ADDITIVE_REVIEW.get(norm2)
        status, source_id = review if review else ("pending_primary_source_check", "JECFA_DB")
        efsa = additive.get("efsa_opinion") or {}
        source_url = SOURCES[source_id]["url"]
        additive["adi_assessment"] = {
            "display_text": additive.get("adi_value", ""),
            "numeric_upper_mg_per_kg_bw_day": (
                additive.get("adi_upper") if status == "source_checked" else None
            ),
            "authority": (
                "EFSA" if source_id.startswith("EFSA_")
                else "WHO/FAO JECFA" if source_id.startswith("WHO_")
                else "JECFA"
            ),
            "source_id": source_id,
            "source_url": source_url,
            "source_locator": (
                "Individual record and monograph not yet recorded."
                if status == "pending_primary_source_check"
                else SOURCES[source_id]["scope"]
            ),
            "verification_status": status,
            "use_for_numeric_compliance": status == "source_checked",
            "review_note": (
                "Verify current authority, population, group members, units and status "
                "against the primary evaluation before calculating an intake limit."
            ),
        }
        additive["allergen_assessment"] = {
            "groups": additive.get("allergen_groups", []),
            "verification_status": (
                "source_checked_with_source_dependent_origin"
                if code == "INS 322(i)"
                else "not_individually_assessed"
            ),
            "source_ids": ["CODEX_CXS_1_1985", "EU_1169_2011"],
            "note": (
                "For lecithin, the source material must be established; the E-number alone "
                "does not establish soy or egg origin."
                if code == "INS 322(i)"
                else "An empty list means no allergen was recorded here, not proof of absence."
            ),
        }
        additive["sources"] = [
            {
                "id": source_id,
                "title": SOURCES[source_id]["title"],
                "url": SOURCES[source_id]["url"],
                "locator": SOURCES[source_id]["scope"],
            }
        ]

        if code == "INS 951":
            additive["adi_assessment"].update({
                "display_text": "JECFA ADI: 0-40 mg/kg thể trọng/ngày",
                "numeric_upper_mg_per_kg_bw_day": 40.0,
                "authority": "WHO/FAO JECFA",
                "source_locator": "WHO/FAO/IARC joint 2023 statement and linked JECFA chemical record.",
            })
            additive["health_considerations"] = [{
                "condition": "pku",
                "severity": "CAUTION",
                "message": (
                    "Aspartame contains phenylalanine. FDA advises people with PKU to "
                    "avoid or restrict it; this is a label-based precaution, not a personalized diagnosis."
                ),
                "source_id": "FDA_ASPARTAME",
                "source_url": SOURCES["FDA_ASPARTAME"]["url"],
                "verification_status": "source_checked",
            }]
        elif code == "INS 621":
            additive["adi_assessment"].update({
                "display_text": "EFSA group ADI: 30 mg/kg thể trọng/ngày, tính theo glutamic acid (E620-E625)",
                "numeric_upper_mg_per_kg_bw_day": None,
                "group_adi_mg_per_kg_bw_day": 30.0,
                "group_members": "E620-E625",
                "group_expression_basis": "glutamic acid",
                "authority": "EFSA",
                "source_locator": SOURCES["EFSA_GLUTAMATES_2017"]["scope"],
                "use_for_numeric_compliance": False,
            })
        elif code == "INS 1420":
            additive["adi_assessment"].update({
                "display_text": "EFSA: không cần ADI dạng số cho modified starches ở mức sử dụng được báo cáo",
                "numeric_upper_mg_per_kg_bw_day": None,
                "authority": "EFSA",
                "source_locator": SOURCES["EFSA_MODIFIED_STARCH_2017"]["scope"],
                "use_for_numeric_compliance": False,
            })
        elif code == "INS 220":
            additive["adi_assessment"].update({
                "display_text": "EFSA 2022 rút ADI nhóm tạm thời; dùng phương pháp margin of exposure",
                "numeric_upper_mg_per_kg_bw_day": None,
                "authority": "EFSA",
                "source_locator": SOURCES["EFSA_SULFITES_2022"]["scope"],
                "use_for_numeric_compliance": False,
            })

        additive["adi_value"] = additive["adi_assessment"]["display_text"]
        additive["adi_upper"] = additive["adi_assessment"].get(
            "numeric_upper_mg_per_kg_bw_day"
        )
        additive["adi_source"] = additive["adi_assessment"].get("source_url", "")
        additive["vn_regulation"] = "Pending official-source review; not for compliance decisions."
        additive["codex_standard"] = "Pending substance, food-category and limit verification."

        if efsa:
            efsa["risk_groups"] = []
            efsa["risk_groups_verification_status"] = "not_used_for_personalized_alerts"
            if code in EFSA_VERIFIED:
                efsa["source_url"] = SOURCES[EFSA_VERIFIED[code][0]]["url"]
                efsa["source_locator"] = EFSA_VERIFIED[code][1]
                efsa["verification_status"] = "source_checked"
            else:
                efsa["verification_status"] = "pending_primary_source_check"

    _write_json(path, additives)
    return additives


def enrich_allergens():
    path = os.path.join(KB_DIR, "allergen_groups.json")
    groups = _read_json(path)
    codex_ids = {"gluten", "milk", "egg", "soy", "peanut", "nuts", "shellfish", "fish", "sulfites"}
    eu_ids = {"gluten", "milk", "egg", "soy", "peanut", "nuts", "shellfish", "fish", "celery", "mustard", "sesame", "sulfites", "lupin", "molluscs"}
    codex_locators = {
        "gluten": "CXS 1-1985, section 4.2.1.4: cereals containing gluten",
        "milk": "CXS 1-1985, section 4.2.1.4: milk and milk products",
        "egg": "CXS 1-1985, section 4.2.1.4: eggs and egg products",
        "soy": "CXS 1-1985, section 4.2.1.4: soybeans and soy products",
        "peanut": "CXS 1-1985, section 4.2.1.4: peanuts and peanut products",
        "nuts": "CXS 1-1985, section 4.2.1.4: tree nuts",
        "shellfish": "CXS 1-1985, section 4.2.1.4: crustacea and products thereof",
        "fish": "CXS 1-1985, section 4.2.1.4: fish and fish products",
        "sulfites": "CXS 1-1985, section 4.2.1.4: sulphite at 10 mg/kg or more",
    }
    eu_locators = {
        "gluten": "Regulation (EU) No 1169/2011, Annex II, item 1",
        "shellfish": "Regulation (EU) No 1169/2011, Annex II, item 2 (crustaceans)",
        "egg": "Regulation (EU) No 1169/2011, Annex II, item 3",
        "fish": "Regulation (EU) No 1169/2011, Annex II, item 4",
        "peanut": "Regulation (EU) No 1169/2011, Annex II, item 5",
        "soy": "Regulation (EU) No 1169/2011, Annex II, item 6",
        "milk": "Regulation (EU) No 1169/2011, Annex II, item 7",
        "nuts": "Regulation (EU) No 1169/2011, Annex II, item 8",
        "celery": "Regulation (EU) No 1169/2011, Annex II, item 9",
        "mustard": "Regulation (EU) No 1169/2011, Annex II, item 10",
        "sesame": "Regulation (EU) No 1169/2011, Annex II, item 11",
        "sulfites": "Regulation (EU) No 1169/2011, Annex II, item 12",
        "lupin": "Regulation (EU) No 1169/2011, Annex II, item 13",
        "molluscs": "Regulation (EU) No 1169/2011, Annex II, item 14",
    }
    fda_ids = {
        "milk", "egg", "fish", "shellfish", "nuts", "peanut", "wheat", "soy", "sesame"
    }
    for group in groups:
        group_id = group["id"]
        if group_id == "gluten":
            ambiguous_terms = {"mì", "bột", "flour"}
            group["keywords"] = [
                keyword for keyword in group.get("keywords", [])
                if keyword.casefold() not in ambiguous_terms
            ]
        if group_id == "soy":
            group["keywords"] = [
                keyword for keyword in group.get("keywords", [])
                if keyword.casefold() != "tương"
            ]
        group["negation_rules"] = {
            keyword: "ALLERGEN_RISK"
            for keyword in group.get("negation_rules", {})
            if keyword in group.get("keywords", [])
        }
        source_ids = []
        if group_id in codex_ids:
            source_ids.append("CODEX_CXS_1_1985")
        if group_id in eu_ids:
            source_ids.append("EU_1169_2011")
        if group_id in fda_ids:
            source_ids.append("FDA_MAJOR_ALLERGENS")

        regulatory_claims = {}
        for jurisdiction, source_id, locator in (
            ("codex", "CODEX_CXS_1_1985", codex_locators.get(group_id)),
            ("eu", "EU_1169_2011", eu_locators.get(group_id)),
            (
                "united_states",
                "FDA_MAJOR_ALLERGENS",
                "FDA major food allergen list; scope depends on the US statutory definitions.",
            ) if group_id in fda_ids else ("", "", None),
        ):
            if not jurisdiction:
                continue
            source_available = bool(locator)
            regulatory_claims[jurisdiction] = {
                "claim": (
                    f"Category appears in the linked {SOURCES[source_id]['title']}."
                    if source_available else None
                ),
                "source_id": source_id if source_available else None,
                "source_url": SOURCES[source_id]["url"] if source_available else None,
                "source_locator": locator,
                "verification_status": (
                    "source_checked_against_linked_text"
                    if source_available else "not_assessed"
                ),
                "use_for_compliance_decision": False,
            }
        regulatory_claims["vietnam"] = {
            "claim": None,
            "source_id": "VN_TT24_2019",
            "source_url": SOURCES["VN_TT24_2019"]["url"] or None,
            "source_locator": None,
            "verification_status": "pending_official_text_and_amendment_check",
            "use_for_compliance_decision": False,
        }

        unsupported = group_id == "beef"
        group["enabled_for_allergen_screening"] = not unsupported
        group["severity"] = "warning" if not unsupported else "not_an_allergen_alert"
        group["verification_status"] = (
            "unsupported_as_major_label_allergen"
            if unsupported else "source_linked_category_scope_aliases_pending_review"
        )
        group["keyword_vocabulary_status"] = (
            "not_used_for_allergen_screening"
            if unsupported else "curated_aliases_pending_expert_review"
        )
        group["source_ids"] = source_ids
        group["source_urls"] = [SOURCES[item]["url"] for item in source_ids]
        group["regulatory_claims"] = regulatory_claims
        group.pop("regulation", None)
        group["screening_limit"] = (
            "Keyword match is a possible label mention only; verify the full ingredient phrase, "
            "source, processing and cross-contact. Absence of a keyword is not proof of absence."
        )
        if group_id == "sulfites":
            group["threshold_note"] = (
                "The linked Codex CXS 1-1985 copy says sulphite at 10 mg/kg or more; "
                "the linked text is amended through 2010. Confirm current rule and jurisdiction."
            )
        if unsupported:
            group["note"] = (
                "Beef remains a food ingredient term but is not treated as one of the "
                "sourced priority allergen-label groups in this dataset."
            )
            group["severity"] = "warning"
        else:
            group["note"] = (
                "Label-screening category only. This record does not establish clinical "
                "allergy, cross-reactivity, an individual's threshold, or a jurisdiction's "
                "current legal requirements."
            )
    _write_json(path, groups)
    return groups


COMMON_INGREDIENTS = [
    ("sugar", "Đường", ["đường", "sugar"], None),
    ("rice", "Gạo", ["gạo", "rice"], None),
    ("soybean", "Đậu nành", ["đậu nành", "đậu tương", "soybean"], "soy"),
    ("palm oil", "Dầu cọ", ["dầu cọ", "palm oil"], None),
    ("olive oil", "Dầu ô liu", ["dầu ô liu", "olive oil"], None),
    ("chicken", "Thịt gà", ["thịt gà", "gà", "chicken"], None),
    ("shrimp", "Tôm", ["tôm", "shrimp"], "shellfish"),
    ("peanut", "Đậu phộng (lạc)", ["đậu phộng", "lạc", "peanut"], "peanut"),
    ("almond", "Hạnh nhân", ["hạnh nhân", "almond"], "nuts"),
    ("gluten", "Gluten", ["gluten"], "gluten"),
    ("cocoa", "Ca cao", ["ca cao", "cocoa"], None),
    ("coconut", "Dừa", ["dừa", "coconut"], None),
    ("tomato", "Cà chua", ["cà chua", "tomato"], None),
    ("potato", "Khoai tây", ["khoai tây", "potato"], None),
    ("onion", "Hành tây", ["hành tây", "onion"], None),
    ("oat", "Yến mạch", ["yến mạch", "oat"], None),
    ("salmon", "Cá hồi", ["cá hồi", "salmon"], "fish"),
    ("tuna", "Cá ngừ", ["cá ngừ", "tuna"], "fish"),
    ("apple", "Táo", ["táo", "apple"], None),
    ("banana", "Chuối", ["chuối", "banana"], None),
    ("lentil", "Đậu lăng", ["đậu lăng", "lentil"], None),
    ("chickpea", "Đậu gà", ["đậu gà", "chickpea"], None),
    ("flour", "Bột", ["bột", "flour"], None),
    ("vinegar", "Giấm", ["giấm", "vinegar"], None),
    ("yeast", "Men", ["men", "yeast"], None),
    ("butter", "Bơ", ["bơ", "butter"], "milk"),
    ("cheese", "Phô mai", ["phô mai", "cheese"], "milk"),
    ("lamb", "Thịt cừu", ["thịt cừu", "lamb"], None),
    ("sesame seed", "Hạt mè (vừng)", ["mè", "vừng", "sesame seed"], "sesame"),
    ("walnut", "Óc chó", ["óc chó", "walnut"], "nuts"),
    ("hazelnut", "Hạt phỉ", ["hạt phỉ", "hazelnut"], "nuts"),
    ("lemon", "Chanh vàng", ["chanh vàng", "lemon"], None),
    ("cucumber", "Dưa chuột", ["dưa chuột", "cucumber"], None),
    ("cinnamon", "Quế", ["quế", "cinnamon"], None),
    ("coriander", "Rau mùi", ["rau mùi", "coriander"], None),
    ("chili pepper", "Ớt", ["ớt", "chili pepper"], None),
]


def write_ingredient_catalog(groups):
    records = []
    for group in groups:
        records.append({
            "id": group["id"],
            "canonical_name_vi": group["name_vi"],
            "canonical_name_en": group["name_en"],
            "label_aliases": group["keywords"],
            "ingredient_type": (
                "allergen_label_category"
                if group.get("enabled_for_allergen_screening")
                else "ingredient_term_not_screened_as_priority_allergen"
            ),
            "allergen_group_id": (
                group["id"] if group.get("enabled_for_allergen_screening") else None
            ),
            "source_ids": group["source_ids"],
            "source_urls": group["source_urls"],
            "verification_status": group["verification_status"],
            "use_for_allergen_screening": group.get("enabled_for_allergen_screening", False),
            "regulatory_claims": group.get("regulatory_claims", {}),
            "scope": (
                "Curated common label vocabulary for screening, not an exhaustive ingredient "
                "ontology, ingredient-composition database, or proof of presence/absence."
            ),
        })

    foodon_catalog = load_foodon_entities(os.path.join(ROOT, "data"))
    if not foodon_catalog:
        raise FileNotFoundError(
            "FoodOn ontology is required to build the source-linked ingredient catalogue."
        )
    for label, name_vi, aliases, allergen_group_id in COMMON_INGREDIENTS:
        entity = find_foodon_exact_label(label, foodon_catalog)
        if not entity:
            raise ValueError(f"No unique exact FoodOn label match for ingredient: {label}")
        records.append({
            "id": entity["curie"].upper(),
            "canonical_name_vi": name_vi,
            "canonical_name_en": entity["label"],
            "label_aliases": aliases,
            "ingredient_type": "common_food_ingredient",
            "foodon_curie": entity["curie"].upper(),
            "foodon_label": entity["label"],
            "allergen_group_id": allergen_group_id,
            "source_ids": ["FOODON"],
            "source_urls": [SOURCES["FOODON"]["url"]],
            "verification_status": "source_checked_for_ontology_identity_only",
            "alias_verification_status": "pending_label_review",
            "use_for_allergen_screening": False,
            "scope": (
                "FoodOn identity is an exact unique label match. Localized aliases and any "
                "allergen-group association are candidate vocabulary only and require review; "
                "they do not establish composition, clinical allergy, or legal status."
            ),
        })
    path = os.path.join(KB_DIR, "ingredients_standard.json")
    _write_json(path, records)
    return records


def main():
    additives = enrich_additives()
    allergens = enrich_allergens()
    ingredients = write_ingredient_catalog(allergens)
    print(
        f"Updated {len(additives)} additive records, "
        f"{sum(group.get('enabled_for_allergen_screening', False) for group in allergens)} "
        "source-linked allergen-label categories, and "
        f"{sum(item['ingredient_type'] == 'common_food_ingredient' for item in ingredients)} "
        "FoodOn-linked common ingredient records."
    )


if __name__ == "__main__":
    main()
