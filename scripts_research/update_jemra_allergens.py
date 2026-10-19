"""Update knowledge_base/allergen_groups.json with FAO/WHO JEMRA (2021) Part 1 standards."""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(PROJECT_ROOT, "knowledge_base", "allergen_groups.json")

with open(path, "r", encoding="utf-8") as f:
    groups = json.load(f)

# Priority mapping based on FAO/WHO JEMRA (2021) Part 1
jemra_map = {
    "gluten": {
        "jemra_priority": "GLOBAL_PRIORITY",
        "immune_mechanism": "IgE_and_coeliac_autoimmune",
        "clinical_severity": "High (Coeliac enteropathy & IgE anaphylaxis)",
        "note": "Codex & JEMRA 2021 global priority: wheat and other Triticum species, rye, barley and hybridized strains."
    },
    "crustacea": {
        "jemra_priority": "GLOBAL_PRIORITY",
        "immune_mechanism": "IgE_mediated",
        "clinical_severity": "High (Anaphylaxis risk; tropomyosin allergen)",
        "note": "Codex & JEMRA 2021 global priority: crustacea and products thereof."
    },
    "shellfish": {
        "jemra_priority": "GLOBAL_PRIORITY",
        "immune_mechanism": "IgE_mediated",
        "clinical_severity": "High (Anaphylaxis risk; tropomyosin allergen)",
        "note": "Codex & JEMRA 2021 global priority: crustacea and products thereof."
    },
    "egg": {
        "jemra_priority": "GLOBAL_PRIORITY",
        "immune_mechanism": "IgE_mediated",
        "clinical_severity": "High (Ovalbumin, ovomucoid anaphylaxis risk)",
        "note": "Codex & JEMRA 2021 global priority: eggs and egg products."
    },
    "fish": {
        "jemra_priority": "GLOBAL_PRIORITY",
        "immune_mechanism": "IgE_mediated",
        "clinical_severity": "High (Parvalbumin anaphylaxis risk)",
        "note": "Codex & JEMRA 2021 global priority: fish and fish products."
    },
    "milk": {
        "jemra_priority": "GLOBAL_PRIORITY",
        "immune_mechanism": "IgE_mediated_milk_protein",
        "clinical_severity": "High (Casein and beta-lactoglobulin anaphylaxis risk)",
        "note": "Codex & JEMRA 2021 global priority: milk and milk products. Non-immune lactose intolerance is classified separately as a food intolerance."
    },
    "peanut": {
        "jemra_priority": "GLOBAL_PRIORITY",
        "immune_mechanism": "IgE_mediated_high_potency",
        "clinical_severity": "High (Extremely low eliciting dose ED01, severe anaphylaxis)",
        "note": "Codex & JEMRA 2021 global priority: peanuts and peanut products."
    },
    "sesame": {
        "jemra_priority": "GLOBAL_PRIORITY",
        "immune_mechanism": "IgE_mediated_high_potency",
        "clinical_severity": "High (Severe systemic allergic reactions confirmed by JEMRA 2021)",
        "note": "Added by FAO/WHO JEMRA 2021 as a global priority allergen alongside Codex GSLPF."
    },
    "nuts": {
        "jemra_priority": "GLOBAL_PRIORITY",
        "immune_mechanism": "IgE_mediated",
        "clinical_severity": "High (Anaphylaxis risk)",
        "note": "JEMRA 2021 specified 6 global priority tree nuts: almond, cashew, hazelnut, pecan, pistachio, walnut. Brazil nut, macadamia, and pine nuts are regional."
    },
    "soy": {
        "jemra_priority": "REGIONAL_PRIORITY",
        "immune_mechanism": "IgE_mediated",
        "clinical_severity": "Moderate (Lower severity and potency compared to peanuts and tree nuts)",
        "note": "JEMRA 2021 recommended soybean as regional priority due to lower prevalence of severe reactions and lower potency."
    },
    "celery": {
        "jemra_priority": "REGIONAL_PRIORITY",
        "immune_mechanism": "IgE_mediated",
        "clinical_severity": "Moderate (High regional prevalence in central Europe)",
        "note": "JEMRA 2021 regional priority allergen."
    },
    "mustard": {
        "jemra_priority": "REGIONAL_PRIORITY",
        "immune_mechanism": "IgE_mediated",
        "clinical_severity": "Moderate (Regional prevalence)",
        "note": "JEMRA 2021 regional priority allergen."
    },
    "lupin": {
        "jemra_priority": "REGIONAL_PRIORITY",
        "immune_mechanism": "IgE_mediated",
        "clinical_severity": "Moderate (Cross-reactivity with peanut)",
        "note": "JEMRA 2021 regional priority allergen."
    },
    "molluscs": {
        "jemra_priority": "REGIONAL_PRIORITY",
        "immune_mechanism": "IgE_mediated",
        "clinical_severity": "Moderate to High",
        "note": "JEMRA 2021 regional priority allergen."
    },
    "sulfites": {
        "jemra_priority": "EXCLUDED_FROM_PRIORITY_ALLERGEN_LIST",
        "immune_mechanism": "non_immune_chemical_hypersensitivity",
        "clinical_severity": "Moderate (Bronchospasm in sensitive individuals at >= 10 mg/kg SO2)",
        "note": "Non-immune chemical hypersensitivity. Excluded by JEMRA 2021 from immune-mediated priority allergen list; retains labeling threshold at 10 mg/kg."
    },
    "beef": {
        "jemra_priority": "REGIONAL_PRIORITY",
        "immune_mechanism": "IgE_or_alpha_gal",
        "clinical_severity": "Low to Moderate",
        "note": "Non-priority allergen unless alpha-gal syndrome or regional condition."
    }
}

for g in groups:
    gid = g["id"]
    if gid in jemra_map:
        info = jemra_map[gid]
        g["jemra_priority"] = info["jemra_priority"]
        g["immune_mechanism"] = info["immune_mechanism"]
        g["clinical_severity"] = info["clinical_severity"]
        g["note"] = info["note"]
        if "FAO_WHO_JEMRA_2021" not in g.get("source_ids", []):
            g["source_ids"] = g.get("source_ids", []) + ["FAO_WHO_JEMRA_2021"]
        jemra_url = "https://cdn.who.int/media/docs/default-source/food-safety/jemra/1st-allergen-summary-report-10may2021.pdf"
        if jemra_url not in g.get("source_urls", []):
            g["source_urls"] = g.get("source_urls", []) + [jemra_url]

        if "regulatory_claims" not in g:
            g["regulatory_claims"] = {}
        g["regulatory_claims"]["fao_who_jemra"] = {
            "claim": f"Evaluated in FAO/WHO JEMRA (2021) Part 1 as {info['jemra_priority']}.",
            "source_id": "FAO_WHO_JEMRA_2021",
            "source_url": jemra_url,
            "source_locator": "JEMRA Summary Report 10 May 2021, Section: Conclusions",
            "verification_status": "source_checked",
            "use_for_compliance_decision": False
        }

# Remove lactose from milk
milk_g = next((g for g in groups if g["id"] == "milk"), None)
if milk_g:
    milk_g["keywords"] = [k for k in milk_g["keywords"] if k != "lactose"]
    milk_g["negation_rules"] = {k: v for k, v in milk_g["negation_rules"].items() if k != "lactose"}
    milk_g["name_en"] = "Milk and milk products (excluding lactose)"

# Add lactose as distinct intolerance group if not present
if not any(g["id"] == "lactose" for g in groups):
    lactose_group = {
        "id": "lactose",
        "name_vi": "Bất dung nạp Lactose (Không phải dị ứng miễn dịch)",
        "name_en": "Lactose intolerance (Non-immune food intolerance)",
        "severity": "warning",
        "keywords": ["lactose", "đường lactose"],
        "negation_rules": {"lactose": "ALLERGEN_RISK", "đường lactose": "ALLERGEN_RISK"},
        "note": "Non-immune disaccharide intolerance caused by intestinal lactase deficiency. Excluded by FAO/WHO JEMRA 2021 from immune-mediated priority allergen list.",
        "enabled_for_allergen_screening": True,
        "verification_status": "source_linked_category_scope_aliases_pending_review",
        "source_ids": ["FAO_WHO_JEMRA_2021"],
        "source_urls": ["https://cdn.who.int/media/docs/default-source/food-safety/jemra/1st-allergen-summary-report-10may2021.pdf"],
        "screening_limit": "Non-immune food intolerance, causing gastrointestinal symptoms but no IgE-mediated anaphylaxis.",
        "jemra_priority": "EXCLUDED_FROM_PRIORITY_ALLERGEN_LIST",
        "immune_mechanism": "non_immune_enzyme_deficiency",
        "clinical_severity": "Gastrointestinal discomfort (Bloating, cramps, diarrhea)",
        "regulatory_claims": {
            "fao_who_jemra": {
                "claim": "Excluded from priority allergen list as non-immune intolerance.",
                "source_id": "FAO_WHO_JEMRA_2021",
                "source_url": "https://cdn.who.int/media/docs/default-source/food-safety/jemra/1st-allergen-summary-report-10may2021.pdf",
                "source_locator": "JEMRA Summary Report 10 May 2021, Section: Conclusions",
                "verification_status": "source_checked",
                "use_for_compliance_decision": False
            },
            "vietnam": {
                "claim": None,
                "source_id": "VN_TT24_2019",
                "source_url": None,
                "source_locator": None,
                "verification_status": "pending_official_text_and_amendment_check",
                "use_for_compliance_decision": False
            }
        },
        "keyword_vocabulary_status": "curated_aliases_pending_expert_review"
    }
    groups.append(lactose_group)

with open(path, "w", encoding="utf-8") as f:
    json.dump(groups, f, ensure_ascii=False, indent=2)

print(f"Updated {len(groups)} allergen groups with FAO/WHO JEMRA (2021) metadata successfully!")
