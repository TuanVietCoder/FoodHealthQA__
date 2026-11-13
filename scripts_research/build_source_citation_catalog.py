import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.claim_locator_resolver import (
    STATUTORY_SOURCE_CATALOG,
    EFSA_TOPIC_MAP,
    JECFA_CHEMICAL_MAP,
)

catalog = {}

# 1. Statutory sources
for loc, meta in STATUTORY_SOURCE_CATALOG.items():
    catalog[loc] = dict(meta)

# 2. EFSA Opinions
for key, meta in EFSA_TOPIC_MAP.items():
    loc = meta["locator"]
    if loc not in catalog:
        catalog[loc] = {
            "citation_id": "EFSA_OPINIONS" if "E171" not in loc else "EFSA_2021_TITANIUM_DIOXIDE_E171",
            "title": meta["title"],
            "url": meta["url"],
            "scope": meta["scope"],
        }

# 3. JECFA Chemical Evaluations
for code, meta in JECFA_CHEMICAL_MAP.items():
    loc = meta["locator"]
    if loc not in catalog:
        catalog[loc] = {
            "citation_id": "JECFA_EVALUATIONS",
            "title": f"WHO/FAO JECFA Chemical Monograph - {meta['name']} (INS {code.upper()})",
            "url": meta["url"],
            "scope": f"Chemical identity and toxicological evaluation; ADI: {meta['adi']}.",
        }

# 4. Fallbacks / Standard general items
if "WHO_FAO_JECFA#Compendium_Additives_ADI" not in catalog:
    catalog["WHO_FAO_JECFA#Compendium_Additives_ADI"] = {
        "citation_id": "JECFA_EVALUATIONS",
        "title": "FAO/WHO JECFA Combined Compendium of Food Additive Specifications & Evaluations",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/",
        "scope": "General JECFA toxicological monographs and ADI evaluations for approved food additives.",
    }

# The catalog is also extended by fetch_jecfa.py and hand-added entries, so never rebuild it from scratch:
# keys already present are kept as they are, only missing keys are added.
CATALOG_PATH = PROJECT_ROOT / "knowledge_base" / "sources" / "catalog" / "source_citation_catalog.json"
existing = json.loads(CATALOG_PATH.read_text(encoding="utf-8")) if CATALOG_PATH.exists() else {}
added = [k for k in catalog if k not in existing]
for k in added:
    existing[k] = catalog[k]
CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
CATALOG_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Catalog {CATALOG_PATH}: {len(existing)} entries ({len(added)} added, existing entries untouched)")
