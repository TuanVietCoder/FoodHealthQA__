"""Dynamic Claim Locator and Source Resolver for FoodHealthQA.

Resolves accurate, entity-specific statutory and scientific claim-level locators
for QA benchmark records dynamically, eliminating hardcoded locator mismatches.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.additive_linker import normalize_ins_key

# ---------------------------------------------------------------------------
# 1. JECFA Chemical Database Locators & URLs
# ---------------------------------------------------------------------------
JECFA_CHEMICAL_MAP: dict[str, dict[str, str]] = {
    # Sweeteners
    "951": {
        "locator": "WHO_FAO_JECFA_2023#Chemical62_Aspartame",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/62",
        "name": "Aspartame",
        "adi": "0-40 mg/kg bw",
    },
    "950": {
        "locator": "WHO_FAO_JECFA#Chemical2434_AcesulfamePotassium",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/2434",
        "name": "Acesulfame potassium",
        "adi": "0-15 mg/kg bw",
    },
    "955": {
        "locator": "WHO_FAO_JECFA#Chemical2340_Sucralose",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/2340",
        "name": "Sucralose",
        "adi": "0-15 mg/kg bw",
    },
    "960": {
        "locator": "WHO_FAO_JECFA#Chemical2605_SteviolGlycosides",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/2605",
        "name": "Steviol glycosides",
        "adi": "0-4 mg/kg bw (as steviol)",
    },
    "960a": {
        "locator": "WHO_FAO_JECFA#Chemical2605_SteviolGlycosides",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/2605",
        "name": "Steviol glycosides",
        "adi": "0-4 mg/kg bw (as steviol)",
    },
    "961": {
        "locator": "WHO_FAO_JECFA#Chemical2339_Neotame",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/2339",
        "name": "Neotame",
        "adi": "0-2 mg/kg bw",
    },
    "420": {
        "locator": "WHO_FAO_JECFA#Chemical1889_Sorbitol_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1889",
        "name": "Sorbitol",
        "adi": "ADI not specified",
    },
    # Flavor enhancers
    "621": {
        "locator": "WHO_FAO_JECFA#Chemical1909_MonosodiumGlutamate_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1909",
        "name": "Monosodium L-glutamate",
        "adi": "ADI not specified",
    },
    "627": {
        "locator": "WHO_FAO_JECFA#Chemical1907_DisodiumGuanylate_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1907",
        "name": "Disodium 5'-guanylate",
        "adi": "ADI not specified",
    },
    "631": {
        "locator": "WHO_FAO_JECFA#Chemical1908_DisodiumInosinate_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1908",
        "name": "Disodium 5'-inosinate",
        "adi": "ADI not specified",
    },
    # Preservatives
    "211": {
        "locator": "WHO_FAO_JECFA#Chemical1910_SodiumBenzoate",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1910",
        "name": "Sodium benzoate",
        "adi": "0-5 mg/kg bw",
    },
    "202": {
        "locator": "WHO_FAO_JECFA#Chemical2443_PotassiumSorbate",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/2443",
        "name": "Potassium sorbate",
        "adi": "0-25 mg/kg bw",
    },
    "220": {
        "locator": "WHO_FAO_JECFA#Chemical2018_SulfurDioxide",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/2018",
        "name": "Sulfur dioxide",
        "adi": "0-0.7 mg/kg bw",
    },
    "221": {
        "locator": "WHO_FAO_JECFA#Chemical2019_SodiumSulfite",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/2019",
        "name": "Sodium sulfite",
        "adi": "0-0.7 mg/kg bw",
    },
    "250": {
        "locator": "WHO_FAO_JECFA#Chemical2444_SodiumNitrite",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/2444",
        "name": "Sodium nitrite",
        "adi": "0-0.07 mg/kg bw",
    },
    "251": {
        "locator": "WHO_FAO_JECFA#Chemical2445_SodiumNitrate",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/2445",
        "name": "Sodium nitrate",
        "adi": "0-3.7 mg/kg bw",
    },
    # Antioxidants
    "300": {
        "locator": "WHO_FAO_JECFA#Chemical1845_AscorbicAcid_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1845",
        "name": "Ascorbic acid",
        "adi": "ADI not specified",
    },
    "320": {
        "locator": "WHO_FAO_JECFA#Chemical1847_BHA",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1847",
        "name": "Butylated hydroxyanisole (BHA)",
        "adi": "0-0.5 mg/kg bw",
    },
    "321": {
        "locator": "WHO_FAO_JECFA#Chemical1848_BHT",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1848",
        "name": "Butylated hydroxytoluene (BHT)",
        "adi": "0-0.3 mg/kg bw",
    },
    "322": {
        "locator": "WHO_FAO_JECFA#Chemical1849_Lecithin_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1849",
        "name": "Lecithin",
        "adi": "ADI not specified",
    },
    "322(i)": {
        "locator": "WHO_FAO_JECFA#Chemical1849_Lecithin_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1849",
        "name": "Lecithin",
        "adi": "ADI not specified",
    },
    # Acidity regulators & raising agents
    "330": {
        "locator": "WHO_FAO_JECFA#Chemical1850_CitricAcid_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1850",
        "name": "Citric acid",
        "adi": "ADI not specified",
    },
    "331(iii)": {
        "locator": "WHO_FAO_JECFA#Chemical1912_TrisodiumCitrate_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1912",
        "name": "Trisodium citrate",
        "adi": "ADI not specified",
    },
    "500(ii)": {
        "locator": "WHO_FAO_JECFA#Chemical1893_SodiumHydrogenCarbonate_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1893",
        "name": "Sodium hydrogen carbonate",
        "adi": "ADI not specified",
    },
    "503(ii)": {
        "locator": "WHO_FAO_JECFA#Chemical1894_AmmoniumHydrogenCarbonate_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1894",
        "name": "Ammonium hydrogen carbonate",
        "adi": "ADI not specified",
    },
    "262(i)": {
        "locator": "WHO_FAO_JECFA#Chemical1891_SodiumAcetate_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1891",
        "name": "Sodium acetate",
        "adi": "ADI not specified",
    },
    # Thickeners, Stabilizers, Emulsifiers
    "401": {
        "locator": "WHO_FAO_JECFA#Chemical1870_SodiumAlginate_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1870",
        "name": "Sodium alginate",
        "adi": "ADI not specified",
    },
    "407": {
        "locator": "WHO_FAO_JECFA#Chemical1872_Carrageenan_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1872",
        "name": "Carrageenan",
        "adi": "ADI not specified",
    },
    "410": {
        "locator": "WHO_FAO_JECFA#Chemical1874_CarobBeanGum_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1874",
        "name": "Carob bean gum",
        "adi": "ADI not specified",
    },
    "412": {
        "locator": "WHO_FAO_JECFA#Chemical1875_GuarGum_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1875",
        "name": "Guar gum",
        "adi": "ADI not specified",
    },
    "414": {
        "locator": "WHO_FAO_JECFA#Chemical1876_GumArabic_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1876",
        "name": "Gum arabic",
        "adi": "ADI not specified",
    },
    "415": {
        "locator": "WHO_FAO_JECFA#Chemical1877_XanthanGum_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1877",
        "name": "Xanthan gum",
        "adi": "ADI not specified",
    },
    "440": {
        "locator": "WHO_FAO_JECFA#Chemical1878_Pectin_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1878",
        "name": "Pectins",
        "adi": "ADI not specified",
    },
    "471": {
        "locator": "WHO_FAO_JECFA#Chemical1880_MonoDiglycerides_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1880",
        "name": "Mono- and diglycerides of fatty acids",
        "adi": "ADI not specified",
    },
    "551": {
        "locator": "WHO_FAO_JECFA#Chemical1897_SiliconDioxide_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1897",
        "name": "Silicon dioxide, amorphous",
        "adi": "ADI not specified",
    },
    "1420": {
        "locator": "WHO_FAO_JECFA#Chemical1900_StarchAcetate_GMP",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1900",
        "name": "Starch acetate",
        "adi": "ADI not specified",
    },
    # Colours
    "150d": {
        "locator": "WHO_FAO_JECFA#Chemical1852_CaramelIV",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1852",
        "name": "Caramel IV - sulfite ammonia caramel",
        "adi": "0-200 mg/kg bw",
    },
    "102": {
        "locator": "WHO_FAO_JECFA#Chemical1854_Tartrazine",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1854",
        "name": "Tartrazine",
        "adi": "0-10 mg/kg bw",
    },
    "104": {
        "locator": "WHO_FAO_JECFA#Chemical1856_QuinolineYellow",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1856",
        "name": "Quinoline Yellow",
        "adi": "0-3 mg/kg bw",
    },
    "110": {
        "locator": "WHO_FAO_JECFA#Chemical1857_SunsetYellow",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1857",
        "name": "Sunset Yellow FCF",
        "adi": "0-4 mg/kg bw",
    },
    "122": {
        "locator": "WHO_FAO_JECFA#Chemical1859_Azorubine",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1859",
        "name": "Azorubine",
        "adi": "0-4 mg/kg bw",
    },
    "124": {
        "locator": "WHO_FAO_JECFA#Chemical1860_Ponceau4R",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1860",
        "name": "Ponceau 4R",
        "adi": "0-4 mg/kg bw",
    },
    "129": {
        "locator": "WHO_FAO_JECFA#Chemical1862_AlluraRed",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1862",
        "name": "Allura Red AC",
        "adi": "0-7 mg/kg bw",
    },
    "133": {
        "locator": "WHO_FAO_JECFA#Chemical1863_BrilliantBlue",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1863",
        "name": "Brilliant Blue FCF",
        "adi": "0-6 mg/kg bw",
    },
    "171": {
        "locator": "WHO_FAO_JECFA#Chemical1866_TitaniumDioxide",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1866",
        "name": "Titanium dioxide",
        "adi": "ADI not limited",
    },
    "160a": {
        "locator": "WHO_FAO_JECFA#BetaCarotenes_Evaluation",
        "url": "https://cdn.who.int/media/docs/default-source/food-safety/jecfa/summary-and-conclusions/jecfa87_4-13-june-2019_summary-and-conclusion.pdf",
        "name": "Beta-carotenes",
        "adi": "0-5 mg/kg bw",
    },
    # Inorganic Phosphates
    "338": {
        "locator": "WHO_FAO_JECFA#Chemical1890_PhosphoricAcid_MTDI70",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1890",
        "name": "Phosphoric acid",
        "adi": "MTDI 70 mg/kg bw (as phosphorus)",
    },
    "339(i)": {
        "locator": "WHO_FAO_JECFA#Chemical1915_SodiumPhosphates_MTDI70",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1915",
        "name": "Sodium dihydrogen phosphate",
        "adi": "MTDI 70 mg/kg bw (as phosphorus)",
    },
    "341(i)": {
        "locator": "WHO_FAO_JECFA#Chemical1916_CalciumPhosphates_MTDI70",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1916",
        "name": "Calcium dihydrogen phosphate",
        "adi": "MTDI 70 mg/kg bw (as phosphorus)",
    },
    "450(i)": {
        "locator": "WHO_FAO_JECFA#Chemical1917_Diphosphates_MTDI70",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1917",
        "name": "Disodium diphosphate",
        "adi": "MTDI 70 mg/kg bw (as phosphorus)",
    },
    "452(i)": {
        "locator": "WHO_FAO_JECFA#Chemical1918_Polyphosphates_MTDI70",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/1918",
        "name": "Sodium polyphosphate",
        "adi": "MTDI 70 mg/kg bw (as phosphorus)",
    },
}

# ---------------------------------------------------------------------------
# 2. EFSA Scientific Opinions Topic & Substance Mapping
# ---------------------------------------------------------------------------
EFSA_TOPIC_MAP: dict[str, dict[str, str]] = {
    "phosphates": {
        "locator": "EFSA_ANS_2019_5674#Phosphates_ADI40",
        "url": "https://doi.org/10.2903/j.efsa.2019.5674",
        "title": "EFSA Journal 2019;17(6):5674 - Re-evaluation of phosphoric acid-phosphates (E 338-341, E 450-452)",
        "scope": "Defines group ADI 40 mg/kg bw/day as phosphorus; flags safety concern for high intake and CKD patients.",
    },
    "azo_dyes_children": {
        "locator": "EFSA_ANS_2008_852#AzoDyes_Hyperactivity_Children",
        "url": "https://doi.org/10.2903/j.efsa.2008.660",
        "title": "EFSA Journal 2008;6(3):660 - Assessment of the results of the study by McCann et al. (2007) on the effect of some colours and sodium benzoate on children's behaviour",
        "scope": "Evaluates effect of E102, E104, E110, E122, E124, E129 on child activity/attention; basis for EU mandatory warning.",
    },
    "nitrites_pregnancy": {
        "locator": "EFSA_ANS_2017_4786#Nitrite_Methemoglobin",
        "url": "https://www.efsa.europa.eu/en/efsajournal/pub/4786",
        "title": "EFSA Journal 2017;15(6):4786 - Re-evaluation of sodium nitrite (E 250) and potassium nitrite (E 249)",
        "scope": "ADI 0.07 mg/kg bw/day; highlights methemoglobinemia risk in young infants and vulnerable groups.",
    },
    "glutamates": {
        "locator": "EFSA_ANS_2017_4910#Glutamates_ADI30",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7009848/",
        "title": "EFSA Journal 2017;15(7):4910 - Re-evaluation of glutamic acid and glutamates (E 620-625)",
        "scope": "Establishes group ADI 30 mg/kg bw/day expressed as glutamic acid.",
    },
    "titanium_dioxide": {
        "locator": "EFSA_ANS_2021_6585#E171_NotSafe",
        "url": "https://www.efsa.europa.eu/en/efsajournal/pub/6585",
        "title": "EFSA Journal 2021;19(5):6585 - Safety assessment of titanium dioxide (E171)",
        "scope": "Genotoxicity concerns could not be ruled out; E171 no longer considered safe as food additive.",
    },
    "modified_starches": {
        "locator": "EFSA_ANS_2017_4911#ModifiedStarches_NoADI",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7009865/",
        "title": "EFSA Journal 2017;15(10):4911 - Re-evaluation of modified starches",
        "scope": "No numerical ADI necessary; no safety concern at reported use levels.",
    },
    "sulfites": {
        "locator": "EFSA_ANS_2022_7594#Sulfites_Safety",
        "url": "https://www.efsa.europa.eu/en/efsajournal/pub/7594",
        "title": "EFSA Journal 2022;20(11):7594 - Follow-up on the re-evaluation of sulfur dioxide and sulfites",
        "scope": "Reaffirms sensitivity in asthmatic individuals and exposure concerns.",
    },
    "bha": {
        "locator": "EFSA_ANS_2011_2392#BHA_ADI1.0",
        "url": "https://www.efsa.europa.eu/en/efsajournal/pub/2392",
        "title": "EFSA Journal 2011;9(10):2392 - Re-evaluation of butylated hydroxyanisole - BHA (E 320)",
        "scope": "Increases ADI from 0.5 to 1.0 mg/kg bw/day based on new toxicological data.",
    },
    "caramel_iv": {
        "locator": "EFSA_ANS_2011_2004#CaramelIV_Exposure",
        "url": "https://www.efsa.europa.eu/en/efsajournal/pub/2004",
        "title": "EFSA Journal 2011;9(3):2004 - Scientific Opinion on the re-evaluation of caramel colours",
        "scope": "Group ADI 300 mg/kg bw/day; 4-MEI exposure within acceptable toxicological limits.",
    },
    "acesulfame_k": {
        "locator": "EFSA_FAP_2025_9317#AcesulfameK_Reevaluation",
        "url": "https://doi.org/10.2903/j.efsa.2025.9317",
        "title": "EFSA Journal 2025 - Re-evaluation of acesulfame K (E 950)",
        "scope": "Reaffirms ADI 0-9 mg/kg bw/day.",
    },
}

# ---------------------------------------------------------------------------
# 3. Complete Statutory Citation Catalog with Primary Source URLs
# ---------------------------------------------------------------------------
STATUTORY_SOURCE_CATALOG: dict[str, dict[str, str]] = {
    "VN_TT24_2019#PhuLuc1_Bang1": {
        "citation_id": "VN_TT24_2019",
        "title": "Thông tư 24/2019/TT-BYT & VBHN 09/VBHN-BYT 2024 - Quản lý và sử dụng phụ gia thực phẩm",
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/9/09-vbhn-byt.pdf",
        "scope": "Phụ lục 1, Bảng 1: Danh mục phụ gia thực phẩm được phép sử dụng tại Việt Nam (Positive List).",
    },
    "CODEX_STAN_192_1995#Table3_GMP": {
        "citation_id": "CODEX_STAN_192_1995",
        "title": "Codex General Standard for Food Additives CXS 192-1995 (Adopted 1995, Revised 2023)",
        "url": "https://www.fao.org/fao-who-codexalimentarius/sh-proxy/en/?lnk=1&url=https%253A%252F%252Fworkspace.fao.org%252Fsites%252Fcodex%252FStandards%252FCXS%2B192-1995%252FCXS_192e.pdf",
        "scope": "Table 3: Additives Permitted for Use in Food in General in Accordance with GMP.",
    },
    "CODEX_CXS_1_1985#Sec4.2.1.4": {
        "citation_id": "CODEX_CXS_1_1985",
        "title": "Codex General Standard for the Labelling of Prepackaged Foods CXS 1-1985 (Rev. 1-1991)",
        "url": "https://www.fao.org/input/download/standards/32/CXS_001e.pdf",
        "scope": "Section 4.2.1.4: Mandatory declaration of foods and ingredients causing hypersensitivity.",
    },
    "CODEX_CXG_36_1989#ClassNames_INS": {
        "citation_id": "CODEX_CXG_36_1989",
        "title": "Codex Class Names and the International Numbering System for Food Additives CXG 36-1989",
        "url": "https://www.fao.org/fao-who-codexalimentarius/sh-proxy/en/?lnk=1&url=https%253A%252F%252Fworkspace.fao.org%252Fsites%252Fcodex%252FStandards%252FCXG%2B36-1989%252FCXG_036e.pdf",
        "scope": "Section 1-3: Functional classes and INS numbering system.",
    },
    "US_FDA_FALCPA_2004#Sec203": {
        "citation_id": "US_FDA_FALCPA",
        "title": "US FDA Food Allergen Labeling and Consumer Protection Act of 2004 (FALCPA) & FASTER Act 2021",
        "url": "https://www.fda.gov/food/nutrition-food-labeling-and-critical-foods/food-allergies",
        "scope": "Section 203: Mandatory labeling of 9 major food allergens in prepackaged foods.",
    },
    "US_FDA_21CFR172.804#b": {
        "citation_id": "US_FDA_FALCPA",
        "title": "US eCFR Title 21 § 172.804 - Aspartame Labeling Requirements",
        "url": "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-B/part-172/section-172.804",
        "scope": "Paragraph (b): Mandatory warning for Phenylketonurics (PKU): 'PHENYLKETONURICS: CONTAINS PHENYLALANINE'.",
    },
    "FAO_WHO_JEMRA_2021_PART1#PriorityAllergens": {
        "citation_id": "WHO_JEMRA_PRIORITY_ALLERGENS",
        "title": "FAO/WHO JEMRA Food Allergen Risk Assessment: Part 1 - Review and validation of Codex priority allergen list",
        "url": "https://cdn.who.int/media/docs/default-source/food-safety/jemra/1st-allergen-summary-report-10may2021.pdf",
        "scope": "Ad hoc Joint FAO/WHO Expert Consultation establishing priority allergen threshold doses.",
    },
    "FAO_WHO_JEMRA_2021_PART3#PAL_Guidance": {
        "citation_id": "FAO_WHO_JEMRA_PART_3",
        "title": "FAO/WHO JEMRA Food Allergen Risk Assessment: Part 3 - Review and establish precautionary allergen labelling",
        "url": "https://www.who.int/publications/i/item/9789240076679",
        "scope": "Principles and application of Precautionary Allergen Labelling (PAL) in food business operations.",
    },
    "EFSA_ANS_2021_6585#E171_NotSafe": {
        "citation_id": "EFSA_2021_TITANIUM_DIOXIDE_E171",
        "title": "EFSA Journal 2021;19(5):6585 - Safety assessment of titanium dioxide (E171)",
        "url": "https://www.efsa.europa.eu/en/efsajournal/pub/6585",
        "scope": "Genotoxicity concern; titanium dioxide can no longer be considered safe.",
    },
    "USDA_FDC#NutrientThresholds": {
        "citation_id": "USDA_FDC",
        "title": "USDA FoodData Central Nutrient Profiles & Daily Value Reference Thresholds",
        "url": "https://fdc.nal.usda.gov/",
        "scope": "Standard reference nutrient database and daily value nutritional criteria.",
    },
}


def extract_ins_keys(row: dict) -> list[str]:
    """Extract normalized INS keys from matched codes and question text."""
    keys: list[str] = []
    
    # 1. Primary: ins_codes_matched
    matched = str(row.get("ins_codes_matched", "")).strip()
    for part in re.split(r"[,;]+", matched):
        part = part.strip()
        if part:
            k = normalize_ins_key(part)
            if k and k not in keys:
                keys.append(k)

    # 2. Secondary: question text
    q = str(row.get("question", ""))
    for m in re.finditer(r"\bINS\s*([0-9]{3,4}(?:\([a-z0-9ivx]+\)|[a-z0-9ivx]+)?)", q, re.I):
        k = normalize_ins_key(m.group(1))
        if k and k not in keys:
            keys.append(k)

    return keys


def resolve_jecfa_locator(row: dict, keys: list[str]) -> str:
    """Resolve substance-specific JECFA evaluation locators."""
    locators: list[str] = []
    q = str(row.get("question", "")).casefold()

    # Prioritize Aspartame if mentioned in keys or question
    if "951" in keys or "aspartame" in q:
        locators.append(JECFA_CHEMICAL_MAP["951"]["locator"])

    for k in keys:
        if k in JECFA_CHEMICAL_MAP:
            loc = JECFA_CHEMICAL_MAP[k]["locator"]
            if loc not in locators:
                locators.append(loc)
                if len(locators) >= 3:
                    break

    if not locators:
        locators.append("WHO_FAO_JECFA#Compendium_Additives_ADI")

    return "; ".join(locators)


def resolve_efsa_locator(row: dict, keys: list[str]) -> str:
    """Resolve topic and substance-specific EFSA opinion locators."""
    q = str(row.get("question", "")).casefold()
    persona = str(row.get("persona_id", "")).strip()
    
    # Priority 1: Child Hyperactivity & Azo Dyes
    azo_keys = {"102", "104", "110", "122", "124", "129"}
    if "tăng động" in q or "hiếu động" in q or (persona == "child_5yo" and ("phẩm màu" in q or "màu" in q)):
        matched_azo = [k for k in keys if k in azo_keys]
        if len(matched_azo) == 1:
            specific_map = {
                "102": "EFSA_ANS_2009_1331#Tartrazine_ADI7.5",
                "104": "EFSA_ANS_2009_1329#QuinolineYellow_ADI0.5",
                "110": "EFSA_ANS_2014_3765#SunsetYellow_ADI4",
                "122": "EFSA_ANS_2009_1332#Azorubine_ADI4",
                "124": "EFSA_ANS_2009_1328#Ponceau4R_ADI4",
                "129": "EFSA_ANS_2009_1327#AlluraRed_ADI7",
            }
            return f"{EFSA_TOPIC_MAP['azo_dyes_children']['locator']}; {specific_map.get(matched_azo[0])}"
        return EFSA_TOPIC_MAP["azo_dyes_children"]["locator"]

    # Priority 2: Phosphates & Renal Disease (CKD)
    phosphate_keys = {"338", "339", "339(i)", "340", "341", "341(i)", "450", "450(i)", "451", "452", "452(i)"}
    if persona == "ckd_stage3" or "suy thận" in q or "photphat" in q or "phosphat" in q or any(k in phosphate_keys for k in keys):
        return EFSA_TOPIC_MAP["phosphates"]["locator"]

    # Priority 3: Nitrites in Pregnancy / Young Infants
    if ("250" in keys or "251" in keys or "nitrit" in q or "nitrat" in q) and (persona == "pregnant_woman" or "mang thai" in q or "thai phụ" in q):
        return EFSA_TOPIC_MAP["nitrites_pregnancy"]["locator"]

    # Priority 4: Specific Additive Opinions
    if "171" in keys or "titan" in q:
        return EFSA_TOPIC_MAP["titanium_dioxide"]["locator"]
    if "1420" in keys or "tinh bột" in q:
        return EFSA_TOPIC_MAP["modified_starches"]["locator"]
    if "220" in keys or "221" in keys or "sulfit" in q:
        return EFSA_TOPIC_MAP["sulfites"]["locator"]
    if "320" in keys or "bha" in q:
        return EFSA_TOPIC_MAP["bha"]["locator"]
    if "150d" in keys or "caramel" in q:
        return EFSA_TOPIC_MAP["caramel_iv"]["locator"]
    if "950" in keys or "acesulfame" in q:
        return EFSA_TOPIC_MAP["acesulfame_k"]["locator"]

    # Priority 5: Glutamates
    glutamate_keys = {"620", "621", "622", "623", "624", "625", "627", "631"}
    if any(k in glutamate_keys for k in keys) or "bột ngọt" in q or "glutamat" in q:
        return EFSA_TOPIC_MAP["glutamates"]["locator"]

    return EFSA_TOPIC_MAP["glutamates"]["locator"]


def resolve_dynamic_locators(row: dict) -> str:
    """Dynamically resolve claim-level locators for all citations in the row."""
    raw_citations = str(row.get("source_citations", ""))
    citations = [c.strip() for c in re.split(r"[,;\n|]+", raw_citations) if c.strip()]
    persona_id = str(row.get("persona_id", "")).strip()
    keys = extract_ins_keys(row)

    locators: list[str] = []
    for c in citations:
        if c == "JECFA_EVALUATIONS":
            locators.append(resolve_jecfa_locator(row, keys))
        elif c == "EFSA_OPINIONS":
            locators.append(resolve_efsa_locator(row, keys))
        elif c == "US_FDA_FALCPA":
            if persona_id == "pku_patient" or "aspartame" in str(row.get("question", "")).casefold():
                locators.append("US_FDA_21CFR172.804#b")
            else:
                locators.append("US_FDA_FALCPA_2004#Sec203")
        elif c == "VN_TT24_2019":
            locators.append("VN_TT24_2019#PhuLuc1_Bang1")
        elif c == "CODEX_STAN_192_1995":
            locators.append("CODEX_STAN_192_1995#Table3_GMP")
        elif c == "CODEX_CXS_1_1985":
            locators.append("CODEX_CXS_1_1985#Sec4.2.1.4")
        elif c == "CODEX_CXG_36_1989":
            locators.append("CODEX_CXG_36_1989#ClassNames_INS")
        elif c == "WHO_JEMRA_PRIORITY_ALLERGENS":
            locators.append("FAO_WHO_JEMRA_2021_PART1#PriorityAllergens")
        elif c == "FAO_WHO_JEMRA_PART_3":
            locators.append("FAO_WHO_JEMRA_2021_PART3#PAL_Guidance")
        elif c == "EFSA_2021_TITANIUM_DIOXIDE_E171":
            locators.append("EFSA_ANS_2021_6585#E171_NotSafe")
        elif c == "USDA_FDC":
            locators.append("USDA_FDC#NutrientThresholds")
        else:
            locators.append(c)

    return "; ".join(locators)


def resolve_source_urls(locators_str: str) -> list[str]:
    """Retrieve verified official primary URLs corresponding to locators."""
    urls: list[str] = []
    for loc in locators_str.split(";"):
        loc = loc.strip()
        if not loc:
            continue
        
        # Check statutory catalog
        if loc in STATUTORY_SOURCE_CATALOG:
            url = STATUTORY_SOURCE_CATALOG[loc]["url"]
            if url not in urls:
                urls.append(url)
            continue

        # Check JECFA map
        found_jecfa = False
        for entry in JECFA_CHEMICAL_MAP.values():
            if entry["locator"] == loc:
                if entry["url"] not in urls:
                    urls.append(entry["url"])
                found_jecfa = True
                break
        if found_jecfa:
            continue

        # Check EFSA topic map
        found_efsa = False
        for entry in EFSA_TOPIC_MAP.values():
            if entry["locator"] == loc:
                if entry["url"] not in urls:
                    urls.append(entry["url"])
                found_efsa = True
                break
        if found_efsa:
            continue

    return urls

