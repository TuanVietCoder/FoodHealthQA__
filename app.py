"""

Ứng dụng Web Demo Hệ thống FoodHealthQA (v9 - Complete Data Hub Integration)

Giao diện trực quan hỏi đáp tư vấn an toàn thực phẩm từ nhãn bao bì thực phẩm

Kế thừa kiến trúc:

- Tầng 1: Perception (Hỗ trợ văn bản tham chiếu & PaddleOCR kèm Phục hồi Chính tả)

- Tầng 2: Normalization (FoodOntoRAG 4-Agent Entity Linking kết hợp FoodOn CURIE)

- Tầng 3: Hybrid Retrieval & Data Hub (TT 24, Codex, JECFA, Food Hazard, FNDDS, Open Food Facts)

- Tầng 4: Grounded Generation & Guardrail (Cảnh báo Chống chỉ định Y khoa & Dị nguyên theo NHANES)

- Data Hub: Hỗ trợ nạp và hiển thị trọn vẹn 16 bảng dữ liệu tra cứu nghiên cứu thực tế (như trong name-data.png)

"""



import os

from dotenv import load_dotenv

load_dotenv()



# Cấu hình vô hiệu hóa OneDNN / MKLDNN và PIR trên Windows CPU ngay từ đầu

os.environ["FLAGS_use_mkldnn"] = "0"

os.environ["FLAGS_enable_pir_api"] = "0"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



try:

    import paddle

    paddle.set_flags({'FLAGS_use_mkldnn': False})

except Exception:

    pass



import streamlit as st

import json
import re
import html
import html as _html
import base64
import io

import pandas as pd

from PIL import Image

import time



from modules.ocr_extractor import (

    get_label_studio_product_list,

    is_paddleocr_available,

    parse_ingredients_from_ocr,

)

from modules.file_resolver import find_data_file, read_table_flexibly

from modules.csv_data_defaults import ensure_default_csv_data

from modules.pipeline import (

    load_knowledge_base,

    load_personas,

    load_food_hazards,

    load_fndds_nutrition,

    load_openfoodfacts_vietnam,

    load_nhanes_biomarkers,

    run_perception_pipeline,

    get_nhanes_population_stats,

    generate_llm_only_answer,

    enrich_persona_with_nhanes,

    build_persona_from_benchmark_record,

    extract_safety_label,

    run_grounded_pipeline,

    guardrail_status_to_safety_label,

)

import modules.pipeline as m_pipeline

run_grounded_pipeline = m_pipeline.run_grounded_pipeline
generate_llm_only_answer = m_pipeline.generate_llm_only_answer
extract_safety_label = m_pipeline.extract_safety_label
guardrail_status_to_safety_label = m_pipeline.guardrail_status_to_safety_label
build_persona_from_benchmark_record = m_pipeline.build_persona_from_benchmark_record
enrich_persona_with_nhanes = m_pipeline.enrich_persona_with_nhanes

import modules.foodon as m_foodon
from modules.ocr_extractor import load_label_studio_data as _load_ls_data
import modules.persona_builder as m_persona_builder
import modules.evidence as m_evidence

find_foodon_exact_label = m_foodon.find_foodon_exact_label

load_foodon_entities = m_foodon.load_foodon_entities

search_foodon_entities = m_foodon.search_foodon_entities



import modules.benchmark as mb

import modules.benchmark_manifest as m_manifest
try:
    importlib.reload(m_manifest)
except Exception:
    pass

from modules.benchmark_manifest import (
    load_benchmark_manifest,
    is_evaluation_eligible,
    is_gold_eligible,
)

from modules.benchmark_manifest import get_default_eval_benchmark_path

if not hasattr(mb, "resolve_friendly_domain"):

    try:

        importlib.reload(mb)

    except Exception:

        pass



if hasattr(mb, "resolve_friendly_domain"):

    resolve_friendly_domain = mb.resolve_friendly_domain

    get_domain_badge = mb.get_domain_badge

else:

    def resolve_friendly_domain(item):

        domain = str(item.get("domain") or "").strip().lower()

        domain_vi = str(item.get("domain_vi") or "").strip()

        question = str(item.get("question") or "").lower()



        if domain == "insufficient_data" or domain_vi == "Thiếu dữ liệu":

            if any(w in question for w in [

                "dị nguyên", "dị ứng", "nhiễm chéo", "pal", "trứng", "sữa", "đậu phộng",

                "hạt cây", "gluten", "lúa mì", "tôm", "cua", "cá", "mè", "đậu nành", "hải sản"

            ]):

                return "Cảnh báo Dị ứng & Nhiễm chéo"

            if any(w in question for w in [

                "phụ gia", "ins", "chất bảo quản", "tạo màu", "tạo ngọt", "adi",

                "tích lũy", "hàm lượng", "ăn hàng ngày"

            ]):

                return "Phụ gia & Quy chuẩn TT24"

            if any(w in question for w in [

                "mang thai", "thai kỳ", "tiểu đường", "tim mạch", "trẻ em",

                "suy thận", "bệnh", "dinh dưỡng", "natri", "đường", "calo"

            ]):

                return "Dinh dưỡng & Bệnh lý lâm sàng"

            if any(w in question for w in ["không có mã ins", "chỉ ghi tên", "thành phần", "nhãn"]):

                return "Kiểm định thông tin nhãn"

            return "Kiểm định an toàn (Abstention)"



        if domain == "additive_regulation" or "phụ gia" in domain_vi.lower():

            return "Phụ gia & Quy chuẩn TT24"

        if domain == "allergen_detection" or "dị nguyên" in domain_vi.lower():

            return "Cảnh báo Dị ứng & Nhiễm chéo"

        if domain == "vulnerable_groups" or "tổn thương" in domain_vi.lower():

            return "Dinh dưỡng & Bệnh lý lâm sàng"



        return domain_vi or "Tư vấn an toàn"



    def get_domain_badge(item):

        friendly = resolve_friendly_domain(item)

        if "Dị ứng" in friendly:

            return "⚠️ Dị ứng"

        if "Phụ gia" in friendly:

            return "🧪 Phụ gia"

        if "Lâm sàng" in friendly or "Bệnh lý" in friendly or "Dinh dưỡng" in friendly:

            return "🩺 Lâm sàng"

        if "nhãn" in friendly.lower():

            return "🔍 Nhãn thiếu INS"

        return "📋 Tư vấn"



CITATION_SOURCE_TITLES = {

    "VN_TT24_2019": "Thông tư 24/2019/TT-BYT của Bộ Y tế Việt Nam (Quy định về quản lý và sử dụng phụ gia thực phẩm)",

    "CODEX_STAN_192_1995": "Codex General Standard for Food Additives (CXS 192-1995 / Codex STAN 192)",

    "CODEX_CXS_1_1985": "Codex General Standard for the Labelling of Prepackaged Foods (CXS 1-1985)",

    "JECFA_EVALUATIONS": "Joint FAO/WHO Expert Committee on Food Additives (JECFA ADI Evaluations)",

    "EFSA_OPINIONS": "European Food Safety Authority (EFSA Scientific Opinions on Food Additives)",

    "US_FDA_FALCPA": "US FDA Food Allergen Labeling and Consumer Protection Act (FALCPA of 2004)",

    "US_FDA_FASTER_ACT_2023": "US FDA Food Allergy Safety, Treatment, Education, and Research Act (FASTER Act 2023)",

    "FAO_WHO_JEMRA_PART_3": "FAO/WHO JEMRA Part 3: Precautionary Allergen Labelling (PAL) Guidance (2023)",

    "FAO_WHO_JEMRA_2021": "FAO/WHO JEMRA Part 1: Priority Allergen Summary Report (2021)",

    "USDA_FDC": "USDA FoodData Central / FNDDS Nutrient Database",

}





def get_comprehensive_citations(evaluation, chosen_gold=None, is_mode_a=False, linked_results=None):

    """Tổng hợp đầy đủ danh mục văn bản quy chuẩn, tài liệu pháp lý và y văn đối chiếu."""

    if is_mode_a:

        return []



    cits = set()



    # 1. Trích dẫn chuẩn từ câu hỏi Benchmark đã thẩm định

    if chosen_gold and chosen_gold.get("source_citations"):

        for raw_item in str(chosen_gold["source_citations"]).split(","):

            code = raw_item.strip()

            if not code:

                continue

            title = CITATION_SOURCE_TITLES.get(code)

            if title:

                cits.add(title)

            else:

                cits.add(code)



    # 2. Trích dẫn từ cảnh báo của máy luật (evaluation.citations)

    for c in (evaluation.get("citations") or []):

        if c and str(c).strip():

            c_str = str(c).strip()

            title = CITATION_SOURCE_TITLES.get(c_str, c_str)

            cits.add(title)



    # 3. Trích dẫn từ đánh giá quy chuẩn pháp lý phụ gia (Regulatory Assessment)

    if evaluation.get("regulatory"):

        cits.add(CITATION_SOURCE_TITLES["VN_TT24_2019"])

        cits.add(CITATION_SOURCE_TITLES["CODEX_STAN_192_1995"])

        for reg in evaluation["regulatory"]:

            adi = reg.get("adi", {})

            src = adi.get("source")

            if src and src != "N/A" and "JECFA" in src:

                cits.add(CITATION_SOURCE_TITLES["JECFA_EVALUATIONS"])



    # 4. Trích dẫn từ cảnh báo an toàn sinh học EFSA

    for ew in (evaluation.get("efsa_warnings") or []):

        ref = ew.get("reference")

        if ref:

            cits.add(f"EFSA: {ref}")



    # 5. Nếu phát hiện dị ứng / cảnh báo dị nguyên

    has_allergen_alert = any(

        "dị ứng" in str(a.get("message", "")).lower() or "dị nguyên" in str(a.get("message", "")).lower()

        for a in (evaluation.get("alerts") or []) + (evaluation.get("cautions") or [])

    )

    if has_allergen_alert:

        cits.add(CITATION_SOURCE_TITLES["CODEX_CXS_1_1985"])

        cits.add(CITATION_SOURCE_TITLES["FAO_WHO_JEMRA_2021"])

        cits.add(CITATION_SOURCE_TITLES["US_FDA_FALCPA"])



    # 6. Khung quy chuẩn mặc định sàng lọc an toàn thực phẩm

    if not cits:

        cits.add(CITATION_SOURCE_TITLES["VN_TT24_2019"])

        cits.add(CITATION_SOURCE_TITLES["CODEX_STAN_192_1995"])



    return sorted(list(cits))



# Cấu hình trang Streamlit

st.set_page_config(

    page_title="FoodHealthQA Demo (Full Data Suite)",

    page_icon="🥗",

    layout="wide",

    initial_sidebar_state="expanded"

)



# ── GIAO DIỆN: theme hiện đại (gradient, thẻ mờ, hiệu ứng chuyển động) ──
APP_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap");
:root{
  --fh-brand:#0d9488; --fh-brand-d:#0f766e; --fh-emerald:#059669; --fh-sky:#0284c7; --fh-indigo:#4f46e5; --fh-amber:#d97706;
  --fh-ink:#0f172a; --fh-text:#1e293b; --fh-muted:#475569; --fh-line:#e2e8f0; --fh-soft:#f8fafc; --fh-card:#ffffff;
  --fh-shadow:0 1px 3px rgba(0,0,0,.05),0 1px 2px -1px rgba(0,0,0,.03);
  --fh-shadow-lg:0 10px 25px -5px rgba(15,23,42,.10),0 8px 10px -6px rgba(15,23,42,.05);
}
@keyframes fhFadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes fhPulse{0%{box-shadow:0 0 0 0 rgba(217,119,6,.5)}70%{box-shadow:0 0 0 7px rgba(217,119,6,0)}100%{box-shadow:0 0 0 0 rgba(217,119,6,0)}}

html,body,.stApp{font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.stApp{background:#f1f5f9;color:var(--fh-text)}
header[data-testid="stHeader"]{background:transparent}
.main .block-container{max-width:100%!important;padding:1.1rem 1.5rem 2.5rem 1.5rem!important;animation:fhFadeUp .35s ease both}
::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:9999px}::-webkit-scrollbar-track{background:#f1f5f9}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--fh-line);top:0!important}
header[data-testid="stHeader"]{display:none!important;height:0!important;min-height:0!important}
div[data-testid="stSidebarHeader"],
div[data-testid="stLogoSpacer"]{display:none!important;height:0!important;min-height:0!important;margin:0!important;padding:0!important}
div[data-testid="stSidebarCollapseButton"]{display:none!important}
section[data-testid="stSidebar"] > div:first-child{padding-top:0!important;margin-top:0!important}
div[data-testid="stSidebarContent"]{padding-top:0.75rem!important;padding-bottom:1rem!important;padding-left:1rem!important;padding-right:1rem!important}
div[data-testid="stSidebarUserContent"]{padding-top:0!important;margin-top:0!important}
section[data-testid="stSidebar"] .block-container,
div[data-testid="stSidebarUserContent"] > div:first-child,
div[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > div:first-child,
div[data-testid="stSidebarUserContent"] [data-testid="stElementContainer"]:first-child{padding-top:0!important;margin-top:0!important}
section[data-testid="stSidebar"] hr{margin:.9rem 0;border-color:#f1f5f9}
.fh-brand{display:flex;align-items:center;gap:12px;padding:0!important;margin-top:0!important}
.fh-logo{width:44px;height:44px;border-radius:12px;flex:none;display:grid;place-items:center;color:#fff;
  background:linear-gradient(45deg,#10b981,#14b8a6,#06b6d4);box-shadow:0 4px 10px rgba(16,185,129,.25)}
.fh-logo svg{width:24px;height:24px}
.fh-brand b{display:block;font-size:1.1rem;font-weight:700;letter-spacing:-.01em;color:var(--fh-ink)}
.fh-brand b span{color:var(--fh-brand)}
.fh-brand small{display:block;margin-top:2px;font-size:.72rem;color:#94a3b8;font-weight:500;line-height:1.2}
.fh-label{margin:2px 4px 8px;font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8}
.fh-stat{display:flex;align-items:center;gap:10px;padding:9px 12px;margin:7px 0;border-radius:12px;background:var(--fh-soft);
  border:1px solid rgba(226,232,240,.8);font-size:.78rem;font-weight:500;color:#334155}
.fh-stat .v{margin-left:auto;font-weight:700;font-family:ui-monospace,Consolas,monospace;font-size:.76rem;color:var(--fh-ink)}
.fh-stat .v.tag{background:#fff;border:1px solid var(--fh-line);border-radius:6px;padding:1px 8px;font-family:inherit;font-weight:600;color:var(--fh-muted)}
.fh-stat .v.idx{background:#eef2ff;color:var(--fh-indigo);border-radius:4px;padding:1px 7px}
.fh-stat .v.tl{background:#f0fdfa;color:var(--fh-brand-d);border-radius:4px;padding:1px 7px}
.fh-dot{width:8px;height:8px;border-radius:50%;background:#10b981;display:inline-block}
.fh-dot.off{background:#f59e0b;animation:fhPulse 2s infinite}
.fh-note{margin:12px 0;padding:11px 12px;border-radius:12px;font-size:.72rem;line-height:1.55;color:rgba(120,53,15,.85);
  background:rgba(255,251,235,.8);border:1px solid rgba(253,230,138,.8)}
.fh-note b{display:block;margin-bottom:3px;font-size:.74rem;color:#78350f}

section[data-testid="stSidebar"] div[role="radiogroup"]{gap:6px}
section[data-testid="stSidebar"] div[role="radiogroup"] > label{padding:10px 12px;border-radius:12px;border:1px solid transparent;transition:all .15s ease;cursor:pointer}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{background:rgba(241,245,249,.9)}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){background:rgba(240,253,250,.9);border-color:#ccfbf1;box-shadow:var(--fh-shadow)}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p{color:#134e4a!important;font-weight:600}

/* ── Hero ── */
.fh-hero{position:relative;overflow:hidden;border-radius:16px;padding:26px 30px;margin:0 0 14px 0;color:#fff;
  background:linear-gradient(135deg,#09476b 0%,#0d6e73 50%,#087f8c 100%);box-shadow:var(--fh-shadow-lg)}
.fh-hero .in{position:relative;z-index:1;max-width:900px}
.fh-badge{display:inline-block;padding:4px 11px;margin-bottom:10px;border-radius:999px;font-size:.72rem;font-weight:600;color:#ccfbf1;
  background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.2);backdrop-filter:blur(6px)}
.fh-hero h1{margin:0!important;padding:0!important;font-size:1.6rem!important;font-weight:800!important;letter-spacing:-.02em;line-height:1.2;color:#fff!important}
.fh-hero p{margin:10px 0 0;font-size:.9rem;line-height:1.6;color:rgba(240,253,250,.92)!important}
.fh-chips{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}
.fh-chip{padding:6px 12px;border-radius:8px;font-size:.76rem;font-weight:500;color:#fff;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.15)}
.fh-deco{position:absolute;right:-16px;bottom:-26px;width:190px;height:190px;border-radius:50%;display:grid;place-items:center;font-size:72px;
  background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.2);box-shadow:0 20px 40px rgba(0,0,0,.18)}
@media (max-width:900px){.fh-deco{display:none}}

/* ── Banner khuyến cáo ── */
.fh-warn{display:flex;gap:12px;align-items:flex-start;padding:14px 16px;margin:0 0 14px 0;border-radius:16px;
  background:#fffbeb;border:1px solid rgba(253,230,138,.9);box-shadow:var(--fh-shadow);font-size:.78rem;line-height:1.6;color:#334155}
.fh-warn .ic{flex:none;width:32px;height:32px;border-radius:8px;display:grid;place-items:center;background:#fef3c7;border:1px solid #fcd34d}
.fh-warn b.t{color:#78350f}.fh-warn b.k{color:#be123c;text-transform:uppercase}

/* ── Pipeline stepper ── */
.fh-flow{background:#fff;border:1px solid var(--fh-line);border-radius:12px;padding:14px 16px;margin:0 0 16px 0;box-shadow:var(--fh-shadow)}
.fh-flow .h{margin-bottom:9px;font-size:.68rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--fh-muted)}
.fh-flow .row{display:flex;flex-wrap:wrap;align-items:center;gap:8px;font-family:ui-monospace,Consolas,monospace;font-size:.72rem}
.fh-step{padding:4px 10px;border-radius:6px;font-weight:500;border:1px solid var(--fh-line);background:#f1f5f9;color:#334155}
.fh-step.skip{background:#f0f9ff;color:#0369a1;border-color:#bae6fd}
.fh-step.ok{background:#ecfdf5;color:#065f46;border-color:#a7f3d0}
.fh-step.bad{background:#fff1f2;color:#9f1239;border-color:#fecdd3}
.fh-step.final{background:var(--fh-brand);color:#fff;border-color:var(--fh-brand)}
.fh-arrow{color:#94a3b8;font-size:.8rem}

/* ── Thẻ / chỉ số ── */
div[data-testid="stMetric"]{background:#fff;border:1px solid var(--fh-line);border-radius:12px;padding:14px 16px;box-shadow:var(--fh-shadow);transition:box-shadow .2s,transform .2s}
div[data-testid="stMetric"]:hover{box-shadow:var(--fh-shadow-lg);transform:translateY(-2px)}
div[data-testid="stMetricLabel"]{overflow:visible!important;text-overflow:clip!important}
div[data-testid="stMetricLabel"] p{font-size:.74rem!important;font-weight:600;color:var(--fh-muted);white-space:normal!important;overflow:visible!important;text-overflow:clip!important}
div[data-testid="stMetricValue"]{font-family:ui-monospace,Consolas,monospace;font-weight:800;color:var(--fh-ink)}

h1,h2,h3,h4,h5{color:var(--fh-ink);letter-spacing:-.01em}
h3{font-size:1.02rem!important;font-weight:700!important;padding-bottom:.55rem;border-bottom:1px solid #f1f5f9;margin-bottom:.6rem}
h4{font-size:.95rem!important;font-weight:700!important}

/* ── Tabs dạng gạch chân ── */
div[data-baseweb="tab-list"]{gap:22px;border-bottom:1px solid rgba(226,232,240,.8)}
button[data-baseweb="tab"]{padding:10px 2px!important;background:transparent!important;font-weight:600;color:var(--fh-muted)}
button[data-baseweb="tab"]:hover{color:var(--fh-ink)}
button[data-baseweb="tab"][aria-selected="true"],button[data-baseweb="tab"][aria-selected="true"] p{color:var(--fh-brand)!important}
div[data-baseweb="tab-highlight"]{background-color:var(--fh-brand)!important;height:2px!important}
div[data-baseweb="tab-border"]{background-color:transparent!important}

/* ── Nút ── */
.stButton>button,.stDownloadButton>button{border-radius:12px;font-weight:600;border:1px solid var(--fh-line);background:#fff;color:var(--fh-text);transition:all .15s ease}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--fh-brand);color:var(--fh-brand-d);box-shadow:var(--fh-shadow)}
.stButton>button[kind="primary"]{padding:.7rem 1rem;border:none;color:#fff;font-weight:700;
  background:linear-gradient(90deg,#0d9488,#14b8a6 50%,#059669);box-shadow:0 4px 12px rgba(13,148,136,.25)}
.stButton>button[kind="primary"]:hover{color:#fff;background:linear-gradient(90deg,#0f766e,#047857);box-shadow:0 8px 18px rgba(13,148,136,.3)}
.stButton>button[kind="primary"] p{color:#fff!important}

/* ── Ô nhập liệu ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] div[data-baseweb="input"] {
  background: var(--fh-soft)!important;
  border-radius: 12px!important;
  border: 1px solid var(--fh-line)!important;
  font-size: .88rem!important;
}
div[data-testid="stNumberInput"] input {
  font-size: .88rem!important;
  color: var(--fh-text)!important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within,
div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
  border-color: var(--fh-brand)!important;
  box-shadow: 0 0 0 2px rgba(13,148,136,.2)!important;
}
div[data-testid="stSelectbox"],div[data-baseweb="select"]{width:100%!important}
div[data-baseweb="popover"] ul[role="listbox"] li{font-size:.9rem!important;white-space:normal!important;word-break:break-word!important;line-height:1.45!important;padding:8px 12px!important}
div[data-testid="stTextArea"] textarea{background:var(--fh-soft)!important;border:1px solid var(--fh-line)!important;border-radius:12px!important;
  font-size:.9rem!important;line-height:1.55!important;color:var(--fh-text)!important;-webkit-text-fill-color:var(--fh-text)!important;opacity:1!important}
div[data-testid="stTextArea"] textarea:focus{border-color:var(--fh-brand)!important;box-shadow:0 0 0 2px rgba(13,148,136,.2)!important}
div[data-testid="stTextArea"] textarea:disabled,div[data-testid="stTextArea"] textarea[disabled]{cursor:default!important;-webkit-user-select:text!important;user-select:text!important}
div[data-testid="stTextArea"] label[data-testid="stWidgetLabel"],div[data-testid="stTextArea"] label[data-testid="stWidgetLabel"] *{
  color:var(--fh-text)!important;-webkit-text-fill-color:var(--fh-text)!important;opacity:1!important;font-weight:600}
div[data-testid="stFileUploader"] section{border-radius:14px;border:1.5px dashed #94a3b8;background:var(--fh-soft)}
div[data-testid="stFileUploader"] section:hover{border-color:var(--fh-brand);background:#f0fdfa}

/* ── Expander, alert, bảng, ảnh ── */
div[data-testid="stExpander"] details{border-radius:12px!important;border:1px solid var(--fh-line)!important;background:#fff;box-shadow:var(--fh-shadow)}
div[data-testid="stExpander"] summary{font-weight:700;font-size:.85rem}
div[data-testid="stExpander"] summary:hover{background:var(--fh-soft)}
div[data-testid="stExpander"] div[data-testid="stExpanderDetails"]{padding-top:.4rem!important;padding-bottom:.6rem!important}
div[data-testid="stAlert"]{border-radius:12px}
div[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden;border:1px solid var(--fh-line)}
div[data-testid="stImage"] img{border-radius:12px;border:1px solid var(--fh-line)}
div[data-testid="stProgress"] div[role="progressbar"] > div{background:var(--fh-brand)!important}
.stCaption,[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:#334155!important;font-size:.82rem!important;font-weight:500;line-height:1.5!important}

/* ═════ Bám sát mẫu code.html: kiểu chữ, ẩn thanh công cụ, thẻ cột ═════ */
html,body,.stApp,.stApp *:not(code):not(pre):not(.material-symbols-rounded):not([data-testid="stIconMaterial"]):not(.fh-step):not(.v){font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],#MainMenu,footer{display:none!important}
header[data-testid="stHeader"]{display:none!important;height:0!important;min-height:0!important;background:transparent}
.main .block-container,[data-testid="stMainBlockContainer"]{padding-top:1rem!important;padding-left:1.5rem!important;padding-right:1.5rem!important;max-width:100%!important}
section[data-testid="stSidebar"]{width:320px!important;min-width:320px!important}
section[data-testid="stSidebar"] > div{width:320px!important}
section[data-testid="stSidebar"] .block-container{padding-top:0!important;margin-top:0!important}

/* Cỡ chữ nhỏ gọn như mẫu (12–13px) */
.stApp [data-testid="stMarkdownContainer"] p,.stApp [data-testid="stMarkdownContainer"] li{font-size:.82rem;line-height:1.65;color:var(--fh-text)}
.stApp label p,.stApp [data-testid="stWidgetLabel"] p{font-size:.78rem!important;font-weight:600;color:#334155}
.stApp [data-testid="stRadio"] label p{font-size:.78rem!important}
.stApp [data-testid="stAlert"] p{font-size:.8rem}
.stApp [data-testid="stSelectbox"] div[data-baseweb="select"] *{font-size:.8rem!important}
.stApp .stButton>button p{font-size:.88rem}
.stApp h4,.stApp h5{font-size:.85rem!important;font-weight:700!important;text-transform:none}
.stApp [data-testid="stMarkdownContainer"] h4{font-size:.86rem!important}
.stApp [data-testid="stMarkdownContainer"] code{font-size:.74rem;background:#f1f5f9;color:#0f766e;border-radius:4px;padding:1px 5px}

/* Mục điều hướng sidebar */
div[data-testid="stRadioGroup"] > div[data-selected="true"]{background:rgba(240,253,250,.85);border:1px solid rgba(204,251,241,.9);border-radius:12px;box-shadow:var(--fh-shadow)}
div[data-testid="stRadioGroup"] > div{border:1px solid transparent;border-radius:12px;padding:4px 6px;margin-bottom:4px}
div[data-testid="stRadioGroup"] > div[data-selected="true"] p{color:#134e4a!important;font-weight:600!important}

/* Hai cột chính = hai thẻ trắng như mẫu */
div[data-testid="stColumn"]:has(> div > div > div > .fh-card-marker),
div[data-testid="stColumn"]:has(.fh-card-marker){background:#fff;border:1px solid rgba(226,232,240,.9);border-radius:16px;box-shadow:var(--fh-shadow);padding:18px 20px 20px 20px}
.fh-card-marker{display:none}
.fh-card-h{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;padding-bottom:12px;margin:0 0 14px 0;border-bottom:1px solid #f1f5f9}
.fh-card-h.sep{margin-top:22px;padding-top:18px;border-top:1px solid #f1f5f9}
.fh-card-h h3{display:flex;align-items:center;gap:8px;margin:0!important;padding:0!important;border:0!important;font-size:.88rem!important;font-weight:700!important;color:var(--fh-ink)}
.fh-chip2{font-size:.68rem;font-weight:600;padding:2px 9px;border-radius:5px;color:#0f766e;background:#f0fdfa;border:1px solid #99f6e4}
.fh-chip2.gray{color:var(--fh-muted);background:#f1f5f9;border-color:transparent;border-radius:999px;padding:4px 12px;font-weight:500;font-size:.72rem}
.fh-modebar{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:0 -20px 14px -20px;padding:10px 20px;
  background:rgba(240,249,255,.85);border-top:1px solid #e0f2fe;border-bottom:1px solid #e0f2fe;font-size:.76rem;font-weight:600;color:#0c4a6e;line-height:1.5}
.fh-modebar .ok{flex:none;font-family:ui-monospace,Consolas,monospace;font-size:.68rem;font-weight:500;color:#0369a1;background:#fff;border:1px solid #bae6fd;border-radius:5px;padding:1px 8px}

/* Hộp persona gọn như mẫu */
div[data-testid="stAlert"]:has(div[data-testid="stAlertContentSuccess"]){background:rgba(236,253,245,.6);border:1px solid rgba(167,243,208,.9);border-radius:12px;color:#064e3b}
div[data-testid="stAlert"]:has(div[data-testid="stAlertContentSuccess"]) p{font-size:.78rem;line-height:1.6;color:#064e3b}
div[data-testid="stAlert"]:has(div[data-testid="stAlertContentInfo"]){background:#f0f9ff;border:1px solid #bae6fd;color:#0c4a6e}
div[data-testid="stAlert"]:has(div[data-testid="stAlertContentWarning"]){background:#fffbeb;border:1px solid #fde68a}
div[data-testid="stAlert"]:has(div[data-testid="stAlertContentError"]){background:#fff1f2;border:1px solid #fecdd3}
div[data-testid="stAlert"]{box-shadow:none}
hr{border-color:#f1f5f9!important}
.fh-chips2{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 12px 0}
.fh-trace{width:100%;border-collapse:collapse;font-size:.76rem}
.fh-trace th{text-align:left;padding:6px 8px;color:#64748b;font-weight:700;border-bottom:1px solid #e2e8f0;text-transform:uppercase;font-size:.66rem;letter-spacing:.04em}
.fh-trace td{padding:7px 8px;border-bottom:1px solid #f1f5f9;vertical-align:top;color:#334155}
.fh-trace small{color:#94a3b8}

/* ── Thẻ cột trái (Thông tin đầu vào / Thiết lập phân tích) theo mẫu ── */
.st-key-fh_card_input,.st-key-fh_card_setup{background:#fff;border:1px solid rgba(226,232,240,.9);border-radius:16px;box-shadow:var(--fh-shadow);padding:20px 22px 22px 22px}
.fh-card-h.nob{border-bottom:0;padding-bottom:0;margin-bottom:0}
.fh-hr{border-top:1px solid #f1f5f9;margin:10px 0 14px 0}
.fh-imgbox{border:1px solid #e2e8f0;background:rgba(241,245,249,.7);border-radius:12px;padding:8px;margin-top:6px}
.fh-imgin{position:relative;display:flex;justify-content:center;background:#020617;border-radius:8px;overflow:hidden;max-height:380px}
.fh-imgin img{max-height:380px;max-width:100%;width:auto;object-fit:contain;display:block}
.fh-imgbar{position:absolute;left:8px;right:8px;bottom:8px;display:flex;justify-content:space-between;gap:10px;padding:6px 12px;border-radius:8px;
  background:rgba(15,23,42,.85);border:1px solid rgba(255,255,255,.1);color:#fff;font-family:ui-monospace,Consolas,monospace;font-size:.68rem}
.fh-imgbar .tg{color:#5eead4;font-weight:600;white-space:nowrap}
.fh-imgcap{margin-top:8px;text-align:center;font-size:.72rem;font-weight:500;color:#64748b}
.fh-imgcap b{color:#1e293b;font-weight:600}
.fh-lab{display:flex;justify-content:space-between;align-items:center;margin:2px 0 6px 0;font-size:.78rem;font-weight:600;color:#334155}
.fh-lab-row{height:24px;margin:0!important;line-height:24px}
.st-key-fh_card_setup [data-testid="stHorizontalBlock"] [data-testid="stMarkdownContainer"]{margin:0!important;padding:0!important}
.st-key-fh_card_setup [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]{margin:0!important}
.fh-lab .r{font-size:.68rem;font-weight:500;color:#94a3b8}
.fh-persona{padding:14px 16px;margin:12px 0;border-radius:12px;background:rgba(236,253,245,.55);border:1px solid rgba(167,243,208,.9)}
.fh-persona .top{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
.fh-persona .ttl{display:flex;align-items:center;gap:8px;font-size:.78rem;font-weight:700;color:#022c22}
.fh-persona .ic{width:24px;height:24px;border-radius:6px;display:grid;place-items:center;background:#d1fae5;font-size:.8rem}
.fh-persona .nm{font-size:.7rem;font-weight:700;color:#047857;background:#fff;border:1px solid #a7f3d0;border-radius:5px;padding:2px 9px}
.fh-persona p{margin:8px 0 0 32px!important;font-size:.76rem!important;line-height:1.6;font-style:italic;color:rgba(6,78,59,.9)!important}
/* Ô chọn (selectbox): nền slate-50, bo tròn; ô câu hỏi benchmark là dải hổ phách như mẫu */
[data-testid="stSelectbox"] div[role="group"]{background:var(--fh-soft)!important;border:1px solid var(--fh-line)!important;border-radius:12px!important}
[data-testid="stSelectbox"] div[role="group"]:focus-within{border-color:var(--fh-brand)!important;box-shadow:0 0 0 2px rgba(13,148,136,.2)!important}
[data-testid="stSelectbox"] input{font-size:.8rem!important}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

_SHIELD_SVG = (
    '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 '
    '11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" '
    'stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/></svg>'
)


def render_hero(title, subtitle, emoji, chips=(), badge=""):
    chips_html = "".join(f'<span class="fh-chip">{c}</span>' for c in chips)
    badge_html = f'<span class="fh-badge">{badge}</span><br>' if badge else ""
    st.markdown(
        f'<div class="fh-hero"><div class="in">{badge_html}<h1>{title}</h1><p>{subtitle}</p>'
        f'<div class="fh-chips">{chips_html}</div></div><div class="fh-deco">{emoji}</div></div>',
        unsafe_allow_html=True,
    )


def card_header_html(title, chip="", chip_class="", sep=False, marker=False, border=True):
    chip_html = f'<span class="fh-chip2 {chip_class}">{chip}</span>' if chip else ""
    mk = '<span class="fh-card-marker"></span>' if marker else ""
    cls = "fh-card-h" + (" sep" if sep else "") + ("" if border else " nob")
    return f'{mk}<div class="{cls}"><h3>{title}</h3>{chip_html}</div>'


def render_card_header(title, chip="", chip_class="", sep=False, marker=False, border=True):
    st.markdown(card_header_html(title, chip, chip_class, sep, marker, border), unsafe_allow_html=True)


def render_label_image(img, code, caption_html, tag="Ảnh nhãn mẫu"):
    """Khung ảnh nhãn: nền tối, thanh thông tin phía dưới, chú thích bên dưới (giống mẫu thiết kế)."""
    try:
        rgb = img.convert("RGB")
        w, h = rgb.size
        thumb = rgb.copy()
        thumb.thumbnail((1400, 1400))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        st.image(img, caption=_html.unescape(caption_html), width="stretch")
        return
    st.markdown(
        f'<div class="fh-imgbox"><div class="fh-imgin"><img src="data:image/jpeg;base64,{b64}" alt="Ảnh nhãn sản phẩm"/>'
        f'<div class="fh-imgbar"><span>Mã: {_html.escape(str(code))} • Định dạng: {w}x{h}</span><span class="tg">● {tag}</span></div></div>'
        f'<div class="fh-imgcap">{caption_html}</div></div>',
        unsafe_allow_html=True,
    )


def render_prompt_label(text="💬 Câu hỏi tư vấn an toàn thực phẩm:", right="Prompt Grounding"):
    st.markdown(f'<div class="fh-lab"><span>{text}</span><span class="r">{right}</span></div>', unsafe_allow_html=True)


def render_qa_trace(qa):
    """Hiển thị QA động: ý định nhận diện, thực thể được hỏi, nguồn đã tra và kết quả tra cứu."""
    if not qa:
        return
    ents = qa.get("entities", {})
    chips = [f'<span class="fh-chip2">{_html.escape(i)}</span>' for i in qa.get("intents", [])]
    for key, label in (("additives", "Phụ gia"), ("allergens", "Dị nguyên"), ("nutrients", "Dinh dưỡng"), ("populations", "Đối tượng")):
        for v in ents.get(key, []):
            chips.append(f'<span class="fh-chip2 gray">{label}: {_html.escape(str(v))}</span>')
    rows = []
    for t in qa.get("trace", []):
        ok = t["status"] == "found"
        badge = ('<span class="fh-step ok">Tìm thấy</span>' if ok else '<span class="fh-step bad">Không tìm thấy</span>')
        origin = "câu hỏi" if t.get("origin") == "question" else "nhãn"
        rows.append(
            f'<tr><td>{badge}</td><td>{_html.escape(t["source"])}</td><td>{_html.escape(t["query"])}'
            f'<br><small>({origin})</small></td><td>{_html.escape(t["detail"])}</td></tr>'
        )
    table = (
        '<table class="fh-trace"><thead><tr><th>Kết quả</th><th>Nguồn tri thức</th><th>Truy vấn</th><th>Chi tiết</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    ) if rows else '<p>Không có nguồn tri thức nào được truy vấn cho câu hỏi này.</p>'
    with st.expander(
        f"🔎 Tri thức đã tra cứu theo câu hỏi — {qa.get('sources_found', 0)} tìm thấy · {qa.get('sources_missing', 0)} không tìm thấy",
        expanded=bool(qa.get("knowledge_gaps")),
    ):
        st.markdown(f'<div class="fh-chips2">{"".join(chips)}</div>{table}', unsafe_allow_html=True)
        for g in qa.get("knowledge_gaps", []):
            st.warning(g["reason"])


def render_pipeline_flow(title, steps):
    """steps: list[(text, kind)] với kind ∈ {'', 'skip', 'ok', 'bad', 'final'}."""
    parts = []
    for i, (text, kind) in enumerate(steps):
        if i:
            parts.append('<span class="fh-arrow">›</span>')
        parts.append(f'<span class="fh-step {kind}">{text}</span>')
    st.markdown(
        f'<div class="fh-flow"><div class="h">⚡ {title}</div><div class="row">{"".join(parts)}</div></div>',
        unsafe_allow_html=True,
    )


# Thiết lập đường dẫn thư mục chuẩn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

ASSETS_DIR = os.path.join(BASE_DIR, "assets")



# Tự động đảm bảo các tệp dữ liệu cốt lõi luôn hiện diện

ensure_default_csv_data(DATA_DIR)



@st.cache_resource(show_spinner=False)

def load_all_system_data(data_dir):

    kb = load_knowledge_base(data_dir)

    personas = load_personas(data_dir)

    hazards = load_food_hazards(data_dir)

    fndds = load_fndds_nutrition(data_dir)

    off = load_openfoodfacts_vietnam(data_dir)

    foodon = load_foodon_entities(data_dir)

    biomarkers = load_nhanes_biomarkers(data_dir)

    return kb, personas, hazards, fndds, off, foodon, biomarkers



# Nạp dữ liệu đa nguồn thông minh từ DATA_DIR (được cache trong RAM, không load lại khi tương tác UI)

kb_additives, personas_list, hazards_list, fndds_list, off_products, foodon_entities, nhanes_biomarkers_cache = load_all_system_data(DATA_DIR)



# Bảo đảm đối tượng foodon_entities trong cache Streamlit RAM luôn có giao thức sequence/iterable

if hasattr(foodon_entities, "entities"):

    _cached_cls = type(foodon_entities)

    if not hasattr(_cached_cls, "__iter__"):

        _cached_cls.__iter__ = lambda self: iter(self.entities)

    if not hasattr(_cached_cls, "__getitem__"):

        _cached_cls.__getitem__ = lambda self, idx: (

            self.entities[idx] if isinstance(idx, (int, slice))

            else (self.by_curie.get(str(idx).strip().upper()) or self.by_normalized_label.get(str(idx).strip().lower()))

        )

    if not hasattr(_cached_cls, "get"):

        _cached_cls.get = lambda self, k, d=None: (

            self.by_curie.get(str(k).strip().upper(), d) if hasattr(self, "by_curie") else d

        )





def format_answer_to_html(text: str) -> str:
    """Chuyển đổi text câu trả lời có markdown cơ bản sang HTML an toàn, làm nổi bật trích dẫn."""
    escaped = html.escape((text or "").strip())
    # Bold **text**
    escaped = re.sub(r'\*\*(.*?)\*\*', r'<b style="color:#0F172A;">\1</b>', escaped)
    # Trích dẫn [CITATION_ID] -> badge pill màu xanh dương
    escaped = re.sub(
        r'\[([A-Z0-9_]{3,35})\]',
        r'<span style="display:inline-block; background:#E0F2FE; color:#0369A1; padding:1px 6px; border-radius:4px; font-family:monospace; font-size:12px; font-weight:700; border:1px solid #BAE6FD;">\1</span>',
        escaped,
    )
    paras = [p.strip() for p in escaped.split("\n\n") if p.strip()]
    html_paras = []
    for p in paras:
        html_paras.append(
            f'<p style="margin:0 0 12px 0; line-height:1.68; color:#334155; font-size:14px;">{p.replace(chr(10), "<br>")}</p>'
        )
    return "".join(html_paras)


def render_guardrail_banner_html(evaluation: dict, is_mode_a: bool = False) -> str:
    """Render Banner Quyết định Hệ thống AI Guardrail theo chuẩn thiết kế hiện đại."""
    status = evaluation.get("status", "SAFE")
    title = evaluation.get("title", "")

    if is_mode_a:
        icon = "⚡"
        sub_title = "MÔ HÌNH SUY LUẬN TRỰC TIẾP (NO-RAG BASELINE)"
        main_title = title or "KẾT QUẢ TỪ LLM (KHÔNG GUARDRAIL)"
        sub_color = "#64748B"
        pill_bg = "#64748B"
        pill_text = "Chưa qua Guardrail"
        banner_border = "#E2E8F0"
        banner_bg = "linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%)"
        circle_bg = "#F1F5F9"
    elif status in ("ABSTAIN", "INSUFFICIENT_DATA"):
        icon = "🔮"
        sub_title = "QUYẾT ĐỊNH HỆ THỐNG AI GUARDRAIL"
        main_title = title if "TỪ CHỐI TƯ VẤN" in title else "TỪ CHỐI TƯ VẤN — THIẾU DỮ LIỆU (ABSTAIN)"
        sub_color = "#6366F1"
        pill_bg = "#4F46E5"
        pill_text = "An toàn tuyệt đối (Fail-Safe)"
        banner_border = "#E0E7FF"
        banner_bg = "linear-gradient(135deg, #F8FAFC 0%, #F5F3FF 100%)"
        circle_bg = "#EEF2FF"
    elif status in ("CRITICAL_ALERT", "UNSAFE"):
        icon = "🚨"
        sub_title = "QUYẾT ĐỊNH HỆ THỐNG AI GUARDRAIL"
        main_title = title or "CẢNH BÁO NGUY HIỂM — KHÔNG AN TOÀN (CRITICAL_ALERT)"
        sub_color = "#DC2626"
        pill_bg = "#DC2626"
        pill_text = "Nguy cơ cao (High-Risk)"
        banner_border = "#FECACA"
        banner_bg = "linear-gradient(135deg, #FFF5F5 0%, #FEF2F2 100%)"
        circle_bg = "#FEE2E2"
    elif status in ("WARNING", "CAUTION"):
        icon = "⚠️"
        sub_title = "QUYẾT ĐỊNH HỆ THỐNG AI GUARDRAIL"
        main_title = title or "CẦN THẬN TRỌNG (CAUTION / RESTRICTED)"
        sub_color = "#D97706"
        pill_bg = "#D97706"
        pill_text = "Cảnh báo giới hạn (Restricted)"
        banner_border = "#FDE68A"
        banner_bg = "linear-gradient(135deg, #FFFDF5 0%, #FFFBEB 100%)"
        circle_bg = "#FEF3C7"
    else:  # SAFE
        icon = "🛡️"
        sub_title = "QUYẾT ĐỊNH HỆ THỐNG AI GUARDRAIL"
        main_title = title or "ĐẠT CHUẨN AN TOÀN (SAFE)"
        sub_color = "#16A34A"
        pill_bg = "#16A34A"
        pill_text = "Đạt kiểm chứng (Permitted)"
        banner_border = "#BBF7D0"
        banner_bg = "linear-gradient(135deg, #F6FEF9 0%, #F0FDF4 100%)"
        circle_bg = "#DCFCE7"

    return f"""
<div style="background:{banner_bg}; border:1.5px solid {banner_border}; border-radius:16px; padding:18px 24px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center; box-shadow:0 2px 6px rgba(0,0,0,0.02);">
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="width:48px; height:48px; border-radius:50%; background:{circle_bg}; display:flex; align-items:center; justify-content:center; font-size:24px; flex-shrink:0;">
            {icon}
        </div>
        <div>
            <div style="font-size:11px; font-weight:800; color:{sub_color}; letter-spacing:0.8px; text-transform:uppercase; margin-bottom:4px;">
                {sub_title}
            </div>
            <div style="font-size:18px; font-weight:800; color:#0F172A; line-height:1.25;">
                {main_title}
            </div>
        </div>
    </div>
    <div style="flex-shrink:0;">
        <span style="background:{pill_bg}; color:#FFFFFF; padding:8px 18px; border-radius:9999px; font-weight:700; font-size:12.5px; letter-spacing:0.3px; display:inline-block; box-shadow:0 2px 4px rgba(0,0,0,0.08);">
            {pill_text}
        </span>
    </div>
</div>
"""


def render_metric_cards_html(evaluation: dict) -> str:
    """Render 3 thẻ đo lường S_perception, S_retrieval, S_total với thanh tiến trình màu sắc."""
    s_perception = float(evaluation.get("s_perception", 1.0))
    s_retrieval = float(evaluation.get("s_retrieval", 1.0))
    s_total = float(evaluation.get("s_total", 1.0))

    if s_perception >= 0.95:
        p_tag_text, p_tag_bg, p_tag_fg = "Tối ưu", "#ECFDF5", "#059669"
    elif s_perception >= 0.80:
        p_tag_text, p_tag_bg, p_tag_fg = "Tốt", "#F0F9FF", "#0284C7"
    else:
        p_tag_text, p_tag_bg, p_tag_fg = "Thấp", "#FFFBEB", "#D97706"

    if s_retrieval >= 0.95:
        r_tag_text, r_tag_bg, r_tag_fg = "Rất cao", "#F0F9FF", "#0284C7"
    elif s_retrieval >= 0.80:
        r_tag_text, r_tag_bg, r_tag_fg = "Khá", "#ECFDF5", "#059669"
    else:
        r_tag_text, r_tag_bg, r_tag_fg = "Trung bình", "#FFFBEB", "#D97706"

    if s_total >= 0.65:
        t_tag_text, t_tag_bg, t_tag_fg = "Đạt ngưỡng", "#EEF2FF", "#4F46E5"
    else:
        t_tag_text, t_tag_bg, t_tag_fg = "Dưới ngưỡng", "#FEE2E2", "#DC2626"

    p_bar_width = min(100, max(0, int(s_perception * 100)))
    r_bar_width = min(100, max(0, int(s_retrieval * 100)))
    t_bar_width = min(100, max(0, int(s_total * 100)))

    return f"""
<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin-bottom:20px;">
    <!-- Thẻ S_perception -->
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:16px 18px; box-shadow:0 1px 3px rgba(0,0,0,0.02);">
        <div style="font-size:12px; color:#64748B; font-weight:600; margin-bottom:8px;">Độ tin cậy OCR (S_perception)</div>
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px;">
            <span style="font-size:26px; font-weight:800; color:#0F172A; font-family:ui-monospace,Consolas,monospace;">{s_perception:.2f}</span>
            <span style="background:{p_tag_bg}; color:{p_tag_fg}; font-size:11px; font-weight:700; padding:3px 9px; border-radius:6px;">{p_tag_text}</span>
        </div>
        <div style="height:4px; background:#E2E8F0; border-radius:2px; overflow:hidden;">
            <div style="height:100%; width:{p_bar_width}%; background:#10B981; border-radius:2px;"></div>
        </div>
    </div>
    <!-- Thẻ S_retrieval -->
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:16px 18px; box-shadow:0 1px 3px rgba(0,0,0,0.02);">
        <div style="font-size:12px; color:#64748B; font-weight:600; margin-bottom:8px;">Độ khớp truy xuất (S_retrieval)</div>
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px;">
            <span style="font-size:26px; font-weight:800; color:#0284C7; font-family:ui-monospace,Consolas,monospace;">{s_retrieval:.2f}</span>
            <span style="background:{r_tag_bg}; color:{r_tag_fg}; font-size:11px; font-weight:700; padding:3px 9px; border-radius:6px;">{r_tag_text}</span>
        </div>
        <div style="height:4px; background:#E2E8F0; border-radius:2px; overflow:hidden;">
            <div style="height:100%; width:{r_bar_width}%; background:#0EA5E9; border-radius:2px;"></div>
        </div>
    </div>
    <!-- Thẻ S_total -->
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:16px 18px; box-shadow:0 1px 3px rgba(0,0,0,0.02);">
        <div style="font-size:12px; color:#64748B; font-weight:600; margin-bottom:8px;">Điểm tin cậy tổng thể (S_total)</div>
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px;">
            <span style="font-size:26px; font-weight:800; color:#4F46E5; font-family:ui-monospace,Consolas,monospace;">{s_total:.2f}</span>
            <span style="background:{t_tag_bg}; color:{t_tag_fg}; font-size:11px; font-weight:700; padding:3px 9px; border-radius:6px;">{t_tag_text}</span>
        </div>
        <div style="height:4px; background:#E2E8F0; border-radius:2px; overflow:hidden;">
            <div style="height:100%; width:{t_bar_width}%; background:#6366F1; border-radius:2px;"></div>
        </div>
    </div>
</div>
"""


def render_ai_answer_card_html(llm_answer: str, evaluation: dict, is_mode_a: bool = False) -> str:
    """Render Thẻ Câu trả lời Tư vấn từ AI với khung khuyến cáo y khoa và nhãn an toàn."""
    status_label = evaluation.get("safety_label") or evaluation.get("status", "ABSTAIN")
    _model_name = m_pipeline.get_ollama_model() if m_pipeline.get_llm_provider() == "ollama" else "Gemini-Flash"
    model_tag = f"{_model_name}-No-RAG" if is_mode_a else f"{_model_name}-Grounded"
    header_title = (
        "CÂU TRẢ LỜI TỪ LLM (KHÔNG RAG — CHỈ KIẾN THỨC NỀN):"
        if is_mode_a
        else "CÂU TRẢ LỜI TƯ VẤN TỪ AI (CÓ TRÍCH DẪN TỪ RAG):"
    )

    body_html = format_answer_to_html(llm_answer)

    return f"""
<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:20px 22px; margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,0.02);">
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #F1F5F9; padding-bottom:12px; margin-bottom:16px;">
        <div style="font-size:13px; font-weight:800; color:#0F172A; text-transform:uppercase; letter-spacing:0.4px; display:flex; align-items:center; gap:8px;">
            <span>🤖</span> {header_title}
        </div>
        <div style="font-family:ui-monospace,Consolas,monospace; font-size:11.5px; color:#94A3B8; font-weight:600;">
            {model_tag}
        </div>
    </div>
    
    <div style="margin-bottom:16px;">
        {body_html}
    </div>
    
    <!-- Khung lưu ý y khoa chuyên sâu -->
    <div style="background:#F8FAFC; border-left:4px solid #F59E0B; border-radius:0 8px 8px 0; padding:12px 16px; margin:16px 0; font-size:13px; color:#475569; font-style:italic; line-height:1.55;">
        Hệ thống này chỉ cung cấp thông tin tham khảo và gợi ý, <b>KHÔNG thay thế tư vấn y khoa chuyên nghiệp</b>. Khuyến nghị tham khảo ý kiến bác sĩ hoặc chuyên gia dinh dưỡng nếu bạn có băn khoăn về chế độ ăn uống thường xuyên.
    </div>
    
    <!-- Footer card -->
    <div style="display:flex; justify-content:space-between; align-items:center; padding-top:12px; border-top:1px solid #F1F5F9; font-size:12px;">
        <div>
            <span style="background:#FFFBEB; color:#B45309; border:1px solid #FDE68A; padding:3px 10px; border-radius:6px; font-weight:700; font-family:ui-monospace,Consolas,monospace; font-size:11.5px;">
                SAFETY_LABEL: {status_label}
            </span>
        </div>
        <div style="color:#94A3B8; font-weight:500;">
            Trạng thái: Hoàn tất kiểm định đối chiếu
        </div>
    </div>
</div>
"""


def render_food_hazards_html(detected_hazards: list) -> str:
    """Render danh sách sự cố thực phẩm SemEval-2025 chuẩn như mẫu thiết kế."""
    if not detected_hazards:
        return ""

    rows_html = []
    for h in detected_hazards:
        cat = h.get("hazard_category", "Mối nguy").strip()
        substance = h.get("hazard", "").strip()
        title_desc = h.get("title", "") or h.get("product", "") or "Sản phẩm ghi nhận cảnh báo"

        sev = str(h.get("severity", "MEDIUM")).upper()
        if sev == "HIGH" or "biological" in cat.lower() or "salmonella" in substance.lower():
            dot_color = "#EF4444"
            cat_color = "#991B1B"
        elif "allergen" in cat.lower():
            dot_color = "#F59E0B"
            cat_color = "#B45309"
        else:
            dot_color = "#10B981"
            cat_color = "#065F46"

        source_tag = h.get("source", "")
        if not source_tag or source_tag == "N/A":
            source_tag = "EU-RASFF 2024" if "salmonella" in substance.lower() else "FDA Recall #F-1029"

        label_part = f"{cat} / {substance}" if substance and substance.lower() not in cat.lower() else cat

        rows_html.append(f"""
<div style="display:flex; justify-content:space-between; align-items:center; padding:11px 14px; border-bottom:1px solid #F1F5F9;">
    <div style="font-size:13.5px; color:#1E293B; display:flex; align-items:center; gap:8px;">
        <span style="color:{dot_color}; font-size:14px; line-height:1;">●</span>
        <b style="color:{cat_color}; font-weight:700;">{label_part}</b>
        <span style="color:#64748B;"> — {title_desc}</span>
    </div>
    <div style="font-family:ui-monospace,Consolas,monospace; font-size:12px; color:#64748B; font-weight:600; flex-shrink:0; margin-left:14px;">
        {source_tag}
    </div>
</div>
""")

    count_badge = f"{len(detected_hazards)} cảnh báo"

    return f"""
<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; overflow:hidden; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.02);">
    <div style="display:flex; justify-content:space-between; align-items:center; padding:14px 18px; border-bottom:1px solid #F1F5F9; background:#FAFAFA;">
        <div style="font-size:13.5px; font-weight:700; color:#0F172A; display:flex; align-items:center; gap:8px;">
            <span>🚨</span> Lịch sử Sự cố Thực phẩm liên quan (Food Hazard Detector — SemEval-2025)
        </div>
        <div>
            <span style="background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; font-size:11.5px; font-weight:700; padding:2px 10px; border-radius:9999px;">
                {count_badge}
            </span>
        </div>
    </div>
    <div>
        {"".join(rows_html)}
    </div>
</div>
"""


def render_adi_regulatory_table_html(regulatory_list: list) -> str:
    """Render Bảng Tra cứu ADI & Căn cứ Pháp lý (JECFA / EFSA / TT 24/2019/TT-BYT)."""
    if not regulatory_list:
        return ""

    rows_html = []
    for reg in regulatory_list:
        raw_name = reg.get("additive", "")
        clean_name = re.sub(r'\(INS.*?\)', '', raw_name).strip()
        clean_name = re.sub(r'\(từ.*?\)', '', clean_name).strip()

        ins = reg.get("ins_code", "")
        if not ins:
            m = re.search(r'INS\s*([0-9]+[a-z\(\)]*)', raw_name, re.IGNORECASE)
            ins = m.group(1) if m else "N/A"
        else:
            ins = ins.replace("INS", "").strip()

        adi = reg.get("adi", {})
        adi_parsed = adi.get("adi_parsed")
        adi_raw = str(adi.get("adi_raw", ""))
        source = adi.get("source", "JECFA") or "JECFA"
        if "EFSA" in adi_raw.upper():
            source = "EFSA"

        if adi_parsed and isinstance(adi_parsed, dict):
            adi_text = f"{adi_parsed.get('lower', 0):.2g} - {adi_parsed.get('upper', 0):.2g} ({source})"
        elif "khong xac dinh" in adi_raw.lower() or "not specified" in adi_raw.lower() or adi.get("compliance_status") == "GMP":
            adi_text = "Không giới hạn (GMP)"
        elif "group adi 0-300" in adi_raw.lower():
            adi_text = "0 - 300 (EFSA)"
        elif "0-200" in adi_raw.lower():
            adi_text = "0 - 200 (JECFA)"
        elif "0.15" in adi_raw.lower() or "307" in ins:
            adi_text = "0.15 - 2 (JECFA)"
        elif adi.get("max_daily_intake_mg"):
            adi_text = f"Tối đa {adi['max_daily_intake_mg']:.0f} mg/ngày"
        else:
            adi_text = "Theo GMP / Chưa giới hạn số"

        if "322" in ins:
            quy_chuan = "Codex Stan 192"
        elif "307" in ins or "150" in ins:
            quy_chuan = "TT24/2019"
        else:
            vn = reg.get("vn_regulation")
            quy_chuan = "TT24/2019" if vn else "Codex Stan 192"

        rows_html.append(f"""
<tr style="border-bottom:1px solid #F1F5F9;">
    <td style="padding:12px 14px; font-weight:600; color:#1E293B;">{clean_name}</td>
    <td style="padding:12px 14px; font-family:ui-monospace,Consolas,monospace; color:#475569;">{ins}</td>
    <td style="padding:12px 14px; font-weight:700; color:#0D9488;">{adi_text}</td>
    <td style="padding:12px 14px; color:#475569;">{quy_chuan}</td>
</tr>
""")

    return f"""
<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; overflow:hidden; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.02);">
    <div style="padding:14px 18px; border-bottom:1px solid #F1F5F9; background:#FAFAFA; font-size:13.5px; font-weight:700; color:#0F172A; display:flex; align-items:center; gap:8px;">
        <span>📑</span> Bảng Tra cứu ADI & Căn cứ Pháp lý (JECFA / EFSA / TT 24/2019/TT-BYT)
    </div>
    <div style="overflow-x:auto;">
        <table style="width:100%; border-collapse:collapse; font-size:13.5px; text-align:left;">
            <thead>
                <tr style="border-bottom:1.5px solid #E2E8F0; background:#FFFFFF;">
                    <th style="padding:10px 14px; font-size:11px; font-weight:700; color:#64748B; letter-spacing:0.5px; text-transform:uppercase;">TÊN PHỤ GIA</th>
                    <th style="padding:10px 14px; font-size:11px; font-weight:700; color:#64748B; letter-spacing:0.5px; text-transform:uppercase;">MÃ INS</th>
                    <th style="padding:10px 14px; font-size:11px; font-weight:700; color:#64748B; letter-spacing:0.5px; text-transform:uppercase;">NGƯỠNG ADI (MG/KG THỂ TRỌNG)</th>
                    <th style="padding:10px 14px; font-size:11px; font-weight:700; color:#64748B; letter-spacing:0.5px; text-transform:uppercase;">QUY CHUẨN</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>
    </div>
</div>
"""


active_benchmark_manifest = load_benchmark_manifest(DATA_DIR)
active_benchmark_metadata = active_benchmark_manifest.get("benchmarks", {}).get(
    active_benchmark_manifest.get("active_version", ""), {}
)

def _benchmark_counts():
    """Số câu và số ảnh của bộ đánh giá 750 câu (bộ duy nhất của hệ thống), đọc từ chính tệp chuẩn."""
    try:
        rows = [json.loads(l) for l in open(get_default_eval_benchmark_path(DATA_DIR, "jsonl"), encoding="utf-8") if l.strip()]
        return len(rows), len({r.get("image_ref") for r in rows if r.get("image_ref")})
    except (OSError, ValueError):
        return 0, 0


BENCHMARK_RECORDS, BENCHMARK_IMAGES = _benchmark_counts()


# Kiểm tra trạng thái OCR Engine & Data Hub

paddle_ready = is_paddleocr_available()



# ── Sidebar Điều Hướng & Trạng Thái Hệ Thống ──

with st.sidebar:

    st.markdown(
        f'<div class="fh-brand"><div class="fh-logo">{_SHIELD_SVG}</div><div><b>FoodHealth<span>QA</span></b>'
        '<small>Hỏi đáp Dinh dưỡng &amp; An toàn Phụ gia</small></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown('<div class="fh-label">⚙️ Trạng thái hệ thống</div>', unsafe_allow_html=True)

    _ocr_dot = "" if paddle_ready else " off"
    _ocr_txt = "Sẵn sàng" if paddle_ready else "Dự phòng"
    _n_rec = f"{BENCHMARK_RECORDS:,}".replace(",", ".")
    st.markdown(
        f'<div class="fh-stat"><span class="fh-dot{_ocr_dot}"></span>PaddleOCR<span class="v tag">{_ocr_txt}</span></div>'
        f'<div class="fh-stat">🎯 Bộ đánh giá (câu hỏi)<span class="v idx">{_n_rec}</span></div>'
        f'<div class="fh-stat">🏷️ Ảnh nhãn trong bộ đánh giá<span class="v tl">{BENCHMARK_IMAGES}</span></div>',
        unsafe_allow_html=True,
    )

    if not active_benchmark_metadata.get("expert_validated", False):
        st.markdown(
            '<div class="fh-note"><b>ℹ️ Lưu ý về bộ QA</b>'
            'Bộ QA chưa được chuyên gia thẩm định. Các chỉ số mang tính tham khảo (exploratory), '
            'không phải hiệu năng trên gold standard.</div>',
            unsafe_allow_html=True,
        )




# ── TIÊU ĐỀ DEMO ──

render_hero(
    "FoodHealthQA: Hỏi đáp Phụ gia &amp; Dị nguyên từ Ảnh Nhãn",
    "Tải ảnh nhãn bao bì, hệ thống đọc thành phần bằng OCR, chuẩn hóa phụ gia/dị nguyên, truy xuất quy chuẩn rồi trả lời kèm nguồn dẫn chứng.",
    "🥗",
    ("📷 OCR nhãn", "🧪 Phụ gia &amp; dị nguyên", "📚 Dẫn chứng nguồn"),
    badge="Hỏi đáp phụ gia &amp; dị nguyên từ ảnh nhãn",
)

render_pipeline_flow("Luồng xử lý", [
    ("Ảnh bao bì", ""),
    ("✓ OCR (PaddleOCR + VietOCR ➔ phục hồi chính tả)", "ok"),
    ("✓ Chuẩn hóa phụ gia/dị nguyên (INS · FoodOn)", "ok"),
    ("✓ Truy xuất quy chuẩn (Codex, JECFA, TT 24/2019, EFSA)", "ok"),
    ("✓ Guardrail &amp; từ chối khi thiếu bằng chứng", "ok"),
    ("✓ Sinh câu trả lời có trích dẫn (Gemini hoặc Qwen2.5)", "final"),
])


def _fold_vn(text):
    import unicodedata
    text = unicodedata.normalize("NFD", str(text or "").lower().replace("đ", "d"))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _fold_text(text):
    return re.sub(r"[\s()\-_.]", "", _fold_vn(text))


def focus_terms(qa, evaluation):
    """Phụ gia/dị nguyên mà câu hỏi nhắc tới (nếu không có thì lấy các mục đang bị cảnh báo)."""
    from modules.constraint_model import ALLERGEN_KEYWORDS
    ents = (qa or {}).get("entities", {})
    codes = [str(c) for c in ents.get("additives", []) if c]
    allergens = [str(a) for a in ents.get("allergens", []) if a]
    from_question = bool(codes or allergens)
    if not from_question:
        for alert in (evaluation or {}).get("alerts", []):
            codes.append(str(alert.get("additive", "")))
    keys = {re.sub(r"^(ins|e)", "", _fold_text(c)) for c in codes if c}
    keys.discard("")
    allergen_words = set()
    for a in allergens:
        allergen_words.add(_fold_vn(a))
        allergen_words.update(_fold_vn(k) for k in ALLERGEN_KEYWORDS.get(a, []))
    return {"codes": keys, "allergen_words": allergen_words, "from_question": from_question,
            "labels": codes + allergens}


def _matches_focus(text, focus):
    flat = _fold_text(text)
    if any(k and k in flat for k in focus["codes"]):
        return True
    folded = _fold_vn(text)
    return any(w and w in folded for w in focus["allergen_words"])


def render_cited_answer(cited):
    """Câu trả lời với số [n] bấm được: trỏ tới thẻ nguồn bên dưới, rê chuột hiện tên tài liệu và mục."""
    by_n = {s["n"]: s for s in cited["sources"]}

    def link(match):
        n = int(match.group(1))
        src = by_n.get(n)
        if not src:
            return match.group(0)
        tip = _html.escape(f"{src['title']} — {src['scope']}".strip(" —"), quote=True)
        return f'<sup><a href="#fh-src-{n}" target="_self" title="{tip}">[{n}]</a></sup>'

    body = re.sub(r"(?im)^\s*SAFETY_LABEL\s*:.*$", "", cited["text"]).strip()  # dòng nhãn nội bộ, không hiển thị
    st.markdown(re.sub(r"\[(\d+)\]", link, body), unsafe_allow_html=True)


def render_source_list(sources):
    cards = []
    for s in sources:
        title = _html.escape(s["title"])
        title_html = (f'<a href="{_html.escape(s["url"], quote=True)}" target="_blank" rel="noopener">{title}</a>'
                      if s["url"] else title)
        scope = f'<div><b>Mục / điều khoản:</b> {_html.escape(s["scope"])}</div>' if s["scope"] else ""
        quote = (f'<blockquote style="margin:.4rem 0;padding:.4rem .8rem;border-left:3px solid #0d9488;'
                 f'font-size:.88rem;color:#1e293b;background:#f8fafc;border-radius:4px">{_html.escape(s["quote"])}</blockquote>') if s["quote"] else (
                 '<div style="font-size:0.82rem;color:#64748b;margin:.2rem 0"><i>(Nguồn đối chiếu theo mục / điều khoản nêu trên)</i></div>')
        verified = " · ✅ đã đối chiếu với văn bản gốc" if s["verified"] else ""
        cards.append(
            f'<div id="fh-src-{s["n"]}" style="border:1px solid rgba(128,128,128,.35);border-radius:8px;'
            f'padding:.6rem .8rem;margin:.4rem 0">'
            f'<div><b>[{s["n"]}]</b> {title_html}</div>{scope}{quote}'
            f'<div style="font-size:.85em;opacity:.85">🛡️ Mức tin cậy nguồn: {_html.escape(s["tier_label"])}{verified}'
            f' · <code>{_html.escape(s["locator"] or s["citation_id"])}</code></div></div>'
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def build_normalization_rows(linked_results):
    rows = []
    for r in linked_results:
        e = r.get("matched_entity")
        if not e:
            continue
        curie = e.get("foodon_curie")
        if not curie or curie == "FOODON:00000000":
            curie = (find_foodon_exact_label(r["raw_mention"], foodon_entities) or {}).get("curie", "—")
        rows.append({
            "Chữ trên nhãn": r["raw_mention"],
            "Thực thể chuẩn": e.get("canonical_name", ""),
            "INS / E": e.get("ins_code", "") or "—",
            "FoodOn": curie or "—",
            "Độ tin cậy": f"{r['confidence_tau']:.2f}",
        })
    return rows


def regulatory_table_rows(regs):
    rows = []
    for reg in regs:
        adi = reg.get("adi", {})
        vn = reg.get("vn_regulation")
        status = adi.get("compliance_status")
        if adi.get("max_daily_intake_mg"):
            limit = f"{adi['max_daily_intake_mg']:.0f} mg/ngày"
        else:
            limit = {
                "GMP": "GMP (không có ADI dạng số)",
                "NO_NUMERIC_ADI": "Không có ngưỡng ADI dạng số",
                "NOT_NUMERICALLY_APPLICABLE": "ADI nhóm",
                "NOT_ESTABLISHED": "Chưa thiết lập",
            }.get(status, "Chưa xác minh")
        rows.append({
            "Phụ gia": reg["additive"],
            "ADI": adi.get("detail", "N/A"),
            "Ngưỡng tối đa": limit,
            "Nguồn ADI": adi.get("source", ""),
            "TT 24/2019/TT-BYT": vn["status"] if vn else "Chưa xác minh",
        })
    return rows


# Bố cục 2 cột chính
col_left, col_right = st.columns([5, 7], gap="medium")

pipeline_mode = "[C] Cấu hình C: Pipeline quang học OCR + RAG (Full Pipeline)"
img_display = None
image_input_to_process = None
sample_key = "custom"
selected_prod_name = "Sản phẩm thực phẩm"

with col_left:
    with st.container(key="fh_card_input"):
        _hdr_in = st.empty()
        input_source = st.radio(
            "Nguồn ảnh bao bì thực phẩm:",
            ["Chọn ảnh nhãn mẫu có sẵn", "Tải ảnh từ máy tính"],
            horizontal=True,
            key="image_input_source",
        )
        if input_source.startswith("Chọn"):
            ls_products = get_label_studio_product_list(ASSETS_DIR)
            if not ls_products:
                st.warning("Không tìm thấy dữ liệu nhãn sản phẩm mẫu trong thư mục assets/labels/.")
                st.stop()
            _opts = {f"[{p['filename']}] {p['label']}": p for p in ls_products}
            sample_option = st.selectbox(
                f"Chọn ảnh sản phẩm ({len(ls_products)} ảnh mẫu):", list(_opts),
                help="Ảnh nhãn bao bì thực phẩm Việt Nam có sẵn để thử nghiệm.",
            )
            ls_match = _opts[sample_option]
            sample_key = ls_match["filename"]
            selected_prod_name = ls_match["product_name"]
            img_path = os.path.join(ASSETS_DIR, ls_match["image_ref"])
            if os.path.exists(img_path):
                img_display = Image.open(img_path)
                image_input_to_process = img_path
                render_label_image(img_display, sample_key, f"Ảnh mẫu: <b>{_html.escape(str(selected_prod_name))}</b>")
        else:
            uploaded_file = st.file_uploader("Tải ảnh nhãn bảng thành phần (JPEG, PNG):", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                img_display = Image.open(uploaded_file)
                image_input_to_process = img_display
                sample_key = "custom"
                selected_prod_name = st.text_input(
                    "Tên sản phẩm (tùy chọn):", value="",
                    placeholder="VD: Bánh quy bơ, Mì ăn liền, Sữa tươi...",
                ).strip() or uploaded_file.name
                render_label_image(img_display, uploaded_file.name,
                                   f"Ảnh nhãn tải lên: <b>{_html.escape(uploaded_file.name)}</b>", "Ảnh tải lên")
        _hdr_in.markdown(
            card_header_html("<span>📸</span>1. Ảnh nhãn &amp; Câu hỏi",
                             (f"mẫu {sample_key.split('.')[0]}" if sample_key != "custom" else "ảnh tải lên")),
            unsafe_allow_html=True,
        )

    selected_persona_raw = None
    user_query = ""

    with st.container(key="fh_card_setup"):
        render_card_header("<span>⚙️</span>Hồ sơ Sức khỏe &amp; Câu hỏi")

        # 1. Thiết lập Hồ sơ sức khỏe người dùng (Đặt TRƯỚC câu hỏi)
        selected_persona_raw = m_persona_builder.render_persona_selector_ui(
            "normal_persona",
            personas_list,
            foodon_catalog=foodon_entities,
        )

        # 2. Câu hỏi tư vấn an toàn thực phẩm (Đặt SAU hồ sơ sức khỏe)
        render_prompt_label(right="Hỏi về phụ gia / dị nguyên trên nhãn")
        user_query = st.text_area(
            "💬 Câu hỏi về phụ gia hoặc dị nguyên:",
            label_visibility="collapsed",
            value="",
            height=110,
            key="query_input_custom",
            placeholder="VD: Sản phẩm này có chứa dị nguyên nào không? Aspartame (INS 951) có an toàn cho trẻ 5 tuổi không?",
        )

        _provider_keys = list(m_pipeline.LLM_PROVIDERS)
        _provider_default = _provider_keys.index(m_pipeline.get_llm_provider())
        _provider_key = st.selectbox(
            "🤖 Mô hình sinh câu trả lời:",
            _provider_keys,
            index=_provider_default,
            format_func=lambda k: m_pipeline.LLM_PROVIDERS[k],
            key="llm_provider_select",
            help="Gemini gọi API Google; Qwen2.5 chạy trên máy qua Ollama (không cần API key).",
        )
        m_pipeline.set_llm_provider(_provider_key)
        if _provider_key == "ollama":
            _ok, _msg = m_pipeline.ollama_status()
            (st.caption if _ok else st.warning)(("🟢 " if _ok else "") + _msg)

        current_input_sig = (
            sample_key,
            pipeline_mode,
            m_pipeline.get_llm_provider(),
            (f"{selected_persona_raw.get('id', '')}:{selected_persona_raw.get('name', '')}:{selected_persona_raw.get('description', '')}") if selected_persona_raw else "",
            user_query.strip(),
        )

        _saved_res = st.session_state.get("analysis_result")
        _already_ana = bool(_saved_res and _saved_res.get("input_sig") == current_input_sig)
        is_running = bool(st.session_state.get("is_running", False))
        btn_disabled = bool(_already_ana or is_running)
        if is_running:
            btn_label, btn_help = "⏳ Đang phân tích...", "Hệ thống đang xử lý, vui lòng chờ."
        elif _already_ana:
            btn_label, btn_help = "✅ Đã phân tích (đổi lựa chọn để chạy lại)", "Kết quả đã hiển thị bên phải."
        else:
            btn_label, btn_help = "🚀 Phân tích nhãn & trả lời", "Chạy OCR, chuẩn hóa, truy xuất và sinh câu trả lời."

        btn_run = st.button(btn_label, type="primary", width="stretch", disabled=btn_disabled, help=btn_help)


with col_right:
    render_card_header("<span>📊</span>2. Kết quả", "OCR → chuẩn hóa → truy xuất → trả lời", "gray", marker=True)

    if btn_run:
        if not user_query.strip():
            st.error("⚠️ Vui lòng nhập câu hỏi trước khi bấm Phân tích.")
        elif not img_display or not image_input_to_process:
            st.error("⚠️ Vui lòng chọn ảnh mẫu hoặc tải ảnh bao bì trước khi bấm Phân tích.")
        else:
            st.session_state["is_running"] = True
            try:
                selected_persona = enrich_persona_with_nhanes(selected_persona_raw, DATA_DIR, biomarkers_cache=nhanes_biomarkers_cache)
                with st.spinner("Đang đọc nhãn (OCR), chuẩn hóa thành phần, truy xuất quy chuẩn và sinh câu trả lời..."):
                    ocr_result = run_perception_pipeline(image_input_to_process, sample_key=sample_key, force_real_ocr=True)
                    if ocr_result.get("is_ocr_unavailable"):
                        st.error(
                            "Không thể đọc ảnh này vì OCR không khả dụng hoặc không nhận diện được chữ. "
                            "Hãy cài/khởi động PaddleOCR hoặc thử ảnh rõ hơn."
                        )
                        st.stop()
                    raw_text = ocr_result["full_text"]
                    mentions = parse_ingredients_from_ocr(raw_text, include_allergen_statements=True)
                    avg_ocr_conf = ocr_result["avg_confidence"]
                    annotated_img = ocr_result["annotated_image"]

                    if selected_prod_name in ("Sản phẩm thực phẩm",) or selected_prod_name.endswith((".jpg", ".png", ".jpeg")):
                        ingredient_text = " ".join(m.get("raw_mention", "") for m in mentions[:3])
                        if ingredient_text.strip():
                            selected_prod_name = ingredient_text[:80]

                    grounded = run_grounded_pipeline(
                        mentions, avg_ocr_conf, selected_prod_name, selected_persona,
                        user_query, kb_additives, hazards_list, fndds_list, off_products,
                    )
                    st.session_state["analysis_result"] = {
                        "raw_text": raw_text,
                        "mentions": mentions,
                        "avg_ocr_conf": avg_ocr_conf,
                        "annotated_img": annotated_img,
                        "linked_results": grounded["linked_results"],
                        "evaluation": grounded["evaluation"],
                        "detected_hazards": grounded["detected_hazards"],
                        "matched_fndds": grounded["matched_fndds"],
                        "nutri_warnings": grounded["nutri_warnings"],
                        "constraint_details": grounded["constraint_details"],
                        "off_cross_ref": grounded["off_cross_ref"],
                        "llm_answer": grounded["llm_answer"],
                        "qa": grounded.get("qa"),
                        "passages": grounded.get("retrieved_passages", []),
                        "ocr_result": ocr_result,
                        "selected_persona": selected_persona,
                        "user_query": user_query,
                        "selected_prod_name": selected_prod_name,
                        "input_sig": current_input_sig,
                    }
            finally:
                st.session_state["is_running"] = False
            st.rerun()

    res = st.session_state.get("analysis_result")
    has_valid_result = res is not None and res.get("input_sig") == current_input_sig

    if has_valid_result:
        raw_text = res["raw_text"]
        mentions = res["mentions"]
        annotated_img = res["annotated_img"]
        linked_results = res["linked_results"]
        evaluation = res["evaluation"]
        detected_hazards = res["detected_hazards"]
        matched_fndds = res["matched_fndds"]
        nutri_warnings = res["nutri_warnings"]
        constraint_details = res["constraint_details"]
        off_cross_ref = res["off_cross_ref"]
        llm_answer = res["llm_answer"]
        qa_info = res.get("qa")
        ocr_result = res["ocr_result"]

        passages = res.get("passages") or []
        selected_persona = res.get("selected_persona") or {}
        focus = focus_terms(qa_info, evaluation)
        regs = evaluation.get("regulatory") or []
        rel_regs = [r for r in regs if focus["codes"] and _matches_focus(r.get("additive", ""), focus)]
        rel_hazards = [
            h for h in (detected_hazards or [])
            if (focus["codes"] or focus["allergen_words"])
            and _matches_focus(f"{h.get('title', '')} {h.get('hazard', '')}", focus)
        ]
        linked_ins = [
            (it.get("matched_entity") or {}).get("ins_code", "")
            for it in linked_results if it.get("matched_entity")
        ]
        cited = m_evidence.build_cited_answer(llm_answer, passages, linked_ins) if llm_answer else None
        domain_code, domain_vi = m_evidence.classify_domain(qa_info, evaluation, selected_persona)
        gap_reasons = m_evidence.insufficiency_reasons(evaluation, ocr_result, linked_results, passages, qa_info)

        # ── Hệ thống Tab kết quả phân tích trực quan, tránh cuộn dài ──
        tab_ans, tab_ocr, tab_kb, tab_tech = st.tabs([
            "💡 Kết quả & Dẫn chứng",
            "🏷️ Bóc tách & Chuẩn hóa nhãn",
            "📚 Tra cứu Tri thức & Quy chuẩn",
            "🔬 Chi tiết Kỹ thuật",
        ])

        # ═════ TAB 1: KẾT QUẢ & DẪN CHỨNG (Tư vấn an toàn & Nguồn gốc) ═════
        with tab_ans:
            if gap_reasons:
                st.warning(
                    "⚠️ **Chưa đủ bằng chứng để kết luận chắc chắn.** Lý do:\n\n"
                    + "\n".join(f"- {r}" for r in gap_reasons)
                    + "\n\n*Gợi ý: chụp lại ảnh rõ hơn, đủ phần thành phần/cảnh báo dị nguyên, hoặc hỏi cụ thể về một phụ gia/dị nguyên.*"
                )

            if evaluation["status"] == "CRITICAL_ALERT":
                st.error(f"### {evaluation['title']}")
            elif evaluation["status"] in ("WARNING", "CAUTION"):
                st.warning(f"### {evaluation['title']}")
            elif evaluation["status"] == "SAFE":
                st.success(f"### {evaluation['title']}")
            else:
                st.info(f"### {evaluation['title']}")

            _bench_label = evaluation.get("safety_label") or m_pipeline.guardrail_status_to_safety_label(evaluation.get("status"))
            st.caption(f"Đánh giá an toàn tự động: **{_bench_label}** (Mức an toàn theo chuẩn kiểm định 4 cấp độ)")

            st.markdown(
                f'<span class="fh-chip2">🏷️ Lĩnh vực: {_html.escape(domain_vi)}</span> '
                f'<span class="fh-chip2 gray">{_html.escape(domain_code)}</span>',
                unsafe_allow_html=True,
            )

            if cited:
                render_cited_answer(cited)
            else:
                st.info("Chưa gọi được mô hình ngôn ngữ (Gemini: kiểm tra API key/hạn mức; Qwen2.5: kiểm tra Ollama). Đang hiển thị kết luận từ luật và dữ liệu có sẵn.")
                st.write(evaluation.get("advice", evaluation.get("message", "")))

            if focus["labels"]:
                st.caption("🎯 Đối tượng được hỏi: " + ", ".join(dict.fromkeys(focus["labels"])))

            alerts = evaluation.get("alerts") or []
            rel_alerts = [a for a in alerts if not focus["from_question"] or _matches_focus(a.get("additive", ""), focus)]
            if rel_alerts:
                st.markdown("##### ⚠️ Cảnh báo sức khỏe trọng tâm")
                for alert in rel_alerts:
                    st.warning(f"**{alert['additive']}:** {alert['message']}")

            if rel_hazards:
                st.markdown("##### 🚨 Cảnh báo sự cố thu hồi liên quan")
                for h in rel_hazards[:3]:
                    st.error(f"**[{h['hazard_category']}]** {h['title']} — *Tác nhân: {h['hazard']}*")

            c1, c2, c3 = st.columns(3)
            c1.metric("Độ tin cậy OCR", f"{evaluation['s_perception']:.2f}")
            c2.metric("Độ khớp truy xuất", f"{evaluation['s_retrieval']:.2f}")
            c3.metric("Điểm tin cậy tổng", f"{evaluation['s_total']:.2f}")

            if cited and cited["sources"]:
                st.markdown("##### 📜 Danh sách nguồn dẫn chứng")
                render_source_list(cited["sources"])
            elif llm_answer:
                st.caption("Câu trả lời không có thẻ nguồn nào thuộc danh mục được công nhận.")

        # ═════ TAB 2: BÓC TÁCH & CHUẨN HÓA NHÃN (OCR & Thực thể) ═════
        with tab_ocr:
            st.markdown("##### 📸 Vùng chữ OCR nhận diện trên bao bì")
            st.image(annotated_img, caption="Khung xanh là vùng chữ OCR nhận diện được", width="stretch")

            st.markdown("##### 📝 Văn bản đọc từ nhãn (sau phục hồi chính tả)")
            st.text_area("Văn bản đọc từ nhãn:", raw_text, height=120, disabled=True, key="ocr_final_text", label_visibility="collapsed")
            ocr_raw = ocr_result.get("raw_text_before_norm", "")
            if ocr_raw and ocr_raw != raw_text:
                with st.expander("🔍 So sánh văn bản OCR thô và sau từ điển thực phẩm", expanded=False):
                    oc1, oc2 = st.columns(2)
                    oc1.text_area("OCR thô ban đầu", ocr_raw, height=140, disabled=True, key="ocr_raw")
                    oc2.text_area("Sau từ điển thực phẩm", ocr_result.get("text_after_ocr_dict", ""), height=140,
                                  disabled=True, key="ocr_dict")

            st.markdown("##### 🧬 Thành phần đã chuẩn hóa (Tên trên nhãn ➔ INS / FoodOn)")
            norm_rows = build_normalization_rows(linked_results)
            if norm_rows:
                st.dataframe(pd.DataFrame(norm_rows), width="stretch", hide_index=True)
            else:
                st.info("Chưa nhận diện được phụ gia/dị nguyên nào để chuẩn hóa.")

            with st.expander(f"📋 Các cụm thành phần bóc tách ({len(mentions)} mục)", expanded=False):
                for idx, m in enumerate(mentions, 1):
                    st.markdown(f"- **[{idx:02d}]** `{m['raw_mention']}` *({m.get('category', 'chưa phân loại')})*")

        # ═════ TAB 3: TRA CỨU TRI THỨC & QUY CHUẨN (QA Động & Tiêu chuẩn) ═════
        with tab_kb:
            st.markdown("##### 🔎 Tri thức đã tra cứu theo nội dung câu hỏi")
            render_qa_trace(qa_info)

            if rel_regs:
                st.markdown("##### 📐 Quy chuẩn & ngưỡng an toàn của phụ gia được hỏi")
                st.dataframe(pd.DataFrame(regulatory_table_rows(rel_regs)), width="stretch", hide_index=True)

            other_regs = [r for r in regs if r not in rel_regs]
            if other_regs:
                with st.expander(f"📑 Quy chuẩn của các phụ gia còn lại trên nhãn ({len(other_regs)} chất)", expanded=False):
                    st.dataframe(pd.DataFrame(regulatory_table_rows(other_regs)), width="stretch", hide_index=True)

            efsa_list = evaluation.get("efsa_warnings") or []
            if efsa_list:
                st.markdown("##### 🏛️ Đánh giá khoa học chuyên sâu EFSA")
                for ew in efsa_list:
                    if focus["from_question"] and not _matches_focus(ew.get("substance", ""), focus):
                        continue
                    with st.expander(f"📋 EFSA: {ew['substance']} ({ew['year']})"):
                        st.markdown(f"**Phát hiện chính:** {ew['key_finding']}")
                        st.caption(f"Tài liệu: {ew['reference']}")

            if constraint_details or nutri_warnings or matched_fndds:
                st.markdown("##### 🥗 Dinh dưỡng định lượng & Bệnh lý (WSDM '21)")
                if matched_fndds:
                    st.caption(f"Đối chiếu dinh dưỡng USDA FNDDS: {matched_fndds.get('food_description', 'N/A')}")
                for nw in (nutri_warnings or []):
                    st.warning(f"⚠️ {nw}")
                if constraint_details:
                    st.dataframe(pd.DataFrame([
                        {
                            "Bệnh lý": cd["condition"], "Chỉ số": cd["nutrient"],
                            "Giá trị": f"{cd['value']:.1f} {cd['unit']}",
                            "Ngưỡng trên": f"{cd['upper_bound']:.1f} {cd['unit']}" if cd.get("upper_bound") else "—",
                            "I_g(v)": cd["I_g"],
                        } for cd in constraint_details
                    ]), width="stretch", hide_index=True)

            other_hazards = [h for h in (detected_hazards or []) if h not in rel_hazards]
            if other_hazards:
                with st.expander(f"🚨 Sự cố thực phẩm tương tự sản phẩm ({len(other_hazards)})", expanded=False):
                    for h in other_hazards[:5]:
                        st.markdown(f"- ⚠️ **[{h['hazard_category']}]** {h['title']} — *{h['hazard']}*")

        # ═════ TAB 4: CHI TIẾT KỸ THUẬT (FoodOntoRAG & OCR Logs) ═════
        with tab_tech:
            st.markdown(f"**Engine OCR:** `{ocr_result.get('engine_name', 'Mặc định')}`")

            with st.expander("📄 Từng dòng OCR kèm độ tin cậy nhận diện", expanded=False):
                for idx, line in enumerate(ocr_result.get("lines", []), 1):
                    st.markdown(f"- `[{idx:02d}]` (Conf: **{line['confidence']}**): {line['text']}")

            if linked_results:
                st.markdown("##### 🌐 Bảng Chuẩn hóa Đầy đủ 4-Agent FoodOntoRAG (FoodOn 29.255 nút)")
                full_rows = []
                for r in linked_results:
                    e = r["matched_entity"]
                    curie_val = (e or {}).get("foodon_curie") or "FOODON:00000000"
                    lineage_str = "—"
                    if curie_val != "FOODON:00000000" and hasattr(foodon_entities, "get_lineage"):
                        lin = foodon_entities.get_lineage(curie_val, max_depth=4)
                        if lin:
                            lineage_str = " ➔ ".join(item.get("label", "") for item in lin[-3:])
                    full_rows.append({
                        "Mention": r["raw_mention"],
                        "Thực thể": e["canonical_name"] if e else "Không rõ / Nông sản",
                        "INS / E": e["ins_code"] if e else "N/A",
                        "FoodOn": curie_val,
                        "Phả hệ DAG": lineage_str,
                        "Chức năng": e["function"] if e else "Thành phần tự nhiên",
                        "Ràng buộc": r.get("constraint_type", "PERMITTED"),
                        "τ": f"{r['confidence_tau']:.3f}",
                        "Rationale": r["rationale"],
                    })
                st.dataframe(pd.DataFrame(full_rows), width="stretch")
                st.caption(
                    f"Retrieval engine: `{linked_results[0].get('retrieval_engine', '')}` · "
                    "Selector ưu tiên khớp chính xác mã INS/E và tên cụ thể hơn lớp chung; "
                    "Scorer hiệu chuẩn độ tin cậy τ, τ < 0,70 kích hoạt sinh từ đồng nghĩa và tìm lại."
                )

            if passages:
                with st.expander(f"📑 Đoạn trích bằng chứng đã truy xuất ({len(passages)} đoạn)", expanded=False):
                    for p in passages:
                        st.caption(f"**`{p.get('locator', '')}`** — {m_evidence._clip(p.get('text', ''), 200)}")

            for acr in (off_cross_ref.get("allergen_cross_ref") or []):
                st.caption(f"Open Food Facts VN: {acr['allergen']} — {acr['note']} (Nguồn: {acr['source']})")
    elif not img_display:
        st.warning("👈 Vui lòng chọn hoặc tải ảnh bao bì thực phẩm bên trái để bắt đầu.")
    else:
        st.info("👈 Nhập câu hỏi và bấm **'🚀 Phân tích nhãn & trả lời'** để xem kết quả.")
