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



# paddle/PaddleOCR/VietOCR KHÔNG import ở đây (mất ~12 giây): modules/ocr_extractor nạp chúng khi cần,
# và luồng nền start_background_loads() khởi tạo sẵn ngay khi server bật.



import streamlit as st

import importlib
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
/* Thanh tiêu đề không chiếm chỗ nhưng KHÔNG bị ẩn: nút mở lại thanh bên (stExpandSidebarButton) nằm trong đó.
   Trước đây ẩn hẳn → khi thanh bên bị thu gọn (cửa sổ hẹp) thì không mở lại được, mất luôn bộ chọn chế độ. */
header[data-testid="stHeader"]{height:0!important;min-height:0!important;background:transparent!important;overflow:visible!important}
div[data-testid="stLogoSpacer"]{display:none!important}
div[data-testid="stSidebarHeader"]{height:0!important;min-height:0!important;margin:0!important;padding:0!important;overflow:visible!important}
div[data-testid="stSidebarCollapseButton"]{display:block!important;position:absolute!important;top:6px;right:6px;z-index:5}
[data-testid="stExpandSidebarButton"]{position:fixed!important;top:12px;left:12px;z-index:999991;background:#fff!important;
  border:1px solid #e2e8f0!important;border-radius:10px!important;box-shadow:0 2px 8px rgba(15,23,42,.12)!important}
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
[data-testid="stToolbar"]{background:transparent!important;height:0!important;min-height:0!important;overflow:visible!important}
[data-testid="stToolbarActions"],[data-testid="stMainMenu"],[data-testid="stAppDeployButton"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],#MainMenu,footer{display:none!important}
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

    off = load_openfoodfacts_vietnam(data_dir)

    foodon = load_foodon_entities(data_dir)

    return kb, personas, hazards, off, foodon


@st.cache_resource(show_spinner=False)
def start_background_loads(data_dir):
    """Chạy nền (một lần cho cả server) các bước nặng chỉ cần khi phân tích ảnh:
    khởi tạo OCR (~12 s import + nạp model), đọc NHANES (~5 s) và USDA FNDDS (~1,5 s).
    Giao diện hiện ngay; lúc người dùng gửi ảnh/câu hỏi đầu tiên thì thường đã nạp xong."""
    from concurrent.futures import ThreadPoolExecutor
    from modules.ocr_extractor import warm_up_ocr
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fh-preload")
    return {
        "ocr": pool.submit(warm_up_ocr),
        "nhanes": pool.submit(load_nhanes_biomarkers, data_dir),
        "fndds": pool.submit(load_fndds_nutrition, data_dir),
    }


def background_result(name):
    """Lấy kết quả nạp nền (chờ nếu chưa xong). Lỗi nạp nền → nạp lại trực tiếp như cách cũ."""
    try:
        return _BG_LOADS[name].result()
    except Exception:
        if name == "nhanes":
            return load_nhanes_biomarkers(DATA_DIR)
        if name == "fndds":
            return load_fndds_nutrition(DATA_DIR)
        return None



# Nạp dữ liệu đa nguồn thông minh từ DATA_DIR (được cache trong RAM, không load lại khi tương tác UI)

kb_additives, personas_list, hazards_list, off_products, foodon_entities = load_all_system_data(DATA_DIR)
_BG_LOADS = start_background_loads(DATA_DIR)



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



def _ocr_status_html():
    """Trạng thái OCR (PaddleOCR + VietOCR): khởi tạo nền ngay lần chạy trang đầu, dùng chung cho cả hai tab và mọi phiên."""
    fut = _BG_LOADS["ocr"]
    if not paddle_ready:
        dot, txt = " off", "Chưa cài"
    elif not fut.done():
        dot, txt = " off", "Đang khởi tạo…"
    else:
        try:
            state = fut.result()
        except Exception:
            state = {}
        if state.get("paddleocr") and state.get("vietocr"):
            dot, txt = "", "Sẵn sàng"
        elif state.get("paddleocr"):
            dot, txt = "", "Sẵn sàng (không VietOCR)"
        else:
            dot, txt = " off", "Lỗi khởi tạo"
    return f'<div class="fh-stat"><span class="fh-dot{dot}"></span>PaddleOCR + VietOCR<span class="v tag">{txt}</span></div>'


@st.fragment(run_every=3)
def _ocr_status_fragment():
    # tự làm mới để "Đang khởi tạo…" chuyển sang "Sẵn sàng" mà không cần tải lại trang
    st.markdown(_ocr_status_html(), unsafe_allow_html=True)


# ── Sidebar Điều Hướng & Trạng Thái Hệ Thống ──

with st.sidebar:

    st.markdown(
        f'<div class="fh-brand"><div class="fh-logo">{_SHIELD_SVG}</div><div><b>FoodHealth<span>QA</span></b>'
        '<small>Hỏi đáp Dinh dưỡng &amp; An toàn Phụ gia</small></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown('<div class="fh-label">⚙️ Trạng thái hệ thống</div>', unsafe_allow_html=True)

    _ocr_status_fragment()
    _n_rec = f"{BENCHMARK_RECORDS:,}".replace(",", ".")
    st.markdown(
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


def _fmt_score(value):
    """Điểm 0–1 hai chữ số; None (không đo được / không áp dụng) → "Không áp dụng"."""
    return "Không áp dụng" if value is None else f"{float(value):.2f}"


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


def _source_origin(url):
    """Tên miền của nguồn gốc, hiện dạng chữ thường (KHÔNG bấm được): dữ liệu mở là bản lưu trong KB của hệ thống."""
    m = re.match(r"https?://([^/]+)", url or "")
    return m.group(1) if m else ""


def _show_png(png, caption):
    """Nhúng ảnh trang PDF trực tiếp (data URI) thay vì st.image: tránh lỗi ảnh vỡ khi kho media của Streamlit
    đã xoá ảnh sau khi trang chạy lại (hiện ra biểu tượng ảnh hỏng)."""
    b64 = base64.b64encode(png).decode("ascii")
    st.markdown(f'<img src="data:image/png;base64,{b64}" style="width:100%;border:1px solid #e2e8f0;border-radius:6px">'
                f'<div style="font-size:.8rem;color:#64748b;margin:.2rem 0 .6rem">{_html.escape(caption)}</div>',
                unsafe_allow_html=True)


def _render_kb_source_view(s, kp):
    """Mở dữ liệu của nguồn từ KB của hệ thống: PDF theo trang (tô sáng dòng phụ gia), HTML/.doc dạng chữ, và đoạn KB đã truy xuất."""
    rel = s.get("local_file")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel) if rel else None
    if not (path and os.path.isfile(path)):
        st.info("Hệ thống chưa lưu bản gốc của nguồn này trong KB. Nội dung bên dưới là đoạn trong KB mà câu trả lời đã dùng.")
        st.write(s.get("quote") or s.get("scope") or "")
        return
    name = os.path.basename(path)
    if name.lower().endswith(".pdf"):
        n_pages = m_evidence.pdf_page_count(rel)
        code = page_no = None
        hits = ()
        if s.get("ins_codes"):
            code = (st.selectbox("Mã INS", s["ins_codes"], key=kp + "-code") if len(s["ins_codes"]) > 1 else s["ins_codes"][0])
            hits = m_evidence.locate_pages(rel, code, locator=s.get("locator", ""))
        if hits:
            labels = {p: f"Trang {p}" for p, _ in hits}
            page_no = (st.radio("Trang có mục này", list(labels), format_func=labels.get, horizontal=True, key=kp + "-page")
                       if len(hits) > 1 else hits[0][0])
            png = m_evidence.render_pdf_row(rel, page_no, code)
            note = "dòng được tô sáng; vị trí do hệ thống dò theo mã INS + tên phụ gia, không phải số trang kiểm chứng thủ công"
        else:
            if s.get("ins_codes"):
                st.caption("Không dò được dòng của mã này; bạn có thể xem từng trang bên dưới.")
            page_no = st.number_input("Trang", 1, max(n_pages, 1), 1, key=kp + "-pg")
            png = m_evidence.render_pdf_page(rel, int(page_no))
            note = "toàn trang"
        if png:
            _show_png(png, f"{name} — trang {page_no}/{n_pages} ({note})")
    else:
        body = m_evidence.local_text(rel)
        if body:
            st.text_area(name, body, height=380, disabled=True, key=kp + "-txt")
    with open(path, "rb") as fh:
        st.download_button(f"📁 Tải file {name}", data=fh.read(), file_name=name, key=kp + "-dl", on_click="ignore")


def render_source_list(sources, key="x"):
    """Thẻ nguồn + chọn một nguồn để mở dữ liệu ngay trong app (từ KB của hệ thống, không dẫn ra link ngoài).
    `key` ổn định theo từng tin nhắn để trạng thái chọn không bị đặt lại khi trang chạy lại."""
    cards = []
    for s in sources:
        origin = _source_origin(s["url"])
        origin_html = f' · <span style="opacity:.7">nguồn gốc: {_html.escape(origin)}</span>' if origin else ""
        scope = f'<div><b>Mục / điều khoản:</b> {_html.escape(s["scope"])}</div>' if s["scope"] else ""
        quote = (f'<blockquote style="margin:.4rem 0;padding:.4rem .8rem;border-left:3px solid #0d9488;'
                 f'font-size:.88rem;color:#1e293b;background:#f8fafc;border-radius:4px">{_html.escape(s["quote"])}</blockquote>') if s["quote"] else (
                 '<div style="font-size:0.82rem;color:#64748b;margin:.2rem 0"><i>(Nguồn đối chiếu theo mục / điều khoản nêu trên)</i></div>')
        verified = " · ✅ đã đối chiếu với văn bản gốc" if s["verified"] else ""
        cards.append(
            f'<div id="fh-src-{s["n"]}" style="border:1px solid rgba(128,128,128,.35);border-radius:8px;'
            f'padding:.6rem .8rem;margin:.4rem 0">'
            f'<div><b>[{s["n"]}]</b> {_html.escape(s["title"])}</div>{scope}{quote}'
            f'<div style="font-size:.85em;opacity:.85">🛡️ Mức tin cậy nguồn: {_html.escape(s["tier_label"])}{verified}'
            f' · <code>{_html.escape(s["locator"] or s["citation_id"])}</code>{origin_html}</div></div>'
        )
    st.markdown("".join(cards), unsafe_allow_html=True)
    kp = f"fh-src-{key}"
    labels = {s["n"]: f"[{s['n']}] {s['title'][:70]}" + ("" if s.get("local_file") else " (chỉ có đoạn trích)") for s in sources}
    chosen = st.selectbox("📂 Mở dữ liệu nguồn trong KB của hệ thống", list(labels), index=None,
                          format_func=labels.get, placeholder="Chọn một nguồn để xem…", key=kp + "-pick")
    if chosen is not None:
        _render_kb_source_view(next(s for s in sources if s["n"] == chosen), f"{kp}-{chosen}")


def render_recognition_summary(rows, key="x", compact=False):
    """Khung 'máy đọc được gì trên nhãn' cho MỌI phụ gia: chữ trên ảnh → chất được nhận diện → ADI → nguồn.
    Dựng bằng code từ kết quả liên kết (không do LLM viết), để người dùng/người kiểm duyệt đối chiếu ngay với ảnh."""
    if not rows:
        return
    linked = [r for r in rows if r["status"] == "linked"]
    unlinked = [r for r in rows if r["status"] != "linked"]
    title = f"🔎 Máy đọc được trên nhãn: {len(linked)} phụ gia được nhận diện" + (
        f", {len(unlinked)} cụm chưa nhận diện được" if unlinked else "")
    options = {}
    with st.expander(title, expanded=not compact):
        st.caption("Chữ ở cột trái là chữ máy đọc từ ảnh. Hãy so với ảnh nhãn để biết máy có đọc nhầm không.")
        lines = []
        for i, r in enumerate(rows):
            raw = r["raw"].replace("`", "'")
            if r["status"] != "linked":
                lines.append(f"- Chữ trên ảnh `{raw}` → ⚠️ **chưa nhận diện được trong cơ sở tri thức** "
                             f"(có thể OCR đọc sai); hệ thống không đoán.")
                continue
            how = f" ({r['how']})" if r["how"] else ""
            adi = f" Theo dữ liệu: {r['adi_text']}." if r["adi_text"] else " Chưa có ADI trong cơ sở tri thức."
            src = ""
            if r["sources"]:
                src = " Nguồn: " + "; ".join(f"*{s['title']}*" for s in r["sources"]) + "."
            lines.append(f"- Chữ trên ảnh `{raw}` → nhận diện là **{r['ins_code']} — {r['name']}**{how}.{adi}{src}")
            for j, s in enumerate(r["sources"]):
                options[(i, j)] = f"{r['ins_code']} · {s['role']} — {s['title'][:60]}" + ("" if s.get("local_file") else " (chỉ có đoạn trích)")
        st.markdown("\n".join(lines))
        if options:
            kp = f"fh-recog-{key}"
            chosen = st.selectbox("📂 Mở nguồn để đối chiếu", list(options), index=None, format_func=options.get,
                                  placeholder="Chọn một dòng để xem dữ liệu nguồn trong KB…", key=kp + "-pick")
            if chosen is not None:
                _render_kb_source_view(rows[chosen[0]]["sources"][chosen[1]], f"{kp}-{chosen[0]}-{chosen[1]}")


def render_nli_faithfulness_widget(cited, evaluation, alerts=None, regs=None, passages=None, label_text=""):
    """Đối chiếu nhận định trong câu trả lời với BẰNG CHỨNG THẬT mà mô hình được đọc: đoạn nguồn đã truy xuất +
    văn bản nhãn. (Trước đây "bằng chứng" là lời khuyên/cảnh báo do chính guardrail sinh ra — hệ thống tự đối chiếu
    với chính nó.) Không truyền LLM nên đây là ĐỐI CHIẾU TỪ KHÓA, huy hiệu ghi đúng tên đó."""
    if not cited or not cited.get("text"):
        return None
    try:
        from modules.nli_verifier import verify_answer_faithfulness
        evidence_parts = [str(label_text or "")] + [str(p.get("text", "")) for p in (passages or []) if isinstance(p, dict)]
        ev_text = "\n".join(part for part in evidence_parts if part.strip())
        nli_res = verify_answer_faithfulness(cited["text"], ev_text, trigger_only=True)
        st.markdown(nli_res["badge_html"], unsafe_allow_html=True)
        return nli_res
    except Exception:
        return None


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
            "Mức khớp (theo luật)": _fmt_score(r.get("confidence_tau")),
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


# ═════════════════════════════════════════════════════════════════════════
# GIAO DIỆN HỎI ĐÁP DẠNG CHAT (kiểu Gemini)
# Người dùng nhập tự do: ảnh nhãn (nút +), câu hỏi, hồ sơ sức khỏe — trong cùng một ô.
# Khi đủ ảnh + câu hỏi → chạy đúng pipeline cũ (OCR → chuẩn hóa → truy xuất → guardrail → trả lời).
# ═════════════════════════════════════════════════════════════════════════
import modules.chat_intake as m_chat

CHAT_CSS = """
<style>
[data-testid="stMainBlockContainer"],.main .block-container{max-width:980px!important;margin:0 auto!important;padding-bottom:7rem!important}
[data-testid="stBottomBlockContainer"]{max-width:980px!important;margin:0 auto!important;padding-bottom:1.2rem!important}
[data-testid="stBottom"],[data-testid="stBottom"] > div{background:transparent!important}
[data-testid="stChatInput"]{border-radius:32px!important;background:#fff!important;border:1px solid #e2e8f0!important;
  box-shadow:0 6px 24px rgba(15,23,42,.10)!important;padding:6px 8px 6px 12px!important}
[data-testid="stChatInput"]:focus-within{border-color:#c7d2fe!important;box-shadow:0 6px 28px rgba(79,70,229,.14)!important}
[data-testid="stChatInput"] textarea{font-size:.98rem!important;background:transparent!important}
[data-testid="stChatMessage"]{background:transparent!important;padding:.4rem 0!important;gap:.8rem}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){flex-direction:row-reverse}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{
  background:#e9eef6;border-radius:20px 4px 20px 20px;padding:10px 16px;max-width:78%;margin-left:auto;flex:0 1 auto}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p{font-size:.92rem!important}
.fh-greet{margin:9vh 0 26px 0}
.fh-greet .hi{font-size:2.5rem;font-weight:800;letter-spacing:-.03em;line-height:1.15;
  background:linear-gradient(90deg,#0d9488,#4f46e5 55%,#db2777);-webkit-background-clip:text;background-clip:text;color:transparent}
.fh-greet .sub{font-size:2.1rem;font-weight:700;letter-spacing:-.03em;color:#c4c7c5;line-height:1.2;margin-top:4px}
.fh-tips{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0 8px}
.fh-tip{background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:14px 16px;font-size:.82rem;line-height:1.55;color:#334155;min-height:96px}
.fh-tip b{display:block;margin-bottom:4px;color:#0f172a;font-size:.84rem}
.fh-tip i{color:#475569}
.fh-ctx{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 0 0}
.fh-ctx span{font-size:.74rem;font-weight:500;color:#334155;background:#fff;border:1px solid #e2e8f0;border-radius:999px;padding:4px 12px}
.fh-ctx span.miss{color:#92400e;background:#fffbeb;border-color:#fde68a}
.fh-ctx span.ok{color:#065f46;background:#ecfdf5;border-color:#a7f3d0}
.st-key-fh_inputbar{position:relative}
.st-key-fh_inputbar [data-testid="stChatInput"] textarea{padding-right:150px!important}
.st-key-fh_inputbar .st-key-llm_provider_select{position:absolute!important;right:60px;bottom:16px;z-index:20;width:104px!important}
/* bỏ viền + nền xám của luật chung cho ô chọn (role="group"), để ô mô hình nằm phẳng trong thanh chat như Gemini */
.st-key-fh_inputbar .st-key-llm_provider_select div[role="group"],
.st-key-fh_inputbar .st-key-llm_provider_select div[role="group"]:focus-within{background:transparent!important;border:none!important;box-shadow:none!important}
.st-key-fh_inputbar .st-key-llm_provider_select div[data-baseweb="select"]{min-height:36px!important}
.st-key-fh_inputbar .st-key-llm_provider_select div[data-baseweb="select"] > div{background:transparent!important;border:none!important;box-shadow:none!important;
  min-height:34px!important;border-radius:999px!important;cursor:pointer}
.st-key-fh_inputbar .st-key-llm_provider_select div[data-baseweb="select"] > div:hover{background:#f1f5f9!important}
.st-key-fh_inputbar .st-key-llm_provider_select div[data-baseweb="select"] *{font-size:.86rem!important;font-weight:600;color:#1f2937!important;justify-content:flex-end}
.fh-hint{text-align:center;font-size:.72rem;color:#64748b;margin-top:6px}
.fh-hint b{color:#334155}
.fh-prod{display:flex;align-items:center;gap:12px;padding:10px 12px;margin:0 0 10px 0;border-radius:14px;background:#f8fafc;border:1px solid #e2e8f0}
.fh-prod img{width:56px;height:56px;object-fit:cover;border-radius:10px;border:1px solid #e2e8f0}
.fh-prod .t{font-size:.84rem;line-height:1.45;color:#334155}
.fh-prod .t b{color:#0f172a}
.fh-prod .t small{display:block;color:#64748b;font-size:.74rem}
</style>
"""
st.markdown(CHAT_CSS, unsafe_allow_html=True)


def _new_ctx():
    return {
        "image": None, "image_name": "", "sample_key": "custom", "product_name": "",
        "ocr_result": None, "mentions": [], "profile": m_chat.empty_profile(), "pending_question": "",
        "image_no": 0, "image_thumb": None, "display_name": "", "image_count": 0, "awaiting_image_choice": False,
        "annotated_images": {},   # số ảnh → ảnh có khung OCR (lưu MỘT lần cho mọi câu trả lời trên ảnh đó)
        "linked_cache": None,     # (số ảnh, kết quả liên kết phụ gia) — chỉ phụ thuộc chữ trên nhãn
        "last_focus": [],         # đối tượng câu hỏi gần nhất (để hiểu câu nối tiếp "chất đó…")
        "last_question": "",      # câu hỏi gần nhất — trả lời lại khi người dùng sửa thông tin nhãn
        "label_corrections": [],  # các câu sửa nhãn của người dùng cho ảnh hiện tại
    }


if "chat" not in st.session_state:
    st.session_state["chat"] = []
if "ctx" not in st.session_state:
    st.session_state["ctx"] = _new_ctx()
chat = st.session_state["chat"]
ctx = st.session_state["ctx"]


def _thumb_bytes(img, size=480):
    try:
        t = img.convert("RGB").copy()
        t.thumbnail((size, size))
        buf = io.BytesIO()
        t.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    except Exception:
        return None


# ── Sidebar: chế độ làm việc (Hỏi đáp / Báo cáo nghiên cứu) ──
WORKSPACE_DEMO = "💬 Hỏi đáp nhãn thực phẩm"
WORKSPACE_REPORT = "🔬 Báo cáo nghiên cứu"
with st.sidebar:
    st.markdown("---")
    app_mode = st.radio(
        "Chế độ:", [WORKSPACE_DEMO, WORKSPACE_REPORT],
        key="app_mode", label_visibility="collapsed",
        help="Hỏi đáp: trò chuyện với hệ thống trên ảnh nhãn. Báo cáo nghiên cứu: kết quả đánh giá 3 cấu hình A/B/C.",
    )

# ═════════════════════════════════════════════════════════════════════════
# TAB "🔬 Báo cáo nghiên cứu": giao diện của commit f2694d50 (chọn ảnh mẫu/tải ảnh, hồ sơ, câu hỏi → kết quả 4 tab).
# Giữ nguyên mã giao diện và xử lý của commit đó; chỉ lấy NHANES/FNDDS từ bản nạp nền. Không có thanh nhập chat.
# ═════════════════════════════════════════════════════════════════════════




if app_mode == WORKSPACE_REPORT:
    # Báo cáo nghiên cứu = KẾT QUẢ ĐÁNH GIÁ BỘ QA (đọc từ các file kết quả của scripts_research), không phải công cụ phân tích từng câu
    import modules.research_report as m_report
    importlib.reload(m_report)  # server Streamlit đang chạy có thể giữ bản cũ của mô-đun trong bộ nhớ
    m_report.render_research_report(DATA_DIR)
    st.stop()

with st.sidebar:
    if st.button("✨ Trò chuyện mới", width="stretch"):
        st.session_state["chat"] = []
        st.session_state["ctx"] = _new_ctx()
        st.rerun()


# ── Xử lý một lượt hội thoại ──
def _load_image_into_ctx(img, name, sample_key, product_name, blocks, source=None):
    """Chạy OCR ngay khi nhận ảnh (một lần), kiểm tra ảnh có phải nhãn thành phần không."""
    with st.spinner(f"Đang đọc nhãn trong ảnh {name} (OCR)..."):
        ocr = run_perception_pipeline(source if source is not None else img, sample_key=sample_key, force_real_ocr=True)
    if ocr.get("is_ocr_unavailable") or not str(ocr.get("full_text", "")).strip():
        blocks.append(("error",
                       f"Không đọc được chữ trên ảnh **{name}** (OCR không khả dụng hoặc ảnh không có chữ). "
                       "Hãy chụp lại rõ nét, đủ sáng, thẳng góc với phần **Thành phần**."))
        return False
    mentions = parse_ingredients_from_ocr(ocr["full_text"], include_allergen_statements=True)
    if not mentions:
        blocks.append(("error",
                       f"Ảnh **{name}** đọc được chữ nhưng không tìm thấy danh sách **thành phần** — có thể đây không phải "
                       "nhãn thực phẩm hoặc chưa chụp đúng mặt có bảng thành phần. Vui lòng gửi ảnh khác."))
        return False
    if not product_name or product_name.lower().endswith(m_chat.PICKER_IMAGE_EXT):
        # tên dùng cho pipeline (đối chiếu sự cố/dinh dưỡng) giữ nguyên cách cũ: ghép 3 thành phần đầu
        guess = " ".join(m.get("raw_mention", "") for m in mentions[:3]).strip()
        product_name = guess[:80] or "Sản phẩm thực phẩm"
    ctx["image_count"] = int(ctx.get("image_count", 0)) + 1
    label_name = m_chat.guess_product_name(ocr["full_text"])
    ctx.update({
        "image": img, "image_name": name, "sample_key": sample_key, "product_name": product_name,
        "ocr_result": ocr, "mentions": mentions, "image_no": ctx["image_count"],
        "image_thumb": _thumb_bytes(img, 160), "awaiting_image_choice": False,
        # không đoán được tên từ nhãn → dùng tên file ảnh
        "display_name": label_name or m_chat.display_name_from_file(name),
        "name_from_label": bool(label_name),
        "linked_cache": None,
        "label_corrections": [],
    })
    ctx.setdefault("annotated_images", {})[ctx["image_no"]] = ocr.get("annotated_image")
    blocks.append(("success", f"Đã nhận **ảnh #{ctx['image_no']}** ({_html.escape(_short_file(name))})"
                              + (f" — sản phẩm: **{_html.escape(label_name)}**" if label_name
                                 else " — chưa đọc được tên sản phẩm trên nhãn, dùng **tên file** để gọi sản phẩm")
                              + f". Đọc được {len(mentions)} cụm thành phần (độ tin cậy OCR {ocr.get('avg_confidence', 0):.2f})."))
    return True


def _run_analysis(question, reused=False):
    persona_raw = m_chat.profile_to_persona(ctx["profile"])
    persona = enrich_persona_with_nhanes(persona_raw, DATA_DIR, biomarkers_cache=background_result("nhanes"))
    ocr = ctx["ocr_result"]
    # Câu nối tiếp ("chất đó…", "nó…", "còn … thì sao") → ghép đối tượng của câu hỏi trước vào câu gửi pipeline
    pipeline_question, followup_terms = m_chat.resolve_followup(question, ctx.get("last_focus"))
    # Liên kết phụ gia chỉ phụ thuộc chữ trên nhãn → tính một lần cho mỗi ảnh, dùng lại cho các câu hỏi sau
    cache = ctx.get("linked_cache")
    cached_links = cache[1] if (cache and cache[0] == ctx.get("image_no")) else None
    with st.spinner("Đang chuẩn hóa thành phần, truy xuất quy chuẩn và sinh câu trả lời..."):
        grounded = run_grounded_pipeline(
            ctx["mentions"], ocr["avg_confidence"], ctx["product_name"], persona,
            pipeline_question, kb_additives, hazards_list, background_result("fndds"), off_products,
            linked_results=cached_links, label_text=ocr.get("full_text", ""),
        )
    if cached_links is None:
        _persona_fields = ("constraint_type", "constraint_rule", "constraint_detail",
                           "priority_tier", "priority_text", "is_refined_oil_exempt")
        ctx["linked_cache"] = (ctx.get("image_no"), [
            {k: v for k, v in it.items() if k not in _persona_fields} for it in grounded["linked_results"]])
    # câu nối tiếp hiểu theo đối tượng của câu vừa hỏi (khi câu đó chính là câu nối tiếp thì giữ đối tượng đã ghép)
    ctx["last_question"] = question     # người dùng sửa thông tin nhãn → trả lời lại câu này
    focus = followup_terms or m_chat.focus_from_turn(
        grounded.get("qa"), question, grounded.get("linked_results"), grounded.get("llm_answer") or "")
    if focus:
        ctx["last_focus"] = focus
    # bản OCR gọn (bỏ ảnh) — ảnh có khung OCR lưu một lần theo số ảnh trong ctx["annotated_images"]
    ocr_light = {k: v for k, v in ocr.items() if k != "annotated_image"}
    return {
        "raw_text": ocr["full_text"],
        "mentions": ctx["mentions"],
        "avg_ocr_conf": ocr["avg_confidence"],
        "followup_terms": followup_terms,
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
        "ocr_result": ocr_light,
        "selected_persona": persona,
        "profile_text": m_chat.profile_summary(ctx["profile"]),
        "profile_declared": bool(ctx["profile"].get("declared")),
        "user_query": question,
        "selected_prod_name": ctx["product_name"],
        "image_name": ctx["image_name"],
        "image_no": ctx.get("image_no", 0),
        "image_thumb": ctx.get("image_thumb"),
        "display_name": ctx.get("display_name", ""),
        "name_from_label": bool(ctx.get("name_from_label")),
        "reused_image": reused,
    }


_PROFILE_PROMPT = (
    "Để đánh giá đúng cho bạn, vui lòng cho biết **hồ sơ sức khỏe** của người sẽ dùng sản phẩm — "
    "tuổi, giới tính, bệnh nền, dị ứng, mang thai/cho con bú. Ví dụ: *“Tôi 40 tuổi, nữ, bị tăng huyết áp, dị ứng tôm”*.\n\n"
    "Nếu không có bệnh nền và không dị ứng, hãy trả lời **“không có”** (hoặc **“bỏ qua”**)."
)

_HELP_TEXT = (
    "Mình là **FoodHealthQA** — trợ lý sàng lọc **phụ gia, dị nguyên và dinh dưỡng** từ ảnh nhãn thực phẩm.\n\n"
    "Bạn chỉ cần gõ vào ô bên dưới (có thể gộp trong một tin nhắn hoặc gửi từng phần):\n"
    "1. **Ảnh nhãn** — bấm nút đính kèm trong ô nhập (" + m_chat.IMAGE_FORMAT_HINT + ").\n"
    "2. **Câu hỏi về sản phẩm** — vd *“Sản phẩm này có chứa dị nguyên nào không?”*\n"
    "3. **Hồ sơ sức khỏe** — vd *“Tôi 40 tuổi, bị tăng huyết áp, dị ứng tôm”*.\n\n"
    "Câu hỏi không kèm ảnh mới sẽ được hiểu là hỏi trên **ảnh gửi gần nhất**.\n\n"
    "Lệnh nhanh: *“xóa hồ sơ”*, *“xóa ảnh”*, *“dùng ảnh cũ”*, *“làm mới”*."
)


def _short_file(filename, max_len=40):
    """Tên file hiển thị (giữ đuôi); tên quá dài rút gọn ở giữa."""
    name = str(filename or "")
    return name if len(name) <= max_len else name[: max_len - 15] + "…" + name[-14:]


def _image_label():
    name = ctx.get("display_name")
    return f"ảnh #{ctx.get('image_no', 0)} — {name}" if name else f"ảnh #{ctx.get('image_no', 0)} ({_short_file(ctx.get('image_name', ''))})"


def handle_turn(text, files, use_old_image=False):
    text = (text or "").strip()
    user_msg = {"role": "user", "text": text, "thumbs": []}
    blocks = []          # (loại, nội dung markdown) cho tin nhắn trả lời
    result = None
    choice = False

    parsed = m_chat.parse_message(text, foodon_entities) if text else None
    kind = parsed["kind"] if parsed else "empty"
    if kind == "use_old_image":
        use_old_image = True

    if kind == "reset":
        st.session_state["chat"] = []
        st.session_state["ctx"] = _new_ctx()
        return

    images = list(files or [])
    for f in images:
        try:
            user_msg["thumbs"].append(_thumb_bytes(Image.open(io.BytesIO(f.getvalue()))))
        except Exception:
            user_msg["thumbs"].append(None)
            user_msg["text"] = (user_msg["text"] + f"\n\n📎 {f.name}").strip()
    if use_old_image and not text:
        user_msg["text"] = "✅ Dùng ảnh cũ"
    chat.append(user_msg)

    profile_changed = False
    if kind == "clear_profile":
        ctx["profile"] = m_chat.empty_profile()
        blocks.append(("info", "Đã xóa hồ sơ sức khỏe. Trước câu hỏi tiếp theo, hệ thống sẽ đề nghị bạn khai báo lại hồ sơ."))
    elif kind == "skip_profile":
        ctx["profile"], _ = m_chat.merge_profile(m_chat.empty_profile(), {"healthy": True, "no_allergy": True})
        profile_changed = True
        blocks.append(("success", "Đã ghi nhận: **không có bệnh nền, không dị ứng**."))
    elif kind == "clear_image":
        keep = ("profile", "pending_question", "image_count", "annotated_images", "last_focus")
        ctx.update({k: v for k, v in _new_ctx().items() if k not in keep})
        blocks.append(("info", "Đã bỏ ảnh nhãn hiện tại. Hãy đính kèm ảnh mới để tiếp tục."))
    elif kind == "greeting":
        blocks.append(("markdown", "Xin chào 👋 " + _HELP_TEXT))
    elif kind == "thanks":
        blocks.append(("markdown", "Rất vui được giúp bạn! Bạn có thể hỏi tiếp về sản phẩm này hoặc gửi ảnh nhãn khác."))
    elif kind == "help":
        blocks.append(("markdown", _HELP_TEXT))

    # 1. Ảnh đính kèm: ảnh lỗi → báo lỗi; không tự ý trả lời bằng ảnh cũ
    new_image = False
    image_failed = False
    if images:
        if len(images) > 1:
            blocks.append(("warning", f"Bạn gửi {len(images)} ảnh — hệ thống chỉ phân tích **một nhãn mỗi lần**, đang dùng ảnh đầu tiên."))
        first = images[0]
        img, err = m_chat.validate_image(first.getvalue(), first.name)
        if err:
            blocks.append(("error", err))
        else:
            new_image = _load_image_into_ctx(img, first.name, "custom", "", blocks)
        image_failed = not new_image

    # 2. Văn bản: lỗi nhập liệu → báo lỗi, không áp dụng gì từ tin nhắn này
    question = ""
    if parsed and kind in ("message", "invalid"):
        for e in parsed["errors"]:
            blocks.append(("error", e))
        for w in parsed["warnings"]:
            blocks.append(("warning", w))
        if not parsed["errors"]:
            upd = parsed["profile"]["update"]
            if upd:
                before = m_chat.profile_summary(ctx["profile"]) if ctx["profile"].get("declared") else ""
                ctx["profile"], notes = m_chat.merge_profile(ctx["profile"], upd)
                after = m_chat.profile_summary(ctx["profile"])
                profile_changed = True
                for n in notes:
                    blocks.append(("warning", n))
                if notes or not before:
                    blocks.append(("success", "Hồ sơ đang áp dụng: " + after))
                elif before != after:
                    blocks.append(("success", f"Đã cập nhật hồ sơ: ~~{_html.escape(before)}~~ → **{_html.escape(after)}**"))
                else:
                    blocks.append(("info", "Hồ sơ không thay đổi: " + after))
            question = parsed["question"]

    # 2b. Người dùng sửa thông tin nhãn ngay trong chat ("không có E621", "mã là E330 chứ không phải E33",
    #     "thành phần đúng là: …") → áp lên thành phần đã đọc rồi trả lời lại câu hỏi gần nhất. Không hiện danh sách
    #     thành phần, không bắt xác nhận.
    correction = (parsed or {}).get("correction")
    if correction:
        if ctx["ocr_result"] is None:
            blocks.append(("info", "Chưa có ảnh nhãn nào để sửa thông tin. Vui lòng đính kèm ảnh nhãn sản phẩm."))
        else:
            new_mentions, done, missing = m_chat.apply_label_correction(
                ctx["mentions"], correction, parse_ingredients_from_ocr)
            if done:
                ctx["mentions"] = new_mentions
                ctx["linked_cache"] = None          # thành phần đổi → liên kết phụ gia tính lại
                ctx.setdefault("label_corrections", []).append(correction)
                blocks.append(("success", "Đã cập nhật theo thông tin bạn sửa: " + "; ".join(done) + "."))
                if not question:
                    question = ctx.get("last_question", "")
                    if question:
                        blocks.append(("info", f"Trả lời lại câu hỏi *“{_html.escape(question)}”* với thông tin đã sửa."))
                    else:
                        blocks.append(("info", "Bạn muốn hỏi gì về sản phẩm này?"))
            if missing:
                blocks.append(("info", "Không thấy " + ", ".join(f"**{_html.escape(x)}**" for x in missing)
                               + " trong phần thành phần đọc được từ ảnh, nên không có gì để sửa ở mục này."))

    # 3. Quyết định bước tiếp theo
    if use_old_image:
        ctx["awaiting_image_choice"] = False
        if ctx["ocr_result"] is None:
            blocks.append(("error", "Chưa có ảnh nào được gửi trước đó. Vui lòng đính kèm ảnh nhãn sản phẩm."))
        elif not question and not ctx["pending_question"]:
            blocks.append(("info", f"Tiếp tục dùng **{_html.escape(_image_label())}**. Bạn muốn hỏi gì về sản phẩm này?"))
    if not question and ctx["pending_question"] and (new_image or profile_changed or use_old_image):
        question = ctx["pending_question"]
        blocks.append(("info", f"Tiếp tục câu hỏi đang chờ: *“{_html.escape(question)}”*"))
    if question and not image_failed and not new_image and ctx.get("awaiting_image_choice"):
        ctx["awaiting_image_choice"] = False   # hỏi tiếp mà không gửi ảnh mới → hiểu là hỏi trên ảnh cũ
    has_old_image = ctx["ocr_result"] is not None
    has_image = has_old_image and not image_failed
    has_profile = bool(ctx["profile"].get("declared"))
    oos = m_chat.abstention.detect_out_of_scope(question) if question else []
    action = m_chat.next_action(has_image, question, oos, has_profile=has_profile)
    if action in ("need_image", "need_profile", "need_image_profile"):
        ctx["pending_question"] = question
    if action == "run":
        ctx["pending_question"] = ""
        result = _run_analysis(question, reused=not new_image)
    elif action == "out_of_scope_no_image":
        blocks.append(("warning", m_chat.out_of_scope_reply(question)))
    elif action in ("need_image", "need_image_profile") and image_failed and has_old_image:
        # ảnh mới lỗi nhưng đã có ảnh trước đó → hỏi người dùng muốn dùng ảnh cũ hay gửi ảnh khác
        ctx["awaiting_image_choice"] = True
        choice = True
        blocks.append(("info", f"Câu hỏi *“{_html.escape(question)}”* đã được giữ lại. Bạn muốn trả lời trên "
                               f"**{_html.escape(_image_label())}** đã gửi trước đó, hay gửi ảnh khác?"))
    elif action in ("need_image", "need_image_profile") and image_failed:
        blocks.append(("info", "Câu hỏi đã được giữ lại. Vui lòng gửi **ảnh nhãn khác** (" + m_chat.IMAGE_FORMAT_HINT + ")."
                       + (" Đồng thời cho biết **hồ sơ sức khỏe** của bạn." if action == "need_image_profile" else "")))
    elif action == "need_image":
        blocks.append(("info", "Đã nhận câu hỏi. Vui lòng **đính kèm ảnh nhãn sản phẩm** bằng nút đính kèm trong ô nhập ("
                               + m_chat.IMAGE_FORMAT_HINT + ") — hệ thống sẽ trả lời ngay khi có ảnh."))
    elif action in ("need_profile", "need_image_profile"):
        missing_img = "\n\nNgoài ra, hãy **đính kèm ảnh nhãn sản phẩm**." if action == "need_image_profile" else ""
        blocks.append(("info", _PROFILE_PROMPT + missing_img))
    elif (new_image or profile_changed) and not blocks_has_error(blocks):
        if not has_image:
            blocks.append(("info", "Đã ghi nhận. Hãy **đính kèm ảnh nhãn** và đặt câu hỏi về sản phẩm để bắt đầu phân tích."))
        else:
            blocks.append(("info", "Bạn muốn hỏi gì về sản phẩm? Ví dụ: *“Có chứa dị nguyên nào không?”*, "
                                   "*“Phụ gia nào cần lưu ý với tôi?”*, *“Có an toàn cho trẻ 5 tuổi không?”*"
                           + ("" if has_profile else "\n\n" + _PROFILE_PROMPT)))
    elif image_failed and not question and not ctx["pending_question"]:
        if has_old_image:
            blocks.append(("info", f"Ảnh trước đó (**{_html.escape(_image_label())}**) vẫn được giữ. "
                                   "Bạn có thể hỏi tiếp trên ảnh đó hoặc gửi ảnh khác."))
    elif not blocks:
        blocks.append(("info", "Hãy nhập câu hỏi hoặc đính kèm ảnh nhãn để bắt đầu."))

    chat.append({"role": "assistant", "blocks": blocks, "result": result, "choice": choice})


def blocks_has_error(blocks):
    return any(k == "error" for k, _ in blocks)


# ── Hiển thị kết quả phân tích (giữ nguyên nội dung của giao diện cũ) ──
# Ý nghĩa của từng nhãn khi chấm bộ QA (dùng ở phần chi tiết)
QA_LABEL_MEANING = {
    "SAFE": "không có luật nào khớp với phần chữ đọc được và hồ sơ — chưa phải xác nhận an toàn",
    "CAUTION": "có phụ gia cần hạn chế hoặc số dinh dưỡng vượt ngưỡng với hồ sơ",
    "UNSAFE": "có dị nguyên trùng dị ứng đã khai (kể cả \"có thể chứa\")",
    "CRITICAL_ALERT": "có chất được quy chuẩn ghi là chống chỉ định với bệnh đã khai",
    "ABSTAIN": "chưa đủ cơ sở để kết luận hoặc câu hỏi ngoài phạm vi",
}


def _allergen_vi(key):
    try:
        from modules.constraint_model import ALLERGEN_KEYWORDS
        words = ALLERGEN_KEYWORDS.get(key) or []
        return next((w for w in words if re.search(r"[à-ỹđ]", w)), words[0] if words else key)
    except Exception:
        return key


def _join_vi(items, limit=3):
    items = list(dict.fromkeys(i for i in items if i))[:limit]
    return ", ".join(items[:-1]) + " và " + items[-1] if len(items) > 1 else (items[0] if items else "")


def _allergy_hits(alert, allergies):
    """Dị ứng đã khai nào khớp với cảnh báo — để gọi tên gọn ("sữa") thay vì cả cụm chữ OCR."""
    if alert.get("allergen_key"):
        return [alert["allergen_key"]]
    try:
        from modules.constraint_model import ALLERGEN_KEYWORDS, contains_label_term
    except Exception:
        return []
    text = f"{alert.get('additive', '')} {alert.get('message', '')}".casefold()
    return [a for a in allergies if any(contains_label_term(text, kw) for kw in ALLERGEN_KEYWORDS.get(a, [a]))]


def verdict_view(evaluation, nutri_warnings=None, allergies=None):
    """Kết luận của bộ luật viết sao cho ĐỌC VÀO LÀ HIỂU: (biểu tượng, tên nhãn, ý nghĩa cụ thể, kiểu khung).
    Chỉ đổi chữ hiển thị — trạng thái và nhãn chấm bộ QA giữ nguyên."""
    status = str(evaluation.get("status") or "").upper()
    alerts = evaluation.get("alerts") or []
    allergies = [str(a) for a in (allergies or [])]
    if status == "CRITICAL_ALERT":
        what = _join_vi([a.get("additive", "") for a in alerts])
        return ("🔴", "Chống chỉ định với bệnh của bạn",
                f"Nhãn có {what or 'một chất'}, được quy chuẩn ghi là chống chỉ định với bệnh bạn đã khai.", "error")
    if status == "WARNING":
        def _names(group):
            keys = [k for a in group for k in _allergy_hits(a, allergies)] or allergies
            return _join_vi([_allergen_vi(k) for k in keys]) or "dị nguyên"
        age = [a for a in alerts if a.get("label_age_rule")]
        alerts = [a for a in alerts if not a.get("label_age_rule")]
        pal = [a for a in alerts if a.get("regulation") == "FAO_WHO_JEMRA_PART_3" or "PAL" in str(a.get("message", ""))]
        direct = [a for a in alerts if a not in pal]
        if age:
            said = _join_vi([f"\"{a.get('label_excerpt', '')}\"" for a in age])
            also = f" Ngoài ra nhãn có {_names(direct)}, trùng với dị ứng bạn đã khai." if direct else ""
            return ("🟠", "Nhãn ghi không dùng cho độ tuổi của bé",
                    f"Nhãn ghi {said} — bé chưa đủ tuổi dùng theo hướng dẫn của nhà sản xuất.{also}", "error")
        if direct:
            return ("🟠", "Có dị nguyên bạn bị dị ứng",
                    f"Nhãn có {_names(direct)}, trùng với dị ứng bạn đã khai.", "error")
        return ("🟠", "Có thể nhiễm chéo dị nguyên",
                f"Nhãn ghi \"có thể chứa\" {_names(pal)} — trùng với dị ứng bạn đã khai "
                "(không có trong thành phần, nhưng có thể lẫn trong quá trình sản xuất).", "warning")
    if status == "CAUTION":
        infant = [a for a in alerts if a.get("infant_rule")]
        if infant and len(infant) == len(alerts):
            return ("🟡", "Có phụ gia không thuộc danh mục cho trẻ dưới 12 tháng",
                    f"Nhãn có {_join_vi([a.get('additive', '') for a in infant])}, không nằm trong danh mục phụ gia được "
                    "phép cho thực phẩm trẻ đến 12 tháng tuổi (TT 24/2019 Phụ lục 2A; Codex GSFA nhóm 13.1, 13.2).", "warning")
        if alerts:
            return ("🟡", "Nên hạn chế",
                    f"Có {_join_vi([a.get('additive', '') for a in alerts])} cần hạn chế với hồ sơ của bạn.", "warning")
        first = str((nutri_warnings or [""])[0])
        first = first if len(first) <= 160 else first[:157] + "…"
        return ("🟡", "Vượt ngưỡng dinh dưỡng", first or "Số liệu dinh dưỡng trên nhãn vượt ngưỡng khuyến nghị cho hồ sơ của bạn.", "warning")
    if status == "ABSTAIN":
        # Từ chối 2 loại, mỗi loại kèm việc người dùng cần làm
        refer = "Đề nghị trao đổi với bác sĩ, dược sĩ hoặc chuyên gia dinh dưỡng phù hợp."
        needs = [g.get("need", "") for g in (evaluation.get("data_gaps") or []) if g.get("need")]
        ask = ("Đề nghị " + "; ".join(dict.fromkeys(needs)) + "." if needs else
               "Đề nghị chụp lại nhãn rõ hơn (đủ phần thành phần, cảnh báo dị nguyên, bảng dinh dưỡng) "
               "hoặc bổ sung thông tin còn thiếu.")
        types = evaluation.get("abstention_types") or []
        if evaluation.get("scope_issues") and "insufficient_data" in types:
            return ("⚪", "Thiếu thông tin và ngoài phạm vi của hệ thống",
                    f"Câu hỏi cần chuyên gia y tế, và nhãn cũng thiếu dữ liệu. {refer} {ask}", "info")
        if evaluation.get("scope_issues"):
            return ("⚪", "Ngoài phạm vi của hệ thống",
                    f"Hệ thống chỉ sàng lọc nhãn thực phẩm, không thay thế tư vấn y khoa. {refer}", "info")
        return ("⚪", "Chưa đủ thông tin để kết luận", f"Nhãn hoặc dữ liệu chưa đủ để trả lời câu hỏi này. {ask}", "info")
    gap = m_evidence.condition_gap_groups(evaluation)
    if gap:
        # bệnh nền được khai nhưng KB không có dữ liệu: KHÔNG hiện ô xanh "không thấy điểm cần lưu ý" (dễ bị hiểu là an toàn)
        return ("⚠️", "Chưa có dữ liệu cho bệnh bạn đã khai",
                f"Hệ thống không có thông tin về ảnh hưởng của các phụ gia trên nhãn đối với {_join_vi(gap)}. "
                "Việc không có cảnh báo KHÔNG có nghĩa là an toàn cho bạn. Hãy hỏi bác sĩ hoặc chuyên gia dinh dưỡng.", "warning")
    return ("🟢", "Không thấy điểm cần lưu ý",
            "Trong phần chữ đọc được từ nhãn, không có thành phần nào khớp với dị ứng hay bệnh bạn đã khai. "
            "Đây là kết quả sàng lọc, chưa phải xác nhận an toàn.", "success")


def _word_edits(before, after, limit=200):
    """Các chỗ khác nhau theo từ giữa chữ OCR thô và chữ sau hậu xử lý: [(trước, sau)]."""
    import difflib
    a, b = str(before or "").split(), str(after or "").split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append((" ".join(a[i1:i2]), " ".join(b[j1:j2])))
            if len(out) >= limit:
                break
    return out


def render_analysis(res, key, compact=False):
    """compact=True: chỉ hiện sản phẩm, kết luận và câu trả lời (dùng cho các câu trả lời cũ trong lịch sử)."""
    raw_text = res["raw_text"]
    mentions = res["mentions"]
    annotated_img = (ctx.get("annotated_images") or {}).get(res.get("image_no")) or res.get("annotated_img")
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

    _thumb = res.get("image_thumb")
    _img_html = (f'<img src="data:image/jpeg;base64,{base64.b64encode(_thumb).decode("ascii")}" alt="Ảnh nhãn"/>'
                 if _thumb else "")
    _reuse = (' · ↩️ dùng ảnh đã gửi trước đó' if res.get("reused_image") else "")
    _profile = res.get("profile_text", "")
    # một dòng ngữ cảnh (không lặp lại câu hỏi người dùng vừa gõ)
    st.markdown(
        f'<div class="fh-prod">{_img_html}<div class="t">📦 <b>{_html.escape(res.get("display_name") or "sản phẩm trong ảnh này")}</b>'
        f'<small>👤 {_html.escape(_profile or "chưa khai báo hồ sơ")}{_reuse}</small></div></div>',
        unsafe_allow_html=True,
    )
    if res.get("followup_terms"):
        st.caption("🔗 Hiểu là hỏi tiếp về: **" + ", ".join(res["followup_terms"]) + "**")

    # ── PHẦN GỌN (luôn hiện), theo đúng 3 mục: [NHẬN DIỆN TỪ ẢNH] · [NỘI DUNG TRẢ LỜI] · [NGUỒN ĐỐI CHIẾU] ──
    _rec_rows = m_evidence.build_recognition_summary(linked_results)
    st.markdown("**[NHẬN DIỆN TỪ ẢNH]**")
    st.markdown(m_evidence.recognition_headline(
        mentions, _rec_rows, res.get("display_name") if res.get("name_from_label") else ""))
    render_recognition_summary(_rec_rows, key, compact)

    st.markdown("**[NỘI DUNG TRẢ LỜI]**")
    _icon, _name, _meaning, _box = verdict_view(evaluation, nutri_warnings, (selected_persona or {}).get("allergies"))
    _rule_label = evaluation.get("safety_label") or m_pipeline.guardrail_status_to_safety_label(evaluation.get("status"))
    _llm_label = m_pipeline.extract_safety_label(llm_answer, default="") if llm_answer else ""
    getattr(st, _box)(f"**{_name}** — {_meaning}", icon=_icon)
    for _pn in evaluation.get("population_notes") or []:
        if _pn.get("code") == "CONDITION_NO_ADDITIVE_EVIDENCE":
            st.warning(_pn["text"], icon="⚠️")  # nổi bật, không phải dòng chữ nhỏ dễ bỏ qua
        else:
            st.caption(f"ℹ️ {_pn['text']}")
    if gap_reasons:
        more = f" (và {len(gap_reasons) - 1} lý do khác — xem chi tiết)" if len(gap_reasons) > 1 else ""
        st.caption(f"⚠️ Chưa đủ bằng chứng để kết luận chắc chắn: {gap_reasons[0]}{more}")
    if cited:
        render_cited_answer(cited)
    elif evaluation.get("answer_refusal"):
        # câu trả lời của mô hình không bám nguồn → từ chối thay vì hiển thị (xem pipeline.check_answer_sources)
        st.error("**Hệ thống từ chối trả lời tự do cho câu hỏi này** vì không đảm bảo câu trả lời có nguồn kiểm chứng: "
                 f"{evaluation['answer_refusal']}. Nhãn ở trên là kết luận của bộ luật; hãy hỏi cụ thể hơn về một "
                 "phụ gia/dị nguyên hoặc chụp lại nhãn rõ hơn.", icon="🚫")
    else:
        if not llm_answer:
            st.caption("Chưa gọi được mô hình ngôn ngữ — đang hiển thị kết luận từ luật và dữ liệu có sẵn.")
        st.write(evaluation.get("advice", evaluation.get("message", "")))

    st.markdown("**[NGUỒN ĐỐI CHIẾU]**")
    _refs = m_evidence.reference_lines(cited["sources"] if cited else [], _rec_rows)
    if _refs:
        st.markdown("\n".join(
            f"- {'[' + str(r['n']) + '] ' if r['n'] else ''}{r['title']}" + (f" — {r['where']}" if r["where"] else "")
            for r in _refs))
    else:
        st.caption("Không có nguồn tra cứu nào cho kết luận này (hệ thống chỉ dựa vào luật nội bộ); không có nguồn để đối chiếu.")
    if cited and cited["sources"]:
        # nhãn expander là markdown, không phải HTML → không mã hóa HTML (trước đây hiện "&#x27;")
        _short = " · ".join(f"[{s['n']}] {s['title'][:48] + ('…' if len(s['title']) > 48 else '')}"
                            for s in cited["sources"])
        with st.expander(f"📜 Mở nguồn: {_short}", expanded=False):
            render_source_list(cited["sources"], key)

    if compact:
        return

    # ── PHẦN CHI TIẾT (bấm "Xem chi tiết phân tích") ──
    nli_res = None
    tab_ans, tab_ocr, tab_kb, tab_tech = st.tabs([
        "📊 Đánh giá & chỉ số",
        "🏷️ Bóc tách & Chuẩn hóa nhãn",
        "📚 Tri thức & Quy chuẩn",
        "🔬 Chi tiết Kỹ thuật",
    ])

    with tab_ans:
        if gap_reasons:
            st.warning(
                "⚠️ **Chưa đủ bằng chứng để kết luận chắc chắn.** Lý do:\n\n"
                + "\n".join(f"- {r}" for r in gap_reasons)
                + "\n\n*Gợi ý: chụp lại ảnh rõ hơn, đủ phần thành phần/cảnh báo dị nguyên, hoặc hỏi cụ thể về một phụ gia/dị nguyên.*"
            )
        st.caption(f"❓ Câu hỏi: {res.get('user_query', '')}  ·  ảnh #{res.get('image_no', '?')} "
                   f"({_short_file(res.get('image_name', ''))})")

        st.caption(f"Nhãn dùng khi chấm bộ QA: **{_rule_label}** — {QA_LABEL_MEANING.get(_rule_label, '')}"
                   + (f" · nhãn LLM tự ghi: **{_llm_label}**" if _llm_label else "")
                   + f" · trạng thái nội bộ của bộ luật: `{evaluation.get('status')}`")

        _inf = evaluation.get("infant_check")
        if _inf:
            with st.expander(f"👶 Đối chiếu danh mục phụ gia cho trẻ dưới 12 tháng ({_inf.get('checked', 0)} phụ gia)",
                             expanded=bool(_inf.get("alerts"))):
                for a in _inf.get("alerts", []):
                    st.markdown(f"- ❌ **{a['additive']}** — không có trong danh mục TT 24/2019 và Codex cho nhóm 13.1, 13.2")
                for p in _inf.get("permitted", []):
                    locs = "; ".join(x["locator"] + f" (tối đa {x['max_level']})" for x in (p["vn"] or p["codex"])[:2])
                    st.markdown(f"- ✅ **{p['additive']}** — {locs}")
                for a in _inf.get("label_age", []):
                    st.markdown(f"- ⛔ Nhãn ghi **\"{a['label_excerpt']}\"** (từ {a['min_age_months']} tháng) — "
                                "Codex CXS 74-1981 mục 8.6.4: nhãn phải ghi tuổi bắt đầu dùng")
                if _inf.get("nutrients"):
                    st.markdown("- ℹ️ Nguồn khoáng/vitamin, không xét như phụ gia (QCVN 11-1:2012/BYT mục 1.3.4; "
                                "Codex CXS 72-1981 mục A3.4): "
                                + ", ".join(n["additive"] for n in _inf["nutrients"]))
                if not _inf.get("checked"):
                    st.caption("Nhãn không có phụ gia mang mã INS để đối chiếu.")

        st.markdown(
            f'<span class="fh-chip2">🏷️ Lĩnh vực: {_html.escape(domain_vi)}</span> '
            f'<span class="fh-chip2 gray">{_html.escape(domain_code)}</span>',
            unsafe_allow_html=True,
        )

        alerts = evaluation.get("alerts") or []
        rel_alerts = [a for a in alerts if not focus["from_question"] or _matches_focus(a.get("additive", ""), focus)]
        if cited:
            nli_res = render_nli_faithfulness_widget(cited, evaluation, alerts=rel_alerts, regs=regs, passages=passages, label_text=raw_text)
        elif llm_answer:
            st.caption("Câu trả lời không có thẻ nguồn nào thuộc danh mục được công nhận.")

        if focus["labels"]:
            st.caption("🎯 Đối tượng được hỏi: " + ", ".join(dict.fromkeys(focus["labels"])))

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
        c2.metric("Liên kết phụ gia (S_retrieval)", _fmt_score(evaluation.get("s_retrieval")),
                  help=evaluation.get("s_retrieval_basis"))
        c3.metric("Điểm tin cậy tổng", f"{evaluation['s_total']:.2f}")

    with tab_ocr:
        st.markdown("##### 📸 Vùng chữ OCR nhận diện trên bao bì")
        if annotated_img is not None:
            st.image(annotated_img, caption="Khung xanh là vùng chữ OCR nhận diện được", width="stretch")
        else:
            st.caption("Ảnh có khung OCR không còn trong phiên này.")

        st.markdown("##### 📝 Văn bản đọc từ nhãn (sau hậu xử lý)")
        st.text_area("Văn bản đọc từ nhãn:", raw_text, height=120, disabled=True, key=f"ocr_final_text_{key}", label_visibility="collapsed")
        ocr_raw = ocr_result.get("raw_text_before_norm", "")
        text_edits = _word_edits(ocr_raw, raw_text) if ocr_raw else []
        phrase_edits = list(dict.fromkeys(
            (o, n) for m in mentions for o, n in (m.get("ocr_corrections") or [])))
        if text_edits or phrase_edits:
            with st.expander(f"🔍 Chữ OCR thô và các chỗ đã sửa ({len(text_edits) + len(phrase_edits)})", expanded=False):
                if phrase_edits:
                    st.markdown("**Sửa chính tả theo cụm** (chỉ trong danh sách thành phần, theo danh mục tên thành phần):")
                    st.markdown("  \n".join(f"- ~~{_html.escape(o)}~~ → **{_html.escape(n)}**" for o, n in phrase_edits))
                if text_edits:
                    st.markdown("**Hậu xử lý chung** (chuẩn hóa mã INS/E, tách từ dính, bỏ ký tự rác):")
                    shown = text_edits[:40]
                    st.markdown("  \n".join(f"- ~~{_html.escape(o) or '∅'}~~ → **{_html.escape(n) or '∅'}**" for o, n in shown)
                                + (f"  \n… và {len(text_edits) - len(shown)} chỗ khác" if len(text_edits) > len(shown) else ""))
                st.text_area("Chữ OCR thô (trước hậu xử lý)", ocr_raw, height=140, disabled=True, key=f"ocr_raw_{key}")

        st.markdown("##### 🧬 Thành phần đã chuẩn hóa (Tên trên nhãn ➔ INS / FoodOn)")
        norm_rows = build_normalization_rows(linked_results)
        if norm_rows:
            st.dataframe(pd.DataFrame(norm_rows), width="stretch", hide_index=True)
        else:
            st.info("Chưa nhận diện được phụ gia/dị nguyên nào để chuẩn hóa.")

        with st.expander(f"📋 Các cụm thành phần bóc tách ({len(mentions)} mục)", expanded=False):
            for idx, m in enumerate(mentions, 1):
                st.markdown(f"- **[{idx:02d}]** `{m['raw_mention']}` *({m.get('category', 'chưa phân loại')})*")

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
                    "Thực thể": (e.get("canonical_name_vi") or e.get("canonical_name") or "") if e else ("Mã không đọc chắc" if r.get("uncertain_code") else "Không rõ / Nông sản"),
                    "INS / E": e.get("ins_code", "") if e else ((r.get("uncertain_code") or {}).get("code_text") or "N/A"),
                    "FoodOn": curie_val,
                    "Phả hệ DAG": lineage_str,
                    "Chức năng": (e.get("function") or e.get("functional_class") or "") if e else ("—" if r.get("uncertain_code") else "Thành phần tự nhiên"),
                    "Ràng buộc": r.get("constraint_type", "PERMITTED"),
                    "Mức khớp (theo luật)": _fmt_score(r.get("confidence_tau")),
                    "Rationale": r["rationale"],
                })
            st.dataframe(pd.DataFrame(full_rows), width="stretch")
            st.caption(
                f"Retrieval engine: `{linked_results[0].get('retrieval_engine', '')}` · "
                "Selector ưu tiên khớp chính xác mã INS/E và tên cụ thể hơn lớp chung; "
                "Mức khớp là điểm theo luật (khớp mã INS 0,97 · tên chuẩn 0,94 · tên đồng nghĩa 0,91…), KHÔNG phải xác suất đã hiệu chỉnh; "
                "dưới 0,70 thì thử tìm lại bằng từ đồng nghĩa. Cụm chứa nhiều mã được tách thành từng mã; mã OCR không có trong danh mục "
                "được ghi là 'mã không đọc chắc'."
            )

        if passages:
            with st.expander(f"📑 Đoạn trích bằng chứng đã truy xuất ({len(passages)} đoạn)", expanded=False):
                for p in passages:
                    st.caption(f"**`{p.get('locator', '')}`** — {m_evidence._clip(p.get('text', ''), 200)}")

        if nli_res and nli_res.get("claims"):
            st.markdown("##### 🛡️ Chi tiết đối chiếu nhận định với bằng chứng")
            nli_table_rows = []
            for c in nli_res["claims"]:
                badge_lbl = "🟢 Có trong bằng chứng" if c["label"] == "ENTAILMENT" else ("🔴 Có dấu hiệu mâu thuẫn" if c["label"] == "CONTRADICTION" else "🟡 Chưa thấy căn cứ")
                nli_table_rows.append({
                    "Khẳng định của AI (Hypothesis)": c["text"],
                    "Kết quả đối chiếu": badge_lbl,
                    "Căn cứ / Giải trình": c["reason"],
                    "Phương pháp": "LLM" if c.get("source") == "llm_nli" else "Trùng từ khóa"
                })
            st.dataframe(pd.DataFrame(nli_table_rows), width="stretch", hide_index=True)

        for acr in (off_cross_ref.get("allergen_cross_ref") or []):
            st.caption(f"Open Food Facts VN: {acr['allergen']} — {acr['note']} (Nguồn: {acr['source']})")


def render_message(msg, i):
    if msg["role"] == "user":
        with st.chat_message("user"):
            for t in msg.get("thumbs") or []:
                if t:
                    st.image(t, width=220)
            if msg.get("text"):
                _t = msg["text"]
                if len(_t) > 600:
                    _t = _t[:600] + f"… *(đã rút gọn, tổng {len(msg['text'])} ký tự)*"
                st.markdown(_html.escape(_t))
        return
    with st.chat_message("assistant", avatar="🥗"):
        _icons = {"error": "❌ ", "warning": "⚠️ ", "success": "✅ "}
        _parts = [_icons.get(kind, "") + body for kind, body in (msg.get("blocks") or []) if body]
        if _parts:
            st.markdown("\n\n".join(_parts))
        if msg.get("choice") and i == len(chat) - 1 and ctx.get("awaiting_image_choice"):
            b1, b2 = st.columns(2)
            if b1.button(f"✅ Dùng {_image_label()}", key=f"use_old_{i}", width="stretch", type="primary"):
                st.session_state["image_choice"] = "old"
            if b2.button("📎 Gửi ảnh khác", key=f"new_img_{i}", width="stretch"):
                st.session_state["image_choice"] = "new"
        if msg.get("result"):
            # Mọi câu trả lời (kể cả câu mới nhất) mặc định GỌN: kết luận · câu trả lời · nguồn 1 dòng.
            # Bấm "Xem chi tiết phân tích" để mở các tab đánh giá, bóc tách nhãn, tri thức, kỹ thuật.
            expanded = st.session_state.get(f"detail_{i}", False)
            render_analysis(msg["result"], i, compact=not expanded)
            label = "▲ Thu gọn chi tiết" if expanded else "🔎 Xem chi tiết phân tích"
            if st.button(label, key=f"toggle_detail_{i}"):
                st.session_state[f"detail_{i}"] = not expanded
                st.rerun()


def render_context_chips():
    img_chip = (f'<span class="ok">🖼️ Đang hỏi trên {_html.escape(_image_label())}</span>' if ctx["ocr_result"] is not None
                else '<span class="miss">🖼️ Chưa có ảnh nhãn</span>')
    prof = ctx["profile"]
    prof_chip = (f'<span class="ok">👤 {_html.escape(m_chat.profile_summary(prof))}</span>' if prof.get("declared")
                 else '<span class="miss">👤 Chưa khai báo hồ sơ</span>')
    pend = (f'<span class="miss">❓ Câu hỏi đang chờ: {_html.escape(ctx["pending_question"][:60])}</span>'
            if ctx["pending_question"] else "")
    st.markdown(f'<div class="fh-ctx">{img_chip}{prof_chip}{pend}</div>', unsafe_allow_html=True)


# ── Màn hình chào (khi chưa có tin nhắn) ──
if not chat:
    st.markdown(
        '<div class="fh-greet"><div class="hi">Xin chào, mình là FoodHealthQA</div>'
        '<div class="sub">Bạn muốn tư vấn sản phẩm nào hôm nay?</div></div>'
        '<div class="fh-tips">'
        '<div class="fh-tip"><b>📎 Đính kèm ảnh nhãn</b>Bấm nút đính kèm trong ô nhập, chụp rõ phần <i>Thành phần</i>.'
        f'<br><small>{_html.escape(m_chat.IMAGE_FORMAT_HINT)}</small></div>'
        '<div class="fh-tip"><b>💬 Đặt câu hỏi</b><i>“Sản phẩm này có chứa dị nguyên nào không?”</i></div>'
        '<div class="fh-tip"><b>👤 Hồ sơ sức khỏe</b><i>“Tôi 40 tuổi, bị tăng huyết áp, dị ứng tôm.”</i></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Có thể gửi cả ba trong một tin nhắn, hoặc từng phần — hệ thống sẽ hỏi lại phần còn thiếu.")

for _i, _msg in enumerate(chat):
    render_message(_msg, _i)

render_context_chips()

# ── Thanh nhập kiểu Gemini: ô chat + bộ chọn mô hình ngay cạnh nút gửi + gợi ý định dạng ảnh ──
_provider_keys = list(m_pipeline.LLM_PROVIDERS)
_PROVIDER_SHORT = {"gemini": "Gemini", "ollama": "Qwen2.5"}
with st.bottom:
    with st.container(key="fh_inputbar"):
        _prompt = st.chat_input(
            "Hỏi FoodHealthQA — đính kèm ảnh nhãn, nhập câu hỏi và hồ sơ sức khỏe…",
            accept_file=True,
            file_type=list(m_chat.PICKER_IMAGE_EXT),
            max_upload_size=m_chat.UPLOAD_HARD_LIMIT_MB,
            key="fh_chat_input",
        )
        _provider_key = st.selectbox(
            "Mô hình sinh câu trả lời", _provider_keys,
            index=_provider_keys.index(m_pipeline.get_llm_provider()),
            format_func=lambda k: _PROVIDER_SHORT.get(k, k),
            key="llm_provider_select", label_visibility="collapsed",
            help="Mô hình sinh câu trả lời — Gemini: gọi API Google · Qwen2.5: chạy cục bộ qua Ollama.",
        )
    m_pipeline.set_llm_provider(_provider_key)
    # Định dạng ảnh đã ghi ở màn hình chào và trong thông báo lỗi; dưới thanh chat chỉ hiện khi Ollama không dùng được.
    if _provider_key == "ollama":
        _ok, _msg = m_pipeline.ollama_status()
        if not _ok:
            st.markdown(f'<div class="fh-hint">⚠️ {_html.escape(_msg)}</div>', unsafe_allow_html=True)
_choice = st.session_state.pop("image_choice", None)

if _prompt is not None or _choice is not None:
    with st.chat_message("assistant", avatar="🥗"):
        try:
            if _choice == "old":
                handle_turn("", [], use_old_image=True)
            elif _choice == "new":
                ctx["awaiting_image_choice"] = False
                chat.append({"role": "user", "text": "📎 Gửi ảnh khác", "thumbs": []})
                chat.append({"role": "assistant", "result": None, "blocks": [(
                    "info", "Hãy bấm nút đính kèm trong ô nhập và chọn ảnh nhãn mới (" + m_chat.IMAGE_FORMAT_HINT
                    + "). Câu hỏi đang chờ sẽ được trả lời ngay khi ảnh hợp lệ.")]})
            else:
                _text, _files = (_prompt.text or ""), list(_prompt.files or [])
                if _text.strip() or _files:
                    handle_turn(_text, _files)
        except Exception as _exc:  # không để lỗi pipeline làm vỡ cuộc trò chuyện
            st.session_state["chat"].append({"role": "assistant", "result": None, "blocks": [
                ("error", f"Đã xảy ra lỗi khi xử lý: `{_html.escape(str(_exc))[:300]}`. Vui lòng thử lại hoặc gửi ảnh khác.")
            ]})
    st.rerun()
