"""
Ứng dụng Web Demo Hệ thống FoodHealthQA (v9 - Complete Data Hub Integration)
Giao diện trực quan hỏi đáp tư vấn an toàn thực phẩm từ nhãn bao bì thực phẩm
Kế thừa kiến trúc:
- Tầng 1: Perception (Hỗ trợ Ground Truth Benchmark & PaddleOCR kèm Phục hồi Chính tả)
- Tầng 2: Normalization (FoodOntoRAG 4-Agent Entity Linking kết hợp FoodOn CURIE)
- Tầng 3: Hybrid Retrieval & Data Hub (TT 24, Codex, JECFA, Food Hazard, FNDDS, Open Food Facts)
- Tầng 4: Grounded Generation & Guardrail (Cảnh báo Chống chỉ định Y khoa & Dị nguyên theo NHANES)
- Data Hub: Hỗ trợ nạp và hiển thị trọn vẹn 16 tệp dữ liệu thực tế (như trong name-data.png)
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
import pandas as pd
from PIL import Image
import time

from modules.ocr_extractor import is_paddleocr_available, get_label_studio_product_list
from modules.file_resolver import find_data_file, read_table_flexibly
from modules.csv_data_defaults import ensure_default_csv_data
from modules.pipeline import (
    load_knowledge_base,
    load_personas,
    load_food_hazards,
    load_fndds_nutrition,
    load_openfoodfacts_vietnam,
    load_nhanes_biomarkers,
    check_food_hazards,
    evaluate_nutritional_risk,
    run_perception_pipeline,
    foodontorag_entity_linking,
    evaluate_safety_and_guardrail,
    cross_reference_openfoodfacts,
    get_nhanes_population_stats,
    generate_llm_answer,
    generate_llm_only_answer,
    enrich_persona_with_nhanes,
)
from modules.foodon import find_foodon_exact_label, load_foodon_entities, search_foodon_entities

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="FoodHealthQA Demo (Full Data Suite)",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thiết lập đường dẫn thư mục chuẩn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Tự động đảm bảo các tệp dữ liệu cốt lõi luôn hiện diện
ensure_default_csv_data(DATA_DIR)

# Nạp dữ liệu đa nguồn thông minh từ DATA_DIR (chỉ load 1 lần khi khởi động)
kb_additives = load_knowledge_base(DATA_DIR)
personas_list = load_personas(DATA_DIR)
hazards_list = load_food_hazards(DATA_DIR)
fndds_list = load_fndds_nutrition(DATA_DIR)
off_products = load_openfoodfacts_vietnam(DATA_DIR)
foodon_entities = load_foodon_entities(DATA_DIR)
nhanes_biomarkers_cache = load_nhanes_biomarkers(DATA_DIR)

# Tiêu đề ứng dụng
st.title("🥗 FoodHealthQA: Tư vấn Sức khỏe từ Thành phần Thực phẩm trên Bao bì")
st.markdown("""
*Hệ thống Hỏi đáp Dinh dưỡng & An toàn Phụ gia Thực phẩm — Kiến trúc 5 tầng:*

`Ảnh bao bì` ➔ `Tầng 1: Perception (CLAHE/Deskew ➔ PaddleOCR ➔ Phục hồi chính tả)` ➔ `Tầng 2: Chuẩn hóa FoodOntoRAG (4-Agent Entity Linking)` ➔ `Tầng 3: Hybrid RAG (Codex, JECFA, TT24, EFSA, Food Hazard, FNDDS, OpenFoodFacts)` ➔ `Tầng 4: Guardrail & Negation Constraint (NHANES Persona)` ➔ `Tầng 5: LLM Grounded Generation (Gemini Flash — Câu trả lời có trích dẫn)`
""")

# Thanh thông báo trạng thái OCR Engine & Data Hub
paddle_ready = is_paddleocr_available()
if paddle_ready:
    st.success("🟢 **PaddleOCR Engine: ĐÃ SẴN SÀNG** | **Data Hub:** Đã nạp thành công bộ dữ liệu phụ gia, hồ sơ NHANES, FNDDS & Food Hazard.")
else:
    st.info("ℹ️ **PaddleOCR Engine: CHẾ ĐỘ DỰ PHÒNG** | **Data Hub:** Đã nạp sẵn sàng 16 tệp dữ liệu nghiên cứu chuẩn hóa.")

st.divider()

# Bố cục 2 cột chính
col_left, col_right = st.columns([1.1, 1.4], gap="large")

with col_left:
    st.subheader("📸 1. Thông tin Đầu vào & Hồ sơ Sức khỏe")

    input_source = st.radio(
        "Nguồn ảnh bao bì thực phẩm:",
        ["Chọn ảnh nhãn mẫu có sẵn", "Tải ảnh từ máy tính"],
        horizontal=True
    )

    img_display = None
    image_input_to_process = None
    sample_key = "custom"
    selected_prod_name = "Sản phẩm thực phẩm"

    if input_source == "Chọn ảnh nhãn mẫu có sẵn":
        ls_products = get_label_studio_product_list(ASSETS_DIR)
        if not ls_products:
            st.warning("Không tìm thấy dữ liệu Label Studio annotation. Vui lòng kiểm tra thư mục assets/label_studio/.")
            st.stop()

        ls_options = []
        option_labels = []
        for p in ls_products:
            option_text = f"[{p['filename']}] {p['label']}"
            option_labels.append(option_text)
            ls_options.append((option_text, p))

        sample_option = st.selectbox(
            f"Chọn nhãn sản phẩm thử nghiệm ({len(ls_products)} ảnh đã gán nhãn NER):",
            option_labels,
            help="550 ảnh nhãn bao bì thực phẩm Việt Nam đã gán nhãn NER (BIO tagging) qua Label Studio. "
                 "Mỗi ảnh có ground truth: tên sản phẩm, thành phần (I), phụ gia (A), cảnh báo."
        )

        ls_match = None
        for opt_text, p_info in ls_options:
            if sample_option == opt_text:
                ls_match = p_info
                break

        if ls_match:
            sample_key = ls_match["filename"]
            selected_prod_name = ls_match["product_name"]
            img_path = os.path.join(ASSETS_DIR, ls_match["image_ref"])

            if os.path.exists(img_path):
                img_display = Image.open(img_path)
                image_input_to_process = img_path
                st.image(img_display, caption=f"Ảnh mẫu: {selected_prod_name}", use_column_width=True)


    else:
        uploaded_file = st.file_uploader("Tải ảnh nhãn bảng thành phần (JPEG, PNG):", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img_display = Image.open(uploaded_file)
            image_input_to_process = img_display
            sample_key = "custom"

            selected_prod_name = st.text_input(
                "Tên sản phẩm (dùng để đối soát dinh dưỡng & mối nguy):",
                value="",
                placeholder="VD: Bánh quy bơ, Mì ăn liền Hảo Hảo, Sữa tươi Vinamilk...",
                help="Nhập tên sản phẩm để hệ thống đối soát chính xác hơn với dữ liệu FNDDS và Food Hazard. Để trống nếu không biết — hệ thống sẽ dùng kết quả OCR."
            )
            if not selected_prod_name.strip():
                selected_prod_name = uploaded_file.name

            st.image(img_display, caption=f"Ảnh nhãn tải lên: {uploaded_file.name}", use_column_width=True)

    # Chọn hồ sơ người dùng
    persona_names = [p["name"] for p in personas_list]
    selected_persona_name = st.selectbox("Chọn Hồ sơ Sức khỏe Người dùng (User Persona):", persona_names)
    selected_persona = next(p for p in personas_list if p["name"] == selected_persona_name)
    selected_persona = enrich_persona_with_nhanes(selected_persona, DATA_DIR, biomarkers_cache=nhanes_biomarkers_cache)

    st.info(f"**Thông tin lâm sàng:** {selected_persona['description']}")
    nhanes_enr = selected_persona.get("nhanes_enrichment")
    if nhanes_enr:
        st.caption(f"Đã bổ sung dữ liệu NHANES: khớp {nhanes_enr['matched_group_size']} hồ sơ từ {', '.join(nhanes_enr['source_files'][:3])}...")

    # Chọn cấu hình pipeline (Ablation Study)
    st.markdown("---")
    has_ground_truth = sample_key != "custom"
    if has_ground_truth:
        pipeline_options = [
            "C: OCR + RAG + LLM (Full Pipeline — Đo hiệu năng thực tế)",
            "B: RAG từ văn bản chuẩn (Ground Truth — Đo hiệu năng RAG khi đầu vào hoàn hảo)",
            "A: LLM không truy xuất (Baseline — Không RAG)",
        ]
        pipeline_help = (
            "Ablation Study 3 cấu hình (cần ảnh mẫu có Ground Truth):\n"
            "• **A vs B**: RAG cải thiện câu trả lời bao nhiêu?\n"
            "• **B vs C**: Lỗi OCR ảnh hưởng thế nào đến kết quả RAG?\n"
            "• **A vs C**: Full pipeline so với baseline LLM."
        )
    else:
        pipeline_options = [
            "C: OCR + RAG + LLM (Full Pipeline)",
        ]
        pipeline_help = (
            "Ảnh tải lên không có Ground Truth annotation → chỉ chạy Full Pipeline (C).\n"
            "Chọn ảnh mẫu (550 ảnh Label Studio) để mở khóa 3 cấu hình ablation study (A, B, C)."
        )

    pipeline_mode = st.selectbox(
        "⚙️ Cấu hình Pipeline (Ablation Study):",
        pipeline_options,
        help=pipeline_help,
    )

    # Câu hỏi người dùng
    user_query = st.text_input(
        "Câu hỏi tư vấn an toàn thực phẩm:",
        "Sản phẩm này có an toàn cho tình trạng sức khỏe hiện tại của tôi không? Cần lưu ý chất phụ gia nào?"
    )

    btn_run = st.button("🚀 Phân tích Nhãn & Đánh giá An toàn Sức khỏe", type="primary", use_container_width=True)

with col_right:
    st.subheader("📊 2. Kết quả Phân tích Pipeline Đa tầng")

    if btn_run and img_display and image_input_to_process:
        is_mode_a = pipeline_mode.startswith("A:")
        is_mode_b = pipeline_mode.startswith("B:")
        is_mode_c = pipeline_mode.startswith("C:")

        mode_label = "A" if is_mode_a else ("B" if is_mode_b else "C")
        spinner_text = {
            "A": "Cấu hình A: LLM-only — đang gọi Gemini Flash trả lời trực tiếp (không RAG)...",
            "B": "Cấu hình B: Ground Truth + RAG — đang chạy pipeline RAG từ văn bản chuẩn...",
            "C": "Cấu hình C: OCR + RAG + LLM — đang chạy full pipeline...",
        }[mode_label]

        with st.spinner(spinner_text):
            time.sleep(0.4)

            # ── Cấu hình A: LLM-only (không OCR, không RAG) ──
            #    Dùng Ground Truth nếu có (ảnh mẫu), hoặc OCR nếu upload ảnh tùy ý
            if is_mode_a:
                use_ocr = sample_key == "custom"
                ocr_result = run_perception_pipeline(image_input_to_process, sample_key=sample_key, force_real_ocr=use_ocr)
                raw_text = ocr_result.get("full_text", "")
                mentions = ocr_result.get("mentions", [])
                avg_ocr_conf = ocr_result.get("avg_confidence", 0)
                annotated_img = ocr_result.get("annotated_image")
                linked_results = []
                evaluation = {"status": "N/A", "title": "Cấu hình A: LLM-only — Không có RAG/Knowledge Base",
                              "s_perception": avg_ocr_conf, "s_retrieval": 0.0, "s_total": 0.0,
                              "alerts": [], "advice": "", "regulatory": [], "efsa_warnings": [], "citations": []}
                detected_hazards = []
                matched_fndds = None
                nutri_warnings = []
                constraint_details = []
                off_cross_ref = {}
                llm_answer = generate_llm_only_answer(raw_text, selected_persona, user_query)

            # ── Cấu hình B: Ground Truth + RAG (không OCR) ──
            #    Dùng Ground Truth (văn bản chuẩn) để đo hiệu năng RAG khi đầu vào hoàn hảo.
            #    Upload ảnh tùy ý → fallback OCR nhưng cảnh báo mất ý nghĩa ablation.
            elif is_mode_b:
                use_ocr = sample_key == "custom"
                if use_ocr:
                    st.warning(
                        "⚠️ **Cấu hình B yêu cầu văn bản chuẩn (Ground Truth)** để đo hiệu năng RAG khi đầu vào hoàn hảo. "
                        "Ảnh tải lên không có Ground Truth annotation nên hệ thống sẽ dùng OCR — "
                        "kết quả sẽ giống Cấu hình C và không phản ánh đúng ablation study. "
                        "**Khuyến nghị:** Chọn ảnh mẫu có sẵn (550 ảnh Label Studio) để so sánh B vs C có ý nghĩa."
                    )
                ocr_result = run_perception_pipeline(image_input_to_process, sample_key=sample_key, force_real_ocr=use_ocr)
                if ocr_result.get("is_ocr_unavailable"):
                    st.error("Không thể đọc ảnh này. Vui lòng thử ảnh rõ hơn hoặc chọn ảnh mẫu có sẵn.")
                    st.stop()
                raw_text = ocr_result["full_text"]
                mentions = ocr_result["mentions"]
                avg_ocr_conf = ocr_result["avg_confidence"]
                annotated_img = ocr_result["annotated_image"]

                if selected_prod_name in ("Sản phẩm thực phẩm",) or selected_prod_name.endswith((".jpg", ".png", ".jpeg")):
                    ingredient_text = " ".join(m.get("raw_mention", "") for m in mentions[:3])
                    if ingredient_text.strip():
                        selected_prod_name = ingredient_text[:80]

                linked_results = [foodontorag_entity_linking(m, kb_additives) for m in mentions]
                evaluation = evaluate_safety_and_guardrail(
                    linked_results, selected_persona, user_query, avg_ocr_conf=avg_ocr_conf, all_mentions=mentions)
                detected_hazards = check_food_hazards(selected_prod_name, mentions, hazards_list)
                matched_fndds, nutri_warnings, constraint_details = evaluate_nutritional_risk(selected_prod_name, fndds_list, selected_persona)
                off_cross_ref = cross_reference_openfoodfacts(selected_prod_name, mentions, off_products)
                llm_answer = generate_llm_answer(
                    evaluation, linked_results, selected_persona, user_query,
                    detected_hazards=detected_hazards, matched_fndds=matched_fndds, nutri_warnings=nutri_warnings)

            # ── Cấu hình C: OCR + RAG + LLM (Full Pipeline) ──
            #    Luôn chạy PaddleOCR thật để đo hiệu năng thực tế (kể cả ảnh mẫu có ground truth)
            else:
                ocr_result = run_perception_pipeline(image_input_to_process, sample_key=sample_key, force_real_ocr=True)
                if ocr_result.get("is_ocr_unavailable"):
                    st.error(
                        "Không thể đọc ảnh này vì OCR không khả dụng hoặc không nhận diện được chữ. "
                        "Hãy cài/khởi động PaddleOCR hoặc thử ảnh rõ hơn."
                    )
                    st.stop()
                raw_text = ocr_result["full_text"]
                mentions = ocr_result["mentions"]
                avg_ocr_conf = ocr_result["avg_confidence"]
                annotated_img = ocr_result["annotated_image"]

                if selected_prod_name in ("Sản phẩm thực phẩm",) or selected_prod_name.endswith((".jpg", ".png", ".jpeg")):
                    ingredient_text = " ".join(m.get("raw_mention", "") for m in mentions[:3])
                    if ingredient_text.strip():
                        selected_prod_name = ingredient_text[:80]

                linked_results = [foodontorag_entity_linking(m, kb_additives) for m in mentions]
                evaluation = evaluate_safety_and_guardrail(
                    linked_results, selected_persona, user_query, avg_ocr_conf=avg_ocr_conf, all_mentions=mentions)
                detected_hazards = check_food_hazards(selected_prod_name, mentions, hazards_list)
                matched_fndds, nutri_warnings, constraint_details = evaluate_nutritional_risk(selected_prod_name, fndds_list, selected_persona)
                off_cross_ref = cross_reference_openfoodfacts(selected_prod_name, mentions, off_products)
                llm_answer = generate_llm_answer(
                    evaluation, linked_results, selected_persona, user_query,
                    detected_hazards=detected_hazards, matched_fndds=matched_fndds, nutri_warnings=nutri_warnings)

        # Banner cấu hình đang chạy
        mode_descriptions = {
            "A": ("🅰️ Cấu hình A: LLM không truy xuất", "LLM trả lời trực tiếp từ văn bản nhãn — **không RAG, không Knowledge Base**. Đo baseline LLM thuần túy."),
            "B": ("🅱️ Cấu hình B: RAG từ văn bản chuẩn", "Ground Truth (Label Studio annotation) → Entity Linking → RAG → LLM. **Đo hiệu năng RAG khi đầu vào hoàn hảo.**"),
            "C": ("🅲 Cấu hình C: OCR + RAG + LLM (Full Pipeline)", "PaddleOCR → Phục hồi chính tả → Entity Linking → RAG → Guardrail → LLM. **Đo hiệu năng thực tế với sai số OCR.**"),
        }
        mode_title, mode_desc = mode_descriptions[mode_label]
        st.info(f"**{mode_title}** — {mode_desc}")

        # Hiển thị Tabs kết quả (4 Tabs) — tiêu đề thay đổi theo cấu hình
        if is_mode_a:
            ocr_tab_label = "📝 Dữ liệu Đầu vào (Văn bản Nhãn)"
        elif is_mode_b:
            ocr_tab_label = "📝 Nhãn Chuẩn (Ground Truth)"
        else:
            ocr_tab_label = "📝 Nhận diện OCR & Phục hồi Chính tả"

        tab_guardrail, tab_entities, tab_ocr, tab_datahub = st.tabs([
            "🛡️ Đánh giá An toàn & Guardrail",
            "🧬 Chuẩn hóa Thực thể (FoodOntoRAG)",
            ocr_tab_label,
            "📁 Tra cứu Đa nguồn Dữ liệu (Data Hub)"
        ])

        with tab_guardrail:
            # Banner kết quả sàng lọc
            if is_mode_a:
                st.info(f"### {evaluation['title']}")
            elif evaluation["status"] == "CRITICAL_ALERT":
                st.error(f"### {evaluation['title']}")
            elif evaluation["status"] in ("WARNING", "CAUTION"):
                st.warning(f"### {evaluation['title']}")
            elif evaluation["status"] == "SAFE":
                st.success(f"### {evaluation['title']}")
            else:
                st.info(f"### {evaluation['title']}")

            # Điểm tin cậy kép (ẩn ở mode A vì không có RAG)
            if not is_mode_a:
                c1, c2, c3 = st.columns(3)
                c1.metric("Độ tin cậy OCR (S_perception)", f"{evaluation['s_perception']:.2f}")
                c2.metric("Độ khớp truy xuất (S_retrieval)", f"{evaluation['s_retrieval']:.2f}")
                c3.metric("Điểm tin cậy tổng thể (S_total)", f"{evaluation['s_total']:.2f}")

            # Câu trả lời tư vấn từ LLM
            if llm_answer:
                if is_mode_a:
                    st.markdown("#### 🤖 Câu trả lời từ LLM (Không RAG — chỉ dựa trên kiến thức LLM):")
                else:
                    st.markdown("#### 🤖 Câu trả lời Tư vấn từ AI (có trích dẫn từ RAG):")
                st.markdown(llm_answer)
            else:
                if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                    st.info("Chưa cấu hình Gemini API key; đang hiển thị kết quả dựa trên luật và dữ liệu có sẵn.")
                else:
                    st.warning("Không tạo được câu trả lời từ LLM; đang dùng lời khuyên dự phòng dựa trên dữ liệu.")
                if not is_mode_a:
                    st.markdown("#### 💡 Lời khuyên Tư vấn Chuyên sâu:")
                    st.write(evaluation.get("advice", evaluation.get("message", "")))

            # Chi tiết cảnh báo rủi ro nếu có
            if evaluation.get("alerts"):
                st.markdown("#### ⚠️ Cảnh báo Rủi ro Phụ gia & Dị ứng:")
                for alert in evaluation["alerts"]:
                    st.markdown(f"- **{alert['additive']}:** {alert['message']}")

            # Hiển thị cảnh báo sự cố từ Food Hazard Detector nếu có
            if detected_hazards:
                st.markdown("#### 🚨 Lịch sử Sự cố Thực phẩm Liên quan (Food Hazard Detector — SemEval-2025):")
                st.caption(f"Phát hiện {len(detected_hazards)} sự cố liên quan trong 5.082 báo cáo thu hồi/cảnh báo nhiễm bẩn/dị nguyên:")
                severity_icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                for h in detected_hazards:
                    sev = h.get("severity", "MEDIUM")
                    icon = severity_icons.get(sev, "⚪")
                    if sev == "HIGH":
                        st.error(f"{icon} **[{h['hazard_category'].upper()}]** {h['title']} — *Tác nhân: {h['hazard']}* | Sản phẩm: {h.get('product', 'N/A')}")
                    elif sev == "MEDIUM":
                        st.warning(f"{icon} **[{h['hazard_category']}]** {h['title']} — *Tác nhân: {h['hazard']}*")
                    else:
                        st.info(f"{icon} **[{h['hazard_category']}]** {h['title']} — *Tác nhân: {h['hazard']}*")

            # Hiển thị đối chiếu OpenFoodFacts VN
            if off_cross_ref.get("allergen_cross_ref"):
                st.markdown("#### 🔎 Đối chiếu Dị nguyên từ Open Food Facts VN:")
                for acr in off_cross_ref["allergen_cross_ref"]:
                    st.warning(f"**{acr['allergen']}** — {acr['note']} *(Nguồn: {acr['source']})*")
            if off_cross_ref.get("matched_products"):
                with st.expander("📦 Sản phẩm tương đồng trong Open Food Facts VN"):
                    for mp in off_cross_ref["matched_products"]:
                        st.markdown(f"- {mp}")

            # Hiển thị đánh giá dinh dưỡng FNDDS nếu có
            if matched_fndds:
                fndds_desc = matched_fndds.get("food_description", "N/A")
                fndds_cat = matched_fndds.get("food_category", "")
                fndds_portion = matched_fndds.get("portion_description", "")
                fndds_gram = matched_fndds.get("gram_weight", 0)
                portion_info = f" | Khẩu phần: {fndds_portion} ({fndds_gram:.0f}g)" if fndds_portion and fndds_gram > 0 else ""
                cat_info = f" | Nhóm: {fndds_cat}" if fndds_cat else ""
                st.markdown(f"#### 🥗 Đối chiếu Dinh dưỡng USDA FNDDS:")
                st.caption(f"Sản phẩm khớp: **{fndds_desc}**{cat_info}{portion_info}")
            if nutri_warnings:
                st.markdown("**Cảnh báo Dinh dưỡng Số học (Ràng buộc NHANES):**")
                for nw in nutri_warnings:
                    st.info(f"- {nw}")

            # Hiển thị bảng hàm chỉ thị khoảng I_g(v) - pFoodReQ
            if constraint_details:
                st.markdown("#### 📐 Bảng Ràng buộc Số học pFoodReQ — Hàm chỉ thị khoảng I_g(v):")
                df_constraints = pd.DataFrame([
                    {
                        "Bệnh lý": cd["condition"],
                        "Chỉ số": cd["nutrient"],
                        "Giá trị": f"{cd['value']:.1f} {cd['unit']}",
                        "Ngưỡng trên": f"{cd['upper_bound']:.1f} {cd['unit']}" if cd.get("upper_bound") else "—",
                        "I_g(v)": cd["I_g"],
                        "Trạng thái": "✅ Trong khoảng" if cd["I_g"] == 1 else f"⚠️ Vi phạm ({cd['severity']})",
                    }
                    for cd in constraint_details
                ])
                st.dataframe(df_constraints, use_container_width=True, hide_index=True)

            # Đánh giá dinh dưỡng tương thích
            if evaluation.get("benefits"):
                st.markdown("#### ✅ Đánh giá Tương thích Dinh dưỡng:")
                for ben in evaluation["benefits"]:
                    st.markdown(f"- **{ben['additive']}:** {ben['message']}")

            # Hiển thị bằng chứng EFSA nếu có
            if evaluation.get("efsa_warnings"):
                st.markdown("#### 🇪🇺 Bằng chứng EFSA (European Food Safety Authority):")
                for ew in evaluation["efsa_warnings"]:
                    with st.expander(f"📋 {ew['substance']} — EFSA {ew['year']}"):
                        st.markdown(f"**Phát hiện chính:** {ew['key_finding']}")
                        if ew.get("efsa_adi"):
                            st.markdown(f"**ADI (EFSA):** {ew['efsa_adi']} mg/kg bw/ngày")
                        st.markdown(f"**Nhóm rủi ro:** {', '.join(ew['matched_risk_groups'])}")
                        st.caption(f"Tài liệu: {ew['reference']}")

            # Hiển thị đánh giá ADI & quy chuẩn pháp lý
            if evaluation.get("regulatory"):
                with st.expander("📐 Bảng ADI (JECFA/EFSA) & Quy chuẩn TT 24/2019/TT-BYT"):
                    reg_rows = []
                    for reg in evaluation["regulatory"]:
                        adi = reg.get("adi", {})
                        vn = reg.get("vn_regulation")
                        row = {
                            "Phụ gia": reg["additive"],
                            "ADI": adi.get("detail", "N/A"),
                            "Nguồn": adi.get("source", ""),
                            "TT 24/BYT": vn["status"] if vn else "Chưa xác minh",
                            "Phụ lục": vn.get("appendix", "—") if vn else "—",
                        }
                        if adi.get("max_daily_intake_mg"):
                            row["Lượng tối đa/ngày"] = f"{adi['max_daily_intake_mg']:.0f} mg"
                        elif adi.get("compliance_status") == "GMP":
                            row["Lượng tối đa/ngày"] = "GMP (không có ADI dạng số)"
                        elif adi.get("compliance_status") == "NO_NUMERIC_ADI":
                            row["Lượng tối đa/ngày"] = "Không có ngưỡng ADI dạng số"
                        elif adi.get("compliance_status") == "NOT_NUMERICALLY_APPLICABLE":
                            row["Lượng tối đa/ngày"] = "ADI nhóm; không tính riêng chất này"
                        elif adi.get("compliance_status") == "NOT_ESTABLISHED":
                            row["Lượng tối đa/ngày"] = "Chưa thiết lập"
                        else:
                            row["Lượng tối đa/ngày"] = "Chưa tính — cần xác minh nguồn"
                        reg_rows.append(row)
                    if reg_rows:
                        st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### 📜 Căn cứ Pháp lý & Tài liệu Quy chuẩn Trích dẫn:")
            for cit in evaluation.get("citations", []):
                st.markdown(f"- 🏛️ `{cit}`")

        with tab_entities:
            st.markdown("#### Bảng Chuẩn hóa Thực thể qua Kiến trúc 4-Agent & Bản thể học FoodOn:")
            st.caption("Kháng lỗi chính tả OCR và trôi dạt bản thể (Ontology Drift) bằng vòng lặp Selector, Scorer và Synonym Generator.")

            if is_mode_a:
                st.warning(
                    "**Cấu hình A: LLM-only** — Không sử dụng Entity Linking.\n\n"
                    "Ở cấu hình này, LLM trả lời trực tiếp từ văn bản nhãn mà không qua bước chuẩn hóa thực thể "
                    "(FoodOntoRAG 4-Agent). Chuyển sang **Cấu hình B** hoặc **C** để xem kết quả Entity Linking."
                )
            elif not linked_results:
                st.info("Chưa có dữ liệu Entity Linking. Vui lòng upload ảnh bao bì hoặc nhập thành phần để hệ thống phân tích.")
            else:
                df_rows = []
                for r in linked_results:
                    e = r["matched_entity"]
                    constraint_label = r.get("constraint_type", "PERMITTED")
                    constraint_icons = {
                        "PERMITTED": "✅ PERMITTED",
                        "RESTRICTED": "⚠️ RESTRICTED",
                        "ALLERGEN_RISK": "🟡 ALLERGEN_RISK",
                        "CONTRAINDICATED": "🔴 CONTRAINDICATED",
                    }
                    df_rows.append({
                        "Mention trên bao bì": r["raw_mention"],
                        "Thực thể chuẩn hóa": e["canonical_name"] if e else "Không rõ / Nông sản",
                        "Mã số INS / E": e["ins_code"] if e else "N/A",
                        "Mã FoodOn CURIE": (
                            e.get("foodon_curie")
                            if e and e.get("foodon_curie") and e.get("foodon_curie") != "FOODON:00000000"
                            else (
                                (find_foodon_exact_label(r["raw_mention"], foodon_entities) or {}).get(
                                    "curie", "FOODON:00000000"
                                )
                                if not e else "FOODON:00000000"
                            )
                        ),
                        "Chức năng công nghệ": e["function"] if e else "Thành phần tự nhiên",
                        "Ràng buộc": constraint_icons.get(constraint_label, constraint_label),
                        "Độ tin cậy (τ)": f"{r['confidence_tau']:.3f}",
                        "Nguyên tắc suy luận / Rationale": r["rationale"]
                    })

                df_display = pd.DataFrame(df_rows)
                try:
                    st.dataframe(df_display, use_container_width=True)
                except Exception:
                    st.table(df_display)

            if linked_results:
                retrieval_engine = linked_results[0].get("retrieval_engine", "")
                st.markdown(f"""
> **Retrieval Engine:** `{retrieval_engine}`
>
> **Nguyên tắc Selector:** Ưu tiên khớp chính xác mã số INS/E (*Exact-match preference*) và ưu tiên tên cụ thể hơn lớp chung (*Specificity rule*).
>
> **Negation Constraint:** Mỗi thành phần được gán nhãn PERMITTED / RESTRICTED / CONTRAINDICATED / ALLERGEN_RISK dựa trên hồ sơ bệnh lý.
""")

        with tab_ocr:
            is_ground_truth = ocr_result.get("is_ground_truth", False)
            is_real_ocr = ocr_result.get("is_real_ocr", False)

            if is_mode_a:
                # ── Mode A: LLM-only — chỉ hiện văn bản đầu vào ──
                st.markdown("#### Văn bản Đầu vào cho LLM")
                if is_ground_truth:
                    st.info(
                        "**Nguồn dữ liệu:** Nhãn Chuẩn (Ground Truth Benchmark) — "
                        "văn bản thành phần đã được chuyên gia kiểm chứng thủ công, độ chính xác 100%."
                    )
                elif is_real_ocr:
                    st.info("**Nguồn dữ liệu:** PaddleOCR quét trực tiếp từ ảnh bao bì.")
                st.warning(
                    "**Cấu hình A: LLM-only** — Không sử dụng pipeline OCR phục hồi chính tả. "
                    "LLM nhận văn bản thô và trả lời trực tiếp dựa trên kiến thức riêng, không tra cứu RAG."
                )
                st.image(annotated_img, caption="Ảnh bao bì gốc", use_column_width=True)
                st.text_area("Văn bản đầu vào LLM nhận được:", raw_text, height=120, disabled=True, key="ocr_input_a")

                st.markdown(f"##### 📋 Danh sách cụm thành phần bóc tách ({len(mentions)} mục):")
                for idx, m in enumerate(mentions, 1):
                    st.markdown(f"{idx}. `{m['raw_mention']}` *(Phân loại sơ bộ: {m['category']})*")

            elif is_mode_b:
                # ── Mode B: Ground Truth + RAG — hiện nhãn chuẩn, không cần pipeline OCR ──
                st.markdown("#### Dữ liệu Nhãn Chuẩn (Ground Truth)")
                if is_ground_truth:
                    st.info(
                        "**Nguồn dữ liệu:** Nhãn Chuẩn (Ground Truth Benchmark) — "
                        "văn bản thành phần đã được chuyên gia kiểm chứng thủ công từ bao bì thực tế, "
                        "độ chính xác 100%. Dùng làm chuẩn đối chiếu (gold standard) "
                        "để đánh giá hiệu năng RAG trong nghiên cứu ablation study."
                    )
                elif is_real_ocr:
                    st.info(
                        "**Nguồn dữ liệu:** PaddleOCR quét trực tiếp từ ảnh tải lên "
                        "(ảnh này không có nhãn chuẩn Ground Truth sẵn)."
                    )
                st.image(annotated_img, caption="Ảnh bao bì", use_column_width=True)
                st.text_area("Văn bản thành phần đầu vào cho RAG:", raw_text, height=120, disabled=True, key="ocr_input_b")

                st.markdown(f"##### 📋 Danh sách cụm thành phần bóc tách ({len(mentions)} mục):")
                for idx, m in enumerate(mentions, 1):
                    st.markdown(f"{idx}. `{m['raw_mention']}` *(Phân loại sơ bộ: {m['category']})*")

            else:
                # ── Mode C: Full Pipeline — hiện đầy đủ OCR + phục hồi chính tả ──
                st.markdown(f"#### Chế độ Hoạt động: `{ocr_result.get('engine_name', 'Mặc định')}`")

                st.image(annotated_img, caption="Ảnh bao bì đã qua xử lý quang học và vẽ Bounding Box màu xanh lá", use_column_width=True)

                ocr_raw = ocr_result.get("raw_text_before_norm", "")
                ocr_dict_text = ocr_result.get("text_after_ocr_dict", "")
                llm_used = ocr_result.get("llm_spell_used", False)

                if ocr_raw and ocr_raw != raw_text:
                    st.markdown("##### 🔄 Pipeline Phục hồi Chính tả OCR (3 giai đoạn):")
                    st.markdown("""
> **Giai đoạn 1 — OCR (PaddleOCR):** Nhận diện ký tự quang học từ ảnh bao bì.
>
> **Giai đoạn 2 — Dictionary Correction (Từ điển Thực phẩm):** Tách từ dính, chuẩn hóa INS, khôi phục dấu thanh bằng kỹ thuật Diacritic-Stripping + Food Domain Lexicon.
>
> **Giai đoạn 3 — LLM Polish (Gemini Flash):** Chỉ sửa các lỗi còn sót mà từ điển không bao phủ được (ký tự lạ, từ chuyên ngành, tên hóa chất).
""")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**① Văn bản OCR thô**")
                        st.text_area("PaddleOCR output:", ocr_raw, height=150, disabled=True, key="ocr_raw")
                    with col2:
                        st.markdown("**② Sau Dictionary Correction**")
                        st.text_area("Từ điển thực phẩm đã sửa:", ocr_dict_text, height=150, disabled=True, key="ocr_dict")
                    with col3:
                        if llm_used:
                            st.markdown("**③ Sau LLM Polish** ✅")
                            st.text_area("Gemini Flash đã sửa nốt:", raw_text, height=150, disabled=True, key="ocr_llm")
                        else:
                            st.markdown("**③ LLM Polish** ⚠️")
                            st.text_area("Chưa kích hoạt LLM:", ocr_dict_text, height=150, disabled=True, key="ocr_llm")
                            st.caption("💡 Đặt `GEMINI_API_KEY` để AI sửa nốt các lỗi còn sót.")
                else:
                    st.text_area("Toàn bộ văn bản trích xuất (Full Text):", raw_text, height=100)

                st.markdown("##### 🔍 Chi tiết từng dòng văn bản quét được:")
                with st.expander("Bấm để xem danh sách dòng chữ kèm điểm tin cậy OCR", expanded=False):
                    for idx, line in enumerate(ocr_result.get("lines", []), 1):
                        st.markdown(f"- `[{idx:02d}]` (Conf: **{line['confidence']}**) : {line['text']}")

                st.markdown(f"##### 📋 Danh sách cụm thành phần bóc tách ({len(mentions)} mục):")
                for idx, m in enumerate(mentions, 1):
                    st.markdown(f"{idx}. `{m['raw_mention']}` *(Phân loại sơ bộ: {m['category']})*")

        with tab_datahub:
            st.markdown("#### 📁 Tra cứu Toàn bộ 16 Nguồn Dữ liệu Nghiên cứu (Data Hub):")
            st.caption("Xem trước nội dung bảng dữ liệu cấu trúc thực tế từ thư mục `data/`:")

            with st.expander("📊 Thống kê Quần thể NHANES (Biomarker Population Stats)"):
                nhanes_stats = get_nhanes_population_stats(DATA_DIR)
                if nhanes_stats and nhanes_stats.get("total_participants", 0) > 0:
                    st.write(f"**Tổng số người tham gia:** {nhanes_stats['total_participants']:,}")
                    stats_rows = []
                    # Hiển thị nhân khẩu học từ P_DEMO
                    demo = nhanes_stats.get("demographics", {})
                    if demo:
                        d_cols = st.columns(5)
                        d_cols[0].metric("Trẻ < 6 tuổi", demo.get("children_under_6", 0))
                        d_cols[1].metric("Trẻ < 18 tuổi", demo.get("children_under_18", 0))
                        d_cols[2].metric("Thai phụ", demo.get("pregnant", 0))
                        d_cols[3].metric("Nam", demo.get("male", 0))
                        d_cols[4].metric("Nữ", demo.get("female", 0))

                    label_map = {
                        "age": ("Tuổi (năm)", "Trẻ em: <6, <18"),
                        "hba1c": ("HbA1c (%)", "Ngưỡng ĐTĐ: ≥6.5%"),
                        "systolic_bp": ("Huyết áp tâm thu (mmHg)", "Ngưỡng THA: ≥140"),
                        "bmi": ("BMI (kg/m²)", "Ngưỡng béo phì: ≥30"),
                        "bun": ("BUN (mmol/L)", "Ngưỡng CKD: >7.1"),
                    }
                    for key, (label, threshold) in label_map.items():
                        s = nhanes_stats.get(key, {})
                        if s:
                            stats_rows.append({
                                "Chỉ số": label,
                                "Trung bình": f"{s['mean']:.2f}",
                                "Độ lệch chuẩn": f"{s['std']:.2f}",
                                "P25": f"{s['p25']:.2f}",
                                "P75": f"{s['p75']:.2f}",
                                "N": s["n"],
                                "Ngưỡng bệnh lý": threshold,
                            })
                    if stats_rows:
                        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("Chưa có dữ liệu NHANES thô trong thư mục data/.")

            all_dataset_options = [
                "1. Phụ gia Chuẩn hóa & Quy chuẩn (additives_standard)",
                "2. Hồ sơ Bệnh án Người dùng (nhanes_personas)",
                "3. Dữ liệu Mối nguy & Thu hồi Thực phẩm (incidents_train)",
                "4. Dữ liệu Bản thể học Thực phẩm (foodon_entities.json)",
                "5. Nhãn Hàng Đóng gói Việt Nam (openfoodfacts_vietnam)",
                "6. Danh mục Món ăn Khảo sát USDA (food)",
                "7. Từ điển Dinh dưỡng USDA (nutrient)",
                "8. Giá trị Dinh dưỡng Định lượng USDA (food_nutrient)",
                "8b. Phân nhóm Thực phẩm WWEIA (food_category)",
                "8c. Khẩu phần Chuẩn USDA (food_portion)",
                "9. NHANES: Glycohemoglobin HbA1c (P_GHB)",
                "10. NHANES: Fasting Glucose (P_GLU)",
                "11. NHANES: Thể chất & BMI (P_BMX)",
                "12. NHANES: Huyết áp (P_BPXO)",
                "13. NHANES: Sinh hóa Gan Thận (P_BIOPRO)",
                "14. NHANES: Nhân khẩu học (P_DEMO)",
                "15. NHANES: Thói quen Ăn uống (P_DBQ)",
            ]

            selected_ds = st.selectbox("Chọn tệp dữ liệu để xem trước bảng chi tiết:", all_dataset_options)

            st.markdown("#### Tra cứu trực tiếp toàn bộ FoodOn")
            st.caption(
                f"Đã nạp {len(foodon_entities):,} thuật ngữ. Tra cứu theo nhãn tiếng Anh hoặc CURIE; "
                "kết quả tra cứu không tự động gán mã nếu nhãn không khớp chính xác."
            )
            foodon_query = st.text_input(
                "Tên khái niệm hoặc CURIE FoodOn:",
                key="foodon_catalog_query",
                placeholder="Ví dụ: food additive hoặc FOODON:00001002",
            )
            if foodon_query.strip():
                foodon_matches = search_foodon_entities(foodon_query, foodon_entities)
                if foodon_matches:
                    st.dataframe(
                        pd.DataFrame(foodon_matches),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Không tìm thấy nhãn hoặc CURIE nào phù hợp.")

            with st.expander("📚 Knowledge base thành phần và nhóm dị nguyên"):
                ingredient_kb_path = os.path.join(
                    BASE_DIR, "knowledge_base", "ingredients_standard.json"
                )
                if os.path.isfile(ingredient_kb_path):
                    with open(ingredient_kb_path, "r", encoding="utf-8") as source:
                        ingredient_records = json.load(source)
                    common_count = sum(
                        item.get("ingredient_type") == "common_food_ingredient"
                        for item in ingredient_records
                    )
                    st.caption(
                        f"{common_count} thành phần có FoodOn exact match; alias và liên kết "
                        "dị nguyên đang chờ rà soát, chưa dùng để kết luận thành phần."
                    )
                    st.dataframe(
                        pd.DataFrame(ingredient_records),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.error(
                        f"Không tìm thấy knowledge base thành phần: {ingredient_kb_path}"
                    )

            file_map_keys = {
                "1. Phụ gia": ["additives_standard"],
                "2. Hồ sơ": ["nhanes_personas"],
                "3. Dữ liệu Mối nguy": ["incidents_train"],
                "4. Dữ liệu Bản thể": ["foodon_entities"],
                "5. Nhãn Hàng": ["openfoodfacts_vietnam"],
                "6. Danh mục Món": ["food"],
                "7. Từ điển": ["nutrient"],
                "8. Giá trị Dinh dưỡng": ["food_nutrient"],
                "8b. Phân nhóm": ["food_category"],
                "8c. Khẩu phần": ["food_portion"],
                "9. NHANES: Glycohemoglobin": ["P_GHB"],
                "10. NHANES: Fasting": ["P_GLU"],
                "11. NHANES: Thể chất": ["P_BMX"],
                "12. NHANES: Huyết áp": ["P_BPXO"],
                "13. NHANES: Sinh hóa": ["P_BIOPRO"],
                "14. NHANES: Nhân khẩu": ["P_DEMO"],
                "15. NHANES: Thói quen": ["P_DBQ"],
            }

            matched_prefix = next((k for k in file_map_keys if k in selected_ds), None)
            if matched_prefix:
                candidates = file_map_keys[matched_prefix]
                actual_file = find_data_file(DATA_DIR, candidates)

                if actual_file:
                    st.write(f"📄 **Tệp tìm thấy:** `{os.path.basename(actual_file)}` *(Dung lượng: {os.path.getsize(actual_file)/1024:.1f} KB)*")
                    if actual_file.lower().endswith(".json"):
                        try:
                            with open(actual_file, "r", encoding="utf-8") as jf:
                                j_content = json.load(jf)
                                if isinstance(j_content, list):
                                    st.dataframe(pd.DataFrame(j_content[:100]), use_container_width=True)
                                else:
                                    st.json(j_content)
                        except Exception as e:
                            st.error(f"Lỗi đọc JSON: {e}")
                    else:
                        df_view = read_table_flexibly(actual_file)
                        st.dataframe(df_view.head(100), use_container_width=True)
                        st.caption(f"Hiển thị tối đa 100 dòng đầu tiên trên tổng số {len(df_view):,} bản ghi.")
                else:
                    st.warning(f"Chưa tìm thấy tệp dữ liệu tương ứng với '{selected_ds}' trong thư mục `data/`.")

    elif not img_display:
        st.warning("Vui lòng chọn hoặc tải ảnh bao bì thực phẩm để bắt đầu phân tích.")
    else:
        st.info("👈 Hãy chọn một ảnh mẫu và bấm **'Phân tích Nhãn & Đánh giá An toàn Sức khỏe'** để chạy demo thử nghiệm.")
