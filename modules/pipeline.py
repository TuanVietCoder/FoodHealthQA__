"""
Module Pipeline cho hệ thống FoodHealthQA (v10 - Full Algorithm Implementation)
Kế thừa kiến trúc:
- Tầng 1: Perception (Hỗ trợ Ground Truth Benchmark hoặc PaddleOCR kèm Post-OCR Normalization)
- Tầng 2: Normalization (FoodOntoRAG 4-agent Entity Linking đầy đủ)
  + Agent 1: Hybrid Retriever (BM25 + TF-IDF/SBERT Semantic + Fusion)
  + Agent 2: Selector Agent (Exact-match preference + Specificity rule)
  + Agent 3: Rule match scorer (điểm khớp theo luật τ, không gọi LLM)
  + Agent 4: Synonym Generator & Feedback Loop
  + Hỗ trợ mở rộng nạp đầy đủ 4.6MB Food Ontology (foodon_entities.json)
- Tầng 3: Hybrid Retrieval & Data Hub:
  + Tra cứu Cơ sở Tri thức Phụ gia & Quy chuẩn TT24 / Codex (additives_standard.csv / additives_mini.json)
  + Đối soát Mối nguy Sự cố Thực phẩm (incidents_train.csv - Food Hazard Detector)
  + Ràng buộc Số học pFoodReQ: Hàm chỉ thị khoảng I_g(v) cho dinh dưỡng định lượng FNDDS
  + Tra cứu Nhãn Sản phẩm & Mã vạch Open Food Facts VN (openfoodfacts_vietnam.csv)
  + Hỗ trợ kiểm tra bộ dữ liệu thô lâm sàng NHANES (P_DEMO, P_GHB, P_GLU, P_BIOPRO, P_BMX, P_BPXO, P_DBQ)
- Tầng 4: Grounded Generation & Guardrail:
  + Negation Constraint Modeling: Gán nhãn PERMITTED/RESTRICTED/CONTRAINDICATED/ALLERGEN_RISK
  + Kiểm tra Chống chỉ định Phụ gia & Dị ứng lâm sàng (personas.json / nhanes_personas.csv)
  + Cảnh báo Xung đột Ngưỡng Dinh dưỡng Số học (Natri, Calo, Đường) qua hàm I_g(v)
  + Tính điểm tin cậy kép S_total và kích hoạt cơ chế từ chối (Principled Abstention)
"""

import os
import time
import json
import logging
import copy
import re
import unicodedata
from copy import deepcopy
import pandas as pd
from modules.ocr_extractor import (
    extract_with_paddle_ocr,
    get_ground_truth_sample,
    run_perception,
    is_paddleocr_available
)
from modules.file_resolver import find_data_file, read_table_flexibly
from modules.foodon import enrich_additive_foodon_curie, load_foodon_entities
from modules import abstention
from modules import dynamic_qa
from modules import infant_rules
from modules import llm_keys
from modules.benchmark import validate_citations, canonicalize_citation, is_recognized_citation, extract_citations
from modules.hybrid_retriever import foodontorag_4agent_linking
from modules.constraint_model import (
    contains_label_term,
    evaluate_nutritional_constraints,
    apply_negation_constraints,
    CONSTRAINT_CONTRAINDICATED,
    CONSTRAINT_ALLERGEN_RISK,
    CONSTRAINT_PRECAUTIONARY_RISK,
    CONSTRAINT_RESTRICTED,
    is_exempt_refined_oil,
    DERIVATIVE_ALLERGEN_MAP,
    evaluate_adi_compliance,
    evaluate_efsa_risk,
    evaluate_regulatory_compliance,
    ALLERGEN_KEYWORDS,
    VERIFIED_ADDITIVE_CONDITIONS,
    strip_allergen_noise,
    allergen_priority_info,
    PRECAUTION_KEYWORDS,
    _ALLERGEN_GROUPS,
)

_LOGGER = logging.getLogger(__name__)


def load_knowledge_base(kb_path_or_dir):
    """
    Nạp cơ sở tri thức phụ gia thực phẩm chuẩn hóa.
    Ưu tiên: knowledge_base/additives_standard.json → data/additives_mini.json → data/additives_standard.csv
    Đồng thời tự động nạp tích hợp foodon_entities.json (nếu có trong thư mục data).
    """
    data_dir = kb_path_or_dir if os.path.isdir(kb_path_or_dir) else os.path.dirname(kb_path_or_dir) or "data"

    # Ưu tiên knowledge_base/ trước
    project_root = os.path.dirname(data_dir)
    kb_dir = os.path.join(project_root, "knowledge_base")
    target_file = None
    if os.path.isdir(kb_dir):
        target_file = find_data_file(kb_dir, ["additives_standard", "additives"])

    if not target_file:
        target_file = find_data_file(data_dir, ["additives_standard", "additives"])
    if not target_file:
        if os.path.isfile(kb_path_or_dir):
            target_file = kb_path_or_dir
        else:
            raise FileNotFoundError(f"Không tìm thấy cơ sở tri thức phụ gia trong: {data_dir}")

    additives = []
    if target_file.lower().endswith(".json"):
        with open(target_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            if "canonical_name" not in item:
                item["canonical_name"] = item.get("canonical_name_vi", item.get("canonical_name_en", ""))
            if "common_names" not in item:
                item["common_names"] = item.get("synonyms", [])
            if "jecfa_adi" not in item:
                item["jecfa_adi"] = item.get("adi_value", "")
            adi_assessment = item.get("adi_assessment")
            if adi_assessment:
                item["jecfa_adi"] = adi_assessment.get("display_text", item["jecfa_adi"])
        additives = raw
    else:
        df = read_table_flexibly(target_file)
        for _, row in df.iterrows():
            synonyms_str = str(row.get("synonyms", ""))
            synonyms = [s.strip() for s in re.split(r"[,;]", synonyms_str) if s.strip()]

            contra_str = str(row.get("contraindications_summary", ""))
            contraindications = []
            if contra_str:
                parts = contra_str.split(";")
                for p in parts:
                    if ":" in p:
                        cond, msg = p.split(":", 1)
                        sev = "CRITICAL" if any(k in msg.upper() for k in ["CHỐNG CHỈ ĐỊNH", "TUYỆT ĐỐI", "CRITICAL"]) else "WARNING"
                        contraindications.append({
                            "condition": cond.strip().lower(),
                            "severity": sev,
                            "message": msg.strip()
                        })

            additives.append({
                "ins_code": str(row.get("ins_code", "")).strip(),
                "e_number": str(row.get("e_number", "")).strip(),
                "canonical_name": str(row.get("canonical_name_vi", row.get("canonical_name", ""))).strip(),
                "canonical_name_en": str(row.get("canonical_name_en", "")).strip(),
                "common_names": synonyms,
                "foodon_curie": str(row.get("foodon_curie", "FOODON:00000000")).strip(),
                "function": str(row.get("function", "")).strip(),
                "jecfa_adi": str(row.get("adi_value", row.get("jecfa_adi", ""))).strip(),
                "vn_regulation": str(row.get("vn_regulation", "Thông tư 24/2019/TT-BYT")).strip(),
                "codex_standard": str(row.get("codex_standard", "Codex STAN 192-1995")).strip(),
                "contraindications": contraindications
            })

    foodon_entities = load_foodon_entities(data_dir)
    if foodon_entities:
        enrich_additive_foodon_curie(additives, foodon_entities)

    return additives


def load_personas(personas_path_or_dir):
    """
    Nạp danh sách hồ sơ sức khỏe và bệnh lý người dùng (NHANES Personas).
    Hỗ trợ cả personas.json (tệp có sẵn trong thư mục người dùng) lẫn nhanes_personas.csv.
    """
    data_dir = personas_path_or_dir if os.path.isdir(personas_path_or_dir) else os.path.dirname(personas_path_or_dir) or "data"


    target_file = find_data_file(data_dir, ["nhanes_personas", "personas", "nhanes_user_personas"])
    if not target_file:
        if os.path.isfile(personas_path_or_dir):
            target_file = personas_path_or_dir
        else:
            raise FileNotFoundError(f"Không tìm thấy file personas trong: {data_dir}")

    if target_file.lower().endswith(".json"):
        with open(target_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        df = read_table_flexibly(target_file)
        personas = []
        for _, row in df.iterrows():
            cond_str = str(row.get("conditions", ""))
            conditions = [c.strip().lower() for c in re.split(r"[,;]", cond_str) if c.strip()]

            allergy_str = str(row.get("allergies", ""))
            allergies = [a.strip().lower() for a in re.split(r"[,;]", allergy_str) if a.strip()]

            personas.append({
                "id": str(row.get("persona_id", row.get("id", ""))).strip(),
                "persona_id": str(row.get("persona_id", row.get("id", ""))).strip(),
                "name": str(row.get("name", "")).strip(),
                "description": str(row.get("description", "")).strip(),
                "conditions": conditions,
                "allergies": allergies,
                "demographics": {
                    "age": int(row.get("age", 30)) if str(row.get("age", "")).isdigit() else 30,
                    "gender": str(row.get("gender", "Unknown")).strip()
                },
                "clinical_metrics": {
                    "bmi": float(row.get("bmi", 22.0)) if str(row.get("bmi", "")).replace(".", "", 1).isdigit() else 22.0,
                    "systolic_bp": int(row.get("systolic_bp", 120)) if str(row.get("systolic_bp", "")).isdigit() else 120,
                    "diastolic_bp": int(row.get("diastolic_bp", 80)) if str(row.get("diastolic_bp", "")).isdigit() else 80,
                    "fasting_glucose_mmol": float(row.get("fasting_glucose_mmol", 5.0)) if str(row.get("fasting_glucose_mmol", "")).replace(".", "", 1).isdigit() else 5.0,
                    "hba1c_pct": float(row.get("hba1c_pct", 5.0)) if str(row.get("hba1c_pct", "")).replace(".", "", 1).isdigit() else 5.0,
                    "bun_mmol": float(row.get("bun_mmol", 5.0)) if str(row.get("bun_mmol", "")).replace(".", "", 1).isdigit() else 5.0
                },
                "special_diets": [d.strip() for d in str(row.get("special_diets", "")).split(",") if d.strip()]
            })
        return personas


def load_food_hazards(hazards_path_or_dir):
    """
    Nạp cơ sở dữ liệu sự cố & mối nguy thực phẩm (incidents_train.csv - Food Hazard Detector).
    """
    data_dir = hazards_path_or_dir if os.path.isdir(hazards_path_or_dir) else os.path.dirname(hazards_path_or_dir) or "data"


    target_file = find_data_file(data_dir, ["incidents_train", "food_hazards", "incidents"])
    if not target_file:
        return []

    if target_file.lower().endswith(".json"):
        with open(target_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        df = read_table_flexibly(target_file)
        hazards = []
        for _, row in df.iterrows():
            hazards.append({
                "title": str(row.get("title", "")).strip(),
                "product": str(row.get("product", "")).strip(),
                "product_category": str(row.get("product-category", row.get("product_category", ""))).strip(),
                "hazard": str(row.get("hazard", "")).strip(),
                "hazard_category": str(row.get("hazard-category", row.get("hazard_category", ""))).strip()
            })
        return hazards


def load_fndds_nutrition(fndds_path_or_dir):
    """
    Nạp dữ liệu dinh dưỡng USDA FNDDS bằng JOIN:
    food.csv + food_nutrient.csv + nutrient.csv + food_category.csv + food_portion.csv
    """
    data_dir = fndds_path_or_dir if os.path.isdir(fndds_path_or_dir) else os.path.dirname(fndds_path_or_dir) or "data"


    food_file = find_data_file(data_dir, ["food"])
    fn_file = find_data_file(data_dir, ["food_nutrient"])
    nutrient_file = find_data_file(data_dir, ["nutrient"])

    if not (food_file and fn_file and nutrient_file):
        return []

    df_food = read_table_flexibly(food_file)
    df_fn = read_table_flexibly(fn_file)
    df_nut = read_table_flexibly(nutrient_file)

    target_nutrient_numbers = {
        208.0: "calories_kcal",
        203.0: "protein_g",
        204.0: "fat_g",
        269.0: "sugars_g",
        307.0: "sodium_mg",
        291.0: "fiber_g",
        606.0: "saturated_fat_g",
    }

    for col in ["nutrient_nbr"]:
        if col in df_nut.columns:
            df_nut[col] = pd.to_numeric(df_nut[col], errors="coerce")
    for col in ["nutrient_id", "fdc_id", "amount"]:
        if col in df_fn.columns:
            df_fn[col] = pd.to_numeric(df_fn[col], errors="coerce")

    nbr_to_col = {}
    for nutrient_number, column_name in target_nutrient_numbers.items():
        matches = df_nut[
            (df_nut["nutrient_nbr"] - nutrient_number).abs() < 1e-9
        ]
        if matches.empty:
            continue
        food_nutrient_id = int(nutrient_number)
        nbr_to_col[food_nutrient_id] = column_name

    target_nbrs = list(nbr_to_col)
    df_fn_filtered = df_fn[df_fn["nutrient_id"].isin(target_nbrs)].copy()

    pivot = df_fn_filtered.pivot_table(
        index="fdc_id", columns="nutrient_id", values="amount", aggfunc="first"
    )
    pivot.columns = [nbr_to_col.get(c, f"nut_{c}") for c in pivot.columns]
    pivot = pivot.reset_index()

    if "fdc_id" in df_food.columns:
        df_food["fdc_id"] = pd.to_numeric(df_food["fdc_id"], errors="coerce")
    if "food_category_id" in df_food.columns:
        df_food["food_category_id"] = pd.to_numeric(df_food["food_category_id"], errors="coerce")

    merged = df_food.merge(pivot, on="fdc_id", how="inner")

    # JOIN food_category.csv → tên nhóm WWEIA
    cat_file = find_data_file(data_dir, ["food_category"])
    if cat_file:
        df_cat = read_table_flexibly(cat_file)
        if "wweia_food_category" in df_cat.columns:
            df_cat["wweia_food_category"] = pd.to_numeric(df_cat["wweia_food_category"], errors="coerce")
            df_cat = df_cat.rename(columns={
                "wweia_food_category": "food_category_id",
                "wweia_food_category_description": "food_category",
            })
            merged = merged.merge(df_cat[["food_category_id", "food_category"]], on="food_category_id", how="left")

    # JOIN food_portion.csv → khẩu phần chuẩn (lấy portion đầu tiên cho mỗi food)
    portion_file = find_data_file(data_dir, ["food_portion"])
    if portion_file:
        df_portion = read_table_flexibly(portion_file)
        if "fdc_id" in df_portion.columns:
            df_portion["fdc_id"] = pd.to_numeric(df_portion["fdc_id"], errors="coerce")
            df_portion["gram_weight"] = pd.to_numeric(df_portion.get("gram_weight", pd.Series(dtype=float)), errors="coerce")
            df_portion_first = df_portion.dropna(subset=["gram_weight"])
            df_portion_first = df_portion_first[df_portion_first["gram_weight"] > 0]
            df_portion_first = df_portion_first.sort_values("seq_num").drop_duplicates(subset=["fdc_id"], keep="first")
            df_portion_first = df_portion_first[["fdc_id", "portion_description", "gram_weight"]]
            merged = merged.merge(df_portion_first, on="fdc_id", how="left")

    records = []
    has_category = "food_category" in merged.columns
    has_portion = "portion_description" in merged.columns

    for _, r in merged.iterrows():
        rec = {
            "food_code": str(int(r["fdc_id"])) if pd.notna(r.get("fdc_id")) else "",
            "food_description": str(r.get("description", "")),
        }
        for col in nbr_to_col.values():
            rec[col] = float(r[col]) if col in r and pd.notna(r.get(col)) else 0.0
        if has_category and pd.notna(r.get("food_category")):
            rec["food_category"] = str(r["food_category"])
        if has_portion:
            rec["portion_description"] = str(r.get("portion_description", "")) if pd.notna(r.get("portion_description")) else ""
            rec["gram_weight"] = float(r["gram_weight"]) if pd.notna(r.get("gram_weight")) else 0.0
        records.append(rec)
    return records


def load_openfoodfacts_vietnam(off_path_or_dir):
    """
    Nạp dữ liệu nhãn sản phẩm và mã vạch Open Food Facts VN (openfoodfacts_vietnam.csv).
    """
    data_dir = off_path_or_dir if os.path.isdir(off_path_or_dir) else os.path.dirname(off_path_or_dir) or "data"


    target_file = find_data_file(data_dir, ["openfoodfacts_vietnam", "openfoodfacts"])
    if not target_file:
        return []

    df = read_table_flexibly(target_file)
    return df.to_dict(orient="records")


def check_nhanes_raw_files(data_dir="data"):
    """
    Kiểm tra các tệp dữ liệu thô lâm sàng NHANES có mặt trong thư mục data hay không.
    (P_DEMO, P_GHB, P_GLU, P_BIOPRO, P_BMX, P_BPXO, P_DBQ)
    """
    nhanes_names = ["P_DEMO", "P_GHB", "P_GLU", "P_BIOPRO", "P_BMX", "P_BPXO", "P_DBQ"]
    found = {}
    for name in nhanes_names:
        p = find_data_file(data_dir, [name])
        found[name] = p is not None
    return found


def load_nhanes_biomarkers(data_dir="data"):
    """
    Nạp và tổng hợp chỉ số sinh hóa + nhân khẩu học + thói quen ăn uống từ các tệp NHANES thô:
    - P_GHB: HbA1c (LBXGH) → đánh giá Đái tháo đường
    - P_GLU: Đường huyết lúc đói (LBXGLU) → sàng lọc tiền ĐTĐ
    - P_BPXO: Huyết áp tâm thu/tâm trương (BPXOSY1/BPXODI1) → Tăng huyết áp
    - P_BMX: BMI (BMXBMI) → Béo phì
    - P_BIOPRO: BUN (LBXSBU) → Chức năng thận
    - P_DEMO: Tuổi (RIDAGEYR), Giới tính (RIAGENDR), Thai kỳ (RIDEXPRG) → children_under_6, pregnancy
    - P_DBQ: Thói quen ăn uống (54 câu hỏi) → special_diets
    Trả về dict {SEQN: {biomarkers...}} để enrich persona.
    """
    biomarkers = {}

    ghb_file = find_data_file(data_dir, ["P_GHB"])
    if ghb_file:
        df = read_table_flexibly(ghb_file)
        for _, r in df.iterrows():
            seqn = r.get("SEQN")
            if pd.notna(seqn):
                seqn = int(seqn)
                biomarkers.setdefault(seqn, {})["hba1c_pct"] = r.get("LBXGH")

    glu_file = find_data_file(data_dir, ["P_GLU"])
    if glu_file:
        df = read_table_flexibly(glu_file)
        for _, r in df.iterrows():
            seqn = r.get("SEQN")
            if pd.notna(seqn):
                seqn = int(seqn)
                biomarkers.setdefault(seqn, {})["fasting_glucose_mg"] = r.get("LBXGLU")
                biomarkers.setdefault(seqn, {})["fasting_glucose_mmol"] = r.get("LBDGLUSI")

    bp_file = find_data_file(data_dir, ["P_BPXO"])
    if bp_file:
        df = read_table_flexibly(bp_file)
        for _, r in df.iterrows():
            seqn = r.get("SEQN")
            if pd.notna(seqn):
                seqn = int(seqn)
                vals = biomarkers.setdefault(seqn, {})
                sys_vals = []
                dia_vals = []
                for c in ["BPXOSY1", "BPXOSY2", "BPXOSY3"]:
                    try:
                        v = float(r.get(c))
                        if pd.notna(v):
                            sys_vals.append(v)
                    except (ValueError, TypeError):
                        pass
                for c in ["BPXODI1", "BPXODI2", "BPXODI3"]:
                    try:
                        v = float(r.get(c))
                        if pd.notna(v):
                            dia_vals.append(v)
                    except (ValueError, TypeError):
                        pass
                if sys_vals:
                    vals["systolic_bp"] = sum(sys_vals) / len(sys_vals)
                if dia_vals:
                    vals["diastolic_bp"] = sum(dia_vals) / len(dia_vals)

    bmx_file = find_data_file(data_dir, ["P_BMX"])
    if bmx_file:
        df = read_table_flexibly(bmx_file)
        for _, r in df.iterrows():
            seqn = r.get("SEQN")
            if pd.notna(seqn):
                seqn = int(seqn)
                if pd.notna(r.get("BMXBMI")):
                    biomarkers.setdefault(seqn, {})["bmi"] = r.get("BMXBMI")

    bio_file = find_data_file(data_dir, ["P_BIOPRO"])
    if bio_file:
        df = read_table_flexibly(bio_file)
        for _, r in df.iterrows():
            seqn = r.get("SEQN")
            if pd.notna(seqn):
                seqn = int(seqn)
                if pd.notna(r.get("LBXSBU")):
                    biomarkers.setdefault(seqn, {})["bun_mmol"] = r.get("LBXSBU")

    # P_DEMO: Nhân khẩu học — tuổi, giới tính, thai kỳ
    demo_file = find_data_file(data_dir, ["P_DEMO"])
    if demo_file:
        df = read_table_flexibly(demo_file)
        for _, r in df.iterrows():
            seqn = r.get("SEQN")
            if pd.notna(seqn):
                seqn = int(seqn)
                vals = biomarkers.setdefault(seqn, {})
                try:
                    age_val = r.get("RIDAGEYR")
                    if pd.notna(age_val) and str(age_val).strip():
                        vals["age"] = int(float(age_val))
                except (ValueError, TypeError):
                    pass
                try:
                    gen_val = r.get("RIAGENDR")
                    if pd.notna(gen_val) and str(gen_val).strip():
                        vals["gender"] = "Male" if int(float(gen_val)) == 1 else "Female"
                except (ValueError, TypeError):
                    pass
                try:
                    prg_val = r.get("RIDEXPRG")
                    if pd.notna(prg_val) and str(prg_val).strip():
                        prg = int(float(prg_val))
                        if prg == 1:
                            vals["pregnant"] = True
                except (ValueError, TypeError):
                    pass

    # P_DBQ: Thói quen ăn uống (Diet Behavior Questionnaire)
    # DBQ700: Tần suất đọc nhãn dinh dưỡng (1=Luôn luôn, 2=Thường, 3=Đôi khi, 4=Hiếm khi, 5=Không bao giờ)
    # DBD895: Số bữa ăn nhanh/tuần
    # DBD900: Số bữa ăn sẵn đóng gói/tuần
    # DBQ930: Có dùng bữa tại trường không (1=Có, 2=Không) — chỉ báo trẻ em
    dbq_file = find_data_file(data_dir, ["P_DBQ"])
    if dbq_file:
        df = read_table_flexibly(dbq_file)
        for _, r in df.iterrows():
            seqn = r.get("SEQN")
            if pd.notna(seqn):
                seqn = int(seqn)
                vals = biomarkers.setdefault(seqn, {})
                diet_habits = []
                try:
                    dbd895 = float(r.get("DBD895", 0))
                    if pd.notna(dbd895) and dbd895 >= 5:
                        diet_habits.append("high_fast_food")
                except (ValueError, TypeError):
                    pass
                try:
                    dbd900 = float(r.get("DBD900", 0))
                    if pd.notna(dbd900) and dbd900 >= 5:
                        diet_habits.append("high_ready_meals")
                except (ValueError, TypeError):
                    pass
                try:
                    dbq700 = float(r.get("DBQ700", 0))
                    if pd.notna(dbq700) and dbq700 <= 2:
                        diet_habits.append("reads_nutrition_labels")
                except (ValueError, TypeError):
                    pass
                if diet_habits:
                    vals["diet_habits"] = diet_habits

    return biomarkers


def derive_conditions_from_biomarkers(biomarker_data):
    """
    Suy diễn bệnh lý từ chỉ số sinh hóa + nhân khẩu học + thói quen NHANES:
    - HbA1c ≥ 6.5% → Diabetes
    - HbA1c 5.7-6.4% → Prediabetes
    - Systolic BP ≥ 140 hoặc Diastolic BP ≥ 90 → Hypertension
    - BMI ≥ 30 → Obesity
    - BMI 25-29.9 → Overweight
    - BUN > 7.1 mmol/L → Chronic Kidney Disease risk
    - Tuổi < 6 (P_DEMO) → children_under_6 (cảnh báo phẩm màu Azo, chất tạo ngọt)
    - Thai kỳ (P_DEMO RIDEXPRG=1) → pregnancy (cảnh báo Aspartame, Caffeine, Quinine)
    - Thói quen ăn nhanh cao (P_DBQ) → high_sodium_risk
    """
    conditions = []
    hba1c = biomarker_data.get("hba1c_pct")
    if hba1c is not None:
        try:
            hba1c = float(hba1c)
            if hba1c >= 6.5:
                conditions.append("Diabetes")
            elif hba1c >= 5.7:
                conditions.append("Prediabetes")
        except (ValueError, TypeError):
            pass

    sbp = biomarker_data.get("systolic_bp")
    dbp = biomarker_data.get("diastolic_bp")
    try:
        if (sbp and float(sbp) >= 140) or (dbp and float(dbp) >= 90):
            conditions.append("Hypertension")
    except (ValueError, TypeError):
        pass

    bmi = biomarker_data.get("bmi")
    if bmi is not None:
        try:
            bmi = float(bmi)
            if bmi >= 30:
                conditions.append("Obesity")
            elif bmi >= 25:
                conditions.append("Overweight")
        except (ValueError, TypeError):
            pass

    bun = biomarker_data.get("bun_mmol")
    if bun is not None:
        try:
            if float(bun) > 7.1:
                conditions.append("Chronic_Kidney_Disease_Stage_3")
        except (ValueError, TypeError):
            pass

    # P_DEMO: Nhân khẩu học
    age = biomarker_data.get("age")
    if age is not None:
        try:
            age = int(age)
            if age < 6:
                conditions.append("children_under_6")
            elif age < 18:
                conditions.append("children_under_18")
        except (ValueError, TypeError):
            pass

    if biomarker_data.get("pregnant"):
        conditions.append("pregnancy")

    # P_DBQ: Thói quen ăn uống
    diet_habits = biomarker_data.get("diet_habits", [])
    if "high_fast_food" in diet_habits or "high_ready_meals" in diet_habits:
        conditions.append("high_sodium_risk")

    return conditions, diet_habits


def get_nhanes_population_stats(data_dir="data"):
    """
    Tính thống kê quần thể NHANES: phân bố các chỉ số sinh hóa.
    Dùng để so sánh hồ sơ persona với mức trung bình dân số.
    """
    biomarkers = load_nhanes_biomarkers(data_dir)
    if not biomarkers:
        return {}

    all_hba1c = [v["hba1c_pct"] for v in biomarkers.values() if "hba1c_pct" in v and pd.notna(v["hba1c_pct"])]
    all_sbp = [v["systolic_bp"] for v in biomarkers.values() if "systolic_bp" in v and pd.notna(v["systolic_bp"])]
    all_bmi = [v["bmi"] for v in biomarkers.values() if "bmi" in v and pd.notna(v["bmi"])]
    all_bun = [v["bun_mmol"] for v in biomarkers.values() if "bun_mmol" in v and pd.notna(v["bun_mmol"])]
    all_age = [v["age"] for v in biomarkers.values() if "age" in v]

    import numpy as np

    def _stats(arr):
        if not arr:
            return {}
        clean = []
        for x in arr:
            try:
                clean.append(float(x))
            except (ValueError, TypeError):
                pass
        if not clean:
            return {}
        a = np.array(clean)
        return {"mean": float(np.mean(a)), "std": float(np.std(a)),
                "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
                "n": len(a)}

    # Thống kê nhân khẩu học từ P_DEMO
    n_children_under_6 = sum(1 for v in biomarkers.values() if v.get("age") is not None and v["age"] < 6)
    n_children_under_18 = sum(1 for v in biomarkers.values() if v.get("age") is not None and v["age"] < 18)
    n_pregnant = sum(1 for v in biomarkers.values() if v.get("pregnant"))
    n_male = sum(1 for v in biomarkers.values() if v.get("gender") == "Male")
    n_female = sum(1 for v in biomarkers.values() if v.get("gender") == "Female")

    return {
        "hba1c": _stats(all_hba1c),
        "systolic_bp": _stats(all_sbp),
        "bmi": _stats(all_bmi),
        "bun": _stats(all_bun),
        "age": _stats(all_age),
        "demographics": {
            "children_under_6": n_children_under_6,
            "children_under_18": n_children_under_18,
            "pregnant": n_pregnant,
            "male": n_male,
            "female": n_female,
        },
        "total_participants": len(biomarkers),
    }


def enrich_persona_with_nhanes(persona, data_dir="data", biomarkers_cache=None):
    """
    Bổ sung thông tin NHANES (P_DEMO, P_DBQ, biomarkers) vào hồ sơ persona.
    Tìm nhóm người tham gia NHANES phù hợp với persona → enrich conditions & demographics.
    biomarkers_cache: dict đã load sẵn từ load_nhanes_biomarkers(), tránh đọc lại file.
    """
    biomarkers = biomarkers_cache if biomarkers_cache is not None else load_nhanes_biomarkers(data_dir)
    if not biomarkers:
        return persona

    enriched = dict(persona)
    conditions = list(persona.get("conditions", []))
    existing_conditions = set(c.lower() for c in conditions)

    # Xác định nhóm NHANES phù hợp dựa trên persona (hỗ trợ cả persona mẫu lẫn persona tự chỉnh)
    persona_id = str(persona.get("id", "")).lower()
    conds_lower = set(str(c).lower() for c in conditions)
    demographics = persona.get("demographics", {})
    user_age = demographics.get("age")
    is_preg = bool(demographics.get("pregnant") or "pregnant" in conds_lower or "pregnancy" in conds_lower or "pregnant" in persona_id)
    has_diabetes = "diabetes" in conds_lower or "diabetes" in persona_id
    has_hypertension = "hypertension" in conds_lower or "hypertension" in persona_id
    has_ckd = any("ckd" in c or "kidney" in c for c in conds_lower) or "ckd" in persona_id
    has_obesity = "obesity" in conds_lower or "obese" in persona_id
    is_child = (user_age is not None and user_age < 18) or "child" in persona_id or "children" in persona_id
    is_healthy = "healthy" in persona_id or (conds_lower == {"healthy"} and not is_preg and not is_child)

    matched_seqns = []

    for seqn, data in biomarkers.items():
        age = data.get("age")
        if age is None:
            continue

        matched = False
        if is_child:
            if (user_age and user_age < 6 and age < 6) or (age < 18):
                matched = True
        elif is_preg:
            if data.get("pregnant"):
                matched = True
        elif has_diabetes:
            hba1c = data.get("hba1c_pct")
            glu = data.get("glucose_fasting_mgdl")
            try:
                if (hba1c and float(hba1c) >= 6.5) or (glu and float(glu) >= 126):
                    matched = True
            except (ValueError, TypeError):
                pass
        elif has_hypertension:
            sbp = data.get("systolic_bp")
            dbp = data.get("diastolic_bp")
            try:
                if (sbp and float(sbp) >= 140) or (dbp and float(dbp) >= 90):
                    matched = True
            except (ValueError, TypeError):
                pass
        elif has_ckd:
            bun = data.get("bun_mmol")
            creat = data.get("creatinine_umol")
            try:
                if (bun and float(bun) > 7.1) or (creat and float(creat) > 115):
                    matched = True
            except (ValueError, TypeError):
                pass
        elif has_obesity:
            bmi = data.get("bmi")
            try:
                if bmi and float(bmi) >= 30:
                    matched = True
            except (ValueError, TypeError):
                pass
        elif is_healthy:
            if 18 <= age <= 40:
                matched = True
        else:
            if user_age and abs(age - user_age) <= 10:
                matched = True

        if matched:
            matched_seqns.append((seqn, data))

    if not matched_seqns:
        return persona

    # Thống kê nhóm NHANES phù hợp
    nhanes_group_size = len(matched_seqns)

    # KHÔNG thêm bệnh/chế độ ăn của người tham gia NHANES vào hồ sơ người dùng. Trước đây bệnh của một người
    # trong khảo sát (vd "Chronic_Kidney_Disease_Stage_3") bị chép vào hồ sơ "trẻ 1 tuổi" rồi kích hoạt luật thận —
    # hồ sơ người dùng chỉ gồm những gì người dùng khai báo. Dữ liệu NHANES chỉ lưu làm tham chiếu, luật không dùng.
    sample_data = matched_seqns[0][1]
    derived, diet_habits = derive_conditions_from_biomarkers(sample_data)
    enriched["conditions"] = conditions
    enriched["nhanes_reference"] = {
        "example_participant_conditions": list(derived),
        "example_participant_diets": list(diet_habits or []),
        "note": "Tham chiếu từ một người tham gia NHANES cùng nhóm; không phải thông tin của người dùng.",
    }
    diet_habits = []

    # Enrich demographics từ P_DEMO
    demographics = enriched.get("demographics", {})
    if sample_data.get("age") is not None:
        demographics["nhanes_age_example"] = sample_data["age"]
    if sample_data.get("gender"):
        demographics["nhanes_gender_example"] = sample_data["gender"]
    enriched["demographics"] = demographics

    # Enrich special_diets từ P_DBQ
    if diet_habits:
        existing_diets = set(enriched.get("special_diets", []))
        existing_diets.update(diet_habits)
        enriched["special_diets"] = list(existing_diets)

    # Metadata cho UI
    enriched["nhanes_enrichment"] = {
        "matched_group_size": nhanes_group_size,
        "source_files": ["P_DEMO.csv", "P_DBQ.csv", "P_GHB.csv", "P_GLU.csv", "P_BPXO.csv", "P_BMX.csv", "P_BIOPRO.csv"],
    }

    return enriched


def cross_reference_openfoodfacts(product_name, mentions, off_products):
    """
    Đối chiếu thành phần quét được với dữ liệu OpenFoodFacts VN:
    - Tìm sản phẩm tương đồng trong OFF
    - So khớp phụ gia (additives_tags) và dị nguyên (allergens_tags)
    - Phát hiện dị nguyên ẩn (undeclared allergens) hoặc phụ gia chưa khai báo
    """
    if not off_products or not product_name:
        return {"matched_products": [], "allergen_cross_ref": [], "additive_cross_ref": []}

    p_lower = product_name.lower()
    mention_texts = [m.get("raw_mention", "").lower() for m in mentions]

    matched = []
    for prod in off_products:
        pn = str(prod.get("product_name", "")).lower()
        if not pn:
            continue
        p_words = set(p_lower.split())
        pn_words = set(pn.split())
        overlap = p_words & pn_words
        if len(overlap) >= 2 or (len(p_words) <= 3 and overlap):
            matched.append(prod)

    allergen_cross = []
    additive_cross = []

    for prod in matched[:5]:
        off_allergens = str(prod.get("allergens_tags", "")).lower()
        off_additives = str(prod.get("additives_tags", "")).lower()

        if off_allergens and off_allergens != "nan":
            allergen_list = [a.strip().replace("en:", "") for a in off_allergens.split(",")]
            for allergen in allergen_list:
                if allergen and not any(allergen in mt for mt in mention_texts):
                    allergen_cross.append({
                        "source": prod.get("product_name", ""),
                        "allergen": allergen,
                        "note": "Dị nguyên khai báo trong OFF nhưng chưa phát hiện trên nhãn OCR"
                    })

        if off_additives and off_additives != "nan":
            add_list = [a.strip().replace("en:", "").upper() for a in off_additives.split(",")]
            for add_code in add_list:
                if add_code:
                    additive_cross.append({
                        "source": prod.get("product_name", ""),
                        "additive_code": add_code,
                        "in_off": True,
                    })

    return {
        "matched_products": [p.get("product_name", "") for p in matched[:5]],
        "allergen_cross_ref": allergen_cross,
        "additive_cross_ref": additive_cross,
    }


_PRODUCT_CATEGORY_MAP = {
    "cookie": ("cereals and bakery products", ["cookie", "biscuit", "bakery", "wafer", "bánh", "cake", "bread"]),
    "bánh": ("cereals and bakery products", ["cookie", "biscuit", "bakery", "wafer", "bánh", "cake", "bread"]),
    "biscuit": ("cereals and bakery products", ["cookie", "biscuit", "bakery", "wafer", "bánh"]),
    "bread": ("cereals and bakery products", ["bread", "bakery", "bánh mì"]),
    "noodle": ("cereals and bakery products", ["noodle", "instant", "pasta", "mì"]),
    "mì": ("cereals and bakery products", ["noodle", "instant", "pasta", "mì"]),
    "beverage": ("non-alcoholic beverages", ["beverage", "soft drink", "cola", "drink", "nước", "juice"]),
    "nước": ("non-alcoholic beverages", ["beverage", "soft drink", "cola", "drink", "nước", "juice"]),
    "cola": ("non-alcoholic beverages", ["cola", "soft drink", "soda"]),
    "lon": ("non-alcoholic beverages", ["canned", "drink", "cola"]),
    "sữa": ("meat, egg and dairy products", ["milk", "dairy", "cheese", "yogurt", "sữa"]),
    "milk": ("meat, egg and dairy products", ["milk", "dairy", "cheese", "yogurt"]),
    "cheese": ("meat, egg and dairy products", ["cheese", "dairy"]),
    "meat": ("meat, egg and dairy products", ["meat", "beef", "pork", "chicken", "thịt"]),
    "thịt": ("meat, egg and dairy products", ["meat", "beef", "pork", "chicken", "thịt"]),
    "seafood": ("seafood", ["fish", "shrimp", "crab", "tôm", "cá", "cua", "seafood"]),
    "tôm": ("seafood", ["shrimp", "prawn", "seafood"]),
    "cá": ("seafood", ["fish", "seafood"]),
    "chocolate": ("cocoa and cocoa preparations, coffee and tea", ["chocolate", "cocoa", "cacao"]),
    "candy": ("confectionery", ["candy", "kẹo", "confectionery", "sweets"]),
    "kẹo": ("confectionery", ["candy", "kẹo", "confectionery"]),
    "kem": ("ices and desserts", ["ice cream", "kem", "dessert", "frozen"]),
    "sauce": ("soups, broths, sauces and condiments", ["sauce", "nước sốt", "condiment", "gia vị"]),
    "nước sốt": ("soups, broths, sauces and condiments", ["sauce", "nước sốt", "condiment"]),
    "nut": ("nuts, nut products and seeds", ["nut", "peanut", "almond", "cashew", "đậu phộng", "hạt"]),
    "snack": ("prepared dishes and snacks", ["snack", "chip", "cracker"]),
}

_HAZARD_INGREDIENT_MAP = {
    "milk": ["sữa", "milk", "bơ", "dairy", "bột sữa", "whey"],
    "egg": ["trứng", "egg"],
    "soy": ["đậu nành", "soy", "soya", "lecithin", "322"],
    "peanut": ["đậu phộng", "peanut", "lạc"],
    "nut": ["hạnh nhân", "hạt", "almond", "walnut", "cashew", "óc chó", "điều"],
    "gluten": ["gluten", "bột mì", "lúa mì", "wheat"],
    "wheat": ["bột mì", "lúa mì", "wheat", "gluten"],
    "fish": ["cá", "fish"],
    "shellfish": ["tôm", "cua", "shrimp", "crab"],
    "sesame": ["mè", "vừng", "sesame"],
    "celery": ["cần tây", "celery"],
    "mustard": ["mù tạt", "mustard"],
    "sulphite": ["sulphite", "sulfite", "SO2", "chất bảo quản"],
    "aspartame": ["aspartame", "951", "phenylalanine"],
    "benzoate": ["benzoat", "benzoate", "211", "natri benzoat"],
    "tartrazine": ["tartrazine", "102", "phẩm màu"],
    "listeria": ["thịt", "sữa", "cheese", "meat", "dairy"],
    "salmonella": ["trứng", "thịt", "egg", "meat", "poultry"],
}

_HAZARD_SEVERITY = {
    "allergens": "HIGH",
    "biological": "HIGH",
    "chemical": "HIGH",
    "food additives and flavourings": "MEDIUM",
    "foreign bodies": "MEDIUM",
    "fraud": "MEDIUM",
    "packaging defect": "LOW",
    "organoleptic aspects": "LOW",
    "migration": "MEDIUM",
}


def check_food_hazards(product_name, mentions, hazards_list):
    """
    Đối soát mối nguy và sự cố thu hồi tiềm ẩn từ tập dữ liệu Food Hazard Detector
    (incidents_train.csv — SemEval-2025 Task 9).
    Hỗ trợ:
    - Đối soát theo nhóm sản phẩm (product-category) với bảng ánh xạ mở rộng
    - Đối soát theo thành phần/dị nguyên (allergens, biological, chemical, food additives)
    - Phân mức nghiêm trọng theo hazard-category
    - Match OR: sản phẩm HOẶC thành phần (không yêu cầu cả hai)
    """
    if not hazards_list:
        return []

    p_name_lower = product_name.lower()
    mentions_text = " ".join([m.get("raw_mention", "").lower() for m in mentions])

    product_keywords = set()
    matched_category = None
    for keyword, (category, prod_terms) in _PRODUCT_CATEGORY_MAP.items():
        if keyword in p_name_lower:
            product_keywords.update(prod_terms)
            matched_category = category
            break

    scored = []
    for item in hazards_list:
        h_prod = item.get("product", "").lower()
        h_sub = item.get("hazard", "").lower()
        h_cat = item.get("hazard_category", "").lower()
        h_prod_cat = item.get("product_category", "").lower()

        score = 0.0

        if product_keywords and any(w in h_prod for w in product_keywords):
            score += 2.0
        elif matched_category and matched_category in h_prod_cat:
            score += 1.5

        for hazard_key, ingredient_kws in _HAZARD_INGREDIENT_MAP.items():
            if hazard_key in h_sub:
                if any(kw in mentions_text for kw in ingredient_kws):
                    score += 3.0
                    break

        if h_cat == "allergens":
            all_allergen_kws = [kw for kws in ALLERGEN_KEYWORDS.values() for kw in kws]
            if any(contains_label_term(mentions_text, kw) for kw in all_allergen_kws):
                score += 1.0

        if h_cat in ("biological", "chemical") and product_keywords:
            if any(w in h_prod for w in product_keywords):
                score += 1.5

        if score > 0:
            severity = _HAZARD_SEVERITY.get(h_cat, "LOW")
            scored.append((score, severity, item))

    scored.sort(key=lambda x: (-x[0], {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x[1], 3)))

    matched = []
    seen_titles = set()
    for _, severity, item in scored[:10]:
        title = item.get("title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        result_item = dict(item)
        result_item["severity"] = severity
        matched.append(result_item)
        if len(matched) >= 5:
            break

    return matched


_FNDDS_KEYWORD_MAP = {
    # Bánh — dài nhất trước
    "bánh quy bơ": ["cookie", "butter", "sugar"],
    "bánh quy socola": ["cookie", "chocolate"],
    "bánh quy": ["cookie", "biscuit"],
    "bánh mì": ["bread", "banh mi"],
    "bánh ngọt": ["cake", "pastry"],
    "bánh": ["cake", "cookie", "bread", "pastry", "bakery", "banh"],
    # Mì / Bún / Phở
    "mì ăn liền": ["ramen noodles", "instant noodle"],
    "mì trứng": ["egg noodle"],
    "mì": ["noodle", "pasta", "ramen"],
    "phở bò": ["pho", "meat"],
    "phở gà": ["pho", "chicken"],
    "phở": ["pho"],
    "bún bò": ["rice noodle", "beef"],
    "bún chả": ["rice noodle", "pork"],
    "bún riêu": ["rice noodle", "crab"],
    "bún": ["rice noodle"],
    # Nước uống
    "nước cam": ["orange juice", "orange"],
    "nước ngọt": ["carbonated", "cola", "soda"],
    "nước ép": ["juice"],
    "nước dừa": ["coconut water"],
    "nước": ["beverage", "drink", "water"],
    "cola": ["cola", "carbonated"],
    "trà đá": ["tea", "iced"],
    "trà sữa": ["tea", "milk"],
    "trà": ["tea"],
    "cà phê sữa": ["coffee", "milk"],
    "cà phê": ["coffee"],
    # Sữa
    "sữa đặc": ["condensed milk"],
    "sữa đậu nành": ["soy milk"],
    "sữa chua": ["yogurt"],
    "sữa tươi": ["whole milk", "milk"],
    "sữa bột": ["milk", "powder"],
    "sữa": ["milk", "dairy"],
    # Thịt / Gia cầm
    "gà nướng": ["chicken", "roasted", "baked"],
    "gà rán": ["chicken", "fried"],
    "gà luộc": ["chicken", "boiled"],
    "thịt gà": ["chicken"],
    "gà": ["chicken"],
    "thịt bò": ["beef"],
    "bò": ["beef"],
    "thịt heo": ["pork"],
    "heo": ["pork"],
    "thịt": ["meat", "beef", "pork", "chicken"],
    "xúc xích": ["sausage", "hot dog"],
    "giò chả": ["pork", "sausage"],
    "chả lụa": ["pork", "sausage"],
    # Hải sản
    "cá hồi": ["salmon"],
    "cá ngừ": ["tuna"],
    "cá": ["fish", "tilapia"],
    "tôm": ["shrimp"],
    "mực": ["squid"],
    "hải sản": ["seafood", "shrimp", "fish"],
    # Trứng / Đậu
    "trứng": ["egg"],
    "đậu phụ": ["tofu"],
    "đậu hũ": ["tofu"],
    # Cơm / Xôi
    "cơm tấm": ["rice", "broken"],
    "cơm chiên": ["fried rice"],
    "cơm": ["rice", "cooked", "white rice"],
    "xôi": ["glutinous rice", "sticky rice"],
    # Cuốn / Chả
    "gỏi cuốn": ["roll", "rice paper", "shrimp"],
    "chả giò": ["spring roll", "fried"],
    "nem cuốn": ["roll", "rice paper"],
    "nem": ["spring roll"],
    # Cháo
    "cháo": ["congee", "porridge"],
    # Gia vị / Nước chấm
    "nước mắm": ["fish sauce"],
    "xì dầu": ["soy sauce"],
    "nước tương": ["soy sauce"],
    "bột ngọt": ["msg", "seasoning"],
    "tương ớt": ["chili sauce", "sriracha"],
    "mayonnaise": ["mayonnaise"],
    "ketchup": ["ketchup"],
    # Snack / Hạt
    "snack": ["chip", "potato chips", "cracker"],
    "khoai tây chiên": ["potato", "fried", "chip"],
    "khoai tây": ["potato"],
    "đậu phộng": ["peanut"],
    "hạt điều": ["cashew"],
    # Dầu / Bơ / Đường
    "dầu ăn": ["oil", "cooking"],
    "dầu": ["oil", "palm oil"],
    "bơ": ["butter"],
    "đường": ["sugar"],
    "mật ong": ["honey"],
    # Kem / Chocolate / Kẹo
    "kem": ["ice cream"],
    "chocolate": ["chocolate", "cocoa", "cacao"],
    "kẹo": ["candy", "confectionery", "sweet"],
    "cookie": ["cookie", "biscuit"],
}

# Bảng Thành phần Thực phẩm Việt Nam (Viện Dinh dưỡng / Bộ Y tế)
# Dữ liệu trên 100g phần ăn được — cho các món truyền thống không có trong FNDDS
_VN_FOOD_COMPOSITION_DB = {
    "bánh chưng": {
        "food_description": "Bánh chưng (Sticky rice cake with pork & mung bean)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 180.0, "protein_g": 5.8, "fat_g": 4.5, "sugars_g": 0.5,
        "sodium_mg": 320.0, "fiber_g": 0.8, "saturated_fat_g": 1.5,
        "portion_description": "1 miếng vừa", "gram_weight": 150.0,
    },
    "bánh tét": {
        "food_description": "Bánh tét (Cylindrical sticky rice cake)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 175.0, "protein_g": 5.2, "fat_g": 4.0, "sugars_g": 0.8,
        "sodium_mg": 280.0, "fiber_g": 0.6, "saturated_fat_g": 1.3,
        "portion_description": "1 khoanh", "gram_weight": 120.0,
    },
    "bánh cuốn": {
        "food_description": "Bánh cuốn (Steamed rice rolls with pork)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 120.0, "protein_g": 5.5, "fat_g": 3.2, "sugars_g": 0.3,
        "sodium_mg": 450.0, "fiber_g": 0.4, "saturated_fat_g": 1.0,
        "portion_description": "1 đĩa", "gram_weight": 200.0,
    },
    "bánh xèo": {
        "food_description": "Bánh xèo (Vietnamese sizzling crepe)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 210.0, "protein_g": 7.5, "fat_g": 12.0, "sugars_g": 1.2,
        "sodium_mg": 380.0, "fiber_g": 1.5, "saturated_fat_g": 4.0,
        "portion_description": "1 cái lớn", "gram_weight": 180.0,
    },
    "bún chả": {
        "food_description": "Bún chả Hà Nội (Grilled pork with rice vermicelli)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 165.0, "protein_g": 11.0, "fat_g": 6.5, "sugars_g": 3.5,
        "sodium_mg": 520.0, "fiber_g": 1.2, "saturated_fat_g": 2.2,
        "portion_description": "1 suất", "gram_weight": 350.0,
    },
    "bún bò huế": {
        "food_description": "Bún bò Huế (Spicy beef noodle soup)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 85.0, "protein_g": 7.2, "fat_g": 3.5, "sugars_g": 1.0,
        "sodium_mg": 680.0, "fiber_g": 0.5, "saturated_fat_g": 1.2,
        "portion_description": "1 tô", "gram_weight": 500.0,
    },
    "bún riêu cua": {
        "food_description": "Bún riêu cua (Crab tomato noodle soup)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 78.0, "protein_g": 6.5, "fat_g": 2.8, "sugars_g": 1.5,
        "sodium_mg": 550.0, "fiber_g": 0.8, "saturated_fat_g": 0.8,
        "portion_description": "1 tô", "gram_weight": 500.0,
    },
    "cơm tấm": {
        "food_description": "Cơm tấm sườn (Broken rice with grilled pork chop)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 195.0, "protein_g": 12.0, "fat_g": 5.5, "sugars_g": 1.0,
        "sodium_mg": 420.0, "fiber_g": 0.8, "saturated_fat_g": 1.8,
        "portion_description": "1 đĩa", "gram_weight": 350.0,
    },
    "hủ tiếu": {
        "food_description": "Hủ tiếu Nam Vang (Phnom Penh-style noodle soup)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 92.0, "protein_g": 7.0, "fat_g": 3.0, "sugars_g": 0.8,
        "sodium_mg": 600.0, "fiber_g": 0.5, "saturated_fat_g": 1.0,
        "portion_description": "1 tô", "gram_weight": 450.0,
    },
    "mì quảng": {
        "food_description": "Mì Quảng (Quang-style turmeric noodle)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 145.0, "protein_g": 9.0, "fat_g": 4.5, "sugars_g": 1.0,
        "sodium_mg": 480.0, "fiber_g": 1.0, "saturated_fat_g": 1.5,
        "portion_description": "1 tô", "gram_weight": 400.0,
    },
    "cao lầu": {
        "food_description": "Cao lầu Hội An (Hoi An-style noodle)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 160.0, "protein_g": 8.5, "fat_g": 5.0, "sugars_g": 2.0,
        "sodium_mg": 520.0, "fiber_g": 1.2, "saturated_fat_g": 1.8,
        "portion_description": "1 tô", "gram_weight": 350.0,
    },
    "chả giò": {
        "food_description": "Chả giò / Nem rán (Fried spring roll)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 220.0, "protein_g": 8.5, "fat_g": 13.0, "sugars_g": 2.0,
        "sodium_mg": 480.0, "fiber_g": 1.2, "saturated_fat_g": 3.5,
        "portion_description": "2 cuốn", "gram_weight": 80.0,
    },
    "gỏi cuốn": {
        "food_description": "Gỏi cuốn (Fresh spring roll with shrimp & pork)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 110.0, "protein_g": 8.0, "fat_g": 2.5, "sugars_g": 1.5,
        "sodium_mg": 380.0, "fiber_g": 1.5, "saturated_fat_g": 0.6,
        "portion_description": "2 cuốn", "gram_weight": 120.0,
    },
    "canh chua": {
        "food_description": "Canh chua cá (Sour fish soup)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 45.0, "protein_g": 5.0, "fat_g": 1.5, "sugars_g": 2.0,
        "sodium_mg": 420.0, "fiber_g": 1.0, "saturated_fat_g": 0.3,
        "portion_description": "1 bát", "gram_weight": 300.0,
    },
    "cá kho tộ": {
        "food_description": "Cá kho tộ (Caramelized fish in clay pot)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 155.0, "protein_g": 18.0, "fat_g": 5.5, "sugars_g": 5.0,
        "sodium_mg": 850.0, "fiber_g": 0.2, "saturated_fat_g": 1.2,
        "portion_description": "1 phần", "gram_weight": 150.0,
    },
    "thịt kho trứng": {
        "food_description": "Thịt kho trứng (Braised pork belly with eggs)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 230.0, "protein_g": 15.0, "fat_g": 16.0, "sugars_g": 4.0,
        "sodium_mg": 780.0, "fiber_g": 0.0, "saturated_fat_g": 5.5,
        "portion_description": "1 phần", "gram_weight": 150.0,
    },
    "gà kho gừng": {
        "food_description": "Gà kho gừng (Chicken braised with ginger)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 180.0, "protein_g": 20.0, "fat_g": 8.0, "sugars_g": 3.5,
        "sodium_mg": 680.0, "fiber_g": 0.3, "saturated_fat_g": 2.2,
        "portion_description": "1 phần", "gram_weight": 150.0,
    },
    "xôi": {
        "food_description": "Xôi (Sticky rice, plain)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 205.0, "protein_g": 4.5, "fat_g": 0.5, "sugars_g": 0.2,
        "sodium_mg": 5.0, "fiber_g": 1.0, "saturated_fat_g": 0.1,
        "portion_description": "1 gói", "gram_weight": 200.0,
    },
    "xôi gấc": {
        "food_description": "Xôi gấc (Sticky rice with gac fruit)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 215.0, "protein_g": 4.0, "fat_g": 2.0, "sugars_g": 3.0,
        "sodium_mg": 8.0, "fiber_g": 1.2, "saturated_fat_g": 0.5,
        "portion_description": "1 gói", "gram_weight": 200.0,
    },
    "cháo gà": {
        "food_description": "Cháo gà (Chicken rice porridge)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 65.0, "protein_g": 4.5, "fat_g": 1.5, "sugars_g": 0.2,
        "sodium_mg": 350.0, "fiber_g": 0.3, "saturated_fat_g": 0.4,
        "portion_description": "1 tô", "gram_weight": 400.0,
    },
    "chè": {
        "food_description": "Chè đậu xanh (Mung bean sweet soup)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 120.0, "protein_g": 3.5, "fat_g": 0.5, "sugars_g": 18.0,
        "sodium_mg": 15.0, "fiber_g": 2.0, "saturated_fat_g": 0.1,
        "portion_description": "1 bát", "gram_weight": 250.0,
    },
    "giò chả": {
        "food_description": "Giò chả / Chả lụa (Vietnamese pork sausage)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 200.0, "protein_g": 22.0, "fat_g": 12.0, "sugars_g": 0.5,
        "sodium_mg": 720.0, "fiber_g": 0.0, "saturated_fat_g": 4.0,
        "portion_description": "100g", "gram_weight": 100.0,
    },
    "chả lụa": {
        "food_description": "Chả lụa (Vietnamese pork roll)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 200.0, "protein_g": 22.0, "fat_g": 12.0, "sugars_g": 0.5,
        "sodium_mg": 720.0, "fiber_g": 0.0, "saturated_fat_g": 4.0,
        "portion_description": "100g", "gram_weight": 100.0,
    },
    "nem chua": {
        "food_description": "Nem chua (Fermented pork roll)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 185.0, "protein_g": 16.0, "fat_g": 11.0, "sugars_g": 3.0,
        "sodium_mg": 680.0, "fiber_g": 0.5, "saturated_fat_g": 3.5,
        "portion_description": "3 miếng", "gram_weight": 60.0,
    },
    "bánh bao": {
        "food_description": "Bánh bao (Steamed bun with pork filling)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 230.0, "protein_g": 9.0, "fat_g": 7.0, "sugars_g": 5.0,
        "sodium_mg": 450.0, "fiber_g": 1.0, "saturated_fat_g": 2.5,
        "portion_description": "1 cái", "gram_weight": 150.0,
    },
    "bánh tráng trộn": {
        "food_description": "Bánh tráng trộn (Mixed rice paper snack)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 160.0, "protein_g": 4.0, "fat_g": 5.0, "sugars_g": 3.0,
        "sodium_mg": 550.0, "fiber_g": 1.5, "saturated_fat_g": 1.0,
        "portion_description": "1 phần", "gram_weight": 120.0,
    },
    "bò lúc lắc": {
        "food_description": "Bò lúc lắc (Shaking beef)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 210.0, "protein_g": 22.0, "fat_g": 12.0, "sugars_g": 2.0,
        "sodium_mg": 550.0, "fiber_g": 0.5, "saturated_fat_g": 4.5,
        "portion_description": "1 đĩa", "gram_weight": 200.0,
    },
    "bò kho": {
        "food_description": "Bò kho (Vietnamese beef stew)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 120.0, "protein_g": 10.0, "fat_g": 5.5, "sugars_g": 2.5,
        "sodium_mg": 600.0, "fiber_g": 1.0, "saturated_fat_g": 2.0,
        "portion_description": "1 tô", "gram_weight": 400.0,
    },
    "cơm chiên": {
        "food_description": "Cơm chiên (Vietnamese fried rice)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 185.0, "protein_g": 6.0, "fat_g": 7.0, "sugars_g": 0.5,
        "sodium_mg": 450.0, "fiber_g": 0.8, "saturated_fat_g": 1.5,
        "portion_description": "1 đĩa", "gram_weight": 300.0,
    },
    "rau muống xào tỏi": {
        "food_description": "Rau muống xào tỏi (Stir-fried water spinach with garlic)",
        "food_category": "Món truyền thống Việt Nam",
        "source": "VCNDD",
        "calories_kcal": 55.0, "protein_g": 3.0, "fat_g": 3.5, "sugars_g": 1.0,
        "sodium_mg": 280.0, "fiber_g": 2.5, "saturated_fat_g": 0.5,
        "portion_description": "1 đĩa", "gram_weight": 150.0,
    },
}


def _match_vn_food(product_name):
    """Tra cứu trực tiếp trong Bảng TPTP Việt Nam (Viện Dinh dưỡng) bằng longest-match."""
    p_lower = product_name.lower()
    sorted_keys = sorted(_VN_FOOD_COMPOSITION_DB.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in p_lower:
            entry = _VN_FOOD_COMPOSITION_DB[key].copy()
            entry["food_code"] = f"VN_{key.replace(' ', '_')}"
            return entry
    return None


def _match_fndds(product_name, fndds_list):
    """Tra cứu trong USDA FNDDS bằng keyword map song ngữ + word overlap."""
    p_lower = product_name.lower()

    matched_terms = []
    sorted_keywords = sorted(_FNDDS_KEYWORD_MAP.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if keyword in p_lower:
            matched_terms.extend(_FNDDS_KEYWORD_MAP[keyword])
            break

    if not matched_terms:
        p_words_check = set(p_lower.split())
        for keyword in sorted_keywords:
            if keyword in p_words_check:
                matched_terms.extend(_FNDDS_KEYWORD_MAP[keyword])
                break

    best_item = None
    best_score = -1
    term_word_sets = [set(t.lower().split()) for t in matched_terms]
    p_words = set(p_lower.split())

    for item in fndds_list:
        desc = str(item.get("food_description", "")).lower()
        cat = str(item.get("food_category", "")).lower()
        desc_words = set(desc.replace(",", " ").replace("(", " ").replace(")", " ").split())
        cat_words = set(cat.replace(",", " ").split())
        score = 0

        for i, term in enumerate(matched_terms):
            tw = term_word_sets[i]
            if tw.issubset(desc_words):
                score += 5 + len(term)
            elif term in desc:
                score += 5 + len(term)
            elif tw.issubset(cat_words):
                score += 3 + len(term)

        score += len(p_words & desc_words)

        if score > best_score:
            best_score = score
            best_item = item

    return best_item, best_score


CODEX_LOW_SODIUM_MG_PER_100G = 120.0   # Codex CAC/GL 23-1997: "low sodium" ≤ 120 mg/100 g
_NUM = r"(\d+(?:[.,]\d+)?)"


def _to_float(text):
    return float(str(text).replace(",", "."))


# chất dinh dưỡng đọc từ nhãn: khóa → (tên trên nhãn, đơn vị)
_LABEL_NUTRIENTS = {
    "sodium_mg": (r"natri|sodium", "mg"),
    "sugars_g": (r"đường tổng số|tổng đường|đường|sugars?", "g"),
}
CODEX_SUGAR_FREE_G_PER_100G = 0.5     # Codex CAC/GL 23-1997: "sugar-free" ≤ 0,5 g/100 g


def _extract_one(low, names, unit):
    head = r"(?:" + names + r")[^0-9]{0,40}?"
    # 1) theo 100 g/ml: "natri 853 mg trên 100 g", "đường tổng số 54,9 g trên 100 g"
    m = re.search(head + _NUM + r"\s*" + unit + r"\s*(?:/|trên|per)\s*100\s*(g|ml)\b", low)
    if m:
        return _to_float(m.group(1)), f"100 {m.group(2)}", m.group(0)
    # 2) theo khẩu phần: "natri 35 mg trên 1 khẩu phần 15 g", "59,5 mg trên 170 ml (1 hộp)"
    m = re.search(head + _NUM + r"\s*" + unit + r"\s*(?:trên|/|per|cho|trong)\s*"
                  r"(?:1\s+)?(?:khẩu phần|kp|gói|ly|hộp|chai|lon|thanh|cái|viên|serving|portion)?[^0-9]{0,12}?"
                  + _NUM + r"\s*(g|ml)\b", low)
    if m and _to_float(m.group(2)) > 0:
        return (round(_to_float(m.group(1)) * 100.0 / _to_float(m.group(2)), 2),
                f"quy đổi từ {m.group(2)} {m.group(3)}", m.group(0))
    # 3) bảng dinh dưỡng: tiêu đề "… trên/per 100 g|ml" rồi dòng "Natri … X mg"
    hdr = re.search(r"(?:tính\s*)?(?:trên|per|/)\s*100\s*(g|ml)\b", low)
    row = re.search(r"(?:" + names + r")\s*[:\-]?\s*" + _NUM + r"\s*" + unit + r"\b", low)
    if hdr and row:
        return _to_float(row.group(1)), f"100 {hdr.group(1)}", row.group(0)
    return None


def extract_label_nutrition(*texts):
    """Số liệu dinh dưỡng GHI TRÊN NHÃN (người dùng nêu trong câu hỏi, hoặc đọc từ bảng dinh dưỡng), quy về /100 g.

    Hiện đọc natri (mg) và đường tổng số (g). Chỉ nhận khi có cả con số lẫn cơ sở tính ("trên/per 100 g|ml" hoặc
    "khẩu phần … X g", quy đổi ×100/X); không đoán khi thiếu. Các cụm nêu NGƯỠNG ("không quá 120 mg/100 g",
    "natri thấp", "không đường") bị bỏ qua để không hiểu nhầm là số của sản phẩm.
    Trả về {"sodium_mg"?, "sugars_g"?, "basis": {...}, "source_text": {...}} hoặc None.
    """
    result, basis, source = {}, {}, {}
    for raw in texts:
        text = unicodedata.normalize("NFC", str(raw or ""))
        if not text.strip():
            continue
        low = text.casefold()
        low = re.sub(r"(?:không quá|không vượt quá|tối đa|dưới|ngưỡng|<=|≤|<)\s*" + _NUM
                     + r"\s*(?:mg|g)\s*(?:/|trên|per)\s*100\s*(?:g|ml)", " ", low)
        low = re.sub(r"(natri|sodium)\s*(thấp|low)", r"tuyên-bố-\2", low)
        low = re.sub(r"(không|ít|giảm)\s*đường", r"tuyên-bố-\1", low)
        for key, (names, unit) in _LABEL_NUTRIENTS.items():
            if key in result:
                continue
            hit = _extract_one(low, names, unit)
            if hit:
                result[key], basis[key], source[key] = hit
    if not result:
        return None
    return dict(result, basis=basis, source_text=source)


def evaluate_nutritional_risk(product_name, fndds_list, persona,
                              label_nutrition=None):
    """
    Đối soát ràng buộc dinh dưỡng định lượng với ngưỡng bệnh lý NHANES User Persona.
    Chiến lược ưu tiên nguồn dữ liệu:
      1. label_nutrition (bảng dinh dưỡng từ nhãn sản phẩm thực tế / OCR) — chính xác nhất
      2. Bảng TPTP Việt Nam (VCNDD) — ước tính cho món Việt
      3. FNDDS USDA — ước tính cho loại thực phẩm tương tự

    Nếu dùng nguồn 2 hoặc 3, đánh dấu is_estimate=True và ghi rõ disclaimer.
    """
    if not fndds_list and not _VN_FOOD_COMPOSITION_DB and not label_nutrition:
        return None, [], []

    # Ưu tiên 1: Dữ liệu từ nhãn sản phẩm thực tế
    if label_nutrition and any(v for v in label_nutrition.values() if v):
        best_item = dict(label_nutrition)
        best_item["data_source_type"] = "product_label"
        best_item["is_estimate"] = False
    else:
        vn_item = _match_vn_food(product_name)
        fndds_item, fndds_score = _match_fndds(product_name, fndds_list) if fndds_list else (None, -1)

        if vn_item and (fndds_score <= 0 or vn_item.get("source") == "VCNDD"):
            best_item = vn_item
            best_item["data_source_type"] = "vcndd_reference"
            best_item["is_estimate"] = True
            best_item["estimate_disclaimer"] = (
                "Số liệu tham chiếu từ Bảng TPTP Việt Nam cho loại thực phẩm tương tự, "
                "không phải giá trị chính xác của sản phẩm đang xét."
            )
        elif fndds_item:
            best_item = fndds_item
            best_item["data_source_type"] = "fndds_reference"
            best_item["is_estimate"] = True
            best_item["estimate_disclaimer"] = (
                "Số liệu tham chiếu từ USDA FNDDS cho loại thực phẩm tương tự, "
                "không phải giá trị chính xác của sản phẩm đang xét. "
                "Không nên dùng để kết luận định lượng chính xác về natri, đường "
                "hoặc các chất dinh dưỡng khác của sản phẩm."
            )
        elif vn_item:
            best_item = vn_item
            best_item["data_source_type"] = "vcndd_reference"
            best_item["is_estimate"] = True
            best_item["estimate_disclaimer"] = (
                "Số liệu tham chiếu từ Bảng TPTP Việt Nam, không phải giá trị chính xác."
            )
        else:
            best_item = None

    if not best_item:
        return None, [], []

    constraint_warnings, constraint_details = evaluate_nutritional_constraints(
        best_item, persona
    )

    warnings = [w["message"] for w in constraint_warnings]

    # Số natri GHI TRÊN NHÃN với người tăng huyết áp: so với điều kiện "natri thấp" của Codex CAC/GL 23-1997
    # (≤ 120 mg/100 g). Chỉ áp dụng cho số liệu của chính sản phẩm, không cho số ước tính.
    conds = {str(c).lower() for c in persona.get("conditions", [])}
    sugars = best_item.get("sugars_g")
    if not best_item.get("is_estimate") and sugars is not None and any("diabet" in c for c in conds):
        sugars = float(sugars)
        ok = sugars <= CODEX_SUGAR_FREE_G_PER_100G
        constraint_details.append({
            "condition": "diabetes", "nutrient": "Đường (điều kiện 'không đường' Codex)", "value": sugars,
            "unit": "g/100g", "upper_bound": CODEX_SUGAR_FREE_G_PER_100G, "lower_bound": None,
            "I_g": 1 if ok else 0, "severity": "OK" if ok else "CAUTION", "citation": "CODEX_CAC_GL_23_1997",
        })
        if not ok:
            warnings.append(
                f"Đường trên nhãn ≈ {sugars:g} g/100 g, vượt {CODEX_SUGAR_FREE_G_PER_100G:g} g/100 g nên không đạt "
                "điều kiện 'không đường' của Codex CAC/GL 23-1997 — người đái tháo đường nên thận trọng với lượng dùng."
            )
    sodium = best_item.get("sodium_mg")
    if not best_item.get("is_estimate") and sodium is not None and conds & {"hypertension", "cardiovascular_risk"}:
        sodium = float(sodium)
        ok = sodium <= CODEX_LOW_SODIUM_MG_PER_100G
        constraint_details.append({
            "condition": "hypertension", "nutrient": "Natri (điều kiện 'natri thấp' Codex)", "value": sodium,
            "unit": "mg/100g", "upper_bound": CODEX_LOW_SODIUM_MG_PER_100G, "lower_bound": None,
            "I_g": 1 if ok else 0, "severity": "OK" if ok else "CAUTION", "citation": "CODEX_CAC_GL_23_1997",
        })
        if not ok and not any("natri thấp" in w for w in warnings):
            warnings.append(
                f"Natri trên nhãn ≈ {sodium:.0f} mg/100 g, vượt {CODEX_LOW_SODIUM_MG_PER_100G:.0f} mg/100 g nên không đạt "
                "điều kiện 'natri thấp' của Codex CAC/GL 23-1997 — người tăng huyết áp nên thận trọng với lượng dùng."
            )

    # Thêm disclaimer vào warnings nếu là dữ liệu ước tính
    if best_item.get("is_estimate") and warnings:
        disclaimer = best_item.get("estimate_disclaimer", "")
        if disclaimer:
            warnings.insert(0, f"⚠️ LƯU Ý: {disclaimer}")

    return best_item, warnings, constraint_details


def run_perception_pipeline(image_input, sample_key="custom", force_real_ocr=False, **kwargs):
    """Tầng 1: Perception Module"""
    return run_perception(image_input, sample_key=sample_key, force_real_ocr=force_real_ocr, **kwargs)


def foodontorag_entity_linking(mention, kb_additives):
    """
    Tầng 2: Normalization Module - FoodOntoRAG 4-Agent Entity Linking Pipeline.
    Delegate sang hybrid_retriever.foodontorag_4agent_linking() triển khai đầy đủ:
      Agent 1: Hybrid Retriever (BM25 Lexical + TF-IDF/SBERT Semantic + Fusion)
      Agent 2: Selector Agent (Exact-match preference + Specificity rule)
      Agent 3: Rule match scorer (điểm khớp theo luật τ, không gọi LLM)
      Agent 4: Synonym Generator & Feedback Loop (khi τ < 0.70)
    """
    return foodontorag_4agent_linking(mention, kb_additives)


def link_all_mentions(mentions, kb_additives):
    """Liên kết mọi cụm thành phần; một cụm chứa NHIỀU mã phụ gia được tách thành một kết quả cho MỖI mã.

    Trước đây mỗi cụm chỉ giữ một thực thể (mã "cụ thể nhất"), nên cụm như "chất tạo ngọt tổng hợp (950, 955)"
    hay "chất ổn định (460(i), 407, 466)" làm mất các mã còn lại — đo trên văn bản nhãn chuẩn của bộ 750 câu:
    1.150/3.032 mã (37,9%) bị bỏ. Việc tách chỉ dùng bộ dò định danh (mã INS/E hoặc tên/tên đồng nghĩa xuất hiện
    nguyên văn trong cụm, modules.additive_linker.link_additive_mentions) — không đoán thêm mã nào.
    """
    from modules.additive_linker import find_uncertain_codes, link_additive_mentions, normalize_ins_key

    kb_by_key = {}
    for additive in kb_additives or []:
        kb_by_key.setdefault(normalize_ins_key(additive.get("ins_code", "")), additive)

    linked = []
    for mention in mentions or []:
        base = foodontorag_entity_linking(mention, kb_additives)
        raw_text = mention.get("raw_mention", "")
        # Mã OCR không có trong danh mục (vd "481(i)" đọc thành "4310") → một mục "phụ gia không liên kết được",
        # để S_retrieval giảm đúng thực tế thay vì bỏ qua mã lạ. Không đoán mã thay thế.
        for unc in find_uncertain_codes(raw_text):
            if unc["kind"] != "unknown_code":
                continue
            linked.append({
                "raw_mention": f"{raw_text} [{unc['code_text']}]",
                "category": "additive_uncertain_code",
                "matched_entity": None,
                "confidence_tau": None,
                "retrieval_confidence": 0.0,
                "match_method": "uncertain_code",
                "code_origin": "unreadable_on_label",
                "uncertain_code": unc,
                "rationale": unc["reason"],
            })
        identities = link_additive_mentions(raw_text)
        subtype_flags = {u["key"]: u for u in find_uncertain_codes(raw_text) if u["kind"] == "uncertain_subtype"}
        if len(identities) < 2:
            ins_key = normalize_ins_key((base.get("matched_entity") or {}).get("ins_code", ""))
            if ins_key in subtype_flags:
                base = dict(base, uncertain_code=subtype_flags[ins_key])
            linked.append(base)
            continue
        for k, ident in enumerate(identities, 1):
            key = normalize_ins_key(ident.get("ins_code", ""))
            rich = kb_by_key.get(key)
            entity = dict(rich or ident["matched_entity"])
            entity.setdefault("identity_only", rich is None)
            item = dict(base)
            item.update({
                "matched_entity": entity,
                "confidence_tau": ident.get("confidence"),
                "retrieval_confidence": ident.get("confidence"),
                "match_method": ident.get("match_method"),
                "code_origin": ident.get("code_origin"),
                "identity_source": ident.get("source", {}),
                "multi_code_index": f"{k}/{len(identities)}",
                **({"uncertain_code": subtype_flags[key]} if key in subtype_flags else {}),
                "rationale": (
                    f"Cụm chứa {len(identities)} mã phụ gia; liên kết định danh {ident.get('match_method')}: "
                    f"{ident.get('matched_alias') or ''} → {entity.get('ins_code', '')} (mã {k}/{len(identities)})."
                ),
            })
            linked.append(item)
    return linked


_ADDITIVE_ENTITY_CATEGORIES = {
    "additive_uncertain_code",
    "flavor_enhancer",
    "preservative",
    "color",
    "sweetener",
    "acidity_regulator",
    "emulsifier",
    "raising_agent",
    "thickener",
    "stabilizer",
    "antioxidant",
    "additive",
}


def _is_additive_link_candidate(item):
    return bool(item.get("matched_entity")) or item.get("category") in _ADDITIVE_ENTITY_CATEGORIES


# Ngưỡng từ chối theo điểm tin cậy tổng: S_total < ngưỡng (và không có cảnh báo nào) → ABSTAIN.
# Giá trị phải được chọn trên phần "phát triển" của bộ đánh giá: scripts_research/calibrate_confidence.py.
CONFIDENCE_ABSTAIN_THRESHOLD = 0.65
# Số liệu dinh dưỡng ƯỚC TÍNH (thực phẩm tương tự trong USDA FNDDS / VCNDD) có được đổi kết luận an toàn không.
# Mặc định False: chỉ bảng dinh dưỡng của chính sản phẩm mới được. True chỉ để tái lập hành vi cũ khi đối chiếu.
NUTRITION_ESTIMATE_CHANGES_LABEL = False
def _evaluate_safety_core(linked_results, persona, question, avg_ocr_conf=0.95, all_mentions=None):
    """
    Tầng 4: Đánh giá an toàn, tính điểm tin cậy kép và kích hoạt Guardrail an toàn y tế.
    Tích hợp Negation Constraint Modeling: gán nhãn PERMITTED/RESTRICTED/CONTRAINDICATED/ALLERGEN_RISK.
    all_mentions: toàn bộ nguyên liệu OCR (bao gồm nguyên liệu thô không phải INS).
    """
    linked_results = apply_negation_constraints(linked_results, persona)

    s_perception = max(0.0, min(float(avg_ocr_conf), 1.0))
    additive_link_results = [
        item for item in linked_results if _is_additive_link_candidate(item)
    ]
    # S_retrieval = tỉ lệ cụm phụ gia trên nhãn liên kết được với cơ sở tri thức. Mọi liên kết được giữ lại đều đã
    # khớp định danh (mã INS/E, tên chuẩn hoặc tên đồng nghĩa có trong chữ trên nhãn) — xem foodontorag_4agent_linking.
    # Trước đây dùng điểm BM25/ngữ nghĩa chia cho điểm cao nhất của lượt tìm (điểm tương đối, ứng viên đầu luôn ≈1)
    # và gán 1.0 cho nhãn không có phụ gia. Nhãn không có cụm phụ gia → không áp dụng (None), S_total = S_perception.
    linked_additives = [item for item in additive_link_results if item.get("matched_entity")]
    if additive_link_results:
        s_retrieval = len(linked_additives) / len(additive_link_results)
        s_retrieval_basis = f"{len(linked_additives)}/{len(additive_link_results)} cụm phụ gia liên kết được với cơ sở tri thức"
        s_total = 0.5 * s_perception + 0.5 * s_retrieval
    else:
        s_retrieval = None
        s_retrieval_basis = "không áp dụng — nhãn không có cụm phụ gia cần liên kết"
        s_total = s_perception
    insufficient_evidence = s_total < CONFIDENCE_ABSTAIN_THRESHOLD
    confidence_metrics = {
        "s_perception": s_perception,
        "s_retrieval": s_retrieval,
        "s_retrieval_basis": s_retrieval_basis,
        "additive_mentions_total": len(additive_link_results),
        "additive_mentions_linked": len(linked_additives),
        "s_total": s_total,
    }

    critical_alerts = []
    warning_alerts = []
    caution_alerts = []
    matched_benefits = []

    user_conditions = persona.get("conditions", [])
    user_allergies = persona.get("allergies", [])

    for item in linked_results:
        entity = item["matched_entity"]
        if not entity:
            continue

        for contra in entity.get("contraindications", []):
            if contra.get("verification_status") != "source_checked":
                continue
            cond = contra.get("condition", "").lower()
            sev = contra.get("severity", "WARNING")
            msg = contra.get("message", "")

            if any(cond in uc.lower() or uc.lower() in cond for uc in user_conditions) or any(a in cond for a in user_allergies):
                alert_item = {
                    "additive": f"{entity['canonical_name']} ({entity.get('ins_code', '')})",
                    "message": msg,
                    "regulation": entity.get("vn_regulation") or "VN_TT24_2019",
                    "codex": entity.get("codex_standard") or "CODEX_STAN_192_1995",
                }
                if sev == "CRITICAL":
                    critical_alerts.append(alert_item)
                elif sev == "WARNING":
                    warning_alerts.append(alert_item)
                elif sev == "CAUTION":
                    caution_alerts.append(alert_item)
                elif sev == "MATCH":
                    matched_benefits.append({
                        "additive": f"{entity['canonical_name']} ({entity.get('ins_code', '')})",
                        "message": msg
                    })

    # Negation Constraint Modeling: phân loại thành phần theo constraint_type
    for item in linked_results:
        ctype = item.get("constraint_type")
        crule = item.get("constraint_rule")
        if not ctype or ctype == "PERMITTED":
            continue
        entity = item.get("matched_entity")
        entity_name = entity["canonical_name"] if entity else item["raw_mention"]
        ins_code = entity.get("ins_code", "") if entity else ""
        label = f"{entity_name} ({ins_code})" if ins_code else entity_name
        rule_desc = f" (Quy tắc: {crule[0]} -> {crule[1]})" if crule else ""

        reg_default = "US_FDA_FALCPA" if ctype in (CONSTRAINT_ALLERGEN_RISK, CONSTRAINT_PRECAUTIONARY_RISK) else (entity.get("vn_regulation") if entity else "VN_TT24_2019")
        codex_default = "CODEX_CXS_1_1985" if ctype in (CONSTRAINT_ALLERGEN_RISK, CONSTRAINT_PRECAUTIONARY_RISK) else (entity.get("codex_standard") if entity else "CODEX_STAN_192_1995")
        constraint_alert = {
            "additive": label,
            "message": (
                f"Nhãn/OCR có thể khớp với thuật ngữ dị nguyên '{item['raw_mention']}'. "
                "Cần xác minh thành phần và nhãn đầy đủ; đây không phải chẩn đoán hay "
                "kết luận về phản ứng cá nhân."
                if ctype == CONSTRAINT_ALLERGEN_RISK
                else f"Quy tắc sàng lọc [{ctype}] khớp với '{item['raw_mention']}'. "
                "Cần xác minh nguồn, phạm vi và ngữ cảnh trước khi diễn giải."
            ),
            "regulation": reg_default,
            "codex": codex_default,
        }
        if ctype == CONSTRAINT_CONTRAINDICATED:
            critical_alerts.append(constraint_alert)
        elif ctype == CONSTRAINT_ALLERGEN_RISK:
            warning_alerts.append(constraint_alert)
        elif ctype == CONSTRAINT_PRECAUTIONARY_RISK:
            pal_alert = dict(constraint_alert)
            pal_alert["message"] = (
                f"Cảnh báo tiếp xúc chéo (Precautionary Allergen Labeling - PAL): "
                f"Nhãn ghi nhận nguy cơ nhiễm chéo hoặc có thể chứa vết của dị nguyên '{item['raw_mention']}'. "
                "Theo khuyến nghị FAO/WHO JEMRA Part 3 (2023), đối tượng có tiền sử dị ứng thực phẩm nghiêm trọng cần thận trọng."
            )
            pal_alert["regulation"] = "FAO_WHO_JEMRA_PART_3"
            pal_alert["codex"] = "CODEX_CXS_1_1985"
            warning_alerts.append(pal_alert)
        elif ctype == CONSTRAINT_RESTRICTED:
            caution_alerts.append(constraint_alert)

    # Dị ứng trực tiếp — scan TOÀN BỘ nguyên liệu (không chỉ phụ gia INS)
    if user_allergies:
        # Gộp nguồn: linked_results + all_mentions (nguyên liệu thô)
        all_texts_to_scan = []
        seen_texts = set()
        for item in linked_results:
            txt = item["raw_mention"]
            if txt.lower() not in seen_texts:
                all_texts_to_scan.append(txt)
                seen_texts.add(txt.lower())
        if all_mentions:
            for m in all_mentions:
                txt = m.get("raw_mention", m.get("text", ""))
                if txt and txt.lower() not in seen_texts:
                    all_texts_to_scan.append(txt)
                    seen_texts.add(txt.lower())

        pal_indicators = ["có thể chứa", "may contain", "vết của", "sản xuất trên", "chế biến trên"]
        has_any_pal_on_label = any(any(p in t.lower() for p in pal_indicators) for t in all_texts_to_scan)

        for raw_text in all_texts_to_scan:
            raw_text_lower = raw_text.lower()
            # Miễn trừ dầu tinh luyện hoàn toàn theo US FDA FALCPA Sec 203
            if is_exempt_refined_oil(raw_text_lower):
                continue

            is_prec_text = any(p in raw_text_lower for p in pal_indicators)
            for allergy in user_allergies:
                keywords = list(ALLERGEN_KEYWORDS.get(allergy, [allergy]))
                derivatives = DERIVATIVE_ALLERGEN_MAP.get(allergy, [])
                combined_terms = keywords + [d for d in derivatives if d not in keywords]

                precaution_hit = (not any(contains_label_term(strip_allergen_noise(raw_text_lower, allergy), kw) for kw in combined_terms)
                                  and any(contains_label_term(raw_text_lower, kw) for kw in PRECAUTION_KEYWORDS.get(allergy, [])))
                if precaution_hit:
                    is_prec_text = True
                if precaution_hit or any(contains_label_term(strip_allergen_noise(raw_text_lower, allergy), kw) for kw in combined_terms):
                    reg_source = "FAO_WHO_JEMRA_PART_3" if is_prec_text else "US_FDA_FALCPA"
                    allergen_alert = {
                        "additive": f"Dị nguyên: {raw_text}",
                        "message": (
                            f"Cảnh báo nguy cơ tiếp xúc chéo (PAL): '{raw_text}' ({allergy}). Theo FAO/WHO JEMRA Part 3, đối tượng dị ứng nặng cần thận trọng."
                            if is_prec_text else
                            f"Nhãn/OCR có thể khớp với dị nguyên '{raw_text}' ({allergy}). "
                            "Kết quả này không xác nhận nguồn, lượng, nhiễm chéo hay phản ứng cá nhân; "
                            "hãy kiểm tra nhãn đầy đủ và hỏi chuyên gia y tế nếu cần."
                        ),
                        "regulation": reg_source,
                        "codex": "CODEX_CXS_1_1985",
                        "allergen_key": allergy,
                    }
                    tier_info = allergen_priority_info(allergy)
                    if tier_info:
                        allergen_alert.update(tier_info)
                        allergen_alert["message"] += f" ({tier_info['priority_text']}.)"
                    warning_alerts.append(allergen_alert)
                    break

    # Phân biệt không dung nạp lactose (enzyme deficiency) vs Dị ứng protein sữa (IgE)
    # Theo FAO/WHO JEMRA 2021: Lactose không thuộc danh mục dị nguyên ưu tiên miễn dịch
    user_cond_lower = [c.lower() for c in user_conditions]
    has_lactose_intolerance = any("lactose" in c for c in user_cond_lower)
    has_milk_allergy = any("milk" in a or "sua" in a for a in user_allergies)
    if has_lactose_intolerance and not has_milk_allergy:
        milk_indicators = ["lactose", "sữa", "sua", "whey", "casein", "bột sữa", "bot sua"]
        detected_milk = any(
            any(kw in item["raw_mention"].lower() for kw in milk_indicators)
            for item in linked_results
        )
        if detected_milk:
            lactose_caution = {
                "additive": "Lactose / Sữa",
                "message": (
                    "Lưu ý không dung nạp đường sữa: Sản phẩm có chứa thành phần từ sữa/lactose. "
                    "Theo FAO/WHO JEMRA 2021, đây là tình trạng thiếu hụt enzyme lactase đường tiêu hóa, "
                    "không phải dị ứng đạm sữa qua trung gian IgE và không gây sốc phản vệ."
                ),
                "regulation": "WHO_JEMRA_PRIORITY_ALLERGENS",
                "codex": "CODEX_CXS_1_1985"
            }
            caution_alerts.append(lactose_caution)

    # Đánh giá quy chuẩn pháp lý tổng hợp: ADI (JECFA) + EFSA + TT 24/2019/TT-BYT
    body_weight = persona.get("weight_kg", 60.0)
    regulatory_assessments = []
    efsa_warnings = []
    for item in linked_results:
        entity = item.get("matched_entity")
        if not entity or entity.get("ins_code", "").startswith("FOODON"):
            continue
        reg = evaluate_regulatory_compliance(entity, persona, body_weight_kg=body_weight)
        regulatory_assessments.append(reg)
        if reg.get("efsa"):
            efsa_warnings.append(reg["efsa"])

    # Trẻ dưới 12 tháng: phụ gia trên nhãn phải nằm trong danh mục cho thực phẩm trẻ nhỏ (TT24 PL2A / Codex GSFA 13.1, 13.2)
    # và nhãn ghi tuổi tối thiểu lớn hơn tuổi của bé -> làm theo nhãn (mức cảnh báo như dị nguyên trực tiếp)
    if infant_rules.is_infant(persona, question):
        caution_alerts.extend(infant_rules.evaluate(linked_results)["alerts"])
        warning_alerts.extend(infant_rules.label_age_alerts(persona, question, _label_texts(linked_results, all_mentions)))

    # Kiểm tra thiếu dữ liệu cho cơ chế từ chối 2 loại
    unlinked_count = sum(
        1 for item in additive_link_results if not item.get("matched_entity")
    )
    total_count = len(additive_link_results)
    kb_coverage_low = total_count > 0 and (unlinked_count / total_count) > 0.5
    has_insufficient_adi = any(
        reg.get("adi", {}).get("compliance_status")
        in {"INSUFFICIENT_DATA", "SOURCE_REVIEW_REQUIRED"}
        for reg in regulatory_assessments
    )

    abstention_reasons = []
    if insufficient_evidence:
        abstention_reasons.append(
            f"Độ tin cậy OCR hoặc độ phù hợp truy xuất dưới ngưỡng an toàn (S_total < {CONFIDENCE_ABSTAIN_THRESHOLD:g})."
        )
    if kb_coverage_low:
        abstention_reasons.append(
            f"Hệ thống không tìm thấy đủ thực thể trong KB ({unlinked_count}/{total_count} "
            "khai báo phụ gia không khớp). Cần tên hoặc mã phụ gia cụ thể hơn."
        )

    q_fold = str(question or "").casefold()
    asks_permission = any(k in q_fold for k in ("được phép", "danh mục", "theo thông tư", "có trong"))
    dose_kw = ("vượt", "vuot", "adi", "tích lũy", "liều", "hàm lượng", "mg/kg")  # a dose question, even if it says "cho phép"
    quantity_kw = dose_kw + ("mỗi ngày", "hằng ngày", "hàng ngày", "thường xuyên")
    effect_kw = ("an toàn", "an toan", "gây hại", "có hại", "ảnh hưởng", "lo ngại")
    generic_user = not set(c.lower() for c in persona.get("conditions", [])) - {"healthy", "healthy_adult"} and not persona.get("allergies")
    # a permission-list question is answered from the list; a question tied to the user's own condition is answered
    # by that condition's rules; only generic exposure / health-effect questions need a dose that the label lacks
    exposure_question = any(k in q_fold for k in dose_kw) or (not asks_permission) and (
        any(k in q_fold for k in quantity_kw) or (generic_user and any(k in q_fold for k in effect_kw)))
    if has_insufficient_adi and exposure_question:
        abstention_reasons.append(
            "Đã định danh được phụ gia nhưng thiếu hàm lượng, khẩu phần hoặc nhóm thực phẩm "
            "để đối chiếu ML/ADI; cần bổ sung dữ liệu định lượng thay vì mã định danh."
        )
    def _collect_canonical_citations(alert_list, include_codex=True):
        cits = []
        for a in alert_list:
            r = canonicalize_citation(a.get("regulation", ""))
            if r and is_recognized_citation(r) and r not in cits:
                cits.append(r)
            if include_codex:
                c = canonicalize_citation(a.get("codex", ""))
                if c and is_recognized_citation(c) and c not in cits:
                    cits.append(c)
        return cits

    if critical_alerts:
        return {
            "status": "CRITICAL_ALERT",
            "safety_label": "CRITICAL_ALERT",
            "badge_color": "red",
            "title": "🔴 CẦN XÁC MINH VỚI CHUYÊN GIA",
            "alerts": critical_alerts,
            "cautions": caution_alerts,
            "advice": (
                "Có quy tắc mức cao khớp với dữ liệu nhãn. Hệ thống không thể xác nhận "
                "nguy cơ cá nhân; hãy kiểm tra nhãn và trao đổi với chuyên gia y tế trước "
                "khi quyết định sử dụng."
            ),
            "citations": (_collect_canonical_citations(critical_alerts) + (["CODEX_CXS_1_1985"] if user_allergies and "CODEX_CXS_1_1985" not in _collect_canonical_citations(critical_alerts) else [])),
            **confidence_metrics,
            "regulatory": regulatory_assessments,
            "efsa_warnings": efsa_warnings,
        }
    elif warning_alerts:
        warn_cits = _collect_canonical_citations(warning_alerts)
        if user_allergies:
            pal_on_label = locals().get("has_any_pal_on_label", False)
            if pal_on_label:
                if "FAO_WHO_JEMRA_PART_3" not in warn_cits:
                    warn_cits.append("FAO_WHO_JEMRA_PART_3")
                if "US_FDA_FALCPA" not in warn_cits:
                    warn_cits.append("US_FDA_FALCPA")
            else:
                if "US_FDA_FALCPA" not in warn_cits:
                    warn_cits.append("US_FDA_FALCPA")
                if "WHO_JEMRA_PRIORITY_ALLERGENS" not in warn_cits:
                    warn_cits.append("WHO_JEMRA_PRIORITY_ALLERGENS")
            if "CODEX_CXS_1_1985" not in warn_cits:
                warn_cits.append("CODEX_CXS_1_1985")
        return {
            "status": "WARNING",
            "safety_label": "UNSAFE",
            "badge_color": "orange",
            "title": "🟡 CẢNH BÁO — CÓ THUẬT NGỮ CẦN KIỂM TRA",
            "alerts": warning_alerts,
            "cautions": caution_alerts,
            "advice": (
                "OCR có thể khớp một thuật ngữ cần kiểm tra. Kết quả này không xác nhận "
                "thành phần, dị ứng cá nhân hay mức nguy hiểm; hãy kiểm tra nhãn đầy đủ "
                "và hỏi chuyên gia y tế nếu cần."
            ),
            "citations": warn_cits,
            **confidence_metrics,
            "regulatory": regulatory_assessments,
            "efsa_warnings": efsa_warnings,
        }
    elif caution_alerts:
        caut_cits = _collect_canonical_citations(caution_alerts, include_codex=True)
        if efsa_warnings:
            has_e171 = any("171" in str(ew.get("substance", "")) or "titanium" in str(ew.get("substance", "")).lower() for ew in efsa_warnings)
            target_efsa = "EFSA_2021_TITANIUM_DIOXIDE_E171" if has_e171 else "EFSA_OPINIONS"
            if target_efsa not in caut_cits:
                caut_cits.append(target_efsa)
        if any(reg.get("adi", {}).get("source") == "JECFA" for reg in regulatory_assessments):
            if "JECFA_EVALUATIONS" not in caut_cits:
                caut_cits.append("JECFA_EVALUATIONS")
        return {
            "status": "CAUTION",
            "safety_label": "CAUTION",
            "badge_color": "yellow",
            "title": "🟡 LƯU Ý — CẦN THÊM THÔNG TIN VỀ LIỀU LƯỢNG",
            "alerts": caution_alerts,
            "cautions": [],
            "advice": (
                "Có thông tin cần lưu ý nhưng dữ liệu nhãn không đủ để xác định mức phơi nhiễm "
                "hoặc đưa ra khuyến nghị cá nhân."
            ),
            "citations": caut_cits,
            **confidence_metrics,
            "regulatory": regulatory_assessments,
            "efsa_warnings": efsa_warnings,
        }
    else:
        if abstention_reasons:
            abstain_cits = []
            q_lower = str(question or "").lower()
            is_additive_query = any(
                k in q_lower for k in ["phụ gia", "adi", "liều lượng", "hàm lượng", "ins", "e-", "ngưỡng", "tích lũy"]
            ) or any("phụ gia" in str(r).lower() or "adi" in str(r).lower() or "định lượng" in str(r).lower() for r in abstention_reasons)
            if is_additive_query and (any(item.get("matched_entity") and item["matched_entity"].get("ins_code") for item in linked_results) or regulatory_assessments):
                for c in ["VN_TT24_2019", "CODEX_STAN_192_1995"]:
                    if c not in abstain_cits:
                        abstain_cits.append(c)
                if any(reg.get("adi", {}).get("source") == "JECFA" for reg in regulatory_assessments):
                    if "JECFA_EVALUATIONS" not in abstain_cits:
                        abstain_cits.append("JECFA_EVALUATIONS")

            is_allergen_query = bool(user_allergies) or any(
                "dị nguyên" in str(r).lower() or "dị ứng" in str(r).lower() for r in abstention_reasons
            ) or any(
                k in q_lower for k in ["dị ứng", "dị nguyên", "nhiễm chéo"]
            )
            if is_allergen_query:
                for c in ["US_FDA_FALCPA", "FAO_WHO_JEMRA_PART_3", "CODEX_CXS_1_1985"]:
                    if c not in abstain_cits:
                        abstain_cits.append(c)
            return {
                "status": "ABSTAIN",
                "safety_label": "ABSTAIN",
                "badge_color": "gray",
                "title": "⚪ TỪ CHỐI TƯ VẤN — CHƯA ĐỦ DỮ LIỆU",
                "abstention_type": "insufficient_data",
                "abstention_reasons": abstention_reasons,
                "message": (
                    "Hệ thống chưa đủ dữ liệu để đưa ra khuyến nghị đáng tin cậy. "
                    + " ".join(abstention_reasons)
                    + " Đề nghị chụp lại nhãn rõ hơn hoặc bổ sung thông tin còn thiếu "
                    "(hàm lượng/khẩu phần, thông tin dị nguyên, bảng dinh dưỡng)."
                ),
                "alerts": [],
                "cautions": [],
                "citations": abstain_cits,
                **confidence_metrics,
                "regulatory": regulatory_assessments,
                "efsa_warnings": efsa_warnings,
            }

        insufficient_data_notes = []
        if has_insufficient_adi:
            insufficient_data_notes.append(
                "Một số phụ gia chưa đủ dữ liệu định lượng để đánh giá mức phơi nhiễm ADI."
            )

        safe_citations = []
        if user_allergies:
            safe_citations.extend(["US_FDA_FALCPA", "WHO_JEMRA_PRIORITY_ALLERGENS", "CODEX_CXS_1_1985"])
        if any(item.get("matched_entity") and item["matched_entity"].get("ins_code") for item in linked_results):
            for c in ["CODEX_STAN_192_1995", "VN_TT24_2019"]:
                if c not in safe_citations:
                    safe_citations.append(c)
        return {
            "status": "SCREENED_NO_ALERT",
            "safety_label": "SAFE",
            "badge_color": "blue",
            "title": "⚪ CHƯA PHÁT HIỆN CẢNH BÁO TRONG DỮ LIỆU ĐÃ SÀNG LỌC",
            "alerts": [],
            "cautions": [],
            "benefits": matched_benefits,
            "insufficient_data_notes": insufficient_data_notes,
            "advice": (
                "Không tìm thấy cảnh báo phù hợp trong các thành phần đã nhận diện. Điều này "
                "không chứng minh sản phẩm an toàn, không xác nhận toàn bộ nhãn và không phải "
                "kết luận tuân thủ pháp luật; hãy kiểm tra bao bì đầy đủ và tham khảo "
                "ý kiến chuyên gia nếu có bệnh nền."
            ),
            "citations": safe_citations,
            **confidence_metrics,
            "regulatory": regulatory_assessments,
            "efsa_warnings": efsa_warnings,
        }


def _label_texts(linked_results, all_mentions=None):
    """Các đoạn chữ của nhãn (mention thô) — dùng để đọc câu cảnh báo như giới hạn tuổi."""
    texts = [str(m.get("raw_mention", "")) if isinstance(m, dict) else str(m) for m in (all_mentions or [])]
    texts += [str(it.get("raw_mention", "")) for it in (linked_results or [])]
    return list(dict.fromkeys(t for t in texts if t))


# Từ dị nguyên trong câu hỏi chỉ được hiểu là "người dùng dị ứng" khi đứng ngay sau một dấu hiệu dị ứng/hỏi thành phần,
# trong cùng cụm (tới động từ hoặc dấu câu kế tiếp). Trước đây mọi từ đều được nhận: "bé ăn sữa chua 'X'" -> dị ứng sữa,
# "điều kiện 'natri thấp'" -> "điều" = hạt điều.
_ALLERGY_CUE = re.compile(r"dị\s*ứng|dị\s*nguyên|mẫn\s*cảm|không\s*dung\s*nạp|tránh|kiêng|allerg|nhiễm\s*chéo|chứa")
_ALLERGY_CUE_END = re.compile(r"\s(có|ăn|uống|dùng|được|thì|nên|không)\b|[?.!]")


def _question_allergies(question):
    """Allergen groups the user names in the question (after an allergy cue); the quoted product name is ignored."""
    q = re.sub(r"'.*?'(?=\s|$|[?,.;:])", " ", str(question or "")).lower()
    windows = []
    for m in _ALLERGY_CUE.finditer(q):
        rest = q[m.end():]
        stop = _ALLERGY_CUE_END.search(rest)
        windows.append(rest[: stop.start() if stop else len(rest)])
    text = " | ".join(windows)
    return [a for a, kws in ALLERGEN_KEYWORDS.items() if any(contains_label_term(text, kw) for kw in kws)]


def evaluate_safety_and_guardrail(linked_results, persona, question, avg_ocr_conf=0.95, all_mentions=None,
                                  extra_gaps=None):
    """Guardrail with two-type abstention (see modules/abstention.py).

    1. Missing data (insufficient_data): the label lacks what the question needs -> ask to re-photograph the
       label or supply the missing information. Independent of OCR confidence, so Config B can abstain too.
    2. Out of scope (out_of_scope): the question asks for diagnosis / treatment / medication decisions
       -> refer to a doctor, pharmacist or dietitian.
    A CRITICAL_ALERT is never downgraded; it only gets the data gaps / referral attached. A direct allergen hit is
    kept as a warning (the label already shows the allergen), with the gaps attached.
    """
    stated = _question_allergies(question)
    if stated:  # allergens the user states in the question are added to the profile, never replacing it
        persona = {**persona, "allergies": list(dict.fromkeys(list(persona.get("allergies", [])) + stated))}
    result = _evaluate_safety_core(linked_results, persona, question, avg_ocr_conf=avg_ocr_conf, all_mentions=all_mentions)
    scope = abstention.detect_out_of_scope(question)
    stated_terms = [kw for a in stated for kw in ALLERGEN_KEYWORDS.get(a, [])]
    gaps = abstention.detect_data_gaps(question, linked_results, all_mentions, allergen_terms=stated_terms)
    gaps = gaps + list(extra_gaps or [])  # khoảng trống tri thức do tầng QA động phát hiện (dynamic_qa)
    status = result.get("status")
    types = list(dict.fromkeys(
        ([abstention.INSUFFICIENT_DATA] if gaps or (status == "ABSTAIN") else [])
        + ([abstention.OUT_OF_SCOPE] if scope else [])))
    result["data_gaps"] = gaps
    result["scope_issues"] = scope
    # nhóm người dùng KB không có luật riêng (người cao tuổi, cho con bú) và phạm vi của luật trẻ dưới 12 tháng:
    # chỉ là dòng lưu ý, không đổi kết luận
    result["population_notes"] = abstention.population_notes(
        question, persona, linked_results=linked_results, covered_conditions=VERIFIED_ADDITIVE_CONDITIONS)
    if infant_rules.is_infant(persona, question):
        result["infant_check"] = infant_rules.evaluate(linked_results)
        result["infant_check"]["label_age"] = [a for a in result.get("alerts", []) if a.get("label_age_rule")]
        cits = result.setdefault("citations", [])
        extra_cits = (["VN_TT24_2019", "CODEX_STAN_192_1995"] if result["infant_check"]["checked"] else []) \
            + (["CODEX_CXS_72_1981", "VN_QCVN_11_1_2012"] if result["infant_check"]["nutrients"] else []) \
            + (["CODEX_CXS_74_1981"] if result["infant_check"]["label_age"] else [])
        for c in extra_cits:
            if c not in cits:
                cits.append(c)
    if not types:
        return result

    # a direct allergen hit keeps its warning; when the question names the allergen, only a hit on that one counts
    # a label age statement ("không dùng cho trẻ dưới 1 tuổi") is read straight from the label, so it is kept the same way
    direct_allergen = any(
        (a.get("regulation") == "US_FDA_FALCPA" and (not stated or a.get("allergen_key") in stated))
        or a.get("label_age_rule")
        for a in result.get("alerts", []))
    only_data_gap = types == [abstention.INSUFFICIENT_DATA]
    keep_alert = status == "CRITICAL_ALERT" or (only_data_gap and direct_allergen and status != "ABSTAIN")
    if keep_alert:
        extra = abstention.describe(types, gaps, scope)
        result["abstention_types"] = types
        result["advice"] = (result.get("advice", "") + " " + extra["message"]).strip()
        return result

    core_reasons = result.get("abstention_reasons", []) if status == "ABSTAIN" else []
    desc = abstention.describe(types, gaps, scope, core_reasons)
    other = [a["additive"] for a in result.get("alerts", []) if a.get("additive")][:3]
    if other:
        desc["message"] += " Lưu ý: nhãn còn khớp mục khác trong hồ sơ của bạn (" + "; ".join(other) + ") — hãy tự kiểm tra nhãn."

    # Tách bạch nguồn trích dẫn đúng theo từng miền vấn đề (domain-aware citations), tránh gán bừa bãi
    all_reasons_text = " ".join(core_reasons + [g.get("reason", "") for g in gaps] + scope).lower()
    q_text = str(question or "").lower()
    is_additive_gap = any("ADDITIVE" in g.get("code", "") for g in gaps) or any(
        k in q_text for k in ["phụ gia", "adi", "liều lượng", "hàm lượng", "ins", "e-", "ngưỡng", "tích lũy"]
    ) or any(k in all_reasons_text for k in ["phụ gia", "adi", "định lượng"])
    is_allergen_gap = any(g.get("code") == "CROSS_CONTACT_STATEMENT" for g in gaps) or bool(stated) or any(
        k in q_text for k in ["dị ứng", "dị nguyên", "nhiễm chéo"]
    ) or any(k in all_reasons_text for k in ["dị nguyên", "dị ứng", "nhiễm chéo"])

    guard_cits = []
    if is_allergen_gap:
        for c in ["US_FDA_FALCPA", "WHO_JEMRA_PRIORITY_ALLERGENS", "CODEX_CXS_1_1985", "FAO_WHO_JEMRA_PART_3"]:
            if c not in guard_cits:
                guard_cits.append(c)
    if is_additive_gap:
        for c in ["VN_TT24_2019", "CODEX_STAN_192_1995"]:
            if c not in guard_cits:
                guard_cits.append(c)
        if any(reg.get("adi", {}).get("source") == "JECFA" for reg in result.get("regulatory", [])):
            if "JECFA_EVALUATIONS" not in guard_cits:
                guard_cits.append("JECFA_EVALUATIONS")

    final_cits = guard_cits if guard_cits else result.get("citations", [])

    return {
        **result,
        "status": "ABSTAIN",
        "safety_label": "ABSTAIN",
        "badge_color": "gray",
        "title": desc["title"],
        "abstention_type": types[0] if len(types) == 1 else "+".join(types),
        "abstention_types": types,
        "abstention_reasons": core_reasons + [g["reason"] for g in gaps] + scope,
        "message": desc["message"],
        "citations": final_cits,
        "suppressed_status": status if status != "ABSTAIN" else None,
    }


# Độ dài câu trả lời — DÙNG CHUNG cho cả Cấu hình A (_LLM_ONLY_PROMPT) và B/C (_LLM_ANSWER_PROMPT) để phép so sánh
# trong bộ QA không lệch vì một bên được dặn viết ngắn hơn.
ANSWER_LENGTH_RULE = ("Trình bày ngắn gọn: 1 câu kết luận trả lời thẳng câu hỏi, sau đó tối đa 4 gạch đầu dòng "
                      "lý do/khuyến nghị; tổng khoảng 120–150 từ; không chép lại danh sách thành phần.")

_LLM_ANSWER_PROMPT = """Bạn là trợ lý tra cứu an toàn thực phẩm Việt Nam. Dựa trên bằng chứng khoa học bên dưới, hãy trả lời câu hỏi của người dùng bằng tiếng Việt.

## Yêu cầu bắt buộc:
- Hệ thống chỉ cung cấp thông tin tham khảo và gợi ý, KHÔNG thay thế tư vấn y khoa chuyên nghiệp.
- Sử dụng ngôn ngữ khuyến nghị: "nên cân nhắc", "khuyến nghị tham khảo ý kiến bác sĩ", "có thể cần thận trọng" — KHÔNG dùng giọng kết luận tuyệt đối như "cấm dùng", "không được ăn", "chắc chắn an toàn".
- Trích dẫn quy chuẩn: Mỗi nhận định rủi ro hoặc an toàn PHẢI kèm mã trích dẫn trong ngoặc vuông từ danh sách nguồn được cung cấp (ví dụ: [US_FDA_FALCPA], [FAO_WHO_JEMRA_PART_3], [VN_TT24_2019]). KHÔNG thêm chữ 'Nguồn:' hoặc link bên trong ngoặc vuông (viết [MÃ_NGUỒN] thay vì [Nguồn: MÃ_NGUỒN]).
- Chỉ dùng chính xác mã nguồn trong danh sách; không tự tạo nguồn, năm xuất bản, URL hoặc mã trích dẫn ngoài danh sách.
- Không bịa thêm thông tin ngoài bằng chứng được cung cấp.
- Nếu bằng chứng không đủ để trả lời:
  + Thiếu dữ liệu nhãn/thành phần: đề nghị chụp lại nhãn rõ hơn hoặc bổ sung thông tin.
  + Vượt phạm vi hệ thống: đề nghị trao đổi với chuyên gia dinh dưỡng hoặc bác sĩ phù hợp.
- Nếu dữ liệu dinh dưỡng là ước tính từ FNDDS (không phải từ nhãn sản phẩm), phải nêu rõ đây là số liệu tham chiếu cho loại thực phẩm tương tự, không phải giá trị chính xác của sản phẩm.
- Không lặp lại nguyên văn bằng chứng, hãy tổng hợp thành câu trả lời mạch lạc.
- {length_rule}
- Tên thành phần đọc từ ảnh (OCR) có thể còn sai chính tả hoặc lẫn chữ của phần khác trên bao bì: khi nhắc tới một thành phần, dùng tên đã chuẩn hóa (phần sau dấu "→" trong bằng chứng) nếu có; không đoán thêm thành phần không có trong bằng chứng.
- Dòng cuối cùng bắt buộc ghi nhãn an toàn chuẩn hóa: SAFETY_LABEL: {status} (chọn đúng 1 trong 5 nhãn sau: SAFE, CAUTION, UNSAFE, CRITICAL_ALERT, ABSTAIN; KHÔNG dùng nhãn WARNING).

## Hồ sơ người dùng:
{persona_info}

## Câu hỏi:
{question}

## Trạng thái đánh giá quy chuẩn: {status}

## Bằng chứng thu thập được:
{evidence}

## Câu trả lời (mang tính gợi ý, khuyến nghị — kèm mã trích dẫn [MÃ_NGUỒN]):
"""


def _get_gemini_api_key():
    """Lấy API key cho Gemini: GEMINI_API_KEY/GOOGLE_API_KEY, hoặc khóa đầu tiên trong danh sách GEMINI_API_KEYS / gemini_keys.txt."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        keys = llm_keys.load_keys()  # tự nạp .env
        api_key = keys[0] if keys else None
    return api_key


PREFERRED_GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
]


_LAST_LLM_MODEL = None


def get_last_llm_model():
    """Tên model đã thực sự trả lời ở lần gọi LLM gần nhất (để ghi vào file dự đoán)."""
    return _LAST_LLM_MODEL


def _call_gemini_with_fallback(prompt, temperature=0.3, max_output_tokens=4096, api_key=None):
    """
    Gọi Gemini với chuỗi model dự phòng và danh sách khóa (modules.llm_keys).

    Dùng lần lượt từng khóa; khi mọi model của khóa hiện tại đều báo hết hạn mức ngày thì chuyển khóa kế.
    Hết toàn bộ khóa: ở chế độ strict ném LLMQuotaExhausted (để dừng batch, mai chạy tiếp với --resume),
    ngoài chế độ strict trả None như trước.
    """
    global _LAST_LLM_MODEL
    pool = llm_keys.get_pool(extra_key=api_key)
    if not pool.keys:
        return None

    try:
        from google import genai
    except ImportError:
        _LOGGER.warning("Thư viện google-genai chưa được cài đặt.")
        return None

    models_to_try = []
    custom_model = os.environ.get("GEMINI_MODEL")
    if custom_model:
        models_to_try.append(custom_model.strip())
    for m in PREFERRED_GEMINI_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    last_error = None
    while True:
        key = pool.acquire()
        if key is None:
            break
        try:
            client = genai.Client(api_key=key)
        except Exception as e:
            _LOGGER.error("Không thể khởi tạo Google GenAI Client: %s", e)
            pool.mark_exhausted(key)
            continue

        daily_hits = 0
        for model_name in models_to_try:
            for attempt in range(3):  # thử lại khi chỉ là giới hạn theo phút
                pool.record_request(key)
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={"temperature": temperature, "max_output_tokens": max_output_tokens},
                    )
                    text = (response.text or "").strip()
                    if text and len(text) > 20:
                        _LAST_LLM_MODEL = model_name
                        return text
                    _LOGGER.warning("Model '%s' trả về nội dung rỗng hoặc quá ngắn.", model_name)
                    break
                except Exception as e:
                    last_error = e
                    kind = llm_keys.classify_error(e)
                    if kind == "daily":
                        daily_hits += 1
                        break
                    if kind == "rate" and attempt < 2:
                        delay = llm_keys.retry_delay_seconds(e)
                        _LOGGER.warning("Gemini '%s' bị giới hạn theo phút, chờ %.0fs rồi thử lại.", model_name, delay)
                        time.sleep(delay)
                        continue
                    _LOGGER.warning("Gemini model '%s' thất bại: %s. Đang chuyển sang model tiếp theo...", model_name, e)
                    break
        if daily_hits and daily_hits >= len(models_to_try):
            _LOGGER.warning("Khóa Gemini %s đã hết hạn mức ngày, chuyển sang khóa kế tiếp.", llm_keys.fingerprint(key))
            pool.mark_exhausted(key)
            continue
        break  # lỗi không phải do hạn mức: không đốt thêm khóa khác

    if pool.acquire() is None and llm_keys.is_strict():
        raise llm_keys.LLMQuotaExhausted("Tất cả khóa Gemini đã hết hạn mức hôm nay.")
    _LOGGER.error("Toàn bộ danh sách model Gemini đều thất bại. Lỗi cuối cùng: %s", last_error)
    return None


# ---------------------------------------------------------------------
# Chọn nhà cung cấp LLM: Gemini (mặc định, giữ nguyên) hoặc Qwen2.5 cục bộ qua Ollama.
# Chọn bằng biến môi trường LLM_PROVIDER=gemini|ollama hoặc set_llm_provider() khi chạy.
# ---------------------------------------------------------------------
LLM_PROVIDERS = {
    "gemini": "Gemini (Google API)",
    "ollama": "Qwen2.5 (Ollama, chạy cục bộ)",
}
_LLM_PROVIDER = None  # None → đọc từ biến môi trường


def set_llm_provider(provider):
    """Đặt nhà cung cấp LLM cho các lần gọi sau ('gemini' | 'ollama'; None = theo env)."""
    global _LLM_PROVIDER
    if provider is not None and provider not in LLM_PROVIDERS:
        raise ValueError(f"LLM provider phải thuộc {sorted(LLM_PROVIDERS)}, nhận: {provider!r}")
    _LLM_PROVIDER = provider


def get_llm_provider():
    if _LLM_PROVIDER:
        return _LLM_PROVIDER
    env = (os.environ.get("LLM_PROVIDER") or "gemini").strip().lower()
    return env if env in LLM_PROVIDERS else "gemini"


def get_ollama_model():
    return (os.environ.get("OLLAMA_MODEL") or "qwen2.5:latest").strip()


def ollama_status():
    """(ok, thông báo) — Ollama đang chạy và đã có model đã chọn chưa."""
    import json as _json
    import urllib.request
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as resp:
            names = [m.get("name", "") for m in _json.loads(resp.read()).get("models", [])]
    except Exception as exc:
        return False, f"Không kết nối được Ollama tại {host}: {exc}"
    model = get_ollama_model()
    if model not in names and not any(n.split(":")[0] == model.split(":")[0] and ":" not in model for n in names):
        return False, f"Ollama chưa có model '{model}'. Hãy chạy: ollama pull {model}"
    return True, f"Ollama sẵn sàng ({model})"


def _llm_ready():
    """True nếu nhà cung cấp LLM đang chọn có thể được gọi."""
    if get_llm_provider() == "ollama":
        return True  # lỗi kết nối được xử lý khi gọi, trả None
    return bool(llm_keys.load_keys())


def _call_ollama(prompt, temperature=0.3, max_output_tokens=4096):
    """Gọi Ollama /api/chat (Qwen2.5). Trả về text hoặc None nếu lỗi."""
    import json as _json
    import urllib.error
    import urllib.request
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = get_ollama_model()
    body = _json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_output_tokens,
            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "8192")),
        },
    }).encode("utf-8")
    req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=int(os.environ.get("OLLAMA_TIMEOUT", "600"))) as resp:
            data = _json.loads(resp.read())
        text = ((data.get("message") or {}).get("content") or "").strip()
        if text and len(text) > 20:
            global _LAST_LLM_MODEL
            _LAST_LLM_MODEL = model
            return text
        _LOGGER.warning("Ollama model '%s' trả về nội dung rỗng hoặc quá ngắn.", model)
    except urllib.error.URLError as exc:
        if llm_keys.is_strict() and not isinstance(exc, urllib.error.HTTPError):
            raise llm_keys.LLMUnavailable(f"Không kết nối được Ollama tại {host}: {exc}") from exc
        _LOGGER.error("Gọi Ollama (%s) thất bại: %s", model, exc)
    except Exception as exc:
        _LOGGER.error("Gọi Ollama (%s) thất bại: %s", model, exc)
    return None


def _call_llm(prompt, temperature=0.3, max_output_tokens=4096, api_key=None):
    """Điểm gọi LLM duy nhất cho Config A/B/C: định tuyến theo nhà cung cấp đang chọn."""
    if get_llm_provider() == "ollama":
        return _call_ollama(prompt, temperature=temperature, max_output_tokens=max_output_tokens)
    return _call_gemini_with_fallback(
        prompt, temperature=temperature, max_output_tokens=max_output_tokens, api_key=api_key,
    )


# Cổng nguồn: câu trả lời của LLM chỉ được hiển thị khi MỌI mã nguồn nằm trong danh sách bằng chứng đã truy xuất và có
# ít nhất một mã (trừ khi chính nhãn là ABSTAIN, khi đó câu trả lời là lời từ chối). Đặt STRICT_SOURCE_GATE=0 để tắt
# (chỉ dùng cho so sánh với số đo cũ).
STRICT_SOURCE_GATE = os.environ.get("STRICT_SOURCE_GATE", "1").strip() != "0"
_BRACKET_TAG = re.compile(r"\[([^\]\n]{2,200})\]")


def _is_source_like_tag(tag):
    """Thẻ trông như mã nguồn: có dấu gạch dưới (JECFA_EVALUATIONS) hoặc mở đầu bằng Nguồn/Source/Ref."""
    return "_" in tag or tag.lower().startswith(("nguồn", "nguon", "source", "ref"))


def check_answer_sources(text, allowed, label=None, also_accept_labels=()):
    """Kiểm tra câu trả lời đã làm sạch có bám nguồn không. Trả về (đạt, lý do khi không đạt).

    Không đạt khi: (1) có mã nguồn nằm ngoài danh sách `allowed` (nguồn bịa hoặc không được truy xuất cho câu này);
    (2) không có mã nguồn nào, trừ khi nhãn là ABSTAIN. Chưa kiểm từng câu có thẻ riêng hay chưa."""
    allowed_set = {str(a) for a in (allowed or [])}
    cited, foreign = [], []
    for match in _BRACKET_TAG.finditer(text or ""):
        for part in re.split(r"[,;&]", match.group(1)):
            part = part.strip()
            if part and _is_source_like_tag(part):
                cited.append(part)
                if part not in allowed_set:
                    foreign.append(part)
    if foreign:
        return False, "câu trả lời dẫn nguồn ngoài danh sách bằng chứng đã truy xuất: " + ", ".join(dict.fromkeys(foreign[:3]))
    if not cited and label != "ABSTAIN":
        return False, ("câu trả lời không dẫn nguồn nào" if allowed_set
                       else "không truy xuất được nguồn nào cho câu hỏi này")
    # (3) Nhãn LLM tự ghi phải trùng nhãn của bộ luật (bộ luật quyết định nhãn, LLM chỉ diễn đạt). Thiếu dòng nhãn
    # thì không chặn (giữ hành vi cũ); chỉ chặn khi LLM ghi một nhãn KHÁC.
    if label:
        llm_label = extract_safety_label(text, default="")
        if llm_label and llm_label != str(label).upper() and llm_label not in {str(a).upper() for a in also_accept_labels}:
            return False, f"nhãn do mô hình ghi ({llm_label}) khác nhãn của bộ luật ({str(label).upper()})"
    return True, ""


def clean_llm_citations_and_label(text, expected_label=None):
    """
    Chuẩn hóa câu trả lời của LLM:
    1. Loại bỏ các chuỗi rác (ví dụ: '[Nguồn: Quy định Việt Nam chưa được xác minh...]').
    2. Chuẩn hóa các trích dẫn dài/viết tay thành mã quy chuẩn chuẩn hóa [MÃ_NGUỒN]
       (ví dụ: '[Nguồn: US FDA FALCPA & Codex (CXS 1-1985)]' -> '[US_FDA_FALCPA] [CODEX_CXS_1_1985]').
    3. Giữ lại [CODEX_CXS_1_1985] gọn gàng trong ngoặc vuông.
    4. Chuẩn hóa dòng cuối: SAFETY_LABEL: <nhãn hợp lệ> (không dùng WARNING).
    """
    if not text:
        return text

    # 1. Xóa bỏ hoàn toàn các chuỗi rác chưa được xác minh
    text = re.sub(
        r'\[\s*(?:Nguồn:?\s*)?[^\]]*chưa được xác minh[^\]]*\]',
        '',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'\[\s*(?:Nguồn:?\s*)?[^\]]*quy định việt nam chưa[^\]]*\]',
        '',
        text,
        flags=re.IGNORECASE,
    )

    # 2. Chuẩn hóa các trích dẫn trong ngoặc vuông
    def _canon_bracket(match):
        inner = match.group(1).strip()
        inner_no_prefix = re.sub(r'^(?:Nguồn|Source|Ref):\s*', '', inner, flags=re.IGNORECASE).strip()

        # Nhận diện các mã quy chuẩn có trong cụm text
        found = []
        if re.search(r'FALCPA', inner_no_prefix, re.I):
            found.append('US_FDA_FALCPA')
        if re.search(r'FASTER\s*ACT', inner_no_prefix, re.I):
            found.append('US_FDA_FASTER_ACT_2023')
        if re.search(r'JEMRA.*Part\s*3|PAL|precautionary', inner_no_prefix, re.I):
            found.append('FAO_WHO_JEMRA_PART_3')
        elif re.search(r'JEMRA', inner_no_prefix, re.I):
            found.append('WHO_JEMRA_PRIORITY_ALLERGENS')
        if re.search(r'CXS\s*1[-_]1985|Codex.*1985|Codex.*Labelling', inner_no_prefix, re.I):
            found.append('CODEX_CXS_1_1985')
        if re.search(r'STAN\s*192|GSFA', inner_no_prefix, re.I):
            found.append('CODEX_STAN_192_1995')
        if re.search(r'TT24|24/2019', inner_no_prefix, re.I):
            found.append('VN_TT24_2019')
        if re.search(r'EFSA.*171|titanium', inner_no_prefix, re.I):
            found.append('EFSA_2021_TITANIUM_DIOXIDE_E171')
        elif re.search(r'EFSA', inner_no_prefix, re.I):
            found.append('EFSA_OPINIONS')
        if re.search(r'JECFA', inner_no_prefix, re.I):
            found.append('JECFA_EVALUATIONS')
        if re.search(r'USDA|FDC|FNDDS', inner_no_prefix, re.I):
            found.append('USDA_FDC')

        if found:
            # Khử trùng lặp giữ nguyên thứ tự
            seen = set()
            dedup = [c for c in found if not (c in seen or seen.add(c))]
            return ' '.join(f'[{c}]' for c in dedup)

        canon = canonicalize_citation(inner_no_prefix)
        if canon and is_recognized_citation(canon):
            return f'[{canon}]'

        return f'[{inner_no_prefix}]'

    text = re.sub(r'\[([^\]]+)\]', _canon_bracket, text)

    # 3. Làm sạch khoảng trắng thừa và dấu chấm phẩy liền kề
    text = re.sub(r'\s+([,.\)])', r'\1', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # 4. Chuẩn hóa nhãn an toàn dòng cuối
    if expected_label:
        norm_exp = guardrail_status_to_safety_label(expected_label)
        text = re.sub(r'SAFETY_LABEL:\s*WARNING', f'SAFETY_LABEL: {norm_exp}', text, flags=re.IGNORECASE)
    else:
        text = re.sub(r'SAFETY_LABEL:\s*WARNING', 'SAFETY_LABEL: UNSAFE', text, flags=re.IGNORECASE)

    return text.strip()


def drop_unrelated_jecfa_passages(passages, linked_results):
    """Bỏ đoạn JECFA nói về phụ gia KHÔNG có trên nhãn. Truy xuất lấp chỗ trống bằng đoạn gần giống nhất, nên với phụ gia
    không có dữ liệu JECFA (vd INS 307c) các đoạn JECFA của chất khác vẫn lọt vào và biến JECFA_EVALUATIONS thành
    nguồn "được phép" cho câu trả lời về chất không có dữ liệu đó. Nguồn khác (Codex, TT 24…) giữ nguyên."""
    from modules.passage_retriever import normalize_ins
    on_label = {
        normalize_ins((it.get("matched_entity") or {}).get("ins_code", ""))
        for it in (linked_results or []) if it.get("matched_entity")
    } - {""}
    return [
        p for p in (passages or [])
        if p.get("citation_id") != "JECFA_EVALUATIONS"
        or on_label & {normalize_ins(c) for c in p.get("ins_codes", [])}
    ]


def generate_llm_answer(evaluation, linked_results, persona, question,
                        detected_hazards=None, matched_fndds=None,
                        nutri_warnings=None, qa_context="", passages=None):
    """
    Gọi Gemini Flash sinh câu trả lời tự nhiên có trích dẫn từ bằng chứng RAG.
    `passages`: các đoạn bằng chứng đã truy xuất (modules.passage_retriever), có thể None.
    Trả về string câu trả lời, hoặc None nếu không có API key / lỗi.
    Nhà cung cấp LLM: xem set_llm_provider() (Gemini hoặc Qwen2.5 qua Ollama).
    """
    api_key = _get_gemini_api_key()
    if not _llm_ready():
        return None

    conditions = ", ".join(persona.get("conditions", [])) or "Không có"
    allergies = ", ".join(persona.get("allergies", [])) or "Không có"
    persona_info = f"Bệnh lý: {conditions}. Dị ứng: {allergies}."
    passages = drop_unrelated_jecfa_passages(passages, linked_results)

    evidence_parts = []

    evidence_parts.append(f"### Kết quả đánh giá: {evaluation.get('title', '')}")
    evidence_parts.append(f"Lời khuyên hệ thống: {evaluation.get('advice', evaluation.get('message', ''))}")

    alerts = evaluation.get("alerts", [])
    if alerts:
        evidence_parts.append("### Cảnh báo rủi ro:")
        for a in alerts:
            reg = canonicalize_citation(a.get("regulation", ""))
            codex = canonicalize_citation(a.get("codex", ""))
            c_tags = " ".join(f"[{c}]" for c in [reg, codex] if c and is_recognized_citation(c))
            evidence_parts.append(f"- {a['additive']}: {a['message']} {c_tags}".strip())

    entity_lines = []
    for item in linked_results:
        e = item.get("matched_entity")
        if e:
            ctype = item.get("constraint_type", "PERMITTED")
            entity_lines.append(f"- {item['raw_mention']} → {e['canonical_name']} ({e.get('ins_code', '')}) | Chức năng: {e.get('function', '')} | ADI: {e.get('jecfa_adi', 'N/A')} | Ràng buộc: {ctype}")
    if entity_lines:
        evidence_parts.append("### Thành phần nhận diện:")
        evidence_parts.extend(entity_lines)

    for reg in evaluation.get("regulatory", []):
        adi = reg.get("adi", {})
        evidence_parts.append(f"- {reg['additive']}: ADI = {adi.get('detail', 'N/A')} [{adi.get('source', 'JECFA')}]")

    for ew in evaluation.get("efsa_warnings", []):
        evidence_parts.append(f"- EFSA ({ew.get('year', '')}): {ew.get('substance', '')} — {ew.get('key_finding', '')} [Ref: {ew.get('reference', '')}]")

    infant = evaluation.get("infant_check")
    if infant:
        evidence_parts.append("### Đối chiếu danh mục phụ gia cho thực phẩm trẻ dưới 12 tháng (TT 24/2019 Phụ lục 2A; Codex GSFA nhóm 13.1, 13.2):")
        for p in infant.get("permitted", []):
            evidence_parts.append(f"- {p['summary']} [VN_TT24_2019]")
        for a in infant.get("alerts", []):
            evidence_parts.append(f"- KHÔNG có trong danh mục: {a['additive']} [VN_TT24_2019] [CODEX_STAN_192_1995]")
        if infant.get("nutrients"):
            evidence_parts.append("- Nguồn khoáng/vitamin, KHÔNG xét như phụ gia: "
                                  + ", ".join(n["additive"] for n in infant["nutrients"]) + " [CODEX_CXS_72_1981] [VN_QCVN_11_1_2012]")
        for a in infant.get("label_age", []):
            evidence_parts.append(f"- NHÃN GHI GIỚI HẠN TUỔI: \"{a['label_excerpt']}\" (từ {a['min_age_months']} tháng) — sản phẩm "
                                  "không dành cho bé ở tuổi này; nêu rõ điều này trong câu trả lời. [CODEX_CXS_74_1981]")
        if not infant.get("checked"):
            evidence_parts.append("- Nhãn không có phụ gia mang mã INS để đối chiếu.")
    for note in evaluation.get("population_notes", []):
        extra = (" TUYỆT ĐỐI không viết rằng sản phẩm 'an toàn', 'ổn' hay 'không cần tránh' đối với người này; hãy nói rõ hệ thống "
                 "không có dữ liệu về bệnh này và khuyên hỏi bác sĩ." if note.get("code") == "CONDITION_NO_ADDITIVE_EVIDENCE" else "")
        evidence_parts.append(f"**LƯU Ý NHÓM NGƯỜI DÙNG ({note['group']})**: {note['text']} Hãy nêu lưu ý này trong câu trả lời.{extra}")

    if detected_hazards:
        evidence_parts.append("### Lịch sử sự cố liên quan:")
        for h in detected_hazards[:3]:
            evidence_parts.append(f"- [{h.get('severity', '')}] {h.get('title', '')} — Tác nhân: {h.get('hazard', '')}")

    if matched_fndds:
        source_type = matched_fndds.get("data_source_type", "fndds_reference")
        is_estimate = matched_fndds.get("is_estimate", True)
        source_label = {
            "product_label": "Nhãn sản phẩm",
            "vcndd_reference": "Bảng TPTP Việt Nam (ước tính)",
            "fndds_reference": "USDA FNDDS (ước tính)",
        }.get(source_type, "Tham chiếu")
        evidence_parts.append(f"### Dinh dưỡng ({source_label}): {matched_fndds.get('food_description', '')}")
        if is_estimate:
            disclaimer = matched_fndds.get("estimate_disclaimer", "")
            if disclaimer:
                evidence_parts.append(f"**LƯU Ý**: {disclaimer}")
        for key in ["calories_kcal", "sodium_mg", "sugars_g", "fat_g", "protein_g"]:
            val = matched_fndds.get(key)
            if val:
                evidence_parts.append(f"- {key}: {val}")
    if nutri_warnings:
        for nw in nutri_warnings:
            evidence_parts.append(f"- Cảnh báo dinh dưỡng: {nw}")
    if qa_context:
        evidence_parts.append(qa_context)
    if passages:
        from modules.passage_retriever import render_passages
        evidence_parts.append(render_passages(passages))

    citations = evaluation.get("citations", [])
    canonical_citations = []
    for c in citations:
        canon = canonicalize_citation(c)
        if canon and is_recognized_citation(canon) and canon not in canonical_citations:
            canonical_citations.append(canon)

    for p_ in passages or []:  # cite only sources that were actually retrieved
        canon = canonicalize_citation(p_.get("citation_id", ""))
        if canon and is_recognized_citation(canon) and canon not in canonical_citations:
            canonical_citations.append(canon)

    if matched_fndds:
        for nc in ["CODEX_CXS_1_1985", "USDA_FDC"]:
            if nc not in canonical_citations:
                canonical_citations.append(nc)

    if evaluation.get("efsa_warnings") or any("efsa" in str(item.get("matched_entity") or {}).lower() for item in linked_results):
        has_e171 = any("171" in str(ew.get("substance", "")) or "titanium" in str(ew.get("substance", "")).lower() for ew in evaluation.get("efsa_warnings", []))
        target_efsa = "EFSA_2021_TITANIUM_DIOXIDE_E171" if has_e171 else "EFSA_OPINIONS"
        if target_efsa not in canonical_citations:
            canonical_citations.append(target_efsa)

    if any((item.get("matched_entity") or {}).get("jecfa_adi") for item in linked_results) or any((reg.get("adi") or {}).get("source") == "JECFA" for reg in evaluation.get("regulatory", [])):
        if "JECFA_EVALUATIONS" not in canonical_citations:
            canonical_citations.append("JECFA_EVALUATIONS")

    if canonical_citations:
        evidence_parts.append("### Danh sách mã nguồn quy chuẩn bắt buộc (hãy trích dẫn đúng mã này trong ngoặc vuông [MÃ_NGUỒN]):")
        for c in canonical_citations:
            evidence_parts.append(f"- [{c}]")

    evidence_text = "\n".join(evidence_parts)

    canonical_safety_status = evaluation.get("safety_label") or guardrail_status_to_safety_label(evaluation.get("status", ""))
    # Bệnh nền chưa có dữ liệu: bộ luật giữ SAFE nhưng prompt cấm mô hình nói "an toàn" (xem LƯU Ý NHÓM NGƯỜI DÙNG),
    # nên mô hình ghi CAUTION là đúng ý hệ thống, không phải mâu thuẫn.
    _accept_labels = ("CAUTION",) if canonical_safety_status == "SAFE" and any(
        n.get("code") == "CONDITION_NO_ADDITIVE_EVIDENCE" for n in evaluation.get("population_notes", [])) else ()

    prompt = _LLM_ANSWER_PROMPT.format(
        length_rule=ANSWER_LENGTH_RULE,
        persona_info=persona_info,
        question=question or "Sản phẩm này có an toàn cho tôi không?",
        status=canonical_safety_status,
        evidence=evidence_text,
    )

    result = _call_llm(
        prompt,
        temperature=0.3,
        max_output_tokens=4096,
        api_key=api_key,
    )
    def _retry_prompt():
        return (
            prompt
            + "\n\nNHẮC LẠI BẮT BUỘC: sau MỖI nhận định phải có mã nguồn trong ngoặc vuông, chỉ dùng các mã: "
            + ", ".join(f"[{c}]" for c in canonical_citations)
            + ". Ví dụ: \"Natri benzoat có ADI 0–5 mg/kg [JECFA_EVALUATIONS].\" Kết thúc bằng dòng SAFETY_LABEL."
        )

    retried = False
    if result and get_llm_provider() == "ollama" and canonical_citations             and not extract_citations(clean_llm_citations_and_label(result, expected_label=canonical_safety_status)):
        # Model cục bộ nhỏ hay quên trích dẫn: gọi lại một lần với nhắc nhở, không tự chèn nguồn thay model.
        retried = True
        again = _call_llm(_retry_prompt(), temperature=0.2, max_output_tokens=4096, api_key=api_key)
        if again:
            result = again
    if result:
        result = clean_llm_citations_and_label(result, expected_label=canonical_safety_status)
        if STRICT_SOURCE_GATE:
            ok, reason = check_answer_sources(result, canonical_citations, canonical_safety_status, _accept_labels)
            if not ok and canonical_citations and not retried:
                # thử lại đúng một lần (mọi nhà cung cấp) trước khi từ chối
                again = _call_llm(_retry_prompt(), temperature=0.2, max_output_tokens=4096, api_key=api_key)
                if again:
                    result = clean_llm_citations_and_label(again, expected_label=canonical_safety_status)
                    ok, reason = check_answer_sources(result, canonical_citations, canonical_safety_status, _accept_labels)
            if not ok:
                _LOGGER.warning("Từ chối câu trả lời của LLM vì không bám nguồn: %s", reason)
                evaluation["answer_refusal"] = reason  # pipeline trả ra cho giao diện và bộ đánh giá
                return None
        citation_check = validate_citations(
            result,
            canonical_citations or citations,
            require_each_sentence=False,
        )
        if not citation_check["valid"]:
            _LOGGER.warning("Citation check did not pass: %s", citation_check)
        return result
    _LOGGER.warning("Gemini returned an empty or too-short answer.")
    return None


_LLM_ONLY_PROMPT = """Bạn là chuyên gia tư vấn an toàn thực phẩm.
Người dùng cung cấp danh sách thành phần đọc từ nhãn bao bì thực phẩm.
Hãy trả lời câu hỏi của họ DỰA TRÊN KIẾN THỨC CỦA BẠN (không có cơ sở dữ liệu tra cứu bên ngoài).

## Lưu ý quan trọng:
- Bạn KHÔNG có cơ sở dữ liệu quy chuẩn, y văn hay bằng chứng khoa học để tra cứu.
- Câu trả lời chỉ mang tính gợi ý dựa trên kiến thức tổng quát, KHÔNG phải kết luận y khoa.
- Sử dụng ngôn ngữ khuyến nghị: "nên cân nhắc", "khuyến nghị tham khảo ý kiến bác sĩ".
- Nếu không chắc chắn, hãy nêu rõ và khuyên tham vấn chuyên gia.
- {length_rule}
- Dòng cuối cùng bắt buộc ghi nhãn an toàn chuẩn hóa: SAFETY_LABEL: SAFE|CAUTION|UNSAFE|CRITICAL_ALERT|ABSTAIN (KHÔNG dùng WARNING)

## Hồ sơ người dùng:
{persona_info}

## Thành phần trên nhãn:
{ingredients_text}

## Câu hỏi:
{question}

## Trả lời (tiếng Việt, ngắn gọn, mang tính gợi ý, không kết luận tuyệt đối):
"""


def generate_llm_only_answer(raw_text, persona, question):
    """
    Cấu hình A — Baseline LLM không truy xuất:
    Gemini nhận thành phần sản phẩm + hồ sơ persona + câu hỏi, nhưng KHÔNG tra cứu KB/RAG.
    Dùng cho ablation study — đo giá trị của retrieval (so sánh A vs B).
    """
    api_key = _get_gemini_api_key()
    if not _llm_ready():
        return None

    conditions = ", ".join(persona.get("conditions", [])) or "Không có"
    allergies = ", ".join(persona.get("allergies", [])) or "Không có"
    persona_info = f"Bệnh lý: {conditions}. Dị ứng: {allergies}."

    prompt = _LLM_ONLY_PROMPT.format(
        length_rule=ANSWER_LENGTH_RULE,
        persona_info=persona_info,
        ingredients_text=raw_text or "(không có thông tin thành phần)",
        question=question or "Sản phẩm này có an toàn cho tôi không?",
    )

    return _call_llm(
        prompt,
        temperature=0.3,
        max_output_tokens=4096,
        api_key=api_key,
    )


_SAFETY_LABEL_PATTERN = re.compile(
    r"\b(SAFE|CAUTION|WARNING|UNSAFE|CRITICAL_ALERT|ABSTAIN)\b",
    re.IGNORECASE,
)


def extract_safety_label(answer_text, default="ABSTAIN"):
    """Bóc nhãn an toàn ở dòng SAFETY_LABEL; không tự suy diễn SAFE khi thiếu nhãn."""
    if not answer_text:
        return default
    for line in reversed(str(answer_text).strip().splitlines()):
        if "SAFETY_LABEL" not in line.upper():
            continue
        match = _SAFETY_LABEL_PATTERN.search(line)
        if match:
            return match.group(1).upper()
    return default


def guardrail_status_to_safety_label(status):
    """Chuẩn hóa trạng thái máy luật về tập nhãn benchmark."""
    normalized = str(status or "").strip().upper()
    mapping = {
        "SCREENED_NO_ALERT": "SAFE",
        "SAFE": "SAFE",
        "CAUTION": "CAUTION",
        # Benchmark v3 gộp cảnh báo nguy cơ cao vào UNSAFE; WARNING vẫn được
        # giữ nguyên trong evaluation.status để UI trình bày mức cảnh báo.
        "WARNING": "UNSAFE",
        "UNSAFE": "UNSAFE",
        "CRITICAL_ALERT": "CRITICAL_ALERT",
        "ABSTAIN": "ABSTAIN",
    }
    return mapping.get(normalized, "ABSTAIN")


def build_persona_from_benchmark_record(record, personas=None):
    """
    Ánh xạ persona theo persona_id của benchmark, không đoán bằng từ khóa câu hỏi.

    `personas` chỉ cung cấp metadata nền; các persona tổng hợp CKD và
    diabetes+hypertension được dựng rõ ràng để cả UI và batch runner dùng chung.
    """
    record = record or {}
    pid = str(record.get("persona_id", "healthy_adult")).strip().lower()
    pname = str(record.get("persona_name", "")).strip()
    available = {}
    for persona in personas or []:
        for key in (persona.get("id"), persona.get("persona_id")):
            if key:
                available[str(key).strip().lower()] = persona

    candidate_ids = {
        "healthy_adult": ("healthy_adult", "user_clinical_007"),
        "pregnant_woman": ("pregnant_woman", "user_clinical_004"),
        "child_5yo": ("child_5yo", "user_clinical_003"),
        "allergy_patient": ("allergy_patient", "user_clinical_006"),
        "pku_patient": ("pku_allergy_patient", "pku_patient", "user_clinical_001"),
        "ckd_stage3": ("ckd_stage3", "user_clinical_005"),
        "diabetes_hypertension": (
            "diabetes_hypertension",
            "diabetes_type2",
            "user_clinical_002",
        ),
    }

    base = None
    for candidate in candidate_ids.get(pid, (pid,)):
        if candidate in available:
            base = deepcopy(available[candidate])
            break
    if base is None:
        base = {
            "id": pid,
            "persona_id": pid,
            "name": pname or pid,
            "description": "",
            "conditions": [],
            "allergies": [],
        }

    exact_profiles = {
        "healthy_adult": {
            "conditions": ["healthy", "healthy_adult"],
            "allergies": [],
        },
        "pregnant_woman": {
            "conditions": ["pregnancy", "pregnancy_second_trimester"],
            "allergies": [],
        },
        "child_5yo": {
            "conditions": ["children_adhd", "children_under_6"],
            "allergies": [],
            "weight_kg": 18.0,
        },
        "ckd_stage3": {
            "conditions": ["chronic_kidney_disease_stage_3", "ckd", "impaired_renal_filtration"],
            "allergies": [],
        },
        "diabetes_hypertension": {
            "conditions": ["diabetes", "hypertension", "cardiovascular_risk"],
            "allergies": [],
        },
        "pku_patient": {
            "conditions": ["phenylketonuria_pku"],
            "allergies": ["phenylalanine"],
        },
    }
    if pid in exact_profiles:
        base.update(deepcopy(exact_profiles[pid]))

    if pid == "allergy_patient":
        condition_text = str(record.get("persona_conditions", "")).casefold()
        question_text = str(record.get("question", "")).casefold()
        combined = f"{condition_text} {question_text}"
        condition_code_map = {
            "soy_allergy": "soy",
            "milk_allergy": "milk",
            "egg_allergy": "egg",
            "wheat_allergy": "wheat",
            "gluten_allergy": "wheat",
            "peanut_allergy": "peanut",
            "treenut_allergy": "tree_nut",
            "tree_nut_allergy": "tree_nut",
            "sesame_allergy": "sesame",
            "fish_allergy": "fish",
            "shrimp_allergy": "shellfish",
            "crab_allergy": "shellfish",
            "seafood_allergy": "shellfish",
            "shellfish_allergy": "shellfish",
        }
        condition_codes = {
            item.strip() for item in re.split(r"[,;]", condition_text) if item.strip()
        }
        allergies = [
            allergen for code, allergen in condition_code_map.items()
            if code in condition_codes
        ]
        allergen_terms = {
            "soy": ("soy_allergy", "đậu nành", "đậu tương", "soy", "lecithin"),
            "milk": ("milk_allergy", "sữa", "milk", "casein", "whey"),
            "egg": ("egg_allergy", "trứng", "egg", "albumin"),
            "wheat": ("wheat_allergy", "gluten_allergy", "gluten", "lúa mì", "wheat", "bột mì"),
            "peanut": ("peanut_allergy", "đậu phộng", "peanut", "lạc"),
            "tree_nut": ("treenut_allergy", "tree_nut_allergy", "hạt cây", "tree nut"),
            "sesame": ("sesame_allergy", "mè", "vừng", "sesame"),
            "fish": ("fish_allergy", "cá", "fish"),
            "shellfish": (
                "shellfish_allergy", "seafood_allergy", "shrimp_allergy",
                "hải sản", "shellfish", "tôm", "cua", "mực",
            ),
        }
        if not allergies:
            allergies = [
                name for name, terms in allergen_terms.items()
                if any(
                    re.search(rf"(?<!\w){re.escape(term)}(?!\w)", combined)
                    for term in terms
                )
            ]
        allergies = list(dict.fromkeys(allergies))
        if not allergies:
            allergies = list(base.get("allergies", []))
        base["allergies"] = allergies
        base["conditions"] = ["severe_food_allergies"] + [f"{item}_allergy" for item in allergies]

    base["benchmark_persona_id"] = pid
    base["name"] = pname or base.get("name") or pid
    benchmark_conditions_vi = str(record.get("persona_conditions_vi", "")).strip()
    if benchmark_conditions_vi:
        base["description"] = f"Hồ sơ benchmark: {benchmark_conditions_vi}."
    base.setdefault("id", pid)
    base.setdefault("persona_id", base["id"])
    base.setdefault("conditions", [])
    base.setdefault("allergies", [])
    return base


def run_grounded_pipeline(
    mentions,
    avg_ocr_conf,
    product_name,
    persona,
    question,
    kb_additives,
    hazards_list,
    fndds_list,
    off_products,
    generate_answer=True,
    dynamic_qa_enabled=True,
    passage_retrieval_enabled=True,
    passage_top_k=5,
    passage_mode=None,
    linked_results=None,
    label_text=None,
):
    """
    Downstream chuẩn dùng chung tuyệt đối cho Config B và Config C.

    linked_results: kết quả liên kết phụ gia đã tính cho CÙNG danh sách mentions (chỉ phụ thuộc chữ trên nhãn, không
    phụ thuộc câu hỏi hay hồ sơ) — truyền vào để không liên kết lại khi hỏi nhiều câu trên cùng một ảnh.

    Hai cấu hình chỉ được phép khác nguồn mentions/confidence trước hàm này:
    B dùng Ground Truth text, C dùng ảnh qua OCR + lexical normalization.
    """
    mentions = list(mentions or [])
    if linked_results is None:
        linked_results = link_all_mentions(mentions, kb_additives)
    else:
        linked_results = copy.deepcopy(list(linked_results))
    # Tầng QA động: hiểu câu hỏi → tra KB theo phụ gia mà câu hỏi nhắc tới → phát hiện khoảng trống tri thức.
    qa_plan, question_additives, qa_gaps = None, [], []
    if dynamic_qa_enabled:
        qa_plan = dynamic_qa.understand_question(question, persona)
        question_additives = dynamic_qa.lookup_question_additives(qa_plan, kb_additives, linked_results)
        qa_gaps = dynamic_qa.knowledge_gaps(question_additives)
    evaluation = evaluate_safety_and_guardrail(
        linked_results,
        persona,
        question,
        avg_ocr_conf=avg_ocr_conf,
        all_mentions=mentions,
        extra_gaps=qa_gaps,
    )
    detected_hazards = check_food_hazards(product_name, mentions, hazards_list)
    # Số liệu dinh dưỡng GHI TRÊN NHÃN: do người dùng nêu trong câu hỏi, hoặc đọc từ bảng dinh dưỡng của nhãn
    label_nutrition = extract_label_nutrition(question, label_text)
    matched_fndds, nutri_warnings, constraint_details = evaluate_nutritional_risk(
        product_name,
        fndds_list,
        persona,
        label_nutrition=label_nutrition,
    )
    has_nutri_conditions = any(
        any(k in str(c).lower() for k in ["tiểu đường", "cao huyết áp", "diabetes", "hypertension", "ckd", "suy thận", "gout", "thận", "tim mạch"])
        for c in persona.get("conditions", [])
    ) or any(
        k in str(question).lower() for k in ["dinh dưỡng", "đường", "muối", "natri", "calo", "béo", "sugar", "sodium", "salt", "fat"]
    )
    if matched_fndds and not matched_fndds.get("is_estimate") and any(
        d.get("citation") == "CODEX_CAC_GL_23_1997" for d in (constraint_details or [])
    ):
        cits = evaluation.setdefault("citations", [])
        if "CODEX_CAC_GL_23_1997" not in cits:
            cits.append("CODEX_CAC_GL_23_1997")
    if matched_fndds and (nutri_warnings or has_nutri_conditions):
        cits = evaluation.setdefault("citations", [])
        for c in ["CODEX_CXS_1_1985", "USDA_FDC"]:
            if c not in cits:
                cits.append(c)
    # Chỉ số liệu dinh dưỡng của CHÍNH sản phẩm (bảng dinh dưỡng trên nhãn) mới được đổi kết luận an toàn.
    # Số liệu ước tính từ thực phẩm "tương tự" (USDA FNDDS / VCNDD) — vd khớp nhầm "Muscle Milk" cho sữa bột trẻ em —
    # chỉ hiển thị làm tham chiếu. NUTRITION_ESTIMATE_CHANGES_LABEL = True để tái lập hành vi cũ khi đối chiếu.
    nutrition_from_label = bool(matched_fndds) and not matched_fndds.get("is_estimate", True)
    if nutri_warnings and evaluation.get("status") == "SCREENED_NO_ALERT" and (
        nutrition_from_label or NUTRITION_ESTIMATE_CHANGES_LABEL
    ):
        evaluation["status"] = "CAUTION"
        evaluation["safety_label"] = "CAUTION"
        evaluation["badge_color"] = "yellow"
        evaluation["title"] = "🟡 LƯU Ý — VƯỢT NGƯỠNG DINH DƯỠNG KHUYẾN NGHỊ"
        evaluation["advice"] = (
            "Chỉ số dinh dưỡng vượt ngưỡng khuyến nghị cho hồ sơ bệnh lý; "
            "hãy cân nhắc liều lượng và tham khảo ý kiến chuyên gia."
        )
    off_cross_ref = cross_reference_openfoodfacts(product_name, mentions, off_products)
    qa_summary, qa_context = None, ""
    if dynamic_qa_enabled:
        cits = evaluation.setdefault("citations", [])
        for c in dynamic_qa.extra_citations(question_additives):
            if c not in cits:
                cits.append(c)
        qa_trace = dynamic_qa.build_trace(
            qa_plan, question_additives, linked_results, evaluation,
            detected_hazards, matched_fndds, nutri_warnings,
        )
        qa_summary = dynamic_qa.summarize(qa_plan, qa_trace, qa_gaps)
        qa_context = dynamic_qa.render_context(qa_plan, qa_trace)
    retrieved_passages = []
    if passage_retrieval_enabled:
        # Truy xuất đoạn bằng chứng: câu hỏi + mã INS đã liên kết từ nhãn (B: văn bản chuẩn, C: OCR).
        try:
            from modules.passage_retriever import get_passage_retriever
            linked_ins = [
                (it.get("matched_entity") or {}).get("ins_code", "")
                for it in linked_results if it.get("matched_entity")
            ]
            linked_ins = [c for c in dict.fromkeys(linked_ins) if c]
            retriever = get_passage_retriever()
            query = f"{question or ''} " + " ".join(linked_ins)
            # mặc định: có dense (PASSAGE_EMBEDDING=bge-m3) → hybrid_dense_exact, không thì hybrid_exact
            mode = passage_mode or ("hybrid_dense_exact" if retriever.has_dense else "hybrid_exact")
            retrieved_passages = retriever.retrieve(
                query, top_k=passage_top_k, mode=mode, ins_codes=linked_ins or None,
            )
        except Exception as exc:  # retrieval must never break the answer path
            _LOGGER.warning("Passage retrieval failed: %s", exc)
    llm_answer = None
    citation_validation = None
    if generate_answer:
        llm_answer = generate_llm_answer(
            evaluation,
            linked_results,
            persona,
            question,
            detected_hazards=detected_hazards,
            # số liệu ước tính không đưa cho LLM như thể là số liệu của sản phẩm
            matched_fndds=matched_fndds if (nutrition_from_label or NUTRITION_ESTIMATE_CHANGES_LABEL) else None,
            nutri_warnings=nutri_warnings if (nutrition_from_label or NUTRITION_ESTIMATE_CHANGES_LABEL) else [],
            qa_context=qa_context,
            passages=retrieved_passages,
        )
        if llm_answer:
            citation_validation = validate_citations(
                llm_answer,
                evaluation.get("citations", []),
                require_each_sentence=False,
            )
    return {
        "linked_results": linked_results,
        "evaluation": evaluation,
        "detected_hazards": detected_hazards,
        "matched_fndds": matched_fndds,
        "nutri_warnings": nutri_warnings,
        "constraint_details": constraint_details,
        "off_cross_ref": off_cross_ref,
        "llm_answer": llm_answer,
        "answer_refusal": evaluation.get("answer_refusal") if not llm_answer else None,
        "citation_validation": citation_validation,
        "qa": qa_summary,
        "retrieved_passages": retrieved_passages,
        "safety_label": guardrail_status_to_safety_label(evaluation.get("status")),
    }
