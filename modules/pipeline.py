"""
Module Pipeline cho hệ thống FoodHealthQA (v10 - Full Algorithm Implementation)
Kế thừa kiến trúc:
- Tầng 1: Perception (Hỗ trợ Ground Truth Benchmark hoặc PaddleOCR kèm Post-OCR Normalization)
- Tầng 2: Normalization (FoodOntoRAG 4-agent Entity Linking đầy đủ)
  + Agent 1: Hybrid Retriever (BM25 + TF-IDF/SBERT Semantic + Fusion)
  + Agent 2: Selector Agent (Exact-match preference + Specificity rule)
  + Agent 3: LLM Scorer (Confidence Calibration τ)
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
import json
import logging
import re
import pandas as pd
from modules.ocr_extractor import (
    extract_with_paddle_ocr,
    get_ground_truth_sample,
    run_perception,
    is_paddleocr_available
)
from modules.file_resolver import find_data_file, read_table_flexibly
from modules.foodon import enrich_additive_foodon_curie, load_foodon_entities
from modules.benchmark import validate_citations
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

    # Xác định nhóm NHANES phù hợp dựa trên persona
    persona_id = persona.get("id", "")
    matched_seqns = []

    for seqn, data in biomarkers.items():
        age = data.get("age")
        if age is None:
            continue

        if "child" in persona_id or "children" in persona_id:
            if age < 18:
                matched_seqns.append((seqn, data))
        elif "pregnant" in persona_id:
            if data.get("pregnant"):
                matched_seqns.append((seqn, data))
        elif "hypertension" in persona_id:
            sbp = data.get("systolic_bp")
            if sbp and float(sbp) >= 140:
                matched_seqns.append((seqn, data))
        elif "diabetes" in persona_id:
            hba1c = data.get("hba1c_pct")
            if hba1c and float(hba1c) >= 6.5:
                matched_seqns.append((seqn, data))
        elif "healthy" in persona_id:
            if 18 <= age <= 40:
                matched_seqns.append((seqn, data))

    if not matched_seqns:
        return persona

    # Thống kê nhóm NHANES phù hợp
    nhanes_group_size = len(matched_seqns)

    # Enrich conditions từ P_DEMO — thêm nếu persona chưa có
    sample_data = matched_seqns[0][1]
    derived, diet_habits = derive_conditions_from_biomarkers(sample_data)
    for cond in derived:
        if cond.lower() not in existing_conditions:
            conditions.append(cond)

    enriched["conditions"] = conditions

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


def evaluate_nutritional_risk(product_name, fndds_list, persona):
    """
    Đối soát ràng buộc dinh dưỡng định lượng với ngưỡng bệnh lý NHANES User Persona.
    Chiến lược: Tra Bảng TPTP Việt Nam trước → FNDDS USDA sau → chọn nguồn tốt nhất.
    """
    if not fndds_list and not _VN_FOOD_COMPOSITION_DB:
        return None, [], []

    vn_item = _match_vn_food(product_name)
    fndds_item, fndds_score = _match_fndds(product_name, fndds_list) if fndds_list else (None, -1)

    if vn_item and (fndds_score <= 0 or vn_item.get("source") == "VCNDD"):
        best_item = vn_item
    elif fndds_item:
        best_item = fndds_item
    elif vn_item:
        best_item = vn_item
    else:
        best_item = None

    if not best_item:
        return None, [], []

    constraint_warnings, constraint_details = evaluate_nutritional_constraints(
        best_item, persona
    )

    warnings = [w["message"] for w in constraint_warnings]

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
      Agent 3: LLM Scorer (Confidence Calibration τ)
      Agent 4: Synonym Generator & Feedback Loop (khi τ < 0.70)
    """
    return foodontorag_4agent_linking(mention, kb_additives)


def evaluate_safety_and_guardrail(linked_results, persona, question, avg_ocr_conf=0.95, all_mentions=None):
    """
    Tầng 4: Đánh giá an toàn, tính điểm tin cậy kép và kích hoạt Guardrail an toàn y tế.
    Tích hợp Negation Constraint Modeling: gán nhãn PERMITTED/RESTRICTED/CONTRAINDICATED/ALLERGEN_RISK.
    all_mentions: toàn bộ nguyên liệu OCR (bao gồm nguyên liệu thô không phải INS).
    """
    linked_results = apply_negation_constraints(linked_results, persona)

    s_perception = max(0.0, min(float(avg_ocr_conf), 1.0))
    retrieval_scores = [
        max(0.0, min(float(item.get("retrieval_confidence", 0.0)), 1.0))
        for item in linked_results
    ]
    s_retrieval = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
    s_total = 0.5 * s_perception + 0.5 * s_retrieval
    insufficient_evidence = s_total < 0.65
    confidence_metrics = {
        "s_perception": s_perception,
        "s_retrieval": s_retrieval,
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
                    "regulation": "Quy định Việt Nam chưa được xác minh cho claim này",
                    "codex": "Giới hạn Codex cần kiểm tra theo phụ gia và nhóm thực phẩm",
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
            "regulation": "Quy định Việt Nam chưa được xác minh cho claim này",
            "codex": "Giới hạn Codex cần kiểm tra theo phụ gia và nhóm thực phẩm",
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
            pal_alert["regulation"] = "FAO/WHO JEMRA Part 3 (Precautionary Allergen Labelling)"
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

        for raw_text in all_texts_to_scan:
            raw_text_lower = raw_text.lower()
            # Miễn trừ dầu tinh luyện hoàn toàn theo US FDA FALCPA Sec 203
            if is_exempt_refined_oil(raw_text_lower):
                continue

            is_prec_text = any(
                p in raw_text_lower for p in ["có thể chứa", "may contain", "vết của", "sản xuất trên", "chế biến trên"]
            )
            for allergy in user_allergies:
                keywords = list(ALLERGEN_KEYWORDS.get(allergy, [allergy]))
                derivatives = DERIVATIVE_ALLERGEN_MAP.get(allergy, [])
                combined_terms = keywords + [d for d in derivatives if d not in keywords]

                if any(contains_label_term(raw_text_lower, kw) for kw in combined_terms):
                    reg_source = (
                        "FAO/WHO JEMRA Part 3 (Precautionary Allergen Labelling)"
                        if is_prec_text else
                        "US FDA FALCPA & FAO/WHO JEMRA 2021 Part 1"
                    )
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
                        "codex": "Codex General Standard for the Labelling of Prepackaged Foods (CXS 1-1985)"
                    }
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
                "regulation": "FAO/WHO JEMRA 2021 Part 1 & US FDA FAQ",
                "codex": "Codex CXS 1-1985"
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
        reg = evaluate_regulatory_compliance(entity, persona, body_weight)
        regulatory_assessments.append(reg)
        if reg.get("efsa"):
            efsa_warnings.append(reg["efsa"])

    if critical_alerts:
        return {
            "status": "CRITICAL_ALERT",
            "badge_color": "red",
            "title": "🔴 CẢNH BÁO MỨC CAO — CẦN XÁC MINH VỚI CHUYÊN GIA",
            "alerts": critical_alerts,
            "cautions": caution_alerts,
            "advice": (
                "Có quy tắc mức cao khớp với dữ liệu nhãn. Hệ thống không thể xác nhận "
                "nguy cơ cá nhân; hãy kiểm tra nhãn và trao đổi với chuyên gia y tế trước "
                "khi quyết định sử dụng."
            ),
            "citations": list(set([a["regulation"] for a in critical_alerts] + [a["codex"] for a in critical_alerts])),
            **confidence_metrics,
            "regulatory": regulatory_assessments,
            "efsa_warnings": efsa_warnings,
        }
    elif warning_alerts:
        return {
            "status": "WARNING",
            "badge_color": "orange",
            "title": "🟡 CẢNH BÁO — CÓ THUẬT NGỮ CẦN KIỂM TRA",
            "alerts": warning_alerts,
            "cautions": caution_alerts,
            "advice": (
                "OCR có thể khớp một thuật ngữ cần kiểm tra. Kết quả này không xác nhận "
                "thành phần, dị ứng cá nhân hay mức nguy hiểm; hãy kiểm tra nhãn đầy đủ "
                "và hỏi chuyên gia y tế nếu cần."
            ),
            "citations": list(set([a["regulation"] for a in warning_alerts] + [a["codex"] for a in warning_alerts])),
            **confidence_metrics,
            "regulatory": regulatory_assessments,
            "efsa_warnings": efsa_warnings,
        }
    elif caution_alerts:
        return {
            "status": "CAUTION",
            "badge_color": "yellow",
            "title": "🟡 LƯU Ý — CẦN THÊM THÔNG TIN VỀ LIỀU LƯỢNG",
            "alerts": caution_alerts,
            "cautions": [],
            "advice": (
                "Có thông tin cần lưu ý nhưng dữ liệu nhãn không đủ để xác định mức phơi nhiễm "
                "hoặc đưa ra khuyến nghị cá nhân."
            ),
            "citations": list(set([a["regulation"] for a in caution_alerts])),
            **confidence_metrics,
            "regulatory": regulatory_assessments,
            "efsa_warnings": efsa_warnings,
        }
    else:
        if insufficient_evidence:
            return {
                "status": "ABSTAIN",
                "badge_color": "gray",
                "title": "⚪ TỪ CHỐI TƯ VẤN (KHÔNG ĐỦ CĂN CỨ)",
                "message": (
                    "Độ tin cậy OCR hoặc độ phù hợp của kết quả truy xuất dưới ngưỡng "
                    "an toàn (S_total < 0.65). Hệ thống không kết luận an toàn khi thiếu căn cứ."
                ),
                "alerts": [],
                "cautions": [],
                "citations": [],
                **confidence_metrics,
                "regulatory": regulatory_assessments,
                "efsa_warnings": efsa_warnings,
            }
        return {
            "status": "SCREENED_NO_ALERT",
            "badge_color": "blue",
            "title": "⚪ CHƯA PHÁT HIỆN CẢNH BÁO TRONG DỮ LIỆU ĐÃ SÀNG LỌC",
            "alerts": [],
            "cautions": [],
            "benefits": matched_benefits,
            "advice": (
                "Không tìm thấy cảnh báo phù hợp trong các thành phần đã nhận diện. Điều này "
                "không chứng minh sản phẩm an toàn, không xác nhận toàn bộ nhãn và không phải "
                "kết luận tuân thủ pháp luật; hãy kiểm tra bao bì đầy đủ."
            ),
            "citations": [],
            **confidence_metrics,
            "regulatory": regulatory_assessments,
            "efsa_warnings": efsa_warnings,
        }


_LLM_ANSWER_PROMPT = """Bạn là chuyên gia tư vấn an toàn thực phẩm Việt Nam. Dựa trên bằng chứng khoa học bên dưới, hãy trả lời câu hỏi của người dùng bằng tiếng Việt.

## Yêu cầu:
- Trả lời ngắn gọn, rõ ràng, dễ hiểu cho người tiêu dùng.
- Mỗi câu phải kết thúc bằng trích dẫn nguồn có trong danh sách nguồn được cung cấp.
- Chỉ dùng chính xác tên nguồn trong danh sách; không tự tạo nguồn, năm xuất bản, URL hoặc mã trích dẫn.
- Nếu có chống chỉ định hoặc dị ứng, cảnh báo rõ ràng và khuyên tham vấn bác sĩ.
- Không bịa thêm thông tin ngoài bằng chứng được cung cấp.
- Nếu không có bằng chứng liên quan để trả lời, hãy nói rõ chưa đủ dữ liệu và không suy đoán.
- Không lặp lại nguyên văn bằng chứng, hãy tổng hợp thành câu trả lời mạch lạc.

## Hồ sơ người dùng:
{persona_info}

## Câu hỏi:
{question}

## Trạng thái đánh giá: {status}

## Bằng chứng thu thập được:
{evidence}

## Câu trả lời tư vấn:
"""


def generate_llm_answer(evaluation, linked_results, persona, question,
                        detected_hazards=None, matched_fndds=None,
                        nutri_warnings=None):
    """
    Gọi Gemini Flash sinh câu trả lời tự nhiên có trích dẫn từ bằng chứng RAG.
    Trả về string câu trả lời, hoặc None nếu không có API key / lỗi.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    conditions = ", ".join(persona.get("conditions", [])) or "Không có"
    allergies = ", ".join(persona.get("allergies", [])) or "Không có"
    persona_info = f"Bệnh lý: {conditions}. Dị ứng: {allergies}."

    evidence_parts = []

    evidence_parts.append(f"### Kết quả đánh giá: {evaluation.get('title', '')}")
    evidence_parts.append(f"Lời khuyên hệ thống: {evaluation.get('advice', evaluation.get('message', ''))}")

    alerts = evaluation.get("alerts", [])
    if alerts:
        evidence_parts.append("### Cảnh báo rủi ro:")
        for a in alerts:
            evidence_parts.append(f"- {a['additive']}: {a['message']} [Nguồn: {a.get('regulation', '')}, {a.get('codex', '')}]")

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

    if detected_hazards:
        evidence_parts.append("### Lịch sử sự cố liên quan:")
        for h in detected_hazards[:3]:
            evidence_parts.append(f"- [{h.get('severity', '')}] {h.get('title', '')} — Tác nhân: {h.get('hazard', '')}")

    if matched_fndds:
        evidence_parts.append(f"### Dinh dưỡng (USDA FNDDS): {matched_fndds.get('food_description', '')}")
        for key in ["calories_kcal", "sodium_mg", "sugars_g", "fat_g", "protein_g"]:
            val = matched_fndds.get(key)
            if val:
                evidence_parts.append(f"- {key}: {val}")
    if nutri_warnings:
        for nw in nutri_warnings:
            evidence_parts.append(f"- Cảnh báo dinh dưỡng: {nw}")

    citations = evaluation.get("citations", [])
    if citations:
        evidence_parts.append("### Nguồn trích dẫn:")
        for c in citations:
            evidence_parts.append(f"- {c}")

    evidence_text = "\n".join(evidence_parts)

    prompt = _LLM_ANSWER_PROMPT.format(
        persona_info=persona_info,
        question=question or "Sản phẩm này có an toàn cho tôi không?",
        status=evaluation.get("status", ""),
        evidence=evidence_text,
    )

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config={"temperature": 0.3, "max_output_tokens": 4096},
        )
        result = (response.text or "").strip()
        if result and len(result) > 20:
            citation_check = validate_citations(
                result,
                citations,
                require_each_sentence=False,
            )
            if citation_check["valid"]:
                return result
            _LOGGER.warning(
                "Citation check did not pass (returning answer anyway): %s",
                citation_check,
            )
            return result
        _LOGGER.warning("Gemini returned an empty or too-short answer.")
    except Exception:
        _LOGGER.exception("Gemini answer generation failed.")
        pass
    return None


_LLM_ONLY_PROMPT = """Bạn là chuyên gia tư vấn an toàn thực phẩm.
Người dùng cung cấp danh sách thành phần đọc từ nhãn bao bì thực phẩm.
Hãy trả lời câu hỏi của họ DỰA TRÊN KIẾN THỨC CỦA BẠN (không có cơ sở dữ liệu tra cứu).

## Hồ sơ người dùng:
{persona_info}

## Thành phần trên nhãn:
{ingredients_text}

## Câu hỏi:
{question}

## Trả lời (tiếng Việt, ngắn gọn, thực tế):
"""


def generate_llm_only_answer(raw_text, persona, question):
    """
    Cấu hình LLM-only: Gemini trả lời trực tiếp từ OCR text, không tra cứu RAG.
    Dùng cho ablation study — so sánh với pipeline có RAG.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    conditions = ", ".join(persona.get("conditions", [])) or "Không có"
    allergies = ", ".join(persona.get("allergies", [])) or "Không có"
    persona_info = f"Bệnh lý: {conditions}. Dị ứng: {allergies}."

    prompt = _LLM_ONLY_PROMPT.format(
        persona_info=persona_info,
        ingredients_text=raw_text or "(không có)",
        question=question or "Sản phẩm này có an toàn cho tôi không?",
    )

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config={"temperature": 0.3, "max_output_tokens": 4096},
        )
        result = (response.text or "").strip()
        if result and len(result) > 20:
            return result
    except Exception:
        pass
    return None
