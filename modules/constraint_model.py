"""
Module Constraint Modeling cho hệ thống FoodHealthQA
Triển khai thuật toán từ pFoodReQ (Personalized Food-health Question answering):

1. Hàm chỉ thị khoảng (Indicator Function):
   I_g(v) = 1 nếu v ∈ range(g); ngược lại I_g(v) = 0

2. Xử lý Ràng buộc Phủ định (Negation Constraint Modeling):
   Gán nhãn loại ràng buộc (Constraint Type Markup) để phân biệt:
   - PERMITTED: thành phần được phép
   - RESTRICTED: thành phần bị giới hạn liều lượng
   - CONTRAINDICATED: thành phần chống chỉ định tuyệt đối
   - ALLERGEN_RISK: thành phần gây dị ứng

3. Ràng buộc Số học (Numerical Constraint):
   Kiểm tra giá trị dinh dưỡng với ngưỡng bệnh lý theo hồ sơ NHANES.
"""

import json
import os
import re


def _load_allergen_groups():
    """Load source-linked allergen label categories enabled for screening."""
    kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base", "allergen_groups.json")
    if not os.path.isfile(kb_path):
        return [], {}, {}
    with open(kb_path, "r", encoding="utf-8") as f:
        groups = [
            group for group in json.load(f)
            if group.get("enabled_for_allergen_screening", True)
        ]
    keywords = {g["id"]: g["keywords"] for g in groups}
    negation = {}
    for g in groups:
        key = f"{g['id']}_allergy"
        negation[key] = {
            kw: CONSTRAINT_ALLERGEN_RISK
            for kw, ctype in g.get("negation_rules", {}).items()
        }
    return groups, keywords, negation


def contains_label_term(text, term):
    """Match a complete label term instead of an arbitrary substring."""
    text = str(text or "")
    term = str(term or "").strip()
    if not text or not term:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.IGNORECASE))


# =====================================================================
# 1. HÀM CHỈ THỊ KHOẢNG (INDICATOR FUNCTION) - pFoodReQ
# =====================================================================

class ConstraintRange:
    """
    Biểu diễn một ràng buộc khoảng giá trị dinh dưỡng.
    I_g(v) = 1 nếu v ∈ [lower, upper]; ngược lại I_g(v) = 0
    """

    def __init__(self, name, lower=None, upper=None, unit="", severity="WARNING",
                 message_template=""):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.unit = unit
        self.severity = severity
        self.message_template = message_template

    def evaluate(self, value):
        """
        Áp dụng hàm chỉ thị: I_g(v).
        Trả về (in_range: bool, violation_direction: str|None)
        """
        if value is None:
            return True, None
        if self.upper is not None and value > self.upper:
            return False, "EXCEEDS_UPPER"
        if self.lower is not None and value < self.lower:
            return False, "BELOW_LOWER"
        return True, None

    def get_violation_message(self, value):
        """Sinh thông báo vi phạm nếu giá trị ngoài khoảng cho phép."""
        in_range, direction = self.evaluate(value)
        if in_range:
            return None
        if self.message_template:
            return self.message_template.format(
                value=value, unit=self.unit, upper=self.upper, lower=self.lower, name=self.name
            )
        if direction == "EXCEEDS_UPPER":
            return (f"{self.name}: {value:.1f} {self.unit} vượt ngưỡng an toàn "
                    f"({self.upper:.1f} {self.unit})")
        return (f"{self.name}: {value:.1f} {self.unit} dưới ngưỡng tối thiểu "
                f"({self.lower:.1f} {self.unit})")


# Bảng ràng buộc dinh dưỡng theo bệnh lý (pFoodReQ Constraint Registry)
_NUTRITION_CONSTRAINTS = {
    "hypertension": [
        ConstraintRange(
            "Natri", upper=600.0, unit="mg/100g", severity="WARNING",
            message_template=(
                "Hàm lượng Natri cao ({value:.0f} {unit}) — "
                "Vượt ngưỡng khuyến nghị AHA cho người tăng huyết áp (<1.500 mg/ngày, "
                "tương đương <{upper:.0f} {unit} mỗi khẩu phần). "
                "I_g(Na) = 0: Vi phạm ràng buộc khoảng."
            )
        ),
    ],
    "cardiovascular_risk": [
        ConstraintRange(
            "Natri", upper=700.0, unit="mg/100g", severity="WARNING",
            message_template=(
                "Hàm lượng Natri ({value:.0f} {unit}) — Nguy cơ tim mạch: "
                "cần giảm tổng natri nạp vào. I_g(Na) = 0."
            )
        ),
        ConstraintRange(
            "Chất béo bão hòa", upper=5.0, unit="g/100g", severity="CAUTION",
            message_template=(
                "Chất béo bão hòa ({value:.1f} {unit}) — Khuyến nghị giảm "
                "để bảo vệ mạch vành. I_g(SatFat) = 0."
            )
        ),
    ],
    "chronic_kidney_disease_stage_3": [
        ConstraintRange(
            "Natri", upper=500.0, unit="mg/100g", severity="CRITICAL",
            message_template=(
                "Natri ({value:.0f} {unit}) — NGUY HIỂM cho CKD giai đoạn 3: "
                "gánh nặng lọc thận nghiêm trọng. I_g(Na) = 0."
            )
        ),
        ConstraintRange(
            "Protein", upper=8.0, unit="g/100g", severity="CAUTION",
            message_template=(
                "Protein ({value:.1f} {unit}) — Cần kiểm soát lượng protein "
                "để giảm gánh nặng lọc thận. I_g(Protein) = 0."
            )
        ),
    ],
    "diabetes": [
        ConstraintRange(
            "Đường", upper=10.0, unit="g/100g", severity="WARNING",
            message_template=(
                "Hàm lượng đường cao ({value:.1f} {unit}) — Nguy cơ tăng đường huyết "
                "sau ăn. I_g(Sugar) = 0: Vi phạm ràng buộc khoảng pFoodReQ."
            )
        ),
        ConstraintRange(
            "Calo", upper=300.0, unit="kcal/100g", severity="CAUTION",
            message_template=(
                "Mật độ năng lượng cao ({value:.0f} {unit}). "
                "Cần kiểm soát khẩu phần cho người tiểu đường. I_g(Cal) = 0."
            )
        ),
    ],
    "obesity": [
        ConstraintRange(
            "Calo", upper=250.0, unit="kcal/100g", severity="WARNING",
            message_template=(
                "Mật độ năng lượng ({value:.0f} {unit}) vượt ngưỡng kiểm soát cân nặng "
                "(<{upper:.0f} {unit}). I_g(Cal) = 0."
            )
        ),
        ConstraintRange(
            "Chất béo", upper=15.0, unit="g/100g", severity="CAUTION",
            message_template=(
                "Chất béo ({value:.1f} {unit}) — Cần giảm chất béo tổng. I_g(Fat) = 0."
            )
        ),
    ],
    "gout": [
        ConstraintRange(
            "Protein", upper=10.0, unit="g/100g", severity="CAUTION",
            message_template=(
                "Protein ({value:.1f} {unit}) — Cần kiểm soát lượng purine "
                "cho người bệnh gút. I_g(Protein) = 0."
            )
        ),
    ],
    "peptic_ulcer": [
        ConstraintRange(
            "Natri", upper=800.0, unit="mg/100g", severity="CAUTION",
            message_template=(
                "Natri ({value:.0f} {unit}) — Cần thận trọng cho người viêm loét "
                "dạ dày. I_g(Na) = 0."
            )
        ),
    ],
}

_NUTRIENT_KEY_MAP = {
    "Natri": "sodium_mg",
    "Calo": "calories_kcal",
    "Đường": "sugars_g",
    "Protein": "protein_g",
    "Chất béo": "fat_g",
    "Chất béo bão hòa": "saturated_fat_g",
}


def evaluate_nutritional_constraints(nutrition_data, persona):
    """
    Đánh giá toàn bộ ràng buộc dinh dưỡng pFoodReQ cho một sản phẩm.
    Áp dụng hàm chỉ thị khoảng I_g(v) cho từng chỉ số vs từng bệnh lý.
    Trả về danh sách cảnh báo và bảng chi tiết I_g.
    """
    if not nutrition_data:
        return [], []

    user_conditions = persona.get("conditions", [])
    warnings = []
    constraint_details = []

    for condition in user_conditions:
        cond_lower = condition.lower().replace(" ", "_")
        constraints = _NUTRITION_CONSTRAINTS.get(cond_lower, [])

        for constraint in constraints:
            nutrient_key = _NUTRIENT_KEY_MAP.get(constraint.name)
            if not nutrient_key:
                continue

            value = nutrition_data.get(nutrient_key)
            if value is None:
                continue
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue

            in_range, direction = constraint.evaluate(value)
            indicator = 1 if in_range else 0

            detail = {
                "condition": condition,
                "nutrient": constraint.name,
                "value": value,
                "unit": constraint.unit,
                "upper_bound": constraint.upper,
                "lower_bound": constraint.lower,
                "I_g": indicator,
                "severity": constraint.severity if not in_range else "OK",
            }
            constraint_details.append(detail)

            if not in_range:
                msg = constraint.get_violation_message(value)
                if msg:
                    warnings.append({
                        "message": msg,
                        "severity": constraint.severity,
                        "condition": condition,
                        "nutrient": constraint.name,
                        "I_g": 0,
                    })

    return warnings, constraint_details


# =====================================================================
# 2. XỬ LÝ RÀNG BUỘC PHỦ ĐỊNH (NEGATION CONSTRAINT MODELING)
# =====================================================================

# Constraint Type Labels
CONSTRAINT_PERMITTED = "PERMITTED"
CONSTRAINT_RESTRICTED = "RESTRICTED"
CONSTRAINT_CONTRAINDICATED = "CONTRAINDICATED"
CONSTRAINT_ALLERGEN_RISK = "ALLERGEN_RISK"

# Bảng ánh xạ ràng buộc phủ định: (bệnh lý) -> (thành phần) -> loại ràng buộc
# Quy tắc bệnh lý (không phải dị ứng) — các quy tắc dị ứng được nạp từ allergen_groups.json
def _load_verified_medical_rules():
    """Only source-checked health considerations may create condition matches."""
    kb_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "knowledge_base",
        "additives_standard.json",
    )
    if not os.path.isfile(kb_path):
        return {}
    with open(kb_path, "r", encoding="utf-8") as source:
        additives = json.load(source)
    rules = {}
    for additive in additives:
        for item in additive.get("health_considerations", []):
            if item.get("verification_status") != "source_checked":
                continue
            condition = str(item.get("condition", "")).strip().lower().replace(" ", "_")
            if not condition:
                continue
            keywords = [
                additive.get("ins_code", ""),
                additive.get("e_number", ""),
                additive.get("canonical_name_en", ""),
                *additive.get("synonyms", []),
            ]
            rules.setdefault(condition, {})
            for keyword in keywords:
                if keyword:
                    rules[condition][str(keyword)] = CONSTRAINT_RESTRICTED
    return rules

# Nạp từ knowledge_base/allergen_groups.json → ALLERGEN_KEYWORDS + _ALLERGEN_NEGATION_RULES
_ALLERGEN_GROUPS, ALLERGEN_KEYWORDS, _ALLERGEN_NEGATION_RULES = _load_allergen_groups()

# Gộp tất cả negation rules: bệnh lý + dị ứng
_NEGATION_RULES = {
    **_load_verified_medical_rules(),
    **_ALLERGEN_NEGATION_RULES,
}


def classify_constraint_type(mention_text, additive, persona):
    """
    Gán nhãn loại ràng buộc (Constraint Type Markup) cho thành phần.
    Phân biệt rõ: PERMITTED vs RESTRICTED vs CONTRAINDICATED vs ALLERGEN_RISK.
    """
    mention_lower = mention_text.lower()
    user_conditions = [c.lower().replace(" ", "_") for c in persona.get("conditions", [])]
    user_allergies = [a.lower().replace(" ", "_") for a in persona.get("allergies", [])]

    constraint_type = CONSTRAINT_PERMITTED
    matched_rule = None

    ins_code = ""
    if additive:
        ins_code = additive.get("ins_code", "")

    all_conditions = user_conditions + [f"{a}_allergy" for a in user_allergies]

    for condition in all_conditions:
        rules = _NEGATION_RULES.get(condition, {})
        for keyword, ctype in rules.items():
            if contains_label_term(mention_lower, keyword) or contains_label_term(ins_code, keyword):
                if _severity_rank(ctype) > _severity_rank(constraint_type):
                    constraint_type = ctype
                    matched_rule = (condition, keyword)

    for allergy in user_allergies:
        keywords = ALLERGEN_KEYWORDS.get(allergy, [allergy])
        for kw in keywords:
            if contains_label_term(mention_lower, kw):
                new_type = CONSTRAINT_ALLERGEN_RISK
                if _severity_rank(new_type) > _severity_rank(constraint_type):
                    constraint_type = new_type
                    matched_rule = (f"{allergy}_allergy", kw)

    return {
        "constraint_type": constraint_type,
        "matched_rule": matched_rule,
    }


def _severity_rank(constraint_type):
    """Thứ tự mức nghiêm trọng của constraint type."""
    ranks = {
        CONSTRAINT_PERMITTED: 0,
        CONSTRAINT_RESTRICTED: 1,
        CONSTRAINT_ALLERGEN_RISK: 2,
        CONSTRAINT_CONTRAINDICATED: 3,
    }
    return ranks.get(constraint_type, 0)


def apply_negation_constraints(linked_results, persona):
    """
    Áp dụng Negation Constraint Modeling lên toàn bộ kết quả entity linking.
    Bổ sung trường constraint_type và constraint_detail cho từng thành phần.
    """
    enriched = []
    for item in linked_results:
        constraint_info = classify_constraint_type(
            item["raw_mention"],
            item.get("matched_entity"),
            persona
        )
        enriched_item = dict(item)
        enriched_item["constraint_type"] = constraint_info["constraint_type"]
        enriched_item["constraint_rule"] = constraint_info["matched_rule"]
        enriched.append(enriched_item)
    return enriched


# =====================================================================
# 3. ĐÁNH GIÁ ADI - JECFA / CODEX STAN 192-1995 (GSFA)
# =====================================================================

def parse_adi_value(adi_str):
    """
    Phân tích chuỗi ADI thành giá trị số (mg/kg thể trọng/ngày).
    Ví dụ: "0 - 40 mg/kg thể trọng/ngày" → (0.0, 40.0)
            "0 - 10 mg/kg thể trọng/ngày" → (0.0, 10.0)
            "Chưa xác định" → None
    """
    if not adi_str or "chưa xác định" in adi_str.lower() or "not specified" in adi_str.lower():
        return None
    match = re.search(r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*mg/kg', adi_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(r'(\d+(?:\.\d+)?)\s*mg/kg', adi_str)
    if match:
        return 0.0, float(match.group(1))
    return None


def parse_ml_value(codex_str):
    """
    Phân tích mức giới hạn tối đa (Maximum Level) từ chuỗi Codex.
    Ví dụ: "Codex STAN 192-1995 (ML: 600 - 1000 mg/kg)" → (600.0, 1000.0)
            "Codex STAN 192-1995 (GMP)" → "GMP"
    """
    if not codex_str:
        return None
    if "GMP" in codex_str.upper():
        return "GMP"
    match = re.search(r'ML:\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*mg/kg', codex_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(r'ML:\s*(\d+(?:\.\d+)?)\s*mg/kg', codex_str)
    if match:
        return 0.0, float(match.group(1))
    match = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*mg/kg', codex_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


def evaluate_adi_compliance(additive_entity, body_weight_kg=60.0):
    """
    Tính toán ngưỡng ADI (Acceptable Daily Intake) theo JECFA/EFSA.
    Trả về dict với thông tin tuân thủ ADI dựa trên cân nặng người dùng.
    Công thức: Lượng nạp tối đa/ngày = ADI_upper × body_weight_kg (mg/ngày)
    """
    assessment = additive_entity.get("adi_assessment") or {}
    adi_str = assessment.get("display_text", additive_entity.get("jecfa_adi", ""))
    codex_str = additive_entity.get("codex_standard", "")
    ins_code = additive_entity.get("ins_code", "")
    name = additive_entity.get("canonical_name", "")

    result = {
        "additive": f"{name} ({ins_code})" if ins_code else name,
        "adi_raw": adi_str,
        "adi_parsed": None,
        "max_daily_intake_mg": None,
        "ml_value": None,
        "compliance_status": "UNKNOWN",
        "source": assessment.get("source_id", ""),
        "detail": "",
    }

    if not assessment or assessment.get("verification_status") != "source_checked":
        result["compliance_status"] = "SOURCE_REVIEW_REQUIRED"
        result["detail"] = (
            "Không tính giới hạn ADI: claim chưa được xác minh bằng nguồn chính."
        )
        if assessment:
            result["source_url"] = assessment.get("source_url", "")
        return result

    if not assessment.get("use_for_numeric_compliance"):
        if ins_code == "INS 621":
            result["compliance_status"] = "NOT_NUMERICALLY_APPLICABLE"
            result["detail"] = (
                "ADI glutamate của EFSA là ADI nhóm E620-E625, biểu thị theo glutamic acid; "
                "không thể áp dụng phép tính số riêng cho INS 621."
            )
        else:
            result["compliance_status"] = "NO_NUMERIC_ADI"
            result["detail"] = (
                "Nguồn đã được đối chiếu nhưng không thiết lập ADI dạng số để quy đổi "
                "thành giới hạn dùng hằng ngày."
            )
        result["source_url"] = assessment.get("source_url", "")
        return result

    adi = parse_adi_value(adi_str)
    if adi:
        result["adi_parsed"] = {"lower": adi[0], "upper": adi[1], "unit": "mg/kg bw/day"}
        max_daily = adi[1] * body_weight_kg
        result["max_daily_intake_mg"] = max_daily
        result["compliance_status"] = "SPECIFIED"

        source_parts = []
        if "JECFA" in adi_str.upper():
            source_parts.append("JECFA (FAO/WHO)")
        if "EFSA" in adi_str.upper():
            source_parts.append("EFSA")
        result["source"] = " / ".join(source_parts) if source_parts else "Codex Alimentarius"

        result["detail"] = (
            f"ADI: {adi[0]:.0f}–{adi[1]:.0f} mg/kg bw/ngày → "
            f"Lượng nạp tối đa cho người {body_weight_kg:.0f} kg: {max_daily:.0f} mg/ngày"
        )
    else:
        if assessment:
            result["compliance_status"] = "SOURCE_REVIEW_REQUIRED"
            result["detail"] = (
                "Không có ngưỡng ADI dạng số đã xác minh để tính lượng dùng cá nhân."
            )
            result["source_url"] = assessment.get("source_url", "")
        elif "GMP" in adi_str.upper() or "GMP" in codex_str.upper():
            result["compliance_status"] = "GMP"
            result["source"] = "Codex STAN 192-1995 (GMP)"
            result["detail"] = "Sử dụng theo nguyên tắc GMP — không giới hạn ADI cụ thể"
        else:
            result["compliance_status"] = "NOT_ESTABLISHED"
            result["source"] = "JECFA"
            result["detail"] = "Chưa xác định ngưỡng ADI — cần thận trọng khi sử dụng"

    ml = parse_ml_value(codex_str)
    if ml == "GMP":
        result["ml_value"] = "GMP"
    elif ml:
        result["ml_value"] = {"lower": ml[0], "upper": ml[1], "unit": "mg/kg sản phẩm"}

    return result


# =====================================================================
# 4. CƠ SỞ TRI THỨC EFSA — Nạp từ knowledge_base/additives_standard.json
# =====================================================================

def _load_efsa_opinions():
    """Nạp EFSA opinions từ knowledge_base/additives_standard.json."""
    kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base", "additives_standard.json")
    if not os.path.isfile(kb_path):
        return {}
    with open(kb_path, "r", encoding="utf-8") as f:
        additives = json.load(f)
    opinions = {}
    for a in additives:
        efsa = a.get("efsa_opinion")
        if efsa:
            ins_code = a["ins_code"]
            opinions[ins_code] = {
                "substance": f"{a.get('canonical_name_en', '')} ({a.get('e_number', '')})",
                "year": efsa["year"],
                "adi_efsa": efsa["adi_efsa"],
                "risk_groups": efsa["risk_groups"],
                "key_finding": efsa["key_finding"],
                "reference": efsa["reference"],
                "verification_status": efsa.get(
                    "verification_status",
                    "pending_primary_source_check",
                ),
            }
    return opinions

_EFSA_OPINIONS_LEGACY = {
    "INS 102": {
        "substance": "Tartrazine (E102)",
        "year": 2009,
        "adi_efsa": 7.5,
        "risk_groups": ["children_under_6", "adhd", "asthma"],
        "key_finding": (
            "EFSA Panel on Food Additives (2009): Tái đánh giá cho thấy Tartrazine "
            "và các phẩm màu Azo có thể làm tăng hành vi hiếu động ở trẻ em. "
            "Khuyến nghị giảm ADI từ 10 xuống 7.5 mg/kg bw/ngày."
        ),
        "reference": "EFSA Journal 2009;7(11):1331",
    },
    "INS 951": {
        "substance": "Aspartame (E951)",
        "year": 2023,
        "adi_efsa": 40.0,
        "risk_groups": ["pku", "pregnancy"],
        "key_finding": (
            "EFSA Panel (2013, cập nhật 2023): ADI 40 mg/kg bw/ngày được duy trì. "
            "Chống chỉ định tuyệt đối cho người mắc Phenylketon niệu (PKU). "
            "IARC (2023) phân loại nhóm 2B (possibly carcinogenic) nhưng JECFA "
            "giữ nguyên ADI."
        ),
        "reference": "EFSA Journal 2013;11(12):3496 + WHO/IARC 2023",
    },
    "INS 621": {
        "substance": "Monosodium L-glutamate (E621)",
        "year": 2017,
        "adi_efsa": 30.0,
        "risk_groups": ["hypertension", "cardiovascular_risk"],
        "key_finding": (
            "EFSA ANS Panel (2017): Thiết lập ADI 30 mg/kg bw/ngày cho glutamate "
            "và các muối của nó. Lượng nạp cao có thể gây đau đầu, nhịp tim nhanh. "
            "Chứa 12.3% natri theo khối lượng — cần lưu ý cho bệnh nhân tăng huyết áp."
        ),
        "reference": "EFSA Journal 2017;15(7):4910",
    },
    "INS 211": {
        "substance": "Sodium Benzoate (E211)",
        "year": 2016,
        "adi_efsa": 5.0,
        "risk_groups": ["asthma", "children_under_6"],
        "key_finding": (
            "EFSA Panel (2016): Đánh giá lại Sodium Benzoate xác nhận ADI 5 mg/kg bw/ngày. "
            "Kết hợp với acid ascorbic (vitamin C) có thể tạo thành benzene. "
            "Nguy cơ kích ứng đường hô hấp ở người hen suyễn."
        ),
        "reference": "EFSA Journal 2016;14(3):4441",
    },
    "INS 1420": {
        "substance": "Acetylated Starch (E1420)",
        "year": 2018,
        "adi_efsa": None,
        "risk_groups": ["diabetes"],
        "key_finding": (
            "EFSA Panel (2018): Tinh bột biến tính được đánh giá không cần ADI cụ thể. "
            "Tuy nhiên, với bệnh nhân tiểu đường cần lưu ý chỉ số đường huyết cao hơn "
            "so với tinh bột tự nhiên."
        ),
        "reference": "EFSA Journal 2018;16(3):5180",
    },
    "INS 322i": {
        "substance": "Lecithin (E322i)",
        "year": 2017,
        "adi_efsa": None,
        "risk_groups": ["soy_allergy", "egg_allergy"],
        "key_finding": (
            "EFSA Panel (2017): Lecithin không cần ADI (JECFA: Not Specified). "
            "Tuy nhiên Lecithin chiết xuất từ đậu nành chứa protein đậu nành dư "
            "có thể gây phản ứng dị ứng ở người nhạy cảm."
        ),
        "reference": "EFSA Journal 2017;15(4):4742",
    },
    "INS 500ii": {
        "substance": "Sodium Hydrogen Carbonate (E500ii)",
        "year": 2019,
        "adi_efsa": None,
        "risk_groups": ["hypertension", "chronic_kidney_disease"],
        "key_finding": (
            "Baking soda (NaHCO₃) — mỗi gram chứa 274 mg natri. "
            "Không có ADI riêng (GMP). Cần kiểm soát cho bệnh nhân tăng huyết áp "
            "và suy thận do tải natri."
        ),
        "reference": "EFSA/Codex GMP Guidelines",
    },
    "INS 471": {
        "substance": "Mono- and Diglycerides of Fatty Acids (E471)",
        "year": 2017,
        "adi_efsa": None,
        "risk_groups": [],
        "key_finding": (
            "EFSA Panel (2017): Mono- và diglycerides là sản phẩm tự nhiên của "
            "quá trình tiêu hóa chất béo. Không cần ADI cụ thể. An toàn ở mức GMP."
        ),
        "reference": "EFSA Journal 2017;15(11):5045",
    },
}

_EFSA_OPINIONS_FROM_KB = _load_efsa_opinions()
_EFSA_OPINIONS = {**_EFSA_OPINIONS_LEGACY, **_EFSA_OPINIONS_FROM_KB}


def get_efsa_opinion(ins_code):
    """Tra cứu bằng chứng EFSA cho phụ gia theo mã INS."""
    return _EFSA_OPINIONS.get(ins_code)


def evaluate_efsa_risk(additive_entity, persona):
    """
    Đánh giá rủi ro dựa trên bằng chứng EFSA cho phụ gia với hồ sơ bệnh lý.
    Trả về cảnh báo EFSA nếu phụ gia thuộc nhóm rủi ro cho bệnh lý của persona.
    """
    ins_code = additive_entity.get("ins_code", "")
    opinion = get_efsa_opinion(ins_code)
    if not opinion:
        return None
    if opinion.get("verification_status") != "source_checked":
        return None

    user_conditions = [c.lower().replace(" ", "_") for c in persona.get("conditions", [])]
    user_allergies = [a.lower().replace(" ", "_") for a in persona.get("allergies", [])]
    all_risks = user_conditions + [f"{a}_allergy" for a in user_allergies]

    matched_risks = [r for r in opinion["risk_groups"] if
                     any(r in ur or ur in r for ur in all_risks)]

    if not matched_risks:
        return None

    return {
        "ins_code": ins_code,
        "substance": opinion["substance"],
        "efsa_adi": opinion["adi_efsa"],
        "key_finding": opinion["key_finding"],
        "reference": opinion["reference"],
        "year": opinion["year"],
        "matched_risk_groups": matched_risks,
    }


# =====================================================================
# 5. QUY CHUẨN PHÁP LÝ VIỆT NAM
# =====================================================================

def get_vn_regulation(ins_code):
    """Return an explicit pending state until an official claim is verified."""
    return {
        "ins_code": ins_code,
        "jurisdiction": "Vietnam",
        "status": "SOURCE_REVIEW_REQUIRED",
        "verification_status": "pending_official_text_and_food_category_check",
        "use_for_compliance_decision": False,
        "note": (
            "Chưa gắn văn bản chính thức, phụ lục hiện hành và điều kiện theo nhóm thực phẩm; "
            "không thể kết luận tuân thủ."
        ),
    }


def evaluate_regulatory_compliance(additive_entity, persona=None, body_weight_kg=60.0):
    """
    Đánh giá tuân thủ quy chuẩn tổng hợp: JECFA ADI + EFSA + TT 24/2019/TT-BYT.
    Trả về dict chi tiết về tình trạng pháp lý và an toàn.
    """
    ins_code = additive_entity.get("ins_code", "")
    name = additive_entity.get("canonical_name", "")
    label = f"{name} ({ins_code})" if ins_code else name

    adi_info = evaluate_adi_compliance(additive_entity, body_weight_kg)

    efsa_info = None
    if persona:
        efsa_info = evaluate_efsa_risk(additive_entity, persona)

    vn_info = additive_entity.get("regulatory_claim") or get_vn_regulation(ins_code)
    if vn_info.get("verification_status") != "source_checked":
        vn_info = {
            "ins_code": ins_code,
            "jurisdiction": "Vietnam",
            "status": "SOURCE_REVIEW_REQUIRED",
            "verification_status": vn_info.get(
                "verification_status",
                "pending_official_text_and_food_category_check",
            ),
            "use_for_compliance_decision": False,
            "note": (
                "Tuyên bố pháp lý chưa được xác minh bằng văn bản chính thức và điều kiện "
                "theo nhóm thực phẩm; không thể kết luận tuân thủ."
            ),
        }

    return {
        "additive": label,
        "ins_code": ins_code,
        "adi": adi_info,
        "efsa": efsa_info,
        "vn_regulation": vn_info,
    }
