"""Mô-đun xây dựng và thẩm định Hồ sơ Sức khỏe Người dùng (User Persona Builder).

Hỗ trợ Chế độ Lai (Hybrid):
1. Hồ sơ mẫu chuẩn lâm sàng (NHANES Personas).
2. Tùy chỉnh hồ sơ cá nhân (Custom Persona):
   - Chọn nhiều bệnh nền kết hợp (Multi-select)
   - Nhập dị nguyên tự do kèm cơ chế phòng vệ chống nhập sai (triệu chứng, thuốc, từ phủ định)
   - Thẩm định và ánh xạ thực thể dị nguyên sang Bản thể học FoodOn (29.255 nút)
   - Thu thập thông tin bổ sung: tuổi, giới tính, thai kỳ, cho con bú
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from modules import foodon as m_foodon

# ── 1. Ánh xạ Bệnh nền & Tình trạng sức khỏe ──

CONDITION_CHOICES = {
    "Đái tháo đường Típ 2 (Cần kiểm soát Đường & Calo)": {
        "key": "diabetes",
        "desc": "Cần kiểm soát nghiêm ngặt đường tự do, tinh bột nhanh và giám sát chất tạo ngọt.",
    },
    "Tăng huyết áp (Cần kiểm soát Natri / Muối)": {
        "key": "hypertension",
        "desc": "Cần giới hạn natri dưới 1.500 mg/ngày, hạn chế muối ăn và bột ngọt (MSG).",
    },
    "Bệnh thận mạn tính CKD (Cần kiểm soát Photphat vô cơ & Đạm)": {
        "key": "chronic_kidney_disease_stage_3",
        "extra_keys": ["ckd"],
        "desc": "Chống chỉ định photphat vô cơ (INS 450, 451, 452) và hạn chế kali/natri cao.",
    },
    "Bệnh nhân Phenylketon niệu PKU (Chống chỉ định Aspartame)": {
        "key": "phenylketonuria_pku",
        "extra_keys": ["pku"],
        "desc": "Rối loạn chuyển hóa bẩm sinh, cấm tuyệt đối nguồn Phenylalanine (Aspartame INS 951).",
    },
    "Phụ nữ mang thai (Thai kỳ 3 tháng đầu/giữa/cuối)": {
        "key": "pregnancy",
        "desc": "Ưu tiên thực phẩm sạch; hạn chế caffeine > 200mg/ngày, tránh TiO2 (INS 171) và nitrit.",
    },
    "Trẻ nhỏ dưới 6 tuổi (Cơ địa hiếu động / Cần tránh phẩm màu Azo)": {
        "key": "children_adhd",
        "extra_keys": ["children", "infants_young_children"],
        "desc": "Cần tránh 6 loại phẩm màu Azo Southampton (INS 102, 110, 122, 124, 129, 104) và chất bảo quản hóa học.",
    },
    "Không dung nạp Lactose (Tránh đường sữa bò)": {
        "key": "lactose_intolerance",
        "extra_keys": ["lactose_allergy"],
        "desc": "Thiếu hụt men lactase, không tiêu hóa được đường lactose trong sữa động vật chưa thủy phân.",
    },
    "Bệnh Gout / Tăng acid uric máu": {
        "key": "gout",
        "desc": "Cần hạn chế thực phẩm giàu purin, phủ tạng động vật và chiết xuất nấm men đậm đặc.",
    },
    "Rối loạn mỡ máu / Tim mạch / Béo phì": {
        "key": "obesity",
        "extra_keys": ["cardiovascular_risk"],
        "desc": "Cần kiểm soát chất béo bão hòa, chất béo chuyển hóa (trans fat) và cholesterol.",
    },
    "Viêm loét dạ dày - tá tràng": {
        "key": "peptic_ulcer",
        "desc": "Tránh các chất kích ứng niêm mạc dạ dày, acid hữu cơ nồng độ cao và gia vị cay nóng.",
    },
    "Hen suyễn (Nhạy cảm với phụ gia Sunfit)": {
        "key": "asthma",
        "desc": "Nhạy cảm cao với các chất bảo quản gốc sunfit (INS 220–228), nguy cơ co thắt phế quản.",
    },
}

# ── 2. Danh mục Dị nguyên Phổ biến (Quick Select) ──

COMMON_ALLERGEN_CHOICES = {
    "Đậu phộng (Lạc)": {"key": "peanut", "curie": "FOODON:03420115"},
    "Sữa bò & Chế phẩm sữa": {"key": "milk", "curie": "FOODON:03420113"},
    "Trứng gà": {"key": "egg", "curie": "FOODON:03420114"},
    "Hải sản giáp xác (Tôm, Cua)": {"key": "shellfish", "curie": "FOODON:03411440"},
    "Đậu nành (Đậu tương)": {"key": "soy", "curie": "FOODON:03420116"},
    "Lúa mì (Gluten)": {"key": "gluten", "curie": "FOODON:03411347"},
    "Mè (Vừng)": {"key": "sesame", "curie": "FOODON:03411358"},
    "Cá biển": {"key": "fish", "curie": "FOODON:03411438"},
    "Hạt cây (Hạt điều, Hạt óc chó, Hạnh nhân)": {"key": "nuts", "curie": "FOODON:03411352"},
}

# ── 3. Bộ từ điển Từ chối / Triệu chứng / Thuốc (Phòng vệ chống nhập sai) ──

_NEGATION_WORDS = {
    "không", "khong", "ko", "k", "none", "no", "không có", "khong co", "ko co",
    "không bị", "khong bi", "chưa phát hiện", "chua phat hien", "bình thường", "binh thuong",
    "khỏe", "khoe", "không dị ứng", "khong di ung"
}

_SYMPTOM_WORDS = {
    "đau đầu": "Đau đầu là triệu chứng thần kinh, không phải thực phẩm gây dị ứng.",
    "dau dau": "Đau đầu là triệu chứng thần kinh, không phải thực phẩm gây dị ứng.",
    "mẩn ngứa": "Mẩn ngứa là triệu chứng ngoài da, không phải tên thực phẩm dị nguyên.",
    "man ngua": "Mẩn ngứa là triệu chứng ngoài da, không phải tên thực phẩm dị nguyên.",
    "nổi mề đay": "Mề đay là triệu chứng lâm sàng, không phải tên thực phẩm dị nguyên.",
    "noi me day": "Mề đay là triệu chứng lâm sàng, không phải tên thực phẩm dị nguyên.",
    "đau bụng": "Đau bụng là triệu chứng tiêu hóa, không phải tên thực phẩm dị nguyên.",
    "dau bung": "Đau bụng là triệu chứng tiêu hóa, không phải tên thực phẩm dị nguyên.",
    "tiêu chảy": "Tiêu chảy là triệu chứng tiêu hóa, không phải tên thực phẩm dị nguyên.",
    "tieu chay": "Tiêu chảy là triệu chứng tiêu hóa, không phải tên thực phẩm dị nguyên.",
    "khó thở": "Khó thở là triệu chứng hô hấp, không phải tên thực phẩm dị nguyên.",
    "kho tho": "Khó thở là triệu chứng hô hấp, không phải tên thực phẩm dị nguyên.",
    "chóng mặt": "Chóng mặt là triệu chứng cơ thể, không phải tên thực phẩm dị nguyên.",
    "chong mat": "Chóng mặt là triệu chứng cơ thể, không phải tên thực phẩm dị nguyên.",
    "buồn nôn": "Buồn nôn là triệu chứng cơ thể, không phải tên thực phẩm dị nguyên.",
    "buon non": "Buồn nôn là triệu chứng cơ thể, không phải tên thực phẩm dị nguyên.",
    "sốt": "Sốt là triệu chứng cơ thể, không phải tên thực phẩm dị nguyên.",
    "sot": "Sốt là triệu chứng cơ thể, không phải tên thực phẩm dị nguyên.",
    "mệt mỏi": "Mệt mỏi là triệu chứng cơ thể, không phải tên thực phẩm dị nguyên.",
    "met moi": "Mệt mỏi là triệu chứng cơ thể, không phải tên thực phẩm dị nguyên.",
}

_DRUG_WORDS = {
    "penicillin": "Penicillin là thuốc kháng sinh dược phẩm, không phải thực phẩm.",
    "aspirin": "Aspirin là thuốc dược phẩm kháng viêm, không phải thực phẩm.",
    "paracetamol": "Paracetamol là thuốc dược phẩm hạ sốt giảm đau, không phải thực phẩm.",
    "panadol": "Panadol là biệt dược thuốc / dược phẩm Paracetamol, không phải thực phẩm.",
    "efferalgan": "Efferalgan là biệt dược thuốc / dược phẩm Paracetamol, không phải thực phẩm.",
    "kháng sinh": "Kháng sinh là thuốc / dược phẩm y tế, không thuộc phạm vi thực phẩm.",
    "khang sinh": "Kháng sinh là thuốc / dược phẩm y tế, không thuộc phạm vi thực phẩm.",
    "amoxicillin": "Amoxicillin là thuốc kháng sinh dược phẩm.",
    "ibuprofen": "Ibuprofen là thuốc dược phẩm kháng viêm phi steroid (NSAID).",
    "thuốc": "Thuốc men / Dược phẩm không thuộc phạm vi nhãn thực phẩm.",
    "thuoc": "Thuốc men / Dược phẩm không thuộc phạm vi nhãn thực phẩm.",
}

_VIETNAMESE_ALLERGEN_MAP = {
    "dau phong": ("peanut", "FOODON:03420115", "Đậu phộng (Lạc)"),
    "lac": ("peanut", "FOODON:03420115", "Lạc / Đậu phộng"),
    "tom": ("shellfish", "FOODON:03411440", "Tôm"),
    "cua": ("shellfish", "FOODON:03411446", "Cua"),
    "hai san": ("shellfish", "FOODON:03411440", "Hải sản giáp xác"),
    "sua": ("milk", "FOODON:03420113", "Sữa động vật"),
    "sua bo": ("milk", "FOODON:03420113", "Sữa bò"),
    "trung": ("egg", "FOODON:03420114", "Trứng gia cầm"),
    "trung ga": ("egg", "FOODON:03420114", "Trứng gà"),
    "dau nanh": ("soy", "FOODON:03420116", "Đậu nành"),
    "dau tuong": ("soy", "FOODON:03420116", "Đậu tương"),
    "lua mi": ("gluten", "FOODON:03411347", "Lúa mì"),
    "gluten": ("gluten", "FOODON:03411347", "Gluten"),
    "me": ("sesame", "FOODON:03411358", "Mè (Vừng)"),
    "vung": ("sesame", "FOODON:03411358", "Vừng"),
    "ca": ("fish", "FOODON:03411438", "Cá"),
    "ca bien": ("fish", "FOODON:03411438", "Cá biển"),
    "hat dieu": ("nuts", "FOODON:03411352", "Hạt điều"),
    "hat oc cho": ("nuts", "FOODON:03301364", "Hạt óc chó"),
    "oc cho": ("nuts", "FOODON:03301364", "Hạt óc chó"),
    "hanh nhan": ("nuts", "FOODON:03411351", "Hạnh nhân"),
    "hat de": ("nuts", "FOODON:03411354", "Hạt dẻ"),
    "hat phi": ("nuts", "FOODON:03411356", "Hạt phỉ"),
    "hat mac ca": ("nuts", "FOODON:03301365", "Hạt mắc ca"),
    "kiwi": ("kiwi", "FOODON:03414363", "Kiwi"),
    "kieu mach": ("buckwheat", "FOODON:03411348", "Kiều mạch"),
    "mu tat": ("mustard", "FOODON:03411357", "Mù tạt"),
    "can tay": ("celery", "FOODON:03411353", "Cần tây"),
    "sunfit": ("sulfites", "FOODON:03411359", "Sunfit"),
    "sulfite": ("sulfites", "FOODON:03411359", "Sulfite"),
}


def _strip_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để đối sánh mềm."""
    text = unicodedata.normalize("NFKD", str(text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("đ", "d")
    return re.sub(r"\s+", " ", text).strip()


def resolve_custom_allergen_term(raw_term: str, foodon_catalog=None) -> Dict[str, Any]:
    """
    Thẩm định và chuẩn hóa một từ dị ứng người dùng nhập.
    Áp dụng cơ chế phòng vệ 3 lớp:
    1. Lọc từ phủ định -> status: 'ignored'
    2. Lọc triệu chứng -> status: 'symptom_warning'
    3. Lọc thuốc/dược phẩm -> status: 'drug_warning'
    4. Thẩm định qua FoodOn DAG và từ điển tiếng Việt -> status: 'valid' (kèm FoodOn CURIE)
    """
    term_clean = str(raw_term or "").strip()
    if not term_clean or len(term_clean) < 2:
        return {"status": "empty"}

    term_norm = _strip_accents(term_clean)

    # 1. Kiểm tra từ phủ định
    if term_norm in _NEGATION_WORDS:
        return {"status": "ignored", "reason": "Từ phủ định hoặc không có dị ứng."}

    # 2. Kiểm tra triệu chứng cơ thể
    for sym_key, sym_msg in _SYMPTOM_WORDS.items():
        if sym_key in term_clean.lower() or sym_key in term_norm:
            return {
                "status": "symptom_warning",
                "term": term_clean,
                "warning": sym_msg,
            }

    # 3. Kiểm tra thuốc / dược phẩm
    for drug_key, drug_msg in _DRUG_WORDS.items():
        if drug_key in term_clean.lower() or drug_key in term_norm:
            return {
                "status": "drug_warning",
                "term": term_clean,
                "warning": drug_msg,
            }

    # 4. Kiểm tra từ điển dị nguyên tiếng Việt
    if term_norm in _VIETNAMESE_ALLERGEN_MAP:
        key, curie, label = _VIETNAMESE_ALLERGEN_MAP[term_norm]
        return {
            "status": "valid",
            "term": term_clean,
            "key": key,
            "curie": curie,
            "label": label,
            "source": "vietnamese_allergen_map",
        }

    # 5. Tra cứu trực tiếp vào Bản thể học FoodOn (29.255 thực thể)
    if foodon_catalog:
        exact = foodon_catalog.get(term_clean) or foodon_catalog.get(term_norm)
        if exact:
            return {
                "status": "valid",
                "term": term_clean,
                "key": term_norm,
                "curie": exact.get("curie", "FOODON:00000000"),
                "label": exact.get("label", term_clean),
                "source": "foodon_exact",
            }

        matches = m_foodon.search_foodon_entities(term_clean, foodon_catalog, limit=1)
        if matches:
            best = matches[0]
            return {
                "status": "valid",
                "term": term_clean,
                "key": term_norm,
                "curie": best.get("curie", "FOODON:00000000"),
                "label": best.get("label", term_clean),
                "source": "foodon_search",
            }

    # Nếu không nằm trong từ điển/FoodOn nhưng là chữ hợp lệ (ví dụ tên hạt lạ):
    # Vẫn cho phép kiểm tra dạng text, không chặn người dùng
    return {
        "status": "valid_raw",
        "term": term_clean,
        "key": term_norm,
        "curie": "FOODON:00000000",
        "label": term_clean,
        "source": "raw_keyword",
    }


def build_custom_persona(
    selected_conditions: List[str],
    selected_common_allergens: List[str],
    custom_allergen_terms: List[Dict[str, Any]],
    age: int = 30,
    gender: str = "Nam",
    is_pregnant: bool = False,
    is_lactating: bool = False,
) -> Dict[str, Any]:
    """Đóng gói thông tin người dùng tự nhập thành một Persona hợp lệ theo schema hệ thống."""
    conditions = []
    condition_names = []

    for label in selected_conditions:
        cfg = CONDITION_CHOICES.get(label)
        if cfg:
            conditions.append(cfg["key"])
            for extra in cfg.get("extra_keys", []):
                if extra not in conditions:
                    conditions.append(extra)
            condition_names.append(label.split("(")[0].strip())

    if is_pregnant and "pregnancy" not in conditions:
        conditions.append("pregnancy")
        condition_names.append("Mang thai")

    if is_lactating and "lactating" not in conditions:
        conditions.append("lactating")
        condition_names.append("Đang cho con bú")

    if not conditions:
        conditions = ["healthy"]
        condition_names = ["Không bệnh nền"]

    allergies = []
    allergy_names = []
    foodon_curies = []

    for label in selected_common_allergens:
        cfg = COMMON_ALLERGEN_CHOICES.get(label)
        if cfg:
            if cfg["key"] not in allergies:
                allergies.append(cfg["key"])
            allergy_names.append(label.split("(")[0].strip())
            if cfg.get("curie"):
                foodon_curies.append(cfg["curie"])

    for item in custom_allergen_terms:
        if item.get("status") in ("valid", "valid_raw"):
            key = item.get("key") or item.get("term")
            if key and key not in allergies:
                allergies.append(key)
            lbl = item.get("label") or item.get("term")
            if lbl and lbl not in allergy_names:
                allergy_names.append(lbl)
            if item.get("curie") and item.get("curie") != "FOODON:00000000":
                foodon_curies.append(item["curie"])

    name_parts = []
    if condition_names and condition_names != ["Không bệnh nền"]:
        name_parts.append(" + ".join(condition_names[:3]))
    if allergy_names:
        name_parts.append(f"Dị ứng: {', '.join(allergy_names[:3])}")
    if not name_parts:
        name_parts = ["Người dùng tự thiết lập"]

    summary_name = " | ".join(name_parts)
    desc = (
        f"{age} tuổi ({gender}). "
        f"Bệnh nền: {', '.join(condition_names)}. "
        f"Dị ứng: {', '.join(allergy_names) if allergy_names else 'Không phát hiện'}. "
        f"{'Đang mang thai. ' if is_pregnant else ''}"
        f"{'Đang cho con bú. ' if is_lactating else ''}"
    )

    return {
        "id": "custom_user_persona",
        "name": f"👤 {summary_name}",
        "description": desc,
        "conditions": conditions,
        "allergies": allergies,
        "foodon_curies": foodon_curies,
        "special_diets": [],
        "dietary_habits": [],
        "demographics": {
            "age": age,
            "gender": gender.lower(),
            "pregnant": is_pregnant,
            "lactating": is_lactating,
        },
        "is_custom": True,
    }


def render_persona_selector_ui(
    key_prefix: str,
    personas_list: Optional[list] = None,
    foodon_catalog=None,
) -> Dict[str, Any]:
    """
    Giao diện Tự chỉnh Hồ sơ Sức khỏe Người dùng (Custom Persona Builder).
    Áp dụng kho CSDL dịch tễ học NHANES và Tầng 2 FoodOntoRAG (FoodOn 29.255 nút).
    """
    import html as _html

    with st.expander("👤 Thiết lập Hồ sơ Sức khỏe của bạn (Bệnh nền & Dị ứng)", expanded=True):
        # 1. Bệnh nền (Multi-select)
        selected_conds = st.multiselect(
            "🏥 Chọn bệnh nền & tình trạng sức khỏe (có thể chọn nhiều):",
            list(CONDITION_CHOICES.keys()),
            default=[],
            help="Tích chọn một hoặc nhiều bệnh lý bạn cần kiểm soát (đái tháo đường, tăng huyết áp, suy thận...).",
            key=f"{key_prefix}_custom_conds",
        )

        # 2. Dị nguyên thực phẩm
        st.markdown("**🥜 Dị ứng Thực phẩm & Dị nguyên (bỏ qua nếu không bị dị ứng):**")
        selected_common = st.multiselect(
            "Chọn nhanh dị nguyên phổ biến:",
            list(COMMON_ALLERGEN_CHOICES.keys()),
            default=[],
            help="9 nhóm dị nguyên ưu tiên toàn cầu (FALCPA & FAO/WHO JEMRA).",
            key=f"{key_prefix}_custom_common_allergens",
        )

        custom_allergen_text = st.text_input(
            "Hoặc gõ món dị ứng khác (cách nhau bởi dấu phẩy, để trống nếu không có):",
            value="",
            placeholder="Ví dụ: hạt óc chó, kiwi, kiều mạch, mù tạt... (để trống nếu không bị)",
            help="Hệ thống tự động dùng Tầng 2 FoodOntoRAG đối chiếu Đồ thị FoodOn 29.255 nút để chuẩn hóa.",
            key=f"{key_prefix}_custom_allergen_text",
        )

        # Phân tích & phòng vệ chống nhập sai (Triệu chứng, Thuốc, Từ phủ định)
        custom_terms_info = []
        if custom_allergen_text.strip():
            raw_tokens = [t.strip() for t in re.split(r"[,;]+", custom_allergen_text) if t.strip()]
            for raw_tok in raw_tokens:
                res = resolve_custom_allergen_term(raw_tok, foodon_catalog=foodon_catalog)
                if res["status"] == "ignored":
                    st.caption(f"ℹ️ Bỏ qua từ phủ định: *'{raw_tok}'* (xác nhận không có dị ứng).")
                elif res["status"] == "symptom_warning":
                    st.warning(f"⚠️ **'{res['term']}'**: {res['warning']} Hệ thống sẽ bỏ qua mục này.")
                elif res["status"] == "drug_warning":
                    st.warning(f"⚠️ **'{res['term']}'**: {res['warning']} Hệ thống chỉ phân tích thực phẩm, vui lòng hỏi bác sĩ về thuốc.")
                elif res["status"] == "valid":
                    st.success(f"✅ Đã nhận diện dị nguyên qua FoodOn: **{res['term']}** ➔ `{res.get('curie', '')}` ({res.get('label', '')})")
                    custom_terms_info.append(res)
                elif res["status"] == "valid_raw":
                    st.info(f"ℹ️ Đã ghi nhận từ khóa dị nguyên: **{res['term']}**")
                    custom_terms_info.append(res)

        # 3. Thông tin nhân khẩu học (Tuổi & Giới tính - xếp song song 50/50 cân đối)
        c_age, c_gen = st.columns(2)
        with c_age:
            user_age = st.number_input(
                "Tuổi:",
                min_value=1,
                max_value=120,
                value=30,
                step=1,
                key=f"{key_prefix}_custom_age",
            )
        with c_gen:
            user_gender = st.selectbox(
                "Giới tính:",
                ["Nam", "Nữ", "Khác"],
                key=f"{key_prefix}_custom_gender",
            )

        # 4. Tình trạng đặc biệt (Thai kỳ & Nuôi con - 2 cột ngang cân đối)
        st.markdown("**Tình trạng thai kỳ & nuôi con (nếu có):**")
        c_preg, c_lact = st.columns(2)
        with c_preg:
            is_preg = st.checkbox("🤰 Đang mang thai", key=f"{key_prefix}_custom_preg")
        with c_lact:
            is_lact = st.checkbox("🤱 Đang cho con bú", key=f"{key_prefix}_custom_lact")

        custom_p = build_custom_persona(
            selected_conditions=selected_conds,
            selected_common_allergens=selected_common,
            custom_allergen_terms=custom_terms_info,
            age=int(user_age),
            gender=user_gender,
            is_pregnant=is_preg,
            is_lactating=is_lact,
        )

        st.markdown(
            f'<div class="fh-persona"><div class="top"><span class="ttl"><span class="ic">✨</span>Hồ sơ tùy chỉnh của bạn</span>'
            f'<span class="nm">{_html.escape(custom_p["name"])}</span></div>'
            f'<p>💡 {_html.escape(custom_p["description"])}</p>'
            f'<div style="font-size:12px;color:#1e40af;margin-top:6px;background:#eff6ff;padding:4px 8px;border-radius:4px;border:1px solid #bfdbfe;">'
            f'🔬 <b>Ứng dụng CSDL NHANES:</b> Hệ thống tự động đối chiếu độ tuổi & bệnh nền với kho dịch tễ học NHANES (P_DEMO, P_GHB, P_BPXO, P_DBQ) để lấy ngưỡng cận lâm sàng chuẩn (HbA1c, Huyết áp, Natri).'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        return custom_p
