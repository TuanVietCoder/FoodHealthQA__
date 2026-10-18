"""
Script sinh bộ QA Benchmark bán tự động cho FoodHealthQA.
Đọc từ Knowledge Base (additives, allergens, personas) để tạo câu hỏi-trả lời
có căn cứ nguồn (source-grounded), phân nhóm theo domain.

Quy trình:
1. Script này sinh câu hỏi từ template + dữ liệu KB → file JSONL draft
2. Nhóm review thủ công bằng file CSV (review spreadsheet)
3. Sau review, chạy script finalize để đánh dấu gold_validated

Chạy: python scripts/generate_qa_benchmark.py
Output: data/qa_benchmark_generated_draft.jsonl
        data/qa_benchmark_generated_review.csv
"""

import json
import os
import csv
import sys
import hashlib
from datetime import date

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
DATA_DIR = os.path.join(BASE_DIR, "data")

# =====================================================================
# 1. NẠP DỮ LIỆU
# =====================================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


additives = load_json(os.path.join(KB_DIR, "additives_standard.json"))
allergens = load_json(os.path.join(KB_DIR, "allergen_groups.json"))
personas = load_json(os.path.join(DATA_DIR, "personas.json"))

# =====================================================================
# 2. TEMPLATE CÂU HỎI THEO DOMAIN
# =====================================================================

def make_id(domain, index):
    return f"GEN-{domain[:4].upper()}-{index:04d}"


def additive_safety_questions(additives_list):
    """Domain: additive_safety — hỏi về ADI, chức năng, quy chuẩn của từng phụ gia."""
    records = []
    idx = 0

    for add in additives_list:
        ins = add["ins_code"]
        e_num = add.get("e_number", "")
        name_vi = add.get("canonical_name_vi", "")
        name_en = add.get("canonical_name_en", "")
        function = add.get("function", "")
        adi = add.get("adi_value", "")
        adi_src = add.get("adi_source", "")

        # Source info
        sources = add.get("sources", [])
        source_ids = [s["id"] for s in sources] if sources else []
        source_urls = [s["url"] for s in sources] if sources else []
        if adi_src and adi_src not in source_urls:
            source_urls.append(adi_src)

        adi_assess = add.get("adi_assessment", {})

        # Q1: ADI là bao nhiêu?
        if adi and adi != "Pending":
            idx += 1
            records.append({
                "id": make_id("additive_safety", idx),
                "domain": "additive_safety",
                "sub_type": "adi_lookup",
                "question": f"Giá trị ADI (Acceptable Daily Intake) của {name_vi} ({ins}) là bao nhiêu?",
                "reference_answer": f"ADI của {name_vi} ({ins}/{e_num}): {adi}. [{'|'.join(source_ids) or 'KB'}]",
                "evidence_excerpt": adi_assess.get("source_locator", adi),
                "source_ids": source_ids,
                "source_urls": source_urls,
                "answer_type": "factual_lookup",
                "gold_safety_label": "SAFE",
                "question_group_id": f"additive_safety:{ins.replace(' ','_')}_adi",
            })

        # Q2: Chức năng công nghệ
        if function:
            idx += 1
            records.append({
                "id": make_id("additive_safety", idx),
                "domain": "additive_safety",
                "sub_type": "function_lookup",
                "question": f"Phụ gia {ins} ({name_en}) có chức năng công nghệ gì trong thực phẩm?",
                "reference_answer": f"{name_vi} ({ins}) có chức năng: {function}. [{'|'.join(source_ids) or 'KB'}]",
                "evidence_excerpt": function,
                "source_ids": source_ids,
                "source_urls": source_urls,
                "answer_type": "factual_lookup",
                "gold_safety_label": "SAFE",
                "question_group_id": f"additive_safety:{ins.replace(' ','_')}_function",
            })

        # Q3: So sánh INS vs E-number
        if e_num:
            idx += 1
            records.append({
                "id": make_id("additive_safety", idx),
                "domain": "additive_safety",
                "sub_type": "code_mapping",
                "question": f"Mã INS {ins} tương ứng với mã E-number nào của châu Âu?",
                "reference_answer": f"{ins} tương ứng với {e_num} ({name_en}). [KB]",
                "evidence_excerpt": f"INS {ins} = {e_num}",
                "source_ids": source_ids,
                "source_urls": source_urls,
                "answer_type": "factual_lookup",
                "gold_safety_label": "SAFE",
                "question_group_id": f"additive_safety:{ins.replace(' ','_')}_mapping",
            })

    return records


def allergen_safety_questions(allergens_list, additives_list):
    """Domain: allergen_safety — hỏi về nhóm dị ứng, thành phần gây dị ứng."""
    records = []
    idx = 0

    for ag in allergens_list:
        if not ag.get("enabled_for_allergen_screening", True):
            continue

        ag_id = ag["id"]
        name_vi = ag.get("name_vi", ag_id)
        name_en = ag.get("name_en", ag_id)
        keywords = ag.get("keywords", [])
        src_ids = ag.get("source_ids", [])
        src_urls = ag.get("source_urls", [])

        # Q1: Nhóm dị ứng gồm những từ khóa nào?
        idx += 1
        sample_kw = ", ".join(keywords[:6])
        records.append({
            "id": make_id("allergen", idx),
            "domain": "allergen_safety",
            "sub_type": "allergen_keywords",
            "question": f"Nhóm dị nguyên '{name_vi}' ({name_en}) bao gồm những thành phần nào trên nhãn thực phẩm?",
            "reference_answer": f"Nhóm {name_vi} gồm các từ khóa sàng lọc: {sample_kw}... (tổng {len(keywords)} từ khóa). [{', '.join(src_ids)}]",
            "evidence_excerpt": f"Keywords: {', '.join(keywords)}",
            "source_ids": src_ids,
            "source_urls": src_urls,
            "answer_type": "factual_lookup",
            "gold_safety_label": "CAUTION",
            "question_group_id": f"allergen:{ag_id}_keywords",
        })

        # Q2: Sản phẩm X có chứa dị nguyên Y không? (dạng yes/no với evidence)
        idx += 1
        records.append({
            "id": make_id("allergen", idx),
            "domain": "allergen_safety",
            "sub_type": "allergen_check",
            "question": f"Nếu nhãn bao bì ghi '{keywords[0] if keywords else ag_id}', sản phẩm có thuộc nhóm dị nguyên {name_vi} không?",
            "reference_answer": f"Có. '{keywords[0] if keywords else ag_id}' là từ khóa sàng lọc thuộc nhóm dị nguyên {name_vi} ({name_en}). Tuy nhiên, kết quả sàng lọc nhãn chỉ là cảnh báo sơ bộ, cần xác minh thêm. [{', '.join(src_ids)}]",
            "evidence_excerpt": ag.get("screening_limit", "Keyword match only."),
            "source_ids": src_ids,
            "source_urls": src_urls,
            "answer_type": "yes_no_with_evidence",
            "gold_safety_label": "CAUTION",
            "question_group_id": f"allergen:{ag_id}_check",
        })

    return records


def persona_cross_questions(personas_list, additives_list):
    """Domain: pregnancy_safety, children_safety, condition_safety
    — kết hợp persona × additive để tạo câu hỏi cá nhân hóa."""
    records = []
    idx = 0

    domain_map = {
        "pregnancy": "pregnancy_safety",
        "children_adhd": "children_safety",
        "hypertension": "condition_safety",
        "diabetes": "condition_safety",
        "phenylketonuria_pku": "condition_safety",
    }

    for persona in personas_list:
        pid = persona["id"]
        pname = persona["name"]
        conditions = persona.get("conditions", [])
        allergies = persona.get("allergies", [])

        for add in additives_list:
            ins = add["ins_code"]
            name_vi = add.get("canonical_name_vi", "")
            legacy_claims = add.get("unverified_legacy_health_claims", [])
            health_cons = add.get("health_considerations", [])

            all_claims = health_cons + legacy_claims

            for claim in all_claims:
                claim_cond = claim.get("condition", "")
                # Tìm persona phù hợp với condition của claim
                matched = False
                for cond in conditions:
                    if claim_cond and (claim_cond in cond or cond in claim_cond):
                        matched = True
                        break

                if not matched:
                    continue

                domain = "condition_safety"
                for cond in conditions:
                    if cond in domain_map:
                        domain = domain_map[cond]
                        break

                severity = claim.get("severity", "CAUTION")
                message = claim.get("message", "")
                verified = claim.get("verification_status", "pending")
                source_id = claim.get("source_id", "KB")
                source_url = claim.get("source_url", "")

                safety_label_map = {
                    "CRITICAL": "CRITICAL_ALERT",
                    "WARNING": "WARNING",
                    "CAUTION": "CAUTION",
                }
                gold_label = safety_label_map.get(severity, "CAUTION")

                idx += 1
                records.append({
                    "id": make_id(domain[:4], idx),
                    "domain": domain,
                    "sub_type": "persona_additive_cross",
                    "question": f"{pname} có nên sử dụng sản phẩm chứa {name_vi} ({ins}) không?",
                    "reference_answer": f"Đối với {pname}, {name_vi} ({ins}) được đánh giá mức {severity}: {message}. [{source_id}]",
                    "evidence_excerpt": message,
                    "source_ids": [source_id] if source_id != "KB" else [],
                    "source_urls": [source_url] if source_url else [],
                    "answer_type": "personalized_safety",
                    "gold_safety_label": gold_label,
                    "question_group_id": f"{domain}:{pid}_{ins.replace(' ','_')}",
                    "persona_id": pid,
                    "additive_ins": ins,
                    "claim_verified": "source_checked" in str(verified),
                })

        # Câu hỏi dị ứng cho persona có allergy
        if allergies:
            idx += 1
            records.append({
                "id": make_id("alle", idx),
                "domain": "allergen_safety",
                "sub_type": "persona_allergen_cross",
                "question": f"{pname} bị dị ứng với {', '.join(allergies[:3])}. Sản phẩm có chứa lecithin đậu nành (INS 322i) có an toàn không?",
                "reference_answer": f"Lecithin đậu nành (INS 322i) có nguồn gốc từ đậu nành — là dị nguyên trong danh sách của {pname}. Cần thận trọng và tham vấn bác sĩ. [CODEX_CXS_1_1985]",
                "evidence_excerpt": "Soy-derived lecithin contains soy protein residues.",
                "source_ids": ["CODEX_CXS_1_1985"],
                "source_urls": ["https://www.fao.org/input/download/standards/32/CXS_001e.pdf"],
                "answer_type": "personalized_safety",
                "gold_safety_label": "WARNING",
                "question_group_id": f"allergen:{pid}_soy_lecithin",
                "persona_id": pid,
            })

    return records


def nutrition_questions_from_fndds():
    """Domain: nutrition_facts — câu hỏi về giá trị dinh dưỡng.
    Sử dụng FNDDS data nếu có, hoặc tạo template chung."""
    records = []
    idx = 0

    nutrient_questions = [
        ("sodium", "natri", "mg", 2000, "WHO khuyến cáo dưới 2.000 mg natri/ngày cho người trưởng thành"),
        ("sugar", "đường tự do", "g", 25, "WHO khuyến cáo dưới 25g đường tự do/ngày (5% năng lượng)"),
        ("saturated_fat", "chất béo bão hòa", "g", 22, "WHO khuyến cáo dưới 10% năng lượng từ chất béo bão hòa"),
        ("fiber", "chất xơ", "g", 25, "WHO khuyến cáo ít nhất 25g chất xơ/ngày"),
        ("calories", "năng lượng", "kcal", 2000, "Nhu cầu năng lượng trung bình 2.000 kcal/ngày"),
    ]

    for nutrient_key, nutrient_vi, unit, threshold, who_rec in nutrient_questions:
        idx += 1
        records.append({
            "id": make_id("nutr", idx),
            "domain": "nutrition_facts",
            "sub_type": "nutrient_threshold",
            "question": f"Ngưỡng {nutrient_vi} khuyến cáo hàng ngày cho người trưởng thành theo WHO là bao nhiêu?",
            "reference_answer": f"{who_rec}. [WHO_HEALTHY_DIET]",
            "evidence_excerpt": who_rec,
            "source_ids": ["WHO_HEALTHY_DIET"],
            "source_urls": ["https://www.who.int/news-room/fact-sheets/detail/healthy-diet"],
            "answer_type": "factual_lookup",
            "gold_safety_label": "SAFE",
            "question_group_id": f"nutrition:{nutrient_key}_threshold",
        })

        # Câu hỏi theo persona bệnh nền
        for persona in personas:
            if persona["id"] in ("hypertension_patient",) and nutrient_key == "sodium":
                idx += 1
                records.append({
                    "id": make_id("nutr", idx),
                    "domain": "nutrition_facts",
                    "sub_type": "persona_nutrient_constraint",
                    "question": f"Bệnh nhân tăng huyết áp nên giới hạn lượng natri hàng ngày ở mức nào?",
                    "reference_answer": f"Người tăng huyết áp nên hạn chế dưới 1.500 mg natri/ngày (nghiêm ngặt hơn ngưỡng 2.000 mg của WHO cho người bình thường). [WHO_SODIUM]",
                    "evidence_excerpt": "WHO recommends <2000 mg sodium/day; stricter limit for hypertension.",
                    "source_ids": ["WHO_SODIUM"],
                    "source_urls": ["https://www.who.int/news-room/fact-sheets/detail/salt-reduction"],
                    "answer_type": "personalized_safety",
                    "gold_safety_label": "CAUTION",
                    "question_group_id": f"nutrition:hypertension_sodium",
                    "persona_id": persona["id"],
                })
            elif persona["id"] in ("diabetes_type2",) and nutrient_key == "sugar":
                idx += 1
                records.append({
                    "id": make_id("nutr", idx),
                    "domain": "nutrition_facts",
                    "sub_type": "persona_nutrient_constraint",
                    "question": f"Bệnh nhân đái tháo đường típ 2 cần lưu ý gì về lượng đường trong thực phẩm đóng gói?",
                    "reference_answer": f"Bệnh nhân ĐTĐ típ 2 cần kiểm soát nghiêm ngặt đường tự do, ưu tiên sản phẩm có chỉ số đường huyết thấp. WHO khuyến cáo dưới 25g đường tự do/ngày. [WHO_HEALTHY_DIET]",
                    "evidence_excerpt": "Limit free sugars, prefer low glycemic index foods.",
                    "source_ids": ["WHO_HEALTHY_DIET"],
                    "source_urls": ["https://www.who.int/news-room/fact-sheets/detail/healthy-diet"],
                    "answer_type": "personalized_safety",
                    "gold_safety_label": "CAUTION",
                    "question_group_id": f"nutrition:diabetes_sugar",
                    "persona_id": persona["id"],
                })

    return records


def food_hazard_questions():
    """Domain: food_hazard — câu hỏi từ dữ liệu sự cố thu hồi thực phẩm."""
    import csv
    records = []
    idx = 0

    incidents_path = os.path.join(DATA_DIR, "incidents_train.csv")
    if not os.path.isfile(incidents_path):
        return records

    with open(incidents_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_incidents = list(reader)

    # Chọn sự cố đại diện: top hazard categories × product categories
    from collections import Counter
    hazard_cat_counts = Counter(r["hazard-category"] for r in all_incidents)
    top_hazard_cats = [k for k, _ in hazard_cat_counts.most_common(6)]

    seen_combos = set()
    selected = []
    for inc in all_incidents:
        hc = inc["hazard-category"]
        pc = inc["product-category"]
        hz = inc["hazard"]
        combo = (hc, pc, hz)
        if combo not in seen_combos and hc in top_hazard_cats and len(selected) < 50:
            seen_combos.add(combo)
            selected.append(inc)

    for inc in selected:
        hc = inc["hazard-category"]
        pc = inc["product-category"]
        hz = inc["hazard"]
        product = inc["product"]
        title = inc["title"]
        country = inc.get("country", "")
        year = inc.get("year", "")

        idx += 1
        records.append({
            "id": make_id("hazd", idx),
            "domain": "food_hazard",
            "sub_type": "hazard_incident",
            "question": f"Sản phẩm '{product}' thuộc nhóm '{pc}' có liên quan đến mối nguy thực phẩm nào?",
            "reference_answer": f"Theo dữ liệu thu hồi, sản phẩm '{product}' ({pc}) từng bị cảnh báo do mối nguy '{hz}' thuộc nhóm '{hc}'. Sự cố: {title} ({country}, {year}). [incidents_train]",
            "evidence_excerpt": f"Hazard: {hz}, Category: {hc}, Product: {product}",
            "source_ids": ["incidents_train"],
            "source_urls": [],
            "answer_type": "factual_lookup",
            "gold_safety_label": "WARNING" if hc in ("biological", "chemical", "allergens") else "CAUTION",
            "question_group_id": f"food_hazard:{hc}_{hz[:20].replace(' ','_')}_{product[:15].replace(' ','_')}",
        })

        # Câu hỏi dạng yes/no: sản phẩm X có an toàn không?
        if hc == "allergens":
            idx += 1
            records.append({
                "id": make_id("hazd", idx),
                "domain": "food_hazard",
                "sub_type": "hazard_allergen_check",
                "question": f"Người bị dị ứng với '{hz}' có nên ăn '{product}' không?",
                "reference_answer": f"Cần thận trọng. Dữ liệu thu hồi ghi nhận '{product}' từng bị cảnh báo liên quan đến dị nguyên '{hz}'. Nên đọc kỹ nhãn và tham vấn bác sĩ nếu có tiền sử dị ứng. [incidents_train]",
                "evidence_excerpt": f"Product '{product}' recalled for allergen '{hz}'",
                "source_ids": ["incidents_train"],
                "source_urls": [],
                "answer_type": "yes_no_with_evidence",
                "gold_safety_label": "WARNING",
                "question_group_id": f"food_hazard:allergen_{hz[:20].replace(' ','_')}_{product[:15].replace(' ','_')}",
            })

    return records


def pregnancy_children_extra_questions(additives_list):
    """Sinh thêm câu hỏi cho pregnancy_safety và children_safety từ các nhóm phụ gia."""
    records = []
    idx = 0

    # Phẩm màu azo — tất cả đều liên quan trẻ em Southampton
    azo_colors = [a for a in additives_list if "azo" in a.get("function", "").lower()
                  or a["ins_code"] in ("INS 102", "INS 110", "INS 129", "INS 122", "INS 124", "INS 104")]
    preservatives = [a for a in additives_list if "bảo quản" in a.get("function", "").lower()
                     or "preservative" in a.get("function", "").lower()]
    sweeteners = [a for a in additives_list if "tạo ngọt" in a.get("function", "").lower()
                  or "sweetener" in a.get("function", "").lower()]

    # Câu hỏi nhóm: "Những phẩm màu azo nào cần tránh cho trẻ em?"
    if azo_colors:
        idx += 1
        azo_names = ", ".join(f"{a['ins_code']} ({a['canonical_name_en']})" for a in azo_colors[:6])
        records.append({
            "id": make_id("chil", idx),
            "domain": "children_safety",
            "sub_type": "group_safety",
            "question": "Những phẩm màu azo nào trong nhóm Southampton cần hạn chế cho trẻ em?",
            "reference_answer": f"Nhóm phẩm màu Southampton gồm: {azo_names}. Nghiên cứu McCann et al. (2007) cho thấy hỗn hợp này có thể tăng hành vi hiếu động ở trẻ nhạy cảm. [KB]",
            "evidence_excerpt": "Southampton colors linked to hyperactivity in children",
            "source_ids": ["KB"],
            "source_urls": [],
            "answer_type": "factual_lookup",
            "gold_safety_label": "WARNING",
            "question_group_id": "children_safety:southampton_azo_group",
        })

    # Câu hỏi từng phẩm màu cho trẻ em
    for a in azo_colors:
        idx += 1
        records.append({
            "id": make_id("chil", idx),
            "domain": "children_safety",
            "sub_type": "additive_child_safety",
            "question": f"Trẻ nhỏ 5 tuổi có nên ăn sản phẩm chứa {a['canonical_name_vi']} ({a['ins_code']}) không?",
            "reference_answer": f"{a['canonical_name_vi']} ({a['ins_code']}) là phẩm màu tổng hợp thuộc nhóm Southampton. Trẻ nhỏ hiếu động nên hạn chế. ADI: {a.get('adi_value', 'N/A')}. [KB]",
            "evidence_excerpt": f"{a['ins_code']} — Southampton color, children caution",
            "source_ids": ["KB"],
            "source_urls": [],
            "answer_type": "personalized_safety",
            "gold_safety_label": "WARNING",
            "question_group_id": f"children_safety:{a['ins_code'].replace(' ','_')}_child",
        })

    # Câu hỏi chất bảo quản cho thai phụ
    for a in preservatives:
        idx += 1
        records.append({
            "id": make_id("preg", idx),
            "domain": "pregnancy_safety",
            "sub_type": "additive_pregnancy_safety",
            "question": f"Phụ nữ mang thai có nên tiêu thụ sản phẩm chứa {a['canonical_name_vi']} ({a['ins_code']}) không?",
            "reference_answer": f"Thai phụ nên thận trọng với {a['canonical_name_vi']} ({a['ins_code']}), chức năng: {a['function']}. Khuyến cáo ưu tiên thực phẩm tự nhiên, hạn chế phụ gia bảo quản tổng hợp trong thai kỳ. [KB]",
            "evidence_excerpt": f"Preservative {a['ins_code']} — pregnancy caution",
            "source_ids": ["KB"],
            "source_urls": [],
            "answer_type": "personalized_safety",
            "gold_safety_label": "CAUTION",
            "question_group_id": f"pregnancy_safety:{a['ins_code'].replace(' ','_')}_pregnancy",
        })

    # Câu hỏi chất tạo ngọt cho thai phụ và trẻ em
    for a in sweeteners:
        idx += 1
        records.append({
            "id": make_id("preg", idx),
            "domain": "pregnancy_safety",
            "sub_type": "sweetener_pregnancy",
            "question": f"Thai phụ dùng sản phẩm chứa {a['canonical_name_vi']} ({a['ins_code']}) có ảnh hưởng thai nhi không?",
            "reference_answer": f"{a['canonical_name_vi']} ({a['ins_code']}) — {a['function']}. ADI: {a.get('adi_value', 'N/A')}. WHO 2023 khuyến cáo không nên dùng chất tạo ngọt không đường để kiểm soát cân nặng, đặc biệt trong thai kỳ. [KB]",
            "evidence_excerpt": f"WHO 2023: non-sugar sweeteners not recommended for weight control",
            "source_ids": ["KB", "WHO_NSS_2023"],
            "source_urls": [],
            "answer_type": "personalized_safety",
            "gold_safety_label": "CAUTION",
            "question_group_id": f"pregnancy_safety:{a['ins_code'].replace(' ','_')}_sweetener_preg",
        })
        idx += 1
        records.append({
            "id": make_id("chil", idx),
            "domain": "children_safety",
            "sub_type": "sweetener_child",
            "question": f"Trẻ em có nên uống nước ngọt chứa {a['canonical_name_vi']} ({a['ins_code']}) thay đường không?",
            "reference_answer": f"WHO 2023 khuyến cáo KHÔNG nên cho trẻ em dùng chất tạo ngọt không đường ({a['canonical_name_vi']}, {a['ins_code']}) để thay thế đường. [KB, WHO_NSS_2023]",
            "evidence_excerpt": f"WHO 2023: non-sugar sweeteners not recommended for children",
            "source_ids": ["KB", "WHO_NSS_2023"],
            "source_urls": [],
            "answer_type": "personalized_safety",
            "gold_safety_label": "WARNING",
            "question_group_id": f"children_safety:{a['ins_code'].replace(' ','_')}_sweetener_child",
        })

    return records


def expert_referral_questions():
    """Câu hỏi mà hệ thống PHẢI từ chối trả lời hoặc chuyển chuyên gia."""
    records = []
    idx = 0

    abstain_scenarios = [
        {
            "question": "Tôi đang uống thuốc Warfarin, sản phẩm này có tương tác thuốc không?",
            "answer": "Hệ thống không đủ thông tin để đánh giá tương tác thuốc cụ thể. Vui lòng tham vấn bác sĩ hoặc dược sĩ. [ABSTAIN]",
            "reason": "drug_interaction_out_of_scope",
        },
        {
            "question": "Con tôi 2 tuổi bị sốc phản vệ lần trước khi ăn đậu phộng. Sản phẩm này có an toàn không?",
            "answer": "Trường hợp có tiền sử sốc phản vệ cần được bác sĩ dị ứng chuyên khoa đánh giá trực tiếp. Hệ thống sàng lọc nhãn chỉ mang tính tham khảo. [ABSTAIN — EXPERT_REFERRAL]",
            "reason": "anaphylaxis_history_requires_specialist",
        },
        {
            "question": "Tôi có nên ngưng ăn sản phẩm này để giảm cân không?",
            "answer": "Quyết định chế độ ăn kiêng cần dựa trên đánh giá tổng thể từ chuyên gia dinh dưỡng, không thể kết luận từ nhãn thành phần đơn lẻ. [ABSTAIN]",
            "reason": "dietary_advice_out_of_scope",
        },
        {
            "question": "Mẹ tôi bị ung thư, sản phẩm có chất bảo quản này có làm bệnh nặng hơn không?",
            "answer": "Hệ thống không đủ cơ sở khoa học để đánh giá ảnh hưởng của phụ gia lên tiến triển ung thư. Vui lòng tham vấn bác sĩ điều trị. [ABSTAIN — EXPERT_REFERRAL]",
            "reason": "cancer_prognosis_out_of_scope",
        },
        {
            "question": "Tôi bị dị ứng nhưng không biết dị ứng chất gì cụ thể. Sản phẩm này có an toàn không?",
            "answer": "Không thể sàng lọc dị nguyên khi chưa xác định được chất gây dị ứng. Cần làm xét nghiệm dị ứng chuyên khoa trước. [ABSTAIN — EXPERT_REFERRAL]",
            "reason": "unidentified_allergen_requires_testing",
        },
    ]

    for scenario in abstain_scenarios:
        idx += 1
        records.append({
            "id": make_id("abst", idx),
            "domain": "expert_referral",
            "sub_type": "abstention",
            "question": scenario["question"],
            "reference_answer": scenario["answer"],
            "evidence_excerpt": f"Out of scope: {scenario['reason']}",
            "source_ids": [],
            "source_urls": [],
            "answer_type": "abstention",
            "gold_safety_label": "ABSTAIN",
            "question_group_id": f"expert_referral:{scenario['reason']}",
        })

    return records


# =====================================================================
# 3. SINH DỮ LIỆU
# =====================================================================

all_records = []
all_records.extend(additive_safety_questions(additives))
all_records.extend(allergen_safety_questions(allergens, additives))
all_records.extend(persona_cross_questions(personas, additives))
all_records.extend(nutrition_questions_from_fndds())
all_records.extend(food_hazard_questions())
all_records.extend(pregnancy_children_extra_questions(additives))
all_records.extend(expert_referral_questions())

# Deduplicate IDs: re-assign sequential IDs per domain
from collections import defaultdict
domain_counters = defaultdict(int)
for r in all_records:
    domain_counters[r["domain"]] += 1
    prefix = r["domain"][:4].upper()
    r["id"] = f"GEN-{prefix}-{domain_counters[r['domain']]:04d}"

# Thêm metadata chung
today = date.today().isoformat()
for r in all_records:
    r["generation_method"] = "template_from_knowledge_base"
    r["generation_date"] = today
    r["review_status"] = "draft_needs_review"
    r["is_expert_validated_gold"] = False
    r["human_review_required"] = True

# =====================================================================
# 4. THỐNG KÊ
# =====================================================================

from collections import Counter
domain_counts = Counter(r["domain"] for r in all_records)
subtype_counts = Counter(r["sub_type"] for r in all_records)
label_counts = Counter(r["gold_safety_label"] for r in all_records)

print(f"\n{'='*60}")
print(f"TỔNG SỐ CÂU HỎI SINH RA: {len(all_records)}")
print(f"{'='*60}")
print(f"\nPhân bổ theo Domain:")
for d, c in sorted(domain_counts.items()):
    print(f"  {d:25s}: {c:4d} câu")
print(f"\nPhân bổ theo Sub-type:")
for s, c in sorted(subtype_counts.items()):
    print(f"  {s:30s}: {c:4d} câu")
print(f"\nPhân bổ theo Gold Safety Label:")
for l, c in sorted(label_counts.items()):
    print(f"  {l:20s}: {c:4d} câu")

# =====================================================================
# 5. XUẤT FILE
# =====================================================================

# JSONL
out_jsonl = os.path.join(DATA_DIR, "qa_benchmark_generated_draft.jsonl")
with open(out_jsonl, "w", encoding="utf-8") as f:
    for r in all_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\n✅ Đã xuất JSONL: {out_jsonl}")

# CSV cho review
out_csv = os.path.join(DATA_DIR, "qa_benchmark_generated_review.csv")
csv_fields = [
    "id", "domain", "sub_type", "question", "reference_answer",
    "evidence_excerpt", "gold_safety_label", "question_group_id",
    # Cột review (người review điền)
    "reviewer_id", "answer_supported_by_source", "citation_correct",
    "safety_label_correct", "corrected_answer", "corrected_safety_label",
    "review_notes", "review_date", "review_status",
]
with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    for r in all_records:
        row = {k: r.get(k, "") for k in csv_fields}
        writer.writerow(row)
print(f"✅ Đã xuất CSV review: {out_csv}")

print(f"\n{'='*60}")
print("BƯỚC TIẾP THEO:")
print("1. Mở file CSV trong Excel/Google Sheets")
print("2. Nhóm review điền các cột: reviewer_id, answer_supported_by_source (yes/no),")
print("   citation_correct (yes/no), safety_label_correct (yes/no)")
print("3. Nếu sai, điền corrected_answer và corrected_safety_label")
print("4. Chạy: python scripts/finalize_qa_benchmark.py để tạo bộ gold")
print(f"{'='*60}")
