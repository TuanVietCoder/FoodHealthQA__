"""
Mở rộng knowledge_base/additives_standard.json thêm 25 phụ gia phổ biến
trên bao bì thực phẩm Việt Nam. Giữ đúng schema hiện có.

Chạy: python scripts/expand_kb_additives.py
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(BASE_DIR, "knowledge_base", "additives_standard.json")

NEW_ADDITIVES = [
    {
        "ins_code": "INS 220",
        "e_number": "E220",
        "canonical_name_vi": "Sulfur dioxide (Lưu huỳnh dioxide)",
        "canonical_name_en": "Sulfur dioxide",
        "synonyms": ["SO2", "lưu huỳnh đioxit", "sulphur dioxide"],
        "function": "Chất bảo quản, chống oxy hóa (Preservative, antioxidant)",
        "adi_value": "EFSA đã rút lại ADI tạm thời năm 2022; JECFA ADI: 0-0.7 mg/kg bw/ngày",
        "adi_upper": 0.7,
        "unverified_legacy_health_claims": [
            {"condition": "asthma_allergy", "severity": "CRITICAL", "message": "Sulfite gây co thắt phế quản nghiêm trọng ở 5-10% bệnh nhân hen suyễn, có thể gây sốc phản vệ", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False},
            {"condition": "pregnancy", "severity": "WARNING", "message": "Khuyến cáo hạn chế sulfite trong thai kỳ", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": ["sulfites"],
    },
    {
        "ins_code": "INS 221",
        "e_number": "E221",
        "canonical_name_vi": "Natri sulfit",
        "canonical_name_en": "Sodium sulfite",
        "synonyms": ["sodium sulphite", "natri sunfit"],
        "function": "Chất bảo quản (Preservative)",
        "adi_value": "JECFA group ADI: 0-0.7 mg/kg bw/ngày (tính theo SO2, nhóm E220-E228)",
        "adi_upper": 0.7,
        "unverified_legacy_health_claims": [
            {"condition": "asthma_allergy", "severity": "CRITICAL", "message": "Nhóm sulfite gây phản ứng nghiêm trọng ở người hen suyễn", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": ["sulfites"],
    },
    {
        "ins_code": "INS 250",
        "e_number": "E250",
        "canonical_name_vi": "Natri nitrit",
        "canonical_name_en": "Sodium nitrite",
        "synonyms": ["sodium nitrite", "chất bảo quản 250", "muối diêm"],
        "function": "Chất bảo quản, ổn định màu thịt chế biến (Preservative, color fixative)",
        "adi_value": "EFSA ADI: 0-0.07 mg/kg bw/ngày (2017)",
        "adi_upper": 0.07,
        "unverified_legacy_health_claims": [
            {"condition": "pregnancy", "severity": "WARNING", "message": "Nitrit kết hợp amin trong thực phẩm có thể tạo nitrosamine gây ung thư; thai phụ nên hạn chế thịt chế biến", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False},
            {"condition": "children_adhd", "severity": "CAUTION", "message": "Trẻ em nên hạn chế thịt chế biến chứa nitrit", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 251",
        "e_number": "E251",
        "canonical_name_vi": "Natri nitrat",
        "canonical_name_en": "Sodium nitrate",
        "synonyms": ["sodium nitrate", "diêm tiêu", "muối diêm tiêu"],
        "function": "Chất bảo quản thịt (Preservative)",
        "adi_value": "EFSA ADI: 0-3.7 mg/kg bw/ngày (2017)",
        "adi_upper": 3.7,
        "unverified_legacy_health_claims": [
            {"condition": "pregnancy", "severity": "CAUTION", "message": "Nitrat chuyển hóa thành nitrit trong cơ thể; thai phụ nên thận trọng", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 320",
        "e_number": "E320",
        "canonical_name_vi": "BHA (Butylated hydroxyanisole)",
        "canonical_name_en": "Butylated hydroxyanisole (BHA)",
        "synonyms": ["BHA", "butylhydroxyanisol"],
        "function": "Chất chống oxy hóa tổng hợp (Synthetic antioxidant)",
        "adi_value": "JECFA ADI: 0-0.5 mg/kg bw/ngày",
        "adi_upper": 0.5,
        "unverified_legacy_health_claims": [
            {"condition": "pregnancy", "severity": "CAUTION", "message": "IARC xếp BHA vào nhóm 2B (có thể gây ung thư ở người); thai phụ nên hạn chế", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False},
            {"condition": "children_adhd", "severity": "CAUTION", "message": "Nên hạn chế chất chống oxy hóa tổng hợp cho trẻ em", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 321",
        "e_number": "E321",
        "canonical_name_vi": "BHT (Butylated hydroxytoluene)",
        "canonical_name_en": "Butylated hydroxytoluene (BHT)",
        "synonyms": ["BHT", "butylhydroxytoluen"],
        "function": "Chất chống oxy hóa tổng hợp (Synthetic antioxidant)",
        "adi_value": "JECFA ADI: 0-0.3 mg/kg bw/ngày",
        "adi_upper": 0.3,
        "unverified_legacy_health_claims": [
            {"condition": "pregnancy", "severity": "CAUTION", "message": "Nên hạn chế BHT trong thai kỳ do thiếu dữ liệu an toàn đầy đủ", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False},
            {"condition": "children_adhd", "severity": "CAUTION", "message": "Khuyến cáo hạn chế cho trẻ nhỏ", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 338",
        "e_number": "E338",
        "canonical_name_vi": "Acid phosphoric",
        "canonical_name_en": "Phosphoric acid",
        "synonyms": ["phosphoric acid", "axit photphoric", "acid orthophosphoric"],
        "function": "Chất điều chỉnh độ chua, hương liệu (Acidity regulator)",
        "adi_value": "EFSA group ADI: 40 mg/kg bw/ngày (tính theo phosphorus, tổng phosphate)",
        "adi_upper": 40.0,
        "unverified_legacy_health_claims": [
            {"condition": "chronic_kidney_disease", "severity": "WARNING", "message": "Phosphate vô cơ gây tăng phospho máu, nguy hiểm cho bệnh nhân suy thận", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False},
            {"condition": "hypertension", "severity": "CAUTION", "message": "Nước ngọt có gas chứa acid phosphoric góp phần tăng huyết áp gián tiếp", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 339(i)",
        "e_number": "E339(i)",
        "canonical_name_vi": "Mononatri phosphat",
        "canonical_name_en": "Monosodium phosphate",
        "synonyms": ["sodium dihydrogen phosphate", "MSP"],
        "function": "Chất điều chỉnh độ chua, chất nhũ hóa (Acidity regulator, emulsifier)",
        "adi_value": "EFSA group ADI: 40 mg/kg bw/ngày (tính theo phosphorus)",
        "adi_upper": 40.0,
        "unverified_legacy_health_claims": [
            {"condition": "chronic_kidney_disease", "severity": "WARNING", "message": "Phosphate natri gây tăng phospho máu và natri máu; nguy hiểm cho bệnh nhân thận", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False},
            {"condition": "hypertension", "severity": "WARNING", "message": "Chứa natri; cần tính vào tổng lượng natri nạp", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 440",
        "e_number": "E440",
        "canonical_name_vi": "Pectin",
        "canonical_name_en": "Pectin",
        "synonyms": ["pectin", "pektin", "chất tạo gel"],
        "function": "Chất tạo gel, chất làm dày, chất ổn định (Gelling agent, thickener)",
        "adi_value": "JECFA: ADI không xác định (Not specified — an toàn ở mức sử dụng thông thường)",
        "adi_upper": None,
        "unverified_legacy_health_claims": [],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 450(i)",
        "e_number": "E450(i)",
        "canonical_name_vi": "Dinatri diphosphat",
        "canonical_name_en": "Disodium diphosphate",
        "synonyms": ["SAPP", "sodium acid pyrophosphate", "dinatri pyrophosphat"],
        "function": "Chất nở, chất nhũ hóa, chất giữ ẩm (Raising agent, emulsifier)",
        "adi_value": "EFSA group ADI: 40 mg/kg bw/ngày (tính theo phosphorus)",
        "adi_upper": 40.0,
        "unverified_legacy_health_claims": [
            {"condition": "chronic_kidney_disease", "severity": "WARNING", "message": "Diphosphate gây tăng phospho máu ở bệnh nhân suy thận", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False},
            {"condition": "hypertension", "severity": "CAUTION", "message": "Chứa natri; cần kiểm soát tổng lượng natri", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 452(i)",
        "e_number": "E452(i)",
        "canonical_name_vi": "Natri polyphosphat",
        "canonical_name_en": "Sodium polyphosphate",
        "synonyms": ["sodium polyphosphate", "sodium hexametaphosphate", "SHMP"],
        "function": "Chất nhũ hóa, chất giữ nước trong thịt chế biến (Emulsifier, moisture retention)",
        "adi_value": "EFSA group ADI: 40 mg/kg bw/ngày (tính theo phosphorus)",
        "adi_upper": 40.0,
        "unverified_legacy_health_claims": [
            {"condition": "chronic_kidney_disease", "severity": "WARNING", "message": "Polyphosphate natri nguy hiểm cho bệnh nhân suy thận — tăng phospho và natri máu", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 955",
        "e_number": "E955",
        "canonical_name_vi": "Sucralose",
        "canonical_name_en": "Sucralose",
        "synonyms": ["sucralose", "splenda", "chất tạo ngọt 955"],
        "function": "Chất tạo ngọt nhân tạo (Sweetener)",
        "adi_value": "JECFA ADI: 0-15 mg/kg bw/ngày",
        "adi_upper": 15.0,
        "unverified_legacy_health_claims": [
            {"condition": "diabetes", "severity": "CAUTION", "message": "Sucralose không calo nhưng một số nghiên cứu gợi ý có thể ảnh hưởng đến hệ vi sinh đường ruột và đáp ứng insulin", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False},
            {"condition": "children_adhd", "severity": "CAUTION", "message": "Nên hạn chế chất tạo ngọt nhân tạo cho trẻ nhỏ theo WHO 2023", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 960a",
        "e_number": "E960a",
        "canonical_name_vi": "Steviol glycoside (Stevia)",
        "canonical_name_en": "Steviol glycosides",
        "synonyms": ["stevia", "stevioside", "rebaudioside A", "chất tạo ngọt stevia"],
        "function": "Chất tạo ngọt tự nhiên (Natural sweetener)",
        "adi_value": "JECFA ADI: 0-4 mg/kg bw/ngày (tính theo steviol)",
        "adi_upper": 4.0,
        "unverified_legacy_health_claims": [
            {"condition": "diabetes", "severity": "CAUTION", "message": "Steviol glycoside không ảnh hưởng đường huyết nhưng cần theo dõi tổng lượng đường thay thế", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 961",
        "e_number": "E961",
        "canonical_name_vi": "Neotame",
        "canonical_name_en": "Neotame",
        "synonyms": ["neotame", "neotam"],
        "function": "Chất tạo ngọt tổng hợp, ngọt gấp 8000 lần đường (Sweetener)",
        "adi_value": "JECFA ADI: 0-0.3 mg/kg bw/ngày",
        "adi_upper": 0.3,
        "unverified_legacy_health_claims": [
            {"condition": "phenylketonuria_pku", "severity": "CAUTION", "message": "Neotame là dẫn xuất aspartame nhưng không phóng thích phenylalanine đáng kể; FDA cho phép không cần cảnh báo PKU", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 621+631+627",
        "e_number": "E621+E631+E627",
        "canonical_name_vi": "Hỗn hợp điều vị (MSG + I+G)",
        "canonical_name_en": "Flavor enhancer blend (MSG + I+G)",
        "synonyms": ["bột nêm", "gia vị tổng hợp", "I+G", "hạt nêm"],
        "function": "Hỗn hợp chất điều vị tăng cường umami (Flavor enhancer blend)",
        "adi_value": "Theo ADI riêng từng thành phần: MSG (EFSA 30 mg/kg bw/ngày), IMP/GMP (JECFA: không xác định)",
        "adi_upper": None,
        "unverified_legacy_health_claims": [
            {"condition": "hypertension", "severity": "WARNING", "message": "Hỗn hợp MSG + I+G chứa natri cao; người cao huyết áp cần kiểm soát", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 401",
        "e_number": "E401",
        "canonical_name_vi": "Natri alginat",
        "canonical_name_en": "Sodium alginate",
        "synonyms": ["sodium alginate", "natri anginat", "algin"],
        "function": "Chất làm dày, chất tạo gel, chất ổn định (Thickener, gelling agent)",
        "adi_value": "JECFA: ADI không xác định (Not specified — an toàn)",
        "adi_upper": None,
        "unverified_legacy_health_claims": [
            {"condition": "hypertension", "severity": "CAUTION", "message": "Natri alginat chứa natri; cần tính vào tổng lượng natri nạp cho bệnh nhân cao huyết áp", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 410",
        "e_number": "E410",
        "canonical_name_vi": "Gôm hạt đậu cào (Carob bean gum)",
        "canonical_name_en": "Carob bean gum (Locust bean gum)",
        "synonyms": ["locust bean gum", "LBG", "gôm carob"],
        "function": "Chất làm dày, chất ổn định (Thickener, stabilizer)",
        "adi_value": "JECFA: ADI không xác định (Not specified — an toàn)",
        "adi_upper": None,
        "unverified_legacy_health_claims": [],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 414",
        "e_number": "E414",
        "canonical_name_vi": "Gôm arabic (Gôm acacia)",
        "canonical_name_en": "Gum arabic (Acacia gum)",
        "synonyms": ["gum arabic", "acacia gum", "gôm acacia", "gôm A-rập"],
        "function": "Chất nhũ hóa, chất ổn định, chất làm dày (Emulsifier, stabilizer)",
        "adi_value": "JECFA: ADI không xác định (Not specified — an toàn)",
        "adi_upper": None,
        "unverified_legacy_health_claims": [],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 133",
        "e_number": "E133",
        "canonical_name_vi": "Brilliant Blue FCF (Xanh sáng FCF)",
        "canonical_name_en": "Brilliant Blue FCF",
        "synonyms": ["FD&C Blue No. 1", "CI 42090", "xanh sáng", "brilliant blue"],
        "function": "Phẩm màu tổng hợp (Synthetic color)",
        "adi_value": "JECFA ADI: 0-12.5 mg/kg bw/ngày",
        "adi_upper": 12.5,
        "unverified_legacy_health_claims": [
            {"condition": "children_adhd", "severity": "CAUTION", "message": "Phẩm màu tổng hợp có thể ảnh hưởng đến hành vi ở trẻ em nhạy cảm (nghiên cứu Southampton)", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False},
            {"condition": "pregnancy", "severity": "CAUTION", "message": "Thai phụ nên hạn chế phẩm màu tổng hợp", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 122",
        "e_number": "E122",
        "canonical_name_vi": "Azorubine (Carmoisine)",
        "canonical_name_en": "Azorubine (Carmoisine)",
        "synonyms": ["carmoisine", "CI 14720", "đỏ azo 122"],
        "function": "Phẩm màu tổng hợp azo (Synthetic azo color)",
        "adi_value": "JECFA ADI: 0-4 mg/kg bw/ngày",
        "adi_upper": 4.0,
        "unverified_legacy_health_claims": [
            {"condition": "children_adhd", "severity": "WARNING", "message": "Thuộc nhóm 6 phẩm màu Southampton có liên quan đến tăng động ở trẻ em", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False},
            {"condition": "asthma_allergy", "severity": "CAUTION", "message": "Phẩm màu azo có thể gây phản ứng ở người nhạy cảm aspirin/hen suyễn", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 124",
        "e_number": "E124",
        "canonical_name_vi": "Ponceau 4R (Đỏ cochineal A)",
        "canonical_name_en": "Ponceau 4R (Cochineal Red A)",
        "synonyms": ["ponceau 4R", "CI 16255", "đỏ ponceau"],
        "function": "Phẩm màu tổng hợp azo (Synthetic azo color)",
        "adi_value": "JECFA ADI: 0-4 mg/kg bw/ngày",
        "adi_upper": 4.0,
        "unverified_legacy_health_claims": [
            {"condition": "children_adhd", "severity": "WARNING", "message": "Thuộc nhóm Southampton — gây tăng động ở trẻ nhạy cảm", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False},
            {"condition": "asthma_allergy", "severity": "CAUTION", "message": "Gây phản ứng dị ứng chéo ở người nhạy cảm aspirin", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 104",
        "e_number": "E104",
        "canonical_name_vi": "Quinoline Yellow",
        "canonical_name_en": "Quinoline Yellow",
        "synonyms": ["CI 47005", "vàng quinoline"],
        "function": "Phẩm màu tổng hợp (Synthetic color)",
        "adi_value": "JECFA ADI: 0-3 mg/kg bw/ngày",
        "adi_upper": 3.0,
        "unverified_legacy_health_claims": [
            {"condition": "children_adhd", "severity": "WARNING", "message": "Thuộc nhóm Southampton — ảnh hưởng hành vi trẻ em", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 171",
        "e_number": "E171",
        "canonical_name_vi": "Titan dioxide",
        "canonical_name_en": "Titanium dioxide",
        "synonyms": ["titanium dioxide", "TiO2", "CI 77891"],
        "function": "Phẩm màu trắng, chất phủ (White pigment, coating)",
        "adi_value": "EFSA 2021: Không còn coi là an toàn (no longer considered safe); EU cấm từ 2022",
        "adi_upper": None,
        "unverified_legacy_health_claims": [
            {"condition": "pregnancy", "severity": "WARNING", "message": "EFSA kết luận không thể loại trừ nguy cơ độc tính gen; EU đã cấm E171 trong thực phẩm từ 08/2022", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False},
            {"condition": "children_adhd", "severity": "WARNING", "message": "Trẻ em tiếp xúc qua kẹo, bánh phủ trắng — EU đã cấm", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 262(i)",
        "e_number": "E262(i)",
        "canonical_name_vi": "Natri acetat",
        "canonical_name_en": "Sodium acetate",
        "synonyms": ["sodium acetate", "natri axetat"],
        "function": "Chất điều chỉnh độ chua, chất bảo quản (Acidity regulator, preservative)",
        "adi_value": "JECFA: ADI không xác định (Not specified — an toàn ở mức GMP)",
        "adi_upper": None,
        "unverified_legacy_health_claims": [
            {"condition": "hypertension", "severity": "CAUTION", "message": "Chứa natri; cần tính vào tổng lượng natri nạp hàng ngày", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
    {
        "ins_code": "INS 341(i)",
        "e_number": "E341(i)",
        "canonical_name_vi": "Monocalci phosphat",
        "canonical_name_en": "Monocalcium phosphate",
        "synonyms": ["calcium dihydrogen phosphate", "MCP", "bột nở"],
        "function": "Chất nở, chất điều chỉnh độ chua (Raising agent, acidity regulator)",
        "adi_value": "EFSA group ADI: 40 mg/kg bw/ngày (tính theo phosphorus)",
        "adi_upper": 40.0,
        "unverified_legacy_health_claims": [
            {"condition": "chronic_kidney_disease", "severity": "WARNING", "message": "Phosphate vô cơ gây tăng phospho máu ở bệnh nhân thận mạn", "verification_status": "pending_primary_source_check", "use_for_personalized_alerts": False}
        ],
        "health_considerations": [],
        "allergen_groups": [],
    },
]


def build_full_entry(raw):
    """Bổ sung các trường chuẩn schema cho entry mới."""
    entry = {
        "ins_code": raw["ins_code"],
        "e_number": raw["e_number"],
        "canonical_name_vi": raw["canonical_name_vi"],
        "canonical_name_en": raw["canonical_name_en"],
        "synonyms": raw.get("synonyms", []),
        "function": raw["function"],
        "adi_value": raw.get("adi_value", ""),
        "adi_upper": raw.get("adi_upper"),
        "adi_source": raw.get("adi_source", ""),
        "vn_regulation": "Pending official-source review; not for compliance decisions.",
        "codex_standard": "Pending substance, food-category and limit verification.",
        "allergen_groups": raw.get("allergen_groups", []),
        "contraindications": [],
        "unverified_legacy_regulatory_claims": {
            "claims": {
                "vietnam": "Pending verification — TT 24/2019/TT-BYT",
                "codex": "Pending verification — Codex STAN 192-1995"
            },
            "verification_status": "pending_primary_source_check",
            "use_for_compliance_decision": False
        },
        "unverified_legacy_health_claims": raw.get("unverified_legacy_health_claims", []),
        "health_considerations": raw.get("health_considerations", []),
        "health_claim_policy": "Only source-checked, explicitly scoped considerations may be shown. Do not infer individual safety, diagnosis, or treatment from an ingredient match.",
        "regulatory_claim": {
            "jurisdiction": "Vietnam",
            "claim": "Pending verification — TT 24/2019/TT-BYT",
            "verification_status": "pending_official_text_and_food_category_check",
            "source_id": "VN_TT24_2019",
            "source_url": "",
            "use_for_compliance_decision": False
        },
        "codex_claim": {
            "claim": "Pending verification — Codex STAN 192-1995",
            "verification_status": "pending_gsfa_food_category_and_maximum_level_check",
            "source_id": "CODEX_GSFA",
            "source_url": "https://www.fao.org/gsfaonline/",
            "use_for_compliance_decision": False
        },
        "adi_assessment": {
            "display_text": raw.get("adi_value", ""),
            "numeric_upper_mg_per_kg_bw_day": raw.get("adi_upper"),
            "authority": "JECFA/EFSA",
            "source_id": "",
            "source_url": "",
            "source_locator": raw.get("adi_value", ""),
            "verification_status": "pending_primary_source_check",
            "use_for_numeric_compliance": False,
            "review_note": "Verify against primary evaluation before use."
        },
        "allergen_assessment": {
            "groups": raw.get("allergen_groups", []),
            "verification_status": "not_individually_assessed",
            "source_ids": ["CODEX_CXS_1_1985", "EU_1169_2011"],
            "note": "An empty list means no allergen was recorded here, not proof of absence."
        },
        "sources": []
    }
    return entry


def main():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)

    existing_codes = {a["ins_code"] for a in existing}
    added = 0
    skipped = 0

    for raw in NEW_ADDITIVES:
        if raw["ins_code"] in existing_codes:
            print(f"  SKIP (already exists): {raw['ins_code']}")
            skipped += 1
            continue
        entry = build_full_entry(raw)
        existing.append(entry)
        existing_codes.add(raw["ins_code"])
        added += 1
        print(f"  ADD: {raw['ins_code']} — {raw['canonical_name_en']}")

    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Kết quả: Thêm {added} phụ gia, bỏ qua {skipped} (đã tồn tại)")
    print(f"Tổng số phụ gia trong KB: {len(existing)}")
    print(f"File: {KB_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
