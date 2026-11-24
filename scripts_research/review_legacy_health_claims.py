"""Rà soát 72 "unverified_legacy_health_claims" của knowledge_base/additives_standard.json với văn bản nguồn đã lưu.

Mỗi tuyên bố cũ được xử lý đúng một trong các cách (ghi trong knowledge_base/sources/derived/legacy_health_claims_review.json):
  covered      – ý đúng đã có trong health_considerations đã kiểm; chỉ thêm bằng chứng trích nguyên văn
  promoted     – phần có nguồn được đưa vào health_considerations (kèm trích nguyên văn + trang)
  removed      – không có nguồn trong KB, hoặc nguồn không nói điều đó: bỏ khỏi KB (vẫn lưu lại ở file rà soát cùng lý do)
Mọi câu "evidence.quote" phải là đoạn NGUYÊN VĂN của đúng trang trong file nguồn (tests/test_legacy_health_review.py kiểm lại).

Chạy: .\venv\Scripts\python.exe scripts_research\review_legacy_health_claims.py [--apply]
Không có --apply: chỉ kiểm các trích dẫn và in tóm tắt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "knowledge_base" / "additives_standard.json"
CATALOG = ROOT / "knowledge_base" / "sources" / "catalog" / "source_citation_catalog.json"
OUT = ROOT / "knowledge_base" / "sources" / "derived" / "legacy_health_claims_review.json"
EXT = ROOT / "knowledge_base" / "sources" / "raw" / "external"


def squash(text: str) -> str:
    return " ".join(text.split())


_PAGES: dict[str, list[str]] = {}


def pages_of(name: str) -> list[str]:
    if name not in _PAGES:
        path = EXT / name
        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as doc:
                _PAGES[name] = [squash(p.get_text()) for p in doc]
        else:
            raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", path.read_text(encoding="utf-8", errors="ignore"))
            _PAGES[name] = [squash(re.sub(r"<[^>]+>", " ", raw))]
    return _PAGES[name]


def evidence(name: str, quote: str) -> dict:
    """Tìm trang chứa đúng đoạn trích nguyên văn; không thấy thì dừng (không ghi trích dẫn không kiểm được)."""
    for number, text in enumerate(pages_of(name), 1):
        if squash(quote) in text:
            return {"source_file": f"knowledge_base/sources/raw/external/{name}", "page": number, "quote": squash(quote)}
    raise SystemExit(f"Không thấy trích dẫn trong {name}: {quote[:80]!r}")


def catalog_url(citation_id: str, fallback: str) -> str:
    for entry in json.loads(CATALOG.read_text(encoding="utf-8")).values():
        if entry.get("citation_id") == citation_id and entry.get("url"):
            return entry["url"]
    return fallback


Q = {
    "sulfite_asthma": ("efsa_2016_4438.pdf", "It is estimated that 3–10% of all asthmatics are sensitive to sulfites."),
    "sulfite_mostly": ("efsa_2016_4438.pdf", "Sulfite sensitivity occurs mostly in asthmatics and may occur in a small number of non-asthmatic individuals."),
    "sulfite_moe": ("efsa_2022_7594.pdf", "MOEs at the maximum of 95th percentile ranges were below 80 for all population groups"),
    "nitrite_infants": ("efsa_2017_4786.pdf", "Nitrite should not be used as an additive in food for infants below the age of 3 months."),
    "nitrite_adi": ("efsa_2017_4786.pdf", "the proposed ADI for nitrite (0.07 mg/kg bw per day nitrite ion)"),
    "nitrate_mechanism": ("efsa_2017_4787.pdf", "the most relevant approach for assessing the toxicity of nitrate would be methaemoglobinaemia induced by nitrite formed from nitrate excreted in the saliva"),
    "nitrate_adi": ("efsa_2017_4787.pdf", "The current acceptable daily intakes (ADIs) for nitrate of 3.7 mg/kg body weight (bw) per day were established by the SCF (1997) and JECFA (2002)."),
    "phos_renal": ("efsa_2019_5674.pdf", "this ADI does not apply to humans with moderate to severe reduction in renal function."),
    "phos_ten_percent": ("efsa_2019_5674.pdf", "Ten per cent of general population might have chronic kidney disease with reduced renal function and they may not tolerate the amount of P per day which is at the level of ADI."),
    "polyol_label": ("EU_REG_1333_2008_ANNEX_V.pdf", "excessive consumption may induce laxative effects"),
    "polyol_scope": ("EU_REG_1333_2008_ANNEX_V.pdf", "The labelling of a table-top sweetener containing polyols and/or aspartame and/or aspartame-acesulfame salt shall bear the following warnings"),
    "benzene_form": ("FDA_BENZENE_QA.html", "Benzene can form at very low levels (ppb level) in some beverages that contain both benzoate salts and ascorbic acid (vitamin C) or erythorbic acid"),
    "benzene_heat": ("FDA_BENZENE_QA.html", "Exposure to heat and light can stimulate the formation of benzene in some beverages that contain benzoate salts and ascorbic acid (vitamin C)."),
    "benzene_cancer": ("FDA_BENZENE_QA.html", "Benzene is a carcinogen that can cause cancer in humans."),
    "pku_label": ("US_CFR_21_172_804.html", "PHENYLKETONURICS: CONTAINS PHENYLALANINE"),
    "tio2_unsafe": ("efsa_2021_6585.pdf", "E 171 can no longer be considered as safe when used as a food additive"),
    "tio2_genotox": ("efsa_2021_6585.pdf", "a concern for genotoxicity of TiO2 particles cannot be ruled out"),
    "annex_v_children": ("EU_REG_1333_2008_ANNEX_V.pdf", "may have an adverse effect on activity and attention in children"),
    "bha_iarc": ("efsa_2011_2392.pdf", "classified BHA as possibly carcinogenic to humans (category 2B), based on inadequate evidence in humans and sufficient evidence in experimental animals"),
}


def ev(*keys):
    return [evidence(*Q[k]) for k in keys]


# Ghi lại thay đổi của health_considerations đã có: ins_code → (điều kiện, thông điệp mới, nguồn mới | None, các trích dẫn)
REWRITES = {
    "INS 220": ("asthma", "Theo EFSA 2016, mẫn cảm với sulfit chủ yếu gặp ở người hen suyễn và có thể gặp ở số ít người không hen; "
                "ước tính 3–10% người hen nhạy cảm với sulfit. EFSA 2022 rút bỏ ADI tạm thời và dùng biên độ phơi nhiễm (MoE): "
                "MoE dưới 80 ở mọi nhóm dân số (phơi nhiễm percentile 95 cao nhất, kịch bản tinh chỉnh).",
                None, ("sulfite_mostly", "sulfite_asthma", "sulfite_moe")),
    "INS 221": ("asthma", "Theo EFSA 2016, mẫn cảm với sulfit chủ yếu gặp ở người hen suyễn và có thể gặp ở số ít người không hen; "
                "ước tính 3–10% người hen nhạy cảm với sulfit. EFSA 2022 rút bỏ ADI tạm thời và dùng biên độ phơi nhiễm (MoE): "
                "MoE dưới 80 ở mọi nhóm dân số (phơi nhiễm percentile 95 cao nhất, kịch bản tinh chỉnh).",
                None, ("sulfite_mostly", "sulfite_asthma", "sulfite_moe")),
    "INS 250": ("infants_young_children", "EFSA 2017 xác định methaemoglobin tăng là tác dụng liên quan để xây dựng ADI nhóm cho nitrit "
                "(đề xuất 0,07 mg ion nitrit/kg thể trọng/ngày). JECFA nêu: không nên dùng nitrit làm phụ gia trong thực phẩm cho trẻ dưới 3 tháng "
                "và ADI không áp dụng cho nhóm tuổi này.", None, ("nitrite_adi", "nitrite_infants")),
    "INS 251": ("infants_young_children", "Nitrat do nước bọt tiết ra được chuyển thành nitrit; EFSA 2017 lấy methaemoglobin do nitrit hình thành từ "
                "nitrat làm hướng đánh giá độc tính phù hợp nhất. ADI nitrat hiện hành 3,7 mg ion nitrat/kg thể trọng/ngày do SCF (1997) và JECFA (2002) "
                "thiết lập.", None, ("nitrate_mechanism", "nitrate_adi")),
    "INS 420": ("general_population", "Liên minh châu Âu yêu cầu chất tạo ngọt dạng viên/bột (table-top) có polyol (gồm sorbitol E 420) phải ghi cảnh báo "
                "'excessive consumption may induce laxative effects' (dùng quá nhiều có thể gây nhuận tràng).",
                ("EU_REG_1333_2008_ANNEX_V", ""), ("polyol_scope", "polyol_label")),
    "INS 211": ("general_population", "FDA nêu: benzen có thể hình thành ở mức rất thấp (ppb) trong một số đồ uống chứa đồng thời muối benzoat và "
                "acid ascorbic (vitamin C) hoặc acid erythorbic; nhiệt và ánh sáng có thể thúc đẩy quá trình này. Benzen là chất gây ung thư ở người.",
                None, ("benzene_form", "benzene_heat", "benzene_cancer")),
    "INS 951": ("pku", "Aspartame chứa phenylalanin. FDA (21 CFR 172.804) yêu cầu nhãn ghi: 'PHENYLKETONURICS: CONTAINS PHENYLALANINE'. "
                "Đây là lưu ý trên nhãn, không phải chẩn đoán cá nhân.", ("US_FDA_21CFR172_804", ""), ("pku_label",)),
    "INS 171": (None, None, None, ("tio2_unsafe", "tio2_genotox")),
}
for code in ("INS 338", "INS 339(i)", "INS 450(i)", "INS 452(i)", "INS 341(i)"):
    REWRITES[code] = ("ckd", "EFSA 2019: ADI nhóm 40 mg/kg thể trọng/ngày (tính theo phospho) không áp dụng cho người giảm chức năng thận mức vừa đến nặng; "
                      "khoảng 10% dân số có thể mắc bệnh thận mạn với chức năng thận giảm và có thể không dung nạp lượng P ở mức ADI.",
                      ("EFSA_PHOSPHATES_2019", "https://doi.org/10.2903/j.efsa.2019.5674"), ("phos_renal", "phos_ten_percent"))
for code in ("INS 102", "INS 110", "INS 129", "INS 122", "INS 124", "INS 104"):
    REWRITES[code] = (None, None, None, ("annex_v_children",))

# Phần có nguồn của tuyên bố cũ được đưa vào health_considerations (chỉ điều kiện 'general_population': không tạo cảnh báo theo bệnh lý)
PROMOTIONS = {
    "INS 320": {"condition": "general_population", "severity": "CAUTION",
                "message": "IARC xếp BHA vào nhóm 2B (có thể gây ung thư ở người), dựa trên bằng chứng chưa đủ ở người và đủ ở động vật thí nghiệm "
                           "(theo EFSA 2011). Tuyên bố cũ về thai phụ không có trong nguồn nên không giữ lại.",
                "source_id": "EFSA_2011_BHA_E320", "source_url": "https://doi.org/10.2903/j.efsa.2011.2392", "keys": ("bha_iarc",)},
}

R, C, P = "removed", "covered", "promoted"
NO_SRC = "Không có nguồn nào trong KB nói điều này."
# số thứ tự = thứ tự duyệt các mục KB (xem in ra bằng --apply)
DECISIONS = {
    1: (R, "EFSA 2017 (glutamat) chỉ ghi huyết áp tăng ở một liều thử nghiệm trên động vật; không có khuyến cáo theo hàm lượng natri."),
    2: (R, NO_SRC), 3: (C, "Đã có lưu ý PKU được kiểm nguồn; nay thêm trích nguyên văn quy định nhãn của 21 CFR 172.804."),
    4: (R, "IARC V134 không có khuyến cáo riêng cho thai phụ."), 5: (C, "Đã có lưu ý 'children' (EU Reg 1333/2008 Annex V) cho 6 màu Southampton."),
    6: (R, "EFSA 2009 chỉ nêu 'có thể gây phản ứng không dung nạp ở một phần nhỏ dân số'; không có kết luận riêng cho người hen."),
    7: (C, "Đã có allergen_assessment (nguồn gốc lecithin quyết định nguy cơ) với nguồn JEMRA/Codex."), 8: (C, "Như dòng 7."),
    9: (R, "Không phải tuyên bố sức khỏe ('rất an toàn'); nhãn WARNING sai nghĩa."), 10: (R, NO_SRC), 11: (R, NO_SRC),
    12: (R, "Nguồn FDA nói về benzen hình thành khi benzoat gặp acid ascorbic, không phải quá mẫn/hen; ý đó đã có trong lưu ý của INS 211."),
    13: (R, NO_SRC), 14: (R, NO_SRC), 15: (R, NO_SRC), 16: (R, NO_SRC),
    17: (R, "EFSA 2011 (caramel) không nói caramel ảnh hưởng hành vi trẻ."),
    18: (R, "EFSA 2011 chỉ nêu caramel nhóm IV được sản xuất với hợp chất sulfit; không kết luận về nguy cơ cho người nhạy sulfit."),
    19: (R, NO_SRC), 20: (R, NO_SRC), 21: (C, "Như dòng 5."), 22: (R, "EFSA 2014 (Sunset Yellow) không có đoạn tương ứng."),
    23: (C, "Như dòng 5."), 24: (R, "EFSA 2009 (Allura Red): có báo cáo phản ứng nhạy cảm nhưng 'chưa kết luận được' về khả năng gây nhạy cảm."),
    25: (R, "Không phải tuyên bố sức khỏe."), 26: (R, NO_SRC), 27: (R, "EFSA 2025 (acesulfame K) không có nội dung về tăng kali máu."),
    28: (R, "Không phải tuyên bố sức khỏe."), 29: (R, "Không phải tuyên bố sức khỏe."),
    30: (R, "Ý kiến EFSA 2018 về carrageenan không có trong KB; không kiểm được."), 31: (R, NO_SRC),
    32: (R, "Không phải tuyên bố sức khỏe."), 33: (R, NO_SRC), 34: (R, "Không phải tuyên bố sức khỏe."), 35: (R, NO_SRC),
    36: (R, "Không phải tuyên bố sức khỏe."), 37: (C, "Đã có lưu ý nhuận tràng của polyol; nay thay số liệu không kiểm được bằng quy định nhãn EU nguyên văn."),
    38: (R, NO_SRC),
    39: (C, "Đã có lưu ý hen suyễn–sulfit; số '5–10%', mức CRITICAL và 'sốc phản vệ' không có trong EFSA 2016 (ghi 3–10%), đã sửa theo nguồn."),
    40: (R, "EFSA 2016 chỉ có nghiên cứu trên động vật mang thai; không có khuyến cáo cho thai phụ."), 41: (C, "Như dòng 39."),
    42: (R, "EFSA 2017 không có khuyến cáo riêng cho thai phụ."), 43: (R, "EFSA 2017 không có khuyến cáo về thịt chế biến cho trẻ em."),
    44: (R, "EFSA 2017 không có khuyến cáo riêng cho thai phụ."),
    45: (P, "Phần IARC 2B có trong EFSA 2011 (đã đưa vào health_considerations); phần 'thai phụ nên hạn chế' không có trong nguồn."),
    46: (R, NO_SRC), 47: (R, "Nguồn về BHT không có trong KB."), 48: (R, "Nguồn về BHT không có trong KB."),
    49: (C, "Đã có lưu ý bệnh thận mạn; nay dùng EFSA 2019 có trích nguyên văn (KDOQI 2020 không có trong KB, số 90–100% không kiểm được)."),
    50: (R, "EFSA 2019 (mục 3.8.4) không kết luận phosphate làm tăng huyết áp."), 51: (C, "Như dòng 49."),
    52: (R, "Như dòng 50."), 53: (C, "Như dòng 49."), 54: (R, "Như dòng 50."), 55: (C, "Như dòng 49."),
    56: (R, NO_SRC), 57: (R, "Không thấy khuyến cáo riêng cho trẻ nhỏ trong bản WHO NSS đã lưu."),
    58: (R, "EFSA 2010 ghi không ảnh hưởng cân bằng glucose; phần 'cần theo dõi' không có trong nguồn."),
    59: (R, "Phần FDA về PKU của neotame không có trong KB; EFSA 2025 chỉ nói lượng phenylalanin sinh ra là an toàn cho dân số chung."),
    60: (R, NO_SRC), 61: (R, NO_SRC), 62: (R, "Brilliant Blue FCF không thuộc 6 màu Southampton."), 63: (R, NO_SRC),
    64: (C, "Như dòng 5."), 65: (R, "EFSA 2009 (Azorubine): có báo cáo phản ứng nhạy cảm nhưng chưa kết luận."),
    66: (C, "Như dòng 5."), 67: (R, "EFSA 2009 (Ponceau 4R): có nghiên cứu ở người hen nhạy aspirin nhưng chưa kết luận."),
    68: (C, "Như dòng 5."), 69: (C, "Đã có lưu ý TiO2 (EFSA 2021: không loại trừ độc tính gen); nguồn không có khuyến cáo riêng cho thai phụ."),
    70: (R, "Nguồn không có khuyến cáo riêng cho trẻ em."), 71: (R, NO_SRC), 72: (C, "Như dòng 49."),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    kb = json.loads(KB.read_text(encoding="utf-8"))
    legacy, n = [], 0
    for item in kb:
        for claim in item.get("unverified_legacy_health_claims") or []:
            n += 1
            legacy.append({"index": n, "ins_code": item["ins_code"], "condition": claim["condition"], "severity": claim["severity"],
                           "message": claim["message"]})
    assert n == 72 and set(DECISIONS) == set(range(1, 73)), (n, sorted(set(range(1, 73)) - set(DECISIONS)))
    assert (legacy[2]["ins_code"], legacy[38]["ins_code"], legacy[44]["ins_code"], legacy[68]["ins_code"]) == ("INS 951", "INS 220", "INS 320", "INS 171")
    by_code = {x["ins_code"]: x for x in kb}
    # 1) kiểm mọi trích dẫn ngay cả khi không --apply
    rewrites = {code: (cond, msg, src, ev(*keys)) for code, (cond, msg, src, keys) in REWRITES.items()}
    promotions = {code: {**spec, "evidence": ev(*spec["keys"])} for code, spec in PROMOTIONS.items()}
    for row in legacy:
        decision, reason = DECISIONS[row["index"]]
        row.update(decision=decision, reason=reason)
    counts = {d: sum(1 for r in legacy if r["decision"] == d) for d in (C, P, R)}
    print(f"72 tuyên bố cũ: giữ ý đã có {counts[C]}, đưa vào KB một phần {counts[P]}, bỏ {counts[R]}")
    print(f"Trích dẫn đã kiểm nguyên văn: {sum(len(v[3]) for v in rewrites.values()) + sum(len(v['evidence']) for v in promotions.values())}")
    if not args.apply:
        return 0

    # 2) sửa health_considerations đã có (giữ nguyên điều kiện/mức độ để không đổi hành vi cảnh báo theo bệnh lý)
    for code, (cond, message, source, quotes) in rewrites.items():
        entry = by_code[code]
        targets = [h for h in entry["health_considerations"] if cond is None or h["condition"] == cond]
        assert targets, (code, cond)
        for h in targets:
            if message:
                h["message_before_review"] = h["message"]
                h["message"] = message
                h["review_note"] = "Đã viết lại theo đúng văn bản nguồn đã lưu; số liệu/chi tiết không có trong nguồn đã bị bỏ."
            if source:
                h["source_id"] = source[0]
                h["source_url"] = source[1] or catalog_url(source[0], h.get("source_url", ""))
            h["evidence"] = quotes
    # 3) phần có nguồn của tuyên bố cũ
    for code, spec in promotions.items():
        entry = by_code[code]
        entry["health_considerations"].append({
            "condition": spec["condition"], "severity": spec["severity"], "message": spec["message"],
            "source_id": spec["source_id"], "source_url": spec["source_url"], "verification_status": "source_checked",
            "evidence": spec["evidence"]})
    # 4) gỡ khỏi KB, lưu lại file rà soát
    for item in kb:
        item["unverified_legacy_health_claims"] = []
    KB.write_text(json.dumps(kb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT.write_text(json.dumps({
        "note": "72 tuyên bố sức khỏe cũ (chưa kiểm nguồn) đã gỡ khỏi additives_standard.json ngày 2026-11-22. covered = ý đúng đã có trong "
                "health_considerations; promoted = phần có nguồn đã đưa vào KB; removed = không có nguồn/nguồn không nói điều đó.",
        "counts": counts, "claims": legacy}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Đã ghi {KB.name} và {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
