"""Nối 5 nguồn mới tải ngày 2026-11-23 vào KB: đăng ký manifest (sha256) và gắn trích dẫn nguyên văn.

  jecfa87_2019_summary.pdf     JECFA kỳ họp 87 (tóm tắt): ADI beta-carotene bị rút bỏ; nguy cơ ở người hút thuốc nặng
  CELEX_02011R1169_2025.pdf    Quy định (EU) 1169/2011, bản hợp nhất 01.04.2025: cảnh báo polyol >10%
  KDOQI_Nutrition_2020.pdf     KDOQI dinh dưỡng trong CKD 2020: cân nhắc phụ gia phospho khi hạn chế phospho
  KDIGO_2024_CKD.pdf, ada_standards_2023_abridged.pdf   chỉ đăng ký (xem ghi chú): chưa có câu nào liên quan phụ gia trong KB

Mọi trích dẫn phải là nguyên văn của đúng trang (review_legacy_health_claims.evidence dừng nếu không thấy).
Chạy: .\\venv\\Scripts\\python.exe scripts_research\\apply_new_sources_2026_11_23.py [--apply]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_legacy_health_claims as rl  # noqa: E402

KB = ROOT / "knowledge_base" / "additives_standard.json"
MANIFEST = ROOT / "knowledge_base" / "sources" / "catalog" / "external_sources_manifest.json"
EXT = ROOT / "knowledge_base" / "sources" / "raw" / "external"

FILES = {
    "jecfa87_2019_summary.pdf": "JECFA kỳ họp 87 (Rome, 4–13/6/2019), Summary and Conclusions, 15 trang",
    "CELEX_02011R1169_2025.pdf": "Quy định (EU) 1169/2011 về thông tin thực phẩm cho người tiêu dùng, bản hợp nhất 01.04.2025 (EUR-Lex, không có hiệu lực pháp lý)",
    "KDOQI_Nutrition_2020.pdf": "KDOQI Clinical Practice Guideline for Nutrition in CKD: 2020 Update (Ikizler và cộng sự), 109 trang",
    # KDIGO_2024_CKD.pdf: đã dời vào _archive_unused/sources/ (2026-11-23), không câu nào gắn phụ gia đủ dùng
    # "KDIGO_2024_CKD.pdf": "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of CKD (Kidney Int 2024;105(4S)), 199 trang",
    # ada_standards_2023_abridged.pdf: đã dời vào _archive_unused/sources/ (2026-11-23), bản 2023 rút gọn không có nội dung về chất tạo ngọt
    # "ada_standards_2023_abridged.pdf": "ADA Standards of Care in Diabetes—2023 Abridged for Primary Care Providers, 28 trang",
}
Q = {
    "bc_withdrawn": ("jecfa87_2019_summary.pdf", "Therefore, the Committee withdrew the two group ADIs of 0–5 mg/kg bw for (1) the sum of the synthetic carotenoids"),
    "bc_general_pop": ("jecfa87_2019_summary.pdf", "because a group ADI is applicable to the general population, which includes heavy smokers."),
    "bc_smokers": ("jecfa87_2019_summary.pdf", "An additional elevated risk of lung cancer and total mortality was seen in heavy smokers (at least one pack per day) and asbestos workers in intervention studies in which participants were administered 20 mg"),
    "polyol_10": ("CELEX_02011R1169_2025.pdf", "Foods containing more than 10 % added polyols authorised pursuant to Regulation (EC) No 1333/ 2008"),
    "polyol_label": ("CELEX_02011R1169_2025.pdf", "excessive consumption may produce laxative effects"),
    "kdoqi_sources": ("KDOQI_Nutrition_2020.pdf", "it is reasonable when making decisions about phosphorus restriction treatment to consider the bioavailability of phosphorus sources (eg, animal, vegetable, additives) (OPINION)"),
    "kdoqi_processed": ("KDOQI_Nutrition_2020.pdf", "the use of phosphorus additives in processed foods"),
}
KDOQI_SOURCE = {"id": "KDOQI_2020_NUTRITION_CKD", "title": "KDOQI Clinical Practice Guideline for Nutrition in CKD: 2020 Update",
                "url": "https://doi.org/10.1053/j.ajkd.2020.05.006",
                "locator": "Am J Kidney Dis 2020;76(3 Suppl 1):S1–S107, khuyến nghị 6.3.2 (file KDOQI_Nutrition_2020.pdf, trang 23)"}
POLYOL_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02011R1169-20250401"


def ev(*keys):
    return [rl.evidence(*Q[k]) for k in keys]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    kb = json.loads(KB.read_text(encoding="utf-8"))
    by = {x["ins_code"]: x for x in kb}
    quotes = {k: rl.evidence(*v) for k, v in Q.items()}
    print(f"Trích dẫn nguyên văn đã kiểm: {len(quotes)} — " + ", ".join(f"{k} p.{v['page']}" for k, v in quotes.items()))
    if not args.apply:
        return 0

    # 1) beta-carotene: bằng chứng cho ADI rút bỏ + lưu ý người hút thuốc nặng (điều kiện 'smokers' đúng như nguồn nói)
    for code in ("INS 160a(i)", "INS 160a(iii)"):
        item = by[code]
        item["adi_assessment"]["evidence"] = ev("bc_withdrawn", "bc_general_pop")
        item["adi_assessment"]["review_note"] = ("Giá trị lấy từ trang JECFA đã lưu và đã đối chiếu với tóm tắt kỳ họp 87 (jecfa87_2019_summary.pdf, "
                                                 "trang 2–3): ADI nhóm 0–5 mg/kg bị rút bỏ năm 2019, không thể thiết lập lại.")
        item["health_considerations"].append({
            "condition": "smokers", "severity": "CAUTION",
            "message": "JECFA 2019: trong các nghiên cứu can thiệp bổ sung 20 mg beta-carotene/ngày (5–8 năm), người hút thuốc nặng (từ một gói/ngày) "
                       "và công nhân tiếp xúc amiăng có thêm nguy cơ ung thư phổi và tử vong chung; vì vậy JECFA không thiết lập được ADI nhóm vì ADI "
                       "áp dụng cho dân số chung, trong đó có người hút thuốc nặng.",
            "source_id": "JECFA_BETA_CAROTENE_2019",
            "source_url": "https://cdn.who.int/media/docs/default-source/food-safety/jecfa/summary-and-conclusions/jecfa87_4-13-june-2019_summary-and-conclusion.pdf",
            "verification_status": "source_checked", "evidence": ev("bc_smokers", "bc_general_pop")})

    # 2) polyol (sorbitol): số liệu 10% và câu cảnh báo nay có văn bản gốc của 1169/2011
    for code in ("INS 420(i)", "INS 420(ii)"):
        for h in by[code]["health_considerations"]:
            if h["condition"] != "general_population":
                continue
            h.setdefault("message_before_review_2026_11_23", h["message"])
            h["message"] = ("Quy định (EU) 1169/2011 (Phụ lục III, mục 2.4) yêu cầu thực phẩm chứa trên 10% polyol bổ sung (gồm sorbitol) ghi: "
                            "'excessive consumption may produce laxative effects' (dùng quá nhiều có thể gây nhuận tràng). Quy định 1333/2008 yêu cầu "
                            "cùng cảnh báo cho chất tạo ngọt dạng viên/bột chứa polyol.")
            h["source_id"], h["source_url"] = "EU_REG_1169_2011", POLYOL_URL
            h["evidence"] = ev("polyol_10", "polyol_label") + [e for e in h.get("evidence", []) if "EU_REG_1333" in e["source_file"]]

    # 3) phosphat + CKD: KDOQI nay có file nên dùng làm bằng chứng bổ sung (giữ nguyên điều kiện 'ckd' và nguồn chính EFSA 2019)
    for code in ("INS 338", "INS 339(i)", "INS 341(i)", "INS 450(i)", "INS 452(i)"):
        item = by[code]
        for h in item["health_considerations"]:
            if h["condition"] == "ckd":
                h["evidence"] = h.get("evidence", []) + ev("kdoqi_sources", "kdoqi_processed")
                h["message"] = h["message"].rstrip() + (" KDOQI 2020 (khuyến nghị 6.3.2): khi quyết định hạn chế phospho, nên cân nhắc sinh khả dụng của "
                                                       "nguồn phospho (động vật, thực vật, phụ gia).")
        if not any(s["id"] == KDOQI_SOURCE["id"] for s in item["sources"]):
            item["sources"].append(copy.deepcopy(KDOQI_SOURCE))
    # 450(iii) cùng nhóm phosphat E 450–452 của EFSA 2019: cùng lưu ý bệnh thận mạn như các phosphat anh em
    template = next(h for h in by["INS 450(i)"]["health_considerations"] if h["condition"] == "ckd")
    if not any(h["condition"] == "ckd" for h in by["INS 450(iii)"]["health_considerations"]):
        by["INS 450(iii)"]["health_considerations"].append(copy.deepcopy(template))
        by["INS 450(iii)"]["sources"].append(copy.deepcopy(KDOQI_SOURCE))
    KB.write_text(json.dumps(kb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 4) manifest
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    have = {x["file"] for x in manifest["manually_downloaded"]}
    for name, description in FILES.items():
        if name in have:
            continue
        data = (EXT / name).read_bytes()
        manifest["manually_downloaded"].append({"file": name, "description": description, "bytes": len(data),
                                                "sha256": hashlib.sha256(data).hexdigest(), "obtained": "downloaded manually by the user"})
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã ghi KB và manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
