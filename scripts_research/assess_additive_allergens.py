"""[ĐÃ THAY THẾ bởi review_allergen_assessments.py — đánh giá lại TẤT CẢ phụ gia, có xét chú thích (1) của Phụ lục II.
Giữ lại để truy vết lần chạy ngày 2026-11-23; không chạy lại.]

Đánh giá dị nguyên cho các phụ gia có allergen_assessment = not_assessed trong knowledge_base/additives_standard.json.

Phương pháp (không có tài liệu nhà sản xuất, nên KHÔNG ghi "source_checked"):
  1. Danh mục chất phải khai báo dị nguyên lấy NGUYÊN VĂN từ hai nguồn đã lưu:
       - Codex CXS 1-1985 (bản sửa 2024), bảng thực phẩm và thành phần phải ghi tên (cxs_001e.pdf, trang 6)
       - Quy định (EU) 1169/2011, Phụ lục II, bản hợp nhất 01.04.2025 (CELEX_02011R1169_2025.pdf, trang 36–37)
  2. Mỗi phụ gia được xếp theo DANH TÍNH HÓA HỌC (tên trong TT24/JECFA) vào một trong ba nhóm:
       not_from_listed_allergen  – chất vô cơ, chất tổng hợp hoặc nguyên liệu không thuộc nhóm nào của hai danh mục
       source_dependent          – nguyên liệu có thể thuộc một nhóm trong danh mục (vd tinh bột từ lúa mì, acid béo từ đậu nành);
                                   nhãn phải ghi rõ nguồn mới kết luận được
       not_determinable          – nguồn nguyên liệu thường gặp chưa có tài liệu trong KB để xếp loại
  3. Trạng thái: "assessed_by_identity_against_allergen_lists" — đối chiếu danh mục đã kiểm, còn nguồn gốc nguyên liệu là
     suy theo danh tính hóa học, CHƯA kiểm với đặc tả JECFA hay tài liệu nhà sản xuất.
allergen_assessment hiện không được pipeline/app đọc (chỉ là dữ liệu KB), nên không đổi nhãn an toàn hay bộ QA.

Chạy: .\\venv\\Scripts\\python.exe scripts_research\\assess_additive_allergens.py [--apply]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_legacy_health_claims as rl  # noqa: E402

KB = ROOT / "knowledge_base" / "additives_standard.json"
STATUS = "assessed_by_identity_against_allergen_lists"
LIST_QUOTES = {
    "codex_cereals": ("cxs_001e.pdf", "Cereals containing glutena − wheat and other Triticum species − rye and other Secale species − barley and other Hordeum species and products thereofb"),
    "codex_others": ("cxs_001e.pdf", "Crustacea and products thereof Crustacea Eggs and products thereof Egg Fish and products thereof Fish Peanuts and products thereof Peanut Milk and products thereof Milk Sesame and products thereof Sesame"),
    "eu_cereals": ("CELEX_02011R1169_2025.pdf", "Cereals containing gluten, namely: wheat (such as spelt and khorasan wheat), rye, barley, oats or their hybridised strains, and products thereof, except: ▼B (a) wheat based glucose syrups including dextrose"),
    "eu_soy": ("CELEX_02011R1169_2025.pdf", "Soybeans and products thereof, except: (a) fully refined soybean oil and fat"),
    "eu_mustard": ("CELEX_02011R1169_2025.pdf", "Mustard and products thereof, except: behenic acid with a minimum of 85 % of purity and obtained after two distillation steps used in the manufacturing of the emulsifiers E 470a, E 471 and E 477"),
}
STARCH = ("Tinh bột biến tính có thể làm từ lúa mì. Hai danh mục coi 'ngũ cốc chứa gluten và sản phẩm của chúng' là chất phải khai báo; "
          "Phụ lục II chỉ miễn siro glucose/dextrose và maltodextrin từ lúa mì, không miễn tinh bột biến tính. Nếu nhãn ghi nguồn ngô, "
          "khoai tây, sắn thì không thuộc danh mục; không ghi nguồn thì không kết luận được.")
FATTY = ("Este/muối của acid béo: acid béo có thể lấy từ dầu đậu nành. Phụ lục II chỉ miễn 'dầu và mỡ đậu nành tinh luyện hoàn toàn', "
         "không nêu các chất dẫn xuất này; nhãn phải ghi nguồn dầu mới kết luận được.")
FATTY_MUSTARD = FATTY + (" Phụ lục II còn nêu acid behenic từ mù tạt dùng sản xuất E 470a/E 471/E 477 chỉ được miễn khi độ tinh khiết "
                         "≥ 85% và chưng cất hai lần.")
A = "not_from_listed_allergen"
B = "source_dependent"
C = "not_determinable"
# mã INS -> (nhóm, nhóm dị nguyên CÓ THỂ liên quan, lý do, các trích dẫn danh mục)
RULES = {
    "INS 451(i)": (A, [], "Muối phosphat vô cơ (pentanatri triphosphat).", ["codex_others", "eu_soy"]),
    "INS 450(iii)": (A, [], "Muối phosphat vô cơ (tetranatri diphosphat).", ["codex_others", "eu_soy"]),
    "INS 170(i)": (A, [], "Khoáng chất vô cơ (calci carbonat).", ["codex_others", "eu_soy"]),
    "INS 508": (A, [], "Muối vô cơ (kali clorid).", ["codex_others", "eu_soy"]),
    "INS 500(i)": (A, [], "Muối vô cơ (natri carbonat).", ["codex_others", "eu_soy"]),
    "INS 319": (A, [], "Chất chống oxy hóa tổng hợp hóa học (TBHQ).", ["codex_others", "eu_soy"]),
    "INS 1520": (A, [], "Hợp chất tổng hợp hóa học (propylen glycol).", ["codex_others", "eu_soy"]),
    "INS 316": (A, [], "Muối của acid erythorbic (hợp chất xác định về hóa học).", ["codex_others", "eu_soy"]),
    "INS 296": (A, [], "Acid malic dạng DL (dạng tổng hợp hóa học).", ["codex_others", "eu_soy"]),
    "INS 260": (A, [], "Acid acetic (hợp chất xác định về hóa học).", ["codex_others", "eu_soy"]),
    "INS 270": (A, [], "Acid lactic: tên 'lactic' không có nghĩa là làm từ sữa; danh mục không nêu acid lactic.", ["codex_others", "eu_soy"]),
    "INS 327": (A, [], "Calci lactat (muối của acid lactic); danh mục không nêu.", ["codex_others", "eu_soy"]),
    "INS 575": (A, [], "Glucono delta-lacton (từ glucose). Nếu glucose từ lúa mì: Phụ lục II miễn siro glucose/dextrose từ lúa mì.", ["eu_cereals"]),
    "INS 466": (A, [], "Dẫn xuất cellulose (nguyên liệu thực vật như gỗ, bông), không thuộc nhóm nào của danh mục.", ["codex_others", "eu_soy"]),
    "INS 460(i)": (A, [], "Cellulose vi tinh thể (nguyên liệu thực vật như gỗ, bông), không thuộc nhóm nào của danh mục.", ["codex_others", "eu_soy"]),
    "INS 418": (A, [], "Gôm gellan (sản phẩm lên men vi khuẩn), không thuộc nhóm nào của danh mục.", ["codex_others", "eu_soy"]),
    "INS 903": (A, [], "Sáp carnauba (lá cọ carnauba), không thuộc nhóm nào của danh mục.", ["codex_others", "eu_soy"]),
    "INS 160c(i)": (A, [], "Oleoresin ớt paprika, không thuộc nhóm nào của danh mục.", ["codex_others", "eu_soy"]),
    "INS 120": (A, [], "Carmin (từ côn trùng cochineal) không thuộc nhóm nào của danh mục. KB chưa có nguồn về mẫn cảm với carmin, "
                       "nên không kết luận thêm.", ["codex_others", "eu_soy"]),
    "INS 476": (A, [], "Este polyglycerol của acid ricinoleic (dầu thầu dầu), không thuộc nhóm nào của danh mục.", ["codex_others", "eu_soy"]),
    "INS 150a": (A, [], "Caramel từ đường/carbohydrat. Nếu làm từ glucose lúa mì: Phụ lục II miễn siro glucose/dextrose từ lúa mì.", ["eu_cereals"]),
    "INS 150c": (A, [], "Caramel từ đường/carbohydrat. Nếu làm từ glucose lúa mì: Phụ lục II miễn siro glucose/dextrose từ lúa mì.", ["eu_cereals"]),
    "INS 1422": (B, ["gluten (lúa mì)"], STARCH, ["codex_cereals", "eu_cereals"]),
    "INS 1414": (B, ["gluten (lúa mì)"], STARCH, ["codex_cereals", "eu_cereals"]),
    "INS 1412": (B, ["gluten (lúa mì)"], STARCH, ["codex_cereals", "eu_cereals"]),
    "INS 1440": (B, ["gluten (lúa mì)"], STARCH, ["codex_cereals", "eu_cereals"]),
    "INS 1442": (B, ["gluten (lúa mì)"], STARCH, ["codex_cereals", "eu_cereals"]),
    "INS 1400": (B, ["gluten (lúa mì)"], STARCH, ["codex_cereals", "eu_cereals"]),
    "INS 422": (B, ["đậu nành"], "Glycerol có thể sản xuất từ dầu/mỡ, trong đó có dầu đậu nành. " + FATTY.split(": ", 1)[1], ["eu_soy"]),
    "INS 475": (B, ["đậu nành"], FATTY, ["eu_soy"]),
    "INS 491": (B, ["đậu nành"], FATTY, ["eu_soy"]),
    "INS 492": (B, ["đậu nành"], FATTY, ["eu_soy"]),
    "INS 477": (B, ["đậu nành", "mù tạt"], FATTY_MUSTARD, ["eu_soy", "eu_mustard"]),
    "INS 470(i)": (B, ["đậu nành", "mù tạt"], FATTY_MUSTARD, ["eu_soy", "eu_mustard"]),
    "INS 635": (C, [], "Dinatri 5'-ribonucleotid: KB chưa có tài liệu về nguyên liệu sản xuất (lên men hay chiết xuất), nên chưa xếp loại được.", []),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    kb = json.loads(KB.read_text(encoding="utf-8"))
    pending = [x["ins_code"] for x in kb if x["allergen_assessment"]["verification_status"] == "not_assessed"]
    missing, extra = sorted(set(pending) - set(RULES)), sorted(set(RULES) - set(pending))
    assert not missing, f"Chưa có quy tắc cho: {missing}"
    quotes = {k: rl.evidence(*v) for k, v in LIST_QUOTES.items()}  # dừng nếu trích dẫn không nguyên văn
    counts = {}
    for code in pending:
        counts[RULES[code][0]] = counts.get(RULES[code][0], 0) + 1
    print(f"{len(pending)} phụ gia chưa đánh giá: " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + (f" | quy tắc thừa (mục đã đánh giá): {extra}" if extra else ""))
    if not args.apply:
        return 0
    for item in kb:
        if item["ins_code"] not in pending:
            continue
        cls, possible, reason, keys = RULES[item["ins_code"]]
        item["allergen_assessment"] = {
            "groups": [],  # không gán nhóm cứng: chỉ khi nhãn ghi nguồn mới kết luận
            "possible_groups_if_source_is_allergenic": possible,
            "classification": cls,
            "verification_status": STATUS,
            "source_ids": ["CODEX_CXS_1_1985", "EU_REG_1169_2011"],
            "note": reason,
            "evidence": [quotes[k] for k in keys],
            "method": "Đối chiếu danh tính hóa học với danh mục dị nguyên phải khai báo (Codex CXS 1 bản 2024, EU 1169/2011 Phụ lục II). "
                      "Nguồn gốc nguyên liệu suy theo danh tính, chưa kiểm với đặc tả JECFA hay tài liệu nhà sản xuất.",
        }
    KB.write_text(json.dumps(kb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Đã ghi additives_standard.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
