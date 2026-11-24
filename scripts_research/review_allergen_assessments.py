"""Rà lại allergen_assessment của TẤT CẢ phụ gia trong knowledge_base/additives_standard.json theo một phương pháp duy nhất.

Thay cho các ghi chú viết tay trước đây (trạng thái "source_checked" nhưng có nhận định lâm sàng/pháp lý không có trong nguồn
đã lưu, vd "EFSA GRAS", "FDA FALCPA miễn", "dị ứng chéo với lạc", "carmine OAS").

Căn cứ duy nhất: danh mục chất phải khai báo, trích NGUYÊN VĂN (kèm trang) từ
  - Codex CXS 1-1985 bản sửa 2024 (cxs_001e.pdf): bảng thực phẩm phải ghi tên (trang 6), sulphite (mục 4.2.1.7, trang 7)
  - Quy định (EU) 1169/2011 Phụ lục II, bản hợp nhất 01.04.2025 (CELEX_02011R1169_2025.pdf, trang 36–37), gồm chú thích (1):
    sản phẩm của siro glucose/maltodextrin lúa mì và của dầu/mỡ đậu nành tinh luyện hoàn toàn cũng được miễn
Mỗi phụ gia xếp theo danh tính hóa học vào:
  listed_substance          – chính chất này có tên trong danh mục (sulfit)
  not_from_listed_allergen  – nguyên liệu không thuộc nhóm nào, hoặc thuộc trường hợp được miễn
  source_dependent          – chỉ kết luận được khi nhãn ghi nguồn nguyên liệu (có ghi nhóm có thể liên quan)
  not_determinable          – KB chưa có tài liệu về nguyên liệu sản xuất
Trạng thái: "source_checked" chỉ cho chất có tên trong danh mục; còn lại "assessed_by_identity_against_allergen_lists"
(danh mục đã kiểm nguyên văn, nguồn gốc nguyên liệu suy theo danh tính hóa học, chưa kiểm tài liệu nhà sản xuất).
Ghi chú cũ được giữ ở "note_before_review". allergen_assessment không được pipeline/app đọc: không đổi nhãn hay bộ QA.

Chạy: .\\venv\\Scripts\\python.exe scripts_research\\review_allergen_assessments.py [--apply]
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
IDENTITY = "assessed_by_identity_against_allergen_lists"
METHOD = ("Đối chiếu danh tính hóa học với danh mục dị nguyên phải khai báo (Codex CXS 1 bản 2024; EU 1169/2011 Phụ lục II). "
          "Nguồn gốc nguyên liệu suy theo danh tính, chưa kiểm với đặc tả JECFA hay tài liệu nhà sản xuất.")
Q = {
    "codex_cereals": ("cxs_001e.pdf", "Cereals containing glutena − wheat and other Triticum species − rye and other Secale species − barley and other Hordeum species and products thereofb"),
    "codex_others": ("cxs_001e.pdf", "Crustacea and products thereof Crustacea Eggs and products thereof Egg Fish and products thereof Fish Peanuts and products thereof Peanut Milk and products thereof Milk Sesame and products thereof Sesame"),
    "codex_sulphite": ("cxs_001e.pdf", "Sulphite when present in concentrations of 10 mg/kg or more in a food,vii shall always be declared"),
    "eu_cereals": ("CELEX_02011R1169_2025.pdf", "Cereals containing gluten, namely: wheat (such as spelt and khorasan wheat), rye, barley, oats or their hybridised strains, and products thereof, except: ▼B (a) wheat based glucose syrups including dextrose"),
    "eu_soy": ("CELEX_02011R1169_2025.pdf", "Soybeans and products thereof, except: (a) fully refined soybean oil and fat"),
    "eu_footnote": ("CELEX_02011R1169_2025.pdf", "And the products thereof, in so far as the process that they have undergone is not likely to increase the level of allergenicity assessed by the Authority for the relevant product from which they originated."),
    "eu_mustard": ("CELEX_02011R1169_2025.pdf", "Mustard and products thereof, except: behenic acid with a minimum of 85 % of purity and obtained after two distillation steps used in the manufacturing of the emulsifiers E 470a, E 471 and E 477"),
    "eu_sulphite": ("CELEX_02011R1169_2025.pdf", "Sulphur dioxide and sulphites at concentrations of more than 10 mg/kg or 10 mg/litre in terms of the total SO 2"),
    "eu_phe": ("CELEX_02011R1169_2025.pdf", "‘contains a source of phenylalanine’ shall appear on the label in cases where aspartame/aspartame- acesulfame salt is designated in the list of ingredients by its specific name"),
}
LISTS = ["codex_others", "eu_soy"]
A, B, C, L = "not_from_listed_allergen", "source_dependent", "not_determinable", "listed_substance"

STARCH = ("Tinh bột biến tính có thể làm từ lúa mì. Hai danh mục coi 'ngũ cốc chứa gluten và sản phẩm của chúng' là chất phải khai báo; "
          "Phụ lục II chỉ miễn siro glucose/dextrose và maltodextrin từ lúa mì (và sản phẩm của chúng), không miễn tinh bột biến tính. "
          "Nhãn ghi nguồn ngô, khoai tây, sắn thì không thuộc danh mục; không ghi nguồn thì không kết luận được.")
SOY_EXEMPT = ("Nếu acid béo/glycerol lấy từ dầu đậu nành tinh luyện hoàn toàn thì được miễn theo Phụ lục II mục 6(a) và chú thích (1) "
              "(sản phẩm của dầu đậu nành tinh luyện, nếu quá trình chế biến không làm tăng tính gây dị ứng).")
MUSTARD = (" Riêng acid béo từ mù tạt: Phụ lục II chỉ miễn acid behenic tinh khiết ≥ 85%, chưng cất hai lần, dùng sản xuất E 470a, "
           "E 471, E 477; nguồn mù tạt khác thì phải khai báo. Nhãn không ghi nguồn thì không kết luận được.")
GLUCOSE = ("Làm từ glucose/carbohydrat. Nếu glucose từ lúa mì: Phụ lục II miễn siro glucose/dextrose từ lúa mì và sản phẩm của chúng "
           "(chú thích (1)).")
FISH = ("Nguyên liệu sản xuất (lên men, chiết từ cá hay thịt) chưa có tài liệu trong KB. Nếu từ cá thì thuộc nhóm 'cá và sản phẩm của "
        "cá' phải khai báo; thịt không thuộc danh mục.")
SULPHITE = ("Sulfit có tên trong cả hai danh mục: Codex yêu cầu ghi khi từ 10 mg/kg trở lên; EU yêu cầu ghi khi trên 10 mg/kg "
            "hoặc 10 mg/lít (tính theo SO2 tổng, ở dạng sẵn sàng sử dụng).")
NOT_LISTED = "Không thuộc nhóm nào của hai danh mục."

RULES = {
    # chất có tên trong danh mục
    "INS 220": (L, ["sulfit"], SULPHITE, ["codex_sulphite", "eu_sulphite"]),
    "INS 221": (L, ["sulfit"], SULPHITE, ["codex_sulphite", "eu_sulphite"]),
    # phụ thuộc nguồn hoặc lượng tồn dư
    "INS 150d": (B, ["sulfit"], "Caramel nhóm IV sản xuất có hợp chất sulfit; phải khai báo khi lượng sulfit tồn dư trong thực phẩm đạt "
                               "ngưỡng của danh mục. " + SULPHITE + " Không xác định được từ mã phụ gia, cần số liệu của sản phẩm.",
                 ["codex_sulphite", "eu_sulphite"]),
    "INS 322(i)": (B, ["đậu nành", "trứng"], "Lecithin có thể từ đậu nành, trứng hay hướng dương. Phụ lục II không miễn lecithin đậu nành "
                                               "(mục 6 chỉ miễn dầu/mỡ tinh luyện, tocopherol, phytosterol, stanol ester); trứng có trong cả "
                                               "hai danh mục. Nhãn phải ghi nguồn.", ["eu_soy", "codex_others"]),
    "INS 471": (B, ["mù tạt"], SOY_EXEMPT + MUSTARD, ["eu_soy", "eu_footnote", "eu_mustard"]),
    "INS 470(i)": (B, ["mù tạt"], SOY_EXEMPT + MUSTARD, ["eu_soy", "eu_footnote", "eu_mustard"]),
    "INS 477": (B, ["mù tạt"], SOY_EXEMPT + MUSTARD, ["eu_soy", "eu_footnote", "eu_mustard"]),
    "INS 631": (B, ["cá"], FISH, ["codex_others"]),
    "INS 627": (B, ["cá"], FISH, ["codex_others"]),
    "INS 621+631+627": (B, ["cá"], "Hỗn hợp: bột ngọt (621) không thuộc danh mục; 631 và 627: " + FISH, ["codex_others"]),
    "INS 635": (C, [], "Dinatri 5'-ribonucleotid: " + FISH, ["codex_others"]),
}
for code in ("INS 1400", "INS 1412", "INS 1414", "INS 1420", "INS 1422", "INS 1440", "INS 1442"):
    RULES[code] = (B, ["gluten (lúa mì)"], STARCH, ["codex_cereals", "eu_cereals"])
for code in ("INS 422", "INS 475", "INS 491", "INS 492"):
    RULES[code] = (A, [], SOY_EXEMPT + " Nguồn dầu khác (cọ, hướng dương...) không thuộc danh mục.", ["eu_soy", "eu_footnote"])
for code in ("INS 420(i)", "INS 420(ii)", "INS 575", "INS 150a", "INS 150c"):
    RULES[code] = (A, [], GLUCOSE, ["eu_cereals", "eu_footnote"])
IDENTITY_NOTES = {
    "INS 451(i)": "Muối phosphat vô cơ.", "INS 450(i)": "Muối phosphat vô cơ.", "INS 450(iii)": "Muối phosphat vô cơ.",
    "INS 452(i)": "Muối phosphat vô cơ.", "INS 338": "Acid vô cơ.", "INS 339(i)": "Muối phosphat vô cơ.", "INS 341(i)": "Muối phosphat vô cơ.",
    "INS 170(i)": "Khoáng chất vô cơ.", "INS 508": "Muối vô cơ.", "INS 500(i)": "Muối vô cơ.", "INS 500(ii)": "Muối vô cơ.",
    "INS 503(ii)": "Muối vô cơ.", "INS 551": "Khoáng chất vô cơ (silic dioxit).", "INS 171": "Khoáng chất vô cơ (titan dioxit).",
    "INS 250": "Muối vô cơ.", "INS 251": "Muối vô cơ.",
    "INS 319": "Chất tổng hợp hóa học.", "INS 320": "Chất tổng hợp hóa học.", "INS 321": "Chất tổng hợp hóa học.",
    "INS 1520": "Chất tổng hợp hóa học.", "INS 316": "Hợp chất xác định về hóa học.", "INS 296": "Acid malic dạng DL (tổng hợp).",
    "INS 260": "Acid acetic.", "INS 262(i)": "Muối của acid acetic.", "INS 270": "Acid lactic: tên 'lactic' không có nghĩa là làm từ sữa.",
    "INS 327": "Muối của acid lactic.", "INS 330": "Acid citric.", "INS 331(iii)": "Muối của acid citric.", "INS 300": "Acid ascorbic.",
    "INS 211": "Muối của acid benzoic.", "INS 202": "Muối của acid sorbic.",
    "INS 102": "Phẩm màu tổng hợp.", "INS 104": "Phẩm màu tổng hợp.", "INS 110": "Phẩm màu tổng hợp.", "INS 122": "Phẩm màu tổng hợp.",
    "INS 124": "Phẩm màu tổng hợp.", "INS 129": "Phẩm màu tổng hợp.", "INS 133": "Phẩm màu tổng hợp.",
    "INS 950": "Chất tạo ngọt tổng hợp.", "INS 955": "Chất tạo ngọt tổng hợp.", "INS 961": "Chất tạo ngọt tổng hợp.",
    "INS 951": "Chất tạo ngọt tổng hợp. Không phải dị nguyên, nhưng EU yêu cầu ghi 'chứa nguồn phenylalanin' (Phụ lục III, xem trích dẫn) "
               "— đây là yêu cầu cho người phenylketon niệu, không phải ghi nhãn dị nguyên.",
    "INS 621": "Bột ngọt (glutamat) không thuộc danh mục.",
    "INS 466": "Dẫn xuất cellulose (nguyên liệu thực vật như gỗ, bông).", "INS 460(i)": "Cellulose (nguyên liệu thực vật như gỗ, bông).",
    "INS 418": "Gôm gellan (lên men vi khuẩn).", "INS 415": "Gôm xanthan (lên men vi khuẩn).", "INS 903": "Sáp lá cọ carnauba.",
    "INS 412": "Gôm guar (hạt guar): danh mục không nêu.", "INS 410": "Gôm hạt carob: danh mục không nêu.",
    "INS 414": "Gôm arabic (cây keo): danh mục không nêu.", "INS 407": "Carrageenan (rong đỏ).", "INS 401": "Alginat (rong nâu).",
    "INS 440": "Pectin (vỏ cam quýt, bã táo).", "INS 960a": "Steviol glycosid (cây cỏ ngọt): danh mục không nêu.",
    "INS 160c(i)": "Oleoresin ớt paprika.", "INS 160a(i)": "Beta-caroten tổng hợp.", "INS 160a(ii)": "Caroten thực vật.",
    "INS 160a(iii)": "Beta-caroten từ nấm Blakeslea trispora.",
    "INS 120": "Carmin (côn trùng cochineal): danh mục không nêu; KB chưa có nguồn về mẫn cảm với carmin nên không kết luận thêm.",
    "INS 476": "Este polyglycerol của acid ricinoleic (dầu thầu dầu).",
}
for code, why in IDENTITY_NOTES.items():
    keys = LISTS + (["eu_phe"] if code == "INS 951" else [])
    RULES[code] = (A, [], why + " " + NOT_LISTED if "danh mục" not in why else why, keys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    kb = json.loads(KB.read_text(encoding="utf-8"))
    codes = [x["ins_code"] for x in kb]
    assert not (set(codes) - set(RULES)), f"Thiếu quy tắc: {sorted(set(codes) - set(RULES))}"
    quotes = {k: rl.evidence(*v) for k, v in Q.items()}
    summary = {}
    changed_class = []
    for item in kb:
        cls, possible, note, keys = RULES[item["ins_code"]]
        summary[cls] = summary.get(cls, 0) + 1
        old = item["allergen_assessment"]
        if old.get("classification") and old["classification"] != cls:
            changed_class.append((item["ins_code"], old["classification"], cls))
        if not args.apply:
            continue
        item["allergen_assessment"] = {
            "groups": possible if cls == L else [],
            "possible_groups_if_source_is_allergenic": possible if cls == B else [],
            "classification": cls,
            "verification_status": "source_checked" if cls == L else IDENTITY,
            "source_ids": ["CODEX_CXS_1_1985", "EU_REG_1169_2011"],
            "note": note,
            "evidence": [quotes[k] for k in keys],
            "method": METHOD,
            **({"note_before_review": old["note"]} if old.get("note") and "note_before_review" not in old and old.get("verification_status") != IDENTITY
               else ({"note_before_review": old["note_before_review"]} if old.get("note_before_review") else {})),
        }
    print(f"{len(kb)} phụ gia: " + ", ".join(f"{k} {v}" for k, v in sorted(summary.items())))
    if changed_class:
        print("Đổi phân loại so với lần đánh giá trước:", changed_class)
    if args.apply:
        KB.write_text(json.dumps(kb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Đã ghi additives_standard.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
