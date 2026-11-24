"""Ghi các nguồn cho nhóm trẻ dưới 12 tháng (người dùng tải tay) vào manifest và danh mục khóa trích dẫn.

- external_sources_manifest.json / manually_downloaded: tên tệp, kích thước, SHA-256 (chỉ thêm tệp chưa có).
- source_citation_catalog.json: chỉ thêm khóa còn thiếu, không ghi đè khóa đã có.
Vị trí (mục, trang PDF) đã đối chiếu với văn bản trích từ chính tệp gốc (PyMuPDF cho PDF, antiword cho .doc).
ND_1002014NĐ_CP.pdf là bản quét (không có lớp chữ) nên chỉ ghi manifest, chưa dùng làm trích dẫn.

Usage: python scripts_research/register_infant_sources.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "knowledge_base" / "sources" / "raw" / "external"
CAT = ROOT / "knowledge_base" / "sources" / "catalog"
OBTAINED = "downloaded manually by the user"

FILES = {
    "complementary_feeding_infants_young_children_6_23_months.pdf":
        "WHO Guideline for complementary feeding of infants and young children 6-23 months of age (2023), ISBN 978-92-4-008186-4",
    "Codex_CXS_72_1981.pdf": "Codex CXS 72-1981 Standard for Infant Formula and Formulas for Special Medical Purposes Intended for Infants (amended 2024)",
    "CXS_74_1981.pdf": "Codex STAN 074-1981, Rev. 1-2006 Processed Cereal-Based Foods for Infants and Young Children",
    "QCVN_11_1_2012_BYT.pdf": "QCVN 11-1:2012/BYT Sản phẩm dinh dưỡng công thức cho trẻ đến 12 tháng tuổi",
    "QCVN_11_2_2012_BYT.doc": "QCVN 11-2:2012/BYT (TT 21/2012/TT-BYT) Sản phẩm dinh dưỡng công thức với mục đích y tế đặc biệt cho trẻ đến 12 tháng tuổi",
    "QCVN_11_3_2012_BYT.doc": "QCVN 11-3:2012/BYT (TT 22/2012/TT-BYT) Sản phẩm dinh dưỡng công thức với mục đích ăn bổ sung cho trẻ từ 6 đến 36 tháng tuổi",
    "QCVN_11_4_2012_BYT.doc": "QCVN 11-4:2012/BYT (TT 23/2012/TT-BYT) Sản phẩm dinh dưỡng chế biến từ ngũ cốc cho trẻ từ 6 đến 36 tháng tuổi",
    "ND_1002014NĐ_CP.pdf": "Nghị định 100/2014/NĐ-CP (bản quét, không có lớp chữ; chưa dùng làm trích dẫn)",
}

CATALOG = {
    "VN_TT24_2019#PhuLuc2A_Nhom13.1_13.2": ("VN_TT24_2019", "Thông tư 24/2019/TT-BYT & VBHN 09/VBHN-BYT 2024",
        "Phụ lục 2A, các dòng nhóm thực phẩm 13.1.1, 13.1.2, 13.1.3, 13.2 (PDF trang 46–357): phụ gia và mức tối đa cho thực phẩm dành cho trẻ."),
    "VN_TT24_2019#PhuLuc3_Muc2_GMP": ("VN_TT24_2019", "Thông tư 24/2019/TT-BYT & VBHN 09/VBHN-BYT 2024",
        "Phụ lục 3 mục 2 (PDF trang 415–418): danh sách nhóm thực phẩm dùng phụ gia theo GMP không có nhóm 13.1 và 13.2."),
    "CODEX_STAN_192_1995#Table2_Cat13.1_13.2": ("CODEX_STAN_192_1995", "Codex General Standard for Food Additives CXS 192-1995",
        "Table Two, food categories 13.1.1, 13.1.2, 13.1.3, 13.2: additives permitted in foods for infants and young children."),
    "CODEX_STAN_192_1995#AnnexTable3_Excl13": ("CODEX_STAN_192_1995", "Codex General Standard for Food Additives CXS 192-1995",
        "Annex to Table Three (PDF page 493): categories 13.1 and 13.2 are excluded from the general GMP conditions of Table Three."),
    "QCVN_11_1_2012#Muc1.3.4": ("VN_QCVN_11_1_2012", FILES["QCVN_11_1_2012_BYT.pdf"],
        "Mục 1.3.4 (PDF trang 7–8): các dạng vitamin, chất khoáng bổ sung theo quy định Bộ Y tế, nếu chưa có thì theo CAC/GL 10-1979 (danh mục hợp chất vitamin và khoáng cho trẻ)."),
    "QCVN_11_1_2012#Muc6_PhuGia": ("VN_QCVN_11_1_2012", FILES["QCVN_11_1_2012_BYT.pdf"],
        "Mục 6 (PDF trang 11): phụ gia dùng trong sản phẩm dinh dưỡng công thức cho trẻ đến 12 tháng tuổi theo quy định của Bộ Y tế."),
    "QCVN_11_4_2012#Muc6_PhuGia": ("VN_QCVN_11_4_2012", FILES["QCVN_11_4_2012_BYT.doc"],
        "Mục 6: phụ gia dùng trong sản phẩm dinh dưỡng từ ngũ cốc cho trẻ 6–36 tháng tuổi theo quy định của Bộ Y tế."),
    "CODEX_CXS_72_1981#SecA3.4": ("CODEX_CXS_72_1981", FILES["Codex_CXS_72_1981.pdf"],
        "Section A 3.4 (PDF page 9): vitamin compounds and mineral salts are selected from the advisory lists of nutrient compounds (CXG 10-1979)."),
    "CODEX_CXS_74_1981#Sec8.6.4": ("CODEX_CXS_74_1981", FILES["CXS_74_1981.pdf"],
        "Section 8.6.4 (PDF page 8): the label shall indicate clearly from which age the product is recommended; not less than six months."),
    "CODEX_CXS_74_1981#Sec8.3.2": ("CODEX_CXS_74_1981", FILES["CXS_74_1981.pdf"],
        "Section 8.3.2 (PDF page 8): the specific name shall be declared for ingredients and food additives."),
    "WHO_CF_2023#Rec2_Milks": ("WHO_COMPLEMENTARY_FEEDING_2023", FILES["complementary_feeding_infants_young_children_6_23_months.pdf"],
        "Recommendation 2 and remarks (PDF page 12): sweetened milks are not appropriate for 6–23 months; flavoured or sweetened milks should not be used."),
    "WHO_CF_2023#Rec5_UnhealthyFoods": ("WHO_COMPLEMENTARY_FEEDING_2023", FILES["complementary_feeding_infants_young_children_6_23_months.pdf"],
        "Recommendation 5 (PDF page 14): foods high in sugar, salt, trans fats, sugar-sweetened beverages and non-sugar sweeteners should not be consumed."),
}


def _dump(path: Path, obj) -> None:
    """Ghi lại JSON giữ nguyên kiểu xuống dòng (CRLF) và dòng trống cuối tệp của tệp gốc."""
    old = path.read_bytes()
    crlf, tail = b"\r\n" in old, old.endswith(b"\n")
    text = json.dumps(obj, ensure_ascii=False, indent=2) + ("\n" if tail else "")
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


def main():
    mpath = CAT / "external_sources_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    have = {x.get("file") for x in manifest["manually_downloaded"]}
    added = 0
    for name, desc in FILES.items():
        p = RAW / name
        if name in have or not p.exists():
            continue
        data = p.read_bytes()
        manifest["manually_downloaded"].append({"file": name, "description": desc, "bytes": len(data),
                                                "sha256": hashlib.sha256(data).hexdigest(), "obtained": OBTAINED})
        added += 1
    _dump(mpath, manifest)

    cpath = CAT / "source_citation_catalog.json"
    catalog = json.loads(cpath.read_text(encoding="utf-8"))
    new_keys = 0
    for key, (cid, title, scope) in CATALOG.items():
        if key not in catalog:
            catalog[key] = {"citation_id": cid, "title": title, "url": "", "scope": scope}
            new_keys += 1
    _dump(cpath, catalog)
    print(f"manifest: +{added} files; catalog: +{new_keys} keys")


if __name__ == "__main__":
    main()
