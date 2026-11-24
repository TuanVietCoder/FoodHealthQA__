"""Mở rộng knowledge_base/additives_standard.json theo tần suất phụ gia xuất hiện trên nhãn của bộ đánh giá.

Chỉ thêm phụ gia mà MỌI trường dữ liệu đều lấy được từ nguồn đã có trong KB, không viết tay:
  - danh tính + chức năng (tiếng Việt) + vị trí trong TT 24/2019 Phụ lục 1: knowledge_base/additive_registry.json
  - ADI: bản ghi kiểm chứng JECFA (derived/jecfa_adi_verification*.json, đối chiếu với trang JECFA đã lưu)
  - Codex CXS 192-1995 Table One: derived/codex_192_gsfa_table_one.json (chạy extract_codex_gsfa.py trước)
Không thêm lưu ý sức khỏe hay nhóm dị nguyên (chưa có nguồn đã kiểm): ghi rõ "not_assessed".

Chạy: .\\venv\\Scripts\\python.exe scripts_research\\expand_additive_kb.py --top 30 [--apply]
Thêm đúng các mã chỉ định: --codes 420i 420ii 1520 (bỏ qua xếp hạng theo tần suất).
Không có --apply: chỉ in danh sách sẽ thêm.

Mục KB mang mã gốc (INS 420, 160a) mà sổ nhận diện chỉ có mã con (420(i), 160a(i)...) không bao giờ gắn được vào nhãn.
SUPERSEDE: khi đủ các mục con trong KB, mục mã gốc được thay bằng các mục con (lưu ý sức khỏe và đánh giá dị nguyên
đã kiểm của mục gốc được chép sang từng mục con).
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.additive_linker import link_additive_mentions  # noqa: E402
import extract_codex_gsfa as codex  # noqa: E402

KB = ROOT / "knowledge_base" / "additives_standard.json"
REGISTRY = ROOT / "knowledge_base" / "additive_registry.json"
DERIVED = ROOT / "knowledge_base" / "sources" / "derived"
EVAL = ROOT / "data" / "foodhealthqa_eval_set.jsonl"
LABELS = ROOT / "assets" / "labels"
TT24_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/9/09-vbhn-byt.pdf"
ROMAN_TAIL = re.compile(r"(?:[ivx]+)$")
SUPERSEDE = {"420": ("420i", "420ii"), "160a": ("160ai", "160aii", "160aiii")}


def norm(code) -> str:
    return re.sub(r"[()\s]", "", re.sub(r"^(ins|e)\s*", "", str(code or "").strip(), flags=re.I)).lower()


def label_frequency() -> collections.Counter:
    """Số ảnh nhãn (trong bộ đánh giá) có mỗi mã INS, tính bằng chính bộ liên kết tên→mã của hệ thống."""
    images = set()
    for line in EVAL.read_text(encoding="utf-8").splitlines():
        ref = json.loads(line).get("image_ref")
        if ref:
            images.add(Path(ref).stem)
    counts: collections.Counter = collections.Counter()
    for image in sorted(images):
        path = LABELS / f"{image}.json"
        if not path.is_file():
            continue
        label = json.loads(path.read_text(encoding="utf-8"))
        text = "; ".join(list(label.get("additive", [])) + list(label.get("ingredient", [])))
        counts.update({norm(r.get("ins_code")) for r in link_additive_mentions(text) if r.get("ins_code")})
    return counts


def load_jecfa() -> dict:
    out = {}
    for name in ("jecfa_adi_verification.json", "jecfa_adi_verification_more.json"):
        data = json.loads((DERIVED / name).read_text(encoding="utf-8"))
        for ins, rec in data.items():
            out[norm(ins)] = rec
    return out


def adi_upper(text: str):
    m = re.match(r"^\s*0\s*[-–]\s*([\d.]+)\s*mg/kg", str(text or ""))
    return float(m.group(1)) if m else None


def adi_display(rec: dict) -> str:
    adi = str(rec.get("latest_adi", "")).strip()
    year = rec.get("latest_evaluation_year", "?")
    up = adi.upper()
    if up.startswith("NOT SPECIFIED"):
        return "JECFA: ADI không xác định (Not Specified) - an toàn ở mức sử dụng thông thường theo GMP"
    if up.startswith("NOT LIMITED"):
        return "JECFA: ADI 'không giới hạn' (Not Limited) - an toàn ở mức sử dụng thông thường theo GMP"
    return f"JECFA (đánh giá {year}): {adi}"


def build_entry(code: str, reg: dict, rec: dict, table_one: dict, digest: str, scope_text: str) -> dict:
    ins = reg["ins_code"]
    names = [reg.get("canonical_name_vi", ""), reg.get("canonical_name_en", "")]
    synonyms = []
    for alias in list(reg.get("common_names") or []) + list(reg.get("aliases") or []):
        if alias and alias not in names and alias not in synonyms:
            synonyms.append(alias)
    adi_text = adi_display(rec)
    upper = adi_upper(rec.get("latest_adi"))
    locator = reg.get("source", {}).get("locator", "")
    matched = codex.codex_match(code, table_one)
    entries = [e for m in matched for e in table_one[m]["entries"]]
    entry = {
        "ins_code": ins,
        "e_number": f"E{code}",  # cùng kiểu với các mục đã có (E322i, E331iii)
        "canonical_name_vi": reg.get("canonical_name_vi", ""),
        "canonical_name_en": reg.get("canonical_name_en", ""),
        "synonyms": synonyms[:8],
        "function": reg.get("function", ""),
        "adi_value": adi_text,
        "adi_upper": upper,
        "adi_source": rec.get("url", ""),
        "vn_regulation": f"Có trong Phụ lục 1 (danh mục phụ gia được phép sử dụng) của TT 24/2019/TT-BYT ({locator}).",
        "codex_standard": "CODEX_STAN_192_1995",
        "allergen_groups": [],
        "contraindications": [],
        "unverified_legacy_regulatory_claims": {"claims": {}, "verification_status": "not_applicable_added_from_primary_sources",
                                                "use_for_compliance_decision": False},
        "unverified_legacy_health_claims": [],
        "health_considerations": [],
        "health_claim_policy": "Only source-checked, explicitly scoped considerations may be shown. Do not infer individual "
                               "safety, diagnosis, or treatment from an ingredient match.",
        "regulatory_claim": {
            "jurisdiction": "Vietnam",
            "claim": f"Có trong Phụ lục 1 TT 24/2019/TT-BYT ({locator}).",
            "verification_status": "source_checked_permitted_substance_level",
            "source_id": reg.get("source", {}).get("id", "VN_TT24_2019_CONSOLIDATED_2024_ANNEX_1"),
            "source_url": reg.get("source", {}).get("url", TT24_URL),
            "source_locator": locator,
            "use_for_compliance_decision": True,
            "compliance_decision_scope": scope_text,
        },
        "codex_claim": {
            "claim": codex.summarize(entries),
            "source_id": "CODEX_GSFA",
            "source_url": "https://www.fao.org/gsfaonline/",
            "verification_status": "source_checked_gsfa_table_one",
            "source_file": "knowledge_base/sources/raw/external/cxs_192e.pdf",
            "sha256": digest,
            "extract_file": "knowledge_base/sources/derived/codex_192_gsfa_table_one.json",
            "codex_codes": [table_one[m]["codex_code"] for m in matched],
            "n_food_category_provisions": len(entries),
            "pages": sorted({e["page"] for e in entries}),
            "use_for_compliance_decision": False,
            "scope": "Đã đối chiếu mã INS với CXS 192-1995 Table One và trích nhóm thực phẩm + mức tối đa. "
                     "Chỉ dùng để tra mức tối đa khi đã xác định đúng mã nhóm thực phẩm của sản phẩm.",
        },
        "adi_assessment": {
            "display_text": adi_text,
            "numeric_upper_mg_per_kg_bw_day": upper,
            "authority": "JECFA",
            "source_id": f"JECFA_CHEMICAL_{rec.get('chemical_id')}",
            "source_url": rec.get("url", ""),
            "source_locator": f"{rec.get('jecfa_name', '')} — lần đánh giá gần nhất {rec.get('latest_evaluation_year', '?')} "
                              f"(file {rec.get('file', '')}, sha256 {str(rec.get('sha256', ''))[:12]})",
            "verification_status": "source_checked",
            "use_for_numeric_compliance": upper is not None,
            "review_note": "Giá trị lấy từ trang JECFA đã lưu trong KB; chưa đối chiếu độc lập với đánh giá của cơ quan khác.",
        },
        "allergen_assessment": {
            "groups": [],
            "verification_status": "not_assessed",
            "note": "Chưa đánh giá nguy cơ dị nguyên của phụ gia này; không được hiểu là 'không có dị nguyên'.",
        },
        "sources": [{"id": f"JECFA_CHEMICAL_{rec.get('chemical_id')}", "title": f"JECFA — {rec.get('jecfa_name', '')}",
                     "url": rec.get("url", ""), "locator": f"ADI {rec.get('latest_adi', '')}"}],
    }
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--codes", nargs="*", default=[], help="thêm đúng các mã này (bỏ qua xếp hạng theo tần suất)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    kb = json.loads(KB.read_text(encoding="utf-8"))
    kb_codes = {norm(x["ins_code"]) for x in kb}
    bare_kb = {c for c in kb_codes if not ROMAN_TAIL.search(c) or c.isdigit()}  # mã không có hậu tố nhóm (vd 420)
    registry = {norm(r.get("ins_code")): r for r in json.loads(REGISTRY.read_text(encoding="utf-8"))}
    jecfa = load_jecfa()
    table_data = json.loads((DERIVED / "codex_192_gsfa_table_one.json").read_text(encoding="utf-8"))
    table_one, digest = table_data["additives"], table_data["sha256"]
    scope_text = next(x["regulatory_claim"]["compliance_decision_scope"] for x in kb if x.get("regulatory_claim", {}).get("compliance_decision_scope"))

    freq = label_frequency()
    picked, skipped = [], collections.Counter()
    wanted = [norm(c) for c in args.codes]
    ranking = [(c, freq.get(c, 0)) for c in wanted] if wanted else freq.most_common()
    for code, n_img in ranking:
        if code in kb_codes:
            continue
        base = ROMAN_TAIL.sub("", code) if not code.isdigit() else code
        if base != code and base in bare_kb and not wanted:
            skipped["biến thể của mã đã có trong KB"] += 1
            continue
        rec = jecfa.get(code)
        if code not in registry:
            skipped["không có trong sổ nhận diện TT24"] += 1
        elif not codex.codex_match(code, table_one):
            skipped["không có trong Codex Table One"] += 1
        elif not rec or not str(rec.get("latest_adi", "")).strip() or not rec.get("passed"):
            skipped["không có ADI JECFA đã kiểm chứng"] += 1
        else:
            picked.append((code, n_img))
        if len(picked) >= args.top and not wanted:
            break

    covered_before = sum(v for c, v in freq.items() if c in kb_codes)
    covered_after = covered_before + sum(v for _, v in picked)
    total = sum(freq.values())
    print(f"KB hiện có {len(kb)} mục; thêm {len(picked)} mục; bỏ qua: {dict(skipped)}")
    print(f"Lượt phụ gia trên nhãn nằm trong KB: {100 * covered_before / total:.1f}% → {100 * covered_after / total:.1f}%")
    for code, n_img in picked:
        print(f"  {registry[code]['ins_code']:<10} {n_img:>3} ảnh  {registry[code].get('canonical_name_en', '')[:40]:<40} "
              f"{adi_display(jecfa[code])[:48]}")
    if args.apply:
        for code, _ in picked:
            kb.append(build_entry(code, registry[code], jecfa[code], table_one, digest, scope_text))
        present = {norm(x["ins_code"]): x for x in kb}
        for base, subs in SUPERSEDE.items():
            if base in present and all(sub in present for sub in subs):
                old = present[base]
                for sub in subs:
                    child = present[sub]
                    if child["allergen_assessment"]["verification_status"] == "not_assessed":
                        child["allergen_assessment"] = copy.deepcopy(old["allergen_assessment"])
                    child["health_considerations"] = copy.deepcopy(old.get("health_considerations") or [])
                kb = [x for x in kb if x is not old]
                print(f"Thay mục gốc INS {base} bằng {len(subs)} mục con {list(subs)} (chép {len(old.get('health_considerations') or [])} lưu ý sức khỏe)")
        KB.write_text(json.dumps(kb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Đã ghi {KB.name}: {len(kb)} mục.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
