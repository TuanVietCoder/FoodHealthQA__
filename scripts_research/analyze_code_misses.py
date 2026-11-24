"""Mã phụ gia bị SÓT trên ảnh thật (code_recall của ocr_regression.py) — lỗi nằm ở khâu nào?

Với mỗi mã có trong danh sách thành phần chuẩn của ảnh nhưng hệ thống không dò được từ chữ OCR:

  A3_parser_cut       chữ OCR CÓ đúng số mã nhưng mã không nằm trong cụm thành phần nào: bộ tách dừng sớm vì một
                      cụm "điểm dừng" (giá trị dinh dưỡng, NSX, hướng dẫn sử dụng…) của KHỐI CHỮ KHÁC bị chèn vào giữa
                      danh sách (thứ tự đọc trộn cột), không tìm thấy tiêu đề "Thành phần", hoặc mã nằm trước tiêu đề
  A2_linker_rejects   mã nằm trong một cụm thành phần nhưng khâu liên kết không nhận (vd "INS 150 a", số trần sau khi
                      tách theo tiền tố INS, tên nhóm đứng trước bị đọc sai)
  A1_suffix_misread   đúng số nhưng hậu tố La Mã bị VietOCR đọc sai ("500(11)", "322()", "450(m)") → liên kết bỏ cả mã
  A1_suffix_variant   đúng số, liên kết ra cùng số nhưng khác hậu tố (500(i) thay vì 500(ii))
  A_postprocess       chữ OCR THÔ có mã đúng nhưng sau hậu xử lý mất → hậu xử lý làm hỏng
  B_misread_char      VietOCR đọc nhầm ký tự: thay số bằng chữ giống hình (O→0, l/I→1, S→5, B→8, …) hoặc tiền tố
                      sai (1NS, lNS, IN5, €)  → sửa được bằng luật nhầm ký tự
  B_misread_digit     sai/thiếu/thừa MỘT chữ số cạnh tiền tố mã (E33 thay vì E330) → nhầm số, khó sửa bằng luật
  C_name_unlinked     mã chuẩn suy ra từ TÊN chất (acid citric → 330), tên có đủ trong chữ OCR → khâu liên kết tên
  D_name_misread      tên chất chỉ có một phần trong chữ OCR (sai dấu/chữ) → VietOCR đọc sai chữ
  E_code_missing      mã ghi rõ trên nhãn nhưng chữ OCR không có dấu vết; tên nhóm đứng trước có hay không được ghi
                      ở cột group_name_in_ocr (có → dòng mã bị cắt/không dò được vùng chữ hoặc nhầm quá nặng)
  E_name_missing      tên chất không có trong chữ OCR

Dùng: python scripts_research/analyze_code_misses.py [--examples 8]
Kết quả: data/analysis/code_miss_breakdown.json + code_misses.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_research"))

import ocr_regression as reg  # noqa: E402
from modules.additive_linker import link_additive_mentions, normalize_ins_key  # noqa: E402
from modules.ocr_extractor import parse_ingredients_from_ocr, reprocess_ocr_text  # noqa: E402

OUT_JSON = ROOT / "data" / "analysis" / "code_miss_breakdown.json"
OUT_CSV = ROOT / "data" / "analysis" / "code_misses.csv"

# ký tự VietOCR hay nhầm với chữ số (đã bỏ dấu, chữ thường)
CONFUSION = {"o": "0", "q": "0", "d": "0", "u": "0", "i": "1", "l": "1", "|": "1", "!": "1", "j": "1", "t": "7",
             "z": "2", "s": "5", "$": "5", "g": "9", "b": "8", "&": "8", "a": "4"}
_PREFIX = r"(?:\b(?:e|ins)\s*[-.:]?\s*|[(,;]\s*)"
_FUZZY_PREFIX = r"(?:(?<![a-z0-9])(?:[e3€]|[i1l|!][nm][s5$]|ln5|1n5|lns|in5|ims)\s*[-.:]?\s*|[(,;]\s*)"
# số đi kèm đơn vị ("350 g", "2,8%", "140 kcal") là khối lượng/hàm lượng, không phải mã đọc nhầm
_CANDIDATE = re.compile(_FUZZY_PREFIX + r"([0-9oqduil|!jtzs$gb&a]{2,5})(?=[a-z]?(?![0-9a-z]{2}))"
                        r"(?!\s*(?:g|mg|mcg|kg|ml|l|%|kcal|kj|cal|gr|gam)\b)(?!\s*%)")


def fold(text):
    t = unicodedata.normalize("NFD", str(text or "").replace("đ", "d").replace("Đ", "D")).casefold()
    return re.sub(r"\s+", " ", "".join(c for c in t if not unicodedata.combining(c)))


def base_digits(key):
    m = re.match(r"(\d{3,4})", key)
    return m.group(1) if m else key


def edit1(a, b):
    """a, b khác nhau đúng một phép sửa (thay/thêm/bớt một ký tự)."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, t = (a, b) if len(a) < len(b) else (b, a)
    return any(t[:i] + t[i + 1:] == s for i in range(len(t)))


def _stop_patterns():
    """Danh sách điểm dừng của bộ tách thành phần (đọc từ mã nguồn để luôn khớp phiên bản đang dùng)."""
    src = (ROOT / "modules" / "ocr_extractor.py").read_text(encoding="utf-8")
    block = src[src.index("stop_patterns = ["):]
    block = block[:block.index("]\n")]
    return [eval(x) for x in re.findall(r'(r"[^"]+")', block)]  # noqa: S307 — chuỗi regex hằng trong mã nguồn


STOPS = _stop_patterns()
STOP_CAUSES = Counter()


def pipeline_stage(key, d, ff, mentions):
    """Chữ OCR có đúng số mã — mất ở bộ tách, khâu liên kết, hay do hậu tố bị đọc sai?"""
    pat = re.compile(r"(?<!\d)" + re.escape(d) + r"(?!\d)")
    hold = [m.get("raw_mention", "") for m in mentions if pat.search(m.get("raw_mention", ""))]
    if not hold:
        heads = list(re.finditer(r"(?:thanh\s*phan|ingredients?)\s*[:\-]?\s*", ff))
        pos = [m.start() for m in pat.finditer(ff)]
        if not heads:
            STOP_CAUSES["không thấy tiêu đề Thành phần"] += 1
        elif all(p < heads[0].start() for p in pos):
            STOP_CAUSES["mã nằm trước tiêu đề"] += 1
        else:
            h = heads[0].end()
            ends = [(h + sm.start(), sp) for sp in STOPS for sm in [re.search(sp, ff[h:], re.I)] if sm]
            STOP_CAUSES["dừng tại: " + (min(ends)[1] if ends else "?")] += 1
        return "A3_parser_cut"
    linked = {normalize_ins_key(h["matched_entity"].get("ins_code", "")) for t in hold for h in link_additive_mentions(t)}
    if any(base_digits(k) == d for k in linked):
        return "A1_suffix_variant"
    after = [t[m.end():m.end() + 6] for t in hold for m in [pat.search(t)]]
    if "(" in key and any(re.match(r"\s*\(?\s*[0-9il|\]\[)(m]{0,4}\)", a) for a in after):
        return "A1_suffix_misread"
    return "A2_linker_rejects"


def gold_hits(ref):
    """{mã: [(nguồn mã, alias khớp, cụm thành phần chuẩn)]}"""
    out = defaultdict(list)
    for m in parse_ingredients_from_ocr(reg.label_text(ref), include_allergen_statements=True):
        for h in link_additive_mentions(m.get("raw_mention", "")):
            key = normalize_ins_key(h["matched_entity"].get("ins_code", ""))
            if key:
                out[key].append((h.get("code_origin"), h.get("matched_alias") or "", m.get("raw_mention", "")))
    return out


def snippet(text, start, end, pad=30):
    return text[max(0, start - pad):end + pad].replace("\n", " ")


def classify(key, hits, full, raw, gold_digits_all, mentions=()):
    d = base_digits(key)
    ff, rf = fold(full), fold(raw)
    explicit = [h for h in hits if h[0] == "explicit_on_label"]
    exact = re.compile(_PREFIX + re.escape(d) + r"(?!\d)")
    group_name = ""
    if explicit:
        m = exact.search(ff)
        if m:
            return pipeline_stage(key, d, ff, mentions), snippet(ff, m.start(), m.end()), explicit[0][2]
        m = exact.search(rf)
        if m:
            return "A_postprocess", snippet(rf, m.start(), m.end()), explicit[0][2]
        for cm in _CANDIDATE.finditer(rf):
            tok = cm.group(1)
            mapped = "".join(CONFUSION.get(c, c) for c in tok)
            if mapped == d:
                return "B_misread_char", snippet(rf, cm.start(), cm.end()), explicit[0][2]
        for cm in _CANDIDATE.finditer(rf):
            mapped = "".join(CONFUSION.get(c, c) for c in cm.group(1))
            if mapped.isdigit() and len(d) >= 3 and edit1(mapped, d) and mapped not in gold_digits_all:
                return "B_misread_digit", snippet(rf, cm.start(), cm.end()), explicit[0][2]
        # tên nhóm đứng trước mã trong cụm chuẩn ("chất điều chỉnh độ acid (330)" → "chat dieu chinh do acid")
        gm = re.match(r"\s*([^()\d]{4,}?)\s*(?:\(|ins|e\s*\d|\d)", fold(explicit[0][2]))
        group_name = gm.group(1).strip(" :-") if gm else ""
        return "E_code_missing", ("group_name_in_ocr" if group_name and group_name in ff else ""), explicit[0][2]
    # mã suy ra từ tên chất
    alias = fold(hits[0][1])
    if alias and alias in ff:
        return "C_name_unlinked", snippet(ff, ff.find(alias), ff.find(alias) + len(alias)), hits[0][2]
    words = re.findall(r"[a-z0-9]+", alias)
    have = sum(re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", ff) is not None for w in words)
    if words and have / len(words) >= 0.5:
        return "D_name_misread", f"{have}/{len(words)} từ của '{alias}'", hits[0][2]
    return "E_name_missing", "", hits[0][2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()
    images = reg.load_images()
    rows, counts = [], Counter()
    total_gold = 0
    for name, raw, ref in images:
        full, _, _ = reprocess_ocr_text(raw)
        mentions = parse_ingredients_from_ocr(full, include_allergen_statements=True)
        got = reg.codes_from_mentions(mentions)
        gh = gold_hits(ref)
        total_gold += len(gh)
        gold_digits_all = {base_digits(k) for k in gh}
        for key in sorted(set(gh) - got):
            cat, evidence, gold_item = classify(key, gh[key], full, raw, gold_digits_all, mentions)
            counts[cat] += 1
            rows.append({"image": name, "code": key, "category": cat, "gold_item": gold_item, "ocr_evidence": evidence})
    missed = len(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "code", "category", "gold_item", "ocr_evidence"])
        w.writeheader()
        w.writerows(rows)
    e_group = sum(1 for r in rows if r["category"] == "E_code_missing" and r["ocr_evidence"] == "group_name_in_ocr")
    report = {"images": len(images), "gold_codes": total_gold, "missed": missed,
              "by_category": dict(counts.most_common()), "E_code_missing_with_group_name_in_ocr": e_group,
              "A3_parser_cut_causes": dict(STOP_CAUSES.most_common())}
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(images)} ảnh · {total_gold} mã chuẩn · bị sót {missed} ({missed / max(total_gold, 1):.1%})")
    for cat, n in counts.most_common():
        extra = f"  (trong đó tên nhóm có trong chữ OCR: {e_group})" if cat == "E_code_missing" else ""
        print(f"  {cat:<18} {n:>4}  {n / max(missed, 1):6.1%} số mã sót{extra}")
    if STOP_CAUSES:
        print("\nA3_parser_cut — vì sao bộ tách bỏ mã:")
        for c, n in STOP_CAUSES.most_common(10):
            print(f"  {n:>4}  {c}")
    for cat in counts:
        ex = [r for r in rows if r["category"] == cat][:args.examples]
        print(f"\n— {cat}:")
        for r in ex:
            print(f"   {r['image']:<12} {r['code']:<7} chuẩn: {r['gold_item'][:45]:<45} | OCR: {r['ocr_evidence'][:80]}")
    print(f"\n→ {OUT_JSON}\n→ {OUT_CSV}")


if __name__ == "__main__":
    main()
