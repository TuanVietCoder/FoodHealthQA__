"""Kiểm con số ADI của từng phụ gia trong knowledge_base/additives_standard.json với văn bản nguồn ĐÃ LƯU.

Với mỗi mục: lấy file nguồn của chính ADI đang hiển thị (EFSA PDF hoặc trang JECFA đúng mã INS), tìm một câu nguyên văn chứa
cả từ chỉ ADI lẫn đúng giá trị ADI của KB (số trên của khoảng, hoặc "not specified", "not limited", "withdrawn", "MTDI").
Kết quả mỗi mục:
  found      – có câu khớp; với --apply câu đó (kèm trang) được ghi vào adi_assessment.evidence
  mismatch   – nguồn có câu về ADI nhưng con số KHÁC giá trị KB (in ra để người xem quyết định, KHÔNG tự sửa số)
  not_found  – không thấy câu ADI nào trong file
  no_file    – không có file nguồn cục bộ
Chạy: .\\venv\\Scripts\\python.exe scripts_research\\verify_adi_evidence.py [--apply] [--report FILE.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_legacy_health_claims as rl  # noqa: E402
from modules import evidence as ev  # noqa: E402

KB = ROOT / "knowledge_base" / "additives_standard.json"
ADI_WORD = re.compile(r"\bADI\b|acceptable daily intake|tolerable (?:daily )?intake|\bMTDI\b|\bTDI\b", re.I)


def source_file(item) -> str:
    """Đường dẫn tương đối (so với raw/external) của file nguồn ADI, hoặc ''."""
    adi = item["adi_assessment"]
    local = ev.local_source_file(str(adi.get("source_id") or ""), str(adi.get("source_url") or ""))
    if not local:
        code = re.sub(r"^INS\s*", "", item["ins_code"].split("+")[0])
        rec = ev._jecfa_records().get(ev._norm_code(code))
        if rec and rec.get("file") and rec.get("passed") and str(adi.get("authority", "")).upper().lstrip().startswith(("JECFA", "WHO")):
            local = f"knowledge_base/sources/raw/external/jecfa/{rec['file']}"
    return local.replace("knowledge_base/sources/raw/external/", "") if local else ""


def expected(adi) -> tuple[str, str]:
    """(loại, mẫu regex của giá trị) theo con số KB."""
    upper = adi.get("numeric_upper_mg_per_kg_bw_day")
    text = str(adi.get("display_text", "")).lower()
    if upper is not None:
        num = f"{float(upper):g}"
        return "numeric", r"(?<![\d.,])" + re.escape(num).replace(r"\.", r"[.,]") + r"(?![\d])"
    if "mtdi" in text:
        return "mtdi", r"MTDI\s*(?:of\s*)?70"
    if "withdrawn" in text or "rut bo" in text or "rút bỏ" in text:
        return "withdrawn", r"withdr[ae]w|WITHDRAWN"
    if "not limited" in text or "không giới hạn" in text:
        return "not_limited", r"not limited|NOT LIMITED"
    if "not specified" in text or "không xác định" in text:
        return "not_specified", r"not specified|NOT SPECIFIED|no need for a numerical ADI|no numerical ADI"
    if "genotox" in text or "không có adi" in text or "no adi" in text or "not be established" in text:
        return "no_adi", r"no ADI|could not be established|cannot be established|no longer be considered"
    return "unknown", r"$^"


def sentences_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(". ", 0, start), text.rfind("; ", 0, start))
    right_candidates = [p for p in (text.find(". ", end), text.find("; ", end)) if p != -1]
    right = min(right_candidates) if right_candidates else len(text)
    left = 0 if left == -1 else left + 2
    if right - left > 420:  # câu quá dài: lấy cửa sổ quanh con số
        left, right = max(left, start - 200), min(right, end + 200)
    return text[left:right].strip()


def check(item) -> dict:
    adi = item["adi_assessment"]
    name = source_file(item)
    kind, value_rx = expected(adi)
    if not name:
        return {"result": "no_file", "kind": kind}
    pages = rl.pages_of(name)
    adi_sentences = []
    for number, text in enumerate(pages, 1):
        for m in re.finditer(value_rx, text, re.I):
            window = sentences_around(text, m.start(), m.end())
            if ADI_WORD.search(window) or kind in ("withdrawn", "not_specified", "not_limited", "mtdi"):
                quote = window
                try:
                    return {"result": "found", "kind": kind, "file": name, "evidence": rl.evidence(name, quote)}
                except SystemExit:
                    continue
        for m in ADI_WORD.finditer(text):
            if len(adi_sentences) < 3:
                adi_sentences.append({"page": number, "text": sentences_around(text, m.start(), m.end())[:300]})
    return {"result": "mismatch" if adi_sentences else "not_found", "kind": kind, "file": name, "adi_sentences_in_source": adi_sentences}


# Mục mà bộ dò tự động không bắt được vì giá trị KB không ở dạng số (đã đọc tay, trích nguyên văn):
MANUAL = {
    "INS 621": [("efsa_2017_4910.pdf", "the Panel derived a group acceptable daily intake (ADI) of 30 mg/kg bw per day, expressed as glutamic acid, for glutamic acid and glutamates (E 620–625)")],
    "INS 621+631+627": [("efsa_2017_4910.pdf", "the Panel derived a group acceptable daily intake (ADI) of 30 mg/kg bw per day, expressed as glutamic acid, for glutamic acid and glutamates (E 620–625)"),
                        ("jecfa/Chemical_2512.html", "INS matches 631 Evaluations Evaluation year: 1993 ADI: NOT SPECIFIED"),
                        ("jecfa/Chemical_822.html", "INS matches 627 Evaluations Evaluation year: 1993 ADI: NOT SPECIFIED")],
    "INS 1420": [("efsa_2017_4911.pdf", "the Panel concluded that there is no safety concern for the use of modiﬁed starches as food additives at the reported uses and use levels for the general population and that there is no need for a numerical ADI")],
    "INS 171": [("efsa_2021_6585.pdf", "Based on all the evidence available, a concern for genotoxicity could not be ruled out, and given the many uncertainties, the Panel concluded that E 171 can no longer be considered as safe when used as a food additive")],
    "INS 491": [("jecfa/Chemical_2271.html", "Evaluation year: 2026 ADI: Temporary group ADI of 0-25 mg/kg bw as total polyoxyethylene (20) sorbitan esters")],
    "INS 492": [("jecfa/Chemical_3983.html", "Evaluation year: 2026 ADI: Temporary group ADI of 0-25 mg/kg bw as total polyoxyethylene (20) sorbitan esters")],
    "INS 160c(i)": [("jecfa/Chemical_4486.html", "ADI: No safety concern at current levels of intake when used as a flavouring agent"),
                    ("jecfa/Chemical_4486.html", "The opinion that paprika oleoresin is acceptable as a spice was based on the report of the fourteenth meeting (1970), which stated that such use is self-limiting and obviates the need for an ADI.")],
}
NOT_LIMITED = "JECFA: ADI 'không giới hạn' (Not Limited) - an toàn ở mức sử dụng thông thường theo GMP"
# Giá trị KB khác văn bản nguồn: sửa theo nguồn (giữ giá trị cũ ở display_text_before_review)
CORRECTIONS = {
    "INS 322(i)": (NOT_LIMITED, None, [("jecfa/Chemical_1477.html", "INS matches 322(i) Evaluations Evaluation year: 1973 ADI: NOT LIMITED")]),
    "INS 330": (NOT_LIMITED, None, [("jecfa/Chemical_3594.html", "Evaluation year: 1973 ADI: NOT LIMITED Comments: Group ADI for citric acid and its calcium, potassium, sodium, and ammonium salts")]),
    "INS 500(ii)": (NOT_LIMITED, None, [("jecfa/Chemical_1892.html", "INS matches 500ii Evaluations Evaluation year: 1965 ADI: NOT LIMITED")]),
    "INS 262(i)": (NOT_LIMITED, None, [("jecfa/Chemical_2970.html", "INS matches 262(i) Evaluations Evaluation year: 1973 ADI: NOT LIMITED")]),
    "INS 471": (NOT_LIMITED, "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/917",
                [("jecfa/Chemical_917.html", "INS matches 471 Evaluations Evaluation year: 1973 ADI: NOT LIMITED")]),
    "INS 260": ("JECFA: ADI nhóm 'không giới hạn' (Not Limited) cho acid acetic và muối kali, natri (1973, giữ nguyên ở kỳ họp 49); "
                "với vai trò chất tạo hương (2004): không đáng lo ngại ở mức tiêu thụ hiện tại", None,
                [("jecfa/Chemical_4785.html", "THE GROUP ADI &#39;NOT LIMITED&#39; FOR ACETIC ACID AND ITS POTASSIUM AND SODIUM SALTS ESTABLISHED AT THE SEVENTEENTH MEETING IN 1973 WAS MAINTAINED AT THE FORTY-NINTH MEETING.")]),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    kb = json.loads(KB.read_text(encoding="utf-8"))
    report, counts = {}, {}
    for item in kb:
        code = item["ins_code"]
        if code in CORRECTIONS or code in MANUAL:
            text, url, quotes = CORRECTIONS.get(code, (None, None, MANUAL.get(code, [])))
            found = [rl.evidence(f, q) for f, q in quotes]  # dừng nếu câu không nguyên văn
            res = {"result": "corrected" if code in CORRECTIONS else "found_manual", "kind": "manual", "evidence_list": found}
            res["kb_display_text"] = item["adi_assessment"].get("display_text", "")
            report[code] = res
            counts[res["result"]] = counts.get(res["result"], 0) + 1
            if args.apply:
                adi = item["adi_assessment"]
                if text and adi.get("display_text") != text:
                    adi.setdefault("display_text_before_review", adi.get("display_text", ""))
                    adi["display_text"] = text
                    item["adi_value"] = text
                    adi["review_note"] = "Đã sửa theo văn bản trang JECFA đã lưu (2026-11-23); giá trị cũ ở display_text_before_review."
                if url:
                    adi["source_url"] = url
                    item["adi_source"] = url
                existing = adi.get("evidence", [])
                adi["evidence"] = existing + [e for e in found if all(e["quote"] != x["quote"] for x in existing)]
            continue
        res = check(item)
        res["kb_display_text"] = item["adi_assessment"].get("display_text", "")
        report[item["ins_code"]] = res
        counts[res["result"]] = counts.get(res["result"], 0) + 1
        if args.apply and res["result"] == "found":
            existing = item["adi_assessment"].get("evidence", [])
            if not any(e["quote"] == res["evidence"]["quote"] for e in existing):
                item["adi_assessment"]["evidence"] = existing + [res["evidence"]]
    print(f"{len(kb)} mục ADI: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for code, res in report.items():
        if res["result"] != "found":
            print(f"  {res['result']:9} {code:16} [{res['kind']}] KB: {res['kb_display_text'][:110]}")
            for s in res.get("adi_sentences_in_source", [])[:2]:
                print(f"             nguồn p.{s['page']}: {s['text'][:200]}")
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.apply:
        KB.write_text(json.dumps(kb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Đã ghi bằng chứng vào additives_standard.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
