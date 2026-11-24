"""Chẩn đoán vì sao VietOCR fine-tune làm mất mã phụ gia (so với bản gốc), trên checkpoint chữ thô.
Dùng: .\venv\Scripts\python.exe scripts_research\diag_ft_code_loss.py [--images 0302.jpeg 0201.jpeg ...] [--top 8]
"""
import argparse, json, re, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts_research"))
sys.stdout.reconfigure(encoding="utf-8")

import eval_vietocr_ft as ev
import ocr_regression as reg
from modules.ocr_extractor import parse_ingredients_from_ocr

HEADING = r"thanh\s*phan|ingredients"
HEAD_RE = re.compile(r"(?i)th[aàáảãạ]nh ph[aàáảãạầ]n")


def fold(s):
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return re.sub(r"[̀-ͯ]", "", s)


def codes_and_mentions(raw):
    full, _, _ = reg.reprocess_ocr_text(raw)
    ments = parse_ingredients_from_ocr(full, include_allergen_statements=True)
    return full, ments, reg.codes_from_mentions(ments)


def window(text, code, width=140):
    num = re.sub(r"\D", "", code)  # "170(i)" -> "170"
    for m in re.finditer(num, text):
        return text[max(0, m.start() - width): m.end() + width].replace("\n", " ")
    return "(không thấy chữ số này trong chữ fine-tune)"


def around_heading(text):
    m = HEAD_RE.search(text)
    return text[m.start(): m.start() + 200] if m else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--top", type=int, default=8, help="nếu không chỉ định ảnh: lấy N ảnh mất nhiều mã nhất")
    args = ap.parse_args()

    done = {}
    for line in open(ev.CKPT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        done[r["image"]] = r

    rows = []
    for path, ing in ev.test_images():
        r = done.get(path.name)
        if not r:
            continue
        gold = reg.codes_from_mentions(parse_ingredients_from_ocr(reg.label_text(ing), include_allergen_statements=True))
        b_full, b_m, b_codes = codes_and_mentions(r["base"])
        f_full, f_m, f_codes = codes_and_mentions(r["finetuned"])
        lost = sorted((gold & b_codes) - f_codes)
        rows.append((path.name, lost, r, b_full, f_full, b_m, f_m))

    if args.images:
        pick = [x for x in rows if x[0] in args.images]
    else:
        pick = sorted((x for x in rows if x[1]), key=lambda x: -len(x[1]))[: args.top]

    tot_lost = sum(len(x[1]) for x in rows)
    no_head = sum(len(x[1]) for x in rows if x[1] and not re.search(HEADING, fold(x[2]["finetuned"])))
    print(f"Tổng mã mất: {tot_lost} | trong ảnh mất tiêu đề 'Thành phần' ở fine-tune: {no_head}\n")

    for name, lost, r, b_full, f_full, b_m, f_m in pick:
        print("=" * 100)
        print(f"{name}: mất {len(lost)} mã -> {lost}")
        print(f"  tiêu đề: gốc={bool(re.search(HEADING, fold(r['base'])))} fine-tune={bool(re.search(HEADING, fold(r['finetuned'])))}")
        print(f"  số mục parser lấy: gốc={len(b_m)} fine-tune={len(f_m)}")
        for c in lost[:3]:
            print(f"  [{c}] quanh mã trong chữ fine-tune:\n      {window(f_full, c)}")
        print("  5 mục cuối (gốc)      :", [m.get("raw_mention", "")[:40] for m in b_m[-5:]])
        print("  5 mục cuối (fine-tune):", [m.get("raw_mention", "")[:40] for m in f_m[-5:]])
        print("  gốc, quanh tiêu đề      :", around_heading(b_full))
        print("  fine-tune, quanh tiêu đề:", around_heading(f_full))


if __name__ == "__main__":
    main()
