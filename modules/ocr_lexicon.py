"""Sửa chính tả chữ OCR THEO CỤM bằng danh mục tên thành phần (không dùng LLM, không sửa từng từ đơn lẻ).

Vì sao theo cụm: chữ OCR sai dấu tiếng Việt thường vẫn là một từ đúng chính tả ("sửa", "cả", "đóng", "khoảng"), nên
nhìn một từ không biết có sai không. Bộ sửa từng từ trước đây làm chữ TỆ hơn (độ phủ từ 87,4% → 82,7%, 499 chỗ sửa
sai, sinh dị nguyên giả "trung"→"trứng", "của"→"cua"). Cụm thì phân biệt được: "dầu cả" không phải tên thành phần,
"dầu cá" thì phải.

Quy tắc:
  * So ở dạng bỏ dấu; cụm 5 → 2 từ liền nhau (chỉ cách nhau bởi khoảng trắng/gạch nối) khớp ĐÚNG một tên trong danh
    mục → thay bằng cách viết có dấu của danh mục (giữ kiểu chữ hoa/thường của từng từ).
  * Không sửa khi chữ gốc đã là một tên hợp lệ trong danh mục, hoặc khi dạng bỏ dấu ứng với ≥2 cách viết có dấu.
  * Từ đơn chỉ sửa khi là tên Latin dài (≥10 ký tự, không dấu) sai đúng MỘT ký tự so với đúng một tên trong danh mục
    ("Bifidobactenum" → "Bifidobacterium").
  * Chỉ áp cho cụm thành phần (ocr_extractor.parse_ingredients_from_ocr), không áp cho phần hướng dẫn/quảng cáo.

Danh mục = tên phụ gia trong knowledge_base/additive_registry.json + từ khóa dị nguyên (constraint_model) + cụm viết
tay cũ (ocr_extractor._FOOD_COMPOUND_LIST) + cụm thành phần của ảnh PHÁT TRIỂN trong bộ đánh giá
(knowledge_base/ocr_ingredient_lexicon.json, tạo bằng scripts_research/build_ocr_lexicon.py — không dùng ảnh KIỂM TRA).
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "knowledge_base" / "additive_registry.json"
DEV_LEXICON = ROOT / "knowledge_base" / "ocr_ingredient_lexicon.json"

MAX_WORDS = 5
MIN_WORDS = 2
LATIN_MIN_LEN = 10
_WORD = re.compile(r"[^\W\d_]+")


def fold(text: str) -> str:
    t = unicodedata.normalize("NFC", str(text or "")).casefold().replace("đ", "d")
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c))


def _nfc_lower(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or "")).casefold()


def _phrases_from_name(name: str):
    """Tên trong danh mục có thể kèm giải thích trong ngoặc / dấu phẩy → tách thành các cụm con."""
    for part in re.split(r"[(),;/\[\]]", str(name or "")):
        words = _WORD.findall(part)
        if words:
            yield " ".join(words)


def _base_terms():
    terms = []
    try:
        for rec in json.loads(REGISTRY.read_text(encoding="utf-8")):
            for name in [rec.get("canonical_name"), rec.get("canonical_name_vi"), rec.get("canonical_name_en"),
                         *(rec.get("aliases") or []), *(rec.get("common_names") or [])]:
                terms.extend(_phrases_from_name(name))
    except Exception:
        pass
    try:
        from modules.constraint_model import ALLERGEN_KEYWORDS
        for kws in ALLERGEN_KEYWORDS.values():
            for kw in kws:
                terms.extend(_phrases_from_name(kw))
    except Exception:
        pass
    try:
        from modules.ocr_extractor import _FOOD_COMPOUND_LIST
        for term in _FOOD_COMPOUND_LIST:
            terms.extend(_phrases_from_name(term))
    except Exception:
        pass
    if DEV_LEXICON.is_file():
        try:
            data = json.loads(DEV_LEXICON.read_text(encoding="utf-8"))
            terms.extend(item["phrase"] for item in data.get("phrases", []))
        except Exception:
            pass
    return terms


@lru_cache(maxsize=1)
def lexicon():
    """(cụm bỏ dấu → {cách viết có dấu}, tập cách viết hợp lệ, {độ dài → [từ Latin dài]})."""
    by_fold, valid, latin = {}, set(), {}
    for term in _base_terms():
        words = _WORD.findall(term)
        if not words or len(words) > MAX_WORDS:
            continue
        form = _nfc_lower(" ".join(words))
        valid.add(form)
        if len(words) >= MIN_WORDS:
            by_fold.setdefault(fold(form), set()).add(form)
        for w in words:
            if len(w) >= LATIN_MIN_LEN and w.isascii():
                latin.setdefault(len(w), set()).add(w.casefold())
    return by_fold, valid, latin


def clear_cache():
    lexicon.cache_clear()


_TONES = {"̀", "́", "̃", "̉", "̣"}   # huyền, sắc, ngã, hỏi, nặng


def _tone_signature(text: str):
    """Mỗi từ: (chữ bỏ dấu, các dấu thanh) — bỏ qua VỊ TRÍ đặt dấu, để "hoá" và "hóa" là một."""
    sig = []
    for word in text.split(" "):
        marks = sorted(c for c in unicodedata.normalize("NFD", word) if c in _TONES)
        shape = unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", word) if c not in _TONES))
        sig.append((shape, tuple(marks)))
    return tuple(sig)


def _match_case(orig: str, new: str) -> str:
    if orig.isupper() and len(orig) > 1:
        return new.upper()
    if orig[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def _edit1(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, t = (a, b) if len(a) < len(b) else (b, a)
    return any(t[:i] + t[i + 1:] == s for i in range(len(t)))


def correct_phrases(text: str):
    """Trả về (chữ đã sửa, [(cụm gốc, cụm sửa)])."""
    text = unicodedata.normalize("NFC", str(text or ""))
    by_fold, valid, latin = lexicon()
    toks = list(_WORD.finditer(text))
    if not toks:
        return text, []
    folded = [fold(m.group(0)) for m in toks]
    edits, changes = [], []          # (start, end, replacement)
    i = 0
    while i < len(toks):
        done = False
        for n in range(min(MAX_WORDS, len(toks) - i), MIN_WORDS - 1, -1):
            seq = toks[i:i + n]
            if any(not re.fullmatch(r"[\s\-]+", text[a.end():b.start()]) for a, b in zip(seq, seq[1:])):
                continue
            forms = by_fold.get(" ".join(folded[i:i + n]))
            if not forms:
                continue
            orig_words = [m.group(0) for m in seq]
            orig_form = _nfc_lower(" ".join(orig_words))
            form = next(iter(forms))
            if (orig_form in valid or len(forms) != 1
                    or _tone_signature(orig_form) == _tone_signature(form)   # chỉ khác chỗ đặt dấu (hoá/hóa)
                    or (fold(form) == form and fold(orig_form) != orig_form)):  # tên không dấu, chữ gốc có dấu
                i += n
                done = True
                break
            new_words = form.split(" ")
            for m, w, nw in zip(seq, orig_words, new_words):
                rep = _match_case(w, nw)
                if rep != w:
                    edits.append((m.start(), m.end(), rep))
            changes.append((" ".join(orig_words), " ".join(_match_case(w, nw) for w, nw in zip(orig_words, new_words))))
            i += n
            done = True
            break
        if done:
            continue
        # Từ đơn KHÔNG được sửa — kể cả tên Latin dài: luật "sai một ký tự" đã thử và đổi từ đúng thành từ khác
        # (Carbohydrate → Carbohydrase, tocopheryl → tocopherol, Hydrolyzed → Hydrolysed), nên đã bỏ.
        i += 1
    for start, end, rep in sorted(edits, reverse=True):
        text = text[:start] + rep + text[end:]
    return text, changes
