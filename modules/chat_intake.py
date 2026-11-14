"""Tiếp nhận tin nhắn kiểu chat (giống Gemini) cho demo FoodHealthQA.

Người dùng gõ tự do trong một ô nhập (kèm ảnh nếu có). Mô-đun này tách tin nhắn thành:
  - câu hỏi về sản phẩm,
  - thông tin hồ sơ sức khỏe (tuổi, giới tính, bệnh nền, dị ứng, mang thai, cho con bú),
  - lệnh điều khiển (làm mới, xóa hồ sơ, chào hỏi, trợ giúp),
và kiểm tra đầu vào không hợp lệ (chuỗi vô nghĩa, câu hỏi ngoài phạm vi, thông tin mâu thuẫn, ảnh hỏng/trống).

Mọi hàm ở đây là hàm thuần (không gọi Streamlit), để kiểm thử độc lập. Khi đủ ảnh + câu hỏi, app gọi lại
đúng pipeline cũ (run_perception_pipeline → run_grounded_pipeline); các luật từ chối (thiếu dữ liệu / vượt phạm vi)
vẫn do pipeline quyết định.
"""

from __future__ import annotations

import io
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from modules import abstention
from modules import persona_builder as pb

MAX_MESSAGE_CHARS = 1500
MAX_IMAGE_MB = 10
MIN_IMAGE_SIDE = 120

# ---------------------------------------------------------------------------
# Chuẩn hóa văn bản
# ---------------------------------------------------------------------------


def fold(text: str) -> str:
    """Chữ thường, bỏ dấu tiếng Việt, đ → d, gộp khoảng trắng."""
    return _fold_map(text)[0]


def _fold_map(text: str) -> Tuple[str, List[int]]:
    """Bỏ dấu và giữ ánh xạ vị trí ký tự đã gấp → vị trí trong chuỗi gốc (để cắt lại cụm có dấu)."""
    src = unicodedata.normalize("NFC", str(text or ""))
    out, idx = [], []
    for i, ch in enumerate(src):
        c = ch.lower()
        if c == "đ":
            c = "d"
        for d in unicodedata.normalize("NFD", c):
            if not unicodedata.combining(d):
                out.append(d)
                idx.append(i)
    return "".join(out), idx


def _orig_slice(text: str, idx: List[int], start: int, end: int) -> str:
    src = unicodedata.normalize("NFC", str(text or ""))
    if start >= end or start >= len(idx):
        return ""
    return src[idx[start]: idx[end - 1] + 1]


# ---------------------------------------------------------------------------
# Phát hiện chuỗi vô nghĩa
# ---------------------------------------------------------------------------

# Âm tiết tiếng Việt (đã bỏ dấu): phụ âm đầu? + nguyên âm (1–3) + phụ âm cuối?
_VN_SYLLABLE = re.compile(r"^(ngh|ng|gh|gi|kh|nh|ph|th|tr|ch|qu|[bcdghklmnpqrstvx])?[aeiouy]{1,3}(ng|nh|ch|[cmnpt])?$")
# Viết tắt / thuật ngữ hợp lệ dù không phải âm tiết tiếng Việt
_ABBREV = {
    "ins", "adi", "pku", "ckd", "tt", "byt", "mg", "ml", "kg", "g", "e", "msg", "bmi", "who", "fda", "efsa",
    "jecfa", "codex", "pal", "ok", "vs", "tv", "kcal", "ppm", "dha", "ara", "hba1c",
}
_DIGIT = re.compile(r"\d")


def _token_ok(tok: str) -> bool:
    if _DIGIT.search(tok) or tok in _ABBREV or _VN_SYLLABLE.match(tok):
        return True
    # từ ngoại (tiếng Anh, tên hóa chất): cần có nguyên âm, không có cụm ≥ 4 phụ âm, không lặp 1 ký tự ≥ 3 lần
    if len(tok) < 3 or len(tok) > 25:
        return False
    if not re.search(r"[aeiouy]", tok):
        return False
    if re.search(r"[^aeiouy\W\d]{4,}", tok) or re.search(r"(.)\1\1", tok):
        return False
    vowels = sum(ch in "aeiouy" for ch in tok)
    return 0.2 <= vowels / len(tok) <= 0.75


def is_gibberish(text: str) -> bool:
    """True nếu đoạn văn chủ yếu là ký tự gõ bừa (vd: 'asdfgh', 'jjjjjj', '!!!???', 'xzqv brrrr')."""
    folded = fold(text)
    tokens = re.findall(r"[^\W_]+", folded)
    if not tokens:
        return True
    bad = sum(not _token_ok(t) for t in tokens)
    return bad / len(tokens) > 0.5


# ---------------------------------------------------------------------------
# Lệnh / chào hỏi
# ---------------------------------------------------------------------------

_RESET = re.compile(r"^\s*(lam moi|bat dau lai|cuoc tro chuyen moi|xoa het|xoa tat ca|reset|clear|new chat)\s*[.!]*\s*$")
_CLEAR_PROFILE = re.compile(r"^\s*(xoa|huy|bo|dat lai|reset)\s+(ho so|thong tin ca nhan|profile)\s*[.!]*\s*$")
_CLEAR_IMAGE = re.compile(r"^\s*(xoa|huy|bo)\s+(anh|hinh)( nhan)?\s*[.!]*\s*$")
_GREETING = re.compile(r"^\s*(xin chao|chao( ban| em| anh| chi)?|hello|hi|hey|alo|halo)\b")
_THANKS = re.compile(r"^\s*(cam on|thank|thanks|tks)\b")
_HELP = re.compile(r"\b(huong dan|help|tro giup|ban la ai|ban lam duoc gi|ban giup duoc gi|cach (dung|su dung))\b")


# ---------------------------------------------------------------------------
# Trích xuất hồ sơ sức khỏe
# ---------------------------------------------------------------------------

# mã bệnh nền → (mẫu nhận diện trên văn bản đã bỏ dấu); khóa phải trùng persona_builder.CONDITION_CHOICES
_CONDITION_PATTERNS = {
    "diabetes": r"tieu duong|dai thao duong|dai duong|diabetes|duong huyet cao",
    "hypertension": r"(tang|cao) huyet ap|huyet ap cao|hypertension",
    "chronic_kidney_disease_stage_3": r"suy than|benh than|than man|\bckd\b|chay than",
    "phenylketonuria_pku": r"\bpku\b|phenylketon",
    "lactose_intolerance": r"khong dung nap (duong )?(sua|lactose)|bat dung nap lactose|lactose intoleran",
    "gout": r"\bgout\b|\bgut\b|acid uric|axit uric",
    "obesity": r"mo mau|tim mach|beo phi|cholesterol cao|thua can",
    "peptic_ulcer": r"(viem|loet) (da day|bao tu|ta trang)|benh da day|dau da day",
    "asthma": r"hen suyen|hen phe quan|\bsuyen\b|bi hen\b",
}
_CONDITION_LABEL = {cfg["key"]: label for label, cfg in pb.CONDITION_CHOICES.items()}
_CONDITION_SHORT = {cfg["key"]: label.split("(")[0].strip() for label, cfg in pb.CONDITION_CHOICES.items()}
_CHILD_KEY = "children_adhd"

_NEGATED_BEFORE = re.compile(r"(khong|chua|khong he|khong co)\s+(bi\s+|mac\s+|co\s+)?$")
_HEALTHY = re.compile(r"\b(khoe manh|khong (co )?benh( nen)?|khong mac benh|binh thuong)\b")

_PREGNANT = re.compile(r"mang thai|co thai|\bmang bau\b|\bdang bau\b|ba bau|thai ky|thai \d+ (tuan|thang)")
_LACTATING = re.compile(r"cho con bu|nuoi con bang sua me|dang cho bu")
_AGE = re.compile(r"(?<![\d.,])(\d{1,3})\s*(tuoi|t\b)")
_AGE_MONTHS = re.compile(r"(?<![\d.,])(\d{1,2})\s*thang tuoi")
# giới tính: kiểm tra trên chữ có dấu vì "nam" (bỏ dấu) trùng "năm"
_MALE = re.compile(r"(?<!việt )\b(nam giới|giới tính\s*:?\s*nam|là nam|đàn ông|con trai|bé trai|nam,|nam\.|nam$)", re.I)
_FEMALE = re.compile(r"\b(nữ giới|giới tính\s*:?\s*nữ|là nữ|phụ nữ|con gái|bé gái|nữ\b)", re.I)
_MALE_FOLD = re.compile(r"\b(nam gioi|gioi tinh\s*:?\s*nam|dan ong)\b")
_FEMALE_FOLD = re.compile(r"\b(nu gioi|gioi tinh\s*:?\s*nu|phu nu|con gai|be gai)\b")

# "dị ứng ..." là lời khai báo (không phải câu hỏi "có gây dị ứng không")
_ALLERGY = re.compile(r"di ung(\s+voi)?\s*:?\s*")
_ALLERGY_NOT_STATEMENT = re.compile(r"(gay|nguy co|chat|tac nhan|phan ung|thanh phan|bi|co)\s+$")
_ALLERGY_SELF_BEFORE = re.compile(r"(toi|minh|em|tui|tao|con|be|chau|vo|chong|bo|me|ba|ong|nguoi|benh nhan|ho so)\s+(\w+\s+){0,3}(bi\s+|co\s+)?$")
_ALLERGY_STOP = re.compile(
    r"\b(thi|nen|co the|co nen|co an|co uong|an|uong|san pham|mon|khong|nhung|ma|vay|lieu|duoc|hay|nay|vi|nhe|nang|rat|"
    r"lam|bi|mac|tuoi|mang thai|benh|dang|cho|hoi|xin|giup|kiem tra|phan tich)\b"
)
_ALLERGY_SPLIT = re.compile(r",|;|/|\+|\bva\b|\bhoac\b|\bvoi\b|\bcung\b")
_ALLERGY_FILLER = {"gi", "nao", "do", "cai gi", "thu gi", "gi het", "gi ca", "a", "ca", "nhieu thu"}
_NO_ALLERGY = re.compile(r"(khong|chua|khong he)\s+(bi\s+|co\s+)?di ung\s*(gi|thu gi|nao)?\b")


def empty_profile() -> Dict[str, Any]:
    return {
        "conditions": [],      # mã bệnh nền (khóa trong CONDITION_CHOICES)
        "allergens": [],       # kết quả resolve_custom_allergen_term (status valid / valid_raw)
        "age": None,
        "gender": None,        # "Nam" | "Nữ"
        "pregnant": False,
        "lactating": False,
        "declared": False,     # người dùng đã khai báo ít nhất một thông tin
        "healthy": False,      # khai báo rõ "không bệnh nền"
        "no_allergy": False,   # khai báo rõ "không dị ứng"
    }


def extract_profile(text: str, foodon_catalog=None) -> Dict[str, Any]:
    """Trích thông tin hồ sơ trong tin nhắn. Trả về {update, errors, warnings, notes, spans}.

    `update` chỉ chứa các trường người dùng thực sự nhắc tới. `spans` là các đoạn (trên chuỗi đã gấp) dùng để
    nhận biết mệnh đề nào thuần khai báo hồ sơ.
    """
    folded, idx = _fold_map(text)
    nfc = unicodedata.normalize("NFC", str(text or ""))
    upd: Dict[str, Any] = {}
    errors: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []
    spans: List[Tuple[int, int]] = []

    # bệnh nền
    add_conds, remove_conds = [], []
    for key, pat in _CONDITION_PATTERNS.items():
        for m in re.finditer(pat, folded):
            spans.append(m.span())
            if _NEGATED_BEFORE.search(folded[max(0, m.start() - 20): m.start()]):
                remove_conds.append(key)
            elif key not in add_conds:
                add_conds.append(key)
    if add_conds:
        upd["conditions_add"] = add_conds
    if remove_conds:
        upd["conditions_remove"] = [k for k in remove_conds if k not in add_conds]
    m = _HEALTHY.search(folded)
    if m:
        spans.append(m.span())
        if not add_conds:
            upd["healthy"] = True

    # thai kỳ / cho con bú
    m = _PREGNANT.search(folded)
    if m:
        spans.append(m.span())
        if _NEGATED_BEFORE.search(folded[max(0, m.start() - 20): m.start()]):
            upd["pregnant"] = False
        else:
            upd["pregnant"] = True
    m = _LACTATING.search(folded)
    if m:
        spans.append(m.span())
        upd["lactating"] = not _NEGATED_BEFORE.search(folded[max(0, m.start() - 20): m.start()])

    # tuổi
    m = _AGE_MONTHS.search(folded)
    if m:
        spans.append(m.span())
        upd["age"] = 1
    else:
        ages = [mm for mm in _AGE.finditer(folded)]
        if ages:
            mm = ages[0]
            spans.append(mm.span())
            age = int(mm.group(1))
            if age < 1 or age > 120:
                errors.append(f"Tuổi **{age}** không hợp lệ — vui lòng nhập tuổi trong khoảng 1–120.")
            else:
                upd["age"] = age
            if len({int(a.group(1)) for a in ages}) > 1:
                warnings.append("Tin nhắn có nhiều độ tuổi khác nhau; hệ thống dùng độ tuổi đầu tiên.")

    # giới tính
    male = bool(_MALE.search(nfc) or _MALE_FOLD.search(folded))
    female = bool(_FEMALE.search(nfc) or _FEMALE_FOLD.search(folded))
    if male and female:
        errors.append("Thông tin giới tính mâu thuẫn (vừa **nam** vừa **nữ**) — vui lòng nhập lại.")
    elif male:
        upd["gender"] = "Nam"
    elif female:
        upd["gender"] = "Nữ"

    # dị ứng
    if _NO_ALLERGY.search(folded):
        upd["no_allergy"] = True
        spans.append(_NO_ALLERGY.search(folded).span())
    else:
        allergens, al_warn, al_err, al_spans = _extract_allergens(text, folded, idx, foodon_catalog)
        if allergens:
            upd["allergens_add"] = allergens
        warnings.extend(al_warn)
        errors.extend(al_err)
        spans.extend(al_spans)

    # mâu thuẫn trong cùng tin nhắn
    if upd.get("pregnant") and upd.get("gender") == "Nam":
        errors.append("Thông tin mâu thuẫn: giới tính **nam** nhưng lại **đang mang thai** — vui lòng kiểm tra lại.")
    if upd.get("lactating") and upd.get("gender") == "Nam":
        errors.append("Thông tin mâu thuẫn: giới tính **nam** nhưng lại **đang cho con bú** — vui lòng kiểm tra lại.")
    age = upd.get("age")
    if age is not None and (upd.get("pregnant") or upd.get("lactating")) and not (10 <= age <= 60):
        errors.append(f"Thông tin không hợp lệ: **{age} tuổi** nhưng lại khai báo mang thai/cho con bú.")
    if upd.get("pregnant") and "gender" not in upd:
        upd["gender"] = "Nữ"

    return {"update": upd, "errors": errors, "warnings": warnings, "notes": notes, "spans": spans}


def _extract_allergens(text, folded, idx, foodon_catalog):
    found, warnings, errors, spans = [], [], [], []
    for m in _ALLERGY.finditer(folded):
        before = folded[max(0, m.start() - 30): m.start()]
        if _ALLERGY_NOT_STATEMENT.search(before) and not _ALLERGY_SELF_BEFORE.search(before):
            continue  # "có gây dị ứng không", "chất gây dị ứng" … là câu hỏi, không phải khai báo
        # danh sách kéo tới hết câu, cắt tại từ dừng
        end = len(folded)
        stop_sentence = re.search(r"[.?!\n]", folded[m.end():])
        if stop_sentence:
            end = m.end() + stop_sentence.start()
        segment = folded[m.end():end]
        stop = _ALLERGY_STOP.search(segment)
        if stop:
            segment = segment[: stop.start()]
            end = m.end() + stop.start()
        if not segment.strip(" ,;"):
            continue
        spans.append((m.start(), end))
        pos = m.end()
        for part in _ALLERGY_SPLIT.split(segment):
            start_in_seg = segment.find(part, pos - m.end()) if part else -1
            if start_in_seg >= 0:
                pos = m.end() + start_in_seg + len(part)
            term_fold = part.strip(" .,:;-")
            if not term_fold or term_fold in _ALLERGY_FILLER:
                continue
            raw = _orig_slice(text, idx, m.end() + start_in_seg, m.end() + start_in_seg + len(part)).strip(" .,:;-") \
                if start_in_seg >= 0 else term_fold
            raw = raw or term_fold
            if len(term_fold) > 40 or len(term_fold.split()) > 5:
                errors.append(f"Không hiểu dị nguyên **'{raw}'** — vui lòng ghi ngắn gọn tên thực phẩm (vd: tôm, đậu phộng, sữa).")
                continue
            if is_gibberish(term_fold):
                errors.append(f"**'{raw}'** không phải tên thực phẩm/dị nguyên hợp lệ — vui lòng nhập lại (vd: tôm, đậu phộng, sữa).")
                continue
            res = pb.resolve_custom_allergen_term(raw, foodon_catalog=foodon_catalog)
            status = res.get("status")
            if status in ("symptom_warning", "drug_warning"):
                warnings.append(f"**'{res.get('term', raw)}'**: {res['warning']} Mục này được bỏ qua.")
            elif status in ("valid", "valid_raw"):
                if not any(_same_allergen(f, res) for f in found):
                    found.append(res)
    return found, warnings, errors, spans


def _same_allergen(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (a.get("key"), fold(a.get("label") or a.get("term", ""))) == (b.get("key"), fold(b.get("label") or b.get("term", "")))


def merge_profile(profile: Dict[str, Any], update: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Gộp thông tin mới vào hồ sơ đang có (thông tin mới ghi đè). Trả về (hồ sơ, ghi chú thay đổi)."""
    p = dict(profile or empty_profile())
    p["conditions"] = list(p.get("conditions", []))
    p["allergens"] = list(p.get("allergens", []))
    notes = []
    if not update:
        return p, notes
    p["declared"] = True
    if update.get("healthy"):
        p["conditions"], p["healthy"] = [], True
    for k in update.get("conditions_add", []):
        if k not in p["conditions"]:
            p["conditions"].append(k)
        p["healthy"] = False
    for k in update.get("conditions_remove", []):
        if k in p["conditions"]:
            p["conditions"].remove(k)
    if update.get("no_allergy"):
        p["allergens"], p["no_allergy"] = [], True
    for a in update.get("allergens_add", []):
        if not any(_same_allergen(x, a) for x in p["allergens"]):
            p["allergens"].append(a)
        p["no_allergy"] = False
    if "age" in update:
        p["age"] = update["age"]
    if "gender" in update:
        if p.get("gender") and p["gender"] != update["gender"]:
            notes.append(f"Đã đổi giới tính trong hồ sơ: {p['gender']} → {update['gender']}.")
        p["gender"] = update["gender"]
    for flag in ("pregnant", "lactating"):
        if flag in update:
            p[flag] = bool(update[flag])
    if (p.get("pregnant") or p.get("lactating")) and p.get("gender") == "Nam":
        p["gender"] = "Nữ"
        notes.append("Hồ sơ có thông tin mang thai/cho con bú nên giới tính được cập nhật thành Nữ.")
    if p.get("age") is not None and p["age"] < 6 and _CHILD_KEY not in p["conditions"]:
        p["conditions"].append(_CHILD_KEY)
    if p.get("age") is not None and p["age"] >= 6 and _CHILD_KEY in p["conditions"]:
        p["conditions"].remove(_CHILD_KEY)
    return p, notes


def profile_to_persona(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Đổi hồ sơ chat thành persona đúng schema mà pipeline đang dùng (qua persona_builder.build_custom_persona)."""
    p = profile or empty_profile()
    labels = [_CONDITION_LABEL[k] for k in p.get("conditions", []) if k in _CONDITION_LABEL]
    persona = pb.build_custom_persona(
        selected_conditions=labels,
        selected_common_allergens=[],
        custom_allergen_terms=list(p.get("allergens", [])),
        age=int(p["age"]) if p.get("age") else 30,
        gender=p.get("gender") or "Khác",
        is_pregnant=bool(p.get("pregnant")),
        is_lactating=bool(p.get("lactating")),
    )
    persona["description"] = profile_summary(p)
    persona["is_declared"] = bool(p.get("declared"))
    return persona


def profile_summary(profile: Dict[str, Any]) -> str:
    p = profile or empty_profile()
    if not p.get("declared"):
        return "Chưa khai báo (đánh giá cho người trưởng thành, không bệnh nền, không dị ứng)"
    parts = []
    if p.get("age"):
        parts.append(f"{p['age']} tuổi")
    if p.get("gender"):
        parts.append(p["gender"])
    if p.get("pregnant"):
        parts.append("đang mang thai")
    if p.get("lactating"):
        parts.append("đang cho con bú")
    conds = [_CONDITION_SHORT.get(k, k) for k in p.get("conditions", []) if k != _CHILD_KEY]
    if _CHILD_KEY in p.get("conditions", []):
        conds.append("Trẻ dưới 6 tuổi")
    if conds:
        parts.append("bệnh nền/đối tượng: " + ", ".join(conds))
    elif p.get("healthy"):
        parts.append("không bệnh nền")
    if p.get("allergens"):
        parts.append("dị ứng: " + ", ".join(a.get("label") or a.get("term", "") for a in p["allergens"]))
    elif p.get("no_allergy"):
        parts.append("không dị ứng")
    return "; ".join(parts) or "Đã khai báo (không có thông tin đặc biệt)"


# ---------------------------------------------------------------------------
# Tách câu hỏi
# ---------------------------------------------------------------------------

_QUESTION_MARK = re.compile(
    r"\?|\b(khong|chua|nao|gi|sao|may|bao nhieu|the nao|duoc khong|co nen|hay khong|phai khong|lieu|a)\s*[.!]*\s*$"
)
_QUESTION_START = re.compile(r"^\s*(lieu|co phai|tai sao|vi sao|bao nhieu|nhu the nao|the nao|cho (toi|minh|em) hoi|hoi)\b")
_IMPERATIVE = re.compile(
    r"\b(kiem tra|phan tich|danh gia|cho (toi|minh|em) biet|xem (giup|gium|ho)|giai thich|tu van|liet ke|tim|doc nhan|"
    r"tra cuu|so sanh)\b"
)
_DOMAIN = re.compile(
    r"\b(san pham|thuc pham|do an|thuc an|mon|nhan|bao bi|thanh phan|an|uong|chua|an toan|phu gia|di ung|di nguyen|ins|adi|"
    r"banh|keo|sua|mi|nuoc|tra|ca phe|bia|ruou|thit|xuc xich|gia vi|nuoc mam|tuong|chat|mau|ngot|beo|duong|muoi|natri|calo|"
    r"gluten|tre|be|mang thai|benh|suc khoe|hop|goi|lon|chai|bao quan|huong lieu|nhiem cheo|dinh duong|han su dung|"
    r"e\d{3}|tieu duong|huyet ap|than|gout)\b"
)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.?!])\s+|\n+", unicodedata.normalize("NFC", str(text or "")))
    return [p.strip() for p in parts if p and p.strip()]


def _is_question(sentence_fold: str) -> bool:
    return bool(_QUESTION_MARK.search(sentence_fold) or _QUESTION_START.search(sentence_fold)
                or _IMPERATIVE.search(sentence_fold))


def _is_profile_only(clause_fold: str, foodon_catalog=None) -> bool:
    """Mệnh đề chỉ khai báo hồ sơ (vd 'tôi 30 tuổi', 'bị tiểu đường', 'dị ứng tôm')."""
    if _is_question(clause_fold):
        return False
    return bool(extract_profile(clause_fold, foodon_catalog)["update"])


def has_domain_signal(question: str) -> bool:
    """Câu hỏi có liên quan tới thực phẩm / nhãn / sức khỏe không (để chặn câu hỏi lạc đề)."""
    q = fold(question)
    if _DOMAIN.search(q) or abstention.detect_out_of_scope(question):
        return True
    try:
        from modules.dynamic_qa import understand_question
        plan = understand_question(question)
    except Exception:
        return False
    if any(i["code"] != "general_safety" for i in plan.get("intents", [])):
        return True
    return any(plan.get(k) for k in ("additives", "unknown_additive_codes", "allergens", "nutrients", "populations"))


# ---------------------------------------------------------------------------
# Phân tích một tin nhắn
# ---------------------------------------------------------------------------


def parse_message(text: str, foodon_catalog=None) -> Dict[str, Any]:
    """Phân loại tin nhắn và tách câu hỏi / hồ sơ.

    Trả về dict:
      kind      : empty | reset | clear_profile | clear_image | greeting | thanks | help | invalid | message
      question  : câu hỏi đã tách ('' nếu không có)
      profile   : kết quả extract_profile (update/errors/warnings)
      errors    : lỗi nhập liệu (chặn xử lý)
      warnings  : cảnh báo (không chặn)
      ignored   : các câu không hiểu (đã bỏ qua)
    """
    raw = unicodedata.normalize("NFC", str(text or "")).strip()
    result = {"kind": "message", "question": "", "profile": {"update": {}, "errors": [], "warnings": []},
              "errors": [], "warnings": [], "ignored": []}
    if not raw:
        result["kind"] = "empty"
        return result
    if len(raw) > MAX_MESSAGE_CHARS:
        result["kind"] = "invalid"
        result["errors"].append(f"Tin nhắn quá dài ({len(raw)} ký tự). Vui lòng rút gọn dưới {MAX_MESSAGE_CHARS} ký tự.")
        return result

    f = fold(raw)
    if _RESET.match(f):
        result["kind"] = "reset"
        return result
    if _CLEAR_PROFILE.match(f):
        result["kind"] = "clear_profile"
        return result
    if _CLEAR_IMAGE.match(f):
        result["kind"] = "clear_image"
        return result
    if is_gibberish(raw):
        result["kind"] = "invalid"
        result["errors"].append(
            "Không hiểu nội dung bạn nhập (có vẻ là ký tự gõ ngẫu nhiên). Vui lòng nhập câu hỏi bằng tiếng Việt có nghĩa, "
            "ví dụ: *“Sản phẩm này có chứa dị nguyên nào không?”*"
        )
        return result
    words = len(re.findall(r"[^\W_]+", f))
    if _GREETING.match(f) and words <= 5:
        result["kind"] = "greeting"
        return result
    if _THANKS.match(f) and words <= 6:
        result["kind"] = "thanks"
        return result
    if _HELP.search(f) and words <= 10:
        result["kind"] = "help"
        return result

    profile = extract_profile(raw, foodon_catalog)
    result["profile"] = profile
    result["errors"].extend(profile["errors"])
    result["warnings"].extend(profile["warnings"])

    question_parts: List[str] = []
    for sent in _split_sentences(raw):
        sf = fold(sent)
        if _is_question(sf):
            # bỏ các mệnh đề thuần khai báo hồ sơ khỏi câu hỏi ("Tôi bị tiểu đường, sản phẩm này ăn được không?")
            clauses = [c.strip() for c in re.split(r",|;", sent) if c.strip()]
            kept = [c for c in clauses if not _is_profile_only(fold(c), foodon_catalog)]
            question_parts.append(", ".join(kept) if kept else sent)
        elif extract_profile(sent, foodon_catalog)["update"] or _ALLERGY.search(sf):
            continue  # câu khai báo hồ sơ
        elif is_gibberish(sent):
            result["ignored"].append(sent)
        elif _DOMAIN.search(sf):
            question_parts.append(sent)  # câu trần thuật nhưng nói về sản phẩm → coi là yêu cầu
        else:
            result["ignored"].append(sent)

    question = " ".join(p for p in question_parts if p).strip()
    if question and len(re.findall(r"[^\W_]+", question)) < 2:
        result["errors"].append("Câu hỏi quá ngắn — vui lòng hỏi đầy đủ hơn, ví dụ: *“Sản phẩm này có an toàn cho trẻ em không?”*")
        question = ""
    if question and not has_domain_signal(question):
        result["errors"].append(
            "Câu hỏi nằm ngoài phạm vi hệ thống. FoodHealthQA chỉ trả lời câu hỏi về **phụ gia, dị nguyên, dinh dưỡng "
            "và an toàn của sản phẩm trên nhãn** (vd: *“Sản phẩm có chứa chất bảo quản nào?”*)."
        )
        question = ""
    result["question"] = question

    if not question and not profile["update"] and not result["errors"] and not result["warnings"]:
        result["kind"] = "invalid"
        result["errors"].append(
            "Chưa nhận ra câu hỏi hay thông tin hồ sơ nào trong tin nhắn. Bạn có thể hỏi về sản phẩm "
            "(*“Có chứa dị nguyên nào không?”*) hoặc khai báo hồ sơ (*“Tôi 40 tuổi, bị tăng huyết áp, dị ứng tôm”*)."
        )
    if result["errors"] or (not question and not profile["update"]):
        result["kind"] = "invalid"
    if result["ignored"] and not result["errors"]:
        result["warnings"].append("Bỏ qua phần không hiểu: " + "; ".join(f"“{s}”" for s in result["ignored"][:3]))
    return result


# ---------------------------------------------------------------------------
# Ảnh
# ---------------------------------------------------------------------------


def validate_image(data: bytes, name: str = "") -> Tuple[Optional[Any], Optional[str]]:
    """Mở và kiểm tra ảnh nhãn. Trả về (PIL.Image, None) hoặc (None, thông báo lỗi)."""
    from PIL import Image, ImageStat, UnidentifiedImageError

    if not data:
        return None, f"Tệp **{name or 'ảnh'}** rỗng — vui lòng chọn ảnh khác."
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        return None, f"Ảnh **{name}** quá lớn (> {MAX_IMAGE_MB} MB). Vui lòng nén hoặc chụp lại."
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return None, f"Không đọc được tệp **{name or 'ảnh'}** — tệp hỏng hoặc không phải ảnh JPEG/PNG."
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) < MIN_IMAGE_SIDE:
        return None, f"Ảnh **{name}** quá nhỏ ({w}×{h}px). Cần ảnh tối thiểu {MIN_IMAGE_SIDE}px mỗi chiều để đọc chữ trên nhãn."
    stat = ImageStat.Stat(img.convert("L"))
    if stat.stddev[0] < 4:
        return None, f"Ảnh **{name}** gần như một màu (trống/đen/trắng) — không thể đọc nhãn. Vui lòng chụp lại."
    return img, None


# ---------------------------------------------------------------------------
# Quyết định bước tiếp theo
# ---------------------------------------------------------------------------


def next_action(has_image: bool, question: str, out_of_scope: Optional[List[str]] = None) -> str:
    """run | need_image | need_question | out_of_scope_no_image"""
    if question and has_image:
        return "run"
    if question and out_of_scope:
        return "out_of_scope_no_image"
    if question:
        return "need_image"
    return "need_question"


def out_of_scope_reply(question: str) -> Optional[str]:
    """Thông báo từ chối theo luật vượt phạm vi (dùng lại abstention.describe) — không cần ảnh."""
    reasons = abstention.detect_out_of_scope(question)
    if not reasons:
        return None
    d = abstention.describe([abstention.OUT_OF_SCOPE], [], reasons)
    return f"**{d.get('title', '')}**\n\n{d.get('message', '')}"
