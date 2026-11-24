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
ALLOWED_IMAGE_EXT = ("jpg", "jpeg", "png", "webp", "bmp")
# Hộp chọn tệp của ô chat chỉ hiện các loại ảnh này; loại nào ngoài ALLOWED_IMAGE_EXT sẽ được báo lỗi tiếng Việt.
PICKER_IMAGE_EXT = ALLOWED_IMAGE_EXT + ("gif", "tif", "tiff", "heic", "heif", "avif")
IMAGE_FORMAT_HINT = "Chỉ nhận ảnh nhãn: " + ", ".join(e.upper() for e in ALLOWED_IMAGE_EXT if e != "jpeg") + f" · tối đa {MAX_IMAGE_MB} MB · 1 ảnh mỗi lần"
# Giới hạn tải lên của ô chat đặt rộng hơn giới hạn thật, để file quá cỡ/định dạng sai vẫn tới được app
# và được báo lỗi rõ ràng trong cuộc trò chuyện (thay vì bị trình duyệt chặn âm thầm).
UPLOAD_HARD_LIMIT_MB = 50

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
# trả lời khi được hỏi có dùng lại ảnh đã gửi trước đó không
_USE_OLD_IMAGE = re.compile(r"^\s*((dung|su dung|giu|lay|hoi tren)( lai)?\s+)?(anh|hinh) (cu|truoc|truoc do|lan truoc|cu di|cu nhe)\s*[.!]*\s*$")
# trả lời khi được hỏi hồ sơ: xác nhận không có bệnh nền / dị ứng
_SKIP_PROFILE = re.compile(
    r"^\s*(bo qua|skip|khong|khong co|khong can|khong co gi|khong co ho so|khong benh gi|khoe manh|binh thuong|"
    r"toi khoe manh|toi binh thuong|khong co benh( nen)? (va|hay|hoac) di ung( gi)?)\s*[.!]*\s*$"
)
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
# nhóm tuổi nêu bằng lời (không có số tuổi): "người trưởng thành", "người cao tuổi"…
_ELDERLY = re.compile(r"\bnguoi (gia|cao tuoi|lon tuoi|cao nien)\b|\bcao tuoi\b")
_ADULT = re.compile(r"\bnguoi (truong thanh|lon|da truong thanh)\b|\bnguoi truong thanh\b")
AGE_GROUP_LABEL = {"adult": "người trưởng thành", "elderly": "người cao tuổi"}
AGE_GROUP_DEFAULT_AGE = {"adult": 30, "elderly": 65}


def age_group_of(age):
    if age is None:
        return None
    return "elderly" if age >= 60 else "adult" if age >= 18 else "child"


# "8 tháng tuổi", hoặc "bé (nhà tôi) 8 tháng" — số tháng ngay sau bé/con/cháu/trẻ mới là tuổi ("dùng 8 tháng" thì không)
_AGE_MONTHS = re.compile(r"(?<![\d.,])(\d{1,2})\s*thang tuoi|\b(?:be|con|chau|tre)(?:\s+(?:nha|toi|minh|em|cua|duoc|"
                         r"moi|chi|tam|khoang|gan|da))*\s+(\d{1,2})\s*thang\b")
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
    r"lam|bi|mac|tuoi|mang thai|benh|dang|cho|hoi|xin|giup|kiem tra|phan tich|"
    r"be|con|chau|toi|minh|em|tui|nguoi|vo|chong|bo|ba|ong)\b"  # đại từ chủ ngữ mở mệnh đề mới
)
_ALLERGY_SPLIT = re.compile(r",|;|/|\+|\bva\b|\bhoac\b|\bvoi\b|\bcung\b")
_ALLERGY_FILLER = {"gi", "nao", "do", "cai gi", "thu gi", "gi het", "gi ca", "a", "ca", "nhieu thu"}
_NO_ALLERGY = re.compile(r"(khong|chua|khong he)\s+(bi\s+|co\s+)?di ung\s*(gi|thu gi|nao)?\b")


def empty_profile() -> Dict[str, Any]:
    return {
        "conditions": [],      # mã bệnh nền (khóa trong CONDITION_CHOICES)
        "allergens": [],       # kết quả resolve_custom_allergen_term (status valid / valid_raw)
        "age": None,
        "age_months": None,    # chỉ khi khai báo theo tháng ("8 tháng tuổi")
        "age_group": None,     # adult | elderly | child (suy từ tuổi hoặc từ cụm "người trưởng thành"…)
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
        # "đang cho con bú, bé 3 tháng tuổi": số tháng là tuổi của con, không phải của người khai báo
        if not upd.get("lactating") and not upd.get("pregnant"):
            upd["age"] = 1
            upd["age_months"] = int(m.group(1) or m.group(2))
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

    # nhóm tuổi bằng lời
    m_group = _ELDERLY.search(folded) or _ADULT.search(folded)
    if m_group:
        spans.append(m_group.span())
        group = "elderly" if _ELDERLY.search(folded) else "adult"
        age_val = upd.get("age")
        if age_val is not None and age_group_of(age_val) != group:
            errors.append(f"Thông tin mâu thuẫn: **{age_val} tuổi** nhưng lại là **{AGE_GROUP_LABEL[group]}** — vui lòng kiểm tra lại.")
        elif age_val is None:
            upd["age_group"] = group

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
                if res.get("source") == "foodon_search" and len(term_fold.replace(" ", "")) <= 3:
                    # từ quá ngắn chỉ khớp mờ trên FoodOn (vd 'bé' → 'Béchamel sauce') → không tin cậy
                    warnings.append(f"Không chắc **'{raw}'** là dị nguyên nào — mục này được bỏ qua, vui lòng ghi rõ tên thực phẩm.")
                    continue
                if not any(_same_allergen(f, res) for f in found):
                    found.append(res)
    return found, warnings, errors, spans


def _same_allergen(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (a.get("key"), fold(a.get("label") or a.get("term", ""))) == (b.get("key"), fold(b.get("label") or b.get("term", "")))


def merge_profile(profile: Dict[str, Any], update: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Cập nhật hồ sơ theo thông tin mới. Trả về (hồ sơ, ghi chú thay đổi).

    - Thông tin mới mâu thuẫn hồ sơ cũ (khác tuổi hoặc giới tính) → coi là người khác: THAY MỚI toàn bộ hồ sơ,
      không gộp bệnh nền/dị ứng của người trước.
    - Ngược lại → BỔ SUNG vào hồ sơ đang có (thêm/bớt bệnh nền, dị ứng; cập nhật thai kỳ…).
    """
    old = profile or empty_profile()
    notes = []
    if not update:
        p = dict(old)
        p["conditions"], p["allergens"] = list(old.get("conditions", [])), list(old.get("allergens", []))
        return p, notes
    conflict = []
    new_group = age_group_of(update["age"]) if update.get("age") is not None else update.get("age_group")
    old_group = age_group_of(old["age"]) if old.get("age") is not None else old.get("age_group")
    if new_group and old_group and new_group != old_group and not (update.get("age") is not None and old.get("age") is not None):
        conflict.append(f"nhóm tuổi {AGE_GROUP_LABEL.get(old_group, 'trẻ em')} → {AGE_GROUP_LABEL.get(new_group, 'trẻ em')}")
    if update.get("age") is not None and old.get("age") is not None and update["age"] != old["age"]:
        conflict.append(f"tuổi {old['age']} → {update['age']}")
    if update.get("gender") and old.get("gender") and update["gender"] != old["gender"]:
        conflict.append(f"giới tính {old['gender']} → {update['gender']}")
    if conflict and old.get("declared"):
        notes.append("Thông tin mới khác hồ sơ trước (" + ", ".join(conflict) + ") nên hồ sơ được **thay mới**, "
                     "không gộp với hồ sơ cũ: " + profile_summary(old) + ".")
        old = empty_profile()
    p = dict(old)
    p["conditions"] = list(old.get("conditions", []))
    p["allergens"] = list(old.get("allergens", []))
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
    if update.get("age_group"):
        p["age_group"] = update["age_group"]
        if p.get("age") is not None and age_group_of(p["age"]) != update["age_group"]:
            p["age"] = None
    if "age" in update:
        p["age_group"] = age_group_of(update["age"])
        p["age"] = update["age"]
        p["age_months"] = update.get("age_months")  # khai báo tuổi theo năm thì bỏ số tháng cũ
    if "gender" in update:
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
        age=int(p["age"]) if p.get("age") else AGE_GROUP_DEFAULT_AGE.get(p.get("age_group"), 30),
        gender=p.get("gender") or "Khác",
        is_pregnant=bool(p.get("pregnant")),
        is_lactating=bool(p.get("lactating")),
    )
    if p.get("age_months") is not None:
        persona.setdefault("demographics", {})["age_months"] = int(p["age_months"])
    persona["description"] = profile_summary(p)
    persona["is_declared"] = bool(p.get("declared"))
    return persona


def profile_summary(profile: Dict[str, Any]) -> str:
    p = profile or empty_profile()
    if not p.get("declared"):
        return "Chưa khai báo"
    parts = []
    if p.get("age_months") is not None:
        parts.append(f"{p['age_months']} tháng tuổi")
    elif p.get("age"):
        parts.append(f"{p['age']} tuổi")
    elif p.get("age_group") in AGE_GROUP_LABEL:
        parts.append(AGE_GROUP_LABEL[p["age_group"]])
    if p.get("gender"):
        parts.append(p["gender"])
    if p.get("pregnant"):
        parts.append("đang mang thai")
    if p.get("lactating"):
        parts.append("đang cho con bú")
    # "Trẻ dưới 6 tuổi" do hệ thống suy ra từ tuổi (merge_profile) nên hiện ở dòng nhóm tuổi, không lẫn vào bệnh nền người dùng khai
    if _CHILD_KEY in p.get("conditions", []):
        parts.append("nhóm tuổi: trẻ dưới 6 tuổi")
    conds = [_CONDITION_SHORT.get(k, k) for k in p.get("conditions", []) if k != _CHILD_KEY]
    if conds:
        parts.append("bệnh nền/đối tượng: " + ", ".join(conds))
    elif p.get("healthy"):
        parts.append("không bệnh nền")
    if p.get("allergens"):
        parts.append("dị ứng: " + ", ".join(a.get("label") or a.get("term", "") for a in p["allergens"]))
    elif p.get("no_allergy"):
        parts.append("không dị ứng")
    if not (conds or p.get("healthy") or p.get("allergens") or p.get("no_allergy")):
        parts.append("chưa nêu bệnh nền/dị ứng")
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


# câu hỏi không có dấu "?" và không kết thúc bằng từ hỏi: "sản phẩm này có an toàn cho người trưởng thành"
_QUESTION_INNER = re.compile(
    r"\bco (an toan|nen|duoc|chua|gay|phu hop|hai|tot|sao|van de)\b|\b(an|uong|dung|su dung) duoc\b|"
    r"\ban toan (cho|voi|khong)\b|\bphu hop (cho|voi)\b"
)


def _is_question(sentence_fold: str) -> bool:
    return bool(_QUESTION_MARK.search(sentence_fold) or _QUESTION_START.search(sentence_fold)
                or _IMPERATIVE.search(sentence_fold) or _QUESTION_INNER.search(sentence_fold))


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
    if _USE_OLD_IMAGE.match(f):
        result["kind"] = "use_old_image"
        return result
    if _SKIP_PROFILE.match(f):
        result["kind"] = "skip_profile"
        return result

    # câu sửa thông tin nhãn ("không có E621", "mã là E330 chứ không phải E33") → tách riêng, phần còn lại xử lý như cũ
    correction, rest = detect_label_correction(raw)
    if correction:
        result["correction"] = correction
        if not rest.strip():
            result["kind"] = "correction"
            return result
        raw = rest

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


_NOT_NAME = re.compile(
    r"thanh phan|ingredient|nsx|hsd|han su dung|ngay san xuat|khoi luong|the tich|cong ty|dia chi|san xuat tai|"
    r"phan phoi|chiu trach nhiem|huong dan|bao quan|dien thoai|www|http|@|ma vach|so lo|xuat xu|net|kcal|gia tri dinh duong|"
    r"su dung|danh cho|huong dan|bao quan|tre (tu|em|duoi|nho)|lac (deu|ky)|dung ngay|dung lien|pha voi|nha san xuat|xuat xu|cach dung|luu y|canh bao|khuyen cao"
)


_PRODUCT_TYPE = re.compile(
    r"\b(banh|keo|sua|mi|mien|pho|bun|chao|nuoc|tra|ca phe|cafe|coffee|snack|xuc xich|lap xuong|cha|gio|pate|"
    r"nuoc mam|nuoc tuong|tuong|dau an|dau|hat nem|bot nem|gia vi|sot|mut|thach|kem|socola|chocolate|ngu coc|"
    r"yen mach|bia|ruou|nuoc ngot|nuoc ep|sua chua|pho mai|bo|ca hop|thit hop|rong bien|khoai tay|bim bim)\b"
)
_NAME_STOP = re.compile(r"[:;.!?\d]|\b(tong so|khoi luong|kich co|the tich|net|thanh phan|nsx|hsd|san xuat|su dung|danh cho|huong dan|bao quan|tre (tu|em|duoi|nho)|lac (deu|ky)|dung ngay|dung lien|pha voi|nha san xuat|xuat xu|cach dung|luu y|canh bao|khuyen cao)\b")


def display_name_from_file(filename: str, max_len: int = 40) -> str:
    """Tên file ảnh (bỏ đuôi) dùng làm tên sản phẩm khi không đoán được từ nhãn; tên quá dài rút gọn ở giữa."""
    stem = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    stem = stem.strip() or "ảnh nhãn"
    return stem if len(stem) <= max_len else stem[: max_len - 13] + "…" + stem[-12:]


def guess_product_name(ocr_text: str, max_len: int = 60, max_words: int = 8) -> str:
    """Đoán tên sản phẩm từ phần chữ đứng TRƯỚC 'Thành phần' trên nhãn ('' nếu không đoán được).

    1) cụm bắt đầu bằng từ chỉ loại sản phẩm (bánh, kẹo, sữa, mì…) — dùng được cả khi OCR trả về một dòng liền;
    2) nếu không có: dòng chữ có nghĩa đầu tiên.
    """
    text = str(ocr_text or "")
    folded, idx = _fold_map(text)
    head_end = re.search(r"thanh\s*phan|ingredients?", folded)
    head = folded[: head_end.start()] if head_end else folded[:300]
    m = _PRODUCT_TYPE.search(head)
    if m:
        stop = _NAME_STOP.search(head, m.start() + 1)
        end = stop.start() if stop else len(head)
        name = _orig_slice(text, idx, m.start(), end).strip(" -,|")
        name = " ".join(name.split()[:max_words])[:max_len].strip()
        if len(name.split()) >= 2:
            return name
    lines = [l.strip(" -:•*|") for l in text.splitlines()]
    for line in lines:
        f = fold(line)
        if re.search(r"thanh\s*phan|ingredients?", f):
            break
        words = re.findall(r"[^\W\d_]{2,}", line)
        if len(words) < 2 or _NOT_NAME.search(f) or is_gibberish(line):
            continue
        return line[:max_len].strip()
    return ""


def validate_image(data: bytes, name: str = "") -> Tuple[Optional[Any], Optional[str]]:
    """Mở và kiểm tra ảnh nhãn. Trả về (PIL.Image, None) hoặc (None, thông báo lỗi)."""
    from PIL import Image, ImageStat, UnidentifiedImageError

    ext = name.rsplit(".", 1)[-1].lower() if "." in (name or "") else ""
    if name and ext not in ALLOWED_IMAGE_EXT:
        return None, (f"Tệp **{name}** không được hỗ trợ (định dạng *.{ext or '?'}*). "
                      "Vui lòng gửi ảnh nhãn dạng **" + ", ".join(e.upper() for e in ALLOWED_IMAGE_EXT if e != "jpeg") + "**.")
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
        return None, f"Không đọc được tệp **{name or 'ảnh'}** — tệp hỏng hoặc không phải ảnh " + ", ".join(e.upper() for e in ALLOWED_IMAGE_EXT if e != "jpeg") + "."
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


# ---------------------------------------------------------------------------
# Câu hỏi nối tiếp ("chất đó có an toàn không?")
# ---------------------------------------------------------------------------

# từ chỉ trỏ tới đối tượng đã nói ở câu trước (văn bản đã bỏ dấu)
_REFERENCE = re.compile(
    r"\b(chat|phu gia|thanh phan|di nguyen|ma|loai|cac chat|nhung chat|cac phu gia)\s+"
    r"(do|nay|kia|ay|tren|vua roi|vua noi|vua hoi|vua neu|o tren)\b"
    r"|\b(no|chung|chung no)\b"
    r"|\bcon\b.{0,40}\bthi sao\b|\bthe con\b|\bvay con\b"
)


def focus_from_qa(qa: dict | None) -> list[str]:
    """Đối tượng câu hỏi vừa được trả lời nhắc tới (để hiểu câu nối tiếp): mã phụ gia + dị nguyên (từ khóa tiếng Việt)."""
    if not qa:
        return []
    ents = qa.get("entities") or {}
    terms = [str(c) for c in ents.get("additives") or [] if c]
    try:
        from modules.constraint_model import ALLERGEN_KEYWORDS
    except Exception:
        ALLERGEN_KEYWORDS = {}
    for a in ents.get("allergens") or []:
        words = ALLERGEN_KEYWORDS.get(a) or [a]
        terms.append(str(words[0]))
    return list(dict.fromkeys(terms))


# nhóm chức năng trong câu hỏi (đã bỏ dấu) → category của phụ gia đã liên kết
_CLASS_WORDS = (
    (r"tao ngot|chat ngot", {"sweetener"}),
    (r"bao quan", {"preservative"}),
    (r"tao mau|pham mau|mau tong hop|mau tu nhien", {"color", "colour"}),
    (r"nhu hoa", {"emulsifier"}),
    (r"dieu vi|bot ngot", {"flavor_enhancer", "flavour_enhancer"}),
    (r"on dinh", {"stabilizer", "stabiliser"}),
    (r"lam day|lam dac", {"thickener"}),
    (r"chong oxy hoa|chong oxi hoa", {"antioxidant"}),
    (r"dieu chinh do acid|dieu chinh do axit|acid|axit", {"acidity_regulator"}),
    (r"tao xop|nang no", {"raising_agent", "leavening_agent"}),
)


def focus_from_turn(qa, question, linked_results=None, answer="", limit=12) -> list[str]:
    """Đối tượng mà một lượt hỏi–đáp thực sự nói tới (để hiểu câu nối tiếp "chất đó…"):
    1) phụ gia/dị nguyên nêu trong câu hỏi; 2) nếu câu hỏi chỉ nêu nhóm ("chất tạo ngọt") → các mã trên nhãn thuộc
    nhóm đó; 3) nếu vẫn không có → các mã INS/E mà câu trả lời nêu ra."""
    terms = focus_from_qa(qa)
    if terms:
        return terms[:limit]
    q = fold(question)
    cats = set()
    for pattern, c in _CLASS_WORDS:
        if re.search(pattern, q):
            cats |= c
    if cats:
        codes = [str((it.get("matched_entity") or {}).get("ins_code") or "") for it in (linked_results or [])
                 if it.get("matched_entity") and str(it.get("category") or "").lower() in cats]
        codes = [c for c in dict.fromkeys(codes) if c]
        if codes:
            return codes[:limit]
    from modules.additive_linker import _EXPLICIT_CODE_RE
    codes = [f"INS {m.group('code').strip()}" for m in _EXPLICIT_CODE_RE.finditer(str(answer or ""))]
    return list(dict.fromkeys(codes))[:limit]


def resolve_followup(question: str, last_focus: list[str] | None) -> tuple[str, list[str]]:
    """Câu hỏi không nêu đối tượng nhưng chỉ trỏ tới câu trước ("chất đó", "nó", "còn … thì sao") → ghép đối tượng
    của câu trước vào câu hỏi gửi cho pipeline. Câu hỏi tự đủ nghĩa (đã nêu phụ gia/dị nguyên) thì giữ nguyên.
    Trả về (câu hỏi gửi pipeline, danh sách đối tượng đã ghép — rỗng nếu không ghép)."""
    question = str(question or "")
    if not question.strip() or not last_focus:
        return question, []
    if not _REFERENCE.search(fold(question)):
        return question, []
    try:
        from modules.dynamic_qa import understand_question
        plan = understand_question(question)
        if plan.get("additives") or plan.get("unknown_additive_codes") or plan.get("allergens"):
            return question, []
    except Exception:
        pass
    terms = list(dict.fromkeys(last_focus))[:12]
    return f"{question} (đối tượng ở câu hỏi trước: {', '.join(terms)})", terms


# ---------------------------------------------------------------------------
# Người dùng sửa thông tin nhãn ngay trong chat ("không có E621", "mã là E330 chứ không phải E33",
# "thành phần đúng là: …") — không hiện danh sách thành phần, không bắt xác nhận; câu sửa chỉ làm hệ thống
# phân tích lại câu hỏi gần nhất với thành phần đã sửa.
# ---------------------------------------------------------------------------

_CODE = r"(?:e|ins)\s*-?\s*\d{2,4}[a-z]?"   # nhận cả 2 chữ số: mã OCR đọc cụt (E33 thay vì E330)
_CORR_REPLACE = (
    re.compile(rf"\b(?:ma\s+)?(?:dung\s+)?la\s+(?P<new>{_CODE})\s*,?\s*(?:chu\s+)?khong\s+phai\s+(?:la\s+)?(?P<old>{_CODE})\b"),
    re.compile(rf"\bkhong\s+phai\s+(?:la\s+)?(?P<old>{_CODE})\s*,?\s*(?:ma\s+)?(?:phai\s+)?la\s+(?P<new>{_CODE})\b"),
    re.compile(rf"\b(?:sua|doi)\s+(?P<old>{_CODE})\s+thanh\s+(?P<new>{_CODE})\b"),
    re.compile(rf"\b(?P<old>{_CODE})\s*,?\s*(?:phai la|dung ra la|that ra la|chinh xac la|dung la)\s+(?P<new>{_CODE})\b"),
)
_CORR_LIST = re.compile(
    r"^\s*(?:(?:dung ra|that ra|thuc ra)\s+)?(?:danh sach\s+)?thanh phan\s+(?:dung|chinh xac|that|that su|tren nhan)"
    r"(?:\s+(?:la|ghi|gom))?\s*:?\s*(?=\S)"
)
_CORR_REMOVE = re.compile(
    r"^\s*(?:(?:nhan|san pham(?: nay)?|sp|no|thanh phan|tren nhan|trong thanh phan)\s+)?"
    r"(?:(?:that ra|thuc ra|thuc te)\s+)?(?:khong (?:he )?(?:co|chua|ghi)|khong he chua)\s+(?P<items>.+?)"
    r"\s*(?:(?:tren|trong)\s+(?:nhan|thanh phan|san pham))?\s*(?:dau|ca|nhe|nha|ma)?\s*[.!]*\s*$"
)
_CORR_MISREAD = re.compile(r"^\s*(?:he thong\s+|ban\s+)?(?:doc|nhan dien|bat)\s+(?:sai|nham)\s+(?:ma\s+|chu\s+)?(?P<items>.+?)\s*[.!]*\s*$")
_CORR_NOT_LABEL = re.compile(r"\b(di ung|benh|toi|minh|tui|em|con toi|be nha)\b")   # khai báo bản thân, không phải sửa nhãn
_CORR_FILLER = {"gi", "nao", "dau", "ca", "chat", "thanh phan", "ma"}


def _canon_code(code_fold: str) -> str:
    m = re.search(r"(\d{2,4}[a-z]?)", code_fold)
    return f"E{m.group(1)}" if m else code_fold.upper()


def _correction_items(sentence, folded, idx, m):
    items = []
    start = m.start("items")
    for part in re.finditer(r"[^,;/]+", folded[start:m.end("items")]):
        for sub in re.split(r"\s+(?:va|hoac|cung nhu)\s+", part.group(0)):
            s = sub.strip()
            if not s or s in _CORR_FILLER:
                continue
            off = start + part.start() + part.group(0).find(s)
            orig = _orig_slice(sentence, idx, off, off + len(s)).strip(" .!")
            items.append(_canon_code(s) if re.fullmatch(_CODE, s) else orig)
    return items


def detect_label_correction(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Tìm câu SỬA thông tin nhãn trong tin nhắn. Trả về (correction | None, phần còn lại của tin nhắn).

    correction = {"remove": [thành phần/mã không có trên nhãn], "replace": [(mã cũ, mã mới)], "ingredients": "…"}.
    Câu hỏi ("không có sữa phải không?") và câu khai báo bản thân ("tôi không có dị ứng") KHÔNG phải câu sửa.
    """
    raw = unicodedata.normalize("NFC", str(text or "")).strip()
    folded, idx = _fold_map(raw)
    m = _CORR_LIST.match(folded)
    if m:
        body = _orig_slice(raw, idx, m.end(), len(folded)).strip()
        if body.count(",") >= 1:
            return {"remove": [], "replace": [], "ingredients": body}, ""
    corr = {"remove": [], "replace": [], "ingredients": ""}
    rest = []
    for sent in _split_sentences(raw):
        sf, sidx = _fold_map(sent)
        found = False
        for pat in _CORR_REPLACE:
            for rm in pat.finditer(sf):
                corr["replace"].append((_canon_code(rm.group("old")), _canon_code(rm.group("new"))))
                found = True
        if not found and not _is_question(sf) and not _CORR_NOT_LABEL.search(sf):
            rm = _CORR_REMOVE.match(sf) or _CORR_MISREAD.match(sf)
            if rm:
                items = _correction_items(sent, sf, sidx, rm)
                if items:
                    corr["remove"].extend(items)
                    found = True
        if not found:
            rest.append(sent)
    if not corr["remove"] and not corr["replace"]:
        return None, raw
    corr["remove"] = list(dict.fromkeys(corr["remove"]))
    corr["replace"] = list(dict.fromkeys(corr["replace"]))
    return corr, " ".join(rest)


def _codes_of(text_fold: str) -> set:
    codes = {m.group(1) for m in re.finditer(r"\b(?:e|ins)\s*-?\s*(\d{2,4}[a-z]?)\b", text_fold)}
    codes |= {m.group(1) for m in re.finditer(r"\((?:[^()]*?[,;]\s*)?(\d{3,4}[a-z]?)(?=\s*[,;)])", text_fold)}
    return codes


def _code_pattern(code: str) -> re.Pattern:
    digits = re.escape(_canon_code(fold(code))[1:])
    return re.compile(rf"(?i)(?:\b(?:E|INS)\s*-?\s*{digits}|(?<![\w.,]){digits})(?!\w)")


def apply_label_correction(mentions, correction, parse_fn=None):
    """Áp câu sửa lên danh sách thành phần đã đọc. Trả về (thành phần mới, các việc đã làm, mục không tìm thấy)."""
    mentions = [dict(m) for m in (mentions or [])]
    done, missing = [], []
    if correction.get("ingredients") and parse_fn is not None:
        body = correction["ingredients"]
        new = parse_fn(body if fold(body).startswith("thanh phan") else f"Thành phần: {body}",
                       include_allergen_statements=True)
        if new:
            return new, [f"dùng danh sách thành phần bạn gửi ({len(new)} thành phần)"], []
        missing.append("danh sách thành phần bạn gửi (không tách được thành phần nào)")
    for old, new_code in correction.get("replace", []):
        pat, hit = _code_pattern(old), False
        for m in mentions:
            if pat.search(m.get("raw_mention", "")):
                m["raw_mention"] = pat.sub(new_code, m["raw_mention"])
                hit = True
        (done.append(f"đổi **{old} → {new_code}**") if hit else missing.append(old))
    for term in correction.get("remove", []):
        tf = fold(term)
        is_code = bool(re.fullmatch(r"e\d{2,4}[a-z]?", tf))
        kept, hit = [], False
        for m in mentions:
            mf = fold(m.get("raw_mention", ""))
            if is_code:
                codes = _codes_of(mf)
                if tf[1:] in codes:
                    hit = True
                    if len(codes) > 1:   # nhóm nhiều mã → chỉ bỏ mã đó
                        text = _code_pattern(term).sub("", m["raw_mention"])
                        text = re.sub(r"\(\s*[,;]\s*", "(", text)
                        text = re.sub(r"\s*[,;]\s*\)", ")", text)
                        text = re.sub(r"([,;])\s*[,;]", r"\1", text)
                        m["raw_mention"] = re.sub(r"\s{2,}", " ", text).strip()
                        kept.append(m)
                    continue
            elif re.search(rf"(?<!\w){re.escape(tf)}(?!\w)", mf):
                hit = True
                continue
            kept.append(m)
        mentions = kept
        (done.append(f"bỏ **{term}**") if hit else missing.append(term))
    return mentions, done, missing


def next_action(has_image: bool, question: str, out_of_scope: Optional[List[str]] = None,
                has_profile: bool = True) -> str:
    """run | need_image | need_profile | need_image_profile | need_question | out_of_scope_no_image

    Không tự gán hồ sơ mặc định: chưa khai báo hồ sơ thì giữ câu hỏi lại và đề nghị người dùng bổ sung
    (hoặc trả lời "bỏ qua" / "không có" để xác nhận không bệnh nền, không dị ứng).
    """
    if not question:
        return "need_question"
    if out_of_scope and not has_image:
        return "out_of_scope_no_image"
    if not has_image and not has_profile:
        return "need_image_profile"
    if not has_image:
        return "need_image"
    if not has_profile:
        return "need_profile"
    return "run"


def out_of_scope_reply(question: str) -> Optional[str]:
    """Thông báo từ chối theo luật vượt phạm vi (dùng lại abstention.describe) — không cần ảnh."""
    reasons = abstention.detect_out_of_scope(question)
    if not reasons:
        return None
    d = abstention.describe([abstention.OUT_OF_SCOPE], [], reasons)
    return f"**{d.get('title', '')}**\n\n{d.get('message', '')}"
