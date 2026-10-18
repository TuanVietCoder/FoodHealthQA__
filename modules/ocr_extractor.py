"""
Module OCR Extractor cho hệ thống FoodHealthQA (v10 - Ultra-Compatible PaddleOCR)
Tích hợp trọn vẹn:
1. Tiền xử lý quang học thích ứng (Adaptive Optical Preprocessing):
   - Deskewing (Khử góc nghiêng bằng MinAreaRect / Contour analysis)
   - Unsharp Masking (Tăng cường độ sắc nét cho chữ in nhỏ < 6pt)
   - Cân bằng tương phản thích ứng (Adaptive CLAHE có kiểm soát độ lệch chuẩn)
2. Bộ nhận diện PaddleOCR tối ưu hóa tham số cho tiếng Việt:
   - Tương thích chéo mọi phiên bản PaddleOCR (v2, v3, PaddleX)
   - Tự động bỏ qua tham số deprecated như show_log
3. Tầng Hậu xử lý Phục hồi Chính tả Thực phẩm (Post-OCR Lexical Normalizer):
   - Chuẩn hóa mã phụ gia quốc tế (INS, E-number, La Mã)
   - Khôi phục dấu thanh và bổ sung ký tự bị mất dựa trên Từ điển Thực phẩm (Domain Lexicon)
   - Bóc tách thành phần với Lookahead Regex (bảo toàn cấu trúc ngoặc đơn)
4. Hỗ trợ song song Chế độ Nhãn Chuẩn (Ground Truth Benchmark) và Chế độ Quét Ảnh Thực Tế.
"""

import os
# Vô hiệu hóa triệt để OneDNN / MKLDNN và PIR trên Windows CPU để tránh lỗi OneDnnContext
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import paddle
    paddle.set_flags({'FLAGS_use_mkldnn': False})
except Exception:
    pass

import re
import unicodedata
import cv2
import numpy as np
from PIL import Image

_PADDLE_AVAILABLE = False
_OCR_ENGINE = None
_INIT_ERROR = None

try:
    from paddleocr import PaddleOCR
    _PADDLE_AVAILABLE = True
except Exception as e:
    _PADDLE_AVAILABLE = False
    _INIT_ERROR = str(e)


def is_paddleocr_available():
    """Kiểm tra xem thư viện PaddleOCR đã được cài đặt và có thể nạp hay không."""
    return _PADDLE_AVAILABLE


def get_ocr_engine():
    """
    Khởi tạo PaddleOCR Engine tương thích linh hoạt mọi phiên bản PaddleOCR (v2, v3, PaddleX).
    Ưu tiên cấu hình tối ưu cho nhãn thực phẩm tiếng Việt.
    """
    global _OCR_ENGINE, _INIT_ERROR
    if not is_paddleocr_available():
        return None
    if _OCR_ENGINE is None:
        configs = [
            # Cấu hình 1: Tối ưu cho nhãn thực phẩm (detection thresholds thấp hơn để bắt chữ nhỏ)
            {
                "use_angle_cls": True,
                "lang": "vi",
                "det_db_thresh": 0.2,
                "det_db_box_thresh": 0.4,
                "det_db_unclip_ratio": 1.8,
                "rec_batch_num": 16,
            },
            # Cấu hình 2: Chuẩn cho PaddleOCR 3.x
            {"lang": "vi"},
            # Cấu hình 3: Mặc định
            {},
        ]
        last_err = None
        for cfg in configs:
            try:
                _OCR_ENGINE = PaddleOCR(**cfg)
                if _OCR_ENGINE is not None:
                    _INIT_ERROR = None
                    break
            except Exception as e:
                last_err = str(e)
                continue
        if _OCR_ENGINE is None:
            _INIT_ERROR = last_err
            print(f"Cảnh báo: Không thể khởi tạo PaddleOCR: {_INIT_ERROR}")
            return None
    return _OCR_ENGINE


def convert_to_cv2_image(image_input):
    """Chuyển đổi linh hoạt đầu vào thành ảnh mảng OpenCV BGR uint8."""
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_input}")
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_input}")
        return img_bgr
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert('RGB'))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    else:
        raise ValueError("Định dạng ảnh không hợp lệ. Hỗ trợ đường dẫn file, PIL.Image hoặc numpy.ndarray.")


# =====================================================================
# 1. THUẬT TOÁN TIỀN XỬ LÝ QUANG HỌC THÍCH ỨNG (OPTICAL PREPROCESSING)
# =====================================================================

def upscale_image(img_bgr, min_height=1500):
    """
    Phóng to ảnh nếu chiều cao nhỏ hơn ngưỡng tối thiểu.
    Chữ nhỏ trên nhãn bao bì cần ít nhất ~1500px chiều cao để OCR chính xác.
    """
    h, w = img_bgr.shape[:2]
    if h < min_height:
        scale = min_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img_bgr


def denoise_image(img_bgr):
    """Khử nhiễu bằng Bilateral Filter: giữ viền chữ sắc nét, loại bỏ nhiễu nền."""
    try:
        return cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    except Exception:
        return img_bgr


def deskew_image(img_bgr):
    """
    Thuật toán Khử góc nghiêng (Deskewing):
    Xác định góc xoay của ảnh dựa trên cấu trúc các khối văn bản và xoay lại thẳng góc.
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return img_bgr

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 0.5 and abs(angle) < 25.0:
            (h, w) = img_bgr.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        return img_bgr
    except Exception:
        return img_bgr


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    """Tăng cường độ tương phản cục bộ ở viền các ký tự."""
    try:
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, 0)
        sharpened = np.minimum(sharpened, 255)
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened
    except Exception:
        return image


def adaptive_clahe(img_bgr):
    """Cân bằng tương phản thích ứng (Adaptive CLAHE) trên kênh L của LAB."""
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        std_dev = np.std(l)
        clip_limit = 3.0 if std_dev < 40.0 else 2.0
        if std_dev < 60.0:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return img_bgr
    except Exception:
        return img_bgr


def binarize_for_ocr(img_bgr):
    """Nhị phân hóa thích ứng cho trường hợp nền phức tạp (gradient, hoa văn)."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=31, C=10
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    except Exception:
        return img_bgr


def adaptive_preprocess_image(img_bgr):
    """Quy trình tiền xử lý thích ứng tổng hợp trước khi đưa vào PaddleOCR."""
    step1 = upscale_image(img_bgr)
    step2 = deskew_image(step1)
    step3 = denoise_image(step2)
    step4 = adaptive_clahe(step3)
    step5 = unsharp_mask(step4, amount=1.5)
    return step5


def preprocess_variant_binary(img_bgr):
    """Biến thể tiền xử lý nhị phân cho multi-pass OCR."""
    step1 = upscale_image(img_bgr)
    step2 = deskew_image(step1)
    step3 = binarize_for_ocr(step2)
    return step3


def preprocess_variant_grayscale_sharp(img_bgr):
    """Biến thể tiền xử lý grayscale tăng cường sắc nét cho multi-pass OCR."""
    step1 = upscale_image(img_bgr)
    step2 = deskew_image(step1)
    gray = cv2.cvtColor(step2, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    sharpened = unsharp_mask(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), amount=2.0)
    return sharpened


# =====================================================================
# 2. TẦNG HẬU XỬ LÝ PHỤC HỒI CHÍNH TẢ THỰC PHẨM (LEXICAL NORMALIZER)
# =====================================================================

_VIET_CHARS = r'a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'

def _fix_word_boundaries(t):
    """Tách các từ bị dính nhau do OCR không nhận diện được khoảng trắng.
    Chỉ xử lý cấu trúc (structural), KHÔNG sửa chính tả — LLM sẽ đảm nhiệm."""
    t = re.sub(rf'([{_VIET_CHARS}])(INS\b)', r'\1 \2', t, flags=re.IGNORECASE)
    t = re.sub(rf'([{_VIET_CHARS}])(\d{{3,4}})', r'\1 \2', t, flags=re.IGNORECASE)
    t = re.sub(rf'(\d{{3,4}})([{_VIET_CHARS}]{{3,}})', r'\1 \2', t, flags=re.IGNORECASE)
    t = re.sub(rf'\)([{_VIET_CHARS}])', r') \1', t, flags=re.IGNORECASE)
    t = re.sub(rf'([{_VIET_CHARS}])\(', r'\1 (', t, flags=re.IGNORECASE)
    return t


def _normalize_ins_codes(t):
    """Chuẩn hóa mã phụ gia quốc tế INS/E-number."""
    # Sửa tiền tố INS bị OCR nhầm
    t = re.sub(r'\b[l1iI|]NS[\.\s]*', 'INS ', t)
    t = re.sub(r'\bE[\.\s]*([0-9]{3,4})', r'E\1', t)

    # Chuẩn hóa dạng ngoặc lồng mã INS
    t = re.sub(r'(?:INS\s*)?322\s*\(\s*i\s*\)?', 'INS 322i ', t, flags=re.IGNORECASE)
    t = re.sub(r'(?:INS\s*)?500\s*\(\s*ii\s*\)?', 'INS 500ii ', t, flags=re.IGNORECASE)
    t = re.sub(r'(?:INS\s*)?503\s*\(\s*ii\s*\)?', 'INS 503ii ', t, flags=re.IGNORECASE)
    t = re.sub(r'(?:INS\s*)?331\s*\(\s*iii\s*\)?', 'INS 331iii ', t, flags=re.IGNORECASE)
    t = re.sub(r'INS[\.\s]*([0-9]{3,4})\s*\(\s*([a-z0-9ivx]+)\s*\)?', r'INS \1\2 ', t, flags=re.IGNORECASE)
    t = re.sub(r'INS[\.\s]*([0-9]{3,4})\s*([ivx]{1,4})\b', r'INS \1\2', t, flags=re.IGNORECASE)
    t = re.sub(r'INS[\.\s]*([0-9]{3,4})(?![0-9ivx])', r'INS \1 ', t, flags=re.IGNORECASE)
    t = re.sub(r'\bINS\s+INS\b', 'INS', t, flags=re.IGNORECASE)

    # Mã số phụ gia riêng lẻ trong ngoặc đơn
    t = re.sub(r'\(([0-9]{3,4}[a-z]*)\)', r'(INS \1)', t)
    t = re.sub(r'\b150d\b', 'INS 150d', t)
    t = re.sub(r'\b1420\b', 'INS 1420', t)
    return t


# ── Hàm tước dấu thanh tiếng Việt (dùng cho dictionary matching) ──
import unicodedata

def _strip_viet(text):
    """Tước bỏ toàn bộ dấu thanh và dấu phụ tiếng Việt → dạng Latin cơ sở."""
    text = text.replace('đ', 'd').replace('Đ', 'D')
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


# =====================================================================
# TỪ ĐIỂN CHÍNH TẢ THỰC PHẨM (OCR Dictionary-based Diacritic Restoration)
# Kỹ thuật: tước dấu thanh (diacritic-stripping) rồi khớp với từ điển
# → tự động sửa mọi biến thể OCR mà không cần liệt kê từng lỗi cụ thể.
# =====================================================================

_FOOD_COMPOUND_LIST = [
    "Có thể chứa vết của", "chất điều chỉnh độ acid",
    "chất tạo ngọt tổng hợp", "hương vani tổng hợp",
    "hương cola tự nhiên", "phẩm màu tổng hợp",
    "Nước bão hòa CO2", "màu caramen nhóm IV",
    "hạt đậu phộng", "Cảnh báo dị ứng",
    "bột sữa béo", "chất nhũ hóa", "chất tạo xốp",
    "chất bảo quản", "chất điều vị", "chất làm dày",
    "trứng gà tươi", "Chứa Phenylalanine",
    "Gói gia vị", "Gói rau sấy",
    "Thành phần", "muối i-ốt", "đường kính", "đậu nành",
    "đậu phộng", "hạnh nhân", "Chứa gluten", "tổng hợp",
    "dầu cọ", "bơ lạt", "Bột mì", "Vắt mì", "Gói nêm",
    "tinh bột", "nước mắm", "xì dầu", "nước tương",
    "bột ngọt", "dầu ăn", "bột nở", "sữa bột",
    "bột năng", "bột gạo", "dầu mè", "mỡ heo",
    "thịt bò", "thịt heo", "thịt gà",
]

_COMPOUND_MAP = {}
for _term in _FOOD_COMPOUND_LIST:
    _key = _strip_viet(_term).lower()
    _COMPOUND_MAP[_key] = _term
_COMPOUND_KEYS = sorted(_COMPOUND_MAP.keys(), key=lambda k: -len(k.split()))

_FOOD_WORD_LIST = [
    "bột", "mì", "bơ", "đường", "trứng", "sữa", "muối", "chất", "dầu",
    "béo", "tươi", "kính", "hương", "tổng", "hợp", "nhũ", "hóa",
    "tạo", "xốp", "quản", "điều", "vị", "phẩm", "màu",
    "ngọt", "dày", "chứa",
    "hạt", "đậu", "phộng", "hạnh", "nhân", "ớt",
    "nếp", "mỡ", "mật", "ong", "tiêu", "hành", "tỏi", "gừng",
    "nghệ", "quế", "giấm", "mắm", "nước", "thịt", "gạo", "mè",
    "vừng", "cọ", "lạt", "nở", "năng",
    "bò", "heo", "gà", "cá", "tôm", "mực", "cua",
    "đỗ", "lạc", "khoai", "sắn", "bắp",
    "chanh", "cam", "ổi", "xoài", "dừa", "bưởi",
    "cải", "rau", "hẹ", "bí", "cà",
    "mạch", "nha", "men", "tương", "xì",
]

_WORD_MAP = {}
for _w in _FOOD_WORD_LIST:
    _k = _strip_viet(_w).lower()
    if _k not in _WORD_MAP:
        _WORD_MAP[_k] = _w

_PUNCT_RE = re.compile(r'^([(\["]*)(.+?)([,.:;!?)\]"]*)$')


def _correct_food_spelling(text):
    """
    Sửa chính tả OCR bằng kỹ thuật Diacritic-Stripping + Food Domain Lexicon.
    Tước dấu thanh → khớp với từ điển → khôi phục dạng đúng.
    Phase 1: Cụm từ (longest match first) → Phase 2: Đơn từ.
    """
    if not text:
        return text

    words = text.split()
    cores = []
    puncts = []
    stripped = []
    for w in words:
        m = _PUNCT_RE.match(w)
        if m:
            pre, core, suf = m.group(1), m.group(2), m.group(3)
        else:
            pre, core, suf = '', w, ''
        cores.append(core)
        puncts.append((pre, suf))
        stripped.append(_strip_viet(core).lower())

    used = [False] * len(words)
    result = list(words)

    for key in _COMPOUND_KEYS:
        key_parts = key.split()
        n = len(key_parts)
        for i in range(len(stripped) - n + 1):
            if any(used[j] for j in range(i, i + n)):
                continue
            if stripped[i:i + n] == key_parts:
                rep_words = _COMPOUND_MAP[key].split()
                for j in range(n):
                    idx = i + j
                    used[idx] = True
                    pre, suf = puncts[idx]
                    if j < len(rep_words):
                        rw = rep_words[j]
                        if cores[idx] and cores[idx][0].isupper() and rw[0].islower():
                            rw = rw[0].upper() + rw[1:]
                        result[idx] = pre + rw + suf
                    else:
                        result[idx] = pre + suf if (pre or suf) else ''

    for i in range(len(words)):
        if used[i]:
            continue
        key = stripped[i]
        if key and key in _WORD_MAP:
            candidate = _WORD_MAP[key]
            if cores[i].lower() != candidate:
                pre, suf = puncts[i]
                rw = candidate
                if cores[i][0].isupper():
                    rw = rw[0].upper() + rw[1:]
                result[i] = pre + rw + suf

    return ' '.join(w for w in result if w)


# =====================================================================
# LLM SPELL CORRECT (Gemini Flash) — chỉ sửa từ mà Dictionary không cover
# =====================================================================

_GEMINI_SPELL_PROMPT = """Bạn là chuyên gia phục hồi văn bản OCR nhãn thực phẩm Việt Nam.
Đoạn text dưới đây đã được sửa cơ bản bằng từ điển chuyên ngành thực phẩm (khôi phục dấu thanh cho các từ phổ biến).
Tuy nhiên vẫn CÒN SÓT một số lỗi mà từ điển không bao phủ được.
Nhiệm vụ: chỉ sửa các lỗi OCR CÒN LẠI — bao gồm:
- Ký tự bị OCR nhầm mà từ điển chưa cover (VD: ö→ố, ü→ủ, ii→ối, phona→phộng, muii-öt→muối i-ốt)
- Từ lạ, tên thương hiệu, tên hóa chất chuyên ngành bị sai dấu
- Từ bị dính nhau mà bước tách từ chưa xử lý được
- Cụm từ cảnh báo dị ứng bị OCR đọc sai
Quy tắc:
- Giữ nguyên mã INS (VD: INS 621, INS 150d), dấu ngoặc, số liệu, cấu trúc câu
- Giữ nguyên các từ đã đúng — KHÔNG sửa lại những gì từ điển đã sửa đúng
- Không thêm, không bớt, không giải thích — chỉ trả về text đã sửa
- Nếu không chắc chắn, giữ nguyên từ gốc

Text:
"""


def _gemini_spell_correct(raw_text):
    """
    Sửa chính tả OCR bằng Gemini 2.0 Flash (miễn phí 15 RPM).
    Chỉ sửa các lỗi mà Dictionary-based correction chưa bao phủ được.
    Trả về text đã sửa, hoặc None nếu thất bại / không có API key.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=_GEMINI_SPELL_PROMPT + raw_text,
            config={"temperature": 0.0, "max_output_tokens": max(1024, len(raw_text) * 6)},
        )
        result = response.text.strip()
        if result and len(result) >= len(raw_text) * 0.5:
            return result
    except Exception:
        pass
    return None


# =====================================================================
# PIPELINE TỔNG HỢP: OCR → Dictionary → LLM
# =====================================================================

def post_process_ocr_spelling(raw_text, use_gemini=True):
    """
    Sửa lỗi chính tả OCR cho nhãn thực phẩm tiếng Việt — Pipeline 3 giai đoạn:
      1. Structural cleanup — tách từ dính, chuẩn hóa INS, dọn ký tự rác
      2. Dictionary correction — khôi phục dấu thanh bằng từ điển thực phẩm (diacritic-stripping)
      3. LLM polish tùy chọn — Gemini Flash sửa nốt các lỗi mà từ điển không cover

    Returns:
        dict: "final", "after_ocr_dict", "after_llm", "llm_used"
    """
    if not raw_text:
        return {"final": "", "after_ocr_dict": "", "after_llm": None, "llm_used": False}

    t = str(raw_text)

    # ── Giai đoạn 1: Structural cleanup ──
    t = _fix_word_boundaries(t)
    t = _normalize_ins_codes(t)
    t = re.sub(r'(?<=\s)[A-Z](?=\s)', '', t)
    t = re.sub(r'\)\)', ')', t)
    t = re.sub(r'\(\(', '(', t)
    t = re.sub(r'(\w)/(\w)', r'\1 / \2', t)

    # ── Giai đoạn 2: Dictionary-based OCR correction ──
    t = _correct_food_spelling(t)
    t = re.sub(r'\b(\w[\w\s]{2,20}?)\s+[/\\]\s+\1\b', r'\1', t)
    t = re.sub(r'\b(\w[\w\s]{2,20}?)\s+\1\b', r'\1', t)
    t = re.sub(r'\s+', ' ', t).strip()

    after_ocr_dict = t

    # ── Giai đoạn 3: LLM polish (chỉ sửa phần dictionary chưa cover) ──
    gemini_result = _gemini_spell_correct(t) if use_gemini else None
    if gemini_result:
        t = gemini_result
        t = _normalize_ins_codes(t)
        t = re.sub(r'\s+', ' ', t).strip()

    return {
        "final": t,
        "after_ocr_dict": after_ocr_dict,
        "after_llm": gemini_result,
        "llm_used": gemini_result is not None,
    }


def parse_ingredients_from_ocr(full_text):
    """
    Thuật toán Bóc tách Thành phần bằng Lookahead Regex:
    - Tìm vị trí xuất hiện của cụm từ 'Thành phần:' hoặc 'Ingredients:'.
    - Cắt tỉa các phần thông tin nhà sản xuất, hạn sử dụng.
    - Tách theo dấu phẩy, chấm phẩy hoặc dấu chấm theo sau bởi khoảng trắng nhưng KHÔNG tách bên trong ngoặc đơn.
      Ví dụ: 'Chất nhũ hóa (INS 322i lecithin, INS 471), muối' -> ['Chất nhũ hóa (INS 322i lecithin, INS 471)', 'muối'].
    - Phân loại danh mục sơ bộ (Category classification).
    """
    folded_chars = []
    source_offsets = []
    for source_index, char in enumerate(full_text):
        normalized = "d" if char.casefold() == "đ" else char.casefold()
        decomposed = unicodedata.normalize("NFKD", normalized)
        for folded_char in decomposed:
            if not unicodedata.combining(folded_char):
                folded_chars.append(folded_char)
                source_offsets.append(source_index)
    folded_text = "".join(folded_chars)

    start_pattern = r"(?:thanh\s*phan|ingredients?)\s*[:\-]?\s*"
    start_matches = list(re.finditer(start_pattern, folded_text, re.IGNORECASE))
    start_match = start_matches[0] if start_matches else None
    if start_match:
        for candidate in start_matches[1:]:
            between_headings = folded_text[start_match.end():candidate.start()]
            if re.fullmatch(r"[\s/|,;:\-]*", between_headings):
                start_match = candidate
            else:
                break
    if start_match:
        start_offset = source_offsets[start_match.end() - 1] + 1 if start_match.end() else 0
        target_end = len(full_text)
        stop_patterns = [
            r"canh\s*bao\s*di\s*ung",
            r"allergen\s*advice",
            r"allergen\s*information",
            r"huong\s*dan\s*bao\s*quan",
            r"cach\s*bao\s*quan",
            r"dieu\s*kien\s*bao\s*quan",
            r"\bbao\s*quan\s*noi\b",
            r"han\s*su\s*dung",
            r"ngay\s*san\s*xuat",
            r"\bhsd\b",
            r"\bnsx\b",
            r"xuat\s*xu",
            r"noi\s*san\s*xuat",
            r"khoi\s*luong\s*tinh",
            r"the\s*tich\s*thuc",
            r"chiu\s*trach\s*nhiem",
            r"nutrition\s*facts",
            r"gia\s*tri\s*dinh\s*duong",
            r"gia\s*tri\s*dinh\s*d\w*",
            r"g(?:ia|i)?\s*tri\s*dinh\s*\w{2,8}",
            r"thong\s*tin\s*dinh\s*duong",
            r"thong\s*tin\s*dinh\s*d\w*",
            r"\bingredients?\b",
            r"net\s*weight",
            r"trong\s*luong\s*tinh",
        ]
        for stop_pattern in stop_patterns:
            stop_match = re.search(
                stop_pattern,
                folded_text[start_match.end():],
                re.IGNORECASE,
            )
            if stop_match:
                folded_stop_index = start_match.end() + stop_match.start()
                target_end = min(target_end, source_offsets[folded_stop_index])
        target_text = full_text[start_offset:target_end]
    else:
        target_text = ""

    # Tách dấu phẩy/chấm phẩy CHỈ KHI NGOÀI ngoặc đơn (bảo toàn cụm trong ngoặc)
    raw_splits = []
    depth = 0
    current = []
    for i, ch in enumerate(target_text):
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch in (',', ';') and depth == 0:
            raw_splits.append(''.join(current))
            current = []
        elif ch == '.' and depth == 0 and i + 2 < len(target_text) and target_text[i + 1] == ' ' and target_text[i + 2].isupper():
            raw_splits.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        raw_splits.append(''.join(current))

    # Hậu xử lý: gộp mục chỉ chứa mã INS rời vào mục liền trước (OCR tách nhầm ngoặc)
    merged_splits = []
    for item in raw_splits:
        stripped = item.strip()
        if merged_splits and re.match(r'^INS\s*\d{3}', stripped, re.IGNORECASE):
            merged_splits[-1] = merged_splits[-1].rstrip(') ') + ', ' + stripped.rstrip(')') + ')'
        else:
            merged_splits.append(item)

    pending_items = []
    for item in merged_splits:
        item = re.sub(r'^(?:vắt mì|gói gia vị|gói nêm|gói rau|thành phần)\s*[:\-]\s*', '', item, flags=re.IGNORECASE)
        item = re.sub(r'\s+', ' ', item).strip(" .-:,\n\r\t")

        # Keep distinct additive codes separate so entity linking cannot collapse
        # several INS substances from one parenthesized label into a single match.
        code_parts = re.split(
            r",\s*(?=(?:INS\s*\d{3,4}|E\s*\d{3,4}))",
            item,
            flags=re.IGNORECASE,
        )
        if len(code_parts) > 1:
            pending_items.extend(code_parts)
        else:
            pending_items.append(item)

    mentions = []
    for item in pending_items:
        item = re.sub(r'\s+', ' ', item).strip(" .-:,\n\r\t")
        _SHORT_VALID = {"bơ", "bò", "mì", "mè", "gà", "cá", "mỡ", "dầu", "bơ", "sữa", "mật"}
        if (len(item) >= 3 or item.lower() in _SHORT_VALID) and not re.match(r'^\d+(\.\d+)?%$', item):
            category = "ingredient"
            lower_item = item.lower()
            if any(k in lower_item for k in ["điều vị", "bột ngọt", "621", "631", "627", "glutamate"]):
                category = "flavor_enhancer"
            elif any(k in lower_item for k in ["bảo quản", "211", "benzoat", "sorbat", "202"]):
                category = "preservative"
            elif any(k in lower_item for k in ["màu", "color", "102", "tartrazine", "110", "129", "150d"]):
                category = "color"
            elif any(k in lower_item for k in ["ngọt", "tạo ngọt", "sweetener", "951", "aspartame", "950"]):
                category = "sweetener"
            elif any(k in lower_item for k in ["acid", "axit", "330", "citric"]):
                category = "acidity_regulator"
            elif any(k in lower_item for k in ["nhũ hóa", "emulsifier", "322", "lecithin", "471"]):
                category = "emulsifier"
            elif any(k in lower_item for k in ["tạo xốp", "500", "503", "baking"]):
                category = "raising_agent"
            elif any(k in lower_item for k in ["làm dày", "1420", "tinh bột"]):
                category = "thickener"
            elif any(k in lower_item for k in ["dị ứng", "allergen", "gluten", "đậu phộng", "đậu nành", "trứng", "sữa"]):
                category = "allergen"

            mentions.append({
                "raw_mention": item,
                "category": category
            })

    return mentions


# =====================================================================
# 3. TRÍCH XUẤT KẾT QUẢ PADDLEOCR & VẼ BOUNDING BOX
# =====================================================================

def parse_paddle_results(results):
    """Phân giải đa hình kết quả trả về từ cả PaddleOCR v2 và PaddleOCR v3 (PaddleX)."""
    lines_data = []
    full_texts = []
    total_conf = 0.0
    boxes = []

    if not results:
        return lines_data, full_texts, 0.70, boxes

    # Định dạng v2 chuẩn: list of [ [ [box], (text, conf) ] ]
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
        for line in results[0]:
            if len(line) >= 2:
                box = line[0]
                text_info = line[1]
                if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                    text, conf = text_info
                    text = str(text).strip()
                    if text:
                        lines_data.append({"text": text, "confidence": round(float(conf), 2)})
                        full_texts.append(text)
                        total_conf += float(conf)
                        boxes.append(box)

    # Định dạng v3 (PaddleX inference pipeline)
    elif isinstance(results, list):
        for res in results:
            texts = []
            scores = []
            polys = []
            if isinstance(res, dict):
                texts = res.get('rec_texts', res.get('rec_text', []))
                scores = res.get('rec_scores', res.get('rec_score', []))
                polys = res.get('dt_polys', [])
            elif hasattr(res, '__dict__'):
                d = res.__dict__
                texts = d.get('rec_texts', d.get('rec_text', []))
                scores = d.get('rec_scores', d.get('rec_score', []))
                polys = d.get('dt_polys', [])

            if isinstance(texts, str):
                texts = [texts]
            if isinstance(scores, (int, float)):
                scores = [scores]

            for i, (t, s) in enumerate(zip(texts, scores)):
                t = str(t).strip()
                if t:
                    lines_data.append({"text": t, "confidence": round(float(s), 2)})
                    full_texts.append(t)
                    total_conf += float(s)
                    if i < len(polys):
                        boxes.append(polys[i])

    avg_conf = total_conf / len(lines_data) if lines_data else 0.70
    return lines_data, full_texts, avg_conf, boxes


def draw_ocr_boxes_from_boxes(img_bgr, boxes):
    """Vẽ bounding box màu xanh lá bán trong suốt bao quanh từng cụm từ trên ảnh."""
    annotated = img_bgr.copy()
    overlay = img_bgr.copy()

    for box in boxes:
        try:
            poly = np.array(box).astype(np.int32)
            cv2.fillPoly(overlay, [poly], color=(0, 255, 100))
            cv2.polylines(annotated, [poly], isClosed=True, color=(0, 200, 0), thickness=2)
        except Exception:
            continue

    alpha = 0.2
    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb)


# =====================================================================
# 4. CHẾ ĐỘ NHÃN CHUẨN (GROUND TRUTH BENCHMARK) & THỰC THI CHÍNH
# =====================================================================

_LABEL_STUDIO_CACHE = None


def load_label_studio_data(assets_dir=None):
    """
    Nạp và parse toàn bộ dữ liệu Label Studio annotation (550 ảnh nhãn bao bì).
    Trả về dict {image_filename: parsed_data} đã cache.
    """
    global _LABEL_STUDIO_CACHE
    if _LABEL_STUDIO_CACHE is not None:
        return _LABEL_STUDIO_CACHE

    if assets_dir is None:
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    json_path = os.path.join(assets_dir, "label_studio", "label_studio_exported_data.json")

    if not os.path.exists(json_path):
        _LABEL_STUDIO_CACHE = {}
        return _LABEL_STUDIO_CACHE

    import json
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    parsed = {}
    for item in raw_data:
        image_ref = item.get("data", {}).get("image", "")
        ann = item["annotations"][0] if item.get("annotations") else None
        if not ann or not image_ref:
            continue

        results = ann.get("result", [])
        id_to_text = {}
        id_to_label = {}
        id_to_bbox = {}

        for r in results:
            rid = r.get("id", "")
            val = r.get("value", {})
            if r.get("type") == "textarea":
                id_to_text[rid] = " ".join(val.get("text", []))
            elif r.get("type") == "rectanglelabels":
                labels = val.get("rectanglelabels", [])
                if labels:
                    id_to_label[rid] = labels[0]
                id_to_bbox[rid] = (
                    val.get("x", 0), val.get("y", 0),
                    val.get("width", 0), val.get("height", 0),
                )

        sorted_ids = sorted(
            id_to_bbox.keys(),
            key=lambda k: (round(id_to_bbox[k][1] / 3), id_to_bbox[k][0]),
        )

        product_parts = []
        full_text_parts = []
        ingredient_parts = []
        additive_parts = []
        warning_parts = []
        lines_data = []

        current_type = None
        current_parts = []

        for rid in sorted_ids:
            text = id_to_text.get(rid, "")
            label = id_to_label.get(rid, "O")
            full_text_parts.append(text)

            if label.startswith("B-"):
                if current_type and current_parts:
                    joined = " ".join(current_parts)
                    if current_type == "PRODUCT_NAME":
                        product_parts.append(joined)
                    elif current_type == "INGREDIENT":
                        ingredient_parts.append(joined)
                    elif current_type == "ADDITIVE":
                        additive_parts.append(joined)
                    elif current_type == "WARNING":
                        warning_parts.append(joined)
                current_type = label[2:]
                current_parts = [text]
            elif label.startswith("I-") and current_type == label[2:]:
                current_parts.append(text)
            else:
                if current_type and current_parts:
                    joined = " ".join(current_parts)
                    if current_type == "PRODUCT_NAME":
                        product_parts.append(joined)
                    elif current_type == "INGREDIENT":
                        ingredient_parts.append(joined)
                    elif current_type == "ADDITIVE":
                        additive_parts.append(joined)
                    elif current_type == "WARNING":
                        warning_parts.append(joined)
                current_type = None
                current_parts = []

        if current_type and current_parts:
            joined = " ".join(current_parts)
            if current_type == "PRODUCT_NAME":
                product_parts.append(joined)
            elif current_type == "INGREDIENT":
                ingredient_parts.append(joined)
            elif current_type == "ADDITIVE":
                additive_parts.append(joined)
            elif current_type == "WARNING":
                warning_parts.append(joined)

        product_name = " ".join(product_parts).strip()
        if not product_name:
            preview = ", ".join((ingredient_parts + additive_parts)[:3])
            product_name = preview[:50] if preview else "Sản phẩm"
        all_components = ingredient_parts + additive_parts
        full_text = "Thành phần: " + ", ".join(all_components) + "."
        if warning_parts:
            full_text += " Cảnh báo: " + " ".join(warning_parts)

        chunk_size = 80
        raw_full = " ".join(full_text_parts)
        words = raw_full.split()
        for i in range(0, len(words), 12):
            line_text = " ".join(words[i:i + 12])
            lines_data.append({"text": line_text, "confidence": 0.99})

        image_filename = os.path.basename(image_ref)
        parsed[image_filename] = {
            "product_name": product_name,
            "full_text": full_text,
            "n_ingredients": len(ingredient_parts),
            "n_additives": len(additive_parts),
            "warning_text": " ".join(warning_parts),
            "lines": lines_data,
            "image_ref": image_ref,
        }

    _LABEL_STUDIO_CACHE = parsed
    return _LABEL_STUDIO_CACHE


def get_label_studio_product_list(assets_dir=None):
    """Trả về danh sách sản phẩm từ Label Studio để hiển thị trong dropdown."""
    ls_data = load_label_studio_data(assets_dir)
    products = []
    for filename, info in sorted(ls_data.items()):
        label = f"{info['product_name']} ({info['n_ingredients']}I, {info['n_additives']}A)"
        products.append({
            "filename": filename,
            "label": label,
            "product_name": info["product_name"],
            "image_ref": info["image_ref"],
        })
    return products


def get_ground_truth_sample(sample_key, image_input):
    """
    Trả về dữ liệu nhãn chuẩn (Ground Truth Benchmark) đã được kiểm chứng.
    Hỗ trợ: (1) 3 mẫu demo gốc, (2) 550 ảnh Label Studio annotation.
    """
    img_bgr = convert_to_cv2_image(image_input)
    key_lower = sample_key.lower()

    # Thử tìm trong Label Studio data trước
    ls_data = load_label_studio_data()
    ls_match = None
    if sample_key in ls_data:
        ls_match = ls_data[sample_key]
    else:
        for filename, info in ls_data.items():
            if sample_key in filename or filename in sample_key:
                ls_match = info
                break

    if ls_match:
        full_text = ls_match["full_text"]
        lines = ls_match["lines"]
        avg_conf = 0.99
        mentions = parse_ingredients_from_ocr(full_text)
        annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        annotated_img = Image.fromarray(annotated_rgb)

        return {
            "is_real_ocr": False,
            "is_ground_truth": True,
            "engine_name": f"Ground Truth — Label Studio Annotation ({ls_match['product_name']})",
            "full_text": full_text,
            "raw_text_before_norm": full_text,
            "text_after_ocr_dict": full_text,
            "llm_spell_used": False,
            "avg_confidence": avg_conf,
            "lines": lines,
            "mentions": mentions,
            "annotated_image": annotated_img,
        }

    # Không tìm thấy ground truth → trả về rỗng để pipeline chuyển sang OCR thật
    annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    annotated_img = Image.fromarray(annotated_rgb)
    return {
        "is_real_ocr": False,
        "is_ground_truth": False,
        "is_ocr_unavailable": True,
        "engine_name": "Không có Ground Truth cho ảnh này",
        "full_text": "",
        "raw_text_before_norm": "",
        "text_after_ocr_dict": "",
        "llm_spell_used": False,
        "avg_confidence": 0.0,
        "lines": [],
        "mentions": [],
        "annotated_image": annotated_img,
    }


def _run_single_ocr_pass(engine, img_bgr):
    """Chạy một lần OCR trên ảnh đã tiền xử lý, trả về (lines_data, full_texts, avg_conf, boxes)."""
    try:
        try:
            results = engine.ocr(img_bgr)
        except TypeError:
            results = engine.ocr(img_bgr, cls=True)
        return parse_paddle_results(results)
    except Exception:
        return [], [], 0.0, []


def _multi_pass_ocr(engine, img_bgr):
    """
    Multi-pass OCR: chạy OCR với nhiều biến thể tiền xử lý khác nhau,
    chọn kết quả có confidence cao nhất và nhiều text nhất.
    """
    preprocess_funcs = [
        ("adaptive", adaptive_preprocess_image),
        ("binary", preprocess_variant_binary),
        ("grayscale_sharp", preprocess_variant_grayscale_sharp),
    ]

    best_result = None
    best_score = -1.0

    for name, preprocess_fn in preprocess_funcs:
        try:
            processed = preprocess_fn(img_bgr)
            lines_data, full_texts, avg_conf, boxes = _run_single_ocr_pass(engine, processed)
            if not full_texts:
                continue
            total_chars = sum(len(t) for t in full_texts)
            score = avg_conf * 0.4 + min(total_chars / 500.0, 1.0) * 0.3 + min(len(full_texts) / 10.0, 1.0) * 0.3
            if score > best_score:
                best_score = score
                best_result = (lines_data, full_texts, avg_conf, boxes, name, processed)
        except Exception:
            continue

    return best_result


def extract_with_paddle_ocr(
    image_input, sample_key="custom", use_gemini_correction=True, **kwargs
):
    """
    Thực hiện quét ảnh thực tế bằng PaddleOCR kết hợp chuỗi thuật toán:
    Multi-pass tiền xử lý ➔ Nhận diện DBNet/SVTR ➔ Phục hồi chính tả tiếng Việt.
    """
    img_bgr = convert_to_cv2_image(image_input)
    engine = get_ocr_engine()
    ocr_error = None

    if engine is not None:
        try:
            result = _multi_pass_ocr(engine, img_bgr)

            if result is not None:
                lines_data, full_texts, avg_conf, boxes, pass_name, processed_img = result
                raw_full_text = " ".join(full_texts)
                spell_result = post_process_ocr_spelling(
                    raw_full_text,
                    use_gemini=use_gemini_correction,
                )
                normalized_full_text = spell_result["final"]
                mentions = parse_ingredients_from_ocr(normalized_full_text)
                annotated_img = draw_ocr_boxes_from_boxes(processed_img, boxes)

                return {
                    "is_real_ocr": True,
                    "is_ground_truth": False,
                    "engine_name": f"PaddleOCR Multi-pass ({pass_name}) + Phục hồi Chính tả Thực phẩm",
                    "full_text": normalized_full_text,
                    "raw_text_before_norm": raw_full_text,
                    "text_after_ocr_dict": spell_result["after_ocr_dict"],
                    "llm_spell_used": spell_result["llm_used"],
                    "avg_confidence": avg_conf,
                    "lines": lines_data,
                    "mentions": mentions,
                    "annotated_image": annotated_img
                }
            else:
                ocr_error = "PaddleOCR không phát hiện được văn bản trong ảnh (đã thử 3 biến thể tiền xử lý)."
        except Exception as e:
            ocr_error = f"Lỗi runtime PaddleOCR: {e}"
            print(ocr_error)
    else:
        ocr_error = _INIT_ERROR or "Chưa cài đặt hoặc chưa nạp được PaddleOCR."

    print(f"PaddleOCR unavailable for image OCR: {ocr_error}")
    img_bgr = convert_to_cv2_image(image_input)
    annotated_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return {
        "is_real_ocr": False,
        "is_ground_truth": False,
        "is_ocr_unavailable": True,
        "engine_name": "OCR không khả dụng",
        "ocr_error": ocr_error,
        "full_text": "",
        "raw_text_before_norm": "",
        "text_after_ocr_dict": "",
        "llm_spell_used": False,
        "avg_confidence": 0.0,
        "lines": [],
        "mentions": [],
        "annotated_image": annotated_img,
    }


def run_perception(image_input, sample_key="custom", force_real_ocr=False, **kwargs):
    """
    Entry-point thống nhất cho Tầng Perception của hệ thống FoodHealthQA.
    Chấp nhận các tham số mở rộng (**kwargs) để đảm bảo không bao giờ bị lỗi TypeError.
    """
    if sample_key != "custom" and not force_real_ocr:
        return get_ground_truth_sample(sample_key, image_input)
    else:
        return extract_with_paddle_ocr(
            image_input,
            sample_key=sample_key,
            **kwargs,
        )
