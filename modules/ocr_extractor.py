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
   - Trích xuất thành phần với Lookahead Regex (bảo toàn cấu trúc ngoặc đơn)
4. Hỗ trợ song song Chế độ Nhãn Chuẩn (Ground Truth Benchmark) và Chế độ Quét Ảnh Thực Tế.
"""

import hashlib
import os
import shutil
import tempfile
import urllib.request
# Vô hiệu hóa triệt để OneDNN / MKLDNN và PIR trên Windows CPU để tránh lỗi OneDnnContext
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Không gọi PyPI để kiểm tra phiên bản mỗi lần Albumentations được import.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import importlib.util
import re
import threading
import unicodedata
import cv2
import numpy as np
from PIL import Image

# Thư viện OCR (paddle, PaddleOCR, VietOCR + torch) rất nặng (~12 giây import). Chỉ nạp khi thật sự cần OCR,
# để mở ứng dụng và các phần không dùng OCR (parse thành phần, script nghiên cứu, test) không phải chờ.
_PADDLE_AVAILABLE = False
_OCR_ENGINE = None
_INIT_ERROR = None
PaddleOCR = None
_OCR_LOCK = threading.RLock()   # luồng khởi động nền và luồng người dùng không khởi tạo trùng


def _load_paddleocr():
    """Import paddle + PaddleOCR (một lần). Trả về True nếu dùng được."""
    global _PADDLE_AVAILABLE, _INIT_ERROR, PaddleOCR
    with _OCR_LOCK:
        if _PADDLE_AVAILABLE:
            return True
        try:
            import paddle
            paddle.set_flags({'FLAGS_use_mkldnn': False})
        except Exception:
            pass
        try:
            from paddleocr import PaddleOCR as _PaddleOCR
            PaddleOCR = _PaddleOCR
            _PADDLE_AVAILABLE = True
            _INIT_ERROR = None
        except Exception as e:
            _PADDLE_AVAILABLE = False
            _INIT_ERROR = str(e)
        return _PADDLE_AVAILABLE

# ── VietOCR (Text Recognition tầng 2) ────────────────────────────────
_VIETOCR_AVAILABLE = False
_VIETOCR_PREDICTOR = None
_VIETOCR_INIT_ERROR = None
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VIETOCR_ASSET_DIR = os.path.join(_PROJECT_ROOT, "models", "vietocr")
_VIETOCR_CONFIG_PATH = os.path.join(_VIETOCR_ASSET_DIR, "vgg_transformer.json")
_VIETOCR_DEFAULT_WEIGHTS_PATH = os.path.join(_VIETOCR_ASSET_DIR, "vgg_transformer.pth")
_VIETOCR_WEIGHTS_URL = "https://vocr.vn/data/vietocr/vgg_transformer.pth"
_VIETOCR_WEIGHTS_SIZE = 151815373
_VIETOCR_WEIGHTS_SHA256 = "380512193a8b6cbf6fad80deacdc9b6939d10d473d199892fc6408d13775ea59"

_VietOCRPredictor = None
_VietOCRCfg = None


def is_paddleocr_available():
    """PaddleOCR đã cài chưa — chỉ kiểm tra gói có trên máy, KHÔNG import (import thật diễn ra khi khởi tạo OCR)."""
    if _PADDLE_AVAILABLE:
        return True
    try:
        return importlib.util.find_spec("paddleocr") is not None
    except (ImportError, ValueError):
        return False


def is_vietocr_available():
    """Kiểm tra VietOCR đã cài và nạp được không (import thật, chỉ một lần)."""
    global _VIETOCR_AVAILABLE, _VIETOCR_INIT_ERROR, _VietOCRPredictor, _VietOCRCfg
    with _OCR_LOCK:
        if not _VIETOCR_AVAILABLE:
            try:
                from vietocr.tool.predictor import Predictor as _VietOCRPredictor
                from vietocr.tool.config import Cfg as _VietOCRCfg
                _VIETOCR_AVAILABLE = True
                _VIETOCR_INIT_ERROR = None
            except Exception as _e:
                _VIETOCR_INIT_ERROR = str(_e)
        return _VIETOCR_AVAILABLE


def warm_up_ocr():
    """Nạp thư viện và khởi tạo sẵn PaddleOCR + VietOCR (gọi từ luồng nền khi ứng dụng khởi động)."""
    engine = get_ocr_engine()
    predictor = get_vietocr_predictor()
    return {"paddleocr": engine is not None, "vietocr": predictor is not None}


def _vietocr_weights_path():
    """Cho phép đổi vị trí model bằng biến VIETOCR_WEIGHTS_PATH khi triển khai."""
    configured = os.environ.get("VIETOCR_WEIGHTS_PATH", "").strip()
    return os.path.abspath(configured) if configured else _VIETOCR_DEFAULT_WEIGHTS_PATH


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_vietocr_weights(path):
    try:
        return (
            os.path.isfile(path)
            and os.path.getsize(path) == _VIETOCR_WEIGHTS_SIZE
            and _file_sha256(path) == _VIETOCR_WEIGHTS_SHA256
        )
    except OSError:
        return False


def _ensure_vietocr_weights():
    """
    Đảm bảo trọng số nằm ở thư mục bền vững của dự án.

    Ưu tiên model local, sau đó tái sử dụng cache Temp của VietOCR cũ; chỉ tải
    từ vocr.vn một lần khi cả hai nơi đều chưa có model hợp lệ. File được kiểm
    tra kích thước và SHA-256 trước khi đưa vào sử dụng.
    """
    target_path = _vietocr_weights_path()
    if _valid_vietocr_weights(target_path):
        return target_path

    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    part_path = f"{target_path}.part-{os.getpid()}"
    legacy_cache = os.path.join(tempfile.gettempdir(), "vgg_transformer.pth")

    try:
        if _valid_vietocr_weights(legacy_cache):
            shutil.copy2(legacy_cache, part_path)
        else:
            request = urllib.request.Request(
                _VIETOCR_WEIGHTS_URL,
                headers={"User-Agent": "FoodHealthQA-VietOCR/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with open(part_path, "wb") as target:
                    shutil.copyfileobj(response, target, length=1024 * 1024)

        if not _valid_vietocr_weights(part_path):
            raise RuntimeError("VietOCR weights failed size/SHA-256 validation")
        os.replace(part_path, target_path)
        return target_path
    finally:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass


def get_vietocr_predictor():
    """Khởi tạo VietOCR từ config/model local; chỉ tải trọng số ở lần đầu."""
    global _VIETOCR_PREDICTOR, _VIETOCR_INIT_ERROR
    if not is_vietocr_available():
        return None
    with _OCR_LOCK:
        return _init_vietocr_predictor()


def _init_vietocr_predictor():
    global _VIETOCR_PREDICTOR, _VIETOCR_INIT_ERROR
    if _VIETOCR_PREDICTOR is None:
        try:
            if not os.path.isfile(_VIETOCR_CONFIG_PATH):
                raise FileNotFoundError(f"Missing local VietOCR config: {_VIETOCR_CONFIG_PATH}")
            weights_path = _ensure_vietocr_weights()
            config = _VietOCRCfg.load_config_from_file(_VIETOCR_CONFIG_PATH)
            config['weights'] = weights_path
            config['pretrain'] = ''
            config['cnn']['pretrained'] = False
            config['device'] = 'cpu'
            config['predictor']['beamsearch'] = False  # faster inference
            _VIETOCR_PREDICTOR = _VietOCRPredictor(config)
            _VIETOCR_INIT_ERROR = None
        except Exception as e:
            _VIETOCR_INIT_ERROR = str(e)
            print(f"[VietOCR] Predictor initialization failed: {e}")
    return _VIETOCR_PREDICTOR


def get_ocr_engine():
    """
    Khởi tạo PaddleOCR Engine tương thích linh hoạt mọi phiên bản PaddleOCR (v2, v3, PaddleX).
    Ưu tiên cấu hình tối ưu cho nhãn thực phẩm tiếng Việt.
    """
    if not _load_paddleocr():
        return None
    with _OCR_LOCK:
        return _init_ocr_engine()


def _init_ocr_engine():
    global _OCR_ENGINE, _INIT_ERROR
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
# PIPELINE TỔNG HỢP: OCR → Dictionary
# =====================================================================

def post_process_ocr_spelling(raw_text):
    """
    Sửa lỗi chính tả OCR cho nhãn thực phẩm tiếng Việt — Pipeline 2 giai đoạn:
      1. Structural cleanup — tách từ dính, chuẩn hóa INS, dọn ký tự rác
      2. Dictionary correction — khôi phục dấu thanh bằng từ điển thực phẩm (diacritic-stripping)

    Returns:
        dict: "final", "after_ocr_dict"
    """
    if not raw_text:
        return {"final": "", "after_ocr_dict": ""}

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

    return {
        "final": t,
        "after_ocr_dict": t,
    }


def parse_allergen_statements(text=None, full_text=None):
    """
    Phân tích 3 khối nhãn dị ứng theo khuyến nghị FAO/WHO JEMRA Part 3 và US FDA FALCPA:
    1. Ingredients list (Danh sách thành phần cấu thành)
    2. Intentional allergen declaration ('Contains' statement / 'Chứa:')
    3. Precautionary Allergen Labeling ('May contain' / PAL / 'Có thể chứa vết...', 'Tiếp xúc chéo')
    
    Hỗ trợ nhận diện riêng khối 'Contains: ...' và khối 'May contain: ...'.
    
    Returns:
        dict: {
            "ingredients_text": str,
            "contains_text": str,
            "may_contain_text": str,
            "pal_text": str,
            "has_contains": bool,
            "has_may_contain": bool,
            "has_pal": bool,
            "contains_allergens": list[str],
            "may_contain_allergens": list[str],
            "pal_allergens": list[str]
        }
    """
    raw_input = text if text is not None else full_text
    if not raw_input:
        return {
            "ingredients_text": "",
            "contains_text": "",
            "may_contain_text": "",
            "pal_text": "",
            "has_contains": False,
            "has_may_contain": False,
            "has_pal": False,
            "contains_allergens": [],
            "may_contain_allergens": [],
            "pal_allergens": []
        }

    target_text = str(raw_input)
    
    # 1. Tìm khối May contain: ... / PAL (Precautionary Allergen Labeling)
    pal_pattern = (
        r"(?:c[oó]\s*th[eể]\s*ch[uứ]a(?:\s*v[eế]t)?(?:\s*c[uủ]a)?|"
        r"may\s*contain(?:\s*traces\s*of)?|"
        r"s[aả]n\s*xu[aấ]t\s*tr[eê]n\s*(?:c[uù]ng\s*)?d[aâ]y\s*chuy[eề]n|"
        r"[đd][uư][oợ]c\s*ch[eế]\s*bi[eế]n\s*tr[eê]n\s*c[uù]ng\s*thi[eế]t\s*b[iị]|"
        r"s[aả]n\s*xu[aấ]t\s*t[aạ]i\s*c[oơ]\s*s[oở]\s*c[oó]\s*x[uử]\s*l[yý])"
        r"\s*[:\-]?\s*([^.\n\r]+?)(?=(?:c[aả]nh\s*b[aá]o\s*d[iị]\s*[uứ]ng|th[oô]ng\s*tin\s*d[iị]\s*[uứ]ng|allergen\b|contains\b|ch[uứ]a\b|[.\n\r;]|$))"
    )
    pal_match = re.search(pal_pattern, target_text, re.IGNORECASE)
    pal_text = pal_match.group(0).strip() if pal_match else ""
    pal_content = pal_match.group(1).strip() if pal_match else ""

    # 2. Tìm khối Contains: ... / Tuyên bố dị ứng chủ ý (Contains statement)
    contains_pattern = (
        r"(?:c[aả]nh\s*b[aá]o\s*d[iị]\s*[uứ]ng|th[oô]ng\s*tin\s*d[iị]\s*[uứ]ng|allergen\s*(?:advice|information|warning))"
        r"\s*[:\-]?\s*(?:ch[uứ]a|c[oó]\s*ch[uứ]a|contains)?\s*[:\-]?\s*([^.\n\r;]+?)(?=(?:may\s*contain|c[oó]\s*th[eể]\s*ch[uứ]a|[.\n\r;]|$))"
    )
    contains_match = re.search(contains_pattern, target_text, re.IGNORECASE)
    if not contains_match:
        direct_contains_pattern = (
            r"(?:^|(?<=[.\s;\n]))(?:ch[uứ]a|c[oó]\s*ch[uứ]a|contains)\s*[:\-]\s*"
            r"([^.\n\r;]+?)(?=(?:may\s*contain|c[oó]\s*th[eể]\s*ch[uứ]a|[.\n\r;]|$))"
        )
        contains_match = re.search(direct_contains_pattern, target_text, re.IGNORECASE)

    contains_text = contains_match.group(0).strip() if contains_match else ""
    contains_content = contains_match.group(1).strip() if contains_match else ""

    def _split_allergen_items(statement):
        if not statement:
            return []
        statement = re.sub(r'^(?:chứa|có chứa|contains)\s*[:\-]?\s*', '', statement, flags=re.IGNORECASE)
        parts = re.split(r'[,;]|(?:\bv[aà]\b)|\band\b', statement, flags=re.IGNORECASE)
        items = []
        for p in parts:
            cleaned = re.sub(r'\s+', ' ', p).strip(" .-:,\n\r\t")
            if cleaned and len(cleaned) >= 2:
                items.append(cleaned)
        return items

    contains_allergens = _split_allergen_items(contains_content)
    pal_allergens = _split_allergen_items(pal_content)

    return {
        "ingredients_text": target_text,
        "contains_text": contains_text,
        "may_contain_text": pal_text,
        "pal_text": pal_text,
        "has_contains": bool(contains_text),
        "has_may_contain": bool(pal_text),
        "has_pal": bool(pal_text),
        "contains_allergens": contains_allergens,
        "may_contain_allergens": pal_allergens,
        "pal_allergens": pal_allergens,
    }


def parse_ingredients_from_ocr(full_text, include_allergen_statements=False):
    """
    Thuật toán Trích xuất thành phần bằng Lookahead Regex:
    - Tìm vị trí xuất hiện của cụm từ 'Thành phần:' hoặc 'Ingredients:'.
    - Cắt tỉa các phần thông tin nhà sản xuất, hạn sử dụng.
    - Tách theo dấu phẩy, chấm phẩy hoặc dấu chấm theo sau bởi khoảng trắng nhưng KHÔNG tách bên trong ngoặc đơn.
      Ví dụ: 'Chất nhũ hóa (INS 322i lecithin, INS 471), muối' -> ['Chất nhũ hóa (INS 322i lecithin, INS 471)', 'muối'].
    - Phân loại danh mục sơ bộ (Category classification).
    - Hỗ trợ gán nhãn khối nguồn: source_block ('ingredient', 'contains', 'precautionary') và is_precautionary flag.
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
            elif any(k in lower_item for k in ["chống oxy hóa", "chống oxy hoá", "antioxidant", "tocopherol"]):
                category = "antioxidant"
            elif any(k in lower_item for k in ["nhũ hóa", "emulsifier", "322", "lecithin", "471"]):
                category = "emulsifier"
            elif any(k in lower_item for k in ["tạo xốp", "500", "503", "baking"]):
                category = "raising_agent"
            elif any(k in lower_item for k in ["làm dày", "1420", "tinh bột biến tính", "modified starch"]):
                category = "thickener"
            elif any(k in lower_item for k in ["ổn định", "ổn đinh", "stabilizer", "stabiliser"]):
                category = "stabilizer"
            elif any(k in lower_item for k in ["hương liệu", "hương tổng hợp", "flavour", "flavor"]):
                category = "flavouring"
            elif any(k in lower_item for k in ["dị ứng", "allergen", "gluten", "đậu phộng", "đậu nành", "trứng", "sữa"]):
                category = "allergen"

            mentions.append({
                "raw_mention": item,
                "category": category,
                "source_block": "ingredient",
                "is_precautionary": False,
            })

    if include_allergen_statements:
        statements = parse_allergen_statements(full_text)
        for ca in statements["contains_allergens"]:
            mentions.append({
                "raw_mention": ca,
                "category": "allergen",
                "source_block": "contains",
                "is_precautionary": False,
            })
        for pa in statements["pal_allergens"]:
            mentions.append({
                "raw_mention": pa,
                "category": "pal_allergen",
                "source_block": "precautionary",
                "is_precautionary": True,
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
    Nạp và parse dữ liệu nhãn sản phẩm benchmark (từ assets/labels/*.json hoặc assets/label_studio/).
    Trả về dict {image_filename: parsed_data} đã cache.
    """
    global _LABEL_STUDIO_CACHE
    if _LABEL_STUDIO_CACHE is not None:
        return _LABEL_STUDIO_CACHE

    if assets_dir is None:
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

    import json
    parsed = {}

    # 1. Nạp từ kho nhãn bóc tách chuẩn hóa assets/labels/*.json (600 sản phẩm thực tế)
    labels_dir = os.path.join(assets_dir, "labels")
    if os.path.isdir(labels_dir):
        label_files = sorted(os.listdir(labels_dir))
        for fname in label_files:
            if not fname.endswith(".json"):
                continue
            stem = os.path.splitext(fname)[0]
            fpath = os.path.join(labels_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as lf:
                    ldata = json.load(lf)
            except Exception:
                continue

            p_raw = ldata.get("product_name")
            p_name = str(p_raw).strip() if p_raw else f"Sản phẩm {stem}"
            i_parts = [str(x).strip() for x in (ldata.get("ingredient") or []) if x and str(x).strip()]
            a_parts = [str(x).strip() for x in (ldata.get("additive") or []) if x and str(x).strip()]
            w_parts = [str(x).strip() for x in (ldata.get("warning") or []) if x and str(x).strip()]

            all_components = i_parts + a_parts
            full_text = "Thành phần: " + ", ".join(all_components) + "."
            if w_parts:
                full_text += " Cảnh báo: " + " ".join(w_parts)

            lines_data = []
            words = full_text.split()
            for i in range(0, len(words), 12):
                lines_data.append({"text": " ".join(words[i:i + 12]), "confidence": 0.99})

            img_rel = f"images/{stem}.jpeg"
            if not os.path.exists(os.path.join(assets_dir, img_rel)):
                for ext in [".jpg", ".png", ".webp"]:
                    candidate = f"images/{stem}{ext}"
                    if os.path.exists(os.path.join(assets_dir, candidate)):
                        img_rel = candidate
                        break

            item_info = {
                "product_name": p_name,
                "full_text": full_text,
                "n_ingredients": len(i_parts),
                "n_additives": len(a_parts),
                "warning_text": " ".join(w_parts),
                "lines": lines_data,
                "image_ref": img_rel,
            }
            parsed[f"{stem}.jpeg"] = item_info
            parsed[stem] = item_info

    # 2. Hỗ trợ dự phòng nếu có file label_studio_exported_data.json
    json_path = os.path.join(assets_dir, "label_studio", "label_studio_exported_data.json")
    if os.path.exists(json_path) and not parsed:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

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

                raw_full = " ".join(full_text_parts)
                words = raw_full.split()
                for i in range(0, len(words), 12):
                    line_text = " ".join(words[i:i + 12])
                    lines_data.append({"text": line_text, "confidence": 0.99})

                image_filename = os.path.basename(image_ref)
                item_info = {
                    "product_name": product_name,
                    "full_text": full_text,
                    "n_ingredients": len(ingredient_parts),
                    "n_additives": len(additive_parts),
                    "warning_text": " ".join(warning_parts),
                    "lines": lines_data,
                    "image_ref": image_ref,
                }
                parsed[image_filename] = item_info
                stem_legacy = os.path.splitext(image_filename)[0]
                parsed[stem_legacy] = item_info
        except Exception:
            pass

    _LABEL_STUDIO_CACHE = parsed
    return _LABEL_STUDIO_CACHE


def get_label_studio_product_list(assets_dir=None):
    """Trả về danh sách sản phẩm mẫu đã bóc tách chuẩn để hiển thị trong dropdown."""
    ls_data = load_label_studio_data(assets_dir)
    products = []
    seen = set()
    for filename, info in sorted(ls_data.items()):
        # Lấy entry chính có phần mở rộng ảnh để tránh duplicate giữa stem và filename
        if not filename.endswith((".jpeg", ".jpg", ".png", ".webp")):
            continue
        stem = os.path.splitext(filename)[0]
        if stem in seen:
            continue
        seen.add(stem)

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
        "avg_confidence": 0.0,
        "lines": [],
        "mentions": [],
        "annotated_image": annotated_img,
    }


# =====================================================================
# PIPELINE 2 TẦNG: PADDLEOCR DETECTION + VIETOCR RECOGNITION
# =====================================================================

def _order_points(pts):
    """Sắp xếp 4 góc polygon: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: tổng nhỏ nhất
    rect[2] = pts[np.argmax(s)]   # bottom-right: tổng lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: hiệu nhỏ nhất
    rect[3] = pts[np.argmax(diff)]  # bottom-left: hiệu lớn nhất
    return rect


def _crop_polygon_region(img_bgr, box):
    """
    Cắt và chỉnh phối cảnh (perspective warp) vùng chữ từ bounding box polygon.
    Trả về ảnh crop đã thẳng, sẵn sàng đưa vào VietOCR.
    """
    try:
        pts = np.array(box, dtype=np.float32)
        if pts.shape != (4, 2):
            pts = pts.reshape(4, 2)
        rect = _order_points(pts)
        (tl, tr, br, bl) = rect

        wA = np.linalg.norm(br - bl)
        wB = np.linalg.norm(tr - tl)
        max_w = max(int(wA), int(wB), 1)

        hA = np.linalg.norm(tr - br)
        hB = np.linalg.norm(tl - bl)
        max_h = max(int(hA), int(hB), 1)

        if max_w < 4 or max_h < 4:
            return None

        dst = np.array([
            [0, 0], [max_w - 1, 0],
            [max_w - 1, max_h - 1], [0, max_h - 1],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img_bgr, M, (max_w, max_h))
    except Exception:
        return None


def _sort_boxes_reading_order(boxes_confs):
    """Sắp xếp các bounding box theo thứ tự đọc: trên→dưới, trái→phải."""
    if not boxes_confs:
        return boxes_confs

    def _key(item):
        pts = np.array(item[0], dtype=np.float32).reshape(-1, 2)
        cy = float(np.mean(pts[:, 1]))
        cx = float(np.mean(pts[:, 0]))
        line_group = round(cy / 15) * 15  # gom các box cùng dòng
        return (line_group, cx)

    return sorted(boxes_confs, key=_key)


def _detect_boxes_paddle_only(engine, img_bgr):
    """
    Chạy PaddleOCR chỉ bước Detection (DBNet), không Recognition.
    Trả về list [(polygon_box, confidence)].
    """
    boxes_confs = []
    try:
        try:
            results = engine.ocr(img_bgr, det=True, rec=False, cls=False)
        except TypeError:
            results = engine.ocr(img_bgr, rec=False)
        if not results:
            return boxes_confs

        # PaddleOCR 2.x detection-only: results[0] = list of [box, score] hoặc [box]
        raw = results[0] if isinstance(results, list) and results else results
        if not raw:
            return boxes_confs
        for item in raw:
            if not isinstance(item, (list, tuple, np.ndarray)) or len(item) < 1:
                continue

            # PaddleOCR 2.9 detection-only trả thẳng một polygon có dạng
            # [[x1,y1], ... [x4,y4]]. Các phiên bản khác có thể bọc thành
            # [polygon, score]. Phải phân biệt hai dạng này trước khi lấy
            # item[0], nếu không polygon sẽ bị rút nhầm còn một điểm [x, y].
            try:
                direct_box = np.asarray(item, dtype=np.float32)
            except (TypeError, ValueError):
                direct_box = None

            if direct_box is not None and direct_box.shape == (4, 2):
                boxes_confs.append((item, 0.85))
                continue

            try:
                nested_box = np.asarray(item[0], dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if nested_box.shape != (4, 2):
                continue

            conf = (
                float(item[1])
                if len(item) > 1 and isinstance(item[1], (int, float, np.number))
                else 0.85
            )
            boxes_confs.append((item[0], conf))
    except Exception as e:
        print(f"[PaddleOCR Detection] Lỗi: {e}")
    return boxes_confs


def _two_step_ocr(engine, vietocr_predictor, img_bgr):
    """
    Pipeline 2 tầng:
      Tầng 1 — PaddleOCR DBNet: phát hiện vùng chữ → polygon bounding boxes
      Tầng 2 — VietOCR VGG-Transformer: nhận diện text tiếng Việt từng vùng

    Trả về (lines_data, full_texts, avg_conf, draw_boxes, pass_name, img_bgr)
    hoặc None nếu không phát hiện được chữ nào.
    """
    boxes_confs = _detect_boxes_paddle_only(engine, img_bgr)
    if not boxes_confs:
        return None

    boxes_confs = _sort_boxes_reading_order(boxes_confs)

    lines_data = []
    full_texts = []
    total_conf = 0.0
    draw_boxes = []

    for box, det_conf in boxes_confs:
        crop = _crop_polygon_region(img_bgr, box)
        if crop is None or crop.size == 0:
            continue
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)
            text = vietocr_predictor.predict(pil_crop)
            if text and text.strip():
                lines_data.append({"text": text.strip(), "confidence": round(det_conf, 3)})
                full_texts.append(text.strip())
                total_conf += det_conf
                draw_boxes.append(box)
        except Exception:
            continue

    if not full_texts:
        return None

    avg_conf = total_conf / len(full_texts)
    return lines_data, full_texts, avg_conf, draw_boxes, "paddle_detect+vietocr_recognize", img_bgr


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


def extract_with_paddle_ocr(image_input, sample_key="custom", **kwargs):
    """
    Thực hiện quét ảnh thực tế bằng PaddleOCR kết hợp chuỗi thuật toán:
    Multi-pass tiền xử lý ➔ Nhận diện DBNet/SVTR ➔ Phục hồi chính tả tiếng Việt.
    """
    img_bgr = convert_to_cv2_image(image_input)
    engine = get_ocr_engine()
    ocr_error = None

    if engine is not None:
        try:
            result = None

            # Tầng 2: Thử pipeline PaddleOCR Detection + VietOCR Recognition trước
            vietocr_predictor = get_vietocr_predictor()
            if vietocr_predictor is not None:
                result = _two_step_ocr(engine, vietocr_predictor, img_bgr)
                pipeline_label = "PaddleOCR DBNet + VietOCR VGG-Transformer"

            # Fallback: Multi-pass PaddleOCR thuần túy
            if result is None:
                result = _multi_pass_ocr(engine, img_bgr)
                pipeline_label = "PaddleOCR Multi-pass"

            if result is not None:
                lines_data, full_texts, avg_conf, boxes, pass_name, processed_img = result
                raw_full_text = " ".join(full_texts)
                spell_result = post_process_ocr_spelling(raw_full_text)
                normalized_full_text = spell_result["final"]
                mentions = parse_ingredients_from_ocr(normalized_full_text)
                annotated_img = draw_ocr_boxes_from_boxes(processed_img, boxes)

                return {
                    "is_real_ocr": True,
                    "is_ground_truth": False,
                    "engine_name": f"{pipeline_label} ({pass_name}) + Phục hồi Chính tả Thực phẩm",
                    "full_text": normalized_full_text,
                    "raw_text_before_norm": raw_full_text,
                    "text_after_ocr_dict": spell_result["after_ocr_dict"],
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
