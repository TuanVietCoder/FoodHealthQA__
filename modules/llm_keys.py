"""
Quản lý danh sách khóa API Gemini cho các lần chạy dài, bị giới hạn hạn mức theo ngày.

Nguồn khóa (theo thứ tự ưu tiên, gộp lại và loại trùng):
  1. Biến môi trường / .env  GEMINI_API_KEYS  (phân tách bằng dấu phẩy, chấm phẩy hoặc xuống dòng)
  2. File gemini_keys.txt ở thư mục gốc dự án (mỗi dòng một khóa, '#' là chú thích)
  3. GEMINI_API_KEY / GOOGLE_API_KEY (một khóa, như trước đây)

Cách dùng khóa: dùng lần lượt từng khóa cho đến khi hết hạn mức ngày (HTTP 429 "PerDay" hoặc chạm
GEMINI_DAILY_LIMIT_PER_KEY), rồi mới chuyển sang khóa kế tiếp. Hết toàn bộ khóa thì ở chế độ strict
(do script sinh dự đoán bật) sẽ ném LLMQuotaExhausted để dừng sạch; ngày hôm sau chạy lại với --resume.

Lưu ý: hạn mức Gemini tính theo *project* Google Cloud chứ không theo khóa. Các khóa tạo trong cùng một
project dùng chung hạn mức; chỉ khóa thuộc các project khác nhau mới thực sự nhân hạn mức.

Không bao giờ ghi khóa ra log/file: trạng thái chỉ lưu dấu vân tay băm (10 ký tự đầu của sha256).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
KEYS_FILE = _ROOT / "gemini_keys.txt"
USAGE_FILE = _ROOT / "data" / "runs" / "gemini_key_usage.json"


class LLMQuotaExhausted(RuntimeError):
    """Mọi khóa Gemini đã hết hạn mức (chỉ ném ở chế độ strict)."""


class LLMUnavailable(RuntimeError):
    """Nhà cung cấp LLM không truy cập được, ví dụ Ollama chưa chạy (chỉ ném ở chế độ strict)."""


_STRICT = False


def set_strict(value: bool) -> None:
    """strict=True: lỗi hạn mức/không kết nối ném exception thay vì trả None (dùng cho chạy batch)."""
    global _STRICT
    _STRICT = bool(value)


def is_strict() -> bool:
    return _STRICT


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def _split(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;\r\n]+", text or "") if p.strip() and not p.strip().startswith("#")]


def load_keys(keys_file: Path | None = None) -> list[str]:
    """Đọc danh sách khóa từ env/.env, gemini_keys.txt và khóa đơn; giữ thứ tự, bỏ trùng."""
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except Exception:
        pass
    keys: list[str] = []
    keys += _split(os.environ.get("GEMINI_API_KEYS", ""))
    path = Path(keys_file) if keys_file else KEYS_FILE
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                keys.append(line)
    except OSError:
        pass
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(var):
            keys.append(os.environ[var].strip())
    return list(dict.fromkeys(k for k in keys if k))


def _quota_day() -> str:
    # Hạn mức Gemini reset theo giờ Thái Bình Dương; dùng UTC-8 làm xấp xỉ cho nhãn ngày.
    return datetime.now(timezone(timedelta(hours=-8))).strftime("%Y-%m-%d")


def classify_error(exc: Exception) -> str:
    """'daily' = hết hạn mức ngày, 'rate' = giới hạn theo phút (chờ rồi thử lại), 'other' = lỗi khác."""
    msg = str(exc)
    low = msg.lower()
    if "429" not in msg and "resource_exhausted" not in low and "quota" not in low:
        return "other"
    if "perday" in low.replace(" ", "").replace("_", "") or "per day" in low or "daily" in low:
        return "daily"
    return "rate"


def retry_delay_seconds(exc: Exception, default: float = 20.0, cap: float = 65.0) -> float:
    m = re.search(r"retry(?:Delay)?\D{0,12}(\d+(?:\.\d+)?)\s*s", str(exc), re.I)
    return min(float(m.group(1)) + 1.0, cap) if m else default


class KeyPool:
    """Danh sách khóa dùng tuần tự, đếm số request theo ngày, đánh dấu khóa đã hết hạn mức."""

    def __init__(self, keys: list[str], daily_limit: int | None = None, usage_file: Path | None = None):
        self.keys = list(keys)
        env_limit = os.environ.get("GEMINI_DAILY_LIMIT_PER_KEY", "").strip()
        self.daily_limit = daily_limit if daily_limit is not None else (int(env_limit) if env_limit.isdigit() else 0)
        self.usage_file = Path(usage_file) if usage_file else USAGE_FILE
        self._exhausted: set[str] = set()  # chỉ trong bộ nhớ: mỗi lần chạy mới sẽ thăm dò lại một lần
        self._lock = threading.Lock()
        self._usage = self._load_usage()

    # ---- bộ đếm theo ngày (bền vững qua các lần chạy) ----
    def _load_usage(self) -> dict:
        try:
            data = json.loads(self.usage_file.read_text(encoding="utf-8"))
            if data.get("date") == _quota_day():
                return data
        except (OSError, ValueError):
            pass
        return {"date": _quota_day(), "requests": {}}

    def _save_usage(self) -> None:
        try:
            self.usage_file.parent.mkdir(parents=True, exist_ok=True)
            self.usage_file.write_text(json.dumps(self._usage, indent=2), encoding="utf-8")
        except OSError:
            pass

    def used_today(self, key: str) -> int:
        return int(self._usage["requests"].get(fingerprint(key), 0))

    def record_request(self, key: str) -> None:
        with self._lock:
            if self._usage.get("date") != _quota_day():
                self._usage = {"date": _quota_day(), "requests": {}}
            fp = fingerprint(key)
            self._usage["requests"][fp] = self._usage["requests"].get(fp, 0) + 1
            self._save_usage()

    # ---- chọn khóa ----
    def mark_exhausted(self, key: str) -> None:
        self._exhausted.add(key)

    def acquire(self) -> str | None:
        """Khóa đầu tiên còn dùng được (dính vào một khóa cho đến khi hết hạn mức)."""
        for key in self.keys:
            if key in self._exhausted:
                continue
            if self.daily_limit and self.used_today(key) >= self.daily_limit:
                continue
            return key
        return None

    def status(self) -> list[dict]:
        return [
            {
                "key": f"#{i + 1} ({fingerprint(k)})",
                "used_today": self.used_today(k),
                "exhausted_this_run": k in self._exhausted,
                "daily_limit": self.daily_limit or None,
            }
            for i, k in enumerate(self.keys)
        ]


_POOL: KeyPool | None = None


def get_pool(extra_key: str | None = None) -> KeyPool:
    """Pool dùng chung; nếu caller truyền một khóa đơn chưa có trong danh sách thì thêm vào cuối."""
    global _POOL
    if _POOL is None:
        _POOL = KeyPool(load_keys())
    if extra_key and extra_key not in _POOL.keys:
        _POOL.keys.append(extra_key)
    return _POOL


def reset_pool() -> None:
    """Dùng trong test hoặc khi đổi danh sách khóa lúc chạy."""
    global _POOL
    _POOL = None
