"""Lưu lịch sử hỏi đáp của giao diện chat thành các file JSON (mỗi cuộc trò chuyện một file).

Thư mục: data/chat_history/ — chứa hồ sơ sức khỏe và ảnh nhãn thu nhỏ của người dùng nên KHÔNG đưa lên git (xem .gitignore).

Hàm ở đây thuần (không gọi Streamlit) để kiểm thử độc lập. Những gì không lưu được dạng JSON (ảnh PIL gốc, ảnh có khung
OCR) bị bỏ; phần còn lại đủ để hiển thị lại các câu trả lời và hỏi tiếp trên ảnh đã gửi (cần `ocr_result` + `mentions`).
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = _ROOT / "data" / "chat_history"

_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
# khóa của ctx không lưu: ảnh PIL gốc và ảnh có khung OCR (nặng, không JSON được)
_CTX_SKIP = {"image", "annotated_images"}
_SKIP_TITLE = ("📎", "✅")


def new_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def is_valid_id(conv_id: Any) -> bool:
    return bool(conv_id) and bool(_ID_RE.match(str(conv_id)))


def encode(obj: Any) -> Any:
    """Đổi về kiểu JSON được. bytes → {"__b64__": …}; tuple/set → list; ảnh PIL → None; số numpy → số thường."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, (bytes, bytearray)):
        return {"__b64__": base64.b64encode(bytes(obj)).decode("ascii")}
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [encode(v) for v in obj]
    if type(obj).__module__.startswith("PIL"):
        return None
    if hasattr(obj, "tolist"):          # numpy mảng / số
        return encode(obj.tolist())
    if hasattr(obj, "item"):
        return encode(obj.item())
    return str(obj)


def decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {"__b64__"}:
            return base64.b64decode(obj["__b64__"])
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj


def make_title(chat: List[dict], ctx: Optional[dict] = None, max_len: int = 45) -> str:
    for msg in chat or []:
        text = str(msg.get("text") or "").strip() if msg.get("role") == "user" else ""
        if text and not text.startswith(_SKIP_TITLE):
            first = text.splitlines()[0].strip()
            return first if len(first) <= max_len else first[: max_len - 1] + "…"
    name = str((ctx or {}).get("display_name") or "").strip()
    return name[:max_len] if name else "Cuộc trò chuyện mới"


def _path(conv_id: str, directory: Optional[Path] = None) -> Path:
    if not is_valid_id(conv_id):
        raise ValueError(f"Mã cuộc trò chuyện không hợp lệ: {conv_id!r}")
    return Path(directory or HISTORY_DIR) / f"{conv_id}.json"


def save_conversation(conv_id: str, chat: List[dict], ctx: Optional[dict] = None, directory: Optional[Path] = None) -> Path:
    """Ghi (đè) một cuộc trò chuyện. Ghi qua file tạm rồi đổi tên để file không bị cụt nếu bị ngắt giữa chừng."""
    path = _path(conv_id, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = None
    if path.is_file():
        try:
            created = json.loads(path.read_text(encoding="utf-8")).get("created")
        except (OSError, ValueError):
            created = None
    now = datetime.now().isoformat(timespec="seconds")
    payload = {
        "id": conv_id,
        "title": make_title(chat, ctx),
        "created": created or now,
        "updated": now,
        "messages": encode(list(chat or [])),
        "ctx": encode({k: v for k, v in (ctx or {}).items() if k not in _CTX_SKIP}),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_conversation(conv_id: str, directory: Optional[Path] = None) -> Optional[Tuple[List[dict], dict]]:
    """Trả về (chat, ctx đã lưu) hoặc None nếu không có / file hỏng."""
    try:
        data = json.loads(_path(conv_id, directory).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return decode(data.get("messages") or []), decode(data.get("ctx") or {})


def list_conversations(limit: int = 30, directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Các cuộc trò chuyện đã lưu, mới nhất trước: [{id, title, updated, n_messages}]."""
    folder = Path(directory or HISTORY_DIR)
    out = []
    if not folder.is_dir():
        return out
    for path in folder.glob("*.json"):
        if not is_valid_id(path.stem):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            out.append({"id": path.stem, "title": data.get("title") or "Cuộc trò chuyện",
                        "updated": data.get("updated") or "", "n_messages": len(data.get("messages") or [])})
    out.sort(key=lambda c: (c["updated"], c["id"]), reverse=True)
    return out[:limit]


def delete_conversation(conv_id: str, directory: Optional[Path] = None) -> bool:
    try:
        _path(conv_id, directory).unlink()
        return True
    except (OSError, ValueError):
        return False
