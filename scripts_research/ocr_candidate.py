"""Thử mô hình DÒ VÙNG CHỮ mới (PaddleOCR 3.x, vd PP-OCRv6_medium_det) trên đúng các ảnh của bộ kiểm tra OCR,
giữ nguyên VietOCR đọc chữ — để so công bằng với hệ thống hiện tại bằng scripts_research/ocr_regression.py.

Vì PaddleOCR 3.x cần paddlepaddle 3.x (khác venv chính đang dùng 2.6.2), script chạy làm HAI bước ở HAI venv:

  Bước 1 — dò vùng chữ (venv_ocr3, chỉ cần paddlepaddle + paddleocr 3.x; KHÔNG import mã dự án):
    venv_ocr3\\Scripts\\python.exe scripts_research\\ocr_candidate.py detect --model PP-OCRv6_medium_det
    → data/ocr_candidate/<model>_boxes.json   (khung chữ + điểm + số giây mỗi ảnh; chạy lại sẽ làm tiếp)

  Bước 2 — đọc chữ trong các khung bằng VietOCR (venv chính):
    venv\\Scripts\\python.exe scripts_research\\ocr_candidate.py recognize --model PP-OCRv6_medium_det
    → data/ocr_candidate/<model>_cache.json   (cùng định dạng bộ đệm OCR; chạy lại sẽ làm tiếp)

  Bước 3 — so với hệ thống hiện tại trên CÙNG tập ảnh:
    venv\\Scripts\\python.exe scripts_research\\ocr_regression.py --cache data\\ocr_candidate\\PP-OCRv6_medium_det_cache.json

Danh sách ảnh lấy từ bộ đệm OCR hiện tại (data/ocr_cache_config_c.json) nên hai bên đo đúng cùng ảnh.
--limit N để chạy thử nhanh N ảnh đầu (bước 3 chỉ so trên các ảnh đã có ở cả hai bên).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_CACHE = PROJECT_ROOT / "data" / "ocr_cache_config_c.json"
OUT_DIR = PROJECT_ROOT / "data" / "ocr_candidate"


def _image_keys(limit=0):
    cache = json.loads(CURRENT_CACHE.read_text(encoding="utf-8")).get("images", {})
    keys = sorted(k for k, v in cache.items() if not v.get("is_ocr_unavailable"))
    return keys[:limit] if limit else keys


def _read_bgr(path):
    import cv2
    import numpy as np
    data = np.fromfile(str(path), dtype=np.uint8)  # đọc được cả đường dẫn có dấu tiếng Việt
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Bước 1: dò vùng chữ bằng PaddleOCR 3.x (venv_ocr3)
# ---------------------------------------------------------------------------
def detect(args):
    from paddleocr import TextDetection

    # oneDNN (MKLDNN) của paddlepaddle 3.x trên Windows CPU lỗi với mô hình PIR:
    # "ConvertPirAttribute2RuntimeAttribute not support" → tắt mặc định (--mkldnn để bật lại)
    kwargs = {"model_name": args.model, "device": args.device, "enable_mkldnn": bool(args.mkldnn)}
    for name in ("thresh", "box_thresh", "unclip_ratio", "limit_side_len"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    model = TextDetection(**kwargs)

    out_path = OUT_DIR / f"{args.model}_boxes.json"
    store = _load(out_path) or {"model": args.model, "params": kwargs, "images": {}}
    keys = [k for k in _image_keys(args.limit) if k not in store["images"]]
    print(f"[detect] {args.model} trên {args.device} · cần dò: {len(keys)} ảnh → {out_path}")
    t0 = time.perf_counter()
    for i, key in enumerate(keys, 1):
        img = _read_bgr(key)
        if img is None:
            store["images"][key] = {"error": "không đọc được ảnh", "boxes": [], "scores": []}
            continue
        t = time.perf_counter()
        res = next(iter(model.predict(img, batch_size=1)))
        polys = [[[float(x), float(y)] for x, y in poly] for poly in res["dt_polys"]]
        scores = [float(s) for s in res.get("dt_scores", [])] or [0.85] * len(polys)
        store["images"][key] = {"boxes": polys, "scores": scores, "seconds": round(time.perf_counter() - t, 2)}
        if i % 10 == 0 or i == len(keys):
            _save(out_path, store)
            eta = (time.perf_counter() - t0) / i * (len(keys) - i)
            print(f"[detect] {i}/{len(keys)} · {len(polys)} khung · còn ~{eta / 60:.0f} phút", flush=True)
    _save(out_path, store)
    print("[detect] xong.")


# ---------------------------------------------------------------------------
# Bước 2: đọc chữ trong khung bằng VietOCR, ghép như hệ thống hiện tại (venv chính)
# ---------------------------------------------------------------------------
def recognize(args):
    sys.path.insert(0, str(PROJECT_ROOT))
    import cv2
    import numpy as np
    from PIL import Image
    from modules import ocr_extractor as ox

    boxes_path = OUT_DIR / f"{args.model}_boxes.json"
    if not boxes_path.is_file():
        sys.exit(f"Chưa có {boxes_path} — chạy bước detect trong venv_ocr3 trước.")
    boxes = _load(boxes_path)
    predictor = ox.get_vietocr_predictor()
    if predictor is None:
        sys.exit("Không khởi tạo được VietOCR trong venv chính.")

    out_path = OUT_DIR / f"{args.model}_cache.json"
    store = _load(out_path) or {"pipeline_version": f"candidate:{args.model}+vietocr", "images": {}}
    keys = [k for k in boxes["images"] if k not in store["images"]]
    if args.limit:
        keys = keys[:args.limit]
    print(f"[recognize] VietOCR đọc {len(keys)} ảnh → {out_path}")
    ox._ensure_torch_threads()
    t0 = time.perf_counter()
    for i, key in enumerate(keys, 1):
        det = boxes["images"][key]
        img = _read_bgr(key)
        if img is None or not det.get("boxes"):
            store["images"][key] = {"raw_text_before_norm": "", "is_ocr_unavailable": img is None, "n_boxes": 0}
            continue
        quads = []
        for poly, score in zip(det["boxes"], det["scores"]):
            pts = np.asarray(poly, dtype=np.float32)
            if pts.shape != (4, 2):  # đa giác nhiều điểm → hình chữ nhật xoay nhỏ nhất
                pts = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32)
            quads.append((pts.tolist(), score))
        quads = ox._sort_boxes_reading_order(quads)
        crops = []
        for box, _ in quads:
            crop = ox._crop_polygon_region(img, box)
            if crop is not None and crop.size:
                crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        t = time.perf_counter()
        texts, probs = ox._vietocr_read_batch(predictor, crops) if crops else ([], [])
        kept = [(tx.strip(), p) for tx, p in zip(texts, probs) if tx and tx.strip()]
        store["images"][key] = {
            "raw_text_before_norm": " ".join(tx for tx, _ in kept),
            "avg_confidence": round(sum(p for _, p in kept) / len(kept), 4) if kept else 0.0,
            "n_boxes": len(quads),
            "det_seconds": det.get("seconds"),
            "rec_seconds": round(time.perf_counter() - t, 2),
        }
        if i % 5 == 0 or i == len(keys):
            _save(out_path, store)
            eta = (time.perf_counter() - t0) / i * (len(keys) - i)
            print(f"[recognize] {i}/{len(keys)} · {len(quads)} khung · còn ~{eta / 60:.0f} phút", flush=True)
    _save(out_path, store)
    print("[recognize] xong. So sánh: venv\\Scripts\\python.exe scripts_research\\ocr_regression.py --cache " + str(out_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["detect", "recognize"])
    ap.add_argument("--model", default="PP-OCRv6_medium_det", help="tên mô hình dò vùng chữ của PaddleOCR 3.x")
    ap.add_argument("--device", default="cpu", help="cpu hoặc gpu:0 (cần paddlepaddle-gpu)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu (thử nhanh)")
    ap.add_argument("--mkldnn", action="store_true", help="bật oneDNN trên CPU (mặc định tắt vì lỗi PIR trên Windows)")
    ap.add_argument("--thresh", type=float, help="ngưỡng điểm ảnh chữ (mặc định của mô hình)")
    ap.add_argument("--box_thresh", type=float, help="ngưỡng giữ khung")
    ap.add_argument("--unclip_ratio", type=float, help="độ nới khung")
    ap.add_argument("--limit_side_len", type=int, help="cạnh ảnh khi dò")
    args = ap.parse_args()
    detect(args) if args.step == "detect" else recognize(args)


if __name__ == "__main__":
    main()
