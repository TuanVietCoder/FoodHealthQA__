"""So VietOCR gốc với bản fine-tune trên các ảnh KIỂM TRA (test) — phần không dùng để huấn luyện.

Cùng một lần dò vùng chữ (DBNet) cho cả hai mô hình, nên khác biệt chỉ đến từ bộ nhận dạng. Chữ OCR thô của mỗi mô hình
được chấm bằng đúng các chỉ số của ocr_regression.py (token_recall, item_recall, code_precision/recall, dị nguyên giả...)
nhưng chỉ trên ảnh test (split_of, dev_fraction 0,5 — giống build_recognizer_dataset.py).

Dùng (venv chính có PaddleOCR):
    python scripts_research/eval_vietocr_ft.py [--limit N] [--device auto|cpu|cuda:0] [--ft models/vietocr/vgg_transformer_ft.pth]
Ra: data/analysis/vietocr_ft_vs_base.json
Checkpoint: data/analysis/vietocr_ft_vs_base.ckpt.jsonl (mỗi ảnh một dòng). Bị ngắt thì chạy lại cùng lệnh để chạy tiếp;
--fresh để bỏ checkpoint. --limit tăng dần (5 → 60 → đủ) cũng dùng lại các ảnh đã chạy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_research"))

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

import calibrate_confidence as cc  # noqa: E402
import ocr_regression as reg  # noqa: E402
from modules import ocr_extractor as oe  # noqa: E402

DEV_FRACTION = 0.5
OUT = ROOT / "data" / "analysis" / "vietocr_ft_vs_base.json"
KEYS = ("token_recall", "item_recall", "code_precision", "code_recall", "allergen_precision", "allergen_recall",
        "fake_allergens", "oversized_items")


CKPT = ROOT / "data" / "analysis" / "vietocr_ft_vs_base.ckpt.jsonl"


def weights_sig(path):
    p = Path(path)
    return f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" if p.is_file() else f"{p.name}:missing"


def make_predictor(weights, device):
    if not oe.is_vietocr_available():  # nạp lười: _VietOCRCfg/_VietOCRPredictor chỉ có sau lệnh này
        raise RuntimeError(f"VietOCR không nạp được: {oe._VIETOCR_INIT_ERROR}")
    cfg = oe._VietOCRCfg.load_config_from_file(oe._VIETOCR_CONFIG_PATH)
    cfg["weights"] = str(weights)
    cfg["pretrain"] = ""
    cfg["cnn"]["pretrained"] = False
    cfg["device"] = device
    cfg["predictor"]["beamsearch"] = False
    return oe._VietOCRPredictor(cfg)


def test_images():
    """[(đường dẫn ảnh, thành phần chuẩn)] của ảnh test, mỗi ảnh một lần."""
    seen, out = set(), []
    for line in open(reg.BENCHMARK, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if cc.split_of(r, DEV_FRACTION) != "test":
            continue
        img = reg._resolve_image(r)
        ingredients = str(r.get("product_ingredients") or "").strip()
        if img is None or not ingredients or str(img) in seen:
            continue
        seen.add(str(img))
        out.append((img, r["product_ingredients"]))
    return sorted(out, key=lambda x: x[0].name)


def read_all(predictor, crops):
    texts = []
    for i in range(0, len(crops), oe.VIETOCR_BATCH_SIZE):
        t, _ = oe._vietocr_read_batch(predictor, crops[i : i + oe.VIETOCR_BATCH_SIZE])
        texts += t
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto (cuda nếu có, không thì cpu) | cpu | cuda:0")
    ap.add_argument("--base", default=str(ROOT / "models" / "vietocr" / "vgg_transformer.pth"))
    ap.add_argument("--ft", default=str(ROOT / "models" / "vietocr" / "vgg_transformer_ft.pth"))
    ap.add_argument("--fresh", action="store_true", help="bỏ checkpoint cũ, chạy lại từ đầu")
    args = ap.parse_args()

    items = test_images()
    if args.limit:
        items = items[: args.limit]
    print(f"Ảnh test: {len(items)}")

    # Checkpoint: mỗi ảnh xong được ghi ngay một dòng; chạy lại chỉ xử lý ảnh còn thiếu.
    # Chữ thô được lưu kèm chữ ký trọng số, nên đổi/huấn luyện lại trọng số thì checkpoint tự mất hiệu lực.
    sig = {"base": weights_sig(args.base), "finetuned": weights_sig(args.ft)}
    done = {}
    if CKPT.is_file() and not args.fresh:
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue  # dòng cuối có thể dở dang nếu bị ngắt giữa chừng
            if r.get("sig") == sig:
                done[r["image"]] = r
    elif args.fresh and CKPT.is_file():
        CKPT.unlink()
    pending = [it for it in items if it[0].name not in done]
    print(f"Đã có trong checkpoint: {len(items) - len(pending)}, còn phải chạy: {len(pending)}")

    if pending:
        engine = oe.get_ocr_engine()
        if engine is None:
            sys.exit("Cần PaddleOCR (chạy bằng venv chính, không phải venv_train).")
        import torch

        device = args.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda") and not torch.cuda.is_available():
            sys.exit("torch trong venv này là bản +cpu nên không thấy GPU. Cài bản CUDA vào venv (xem hướng dẫn) "
                     "hoặc chạy với --device cpu.")
        print(f"VietOCR chạy trên: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.startswith("cuda") else ""))
        models = {"base": make_predictor(args.base, device), "finetuned": make_predictor(args.ft, device)}
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        with open(CKPT, "a", encoding="utf-8") as ck:
            for n, (path, ingredients) in enumerate(pending, 1):
                img = oe.convert_to_cv2_image(str(path))
                boxes = oe._sort_boxes_reading_order(oe._detect_boxes_paddle_only(engine, img))
                crops = []
                for box, _ in boxes:
                    c = oe._crop_polygon_region(img, box)
                    if c is not None and c.size:
                        crops.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                rec = {"image": path.name, "sig": sig}
                for name, predictor in models.items():
                    rec[name] = " ".join(t.strip() for t in read_all(predictor, crops) if t and t.strip()) if crops else ""
                ck.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ck.flush()
                done[path.name] = rec
                if n % 10 == 0:
                    print(f"  {n}/{len(pending)}")

    per_model = {name: [(p.name, done[p.name][name], ing) for p, ing in items] for name in ("base", "finetuned")}
    summaries = {name: reg.evaluate(rows)[0] for name, rows in per_model.items()}
    print(f"\n{'chỉ số':<20}{'gốc':>10}{'fine-tune':>12}")
    for k in KEYS:
        b, f = summaries["base"].get(k), summaries["finetuned"].get(k)
        print(f"{k:<20}{reg._fmt(b):>10}{reg._fmt(f):>12}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"images": len(items), "summaries": summaries}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nĐã ghi {OUT}")


if __name__ == "__main__":
    main()
