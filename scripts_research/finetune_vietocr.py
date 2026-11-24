"""Fine-tune bộ nhận dạng VietOCR (vgg_transformer) trên cặp (crop, text) do build_recognizer_dataset.py tạo.

Khởi tạo từ trọng số hiện có models/vietocr/vgg_transformer.pth, huấn luyện trên GPU, lưu trọng số tốt nhất vào
models/vietocr/vgg_transformer_ft.pth. Để pipeline dùng bản mới:
    set VIETOCR_WEIGHTS_PATH=models\\vietocr\\vgg_transformer_ft.pth     (PowerShell: $env:VIETOCR_WEIGHTS_PATH=...)
Sau đó xóa/đổi OCR_PIPELINE_VERSION để bộ đệm OCR cũ (data/ocr_cache_config_c.json) được chạy lại.

YÊU CẦU: torch bản CUDA. venv hiện có torch 2.3.1+cpu nên huấn luyện trên GPU cần bản CUDA, ví dụ trong một venv riêng:
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
    pip install vietocr==0.3.13 opencv-python pillow
(Không cài đè vào venv chạy ứng dụng nếu muốn giữ nguyên bản CPU.)

Dùng:
    python scripts_research/finetune_vietocr.py --iters 5000 --batch-size 16
    python scripts_research/finetune_vietocr.py --device cpu --iters 50    # chạy thử luồng, không cần GPU
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ocr_train"
MODEL_DIR = ROOT / "models" / "vietocr"
BASE_WEIGHTS = MODEL_DIR / "vgg_transformer.pth"
BASE_CONFIG = MODEL_DIR / "vgg_transformer.json"
OUT_WEIGHTS = MODEL_DIR / "vgg_transformer_ft.pth"
MAX_LEN = 100  # VietOCR giải mã tối đa 128 ký tự gồm token đặc biệt


def filtered_annotation(src: Path, dst: Path, vocab: set[str]) -> tuple[int, int]:
    """Bỏ mẫu có ký tự ngoài bảng ký tự của mô hình (Vocab.encode sẽ lỗi KeyError) hoặc quá dài."""
    kept, dropped = [], 0
    for line in src.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        rel, text = line.split("\t", 1)
        text = unicodedata.normalize("NFC", text)
        if not text or len(text) > MAX_LEN or any(c not in vocab for c in text):
            dropped += 1
            continue
        kept.append(f"{rel}\t{text}")
    dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return len(kept), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5, help="thấp hơn mặc định (1e-3) vì là fine-tune, tránh quên chữ đã học")
    ap.add_argument("--valid-every", type=int, default=250)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    for f in (BASE_WEIGHTS, BASE_CONFIG, DATA / "train.txt", DATA / "val.txt"):
        if not f.is_file():
            sys.exit(f"Thiếu {f} (chạy build_recognizer_dataset.py trước).")

    import torch
    from vietocr.model.trainer import Trainer
    from vietocr.tool.config import Cfg

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("torch không thấy CUDA (có thể đang dùng bản +cpu). Xem hướng dẫn đầu file hoặc dùng --device cpu.")

    cfg = Cfg.load_config_from_file(str(BASE_CONFIG))
    vocab = set(cfg["vocab"])
    n_tr, d_tr = filtered_annotation(DATA / "train.txt", DATA / "train_filtered.txt", vocab)
    n_va, d_va = filtered_annotation(DATA / "val.txt", DATA / "val_filtered.txt", vocab)
    print(f"train {n_tr} (bỏ {d_tr}), val {n_va} (bỏ {d_va})")
    if n_tr < 50 or n_va < 10:
        sys.exit("Quá ít mẫu; chạy lại build_recognizer_dataset.py trên toàn bộ ảnh hoặc hạ --min-score.")

    cfg["pretrain"] = ""
    cfg["weights"] = str(BASE_WEIGHTS)
    cfg["device"] = args.device
    cfg["cnn"]["pretrained"] = False
    cfg["dataset"].update({
        "data_root": str(DATA), "train_annotation": "train_filtered.txt", "valid_annotation": "val_filtered.txt",
        "name": "foodlabel", "image_height": 32, "image_min_width": 32, "image_max_width": 512,
    })
    cfg["dataloader"].update({"num_workers": 0})  # Windows: tránh lỗi multiprocessing
    cfg["optimizer"]["max_lr"] = args.lr
    cfg["trainer"].update({
        "batch_size": args.batch_size, "iters": args.iters, "valid_every": args.valid_every,
        "print_every": 50, "checkpoint": str(MODEL_DIR / "ft_checkpoint.pth"), "export": str(OUT_WEIGHTS),
        "metrics": 10000,
    })
    cfg["aug"] = {"image_aug": True, "masked_language_model": False}

    trainer = Trainer(cfg, pretrained=False)
    trainer.load_weights(str(BASE_WEIGHTS))  # nạp trọng số gốc làm điểm xuất phát
    trainer.train()
    json.dump({"iters": args.iters, "lr": args.lr, "train": n_tr, "val": n_va, "weights": str(OUT_WEIGHTS)},
              open(MODEL_DIR / "ft_run.json", "w", encoding="utf-8"), indent=2)
    print(f"Xong. Trọng số: {OUT_WEIGHTS}")


if __name__ == "__main__":
    main()
