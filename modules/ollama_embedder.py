"""
Embedding đa ngôn ngữ chạy qua Ollama (mặc định bge-m3), không cần torch/sentence-transformers.

    ollama pull bge-m3
    enc = OllamaEmbedder("bge-m3")
    vectors = enc(["câu 1", "câu 2"])        # np.ndarray (n, dim), đã chuẩn hóa L2

Vector được cache trên đĩa theo (model, sha256(text)) nên 1.095 đoạn KB chỉ phải embed một lần,
và các truy vấn lặp lại (nhiều top-k, nhiều lần chạy) không gọi lại Ollama.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / "data" / "cache"


class OllamaEmbedder:
    def __init__(self, model: str = "bge-m3", host: str | None = None, batch_size: int = 16,
                 cache_dir: Path | None = None, timeout: int = 300):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in model)
        self.cache_path = Path(cache_dir or CACHE_DIR) / f"embeddings_{safe}.npz"
        self._mem: dict[str, np.ndarray] = {}
        self._dirty = 0
        self._load_cache()

    # ---- cache ----
    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def _load_cache(self) -> None:
        try:
            with np.load(self.cache_path, allow_pickle=False) as data:
                self._mem = dict(zip(data["keys"].tolist(), data["vecs"]))
        except (OSError, ValueError, KeyError):
            self._mem = {}

    def save_cache(self) -> None:
        if not self._mem:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._mem), dtype="U24")
        vecs = np.stack(list(self._mem.values())).astype(np.float32)
        temporary = self.cache_path.with_name(self.cache_path.stem + ".tmp.npz")
        np.savez(temporary, keys=keys, vecs=vecs)
        os.replace(temporary, self.cache_path)
        self._dirty = 0

    # ---- encode ----
    def _embed_remote(self, texts: list[str]) -> np.ndarray:
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(f"{self.host}/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            vecs = np.asarray(json.loads(resp.read())["embeddings"], dtype=np.float32)
        if vecs.shape[0] != len(texts):
            raise RuntimeError(f"Ollama trả {vecs.shape[0]} vector cho {len(texts)} văn bản")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms

    def __call__(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self._mem]
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start:start + self.batch_size]
            vecs = self._embed_remote([t for _, t in chunk])
            for (k, _), v in zip(chunk, vecs):
                self._mem[k] = v
            self._dirty += len(chunk)
        if self._dirty >= 200 or (missing and len(texts) > 50):
            self.save_cache()  # lô lớn (embed cả corpus) lưu ngay; truy vấn lẻ lưu theo đợt
        return np.stack([self._mem[k] for k in keys]) if keys else np.zeros((0, 0), dtype=np.float32)
