"""Multilingual sentence encoder (intfloat/multilingual-e5-small), loaded from local files only."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DEFAULT_ENCODER_DIR = Path(__file__).resolve().parent.parent / "models" / "encoder"

# Never reach out to the internet at runtime.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class Encoder:
    def __init__(self, path: Path = DEFAULT_ENCODER_DIR, batch_size: int = 32):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.model = SentenceTransformer(str(path), device="cpu", local_files_only=True)
        self.model.max_seq_length = 512
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)
