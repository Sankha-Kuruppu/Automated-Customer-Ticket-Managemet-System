"""Build models/encoder_ft = pinned base encoder (models/encoder) + fine-tuned transformer-layer delta.

Run after download_encoder.py (the Dockerfile does both at build time). No-op if no delta exists.
"""
import shutil
from pathlib import Path

from safetensors.torch import load_file
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent.parent
ENC = ROOT / "models" / "encoder"
DELTA = ROOT / "models" / "encoder_ft_delta.safetensors"
OUT = ROOT / "models" / "encoder_ft"

if __name__ == "__main__":
    if not DELTA.exists():
        print("no fine-tuned delta found; using base encoder")
        raise SystemExit(0)
    model = AutoModel.from_pretrained(str(ENC), local_files_only=True)
    delta = {k: v.float() for k, v in load_file(str(DELTA)).items()}
    missing, unexpected = model.load_state_dict(delta, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert all(not k.startswith("encoder.") for k in missing), "delta does not cover all encoder layers"
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(ENC, OUT, ignore=shutil.ignore_patterns("model.safetensors", ".cache", "onnx"))
    model.save_pretrained(str(OUT), safe_serialization=True)
    print(f"applied {len(delta)} tensors from {DELTA.name} -> {OUT}")
