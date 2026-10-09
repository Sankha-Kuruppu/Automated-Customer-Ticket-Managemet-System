"""Download the multilingual sentence encoder into models/encoder (run once; Docker runs it at build time).

The service never downloads anything at runtime: the image ships these files.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

ENCODER_REPO = "intfloat/multilingual-e5-small"
ENCODER_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"  # pinned commit
TARGET = Path(__file__).resolve().parent.parent / "models" / "encoder"

if __name__ == "__main__":
    snapshot_download(
        repo_id=ENCODER_REPO,
        revision=ENCODER_REVISION,
        local_dir=TARGET,
        allow_patterns=["*.json", "*.txt", "model.safetensors", "sentencepiece.bpe.model", "1_Pooling/*",
                        "modules.json", "tokenizer*"],
    )
    print("encoder saved to", TARGET)
