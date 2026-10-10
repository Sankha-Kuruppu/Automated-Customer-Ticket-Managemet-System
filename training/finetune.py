"""Fine-tune multilingual-e5-small on the ticket tasks (CPU friendly).

    python -m training.finetune --mode eval            # fit on train, score on validation each epoch
    python -m training.finetune --mode final --epochs N # fit on train + validation, save artifacts

Design:
* Multi-task: one shared encoder, three linear heads (category 11-way, secondary 12-way incl. "none",
  urgency binary). Loss = CE(category) + 0.5 CE(secondary) + 0.5 BCE(urgency), label smoothing 0.1.
* The 96M-parameter word-embedding matrix is frozen. Only the 12 transformer layers (~21M params)
  are trained, which (a) regularises: rare Sinhala/Tamil sub-words keep their pretrained vectors,
  and (b) keeps the artifact small: we store only the changed layers as a ~42 MB fp16 delta
  (models/encoder_ft_delta.safetensors) applied on top of the pinned base model at build time.
* Heads read the same mean-pooled, L2-normalised vector the serving Encoder produces, so serving
  can apply them with numpy (see app.model.SoftmaxHead).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.labels import NO_SECONDARY  # noqa: E402
from app.preprocess import encoder_input  # noqa: E402
from training.train import CAT_SORTED, load  # noqa: E402

BASE = ROOT / "models" / "encoder"
OUT = ROOT / "models"
SEED = 13
MAX_LEN = 256
SCALE = 20.0  # heads see unit-norm vectors; a fixed scale lets logits grow large enough


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class MultiTask(torch.nn.Module):
    def __init__(self, n_cat, n_sec):
        super().__init__()
        self.enc = AutoModel.from_pretrained(str(BASE), local_files_only=True)
        for p in self.enc.embeddings.parameters():
            p.requires_grad = False
        h = self.enc.config.hidden_size
        self.drop = torch.nn.Dropout(0.1)
        self.cat = torch.nn.Linear(h, n_cat)
        self.sec = torch.nn.Linear(h, n_sec)
        self.urg = torch.nn.Linear(h, 1)

    def embed(self, ids, mask):
        out = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return F.normalize(pooled, dim=-1)

    def forward(self, ids, mask):
        z = self.drop(self.embed(ids, mask)) * SCALE
        return self.cat(z), self.sec(z), self.urg(z).squeeze(-1)


def encode_rows(tok, rows):
    texts = [encoder_input(r.get("subject"), r["text"]) for r in rows]
    return [tok(t, truncation=True, max_length=MAX_LEN)["input_ids"] for t in texts]


def batches(ids, idx, bs, shuffle):
    """Length-bucketed batches with dynamic padding (much faster on CPU)."""
    idx = list(idx)
    if shuffle:
        random.shuffle(idx)
        chunks = [sorted(idx[i:i + bs * 20], key=lambda j: len(ids[j])) for i in range(0, len(idx), bs * 20)]
        groups = [c[i:i + bs] for c in chunks for i in range(0, len(c), bs)]
        random.shuffle(groups)
    else:
        idx = sorted(idx, key=lambda j: len(ids[j]))
        groups = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    for g in groups:
        L = max(len(ids[j]) for j in g)
        x = torch.full((len(g), L), 1, dtype=torch.long)  # XLM-R pad id = 1
        m = torch.zeros((len(g), L), dtype=torch.long)
        for k, j in enumerate(g):
            x[k, :len(ids[j])] = torch.tensor(ids[j])
            m[k, :len(ids[j])] = 1
        yield g, x, m


@torch.no_grad()
def predict(model, ids, bs=64):
    model.eval()
    n = len(ids)
    pc, ps, pu = np.zeros((n, model.cat.out_features)), np.zeros((n, model.sec.out_features)), np.zeros(n)
    for g, x, m in batches(ids, range(n), bs, False):
        c, s, u = model(x, m)
        pc[g], ps[g], pu[g] = F.softmax(c, -1).numpy(), F.softmax(s, -1).numpy(), torch.sigmoid(u).numpy()
    return pc, ps, pu


def train(rows, epochs, eval_rows=None, lr=5e-5, bs=16):
    set_seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    sec_classes = sorted({r["secondary_category"] or NO_SECONDARY for r in rows})
    model = MultiTask(len(CAT_SORTED), len(sec_classes))
    ids = encode_rows(tok, rows)
    y_cat = torch.tensor([CAT_SORTED.index(r["category"]) for r in rows])
    y_sec = torch.tensor([sec_classes.index(r["secondary_category"] or NO_SECONDARY) for r in rows])
    y_urg = torch.tensor([float(r["is_urgent"]) for r in rows])
    # class-balanced weights for the primary category
    counts = np.bincount(y_cat.numpy(), minlength=len(CAT_SORTED))
    w_cat = torch.tensor((counts.sum() / (len(counts) * np.maximum(counts, 1))), dtype=torch.float)
    pos_w = torch.tensor((len(y_urg) - y_urg.sum()) / max(y_urg.sum(), 1))

    enc_params = [p for p in model.enc.parameters() if p.requires_grad]
    head_params = list(model.cat.parameters()) + list(model.sec.parameters()) + list(model.urg.parameters())
    opt = torch.optim.AdamW([{"params": enc_params, "lr": lr}, {"params": head_params, "lr": 1e-3}],
                            weight_decay=0.01)
    steps = epochs * math.ceil(len(rows) / bs)
    warm = int(0.06 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)))

    eval_ids = encode_rows(tok, eval_rows) if eval_rows else None
    history = []
    step = 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for g, x, m in batches(ids, range(len(rows)), bs, True):
            c, s, u = model(x, m)
            loss = F.cross_entropy(c, y_cat[g], weight=w_cat, label_smoothing=0.1) \
                + 0.5 * F.cross_entropy(s, y_sec[g], label_smoothing=0.1) \
                + 0.5 * F.binary_cross_entropy_with_logits(u, y_urg[g], pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"  ep{ep} step {step}/{steps} loss={tot / 50:.3f} ({time.time() - t0:.0f}s)", flush=True)
                tot = 0.0
        rec = {"epoch": ep, "seconds": round(time.time() - t0)}
        if eval_rows:
            pc, ps, pu = predict(model, eval_ids)
            yv = np.array([r["category"] for r in eval_rows])
            pred = np.array(CAT_SORTED)[pc.argmax(1)]
            yu = np.array([int(r["is_urgent"]) for r in eval_rows])
            rec.update({"val_acc": round(float((pred == yv).mean()), 4),
                        "val_macro_f1": round(f1_score(yv, pred, average="macro"), 4),
                        "val_urgency_f1": round(f1_score(yu, (pu >= 0.5).astype(int)), 4)})
            np.savez(OUT / f"ft_val_probs_ep{ep}.npz", cat=pc, sec=ps, urg=pu, sec_classes=np.array(sec_classes))
        history.append(rec)
        print("  ", rec, flush=True)
    return model, tok, sec_classes, history


def save_artifacts(model, sec_classes, history):
    # 1. encoder delta: transformer layers only (embeddings were frozen), fp16
    delta = {f"encoder.{k}": v.detach().half().contiguous() for k, v in model.enc.encoder.state_dict().items()}
    save_file(delta, str(OUT / "encoder_ft_delta.safetensors"))
    # 2. heads as plain numpy (applied by app.model.SoftmaxHead; urgency uses sigmoid=True)
    heads = {
        "scale": SCALE,
        "category": {"W": model.cat.weight.detach().numpy(), "b": model.cat.bias.detach().numpy(), "classes": CAT_SORTED},
        "secondary": {"W": model.sec.weight.detach().numpy(), "b": model.sec.bias.detach().numpy(), "classes": sec_classes},
        "urgency": {"W": model.urg.weight.detach().numpy(), "b": model.urg.bias.detach().numpy()},
    }
    np.save(OUT / "ft_heads.npy", heads, allow_pickle=True)
    with open(OUT / "finetune_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("saved encoder delta + heads")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "final"], default="eval")
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()
    train_rows, val_rows = load("train"), load("validation")
    if args.mode == "eval":
        _, _, _, hist = train(train_rows, args.epochs, eval_rows=val_rows)
        with open(OUT / "finetune_eval_history.json", "w") as f:
            json.dump(hist, f, indent=2)
    else:
        model, _, sec_classes, hist = train(train_rows + val_rows, args.epochs)
        save_artifacts(model, sec_classes, hist)


if __name__ == "__main__":
    main()
