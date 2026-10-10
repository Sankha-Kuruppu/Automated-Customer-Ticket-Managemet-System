"""Can a stacking meta-model beat the fixed-weight ensemble? Honest 5-fold CV on validation.

Inputs per ticket: TF-IDF and fine-tuned-encoder probabilities (both from train-only models)
plus script features (share of Latin / Sinhala / Tamil characters). The language label is NOT used.
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.model import apply_temperature  # noqa: E402
from training.ensemble_eval import tfidf_val_probs  # noqa: E402
from training.train import CAT_SORTED, load, to_xy  # noqa: E402


def script_feats(rows):
    out = []
    for r in rows:
        t = (r.get("subject") or "") + " " + r["text"]
        letters = max(1, len(re.findall(r"\w", t)))
        lat = len(re.findall(r"[A-Za-z]", t)) / letters
        si = len(re.findall(r"[\u0D80-\u0DFF]", t)) / letters
        ta = len(re.findall(r"[\u0B80-\u0BFF]", t)) / letters
        out.append([lat, si, ta])
    return np.array(out)


def main():
    val = load("validation")
    _, yc, _, _, lang = to_xy(val)
    tc, _, _ = tfidf_val_probs()
    fc = np.load(ROOT / "models" / "ft_val_probs_ep3.npz")["cat"]
    S = script_feats(val)
    base = np.array(CAT_SORTED)[apply_temperature(0.1 * tc + 0.9 * fc, 0.5).argmax(1)]

    def logit(p):
        return np.log(np.clip(p, 1e-6, 1))

    variants = {
        "probs only": np.hstack([logit(tc), logit(fc)]),
        "probs + script": np.hstack([logit(tc), logit(fc), S]),
        "probs x script": np.hstack([logit(tc), logit(fc), S, logit(tc) * S[:, :1], logit(fc) * S[:, :1]]),
    }
    print(f"current fixed ensemble: acc={np.mean(base == yc):.4f}  singlish={np.mean(base[lang == 'singlish'] == yc[lang == 'singlish']):.3f}")
    for name, X in variants.items():
        for C in (0.1, 1.0):
            pred = np.empty(len(yc), dtype=object)
            for seed in (0, 1, 2):
                skf = StratifiedKFold(5, shuffle=True, random_state=seed)
                for tr, te in skf.split(X, yc):
                    m = LogisticRegression(C=C, max_iter=5000).fit(X[tr], yc[tr])
                    pred[te] = m.predict(X[te])
                acc = np.mean(pred == yc)
                by = {l: round(float(np.mean(pred[lang == l] == yc[lang == l])), 3) for l in ("en", "singlish", "si", "ta")}
                print(f"{name:16s} C={C:<4} seed={seed} acc={acc:.4f} {by}")


if __name__ == "__main__":
    main()
