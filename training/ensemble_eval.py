"""Evaluate TF-IDF + fine-tuned-encoder ensembles on validation for each fine-tuning epoch.

    python -m training.ensemble_eval
Needs models/ft_val_probs_ep*.npz from `python -m training.finetune --mode eval`.
Writes models/ensemble_eval.json (choice of epoch, weights, thresholds, temperature).
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.labels import NO_SECONDARY  # noqa: E402
from app.model import apply_temperature, decide  # noqa: E402
from training.train import CAT_SORTED, CACHE, cat_metrics, load, secondary_f1, svm_cal, to_xy, word_char_vec  # noqa: E402


def tfidf_val_probs():
    path = CACHE / "tfidf_val_probs.npz"
    if path.exists():
        d = np.load(path)
        return d["cat"], d["sec"], d["urg"]
    Xtr, yc, ys, yu, _ = to_xy(load("train"))
    Xva = to_xy(load("validation"))[0]
    v = word_char_vec()
    A, B = v.fit_transform(Xtr), v.transform(Xva)
    pc = svm_cal(1.0).fit(A, yc).predict_proba(B)
    ps = svm_cal(1.0).fit(A, ys).predict_proba(B)
    pu = svm_cal(1.0).fit(A, yu).predict_proba(B)[:, 1]
    np.savez(path, cat=pc, sec=ps, urg=pu)
    return pc, ps, pu


def main():
    val = load("validation")
    _, yc, ys, yu, lang = to_xy(val)
    tc, ts, tu = tfidf_val_probs()
    sec_classes = sorted(set(ys))
    results = []
    for f in sorted(glob.glob(str(ROOT / "models" / "ft_val_probs_ep*.npz"))):
        ep = int(f.split("ep")[-1].split(".")[0])
        d = np.load(f)
        fc, fs, fu = d["cat"], d["sec"], d["urg"]
        assert list(d["sec_classes"]) == sec_classes
        # Rule: among weights within 0.01 macro-F1 of the best, take the lowest log loss
        # (better calibrated, and keeps TF-IDF diversity as insurance against distribution shift).
        cands = []
        for w in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0):
            m = cat_metrics(yc, (1 - w) * tc + w * fc, lang)
            cands.append({"weight_emb": w, **{k: m[k] for k in ("accuracy", "macro_f1", "log_loss", "ece")}})
        top = max(c["macro_f1"] for c in cands)
        best = min((c for c in cands if c["macro_f1"] >= top - 0.01), key=lambda c: c["log_loss"])
        p = (1 - best["weight_emb"]) * tc + best["weight_emb"] * fc
        temp = min([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0],
                   key=lambda t: log_loss(yc, np.clip(apply_temperature(p, t), 1e-12, 1), labels=CAT_SORTED))
        pcat = apply_temperature(p, temp)
        cm = cat_metrics(yc, pcat, lang)
        base = {"category_classes": CAT_SORTED, "secondary_classes": sec_classes, "review_threshold": 0.0}
        bs = None
        for w in (0.0, 0.3, 0.5, 0.7, 1.0):
            psec = (1 - w) * ts + w * fs
            for t in np.round(np.arange(0.1, 0.81, 0.025), 3):
                pr = decide({**base, "secondary_threshold": float(t), "urgency_threshold": 1.1}, pcat, psec, np.zeros(len(val)))
                sp = np.array([x["secondary_category"] or NO_SECONDARY for x in pr])
                f1, pp, rr = secondary_f1(ys, sp)
                if bs is None or f1 > bs["f1"]:
                    bs = {"weight_emb": w, "threshold": float(t), "f1": round(f1, 4), "precision": round(pp, 4), "recall": round(rr, 4)}
        bu = None
        for w in (0.0, 0.3, 0.5, 0.7, 1.0):
            pu = (1 - w) * tu + w * fu
            for t in np.round(np.arange(0.1, 0.91, 0.025), 3):
                pred = (pu >= t).astype(int)
                f1 = f1_score(yu, pred, zero_division=0)
                if bu is None or f1 > bu["f1"]:
                    bu = {"weight_emb": w, "threshold": float(t), "f1": round(f1, 4),
                          "precision": round(precision_score(yu, pred), 4), "recall": round(recall_score(yu, pred), 4)}
        ft_alone = cat_metrics(yc, fc, lang)
        r = {"epoch": ep, "ft_alone_acc": ft_alone["accuracy"], "ft_alone_macro_f1": ft_alone["macro_f1"],
             "category": {**best, "temperature": temp, "ece_after_temp": cm["ece"],
                          "accuracy_by_language": cm["accuracy_by_language"]},
             "secondary": bs, "urgency": bu}
        results.append(r)
        print(json.dumps(r))
    best = max(results, key=lambda r: (r["category"]["macro_f1"], r["urgency"]["f1"]))
    with open(ROOT / "models" / "ensemble_eval.json", "w") as f:
        json.dump({"per_epoch": results, "chosen": best}, f, indent=2)
    print("CHOSEN epoch", best["epoch"])


if __name__ == "__main__":
    main()
