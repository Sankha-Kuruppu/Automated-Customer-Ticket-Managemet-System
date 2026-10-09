"""Train, validate and package the RideEat ticket classifier.

Usage:
    python training/download_encoder.py   # once: fetch the multilingual encoder into models/encoder
    python -m training.train

Stages (results are printed and written to models/metrics.json):
  1. Category experiments, all fit on train.jsonl and scored on validation.jsonl:
     baseline Naive Bayes -> TF-IDF linear models -> sentence embeddings -> ensemble.
  2. Tuning of the ensemble (embedding-model C, ensemble weight, temperature) on validation.
  3. Secondary-category and urgency heads: ensemble weight and decision threshold on validation.
  4. Final model: refit on train + validation with the chosen settings and saved to
     models/ticket_model.joblib, with its SHA-256 and version tag.

Why validation (not cross-validation on train) drives selection: train contains near-duplicate
templates, so 5-fold CV on train reaches ~99% accuracy while validation (reworded tickets)
is ~60-65% for lexical models. Validation is the honest estimate of unseen wording.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.labels import CATEGORIES, NO_SECONDARY  # noqa: E402
from app.model import apply_temperature, decide, file_sha256  # noqa: E402
from app.preprocess import build_input, encoder_input  # noqa: E402

MODEL_VERSION = "v1.1.0"
DATA = ROOT / "data"
OUT = ROOT / "models"
CACHE = ROOT / "training" / "cache"
SEED = 42
CAT_SORTED = sorted(CATEGORIES)


# ---------------------------------------------------------------- data
def load(name: str) -> list[dict]:
    with open(DATA / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_xy(rows):
    X = [build_input(r["channel"], r.get("subject"), r["text"]) for r in rows]
    y_cat = np.array([r["category"] for r in rows])
    y_sec = np.array([r["secondary_category"] or NO_SECONDARY for r in rows])
    y_urg = np.array([bool(r["is_urgent"]) for r in rows]).astype(int)
    lang = np.array([r["language"] for r in rows])
    return X, y_cat, y_sec, y_urg, lang


_encoder = None


def embeddings(name: str, rows) -> np.ndarray:
    """Sentence embeddings for a split, cached on disk (same encoder code path as serving)."""
    global _encoder
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"e5_{name}.npy"
    if path.exists():
        E = np.load(path)
        if E.shape[0] == len(rows):
            return E
    if _encoder is None:
        from app.encoder import Encoder
        _encoder = Encoder()
    t = time.time()
    E = _encoder.encode([encoder_input(r.get("subject"), r["text"]) for r in rows])
    np.save(path, E)
    print(f"  embedded {name}: {E.shape} in {time.time() - t:.0f}s")
    return E


# ---------------------------------------------------------------- features / models
def word_vec():
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w+\b")


def char_vec():
    # char_wb n-grams cope with Sinhala/Tamil script and inconsistent Singlish/Tanglish spelling.
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                           max_features=300_000)


def word_char_vec():
    return FeatureUnion([("word", word_vec()), ("char", char_vec())])


def lr(C: float):
    return LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=SEED)


def svm_cal(C: float):
    return CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced", random_state=SEED),
                                  method="sigmoid", cv=5)


# ---------------------------------------------------------------- metrics
def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the top-class confidence."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def cat_metrics(y_true, proba, lang):
    """proba columns are in CAT_SORTED order."""
    pred = np.array(CAT_SORTED)[proba.argmax(1)]
    correct = pred == y_true
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "macro_f1": round(f1_score(y_true, pred, average="macro", labels=CATEGORIES, zero_division=0), 4),
        "log_loss": round(log_loss(y_true, np.clip(proba, 1e-12, 1), labels=CAT_SORTED), 4),
        "ece": round(ece(proba.max(1), correct), 4),
        "macro_f1_by_language": {
            lg: round(f1_score(y_true[lang == lg], pred[lang == lg], average="macro", labels=CATEGORIES,
                               zero_division=0), 4)
            for lg in sorted(set(lang))
        },
        "accuracy_by_language": {lg: round(float(correct[lang == lg].mean()), 4) for lg in sorted(set(lang))},
    }


def secondary_f1(y_sec, sec_pred):
    """F1 over secondary labels: a hit is a predicted non-null secondary equal to the true one."""
    pred_pos = sec_pred != NO_SECONDARY
    true_pos = y_sec != NO_SECONDARY
    hits = (pred_pos & (sec_pred == y_sec)).sum()
    p = hits / pred_pos.sum() if pred_pos.sum() else 0.0
    r = hits / true_pos.sum() if true_pos.sum() else 0.0
    return float(2 * p * r / (p + r)) if p + r else 0.0, float(p), float(r)


def log(msg: str):
    print(msg, flush=True)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    train, val = load("train"), load("validation")
    Xtr, ytr_cat, ytr_sec, ytr_urg, _ = to_xy(train)
    Xva, yva_cat, yva_sec, yva_urg, lang_va = to_xy(val)
    Etr, Eva = embeddings("train", train), embeddings("validation", val)
    report: dict = {
        "model_version": MODEL_VERSION,
        "data": {"train": len(train), "validation": len(val),
                 "train_category_counts": dict(Counter(ytr_cat.tolist())),
                 "train_secondary_rate": round(float((ytr_sec != NO_SECONDARY).mean()), 4),
                 "train_urgent_rate": round(float(ytr_urg.mean()), 4)},
        "encoder": "intfloat/multilingual-e5-small (frozen, mean-pooled, L2-normalised)",
    }

    # ---- Stage 1: category experiments -----------------------------------------------
    log("Stage 1: category experiments (fit on train, score on validation)")
    wv, cv, wcv = word_vec(), char_vec(), word_char_vec()
    W_tr, W_va = wv.fit_transform(Xtr), wv.transform(Xva)
    C_tr, C_va = cv.fit_transform(Xtr), cv.transform(Xva)
    A_tr, A_va = wcv.fit_transform(Xtr), wcv.transform(Xva)

    fitted = {}

    def run(name, model, Xa, Xb):
        m = model.fit(Xa, ytr_cat)
        p = m.predict_proba(Xb)
        fitted[name] = p
        r = {"name": name, **cat_metrics(yva_cat, p, lang_va)}
        log(f"  {name:55s} acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} ece={r['ece']:.4f}")
        return r

    exps = [
        run("E1 baseline: word TF-IDF + Complement NB", ComplementNB(alpha=0.3), W_tr, W_va),
        run("E2 word TF-IDF + LogReg", lr(30.0), W_tr, W_va),
        run("E3 char n-gram TF-IDF + LogReg", lr(30.0), C_tr, C_va),
        run("E4 word+char TF-IDF + LinearSVC (calibrated)", svm_cal(1.0), A_tr, A_va),
        run("E5 multilingual-e5 embeddings + LogReg", lr(100.0), Etr, Eva),
    ]
    p_ens = 0.5 * fitted["E4 word+char TF-IDF + LinearSVC (calibrated)"] + \
        0.5 * fitted["E5 multilingual-e5 embeddings + LogReg"]
    r = {"name": "E6 ensemble E4 + E5 (equal weights)", **cat_metrics(yva_cat, p_ens, lang_va)}
    log(f"  {r['name']:55s} acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} ece={r['ece']:.4f}")
    exps.append(r)
    report["category_experiments"] = exps

    # ---- Stage 2: tune ensemble ---------------------------------------------------------
    log("Stage 2: tuning the category ensemble on validation")
    p_tfidf = fitted["E4 word+char TF-IDF + LinearSVC (calibrated)"]
    emb_probs = {C: lr(C).fit(Etr, ytr_cat).predict_proba(Eva) for C in (30.0, 100.0, 300.0)}
    tuning = []
    for C, p_emb in emb_probs.items():
        for w in (0.4, 0.5, 0.6, 0.7):
            p = (1 - w) * p_tfidf + w * p_emb
            m = cat_metrics(yva_cat, p, lang_va)
            tuning.append({"emb_C": C, "weight_emb": w, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                           "log_loss": m["log_loss"]})
    best = max(tuning, key=lambda t: (t["macro_f1"], -t["log_loss"]))
    p_best = (1 - best["weight_emb"]) * p_tfidf + best["weight_emb"] * emb_probs[best["emb_C"]]
    # Temperature scaling for calibrated confidence (does not change the argmax).
    temps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    temp = min(temps, key=lambda t: log_loss(yva_cat, np.clip(apply_temperature(p_best, t), 1e-12, 1),
                                             labels=CAT_SORTED))
    best["temperature"] = temp
    report["category_tuning"] = tuning
    report["category_choice"] = best
    log(f"  chosen: emb C={best['emb_C']} weight_emb={best['weight_emb']} temperature={temp} "
        f"macroF1={best['macro_f1']}")
    p_cat_va = apply_temperature(p_best, temp)

    # ---- Stage 3: secondary and urgency heads -----------------------------------------
    log("Stage 3: secondary and urgency heads")
    sec_classes = sorted(set(ytr_sec))
    sec_tfidf = svm_cal(1.0).fit(A_tr, ytr_sec).predict_proba(A_va)
    sec_emb = lr(100.0).fit(Etr, ytr_sec).predict_proba(Eva)
    urg_tfidf = svm_cal(1.0).fit(A_tr, ytr_urg).predict_proba(A_va)[:, 1]
    urg_emb = lr(100.0).fit(Etr, ytr_urg).predict_proba(Eva)[:, 1]

    base = {"category_classes": CAT_SORTED, "secondary_classes": sec_classes, "review_threshold": 0.0}
    dummy_urg = np.zeros(len(val))

    best_sec = None
    for w in (0.0, 0.3, 0.5, 0.7, 1.0):
        p_sec = (1 - w) * sec_tfidf + w * sec_emb
        for t in np.round(np.arange(0.1, 0.81, 0.025), 3):
            preds = decide({**base, "secondary_threshold": float(t), "urgency_threshold": 1.1},
                           p_cat_va, p_sec, dummy_urg)
            sp = np.array([p["secondary_category"] or NO_SECONDARY for p in preds])
            f1, p_, r_ = secondary_f1(yva_sec, sp)
            if best_sec is None or f1 > best_sec["f1"]:
                best_sec = {"weight_emb": w, "threshold": float(t), "f1": round(f1, 4),
                            "precision": round(p_, 4), "recall": round(r_, 4),
                            "exact_accuracy": round(float((sp == yva_sec).mean()), 4)}
    log(f"  secondary: {best_sec}")

    best_urg = None
    for w in (0.0, 0.3, 0.5, 0.7, 1.0):
        pu = (1 - w) * urg_tfidf + w * urg_emb
        for t in np.round(np.arange(0.1, 0.91, 0.025), 3):
            pred = (pu >= t).astype(int)
            f1 = f1_score(yva_urg, pred, zero_division=0)
            if best_urg is None or f1 > best_urg["f1"]:
                best_urg = {"weight_emb": w, "threshold": float(t), "f1": round(f1, 4),
                            "precision": round(precision_score(yva_urg, pred, zero_division=0), 4),
                            "recall": round(recall_score(yva_urg, pred, zero_division=0), 4)}
    log(f"  urgency: {best_urg}")

    # Human-review threshold: smallest confidence cut-off at which auto-routed tickets reach
    # 90% category accuracy on validation.
    conf = p_cat_va.max(1)
    correct = np.array(CAT_SORTED)[p_cat_va.argmax(1)] == yva_cat
    review_t = 0.5
    for t in np.round(np.arange(0.2, 0.95, 0.01), 2):
        auto = conf >= t
        if auto.sum() >= 50 and correct[auto].mean() >= 0.90:
            review_t = float(t)
            break
    flagged = conf < review_t
    review = {
        "threshold": review_t,
        "target_auto_accuracy": 0.90,
        "share_flagged": round(float(flagged.mean()), 4),
        "accuracy_auto_routed": round(float(correct[~flagged].mean()), 4) if (~flagged).any() else None,
        "accuracy_flagged": round(float(correct[flagged].mean()), 4) if flagged.any() else None,
    }
    log(f"  human review: {review}")

    # Full validation scoring through the exact serving decision function.
    p_sec_va = (1 - best_sec["weight_emb"]) * sec_tfidf + best_sec["weight_emb"] * sec_emb
    p_urg_va = (1 - best_urg["weight_emb"]) * urg_tfidf + best_urg["weight_emb"] * urg_emb
    final_cfg = {**base, "secondary_threshold": best_sec["threshold"], "urgency_threshold": best_urg["threshold"],
                 "review_threshold": review_t}
    preds = decide(final_cfg, p_cat_va, p_sec_va, p_urg_va)
    cat_pred = np.array([p["category"] for p in preds])
    sec_pred = np.array([p["secondary_category"] or NO_SECONDARY for p in preds])
    urg_pred = np.array([p["is_urgent"] for p in preds]).astype(int)
    sf1, sp_, sr_ = secondary_f1(yva_sec, sec_pred)
    cm = confusion_matrix(yva_cat, cat_pred, labels=CATEGORIES)
    report["validation"] = {
        "category": cat_metrics(yva_cat, p_cat_va, lang_va),
        "heads": {
            "category_macro_f1": round(f1_score(yva_cat, cat_pred, average="macro", zero_division=0), 4),
            "secondary_f1": round(sf1, 4), "secondary_precision": round(sp_, 4), "secondary_recall": round(sr_, 4),
            "secondary_exact_accuracy": round(float((sec_pred == yva_sec).mean()), 4),
            "urgency_f1": round(f1_score(yva_urg, urg_pred, zero_division=0), 4),
            "urgency_precision": round(precision_score(yva_urg, urg_pred, zero_division=0), 4),
            "urgency_recall": round(recall_score(yva_urg, urg_pred, zero_division=0), 4),
            "spam_rule_violations": int(sum(bool(p["category"] == "spam_irrelevant" and (p["is_urgent"] or p["secondary_category"])) for p in preds)),
        },
        "human_review": review,
        "per_class_f1": {c: round(float(f), 4) for c, f in
                         zip(CATEGORIES, f1_score(yva_cat, cat_pred, labels=CATEGORIES, average=None, zero_division=0))},
        "confusion_matrix": {"labels": CATEGORIES, "matrix": cm.tolist()},
    }
    report["thresholds"] = {"urgency": best_urg["threshold"], "secondary": best_sec["threshold"], "review": review_t}
    report["head_choices"] = {"secondary": best_sec, "urgency": best_urg}
    log(f"  VALIDATION: {json.dumps(report['validation']['heads'])}")
    log(f"  by language (acc): {report['validation']['category']['accuracy_by_language']}")

    # ---- Stage 4: final refit on train + validation -----------------------------------
    log("Stage 4: refit on train + validation")
    X_all = Xtr + Xva
    E_all = np.vstack([Etr, Eva])
    y_cat_all = np.concatenate([ytr_cat, yva_cat])
    y_sec_all = np.concatenate([ytr_sec, yva_sec])
    y_urg_all = np.concatenate([ytr_urg, yva_urg])
    vec = word_char_vec()
    A_all = vec.fit_transform(X_all)

    def head(y, w, C_emb, temperature=1.0):
        return {"tfidf": svm_cal(1.0).fit(A_all, y),
                "emb": lr(C_emb).fit(E_all, y) if w > 0 else None,
                "weight_emb": w, "temperature": temperature}

    bundle = {
        "model_version": MODEL_VERSION,
        "vectorizer": vec,
        "heads": {
            "category": head(y_cat_all, best["weight_emb"], best["emb_C"], temp),
            "secondary": head(y_sec_all, best_sec["weight_emb"], 100.0),
            "urgency": head(y_urg_all, best_urg["weight_emb"], 100.0),
        },
        "category_classes": CAT_SORTED,
        "secondary_classes": sec_classes,
        "secondary_threshold": best_sec["threshold"],
        "urgency_threshold": best_urg["threshold"],
        "review_threshold": review_t,
        "encoder": "intfloat/multilingual-e5-small",
    }
    assert list(bundle["heads"]["category"]["tfidf"].classes_) == CAT_SORTED
    assert list(bundle["heads"]["secondary"]["tfidf"].classes_) == sec_classes

    OUT.mkdir(exist_ok=True)
    path = OUT / "ticket_model.joblib"
    joblib.dump(bundle, path, compress=3)
    report["artifact"] = {"path": "models/ticket_model.joblib", "sha256": file_sha256(path),
                          "trained_on": "train + validation", "training_seconds": round(time.time() - t0, 1)}
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
