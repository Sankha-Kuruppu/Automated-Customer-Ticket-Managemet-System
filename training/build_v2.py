"""Package model v2.x: TF-IDF heads + fine-tuned encoder heads + language-aware mixing.

Selection (ensemble weights, Singlish weight, temperature, thresholds) uses models fit on train
and scored on validation; the shipped components are then refit on train + validation.

Pipeline (see README "How it was trained"):
    python -m training.train                              # v1 experiments + metrics.json
    python -m training.finetune --mode eval --epochs 4    # per-epoch validation probabilities
    python -m training.ensemble_eval                      # choose epoch, weights, thresholds
    python -m training.finetune --mode final --epochs N   # refit encoder on train + validation
    python -m training.build_v2                           # this script -> models/ticket_model.joblib
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.labels import CATEGORIES, NO_SECONDARY  # noqa: E402
from app.model import SoftmaxHead, apply_temperature, decide, file_sha256  # noqa: E402
from training.ensemble_eval import tfidf_val_probs  # noqa: E402
from training.train import CAT_SORTED, cat_metrics, char_vec, load, lr, secondary_f1, svm_cal, to_xy, word_char_vec  # noqa: E402

MODEL_VERSION = "v2.1.0"
ROMANISED = ["singlish"]  # Tanglish excluded: only 100 train tickets, TF-IDF is weak there
REVIEW_TARGET = 0.95  # human-review threshold: auto-routed tickets must reach this accuracy
MODELS = ROOT / "models"


def main():
    ev = json.load(open(MODELS / "ensemble_eval.json"))["chosen"]
    ep = ev["epoch"]
    wc, temp = ev["category"]["weight_emb"], ev["category"]["temperature"]
    ws, ts = ev["secondary"]["weight_emb"], ev["secondary"]["threshold"]
    wu, tu = ev["urgency"]["weight_emb"], ev["urgency"]["threshold"]

    # ---- validation report for the chosen configuration (models fit on train only)
    train, val = load("train"), load("validation")
    _, yc, ys, yu, lang = to_xy(val)
    tc, tsec, turg = tfidf_val_probs()
    d = np.load(MODELS / f"ft_val_probs_ep{ep}.npz")
    sec_classes = [str(c) for c in d["sec_classes"]]
    # Language-aware weighting (v2.1): a language-ID model trained on train predicts whether a
    # ticket is Singlish; those tickets get their own ensemble weight for the category head.
    Xtr_raw, Xva_raw = to_xy(train)[0], to_xy(val)[0]
    lid_vec = char_vec()
    lid = lr(10.0).fit(lid_vec.fit_transform(Xtr_raw), to_xy(train)[4])
    lid_pred = lid.predict(lid_vec.transform(Xva_raw))
    roman = np.isin(lid_pred, ROMANISED)
    lang_id_acc = round(float(np.mean(lid_pred == lang)), 4)
    grid = []
    for wr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        w = np.where(roman, wr, wc)[:, None]
        m = cat_metrics(yc, apply_temperature((1 - w) * tc + w * d["cat"], temp), lang)
        grid.append({"weight_emb_roman": wr, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "log_loss": m["log_loss"], "singlish_acc": m["accuracy_by_language"]["singlish"]})
    top = max(g["macro_f1"] for g in grid)
    wr = min((g for g in grid if g["macro_f1"] >= top - 0.01), key=lambda g: g["log_loss"])["weight_emb_roman"]
    w = np.where(roman, wr, wc)[:, None]
    p_raw = (1 - w) * tc + w * d["cat"]
    temp = min([0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
               key=lambda t: log_loss(yc, np.clip(apply_temperature(p_raw, t), 1e-12, 1), labels=CAT_SORTED))
    pcat = apply_temperature(p_raw, temp)
    print(f"language-ID acc={lang_id_acc}; romanised weight={wr}; temperature={temp}")
    psec = (1 - ws) * tsec + ws * d["sec"]
    purg = (1 - wu) * turg + wu * d["urg"]

    conf = pcat.max(1)
    correct = np.array(CAT_SORTED)[pcat.argmax(1)] == yc
    review_t = 0.5
    for t in np.round(np.arange(0.2, 0.95, 0.01), 2):
        auto = conf >= t
        if auto.sum() >= 50 and correct[auto].mean() >= REVIEW_TARGET:
            review_t = float(t)
            break
    flagged = conf < review_t

    cfg = {"category_classes": CAT_SORTED, "secondary_classes": sec_classes,
           "secondary_threshold": ts, "urgency_threshold": tu, "review_threshold": review_t}
    preds = decide(cfg, pcat, psec, purg)
    cp = np.array([p["category"] for p in preds])
    sp = np.array([p["secondary_category"] or NO_SECONDARY for p in preds])
    up = np.array([p["is_urgent"] for p in preds]).astype(int)
    sf1, spp, srr = secondary_f1(ys, sp)
    validation = {
        "category": cat_metrics(yc, pcat, lang),
        "heads": {
            "category_macro_f1": round(f1_score(yc, cp, average="macro"), 4),
            "secondary_f1": round(sf1, 4), "secondary_precision": round(spp, 4), "secondary_recall": round(srr, 4),
            "secondary_exact_accuracy": round(float((sp == ys).mean()), 4),
            "urgency_f1": round(f1_score(yu, up), 4), "urgency_precision": round(precision_score(yu, up), 4),
            "urgency_recall": round(recall_score(yu, up), 4),
            "spam_rule_violations": int(sum(bool(p["category"] == "spam_irrelevant" and (p["is_urgent"] or p["secondary_category"])) for p in preds)),
        },
        "human_review": {
            "threshold": review_t, "target_auto_accuracy": REVIEW_TARGET,
            "share_flagged": round(float(flagged.mean()), 4),
            "accuracy_auto_routed": round(float(correct[~flagged].mean()), 4),
            "accuracy_flagged": round(float(correct[flagged].mean()), 4) if flagged.any() else None,
        },
        "per_class_f1": {c: round(float(f), 4) for c, f in
                         zip(CATEGORIES, f1_score(yc, cp, labels=CATEGORIES, average=None, zero_division=0))},
        "confusion_matrix": {"labels": CATEGORIES, "matrix": confusion_matrix(yc, cp, labels=CATEGORIES).tolist()},
    }
    print("VALIDATION v2:", json.dumps(validation["heads"]), json.dumps(validation["human_review"]))
    print("by language:", validation["category"]["accuracy_by_language"])

    # ---- final components, all fit on train + validation
    Xtr, yc_tr, ys_tr, yu_tr, _ = to_xy(train)
    Xva = to_xy(val)[0]
    vec = word_char_vec()
    A = vec.fit_transform(Xtr + Xva)
    y_cat, y_sec, y_urg = np.concatenate([yc_tr, yc]), np.concatenate([ys_tr, ys]), np.concatenate([yu_tr, yu])
    lid_vec_all = char_vec()
    lid_all = lr(10.0).fit(lid_vec_all.fit_transform(Xtr + Xva),
                           np.concatenate([to_xy(train)[4], lang]))
    heads_np = np.load(MODELS / "ft_heads.npy", allow_pickle=True).item()
    sc = heads_np["scale"]
    assert heads_np["secondary"]["classes"] == sec_classes
    bundle = {
        "model_version": MODEL_VERSION,
        "vectorizer": vec,
        "heads": {
            "category": {"tfidf": svm_cal(1.0).fit(A, y_cat), "weight_emb": wc, "weight_emb_roman": wr,
                         "temperature": temp,
                         "emb": SoftmaxHead(heads_np["category"]["W"], heads_np["category"]["b"], sc, CAT_SORTED)},
            "secondary": {"tfidf": svm_cal(1.0).fit(A, y_sec), "weight_emb": ws,
                          "emb": SoftmaxHead(heads_np["secondary"]["W"], heads_np["secondary"]["b"], sc, sec_classes)},
            "urgency": {"tfidf": svm_cal(1.0).fit(A, y_urg), "weight_emb": wu,
                        "emb": SoftmaxHead(heads_np["urgency"]["W"], heads_np["urgency"]["b"], sc, sigmoid=True)},
        },
        "category_classes": CAT_SORTED,
        "secondary_classes": sec_classes,
        "secondary_threshold": ts,
        "urgency_threshold": tu,
        "review_threshold": review_t,
        "encoder": "intfloat/multilingual-e5-small fine-tuned (models/encoder_ft_delta.safetensors)",
        "encoder_dir": "encoder_ft",
        "lang_id": {"vectorizer": lid_vec_all, "model": lid_all, "romanised": ROMANISED},
    }
    assert list(bundle["heads"]["category"]["tfidf"].classes_) == CAT_SORTED
    assert list(bundle["heads"]["secondary"]["tfidf"].classes_) == sec_classes
    path = MODELS / "ticket_model.joblib"
    joblib.dump(bundle, path, compress=3)

    # ---- merge into metrics.json (keeps v1 experiments as the improvement history)
    metrics = json.load(open(MODELS / "metrics.json", encoding="utf-8"))
    ft_hist = json.load(open(MODELS / "finetune_eval_history.json"))
    v1_val = metrics.get("validation")
    metrics["model_version"] = MODEL_VERSION
    metrics["encoder"] = "intfloat/multilingual-e5-small, fine-tuned multi-task (embeddings frozen)"
    exps = [e for e in metrics["category_experiments"] if not e["name"].startswith(("E7", "E8", "E9"))]
    best_ft = max(ft_hist, key=lambda h: h.get("val_macro_f1", 0))
    exps.append({"name": f"E7 fine-tuned e5 (multi-task, epoch {best_ft['epoch']}) alone",
                 "accuracy": best_ft["val_acc"], "macro_f1": best_ft["val_macro_f1"],
                 "ece": None, "log_loss": None})
    p_e8 = apply_temperature((1 - wc) * tc + wc * d["cat"], ev["category"]["temperature"])
    m8 = cat_metrics(yc, p_e8, lang)
    exps.append({"name": f"E8 ensemble E4 + fine-tuned e5 (epoch {ep}, w={wc})",
                 "accuracy": m8["accuracy"], "macro_f1": m8["macro_f1"], "ece": m8["ece"], "log_loss": m8["log_loss"]})
    exps.append({"name": f"E9 E8 + language-aware weight (Singlish w={wr}, T={temp})",
                 "accuracy": validation["category"]["accuracy"], "macro_f1": validation["category"]["macro_f1"],
                 "ece": validation["category"]["ece"], "log_loss": validation["category"]["log_loss"]})
    metrics["language_aware"] = {"language_id_accuracy": lang_id_acc, "grid": grid, "chosen_weight_roman": wr,
                                 "temperature": temp}
    metrics["category_experiments"] = exps
    metrics["finetune_history"] = ft_hist
    metrics["ensemble_eval"] = json.load(open(MODELS / "ensemble_eval.json"))
    metrics["v1_validation"] = metrics.get("v1_validation", v1_val)
    metrics["validation"] = validation
    metrics["thresholds"] = {"urgency": tu, "secondary": ts, "review": review_t}
    metrics["category_choice"] = {"model": "language-aware ensemble tfidf-svm + fine-tuned e5", "C": 1.0,
                                  "weight_emb": wc, "weight_emb_roman": wr, "temperature": temp, "epoch": ep}
    metrics["artifact"] = {"path": "models/ticket_model.joblib", "sha256": file_sha256(path),
                           "encoder_delta_sha256": file_sha256(MODELS / "encoder_ft_delta.safetensors"),
                           "trained_on": "train + validation"}
    json.dump(metrics, open(MODELS / "metrics.json", "w", encoding="utf-8"), indent=2)
    print("saved", path, "version", MODEL_VERSION)


if __name__ == "__main__":
    main()
